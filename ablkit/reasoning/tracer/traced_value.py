"""
TracedValue -- 带依赖集追踪的值对象 + 编译求值器。

TracedValue 支持的运算:
  算术: +, -, *, //, %, **, abs, neg
  比较: ==, !=, <, >, <=, >= (返回 TracedValue 而非 bool)
  布尔: |, &, ~  (Python 短路 and/or/not 无法追踪)
  转换: int(), float(), bool()
"""

from typing import FrozenSet, Optional


# ============================================================
# 追踪值 (TracedValue) -- 带依赖集追踪的值对象
# ============================================================


def _safe_div(a, b):
    return a / b if b else 0


def _safe_floordiv(a, b):
    return a // b if b else 0


class TracedValue:
    """A value that tracks which input variables it depends on.

    Also records operations for compiled-evaluator support: each node stores
    its graph position (_pos) and the operation that created it (_op_record).
    """

    __slots__ = ("value", "sources", "_graph", "_pos", "_op_record")

    def __init__(self, value, sources: FrozenSet[int], graph: Optional[list] = None):
        self.value = value
        self.sources = sources
        self._graph = graph
        self._op_record = None
        if graph is not None:
            self._pos = len(graph)
            graph.append(self)
        else:
            self._pos = -1

    def _binop(self, other, fn):
        ov = other.value if isinstance(other, TracedValue) else other
        os = other.sources if isinstance(other, TracedValue) else frozenset()
        result = TracedValue(fn(self.value, ov), self.sources | os, self._graph)
        other_pos = other._pos if isinstance(other, TracedValue) else None
        result._op_record = (fn, self._pos, other_pos, ov if other_pos is None else None)
        return result

    def _rbinop(self, other, fn):
        ov = other.value if isinstance(other, TracedValue) else other
        os = other.sources if isinstance(other, TracedValue) else frozenset()
        result = TracedValue(fn(ov, self.value), self.sources | os, self._graph)
        other_pos = other._pos if isinstance(other, TracedValue) else None
        result._op_record = (fn, other_pos, self._pos, ov if other_pos is None else None)
        return result

    def _unaryop(self, fn):
        result = TracedValue(fn(self.value), self.sources, self._graph)
        result._op_record = (fn, self._pos, None, None)
        return result

    # -- Arithmetic --
    def __add__(self, o):      return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o):     return self._rbinop(o, lambda a, b: a + b)
    def __sub__(self, o):      return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o):     return self._rbinop(o, lambda a, b: a - b)
    def __mul__(self, o):      return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o):     return self._rbinop(o, lambda a, b: a * b)
    def __neg__(self):         return self._unaryop(lambda a: -a)
    def __abs__(self):         return self._unaryop(abs)
    def __floordiv__(self, o): return self._binop(o, _safe_floordiv)
    def __rfloordiv__(self, o):return self._rbinop(o, _safe_floordiv)
    def __mod__(self, o):      return self._binop(o, lambda a, b: a % b if b else 0)
    def __pow__(self, o):      return self._binop(o, lambda a, b: a ** b)
    def __truediv__(self, o):  return self._binop(o, _safe_div)
    def __rtruediv__(self, o): return self._rbinop(o, _safe_div)

    # -- Comparison (return TracedValue to keep tracking) --
    def __eq__(self, o):  return self._binop(o, lambda a, b: int(a == b))
    def __ne__(self, o):  return self._binop(o, lambda a, b: int(a != b))
    def __lt__(self, o):  return self._binop(o, lambda a, b: int(a < b))
    def __le__(self, o):  return self._binop(o, lambda a, b: int(a <= b))
    def __gt__(self, o):  return self._binop(o, lambda a, b: int(a > b))
    def __ge__(self, o):  return self._binop(o, lambda a, b: int(a >= b))

    # -- Boolean (for `or`, `and`, `not` in KB logic) --
    def __or__(self, o):  return self._binop(o, lambda a, b: int(bool(a) or bool(b)))
    def __ror__(self, o): return self._rbinop(o, lambda a, b: int(bool(a) or bool(b)))
    def __and__(self, o): return self._binop(o, lambda a, b: int(bool(a) and bool(b)))
    def __rand__(self, o):return self._rbinop(o, lambda a, b: int(bool(a) and bool(b)))
    def __invert__(self): return self._unaryop(lambda a: int(not bool(a)))

    # -- Conversion --
    def __int__(self):   return int(self.value)
    def __float__(self): return float(self.value)
    def __bool__(self):  return bool(self.value)
    def __hash__(self):  return hash(self.value)
    def __round__(self, ndigits=None): return self._unaryop(lambda a: round(a, ndigits))
    def __repr__(self):  return f"T({self.value}, {set(self.sources)})"


# ============================================================
# 编译求值器 -- 将 TracedValue 图编译为快速回放函数 (~100x 加速)
# ============================================================


def compile_graph(graph, n, target_positions):
    """Compile a TracedValue graph into a fast replay function.

    After one full TracedValue trace, this builds a function that replays
    the same computation with plain Python values -- no TracedValue objects,
    no frozenset tracking. ~100x faster than re-tracing.

    Args:
        graph: list of TracedValue nodes from a traced execution.
        n: number of input variables (first n graph nodes are inputs).
        target_positions: list of graph positions to extract values from.

    Returns:
        A function: inputs (list of n values) -> tuple of values at target_positions.
        Returns None if the graph cannot be compiled (missing op records).
    """
    instructions = []
    for i in range(n, len(graph)):
        rec = graph[i]._op_record
        if rec is None:
            return None
        fn, a1, a2, const = rec
        if a1 is not None and a2 is not None:
            instructions.append((0, fn, a1, a2))
        elif a1 is not None and a2 is None and const is not None:
            instructions.append((1, fn, a1, const))
        elif a1 is None and a2 is not None and const is not None:
            instructions.append((2, fn, a2, const))
        elif a2 is None and const is None:
            instructions.append((3, fn, a1, None))
        else:
            return None

    _instrs = instructions
    _targets = target_positions

    def evaluate(inputs):
        vals = list(inputs)
        for op, fn, a, b in _instrs:
            if op == 0:
                vals.append(fn(vals[a], vals[b]))
            elif op == 1:
                vals.append(fn(vals[a], b))
            elif op == 2:
                vals.append(fn(b, vals[a]))
            else:
                vals.append(fn(vals[a]))
        return tuple(vals[p] for p in _targets)

    return evaluate
