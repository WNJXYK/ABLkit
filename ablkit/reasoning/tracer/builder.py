"""
分解构建器 -- 预计算转移表 + 构建 Decomposition。

包含:
- CSS 提取 (三层策略: 精确匹配 / frontier / raw pass-through)
- y-part 分配与 y-pruning
- 转移表预计算 (eager / lazy 两种模式)
"""

import itertools
from collections import defaultdict
from typing import Any, Callable, List, Optional

from ..ifw_dp import Decomposition, make_tree
from .traced_value import TracedValue, compile_graph


# ============================================================
# 代价估计辅助 -- subtree 变量集合 + y-part 分配
# ============================================================


def _compute_subtree_vars(var_groups, children, root):
    """Compute subtree_vars[i] = frozenset of all variables in subtree rooted at i."""
    num_nodes = len(var_groups)
    subtree_vars = [None] * num_nodes

    def _recurse(i):
        result = set(var_groups[i])
        for ch in children[i]:
            _recurse(ch)
            result |= subtree_vars[ch]
        subtree_vars[i] = frozenset(result)

    _recurse(root)
    return subtree_vars


def _assign_y_parts(subtree_vars, root, y_check_info):
    """
    Assign y-parts to tree nodes. For each y_part with z_sources S,
    find the smallest non-root node whose subtree covers S.

    Returns:
        {node_idx: [y_part_indices]}
    """
    if not y_check_info:
        return {}

    result = {}
    num_nodes = len(subtree_vars)

    for yi, z_src in y_check_info.items():
        best_node = None
        best_size = len(subtree_vars[root]) + 1
        for node_idx in range(num_nodes):
            if node_idx == root:
                continue
            if z_src <= subtree_vars[node_idx] and len(subtree_vars[node_idx]) < best_size:
                best_node = node_idx
                best_size = len(subtree_vars[node_idx])
        if best_node is not None:
            result.setdefault(best_node, []).append(yi)

    return result


# ============================================================
# 分解构建
# ============================================================


def _build_decomposition(
    logic_forward, n, K, var_groups, children, root, max_precompute,
    y_check_info=None, y_decompose_fn=None, min_css_pos=None,
    lazy_precompute: bool = False, max_states: int = 0,
    compiled_cache: dict = None, y_size: int = 0,
):
    """
    构建带预计算转移表的树形 Decomposition。

    根据 y_decompose_fn + min_css_pos 是否提供，分为两种模式:

    **最小 CSS 模式** (有 y_decompose_fn):
      CSS = 每个源集合的最小域值 (如加法的进位，H=2)。
      每步 (含根) 用 y 分量验证。根节点返回 ("__target__", y)。

    **完整 CSS 模式** (无 y_decompose_fn):
      CSS = 源集合处所有 TracedValue 节点的元组。
      根节点直接检查 KB(z)==y。
    """
    num_nodes = len(var_groups)
    subtree_vars = _compute_subtree_vars(var_groups, children, root)
    _use_min_css = bool(min_css_pos)
    _use_y_cond = bool(y_decompose_fn and min_css_pos)

    # -- Helpers --

    def _run_kb(z_values):
        res = logic_forward(z_values)
        return res.value if isinstance(res, TracedValue) else res

    _compiled = compiled_cache if compiled_cache is not None else {}

    def _find_frontier(graph, target_sources):
        """找到计算图中 subtree -> external 的信息出口 (frontier)。

        frontier = sources <= target_sources 且至少有一个 external user 的节点。
        外部节点 = sources 不是 target_sources 的子集。
        frontier 形成正确的 cut: subtree 变量到输出的每条路径
        必经过至少一个 frontier 节点。

        相比 raw pass-through，frontier 利用了中间计算的压缩:
          subtree_vars = {0,1,2,4,5}, 存在瓶颈 carry0 (sources={0,4}):
          raw: (z0,z1,z2,z4,z5) -> H = K^5
          frontier: (carry0, digit0, carry1, digit1, z2) -> H 远小于 K^5
        """
        used_by = defaultdict(set)
        for j in range(n, len(graph)):
            rec = graph[j]._op_record
            if rec is None:
                continue
            _, left_pos, right_pos, _ = rec
            if left_pos is not None:
                used_by[left_pos].add(j)
            if right_pos is not None:
                used_by[right_pos].add(j)

        internal = {j for j, nd in enumerate(graph) if nd.sources <= target_sources}

        frontier = []
        for j in sorted(internal):
            users = used_by.get(j, set())
            if not users or any(u not in internal for u in users):
                frontier.append(j)

        return frontier

    def _extract_css(z_values, target_sources):
        """Extract CSS for a non-root node from TracedValue graph.

        Frontier-first strategy:
        1. **Frontier CSS**: 计算图中 subtree -> external 的信息出口。
           捕获所有从子树流向外部的信息，对下游无损。
           当 min_css 模式启用时，尝试从 frontier 中选取最小域位置。
        2. **Raw pass-through**: 兜底 (frontier 为空时)，CSS = 变量原始值元组。

        首次调用编译 evaluator (~100x faster)，后续走缓存。
        """
        entry = _compiled.get(target_sources)
        if entry is not None:
            all_vals, min_pos = entry
            vals = all_vals(z_values)
            if _use_min_css and min_pos is not None and min_pos < len(vals):
                return vals[min_pos]
            return vals

        # Slow path: full TracedValue trace (once per target_sources)
        graph = []
        inputs = [TracedValue(z_values[i], frozenset({i}), graph)
                  for i in range(len(z_values))]
        try:
            logic_forward(inputs)
        except Exception:
            sorted_src = sorted(target_sources)
            _compiled[target_sources] = (lambda z, _s=sorted_src: tuple(z[v] for v in _s), None)
            return tuple(z_values[v] for v in sorted_src)

        # Frontier CSS: 子树的信息出口 (对下游无损)
        frontier = _find_frontier(graph, target_sources)
        if frontier:
            evaluator = compile_graph(graph, n, frontier)
            # min_css: 从 frontier 中选取 min_css 候选位置
            min_pos_in_list = None
            if min_css_pos:
                global_pos = min_css_pos.get(target_sources)
                if global_pos is not None and global_pos in frontier:
                    min_pos_in_list = frontier.index(global_pos)
            _compiled[target_sources] = (evaluator, min_pos_in_list) if evaluator else None

            if _use_min_css and min_pos_in_list is not None:
                return graph[frontier[min_pos_in_list]].value
            return tuple(graph[p].value for p in frontier)

        # Raw pass-through: 兜底
        sorted_src = sorted(target_sources)
        _compiled[target_sources] = (lambda z, _s=sorted_src: tuple(z[v] for v in _s), None)
        return tuple(z_values[v] for v in sorted_src)

    # -- Y-part assignment --

    y_node_asgn = {}
    if y_check_info and y_decompose_fn:
        y_node_asgn = _assign_y_parts(subtree_vars, root, y_check_info)
        if _use_y_cond:
            assigned = set()
            for yis in y_node_asgn.values():
                assigned.update(yis)
            # Unassigned: in y_check_info but not yet assigned to any non-root node
            unassigned = [yi for yi in y_check_info if yi not in assigned]
            if unassigned:
                y_node_asgn.setdefault(root, []).extend(unassigned)
            # Missing: y-parts whose z-sources were never discovered by _discover_y_checks
            # (e.g. they only appear after earlier y-checks pass, making them very rare).
            # They depend on ALL z-sources, so must be checked at the root.
            if y_size > 0:
                discovered = set(y_check_info.keys())
                missing = [yi for yi in range(y_size) if yi not in discovered]
                if missing:
                    y_node_asgn.setdefault(root, []).extend(missing)

    # -- Post-order traversal --

    order = []

    def _postorder(i):
        for ch in children[i]:
            _postorder(ch)
        order.append(i)

    _postorder(root)

    # -- Precompute transition tables --

    trans_table = [{} for _ in range(num_nodes)]
    partial_table = [{} for _ in range(num_nodes)]
    css_states = [set() for _ in range(num_nodes)]
    css_representative = [{} for _ in range(num_nodes)]
    total_kb_calls = 0
    budget_exceeded = False
    precompute_skipped = False

    for node in order:
        if budget_exceeded:
            break

        ch = children[node]
        step_vars = var_groups[node]
        r = len(step_vars)

        # Lazy mode only warms leaf states eagerly
        if lazy_precompute and ch:
            precompute_skipped = True
            continue

        if ch:
            child_h_lists = [sorted(css_states[c]) for c in ch]
            if any(len(hl) == 0 for hl in child_h_lists):
                child_combos = []
            else:
                estimated = K ** r
                for hl in child_h_lists:
                    h_count = min(len(hl), max_states) if max_states > 0 else len(hl)
                    estimated *= h_count
                if estimated > max_precompute - total_kb_calls:
                    budget_exceeded = True
                    break
                child_combos = list(itertools.product(*child_h_lists))
        else:
            child_combos = [()]

        need_partial = node in y_node_asgn

        for h_combo in child_combos:
            if budget_exceeded:
                break

            rep = {}
            for ci, c in enumerate(ch):
                child_rep = css_representative[c].get(h_combo[ci], {})
                rep.update(child_rep)

            for z_vals in itertools.product(range(K), repeat=r):
                total_kb_calls += 1
                if total_kb_calls > max_precompute:
                    budget_exceeded = True
                    break

                z = [0] * n
                for vid, val in rep.items():
                    z[vid] = val
                for j, vid in enumerate(step_vars):
                    z[vid] = z_vals[j]

                if node == root and not _use_min_css:
                    css_next = _run_kb(z)
                else:
                    css_next = _extract_css(z, subtree_vars[node])

                trans_table[node][(h_combo, z_vals)] = css_next
                css_states[node].add(css_next)

                if need_partial:
                    partial_table[node][(h_combo, z_vals)] = _run_kb(z)

                if css_next not in css_representative[node]:
                    new_rep = dict(rep)
                    for j, vid in enumerate(step_vars):
                        new_rep[vid] = z_vals[j]
                    css_representative[node][css_next] = new_rep

        # Beam-prune css_states
        if max_states > 0 and len(css_states[node]) > max_states:
            freq = defaultdict(int)
            for css in trans_table[node].values():
                freq[css] += 1
            top = set(s for s, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_states])
            css_states[node] = top

    # Representative lookup for on-the-fly fallback
    _reps = {}
    for node_idx in range(num_nodes):
        for css_val, rep in css_representative[node_idx].items():
            _reps[(node_idx, css_val)] = rep

    _tt = trans_table
    _pt = partial_table
    _vg = var_groups
    _ch = children
    _n = n

    def _reconstruct_z(h_children, z_vals, node):
        rep = {}
        for ci, c in enumerate(_ch[node]):
            child_rep = _reps.get((c, h_children[ci]))
            if child_rep is None:
                return None
            rep.update(child_rep)
        z = [0] * _n
        for vid, val in rep.items():
            z[vid] = val
        for j, vid in enumerate(_vg[node]):
            z[vid] = z_vals[j]
        return z

    # -- Transition function --

    _y_cache = {}

    def _get_y_parts(y):
        if y not in _y_cache:
            _y_cache[y] = y_decompose_fn(y)
        return _y_cache[y]

    def _y_check(node, key, h_children, z_vals, y):
        """Check assigned y-parts against partial KB output. Returns False if mismatch."""
        if node not in y_node_asgn:
            return True
        partial = _pt[node].get(key)
        if partial is None:
            z = _reconstruct_z(h_children, z_vals, node)
            if z is None:
                return True
            partial = _run_kb(z)
            _pt[node][key] = partial
        try:
            h_parts = y_decompose_fn(partial)
        except Exception:
            return True
        y_parts = _get_y_parts(y)
        for j in y_node_asgn[node]:
            if j < len(h_parts) and j < len(y_parts):
                if h_parts[j] != y_parts[j]:
                    return False
        return True

    def transition_fn(h_children, z_vals, node, y):
        is_root = (node == root)
        key = (h_children, z_vals)

        # 1. Get CSS
        css = _tt[node].get(key)
        if css is None:
            z = _reconstruct_z(h_children, z_vals, node)
            if z is None:
                return None
            if is_root and not _use_min_css:
                css = _run_kb(z)
            else:
                css = _extract_css(z, subtree_vars[node])
            _tt[node][key] = css
            if (node, css) not in _reps:
                new_rep = {}
                for ci, c in enumerate(_ch[node]):
                    cr = _reps.get((c, h_children[ci]))
                    if cr:
                        new_rep.update(cr)
                for j, vid in enumerate(_vg[node]):
                    new_rep[vid] = z_vals[j]
                _reps[(node, css)] = new_rep

        # 2. Y-check (all nodes including root when min_css)
        if y_node_asgn and not _y_check(node, key, h_children, z_vals, y):
            return None

        # 3. Return
        if is_root:
            if _use_y_cond:
                # All y-parts are checked via y_node_asgn (including missing ones
                # at the root via the y-check mechanism above). If we reach here,
                # all y-checks passed → accept.
                return ("__target__", y)
            elif _use_min_css:
                z = _reconstruct_z(h_children, z_vals, node)
                if z is not None and _run_kb(z) == y:
                    return ("__target__", y)
                return None
            else:
                if css == y:
                    return ("__target__", y)
                return None

        return css

    if y_node_asgn:
        print(f"  [Tracer] y-pruning enabled: {y_node_asgn}")

    decomp = make_tree(
        var_groups=var_groups,
        children=children,
        root=root,
        transition_fn=transition_fn,
        h_final_fn=lambda y: ("__target__", y),
        n=n, H=0,
    )
    # Expose representatives for BatchDPEngine's root_output_fn
    decomp._reps = _reps

    _css_domain_sizes = [
        len(css_states[node]) for node in range(num_nodes) if node != root
    ]

    info = {
        "precompute_complete": not (budget_exceeded or precompute_skipped),
        "precompute_kb_calls": total_kb_calls,
        "precompute_mode": "lazy" if lazy_precompute else "budgeted",
        "css_domain_sizes": _css_domain_sizes,
        "var_groups": var_groups,
        "children": children,
        "root": root,
        "y_node_assignment": y_node_asgn if y_node_asgn else None,
    }
    return decomp, info
