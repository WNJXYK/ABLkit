"""
主编排器 -- 发现瓶颈 -> 拓扑 -> 构建 + 验证。

包含:
- discover_decomposition: 完整流水线
- _verify_decomposition: 随机检验分解正确性
- _build_brute_chain: 暴力搜索回退
"""

import random
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Any

from ..ifw_dp import Decomposition, dp_map, make_chain
from .traced_value import TracedValue
from .static_graph import find_bottlenecks
from .topology import _jt_topology
from .builder import _build_decomposition


# ============================================================
# 代价计算
# ============================================================


def _actual_cost(var_groups, children, root, css_domain_sizes, K):
    """从实际 CSS 域大小计算 DP 代价。

    cost = sum_node K^r_i * prod(child_css_domain)
    其中 r_i = |var_groups[i]|, child_css_domain 来自 build 后的实际域大小。
    """
    num_nodes = len(var_groups)
    domain_by_node = {}
    idx = 0
    for i in range(num_nodes):
        if i != root:
            if idx < len(css_domain_sizes):
                domain_by_node[i] = css_domain_sizes[idx]
                idx += 1
            else:
                domain_by_node[i] = K ** len(var_groups[i])

    total = 0
    for i in range(num_nodes):
        r = len(var_groups[i])
        child_product = 1
        for ch in children[i]:
            child_product *= domain_by_node.get(ch, K)
        total += child_product * (K ** r)
    return total


# ============================================================
# 主编排器
# ============================================================


def discover_decomposition(
    logic_forward: Callable,
    n: int,
    K: int,
    num_samples: int = 500,
    seed: int = 0,
    min_compression: float = 1.5,
    max_precompute: int = 100_000,
    max_states: int = 0,
    lazy_precompute: bool = False,
    constraint_fn: Optional[Callable] = None,
    y_size: int = 0,
    Y_domains: Optional[List[int]] = None,
    y_decompose_fn: Optional[Callable] = None,
) -> Tuple[Decomposition, list]:
    """
    从追踪的静态图自动发现 IFW 分解。

    流水线: 发现瓶颈 -> 生成 JT+CSS 拓扑 -> 构建 + 验证 ->
    验证失败时回退 brute-force (安全网)。
    """
    brute_cost = K ** n

    # 1. 发现瓶颈 + y-check 结构
    bottlenecks, static_graph, y_check_info = find_bottlenecks(
        logic_forward, n, K, num_samples, seed,
        constraint_fn=constraint_fn, y_size=y_size, Y_domains=Y_domains,
        y_decompose_fn=y_decompose_fn,
    )
    if not static_graph["is_static"]:
        print(
            "  [Tracer] WARNING: trace schema mismatches detected "
            f"({static_graph['schema_mismatches']} mismatches over {static_graph['num_traces']} accepted traces)"
        )
    y_info = y_check_info if y_decompose_fn else None

    # CSS 候选位置: 每个 source set 处按域大小排序，逐级尝试
    _css_by_src = defaultdict(list)
    for pos, entry in static_graph["node_by_pos"].items():
        src = entry["sources"]
        if src is not None:
            _css_by_src[src].append((len(entry["values"]), pos))
    for src in _css_by_src:
        _css_by_src[src].sort()
        deduped = []
        seen_doms = set()
        for dom, pos in _css_by_src[src]:
            if dom not in seen_doms:
                seen_doms.add(dom)
                deduped.append((dom, pos))
        _css_by_src[src] = deduped

    def _css_candidates(max_tries=8):
        """Yield min_css_pos dicts with progressively larger domains, then None."""
        for idx in range(max_tries):
            css_pos = {}
            for src, cands in _css_by_src.items():
                i = min(idx, len(cands) - 1)
                css_pos[src] = cands[i][1]
            yield css_pos
        yield None  # full tuple fallback

    def _make_info(decomp_type, extra=None):
        info = {
            "decomposition_type": decomp_type,
            "bottlenecks": bottlenecks[:10],
            "static_graph": {
                "is_static": static_graph["is_static"],
                "num_traces": static_graph["num_traces"],
                "schema_mismatches": static_graph["schema_mismatches"],
                "num_nodes": static_graph["num_nodes"],
                "css_candidates": static_graph.get("css_candidates", []),
            },
            "estimated_brute-force_cost": brute_cost,
        }
        if extra:
            info.update(extra)
        return info

    # 2. 生成 JT 拓扑
    topo = _jt_topology(bottlenecks, static_graph, n, K, min_compression)

    if topo is not None:
        vg, ch, rt, meta = topo

        # 3. 构建 + 验证 (从最小 CSS 域逐级尝试)
        _shared_compiled = {}
        for css_pos in _css_candidates():
            try:
                decomp, build_info = _build_decomposition(
                    logic_forward, n, K, vg, ch, rt, max_precompute,
                    y_check_info=y_info, y_decompose_fn=y_decompose_fn,
                    min_css_pos=css_pos,
                    lazy_precompute=lazy_precompute,
                    max_states=max_states,
                    compiled_cache=_shared_compiled,
                    y_size=y_size,
                )
            except Exception:
                continue

            if not _verify_decomposition(decomp, logic_forward, n, K, max_states=max_states):
                continue

            # 验证通过
            css_domains = build_info.get("css_domain_sizes", [])
            jt_cost = _actual_cost(vg, ch, rt, css_domains, K)

            build_info.update(meta)
            build_info.setdefault(
                "state_mode",
                "minimal-css" if (y_decompose_fn and css_pos is not None) else "css",
            )
            build_info.update(_make_info("jt"))
            build_info["estimated_jt_cost"] = jt_cost
            _print_result(n, K, brute_cost, jt_cost, css_domains)
            return decomp, build_info

    # JT 拓扑不存在或所有 CSS 级别验证失败 -> brute-force (安全网)
    print(f"  [Tracer] JT failed (n={n}, K={K}), falling back to brute-force")
    return _build_brute_chain(logic_forward, n, K), _make_info("brute-force")


def _print_result(n, K, brute_cost, jt_cost, css_domains):
    """打印分解结果摘要。"""
    speedup = brute_cost / max(jt_cost, 1)
    print(f"  [Tracer] JT decomposition (n={n}, K={K}): "
          f"cost={jt_cost:,} (speedup: {speedup:,.1f}x, css={css_domains})")


# ============================================================
# 验证与回退 -- 随机检验分解正确性，失败时退回暴力搜索
# ============================================================


def _verify_decomposition(decomp, logic_forward, n, K, num_checks=20, seed=12345, max_states=0):
    """
    Verify a decomposition by checking random inputs against dp_map.

    Runs logic_forward on random valid inputs, then checks if dp_map
    with perfect log-probs (true z gets 0, others get -inf) recovers
    the exact original z. Returns True if all checks pass.
    """
    rng = random.Random(seed)
    for _ in range(num_checks):
        z = [rng.randint(0, K - 1) for _ in range(n)]
        res = logic_forward(z)
        if isinstance(res, TracedValue):
            res = res.value
        if res is None:
            continue

        log_p = [[-100.0] * K for _ in range(n)]
        for i in range(n):
            log_p[i][z[i]] = 0.0

        z_hat, score = dp_map(decomp, K, res, log_p, max_states=max_states)
        if score <= float("-inf") or z_hat != z:
            return False
        y_hat = logic_forward(z_hat)
        if isinstance(y_hat, TracedValue):
            y_hat = y_hat.value
        if y_hat != res:
            return False
    return True


def _build_brute_chain(logic_forward, n, K):
    """Fallback: one variable per step, no compression."""
    def transition_fn(h_prev, z_vals, step, y):
        prev = dict(h_prev) if h_prev else {}
        prev[step] = z_vals[0]
        if step == n - 1:
            z = [prev.get(i, 0) for i in range(n)]
            res = logic_forward(z)
            if isinstance(res, TracedValue):
                res = res.value
            if res == y:
                return ("__target__", y)
            return None
        return tuple(sorted(prev.items()))

    return make_chain(
        L=n, var_groups=[[i] for i in range(n)],
        transition_fn=transition_fn, h_init=(),
        h_final_fn=lambda y: ("__target__", y), n=n, H=0,
    )
