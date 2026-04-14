"""
自动静态图提取与 CSS 引导的 IFW 分解 (Automatic IFW Decomposition Discovery)

本包假设 KB 在 TracedValue 输入上遵循固定的、无分支的执行图。
在此假设下，完成以下流水线:

1. 多次追踪执行 -> 恢复静态节点-位置/源集合图
2. 基于观测域压缩率评估 CSS 候选
3. 生成候选拓扑 (统一 JT+CSS: source-set tree 优先，interaction-graph fallback)
4. 可选: 将 y-pruning 纳入代价估计
5. 构建经验证的 IFW 分解，失败时自动回退

模块结构:
- traced_value: TracedValue 追踪对象 + 编译求值器
- static_graph: 静态图提取与瓶颈发现
- topology: 拓扑生成 (source-set tree + interaction-graph JT)
- builder: 分解构建器 (CSS 提取 + 转移表预计算)
- discover: 主编排器 (发现 + 验证 + 回退)
"""

from .traced_value import TracedValue, compile_graph
from .static_graph import _extract_static_graph, find_bottlenecks
from .topology import _interaction_graph_jt, _source_set_tree, _jt_topology
from .builder import _build_decomposition, _compute_subtree_vars, _assign_y_parts
from .discover import discover_decomposition, _verify_decomposition, _build_brute_chain

__all__ = [
    # Public API
    "TracedValue",
    "compile_graph",
    "discover_decomposition",
    "find_bottlenecks",
    # Internal (used by tests)
    "_extract_static_graph",
    "_interaction_graph_jt",
    "_verify_decomposition",
]
