"""
拓扑生成 -- 统一 JT+CSS (source-set tree + interaction-graph fallback)。

两阶段策略:
1. Source-set tree (主策略): 从 bottleneck source set 的 Hasse 图构建
2. Interaction-graph JT (fallback): 交互图 + min-fill 消元
"""


# ============================================================
# 工具函数 -- 验证、剪枝、二叉化、压缩
# ============================================================


def _verify_coverage(var_groups, n):
    """Check that var_groups partitions {0, ..., n-1} exactly."""
    all_vars = sorted(v for vg in var_groups for v in vg)
    return all_vars == list(range(n))


def _prune_empty(var_groups, children, root):
    """
    剪枝空节点 (var_groups 为空且无有效子节点的节点)。

    - 空叶节点: 直接删除 (标记 var_groups 为 None)
    - 空中间节点 (仅一个子节点): 折叠，将子节点提升到父节点位置
    - 反复迭代直到无变化

    Returns: 更新后的 root 索引
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(var_groups)):
            if var_groups[i] is None:
                continue
            children[i] = [c for c in children[i] if var_groups[c] is not None]
            if not var_groups[i] and not children[i] and i != root:
                var_groups[i] = None
                changed = True
            elif not var_groups[i] and len(children[i]) == 1 and i != root:
                child = children[i][0]
                var_groups[i] = var_groups[child]
                children[i] = children[child]
                var_groups[child] = None
                children[child] = []
                changed = True
    children[root] = [c for c in children[root] if var_groups[c] is not None]
    while not var_groups[root] and len(children[root]) == 1:
        old_root = root
        root = children[old_root][0]
        var_groups[old_root] = None
        children[old_root] = []
    return root


def _compact(var_groups, children, root):
    """
    压缩: 删除被标记为 None 的节点，重新编号为连续索引。

    Returns: (new_var_groups, new_children, new_root)
    """
    alive = [i for i in range(len(var_groups)) if var_groups[i] is not None]
    remap = {old: new for new, old in enumerate(alive)}
    new_vg = [var_groups[i] for i in alive]
    new_ch = [[remap[c] for c in children[i]] for i in alive]
    new_root = remap[root]
    return new_vg, new_ch, new_root


def _binarize(var_groups, children):
    """
    将多叉树原地二叉化 (右倾)。

    当某节点有 m > 2 个子节点 [c0, c1, ..., c_{m-1}] 时，
    插入中间节点 (空 var_groups) 将其拆分为二叉:

        node                    node
       / | \\      ->           /    \\
      c0 c1 c2              c0    aux0
                                 /    \\
                               c1     c2

    这样 DP 分配修改量从 O(k^{m-1}) 降为 O(k) per node。
    """
    i = 0
    while i < len(var_groups):
        if len(children[i]) > 2:
            while len(children[i]) > 2:
                right = children[i].pop()
                left = children[i].pop()
                aux = len(var_groups)
                var_groups.append([])
                children.append([left, right])
                children[i].append(aux)
        i += 1


# ============================================================
# Interaction-graph JT (fallback 策略)
# ============================================================


def _build_interaction_graph(node_by_pos, n):
    """从静态图的 source set 构建变量交互图 (邻接表)。

    同一个 source set 中的所有变量两两连边，表示它们在 KB 计算中存在交互。
    排除全集 source set (= 最终输出节点)，因为它会使图变为完全图。
    """
    full = frozenset(range(n))
    adj = [set() for _ in range(n)]
    for pos, entry in node_by_pos.items():
        src = entry["sources"]
        if src is None or len(src) <= 1 or src == full:
            continue
        src_list = sorted(src)
        for i in range(len(src_list)):
            for j in range(i + 1, len(src_list)):
                u, v = src_list[i], src_list[j]
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _minfill_eliminate(adj, n):
    """Min-fill 贪心消元: 每步选填充边最少的变量消去。

    Returns:
        ordering: 消元顺序 [v0, v1, ..., v_{n-1}]
        bags: 每步的 bag = frozenset({被消去的变量} | {活跃邻居})
    """
    neighbors = [set(a) for a in adj]
    eliminated = set()
    ordering = []
    bags = []

    for _ in range(n):
        best_v, best_fill = -1, float("inf")
        best_active = []
        for v in range(n):
            if v in eliminated:
                continue
            active = [u for u in neighbors[v] if u not in eliminated]
            fill = 0
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    if active[j] not in neighbors[active[i]]:
                        fill += 1
            if fill < best_fill or (fill == best_fill and len(active) < len(best_active)):
                best_fill = fill
                best_v = v
                best_active = active

        bag = frozenset({best_v} | set(best_active))
        bags.append(bag)
        ordering.append(best_v)

        for i in range(len(best_active)):
            for j in range(i + 1, len(best_active)):
                neighbors[best_active[i]].add(best_active[j])
                neighbors[best_active[j]].add(best_active[i])

        eliminated.add(best_v)

    return ordering, bags


def _interaction_graph_jt(static_graph, n, K):
    """
    从交互图经 min-fill 消元构建 Junction Tree 拓扑 (fallback 策略)。

    当 source set Hasse 图无法产生有效分解时，退回到交互图 + min-fill。
    注意: 交互图是成对关系，会丢失 CSS 压缩结构 (source set 中多变量
    共享的中间值被展开为 clique)，因此代价通常高于 source-set tree。

    算法:
    1. 从 source set 构建交互图
    2. Min-fill 消元 -> bags
    3. 构建 clique tree (parent = 第一个共享变量的后续 bag)
    4. 合并子集 bag (bag_i <= bag_{parent_i})
    5. 后序分配变量到最深的 bag
    6. 剪枝、二叉化、压缩

    Returns: (var_groups, children, root, meta) or None
    """
    node_by_pos = static_graph["node_by_pos"]
    adj = _build_interaction_graph(node_by_pos, n)

    if not any(adj):
        return None

    ordering, raw_bags = _minfill_eliminate(adj, n)
    treewidth = max(len(b) for b in raw_bags) - 1

    # 构建 clique tree: parent[i] = 第一个与 bag_i 共享变量的后续 bag
    num_raw = len(raw_bags)
    parent = [None] * num_raw
    for i in range(num_raw):
        for j in range(i + 1, num_raw):
            if raw_bags[i] & raw_bags[j]:
                parent[i] = j
                break

    # 合并子集 bag
    alive = [True] * num_raw
    for i in range(num_raw):
        if parent[i] is not None and raw_bags[i] <= raw_bags[parent[i]]:
            alive[i] = False

    def _find_alive_ancestor(i):
        p = parent[i]
        while p is not None and not alive[p]:
            p = parent[p]
        return p

    idx_map = {}
    final_bags = []
    for i in range(num_raw):
        if alive[i]:
            idx_map[i] = len(final_bags)
            final_bags.append(raw_bags[i])

    num = len(final_bags)
    if num <= 1:
        return None

    children_list = [[] for _ in range(num)]

    for i in range(num_raw):
        if not alive[i]:
            continue
        ancestor = _find_alive_ancestor(i)
        if ancestor is not None and alive[ancestor]:
            ni, na = idx_map[i], idx_map[ancestor]
            children_list[na].append(ni)

    has_parent = set()
    for ch_list in children_list:
        has_parent.update(ch_list)
    roots = [i for i in range(num) if i not in has_parent]

    if len(roots) > 1:
        vr = num
        final_bags.append(frozenset())
        children_list.append(roots)
        root = vr
        num += 1
    elif len(roots) == 1:
        root = roots[0]
    else:
        return None

    # 后序分配变量到最深 bag
    var_groups = [[] for _ in range(num)]
    assigned = set()

    def _assign(nd):
        for ch in children_list[nd]:
            _assign(ch)
        bag = final_bags[nd] if nd < len(final_bags) else frozenset()
        for v in sorted(bag):
            if v not in assigned:
                var_groups[nd].append(v)
                assigned.add(v)

    _assign(root)

    for v in range(n):
        if v not in assigned:
            var_groups[root].append(v)
            assigned.add(v)

    if not _verify_coverage(var_groups, n):
        return None

    root = _prune_empty(var_groups, children_list, root)
    non_empty = sum(1 for vg in var_groups if vg is not None)
    if non_empty <= 1:
        return None

    _binarize(var_groups, children_list)
    var_groups, children_list, root = _compact(var_groups, children_list, root)

    bag_cost = sum(K ** len(b) for b in final_bags if b)
    meta = {
        "treewidth": treewidth,
        "bag_cost": bag_cost,
    }
    return var_groups, children_list, root, meta


# ============================================================
# Source-set tree (主策略)
# ============================================================


def _source_set_tree(bottlenecks, n, min_compression=1.5):
    """
    从 bottleneck source set 的 Hasse 图构建树拓扑 (主策略)。

    Source set 的包含层次自然保留 CSS 压缩: 每个 source set 对应一个
    瓶颈节点，其 CSS 域大小远小于 K^|vars| (如加法进位 H=2)。

    与链式拓扑不同，树拓扑允许互不包含的 source set 作为兄弟节点，
    从而覆盖更多可分解的约束结构。链式拓扑是树的特例 (所有 source set
    嵌套时退化为链)。

    算法:
    1. 筛选有效 bottleneck (压缩率 >= min_compression，非常量，非全集)
    2. 按包含关系构建 Hasse 图 (偏序的直接覆盖关系)
    3. 每个 source set 对应一个节点，var_groups = 本集合减去子集合的并
    4. 无父节点的 source set 共享根节点 (根处理剩余变量)

    Returns:
        (var_groups, children, root, meta) or None
    """
    valid = []
    for b in bottlenecks:
        if b["domain_size"] <= 1:
            continue
        if b["compression"] < min_compression:
            continue
        if b["sources"] == frozenset(range(n)):
            continue
        valid.append(b)

    if not valid:
        return None

    best_by_src = {}
    for b in valid:
        src = b["sources"]
        if src not in best_by_src or b["compression"] > best_by_src[src]["compression"]:
            best_by_src[src] = b
    sets = sorted(best_by_src.keys(), key=lambda s: len(s))

    if not sets:
        return None

    # 构建 Hasse 图: parent[i] = 包含 sets[i] 的最小 sets[j] (直接覆盖)
    num = len(sets)
    parent = [None] * num
    for i in range(num):
        for j in range(i + 1, num):
            if sets[i] < sets[j]:
                parent[i] = j
                break

    inner_roots = [i for i in range(num) if parent[i] is None]

    children = [[] for _ in range(num)]
    for i in range(num):
        if parent[i] is not None:
            children[parent[i]].append(i)

    assigned = set()
    var_groups = [[] for _ in range(num)]

    def _assign_vars(i):
        for ch in children[i]:
            _assign_vars(ch)
        for v in sorted(sets[i]):
            if v not in assigned:
                var_groups[i].append(v)
                assigned.add(v)

    for r in inner_roots:
        _assign_vars(r)

    root_vars = sorted(set(range(n)) - assigned)

    root = num
    var_groups.append(root_vars)
    children.append(inner_roots)

    if not _verify_coverage(var_groups, n):
        return None

    if num == 0:
        return None

    root = _prune_empty(var_groups, children, root)

    non_empty = sum(1 for vg in var_groups if vg is not None)
    if non_empty <= 1:
        return None

    _binarize(var_groups, children)

    var_groups, children, root = _compact(var_groups, children, root)

    meta = {}
    return var_groups, children, root, meta


# ============================================================
# 统一入口
# ============================================================


def _jt_topology(bottlenecks, static_graph, n, K, min_compression=1.5):
    """
    统一 JT+CSS 拓扑生成器。

    两阶段策略:
    1. **Source-set tree** (主策略): 从 bottleneck source set 的 Hasse 图
       构建树拓扑。保留 CSS 压缩结构，链式拓扑是其嵌套特例。
    2. **Interaction-graph JT** (fallback): 当 source set 无法产生有效
       分解时 (无有效瓶颈)，退回到交互图 + min-fill 消元。代价较高
       但保证能处理任意交互结构。

    Returns: (var_groups, children, root, meta) or None
    """
    result = _source_set_tree(bottlenecks, n, min_compression)
    if result is not None:
        return result

    return _interaction_graph_jt(static_graph, n, K)
