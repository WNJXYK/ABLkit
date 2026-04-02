# SA-ABL: Structure-Aware Abductive Learning

## 1. Motivation

ABL (Abductive Learning) bridges neural perception and symbolic reasoning: a perception model predicts pseudo-labels z from raw inputs x, and a knowledge base (KB) revises z via abductive reasoning to be consistent with ground-truth label y. The revised z then supervises the perception model.

**The scalability bottleneck is abductive search.** For n-digit MNIST addition:

| Method | Search Space | n=5 (k=10) |
|--------|-------------|-------------|
| ABL (exhaustive) | O(k^{2n}) | 10^{10} |
| C-ABL (value curriculum) | O(\|Z_p\|^{2n}) | still exponential |
| GroundKB (pre-enumeration) | O(k^{2n}) offline + O(1) online | OOM at build time |
| DA-B&B (branch-and-bound) | O(k^{2n}) worst case, better average | still exponential worst case |

**SA-ABL reduces search to O(k^c)**, where c is the arity of the failing constraint — a small constant independent of n.

## 2. Core Idea

SA-ABL exploits the **structure of the KB** — specifically, the dependency graph among constraints and variables — to localize abductive search.

Instead of searching over all symbol positions jointly, SA-ABL:

1. **Extracts the constraint dependency graph** from the KB (offline, once)
2. **Executes logic_forward with taint tracking** — each input symbol carries a source tag; tags propagate through computation
3. **Finds the first failing constraint** along the topological order of the dependency graph
4. **Reads the taint set** of that constraint — these are the only positions that need revision
5. **Searches only over the tainted positions** — O(k^c) where c = |taint set|

## 3. Theoretical Framework

### 3.1 Constraint Dependency Graph

Given a KB with rules {R_1, ..., R_m} over symbol positions {z_1, ..., z_n}:

- **Variable nodes**: z_1, ..., z_n (the pseudo-label positions)
- **Constraint nodes**: R_1, ..., R_m (each rule/goal in the KB)
- **Edges**: z_i — R_j if rule R_j involves variable z_i

This is a **factor graph** (bipartite graph between variables and constraints).

### 3.2 Taint Tracking

Each input symbol z_i is wrapped as a TaintedValue(value, source={i}). During execution of logic_forward:

- **Arithmetic**: taint(a op b) = taint(a) ∪ taint(b)
- **Unification**: taint(X) = taint(X) ∪ taint(Y) when X unifies with Y
- **Comparison**: when a comparison goal fails, report taint(LHS) ∪ taint(RHS)

The **first failing goal** in execution order produces a taint set T ⊆ {1, ..., n}.
SA-ABL searches only over positions in T: at most k^|T| candidates.

### 3.3 Acyclic Case: Topological Processing

When the constraint dependency graph is a **DAG** (or tree):

- Process constraints in topological order
- Each constraint's inputs are either:
  - Already verified (earlier in topological order), or
  - Fresh (not yet checked)
- The first failing constraint's taint set is minimal — upstream positions are already correct

**Examples**:
- n-digit addition with carry: chain structure, taint = {d1_i, d2_i} at failing position → O(k^2)
- HWF formula evaluation: tree structure (expression tree), taint = operator + operands → O(k^3)

### 3.4 Cyclic Case: Chordalization + Tree Decomposition

When the constraint dependency graph has **cycles** (e.g., Sudoku: row/column/box constraints share cells):

1. **Chordalize** the constraint graph (triangulate: add edges until every cycle ≥ 4 has a chord)
2. **Tree decomposition**: build a junction tree whose nodes are cliques of the chordal graph
3. **Process along the junction tree** in topological order (leaf → root or any valid elimination order)
4. Each junction tree node is a **cluster** of variables; inter-cluster dependencies pass through **separators** (auxiliary variables)

Search space per step: **O(k^w)** where w = treewidth + 1 = max clique size.

In practice, taint tracking further reduces this: only the actually-tainted variables within a cluster are searched, giving **O(k^min(c, w))**.

| Task | Graph Structure | Treewidth | Taint Size (typical) | Search/step |
|------|----------------|-----------|---------------------|-------------|
| n-digit addition | chain | 2 | 2 | k^2 = 100 |
| HWF formula | tree | 2-3 | 2-3 | k^3 ≈ 1000 |
| 4×4 Sudoku | cyclic | ~7 | 2 | k^2 = 16 |
| 9×9 Sudoku | cyclic | ~20 | 2-4 | k^4 ≈ 6561 |
| Chess attack | sparse | 2-4 | 2-4 | k^4 ≈ 4096 |

### 3.5 Formal Guarantee

> **Theorem (Search Space Bound).**
> Let KB be a knowledge base whose constraint dependency graph has treewidth w,
> and let c be the maximum number of input symbols involved in any single constraint.
> Then SA-ABL's per-sample abductive search explores at most O(k^min(c, w+1)) candidates,
> independent of the total number of symbol positions n.

## 4. Algorithm

```
Algorithm: SA-ABL Abduction

Input:  pseudo_labels z = [z_1, ..., z_n]  (from perception model)
        ground_truth y
        KB with constraint dependency graph G and topological order π

Output: revised z' consistent with KB, or z if already consistent

1. TAINT INITIALIZATION
   For i = 1, ..., n:
       z̃_i ← TaintedValue(z_i, source={i})

2. TAINT-TRACKED EXECUTION
   Run logic_forward(z̃) with taint-aware Prolog interpreter
   Record execution trace: [(goal_1, result_1, taint_1), ...]

3. FIRST-FAIL DETECTION
   If logic_forward(z̃) == y:
       return z  (already consistent, no revision needed)

   Find first goal g_k in execution trace where g_k fails
   T ← taint set of g_k  (the input positions responsible)

4. LOCAL SEARCH
   For each candidate assignment c ∈ Z^|T|:  (k^|T| candidates)
       z' ← z with positions T replaced by c
       If logic_forward(z') == y:
           Collect (z', cost(z', z))  where cost = confidence distance

   Return z' with minimum cost (closest to model's prediction)
```

## 5. Implementation Plan

### 5.1 Lightweight Prolog Interpreter with Taint Tracking

Location: `ablkit/reasoning/prolog_taint/`

Components:

```
prolog_taint/
├── __init__.py
├── types.py          # TaintedValue, Term, Variable, Compound, Atom
├── parser.py         # Prolog text → AST (Horn clauses, arithmetic, lists)
├── unifier.py        # Unification with taint propagation
├── interpreter.py    # Goal resolution + execution trace recording
└── builtins.py       # is/2, =:=/2, \=/2, member/2, abs/1, etc.
```

Supported Prolog subset:
- Horn clauses (facts + rules)
- Conjunction (`,`), disjunction (`;`)
- Unification (`=`, `\=`)
- Arithmetic (`is`, `+`, `-`, `*`, `//`, `mod`, `abs`)
- Comparison (`=:=`, `=\=`, `<`, `>`, `>=`, `=<`)
- Lists (`[H|T]`, `member/2`)
- Negation-as-failure (`\+`)

Taint propagation:
- TaintedValue wraps (value, frozenset_of_source_indices)
- All operations merge source sets
- On goal failure: return (goal, merged_taint_of_all_variables_in_goal)

### 5.2 SA-ABL Reasoner

Location: `ablkit/reasoning/sabl_reasoner.py`

```python
class SABLReasoner(Reasoner):
    """Structure-Aware ABL reasoner.

    Uses taint-tracked Prolog execution to localize abductive search
    to only the positions responsible for KB constraint failure.
    """

    def __init__(self, kb, pl_source, ...):
        # Parse Prolog KB once
        # Build constraint dependency graph
        # Compute tree decomposition if cyclic

    def abduce(self, data_example):
        # 1. Wrap pseudo-labels as TaintedValues
        # 2. Run taint-tracked logic_forward
        # 3. Find first-fail, get taint set T
        # 4. Search over k^|T| candidates at positions T
        # 5. Return best candidate by confidence distance
```

### 5.3 KB Examples in Prolog

**n-digit addition (carry-aware):**
```prolog
% logic_forward(Digits, Sum) — Digits = [D1_1,...,D1_n, D2_1,...,D2_n]
logic_forward(Digits, Sum) :-
    length(Digits, Len),
    Half is Len // 2,
    split(Digits, Half, Num1Digits, Num2Digits),
    reverse(Num1Digits, R1),
    reverse(Num2Digits, R2),
    add_carry(R1, R2, 0, RevResult),
    reverse(RevResult, Result),
    digits_to_number(Result, Sum).

add_carry([], [], Carry, [Carry]) :- Carry > 0.
add_carry([], [], 0, []).
add_carry([D1|T1], [D2|T2], CIn, [S|Rest]) :-
    S0 is D1 + D2 + CIn,
    S is S0 mod 10,
    COut is S0 // 10,
    add_carry(T1, T2, COut, Rest).
```

Dependency graph: chain (D1_i, D2_i) → carry_i → (D1_{i-1}, D2_{i-1})
Taint per step: {D1_i, D2_i} → search k^2 = 100

**Chess queen attack:**
```prolog
logic_forward([R1, C1, R2, C2], Attack) :-
    (   R1 =:= R2, Attack = true
    ;   C1 =:= C2, Attack = true
    ;   DR is abs(R1 - R2), DC is abs(C1 - C2),
        DR =:= DC, Attack = true
    ;   Attack = false
    ).
```

**4×4 Sudoku (uniqueness constraints):**
```prolog
logic_forward(Board, valid) :-
    Board = [R1, R2, R3, R4],
    maplist(all_different, [R1, R2, R3, R4]),
    transpose(Board, Cols),
    maplist(all_different, Cols),
    boxes(Board, Boxes),
    maplist(all_different, Boxes).

all_different([]).
all_different([H|T]) :-
    not_member(H, T),
    all_different(T).

not_member(_, []).
not_member(X, [H|T]) :-
    X \= H,
    not_member(X, T).
```

Dependency graph: cyclic (row/column/box share cells)
Tree decomposition needed; taint per failing `\=` check: 2 cells → search k^2

### 5.4 Integration with ABL Framework

In `ablkit/reasoning/__init__.py`, export `SABLReasoner`.

In example `main.py`, add `--kb sabl` option:
```python
elif kb_type == "sabl":
    kb = AddKBBase()
    reasoner = SABLReasoner(kb, pl_file="add_carry.pl")
```

### 5.5 Experiments

| Experiment | Purpose |
|-----------|---------|
| mnist_add n=1,2,3,5 | Compare search counts: base / dabb / sabl |
| mnist_add n=5 direct training | Verify SA-ABL can train where base/cached cannot |
| mnist_add n=2 → n=5 transfer | Combine SA-ABL with curriculum transfer |
| Chess attack (if time permits) | Validate generality on non-chain KB |

Key metrics:
- `kb_search_count` per epoch (already instrumented)
- Symbol accuracy, reasoning accuracy
- Wall-clock time per epoch
- Convergence speed (epochs to 95% final accuracy)

## 6. Summary

SA-ABL's contribution is a single, clean idea:

**Don't search where the KB tells you not to.**

The constraint dependency graph of the KB reveals which positions can possibly be responsible for a constraint violation. Taint tracking identifies them precisely. The result is a search space reduction from O(k^n) to O(k^c) — exponential to constant — for any KB with bounded constraint arity.

The implementation requires:
1. A taint-tracking Prolog interpreter (~600 lines Python)
2. A new SABLReasoner class (~200 lines)
3. Prolog KB files for each task (~50 lines each)

Total new code: ~850 lines.
