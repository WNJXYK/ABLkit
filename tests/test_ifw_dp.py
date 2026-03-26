"""Tests for IFW-DP: CPU and GPU modules."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ablkit.reasoning.ifw_dp import (
    make_addition_decomposition,
    make_mod_decomposition,
    precompute_transitions,
    dp_map,
    dp_marginal,
    sparse_dp_map,
    sparse_dp_marginal,
)


def test_addition_map_basic():
    """Test MAP DP on 2-digit addition: 38 + 19 = 57."""
    d = 2
    K = 10
    decomp = make_addition_decomposition(d)
    y = 57

    # z = [a0, a1, b0, b1] = [8, 3, 9, 1] (38 + 19)
    # Simulate: model correctly predicts each digit
    log_p = [[math.log(1e-6)] * K for _ in range(2 * d)]
    # Ground truth: a=38 → a0=8, a1=3; b=19 → b0=9, b1=1
    log_p[0][8] = 0.0  # a0 = 8
    log_p[1][3] = 0.0  # a1 = 3
    log_p[2][9] = 0.0  # b0 = 9
    log_p[3][1] = 0.0  # b1 = 1

    trans = precompute_transitions(decomp, K, y)
    z_hat, score = dp_map(decomp, trans, log_p)

    assert z_hat == [8, 3, 9, 1], f"Expected [8,3,9,1], got {z_hat}"
    print(f"  MAP basic: z_hat={z_hat}, score={score:.4f}  ✓")


def test_addition_map_ambiguous():
    """Test MAP DP when model is uncertain between two valid decompositions."""
    d = 1
    K = 10
    decomp = make_addition_decomposition(d)
    y = 7  # a + b = 7

    # Model thinks a0≈3 and b0≈4, but also somewhat likely a0=2, b0=5
    log_p = [[math.log(0.01)] * K for _ in range(2)]
    log_p[0][3] = math.log(0.5)  # a=3 most likely
    log_p[0][2] = math.log(0.3)  # a=2 second
    log_p[1][4] = math.log(0.5)  # b=4 most likely
    log_p[1][5] = math.log(0.3)  # b=5 second

    trans = precompute_transitions(decomp, K, y)
    z_hat, score = dp_map(decomp, trans, log_p)

    a, b = z_hat[0], z_hat[1]
    assert a + b == y, f"Constraint violated: {a} + {b} != {y}"
    assert a == 3 and b == 4, f"Expected [3,4], got [{a},{b}]"
    print(f"  MAP ambiguous: z_hat={z_hat}, score={score:.4f}  ✓")


def test_addition_marginal():
    """Test marginal DP on 1-digit addition."""
    d = 1
    K = 10
    decomp = make_addition_decomposition(d)
    y = 5  # a + b = 5

    # Uniform perception
    p = [[1.0 / K] * K for _ in range(2)]

    trans = precompute_transitions(decomp, K, y)
    q, Z = dp_marginal(decomp, trans, p)

    # Check: q[0] and q[1] should be proper distributions
    sum_q0 = sum(q[0])
    sum_q1 = sum(q[1])
    assert abs(sum_q0 - 1.0) < 1e-6, f"q[0] sums to {sum_q0}"
    assert abs(sum_q1 - 1.0) < 1e-6, f"q[1] sums to {sum_q1}"

    # There are 6 valid pairs: (0,5),(1,4),(2,3),(3,2),(4,1),(5,0)
    # With uniform prior, each should have equal posterior
    for a in range(6):
        assert abs(q[0][a] - 1.0 / 6) < 1e-6, f"q[0][{a}] = {q[0][a]}"
    for a in range(6, K):
        assert abs(q[0][a]) < 1e-6, f"q[0][{a}] should be 0, got {q[0][a]}"

    print(f"  Marginal: Z={Z:.6f}, q[0][:6]={[round(x,3) for x in q[0][:6]]}  ✓")


def test_addition_2digit_marginal():
    """Test marginal DP on 2-digit addition with carry."""
    d = 2
    K = 10
    decomp = make_addition_decomposition(d)
    y = 100  # e.g. 50 + 50, 91 + 09, etc.

    # Peaked perception
    p = [[0.02] * K for _ in range(4)]
    # a=55 → a0=5,a1=5; b=45 → b0=5,b1=4
    p[0][5] = 0.8
    p[1][5] = 0.8
    p[2][5] = 0.8
    p[3][4] = 0.8

    trans = precompute_transitions(decomp, K, y)
    q, Z = dp_marginal(decomp, trans, p)

    assert Z > 0, "Z should be positive"
    # q[0][5] should be high (a0=5 is consistent with 55+45=100)
    assert q[0][5] > 0.3, f"q[0][5] = {q[0][5]}, expected > 0.3"
    print(f"  2-digit marginal: Z={Z:.6e}, q[0][5]={q[0][5]:.4f}  ✓")


def test_mod_arithmetic():
    """Test modular arithmetic decomposition."""
    n = 4
    p_mod = 7
    K = 10
    decomp = make_mod_decomposition(n, p_mod, op="add")

    # Target: z0 + z1 + z2 + z3 ≡ 3 (mod 7)
    y = 3

    log_p = [[math.log(1.0 / K)] * K for _ in range(n)]

    trans = precompute_transitions(decomp, K, y)
    z_hat, score = dp_map(decomp, trans, log_p)

    total = sum(z_hat) % p_mod
    assert total == y, f"Constraint violated: sum={sum(z_hat)}, mod 7 = {total}, expected {y}"
    print(f"  Mod arithmetic: z_hat={z_hat}, sum%7={total}  ✓")


def test_sparse_dp():
    """Test sparse DP (hash-table version) gives same result as dense."""
    d = 2
    K = 10
    decomp = make_addition_decomposition(d)
    y = 57

    log_p = [[math.log(0.05)] * K for _ in range(4)]
    log_p[0][8] = math.log(0.6)
    log_p[1][3] = math.log(0.6)
    log_p[2][9] = math.log(0.6)
    log_p[3][1] = math.log(0.6)

    # Dense DP
    trans = precompute_transitions(decomp, K, y)
    z_dense, score_dense = dp_map(decomp, trans, log_p)

    # Sparse DP
    z_sparse, score_sparse = sparse_dp_map(decomp, K, y, log_p)

    assert z_dense == z_sparse, f"Dense {z_dense} != Sparse {z_sparse}"
    assert abs(score_dense - score_sparse) < 1e-6
    print(f"  Sparse DP: matches dense  ✓")


def test_sparse_marginal():
    """Test sparse marginal DP."""
    d = 1
    K = 10
    decomp = make_addition_decomposition(d)
    y = 5

    p = [[1.0 / K] * K for _ in range(2)]

    # Dense
    trans = precompute_transitions(decomp, K, y)
    q_dense, Z_dense = dp_marginal(decomp, trans, p)

    # Sparse
    q_sparse, Z_sparse = sparse_dp_marginal(decomp, K, y, p)

    assert abs(Z_dense - Z_sparse) < 1e-6
    for i in range(2):
        for k in range(K):
            assert abs(q_dense[i][k] - q_sparse[i][k]) < 1e-6
    print(f"  Sparse marginal: matches dense  ✓")


def test_tracer():
    """Test operator overloading tracer."""
    from ablkit.reasoning.tracer import trace_kb, find_bottlenecks

    # KB: digit-level addition (must use TracedValue-compatible ops)
    def kb_digit_add(z):
        d = len(z) // 2
        # Build numbers digit by digit (TracedValue-compatible)
        a = z[0]
        for i in range(1, d):
            a = a + z[i] * (10**i)
        b = z[d]
        for i in range(1, d):
            b = b + z[d + i] * (10**i)
        return a + b

    # Trace with concrete values
    graph = trace_kb(kb_digit_add, [8, 3, 9, 1], n=4)
    print(f"  Tracer: {len(graph.nodes)} nodes recorded")
    graph.print_graph()

    # Find bottlenecks
    bottlenecks = find_bottlenecks(kb_digit_add, n=4, K=10, n_trials=100)
    print(f"  Found {len(bottlenecks)} bottlenecks:")
    for b in bottlenecks[:5]:
        print(
            f"    sources={sorted(b.sources)}, domain={b.domain_size}, "
            f"compression={b.compression_ratio:.1f}x, op={b.node_op}"
        )
    print("  ✓")


def test_gpu_basic():
    """Test GPU batched DP (using numpy as backend)."""
    try:
        import numpy as np
    except ImportError:
        print("  GPU test: numpy not available, skipping")
        return

    from ablkit.reasoning.ifw_gpu import BatchedAdditionDP

    d = 2
    K = 10
    B = 4
    engine = BatchedAdditionDP(d=d, K=K, xp=np)

    # Batch of 4 samples
    p = np.full((B, 2 * d, K), 0.02)

    # Sample 0: 38+19=57
    p[0, 0, 8] = 0.8; p[0, 1, 3] = 0.8; p[0, 2, 9] = 0.8; p[0, 3, 1] = 0.8
    # Sample 1: 25+32=57
    p[1, 0, 5] = 0.8; p[1, 1, 2] = 0.8; p[1, 2, 2] = 0.8; p[1, 3, 3] = 0.8
    # Sample 2: 10+10=20
    p[2, 0, 0] = 0.8; p[2, 1, 1] = 0.8; p[2, 2, 0] = 0.8; p[2, 3, 1] = 0.8
    # Sample 3: 99+01=100
    p[3, 0, 9] = 0.8; p[3, 1, 9] = 0.8; p[3, 2, 1] = 0.8; p[3, 3, 0] = 0.8

    y = np.array([57, 57, 20, 100])

    q, Z = engine.marginal(p, y)

    print(f"  GPU marginal: Z={Z}")
    for b in range(B):
        a0 = int(np.argmax(q[b, 0]))
        a1 = int(np.argmax(q[b, 1]))
        b0 = int(np.argmax(q[b, 2]))
        b1 = int(np.argmax(q[b, 3]))
        num_a = a1 * 10 + a0
        num_b = b1 * 10 + b0
        print(f"    sample {b}: {num_a}+{num_b}={num_a+num_b} (target={y[b]})")
    print("  ✓")


if __name__ == "__main__":
    print("=== IFW-DP Tests ===\n")

    print("[1] Addition MAP (basic)")
    test_addition_map_basic()

    print("[2] Addition MAP (ambiguous)")
    test_addition_map_ambiguous()

    print("[3] Addition Marginal (1-digit)")
    test_addition_marginal()

    print("[4] Addition Marginal (2-digit with carry)")
    test_addition_2digit_marginal()

    print("[5] Modular Arithmetic")
    test_mod_arithmetic()

    print("[6] Sparse DP (MAP)")
    test_sparse_dp()

    print("[7] Sparse DP (Marginal)")
    test_sparse_marginal()

    print("[8] Tracer")
    test_tracer()

    print("[9] GPU Batched DP")
    test_gpu_basic()

    print("\n=== All tests passed ===")
