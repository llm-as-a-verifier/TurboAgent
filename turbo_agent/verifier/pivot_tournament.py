"""
Probabilistic Pivot Tournament (PPT): O(N·k) best-of-N selection.

A round-robin tournament compares all C(N, 2) pairs of candidates — O(N^2)
verifier calls per request. PPT reaches the same selection with O(N·k)
comparisons (k = number of pivots, k << N) in three steps:

  1) Ring pass. Sample a uniformly random Hamiltonian cycle gamma over the N
     candidates and score the N adjacent directed pairs
     {(gamma_t, gamma_{t+1 mod N})}. Because the cycle is a single loop, every
     candidate appears exactly once in the "A" slot and once in the "B" slot of
     the verifier prompt, so any systematic preference of the verifier for one
     slot over the other cancels in expectation across the ring.

  2) Pivot selection. Rank candidates by their ring-pass mean preference
     w_i / c_i and take the top-k as the pivot set P. Pivots are the empirical
     leaders, so the remaining budget is spent distinguishing the strongest
     candidates rather than re-scoring weak anchors.

  3) Pivot rounds. With P fixed, score every non-pivot-vs-pivot directed pair
     (i, p) with i not in P, p in P, and every pivot-vs-pivot pair within P.
     All ring and pivot-round comparisons are aggregated into the same w_i, c_i
     and the winner is argmax_i w_i / c_i. Normalizing by c_i removes the bias
     that pivots take part in more comparisons than non-pivots.

  Total comparisons: N + k(N - k) + C(k, 2), i.e. linear in N for fixed k.

A comparison's two fine-grained rewards (R_a, R_b) become a soft win via the
Bradley-Terry model, p(a beats b) = sigmoid(R_a - R_b). This module is
selection-agnostic: the caller supplies the directed rewards per pair.
"""

import math
from itertools import combinations

DEFAULT_PIVOTS = 2


def ring_cycle(n, rng):
    """Return the N directed adjacent pairs of a uniformly random Hamiltonian
    cycle over `n` candidates: [(gamma_t, gamma_{t+1 mod N}) for t in range(N)].
    For n <= 1 there are no comparisons."""
    if n <= 1:
        return []
    perm = list(range(n))
    rng.shuffle(perm)
    return [(perm[t], perm[(t + 1) % n]) for t in range(n)]


def bradley_terry(ra, rb):
    """p(a beats b) under the Bradley-Terry model on rewards in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-(ra - rb)))


def accumulate(pairs, reward, w, c):
    """Aggregate soft wins into w, c in place from a directed reward map.

    `reward` maps a directed pair (a, b) to its fine-grained rewards
    (R_a, R_b)."""
    for a, b in pairs:
        ra, rb = reward[(a, b)]
        p = bradley_terry(ra, rb)
        w[a] += p
        c[a] += 1
        w[b] += 1.0 - p
        c[b] += 1


def select_pivots(w, c, k):
    """Top-k candidates by mean preference w_i / c_i (ties broken by index)."""
    n = len(w)
    k = min(k, n)
    order = sorted(
        range(n),
        key=lambda i: (-(w[i] / c[i] if c[i] else 0.0), i))
    return order[:k]


def pivot_round_pairs(n, pivots):
    """Directed pairs for step 3: every non-pivot vs pivot, plus pivot vs
    pivot. Non-pivots take slot A; within P the lower index takes slot A."""
    pivot_set = set(pivots)
    non_pivots = [i for i in range(n) if i not in pivot_set]
    pairs = [(i, p) for i in non_pivots for p in pivots]
    pairs += list(combinations(sorted(pivots), 2))
    return pairs


def best_index(w, c):
    """argmax_i w_i / c_i, ties broken toward the lower index."""
    n = len(w)
    return max(range(n), key=lambda i: (w[i] / c[i] if c[i] else 0.0, -i))


def mean_preferences(w, c):
    """Per-candidate mean preference w_i / c_i in [0, 1]."""
    return [w[i] / c[i] if c[i] else 0.0 for i in range(len(w))]
