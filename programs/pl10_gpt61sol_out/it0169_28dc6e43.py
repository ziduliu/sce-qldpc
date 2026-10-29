"""Initial program: lifted-product CSS code, ONE (group, shape) bed per
program, N_SLOTS (env CM_NSLOTS) freely-filled candidates per generation.

SEARCH SPACE (the only constraints; everything else is free):
- The group and the protograph shapes are fixed for the whole program
  (genotype, editable between generations). Each of the 20 slots fills
  the cells however it wants: PER SLOT, every cell holds 0..5 DISTINCT
  group elements (an empty cell is a zero block) - term counts AND
  elements are both the slot's choice. Identical fillings are
  deduplicated downstream and waste the slot.
- Size law: n = (ca*cb + ra*rb) * q for A of shape (ra, ca), B of shape
  (rb, cb), |G| = q. n must land in [CM_NMIN, CM_NMAX) (default
  [100, 600)); the scaffold picks the largest valid q.
- PER-SLOT validity (scaffold-checked; a violating slot is voided, not
  repaired): max row weight <= CM_WMAX (an HX row sums A-row i with
  B-column j; an HZ row sums a B-row with an A-column - check BOTH),
  no all-zero protograph row or column on either side (a zero A-row
  leaves its sector-2 block untouched by every Z check, which is a
  weight-1 logical - d=1 - unless the matching B column is a single
  1-term cell; symmetric for the other three cases), cells distinct,
  sizes 0..5.
- k > 0 required; k is measured on the built matrices per slot.

GROUPS: GTYPE "cyclic" (Z_q), "abelian" (Z_A0 x Z_{q/A0}) or "custom".
A custom group is a code string defining valid_q(q) -> bool and
custom_ops(q) -> dict(elements, ident, mul, inv). Any finite group you
can write down qualifies; the scaffold validates identity, inverses and
associativity on samples before building.
"""
import json
import os
import random

N_MIN = int(os.environ.get("CM_NMIN", "100"))
N_MAX = int(os.environ.get("CM_NMAX", "600"))
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("On dicyclic Dic14, order 56, shapes (1,3)x(1,3), replacing order-three complementary voltages by order-four voltages and selecting fillings with stabilizer-aware BP probes should improve worse-basis circuit fitness above the F57 parent's 4.8170.")
GROUP = ("New nonsplit dicyclic construction: x^28=1, y^2=x^14, y*x*y^-1=x^-1. Dic14 has no order-three elements and exactly one involution; it is genuinely different from the order-57 Frobenius parent. This order-56 bed is absent from BEDS. Strong lower-rate nonabelian beds motivate testing it without establishing a high-rate group advantage.")
SHAPES = ("Keep equal (1,3) sides: each component has 56 checks on 168 variables, n=560 and guaranteed k>=224, rate 0.400. Require verified complete k=224. Parent witnesses and failures spread across sector-1 Acol/Bcol blocks, with mostly 1.0-1.1x enrichment and collective flips across all 228 logicals; there is no demonstrated isolated Brow hotspot.")
FILLING = ("Twenty indexed pairs of automorphism-quotiented rotation-step geometries prescribe fresh independent trinomial searches. Each geometry contains an order-28 rotation; complementary terms have order four. Verified component lifts screen rank, four-cycles, kernels through four, weight-six multiplicity, expansion and triple boundaries. Complete-code finalists are compared with common-random-number BP probes, classifying settled residuals modulo actual stabilizers. These probes omit measurement histories and ASC ordering and are pL heuristics, not circuit measurements. Both row-weight sums remain 9+3=12, with 24192 six-round CNOTs. Target broad failures such as parent slot14 X=102/5000 despite d2s=8; require final k=224, CSS and no logical of weight <=4.")
FAMILY_NAME = "r040_Dic14_13x13_order4_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 56

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic14 requires q=56")
    # Normal forms x^a*y^b, 0<=a<28, 0<=b<2.
    # x^28=1, y^2=x^14, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(28)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 14 * (b // 2)) % 28, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + 14*b*d) % 28,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a + 14) % 28, 1) if b else ((-a) % 28, 0)

    def idx(g):
        a, b = red(g)
        return a + 28*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _f57_columns(rows, n):
    """Column syndromes of a verified lifted matrix."""
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _d56_briefs():
    """Relative step geometries, not a bank of saved component fillings."""
    from itertools import combinations
    from math import gcd

    units = [u for u in range(1, 28) if gcd(u, 28) == 1]

    def canonical(ds):
        return min(tuple(sorted(
            min(u*d % 28, 28 - u*d % 28) for d in ds))
            for u in units)

    # Avoid order-two/four rotation steps. At least one odd step
    # supplies an order-28 rotation, so the relative labels generate G.
    steps = [d for d in range(1, 14) if d != 7]
    representatives = {
        canonical(ds) for ds in combinations(steps, 3)
        if any(d % 2 for d in ds)
    }

    def priority(ds):
        orders = [28 // gcd(d, 28) for d in ds]
        return (-min(orders), -sum(orders), ds)

    book = sorted(representatives, key=priority)
    assert len(book) >= 20
    return book


def _d56_draw(rng, G, brief):
    """Normalize one common automorphism gauge; search relative phases."""
    phases = (0, rng.randrange(28), rng.randrange(28))
    return [[
        [G.e, (d, 0), (t, 1)]
        for d, t in zip(brief, phases)
    ]]


def _d56_distinct_differences(G, grid, side):
    """Action-aware prefilter; verified component matrices remain decisive."""
    seen = set()
    for cell in grid[0]:
        for a in cell:
            for b in cell:
                if G._key(a) == G._key(b):
                    continue
                if side == "A":
                    relative = G.mul(a, G.inv(b))
                else:
                    relative = G.mul(G.inv(a), b)
                difference = G._key(relative)
                if difference in seen:
                    return False
                seen.add(difference)
    return True


def _f57_component(G, grid, side):
    """Extract the actual A-left or B-right component using a unit partner."""
    unit = [[[G.e]]]
    if side == "A":
        rows = lifted_matrices(G, grid, unit)[0]
    else:
        rows = lifted_matrices(G, unit, grid)[1]
    width = 3 * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _f57_clean4(columns):
    """Exact exclusion of component kernels through weight four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = set()
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome == 0 or syndrome in singles or syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _f57_cycles(rows, columns):
    """Verified four-cycles and exact six-cycles when four-cycles are absent."""
    m = len(rows)
    adjacency = [0] * m
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            common = (row & rows[j]).bit_count()
            four += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    if four:
        return four, None

    triangles = 0
    for i in range(m):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6
                for d in (c.bit_count() for c in columns))
    six = triangles - stars
    assert six >= 0
    return 0, six


def _f57_six_boundary(columns, limit=None):
    """Exact weight-six kernel count and small-boundary triple exposure.

    Every column has odd weight three, excluding odd kernel weights.
    After the <=4 exclusion, distinct matching triples are disjoint.
    Every six-set has ten unordered triple partitions. No group-orbit
    shortcut is used: order-three stabilizers cannot corrupt this count.
    """
    buckets = {}
    collisions = 0
    exposure = 0
    boundary_one = 0
    size = len(columns)
    for a in range(size - 2):
        x = columns[a]
        for b in range(a + 1, size - 1):
            pair = x ^ columns[b]
            for c in range(b + 1, size):
                syndrome = pair ^ columns[c]
                previous = buckets.get(syndrome, 0)
                collisions += previous
                if limit is not None and collisions > 10 * limit:
                    return None
                buckets[syndrome] = previous + 1
                weight = syndrome.bit_count()
                if weight <= 5:
                    exposure += (7 - weight) ** 2
                    boundary_one += int(weight == 1)
    assert collisions % 10 == 0
    return collisions // 10, boundary_one, exposure


def _d56_select(rng, G, side, brief):
    """Keep alternatives for joint decoder-aware selection."""
    import numpy as np

    pool = []
    seen = set()
    accepted = 0
    for trial in range(1800):
        grid = _d56_draw(rng, G, brief)
        if not _d56_distinct_differences(G, grid, side):
            continue
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in grid[0])
        if signature in seen:
            continue
        seen.add(signature)
        rows = _f57_component(G, grid, side)
        if gf2_rank(rows) != G.q:
            continue
        columns = _f57_columns(rows, 3 * G.q)
        assert all(c.bit_count() == 3 for c in columns)
        four, cycles = _f57_cycles(rows, columns)
        if four or not _f57_clean4(columns):
            continue
        accepted += 1
        pool.append((cycles, signature, grid, rows, columns))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]
        if accepted >= 64:
            break

    if not pool:
        raise ValueError("no eligible dicyclic component for geometry")

    entries = []
    for cycles, signature, grid, rows, columns in pool:
        # Spectrum of the ACTUAL verified check-overlap graph.
        overlap = np.zeros((G.q, G.q), dtype=np.float64)
        for i, row in enumerate(rows):
            for j in range(i):
                shared = (row & rows[j]).bit_count()
                overlap[i, j] = overlap[j, i] = shared
        eigenvalues = np.linalg.eigvalsh(overlap)
        radius = round(float(max(
            abs(eigenvalues[0]), abs(eigenvalues[-2]))), 9)
        # Every check has overlap degree 9*(3-1)=18.
        if radius >= 18 - 1e-8:
            continue

        limit = entries[-1][0][0] if len(entries) >= 3 else None
        census = _f57_six_boundary(columns, limit)
        if census is None:
            continue
        words6, one, exposure = census
        key = (words6, one, radius, exposure, cycles, signature)
        entries.append((key, grid))
        entries.sort(key=lambda item: item[0])
        del entries[3:]
    if not entries:
        raise ValueError("no connected screened dicyclic component")
    return entries


def _d56_probe(checks, stabilizers, n, seed, shots=48):
    """BP data-noise surrogate, NOT the scored ASC memory experiment.

    Common random numbers compare proposals. Unsettled shots are
    penalized; settled residuals are tested modulo actual stabilizers.
    Correlated neighbor pairs approximate hooks without claiming an
    ASC ordering or including measurement histories.
    """
    import numpy as np

    neighbors_by_check = []
    for row in checks:
        neighbors = []
        while row:
            bit = row & -row
            neighbors.append(bit.bit_length() - 1)
            row ^= bit
        neighbors_by_check.append(neighbors)
    weight = len(neighbors_by_check[0])
    assert all(len(ns) == weight for ns in neighbors_by_check)
    incidence = np.asarray(neighbors_by_check, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(checks), len(flat)
    degrees = np.bincount(flat, minlength=n)
    assert np.all(degrees > 0)
    variable_edges = np.full(
        (n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, c in enumerate(flat):
        variable_edges[c, used[c]] = edge
        used[c] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.008, 0.014, 0.022))
    rates = levels[np.arange(shots) % 3, None] * degrees[None, :] / 3
    error = channel.random((shots, n)) < rates
    hook_rows = channel.integers(nr, size=shots)
    first = channel.integers(weight, size=shots)
    second = channel.integers(weight - 1, size=shots)
    second += second >= first
    hooks = channel.random(shots) < 0.30
    for shot in np.flatnonzero(hooks):
        r = hook_rows[shot]
        error[shot, incidence[r, first[shot]]] ^= True
        error[shot, incidence[r, second[shot]]] ^= True

    prior_rates = np.minimum(
        rates + 0.6 * degrees[None, :] / edges, 0.25)
    prior = np.log((1 - prior_rates) / prior_rates)
    target = (
        error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    failed = iteration_sum = 0
    for iteration in range(1, 25):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logarithms = np.log(np.clip(
            np.tanh(np.abs(shaped) / 2), 1e-12, 1 - 1e-7))
        products = np.minimum(np.exp(
            logarithms.sum(axis=2, keepdims=True) - logarithms),
            1 - 1e-7)
        magnitude = np.log1p(products) - np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)

        padded = np.zeros((count, edges + 1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior + padded[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        syndrome = (
            hard[:, flat].reshape(count, nr, weight).sum(axis=2) & 1
        ).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local] ^ hard[local]):
                word |= 1 << int(c)
            failed += int(nontrivial(word))
            iteration_sum += iteration

        keep = ~settled
        if not np.any(keep):
            return failed, 0, iteration_sum
        messages = np.clip(
            0.35 * messages
            + 0.65 * (posterior[:, flat] - outgoing), -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failed + unresolved, unresolved, iteration_sum + 25*unresolved


def _f57_small_logical(checks, stabilizers, n):
    """Exact existence test for a quantum logical of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    columns = _f57_columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is not None:
            if nontrivial(word ^ previous):
                return True
        else:
            seen[syndrome] = word
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(syndrome)
            if previous is not None:
                if nontrivial(word ^ previous):
                    return True
            else:
                seen[syndrome] = word
    return False


def NEW_CTX(rng, G, idx):
    """Prescribe new geometry pairs, then select complete verified codes."""
    assert G.q == 56
    assert A_SHAPE == B_SHAPE == (1, 3)
    supports = [[3, 3, 3]]
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 12
    assert 9 + 3 <= min(12, W_MAX)  # Both HX and HZ budgets.

    book = _d56_briefs()
    cards, seen = [], set()
    # Remove obvious side-exchanged brief pairs. Additional harness
    # batches receive new geometry pairs, not transformed saved codes.
    for separation in (7, 11, 3):
        for a in range(len(book)):
            b = (a + separation) % len(book)
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                cards.append((a, b))
    assert len(cards) >= 20
    pa, pb = cards[idx % len(cards)]
    ba, bb = book[pa], book[pb]
    seed = 560014003 + 1009*idx

    for attempt in range(3):
        aa = _d56_select(rng, G, "A", ba)
        bb_entries = _d56_select(rng, G, "B", bb)
        proposals = []
        for ia, (ka, A) in enumerate(aa):
            for ib, (kb, B) in enumerate(bb_entries):
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 560 or k != 224:
                    continue
                if not css_ok(hx, hz):
                    continue
                if max(row.bit_count() for row in hx + hz) != 12:
                    continue
                geometry = tuple(
                    value for t in range(5)
                    for value in (max(ka[t], kb[t]), ka[t] + kb[t])
                )
                proposals.append(
                    (geometry, ia, ib, A, B, hx, hz, n, k))
        proposals.sort(key=lambda item: item[:3])

        finalists = []
        for geometry, ia, ib, A, B, hx, hz, n, k in proposals[:4]:
            if _f57_small_logical(hx, hz, n):
                continue
            if _f57_small_logical(hz, hx, n):
                continue
            # Common random numbers across proposals, separately by basis.
            x = _d56_probe(hz, hx, n, seed, 48)
            z = _d56_probe(hx, hz, n, seed + 1, 48)
            quality = (
                max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                x[2] + z[2],
            ) + geometry + (ia, ib)
            finalists.append((quality, A, B, k, x, z))

        if finalists:
            finalists.sort(key=lambda item: item[0])
            quality, A, B, k, x, z = finalists[0]
            return {"idx": idx, "briefs": (ba, bb), "k": k,
                    "bp_surrogate": (x, z), "screen": quality,
                    "cells": {"A": A, "B": B}}
    raise ValueError("no screened dicyclic code in rate cell 0.400")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone give reliable r040f memory: parent fitness 3.2187, all five measured codes hit a failed-shot cap, with weight-4 sector-1 witnesses and broadly distributed logical flips.
# REFUTED: Balanced short-collision optimization on D58 (1,3)x(1,3) ensures rare circuit failures: fitness was 4.3650, every measured basis hit the failed-shot cap, and slot 0 had 105/3500 X failures despite d2s=8.
# REFUTED: Removing involutions and excluding component four-cycles and kernels through four alone guarantees rare memory failures: F57 fitness was 4.8170, all ten measured bases hit failed-shot caps, and slot14 had 102/5000 X failures despite d2s=8, with flips across all 228 logicals.
# EVOLVE-BLOCK-END


class _G(object):
    def __init__(self, ops, q):
        self.q = q
        self.elements = list(ops["elements"])
        self.e = ops["ident"]
        self.mul = ops["mul"]
        self.inv = ops["inv"]
        self._pos = {self._key(g): i for i, g in enumerate(self.elements)}

    @staticmethod
    def _key(g):
        return tuple(g) if isinstance(g, (list, tuple)) else g

    def idx(self, g):
        return self._pos[self._key(g)]

    def pick(self, rng):
        return self.elements[rng.randrange(self.q)]


# ---- verified construction helpers (2026-09-09) --------------------------
# lifted_matrices(G, cells_A, cells_B) builds EXACTLY the HX / HZ the
# evaluator builds (same lift conventions, same column order), as Python-int
# bit rows; gf2_rank(rows) is the rank behind k; lifted_k(G, A, B) is the k the table will report;
# css_ok(hx, hz) checks HX HZ^T = 0. Use these for any in-program screening -
# a hand-written lift that gets the left/right actions wrong screens a
# different (usually non-CSS) matrix pair than the one that is measured.
def lifted_matrices(G, cells_A, cells_B):
    """cells_A[i][j], cells_B[a][b]: lists of group elements (as CELL returns
    them) or element indices. Returns (hx_rows, hz_rows, n): lists of Python
    ints, bit c set = a 1 in column c; n = number of columns (qubits)."""
    q = G.q
    ra, ca = len(cells_A), len(cells_A[0])
    rb, cb = len(cells_B), len(cells_B[0])
    els = G.elements

    def gi(g):
        try:
            return G.idx(g)
        except (KeyError, TypeError):
            if isinstance(g, int) and 0 <= g < q:
                return g
            raise
    mul_idx = [[G.idx(G.mul(els[x], els[y])) for y in range(q)] for x in range(q)]
    inv_idx = [G.idx(G.inv(els[x])) for x in range(q)]
    A = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_A]
    B = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_B]
    n1 = ca * cb * q
    n = n1 + ra * rb * q
    hx = []
    for i in range(ra):
        for beta in range(cb):
            for r in range(q):
                row = 0
                for j in range(ca):                       # sector 1: A_{ij} left-lifted
                    base = (j * cb + beta) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[inv_idx[g]][r])   # h with g*h = r
                for alpha in range(rb):                   # sector 2: B_{alpha beta}^T right-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][inv_idx[g]])   # c = r*g^-1
                hx.append(row)
    hz = []
    for j in range(ca):
        for alpha in range(rb):
            for r in range(q):
                row = 0
                for beta in range(cb):                    # sector 1: B_{alpha beta} right-lifted
                    base = (j * cb + beta) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][g])           # h with h*g^-1 = r
                for i in range(ra):                       # sector 2: A_{ij}^T left-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[g][r])           # c = g*r
                hz.append(row)
    return hx, hz, n


def _popcount(x):
    return bin(x).count("1")



def gf2_rank(rows):
    """Rank over GF(2) of a list of bit rows."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return len(pivots)


def lifted_k(G, cells_A, cells_B):
    """k = n - rank(HX) - rank(HZ) of the code the evaluator will build."""
    hx, hz, n = lifted_matrices(G, cells_A, cells_B)
    return n - gf2_rank(hx) - gf2_rank(hz)


def css_ok(hx, hz):
    """True iff every X check commutes with every Z check (HX HZ^T = 0)."""
    return all(_popcount(x & z) % 2 == 0 for x in hx for z in hz)


def _grid_weights(sa, sb):
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    wx = max(sum(sa[i]) + sum(sb[r_][j] for r_ in range(rb))
             for i in range(ra) for j in range(cb))
    wz = max(sum(sb[r_]) + sum(sa[i][c] for i in range(ra))
             for r_ in range(rb) for c in range(ca))
    return max(wx, wz)


def _no_dead_lines(sa, sb):
    return all(
        all(sum(row) > 0 for row in g)
        and all(sum(r_[c] for r_ in g) > 0 for c in range(len(g[0])))
        for g in (sa, sb))


def roll_supports(rng):
    """Random per-cell term counts (1..CELL_MAX), repaired down to the
    weight cap with 1 as the floor - the roller never emits zero blocks
    (a slot's CELL may still return [] deliberately). A scaffold
    utility available to NEW_CTX; always returns legal grids."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    sa = [[rng.randint(1, CELL_MAX) for _ in range(ca)] for _ in range(ra)]
    sb = [[rng.randint(1, CELL_MAX) for _ in range(cb)] for _ in range(rb)]
    for _ in range(4000):
        if _grid_weights(sa, sb) <= W_MAX:
            return sa, sb
        g = sa if rng.random() < 0.5 else sb
        i = rng.randrange(len(g))
        c = rng.randrange(len(g[0]))
        if g[i][c] >= 2:            # 1 is the floor: the roller never
            g[i][c] -= 1            # emits zero blocks (evolution may)
    raise ValueError("roll_supports could not fit the weight cap "
                     "(all-ones row exceeds it: shape too wide)")


def _registry_ops(gtype, q, a0):
    if gtype == "cyclic":
        return dict(elements=list(range(q)), ident=0,
                    mul=lambda g, h: (g + h) % q,
                    inv=lambda g: (-g) % q)
    if gtype == "abelian":
        a = int(a0)
        if a < 1 or q % a:
            raise ValueError("A0 must divide q")
        b = q // a
        return dict(elements=[(x, y) for x in range(a) for y in range(b)],
                    ident=(0, 0),
                    mul=lambda g, h: ((g[0] + h[0]) % a, (g[1] + h[1]) % b),
                    inv=lambda g: ((-g[0]) % a, (-g[1]) % b))
    raise ValueError("GTYPE %r needs CUSTOM_GROUP" % gtype)


def _make_ops(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        if not ns["valid_q"](q):
            raise ValueError("valid_q rejects q=%d" % q)
        return ns["custom_ops"](q)
    return _registry_ops(GTYPE, q, A0)


def _valid_q(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        return bool(ns["valid_q"](q))
    if GTYPE == "abelian":
        return A0 is not None and int(A0) >= 1 and q % int(A0) == 0
    return q >= 2


def _validate_group(G):
    rng = random.Random(7)
    sample = [G.e] + [G.pick(rng) for _ in range(min(12, G.q))]
    for g in sample:
        assert G._key(G.mul(G.e, g)) == G._key(g), "identity fails"
        assert G._key(G.mul(g, G.inv(g))) == G._key(G.e), "inverse fails"
    for g in sample[:6]:
        for h in sample[:6]:
            for k in sample[:6]:
                assert G._key(G.mul(G.mul(g, h), k)) \
                    == G._key(G.mul(g, G.mul(h, k))), "associativity fails"
    assert len(G._pos) == len(G.elements) == G.q, "element list broken"


def _choose_q():
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if min(ra, ca, rb, cb) < 1:
        raise ValueError("protograph sides need >= 1 row and >= 1 column")
    f = ca * cb + ra * rb
    best = None
    for q in range(max(2, N_MIN // f), N_MAX // f + 1):
        if f * q < N_MIN or f * q >= N_MAX:
            continue
        if _valid_q(q):
            best = q
    if best is None:
        raise ValueError("no valid q puts n in [%d, %d)" % (N_MIN, N_MAX))
    return best


def _fill_slot(rng, G, idx):
    ctx = NEW_CTX(rng, G, idx)
    grids = {}
    rgrids = {}
    for side, (rows, cols) in (("A", A_SHAPE), ("B", B_SHAPE)):
        grid = []
        rgrid = []
        for i in range(rows):
            row = []
            rrow = []
            for j in range(cols):
                cell = CELL(rng, G, side, i, j, ctx)
                if not isinstance(cell, (list, tuple)):
                    raise ValueError("%s[%d][%d]: not a list" % (side, i, j))
                if len(cell) > CELL_MAX:
                    raise ValueError("%s[%d][%d]: >%d terms"
                                     % (side, i, j, CELL_MAX))
                keys = {G._key(g) for g in cell}
                if len(keys) != len(cell):
                    raise ValueError("%s[%d][%d]: duplicate element"
                                     % (side, i, j))
                pairs = sorted((G.idx(g), G._key(g)) for g in cell)
                row.append([i for i, _ in pairs])
                rrow.append([list(e) if isinstance(e, tuple) else e
                             for _, e in pairs])
            grid.append(row)
            rgrid.append(rrow)
        grids[side] = grid
        rgrids[side] = rgrid
    sa = [[len(c) for c in row] for row in grids["A"]]
    sb = [[len(c) for c in row] for row in grids["B"]]
    w = _grid_weights(sa, sb)
    if w > W_MAX:
        raise ValueError("row weight %d exceeds cap %d" % (w, W_MAX))
    if not _no_dead_lines(sa, sb):
        raise ValueError("all-zero protograph row or column")
    return grids["A"], grids["B"], rgrids["A"], rgrids["B"]


def get_candidates():
    """-> request dict for the worker: shared group data + 20 slots."""
    q = _choose_q()
    ops = _make_ops(q)
    G = _G(ops, q)
    _validate_group(G)
    mul_table = [[G.idx(G.mul(G.elements[a], G.elements[b]))
                  for b in range(q)] for a in range(q)]
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    slots = []
    for s in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + s)
        try:
            A, B, Ar, Br = _fill_slot(rng, G, s)
            slots.append({"idx": s, "A": A, "B": B, "Ar": Ar, "Br": Br})
        except Exception as e:
            slots.append({"idx": s, "error": str(e)[:120]})
    return {"family": FAMILY_NAME, "concept": CONCEPT,
            "gtype": GTYPE, "q": q, "mul": mul_table,
            "a_shape": [ra, ca], "b_shape": [rb, cb],
            "n": (ca * cb + ra * rb) * q,
            "slots": slots}


if __name__ == "__main__":
    r = get_candidates()
    ok = sum(1 for s in r["slots"] if "error" not in s)
    print(json.dumps({"family": r["family"], "q": r["q"], "n": r["n"],
                      "slots_ok": ok}))
