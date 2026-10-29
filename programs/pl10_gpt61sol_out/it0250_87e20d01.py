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
CONCEPT = ("For Q8 direct product C7, order 56, shapes (1,3)x(1,3), quaternion-direction trinomials screened for short dependencies, check-graph expansion and stabilizer-aware BP performance should improve worse-basis ASC memory above the square-affine parent's fitness 4.8678.")
GROUP = ("Jump to the direct product of the quaternion group and C7. Its centre has order fourteen and it has exactly one involution, distinguishing it from both the centre-one order-55 parent and the listed centre-two order-56 bed. This bed is absent from BEDS. Three quaternion directions supply noncommuting relative geometries; neither this algebra nor lower-rate nonabelian results establishes a circuit advantage.")
SHAPES = ("Retain equal (1,3) sides for cell 0.400: components have 56 checks and 168 variables, n=560, kb=224 and guaranteed rate 0.400. Require verified complete k=224. Parent weight-eight witnesses and failures span many sector-1 Acol/Bcol blocks; data and Acol/Brow check enrichment is mostly 1.0-1.1, so there is no demonstrated persistent hotspot to reinforce.")
FILLING = ("Twenty contracts prescribe multisets of relative-order triangle signatures (7,28,28), (14,28,28), (4,28,28) and (28,28,28). Fresh independent voltage searches use helper-derived singleton actions, an action-aware difference prefilter, verified four-cycle and kernel-through-four exclusion, exact weight-six multiplicity, sampled witnesses, check-overlap expansion and triple boundaries. Complete finalists receive 192-shot common-random-number BP probes with residuals classified modulo actual stabilizers; these omit ASC ordering and measurement histories and are not scored circuit experiments. Both row-weight sums remain 9+3=12; six-round CNOT count is 24192, slightly above the parent's 23760. Require complete verified CSS, k=224 and no logical through four. Target all ten parent bases reaching failure caps, including slot1 X=102/4500, with collective flips across all 220 logicals.")
FAMILY_NAME = "r040_Q8xC7_q56_13x13_ordertriangle_expansion_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 56

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 direct product C7 requires q=56")
    # (s,a,b,t) encodes (-1)^s i^a j^b z^t.
    # i^2=j^2=-1, ji=-ij, z^7=1 and z is central.
    els = [(s,a,b,t) for t in range(7) for b in range(2)
           for a in range(2) for s in range(2)]

    def red(g):
        return (int(g[0]) % 2, int(g[1]) % 2,
                int(g[2]) % 2, int(g[3]) % 7)

    def mul(g, h):
        s,a,b,t = red(g)
        u,c,d,v = red(h)
        return ((s+u+a*c+b*d+b*c) % 2,
                (a+c) % 2, (b+d) % 2, (t+v) % 7)

    def inv(g):
        s,a,b,t = red(g)
        return ((s+a+b+a*b) % 2, a, b, (-t) % 7)

    def idx(g):
        s,a,b,t = red(g)
        return s + 2*a + 4*b + 8*t

    return dict(elements=els, ident=(0,0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
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


def _f57_briefs():
    """Twenty relative-order architectures, not saved filling images."""
    from itertools import combinations_with_replacement

    # Sorted orders of the three undirected relative differences:
    # 0: (7,28,28), 1: (14,28,28),
    # 2: (4,28,28), 3: (28,28,28).
    # These signatures are invariant under group automorphisms and
    # cell translations. Arbitrary Tanner equivalence is checked later.
    book = list(combinations_with_replacement(range(4), 3))
    assert len(book) == 20
    return book


def _f57_draw(rng, G, brief):
    """Construct each prescribed signature on independent quaternion axes."""
    axes = ((1,0), (0,1), (1,1))
    cells = []
    for j, kind in enumerate(brief):
        a, b = axes[j]
        c, d = axes[(j+1) % 3]
        if kind < 2:
            # A central ratio has order seven or fourteen. The other
            # two ratios have order twenty-eight by phase exclusion.
            delta = rng.randrange(1, 7)
            phase = rng.choice([t for t in range(1, 7) if t != delta])
            u = (kind, 0, 0, delta)
            v = (rng.randrange(2), a, b, phase)
        elif kind == 2:
            # Exactly one order-four ratio; independent axes do not commute.
            u = (rng.randrange(2), a, b, 0)
            v = (rng.randrange(2), c, d, rng.randrange(1, 7))
        else:
            # Nonzero distinct C7 phases make all three ratios order 28.
            first, second = rng.sample(range(1, 7), 2)
            u = (rng.randrange(2), a, b, first)
            v = (rng.randrange(2), c, d, second)
        cell = [G.e, u, v]
        assert len({G._key(g) for g in cell}) == 3
        if kind >= 2:
            assert G.mul(u, v) != G.mul(v, u)
        cells.append(cell)
    return [cells]


def _f57_distinct_differences(G, grid, side):
    """Action-aware prefilter; actual verified component checks decide."""
    seen = set()
    for cell in grid[0]:
        for a in cell:
            for b in cell:
                if G._key(a) == G._key(b):
                    continue
                relative = (G.mul(a, G.inv(b)) if side == "A"
                            else G.mul(G.inv(a), b))
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


def _f55_sampled_protection(columns, seed):
    """Sample valid kernel words; estimates are upper bounds, not certificates."""
    srng = random.Random(seed)
    width = len(columns)
    permutation = list(range(width))
    minimum = width + 1
    short = set()
    for trial in range(32):
        srng.shuffle(permutation)
        pivots, basis = {}, []
        for c in permutation:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                basis.append(word)
        for i, word in enumerate(basis):
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 8:
                short.add(word)
            for other in basis[:i]:
                candidate = word ^ other
                weight = candidate.bit_count()
                minimum = min(minimum, weight)
                if weight <= 8:
                    short.add(candidate)
    return min(minimum, 10), len(short)


def _q56_expansion(rows):
    """Nontrivial check-overlap spectral radius of the verified component."""
    import numpy as np

    size = len(rows)
    overlap = np.zeros((size, size), dtype=np.float64)
    degrees = []
    for i, row in enumerate(rows):
        for j in range(i):
            shared = (row & rows[j]).bit_count()
            overlap[i, j] = overlap[j, i] = shared
        degrees.append(sum((row & other).bit_count()
                           for j, other in enumerate(rows) if j != i))
    assert len(set(degrees)) == 1 and degrees[0] == 18
    eigenvalues = np.linalg.eigvalsh(overlap)
    radius = max(abs(float(eigenvalues[0])),
                 abs(float(eigenvalues[-2])))
    # A second eigenvalue eighteen signals disconnected components.
    return round(radius, 9)


def _f57_select(rng, G, side, brief, batch, actions):
    """Keep independently searched alternatives for complete-code selection."""
    q = G.q
    pool, seen = [], set()
    accepted = 0

    def component(grid):
        # Exact linear composition of helper-derived singleton blocks.
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    for trial in range(2400):
        grid = _f57_draw(rng, G, brief)
        if not _f57_distinct_differences(G, grid, side):
            continue
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in grid[0])
        if signature in seen:
            continue
        seen.add(signature)
        rows = component(grid)
        if gf2_rank(rows) != q:
            continue
        columns = _f57_columns(rows, 3*q)
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
        raise ValueError("no eligible quaternion-product component for contract")

    entries = []
    probe_seed = 560030 + 29*batch + sum(
        (j+1)*kind for j, kind in enumerate(brief))
    for cycles, signature, grid, rows, columns in pool:
        radius = _q56_expansion(rows)
        if radius >= 18-1e-8:
            continue
        limit = entries[-1][0][0] if len(entries) >= 3 else None
        census = _f57_six_boundary(columns, limit)
        if census is None:
            continue
        words6, one, exposure = census
        distance, short8 = _f55_sampled_protection(columns, probe_seed)
        tail = ((cycles, exposure) if batch % 2
                else (exposure, cycles))
        key = (words6, 10-distance, short8, radius, one) + tail + (signature,)
        entries.append((key, grid, rows))
        entries.sort(key=lambda item: item[0])
        del entries[3:]
    if not entries:
        raise ValueError("quaternion-product component refinement failed")
    return entries


def _f55_probe(checks, stabilizers, n, seed, shots=192):
    """Common-random-number BP surrogate, not an ASC memory experiment.

    Includes degree-weighted data noise and sampled correlated pairs.
    Settled residuals are classified modulo the actual stabilizers.
    Measurement histories, relay decoding and ASC ordering are omitted.
    """
    import numpy as np

    neighbors = []
    for row in checks:
        current = []
        while row:
            bit = row & -row
            current.append(bit.bit_length()-1)
            row ^= bit
        neighbors.append(current)
    weight = len(neighbors[0])
    assert all(len(row) == weight for row in neighbors)
    incidence = np.asarray(neighbors, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(checks), len(flat)
    degrees = np.bincount(flat, minlength=n)
    assert np.all(degrees > 0)
    variable_edges = np.full(
        (n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.012, 0.022, 0.035))
    rates = levels[np.arange(shots) % 3, None] * degrees[None, :] / 3
    error = channel.random((shots, n)) < rates
    hook_rows = channel.integers(nr, size=shots)
    first = channel.integers(weight, size=shots)
    second = channel.integers(weight-1, size=shots)
    second += second >= first
    hooks = channel.random(shots) < 0.25
    for shot in np.flatnonzero(hooks):
        r = hook_rows[shot]
        error[shot, incidence[r, first[shot]]] ^= True
        error[shot, incidence[r, second[shot]]] ^= True

    prior_rates = np.minimum(
        rates + 0.5*degrees[None, :]/edges, 0.30)
    prior = np.log((1-prior_rates)/prior_rates)
    target = (
        error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length()-1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    failed = iteration_sum = 0
    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logarithms = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
        products = np.minimum(np.exp(
            logarithms.sum(axis=2, keepdims=True)-logarithms), 1-1e-7)
        magnitude = np.log1p(products)-np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)

        padded = np.zeros((count, edges+1), dtype=np.float64)
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
            0.35*messages + 0.65*(posterior[:, flat]-outgoing),
            -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failed+unresolved, unresolved, iteration_sum+33*unresolved


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
    """Optimize a relative-order contract, then compare verified complete codes."""
    q = G.q
    assert q == 56
    assert A_SHAPE == B_SHAPE == (1, 3)
    supports = [[3, 3, 3]]
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 12
    assert 9+3 <= min(12, W_MAX)  # HX and HZ, separately.

    book = _f57_briefs()
    batch, design = divmod(idx, 20)
    ba = book[design]
    bb = book[(design+7+batch) % 20]

    # Cache verified singleton actions only. No handwritten lift is used.
    mask = (1 << q)-1
    actions = {"A": {}, "B": {}}
    zero = [[[]]]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], zero)
        _, hz, _ = lifted_matrices(G, zero, [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    seed = 560013003 + 1009*idx
    for attempt in range(3):
        aa = _f57_select(rng, G, "A", ba, batch, actions["A"])
        bb_entries = _f57_select(
            rng, G, "B", bb, batch, actions["B"])
        proposals = []
        for ia, (ka, A, ar) in enumerate(aa):
            for ib, (kb, B, br) in enumerate(bb_entries):
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 560 or k != 224:
                    continue
                if not css_ok(hx, hz):
                    continue
                if max(row.bit_count() for row in hx+hz) != 12:
                    continue

                # Confirm exact equality with the final submitted lift.
                actual_a = [
                    sum(((row >> (3*j*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]
                ]
                actual_b = [
                    row & ((1 << (3*q))-1) for row in hz[:q]
                ]
                assert actual_a == ar and actual_b == br
                geometry = tuple(
                    value for t in range(7)
                    for value in (max(ka[t], kb[t]), ka[t]+kb[t])
                )
                proposals.append(
                    (geometry, ia, ib, A, B, hx, hz, n, k))
        proposals.sort(key=lambda item: item[:3])

        finalists = []
        for geometry, ia, ib, A, B, hx, hz, n, k in proposals[:5]:
            if _f57_small_logical(hx, hz, n):
                continue
            if _f57_small_logical(hz, hx, n):
                continue
            # Identical random streams across proposals in each basis.
            x = _f55_probe(hz, hx, n, seed)
            z = _f55_probe(hx, hz, n, seed+1)
            quality = (
                max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                x[2]+z[2],
            ) + geometry + (ia, ib)
            assert 6*sum(row.bit_count() for row in hx+hz) == 24192
            finalists.append((quality, A, B, k, x, z))

        if finalists:
            finalists.sort(key=lambda item: item[0])
            quality, A, B, k, x, z = finalists[0]
            return {"idx": idx, "briefs": (ba, bb), "k": k,
                    "bp_surrogate": (x, z), "screen": quality,
                    "cells": {"A": A, "B": B}}
    raise ValueError("no quaternion-product filling passed rate-0.400 screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone give reliable r040f memory: parent fitness 3.2187, all five measured codes hit a failed-shot cap, with weight-4 sector-1 witnesses and broadly distributed logical flips.
# REFUTED: Balanced short-collision optimization on D58 (1,3)x(1,3) ensures rare circuit failures: fitness was 4.3650, every measured basis hit the failed-shot cap, and slot 0 had 105/3500 X failures despite d2s=8.
# REFUTED: Removing involutions and independently optimizing four-cycle-free components through weight-six multiplicity ensures rare memory failures: F57 fitness was 4.8170, all ten measured bases hit failed-shot caps, and slot14 had 102/5000 X failures with flips across all 228 logical qubits.
# REFUTED: Square-affine triangles with short-kernel screening and complete-code BP selection ensure rare ASC failures: fitness was 4.8678, all ten measured bases reached failure caps, and slot1 X failed 102/4500 shots with collective flips across all 220 logicals and mostly 1.0-1.1x block enrichment.
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
