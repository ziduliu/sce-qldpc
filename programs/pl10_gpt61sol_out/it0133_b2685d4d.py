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
CONCEPT = ("For cyclic Z13 (metacyclic Z13:Z1, s=1), order 13, shapes (2,5)x(3,8), reinforcing A while preserving degree-three B columns should raise worse-basis circuit fitness above 4.9989 when short-kernel multiplicity and poor component expansion are screened together.")
GROUP = ("Frozen L1 group: retain cyclic Z13 and its exact construction. This bed already reaches s=5.202, so filling is the controlled variable. The size factor 46 admits q<=13 and gives n=598. ord_13(2)=12 motivates distinguishing unavoidable augmentation-rank deficits from additional verified rank loss; it is not a circuit-performance guarantee.")
SHAPES = ("Frozen (2,5)x(3,8), targeting cell 0.350: kb=195, guaranteed rate 15/46=0.326087, with verified 195<=k<=201. A has 65 variables/26 checks and B has 104 variables/39 checks. Parent failures span sector-one Acol/Bcol and Zchk:Acol/Brow blocks with only 1.1-1.2x enrichment; no isolated block is a demonstrated cause. Singly reinforced B slots 1/5/9 are X-limited despite weight-eight witnesses in both bases.")
FILLING = ("Twenty contracts cross five A architectures containing 16-18 terms with four degree-preserving B redistributions. A columns have degree three/four and rows load eight/nine; B columns have degree three and all rows load eight. Both CSS weights stay <=12, with six-round CNOT budgets 26832/27690/28548, none matching a parent budget. Optimize verified singleton-action compositions; screen short kernels, four-cycles, exact A weight-eight multiplicity, anchored triple boundaries and common-random-number longer-witness probes. Final helper-built components, CSS, ranks and gate counts are verified. Targets are the parent's 1.1-1.4% X failure fractions and collective logical flips, not distance alone; these structural screens remain hypotheses about measured pL.")
FAMILY_NAME = "r035_m13r1s1_25x38_Areinforced_boundary8_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Same cyclic metacyclic construction; factor 46 requires q<=13.
    # Z_13 x| Z_1: y x y^-1 = x^1.
    if q != 13:
        raise ValueError("this bed requires order 13")
    m, r, s = 13, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty reinforcement/redistribution contracts on the frozen bed."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    design, batch = idx % 20, idx // 20

    # Explicit upper/lower counts: one, two or three degree-four columns.
    # Different concentration patterns are designs, not permutations.
    a_profiles = (
        ((2, 2, 2, 1, 1), (2, 1, 1, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 2, 2, 2, 2)),
        ((3, 1, 2, 1, 1), (1, 3, 1, 2, 2)),
        ((3, 3, 1, 1, 1), (1, 1, 3, 2, 2)),
    )
    sa = [list(row) for row in a_profiles[design // 4]]
    sb = [[1] * cb for _ in range(rb)]

    # Transfer a singleton to another row of the SAME column.
    # Every B column remains degree three; every row remains load eight.
    transfers = (
        (),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 2), (2, 0)),
        ((0, 1), (1, 2), (2, 0),
         (0, 2), (2, 1), (1, 0)),
    )[design % 4]
    for j, (source, destination) in enumerate(transfers):
        sb[source][j] = 0
        sb[destination][j] = 2

    counts = {"A": sa, "B": sb}
    assert all(sum(row[j] for row in sa) in (3, 4)
               for j in range(ca))
    assert all(sum(row[j] for row in sb) == 3
               for j in range(cb))
    assert all(sum(row) == 8 for row in sb)
    # HX <=9+3; HZ <=8+4. No support is repaired after submission.
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    mask = (1 << q) - 1
    unit = [[[G.e]]]

    def transpose(rows, width):
        cols = [0] * width
        sector_mask = (1 << width) - 1
        for r, row in enumerate(rows):
            bits = row & sector_mask
            while bits:
                bit = bits & -bits
                cols[bit.bit_length() - 1] ^= 1 << r
                bits ^= bit
        return cols

    # Cache only actions returned by the verified helper. This is linear
    # composition of measured blocks, not a guessed left/right lift.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        ar = lifted_matrices(G, [[[g]]], unit)[0]
        br = lifted_matrices(G, unit, [[[g]]])[1]
        actions["A"][G._key(g)] = tuple(transpose(ar, q))
        actions["B"][G._key(g)] = tuple(transpose(br, q))

    def draw(side):
        return [[rng.sample(G.elements, t) for t in row]
                for row in counts[side]]

    def component_columns(side, grid):
        width = len(grid[0]) * q
        cols = [0] * width
        for i, row in enumerate(grid):
            shift = i * q
            for j, cell in enumerate(row):
                base = j * q
                for g in cell:
                    term = actions[side][G._key(g)]
                    for t, syndrome in enumerate(term):
                        cols[base + t] ^= syndrome << shift
        expected = [sum(row[j] for row in counts[side])
                    for j in range(len(counts[side][0]))]
        assert all(c.bit_count() == expected[j // q]
                   for j, c in enumerate(cols))
        return cols

    def cycle_data(cols):
        """Exact four-cycle count for either component and mixed degrees."""
        pairs = {}
        adj = [0] * max(c.bit_length() for c in cols)
        conflicts = 0
        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a, u in enumerate(nodes):
                for v in nodes[a + 1:]:
                    key = (1 << u) | (1 << v)
                    old = pairs.get(key, 0)
                    conflicts += old
                    pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return conflicts, adj

    def kernel_metric(cols, adj):
        """Short syndrome ambiguities, including odd-weight B witnesses.

        Counts are collision proxies if shorter words exist. They are
        not quantum distances and do not distinguish stabilizers.
        """
        singles = {}
        for syn in cols:
            singles[syn] = singles.get(syn, 0) + 1
        n2 = sum(v * (v - 1) // 2 for v in singles.values())
        pair_hist = {}
        n3 = n4 = 0
        nc = len(cols)
        for a in range(nc - 1):
            for b in range(a + 1, nc):
                syn = cols[a] ^ cols[b]
                n3 += singles.get(syn, 0)
                old = pair_hist.get(syn, 0)
                n4 += old
                pair_hist[syn] = old + 1

        triangles = 0
        for u in range(len(adj)):
            bits = adj[u] & ~((1 << (u + 1)) - 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() - 1
                triangles += (
                    adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                ).bit_count()
                bits ^= bit
        stars = sum(c.bit_count() * (c.bit_count() - 1)
                    * (c.bit_count() - 2) // 6 for c in cols)
        # Exact with no C4; otherwise only a nonnegative cycle proxy.
        # The new search also retains unavoidable-C4 local minima.
        n6cycles = max(0, triangles - stars)
        if n2 or n3 or n4:
            return (n2, n3, n4, 10**9, 10**9, n6cycles)

        triple_hist = {}
        n5 = n6 = 0
        for a in range(nc - 2):
            for b in range(a + 1, nc - 1):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc):
                    syn = ab ^ cols[c]
                    n5 += pair_hist.get(syn, 0)
                    old = triple_hist.get(syn, 0)
                    n6 += old
                    triple_hist[syn] = old + 1
        return (0, 0, 0, n5, n6, n6cycles)

    def rows_from_columns(cols):
        """Transpose the columns extracted from the verified lift."""
        rows = [0] * max(c.bit_length() for c in cols)
        for j, col in enumerate(cols):
            while col:
                bit = col & -col
                rows[bit.bit_length() - 1] |= 1 << j
                col ^= bit
        return rows

    def kernel_probe(cols, trials):
        """Sample actual kernel words, then reduce their basis combinations.

        Common random orders compare finalists more consistently.
        Missing a short witness never certifies a component distance.
        """
        nc = len(cols)
        probe_rng = random.Random(130001 + 271 * idx + nc)
        minimum = nc + 1
        small = set()
        order = list(range(nc))

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 10:
                small.add(word)

        for _ in range(trials):
            probe_rng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p in pivots:
                        s, v = pivots[p]
                        syndrome ^= s
                        word ^= v
                    else:
                        pivots[p] = (syndrome, word)
                        break
                if syndrome == 0:
                    record(word)
                    basis.append(word)

            for sweep in range(2):
                probe_rng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for j in range(i):
                        candidate = word ^ basis[j]
                        record(candidate)
                        w = candidate.bit_count()
                        if candidate and w < weight:
                            word, weight = candidate, w
                    basis[i] = word
            if minimum <= 6:
                break

        exposure = (
            sum(word.bit_count() == minimum for word in small),
            sum(word.bit_count() <= 8 for word in small),
            len(small),
        )
        return minimum, exposure

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def optimize(side):
        import math
        from itertools import combinations

        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        number_rows = len(counts[side])
        augmentation = [
            sum((t % 2) << j for j, t in enumerate(row))
            for row in counts[side]
        ]
        allowed_deficit = number_rows - gf2_rank(augmentation)

        def rank_penalty(columns):
            deficit = number_rows * q - gf2_rank(
                rows_from_columns(columns))
            return max(0, deficit - allowed_deficit)

        def short_quality(columns):
            singles = {}
            for s in columns:
                singles[s] = singles.get(s, 0) + 1
            n2 = sum(v * (v - 1) // 2 for v in singles.values())
            pairs = {}
            n3 = n4 = 0
            for i, x in enumerate(columns):
                for j in range(i):
                    s = x ^ columns[j]
                    n3 += singles.get(s, 0)
                    old = pairs.get(s, 0)
                    n4 += old
                    pairs[s] = old + 1
            return n2, n3, n4

        pool, tested = [], set()

        def retain(grid, columns, cost, adj):
            sig = signature(grid)
            if sig in tested:
                return
            tested.add(sig)
            quick = ((rank_penalty(columns),)
                     + short_quality(columns) + (cost,))
            pool.append((quick, sig, grid, columns, adj))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        # Cached verified actions make multiple independent walks cheap.
        # Keep nonzero-C4 minima too: never replace a difficult contract
        # by an unscreened fallback merely because C4=0 was not found.
        for restart in range(3):
            current = draw(side)
            columns = component_columns(side, current)
            cost, adj = cycle_data(columns)
            local_min = cost
            retain(current, columns, cost, adj)
            steps = 900 + 150 * (batch % 3)

            for step in range(steps):
                i, j = rng.choice(sites)
                trial = [list(row) for row in current]
                trial[i][j] = rng.sample(G.elements, counts[side][i][j])
                tc = component_columns(side, trial)
                trial_cost, ta = cycle_data(tc)
                temperature = 1.2 * (0.993 ** step) + 0.025
                delta = (trial_cost - cost) / q
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current, columns = trial, tc
                    cost, adj = trial_cost, ta

                improved = cost < local_min
                local_min = min(local_min, cost)
                if improved or (
                        step % 12 == 0 and cost <= local_min + q):
                    retain(current, columns, cost, adj)
            retain(current, columns, cost, adj)

        finalists = []
        for quick, sig, grid, columns, adj in pool:
            metric = (quick[0],) + kernel_metric(columns, adj)
            # Short kernels take precedence over cycle heuristics.
            pre = metric[:-1] + (quick[-1], metric[-1])
            finalists.append((pre, sig, grid, columns, metric, quick[-1]))
        finalists.sort(key=lambda item: item[:2])

        def anchored_boundaries(columns):
            """Worst-block and total small triple-boundary exposure.

            Each anchor represents one lift orbit. These are expansion/
            trapping-set surrogates, not estimated circuit failure rates.
            """
            block_costs = []
            nc = len(columns)
            for anchor in range(0, nc, q):
                total = 0
                for b in range(nc - 1):
                    if b == anchor:
                        continue
                    ab = columns[anchor] ^ columns[b]
                    for c in range(b + 1, nc):
                        if c == anchor:
                            continue
                        w = (ab ^ columns[c]).bit_count()
                        if w <= 4:
                            total += (5 - w) ** 2
                block_costs.append(total)
            return max(block_costs), sum(block_costs)

        def exact_weight8(columns):
            """Exact component weight-eight count after weights <=6 vanish.

            Equal four-support syndromes cannot overlap in this case.
            Each eight-set has 35 unordered four/four partitions.
            This counts component words, not quantum logical operators.
            """
            histogram = {}
            collisions = 0
            for a, b, c, d in combinations(columns, 4):
                syndrome = a ^ b ^ c ^ d
                old = histogram.get(syndrome, 0)
                collisions += old
                histogram[syndrome] = old + 1
            assert collisions % 35 == 0
            return collisions // 35

        best = None
        for pre, sig, grid, columns, metric, c4 in finalists[:4]:
            short_free = not any(metric[1:6])
            n8 = (exact_weight8(columns)
                  if side == "A" and short_free else
                  (0 if side == "B" else 10**9))
            traps = anchored_boundaries(columns)
            distance_upper, exposure = kernel_probe(
                columns, 128 + 32 * (batch % 3))
            quality = (
                metric[:-1] + (c4, n8) + traps
                + (-min(distance_upper, 10), exposure, metric[-1], sig)
            )
            if best is None or quality < best[0]:
                best = quality, grid
        if best is None:
            raise ValueError("no screened component in filling contract")
        return best[1]

    # Retry labels only. The evaluator's full ranks determine the rate.
    for attempt in range(3):
        cells = {"A": optimize("A"), "B": optimize("B")}
        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (195 <= k <= 201):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified CSS commutation failed")

        # Independently verify that every cached action composition
        # screened precisely the final helper-built component.
        for side, grid in cells.items():
            rows = (lifted_matrices(G, grid, unit)[0]
                    if side == "A" else
                    lifted_matrices(G, unit, grid)[1])
            width = len(grid[0]) * q
            assert transpose(rows, width) == component_columns(side, grid)

        assert n == 598
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        total_a = sum(map(sum, counts["A"]))
        expected_cnots = 6 * q * (11 * total_a + 168)
        assert expected_cnots in (26832, 27690, 28548)
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        return {"idx": idx, "cells": cells, "k": k}
    raise ValueError("no verified filling in the target rate interval")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases imply balanced memory reliability: current parent slot 20 had 100/8250 Z failures versus 100/50250 X failures, with only diffuse block enrichment.
# REFUTED: A single added B doubleton reliably balances the Z13 bed: slots 1/5/9 retained d2s=8 but had about 1.1-1.4% X failures versus 0.3-0.5% Z failures, and s=4.851-4.962 at depth17; unreinforced slot0 scored 5.202 at depth16.
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
