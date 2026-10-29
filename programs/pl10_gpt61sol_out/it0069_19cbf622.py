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
CONCEPT = ("For cyclic Z17 (metacyclic Z17:Z1, s=1), order 17, shapes (2,5)x(2,6), disjoint A/B intra-cell difference palettes and verified weight-eight multiplicity screening should raise worse-basis circuit fitness above 4.7738 without increasing gate count.")
GROUP = ("Frozen L1 Z17 bed: its measured filling improvement from 4.0702 to 4.7738 motivates another element-model experiment. These shapes admit at most q=17 under n<600, giving n=578; neither the group nor its encoding changes.")
SHAPES = ("Frozen (2,5)x(2,6) targets cell 0.350: guaranteed k>=204 and rate>=12/34=0.352941. Components have 34 checks and unequal lengths 85/102. Parent failures affect all 204 logical qubits; Acol/Bcol enrichments are only 1.1-1.2x, including Acol0/Bcol0, Acol4/Bcol2 and Acol2/Bcol1. This supports global graph refinement, not a claimed single-block diagnosis.")
FILLING = ("Twenty support/palette architectures retain degree-three columns, A row totals 7/8, B row totals 9/9, HX<=11, HZ=12 and 25092 six-round CNOTs. Disjoint undirected intra-cell differences exclude mixed-sector four-cycles when both component graphs are four-cycle-free. Local searches use verified component projections; finalists receive a translation-orbit weight-eight census only after excluding component kernels through weight six. Joint verified-matrix selection screens full four-cycles and actual k, preferring k=204. The hypothesis targets parent Z failures of 100-105 in 5250-8250 shots versus about 100 in 23000-50250 X shots; these structural screens do not replace measured circuit pL.")
FAMILY_NAME = "r035_m17r1s1_25x26_disjoint_palette_w8_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # metacyclic Z_17 x| Z_1: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 17, 1, 1
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
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Joint difference separation, followed by verified component searches."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    from itertools import combinations

    # Upper-row counts; lower-row counts are their complements to three.
    # The second entry is the minimum number of difference classes needed
    # by either row if it is to have no repeated check-pair intersections.
    a_profiles = (
        ((2, 2, 1, 1, 1), 3),
        ((0, 2, 2, 2, 1), 4),
        ((3, 0, 2, 1, 1), 5),
    )
    b_profiles = (
        ((2, 2, 2, 1, 1, 1), 3),
        ((0, 2, 2, 2, 2, 1), 4),
        ((3, 0, 2, 2, 1, 1), 5),
    )
    classes = tuple(range(1, (q + 1) // 2))

    def difference_class(d):
        d %= q
        return min(d, q - d)

    pattern_cache = {}

    def patterns(palette, weight):
        """Normalized supports, not an independently implemented lift."""
        key = tuple(palette), weight
        if key in pattern_cache:
            return pattern_cache[key]
        allowed = set(palette)
        if weight == 0:
            result = [()]
        elif weight == 1:
            result = [(0,)]
        elif weight == 2:
            result = [(0, d) for d in range(1, q)
                      if difference_class(d) in allowed]
        else:
            result = []
            for a, b in combinations(range(1, q), 2):
                diffs = {difference_class(a), difference_class(b),
                         difference_class(b - a)}
                # A repeated difference inside a trinomial already
                # produces a component four-cycle.
                if len(diffs) == 3 and diffs <= allowed:
                    result.append((0, a, b))
        pattern_cache[key] = result
        return result

    def canonical_palette(palette):
        # Remove multiplicative-automorphism copies of a partition.
        return min(tuple(sorted({
            difference_class(u * d) for d in palette
        })) for u in range(1, q))

    catalogs = []
    for ai, (ap, amin) in enumerate(a_profiles):
        for bi, (bp, bmin) in enumerate(b_profiles):
            if amin + bmin > len(classes):
                continue
            choices = []
            for size in range(amin, len(classes) - bmin + 1):
                for palette_a in combinations(classes, size):
                    if palette_a != canonical_palette(palette_a):
                        continue
                    palette_b = tuple(d for d in classes
                                      if d not in palette_a)
                    if 3 in ap and not patterns(palette_a, 3):
                        continue
                    if (3 in ap or 0 in ap) and not patterns(palette_a, 3):
                        continue
                    if (3 in bp or 0 in bp) and not patterns(palette_b, 3):
                        continue
                    choices.append((ai, bi, palette_a, palette_b))
            choices.sort(key=lambda x: (abs(len(x[2]) - 4), x[2]))
            if choices:
                catalogs.append(choices)

    # Interleave dense and concentrated support architectures rather than
    # filling twenty slots with relabelings of one support/voltage model.
    briefs = []
    level = 0
    while len(briefs) < 20:
        added = False
        for catalog in catalogs:
            if level < len(catalog):
                briefs.append(catalog[level])
                added = True
                if len(briefs) == 20:
                    break
        if not added:
            raise ValueError("fewer than twenty palette architectures")
        level += 1

    design = idx % 20
    mode = (idx // 20) % 3
    ai, bi, palette_a, palette_b = briefs[design]
    pa, pb = a_profiles[ai][0], b_profiles[bi][0]
    palettes = {"A": palette_a, "B": palette_b}
    assert not set(palette_a) & set(palette_b)
    counts = {
        "A": [list(pa), [3 - x for x in pa]],
        "B": [list(pb), [3 - x for x in pb]],
    }
    assert _grid_weights(counts["A"], counts["B"]) <= W_MAX
    assert _no_dead_lines(counts["A"], counts["B"])

    dummy_a = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    dummy_b = [[[G.e] for _ in range(cb)] for _ in range(rb)]
    mask = (1 << q) - 1

    libraries = {
        side: {w: patterns(palettes[side], w) for w in range(4)}
        for side in ("A", "B")
    }
    for side in ("A", "B"):
        for row in counts[side]:
            for weight in row:
                assert libraries[side][weight]

    def draw_cell(side, weight):
        offsets = rng.choice(libraries[side][weight])
        anchor = G.pick(rng)
        return [G.mul(anchor, (d, 0)) for d in offsets]

    def draw(side):
        return [[draw_cell(side, t) for t in row]
                for row in counts[side]]

    def component_columns(side, grid):
        # Project actual helper-built matrices; do not substitute a lift.
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, dummy_b)
            cols = [0] * (ca * q)
            for i in range(ra):
                for r in range(q):
                    row = hx[(i * cb) * q + r]  # B-column zero
                    check_bit = 1 << (i * q + r)
                    for j in range(ca):
                        bits = (row >> (j * cb * q)) & mask
                        while bits:
                            bit = bits & -bits
                            cols[j * q + bit.bit_length() - 1] ^= check_bit
                            bits ^= bit
        else:
            _, hz, _ = lifted_matrices(G, dummy_a, grid)
            cols = [0] * (cb * q)
            sector_mask = (1 << (cb * q)) - 1
            for a in range(rb):
                for r in range(q):
                    bits = hz[a * q + r] & sector_mask  # A-column zero
                    check_bit = 1 << (a * q + r)
                    while bits:
                        bit = bits & -bits
                        cols[bit.bit_length() - 1] ^= check_bit
                        bits ^= bit
        assert all(c.bit_count() == 3 for c in cols)
        return cols

    def cycle_data(cols):
        """Exact component four-cycle count and check adjacency."""
        pairs = {}
        adj = [0] * (2 * q)
        conflicts = 0
        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a in range(3):
                u = nodes[a]
                for b in range(a + 1, 3):
                    v = nodes[b]
                    key = (1 << u) | (1 << v)
                    old = pairs.get(key, 0)
                    conflicts += old
                    pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return conflicts, adj

    def kernel_metric(cols, adj):
        """Short kernels and triple boundaries on the verified component."""
        pair_hist = {}
        collisions = 0
        nc = len(cols)
        for a in range(nc - 1):
            for b in range(a + 1, nc):
                syn = cols[a] ^ cols[b]
                old = pair_hist.get(syn, 0)
                collisions += old
                pair_hist[syn] = old + 1
        # This is used only on four-cycle-free degree-three components,
        # whose columns are distinct and nonzero.
        n4 = collisions // 3

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
        n6cycles = triangles - nc
        if n4:
            return (n4, 10**9, n6cycles, 10**9)

        triple_hist = {}
        collisions = boundary = 0
        for a in range(nc - 2):
            for b in range(a + 1, nc - 1):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc):
                    syn = ab ^ cols[c]
                    old = triple_hist.get(syn, 0)
                    collisions += old
                    triple_hist[syn] = old + 1
                    weight = syn.bit_count()
                    if weight < 7:
                        boundary += (7 - weight) ** 2
        return (0, collisions // 10, n6cycles, boundary)

    def weight_eight_count(cols):
        """Exact component weight-eight census when kernels <=6 are absent.

        Enumerate one representative per translation orbit of quadruple
        supports. Canonicalize their VERIFIED syndromes under simultaneous
        cyclic rotation of the two check blocks; no new lift is built.

        Quartet support and nonzero quartet-syndrome orbits have size 17:
        four terms cannot be translation-invariant, and a syndrome of
        weight <=12 cannot contain a complete 17-check orbit. Thus each
        canonical syndrome collision contributes 17 ordinary collisions.
        With no kernels <=6, colliding quadruples are disjoint, and each
        weight-eight word has 35 unordered quartet partitions.
        """
        nc = len(cols)
        seen = {}
        collisions = 0

        def canonical_syndrome(s):
            low, high = s & mask, s >> q
            best = s
            for shift in range(1, q):
                lo = ((low << shift) & mask) | (low >> (q - shift))
                hi = ((high << shift) & mask) | (high >> (q - shift))
                value = lo | (hi << q)
                if value < best:
                    best = value
            return best

        for a in range(0, nc, q):
            first_block = a // q
            for b in range(a + 1, nc - 2):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, nc):
                        support = (a, b, c, d)
                        representative = True
                        # Any competing normalized representative must
                        # shift one term in the first occupied block to 0.
                        for anchor in support[1:]:
                            if anchor // q != first_block:
                                break
                            shift = anchor % q
                            moved = tuple(sorted(
                                (v // q) * q + (v % q - shift) % q
                                for v in support))
                            if moved < support:
                                representative = False
                                break
                        if not representative:
                            continue
                        syndrome = abc ^ cols[d]
                        if not syndrome:
                            raise ValueError("weight-eight census saw a quartet kernel")
                        key = canonical_syndrome(syndrome)
                        old = seen.get(key, 0)
                        collisions += old
                        seen[key] = old + 1

        total = q * collisions
        assert total % 35 == 0
        return total // 35

    def full_four_cycles(rows):
        return sum(t * (t - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for t in [(row & other).bit_count()])

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def optimize(side):
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        current = draw(side)
        cols = component_columns(side, current)
        cost, adj = cycle_data(cols)
        fallback, fallback_cost = current, cost
        pool = []
        tested = set()

        for step in range(2400):
            if cost == 0:
                sig = signature(current)
                if sig not in tested:
                    tested.add(sig)
                    metric = kernel_metric(cols, adj)
                    # Keep several genuinely different kernels for the
                    # more expensive weight-eight and joint-rank screen.
                    quick = (metric[0], metric[1], metric[3], metric[2])
                    pool.append((quick, sig, metric, current, cols))
                    pool.sort(key=lambda item: (item[0], item[1]))
                    del pool[3:]
                    if (len(tested) >= 32
                            and pool[0][2][0] == pool[0][2][1] == 0):
                        break

            i, j = sites[rng.randrange(len(sites))]
            trial = [list(row) for row in current]
            trial[i][j] = draw_cell(side, counts[side][i][j])
            trial_cols = component_columns(side, trial)
            trial_cost, trial_adj = cycle_data(trial_cols)
            if trial_cost < fallback_cost:
                fallback, fallback_cost = trial, trial_cost

            if trial_cost <= cost or rng.random() < 0.012:
                current, cols = trial, trial_cols
                cost, adj = trial_cost, trial_adj
            if step % 240 == 239 and cost:
                current = draw(side)
                cols = component_columns(side, current)
                cost, adj = cycle_data(cols)
                if cost < fallback_cost:
                    fallback, fallback_cost = current, cost

        if not pool:
            # The fallback is legal and still respects difference
            # separation, but is not claimed to pass the graph screen.
            fc = component_columns(side, fallback)
            metric = (10**9, 10**9, 10**9, 10**9)
            return [(metric, 10**12, fallback)]

        result = []
        # Bound the expensive census to two distinct finalists per side.
        for _, _, metric, grid, cc in pool[:2]:
            eight = (weight_eight_count(cc)
                     if metric[0] == metric[1] == 0 else 10**12)
            result.append((metric, eight, grid))
        return result

    apool, bpool = optimize("A"), optimize("B")
    best = None
    for am, ae, A in apool:
        for bm, be, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # Stay in the requested rate cell even if a rare rank
            # deficit survives. Exact k=204 is preferred below.
            if not (204 <= k <= 216):
                continue
            fx, fz = full_four_cycles(hx), full_four_cycles(hz)
            short = (max(am[0], bm[0]), am[0] + bm[0],
                     max(am[1], bm[1]), am[1] + bm[1])
            cycles = (max(fx, fz), fx + fz)

            if mode == 1:
                tail = (bm[3], am[3], be, ae, bm[2], am[2])
            elif mode == 2:
                tail = (max(ae, be), ae + be,
                        max(am[2], bm[2]), am[2] + bm[2],
                        bm[3], am[3])
            else:
                # B-kernel weaknesses replicate X-type logicals across
                # A columns, relevant to the parent's Z-memory bottleneck.
                tail = (be, ae, bm[3], am[3], bm[2], am[2])
            key = short + cycles + (int(k != 204),) + tail
            if best is None or key < best[0]:
                best = key, A, B, hx, hz, k

    if best is None:
        raise ValueError("no jointly screened filling in the 0.350 rate cell")
    key, A, B, hx, hz, k = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    cells = {"A": A, "B": B}
    return {"idx": idx, "design": design, "mode": mode,
            "cells": cells, "k": k, "screen": key,
            "palettes": palettes}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: A shared weight-eight screen establishes balanced circuit protection in this Z17 bed: slot 8 failed 105/5250 Z shots versus 101/23000 X shots, while slot 20 failed 100/8250 versus 100/50250; all 204 logical qubits were affected.
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
