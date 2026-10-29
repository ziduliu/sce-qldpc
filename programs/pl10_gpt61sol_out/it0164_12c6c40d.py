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
CONCEPT = ("Cyclic Z17, order 17, shapes (2,4)x(3,7): side-disjoint internal differences and distributed B reinforcement should improve worse-basis circuit reliability while reducing extraction load and retaining every submitted code in cell 0.250.")
GROUP = ("Keep the exact metacyclic implementation m=1,r=17,s=1. The new size factor remains 34, so n=578 without an order change. The Z17 bed has reached about 7.2; the related Z29 two-row bed reached 7.185 with weight-nine extraction. These motivate retaining the family and testing lower gate exposure, not a group-only advantage.")
SHAPES = ("A=(2,4) has 68 variables and 34 checks; B=(3,7) has 119 variables and 51 checks. The guaranteed bound is kb=136, rate 4/17=0.235294. Controlled component ranks predict k=136..148, and complete verified CSS ranks require that interval, giving rates 0.2353..0.2561 in cell 0.250. Parent failure maps implicated multiple Acol/Bcol and Acol/Brow blocks; slot50 had four Z failures, slot56 two X failures, and slot15 thirty Z failures, without a consistent isolated hotspot.")
FILLING = ("Twenty count architectures trade degree-three/four A columns against distributed B doubletons or diagonal trinomials. Additional slots receive new canonical internal-geometry contracts, never transformations of saved codes. Both CSS weight sums are <=12; extraction uses 25092..28764 CNOTs versus 27744..34272 in the parent's primary architectures. A/B internal undirected differences are disjoint, removing mixed-sector four-cycles; final verified matrices check this identity. Optimize geometry-preserving cell voltages, require component ranks with only augmentation-forced deficits, test kernel words through six, then cap sampled protection at twelve and reduce normalized witness multiplicity. These are circuit-pL hypotheses, not distance or ASC certificates.")
FAMILY_NAME = "r025_m1r17s1_24x37_disjoint_geometry_gatelean_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same metacyclic construction rule; factor 34 requires q<=17.
    # Z_1 x| Z_17 is cyclic, with the unchanged trivial action.
    if not valid_q(q):
        raise ValueError("this shape bed requires order 17")
    m, r, s = 1, 17, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize a count/geometry contract using verified component lifts."""
    import math
    from itertools import combinations, permutations

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    q = G.q
    assert (ra, ca, rb, cb, q) == (2, 4, 3, 7, 17)
    batch, design = divmod(idx, 20)

    # A0: columns three, rows six/six.
    # A1: one degree-four column, rows seven/six.
    # A2: two degree-four columns, rows eight/six.
    # A3: zero/trinomial incidence, columns three, rows six/six.
    a_profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((2, 1, 2, 2), (2, 2, 1, 1)),
        ((2, 2, 2, 2), (2, 2, 1, 1)),
        ((0, 2, 2, 2), (3, 1, 1, 1)),
    )
    b_edits = (
        (),
        ((0, 0, 2),),
        ((0, 0, 2), (1, 1, 2)),
        ((0, 0, 2), (1, 1, 2), (2, 2, 2)),
        ((0, 0, 2), (0, 1, 2)),
        ((0, 0, 3),),
        ((0, 0, 3), (1, 1, 3)),
        ((0, 0, 3), (1, 1, 3), (2, 2, 3)),
    )
    designs = (
        (0, 0), (0, 1), (0, 2), (0, 3),
        (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 5), (3, 6), (3, 7),
    )
    pa, pb = designs[design]
    sa = [list(row) for row in a_profiles[pa]]
    sb = [[1] * cb for _ in range(rb)]
    for i, j, weight in b_edits[pb]:
        sb[i][j] = weight
    counts = {"A": sa, "B": sb}

    # Degree-four A columns pair only with B rows <=8.
    # B trinomials pair with A columns three and A rows six:
    # HX <=6+5=11; HZ <=9+3=12.
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= CELL_MAX
               for grid in counts.values() for row in grid for w in row)

    def augmentation_rank(support):
        return gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support
        ])

    deficits = {
        side: len(support) - augmentation_rank(support)
        for side, support in counts.items()
    }

    def direction(value):
        value %= q
        return min(value, q - value)

    def differences(cell):
        return {direction(x - y)
                for x, y in combinations(cell, 2)}

    def normalized(cell, multiplier):
        if not cell:
            return ()
        values = [multiplier * x % q for x in cell]
        return min(tuple(sorted((x - origin) % q for x in values))
                   for origin in values)

    def geometry_key(grids):
        # Remove internal origins, global cyclic automorphisms and
        # component row/column permutations. This is not a complete
        # Tanner-isomorphism classifier; downstream deduplication rules.
        images = []
        for multiplier in range(1, q):
            sides = []
            for side in ("A", "B"):
                support = counts[side]
                grid = grids[side]
                row_images = []
                for order in permutations(range(len(support))):
                    row_images.append(tuple(sorted(
                        tuple((support[i][j],
                               normalized(grid[i][j], multiplier))
                              for i in order)
                        for j in range(len(support[0]))
                    )))
                sides.append(min(row_images))
            images.append(tuple(sides))
        return min(images)

    book_rng = random.Random(170250701 + 1009 * design)
    seen = set()
    number = 0
    geometry = None
    for proposal in range(12000):
        step = book_rng.randrange(1, q)
        b_triangle = (0, step, 3 * step % q)
        b_palette = sorted(differences(b_triangle))
        a_palette = [d for d in range(1, (q + 1) // 2)
                     if d not in b_palette]
        a_triangles = [
            (0, x, y)
            for x, y in combinations(range(1, q), 2)
            if len(differences((0, x, y))) == 3
            and differences((0, x, y)) <= set(a_palette)
        ]
        if not a_triangles:
            continue

        grids = {}
        legal = True
        for side, palette in (("A", a_palette), ("B", b_palette)):
            grid = []
            for row in counts[side]:
                used = set()
                cells = []
                for weight in row:
                    if weight == 0:
                        cell = ()
                    elif weight == 1:
                        cell = (0,)
                    elif weight == 2:
                        choices = [d for d in palette if d not in used]
                        if not choices:
                            legal = False
                            break
                        cell = (0, book_rng.choice(choices))
                    else:
                        choices = (a_triangles if side == "A"
                                   else [b_triangle])
                        choices = [c for c in choices
                                   if not (differences(c) & used)]
                        if not choices:
                            legal = False
                            break
                        cell = book_rng.choice(choices)
                    used.update(differences(cell))
                    cells.append(cell)
                if not legal:
                    break
                grid.append(cells)
            if not legal:
                break
            grids[side] = grid
        if not legal:
            continue
        # A degree-four column also needs unequal top/bottom gaps.
        if any(sa[0][j] == sa[1][j] == 2
               and differences(grids["A"][0][j])
               == differences(grids["A"][1][j])
               for j in range(ca)):
            continue
        key = geometry_key(grids)
        if key in seen:
            continue
        seen.add(key)
        if number == batch:
            geometry = grids
            break
        number += 1
    if geometry is None:
        raise ValueError("no unused side-disjoint internal geometry")

    internal = {
        side: set().union(*(differences(cell)
                            for row in geometry[side] for cell in row))
        for side in ("A", "B")
    }
    assert not (internal["A"] & internal["B"])

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw(support, side):
        grid = []
        for i, row in enumerate(support):
            cells = []
            for j, weight in enumerate(row):
                origin = G.pick(rng)
                cell = [G.mul(origin, (0, delta))
                        for delta in geometry[side][i][j]]
                assert len(cell) == weight
                assert len({G._key(g) for g in cell}) == weight
                cells.append(cell)
            grid.append(cells)
        return grid

    positions = {
        side: [(i, j) for i, row in enumerate(support)
               for j, weight in enumerate(row) if weight]
        for side, support in counts.items()
    }

    component_cache = {}

    def component_rows(A, B):
        # Zero companions are probes, never submitted slot matrices.
        # Project components only from the verified scaffold lift.
        # Cache the unchanged side during single-label mutations.
        result = []
        for side, grid, width in (("A", A, ca), ("B", B, cb)):
            fingerprint = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            old = component_cache.get(side)
            if old is None or old[0] != fingerprint:
                if side == "A":
                    rows = lifted_matrices(G, grid, [[[]]])[0]
                else:
                    rows = lifted_matrices(G, [[[]]], grid)[1]
                mask = (1 << (width * q)) - 1
                rows = [row & mask for row in rows]
                component_cache[side] = (fingerprint, rows)
            result.append(component_cache[side][1])
        return result[0], result[1]

    def cycle_counts(rows, nc):
        # Exact four-cycle count; triangle proxy subtracts single-column
        # stars and equals the six-cycle count when there are no 4-cycles.
        m = len(rows)
        adj = [0] * m
        degrees = [0] * nc
        four = 0
        for i, row in enumerate(rows):
            bits = row
            while bits:
                bit = bits & -bits
                degrees[bit.bit_length() - 1] += 1
                bits ^= bit
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(m):
            later = adj[i] >> (i + 1)
            while later:
                bit = later & -later
                j = i + 1 + bit.bit_length() - 1
                triangles += ((adj[i] & adj[j]) >> (j + 1)).bit_count()
                later ^= bit
        stars = sum(d * (d - 1) * (d - 2) // 6 for d in degrees)
        return four, max(0, triangles - stars)

    def analyze(A, B):
        ha, hb = component_rows(A, B)
        a4, a6 = cycle_counts(ha, ca * q)
        b4, b6 = cycle_counts(hb, cb * q)
        four, six = a4 + b4, a6 + b6
        energy = (100.0 * four + six / 5.0) / q
        return energy, ha, hb, four, six

    def column_syndromes(rows, nc):
        columns = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= tag
                row ^= bit
        return columns

    def short_kernel(rows, nc):
        """Exact existence test for a nonzero component word <=6.

        Every subset of size <=3 is visited once. Equal syndromes of
        distinct subsets give a nonzero kernel word of weight <=6.
        This tests components, not quantum logicals or circuit faults.
        """
        columns = column_syndromes(rows, nc)
        seen = {0}
        for syndrome in columns:
            if syndrome in seen:
                return True
            seen.add(syndrome)
        for i in range(nc):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
        for i in range(nc):
            for j in range(i):
                pair = columns[i] ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    if syndrome in seen:
                        return True
                    seen.add(syndrome)
        return False

    def kernel_scan(rows, nc, seed, trials):
        """Sample actual fundamental kernel words and their pair XORs.

        Returned minima are upper bounds only. Pair XORs improve the
        witness search without treating a missed word as a certificate.
        """
        srng = random.Random(seed)
        columns = column_syndromes(rows, nc)
        order = list(range(nc))
        best = nc + 1
        small = set()

        def record(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight <= 14:
                small.add(word)

        for _ in range(trials):
            srng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    record(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    record(word ^ other)
            if best <= 6:
                break
        return best, len(small)

    winner = None
    winner_quality = None
    for restart in range(6):
        A, B = draw(sa, "A"), draw(sb, "B")
        current = analyze(A, B)
        local_A, local_B = clone(A), clone(B)
        local = current
        for step in range(300):
            # B has more cells and a higher component rate, so allocate
            # more voltage proposals there rather than preserve 0.65 A.
            side = "B" if rng.random() < 0.65 else "A"
            grid = B if side == "B" else A
            i, j = rng.choice(positions[side])
            old = grid[i][j]
            shift = rng.choice(G.elements[1:])
            # Preserve the indexed internal geometry and disjoint
            # difference sets. Empty cells are never mutation targets.
            grid[i][j] = [G.mul(shift, g) for g in old]
            proposed = analyze(A, B)
            delta = proposed[0] - current[0]
            temperature = 18.0 * (0.985 ** step) + 0.12
            accept = (delta <= 0 or
                      rng.random() < math.exp(-min(700.0,
                                                   delta / temperature)))
            if accept:
                current = proposed
                if (current[0] < local[0] or
                        (current[0] == local[0] and rng.random() < 0.04)):
                    local_A, local_B = clone(A), clone(B)
                    local = current
            else:
                grid[i][j] = old

        A, B = local_A, local_B
        # Only augmentation-forced component deficits are admitted.
        # Complete CSS ranks, not this model, decide the submitted k.
        if gf2_rank(local[1]) != ra * q - deficits["A"]:
            continue
        if gf2_rank(local[2]) != rb * q - deficits["B"]:
            continue
        k = lifted_k(G, A, B)
        if not 136 <= k <= 148:
            continue
        bad_a = short_kernel(local[1], ca * q)
        bad_b = short_kernel(local[2], cb * q)
        da, na = kernel_scan(local[1], ca * q,
                             271828 + idx * 1009, 96)
        db, nb = kernel_scan(local[2], cb * q,
                             314159 + idx * 1009, 128)
        # Cap the sampled protection reward at twelve. Normalize
        # multiplicity because the component lengths now differ.
        ea, eb = na / (ca * q), nb / (cb * q)
        quality = (int(not bad_a and not bad_b),
                   -int(bad_a) - int(bad_b),
                   min(da, db, 12),
                   -max(ea, eb),
                   -(ea + eb),
                   -local[3],
                   -local[4])
        if winner_quality is None or quality > winner_quality:
            winner_quality = quality
            winner = {"A": clone(A), "B": clone(B)}

    if winner is None:
        # Preserve the same legal contract if annealing missed its ranks.
        # Fresh candidates are screened before submission, not repaired.
        for trial in range(64):
            A, B = draw(sa, "A"), draw(sb, "B")
            ha, hb = component_rows(A, B)
            if gf2_rank(ha) != ra * q - deficits["A"]:
                continue
            if gf2_rank(hb) != rb * q - deficits["B"]:
                continue
            k = lifted_k(G, A, B)
            if not 136 <= k <= 148:
                continue
            stats = analyze(A, B)
            quality = (-1, -stats[3], -stats[4])
            if winner_quality is None or quality > winner_quality:
                winner_quality = quality
                winner = {"A": clone(A), "B": clone(B)}
        if winner is None:
            raise ValueError("count/geometry contract missed verified rate")

    hx, hz, n = lifted_matrices(G, winner["A"], winner["B"])
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert css_ok(hx, hz)
    assert n == 578 and 136 <= k <= 148
    assert 9 * n <= 40 * k < 11 * n
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)

    ha, hb = component_rows(winner["A"], winner["B"])
    mask = (1 << q) - 1
    # Confirm component projections against the complete submitted lift.
    projected_a = [
        sum(((hx[(i * cb) * q + t] >> (j * cb * q)) & mask)
            << (j * q) for j in range(ca))
        for i in range(ra) for t in range(q)
    ]
    projected_b = [row & ((1 << (cb * q)) - 1)
                   for row in hz[:rb * q]]
    assert ha == projected_a and hb == projected_b

    def four_cycles(rows):
        return sum(v * (v - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for v in [(row & other).bit_count()])

    ac4, bc4 = four_cycles(ha), four_cycles(hb)
    xc4, zc4 = four_cycles(hx), four_cycles(hz)
    # Disjoint internal differences exclude mixed-sector contributions.
    assert xc4 == cb * ac4 + ra * bc4
    assert zc4 == rb * ac4 + ca * bc4
    expected_cnots = 6 * q * (
        (cb + rb) * sum(map(sum, sa))
        + (ca + ra) * sum(map(sum, sb))
    )
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
    return {"idx": idx, "architecture": (pa, pb), "geometry": batch,
            "k": k, "screen": winner_quality, "cells": winner,
            "cnots": expected_cnots, "four_cycles": (xc4, zc4)}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Stage-1 settlement is not a reliability certificate: parent slot 4 settled 1498/1500 X shots at stage 1 yet failed 102/1500.
# REFUTED: Check weight <=12 and exclusion of weight-six logicals certify zero circuit failures: all five current measured codes still had 5..11 X failures in 200000 shots, despite minimum found Z-type witness weight 12.
# REFUTED: Component kernel/cycle screening ensures ASC measurability: current slot16 had k=84, w=12 and d2s=14 but both ASC schedules timed out, leaving only four measured codes.
# REFUTED: Equal d2s=14 ensures comparable circuit reliability: current slot15 had thirty Z failures/200000 and s=6.372, while slot56 had at most two failures per basis and s=7.112.
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
