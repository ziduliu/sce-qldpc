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
CONCEPT = ("For generalized quaternion Q16, order 16, shapes (1,4)x(2,7), noncommuting A geometries and triangle-free graphic B components should improve worse-basis circuit reliability above 3.1933 while retaining the 0.500 rate cell.")
GROUP = ("Jump from cyclic C19 to Q16: x^8=1, y^2=x^4, y*x*y^-1=x^-1. Noncentral ratios have orders four or eight, permitting noncommuting supports without involutory binomial ratios. BEDS has no order-16 (1,4)x(2,7) bed; results on other order-16 shapes do not establish an advantage here. The fixed lift gives n=480.")
SHAPES = ("Retain (1,4)x(2,7): A has 64 variables/16 checks and B has 112 variables/32 checks. n=480 and kb=240 give guaranteed rate 0.500; complete verified ranks require k=243, rate 0.50625. Parent failures affected all 285 logicals across many Acol/Bcol blocks, with diffuse 1.0-1.2x data/check enrichment, so neither an Acol nor a Brow hotspot is established.")
FILLING = ("Twenty contracts cross five canonical A geometries with four B graph architectures; additional harness batches consume additional canonical geometries and change B voltage-coset contracts. A column weights are [2,2,3,3], B column weights all two. HX weight is twelve and HZ weight at most eleven; six-round CNOT count is 15360 versus 19494. Verified component screening excludes weights <=3, then full-CSS stabilizer quotienting counts weight-four ambiguities and possible shared-ancilla exposure in both bases. Target the parent's 153-171 X failures/250 shots and widespread logical flips; these structural costs neither simulate ASC hooks nor certify pL.")
FAMILY_NAME = "r050_Q16_14x27_noncommuting_graph_hookbudget_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q16 requires order 16")
    # Canonical x^a*y^b: x^8=1, y^2=x^4, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 4*b*d) % 8,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 8, 0) if b == 0 else ((a + 4) % 8, 1)

    def idx(g):
        a, b = red(g)
        return a + 8*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Four graphic B architectures, all with seven degree-two columns.
# Row totals are 7/7 or 8/6; zeros are prescribed, not repairs.
_B_PROFILES = (
    ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(0,2),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
)


def _kernel56(rows, ncols):
    """Exact weight-five/six counts when kernel weights <=4 are absent.

    Pair/triple syndrome matches count each weight-five word ten times.
    Distinct triple collisions count each weight-six word ten times.
    Smaller-word exclusion makes these colliding supports disjoint.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            s = x ^ y
            pairs[s] = pairs.get(s, 0) + 1
    triples = {}
    five = six = 0
    for i in range(ncols):
        for j in range(i):
            s = columns[i] ^ columns[j]
            for h in range(j):
                value = s ^ columns[h]
                five += pairs.get(value, 0)
                six += triples.get(value, 0)
                triples[value] = triples.get(value, 0) + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def NEW_CTX(rng, G, idx):
    """Canonical quaternion A contracts and optimized graphic B voltages."""
    from itertools import combinations
    from math import exp

    q = G.q
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    assert q == 16

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        # Zero partners are extraction probes, not submitted codes.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    # Automorphisms x->x^u, y->x^v*y, u odd. Independent right
    # translations are A-column gauges. Canonicalization removes these
    # obvious copies; arbitrary Tanner isomorphism remains downstream.
    automorphisms = [
        tuple(G.idx(((u*a + v*b) % 8, b))
              for a, b in G.elements)
        for u in (1, 3, 5, 7) for v in range(8)
    ]

    def normal(cell, transform):
        values = [G.elements[transform[G.idx(g)]] for g in cell]
        return min(tuple(sorted(
            G.idx(G.mul(g, G.inv(origin))) for g in values))
                   for origin in values)

    triangles = [
        (G.e, (d, 0), (a, 1))
        for d in (1, 2, 3, 5, 6, 7) for a in range(8)
    ]
    binomial_pairs = (
        ((G.e, (1, 0)), (G.e, (3, 0))),
        ((G.e, (1, 0)), (G.e, (0, 1))),
        ((G.e, (0, 1)), (G.e, (1, 1))),
    )
    all_cells = set(triangles)
    for pair in binomial_pairs:
        all_cells.update(pair)
    normal_forms = {
        cell: tuple(normal(cell, transform)
                    for transform in automorphisms)
        for cell in all_cells
    }

    book, seen_geometry = [], set()
    for bins in binomial_pairs:
        for first, second in combinations(triangles, 2):
            cells = bins + (first, second)
            signature = min(
                tuple(sorted(normal_forms[cell][t] for cell in cells))
                for t in range(len(automorphisms))
            )
            if signature in seen_geometry:
                continue
            seen_geometry.add(signature)
            grid = [[[G.elements[g] for g in cell]
                     for cell in signature]]
            rows, width = component(grid, "A")
            stats = _kernel_statistics(rows, width)
            if stats[0] or stats[1] or gf2_rank(rows) != q:
                continue
            key = (stats[2], stats[3], _four_cycles(rows),
                   _six_cycles(rows), signature)
            book.append((key, grid))
    book.sort(key=lambda item: item[0])
    wanted = 5 * batch + marker
    if wanted >= len(book):
        raise ValueError("Canonical quaternion A contract book exhausted")
    akey, A = book[wanted]

    profile = _B_PROFILES[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2, 2, 3, 3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    assert all(sum(p) == 2 for p in profile)
    assert max(map(sum, sb)) <= 8

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [(1, j) for j in cross[1:]]
    outside_count = 1 + (marker + batch) % (len(cross) - 1)
    outside_sites = set(cross[1:1 + outside_count])
    rotations = [(a, 0) for a in range(1, 8)]
    outside = [(a, 1) for a in range(8)]
    noncentral = rotations[:3] + rotations[4:] + outside
    internal_palette = (
        noncentral,
        [(a, 0) for a in (1, 3, 5, 7)],
        [(a, 0) for a in (1, 3, 5, 7)] + outside,
    )[batch % 3]

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for i, j in internal:
            grid[i][j] = [G.e, rng.choice(internal_palette)]
        rv = rng.sample(rotations, len(cross) - 1 - outside_count)
        ov = rng.sample(outside, outside_count)
        for j in cross:
            grid[0][j] = [G.e]
            g = (G.e if j == cross[0] else
                 ov.pop() if j in outside_sites else rv.pop())
            grid[1][j] = [g]
        return grid

    def assess_B(grid):
        rows, width = component(grid, "B")
        stats = _kernel_statistics(rows, width)
        # B is an incidence matrix. These are parallel-edge, triangle
        # and four-cycle statistics on the actual helper-built graph.
        energy = (1000 * stats[0] + 100 * stats[1] + stats[2]) / q
        return energy, stats, rows

    pool, seen_B = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != 2*q - 1:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen_B:
            return
        seen_B.add(signature)
        serial += 1
        key = (stats[2], stats[3])
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[6:]

    for restart in range(4):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(220):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [G.e, rng.choice(internal_palette)]
            else:
                palette = outside if j in outside_sites else rotations
                occupied = {proposal[1][c][0] for c in cross if c != j}
                choices = [g for g in palette if g not in occupied]
                if not choices:
                    continue
                proposal[i][j] = [rng.choice(choices)]
            proposed, ps, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 4.0 * (0.982 ** step) + 0.04
            if delta <= 0 or rng.random() < exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No connected triangle-free quaternion B contract")

    def quotient_columns(stabilizers, width):
        # Linear canonical remainders modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residual = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                residual[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def logical_risk(checks, stabilizers, width):
        """Exact short-logical classification; hook costs are surrogates.

        A shared opposite-basis check marks possible ancilla spread,
        not a claim about the ordering of the measured ASC circuit.
        """
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        exposure = [a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]
        buckets = {}
        hooks = double_hooks = weighted = partitions = 0

        for i, si in enumerate(syndrome):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                value = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(value)
                if third is not None and third != label:
                    return None
                h = (hi & spread[j]).bit_count()
                w = exposure[i] * exposure[j]
                bucket = buckets.get(value)
                if bucket is None:
                    buckets[value] = [1, h, w, {label: [1, h, w]}]
                    continue
                same = bucket[3].get(label)
                sc, sh, sw = same if same is not None else (0, 0, 0)
                dc = bucket[0] - sc
                dh = bucket[1] - sh
                dw = bucket[2] - sw
                partitions += dc
                hooks += h * dc + dh
                double_hooks += h * dh
                weighted += w * dc + dw
                bucket[0] += 1
                bucket[1] += h
                bucket[2] += w
                if same is None:
                    bucket[3][label] = [1, h, w]
                else:
                    same[0] += 1
                    same[1] += h
                    same[2] += w
        assert partitions % 3 == 0
        return hooks, double_hooks, weighted, partitions // 3

    winner = None
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 480 or k != 243:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        score = tuple(value for t in range(4)
                      for value in (max(px[t], pz[t]), px[t] + pz[t]))
        score += coarse
        if winner is None or score < winner[0]:
            winner = (score, B, hx, hz, px, pz)

    if winner is None:
        raise ValueError("No k=243 quaternion CSS filling passed screening")
    score, B, hx, hz, px, pz = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    probe_a, _ = component(A, "A")
    probe_b, _ = component(B, "B")
    assert actual_a == probe_a and actual_b == probe_b
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert 6 * sum(row.bit_count() for row in hx + hz) == 15360
    assert 19 * 480 <= 40 * 243 < 21 * 480
    return {"A": A, "B": B, "design": (architecture, wanted),
            "voltage_contract": (outside_count, batch % 3),
            "k": 243, "logical_risk": (px, pz),
            "A_contract": akey, "cycle_score": score}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the already screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: A<=3/B<=4 component exclusion balances C19 (1,4)x(2,7) circuit memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored 3.1933 despite substantially lower Z failed-shot rates.
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
