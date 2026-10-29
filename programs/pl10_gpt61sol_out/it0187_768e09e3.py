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
CONCEPT = ("For the square-slope affine matrix group over F11, order 55, shapes (1,3)x(1,3), independently screened noncommuting trinomials should reduce collective circuit failures relative to the D14 parent's fitness 4.2963; measured worse-basis pL decides.")
GROUP = ("Jump from the dihedral presentation to affine matrices [[a,t],[0,1]] over F11 with a a nonzero square. This group has order 55, no involutions, and is isomorphic to C11:C5. Its bed is absent from BEDS. The related order-57 Frobenius rate-0.400 bed reached 4.8676, motivating a distinct odd-order experiment without establishing a group advantage.")
SHAPES = ("Equal (1,3) sides give 165-variable, 55-check components, n=550 and guaranteed k>=220, rate 0.400. Require complete verified k=220. The parent's weight-six witnesses span many sector-2 Arow/Brow blocks, with diffuse approximately 1.0-1.2x data/check enrichment including Acol/Brow checks; redesign both factors rather than infer one hotspot.")
FILLING = ("Twenty multisets of four quotient-triangle types prescribe independent voltage searches, paired with different companion contracts. Every cell has three distinct terms. Singleton actions extracted from verified lifts supply exact component screens for full rank, Tanner four-cycles and kernels through weight four. Finalists minimize exact weight-six kernel count, then small triple boundaries and a check-graph expansion heuristic. Complete verified lifts require CSS, k=220 and no logical through weight four. Both row-weight sums are 9+3=12 and six-round CNOT count is 23760; its reduction from 24192 comes from smaller n, not fewer gates per qubit. Target the parent's collective flips and 100-failure caps at 1500-2750 shots; structural screens are not circuit fitness.")
FAMILY_NAME = "r040_AffSquare11_q55_13x13_triangle20_exact6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("square-slope affine F11 requires order 55")
    # (a,t) encodes [[a,t],[0,1]]; square slopes form a subgroup.
    slopes = tuple(sorted({x*x % 11 for x in range(1, 11)}))
    positions = {a: i for i, a in enumerate(slopes)}
    els = [(a, t) for a in slopes for t in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 11)

    def mul(g, h):
        a, t = red(g)
        c, u = red(h)
        return (a*c % 11, (t + a*u) % 11)

    def inv(g):
        a, t = red(g)
        ai = pow(a, 9, 11)
        return (ai, -ai*t % 11)

    def idx(g):
        a, t = red(g)
        return 11*positions[a] + t

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _af_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _af_rows(columns, checks):
    rows = [0] * checks
    for c, word in enumerate(columns):
        while word:
            bit = word & -word
            rows[bit.bit_length() - 1] |= 1 << c
            word ^= bit
    return rows


def _af_clean4(columns):
    """Exact component exclusion through four; all columns have odd degree."""
    if 0 in columns or len(set(columns)) != len(columns):
        return False
    seen = set()
    for i, x in enumerate(columns):
        for j in range(i):
            value = x ^ columns[j]
            if value in seen:
                return False
            seen.add(value)
    # Odd numbers of odd-weight columns cannot sum to zero.
    return True


def _af_graph(columns, checks):
    """Exact C4; exact Tanner C6 after C4 exclusion."""
    adjacency = [0] * checks
    pairs = {}
    four = 0
    for word in columns:
        vertices = []
        while word:
            bit = word & -word
            vertices.append(bit.bit_length() - 1)
            word ^= bit
        assert len(vertices) == 3
        for i, u in enumerate(vertices):
            for v in vertices[:i]:
                key = (min(u, v), max(u, v))
                old = pairs.get(key, 0)
                four += old
                pairs[key] = old + 1
                adjacency[u] |= 1 << v
                adjacency[v] |= 1 << u
    if four:
        return four, None, adjacency
    triangles = 0
    for u in range(checks):
        later = adjacency[u] & ~((1 << (u + 1)) - 1)
        while later:
            bit = later & -later
            v = bit.bit_length() - 1
            triangles += (adjacency[u] & adjacency[v]
                          & ~((1 << (v + 1)) - 1)).bit_count()
            later ^= bit
    six = triangles - len(columns)
    assert six >= 0
    return 0, six, adjacency


def _af_radius(adjacency):
    """Approximate nontrivial spectral radius, not a certified bound."""
    from math import sin, sqrt
    checks = len(adjacency)
    neighbors = []
    for word in adjacency:
        row = []
        while word:
            bit = word & -word
            row.append(bit.bit_length() - 1)
            word ^= bit
        neighbors.append(row)
    assert len({len(row) for row in neighbors}) == 1
    result = 0.0
    for seed in (1, 5):
        vector = [sin(seed * (i + 1)) for i in range(checks)]
        mean = sum(vector) / checks
        vector = [v - mean for v in vector]
        norm = sqrt(sum(v*v for v in vector))
        vector = [v / norm for v in vector]
        for step in range(24):
            image = [sum(vector[j] for j in row) for row in neighbors]
            mean = sum(image) / checks
            image = [v - mean for v in image]
            norm = sqrt(sum(v*v for v in image))
            if norm < 1e-12:
                break
            vector = [v / norm for v in image]
        result = max(result, norm)
    return round(result, 8)


def _af_six_boundary(columns):
    """Exact six-word count following <=4 exclusion; boundary surrogates.

    Distinct triples with equal syndrome are disjoint. Every six-set
    has ten unordered triple partitions. No group-orbit shortcut is
    used, so stabilizers of supports cannot corrupt the count.
    """
    histogram = {}
    collisions = one = three = 0
    size = len(columns)
    for a in range(size - 2):
        x = columns[a]
        for b in range(a + 1, size - 1):
            pair = x ^ columns[b]
            for c in range(b + 1, size):
                value = pair ^ columns[c]
                old = histogram.get(value, 0)
                collisions += old
                histogram[value] = old + 1
                weight = value.bit_count()
                one += int(weight == 1)
                three += int(weight == 3)
    assert collisions % 10 == 0
    return collisions // 10, one, three


def _af_logical4(checks, stabilizers, width):
    """Exact existence of a nontrivial quantum logical through weight four."""
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

    columns = _af_columns(checks, width)
    seen = {0: 0}
    for i, value in enumerate(columns):
        word = 1 << i
        if value in seen:
            if nontrivial(word ^ seen[value]):
                return True
        else:
            seen[value] = word
    for i, x in enumerate(columns):
        for j in range(i):
            value = x ^ columns[j]
            word = (1 << i) | (1 << j)
            if value in seen:
                if nontrivial(word ^ seen[value]):
                    return True
            else:
                seen[value] = word
    return False


def NEW_CTX(rng, G, idx):
    """Twenty quotient architectures; independently optimize actual components."""
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 55 and A_SHAPE == B_SHAPE == (1, 3)
    counts = [[3, 3, 3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)
    assert 9 + 3 <= min(12, W_MAX)
    mask = (1 << q) - 1
    width = 3 * q
    unit = [[[G.e]]]

    # Obtain singleton actions exclusively from verified lifts. XOR
    # composition below is exact linearity of these verified actions.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], unit)
        _, hz, _ = lifted_matrices(G, unit, [[[g]]])
        actions["A"][G.idx(g)] = _af_columns(
            [row & mask for row in hx], q)
        actions["B"][G.idx(g)] = _af_columns(
            [row & mask for row in hz], q)

    # Types R1/R2 have quotient exponents (0,0,1)/(0,0,2).
    # Types T1/T2 have (0,1,2)/(0,1,3). Their quotient triangle
    # architectures differ; the twenty multisets are deliberate
    # search constraints, not twenty transformations of one filling.
    book = list(combinations_with_replacement(range(4), 3))
    assert len(book) == 20
    design, batch = idx % 20, idx // 20
    brief_a = book[design]
    brief_b = book[(design + 7 + batch % 3) % 20]

    def draw(brief, marker):
        rotations = sum(kind < 2 for kind in brief)
        if rotations == 3:
            steps = (1, 2, 3) if marker % 2 == 0 else (1, 2, 4)
        elif rotations == 2:
            steps = (1, 2) if marker % 2 == 0 else (1, 3)
        else:
            steps = (1,)
        cells = []
        rpos = 0
        for position, kind in enumerate(brief):
            if kind < 2:
                exponent = kind + 1
                phase = 0 if rpos == 0 else rng.randrange(11)
                u = (1, steps[rpos])
                v = (pow(3, exponent, 11), phase)
                rpos += 1
            else:
                exponent = 2 if kind == 2 else 3
                # Normalize one noncommuting triangle's affine gauge
                # when the contract has no rotation-containing cell.
                a, b = ((0, 1) if rotations == 0 and position == 0
                        else (rng.randrange(11), rng.randrange(11)))
                u, v = (3, a), (pow(3, exponent, 11), b)
            if G.mul(u, v) == G.mul(v, u):
                return None
            cells.append([G.e, u, v])
        return [cells]

    def component(grid, side):
        columns = []
        for cell in grid[0]:
            terms = [actions[side][G.idx(g)] for g in cell]
            columns.extend(terms[0][c] ^ terms[1][c] ^ terms[2][c]
                           for c in range(q))
        return columns

    def select(side, brief, marker):
        pool, seen = [], set()
        accepted = 0
        for trial in range(1100):
            grid = draw(brief, marker)
            if grid is None:
                continue
            signature = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell))
                for cell in grid[0]))
            if signature in seen:
                continue
            seen.add(signature)

            # Cheap ordered-difference prefilter. Actual component
            # cycle and kernel checks below remain authoritative.
            differences = set()
            repeated = False
            for cell in grid[0]:
                for g in cell:
                    for h in cell:
                        if g == h:
                            continue
                        value = G.idx(G.mul(g, G.inv(h)))
                        if value in differences:
                            repeated = True
                        differences.add(value)
            if repeated:
                continue

            columns = component(grid, side)
            four, cycles, adjacency = _af_graph(columns, q)
            if four:
                continue
            rows = _af_rows(columns, q)
            if gf2_rank(rows) != q or not _af_clean4(columns):
                continue
            accepted += 1
            pool.append((cycles, trial, grid, rows, columns, adjacency))
            pool.sort(key=lambda item: item[:2])
            del pool[16:]
            if accepted >= 64:
                break

        if not pool:
            raise ValueError("no clean full-rank component for affine brief")
        refined = []
        for cycles, trial, grid, rows, columns, adjacency in pool[:8]:
            six, one, three = _af_six_boundary(columns)
            radius = _af_radius(adjacency)
            key = ((six, one, radius, three, cycles) if batch % 2 == 0
                   else (six, one, three, radius, cycles))
            refined.append((key, trial, grid, rows))
        refined.sort(key=lambda item: item[:2])
        return refined[:3]

    aa = select("A", brief_a, design + batch)
    bb = select("B", brief_b, design // 4 + batch)
    options = []
    for ka, ta, A, ar in aa:
        for kb, tb, B, br in bb:
            joint = tuple(v for a, b in zip(ka, kb)
                          for v in (max(a, b), a + b))
            options.append((joint, ta, tb, A, B, ar, br))
    options.sort(key=lambda item: item[:3])

    for score, _, _, A, B, ar, br in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or k != 220:
            continue

        # Check the screened actions against the complete final lift,
        # including the actual right-action B factor.
        actual_a = [
            sum(((row >> (3*j*q)) & mask) << (j*q)
                for j in range(3))
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << width) - 1) for row in hz[:q]]
        assert actual_a == ar and actual_b == br
        if _af_logical4(hx, hz, n) or _af_logical4(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 12
        assert 6 * sum(row.bit_count() for row in hx + hz) == 23760
        return {"A": A, "B": B, "k": k,
                "contract": (brief_a, brief_b),
                "component_screen": score}
    raise ValueError("no complete affine filling passed target-rate screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform-label supports adequately protect this D14 (6,2)x(6,2) bed: all five measured parent codes have sector-2 weight-four witnesses and worse-basis failed-shot fractions of 0.268-0.452, despite mostly stage-1 settlement.
# REFUTED: Balanced D14 matching components screened against short dependencies suffice for rare circuit failures: fitness was 4.2963, all five measured codes retained abundant sector-2 weight-six witnesses, and worst-basis failures reached about 100 in 1500-2250 shots with flips spanning all 224 logicals.
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
