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
CONCEPT = ("Hypothesis: AGL(1,8) x C2, order 112, shapes (2,3)x(2,3), with order-14 binomials and verified-lift localized/full-component kernel screening, eliminates the parent's weight-40 collapse and yields five measured distance readings >=64 in cell 0.100; no distance lower bound is claimed.")
GROUP = ("Keep the frozen affine-F8 times central-C2 group: this bed leads cell 0.100 at fitness 278.7232, with four measured readings 63..66. Distinct multiplicative coordinates and opposite central coordinates give binomial relative order 14; actual component restrictions screen small common invariant kernels.")
SHAPES = ("Keep equal (2,3) components with 336 variables and 224 checks each; q=112 gives n=1456, k>=112, and guaranteed rate 1/13. Parent measured minima were sector 1: slots 19/34 used Acol2 across B columns, slot 51 used Acol1, and slot 40 collapsed to weight 40 on Acol0/1 across Bcol1/2. Exact ranks retain cell 0.100.")
FILLING = ("Twenty explicit support pairs have column degrees 4..5, full-rank binary augmentation, and both check-weight maxima <=12. Fresh labels force every binomial ratio to order 14. Verified component columns reject extra single-column kernel dimensions, inspect two-column kernel words, and compete by randomized full-component kernel readings; four cycles break ties. These are classical upper readings, not quantum distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r010_affineF8xC2_23x23_local_kernel_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    # (u,v,z): affine map t -> alpha^v*t + u over F8,
    # times central C2. F8 uses alpha^3 + alpha + 1 = 0.
    if not valid_q(q):
        raise ValueError("this affine-field bed requires order 112")
    powers = (1, 2, 4, 3, 6, 7, 5)
    els = [(u, v, z) for z in range(2)
           for v in range(7) for u in range(8)]

    def field_mul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return out

    def red(g):
        return (g[0] & 7, g[1] % 7, g[2] & 1)

    def mul(g, h):
        return (g[0] ^ field_mul(powers[g[1]], h[0]),
                (g[1] + h[1]) % 7, g[2] ^ h[2])

    def inv(g):
        vi = (-g[1]) % 7
        return (field_mul(powers[vi], g[0]), vi, g[2])

    def idx(g):
        u, v, z = red(g)
        return (7 * z + v) * 8 + u

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


# Different density allocations, not permutations of one labeling.
# Every column has >=4 terms. Templates 3..8 are paired only with
# template 0: its row/column totals of 6/4 accommodate their density.
_SUPPORT_TEMPLATES = (
    ((2, 2, 2), (2, 2, 2)),  # 0: uniform baseline
    ((3, 2, 2), (2, 2, 2)),  # 1: one reinforced cell
    ((3, 2, 2), (2, 3, 2)),  # 2: balanced reinforcement
    ((3, 2, 2), (3, 2, 2)),  # 3: one degree-six column
    ((3, 3, 2), (2, 2, 3)),  # 4: asymmetric row density
    ((3, 3, 2), (3, 2, 3)),  # 5: distributed dense side
    ((3, 3, 2), (3, 3, 2)),  # 6: two degree-six columns
    ((4, 2, 2), (2, 4, 2)),  # 7: concentrated dense side
    ((4, 2, 2), (2, 3, 3)),  # 8: mixed concentration
    ((3, 3, 1), (2, 2, 3)),  # 9: uneven cells, balanced degrees
    ((4, 1, 2), (1, 4, 2)),  # 10: crossed concentrated cells
    ((3, 3, 1), (2, 1, 3)),  # 11: asymmetric balanced degrees
)
_SUPPORT_PAIRS = (
    (2, 2), (1, 2), (0, 2), (1, 1), (0, 1),
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (2, 9), (1, 9), (0, 9), (9, 9),
    (2, 10), (1, 10), (9, 10), (10, 10), (2, 11),
)


def _set_bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _affine_grid(rng, supports, phase):
    """Distinct v coordinates make every within-cell ratio non-involutory."""
    grid = []
    for i, counts in enumerate(supports):
        row = []
        for j, count in enumerate(counts):
            vs = list(range(7))
            rng.shuffle(vs)
            flip = rng.randrange(2)
            cell = []
            for t in range(count):
                if phase == 0:
                    z = flip ^ (t & 1)
                elif phase == 1:
                    z = flip ^ ((t & 1) if (i + j) & 1 else 0)
                else:
                    z = rng.randrange(2)
                cell.append((rng.randrange(8), vs[t], z))
            row.append(cell)
        grid.append(row)
    return grid


def _component_stats(rows, ncols):
    """Four cycles and check-projection triangle excess, not distance."""
    columns = [0] * ncols
    pairs = {}
    four = 0
    for r, row in enumerate(rows):
        ns = list(_set_bits(row))
        for u in ns:
            columns[u] |= 1 << r
        for a, u in enumerate(ns):
            for v in ns[a + 1:]:
                key = u * ncols + v
                previous = pairs.get(key, 0)
                four += previous
                pairs[key] = previous + 1

    # Reject isolated variables and identical component columns:
    # the latter supply classical weight-two kernel vectors.
    if not all(columns) or len(set(columns)) != ncols:
        return None

    adjacency = [0] * len(rows)
    forced_triangles = 0
    for signature in columns:
        degree = signature.bit_count()
        forced_triangles += degree * (degree - 1) * (degree - 2) // 6
        for r in _set_bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    # Require a connected component Tanner graph.
    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _set_bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _set_bits(neighbors >> (u + 1)):
            v += u + 1
            triangles += (neighbors & adjacency[v] &
                          (-1 << (v + 1))).bit_count()

    # With no four cycles, this is the number of genuine six cycles.
    # Otherwise it remains only a secondary projection statistic.
    return four, max(0, triangles - forced_triangles)


def _verified_components(G, hx, hz):
    """Extract both classical lifts from the scaffold's verified matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            lifted_row = hx[i * cb * q + r]  # beta = 0
            ar.append(sum(((lifted_row >> (j * cb * q)) & mask)
                          << (j * q) for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + r] & bmask             # A-column j = 0
          for a in range(rb) for r in range(q)]
    return ar, br


def NEW_CTX(rng, G, idx):
    """Screen fresh labels by actual localized and full component kernels."""
    from itertools import combinations

    # Four degree-14 and four degree-13 components. Every template has
    # binary augmentation rank two; no all-even component row is used.
    profiles = (
        ((3,2,2), (2,3,2)),
        ((3,3,1), (2,2,3)),
        ((4,1,2), (1,4,2)),
        ((4,2,1), (1,3,3)),
        ((1,1,4), (3,3,1)),
        ((1,2,3), (3,2,2)),
        ((1,3,2), (3,1,3)),
        ((2,3,1), (2,1,4)),
    )
    plans = (
        (0,0), (0,1), (0,2), (0,3), (1,1),
        (1,2), (1,3), (2,2), (2,3), (3,3),
        (0,4), (1,4), (2,4), (3,4), (0,5),
        (1,5), (2,5), (3,5), (0,6), (1,7),
    )
    ia, ib = plans[idx % len(plans)]
    sa, sb = profiles[ia], profiles[ib]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    for support in (sa, sb):
        augmentation = [
            sum((count & 1) << j for j, count in enumerate(row))
            for row in support
        ]
        assert gf2_rank(augmentation) == 2

    q, width = G.q, 3 * G.q
    diagnostic_seed = rng.randrange(1 << 62)

    def labels(support, side):
        grid = []
        for i, counts in enumerate(support):
            row = []
            for j, count in enumerate(counts):
                vs = rng.sample(range(7), count)
                flip = rng.randrange(2)
                if count == 1:
                    zs = [flip]
                elif count == 2:
                    # Distinct v and opposite z: relative order exactly 14.
                    zs = [flip, flip ^ 1]
                else:
                    ones = (1 + (idx + i + 2*j + side) %
                            (count - 1))
                    zs = [1] * ones + [0] * (count - ones)
                    rng.shuffle(zs)
                    zs = [z ^ flip for z in zs]
                row.append([(rng.randrange(8), v, z)
                            for v, z in zip(vs, zs)])
            grid.append(row)
        return grid

    def columns_and_cycles(rows):
        columns = [0] * width
        pairs = {}
        four = 0
        for r, row in enumerate(rows):
            ns = list(_set_bits(row))
            for c in ns:
                columns[c] |= 1 << r
            for a, b in combinations(ns, 2):
                key = a * width + b
                old = pairs.get(key, 0)
                four += old
                pairs[key] = old + 1
        if not all(columns) or len(set(columns)) != width:
            return None
        return columns, four

    def dependency_basis(columns, order):
        """Actual kernel vectors, represented in original column order."""
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
        return basis

    def basis_reading(basis, exhaustive=False):
        if not basis:
            return width + 1
        reading = min(word.bit_count() for word in basis)
        if exhaustive and len(basis) <= 14:
            # Gray-code enumeration is exact for this restricted kernel.
            word = 0
            for t in range(1, 1 << len(basis)):
                word ^= basis[(t & -t).bit_length() - 1]
                reading = min(reading, word.bit_count())
        else:
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    reading = min(reading, (word ^ other).bit_count())
        return reading

    def localized_reading(columns):
        """Inspect restrictions of the VERIFIED component, not a new lift."""
        blocks = [list(range(j*q, (j+1)*q)) for j in range(3)]
        reading = width + 1
        for block in blocks:
            basis = dependency_basis(columns, block)
            # One all-ones orbit is allowed; extra dimensions are not.
            if len(basis) > 1:
                return None
            reading = min(reading, basis_reading(basis, True))
        for a, b in combinations(range(3), 2):
            basis = dependency_basis(columns, blocks[a] + blocks[b])
            reading = min(reading, basis_reading(basis, True))
        return reading

    def sampled_reading(columns, seed, trials):
        """Fundamental words and their pair sums: upper readings only."""
        diagnostic = random.Random(seed)
        reading = width + 1
        for _ in range(trials):
            order = list(range(width))
            diagnostic.shuffle(order)
            basis = dependency_basis(columns, order)
            reading = min(reading, basis_reading(basis))
        return reading

    pool = []
    for trial in range(96):
        if trial >= 24 and len(pool) >= 4:
            break
        A, B = labels(sa, 0), labels(sb, 1)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _verified_components(G, hx, hz)
        ast, bst = columns_and_cycles(ar), columns_and_cycles(br)
        if ast is None or bst is None:
            continue
        ac, afour = ast
        bc, bfour = bst
        la, lb = localized_reading(ac), localized_reading(bc)
        if la is None or lb is None:
            continue

        # Common diagnostic permutations across candidates reduce one
        # source of selection noise. Finalists receive a longer probe.
        da = min(la, sampled_reading(ac, diagnostic_seed + 17, 2))
        db = min(lb, sampled_reading(bc, diagnostic_seed + 31, 2))
        key = (min(da, db), da + db,
               -max(afour, bfour), -(afour + bfour))
        if len(pool) >= 4 and key <= pool[-1]["key"]:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 8*k < n):
            continue
        item = dict(key=key, A=A, B=B, ac=ac, bc=bc,
                    da=da, db=db, four=(afour, bfour), k=k)
        pool.append(item)
        pool.sort(key=lambda item: item["key"], reverse=True)
        pool = pool[:4]

    if not pool:
        raise ValueError("no rank-legal labeling passed localized screening")

    best = None
    for item in pool:
        da = min(item["da"],
                 sampled_reading(item["ac"], diagnostic_seed + 101, 12))
        db = min(item["db"],
                 sampled_reading(item["bc"], diagnostic_seed + 211, 12))
        afour, bfour = item["four"]
        key = (min(da, db), da + db,
               -max(afour, bfour), -(afour + bfour),
               -(item["k"] - q))
        if best is None or key > best[0]:
            best = (key, item)

    key, item = best
    hx, hz, n = lifted_matrices(G, item["A"], item["B"])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert n - gf2_rank(hx) - gf2_rank(hz) == item["k"]
    return {"cells": {"A": item["A"], "B": item["B"]},
            "screen": key, "k": item["k"], "design": idx}


def CELL(rng, G, side, i, j, ctx):
    """Return the full kernel-screened cell without changing its labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A row-weight cap of 12 alone excludes abundant weight-6 logicals: the rolled C37 (2,6)x(2,6) parent had d=6 in all five measured slots, entirely in sector 1, despite k-kb ranging from 0 to 10.
# REFUTED: Non-involutory within-cell ratios and short-cycle selection exclude weight-40 sector-1 logicals on this affine bed: parent slot 40 fell from d2s=70 to d30s=40, with X witnesses on Acol0/1 across Bcol1/2.
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
