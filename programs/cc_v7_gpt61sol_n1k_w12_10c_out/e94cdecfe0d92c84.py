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
CONCEPT = ("On (F16 additive semidirect C5) x C2, order 160, A=(1,2), B=(1,3), noncommuting trinomial cells selected against short component kernels should produce five new measured distance readings >=14 in cell 0.300. Search readings remain provisional upper bounds.")
GROUP = ("Jump to an affine-field subgroup times C2: the normal subgroup is C2^4, unlike the parent's cyclic order-27 normal subgroup. The related affine-field order-112 bed has strong provisional readings, motivating this untested order-160 bed. Distinct C5 coordinates give within-cell ratios order five or ten, never two.")
SHAPES = ("A=(1,2), B=(1,3): component lengths 320 and 480 with 160 checks each; n=1120 and k>=320 give guaranteed rate 2/7=0.285714 in cell 0.300. The parent measured weight-five minima entirely in sector one, spread across Acol/Bcol blocks. Remove degree-two columns from both components rather than reinforce one witness block.")
FILLING = ("All cells are trinomials: A-row total six and B-row total nine give HX=6+3=9 and HZ=9+3=12. Twenty cards cross five quotient-coordinate frames with four central-mixing schedules. Generate noncommuting relative labels, shortlist verified component geometry, exclude component kernels through weight four, and compare independently sampled information-set kernel witnesses. Full verified lifts gate kernels through four, CSS, and actual rate in [0.275,0.325). None of the selection readings certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r030_affineF16C5xC2_12x13_trinomial_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 160

def custom_ops(q):
    # Affine maps t -> beta^v*t+u over F16, times central C2.
    # F16 = F2[alpha]/(alpha^4+alpha+1); beta=alpha^3 has order 5.
    if not valid_q(q):
        raise ValueError("this affine-field group requires order 160")
    powers = (1, 8, 12, 10, 15)
    els = [(u, v, z) for z in range(2)
           for v in range(5) for u in range(16)]

    def field_mul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 16:
                a ^= 19
        return out

    def red(g):
        return (g[0] & 15, g[1] % 5, g[2] & 1)

    def mul(g, h):
        return (g[0] ^ field_mul(powers[g[1]], h[0]),
                (g[1] + h[1]) % 5, g[2] ^ h[2])

    def inv(g):
        vi = (-g[1]) % 5
        return (field_mul(powers[vi], g[0]), vi, g[2])

    def idx(g):
        u, v, z = red(g)
        return (5*z + v)*16 + u

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _syndrome_columns(rows, n):
    """Columns of a verified bit-row matrix, as syndrome integers."""
    cols = [0] * n
    mask = (1 << n) - 1
    for r, row in enumerate(rows):
        x = row & mask
        while x:
            bit = x & -x
            cols[bit.bit_length() - 1] |= 1 << r
            x ^= bit
    return cols


def _no_kernel4(cols):
    """Exact exclusion of every nonzero kernel vector of weight <=4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    # A pair matching a singleton gives weight three; matching an
    # earlier pair gives weight four. Shared indices would imply
    # duplicate columns, already excluded above.
    seen = singles.copy()
    for i, a in enumerate(cols):
        for j in range(i):
            v = a ^ cols[j]
            if v in seen:
                return False
            seen.add(v)
    return True


# Complement-coordinate prescriptions, not automorphism-generated slots.
# Each card searches independent normal-coordinate fillings of its frame.
_FIELD_TRIPLES = (
    (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 3), (0, 2, 4), (0, 3, 4),
)
_FIELD_FRAMES = (
    ((0, 0), (0, 0, 0)),
    ((0, 0), (0, 1, 2)),
    ((0, 1), (0, 1, 3)),
    ((0, 2), (1, 3, 4)),
    ((0, 3), (2, 4, 5)),
)
# Numbers of cells containing both central coordinates.
_FIELD_MIXING = ((2, 3), (2, 2), (1, 3), (2, 1))


def _field_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _field_cells(rng, G, frames, mixed, phase):
    """Distinct multipliers and noncommuting within-cell relative labels."""
    cells = []
    for j, frame in enumerate(frames):
        vs = _FIELD_TRIPLES[frame]
        zs = [0, 0, 0]
        if j < mixed:
            zs[(phase + j) % 3] = 1
        for _ in range(128):
            cell = [(rng.randrange(16), v, z)
                    for v, z in zip(vs, zs)]
            first_inv = G.inv(cell[0])
            g = G.mul(first_inv, cell[1])
            h = G.mul(first_inv, cell[2])
            if G._key(G.mul(g, h)) == G._key(G.mul(h, g)):
                continue
            assert len({G._key(x) for x in cell}) == 3
            cells.append(cell)
            break
        else:
            raise ValueError("noncommuting trinomial generation failed")
    return [cells]


def _field_component(G, cells, side):
    """Extract only through the verified lift; dummy zeros are never submitted."""
    width = len(cells[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, cells, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], cells)
    rows = [row & mask for row in rows]
    return rows, _syndrome_columns(rows, width)


def _field_geometry(rows, columns):
    """Exact Tanner C4 and a secondary projection-triangle statistic."""
    if 0 in columns or len(set(columns)) != len(columns):
        return None
    four = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            common = (row & other).bit_count()
            four += common * (common - 1) // 2

    adjacency = [0] * len(rows)
    forced = 0
    for signature in columns:
        degree = signature.bit_count()
        forced += degree * (degree - 1) * (degree - 2) // 6
        for r in _field_bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    reached = frontier = 1
    while frontier:
        neighbors = 0
        for r in _field_bits(frontier):
            neighbors |= adjacency[r]
        frontier = neighbors & ~reached
        reached |= frontier
    if reached != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _field_bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v]
                          & (-1 << (v + 1))).bit_count()
    return four, max(0, triangles - forced)


def _field_kernel_reading(columns, rng, trials):
    """Sample genuine component-kernel words in independent information sets.

    Fundamental dependencies and their sums are witnesses only.
    A large reading is not evidence of a certified minimum.
    """
    width = len(columns)
    reading = width + 1
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
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
                reading = min(reading, word.bit_count())
                words.append(word)

        words.sort(key=int.bit_count)
        short = words[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                reading = min(reading, (word ^ other).bit_count())
        if len(words) >= 3:
            for _ in range(128):
                a, b, c = rng.sample(words, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
        if reading <= 6:
            break
    return reading


def _field_pool(rng, G, side, frames, mixed, phase):
    """Geometry shortlist followed by two independent witness-search budgets."""
    pool = []
    for trial in range(48):
        cells = _field_cells(rng, G, frames, mixed, phase)
        rows, columns = _field_component(G, cells, side)
        geometry = _field_geometry(rows, columns)
        if geometry is None:
            continue
        pool.append((geometry, trial, cells, columns))
        if len(pool) >= 32:
            break
    pool.sort(key=lambda entry: entry[:2])

    ranked = []
    for geometry, trial, cells, columns in pool:
        if not _no_kernel4(columns):
            continue
        reading = _field_kernel_reading(columns, rng, 8)
        merit = (reading, -geometry[0], -geometry[1])
        ranked.append((merit, trial, cells, columns, geometry))
        if len(ranked) >= 8:
            break
    if not ranked:
        raise ValueError("no connected trinomial component passed kernel-4")
    ranked.sort(key=lambda entry: entry[0], reverse=True)

    finalists = []
    for merit, trial, cells, columns, geometry in ranked[:4]:
        reading = min(merit[0],
                      _field_kernel_reading(columns, rng, 24))
        key = (reading, -geometry[0], -geometry[1])
        finalists.append((key, trial, cells))
    finalists.sort(key=lambda entry: entry[0], reverse=True)
    return finalists[:3]


def NEW_CTX(rng, G, idx):
    """Twenty quotient/mixing cards, optimized before returning any cells."""
    card = idx % 20
    af, bf = _FIELD_FRAMES[card % 5]
    am, bm = _FIELD_MIXING[card // 5]
    cohort = idx // 20
    phase_a = (card + cohort) % 3
    phase_b = (card // 5 + 2*cohort) % 3
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    for attempt in range(3):
        apool = _field_pool(rng, G, "A", af, am, phase_a)
        bpool = _field_pool(rng, G, "B", bf, bm, phase_b)
        proposals = []
        for ak, at, A in apool:
            for bk, bt, B in bpool:
                merit = (min(ak[0], bk[0]), ak[0] + bk[0],
                         ak[1] + bk[1], ak[2] + bk[2])
                proposals.append((merit, at, bt, A, B))
        proposals.sort(key=lambda entry: entry[0], reverse=True)

        for merit, at, bt, A, B in proposals:
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 1120
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= 320 and 275*n <= 1000*k < 325*n):
                continue
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            assert max(row.bit_count() for row in hx + hz) <= 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            return {"A": A, "B": B, "idx": idx, "k": k,
                    "card": card, "component_merit": merit}
    raise ValueError("no screened affine-field lift remained in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return fresh lists from the selected, fully screened filling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial, and exact kernel-4 exclusion suppress the order-81 parent's light logical family; all five measured codes instead returned weight five with hundreds of sector-1 minimum witnesses of both types.
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
