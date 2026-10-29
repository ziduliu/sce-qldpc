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
CONCEPT = ("C13 semidirect C3 with action 3, order 39, shapes (2,3)x(2,3): balanced component column degrees plus rejection of logicals of weight <=4 should yield five measured d30s>4 codes in cell 0.100.")
GROUP = ("Untried odd-order nonabelian group, unlike the parent's order-58 dihedral group: there are no involutions, and 3^3=1 mod 13 validates the action. Whether this improves distance is experimental.")
SHAPES = ("Equal length-three, two-check component protographs give n=507 and guaranteed rate 1/13. Both single-row parent components are replaced because its measured weight-4 witnesses span sector-1 Acol/Bcol blocks. Exact lifted ranks keep accepted candidates below rate 0.125.")
FILLING = ("Twenty support/coset designs, not automorphism copies: optimize component four- and six-cycle counts using verified lifts, then screen exact k and all logical supports of weight <=4. Profiles have positive cells, column degree >=3, and both check-weight sums <=12.")
FAMILY_NAME = "r010_C13sdC3_23x23_cycle_smalllogical"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # x^13=y^3=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    els = [(a, b) for b in range(3) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    def mul(g, h):
        return ((g[0] + pow(3, g[1] % 3, 13)*h[0]) % 13,
                (g[1] + h[1]) % 3)

    def inv(g):
        b = (-g[1]) % 3
        return ((-pow(3, b, 13)*g[0]) % 13, b)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _cycle_counts(rows):
    """Four/six cycles of a binary component Tanner graph."""
    size = len(rows)
    overlap = [[0]*size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i):
            t = (rows[i] & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = t
            four += t*(t-1)//2
    six = 0
    for i in range(size):
        for j in range(i):
            u = overlap[i][j]
            if not u:
                continue
            common = rows[i] & rows[j]
            for k in range(j):
                v, w = overlap[j][k], overlap[i][k]
                if v and w:
                    t = (common & rows[k]).bit_count()
                    six += u*v*w - t*(u+v+w-2)
    return four, six


def _component_objective(G, A, B, design):
    # Extract components only from the scaffold's verified matrices.
    hx, hz, n = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q)-1
    ar = []
    for i in range(2):
        for r in range(q):
            row = hx[i*3*q+r]
            ar.append(sum(((row >> (j*3*q)) & mask) << (j*q)
                          for j in range(3)))
    br = [row & ((1 << (3*q))-1) for row in hz[:2*q]]
    a4, a6 = _cycle_counts(ar)
    b4, b6 = _cycle_counts(br)
    # Later slots also explore different A/B priorities and cycle tradeoffs.
    priority = design % 3
    wa, wb = ((1, 1), (2, 1), (1, 2))[priority]
    six_weight = (1, 2, 4)[(design//3) % 3]
    score = (1000*max(a4, b4) + 200*(wa*a4+wb*b4)
             + six_weight*(wa*a6+wb*b6))
    return score, hx, hz, n


def _row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]
    return basis


def _in_span(v, basis):
    while v:
        p = v.bit_length()-1
        if p not in basis:
            return False
        v ^= basis[p]
    return True


def _small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1, 2, 3 or 4."""
    basis = _row_basis(stabilizers)
    syndromes = [0]*n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndromes[low.bit_length()-1] |= bit
            row ^= low
    singles = {}
    for i, s in enumerate(syndromes):
        v = 1 << i
        if s == 0 and not _in_span(v, basis):
            return True
        if s in singles:
            if not _in_span(v ^ singles[s], basis):
                return True
        else:
            singles[s] = v
    pairs = {}
    for i in range(n):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            s = si ^ syndromes[j]
            v = vi | (1 << j)
            if s == 0 and not _in_span(v, basis):
                return True
            if s in singles and not _in_span(v ^ singles[s], basis):
                return True
            if s in pairs:
                if not _in_span(v ^ pairs[s], basis):
                    return True
            else:
                pairs[s] = v
    return False


def NEW_CTX(rng, G, idx):
    """Five support geometries times four quotient-label constructions."""
    p0 = ((3, 2, 2), (2, 3, 2))
    p1 = ((3, 3, 1), (2, 2, 2))
    p2 = ((3, 2, 2), (1, 3, 2))
    p3 = ((3, 3, 2), (1, 1, 2))
    p4 = ((2, 2, 2), (2, 2, 2))
    profiles = ((p0, p0), (p1, p0), (p2, p2),
                (p3, p4), (p4, p0))
    sa, sb = profiles[(idx % 20)//4]
    model = idx % 4
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def quotient_pattern(side, i, j, weight):
        offset = (i + 2*j + (side == "B")) % 3
        if weight == 1:
            return [offset]
        if model == 0:
            return [(offset+t) % 3 for t in range(weight)]
        if model == 1:
            return ([0, 1] if weight == 2 else [0, 0, 1])
        if model == 2:
            return ([1, 2] if weight == 2 else [0, 1, 2])
        if weight == 2:
            return [offset, (offset+1+(i+j) % 2) % 3]
        return [offset, offset, (offset+2) % 3]

    def draw_cell(side, i, j, weight):
        cell = []
        for b in quotient_pattern(side, i, j, weight):
            available = [(a, b) for a in range(13) if (a, b) not in cell]
            cell.append(available[rng.randrange(len(available))])
        return cell

    def draw_grid(side, supports):
        return [[draw_cell(side, i, j, weight)
                 for j, weight in enumerate(row)]
                for i, row in enumerate(supports)]

    pool = []
    for _ in range(6):
        A, B = draw_grid("A", sa), draw_grid("B", sb)
        score, hx, hz, n = _component_objective(G, A, B, idx)
        pool.append((score, A, B, hx, hz, n))
    pool.sort(key=lambda item: item[0])
    best = pool[0]

    # Mutations preserve cell counts and quotient quotas, not merely labels
    # under a common automorphism. Each slot optimizes its own design.
    for _ in range(40):
        _, A, B, _, _, _ = best
        A = [[list(cell) for cell in row] for row in A]
        B = [[list(cell) for cell in row] for row in B]
        grid = A if rng.randrange(2) == 0 else B
        i, j = rng.randrange(2), rng.randrange(3)
        cell = grid[i][j]
        t = rng.randrange(len(cell))
        b = cell[t][1]
        available = [(a, b) for a in range(13) if (a, b) not in cell]
        cell[t] = available[rng.randrange(len(available))]
        score, hx, hz, n = _component_objective(G, A, B, idx)
        candidate = (score, A, B, hx, hz, n)
        if score <= best[0]:
            best = candidate
            pool.append(candidate)

    pool.sort(key=lambda item: item[0])
    seen = set()
    tested = 0
    for score, A, B, hx, hz, n in pool:
        key = tuple(tuple(sorted(cell)) for grid in (A, B)
                    for row in grid for cell in row)
        if key in seen:
            continue
        seen.add(key)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 8*k < n):
            continue
        tested += 1
        if not _small_logical(hz, hx, n) and not _small_logical(hx, hz, n):
            assert css_ok(hx, hz)
            return {"A": A, "B": B}
        if tested >= 8:
            break
    # Do not submit a known small-logical code merely to populate a slot.
    raise ValueError("design found no rate-legal code without weight<=4 logicals")


def CELL(rng, G, side, i, j, ctx):
    """Return the already optimized full cell; no call-order dependence."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A large dihedral lift with legal rolled supports suffices for d30s>4: all five measured parent codes instead have d30s=4, with sector-1 witnesses.
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
