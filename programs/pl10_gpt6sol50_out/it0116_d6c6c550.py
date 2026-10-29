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
CONCEPT = ("In nonabelian C9 semidirect C4 of order 36, A=(2,3) and B=(2,4), order-nine binomial gaps and sparse B supports should reduce worst-basis circuit failures.")
GROUP = ("Keep the measured order-36 nonabelian bed for a filling-only comparison: its parent scored 6.9588, with Z-basis failures dominating its five measured slots.")
SHAPES = ("Keep (2,3)x(2,4): n=576, guaranteed k=72 and rate 0.125. Its unequal component lengths are 108 and 144. Parent Z failures enriched Acol/Bcol data, including Bcol1, and modestly enriched Xchk:Arow0/Bcol1 and Zchk:Acol2/Brow0.")
FILLING = ("Twenty indexed support-and-gap motifs keep A binomial and put one or two singleton cells in each B row. Distinct same-coset binomial elements have order-nine relative gaps. Screen independent fillings for component relations and short cycles, then verify k; test for fewer than the parent's 2-8 Z failures per 200000 shots without giving up k above the bound.")
FAMILY_NAME = "c9_semi_c4_23x24_order9_sparseB_cycle_screen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # x^9=y^4=1, y*x*y^-1=x^-1. In particular y has order four,
    # unlike the reflection generator of a dihedral group.
    els = [(a, b) for b in range(4) for a in range(9)]

    def red(g):
        return (int(g[0]) % 9, int(g[1]) % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 9, (b + d) % 4)

    def inv(g):
        a, b = red(g)
        return ((a if b & 1 else -a) % 9, (-b) % 4)

    def idx(g):
        a, b = red(g)
        return 9 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _support_profile(idx):
    """Five B degree motifs crossed with four binomial-gap domains."""
    motif, phase = idx % 5, idx // 5
    shift = phase % 4
    sa = [[2] * 3 for _ in range(2)]
    sb = [[2] * 4 for _ in range(2)]
    offsets = (
        ((0,), (2,)),
        ((0, 1), (2, 3)),
        ((0,), (1, 2)),
        ((0, 2), (1,)),
        ((0, 1), (1, 3)),
    )
    for i, positions in enumerate(offsets[motif]):
        for position in positions:
            sb[i][(shift + position) % 4] = 1
    return sa, sb, (phase % 4)


def _order_nine_cells(rng, counts, domain):
    """A binomial's two distinct elements differ by a generator of C9."""
    domains = ((1, 2, 4), (1, 2), (1, 4), (2, 4))
    out = []
    for row in counts:
        filled = []
        for count in row:
            a, b = rng.randrange(9), rng.randrange(4)
            cell = [(a, b)]
            if count == 2:
                gap = rng.choice(domains[domain])
                if rng.randrange(2):
                    gap = -gap
                cell.append(((a + gap) % 9, b))
            filled.append(cell)
        out.append(filled)
    return out


def _cosets_connected(grid):
    """Require row-to-row coset differences to generate C4."""
    differences = {
        (grid[0][j][0][1] - grid[1][j][0][1]) % 4
        for j in range(len(grid[0]))
    }
    return any(d & 1 for d in differences) and len(differences) >= 2


def _component_defects(G, grid, side):
    """Short kernel relations and four-cycles of an actual lift action.

    A variable h meets check g*h; B uses h*g^-1. These are the
    component actions used by lifted_matrices, not cyclic-shift proxies.
    """
    columns = []
    for row_column in range(len(grid[0])):
        for h in G.elements:
            syndrome = 0
            for i, row in enumerate(grid):
                for g in row[row_column]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)

    pairs = {}
    twos = fours = cycles = 0
    for u, first in enumerate(columns):
        for second in columns[u + 1:]:
            syndrome = first ^ second
            twos += syndrome == 0
            fours += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
            overlap = (first & second).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return twos, fours, cycles


def _component_sixes(G, grid, side):
    """Count six-cycle candidates through one check per lift orbit."""
    q = G.q
    checks = [0] * (len(grid) * q)
    for i, row in enumerate(grid):
        for r in G.elements:
            mask = 0
            for j, cell in enumerate(row):
                for g in cell:
                    h = (G.mul(G.inv(g), r) if side == "A"
                         else G.mul(r, g))
                    mask |= 1 << (j * q + G.idx(h))
            checks[i * q + G.idx(r)] = mask

    total = 0
    for i in range(len(grid)):
        root_index = i * q + G.idx(G.e)
        root = checks[root_index]
        nearby = [(v, root & mask) for v, mask in enumerate(checks)
                  if v != root_index and root & mask]
        for a in range(len(nearby)):
            u, ru = nearby[a]
            for b in range(a):
                v, rv = nearby[b]
                if ((ru | rv).bit_count() >= 2
                        and (checks[u] & checks[v] & ~root)):
                    total += 1
    return total


def NEW_CTX(rng, G, idx):
    """Screen fresh complete fillings of an indexed support-and-gap motif."""
    sa, sb, domain = _support_profile(idx)
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    shortlist = []
    for _ in range(30 + idx % 7):
        a = _order_nine_cells(rng, sa, domain)
        b = _order_nine_cells(rng, sb, (domain + 1) % 4)
        if not (_cosets_connected(a) and _cosets_connected(b)):
            continue
        da = _component_defects(G, a, "A")
        db = _component_defects(G, b, "B")
        key = (da[0] + db[0], max(da[2], db[2]),
               da[2] + db[2], 4 * da[1] + 6 * db[1],
               rng.random())
        shortlist.append((key, a, b))
    if not shortlist:
        raise ValueError("no connected-coset filling")
    shortlist.sort(key=lambda item: item[0])

    best_key, best = None, None
    for defect, a, b in shortlist[:8]:
        k = lifted_k(G, a, b)
        sixes = (_component_sixes(G, a, "A")
                 + _component_sixes(G, b, "B"))
        # n=576: k=73..100 is strictly inside the 0.150 rate cell.
        key = (not 73 <= k <= 100, max(0, 78 - k),
               defect[:3], sixes, defect[3], abs(k - 86))
        if best_key is None or key < best_key:
            best_key, best = key, {"A": a, "B": b}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return one complete, independently screened cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Uniform rolled D29 (1,3)x(1,3) did not protect the parent: its five
# measured slots had abundant weight-four logicals and at least one
# basis reached the 100-failure cap within 750 shots.
# Higher row weight does not always suppress X failures: the Z43
# parent's w12 slot 3 had two in 200000 shots; w11 slots 9 and 39 had zero.
# Component four-/six-cycle screening on Z31 (2,3)x(2,5) did not
# balance the bases: all five measured slots had zero X failures,
# while Z reached 100 failures in 66750-170250 shots.
# Unrestricted mixed-element binomials did not balance this C9:C4 bed:
# all five measured slots had 2-8 Z failures per 200000 shots, versus
# 0-1 X failures; the Z decoder also reached stage 3 in 7-31 shots.
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
