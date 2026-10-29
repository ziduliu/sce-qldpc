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
CONCEPT = ("On dihedral D13 of order 26 with A=(1,3), B=(2,7), screening distinct support/coset motifs for short component dependencies should produce five d30s>=6 codes in cell 0.450.")
GROUP = ("Keep the dihedral construction rule but use D13: its measured (1,3)x(2,7) bed reached d30s=5 and fitness 10.9. Order 26 gives n=598 with these shapes.")
SHAPES = ("A=(1,3), B=(2,7) give k>=26*(3-1)*(7-2)=260 and guaranteed rate 260/598=0.435 in cell 0.450. The one-row A and two-row B components have unequal lengths. The parent's lightest sector-1 witnesses used Acol/Bcol blocks, including the weight-four X witnesses on Acol0-1/Bcol1.")
FILLING = ("Cross five weight-legal A/B support motifs with four coset rules. Screen verified-lift component columns for short dependencies, then select near-bound k. This targets the parent's weight-four/six witnesses; every HX and HZ row sum is at most 12.")
FAMILY_NAME = "r045_D13_13x27_component_motifs"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 26

def custom_ops(q):
    # D13 = <x,y | x^13=y^2=1, yxy=x^-1>; (a,b) denotes x^a y^b.
    p = 13
    els = [(a, b) for b in range(2) for a in range(p)]

    def red(g):
        return (g[0] % p, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c)) % p, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a if b else -a) % p, b)

    def idx(g):
        a, b = red(g)
        return b * p + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _component_columns(hx, hz, q):
    """Read A and B component syndromes from the verified LP lifts."""
    mask = (1 << q) - 1
    ac = [0] * (3 * q)
    bc = [0] * (7 * q)
    for r in range(q):
        row = hx[r]             # Arow 0, Bcol 0
        for j in range(3):
            bits = (row >> (j * 7 * q)) & mask
            while bits:
                bit = bits & -bits
                ac[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
    for alpha in range(2):
        for r in range(q):
            row = hz[alpha * q + r]  # Acol 0, Brow alpha
            for j in range(7):
                bits = (row >> (j * q)) & mask
                while bits:
                    bit = bits & -bits
                    bc[j * q + bit.bit_length() - 1] |= 1 << (alpha * q + r)
                    bits ^= bit
    return ac, bc


def _short_cost(cols, rng):
    """Count dependencies through four and sample distinct weight-five words."""
    singles = set(cols)
    bad2 = len(cols) - len(singles) + 10 * (0 in singles)
    pairs = {}
    bad3 = bad4 = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            bad3 += value in singles
            bad4 += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
    bad5 = 0
    for _ in range(2200):
        i, j, k = rng.sample(range(len(cols)), 3)
        bad5 += (cols[i] ^ cols[j] ^ cols[k]) in pairs
    return bad2, bad3, bad4, bad5


def NEW_CTX(rng, G, idx):
    """Cross five support constructions with four coset constructions."""
    motif, mode = idx % 5, (idx // 5) % 4
    order = rng.sample(range(7), 7)
    A_counts = [3, 3, 3] if motif == 0 else [2, 3, 3]
    if motif != 0:
        rng.shuffle(A_counts)

    B_counts = [[1] * 7 for _ in range(2)]
    if motif == 0 or motif == 2:
        extra = (order[:2], order[2:4])
    elif motif == 1:
        extra = (order[:2], (order[0], order[2]))
    elif motif == 3:
        extra = (order[:1], order[1:3])
    else:
        extra = ((order[0], order[0]), order[1:3])
    for alpha in range(2):
        for j in extra[alpha]:
            B_counts[alpha][j] += 1

    # A-row sums are 9 or 8. B rows sum at most 9; B columns
    # sum at most 3 in motif 0 and at most 4 in all other motifs.
    assert _grid_weights([A_counts], B_counts) <= 12

    def a_cell(count, j):
        if count == 2:
            flag = (mode + j) % 2
            return [(a, flag) for a in rng.sample(range(13), 2)]
        rotations = (2, 2, 3, 1)[mode]
        if rotations == 3 and j == 2:
            rotations = 2
        return ([(a, 0) for a in rng.sample(range(13), rotations)] +
                [(a, 1) for a in rng.sample(range(13),
                                            3 - rotations)])

    def b_grid():
        axes = rng.sample(range(13), 7)
        rows = [[], []]
        for j in range(7):
            base = rng.randrange(13)
            flags = (rng.randrange(2), rng.randrange(2))
            for alpha in range(2):
                count = B_counts[alpha][j]
                flag = flags[alpha] if mode != 2 else flags[0]
                first = base if alpha == 0 else (base + axes[j]) % 13
                cell = [(first, flag)]
                if count > 1:
                    gap = rng.randrange(1, 13)
                    cell.append(((first + gap) % 13,
                                 flag ^ int(mode == 3)))
                if count > 2:
                    gap = rng.choice([d for d in range(1, 13)
                                      if (first + d) % 13 != cell[1][0]
                                      or cell[1][1] != flag])
                    cell.append(((first + gap) % 13, flag))
                rows[alpha].append(cell)
        return rows

    shortlist = []
    for _ in range(16):
        A = [[a_cell(count, j) for j, count in enumerate(A_counts)]]
        B = b_grid()
        hx, hz, _ = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        sa, sb = _short_cost(ac, rng), _short_cost(bc, rng)
        grade = (sa[0] + sb[0], sa[1] + sb[1],
                 sa[2] + sb[2], sa[3] + sb[3], rng.random())
        shortlist.append((grade, A, B, hx, hz))

    best = None
    for grade, A, B, hx, hz in sorted(shortlist,
                                      key=lambda item: item[0])[:5]:
        k = len(hx) * 0 + 598 - gf2_rank(hx) - gf2_rank(hz)
        key = (not 260 <= k < 284, grade, abs(k - 260))
        if best is None or key < best[0]:
            best = (key, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, without changing shared state."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled singleton-heavy cyclic-q21 fillings suffice for d30s>=8 at
# rate ~0.30: refuted in this bed by five measured distances of 6.
# REFUTED: Four screened labels per D42 slot suffice for five d30s>=8
# codes: the parent's measured distances were 8, 8, 4, 6 and 8.
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
