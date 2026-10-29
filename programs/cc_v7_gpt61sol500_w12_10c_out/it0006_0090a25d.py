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
CONCEPT = ("For cyclic Z17 (metacyclic Z17:Z1), order 17, shapes (2,5)x(2,6), balanced column degrees and difference-packed labels selected against short component codewords should beat the parent's five-code score 8.824; this is a measured-distance hypothesis, not a distance certificate.")
GROUP = ("Frozen L1 bed: Z17 already gives five measured n=578, k=204 codes with distance reading 5. Keep the group and test whether structured filling removes the observed weight-5 plateau.")
SHAPES = ("Frozen (2,5)x(2,6): n=34*17=578, guaranteed k=12*17=204 and rate 0.35294 in cell 0.350; q=17 is the largest admissible lift. Component checks are 34x85 and 34x102. All reported parental minimum logicals occupy sector-1 Acol/Bcol blocks, with no consistently isolated weak block.")
FILLING = ("Twenty support profiles: nine all-degree-three designs, nine balanced degree-two/four perturbations, and two asymmetric designs. Construct labels by cyclic difference packing, then extract component checks from verified lifted_matrices and screen codewords through weight six. Prefer no short witnesses, otherwise fewer weight-six witnesses; require verified k=204. CELL only returns the selected cells. Component screening does not certify quantum distance.")
FAMILY_NAME = "r035_m17r1s1_25x26_balanced_shortword_v1"

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
    """Construct a structural profile, then optimize its actual lifts."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = q * (ca - ra) * (cb - rb)

    # Entries are column term counts (top row, bottom row).
    # Each A profile has row totals 7/8; each B profile has 9/9.
    aprofiles = [
        [(2, 1), (2, 1), (1, 2), (1, 2), (1, 2)],
        [(3, 0), (2, 1), (1, 2), (1, 2), (1, 2)],
        [(3, 0), (0, 3), (2, 1), (1, 2), (1, 2)],
    ]
    bprofiles = [
        [(2, 1)] * 3 + [(1, 2)] * 3,
        [(3, 0), (2, 1)] + [(1, 2)] * 4,
        [(3, 0), (0, 3)] + [(2, 1)] * 2 + [(1, 2)] * 2,
    ]
    design = idx % 20
    if design < 9:
        acounts = aprofiles[design // 3]
        bcounts = bprofiles[design % 3]
    elif design < 18:
        acounts = aprofiles[(design - 9) // 3]
        t = (design - 9) % 3 + 1
        bcounts = ([(1, 1)] * t + [(2, 2)] * t
                   + [(2, 1)] * (3 - t) + [(1, 2)] * (3 - t))
    elif design == 18:
        acounts = aprofiles[0]
        bcounts = [(1, 1), (3, 1), (2, 1)] + [(1, 2)] * 3
    else:
        acounts = aprofiles[2]
        bcounts = [(1, 1), (4, 0)] + [(1, 2)] * 4

    sa = [[col[i] for col in acounts] for i in range(2)]
    sb = [[col[i] for col in bcounts] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def pack(counts):
        """Construct a cyclic pair packing, not relabel an existing code.

        A within-row check-pair orbit uses an unsigned difference;
        a cross-row orbit uses an oriented difference. Never reuse
        either orbit within this component.
        """
        width = len(counts)
        for restart in range(64):
            used = [set(), set()]
            cross_used = set()
            out = [None] * width
            order = list(range(width))
            rng.shuffle(order)
            order.sort(key=lambda j: -sum(counts[j]))
            complete = True
            for j in order:
                a, b = counts[j]
                chosen = None
                for attempt in range(256):
                    x = rng.sample(range(q), a)
                    y = rng.sample(range(q), b)
                    anchor = x[0] if x else y[0]
                    x = sorted((v - anchor) % q for v in x)
                    y = sorted((v - anchor) % q for v in y)
                    local = []
                    legal = True
                    for row, vals in enumerate((x, y)):
                        diffs = []
                        for u in range(len(vals)):
                            for v in range(u + 1, len(vals)):
                                d = (vals[v] - vals[u]) % q
                                diffs.append(min(d, q - d))
                        ds = set(diffs)
                        if len(ds) != len(diffs) or ds & used[row]:
                            legal = False
                            break
                        local.append(ds)
                    if not legal:
                        continue
                    cross = [(u - v) % q for u in x for v in y]
                    cs = set(cross)
                    if len(cs) != len(cross) or cs & cross_used:
                        continue
                    chosen = (x, y)
                    used[0].update(local[0])
                    used[1].update(local[1])
                    cross_used.update(cs)
                    break
                if chosen is None:
                    complete = False
                    break
                out[j] = chosen
            if complete:
                return [[[G.elements[v] for v in out[j][i]]
                         for j in range(width)] for i in range(2)]
        return None

    def syndrome_columns(rows, positions):
        """Read columns of a verified matrix; do not implement a lift."""
        result = []
        for pos in positions:
            syndrome = 0
            for r, row in enumerate(rows):
                if (row >> pos) & 1:
                    syndrome |= 1 << r
            result.append(syndrome)
        return result

    def short_words(columns):
        """Exact rejection of words <=5; count weight-six words.

        Return 7 when no word <=6 exists: this is a screening flag,
        not an assertion that the component distance equals seven.
        After excluding words <=4, colliding triples are disjoint,
        so each weight-six word contributes ten triple collisions.
        """
        low = {}
        for s in columns:
            if not s:
                return (1, 1)
            if s in low:
                return (2, 1)
            low[s] = 1
        length = len(columns)
        for i in range(length):
            x = columns[i]
            for j in range(i + 1, length):
                s = x ^ columns[j]
                if not s:
                    return (2, 1)
                previous = low.get(s, 0)
                if previous:
                    return (3 if previous == 1 else 4, 1)
                low[s] = 2

        triples = {}
        get = triples.get
        collisions = 0
        for i in range(length - 2):
            x = columns[i]
            for j in range(i + 1, length - 1):
                xy = x ^ columns[j]
                for h in range(j + 1, length):
                    s = xy ^ columns[h]
                    if not s:
                        return (3, 1)
                    if s in low:
                        # A singleton collision would imply weight four,
                        # already excluded by the pair enumeration.
                        return (5, 1)
                    previous = get(s, 0)
                    collisions += previous
                    triples[s] = previous + 1
        if collisions:
            return (6, collisions // 10)
        return (7, 0)

    best = None
    best_score = None
    accepted = 0
    # Beyond twenty slots, retain the structural design but enlarge its
    # independently seeded optimization search; do not emit orbit copies.
    quota = 28 + 4 * (idx // 20)
    for trial in range(3 * quota):
        A = pack(acounts)
        B = pack(bcounts)
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != kb:
            continue

        # HX at Bcol=0 restricts to the left-lifted A component.
        arows = [hx[(i * cb) * q + r]
                 for i in range(ra) for r in range(q)]
        apos = [j * cb * q + r for j in range(ca) for r in range(q)]
        # HZ at Acol=0 restricts to the right-lifted B component.
        brows = hz[:rb * q]
        bpos = list(range(cb * q))
        pa = short_words(syndrome_columns(arows, apos))
        if pa[0] <= 4:
            continue
        pb = short_words(syndrome_columns(brows, bpos))
        if pb[0] <= 4:
            continue

        score = (min(pa[0], pb[0]), pa[0] + pb[0],
                 -max(pa[1], pb[1]), -(pa[1] + pb[1]))
        if best_score is None or score > best_score:
            best_score = score
            best = (A, B)
        accepted += 1
        if accepted >= quota:
            break

    if best is None:
        raise ValueError("no rank-bound candidate passed short-word screening")
    A, B = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    assert n - gf2_rank(hx) - gf2_rank(hz) == kb
    return {"idx": idx, "cells": {"A": A, "B": B},
            "component_screen": best_score}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
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
