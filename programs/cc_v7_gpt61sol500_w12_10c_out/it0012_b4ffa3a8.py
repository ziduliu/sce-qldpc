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
CONCEPT = ("On cyclic Z14, encoded as metacyclic Z1:Z14 with s=1, q=14, shapes (2,5)x(3,7), balanced column degrees and component-kernel screening should improve the parent's measured distance-6 plateau.")
GROUP = ("Keep the parent's custom metacyclic rule and trivial action. Resize only its cyclic order to 14: the new size factor is 41, so q=21 would give n=861, while the largest fitting order gives n=574.")
SHAPES = ("A=(2,5), B=(3,7) guarantee k>=168 and rate 12/41=0.292683. Component lengths/check counts are 70/28 and 98/42. Parent minimum witnesses were entirely in sector 1, often involving Acol1/Acol2 but spread across B columns; use three B rows and component columns of degree at least three.")
FILLING = ("Twenty design cards cross four balanced support budgets with zero through four even-step binomials. Build B's monomial scaffold with distinct cyclic row differences, then add controlled binomials. Select among 48 fillings per card using component witnesses through weight six, four-cycle counts, and exact lifted rank; retain only rates in [0.275,0.325). This screening does not certify quantum distance.")
FAMILY_NAME = "r030_m1r14s1_25x37_balanced_kernel6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # Same metacyclic rule and trivial action; order resized for n=574.
    m, r, s = 1, 14, 1
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
B_SHAPE = (3, 7)               # (rb, cb)


def _project_columns(rows, positions):
    """Column syndromes projected from the verified lifted matrices."""
    return [sum(((row >> p) & 1) << r
                for r, row in enumerate(rows)) for p in positions]


def _kernel_through_six(cols):
    """Exact existence test through weight six; 7 means none found.
    The collision count is only a tie-breaker, not a witness census."""
    seen = {0: 0}
    best, hits = 7, 0
    length = len(cols)
    for i, c in enumerate(cols):
        mask = 1 << i
        old = seen.get(c)
        if old is not None:
            w = (old ^ mask).bit_count()
            if w < best:
                best, hits = w, 1
            elif w == best:
                hits += 1
        else:
            seen[c] = mask
    for i in range(length):
        ci, mi = cols[i], 1 << i
        for j in range(i + 1, length):
            syndrome = ci ^ cols[j]
            mask = mi | (1 << j)
            old = seen.get(syndrome)
            if old is not None:
                w = (old ^ mask).bit_count()
                if w < best:
                    best, hits = w, 1
                elif w == best:
                    hits += 1
            else:
                seen[syndrome] = mask
    if best <= 4:
        return best, hits
    for i in range(length):
        ci, mi = cols[i], 1 << i
        for j in range(i + 1, length):
            cij, mij = ci ^ cols[j], mi | (1 << j)
            for t in range(j + 1, length):
                syndrome = cij ^ cols[t]
                mask = mij | (1 << t)
                old = seen.get(syndrome)
                if old is not None:
                    w = (old ^ mask).bit_count()
                    if w <= 5:
                        # Singles/pairs already excluded every weight <=4.
                        return 5, 1
                    if w == 6:
                        best, hits = 6, hits + 1
                else:
                    seen[syndrome] = mask
    return best, hits


def _four_cycles(cols):
    total = 0
    for i, c in enumerate(cols):
        for d in cols[:i]:
            common = (c & d).bit_count()
            total += common * (common - 1) // 2
    return total


def _monomial_B(rng, G):
    """Cyclic difference design, not a separate hand-written lift."""
    q = G.q
    for _ in range(100):
        us, vs, ds = set(), set(), set()
        pairs = []
        for j in range(B_SHAPE[1]):
            choices = [(u, v) for u in range(q) for v in range(q)
                       if u not in us and v not in vs
                       and (u - v) % q not in ds]
            if not choices:
                break
            u, v = choices[rng.randrange(len(choices))]
            pairs.append((u, v))
            us.add(u)
            vs.add(v)
            ds.add((u - v) % q)
        if len(pairs) == B_SHAPE[1]:
            return [[[G.e] for _ in pairs],
                    [[G.elements[u]] for u, v in pairs],
                    [[G.elements[v]] for u, v in pairs]]
    raise ValueError("could not complete cyclic difference design")


def NEW_CTX(rng, G, idx):
    """Four support budgets x five binomial-cycle profiles, then screening."""
    style = idx % 4
    even_count = (idx // 4) % 5
    templates = (
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((2, 2, 1, 1, 2), (2, 1, 2, 2, 1)),
        ((2, 2, 2, 1, 1), (2, 2, 1, 2, 2)),
        ((2, 2, 2, 2, 1), (2, 2, 2, 1, 2)),
    )
    sa = [list(row) for row in templates[style]]
    sb = [[1] * B_SHAPE[1] for _ in range(B_SHAPE[0])]
    if style == 0:
        for i in range(3):
            sb[i][2 * i] = sb[i][2 * i + 1] = 2
    elif style == 1:
        for i in range(3):
            sb[i][2 * i] = 2

    # HX maxima: 12,12,12,12; HZ maxima: 12,12,11,11.
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    extra = [(side, i, j)
             for side, grid in (("A", sa), ("B", sb))
             for i, row in enumerate(grid)
             for j, count in enumerate(row) if count == 2]
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    apos = [j * cb * q + t for j in range(ca) for t in range(q)]
    bpos = list(range(cb * q))
    best_score, selected = None, None

    for attempt in range(48):
        A = [[[G.pick(rng)] for j in range(ca)] for i in range(ra)]
        B = _monomial_B(rng, G)
        positions = list(extra)
        rng.shuffle(positions)
        even_positions = set(positions[:even_count])
        for side, i, j in extra:
            # Order-14 differences give length-28 cell cycles;
            # order-7 differences give length-14 cell cycles.
            pool = (2, 4, 6, 8, 10, 12) if (
                side, i, j) in even_positions else (1, 3, 5, 9, 11, 13)
            delta = pool[rng.randrange(len(pool))]
            grid = A if side == "A" else B
            grid[i][j].append(G.mul(grid[i][j][0], G.elements[delta]))

        hx, hz, n = lifted_matrices(G, A, B)
        arows = [hx[(i * cb) * q + t]
                 for i in range(ra) for t in range(q)]
        acols = _project_columns(arows, apos)
        bcols = _project_columns(hz[:rb * q], bpos)
        da, ha = _kernel_through_six(acols)
        db, hb = _kernel_through_six(bcols)
        coarse = (min(da, db), da + db, -ha - hb)
        if best_score is not None and coarse < best_score[:3]:
            continue
        score = coarse + (-_four_cycles(acols) - _four_cycles(bcols),)
        if best_score is not None and score <= best_score:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (2750 * n <= 10000 * k < 3250 * n):
            continue
        best_score = score
        selected = (A, B, hx, hz, k)

    if selected is None:
        raise ValueError("no candidate with exact rank in target cell")
    A, B, hx, hz, k = selected
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": best_score, "k": k}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
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
