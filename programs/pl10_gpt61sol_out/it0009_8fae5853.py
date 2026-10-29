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
CONCEPT = ("For Dic_11 of order 44 and equal (2,3)x(2,3) shapes, balanced support profiles and exact-lift short-cycle screening should improve worse-basis circuit failure rates over the rolled D14 parent. Exact ranks keep every retained code below rate 0.125, targeting the empty 0.100 cell.")
GROUP = ("Dic_11, order 44, is not the parent's order-14 dihedral group. It has a central involution but no other involutions; mixed-coset differences have order four. Order 44 is the largest dicyclic order fitting n=13*q<600. The D28 (2,4)x(2,4) bed scored 5.766, motivating another nonabelian balanced bed, not proving a group-only advantage.")
SHAPES = ("A=B=(2,3) gives n=572 and guaranteed k>=44, rate >=1/13. Both lifted component matrices have 88 checks and 132 variables. Exact ranks require k/n<0.125. Parent witnesses were weight four entirely in Arow/Brow sector 2, with broadly weak 1.0-1.2 failure enrichment rather than one strongly localized block.")
FILLING = ("Slots select distinct canonical support-profile pairs: positive 1-3 term cells, row sums 5-7 and column sums 3-5, so both check-weight sums are <=12. Mixed-coset cells exclude central-involution differences. Candidate fillings use verified lifts and ranks, reject repeated syndrome columns, and minimize same-basis short-cycle overlaps. This tests whether better neighborhoods reduce incorrect settled decoding; it does not claim that the surrogate measures circuit pL or reduces total CNOTs.")
FAMILY_NAME = "r010_dic11_23x23_profile_overlap_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic_11: x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    # Canonical element (a,b) represents x^a*y^b, b in {0,1}.
    if not valid_q(q):
        raise ValueError("Dic_11 requires order 44")
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((a + 11 * (b // 2)) % 22, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11*b*d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return a + 22*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _support_pair(idx):
    """Canonicalize supports, not random labelings or orbit copies."""
    import itertools
    perms = list(itertools.permutations(range(3)))
    profiles = set()
    for f in itertools.product((1, 2, 3), repeat=6):
        rows = (f[:3], f[3:])
        rs = [sum(row) for row in rows]
        cs = [rows[0][j] + rows[1][j] for j in range(3)]
        if min(rs) < 5 or max(rs) > 7 or sum(rs) < 11:
            continue
        if min(cs) < 3 or max(cs) > 5:
            continue
        images = []
        for rr in (rows, rows[::-1]):
            for p in perms:
                images.append(tuple(rr[i][j] for i in range(2) for j in p))
        profiles.add(min(images))

    def margins(f):
        return ([sum(f[:3]), sum(f[3:])],
                [f[j] + f[j+3] for j in range(3)])

    def priority(pair):
        a, b = pair
        ar, ac = margins(a)
        br, bc = margins(b)
        deficit = sum(max(0, 4-c) for c in ac + bc)
        peak = max(max(ar) + max(bc), max(br) + max(ac))
        imbalance = abs(ar[0]-ar[1]) + abs(br[0]-br[1])
        spread = max(ac)-min(ac) + max(bc)-min(bc)
        return (deficit, peak, sum(a)+sum(b), imbalance, spread, a, b)

    bank = []
    ordered = sorted(profiles)
    for u, a in enumerate(ordered):
        for b in ordered[u:]:
            sa, sb = [list(a[:3]), list(a[3:])], [list(b[:3]), list(b[3:])]
            if _grid_weights(sa, sb) <= W_MAX:
                bank.append((a, b))
    bank.sort(key=priority)
    if idx >= len(bank):
        raise ValueError("more slots than deliberate support designs")
    a, b = bank[idx]
    return {"A": [list(a[:3]), list(a[3:])],
            "B": [list(b[:3]), list(b[3:])]}


def _cell_allowed(G, cell):
    keys = [G._key(g) for g in cell]
    if len(set(keys)) != len(keys):
        return False
    # Exclude the unique order-two relative displacement.
    for u, g in enumerate(cell):
        for h in cell[u+1:]:
            if G._key(G.mul(G.inv(g), h)) == (11, 0):
                return False
    return True


def _draw_design(rng, G, supports, pools, idx):
    cells = {}
    for side in ("A", "B"):
        grid = []
        for i in range(2):
            row = []
            for j in range(3):
                t = supports[side][i][j]
                phase = (idx + i + j + (side == "B")) % 2
                if t == 1:
                    counts = [1-phase, phase]
                else:
                    counts = [t//2, t-t//2]
                    if phase:
                        counts.reverse()
                while True:
                    cell = (rng.sample(pools[0], counts[0]) +
                            rng.sample(pools[1], counts[1]))
                    if _cell_allowed(G, cell):
                        break
                row.append(cell)
            grid.append(row)
        cells[side] = grid
    return cells


def _matrix_screen(G, cells):
    """Exact construction; returned cost is a heuristic, not fitness."""
    hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k <= 0 or 8*k >= n:
        return None

    cost = 0
    for rows in (hx, hz):
        columns = [0] * n
        for r, row in enumerate(rows):
            bits = row
            while bits:
                bit = bits & -bits
                columns[bit.bit_length()-1] |= 1 << r
                bits ^= bit
        # Conservatively reject all repeated syndrome columns,
        # without assuming whether their weight-two kernels are logical.
        if len(set(columns)) != n:
            return None
        for r, row in enumerate(rows):
            for other in rows[:r]:
                overlap = (row & other).bit_count()
                if overlap >= 2:
                    cost += overlap*(overlap-1)//2
                    cost += 6*max(0, overlap-2)**2
    # Prefer smaller rank deficits only when neighborhood costs tie.
    return (cost, k-G.q)


def NEW_CTX(rng, G, idx):
    """One deliberate support design, then screened label optimization."""
    supports = _support_pair(idx)
    pools = ([g for g in G.elements if g[1] == 0],
             [g for g in G.elements if g[1] == 1])
    best, best_cost = None, None
    for attempt in range(128):
        trial = _draw_design(rng, G, supports, pools, idx)
        value = _matrix_screen(G, trial)
        if value is not None and (best_cost is None or value < best_cost):
            best, best_cost = trial, value
        if attempt >= 31 and best is not None:
            break
    if best is None:
        raise ValueError("no filling passed exact rank/neighborhood screen")

    # Preserve this slot's term counts and coset allocation while
    # replacing actual labels; these are not automorphism moves.
    for step in range(80):
        side = ("A", "B")[rng.randrange(2)]
        i, j = rng.randrange(2), rng.randrange(3)
        old = best[side][i][j]
        pos = rng.randrange(len(old))
        replacement = pools[old[pos][1]][rng.randrange(22)]
        cell = list(old)
        cell[pos] = replacement
        if not _cell_allowed(G, cell):
            continue
        trial = {s: [[list(c) for c in row] for row in best[s]]
                 for s in ("A", "B")}
        trial[side][i][j] = cell
        value = _matrix_screen(G, trial)
        if value is not None and value < best_cost:
            best, best_cost = trial, value

    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    if not css_ok(hx, hz):
        raise ValueError("verified lift failed CSS commutation")
    return {"idx": idx, "cells": best}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-12 rolling alone removes the D14 (6,2)x(6,2) sector-2 bottleneck: parent measurements instead found weight-four logicals in both bases and worse-basis scores 3.30-3.59.
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
