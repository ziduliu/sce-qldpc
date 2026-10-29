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
CONCEPT = ("Falsifiable principle: for cyclic C23 of order 23 on A=(2,4), B=(3,5), regular 7/8-term row profiles selected by verified-lift screening will raise five-code fitness above 23.7774 while keeping the mean rate in the 0.150 cell.")
GROUP = ("Keep C23: the current n=598 program measures k=102 and d30s=11-12, while the C23 (2,4)x(3,5) bed in RUN STATE has reached fitness 25.0. The group is frozen this round.")
SHAPES = ("Keep (2,4)x(3,5): the guaranteed rate is 92/598 ~= 0.154, and q=23 gives n=598, the largest lift order allowed by this shape factor. The unequal sides give sector lengths 460 and 138. The parent's measured weight-12 Z logicals were reported across all Acol/Bcol blocks, with many touching Bcol2 and Bcol3.")
FILLING = ("Give each A row seven terms, with one 1-term cell and three 2-term cells; A column sums are 3 or 4. Give each B row eight terms, with three 2-term cells and two 1-term cells; B column sums are 4 or 5. Both check-row weights are at most 12. For each slot, screen 16 independently labeled candidates using the verified lifts: first by rate, k and 4-cycles, then by non-stabilizer logicals through weight 4 in both sectors.")
FAMILY_NAME = "r015_m1r23s1_24x35_regular78_logical4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # metacyclic Z_1 x| Z_23: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 1, 23, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search regular row profiles and select labels using verified lifts."""
    def supports(trial):
        # Each A row has sum 7; the distinct singleton columns give
        # column totals 3 or 4.
        p, q = [2] * 4, [2] * 4
        u = (idx + trial) % 4
        v = (idx + 2 * trial + 1) % 4
        if v == u:
            v = (v + 1) % 4
        p[u], q[v] = 1, 1
        sa = [p, q]

        # Every B row has sum 8. One column totals 4 and the others 5.
        low = (idx + trial) % 5
        stubs = [low]
        for j in range(5):
            if j != low:
                stubs.extend((j, j))
        inc = None
        for _ in range(100):
            order = list(stubs)
            rng.shuffle(order)
            rows = [order[0:3], order[3:6], order[6:9]]
            if all(len(set(row)) == 3 for row in rows):
                inc = rows
                break
        if inc is None:
            inc = [[low, (low + 1) % 5, (low + 2) % 5],
                   [(low + 1) % 5, (low + 3) % 5, (low + 4) % 5],
                   [(low + 2) % 5, (low + 3) % 5, (low + 4) % 5]]
        sb = [[1 + int(j in inc[a]) for j in range(5)]
              for a in range(3)]
        return sa, sb

    def draw_grid(sup):
        return [[rng.sample(G.elements, count) for count in row]
                for row in sup]

    def cycles4(rows):
        seen, total = {}, 0
        for row in rows:
            cols = []
            while row:
                bit = row & -row
                cols.append(bit.bit_length() - 1)
                row ^= bit
            for x in range(len(cols)):
                for y in range(x):
                    pair = (cols[y], cols[x])
                    total += seen.get(pair, 0)
                    seen[pair] = seen.get(pair, 0) + 1
        return total

    def logical_floor(checks, stabs, n):
        """Return the minimum non-stabilizer logical weight through 4, or 5."""
        basis = {}
        for row in stabs:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in basis:
                    x ^= basis[p]
                else:
                    basis[p] = x
                    break

        def remainder(x):
            while x:
                p = x.bit_length() - 1
                if p not in basis:
                    break
                x ^= basis[p]
            return x

        rem = [remainder(1 << i) for i in range(n)]
        synd = [0] * n
        for ri, row in enumerate(checks):
            bit = 1 << ri
            while row:
                low = row & -row
                synd[low.bit_length() - 1] |= bit
                row ^= low

        by_synd = {}
        for i, sig in enumerate(synd):
            if sig == 0 and rem[i]:
                return 1
            for j in by_synd.get(sig, ()):
                if rem[i] ^ rem[j]:
                    return 2
            by_synd.setdefault(sig, []).append(i)

        for i in range(n):
            for j in range(i + 1, n):
                target = synd[i] ^ synd[j]
                for k in by_synd.get(target, ()):
                    if k > j and rem[i] ^ rem[j] ^ rem[k]:
                        return 3

        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                sig = synd[i] ^ synd[j]
                residue = rem[i] ^ rem[j]
                old = pairs.get(sig)
                if old is not None:
                    for a, b, previous in old:
                        if a != i and a != j and b != i and b != j:
                            if previous ^ residue:
                                return 4
                    old.append((i, j, residue))
                else:
                    pairs[sig] = [(i, j, residue)]
        return 5

    pool = []
    for trial in range(16):
        sa, sb = supports(trial)
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            continue
        A, B = draw_grid(sa), draw_grid(sb)
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        c4 = cycles4(hx) + cycles4(hz)
        pool.append({"A": A, "B": B, "hx": hx, "hz": hz, "n": n,
                     "k": k, "c4": c4})

    pool.sort(key=lambda c: (int(92 <= c["k"] <= 104), -c["c4"],
                             -abs(c["k"] - 100), c["k"]), reverse=True)
    best = None
    for c in pool[:5]:
        dx = logical_floor(c["hz"], c["hx"], c["n"])
        dz = logical_floor(c["hx"], c["hz"], c["n"])
        d = min(dx, dz)
        key = (int(92 <= c["k"] <= 104), d,
               c["k"] * d * d / c["n"], dx + dz,
               -c["c4"], c["k"])
        if best is None or key > best[0]:
            best = (key, c["A"], c["B"])

    if best is None:
        raise ValueError("no positive-k legal candidate")
    return {"cells": {"A": best[1], "B": best[2]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the preselected full cell for this side and protograph position."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled supports with uniform labels produced sector-1 Z d30s=6 in four of five measured slots (4, 6, 12, 13); the balanced-profile, low-4-cycle filling is the new hypothesis.
# The balanced all-2-term A profile with 4-cycle-only selection measured d30s 11, 11, 11, 12, 12; test complementary A degrees and a 6-cycle tie-break.
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
