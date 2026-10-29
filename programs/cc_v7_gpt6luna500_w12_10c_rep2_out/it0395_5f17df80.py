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
CONCEPT = ("Falsifiable principle: for cyclic C17 of order 17 with A=(2,5) and B=(2,6), increasing the screened label proposals from 32 to 64 will raise five-code fitness above 14.6824 in the 0.350 rate cell.")
GROUP = ("Keep C17: the group and order are frozen, and the current bed's measured codes provide the baseline for testing only the filling.")
SHAPES = ("Keep (2,5)x(2,6): the rank-count bound is k>=17*(5-2)*(6-2)=204, or 204/578~=0.353; q=17 is the largest lift order allowed by the size window for these shapes. The component layouts have 5 and 6 column groups, so their nominal dimensions differ. The parent's found weight-6 logicals were broadly distributed across sector-1 Acol/Bcol blocks rather than concentrated in one block.")
FILLING = ("Keep the one/two-term support profile and existing rank, 4-cycle, weight-at-most-4, and 6-cycle screens, but generate 64 proposals per slot instead of 32. The intended change is a better chance of selecting labelings that avoid the recurring weight-6 witnesses; this screen does not guarantee a larger distance.")
FAMILY_NAME = "r035_m17r1s1_25x26_balanced12_screen64"

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
    """Search legal support/label proposals once per slot."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    def four_cycles(rows):
        seen = {}
        count = 0
        for row in rows:
            bits = []
            while row:
                bit = row & -row
                bits.append(bit.bit_length() - 1)
                row ^= bit
            for x in range(len(bits)):
                for y in range(x):
                    pair = (bits[y], bits[x])
                    count += seen.get(pair, 0)
                    seen[pair] = seen.get(pair, 0) + 1
        return count

    def six_cycles(rows, n):
        var_checks = [[] for _ in range(n)]
        for c, row in enumerate(rows):
            while row:
                bit = row & -row
                var_checks[bit.bit_length() - 1].append(c)
                row ^= bit
        shared = {}
        adj = [set() for _ in rows]
        for v, checks in enumerate(var_checks):
            for x, a in enumerate(checks):
                for b in checks[:x]:
                    shared.setdefault((b, a), []).append(v)
                    adj[a].add(b)
                    adj[b].add(a)
        count = 0
        for a in range(len(rows)):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[a].intersection(adj[b]):
                    if c <= b:
                        continue
                    xs = shared[(a, b)]
                    ys = shared[(b, c)]
                    zs = shared[(a, c)]
                    count += sum(1 for x in xs for y in ys for z in zs
                                 if x != y and x != z and y != z)
        return count

    def has_small_logical(checks, stabilizers, n):
        col = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                col[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        basis = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in basis:
                    row ^= basis[p]
                else:
                    basis[p] = row
                    break

        def in_stabilizer_span(mask):
            while mask:
                p = mask.bit_length() - 1
                if p not in basis:
                    return False
                mask ^= basis[p]
            return True

        by_syndrome = {}
        for i, s in enumerate(col):
            by_syndrome.setdefault(s, []).append(i)

        for i, s in enumerate(col):
            if s == 0 and not in_stabilizer_span(1 << i):
                return True
        for i in range(n):
            for j in range(i + 1, n):
                if col[i] == col[j] and not in_stabilizer_span((1 << i) | (1 << j)):
                    return True
        for i in range(n):
            for j in range(i + 1, n):
                target = col[i] ^ col[j]
                for k in by_syndrome.get(target, ()):
                    if k > j and not in_stabilizer_span((1 << i) | (1 << j) | (1 << k)):
                        return True

        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                sig = col[i] ^ col[j]
                old = pairs.get(sig)
                if old is not None:
                    for a, b in old:
                        if a != i and a != j and b != i and b != j:
                            mask = (1 << a) | (1 << b) | (1 << i) | (1 << j)
                            if not in_stabilizer_span(mask):
                                return True
                    old.append((i, j))
                else:
                    pairs[sig] = [(i, j)]
        return False

    def in_rate_band(c):
        return 8 * c["k"] < 3 * c["n"]

    pool = []
    for _ in range(64):
        # A: one doubled cell per column, with 2/3 doubles in its rows.
        da = [0, 0, 1, 1, 1]
        rng.shuffle(da)
        sa = [[1 + int(da[j] == i) for j in range(ca)] for i in range(ra)]

        # B: one doubled cell per column, with three doubles per row.
        db = [0, 0, 0, 1, 1, 1]
        rng.shuffle(db)
        sb = [[1 + int(db[j] == a) for j in range(cb)] for a in range(rb)]

        # These balanced profiles obey the frozen bed's row-weight cap.

        A = [[rng.sample(G.elements, sa[i][j]) for j in range(ca)]
             for i in range(ra)]
        B = [[rng.sample(G.elements, sb[i][j]) for j in range(cb)]
             for i in range(rb)]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        total = sum(map(sum, sa)) + sum(map(sum, sb))
        pool.append({
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "total": total,
            "c4": four_cycles(hx) + four_cycles(hz),
            "tie": rng.random()
        })

    pool.sort(key=lambda c: (int(in_rate_band(c)), c["k"], -c["c4"],
                             -abs(c["total"] - 33), c["tie"]), reverse=True)
    finalists = pool[:10]
    for c in finalists:
        c["small"] = (has_small_logical(c["hz"], c["hx"], c["n"])
                      or has_small_logical(c["hx"], c["hz"], c["n"]))

    finalists.sort(key=lambda c: (int(in_rate_band(c)), int(not c["small"]),
                                  c["k"], -abs(c["total"] - 33), -c["c4"],
                                  c["tie"]), reverse=True)
    finalists = finalists[:4]
    for c in finalists:
        c["c6"] = six_cycles(c["hx"], c["n"]) + six_cycles(c["hz"], c["n"])
    best = max(finalists, key=lambda c: (int(in_rate_band(c)),
                                         int(not c["small"]), c["k"],
                                         -c["c4"], -c["c6"],
                                         -abs(c["total"] - 33), c["tie"]))
    return {"A": best["A"], "B": best["B"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the full cell selected by the slot-level screen."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# The 32-proposal filling produced measured d30s readings 6, 6, 6, 6, and 8; it did not reliably avoid the recurring weight-6 witnesses.
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
