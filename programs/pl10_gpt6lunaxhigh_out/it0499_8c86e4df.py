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
CONCEPT = ("Falsifiable: for Q8xC4 of order 32 on shapes A=(1,4), B=(1,4), retaining the sparse, low-weight-screened filling will produce a five-code measured mean above the 0.550-cell frontier best of 3.4.")
GROUP = ("Replace the parent's cyclic C33 with the nonabelian group Q8xC4, of order 32. The current C33 program scores 3.2191, below the 0.550-cell frontier best of 3.4. The quaternion factor changes the multiplication law, while q=32 gives n=544 within the size window.")
SHAPES = ("Keep (1,4)x(1,4): at q=32, n=544 and the guaranteed rate is 9/17 ~= 0.5294, placing every valid code in the 0.550 cell. Equal shapes give equal-length component structures of length 128, each with 32 checks and nominal dimension 96. The parent's measured failures had mild, varying enrichments across Acol/Bcol blocks, with no consistent hotspot to motivate a shape change.")
FILLING = ("Keep the existing all-nonzero 1/2-term supports and exact-lift screening through weight 3. The parent's measured codes had weight-3 logical witnesses and scores around 3.1-3.4; their failure maps do not identify a single block to target. This group-only test asks whether the new regular action improves measured circuit performance. Each side's row total is at most 8 and each opposing cell at most 2 terms, so both row-weight sums are at most 10.")
FAMILY_NAME = "q8xc4_q32_14x14_sparse_d3screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # Q8 x C4. The first two coordinates encode i^a j^b in Q8,
    # where i^4=1, j^2=i^2, and j*i*j^-1=i^-1.
    els = [(a, b, c) for c in range(4)
           for b in range(2) for a in range(4)]

    def red(g):
        return (int(g[0]) % 4, int(g[1]) % 2, int(g[2]) % 4)

    def idx(g):
        a, b, c = red(g)
        return c * 8 + b * 4 + a

    def mul(g, h):
        a, b, c = red(g)
        u, v, w = red(h)
        return ((a + (-1 if b else 1) * u + 2 * b * v) % 4,
                (b + v) % 2, (c + w) % 4)

    def inv(g):
        a, b, c = red(g)
        ai = (-a) % 4 if b == 0 else (a + 2) % 4
        return (ai, b, (-c) % 4)

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Select a sparse, slot-stratified filling by exact lifted screens."""
    # Each profile gives the A-row and B-row term totals. Adjacent
    # profiles let the exact screen trade logical weight against gate count.
    profiles = ((6, 6), (6, 7), (7, 6), (6, 8), (7, 7),
                (8, 6), (7, 8), (8, 7), (8, 8))
    base = idx % len(profiles)
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    def supports(total):
        twos = set(rng.sample(range(4), total - 4))
        return [[2 if j in twos else 1 for j in range(4)]]

    def cells(s):
        return [[rng.sample(G.elements, count) for count in row]
                for row in s]

    def four_cycles(rows):
        total = 0
        for i, x in enumerate(rows):
            for y in rows[:i]:
                common = _popcount(x & y)
                total += common * (common - 1) // 2
        return total

    def low_profile(checks, stabilizers, n):
        # Find the minimum non-stabilizer codeword weight through 3.
        pivots = {}
        for row in stabilizers:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in pivots:
                    x ^= pivots[p]
                else:
                    pivots[p] = x
                    break

        def outside_span(x):
            while x:
                p = x.bit_length() - 1
                if p not in pivots:
                    return True
                x ^= pivots[p]
            return False

        signatures = [0] * n
        for ri, row in enumerate(checks):
            x = row
            marker = 1 << ri
            while x:
                bit = x & -x
                signatures[bit.bit_length() - 1] |= marker
                x ^= bit

        by_signature = {}
        for col, sig in enumerate(signatures):
            by_signature.setdefault(sig, []).append(col)

        cap = 64
        w1 = 0
        for col, sig in enumerate(signatures):
            if sig == 0 and outside_span(1 << col):
                w1 += 1
                if w1 >= cap:
                    return 1, cap
        if w1:
            return 1, w1

        w2 = 0
        for cols in by_signature.values():
            for u in range(len(cols)):
                for v in range(u + 1, len(cols)):
                    if outside_span((1 << cols[u]) | (1 << cols[v])):
                        w2 += 1
                        if w2 >= cap:
                            return 2, cap
        if w2:
            return 2, w2

        w3 = 0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                target = signatures[i] ^ signatures[j]
                for k in by_signature.get(target, ()):
                    if k > j and outside_span((1 << i) | (1 << j) | (1 << k)):
                        w3 += 1
                        if w3 >= cap:
                            return 3, cap
        if w3:
            return 3, w3
        return 4, 0

    best_key = None
    best_cells = None
    seen = set()
    for trial in range(12):
        profile_idx = max(0, min(len(profiles) - 1,
                                 base + (trial % 3) - 1))
        total_a, total_b = profiles[profile_idx]
        sa, sb = supports(total_a), supports(total_b)
        if _grid_weights(sa, sb) > W_MAX:
            continue

        A, B = cells(sa), cells(sb)
        sig = tuple(tuple(sorted(G.idx(g) for g in cell))
                    for grid in (A, B) for row in grid for cell in row)
        if sig in seen:
            continue
        seen.add(sig)

        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue

        dx, cx = low_profile(hz, hx, n)
        dz, cz = low_profile(hx, hz, n)
        terms = total_a + total_b
        degrees = [sa[0][j] + sb[0][b]
                   for j in range(ca) for b in range(cb)]
        degree_sum = sum(degrees)
        balance = sum((len(degrees) * x - degree_sum) ** 2
                      for x in degrees)
        cycles = four_cycles(hx) + four_cycles(hz)
        key = (min(dx, dz), dx + dz, -max(cx, cz), -(cx + cz),
               -terms, -cycles, -balance)
        if best_key is None or key > best_key:
            best_key = key
            best_cells = {"A": A, "B": B}

    if best_cells is None:
        raise ValueError("no valid filling candidate")
    labels = {
        side: [[[G.idx(g) for g in cell] for cell in row]
               for row in best_cells[side]]
        for side in ("A", "B")
    }
    return {"idx": idx, "labels": labels}


def CELL(rng, G, side, i, j, ctx):
    """Materialize the preselected cell from its group-element indices."""
    return [G.elements[t] for t in ctx["labels"][side][i][j]]

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# The previous 15-17-term, 4-cycle-first filling did not prevent weight-3 logical witnesses in the five measured codes; their measured scores were 3.06-3.34 with diffuse failure maps.
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
