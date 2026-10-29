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
CONCEPT = ("On C8 semidirect C4 (action 3), order 32, with "
           "(1,4)x(1,4) sides, full-syndrome screening of even-weight "
           "gap-order motifs should yield five inequivalent codes "
           "with k>=304 and d30s>=4.")
GROUP = ("Keep the frozen order-32 C8 semidirect C4 group. This bed "
         "produced measured k=304, d30s=4 codes, establishing a "
         "filling-only baseline.")
SHAPES = ("Keep equal (1,4)x(1,4) sides: their four-block components "
          "have equal length, n=544, and guaranteed k>=288, rate "
          "9/17 in the open-ended 0.550 cell. The parent's lightest "
          "X and Z logicals occupied sector-1 Acol/Bcol blocks, "
          "particularly Acol1-2 and Bcol1-3.")
FILLING = ("Use twenty indexed mixtures of order-eight and order-four "
           "binomial gaps, plus four-term profiles. All cells have "
           "even weight and both row sums are at most 12. Reject "
           "weight-one and weight-two kernels in both verified full "
           "matrices, then prefer larger verified k; use component "
           "weight-four counts only to break ties.")
FAMILY_NAME = "r055_c8sd3c4_q32_14x14_even_gap_fullrank_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # x^8=y^4=1, y*x*y^-1=x^3; 3^4=1 modulo 8.
    m, r, s = 8, 4, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return m * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m) * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m) * a) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _component_columns(rows, starts, q):
    """Transpose component blocks from the verified LP check matrices."""
    columns = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:q]):
        for block, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                low = bits & -bits
                columns[block * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return columns


def _all_columns(rows, n):
    columns = [0] * n
    for check, row in enumerate(rows):
        mark = 1 << check
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= mark
            row ^= low
    return columns


def _short_relations(columns, limit=None):
    """Count zero columns, repeated columns and pair-syndrome collisions."""
    bad = int(0 in columns) + len(columns) - len(set(columns))
    pairs = set()
    collisions = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            if syndrome in pairs:
                collisions += 1
                if limit is not None and collisions >= limit:
                    return bad, collisions
            pairs.add(syndrome)
    return bad, collisions


def _component_grade(columns):
    """Count component kernel relations of weights two, three and four."""
    singles = set(columns)
    bad2 = int(0 in singles) + len(columns) - len(singles)
    pairs = {}
    bad3 = bad4 = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            bad3 += syndrome in singles
            bad4 += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return bad2, bad3, bad4


def _design(rng, G, idx, trial):
    """Build a fresh instance of an indexed, even-weight gap motif."""
    motif, batch = idx % 20, (idx // 20) % 3
    eight = [(a, b) for b in (0, 2) for a in (1, 3, 5, 7)]
    odd_four = [(a, b) for b in (1, 3) for a in range(8)]
    even_four = [(a, b) for b in (0, 2) for a in (2, 6)]
    pools = {"eight": eight, "odd4": odd_four, "even4": even_four}
    grids = {}

    for side in ("A", "B"):
        digit = motif % 5 if side == "A" else motif // 5
        n_eight = (4 - digit) if batch != 1 else digit
        heavy = batch == 2 and ((motif % 2 == 1) == (side == "A"))
        counts = [2, 2, 2, 2]
        if heavy:
            counts[rng.randrange(4)] = 4
            n_eight = min(n_eight, 3)

        remaining = 4 - n_eight
        n_even_four = 0 if batch == 0 else min(
            2, remaining, (motif + int(side == "B") + trial % 2) % 3)
        kinds = (["eight"] * n_eight
                 + ["even4"] * n_even_four
                 + ["odd4"] * (remaining - n_even_four))
        rng.shuffle(kinds)

        used_gaps = set()
        row = []
        for j, (kind, count) in enumerate(zip(kinds, counts)):
            base = G.pick(rng)
            if count == 2:
                options = [
                    gap for gap in pools[kind]
                    if min(G.idx(gap), G.idx(G.inv(gap)))
                    not in used_gaps
                ]
                if not options:
                    raise ValueError("gap-order motif exhausted")
                gap = rng.choice(options)
                used_gaps.add(min(G.idx(gap), G.idx(G.inv(gap))))
                cell = [base, G.mul(base, gap)]
            elif (motif + trial) % 3 == 0:
                cell = rng.sample(G.elements, 4)
            else:
                # Unlike a relabeling, this changes the support topology.
                for _ in range(40):
                    h = rng.choice(eight + odd_four)
                    t = rng.choice(eight + odd_four)
                    cell = [base, G.mul(base, h), G.mul(base, t),
                            G.mul(G.mul(base, h), t)]
                    if len(set(cell)) == 4:
                        break
                else:
                    cell = rng.sample(G.elements, 4)
            row.append(cell)
        grids[side] = [row]
    return grids


def NEW_CTX(rng, G, idx):
    """Check exact short syndromes before rewarding verified dimension."""
    q = G.q
    best_key = best = None
    for trial in range(18):
        grids = _design(rng, G, idx, trial)
        hx, hz, n = lifted_matrices(G, grids["A"], grids["B"])

        # All constructed cells have even cardinality. Thus every
        # check-matrix column has even parity: once zero and repeated
        # syndromes are excluded, no kernel word has weight below four.
        cx, cz = _all_columns(hx, n), _all_columns(hz, n)
        bad = (int(0 in cx) + n - len(set(cx))
               + int(0 in cz) + n - len(set(cz)))
        if bad and best_key is not None and best_key[0] == 0:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        ac = _component_columns(hx, [4*j*q for j in range(4)], q)
        bc = _component_columns(hz, [j*q for j in range(4)], q)
        a, b = _component_grade(ac), _component_grade(bc)
        key = (bad, -k, a[0] + b[0],
               max(a[2], b[2]), a[2] + b[2], trial)
        if best_key is None or key < best_key:
            best_key, best = key, grids
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support counts and uniform labels on the order-58
# dihedral parent did not suppress sector-1 weight-four logicals:
# all five measured parent slots had d30s=4.
# REFUTED: Action-coset component screening on the order-55 parent
# raises its measured distance above eight: all five selected slots
# had d30s=8 despite distinct fillings.
# REFUTED: Odd-action-coset binomials and the parent's short-relation
# screen raise distance above four on the order-32 (1,4)x(1,4) bed:
# all five measured slots had d30s=4 and k=296.
# REFUTED: Adding factored four-term cells to the order-32 bed removes
# its sector-1 weight-four ceiling: all five measured slots still had
# d30s=4, including slots with k=304.
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
