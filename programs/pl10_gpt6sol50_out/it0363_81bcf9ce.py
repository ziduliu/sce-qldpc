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
CONCEPT = ("In dihedral Z29:Z2 with inversion, order 58, A=(1,3) and B=(1,3), balanced mixed-coset voltage motifs screened for verified component cycles should reduce worse-basis circuit failures.")
GROUP = ("Keep the measured order-58 dihedral (1,3)x(1,3) bed fixed for a filling-only comparison; its best reported fitness is about 4.5.")
SHAPES = ("Keep equal (1,3)x(1,3) sides: n=580, guaranteed k>=232 and rate>=0.400, with two 174-variable single-row components. The parent's weight-eight witnesses occupied sector-1 Acol/Bcol blocks; failed-shot maps showed no persistent check-block hotspot.")
FILLING = ("Cross twenty indexed coset-and-gap motifs with dense mixed triples and legal 3/3/2 or 4/2/2 profiles. Screen actual lifted components for short relations and rooted Tanner cycles. Test for fewer than the parent's roughly 100 worse-basis failures within 1500-5750 shots; distance proxy is not fitness.")
FAMILY_NAME = "r040_m29r2s28_13x13_balanced_coset_cycle_screen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 58

def custom_ops(q):
    # Same metacyclic inversion rule: Z_29 x| Z_2.
    m, r, s = 29, 2, 28
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb); n = 10 * 58 = 580


def _component_columns(rows, q, stride):
    """Extract component columns from the exact HX/HZ lift."""
    columns = [0] * (3 * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:q]):
        for j in range(3):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                bit = bits & -bits
                columns[j * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit
    return columns


def _short_relations(columns):
    """Count component kernel words of weights two through four."""
    singles = set(columns)
    pairs = {}
    twos = sum(value == 0 for value in columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            twos += syndrome == 0
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    triples = sum(pairs.get(value, 0) for value in singles)
    fours = sum(v * (v - 1) // 2 for v in pairs.values())
    return twos, triples, fours


def _rooted_cycles(columns, q):
    """Count rooted component Tanner cycles of lengths four, six, eight."""
    checks = [0] * q
    var_checks = []
    for variable, syndrome in enumerate(columns):
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            check = bit.bit_length() - 1
            checks[check] |= 1 << variable
            neighbors.append(check)
            syndrome ^= bit
        var_checks.append(neighbors)

    root = checks[0]
    nearby = []
    fours = sixes = eights = 0
    for check in range(1, q):
        common = root & checks[check]
        if common:
            size = common.bit_count()
            fours += size * (size - 1) // 2
            nearby.append((check, common))
    for i, (u, ru) in enumerate(nearby):
        for v, rv in nearby[:i]:
            sixes += ((ru & ~rv).bit_count() *
                      (rv & ~ru).bit_count() *
                      (checks[u] & checks[v] & ~root).bit_count())

    paths = {}
    start = root
    while start:
        bit = start & -start
        x = bit.bit_length() - 1
        start ^= bit
        for u in var_checks[x]:
            if u == 0:
                continue
            second = checks[u] & ~(1 << x) & ~root
            while second:
                bit_y = second & -second
                y = bit_y.bit_length() - 1
                second ^= bit_y
                for v in var_checks[y]:
                    if v == 0 or v == u:
                        continue
                    previous = paths.setdefault(v, [])
                    for old_x, old_y, old_u in previous:
                        if (old_u != u and old_x != x and
                                len({x, y, old_x, old_y}) == 4):
                            eights += 1
                    previous.append((x, y, u))
    return fours, sixes, eights


def _root_check_fours(rows, q):
    """Four-cycle count at one representative of each full check block."""
    total = 0
    for block in range(3):
        root_index = block * q
        root = rows[root_index]
        for index, row in enumerate(rows):
            if index != root_index:
                shared = (root & row).bit_count()
                total += shared * (shared - 1) // 2
    return total


def _cell_motif(rng, count, gap, slope, majority):
    """Correlate rotation/reflection offsets within one distinct cell."""
    a = rng.randrange(29)
    direction = rng.choice((-1, 1))
    d = direction * gap
    if count == 2:
        # Same-coset binomials have a rotation of order 29 as their gap.
        b = rng.randrange(2)
        return [(a, b), ((a + d) % 29, b)]
    if count == 3:
        if majority == 0:
            return [(a, 0), ((a + d) % 29, 0),
                    ((a + slope * d) % 29, 1)]
        return [(a, 0), ((a + d) % 29, 1),
                ((a + slope * d) % 29, 1)]
    # Two rotations and two reflections, with no repeated offset
    # within either coset.
    return [(a, 0), ((a + d) % 29, 0),
            ((a + slope * d) % 29, 1),
            ((a + (slope + 1) * d) % 29, 1)]


def NEW_CTX(rng, G, idx):
    """Select an indexed motif by verified component and full-check tests."""
    design, tier = idx % 20, idx // 20
    mode, palette = divmod(design, 5)
    slope = (2, 3, 5, 8, 11)[palette]
    ac, bc = [3, 3, 3], [3, 3, 3]
    if tier == 1:
        ac[design % 3] = 2
        bc[(design // 3) % 3] = 2
    elif tier >= 2:
        ac = [2, 2, 2]
        bc = [2, 2, 2]
        ac[design % 3] = 4
        bc[(design // 3) % 3] = 4
    assert _grid_weights([ac], [bc]) <= W_MAX
    assert _no_dead_lines([ac], [bc])

    def draw(counts, side):
        cells = []
        for j, count in enumerate(counts):
            gap = rng.randrange(1, 15)
            # Four coset profiles and five normalized offset slopes
            # define twenty motifs; independent gaps distinguish slots.
            majority = ((mode == 1) if side == "A" else (mode == 0))
            if mode >= 2:
                majority = (mode + j + (side == "B")) % 2
            cell = _cell_motif(
                rng, count, gap,
                (slope + j + 2 * (side == "B")) % 27 + 2,
                majority)
            rng.shuffle(cell)
            cells.append(cell)
        return [cells]

    shortlist = []
    for _ in range(20):
        A, B = draw(ac, "A"), draw(bc, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        a_cols = _component_columns(hx, G.q, 3)
        b_cols = _component_columns(hz, G.q, 1)
        ar, br = _short_relations(a_cols), _short_relations(b_cols)
        short = (ar[0] + br[0], ar[1] + br[1],
                 max(ar[2], br[2]), ar[2] + br[2])
        full_fours = (_root_check_fours(hx, G.q) +
                      _root_check_fours(hz, G.q))
        shortlist.append((short, full_fours, rng.random(),
                          A, B, hx, hz, n, a_cols, b_cols))
    shortlist.sort(key=lambda item: item[:3])

    best_key = best = None
    for short, full_fours, tie, A, B, hx, hz, n, acols, bcols in shortlist[:7]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        ca = _rooted_cycles(acols, G.q)
        cb = _rooted_cycles(bcols, G.q)
        key = (not 0.375 <= k / n < 0.425, short[:2],
               max(ca[0], cb[0]), ca[0] + cb[0],
               max(ca[1], cb[1]), ca[1] + cb[1],
               max(ca[2], cb[2]), ca[2] + cb[2],
               short[2:], full_fours, abs(k - 232), tie)
        if best_key is None or key < best_key:
            best_key, best = key, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing terms."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Parent 6x2 rolled bed did not protect its sector-2 blocks: all five
# measured slots had weight-four logicals there and >=100 failures/basis
# within 250-1750 shots.
# REFUTED: Shrinking to (1,3)x(1,5) with screened mixed-count cells
# removes the dominant failures: all five measured slots still had
# sector-1 weight-two witnesses and 201-243 Z-basis failures per
# 250 shots, for five-code fitness 2.4912.
# REFUTED: Short-component-relation screening on the parent's
# order-34 (1,4)x(1,4) bed reliably clears its sector-1 failures:
# five-code fitness was 3.2276; measured short witnesses remained,
# and each worse basis hit the failure cap after 250 shots.
# REFUTED: Moving to order-28 (1,4)x(1,5) odd-voltage binomials
# improves circuit failures: fitness fell to 2.5800, all measured
# slots had sector-1 weight-two witnesses, and their worse bases
# failed in 248-250 of 250 scored shots.
# REFUTED: Paired rotation-gap spectra with one reflection per triple
# reliably suppress failures on this bed: fitness was 4.4389; measured
# slot 19 had 111 Z failures in 1500 shots, and slot 23 had 100 X
# failures in 5500 shots.
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
