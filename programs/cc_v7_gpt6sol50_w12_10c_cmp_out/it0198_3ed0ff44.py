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
CONCEPT = ("On C3 x SD16, order 48, with A=(1,2) and B=(2,5), independently screened long-order label motifs should produce more distance-12 codes at rate 0.250.")
GROUP = ("C3 x SD16 has centre of order six, unlike the parent's GL(2,3), whose centre has order two. Order-48 (1,2)x(2,5) beds have reached fitness 37.4, whereas the parent's measured distances were 8, 9 and 10.")
SHAPES = ("Keep (1,2)x(2,5): n=576, guaranteed k=144 and rate 0.250. The one-row A and two-row B components have 96 and 240 variables. The parent's lightest logicals occupied sector-1 Acol/Bcol blocks, so screen both components.")
FILLING = ("Use four terms per A cell and eight per B row, giving HX and HZ weights at most 12. Cross four B support profiles with five order-24/-12/-8/-6 difference motifs. Screen independently labelled verified lifts for short component relations, then check k above 144.")
FAMILY_NAME = "r025_c3xsd16_12x25_order_mix_five_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # C3 x SD16: x^8=y^2=z^3=1, y*x*y^-1=x^3;
    # z commutes with x and y. SD16 has five involutions,
    # unlike quaternion Q16 or dihedral D16.
    els = [(a, b, c) for c in range(3)
           for b in range(2) for a in range(8)]

    def red(g):
        return (g[0] % 8, g[1] % 2, g[2] % 3)

    def idx(g):
        a, b, c = red(g)
        return 16*c + 8*b + a

    def mul(g, h):
        a, b, c = red(g)
        u, v, t = red(h)
        return ((a + pow(3, b, 8)*u) % 8,
                (b + v) % 2, (c + t) % 3)

    def inv(g):
        a, b, c = red(g)
        return ((-pow(3, b, 8)*a) % 8, b, (-c) % 3)

    return dict(elements=els, ident=(0, 0, 0), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _component_obstructions(rows, starts, q, rng=None):
    """Short column relations from an exact verified component lift."""
    mask = (1 << q) - 1
    cols = [0] * (len(starts) * q)
    for r, row in enumerate(rows):
        for block, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                cols[block*q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit

    singles = set(cols)
    bad2 = int(0 in singles) + len(cols) - len(singles)
    bad3 = bad4 = close = 0
    pairs = {}
    for i, left in enumerate(cols):
        for j in range(i):
            syndrome = left ^ cols[j]
            bad3 += syndrome in singles
            bad4 += syndrome in pairs
            if syndrome.bit_count() <= 4:
                close += 1
            if syndrome not in pairs:
                pairs[syndrome] = (j, i)

    if rng is None or bad2 or bad3 or bad4:
        return bad2, bad3, bad4, 0, 0, close

    # Any odd-weight zero-syndrome word has an even-degree column.
    # Component translations allow that column to be rooted at the
    # identity in its protograph block.
    roots = [block*q for block in range(len(starts))
             if cols[block*q].bit_count() % 2 == 0]
    five = 0
    for root in roots:
        pivot = cols[root]
        for i in range(len(cols)):
            if i == root:
                continue
            for j in range(i):
                if j == root:
                    continue
                pair = pairs.get(pivot ^ cols[i] ^ cols[j])
                if pair is not None and pair[0] not in (root, i, j) \
                        and pair[1] not in (root, i, j):
                    five += 1
                    if five >= 32:
                        break
            if five >= 32:
                break
        if five >= 32:
            break

    # A sampled rooted-six probe breaks ties between the finalists.
    six = 0
    if not five:
        for root in range(0, len(cols), q):
            for _ in range(1200):
                i, j, k = rng.sample(range(len(cols)), 3)
                if root in (i, j, k):
                    continue
                pair = pairs.get(cols[root] ^ cols[i] ^ cols[j] ^ cols[k])
                if pair is not None and all(
                        t not in (root, i, j, k) for t in pair):
                    six += 1
                    if six >= 4:
                        break
            if six >= 4:
                break
    return bad2, bad3, bad4, five, six, close


def _b_support(rng, mode):
    """Four row-eight motifs; every B column totals at most four."""
    profiles = (
        ([2, 2, 2, 1, 1], [1, 1, 2, 2, 2]),
        ([3, 2, 1, 1, 1], [1, 1, 2, 2, 2]),
        ([2, 2, 2, 1, 1], [2, 2, 1, 2, 1]),
        ([3, 2, 1, 1, 1], [1, 1, 1, 2, 3]),
    )
    order = rng.sample(range(5), 5)
    rows = [[row[j] for j in order] for row in profiles[mode]]
    if rng.randrange(2):
        rows.reverse()
    return rows


def _a_cell(rng, G, gap24, occupancy):
    """Four distinct terms; vary 2+2 and 3+1 action-coset motifs."""
    first = (rng.randrange(8), 0, rng.randrange(3))
    cell = [first, G.mul(first, rng.choice(gap24))]
    if occupancy == 3:
        cell.append(rng.choice([g for g in G.elements
                                if g[1] == 0 and g not in cell]))
    second = (rng.randrange(8), 1, rng.randrange(3))
    cell.append(second)
    if occupancy == 2:
        cell.append(G.mul(second, rng.choice(gap24)))
    return cell


def NEW_CTX(rng, G, idx):
    """Screen independent fillings across twenty support/order motifs."""
    q = G.q
    mode, support_mode = idx % 5, (idx // 5) % 4
    odd = (1, 3, 5, 7)
    gap24 = [(a, 0, c) for a in odd for c in (1, 2)]
    gap12 = [(a, 1, c) for a in odd for c in (1, 2)]
    gap8 = [(a, 0, 0) for a in odd]
    gap6 = [(a, 1, c) for a in (0, 2, 4, 6) for c in (1, 2)]
    pools = (
        ((90, gap24), (10, gap12)),
        ((65, gap24), (35, gap12)),
        ((45, gap24), (45, gap12), (10, gap8)),
        ((30, gap24), (40, gap12), (30, gap6)),
        ((70, gap24), (10, gap12), (20, gap8)),
    )

    def difference(used):
        ticket = rng.randrange(100)
        for mass, pool in pools[mode]:
            ticket -= mass
            if ticket < 0:
                options = [g for g in pool
                           if min(G.idx(g), G.idx(G.inv(g))) not in used]
                if not options:
                    options = pool
                gap = rng.choice(options)
                used.add(min(G.idx(gap), G.idx(G.inv(gap))))
                return gap
        raise ValueError("difference distribution is not normalized")

    def b_cell(count, used):
        first = G.pick(rng)
        if count == 1:
            return [first]
        second = G.mul(first, difference(used))
        cell = [first, second]
        if count == 3:
            cell.append(rng.choice(
                [g for g in G.elements
                 if g[1] != first[1] and g not in cell]))
        return cell

    trials = []
    for trial in range(24):
        A = [[_a_cell(rng, G, gap24, 3 if mode == 4 else 2)
              for _ in range(2)]]
        B = []
        for counts in _b_support(rng, support_mode):
            used = set()
            B.append([b_cell(count, used) for count in counts])
        hx, hz, n = lifted_matrices(G, A, B)
        a_starts = [j*B_SHAPE[1]*q for j in range(2)]
        b_starts = [j*q for j in range(5)]
        a = _component_obstructions(hx[:q], a_starts, q)
        b = _component_obstructions(hz[:2*q], b_starts, q)
        screen = (a[0]+b[0], a[1]+b[1], a[2]+b[2],
                  max(a[2], b[2]), a[5]+b[5])
        trials.append((screen, trial, A, B, hx, hz, n))

    best = None
    for screen, trial, A, B, hx, hz, n in sorted(
            trials, key=lambda item: item[:2])[:6]:
        a = _component_obstructions(
            hx[:q], [j*B_SHAPE[1]*q for j in range(2)], q, rng)
        b = _component_obstructions(
            hz[:2*q], [j*q for j in range(5)], q, rng)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (int(k > 157), screen[:4], a[3]+b[3],
                 a[4]+b[4], screen[4], -k, trial)
        if best is None or grade < best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Parent's random B-heavy w=12 fillings did not reliably remove sector-1
# Z logicals of weight 6: measured slots 4, 6, 12 and 13 retained them.
# The order-34 (1,3)x(1,5) bed measured four weight-2 codes among its
# top five, refuting that the rate-0.500 shape alone prevents d=2.
# REFUTED: Bipartite binomial differences on the parent's abelian bed
# provide five distinct improved codes: only four of 60 slots survived
# isomorphism deduplication, and every measured slot had d30s=4.
# REFUTED: The parent's GL(2,3) balanced-component screen reliably
# reaches distance 12: its five measured d30s readings were 8, 9,
# 10, 10 and 10, with light logicals in sector-1 Acol/Bcol blocks.
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
