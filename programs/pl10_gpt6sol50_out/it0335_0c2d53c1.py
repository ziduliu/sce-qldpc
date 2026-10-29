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
CONCEPT = ("In C3 x S3 of order 18, A=(1,4) and B=(2,7), screened support-and-coset designs may reduce worst-basis circuit failures near rate 0.500.")
GROUP = ("Replace cyclic Z19 with the nonabelian direct product C3 x S3. Its order-18 lift gives n=540; noncommuting labels test a group action absent from the parent's matrices.")
SHAPES = ("Retain (1,4)x(2,7): n=540 and guaranteed k>=270, rate>=0.500. The one-row four-column A and two-row seven-column B components have unequal lengths. Parent failures were spread across sector-1 Acol/Bcol data blocks, with check-block enrichments near 1.0.")
FILLING = ("Cross four weight-legal support motifs with five relative-element patterns. Use order-six relative elements in binomials, vary cosets in larger cells, and screen verified component relations and check overlaps before checking k. Test for fewer worst-basis failed shots, rather than optimizing distance alone.")
FAMILY_NAME = "r050_c3_times_s3_14x27_support_coset_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 18

def custom_ops(q):
    # Direct product C3 x (C3 : C2), with C2 inverting its C3.
    els = [(a, b, c) for a in range(3)
           for b in range(3) for c in range(2)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3, int(g[2]) % 2)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + d) % 3,
                (b + (-e if c else e)) % 3,
                (c + f) % 2)

    def inv(g):
        a, b, c = red(g)
        return ((-a) % 3, (b if c else -b) % 3, c)

    def idx(g):
        a, b, c = red(g)
        return 6 * a + 2 * b + c

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)





def _verified_component_signatures(hx, hz, q):
    """Extract both components from the exact LP lifts, not a guessed action."""
    mask = (1 << q) - 1
    a, b = [0] * (4 * q), [0] * (7 * q)
    for r in range(q):
        for j in range(4):
            bits = (hx[r] >> (j * 7 * q)) & mask
            while bits:
                bit = bits & -bits
                a[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
    for alpha in range(2):
        for r in range(q):
            for j in range(7):
                bits = (hz[alpha * q + r] >> (j * q)) & mask
                while bits:
                    bit = bits & -bits
                    b[j * q + bit.bit_length() - 1] |= 1 << (alpha * q + r)
                    bits ^= bit
    return a, b


def _component_relations(columns):
    """Count short relations among verified A-component columns."""
    singles = set(columns)
    pair_counts = {}
    one = two = three = four = 0
    for i, left in enumerate(columns):
        one += left == 0
        for right in columns[:i]:
            value = left ^ right
            two += value == 0
            three += value in singles
            four += pair_counts.get(value, 0)
            pair_counts[value] = pair_counts.get(value, 0) + 1
    return one, two, three, four


def _element_order(G, g):
    value = G.e
    for order in range(1, G.q + 1):
        value = G.mul(value, g)
        if value == G.e:
            return order
    raise ValueError("group element has no order")


def _relative_pools(G):
    pools = {2: [], 3: [], 6: []}
    for g in G.elements:
        order = _element_order(G, g)
        if order in pools:
            pools[order].append(g)
    assert pools[6] and pools[3] and pools[2]
    return pools


def _support_motif(idx):
    """Twenty indexed designs: four supports times five coset patterns."""
    kind, phase = idx % 4, (idx // 4) % 5
    if kind == 0:
        a_counts = [3, 3, 2, 2]
        b_counts = [[1] * 7 for _ in range(2)]
    elif kind == 1:
        a_counts = [4, 2, 2, 2]
        b_counts = [[1] * 7 for _ in range(2)]
    elif kind == 2:
        a_counts = [3, 2, 2, 2]
        b_counts = [[1] * 7 for _ in range(2)]
        for j in ((0, 2), (1, 3))[0]:
            b_counts[0][(j + phase) % 7] = 2
        for j in (4, 6):
            b_counts[1][(j + phase) % 7] = 2
    else:
        a_counts = [2, 2, 2, 2]
        b_counts = [[1] * 7 for _ in range(2)]
        for j in (0, 2, 4):
            b_counts[0][j] = 2
        for j in ((1, 3, 5), (0, 3, 5), (0, 2, 5),
                  (0, 2, 4), (1, 2, 4))[phase]:
            b_counts[1][j] = 2
    sa = [a_counts]
    assert _grid_weights(sa, b_counts) <= 12
    assert _no_dead_lines(sa, b_counts)
    return sa, b_counts, phase


def _draw_cell(rng, G, count, phase, pools, side):
    """Distinct labels; binomial relative elements have order six."""
    first = G.pick(rng)
    if count == 1:
        return [first]
    relatives = [rng.choice(pools[6])]
    if count >= 3:
        # Five distinct coset patterns change the relative-element model.
        second_order = (6, 3, 2, 3, 6)[phase]
        choices = [g for g in pools[second_order]
                   if g not in (G.e, relatives[0])]
        relatives.append(rng.choice(choices))
    if count == 4:
        choices = [g for g in pools[3] + pools[6]
                   if g not in (G.e, *relatives)]
        relatives.append(rng.choice(choices))
    return [first] + [G.mul(first, g) for g in relatives]


def _root_four_cost(rows, q):
    """Rooted four-cycle pressure across all lifted check blocks."""
    total = 0
    for root in range(0, len(rows), q):
        for j, row in enumerate(rows):
            if j == root:
                continue
            common = (rows[root] & row).bit_count()
            total += common * (common - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Screen one indexed design on verified LP matrices and rank."""
    sa, sb, phase = _support_motif(idx)
    pools = _relative_pools(G)
    shortlist = []
    for _ in range(12):
        a = [[_draw_cell(rng, G, count, phase, pools, "A")
              for count in sa[0]]]
        b = [[_draw_cell(rng, G, count, phase, pools, "B")
              for count in row] for row in sb]
        hx, hz, n = lifted_matrices(G, a, b)
        ac, bc = _verified_component_signatures(hx, hz, G.q)
        ar, br = _component_relations(ac), _component_relations(bc)
        cost = (ar[0] + br[0], ar[1] + br[1],
                max(ar[2], br[2]), ar[2] + br[2],
                max(ar[3], br[3]),
                _root_four_cost(hx, G.q) +
                _root_four_cost(hz, G.q), rng.random())
        shortlist.append((cost, a, b, hx, hz, n))
    shortlist.sort(key=lambda item: item[0])

    best = None
    for cost, a, b, hx, hz, n in shortlist[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # The 0.500 cell at n=540 contains integer k=257..283.
        key = (not 257 <= k <= 283, cost, abs(k - 273))
        if best is None or key < best[0]:
            best = (key, a, b)
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing its terms."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Separated differences with the parent's mixed B doubles did not prevent 185-199/250 X-basis failures or sector-1 weight-three Z witnesses in its five measured slots.
# Mostly monomial B with distinct A gaps did not give diverse, reliably protected codes: only seven slots were distinct, and a measured slot failed 179/250 X-basis shots.
# Triangle-free A-binomial gaps with distributed B doubles did not improve the sparse-B parent: fitness fell from 3.3369 to 3.2638, and the five measured slots had 131-166 X failures per 250 shots.
# Two A trinomials with monomial B did not yield twenty inequivalent designs: 48 of 60 slots were isomorphic duplicates; the five measured slots scored only 3.32-3.42.
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
