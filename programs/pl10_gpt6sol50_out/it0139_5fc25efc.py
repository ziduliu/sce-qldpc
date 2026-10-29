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
CONCEPT = ("For cyclic Z19 of order 19 with A=(2,5) and B=(2,5), degree-four component columns and short-relation screening should reduce circuit-level failures in the 0.300 rate cell.")
GROUP = ("Keep the measured cyclic Z19 order-19 bed fixed for a filling-only comparison with the 5.3789 parent; prime order permits varied nonzero voltage-gap sets.")
SHAPES = ("Keep equal (2,5)x(2,5) sides: n=551, guaranteed k=171 and rate 0.310, with two length-95 component codes. The parent's weight-eight witnesses occupied sector-1 Acol/Bcol blocks; its failed-shot maps showed broadly distributed data errors and no dominant check block.")
FILLING = ("Use mostly six-surplus-term components, each with one degree-four column, plus five-surplus lower-noise comparisons. Selected designs use a trinomial instead of two binomials. Index distinct gap sets, screen both components for short relations, and verify target-cell k. Test for fewer worse-basis failures and decoder-struggling shots, rather than scoring distance or excess k.")
FAMILY_NAME = "r030_z19_25x25_degree4_relation_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return g[0] % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb); n = 29 * 19 = 551


def _support_profile(rng, surplus, trinomial):
    """Rows total at most eight; each column has three or four terms."""
    order = rng.sample(range(5), 5)
    counts = [[1] * 5 for _ in range(2)]
    for j in order[:3]:
        counts[0][j] = 2
    for j in order[3:]:
        counts[1][j] = 2
    if surplus == 6:
        counts[1][order[0]] = 2
        if trinomial:
            # Move one row-zero surplus to an exclusive column. The
            # column totals remain 4,3,3,3,3 and both row totals stay 8.
            counts[0][order[0]] = 1
            counts[0][order[1]] = 3
    return counts


def _draw_component(rng, counts, gap_index):
    """Fill cells from an indexed spectrum of distinct cyclic gaps."""
    from itertools import combinations

    surplus = sum(count - 1 for row in counts for count in row)
    spectra = list(combinations(range(1, 10), surplus))
    gaps = list(spectra[gap_index % len(spectra)])
    rng.shuffle(gaps)
    component = []
    for row in counts:
        out = []
        for count in row:
            start = rng.randrange(19)
            cell = [(start, 0)]
            for _ in range(count - 1):
                gap = gaps.pop()
                if rng.randrange(2):
                    gap = -gap
                cell.append(((start + gap) % 19, 0))
            out.append(cell)
        component.append(out)
    return component


def _component_columns(G, grid, side):
    """Column syndromes under the scaffold's left/right lift actions."""
    columns = []
    for j in range(5):
        for h in G.elements:
            syndrome = 0
            for i in range(2):
                for g in grid[i][j]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)
    return columns


def _component_defects(columns):
    """Count short relations and pairs sharing two check neighbors."""
    singles = set(columns)
    seen = {}
    bad2 = sum(value == 0 for value in columns)
    bad3 = bad4 = four_cycles = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = left ^ right
            bad2 += value == 0
            bad3 += value in singles
            bad4 += seen.get(value, 0)
            seen[value] = seen.get(value, 0) + 1
            common = (left & right).bit_count()
            four_cycles += common * (common - 1) // 2
    return bad2, bad3, bad4, four_cycles


def _five_six_defects(columns, limit=500):
    """Bound disjoint pair/triple and triple/triple collisions."""
    pairs = {}
    length = len(columns)
    for i in range(length):
        for j in range(i + 1, length):
            pairs.setdefault(columns[i] ^ columns[j], (i, j))

    triples = {}
    bad5 = bad6 = 0
    for i in range(length - 2):
        for j in range(i + 1, length - 1):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, length):
                value = partial ^ columns[k]
                pair = pairs.get(value)
                if pair is not None and i not in pair and j not in pair \
                        and k not in pair:
                    bad5 += 1
                previous = triples.get(value)
                if previous is None:
                    triples[value] = (i, j, k)
                elif i not in previous and j not in previous \
                        and k not in previous:
                    bad6 += 1
                if bad5 + bad6 >= limit:
                    return bad5, bad6
    return bad5, bad6


def NEW_CTX(rng, G, idx):
    """Screen an indexed support and voltage-spectrum design."""
    mode = idx % 10
    surplus_a = 5 if mode in (3, 8) else 6
    surplus_b = 5 if mode in (4, 9) else 6
    sa = _support_profile(rng, surplus_a, mode in (2, 7))
    sb = _support_profile(rng, surplus_b, mode in (1, 6))
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    shortlist = []
    for trial in range(18 + idx % 4):
        A = _draw_component(rng, sa, 11 * idx + 7 * trial)
        B = _draw_component(rng, sb, 11 * idx + 7 * trial + 29)
        acols = _component_columns(G, A, "A")
        bcols = _component_columns(G, B, "B")
        ad = _component_defects(acols)
        bd = _component_defects(bcols)
        preliminary = (ad[0] + bd[0], ad[1] + bd[1],
                       max(ad[2], bd[2]), ad[2] + bd[2],
                       max(ad[3], bd[3]), ad[3] + bd[3],
                       rng.random())
        shortlist.append((preliminary, A, B, acols, bcols))
    shortlist.sort(key=lambda entry: entry[0])

    chosen = None
    chosen_key = None
    # n=551 stays in the 0.300 cell when 171 <= k <= 179.
    for preliminary, A, B, acols, bcols in shortlist[:3]:
        k = lifted_k(G, A, B)
        a5, a6 = _five_six_defects(acols)
        b5, b6 = _five_six_defects(bcols)
        key = (not 171 <= k <= 179, preliminary[:4],
               max(a5 + a6, b5 + b6), a5 + a6 + b5 + b6,
               preliminary[4:], abs(k - 171))
        if chosen_key is None or key < chosen_key:
            chosen_key = key
            chosen = {"A": A, "B": B}
    return chosen


def CELL(rng, G, side, i, j, ctx):
    """Return the independently screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled (2,4)x(3,5) supports protect both bases comparably:
# all five measured parent slots had X as their worse basis.
# REFUTED: The Z23 (2,3)x(2,5) gap-repeat screen balances the bases:
# its five measured slots had 95-100 Z failures, versus 0-17 X failures.
# REFUTED: Column-covering sparse Z19 profiles alone eliminate frequent failures; all five measured slots retained weight-eight witnesses and reached the 100-failure cap in at least one basis.
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
