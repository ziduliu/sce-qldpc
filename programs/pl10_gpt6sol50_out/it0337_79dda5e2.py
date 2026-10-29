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
CONCEPT = ("In cyclic Z17 of order 17, A=(2,5) and B=(2,6), balanced degree-three component columns screened for short relations should reduce worst-basis circuit failures.")
GROUP = ("Keep the cyclic construction rule but use order 17 for the new shapes. The measured Z17 (2,5)x(2,6) bed reached about 4.7 with weight-eight witnesses; this tests whether improved incidence screening can outperform the parent's Z13 bed.")
SHAPES = ("(2,5)x(2,6) gives n=578, guaranteed k>=204 and rate>=0.353 in cell 0.350. Its unequal component lengths are 85 and 102. The parent's weight-six witnesses and failures spread over sector-1 Acol/Bcol blocks; slot 34 had 101/9750 Z and 103/11250 X failed shots, without a persistent check-block hotspot.")
FILLING = ("Every A and B column has three terms, distributed as (2,1), (1,2), or balanced paired (3,0)/(0,3) patterns. Both check-row sums stay <=12. Indexed gap motifs and verified-lift relation screens test whether fewer shots fail and fewer logical qubits flip; neither k excess nor distance proxy is the fitness.")
FAMILY_NAME = "r035_z17_25x26_balanced_degree3_relations_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same additive cyclic construction rule, now Z17, encoded as pairs.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return int(g[0]) % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb); n = 34 * 17 = 578


def _verified_component_columns(hx, hz, q):
    """Extract component columns from the exact scaffold LP lifts."""
    mask = (1 << q) - 1
    a = [0] * (A_SHAPE[1] * q)
    b = [0] * (B_SHAPE[1] * q)
    for i in range(A_SHAPE[0]):
        for r in range(q):
            row = hx[(i * B_SHAPE[1]) * q + r]
            for j in range(A_SHAPE[1]):
                bits = (row >> ((j * B_SHAPE[1]) * q)) & mask
                while bits:
                    bit = bits & -bits
                    a[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    bits ^= bit
    for alpha in range(B_SHAPE[0]):
        for r in range(q):
            row = hz[alpha * q + r]
            for beta in range(B_SHAPE[1]):
                bits = (row >> (beta * q)) & mask
                while bits:
                    bit = bits & -bits
                    b[beta * q + bit.bit_length() - 1] |= (
                        1 << (alpha * q + r))
                    bits ^= bit
    return a, b


def _pair_cost(columns):
    """Short component dependencies and pairs of shared checks."""
    singles = set(columns)
    pairs = {}
    two = three = four = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            syndrome = left ^ right
            two += syndrome == 0
            three += syndrome in singles
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = (i, j)
            elif i not in previous and j not in previous:
                four += 1
            shared = (left & right).bit_count()
            cycles += shared * (shared - 1) // 2
    return two, three, four, cycles


def _six_cost(columns, cap=2000):
    """Count disjoint repeated triple syndromes up to a screen cap."""
    seen = {}
    defects = 0
    length = len(columns)
    for i in range(length - 2):
        for j in range(i + 1, length - 1):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, length):
                syndrome = partial ^ columns[k]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = (i, j, k)
                elif i not in old and j not in old and k not in old:
                    defects += 1
                    if defects >= cap:
                        return cap
    return defects


def _counts(rng, width, triple_pair):
    """Balance every column at degree three and both component rows."""
    order = rng.sample(range(width), width)
    top = set(order[:width // 2])
    grid = [[2 if j in top else 1 for j in range(width)],
            [1 if j in top else 2 for j in range(width)]]
    if triple_pair:
        upper = rng.choice(sorted(top))
        lower = rng.choice(sorted(set(range(width)) - top))
        grid[0][upper], grid[1][upper] = 3, 0
        grid[0][lower], grid[1][lower] = 0, 3
    return grid


def _draw_grid(rng, q, supports, indexed_gaps):
    """Label binomials with distinct gaps and triples with spaced terms."""
    gaps = list(indexed_gaps)
    rng.shuffle(gaps)
    result = []
    for row in supports:
        filled = []
        for count in row:
            if count == 0:
                cell = []
            elif count == 1:
                cell = [(rng.randrange(q), 0)]
            elif count == 2:
                start = rng.randrange(q)
                gap = gaps.pop() * rng.choice((-1, 1))
                cell = [(start, 0), ((start + gap) % q, 0)]
            else:
                while True:
                    values = rng.sample(range(q), 3)
                    distances = [min((values[u] - values[v]) % q,
                                     (values[v] - values[u]) % q)
                                 for u in range(3) for v in range(u)]
                    if min(distances) >= 2:
                        break
                cell = [(value, 0) for value in values]
            filled.append(cell)
        result.append(filled)
    return result


def NEW_CTX(rng, G, idx):
    """Select one of twenty indexed incidence/gap designs."""
    from itertools import combinations

    design = idx % 20
    mode = design % 4
    a_gaps = list(combinations(range(1, 9), 5))[
        (9 * design + idx // 20) % 56]
    b_gaps = list(combinations(range(1, 9), 6))[
        (11 * design + idx // 20) % 28]
    shortlist = []
    for _ in range(15 + idx % 5):
        sa = _counts(rng, 5, mode in (2, 3))
        sb = _counts(rng, 6, mode in (1, 3))
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        A = _draw_grid(rng, G.q, sa, a_gaps)
        B = _draw_grid(rng, G.q, sb, b_gaps)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _verified_component_columns(hx, hz, G.q)
        af, bf = _pair_cost(ac), _pair_cost(bc)
        key = (af[0] + bf[0], af[1] + bf[1],
               max(af[2], bf[2]), af[2] + bf[2],
               af[3] + bf[3], rng.random())
        shortlist.append((key, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[0])

    best_key = best = None
    for key, A, B, hx, hz, n, ac, bc in shortlist[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 204 <= k <= 216:
            continue
        six_a = _six_cost(ac)
        six_b = _six_cost(bc)
        choice = (key[:4], max(six_a, six_b), six_a + six_b,
                  key[4], k - 204, key[5])
        if best_key is None or choice < best_key:
            best_key, best = choice, {"A": A, "B": B}
    if best is None:
        raise ValueError("no positive-k filling in target cell")
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Predominantly single-term rolled D7 cells suffice for low circuit failure:
# selected parent codes had 101-113 worst-basis failures in 250 shots and
# reported weight-four sector-2 logical operators.
# Pair-signature screening alone did not eliminate the current bed's weight-3/4 sector-1 logicals or its frequent measured failures.
# Penalizing component weights 2-4 across the previous five profiles did not
# improve this bed: fitness was 3.3540 versus the preceding 3.3584, and
# every measured slot was limited by Z-basis failures.
# REFUTED: Adding a B triple to the order-42 single-row bed reliably
# suppresses circuit failures: this program scored 3.5582, and every
# measured slot still reached the failed-shot cap in both bases.
# REFUTED: Reinforced B columns and verified short-relation screening
# alone made the Z13 (2,5)x(3,8) circuit failure-rare: the resulting
# fitness was 4.8521; all five measured slots reached the failed-shot
# cap in both bases, with widespread logical flips.
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
