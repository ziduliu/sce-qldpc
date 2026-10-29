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
CONCEPT = ("On cyclic C37 of order 37 with A=(2,3), B=(2,4), screen both exact lifted components for short relations; five fresh fillings with stronger component readings may improve the rate-0.150 frontier.")
GROUP = ("C37 allows n=592 in these shapes. Its measured bed reached fitness 42.7 with k=79-83 and distance-17/18 readings; the parent's order-55 bed reached distance eight in another rate cell.")
SHAPES = ("A=(2,3), B=(2,4) give k>=74, guaranteed rate 0.125, and unequal two-check components of three and four variable blocks. Measured C37 rank deficits put codes near rate 0.150. The parent's lightest logicals occupied sector-1 Acol/Bcol blocks, so both corresponding components are screened.")
FILLING = ("Twenty indexed support-and-gap designs mostly pair binomial A cells with complementary B rows. Exact component columns penalize short relations and probe rooted words; exact rank keeps finalists in the target cell. A-row plus B-column weights are at most 12, as are B-row plus A-column weights.")
FAMILY_NAME = "r015_c37_23x24_independent_component_probe"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # C37, represented by one-coordinate tuples for the custom interface.
    els = [(a,) for a in range(37)]

    def red(g):
        return (g[0] % 37,)

    def idx(g):
        return g[0] % 37

    def mul(g, h):
        return ((g[0] + h[0]) % 37,)

    def inv(g):
        return ((-g[0]) % 37,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _component_columns(rows, q, side):
    """Extract every check row of one component from the verified lift."""
    blocks = 3 if side == "A" else 4
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for check in range(2 * q):
        alpha, r = divmod(check, q)
        # HX has row-block order (A-row, B-column); fix B-column 0.
        # HZ has row-block order (A-column, B-row); fix A-column 0.
        row = rows[(alpha * 4 * q + r) if side == "A" else check]
        for block in range(blocks):
            start = (block * 4 * q) if side == "A" else (block * q)
            bits = (row >> start) & mask
            while bits:
                low = bits & -bits
                columns[block * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return columns


def _short_relations(columns):
    """Penalize zero columns, duplicate columns and pair collisions."""
    singles = set(columns)
    bad = int(0 in singles) + len(columns) - len(singles)
    pairs = set()
    fours = overlaps = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            fours += syndrome in pairs
            pairs.add(syndrome)
            shared = (left & right).bit_count()
            overlaps += shared * (shared - 1) // 2
    return bad, fours, overlaps


def _rooted_word(columns, q, limit, node_cap):
    """Budgeted component-kernel probe, rooted once per variable block."""
    adjacency = [[] for _ in range(2 * q)]
    degree = max(x.bit_count() for x in columns)
    for v, syndrome in enumerate(columns):
        bits = syndrome
        while bits:
            low = bits & -bits
            adjacency[low.bit_length() - 1].append(v)
            bits ^= low

    best = limit + 1
    for block in range(len(columns) // q):
        root = block * q
        budget = [node_cap]

        def visit(syndrome, used, depth):
            nonlocal best
            if not syndrome:
                best = min(best, depth)
                return
            if depth >= min(limit, best) or budget[0] <= 0:
                return
            if syndrome.bit_count() > degree * (min(limit, best) - depth):
                return
            budget[0] -= 1
            bits = syndrome
            pivot = None
            fewest = len(columns) + 1
            for _ in range(min(4, bits.bit_count())):
                low = bits & -bits
                bits ^= low
                check = low.bit_length() - 1
                available = sum(not ((used >> v) & 1)
                                for v in adjacency[check])
                if available < fewest:
                    fewest, pivot = available, check
            if not fewest:
                return
            options = [v for v in adjacency[pivot]
                       if not ((used >> v) & 1)]
            options.sort(key=lambda v: (columns[v] & syndrome).bit_count(),
                         reverse=True)
            for v in options:
                visit(syndrome ^ columns[v], used | (1 << v), depth + 1)
                if budget[0] <= 0:
                    return

        visit(columns[root], 1 << root, 1)
    return best


def _cell(rng, count, mode):
    """Sample a distinct cyclic support with an indexed gap rule."""
    if count == 0:
        return []
    if count == 1:
        return [(rng.randrange(37),)]
    if count == 2:
        base = rng.randrange(37)
        gap = rng.randrange(1 + mode, 19)
        return [(base,), ((base + gap) % 37,)]
    for _ in range(60):
        values = rng.sample(range(37), count)
        distances = [min((x - y) % 37, (y - x) % 37)
                     for i, x in enumerate(values) for y in values[:i]]
        if min(distances) >= 1 + mode:
            return [(a,) for a in values]
    return [(a,) for a in rng.sample(range(37), count)]


def NEW_CTX(rng, G, idx):
    """Choose fresh labels for one of twenty support-and-gap designs."""
    design = idx % 20
    mode = (design // 5) % 4
    profiles = ((3, 1, 3, 1), (3, 2, 2, 1),
                (3, 3, 1, 1), (4, 1, 2, 1),
                (2, 3, 1, 2))
    if design < 15:
        a_counts = [[2, 2, 2], [2, 2, 2]]
        top = profiles[design % 5]
        b_counts = [list(top), [4 - x for x in top]]
        # A rows sum to 6, B rows to 8; B columns sum to 4.
        # Thus HX <= 6+4 and HZ <= 8+4.
    else:
        a_counts = [[3, 2, 2], [2, 3, 2]]
        shift = design % 4
        top = (3, 1, 2, 1)
        bottom = (1, 3, 1, 2)
        b_counts = [
            list(top[shift:] + top[:shift]),
            list(bottom[shift:] + bottom[:shift]),
        ]
        # A rows sum to 7, B rows to 7; their column maxima
        # are respectively 5 and 4. HX <= 11, HZ <= 12.

    proposals = []
    q = G.q
    for trial in range(20):
        grids = {
            "A": [[_cell(rng, count, mode) for count in row]
                  for row in a_counts],
            "B": [[_cell(rng, count, mode) for count in row]
                  for row in b_counts],
        }
        hx, hz, n = lifted_matrices(G, grids["A"], grids["B"])
        acol = _component_columns(hx, q, "A")
        bcol = _component_columns(hz, q, "B")
        a = _short_relations(acol)
        b = _short_relations(bcol)
        coarse = (a[0] + b[0], max(a[1], b[1]),
                  a[1] + b[1], a[2] + b[2], rng.random())
        proposals.append((coarse, grids, acol, bcol, hx, hz, n))

    proposals.sort(key=lambda item: item[0])
    finalists = []
    for coarse, grids, acol, bcol, hx, hz, n in proposals[:5]:
        da = _rooted_word(acol, q, 12, 850)
        db = _rooted_word(bcol, q, 12, 850)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (75 <= k <= 103, min(da, db), da + db,
                 -coarse[0], -coarse[1], -coarse[2],
                 k, -coarse[3], -coarse[4])
        finalists.append((grade, grids))
    return max(finalists, key=lambda item: item[0])[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support counts and uniform labels on the order-58
# dihedral parent did not suppress sector-1 weight-four logicals:
# all five measured parent slots had d30s=4.
# REFUTED: Pair-relation screening with mostly three-distinct-coset
# triples on this order-55 bed exceeded distance eight: its five
# measured slots all had k=220 and d30s=8.
# REFUTED: Rooted weight-eight screening on the order-55 (1,3)x(1,3)
# bed raised its five-code score above 25.6: the current program's
# five measured slots again had k=220 and d30s=8.
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
