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
CONCEPT = ("On cyclic C37 of order 37 with A=(2,3) and B=(2,4), complementary B-column profiles and distinct A-binomial gaps can yield five rate-0.150 codes with d30s above 18.")
GROUP = ("Jump from nonabelian order 56 to cyclic C37. The measured C37 (2,3)x(2,4) bed reached fitness 42.7 at n=592, with selected k=79-83 and d30s=17-18.")
SHAPES = ("A=(2,3), B=(2,4) give n=592 and guaranteed k>=74, rate 0.125; measured rank excess has placed this bed in cell 0.150. The unequal-length sides have two-check, three- and four-column components. Both the current parent's and the target bed's lightest reported logicals occupied sector-1 Acol/Bcol blocks.")
FILLING = ("Use distinct-gap A binomials and twenty indexed B count profiles whose rows sum to eight and columns to four. HX weighs at most 6+4=10 and HZ at most 8+4=12. Screen cyclic component relations, then verified component columns and exact k, seeking to replace the target bed's weight-17/18 sector-1 witnesses.")
FAMILY_NAME = "r015_c37_23x24_complementary_minor_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic C37, encoded by one-coordinate tuples.
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


def _component_columns(rows, starts, q):
    """Transpose one component directly from verified lifted check rows."""
    columns = [0] * (3 * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:q]):
        for block, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                columns[block * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit
    return columns


def _pair_grade(columns):
    """Count repeated columns, weight-four relations, and check-pair overlaps."""
    repeated = len(columns) - len(set(columns))
    pairs = {}
    fours = overlaps = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            previous = pairs.get(syndrome, 0)
            fours += previous
            pairs[syndrome] = previous + 1
            shared = (left & right).bit_count()
            overlaps += shared * (shared - 1) // 2
    return repeated, fours, overlaps


def _rooted_min(columns, q, limit, node_cap):
    """Budget-search component words rooted in each variable block.

    Regular translations let a word touching a block be rooted at that
    block's identity variable. A budget exhaustion is not a distance proof.
    """
    adjacency = [[] for _ in range(q)]
    for variable, syndrome in enumerate(columns):
        bits = syndrome
        while bits:
            bit = bits & -bits
            adjacency[bit.bit_length() - 1].append(variable)
            bits ^= bit

    exhausted = 0
    for block in range(3):
        root = block * q
        budget = [node_cap]

        def visit(syndrome, used, depth):
            if not syndrome:
                return depth
            if (depth >= limit or
                    syndrome.bit_count() > 3 * (limit - depth) or
                    budget[0] <= 0):
                return None
            budget[0] -= 1

            # Any completion must flip an odd check; try a constrained
            # check and the variables that cancel most odd checks first.
            bits = syndrome
            pivot = None
            available = len(columns) + 1
            for _ in range(min(4, bits.bit_count())):
                bit = bits & -bits
                check = bit.bit_length() - 1
                bits ^= bit
                choices = sum(not ((used >> v) & 1)
                              for v in adjacency[check])
                if choices < available:
                    available, pivot = choices, check
            if not available:
                return None

            options = [v for v in adjacency[pivot]
                       if not ((used >> v) & 1)]
            options.sort(
                key=lambda v: (columns[v] & syndrome).bit_count(),
                reverse=True)
            for v in options:
                result = visit(syndrome ^ columns[v],
                               used | (1 << v), depth + 1)
                if result is not None:
                    return result
            return None

        found = visit(columns[root], 1 << root, 1)
        if found is not None:
            return found, exhausted
        exhausted += budget[0] == 0
    return limit + 1, exhausted


def _cyclic_minor_grade(grid, inverse=False):
    """Screen short two-check component relations over F2[C37]."""
    from itertools import combinations

    q = 37
    mask = (1 << q) - 1

    def rotate(bits, shift):
        return ((bits << shift) | (bits >> (q - shift))) & mask

    def product(left, right):
        value = 0
        while right:
            bit = right & -right
            value ^= rotate(left, bit.bit_length() - 1)
            right ^= bit
        return value

    blocks = []
    for row in grid:
        blocks.append([
            sum(1 << ((-g[0] if inverse else g[0]) % q) for g in cell)
            for cell in row
        ])

    words = []
    width = len(grid[0])
    for chosen in combinations(range(width), 3):
        parts = [0] * width
        for place, j in enumerate(chosen):
            u, v = (chosen[t] for t in range(3) if t != place)
            parts[j] = (product(blocks[0][u], blocks[1][v]) ^
                        product(blocks[1][u], blocks[0][v]))
        word = sum(part << (q * j) for j, part in enumerate(parts))
        if not word:
            return (0, -100000, -100000)
        words.append(word)

    shifted = []
    for word in words:
        shifted.append([
            sum(rotate((word >> (q * j)) & mask, shift) << (q * j)
                for j in range(width))
            for shift in range(q)
        ])

    minimum = min(word.bit_count() for word in words)
    light16 = light18 = 0
    for i, word in enumerate(words):
        for j in range(i, len(words)):
            for other in shifted[j]:
                combined = word ^ other
                if combined:
                    weight = combined.bit_count()
                    minimum = min(minimum, weight)
                    light16 += weight <= 16
                    light18 += weight <= 18
    return (minimum, -light16, -light18)


def _verified_columns(rows, starts, q):
    """Transpose every supplied component row from the verified lift."""
    columns = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        mark = 1 << check
        for block, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                columns[block * q + bit.bit_length() - 1] |= mark
                bits ^= bit
    return columns


def NEW_CTX(rng, G, idx):
    """Compare fresh labels within one of twenty distinct count profiles."""
    from itertools import permutations

    profiles = (sorted(set(permutations((3, 2, 2, 1)))) +
                sorted(set(permutations((3, 3, 1, 1)))) +
                [(2, 2, 2, 2), (4, 2, 1, 1)])
    top = profiles[idx % 20]
    bottom = tuple(4 - count for count in top)
    q = G.q

    def draw_a():
        gaps = rng.sample(range(1, 19), 6)
        rng.shuffle(gaps)
        return [[
            [(start,), ((start + gaps[3 * i + j]) % q,)]
            for j, start in enumerate(rng.randrange(q) for _ in range(3))]
            for i in range(2)
        ]

    def draw_b():
        grid = []
        for counts in (top, bottom):
            row = []
            for count in counts:
                if not count:
                    row.append([])
                    continue
                # Avoid repeated undirected gaps inside multi-term cells.
                while True:
                    labels = rng.sample(range(q), count)
                    gaps = {
                        min((labels[u] - labels[v]) % q,
                            (labels[v] - labels[u]) % q)
                        for u in range(count) for v in range(u)
                    }
                    if len(gaps) == count * (count - 1) // 2:
                        break
                row.append([(value,) for value in labels])
            grid.append(row)
        return grid

    proposals = []
    for trial in range(32):
        A, B = draw_a(), draw_b()
        am = _cyclic_minor_grade(A, inverse=True)
        bm = _cyclic_minor_grade(B)
        preliminary = (min(am[0], bm[0]), am[0] + bm[0],
                       am[1] + bm[1], am[2] + bm[2], -trial)
        proposals.append((preliminary, A, B))

    best = None
    for preliminary, A, B in sorted(
            proposals, key=lambda entry: entry[0], reverse=True)[:5]:
        hx, hz, n = lifted_matrices(G, A, B)
        a_rows = [hx[i * 4 * q + r] for i in range(2)
                  for r in range(q)]
        b_rows = hz[:2 * q]
        ac = _verified_columns(a_rows, (0, 4*q, 8*q), q)
        bc = _verified_columns(b_rows, (0, q, 2*q, 3*q), q)
        ag, bg = _pair_grade(ac), _pair_grade(bc)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (74 <= k <= 103, preliminary[0],
                 -(ag[0] + bg[0]), -(ag[1] + bg[1]),
                 preliminary[1], preliminary[2], preliminary[3],
                 -(ag[2] + bg[2]), -abs(k - 83))
        if best is None or grade > best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell, including deliberate zero blocks."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support variation on the parent's (1,3)x(1,3) bed
# suffices to remove weight-4 logicals: all five measured slots had d30s=4.
# REFUTED: The previous (1,4)x(1,4) penalty screen suffices to remove
# short logicals: its five measured slots all retained d30s=3 witnesses.
# REFUTED: Triangle-free binomial gaps eliminate weight-four logicals on (1,4)x(1,4): all five measured parent slots retained d30s=4.
# REFUTED: The order-56 parent's coset-diverse triplets raise k above
# its rank-count bound: all 60 distinct slots had k=224, and its five
# measured slots retained sector-1 distance-eight witnesses.
# REFUTED: Action-coset-spread triples with rooted screening lift that
# order-56 distance-eight ceiling: the current five measured codes
# still had k=224 and d30s=8, with sector-1 witnesses.
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
