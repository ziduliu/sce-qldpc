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
CONCEPT = ("On cyclic C37 of order 37 with A=(2,3), B=(2,4), "
           "globally separated support gaps and deeper independent "
           "verified-component screening will select more distance-18 "
           "codes in rate cell 0.150.")
GROUP = ("Keep cyclic C37 at order 37: this n=592 bed has measured "
         "distance-18 codes and frontier fitness 42.6820. An unchanged "
         "group makes this a direct filling-only test.")
SHAPES = ("Keep (2,3)x(2,4): n=16*37=592, guaranteed k>=74 and "
          "rate>=0.125; measured k=83-88 lies in cell 0.150. The "
          "unequal two-row components have 111 and 148 variables, "
          "whereas equal shapes give equal lengths. The parent's "
          "lightest X witnesses spanned sector-1 Acol/Bcol blocks "
          "across the A columns.")
FILLING = ("Concentrate on balanced complementary B columns, retain "
           "legal zero/four-term alternatives, and test one mixed-A "
           "control. Separate binomial gaps across each entire side "
           "and discourage repeated dense-cell gaps. A rows weigh "
           "at most seven, B rows eight, B columns four: HX<=11 and "
           "HZ<=12. Independently screen more A labels and probe "
           "shortlisted verified kernels to suppress the parent's "
           "weight-16 witnesses while checking k above 74.")
FAMILY_NAME = "r015_c37_23x24_global_gaps_deep_kernel_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic C37, encoded as two-coordinate tuples.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def idx(g):
        return g[0] % q

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _counts(rng, idx):
    """Four balanced B motifs and one mixed-A control, times four regimes."""
    patterns = (
        ((1, 3, 2, 2), (3, 1, 2, 2)),
        ((1, 1, 3, 3), (3, 3, 1, 1)),
        ((0, 3, 2, 3), (4, 1, 2, 1)),
        ((0, 4, 2, 2), (4, 0, 2, 2)),
        ((1, 1, 3, 3), (3, 3, 1, 1)),
    )
    A = [[2] * 3 for _ in range(2)]
    if idx % 5 == 4:
        j = rng.randrange(3)
        A[0][j], A[1][j] = 3, 1
    rows = patterns[idx % 5]
    order = rng.sample(range(4), 4)
    B = [[row[j] for j in order] for row in rows]
    if rng.randrange(2):
        B.reverse()
    assert _grid_weights(A, B) <= 12
    assert _no_dead_lines(A, B)
    return A, B


def _labels(rng, counts, regime):
    """Separate unsigned differences across the whole component side."""
    pools = (tuple(range(1, 19)), tuple(range(3, 19)),
             tuple(range(1, 19, 2)), tuple(range(2, 19, 2)))
    result = []
    used = set()
    for row in counts:
        labelled = []
        for count in row:
            if count == 2:
                choices = [d for d in pools[regime] if d not in used]
                gap = rng.choice(choices or pools[regime])
                used.add(gap)
                start = rng.randrange(37)
                shifts = [start, (start + gap) % 37]
            elif count >= 3:
                shifts = None
                best_penalty = count * count + 19
                for _ in range(48 if count == 4 else 32):
                    candidate = rng.sample(range(37), count)
                    differences = {
                        min((candidate[u] - candidate[v]) % 37,
                            (candidate[v] - candidate[u]) % 37)
                        for u in range(count) for v in range(u)
                    }
                    penalty = (count * (count - 1) // 2
                               - len(differences)
                               + len(differences & used))
                    if penalty < best_penalty:
                        shifts, best_penalty = candidate, penalty
                    if penalty == 0:
                        break
                used.update(
                    min((shifts[u] - shifts[v]) % 37,
                        (shifts[v] - shifts[u]) % 37)
                    for u in range(count) for v in range(u))
            else:
                shifts = rng.sample(range(37), count)
            labelled.append([(a, 0) for a in shifts])
        result.append(labelled)
    return result


def _component_columns(rows, q, blocks, stride, check_blocks):
    """Read both component-check rows from the verified LP matrices."""
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            row = rows[i * check_blocks * q + r]
            mark = 1 << (i * q + r)
            for j in range(blocks):
                bits = (row >> (j * stride * q)) & mask
                while bits:
                    bit = bits & -bits
                    columns[j * q + bit.bit_length() - 1] |= mark
                    bits ^= bit
    return columns


def _rotate_bits(x, shift, q):
    mask = (1 << q) - 1
    shift %= q
    return ((x << shift) | (x >> (q - shift))) & mask


def _cyclic_product(x, y, q):
    result = 0
    while y:
        bit = y & -y
        result ^= _rotate_bits(x, bit.bit_length() - 1, q)
        y ^= bit
    return result


def _halve_even(x, q):
    """Solve (1+t)y=x in the cyclic ring, when x has even parity."""
    if x.bit_count() & 1:
        return None
    y = previous = 0
    for position in range(1, q):
        previous ^= (x >> position) & 1
        y |= previous << position
    return y if ((y >> (q - 1)) & 1) == (x & 1) else None


def _kernel_word(word, columns):
    syndrome = 0
    while word:
        bit = word & -word
        syndrome ^= columns[bit.bit_length() - 1]
        word ^= bit
    return syndrome == 0


def _minor_merit(columns, q, blocks, threshold):
    """Screen genuine two-row component words in the verified lift.

    A three-column minor is a kernel word. Divide its even block
    polynomials by 1+t only when the resulting word also verifies.
    Translated sums expose words that an individual minor misses.
    """
    from itertools import combinations

    mask = (1 << q) - 1
    polynomials = [
        [(columns[j * q] >> (i * q)) & mask for j in range(blocks)]
        for i in range(2)
    ]
    words = []
    for triple in combinations(range(blocks), 3):
        parts = [0] * blocks
        for j in triple:
            u, v = [t for t in triple if t != j]
            parts[j] = (
                _cyclic_product(polynomials[0][u],
                                polynomials[1][v], q)
                ^ _cyclic_product(polynomials[1][u],
                                  polynomials[0][v], q))
        for _ in range(3):
            word = sum(part << (j * q) for j, part in enumerate(parts))
            if word and _kernel_word(word, columns):
                words.append(word)
                # On an even-cell component, toggling a full block can
                # substantially shorten a divided minor. Verify it too.
                compact = word
                for j, part in enumerate(parts):
                    if part.bit_count() > q // 2:
                        compact ^= mask << (j * q)
                if compact != word and _kernel_word(compact, columns):
                    words.append(compact)
            divided = [_halve_even(part, q) for part in parts]
            if any(part is None for part in divided):
                break
            parts = divided

    words = list(dict.fromkeys(words))
    if not words:
        return (0, -100000)
    minimum = min(word.bit_count() for word in words)
    short = sum(word.bit_count() <= threshold for word in words)

    def observe(word):
        nonlocal minimum, short
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            short += weight <= threshold

    for i, left in enumerate(words):
        for j in range(i, len(words)):
            right = words[j]
            for shift in range(1 if i == j else 0, q):
                translated = 0
                for block in range(blocks):
                    part = (right >> (block * q)) & mask
                    translated |= _rotate_bits(part, shift, q) << (block * q)
                observe(left ^ translated)

    # Three translates of one relation need not resemble any pair.
    for base in words[:2]:
        shifted = []
        for shift in range(1, q):
            translated = 0
            for block in range(blocks):
                part = (base >> (block * q)) & mask
                translated |= _rotate_bits(part, shift, q) << (block * q)
            shifted.append(translated)
        for i, left in enumerate(shifted):
            for right in shifted[:i]:
                observe(base ^ left ^ right)
    return minimum, -short


def _kernel_probe(columns, rng):
    """Search verified component kernels using several pivot orders."""
    smallest = len(columns) + 1
    for _ in range(6):
        order = list(range(len(columns)))
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                pivot = syndrome.bit_length() - 1
                previous = pivots.get(pivot)
                if previous is None:
                    pivots[pivot] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            else:
                words.append(word)
                smallest = min(smallest, word.bit_count())
        words.sort(key=int.bit_count)
        short = words[:24]
        for i, left in enumerate(short):
            for right in short[:i]:
                combined = left ^ right
                if combined:
                    smallest = min(smallest, combined.bit_count())
        for i, left in enumerate(short[:16]):
            for j, middle in enumerate(short[:i]):
                pair = left ^ middle
                for right in short[:j]:
                    combined = pair ^ right
                    if combined:
                        smallest = min(smallest, combined.bit_count())
    return smallest


def NEW_CTX(rng, G, idx):
    """Screen A and B independently, then rank their verified pairings."""
    sa, sb = _counts(rng, idx)
    regime = (idx // 5) % 4
    anchor_b = _labels(rng, sb, (regime + 2) % 4)
    a_pool = []
    for trial in range(108):
        A = _labels(rng, sa, regime)
        hx, _, _ = lifted_matrices(G, A, anchor_b)
        ac = _component_columns(hx, G.q, 3, 4, 4)
        a_pool.append((_minor_merit(ac, G.q, 3, 17),
                       -trial, A, ac))

    a_pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
    anchor_a = a_pool[0][2]
    b_pool = []
    for trial in range(36):
        B = _labels(rng, sb, (regime + 2) % 4)
        _, hz, _ = lifted_matrices(G, anchor_a, B)
        bc = _component_columns(hz, G.q, 4, 1, 1)
        b_pool.append((_minor_merit(bc, G.q, 4, 19),
                       -trial, B, bc))

    def shortlist(pool):
        pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = []
        for merit, tie, grid, columns in pool[:10]:
            observed = min(merit[0], _kernel_probe(columns, rng))
            selected.append(((observed, merit[0], merit[1], tie), grid))
        selected.sort(key=lambda item: item[0], reverse=True)
        return selected[:4]

    a_best, b_best = shortlist(a_pool), shortlist(b_pool)
    best = None
    for a_grade, A in a_best:
        for b_grade, B in b_best:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            predicted = min(a_grade[0], b_grade[0], 21)
            grade = (75 <= k <= 103, k * predicted * predicted,
                     predicted, a_grade, b_grade)
            if best is None or grade > best[0]:
                best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Rolled singleton-heavy supports do not reliably avoid light logicals:
# all five measured parent slots had d30s=5, despite distinct labels.
# REFUTED: Disjoint doubles plus screening through component weight five
# suffice to raise distance above six: all five current measured slots
# retained d30s=6, with weight-six sector-1 X witnesses.
# REFUTED: The previous gap-orbit screen yielded five distance-eight
# readings: its measured distances were 8, 8, 8, 6, 6. Its A extraction
# inspected only the first of two component-check rows.
# REFUTED: Screening component relations only through weight four
# reliably selects distance-18 codes on C37 (2,3)x(2,4): four of the
# current parent's five measured readings were 16 or 17.
# REFUTED: Independent divided-minor screening of 32 labels per side
# improves the preceding 41.0074 fitness: its fitness was 38.1146,
# and four of five measured codes retained weight-16 X witnesses.
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
