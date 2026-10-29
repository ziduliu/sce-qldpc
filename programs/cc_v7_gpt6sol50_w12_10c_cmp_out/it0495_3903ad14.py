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
CONCEPT = ("On C19 x S3 of order 114 with A=(1,2), B=(1,2), screened four-term components should produce five novel cell-0.200 codes with higher fitness than the cyclic order-19 parent.")
GROUP = ("Jump to C19 x S3: its centre has order 19 and it has 57 conjugacy classes, matching the identifying facts of the order-114 bed whose best fitness is 105.4. Order 114 gives n=570 on these shapes.")
SHAPES = ("Equal (1,2)x(1,2) sides give equal-length, 228-variable X/Z components. Guaranteed k=114 and rate 114/570=0.200 target cell 0.200. The parent's lightest X and Z logicals were entirely in sector-1 Acol/Bcol blocks.")
FILLING = ("Use four distinct terms per cell, making both check-row weights 12. Twenty indexed S3-coset occupancy profiles receive fresh C19 labels. Screen exact verified lifts for rate, sampled component kernel words and repeated neighbour pairs; test whether the larger nonabelian components remove the parent's weight-eight witnesses.")
FAMILY_NAME = "r020_c19xs3_12x12_coset_component_probe_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 114

def custom_ops(q):
    # C19 x (C3 semidirect C2); the C2 factor inverts C3.
    els = [(a, b, c) for c in range(2)
           for b in range(3) for a in range(19)]

    def red(g):
        return (g[0] % 19, g[1] % 3, g[2] % 2)

    def idx(g):
        a, b, c = red(g)
        return a + 19 * b + 57 * c

    def mul(g, h):
        a, b, c = red(g)
        u, v, d = red(h)
        return ((a + u) % 19, (b + (-v if c else v)) % 3,
                (c + d) % 2)

    def inv(g):
        a, b, c = red(g)
        return ((-a) % 19, (b if c else -b) % 3, c)

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _four_term_side(rng, occupancies):
    """Four distinct elements per cell with indexed S3-coset counts."""
    row = []
    for even_count in occupancies:
        even = rng.sample(range(3), even_count)
        odd = rng.sample(range(3), 4 - even_count)
        cosets = [(b, 0) for b in even] + [(b, 1) for b in odd]
        labels = rng.sample(range(19), 4)
        cell = [(a, b, c) for a, (b, c) in zip(labels, cosets)]
        rng.shuffle(cell)
        row.append(cell)
    return [row]


def _two_component_columns(hx, hz, q):
    """Extract two-column component checks from the verified LP matrices."""
    mask = (1 << q) - 1
    result = []
    for rows, stride in ((hx, 2), (hz, 1)):
        columns = [0] * (2 * q)
        for r in range(q):
            for j in range(2):
                bits = (rows[r] >> (j * stride * q)) & mask
                while bits:
                    bit = bits & -bits
                    columns[j * q + bit.bit_length() - 1] |= 1 << r
                    bits ^= bit
        result.append(columns)
    return result


def _component_probe(columns, rng):
    """Sample actual component-kernel words; this is not a distance proof."""
    words = set()
    for _ in range(3):
        pivots = {}
        for j in rng.sample(range(len(columns)), len(columns)):
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                prior = pivots.get(p)
                if prior is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= prior[0]
                word ^= prior[1]
            else:
                words.add(word)
    shortest = sorted(words, key=int.bit_count)[:24]
    minimum = min((w.bit_count() for w in shortest),
                  default=len(columns) + 1)
    light = sum(w.bit_count() <= 24 for w in shortest)
    for i, left in enumerate(shortest):
        for right in shortest[:i]:
            weight = (left ^ right).bit_count()
            if weight:
                minimum = min(minimum, weight)
                light += weight <= 24
    return minimum, light


def _repeated_neighbours(columns):
    """Count pairs of variables sharing two or more component checks."""
    repeats = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            shared = (x & y).bit_count()
            if shared >= 2:
                repeats += shared * (shared - 1) // 2
    return repeats


def _component_columns(hx, hz, q):
    """Extract both two-row component checks from the verified LP lifts."""
    mask = (1 << q) - 1
    a_cols = [0] * (5 * q)
    b_cols = [0] * (5 * q)
    for i in range(2):
        for r in range(q):
            # HX row (Arow i, Bcol 0), restricted to its A blocks.
            a_row = hx[i * 5 * q + r]
            # HZ row (Acol 0, Brow i), restricted to its B blocks.
            b_row = hz[i * q + r]
            mark = 1 << (i * q + r)
            for j in range(5):
                bits = (a_row >> (j * 5 * q)) & mask
                while bits:
                    bit = bits & -bits
                    a_cols[j * q + bit.bit_length() - 1] |= mark
                    bits ^= bit
                bits = (b_row >> (j * q)) & mask
                while bits:
                    bit = bits & -bits
                    b_cols[j * q + bit.bit_length() - 1] |= mark
                    bits ^= bit
    return a_cols, b_cols


def _component_reading(columns, rng):
    """Detect component kernel words through six; sample weight seven."""
    size = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != size:
        return 2

    pairs = set()
    for i in range(size):
        ci = columns[i]
        for j in range(i + 1, size):
            syndrome = ci ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs.add(syndrome)

    triples = set()
    for i in range(size - 2):
        ci = columns[i]
        for j in range(i + 1, size - 1):
            cij = ci ^ columns[j]
            for k in range(j + 1, size):
                syndrome = cij ^ columns[k]
                if syndrome == 0:
                    return 3
                if syndrome in pairs:
                    return 5
                if syndrome in triples:
                    return 6
                triples.add(syndrome)

    for _ in range(30000):
        i, j, k, ell = rng.sample(range(size), 4)
        syndrome = (columns[i] ^ columns[j] ^
                    columns[k] ^ columns[ell])
        if syndrome in triples:
            return 7
    return 8                    # a screened reading, not a distance proof


def _minor_grade(hx, hz, q):
    """Grade three-column relations in both verified cyclic components."""
    from itertools import combinations

    mask = (1 << q) - 1

    def rotate(x, shift):
        return ((x << shift) | (x >> (q - shift))) & mask

    def product(x, y):
        result = 0
        while y:
            bit = y & -y
            result ^= rotate(x, bit.bit_length() - 1)
            y ^= bit
        return result

    def one_side(rows, stride):
        blocks = [[(rows[i] >> (j * stride * q)) & mask
                   for j in range(5)] for i in range(2)]
        words = []
        for columns in combinations(range(5), 3):
            parts = [0] * 5
            for position, j in enumerate(columns):
                u, v = [columns[t] for t in range(3) if t != position]
                parts[j] = (product(blocks[0][u], blocks[1][v]) ^
                            product(blocks[1][u], blocks[0][v]))
            word = sum(part << (q * j) for j, part in enumerate(parts))
            if not word:
                return (0, -10000, -10000)
            words.append(word)

        shifted = []
        for word in words:
            shifted.append([
                sum(rotate((word >> (q * j)) & mask, s) << (q * j)
                    for j in range(5))
                for s in range(q)
            ])

        minimum = min(word.bit_count() for word in words)
        seven = eight = 0
        for i, word in enumerate(words):
            weight = word.bit_count()
            seven += weight <= 7
            eight += weight == 8
            for j in range(i, len(words)):
                for other in shifted[j]:
                    combined = word ^ other
                    if not combined:
                        continue
                    weight = combined.bit_count()
                    minimum = min(minimum, weight)
                    seven += weight <= 7
                    eight += weight == 8
        return minimum, -seven, -eight

    a = one_side((hx[0], hx[5 * q]), 5)
    b = one_side((hz[0], hz[q]), 1)
    return a, b


def NEW_CTX(rng, G, idx):
    """Test an indexed occupancy design using only verified matrix lifts."""
    from itertools import combinations_with_replacement

    side_profiles = list(combinations_with_replacement((1, 2, 3), 2))
    paired = [(a, b) for i, a in enumerate(side_profiles)
              for b in side_profiles[i:]]
    a_profile, b_profile = paired[1 + idx % 20]
    best, best_key = None, None

    for _ in range(9):
        a = _four_term_side(rng, a_profile)
        b = _four_term_side(rng, b_profile)
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        acols, bcols = _two_component_columns(hx, hz, G.q)
        da = _component_probe(acols, rng)
        db = _component_probe(bcols, rng)
        overlaps = (_repeated_neighbours(acols)
                    + _repeated_neighbours(bcols))
        key = (int(114 <= k <= 128), min(da[0], db[0]),
               -(da[1] + db[1]), -overlaps, k, rng.random())
        if best_key is None or key > best_key:
            best_key = key
            best = {"A": a, "B": b}
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Uniform rolled supports do not reliably avoid short witnesses: parent
# slots 1 and 2 had d2s=2 despite satisfying the row-weight cap.
# Screening only kernel words through weight four does not eliminate the
# distance-7/8 sector-1 witnesses: all five measured parent slots retained them.
# REFUTED: Zero/trinomial exchanges with short-component screening suffice
# for d30s>8: all five current-parent measured slots retained d30s=8.
# REFUTED: Mixing rank-one-parity profiles with regular and denser profiles
# makes k=179 consistent among selected codes: only two of the current
# parent's five measured slots had k=179; the other three had k=174.
# REFUTED: Independently screened cyclic order-19 rank-one components
# raise distance beyond eight: all five current measured slots had d30s=8.
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
