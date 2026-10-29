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
CONCEPT = ("On cyclic Z19 of order 19 with A=(2,5), B=(2,5), independently screened odd-degree rank-one components with twenty paired gap spectra should give five distinct k=179, d30s>=8 codes.")
GROUP = ("Keep the measured order-19 Z19 bed fixed: n=551, and its recorded five-code score reaches 20.791, permitting a filling-only comparison.")
SHAPES = ("Keep equal (2,5)x(2,5) sides and equal-length X/Z component codes. The guaranteed k is 19*(5-2)^2=171, rate 171/551=0.3103 in cell 0.300. The current program's weight-seven and -eight X/Z witnesses occupied sector-1 Acol/Bcol blocks.")
FILLING = ("Give each side four (2,1) columns and one (0,3) column: rows total 8/7, columns total three, and LP row weight is at most 11. Twenty joint-automorphism-inequivalent binomial-gap-spectrum pairs receive fresh labels; trinomial gaps avoid the binomial gaps. Screen exact verified component checks through weight six, then select exact k=179 pairings. This targets three measured weight-seven readings without the parent's row-weight-12 supports.")
FAMILY_NAME = "r030_m1r19s1_25x25_oddcolumn_paired_sixscreen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    # Same metacyclic presentation, with the order required by the new shape.
    m, r, s = 1, 19, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _row_masks(rng, doubled, intersection):
    first = set(rng.sample(range(5), doubled))
    second = set(rng.sample(sorted(first), intersection))
    second.update(rng.sample(sorted(set(range(5)) - first),
                             doubled - intersection))
    return first, second


def _component_four_cycles(grid, q):
    """Count repeated pairs of component-code variable nodes."""
    seen = set()
    repeats = 0
    width = 5 * q
    for row in grid:
        for r in range(q):
            variables = [j * q + (r - g[1]) % q
                         for j, cell in enumerate(row) for g in cell]
            for a in range(len(variables)):
                for b in range(a + 1, len(variables)):
                    u, v = sorted((variables[a], variables[b]))
                    key = u * width + v
                    if key in seen:
                        repeats += 1
                    else:
                        seen.add(key)
    return repeats


def _has_short_kernel(rows, n):
    """Detect zero-syndrome words of weights 1 through 4."""
    signatures = [0] * n
    for i, bits in enumerate(rows):
        while bits:
            bit = bits & -bits
            signatures[bit.bit_length() - 1] |= 1 << i
            bits ^= bit
    singles = set(signatures)
    if 0 in singles or len(singles) != n:
        return True
    pairs = set()
    for a in range(n):
        for b in range(a + 1, n):
            syndrome = signatures[a] ^ signatures[b]
            if syndrome in singles or syndrome in pairs:
                return True
            pairs.add(syndrome)
    return False


def _label_grid(rng, G, support):
    grid = []
    for row in support:
        out = []
        for count in row:
            cell = []
            while len(cell) < count:
                g = G.pick(rng)
                if g not in cell:
                    cell.append(g)
            out.append(cell)
        grid.append(out)
    return grid


def _verified_component_columns(hx, hz, q):
    """Read A and B component signatures from the scaffold's exact lifts."""
    mask = (1 << q) - 1
    a_cols = [0] * (5 * q)
    b_cols = [0] * (5 * q)
    for i in range(2):
        for r in range(q):
            row = hx[i * 5 * q + r]
            check = 1 << (i * q + r)
            for j in range(5):
                bits = (row >> (j * 5 * q)) & mask
                while bits:
                    bit = bits & -bits
                    a_cols[j * q + bit.bit_length() - 1] |= check
                    bits ^= bit
    for i in range(2):
        for r in range(q):
            bits = hz[i * q + r] & ((1 << (5 * q)) - 1)
            check = 1 << (i * q + r)
            while bits:
                bit = bits & -bits
                b_cols[bit.bit_length() - 1] |= check
                bits ^= bit
    return a_cols, b_cols


def _component_reading(rng, cols):
    """Detect relations through six; sample quadruples for weight seven."""
    singles = set(cols)
    if 0 in singles:
        return 1, 1000
    if len(singles) != len(cols):
        return 2, 1000
    pairs = set()
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            if value in singles:
                return 3, 1000
            if value in pairs:
                return 4, 1000
            pairs.add(value)

    triples = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            partial = cols[i] ^ cols[j]
            for k in range(j + 1, len(cols)):
                value = partial ^ cols[k]
                if value in pairs:
                    return 5, 1000
                if value in triples:
                    return 6, 1000
                triples.add(value)

    hits = 0
    for _ in range(14000):
        indices = rng.sample(range(len(cols)), 4)
        value = cols[indices[0]] ^ cols[indices[1]]
        value ^= cols[indices[2]] ^ cols[indices[3]]
        if value in triples:
            hits += 1
    return 7, hits


def _cyclic_minor_grade(grid):
    """Grade known two-row component-kernel words, including shifted pairs."""
    from itertools import combinations

    q = 19
    mask = (1 << q) - 1

    def rotate(x, s):
        return ((x << s) | (x >> (q - s))) & mask

    def product(x, y):
        result = 0
        while y:
            bit = y & -y
            result ^= rotate(x, bit.bit_length() - 1)
            y ^= bit
        return result

    blocks = [[sum(1 << g[1] for g in cell) for cell in row]
              for row in grid]
    minors = {}
    for u, v in combinations(range(5), 2):
        minors[u, v] = (product(blocks[0][u], blocks[1][v]) ^
                        product(blocks[1][u], blocks[0][v]))

    words = []
    for u, v, w in combinations(range(5), 3):
        pieces = ((u, minors[v, w]), (v, minors[u, w]),
                  (w, minors[u, v]))
        word = sum(bits << (j * q) for j, bits in pieces)
        if not word:
            return (0, -100000, -100000)
        words.append(word)

    def shifted(word, shift):
        return sum(rotate((word >> (j * q)) & mask, shift) << (j * q)
                   for j in range(5))

    minimum = min(word.bit_count() for word in words)
    below_eight = at_eight = 0
    for i, word in enumerate(words):
        for j in range(i, len(words)):
            for s in range(q):
                other = shifted(words[j], s)
                combined = word ^ other
                if not combined:
                    continue
                weight = combined.bit_count()
                minimum = min(minimum, weight)
                below_eight += weight < 8
                at_eight += weight == 8
    return minimum, -below_eight, -at_eight


def _rankone_spectra():
    """Cache twenty distinct paired four-gap orbits under Z19 multipliers."""
    from functools import lru_cache
    from itertools import combinations

    @lru_cache(None)
    def build():
        spectra = list(combinations(range(1, 10), 4))
        images = {
            (gaps, unit):
            tuple(sorted(min(unit * d % 19, (-unit * d) % 19)
                         for d in gaps))
            for gaps in spectra for unit in range(1, 10)
        }
        representatives = {}
        for i, ag in enumerate(spectra):
            for bg in spectra[i:]:
                orbit = min(
                    tuple(sorted((images[ag, unit], images[bg, unit])))
                    for unit in range(1, 10)
                )
                representatives.setdefault(orbit, (ag, bg))
        keys = sorted(representatives)
        return tuple(representatives[
            keys[(2 * t + 1) * len(keys) // 40]] for t in range(20))

    return build()


def _rankone_side(rng, gaps):
    """Four distinct-gap binomials and a disjoint-gap trinomial."""
    from itertools import combinations

    rows = [[[] for _ in range(5)] for _ in range(2)]
    positions = rng.sample(range(5), 5)
    assigned = list(gaps)
    rng.shuffle(assigned)
    for j, gap in zip(positions[:4], assigned):
        start = rng.randrange(19)
        signed = gap if rng.randrange(2) else -gap
        rows[0][j] = [(0, start), (0, (start + signed) % 19)]
        rows[1][j] = [(0, rng.randrange(19))]

    def acceptable(shifts):
        distances = {
            min((x - y) % 19, (y - x) % 19)
            for p, x in enumerate(shifts) for y in shifts[:p]
        }
        return len(distances) == 3 and not (distances & set(gaps))

    triple = None
    for _ in range(70):
        candidate = rng.sample(range(19), 3)
        if acceptable(candidate):
            triple = candidate
            break
    if triple is None:
        options = [candidate for candidate in
                   combinations(range(19), 3) if acceptable(candidate)]
        triple = rng.choice(options)
    rows[1][positions[4]] = [(0, shift) for shift in triple]
    if rng.randrange(2):
        rows.reverse()
    return rows


def _even_component_grade(columns):
    """Exclude component words through six using verified check signatures.

    Every constructed component column has odd check degree three.
    Its kernel therefore has no odd-weight words.
    """
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    for i, x in enumerate(columns):
        for y in columns[:i]:
            value = x ^ y
            if value in pairs:
                return 4
            pairs.add(value)

    triples = set()
    for i in range(len(columns) - 2):
        x = columns[i]
        for j in range(i + 1, len(columns) - 1):
            partial = x ^ columns[j]
            for k in range(j + 1, len(columns)):
                value = partial ^ columns[k]
                if value in triples:
                    return 6
                triples.add(value)
    return 8


def NEW_CTX(rng, G, idx):
    """Screen each side independently, then verify newly paired full lifts."""
    ag, bg = _rankone_spectra()[idx % 20]
    a_pool, b_pool = [], []
    for _ in range(54):
        a = _rankone_side(rng, ag)
        b = _rankone_side(rng, bg)
        qa, qb = _cyclic_minor_grade(a), _cyclic_minor_grade(b)
        hx, hz, _ = lifted_matrices(G, a, b)
        a_pool.append((qa, rng.random(), a, hx, hz))
        b_pool.append((qb, rng.random(), b, hx, hz))

    def shortlist(pool, side):
        pool.sort(key=lambda item: item[:2], reverse=True)
        selected = []
        for minor, tie, grid, hx, hz in pool[:17]:
            ac, bc = _verified_component_columns(hx, hz, G.q)
            grade = _even_component_grade(ac if side == "A" else bc)
            selected.append((grade, minor, tie, grid))
        selected.sort(key=lambda item: item[:3], reverse=True)
        return selected[:5]

    candidates_a = shortlist(a_pool, "A")
    candidates_b = shortlist(b_pool, "B")
    best_key = best_pair = None
    for da, ma, ta, a in candidates_a:
        for db, mb, tb, b in candidates_b:
            k = lifted_k(G, a, b)
            key = (int(k == 179), min(da, db), da + db,
                   min(ma[0], mb[0]), ma[1] + mb[1],
                   ma[2] + mb[2], -abs(k - 179), ta + tb)
            if best_key is None or key > best_key:
                best_key, best_pair = key, (a, b)
    return {"A": best_pair[0], "B": best_pair[1]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Uniform rolled supports do not reliably avoid short witnesses: parent
# slots 1 and 2 had d2s=2 despite satisfying the row-weight cap.
# REFUTED: Screening balanced supports for four-cycles, full rank and
# kernels through weight four does not suffice for d30s>=9: the current
# parent's five measured distances were 8, 8, 8, 7 and 7.
# REFUTED: Screening mixed balanced/asymmetric components through weight
# six and sampling weight seven suffices for d30s>=9: all five measured
# current-parent slots had d30s=8 and k-kb=0.
# REFUTED: Parity-aligned row-eight supports screened by cyclic minors
# retain d30s>=8 across the selected codes: the current program measured
# three distance-seven codes and only two distance-eight codes.
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
