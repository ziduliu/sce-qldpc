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
CONCEPT = ("On nonabelian C13 semidirect_3 C9 of order 117 with A=(1,2), B=(1,2), quotient-screened four-term action-class motifs should yield five rate-0.200 codes with d30s>23.")
GROUP = ("Jump from cyclic order 19 to C13 semidirect_3 C9. The measured order-117 (1,2)x(1,2) bed leads cell 0.200 at 113.3, with selected distances 23-24; its order-39 quotient permits cheaper component screening.")
SHAPES = ("Equal (1,2)x(1,2) sides give n=5*117=585, guaranteed k=117 and rate 0.200, with equal-length two-column components. The parent's weight-eight X/Z witnesses were exclusively sector-1 Acol/Bcol, prominent on Acol1-2 and Bcol1-2 respectively.")
FILLING = ("Put four distinct terms spanning three action classes in every cell, so HX and HZ row weights are 12. Use twenty indexed, separately labelled class-occupancy motifs; reject short order-39 quotient component relations, then check full-lift rank and neighbour overlap. This tests whether the parent's weight-eight spectrum gives way to five distances above 23 without excessive rank-bound surplus.")
FAMILY_NAME = "r020_c13sd3c9_q117_12x12_quotient_class_motifs_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 117

def custom_ops(q):
    # x^13=y^9=1 and y*x*y^-1=x^3; 3 has order 3 modulo 13.
    m, r, s = 13, 9, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return m * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m) * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m) * a) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _support_profile(rng, kind):
    """Three distinct count profiles; permute columns before labelling."""
    profiles = (
        # Parity rank one: four (2,1) columns and one (0,3).
        [[2, 2, 2, 2, 0], [1, 1, 1, 1, 3]],
        # Parity rank two: rows 8/8, column degrees 3,3,4,3,3.
        [[2, 2, 2, 1, 1], [1, 1, 2, 2, 2]],
        # Parity rank two: rows 8/9, column degrees 3,3,3,4,4.
        [[2, 2, 2, 1, 1], [1, 1, 1, 3, 3]],
    )
    order = rng.sample(range(5), 5)
    rows = [[row[j] for j in order] for row in profiles[kind]]
    if rng.randrange(2):
        rows.reverse()
    return rows


def _label_grid(rng, support, motif, side):
    """Independently label cells using an indexed binomial-gap pool."""
    from itertools import combinations

    pools = tuple(combinations(range(1, 10), 6))
    preferred = pools[(17 * motif + 31 * side) % len(pools)]
    grid = []
    for row in support:
        used_gaps = set()
        labelled = []
        for count in row:
            if count == 0:
                labelled.append([])
                continue
            if count == 2:
                available = [gap for gap in preferred
                             if gap not in used_gaps]
                gap = rng.choice(available or preferred)
                start = rng.randrange(19)
                chosen = [start, (start + gap) % 19]
                used_gaps.add(gap)
            else:
                chosen = None
                for _ in range(100):
                    shifts = rng.sample(range(19), count)
                    gaps = {
                        min((x - y) % 19, (y - x) % 19)
                        for p, x in enumerate(shifts) for y in shifts[:p]
                    }
                    if len(gaps) == count * (count - 1) // 2 \
                            and not (gaps & used_gaps):
                        chosen = shifts
                        used_gaps.update(gaps)
                        break
                if chosen is None:
                    chosen = rng.sample(range(19), count)
            labelled.append([(0, shift) for shift in chosen])
        grid.append(labelled)
    return grid


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
    return (min(a[0], b[0]), a[0] + b[0],
            a[1] + b[1], a[2] + b[2])


def _quotient39():
    """Quotient by the central order-three subgroup generated by y^3."""
    els = [(a, b) for b in range(3) for a in range(13)]

    def mul(g, h):
        return ((g[0] + pow(3, g[1], 13) * h[0]) % 13,
                (g[1] + h[1]) % 3)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-pow(3, bi, 13) * g[0]) % 13, bi)

    return _G(dict(elements=els, ident=(0, 0), mul=mul, inv=inv), 39)


def _action_cell(rng, repeated, spread):
    """Four distinct elements, with action-class counts 2,1,1."""
    heights = [rng.randrange(3) for _ in range(3)]
    bs = [c + 3 * heights[c] for c in range(3)]
    bs.append(repeated + 3 * rng.choice(
        [h for h in range(3) if h != heights[repeated]]))
    for _ in range(80):
        aa = rng.sample(range(13), 4)
        gaps = {min((aa[i] - aa[j]) % 13,
                    (aa[j] - aa[i]) % 13)
                for i in range(4) for j in range(i)}
        if len(gaps) >= spread:
            break
    cell = list(zip(aa, bs))
    rng.shuffle(cell)
    return cell


def _two_component_columns(hx, hz, q):
    """Extract A and B from the scaffold's exact two-column LP lifts."""
    mask = (1 << q) - 1
    answer = []
    for rows, bases in ((hx, (0, 2 * q)), (hz, (0, q))):
        cols = [0] * (2 * q)
        for r in range(q):
            for block, base in enumerate(bases):
                bits = (rows[r] >> base) & mask
                while bits:
                    low = bits & -bits
                    cols[block * q + low.bit_length() - 1] |= 1 << r
                    bits ^= low
        answer.append(cols)
    return answer


def _quotient_grade(cols):
    """Detect component words through six and count repeated check pairs."""
    singles = set(cols)
    if 0 in singles:
        return (1, -100000)
    if len(singles) != len(cols):
        return (2, -100000)
    pairs = set()
    overlap = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            s = x ^ y
            if s in singles:
                return (3, -100000)
            if s in pairs:
                return (4, -100000)
            pairs.add(s)
            common = (x & y).bit_count()
            overlap += common * (common - 1) // 2

    triples = set()
    for i in range(len(cols) - 2):
        x = cols[i]
        for j in range(i + 1, len(cols) - 1):
            partial = x ^ cols[j]
            for k in range(j + 1, len(cols)):
                syndrome = partial ^ cols[k]
                if syndrome in pairs:
                    return (5, -overlap)
                if syndrome in triples:
                    return (6, -overlap)
                triples.add(syndrome)
    return (7, -overlap)


def _quotient_seven(cols, q):
    """Find seven-term relations with a translated pivot in either block."""
    triples = set()
    length = len(cols)
    for i in range(length - 2):
        x = cols[i]
        for j in range(i + 1, length - 1):
            partial = x ^ cols[j]
            for k in range(j + 1, length):
                triples.add(partial ^ cols[k])

    for pivot in (0, q):
        for i in range(length - 2):
            if i == pivot:
                continue
            x = cols[pivot] ^ cols[i]
            for j in range(i + 1, length - 1):
                if j == pivot:
                    continue
                partial = x ^ cols[j]
                for k in range(j + 1, length):
                    if k != pivot and (partial ^ cols[k]) in triples:
                        return True
    return False


def _full_overlap(cols):
    """Count pairs of distinct variables sharing multiple checks."""
    penalty = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            common = (x & y).bit_count()
            penalty += common * (common - 1) // 2
    return penalty


def NEW_CTX(rng, G, idx):
    """Screen an indexed action motif in the quotient, then verify full k."""
    motif = idx % 20
    signature = tuple((motif // (3 ** position)) % 3
                      for position in range(4))
    spread = 5 + (motif % 3 == 0)
    quotient = _quotient39()
    trials = []

    for _ in range(18):
        cells = {
            "A": [[_action_cell(rng, signature[j], spread)
                   for j in range(2)]],
            "B": [[_action_cell(rng, signature[2 + j], spread)
                   for j in range(2)]],
        }
        projected = {
            side: [[[(a, b % 3) for a, b in cell] for cell in row]
                   for row in cells[side]]
            for side in ("A", "B")
        }
        hx, hz, _ = lifted_matrices(
            quotient, projected["A"], projected["B"])
        ac, bc = _two_component_columns(hx, hz, quotient.q)
        ga, gb = _quotient_grade(ac), _quotient_grade(bc)
        preliminary = (min(ga[0], gb[0]), ga[0] + gb[0],
                       ga[1] + gb[1], rng.random())
        trials.append((preliminary, cells, ac, bc, ga, gb))

    shortlisted = []
    for preliminary, cells, ac, bc, ga, gb in sorted(
            trials, key=lambda item: item[0], reverse=True)[:6]:
        da = (8 if ga[0] == 7
              and not _quotient_seven(ac, quotient.q) else ga[0])
        db = (8 if gb[0] == 7
              and not _quotient_seven(bc, quotient.q) else gb[0])
        grade = (min(da, db), da + db, preliminary[2],
                 preliminary[3])
        shortlisted.append((grade, cells))

    best = best_key = None
    for quotient_grade, cells in sorted(
            shortlisted, key=lambda item: item[0], reverse=True)[:4]:
        a, b = cells["A"], cells["B"]
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        ac, bc = _two_component_columns(hx, hz, G.q)
        overlap = _full_overlap(ac) + _full_overlap(bc)
        key = (117 <= k <= 131, quotient_grade[0],
               quotient_grade[1], -overlap,
               -abs(k - 121), quotient_grade[2:])
        if best_key is None or key > best_key:
            best_key, best = key, cells
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
# REFUTED: The parent's generic minor screen reliably preserves distance
# eight on rank-one-parity fillings: eight of its ten k=179 slots had d2s=6.
# REFUTED: Indexed gap-pool rank-one fillings eliminate weight-eight
# logicals on the order-19 bed: all five current measured slots had d30s=8.
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
