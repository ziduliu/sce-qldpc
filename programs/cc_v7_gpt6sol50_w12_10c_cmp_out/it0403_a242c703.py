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
CONCEPT = ("On cyclic C46 of order 46 with A=B=(3,2), independently "
           "screened, coset-stratified component fillings will produce "
           "five distance readings of at least 29 in cell 0.100.")
GROUP = ("Jump from C37 to C46. The measured order-46 (3,2)x(3,2) "
         "frontier has five n=598, k=46, d30s=28 codes; exceeding "
         "that reading by one is the falsifiable target.")
SHAPES = ("Equal (3,2)x(3,2) sides give n=13*46=598, guaranteed "
          "k>=46 and rate>=0.077 in cell 0.100. Their equal-length "
          "components each have three variable blocks. The C37 "
          "parent's lightest witnesses used sector-one Acol/Bcol "
          "blocks, whereas measured order-46 witnesses used "
          "sector-two Arow/Brow blocks; screen the latter.")
FILLING = ("Cross five A and four B support motifs, each with row "
           "totals at most five and column totals seven: HX and HZ "
           "weigh at most 12. Independently vary C2-parity occupancy "
           "of distinct C46 labels. Screen sector-two component "
           "minors extracted from verified lifts and translated "
           "combinations for weight-28 witnesses, then check exact "
           "k above the guaranteed bound of 46.")
FAMILY_NAME = "r010_c46_32x32_sector2_coset_minor_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # C46; tuple exponent t indexes the cyclic regular representation.
    els = [(t, 0) for t in range(q)]

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

A_SHAPE = (3, 2)               # (ra, ca)
B_SHAPE = (3, 2)               # (rb, cb)


def _counts(idx):
    """Five A motifs crossed with four B motifs; each column sums to seven."""
    profiles = (
        ((4, 1), (2, 3), (1, 3)),
        ((3, 2), (2, 3), (2, 2)),
        ((3, 2), (3, 2), (1, 3)),
        ((2, 3), (3, 2), (2, 2)),
        ((2, 2), (4, 1), (1, 4)),
    )
    A = [list(row) for row in profiles[idx % 5]]
    B = [list(row) for row in profiles[(idx // 5) % 4]]
    assert _grid_weights(A, B) <= 12
    assert _no_dead_lines(A, B)
    return A, B


def _labels(rng, counts, mode):
    """Fresh supports, with an indexed distribution across C46 parity cosets."""
    grid = []
    used = set()
    for row in counts:
        labelled = []
        for count in row:
            best = None
            best_penalty = 100
            for _ in range(28):
                if count >= 3 and mode:
                    even = (2 if mode == 1 else
                            (count - 1 if mode == 2 else 1))
                    even = min(even, count)
                    values = (rng.sample(range(0, 46, 2), even) +
                              rng.sample(range(1, 46, 2), count - even))
                elif count == 2 and mode == 1:
                    values = [rng.choice(range(0, 46, 2)),
                              rng.choice(range(1, 46, 2))]
                else:
                    values = rng.sample(range(46), count)
                gaps = [
                    min((x - y) % 46, (y - x) % 46)
                    for p, x in enumerate(values) for y in values[:p]
                ]
                penalty = (len(gaps) - len(set(gaps))
                           + len(set(gaps) & used))
                if penalty < best_penalty:
                    best, best_penalty = values, penalty
                if penalty == 0:
                    break
            used.update(
                min((x - y) % 46, (y - x) % 46)
                for p, x in enumerate(best) for y in best[:p])
            labelled.append([(x, 0) for x in best])
        grid.append(labelled)
    return grid


def _sector_two_columns(hx, hz, q, side):
    """Extract actual two-check, three-variable sector-two components."""
    n1 = 4 * q
    rows = hz if side == "A" else hx
    columns = [0] * (3 * q)
    mask = (1 << q) - 1
    for check_block in range(2):
        for r in range(q):
            # HZ: fix Brow 0, vary Acol j. HX: fix Arow 0,
            # vary Bcol beta. Both choices retain three sector-2 blocks.
            check_row = (check_block * 3 * q + r
                         if side == "A" else check_block * q + r)
            row = rows[check_row]
            for j in range(3):
                base = n1 + (3 * j * q if side == "A" else j * q)
                bits = (row >> base) & mask
                while bits:
                    bit = bits & -bits
                    columns[j * q + bit.bit_length() - 1] |= (
                        1 << (check_block * q + r))
                    bits ^= bit
    return columns


def _component_merit(columns, q):
    """Screen verified component kernels and translated minor combinations."""
    mask = (1 << q) - 1
    polynomials = [
        [(columns[j * q] >> (i * q)) & mask for j in range(3)]
        for i in range(2)
    ]

    def rotate(x, shift):
        shift %= q
        return ((x << shift) | (x >> (q - shift))) & mask

    def product(x, y):
        answer = 0
        while y:
            bit = y & -y
            answer ^= rotate(x, bit.bit_length() - 1)
            y ^= bit
        return answer

    def verified(parts):
        word = sum(part << (j * q) for j, part in enumerate(parts))
        syndrome = 0
        bits = word
        while bits:
            bit = bits & -bits
            syndrome ^= columns[bit.bit_length() - 1]
            bits ^= bit
        return word if word and not syndrome else 0

    parts = [
        product(polynomials[0][1], polynomials[1][2])
        ^ product(polynomials[1][1], polynomials[0][2]),
        product(polynomials[0][0], polynomials[1][2])
        ^ product(polynomials[1][0], polynomials[0][2]),
        product(polynomials[0][0], polynomials[1][1])
        ^ product(polynomials[1][0], polynomials[0][1]),
    ]
    words = []
    for _ in range(5):
        word = verified(parts)
        if word:
            words.append(word)
        if any(part.bit_count() & 1 for part in parts):
            break
        divided = []
        for part in parts:
            previous = value = 0
            for position in range(1, q):
                previous ^= (part >> position) & 1
                value |= previous << position
            divided.append(value)
        if divided == parts:
            break
        parts = divided

    if not words:
        return (0, -100000, -100000)

    minimum = 3 * q + 1
    light28 = light30 = 0

    def observe(word):
        nonlocal minimum, light28, light30
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            light28 += weight <= 28
            light30 += weight <= 30

    for word in words:
        observe(word)
        shifted = [
            sum(rotate((word >> (j * q)) & mask, s) << (j * q)
                for j in range(3))
            for s in range(q)
        ]
        for other in shifted[1:]:
            observe(word ^ other)
        # Fix one translate at zero; inspect all three-word sums.
        for s in range(1, q):
            for t in range(s + 1, q):
                observe(word ^ shifted[s] ^ shifted[t])

    for i, left in enumerate(words):
        for right in words[i + 1:]:
            for s in range(q):
                translated = sum(
                    rotate((right >> (j * q)) & mask, s) << (j * q)
                    for j in range(3))
                observe(left ^ translated)

    return minimum, -light28, -light30


def NEW_CTX(rng, G, idx):
    """Select independently screened sector-two components of one motif."""
    sa, sb = _counts(idx)
    anchor_a = _labels(rng, sa, idx % 4)
    anchor_b = _labels(rng, sb, (idx // 5) % 4)
    options = {"A": [], "B": []}

    for side in ("A", "B"):
        for trial in range(38):
            grid = _labels(rng, sa if side == "A" else sb,
                           (idx + trial // 10 + (side == "B")) % 4)
            A = grid if side == "A" else anchor_a
            B = grid if side == "B" else anchor_b
            hx, hz, _ = lifted_matrices(G, A, B)
            columns = _sector_two_columns(hx, hz, G.q, side)
            merit = _component_merit(columns, G.q)
            options[side].append((merit, -trial, grid))
        options[side].sort(key=lambda entry: entry[:2], reverse=True)

    best = None
    for am, _, A in options["A"][:4]:
        for bm, _, B in options["B"][:4]:
            k = lifted_k(G, A, B)
            predicted = min(am[0], bm[0], 31)
            grade = (46 <= k < 74, k * predicted * predicted,
                     predicted, am, bm)
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
# REFUTED: Screening only component relations through weight four with
# 18 label trials routinely removes the lightest logicals on this bed:
# the parent measured distances 16, 17, 18, 17, 17, with lightest X
# witnesses in sector-1 Acol/Bcol blocks.
# REFUTED: Selecting C37 labels by translated component minors yields
# five distance-18 codes: the current program measured 18, 18, 18,
# 17, 17; all its lightest reported witnesses remained in sector one.
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
