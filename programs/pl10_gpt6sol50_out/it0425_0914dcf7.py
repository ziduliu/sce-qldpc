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
CONCEPT = ("In cyclic Z13 of order 13 with A=(2,5) and B=(3,8), denser A-binomial coverage with B columns held at weight three should reduce worst-basis circuit failures.")
GROUP = ("Keep cyclic Z13 and order 13 for a direct filling-only comparison with the current bed's measured 4.7762 fitness.")
SHAPES = ("Keep (2,5)x(3,8): n=598, guaranteed k>=195 and rate>=0.326 in cell 0.350. The unequal A and B component lengths are 65 and 104. The parent's weight-eight witnesses and enriched failures spread across sector-1 Acol/Bcol blocks; no Brow or check block was a persistent hotspot.")
FILLING = ("Cross five indexed A-density/B-exchange motifs with four A-gap palettes. Give A six to eight binomials, while balanced B exchanges keep each B row at eight terms and every B column at three. Screen verified component relations and k, prioritizing B defects before A defects. Test whether this changes the roughly 100-failure worst-basis caps, including X-basis caps after 3250-11000 shots; distance remains only a selection screen.")
FAMILY_NAME = "r035_z13_25x38_dense_A_balanced_B_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Additive cyclic group Z13, encoded as pairs.
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
B_SHAPE = (3, 8)               # (rb, cb); n = 46 * 13 = 598


def _verified_component_columns(hx, hz, q):
    """Read both components from the scaffold's exact LP matrices."""
    mask = (1 << q) - 1
    a = [0] * (5 * q)
    b = [0] * (8 * q)
    for i in range(2):
        for r in range(q):
            row = hx[(i * 8) * q + r]  # X check at B-column zero
            for j in range(5):
                bits = (row >> ((j * 8) * q)) & mask
                while bits:
                    bit = bits & -bits
                    a[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    bits ^= bit
    for alpha in range(3):
        for r in range(q):
            row = hz[alpha * q + r]  # Z check at A-column zero
            for beta in range(8):
                bits = (row >> (beta * q)) & mask
                while bits:
                    bit = bits & -bits
                    b[beta * q + bit.bit_length() - 1] |= (
                        1 << (alpha * q + r))
                    bits ^= bit
    return a, b


def _component_flaws(columns):
    """Count short dependencies and repeated check-pairs."""
    singles = {}
    for col in columns:
        singles[col] = singles.get(col, 0) + 1
    twos = sum(v * (v - 1) // 2 for v in singles.values())
    pairs = {}
    threes = fours = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            syndrome = left ^ columns[j]
            if not syndrome:
                continue
            threes += singles.get(syndrome, 0)
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = (i, j)
            elif i not in previous and j not in previous:
                fours += 1

    check_pairs = {}
    cycles = 0
    for col in columns:
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length() - 1)
            col ^= bit
        for a in range(len(checks)):
            for b in range(a + 1, len(checks)):
                pair = (checks[a], checks[b])
                cycles += check_pairs.get(pair, 0)
                check_pairs[pair] = check_pairs.get(pair, 0) + 1
    return twos, threes, fours, cycles


def _b_voltages(rng):
    """Choose eight triples with distinct differences for each row pair."""
    for _ in range(100):
        options = [(u, v) for u in range(13) for v in range(13)]
        rng.shuffle(options)
        chosen = []
        used_u, used_v, used_difference = set(), set(), set()
        for u, v in options:
            difference = (u - v) % 13
            if (u in used_u or v in used_v or
                    difference in used_difference):
                continue
            chosen.append((u, v))
            used_u.add(u)
            used_v.add(v)
            used_difference.add(difference)
            if len(chosen) == 8:
                rng.shuffle(chosen)
                return chosen
    raise ValueError("could not construct eight B voltage triples")


def _six_relations(columns, cap=3000):
    """Count disjoint triple collisions, a component weight-six signal."""
    seen = {}
    defects = 0
    for i in range(len(columns) - 2):
        for j in range(i + 1, len(columns) - 1):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, len(columns)):
                syndrome = partial ^ columns[k]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = (i, j, k)
                elif i not in old and j not in old and k not in old:
                    defects += 1
                    if defects >= cap:
                        return cap
    return defects


def _draw_profile(rng, idx):
    """Indexed A-density and B-exchange designs with fresh voltages."""
    motif, gap_profile = idx % 5, (idx // 5) % 4
    # Each pair is (A binomials, disjoint three-column B exchanges).
    a_doubles, exchanges = (
        (6, 1), (7, 1), (8, 1), (6, 2), (7, 2)
    )[motif]

    sa = [[1] * 5 for _ in range(2)]
    columns = rng.sample(range(5), 5)
    first = rng.randrange(2)
    for t, j in enumerate(columns):
        sa[(t + first) % 2][j] = 2
    while sum(count == 2 for row in sa for count in row) < a_doubles:
        available = [(i, j) for i in range(2) for j in range(5)
                     if sa[i][j] == 1 and sum(sa[i]) < 9]
        i, j = rng.choice(available)
        sa[i][j] = 2

    # A balanced exchange replaces three monomials by a zero,
    # a binomial and a monomial in each row. Every B column retains
    # three terms, including columns touched by two exchanges.
    sb = [[1] * 8 for _ in range(3)]
    selected = rng.sample(range(8), 3 * exchanges)
    for c in range(exchanges):
        triplet = selected[3 * c:3 * c + 3]
        for row in range(3):
            sb[row][triplet[row]] = 0
            sb[row][triplet[(row + 1 + c) % 3]] = 2

    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) == 8 for row in sb)
    assert all(sum(sb[i][j] for i in range(3)) == 3
               for j in range(8))

    palettes = (
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (1, 2, 4, 5, 6),
    )
    A = []
    for row in sa:
        gaps = rng.sample(palettes[gap_profile],
                          sum(count == 2 for count in row))
        rng.shuffle(gaps)
        cells = []
        for count in row:
            start = rng.randrange(13)
            cell = [(start, 0)]
            if count == 2:
                gap = gaps.pop() * rng.choice((-1, 1))
                cell.append(((start + gap) % 13, 0))
            cells.append(cell)
        A.append(cells)

    points = _b_voltages(rng)
    B = [[None] * 8 for _ in range(3)]
    for j, (u, v) in enumerate(points):
        origin = rng.randrange(13)
        offsets = (origin, (origin + u) % 13, (origin + v) % 13)
        for row in range(3):
            count = sb[row][j]
            if count == 0:
                B[row][j] = []
            elif count == 1:
                B[row][j] = [(offsets[row], 0)]
            else:
                gap = rng.randrange(1, 13)
                B[row][j] = [(offsets[row], 0),
                             ((offsets[row] + gap) % 13, 0)]
    return A, B


def NEW_CTX(rng, G, idx):
    """Screen indexed designs on exact lifts, then verify their rank."""
    shortlist = []
    for _ in range(40 + idx % 7):
        A, B = _draw_profile(rng, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _verified_component_columns(hx, hz, G.q)
        af, bf = _component_flaws(ac), _component_flaws(bc)
        # Preserve the B component first: the new density is in A.
        short = (bf[0], bf[1], bf[2], af[0], af[1], af[2],
                 bf[3] + af[3])
        shortlist.append((short, rng.random(), A, B,
                          hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[:2])

    best_key = best = None
    for short, tie, A, B, hx, hz, n, ac, bc in shortlist[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        six_b = _six_relations(bc)
        six_a = _six_relations(ac)
        choice = (not 195 <= k <= 224, short[:6],
                  six_b, six_a, short[6],
                  abs(k - 195), tie)
        if best_key is None or choice < best_key:
            best_key, best = choice, {"A": A, "B": B}
    if best is None:
        raise ValueError("no positive-k dense-A filling")
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
# REFUTED: Distinct B voltages with occasional extra terms suffice on
# Z13 (2,5)x(3,8): the parent scored 4.8521; all five measured codes
# reached the failure cap in both bases and had weight-six witnesses.
# REFUTED: The balanced B exchanges alone improve the Z13 bed: fitness
# fell from 4.8521 to 4.7762; slots 44 and 49, both with two exchanges,
# reached the X-basis failure cap after 4500 and 3250 shots.
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
