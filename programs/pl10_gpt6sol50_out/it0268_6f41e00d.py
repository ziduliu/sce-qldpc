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
CONCEPT = ("In cyclic Z13 of order 13 with A=(2,5), B=(3,8), distinct B-row voltage differences and screened A binomials should reduce worst-basis circuit failures.")
GROUP = ("Keep the parent's trivial-action metacyclic Z13 rule and order for a filling-only test. This n=598 bed scored 4.6001, leaving its element model directly testable.")
SHAPES = ("Keep (2,5)x(3,8): guaranteed k>=195 of n=598, rate>=0.3261 in the 0.350 cell; its unequal component lengths are 65 and 104. The parent's weight-six witnesses occupied sector-1 Acol/Bcol blocks. Slots 22 and 31 had 106/2500 and 108/4250 X-basis failures, without a persistent check-block hotspot.")
FILLING = ("Most indexed designs use six to eight A binomials and monomial B cells. Give each B column a triple of row-relative voltages with distinct differences for every pair of rows; retain a few lighter or B-augmented profiles. Screen exact lifted component relations, then verified rank and Tanner cycles. Test whether the parent's frequent failed shots and widespread logical flips decrease, rather than scoring distance itself.")
FAMILY_NAME = "r035_m13r1s1_25x38_distinct_B_voltages_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Same metacyclic rule, now Z_13 x| Z_1 with trivial action.
    m, r, s = 13, 1, 1
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
B_SHAPE = (3, 8)               # (rb, cb); n = 46 * 13 = 598


def _four_cycles(rows):
    """Count pairs of checks sharing two qubits in the verified lift."""
    total = 0
    for i, row in enumerate(rows):
        for earlier in rows[:i]:
            overlap = (row & earlier).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _support_profile(rng, idx):
    """Twenty indexed density motifs; every check-row sum is at most 12."""
    motif = idx % 20
    if motif < 12:
        left, right = 3, 3
    elif motif < 16:
        left, right = 3, 4
    elif motif == 16:
        left, right = 4, 4
    else:
        left, right = 2, 3

    first = set(rng.sample(range(5), left))
    remaining = [j for j in range(5) if j not in first]
    second = set(remaining)
    second.update(rng.sample(sorted(first), right - len(remaining)))
    sa = [[1 + (j in first) for j in range(5)],
          [1 + (j in second) for j in range(5)]]
    sb = [[1] * 8 for _ in range(3)]

    # Only the five-binomial A profile admits a B-row extra:
    # its A columns each weigh three and its A rows weigh at most eight.
    if motif >= 18:
        positions = rng.sample(range(8), motif - 17)
        for row, column in enumerate(positions):
            sb[row][column] = 2

    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _draw_A(rng, sa, idx):
    """Spread A's undirected binomial gaps before choosing cell origins."""
    number = sum(count == 2 for row in sa for count in row)
    gaps = list(range(1, 7))
    rng.shuffle(gaps)
    if number < 6:
        gaps = gaps[:number]
    elif number > 6:
        gaps.extend(rng.sample(range(1, 7), number - 6))
    rng.shuffle(gaps)

    A = []
    for row in sa:
        filled = []
        for count in row:
            start = rng.randrange(13)
            if count == 1:
                filled.append([(start, 0)])
            else:
                gap = gaps.pop()
                if rng.randrange(2):
                    gap = -gap
                filled.append([(start, 0),
                               ((start + gap) % 13, 0)])
        A.append(filled)
    return A


def _draw_B(rng, sb):
    """Make eight voltage triples with distinct row-pair differences.

    A monomial B column can be column-gauged to voltages (0,u,v).
    Distinct u, v and u-v across columns avoid repeated differences
    between any chosen pair of its three protograph rows.
    """
    q = 13
    points = None
    for _ in range(30):
        available = [(u, v) for u in range(q) for v in range(q)]
        rng.shuffle(available)
        chosen = []
        used_u, used_v, used_difference = set(), set(), set()
        for u, v in available:
            difference = (u - v) % q
            if (u in used_u or v in used_v
                    or difference in used_difference):
                continue
            chosen.append((u, v))
            used_u.add(u)
            used_v.add(v)
            used_difference.add(difference)
            if len(chosen) == 8:
                points = chosen
                break
        if points is not None:
            break
    if points is None:
        raise ValueError("could not construct eight distinct B voltages")

    rng.shuffle(points)
    B = [[None] * 8 for _ in range(3)]
    for j, (u, v) in enumerate(points):
        origin = rng.randrange(q)
        starts = (origin, (origin + u) % q, (origin + v) % q)
        for row in range(3):
            terms = [(starts[row], 0)]
            if sb[row][j] == 2:
                step = rng.randrange(1, q)
                terms.append(((starts[row] + step) % q, 0))
            B[row][j] = terms
    return B


def _component_columns(hx, hz, q):
    """Read both component column syndromes from the verified LP lift."""
    a = [0] * (5 * q)
    b = [0] * (8 * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            row = hx[(i * 8) * q + r]  # B-column zero
            for j in range(5):
                bits = (row >> ((j * 8) * q)) & mask
                while bits:
                    bit = bits & -bits
                    a[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    bits ^= bit
    for alpha in range(3):
        for r in range(q):
            row = hz[alpha * q + r]  # A-column zero
            for beta in range(8):
                bits = (row >> (beta * q)) & mask
                while bits:
                    bit = bits & -bits
                    b[beta * q + bit.bit_length() - 1] |= (
                        1 << (alpha * q + r))
                    bits ^= bit
    return a, b


def _pair_defects(columns):
    """Count actual weight-two/three/four component relations."""
    singles = set(columns)
    seen = {}
    twos = threes = fours = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            syndrome = left ^ columns[j]
            twos += syndrome == 0
            threes += syndrome in singles
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = (i, j)
            elif i not in old and j not in old:
                fours += 1
    return twos, threes, fours


def _six_defects(columns, cap=1000):
    """Count disjoint repeated triple syndromes, up to a screening cap."""
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


def NEW_CTX(rng, G, idx):
    """Screen a slot-specific support motif and fresh B-voltage designs."""
    sa, sb = _support_profile(rng, idx)
    shortlist = []
    for _ in range(16 + idx % 5):
        A = _draw_A(rng, sa, idx)
        B = _draw_B(rng, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        ar, br = _pair_defects(ac), _pair_defects(bc)
        key = (ar[0] + br[0], ar[1] + br[1],
               max(ar[2], br[2]), ar[2] + br[2],
               rng.random())
        shortlist.append((key, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[0])

    best = None
    for key, A, B, hx, hz, n, ac, bc in shortlist[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 195 <= k <= 224:
            continue
        six_a = _six_defects(ac)
        six_b = _six_defects(bc)
        quality = (key[:4], max(six_a, six_b), six_a + six_b,
                   _four_cycles(hx) + _four_cycles(hz),
                   k - 195, key[4])
        if best is None or quality < best[0]:
            best = (quality, {"A": A, "B": B})
    if best is None:
        raise ValueError("no positive-k target-cell filling")
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without redrawing its labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Singleton-heavy rolled labels suffice on this bed: refuted by the seed's
# 4.0553 fitness, weight-5 X-type witnesses, and 109/750 Z-basis failures
# in measured slot 50.
# A mixed-coset (1,3)x(2,5) filling by itself removes the short
# sector-1 failure mode: refuted by the measured D17 bed's 3.0007
# fitness and X-basis failures reaching the cap after 250 shots.
# Rank-excess and Tanner-four-cycle selection alone make this Z13 bed
# reliably low-failure: refuted by the parent's 4.6001 fitness; its
# measured slots 22 and 31 had 106/2500 and 108/4250 X-basis failures.
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
