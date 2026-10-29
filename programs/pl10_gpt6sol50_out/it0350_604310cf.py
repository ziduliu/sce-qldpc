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
CONCEPT = ("In cyclic Z20 of order 20 with A=(2,5), B=(2,5), a parity-rank-one monomial splice should reduce worse-basis circuit failures while retaining rate near 0.350.")
GROUP = ("Keep cyclic Z20 and order 20 for a filling-only test. This measured bed reached 5.2; the current zero/binomial filling scored 4.9643, leaving a direct comparison for different component parity and gate counts.")
SHAPES = ("Keep equal (2,5)x(2,5): n=580, guaranteed k>=180, and two 100-column components. Screen verified k for 189..217, the 0.350 cell. Four of the parent's five measured slots were X-limited; their enriched Acol/Bcol data blocks differed, and no check block was a persistent hotspot.")
FILLING = ("Keep A even-supported but give B shared singleton columns in its two rows. Separate the singleton columns' relative Z20 voltages by a generator to avoid their shortest mutual cycle. Cross five support motifs with four gap-order profiles, retaining row weight <=12; screen verified rank, short relations and Tanner cycles. Test failed-shot rates and decoder difficulty, not the distance proxy.")
FAMILY_NAME = "r035_z20_25x25_parity_one_monomial_splice_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Additive Z20, encoded as pairs.
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
B_SHAPE = (2, 5)               # (rb, cb); n = 29 * 20 = 580


def _four_cycles(rows):
    """Count Tanner four-cycles in the verified check rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            shared = (row & other).bit_count()
            total += shared * (shared - 1) // 2
    return total


def _component_columns(hx, hz, q):
    """Extract component columns using the evaluator's actual lift actions."""
    ca, cb, rb = A_SHAPE[1], B_SHAPE[1], B_SHAPE[0]
    a = []
    for j in range(ca):
        for t in range(q):
            mask = 0
            for i in range(A_SHAPE[0]):
                for r in range(q):
                    row = hx[i * cb * q + r]     # B-column beta = 0
                    mask |= ((row >> (j * cb * q + t)) & 1) << (i * q + r)
            a.append(mask)
    b = []
    for beta in range(cb):
        for t in range(q):
            mask = 0
            for alpha in range(rb):
                for r in range(q):
                    row = hz[alpha * q + r]     # A-column j = 0
                    mask |= ((row >> (beta * q + t)) & 1) << (alpha * q + r)
            b.append(mask)
    return a, b


def _short_relations(cols):
    """Count genuine component kernel words of weights two through four."""
    singles = set(cols)
    pairs = {}
    counts = [int(0 in singles), 0, 0, 0]
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            value = left ^ cols[j]
            counts[1] += value == 0
            counts[2] += value in singles
            previous = pairs.get(value)
            if previous is not None:
                counts[3] += sum(i not in pair and j not in pair
                                 for pair in previous)
                previous.append((i, j))
            else:
                pairs[value] = [(i, j)]
    return tuple(counts)


def _six_relations(cols, cap=400):
    """Capped disjoint triple collisions: a screening signal, not distance."""
    seen = {}
    sixes = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            partial = cols[i] ^ cols[j]
            for k in range(j + 1, len(cols)):
                value = partial ^ cols[k]
                previous = seen.get(value)
                if previous is None:
                    seen[value] = (i, j, k)
                elif i not in previous and j not in previous \
                        and k not in previous:
                    sixes += 1
                    if sixes >= cap:
                        return cap
    return sixes


def _counts(rng, motif):
    """Five parity-one support motifs with no dead protograph lines."""
    sa = [[2] * 5 for _ in range(2)]
    sb = [[2] * 5 for _ in range(2)]

    if motif == 1:
        # Disjoint zero pairs lower A's gate count.
        missing = rng.sample(range(5), 4)
        for i in range(2):
            for j in missing[2 * i:2 * i + 2]:
                sa[i][j] = 0
    else:
        missing = rng.sample(range(5), 2)
        sa[0][missing[0]] = 0
        sa[1][missing[1]] = 0

    shared = rng.sample(range(5), 2)
    for i in range(2):
        for j in shared:
            sb[i][j] = 1

    if motif == 2:
        # One shared singleton and different zero columns: seven
        # terms per B row, with the same odd-parity pattern in both.
        for j in shared[1:]:
            sb[0][j] = sb[1][j] = 2
        others = [j for j in range(5) if j != shared[0]]
        absent = rng.sample(others, 2)
        sb[0][absent[0]] = 0
        sb[1][absent[1]] = 0
    elif motif == 3:
        # Exchange two doubles for a four-term/zero pair, keeping
        # B-column weight at most four.
        others = [j for j in range(5) if j not in shared]
        quad, zero = rng.sample(others, 2)
        sb[0][quad], sb[1][quad] = 4, 0
        sb[0][zero] = 0
    elif motif == 4:
        # The same exchange on A; align its opposite-row zero.
        quad = missing[1]
        extra = rng.choice([j for j in range(5)
                            if j not in missing])
        sa[0][quad], sa[0][extra] = 4, 0

    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _draw(rng, G, supports, phase, side):
    """Draw full cells, with a unit-order separation between B monomials."""
    grid = []
    odds = (1, 3, 7, 9)
    evens = (2, 4, 6, 8)
    for row_index, row in enumerate(supports):
        doubles = sum(count == 2 for count in row)
        even_count = min(doubles - 1, (phase + row_index) % 4)
        if side == "A":
            even_count = min(doubles - 1, phase)
        gaps = (rng.sample(evens, even_count)
                + rng.sample(odds, doubles - even_count))
        rng.shuffle(gaps)
        filled = []
        for count in row:
            if count == 0:
                filled.append([])
                continue
            start = rng.randrange(G.q)
            if count == 1:
                filled.append([(start, 0)])
            elif count == 2:
                step = gaps.pop() * rng.choice((-1, 1))
                filled.append([(start, 0),
                               ((start + step) % G.q, 0)])
            else:
                offsets = rng.sample(range(1, G.q), count - 1)
                filled.append([(start, 0)] +
                              [((start + d) % G.q, 0)
                               for d in offsets])
        grid.append(filled)

    if side == "B":
        common = [j for j in range(5)
                  if supports[0][j] == supports[1][j] == 1]
        if len(common) == 2:
            j, h = common
            first = (grid[1][j][0][0] - grid[0][j][0][0]) % G.q
            unit_gap = rng.choice((1, 3, 7, 9, 11, 13, 17, 19))
            grid[1][h] = [((grid[0][h][0][0] + first
                            + unit_gap) % G.q, 0)]
    return grid


def NEW_CTX(rng, G, idx):
    """Screen an indexed support motif and gap-order profile on exact lifts."""
    motif, phase = idx % 5, (idx // 5) % 4
    sa, sb = _counts(rng, motif)
    finalists = []
    for _ in range(16 + idx % 5):
        A = _draw(rng, G, sa, phase, "A")
        B = _draw(rng, G, sb, phase, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        ar, br = _short_relations(ac), _short_relations(bc)
        short = (ar[0] + br[0], ar[1] + br[1],
                 ar[2] + br[2], max(ar[3], br[3]),
                 ar[3] + br[3])
        finalists.append((short, rng.random(), A, B, hx, hz, n,
                          ac, bc))
    finalists.sort(key=lambda item: item[:2])

    best = None
    for short, tie, A, B, hx, hz, n, ac, bc in finalists[:6]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        cycles = _four_cycles(hx) + _four_cycles(hz)
        quality = (not 189 <= k <= 217, short[:3],
                   cycles, short[3:],
                   max(_six_relations(ac, 120),
                       _six_relations(bc, 120)),
                   abs(k - 193), tie)
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k monomial-splice filling")
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without redrawing its labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# A rolled, weight-12 Z17 (2,5)x(2,6) filling suffices for balanced
# memory performance: parent slots 31 and 50 had worst-Z-basis scores
# 3.691 and 3.817 despite better X-basis scores.
# Unit-separated, four-cycle-screened Z35 (1,3)x(2,5) fillings remove
# the weak X-basis mode: refuted by measured X-basis failure caps after
# 1500-3250 shots and the resulting 4.342 fitness.
# Short-component-word and four-cycle screening on the Z13
# (2,5)x(3,8) bed suffices for rare failures: refuted by the current
# parent's 4.5405 fitness; its measured slots reached the failure cap
# after as few as 3250-4250 shots in their worse basis.
# Column-spread Z17 (2,5)x(2,6) screening alone reliably improves
# worse-basis failures: refuted by its 4.5242 fitness and measured
# Z-basis failure caps after 2500-5000 shots.
# REFUTED: The Z20 zero/binomial profile remains predominantly Z-limited; four of its five measured slots were instead X-limited.
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
