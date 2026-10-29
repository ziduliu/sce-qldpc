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
CONCEPT = ("In C7 semidirect C8 with inversion action, order 56, A=(1,3) and B=(1,3), new quotient-coset profiles screened on full verified checks should reduce worst-basis circuit failures.")
GROUP = ("Keep the order-56 C7 semidirect C8 group fixed. This bed scored 4.6994, close to the measured order-56 Q8 times C7 bed's 4.6988, making a filling-only comparison worthwhile.")
SHAPES = ("Keep equal (1,3)x(1,3) sides: n=560, two equal 168-variable components, guaranteed k>=224 and rate 0.400. Parent weight-eight witnesses and failures occupied sector-1 Acol/Bcol blocks; check-block enrichment showed no persistent hotspot.")
FILLING = ("Cross twenty new A/B quotient-pattern designs, then test repeated-coset triples and one-binomial variants. Screen verified component relations and rooted full-HX/HZ cycles, retaining target-cell k. Aim to reduce the parent's roughly 100 limiting failures within 3500-6250 shots and its decoder fallthroughs, without seeking k above the bound.")
FAMILY_NAME = "r040_c7_by_c8_inversion_13x13_fullcheck_cosets_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 56

def custom_ops(q):
    # C7 semidirect C8: y*x*y^-1 = x^-1.
    els = [(a, b) for b in range(8) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 8)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 7, (b + d) % 8)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 8
        return ((a if bi & 1 else -a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb); n = 10 * 56 = 560


def _verified_component_flaws(rows, q, bases):
    """Short relations and four-cycles in verified component columns."""
    columns = [0] * (len(bases) * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        for j, base in enumerate(bases):
            bits = (row >> base) & mask
            while bits:
                bit = bits & -bits
                columns[j * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit

    singles = {}
    for value in columns:
        singles[value] = singles.get(value, 0) + 1
    twos = sum(v * (v - 1) // 2 for v in singles.values())
    pair_counts, pair_examples = {}, {}
    triples = fours = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            syndrome = left ^ right
            triples += singles.get(syndrome, 0)
            fours += pair_counts.get(syndrome, 0)
            pair_counts[syndrome] = pair_counts.get(syndrome, 0) + 1
            pair_examples.setdefault(syndrome, (i, j))
            overlap = (left & right).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return (twos, triples // 3, fours, cycles), columns, pair_examples


def _sample_five_relations(rng, columns, pairs, draws=1400):
    """Sample disjoint triple/pair component-kernel relations."""
    found = 0
    for _ in range(draws):
        i, j, k = rng.sample(range(len(columns)), 3)
        pair = pairs.get(columns[i] ^ columns[j] ^ columns[k])
        if pair is not None and i not in pair and j not in pair \
                and k not in pair:
            found += 1
    return found


# Each ordered triple specifies C8 offsets inside one C7-by-C8 cell.
# The first ten patterns have distinct quotient coordinates and are
# different from the parent's four choices. The last four deliberately
# repeat a quotient coordinate while retaining distinct group elements.
_QUOTIENT_PATTERNS = (
    (0, 1, 2), (0, 1, 5), (0, 1, 6), (0, 2, 3), (0, 2, 7),
    (0, 3, 4), (0, 3, 5), (0, 4, 5), (0, 4, 7), (0, 5, 6),
    (0, 0, 1), (0, 0, 3), (0, 1, 1), (0, 2, 2),
)


def _draw_side(rng, counts, tokens):
    """Draw complete cells, including distinct C7 labels in repeated cosets."""
    row = []
    for count, token in zip(counts, tokens):
        offsets = list(_QUOTIENT_PATTERNS[token])
        if count == 2:
            offsets = rng.sample(offsets, 2)
        shift = rng.randrange(8)
        cell = []
        for offset in offsets:
            quotient = (shift + offset) % 8
            available = [a for a in range(7)
                         if (a, quotient) not in cell]
            cell.append((rng.choice(available), quotient))
        rng.shuffle(cell)
        row.append(cell)
    return [row]


def _rooted_full_cycles(rows, q):
    """Four- and six-cycle costs at roots in every verified check block."""
    fours = sixes = 0
    for block in range(3):
        for coordinate in (0, 1):
            root_index = block * q + coordinate
            root = rows[root_index]
            nearby = []
            for index, row in enumerate(rows):
                if index == root_index:
                    continue
                common = root & row
                if common:
                    degree = common.bit_count()
                    fours += degree * (degree - 1) // 2
                    nearby.append((index, common))
            for pos, (u, ru) in enumerate(nearby):
                for v, rv in nearby[:pos]:
                    outside = rows[u] & rows[v] & ~root
                    if outside:
                        sixes += ((ru & ~rv).bit_count()
                                  * (rv & ~ru).bit_count()
                                  * outside.bit_count())
    return fours, sixes


def NEW_CTX(rng, G, idx):
    """Screen one of twenty profile pairs, with distinct later-tier tests."""
    design, tier = idx % 20, idx // 20
    a_mode, b_mode = design % 5, design // 5
    ac, bc = [3, 3, 3], [3, 3, 3]

    # Tier zero replaces the parent's distinct-coset patterns. Tier one
    # changes one cell on EACH side to a repeated-coset triple. Tier two
    # tests one binomial on one side; further tiers test both sides.
    a_tokens = [(a_mode + 2 * j) % 10 for j in range(3)]
    b_tokens = [(2 * b_mode + a_mode + 3 * j + 1) % 10
                for j in range(3)]
    if tier == 1:
        a_tokens[design % 3] = 10 + a_mode % 4
        b_tokens[(design // 3) % 3] = 10 + b_mode
    elif tier >= 2:
        if design % 2:
            ac[design % 3] = 2
        else:
            bc[design % 3] = 2
        if tier >= 3:
            ac[design % 3] = 2
            bc[(design + 1) % 3] = 2

    assert _grid_weights([ac], [bc]) <= W_MAX
    assert _no_dead_lines([ac], [bc])
    q = G.q
    finalists = []
    for _ in range(28 + idx % 5):
        A = _draw_side(rng, ac, a_tokens)
        B = _draw_side(rng, bc, b_tokens)
        hx, hz, n = lifted_matrices(G, A, B)
        a = _verified_component_flaws(
            hx[:q], q, [j * B_SHAPE[1] * q
                        for j in range(A_SHAPE[1])])
        b = _verified_component_flaws(
            hz[:q], q, [j * q for j in range(B_SHAPE[1])])
        af, bf = a[0], b[0]
        short = (max(af[0], bf[0]), af[0] + bf[0],
                 max(af[1], bf[1]), af[1] + bf[1],
                 max(af[2], bf[2]), af[2] + bf[2],
                 max(af[3], bf[3]), af[3] + bf[3])
        finalists.append((short, rng.random(), A, B, hx, hz, n, a, b))
    finalists.sort(key=lambda item: item[:2])

    best = None
    for short, tie, A, B, hx, hz, n, a, b in finalists[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        fa = _sample_five_relations(rng, a[1], a[2], draws=1100)
        fb = _sample_five_relations(rng, b[1], b[2], draws=1100)
        x_cycles = _rooted_full_cycles(hx, q)
        z_cycles = _rooted_full_cycles(hz, q)
        choice = (not 224 <= k <= 237, short[:4],
                  max(fa, fb), fa + fb,
                  max(x_cycles[0], z_cycles[0]),
                  x_cycles[0] + z_cycles[0],
                  max(x_cycles[1], z_cycles[1]),
                  x_cycles[1] + z_cycles[1],
                  short[4:], abs(k - 224), tie)
        if best is None or choice < best[0]:
            best = (choice, A, B)
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Predominantly single-term rolled D7 cells suffice for low circuit failure:
# selected parent codes had 101-113 worst-basis failures in 250 shots and
# reported weight-four sector-2 logical operators.
# Unrestricted 2/3-term labels selected by repeated-pair screening alone
# improve this bed: measured five-code fitness fell to 3.3584 versus the
# preceding program's 3.4438, with weight-three witnesses in several slots.
# Parity-crossing pairs and repeated-parity triples do not by themselves
# make X-basis failures rare: all five selected slots still reached the
# failure cap, with roughly 122-157 X failures per 500 shots.
# Full-order gaps and mixed-quotient triples did not improve that failure rate: fitness fell to 3.5688, with 130-195 X failures per 500 shots in the five measured slots.
# The C19 balanced singleton/binomial profiles do not make failures rare:
# measured slot 1 had 102 X failures in 20250 shots and slot 5 had
# 101 Z failures in 29250 shots.
# C59 support/palette screening did not preserve the C19 program's
# 5.4015 fitness: it scored 4.4465, with each measured slot reaching
# roughly 100 worse-basis failures within 1750-5000 shots.
# Cross-coset triples screened only on component relations did not make
# order-56 failures rare: fitness was 4.6994, with every measured slot
# reaching roughly 100 failures in a basis within 3500-6250 shots.
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
