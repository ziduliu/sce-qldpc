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
CONCEPT = ("In cyclic Z34 of order 34, A=(1,4) and B=(1,4), non-equivalent paired binomial gaps with parity-mixed trinomials should reduce short component relations and worse-basis circuit failures.")
GROUP = ("Keep cyclic Z34 and order 34 fixed for a filling-only comparison. This n=578 bed's current five-code fitness is 3.4845.")
SHAPES = ("Keep equal (1,4)x(1,4) one-row sides with two 136-variable components. They guarantee k>=306 and k/n>=9/17=0.5294 in cell 0.550. The parent's weight-four witnesses occupied sector-1 Acol/Bcol blocks; failed-shot enrichment shifted among those blocks, with no persistent check-block hotspot.")
FILLING = ("Use three binomials and one parity-mixed trinomial per side, giving row weight 12. Give slots different paired three-gap spectra and screen independent labels on verified component relations and Tanner cycles. Test for fewer broadly distributed worse-basis failures than the parent's 101-194 failures per 250-500 shots, while keeping k near 306.")
FAMILY_NAME = "r055_z34_14x14_paired_threegap_mixedtri_cycle_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: g[0] % q, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _component_relations(rows, bases, q):
    """Short words and Tanner cycles in an actual lifted component."""
    mask = (1 << q) - 1
    columns = [0] * (len(bases) * q)
    for check, row in enumerate(rows[:q]):
        for j, base in enumerate(bases):
            bits = (row >> base) & mask
            while bits:
                bit = bits & -bits
                columns[j * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit

    locations = {}
    for i, value in enumerate(columns):
        locations.setdefault(value, []).append(i)
    pairs = {}
    twos = sum(value == 0 for value in columns)
    triples = fours = cycles4 = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = left ^ right
            twos += value == 0
            triples += sum(k != i and k != j
                           for k in locations.get(value, ()))
            common = (left & right).bit_count()
            cycles4 += common * (common - 1) // 2
            entry = pairs.get(value)
            if entry is None:
                pairs[value] = [1, {i: 1, j: 1}]
            else:
                count, incidence = entry
                fours += count - incidence.get(i, 0) - incidence.get(j, 0)
                entry[0] = count + 1
                incidence[i] = incidence.get(i, 0) + 1
                incidence[j] = incidence.get(j, 0) + 1

    check_rows = [0] * q
    for column, signature in enumerate(columns):
        while signature:
            bit = signature & -signature
            check_rows[bit.bit_length() - 1] |= 1 << column
            signature ^= bit
    intersections = [[0] * q for _ in range(q)]
    for i in range(q):
        for j in range(i):
            intersections[i][j] = intersections[j][i] = (
                check_rows[i] & check_rows[j]).bit_count()
    cycles6 = 0
    for i in range(q):
        for j in range(i):
            for k in range(j):
                a, b, c = (intersections[i][j],
                           intersections[j][k],
                           intersections[k][i])
                if a and b and c:
                    common = (check_rows[i] & check_rows[j]
                              & check_rows[k]).bit_count()
                    cycles6 += a * b * c - common * (a + b + c) \
                               + 2 * common
    return twos, triples // 3, fours // 3, cycles4, cycles6


@__import__("functools").lru_cache(maxsize=1)
def _gap_orbits_z34():
    """Three-gap spectra, canonical under automorphisms and reversal."""
    from itertools import combinations
    units = tuple(u for u in range(1, 34, 2) if u != 17)
    representatives = set()
    for gaps in combinations(range(1, 17), 3):
        if not any(d & 1 for d in gaps):
            continue
        a, b, c = gaps
        if any(value % 34 == 0 for value in
               (a + b + c, a + b - c, a - b + c, -a + b + c)):
            continue
        representatives.add(min(
            tuple(sorted(min(u * d % 34, (-u * d) % 34)
                         for d in gaps))
            for u in units))
    return tuple(sorted(representatives))


@__import__("functools").lru_cache(maxsize=1)
def _paired_gap_designs_z34():
    """Pairs distinct even after independently canonicalizing each side."""
    orbits = _gap_orbits_z34()
    designs = [(orbits[i], orbits[j])
               for i in range(len(orbits))
               for j in range(i, len(orbits))]
    random.Random(341404).shuffle(designs)
    return tuple(designs)


def NEW_CTX(rng, G, idx):
    """Screen fresh labels for one slot-specific paired gap spectrum."""
    designs = _paired_gap_designs_z34()
    if idx >= len(designs):
        raise ValueError("insufficient distinct paired gap designs")
    ag, bg = designs[idx]
    q = G.q
    supports = [[2, 2, 2, 3]]
    assert _grid_weights(supports, supports) == 12
    assert _no_dead_lines(supports, supports)

    def draw(gaps, side):
        row = []
        for gap in rng.sample(list(gaps), 3):
            start = rng.randrange(q)
            step = gap if rng.randrange(2) else -gap
            row.append([(start, 0), ((start + step) % q, 0)])

        start = rng.randrange(q)
        if (idx + (side == "B")) % 2:
            offsets = rng.sample(range(1, q, 2), 2)
        else:
            offsets = [2 * rng.randrange(1, 17),
                       2 * rng.randrange(17) + 1]
        row.append([((start + offset) % q, 0)
                    for offset in (0,) + tuple(offsets)])
        rng.shuffle(row)
        return [row]

    shortlist = []
    for _ in range(24):
        A, B = draw(ag, "A"), draw(bg, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        ar = _component_relations(hx, [4 * j * q for j in range(4)], q)
        br = _component_relations(hz, [j * q for j in range(4)], q)
        cost = (ar[0] + br[0], ar[1] + br[1],
                max(ar[2], br[2]), ar[2] + br[2],
                max(ar[4], br[4]), ar[4] + br[4],
                ar[3] + br[3], rng.random())
        shortlist.append((cost, A, B, hx, hz, n))
    shortlist.sort(key=lambda item: item[0])

    chosen_key = chosen = None
    for cost, A, B, hx, hz, n in shortlist[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        key = (k <= 0, k != 306, cost)
        if chosen_key is None or key < chosen_key:
            chosen_key, chosen = key, {"A": A, "B": B}
    return chosen


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell, without redrawing its terms."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Parent 6x2 rolled bed did not protect its sector-2 blocks: all five
# measured slots had weight-four logicals there and >=100 failures/basis
# within 250-1750 shots.
# REFUTED: Shrinking to (1,3)x(1,5) with screened mixed-count cells
# removes the dominant failures: all five measured slots still had
# sector-1 weight-two witnesses and 201-243 Z-basis failures per
# 250 shots, for five-code fitness 2.4912.
# REFUTED: Short-component-relation screening in the dihedral order-34
# (1,4)x(1,4) bed reliably suppresses sector-1 failures: its five
# measured slots scored 3.10-3.29, and all reached the failure cap
# after only 250 shots in their worse basis.
# REFUTED: Indexing four-binomial gaps supplies twenty non-isomorphic
# initial Z34 designs: slots 0-19 collapsed to two isomorphism classes.
# The five measured mixed-support slots still had sector-1 weight-four
# witnesses and 101-194 worse-basis failures per 250-500 shots.
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
