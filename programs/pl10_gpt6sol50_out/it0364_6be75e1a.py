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
CONCEPT = ("In cyclic Z29 of order 29 with A=(2,4) and B=(2,4), matching the sides' indexed six-gap spectra and rejecting short component words should lower worst-basis circuit failures.")
GROUP = ("Keep cyclic Z29 and order 29 for a filling-only test. Its measured (2,4)x(2,4) bed has reached about 7.3 fitness; every nonzero gap has full order.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: q=29 gives n=580, guaranteed k>=116 and rate>=0.200, and two 116-variable component codes. The parent's X failures involved several sector-1 Acol/Bcol blocks rather than one persistent hotspot.")
FILLING = ("Use a distinct automorphism-class six-gap spectrum per indexed design, shared by A and B but placed independently. Both sides retain singleton anchors and check weight 11. Parent matched-spectrum slots 2 and 27 had 0 and 2 X failures; zero-overlap slot 5 had 15 and partial-overlap slot 16 had 6. Screen exact incidences for weight-five and weight-six component words, then verified ranks and rooted cycles; test failed shots, not d2s.")
FAMILY_NAME = "r020_z29_24x24_matched_spectra_shortwords_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Prime-order cyclic group; tuple encoding preserves the cell API.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return g[0] % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


# Both sides have one singleton in each row of column zero. Their
# binomial gaps share an indexed spectrum, but not cell placements or
# origins. Each row has seven terms and each column at most four, so
# both kinds of lifted check have weight at most 11.


def _gap_spectra():
    """Canonical six-gap sets under multiplication by units of Z29."""
    from itertools import combinations
    representatives = set()
    for gaps in combinations(range(1, 15), 6):
        representative = min(
            tuple(sorted(min((unit * gap) % 29,
                             29 - (unit * gap) % 29)
                         for gap in gaps))
            for unit in range(1, 15)
        )
        representatives.add(representative)
    return tuple(sorted(representatives))


_GAP_SPECTRA = _gap_spectra()


def _counts(idx):
    a = [[1, 2, 2, 2] for _ in range(2)]
    b = [[1, 2, 2, 2] for _ in range(2)]
    assert _grid_weights(a, b) == 11
    assert _no_dead_lines(a, b)
    return a, b


def _draw_side(rng, counts, gaps):
    """Independently place six distinct undirected gaps and cell origins."""
    assigned = list(gaps)
    rng.shuffle(assigned)
    cells = []
    for row in counts:
        filled = []
        for count in row:
            start = rng.randrange(29)
            cell = [(start, 0)]
            if count == 2:
                gap = assigned.pop()
                if rng.randrange(2):
                    gap = -gap
                cell.append(((start + gap) % 29, 0))
            filled.append(cell)
        cells.append(filled)
    assert not assigned
    return cells


def _component_columns(hx, hz, q):
    """Read both component incidences from the evaluator's exact lifts."""
    a_rows = [hx[(i * 4) * q + r]
              for i in range(2) for r in range(q)]
    b_rows = [hz[i * q + r]
              for i in range(2) for r in range(q)]
    a_cols = [
        sum(1 << i for i, row in enumerate(a_rows)
            if (row >> ((j * 4) * q + h)) & 1)
        for j in range(4) for h in range(q)
    ]
    b_cols = [
        sum(1 << i for i, row in enumerate(b_rows)
            if (row >> (j * q + h)) & 1)
        for j in range(4) for h in range(q)
    ]
    return a_cols, b_cols


def _relations(cols):
    """Deterministic short component relations and Tanner four-cycles."""
    singles = set(cols)
    pairs = {}
    bad2 = bad3 = bad4 = cycles4 = 0
    for a, x in enumerate(cols):
        for b in range(a + 1, len(cols)):
            y = cols[b]
            value = x ^ y
            bad2 += value == 0
            bad3 += value in singles
            old = pairs.get(value)
            if old is None:
                pairs[value] = (a, b)
            elif a not in old and b not in old:
                bad4 += 1
            common = (x & y).bit_count()
            cycles4 += common * (common - 1) // 2
    return bad2, bad3, bad4, cycles4


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for earlier in rows[:i]:
            overlap = (row & earlier).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _root_cycles(rows, q):
    """Balance simple four- and six-cycles over lifted check blocks.

    Group translation makes coordinate zero representative of each
    check block. The six-cycle count excludes a repeated data vertex
    by subtracting intersections common to all three checks.
    """
    largest_four = sum_four = largest_six = sum_six = 0
    for block in range(len(rows) // q):
        root_index = block * q
        root = rows[root_index]
        adjacent = []
        four = 0
        for index, row in enumerate(rows):
            if index == root_index:
                continue
            common = root & row
            if common:
                degree = common.bit_count()
                four += degree * (degree - 1) // 2
                adjacent.append((index, common, degree))

        six = 0
        for t, (u, ru, nu) in enumerate(adjacent):
            for v, rv, nv in adjacent[:t]:
                uv = (rows[u] & rows[v]).bit_count()
                if uv:
                    triple = (ru & rv).bit_count()
                    six += nu * nv * uv - triple * (
                        nu + nv + uv - 2)
        largest_four = max(largest_four, four)
        sum_four += four
        largest_six = max(largest_six, six)
        sum_six += six
    return largest_four, sum_four, largest_six, sum_six


def _small_component_words(columns):
    """Count disjoint 2+3 relations and rooted 1+2+3 relations.

    Equality of the corresponding column XORs gives a kernel word.
    Root one variable in each protograph column; translation makes
    coordinate zero representative of its other 28 lift positions.
    """
    count = len(columns)
    pairs = {}
    for a in range(count):
        abit = 1 << a
        for b in range(a + 1, count):
            value = columns[a] ^ columns[b]
            pairs.setdefault(value, []).append(abit | (1 << b))

    targets = {}
    for root in range(0, count, 29):
        rootbit = 1 << root
        root_value = columns[root]
        for value, masks in pairs.items():
            eligible = [mask | rootbit for mask in masks
                        if not mask & rootbit]
            if eligible:
                targets.setdefault(value ^ root_value, []).extend(eligible)

    fives = sixes = 0
    for a in range(count - 2):
        abit = 1 << a
        for b in range(a + 1, count - 1):
            ab = columns[a] ^ columns[b]
            abits = abit | (1 << b)
            for c in range(b + 1, count):
                value = ab ^ columns[c]
                bits = abits | (1 << c)
                for other in pairs.get(value, ()):
                    fives += not (bits & other)
                for other in targets.get(value, ()):
                    sixes += not (bits & other)
    return fives, sixes


def NEW_CTX(rng, G, idx):
    """Select a matched-spectrum design using exact LP incidences."""
    counts_a, counts_b = _counts(idx)
    assert len(_GAP_SPECTRA) >= N_SLOTS
    # Distinct canonical gap spectra make the indexed designs more
    # than fresh labels of a single spectrum.
    spectrum_index = idx * len(_GAP_SPECTRA) // N_SLOTS
    gaps = _GAP_SPECTRA[spectrum_index]

    shortlist = []
    for _ in range(42 + idx % 7):
        A = _draw_side(rng, counts_a, gaps)
        B = _draw_side(rng, counts_b, gaps)
        hx, hz, n = lifted_matrices(G, A, B)
        acols, bcols = _component_columns(hx, hz, G.q)
        fa, fb = _relations(acols), _relations(bcols)
        structural = (
            fa[0] + fb[0], fa[1] + fb[1],
            max(fa[2], fb[2]), fa[2] + fb[2],
            max(fa[3], fb[3]), fa[3] + fb[3], rng.random()
        )
        shortlist.append((structural, A, B, hx, hz, n, acols, bcols))
    shortlist.sort(key=lambda item: item[0])

    best_key = best = None
    for structural, A, B, hx, hz, n, acols, bcols in shortlist[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        wa = _small_component_words(acols)
        wb = _small_component_words(bcols)
        xc = _root_cycles(hx, G.q)
        zc = _root_cycles(hz, G.q)
        # k=116..130 places an individual n=580 code in cell 0.200.
        key = (
            not 116 <= k <= 130, abs(k - 122),
            max(wa[0], wb[0]), wa[0] + wb[0],
            max(wa[1], wb[1]), wa[1] + wb[1],
            max(xc[2], zc[2]), xc[2] + zc[2],
            max(xc[0], zc[0]), xc[0] + zc[0],
            structural
        )
        if best_key is None or key < best_key:
            best_key, best = key, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell, with distinct elements."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Full-order differences alone reliably balance circuit failures:
# refuted by current-program slot 7 (100 X failures in 98500 shots)
# versus slot 6 (6 X failures in 200000 shots).
# Complementary singleton placement and parity-gauge rejection suffice
# to reach this cell's frontier: refuted by fitness 6.3555 versus 7.0;
# measured slot 9 had 78 X failures in 200000 shots.
# Distinct gaps plus sampled component relations and full-matrix
# four-cycle counts suffice to saturate this bed: refuted by the
# parent's 7.1223 fitness; slot 20 had 17 X failures despite d2s=13.
# Cross-side gap-overlap variation plus rooted-cycle screening suffices
# for zero failures: refuted by this parent's 7.2293 fitness; measured
# slots 5 and 16 had 15 and 6 X failures in 200000 shots, respectively.
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
