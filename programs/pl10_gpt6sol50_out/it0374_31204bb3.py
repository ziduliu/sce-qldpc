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
CONCEPT = ("In cyclic Z29 of order 29 with A=(2,4), B=(2,4), automorphism-distinct gap spectra screened for balanced short words and full-check cycles may reduce worse-basis circuit failures.")
GROUP = ("Keep cyclic Z29 and order 29 for a filling-only test. Its measured (2,4)x(2,4) bed has reached about 7.3; every nonzero binomial gap has full order.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: n=580, guaranteed k>=116 and rate>=0.200, with two 116-variable components. Parent X failures enriched sector-1 Acol0/Bcol2 and Acol3/Bcol2 data and Xchk:Arow1/Bcol0 checks; these shapes retain equal-length protection on both sides.")
FILLING = ("Index distinct six-gap spectra, vary singleton layouts and cross-side gap overlap, and keep check weight <=11. Screen exact component incidences for short words and verified HX/HZ for balanced rooted cycles. Test whether slot 50's eight X failures and 69/11 later-stage settlements become rarer; d2s is only a selection screen.")
FAMILY_NAME = "r020_z29_24x24_spectral_words_fullcycles_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Additive cyclic group Z29, encoded with two-coordinate elements.
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
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 29 = 580


# Mask bits mark singleton cells. Every row has at most seven terms,
# and every column at most four, so both check-row weights are <=11.
_SUPPORT_MOTIFS = (
    ((1, 1), (1, 1)),       # both components have an aligned anchor
    ((1, 1), (2, 2)),       # aligned anchors in different columns
    ((1, 1), (1, 2)),       # aligned A, separated B
    ((1, 2), (1, 1)),       # separated A, aligned B
    ((3, 1), (1, 1)),       # an additional A-row singleton
)


def _gap_spectra():
    """Six-gap sets, one representative per Z29-automorphism orbit."""
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


def _counts(masks):
    return [[2 - ((mask >> j) & 1) for j in range(4)]
            for mask in masks]


def _draw_side(rng, counts, spectrum):
    """Place distinct gaps and independent origins in complete cells."""
    pairs = sum(value == 2 for row in counts for value in row)
    gaps = rng.sample(spectrum, pairs)
    cells = []
    for row in counts:
        filled = []
        for count in row:
            start = rng.randrange(29)
            cell = [(start, 0)]
            if count == 2:
                gap = gaps.pop()
                if rng.randrange(2):
                    gap = -gap
                cell.append(((start + gap) % 29, 0))
            filled.append(cell)
        cells.append(filled)
    return cells


def _component_columns(hx, hz, q, side):
    """Extract both component incidences from the verified LP matrices."""
    if side == "A":
        rows = [hx[(i * B_SHAPE[1]) * q + r]
                for i in range(A_SHAPE[0]) for r in range(q)]
        stride = B_SHAPE[1]
    else:
        rows = [hz[i * q + r]
                for i in range(B_SHAPE[0]) for r in range(q)]
        stride = 1
    columns = [0] * (4 * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        for j in range(4):
            block = (row >> (j * stride * q)) & mask
            while block:
                bit = block & -block
                columns[j * q + bit.bit_length() - 1] |= 1 << check
                block ^= bit
    return columns


def _component_defects(columns):
    """Short kernel relations and pairs of variables sharing checks."""
    singles = set(columns)
    pairs = {}
    bad2 = sum(value == 0 for value in columns)
    bad3 = bad4 = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = left ^ right
            bad2 += value == 0
            bad3 += value in singles
            bad4 += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
            common = (left & right).bit_count()
            cycles += common * (common - 1) // 2
    return bad2, bad3, bad4, cycles


def _root_sixes(columns, q):
    """Count component six-cycle configurations at both check-row roots."""
    checks = [0] * (2 * q)
    for variable, column in enumerate(columns):
        while column:
            bit = column & -column
            checks[bit.bit_length() - 1] |= 1 << variable
            column ^= bit

    total = 0
    for root_index in (0, q):
        root = checks[root_index]
        nearby = [(i, root & checks[i]) for i in range(2 * q)
                  if i != root_index and root & checks[i]]
        for a, (u, ru) in enumerate(nearby):
            for v, rv in nearby[:a]:
                total += ((ru & ~rv).bit_count()
                          * (rv & ~ru).bit_count()
                          * (checks[u] & checks[v] & ~root).bit_count())
    return total


def _six_word_collisions(columns):
    """Triple-syndrome collisions signal component weight-six words.

    When the earlier short-relation screen finds no weight-two or
    weight-four word, colliding triples are disjoint. Counting their
    multiplicity distinguishes components with one versus several
    translated families of the parent's weight-12 LP witnesses.
    """
    seen = {}
    collisions = 0
    for i, first in enumerate(columns):
        for j in range(i + 1, len(columns)):
            partial = first ^ columns[j]
            for third in columns[j + 1:]:
                value = partial ^ third
                previous = seen.get(value, 0)
                collisions += previous
                seen[value] = previous + 1
    return collisions


def _full_root_load(rows, q):
    """Four- and simple six-cycle loads at each exact check-block root."""
    max_four = sum_four = max_six = sum_six = 0
    for block in range(len(rows) // q):
        root_index = block * q
        root = rows[root_index]
        neighbors = []
        fours = 0
        for index, row in enumerate(rows):
            if index == root_index:
                continue
            shared = root & row
            if shared:
                degree = shared.bit_count()
                fours += degree * (degree - 1) // 2
                neighbors.append((index, shared, degree))

        sixes = 0
        for t, (u, ru, nu) in enumerate(neighbors):
            for v, rv, nv in neighbors[:t]:
                uv = (rows[u] & rows[v]).bit_count()
                if uv:
                    triple = (ru & rv).bit_count()
                    sixes += nu * nv * uv - triple * (
                        nu + nv + uv - 2)
        max_four = max(max_four, fours)
        sum_four += fours
        max_six = max(max_six, sixes)
        sum_six += sixes
    return max_four, sum_four, max_six, sum_six


def NEW_CTX(rng, G, idx):
    """Choose an indexed spectral design using verified component/full lifts."""
    motif = idx % 5
    a_masks, b_masks = _SUPPORT_MOTIFS[motif]
    sa, sb = _counts(a_masks), _counts(b_masks)
    assert _grid_weights(sa, sb) <= 11
    assert _no_dead_lines(sa, sb)
    assert len(_GAP_SPECTRA) >= N_SLOTS

    # Different indexed A spectra are in different automorphism orbits.
    position = idx * len(_GAP_SPECTRA) // N_SLOTS
    gaps_a = _GAP_SPECTRA[position]
    complement = [g for g in range(1, 15) if g not in gaps_a]
    overlap = (0, 2, 4, 6)[(idx // 5) % 4]

    shortlist = []
    for _ in range(23 + idx % 5):
        gaps_b = (rng.sample(gaps_a, overlap)
                  + rng.sample(complement, 6 - overlap))
        A = _draw_side(rng, sa, gaps_a)
        B = _draw_side(rng, sb, gaps_b)
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, hz, G.q, "A")
        bc = _component_columns(hx, hz, G.q, "B")
        da, db = _component_defects(ac), _component_defects(bc)
        key = (da[0] + db[0], da[1] + db[1],
               max(da[2], db[2]), da[2] + db[2],
               max(da[3], db[3]), da[3] + db[3],
               rng.random())
        shortlist.append((key, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[0])

    best = None
    target_k = 122 if motif < 2 else 118
    for structural, A, B, hx, hz, n, ac, bc in shortlist[:6]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        words_a = _six_word_collisions(ac)
        words_b = _six_word_collisions(bc)
        xc = _full_root_load(hx, G.q)
        zc = _full_root_load(hz, G.q)
        # At n=580, k=116..130 stays in the requested rate cell.
        quality = (
            not 116 <= k <= 130, abs(k - target_k),
            structural[:4],
            max(words_a, words_b), words_a + words_b,
            max(xc[0], zc[0]), xc[0] + zc[0],
            max(xc[2], zc[2]), xc[2] + zc[2],
            max(xc[3], zc[3]), xc[3] + zc[3],
            structural[4:]
        )
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k Z29 filling")
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing its labels."""
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
# Complementary supports and four-cycle selection alone make X-basis
# failures rare: refuted by current-parent slots 22 and 31, with
# 106/2500 and 108/4250 X-basis failed shots respectively.
# Sparse Z17 component-word screening suffices to match the 0.350
# frontier: refuted by this parent's 4.3509 fitness; all five measured
# slots reached the Z-basis failure cap within 2000-4000 shots.
# Dense noncommuting order-84 action-mix cells balance memory failures:
# refuted by the current program's 5.4331 fitness; measured slot 27
# had 105 Z failures in 4750 shots but only 3 X failures in 200000.
# The parent's four gap palettes and component-cycle screen guarantee
# balanced, failure-free Z29 measurements: refuted by slot 50's eight
# X failures versus one Z failure in 200000 shots per basis, and slot
# 7's three X failures versus zero Z failures.
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
