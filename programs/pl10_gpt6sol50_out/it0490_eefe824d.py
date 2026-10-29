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
CONCEPT = ("In cyclic Z37 of order 37 with A=(2,3), B=(2,4), indexed mixed-support and odd gap-overlap designs screened for component ambiguities should reduce Z-basis circuit failures.")
GROUP = ("Keep the measured order-37 cyclic bed, which has reached fitness 7.4. Its prime-order lift has no proper nontrivial subgroup in which a nonzero binomial gap can be confined.")
SHAPES = ("Keep (2,3)x(2,4), whose largest permitted Z37 lift has n=592, guaranteed k>=74 and rate>=0.125; k=88 has rate 0.149. The unequal A/B component lengths are 111/148, unlike equal-shaped sides. Parent Z failures enriched varying sector-1 Acol/Bcol blocks, including Acol0/Bcol1 and Acol1/Bcol1, rather than a persistent check-block hotspot.")
FILLING = ("Cross distinct A-gap spectra with odd A/B overlap counts and indexed all-binomial, one-singleton and two-singleton profiles. Screen verified component pair-syndrome collisions and HX/HZ check-cycle concentration, then verify k. Test whether the parent's one- and two-shot Z failures and decoder fall-throughs become rarer while retaining five measurable rate-0.150 codes.")
FAMILY_NAME = "r015_z37_23x24_odd_overlap_mixed_kernel_spread_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Additive Z37, encoded as pairs.
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

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 16 * 37 = 592


def _unit_gaps():
    return tuple(range(1, 19))


from functools import lru_cache


@lru_cache(None)
def _gap_spectra():
    """Six-gap A spectra, one per Z37 automorphism orbit."""
    from itertools import combinations

    seen = set()
    representatives = []
    for subset in combinations(_unit_gaps(), 6):
        if subset in seen:
            continue
        orbit = {
            tuple(sorted(min((u * d) % 37, (-u * d) % 37)
                         for d in subset))
            for u in range(1, 37)
        }
        seen.update(orbit)
        representatives.append(min(orbit))
    return tuple(sorted(representatives))


def _draw_voltage_side(rng, gaps, counts, q):
    """Assign distinct signed gaps only to binomial cells."""
    assigned = list(gaps)
    rng.shuffle(assigned)
    gap_iter = iter(assigned)
    grid = []
    for row_counts in counts:
        row = []
        for count in row_counts:
            start = rng.randrange(q)
            if count == 1:
                row.append([(start, 0)])
            else:
                gap = next(gap_iter)
                if rng.randrange(2):
                    gap = -gap
                row.append([(start, 0), ((start + gap) % q, 0)])
        grid.append(row)
    return grid


def _indexed_counts(idx):
    """Twenty indexed gap designs crossed with three support phases."""
    design, phase = idx % 20, idx // 20
    kind = design % 5
    sa = [[2] * 3 for _ in range(2)]
    sb = [[2] * 4 for _ in range(2)]

    if kind == 1 or kind == 3:
        row = (design + phase) % 2
        col = (design // 2 + phase) % 4
        sb[row][col] = 1
        if kind == 3 and phase >= 1:
            sb[1 - row][(col + phase) % 4] = 1
    if kind == 2 or (kind == 4 and phase >= 1):
        sa[(design // 3 + phase) % 2][(design + phase) % 3] = 1
    if kind == 4 and phase >= 2:
        sb[(design + phase) % 2][(design // 2 + 1) % 4] = 1

    return sa, sb


def _verified_component_rows(G, A, B):
    """Extract both actual component actions from the verified CSS lift."""
    hx, hz, _ = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q) - 1
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    a_rows = []
    for i in range(A_SHAPE[0]):
        for r in range(q):
            lifted_row = hx[(i * cb) * q + r]  # beta=0
            a_rows.append(sum(
                ((lifted_row >> ((j * cb) * q)) & mask) << (j * q)
                for j in range(ca)
            ))
    b_rows = []
    for alpha in range(B_SHAPE[0]):
        for r in range(q):
            lifted_row = hz[alpha * q + r]  # j=0
            b_rows.append(sum(
                ((lifted_row >> (beta * q)) & mask) << (beta * q)
                for beta in range(cb)
            ))
    return a_rows, b_rows


def _component_quality(rows, q, width):
    """Short kernel relations and rooted four-/six-cycle indicators."""
    columns = [0] * (width * q)
    for check, bits in enumerate(rows):
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << check
            bits ^= bit

    seen = {}
    doubles = fours = 0
    for i, first in enumerate(columns):
        for second in columns[:i]:
            syndrome = first ^ second
            doubles += syndrome == 0
            fours += seen.get(syndrome, 0)
            seen[syndrome] = seen.get(syndrome, 0) + 1

    cycles4 = cycles6 = 0
    for root_index in (0, q):
        root = rows[root_index]
        nearby = []
        for check, row in enumerate(rows):
            if check == root_index:
                continue
            overlap = root & row
            count = overlap.bit_count()
            cycles4 += count * (count - 1) // 2
            if count:
                nearby.append((row, overlap, count))
        for i, (left, left_root, left_count) in enumerate(nearby):
            for right, right_root, right_count in nearby[:i]:
                middle = (left & right & ~root).bit_count()
                if middle:
                    cycles6 += middle * (
                        left_count * right_count
                        - (left_root & right_root).bit_count()
                    )
    return doubles, fours, cycles4, cycles6


def _css_four_spread(rows, q):
    """Four-cycle concentration at one check in each verified lift block."""
    rooted = []
    for block in range(len(rows) // q):
        root_index = block * q
        root = rows[root_index]
        count = 0
        for index, other in enumerate(rows):
            if index != root_index:
                common = (root & other).bit_count()
                count += common * (common - 1) // 2
        rooted.append(count)
    return max(rooted), sum(rooted)


def NEW_CTX(rng, G, idx):
    """Select an indexed support/gap design using verified LP matrices."""
    sa, sb = _indexed_counts(idx)
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    spectra = _gap_spectra()
    design = idx % 20
    # Spread the twenty A spectra across automorphism orbits rather
    # than using the parent's orbit positions.
    spectrum = spectra[(len(spectra) * (design + 1)) // 21]
    a_doubles = sum(count == 2 for row in sa for count in row)
    b_doubles = sum(count == 2 for row in sb for count in row)
    finalists = []

    for _ in range(42 + idx % 11):
        a_gaps = rng.sample(spectrum, a_doubles)
        overlap = min(1 + 2 * (design % 3), a_doubles, b_doubles)
        b_gaps = rng.sample(a_gaps, overlap)
        b_gaps += rng.sample(
            [gap for gap in _unit_gaps() if gap not in a_gaps],
            b_doubles - overlap
        )
        A = _draw_voltage_side(rng, a_gaps, sa, G.q)
        B = _draw_voltage_side(rng, b_gaps, sb, G.q)
        a_rows, b_rows = _verified_component_rows(G, A, B)
        aq = _component_quality(a_rows, G.q, 3)
        bq = _component_quality(b_rows, G.q, 4)
        # Pair-syndrome collisions expose short component ambiguities;
        # rooted cycles are secondary decoder-screening signals.
        quality = (aq[0] + bq[0],
                   3 * bq[1] + 4 * aq[1],
                   max(3 * bq[1], 4 * aq[1]),
                   max(aq[2], bq[2]), aq[2] + bq[2],
                   max(aq[3], bq[3]), aq[3] + bq[3],
                   rng.random())
        finalists.append((quality, A, B))
    finalists.sort(key=lambda entry: entry[0])

    chosen, chosen_key = None, None
    # n=592: integer k=75..103 lies in the requested rate cell.
    for quality, A, B in finalists[:12]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        xc = _css_four_spread(hx, G.q)  # HX detects Z errors.
        zc = _css_four_spread(hz, G.q)
        key = (not 75 <= k <= 103, quality[:3],
               xc[0], xc[1], max(xc[0], zc[0]),
               zc[0] + xc[0], quality[3:],
               abs(k - 88))
        if chosen_key is None or key < chosen_key:
            chosen_key, chosen = key, {"A": A, "B": B}
    return chosen


def CELL(rng, G, side, i, j, ctx):
    """Return a screened full cell without redrawing its terms."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Singleton-dominated rolled A protects X-basis adequately: refuted by all five measured slots; slot 12 had 109 X-basis failures in 750 shots versus 2 Z-basis failures in 200000.
# Dense A with at most five B double cells prevents circuit failures:
# refuted by parent slot 10's 60 X failures and slot 12's 28 Z failures,
# each in 200000 shots.
# Unit-gap, component-screened Z35 binomials guarantee zero failures:
# refuted by this parent's measured slots 5, 10 and 32, with respectively
# 3, 4 and 1 Z-basis failures in 200000 shots.
# Distinct indexed A-gap spectra suffice to make every Z37 slot distinct:
# refuted by the immediate parent's 26 distinct slots out of 60.
# Component-cycle-first, all-binomial screening eliminates Z failures:
# refuted by immediate-parent measured slots 0, 2 and 43, with 1, 2
# and 1 Z-basis failures respectively in 200000 shots.
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
