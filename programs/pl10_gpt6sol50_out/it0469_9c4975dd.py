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
CONCEPT = ("In cyclic Z28 of order 28 with A=(2,4), B=(2,4), automorphism-distinct mixed-order gap spectra screened for short component words should lower circuit-level failures in cell 0.200.")
GROUP = ("Jump from Z29 to genuinely different Z28. The measured Z28 (2,4)x(2,4) bed reached 6.6, below the Z29 parent's 7.4292; mixed gap orders make this filling a new test, not an assumed improvement.")
SHAPES = ("Keep equal (2,4) sides: n=560, guaranteed k>=112 and rate>=0.200, with two 112-variable components. Parent failures implicated differing sector-1 blocks, including Acol3/Bcol2 in slot 17, while slots 20 and 24 had none in either basis.")
FILLING = ("Use an aligned singleton column and six distinct binomial gaps per side, keeping both check weights at 11. Give indexed A designs different Z28-automorphism gap spectra with three to five full-order gaps; independently vary B overlap. Screen exact lifts for short component words and balanced cycles, then seek k near 120 to test fewer failed shots and multi-logical flips.")
FAMILY_NAME = "r020_z28_24x24_mixed_order_orbit_shortwords_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Additive cyclic group Z28, encoded as pairs.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def idx(g):
        return int(g[0]) % q

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 28 = 560


def _component_short_cycles(G, cells, side):
    """Count rooted simple Tanner cycles of lengths four, six and eight.

    Group translation maps every lift coordinate of a given check row
    onto every other one, so one root per protograph check row suffices.
    Use the component's actual left/right lift conventions.
    """
    q = G.q
    check_vars = [[] for _ in range(2 * q)]
    var_checks = [[] for _ in range(len(cells[0]) * q)]
    for i, row in enumerate(cells):
        for r in G.elements:
            check = i * q + G.idx(r)
            for j, cell in enumerate(row):
                for g in cell:
                    c = G.mul(G.inv(g), r) if side == "A" else G.mul(r, g)
                    variable = j * q + G.idx(c)
                    check_vars[check].append(variable)
                    var_checks[variable].append(check)

    adjacent = [[] for _ in check_vars]
    for check, variables in enumerate(check_vars):
        for variable in variables:
            adjacent[check].extend(
                (other, variable) for other in var_checks[variable]
                if other != check
            )

    fours = sixes = eights = 0
    identity = G.idx(G.e)
    for i in range(2):
        root = i * q + identity
        root_edges = {}
        for other, variable in adjacent[root]:
            old = root_edges.setdefault(other, [])
            fours += len(old)
            old.append(variable)

        paths = {}
        for middle, first_var in adjacent[root]:
            for end, second_var in adjacent[middle]:
                if end == root or second_var == first_var:
                    continue
                for last_var in root_edges.get(end, ()):
                    if last_var != first_var and last_var != second_var:
                        sixes += 1

                old_paths = paths.setdefault(end, [])
                for old_middle, old_first, old_second in old_paths:
                    if old_middle != middle and len({
                        first_var, second_var, old_first, old_second
                    }) == 4:
                        eights += 1
                old_paths.append((middle, first_var, second_var))
    return fours, sixes, eights


def _gap_spectra28():
    """Distinct six-gap orbits containing three to five full-order gaps."""
    from itertools import combinations
    from math import gcd
    units = tuple(u for u in range(1, 28) if gcd(u, 28) == 1)
    full = (1, 3, 5, 9, 11, 13)
    shorter = (2, 4, 6, 8, 10, 12)  # exclude order-two and order-four gaps
    representatives = set()
    for number_full in (5, 4, 3):
        for a in combinations(full, number_full):
            for b in combinations(shorter, 6 - number_full):
                gaps = a + b
                representative = min(
                    tuple(sorted(min((u * d) % 28, (-(u * d)) % 28)
                                 for d in gaps))
                    for u in units
                )
                representatives.add(representative)
    return tuple(sorted(representatives))


_GAP_SPECTRA28 = _gap_spectra28()


def _trial_cells(rng, G, idx):
    """Draw fresh labels from a slot-indexed A orbit and varied B overlap."""
    spectrum = idx * len(_GAP_SPECTRA28) // N_SLOTS
    a_gaps = _GAP_SPECTRA28[spectrum]
    overlap = (0, 2, 4)[idx % 3]
    available = [g for g in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13)
                 if g not in a_gaps]
    b_gaps = (rng.sample(a_gaps, overlap)
              + rng.sample(available, 6 - overlap))

    def fill(gaps):
        gaps = list(gaps)
        rng.shuffle(gaps)
        grid = []
        for i in range(2):
            row = []
            for j in range(4):
                start = rng.randrange(G.q)
                cell = [(start, 0)]
                if j:
                    gap = gaps.pop()
                    if rng.randrange(2):
                        gap = -gap
                    cell.append(((start + gap) % G.q, 0))
                row.append(cell)
            grid.append(row)
        return grid

    return fill(a_gaps), fill(b_gaps)


def _exact_component_rows(hx, hz, q):
    """Extract both component parity checks from the verified LP lifts."""
    mask = (1 << q) - 1
    a_rows = []
    b_rows = []
    for i in range(2):
        for r in range(q):
            source = hx[(i * 4) * q + r]
            a_rows.append(sum(
                ((source >> (j * 4 * q)) & mask) << (j * q)
                for j in range(4)))
            source = hz[i * q + r]
            b_rows.append(sum(
                ((source >> (j * q)) & mask) << (j * q)
                for j in range(4)))
    return a_rows, b_rows


def _short_word_profile(rows, ncols, rng):
    """Sample locally minimal words in an exact component kernel."""
    pivots = {}
    for source in rows:
        row = source
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    ordered = sorted(pivots)
    basis = []
    for free in range(ncols):
        if free in pivots:
            continue
        word = 1 << free
        for p in ordered:
            if (word & pivots[p]).bit_count() & 1:
                word ^= 1 << p
        basis.append(word)
    if not basis:
        return (0, 0, 0, 0)

    minima = set()
    starts = rng.sample(basis, min(12, len(basis)))
    for _ in range(24):
        word = 0
        for _ in range(rng.randrange(1, 6)):
            word ^= rng.choice(basis)
        if word:
            starts.append(word)
    for word in starts:
        while True:
            weight = word.bit_count()
            improved = None
            for vector in basis:
                candidate = word ^ vector
                if candidate and candidate.bit_count() < weight:
                    improved, weight = candidate, candidate.bit_count()
            if improved is None:
                break
            word = improved
        minima.add(word)

    weights = [word.bit_count() for word in minima]
    return (sum(w <= 11 for w in weights),
            sum(w <= 12 for w in weights),
            -min(weights),
            sum(w <= 13 for w in weights))


def _root_four_balance(rows, q):
    """Four-cycle concentration over the exact lifted check blocks."""
    counts = []
    for root in rows[::q]:
        count = 0
        for row in rows:
            shared = (root & row).bit_count()
            count += shared * (shared - 1) // 2
        counts.append(count)
    return max(counts), sum(counts)


def NEW_CTX(rng, G, idx):
    """Select an indexed design by component words and exact check cycles."""
    counts = [[1, 2, 2, 2], [1, 2, 2, 2]]
    assert _grid_weights(counts, counts) == 11
    assert _no_dead_lines(counts, counts)
    assert len(_GAP_SPECTRA28) >= N_SLOTS

    finalists = []
    for _ in range(28 + idx % 5):
        A, B = _trial_cells(rng, G, idx)
        ac = _component_short_cycles(G, A, "A")
        bc = _component_short_cycles(G, B, "B")
        cycle_key = (max(ac[0], bc[0]), ac[0] + bc[0],
                     max(ac[1], bc[1]), ac[1] + bc[1],
                     max(ac[2], bc[2]), ac[2] + bc[2],
                     rng.random())
        finalists.append((cycle_key, A, B))
    finalists.sort(key=lambda entry: entry[0])

    selected = None
    for cycle_key, A, B in finalists[:9]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        ar, br = _exact_component_rows(hx, hz, G.q)
        aw = _short_word_profile(ar, 4 * G.q, rng)
        bw = _short_word_profile(br, 4 * G.q, rng)
        xf = _root_four_balance(hx, G.q)
        zf = _root_four_balance(hz, G.q)
        # At n=560, k=112..125 remains in rate cell 0.200.
        key = (not 112 <= k <= 125, abs(k - 120),
               max(aw[0], bw[0]), aw[0] + bw[0],
               max(aw[1], bw[1]), aw[1] + bw[1],
               max(aw[2], bw[2]), aw[2] + bw[2],
               max(aw[3], bw[3]), aw[3] + bw[3],
               max(xf[0], zf[0]), xf[0] + zf[0],
               cycle_key)
        if selected is None or key < selected[0]:
            selected = (key, A, B)
    return {"A": selected[1], "B": selected[2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: cyclic-order-21 rolled sparse fillings do not consistently
# suppress sector-1 weight-six logicals or worst-basis failures: the five
# measured parent slots scored only 3.91-4.63.
# Refuted: minimizing component four-cycles alone consistently prevents
# circuit failures on this D23 bed. Parent slot 42 had six X-basis failures
# in 200000 shots despite w10 and a d2s estimate of 22.
# Refuted: balanced-cycle D23 binomials guarantee zero failures; current
# slots 2, 6 and 27 had respectively 1, 2 and 1 Z-basis failed shots.
# Refuted: Z37 component-cycle screening guarantees zero Z failures:
# measured slot 1 had two in 200000 shots, flipping 74 logical qubits.
# Refuted: changing Z37 to Z36 while keeping unequal all-binomial
# components improves the five-code score: fitness fell from 7.3872
# to 7.2252; measured Z36 slots 21 and 23 had four and three Z failures.
# Refuted: light-cell motifs screened only for component cycles and pair collisions reliably prevent circuit failures on Z29; slot 6 had eight Z failures and slot 42 had six X failures in 200000 shots each.
# Refuted: short-component-word screening guarantees zero circuit failures
# on Z29; measured slots 17, 28 and 29 still had worse-basis failures.
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
