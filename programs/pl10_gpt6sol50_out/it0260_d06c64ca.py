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
CONCEPT = ("Cyclic Z29 of order 29, A=(2,4), B=(2,4): selecting binomial-gap designs by connected-error syndrome expansion should reduce worse-basis circuit failures.")
GROUP = ("Keep Z29 at q=29 for a filling-only test. This measured bed has produced strong codes, while the current program scores 7.0157.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: n=580, two 116-variable components, and guaranteed k/n>=116/580=0.200. The parent's all-binomial codes have k=132, rate 0.228 in cell 0.250. Minimum-weight witnesses occupy sector-1 Acol/Bcol blocks; slots 26 and 33 had 13 and 15 X failures with differing, modest block enrichment.")
FILLING = ("Use distinct indexed eight-gap spectra and independent binomial labels, retaining row weight 12. Screen verified component cycles, then select for larger syndromes of connected four- through eight-qubit errors on both components and verify k=132. Test for fewer worst-basis failures and stage-three fallthroughs than cycle-only screening; d2s=13 did not distinguish the parent's slots 24, 26 and 33.")
FAMILY_NAME = "r025_z29_24x24_connected_syndrome_expansion_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Cyclic group as the metacyclic presentation Z_m : Z_r,
    # with m=29, r=1 and trivial action s=1.
    m, r, s = 29, 1, 1
    els = [(a, 0) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        return ((-g[0]) % m, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: g[0] % m,
                red=lambda g: (g[0] % m, 0))
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _component_cycles(hx, hz, q):
    """Rooted 4-, 6-, and 8-cycle counts from the verified component lifts.

    HX at B-column zero contains A; HZ at A-column zero contains B.
    Two check roots per component suffice for the cyclic translations.
    """
    profiles = []
    mask = (1 << q) - 1
    for rows, stride in ((hx, 4), (hz, 1)):
        check_vars = [[] for _ in range(2 * q)]
        var_checks = [[] for _ in range(4 * q)]
        for i in range(2):
            for r in range(q):
                check = i * q + r
                bits = rows[(i * stride) * q + r]
                for j in range(4):
                    present = (bits >> (j * stride * q)) & mask
                    while present:
                        low = present & -present
                        var = j * q + low.bit_length() - 1
                        check_vars[check].append(var)
                        var_checks[var].append(check)
                        present ^= low

        four = six = eight = 0
        for root in (0, q):
            root_vars = set(check_vars[root])
            shared = {}
            paths = {}
            for first in check_vars[root]:
                for middle in var_checks[first]:
                    if middle == root:
                        continue
                    shared.setdefault(middle, []).append(first)
                    for second in check_vars[middle]:
                        if second == first:
                            continue
                        for end in var_checks[second]:
                            if end != root and end != middle:
                                paths.setdefault(end, []).append(
                                    (first, second, middle))
            four += sum(len(v) * (len(v) - 1) // 2
                        for v in shared.values())
            for end, entries in paths.items():
                closing = root_vars.intersection(check_vars[end])
                for first, second, _ in entries:
                    six += sum(last != first and last != second
                               for last in closing)
                for p, (first, second, middle) in enumerate(entries):
                    for other_first, other_second, other_middle in entries[p + 1:]:
                        if (middle != other_middle
                                and len({first, second, other_first,
                                         other_second}) == 4):
                            eight += 1
        profiles.append((four, six // 2, eight))
    return profiles


def _indexed_gap_spectra():
    """One representative per Z29 scaling orbit of eight-gap spectra."""
    from itertools import combinations
    representatives = {}
    for gaps in combinations(range(1, 15), 8):
        orbit_key = min(
            tuple(sorted(min((u * d) % 29, (-u * d) % 29)
                         for d in gaps))
            for u in range(1, 29))
        representatives.setdefault(orbit_key, gaps)
    designs = sorted(representatives)
    random.Random(19063).shuffle(designs)
    return tuple(designs)


_GAP_SPECTRA = _indexed_gap_spectra()


def _component_columns_from_lift(rows, q, stride):
    """Read component columns from the verified HX or HZ lift."""
    columns = [0] * (4 * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            check_bit = 1 << (i * q + r)
            row = rows[i * stride * q + r]
            for j in range(4):
                chunk = (row >> (j * stride * q)) & mask
                while chunk:
                    low = chunk & -chunk
                    columns[j * q + low.bit_length() - 1] |= check_bit
                    chunk ^= low
    return columns


def _connected_expansion(columns, q, beam_width=20):
    """Find small syndromes among connected errors, at four lift orbits.

    This is a screening signal, not a distance estimate or a decoder.
    The four roots represent the four variable-block translation orbits.
    """
    checks = [0] * (2 * q)
    for v, syndrome in enumerate(columns):
        bits = syndrome
        while bits:
            low = bits & -bits
            checks[low.bit_length() - 1] |= 1 << v
            bits ^= low

    adjacent = [0] * len(columns)
    for variables in checks:
        bits = variables
        while bits:
            low = bits & -bits
            v = low.bit_length() - 1
            adjacent[v] |= variables ^ low
            bits ^= low

    smallest = [4 * depth for depth in range(9)]
    for block in range(4):
        root = block * q
        beam = [(1 << root, columns[root], adjacent[root])]
        for depth in range(2, 9):
            candidates = {}
            for chosen, syndrome, frontier in beam:
                available = frontier & ~chosen
                while available:
                    low = available & -available
                    v = low.bit_length() - 1
                    new_chosen = chosen | low
                    if new_chosen not in candidates:
                        candidates[new_chosen] = (
                            syndrome ^ columns[v], frontier | adjacent[v])
                    available ^= low
            if not candidates:
                break
            ranked = sorted(candidates.items(),
                            key=lambda item: (item[1][0].bit_count(),
                                              item[0]))
            beam = [(chosen, value[0], value[1])
                    for chosen, value in ranked[:beam_width]]
            smallest[depth] = min(smallest[depth],
                                  beam[0][1].bit_count())
    return tuple(smallest[4:9])


def NEW_CTX(rng, G, idx):
    """Compare independent labels for this slot's two gap-orbit designs."""
    pa = _GAP_SPECTRA[(2 * idx) % len(_GAP_SPECTRA)]
    pb = _GAP_SPECTRA[(2 * idx + 1) % len(_GAP_SPECTRA)]
    counts = [[2] * 4 for _ in range(2)]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    def draw(spectrum):
        gaps = rng.sample(spectrum, 8)
        grid = []
        for i in range(2):
            row = []
            for j in range(4):
                gap = gaps[4 * i + j]
                step = gap if rng.randrange(2) else -gap
                start = rng.randrange(G.q)
                row.append([(start, 0),
                            ((start + step) % G.q, 0)])
            grid.append(row)
        return grid

    trials = []
    for _ in range(24):
        A, B = draw(pa), draw(pb)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_cycles(hx, hz, G.q)
        cycles = (max(ac[0], bc[0]), ac[0] + bc[0],
                  max(ac[1], bc[1]), ac[1] + bc[1],
                  max(ac[2], bc[2]), ac[2] + bc[2])
        trials.append((cycles, rng.random(), A, B, hx, hz, n))
    trials.sort(key=lambda item: item[:2])

    best_key, best = None, None
    for cycles, tie, A, B, hx, hz, n in trials[:6]:
        a = _connected_expansion(
            _component_columns_from_lift(hx, G.q, 4), G.q)
        b = _connected_expansion(
            _component_columns_from_lift(hz, G.q, 1), G.q)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Compare the most vulnerable component first, at error
        # weights six, seven and eight; retain the target rate cell.
        quality = (k != 132,
                   -min(a[2], b[2]), -min(a[3], b[3]),
                   -min(a[4], b[4]),
                   -(a[2] + b[2]), -(a[3] + b[3]),
                   -min(a[1], b[1]), cycles, tie)
        if best_key is None or quality < best_key:
            best_key, best = quality, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, without changing its labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Parent (6,2)x(6,2) mostly-monomial fillings did not suppress sector-2
# weight-4 logicals: its five measured scores were 3.30--3.63.
# Rank and Tanner-four-cycle screening alone did not prevent weak X-basis
# slots at q=24: slots 37 and 43 failed 100/10750 and 102/25500 shots.
# REFUTED: Balanced mixed-count Z23 fillings adequately balance the two
# bases: parent slot 3 had 101 Z failures in 35750 shots but only 10 X
# failures in 200000, with broadly distributed sector-1 data enrichment.
# REFUTED: The current all-even, short-relation/four-cycle screen reliably
# balances both bases: slot 1 had 33 Z versus 1 X failures, while slot 3
# had 45 X versus 9 Z failures, each basis sampled for 200000 shots.
# REFUTED: Balancing component four-, six- and eight-cycle counts alone
# makes d2s=13 slots similarly reliable: current slots 26 and 33 had
# 13 and 15 X failures, versus four X failures in slot 24, in 200000
# shots per basis despite all three having k=132 and d2s=13.
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
