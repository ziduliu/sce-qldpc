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
CONCEPT = ("In cyclic metacyclic Z13 x Z2 of order 26, A=(1,3) and B=(2,7), sparse B fillings screened to exclude component words below weight six should improve five-code circuit fitness over 4.1191.")
GROUP = ("Keep m=13, r=2, s=1 and order 26 unchanged: this bed produced 57 distinct codes and fitness 4.1191, permitting a filling-only comparison.")
SHAPES = ("Keep (1,3)x(2,7): n=598, guaranteed k>=260 and rate>=0.435 in cell 0.450; its unequal components have 78 and 182 variables. Parent failures spread across Acol/Bcol data blocks, with slot 40's Z failures enriched at Bcol4, not at a dominant check block.")
FILLING = ("Use trinomial A cells and indexed B profiles with two to four reinforced split cells or two isolated cells. Screen verified component columns for weight-two through weight-five words; among clean candidates favor fewer B terms and fewer weight-six A relations. Test for fewer failed shots and X-basis stage-three fallthroughs, not excess k.")
FAMILY_NAME = "r045_m13r2s1_13x27_sparseB_fiveword_screen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 26

def custom_ops(q):
    # Same metacyclic rule, now Z_13 x| Z_2 with trivial action.
    # Coprime factors make this order-26 group cyclic.
    m, r, s = 13, 2, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb); n = 23 * 26 = 598


def _verified_component_columns(rows, q, stride, length, check_rows=1):
    """Read all component check rows from the scaffold's exact lift."""
    columns = [0] * (length * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:check_rows * q]):
        for j in range(length):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return columns


def _component_defects(columns):
    """Count short relations and pairs of columns sharing two checks."""
    singles = {}
    for col in columns:
        singles[col] = singles.get(col, 0) + 1
    twos = sum(v * (v - 1) // 2 for v in singles.values())

    pairs = {}
    threes = fours = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            if syndrome:
                threes += singles.get(syndrome, 0)
                fours += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0) + 1

    check_pairs = {}
    cycles = 0
    for col in columns:
        checks = []
        while col:
            low = col & -col
            checks.append(low.bit_length() - 1)
            col ^= low
        for a in range(len(checks)):
            for b in range(a + 1, len(checks)):
                pair = (checks[a], checks[b])
                cycles += check_pairs.get(pair, 0)
                check_pairs[pair] = check_pairs.get(pair, 0) + 1
    return twos, threes // 3, fours, cycles


def _voltage(t):
    """CRT coordinates for a voltage in the order-26 metacyclic group."""
    return (t % 13, t % 2)


def _relative_collisions(shifts):
    """Repeated differences among B-column offsets."""
    seen = {}
    collisions = 0
    for i, x in enumerate(shifts):
        for y in shifts[:i]:
            difference = min((x - y) % 26, (y - x) % 26)
            collisions += seen.get(difference, 0)
            seen[difference] = seen.get(difference, 0) + 1
    return collisions


def _five_relations(columns, cap=16):
    """Count disjoint pair/triple collisions: genuine weight-five words."""
    pairs = {}
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            pairs.setdefault(left ^ columns[j], []).append((i, j))

    found = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, len(columns)):
                for u, v in pairs.get(partial ^ columns[k], ()):
                    if u != i and u != j and u != k \
                            and v != i and v != j and v != k:
                        found += 1
                        if found >= cap:
                            return cap
    return found


def _six_relations(columns, cap=600):
    """Capped disjoint triple collisions in the smaller A component."""
    seen = {}
    found = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, len(columns)):
                value = partial ^ columns[k]
                previous = seen.get(value)
                if previous is None:
                    seen[value] = (i, j, k)
                elif i not in previous and j not in previous \
                        and k not in previous:
                    found += 1
                    if found >= cap:
                        return cap
    return found


def NEW_CTX(rng, G, idx):
    """Select a fresh voltage labeling of one of twenty support designs."""
    design = idx % 20
    profile, mode = design % 4, design // 4
    extras = ((1, 1), (1, 2), (2, 2), (1, 1))[profile]
    odd_gaps = (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)

    def triple(j, for_b=False):
        for _ in range(100):
            if for_b or mode == 4:
                terms = rng.sample(range(26), 3)
            elif mode == 3:
                terms = rng.sample(range((j + idx) % 2, 26, 2), 3)
            else:
                pair_even = mode == 0 or (mode == 2 and j != 1)
                primary = range(0, 26, 2) if pair_even else range(1, 26, 2)
                other = range(1, 26, 2) if pair_even else range(0, 26, 2)
                terms = rng.sample(primary, 2) + [rng.choice(other)]
            gaps = [min((x - y) % 26, (y - x) % 26)
                    for t, x in enumerate(terms) for y in terms[:t]]
            if min(gaps) >= 3 and 13 not in gaps:
                return [_voltage(t) for t in terms]
        return [_voltage(t) for t in rng.sample(range(26), 3)]

    shortlist = []
    for _ in range(42 + idx % 7):
        A = [[triple(j) for j in range(3)]]
        order = rng.sample(range(7), 7)
        isolated = order[:2] if profile == 3 else []
        split = [j for j in order if j not in isolated]
        top = set(split[:extras[0]])
        bottom = set(split[extras[0]:sum(extras)])
        monomial = [j for j in split if j not in top and j not in bottom]

        best_offsets = None
        for _trial in range(28):
            offsets = rng.sample(range(26), 7)
            cost = (_relative_collisions([offsets[j] for j in monomial]),
                    rng.random())
            if best_offsets is None or cost < best_offsets[0]:
                best_offsets = (cost, offsets)
        offsets = best_offsets[1]

        B = [[None] * 7 for _ in range(2)]
        for j in range(7):
            start = rng.randrange(26)
            B[0][j] = [_voltage(start)]
            B[1][j] = [_voltage(start + offsets[j])]
        for row, j in enumerate(isolated):
            B[row][j] = triple(j, for_b=True)
            B[1 - row][j] = []
        for row, marked in ((0, top), (1, bottom)):
            for j in marked:
                first = B[row][j][0]
                start = next(t for t in range(26) if _voltage(t) == first)
                B[row][j].append(_voltage(start + rng.choice(odd_gaps)))

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        ac = _verified_component_columns(
            hx, G.q, B_SHAPE[1], A_SHAPE[1])
        bc = _verified_component_columns(
            hz, G.q, 1, B_SHAPE[1], B_SHAPE[0])
        a, b = _component_defects(ac), _component_defects(bc)
        cheap = (a[0] + b[0], a[1] + b[1],
                 a[2] + b[2], max(a[3], b[3]),
                 a[3] + b[3], rng.random())
        shortlist.append((cheap, A, B, hx, hz, n, ac, bc))

    shortlist.sort(key=lambda item: item[0])
    finalists = []
    for cheap, A, B, hx, hz, n, ac, bc in shortlist[:14]:
        clean_four = cheap[:3] == (0, 0, 0)
        five = _five_relations(bc) if clean_four else 16
        six_a = _six_relations(ac) if clean_four and five == 0 else 600
        terms_b = sum(len(cell) for row in B for cell in row)
        quality = (cheap[:3], five, terms_b, six_a,
                   cheap[3], cheap[4], cheap[5])
        finalists.append((quality, A, B, hx, hz, n))

    finalists.sort(key=lambda item: item[0])
    for _, A, B, hx, hz, n in finalists:
        if n - gf2_rank(hx) - gf2_rank(hz) == 260:
            return {"A": A, "B": B}
    _, A, B, _, _, _ = finalists[0]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Predominantly single-term rolled D7 cells suffice for low circuit failure:
# selected parent codes had 101-113 worst-basis failures in 250 shots and
# reported weight-four sector-2 logical operators.
# Pair-signature screening alone did not eliminate the current bed's weight-3/4 sector-1 logicals or its frequent measured failures.
# Penalizing component-kernel weights 2-4 across five profiles did not
# balance the bases: fitness fell to 3.3540 and all five measured slots
# were Z-limited, including slot 48 with 160 failures in 250 shots.
# Further density-and-cycle screening on Z7 x| Z6 did not remove
# frequent failures: the current parent scored 3.5312, with slot 10
# reaching 192 X-basis failures in 500 shots.
# Screening weight-four relations and cycles alone did not make this
# order-26 bed failure-rare: all five measured slots reached the
# failed-shot cap in both bases, despite five-code fitness 4.1191.
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
