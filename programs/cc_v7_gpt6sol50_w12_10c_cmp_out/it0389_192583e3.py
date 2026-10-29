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
CONCEPT = ("On cyclic C26 of order 26 with A=(1,3), B=(2,7), "
           "screened triple/zero and asymmetric-doubleton B profiles "
           "can give five new cell-0.450 codes without measured "
           "logicals below weight six.")
GROUP = ("Keep cyclic C26 at order 26 for a filling-only comparison. "
         "This bed produced 60 distinct codes and five measured "
         "distance-six codes with the current filling.")
SHAPES = ("Keep (1,3)x(2,7): n=23*26=598, guaranteed k=260 and "
          "rate 0.435. The three-column, one-check A component and "
          "seven-column, two-check B component have unequal lengths. "
          "The parent's lightest X/Z witnesses spread across "
          "sector-1 Acol/Bcol blocks.")
FILLING = ("Cross four three-term A parity profiles with five legal "
           "B profiles, including paired triple/zero cells and unequal "
           "doubleton counts. A rows weigh nine, B rows at most nine, "
           "and B columns at most three, so both check types weigh at "
           "most 12. Screen verified component and full lifts for "
           "short relations and monitor k above the 260 bound.")
FAMILY_NAME = "r045_c26_13x27_heterogeneous_B_shortscreen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 26

def custom_ops(q):
    # The cyclic group C26, encoded as one-coordinate tuples.
    els = [(a,) for a in range(q)]

    def red(g):
        return (g[0] % q,)

    def idx(g):
        return g[0] % q

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _columns(rows, starts, q):
    """Extract component columns from verified LP check rows."""
    cols = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for ri, row in enumerate(rows):
        mark = 1 << ri
        for j, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                cols[j * q + bit.bit_length() - 1] |= mark
                bits ^= bit
    return cols


def _all_columns(rows, n):
    cols = [0] * n
    for ri, row in enumerate(rows):
        mark = 1 << ri
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= mark
            row ^= bit
    return cols


def _short_grade(cols, roots=()):
    """Count relations through four; test rooted five-words if requested."""
    singles = set(cols)
    bad2 = int(0 in singles) + len(cols) - len(singles)
    if bad2:
        return (bad2, 0, 0, 0)
    pairs = {}
    bad3 = bad4 = 0
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            bad3 += syndrome in singles
            bad4 += syndrome in pairs
            pairs.setdefault(syndrome, (j, i))
    if not roots or bad3 or bad4:
        return (0, bad3, bad4, 0)

    # Cyclic translation takes any component word to one rooted at the
    # identity in one of its protograph blocks.
    fives = 0
    for root in roots:
        pivot = cols[root]
        for i in range(len(cols)):
            if i == root:
                continue
            for j in range(i):
                if j == root:
                    continue
                pair = pairs.get(pivot ^ cols[i] ^ cols[j])
                if pair is not None and root not in pair \
                        and i not in pair and j not in pair:
                    fives += 1
                    if fives >= 24:
                        return (0, 0, 0, fives)
    return (0, 0, 0, fives)


def _draw_design(rng, idx):
    """Draw one member of an indexed A-parity/B-support design."""
    parity_mode, topology = divmod(idx % 20, 5)
    A = [[]]
    for j in range(3):
        if parity_mode == 0:
            parity = [0, 0, 1]
        elif parity_mode == 1:
            parity = [0, 1, 1]
        elif parity_mode == 2:
            parity = [j % 2, j % 2, 1 - j % 2]
        else:
            parity = [j % 2, 1 - j % 2, 1 - j % 2]
        rng.shuffle(parity)
        # Distinct residues modulo 13 avoid coincident labels after
        # projecting the lift onto its order-13 quotient.
        residues = rng.sample(range(13), 3)
        A[0].append([((2 * a + p) % 26,)
                     for a, p in zip(residues, parity)])

    counts = [[1] * 7 for _ in range(2)]
    sites = rng.sample(range(7), 7)
    if topology == 0:
        # One triple/zero exchange; the opposite row has three doubles.
        counts[0][sites[0]], counts[1][sites[0]] = 3, 0
        for col in sites[1:4]:
            counts[1][col] = 2
    elif topology == 1:
        # Two exchanges in opposite directions and one double per row.
        counts[0][sites[0]], counts[1][sites[0]] = 3, 0
        counts[1][sites[1]], counts[0][sites[1]] = 3, 0
        counts[0][sites[2]] = 2
        counts[1][sites[3]] = 2
    elif topology == 2:
        for col in sites[:2]:
            counts[0][col] = 2
        counts[1][sites[2]] = 2
    elif topology == 3:
        counts[0][sites[0]] = 2
        for col in sites[1:3]:
            counts[1][col] = 2
    else:
        counts[0][sites[0]], counts[1][sites[0]] = 3, 0
        for col in sites[1:3]:
            counts[1][col] = 2

    B = [[], []]
    for r in range(2):
        for j in range(7):
            count = counts[r][j]
            if count == 0:
                cell = []
            elif count == 1:
                cell = [(rng.randrange(26),)]
            elif count == 2:
                start = rng.randrange(26)
                gap = rng.choice((1, 2, 3, 5, 7, 9, 10, 11))
                cell = [(start,), ((start + gap) % 26,)]
            else:
                residues = rng.sample(range(13), 3)
                parity = [0, 1, rng.randrange(2)]
                rng.shuffle(parity)
                cell = [((2 * a + p) % 26,)
                        for a, p in zip(residues, parity)]
            B[r].append(cell)

    sa = [[len(cell) for cell in A[0]]]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return A, B


def NEW_CTX(rng, G, idx):
    """Select independently filled motifs using exact lifted matrices."""
    q = G.q
    a_starts = [j * 7 * q for j in range(3)]
    b_starts = [j * q for j in range(7)]
    roots_a = tuple(j * q for j in range(3))
    roots_b = tuple(j * q for j in range(7))
    shortlist = []

    for trial in range(24):
        A, B = _draw_design(rng, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _columns(hx[:q], a_starts, q)
        bc = _columns(hz[:2*q], b_starts, q)
        sa = _short_grade(ac)
        sb = _short_grade(bc)
        key = (sa[0] + sb[0], sa[1] + sb[1],
               sa[2] + sb[2], max(sa[2], sb[2]))
        shortlist.append((key, trial, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[:2])

    best = None
    for key, trial, A, B, hx, hz, n, ac, bc in shortlist[:8]:
        sa = _short_grade(ac, roots_a)
        sb = _short_grade(bc, roots_b)
        sx = _short_grade(_all_columns(hx, n))
        sz = _short_grade(_all_columns(hz, n))
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (int(not 0.425 <= k / n < 0.475),
                 sx[0] + sz[0], sx[1] + sz[1],
                 sx[2] + sz[2], key,
                 sa[3] + sb[3], abs(k - 260), trial)
        if best is None or grade < best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return a complete screened cell without redrawing its labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support variation on the parent's (1,3)x(1,3) bed
# suffices to remove weight-4 logicals: all five measured slots had d30s=4.
# REFUTED: The previous (1,4)x(1,4) penalty screen suffices to remove
# short logicals: its five measured slots all retained d30s=3 witnesses.
# REFUTED: Triangle-free binomial gaps on the parent's order-34
# (1,4)x(1,4) bed eliminate weight-four logicals: all five measured
# slots retained d30s=4, with thousands of verified weight-four words.
# REFUTED: Noncommuting order-six differences in S3 x C3 on
# (1,4)x(2,7) lift the rate-0.500 distance above four: all five
# measured slots retained sector-1 weight-four logicals.
# REFUTED: On the earlier C26 disjoint-doubleton filling, screening
# sixteen label trials reliably made every slot distance six: its
# distinct slots also included distance-four and distance-five readings.
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
