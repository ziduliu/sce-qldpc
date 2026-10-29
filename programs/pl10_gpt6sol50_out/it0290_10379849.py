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
CONCEPT = ("In C7 x A4 of order 84, A=(1,2) and B=(1,3), coset-structured voltages screened for short component relations may reduce worst-basis circuit failure.")
GROUP = ("Jump from dihedral order 58 to the nonabelian direct product C7 x A4. Order 84 gives n=588; measured order-84 (1,2)x(1,3) beds provide comparisons, while this group tests a different voltage structure.")
SHAPES = ("(1,2)x(1,3) gives n=588 and guaranteed k>=168, rate 0.286 in cell 0.300. Its one-row components have unequal lengths 168 and 252. The parent's short witnesses occupied sector-1 Acol/Bcol blocks, with no persistent failed-shot check-block hotspot.")
FILLING = ("Cross five legal support designs with four A4-coset patterns, and vary C7 gaps within each design. Screen independent fillings on verified component columns; check final rank and rooted Tanner cycles. This tests whether the parent's roughly 100 worse-basis failures within 1500-3250 shots become rarer, without treating d2s as fitness.")
FAMILY_NAME = "r030_c7xa4_q84_12x13_coset_component_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Direct product C7 x A4; A4 is the even permutations of four points.
    from itertools import permutations

    ps = []
    for p in permutations(range(4)):
        inversions = sum(p[i] > p[j] for i in range(4)
                         for j in range(i + 1, 4))
        if inversions % 2 == 0:
            ps.append(p)
    ps.sort()
    pos = {p: i for i, p in enumerate(ps)}
    compose = [[pos[tuple(p[t] for t in h)] for h in ps] for p in ps]
    inverses = [pos[tuple(p.index(t) for t in range(4))] for p in ps]
    els = [(a, b) for b in range(12) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 12)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    def mul(g, h):
        return ((g[0] + h[0]) % 7, compose[g[1]][h[1]])

    def inv(g):
        return ((-g[0]) % 7, inverses[g[1]])

    return dict(elements=els, ident=(0, pos[(0, 1, 2, 3)]),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb); n = 7 * 84 = 588


def _component_columns(rows, q, width, stride):
    """Read component columns from the evaluator-identical lifted rows."""
    columns = [0] * (width * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:q]):
        for j in range(width):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                bit = bits & -bits
                columns[j * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit
    return columns


def _short_signature(columns):
    """Component pair relations and Tanner four-cycles."""
    two = len(columns) - len(set(columns))
    pairs = {}
    four = cycles = 0
    for j, right in enumerate(columns):
        for i in range(j):
            left = columns[i]
            overlap = (left & right).bit_count()
            cycles += overlap * (overlap - 1) // 2
            syndrome = left ^ right
            first = pairs.get(syndrome)
            if first is None:
                pairs[syndrome] = (i, j)
            elif i not in first and j not in first:
                four += 1
    return two, four, cycles


def _a4_cosets(G):
    """The V4 subgroup and its two cosets inside A4."""
    normal = [b for b in range(12)
              if G.mul((0, b), (0, b)) == G.e]
    generator = next(b for b in range(12) if b not in normal)
    first = [G.mul((0, generator), (0, b))[1] for b in normal]
    second = [G.mul((0, generator), (0, b))[1] for b in first]
    assert len(normal) == len(first) == len(second) == 4
    assert len(set(normal + first + second)) == 12
    return normal, first, second


def _draw_cell(rng, cosets, count, mode, ratio, position):
    """Use distinct C7 x A4 labels with a prescribed quotient pattern."""
    patterns = ((0, 1, 2), (1, 1, 2), (0, 0, 1), (0, 1, 1))
    classes = list(patterns[mode][:count])
    if count > 3:
        classes.append((mode + position) % 3)
    start = rng.randrange(7)
    step = rng.randrange(1, 7)
    offsets = (0, 1, ratio, (ratio + 2) % 7)
    cell = []
    for term, cls in enumerate(classes):
        a = (start + step * offsets[term]) % 7
        choices = [(a, b) for b in cosets[cls] if (a, b) not in cell]
        cell.append(rng.choice(choices))
    rng.shuffle(cell)
    return cell


def _root_four_cycles(rows, q):
    """Count full-matrix four-cycles at one check per lifted block."""
    total = 0
    for root in range(0, len(rows), q):
        row = rows[root]
        for other, candidate in enumerate(rows):
            if other == root:
                continue
            shared = (row & candidate).bit_count()
            total += shared * (shared - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Screen the slot's support/coset design, not orbit relabelings."""
    profiles = (
        ((3, 3), (3, 3, 3)),
        ((3, 3), (3, 3, 3)),
        ((4, 3), (3, 3, 2)),
        ((4, 4), (3, 3, 2)),
        ((4, 4), (2, 2, 2)),
    )
    motif, mode = idx % 5, (idx // 5) % 4
    ac, bc = profiles[motif]
    if motif == 1:
        # A separate dense design: exchange the quotient patterns
        # between sides rather than relabeling an existing filling.
        mode_a, mode_b = mode, (mode + 2) % 4
    else:
        mode_a, mode_b = mode, (mode + 1) % 4
    sa, sb = [list(ac)], [list(bc)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    cosets = _a4_cosets(G)
    shortlist = []

    for trial in range(12):
        grids = {}
        for side, counts, pattern in (
                ("A", ac, mode_a), ("B", bc, mode_b)):
            row = []
            for j, count in enumerate(counts):
                ratio = 2 + ((idx + 2 * j + 3 * (side == "B")
                              + trial // 4) % 5)
                row.append(_draw_cell(rng, cosets, count, pattern,
                                      ratio, j + (side == "B")))
            grids[side] = [row]

        hx, hz, n = lifted_matrices(G, grids["A"], grids["B"])
        a = _short_signature(
            _component_columns(hx, G.q, A_SHAPE[1], B_SHAPE[1]))
        b = _short_signature(
            _component_columns(hz, G.q, B_SHAPE[1], 1))
        key = (a[0] + b[0], max(a[1], b[1]), a[1] + b[1],
               max(a[2], b[2]), a[2] + b[2], rng.random())
        shortlist.append((key, grids, hx, hz, n))

    shortlist.sort(key=lambda item: item[0])
    best_key = best = None
    for key, grids, hx, hz, n in shortlist[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        choice = (not 168 <= k <= 190, key[:3],
                  _root_four_cycles(hx, G.q)
                  + _root_four_cycles(hz, G.q),
                  key[3:], abs(k - 168))
        if best_key is None or choice < best_key:
            best_key, best = choice, grids
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the complete, screened cell without redrawing it."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled singleton-heavy supports suffice to suppress short sector-1 logicals at this bed: all five measured parent slots had weight-4 witnesses and worse-basis scores of only 3.17-3.34.
# REFUTED: Screening the current dihedral bed's coset-spanning cells for weight-2/4 signatures suffices to beat the shown cyclic (1,3)x(1,3) result: measured fitness was 4.4224 versus 4.5365, with every selected worse basis reaching the failure cap.
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
