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
CONCEPT = ("On Q8 semidirect C3 of order 24 with A=(1,3), B=(2,7), "
           "action-spanning A cells and screened B profiles can yield "
           "five d30s>=6 codes, potentially including distance seven.")
GROUP = ("Jump from cyclic Z26 to the nonabelian order-24 quaternion "
         "semidirect C3 group. The related Dic6 order-24 bed measured "
         "five distance-six codes, but this group's action and its "
         "order-four and order-six elements give a different lift.")
SHAPES = ("Keep (1,3)x(2,7): n=23*24=552 and guaranteed k>=240, "
          "rate 0.435 in cell 0.450. A is a three-column one-check "
          "component; B is a seven-column two-check component. The "
          "parent's lightest X/Z logicals occupied sector-1 Acol/Bcol "
          "blocks, motivating screens on both components.")
FILLING = ("Cross five distinct legal support profiles with four "
           "action-coset modes; independently draw noncommuting labels "
           "and select against short relations in verified components. "
           "Exact-rank selection keeps k near 240. Every profile has "
           "HX and HZ row weights at most 12.")
FAMILY_NAME = "r045_q8sd_c3_q24_13x27_action_component_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Q8 x| C3: the C3 generator cycles i -> j -> k -> i.
    els = [(u, b) for b in range(3) for u in range(8)]
    basis = ((0, 1, 2, 3), (1, 0, 3, 2),
             (2, 3, 0, 1), (3, 2, 1, 0))
    negative = ((0, 0, 0, 0), (0, 1, 0, 1),
                (0, 1, 1, 0), (0, 0, 1, 1))
    action = ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2))

    def red(g):
        return (g[0] % 8, g[1] % 3)

    def qmul(u, v):
        x, y = u % 4, v % 4
        return basis[x][y] + 4 * ((u // 4) ^ (v // 4)
                                  ^ negative[x][y])

    def apply(u, b):
        return action[b][u % 4] + 4 * (u // 4)

    def mul(g, h):
        u, b = red(g)
        v, d = red(h)
        return (qmul(u, apply(v, b)), (b + d) % 3)

    def inv(g):
        u, b = red(g)
        ui = u if u % 4 == 0 else u ^ 4
        bi = (-b) % 3
        return (apply(ui, bi), bi)

    def idx(g):
        u, b = red(g)
        return 8 * b + u

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _component_columns(rows, q, side):
    """Extract component columns from the evaluator-equivalent LP lift."""
    blocks = 3 if side == "A" else 7
    check_rows = q if side == "A" else 2 * q
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for r, row in enumerate(rows[:check_rows]):
        for j in range(blocks):
            base = j * 7 * q if side == "A" else j * q
            bits = (row >> base) & mask
            while bits:
                bit = bits & -bits
                columns[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
    return columns


def _short_relations(columns):
    """Screen zero, repeated, three-word and four-word syndromes."""
    singles = set(columns)
    bad2 = len(columns) - len(singles) + 20 * int(0 in singles)
    pairs = {}
    bad3 = bad4 = 0
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            syndrome = first ^ second
            bad3 += syndrome in singles
            bad4 += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return bad2, bad3, bad4


def _rooted_fives(columns, q, cap=12):
    """Find five-word component relations with one translated root."""
    pairs = {}
    for i, first in enumerate(columns):
        for j in range(i):
            pairs.setdefault(first ^ columns[j], (j, i))
    hits = 0
    for root in range(0, len(columns), q):
        pivot = columns[root]
        for i, first in enumerate(columns):
            if i == root:
                continue
            for j in range(i):
                if j == root:
                    continue
                pair = pairs.get(pivot ^ first ^ columns[j])
                if pair is not None and root not in pair \
                        and i not in pair and j not in pair:
                    hits += 1
                    if hits >= cap:
                        return hits
    return hits


def _a_sixes(columns, cap=16):
    """Count disjoint triple-syndrome collisions in the A component."""
    triples = {}
    hits = 0
    size = len(columns)
    for i in range(size - 2):
        for j in range(i + 1, size - 1):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, size):
                syndrome = partial ^ columns[k]
                prior = triples.get(syndrome)
                if prior is None:
                    triples[syndrome] = (i, j, k)
                elif i not in prior and j not in prior and k not in prior:
                    hits += 1
                    if hits >= cap:
                        return hits
    return hits


def _group_order(G, g):
    product = G.e
    for order in range(1, G.q + 1):
        product = G.mul(product, g)
        if product == G.e:
            return order
    raise ValueError("element order exceeds group order")


def _design(rng, G, motif, mode, order4, order6):
    """Draw an independent member of one indexed support/coset design."""
    sites = rng.sample(range(7), 7)
    ac = ([3, 3, 2] if motif == 3 else
          [4, 3, 2] if motif == 4 else [3, 3, 3])
    rng.shuffle(ac)
    bc = [[1] * 7 for _ in range(2)]

    if motif == 0:
        for col in sites[:2]:
            bc[0][col] = 2
        for col in sites[2:4]:
            bc[1][col] = 2
    elif motif == 1:
        bc[0][sites[0]], bc[1][sites[0]] = 3, 0
        for col in sites[1:4]:
            bc[1][col] = 2
    elif motif == 2:
        bc[0][sites[0]], bc[1][sites[0]] = 3, 0
        bc[1][sites[1]], bc[0][sites[1]] = 3, 0
        bc[0][sites[2]], bc[1][sites[3]] = 2, 2
    elif motif == 3:
        for col in sites[:2]:
            bc[0][col] = 2
        for col in (sites[0], sites[2]):
            bc[1][col] = 2
    else:
        bc[0][sites[0]], bc[1][sites[1]] = 2, 2

    assert _grid_weights([ac], bc) <= 12

    def a_cell(count, j):
        if count == 2:
            return [rng.choice([(u, 0) for u in range(8)]),
                    rng.choice([(u, 1 + (mode % 2))
                                for u in range(8)])]
        if mode == 0:
            cosets = [0, 1, 2]
        elif mode == 1:
            cosets = [0, 0, 1]
        elif mode == 2:
            cosets = [0, 1, 2]
        else:
            cosets = [1, 1, 2]
        if count == 4:
            cosets.append((j + mode) % 3)
        cell = []
        for coset in cosets:
            options = [(u, coset) for u in range(8)
                       if (u, coset) not in cell]
            cell.append(rng.choice(options))
        return cell

    A = [[a_cell(count, j) for j, count in enumerate(ac)]]
    B = [[], []]
    for r in range(2):
        for j in range(7):
            count = bc[r][j]
            if not count:
                cell = []
            elif count == 1:
                cell = [G.pick(rng)]
            else:
                base = G.pick(rng)
                pool = (order6 if mode == 0 else
                        order4 if mode == 1 else
                        order4 + order6 if mode == 2 else order6)
                gap = rng.choice(pool)
                cell = [base, G.mul(base, gap)]
                if count == 3:
                    # A third, independently drawn action-coset label
                    # distinguishes a triple from a repeated binomial.
                    options = [g for g in G.elements
                               if g not in cell and
                               g[1] != cell[0][1]]
                    cell.append(rng.choice(options))
            B[r].append(cell)
    return A, B


def NEW_CTX(rng, G, idx):
    """Select an independently filled indexed design on verified lifts."""
    motif, mode = idx % 5, (idx // 5) % 4
    order4 = [g for g in G.elements if _group_order(G, g) == 4]
    order6 = [g for g in G.elements if _group_order(G, g) == 6]
    candidates = []

    for trial in range(24):
        A, B = _design(rng, G, motif, mode, order4, order6)
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, "A")
        bc = _component_columns(hz, G.q, "B")
        sa, sb = _short_relations(ac), _short_relations(bc)
        preliminary = (sa[0] + sb[0], sa[1] + sb[1],
                       sa[2] + sb[2], max(sa[2], sb[2]))
        candidates.append((preliminary, trial, A, B, hx, hz, n,
                           ac, bc))

    best = None
    for preliminary, trial, A, B, hx, hz, n, ac, bc in sorted(
            candidates, key=lambda item: item[:2])[:7]:
        fives = (_rooted_fives(ac, G.q) +
                 _rooted_fives(bc, G.q))
        sixes = _a_sixes(ac) if preliminary[:3] == (0, 0, 0) else 16
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (not 240 <= k < 263, preliminary[:3],
                 fives, sixes, preliminary[3],
                 abs(k - 240), trial)
        if best is None or grade < best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Uniform rolled Z23 labels at (2,4)x(3,5) did not usually exceed Z-distance six: four of the parent's five measured slots had d30s=6.
# REFUTED: Minimizing Tanner four-cycles on D23 (1,3)x(1,4) did not raise measured distance above four: all five selected parent slots had d30s=4.
# REFUTED: Eight-trial component screening with sparse B on D13 (1,3)x(2,7) reliably removes weight-four logicals: four of five measured slots had d30s=4.
# REFUTED: Distinct D13 B inter-row rotation differences and 28 trials
# suffice to exceed distance five: all five selected slots had d30s=5.
# REFUTED: Cyclic Z26 order-pattern screening consistently exceeds five:
# four of its five measured slots had d30s=5; only slot 18 reached six.
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
