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
CONCEPT = ("In cyclic Z26 of order 26, A=(1,3) and B=(2,7), indexed sparse B profiles screened for short component relations should improve worst-basis circuit fitness.")
GROUP = ("Keep the frozen Z1:Z26 metacyclic presentation: this bed scored 4.0817 with k=260 in its five measured codes, providing a filling-only comparison.")
SHAPES = ("Keep (1,3)x(2,7): n=598, guaranteed k=260 and rate 0.435 in cell 0.450. The unequal components have lengths 78 and 182. Parent failures were diffuse across sector-1 Acol/Bcol blocks, including Bcol2, Bcol4 and Bcol6, without a consistent check-block hotspot.")
FILLING = ("Cross five B support profiles, including eight-term rows and balanced 3/0 cells, with four A trinomial voltage motifs. Screen verified component lifts for relations through weight five, then verify k=260. Test whether fewer B terms and short relations reduce the parent's roughly 100-114 limiting-basis failures and frequent X-basis stage-three fallthroughs.")
FAMILY_NAME = "r045_m1r26s1_13x27_sparse_profiles_fiveword_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 26

def custom_ops(q):
    # Same metacyclic rule: Z_1 x| Z_26, with trivial action.
    m, r, s = 1, 26, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb); n = 23 * 26 = 598


def _component_relations(G, grid, side):
    """Short dependencies and four-cycles in the actual cyclic component."""
    columns = []
    for j in range(len(grid[0])):
        for h in G.elements:
            syndrome = 0
            for i, row in enumerate(grid):
                for g in row[j]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)

    singles = set(columns)
    pairs = {}
    twos = threes = fours = cycles = 0
    for i, x in enumerate(columns):
        for y in columns[i + 1:]:
            value = x ^ y
            twos += value == 0
            threes += value in singles
            fours += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
            shared = (x & y).bit_count()
            cycles += shared * (shared - 1) // 2
    return twos, threes, fours, cycles


def _verified_component_columns(hx, hz, q):
    """Extract A and B component columns from the exact LP matrices."""
    mask = (1 << q) - 1
    ac = [0] * (3 * q)
    bc = [0] * (7 * q)
    for check in range(q):
        row = hx[check]  # A-row zero, B-column zero
        for j in range(3):
            bits = (row >> (j * 7 * q)) & mask
            while bits:
                low = bits & -bits
                ac[j * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    for check in range(2 * q):
        row = hz[check]  # A-column zero, both B rows
        for j in range(7):
            bits = (row >> (j * q)) & mask
            while bits:
                low = bits & -bits
                bc[j * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return ac, bc


def _has_five_word(columns):
    """Detect a disjoint pair/triple collision in a component kernel."""
    pairs = {}
    for i, x in enumerate(columns):
        for j in range(i):
            value = x ^ columns[j]
            examples = pairs.setdefault(value, [])
            if len(examples) < 4:
                examples.append((i, j))
    for i, x in enumerate(columns):
        for j in range(i):
            partial = x ^ columns[j]
            for k in range(j):
                for u, v in pairs.get(partial ^ columns[k], ()):
                    if u not in (i, j, k) and v not in (i, j, k):
                        return True
    return False


def _relative_collision_cost(offsets):
    """Count repeated undirected differences between monomial columns."""
    seen = {}
    cost = 0
    for i, x in enumerate(offsets):
        for y in offsets[:i]:
            delta = (x - y) % 26
            delta = min(delta, 26 - delta)
            cost += seen.get(delta, 0)
            seen[delta] = seen.get(delta, 0) + 1
    return cost


def NEW_CTX(rng, G, idx):
    """Cross five support motifs with four voltage motifs; screen final lifts."""
    profile = idx % 5
    mode = (idx // 5) % 4
    odd_gaps = (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)
    even = tuple(range(0, 26, 2))
    odd = tuple(range(1, 26, 2))

    def triple(j, for_b=False):
        for _ in range(90):
            if for_b or mode == 3:
                terms = rng.sample(range(26), 3)
            elif mode == 2:
                start = rng.randrange(26)
                gap = rng.choice(odd_gaps)
                ratio = (4, 6, 8)[j]
                terms = [start, (start + gap) % 26,
                         (start + ratio * gap) % 26]
            else:
                primary = even if (mode == 0 or j == 1) else odd
                other = odd if primary is even else even
                terms = rng.sample(primary, 2) + [rng.choice(other)]
            distances = [
                min((x - y) % 26, (y - x) % 26)
                for t, x in enumerate(terms) for y in terms[:t]
            ]
            if len(set(terms)) == 3 and min(distances) >= 3 \
                    and 13 not in distances:
                return [(0, t) for t in terms]
        return [(0, t) for t in rng.sample(range(26), 3)]

    shortlist = []
    for _ in range(28 + idx % 7):
        A = [[triple(j) for j in range(3)]]
        order = rng.sample(range(7), 7)
        isolated = []
        reinforced = []
        if profile == 0:       # B rows 9/9: four split binomials
            reinforced = [(0, j) for j in order[:2]]
            reinforced += [(1, j) for j in order[2:4]]
        elif profile == 1:     # B rows 8/8: five monomial columns
            reinforced = [(0, order[0]), (1, order[1])]
        elif profile == 2:     # B rows 9/8: four monomial columns
            reinforced = [(0, j) for j in order[:2]]
            reinforced += [(1, order[2])]
        elif profile == 3:     # B rows 9/9: one isolated 3/0 per row
            isolated = [(0, order[0]), (1, order[1])]
            reinforced = [(0, order[2]), (1, order[3])]
        else:                  # B rows 8/8: isolated 3/0 cells
            isolated = [(0, order[0]), (1, order[1])]

        active = {j for _, j in isolated + reinforced}
        monomial = [j for j in range(7) if j not in active]
        offsets = min(
            (rng.sample(range(26), 7) for _ in range(45)),
            key=lambda values: (
                _relative_collision_cost([values[j] for j in monomial]),
                rng.random()))

        B = [[None] * 7 for _ in range(2)]
        for j in range(7):
            start = rng.randrange(26)
            B[0][j] = [(0, start)]
            B[1][j] = [(0, (start + offsets[j]) % 26)]
        for row, j in isolated:
            B[row][j] = triple(j % 3, for_b=True)
            B[1 - row][j] = []
        for row, j in reinforced:
            start = B[row][j][0][1]
            gap = rng.choice(odd_gaps)
            B[row][j].append((0, (start + gap) % 26))

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)

        ar = _component_relations(G, A, "A")
        br = _component_relations(G, B, "B")
        cheap = (ar[0] + br[0], ar[1] + br[1],
                 max(ar[2], br[2]), ar[2] + br[2],
                 max(ar[3], br[3]), ar[3] + br[3],
                 _relative_collision_cost(
                     [offsets[j] for j in monomial]),
                 rng.random())
        shortlist.append((cheap, A, B))

    shortlist.sort(key=lambda item: item[0])
    finalists = []
    for cheap, A, B in shortlist[:6]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 260:
            continue
        ac, bc = _verified_component_columns(hx, hz, G.q)
        b_five = _has_five_word(bc)
        a_five = _has_five_word(ac)
        terms = sum(len(cell) for row in B for cell in row)
        quality = (b_five or a_five, cheap[:4], terms,
                   cheap[4:])
        finalists.append((quality, A, B))
    if finalists:
        finalists.sort(key=lambda entry: entry[0])
        return {"A": finalists[0][1], "B": finalists[0][2]}
    _, A, B = shortlist[0]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Parent slot 12 refutes that w=12 and ASC depth 13 suffice for low pL:
# its X basis had 109 failures in 750 shots and scored s=3.499.
# Spaced A trinomials with otherwise unscreened sparse B voltages do
# not suffice on this bed: all five measured codes hit the failure cap
# in both bases, with fitness 3.8805 and weight-six sector-1 witnesses.
# Full-order gaps and parity-patterned A trinomials did not make
# failures rare: the current program scored 4.0817, and all five
# measured codes hit the failed-shot cap in both bases.
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
