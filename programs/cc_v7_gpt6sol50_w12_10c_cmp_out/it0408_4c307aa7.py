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
CONCEPT = ("On (C4 x C4) semidirect C2 of order 32 with A=(1,3), "
           "B=(2,5), screened, distinct binomial-order profiles will "
           "produce five rate-0.350 codes scoring above 22.588.")
GROUP = ("Replace C8 semidirect_3 C4 by the factor-swapping action on "
         "C4 x C4. Its cyclic centre C4 differs from the parent's "
         "C2 x C2 centre. Order 32 preserves n=544 for a direct test.")
SHAPES = ("Keep A=(1,3), B=(2,5): n=17*32=544 and guaranteed "
          "k=6*32=192, rate 0.353. The unequal one-row and two-row "
          "components have 96 and 160 variables. The parent's lightest "
          "X and Z witnesses used sector-one Acol/Bcol blocks; Z "
          "witnesses prominently touched Bcol2-3.")
FILLING = ("Use three distinct elements per A cell and a singleton "
           "plus a binomial per B column. A weighs nine and B rows "
           "eight and seven, giving HX<=12 and HZ<=11. Twenty "
           "different counts of order-eight and two kinds of order-four "
           "B gaps define intrinsic profiles; independently screen both "
           "verified components through weight six. Test whether the "
           "parent's weight-eight witnesses change while k stays near 192.")
FAMILY_NAME = "r035_c4wr2_13x25_gap_order_component_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # (C4 x C4) semidirect C2; its generator swaps the C4 factors.
    els = [(a, c, b) for b in range(2)
           for c in range(4) for a in range(4)]

    def red(g):
        return (g[0] % 4, g[1] % 4, g[2] % 2)

    def idx(g):
        a, c, b = red(g)
        return 16 * b + 4 * c + a

    def mul(g, h):
        a, c, b = red(g)
        d, e, f = red(h)
        if b:
            d, e = e, d
        return ((a + d) % 4, (c + e) % 4, (b + f) % 2)

    def inv(g):
        a, c, b = red(g)
        if b:
            return ((-c) % 4, (-a) % 4, 1)
        return ((-a) % 4, (-c) % 4, 0)

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


_GAP_PROFILES = None


def _gap_profiles():
    """Twenty different counts of intrinsic B-gap order classes.

    The omitted twenty-first profile has only normal-subgroup gaps.
    Counts, unlike element labels or column permutations, cannot be
    changed by a group automorphism.
    """
    global _GAP_PROFILES
    if _GAP_PROFILES is None:
        _GAP_PROFILES = [
            (eight, swap_four)
            for eight in range(6)
            for swap_four in range(6 - eight)
            if (eight, swap_four) != (0, 0)
        ]
    return _GAP_PROFILES


def _component_columns(rows, q, blocks, stride, check_blocks, nrows):
    """Extract every component-check row from the verified LP lift."""
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for i in range(nrows):
        for r in range(q):
            row = rows[(i * check_blocks) * q + r]
            check = 1 << (i * q + r)
            for j in range(blocks):
                bits = (row >> (j * stride * q)) & mask
                while bits:
                    bit = bits & -bits
                    columns[j * q + bit.bit_length() - 1] |= check
                    bits ^= bit
    return columns


def _component_merit(columns):
    """Lexicographically count words of weights 2, 4, 6 and short cycles.

    All constructed columns have odd weight three, so odd-weight
    component words cannot be in the kernel. Pair and triple syndrome
    collisions detect the remaining weights through six.
    """
    if 0 in columns or len(set(columns)) != len(columns):
        return (1, 0, 0, 0)

    pairs = set()
    overlap = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in pairs:
                return (0, 1, 0, 0)
            pairs.add(syndrome)
            shared = (x & y).bit_count()
            overlap += shared * (shared - 1) // 2

    triples = set()
    sixes = 0
    for i in range(len(columns) - 2):
        x = columns[i]
        for j in range(i + 1, len(columns) - 1):
            xy = x ^ columns[j]
            for k in range(j + 1, len(columns)):
                syndrome = xy ^ columns[k]
                if syndrome in triples:
                    sixes += 1
                    if sixes >= 102:
                        return (0, 0, sixes, overlap)
                else:
                    triples.add(syndrome)
    return (0, 0, sixes, overlap)


def _label_a(rng, G, motif):
    """Three distinct terms per cell, straddling the swapping coset."""
    normal = [g for g in G.elements if g[2] == 0]
    swapped = [g for g in G.elements if g[2] == 1]
    row = []
    for j in range(3):
        ns = 1 + ((motif + j) % 2)
        row.append(rng.sample(normal, ns) +
                   rng.sample(swapped, 3 - ns))
    return [row]


def _label_b(rng, G, profile):
    """One singleton and one binomial per column, with typed gaps."""
    eight, swap_four = profile
    types = (["eight"] * eight + ["swap_four"] * swap_four +
             ["normal_four"] * (5 - eight - swap_four))
    rng.shuffle(types)
    pools = {
        "eight": [g for g in G.elements
                  if g[2] == 1 and (g[0] + g[1]) % 2 == 1],
        "swap_four": [g for g in G.elements
                      if g[2] == 1 and (g[0] + g[1]) % 4 == 2],
        "normal_four": [g for g in G.elements
                        if g[2] == 0 and (g[0] % 2 or g[1] % 2)],
    }
    doubled_in_first = set(rng.sample(range(5), 3))
    used_gaps = set()
    grid = [[None] * 5 for _ in range(2)]
    for j, kind in enumerate(types):
        available = [g for g in pools[kind] if G.idx(g) not in used_gaps]
        gap = rng.choice(available or pools[kind])
        used_gaps.add(G.idx(gap))
        doubled = 0 if j in doubled_in_first else 1
        base = G.pick(rng)
        grid[doubled][j] = [base, G.mul(base, gap)]
        grid[1 - doubled][j] = [G.pick(rng)]
    return grid


def NEW_CTX(rng, G, idx):
    """Screen independently labeled components of one intrinsic profile."""
    profile = _gap_profiles()[idx % 20]
    best_a = best_b = None
    merit_a = merit_b = None

    for trial in range(26):
        A = (best_a if merit_a is not None and merit_a[:3] == (0, 0, 0)
             else _label_a(rng, G, idx % 20))
        B = (best_b if merit_b is not None and merit_b[:3] == (0, 0, 0)
             else _label_b(rng, G, profile))
        hx, hz, _ = lifted_matrices(G, A, B)

        if merit_a is None or merit_a[:3] != (0, 0, 0):
            a_columns = _component_columns(hx, G.q, 3, 5, 5, 1)
            sa = _component_merit(a_columns)
            if merit_a is None or sa < merit_a:
                merit_a, best_a = sa, A

        if merit_b is None or merit_b[:3] != (0, 0, 0):
            b_columns = _component_columns(hz, G.q, 5, 1, 1, 2)
            sb = _component_merit(b_columns)
            if merit_b is None or sb < merit_b:
                merit_b, best_b = sb, B

        if merit_a[:3] == (0, 0, 0) and merit_b[:3] == (0, 0, 0):
            break

    return {"A": best_a, "B": best_b}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Rolled singleton-heavy supports do not reliably avoid light logicals:
# all five measured parent slots had d30s=5, despite distinct labels.
# REFUTED: Disjoint doubles plus screening through component weight five
# suffice to raise distance above six: all five current measured slots
# retained d30s=6, with weight-six sector-1 X witnesses.
# REFUTED: The previous gap-orbit screen yielded five distance-eight
# readings: its measured distances were 8, 8, 8, 6, 6. Its A extraction
# inspected only the first of two component-check rows.
# REFUTED: Screening three-regular components on C8 semidirect_3 C4
# beats the prior distance-eight score: its five measured slots all
# retained k=192 and d30s=8, giving the unchanged fitness 22.588.
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
