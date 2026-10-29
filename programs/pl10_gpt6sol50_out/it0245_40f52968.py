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
CONCEPT = ("In C4 x C4 of order 16, shapes (2,5)x(2,6), degree-three component columns with distinct order-four binomial gaps may reduce worst-basis circuit failures.")
GROUP = ("Jump from cyclic Z13 to noncyclic C4 x C4. Its order 16 gives n=544 on this untested group-shape bed. The related Z17 shape bed reached about 4.6, but whether fewer gates offset a shorter lift requires measurement.")
SHAPES = ("(2,5)x(2,6) guarantees k>=192 of n=544, rate>=0.3529 in cell 0.350; its component lengths are 80 and 96, not equal. The parent's weight-six witnesses and enriched failures occupied sector-1 Acol/Bcol blocks. Its slot 22 had 106 X failures in 2500 shots versus 103 Z failures in 11250 shots.")
FILLING = ("Give every component column one binomial and one monomial. Distinct order-four gaps within each side avoid order-two binomials; indexed supports and independently screened labels distinguish complete designs. HX<=11 and HZ<=12. Screen verified component relations, rank and Tanner cycles to test whether the parent's frequent X failures fall; distance is not the fitness.")
FAMILY_NAME = "r035_c4xc4_q16_25x26_order4_degree3_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Direct product C4 x C4, not a cyclic group of order 16.
    els = [(a, b) for b in range(4) for a in range(4)]

    def red(g):
        return (g[0] % 4, g[1] % 4)

    def mul(g, h):
        return ((g[0] + h[0]) % 4, (g[1] + h[1]) % 4)

    def inv(g):
        return ((-g[0]) % 4, (-g[1]) % 4)

    def idx(g):
        a, b = red(g)
        return 4 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb); n = 34 * 16 = 544


def _four_cycles(rows):
    """Count pairs of verified check rows sharing two qubits."""
    total = 0
    for i, row in enumerate(rows):
        for earlier in rows[:i]:
            overlap = (row & earlier).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _supports(rng, idx):
    """Twenty indexed complementary-column support motifs."""
    a_order = rng.sample(range(5), 5)
    b_order = rng.sample(range(6), 6)
    # The twenty three-of-six B choices are distinct indexed motifs.
    from itertools import combinations
    b_first = set(combinations(range(6), 3)[idx % 20])
    a_first = set(a_order[:2 + (idx // 20) % 2])
    sa = [[1 + (j in a_first) for j in range(5)],
          [1 + (j not in a_first) for j in range(5)]]
    sb = [[1 + (t in b_first) for t in range(6)],
          [1 + (t not in b_first) for t in range(6)]]
    # Permute column labels independently of the indexed double-row
    # incidence; fresh element choices define complete, non-orbit designs.
    sb = [[row[b_order[j]] for j in range(6)] for row in sb]
    assert _grid_weights(sa, sb) == 12
    return sa, sb


def _order_four_gaps(G):
    """Six inverse classes of order-four elements in C4 x C4."""
    gaps, seen = [], set()
    for g in G.elements:
        if G.mul(g, g) == G.e or G.idx(g) in seen:
            continue
        seen.add(G.idx(g))
        seen.add(G.idx(G.inv(g)))
        gaps.append(g)
    assert len(gaps) == 6
    return gaps


def _draw_side(rng, G, supports, gaps):
    """Draw monomials and distinct-label, order-four binomials."""
    gaps = list(gaps)
    rng.shuffle(gaps)
    grid = []
    for row in supports:
        filled = []
        for count in row:
            start = G.pick(rng)
            if count == 1:
                filled.append([start])
            else:
                gap = gaps.pop()
                if rng.randrange(2):
                    gap = G.inv(gap)
                filled.append([start, G.mul(start, gap)])
        grid.append(filled)
    return grid


def _component_columns(hx, hz, q):
    """Read both component incidences from the verified LP matrices."""
    a, b = [], []
    for j in range(5):
        for t in range(q):
            syndrome = 0
            for i in range(2):
                for r in range(q):
                    row = hx[i * 6 * q + r]  # B column zero
                    syndrome |= ((row >> ((j * 6) * q + t)) & 1) << (i * q + r)
            a.append(syndrome)
    for beta in range(6):
        for t in range(q):
            syndrome = 0
            for alpha in range(2):
                for r in range(q):
                    row = hz[alpha * q + r]  # A column zero
                    syndrome |= ((row >> (beta * q + t)) & 1) << (alpha * q + r)
            b.append(syndrome)
    return a, b


def _pair_quality(columns):
    """Count repeated component columns, four-relations and four-cycles."""
    seen = {}
    twos = fours = cycles = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = left ^ right
            twos += value == 0
            fours += seen.get(value, 0)
            seen[value] = seen.get(value, 0) + 1
            shared = (left & right).bit_count()
            cycles += shared * (shared - 1) // 2
    return twos, fours, cycles


def _six_defects(columns, limit=40):
    """Bound disjoint equal-syndrome triples in a component."""
    seen = {}
    defects = 0
    for i in range(len(columns) - 2):
        for j in range(i + 1, len(columns) - 1):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, len(columns)):
                value = partial ^ columns[k]
                old = seen.get(value)
                if old is None:
                    seen[value] = (i, j, k)
                elif i not in old and j not in old and k not in old:
                    defects += 1
                    if defects >= limit:
                        return defects
    return defects


def NEW_CTX(rng, G, idx):
    """Select independent complete fillings for this indexed motif."""
    sa, sb = _supports(rng, idx)
    gaps = _order_four_gaps(G)
    pool = []
    for _ in range(14 + idx % 5):
        # B receives all six inverse gap classes; A receives five.
        # Starts are independent, so these are not relabelings of one code.
        A = _draw_side(rng, G, sa, rng.sample(gaps, 5))
        B = _draw_side(rng, G, sb, gaps)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        aq, bq = _pair_quality(ac), _pair_quality(bc)
        short = (aq[0] + bq[0], bq[1], aq[1],
                 bq[2], aq[2])
        pool.append((short, rng.random(), A, B, hx, hz, n, ac, bc))
    pool.sort(key=lambda item: item[:2])

    best, best_key = None, None
    for short, _, A, B, hx, hz, n, ac, bc in pool[:5]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        six_b = _six_defects(bc)
        six_a = _six_defects(ac)
        key = (k > 203, short[:3], six_b, six_a,
               short[3:], _four_cycles(hx) + _four_cycles(hz),
               max(0, k - 192))
        if best_key is None or key < best_key:
            best_key, best = key, {"A": A, "B": B}
    if best is None:
        raise ValueError("no positive-k C4 x C4 filling")
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing labels."""
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
# Complementary sparse Z13 supports plus rank/four-cycle selection
# suffice to balance both bases: refuted by the current parent's 4.6001
# fitness; measured slot 22 had 106 X failures in 2500 shots, versus
# 103 Z failures in 11250 shots.
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
