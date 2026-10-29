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
CONCEPT = ("In cyclic Z37 of order 37, A=(2,3) and B=(2,4), independently screened binomial gap spectra may produce five distinct low-failure codes in rate cell 0.150.")
GROUP = ("Keep the cyclic construction rule but change its order to 37. The measured Z37 (2,3)x(2,4) bed reached 7.4, and order 37 gives n=592 for these shapes.")
SHAPES = ("(2,3)x(2,4) gives n=592, guaranteed k>=74 and rate>=0.125; rank deficits can put measured k in cell 0.150. Its A and B components have unequal lengths, 111 and 148. The parent's weight-four witnesses occupied Acol/Bcol sector-1 blocks, while failed shots had no persistent check-block hotspot.")
FILLING = ("Give every cell two distinct elements: HX rows weigh 6+4=10 and HZ rows weigh 8+4=12. Index inequivalent A and B gap spectra, screen independently labelled candidates on verified component lifts for short relations and cycles, and check k. Test for fewer than the parent's 110-131 Z failures per 250 shots and frequent X failures, not just a higher distance proxy.")
FAMILY_NAME = "r015_z37_23x24_indexed_binomial_spectra_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # The same cyclic rule, with m=37, r=1 and trivial action s=1.
    m, r, s = 37, 1, 1
    els = [(a, 0) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, 0)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        return ((-g[0]) % m, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: int(g[0]) % m, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 16 * 37 = 592





def _gap_orbit(gaps):
    """Undirected gap multiset, canonical under Z37 automorphisms."""
    return min(tuple(sorted(min((u * d) % 37, 37 - (u * d) % 37)
                            for d in gaps))
               for u in range(1, 37))


def _spectral_designs():
    """Give each slot a different A spectrum, not an orbit copy."""
    source = random.Random(370231)
    designs, seen_a, seen_b = [], set(), set()
    while len(designs) < max(20, N_SLOTS):
        ag = tuple(sorted(source.sample(range(1, 19), 6)))
        bg = tuple(sorted(source.sample(range(1, 19), 8)))
        ak, bk = _gap_orbit(ag), _gap_orbit(bg)
        if ak in seen_a or bk in seen_b:
            continue
        seen_a.add(ak)
        seen_b.add(bk)
        designs.append((ag, bg))
    return tuple(designs)


_SPECTRAL_DESIGNS = _spectral_designs()


def _label_component(rng, gaps, width):
    """Fresh origins and signs for a complete two-row binomial side."""
    assignments = rng.sample(list(gaps), 2 * width)
    grid = []
    for i in range(2):
        row = []
        for j in range(width):
            start = rng.randrange(37)
            gap = assignments[i * width + j]
            if rng.randrange(2):
                gap = -gap
            row.append([(start, 0), ((start + gap) % 37, 0)])
        grid.append(row)
    return grid


def _verified_component_signatures(hx, hz, q):
    """Read both components from the exact HX/HZ sector-1 blocks."""
    mask = (1 << q) - 1
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    a, b = [0] * (ca * q), [0] * (cb * q)
    for i in range(2):
        for r in range(q):
            row = hx[(i * cb) * q + r]  # B-column zero
            for j in range(ca):
                bits = (row >> (j * cb * q)) & mask
                while bits:
                    bit = bits & -bits
                    a[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    bits ^= bit
    for alpha in range(2):
        for r in range(q):
            row = hz[alpha * q + r]  # A-column zero
            for beta in range(cb):
                bits = (row >> (beta * q)) & mask
                while bits:
                    bit = bits & -bits
                    b[beta * q + bit.bit_length() - 1] |= 1 << (alpha * q + r)
                    bits ^= bit
    return a, b


def _component_relations(columns):
    """Screen component kernel relations of weights one through four."""
    singles = set(columns)
    pairs = {}
    one = two = three = four = 0
    for i, left in enumerate(columns):
        one += left == 0
        for right in columns[:i]:
            value = left ^ right
            two += value == 0
            three += value in singles
            four += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
    return one, two, three, four


def _component_cycles(columns, q):
    """Count rooted four- and six-cycles of the verified component."""
    checks = [0] * (2 * q)
    for variable, syndrome in enumerate(columns):
        bits = syndrome
        while bits:
            bit = bits & -bits
            checks[bit.bit_length() - 1] |= 1 << variable
            bits ^= bit

    fours = sixes = 0
    for root in (0, q):
        neighbors = []
        for j, row in enumerate(checks):
            if j == root:
                continue
            shared = checks[root] & row
            size = shared.bit_count()
            fours += size * (size - 1) // 2
            if shared:
                neighbors.append((j, shared, size))
        for t, (j, rj, nj) in enumerate(neighbors):
            for k, rk, nk in neighbors[:t]:
                jk = (checks[j] & checks[k]).bit_count()
                common = (rj & rk).bit_count()
                sixes += (nj * jk * nk -
                          common * (nj + jk + nk) + 2 * common)
    return fours, sixes


def NEW_CTX(rng, G, idx):
    """Screen labels within one indexed, inequivalent gap-spectrum design."""
    agaps, bgaps = _SPECTRAL_DESIGNS[idx]
    finalists = []
    for _ in range(19 + idx % 6):
        a = _label_component(rng, agaps, 3)
        b = _label_component(rng, bgaps, 4)
        hx, hz, _ = lifted_matrices(G, a, b)
        ac, bc = _verified_component_signatures(hx, hz, G.q)
        ar, br = _component_relations(ac), _component_relations(bc)
        ax, bx = _component_cycles(ac, G.q), _component_cycles(bc, G.q)
        key = (ar[0] + br[0], ar[1] + br[1], ar[2] + br[2],
               max(ax[0], bx[0]), ax[0] + bx[0],
               max(ax[1], bx[1]), ax[1] + bx[1],
               ar[3] + br[3], rng.random())
        finalists.append((key, a, b, hx, hz))
    finalists.sort(key=lambda item: item[0])

    chosen = None
    for cost, a, b, hx, hz in finalists[:6]:
        k = 592 - gf2_rank(hx) - gf2_rank(hz)
        # Cell 0.150 contains integer k=74..103 at n=592.
        quality = (not 74 <= k <= 103, abs(k - 88), cost)
        if chosen is None or quality < chosen[0]:
            chosen = (quality, a, b)
    return {"A": chosen[1], "B": chosen[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete binomial cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Separated differences with the parent's mixed B doubles did not prevent 185-199/250 X-basis failures or sector-1 weight-three Z witnesses in its five measured slots.
# Mostly monomial B with distinct A gaps did not give diverse, reliably protected codes: only seven slots were distinct, and a measured slot failed 179/250 X-basis shots.
# Triangle-free A-binomial gaps with distributed B doubles did not improve the sparse-B parent: fitness fell from 3.3369 to 3.2638, and the five measured slots had 131-166 X failures per 250 shots.
# Two A trinomials with monomial B did not suppress circuit failures: the five measured slots had 110-131 Z failures per 250 shots and many equivalent fillings were deduplicated.
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
