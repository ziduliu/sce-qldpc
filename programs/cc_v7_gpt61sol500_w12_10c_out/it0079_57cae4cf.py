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
CONCEPT = ("SL(2,3) x C1, order 24, shapes (1,3)x(1,5): order-four/order-six binomials selected by exact CSS logical classes should turn enough short commuting cycles into stabilizers to raise the top-five measured distance floor above four at rate 0.500.")
GROUP = ("Preserve the determinant-one matrix-group times cyclic-factor construction. Reduce the independent cyclic factor from order two to one: q=24 fits the new factor 16 and gives n=384. The parent's q=48 bed measured twelve on all five leaders; retaining its matrix factor is motivated by that reading, not a prediction of high-rate distance.")
SHAPES = ("Target cell 0.500: A=(1,3), B=(1,5), n=384 and kb=192, guaranteed rate 0.500. Components have 24 checks and 72/120 variables. Parent minima broadly occupied sector-one Acol/Bcol blocks. Test complete CSS logical classes in both sectors and require exact lifted k/n in [0.475,0.525).")
FILLING = ("All eight cells are distinct-term binomials: HX weight eight, HZ weight twelve. Twenty relative-order/anchor constraints search fresh fillings, with relative orders four or six and no repeated inverse-generator pair within either side. Exact syndrome collisions modulo the opposite stabilizer span exclude logicals through weight four when possible; otherwise minimize their exact weight-four count. Rank and permutation-invariant logical-class collision fingerprints distinguish selected designs. Evaluator distance readings remain upper bounds.")
FAMILY_NAME = "r050_SL23xC1_13x15_binomial_logical4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Same matrix-group times cyclic-factor rule; cyclic order is now one.
    # Encoding: (a,b,c,d,z), with a*d-b*c = 1 mod 3.
    if not valid_q(q):
        raise ValueError("SL(2,3) x C1 requires order 24")
    els = [(a,b,c,d,z)
           for z in range(1)
           for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3 == 1]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(g[i] % (1 if i == 4 else 3) for i in range(5))

    def mul(g, h):
        a,b,c,d,z = red(g)
        e,f,u,v,w = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3, (z+w) % 1)

    def inv(g):
        a,b,c,d,z = red(g)
        return (d, (-b) % 3, (-c) % 3, a, z)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)

from itertools import combinations_with_replacement
from functools import lru_cache


@lru_cache(maxsize=1)
def _support_designs():
    """Twenty order-occupancy/anchor constraints, not twenty orbit images.

    Entries specify the number of order-four relatives in A and B,
    followed by the anchor model. Every remaining relative has order six.
    A always includes an order-six relative.
    """
    return tuple(
        (a4, b4, model)
        for model in range(2)
        for a4 in range(3)
        for b4 in range(1, 4)
    ) + ((1, 2, 2), (2, 1, 2))


@lru_cache(maxsize=1)
def _edge_pools(G):
    """Represent distinct inverse pairs of order-four/order-six elements."""
    pools = {4: [], 6: []}
    for g in G.elements:
        power, order = G.e, 0
        for t in range(1, 13):
            power = G.mul(power, g)
            if power == G.e:
                order = t
                break
        if order in pools and G.idx(g) < G.idx(G.inv(g)):
            pools[order].append(g)
    if len(pools[4]) < 3 or len(pools[6]) < 4:
        raise ValueError("unexpected SL(2,3) inverse-pair inventory")
    return pools


@lru_cache(maxsize=1)
def _label_pools(G):
    """Exclude equal terms and relative involutions, including central ones."""
    compatible = {}
    for h in G.elements:
        hi = G.inv(h)
        compatible[h] = frozenset(
            g for g in G.elements
            if g != h and G.mul(G.mul(hi, g), G.mul(hi, g)) != G.e)
    return compatible


def _label_grid(rng, G, width, four_count, model):
    """Fresh binomials with prescribed relative orders and anchor support."""
    pools = _edge_pools(G)
    steps = (rng.sample(pools[4], four_count)
             + rng.sample(pools[6], width-four_count))
    rng.shuffle(steps)
    if model == 0:
        anchors = [g for g in G.elements if (g[0]+g[3]) % 3 == 0]
    elif model == 2:
        anchors = pools[6] + [G.inv(g) for g in pools[6]]
    else:
        anchors = G.elements
    row = []
    for step in steps:
        if rng.randrange(2):
            step = G.inv(step)
        anchor = rng.choice(anchors)
        other = G.mul(anchor, step)
        assert anchor != other
        row.append([anchor, other])
    return [row]


def _columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _component_columns(hx, hz, q):
    """Extract both component actions only from the verified LP matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = []
    for i in range(ra):
        for h in range(q):
            row = hx[(i*cb)*q+h]
            packed = 0
            for j in range(ca):
                packed |= ((row >> (j*cb*q)) & mask) << (j*q)
            ar.append(packed)
    bmask = (1 << (cb*q))-1
    br = [row & bmask for row in hz[:rb*q]]
    return _columns(ar, ca*q), _columns(br, cb*q)


def _cycles(cols):
    counts, total = {}, 0
    for syndrome in cols:
        active = []
        while syndrome:
            bit = syndrome & -syndrome
            active.append(bit.bit_length()-1)
            syndrome ^= bit
        for i, a in enumerate(active):
            for b in active[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                total += old
                counts[key] = old+1
    return total


def _kernel_probe(rng, cols, trials, best):
    """Actual classical kernel witnesses; no distance certification."""
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                best = min(best, word.bit_count())
        short = sorted(dependencies, key=int.bit_count)[:40]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 4:
            break
    return best


def _small_logical(checks, stabilizers, n):
    """Exact existence test for quantum logical supports of weight <=4."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(word):
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    cols = _columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif outside(word ^ old):
            return True
    for i, syndrome in enumerate(cols):
        bit = 1 << i
        for j in range(i):
            s = syndrome ^ cols[j]
            word = bit | (1 << j)
            old = seen.get(s)
            if old is None:
                seen[s] = word
            elif outside(word ^ old):
                return True
    return False


def _logical4_stats(checks, stabilizers, n):
    """Exact logical existence through four and an invariant collision census.

    Once logical weights <=3 are excluded, differing logical classes of
    equal-syndrome pairs have disjoint supports. Each weight-four logical
    then contributes exactly three unordered pair-partition collisions.
    Small stabilizers are permitted throughout.
    """
    basis = {}
    for original in stabilizers:
        word = original
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    pivots = sorted(basis, reverse=True)

    def reduce(word):
        for p in pivots:
            if (word >> p) & 1:
                word ^= basis[p]
        return word

    cols = _columns(checks, n)
    classes = [reduce(1 << i) for i in range(n)]
    singles = {}
    small = False
    for syndrome, cls in zip(cols, classes):
        if syndrome == 0 and cls:
            small = True
        previous = singles.get(syndrome)
        if previous is None:
            singles[syndrome] = cls
        elif previous != cls:
            small = True

    pairs = {}
    for i, syndrome in enumerate(cols):
        ci = classes[i]
        for j in range(i):
            s = syndrome ^ cols[j]
            cls = ci ^ classes[j]
            if s == 0 and cls:
                small = True
            singleton = singles.get(s)
            if singleton is not None and singleton != cls:
                small = True
            bucket = pairs.get(s)
            if bucket is None:
                pairs[s] = {cls: 1}
            else:
                bucket[cls] = bucket.get(cls, 0)+1

    cross, histogram = 0, {}
    for syndrome, bucket in pairs.items():
        occupancies = tuple(sorted(bucket.values()))
        total = sum(occupancies)
        cross += (total*(total-1)
                  - sum(c*(c-1) for c in occupancies)) // 2
        key = (syndrome.bit_count(), occupancies)
        histogram[key] = histogram.get(key, 0)+1
    return small, cross, tuple(sorted(histogram.items()))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Optimize twenty constraints and retain only invariant-distinct codes.

    The deterministic cache does not alter the returned cells. Extra harness
    slots continue the indexed searches with fresh seeds and fingerprint
    exclusion, rather than returning permutations of earlier designs.
    """
    plans = _support_designs()
    sa = ((2, 2, 2),)
    sb = ((2, 2, 2, 2, 2),)
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    bank, used = [], set()

    for task in range(N_SLOTS):
        a4, b4, model = plans[task % len(plans)]
        local = random.Random(BASE_SEED*31013 + 50024 + task)
        best, best_score = None, None
        accepted = 0

        for attempt in range(96):
            A = _label_grid(local, G, 3, a4, model)
            B = _label_grid(local, G, 5, b4, model)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if k <= 0 or not (475*n <= 1000*k < 525*n):
                continue

            sx = _logical4_stats(hz, hx, n)
            sz = _logical4_stats(hx, hz, n)
            # Each histogram records syndrome weight and the multiset of
            # pair multiplicities in stabilizer-equivalence classes.
            # These survive check/qubit permutations and global X/Z swap.
            fingerprint = (
                k, tuple(sorted((sx[2], sz[2]))),
                tuple(sorted((
                    tuple(sorted(r.bit_count() for r in hx)),
                    tuple(sorted(r.bit_count() for r in hz))))))
            if fingerprint in used:
                continue

            small = sx[0] or sz[0]
            collisions = sx[1]+sz[1]
            ac, bc = _component_columns(hx, hz, G.q)
            cycles = _cycles(ac)+_cycles(bc)
            score = (not small, collisions == 0,
                     -collisions, k, -cycles)
            if best_score is None or score > best_score:
                best_score = score
                best = (A, B, hx, hz, fingerprint)
            accepted += 1
            if accepted >= 16:
                break

        if best is None:
            # Do not replace an exhausted search with an orbit copy.
            continue
        A, B, hx, hz, fingerprint = best
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= 12
        used.add(fingerprint)
        bank.append((A, B))

    if len(bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct half-rate designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return a separately optimized, fingerprint-distinguished design."""
    bank = _design_bank(G)
    if idx >= len(bank):
        raise ValueError("no further invariant-distinct half-rate design")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free rotational binomials plus a trinomial on D34 (1,4)x(1,4) did not raise the distance reading above four: all 60 distinct slots had d2s=4 and all five measured slots had d30s=4.
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
