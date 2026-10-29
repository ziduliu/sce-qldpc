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
CONCEPT = ("On Q8 x Z5, order 40, shapes (1,3)x(1,4), index-two cyclic binomial components paired with quaternion-unit cells should raise fitness at certified distance four; at most three k=247 designs supplement k=246 designs without leaving rate cell 0.450.")
GROUP = ("Frozen L1 bed: the parent scored 7.4218 and all five measured distances were four. Q8 x Z5 contains C20=<x*z> of index two. Over F2, odd-augmentation polynomials supported on the 2-group Q8 are units, permitting controlled rank deficits rather than another failed attempt to eliminate all weight-four logicals.")
SHAPES = ("Frozen (1,3)x(1,4): n=520, kb=240 and guaranteed rate 6/13. Component lengths are 120 and 160 with 40 checks each; the size window permits q<=46 but valid_q fixes q=40. Current minimum witnesses were entirely in sector-1 Acol/Bcol blocks. Three k=247 and two k=246 leaders have mean rate 246.6/520=0.474231, inside cell 0.450.")
FILLING = ("Enumerate structural binomial/unit cards and retain Tanner-invariant-distinct designs. Normal cards use connected C20 A binomials and a pure-Q8 unit trinomial in B; verified k must be 246. At most three connected even/even cards have verified k=247. Exclude all commuting weights 1..3, then verify a nontrivial weight-four logical with the existing exact stabilizer-quotient screen. No low-distance fallback. Both row caps are legal; expected top-five score at reading four is 7.5877, not a claim of increased distance.")
FAMILY_NAME = "r045_Q8xZ5_13x14_rank38_exact4_portfolio"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    # (a,b,c) = x^a y^b z^c:
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1; z central of order 5.
    if q != 40:
        raise ValueError("Q8 x Z5 requires order 40")
    els = [(a, b, c) for c in range(5)
           for b in range(2) for a in range(4)]

    def red(g):
        return (g[0] % 4, g[1] % 2, g[2] % 5)

    def mul(g, h):
        a, b, c = g
        d, e, f = h
        return ((a + (-1 if b else 1)*d + 2*b*e) % 4,
                (b + e) % 2, (c + f) % 5)

    def inv(g):
        a, b, c = g
        return (((a + 2) if b else -a) % 4, b, (-c) % 5)

    def idx(g):
        a, b, c = red(g)
        return (2*c + b)*4 + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _q85_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _q85_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return True
        v ^= basis[p]
    return False


def _q85_short_screen(hx, hz, n):
    """Exact existence test for nontrivial CSS logicals of weight <=4.

    Every such support splits into two supports of size <=2.
    Equal-syndrome collisions are tested modulo the actual stabilizer
    row space. Keeping one representative per syndrome is sufficient:
    if all collisions with it are stabilizers, their pairwise differences
    are stabilizers too.
    """
    cycles = 0
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = _q85_basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Count actual Tanner 4-cycles, not protograph coincidences.
        overlaps = {}
        for col in cols:
            ids = []
            v = col
            while v:
                bit = v & -v
                ids.append(bit.bit_length() - 1)
                v ^= bit
            for a in range(len(ids)):
                for b in range(a):
                    key = (ids[b], ids[a])
                    old = overlaps.get(key, 0)
                    cycles += old
                    overlaps[key] = old + 1

        representatives = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            mask = bits[i]
            if syndrome in representatives:
                v = mask ^ representatives[syndrome]
                if _q85_outside(v, basis):
                    return v.bit_count(), cycles
            else:
                representatives[syndrome] = mask

        for i in range(n):
            si, bi = cols[i], bits[i]
            for j in range(i):
                syndrome = si ^ cols[j]
                mask = bi | bits[j]
                if syndrome in representatives:
                    v = mask ^ representatives[syndrome]
                    if _q85_outside(v, basis):
                        return v.bit_count(), cycles
                else:
                    representatives[syndrome] = mask

    # Sentinel: no logical of weight <=4, not a distance estimate.
    return 5, cycles


from functools import lru_cache


def _q85_kernel_profile(rows, n):
    """Exact exclusion of commuting weights <=3, plus graph invariants.

    With distinct nonzero columns, every pair-syndrome collision uses
    disjoint pairs. Each weight-four kernel support contributes exactly
    three unordered pair partitions.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None

    counts = {}
    collisions = 0
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                return None
            old = counts.get(syndrome, 0)
            collisions += old
            counts[syndrome] = old + 1
    assert collisions % 3 == 0

    pair_hist = {}
    for syndrome, multiplicity in counts.items():
        key = (multiplicity, syndrome.bit_count())
        pair_hist[key] = pair_hist.get(key, 0) + 1

    def weight_hist(words):
        hist = {}
        for word in words:
            weight = word.bit_count()
            hist[weight] = hist.get(weight, 0) + 1
        return tuple(sorted(hist.items()))

    cycles = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            cycles += overlap * (overlap - 1) // 2

    return (collisions // 3, cycles, weight_hist(rows),
            weight_hist(cols), tuple(sorted(pair_hist.items())))


@lru_cache(maxsize=1)
def _q85_rank_bank(G, slots):
    """Pure deterministic construction; caching only avoids repeated work.

    Different retained fingerprints certify different Tanner graphs.
    Equal fingerprints are conservatively discarded even though they
    need not imply isomorphism.
    """
    from itertools import combinations
    order = random.Random(45040246)
    e = G.e
    odd_steps = (1, 3, 5, 7, 9)
    step_pairs = tuple(combinations(odd_steps, 2))

    def xz(s):
        # (x*z)^s, a generator of the index-two C20 for primitive s.
        return (s % 4, 0, s % 5)

    def yz(s):
        # (y*z)^s for odd s; y^3=x^2*y.
        return (0 if s % 4 == 1 else 2, 1, s % 5)

    def bins(elements):
        return [[[e, g] for g in elements]]

    def examine(A, B, target_k):
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != target_k:
            return None
        sx = _q85_kernel_profile(hx, n)
        if sx is None:
            return None
        sz = _q85_kernel_profile(hz, n)
        if sz is None:
            return None

        # The previous step excludes every smaller commuting support.
        # This step supplies an actual nontrivial weight-four witness.
        low, _ = _q85_short_screen(hx, hz, n)
        if low != 4:
            return None

        fingerprint = (k, tuple(sorted((sx, sz))))
        objective = (sx[0] + sz[0], sx[1] + sz[1])
        return fingerprint, objective, A, B

    def retain(table, result):
        if result is None:
            return
        fingerprint, objective, A, B = result
        previous = table.get(fingerprint)
        if previous is None or objective < previous[0]:
            table[fingerprint] = (objective, A, B)

    # Connected Cayley components on the entire order-40 group have
    # one-dimensional binomial rank deficits on both sides. Their
    # verified quantum dimension is 247. Keep at most three such codes:
    # four would move a five-code mean with k=246 partners above 0.475.
    connected_plans = [
        (a, b, c, d)
        for a in step_pairs for b in odd_steps
        for c in step_pairs for d in step_pairs
    ]
    order.shuffle(connected_plans)
    connected = {}
    reserve_count = min(3, max(0, slots - 2))
    for attempt, (ap, ay, bx, by) in enumerate(connected_plans):
        if attempt >= 96 and len(connected) >= reserve_count:
            break
        if attempt >= 320:
            break
        A = bins((xz(ap[0]), xz(ap[1]), yz(ay)))
        B = bins((xz(bx[0]), xz(bx[1]), yz(by[0]), yz(by[1])))
        retain(connected, examine(A, B, 247))
    reserves = sorted(connected.values(), key=lambda item: item[0])
    reserves = reserves[:reserve_count]

    # Four unsigned C20 generator spectra, not translated cell copies.
    # Odd steps make these component graphs bipartite; the presence
    # of step 1 connects each of the two C20 cosets.
    a_profiles = ((1, 3, 7), (1, 3, 5),
                  (1, 5, 7), (1, 5, 9))
    quaternion_axes = (
        (1, 0, 0), (3, 0, 0), (0, 1, 0),
        (2, 1, 0), (1, 1, 0), (3, 1, 0),
    )
    unit_pairs = tuple(
        (p, t) for p, t in combinations(quaternion_axes, 2)
        if (p[0] % 2, p[1]) != (t[0] % 2, t[1])
    )
    normal_plans = [
        (profile, pair, charge, p, t)
        for profile in a_profiles for pair in step_pairs
        for charge in range(5) for p, t in unit_pairs
    ]
    order.shuffle(normal_plans)
    normal = {}
    need = slots - len(reserves)
    budget = max(160, 8 * slots)
    cap = max(640, 16 * slots)
    for attempt, (profile, pair, charge, p, t) in enumerate(normal_plans):
        if attempt >= budget and len(normal) >= need:
            break
        if attempt >= cap:
            break
        A = bins(tuple(xz(s) for s in profile))
        B = bins((xz(pair[0]), xz(pair[1]), (0, 1, charge)))
        # Odd augmentation in F2[Q8] makes this block invertible.
        # All three terms have central coordinate zero.
        B[0].append([e, p, t])
        retain(normal, examine(A, B, 246))

    selected = reserves + sorted(
        normal.values(), key=lambda item: item[0])[:need]
    if len(selected) < min(5, slots):
        raise ValueError("fewer than five invariant-distinct exact-four designs")

    bank = []
    for objective, A, B in selected:
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        bank.append((A, B))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign an invariant-distinct, rank-controlled structural design."""
    bank = _q85_rank_bank(G, N_SLOTS)
    if idx >= len(bank):
        raise ValueError("no further invariant-distinct design; no orbit fallback")
    A, B = bank[idx]
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without resampling or mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Distinct-axis order-20 binomials with the exact-four screen's legal fallback raise this Q8 x Z5 bed above distance four; all five measured current codes retained d30s=4 and sector-1 witnesses.
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
