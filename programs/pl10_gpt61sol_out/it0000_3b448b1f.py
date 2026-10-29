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
CONCEPT = ("Dic11, order 44, shapes (2,3)x(2,3): asymmetric binomials with order-22 rotation differences in A and order-four mixed-coset differences in B should retain near-zero circuit failures while increasing measured k to 60-70 in the 0.100 cell.")
GROUP = ("Keep the frozen Dic11 bed: current fitness is 7.1162 and four measured codes had zero failures in both bases. Its C4 abelianization permits different binomial vanishing orders, motivating a filling-only rank-deficit experiment; a circuit advantage remains unverified.")
SHAPES = ("Frozen equal (2,3) components have 132 variables and 88 checks each; n=572 and kb=44 give guaranteed rate 1/13. Exact ranks require 60<=k<=70, still below rate 0.125. Parent slot 34 had two Z failures, with Acol1/Bcol2 enriched 2.0x, but four measured siblings had none; this is insufficient evidence for a unique block repair.")
FILLING = ("All cells are doubletons: component rows have weight six and columns weight four, so CSS check weight is ten and six-round CNOT count is 31680. Slots use canonical order-22 ratio architectures, not saved-code relabelings. A uses same-coset odd rotation displacements excluding the central involution; B uses mixed cosets. Verified lifts screen exact k, component words through weight four, cycles and small-boundary triples, then actual short quantum logicals and CSS. Sampled component-distance rewards saturate at 22. This targets the parent's ceiling limitation and rare collective Z failures, not distance as fitness.")
FAMILY_NAME = "r010_dic11_23x23_asym_binomial_boundary_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic11: x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    # Encoding (a,b) means x^a*y^b, with b in {0,1}.
    assert q == 44
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        return (g[0] % 22, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11*b*d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return 22*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _dv_columns(rows, n):
    """Column syndromes of an already verified lifted matrix."""
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _dv_pivots(rows):
    piv = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in piv:
                row ^= piv[p]
            else:
                piv[p] = row
                break
    return piv


def _dv_remainder(word, piv):
    while word:
        p = word.bit_length() - 1
        if p not in piv:
            return word
        word ^= piv[p]
    return 0


def _dv_low4(rows, stabilizers, n):
    """Exact existence test for a nontrivial kernel word of weight <=4."""
    cols = _dv_columns(rows, n)
    piv = _dv_pivots(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if _dv_remainder(word ^ seen[syndrome], piv):
                return True
        else:
            seen[syndrome] = word
    for i in range(n):
        si = cols[i]
        wi = 1 << i
        for j in range(i):
            syndrome = si ^ cols[j]
            word = wi | (1 << j)
            if syndrome in seen:
                if _dv_remainder(word ^ seen[syndrome], piv):
                    return True
            else:
                seen[syndrome] = word
    return False


def _dv_components(hx, hz, q):
    """Extract A-left and B-right components from the scaffold's lift."""
    mask = (1 << q) - 1
    arows = []
    for i in range(2):
        for r in range(q):
            row = hx[(i * 3) * q + r]
            packed = 0
            for j in range(3):
                packed |= ((row >> (j * 3 * q)) & mask) << (j * q)
            arows.append(packed)
    # j=0 in HZ gives the B-right component on contiguous sector-1 blocks.
    bmask = (1 << (3 * q)) - 1
    brows = [hz[a * q + r] & bmask
             for a in range(2) for r in range(q)]
    return arows, brows


def _dv_cycles(rows, n):
    """Four-cycle count and a row-intersection triangle heuristic."""
    nr = len(rows)
    adjacency = [0] * nr
    c4 = 0
    for i in range(nr):
        for j in range(i):
            common = (rows[i] & rows[j]).bit_count()
            c4 += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    triangles = 0
    for i in range(nr):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = 0
    for col in _dv_columns(rows, n):
        degree = col.bit_count()
        stars += degree * (degree - 1) * (degree - 2) // 6
    # Exact six-cycle count when intersections are simple; heuristic otherwise.
    return c4, max(0, triangles - stars)


def _dv_kernel_estimate(rng, rows, n, trials):
    """Upper bound from randomized fundamental kernel circuits."""
    cols = _dv_columns(rows, n)
    order = list(range(n))
    best = n + 1
    for _ in range(trials):
        rng.shuffle(order)
        piv = {}
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in piv:
                    s, w = piv[p]
                    syndrome ^= s
                    word ^= w
                else:
                    piv[p] = (syndrome, word)
                    break
            if not syndrome and word:
                best = min(best, word.bit_count())
    return best


def NEW_CTX(rng, G, idx):
    """Choose a ratio architecture, then optimize its actual voltages."""
    import itertools

    q = G.q
    component_n = 3 * q
    supports = [[2, 2, 2], [2, 2, 2]]
    assert _grid_weights(supports, supports) <= W_MAX
    assert _no_dead_lines(supports, supports)

    # Undirected generators of <x>: every displacement has order 22.
    # Canonicalize ratio architectures under component row/column
    # permutations and x -> x^u automorphisms. This is not a claim of
    # complete Tanner-isomorphism classification.
    units = (1, 3, 5, 7, 9)
    permutations = list(itertools.permutations(range(3)))

    def canonical(f):
        images = []
        for swap in (0, 1):
            for p in permutations:
                sequence = [f[3 * (i ^ swap) + j]
                            for i in range(2) for j in p]
                for u in units:
                    image = []
                    for d in sequence:
                        value = (u * d) % 22
                        image.append(min(value, 22 - value))
                    images.append(tuple(image))
        return min(images)

    # Enumerate equality architectures rather than drawing twenty
    # unconstrained copies of the same labeling model.
    patterns = []

    def extend(prefix, largest):
        if len(prefix) == 6:
            patterns.append(tuple(prefix))
            return
        for value in range(min(4, largest + 1) + 1):
            extend(prefix + [value], max(largest, value))

    extend([0], 0)
    embeddings = (
        (1, 3, 5, 7, 9),
        (1, 5, 9, 3, 7),
        (1, 7, 3, 9, 5),
    )
    bank = sorted({
        canonical(tuple(embedding[t] for t in pattern))
        for pattern in patterns for embedding in embeddings
    })
    assert len(bank) >= 20
    # Spread the supplied slot budget across the canonical bank.
    architecture = bank[(idx * len(bank) // max(1, N_SLOTS)) % len(bank)]

    def draw():
        A = []
        for i in range(2):
            row = []
            for j in range(3):
                # Nonconstant quotient holonomy keeps this from being
                # an all-rotation, disconnected component construction.
                b = int(i == 1 and j != 0)
                a = rng.randrange(22)
                d = architecture[3 * i + j]
                row.append([(a, b), ((a + d) % 22, b)])
            A.append(row)
        B = [[[(rng.randrange(22), 0), (rng.randrange(22), 1)]
              for j in range(3)] for i in range(2)]
        return A, B

    def boundary_exposure(rows):
        """Exact count-weighted small boundaries of component triples.

        This is a trapping-set heuristic, not a pL measurement or a
        logical-distance certificate. Both components are projections
        of the verified CSS matrices.
        """
        cols = _dv_columns(rows, component_n)
        exposure = 0
        for i in range(component_n - 2):
            for j in range(i + 1, component_n - 1):
                pair = cols[i] ^ cols[j]
                for t in range(j + 1, component_n):
                    weight = (pair ^ cols[t]).bit_count()
                    if weight <= 4:
                        exposure += (6 - weight) ** 2
        return exposure

    # All rank decisions use the complete evaluator-convention lift.
    # The algebra motivates the pool; it never substitutes for its k.
    pool = []
    for attempt in range(96):
        A, B = draw()
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 60 <= k <= 70:
            continue
        ar, br = _dv_components(hx, hz, q)
        if _dv_low4(ar, [], component_n):
            continue
        if _dv_low4(br, [], component_n):
            continue
        a4, a6 = _dv_cycles(ar, component_n)
        b4, b6 = _dv_cycles(br, component_n)
        cost = (max(a4, b4), a4 + b4,
                max(a6, b6), a6 + b6)
        item = (cost, attempt, A, B, hx, hz, ar, br, n, k)
        pool.append(item)
        pool.sort(key=lambda entry: (entry[0], -entry[9], entry[1]))
        del pool[6:]

    if not pool:
        raise ValueError("no asymmetric binomial passed rank/short-word screens")

    finalists = []
    for item in pool:
        cost, attempt, A, B, hx, hz, ar, br, n, k = item
        da = _dv_kernel_estimate(rng, ar, component_n, 40)
        db = _dv_kernel_estimate(rng, br, component_n, 40)
        ba = boundary_exposure(ar)
        bb = boundary_exposure(br)
        # Avoid chasing large, optimistic sampled-distance estimates.
        # Once the basic sampled protection threshold is met, reduce
        # the worse component's small-boundary exposure.
        key = (int(min(da, db) >= 22), min(da, db, 22),
               -max(ba, bb), -(ba + bb),
               -cost[0], -cost[1], -cost[2], -cost[3], k)
        finalists.append((key, attempt, item))
    finalists.sort(key=lambda entry: (entry[0], -entry[1]), reverse=True)

    for key, attempt, item in finalists:
        _, _, A, B, hx, hz, _, _, n, k = item
        if _dv_low4(hx, hz, n) or _dv_low4(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 10
        return {"idx": idx, "architecture": architecture, "k": k,
                "screen": key, "cells": {"A": A, "B": B}}
    raise ValueError("all finalists contained a quantum logical of weight <=4")


def CELL(rng, G, side, i, j, ctx):
    """Return the rank- and neighborhood-screened full doubleton."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal uniform D7 labels on (6,2)x(6,2) do not suffice for reliable memory: measured s=3.30-3.59, abundant weight-four sector-two logicals, and roughly 10-45 percent failed-shot rates.
# REFUTED: d2s=26 guarantees ceiling-level circuit performance on this Dic11 bed: parent slot 34 had two Z failures, 38 collective logical flips and s=6.864, while slot 26 with the same d2s had no failures.
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
