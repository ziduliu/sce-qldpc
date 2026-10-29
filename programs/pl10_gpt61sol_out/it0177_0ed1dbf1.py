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
CONCEPT = ("Dic11, order 44, shapes (2,3)x(2,3): an order-22/order-11 row split in A and identical C4-quotient rows in B should yield fresh k=71 codes retaining zero-failure circuit performance at check weight ten.")
GROUP = ("Keep the exact frozen Dic11 group: this bed achieved 7.3384 with five zero-failure measurements in both bases. Its C4 abelianization motivates a filling-only ceiling experiment; it does not establish a group advantage.")
SHAPES = ("Frozen equal (2,3) components each have 132 variables and 88 checks. At q=44, n=572 and kb=44 give guaranteed rate 1/13; require verified k=71, rate 0.124126, still in cell 0.100. The size factor permits q<=46 but the frozen group fixes q=44. Parent measurements have no failed-shot attribution; minimum-weight witnesses mainly occupy sector-one Acol/Bcol blocks.")
FILLING = ("Every cell remains a doubleton: component row/column weights six/four, CSS weight ten and 31680 six-round CNOTs. Canonical paired-step geometries prescribe odd order-22 differences in A row zero and even order-11 differences in row one. B mixed-coset terms have equal exponent parity, shared between its two rows in each column, making the C4-quotient rows identical. The quotient motivates k=71, but only verified full ranks accept it. Screen component words<=4, triple boundaries, capped sampled kernel witnesses and actual quantum logicals<=4. Parent k=67 codes all saturated despite d2s=20..22 and occasional LSD resolutions; the hypothesis is retained reliability at a higher ceiling, not distance as fitness.")
FAMILY_NAME = "r010_dic11_23x23_odd_even_quotientrank1_v3"

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
    """Choose an odd/even geometry and optimize fresh k=71 voltages."""
    from itertools import combinations_with_replacement, product

    q = G.q
    component_n = 3 * q
    supports = [[2, 2, 2], [2, 2, 2]]
    assert _grid_weights(supports, supports) <= W_MAX
    assert _no_dead_lines(supports, supports)

    # The row colors are algebraically different: odd generator
    # differences have order 22; nonzero even differences have order 11.
    # Quotient column permutations, common automorphisms and independent
    # doubleton reversal, rather than spending slots on those images.
    units = (1, 3, 5, 7, 9)
    column_types = list(product((1, 3, 5, 7, 9),
                                (2, 4, 6, 8, 10)))

    def undirected(value):
        value %= 22
        return min(value, 22 - value)

    def canonical(columns):
        return min(
            tuple(sorted((undirected(u * odd), undirected(u * even))
                         for odd, even in columns))
            for u in units
        )

    geometries = set()
    for columns in combinations_with_replacement(column_types, 3):
        # Avoid constant-step rows; their equality architectures are
        # particularly restrictive, even before voltage optimization.
        if len({c[0] for c in columns}) < 2:
            continue
        if len({c[1] for c in columns}) < 2:
            continue
        geometries.add(canonical(columns))

    bank = sorted(geometries, key=lambda card: (
        len(set(card)),
        len({c[0] for c in card}) + len({c[1] for c in card}),
        card))
    assert len(bank) >= max(20, N_SLOTS)
    position = idx * len(bank) // max(20, N_SLOTS)
    architecture = bank[position]

    def draw():
        A = [[], []]
        for j, (odd, even) in enumerate(architecture):
            a = rng.randrange(22)
            A[0].append([(a, 0), ((a + odd) % 22, 0)])
            a = rng.randrange(22)
            b = int(j != 0)
            A[1].append([(a, b), ((a + even) % 22, b)])

        # In the C4 abelianization x -> z^2, y -> z.
        # With equal a/c parity t, a mixed doubleton projects to
        # u + t*u^3, where u=1+z. Share t between the two rows,
        # so their quotient cells are identical column by column.
        # Normal-subgroup voltages remain independently variable.
        parity = (0, 1, 0)
        B = [[[(2 * rng.randrange(11) + parity[j], 0),
               (2 * rng.randrange(11) + parity[j], 1)]
              for j in range(3)] for i in range(2)]
        return A, B

    def connected(rows):
        """Connectivity of the verified component Tanner graph."""
        cols = _dv_columns(rows, component_n)
        parent = list(range(len(rows)))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for syndrome in cols:
            if not syndrome:
                return False
            first = syndrome & -syndrome
            a = first.bit_length() - 1
            syndrome ^= first
            while syndrome:
                bit = syndrome & -syndrome
                b = bit.bit_length() - 1
                parent[root(b)] = root(a)
                syndrome ^= bit
        return len({root(i) for i in range(len(rows))}) == 1

    def boundary_exposure(rows):
        """Triple-boundary exposure, balanced across component blocks.

        Counts come from verified component syndromes. They are a BP
        trapping-set surrogate, not a circuit or distance certificate.
        """
        cols = _dv_columns(rows, component_n)
        total = 0
        exposure = [0, 0, 0]
        for i in range(component_n - 2):
            for j in range(i + 1, component_n - 1):
                pair = cols[i] ^ cols[j]
                for t in range(j + 1, component_n):
                    weight = (pair ^ cols[t]).bit_count()
                    if weight <= 4:
                        penalty = (6 - weight) ** 2
                        total += penalty
                        for block in {i // q, j // q, t // q}:
                            exposure[block] += penalty
        return max(exposure), total

    # Over F2[C4], A has quotient invariant exponents (2,4),
    # while B has (1,4). Together these motivate quotient k=31
    # and total k=40+31=71 when the remaining sectors have full
    # component rank. Only full verified CSS ranks decide acceptance.
    for batch in range(2):
        pool = []
        for attempt in range(128):
            A, B = draw()
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 572 or k != 71 or 8 * k >= n:
                continue
            ar, br = _dv_components(hx, hz, q)
            if not connected(ar) or not connected(br):
                continue
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
            pool.sort(key=lambda entry: (entry[0], entry[1]))
            del pool[10:]

        finalists = []
        for item in pool:
            cost, attempt, A, B, hx, hz, ar, br, n, k = item
            da = _dv_kernel_estimate(rng, ar, component_n, 48)
            db = _dv_kernel_estimate(rng, br, component_n, 48)
            ba, bb = boundary_exposure(ar), boundary_exposure(br)
            distance = min(da, db)
            # Parent d2s=20 codes already saturated. Use twenty as
            # a sampled warning threshold, then prioritize boundaries
            # rather than chasing arbitrarily optimistic upper bounds.
            key = (int(distance < 20),
                   max(ba[0], bb[0]), ba[0] + bb[0],
                   max(ba[1], bb[1]), ba[1] + bb[1],
                   cost, -min(distance, 22),
                   abs(da - db), attempt)
            finalists.append((key, item))
        finalists.sort(key=lambda entry: entry[0])

        for key, item in finalists:
            _, _, A, B, hx, hz, _, _, n, k = item
            if _dv_low4(hx, hz, n) or _dv_low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 10
            assert sum(row.bit_count() for row in hx + hz) * 6 == 31680
            return {"idx": idx, "architecture": architecture, "k": k,
                    "screen": key, "cells": {"A": A, "B": B}}

    raise ValueError("no connected k=71 quotient-rank-one filling passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened doubleton without late relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal uniform D7 labels on (6,2)x(6,2) do not suffice for reliable memory: measured s=3.30-3.59, abundant weight-four sector-two logicals, and roughly 10-45 percent failed-shot rates.
# REFUTED: d2s=26 guarantees ceiling-level circuit performance on this Dic11 bed: parent slot 34 had two Z failures, 38 collective logical flips and s=6.864, while slot 26 with the same d2s had no failures.
# REFUTED: Increasing d2s at fixed k necessarily raises measured fitness: Dic11 slots 15 and 17 had k=67 and d2s=22 versus 20, yet both had zero failures in both bases and the same saturated s=7.338.
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
