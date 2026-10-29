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
CONCEPT = ("Cyclic Z9 encoded as metacyclic Z1:Z9, order 9, shapes (2,7)x(2,8): redistributing extra terms across both components and screening weight-four words modulo stabilizers should outperform the parent's A-only triple upgrades.")
GROUP = ("Frozen L1 group, order and encoding. This exact Z9 bed measured five d30s=4 readings at k=283; test a filling change without assuming a group-driven distance advantage.")
SHAPES = ("Frozen (2,7)x(2,8): n=540, kb=270, guaranteed rate 0.500, and component lengths/check counts 63/18 and 72/18. Parent minima were entirely sector-1 Acol/Bcol operators spread across both components. Verified ranks retain 0.475<=k/n<0.525; fresh k=283 fallbacks preserve the parent's rate.")
FILLING = ("Twenty cards test balanced degree-two/three columns, pure-row cells, and selected degree-four columns on both sides. Extract components from verified lifts, reject short component kernels, then exactly count weight-four logicals modulo the actual stabilizer spaces. Prefer exclusion through weight four, otherwise k=283 B-triple fallbacks and fewer logical quartets. Permutation-invariant fingerprints conservatively exclude emitted orbit copies. Screening certifies only the explicitly excluded weights, not a searched distance.")
FAMILY_NAME = "r050_m1r9s1_27x28_twosided_logical4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 9

def custom_ops(q):
    # Same metacyclic rule; the new shape requires a smaller lift.
    # Z_1 x| Z_9 with s=1 is cyclic of order 9.
    m, r, s = 1, 9, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)

# Each template changes internal triple geometry, not just its translate.
_TRIPLE_GEOMETRIES = (
    (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 1, 5), (0, 1, 6), (0, 1, 7),
    (0, 2, 4), (0, 2, 5), (0, 3, 6),
)
_UNITS_9 = (1, 2, 4, 5, 7, 8)


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _small_logical_3(checks, stabilizers, width):
    """Exactly detect any non-stabilizer logical of weight one to three."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]

    def in_span(v):
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                return False
            v ^= basis[p]
        return True

    syndromes = _component_columns(checks, width)
    singles = {}
    for i, syndrome in enumerate(syndromes):
        v = 1 << i
        if syndrome == 0 and not in_span(v):
            return True
        if syndrome in singles:
            if not in_span(v ^ singles[syndrome]):
                return True
        else:
            singles[syndrome] = v

    # After the singleton test, equal-syndrome singleton representatives
    # differ only by stabilizers, so one representative per syndrome suffices.
    for i in range(width):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            v = vi | (1 << j)
            if syndrome == 0 and not in_span(v):
                return True
            if syndrome in singles:
                if not in_span(v ^ singles[syndrome]):
                    return True
    return False


from functools import lru_cache


@lru_cache(maxsize=1)
def _twosided_bank(G):
    """Optimize structural cards once, keeping invariant-distinct outputs."""
    q = G.q
    assert q == 9
    mask = (1 << q) - 1

    # The first sixteen cards have row totals 9/9 on both sides.
    # Column totals are 2 or 3, so both CSS row caps are twelve.
    aprofiles = (
        ((2, 1), (2, 1), (1, 2), (1, 2)),
        ((3, 0), (2, 1), (1, 2), (1, 2)),
        ((3, 0), (0, 3), (2, 1), (1, 2)),
        ((3, 0), (3, 0), (0, 3), (0, 3)),
    )
    bprofiles = (
        ((2, 1), (1, 2)) + ((1, 1),) * 6,
        ((3, 0), (0, 3)) + ((1, 1),) * 6,
        ((2, 1), (1, 2), (2, 0), (0, 2)) + ((1, 1),) * 4,
        ((3, 0), (0, 3), (2, 0), (0, 2)) + ((1, 1),) * 4,
    )
    geometries = ((0, 1, 2), (0, 1, 3),
                  (0, 1, 4), (0, 2, 5))

    def recipe(card):
        if card < 16:
            ac = aprofiles[card // 4] + ((1, 1),) * 3
            bc = bprofiles[card % 4]
        else:
            # A totals 8/8 and has maximum column degree three.
            # B totals 9/9 with one degree-four column.
            pure, internal = divmod(card - 16, 2)
            ac = (((3, 0), (0, 3)) if pure else
                  ((2, 1), (1, 2))) + ((1, 1),) * 5
            bc = (((2, 2), (2, 0), (0, 2)) + ((1, 1),) * 5
                  if internal else
                  ((2, 2),) + ((1, 1),) * 7)
        return ac, bc

    def draw_cell(rng, count, model):
        if count == 0:
            return []
        base = rng.randrange(q)
        if count == 1:
            offsets = (0,)
        elif count == 2:
            # Order-nine relative labels avoid a within-cell
            # order-three cycle; complete matrices still decide validity.
            offsets = (0, rng.choice(_UNITS_9))
        else:
            geometry = geometries[(model + rng.randrange(4)) % 4]
            unit = rng.choice(_UNITS_9)
            offsets = tuple(unit * t % q for t in geometry)
        return [(0, (base + t) % q) for t in offsets]

    def draw_grid(rng, counts, side, model):
        width = len(counts)
        grid = [[None] * width for _ in range(2)]
        mono = [j for j, pair in enumerate(counts) if pair == (1, 1)]
        relatives = rng.sample(range(q), len(mono))
        for j, d in zip(mono, relatives):
            grid[0][j] = [G.e]
            grid[1][j] = [(0, d if side == "A" else (-d) % q)]
        for j, pair in enumerate(counts):
            if pair == (1, 1):
                continue
            for i in range(2):
                grid[i][j] = draw_cell(rng, pair[i], model + i + j)
        return grid

    def reserve(rng, model):
        # Unlike the parent, B has two odd triple cells in different
        # columns. A remains monomial. Row totals are A=7/7, B=10/10;
        # maximum column totals are A=2, B=4.
        A = draw_grid(rng, ((1, 1),) * 7, "A", model)
        B = draw_grid(rng, ((1, 1),) * 8, "B", model)
        B[0][0] = draw_cell(rng, 3, model)
        B[1][1] = draw_cell(rng, 3, model + 1)
        return A, B

    def components(hx, hz):
        ar = []
        for i in range(2):
            for r in range(q):
                row = hx[(i * 8) * q + r]
                ar.append(sum(((row >> (j * 8 * q)) & mask) << (j * q)
                              for j in range(7)))
        br = [row & ((1 << (8 * q)) - 1) for row in hz[:2 * q]]
        return (_component_columns(ar, 7 * q),
                _component_columns(br, 8 * q))

    def component_stats(cols):
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        pairs = {}
        collisions = 0
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                if s in singles:
                    return None
                old = pairs.get(s, 0)
                collisions += old
                pairs[s] = old + 1
        return collisions // 3, _component_four_cycles(cols)

    def logical4_stats(checks, stabilizers, n):
        """Exact logical quartet census after excluding all kernel words <=3.

        Equal pair syndromes have disjoint supports. Reducing each qubit
        modulo the actual stabilizer row space distinguishes stabilizer
        quartets from logical quartets. Each quartet has three partitions.
        """
        cols = _component_columns(checks, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return None
        pairs = {}
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                if s in singles:
                    return None
                packed = i * n + j
                old = pairs.get(s)
                if old is None:
                    pairs[s] = packed
                elif isinstance(old, int):
                    pairs[s] = [old, packed]
                else:
                    old.append(packed)

        basis = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = row
                    break
                row ^= old

        def remainder(v):
            # Unlike an outside-span test, retain nonpivot coordinates
            # while continuing to reduce all pivot coordinates.
            for p in sorted(basis, reverse=True):
                if (v >> p) & 1:
                    v ^= basis[p]
            return v

        quotient = [remainder(1 << i) for i in range(n)]
        participation = [0] * n
        histogram = {}
        logical_collisions = 0
        for syndrome, value in pairs.items():
            if isinstance(value, int):
                key = (1, syndrome.bit_count(), (1,))
                histogram[key] = histogram.get(key, 0) + 1
                continue
            frequency = {}
            labeled = []
            for packed in value:
                i, j = divmod(packed, n)
                label = quotient[i] ^ quotient[j]
                frequency[label] = frequency.get(label, 0) + 1
                labeled.append((i, j, label))
            count = len(value)
            classes = tuple(sorted(frequency.values()))
            key = (count, syndrome.bit_count(), classes)
            histogram[key] = histogram.get(key, 0) + 1
            logical_collisions += (
                count * (count - 1) -
                sum(c * (c - 1) for c in classes)) // 2
            for i, j, label in labeled:
                outside = count - frequency[label]
                participation[i] += outside
                participation[j] += outside

        assert logical_collisions % 3 == 0
        assert all(p % 3 == 0 for p in participation)
        return (logical_collisions // 3, tuple(sorted(histogram.items())),
                tuple(c.bit_count() for c in cols),
                tuple(p // 3 for p in participation))

    def fingerprint(hx, hz, sx, sz, k):
        types = tuple(sorted((
            (sx[0], sx[1], tuple(sorted(r.bit_count() for r in hx))),
            (sz[0], sz[1], tuple(sorted(r.bit_count() for r in hz))),
        )))
        joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
        swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
        return k, types, min(joint, swapped)

    fingerprints = set()
    bank = []
    for slot in range(N_SLOTS):
        card = slot % 20
        acounts, bcounts = recipe(card)
        sa = [[c[i] for c in acounts] for i in range(2)]
        sb = [[c[i] for c in bcounts] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED * 130003 + 940041 + slot)
        winner = None
        seen = set()

        for batch in range(3):
            pools = [[], []]
            for kind, quota in ((0, 48), (1, 24)):
                for attempt in range(quota):
                    if kind == 0:
                        A = draw_grid(rng, acounts, "A", card)
                        B = draw_grid(rng, bcounts, "B", card + 1)
                    else:
                        A, B = reserve(rng, card + attempt)
                    key = tuple(tuple(sorted(cell)) for grid in (A, B)
                                for row in grid for cell in row)
                    if key in seen:
                        continue
                    seen.add(key)
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (475 * n <= 1000 * k < 525 * n):
                        continue
                    if kind == 1 and k != 283:
                        continue
                    a, b = components(hx, hz)
                    pa, pb = component_stats(a), component_stats(b)
                    if pa is None or pb is None:
                        continue
                    short = pa[0] + pb[0]
                    priority = (short == 0, -short, k,
                                -pa[1] - pb[1])
                    pools[kind].append(
                        (priority, A, B, hx, hz, n, k))
                    pools[kind].sort(key=lambda item: item[0], reverse=True)
                    del pools[kind][4:]

            for pool in pools:
                for _, A, B, hx, hz, n, k in pool:
                    sx = logical4_stats(hx, hz, n)
                    if sx is None:
                        continue
                    sz = logical4_stats(hz, hx, n)
                    if sz is None:
                        continue
                    fp = fingerprint(hx, hz, sx, sz, k)
                    if fp in fingerprints:
                        continue
                    quartets = sx[0] + sz[0]
                    # Five is only an exclusion-through-four flag.
                    # It is not a budget-limited distance reading.
                    lower = 5 if quartets == 0 else 4
                    objective = (k * lower * lower, lower,
                                 -max(sx[0], sz[0]), -quartets)
                    if winner is None or objective > winner[0]:
                        winner = (objective, fp, A, B, hx, hz, k)
            if winner is not None:
                break

        if winner is None:
            raise ValueError("no invariant-distinct screened card filling")
        objective, fp, A, B, hx, hz, k = winner
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        fingerprints.add(fp)
        bank.append((A, B, k, objective))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the jointly optimized, invariant-distinct structural card."""
    A, B, k, screen = _twosided_bank(G)[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "logical4_screen": screen}


def CELL(rng, G, side, i, j, ctx):
    """Return the full two-sided selection without any relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: A-only odd triple upgrades with unchanged monomial B on Z9 (2,7)x(2,8) remove the weight-four plateau: all five measured codes had d30s=4 and thousands of sector-1 weight-four witnesses.
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
