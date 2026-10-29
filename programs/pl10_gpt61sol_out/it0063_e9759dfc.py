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
CONCEPT = ("On metacyclic Z29:Z2(s=28), order 58, shapes (1,3)x(1,3), all-mixed trinomials with minimal component four-cycle overlap, screened weight-six kernels and improved check-graph expansion should reduce worse-basis circuit failures.")
GROUP = ("Frozen by L1. Filling changes improved this exact order-58 bed from 3.2187 to 4.3650, supporting another element-model refinement without attributing gains to a group change. Mixed trinomials connect both rotation cosets; their involutory differences still impose a circuit-level risk.")
SHAPES = ("Frozen equal (1,3) sides: n=580, guaranteed k=232 and rate 0.400, with two 174-variable/58-check components and q=58 fixed. Parent minimum logicals occupy sector-1 Acol/Bcol blocks; Acol/Brow check enrichment is diffuse at approximately 1.0-1.1, so protect both components rather than patching one block.")
FILLING = ("Canonical rotation-step geometry pairs define deliberate designs. Each cell is a mixed trinomial; distinct reflection differences remove avoidable component four-cycles while check weight remains 12 and six-round CNOT count remains 25056. Independently search the actual left/right components extracted from verified lifts, prioritize low nontrivial check-graph spectral radius, and count weight-six kernels exactly through group-orbit triple syndromes after excluding weight-four kernels. Final verified ranks require k=232 and CSS commutation. This targets the parent's three measured weight-six-logical slots and collective failures, including slot 0 X=105/3500 and slot 21 Z=106/2500; expansion and kernel statistics are circuit-pL hypotheses, not the score.")
FAMILY_NAME = "r040f_m29r2s28_13x13_mixed3_orbit_expansion_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 58

def custom_ops(q):
    # metacyclic Z_29 x| Z_2: y x y^-1 = x^28 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 29, 2, 28
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
B_SHAPE = (1, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search two deliberate geometries; all mutable state remains local."""
    from itertools import combinations
    from math import cos, sin, pi, hypot

    q, m = G.q, 29
    nc = 3 * q
    mask = (1 << m) - 1
    cmask = (1 << nc) - 1
    unit = [[[G.e]]]
    els = G.elements
    mul = [[G.idx(G.mul(g, h)) for h in els] for g in els]
    inv = [G.idx(G.inv(g)) for g in els]

    # Rotation-step triples modulo all multipliers and column exchanges.
    # The quotient has 26 geometries; slots choose different geometry
    # pairs, not automorphism images of one saved filling.
    representatives = set()
    for ds in combinations(range(1, 15), 3):
        images = []
        for u in range(1, m):
            images.append(tuple(sorted(
                min((u * d) % m, m - (u * d) % m) for d in ds)))
        representatives.add(min(images))
    book = sorted(representatives)
    a = idx % len(book)
    b = (a + 7 + idx // len(book)) % len(book)

    # One representative of every triple of component variables under
    # common right multiplication. A triple has trivial stabilizer:
    # its size is three and three does not divide |D29|=58.
    triple_orbits = []
    for block in range(3):
        anchor = block * q
        for u in range(anchor + 1, nc):
            for v in range(u + 1, nc):
                triple = (anchor, u, v)
                canonical = True
                for x in triple:
                    if x // q != block:
                        continue
                    h = inv[x % q]
                    image = tuple(sorted(
                        (y // q) * q + mul[y % q][h] for y in triple))
                    if image < triple:
                        canonical = False
                        break
                if canonical:
                    triple_orbits.append(triple)
    assert len(triple_orbits) * q == nc * (nc - 1) * (nc - 2) // 6

    trig = [
        ([cos(2 * pi * k * t / m) for t in range(m)],
         [sin(2 * pi * k * t / m) for t in range(m)])
        for k in range(1, 15)
    ]

    def rotate(word, shift):
        shift %= m
        return ((word << shift) | (word >> (m - shift))) & mask

    def action(word, h):
        """Permute check labels by right multiplication; not a lift."""
        lo, hi = word & mask, word >> m
        aa, bb = h
        lo, hi = rotate(lo, aa), rotate(hi, -aa)
        if bb:
            lo, hi = hi, lo
        return lo | (hi << m)

    def canonical_syndrome(word):
        """Normalize at each occupied check; take the smallest image.

        An odd syndrome of weight <=9 cannot have a nontrivial
        stabilizer in D29. Its orbit therefore contains exactly 58
        syndromes, just like an orbit of variable triples.
        """
        lo, hi = word & mask, word >> m
        answer = None
        occupied = lo
        while occupied:
            bit = occupied & -occupied
            t = bit.bit_length() - 1
            image = rotate(lo, -t) | (rotate(hi, t) << m)
            if answer is None or image < answer:
                answer = image
            occupied ^= bit
        occupied = hi
        while occupied:
            bit = occupied & -occupied
            t = bit.bit_length() - 1
            image = rotate(hi, -t) | (rotate(lo, t) << m)
            if answer is None or image < answer:
                answer = image
            occupied ^= bit
        return answer

    def columns(rows, size):
        result = [0] * size
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def component(grid, side):
        """Extract the actual component from the verified scaffold lift."""
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = [row & cmask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            source = [row & cmask for row in hz]
            # Invert both variable and check group coordinates. This
            # permutes the verified right-action component into the
            # left-action coordinates used by the orbit census.
            rows = []
            for r in range(q):
                word = source[inv[r]]
                permuted = 0
                while word:
                    bit = word & -word
                    c = bit.bit_length() - 1
                    permuted |= 1 << ((c // q) * q + inv[c % q])
                    word ^= bit
                rows.append(permuted)
        return rows, columns(rows, nc)

    def no_four(cols):
        # All columns have odd weight three, excluding odd kernels.
        if len(set(cols)) != nc or any(x.bit_count() != 3 for x in cols):
            return False
        seen = set()
        for i, x in enumerate(cols):
            for y in cols[:i]:
                syndrome = x ^ y
                if syndrome == 0 or syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def graph_statistics(rows):
        """Actual check overlaps, Tanner cycles and nontrivial spectrum."""
        overlap = [[0] * q for _ in range(q)]
        four = squares = triangles = 0
        for i in range(q):
            for j in range(i):
                w = (rows[i] & rows[j]).bit_count()
                overlap[i][j] = overlap[j][i] = w
                four += w * (w - 1) // 2
                squares += w * w
        for i in range(q):
            for j in range(i + 1, q):
                ij = overlap[i][j]
                if ij:
                    for k in range(j + 1, q):
                        triangles += ij * overlap[i][k] * overlap[j][k]
        # Each degree-three variable supplies one non-Tanner triangle.
        six = triangles - squares + 2 * nc
        assert six >= 0

        # Fourier blocks of this verified D29-equivariant overlap graph.
        # The trivial degree eigenvalue is excluded; retain the sign
        # representation and both eigenvalues of every 2D block.
        weights = overlap[0]
        rotation_mass = sum(weights[:m])
        reflection_mass = sum(weights[m:])
        radius = abs(rotation_mass - reflection_mass)
        for cs, ss in trig:
            diagonal = sum(weights[t] * cs[t] for t in range(m))
            real = sum(weights[m + t] * cs[t] for t in range(m))
            imag = sum(weights[m + t] * ss[t] for t in range(m))
            radius = max(radius, abs(diagonal) + hypot(real, imag))
        return four, six, round(radius, 9)

    def six_kernels(cols, limit):
        """Exact weight-six count, after excluding smaller kernels.

        Equal canonical triple syndromes correspond to 58 ordinary
        syndrome collisions. Every six-set has ten unordered triple
        partitions. Orbit counting avoids enumerating 862924 triples.
        """
        buckets = {}
        collisions = 0
        boundary = [0] * 10
        for i, j, k in triple_orbits:
            syndrome = cols[i] ^ cols[j] ^ cols[k]
            weight = syndrome.bit_count()
            assert weight in (1, 3, 5, 7, 9)
            boundary[weight] += 1
            key = canonical_syndrome(syndrome)
            old = buckets.get(key, 0)
            collisions += old
            if limit is not None and collisions > limit:
                return None
            buckets[key] = old + 1
        assert (q * collisions) % 10 == 0
        return collisions, q * collisions // 10, tuple(boundary)

    def select(side, geometry):
        # A common reflection translation is gauge: fix the first t=0.
        # Search the remaining offsets, not translated copies.
        parameters = list(combinations(range(m), 2))
        parameters += [(v, u) for u, v in parameters]
        parameters += [(t, t) for t in range(m)]
        rng.shuffle(parameters)
        pool = []
        for t1, t2 in parameters:
            ts = (0, t1, t2)
            # For {1,x^d,x^t y}, the two involutory check differences
            # are x^t y and x^(t+d) y.
            reflection_differences = {
                value for d, t in zip(geometry, ts)
                for value in (t, (t + d) % m)
            }
            if len(reflection_differences) != 6:
                continue
            grid = [[
                [G.e, (d, 0), (t, 1)]
                for d, t in zip(geometry, ts)
            ]]
            rows, cols = component(grid, side)
            if gf2_rank(rows) != q or not no_four(cols):
                continue
            four, cycles6, radius = graph_statistics(rows)
            # Two forced involutory cycles per cell, with no additional
            # overlap: 3 cells * 2 * (q/2) = 3q.
            if four != 3 * q:
                continue
            pool.append(((radius, cycles6, ts), grid, cols))
            if len(pool) >= 160:
                break
        if not pool:
            raise ValueError("no minimally overlapping mixed component")
        pool.sort(key=lambda item: item[0])

        best = None
        best_key = None
        for cheap, grid, cols in pool[:32]:
            # Confirm the orbit action against columns of the verified
            # lift, including the permuted right-action B component.
            for h in ((1, 0), (0, 1)):
                hidx = G.idx(h)
                for block in range(3):
                    for c in range(q):
                        assert action(cols[block * q + c], h) == (
                            cols[block * q + mul[c][hidx]])
            census = six_kernels(
                cols, None if best_key is None else best_key[0])
            if census is None:
                continue
            collisions, words6, boundary = census
            key = (collisions, cheap[0], cheap[1],
                   boundary[1], boundary[3], cheap[2])
            if best_key is None or key < best_key:
                best_key = key
                best = (grid, words6, cheap[0], cheap[1])
            # Pool order protects expansion among the screened
            # zero-weight-six components.
            if collisions == 0:
                break
        if best is None:
            raise ValueError("component orbit screening found no candidate")
        return best

    aa = select("A", book[a])
    bb = select("B", book[b])
    A, B = aa[0], bb[0]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k != 232 or not css_ok(hx, hz):
        raise ValueError("verified complete-code rank or CSS check failed")
    assert max(row.bit_count() for row in hx + hz) == 12
    return {"idx": idx, "cells": {"A": A, "B": B},
            "geometry": (book[a], book[b]),
            "component_screen": (aa[1:], bb[1:])}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone give reliable r040f memory: parent fitness 3.2187, all five measured codes hit a failed-shot cap, with weight-4 sector-1 witnesses and broadly distributed logical flips.
# REFUTED: Balanced collision screening plus d2s=8 certifies robust circuit memory: current slot 0 still had 105/3500 X failures and slot 21 had 106/2500 Z failures, with logical flips spanning all 232 logical qubits.
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
