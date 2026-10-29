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
CONCEPT = ("On SL(2,3), order 24, shapes (1,3)x(2,7), quaternion-quotient unit trinomials and noncommuting degree-two B connections screened through weight six should improve the 0.450 frontier's circuit fitness beyond 4.0835.")
GROUP = ("Jump from A4 x C2 to the determinant-one matrix group over F3. SL(2,3) has a unique involution, a normal quaternion subgroup Q8, quotient C3, and order-four relative directions absent from the parent. The related order-24 seven-class bed scored about 4.1 on (1,3)x(2,5), but that different-shape reading establishes no group advantage. Quotient-span-one/two trinomials retain the monomial-image unit construction.")
SHAPES = ("Keep (1,3)x(2,7) for cell 0.450: q=24, n=552, kb=240 and guaranteed rate 10/23=0.43478. Components have 72 variables/24 checks and 168 variables/48 checks. Require verified k=240. Parent weight-five/six witnesses occupied many sector-one Acol/Bcol blocks; failure enrichment was diffuse near 1.0-1.1x, without a consistent Brow hotspot, motivating global component repair.")
FILLING = ("Twenty contracts cross four A relative-order patterns with five support architectures. Four architectures have A counts (3,3,3), B row totals (9,9), four degree-three columns and three singleton-pair degree-two columns; those three connections have noncommuting relative-order spectrum (4,6,6). A fifth architecture tests (2,3,3) and one degree-four B column. Both CSS weight sums are <=12; CNOT budgets remain 20736-22032. Fresh verified lifts shortlist exact weight-three/four kernels, then count component weight-five/six words exactly when shorter words are absent and balance small triple boundaries before multiplicity. Full matrices verify ranks, CSS and short logicals. Target the parent's 115/750 Z and 142/750 X failures and widespread collective flips despite d2s reaching six; only ASC measurements test the proposed improvement.")
FAMILY_NAME = "r045_SL2F3_q24_13x27_quaternion_sixscreen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    from itertools import product
    if not valid_q(q):
        raise ValueError("SL(2,3) requires order 24")
    # (a,b,c,d) represents [[a,b],[c,d]] over F3.
    els = [g for g in product(range(3), repeat=4)
           if (g[0]*g[3] - g[1]*g[2]) % 3 == 1]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 3 for x in g)

    def mul(g, h):
        a, b, c, d = red(g)
        e, f, u, v = red(h)
        return ((a*e + b*u) % 3, (a*f + b*v) % 3,
                (c*e + d*u) % 3, (c*f + d*v) % 3)

    def inv(g):
        a, b, c, d = red(g)
        return (d, (-b) % 3, (-c) % 3, a)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _component_spectrum56(cols, short):
    """Exact weight-five/six counts after exclusion through weight four.

    Unique nonzero columns and zero weight-three/four counts ensure
    matched supports are disjoint. Each weight-five word has ten
    pair/triple partitions; each weight-six word has ten unordered
    triple/triple partitions. Boundary exposure is only a BP surrogate.
    """
    if short[0] or short[1]:
        return (10**12, 10**12, 10**12)
    pairs = {}
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            pairs[value] = pairs.get(value, 0) + 1
    triples = {}
    five = six = boundary = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                five += pairs.get(value, 0)
                old = triples.get(value, 0)
                six += old
                triples[value] = old + 1
                weight = value.bit_count()
                if weight <= 3:
                    boundary += (4 - weight) ** 2
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10, boundary


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Five support contracts crossed with four quaternion-order briefs."""
    from itertools import combinations

    q = G.q
    assert q == 24
    architecture = (idx // 4) % 5
    mode = idx % 4
    extension = idx // 20
    sa = [[3, 3, 3]] if architecture < 4 else [[2, 3, 3]]
    column_profiles = (
        ((2, 1), (2, 1), (1, 2), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (1, 2), (1, 2), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (0, 3), (2, 1), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (3, 0), (0, 3), (0, 3),
         (1, 1), (1, 1), (1, 1)),
        ((2, 2), (2, 1), (1, 2), (1, 1),
         (1, 1), (1, 1), (1, 1)),
    )
    profile = column_profiles[architecture]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert tuple(map(sum, sb)) == (9, 9)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    # Check BOTH weight sums explicitly.
    assert max(map(sum, sa)) + max(map(sum, profile)) <= 12
    assert max(map(sum, sb)) + max(map(sum, zip(*sa))) <= 12

    orders = _orders(G)
    # In SL(2,3), exactly the elements of orders 1,2,4 form Q8.
    quaternion = [g for g in G.elements
                  if orders[G.idx(g)] in (1, 2, 4)]
    assert len(quaternion) == 8

    def quotient(g):
        return min(G.idx(G.mul(h, g)) for h in quaternion)

    quotients = {G.idx(g): quotient(g) for g in G.elements}
    assert len(set(quotients.values())) == 3
    nonidentity = [g for g in G.elements
                   if G._key(g) != G._key(G.e)]
    ratio_pools = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (4, 6)
    }
    assert all(ratio_pools.values())

    a_patterns = (
        ((4, 6, 6), (4, 6, 6), (4, 6, 6)),
        ((3, 4, 6), (4, 6, 6), (4, 6, 6)),
        ((3, 3, 4), (3, 4, 6), (4, 6, 6)),
        ((4, 4, 4), (3, 4, 6), (4, 6, 6)),
    )
    triangles = {signature: []
                 for pattern in a_patterns for signature in pattern}
    for u, v in combinations(nonidentity, 2):
        relative = G.mul(G.inv(u), v)
        signature = tuple(sorted((
            orders[G.idx(u)], orders[G.idx(v)],
            orders[G.idx(relative)])))
        if signature not in triangles:
            continue
        cell = [G.e, u, v]
        if len({quotients[G.idx(g)] for g in cell}) not in (1, 2):
            continue
        if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
            continue
        triangles[signature].append(cell)
    assert all(triangles.values())

    def triple(number, side):
        policy = mode if side == "A" else (mode + extension) % 4
        signature = a_patterns[policy][number % 3]
        relative = list(rng.choice(triangles[signature]))
        if side == "A":
            return relative
        base = G.pick(rng)
        return [G.mul(g, base) for g in relative]

    def draw_grid(side, support):
        grid = []
        heavy = 0
        # For singleton pairs, t=b0*b1^-1 is the actual right-component
        # connection. Avoid searching monomial column-gauge copies.
        pair_sites = [j for j, column in enumerate(profile)
                      if column == (1, 1)]
        paired = {}
        if side == "B":
            holonomies = list(rng.choice(triangles[(4, 6, 6)]))
            while len(holonomies) < len(pair_sites):
                choices = [g for g in nonidentity
                           if orders[G.idx(g)] in (4, 6)
                           and g not in holonomies]
                holonomies.append(rng.choice(choices))
            paired = dict(zip(pair_sites, holonomies))

        for i, row_counts in enumerate(support):
            row = []
            used = set()
            binary = 0
            for j, count in enumerate(row_counts):
                if side == "B" and j in paired:
                    t = paired[j]
                    cell = [G.e] if i == 0 else [G.inv(t)]
                elif count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    if side == "A":
                        order = 6
                    else:
                        palettes = ((6,), (4,), (4, 6))
                        palette = palettes[extension % 3]
                        order = palette[(binary + mode) % len(palette)]
                    choices = [g for g in ratio_pools[order]
                               if G._key(g) not in used]
                    if not choices:
                        raise ValueError("quaternion binomial brief exhausted")
                    delta = rng.choice(choices)
                    used.update((G._key(delta), G._key(G.inv(delta))))
                    base = G.e if side == "A" else G.pick(rng)
                    cell = [base, G.mul(delta, base)]
                    binary += 1
                else:
                    cell = triple(heavy, side)
                    heavy += 1
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                row.append(cell)
            grid.append(row)
        return grid

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def components(hx, hz):
        # Extract exclusively from the helper-built complete matrices.
        # Bcol0 X checks contain the left-lifted A component.
        mask = (1 << q) - 1
        ar = []
        for row in hx[:q]:
            compact = 0
            for j in range(A_SHAPE[1]):
                base = j * B_SHAPE[1] * q
                compact |= ((row >> base) & mask) << (j * q)
            ar.append(compact)
        # Acol0 Z checks contain BOTH right-lifted B check rows.
        width = B_SHAPE[1] * q
        br = [row & ((1 << width) - 1)
              for row in hz[:B_SHAPE[0] * q]]
        return ar, br

    def kernel_stats(cols):
        """Exact weight-three/four kernel counts for unique columns."""
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        singles = set(cols)
        pairs = {}
        triples = collisions = cycles = 0
        weights = [c.bit_count() for c in cols]
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                triples += int(s in singles)
                old = pairs.get(s, 0)
                collisions += old
                pairs[s] = old + 1
                overlap = (weights[i] + weights[j] - s.bit_count()) // 2
                cycles += overlap * (overlap - 1) // 2
        assert triples % 3 == 0 and collisions % 3 == 0
        return triples // 3, collisions // 3, cycles

    def logical_stats(checks, stabilizers, n):
        """Exact logical counts and shared-check hook opportunities.

        Unique nonzero columns exclude weights one and two. Quotient
        residues classify collisions modulo the stabilizer row span.
        Hook statistics count possible fault-pair partitions, not
        ASC-certified faults or an estimated circuit failure rate.
        """
        cols = columns(checks, n)
        if 0 in cols or len(set(cols)) != n:
            return None
        spread = columns(stabilizers, n)
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items(), reverse=True)
        residuals = []
        for i in range(n):
            v = 1 << i
            for p, row in ordered:
                if (v >> p) & 1:
                    v ^= row
            residuals.append(v)

        singles = {s: residuals[i] for i, s in enumerate(cols)}
        totals = {}
        classes = {}
        triples = fours = double_hook = single_hook = 0
        for i, x in enumerate(cols):
            ri = residuals[i]
            hi = spread[i]
            for j in range(i):
                s = x ^ cols[j]
                residue = ri ^ residuals[j]
                single = singles.get(s)
                if single is not None and residue != single:
                    triples += 1
                hook = (hi & spread[j]).bit_count()
                key = (s, residue)
                nt, ht = totals.get(s, (0, 0))
                nc, hc = classes.get(key, (0, 0))
                different = nt - nc
                different_hooks = ht - hc
                fours += different
                double_hook += hook * different_hooks
                single_hook += hook * different + different_hooks
                totals[s] = (nt + 1, ht + hook)
                classes[key] = (nc + 1, hc + hook)
        assert triples % 3 == 0 and fours % 3 == 0
        return triples // 3, fours // 3, double_hook, single_hook

    shortlist = []
    exact_seen = set()
    for trial in range(320):
        try:
            A = draw_grid("A", sa)
            B = draw_grid("B", sb)
        except ValueError:
            continue
        identity = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row
        )
        if identity in exact_seen:
            continue
        exact_seen.add(identity)

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = components(hx, hz)
        if gf2_rank(ar) != A_SHAPE[0] * q:
            continue
        if gf2_rank(br) != B_SHAPE[0] * q:
            continue
        ast = kernel_stats(columns(ar, A_SHAPE[1] * q))
        bst = kernel_stats(columns(br, B_SHAPE[1] * q))
        if ast is None or bst is None:
            continue

        # Component words can replicate across opposite-side columns.
        # This coarse weighting is only a shortlisting heuristic.
        a3, b3 = B_SHAPE[1] * ast[0], A_SHAPE[1] * bst[0]
        a4, b4 = B_SHAPE[1] * ast[1], A_SHAPE[1] * bst[1]
        ca, cb = ast[2] / len(ar), bst[2] / len(br)
        key = (max(a3, b3), a3 + b3,
               max(a4, b4), a4 + b4,
               max(ca, cb), ca + cb)
        shortlist.append((key, identity, A, B, hx, hz, n))
        shortlist.sort(key=lambda item: item[:2])
        del shortlist[8:]

    best = None
    best_key = None
    for component_key, identity, A, B, hx, hz, n in shortlist:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 10 * q or not (425 * n <= 1000 * k < 475 * n):
            continue
        ar, br = components(hx, hz)
        acols = columns(ar, A_SHAPE[1] * q)
        bcols = columns(br, B_SHAPE[1] * q)
        ast, bst = kernel_stats(acols), kernel_stats(bcols)
        if ast is None or bst is None:
            continue
        asp = _component_spectrum56(acols, ast)
        bsp = _component_spectrum56(bcols, bst)
        lx = logical_stats(hx, hz, n)
        lz = logical_stats(hz, hx, n)
        if lx is None or lz is None:
            continue
        fx = _same_basis_cycles(hx)
        fz = _same_basis_cycles(hz)
        # Exact component counts are NOT an exact quantum logical
        # census. Replication weights and triple boundaries remain
        # uncalibrated circuit-risk surrogates.
        a5, b5 = B_SHAPE[1] * asp[0], A_SHAPE[1] * bsp[0]
        a6, b6 = B_SHAPE[1] * asp[1], A_SHAPE[1] * bsp[1]
        ba, bb = asp[2] / len(ar), bsp[2] / len(br)
        key = (max(lx[0], lz[0]), lx[0] + lz[0],
               max(lx[1], lz[1]), lx[1] + lz[1],
               max(ast[0], bst[0]), ast[0] + bst[0],
               max(ast[1], bst[1]), ast[1] + bst[1],
               int(bool(a5 or b5)), int(bool(a6 or b6)),
               max(ba, bb), ba + bb,
               max(a5, b5), a5 + b5,
               max(a6, b6), a6 + b6,
               max(lx[2], lz[2]), lx[2] + lz[2],
               max(lx[3], lz[3]), lx[3] + lz[3],
               max(fx / len(hx), fz / len(hz)),
               fx + fz, component_key, identity)
        if best_key is None or key < best_key:
            best_key = key
            best = (A, B, hx, hz, k, lx, lz)

    if best is None:
        raise ValueError("no full-rank two-check filling passed screening")
    A, B, hx, hz, k, lx, lz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    edges = sum(row.bit_count() for row in hx + hz)
    expected = q * (
        (B_SHAPE[1] + B_SHAPE[0]) * sum(map(sum, sa))
        + (A_SHAPE[1] + A_SHAPE[0]) * sum(map(sum, sb)))
    assert edges == expected
    return {"idx": idx, "architecture": architecture, "mode": mode,
            "extension": extension, "k": k, "screen": best_key,
            "logical_profile": (lx, lz), "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing four-cycles alone did not protect this A4 x C3 bed: fitness 3.2178, Z failures 112..137/250, and widespread weight-three sector-1 witnesses despite mostly stage-1 settlement.
# REFUTED: Low gate count and connected binomial A suffice for reliable balanced memory: the triangle-transfer parent scored 3.4383, retained 48 weight-three Z witnesses, and failed 108/250 to 181/500 X shots despite first-stage settlement.
# REFUTED: Unit trinomials, full component ranks and exact weight-three/four hook screening suffice for rare collective failures on A4 x C2 (1,3)x(2,7): fitness was 3.9832; measured slots reached failed-shot caps, retained abundant sector-one weight-five/six witnesses, and failures touched all 240 logicals.
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
