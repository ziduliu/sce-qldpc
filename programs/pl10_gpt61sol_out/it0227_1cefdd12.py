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
CONCEPT = ("For A4 x C2, order 24, shapes A=(1,4), B=(1,4), order-six binomials and algebraic-unit trinomials with reduced verified syndrome-boundary exposure should improve worse-basis circuit fitness beyond the approximately 3.5 record of cell 0.550.")
GROUP = ("Retain the A4 x cyclic direct-product construction rule, changing only the cyclic factor from three to two. Order 36 cannot fit the smallest balanced target shape because 17*36=612; order 24 gives n=408. The normal 2-subgroup V4 x C2 supplies unit trinomials, and order-six differences remain available. The parent's 3.3437 measurement establishes no advantage for either order.")
SHAPES = ("Equal (1,4) components each have 24 checks on 96 variables. n=408 and kb=216 give guaranteed rate 9/17=0.529412, already inside the open-ended 0.550 cell. Require verified 216<=k<=225. Parent weight-three/four witnesses occupied sector-one Acol/Bcol blocks, while failure enrichment was diffuse at 1.0-1.2x without a persistent Brow hotspot; redesign both components rather than infer a localized repair.")
FILLING = ("Twenty contracts cross five support allocations with four unit-trinomial geometries: normal-2-subgroup support, noncommuting ratio spectra (2,6,6) and (2,3,3), and commuting spectrum (2,3,6). Every binomial has order-six difference. Additional harness batches change the A/B geometry pairing and normal-support contract. Verified component lifts limit rank deficits and compare short kernels and small pair boundaries. Complete lifts classify actual logical supports through weight four, then prioritize opposite-check-weighted pair exposure after protection through weight three. Both check-weight sums are <=12; six-round CNOT budgets are 12240-12960 rather than 15120. Target the parent's 151/250 Z and 186/500 X failures and widespread collective flips; the exposure statistic is not an ASC-certified hook count or calibrated pL estimate.")
FAMILY_NAME = "r055_A4xC2_14x14_unit6_pairboundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C2 requires order 24")
    # Same direct-product construction, with cyclic factor order two.
    central_order = 2
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(central_order)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (
            int(g[4]) % central_order,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + (
            (g[4] + h[4]) % central_order,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % central_order,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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


def _designed_cell(rng, G, count, target_order, mode, orders):
    """Avoid order-two differences; prescribe binomial cycle lengths."""
    for _ in range(512):
        cell = rng.sample(G.elements, count)
        # Distinct projected permutations prevent pure-central pairs.
        if len({tuple(g[:4]) for g in cell}) != count:
            continue
        relative = [
            orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))]
            for a in range(count) for b in range(a + 1, count)
        ]
        if min(relative) < 3:
            continue
        if count == 2:
            if relative[0] != target_order:
                continue
        else:
            central_span = len({g[4] for g in cell})
            wanted = 3 if mode == 0 else 2
            if central_span != wanted:
                continue
        return cell
    raise ValueError("could not realize cell design")


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Optimize a target-rate support/unit contract on verified lifts."""
    profiles = (
        ((2, 2, 2, 3), (2, 2, 2, 3)),
        ((2, 2, 2, 2), (2, 2, 3, 3)),
        ((2, 2, 2, 2), (1, 3, 3, 3)),
        ((2, 2, 2, 3), (2, 2, 2, 2)),
        ((1, 3, 3, 3), (2, 2, 2, 2)),
    )
    case, batch = idx % 20, idx // 20
    pa, pb = profiles[case // 4]
    mode = case % 4
    side_modes = {"A": mode, "B": (mode + batch) % 4}
    sa, sb = [list(pa)], [list(pb)]
    q, e = G.q, G.e
    assert q == 24 and A_SHAPE == B_SHAPE == (1, 4)
    # Check both budgets, including the unequal support allocations.
    wx = sum(pa) + max(pb)
    wz = sum(pb) + max(pa)
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    orders = _orders(G)
    six = [g for g in G.elements if orders[G.idx(g)] == 6]
    normal = {G._key(g) for g in G.elements
              if orders[G.idx(g)] <= 2}
    assert len(normal) == 8 and len(six) == 8

    # V4 x C2 is a normal 2-subgroup. Odd augmentation in its
    # group algebra gives a unit. A trinomial with two terms in
    # one normal coset and the third in another also maps to a
    # monomial modulo it; the kernel ideal is nilpotent.
    nonidentity = [g for g in G.elements if G._key(g) != G._key(e)]
    triangles = {m: [] for m in range(4)}
    for a, g in enumerate(nonidentity):
        for h in nonidentity[a + 1:]:
            cell = [e, g, h]
            relative = G.mul(G.inv(g), h)
            spectrum = tuple(sorted((
                orders[G.idx(g)], orders[G.idx(h)],
                orders[G.idx(relative)],
            )))
            commute = G._key(G.mul(g, h)) == G._key(G.mul(h, g))
            if all(G._key(x) in normal for x in cell):
                span = len({x[4] for x in cell})
                central_pair = any(
                    tuple(u[:4]) == tuple(v[:4])
                    for t, u in enumerate(cell) for v in cell[:t]
                )
                normal_policy = batch % 3
                legal = (
                    (span == 2 and not central_pair)
                    if normal_policy == 0 else
                    (span == 1)
                    if normal_policy == 1 else
                    central_pair
                )
                if legal:
                    triangles[0].append(cell)
            elif spectrum == (2, 6, 6) and not commute:
                triangles[1].append(cell)
            elif spectrum == (2, 3, 3) and not commute:
                triangles[2].append(cell)
            elif spectrum == (2, 3, 6):
                # Further batches change an intrinsic commutation
                # contract, not a saved code's labeling orbit.
                if commute == (batch % 2 == 0):
                    triangles[3].append(cell)
    if not all(triangles.values()):
        raise ValueError("empty algebraic-unit triangle contract")

    # Independently verify the claimed unit property using the exact
    # A-left singleton block, not a different lift convention.
    for book in triangles.values():
        for cell in book:
            rows = lifted_matrices(G, [[cell]], [[[]]])[0]
            if gf2_rank(rows) != q:
                raise ValueError("unit triangle failed verified rank test")

    def columns(rows, ncols):
        result = [0] * ncols
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                low = row & -row
                result[low.bit_length() - 1] |= bit
                row ^= low
        return result

    def component_quality(side, T):
        # A one-cell opposite-side probe extracts the exact action used
        # by the scaffold. No independent left/right lift is implemented.
        if side == "A":
            rows = lifted_matrices(G, T, [[[e]]])[0]
        else:
            rows = lifted_matrices(G, [[[e]]], T)[1]
        ncols = len(T[0]) * q
        mask = (1 << ncols) - 1
        rows = [row & mask for row in rows]
        deficit = q - gf2_rank(rows)
        if deficit > 1:
            return None
        cols = columns(rows, ncols)
        if 0 in cols or len(set(cols)) != ncols:
            return None
        singles = set(cols)
        pairs = {}
        n3 = n4 = cycles = exposure = 0
        weights = [c.bit_count() for c in cols]
        for a, x in enumerate(cols):
            for b in range(a):
                s = x ^ cols[b]
                n3 += int(s in singles)
                old = pairs.get(s, 0)
                n4 += old
                pairs[s] = old + 1
                boundary = s.bit_count()
                exposure += (weights[a] * weights[b]
                             * max(0, 5 - boundary) ** 2)
                overlap = (weights[a] + weights[b] - boundary) // 2
                cycles += overlap * (overlap - 1) // 2
        # Nonzero distinct columns make these exact classical counts.
        # Pair-boundary exposure is only a neighborhood heuristic.
        assert n3 % 3 == 0 and n4 % 3 == 0
        return (n3 // 3, n4 // 3, exposure, cycles, deficit)

    def design_side(side, counts):
        # Order-three binomials create isolated weight-three cycles.
        # Use order six on every doubleton, with distinct inverse pairs.
        options = []
        for w in counts:
            if w == 1:
                options.append([[e]])
            elif w == 2:
                options.append([[e, g] for g in six])
            elif w == 3:
                options.append(triangles[side_modes[side]])
            else:
                raise ValueError("unsupported target-shape term count")

        def valid_ratios(T):
            used = set()
            for cell in T[0]:
                if len(cell) != 2:
                    continue
                g = cell[1]
                key, inverse = G._key(g), G._key(G.inv(g))
                if key in used or inverse in used:
                    return False
                used.update((key, inverse))
            return True

        def fingerprint(T):
            return tuple(tuple(sorted(G.idx(g) for g in cell))
                         for cell in T[0])

        pool = []
        seen = set()

        def retain(T):
            signature = fingerprint(T)
            if signature in seen or not valid_ratios(T):
                return None
            seen.add(signature)
            quality = component_quality(side, T)
            if quality is None:
                return None
            pool.append((quality, signature, T))
            pool.sort(key=lambda item: (item[0], item[1]))
            del pool[5:]
            return quality

        for trial in range(80):
            T = [[list(rng.choice(choices)) for choices in options]]
            retain(T)
        if not pool:
            raise ValueError("no distinct-syndrome component passed rank screen")

        # Whole-relative-cell mutations preserve this slot's count and
        # ratio/phase architecture. Equal-cost moves explore plateaus.
        starts = list(pool[:3])
        mutable = [j for j, choices in enumerate(options) if len(choices) > 1]
        for quality, signature, initial in starts:
            current = [[list(cell) for cell in initial[0]]]
            current_quality = quality
            for step in range(32):
                j = rng.choice(mutable)
                trial = [[list(cell) for cell in current[0]]]
                trial[0][j] = list(rng.choice(options[j]))
                candidate_quality = retain(trial)
                if candidate_quality is not None and (
                        candidate_quality < current_quality or
                        (candidate_quality == current_quality
                         and rng.random() < 0.35)):
                    current, current_quality = trial, candidate_quality
        return pool

    apool = design_side("A", pa)
    bpool = design_side("B", pb)

    def logical_counts(checks, stabilizers, n):
        """Exact non-stabilizer kernel-support counts through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(v):
            while v:
                p = v.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                v ^= old
            return False

        cols = columns(checks, n)
        bits = [1 << a for a in range(n)]
        singles = {}
        result = [0, 0, 0, 0]
        for a, syndrome in enumerate(cols):
            singles.setdefault(syndrome, []).append(a)
            if syndrome == 0 and nontrivial(bits[a]):
                result[0] += 1
        pairs = {}
        for a, x in enumerate(cols):
            for b in range(a + 1, n):
                s = x ^ cols[b]
                support = bits[a] | bits[b]
                if s == 0 and nontrivial(support):
                    result[1] += 1
                for c in singles.get(s, ()):
                    if c > b and nontrivial(support | bits[c]):
                        result[2] += 1
                bucket = pairs.setdefault(s, [])
                # c < d < a < b selects exactly one partition of each
                # ordered weight-four support; overlapping pairs are ignored.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        result[3] += 1
                bucket.append((a, b, support))
        return tuple(result)

    finalists = []
    for aq, af, A in apool:
        for bq, bf, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 408 or not (216 <= k <= 225):
                continue
            key = (max(aq[0], bq[0]), aq[0] + bq[0],
                   max(aq[1], bq[1]), aq[1] + bq[1],
                   max(aq[2], bq[2]), aq[2] + bq[2],
                   aq[3] + bq[3], af, bf)
            finalists.append((key, A, B, hx, hz, n, k))
    finalists.sort(key=lambda item: item[0])

    def pair_exposure(checks, other, n):
        """Opposite-check-weighted small pair boundaries.

        Shared opposite checks mark potential correlated-fault
        opportunities, not hooks certified for the ASC ordering.
        """
        cols = columns(checks, n)
        mate = columns(other, n)
        degree = [a.bit_count() + b.bit_count()
                  for a, b in zip(cols, mate)]
        total = 0
        for a, x in enumerate(cols):
            for b in range(a):
                boundary = (x ^ cols[b]).bit_count()
                if boundary <= 4:
                    shared = (mate[a] & mate[b]).bit_count()
                    total += (degree[a] * degree[b]
                              * (5 - boundary) ** 2 * (1 + shared))
        return total / len(checks)

    best = None
    for component_key, A, B, hx, hz, n, k in finalists[:5]:
        if not css_ok(hx, hz):
            continue
        x = logical_counts(hz, hx, n)
        z = logical_counts(hx, hz, n)
        # Exact short protection first. Once logicals through weight
        # three are matched, compare fault-opportunity exposure before
        # further optimizing the distance-screen multiplicity.
        short = tuple(v for t in range(3)
                      for v in (max(x[t], z[t]), x[t] + z[t]))
        ex = pair_exposure(hx, hz, n)
        ez = pair_exposure(hz, hx, n)
        edges = sum(row.bit_count() for row in hx + hz)
        key = short + (
            max(ex, ez), ex + ez,
            max(x[3], z[3]), x[3] + z[3],
            edges, component_key,
        )
        if best is None or key < best[0]:
            best = (key, A, B, k, hx, hz, ex, ez)
    if best is None:
        raise ValueError("no screened filling retained the 0.550 rate cell")
    key, A, B, k, hx, hz, ex, ez = best
    assert lifted_k(G, A, B) == k
    assert css_ok(hx, hz)
    assert k / 408 > 0.525
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    expected_edges = q * (
        (B_SHAPE[1] + B_SHAPE[0]) * sum(pa)
        + (A_SHAPE[1] + A_SHAPE[0]) * sum(pb)
    )
    assert sum(row.bit_count() for row in hx + hz) == expected_edges
    assert 6 * expected_edges in (12240, 12960)
    return {"idx": idx, "case": case, "batch": batch, "k": k,
            "screen": key, "unit_modes": side_modes,
            "pair_exposure": (ex, ez),
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory differences and minimizing four-cycles suffices for reliable r045 memory: fitness was 3.2178, measured Z failures were 112..137/250 shots, and every measured code had weight-three X-type logical witnesses.
# REFUTED: Verified weight-three/four multiplicity minimization suffices for rare collective failures on A4 x C3 (1,3)x(1,4): fitness was 3.3437, every measured basis reached the failed-shot cap, and failures affected essentially every logical qubit despite predominantly first-pass decoder settlement.
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
