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
CONCEPT = ("Cyclic C17, order 17, shapes (2,5)x(2,6): degree-three components and stabilizer-aware complete-code BP selection should raise the weaker 0.350 frontier entry above its measured 5.1, rather than refine the near-saturated 0.100 cell.")
GROUP = ("Jump from nonabelian Dic4 to abelian C17. All nonidentity differences have order seventeen, so there are no involutions or short proper-subgroup differences. The measured C17 (3,5)x(3,5) bed reached about 7.2, whereas the destination (2,5)x(2,6) bed reached 4.8. These readings motivate a new filling experiment but do not establish a causal group advantage.")
SHAPES = ("A=(2,5), B=(2,6) have 85 and 102 component variables, respectively, and 34 checks each. The size factor is 34, allowing q<=17; q=17 gives n=578, kb=204 and guaranteed rate 0.352941 in cell 0.350. Verified full ranks retain 204<=k<=216. Parent measured failures enriched different Acol/Bcol blocks, including Acol2/Bcol0 and Acol4/Bcol0; one or two failures per worse basis do not identify a persistent Brow defect. Redesign both components rather than patching one block.")
FILLING = ("Twenty contracts cross four degree-three support pairs with five canonical joint relative geometries; extra slots extend the geometry book, not saved-code relabelings. A row totals are seven/eight and B totals nine/nine, so HX weights are at most eleven and HZ twelve. Six-round extraction uses 25092 CNOTs versus the parent's 27648-28416. Fresh offset searches use helper-derived actions, actual component ranks, short-kernel exclusion, weight-six multiplicity, triple boundaries and capped sampled witnesses. Complete helper-built finalists screen actual k, CSS and logicals<=4, then compare common-random-number damped BP probes modulo actual stabilizers. The probes omit ASC ordering and measurement histories and are not circuit pL measurements. Target broad collective ambiguity and late decoder settlement, not a larger distance upper bound.")
FAMILY_NAME = "r035_C17_25x26_degree3_jointgeometry_verifiedbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This cyclic bed requires order seventeen")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


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
    """Project both components from the verified full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    arows = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            packed = 0
            for j in range(ca):
                packed |= ((row >> (j * cb * q)) & mask) << (j * q)
            arows.append(packed)
    bmask = (1 << (cb * q)) - 1
    brows = [hz[a * q + r] & bmask
             for a in range(rb) for r in range(q)]
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
    """Actual sampled kernel witnesses: upper bounds, never certificates."""
    cols = _dv_columns(rows, n)
    order = list(range(n))
    best = n + 1
    for _ in range(trials):
        rng.shuffle(order)
        piv, basis = {}, []
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
                basis.append(word)
                best = min(best, word.bit_count())
        # Every XOR remains in the kernel of the verified component.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposal = word ^ other
                    if proposal:
                        pw = proposal.bit_count()
                        best = min(best, pw)
                        if pw < weight:
                            word, weight = proposal, pw
                basis[i] = word
        if best <= 6:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Indexed relative geometries, verified components, then joint BP."""
    import numpy as np
    from collections import Counter

    q = G.q
    assert q == 17
    assert A_SHAPE == (2, 5) and B_SHAPE == (2, 6)
    design, mode = idx % 20, idx // 20
    architecture, geometry_number = divmod(design, 5)
    geometry_number += 5 * mode

    # Columns always total three. The four support pairs differ in
    # whether either component contains a zero/trinomial column.
    first_a = ((1, 1, 1, 2, 2), (0, 1, 2, 2, 2))
    first_b = ((1, 1, 1, 2, 2, 2), (0, 1, 2, 2, 2, 2))
    pa, pb = ((0, 0), (0, 1), (1, 0), (1, 1))[architecture]
    sa = [list(first_a[pa]), [3-t for t in first_a[pa]]]
    sb = [list(first_b[pb]), [3-t for t in first_b[pb]]]
    supports = (sa, sb)
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert (wx, wz) == (11, 12)
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    for counts in supports:
        assert all(sum(column) == 3 for column in zip(*counts))
        assert gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in counts]) == 2

    # Cache only singleton actions obtained from the verified helper.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G._key(g)] = tuple(row & mask for row in hx)
        actions[1][G._key(g)] = tuple(row & mask for row in hz)

    def component_rows(grid, side):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G._key(g)]):
                        out[t] ^= bits << (j*q)
            rows.extend(out)
        return rows

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def normalized(cell, multiplier):
        if not cell:
            return ()
        values = [multiplier*t % q for t in cell]
        return min(tuple(sorted((t-origin) % q for t in values))
                   for origin in values)

    def geometry_key(grids):
        """Remove explicit affine/row/column images, not arbitrary graphs."""
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                columns = [
                    (normalized(grid[0][j], u),
                     normalized(grid[1][j], u))
                    for j in range(len(grid[0]))]
                sides.append(min(
                    tuple(sorted(columns)),
                    tuple(sorted((b, a) for a, b in columns))))
            images.append(tuple(sides))
        return min(images)

    def internal_exposure(grids):
        penalties = []
        for grid in grids:
            for row in grid:
                differences = Counter(
                    (a-b) % q for cell in row
                    for a in cell for b in cell if a != b)
                penalties.append(sum(v*(v-1)//2
                                     for v in differences.values()))
        return max(penalties), sum(penalties)

    # This deterministic book assigns relative structures before offset
    # optimization. Extra slots extend the book rather than resampling
    # an already assigned relative architecture.
    grng = random.Random(170025006 + 1009*architecture)
    geometries, seen = [], set()
    for attempt in range(256 + 16*geometry_number):
        grids = [
            [[([] if count == 0 else
               [0] + grng.sample(range(1, q), count-1))
              for count in row] for row in counts]
            for counts in supports]
        identity = geometry_key(grids)
        if identity in seen:
            continue
        seen.add(identity)
        geometries.append((internal_exposure(grids), identity, grids))
    geometries.sort(key=lambda entry: entry[:2])
    if len(geometries) <= geometry_number:
        raise ValueError("relative-geometry book exhausted")
    relative = geometries[geometry_number][2]

    def draw(side):
        return [[[((origin+t) % q,) for t in cell]
                 for cell in row
                 for origin in [rng.randrange(q)]]
                for row in relative[side]]

    def connected(rows, width):
        cols = _dv_columns(rows, width)
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
            a = first.bit_length()-1
            syndrome ^= first
            while syndrome:
                bit = syndrome & -syndrome
                parent[root(bit.bit_length()-1)] = root(a)
                syndrome ^= bit
        return len({root(a) for a in range(len(rows))}) == 1

    def boundary_profile(rows, width):
        """Exact weight-six census after <=4 exclusion; boundary surrogate."""
        cols = _dv_columns(rows, width)
        assert all(value.bit_count() == 3 for value in cols)
        triples = {}
        collisions = boundary = 0
        loads = [0] * (width // q)
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    old = triples.get(syndrome, 0)
                    collisions += old
                    triples[syndrome] = old + 1
                    weight = syndrome.bit_count()
                    if weight <= 3:
                        penalty = (5-weight)**2
                        boundary += penalty
                        for block in {i//q, j//q, h//q}:
                            loads[block] += penalty
        # Odd column weights exclude odd kernel words. With <=4 absent,
        # matching triples are disjoint; each six-set has ten partitions.
        assert collisions % 10 == 0
        return collisions//10, max(loads), boundary

    def optimize(side):
        width = len(supports[side][0]) * q
        positions = [(i, j) for i, row in enumerate(relative[side])
                     for j, cell in enumerate(row) if cell]
        pool = []
        for trial in range(128):
            grid = draw(side)
            rows = component_rows(grid, side)
            coarse = _dv_cycles(rows, width)
            pool.append((coarse, trial, grid, rows))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        finalists = []
        for coarse, trial, grid, rows in pool:
            # Cell translations change inter-cell holonomy but preserve
            # the deliberate slot's relative-support architecture.
            for step in range(80):
                proposal = clone(grid)
                i, j = rng.choice(positions)
                shift = rng.randrange(1, q)
                proposal[i][j] = [((g[0]+shift) % q,)
                                  for g in proposal[i][j]]
                proposed_rows = component_rows(proposal, side)
                proposed = _dv_cycles(proposed_rows, width)
                if proposed <= coarse:
                    grid, rows, coarse = proposal, proposed_rows, proposed
            if gf2_rank(rows) != 2*q or not connected(rows, width):
                continue
            if _dv_low4(rows, [], width):
                continue
            words6, peak, boundary = boundary_profile(rows, width)
            distance = _dv_kernel_estimate(rng, rows, width, 40)
            # Only a sampled warning floor; never a distance certificate.
            key = (int(distance < 8), words6, peak, boundary,
                   coarse, -min(distance, 10), trial)
            finalists.append((key, grid, rows))
        finalists.sort(key=lambda item: item[0])
        return finalists[:3]

    def probe(checks, stabilizers, n, combined_degrees, seed):
        """Damped data-noise BP surrogate, not ASC memory or relay-BP.

        Includes sampled shared-check pairs, but no claim about ASC hook
        ordering. Settled residuals are classified in the actual CSS
        quotient. Unresolved probes are conservatively counted as failures.
        """
        shots = 48
        neighborhoods = []
        flat, rr, cc = [], [], []
        for r, row in enumerate(checks):
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            neighborhoods.append(ns)
            for c, variable in enumerate(ns):
                flat.append(variable)
                rr.append(r)
                cc.append(c)
        edges = len(flat)
        flat = np.asarray(flat, dtype=np.int32)
        rr = np.asarray(rr, dtype=np.int32)
        cc = np.asarray(cc, dtype=np.int32)
        nr = len(checks)
        maximum = max(map(len, neighborhoods))
        check_edges = np.full((nr, maximum), edges, dtype=np.int32)
        check_variables = np.full((nr, maximum), n, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            check_variables[r, :len(ns)] = ns
        check_edges[rr, cc] = np.arange(edges)
        valid = check_edges != edges

        degrees = np.bincount(flat, minlength=n)
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (n, int(degrees.max())), edges, dtype=np.int32)
        used = np.zeros(n, dtype=np.int32)
        for edge, variable in enumerate(flat):
            variable_edges[variable, used[variable]] = edge
            used[variable] += 1

        channel = np.random.default_rng(seed)
        levels = np.asarray((0.010, 0.018, 0.028))
        load = np.asarray(combined_degrees, dtype=np.float64)
        load /= load.mean()
        rates = np.minimum(
            levels[np.arange(shots) % 3, None] * load[None, :], 0.20)
        error = channel.random((shots, n)) < rates
        opposite = []
        for row in stabilizers:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            opposite.append(ns)
        for shot in range(shots):
            if channel.random() < 0.20:
                ns = opposite[int(channel.integers(len(opposite)))]
                pair = channel.choice(len(ns), 2, replace=False)
                error[shot, ns[int(pair[0])]] ^= True
                error[shot, ns[int(pair[1])]] ^= True

        prior_rates = np.minimum(rates + 0.001, 0.25)
        prior = np.log((1-prior_rates)/prior_rates)
        padded_error = np.zeros((shots, n+1), dtype=bool)
        padded_error[:, :n] = error
        target = (padded_error[:, check_variables].sum(axis=2) & 1)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        span = _dv_pivots(stabilizers)
        failed = iteration_sum = 0

        for iteration in range(1, 33):
            count = len(error)
            padded = np.zeros((count, edges+1), dtype=np.float64)
            padded[:, :edges] = messages
            shaped = padded[:, check_edges]
            signs = np.where(shaped < 0, -1.0, 1.0)
            logarithms = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
            logarithms = np.where(valid[None, :, :], logarithms, 0.0)
            products = np.minimum(np.exp(
                logarithms.sum(axis=2, keepdims=True)-logarithms),
                1-1e-7)
            magnitude = np.log1p(products)-np.log1p(-products)
            outgoing = (
                magnitude * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None])[:, rr, cc]
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            hard = posterior < 0
            padded_hard = np.zeros((count, n+1), dtype=bool)
            padded_hard[:, :n] = hard
            syndrome = padded_hard[:, check_variables].sum(axis=2) & 1
            settled = np.all(syndrome == target, axis=1)
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(error[local] ^ hard[local]):
                    word |= 1 << int(c)
                failed += int(bool(_dv_remainder(word, span)))
                iteration_sum += iteration
            keep = ~settled
            if not np.any(keep):
                return failed, 0, iteration_sum
            messages = np.clip(
                0.35*messages + 0.65*(posterior[:, flat]-outgoing),
                -24, 24)[keep]
            prior, error = prior[keep], error[keep]
            target, target_sign = target[keep], target_sign[keep]
        unresolved = len(error)
        return failed+unresolved, unresolved, iteration_sum+33*unresolved

    for batch in range(3):
        apool, bpool = optimize(0), optimize(1)
        options = []
        for ak, A, ar in apool:
            for bk, B, br in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 578 or not 204 <= k <= 216:
                    continue
                actual_a, actual_b = _dv_components(hx, hz, q)
                assert ar == actual_a and br == actual_b
                if _dv_low4(hx, hz, n) or _dv_low4(hz, hx, n):
                    continue
                key = tuple(v for t in range(4)
                            for v in (max(ak[t], bk[t]), ak[t]+bk[t]))
                key += (ak[4], bk[4], ak[5], bk[5], ak[-1], bk[-1])
                options.append((key, A, B, hx, hz, n, k))
        options.sort(key=lambda item: item[0])

        finalists = []
        seed = 170035006 + 1009*idx
        for key, A, B, hx, hz, n, k in options[:4]:
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx) == 11
            assert max(row.bit_count() for row in hz) == 12
            assert 6*sum(row.bit_count() for row in hx+hz) == 25092
            xc, zc = _dv_columns(hx, n), _dv_columns(hz, n)
            degrees = [x.bit_count()+z.bit_count()
                       for x, z in zip(xc, zc)]
            # Identical random streams across complete alternatives.
            x = probe(hz, hx, n, degrees, seed)
            z = probe(hx, hz, n, degrees, seed+1)
            quality = (max(x[0], z[0]), x[0]+z[0],
                       max(x[1], z[1]), x[1]+z[1],
                       x[2]+z[2]) + key
            finalists.append((quality, A, B, k, x, z))
        if finalists:
            finalists.sort(key=lambda item: item[0])
            quality, A, B, k, xp, zp = finalists[0]
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": design, "mode": mode, "k": k,
                    "relative_geometry": relative,
                    "bp_surrogate": (xp, zp), "screen": quality}

    raise ValueError("no degree-three C17 contract passed verified screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed distinct elements; no additional randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal uniform D7 labels on (6,2)x(6,2) do not suffice for reliable memory: measured s=3.30-3.59, abundant weight-four sector-two logicals, and roughly 10-45 percent failed-shot rates.
# REFUTED: d2s=26 guarantees ceiling-level Dic11 circuit performance: parent slot34 had two Z failures, 38 collective logical flips and s=6.864, while slot26 with the same d2s had zero failures in both bases.
# REFUTED: The Dic4 group/shape jump raises the Dic11 parent's recorded measured average: fitness decreased from 7.1162 to 7.0942, and every measured Dic4 finalist had one or two failures in its worse basis despite k=64 and d2s=14.
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
