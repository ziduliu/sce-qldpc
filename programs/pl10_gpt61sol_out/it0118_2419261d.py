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
CONCEPT = ("On dihedral Z29:Z2(s=28), order 58, shapes (1,3)x(1,3), one pure and two mixed trinomials per component, jointly selected for fewer stabilizer-aware BP probe failures, should improve worse-basis circuit memory at unchanged gate cost; actual circuit measurement can refute this.")
GROUP = ("Frozen by L1: order 58 gives n=580. Filling changes have improved this bed, but the current fitness 4.4188 remains failure-cap limited. Keep the group to isolate whether decoder-aware voltage selection improves its memory circuit.")
SHAPES = ("Frozen equal (1,3) sides give two 174-variable/58-check components, n=580 and guaranteed k>=232, rate 0.400. All five measured parent codes attain k=232. Their sector-1 witnesses span Acol/Bcol blocks broadly, and Acol/Brow check enrichment is mostly 1.0-1.1x; strengthen both components without claiming a unique hotspot.")
FILLING = ("Every cell is a trinomial. Canonical pure-triangle/mixed-step architecture pairs prescribe distinct design constraints; offsets are optimized, not obtained by transforming saved fillings. One pure cell per side lowers unavoidable component four-cycles from 174 in an all-mixed component to 116. Verified lifts screen rank, short dependencies and expansion, preferring no anchored weight-six collisions. Full-code finalists undergo common-random-number BP probes with independent data noise and correlated check-neighbor pairs; settled residuals are classified modulo actual stabilizers, not merely by syndrome convergence. These probes omit the ASC circuit and measurement histories and are only pL heuristics. Final verified k=232 and CSS screens retain cell 0.400. Check weight 12 and 25056 six-round CNOTs are unchanged. Target the parent's collective failures, including slot0 Z=125/1750 and slot2 Z=106/3250, rather than increasing a distance proxy.")
FAMILY_NAME = "r040f_m29r2s28_13x13_pure1_jointbp_v3"

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
    """Prescribe an architecture, then jointly screen new voltage fillings."""
    from itertools import combinations
    import numpy as np

    q, m = G.q, 29
    width = 3 * q
    assert _grid_weights([[3,3,3]], [[3,3,3]]) <= min(12, W_MAX)

    def direction(a):
        a %= m
        return min(a, m - a)

    # A pure triangle contributes three rotation directions; the two
    # mixed cells contribute one each. Keep all five distinct.
    # Quotient obvious multipliers, pure-cell origin changes and exchange
    # of the mixed cells. This is not a full Tanner-isomorphism claim.
    representatives = {}
    for t in range(2, m):
        triangle = (1, t, t - 1)
        occupied = {direction(a) for a in triangle}
        if len(occupied) != 3:
            continue
        free = [d for d in range(1, 15) if d not in occupied]
        for d, e in combinations(free, 2):
            key = min(
                (tuple(sorted(direction(u * a) for a in triangle)),
                 tuple(sorted((direction(u * d), direction(u * e)))))
                for u in range(1, m)
            )
            representatives.setdefault(key, (t, d, e))
    book = [representatives[key] for key in sorted(representatives)]

    # Different unordered architecture pairs, including extra harness
    # batches, rather than repeated draws from one support policy.
    pairs, seen_pairs = [], set()
    for a in range(len(book)):
        b = (7 * a + 11) % len(book)
        key = tuple(sorted((a, b)))
        if key not in seen_pairs:
            seen_pairs.add(key)
            pairs.append((a, b))
    assert len(pairs) >= 20
    pa, pb = pairs[idx % len(pairs)]
    ga, gb = book[pa], book[pb]

    def make_side(geometry, offset):
        t, d, e = geometry
        # A common conjugation fixes the first reflection offset.
        # Only the remaining relative offset is searched.
        return [[
            [(0,0), (1,0), (t,0)],
            [(0,0), (d,0), (0,1)],
            [(0,0), (e,0), (offset,1)],
        ]]

    def transpose_bits(rows, width):
        cols = [0] * width
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                b = row & -row
                cols[b.bit_length()-1] |= bit
                row ^= b
        return cols

    def component(side, grid):
        """Extract the actual left/right component using an identity partner."""
        unit = [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
        cmask = (1 << width) - 1
        rows = [row & cmask for row in rows]
        return rows, transpose_bits(rows, width)

    def pair_stats(cols):
        singles = set(cols)
        low = len(cols) - len(singles)
        pairs = {}
        cycles = 0
        weights = [c.bit_count() for c in cols]
        for i, x in enumerate(cols):
            for j in range(i+1, len(cols)):
                s = x ^ cols[j]
                if s == 0 or s in singles:
                    low += 1
                old = pairs.get(s, 0)
                low += old
                pairs[s] = old + 1
                shared = (weights[i] + weights[j] - s.bit_count()) // 2
                cycles += max(0, shared-1)
        return low, cycles, pairs

    def anchored_spectrum(cols, pairs):
        # A left-lift component has a right-translation symmetry;
        # B's right-lift component has a left-translation symmetry.
        # Every small codeword orbit therefore has a representative
        # containing an identity-coordinate qubit in one of its blocks.
        # Compare all triples with triples containing such an anchor.
        anchors = (0, q, 2*q)
        aset = set(anchors)
        anchored = {}
        n = len(cols)
        for a in anchors:
            for b in range(n):
                if b == a or (b in aset and b < a):
                    continue
                ab = cols[a] ^ cols[b]
                for c in range(b+1, n):
                    if c == a or (c in aset and c < a):
                        continue
                    s = ab ^ cols[c]
                    anchored[s] = anchored.get(s, 0) + 1
        five = sum(count * pairs.get(s, 0) for s, count in anchored.items())
        six = 0
        get = anchored.get
        for a in range(n-2):
            x = cols[a]
            for b in range(a+1, n-1):
                ab = x ^ cols[b]
                for c in range(b+1, n):
                    six += get(ab ^ cols[c], 0)
        # Remove each anchored triple's comparison with itself.
        # If pair screening is clean, distinct matching triples are
        # disjoint, so the remaining matches witness weight six.
        six -= sum(anchored.values())
        return five, six

    def graph_statistics(cols):
        """Statistics of the verified component's check-intersection graph."""
        adjacency = np.zeros((q, q), dtype=np.float64)
        for column in cols:
            neighbors = []
            while column:
                bit = column & -column
                neighbors.append(bit.bit_length() - 1)
                column ^= bit
            for i, j in combinations(neighbors, 2):
                adjacency[i, j] += 1
                adjacency[j, i] += 1
        eigenvalues = np.linalg.eigvalsh(adjacency)
        radius = float(max(abs(eigenvalues[0]), abs(eigenvalues[-2])))
        triangles = int(round(
            np.trace(adjacency @ adjacency @ adjacency) / 6))
        squares = int(round(np.sum(adjacency * adjacency) / 2))
        # Exact for degree-three variables, including overlaps of size two.
        cycles6 = triangles - squares + 2 * width
        assert cycles6 >= 0
        spectrum = tuple(round(float(v), 7) for v in eigenvalues)
        return round(radius, 9), cycles6, spectrum

    def select_components(side, geometry):
        pool, spectra = [], set()
        offsets = list(range(m))
        rng.shuffle(offsets)
        for offset in offsets:
            grid = make_side(geometry, offset)
            rows, cols = component(side, grid)
            low, four, pair_hist = pair_stats(cols)
            if low or four != 2 * q or gf2_rank(rows) != q:
                continue
            radius, cycles6, spectrum = graph_statistics(cols)
            if radius >= 18 - 1e-8 or spectrum in spectra:
                continue
            spectra.add(spectrum)
            pool.append(((radius, cycles6, offset),
                         grid, cols, pair_hist))
        if not pool:
            raise ValueError("architecture has no eligible component")
        pool.sort(key=lambda item: item[0])

        screened = []
        for cheap, grid, cols, pair_hist in pool[:8]:
            five, six = anchored_spectrum(cols, pair_hist)
            assert five == 0
            screened.append(((six, cheap[0], cheap[1]), grid))
        screened.sort(key=lambda item: item[0])
        safe = [item for item in screened if item[0][0] == 0]
        # Prefer the weight-eight-capable pool without treating a missed
        # longer witness as a distance certificate.
        return (safe or screened)[:3]

    def basis(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        return pivots

    def nontrivial(word, pivots):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    def low4(checks, stabilizers, n):
        """Exact logical-existence test through weight four."""
        cols = transpose_bits(checks, n)
        pivots = basis(stabilizers)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            if syndrome in seen:
                if nontrivial(word ^ seen[syndrome], pivots):
                    return True
            else:
                seen[syndrome] = word
        for i, syndrome in enumerate(cols):
            for j in range(i):
                s = syndrome ^ cols[j]
                word = (1 << i) | (1 << j)
                if s in seen:
                    if nontrivial(word ^ seen[s], pivots):
                        return True
                else:
                    seen[s] = word
        return False

    def bp_probe(checks, stabilizers, n, seed, shots):
        """Correlated-data-noise BP surrogate, NOT an ASC memory experiment.

        Common random numbers compare proposals fairly. There is no
        measurement noise or time dimension. Unsettled shots are penalized;
        settled shots fail only for a nontrivial stabilizer-quotient residual.
        """
        check_columns = []
        for row in checks:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length() - 1)
                row ^= bit
            check_columns.append(ns)
        row_weight = len(check_columns[0])
        assert all(len(ns) == row_weight for ns in check_columns)
        incidence = np.asarray(check_columns, dtype=np.int32)
        flat = incidence.ravel()
        nr, edges = len(checks), len(flat)
        degrees = np.bincount(flat, minlength=n)
        neighbors = np.full((n, int(degrees.max())),
                            edges, dtype=np.int32)
        used = np.zeros(n, dtype=np.int32)
        for edge, c in enumerate(flat):
            neighbors[c, used[c]] = edge
            used[c] += 1

        channel_rng = np.random.default_rng(seed)
        levels = np.asarray((0.012, 0.021, 0.030), dtype=np.float64)
        rates = levels[np.arange(shots) % 3, None] * degrees[None, :] / 3
        error = channel_rng.random((shots, n)) < rates

        # A deliberately approximate hook-like fault: one correlated
        # pair among actual neighbors of a sampled check. It is not
        # asserted to reproduce any particular ASC CNOT ordering.
        hook_rows = channel_rng.integers(nr, size=shots)
        hook_first = channel_rng.integers(row_weight, size=shots)
        hook_second = channel_rng.integers(row_weight - 1, size=shots)
        hook_second += hook_second >= hook_first
        hooks = channel_rng.random(shots) < 0.35
        for shot in np.flatnonzero(hooks):
            r = hook_rows[shot]
            error[shot, incidence[r, hook_first[shot]]] ^= True
            error[shot, incidence[r, hook_second[shot]]] ^= True

        prior_rates = np.minimum(
            rates + 0.7 * degrees[None, :] / edges, 0.25)
        prior = np.log((1 - prior_rates) / prior_rates)
        target = (
            error[:, flat].reshape(shots, nr, row_weight).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        active = np.arange(shots)
        pivots = basis(stabilizers)
        failed = unresolved = iteration_sum = 0

        for iteration in range(1, 33):
            count = len(active)
            shaped = messages.reshape(count, nr, row_weight)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logarithms = np.log(np.clip(
                np.tanh(np.abs(shaped) / 2), 1e-12, 1 - 1e-7))
            products = np.minimum(np.exp(
                logarithms.sum(axis=2, keepdims=True) - logarithms),
                1 - 1e-7)
            magnitudes = np.log1p(products) - np.log1p(-products)
            outgoing = (
                magnitudes * signs *
                signs.prod(axis=2, keepdims=True) *
                target_sign[:, :, None]
            ).reshape(count, edges)

            padded = np.zeros((count, edges + 1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, neighbors].sum(axis=2)
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, nr, row_weight).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)

            for local in np.flatnonzero(settled):
                residual = error[active[local]] ^ hard[local]
                word = 0
                for c in np.flatnonzero(residual):
                    word |= 1 << int(c)
                failed += int(nontrivial(word, pivots))
                iteration_sum += iteration

            keep = ~settled
            if not np.any(keep):
                return failed, unresolved, iteration_sum
            messages = np.clip(
                0.25 * messages +
                0.75 * (posterior[:, flat] - outgoing), -24, 24)
            messages = messages[keep]
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            active = active[keep]

        unresolved = len(active)
        failed += unresolved
        iteration_sum += 33 * unresolved
        return failed, unresolved, iteration_sum

    aa = select_components("A", ga)
    bb = select_components("B", gb)
    proposals = []
    probe_seed = 580029003 + 1009 * idx

    def quality(x, z, geometry_cost):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                x[2] + z[2]) + geometry_cost

    for ak, A in aa:
        for bk, B in bb:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 580 or k != 232:
                continue
            geometry_cost = (
                max(ak[0], bk[0]), ak[0] + bk[0],
                max(ak[1], bk[1]), ak[1] + bk[1],
                ak[2] + bk[2],
            )
            x = bp_probe(hz, hx, n, probe_seed, 72)
            z = bp_probe(hx, hz, n, probe_seed + 1, 72)
            key = quality(x, z, geometry_cost)
            proposals.append((key, geometry_cost, x, z,
                              A, B, hx, hz, n, k))
    if not proposals:
        raise ValueError("no verified rate-0.400 proposal")
    proposals.sort(key=lambda item: item[0])

    finalists = []
    for item in proposals[:3]:
        _, cost, old_x, old_z, A, B, hx, hz, n, k = item
        x = bp_probe(hz, hx, n, probe_seed + 100003, 216)
        z = bp_probe(hx, hz, n, probe_seed + 100004, 216)
        x = tuple(a + b for a, b in zip(x, old_x))
        z = tuple(a + b for a, b in zip(z, old_z))
        finalists.append((quality(x, z, cost), A, B, hx, hz, n, k))
    finalists.sort(key=lambda item: item[0])

    for key, A, B, hx, hz, n, k in finalists:
        if low4(hx, hz, n) or low4(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 12
        return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "geometry": (ga, gb), "bp_surrogate": key}
    raise ValueError("all BP finalists contained a logical of weight <=4")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Screen d2s=4 alone implies reliable circuit memory; the parent's measured slots still had worst-basis failures of 120-144/250 shots, with broad sector-1 weight-four witnesses.
# REFUTED: Balanced supports and d2s=8 alone make this bed's circuit failures rare; current-parent slot0 had 125/1750 Z failures affecting all 232 logical qubits, and all five measured codes reached failed-shot caps.
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
