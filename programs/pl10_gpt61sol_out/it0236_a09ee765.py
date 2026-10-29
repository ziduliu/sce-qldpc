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
CONCEPT = ("For the square-slope affine matrix group over F11, order 55, shapes (1,3)x(1,3), mixed order-five quotient gaps and stabilizer-aware joint BP selection should improve worse-basis ASC circuit fitness above the F57 parent's 4.8676; measured pL decides.")
GROUP = ("Jump from order-57 C19:C3 to matrices [[a,t],[0,1]] over F11 with square nonzero slopes, order 55. Nontranslation elements have order five rather than three. This genuinely different group's existing rate-0.400 bed scores about 4.9, a competitive measured destination without proving a group advantage. Its fixed order gives n=550.")
SHAPES = ("Retain equal (1,3) sides for cell 0.400: each component has 165 variables and 55 checks, n=550 and guaranteed k>=220. Require verified complete k=220. The parent has broad sector-1 Acol/Bcol weight-eight witnesses and flips across all 228 logicals; mostly 1.0-1.1x data/check enrichment, including Acol/Brow checks, does not justify a localized repair.")
FILLING = ("Twenty prescribed pairs mix positive and inverse order-five quotient gaps, including gaps absent from the shown affine briefs. Fresh independent searches vary rotation directions and every translation parameter. Verified singleton lifts supply exact component actions; require full rank, no Tanner four-cycles and no kernel through four. Quotient obvious affine-automorphism, cell-gauge and column copies within each search. Shortlisted components retain low six-word exposure, boundaries and spectral radius, but complete verified finalists compete first on common-random-number BP probes with degree-weighted data errors, opposite-check pair faults and stabilizer-aware settled residuals. These probes omit ASC ordering and measurement histories and are hypotheses, not circuit fitness. Final matrices require CSS, k=220 and no logical through four. Both row sums are 9+3=12 and six-round CNOT count is 23760; the reduction from 24624 is due to smaller n. Target the parent's ten failed-shot caps and collective flips, not distance alone.")
FAMILY_NAME = "r040_AffSquare11_q55_13x13_inversegap_jointbp_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("square-slope affine F11 requires q=55")
    # (a,t) represents [[a,t],[0,1]]. The square slopes form C5.
    slopes = tuple(pow(3, b, 11) for b in range(5))
    position = {a: b for b, a in enumerate(slopes)}
    els = [(a, t) for a in slopes for t in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 11)

    def mul(g, h):
        a, t = red(g)
        c, u = red(h)
        return (a*c % 11, (t + a*u) % 11)

    def inv(g):
        a, t = red(g)
        ai = pow(a, 9, 11)
        return (ai, -ai*t % 11)

    def idx(g):
        a, t = red(g)
        return 11*position[a] + t

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _f_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _f_low4(rows, stabilizers, width):
    """Exact existence of a nontrivial kernel word of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    cols = _f_columns(rows, width)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if nontrivial(word ^ seen[syndrome]):
                return True
        else:
            seen[syndrome] = word
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            word = (1 << i) | (1 << j)
            if syndrome in seen:
                if nontrivial(word ^ seen[syndrome]):
                    return True
            else:
                seen[syndrome] = word
    return False


def _f_graph(cols, nr):
    """Exact four/six-cycle statistics when four-cycles are absent."""
    adjacency = [0] * nr
    multiplicity = {}
    for column in cols:
        neighbors = []
        while column:
            bit = column & -column
            neighbors.append(bit.bit_length() - 1)
            column ^= bit
        assert len(neighbors) == 3
        for a in range(3):
            for b in range(a):
                i, j = sorted((neighbors[a], neighbors[b]))
                multiplicity[i, j] = multiplicity.get((i, j), 0) + 1
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    four = sum(v * (v - 1) // 2 for v in multiplicity.values())
    if four:
        return four, None, adjacency
    triangles = 0
    for i in range(nr):
        higher = adjacency[i] & ~((1 << (i + 1)) - 1)
        while higher:
            bit = higher & -higher
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ~((1 << (j + 1)) - 1)).bit_count()
            higher ^= bit
    return 0, triangles - len(cols), adjacency


def _f_radius(adjacency):
    """Power-iteration expansion heuristic, not a certified bound."""
    from math import sin, sqrt
    nr = len(adjacency)
    neighbors = []
    for word in adjacency:
        row = []
        while word:
            bit = word & -word
            row.append(bit.bit_length() - 1)
            word ^= bit
        neighbors.append(row)
    assert len({len(row) for row in neighbors}) == 1
    result = 0.0
    for seed in (1, 3, 7):
        vector = [sin((i + 1) * seed) for i in range(nr)]
        mean = sum(vector) / nr
        vector = [v - mean for v in vector]
        norm = sqrt(sum(v * v for v in vector))
        vector = [v / norm for v in vector]
        for step in range(24):
            image = [sum(vector[j] for j in row) for row in neighbors]
            mean = sum(image) / nr
            image = [v - mean for v in image]
            norm = sqrt(sum(v * v for v in image))
            if norm < 1e-12:
                break
            vector = [v / norm for v in image]
        result = max(result, norm)
    return round(result, 8)


def _f_exposure(cols, q):
    """Anchored six-word collisions and small triple boundaries.

    All columns have odd weight three, excluding odd-weight kernels.
    After weight<=4 exclusion, distinct equal-syndrome triples are
    disjoint. Counts below are anchored exposure, not an exact spectrum.
    """
    size = len(cols)
    anchors = tuple(range(0, size, q))
    aset = set(anchors)
    anchored = {}
    boundary = 0
    for a in anchors:
        for b in range(size):
            if b == a or (b in aset and b < a):
                continue
            ab = cols[a] ^ cols[b]
            for c in range(b + 1, size):
                if c == a or (c in aset and c < a):
                    continue
                syndrome = ab ^ cols[c]
                anchored[syndrome] = anchored.get(syndrome, 0) + 1
                weight = syndrome.bit_count()
                if weight <= 3:
                    boundary += (4 - weight) ** 2
    six = -sum(anchored.values())
    get = anchored.get
    for a in range(size - 2):
        x = cols[a]
        for b in range(a + 1, size - 1):
            ab = x ^ cols[b]
            for c in range(b + 1, size):
                six += get(ab ^ cols[c], 0)
    assert six >= 0
    return six, boundary


def _aff_probe(checks, stabilizers, n, seed, shots=96):
    """Complete-code BP surrogate, not an ASC circuit measurement.

    Degree-weighted data errors and random opposite-check pairs probe
    decoding ambiguity. Settled residuals are classified modulo the
    verified stabilizer span. Measurement histories are not simulated.
    """
    import numpy as np

    def neighborhoods(rows):
        result = []
        for word in rows:
            ns = []
            while word:
                bit = word & -word
                ns.append(bit.bit_length() - 1)
                word ^= bit
            result.append(ns)
        return result

    incidence = np.asarray(neighborhoods(checks), dtype=np.int32)
    opposite = np.asarray(neighborhoods(stabilizers), dtype=np.int32)
    nr, weight = incidence.shape
    flat = incidence.ravel()
    edges = len(flat)
    degree = np.bincount(flat, minlength=n)
    exposure = degree + np.bincount(opposite.ravel(), minlength=n)
    assert np.all(degree > 0)

    variable_edges = np.full(
        (n, int(degree.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.006, 0.010, 0.014))
    rates = levels[np.arange(shots) % 3, None] * exposure[None, :] / 6
    error = channel.random((shots, n)) < rates

    # Random pairs under opposite-basis ancillas approximate a fault
    # mechanism; they are not claims about actual ASC hook ordering.
    rows = channel.integers(len(opposite), size=shots)
    first = channel.integers(opposite.shape[1], size=shots)
    second = channel.integers(opposite.shape[1] - 1, size=shots)
    second += second >= first
    for shot in np.flatnonzero(channel.random(shots) < 0.25):
        error[shot, opposite[rows[shot], first[shot]]] ^= True
        error[shot, opposite[rows[shot], second[shot]]] ^= True

    prior_rates = np.minimum(
        rates + 0.5 * np.bincount(opposite.ravel(), minlength=n)[None, :]
        / opposite.size, 0.25)
    prior = np.log((1 - prior_rates) / prior_rates)
    target = (
        error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

    pivots = {}
    for word in stabilizers:
        while word:
            p = word.bit_length() - 1
            if p in pivots:
                word ^= pivots[p]
            else:
                pivots[p] = word
                break

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    failed = work = 0
    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logarithms = np.log(np.clip(
            np.tanh(np.abs(shaped) / 2), 1e-12, 1 - 1e-7))
        products = np.minimum(np.exp(
            logarithms.sum(axis=2, keepdims=True) - logarithms),
            1 - 1e-7)
        magnitude = np.log1p(products) - np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)

        padded = np.zeros((count, edges + 1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior + padded[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        syndrome = (
            hard[:, flat].reshape(count, nr, weight).sum(axis=2) & 1
        ).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local] ^ hard[local]):
                word |= 1 << int(c)
            failed += int(nontrivial(word))
            work += iteration

        keep = ~settled
        if not np.any(keep):
            return failed, 0, work
        messages = np.clip(
            0.30 * messages
            + 0.70 * (posterior[:, flat] - outgoing), -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failed + unresolved, unresolved, work + 33*unresolved


def NEW_CTX(rng, G, idx):
    """Twenty inverse-gap contracts; jointly select fresh complete codes."""
    q = G.q
    assert q == 55 and A_SHAPE == B_SHAPE == (1, 3)
    width = 3 * q
    mask = (1 << q) - 1
    supports = [[3, 3, 3]]
    assert _grid_weights(supports, supports) == 12
    assert _no_dead_lines(supports, supports)
    assert 9 + 3 <= min(12, W_MAX)

    # R1,R2,T12,T13,R3,R4. R3/R4 introduce inverse quotient
    # gaps rather than replaying the earlier four-type affine book.
    a_cards = (
        (0, 0, 4), (0, 1, 4), (1, 4, 5),
        (0, 2, 4), (2, 3, 5),
    )
    b_cards = (
        (1, 1, 5), (0, 4, 4), (1, 3, 5), (2, 2, 4),
    )
    design, batch = idx % 20, idx // 20
    brief_a = a_cards[design // 4]
    brief_b = b_cards[design % 4]
    rotation_gap = {0: 1, 1: 2, 4: 3, 5: 4}

    actions = {"A": {}, "B": {}}
    unit = [[[G.e]]]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], unit)
        _, hz, _ = lifted_matrices(G, unit, [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                term = actions[side][G.idx(g)]
                for r, word in enumerate(term):
                    rows[r] ^= word << (j * q)
        return rows

    def draw(brief):
        number = sum(kind in rotation_gap for kind in brief)
        # Distinct undirected translation directions avoid an obvious
        # source of repeated two-check overlaps. Actual matrices decide.
        directions = iter(rng.sample(range(1, 6), number))
        cells = []
        for kind in brief:
            if kind in rotation_gap:
                gap = 1 + (rotation_gap[kind] - 1 + batch) % 4
                u = (1, next(directions))
                v = (pow(3, gap, 11), rng.randrange(11))
            else:
                exponent = 2 if kind == 2 else 3
                if batch % 2:
                    exponent = 5 - exponent
                u = (3, rng.randrange(11))
                v = (pow(3, exponent, 11), rng.randrange(11))
                if G.mul(u, v) == G.mul(v, u):
                    return None
            cell = [G.e, u, v]
            assert len(set(cell)) == 3
            cells.append(cell)
        return [cells]

    def canonical(grid):
        """Remove obvious within-search images, not arbitrary Tanner copies."""
        normalizations = []
        for cell in grid[0]:
            normalizations.append([
                [G.mul(g, G.inv(anchor)) for g in cell]
                for anchor in cell
            ])
        images = []
        # All maps (a,t)->(a,u*t+v*(1-a)) below are automorphisms.
        for u in range(1, 11):
            for v in range(11):
                normalized = []
                for choices in normalizations:
                    normalized.append(min(
                        tuple(sorted(
                            (a, (u*t + v*(1-a)) % 11)
                            for a, t in choice))
                        for choice in choices
                    ))
                images.append(tuple(sorted(normalized)))
        return min(images)

    def select(side, brief):
        pool, seen = [], set()
        accepted = 0
        for trial in range(1600):
            grid = draw(brief)
            if grid is None:
                continue
            raw = tuple(tuple(sorted(cell)) for cell in grid[0])
            if raw in seen:
                continue
            seen.add(raw)
            rows = component(grid, side)
            cols = _f_columns(rows, width)
            four, cycles, adjacency = _f_graph(cols, q)
            if four or gf2_rank(rows) != q:
                continue
            accepted += 1
            pool.append((cycles, trial, grid, rows, cols, adjacency))
            pool.sort(key=lambda entry: entry[:2])
            del pool[24:]
            if accepted >= 80:
                break
        if not pool:
            raise ValueError("no eligible affine component for inverse-gap card")

        refined, identities = [], set()
        for cycles, trial, grid, rows, cols, adjacency in pool:
            if _f_low4(rows, [], width):
                continue
            identity = canonical(grid)
            if identity in identities:
                continue
            identities.add(identity)
            six, boundary = _f_exposure(cols, q)
            radius = _f_radius(adjacency)
            key = (six, boundary, radius, cycles)
            refined.append((key, trial, grid, rows))
            if len(refined) >= 8:
                break
        refined.sort(key=lambda entry: entry[:2])
        if not refined:
            raise ValueError("affine component short-kernel screen failed")
        return refined[:3]

    # Retry optimization, never repair cells after they are submitted.
    for attempt in range(3):
        aa = select("A", brief_a)
        bb = select("B", brief_b)
        options = []
        for ka, ta, A, ar in aa:
            for kb, tb, B, br in bb:
                joint = tuple(v for a, b in zip(ka, kb)
                              for v in (max(a, b), a + b))
                options.append((joint, ta, tb, A, B, ar, br))
        options.sort(key=lambda entry: entry[:3])

        finalists = []
        for geometry, ta, tb, A, B, ar, br in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 550 or k != 220:
                continue
            actual_a = [
                sum(((row >> (3*j*q)) & mask) << (j*q)
                    for j in range(3))
                for row in hx[:q]
            ]
            actual_b = [row & ((1 << width) - 1) for row in hz[:q]]
            assert actual_a == ar and actual_b == br
            if _f_low4(hx, hz, n) or _f_low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 12
            assert 6 * sum(row.bit_count() for row in hx + hz) == 23760

            # Identical random streams across proposals, separately by
            # basis. Probe outcomes are not recorded as measured pL.
            seed = 550011403 + 1009*idx + 37*attempt
            x = _aff_probe(hz, hx, n, seed)
            z = _aff_probe(hx, hz, n, seed + 1)
            quality = (
                max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                x[2] + z[2],
            ) + geometry + (ta, tb)
            finalists.append((quality, A, B, k, x, z))
            if len(finalists) >= 4:
                break

        if finalists:
            finalists.sort(key=lambda entry: entry[0])
            quality, A, B, k, x, z = finalists[0]
            return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
                    "contract": (brief_a, brief_b, batch),
                    "bp_surrogate": (x, z), "screen": quality}
    raise ValueError("no complete inverse-gap affine code passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell, without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Screen d2s=4 alone implies reliable circuit memory; the parent's measured slots still had worst-basis failures of 120-144/250 shots, with broad sector-1 weight-four witnesses.
# REFUTED: Balanced dihedral trinomials with d2s=8 suffice for rare circuit failures; all five measured parent codes reached failed-shot caps, including slot 0 Z=125/1750, with flips spanning all 232 logical qubits.
# REFUTED: F57 four-cycle exclusion plus component boundary/spectral screening guarantees rare circuit failures; all ten measured parent bases reached failed-shot caps, slot 9 X failed 103/3500, and failures collectively touched all 228 logical qubits despite d2s=8.
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
