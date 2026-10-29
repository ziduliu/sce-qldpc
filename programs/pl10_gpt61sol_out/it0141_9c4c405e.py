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
CONCEPT = ("Generalized quaternion Q16, order 16, shapes (2,5)x(2,6): uniformly degree-three component columns and balanced short-boundary exposure should reduce worse-basis circuit failures relative to the parent's ubiquitous degree-two/four-body vulnerabilities.")
GROUP = ("Jump from C2 x C2 to Q16: x^8=1, y^2=x^4, y*x*y^-1=x^-1. This order-16 quaternion bed is absent from BEDS. The cyclic degree-three Z20 bed scored 5.8, while the related Z17 (2,5)x(2,6) bed scored 4.8; these motivate a noncommuting-label experiment, not a measured quaternion advantage. Binomial ratios use orders eight or four, never the central involution.")
SHAPES = ("Target cell 0.350 instead of repeating the high-failure 0.500 bed. Components (2,5) and (2,6) have 80/96 variables and 32 checks each; n=544 and kb=192 give guaranteed rate 12/34=0.35294. Every component column has degree three. Parent measured codes had k=256, abundant weight-four sector-1 witnesses, and diffuse 1.1-1.2x Acol/Bcol enrichment; replacing both component geometries is better supported than repairing one alleged Brow hotspot. Verified full ranks require k=192.")
FILLING = ("Twenty contracts cross four A and five B relative-ratio architectures. Counts are A=[[2,2,1,1,1],[1,1,2,2,2]] and B=[[2,2,2,1,1,1],[1,1,1,2,2,2]]: A rows 7/8, B rows 9/9, all columns three, HX weights 10/11 and HZ weight twelve. Six-round CNOT count is 23616. Independent multistart optimization uses only verified singleton actions. Finalists prioritize short component words, weight-six multiplicity, small triple boundaries, connectivity and capped sampled protection, then actual CSS short-logical classification. Rank-legal weaker finalists remain available rather than voiding a contract. These surrogates target the parent's 125-164 failures per 250 scored shots and collective logical flips; they do not simulate ASC or certify pL.")
FAMILY_NAME = "r035_q16_25x26_trivalent_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q16 requires q=16")
    # Generalized quaternion: x^8=1, y^2=x^4, y*x*y^-1=x^-1.
    # Canonical encoding (a,b) = x^a*y^b, b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        quotient, b = divmod(int(g[1]), 2)
        return ((int(g[0]) + 4 * quotient) % 8, b)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 4*b*d) % 8,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 8, 0) if b == 0 else ((a + 4) % 8, 1)

    def idx(g):
        a, b = red(g)
        return 8*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _r050_columns(rows, n):
    """Column syndrome signatures of an already verified lifted matrix."""
    columns = [0] * n
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _r050_small_profile(checks, stabilizers, n):
    """Reject logicals of weight <=3; count weight-four fault opportunities.

    A shared stabilizer check makes a pair a possible ancilla-spread pair.
    Two such pairs forming a logical are a two-fault opportunity, without
    asserting that the unknown ASC ordering actually realizes that hook.
    Each quartet is counted once per disjoint pair partition.
    """
    syndrome = _r050_columns(checks, n)
    shared = _r050_columns(stabilizers, n)

    # Row-space membership only. Lifts and reported k use scaffold helpers.
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break

    def in_span(word):
        while word:
            p = word.bit_length() - 1
            pivot = basis.get(p)
            if pivot is None:
                return False
            word ^= pivot
        return True

    bits = [1 << i for i in range(n)]
    singles = {}
    for i, s in enumerate(syndrome):
        if s == 0 and not in_span(bits[i]):
            return None
        singles.setdefault(s, []).append(bits[i])

    pairs = {}
    hook_count = 0
    logical_four = 0
    hits = [0] * n
    for i in range(n):
        si, bi = syndrome[i], bits[i]
        for j in range(i + 1, n):
            s = si ^ syndrome[j]
            mask = bi | bits[j]
            if s == 0 and not in_span(mask):
                return None
            for third in singles.get(s, ()):
                if not (mask & third) and not in_span(mask | third):
                    return None

            hook = bool(shared[i] & shared[j])
            entry = (mask, hook)
            old = pairs.get(s)
            if old is None:
                pairs[s] = entry
                continue

            bucket = old if isinstance(old, list) else (old,)
            for other_mask, other_hook in bucket:
                if mask & other_mask:
                    continue
                quartet = mask | other_mask
                if in_span(quartet):
                    continue
                logical_four += 1
                if hook and other_hook:
                    hook_count += 1
                    word = quartet
                    while word:
                        bit = word & -word
                        hits[bit.bit_length() - 1] += 1
                        word ^= bit
            if isinstance(old, list):
                old.append(entry)
            else:
                pairs[s] = [old, entry]

    return hook_count, logical_four, max(hits)


# Indices refer to doubleton positions in component row-major order.
# Mixed ratios lie outside <x> and have order four. Other ratios are
# rotations of orders eight or four, excluding x^4.
_Q16_A_MIXED = (
    (),
    (0, 2),
    (2, 3, 4),
    (0, 1, 2, 3, 4),
)
_Q16_B_MIXED = (
    (),
    (0,),
    (0, 3),
    (0, 1, 3),
    (0, 3, 4, 5),
)


def _r050_component(G, side, grid):
    """Use a zero opposite-side probe in the verified construction."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _r050_component_score(rows, width):
    """Classical short-word counts, not quantum distance or circuit pL.

    With nonzero distinct columns, each triple and quartet contributes
    three pair partitions. Earlier penalties dominate when duplicates
    would make those normalizations inexact.
    """
    cols = _r050_columns(rows, width)
    singles = {}
    one = two = 0
    for s in cols:
        one += int(s == 0)
        two += singles.get(s, 0)
        singles[s] = singles.get(s, 0) + 1

    pairs = {}
    three = four = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            s = x ^ y
            three += singles.get(s, 0)
            four += pairs.get(s, 0)
            pairs[s] = pairs.get(s, 0) + 1

    cycles = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return one + two, three // 3, four // 3, cycles


def _r050_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _q16_actions(G):
    """Cache singleton blocks obtained from the verified construction."""
    mask = (1 << G.q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in hx)
        actions["B"][G._key(g)] = tuple(row & mask for row in hz)
    return actions


def _q16_component(grid, side, actions, q):
    """XOR verified singleton blocks; no independent lift convention."""
    rows = []
    for row in grid:
        out = [0] * q
        for j, cell in enumerate(row):
            for g in cell:
                block = actions[side][tuple(g)]
                for t, bits in enumerate(block):
                    out[t] ^= bits << (j * q)
        rows.extend(out)
    return rows


def _q16_blueprint(side, policy):
    if side == "A":
        counts = [[2,2,1,1,1], [1,1,2,2,2]]
        prescription = _Q16_A_MIXED[policy]
    else:
        counts = [[2,2,2,1,1,1], [1,1,1,2,2,2]]
        prescription = _Q16_B_MIXED[policy]
    doubles = [(i, j) for i in range(2)
               for j in range(len(counts[0])) if counts[i][j] == 2]
    return counts, {doubles[t] for t in prescription}


def _q16_cell(rng, G, counts, mixed, grid, i, j):
    """Preserve the ratio family and distinct inverse classes per row."""
    g = G.pick(rng)
    if counts[i][j] == 1:
        return [g]

    occupied = set()
    for h, cell in enumerate(grid[i]):
        if h == j or len(cell) != 2:
            continue
        d = G.mul(G.inv(cell[0]), cell[1])
        occupied.add(("M", d[0] % 4) if d[1]
                     else ("R", min(d[0], 8 - d[0])))

    if (i, j) in mixed:
        choices = [a for a in range(4) if ("M", a) not in occupied]
        a = rng.choice(choices) + 4 * rng.randrange(2)
        d = (a, 1)
    else:
        number = sum(counts[i][h] == 2 and (i, h) not in mixed
                     for h in range(len(counts[0])))
        palette = (1, 3) if number <= 2 else (1, 2, 3)
        choices = [a for a in palette if ("R", a) not in occupied]
        a = rng.choice(choices)
        d = ((a if rng.randrange(2) else -a) % 8, 0)
    return [g, G.mul(g, d)]


def _q16_cycles(rows, width):
    """Exact component four/six cycles and check-graph connectivity."""
    columns = _r050_columns(rows, width)
    size = len(rows)
    overlap = [[0] * size for _ in rows]
    adjacency = [0] * size
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            common = (row & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = common
            four += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    six = 0
    for i in range(size):
        for j in range(i):
            if not overlap[i][j]:
                continue
            common = adjacency[i] & adjacency[j] & ((1 << j) - 1)
            while common:
                bit = common & -common
                h = bit.bit_length() - 1
                six += overlap[i][j] * overlap[i][h] * overlap[j][h]
                common ^= bit

    # Every submitted component variable has exactly three neighbors.
    for col in columns:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length() - 1)
            col ^= bit
        assert len(neighbors) == 3
        a, b, c = neighbors
        six -= overlap[a][b] + overlap[a][c] + overlap[b][c] - 2

    seen = 1
    pending = 1
    while pending:
        bit = pending & -pending
        pending ^= bit
        i = bit.bit_length() - 1
        new = adjacency[i] & ~seen
        seen |= new
        pending |= new
    return four, six, seen.bit_count() == size


def _q16_triples(columns, exact):
    """Triple-boundary exposure and weight-six collision multiplicity.

    If component weights <=4 are absent, equal triple syndromes have
    disjoint supports and each weight-six word has ten partitions.
    Otherwise the collision count is only a lower-priority surrogate.
    """
    frequencies = {}
    collisions = boundary = 0
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                weight = syndrome.bit_count()
                if weight <= 3:
                    boundary += (5 - weight) ** 2
                collisions += frequencies.get(syndrome, 0)
                frequencies[syndrome] = frequencies.get(syndrome, 0) + 1
    if exact:
        assert collisions % 10 == 0
        return collisions // 10, boundary
    return collisions, boundary


def _q16_kernel_sample(rng, columns, trials):
    """Sample real component words; returned minima are upper bounds."""
    order = list(range(len(columns)))
    best = len(columns) + 1
    witnesses = set()
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if syndrome == 0:
                basis.append(word)
                best = min(best, word.bit_count())
                if word.bit_count() <= 10:
                    witnesses.add(word)

        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposed = word ^ other
                    w = proposed.bit_count()
                    best = min(best, w)
                    if w <= 10:
                        witnesses.add(proposed)
                    if w < weight:
                        word, weight = proposed, w
                basis[i] = word
    return min(best, 10), len(witnesses)


def _q16_component_pool(rng, G, side, policy, batch, actions):
    """Optimize relative labels, not automorphism images of saved codes."""
    import math

    counts, mixed = _q16_blueprint(side, policy)
    width = len(counts[0]) * G.q
    positions = [(i, j) for i in range(2)
                 for j in range(len(counts[0]))]
    bank = []
    signatures = set()

    def remember(grid):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in signatures:
            return
        signatures.add(signature)
        rows = _q16_component(grid, side, actions, G.q)
        if gf2_rank(rows) != 2 * G.q:
            return
        short = _r050_component_score(rows, width)
        four, six, connected = _q16_cycles(rows, width)
        key = short[:3] + (not connected, four, six)
        bank.append((key, _r050_copy(grid), rows))
        bank.sort(key=lambda item: item[0])
        del bank[10:]

    for restart in range(6):
        grid = [[[] for _ in counts[0]] for _ in range(2)]
        for i, j in positions:
            grid[i][j] = _q16_cell(
                rng, G, counts, mixed, grid, i, j)
        remember(grid)
        rows = _q16_component(grid, side, actions, G.q)
        four, six, _ = _q16_cycles(rows, width)
        cost = (60 * four + six) / G.q
        local_grid, local_cost = _r050_copy(grid), cost
        steps = 200 + 20 * (batch % 3)

        for step in range(steps):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = _q16_cell(
                rng, G, counts, mixed, grid, i, j)
            rows = _q16_component(grid, side, actions, G.q)
            c4, c6, _ = _q16_cycles(rows, width)
            proposed = (60 * c4 + c6) / G.q
            temperature = 18 * (1 - step / steps) + 0.15
            delta = proposed - cost
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                cost = proposed
                if cost < local_cost:
                    local_grid, local_cost = _r050_copy(grid), cost
            else:
                grid[i][j] = old
        remember(local_grid)

    finalists = []
    for coarse, grid, rows in bank[:4]:
        columns = _r050_columns(rows, width)
        exact = not any(coarse[:3])
        six_words, boundary = _q16_triples(columns, exact)
        protection, witnesses = _q16_kernel_sample(
            rng, columns, 24 + 8 * (batch % 3))
        key = (coarse[0], coarse[1], coarse[2], coarse[3],
               six_words, boundary, -protection, witnesses,
               coarse[4], coarse[5])
        finalists.append((key, grid, rows))
    finalists.sort(key=lambda item: item[0])
    if not finalists:
        raise ValueError("no full-rank quaternion component")
    return finalists[:3]


def NEW_CTX(rng, G, idx):
    """Twenty quaternion contracts, with verified final CSS pairing."""
    design, batch = idx % 20, idx // 20
    pa, pb = divmod(design, 5)
    sa, _ = _q16_blueprint("A", pa)
    sb, _ = _q16_blueprint("B", pb)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)

    actions = _q16_actions(G)
    ap = _q16_component_pool(rng, G, "A", pa, batch, actions)
    bp = _q16_component_pool(rng, G, "B", pb, batch, actions)
    options = []
    for ai, (ak, a, ar) in enumerate(ap):
        for bi, (bk, b, br) in enumerate(bp):
            # Sector-1 component words repeat over six B columns or
            # five A columns. This weighting is only an exposure proxy.
            coarse = tuple(v for t in range(6)
                           for v in (max(6 * ak[t], 5 * bk[t]),
                                     6 * ak[t] + 5 * bk[t]))
            coarse += (max(ak[6], bk[6]), ak[7] + bk[7],
                       ak[8] + bk[8], ak[9] + bk[9])
            options.append((coarse, ai, bi, a, b, ar, br))
    options.sort(key=lambda item: item[:3])

    best = None
    for coarse, ai, bi, a, b, ar, br in options[:3]:
        # Cached actions must screen exactly the final verified components.
        assert ar == _r050_component(G, "A", a)
        assert br == _r050_component(G, "B", b)
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 192 or not (13 * n <= 40 * k < 15 * n):
            continue
        assert n == 544
        assert max(row.bit_count() for row in hx + hz) <= W_MAX

        pz = _r050_small_profile(hx, hz, n)
        px = _r050_small_profile(hz, hx, n)
        if px is None or pz is None:
            score = (1,) + coarse
        else:
            score = (0, max(px[1], pz[1]), px[1] + pz[1],
                     max(px[0], pz[0]), px[0] + pz[0]) + coarse
        if best is None or score < best[0]:
            best = (score, a, b, hx, hz, k)

    if best is None:
        raise ValueError("no verified quaternion pairing in cell 0.350")
    score, a, b, hx, hz, k = best
    assert css_ok(hx, hz)
    assert sum(row.bit_count() for row in hx + hz) * 6 == 23616
    return {"idx": idx, "design": (pa, pb, batch), "k": k,
            "cells": {"A": a, "B": b}, "screen": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete pre-screened cell for either new shape."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap compliance with rolled labels is sufficient for low circuit failure in the q14 (6,2)x(6,2) bed; measured s=3.30-3.59, basis failures up to 113/250, and numerous weight-four sector-2 logicals contradict it.
# REFUTED: The systematic [1,3,3,3] A row plus degree-two B hook screen sufficiently suppresses collective circuit failures: fitness was 3.1012, measured codes failed 107-179/250 shots, all 213 logicals were affected, and weight-four witnesses remained abundant.
# REFUTED: Twenty parent structural plans guarantee twenty inequivalent codes: only 14 distinct fillings survived across 60 slots, with numerous isomorphic duplicates.
# REFUTED: Balanced degree-two/three columns and three-row C2 x C2 geometry sufficiently suppress collective circuit failures: fitness remained 3.1707, measured worse-basis failures were 147-164/250, all 256 logicals were affected, and every measured code retained abundant weight-four sector-1 witnesses.
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
