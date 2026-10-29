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
CONCEPT = ("For cyclic C20 in the frozen metacyclic encoding, order 20, shapes (2,5)x(2,5), separating A/B intra-cell directions and reducing four-variable absorbing sets should improve worse-basis circuit memory at unchanged gate count.")
GROUP = ("Keep the exact custom group and order byte-for-byte under L1. This bed already reaches approximately 5.8 in RUN STATE, while the current parent scores 5.5627; this experiment isolates filling effects.")
SHAPES = ("Frozen equal (2,5) components have 100 variables and 40 checks each: n=580, kb=180 and guaranteed rate 9/29=0.310345 in cell 0.300. Parent failures span sector-1 Acol/Bcol blocks with mostly 1.1-1.2x enrichment and affect all 180 logicals; no isolated Acol/Brow hotspot is established.")
FILLING = ("Keep every column at degree three and component row totals seven/eight: maximum CSS check weight eleven and 25200 six-round CNOTs. Twenty canonical geometry cards prescribe disjoint three/four-direction alphabets, plus a one-shared-direction control. Directions have orders 5, 10 or 20, never 2 or 4. Search only non-gauge offsets using verified component lifts, then refine by short kernel multiplicity and translation-anchored four-variable absorbing sets. Coordinate both components rather than optimize labels independently. The parent's five measured codes all reached failure caps, with worse-basis s=5.402-5.782 and widespread logical flips. Require verified CSS and actual rate in cell 0.300, permitting small genuine rank deficits. Direction separation and absorbing-set counts are hypotheses for circuit pL, not distance certificates or scored decoder measurements.")
FAMILY_NAME = "r030_m1r20s1_25x25_segregated_gap_absorbing4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Same metacyclic rule; order resized because 29*21 exceeds the window.
    m, r, s = 1, 20, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# Columns are prescribed term-count pairs, not random support draws.
# Every profile has five columns of weight 3 or 4 and row sums <=8.
_SUPPORT_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,2)),
    ((2,1),(2,1),(1,2),(1,2),(2,2)),
    ((2,1),(1,2),(1,2),(1,2),(3,1)),
    ((3,0),(2,1),(1,2),(1,2),(1,2)),
    ((3,0),(1,2),(1,2),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(2,2)),
    ((3,0),(1,2),(1,2),(1,2),(2,2)),
)
_PROFILE_PAIRS = (
    (0,0),(0,1),(0,2),(1,1),(1,2),(2,2),
    (0,3),(0,4),(1,3),(1,4),(2,3),(2,4),
    (3,3),(3,4),(4,4),(0,5),(1,5),(2,5),
    (0,6),(0,7),
)


def _support_profile(rng, number):
    columns = list(_SUPPORT_PROFILES[number])
    rng.shuffle(columns)
    swap = rng.randrange(2)
    return [[column[i ^ swap] for column in columns] for i in range(2)]


def _component_rows(G, side, grid):
    """Extract the component action from the verified construction.

    The opposite side is a zero 1x1 probe, not a submitted component.
    This preserves the evaluator's left/right conventions exactly.
    """
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _cycle_counts(rows):
    """Exact component Tanner four- and six-cycle counts."""
    size = len(rows)
    intersections = [[0] * size for _ in range(size)]
    counts = [[0] * size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i + 1, size):
            common = rows[i] & rows[j]
            number = common.bit_count()
            intersections[i][j] = common
            counts[i][j] = number
            four += number * (number - 1) // 2
    six = 0
    for i in range(size):
        for j in range(i + 1, size):
            a = counts[i][j]
            if not a:
                continue
            for k in range(j + 1, size):
                b, c = counts[i][k], counts[j][k]
                if b and c:
                    common = (intersections[i][j] & rows[k]).bit_count()
                    # Count distinct variable choices, excluding stars.
                    six += a*b*c - common*(a+b+c) + 2*common
    return four, six


def _short_collision_counts(rows, columns):
    """Cheap component-codeword screen through weight four.

    Pair collisions with overlapping supports can overcount when
    weight-one/two words exist; those are already penalized first.
    """
    syndromes = [0] * columns
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] |= 1 << check
            row ^= bit
    singles = {}
    one = two = 0
    for syndrome in syndromes:
        one += int(syndrome == 0)
        two += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
    pairs = {}
    three = four = 0
    for i in range(columns):
        for j in range(i + 1, columns):
            syndrome = syndromes[i] ^ syndromes[j]
            three += singles.get(syndrome, 0)
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return one, two, three, four


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _design_component(rng, G, side, support, idx):
    """Independent multistart label search; no orbit-copy construction."""
    import math
    positions = [(i,j) for i in range(2) for j in range(5)
                 if support[i][j]]
    selected = None
    selected_quality = None
    # Extra harness slots use different search pressure as well as labels.
    cycle_weight = 200 + 20 * (idx // len(_PROFILE_PAIRS))
    for restart in range(4):
        grid = [[rng.sample(G.elements, support[i][j])
                 for j in range(5)] for i in range(2)]
        rows = _component_rows(G, side, grid)
        cycles = _cycle_counts(rows)
        score = cycle_weight * cycles[0] + cycles[1]
        local_best = _copy_cells(grid)
        local_score = score

        for step in range(180):
            i, j = rng.choice(positions)
            cell = grid[i][j]
            position = rng.randrange(len(cell))
            old = cell[position]
            occupied = {G._key(g) for t,g in enumerate(cell)
                        if t != position}
            replacement = G.pick(rng)
            while G._key(replacement) in occupied:
                replacement = G.pick(rng)
            if G._key(replacement) == G._key(old):
                continue
            cell[position] = replacement
            trial_cycles = _cycle_counts(_component_rows(G, side, grid))
            trial_score = (cycle_weight * trial_cycles[0]
                           + trial_cycles[1])
            temperature = 35.0 * (1.0 - step / 180.0) + 0.3
            delta = trial_score - score
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                score = trial_score
                if score < local_score:
                    local_score = score
                    local_best = _copy_cells(grid)
            else:
                cell[position] = old

        rows = _component_rows(G, side, local_best)
        quality = (
            len(rows) - gf2_rank(rows),
        ) + _short_collision_counts(rows, 5 * G.q) + _cycle_counts(rows)
        if selected_quality is None or quality < selected_quality:
            selected_quality = quality
            selected = local_best
    return selected


def NEW_CTX(rng, G, idx):
    """Coordinated direction cards and verified absorbing-set screening."""
    import math

    q = G.q
    counts = [[2, 2, 1, 1, 1], [1, 1, 2, 2, 2]]
    assert q == 20
    assert _grid_weights(counts, counts) <= min(11, W_MAX)
    assert _no_dead_lines(counts, counts)

    # Four architecture experiments, five inequivalent cards each.
    # Larger harness batches advance through the same deterministic book.
    architectures = ((3, 3, 0), (3, 4, 0),
                     (4, 4, 0), (4, 4, 1))
    architecture = (idx % 20) // 5
    na, nb, shared = architectures[architecture]
    wanted = idx % 5 + 5 * (idx // 20)
    units = (1, 3, 7, 9)
    other = (2, 4, 6, 8)
    multipliers = (1, 3, 7, 9, 11, 13, 17, 19)

    def direction(d):
        d %= q
        return min(d, q - d)

    def draw_geometry(grng, size):
        # Two generator directions on each side. The other directions
        # have order ten or five. Every selected direction is used.
        alphabet = grng.sample(units, 2)
        alphabet += grng.sample(other, size - 2)
        while True:
            top = grng.sample(alphabet, 2)
            bottom = grng.sample(alphabet, 3)
            if len(set(top + bottom)) == size:
                return [[top[0], top[1], 0, 0, 0],
                        [0, 0, bottom[0], bottom[1], bottom[2]]]

    def used_directions(steps):
        return {d for row in steps for d in row if d}

    def geometry_key(pair):
        """Quotient obvious automorphisms, row/column and side exchange.

        This canonicalizes design briefs, not arbitrary Tanner graphs.
        The evaluator remains authoritative for full-code isomorphism.
        """
        images = []
        for u in multipliers:
            sides = []
            for steps in pair:
                columns = [
                    (counts[0][j], direction(u * steps[0][j]),
                     counts[1][j], direction(u * steps[1][j]))
                    for j in range(5)
                ]
                normal = tuple(sorted(columns))
                swapped = tuple(sorted(
                    (c, d, a, b) for a, b, c, d in columns))
                sides.append(min(normal, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    grng = random.Random(2000029 + 101 * architecture)
    book, seen = [], set()
    for trial in range(40000):
        ga = draw_geometry(grng, na)
        gb = draw_geometry(grng, nb)
        if len(used_directions(ga) & used_directions(gb)) != shared:
            continue
        identity = geometry_key((ga, gb))
        if identity in seen:
            continue
        seen.add(identity)
        book.append((ga, gb))
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("inequivalent direction-card book exhausted")
    geometry = book[wanted]

    def labels(steps):
        """Remove column and check-row translation gauges.

        Top-cell anchors and the first bottom-cell anchor are zero.
        The four remaining bottom phases carry the offset freedom.
        """
        grid = [[None] * 5 for _ in range(2)]
        for i in range(2):
            for j in range(5):
                phase = 0 if i == 0 or j == 0 else rng.randrange(q)
                g = (0, phase)
                cell = [g]
                if counts[i][j] == 2:
                    cell.append(G.mul(g, (0, steps[i][j])))
                grid[i][j] = cell
        return grid

    def columns(rows):
        out = [0] * (5 * q)
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def assess(side, grid):
        rows = _component_rows(G, side, grid)
        four, six = _cycle_counts(rows)
        deficit = len(rows) - gf2_rank(rows)
        # Normalize regular-lift orbit changes before annealing.
        energy = 80 * deficit + (80 * four + 0.5 * six) / q
        return energy, rows, deficit, four, six

    def six_word_cost(cols, short):
        """Triple collisions count weight-six words after <=4 exclusion."""
        histogram = {}
        collisions = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    previous = histogram.get(syndrome, 0)
                    collisions += previous
                    histogram[syndrome] = previous + 1
        if not any(short):
            # Distinct colliding triples must then be disjoint;
            # each six-support has ten unordered triple partitions.
            assert collisions % 10 == 0
            return collisions // 10
        # Do not call the collision statistic an exact spectrum here.
        return 10000000 + collisions

    def absorbing_four(cols):
        """Exact translation-anchored four-variable absorbing counts.

        A proper absorbing set gives each variable strictly more
        satisfied than unsatisfied incident checks. Also count a
        one-neighbor-relaxed class as a secondary risk surrogate.

        Every four-support contributes four anchor incidences over all
        q translations. Thus q/4 times the anchored incidence sum is
        the full count, including shorter translation orbits.
        """
        width = len(cols)
        degrees = [c.bit_count() for c in cols]
        severe = proper = relaxed = 0
        exposure = [0] * 5
        for block in range(5):
            anchor = block * q
            ca = cols[anchor]
            da = degrees[anchor]
            candidates = [b for b in range(width) if b != anchor]
            for bi, b in enumerate(candidates):
                ab = ca ^ cols[b]
                cb, db = cols[b], degrees[b]
                for ci in range(bi + 1, len(candidates)):
                    c = candidates[ci]
                    abc = ab ^ cols[c]
                    cc, dc = cols[c], degrees[c]
                    for di in range(ci + 1, len(candidates)):
                        d = candidates[di]
                        syndrome = abc ^ cols[d]
                        boundary = syndrome.bit_count()
                        if boundary == 0 or boundary > 4:
                            continue
                        ua = (syndrome & ca).bit_count()
                        ub = (syndrome & cb).bit_count()
                        uc = (syndrome & cc).bit_count()
                        ud = (syndrome & cols[d]).bit_count()
                        dd = degrees[d]
                        good = (2*ua < da and 2*ub < db and
                                2*uc < dc and 2*ud < dd)
                        if good:
                            proper += 1
                            severe += int(boundary <= 2)
                            exposure[block] += 1
                        if (2*ua <= da + 1 and 2*ub <= db + 1 and
                                2*uc <= dc + 1 and 2*ud <= dd + 1):
                            relaxed += 1
        factor = q // 4
        return (factor * severe, factor * proper,
                max(exposure), factor * relaxed)

    def select(side, steps):
        proposals = []
        positions = [(1, j) for j in range(1, 5)]
        for restart in range(6):
            grid = labels(steps)
            current = assess(side, grid)
            best_grid = _copy_cells(grid)
            best_energy = current[0]
            for step in range(240):
                i, j = rng.choice(positions)
                old = grid[i][j]
                shift = (0, rng.randrange(1, q))
                grid[i][j] = [G.mul(g, shift) for g in old]
                proposed = assess(side, grid)
                delta = proposed[0] - current[0]
                temperature = 18 * (1 - step / 240.0) ** 2 + 0.4
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if current[0] < best_energy:
                        best_energy = current[0]
                        best_grid = _copy_cells(grid)
                else:
                    grid[i][j] = old
                if step in (59, 119, 179, 239):
                    proposals.append(_copy_cells(grid))
            proposals.append(best_grid)

        screened = []
        identities = set()
        for serial, grid in enumerate(proposals):
            identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for row in grid for cell in row)
            if identity in identities:
                continue
            identities.add(identity)
            _, rows, deficit, four, six = assess(side, grid)
            short = _short_collision_counts(rows, 5 * q)
            cheap = (deficit,) + short + (four, six, serial)
            screened.append((cheap, grid, rows))
        screened.sort(key=lambda item: item[0])

        finalists = []
        for cheap, grid, rows in screened[:8]:
            cols = columns(rows)
            short = cheap[1:5]
            kernels = six_word_cost(cols, short)
            traps = absorbing_four(cols)
            key = cheap[:5] + (kernels,) + traps + cheap[5:]
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        if not finalists:
            raise ValueError("offset search produced no candidate")
        return finalists[:4]

    for attempt in range(2):
        aa = select("A", geometry[0])
        bb = select("B", geometry[1])
        options = []
        for ia, (ak, A) in enumerate(aa):
            for ib, (bk, B) in enumerate(bb):
                # Protect the worse component before reducing aggregate
                # exposure. The final serial numbers only break ties.
                key = tuple(v for a, b in zip(ak[:-1], bk[:-1])
                            for v in (max(a, b), a + b))
                options.append((key, ia, ib, A, B))
        options.sort(key=lambda item: item[:3])
        for key, ia, ib, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 11 * n <= 40 * k < 13 * n):
                continue
            assert n == 580
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "k": k, "architecture": architecture,
                    "geometry_card": wanted, "screen": key}
    raise ValueError("no segregated-gap filling remains in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell, without mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Independent component short-cycle optimization suffices for rare circuit failures: all five current-parent measured codes reached failure caps, with worse-basis s=5.402-5.782 and widespread logical flips.
# REFUTED: Exact-k=180 screening retains every optimized support architecture: current-parent slots 7, 9, 11, 13 and others were voided because optimized fillings did not attain the target rank.
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
