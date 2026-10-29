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
CONCEPT = ("For C5 semidirect C8 with action x->x^2, order 40, shapes (1,2)x(2,6), involution-free noncommuting triangles and joint full-code BP screening should improve worse-basis circuit reliability beyond the parent's 6.1046 without increasing extraction load.")
GROUP = ("Jump from Q8 x C5 to C5:C8(s=2). Since 2^8=1 mod 5, the action defines an associative semidirect product. The new group has centre order two and ten conjugacy classes, unlike the parent's centre-ten/25-class group. No matching order-40 bed appears in BEDS. Its available relative orders 4,5,8,10 motivate a new geometry test; neither those orders nor related nonabelian beds establish a circuit advantage.")
SHAPES = ("Keep A=(1,2), B=(2,6) for cell 0.300. Components have 80 variables/40 checks and 240 variables/80 checks; n=560 and kb=160 give guaranteed rate 2/7. Complete verified ranks require 0.275<=k/n<0.325. Parent failures collectively affected all 160 logicals and diffusely enriched Acol/Bcol blocks, usually 1.0-1.2x. Acol/Brow check enrichment was near one, so no isolated Brow repair or transferable worst-logical index is assumed.")
FILLING = ("Twenty indexed contracts cross four balanced B support architectures with five noncommuting A triangle spectra. All batches retain component column weight three, A row total six and B row totals nine: HX/HZ weights are 9/12 and six-round extraction uses 24480 CNOTs, unchanged from the parent's first measured slots. A deterministic joint relative-support book removes obvious automorphism/translation/protograph copies, not arbitrary Tanner equivalence. Search verified singleton actions, prefer connectivity, ranks, exact component exclusion through four and through six on A, capped sampled witnesses and anchored boundaries. Then compare complete verified CSS codes using common-noise sum-product probes with opposite-check correlated pairs and stabilizer-quotient logical classification. These probes omit ASC scheduling and measurement histories and are not measured pL. Parent slots1/5 both had k=160,d2s=10,w=12,depth15 but worse-basis s=6.321/5.808; joint decoder screening replaces a purely component-based ordering.")
FAMILY_NAME = "r030_C5sdC8_s2_12x26_indexed_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect C8 requires order 40")
    # x^5=y^8=1, y*x*y^-1=x^2; elements are x^a*y^b.
    # The action has order four; y^4 generates the central C2.
    els = [(a, b) for b in range(8) for a in range(5)]
    action = (1, 2, 4, 3, 1, 2, 4, 3)

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 8)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 5, (b+d) % 8)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 8
        return ((-action[bi]*a) % 5, bi)

    def idx(g):
        a, b = red(g)
        return 5*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


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


def _retired_c20_ctx(rng, G, idx):
    """Inactive parent experiment; the Q8 x C5 builder below replaces it."""
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


def _retired_quaternion_ctx(rng, G, idx):
    """Inactive historical quaternion search; new slots never call it."""
    import math

    q = G.q
    assert q == 40
    design, mode = idx % 20, (idx // 20) % 3
    architecture, policy = divmod(design, 5)

    # Baseline row totals are (9,9), with four different allocations.
    profiles = (
        ((2,1),(2,1),(2,1),(1,2),(1,2),(1,2)),
        ((3,0),(0,3),(2,1),(2,1),(1,2),(1,2)),
        ((1,1),(2,2),(2,1),(2,1),(1,2),(1,2)),
        ((3,0),(3,0),(0,3),(0,3),(2,1),(1,2)),
    )
    sa = [[3, 4 if mode == 2 else 3]]
    sb = [[column[i] for column in profiles[architecture]]
          for i in range(2)]
    if mode:
        for i in range(1 if mode == 1 else 2):
            sites = [j for j in range(6) if sb[i][j] == 2]
            sb[i][sites[(policy + i) % len(sites)]] -= 1

    wx = sum(sa[0]) + max(sum(row[j] for row in sb)
                          for j in range(6))
    wz = max(map(sum, sb)) + max(sa[0])
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    elements = list(G.elements)

    def order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("element order exceeds group order")

    orders = {g: order(g) for g in elements}
    palettes = {t: [g for g in elements if orders[g] == t]
                for t in (4, 5, 10, 20)}
    assert all(palettes.values())

    def relative_order(g, h):
        return orders[G.mul(G.inv(g), h)]

    # Two genuinely different internal triangle spectra.
    spectra = ((20,20,20), (4,20,20))
    triangles = [[], []]
    usable = palettes[4] + palettes[20]
    for u in usable:
        for v in usable:
            if G.mul(u, v) == G.mul(v, u):
                continue
            spectrum = tuple(sorted(
                (orders[u], orders[v], relative_order(u, v))))
            if spectrum in spectra:
                triangles[spectra.index(spectrum)].append((u, v))
    assert all(triangles)

    a_types = ((0,0), (0,1), (1,1), (0,0), (0,1))[policy]
    b_orders = ((20,), (20,), (20,10),
                (20,20,5), (20,20,4))[policy]

    # Cache only singleton blocks extracted from lifted_matrices.
    # Assembly below is an XOR of these verified actions, not a new lift.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][g] = tuple(row & mask for row in hx)
        actions[1][g] = tuple(row & mask for row in hz)

    unit_cache = {}

    def unit_cell(normalized):
        signature = tuple(sorted(normalized))
        if signature not in unit_cache:
            rows = [0] * q
            for g in normalized:
                for r, bits in enumerate(actions[0][g]):
                    rows[r] ^= bits
            unit_cache[signature] = gf2_rank(rows) == q
        return unit_cache[signature]

    def draw_cell(side, i, j):
        count = (sa if side == 0 else sb)[i][j]
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        if side == 1 and count == 2:
            t = b_orders[(j + 2*i + mode) % len(b_orders)]
            base = G.pick(rng)
            return [base, G.mul(base, rng.choice(palettes[t]))]

        kind = (a_types[j] if side == 0 else
                (policy + i + j + mode) % 2)
        for _ in range(96):
            u, v = rng.choice(triangles[kind])
            normalized = [G.e, u, v]
            if count == 4:
                extensions = [
                    w for w in elements
                    if all(relative_order(t, w) in (4,5,10,20)
                           for t in normalized)
                ]
                normalized.append(rng.choice(extensions))
            isolated = (side == 1 and
                        sum(sb[r][j] > 0 for r in range(2)) == 1)
            if isolated and not unit_cell(normalized):
                continue
            # Fix single-row A whole-cell gauges. B offsets retain
            # inter-row holonomy and are searched independently.
            base = G.e if side == 0 else G.pick(rng)
            cell = [G.mul(base, g) for g in normalized]
            assert len(set(cell)) == count
            return cell
        raise ValueError("no invertible isolated quaternion trinomial")

    def rows_of(grid, side):
        result = []
        for grid_row in grid:
            rows = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        rows[r] ^= bits << (j*q)
            result.extend(rows)
        return result

    def columns(rows, width):
        cols = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return cols

    def coarse(grid, side):
        rows = rows_of(grid, side)
        four, six = _cycle_counts(rows)
        cost = (len(rows) - gf2_rank(rows), four, six)
        energy = 120*cost[0] + (80*four + 0.25*six)/q
        return cost, energy, rows

    def boundary(cols):
        """Normalized anchored triple boundaries; a BP-risk surrogate."""
        loads = []
        denominator = max(1, (len(cols)-1)*(len(cols)-2)//2)
        for anchor in range(0, len(cols), q):
            load = 0
            for i in range(len(cols)):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ cols[i]
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight < 5:
                        load += (5-weight)**2
            loads.append(load / denominator)
        return max(loads), sum(loads)/len(loads)

    def kernel_probe(cols):
        """Sample true kernel words; no distance lower-bound claim."""
        permutation = list(range(len(cols)))
        minimum = len(cols) + 1
        vulnerable = set()

        def inspect(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    vulnerable.add(word)

        for _ in range(28):
            rng.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    inspect(word)

            # Elementary basis reductions preserve kernel membership.
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposal = word ^ other
                    inspect(proposal)
                    if proposal and proposal.bit_count() < weight:
                        word, weight = proposal, proposal.bit_count()
                basis[i] = word
            if minimum <= 4:
                break

        block_mask = (1 << q) - 1
        loads = [sum(bool((word >> (j*q)) & block_mask)
                     for word in vulnerable)
                 for j in range(len(cols)//q)]
        return minimum, len(vulnerable), max(loads, default=0)

    def select(side):
        counts = sa if side == 0 else sb
        width = len(counts[0])*q
        positions = [(i,j) for i, row in enumerate(counts)
                     for j, count in enumerate(row) if count]
        pool, seen = [], set()

        def retain(grid, result, serial):
            signature = tuple(tuple(sorted(cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            cost, _, rows = result
            pool.append((cost, serial, _copy_cells(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(4):
            grid = [[draw_cell(side, i, j)
                     for j in range(len(row))]
                    for i, row in enumerate(counts)]
            current = coarse(grid, side)
            best_grid, best = _copy_cells(grid), current
            retain(grid, current, restart*1000)

            for step in range(160):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed = coarse(grid, side)
                delta = proposed[1] - current[1]
                temperature = 18*(0.976**step) + 0.2
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current = proposed
                    if current[0] < best[0]:
                        best_grid, best = _copy_cells(grid), current
                    if step % 40 == 39:
                        retain(grid, current, restart*1000 + step + 1)
                else:
                    grid[i][j] = old
            retain(best_grid, best, restart*1000 + 999)

        finalists = []
        for cost, serial, grid, rows in pool[:5]:
            cols = columns(rows, width)
            short = _short_collision_counts(rows, width)
            distance, low, peak = kernel_probe(cols)
            bpeak, bmean = boundary(cols)
            key = (int(any(short)), int(distance < 10),
                   bpeak, bmean, low, peak,
                   cost[1], cost[2], cost[0],
                   -min(distance, 12), serial)
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        return finalists

    def connected(hx, hz, n):
        parent = list(range(n))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in hx + hz:
            if not row:
                return False
            first = row & -row
            a = first.bit_length() - 1
            row ^= first
            while row:
                bit = row & -row
                b = bit.bit_length() - 1
                aa, bb = root(a), root(b)
                if aa != bb:
                    parent[bb] = aa
                row ^= bit
        return len({root(i) for i in range(n)}) == 1

    def short_logical(hx, hz, n):
        """Exact non-stabilizer logical existence through weight four."""
        for checks, stabilizers in ((hx, hz), (hz, hx)):
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

            cols = columns(checks, n)
            seen = {0: 0}
            for i, syndrome in enumerate(cols):
                word = 1 << i
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
            for i, syndrome in enumerate(cols):
                for j in range(i):
                    value = syndrome ^ cols[j]
                    word = (1 << i) | (1 << j)
                    previous = seen.get(value)
                    if previous is None:
                        seen[value] = word
                    elif nontrivial(word ^ previous):
                        return True
        return False

    fallback = None
    for batch in range(2):
        aa, bb = select(0), select(1)
        options = []
        for ia, (ak, A) in enumerate(aa):
            for ib, (bk, B) in enumerate(bb):
                # Worst-component risk precedes aggregate exposure.
                key = tuple(v for a, b in zip(ak[:-1], bk[:-1])
                            for v in (max(a,b), a+b))
                options.append((key, ia, ib, A, B))
        options.sort(key=lambda item: item[:3])

        examined = 0
        for key, ia, ib, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 560 and k > 0 and
                    11*n <= 40*k < 13*n):
                continue
            if not connected(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
            expected_cnots = (24480, 23760, 24960)[mode]
            assert 6*sum(row.bit_count() for row in hx + hz) == expected_cnots
            # Independently confirm cached assembly against the helper.
            assert rows_of(A, 0) == _component_rows(G, "A", A)
            assert rows_of(B, 1) == _component_rows(G, "B", B)

            candidate = {
                "idx": idx, "cells": {"A": A, "B": B},
                "k": k, "architecture": architecture,
                "policy": policy, "mode": mode, "screen": key,
                "small_logical_safe": False,
            }
            if fallback is None:
                fallback = candidate
            if not short_logical(hx, hz, n):
                candidate["small_logical_safe"] = True
                return candidate
            examined += 1
            if examined >= 6:
                break

    # A legal target-band code is preferable to voiding the slot solely
    # because a protection preference failed. The evaluator still screens it.
    if fallback is not None:
        return fallback
    raise ValueError("no connected Q8 x C5 filling retained cell 0.300")


def _sc8_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _sc8_logical4(checks, stabilizers, n):
    """Exact logical existence through four on the verified CSS matrices."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    # Linear representatives modulo the stabilizer row space.
    residues = [1 << c for c in range(n)]
    for p in sorted(pivots):
        tail, value = pivots[p] ^ (1 << p), 0
        while tail:
            bit = tail & -tail
            value ^= residues[bit.bit_length()-1]
            tail ^= bit
        residues[p] = value

    cols = _sc8_columns(checks, n)
    seen = {0: 0}
    for c, syndrome in enumerate(cols):
        value = residues[c]
        old = seen.get(syndrome)
        if old is not None and old != value:
            return True
        seen[syndrome] = value
    for c, syndrome in enumerate(cols):
        for d in range(c):
            value = residues[c] ^ residues[d]
            pair = syndrome ^ cols[d]
            old = seen.get(pair)
            if old is not None and old != value:
                return True
            seen[pair] = value
    return False


def _sc8_bp(checks, stabilizers, n, seed):
    """Common-noise decoder surrogate, NOT the scored ASC experiment.

    Shared opposite checks suggest possible correlated data pairs.
    No pair is asserted to be an ASC-certified hook. Measurement faults,
    extraction ordering and relay/LSD stages are absent from this probe.
    """
    import numpy as np

    def neighborhoods(rows):
        result = []
        for row in rows:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            result.append(ns)
        return result

    ns, hooks = neighborhoods(checks), neighborhoods(stabilizers)
    weight = len(ns[0])
    assert all(len(row) == weight for row in ns)
    incidence = np.asarray(ns, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(ns), len(flat)
    degrees = np.bincount(flat, minlength=n)
    assert np.all(degrees > 0)
    variable_edges = np.full(
        (n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    shots = 48
    channel = np.random.default_rng(seed)
    levels = np.asarray((0.010, 0.020, 0.035))
    rates = levels[np.arange(shots) % 3, None]*degrees[None, :]/3
    error = channel.random((shots, n)) < rates
    for shot in range(shots):
        if channel.random() < 0.4:
            row = hooks[int(channel.integers(len(hooks)))]
            a = int(channel.integers(len(row)))
            b = int(channel.integers(len(row)-1))
            b += b >= a
            error[shot, row[a]] ^= True
            error[shot, row[b]] ^= True

    prior_rates = np.minimum(rates+0.8/n, 0.25)
    prior = np.log((1-prior_rates)/prior_rates)
    target = (error[:, flat].reshape(shots, nr, weight)
              .sum(axis=2) & 1).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(word):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    failures = effort = 0
    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logs = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
        products = np.minimum(
            np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-7)
        magnitude = np.log1p(products)-np.log1p(-products)
        outgoing = (
            magnitude*signs*signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]).reshape(count, edges)
        padded = np.zeros((count, edges+1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior+padded[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        syndrome = (hard[:, flat].reshape(count, nr, weight)
                    .sum(axis=2) & 1).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local] ^ hard[local]):
                word |= 1 << int(c)
            failures += int(nontrivial(word))
            effort += iteration
        keep = ~settled
        if not np.any(keep):
            return failures, 0, effort
        messages = np.clip(
            0.3*messages+0.7*(posterior[:, flat]-outgoing),
            -24, 24)[keep]
        prior, target = prior[keep], target[keep]
        target_sign, error = target_sign[keep], error[keep]
    unresolved = len(error)
    return failures+unresolved, unresolved, effort+33*unresolved


def NEW_CTX(rng, G, idx):
    """Indexed C5:C8 geometries, verified actions and joint decoder selection."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 40 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    assert W_MAX >= 12
    elements = list(G.elements)
    multiplication = [[G.idx(G.mul(g, h)) for h in elements]
                      for g in elements]
    inverses = [G.idx(G.inv(g)) for g in elements]
    identity = G.idx(G.e)
    orders = []
    for g in range(q):
        power = identity
        for order in range(1, q+1):
            power = multiplication[power][g]
            if power == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid element order")
    palettes = {o: [g for g in range(q) if orders[g] == o]
                for o in (4, 5, 8, 10)}
    assert all(palettes.values())

    # All singleton actions come from the authoritative scaffold helper.
    mask = (1 << q)-1
    actions = {"A": [], "B": []}
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"].append(tuple(row & mask for row in hx))
        actions["B"].append(tuple(row & mask for row in hz))

    def rows_of(grid, side):
        result = []
        for row in grid:
            lifted = [0]*q
            for j, cell in enumerate(row):
                for g in cell:
                    for r, bits in enumerate(actions[side][G.idx(g)]):
                        lifted[r] ^= bits << (j*q)
            result.extend(lifted)
        return result

    spectra = ((5, 8, 8), (8, 8, 10),
               (4, 8, 8), (4, 4, 10))
    triangles = {s: [] for s in spectra}
    unit_triangles = {s: [] for s in spectra}
    for u, v in combinations(range(q), 2):
        if identity in (u, v):
            continue
        if multiplication[u][v] == multiplication[v][u]:
            continue
        relative = multiplication[inverses[u]][v]
        spectrum = tuple(sorted((orders[u], orders[v], orders[relative])))
        if spectrum not in triangles:
            continue
        template = (identity, u, v)
        triangles[spectrum].append(template)
        probe = [actions["A"][identity][r]
                 ^ actions["A"][u][r] ^ actions["A"][v][r]
                 for r in range(q)]
        if gf2_rank(probe) == q:
            unit_triangles[spectrum].append(template)
    assert all(triangles.values())
    # Prefer units only where they actually exist; never exhaust an
    # unrealizable hard unit palette as the earlier order-39 model did.
    books = {s: unit_triangles[s] or triangles[s] for s in spectra}
    a_cards = (
        (spectra[0], spectra[0]),
        (spectra[1], spectra[1]),
        (spectra[0], spectra[1]),
        (spectra[2], spectra[2]),
        (spectra[3], spectra[2]),
    )
    first_rows = (
        (1, 1, 1, 2, 2, 2),
        (0, 1, 2, 2, 2, 2),
        (0, 1, 1, 2, 2, 3),
        (0, 0, 2, 2, 2, 3),
    )
    double_cards = (
        (10, 10, 10, 10, 10, 10),
        (8, 8, 8, 8, 8, 8),
        (10, 8, 10, 8, 10, 8),
        (5, 10, 8, 5, 10, 8),
        (4, 8, 10, 4, 8, 10),
    )

    # Automorphisms x->x^u, y->x^v*y^t with t=1 or 5.
    # This is a geometry invariant, not an independently implemented lift.
    prefix = (0, 1, 3, 2, 0, 1, 3, 2)
    automorphisms = []
    for u in range(1, 5):
        for v in range(5):
            for t in (1, 5):
                image = tuple(G.idx(((u*a+v*prefix[b]) % 5,
                                     (t*b) % 8))
                              for a, b in elements)
                assert len(set(image)) == q
                automorphisms.append(image)

    def normalized(cell, image):
        if not cell:
            return ()
        values = [image[g] for g in cell]
        return min(tuple(sorted(
            multiplication[inverses[origin]][g] for g in values))
            for origin in values)

    def geometry_key(grids):
        images = []
        for image in automorphisms:
            a = tuple(sorted(normalized(cell, image)
                             for cell in grids["A"][0]))
            columns = [
                (normalized(grids["B"][0][j], image),
                 normalized(grids["B"][1][j], image))
                for j in range(6)]
            b = min(tuple(sorted(columns)),
                    tuple(sorted((y, x) for x, y in columns)))
            images.append((a, b))
        return min(images)

    # Each index consumes a different joint relative-support contract.
    # Higher harness batches consume new entries, not saved-code images.
    grng = random.Random(508040731)
    book, seen = [], set()
    for proposal in range(20000):
        card = len(book) % 20
        architecture, policy = divmod(card, 5)
        phase = len(book)//20
        top = first_rows[architecture]
        supports = {"A": [[3, 3]],
                    "B": [list(top), [3-w for w in top]]}
        grids = {}
        for side in ("A", "B"):
            grid = []
            for i, counts in enumerate(supports[side]):
                row = []
                for j, count in enumerate(counts):
                    if count == 0:
                        cell = ()
                    elif count == 1:
                        cell = (identity,)
                    elif count == 2:
                        order = double_cards[policy][(j+phase) % 6]
                        cell = (identity, grng.choice(palettes[order]))
                    else:
                        spectrum = (a_cards[policy][j] if side == "A"
                                    else spectra[(policy+i+j+phase) % 4])
                        cell = grng.choice(books[spectrum])
                    row.append(cell)
                grid.append(row)
            grids[side] = grid
        signature = geometry_key(grids)
        if signature in seen:
            continue
        seen.add(signature)
        book.append((supports, grids, signature))
        if len(book) > idx:
            break
    if len(book) <= idx:
        raise ValueError("indexed semidirect geometry book exhausted")
    supports, geometry, contract = book[idx]
    sa, sb = supports["A"], supports["B"]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert sum(sa[0])+max(map(sum, zip(*sb))) == 9
    assert max(map(sum, sb))+max(sa[0]) == 12
    assert all(sum(row) == 9 for row in sb)
    assert all(sum(column) == 3 for column in zip(*sb))

    def draw_cell(side, i, j):
        origin = G.idx(G.pick(rng))
        return [elements[multiplication[origin][g]]
                for g in geometry[side][i][j]]

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(supports[side])]

    def graph_stats(rows, cols):
        adjacency = [dict() for _ in rows]
        neighborhoods = []
        for syndrome in cols:
            ns = []
            while syndrome:
                bit = syndrome & -syndrome
                ns.append(bit.bit_length()-1)
                syndrome ^= bit
            neighborhoods.append(ns)
            for a, b in combinations(ns, 2):
                adjacency[a][b] = adjacency[a].get(b, 0)+1
                adjacency[b][a] = adjacency[b].get(a, 0)+1
        four = sum(v*(v-1)//2
                   for a, ns in enumerate(adjacency)
                   for b, v in ns.items() if a < b)
        six = 0
        for a, ns in enumerate(adjacency):
            for b, ab in ns.items():
                if b <= a:
                    continue
                for c, bc in adjacency[b].items():
                    if c > b and c in ns:
                        six += ab*bc*ns[c]
        for ns in neighborhoods:
            for a, b, c in combinations(ns, 3):
                six -= (adjacency[a][b]+adjacency[a][c]
                        +adjacency[b][c]-2)
        unseen, parts = set(range(len(rows))), 0
        while unseen:
            todo = [unseen.pop()]
            parts += 1
            while todo:
                for b in adjacency[todo.pop()]:
                    if b in unseen:
                        unseen.remove(b)
                        todo.append(b)
        return parts, four, six

    def refine(side, rows, cols):
        short = _short_collision_counts(rows, len(cols))
        bad6 = int(any(short))
        if side == "A" and not bad6:
            seen_syndromes = {0}
            seen_syndromes.update(cols)
            for i, x in enumerate(cols):
                for y in cols[:i]:
                    seen_syndromes.add(x ^ y)
            for i, x in enumerate(cols):
                if bad6:
                    break
                for j in range(i):
                    if bad6:
                        break
                    pair = x ^ cols[j]
                    for h in range(j):
                        syndrome = pair ^ cols[h]
                        if syndrome in seen_syndromes:
                            bad6 = 1
                            break
                        seen_syndromes.add(syndrome)

        loads = []
        for anchor in range(0, len(cols), q):
            burden = 0
            for i, x in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ x
                for j in range(i):
                    if j != anchor:
                        weight = (pair ^ cols[j]).bit_count()
                        if weight <= 3:
                            burden += (5-weight)**2
            loads.append(burden)

        probe = random.Random(508040919+1009*idx+(side == "B"))
        permutation = list(range(len(cols)))
        minimum, vulnerable = len(cols)+1, set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10 and len(vulnerable) < 4096:
                    vulnerable.add(word)

        for trial in range(32 if side == "A" else 20):
            probe.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)
            probe.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    candidate = word ^ other
                    offer(candidate)
                    cw = candidate.bit_count()
                    if candidate and cw < weight:
                        word, weight = candidate, cw
                basis[i] = word
            if minimum <= 4:
                break
        return (int(any(short)), bad6, max(0, 10-minimum),
                len(vulnerable), max(loads), sum(loads))

    def portfolio(side):
        width = len(supports[side][0])*q
        sites = [(i, j) for i, row in enumerate(supports[side])
                 for j, count in enumerate(row) if count]
        pool, identities = [], set()
        serial = 0

        def assess(grid):
            rows = rows_of(grid, side)
            cols = _sc8_columns(rows, width)
            assert all(c.bit_count() == 3 for c in cols)
            parts, four, six = graph_stats(rows, cols)
            deficit = len(rows)-gf2_rank(rows)
            cost = (parts-1, deficit, four, six)
            energy = 100*(parts-1)+80*deficit+(80*four+six/6)/q
            return cost, energy, rows, cols

        def retain(grid, state):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in identities:
                return
            identities.add(signature)
            serial += 1
            cost, _, rows, cols = state
            pool.append((cost, serial, _copy_cells(grid), rows, cols))
            pool.sort(key=lambda item: item[:2])
            del pool[5:]

        for restart in range(3):
            grid = draw_grid(side)
            current = assess(grid)
            local_grid, local = _copy_cells(grid), current
            retain(grid, current)
            for step in range(180):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed = assess(grid)
                delta = proposed[1]-current[1]
                temperature = 9*(0.980**step)+0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current = proposed
                    if current[0] < local[0]:
                        local_grid, local = _copy_cells(grid), current
                else:
                    grid[i][j] = old
                if step in (89, 179):
                    retain(grid, current)
            retain(local_grid, local)

        finalists = []
        for cost, number, grid, rows, cols in pool[:4]:
            quality = refine(side, rows, cols)
            key = (cost[0], quality[0], quality[1], cost[1],
                   quality[2], quality[3], quality[4], quality[5],
                   cost[2], cost[3], number)
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        return finalists[:3]

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                return False
            first = (row & -row).bit_length()-1
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length()-1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(c) for c in range(width)}) == 1

    apool, bpool = portfolio("A"), portfolio("B")
    options = []
    for ak, A in apool:
        for bk, B in bpool:
            structure = tuple(v for a, b in zip(ak[:-1], bk[:-1])
                              for v in (max(a, b), a+b))
            options.append((structure, A, B))
    options.sort(key=lambda item: item[0])
    finalists, fallback = [], None
    for structure, A, B in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 560 and 275*n <= 1000*k < 325*n):
            continue
        candidate = {"idx": idx, "cells": {"A": A, "B": B},
                     "k": k, "contract": contract,
                     "component_screen": structure}
        if fallback is None:
            fallback = candidate
        if not connected(hx+hz, n):
            continue
        bad = int(_sc8_logical4(hx, hz, n)
                  or _sc8_logical4(hz, hx, n))
        seed = 508040733+1009*idx
        x = _sc8_bp(hz, hx, n, seed)
        z = _sc8_bp(hx, hz, n, seed+1)
        key = (bad, max(x[0], z[0]), x[0]+z[0],
               max(x[1], z[1]), x[1]+z[1],
               x[2]+z[2], structure)
        candidate["bp_surrogate"] = (x, z)
        candidate["joint_screen"] = key
        finalists.append((key, candidate))
        if len(finalists) >= 4:
            break

    if finalists:
        finalists.sort(key=lambda item: item[0])
        winner = finalists[0][1]
    elif fallback is not None:
        winner = fallback
    else:
        winner = None
        # Preserve this indexed relative-support contract in the rescue.
        for attempt in range(96):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if 275*n <= 1000*k < 325*n:
                winner = {"idx": idx, "cells": {"A": A, "B": B},
                          "k": k, "contract": contract,
                          "component_screen": ("rescue", attempt)}
                break
        if winner is None:
            raise ValueError("indexed C5:C8 contract missed cell 0.300")

    A, B = winner["cells"]["A"], winner["cells"]["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    assert n == 560 and css_ok(hx, hz)
    assert max(row.bit_count() for row in hx) == 9
    assert max(row.bit_count() for row in hz) == 12
    assert 6*sum(row.bit_count() for row in hx+hz) == 24480
    assert rows_of(A, "A") == _component_rows(G, "A", A)
    assert rows_of(B, "B") == _component_rows(G, "B", B)
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    assert winner["k"] == n-gf2_rank(hx)-gf2_rank(hz)
    return winner


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without subsequent randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Independent component short-cycle optimization suffices for rare circuit failures: all five current-parent measured codes reached failure caps, with worse-basis s=5.402-5.782 and widespread logical flips.
# REFUTED: Exact-k=180 screening retains every optimized support architecture: current-parent slots 7, 9, 11, 13 and others were voided because optimized fillings did not attain the target rank.
# REFUTED: Canonical direction-separated geometry cards guarantee distinct Tanner codes: the measured parent produced only eight distinct codes from sixty slots.
# REFUTED: Direction separation eliminates failure-cap events on the cyclic C20 bed: every measured parent code still reached a failed-shot cap, with worse-basis s=5.608..5.809.
# REFUTED: Equal k=160,d2s=10,w=12 and ASC depth15 certify comparable circuit reliability on Q8 x C5: current slots1/5 scored 6.321/5.808, with Z failures 73/200000 versus 100/81500 and collective flips across all 160 logicals.
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
