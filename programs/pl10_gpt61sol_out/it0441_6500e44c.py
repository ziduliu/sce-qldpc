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
CONCEPT = ("Cyclic C19, order 19, shapes (1,4)x(2,7): reallocating one A term to five alternative eighteen-term mixed-degree B architectures, followed by noisy-syndrome holdout selection, should raise cell-0.500 circuit fitness above 3.3458.")
GROUP = ("Frozen under L1: retain C19, order 19, multiplication and encoding byte-for-byte. This exact bed reached 3.3458, while the current anchored filling scored 3.2583. These readings motivate a filling comparison, not a group-only reliability claim.")
SHAPES = ("Frozen (1,4)x(2,7): size factor thirty permits q<=19; q=19 gives n=570, kb=285 and guaranteed rate 0.500. Unequal components have 76 variables/19 checks and 133 variables/38 checks. Require complete verified k=285 and zero k-kb, avoiding proxy selection through rank inflation. Parent failures reached all 285 logicals with mostly 1.0-1.1x Acol/Bcol data and Acol/Brow check enrichment, establishing no persistent localized repair site.")
FILLING = ("Twenty contracts cross four canonical A component classes with five B support architectures; additional slots consume additional A classes. A has three binomials and one trinomial, weights [2,2,2,3]. B has three degree-two and four degree-three columns, row totals 9/9 and eighteen terms. Four architectures use three cross-row weak columns with different strong-column concentration; the fifth has internal weak edges. Helper-derived singleton actions screen exact short component words, ranks and Tanner cycles; rooted component invariants conservatively discard repeated proposals. Complete verified CSS finalists exclude logicals through three and retain k=285. Two-interval BP probes use a noisy intermediate syndrome, a perfect final syndrome, degree-scaled data noise and opposite-check proper suffix faults; independent holdouts lead selection. They omit actual ASC ordering, six-round measurement history and relay/LSD and are not calibrated pL. Both CSS weight caps are twelve; extraction increases from 17670 to19494 CNOTs. Test whether stronger B protection offsets that increase and reduces the parent's 135-160 Z failures/250, 116-145 X failures/500 and collective flips.")
FAMILY_NAME = "r050_C19_14x27_B18_noisysyndrome_holdout_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order 19")
    els = [(a,) for a in range(19)]

    def red(g):
        return (int(g[0]) % 19,)

    def mul(g, h):
        return ((g[0] + h[0]) % 19,)

    def inv(g):
        return ((-g[0]) % 19,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Five B architectures: column totals 3,3,3,3,2,2,2;
# both rows total nine. Zero cells are deliberate, never repairs.
_B_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,1),(1,1),(1,1)),
    ((3,0),(1,2),(1,2),(1,2),(1,1),(1,1),(1,1)),
    ((3,0),(0,3),(2,1),(1,2),(1,1),(1,1),(1,1)),
    ((2,1),(2,1),(1,2),(1,2),(2,0),(0,2),(1,1)),
    ((3,0),(1,2),(1,2),(1,2),(2,0),(0,2),(1,1)),
)


def _kernel56(rows, ncols):
    """Exact weight-five/six counts when kernel weights <=4 are absent.

    Pair/triple syndrome matches count each weight-five word ten times.
    Distinct triple collisions count each weight-six word ten times.
    Smaller-word exclusion makes these colliding supports disjoint.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            s = x ^ y
            pairs[s] = pairs.get(s, 0) + 1
    triples = {}
    five = six = 0
    for i in range(ncols):
        for j in range(i):
            s = columns[i] ^ columns[j]
            for h in range(j):
                value = s ^ columns[h]
                five += pairs.get(value, 0)
                six += triples.get(value, 0)
                triples[value] = triples.get(value, 0) + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def _retired_anchor13_CTX(rng, G, idx):
    """Inactive anchored-incidence filling retained for provenance."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 19
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    load_weight = (0.0, 0.2, 0.5, 1.0, 2.0)[marker]

    def direction(d):
        d %= q
        return min(d, q - d)

    def directions(cell):
        return {direction(a - b) for a, b in combinations(cell, 2)}

    def normalized(cell, multiplier=1):
        values = [(multiplier * a) % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def geometry_key(cells):
        # Quotient cyclic multipliers, independent cell origins and
        # column permutations. Arbitrary Tanner equivalence remains
        # the evaluator's responsibility.
        return min(tuple(sorted(normalized(cell, u) for cell in cells))
                   for u in range(1, q))

    # Cache the two actual singleton actions, not a hand-written lift.
    # Cell addition and block placement are linear over F2. The selected
    # components are checked against the complete verified lift below.
    block_mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz1)

    def component(grid, side):
        width = len(grid[0]) * q
        rows = [0] * (len(grid) * q)
        for i, grid_row in enumerate(grid):
            for j, cell in enumerate(grid_row):
                shift = j * q
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        rows[i * q + t] ^= bits << shift
        return rows, width

    # Restore difference-disjoint Sidon geometry on A, but never the
    # archived fourteen-term B construction. The complete new codes
    # have a different gate count, B rank and logical dimension.
    # Normalize one binomial direction by a common automorphism.
    trinomials = sorted({
        normalized((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if len(directions((0, a, b))) == 3
    })
    geometries = set()
    for first, second in combinations(trinomials, 2):
        df, ds = directions(first), directions(second)
        if df & ds or 1 in df or 1 in ds:
            continue
        for b in range(2, (q + 1) // 2):
            if b in df or b in ds:
                continue
            geometries.add(geometry_key(
                ((0, 1), (0, b), first, second)))

    book = []
    for signature in sorted(geometries):
        grid = [[[(a,) for a in cell] for cell in signature]]
        rows, width = component(grid, "A")
        stats = _kernel_statistics(rows, width)
        if stats[0] or stats[1] or gf2_rank(rows) != q:
            continue
        # Actual short-word multiplicity precedes Tanner-cycle costs.
        # Neither statistic is asserted to predict circuit pL.
        key = (stats[2], _four_cycles(rows), _six_cycles(rows),
               stats[3], signature)
        book.append((key, grid))
    book.sort(key=lambda entry: entry[0])
    assert len(book) >= 5
    wanted = 5 * batch + marker
    if wanted >= len(book):
        raise ValueError("No unused canonical A geometry for this batch")
    akey, A = book[wanted]
    bin_directions = {
        next(iter(directions(tuple(g[0] for g in cell))))
        for cell in A[0] if len(cell) == 2
    }

    # Column zero anchors every check in the upper orbit. Other columns
    # are edges. Internal edges belong ONLY to the unanchored orbit:
    # an edge between two anchored checks would create a weight-three
    # word. These four briefs genuinely change check-degree profiles.
    profiles = (
        ((1,0),(1,1),(1,1),(1,1),(0,2),(0,2),(0,2)),
        ((1,0),(1,1),(1,1),(1,1),(1,1),(0,2),(0,2)),
        ((1,0),(1,1),(1,1),(1,1),(1,1),(1,1),(0,2)),
        ((1,0),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    )
    profile = profiles[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2, 2, 3, 3]
    assert sum(map(sum, sb)) == 13
    assert tuple(map(sum, sb)) == (
        (4, 9), (5, 8), (6, 7), (7, 6))[architecture]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [(1, j) for j in cross]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def gauge(grid):
        # Remove the common cross-row voltage origin, not a new draw.
        origin = grid[1][cross[0]][0][0]
        for j in cross:
            grid[1][j] = [((grid[1][j][0][0] - origin) % q,)]
        return grid

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        grid[0][0] = [(0,)]
        for i, j in internal:
            assert i == 1
            grid[i][j] = [(0,), (rng.randrange(1, (q + 1) // 2),)]
        ratios = rng.sample(range(q), len(cross))
        for j, ratio in zip(cross, ratios):
            grid[0][j], grid[1][j] = [(0,)], [(ratio,)]
        return gauge(grid)

    def assess_B(grid):
        """Exact short words of a graph incidence matrix with anchors.

        With distinct columns, weight-three words are triangles or
        an edge plus its two endpoint anchors. Weight-four words are
        graph quartets or a two-edge path plus its endpoint anchors.
        All columns are obtained from verified component actions.
        """
        rows, width = component(grid, "B")
        endpoints = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                endpoints[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        edges, anchors = set(), set()
        adjacency = [0] * len(rows)
        for value in endpoints:
            degree = value.bit_count()
            assert degree in (1, 2)
            low = value & -value
            a = low.bit_length() - 1
            if degree == 1:
                assert a < q
                anchors.add(a)
                continue
            b = (value ^ low).bit_length() - 1
            edges.add((a, b))
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a
        repeated = width - len(set(endpoints))
        assert len(anchors) == q
        anchor_mask = sum(1 << a for a in anchors)

        triangle_sum = sum(
            (adjacency[a] & adjacency[b]).bit_count()
            for a, b in edges)
        assert triangle_sum % 3 == 0
        anchor_edges = sum(a in anchors and b in anchors
                           for a, b in edges)
        triangles = triangle_sum // 3 + anchor_edges

        partitions = 0
        for a, neighbors in enumerate(adjacency):
            for other in adjacency[:a]:
                overlap = (neighbors & other).bit_count()
                partitions += overlap * (overlap - 1) // 2
        assert partitions % 2 == 0
        graph_fours = partitions // 2
        anchor_degree = [
            (neighbors & anchor_mask).bit_count()
            for neighbors in adjacency
        ]
        anchor_fours = sum(d * (d - 1) // 2 for d in anchor_degree)
        quartets = graph_fours + anchor_fours

        peak = total_load = 0
        if not repeated and not triangles:
            for a, b in edges:
                neighbors = adjacency[a] & ~(1 << b)
                load = 0
                while neighbors:
                    bit = neighbors & -neighbors
                    c = bit.bit_length() - 1
                    load += (
                        adjacency[c] & adjacency[b]).bit_count() - 1
                    neighbors ^= bit
                if a in anchors:
                    load += anchor_degree[b] - 1
                if b in anchors:
                    load += anchor_degree[a] - 1
                peak = max(peak, load)
                total_load += load

            # Anchor variables themselves also participate in the
            # path words and must not disappear from concentration.
            for a in anchors:
                neighbors, load = adjacency[a], 0
                while neighbors:
                    bit = neighbors & -neighbors
                    b = bit.bit_length() - 1
                    load += anchor_degree[b] - 1
                    neighbors ^= bit
                peak = max(peak, load)
                total_load += load
            assert total_load == 4 * quartets

        stats = (repeated, triangles, quartets, peak)
        energy = ((300 * repeated + 50 * triangles + quartets) / q
                  + load_weight * peak)
        return energy, stats, rows

    pool, seen = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != 2 * q:
            return
        signature = tuple(tuple(g[0] for g in cell)
                          for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        coupling = sum(
            direction(grid[i][j][1][0] - grid[i][j][0][0])
            in bin_directions for i, j in internal)
        key = (stats[2] / q + load_weight * stats[3],
               stats[3], stats[2], coupling)
        serial += 1
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    # Cached verified actions and graph intersection counts permit a
    # substantially deeper search than the parent's pair-syndrome census.
    for restart in range(6):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(960):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [
                    (0,), (rng.randrange(1, (q + 1) // 2),)]
            else:
                proposal[i][j] = [(rng.randrange(q),)]
            gauge(proposal)
            proposed, ps, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 6.0 * (0.994 ** step) + 0.03
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No full-rank anchored B component passed its contract")

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def quotient_columns(stabilizers, n):
        # A linear canonical remainder modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residual = [0] * n
        for c in range(n):
            row = pivots.get(c)
            if row is None:
                residual[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def logical_risk(checks, stabilizers, n):
        """Exact <=3 rejection and an uncalibrated fault-class loss.

        Equal syndrome is insufficient for equivalence: aggregate mass
        only when the actual stabilizer quotient labels also agree.
        Sum-minus-largest class mass is an ideal class-decoder loss
        for this truncated artificial model, NOT measured circuit pL.
        """
        import random

        cs = columns(checks, n)
        spread = columns(stabilizers, n)
        if 0 in cs or len(set(cs)) != n:
            return None
        residual = quotient_columns(stabilizers, n)
        singles = dict(zip(cs, residual))
        exposure = [2 + s.bit_count() + h.bit_count()
                    for s, h in zip(cs, spread)]
        buckets = {0: {0: 1000000000}}

        def add(syndrome, label, mass):
            classes = buckets.setdefault(syndrome, {})
            classes[label] = classes.get(label, 0) + mass

        for i, syndrome in enumerate(cs):
            add(syndrome, residual[i], 4096 * exposure[i])

        for i, si in enumerate(cs):
            ri = residual[i]
            for j in range(i):
                syndrome = si ^ cs[j]
                label = ri ^ residual[j]
                third = singles.get(syndrome)
                if third is not None and third != label:
                    return None
                # Degree weighting distinguishes low-exposure anchor
                # variables from the parent's uniformly degree-two B.
                add(syndrome, label,
                    16 * exposure[i] * exposure[j])

        source = []
        for row in stabilizers:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length() - 1)
                row ^= bit
            source.append(sites)

        # Three independently permuted check orders approximate possible
        # ancilla suffixes. Each order can be serialized legally within
        # one basis, but none is asserted to be the actual joint ASC
        # order. Common streams compare finalists without selecting a
        # specially favorable order for each labeling.
        order_rng = random.Random(
            190513570 + 1009 * idx + len(checks))
        for repeat in range(3):
            for original in source:
                sites = list(original)
                if len(sites) < 2:
                    continue
                order_rng.shuffle(sites)
                mass = max(
                    1, 256 * len(sites) // (3 * (len(sites) - 1)))
                syndrome = label = 0
                for position, c in enumerate(reversed(sites)):
                    syndrome ^= cs[c]
                    label ^= residual[c]
                    if position + 1 < len(sites):
                        add(syndrome, label, mass)

        loss = peak = ambiguous = zero_loss = 0
        for syndrome, classes in buckets.items():
            if len(classes) < 2:
                continue
            error = sum(classes.values()) - max(classes.values())
            loss += error
            peak = max(peak, error)
            ambiguous += 1
            if syndrome == 0:
                zero_loss += error
        return loss, zero_loss, peak, ambiguous

    winner = None
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 570 or k != 285:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        # Protect the weaker channel using aggregated logical-class
        # loss, then undetectable suffix mass and concentration.
        # These costs do not replace the evaluator's ASC experiment.
        score = tuple(value for t in (0, 1, 2, 3)
                      for value in (max(px[t], pz[t]), px[t] + pz[t]))
        score += coarse
        if winner is None or score < winner[0]:
            winner = (score, B, hx, hz, px, pz)

    if winner is None:
        raise ValueError("No k=285 anchored filling passed the CSS logical screen")
    score, B, hx, hz, px, pz = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == component(A, "A")[0]
    assert actual_b == component(B, "B")[0]
    assert all(len(directions(tuple(g[0] for g in cell))) == 3
               for cell in A[0] if len(cell) == 3)
    assert B[0][0] == [(0,)] and not B[1][0]
    assert gf2_rank(actual_a) == q
    assert gf2_rank(actual_b) == 2 * q
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 285
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 17670
    assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)

    _, bstats, _ = assess_B(B)
    exact_b = _kernel_statistics(actual_b, 7 * q)
    assert exact_b[:3] == bstats[:3]
    assert not any(exact_b[:2])
    return {"A": A, "B": B, "design": (architecture, wanted),
            "k": 285, "fault_class_loss": (px, pz),
            "component_contract": akey, "selection_score": score,
            "quartet_contract": load_weight,
            "B_quartets_and_peak": bstats[2:], "cnots": 17670}


def NEW_CTX(rng, G, idx):
    """Nine-term A, eighteen-term B and independent noisy-syndrome selection."""
    import hashlib
    import math
    import random
    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 19
    architecture, wanted = idx % 5, idx // 5
    mask = (1 << q) - 1

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    # These are the scaffold's two actual actions, not an independent lift.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz1)

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, word in enumerate(actions[side][G.idx(g)]):
                        rows[i*q+t] ^= word << (j*q)
        return rows, len(grid[0]) * q

    def fingerprint(rows, width):
        """Conservative rooted component invariant, not full canonization."""
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, value in enumerate(columns(rows, width)):
            while value:
                bit = value & -value
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                value ^= bit
        images = []
        # Regular translations cover every check in each row orbit.
        for root in range(0, nr, q):
            colors = [0] * nr + [1] * width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(10):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr + width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            images.append(digest.hexdigest())
        return tuple(sorted(images))

    def normal(cell, multiplier=1):
        values = [multiplier*a % q for a in cell]
        return min(tuple(sorted((a-origin) % q for a in values))
                   for origin in values)

    def geometry(cells):
        return min(tuple(sorted(normal(cell, u) for cell in cells))
                   for u in range(1, q))

    # The deterministic catalogue is shared conceptually by every slot.
    # Indexed component classes, not independent origin draws, are contracts.
    steps = []
    for b, c in combinations(range(2, 10), 2):
        ds = (1, b, c)
        signed = ds + tuple(-d for d in ds)
        if all((a+b+c) % q != 0
               for a in signed for b in signed for c in signed):
            steps.append(ds)
    assert steps
    triples = sorted({
        normal((0, a, b))
        for a, b in combinations(range(1, q), 2)
    })
    catalogue = random.Random(190918037)
    raw, signatures = [], set()
    target = max(96, 4*wanted + 48)
    for trial in range(12000):
        ds = catalogue.choice(steps)
        terminal = catalogue.choice(triples)
        signature = geometry(
            tuple((0, d) for d in ds) + (terminal,))
        if signature in signatures:
            continue
        signatures.add(signature)
        A = [[[(a,) for a in cell] for cell in signature]]
        ar, width = component(A, "A")
        stats = _kernel_statistics(ar, width)
        if any(stats[:2]) or gf2_rank(ar) != q:
            continue
        key = (stats[2], _four_cycles(ar), stats[3],
               _six_cycles(ar), signature)
        raw.append((key, A, ar))
        if len(raw) >= target:
            break

    bank, classes = [], set()
    for key, A, ar in sorted(raw, key=lambda item: item[0]):
        identity = fingerprint(ar, 4*q)
        if identity in classes:
            continue
        classes.add(identity)
        bank.append((key, identity, A, ar))
        if len(bank) > wanted:
            break
    if wanted >= len(bank):
        raise ValueError("Distinct nine-term C19 A catalogue exhausted")
    akey, aid, A, ar = bank[wanted]

    weak = ((1,1),) * 3
    profiles = (
        weak + ((2,1),(2,1),(1,2),(1,2)),
        weak + ((3,0),(1,2),(1,2),(1,2)),
        weak + ((3,0),(0,3),(2,1),(1,2)),
        weak + ((3,0),(3,0),(0,3),(0,3)),
        ((2,0),(0,2),(1,1),(2,1),(2,1),(1,2),(1,2)),
    )
    profile = profiles[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2,2,2,3]
    assert tuple(map(sum, sb)) == (9, 9)
    assert sorted(map(sum, zip(*sb))) == [2,2,2,3,3,3,3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    anchors = [next(i for i in range(2) if sb[i][j])
               for j in range(7)]
    cross = [j for j, p in enumerate(profile) if p == (1,1)]
    fixed_cross = cross[0]
    sites = [
        (i, j) for i in range(2) for j in range(7)
        if sb[i][j]
        and not (i == anchors[j] and sb[i][j] == 1)
        and (i, j) != (1, fixed_cross)
    ]
    nonidentity = [g for g in G.elements if g != G.e]

    def new_cell(i, j):
        count = sb[i][j]
        if i == anchors[j]:
            return [G.e] + rng.sample(nonidentity, count-1)
        return rng.sample(G.elements, count)

    def draw_B():
        B = [[new_cell(i, j) if sb[i][j] else []
              for j in range(7)] for i in range(2)]
        B[1][fixed_cross] = [G.e]
        return B

    def assess_B(B):
        br, width = component(B, "B")
        stats = _kernel_statistics(br, width)
        deficit = 2*q - gf2_rank(br)
        cycles = _four_cycles(br)
        energy = (
            1000*stats[0] + 100*stats[1] + stats[2]
            + 500*deficit + 0.015*cycles) / q
        return energy, stats, deficit, cycles, br

    pool, labels = [], set()
    serial = 0

    def remember(B, stats, deficit, cycles, br):
        nonlocal serial
        if any(stats[:2]) or deficit:
            return
        key = (stats[2], stats[3], cycles)
        if len(pool) == 8 and key > pool[-1][0]:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
        if signature in labels:
            return
        labels.add(signature)
        serial += 1
        pool.append((key, serial, clone(B), list(br)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    for restart in range(4):
        B = draw_B()
        energy, stats, deficit, cycles, br = assess_B(B)
        remember(B, stats, deficit, cycles, br)
        for step in range(500):
            proposal = clone(B)
            i, j = rng.choice(sites)
            proposal[i][j] = new_cell(i, j)
            proposed, ps, pd, pc, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 8.0*(0.988**step) + 0.04
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                B, energy = proposal, proposed
                stats, deficit, cycles, br = ps, pd, pc, pr
                remember(B, stats, deficit, cycles, br)
    if not pool:
        raise ValueError("No full-rank eighteen-term B passed its contract")

    def quotient(stabilizers, width):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residue = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                residue[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residue[bit.bit_length() - 1]
                tail ^= bit
            residue[c] = value
        return residue

    def short_guard(checks, stabilizers, width):
        """Exact rejection through three for distinct nonzero columns."""
        cs = columns(checks, width)
        if 0 in cs or len(set(cs)) != width:
            return None
        residue = quotient(stabilizers, width)
        singles = dict(zip(cs, residue))
        for i, si in enumerate(cs):
            for j in range(i):
                third = singles.get(si ^ cs[j])
                if third is not None and third != residue[i] ^ residue[j]:
                    return None
        return residue

    def noisy_probe(checks, stabilizers, width, residue, seed, shots):
        """Two-interval noisy-syndrome surrogate, NOT ASC measurement.

        H*e1+m=s1; H*(e1+e2)=s2. Proper opposite-check suffixes
        use common random orders, not the evaluator's ASC ordering.
        """
        import numpy as np

        def sites_of(row):
            result = []
            while row:
                bit = row & -row
                result.append(bit.bit_length() - 1)
                row ^= bit
            return result

        m = len(checks)
        variables = 2*width + m
        augmented = [
            row | (1 << (2*width+r))
            for r, row in enumerate(checks)
        ] + [row | (row << width) for row in checks]
        ns = [sites_of(row) for row in augmented]
        row_width = max(map(len, ns))
        incidence = np.full(
            (2*m, row_width), variables, dtype=np.int32)
        for r, row in enumerate(ns):
            incidence[r, :len(row)] = row
        flat = incidence.ravel()
        valid = flat != variables
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        ve = np.full(
            (variables+1, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            ve[c, used[c]] = edge
            used[c] += 1

        dc = np.asarray([c.bit_count()
                         for c in columns(checks, width)], dtype=float)
        ds = np.asarray([c.bit_count()
                         for c in columns(stabilizers, width)], dtype=float)
        levels = np.asarray((0.40, 0.70, 1.0))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+dc+ds)*2/3
        measure_mu = np.asarray([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(
            -2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measure_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :width] = channel.random((shots, width)) < data_p
        error[:, width:2*width] = (
            channel.random((shots, width)) < data_p)
        error[:, 2*width:variables] = (
            channel.random((shots, m)) < measure_p)

        for row in stabilizers:
            ns_hook = sites_of(row)
            if len(ns_hook) < 2:
                continue
            order = channel.permutation(ns_hook)
            event_p = levels*3*0.002*len(ns_hook)*2/3
            for interval in range(2):
                active = channel.random(shots) < event_p
                for shot in np.flatnonzero(active):
                    cut = int(channel.integers(1, len(order)))
                    error[shot, interval*width+order[cut:]] ^= True

        # This marginal allowance is heuristic, not a calibrated channel.
        marginal_mu = data_mu + 3*0.002*ds*0.35
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate(
            (marginal_p, marginal_p,
             np.clip(measure_p, 0.002, 0.25)), axis=1)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log(
            (1-probabilities)/probabilities)
        target = (
            error[:, flat].reshape(
                shots, 2*m, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        true_final = error[:, :width] ^ error[:, width:2*width]
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        done = np.zeros(shots, dtype=bool)
        failed = np.zeros(shots, dtype=bool)
        completion = np.full(shots, 37, dtype=np.int32)

        for iteration in range(1, 37):
            shaped = messages.reshape(shots, 2*m, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(shots, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((shots, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, ve].sum(axis=2)
            posterior[:, variables] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(
                    shots, 2*m, row_width).sum(axis=2) & 1
            ).astype(bool)
            newly = np.all(syndrome == target, axis=1) & ~done
            final = hard[:, :width] ^ hard[:, width:2*width]
            for shot in np.flatnonzero(newly):
                label = 0
                for c in np.flatnonzero(final[shot] ^ true_final[shot]):
                    label ^= residue[int(c)]
                failed[shot] = bool(label)
                completion[shot] = iteration
            done |= newly
            if np.all(done):
                break
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing), -30, 30)
            messages[:, ~valid] = 30.0
        return (int((failed | ~done).sum()), int((~done).sum()),
                int((completion > 12).sum()), int(completion.sum()))

    options, bclasses = [], set()
    for coarse, serial, B, br in pool:
        bid = fingerprint(br, 7*q)
        if bid in bclasses:
            continue
        bclasses.add(bid)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 570 or k != 285:
            continue
        rz = short_guard(hz, hx, n)
        rx = short_guard(hx, hz, n)
        if rz is None or rx is None:
            continue
        stats = _kernel_statistics(br, 7*q)
        higher = (_kernel56(br, 7*q)
                  if not any(stats[:3]) else (10**9, 10**9))
        structural = (stats[2], higher, stats[3], coarse[2], serial)
        options.append(
            (structural, B, br, bid, hx, hz, rz, rx))
        if len(options) == 4:
            break
    if not options:
        raise ValueError("No k=285 mixed-degree CSS finalist")
    options.sort(key=lambda item: item[0])

    def quality(z, x):
        return (max(z[0], x[0]), z[0]+x[0],
                max(z[1], x[1]), z[1]+x[1],
                max(z[2], x[2]), z[2]+x[2], z[3]+x[3])

    seed = 190918570 + 1009*idx
    training = []
    for number, option in enumerate(options[:3]):
        structural, B, br, bid, hx, hz, rz, rx = option
        z = noisy_probe(hz, hx, 570, rz, seed, 24)
        x = noisy_probe(hx, hz, 570, rx, seed+1, 24)
        training.append(
            (quality(z, x)+structural+(number,), number, option, z, x))
    training.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, option, tz, tx in training[:2]:
        structural, B, br, bid, hx, hz, rz, rx = option
        z = noisy_probe(hz, hx, 570, rz, seed+1000003, 64)
        x = noisy_probe(hx, hz, 570, rx, seed+1000004, 64)
        score = quality(z, x)+quality(tz, tx)+structural+(number,)
        finalists.append((score, option, z, x))
    finalists.sort(key=lambda item: item[0])
    score, option, z, x = finalists[0]
    structural, B, br, bid, hx, hz, rz, rx = option

    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == ar and actual_b == br
    assert gf2_rank(actual_a) == q
    assert gf2_rank(actual_b) == 2*q
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 285
    assert max(row.bit_count() for row in hx+hz) == 12
    assert 6*sum(row.bit_count() for row in hx+hz) == 19494
    assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": 285,
            "design": (architecture, wanted),
            "component_classes": (aid, bid),
            "A_contract": akey, "B_contract": structural,
            "noisy_syndrome_holdout": (z, x),
            "selection_score": score, "cnots": 19494}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified mixed-degree cell without late redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: Asymmetric A<=3/B<=4 component exclusion balances C19 memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored only 3.1933 despite much lower Z failed-shot rates.
# REFUTED: Difference-disjoint Sidon A cells, triangle-free graphic B and shared-check quartic minimization suffice for rare C19 circuit failures: fitness was 3.3458, measured Z failures were 106..148/250, and every measured code's failures collectively affected all 288 logicals.
# REFUTED: AP-containing A and worst-edge quartet penalties eliminate collective failures at unchanged extraction cost: fitness was 3.3405, Z failed 111..141/250 shots, X failed 163..200/500, and failures affected all 288 logicals.
# REFUTED: Singleton-anchored B13 and truncated logical-class fault-mass selection improve fitness above 3.3405 while reducing gates: measured fitness fell to 3.2583, Z failed 135..160/250 shots, failures reached all 285 logicals, and sixty slots collapsed to eight distinct codes. This does not isolate gate count from component geometry or selection.
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
