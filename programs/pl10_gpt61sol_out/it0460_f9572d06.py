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
CONCEPT = ("For cyclic C13, order 13, shapes (2,5)x(3,8), degree-three A columns paired with mixed-row forest B22/B23 components and six-interval noisy-syndrome holdout selection should raise cell-0.350 circuit fitness above 5.1205 through lower extraction load and fewer collective failures.")
GROUP = ("Frozen L1 group: keep the exact C13 implementation, order and encoding byte-for-byte. This bed leads cell 0.350 at fitness 5.1205. Size factor forty-six permits q<=13, giving n=598 at the largest legal lift. This is a filling experiment, not evidence for a group-only advantage.")
SHAPES = ("Frozen (2,5)x(3,8): A has 65 variables/26 checks and B has 104 variables/39 checks. kb=195 gives guaranteed rate 15/46=0.326087; complete verified ranks require k=195. Parent failures reached all 195 logicals, with diffuse 1.1-1.2x Acol/Bcol data and Acol/Brow check enrichment. Repeatedly vulnerable logical labels included #78 and #66, but neither the labels nor the block maps establish a persistent isolated repair site.")
FILLING = ("Twenty contracts cross two fresh A-gap families, two B budgets and five canonical B architectures. A retains fifteen terms, degree-three columns and row loads 8/7. B22 has two mixed-row degree-two columns using different row pairs, hence a forest; B23 has one mixed-row degree-two column. All remaining B columns have degree three, with row loads 7/7/8 or 7/8/8. CSS weights stay <=11 and six-round CNOT counts fall from 25974 to 24882/25428. Additional batches consume further canonical A cards beyond the parent's first fifteen. Helper-derived component screens exclude kernels through four and retain diverse alternatives. Complete verified CSS pairs are selected using six independent fault intervals, five noisy syndrome observations, a perfect final observation, degree-scaled data faults, opposite-check proper-suffix faults and stabilizer-aware BP residuals. An independent holdout leads the final choice. This omits actual ASC ordering and relay/LSD and is not measured pL. Target the parent's 100-102 X failures in 11750-16750 shots versus 100-104 Z failures in 28500-43000 shots, rather than maximizing a distance upper bound.")
FAMILY_NAME = "r035_C13_25x38_mixedforest_sixround_holdout_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 requires order 13")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


# Four graphic B architectures, all with seven degree-two columns.
# Row totals are 7/7 or 8/6; zeros are prescribed, not repairs.
_B_PROFILES = (
    ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(0,2),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
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


def _retired_quaternion_graph_CTX(rng, G, idx):
    """Inactive previous-bed filling, retained only for provenance."""
    from itertools import combinations
    from math import exp

    q = G.q
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    assert q == 16

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        # Zero partners are extraction probes, not submitted codes.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    # Automorphisms x->x^u, y->x^v*y, u odd. Independent right
    # translations are A-column gauges. Canonicalization removes these
    # obvious copies; arbitrary Tanner isomorphism remains downstream.
    automorphisms = [
        tuple(G.idx(((u*a + v*b) % 8, b))
              for a, b in G.elements)
        for u in (1, 3, 5, 7) for v in range(8)
    ]

    def normal(cell, transform):
        values = [G.elements[transform[G.idx(g)]] for g in cell]
        return min(tuple(sorted(
            G.idx(G.mul(g, G.inv(origin))) for g in values))
                   for origin in values)

    triangles = [
        (G.e, (d, 0), (a, 1))
        for d in (1, 2, 3, 5, 6, 7) for a in range(8)
    ]
    binomial_pairs = (
        ((G.e, (1, 0)), (G.e, (3, 0))),
        ((G.e, (1, 0)), (G.e, (0, 1))),
        ((G.e, (0, 1)), (G.e, (1, 1))),
    )
    all_cells = set(triangles)
    for pair in binomial_pairs:
        all_cells.update(pair)
    normal_forms = {
        cell: tuple(normal(cell, transform)
                    for transform in automorphisms)
        for cell in all_cells
    }

    book, seen_geometry = [], set()
    for bins in binomial_pairs:
        for first, second in combinations(triangles, 2):
            cells = bins + (first, second)
            signature = min(
                tuple(sorted(normal_forms[cell][t] for cell in cells))
                for t in range(len(automorphisms))
            )
            if signature in seen_geometry:
                continue
            seen_geometry.add(signature)
            grid = [[[G.elements[g] for g in cell]
                     for cell in signature]]
            rows, width = component(grid, "A")
            stats = _kernel_statistics(rows, width)
            if stats[0] or stats[1] or gf2_rank(rows) != q:
                continue
            key = (stats[2], stats[3], _four_cycles(rows),
                   _six_cycles(rows), signature)
            book.append((key, grid))
    book.sort(key=lambda item: item[0])
    wanted = 5 * batch + marker
    if wanted >= len(book):
        raise ValueError("Canonical quaternion A contract book exhausted")
    akey, A = book[wanted]

    profile = _B_PROFILES[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2, 2, 3, 3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    assert all(sum(p) == 2 for p in profile)
    assert max(map(sum, sb)) <= 8

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [(1, j) for j in cross[1:]]
    outside_count = 1 + (marker + batch) % (len(cross) - 1)
    outside_sites = set(cross[1:1 + outside_count])
    rotations = [(a, 0) for a in range(1, 8)]
    outside = [(a, 1) for a in range(8)]
    noncentral = rotations[:3] + rotations[4:] + outside
    internal_palette = (
        noncentral,
        [(a, 0) for a in (1, 3, 5, 7)],
        [(a, 0) for a in (1, 3, 5, 7)] + outside,
    )[batch % 3]

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for i, j in internal:
            grid[i][j] = [G.e, rng.choice(internal_palette)]
        rv = rng.sample(rotations, len(cross) - 1 - outside_count)
        ov = rng.sample(outside, outside_count)
        for j in cross:
            grid[0][j] = [G.e]
            g = (G.e if j == cross[0] else
                 ov.pop() if j in outside_sites else rv.pop())
            grid[1][j] = [g]
        return grid

    def assess_B(grid):
        rows, width = component(grid, "B")
        stats = _kernel_statistics(rows, width)
        # B is an incidence matrix. These are parallel-edge, triangle
        # and four-cycle statistics on the actual helper-built graph.
        energy = (1000 * stats[0] + 100 * stats[1] + stats[2]) / q
        return energy, stats, rows

    pool, seen_B = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != 2*q - 1:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen_B:
            return
        seen_B.add(signature)
        serial += 1
        key = (stats[2], stats[3])
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[6:]

    for restart in range(4):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(220):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [G.e, rng.choice(internal_palette)]
            else:
                palette = outside if j in outside_sites else rotations
                occupied = {proposal[1][c][0] for c in cross if c != j}
                choices = [g for g in palette if g not in occupied]
                if not choices:
                    continue
                proposal[i][j] = [rng.choice(choices)]
            proposed, ps, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 4.0 * (0.982 ** step) + 0.04
            if delta <= 0 or rng.random() < exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No connected triangle-free quaternion B contract")

    def quotient_columns(stabilizers, width):
        # Linear canonical remainders modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residual = [0] * width
        for c in range(width):
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

    def logical_risk(checks, stabilizers, width):
        """Exact short-logical classification; hook costs are surrogates.

        A shared opposite-basis check marks possible ancilla spread,
        not a claim about the ordering of the measured ASC circuit.
        """
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        exposure = [a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]
        buckets = {}
        hooks = double_hooks = weighted = partitions = 0

        for i, si in enumerate(syndrome):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                value = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(value)
                if third is not None and third != label:
                    return None
                h = (hi & spread[j]).bit_count()
                w = exposure[i] * exposure[j]
                bucket = buckets.get(value)
                if bucket is None:
                    buckets[value] = [1, h, w, {label: [1, h, w]}]
                    continue
                same = bucket[3].get(label)
                sc, sh, sw = same if same is not None else (0, 0, 0)
                dc = bucket[0] - sc
                dh = bucket[1] - sh
                dw = bucket[2] - sw
                partitions += dc
                hooks += h * dc + dh
                double_hooks += h * dh
                weighted += w * dc + dw
                bucket[0] += 1
                bucket[1] += h
                bucket[2] += w
                if same is None:
                    bucket[3][label] = [1, h, w]
                else:
                    same[0] += 1
                    same[1] += h
                    same[2] += w
        assert partitions % 3 == 0
        return hooks, double_hooks, weighted, partitions // 3

    winner = None
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 480 or k != 243:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        score = tuple(value for t in range(4)
                      for value in (max(px[t], pz[t]), px[t] + pz[t]))
        score += coarse
        if winner is None or score < winner[0]:
            winner = (score, B, hx, hz, px, pz)

    if winner is None:
        raise ValueError("No k=243 quaternion CSS filling passed screening")
    score, B, hx, hz, px, pz = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    probe_a, _ = component(A, "A")
    probe_b, _ = component(B, "B")
    assert actual_a == probe_a and actual_b == probe_b
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert 6 * sum(row.bit_count() for row in hx + hz) == 15360
    assert 19 * 480 <= 40 * 243 < 21 * 480
    return {"A": A, "B": B, "design": (architecture, wanted),
            "voltage_contract": (outside_count, batch % 3),
            "k": 243, "logical_risk": (px, pz),
            "A_contract": akey, "cycle_score": score}


def NEW_CTX(rng, G, idx):
    """Light forest contracts and complete-code noisy-history holdout screening."""
    import hashlib
    import math
    import random
    from collections import Counter
    from itertools import combinations, combinations_with_replacement, permutations

    q = G.q
    assert q == 13
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    budget, a_phase = divmod(architecture, 2)
    # The parent's sixty slots consumed cards 0..14. These twenty
    # contracts use ten different cards per batch, starting at fifteen.
    wanted = 15 + 10 * batch + 5 * a_phase + marker
    sa = [[2, 2, 2, 1, 1], [1, 1, 1, 2, 2]]

    # A single mixed-row degree-two column is a matching. Two such
    # columns on different row pairs form a forest for every voltage.
    weak = (((1, 1, 0), (1, 0, 1)) if budget == 0
            else ((1, 1, 0),))
    strong_types = sorted(
        {(1, 1, 1)} | set(permutations((2, 1, 0))))
    desired_loads = ([7, 7, 8] if budget == 0 else [7, 8, 8])

    def canonical_support(columns):
        # Remove count-card row/column copies. Complete Tanner
        # isomorphism and the run archive remain evaluator decisions.
        return min(
            tuple(sorted(tuple(column[i] for i in order)
                         for column in columns))
            for order in permutations(range(3)))

    support_book = set()
    for tail in combinations_with_replacement(
            strong_types, 8 - len(weak)):
        if not 1 <= tail.count((1, 1, 1)) <= 5:
            continue
        columns_card = weak + tail
        grid = [[column[i] for column in columns_card] for i in range(3)]
        if sorted(map(sum, grid)) != desired_loads:
            continue
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in grid]
        if gf2_rank(augmentation) == 3:
            support_book.add(canonical_support(columns_card))
    support_book = sorted(support_book)
    assert len(support_book) >= 5
    profile = support_book[marker * (len(support_book) - 1) // 4]
    sb = [[p[i] for p in profile] for i in range(3)]
    supports = {"A": sa, "B": sb}
    expected_cnots = 24882 + 546 * budget

    assert tuple(map(sum, sa)) == (8, 7)
    assert sorted(map(sum, sb)) == desired_loads
    assert all(sum(p) == 3 for p in zip(*sa))
    assert sorted(map(sum, profile)) == (
        [2] * (2 - budget) + [3] * (6 + budget))
    assert sum(map(sum, sb)) == 22 + budget
    assert _grid_weights(sa, sb) == 11 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(gf2_rank([
        sum((count & 1) << j for j, count in enumerate(row))
        for row in grid]) == len(grid)
        for grid in (sa, sb))

    def direction(value):
        value %= q
        return min(value, q - value)

    def gap_key(upper, lower):
        # Common group automorphisms and permutations of equal-support
        # columns are gauges, not separate submitted designs.
        return min(
            (tuple(sorted(direction(u * d) for d in upper)),
             tuple(sorted(direction(u * d) for d in lower)))
            for u in range(1, q))

    gap_book = sorted({
        gap_key(upper, lower)
        for upper in combinations(range(1, 7), 3)
        for lower in combinations(range(1, 7), 2)
    })
    if wanted >= len(gap_book):
        raise ValueError("Unused canonical odd-degree A gap cards exhausted")
    upper, lower = gap_book[wanted]
    gaps = [list(upper) + [0, 0], [0, 0, 0] + list(lower)]
    contract = (budget, a_phase, marker, wanted, upper, lower)

    # These singleton actions come only from the verified construction.
    # Final component projections are checked against the complete lift.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz1)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        rows[i * q + t] ^= bits << (j * q)
        return rows, len(grid[0]) * q

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def rooted_identity(rows, width):
        """Conservative Tanner invariant, not complete canonization."""
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
        # Cyclic translations are transitive within each check-row orbit.
        for root in range(0, nr, q):
            colors = [0] * nr + [1] * width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(9):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr + width)]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            images.append(digest.hexdigest())
        return min(images)

    def cycle_metric(rows, width):
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (
                    neighbors & adjacency[j] & ((1 << j) - 1)
                ).bit_count()
                lower ^= bit
        # Subtract the star triangles of the actual mixed-degree columns.
        # Without C4 this gives exact C6; otherwise it is a heuristic.
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count()
                              for c in columns(rows, width)))
        six = max(0, triangles - stars)
        return (80 * four + six / 5.0) / q, four, six

    def sample_words(rows, width, side):
        """Actual kernel witnesses; sampled minima remain upper bounds."""
        srng = random.Random(13035817 + 1009 * idx + (side == "B"))
        cs = columns(rows, width)
        order = list(range(width))
        minimum = width + 1
        short = set()

        def offer(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 10 and len(short) < 8192:
                short.add(word)

        for _ in range(28):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cs[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)

        loads = [0] * (width // q)
        burden = 0
        for word in short:
            cost = (12 - word.bit_count()) ** 2
            burden += cost
            for j in range(len(loads)):
                loads[j] += cost * ((word >> (j * q)) & mask).bit_count()
        return minimum, burden / width, max(loads) / q

    def bp_burden(rows, width, side):
        """Classical component stress test, NOT ASC circuit simulation.

        A settled but wrong solution is a nonzero component kernel word.
        Such settlement is penalized more than unresolved BP, because
        a later decoder stage cannot rescue an undetected miscorrection.
        """
        import numpy as np

        check_edges = [[] for _ in rows]
        variable_edges = [[] for _ in range(width)]
        edge_variables = []
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                edge = len(edge_variables)
                edge_variables.append(c)
                check_edges[r].append(edge)
                variable_edges[c].append(edge)
                row ^= bit
        ne = len(edge_variables)
        ev = np.asarray(edge_variables, dtype=np.int32)
        ce = np.full(
            (len(rows), max(map(len, check_edges))), ne, dtype=np.int32)
        ve = np.full(
            (width, max(map(len, variable_edges))), ne, dtype=np.int32)
        for r, edges in enumerate(check_edges):
            ce[r, :len(edges)] = edges
        for c, edges in enumerate(variable_edges):
            ve[c, :len(edges)] = edges
        valid = ce != ne
        cv = np.asarray(edge_variables + [width], dtype=np.int32)[ce]

        shots = 192
        channel = np.random.default_rng(
            13035919 + 1009 * idx + 7919 * (side == "B"))
        error = np.zeros((shots, width + 1), dtype=bool)
        for shot in range(shots):
            weight = 2 + shot // 64
            error[shot, channel.choice(
                width, weight, replace=False)] = True
        target = np.logical_xor.reduce(error[:, cv], axis=2)
        signs = 1.0 - 2.0 * target.astype(float)
        p = max(0.025, min(0.08, 3.0 / width))
        prior = math.log((1.0 - p) / p)
        v2c = np.full((shots, ne + 1), prior, dtype=float)
        v2c[:, ne] = 30.0
        c2v = np.zeros_like(v2c)
        active = np.ones(shots, dtype=bool)
        ones = np.ones((shots, len(rows), 1))
        wrong = work = 0

        for iteration in range(36):
            work += int(active.sum())
            t = np.tanh(np.clip(v2c[:, ce], -18, 18) / 2)
            t[:, ~valid] = 1.0
            prefix = np.concatenate(
                (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            following = np.concatenate(
                (suffix[:, :, 1:], ones), axis=2)
            outgoing = 2 * np.arctanh(np.clip(
                signs[:, :, None] * prefix * following,
                -0.999999, 0.999999))
            c2v[:, ce[valid]] = outgoing[:, valid]
            posterior = prior + c2v[:, ve].sum(axis=2)
            hard = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)), axis=1)
            matched = np.logical_xor.reduce(hard[:, cv], axis=2)
            settled = active & np.all(matched == target, axis=1)
            incorrect = np.any(
                hard[:, :width] != error[:, :width], axis=1)
            wrong += int(np.count_nonzero(settled & incorrect))
            active[settled] = False
            if not active.any():
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, ev] - c2v[:, :ne]
            v2c[:, :ne] = (
                damping * v2c[:, :ne] + (1 - damping) * proposal)

        unresolved = int(active.sum())
        return ((2 * wrong + unresolved) / shots,
                wrong / shots, unresolved / shots,
                work / (36.0 * shots))

    def portfolio(side):
        support = supports[side]
        nr, nc = len(support), len(support[0])
        anchors = [
            next(i for i in range(nr) if support[i][j])
            for j in range(nc)]

        if side == "A":
            sites = [(i, j) for i in range(nr) for j in range(nc)
                     if support[i][j] == 2]
        else:
            sites = [(i, j) for i in range(nr) for j in range(nc)
                     if support[i][j] and not (
                         i == anchors[j] and support[i][j] == 1)]

        def new_cell(i, j):
            count = support[i][j]
            if side == "A":
                if count == 1:
                    return [G.e]
                origin = rng.randrange(q)
                gap = gaps[i][j]
                return [(origin,), ((origin + gap) % q,)]
            if i == anchors[j]:
                return [G.e] + rng.sample(
                    [g for g in G.elements if g != G.e], count - 1)
            return rng.sample(G.elements, count)

        def draw():
            return [[new_cell(i, j) if support[i][j] else []
                     for j in range(nc)] for i in range(nr)]

        pool, labels, classes = [], set(), set()
        serial = 0

        def remember(grid, rows, metric):
            nonlocal serial
            key = (metric[1], metric[2])
            if len(pool) == 6 and key > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in labels:
                return
            labels.add(signature)
            width = nc * q
            stats = _kernel_statistics(rows, width)
            if any(stats[:3]) or gf2_rank(rows) != nr * q:
                return
            if side == "B":
                weak_columns = [
                    c for c in columns(rows, width) if c.bit_count() == 2]
                if gf2_rank(weak_columns) != len(weak_columns):
                    return
            identity = rooted_identity(rows, width)
            if identity in classes:
                return
            classes.add(identity)
            serial += 1
            pool.append((key, serial, clone(grid), list(rows), identity))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(4):
            grid = draw()
            rows, width = component(grid, side)
            current = cycle_metric(rows, width)
            remember(grid, rows, current)
            for step in range(440):
                proposal = clone(grid)
                i, j = rng.choice(sites)
                proposal[i][j] = new_cell(i, j)
                trial_rows, _ = component(proposal, side)
                trial = cycle_metric(trial_rows, width)
                delta = trial[0] - current[0]
                temperature = 10.0 * (0.988 ** step) + 0.05
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, rows, current = proposal, trial_rows, trial
                    if delta <= 0 or step % 37 == 0:
                        remember(grid, rows, current)

        if not pool:
            # Fresh complete proposals preserve the same legal contract.
            # No submitted illegal cell is repaired.
            for _ in range(384):
                grid = draw()
                rows, width = component(grid, side)
                remember(grid, rows, cycle_metric(rows, width))
                if len(pool) >= 3:
                    break
        if not pool:
            raise ValueError("No full-rank component excluded kernels <=4")

        finalists = []
        for coarse, serial, grid, rows, identity in pool:
            width = nc * q
            cs = columns(rows, width)
            degrees = [sum(row[j] for row in support) for j in range(nc)]
            assert all(c.bit_count() == degrees[j // q]
                       for j, c in enumerate(cs))
            # remember() excluded kernels <=4, so these mixed-degree
            # five/six counts are exact component counts.
            five, six = _kernel56(rows, width)
            distance, burden, peak = sample_words(rows, width, side)
            bp = bp_burden(rows, width, side)
            key = (int(five > 0), int(six > 0),
                   (five + six) / width, bp[0], bp[1],
                   max(0, 8 - distance), burden, peak,
                   bp[2], bp[3], coarse)
            finalists.append({
                "grid": grid, "rows": rows, "identity": identity,
                "key": key, "five_count": five,
                "six": six / width, "six_count": six,
                "bp": bp, "distance_upper": distance,
                "burden": burden, "peak": peak, "coarse": coarse})
        finalists.sort(key=lambda item: item["key"])
        return finalists[:4]

    def no_logicals4(checks, stabilizers, width):
        """Exact stabilizer-quotiented exclusion through weight four."""
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
            else:
                tail, value = row ^ (1 << c), 0
                while tail:
                    bit = tail & -tail
                    value ^= residue[bit.bit_length() - 1]
                    tail ^= bit
                residue[c] = value

        cs = columns(checks, width)
        seen = {0: 0}
        for i, syndrome in enumerate(cs):
            label = residue[i]
            old = seen.get(syndrome)
            if old is not None and old != label:
                return False
            seen[syndrome] = label
        for i, syndrome in enumerate(cs):
            for j in range(i):
                value = syndrome ^ cs[j]
                label = residue[i] ^ residue[j]
                old = seen.get(value)
                if old is not None and old != label:
                    return False
                seen[value] = label
        return True

    def history_burden(checks, stabilizers, n, seed, shots):
        """Six-interval noisy-syndrome benchmark, NOT an ASC circuit.

        Difference equations are H*e_r+m_r+m_(r-1)=delta_s_r.
        Boundary measurement errors m_0 and m_6 are zero: five
        intermediate observations are noisy and the last is perfect.
        Correlated faults use random proper opposite-check suffixes,
        not a claimed ASC gate order. BP is not relay-BP or LSD.
        """
        import numpy as np

        rounds, m = 6, len(checks)
        width = rounds * n + (rounds - 1) * m
        augmented = []
        for interval in range(rounds):
            for r, row in enumerate(checks):
                word = row << (interval * n)
                if interval < rounds - 1:
                    word |= 1 << (rounds * n + interval * m + r)
                if interval:
                    word |= 1 << (rounds * n + (interval - 1) * m + r)
                augmented.append(word)

        cs = columns(augmented, width)
        check_edges = [[] for _ in augmented]
        variable_edges = [[] for _ in range(width)]
        edge_variables = []
        for v, syndrome in enumerate(cs):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                edge = len(edge_variables)
                edge_variables.append(v)
                check_edges[r].append(edge)
                variable_edges[v].append(edge)
                syndrome ^= bit
        ne, nr = len(edge_variables), len(augmented)
        ev = np.asarray(edge_variables, dtype=np.int32)
        ce = np.full(
            (nr, max(map(len, check_edges))), ne, dtype=np.int32)
        ve = np.full(
            (width, max(map(len, variable_edges))), ne, dtype=np.int32)
        for r, edges in enumerate(check_edges):
            ce[r, :len(edges)] = edges
        for v, edges in enumerate(variable_edges):
            ve[v, :len(edges)] = edges
        valid = ce != ne
        cv = np.asarray(edge_variables + [width], dtype=np.int32)[ce]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        channel = np.random.default_rng(seed)
        levels = np.asarray((0.40, 0.65, 0.90))[
            np.arange(shots) % 3]
        degree = np.asarray([
            c.bit_count() for c in columns(checks + stabilizers, n)
        ], dtype=float)
        single_mu = 0.002 * (2.0 / 3.0) * degree
        single_p = 0.5 * (
            1.0 - np.exp(-2.0 * levels[:, None] * single_mu))
        error = np.zeros((shots, width + 1), dtype=bool)
        for interval in range(rounds):
            start = interval * n
            error[:, start:start + n] = (
                channel.random((shots, n)) < single_p)

        # A fixed random order per opposite check, reused over intervals.
        # Marginal priors account for each site's actual suffix exposure.
        hook_mu = np.zeros(n, dtype=float)
        for row in stabilizers:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length() - 1)
                row ^= bit
            if len(sites) < 2:
                continue
            order = channel.permutation(sites)
            base = 0.002 * (len(sites) + 1) / 3.0
            hook_mu[order] += (
                base * np.arange(len(sites)) / (len(sites) - 1))
            for interval in range(rounds):
                events = np.flatnonzero(
                    channel.random(shots) < levels * base)
                for shot in events:
                    cut = int(channel.integers(1, len(sites)))
                    error[shot, interval * n + order[cut:]] ^= True

        measurement_mu = 0.002 * (2.0 / 3.0) * np.asarray([
            row.bit_count() + 2 for row in checks], dtype=float)
        measurement_p = 0.5 * (
            1.0 - np.exp(
                -2.0 * levels[:, None] * measurement_mu))
        for interval in range(rounds - 1):
            start = rounds * n + interval * m
            error[:, start:start + m] = (
                channel.random((shots, m)) < measurement_p)

        marginal_p = np.clip(0.5 * (
            1.0 - np.exp(
                -2.0 * levels[:, None] * (single_mu + hook_mu))),
            0.001, 0.20)
        probabilities = np.concatenate(
            [marginal_p] * rounds
            + [np.clip(measurement_p, 0.001, 0.20)] * (rounds - 1),
            axis=1)
        prior = np.log((1.0 - probabilities) / probabilities)
        target = np.logical_xor.reduce(error[:, cv], axis=2)
        signs = 1.0 - 2.0 * target.astype(float)
        true_final = np.logical_xor.reduce(
            error[:, :rounds * n].reshape(shots, rounds, n), axis=1)

        v2c = np.empty((shots, ne + 1), dtype=float)
        v2c[:, :ne] = prior[:, ev]
        v2c[:, ne] = 30.0
        c2v = np.zeros_like(v2c)
        active = np.ones(shots, dtype=bool)
        ones = np.ones((shots, nr, 1))
        wrong = work = 0
        early = None
        iterations = 36

        for iteration in range(iterations):
            work += int(active.sum())
            t = np.tanh(np.clip(v2c[:, ce], -18.0, 18.0) / 2.0)
            t[:, ~valid] = 1.0
            prefix = np.concatenate(
                (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            following = np.concatenate(
                (suffix[:, :, 1:], ones), axis=2)
            outgoing = 2.0 * np.arctanh(np.clip(
                signs[:, :, None] * prefix * following,
                -0.999999, 0.999999))
            c2v[:, ce[valid]] = outgoing[:, valid]
            posterior = prior + c2v[:, ve].sum(axis=2)
            hard = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)),
                axis=1)
            matched = np.logical_xor.reduce(hard[:, cv], axis=2)
            settled = active & np.all(matched == target, axis=1)
            estimated_final = np.logical_xor.reduce(
                hard[:, :rounds * n].reshape(shots, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                residual = estimated_final[shot] ^ true_final[shot]
                word = sum(1 << int(v)
                           for v in np.flatnonzero(residual))
                wrong += int(nontrivial(word))
            active[settled] = False
            if iteration == 11:
                early = int(active.sum())
            if not active.any():
                if early is None:
                    early = 0
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, ev] - c2v[:, :ne]
            v2c[:, :ne] = (
                damping * v2c[:, :ne] + (1.0 - damping) * proposal)

        unresolved = int(active.sum())
        if early is None:
            early = unresolved
        return ((wrong + unresolved) / shots, wrong / shots,
                early / shots, work / (iterations * shots))

    def history_quality(x, z):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                max(x[2], z[2]), x[2] + z[2], x[3] + z[3])

    apool, bpool = portfolio("A"), portfolio("B")
    prepared = []
    serial = 0
    for a in apool:
        for b in bpool:
            guard = (int(a["five_count"] + b["five_count"] > 0),
                     int(a["six_count"] + b["six_count"] > 0))
            coarse = (
                max(a["six"], b["six"]), a["six"] + b["six"],
                max(a["bp"][0], b["bp"][0]),
                a["bp"][0] + b["bp"][0],
                max(a["bp"][1], b["bp"][1]),
                max(0, 8 - a["distance_upper"])
                + max(0, 8 - b["distance_upper"]),
                max(a["burden"], b["burden"]),
                a["burden"] + b["burden"],
                max(a["peak"], b["peak"]),
                a["coarse"], b["coarse"])
            prepared.append((guard + coarse, serial, guard, coarse, a, b))
            serial += 1
    prepared.sort(key=lambda item: item[:2])

    # Component evidence shortlists pairings, but does not make the
    # final circuit-risk decision. Banks are common across alternatives.
    seed = 130356701 + 1009 * idx
    training = []
    for _, serial, guard, coarse, a, b in prepared:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 598 or k != 195:
            continue
        if not no_logicals4(hx, hz, n):
            continue
        if not no_logicals4(hz, hx, n):
            continue
        actual_a, actual_b = _component_rows(hx, hz, q)
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 11 <= W_MAX
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        tx = history_burden(hz, hx, n, seed, 36)
        tz = history_burden(hx, hz, n, seed + 1, 36)
        key = guard + history_quality(tx, tz) + coarse + (serial,)
        training.append(
            (key, serial, guard, coarse, a, b, hx, hz, tx, tz))
        if len(training) == 3:
            break
    training.sort(key=lambda item: item[:2])

    finalists = []
    for entry in training[:2]:
        _, serial, guard, coarse, a, b, hx, hz, tx, tz = entry
        # Independent holdout leads selection; training breaks ties.
        x = history_burden(hz, hx, 598, seed + 1000003, 72)
        z = history_burden(hx, hz, 598, seed + 1000004, 72)
        key = (guard + history_quality(x, z)
               + history_quality(tx, tz) + coarse + (serial,))
        finalists.append((key, serial, a, b, x, z, tx, tz))
    finalists.sort(key=lambda item: item[:2])

    if not finalists:
        raise ValueError("No verified k=195 forest CSS finalist")
    key, serial, a, b, x, z, tx, tz = finalists[0]
    A, B = a["grid"], b["grid"]
    assert lifted_k(G, A, B) == 195
    assert all(len(cell) <= CELL_MAX and
               len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    weak_columns = [
        c for c in columns(b["rows"], 8 * q) if c.bit_count() == 2]
    assert len(weak_columns) == (2 - budget) * q
    assert gf2_rank(weak_columns) == len(weak_columns)
    return {"A": A, "B": B, "k": 195, "contract": contract,
            "component_identities": (a["identity"], b["identity"]),
            "component_five_six_counts":
                ((a["five_count"], a["six_count"]),
                 (b["five_count"], b["six_count"])),
            "component_bp": (a["bp"], b["bp"]),
            "training_history_burden": (tx, tz),
            "holdout_history_burden": (x, z),
            "selection_score": key, "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete holdout-selected forest-contract cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: A<=3/B<=4 component exclusion balances C19 (1,4)x(2,7) circuit memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored 3.1933 despite substantially lower Z failed-shot rates.
# REFUTED: Noncommuting Q16 A geometries, triangle-free graphic B and shared-check quartet minimization suffice for rare high-rate memory failures: fitness was 3.2584, measured Z failures were 122..141/250, and collective flips affected essentially all 243 logicals despite first-pass Z settlement.
# REFUTED: Equal degree-three components and weight-eight witnesses imply comparable X/Z circuit failure rates on C13: all five current-parent codes were X-limited, with 100..102 X failures in 11750..16750 shots versus 100..104 Z failures in 28500..43000 shots, and failures reached all 195 logical qubits.
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
