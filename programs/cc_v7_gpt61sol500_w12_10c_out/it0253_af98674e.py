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
CONCEPT = ("Frozen C7:C4, action 6, order 28, A=(1,4), B=(1,5): aligned singular-cell row dependencies may yield k>345 while excluding logical weights below four, or exclude all logicals through four; either outcome would challenge the measured [588,345,4] plateau.")
GROUP = ("Keep the exact L1 group and order. This bed measured five [588,345,4] leaders with fitness 9.388. Elements of orders 2,4,7,14 furnish different singular-block geometries. Their ranks and dependency spaces are measured on verified lifts; no distance advantage is assumed.")
SHAPES = ("Keep (1,4)x(1,5) for open-ended cell 0.550: n=588, kb=336, guaranteed rate 4/7, and maximum admissible lift order 28. Components have 112/140 variables and 28 checks each. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly, especially Acol2 and Bcol4; listed minima did not require Arow/Brow blocks.")
FILLING = ("Build singular-cell libraries from verified left/right blocks and cross legal cardinality architectures with four dependency-dimension targets. Also search inverse-pair occupancy strata including triangular graphs and involutory relatives, leaving acceptance to exact stabilizer-aware logical screening. Blacklist displayed parent geometries and retain unequal uncoloured incidence signatures. Select by verified k times squared exact short-logical floor; floor five means exclusion through four, not a distance reading. Fresh floor-three designs are lower-priority fallbacks.")
FAMILY_NAME = "r055_m7r4s6_14x15_dependencystrata_rankflex_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Same metacyclic semidirect construction, with new parameters.
    # C7:C4: y*x*y^-1=x^6; 6^4=1 mod 7.
    if not valid_q(q):
        raise ValueError("C7:C4 requires order 28")
    m, r, s = 7, 4, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)

from itertools import combinations, combinations_with_replacement
from functools import lru_cache


def _unsigned_step(x, m):
    x %= m
    return min(x, (-x) % m)


def _normal_side(side, u, m):
    steps, tri = side
    ds = tuple(sorted(_unsigned_step(u * d, m) for d in steps))
    if tri[0] == 0:
        # Mixed trinomial: normalize its reflection by conjugation.
        ts = (0, _unsigned_step(u * tri[1], m))
    else:
        # Rotation trinomial: remove translation and independent inversion.
        support = tri[1:]
        ts = (1,) + min(
            tuple(sorted((sgn * u * (x - base)) % m for x in support))
            for base in support for sgn in (1, -1))
    return ds, ts


def _canonical_plan(a, b, m):
    # Remove common group automorphisms, column permutations, and side swap.
    return min(
        tuple(sorted((_normal_side(a, u, m), _normal_side(b, u, m))))
        for u in range(1, m))


def _plan_cells(side):
    steps, tri = side
    cells = [((0, 0), (d, 0)) for d in steps]
    if tri[0] == 0:
        cells.append(((0, 0), (tri[1], 0), (0, 1)))
    else:
        cells.append(tuple((a, 0) for a in tri[1:]))
    return (tuple(cells),)


def _small_kernel_stats(rows, n):
    """Exact commuting-vector screening through weight three.

    With distinct nonzero column syndromes, equal pair syndromes
    involve disjoint pairs. Every weight-four kernel vector has
    exactly three pair partitions, permitting exact counting.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= bit
            row ^= lo

    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None

    pairs = {}
    for i, a in enumerate(cols):
        for j in range(i + 1, n):
            syndrome = a ^ cols[j]
            if syndrome in singles:
                return None
            packed = i * n + j
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = packed
            elif isinstance(previous, int):
                pairs[syndrome] = [previous, packed]
            else:
                previous.append(packed)

    histogram = {}
    participation = [0] * n
    collisions = 0
    for syndrome, value in pairs.items():
        count = 1 if isinstance(value, int) else len(value)
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
        if count > 1:
            collisions += count * (count - 1) // 2
            for packed in value:
                i, j = divmod(packed, n)
                participation[i] += count - 1
                participation[j] += count - 1

    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in cols),
            tuple(p // 3 for p in participation))


def _incidence_fingerprint(hx, hz, sx, sz):
    # Invariant under qubit/check permutations and a global X/Z swap.
    joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
    swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
    row_profiles = tuple(sorted((
        tuple(sorted(r.bit_count() for r in hx)),
        tuple(sorted(r.bit_count() for r in hz)))))
    return (tuple(sorted((sx[1], sz[1]))),
            min(joint, swapped), row_profiles)


def _binomial_canonical(G, left, right):
    """Remove known gauges before the actual graph-invariant screen.

    x -> x^u, y -> x^v*y are automorphisms of C7:C4.
    Replacing a binomial relative by its inverse is a column gauge.
    These operations generate candidates, never a second matrix lift.
    """
    def image(side, u, v):
        result = []
        for a, b in side:
            g = ((u*a + v*(b & 1)) % 7, b)
            result.append(min(G.idx(g), G.idx(G.inv(g))))
        return tuple(sorted(result))

    return min((image(left, u, v), image(right, u, v))
               for u in range(1, 7) for v in range(7))


def _binomial_graph_profile(hx, hz, n):
    """Uncoloured combined Tanner invariant.

    Different profiles certify nonisomorphism; equal profiles are
    conservatively discarded, not asserted to be isomorphic.
    """
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    neighbours = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if not overlap:
                continue
            a = (weights[j], overlap)
            b = (weights[i], overlap)
            neighbours[i][a] = neighbours[i].get(a, 0) + 1
            neighbours[j][b] = neighbours[j].get(b, 0) + 1

    checks = tuple(sorted(
        (weights[i], tuple(sorted(profile.items())))
        for i, profile in enumerate(neighbours)))
    qubits = [[] for _ in range(n)]
    for weight, source in zip(weights, rows):
        while source:
            bit = source & -source
            qubits[bit.bit_length() - 1].append(weight)
            source ^= bit
    return checks, tuple(sorted(tuple(sorted(p)) for p in qubits))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Pure deterministic architecture search on the new shape bed."""
    assert G.q == 28
    search = random.Random(550280145)
    representatives = [
        g for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
        and G.idx(g) < G.idx(G.inv(g))
    ]

    def library(count):
        buckets = {}
        for side in combinations(representatives, count):
            signed = set(side)
            signed.update(G.inv(g) for g in side)
            # A label-architecture criterion: inverse-distinct Cayley
            # edges with no triangular relation. The final exclusion
            # screen still uses the scaffold's complete HX/HZ matrices.
            if any(G.mul(g, h) in signed
                   for g in signed for h in signed):
                continue
            occupancy = (
                sum(g[1] == 0 for g in side),
                sum(g[1] == 2 for g in side),
                sum(g[1] & 1 for g in side))
            buckets.setdefault(occupancy, []).append(side)
        for values in buckets.values():
            search.shuffle(values)
        return buckets

    alib, blib = library(4), library(5)
    if not alib or not blib:
        raise ValueError("empty triangle-free component library")

    # The all-odd control is evaluated first. Subsequent buckets change
    # the number of rotation, even-quotient and odd-quotient generators.
    strata = sorted(
        ((a, b) for a in alib for b in blib),
        key=lambda p: (p != ((0, 0, 4), (0, 0, 5)), p))
    pair_buckets = []
    for a, b in strata:
        pairs = [(left, right)
                 for left in alib[a] for right in blib[b]]
        search.shuffle(pairs)
        pair_buckets.append(pairs)

    budget = max(320, 8 * N_SLOTS)
    plans, known = [], set()
    # Preserve the all-odd control before round-robin occupancy policies.
    for left, right in pair_buckets[0]:
        key = _binomial_canonical(G, left, right)
        if key not in known:
            known.add(key)
            plans.append(key)

    positions = [0] * len(pair_buckets)
    while len(plans) < budget:
        added = False
        for t, pairs in enumerate(pair_buckets):
            while positions[t] < len(pairs):
                left, right = pairs[positions[t]]
                positions[t] += 1
                key = _binomial_canonical(G, left, right)
                if key in known:
                    continue
                known.add(key)
                plans.append(key)
                added = True
                break
            if len(plans) >= budget:
                break
        if not added:
            break

    keep = {}
    for plan in plans[:budget]:
        A = (tuple((G.e, G.elements[t]) for t in plan[0]),)
        B = (tuple((G.e, G.elements[t]) for t in plan[1]),)
        sa = [[len(cell) for cell in A[0]]]
        sb = [[len(cell) for cell in B[0]]]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        sx = _small_kernel_stats(hx, n)
        if sx is None:
            continue
        sz = _small_kernel_stats(hz, n)
        if sz is None:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        assert n == 588 and k >= 336

        # Check degrees 10 and 12 intrinsically separate the two types
        # here, so the existing X/Z-swap invariant is also safe against
        # uncoloured Tanner relabelings. Strengthen it with overlaps.
        fingerprint = (
            _incidence_fingerprint(hx, hz, sx, sz),
            _binomial_graph_profile(hx, hz, n))
        floor = 5 if sx[0] == sz[0] == 0 else 4
        # The floor is an exhaustive kernel-screen certificate only.
        # Weight-four kernels are not automatically called logicals.
        objective = (-k * floor * floor, sx[0] + sz[0], plan)
        previous = keep.get(fingerprint)
        if previous is None or objective < previous[0]:
            keep[fingerprint] = (objective, A, B)

    selected = sorted(keep.values(), key=lambda item: item[0])[:N_SLOTS]
    if len(selected) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct screened designs")

    bank = []
    for objective, A, B in selected:
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert n - gf2_rank(hx) - gf2_rank(hz) >= 336
        assert all(len(cell) == len(set(cell)) == 2
                   for grid in (A, B) for row in grid for cell in row)
        bank.append((A, B))
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Shared deterministic search; caching does not affect final cells."""
    if idx >= 0:
        bank = NEW_CTX(None, G, -1)
        if idx >= len(bank):
            raise ValueError("no further fresh invariant-distinct design")
        A, B = bank[idx]
        return {"A": A, "B": B}

    from collections import Counter
    assert G.q == 28
    local = random.Random(BASE_SEED * 100003 + 28055097)
    elements = tuple(G.elements)
    central = (0, 2)

    def columns(rows, n):
        result = [0] * n
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                result[bit.bit_length() - 1] |= 1 << r
                source ^= bit
        return result

    def logical_floor(checks, stabilizers, n):
        """Exact logical existence through four, allowing stabilizers.

        Return 5 only when all logical weights <=4 were excluded.
        This sentinel is a lower screening floor, not an upper reading.
        """
        basis = {}
        for source in stabilizers:
            word = source
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old

        # Linear normal forms of unit vectors modulo the row span.
        residues = []
        for c in range(n):
            row = basis.get(c)
            if row is None:
                residues.append(1 << c)
                continue
            tail, residue = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                residue ^= residues[bit.bit_length() - 1]
                tail ^= bit
            residues.append(residue)

        cols = columns(checks, n)
        singles = {0: 0}
        for syndrome, residue in zip(cols, residues):
            if syndrome == 0 and residue:
                return 1
            previous = singles.get(syndrome)
            if previous is None:
                singles[syndrome] = residue
            elif previous != residue:
                return 2

        # Pair/single collisions cover every logical through weight three.
        for i, syndrome in enumerate(cols):
            ri = residues[i]
            for j in range(i):
                previous = singles.get(syndrome ^ cols[j])
                if previous is not None and previous != (ri ^ residues[j]):
                    return 3

        # After the preceding exclusions, one pair representative per
        # syndrome suffices: equal quotient residues differ by stabilizers.
        seen = dict(singles)
        for i, syndrome in enumerate(cols):
            ri = residues[i]
            for j in range(i):
                key = syndrome ^ cols[j]
                residue = ri ^ residues[j]
                previous = seen.get(key)
                if previous is None:
                    seen[key] = residue
                elif previous != residue:
                    return 4
        return 5

    def graph_signature(hx, hz, n):
        """Uncoloured Tanner invariant, independent of X/Z check labels.

        Unequal signatures certify nonisomorphism. Equal signatures are
        discarded conservatively; equality is not an isomorphism claim.
        """
        rows = list(hx) + list(hz)
        degrees = [row.bit_count() for row in rows]
        links = [[] for _ in rows]
        adjacency = [0] * len(rows)
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    links[i].append((j, overlap))
                    links[j].append((i, overlap))
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        initial = []
        for i, neighbours in enumerate(adjacency):
            triangles, remaining = 0, neighbours
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() - 1
                triangles += (neighbours & adjacency[j]).bit_count()
                remaining ^= bit
            profile = Counter((degrees[j], overlap)
                              for j, overlap in links[i])
            initial.append((degrees[i], triangles // 2,
                            tuple(sorted(profile.items()))))

        history = []
        keys = initial
        for _ in range(3):
            histogram = Counter(keys)
            history.append(tuple(sorted(histogram.items())))
            palette = {key: c for c, key in enumerate(sorted(histogram))}
            colours = [palette[key] for key in keys]
            keys = [
                (colours[i], tuple(sorted(Counter(
                    (colours[j], overlap)
                    for j, overlap in links[i]).items())))
                for i in range(len(rows))
            ]

        qubits = [[] for _ in range(n)]
        for degree, source in zip(degrees, rows):
            while source:
                bit = source & -source
                qubits[bit.bit_length() - 1].append(degree)
                source ^= bit
        return (tuple(history),
                tuple(sorted(tuple(sorted(p)) for p in qubits)))

    def cells_from_plan(plan):
        return tuple((tuple((G.e, elements[t]) for t in side),)
                     for side in plan)

    # The current parent's final cells belong to this entire all-odd
    # universe. Blacklist its matrix invariants, including orbit images.
    odd = [(a, 1) for a in range(7)]
    old_plans = {
        _binomial_canonical(G, a, b)
        for a in combinations(odd, 4)
        for b in combinations(odd, 5)
    }
    archived = set()
    for plan in sorted(old_plans):
        A, B = cells_from_plan(plan)
        hx, hz, n = lifted_matrices(G, A, B)
        archived.add(graph_signature(hx, hz, n))

    # Displayed current/earlier even-relative geometries. This supplements
    # the all-odd blacklist; equality is conservatively rejected.
    known_cards = (
        "10 01 21 41|30 11 21 31 32",
        "10 01 31 12|01 11 21 41 51",
        "10 01 21 12|30 01 11 21 32",
        "01 11 22 32|01 11 21 41 51",
        "10 01 21 41|10 01 31 51 12",
        "10 01 31 12|30 01 11 21 32",
        "01 11 21 32|01 11 21 41 51",
        "01 11 21 31|30 01 11 21 32",
        "10 01 31 22|20 01 31 41 22",
        "10 01 31 22|10 01 31 51 12",
        "10 01 21 41|01 21 41 51 61",
        "01 12 22 32|01 11 21 31 41",
        "01 11 21 41|10 01 21 41 12",
        "10 01 31 12|10 01 21 41 12",
        "10 01 31 12|20 11 21 51 22",
        "01 11 21 31|10 01 21 41 12",
        "10 01 21 12|10 01 21 41 12",
        "01 11 21 32|01 11 31 51 61",
        "10 01 31 22|10 01 21 41 12",
        "10 01 31 22|30 01 11 21 32",
        "01 12 22 32|10 01 21 41 12",
        "10 01 21 41|10 01 21 41 12",
        "10 01 21 41|20 01 11 41 22",
        "01 11 22 32|01 11 21 31 41",
        "10 01 21 41|01 11 21 31 41",
    )
    for card in known_cards:
        grids = []
        for text in card.split("|"):
            grids.append((tuple(
                (G.e, (int(token[0]), int(token[1])))
                for token in text.split()),))
        hx, hz, n = lifted_matrices(G, grids[0], grids[1])
        archived.add(graph_signature(hx, hz, n))

    def dependency_space(rows):
        """Canonical dependencies between rows of a verified block."""
        pivots, dependencies = {}, []
        for r, original in enumerate(rows):
            word, coefficient = original, 1 << r
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (word, coefficient)
                    break
                word ^= previous[0]
                coefficient ^= previous[1]
            if not word:
                dependencies.append(coefficient)

        basis = {}
        for word in dependencies:
            while word:
                p = word.bit_length() - 1
                previous = basis.get(p)
                if previous is None:
                    basis[p] = word
                    break
                word ^= previous
        for p in sorted(basis):
            for higher in sorted(basis):
                if higher > p and ((basis[higher] >> p) & 1):
                    basis[higher] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    # A single scaffold lift supplies both action conventions. Empty
    # auxiliary cells are extraction devices, never submitted fillings.
    q = G.q
    mask = (1 << q) - 1
    singular = {(side, count): {}
                for side in ("A", "B") for count in range(2, 6)}
    supports = {
        2: [(G.e, g) for g in elements if g != G.e],
        3: [(G.e, elements[a], elements[b])
            for a, b in combinations(range(1, q), 2)],
    }
    for count in (4, 5):
        supports[count] = sorted({
            tuple(sorted((G.e,) + tuple(local.sample(
                elements[1:], count - 1))))
            for _ in range(256)
        })

    anchors = (G.e, (1, 0), (0, 1), (1, 1))
    for count in range(2, 6):
        for support in supports[count]:
            for anchor in anchors:
                cell = tuple(sorted(G.mul(anchor, g) for g in support))
                A = ((cell, (), (), ()),)
                B = ((cell, (), (), (), ()),)
                hx, hz, _ = lifted_matrices(G, A, B)
                for side, rows in (
                        ("A", [row & mask for row in hx[:q]]),
                        ("B", [row & mask for row in hz[:q]])):
                    space = dependency_space(rows)
                    if not space or len(space) == q:
                        continue
                    incidence = tuple(sorted(columns(rows, q)))
                    bucket = singular[(side, count)].setdefault(space, {})
                    if incidence not in bucket and len(bucket) < 12:
                        bucket[incidence] = cell

    def locked_side(counts, side, policy):
        """Target actual shared dependencies, not element-label equality."""
        target = (1, 2, 4, 7)[policy]
        available = [
            (count, space)
            for count in set(counts) if count > 1
            for space in singular[(side, count)]
        ]
        if not available:
            return None
        near = min(abs(len(space) - target)
                   for _, space in available)
        references = [space for _, space in available
                      if abs(len(space) - target) == near]
        reference = local.choice(references)
        row, used = [], set()
        for count in counts:
            if count == 1:
                row.append((local.choice(elements),))
                continue
            buckets = singular[(side, count)]
            if not buckets:
                return None

            def affinity(space):
                intersection = (len(reference) + len(space)
                                - gf2_rank(reference + space))
                return (intersection, -abs(len(space) - target))

            best = max(affinity(space) for space in buckets)
            spaces = [space for space in buckets
                      if affinity(space) == best]
            space = local.choice(spaces)
            pool = list(buckets[space].values())
            fresh = [cell for cell in pool if cell not in used]
            cell = local.choice(fresh if fresh else pool)
            used.add(cell)
            row.append(cell)
        return (tuple(row),)

    keep = {}

    def admit(A, B):
        sa = [[len(c) for c in A[0]]]
        sb = [[len(c) for c in B[0]]]
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            return
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            return

        fx = logical_floor(hz, hx, n)
        if fx < 3:
            return
        fz = logical_floor(hx, hz, n)
        floor = min(fx, fz)
        if floor < 3:
            return
        # Floor-three designs are fresh fallbacks only. The objective
        # below keeps every comparable floor-four/five improvement ahead
        # of them; never silently refill an exhausted slot with an orbit.

        signature = graph_signature(hx, hz, n)
        if signature in archived:
            return
        objective = (k * floor * floor, floor, k, fx + fz,
                     -sum(row.bit_count() ** 2 for row in hx + hz))
        previous = keep.get(signature)
        if previous is None or objective > previous[0]:
            frozen = tuple(
                tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for grid in (A, B))
            keep[signature] = (objective, frozen[0], frozen[1])

    # Deliberate support architectures, not a single rolled count grid.
    profiles = []
    for aw in combinations_with_replacement(range(1, 6), 4):
        for bw in combinations_with_replacement(range(1, 6), 5):
            if sum(aw) + max(bw) > W_MAX:
                continue
            if sum(bw) + max(aw) > W_MAX:
                continue
            if aw == (2, 2, 2, 2) and bw == (2, 2, 2, 2, 2):
                continue
            profiles.append((aw, bw))
    profiles.sort(key=lambda p: (
        p[0].count(1) + p[1].count(1),
        -sum(p[0]) - sum(p[1]), p))

    def draw_side(counts, model, phase):
        row = []
        for j, count in enumerate(counts):
            if model == 0:
                cell = local.sample(elements, count)
            elif model == 1:
                # Complete central-involution pairs plus an optional
                # unpaired term. Short stabilizers are allowed by design.
                cell = []
                while len(cell) + 2 <= count:
                    available = [
                        g for g in elements
                        if g not in cell and G.mul(g, central) not in cell
                    ]
                    g = local.choice(available)
                    cell.extend((g, G.mul(g, central)))
                if len(cell) < count:
                    cell.append(local.choice(
                        [g for g in elements if g not in cell]))
            else:
                # Explicit parity-occupancy changes, not common gauges.
                amount = (phase + j) % (count + 1)
                even = [g for g in elements if not (g[1] & 1)]
                odd_pool = [g for g in elements if g[1] & 1]
                cell = (local.sample(even, amount)
                        + local.sample(odd_pool, count - amount))
            assert len(cell) == len(set(cell))
            row.append(cell)
        return [row]

    for card, (aw, bw) in enumerate(profiles):
        for attempt in range(16):
            model = attempt % 3
            admit(draw_side(aw, model, card + attempt),
                  draw_side(bw, model, 2 * card + attempt + 1))
        for policy in range(4):
            for attempt in range(4):
                A = locked_side(aw, "A", policy)
                B = locked_side(bw, "B", policy)
                if A is not None and B is not None:
                    admit(A, B)

    # Include involutory and triangular controls. Their small kernels
    # must be classified modulo the actual stabilizer spans, rather
    # than rejected solely from a Cayley-label relation.
    representatives = [
        g for g in elements if g != G.e
        and G.idx(g) <= G.idx(G.inv(g))
    ]
    orders = {}
    for g in representatives:
        value = G.e
        for exponent in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break

    def library(count):
        buckets = {}
        for side in combinations(representatives, count):
            occupancy = tuple(
                sum(orders[g] == order for g in side)
                for order in (2, 4, 7, 14))
            buckets.setdefault(occupancy, []).append(side)
        for values in buckets.values():
            local.shuffle(values)
        return buckets

    alib, blib = library(4), library(5)
    strata = [(a, b) for a in alib for b in blib
              if not (a[1] == 4 and b[1] == 5)]
    # Even-relative controls can have disconnected component lifts,
    # hence larger rank deficits. Their triangles may still be logical;
    # the full verified screen, not this priority, decides acceptance.
    strata.sort(key=lambda p: (
        p[0][1] + p[1][1],
        p[0][0] + p[1][0], p))
    budget = max(1536, 48 * N_SLOTS)
    plans = set()
    for sweep in range(48):
        for a, b in strata:
            left = local.choice(alib[a])
            right = local.choice(blib[b])
            plans.add(_binomial_canonical(G, left, right))
            if len(plans) >= budget:
                break
        if len(plans) >= budget:
            break

    # Every retained slot is selected by a matrix invariant, not by
    # occupancy cards or automorphism-normalized labels alone.
    for plan in sorted(plans):
        admit(*cells_from_plan(plan))

    selected = sorted(keep.values(), key=lambda item: item[0],
                      reverse=True)[:N_SLOTS]
    if len(selected) < min(5, N_SLOTS):
        raise ValueError("fewer than five fresh short-logical-screened designs")

    bank = []
    for objective, A, B in selected:
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        assert n == 588
        assert n - gf2_rank(hx) - gf2_rank(hz) >= 336
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert all(1 <= len(cell) <= CELL_MAX
                   and len(cell) == len(set(cell))
                   for grid in (A, B) for row in grid for cell in row)
        bank.append((A, B))
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly screened cell without additional draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Canonical triangle-free binomial enumeration plus incidence fingerprints supplies sixty distinct improving C7:C4 designs: the current parent produced thirteen isomorphism classes, voided slots 21-59, and all five measured codes remained [588,345,4].
# REFUTED: Stabilizer-aware random cardinality experiments plus a capped triangle-free control bank break the C7:C4 plateau: the current program still measured five [588,345,4] leaders, scored 9.388, and reproduced five archived codes.
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
