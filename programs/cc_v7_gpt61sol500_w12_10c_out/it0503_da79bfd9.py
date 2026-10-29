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
CONCEPT = ("Frozen C7:C4, action 6, order 28, A=(1,4), B=(1,5): coordinated minimization of the span of short logical classes may produce five fresh k>=347 codes retaining distance-four readings, or codes excluding logicals through four, and improve cell-0.550 fitness above 9.429. Evaluation can refute this.")
GROUP = ("Frozen byte-for-byte under L1. This bed repeatedly measured [588,345,4], including the current parent's five leaders. C7 singular supports and the central involution permit coordinated filling experiments, but neither greater rank surplus nor improved distance is assumed.")
SHAPES = ("Frozen for open-ended cell 0.550: n=588, kb=336, guaranteed rate 4/7, and maximum admissible lift order 28. Unequal components have 112 and 140 variables with 28 checks each. Parent minima occupied sector-one Acol/Bcol blocks broadly, often emphasizing Acol2 or Bcol4; no listed minimum used Arow/Brow.")
FILLING = ("Cross positive legal count profiles with unrestricted, C7-singular, central-paired and quotient-occupancy supports. Beam search uses simultaneous cell replacements and legal term transfers, minimizing exact short-logical quotient-span rank and collision load rather than repairing only one witness. Screen scaffold lifts modulo actual stabilizers through four, exhaust additional binomial controls, blacklist displayed archived geometries, and separate outputs with uncoloured three-check-intersection fingerprints. Five is an exclusion floor, not a sampled distance. No archived or distance-three padding is emitted.")
FAMILY_NAME = "r055_m7r4s6_14x15_jointlogicalspan_tripleincidence_v5"

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
    """Pure shared optimization; cache even an exhausted immutable bank."""
    if idx >= 0:
        bank = NEW_CTX(None, G, -1)
        if idx >= len(bank):
            raise ValueError("no further fresh jointly screened design")
        A, B = bank[idx]
        return {"A": A, "B": B}

    from collections import Counter
    assert G.q == 28
    q = G.q
    elements = tuple(G.elements)
    local = random.Random(BASE_SEED * 100003 + 28055519)

    def columns(rows, n):
        out = [0] * n
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                out[bit.bit_length()-1] |= 1 << r
                source ^= bit
        return out

    def insert(word, basis):
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                return
            word ^= old

    def short_statistics(checks, stabilizers, n):
        """Exact logical minimum through four and quotient-span statistics.

        Supports with equal syndrome differ by a commuting word.
        Their linear normal forms determine whether it is a stabilizer.
        All singleton/pair comparisons precede pair/pair comparisons.
        """
        basis = {}
        for word in stabilizers:
            insert(word, basis)
        residues = []
        for c in range(n):
            row = basis.get(c)
            if row is None:
                residues.append(1 << c)
            else:
                tail, residue = row ^ (1 << c), 0
                while tail:
                    bit = tail & -tail
                    residue ^= residues[bit.bit_length()-1]
                    tail ^= bit
                residues.append(residue)

        cols = columns(checks, n)
        bits = [1 << c for c in range(n)]
        singles = {0: (0, 0)}
        span, load, pressure = {}, 0, 0
        floor = 5

        def observe(residue, word):
            nonlocal load, pressure, floor
            if residue:
                weight = word.bit_count()
                assert 1 <= weight <= 4
                floor = min(floor, weight)
                insert(residue, span)
                load += 1
                pressure |= word

        for c, syndrome in enumerate(cols):
            entry = (residues[c], bits[c])
            previous = singles.get(syndrome)
            if previous is None:
                singles[syndrome] = entry
            else:
                observe(entry[0] ^ previous[0],
                        entry[1] ^ previous[1])

        for i, syndrome in enumerate(cols):
            ri, bi = residues[i], bits[i]
            for j in range(i):
                previous = singles.get(syndrome ^ cols[j])
                if previous is not None:
                    observe(ri ^ residues[j] ^ previous[0],
                            bi ^ bits[j] ^ previous[1])
        if floor < 4:
            # The span/load here cover all compared logicals through three.
            return floor, len(span), load, pressure

        seen = dict(singles)
        for i, syndrome in enumerate(cols):
            ri, bi = residues[i], bits[i]
            for j in range(i):
                key = syndrome ^ cols[j]
                entry = (ri ^ residues[j], bi ^ bits[j])
                previous = seen.get(key)
                if previous is None:
                    seen[key] = entry
                else:
                    observe(entry[0] ^ previous[0],
                            entry[1] ^ previous[1])
        return floor, len(span), load, pressure

    def fingerprint(hx, hz, n):
        """Uncoloured incidence invariant, including triple intersections.

        Inequality certifies nonisomorphism. Equality is discarded
        conservatively and is not asserted to prove isomorphism.
        """
        rows = list(hx) + list(hz)
        weights = [row.bit_count() for row in rows]
        incidence = [[] for _ in range(n)]
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                incidence[bit.bit_length()-1].append(r)
                source ^= bit

        pair = Counter()
        triple = Counter()
        for ids in incidence:
            pair.update(combinations(ids, 2))
            triple.update(combinations(ids, 3))

        roots = [Counter() for _ in rows]
        for (a, b, c), count in triple.items():
            for i, j, k in ((a, b, c), (b, a, c), (c, a, b)):
                left = (weights[j], pair[tuple(sorted((i, j)))])
                right = (weights[k], pair[tuple(sorted((i, k)))])
                roots[i][(
                    tuple(sorted((left, right))),
                    pair[tuple(sorted((j, k)))], count)] += 1

        keys = [
            (weights[i], tuple(sorted(roots[i].items())))
            for i in range(len(rows))]
        links = [[] for _ in rows]
        for (i, j), count in pair.items():
            links[i].append((j, count))
            links[j].append((i, count))

        history = []
        colours = []
        for _ in range(3):
            histogram = Counter(keys)
            history.append(tuple(sorted(histogram.items())))
            palette = {key: c for c, key in enumerate(sorted(histogram))}
            colours = [palette[key] for key in keys]
            keys = [
                (colours[i], tuple(sorted(Counter(
                    (colours[j], count)
                    for j, count in links[i]).items())))
                for i in range(len(rows))]
        variables = Counter(
            tuple(sorted(colours[r] for r in ids))
            for ids in incidence)
        return n, tuple(history), tuple(sorted(variables.items()))

    # Displayed parent and earlier exact-bed codes are blacklist entries,
    # never candidate seeds or an emitted fallback.
    archived = set()
    cards = (
        "10 01 31 22|10 01 21 41 12",
        "10 01 31 22|30 01 11 21 32",
        "01 12 22 32|10 01 21 41 12",
        "10 01 21 41|10 01 21 41 12",
        "10 01 21 41|20 01 11 41 22",
        "10 01 31 12|10 01 21 41 12",
        "10 01 31 12|20 01 11 41 22",
        "01 11 21 22|01 11 21 31 41",
        "01 11 22 32|01 11 21 31 41",
        "01 11 21 41|10 01 21 41 12",
        "10 01 21 41|01 11 21 31 41",
        "01 11 21 31|10 01 21 41 12",
        "10 01 21 12|10 01 21 41 12",
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
        "10 01 31 12|20 11 21 51 22",
        "01 11 21 32|01 11 31 51 61",
        "10 01 21 12|20 01 31 61 22",
        "10 01 21 12|01 11 21 31 51",
        "10 20 30 12|10 20 30 12 22",
        "10 20 30 12|10 20 30 22 32",
        "10 20 30 12|10 01 21 41 12",
        "10 20 30 12|30 01 11 21 32",
        "10 20 30 12|01 11 21 31 41",
        "10 20 30 12|10 20 30 01 11",
        "10 01 21 12|01 11 21 41 51",
        "01 11 21 31|20 01 11 41 22",
        "01 11 21 31|01 11 21 31 51",
        "01 11 21 31|01 11 21 41 51",
        "01 11 21 32|20 01 11 41 22",
        "10 01 21 41|01 11 21 41 51",
        "10 01 31 12|01 11 21 31 51",
        "10 01 31 22|01 11 21 41 51",
        "01 11 21 32|10 01 21 41 12",
        "01 11 21 32|30 01 11 21 32",
        "01 11 21 32|01 11 21 31 41",
        "01 12 22 32|30 01 11 21 32",
    )
    for card in cards:
        grids = []
        for text in card.split("|"):
            grids.append((tuple(
                (G.e, (int(t[0]), int(t[1])))
                for t in text.split()),))
        hx, hz, n = lifted_matrices(G, *grids)
        archived.add(fingerprint(hx, hz, n))

    keep, seen_labels = {}, set()

    def freeze(A, B):
        return tuple(
            tuple(tuple(tuple(cell) for cell in row) for row in grid)
            for grid in (A, B))

    def read(A, B):
        sa = [[len(c) for c in A[0]]]
        sb = [[len(c) for c in B[0]]]
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            return None
        frozen = freeze(A, B)
        labels = tuple(tuple(sorted(c))
                       for grid in frozen for row in grid for c in row)
        if labels in seen_labels:
            return None
        seen_labels.add(labels)
        hx, hz, n = lifted_matrices(G, *frozen)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            return None
        z = short_statistics(hx, hz, n)
        x = short_statistics(hz, hx, n)
        floor = min(x[0], z[0])
        item = dict(A=frozen[0], B=frozen[1], hx=hx, hz=hz,
                    n=n, k=k, x=x, z=z, floor=floor)
        item["beam"] = (
            floor, -x[1]-z[1], -x[2]-z[2], k)
        item["selection"] = (
            k*floor*floor, floor, k, -x[1]-z[1], -x[2]-z[2])
        if floor >= 4:
            fp = fingerprint(hx, hz, n)
            if fp not in archived:
                previous = keep.get(fp)
                if previous is None or item["selection"] > previous["selection"]:
                    keep[fp] = item
        return item

    profiles = [
        (a, b)
        for a in combinations_with_replacement(range(1, 6), 4)
        for b in combinations_with_replacement(range(1, 6), 5)
        if sum(a)+max(b) <= W_MAX and sum(b)+max(a) <= W_MAX
        and (a, b) != ((2, 2, 2, 2), (2, 2, 2, 2, 2))]
    profiles.sort(key=lambda p: (
        p[0].count(1)+p[1].count(1), -sum(p[0])-sum(p[1]), p))

    def draw_cell(count, policy, phase):
        if policy == 1 and count == 3:
            # C7 minimal-polynomial supports, coordinated across cells.
            d = 1 + phase % 6
            support = (G.e, (d, 0), (3*d % 7, 0))
        elif policy == 2 and count >= 2:
            support = []
            while len(support)+2 <= count:
                available = [
                    g for g in elements
                    if g not in support
                    and G.mul(g, (0, 2)) not in support]
                g = local.choice(available)
                support.extend((g, G.mul(g, (0, 2))))
            if len(support) < count:
                support.append(local.choice(
                    [g for g in elements if g not in support]))
        elif policy == 3:
            occupancy = (phase+j for j in range(count))
            support = []
            for b in occupancy:
                support.append(local.choice(
                    [g for g in elements
                     if g[1] == b % 4 and g not in support]))
        else:
            support = local.sample(elements, count)
        anchor = local.choice(elements)
        result = tuple(G.mul(g, anchor) for g in support)
        assert len(result) == len(set(result)) == count
        return result

    # Twenty construction policies: five count architectures crossed
    # with four element models. Extra slots use additional survivors,
    # never transported copies of these policies.
    for card in range(20):
        architecture, policy = divmod(card, 4)
        position = architecture * max(1, len(profiles)//5)
        aw, bw = profiles[min(position, len(profiles)-1)]
        elites = []

        def retain(item):
            if item is not None:
                elites.append(item)
                elites.sort(key=lambda x: x["beam"], reverse=True)
                del elites[6:]

        for attempt in range(8):
            A = [[draw_cell(c, policy, card+j+attempt)
                  for j, c in enumerate(aw)]]
            B = [[draw_cell(c, policy, 2*card+j+attempt)
                  for j, c in enumerate(bw)]]
            retain(read(A, B))

        for step in range(24):
            if not elites:
                break
            parent = local.choice(elites[:min(4, len(elites))])
            for branch in range(4):
                A = [[list(c) for c in parent["A"][0]]]
                B = [[list(c) for c in parent["B"][0]]]
                grids = {"A": A, "B": B}
                side = "A" if parent["z"][0] <= parent["x"][0] else "B"
                if branch == 3:
                    side = local.choice(("A", "B"))
                witness = parent["z"][3] if side == "A" else parent["x"][3]
                pressure = [1] * len(grids[side][0])
                while witness:
                    bit = witness & -witness
                    c = bit.bit_length()-1
                    if c < 20*q:
                        j = c//(5*q) if side == "A" else (c//q) % 5
                        pressure[j] += 1
                    witness ^= bit
                row = grids[side][0]
                j = local.choices(range(len(row)), weights=pressure)[0]
                row[j] = list(draw_cell(
                    len(row[j]), (policy+branch) % 4, card+step+j))
                if branch in (1, 2):
                    # Coordinated replacement, not a single-witness repair.
                    other = local.choice([t for t in range(len(row)) if t != j])
                    row[other] = list(draw_cell(
                        len(row[other]), policy, 2*card+step+other))
                if branch == 3:
                    positions = [(s, t) for s in ("A", "B")
                                 for t in range(len(grids[s][0]))]
                    sources = [(s, t) for s, t in positions
                               if len(grids[s][0][t]) > 1]
                    destinations = [(s, t) for s, t in positions
                                    if len(grids[s][0][t]) < CELL_MAX]
                    if sources and destinations:
                        source = local.choice(sources)
                        destination = local.choice(destinations)
                        if source != destination:
                            src = grids[source[0]][0][source[1]]
                            dst = grids[destination[0]][0][destination[1]]
                            g = local.choice(src)
                            if g not in dst:
                                src.remove(g)
                                dst.append(g)
                retain(read(A, B))

    # Exhaust the small inverse-pair library, including the involution.
    # Acceptance is stabilizer-aware, not a no-small-kernel assumption.
    representatives = [
        g for g in elements if g != G.e
        and G.idx(g) <= G.idx(G.inv(g))]

    def library(count):
        result = []
        for side in combinations(representatives, count):
            signed = set(side) | {G.inv(g) for g in side}
            if not any(G.mul(g, h) in signed
                       for g in signed for h in signed):
                result.append(side)
        return result

    plans = {
        _binomial_canonical(G, left, right)
        for left in library(4) for right in library(5)}
    for plan in sorted(plans):
        A = (tuple((G.e, elements[t]) for t in plan[0]),)
        B = (tuple((G.e, elements[t]) for t in plan[1]),)
        read(A, B)

    selected = sorted(
        keep.values(), key=lambda x: x["selection"],
        reverse=True)[:N_SLOTS]
    bank = []
    for item in selected:
        A, B = item["A"], item["B"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 588 and item["k"] >= 336
        assert css_ok(hx, hz)
        assert max(r.bit_count() for r in hx+hz) <= W_MAX
        assert all(1 <= len(c) <= CELL_MAX and len(c) == len(set(c))
                   for grid in (A, B) for row in grid for c in row)
        bank.append((A, B))
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly selected cell without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Canonical triangle-free binomial enumeration plus pair-incidence fingerprints supplies sixty distinct improving C7:C4 designs: the measured parent produced thirteen isomorphism classes, voided slots 21-59, and all five measured leaders remained [588,345,4].
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
