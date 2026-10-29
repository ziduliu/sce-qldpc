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
CONCEPT = ("A4 x C7, order 84, shapes (1,2)x(1,3): independently optimized noncommuting cubic cells with different central-coordinate architectures should supply five fresh measured distance-twelve codes in cell 0.300; at the rank bound these would score 41.1429, above the cell's 37.9538 record.")
GROUP = ("Jump to the direct product (V4 semidirect C3) x C7, genuinely different from the parent's order-28 metacyclic group. No centre-7, 28-class order-84 bed appears in BEDS. The order-84 dicyclic bed reached 38.0 while other order-84 beds reached 27.1-34.5, motivating comparison rather than certifying an advantage. Central C7 voltages supplement noncommuting A4 directions.")
SHAPES = ("Redirect from the repeated distance-four 0.550 plateau to cell 0.300. A=(1,2), B=(1,3) give n=588, kb=168 and guaranteed rate 2/7, with 168/252-variable components and 84 checks each. Parent minima occupied sector-one Acol/Bcol blocks broadly; screen both components and actual full-code logical representatives. Verified ranks retain 0.275<=k/n<0.325.")
FILLING = ("Five C3-quotient architectures crossed with four central-coordinate models define twenty policies. All cells have three distinct terms and HX/HZ weights 9/12. Compile component actions only from verified unit lifts, exclude component kernels through four, and independently deepen retained true kernel witnesses. Assemble rank-legal finalists, test component embeddings modulo exact stabilizer spans, dress logicals and run full-code probes. Unequal uncoloured Tanner signatures separate emitted designs; bounded-search readings are never distance certificates.")
FAMILY_NAME = "r030_A4xC7_12x13_cubic_central_architectures_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # A4 x C7 = (V4 semidirect C3) x C7.
    # V4 uses XOR; the C3 action cycles 1 -> 2 -> 3 -> 1.
    if not valid_q(q):
        raise ValueError("A4 x C7 requires order 84")
    actions = ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2))
    els = [(a, b, c) for c in range(7)
           for b in range(3) for a in range(4)]

    def red(g):
        return (g[0] % 4, g[1] % 3, g[2] % 7)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return (a ^ actions[b][d], (b + e) % 3, (c + f) % 7)

    def inv(g):
        a, b, c = red(g)
        bi = (-b) % 3
        return (actions[bi][a], bi, (-c) % 7)

    def idx(g):
        a, b, c = red(g)
        return 12*c + 4*b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)

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
def _legacy_triangle_NEW_CTX(rng, G, idx):
    """Inactive historical C7:C4 optimizer; superseded below."""
    if idx >= 0:
        bank = _legacy_triangle_NEW_CTX(None, G, -1)
        if idx >= len(bank):
            raise ValueError("no further fresh invariant-distinct design")
        A, B = bank[idx]
        return {"A": A, "B": B}

    from collections import Counter
    assert G.q == 28
    local = random.Random(BASE_SEED * 100003 + 28055042)
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
        if fx < 4:
            return
        fz = logical_floor(hx, hz, n)
        floor = min(fx, fz)
        if floor < 4:
            return

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
        for attempt in range(24):
            model = attempt % 3
            admit(draw_side(aw, model, card + attempt),
                  draw_side(bw, model, 2 * card + attempt + 1))

    # Fresh controls change intrinsic Cayley geometry by using even
    # relatives. Enumerate triangle-free sides rather than perturbing
    # labels of the parent's exhausted all-odd designs.
    representatives = [
        g for g in elements
        if g != G.e and G.mul(g, g) != G.e
        and G.idx(g) < G.idx(G.inv(g))
    ]

    def library(count):
        result = []
        for side in combinations(representatives, count):
            signed = set(side)
            signed.update(G.inv(g) for g in side)
            if any(G.mul(g, h) in signed
                   for g in signed for h in signed):
                continue
            result.append(side)
        return result

    alib, blib = library(4), library(5)
    plans = set()
    for left in alib:
        for right in blib:
            if all(g[1] & 1 for g in left + right):
                continue
            plans.add(_binomial_canonical(G, left, right))
    plans = sorted(plans)
    local.shuffle(plans)

    for plan in plans[:max(640, 12 * N_SLOTS)]:
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


# Historical binomial helpers above are inactive on the new bed.
# All active parity-matrix construction uses verified scaffold lifts.

_A47_PATTERNS = (
    (0, 0, 1), (0, 1, 2), (0, 1, 1),
    (0, 0, 2), (0, 2, 2),
)
_A47_CARDS = (
    (0, 1, 0, 1, 2),
    (1, 1, 1, 1, 1),
    (0, 2, 1, 3, 4),
    (3, 4, 0, 2, 1),
    (2, 1, 4, 1, 3),
)


def _a47_columns(rows, width):
    cols = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    return cols


def _a47_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _a47_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _a47_short4(cols):
    """Exact exclusion through four for odd-weight cubic columns."""
    if 0 in cols or len(set(cols)) != len(cols):
        return False
    assert all(col.bit_count() == 3 for col in cols)
    # Odd column weights exclude every odd-weight kernel.
    pairs = set()
    for i, col in enumerate(cols):
        for other in cols[:i]:
            syndrome = col ^ other
            if syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _a47_six_load(cols, height):
    """Tanner six-cycle count after excluding four-cycles."""
    adjacency = [0] * height
    for col in cols:
        positions = []
        while col:
            bit = col & -col
            positions.append(bit.bit_length() - 1)
            col ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                adjacency[a] |= 1 << b
                adjacency[b] |= 1 << a
    triangles = 0
    for neighbours in adjacency:
        remaining = neighbours
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            triangles += (neighbours & adjacency[j]).bit_count()
            remaining ^= bit
    # Each cubic column creates one check-intersection triangle.
    return triangles // 6 - len(cols)


@lru_cache(maxsize=1)
def _a47_atlas(G):
    """Compile both actions from verified single-term LP matrices."""
    q = G.q
    mask = (1 << q) - 1
    stub_a = [[[G.e], [G.e]]]
    stub_b = [[[G.e], [G.e], [G.e]]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                A, B = [[[g], []]], stub_b
            else:
                A, B = stub_a, [[[g], [], []]]
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            rows = [row & mask for row in source[:q]]
            blocks.append(tuple(_a47_columns(rows, q)))
        atlas[side] = tuple(blocks)
    return atlas


def _a47_compile(G, atlas, side, cells):
    """F2-linear superposition of scaffold-generated cell blocks."""
    q = G.q
    cols = [0] * (len(cells[0]) * q)
    for j, cell in enumerate(cells[0]):
        for g in cell:
            block = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(block):
                cols[j*q + h] ^= syndrome
    return cols


def _a47_draw(local, G, kinds, mode, slot, side):
    """Different quotient/central occupancies, not common orbit images."""
    row, used = [], set()
    for j, kind in enumerate(kinds):
        pattern = _A47_PATTERNS[kind]
        for retry in range(160):
            phase = local.randrange(3)
            direction = local.choice((1, 2))
            bs = [(phase + direction*b) % 3 for b in pattern]
            base, step = local.randrange(7), local.randrange(1, 7)
            if mode == 0:
                cs = local.sample(range(7), 3)
            elif mode == 1:
                cs = [(base + t*step) % 7 for t in range(3)]
            elif mode == 2:
                cs = [base, base, (base + step) % 7]
                local.shuffle(cs)
            else:
                cs = [local.randrange(7) for _ in range(3)]
            aa = [local.randrange(4) for _ in range(3)]
            cell = list(zip(aa, bs, cs))
            if len(set(cell)) != 3:
                continue
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if G.mul(relatives[0], relatives[1]) == G.mul(
                    relatives[1], relatives[0]):
                continue
            differences = {
                G.mul(G.inv(g), h)
                for g in cell for h in cell if g != h
            }
            if len(differences) != 6 or differences & used:
                continue
            row.append(cell)
            used.update(differences)
            break
        else:
            return None
    return [row]


def _a47_probe(cols, local, trials, seeds=(), stabilizers=None):
    """Found kernel/logical witnesses; a high reading proves nothing.

    Information-set elimination carries physical supports. For full
    codes only non-stabilizer words update the reading; short
    stabilizers remain useful descent directions.
    """
    width = len(cols)
    span = None if stabilizers is None else _a47_basis(stabilizers)
    generators = () if stabilizers is None else tuple(stabilizers)
    memory = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:32]
    logical_memory = []
    best = width + 1
    order = list(range(width))

    def consider(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        if span is None or _a47_outside(word, span):
            best = min(best, weight)

    for word in memory:
        consider(word)

    for trial in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                consider(word)

        if span is None:
            starts = dependencies
        else:
            starts = sorted(
                (w for w in dependencies if _a47_outside(w, span)),
                key=int.bit_count)[:12]
        words = sorted(set(starts + memory + logical_memory),
                       key=lambda w: (w.bit_count(), w))[:48]
        donors = sorted(set(dependencies + memory),
                        key=int.bit_count)[:32]

        for sweep in range(2):
            local.shuffle(donors)
            reduced = []
            for word in words:
                weight = word.bit_count()
                for donor in donors:
                    proposal = word ^ donor
                    pw = proposal.bit_count()
                    if 0 < pw < weight:
                        if span is not None and _a47_outside(word, span):
                            if not _a47_outside(proposal, span):
                                continue
                        word, weight = proposal, pw
                if generators and _a47_outside(word, span):
                    for check in generators:
                        proposal = word ^ check
                        pw = proposal.bit_count()
                        if 0 < pw < weight:
                            word, weight = proposal, pw
                consider(word)
                reduced.append(word)
            words = sorted(set(reduced), key=int.bit_count)
            donors = sorted(set(words + donors), key=int.bit_count)[:32]

        discoveries = set(words + memory + logical_memory)
        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                proposal = word ^ other
                if proposal:
                    consider(proposal)
                    discoveries.add(proposal)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        if span is not None:
            logical_memory = sorted(
                (w for w in discoveries if _a47_outside(w, span)),
                key=int.bit_count)[:8]
        if best <= 4:
            break

    return best, tuple(memory + logical_memory)


def _a47_pool(local, G, atlas, side, kinds, mode, slot, batch):
    """Independent constituent optimization with equal finalist budgets."""
    q = G.q
    quota = (48 if side == "A" else 96) + 24*batch
    pool, seen = [], set()

    def merit(item):
        return (item["d"], -item["six"], item["rank"])

    for attempt in range(quota):
        cells = _a47_draw(local, G, kinds, mode, slot, side)
        if cells is None:
            continue
        labels = tuple(tuple(sorted(c)) for c in cells[0])
        if labels in seen:
            continue
        seen.add(labels)
        cols = _a47_compile(G, atlas, side, cells)
        rank = gf2_rank(cols)
        if rank < q - 5 or not _a47_short4(cols):
            continue
        reading, words = _a47_probe(cols, local, 2)
        pool.append(dict(
            cells=cells, cols=cols, rank=rank, d=reading, words=words,
            six=_a47_six_load(cols, q)))
        pool.sort(key=merit, reverse=True)
        del pool[6:]

    for item in pool:
        reading, words = _a47_probe(
            item["cols"], local, 12, item["words"])
        item["d"] = min(item["d"], reading)
        item["words"] = words
    pool.sort(key=merit, reverse=True)
    pool = pool[:3]
    for item in pool:
        reading, words = _a47_probe(
            item["cols"], local, 32, item["words"])
        item["d"] = min(item["d"], reading)
        item["words"] = words
    return sorted(pool, key=merit, reverse=True)


def _a47_full_reading(item, local, q):
    """Verify and dress embeddings, then search both full logical types."""
    hx, hz, n = item["hx"], item["hz"], item["n"]
    mask = (1 << q) - 1
    best = n + 1

    for side, words, checks, stabilizers, fixed_count in (
            ("A", item["aw"], hx, hz, 3),
            ("B", item["bw"], hz, hx, 2)):
        span = _a47_basis(stabilizers)
        seeds = []
        for seed in sorted(set(words), key=int.bit_count)[:8]:
            for fixed in range(fixed_count):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask) << ((3*j+fixed)*q)
                        for j in range(2))
                else:
                    word = seed << (fixed*3*q)
                # No unverified convention is assumed for an embedding.
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _a47_outside(word, span):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        proposal = word ^ row
                        pw = proposal.bit_count()
                        nearby.append((pw, row))
                        if 0 < pw < weight:
                            chosen, weight = proposal, pw
                    nearby.sort(key=lambda pair: pair[0])
                    donors = [row for _, row in nearby[:12]]
                    for i, row in enumerate(donors):
                        for other in donors[:i]:
                            proposal = word ^ row ^ other
                            pw = proposal.bit_count()
                            if 0 < pw < weight:
                                chosen, weight = proposal, pw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
                seeds.append(word)

        cols = _a47_columns(checks, n)
        reading, _ = _a47_probe(
            cols, local, 6, seeds, stabilizers=stabilizers)
        best = min(best, reading)
    return best


def _a47_signature(hx, hz, n):
    """Uncoloured Tanner invariant; unequal signatures exclude isomorphism.

    Equal signatures are conservatively rejected, never asserted to
    prove isomorphism. Squared check-overlap neighbourhoods strengthen
    degree and immediate-overlap profiles.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length() - 1].append(r)
            source ^= bit

    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    degrees = [row.bit_count() for row in rows]
    for r, neighbours in enumerate(adjacency):
        square = Counter()
        for s, multiplicity in neighbours.items():
            for t, other in adjacency[s].items():
                square[t] += multiplicity * other
        profiles.append((
            degrees[r],
            tuple(sorted(Counter(
                (degrees[s], multiplicity)
                for s, multiplicity in neighbours.items()).items())),
            square.get(r, 0),
            sum(neighbours.get(t, 0)*value
                for t, value in square.items()),
            tuple(sorted(Counter(
                (degrees[t], value)
                for t, value in square.items()).items()))))
    return (
        tuple(sorted(Counter(map(len, incidents)).items())),
        tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _a47_bank(G):
    """Deterministic structural-policy bank without orbit fallbacks."""
    assert G.q == 84
    assert A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    q = G.q
    mask = (1 << q) - 1
    atlas = _a47_atlas(G)
    bank, emitted = [], set()

    def merit(item):
        d = min(item["da"], item["db"], item.get("full", item["n"]+1))
        return (item["k"]*d*d/item["n"], d,
                item["da"]+item["db"], -item["six"])

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        lane = slot // 20
        card = _A47_CARDS[architecture]
        ak = card[:2]
        bk = tuple((kind + lane) % len(_A47_PATTERNS)
                   for kind in card[2:])
        chosen = None

        for batch in range(2):
            local = random.Random(
                BASE_SEED*100003 + 84703071
                + 104729*slot + 7919*batch)
            ap = _a47_pool(
                local, G, atlas, "A", ak, mode, slot, batch)
            bp = _a47_pool(
                local, G, atlas, "B", bk, mode, slot, batch)
            assembled = []
            for a in ap:
                for b in bp:
                    A, B = a["cells"], b["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (275*n <= 1000*k < 325*n):
                        continue

                    # Confirm every compiled action against a complete lift.
                    ar = [
                        (row & mask)
                        | (((row >> (3*q)) & mask) << q)
                        for row in hx[:q]]
                    br = [row & ((1 << (3*q))-1) for row in hz[:q]]
                    assert _a47_columns(ar, 2*q) == a["cols"]
                    assert _a47_columns(br, 3*q) == b["cols"]
                    assembled.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        da=a["d"], db=b["d"],
                        aw=a["words"], bw=b["words"],
                        six=a["six"]+b["six"]))

            finalists = sorted(assembled, key=merit, reverse=True)[:3]
            for item in finalists:
                item["full"] = _a47_full_reading(item, local, q)

            for item in sorted(finalists, key=merit, reverse=True):
                signature = _a47_signature(
                    item["hx"], item["hz"], item["n"])
                if signature in emitted:
                    continue
                assert css_ok(item["hx"], item["hz"])
                assert item["n"] == 588 and item["k"] >= 168
                assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
                assert _no_dead_lines([[3, 3]], [[3, 3, 3]])
                assert max(row.bit_count()
                           for row in item["hx"]+item["hz"]) == 12
                assert all(len(cell) == len(set(cell)) == 3
                           for grid in (item["A"], item["B"])
                           for row in grid for cell in row)
                emitted.add(signature)
                chosen = tuple(
                    tuple(tuple(tuple(cell) for cell in row) for row in grid)
                    for grid in (item["A"], item["B"]))
                break
            if chosen is not None:
                break

        # Exhausted policies are not replaced with an emitted relabeling.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this policy's independently screened, rank-legal filling."""
    selected = _a47_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh cubic filling passed this policy")
    A, B = selected
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cubic cell without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Canonical triangle-free binomial enumeration plus incidence fingerprints supplies sixty distinct improving C7:C4 designs: the current parent produced thirteen isomorphism classes, voided slots 21-59, and all five measured codes remained [588,345,4].
# REFUTED: Stabilizer-aware cardinality experiments and fresh triangle-free controls break the C7:C4 plateau: the current program again measured five [588,345,4] leaders, scored 9.388, and reproduced five archived codes.
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
