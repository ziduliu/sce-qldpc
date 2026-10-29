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
CONCEPT = ("Metacyclic C11:C5, action 3, order 55, A=B=(1,3): packed noncommuting cubic supports, exact constituent exclusion through eight and independent stabilizer-dressed logical rescreens should supply five target-rate codes with measured readings at least ten; rank surplus or larger readings would improve cell-0.400 fitness above forty.")
GROUP = ("Preserve the parent's cyclic-kernel/cyclic-complement semidirect construction rule, resizing its parameters to m=11,r=5,s=3. Here 3 has order five modulo eleven and n=550 fits the new factor ten. The recorded order-55, centre-one, seven-class bed reached fitness 25.6; stronger constituent screening is a testable response, not a presumed group advantage. The order-56 affine bed's forty is a comparison, not a construction copied across families.")
SHAPES = ("Target cell 0.400 with equal (1,3) sides: n=550, kb=220 and guaranteed rate 0.400. Each constituent has 165 variables and 55 checks; the shape permits q<=59 and the selected faithful metacyclic order is 55. Cubic cells give HX=HZ=12. Parent minima were entirely sector-one Acol/Bcol witnesses, broadly distributed and often emphasizing Acol2 or Bcol4. Gate every complete verified rank inside 0.375<=k/n<0.425 and probe both LP sectors.")
FILLING = ("Twenty policies cross five relative quotient architectures with four coordinate models. Compile cubic constituent columns solely from scaffold unit lifts, pack noncommuting directed differences, exclude constituent kernels through six and test through eight without a search cutoff, then repair cells carrying actual witnesses. Retain independently optimized six-weight fallbacks if stronger constituent exclusions fail. Complete lifts require exact kernel exclusion through four and target-band ranks. Compare real non-stabilizer witnesses after equal independent rescreens and single/pair stabilizer dressing. Unequal uncoloured Tanner walk/refinement signatures certify emitted diversity; equal signatures are conservatively rejected. No sampled miss is called a distance certificate.")
FAMILY_NAME = "r040_metacyclic_m11r5s3_13x13_cubic_exact8_dressed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    # Same metacyclic semidirect rule, resized for the new shapes.
    # C11:C5: y*x*y^-1=x^3; 3 has order five modulo eleven.
    if not valid_q(q):
        raise ValueError("C11:C5 requires order 55")
    m, r, s = 11, 5, 3
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
def _legacy_product_ctx(rng, G, idx):
    """Historical (1,4)x(1,5) policy; never called by the new bank."""
    if idx >= 0:
        bank = _legacy_product_ctx(None, G, -1)
        if idx >= len(bank):
            raise ValueError("no further fresh logical-four-screened design")
        A, B = bank[idx]
        return {"A": A, "B": B}

    from collections import Counter
    assert G.q == 28
    q = G.q
    mask = (1 << q) - 1
    elements = tuple(G.elements)
    local = random.Random(BASE_SEED * 100003 + 28055183)

    def columns(rows, width):
        result = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                result[bit.bit_length() - 1] |= 1 << r
                source ^= bit
        return tuple(result)

    def logical_floor(checks, stabilizers, n):
        """Exact logical minimum through four, including both LP sectors.

        Five is an exclusion sentinel. Any returned witness is outside
        the exact stabilizer span; small stabilizers are permitted.
        """
        basis = {}
        for word in stabilizers:
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old

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
        bits = [1 << c for c in range(n)]
        for c, syndrome in enumerate(cols):
            if syndrome == 0 and residues[c]:
                return 1, bits[c]

        singles = {0: (0, 0)}
        for c, syndrome in enumerate(cols):
            old = singles.get(syndrome)
            if old is None:
                singles[syndrome] = (residues[c], bits[c])
            elif old[0] != residues[c]:
                return 2, old[1] ^ bits[c]

        # Complete pair/single comparison precedes the weight-four test.
        for i, syndrome in enumerate(cols):
            ri = residues[i]
            for j in range(i):
                old = singles.get(syndrome ^ cols[j])
                if old is not None and old[0] != (ri ^ residues[j]):
                    word = old[1] ^ bits[i] ^ bits[j]
                    assert word.bit_count() == 3
                    return 3, word

        seen = dict(singles)
        for i, syndrome in enumerate(cols):
            ri = residues[i]
            for j in range(i):
                key = syndrome ^ cols[j]
                residue = ri ^ residues[j]
                old = seen.get(key)
                if old is None:
                    seen[key] = (residue, bits[i] ^ bits[j])
                elif old[0] != residue:
                    word = old[1] ^ bits[i] ^ bits[j]
                    assert word.bit_count() == 4
                    return 4, word
        return 5, 0

    def signature(hx, hz, n):
        """Uncoloured Tanner invariant, independent of X/Z labels.

        Unequal signatures certify nonisomorphism. Equal signatures
        are conservatively discarded, not called an isomorphism proof.
        """
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        links = [dict() for _ in rows]
        adjacency = [0] * len(rows)
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    links[i][j] = links[j][i] = overlap
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        keys = []
        for i, neighbors in enumerate(adjacency):
            triangles, remaining = 0, neighbors
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() - 1
                triangles += (neighbors & adjacency[j]).bit_count()
                remaining ^= bit
            profile = Counter(
                (weights[j], overlap,
                 (adjacency[i] & adjacency[j]).bit_count())
                for j, overlap in links[i].items())
            keys.append((weights[i], triangles // 2,
                         tuple(sorted(profile.items()))))

        history = []
        colours = []
        for _ in range(4):
            histogram = Counter(keys)
            history.append(tuple(sorted(histogram.items())))
            palette = {key: c for c, key in enumerate(sorted(histogram))}
            colours = [palette[key] for key in keys]
            keys = [
                (colours[i], tuple(sorted(Counter(
                    (colours[j], overlap)
                    for j, overlap in links[i].items()).items())))
                for i in range(len(rows))
            ]

        qubits = [[] for _ in range(n)]
        for colour, source in zip(colours, rows):
            while source:
                bit = source & -source
                qubits[bit.bit_length() - 1].append(colour)
                source ^= bit
        coarse = (n, tuple(history), tuple(sorted(Counter(
            tuple(sorted(v)) for v in qubits).items())))

        # Local refinement can merge genuinely different regular lifts.
        # Use a GLOBAL, exact finite-field invariant as additional data.
        # For combined incidence H, Gram=H H^T uses integer overlaps.
        # Tanner relabelings conjugate Gram and its resolvent by the
        # same check permutation, preserving every profile below.
        height = len(rows)
        gram = [[0] * height for _ in rows]
        for i in range(height):
            gram[i][i] = weights[i]
            for j, overlap in links[i].items():
                gram[i][j] = overlap

        try:
            import numpy as np
        except ImportError:
            np = None

        prime = 65521
        points = (37, 41, 43, 47, 53, 59, 61, 67)

        def inverse_at(point):
            """Exact modular Gauss-Jordan, with a portable fallback."""
            if np is not None:
                eye = np.eye(height, dtype=np.int64)
                left = (point * eye
                        - np.asarray(gram, dtype=np.int64)) % prime
                aug = np.concatenate((left, eye), axis=1)
            else:
                aug = [
                    [((point if i == j else 0) - gram[i][j]) % prime
                     for j in range(height)]
                    + [int(i == j) for j in range(height)]
                    for i in range(height)
                ]
            determinant = 1
            for k in range(height):
                pivot_row = next(
                    (r for r in range(k, height) if int(aug[r][k])),
                    None)
                if pivot_row is None:
                    return None
                if pivot_row != k:
                    if np is not None:
                        aug[[k, pivot_row]] = aug[[pivot_row, k]]
                    else:
                        aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
                    determinant = (-determinant) % prime

                pivot = int(aug[k][k])
                determinant = determinant * pivot % prime
                scale = pow(pivot, prime - 2, prime)
                if np is not None:
                    aug[k] = aug[k] * scale % prime
                    factors = aug[:, k].copy()
                    factors[k] = 0
                    # Products are below prime^2, safely inside int64.
                    aug[:] = (aug - np.outer(factors, aug[k])) % prime
                else:
                    aug[k] = [value * scale % prime for value in aug[k]]
                    pivot_values = aug[k]
                    for r in range(height):
                        if r == k:
                            continue
                        factor = aug[r][k]
                        if factor:
                            aug[r] = [
                                (x - factor * y) % prime
                                for x, y in zip(aug[r], pivot_values)
                            ]
            inverse = (aug[:, height:] if np is not None
                       else [row[height:] for row in aug])
            return determinant, inverse

        resolved = None
        for point in points:
            resolved = inverse_at(point)
            if resolved is not None:
                break
        if resolved is None:
            # A conservative invariant fallback, never a validity failure.
            return coarse + ((prime, points, "singular"),)

        determinant, inverse = resolved
        diagonal = [int(inverse[i][i]) for i in range(height)]
        profiles = []
        for i in range(height):
            environment = Counter(
                (weights[j], diagonal[j], gram[i][j],
                 int(inverse[i][j]))
                for j in range(height) if j != i)
            profiles.append(
                (weights[i], diagonal[i],
                 tuple(sorted(environment.items()))))
        global_profile = (
            prime, point, determinant,
            tuple(sorted(Counter(profiles).items())))
        return coarse + (global_profile,)

    archived = set()

    def archive(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        archived.add(signature(hx, hz, n))

    # Compile the unchanged parent's actual bank only for blacklisting.
    # It is never used as a candidate bank or a fallback.
    for A, B in _design_bank(G):
        archive(A, B)

    # Additional displayed codes from earlier programs on this exact bed.
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
        "10 01 21 12|20 01 31 61 22",
        "10 01 21 12|01 11 21 31 51",
        "10 20 30 12|10 20 30 12 22",
        "10 20 30 12|10 20 30 22 32",
        "10 20 30 12|10 01 21 41 12",
        "10 20 30 12|30 01 11 21 32",
        "10 20 30 12|01 11 21 31 41",
        "10 20 30 12|10 20 30 01 11",
        # All six emitted current-parent slots, including the iso copy
        # represented by slot zero. These are blacklist entries only.
        "10 01 21 12|01 11 21 41 51",
        "01 11 21 31|20 01 11 41 22",
        "01 11 21 31|01 11 21 31 51",
        "01 11 21 31|01 11 21 41 51",
        "01 11 21 32|20 01 11 41 22",
    )
    for card in known_cards:
        grids = []
        for text in card.split("|"):
            grids.append((tuple(
                (G.e, (int(token[0]), int(token[1])))
                for token in text.split()),))
        archive(*grids)

    # Single-term actions are extracted from VERIFIED unit lifts.
    # Their F2 superpositions compile cells, not a hand-written lift.
    atlas = {"A": [], "B": []}
    for g in elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        atlas["A"].append(columns([r & mask for r in hx], q))
        atlas["B"].append(columns([r & mask for r in hz], q))

    def block_columns(cell, side):
        out = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h, value in enumerate(contribution):
                out[h] ^= value
        return tuple(out)

    def clean3(cols):
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return False
        return not any(
            (value ^ cols[j]) in singles
            for i, value in enumerate(cols) for j in range(i))

    # Exhaust normalized doubletons/trinomials. Larger menus cross
    # actual singularity strata rather than drawing only generic cells.
    menus = {}
    for count in range(2, 6):
        if count <= 3:
            supports = [
                (G.e,) + tail
                for tail in combinations(elements[1:], count - 1)]
        else:
            supports = sorted({
                tuple(sorted((G.e,) + tuple(
                    local.sample(elements[1:], count - 1))))
                for _ in range(384)
            })
        for side in ("A", "B"):
            records = []
            for cell in supports:
                cols = block_columns(cell, side)
                records.append((int(clean3(cols)), gf2_rank(cols), cell))
            menus[side, count] = tuple(sorted(records))

    profiles = [
        (aw, bw)
        for aw in combinations_with_replacement(range(1, 6), 4)
        for bw in combinations_with_replacement(range(1, 6), 5)
        if sum(aw) + max(bw) <= W_MAX
        and sum(bw) + max(aw) <= W_MAX
        and (aw, bw) != ((2, 2, 2, 2), (2, 2, 2, 2, 2))
    ]
    profiles.sort(key=lambda p: (
        p[0].count(1) + p[1].count(1),
        -sum(p[0]) - sum(p[1]), p))

    def draw_cell(count, side, policy, phase):
        if count == 1:
            return (local.choice(elements),)
        if policy == 0:
            return tuple(local.sample(elements, count))
        if policy in (1, 2):
            records = menus[side, count]
            if policy == 1:
                clean = [r for r in records if r[0]]
                pool = clean if clean else records
            else:
                minimum = min(r[1] for r in records)
                pool = [r for r in records if r[1] <= minimum + 2]
            cell = local.choice(pool)[2]
            anchor = local.choice(elements)
            return tuple(G.mul(anchor, g) for g in cell)
        if policy == 4:
            # Deliberate group-algebra product supports, not generic
            # samples selected merely for singular block rank.
            # Four terms are the expansion of (e+a)(e+b).
            for _ in range(96):
                a, b = local.sample(elements[1:], 2)
                if G.mul(a, b) == G.mul(b, a):
                    continue
                support = [G.e, a]
                if count >= 3:
                    support.append(b)
                if count >= 4:
                    support.append(G.mul(a, b))
                if count == 5:
                    # Ordered length-three extension changes intrinsic
                    # support geometry rather than transporting a code.
                    support.append(G.mul(G.mul(a, b), a))
                if len(support) != count or len(set(support)) != count:
                    continue
                anchor = local.choice(elements)
                cell = tuple(G.mul(g, anchor) for g in support)
                assert len(cell) == len(set(cell)) == count
                return cell
            # A fresh legal draw, not an archived or emitted fallback.
            return tuple(local.sample(elements, count))
        even = [g for g in elements if not (g[1] & 1)]
        odd = [g for g in elements if g[1] & 1]
        amount = phase % (count + 1)
        return tuple(local.sample(even, amount)
                     + local.sample(odd, count - amount))

    keep, labels_seen = {}, set()

    def read(A, B):
        key = tuple(tuple(sorted(cell))
                    for grid in (A, B) for row in grid for cell in row)
        if key in labels_seen:
            return None
        labels_seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        fx, wx = logical_floor(hz, hx, n)
        fz, wz = logical_floor(hx, hz, n)
        floor = min(fx, fz)
        return dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    fx=fx, fz=fz, floor=floor, wx=wx, wz=wz,
                    score=(k * floor * floor, floor, k, fx + fz))

    def admit(item):
        if item is None or item["floor"] < 4 or item["k"] <= 0:
            return
        fp = signature(item["hx"], item["hz"], item["n"])
        if fp in archived:
            return
        previous = keep.get(fp)
        if previous is None or item["score"] > previous["score"]:
            keep[fp] = item

    for card, (aw, bw) in enumerate(profiles):
        elites = []
        for attempt in range(32):
            policy = attempt % 5
            A = (tuple(draw_cell(c, "A", policy, card + j + attempt)
                       for j, c in enumerate(aw)),)
            B = (tuple(draw_cell(c, "B", policy, 2*card + j + attempt)
                       for j, c in enumerate(bw)),)
            item = read(A, B)
            admit(item)
            if item is not None:
                elites.append(item)
                elites.sort(key=lambda x: x["score"], reverse=True)
                del elites[6:]

        # Repair the side that supplied the actual lighter witness.
        # A changed matrix is always rescreened from scratch.
        for step in range(32):
            if not elites:
                break
            parent = local.choice(elites[:min(4, len(elites))])
            side = "A" if parent["fz"] <= parent["fx"] else "B"
            witness = parent["wz"] if side == "A" else parent["wx"]
            count_grid = aw if side == "A" else bw
            pressure = [1] * len(count_grid)
            remaining = witness
            while remaining:
                bit = remaining & -remaining
                position = bit.bit_length() - 1
                if position < 20*q:
                    j = (position // (5*q) if side == "A"
                         else (position // q) % 5)
                    pressure[j] += 1
                remaining ^= bit
            j = local.choices(range(len(count_grid)), weights=pressure)[0]
            A = [[list(cell) for cell in parent["A"][0]]]
            B = [[list(cell) for cell in parent["B"][0]]]
            target = A if side == "A" else B
            target[0][j] = list(draw_cell(
                count_grid[j], side, (card + step) % 5, card + step))
            if step % 6 == 5:
                other = (j + 1 + local.randrange(len(count_grid)-1)) \
                        % len(count_grid)
                target[0][other] = list(draw_cell(
                    count_grid[other], side, (step + 1) % 5, step))
            item = read(A, B)
            admit(item)
            if item is not None:
                elites.append(item)
                elites.sort(key=lambda x: x["score"], reverse=True)
                del elites[6:]

    # Exhaust this control library rather than sample another capped
    # collection of occupancy strata. Include the central involution;
    # only the complete stabilizer-aware test decides its acceptance.
    representatives = [
        g for g in elements if g != G.e
        and G.idx(g) <= G.idx(G.inv(g))
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

    plans = {
        _binomial_canonical(G, left, right)
        for left in library(4) for right in library(5)
    }
    for plan in sorted(plans):
        A = (tuple((G.e, elements[t]) for t in plan[0]),)
        B = (tuple((G.e, elements[t]) for t in plan[1]),)
        admit(read(A, B))

    selected = sorted(keep.values(), key=lambda x: x["score"],
                      reverse=True)[:N_SLOTS]
    bank = []
    for item in selected:
        A, B = item["A"], item["B"]
        hx, hz, n = lifted_matrices(G, A, B)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert n == 588 and item["k"] >= 336
        assert css_ok(hx, hz)
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert all(1 <= len(cell) <= CELL_MAX
                   and len(cell) == len(set(cell))
                   for grid in (A, B) for row in grid for cell in row)
        bank.append(tuple(
            tuple(tuple(tuple(cell) for cell in row) for row in grid)
            for grid in (A, B)))
    # Cache even an exhausted bank. Do not rerun a failed shared search
    # once per harness slot, and never pad it with archived orbit copies.
    return tuple(bank)


# Five quotient architectures crossed with four coordinate models.
# These specify intrinsic supports, not automorphism images of saved codes.
_C55_CARDS = (
    ((0, 1), (1, 2), (2, 4)),
    ((0, 2), (1, 3), (1, 4)),
    ((1, 2), (1, 3), (2, 4)),
    ((1, 2), (1, 4), (2, 3)),
    ((1, 3), (2, 3), (3, 4)),
)


def _c55_columns(rows, width):
    out = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            out[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    return tuple(out)


def _c55_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous
    return basis


def _c55_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _c55_syndrome(cols, word):
    value = 0
    while word:
        bit = word & -word
        value ^= cols[bit.bit_length() - 1]
        word ^= bit
    return value


def _c55_clean4(cols):
    """Exact conservative kernel exclusion through four.

    All columns used here have odd weight. Thus weight-three kernel
    words are impossible; singleton/pair collisions cover the rest.
    Small stabilizers are deliberately rejected as well.
    """
    assert all(c.bit_count() & 1 for c in cols)
    seen = {0}
    for value in cols:
        if value in seen:
            return False
        seen.add(value)
    for i, value in enumerate(cols):
        for j in range(i):
            pair = value ^ cols[j]
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _c55_exact_word(cols, q, through):
    """Exhaust constituent dependencies through six or eight after clean4.

    Cubic columns impose even kernel parity. Singleton/pair tails are
    unique after clean4. At an unsatisfied check, partition completions
    by their first selected available variable, banning earlier options.
    There is no node or time cutoff.

    Each constituent regular action is transitive within each q-column
    block, so the three anchors exhaust constituent support orbits.
    This reduction is NOT used for complete quantum matrices.
    """
    bits = [1 << j for j in range(len(cols))]
    incidence = [0] * q
    tails = {0: 0}
    for j, source in enumerate(cols):
        assert source.bit_count() == 3
        tails[source] = bits[j]
        while source:
            bit = source & -source
            incidence[bit.bit_length() - 1] |= bits[j]
            source ^= bit
    for j, value in enumerate(cols):
        for h in range(j):
            key = value ^ cols[h]
            assert key not in tails
            tails[key] = bits[j] | bits[h]

    def visit(value, remaining, banned, word):
        if not value:
            return word
        if (value.bit_count() ^ remaining) & 1:
            remaining -= 1
        if remaining <= 0 or value.bit_count() > 3 * remaining:
            return 0
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not (tail & banned)):
                return word | tail
            return 0

        pending, choices, least = value, 0, len(cols) + 1
        while pending:
            bit = pending & -pending
            options = incidence[bit.bit_length() - 1] & ~banned
            count = options.bit_count()
            if not count:
                return 0
            if count < least:
                choices, least = options, count
                if count == 1:
                    break
            pending ^= bit

        alternatives = []
        while choices:
            bit = choices & -choices
            choices ^= bit
            alternatives.append(bit.bit_length() - 1)
        alternatives.sort(
            key=lambda j: (-(cols[j] & value).bit_count(), j))
        excluded = banned
        for j in alternatives:
            excluded |= bits[j]
            answer = visit(value ^ cols[j], remaining - 1,
                           excluded, word | bits[j])
            if answer:
                return answer
        return 0

    for anchor in range(0, len(cols), q):
        word = visit(cols[anchor], through - 1,
                     bits[anchor], bits[anchor])
        if word:
            assert word.bit_count() <= through
            assert _c55_syndrome(cols, word) == 0
            return word
    return 0


def _c55_probe(cols, local, trials, seeds=(), stabilizers=()):
    """Find real kernel/logical witnesses; readings are upper bounds.

    Retain logical witnesses separately from short stabilizers. Every
    recombination has zero syndrome; full-code observations additionally
    test the exact stabilizer span.
    """
    width = len(cols)
    span = _c55_basis(stabilizers) if stabilizers else None
    memory = sorted(
        {w for w in seeds if w and _c55_syndrome(cols, w) == 0},
        key=lambda w: (w.bit_count(), w))[:24]
    found = set()
    best = width + 1

    def observe(word):
        nonlocal best
        if word and (span is None or _c55_outside(word, span)):
            found.add(word)
            best = min(best, word.bit_count())

    for word in memory:
        observe(word)
    order = list(range(width))
    for _ in range(trials):
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
                observe(word)

        words = sorted(set(memory + dependencies),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(words[:32])
        donors = words[:24]
        for start in words[:48]:
            word = start
            for donor in donors:
                candidate = word ^ donor
                if candidate and candidate.bit_count() < word.bit_count():
                    word = candidate
            observe(word)
            discoveries.add(word)
        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:24]
        found = set(sorted(found,
                           key=lambda w: (w.bit_count(), w))[:24])
        if span is None and best <= 6:
            break

    if stabilizers:
        for start in sorted(found, key=int.bit_count)[:6]:
            word = start
            for _ in range(3):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda item: item[0])
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                moves = [row for _, row in nearby[:12]]
                for i, row in enumerate(moves):
                    for other in moves[:i]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                for donor in memory:
                    candidate = word ^ donor
                    cw = candidate.bit_count()
                    if (0 < cw < weight
                            and _c55_outside(candidate, span)):
                        chosen, weight = candidate, cw
                if chosen == word:
                    break
                word = chosen
                observe(word)
            observe(word)

    witnesses = tuple(sorted(found,
                             key=lambda w: (w.bit_count(), w))[:24])
    assert all(_c55_syndrome(cols, w) == 0 for w in witnesses)
    return best, witnesses


def _c55_draw(local, G, architecture, model, side):
    """Pack actual noncommuting supports, with four coordinate policies."""
    quadratic = (1, 3, 4, 5, 9)
    nonquadratic = (2, 6, 7, 8, 10)
    for _ in range(24):
        row, used = [], set()
        for j, (b, d) in enumerate(_C55_CARDS[architecture]):
            accepted = None
            for attempt in range(64):
                if model == 0:
                    a, c = local.randrange(11), local.randrange(11)
                elif model == 1:
                    a, c = local.sample(range(1, 11), 2)
                elif model == 2:
                    a = local.randrange(11)
                    c = (a + (1, 3, 5)[j]
                         + 2 * architecture + attempt % 3) % 11
                else:
                    a = local.choice(quadratic)
                    c = local.choice(nonquadratic)
                g, h = (a, b), (c, d)
                cell = (G.e, g, h)
                if len(set(cell)) != 3:
                    continue
                if G.mul(g, h) == G.mul(h, g):
                    continue
                differences = {
                    (G.mul(x, G.inv(y)) if side == "A"
                     else G.mul(G.inv(x), y))
                    for x in cell for y in cell if x != y
                }
                if len(differences) != 6 or differences & used:
                    continue
                accepted = cell
                used.update(differences)
                break
            if accepted is None:
                break
            row.append(accepted)
        if len(row) == 3:
            return (tuple(row),)
    return None


def _c55_signature(hx, hz, n):
    """Uncoloured Tanner invariant, including global closed-walk data.

    Inequality certifies nonisomorphism; equality is rejected
    conservatively. X/Z labels and protograph labels are absent.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [r.bit_count() for r in rows]
    links = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                links[i][j] = links[j][i] = overlap

    keys = []
    for i, neighbors in enumerate(links):
        second = Counter()
        for j, weight in neighbors.items():
            for h, other in links[j].items():
                second[h] += weight * other
        third = sum(second[j] * weight
                    for j, weight in neighbors.items())
        fourth = sum(value * value for value in second.values())
        profile = Counter((weights[j], weight)
                          for j, weight in neighbors.items())
        keys.append((weights[i], third, fourth,
                     tuple(sorted(profile.items()))))

    history = []
    colours = []
    for _ in range(3):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: c for c, key in enumerate(sorted(histogram))}
        colours = [palette[key] for key in keys]
        keys = [
            (colours[i], tuple(sorted(Counter(
                (colours[j], weight)
                for j, weight in neighbors.items()).items())))
            for i, neighbors in enumerate(links)
        ]
    incidents = [[] for _ in range(n)]
    for colour, source in zip(colours, rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length() - 1].append(colour)
            source ^= bit
    return (n, tuple(history), tuple(sorted(Counter(
        tuple(sorted(v)) for v in incidents).items())))


@lru_cache(maxsize=1)
def _c55_bank(G):
    """Shared deterministic search; failed cards never emit orbit copies."""
    assert G.q == 55 and A_SHAPE == B_SHAPE == (1, 3)
    q, width = G.q, 3 * G.q
    mask = (1 << q) - 1
    zero = [[[], [], []]]
    atlas = {}

    # Only verified unit lifts define the two component actions.
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(_c55_columns(
                [row & mask for row in source[:q]], q))
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        out = [0] * width
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for h, value in enumerate(atlas[side][G.idx(g)]):
                    out[j * q + h] ^= value
        return tuple(out)

    @lru_cache(maxsize=None)
    def component_pool(side, architecture, model):
        local = random.Random(
            BASE_SEED * 100003 + 55040091
            + 1009 * architecture + 100003 * model
            + 7919 * int(side == "B"))
        raw, fallback, seen = [], [], set()

        def merit(item):
            reading = min(item["reading"], 12)
            load = sum(w.bit_count() <= reading
                       for w in item["words"])
            return (reading, -load, item["reading"], -item["serial"])

        def admit(grid, serial):
            if grid is None or grid in seen:
                return
            seen.add(grid)
            cols = compile_component(side, grid)
            if not _c55_clean4(cols):
                return
            rank = gf2_rank(cols)
            if rank < q - 2:
                return
            reading, words = _c55_probe(cols, local, 2)
            witness = _c55_exact_word(cols, q, 6)
            if witness:
                reading, words = _c55_probe(
                    cols, local, 1, words + (witness,))
            item = dict(
                grid=grid, cols=cols, rank=rank, reading=reading,
                words=words, witness=witness, serial=serial,
                floor=6 if witness else 8)
            destination = fallback if witness else raw
            destination.append(item)
            destination.sort(key=merit, reverse=True)
            del destination[6 if not witness else 3:]

        for attempt in range(96):
            admit(_c55_draw(local, G, architecture, model, side), attempt)

        for step in range(32):
            replacement = _c55_draw(
                local, G, architecture, model, side)
            if replacement is None:
                continue
            parents = raw if raw else fallback
            if not parents:
                admit(replacement, 96 + step)
                continue
            parent = local.choice(parents[:min(4, len(parents))])
            witnesses = parent["words"][:8]
            if parent["witness"]:
                witnesses = (parent["witness"],) + tuple(witnesses)
            pressure = [
                1 + sum(((word >> (j * q)) & mask).bit_count()
                        for word in witnesses)
                for j in range(3)]
            j = local.choices(range(3), weights=pressure, k=1)[0]
            row = list(parent["grid"][0])
            row[j] = replacement[0][j]
            if step % 7 == 0:
                h = (j + 1 + local.randrange(2)) % 3
                row[h] = replacement[0][h]
            admit((tuple(row),), 96 + step)

        selected = raw if raw else fallback
        for item in selected:
            reading, words = _c55_probe(
                item["cols"], local, 16, item["words"])
            item["reading"] = min(item["reading"], reading)
            item["words"] = words
            if item["floor"] == 8:
                witness = _c55_exact_word(item["cols"], q, 8)
                if witness:
                    reading, words = _c55_probe(
                        item["cols"], local, 1, words + (witness,))
                    item["reading"] = min(item["reading"], reading)
                    item["words"] = words
                else:
                    item["floor"] = 10
            assert item["words"]
            assert item["reading"] >= item["floor"]
        selected.sort(
            key=lambda item: (item["floor"], merit(item)), reverse=True)
        return tuple(selected[:4])

    emitted, bank = set(), []
    counts = [[3, 3, 3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    for slot in range(N_SLOTS):
        architecture, model = divmod(slot % 20, 4)
        lane = slot // 20
        ap = component_pool("A", architecture, model)
        bp = component_pool(
            "B", (architecture + 2 + lane) % 5, (model + 2) % 4)
        local = random.Random(
            BASE_SEED * 100007 + 55040137 + 104729 * slot)
        pairs = sorted(
            ((min(a["reading"], b["reading"]),
              a["floor"] + b["floor"],
              a["reading"] + b["reading"], ai, bi, a, b)
             for ai, a in enumerate(ap) for bi, b in enumerate(bp)),
            key=lambda entry: entry[:5], reverse=True)
        finalists = []

        for _, _, _, _, _, a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (375 * n <= 1000 * k < 425 * n):
                continue

            # Independently check the compiled components against the
            # complete evaluator lift before using constituent words.
            ar = [
                sum(((row >> (3 * j * q)) & mask) << (j * q)
                    for j in range(3))
                for row in hx[:q]]
            br = [row & ((1 << width) - 1) for row in hz[:q]]
            assert _c55_columns(ar, width) == a["cols"]
            assert _c55_columns(br, width) == b["cols"]
            xc, zc = _c55_columns(hz, n), _c55_columns(hx, n)
            if not _c55_clean4(xc) or not _c55_clean4(zc):
                continue
            signature = _c55_signature(hx, hz, n)
            if signature in emitted:
                continue

            xseeds = tuple(
                word << (j * width)
                for word in b["words"][:8] for j in range(3))
            zseeds = tuple(
                sum(((word >> (j * q)) & mask)
                    << ((3 * j + beta) * q) for j in range(3))
                for word in a["words"][:8] for beta in range(3))
            assert all(_c55_syndrome(xc, w) == 0 for w in xseeds)
            assert all(_c55_syndrome(zc, w) == 0 for w in zseeds)
            dx, wx = _c55_probe(xc, local, 3, xseeds, hx)
            dz, wz = _c55_probe(zc, local, 3, zseeds, hz)
            if dx > n or dz > n:
                continue
            finalists.append(dict(
                A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                xc=xc, zc=zc, dx=dx, dz=dz, wx=wx, wz=wz,
                xseeds=xseeds, zseeds=zseeds, a=a, b=b,
                signature=signature))
            if len(finalists) == 4:
                break

        for item in finalists:
            dx, _ = _c55_probe(
                item["xc"], local, 8,
                item["xseeds"] + item["wx"], item["hx"])
            dz, _ = _c55_probe(
                item["zc"], local, 8,
                item["zseeds"] + item["wz"], item["hz"])
            item["dx"] = min(item["dx"], dx)
            item["dz"] = min(item["dz"], dz)

        def merit(item):
            # These are real found witnesses, not exclusion sentinels.
            reading = min(
                item["dx"], item["dz"],
                item["a"]["reading"], item["b"]["reading"], 12)
            return (item["k"] * reading * reading / item["n"],
                    reading, item["k"],
                    item["a"]["floor"] + item["b"]["floor"])

        if not finalists:
            bank.append(None)
            continue
        winner = max(finalists, key=merit)
        A, B = winner["A"], winner["B"]
        assert winner["n"] == 550 and winner["k"] >= 220
        assert 375 * 550 <= 1000 * winner["k"] < 425 * 550
        assert css_ok(winner["hx"], winner["hz"])
        assert max(row.bit_count()
                   for row in winner["hx"] + winner["hz"]) == 12
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(winner["signature"])
        bank.append((A, B))

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this structural card's jointly screened fresh filling."""
    bank = _c55_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("no fresh target-rate cubic design passed this card")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return every complete cell of the selected balanced cubic design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Canonical triangle-free binomial enumeration plus incidence fingerprints supplies sixty distinct improving C7:C4 designs: the measured parent produced thirteen isomorphism classes, voided slots 21-59, and all five measured leaders remained [588,345,4].
# REFUTED: Singular-cell menus and short-logical witness repairs raise C7:C4 fitness above 9.388: all five current measured leaders remained [588,345,4], with only six emitted slots and one isomorphism duplicate.
# REFUTED: Noncommuting product-support repairs and modular resolvent discrimination improve the C7:C4 high-rate bed: all five measured leaders remained [588,345,4], and only seven distinct binomial fillings survived.
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
