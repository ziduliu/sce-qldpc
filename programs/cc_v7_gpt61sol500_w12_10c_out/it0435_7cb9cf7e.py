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
CONCEPT = ("C7:C6, action 3, order 42, A=(1,3), B=(1,4): even-quotient cubic A cells and three odd-quotient B binomials plus an even-quotient trinomial should supply five fresh cell-0.500 codes excluding all full-code kernels through five.")
GROUP = ("Jump from C7:C4 to the faithful affine group C7:C6: 3 has order six modulo seven. The even quotient contains C7:C3; odd-quotient binomial differences have order six rather than four. RUN STATE reports a centre-one, seven-class order-42 bed at these shapes scoring 8.0, so improvement is a filling hypothesis, not an assumed group advantage.")
SHAPES = ("Target cell 0.500: n=13*42=546, kb=252 and guaranteed rate 6/13. Components have 126 and 168 variables with 42 checks each; these shapes admit q<=46. The even-quotient cubic projection supplies a component rank deficit, but only exact assembled ranks inside 0.475<=k/n<0.525 admit a code. Parent minima occupied sector-one blocks broadly, especially Acol2 and Bcol4; replace both component architectures.")
FILLING = ("Twenty cards cross five A coefficient architectures with four even-quotient B trinomial geometries. A counts are (3,3,3), B counts (2,2,2,3): both row sums nine, HX=11/12 and HZ=12. Compile component actions only from verified unit lifts; retain short-kernel-free components and screen complete verified matrices exhaustively through five without a node cutoff. This conservative screen rejects small stabilizers too. Select verified target-band ranks and unequal uncoloured Tanner invariants; six is an exclusion floor, never a sampled upper distance reading.")
FAMILY_NAME = "r050_F42_13x14_even_cubic_odd_sidon_exact5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # (a,b) acts on F7 as z -> 3^b*z+a.
    # 3 has order six modulo seven.
    if not valid_q(q):
        raise ValueError("C7:C6 requires order 42")
    els = [(a, b) for b in range(6) for a in range(7)]
    action = [pow(3, b, 7) for b in range(6)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-action[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)

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
def _retired_C7C4_ctx(rng, G, idx):
    """Inactive historical search; never called by the new group policy."""
    if idx >= 0:
        bank = _retired_C7C4_ctx(None, G, -1)
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
        return (n, tuple(history), tuple(sorted(Counter(
            tuple(sorted(v)) for v in qubits).items())))

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
        for attempt in range(24):
            policy = attempt % 4
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
        for step in range(24):
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
                count_grid[j], side, (card + step) % 4, card + step))
            if step % 6 == 5:
                other = (j + 1 + local.randrange(len(count_grid)-1)) \
                        % len(count_grid)
                target[0][other] = list(draw_cell(
                    count_grid[other], side, (step + 1) % 4, step))
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


def _six_columns(rows, width):
    """Transpose matrices obtained from the verified construction."""
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return tuple(cols)


def _six_no_small(cols, through=5, anchors=None):
    """Exhaustive conservative kernel exclusion through four or five.

    No cutoff is used. Unique singleton/pair syndromes exclude all
    kernels through four. An odd check drives the remaining completion
    search; earlier alternatives are banned to partition the search.
    This rejects small stabilizers as well as small logicals.
    """
    width = len(cols)
    bits = [1 << j for j in range(width)]
    tails = {0: 0}
    for j, value in enumerate(cols):
        if value in tails:
            return False
        tails[value] = bits[j]
    for j, value in enumerate(cols):
        for h in range(j):
            key = value ^ cols[h]
            if key in tails:
                return False
            tails[key] = bits[j] | bits[h]

    if through <= 4:
        return True
    assert through == 5
    # With exclusively odd-weight columns every kernel has even weight.
    if all(value.bit_count() & 1 for value in cols):
        return True

    height = max(value.bit_length() for value in cols)
    rows = [0]*height
    maximum = max(value.bit_count() for value in cols)
    for j, source in enumerate(cols):
        while source:
            bit = source & -source
            rows[bit.bit_length()-1] |= bits[j]
            source ^= bit

    def exists(value, remaining, banned):
        if not value:
            return True
        if remaining <= 0 or value.bit_count() > maximum*remaining:
            return False
        if remaining <= 2:
            tail = tails.get(value)
            return (tail is not None
                    and tail.bit_count() <= remaining
                    and not (tail & banned))

        pending, choices, least = value, 0, width+1
        while pending:
            bit = pending & -pending
            available = rows[bit.bit_length()-1] & ~banned
            count = available.bit_count()
            if not count:
                return False
            if count < least:
                choices, least = available, count
                if count == 1:
                    break
            pending ^= bit

        options = []
        while choices:
            bit = choices & -choices
            options.append(bit.bit_length()-1)
            choices ^= bit
        options.sort(key=lambda j: (-(cols[j] & value).bit_count(), j))
        excluded = banned
        for j in options:
            excluded |= bits[j]
            if exists(value ^ cols[j], remaining-1, excluded):
                return True
        return False

    roots = range(width) if anchors is None else anchors
    return not any(exists(cols[j], 4, bits[j]) for j in roots)


@lru_cache(maxsize=1)
def _six_atlas(G):
    """Compile both constituent actions from VERIFIED single-term lifts."""
    q, mask = G.q, (1 << G.q)-1
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            zero, unit = [[[]]], [[[g]]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 2*q
            source = hx if side == "A" else hz
            block = _six_columns([r & mask for r in source[:q]], q)
            assert all(value.bit_count() == 1 for value in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)
    return atlas


def _six_component(G, atlas, side, grid):
    """F2 superposition of scaffold-generated actions, not another lift."""
    q = G.q
    cols = [0]*(len(grid[0])*q)
    for j, cell in enumerate(grid[0]):
        for g in cell:
            for h, value in enumerate(atlas[side][G.idx(g)]):
                cols[j*q+h] ^= value
    return tuple(cols)


def _six_cycles(cols):
    """Actual constituent Tanner four-cycle count, only a tie-breaker."""
    total = 0
    for j, col in enumerate(cols):
        for other in cols[:j]:
            overlap = (col & other).bit_count()
            total += overlap*(overlap-1)//2
    return total


def _six_draw_a(local, G, geometry):
    """Five coefficient architectures inside the even C7:C3 subgroup."""
    if geometry == 0:
        aa = [local.randrange(7) for _ in range(3)]
        bb = [local.choice([b for b in range(7) if b != 3*a % 7])
              for a in aa]
    elif geometry == 1:
        aa = local.sample(range(7), 3)
        bb = [local.choice([b for b in range(7) if b != 3*a % 7])
              for a in aa]
    elif geometry == 2:
        base, step = local.randrange(7), local.randrange(1, 7)
        aa = [(base+t*step) % 7 for t in range(3)]
        bb = [local.choice([b for b in range(7) if b != 3*a % 7])
              for a in aa]
    elif geometry == 3:
        aa = local.sample(range(7), 3)
        delta = local.randrange(1, 7)
        bb = [(3*a+delta) % 7 for a in aa]
    else:
        aa = local.sample(range(7), 3)
        deltas = local.sample(range(1, 7), 3)
        bb = [(3*a+d) % 7 for a, d in zip(aa, deltas)]

    row = []
    for a, b in zip(aa, bb):
        cell = (G.e, (a, 2), (b, 4))
        assert len(set(cell)) == 3
        assert G.mul(cell[1], cell[2]) != G.mul(cell[2], cell[1])
        row.append(cell)
    return (tuple(row),)


def _six_draw_b(local, G, model, sidon):
    """Three degree-two columns with order-six differences, then a cubic."""
    relatives = local.choice(sidon)
    row = [(G.e, (a, 1)) for a in relatives]
    # Different quotient occupancies are intrinsic support experiments.
    patterns = ((0, 2, 4), (0, 0, 2),
                (0, 0, 4), (0, 0, 0))
    _, b, d = patterns[model]
    if model == 0:
        a = local.randrange(7)
        c = local.choice([h for h in range(7) if h != 3*a % 7])
    elif model in (1, 2):
        a, c = local.randrange(1, 7), local.randrange(7)
    else:
        legal = [
            (a, c) for a in range(1, 7) for c in range(1, 7)
            if len({(x-y) % 7 for x in (0, a, c)
                    for y in (0, a, c) if x != y}) == 6]
        a, c = local.choice(legal)
    cell = (G.e, (a, b), (c, d))
    assert len(set(cell)) == 3
    row.append(cell)
    return (tuple(row),)


def _six_signature(hx, hz, n):
    """Uncoloured Tanner invariant: inequality certifies nonisomorphism.

    Weighted closed walks and common-neighbour counts supplement the
    inherited degree/overlap invariant. Equal signatures are discarded
    conservatively, not asserted to establish isomorphism.
    """
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    links = [dict() for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                links[i][j] = links[j][i] = overlap
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    profiles = []
    for i, neighbours in enumerate(links):
        local = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbours.items())
        second = Counter()
        for j, overlap in neighbours.items():
            for h, other in links[j].items():
                second[h] += overlap*other
        third = sum(second[j]*overlap
                    for j, overlap in neighbours.items())
        fourth = sum(value*value for value in second.values())
        profiles.append(
            (weights[i], tuple(sorted(local.items())), third, fourth))
    return (n, tuple(sorted(Counter(profiles).items())),
            _binomial_graph_profile(hx, hz, n))


@lru_cache(maxsize=1)
def _six_bank(G):
    """Pure deterministic bank of independently optimized structural cards."""
    assert G.q == 42 and A_SHAPE == (1, 3) and B_SHAPE == (1, 4)
    q, mask = G.q, (1 << G.q)-1
    atlas = _six_atlas(G)
    sa, sb = [[3, 3, 3]], [[2, 2, 2, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    # Three C7 residues with all six ordered differences distinct.
    # Together with quotient-one relatives this prevents the binomial
    # control's immediate four-cycle obstruction. Actual screens decide.
    sidon = tuple(
        triple for triple in combinations(range(7), 3)
        if len({(a-b) % 7 for a in triple for b in triple if a != b}) == 6)
    assert sidon
    emitted, bank = set(), []

    for slot in range(N_SLOTS):
        geometry, model = divmod(slot % 20, 4)
        local = random.Random(BASE_SEED*100003 + 42050193 + 104729*slot)
        chosen = None

        def pool(side, batch):
            keep, labels_seen = [], set()
            quota = ((96+48*batch) if side == "A"
                     else (160+64*batch))
            for attempt in range(quota):
                grid = (_six_draw_a(local, G, geometry) if side == "A"
                        else _six_draw_b(local, G, model, sidon))
                labels = tuple(tuple(sorted(cell)) for cell in grid[0])
                if labels in labels_seen:
                    continue
                labels_seen.add(labels)
                cols = _six_component(G, atlas, side, grid)
                rank = gf2_rank(cols)
                if side == "A":
                    # The quotient gives rank<=38, not a full-code k formula.
                    # Keep a narrow useful surplus range; assembled rank gates
                    # below remain the authority on rate.
                    if not (34 <= rank <= 38):
                        continue
                elif rank != q:
                    continue

                # A constituent regular action is transitive within each
                # q-column block, so one anchor per block covers its support
                # orbits. This reduction is NEVER used on the full code.
                anchors = range(0, len(cols), q)
                if not _six_no_small(cols, 5, anchors):
                    continue
                cycles = _six_cycles(cols)
                priority = ((-rank, -cycles, -attempt) if side == "A"
                            else (-cycles, -attempt))
                keep.append(dict(
                    grid=grid, cols=cols, rank=rank, cycles=cycles,
                    priority=priority))
                keep.sort(key=lambda item: item["priority"], reverse=True)
                del keep[6:]
            return keep

        for batch in range(3):
            ap, bp = pool("A", batch), pool("B", batch)
            pairs = sorted(
                ((a["rank"], a["cycles"]+b["cycles"], ai, bi, a, b)
                 for ai, a in enumerate(ap) for bi, b in enumerate(bp)),
                key=lambda entry: entry[:4])
            finalists = []
            for _, cycles, _, _, a, b in pairs[:18]:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 546 or not (475*n <= 1000*k < 525*n):
                    continue

                # Confirm component compilation against complete matrices.
                ar = [
                    sum(((row >> (4*j*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (4*q))-1) for row in hz[:q]]
                assert _six_columns(ar, 3*q) == a["cols"]
                assert _six_columns(br, 4*q) == b["cols"]

                # Complete-code tests cover EVERY variable in BOTH sectors.
                # No constituent-distance assumption substitutes for these.
                if not _six_no_small(_six_columns(hx, n), 5):
                    continue
                if not _six_no_small(_six_columns(hz, n), 5):
                    continue
                fingerprint = _six_signature(hx, hz, n)
                if fingerprint in emitted:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    fingerprint=fingerprint, cycles=cycles))
                if len(finalists) == 3:
                    break

            if not finalists:
                continue
            # All admitted matrices have the same exact exclusion floor.
            # Rank and cycle load select among them; no search miss earns
            # an invented upper distance estimate.
            winner = max(finalists,
                         key=lambda item: (item["k"], -item["cycles"]))
            assert css_ok(winner["hx"], winner["hz"])
            assert lifted_k(G, winner["A"], winner["B"]) == winner["k"]
            assert max(row.bit_count()
                       for row in winner["hx"]+winner["hz"]) <= W_MAX
            assert all(len(cell) == len(set(cell)) <= CELL_MAX
                       for grid in (winner["A"], winner["B"])
                       for row in grid for cell in row)
            assert 475*546 <= 1000*winner["k"] < 525*546
            emitted.add(winner["fingerprint"])
            chosen = tuple(
                tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for grid in (winner["A"], winner["B"]))
            break

        # Preserve failed card indices; never pad with an orbit or archive copy.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this card's fresh, target-band, full-five-excluded design."""
    item = _six_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh full-five-excluded filling for this card")
    A, B = item
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional random labeling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Canonical triangle-free binomial enumeration plus incidence fingerprints supplies sixty distinct improving C7:C4 designs: the measured parent produced thirteen isomorphism classes, voided slots 21-59, and all five measured leaders remained [588,345,4].
# REFUTED: Singular-cell menus and short-logical witness repairs raise C7:C4 fitness above 9.388: the current program emitted six slots collapsing to five classes, and every measured leader remained [588,345,4].
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
