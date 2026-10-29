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
CONCEPT = ("Semidihedral SD32, order 32, A=(1,3), B=(2,5): cubic components with a prescribed C8 image deficit should supply five fresh k=200 codes with measured distance readings above eight, improving the 0.350 frontier's 23.529 fitness.")
GROUP = ("New semidihedral construction: x^16=y^2=1 and y*x*y^-1=x^7. This order-32 group differs genuinely from the parent's order-28 C7:C4. As a 2-group, odd-augmentation elements of F2[SD32] are units, motivating cubic pivot blocks. The related Dic8 cubic bed measured k=200 and distance eight; neither its distance nor its score is assumed to transfer.")
SHAPES = ("Target 0.350: (1,3)x(2,5) gives n=544, kb=192 and guaranteed rate 6/17. Components have 96/32 and 160/64 variables/checks. The size window permits q<=35; this family fixes q=32. A four-dimensional B image deficit targets k=200, rate 0.367647, but complete scaffold ranks decide admission. Parent weight-four minima occupied sector-one Acol/Bcol blocks broadly, especially Acol2 and Bcol4.")
FILLING = ("Twenty cards cross five A rotation-order recipes with four B coupling policies. A contains three mixed-coset trinomials. B contains four top C8 binomials, four bottom monomials and one bottom-only trinomial; every component column is cubic. HX weight is twelve and HZ maximum eleven. Compile component actions from verified unit lifts, optimize actual kernel witnesses, perform bounded searches through eight with explicit completion flags, then exclude full logicals through four and search both complete logical quotients with verified embeddings. Unequal uncoloured Tanner invariants distinguish emitted designs; search misses are never distance certificates.")
FAMILY_NAME = "r035_SD32_13x25_C8image_cubic_witness8_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # SD32: x^16=y^2=1, y*x*y^-1=x^7; 7^2=1 mod 16.
    if not valid_q(q):
        raise ValueError("SD32 requires order 32")
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        return (g[0] % 16, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else 7)*c) % 16,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a*(1 if b == 0 else 7)) % 16, b)

    def idx(g):
        a, b = red(g)
        return 16*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

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
    """Pure shared search; the negative index memoizes its immutable bank."""
    if idx >= 0:
        bank = NEW_CTX(None, G, -1)
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


# The historical C7:C4 helpers above are inactive on this new bed.
# This bank and the final NEW_CTX/CELL definitions replace their optimizer.
_SD_A_RECIPES = (
    (1, 1, 1), (1, 1, 2), (1, 2, 2),
    (1, 1, 4), (1, 2, 4),
)
_SD_SA = [[3, 3, 3]]
_SD_SB = [[2, 2, 2, 2, 0], [1, 1, 1, 1, 3]]


def _sd_columns(rows, width):
    out = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            out[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return tuple(out)


def _sd_basis(rows):
    out = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            previous = out.get(p)
            if previous is None:
                out[p] = word
                break
            word ^= previous
    return out


def _sd_outside(word, basis):
    while word:
        p = word.bit_length()-1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _sd_clean4(cols):
    """Exact exclusion of component kernels through four."""
    seen = {0}
    for value in cols:
        if value in seen:
            return False
        seen.add(value)
    for i, value in enumerate(cols):
        for other in cols[:i]:
            pair = value ^ other
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _sd_small_logical(checks, stabilizers, n):
    """Exact logical-existence test through four, allowing stabilizers."""
    span = _sd_basis(stabilizers)
    residues = []
    for c in range(n):
        row = span.get(c)
        if row is None:
            residues.append(1 << c)
        else:
            tail, residue = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                residue ^= residues[bit.bit_length()-1]
                tail ^= bit
            residues.append(residue)

    cols = _sd_columns(checks, n)
    seen = {0: 0}
    for value, residue in zip(cols, residues):
        old = seen.get(value)
        if old is None:
            seen[value] = residue
        elif old != residue:
            return True
    for i, value in enumerate(cols):
        for j in range(i):
            key = value ^ cols[j]
            residue = residues[i] ^ residues[j]
            old = seen.get(key)
            if old is None:
                seen[key] = residue
            elif old != residue:
                return True
    return False


@lru_cache(maxsize=1)
def _sd_atlas(G):
    """Compile unit actions exclusively from the complete verified lift."""
    q, mask = G.q, (1 << G.q)-1
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[], [], []]]
            B = [[[] for _ in range(5)] for _ in range(2)]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 17*q
            source = hx if side == "A" else hz
            cols = _sd_columns([r & mask for r in source[:q]], q)
            assert all(c.bit_count() == 1 for c in cols)
            blocks.append(cols)
        atlas[side] = tuple(blocks)
    return atlas


def _sd_compile(G, atlas, side, grid):
    q = G.q
    cols = [0]*(len(grid[0])*q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                for h, value in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+h] ^= value << (i*q)
    return tuple(cols)


def _sd_cycles(cols):
    counts, total = {}, 0
    for source in cols:
        positions = []
        while source:
            bit = source & -source
            positions.append(bit.bit_length()-1)
            source ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                total += old
                counts[key] = old + 1
    return total


def _sd_probe(cols, rng, trials, seeds=(), stabilizers=None):
    """Actual kernel/logical witnesses; high readings are only upper bounds."""
    span = None if stabilizers is None else _sd_basis(stabilizers)
    memory = sorted({w for w in seeds if w}, key=int.bit_count)[:32]
    order = list(range(len(cols)))
    best = len(cols)+1

    def observe(word):
        nonlocal best
        if (word and word.bit_count() < best
                and (span is None or _sd_outside(word, span))):
            best = word.bit_count()

    for word in memory:
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        assert syndrome == 0
        observe(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                observe(word)

        discoveries = set(dependencies + memory)
        directions = sorted(discoveries, key=int.bit_count)[:40]
        for seed in directions[:16]:
            word = seed
            for sweep in range(2):
                changed = False
                for direction in directions:
                    candidate = word ^ direction
                    if candidate and candidate.bit_count() < word.bit_count():
                        word = candidate
                        changed = True
                if not changed:
                    break
            observe(word)
            discoveries.add(word)

        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    if candidate.bit_count() <= min(best+4, 20):
                        discoveries.add(candidate)

        if stabilizers is not None:
            logical = [w for w in short if _sd_outside(w, span)]
            for word in logical[:4]:
                for sweep in range(2):
                    previous = word.bit_count()
                    nearby = sorted(stabilizers,
                                    key=lambda r: (word ^ r).bit_count())[:12]
                    for i, row in enumerate(nearby):
                        candidate = word ^ row
                        if candidate.bit_count() < word.bit_count():
                            word = candidate
                        for other in nearby[:i]:
                            candidate = word ^ row ^ other
                            if candidate.bit_count() < word.bit_count():
                                word = candidate
                    if word.bit_count() == previous:
                        break
                observe(word)
                discoveries.add(word)
        memory = sorted((w for w in discoveries if w),
                        key=int.bit_count)[:32]
    return best, tuple(memory)


def _sd_bounded_words(cols, q, height, maximum=8, node_cap=24000):
    """Constituent search with explicit completion status.

    Regular actions are transitive within each constituent block, so
    block anchors cover constituent orbits. No corresponding symmetry
    is assumed for the complete quantum code. A cutoff is not exclusion.
    """
    assert _sd_clean4(cols)
    bits = [1 << i for i in range(len(cols))]
    incidence = [0]*height
    tails = {0: 0}
    for i, source in enumerate(cols):
        assert source.bit_count() == 3
        tails[source] = bits[i]
        while source:
            bit = source & -source
            incidence[bit.bit_length()-1] |= bits[i]
            source ^= bit
    for i, value in enumerate(cols):
        for j in range(i):
            key = value ^ cols[j]
            assert key not in tails
            tails[key] = bits[i] | bits[j]

    found, nodes, stopped = set(), 0, False

    def visit(value, word, banned, weight):
        nonlocal nodes, stopped
        if stopped:
            return
        if nodes >= node_cap:
            stopped = True
            return
        nodes += 1
        if not value:
            found.add(word)
            if len(found) >= 8:
                stopped = True
            return
        remaining = maximum-weight
        if remaining <= 0 or value.bit_count() > 3*remaining:
            return
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not tail & banned):
                found.add(word | tail)
                if len(found) >= 8:
                    stopped = True
            return

        source, choices, least = value, 0, len(cols)+1
        while source:
            bit = source & -source
            options = incidence[bit.bit_length()-1] & ~banned
            count = options.bit_count()
            if not count:
                return
            if count < least:
                choices, least = options, count
            source ^= bit

        alternatives = []
        while choices:
            bit = choices & -choices
            alternatives.append(bit.bit_length()-1)
            choices ^= bit
        alternatives.sort(key=lambda i: (-(cols[i] & value).bit_count(), i))
        excluded = banned
        for i in alternatives:
            excluded |= bits[i]
            visit(value ^ cols[i], word | bits[i], excluded, weight+1)
            if stopped:
                break

    for anchor in range(0, len(cols), q):
        visit(cols[anchor], bits[anchor], bits[anchor], 1)
        if stopped:
            break
    words = tuple(sorted(found, key=lambda w: (w.bit_count(), w)))
    for word in words:
        syndrome, source = 0, word
        while source:
            bit = source & -source
            syndrome ^= cols[bit.bit_length()-1]
            source ^= bit
        assert word and syndrome == 0 and word.bit_count() <= maximum
    return words, not stopped, nodes


def _sd_triple(rng, divisor):
    from math import gcd
    a, b = rng.randrange(16), rng.randrange(2)
    d = rng.choice([v for v in range(1, 16) if gcd(v, 16) == divisor])
    return ((a, b), ((a+d) % 16, b), (rng.randrange(16), 1-b))


def _sd_draw(rng, side, recipe, mode, lane):
    if side == "A":
        return (tuple(_sd_triple(rng, d) for d in recipe),)

    gaps = list(((2, 2, 6, 6), (2, 2, 6, 6),
                 (2, 2, 2, 6), (2, 2, 2, 2))[mode])
    rng.shuffle(gaps)
    top, bottom = [], []
    for j, gap in enumerate(gaps):
        a, b = rng.randrange(16), rng.randrange(2)
        delta = rng.choice((gap, 16-gap))
        top.append(((a, b), ((a+delta) % 16, b)))
        if mode == 1:
            # Change coupling occupancy, not a common group transport.
            quotient = (b+j+lane) & 1
        elif mode == 3:
            quotient = (j+lane) & 1
        else:
            quotient = rng.randrange(2)
        bottom.append(((rng.randrange(16), quotient),))
    top.append(())
    bottom.append(_sd_triple(rng, 1 if mode & 1 else 2))
    return (tuple(top), tuple(bottom))


def _sd_pool(rng, G, atlas, side, recipe, mode, lane, quota):
    pool, seen = [], set()

    def admit(grid):
        label = tuple(tuple(sorted(c)) for row in grid for c in row)
        if label in seen:
            return
        seen.add(label)
        cols = _sd_compile(G, atlas, side, grid)
        assert all(c.bit_count() == 3 for c in cols)
        if not _sd_clean4(cols):
            return
        expected = 32 if side == "A" else 60
        if gf2_rank(cols) != expected:
            return
        reading, words = _sd_probe(cols, rng, 2)
        if reading < 6:
            return
        cycles = _sd_cycles(cols)
        item = dict(grid=grid, cols=cols, d=reading, words=words,
                    cycles=cycles, clean8=False)
        pool.append(item)
        pool.sort(key=lambda x: (min(x["d"], 12), -x["cycles"]),
                  reverse=True)
        del pool[10:]

    for _ in range(quota):
        admit(_sd_draw(rng, side, recipe, mode, lane))
    for step in range(quota//3):
        replacement = _sd_draw(rng, side, recipe, mode, lane+step)
        if not pool:
            admit(replacement)
            continue
        parent = rng.choice(pool[:min(4, len(pool))])
        grid = [[tuple(c) for c in row] for row in parent["grid"]]
        if side == "A":
            j = rng.randrange(3)
            grid[0][j] = replacement[0][j]
        elif step % 5 == 0:
            grid[1][4] = replacement[1][4]
        else:
            j = rng.randrange(4)
            grid[1][j] = replacement[1][j]
        admit(tuple(tuple(row) for row in grid))

    for item in pool:
        hits, complete, nodes = _sd_bounded_words(
            item["cols"], G.q, G.q if side == "A" else 2*G.q)
        item["clean8"] = complete and not hits
        item["nodes8"] = nodes
        item["hits8"] = len(hits)
        item["d"], item["words"] = _sd_probe(
            item["cols"], rng, 18, item["words"] + hits)
        if hits:
            item["d"] = min(item["d"], min(w.bit_count() for w in hits))
    pool.sort(key=lambda x: (
        min(x["d"], 12), x["clean8"], -x["hits8"], -x["cycles"]),
        reverse=True)
    return pool[:4]


def _sd_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; unequal values certify nonisomorphism."""
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [r.bit_count() for r in rows]
    links = [dict() for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                links[i][j] = links[j][i] = overlap
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    keys = []
    for i, neighbors in enumerate(links):
        profile = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbors.items())
        keys.append((weights[i], tuple(sorted(profile.items()))))
    history = []
    for _ in range(3):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: c for c, key in enumerate(sorted(histogram))}
        colours = [palette[key] for key in keys]
        keys = [
            (colours[i], tuple(sorted(Counter(
                (colours[j], overlap)
                for j, overlap in links[i].items()).items())))
            for i in range(len(rows))]
    incidents = [[] for _ in range(n)]
    for colour, source in zip(colours, rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length()-1].append(colour)
            source ^= bit
    return (n, tuple(history), tuple(sorted(Counter(
        tuple(sorted(v)) for v in incidents).items())))


@lru_cache(maxsize=1)
def _sd_bank(G):
    """Independently optimize structural cards; never pad with orbit copies."""
    assert G.q == 32
    assert _grid_weights(_SD_SA, _SD_SB) == 12
    assert _no_dead_lines(_SD_SA, _SD_SB)
    atlas = _sd_atlas(G)
    q, mask = G.q, (1 << G.q)-1
    bank, emitted = [], set()

    def embeddings(words, side):
        if side == "B":
            return tuple(word << (j*5*q)
                         for word in words[:12] for j in range(3))
        return tuple(
            sum(((word >> (j*q)) & mask) << ((j*5+beta)*q)
                for j in range(3))
            for word in words[:12] for beta in range(5))

    def merit(item):
        d = min(item["da"], item["db"], item["dx"], item["dz"], 12)
        return (item["k"]*d*d/item["n"], d,
                item["clean8"], -item["cycles"])

    for slot in range(N_SLOTS):
        model, mode = divmod(slot % 20, 4)
        lane = slot//20
        recipe = _SD_A_RECIPES[model]
        rng = random.Random(BASE_SEED*100003 + 32350217 + 104729*slot)
        winner = None
        for retry in range(2):
            ap = _sd_pool(rng, G, atlas, "A", recipe, mode, lane,
                          112+64*retry)
            bp = _sd_pool(rng, G, atlas, "B", recipe, mode, lane,
                          96+64*retry)
            proposals = sorted(
                ((min(a["d"], b["d"]),
                  a["clean8"]+b["clean8"],
                  -a["cycles"]-b["cycles"], ai, bi, a, b)
                 for ai, a in enumerate(ap)
                 for bi, b in enumerate(bp)),
                key=lambda x: x[:5], reverse=True)
            finalists = []
            for _, _, _, _, _, a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 544 or k != 200:
                    continue
                # Verify compiled component actions and column ordering.
                ar = [
                    sum(((row >> (j*5*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
                assert _sd_columns(ar, 3*q) == a["cols"]
                assert _sd_columns(br, 5*q) == b["cols"]
                fp = _sd_fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                if (_sd_small_logical(hz, hx, n)
                        or _sd_small_logical(hx, hz, n)):
                    continue
                xc, zc = _sd_columns(hz, n), _sd_columns(hx, n)
                dx, xwords = _sd_probe(
                    xc, rng, 8, embeddings(b["words"], "B"), hx)
                dz, zwords = _sd_probe(
                    zc, rng, 8, embeddings(a["words"], "A"), hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k, fp=fp,
                    xc=xc, zc=zc, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    da=a["d"], db=b["d"],
                    clean8=a["clean8"]+b["clean8"],
                    cycles=a["cycles"]+b["cycles"]))
                if len(finalists) == 4:
                    break
            for item in finalists:
                dx, _ = _sd_probe(
                    item["xc"], rng, 20, item["xwords"], item["hx"])
                dz, _ = _sd_probe(
                    item["zc"], rng, 20, item["zwords"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["n"] == 544 and winner["k"] == 200
        assert 325*winner["n"] <= 1000*winner["k"] < 375*winner["n"]
        assert max(r.bit_count()
                   for r in winner["hx"]+winner["hz"]) <= W_MAX
        assert [[len(c) for c in row] for row in A] == _SD_SA
        assert [[len(c) for c in row] for row in B] == _SD_SB
        assert all(len(c) == len(set(c)) <= CELL_MAX
                   for grid in (A, B) for row in grid for c in row)
        emitted.add(winner["fp"])
        bank.append(dict(A=A, B=B, k=winner["k"],
                         card=(recipe, mode, lane), selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the final new-group optimizer, not any historical bank."""
    item = _sd_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh rank-200 SD32 design passed this card")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell, including deliberate zero blocks."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Canonical triangle-free binomial enumeration plus incidence fingerprints supplies sixty distinct improving C7:C4 designs: the measured parent produced thirteen isomorphism classes, voided slots 21-59, and all five measured leaders remained [588,345,4].
# REFUTED: Singular-cell menus and exact logical-four witness repairs improve C7:C4 beyond 9.388: the current measured program retained five [588,345,4] leaders, emitted only six slots, and one was an isomorphism duplicate.
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
