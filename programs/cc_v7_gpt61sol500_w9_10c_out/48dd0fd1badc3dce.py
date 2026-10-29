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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dicyclic Dic27, order 108, A=B=(1,2): independently optimized noncommuting trinomials should supply five novel distance readings >=16 in cell 0.200. At k=108 this would score 51.2; neither that reading nor catalogue novelty is a distance certificate.")
GROUP = ("Jump from cyclic C11 to Dic27: x^54=1, y^2=x^27, y*x*y^-1=x^-1. Order 108 gives n=540 and is absent from BEDS. Strong nonabelian order-105/110/117 (1,2)x(1,2) beds motivate this shape and density, not a distance guarantee for this different group. Long rotation ratios have order 27 or 54; outside elements have order four.")
SHAPES = ("Equal (1,2) constituents have 216 variables and 108 checks. Size factor five allows q<=119; this family fixes q=108. Rank counting gives kb=108 and guaranteed rate 0.200. Verified constituent and full ranks enforce the target band. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly; replacing both degree-two incidence constituents targets that bottleneck rather than moving its labels.")
FILLING = ("Twenty ratio-pattern cards independently optimize mixed rotation/outside and two-outside trinomials. Every cell has three distinct elements: A-row+B-column=6+3=9 and B-row+A-column=6+3=9, below both nine and twelve. All component actions come from verified lifts. Exact kernel-four exclusion, full constituent ranks, equal-budget kernel-witness refinement, full logical-class probes and stabilizer descent guide selection. An immutable portfolio rejects repeated full uncoloured Tanner invariants and exact fillings; final lifts verify CSS, both budgets and actual target rate. Unknown archive coverage and larger measured readings remain hypotheses.")
FAMILY_NAME = "r020_dic27_12x12_noncommuting_trinomial_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 108

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic27 requires order 108")
    # (a,b)=x^a*y^b; x^54=1, y^2=x^27, y*x*y^-1=x^-1.
    m, modulus = 27, 54
    els = [(a, b) for b in range(2) for a in range(modulus)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + m*(b // 2)) % modulus, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + m*b*d) % modulus,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a + m) % modulus, b)

    def idx(g):
        a, b = red(g)
        return modulus*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _component_spec(rng, mode, q):
    """Matching columns, or two cycle columns plus five matchings."""
    if mode == 0:
        return (), tuple(sorted(rng.sample(range(q), 7)))
    if mode == 1:
        d = rng.randrange(1, q)
        phase = rng.randrange(q)
        # Five independent vertices of the eleven-cycle with step d.
        labels = tuple(sorted((phase + d * t) % q
                              for t in (0, 2, 4, 6, 8)))
        return (d, d), labels
    return (rng.randrange(1, q), rng.randrange(1, q)), \
        tuple(sorted(rng.sample(range(q), 5)))


def _component_cells(G, spec):
    ds, labels = spec
    e, els = G.e, G.elements
    if not ds:
        return [[[e] for _ in labels],
                [[els[s]] for s in labels]]
    # Column weights remain two; both row weights remain seven.
    return [[[e, els[ds[0]]]] + [[e] for _ in labels] + [[]],
            [[]] + [[els[s]] for s in labels] + [[e, els[ds[1]]]]]


def _translation_key(labels, q):
    return min(tuple(sorted((s + t) % q for s in labels))
               for t in range(q))


def _scaled_spec_key(spec, u, q):
    ds, labels = spec
    s = tuple((u * x) % q for x in labels)
    forward = _translation_key(s, q)
    backward = _translation_key(tuple((-x) % q for x in s), q)
    if not ds:
        return (0, min(forward, backward))
    d0, d1 = ((u * d) % q for d in ds)
    d0, d1 = min(d0, q - d0), min(d1, q - d1)
    return min((1, d0, d1, forward),
               (1, d1, d0, backward))


def _pair_key(a, b, q):
    """Remove translations, row swaps, common automorphisms and A/B swap."""
    keys = []
    for u in range(1, q):
        ka = _scaled_spec_key(a, u, q)
        kb = _scaled_spec_key(b, u, q)
        keys.append(tuple(sorted((ka, kb))))
    return min(keys)


def _tanner_signature(hx, hz):
    """An uncoloured Tanner-isomorphism invariant, not an isomorphism test.

    Check intersections define two adjacency relations. Each check's
    profile counts common neighbours in both relations for every other
    check. Different signatures certify non-isomorphic Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                adj = one if c == 1 else two
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            kind = ((one[i] >> j) & 1) + 2 * ((two[i] >> j) & 1)
            key = (kind,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & two[j]).bit_count()
                   + (two[i] & one[j]).bit_count(),
                   (two[i] & two[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(tuple(sorted(p.items())) for p in profiles))


def _no_short_kernel(rows, n):
    """Exactly exclude kernel vectors of weights one, two and three."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    seen = set(columns)
    if 0 in seen or len(seen) != n:
        return False
    for i, c in enumerate(columns):
        for j in range(i):
            if (c ^ columns[j]) in seen:
                return False
    return True


def _retired_c11_joint_context(rng, G, idx):
    """Inactive historical C11 catalogue; never fills the new bed.

    Constituent automorphisms are quotiented before pairing, but their
    RELATIVE scaling is retained. Different intrinsic fingerprints
    certify nonisomorphism; equal fingerprints are discarded.
    """
    from itertools import combinations
    from collections import Counter

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    element = {G._key(g): g for g in G.elements}

    def x(t):
        return element[(t % q, 0)]

    def label_key(values):
        # Every minimum translated tuple starts at zero.
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    def regular(upper, lower, labels):
        p = len(upper)
        assert len(lower) == p and 2*p + len(labels) == 7
        grid = [
            [[e, x(d)] for d in upper]
            + [[e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(s)] for s in labels]
            + [[e, x(d)] for d in lower],
        ]
        assert all(sum(map(len, row)) == 7 for row in grid)
        assert all(len(grid[0][j]) + len(grid[1][j]) == 2
                   for j in range(7))
        return grid

    def scaled(card, u):
        upper, lower, labels = card
        a = tuple(sorted(min(u*d % q, (-u*d) % q) for d in upper))
        b = tuple(sorted(min(u*d % q, (-u*d) % q) for d in lower))
        return a, b, label_key(tuple(u*s % q for s in labels))

    def canonical(card):
        images = []
        for u in range(1, q):
            a, b, values = scaled(card, u)
            images.append((a, b, values))
            images.append((b, a, label_key(
                tuple((-s) % q for s in values))))
        return min(images)

    def signed_steps(steps):
        return {d % q for d in steps} | {(-d) % q for d in steps}

    # Unlike the parent, upper and lower internal steps are independent.
    # Anchoring one matching at zero removes redundant translations.
    descriptors = set()
    for p in range(3):
        options = []
        for steps in combinations(range(1, (q + 1)//2), p):
            signed = signed_steps(steps)
            if any((a+b) % q in signed for a in signed for b in signed):
                continue
            options.append(steps)
        for ai, upper in enumerate(options):
            for lower in options[ai:]:
                forbidden = signed_steps(upper) | signed_steps(lower)
                for tail in combinations(range(1, q), 6 - 2*p):
                    labels = (0,) + tail
                    if any((a-b) % q in forbidden
                           for a, b in combinations(labels, 2)):
                        continue
                    descriptors.add(canonical((upper, lower, labels)))

    bank = []
    component_mask = (1 << (7*q)) - 1
    for card in sorted(descriptors):
        A = regular(*card)
        # Exact constituent action from the scaffold, with a zero side.
        rows, _, _ = lifted_matrices(G, A, [[[]]])
        rows = [row & component_mask for row in rows]
        if gf2_rank(rows) != 2*q - 1:
            continue
        if not _no_short_kernel(rows, 7*q):
            continue
        bank.append(card)
    bank.sort(key=lambda c: (c[0] == c[1], -len(c[0]), c))
    if not bank:
        raise ValueError("no connected triangle-free regular constituents")

    def columns(rows, width):
        cols = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= tag
                row ^= bit
        return cols

    def joint_fingerprint(hx, hz, width):
        """Intrinsic invariant extracted from the complete VERIFIED lift.

        Regular codes have degree-four and degree-fourteen qubits.
        Every check meets exactly two degree-fourteen qubits. Contract
        that check to a weighted edge. Any Tanner isomorphism preserves
        this multigraph, without needing to preserve X/Z check types.

        Rooted walk diagonals through length twelve and joint walk
        profiles through length eight are permutation invariants.
        Translation symmetry only accelerates their exact computation.
        """
        rows = hx + hz
        cols = columns(rows, width)
        degrees = tuple(sorted(Counter(c.bit_count() for c in cols).items()))
        high = [j for j, c in enumerate(cols) if c.bit_count() == 14]
        if len(high) != 4*q:
            raise ValueError("regular lift lost its intrinsic degree class")
        position = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        nv = len(high)
        adjacency = [Counter() for _ in range(nv)]
        for row in rows:
            bits = row & high_mask
            if bits.bit_count() != 2:
                raise ValueError("check does not contract to a high-degree edge")
            first = bits & -bits
            u = position[first.bit_length() - 1]
            v = position[(bits ^ first).bit_length() - 1]
            adjacency[u][v] += 1
            adjacency[v][u] += 1
        neighbours = [tuple(sorted(a.items())) for a in adjacency]

        # Verify the symmetry on the EXTRACTED graph before exploiting it.
        symmetric = all(
            adjacency[i*q+h].get(j*q+t, 0)
            == adjacency[i*q].get(j*q+(t-h) % q, 0)
            for i in range(4) for j in range(4)
            for h in range(q) for t in range(q))
        roots = list(range(0, nv, q)) if symmetric else list(range(nv))
        diagonals, pair_features = [], []
        for root in roots:
            vector = [0] * nv
            vector[root] = 1
            diagonal = []
            features = [[] for _ in range(nv)]
            for length in range(1, 13):
                following = [0] * nv
                for v, count in enumerate(vector):
                    if count:
                        for w, multiplicity in neighbours[v]:
                            following[w] += count * multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 8:
                    for v, count in enumerate(vector):
                        features[v].append(count)
            diagonals.append(tuple(diagonal))
            pair_features.append(features)

        colors = [
            (sum(adjacency[v].values()),
             diagonals[v//q if symmetric else v])
            for v in range(nv)]
        profiles = []
        for ri, root in enumerate(roots):
            profile = (
                colors[root],
                tuple(sorted((colors[v], tuple(pair_features[ri][v]))
                             for v in range(nv))))
            profiles.extend([profile] * (q if symmetric else 1))
        return (
            tuple(sorted(Counter(row.bit_count() for row in rows).items())),
            degrees, tuple(sorted(profiles)))

    def matching(values):
        return regular((), (), values)

    def ring(d0, d1, values):
        return regular((d0,), (d1,), values)

    # Shown archived regular distance-four classes. Singleton codes
    # have a different intrinsic qubit-degree distribution and cannot
    # be isomorphic to any candidate submitted by this catalogue.
    two = regular((1, 3), (1, 3), (0, 2, 4))
    other_two = regular((1, 3), (1, 3), (0, 2, 6))
    archived = [
        (matching((0, 1, 2, 4, 5, 6, 10)),
         matching((0, 2, 3, 5, 6, 7, 9))),
        (ring(10, 10, (0, 2, 5, 7, 9)),
         ring(10, 10, (0, 2, 4, 7, 9))),
        (matching(tuple(range(7))), two),
        (ring(1, 1, (0, 2, 4, 6, 8)), two),
        (two, two),
        (ring(1, 1, (0, 2, 4, 6, 8)), other_two),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
    ]
    archive_keys = set()
    for A, B in archived:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(joint_fingerprint(hx, hz, n))

    pairs = [(ai, bi) for ai in range(len(bank))
             for bi in range(ai, len(bank))]
    pairs.sort(key=lambda ab: (
        bank[ab[0]][0] == bank[ab[0]][1]
        and bank[ab[1]][0] == bank[ab[1]][1],
        -(len(bank[ab[0]][0]) + len(bank[ab[1]][0])), ab))
    scales = tuple(range(2, q-1)) + (1, q-1)

    graph_keys, orbit_keys = set(), set()
    accepted = 0
    # Interleave intrinsic component pairs and relative orientations.
    # Scaling B alone is not treated as a common group automorphism.
    for turn in range(len(scales)):
        for ai, bi in pairs:
            a = bank[ai]
            u = scales[(turn + 3*ai + bi) % len(scales)]
            b = scaled(bank[bi], u)
            orbit = min(tuple(sorted((scaled(a, v), scaled(b, v))))
                        for v in range(1, q))
            if orbit in orbit_keys:
                continue
            orbit_keys.add(orbit)
            A, B = regular(*a), regular(*b)
            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            if n != 583 or max(r.bit_count() for r in hx + hz) > cap:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != 287 or not (475*n <= 1000*k < 525*n):
                continue
            signature = joint_fingerprint(hx, hz, n)
            if signature in archive_keys or signature in graph_keys:
                continue
            if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
                continue
            graph_keys.add(signature)

            if accepted == idx:
                if not css_ok(hx, hz):
                    raise ValueError("verified joint-voltage lift failed CSS")
                return {"A": A, "B": B, "k": k,
                        "card": (a, b, u), "kernel_floor": 4,
                        "joint_walk_depth": 12}
            accepted += 1

    # Exhaustion voids a slot rather than publishing an archived copy.
    raise ValueError("novel regular joint-voltage catalogue exhausted")


def _d108_columns(rows, width):
    columns = [0] * width
    for i, row in enumerate(rows):
        tag = 1 << i
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    return columns


def _d108_clear4(rows, width):
    """Exactly exclude every nonzero kernel word of weights one to four."""
    columns = _d108_columns(rows, width)
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(columns):
        for second in columns[:i]:
            syndrome = first ^ second
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _d108_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    return pivots


def _d108_normal(word, pivots):
    result = 0
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            bit = 1 << p
            result |= bit
            word ^= bit
        else:
            word ^= old
    return result


def _d108_probe(local, checks, width, passes, stabilizers=(), seeds=()):
    """Heuristic search for actual kernel or nontrivial logical witnesses.

    Recorded weights are upper readings. Logical labels are exact
    stabilizer normal forms; no hand-written lift is used.
    """
    columns = _d108_columns(checks, width)
    span = _d108_span(stabilizers)
    best, weight = 0, width + 1

    def label(word):
        return _d108_normal(word, span) if span else word

    def record(word, logical):
        nonlocal best, weight
        if word and logical:
            w = word.bit_count()
            if w < weight:
                best, weight = word, w

    for word in seeds:
        if word and all((row & word).bit_count() % 2 == 0
                        for row in checks):
            record(word, label(word))

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
        pivots, basis = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                logical = label(word)
                basis.append((word, logical))
                record(word, logical)

        # Basis exchanges preserve the kernel and track exact classes.
        local.shuffle(basis)
        for i, (word, logical) in enumerate(basis):
            w = word.bit_count()
            for j, (other, other_class) in enumerate(basis):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, logical, w = trial, logical ^ other_class, tw
            basis[i] = word, logical
            record(word, logical)

        short = sorted(basis, key=lambda z: z[0].bit_count())[:32]
        for i, (first, first_class) in enumerate(short):
            for j in range(i):
                pair = first ^ short[j][0]
                pair_class = first_class ^ short[j][1]
                record(pair, pair_class)
                if i < 24:
                    for h in range(j):
                        record(pair ^ short[h][0],
                               pair_class ^ short[h][1])

        if best:
            starting = best
            starting_class = label(starting)
            for first, first_class in short:
                record(starting ^ first, starting_class ^ first_class)
                for second, second_class in short:
                    record(starting ^ first ^ second,
                           starting_class ^ first_class ^ second_class)

    # Reduce an actual logical representative inside its stabilizer coset.
    if best and stabilizers:
        for _ in range(3):
            previous = best
            for row in stabilizers:
                trial = best ^ row
                tw = trial.bit_count()
                if 0 < tw < weight:
                    best, weight = trial, tw
            starting = best
            moves = [row for row in stabilizers
                     if (row & starting).bit_count() >= 2]
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    trial = starting ^ first ^ second
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        best, weight = trial, tw
            if best == previous:
                break

    if best:
        assert label(best)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
    return weight, best


def _d108_graph_key(hx, hz, width):
    """Intrinsic full-Tanner invariant, not a complete isomorphism test."""
    from collections import Counter
    import hashlib

    rows = list(hx) + list(hz)
    columns = _d108_columns(rows, width)
    m = len(rows)
    weights = [r.bit_count() for r in rows]
    degree_counts = Counter(c.bit_count() for c in columns)
    row_colors = []
    for row in rows:
        degrees = Counter()
        while row:
            bit = row & -row
            degrees[columns[bit.bit_length() - 1].bit_count()] += 1
            row ^= bit
        row_colors.append(tuple(sorted(degrees.items())))

    one, many = [0]*m, [0]*m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps.append((i, j, overlap))
            if overlap:
                relation = one if overlap == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i, j, overlap in overlaps:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(weights[j], row_colors[j], overlap, common)] += 1
        profiles[j][(weights[i], row_colors[i], overlap, common)] += 1

    intrinsic = (
        tuple(sorted(degree_counts.items())),
        tuple(sorted(
            (weights[i], row_colors[i], tuple(sorted(profiles[i].items())))
            for i in range(m))),
    )
    return hashlib.sha256(repr(intrinsic).encode("ascii")).digest()


def _d108_design(local, G, idx, excluded):
    """Optimize a ratio-pattern card using only verified matrix actions."""
    q, width = G.q, 2*G.q
    cap = min(9, W_MAX)
    card, mode = idx % 20, idx % 5
    elements = {G._key(g): g for g in G.elements}
    unit = [[[G.e]]]
    mask = (1 << width) - 1

    def draw(side):
        row = []
        for j in range(2):
            parity = (card // 5 + j + int(side == "B")) % 2
            rotations = [a for a in range(1, 54)
                         if a % 3 and a % 2 == parity]
            ratio = local.choice(rotations)
            residue = (card + j + int(side == "B")) % 3
            anchor = local.choice([a for a in range(54)
                                   if a % 3 == residue])
            two_outside = (
                mode == 3
                or (mode == 2 and j == int(side == "B"))
                or (mode == 4 and side == "B" and j == 1))
            if two_outside:
                cell = [G.e, elements[(anchor, 1)],
                        elements[((anchor + ratio) % 54, 1)]]
            else:
                cell = [G.e, elements[(ratio, 0)],
                        elements[(anchor, 1)]]
            assert len({G._key(g) for g in cell}) == 3
            row.append(cell)
        return [row]

    def component(grid, side):
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
        return [row & mask for row in rows[:q]]

    def filling_key(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def cycle_cost(rows):
        cost = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                cost += overlap*(overlap - 1)//2
        return cost

    def bank(side):
        raw, seen = [], set()
        for trial in range(48):
            grid = draw(side)
            key = filling_key(grid)
            if key in seen:
                continue
            seen.add(key)
            rows = component(grid, side)
            if gf2_rank(rows) != q or not _d108_clear4(rows, width):
                continue
            raw.append(dict(grid=grid, rows=rows, key=key,
                            cost=cycle_cost(rows), trial=trial))
        raw.sort(key=lambda z: (z["cost"], z["trial"]))

        shortlist = []
        # Equal search budgets, rather than selecting noisy unequal probes.
        for entry in raw[:12]:
            reading, witness = _d108_probe(
                local, entry["rows"], width, 3)
            if not witness:
                continue
            entry.update(reading=reading, witness=witness)
            shortlist.append(entry)
        shortlist.sort(
            key=lambda z: (z["reading"], -z["cost"]), reverse=True)

        finalists = shortlist[:3]
        for entry in finalists:
            reading, witness = _d108_probe(
                local, entry["rows"], width, 12,
                seeds=(entry["witness"],))
            entry.update(reading=reading, witness=witness)
        finalists.sort(
            key=lambda z: (z["reading"], -z["cost"]), reverse=True)
        return finalists

    aa, bb = bank("A"), bank("B")
    if not aa or not bb:
        return None
    plans = []
    for a in aa:
        for b in bb:
            preliminary = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                -a["cost"] - b["cost"])
            plans.append((preliminary, a, b))
    plans.sort(key=lambda z: z[0], reverse=True)

    survivors, signatures = [], set()
    for preliminary, a, b in plans:
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 540 and k > 0
                and 175*n <= 1000*k < 225*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if not (_d108_clear4(hx, n) and _d108_clear4(hz, n)):
            continue
        signature = _d108_graph_key(hx, hz, n)
        if signature in excluded or signature in signatures:
            continue
        signatures.add(signature)
        if not css_ok(hx, hz):
            raise ValueError("verified Dic27 lift failed CSS")

        xseeds = [b["witness"] << (block*2*q) for block in range(2)]
        zseeds = []
        for block in range(2):
            word, bits = 0, a["witness"]
            while bits:
                bit = bits & -bits
                j, h = divmod(bit.bit_length() - 1, q)
                word |= 1 << ((2*j + block)*q + h)
                bits ^= bit
            zseeds.append(word)

        dx, wx = _d108_probe(
            local, hz, n, 4, stabilizers=hx, seeds=xseeds)
        dz, wz = _d108_probe(
            local, hx, n, 4, stabilizers=hz, seeds=zseeds)
        if not wx or not wz:
            continue
        reading = min(dx, dz)
        quality = (k*reading*reading, reading, preliminary)
        survivors.append(dict(
            A=A, B=B, k=k, signature=signature,
            reading=reading, component_readings=(
                a["reading"], b["reading"]), quality=quality,
            filling=(a["key"], b["key"])))
        if len(survivors) >= 3:
            break

    if not survivors:
        return None
    winner = max(survivors, key=lambda z: z["quality"])
    assert lifted_k(G, winner["A"], winner["B"]) == winner["k"]
    return winner


from functools import lru_cache


@lru_cache(maxsize=1)
def _d108_portfolio(G):
    """Immutable deterministic portfolio; call order cannot change outputs.

    Only unequal intrinsic signatures are retained. The evaluator's
    archive remains authoritative for novelty against earlier programs.
    """
    if G.q != 108:
        raise ValueError("new bed requires Dic27 of order 108")
    entries, excluded, fillings = [], set(), set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for idx in range(N_SLOTS):
        entry = None
        for restart in range(3):
            local = random.Random(
                BASE_SEED*10007 + 0xD108020
                + 65537*idx + 1000003*restart)
            winner = _d108_design(local, G, idx, excluded)
            if winner is None or winner["filling"] in fillings:
                continue
            excluded.add(winner["signature"])
            fillings.add(winner["filling"])
            entry = (
                freeze(winner["A"]), freeze(winner["B"]),
                winner["k"], winner["component_readings"],
                winner["reading"], winner["signature"])
            break
        entries.append(entry)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an independently optimized ratio card."""
    entry = _d108_portfolio(G)[idx]
    if entry is None:
        raise ValueError("Dic27 invariant-distinct target-rate card exhausted")
    A, B, k, components, reading, signature = entry
    return {
        "A": A, "B": B, "screen_k": k,
        "ratio_card": idx % 20,
        "component_witness_readings": components,
        "logical_witness_reading": reading,
        "complete_kernel4_excluded": True,
        "intrinsic_signature": signature.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell from the verified Dic27 portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton and symmetric-layer catalogue supplies twenty inequivalent designs: sixty measured-table slots collapsed to five distinct codes, all with d30s=4.
# REFUTED: Independent regular layers and joint-walk fingerprints through length twelve supply five novel measured classes: the current sixty-slot evaluation retained only two classes, both with d30s=4, scoring 3.1508.
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
