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
CONCEPT = ("Custom cyclic C11, order 11, A=B=(2,7): exhaustive relative seven-matching pairings admitted by intrinsic full-lift invariants should supply five archive-new codes with certified distance four and k=287 in cell 0.500, scoring 7.8765.")
GROUP = ("Jump from C8 to the genuinely different order-eleven cyclic group, implemented explicitly through CUSTOM_GROUP. The C11 (2,7)x(2,7) bed measured distance-four leaders at k=287 or 280. This construction replaces its displayed internal-binomial geometry with seven distinct cross-matchings; it does not reuse archived cells as candidate seeds.")
SHAPES = ("Equal (2,7) constituents have 77 variables and 22 checks. The shape allows q<=11, and q=11 gives n=583, kb=275 and guaranteed rate 25/53=0.471698. Verified complete ranks require k=287, rate 0.492281 in cell 0.500. Parent weight-three minima occupied sector-1 Acol/Bcol blocks broadly. Removing its singleton pivots targets that mechanism. Five novel certified-distance-four codes score 287*16/583=7.876501, slightly above the current target-cell incumbent.")
FILLING = ("Enumerate every seven-element matching set modulo translations, then unordered relative pairings modulo common C11 automorphisms. Every constituent row totals seven and every column totals two, giving BOTH CSS check maxima nine, also below twelve. Complete verified lifts exclude all commuting supports through weight three. A four-cycle extracted from the verified B action supplies an explicitly checked non-stabilizer weight-four witness. Unequal full-Tanner intersection and intrinsic degree-fourteen projection signatures screen structural diversity; the displayed C11 frontier filling is an exclusion reference only. An immutable indexed catalogue never pads exhaustion with orbit copies. Catalogue yield and novelty against the unshown archive remain hypotheses.")
FAMILY_NAME = "r050_c11_27x27_crossmatching_exact4_relative_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C11 requires order eleven")
    els = [(a,) for a in range(11)]

    def red(g):
        return (int(g[0]) % 11,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 11,)

    def inv(g):
        return ((-int(g[0])) % 11,)

    def idx(g):
        return int(g[0]) % 11

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _retired_z11_singleton_context(rng, G, idx):
    """Inactive parent catalogue; never used to fill the new shapes.

    A is a triangle-free incidence component. B has one singleton
    column and six distinct cross-layer matchings. Thus B has no
    kernel of weight <=3 despite its degree-one column: a short path
    between singleton endpoints needs at least two matching edges.
    No completed code is varied solely by a common group automorphism.
    """
    from itertools import combinations

    q = G.q
    e, els = G.e, G.elements

    def affine_key(labels):
        # Remove component translations and scalings before making
        # cards. Relative A/B voltage orientations are explored later.
        return min(_translation_key(tuple((u * s) % q for s in labels), q)
                   for u in range(1, q))

    def matching_cards(count):
        return sorted({affine_key(labels)
                       for labels in combinations(range(q), count)})

    acards = matching_cards(7)
    bcards = matching_cards(6)

    def matching_A(labels):
        return [[[e] for _ in labels],
                [[els[s]] for s in labels]]

    def cycle_A(steps, labels):
        # Each internal cycle column contributes two terms to one
        # row and none to the other. Matching columns contribute one
        # to each row. All seven columns retain total weight two.
        p = len(steps)
        upper = [[e, els[d]] for d in steps]
        lower = [[] for _ in steps]
        upper += [[e] for _ in labels]
        lower += [[els[s]] for s in labels]
        upper += [[] for _ in steps]
        lower += [[e, els[d]] for d in steps]
        assert len(upper) == len(lower) == 7
        assert sum(map(len, upper)) == sum(map(len, lower)) == 7
        return [upper, lower]

    afillings = [matching_A(labels) for labels in acards]

    # One eleven-cycle on each layer; five cross-matchings form an
    # independent set in the step-one cycle.
    afillings.append(cycle_A((1,), (0, 2, 4, 6, 8)))

    # Two cycles per layer. Steps 1 and 3 have no internal triangles;
    # these matching triples have no differences +/-1 or +/-3.
    # The arithmetic-progression and non-progression triples are
    # genuinely different voltage cards, not random relabelings.
    afillings.append(cycle_A((1, 3), (0, 2, 4)))
    afillings.append(cycle_A((1, 3), (0, 2, 6)))

    def singleton_B(labels, scale):
        shifts = sorted((scale * s) % q for s in labels)
        # Column six is the singleton. Row totals are seven and six,
        # column totals are two for columns 0..5 and one for column 6.
        return [[[e] for _ in range(7)],
                [[els[s]] for s in shifts] + [[]]]

    # Diagonal ordering spreads the first twenty entries over both
    # component catalogues and all three A support profiles. Extra
    # slots explore relative voltage orientations, subject to the
    # same full-graph invariant rejection.
    plans = [(a, (turn + 3 * a) % len(bcards))
             for turn in range(len(bcards))
             for a in range(len(afillings))]

    graph_keys = set()
    accepted = 0
    for scale in range(1, q):
        for ai, bi in plans:
            A = afillings[ai]
            B = singleton_B(bcards[bi], scale)
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb):
                continue
            if _grid_weights(sa, sb) > min(9, W_MAX):
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != 280 or not (475 * n <= 1000 * k < 525 * n):
                continue

            signature = _tanner_signature(hx, hz)
            if signature in graph_keys:
                continue
            graph_keys.add(signature)

            if accepted == idx:
                # Screen the actual evaluator matrices, not a
                # separately implemented incidence or voltage lift.
                if not (_no_short_kernel(hx, n)
                        and _no_short_kernel(hz, n)):
                    raise ValueError("asymmetric card has a kernel of weight <=3")
                if not css_ok(hx, hz):
                    raise ValueError("verified asymmetric lift failed CSS")
                return {"A": A, "B": B}
            accepted += 1

    raise ValueError("non-isomorphic asymmetric catalogue exhausted")


def _retired_c8_two_pivot_context(rng, G, idx):
    """Inactive historical catalogue; never publishes the C11 designs.

    No random orbit copies or mutable cross-slot cache. Equal invariant
    keys are conservatively discarded; different keys certify that the
    corresponding graphs cannot be isomorphic.
    """
    from itertools import combinations

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    if q != 8:
        raise ValueError("two-pivot catalogue requires order eight")
    elements = {G._key(g): g for g in G.elements}

    def x(t):
        return elements[(t % q, 0)]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def clear_two(rows, width):
        cols = columns(rows, width)
        return 0 not in cols and len(set(cols)) == width

    def walk_key(adj):
        """Exact permutation invariant of an undirected multigraph.

        Repeated neighbours retain edge multiplicities. Joint walk
        profiles through length eight are structural invariants, not
        distance estimates or complete isomorphism tests.
        """
        size = len(adj)
        power = [[int(u == v) for v in range(size)]
                 for u in range(size)]
        features = [[[] for _ in range(size)] for _ in range(size)]
        diagonal = [[] for _ in range(size)]
        for length in range(1, 9):
            power = [
                [sum(power[w][v] for w in adj[u])
                 for v in range(size)]
                for u in range(size)
            ]
            for u in range(size):
                diagonal[u].append(power[u][u])
                for v in range(size):
                    features[u][v].append(power[u][v])
        return tuple(sorted(
            (len(adj[u]), tuple(diagonal[u]),
             tuple(sorted(
                 (len(adj[v]), tuple(diagonal[v]),
                  tuple(features[u][v]))
                 for v in range(size))))
            for u in range(size)))

    def build(desc):
        upper, lower, labels = desc
        grid = [
            [[e, x(d)] for d in upper]
            + [[e] for _ in labels]
            + [[] for _ in lower] + [[e], []],
            [[] for _ in upper]
            + [[x(t)] for t in labels]
            + [[e, x(d)] for d in lower] + [[], [e]],
        ]
        assert all(len(row) == 8 for row in grid)
        assert all(sum(map(len, row)) == 7 for row in grid)
        assert [len(grid[0][j]) + len(grid[1][j])
                for j in range(8)] == [2] * 6 + [1, 1]
        return grid

    def component_key(grid):
        # Extract the exact A action using the scaffold's unit-side lift.
        hx, _, _ = lifted_matrices(G, grid, [[[e]]])
        width = 8 * q
        mask = (1 << width) - 1
        rows = [row & mask for row in hx]
        cols = columns(rows, width)
        if 0 in cols or len(set(cols)) != width:
            return None
        if gf2_rank(rows) != 2 * q:
            return None
        adj = [[] for _ in rows]
        for col in cols:
            if col.bit_count() == 1:
                continue
            if col.bit_count() != 2:
                return None
            first = col & -col
            u = first.bit_length() - 1
            v = (col ^ first).bit_length() - 1
            adj[u].append(v)
            adj[v].append(u)
        assert all(len(neighbours) == 6 for neighbours in adj)
        return walk_key(adj)

    # Noninvolutory inverse classes 1,2,3 give simple internal cycles.
    # Step four is excluded: its binomial produces duplicate columns.
    representatives = {}
    for p in range(4):
        steps = list(combinations(range(1, 4), p))
        cross = 6 - 2 * p
        label_sets = sorted({
            _translation_key(values, q)
            for values in combinations(range(q), cross)
        })
        for ai, upper in enumerate(steps):
            for lower in steps[ai:]:
                for labels in label_sets:
                    desc = (upper, lower, labels)
                    grid = build(desc)
                    key = component_key(grid)
                    if key is not None and key not in representatives:
                        representatives[key] = (desc, grid)

    # Interleave internal-cycle support profiles rather than exhausting
    # the matching-only family in the initial measured portfolio.
    by_profile = [
        sorted((item for item in representatives.values()
                if len(item[0][0]) == p), key=lambda item: item[0])
        for p in range(4)
    ]
    bank = []
    for turn in range(max(map(len, by_profile))):
        for p in (2, 1, 0, 3):
            if turn < len(by_profile[p]):
                bank.append(by_profile[p][turn])
    if not bank:
        raise ValueError("no simple two-pivot constituents")

    def full_key(hx, hz, width):
        """Intrinsic invariants of the complete, uncoloured Tanner graph.

        Degree fourteen identifies sector-two qubits without knowing
        their protograph labels. Checks meeting two such qubits become
        multigraph edges; singleton checks are omitted from this
        projection. Both this projection and the complete check
        intersection invariant survive arbitrary Tanner relabelings.
        """
        rows = hx + hz
        cols = columns(rows, width)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() == 14]
        if len(high) != 4 * q:
            raise ValueError("intrinsic high-degree sector changed")
        position = {j: v for v, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        adj = [[] for _ in high]
        markers = [0] * len(high)
        for row in rows:
            support = row & mask
            first = support & -support
            if support.bit_count() == 1:
                markers[position[first.bit_length() - 1]] += 1
            elif support.bit_count() == 2:
                u = position[first.bit_length() - 1]
                v = position[(support ^ first).bit_length() - 1]
                adj[u].append(v)
                adj[v].append(u)
            else:
                raise ValueError("unexpected projected check incidence")
        assert all(count == 2 for count in markers)
        return (_tanner_signature(hx, hz), walk_key(adj))

    def three_witness(B):
        # Derive endpoint syndromes from the VERIFIED B action.
        _, hz, _ = lifted_matrices(G, [[[e]]], B)
        width = 8 * q
        mask = (1 << width) - 1
        cols = columns([row & mask for row in hz], width)
        single = {col: 1 << j for j, col in enumerate(cols)
                  if col.bit_count() == 1}
        for j, col in enumerate(cols):
            if col.bit_count() == 2:
                first = col & -col
                if first in single and (col ^ first) in single:
                    return (1 << j) | single[first] | single[col ^ first]
        raise ValueError("two-pivot constituent lost its weight-three word")

    graph_keys = set()
    accepted = 0
    # Cover different intrinsic constituent pairings before exploring
    # relative orientations. These are structural cards, not draws.
    for separation in range(len(bank)):
        for ai in range(len(bank) - separation):
            bi = ai + separation
            A, B = bank[ai][1], bank[bi][1]
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            hx, hz, n = lifted_matrices(G, A, B)
            if n != 544 or max(row.bit_count() for row in hx + hz) > cap:
                continue
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if k != 288 or 40 * k <= 21 * n:
                continue
            signature = full_key(hx, hz, n)
            if signature in graph_keys:
                continue
            if not (clear_two(hx, n) and clear_two(hz, n)):
                continue
            graph_keys.add(signature)

            if accepted == idx:
                witness = three_witness(B)  # Acol0, entirely sector one.
                if witness.bit_count() != 3:
                    raise ValueError("invalid weight-three witness")
                if any((row & witness).bit_count() & 1 for row in hz):
                    raise ValueError("verified witness does not commute")
                if gf2_rank(hx + [witness]) != rx + 1:
                    raise ValueError("weight-three word is a stabilizer")
                if not css_ok(hx, hz):
                    raise ValueError("verified two-pivot lift failed CSS")
                return {
                    "A": A, "B": B, "k": k,
                    "card": (bank[ai][0], bank[bi][0]),
                    "distance_certificate": 3,
                    "logical_witness": witness,
                }
            accepted += 1

    raise ValueError("invariant-distinct two-pivot catalogue exhausted")


def _c11_columns(rows, width):
    """Column syndromes of the supplied scaffold-built matrix."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _c11_basis(rows):
    basis = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            old = basis.get(pivot)
            if old is None:
                basis[pivot] = word
                break
            word ^= old
    return basis


def _c11_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        old = basis.get(pivot)
        if old is None:
            return True
        word ^= old
    return False


def _c11_four_word(G, B):
    """Find a four-cycle using ONLY the verified constituent action.

    A pair of check vertices with two common neighbours gives four
    distinct incidence columns. Their sum is an actual kernel word.
    """
    _, full_rows, _ = lifted_matrices(G, [[[G.e]]], B)
    width = 7 * G.q
    mask = (1 << width) - 1
    rows = [row & mask for row in full_rows[:2 * G.q]]
    columns = _c11_columns(rows, width)
    adjacency = [dict() for _ in rows]

    for j, syndrome in enumerate(columns):
        if syndrome.bit_count() != 2:
            raise ValueError("matching constituent lost degree two")
        bit = syndrome & -syndrome
        u = bit.bit_length() - 1
        v = (syndrome ^ bit).bit_length() - 1
        if v in adjacency[u]:
            raise ValueError("parallel constituent edges")
        adjacency[u][v] = j
        adjacency[v][u] = j

    for u in range(len(rows)):
        for v in range(u):
            common = sorted(set(adjacency[u]) & set(adjacency[v]))
            if len(common) < 2:
                continue
            a, b = common[:2]
            positions = (
                adjacency[u][a], adjacency[v][a],
                adjacency[u][b], adjacency[v][b],
            )
            word = sum(1 << j for j in positions)
            if word.bit_count() != 4:
                continue
            if all((row & word).bit_count() % 2 == 0 for row in rows):
                return word
    raise ValueError("dense matching constituent unexpectedly lacks a four-cycle")


def _c11_graph_key(hx, hz, width):
    """Intrinsic invariant of the full UNCOLOURED Tanner graph.

    Qubit degree identifies the high-degree class without sector labels.
    Its weighted check projection, rooted refinement and joint walks
    survive arbitrary qubit/check relabelings. Equality is inconclusive
    and is discarded conservatively.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _c11_columns(rows, width)
    high = [j for j, syndrome in enumerate(columns)
            if syndrome.bit_count() == 14]
    if len(high) != 44:
        raise ValueError("unexpected intrinsic high-degree class")
    position = {j: v for v, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]

    for row in rows:
        endpoints = row & mask
        if endpoints.bit_count() != 2:
            raise ValueError("unexpected high-degree check projection")
        bit = endpoints & -endpoints
        u = position[bit.bit_length() - 1]
        v = position[(endpoints ^ bit).bit_length() - 1]
        adjacency[u][v] += 1
        adjacency[v][u] += 1

    size = len(high)
    labels = [
        (sum(neighbours.values()),
         tuple(sorted(Counter(neighbours.values()).items())))
        for neighbours in adjacency
    ]
    rooted = []
    for root in range(size):
        initial = [(labels[v], int(v == root)) for v in range(size)]
        palette = {feature: i for i, feature in
                   enumerate(sorted(set(initial)))}
        colors = [palette[feature] for feature in initial]
        history = []

        for _ in range(6):
            features = [
                (colors[u], tuple(sorted(
                    (multiplicity, colors[v])
                    for v, multiplicity in adjacency[u].items())))
                for u in range(size)
            ]
            history.append(tuple(sorted(Counter(features).items())))
            palette = {feature: i for i, feature in
                       enumerate(sorted(set(features)))}
            colors = [palette[feature] for feature in features]

        vector = [0] * size
        vector[root] = 1
        walks = [[] for _ in high]
        for _ in range(8):
            following = [0] * size
            for u, count in enumerate(vector):
                if count:
                    for v, multiplicity in adjacency[u].items():
                        following[v] += count * multiplicity
            vector = following
            for v, count in enumerate(vector):
                walks[v].append(count)

        rooted.append((
            tuple(history),
            tuple(walks[root]),
            tuple(sorted(
                (labels[v], colors[v], tuple(walks[v]))
                for v in range(size))),
        ))

    intrinsic = (
        width,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in columns).items())),
        tuple(sorted(rooted)),
        _tanner_signature(hx, hz),
    )
    return sha256(repr(intrinsic).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _c11_matching_portfolio(G, count):
    """Deterministic immutable portfolio of screened relative designs.

    Enumeration-order shuffling is not per-slot sampling. Each published
    code acquires a different intrinsic full-graph invariant.
    """
    from itertools import combinations, combinations_with_replacement

    if G.q != 11:
        raise ValueError("C11 matching portfolio requires order eleven")
    support = [[1] * 7, [1] * 7]
    cap = min(9, W_MAX)
    if not _no_dead_lines(support, support):
        raise ValueError("matching support has a dead line")
    if _grid_weights(support, support) > cap:
        raise ValueError("both matching budgets require cap nine")

    def normalize(values):
        return _translation_key(tuple(values), 11)

    def build(card):
        return [
            [[G.e] for _ in card],
            [[G.elements[s]] for s in card],
        ]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    # Translation gauges are removed, but independently scaled
    # constituents are retained: their relative orientation can change
    # the complete LP graph.
    cards = sorted({
        normalize(values)
        for values in combinations(range(11), 7)
    })
    grids = [build(card) for card in cards]
    images = [
        tuple(normalize((u * s) % 11 for s in card)
              for u in range(1, 11))
        for card in cards
    ]

    def orbit_key(a, b):
        return min(tuple(sorted((x, y)))
                   for x, y in zip(images[a], images[b]))

    # Displayed frontier filling: exclusion reference ONLY.
    # This also excludes any isomorphic image that our intrinsic key
    # recognizes; the unshown archive remains the evaluator's authority.
    displayed = (
        ((0, 1), (0, 3), (0,), (0,), (0,), (), ()),
        ((), (), (0,), (2,), (4,), (0, 1), (0, 3)),
    )
    reference = [
        [[G.elements[s] for s in cell] for cell in row]
        for row in displayed
    ]
    rhx, rhz, rn = lifted_matrices(G, reference, reference)
    blocked = {_c11_graph_key(rhx, rhz, rn)}

    # Exhaust every unordered relative prescription. Shuffle its order
    # once with a local fixed seed, then apply joint novelty gates.
    plans = list(combinations_with_replacement(range(len(cards)), 2))
    local = random.Random(BASE_SEED ^ 0xC112704)
    local.shuffle(plans)
    orbits = set()
    entries = []

    for ai, bi in plans:
        orbit = orbit_key(ai, bi)
        if orbit in orbits:
            continue
        orbits.add(orbit)
        A, B = grids[ai], grids[bi]
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 583:
            raise ValueError("unexpected C11 matching size")
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 287 or not (475 * n <= 1000 * k < 525 * n):
            continue

        # Exact complete-kernel exclusions, including mixed-sector
        # supports. These are stronger than logical-only exclusions.
        if not (_no_short_kernel(hx, n)
                and _no_short_kernel(hz, n)):
            continue

        # The verified B four-cycle embeds in Acol0. Test its actual
        # commutation and stabilizer-span membership before publication.
        witness = _c11_four_word(G, B)
        if witness.bit_count() != 4:
            continue
        if any((row & witness).bit_count() % 2 for row in hz):
            continue
        if not _c11_outside(witness, _c11_basis(hx)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified matching lift failed CSS")

        signature = _c11_graph_key(hx, hz, n)
        if signature in blocked:
            continue
        blocked.add(signature)
        entries.append((
            freeze(A), freeze(B), cards[ai], cards[bi], witness,
        ))
        if len(entries) >= count:
            break

    # Never manufacture extra slots by relabeling an accepted code.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed selection from an immutable verified portfolio."""
    entries = _c11_matching_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("invariant-distinct C11 matching catalogue exhausted")
    A, B, a, b, witness = entries[idx]
    return {
        "A": A, "B": B, "k": 287,
        "exact_distance": 4,
        "logical_witness": witness,
        "matching_cards": (a, b),
        "catalogue_position": idx,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without resampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct good designs: 60 evaluated slots retained only four distinct codes, with d30s readings three or four.
# REFUTED: One bare singleton plus affine matching-set variation supplies a broad high-fitness portfolio: sixty parent slots collapsed to one measured k=280, d30s=4 code, scoring 1.5368.
# REFUTED: The C8 two-pivot walk catalogue supplies twenty inequivalent designs: the current sixty-slot evaluation retained only six distinct codes, with all five measured leaders at k=288 and d30s=3.
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
