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
CONCEPT = ("Cyclic Z8 in the frozen metacyclic presentation, order 8, A=(2,8), B=(2,8): rank-first logical screening of sparse and trinomial support cards may retain k>288 without logicals below weight three; an archive-filtered saturated-incidence portfolio supplies the baseline.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. Order eight is the largest lift allowed by size factor 68. This bed measured five k=288,d30s=3 finalists, and the joint-deck move still retained only six evaluator-distinct classes. Neither rank surplus nor greater distance is inferred from voltage labels.")
SHAPES = ("Keep equal 64-variable, 16-check constituents: n=544, kb=288 and guaranteed rate 36/68=0.529412 in cell 0.550; q<=8. Parent weight-three X witnesses concentrated at Bcol6/7 and Z witnesses at Acol6/7. The saturated singleton pivots force exact k=288, so the new rank-surplus branch tests different supports rather than further pivot relabelings.")
FILLING = ("Jointly screen deliberate lower-density incidence and degree-three-column cards, allowing commuting weight-one/two words only when they lie in the opposite stabilizer span. Supplement with exhaustive normalized saturated two-pivot pairings. Every published code passes verified ranks, CSS, legal term budgets and exact logical exclusion through weight two; an exact weight-three screen determines certified floor three or four. Rank-ranked immutable entries replace hash ownership, and intrinsic full-Tanner invariants exclude repeated entries and the reported current/direct-parent representatives. Strict improvement requires a surviving rank-surplus or floor-four code; baseline floor-three codes alone cannot exceed 81/17.")
FAMILY_NAME = "r055_m8r1s1_28x28_rankfirst_logical_supportfolio_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same metacyclic construction rule; factor 68 requires q <= 8.
    # Z_8 x| Z_1, with s^r = 1 mod m.
    m, r, s = 8, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 8)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)


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


def _retired_jointdeck_context(rng, G, idx):
    """Inactive archive-filtered catalogue of joint voltage prescriptions.

    Constituent graph equivalence need not preserve its specified deck
    action in a coupled lift. Retain voltage cards until full-graph
    screening; unequal intrinsic buckets separate different slots.
    """
    from itertools import combinations
    from hashlib import sha256

    if not (0 <= idx < N_SLOTS):
        raise ValueError("slot index outside catalogue")

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
    voltage_cards = {}
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
                    if key is not None:
                        # Keep the deck-labelled prescription even when
                        # another constituent has the same walk profile.
                        voltage_cards[desc] = grid
                        if key not in representatives:
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

        # Refine the COMPLETE bipartite incidence graph, rather than
        # only its check intersection or high-degree-sector projection.
        # Initial colours use degree alone: no group labels, CSS colours,
        # protograph coordinates or vertex ordering enter the invariant.
        from collections import Counter

        m = len(rows)
        incidence = [[] for _ in range(m + width)]
        for j, col in enumerate(cols):
            vertex = m + j
            while col:
                bit = col & -col
                check = bit.bit_length() - 1
                incidence[check].append(vertex)
                incidence[vertex].append(check)
                col ^= bit

        colours = [len(neighbours) for neighbours in incidence]
        refinements = []
        for _ in range(6):
            features = [
                (colours[v],
                 tuple(sorted(colours[t] for t in neighbours)))
                for v, neighbours in enumerate(incidence)
            ]
            refinements.append(tuple(sorted(Counter(features).items())))
            palette = {feature: c for c, feature in
                       enumerate(sorted(set(features)))}
            colours = [palette[feature] for feature in features]

        return (_tanner_signature(hx, hz), walk_key(adj),
                tuple(refinements))

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

    def normalize(desc, unit=1):
        """Known constituent gauges, retaining its cyclic deck action.

        Reverse internal edges independently, translate matching offsets,
        and exchange the two check layers. Do NOT quotient independent
        unit-three deck orientations before coupling A and B.
        """
        upper, lower, labels = desc

        def steps(values):
            return tuple(sorted(
                min((unit * d) % q, (-unit * d) % q)
                for d in values))

        top, bottom = steps(upper), steps(lower)
        forward = _translation_key(
            tuple((unit * t) % q for t in labels), q)
        backward = _translation_key(
            tuple((-unit * t) % q for t in labels), q)
        return min((top, bottom, forward),
                   (bottom, top, backward))

    # These are exclusion references, never candidate outputs. They
    # encode all six distinct representatives in the current table.
    c0 = ((1, 2), (1, 2), (0, 1))
    c1 = ((1,), (1,), (0, 1, 2, 3))
    c2 = ((), (), (0, 1, 2, 3, 4, 5))
    c6 = ((1, 2), (1, 3), (0, 1))
    c18 = ((1,), (2,), (0, 1, 2, 3))
    c59a = ((1,), (2,), (0, 2, 4, 6))
    c59b = ((1,), (3,), (0, 1, 2, 3))
    archived = (
        (c0, c0), (c1, c1), (c2, c2),
        (c6, c6), (c18, c18), (c59a, c59b),
    )
    blocked = set()
    for da, db in archived:
        hx, hz, n = lifted_matrices(G, build(da), build(db))
        blocked.add(full_key(hx, hz, n))

    # Enumerate voltage prescriptions, not one representative per
    # unlabelled constituent graph. All candidates retain legal simple
    # incidence columns and the same two rank pivots.
    descriptions = sorted({normalize(desc) for desc in voltage_cards})
    if not descriptions:
        raise ValueError("no legal joint voltage cards")

    # Unit five differs from three by simultaneous inversion; units
    # one/three suffice as relative orientation proposals. Full-graph
    # invariants still decide whether either proposal is genuinely new.
    plans = [
        (a, b, twist)
        for a in range(len(descriptions))
        for b in range(a, len(descriptions))
        for twist in (1, 3)
    ]
    local = random.Random(BASE_SEED ^ 0x8DECC3)
    local.shuffle(plans)
    seen_orbits = set()

    for ai, bi, twist in plans:
        da = descriptions[ai]
        db = normalize(descriptions[bi], twist)

        # Only a COMMON group automorphism is removed here. Independently
        # applying it to B is a proposal, not an inequivalence claim.
        orbit = min(
            tuple(sorted((normalize(da, unit), normalize(db, unit))))
            for unit in (1, 3, 5, 7)
        )
        if orbit in seen_orbits:
            continue
        seen_orbits.add(orbit)

        A, B = build(da), build(db)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 544 or max(row.bit_count() for row in hx + hz) > cap:
            continue

        signature = full_key(hx, hz, n)
        if signature in blocked:
            continue

        # Every isomorphic full graph has the same intrinsic signature
        # and therefore the same owner. Distinct slots cannot accept
        # isomorphic graphs under these invariants. Hash collisions only
        # put additional non-isomorphic proposals in the same bucket.
        digest = sha256(repr(signature).encode("ascii")).digest()
        owner = int.from_bytes(digest[:8], "big") % N_SLOTS
        if owner != idx:
            continue

        if not (clear_two(hx, n) and clear_two(hz, n)):
            continue
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 288 or 40 * k <= 21 * n:
            continue

        witness = three_witness(B)  # Acol0, entirely sector one.
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified joint-deck lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")

        return {
            "A": A, "B": B, "k": k,
            "card": (da, db),
            "relative_orientation_proposal": twist,
            "intrinsic_owner": owner,
            "distance_certificate": 3,
            "logical_witness": witness,
        }

    raise ValueError("archive-new joint-deck bucket exhausted")


from functools import lru_cache


@lru_cache(maxsize=1)
def _rankfirst_supportfolio(G, count):
    """Immutable portfolio; memoization does not change selected cells.

    All algebraic and logical screens use the scaffold's actual lift.
    Different intrinsic signatures certify Tanner non-isomorphism;
    equal signatures are discarded conservatively.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations, combinations_with_replacement

    q, e, cap = G.q, G.e, min(9, W_MAX)
    if q != 8:
        raise ValueError("the frozen supportfolio requires order eight")
    elements = {G._key(g): g for g in G.elements}

    def x(t):
        return elements[(t % q, 0)]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def logical_two(cols, own):
        """Exactly exclude nontrivial logicals of weights one and two.

        Equal single-column syndromes give all commuting two-words.
        Keeping one representative is sufficient: differences between
        any two accepted representatives are sums of stabilizers.
        """
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            word = 1 << j
            if syndrome in seen:
                if outside(word ^ seen[syndrome], own):
                    return False
            else:
                seen[syndrome] = word
        return True

    def logical_three(cols, own):
        """Find a weight-three logical, or exactly exclude all of them."""
        singles = {}
        for j, syndrome in enumerate(cols):
            singles.setdefault(syndrome, j)
        for i, first in enumerate(cols):
            for j in range(i):
                h = singles.get(first ^ cols[j])
                if h is None or h == i or h == j:
                    continue
                word = (1 << i) | (1 << j) | (1 << h)
                if outside(word, own):
                    return word
        return 0

    def graph_key(hx, hz, width):
        """Intrinsic full-Tanner and marked weighted-walk invariants.

        Sector-one qubits have degree <=6 in these prescriptions.
        Sector-two qubits have degree >=12. Thus the high-degree class
        is identifiable without group, protograph or CSS-type labels.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, width)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() >= 10]
        position = {j: i for i, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        adj = [Counter() for _ in high]
        marks = [Counter() for _ in high]

        for row in rows:
            support = row & mask
            endpoints = []
            while support:
                bit = support & -support
                endpoints.append(position[bit.bit_length() - 1])
                support ^= bit
            marker = (len(endpoints), row.bit_count())
            for u in endpoints:
                marks[u][marker] += 1
                for v in endpoints:
                    if u != v:
                        adj[u][v] += 1

        size = len(high)
        labels = [
            (cols[j].bit_count(), tuple(sorted(marks[u].items())))
            for u, j in enumerate(high)
        ]
        power = [[int(u == v) for v in range(size)]
                 for u in range(size)]
        walks = [[[] for _ in range(size)] for _ in range(size)]
        for _ in range(6):
            following = [[0] * size for _ in range(size)]
            for u, neighbors in enumerate(adj):
                target = following[u]
                for w, multiplicity in neighbors.items():
                    for v, value in enumerate(power[w]):
                        target[v] += multiplicity * value
            power = following
            for u in range(size):
                for v in range(size):
                    walks[u][v].append(power[u][v])

        profiles = tuple(sorted(
            (labels[u], tuple(walks[u][u]),
             tuple(sorted(
                 (labels[v], tuple(walks[v][v]), tuple(walks[u][v]))
                 for v in range(size))))
            for u in range(size)
        ))
        intrinsic = (
            tuple(sorted(Counter(row.bit_count() for row in rows).items())),
            tuple(sorted(Counter(col.bit_count() for col in cols).items())),
            _tanner_signature(hx, hz), profiles,
        )
        return sha256(repr(intrinsic).encode("ascii")).digest()

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
        return grid

    def normalize(desc, unit=1):
        upper, lower, labels = desc

        def steps(values):
            return tuple(sorted(
                min(unit * d % q, -unit * d % q) for d in values))

        top, bottom = steps(upper), steps(lower)
        forward = _translation_key(
            tuple(unit * t % q for t in labels), q)
        backward = _translation_key(
            tuple(-unit * t % q for t in labels), q)
        return min((top, bottom, forward),
                   (bottom, top, backward))

    # Exclusion references, never candidate seeds. First the six
    # evaluator-distinct current representatives, then the direct parent.
    reported = (
        (((1,), (3,), (0,1,3,5)),
         ((1,), (1,), (0,2,4,6))),
        (((1,), (3,), (0,1,2,6)),
         ((2,), (3,), (0,1,2,3))),
        (((1,), (1,), (0,1,2,5)),
         ((3,), (3,), (0,1,3,4))),
        (((1,3), (2,3), (0,1)),
         ((3,), (3,), (0,1,2,3))),
        (((1,), (2,), (0,1,2,5)),
         ((1,2), (2,3), (0,1))),
        (((1,2), (2,3), (0,2)),
         ((1,2), (2,3), (0,2))),
        (((1,2), (1,2), (0,1)),
         ((1,2), (1,2), (0,1))),
        (((1,), (1,), (0,1,2,3)),
         ((1,), (1,), (0,1,2,3))),
        (((), (), (0,1,2,3,4,5)),
         ((), (), (0,1,2,3,4,5))),
        (((1,2), (1,3), (0,1)),
         ((1,2), (1,3), (0,1))),
        (((1,), (2,), (0,1,2,3)),
         ((1,), (2,), (0,1,2,3))),
        (((1,), (2,), (0,2,4,6)),
         ((1,), (3,), (0,1,2,3))),
    )
    blocked = set()
    for da, db in reported:
        hx, hz, n = lifted_matrices(G, build(da), build(db))
        blocked.add(graph_key(hx, hz, n))

    results, filling_keys = [], set()

    def consider(A, B, prescription):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            return
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            return

        hx, hz, n = lifted_matrices(G, A, B)
        if n != 544 or max(row.bit_count() for row in hx + hz) > cap:
            return
        bx, bz = basis(hx), basis(hz)
        k = n - len(bx) - len(bz)
        if k < 288 or 1000 * k <= 525 * n:
            return

        cx, cz = columns(hx, n), columns(hz, n)
        if not logical_two(cz, bx) or not logical_two(cx, bz):
            return
        signature = graph_key(hx, hz, n)
        if signature in blocked:
            return

        wx = logical_three(cz, bx)
        wz = logical_three(cx, bz)
        floor = 3 if wx or wz else 4
        if not css_ok(hx, hz):
            raise ValueError("verified supportfolio lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")

        # A floor-four result means an exact exclusion through three,
        # not a claim that its distance is exactly four.
        blocked.add(signature)
        quality = (k * floor * floor / n, floor, k, signature)
        results.append((
            quality, freeze(A), freeze(B), k, floor,
            ("X", wx) if wx else (("Z", wz) if wz else None),
            prescription, signature.hex(),
        ))

    # Deliberate non-saturated support cards. Unlike the old kernel
    # gate, the full logical gate can retain genuine degeneracy.
    supports = (
        ((1,1,1,1,1,1,0,0), (0,0,1,1,0,0,2,2)),
        ((1,1,1,1,1,1,0,0), (0,0,0,1,1,1,2,2)),
        ((1,1,1,1,1,1,0,0), (0,0,0,0,1,1,2,2)),
        ((1,1,1,1,1,1,0,0), (0,0,0,0,1,0,2,3)),
        ((1,1,1,1,1,0,0,0), (0,0,0,0,1,1,2,3)),
    )
    sparse_bank = []
    for policy, support in enumerate(supports):
        for trial in range(16):
            local = random.Random(
                BASE_SEED ^ (0x8A3100 + 1009 * policy + trial))
            grid = [
                [local.sample(G.elements, terms) for terms in row]
                for row in support
            ]
            sparse_bank.append((policy, trial, grid))
    for ai, bi in combinations_with_replacement(range(len(sparse_bank)), 2):
        a, b = sparse_bank[ai], sparse_bank[bi]
        consider(a[2], b[2], ("rank-surplus-support", a[:2], b[:2]))

    # Exhaustive saturated prescriptions, retaining deck-labelled cards
    # until the complete lift has been screened. Step four is excluded
    # here because its binomial has repeated constituent columns.
    descriptions = set()
    for p in range(4):
        options = list(combinations((1, 2, 3), p))
        labels = sorted({
            _translation_key(values, q)
            for values in combinations(range(q), 6 - 2 * p)
        })
        for upper in options:
            for lower in options:
                for values in labels:
                    descriptions.add(normalize((upper, lower, values)))
    descriptions = sorted(descriptions)

    plans = [
        (ai, bi, twist)
        for ai in range(len(descriptions))
        for bi in range(ai, len(descriptions))
        for twist in (1, 3)
    ]
    local = random.Random(BASE_SEED ^ 0x8A31F3)
    local.shuffle(plans)
    orbits = set()
    target = max(count, 5) + 12
    for ai, bi, twist in plans:
        da = descriptions[ai]
        db = normalize(descriptions[bi], twist)
        orbit = min(
            tuple(sorted((normalize(da, u), normalize(db, u))))
            for u in (1, 3, 5, 7)
        )
        if orbit in orbits:
            continue
        orbits.add(orbit)
        consider(build(da), build(db),
                 ("saturated-relative-card", da, db))
        if len(results) >= target:
            break

    # No padding with relabeled copies, and no distance-two fallback.
    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed lookup in the jointly screened immutable portfolio."""
    if not 0 <= idx < N_SLOTS:
        raise ValueError("slot index outside supportfolio")
    entries = _rankfirst_supportfolio(G, N_SLOTS)
    if idx >= len(entries):
        raise ValueError("novel logical-screened supportfolio exhausted")
    quality, A, B, k, floor, witness, card, signature = entries[idx]
    return {
        "A": A, "B": B, "k": k, "quality": quality,
        "distance_lower_bound": floor,
        "verified_three_logical": witness,
        "support_prescription": card,
        "intrinsic_tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell from the joint logical-screened portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct good designs: 60 evaluated slots retained only four distinct codes, with d30s readings three or four.
# REFUTED: One bare singleton plus affine matching-set variation supplies a broad high-fitness portfolio: sixty parent slots collapsed to one measured k=280, d30s=4 code, scoring 1.5368.
# REFUTED: Constituent-walk representatives and the parent full-graph gates supply twenty evaluator-distinct two-pivot Z8 codes: sixty evaluated slots retained only six distinct classes; all five measured codes had k=288 and d30s=3.
# REFUTED: Joint-deck orientations and intrinsic bucket ownership increase evaluator-distinct yield beyond six on this bed: the current sixty-slot evaluation still retained six classes and fitness 4.7650.
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
