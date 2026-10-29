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
CONCEPT = ("GL2(F3), order 48, A=B=(1,3): noncommuting order-six/eight binomial constituents screened on the actual lifts should supply five novel distance-at-least-five finalists in cell 0.400.")
GROUP = ("Jump from cyclic C8 to the matrix group GL2(F3). Its order-six/eight directions avoid the immediate cycles from order-two, three and four binomials. The order-55 (1,3)x(1,3) frontier measured distance five and fitness 10.2002, motivating the kernel-four screen on this different group, not establishing its performance. Order 48 gives legal n=480.")
SHAPES = ("Equal constituents have 144 variables and 48 checks; n=480 and kb=192 give guaranteed rate 0.400. All-binomial fillings have at least three check dependencies per CSS type, hence k>=198 and rate>=0.4125; actual ranks must remain below rate 0.425. Five k=198,d=5 codes would score 10.3125. Parent weight-three minima concentrated on sector-1 Bcol6/7 and Acol6/7 singleton blocks, which are removed.")
FILLING = ("Enumerate inverse-normalized direction triples modulo inner conjugations, screen both verified constituent actions through weight four, then pair different triples and relative determinant-twist orientations. Both all-binomial row totals are six and both LP maxima eight. One-trinomial B alternatives have row totals six/seven and both maxima nine. Every emitted full lift passes exact kernel rejection through four, actual-rate screening, CSS and distinct intrinsic uncoloured Tanner invariants. Immutable memoization only avoids recomputation; slot outputs do not depend on call order. Catalogue yield, archive novelty and larger distance readings remain measured hypotheses.")
FAMILY_NAME = "r040_gl2f3_13x13_binomial_fullkernel4_catalog_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("GL2(F3) has order 48")
    # Row-major invertible 2x2 matrices over F_3.
    els = [(a, b, c, d)
           for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d - b*c) % 3]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(t) % 3 for t in g)

    def mul(g, h):
        a, b, c, d = red(g)
        e, f, u, v = red(h)
        return ((a*e + b*u) % 3, (a*f + b*v) % 3,
                (c*e + d*u) % 3, (c*f + d*v) % 3)

    def inv(g):
        a, b, c, d = red(g)
        determinant = (a*d - b*c) % 3
        if not determinant:
            raise ValueError("singular matrix")
        # Each nonzero element of F_3 is its own inverse.
        return ((determinant*d) % 3, (-determinant*b) % 3,
                (-determinant*c) % 3, (determinant*a) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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


def _retired_z8_jointdeck_context(rng, G, idx):
    """Pure archive-filtered catalogue of joint voltage prescriptions.

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


def _jp_columns(rows, width):
    """Syndromes from a supplied verified binary matrix."""
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _jp_clean4(rows, width):
    """Exactly exclude every nonzero kernel word of weights one to four."""
    cols = _jp_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _jp_component_rows(G, side, grid):
    """Extract the actual left/right action using a unit dummy side."""
    dummy = [[[G.e]]]
    mask = (1 << (3 * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz]


def _jp_graph_key(hx, hz, n):
    """Intrinsic full-Tanner invariants, compressed without label data.

    Unequal keys certify non-isomorphism. Equal keys are discarded
    conservatively; this is not a complete isomorphism algorithm.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _jp_columns(rows, n)
    m = len(rows)
    adjacency = [0] * m
    for col in cols:
        bits = col
        while bits:
            bit = bits & -bits
            adjacency[bit.bit_length() - 1] |= col ^ bit
            bits ^= bit

    def shells(root):
        visited = frontier = 1 << root
        counts = []
        while frontier:
            neighbors = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                neighbors |= adjacency[bit.bit_length() - 1]
                bits ^= bit
            frontier = neighbors & ~visited
            if frontier:
                counts.append(frontier.bit_count())
                visited |= frontier
        return tuple(counts)

    # Sector two is identified by intrinsic qubit degree, not coordinates.
    high = [j for j, col in enumerate(cols) if col.bit_count() >= 12]
    position = {j: i for i, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    projected = [Counter() for _ in high]
    colors = [Counter() for _ in high]
    for row in rows:
        bits = row & mask
        endpoints = []
        while bits:
            bit = bits & -bits
            endpoints.append(position[bit.bit_length() - 1])
            bits ^= bit
        for u in endpoints:
            colors[u][row.bit_count()] += 1
            for v in endpoints:
                if u != v:
                    projected[u][v] += 1

    size = len(high)
    power = [[int(u == v) for v in range(size)]
             for u in range(size)]
    walks = [[[] for _ in range(size)] for _ in range(size)]
    for _ in range(8):
        following = [[0] * size for _ in range(size)]
        for u, neighbors in enumerate(projected):
            target = following[u]
            for w, multiplicity in neighbors.items():
                source = power[w]
                for v, value in enumerate(source):
                    target[v] += multiplicity * value
        power = following
        for u in range(size):
            for v in range(size):
                walks[u][v].append(power[u][v])

    labels = [(cols[j].bit_count(), tuple(sorted(colors[u].items())))
              for u, j in enumerate(high)]
    walk_profiles = tuple(sorted(
        (labels[u], tuple(walks[u][u]),
         tuple(sorted((labels[v], tuple(walks[v][v]),
                       tuple(walks[u][v]))
                      for v in range(size))))
        for u in range(size)
    ))
    invariant = (
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        tuple(sorted((rows[i].bit_count(), shells(i))
                     for i in range(m))),
        _tanner_signature(hx, hz),
        walk_profiles,
    )
    return sha256(repr(invariant).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _jp_portfolio(G):
    """Deterministic immutable catalogue; memoization changes no outputs."""
    from itertools import combinations

    if G.q != 48:
        raise ValueError("GL2(F3) catalogue requires order 48")
    cap = min(9, W_MAX)
    width = 3 * G.q

    def order(g):
        power = G.e
        for length in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return length
        raise ValueError("invalid matrix-group element order")

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    # Short-order binomials are excluded constructively. Remaining
    # Cayley triples are still tested exactly; orders alone are insufficient.
    directions = sorted({
        inverse_class(g) for g in G.elements if order(g) in (6, 8)
    })
    conjugations = []
    for h in G.elements:
        hi = G.inv(h)
        conjugations.append({
            t: inverse_class(G.mul(G.mul(h, G.elements[t]), hi))
            for t in directions
        })

    cards = set()
    for triple in combinations(directions, 3):
        generators = [G.elements[t] for t in triple]
        if any(G.mul(generators[i], generators[j])
               == G.mul(generators[j], generators[i])
               for i in range(3) for j in range(i)):
            continue
        cards.add(min(tuple(sorted(action[t] for t in triple))
                      for action in conjugations))

    def grid(card):
        return [[[G.e, G.elements[t]] for t in card]]

    bank = []
    for card in sorted(cards):
        C = grid(card)
        left = _jp_component_rows(G, "A", C)
        if gf2_rank(left) != G.q - 1 or not _jp_clean4(left, width):
            continue
        right = _jp_component_rows(G, "B", C)
        if gf2_rank(right) != G.q - 1 or not _jp_clean4(right, width):
            continue
        bank.append((card, C))
    if not bank:
        raise ValueError("no kernel-four-clean GL2(F3) binomial constituent")

    def determinant_twist(C):
        # phi(g)=det(g)*g is a genuine involutory automorphism of GL2(F3).
        # Its relative use is a proposal, not an inequivalence assertion.
        out = []
        for row in C:
            new_row = []
            for cell in row:
                transformed = []
                for a, b, c, d in cell:
                    determinant = (a*d - b*c) % 3
                    transformed.append(tuple(
                        determinant*t % 3 for t in (a, b, c, d)))
                new_row.append(transformed)
            out.append(new_row)
        return out

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    signatures, fillings = set(), set()
    entries = []

    def consider(A, B, prescription):
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for C in (A, B) for row in C for cell in row)
        if filling in fillings:
            return
        fillings.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            return

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 480 or not (375*n <= 1000*k < 425*n):
            return
        if max(row.bit_count() for row in hx + hz) > cap:
            return
        # These exact tests cover both sectors and both CSS types.
        # They establish distance >=5, not a heuristic search reading.
        if not (_jp_clean4(hx, n) and _jp_clean4(hz, n)):
            return
        signature = _jp_graph_key(hx, hz, n)
        if signature in signatures:
            return
        if not css_ok(hx, hz):
            raise ValueError("verified GL2(F3) lift failed CSS")
        signatures.add(signature)
        entries.append((freeze(A), freeze(B), k, prescription))

    # Exhaustive structural pairings precede the support extension.
    # Independent inner conjugations are removed, but the relative
    # determinant twist is retained until full-graph screening.
    plans = [(a, b, twist)
             for a in range(len(bank))
             for b in range(a, len(bank))
             for twist in (0, 1)]
    plans.sort(key=lambda p: (p[0] == p[1], p[2], p[0], p[1]))
    for ai, bi, twist in plans:
        A = bank[ai][1]
        B = determinant_twist(bank[bi][1]) if twist else bank[bi][1]
        consider(A, B, ("binomial-pair", bank[ai][0],
                        bank[bi][0], twist))
        if len(entries) >= N_SLOTS:
            return tuple(entries)

    # Extend B from counts (2,2,2) to (3,2,2), in every column and
    # with every possible third element. A's six terms plus B's
    # maximum column three, and B's seven plus A's two, both give nine.
    # This is an exhaustive support/element catalogue, not slot draws.
    for bi, (card, source) in enumerate(bank):
        for j in range(3):
            for third in G.elements:
                if third in source[0][j]:
                    continue
                B = [[list(cell) for cell in source[0]]]
                B[0][j].append(third)
                right = _jp_component_rows(G, "B", B)
                if not _jp_clean4(right, width):
                    continue
                for ai, (acard, A) in enumerate(bank):
                    consider(A, B, ("one-trinomial", acard, card,
                                    j, G.idx(third)))
                    if len(entries) >= N_SLOTS:
                        return tuple(entries)

    # Never weaken the floor or pad the catalogue with equivalent codes.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed lookup; outputs are independent of call order."""
    entries = _jp_portfolio(G)
    if not (0 <= idx < len(entries)):
        raise ValueError("distinct kernel-four-clean GL2(F3) catalogue exhausted")
    A, B, k, prescription = entries[idx]
    return {"A": A, "B": B, "k": k,
            "verified_kernel_floor": 5,
            "prescription": prescription}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell from the immutable screened catalogue."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct good designs: 60 evaluated slots retained only four distinct codes, with d30s readings three or four.
# REFUTED: One bare singleton plus affine matching-set variation supplies a broad high-fitness portfolio: sixty parent slots collapsed to one measured k=280, d30s=4 code, scoring 1.5368.
# REFUTED: Constituent-walk representatives and the parent full-graph gates supply twenty evaluator-distinct two-pivot Z8 codes: sixty evaluated slots retained only six distinct classes; all five measured codes had k=288 and d30s=3.
# REFUTED: Retaining Z8 deck descriptions and relative orientations expands the measured two-pivot portfolio beyond six classes: the current sixty-slot evaluation retained six distinct codes and unchanged fitness 4.7650.
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
