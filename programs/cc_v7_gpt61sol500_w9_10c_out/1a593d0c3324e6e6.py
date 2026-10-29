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
CONCEPT = ("Metacyclic C5 semidirect_-1 C2, order 10, A=B=(3,7): independently optimized mixed degree-two/three constituents should supply five novel measured readings >=8 in cell 0.300; at k=160 these would score 17.6552, not certify distance.")
GROUP = ("Keep the metacyclic construction rule, changing only m,r,s to 5,2,4. New size factor 58 makes the parent's order 11 invalid; order 10 is the largest admissible lift and gives n=580. This dihedral group's (3,7)x(3,7) bed reached fitness 13.5. Improve that measured bed without assuming a group-distance advantage.")
SHAPES = ("Change both sides to (3,7), with equal 70-variable, 30-check constituents. Guaranteed kb=160 and rate 16/58=0.27586 lie in cell 0.300; complete verified ranks enforce [0.275,0.325). Aim away from the named 0.500 cell because the parent's degree-two catalogue retained only five classes, all measured at four. Its minima broadly occupied sector-1 Acol/Bcol blocks; three rows permit cubic columns within the strict weight-nine budget.")
FILLING = ("Use four cubic columns and three degree-two columns, each constituent row totaling six, so BOTH CSS maxima are nine. Twenty quotient-holonomy prescriptions receive equal constituent search and witness-refinement budgets. All actions come from verified lifts. Exact kernel-four exclusions, actual ranks and CSS gate final pairs; embedded and randomized mixed-sector logical witnesses guide selection. An immutable portfolio conservatively rejects repeated intrinsic uncoloured Tanner signatures. Readings >=8 and novelty against the unseen archive remain measurement hypotheses.")
FAMILY_NAME = "r030_m5r2s4_37x37_mixedcolumn_witness_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    # Same metacyclic construction rule; factor 58 requires q <= 10.
    # C5 semidirect C2: s=4 has order two modulo five.
    if not valid_q(q):
        raise ValueError("this metacyclic parameter case has order ten")
    m, r, s = 5, 2, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 7)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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


def _retired_singleton_ctx(rng, G, idx):
    """Uncalled historical Z11 catalogue, not a policy for the new shapes.

    Every accepted code, not merely its first twenty entries, passes
    the complete verified-matrix kernel screen through weight three.
    Different fingerprints certify non-isomorphism; equal fingerprints
    are conservatively discarded without claiming an isomorphism.
    """
    from itertools import combinations

    q, e, els = G.q, G.e, G.elements
    cap = min(9, W_MAX)

    def label_key(labels):
        return min(tuple(sorted((sign * s + t) % q for s in labels))
                   for sign in (1, -1) for t in range(q))

    def regular(upper, lower, labels):
        """Seven degree-two columns; row totals seven."""
        p = len(upper)
        assert len(lower) == p and len(labels) + 2 * p == 7
        return (
            [[e, els[d]] for d in upper]
            + [[e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[els[s]] for s in labels]
            + [[e, els[d]] for d in lower],
        )

    def card_cells(card):
        _, steps, labels = card
        return [list(row) for row in regular(steps, steps, labels)]

    # The singleton layer has no internal edges. On the opposite
    # layer, step one is an eleven-cycle. Matching offsets form an
    # independent set in that cycle, excluding component triangles.
    singleton_cycle = [
        [[e] for _ in range(5)] + [[], [e]],
        [[els[s]] for s in (0, 2, 4, 6, 8)]
        + [[e, els[1]], []],
    ]
    assert [sum(map(len, row)) for row in singleton_cycle] == [6, 7]

    # Canonical regular constituent cards. Common voltage scaling is
    # NOT removed here: the singleton-cycle fixes a relative step.
    by_profile = []
    for p in range(3):
        cards = set()
        for steps in combinations(range(1, (q + 1) // 2), p):
            signed = {d % q for d in steps}
            signed.update((-d) % q for d in steps)
            # Constructive triangle exclusion for internal edges.
            if any((a + b) % q in signed
                   for a in signed for b in signed):
                continue
            for labels in combinations(range(q), 7 - 2 * p):
                if any((a - b) % q in signed
                       for a, b in combinations(labels, 2)):
                    continue
                cards.add((p, steps, label_key(labels)))
        by_profile.append(sorted(cards))

    # Interleave support profiles instead of spending an initial
    # prefix entirely on one regular matching model.
    cards = []
    for turn in range(max(map(len, by_profile))):
        for profile in (2, 1, 0):
            if turn < len(by_profile[profile]):
                cards.append(by_profile[profile][turn])

    def columns(rows, n):
        result = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def fingerprint(hx, hz, n):
        """Intrinsic, uncoloured Tanner invariants.

        In addition to full check intersections, project each high
        qubit-degree class onto checks. Degree classes and the resulting
        adjacency/common-neighbour profiles survive arbitrary Tanner
        relabelings, including interchange of the two CSS types.
        """
        rows = hx + hz
        cols = columns(rows, n)
        degree_counts = {}
        for col in cols:
            d = col.bit_count()
            degree_counts[d] = degree_counts.get(d, 0) + 1

        projected = []
        for degree in sorted(d for d in degree_counts if d >= 10):
            adj = [0] * len(rows)
            for col in cols:
                if col.bit_count() != degree:
                    continue
                bits = col
                while bits:
                    bit = bits & -bits
                    adj[bit.bit_length() - 1] |= col ^ bit
                    bits ^= bit
            profiles = []
            for i, neighbors in enumerate(adj):
                histogram = {}
                for j, other in enumerate(adj):
                    if i == j:
                        continue
                    key = (rows[j].bit_count(), other.bit_count(),
                           (neighbors >> j) & 1,
                           (neighbors & other).bit_count())
                    histogram[key] = histogram.get(key, 0) + 1
                profiles.append(
                    (rows[i].bit_count(), neighbors.bit_count(),
                     tuple(sorted(histogram.items()))))
            projected.append((degree, tuple(sorted(profiles))))
        return (_tanner_signature(hx, hz),
                tuple(sorted(degree_counts.items())),
                tuple(projected))

    # Explicitly exclude the shown archived Z11 codes. These matrices
    # are only fingerprint references, never submitted as candidates.
    def old_cycle(d0, d1, labels):
        return [list(row) for row in regular((d0,), (d1,), labels)]

    def old_matching(labels):
        return [list(row) for row in regular((), (), labels)]

    archived = [
        (old_matching((0, 1, 2, 4, 5, 6, 10)),
         old_matching((0, 2, 3, 5, 6, 7, 9))),
        (old_cycle(10, 10, (0, 2, 5, 7, 9)),
         old_cycle(10, 10, (0, 2, 4, 7, 9))),
        (old_cycle(6, 7, (0, 1, 4, 6, 8)),
         old_cycle(4, 4, (0, 3, 6, 8, 9))),
        (old_cycle(1, 7, (3, 4, 5, 6, 8)),
         old_cycle(4, 6, (0, 1, 3, 6, 10))),
        (old_matching(tuple(range(7))),
         [list(row) for row in
          regular((1, 3), (1, 3), (0, 2, 4))]),
        (old_matching(tuple(range(7))),
         [[[e] for _ in range(7)],
          [[els[s]] for s in range(6)] + [[]]]),
    ]
    archive_keys = set()
    for A, B in archived:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(fingerprint(hx, hz, n))

    def scaled_card(card, u):
        p, steps, labels = card
        ds = tuple(sorted(min((u * d) % q, (-u * d) % q)
                          for d in steps))
        return p, ds, label_key(tuple((u * s) % q for s in labels))

    def regular_pair_key(a, b):
        # Common automorphisms, column gauges, row swaps and A/B swap.
        return min(tuple(sorted((scaled_card(a, u),
                                 scaled_card(b, u))))
                   for u in range(1, q))

    # Primary asymmetric cards, then cycle-rich regular pairs, then
    # the remaining canonical pairs if extra slots are requested.
    cycle_cards = by_profile[2] + by_profile[1]
    plans = [(a, None) for a in cards]
    for turn in range(len(cycle_cards)):
        for ai, a in enumerate(cards):
            plans.append((a, cycle_cards[(turn + ai) % len(cycle_cards)]))
    plans.extend((a, b) for ai, a in enumerate(cards)
                 for b in cards[ai:])

    graph_keys, orbit_keys = set(), set()
    accepted = 0
    for a, b in plans:
        orbit = (("opposite-cycle-singleton", a) if b is None
                 else ("regular", regular_pair_key(a, b)))
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A = card_cells(a)
        B = singleton_cycle if b is None else card_cells(b)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k not in (280, 287) or not (475 * n <= 1000 * k < 525 * n):
            continue

        signature = fingerprint(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        # This stronger-than-logical screen excludes all commuting
        # supports of weights one, two and three in both CSS types.
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified singleton-cycle lift failed CSS")
        graph_keys.add(signature)

        if accepted == idx:
            return {"A": A, "B": B, "k": k,
                    "card": (a, b), "kernel_floor": 4}
        accepted += 1

    raise ValueError("novel kernel-three-screened catalogue exhausted")


def _shape_columns(rows, width):
    """Syndromes in the coordinates of a supplied verified lift."""
    columns = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    return columns


def _shape_kernel4(rows, width):
    """Exact exclusion of all nonzero kernel words of weights 1..4.

    Every such word splits into two supports of weight at most two.
    A collision between their syndromes detects it.
    """
    columns = _shape_columns(rows, width)
    seen = {0}
    for syndrome in columns:
        if syndrome in seen:
            return None
        seen.add(syndrome)
    squares = 0
    for i, first in enumerate(columns):
        for second in columns[:i]:
            syndrome = first ^ second
            if syndrome in seen:
                return None
            seen.add(syndrome)
            overlap = (first & second).bit_count()
            squares += overlap * (overlap - 1) // 2
    return columns, squares


def _shape_probe(local, columns, passes, seed=0):
    """Heuristic kernel upper reading carrying an actual witness."""
    width = len(columns)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
        pivots, basis = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                basis.append(word)

        # Elementary exchanges preserve the whole kernel span.
        local.shuffle(basis)
        for i in range(len(basis)):
            current = basis[i]
            w = current.bit_count()
            for j, other in enumerate(basis):
                if i == j:
                    continue
                trial = current ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    current, w = trial, tw
            basis[i] = current
            inspect(current)

        for i, first in enumerate(basis):
            for j in range(i):
                pair = first ^ basis[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ basis[h])

        for _ in range(2):
            starting = best
            for i, first in enumerate(basis):
                inspect(starting ^ first)
                for second in basis[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break
        if weight <= 6:
            break

    if not best:
        raise ValueError("mixed-column constituent has no kernel witness")
    syndrome = 0
    bits = best
    while bits:
        bit = bits & -bits
        syndrome ^= columns[bit.bit_length() - 1]
        bits ^= bit
    assert syndrome == 0
    return weight, best


def _shape_component_bank(G, side):
    """Twenty deliberate quotient cards, independently optimized per side."""
    from itertools import combinations_with_replacement

    q, width = G.q, 7 * G.q
    if q != 10:
        raise ValueError("mixed-column bank requires order ten")
    support = (
        (1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, 0, 1),
        (1, 1, 1, 1, 0, 1, 1),
    )
    assert all(sum(row) == 6 for row in support)
    assert [sum(row[j] for row in support) for j in range(7)] == \
        [3, 3, 3, 3, 2, 2, 2]
    assert _grid_weights(support, support) == 9 <= min(9, W_MAX)

    # After fixing the first cubic column and each column's first
    # incident label, these prescribe quotient holonomies, not gauges
    # of one completed constituent. There are exactly twenty cards.
    models = tuple(combinations_with_replacement(
        ((0, 0), (0, 1), (1, 0), (1, 1)), 3))
    assert len(models) == 20
    elements = {G._key(g): g for g in G.elements}

    def element(a, b):
        return elements[(a % 5, b % 2)]

    def draw(local, model, card):
        grid = [[[] for _ in range(7)] for _ in range(3)]
        for i in range(3):
            grid[i][0] = [G.e]
        for j, (b, c) in enumerate(model, 1):
            grid[0][j] = [G.e]
            grid[1][j] = [element(local.randrange(5), b)]
            grid[2][j] = [element(local.randrange(5), c)]

        # Triangle of degree-two column orbits. At least one reflection
        # is used; the card also varies its quotient cycle holonomy.
        quotients = (
            1, (card // 2 + int(side == "B")) % 2, card % 2)
        for j, (u, v), b in zip(
                (4, 5, 6), ((0, 1), (0, 2), (1, 2)), quotients):
            grid[u][j] = [G.e]
            grid[v][j] = [element(local.randrange(5), b)]
        return grid

    def component(grid):
        # The auxiliary zero cell is never published. These are exactly
        # the scaffold's two constituent actions, not a separate lift.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    bank, fillings = [], set()
    for card, model in enumerate(models):
        local = random.Random(
            BASE_SEED ^ (0x371008 + 104729 * card
                         + 1000003 * int(side == "B")))
        raw = []
        for trial in range(128):
            grid = draw(local, model, card)
            filling = tuple(
                tuple(G.idx(g) for g in cell)
                for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = component(grid)
            rank = gf2_rank(rows)
            if rank < 3 * q - 2:
                continue
            stats = _shape_kernel4(rows, width)
            if stats is None:
                continue
            columns, squares = stats

            # Check-projection triangles, less the triangles internal
            # to individual cubic columns, penalize short Tanner cycles.
            adjacency = [0] * len(rows)
            internal = 0
            for column in columns:
                degree = column.bit_count()
                internal += degree * (degree - 1) * (degree - 2) // 6
                bits = column
                while bits:
                    bit = bits & -bits
                    adjacency[bit.bit_length() - 1] |= column ^ bit
                    bits ^= bit
            triangles = 0
            for u, neighbors in enumerate(adjacency):
                bits = neighbors >> (u + 1)
                while bits:
                    bit = bits & -bits
                    v = bit.bit_length() + u
                    triangles += (
                        (neighbors & adjacency[v]) >> (v + 1)).bit_count()
                    bits ^= bit
            cost = (12 + 4 * (card % 5)) * squares \
                + max(0, triangles - internal)
            raw.append(dict(
                grid=grid, rows=rows, columns=columns,
                rank=rank, cost=cost, trial=trial, card=card))
        raw.sort(key=lambda r: (r["cost"], r["trial"]))

        shortlist = []
        for record in raw[:8]:
            reading, witness = _shape_probe(
                local, record["columns"], 4)
            record.update(distance=reading, witness=witness)
            shortlist.append(record)
        shortlist.sort(key=lambda r: (
            -r["distance"], r["cost"], r["trial"]))

        # Every retained constituent receives the same refinement budget.
        for record in shortlist[:2]:
            reading, witness = _shape_probe(
                local, record["columns"], 24, record["witness"])
            assert all((row & witness).bit_count() % 2 == 0
                       for row in record["rows"])
            record.update(distance=reading, witness=witness)
            bank.append(record)
    bank.sort(key=lambda r: (
        -r["distance"], r["cost"], r["card"], r["trial"]))
    return bank


def _shape_logical_reading(G, a, b, hx, hz, local):
    """Actual nontrivial witnesses, including mixed-sector proposals.

    This is a heuristic upper reading, not an exhaustive distance search.
    """
    q, width = G.q, 58 * G.q
    overall = None
    for kind, checks, stabilizers in (("X", hz, hx), ("Z", hx, hz)):
        span = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                previous = span.get(p)
                if previous is None:
                    span[p] = row
                    break
                row ^= previous

        def outside(word):
            while word:
                p = word.bit_length() - 1
                previous = span.get(p)
                if previous is None:
                    return True
                word ^= previous
            return False

        seeds = {}

        def inspect(word):
            if not word:
                return
            w = word.bit_count()
            if len(seeds) >= 4 and w >= max(seeds.values()):
                return
            if not outside(word):
                return
            seeds[word] = w
            if len(seeds) > 4:
                del seeds[max(seeds, key=seeds.get)]

        # Both witnesses already use their VERIFIED constituent action.
        for block in range(7):
            if kind == "X":
                word = b["witness"] << (block * 7 * q)
            else:
                word, bits = 0, a["witness"]
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 7 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            inspect(word)

        columns = _shape_columns(checks, width)
        for pass_number in range(2):
            order = list(range(width))
            local.shuffle(order)
            if pass_number:
                # Include a deliberately sector-two-biased information set.
                order.sort(key=lambda j: j < 49 * q)
            pivots, basis = {}, []
            for j in order:
                syndrome, word = columns[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    inspect(word)

            short = sorted(basis, key=int.bit_count)[:48]
            for i, first in enumerate(short):
                for second in short[:i]:
                    inspect(first ^ second)
            tiny = short[:16]
            for i, first in enumerate(tiny):
                for j in range(i):
                    pair = first ^ tiny[j]
                    for h in range(j):
                        inspect(pair ^ tiny[h])

        if not seeds:
            return None
        best = width + 1
        for starting in list(seeds):
            word, weight = starting, starting.bit_count()
            for _ in range(3):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    if tw < weight:
                        word, weight = trial, tw
                anchor = word
                moves = [row for row in stabilizers
                         if (row & anchor).bit_count() >= 2]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = anchor ^ first ^ second
                        tw = trial.bit_count()
                        if tw < weight:
                            word, weight = trial, tw
                if word == previous:
                    break
            assert word and outside(word)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            best = min(best, weight)
        overall = best if overall is None else min(overall, best)
    return overall


def _shape_graph_key(hx, hz, width):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive."""
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    columns = _shape_columns(rows, width)
    degrees = [column.bit_count() for column in columns]
    profiles = []
    for row in rows:
        neighborhood = Counter()
        bits = row
        while bits:
            bit = bits & -bits
            neighborhood[degrees[bit.bit_length() - 1]] += 1
            bits ^= bit
        profiles.append((row.bit_count(), tuple(sorted(neighborhood.items()))))
    signature = (
        width, len(rows),
        tuple(sorted(Counter(degrees).items())),
        tuple(sorted(Counter(profiles).items())),
        _tanner_signature(hx, hz))
    return sha256(repr(signature).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _shape_portfolio(G, count):
    """Immutable deterministic portfolio; memoization changes only cost."""
    aa = _shape_component_bank(G, "A")
    bb = _shape_component_bank(G, "B")
    if not aa or not bb:
        raise ValueError("mixed-column constituent bank exhausted")

    plans = sorted(
        (-min(a["distance"], b["distance"]),
         -a["distance"] - b["distance"],
         a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(aa)
        for bi, b in enumerate(bb))
    results, graph_keys, filling_keys = [], set(), set()
    cap = min(9, W_MAX)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for preliminary, total, cost, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        filling = tuple(
            tuple(G.idx(g) for g in cell)
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (k > 0 and 275*n <= 1000*k < 325*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = _shape_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        # Complete screens include mixed-sector supports, not just
        # constituent words. No hand-written matrix action is used.
        if (_shape_kernel4(hx, n) is None
                or _shape_kernel4(hz, n) is None):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified mixed-column lift failed CSS")

        local = random.Random(
            BASE_SEED ^ (0x3710D0 + 65537 * ai + 1009 * bi))
        reading = _shape_logical_reading(G, a, b, hx, hz, local)
        if reading is None:
            continue
        quality = (
            k * reading * reading / n, reading,
            -cost, -preliminary, -total, signature)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["distance"], b["distance"]), signature.hex()))
        if len(results) >= max(count, 5) + 16:
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed selection of a complete independently optimized design."""
    catalogue = _shape_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("invariant-distinct mixed-column portfolio exhausted")
    quality, A, B, k, reading, cards, components, signature = catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "design_cards": cards, "component_readings": components,
        "quantum_witness_reading": reading,
        "kernel4_excluded": True, "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without resampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus cycle-rich regular Z11 catalogue supplies twenty inequivalent designs or escapes distance four: sixty evaluated slots retained five classes, all five measured at four.
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
