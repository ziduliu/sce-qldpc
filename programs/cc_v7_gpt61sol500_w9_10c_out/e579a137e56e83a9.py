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
CONCEPT = ("Cyclic metacyclic Z8:Z1, order 8, A=B=(3,7): saturated degree-two/three constituents screened through weight seven should supply five novel measured distance readings >=8 in rate cell 0.300; this is a falsifiable measurement hypothesis.")
GROUP = ("Keep the metacyclic construction rule, cyclic specialization and order eight unchanged. The new factor 58 gives n=464, so no order repair is needed. Related Z8 (3,8) portfolios measured six, but this parent's five finalists measured five. These comparisons motivate a shape change without establishing a distance advantage.")
SHAPES = ("Equal (3,7) sides have 56-variable, 24-check constituents. The shape permits q<=10; the retained group fixes q=8, n=464 and kb=128, with guaranteed rate 16/58=0.275862. Verified full ranks enforce [0.275,0.325). Parent minimum logicals were sector-1 Acol/Bcol witnesses, prominently involving Acol2/3, not sector-2 Arow/Brow blocks.")
FILLING = ("Three degree-two and four degree-three columns saturate six terms per constituent row, giving BOTH CSS maxima nine. Search inequivalent count profiles and relative voltages. Verified constituents exclude kernels through five, count weight-six words, and exactly test weight seven when six is absent; inversion equivalence is checked against the verified right lift. Full lifts enforce actual rate, CSS and complete kernel-five exclusion. Explicit nontrivial logical witnesses receive equal search and stabilizer-descent budgets. An immutable portfolio conservatively rejects repeated full uncoloured Tanner invariants, replacing frequently empty ownership bins. Constituent floors do not certify the same quantum floor; larger witness readings remain upper bounds.")
FAMILY_NAME = "r030_m8r1s1_37x37_saturated_kernel7_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same metacyclic construction rule; factor 73 requires q <= 8.
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


def _retired_z11_ctx(rng, G, idx):
    """Uncalled historical catalogue; never used for the new shapes.

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


def _retired_distributed_z8_ctx(rng, G, idx):
    """Historical eight-column optimizer; never called by the new entry point."""
    from collections import Counter
    import hashlib

    if G.q != 8:
        raise ValueError("distributed-core catalogue requires order eight")
    q, width = G.q, 8 * G.q
    cap = min(9, W_MAX)
    # Profiles are column-major here. All row totals are exactly six.
    profiles = (
        ((1,0,0), (1,1,0), (0,1,1), (0,1,1), (1,0,1),
         (2,1,0), (0,2,1), (1,0,2)),
        ((1,0,0), (1,1,0), (0,1,1), (0,1,1), (1,0,1),
         (1,1,1), (2,0,1), (0,2,1)),
        ((1,0,0), (1,1,0), (1,1,0), (1,0,1), (1,0,1),
         (1,2,0), (0,0,3), (0,2,1)),
        ((1,0,0), (1,1,0), (1,1,0), (1,0,1), (0,1,1),
         (2,1,0), (0,2,1), (0,0,3)),
        # Measured distance-six support mechanism, used only as fallback.
        ((1,0,0), (1,1,0), (0,1,1), (0,1,1), (0,1,1),
         (2,0,1), (2,0,1), (0,2,1)),
    )
    card = idx % 20
    pressure = 1 + card // 4
    pa, pb = card % 4, (card % 4 + card // 4) % 4

    def support(profile):
        return [[column[i] for column in profile] for i in range(3)]

    for profile in profiles:
        counts = support(profile)
        assert all(sum(row) == 6 for row in counts)
        assert all(1 <= sum(column) <= 3 for column in profile)
        assert _no_dead_lines(counts, counts)
        assert _grid_weights(counts, counts) <= cap

    def columns(rows, size):
        out = [0] * size
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def short_screen(cols, through_five):
        """Exact exclusion through four/five; never a heuristic reading."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        pairs = set()
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs.add(syndrome)
        if through_five:
            for i, first in enumerate(cols):
                for j in range(i):
                    pair = first ^ cols[j]
                    for h in range(j):
                        if (pair ^ cols[h]) in pairs:
                            return None
        return pairs

    def six_count(cols):
        """Exact AFTER exclusion through five; ten partitions per word."""
        counts, collisions = {}, 0
        for i, first in enumerate(cols):
            for j in range(i):
                pair = first ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    old = counts.get(syndrome, 0)
                    collisions += old
                    counts[syndrome] = old + 1
        assert collisions % 10 == 0
        return collisions // 10

    def draw(profile):
        grid = [[[] for _ in range(8)] for _ in range(3)]
        for j, counts in enumerate(profile):
            for i, count in enumerate(counts):
                if count:
                    labels = rng.sample(range(q), count)
                    grid[i][j] = [G.elements[t] for t in labels]
            anchor = next(grid[i][j][0] for i in range(3)
                          if grid[i][j])
            inverse = G.inv(anchor)
            for i in range(3):
                grid[i][j] = [G.mul(inverse, g) for g in grid[i][j]]
        return grid

    def component_rows(grid, side):
        # Obtain the actual left/right actions from the verified helper.
        dummy = [[[G.e]]]
        mask = (1 << width) - 1
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, dummy)
        else:
            _, rows, _ = lifted_matrices(G, dummy, grid)
        return [row & mask for row in rows]

    def witness_probe(cols):
        """Equal-budget explicit kernel witnesses: upper readings only."""
        best = len(cols) + 1
        order = list(range(len(cols)))
        for _ in range(16):
            rng.shuffle(order)
            pivots, basis = {}, []
            for j in order:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
            for i in range(len(basis)):
                current = basis[i]
                for j in range(len(basis)):
                    trial = current ^ basis[j]
                    if trial and trial.bit_count() < current.bit_count():
                        current = trial
                basis[i] = current
                best = min(best, current.bit_count())
            for i, first in enumerate(basis):
                for j in range(i):
                    pair = first ^ basis[j]
                    best = min(best, pair.bit_count())
                    for h in range(j):
                        best = min(best, (pair ^ basis[h]).bit_count())
        return best

    def pool(profile_number, side):
        records, seen = [], set()
        for trial in range(256):
            # Broaden only after spending most proposals on the new core.
            profile = profiles[profile_number if trial < 224 else 4]
            grid = draw(profile)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in seen:
                continue
            seen.add(filling)
            rows = component_rows(grid, side)
            cols = columns(rows, width)
            pairs = short_screen(cols, False)
            if pairs is None:
                continue
            strong = short_screen(cols, True) is not None
            six = six_count(cols) if strong else 10**9
            floor = 7 if strong and six == 0 else 6 if strong else 5
            records.append((floor, six, trial, grid, cols))
        records.sort(key=lambda z: (-z[0], z[1], z[2]))
        finalists = []
        for floor, six, trial, grid, cols in records[:16]:
            reading = witness_probe(cols)
            finalists.append((floor, six, reading, trial, grid))
        finalists.sort(key=lambda z: (-z[0], -z[2], z[1], z[3]))
        return finalists

    def signature(hx, hz, n):
        """Intrinsic uncoloured Tanner invariant, not a labeling key."""
        rows = hx + hz
        cols = columns(rows, n)
        incidence = [[] for _ in rows]
        overlap = [Counter() for _ in rows]
        for c in cols:
            vertices = []
            while c:
                bit = c & -c
                vertices.append(bit.bit_length() - 1)
                c ^= bit
            degree = len(vertices)
            for u in vertices:
                incidence[u].append(degree)
                for v in vertices:
                    if u != v:
                        overlap[u][v] += 1
        adjacency = [sum(1 << v for v in neighbors)
                     for neighbors in overlap]
        labels = [(rows[u].bit_count(), tuple(sorted(incidence[u])))
                  for u in range(len(rows))]
        profiles_out = []
        for u, neighbors in enumerate(overlap):
            features = tuple(sorted(
                (labels[v], multiplicity,
                 (adjacency[u] & adjacency[v]).bit_count())
                for v, multiplicity in neighbors.items()))
            profiles_out.append((labels[u], features))
        qubit_profiles = []
        for c in cols:
            neighbors = []
            while c:
                bit = c & -c
                neighbors.append(labels[bit.bit_length() - 1])
                c ^= bit
            qubit_profiles.append(tuple(sorted(neighbors)))
        return (tuple(sorted(profiles_out)),
                tuple(sorted(Counter(qubit_profiles).items())))

    def mixed_five(cols, pairs, n1):
        """Exact mixed-sector test using one anchor per translation orbit.

        Any mixed weight-five word can be translated so one sector-2
        qubit has group coordinate zero. The constituent screens already
        exclude all pure-sector-1 words through five. Exclusion through
        four makes a triple/pair collision necessarily disjoint.
        """
        for anchor in range(n1, len(cols), q):
            first = cols[anchor]
            for j, second in enumerate(cols):
                if j == anchor:
                    continue
                pair = first ^ second
                for h in range(j):
                    if h != anchor and (pair ^ cols[h]) in pairs:
                        return True
        return False

    best, best_quality = None, None
    for batch in range(3):
        aa, bb = pool(pa, "A"), pool(pb, "B")
        plans = []
        for ai, a in enumerate(aa):
            for bi, b in enumerate(bb):
                quality = (min(a[0], b[0]), min(a[2], b[2]),
                           -(pressure*a[1] + (6-pressure)*b[1]),
                           a[0] + b[0])
                plans.append((quality, ai, bi))
        plans.sort(reverse=True)
        owned = []
        for preliminary, ai, bi in plans:
            a, b = aa[ai], bb[bi]
            A, B = a[4], b[4]
            hx, hz, n = lifted_matrices(G, A, B)
            if n != 584 or max(r.bit_count() for r in hx + hz) > cap:
                continue
            intrinsic = signature(hx, hz, n)
            digest = hashlib.blake2b(
                repr(intrinsic).encode("ascii"), digest_size=16).digest()
            if int.from_bytes(digest, "little") % N_SLOTS != idx:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 325*n <= 1000*k < 375*n):
                continue
            px = short_screen(columns(hx, n), False)
            if px is None:
                continue
            pz = short_screen(columns(hz, n), False)
            if pz is None:
                continue
            owned.append((preliminary, a, b, A, B, hx, hz, n, k))
            if len(owned) >= 3:
                break

        for preliminary, a, b, A, B, hx, hz, n, k in owned:
            full_floor = 5
            if min(a[0], b[0]) >= 6:
                legal_five = True
                for checks in (hx, hz):
                    cols = columns(checks, n)
                    pairs = short_screen(cols, False)
                    if mixed_five(cols, pairs, 64*q):
                        legal_five = False
                        break
                if legal_five:
                    full_floor = 6
            quality = (min(full_floor, preliminary[0]),
                       preliminary, k)
            if best_quality is None or quality > best_quality:
                assert css_ok(hx, hz)
                best_quality = quality
                best = {"A": A, "B": B, "k": k,
                        "card": (pa, pb, pressure, batch),
                        "component_floor": (a[0], b[0]),
                        "component_six_counts": (a[1], b[1]),
                        "component_witness_readings": (a[2], b[2]),
                        "full_kernel_floor": full_floor}
        if best is not None and best["full_kernel_floor"] >= 6:
            break

    if best is None:
        raise ValueError("no screened target-rate design in intrinsic slot bin")
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _saturated37_portfolio(G, count):
    """Deterministic computation with immutable cached publication data.

    Memoization changes build cost only. Each slot gets a different
    retained full-Tanner invariant, not a voltage orbit or ownership bin.
    """
    from itertools import combinations, permutations
    from collections import Counter

    if G.q != 8:
        raise ValueError("seven-column portfolio requires the retained Z8")
    q, width, cap = G.q, 7 * G.q, min(9, W_MAX)
    local = random.Random(BASE_SEED ^ 0x307708)
    zero = [[[]]]
    mask = (1 << width) - 1
    inverse_positions = [G.idx(G.inv(g)) for g in G.elements]

    def columns(rows, size):
        out = [0] * size
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def pairs4(cols):
        """Exactly exclude all kernel supports of weights one through four."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        pairs = set()
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs.add(syndrome)
        return pairs

    def constituent_screen(rows):
        """Exclude <=5, count six, and exactly test seven when six is absent."""
        cols = columns(rows, width)
        pairs = pairs4(cols)
        if pairs is None:
            return None
        triples, collisions = {}, 0
        for i, first in enumerate(cols):
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    syndrome = partial ^ cols[h]
                    if syndrome in pairs:
                        return None
                    previous = triples.get(syndrome, 0)
                    collisions += previous
                    triples[syndrome] = previous + 1
        # With <=5 excluded, colliding triples have disjoint supports.
        # Each six-word has ten unordered partitions into two triples.
        assert collisions % 10 == 0
        sixes = collisions // 10
        if sixes:
            return 6, sixes

        # With <=6 excluded, a quadruple/triple collision necessarily
        # has seven distinct endpoints. This tests every seven-word.
        for i, first in enumerate(cols):
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    triple = partial ^ cols[h]
                    for t in range(h):
                        if (triple ^ cols[t]) in triples:
                            return 7, 0
        return 8, 0

    def invariant(rows, size, walks=False):
        """Conservative uncoloured Tanner invariant.

        Inequality certifies nonisomorphism; equality is inconclusive
        and is discarded. Closed walks enrich small constituent keys.
        """
        cols = columns(rows, size)
        basic = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            _tanner_signature(rows, []),
        )
        if not walks:
            return basic
        adjacency = [[] for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adjacency[i].append((j, overlap))
                    adjacency[j].append((i, overlap))
        profiles = []
        for root in range(len(rows)):
            vector = [0] * len(rows)
            vector[root] = 1
            diagonal = []
            for length in range(8):
                following = [0] * len(rows)
                for i, value in enumerate(vector):
                    if value:
                        for j, multiplicity in adjacency[i]:
                            following[j] += value * multiplicity
                vector = following
                diagonal.append(vector[root])
            profiles.append((rows[root].bit_count(), tuple(diagonal)))
        return basic, tuple(sorted(profiles))

    def support_key(profile):
        return min(
            tuple(sorted(tuple(column[i] for i in order)
                         for column in profile))
            for order in permutations(range(3)))

    # Both seeds have row totals six and degrees 2,2,2,3,3,3,3.
    # Neither contains the parent's degree-one pendant column.
    profiles = [
        ((1,1,0), (0,1,1), (1,0,1),
         (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
        ((2,0,0), (0,2,0), (1,0,1),
         (1,1,1), (1,1,1), (1,1,1), (0,1,2)),
    ]
    keys = {support_key(p) for p in profiles}
    doubles = sorted(set(permutations((1,1,0)))
                     | set(permutations((2,0,0))))
    triples = sorted(set(permutations((2,1,0)))
                     | set(permutations((3,0,0)))
                     | {(1,1,1)})
    for attempt in range(32768):
        profile = tuple(
            [local.choice(doubles) for _ in range(3)]
            + [local.choice(triples) for _ in range(4)])
        if any(sum(c[i] for c in profile) != 6 for i in range(3)):
            continue
        # Prefer at least two independent augmentation checks. This
        # is a support-selection criterion, not a lifted-rank assertion.
        augmentation = [
            sum((column[i] & 1) << j
                for j, column in enumerate(profile))
            for i in range(3)]
        if gf2_rank(augmentation) < 2:
            continue
        key = support_key(profile)
        if key not in keys:
            keys.add(key)
            profiles.append(profile)
        if len(profiles) >= 20:
            break

    def draw(profile):
        grid = [[[] for _ in range(7)] for _ in range(3)]
        for j, counts in enumerate(profile):
            for i, terms in enumerate(counts):
                if not terms:
                    continue
                if terms == 2 and sum(counts) == 2:
                    # A sole internal binomial gets an eight-cycle,
                    # rather than an immediate two/four-cycle word.
                    labels = [0, local.choice((1, 3, 5, 7))]
                else:
                    labels = local.sample(range(q), terms)
                grid[i][j] = [G.elements[t] for t in labels]
            anchor = next(grid[i][j][0] for i in range(3)
                          if grid[i][j])
            inv = G.inv(anchor)
            for i in range(3):
                grid[i][j] = [G.mul(inv, g) for g in grid[i][j]]
        return grid

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = word
                    break
                word ^= previous
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                return True
            word ^= previous
        return False

    def probe(rows, size, search_rng, passes, seed=0, own=None):
        """Equal-budget information-set search with actual kernel witnesses."""
        cols = columns(rows, size)
        best = seed
        weight = seed.bit_count() if seed else size + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if not word or w >= weight:
                return
            if own is not None and not outside(word, own):
                return
            best, weight = word, w

        order = list(range(size))
        for _ in range(passes):
            search_rng.shuffle(order)
            pivots, basis = {}, []
            for j in order:
                syndrome, word = cols[j], 1 << j
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
            search_rng.shuffle(basis)
            for i in range(len(basis)):
                word = basis[i]
                for j, other in enumerate(basis):
                    if i == j:
                        continue
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
                basis[i] = word
                inspect(word)
            short = sorted(basis, key=int.bit_count)[:28]
            for i, first in enumerate(short):
                for j in range(i):
                    partial = first ^ short[j]
                    inspect(partial)
                    for h in range(j):
                        inspect(partial ^ short[h])
            if best:
                starting = best
                for i, first in enumerate(short):
                    inspect(starting ^ first)
                    for second in short[:i]:
                        inspect(starting ^ first ^ second)
        if best:
            assert all((row & best).bit_count() % 2 == 0 for row in rows)
            assert own is None or outside(best, own)
        return best

    records, fillings, component_keys = [], set(), set()
    clean_eight = 0
    for attempt in range(12288):
        pi = attempt % len(profiles)
        C = draw(profiles[pi])
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in C for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        left, _, _ = lifted_matrices(G, C, zero)
        left = [row & mask for row in left]
        if gf2_rank(left) < 22:
            continue
        screen = constituent_screen(left)
        if screen is None:
            continue
        floor, sixes = screen
        signature = invariant(left, width, True)
        if signature in component_keys:
            continue
        component_keys.add(signature)

        # Check the entire inversion permutation against the actual
        # right lift, rather than merely assuming a convention transfer.
        _, right, _ = lifted_matrices(G, zero, C)
        right = [row & mask for row in right]
        for i in range(3):
            for h in range(q):
                source = left[i*q+h]
                transferred = 0
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    transferred |= 1 << (j*q + inverse_positions[t])
                    source ^= bit
                assert transferred == right[i*q+inverse_positions[h]]

        records.append(dict(
            grid=C, rows=left, right=right, floor=floor,
            sixes=sixes, filling=filling, profile=pi, trial=attempt))
        clean_eight += int(floor == 8)
        if attempt >= 2048 and clean_eight >= 36:
            break
    if len(records) < 8:
        raise ValueError("too few distinct kernel-five-clean constituents")
    records.sort(key=lambda r: (
        -r["floor"], r["sixes"], r["profile"], r["filling"]))
    bank = records[:max(40, min(80, count + 24))]

    for record in bank:
        search_rng = random.Random(
            BASE_SEED ^ (0x370800 + 1009 * record["trial"]))
        word = probe(record["rows"], width, search_rng, 16)
        if not word:
            raise ValueError("screened constituent has no kernel witness")
        assert word.bit_count() >= record["floor"]
        right_word, source = 0, word
        while source:
            bit = source & -source
            j, h = divmod(bit.bit_length() - 1, q)
            right_word |= 1 << (j*q + inverse_positions[h])
            source ^= bit
        assert all((row & right_word).bit_count() % 2 == 0
                   for row in record["right"])
        record["word"], record["right_word"] = word, right_word

    plans = sorted(
        (-min(a["floor"], b["floor"]),
         -min(a["word"].bit_count(), b["word"].bit_count(), 8),
         a["sixes"] + b["sixes"], ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))

    def mixed_five(cols, pairs, n1):
        """Translation-complete mixed-sector weight-five exclusion.

        Pure sector-1 words split into already screened constituent
        kernels. Every remaining word can be translated to put one
        sector-2 endpoint at coordinate zero. With <=4 excluded,
        a triple/pair collision is necessarily disjoint.
        """
        for anchor in range(n1, len(cols), q):
            first = cols[anchor]
            for j, second in enumerate(cols):
                if j == anchor:
                    continue
                partial = first ^ second
                for h in range(j):
                    if h != anchor and (partial ^ cols[h]) in pairs:
                        return True
        return False

    def logical_reading(a, b, hx, hz, n, search_rng):
        """Explicit nontrivial witnesses; the result is only an upper bound."""
        readings = []
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = span(stabilizers)
            seeds = []
            for block in range(7):
                if kind == "X":
                    word = b["right_word"] << (block * width)
                else:
                    word, source = 0, a["word"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j*7+block)*q+h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if outside(word, own):
                    seeds.append(word)
            seed = min(seeds, key=int.bit_count) if seeds else 0
            word = probe(checks, n, search_rng, 3, seed, own)
            if not word:
                return None
            for descent in range(3):
                starting = word
                ranked = sorted(
                    stabilizers,
                    key=lambda row: (starting ^ row).bit_count())
                for row in ranked:
                    trial = starting ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                short = ranked[:32]
                for i, first in enumerate(short):
                    for second in short[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < word.bit_count():
                            word = trial
                if word == starting:
                    break
            assert word and outside(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            readings.append(word.bit_count())
        return min(readings)

    results, graph_keys = [], set()
    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert all(sum(row) == 6 for row in sa + sb)
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= cap
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 464 or not (275*n <= 1000*k < 325*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = invariant(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        passed = True
        for checks in (hx, hz):
            cols = columns(checks, n)
            pairs = pairs4(cols)
            if pairs is None or mixed_five(cols, pairs, 49*q):
                passed = False
                break
        if not passed:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified seven-column portfolio failed CSS")
        search_rng = random.Random(
            BASE_SEED ^ (0x3077D0 + 1009*ai + bi))
        reading = logical_reading(a, b, hx, hz, n, search_rng)
        if reading is None:
            continue
        assert reading >= 6
        assert lifted_k(G, A, B) == k

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        # Cap heuristic readings at the stated target. Weak searches
        # returning much larger witnesses cannot dominate publication.
        quality = (
            k * min(reading, 8)**2 / n,
            min(a["floor"], b["floor"]),
            -a["sixes"] - b["sixes"])
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["profile"], b["profile"]),
            (a["floor"], b["floor"]),
            (a["sixes"], b["sixes"])))
        if len(results) >= max(count, 5) + 16:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable, screened, invariant-separated design."""
    portfolio = _saturated37_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(portfolio)):
        raise ValueError("distinct seven-column target-rate portfolio exhausted")
    quality, A, B, k, reading, profiles, floors, sixes = portfolio[idx]
    return dict(
        A=A, B=B, k=k, quality=quality,
        support_cards=profiles, component_kernel_floors=floors,
        component_six_counts=sixes, full_kernel_floor=6,
        logical_witness_upper_bound=reading)


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell from immutable screened catalogue data."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus cycle-rich regular Z11 catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five classes, all with measured distance four.
# REFUTED: Distributed degree-two cores with six-multiplicity optimization and ownership bins supply five readings above six: the measured parent retained thirteen distinct codes, many bins failed, and all five finalists had d30s=5.
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
