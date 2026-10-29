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
CONCEPT = ("Frobenius C11 semidirect C5 with action 3, order 55, A=B=(1,3): independently oriented girth-five incidence constituents should supply five novel k=226 codes with verified distance five, targeting fitness 226*25/550=10.2727 in cell 0.400.")
GROUP = ("Jump from the parent's cyclic C11 to the genuinely nonabelian order-55 Frobenius group. Action 3 has order five modulo eleven. The reported order-55 (1,3)x(1,3) bed leads cell 0.400 at 10.2 and measured k=226,d30s=5 codes; this motivates exact screening and new relative orientations, not a transferred distance assumption.")
SHAPES = ("Change both sides to equal (1,3) constituents with 165 variables and 55 checks. n=550, q_max=59, kb=220 and guaranteed rate 0.400. Connected binomial constituents have verified rank 54; complete ranks must give k=226, excess six and rate 0.41091. Parent weight-four minima broadly occupied sector-1 Acol/Bcol blocks. Five novel certified distance-five codes would replace the destination incumbent's mixture of k=222 and k=226 finalists.")
FILLING = ("Enumerate inverse-class generator triples and quotient explicit constituent automorphisms, then restore independent relative group orientations when coupling. Verified constituent actions exclude every kernel support through four and supply actual five-cycle witnesses. Complete verified lifts require k=226, CSS, both row-weight sums eight, kernel exclusion through four and a non-stabilizer weight-five witness. Full uncoloured Tanner refinement uses ALL intrinsic degree-twelve roots, avoiding an unsupported nonabelian translation shortcut. Exclude every action interpretation of the shown order-55 frontier reference and conservatively discard equal graph invariants. Catalogue yield and novelty against the unseen archive remain hypotheses.")
FAMILY_NAME = "r040_frobenius11x5_action3_13x13_girth5_relative_certified_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C11 semidirect C5 has order 55")
    # x^11=y^5=1, y*x*y^-1=x^3; (a,b)=x^a*y^b.
    m, r = 11, 5
    powers = [pow(3, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _retired_c11_singleton_context(rng, G, idx):
    """Inactive historical C11 catalogue; never publishes new cells.

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


from functools import lru_cache


def _f55_columns(rows, width):
    """Column syndromes of matrices supplied by the verified helper."""
    result = [0] * width
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            result[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return result


def _f55_clear_four(rows, width):
    """Exact exclusion of every nonzero kernel support of weights 1..4."""
    columns = _f55_columns(rows, width)
    seen = {0}
    for syndrome in columns:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _f55_component_rows(G, grid, side):
    """Expose the evaluator's exact left or right constituent action."""
    dummy = [[[G.e]]]
    width = 3 * G.q
    mask = (1 << width)-1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:G.q]]


def _f55_five_cycle(rows, width):
    """Return an actual weight-five constituent kernel word."""
    columns = _f55_columns(rows, width)
    adjacency = [[] for _ in rows]
    for j, syndrome in enumerate(columns):
        if syndrome.bit_count() != 2:
            raise ValueError("binomial constituent is not an incidence matrix")
        first = syndrome & -syndrome
        u = first.bit_length()-1
        v = (syndrome ^ first).bit_length()-1
        adjacency[u].append((v, j))
        adjacency[v].append((u, j))

    def walk(vertex, visited, edges):
        if len(edges) == 4:
            for other, edge in adjacency[vertex]:
                if other == 0 and edge not in edges:
                    return sum(1 << j for j in edges + (edge,))
            return 0
        for other, edge in adjacency[vertex]:
            if other in visited:
                continue
            word = walk(other, visited | {other}, edges + (edge,))
            if word:
                return word
        return 0

    # The connected regular constituent is vertex transitive. A
    # girth-five six-regular graph on 55 vertices has a five-cycle:
    # girth >=6 would require at least 62 vertices by the Moore bound.
    word = walk(0, {0}, ())
    if word:
        assert word.bit_count() == 5
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
    return word


def _f55_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _f55_outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _f55_graph_key(hx, hz, n):
    """Intrinsic rooted refinement of the COMPLETE uncoloured Tanner graph.

    Checks are pooled without CSS colours. Every degree-twelve qubit
    is used as a root; no transitivity of a nonabelian LP is assumed.
    Unequal keys certify non-isomorphism, while equality is discarded
    conservatively. Hash collisions can only discard additional codes.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    adjacency = [[] for _ in range(n + len(rows))]
    for r, original in enumerate(rows):
        row = original
        check = n+r
        while row:
            bit = row & -row
            j = bit.bit_length()-1
            adjacency[j].append(check)
            adjacency[check].append(j)
            row ^= bit
    degrees = [len(a) for a in adjacency]
    roots = [j for j in range(n) if degrees[j] == 12]
    if len(roots) != 55:
        raise ValueError("unexpected intrinsic high-degree root class")

    rooted = []
    for root in roots:
        labels = [(int(v >= n), degrees[v], int(v == root))
                  for v in range(len(adjacency))]
        unique = sorted(set(labels))
        palette = {key: c for c, key in enumerate(unique)}
        colours = [palette[key] for key in labels]
        digest = sha256()
        digest.update(repr(tuple(sorted(Counter(labels).items())))
                      .encode("ascii"))
        previous = len(unique)
        for depth in range(8):
            keys = [
                (colours[v],
                 tuple(sorted(colours[w] for w in neighbours)))
                for v, neighbours in enumerate(adjacency)
            ]
            digest.update(repr(tuple(sorted(Counter(keys).items())))
                          .encode("ascii"))
            unique = sorted(set(keys))
            palette = {key: c for c, key in enumerate(unique)}
            colours = [palette[key] for key in keys]
            if len(unique) == previous:
                break
            previous = len(unique)
        rooted.append(digest.digest())

    return (
        tuple(sorted(Counter(degrees[:n]).items())),
        tuple(sorted(Counter(degrees[n:]).items())),
        tuple(sorted(rooted)),
    )


@lru_cache(maxsize=1)
def _f55_portfolio(G, count):
    """Immutable finite-card portfolio, independent of slot call order."""
    from itertools import combinations

    if G.q != 55 or min(W_MAX, 9) < 8:
        raise ValueError("order-55 binomial bed requires row budget eight")
    q, width = G.q, 3*G.q
    local = random.Random(BASE_SEED ^ 0xF551335)

    def inverse_class(t):
        return min(t, G.idx(G.inv(G.elements[t])))

    # All automorphisms x->x^u, y->x^v*y, u != 0 modulo eleven.
    # The geometric sums encode (x^v*y)^b exactly.
    sums = [0]
    for b in range(4):
        sums.append((sums[-1] + pow(3, b, 11)) % 11)
    automorphisms = [
        tuple(G.idx(((u*a + v*sums[b]) % 11, b))
              for a, b in G.elements)
        for u in range(1, 11) for v in range(11)
    ]
    class_maps = [
        tuple(inverse_class(t) for t in action)
        for action in automorphisms
    ]
    directions = [
        t for t, g in enumerate(G.elements)
        if g != G.e and inverse_class(t) == t
    ]
    cards = sorted({
        min(tuple(sorted(action[t] for t in triple))
            for action in class_maps)
        for triple in combinations(directions, 3)
    })

    def build(card):
        return [[[G.e, G.elements[t]] for t in card]]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    bank = []
    for card in cards:
        grid = build(card)
        left = _f55_component_rows(G, grid, "A")
        if gf2_rank(left) != q-1:
            continue
        if not _f55_clear_four(left, width):
            continue
        right = _f55_component_rows(G, grid, "B")
        if gf2_rank(right) != q-1:
            continue
        if not _f55_clear_four(right, width):
            continue
        wa = _f55_five_cycle(left, width)
        wb = _f55_five_cycle(right, width)
        if not wa or not wb:
            continue
        bank.append((card, freeze(grid), wa, wb))
    if not bank:
        raise ValueError("no verified girth-five generating constituents")

    # Exclusion references ONLY. The prompt does not specify which
    # order-five action its archived encoding used, so exclude all four
    # identifications y_old -> y^power instead of guessing one action.
    archive_keys = set()
    for power in range(1, 5):
        aa = ((1, 0), (0, power), (5, 2*power % 5))
        bb = ((1, 0), (0, 2*power % 5), (5, 2*power % 5))
        A = [[[G.e, g] for g in aa]]
        B = [[[G.e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(_f55_graph_key(hx, hz, n))

    pairs = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    local.shuffle(pairs)
    orientations = {}
    for pair in pairs:
        order = list(range(len(automorphisms)))
        local.shuffle(order)
        orientations[pair] = tuple(order)

    def joint_orbit(a, b):
        return min(
            tuple(sorted((
                tuple(sorted(action[t] for t in a)),
                tuple(sorted(action[t] for t in b)))))
            for action in class_maps
        )

    def move_word(word, action):
        moved = 0
        while word:
            bit = word & -word
            block, h = divmod(bit.bit_length()-1, q)
            moved |= 1 << (block*q + action[h])
            word ^= bit
        return moved

    def logical_witness(source, kind, hx, hz):
        checks, stabilizers = (hz, hx) if kind == "X" else (hx, hz)
        own = _f55_span(stabilizers)
        for block in range(3):
            if kind == "X":
                word = source << (block*3*q)
            else:
                word = 0
                bits = source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*3 + block)*q + h)
                    bits ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                continue
            if _f55_outside(word, own):
                return word
        return 0

    entries, orbit_keys, graph_keys = [], set(), set()
    # Interleave constituent-family pairs before spending additional
    # slots on orientations of one pairing. These are finite structural
    # prescriptions screened jointly, not twenty independent draws.
    for depth in range(len(automorphisms)):
        for ai, bi in pairs:
            a, b = bank[ai], bank[bi]
            orientation = orientations[(ai, bi)][depth]
            action = automorphisms[orientation]
            bcard = tuple(sorted(class_maps[orientation][t]
                                 for t in b[0]))
            orbit = joint_orbit(a[0], bcard)
            if orbit in orbit_keys:
                continue
            orbit_keys.add(orbit)

            A = a[1]
            # Keep the actual automorphism images, without independently
            # reversing binomials and invalidating the carried witness.
            B = [[[G.e, G.elements[action[t]]] for t in b[0]]]
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) != 8:
                continue
            hx, hz, n = lifted_matrices(G, A, B)
            if n != 550 or max(row.bit_count() for row in hx+hz) != 8:
                continue
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if k != 226 or not (375*n <= 1000*k < 425*n):
                continue

            # Complete gates include mixed-sector commuting supports.
            if not (_f55_clear_four(hx, n)
                    and _f55_clear_four(hz, n)):
                continue
            wx = logical_witness(move_word(b[3], action), "X", hx, hz)
            wz = logical_witness(a[2], "Z", hx, hz)
            if not wx or not wz:
                continue
            if wx.bit_count() != 5 or wz.bit_count() != 5:
                raise ValueError("five-cycle embedding changed weight")
            if not css_ok(hx, hz):
                raise ValueError("verified Frobenius lift failed CSS")

            signature = _f55_graph_key(hx, hz, n)
            if signature in archive_keys or signature in graph_keys:
                continue
            graph_keys.add(signature)
            entries.append((
                freeze(A), freeze(B), k, wx, wz,
                (a[0], b[0], orientation),
            ))
            if len(entries) >= count:
                return tuple(entries)

    # Never pad an exhausted portfolio with orbit copies.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of a verified, immutable complete filling."""
    entries = _f55_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("novel certified distance-five portfolio exhausted")
    A, B, k, wx, wz, card = entries[idx]
    return {
        "A": A, "B": B, "k": k, "card": card,
        "exact_distance": 5,
        "verified_X_logical": wx, "verified_Z_logical": wz,
        "complete_kernel_exclusion_through": 4,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell without resampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric regular catalogue supplies twenty inequivalent designs: the current sixty-slot evaluation retained only five classes, all five measured at distance four.
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
