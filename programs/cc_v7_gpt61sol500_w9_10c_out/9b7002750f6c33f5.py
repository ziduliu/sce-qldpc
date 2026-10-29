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
CONCEPT = ("Metacyclic Z5:Z2(s=4), order 10, A=B=(3,7): fifteen-cycle degree-two cores reinforced by four cubic columns, followed by independently refined constituent selection, should supply five novel distance-eight-or-better readings in cell 0.300; this is a measurement hypothesis.")
GROUP = ("Preserve the metacyclic semidirect-product construction rule. New size factor 58 excludes the parent's order 11 because n=638; m=5,r=2,s=4 gives order ten and n=580, the largest lift allowed. The recorded Z5:Z2(s=4) (3,7)x(3,7) bed reached fitness 13.5. Improve constituent geometry on that demonstrated bed rather than claiming that the group establishes distance.")
SHAPES = ("Equal (3,7) constituents have 70 variables and 30 checks. n=580, kb=160 and guaranteed rate 16/58=0.275862 target cell 0.300. Complete verified ranks enforce [0.275,0.325). Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly; replace its degree-two/grounding columns with four cubic columns and a three-row matching ring.")
FILLING = ("Three degree-two columns form a matching ring whose girth fifteen is verified from scaffold constituent rows. Four cubic columns contribute four terms to every row: constituent row totals six, column maxima three, and BOTH CSS maxima nine. Twenty reinforcement/coset cards receive independent constituent optimization, exact kernel exclusion through six, and equal-budget witness refinement. Joint selection uses actual ranks, CSS, complete kernel-four exclusion, explicit nontrivial logical witnesses and stabilizer-pair descent. Immutable outputs have different intrinsic full uncoloured Tanner invariants; larger distance and novelty against the unseen archive remain evaluator-tested.")
FAMILY_NAME = "r030_m5r2s4_37x37_core15_cubic_joint_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    # Same metacyclic construction rule; factor 58 requires q <= 10.
    # x^5=y^2=1 and y*x*y^-1=x^4; 4^2=1 modulo five.
    if not valid_q(q):
        raise ValueError("this metacyclic group has order ten")
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


def _retired_order11_context(rng, G, idx):
    """Inactive historical order-eleven catalogue; never publishes cells.

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


def _core15_columns(rows, width):
    """Column syndromes of supplied verified matrix rows."""
    cols = [0] * width
    mask = (1 << width) - 1
    for r, original in enumerate(rows):
        row = original & mask
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _core15_rows(G, grid, side):
    """Expose exactly the scaffold's left A or right B constituent."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:len(grid) * G.q]]


def _core15_basis(rows):
    span = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                span[p] = word
                break
            word ^= old
    return span


def _core15_outside(word, span):
    while word:
        p = word.bit_length() - 1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _core15_clean_four(rows, width):
    """Exactly exclude every nonzero kernel support of weights 1..4."""
    cols = _core15_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _core15_clean_six(rows, width):
    """Exact constituent kernel exclusion through six.

    Distinct supports of size at most three must have distinct
    syndromes. Their XOR detects every nonzero support of size <=6.
    """
    cols = _core15_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in seen:
                return False
            seen.add(syndrome)
    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
    return True


def _core15_girth(rows, q):
    """Girth of the last three column blocks in the VERIFIED component."""
    cols = _core15_columns(rows, 7*q)[4*q:]
    adjacency = [[] for _ in rows]
    for edge, syndrome in enumerate(cols):
        if syndrome.bit_count() != 2:
            raise ValueError("matching-core column is not an edge")
        first = syndrome & -syndrome
        u = first.bit_length() - 1
        v = (syndrome ^ first).bit_length() - 1
        adjacency[u].append((v, edge))
        adjacency[v].append((u, edge))

    girth = len(rows) + 1
    for root in range(len(rows)):
        distances = [-1] * len(rows)
        parent_edge = [-1] * len(rows)
        distances[root] = 0
        queue = [root]
        for u in queue:
            for v, edge in adjacency[u]:
                if distances[v] < 0:
                    distances[v] = distances[u] + 1
                    parent_edge[v] = edge
                    queue.append(v)
                elif edge != parent_edge[u] and edge != parent_edge[v]:
                    girth = min(
                        girth, distances[u] + distances[v] + 1)
    return girth


def _core15_probe(local, rows, width, passes, seed=0):
    """Return an actual constituent kernel word, not an optimistic cap."""
    cols = _core15_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
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
                kernel.append(word)
                inspect(word)

        # Elementary basis exchanges preserve the entire kernel.
        local.shuffle(kernel)
        for i in range(len(kernel)):
            current = kernel[i]
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = current ^ other
                if 0 < trial.bit_count() < current.bit_count():
                    current = trial
            kernel[i] = current
            inspect(current)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])

        for _ in range(3):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

    if not best:
        raise ValueError("constituent probe returned no kernel witness")
    if any((row & best).bit_count() & 1 for row in rows):
        raise ValueError("constituent witness failed verified commutation")
    return best


def _core15_graph_key(hx, hz, width):
    """Intrinsic invariant of the complete UNCOLOURED Tanner graph.

    Unequal keys certify non-isomorphism. Equal keys are conservatively
    rejected; they do not establish isomorphism.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _core15_columns(rows, width)
    row_profiles = []
    for row in rows:
        degrees = Counter()
        bits = row
        while bits:
            bit = bits & -bits
            degrees[cols[bit.bit_length() - 1].bit_count()] += 1
            bits ^= bit
        row_profiles.append(
            (row.bit_count(), tuple(sorted(degrees.items()))))
    intrinsic = (
        width, len(rows),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted(row_profiles)),
        _tanner_signature(list(hx), list(hz)),
    )
    return sha256(repr(intrinsic).encode("ascii")).digest()


def _core15_logical_reading(G, a, b, hx, hz):
    """Embed actual constituent witnesses and reduce by stabilizers.

    Every retained support is checked for both commutation and
    nontriviality. Its weight is an UPPER bound, not a distance proof.
    """
    q, width = G.q, 7*G.q
    best = None
    for kind, source, checks, stabilizers in (
            ("X", b["right_word"], hz, hx),
            ("Z", a["left_word"], hx, hz)):
        span = _core15_basis(stabilizers)
        found = False
        for block in range(7):
            if kind == "X":
                word = source << (block * width)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 7 + block) * q + h)
                    bits ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("verified constituent embedding failed")
            if not _core15_outside(word, span):
                continue
            found = True

            for _ in range(3):
                starting = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                anchor = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & anchor).bit_count() >= 2),
                    key=lambda row: -(row & anchor).bit_count())[:48]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = anchor ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break

            if not word or not _core15_outside(word, span):
                raise ValueError("stabilizer descent lost the logical class")
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("stabilizer descent lost commutation")
            w = word.bit_count()
            best = w if best is None else min(best, w)
        if not found:
            return None
    return best


@lru_cache(maxsize=1)
def _core15_portfolio(G, count):
    """Immutable deterministic optimization; memoization changes only cost."""
    from collections import Counter
    from hashlib import sha256

    if G.q != 10 or A_SHAPE != (3, 7) or B_SHAPE != (3, 7):
        raise ValueError("core15 portfolio requires the new order-ten bed")
    q, width, cap = G.q, 7*G.q, min(9, W_MAX)
    if cap < 9:
        raise ValueError("core15 cubic supports require row cap nine")

    local = random.Random(BASE_SEED ^ 0x103715C)
    elements = {G._key(g): g for g in G.elements}
    e, x = G.e, elements[(1, 0)]

    # Each column below sums to three; each row sums to four.
    # Together with the fixed matching ring, row totals become six.
    reinforcement_columns = (
        ((1,1,1), (1,1,1), (1,1,1), (1,1,1)),
        ((1,1,1), (1,1,1), (2,1,0), (0,1,2)),
        ((1,1,1), (2,1,0), (0,2,1), (1,0,2)),
        ((2,1,0), (0,1,2), (2,1,0), (0,1,2)),
        ((3,0,0), (0,3,0), (0,0,3), (1,1,1)),
    )
    profiles = tuple(
        tuple(tuple(columns[j][i] for j in range(4))
              for i in range(3))
        for columns in reinforcement_columns)
    assert all(
        all(sum(row) == 4 for row in profile)
        and all(sum(profile[i][j] for i in range(3)) == 3
                for j in range(4))
        for profile in profiles)

    rotations = [g for g in G.elements if g[1] == 0]
    reflections = [g for g in G.elements if g[1] == 1]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def draw(profile, policy, trial):
        grid = [[[] for _ in range(7)] for _ in range(3)]
        for i in range(3):
            for j in range(4):
                terms = profile[i][j]
                if not terms:
                    continue
                if terms == 1:
                    if policy == 0:
                        pool = (reflections if
                                (i + j + trial) % 3 == 0 else rotations)
                    elif policy == 1:
                        pool = reflections if (i + j) % 2 else rotations
                    elif policy == 2:
                        pool = reflections if j % 2 else rotations
                    else:
                        pool = G.elements
                    cell = [local.choice(pool)]
                else:
                    first = G.pick(local)
                    if terms == 2:
                        if policy in (0, 2):
                            relative = elements[(local.randrange(1, 5), 0)]
                        elif policy == 1:
                            relative = local.choice(reflections)
                        else:
                            relative = local.choice(
                                [g for g in G.elements if g != e])
                        cell = [first, G.mul(first, relative)]
                    else:
                        # Change relative supports, not merely anchors.
                        if policy in (0, 1):
                            a = elements[(local.randrange(1, 5), 0)]
                            b = local.choice(reflections)
                        elif policy == 2:
                            a, b = local.sample(
                                [g for g in rotations if g != e], 2)
                        else:
                            a, b = local.sample(
                                [g for g in G.elements if g != e], 2)
                        cell = [first, G.mul(first, a), G.mul(first, b)]
                if len({G._key(g) for g in cell}) != terms:
                    raise ValueError("cubic reinforcement duplicated an element")
                grid[i][j] = cell

        # Three row-pair matchings. The cycle voltage is a rotation
        # of order five, giving two lifted fifteen-cycles.
        grid[0][4], grid[1][4] = [e], [e]
        grid[1][5], grid[2][5] = [e], [e]
        grid[2][6], grid[0][6] = [e], [x]
        counts = [[len(cell) for cell in row] for row in grid]
        assert all(sum(row) == 6 for row in counts)
        assert tuple(sum(counts[i][j] for i in range(3))
                     for j in range(7)) == (3,3,3,3,2,2,2)
        return grid

    def cycle_penalty(rows):
        # An intrinsic cheap tie-break; never a distance certificate.
        cols = _core15_columns(rows, width)
        c4 = 0
        for i, col in enumerate(cols):
            for other in cols[:i]:
                overlap = (col & other).bit_count()
                c4 += overlap * (overlap - 1) // 2
        return c4

    bank, component_keys, filling_keys = [], set(), set()
    for card in range(20):
        profile, policy = profiles[card // 4], card % 4
        raw = []
        for trial in range(384):
            C = draw(profile, policy, trial)
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in C for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            left = _core15_rows(G, C, "A")
            if gf2_rank(left) != 3*q:
                continue
            if not _core15_clean_four(left, width):
                continue
            raw.append((cycle_penalty(left), trial, C, left))
        raw.sort(key=lambda record: (record[0], record[1]))

        refined = []
        for penalty, trial, C, left in raw[:48]:
            if not _core15_clean_six(left, width):
                continue
            right = _core15_rows(G, C, "B")
            if gf2_rank(right) != 3*q:
                continue
            if not _core15_clean_six(right, width):
                continue
            if (_core15_girth(left, q) != 15
                    or _core15_girth(right, q) != 15):
                raise ValueError("verified matching core is not girth fifteen")
            wl = _core15_probe(local, left, width, 4)
            wr = _core15_probe(local, right, width, 4)
            refined.append((
                min(wl.bit_count(), wr.bit_count()),
                penalty, trial, C, left, right, wl, wr))
        refined.sort(key=lambda record:
                     (-min(record[0], 10), record[1], record[2]))

        finalists = []
        for record in refined[:8]:
            _, penalty, trial, C, left, right, wl, wr = record
            # All finalists receive the same additional budget.
            wl = _core15_probe(local, left, width, 24, wl)
            wr = _core15_probe(local, right, width, 24, wr)
            reading = min(wl.bit_count(), wr.bit_count())
            assert reading >= 7
            finalists.append((
                reading, penalty, trial, C, left, right, wl, wr))
        finalists.sort(key=lambda record:
                       (-min(record[0], 10), record[1], record[2]))

        kept = 0
        for reading, penalty, trial, C, left, right, wl, wr in finalists:
            signature = sha256(repr((
                _tanner_signature(left, []),
                _tanner_signature(right, []),
                tuple(sorted(Counter(
                    c.bit_count()
                    for c in _core15_columns(left, width)).items())),
            )).encode("ascii")).digest()
            if signature in component_keys:
                continue
            component_keys.add(signature)
            bank.append({
                "grid": freeze(C), "reading": reading,
                "left_word": wl, "right_word": wr,
                "penalty": penalty, "card": card, "trial": trial,
                "component_signature": signature,
            })
            kept += 1
            if kept >= 4:
                break

    if len(bank) < 2:
        raise ValueError("too few kernel-six-clean cubic constituents")
    bank.sort(key=lambda record: (
        -min(record["reading"], 10), record["penalty"],
        record["card"], record["trial"]))

    # Joint support cards vary the two sides, not gauges of one filling.
    profile_pairs = tuple(
        (a, b) for a in range(5) for b in range(a, 5))
    plans = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    tested, graph_keys, entries = set(), set(), []

    for position in range(count):
        card = position % 20
        wanted = profile_pairs[card % len(profile_pairs)]
        pressure = 1 + (position // 5) % 3
        ranked = list(plans)
        local.shuffle(ranked)
        ranked.sort(key=lambda pair: (
            -min(bank[pair[0]]["reading"],
                 bank[pair[1]]["reading"], 10),
            tuple(sorted((bank[pair[0]]["card"] // 4,
                          bank[pair[1]]["card"] // 4))) != wanted,
            -(pressure * min(bank[pair[0]]["reading"], 10)
              + (4-pressure) * min(bank[pair[1]]["reading"], 10)),
            bank[pair[0]]["card"] == bank[pair[1]]["card"],
            bank[pair[0]]["penalty"] + bank[pair[1]]["penalty"],
        ))

        survivors = []
        for ai, bi in ranked:
            if (ai, bi) in tested:
                continue
            tested.add((ai, bi))
            a, b = bank[ai], bank[bi]
            A, B = a["grid"], b["grid"]
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            if n != 580 or max(row.bit_count() for row in hx+hz) > cap:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 275*n <= 1000*k < 325*n):
                continue
            signature = _core15_graph_key(hx, hz, n)
            if signature in graph_keys:
                continue
            # Reserve even discarded survivors: no recycling an invariant
            # class into a later optimization card.
            graph_keys.add(signature)
            if not (_core15_clean_four(hx, n)
                    and _core15_clean_four(hz, n)):
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified core15 cubic lift failed CSS")

            reading = _core15_logical_reading(G, a, b, hx, hz)
            if reading is None or reading < 7:
                continue
            capped = min(reading, 10)
            quality = (
                k * capped * capped / n, capped, k,
                min(a["reading"], b["reading"]),
                -a["penalty"] - b["penalty"])
            survivors.append((
                quality, A, B, k, reading,
                (card, a["card"], b["card"],
                 a["trial"], b["trial"]),
                signature))
            if len(survivors) >= 3:
                break

        if not survivors:
            # Preserve any already constructed distinct entries; never
            # pad an exhausted catalogue with relabelings.
            break
        winner = max(survivors, key=lambda record: record[0])
        _, A, B, k, reading, prescription, signature = winner
        entries.append((A, B, k, reading, prescription, signature))

    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an immutable optimized complete filling."""
    entries = _core15_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("distinct target-rate core15 cubic portfolio exhausted")
    A, B, k, reading, prescription, signature = entries[idx]
    return {
        "A": A, "B": B, "k": k,
        "logical_witness_upper_bound": reading,
        "constituent_kernel_exclusion_through": 6,
        "complete_kernel_exclusion_through": 4,
        "design_card": prescription,
        "intrinsic_signature": signature.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell from the verified new-shape portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: Opposite-cycle singleton and regular pairing escapes the distance-four plateau: all five current measured finalists had d30s=4, with their minimum logical families entirely in sector 1.
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
