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
CONCEPT = ("Cyclic C37, order 37, A=(2,3), B=(2,4): independently optimized degree-three constituents should supply five novel finalists with measured distance readings above fourteen in cell 0.150.")
GROUP = ("Jump from C11 to the genuinely nonisomorphic C37. Order 37 is the largest admissible lift for size factor sixteen, giving n=592. The reported C37 bed best is 25.2, close to the cell's 25.3 record; optimize constituent geometry rather than assuming an advantage from the group. No shown archived labeling is reused.")
SHAPES = ("A has 111 variables and 74 checks; B has 148 variables and 74 checks. With n=592 and kb=74, the guaranteed rate is 2/16=0.125. Verified complete ranks retain [0.125,0.175), targeting cell 0.150 instead of the saturated parental cell 0.500. Parent weight-four minima broadly occupied sector-1 Acol/Bcol blocks; degree-three columns replace that degree-two incidence mechanism.")
FILLING = ("Enumerate balanced column-weight-three count cards with constituent rows at most six, giving BOTH CSS maxima at most nine, also below twelve. Screen verified constituent actions for rank and complete kernel exclusion through four; exact six-word multiplicities and equal-budget actual kernel-witness searches optimize independent labels. Pair shortlisted constituents, enforce actual full rate, CSS and complete kernel-four exclusion, then rank explicit embedded and randomized mixed-sector logical readings. Unequal intrinsic full-Tanner invariants separate immutable catalogue entries without ownership bins. Larger witness readings are upper bounds; archive novelty, catalogue yield and measurement improvement remain hypotheses.")
FAMILY_NAME = "r015_cyclic37_23x24_cubic_sixcount_mixedportfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C37 requires order 37")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def _retired_c11_context(rng, G, idx):
    """Uncalled historical C11 filling; never used for the new bed.

    Candidate proposals are finite structural cards. Slot ownership is
    a function of the built graph, not its labels or proposal number.
    No mutable cross-slot state or saved parent filling is published.
    """
    from itertools import combinations
    from collections import Counter
    import hashlib

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    element = {G._key(g): g for g in G.elements}

    def x(t):
        return element[(t % q, 0)]

    def labels_key(values):
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    def signed(steps):
        return set(steps) | {(-d) % q for d in steps}

    def image(card, scale):
        kind, upper, lower, values = card
        a = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in upper))
        b = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in lower))
        v = labels_key(tuple(scale*s % q for s in values))
        return kind, a, b, v

    def canonical(card):
        images = []
        for u in range(1, q):
            kind, a, b, v = image(card, u)
            images.append((kind, a, b, v))
            if kind == 0:
                images.append((kind, b, a,
                               labels_key(tuple(-s % q for s in v))))
        return min(images)

    def regular(upper, lower, values):
        assert len(upper) == len(lower)
        assert len(values) + len(upper) + len(lower) == 7
        return [
            [[e, x(d)] for d in upper]
            + [[e] for _ in values] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(v)] for v in values]
            + [[e, x(d)] for d in lower],
        ]

    def build(card):
        kind, upper, lower, values = card
        if kind == 0:
            grid = regular(upper, lower, values)
        elif kind == 1:
            # Six cross matchings and one grounding singleton.
            # Unlike the parent's boundary, there is no internal ring.
            assert len(values) == 6
            grid = [
                [[e] for _ in values] + [[e]],
                [[x(v)] for v in values] + [[]],
            ]
        else:
            # Five matchings, opposite-layer ring, grounding singleton.
            assert len(values) == 5 and len(lower) == 1
            grid = [
                [[e] for _ in values] + [[], [e]],
                [[x(v)] for v in values] + [[e, x(lower[0])], []],
            ]
        assert all(len(row) == 7 for row in grid)
        assert all(len({G._key(g) for g in cell}) == len(cell)
                   for row in grid for cell in row)
        return grid

    def columns(rows, width):
        out = [0] * width
        mask = (1 << width) - 1
        for r, row in enumerate(rows):
            row &= mask
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def component_info(grid):
        # A zero one-cell side exposes the EXACT scaffold A action.
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
        width = 7*q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        if not _no_short_kernel(rows, width):
            return None
        cols = columns(rows, width)
        multiplicities = {}
        collisions = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                old = multiplicities.get(syndrome, 0)
                collisions += old
                multiplicities[syndrome] = old + 1
        # After exclusion through three, every collision has disjoint
        # pair supports. A weight-four word has three such partitions.
        assert collisions % 3 == 0
        return gf2_rank(rows), collisions // 3

    descriptors = set()
    for p in range(3):
        options = []
        for steps in combinations(range(1, (q + 1)//2), p):
            ss = signed(steps)
            if not any((a+b) % q in ss for a in ss for b in ss):
                options.append(steps)
        for ai, upper in enumerate(options):
            for lower in options[ai:]:
                forbidden = signed(upper) | signed(lower)
                for tail in combinations(range(1, q), 6 - 2*p):
                    values = (0,) + tail
                    if any((a-b) % q in forbidden
                           for a, b in combinations(values, 2)):
                        continue
                    descriptors.add(canonical(
                        (0, upper, lower, values)))

    # Deliberately vary the intrinsic six-matching graph, not just its
    # grounding location or a translation of one matching set.
    for tail in combinations(range(1, q), 5):
        descriptors.add(canonical((1, (), (), (0,) + tail)))
    for d in range(1, (q + 1)//2):
        for tail in combinations(range(1, q), 4):
            values = (0,) + tail
            if any((a-b) % q in (d, (-d) % q)
                   for a, b in combinations(values, 2)):
                continue
            descriptors.add(canonical((2, (), (d,), values)))

    regular_bank, boundary_bank = [], []
    for card in sorted(descriptors):
        grid = build(card)
        info = component_info(grid)
        if info is None:
            continue
        record = (card, info, grid)
        (regular_bank if card[0] == 0 else boundary_bank).append(record)
    if not regular_bank:
        raise ValueError("no kernel-three-screened regular constituents")

    def fingerprint(hx, hz, width):
        """Intrinsic invariant of the complete uncoloured Tanner graph.

        High-degree qubits identify sector 2 without protograph labels.
        A check meeting two such qubits becomes a weighted edge; a check
        meeting one becomes a marked grounding loop. Check weights label
        these edges, and qubit degrees label vertices.

        Every feature is invariant under check/qubit relabeling.
        Equal fingerprints are discarded conservatively.
        """
        rows = hx + hz
        cols = columns(rows, width)
        degree_counts = Counter(c.bit_count() for c in cols)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        nv = len(high)
        if nv != 4*q:
            raise ValueError("unexpected high-degree contraction")
        position = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        adjacency = [Counter() for _ in high]
        edge_weights = [Counter() for _ in high]
        for row in rows:
            bits = row & high_mask
            degree = bits.bit_count()
            if degree not in (1, 2):
                raise ValueError("check contraction is not an edge or anchor")
            first = bits & -bits
            u = position[first.bit_length() - 1]
            v = (u if degree == 1 else
                 position[(bits ^ first).bit_length() - 1])
            adjacency[u][v] += 1
            edge_weights[u][row.bit_count()] += 1
            if u != v:
                adjacency[v][u] += 1
                edge_weights[v][row.bit_count()] += 1

        neighbours = [tuple(sorted(a.items())) for a in adjacency]
        # Exploit translation only after checking it on the extracted
        # graph and its intrinsic vertex labels.
        vertex_labels = [
            (cols[j].bit_count(), tuple(sorted(edge_weights[v].items())))
            for v, j in enumerate(high)]
        symmetric = all(
            vertex_labels[i*q+h] == vertex_labels[i*q]
            and all(
                adjacency[i*q+h].get(j*q+t, 0)
                == adjacency[i*q].get(j*q+(t-h) % q, 0)
                for j in range(4) for t in range(q))
            for i in range(4) for h in range(q))
        roots = list(range(0, nv, q)) if symmetric else list(range(nv))
        diagonals, features = [], []
        for root in roots:
            vector = [0] * nv
            vector[root] = 1
            diagonal, pair = [], [[] for _ in high]
            for length in range(1, 11):
                following = [0] * nv
                for v, count in enumerate(vector):
                    if count:
                        for w, multiplicity in neighbours[v]:
                            following[w] += count * multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 6:
                    for v, count in enumerate(vector):
                        pair[v].append(count)
            diagonals.append(tuple(diagonal))
            features.append(pair)

        colors = [
            (vertex_labels[v],
             diagonals[v//q if symmetric else v])
            for v in range(nv)]
        profiles = []
        for ri, root in enumerate(roots):
            profile = (
                colors[root],
                tuple(sorted((colors[v], tuple(features[ri][v]))
                             for v in range(nv))))
            profiles.extend([profile] * (q if symmetric else 1))
        return (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(degree_counts.items())),
            tuple(sorted(profiles)))

    def matching(values):
        return regular((), (), values)

    def ring(a, b, values):
        return regular((a,), (b,), values)

    two = regular((1, 3), (1, 3), (0, 2, 4))
    other_two = regular((1, 3), (1, 3), (0, 2, 6))
    ring_one = ring(1, 1, (0, 2, 4, 6, 8))
    grounded = build((2, (), (1,), (0, 2, 4, 6, 8)))
    six_grounded = build((1, (), (), tuple(range(6))))
    shown_archive = [
        (matching((0, 1, 2, 4, 5, 6, 10)),
         matching((0, 2, 3, 5, 6, 7, 9))),
        (ring(10, 10, (0, 2, 5, 7, 9)),
         ring(10, 10, (0, 2, 4, 7, 9))),
        (ring(6, 7, (0, 1, 4, 6, 8)),
         ring(4, 4, (0, 3, 6, 8, 9))),
        (ring(1, 7, (3, 4, 5, 6, 8)),
         ring(4, 6, (0, 1, 3, 6, 10))),
        (matching(tuple(range(7))), two),
        (ring_one, two),
        (two, two),
        (ring_one, other_two),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
        (two, grounded),
        (matching(tuple(range(7))), grounded),
        (matching(tuple(range(7))), six_grounded),
        (two, regular((2, 5), (2, 5), (0, 3, 7))),
        (two, regular((2, 3), (2, 3), (0, 1, 5))),
    ]
    archive_keys = set()
    for A, B in shown_archive:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(fingerprint(hx, hz, n))
    # These references are never candidate outputs. The complete run's
    # archive remains authoritative for unshown previously built codes.

    plans = []
    for ai in range(len(regular_bank)):
        for bi in range(ai, len(regular_bank)):
            for scale in range(1, q):
                plans.append((0, ai, bi, scale))
        for bi in range(len(boundary_bank)):
            for scale in range(1, q):
                plans.append((1, ai, bi, scale))
    rng.shuffle(plans)

    best, best_quality = None, None
    filling_keys, graph_keys = set(), set()
    owned_regular = 0
    # Different slots have different multiplicity pressures, but their
    # ownership bins are disjoint for the ENTIRE built-graph invariant.
    pressure = 1 + idx % 5
    for boundary, ai, bi, scale in plans[:1536]:
        a = regular_bank[ai]
        b = (boundary_bank if boundary else regular_bank)[bi]
        A = a[2]
        bcard = image(b[0], scale)
        B = build(bcard)
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if n != 583 or max(r.bit_count() for r in hx + hz) > cap:
            continue
        signature = fingerprint(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        graph_keys.add(signature)
        digest = hashlib.blake2b(
            repr(signature).encode("ascii"), digest_size=16).digest()
        if int.from_bytes(digest, "little") % N_SLOTS != idx:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            continue
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified owned-bin filling failed CSS")

        qa, qb = a[1][1], b[1][1]
        # Both floors are four. Prefer rank surplus first, then fewer
        # actual constituent weight-four words, with opposing pressures.
        # Multiplicity is a tie-break, NOT a quantum-distance estimate.
        quality = (k, -max(qa, qb),
                   -(pressure*qa + (6-pressure)*qb),
                   -(qa + qb), digest)
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best = {
                "A": A, "B": B, "k": k,
                "card": (a[0], bcard),
                "kernel_floor": 4,
                "component_four_counts": (qa, qb),
                "owner": idx,
            }
        if not boundary:
            owned_regular += 1
            if owned_regular >= 12:
                break

    if best is None:
        raise ValueError("no novel target-rate filling in intrinsic owner bin")
    return best


from functools import lru_cache


def _c37_columns(rows, width):
    """Column syndromes of matrices supplied by the verified helper."""
    out = [0] * width
    mask = (1 << width) - 1
    for r, original in enumerate(rows):
        row = original & mask
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _c37_clear4(cols):
    """Exactly exclude nonzero kernel supports of weights one to four."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _c37_six_count(cols):
    """Exact six-word count for clean, odd-column-weight constituents.

    All columns have weight three, so odd-weight kernel words cannot
    occur. After exclusion through four, colliding triples are disjoint;
    each six-word has ten unordered triple partitions.
    """
    if not all(c.bit_count() == 3 for c in cols):
        raise ValueError("constituent columns are not cubic")
    triples, collisions = {}, 0
    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                old = triples.get(value, 0)
                collisions += old
                triples[value] = old + 1
    if collisions % 10:
        raise ValueError("inconsistent verified six-word count")
    return collisions // 10


def _c37_probe(cols, local, passes, seed=0):
    """Heuristic search with an explicit nonincreasing kernel witness."""
    best = seed
    weight = seed.bit_count() if seed else len(cols) + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        order = list(range(len(cols)))
        local.shuffle(order)
        pivots, nulls = {}, []
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
                nulls.append(word)

        local.shuffle(nulls)
        for i, word in enumerate(nulls):
            w = word.bit_count()
            for j, other in enumerate(nulls):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            nulls[i] = word
            inspect(word)
        short = sorted(set(nulls), key=int.bit_count)[:40]
        for i, first in enumerate(short):
            for j in range(i):
                inspect(first ^ short[j])
        tiny = short[:16]
        for i, first in enumerate(tiny):
            for j in range(i):
                pair = first ^ tiny[j]
                for h in range(j):
                    inspect(pair ^ tiny[h])
        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)

    if not best:
        raise ValueError("kernel witness search returned no word")
    syndrome = 0
    bits = best
    while bits:
        bit = bits & -bits
        syndrome ^= cols[bit.bit_length() - 1]
        bits ^= bit
    if syndrome:
        raise ValueError("kernel witness failed verified syndrome")
    return weight, best


def _c37_span(rows):
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


def _c37_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c37_logical_reading(G, a, b, hx, hz, n, local):
    """Search actual logical witnesses, including mixed-sector words.

    Every retained witness is checked for commutation and nonmembership
    in the corresponding stabilizer span. This is not exhaustive.
    """
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    overall = None
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _c37_span(stabilizers)
        best, weight = 0, n + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if word and w < weight and _c37_outside(word, own):
                best, weight = word, w

        # Component witnesses use exactly the evaluator's A-left and
        # B-right actions. Try every opposite protograph-column embedding.
        if kind == "X":
            for j in range(ca):
                word = b["witness"] << (j * cb * q)
                if all((row & word).bit_count() % 2 == 0
                       for row in checks):
                    inspect(word)
        else:
            for beta in range(cb):
                word, source = 0, a["witness"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + beta) * q + h)
                    source ^= bit
                if all((row & word).bit_count() % 2 == 0
                       for row in checks):
                    inspect(word)

        def shorten(word, rounds=2):
            w = word.bit_count()
            for _ in range(rounds):
                changed = False
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w, changed = trial, tw, True
                if not changed:
                    break
            return word

        if best:
            inspect(shorten(best, 3))
        cols = _c37_columns(checks, n)
        for _ in range(2):
            order = list(range(n))
            local.shuffle(order)
            pivots, nulls = {}, []
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
                    nulls.append(word)
                    inspect(word)

            short = sorted(set(nulls), key=int.bit_count)[:24]
            reduced = []
            for word in short:
                word = shorten(word)
                inspect(word)
                reduced.append(word)
            for i, first in enumerate(reduced):
                for second in reduced[:i]:
                    inspect(first ^ second)
            tiny = reduced[:12]
            for i, first in enumerate(tiny):
                for j in range(i):
                    pair = first ^ tiny[j]
                    for h in range(j):
                        inspect(pair ^ tiny[h])
            if best:
                starting = best
                for i, first in enumerate(stabilizers):
                    inspect(starting ^ first)
                    for second in stabilizers[:i]:
                        inspect(starting ^ first ^ second)

        if not best:
            return None
        if not _c37_outside(best, own):
            raise ValueError("logical witness became a stabilizer")
        if any((row & best).bit_count() & 1 for row in checks):
            raise ValueError("logical witness failed commutation")
        overall = weight if overall is None else min(overall, weight)
    return overall


@lru_cache(maxsize=1)
def _c37_portfolio(G, count):
    """Deterministic optimized portfolio with immutable published cells.

    Memoization changes cost, not outputs. No slot depends on a previous
    slot's random state, and an exhausted catalogue is never padded.
    """
    from itertools import combinations_with_replacement
    from collections import Counter
    from hashlib import sha256

    if G.q != 37:
        raise ValueError("new portfolio requires C37")
    q, cap = G.q, min(9, W_MAX)
    dummy = [[[]]]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def profiles(c):
        # Each column has three terms. Remove row exchange and column
        # permutations; retain only balanced constituent row budgets.
        cards = set()
        for top in combinations_with_replacement(range(4), c):
            bottom = tuple(3 - t for t in top)
            totals = (sum(top), sum(bottom))
            if min(totals) < 4 or max(totals) > 6:
                continue
            cards.add(min(tuple(top), tuple(sorted(bottom))))
        return sorted(cards)

    def draw(c, card, local):
        grid = [[[] for _ in range(c)] for _ in range(2)]
        for j, top in enumerate(card):
            counts = (top, 3 - top)
            labels = [local.sample(range(q), size) for size in counts]
            # Fix one column gauge only. Relative offsets and binomial/
            # trinomial gaps remain independent optimization variables.
            anchor = next(values[0] for values in labels if values)
            for i in range(2):
                grid[i][j] = [
                    G.elements[(t - anchor) % q]
                    for t in sorted(labels[i])
                ]
        return grid

    def component(grid, side):
        c = len(grid[0])
        width = c * q
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, dummy)
        else:
            _, rows, _ = lifted_matrices(G, dummy, grid)
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    def cycle_cost(rows):
        # Penalize actual four-cycles of the verified component.
        return sum(
            overlap * (overlap - 1) // 2
            for i, row in enumerate(rows)
            for other in rows[:i]
            for overlap in [(row & other).bit_count()]
        )

    def make_bank(side, c):
        bank, fillings = [], set()
        for ci, card in enumerate(profiles(c)):
            local = random.Random(
                BASE_SEED ^ (0xC37000 + 100003 * c + 7919 * ci))
            raw = []
            for trial in range(192):
                grid = draw(c, card, local)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows = component(grid, side)
                if gf2_rank(rows) < 2 * q - 2:
                    continue
                cols = _c37_columns(rows, c * q)
                if not _c37_clear4(cols):
                    continue
                raw.append((cycle_cost(rows), trial, grid, cols))
            raw.sort(key=lambda item: item[:2])

            finalists = []
            # Every shortlist entry receives the same witness budget
            # and exact six-word calculation.
            for cost, trial, grid, cols in raw[:10]:
                six = _c37_six_count(cols)
                reading, witness = _c37_probe(cols, local, 8)
                finalists.append(dict(
                    grid=freeze(grid), cols=cols, cost=cost,
                    trial=trial, card=card, six=six,
                    reading=reading, witness=witness))
            finalists.sort(key=lambda r: (
                -r["reading"], r["six"], r["cost"], r["trial"]))
            for rec in finalists[:4]:
                reading, witness = _c37_probe(
                    rec["cols"], local, 12, rec["witness"])
                rec["reading"], rec["witness"] = reading, witness
                bank.append(rec)
        bank.sort(key=lambda r: (
            -r["reading"], r["six"], r["cost"], r["card"], r["trial"]))
        return bank

    A_bank = make_bank("A", 3)
    B_bank = make_bank("B", 4)
    if not A_bank or not B_bank:
        raise ValueError("optimized cubic constituent bank exhausted")

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"],
         a["six"] + b["six"], a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(A_bank)
        for bi, b in enumerate(B_bank))

    def graph_key(hx, hz, n):
        # All features are intrinsic to the full uncoloured Tanner graph.
        # Unequal keys certify nonisomorphism; equality is discarded
        # conservatively, without claiming an isomorphism.
        rows = list(hx) + list(hz)
        cols = _c37_columns(rows, n)
        key = (
            n, len(rows),
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            _tanner_signature(list(hx), list(hz)),
        )
        return sha256(repr(key).encode("ascii")).digest()

    results, graph_keys = [], set()
    for _, _, _, _, ai, bi in plans:
        a, b = A_bank[ai], B_bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        # Explicitly check BOTH sums before invoking the lift.
        wx = max(sum(row) + sum(B[r][j].__len__() for r in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(A[i][j].__len__() for i in range(2))
                 for row in sb for j in range(3))
        if max(wx, wz) > cap or not _no_dead_lines(sa, sb):
            continue
        if _grid_weights(sa, sb) != max(wx, wz):
            raise ValueError("row-budget helpers disagree")

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 592 or not (125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if not (_c37_clear4(_c37_columns(hx, n))
                and _c37_clear4(_c37_columns(hz, n))):
            continue
        signature = graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified C37 lift failed CSS")
        graph_keys.add(signature)

        local = random.Random(BASE_SEED ^ (0xC37F00 + 1009 * ai + bi))
        reading = _c37_logical_reading(G, a, b, hx, hz, n, local)
        if reading is None:
            continue
        quality = (k * reading * reading / n, reading,
                   -a["six"] - b["six"], -a["cost"] - b["cost"], k)
        results.append((
            quality, A, B, k, reading,
            (a["card"], b["card"]),
            (a["reading"], b["reading"]),
            (a["six"], b["six"]), signature.hex()))
        # Refine beyond the requested portfolio instead of emitting the
        # first labels that happen to satisfy a slot-specific hash bin.
        if len(results) >= max(count, 5) + 24:
            break

    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed selection of a full-matrix-screened C37 design."""
    entries = _c37_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("distinct C37 target-rate portfolio exhausted")
    quality, A, B, k, reading, cards, components, six, signature = entries[idx]
    return {
        "A": A, "B": B, "k": k, "quality": quality,
        "support_cards": cards, "component_readings": components,
        "component_six_counts": six,
        "logical_witness_reading": reading,
        "complete_kernel4_excluded": True,
        "intrinsic_tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the full optimized cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: Independent grounded C11 constituents and intrinsic ownership bins supply twenty inequivalent designs or escape distance four: the current sixty-slot evaluation retained only five distinct codes, all five measured at distance four.
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
