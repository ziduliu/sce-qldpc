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
CONCEPT = ("Metacyclic Z5:Z2 with inversion action s=4, order 10, A=B=(3,7): balanced degree-two/three constituents screened against short kernel words should supply five novel distance-eight readings in cell 0.300; this is a measurement hypothesis, not a distance certificate.")
GROUP = ("Retain the metacyclic construction rule, changing parameters to m=5,r=2,s=4. The new factor 58 makes order 11 inadmissible: n=638. Order ten gives n=580. The reported order-ten metacyclic (3,7)x(3,7) bed reached fitness 13.5; target distance eight rather than claiming a group-distance advantage.")
SHAPES = ("Equal (3,7) sides give 70-variable, 30-check constituents, n=580 and kb=160, guaranteed rate 16/58=0.275862. Verified actual ranks retain [0.275,0.325). Deliberately target cell 0.300 instead of forcing a wider degree-two construction back into 0.500. Parent weight-four minima broadly occupied sector-1 Acol/Bcol blocks; cubic columns change that constituent mechanism.")
FILLING = ("Twenty cards pair four balanced support profiles with two element models. Every constituent row has six terms; columns have degrees two or three, giving BOTH LP maxima nine. Verified constituent lifts exclude kernel weights through four, preferentially through six, before independent witness refinement. Complete lifts enforce actual rate, kernel-four exclusion and CSS, then probe nontrivial logical witnesses including mixed sectors. Unequal intrinsic uncoloured Tanner signatures separate published slots; no ownership hash bucket or archived filling seeds the search. Distance-eight survival and novelty against the unseen archive remain hypotheses.")
FAMILY_NAME = "r030_m5r2s4_37x37_balanced_cubic_kernel_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this metacyclic parameter case has order ten")
    # Same metacyclic construction rule. Factor 58 excludes order eleven.
    # Z_5 x| Z_2, with 4^2 = 1 mod 5.
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


def _retired_z11_context(rng, G, idx):
    """Inactive historical context; never fills the new shapes.

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


_CUBIC_PROFILES = (
    ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ((1, 1, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2)),
    ((1, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2)),
    ((2, 1, 0), (0, 1, 2), (2, 0, 1), (0, 2, 1)),
)


def _shape_columns(rows, width):
    """Column syndromes of a supplied VERIFIED matrix."""
    out = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _shape_clear(rows, width, six=False):
    """Exact exclusion through four, or through six.

    Equal syndromes of distinct supports of size <=2 or <=3 expose
    every nonzero kernel word in the corresponding weight range.
    """
    cols = _shape_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    if six:
        for i, first in enumerate(cols):
            for j in range(i):
                pair = first ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return False
                    seen.add(syndrome)
    return True


def _shape_span(rows):
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


def _shape_reduce(word, pivots):
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


def _shape_probe(local, rows, width, passes, seed=0, stabilizers=None):
    """Actual kernel/logical witnesses; absence of a lighter find is not proof."""
    cols = _shape_columns(rows, width)
    span = _shape_span(stabilizers) if stabilizers is not None else {}
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def label(word):
        return _shape_reduce(word, span) if stabilizers is not None else word

    def record(word, cls=None):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if cls is None:
            cls = label(word)
        if cls:
            best, weight = word, word.bit_count()

    if seed:
        assert label(seed)
        assert all((row & seed).bit_count() % 2 == 0 for row in rows)

    for turn in range(passes):
        order = list(range(width))
        local.shuffle(order)
        # Alternate which sector reaches the information set first.
        if stabilizers is not None and turn % 2:
            order.sort(key=lambda j: j < 49 * 10)

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
                cls = label(word)
                basis.append((word, cls))
                record(word, cls)

        local.shuffle(basis)
        for i, (word, cls) in enumerate(basis):
            w = word.bit_count()
            for j, (other, other_cls) in enumerate(basis):
                if i == j:
                    continue
                candidate = word ^ other
                new_cls = cls ^ other_cls
                nw = candidate.bit_count()
                if new_cls and 0 < nw < w:
                    word, cls, w = candidate, new_cls, nw
            basis[i] = (word, cls)
            record(word, cls)

        short = sorted(basis, key=lambda item: item[0].bit_count())[:24]
        for i, (first, cls) in enumerate(short):
            for j in range(i):
                pair = first ^ short[j][0]
                pair_cls = cls ^ short[j][1]
                record(pair, pair_cls)
                for h in range(j):
                    record(pair ^ short[h][0],
                           pair_cls ^ short[h][1])

        if best:
            starting, starting_cls = best, label(best)
            for i, (first, cls) in enumerate(short):
                record(starting ^ first, starting_cls ^ cls)
                for second, other_cls in short[:i]:
                    record(starting ^ first ^ second,
                           starting_cls ^ cls ^ other_cls)
        if weight <= 6:
            break

    if stabilizers is not None and best:
        for _ in range(3):
            previous = best
            for row in stabilizers:
                record(best ^ row)
            starting = best
            moves = [row for row in stabilizers
                     if (row & starting).bit_count() >= 2]
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    record(starting ^ first ^ second)
            if best == previous:
                break

    if best:
        assert label(best)
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return weight, best


def _shape_component(G, grid, side):
    """Expose precisely the scaffold's constituent action."""
    unit = [[[G.e]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    width = 7 * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows[:3 * G.q]]


def _shape_cycle_cost(rows):
    cost = 0
    for i, first in enumerate(rows):
        for second in rows[:i]:
            overlap = (first & second).bit_count()
            cost += overlap * (overlap - 1) // 2
    return cost


def _shape_signature(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    No CSS colour, group coordinate or protograph position enters it.
    Unequal signatures certify non-isomorphism; equality is inconclusive
    and therefore causes conservative rejection.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    cols = _shape_columns(rows, width)
    m = len(rows)
    colours = []
    for row in rows:
        degrees = Counter()
        bits = row
        while bits:
            bit = bits & -bits
            degrees[cols[bit.bit_length() - 1].bit_count()] += 1
            bits ^= bit
        colours.append((row.bit_count(), tuple(sorted(degrees.items()))))

    one, many = [0] * m, [0] * m
    pairs = []
    for i, first in enumerate(rows):
        for j in range(i):
            overlap = (first & rows[j]).bit_count()
            pairs.append((i, j, overlap))
            if overlap:
                relation = one if overlap == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i, j, overlap in pairs:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(colours[j], overlap, common)] += 1
        profiles[j][(colours[i], overlap, common)] += 1
    return (
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted(
            (colours[i], tuple(sorted(profiles[i].items())))
            for i in range(m))),
    )


@lru_cache(maxsize=8)
def _shape_bank(G, profile, mode, side):
    """Immutable independently optimized constituent bank.

    Four cubic columns and three cross-row degree-two columns give
    row totals six and column totals (3,3,3,3,2,2,2).
    """
    local = random.Random(
        BASE_SEED * 10007 + 0x103707
        + 65537 * profile + 1000003 * mode
        + (170003 if side == "B" else 0))
    cubic = _CUBIC_PROFILES[profile]
    vectors = cubic + ((1, 1, 0), (1, 0, 1), (0, 1, 1))
    counts = tuple(tuple(column[i] for column in vectors)
                   for i in range(3))
    assert tuple(map(sum, counts)) == (6, 6, 6)
    assert tuple(map(sum, vectors)) == (3, 3, 3, 3, 2, 2, 2)

    rotations = [g for g in G.elements if g[1] == 0 and g != G.e]
    reflections = [g for g in G.elements if g[1] == 1]
    ratios = rotations if mode == 0 else reflections

    def draw():
        grid = []
        for i, row in enumerate(counts):
            cells = []
            for j, count in enumerate(row):
                if not count:
                    cells.append([])
                    continue
                first = G.pick(local)
                if count == 1:
                    cell = [first]
                else:
                    relative = local.choice(ratios)
                    cell = [first, G.mul(first, relative)]
                assert len({G._key(g) for g in cell}) == count
                cells.append(cell)
            grid.append(cells)
        return grid

    raw, fillings = [], set()
    for trial in range(160):
        grid = draw()
        key = tuple(tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
        if key in fillings:
            continue
        fillings.add(key)
        rows = _shape_component(G, grid, side)
        rank = gf2_rank(rows)
        if rank < 3 * G.q - 2:
            continue
        if not _shape_clear(rows, 7 * G.q):
            continue
        reading, witness = _shape_probe(
            local, rows, 7 * G.q, 2)
        cost = _shape_cycle_cost(rows)
        raw.append(dict(grid=grid, rows=rows, rank=rank,
                        reading=reading, witness=witness,
                        cost=cost, trial=trial))
    raw.sort(key=lambda item: (
        -item["reading"], item["cost"], item["trial"]))

    clean, fallback = [], []
    for record in raw[:64]:
        clean6 = _shape_clear(record["rows"], 7 * G.q, six=True)
        record["clean6"] = clean6
        (clean if clean6 else fallback).append(record)
        if len(clean) >= 12:
            break
    selected = clean + fallback[:max(0, 12 - len(clean))]
    if not selected:
        raise ValueError("balanced cubic constituent bank exhausted")

    for record in selected:
        reading, witness = _shape_probe(
            local, record["rows"], 7 * G.q, 12,
            seed=record["witness"])
        record.update(reading=reading, witness=witness)
        if record["clean6"]:
            assert reading >= 7
        record["quality"] = (
            reading, int(record["clean6"]), -record["cost"],
            -record["trial"])
    selected.sort(key=lambda item: item["quality"], reverse=True)
    return tuple(selected)


def _shape_logical_reading(local, G, A, B, hx, hz, n, a, b):
    """Embed verified constituent witnesses, then search the full CSS pair."""
    q = G.q
    best = n + 1
    for kind, checks, stabilizers, source in (
            ("X", hz, hx, b), ("Z", hx, hz, a)):
        span = _shape_span(stabilizers)
        seed = 0
        for block in range(7):
            if kind == "X":
                word = source["witness"] << (block * 7 * q)
            else:
                word = 0
                bits = source["witness"]
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 7 + block) * q + h)
                    bits ^= bit
            if not word:
                continue
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _shape_reduce(word, span):
                continue
            if not seed or word.bit_count() < seed.bit_count():
                seed = word
        reading, witness = _shape_probe(
            local, checks, n, 4, seed=seed,
            stabilizers=stabilizers)
        if witness:
            best = min(best, reading)
    return best


@lru_cache(maxsize=1)
def _shape_portfolio(G):
    """Deterministic immutable portfolio of twenty structural cards.

    Extra harness slots independently optimize the same card families,
    but cannot publish a repeated intrinsic full-Tanner signature.
    """
    from itertools import combinations_with_replacement

    if G.q != 10:
        raise ValueError("new shapes require the order-ten parameter case")
    pair_cards = tuple(combinations_with_replacement(range(4), 2))
    entries, excluded, attempted = [], set(), set()
    cap = min(9, W_MAX)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for idx in range(N_SLOTS):
        card = idx % 20
        ap, bp = pair_cards[card % 10]
        mode = card // 10
        aa = _shape_bank(G, ap, mode, "A")
        bb = _shape_bank(G, bp, mode, "B")
        local = random.Random(
            BASE_SEED * 10007 + 0x103708 + 65537 * idx)

        plans = []
        for ai, a in enumerate(aa):
            for bi, b in enumerate(bb):
                weak = min(a["reading"], b["reading"])
                plans.append((
                    -weak,
                    -int(a["clean6"]) - int(b["clean6"]),
                    a["cost"] + b["cost"],
                    local.random(), ai, bi))
        plans.sort()
        survivors = []
        tested = 0

        for _, _, _, _, ai, bi in plans:
            a, b = aa[ai], bb[bi]
            A, B = a["grid"], b["grid"]
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in attempted:
                continue
            attempted.add(filling)
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 580 and k > 0
                    and 275 * n <= 1000 * k < 325 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue
            if not (_shape_clear(hx, n) and _shape_clear(hz, n)):
                continue
            signature = _shape_signature(hx, hz, n)
            if signature in excluded:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified cubic lift failed CSS")

            reading = _shape_logical_reading(
                local, G, A, B, hx, hz, n, a, b)
            if reading == n + 1:
                continue
            weak = min(reading, a["reading"], b["reading"])
            survivors.append(dict(
                A=A, B=B, k=k, signature=signature,
                reading=reading,
                component_readings=(a["reading"], b["reading"]),
                component_clear6=(a["clean6"], b["clean6"]),
                quality=(k * weak * weak, weak,
                         -a["cost"] - b["cost"])))
            tested += 1
            if tested >= 3:
                break

        if not survivors:
            entries.append(None)
            continue
        winner = max(survivors, key=lambda item: item["quality"])
        excluded.add(winner["signature"])
        assert lifted_k(G, winner["A"], winner["B"]) == winner["k"]
        entries.append((
            freeze(winner["A"]), freeze(winner["B"]),
            winner["k"], winner["component_readings"],
            winner["component_clear6"], winner["reading"]))
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete, independently screened structural card."""
    entry = _shape_portfolio(G)[idx]
    if entry is None:
        raise ValueError("novel balanced-cubic card exhausted")
    A, B, k, readings, clear6, logical_reading = entry
    return {
        "A": A, "B": B, "k": k, "card": idx % 20,
        "component_witness_readings": readings,
        "component_kernel6_excluded": clear6,
        "logical_witness_reading": logical_reading,
        "complete_kernel4_excluded": True,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell; never resample or mutate the portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: Intrinsic ownership bins ensure one evaluator-distinct code per slot on the current Z11 incidence catalogue: sixty evaluated slots retained only five classes, and all five measured distances remained four.
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
