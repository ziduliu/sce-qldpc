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
CONCEPT = ("C11 semidirect C4 with inversion, order 44, A=B=(2,3): saturated constituents with new zero-cell incidences and actual k>=48 should supply five novel d30s>=24 readings, exceeding the 46.5234 frontier in cell 0.100. This is a falsifiable measurement hypothesis.")
GROUP = ("Keep the parent's cyclic-by-cyclic semidirect construction rule, changing its parameters to m=11,r=4,s=10 because the larger shape excludes q=110. Inversion has order two dividing four. This order-44 group is isomorphic to the measured Dic11 bed that leads cell 0.100; its distance-24 readings motivate the experiment but do not certify distance for new fillings.")
SHAPES = ("Change both sides to (2,3): equal constituents have 132 variables and 88 checks. Factor thirteen permits q<=46; q=44 gives n=572 and guaranteed kb=44. Verified complete ranks retain 48<=k<=71, entirely in open-ended cell 0.100. Five k=48,d30s=24 codes would score 48.3357, above its 46.5234 incumbent. The parent's measured weight-eighteen minima occupied only sector-1 Acol/Bcol blocks, motivating replacement of both single-row constituents.")
FILLING = ("Twenty saturated-support/C4-parity cards include zero-cell profiles (0,2,3)/(3,1,1) and (0,1,4)/(3,2,0), not merely relabelings of dense supports. Four-term cells use three even-quotient terms and one odd term to avoid automatically cancelling every normal-C11 coset. Every constituent row totals five and every column is at most four, so BOTH CSS maxima are nine, also below twelve. Verified lifts provide exact kernel-four rejection, exhaustive normal-C11 invariant witnesses and equally refined constituent probes. Full lifts enforce actual rank, CSS, nontrivial embedded witnesses and mixed-sector stabilizer-aware probes. Immutable publication rejects equal uncoloured Tanner invariants conservatively; larger readings and archive novelty remain measurement questions.")
FAMILY_NAME = "r010_m11r4s10_q44_23x23_zeroedge_rank48_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C11 semidirect C4 requires order 44")
    # Same semidirect cyclic-by-cyclic rule as the parent.
    # x^11=y^4=1, y*x*y^-1=x^-1.
    m, r, s = 11, 4, 10
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi] * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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


def _unused_z11_owned_catalogue(rng, G, idx):
    """Uncalled historical Z11 catalogue, not valid for the new bed.

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


def _jump_columns(rows, width):
    """Column syndromes of supplied verified rows."""
    cols = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _jump_kernel4(rows, width):
    """Exact kernel exclusion through four, plus a cycle tie-break."""
    cols = _jump_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return None
        seen.add(syndrome)
    squares = 0
    for i, first in enumerate(cols):
        for j in range(i):
            second = cols[j]
            syndrome = first ^ second
            if syndrome in seen:
                return None
            seen.add(syndrome)
            overlap = (first & second).bit_count()
            squares += overlap * (overlap - 1) // 2
    return cols, squares


def _jump_probe(rng, cols, passes, seed=0):
    """Heuristic kernel search carrying an actual, nonincreasing witness.

    All coordinates are those of the verified matrix. Larger readings
    are upper bounds, not certificates. Basis exchanges preserve span.
    """
    width = len(cols)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def record(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
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

        rng.shuffle(basis)
        for i in range(len(basis)):
            current = basis[i]
            current_weight = current.bit_count()
            for j in range(len(basis)):
                if i == j:
                    continue
                trial = current ^ basis[j]
                w = trial.bit_count()
                if 0 < w < current_weight:
                    current, current_weight = trial, w
            basis[i] = current
            record(current)

        for i, first in enumerate(basis):
            for j in range(i):
                record(first ^ basis[j])
        short = sorted(basis, key=int.bit_count)[:40]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    record(pair ^ short[h])

        for descent in range(2):
            starting = best
            for i, first in enumerate(basis):
                record(starting ^ first)
                for j in range(i):
                    record(starting ^ first ^ basis[j])
            if best == starting:
                break
        if weight <= 10:
            break

    if not best:
        raise ValueError("constituent kernel witness absent")
    syndrome = 0
    bits = best
    while bits:
        bit = bits & -bits
        syndrome ^= cols[bit.bit_length() - 1]
        bits ^= bit
    assert syndrome == 0
    return weight, best


def _jump_orbit_seed(rng, G, cols, blocks):
    """Probe the C7-invariant subspace using actual column syndromes.

    This is a witness search on the complete invariant subspace, not
    exhaustive enumeration of its exponentially many kernel words.
    """
    words, syndromes = [], []
    for j in range(blocks):
        for b in range(8):
            word = syndrome = 0
            for a in range(7):
                position = j * G.q + G.idx((a, b))
                word |= 1 << position
                syndrome ^= cols[position]
            words.append(word)
            syndromes.append(syndrome)
    _, compressed = _jump_probe(rng, syndromes, 12)
    expanded = 0
    while compressed:
        bit = compressed & -compressed
        expanded ^= words[bit.bit_length() - 1]
        compressed ^= bit
    return expanded


def _jump_logical_reading(G, Arec, Brec, hx, hz):
    """Embed and reduce explicit nontrivial logical witnesses.

    Membership, commutation and the final logical class are checked.
    Stabilizer-pair descent may enter sector 2. It is not exhaustive.
    """
    q = G.q
    best = None
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        for block in range(2 if kind == "X" else 4):
            if kind == "X":
                word = Brec["witness"] << (block * 4 * q)
            else:
                word = 0
                source = Arec["witness"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 4 + block) * q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not nontrivial(word):
                continue
            weight = word.bit_count()
            for descent in range(2):
                starting = word
                for i, first in enumerate(stabilizers):
                    trial = starting ^ first
                    w = trial.bit_count()
                    if w < weight:
                        word, weight = trial, w
                    for j in range(i):
                        trial = starting ^ first ^ stabilizers[j]
                        w = trial.bit_count()
                        if w < weight:
                            word, weight = trial, w
                if word == starting:
                    break
            assert word and nontrivial(word)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            best = weight if best is None else min(best, weight)
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _f110_catalogue(G, count):
    """Deterministic portfolio; cached cells are immutable.

    Memoization changes computation cost, not the selected matrices.
    All screening matrices come from the scaffold's verified lift.
    """
    from collections import Counter
    from hashlib import sha256

    q, e, width = G.q, G.e, 2 * G.q
    if q != 110:
        raise ValueError("faithful cubic portfolio requires order 110")
    cap = min(9, 12, W_MAX)
    support = [[3, 3]]
    wx = max(sum(support[0]) + support[0][j] for j in range(2))
    wz = max(sum(support[0]) + support[0][j] for j in range(2))
    assert wx == wz == 9 <= cap
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) == 9

    models = (
        ((0, 1), (1, 2)), ((0, 1), (1, 3)),
        ((0, 3), (2, 3)), ((1, 2), (1, 4)),
        ((1, 3), (2, 4)), ((1, 4), (2, 6)),
        ((1, 6), (3, 4)), ((2, 4), (1, 7)),
        ((2, 6), (3, 7)), ((3, 4), (4, 7)),
    )
    mask = (1 << width) - 1

    def component(grid, right=False):
        # The zero auxiliary cell is never published.
        if right:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        else:
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        assert len(rows) == q
        assert all(row.bit_length() <= width for row in rows)
        return [row & mask for row in rows]

    def draw(local, card):
        row = []
        for j, (b0, b1) in enumerate(models[card % 10]):
            # The second ten cards change only the second triangle's
            # quotient orders, not a gauge of a completed constituent.
            if card >= 10 and j == 1:
                b0, b1 = (b0 + 5) % 10, (b1 + 5) % 10
            for _ in range(64):
                a = local.randrange(1, 11) if b0 == 0 else local.randrange(11)
                c = local.randrange(1, 11) if b1 == 0 else local.randrange(11)
                g, h = (a, b0), (c, b1)
                if g == e or h == e or g == h:
                    continue
                if G.mul(g, h) == G.mul(h, g):
                    continue
                row.append([e, g, h])
                break
            else:
                return None
        return [row]

    def cycle_cost(cols, squares):
        adjacency = [0] * q
        internal = 0
        for col in cols:
            degree = col.bit_count()
            internal += degree * (degree - 1) * (degree - 2) // 6
            remaining = col
            while remaining:
                bit = remaining & -remaining
                adjacency[bit.bit_length() - 1] |= col ^ bit
                remaining ^= bit
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            remaining = neighbors >> (i + 1)
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() + i
                triangles += ((neighbors & adjacency[j]) >> (j + 1)).bit_count()
                remaining ^= bit
        # This is a selection penalty, not a distance lower bound.
        return 12 * squares + max(0, triangles - internal)

    def orbit_seed(local, cols):
        """Search actual normal-C11 coset kernel words."""
        atoms, syndromes = [], []
        for j in range(2):
            for b in range(10):
                word = syndrome = 0
                for a in range(11):
                    position = j * q + G.idx((a, b))
                    word |= 1 << position
                    syndrome ^= cols[position]
                atoms.append(word)
                syndromes.append(syndrome)
        _, compressed = _jump_probe(local, syndromes, 12)
        word = 0
        while compressed:
            bit = compressed & -compressed
            word ^= atoms[bit.bit_length() - 1]
            compressed ^= bit
        return word

    bank, filling_keys = [], set()
    for card in range(20):
        local = random.Random(BASE_SEED ^ (0xF110C0 + 104729 * card))
        raw = []
        for trial in range(96):
            grid = draw(local, card)
            if grid is None:
                continue
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for cell in grid[0])
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            rows = component(grid)
            stats = _jump_kernel4(rows, width)
            if stats is None:
                continue
            cols, squares = stats
            if gf2_rank(rows) < q - 6:
                continue
            raw.append((cycle_cost(cols, squares), trial, grid, rows, cols))
        raw.sort(key=lambda item: item[:2])

        shortlisted = []
        for cost, trial, grid, rows, cols in raw[:12]:
            seed = orbit_seed(local, cols)
            reading, witness = _jump_probe(local, cols, 4, seed)
            shortlisted.append(dict(
                grid=grid, rows=rows, cols=cols, cost=cost,
                trial=trial, card=card, distance=reading, witness=witness))
        shortlisted.sort(key=lambda r: (
            -r["distance"], r["cost"], r["trial"]))

        # Every retained finalist receives the same 4+24-pass budget.
        for record in shortlisted[:3]:
            reading, witness = _jump_probe(
                local, record["cols"], 24, record["witness"])
            if reading < 12:
                continue
            record["distance"], record["witness"] = reading, witness
            right_rows = component(record["grid"], True)
            right_word, source = 0, witness
            while source:
                bit = source & -source
                j, h = divmod(bit.bit_length() - 1, q)
                right_word |= 1 << (j * q + G.idx(G.inv(G.elements[h])))
                source ^= bit
            # Inversion transfer is checked against the actual right lift.
            assert all((row & right_word).bit_count() % 2 == 0
                       for row in right_rows)
            assert gf2_rank(right_rows) == gf2_rank(record["rows"])
            record["right_witness"] = right_word
            bank.append(record)
    if len(bank) < 5:
        raise ValueError("too few optimized faithful cubic constituents")
    bank.sort(key=lambda r: (
        -r["distance"], r["cost"], r["card"], r["trial"]))

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

    def graph_key(hx, hz, n):
        """Intrinsic uncoloured full-Tanner invariant.

        Checks have degree nine and qubits degree six/twelve, so their
        vertex classes are distinguished without CSS colouring.
        Equality is inconclusive and is rejected conservatively.
        """
        rows = hx + hz
        cols = _jump_columns(rows, n)
        incidence, shared = [], Counter()
        for col in cols:
            checks = []
            while col:
                bit = col & -col
                checks.append(bit.bit_length() - 1)
                col ^= bit
            incidence.append(checks)
            for i, a in enumerate(checks):
                for b in checks[:i]:
                    shared[(min(a, b), max(a, b))] += 1
        adjacency = [{} for _ in rows]
        for (a, b), multiplicity in shared.items():
            adjacency[a][b] = multiplicity
            adjacency[b][a] = multiplicity
        masks = [sum(1 << j for j in neighbors)
                 for neighbors in adjacency]
        features = []
        for i, neighbors in enumerate(adjacency):
            profile = Counter(
                (rows[j].bit_count(), multiplicity,
                 (masks[i] & masks[j]).bit_count())
                for j, multiplicity in neighbors.items())
            walk4 = sum((masks[i] & other).bit_count() ** 2
                        for other in masks)
            features.append((
                rows[i].bit_count(), masks[i].bit_count(),
                tuple(sorted(profile.items())), walk4))
        palette = {f: c for c, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]
        history = [tuple(sorted(Counter(features).items()))]
        for _ in range(2):
            qfeatures = [tuple(sorted(colors[r] for r in checks))
                         for checks in incidence]
            qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
            profiles = [[] for _ in rows]
            for checks, feature in zip(incidence, qfeatures):
                for r in checks:
                    profiles[r].append(qp[feature])
            refined = [(colors[r], tuple(sorted(profiles[r])))
                       for r in range(len(rows))]
            history.append((
                tuple(sorted(Counter(qfeatures).items())),
                tuple(sorted(Counter(refined).items()))))
            palette = {f: c for c, f in enumerate(sorted(set(refined)))}
            colors = [palette[f] for f in refined]
        signature = (
            n, len(rows),
            tuple(sorted(Counter(len(c) for c in incidence).items())),
            tuple(history))
        return sha256(repr(signature).encode("ascii")).digest()

    def logical_reading(a, b, hx, hz, local):
        """Actual logical witnesses, including randomized mixed sectors.

        All candidates commute, and stabilizer-span membership is checked.
        A high returned value is only an upper bound.
        """
        overall = None
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = span(stabilizers)
            best, weight = 0, 5 * q + 1

            def inspect(word):
                nonlocal best, weight
                w = word.bit_count()
                if word and w < weight and outside(word, own):
                    best, weight = word, w

            for block in range(2):
                if kind == "X":
                    word = b["right_witness"] << (block * width)
                else:
                    word, source = 0, a["witness"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 2 + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                inspect(word)

            cols = _jump_columns(checks, 5 * q)
            order = list(range(5 * q))
            for _ in range(2):
                local.shuffle(order)
                pivots, nulls = {}, []
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
                        nulls.append(word)
                        inspect(word)

                # Shorten actual kernel words by actual stabilizers.
                reduced = []
                for word in nulls:
                    w = word.bit_count()
                    for row in stabilizers:
                        trial = word ^ row
                        tw = trial.bit_count()
                        if 0 < tw < w:
                            word, w = trial, tw
                    inspect(word)
                    if outside(word, own):
                        reduced.append(word)
                reduced = sorted(set(reduced), key=int.bit_count)
                short = reduced[:28]
                for i, first in enumerate(short):
                    for second in short[:i]:
                        inspect(first ^ second)
                tiny = short[:16]
                for i, first in enumerate(tiny):
                    for j in range(i):
                        pair = first ^ tiny[j]
                        for h in range(j):
                            inspect(pair ^ tiny[h])

                if best:
                    for descent in range(2):
                        starting = best
                        for i, first in enumerate(stabilizers):
                            inspect(starting ^ first)
                            for second in stabilizers[:i]:
                                inspect(starting ^ first ^ second)
                        if best == starting:
                            break
            if not best:
                return None
            assert outside(best, own)
            assert all((row & best).bit_count() % 2 == 0 for row in checks)
            overall = weight if overall is None else min(overall, weight)
        return overall

    plans = sorted(
        (-min(a["distance"], b["distance"]),
         -a["distance"] - b["distance"], a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))
    results, graph_keys = [], set()
    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or not (175 * n <= 1000 * k < 225 * n):
            continue
        assert k >= 110
        signature = graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        if (_jump_kernel4(hx, n) is None
                or _jump_kernel4(hz, n) is None):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified faithful cubic lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) == 9 <= cap
        local = random.Random(BASE_SEED ^ (0xF110D0 + 1009 * ai + bi))
        reading = logical_reading(a, b, hx, hz, local)
        if reading is None:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, reading,
                   -a["cost"] - b["cost"], k)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["distance"], b["distance"]), signature.hex()))
        # Refine a wider equally screened portfolio before publication.
        if len(results) >= max(count, 5) + 16:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def _s44_span(rows):
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


def _s44_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _s44_component(G, grid, right=False):
    """Only the verified scaffold determines regular-action conventions."""
    if right:
        _, rows, _ = lifted_matrices(G, [[[]]], grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
    assert len(rows) == 2 * G.q
    assert all(row.bit_length() <= 3 * G.q for row in rows)
    return rows


def _s44_orbit_seed(G, rows):
    """Exhaustive normal-C11 invariant kernel, with an actual witness."""
    cols = _jump_columns(rows, 3 * G.q)
    pivots, basis = {}, []
    for j in range(3):
        for b in range(4):
            word = syndrome = 0
            for a in range(11):
                position = j * G.q + G.idx((a, b))
                word |= 1 << position
                syndrome ^= cols[position]
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

    # Twelve disjoint coset atoms make this enumeration small.
    best = 0
    previous = word = 0
    for number in range(1, 1 << len(basis)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous).bit_length() - 1
        word ^= basis[changed]
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word
        previous = gray
    assert best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _s44_logical(checks, stabilizers, n, local, seeds):
    """Actual logical upper bound, including both LP sectors.

    Kernel construction, row-span membership and stabilizer descent all
    use the verified complete matrices. This is not exhaustive beyond
    the separate exact kernel-four gate.
    """
    own = _s44_span(stabilizers)
    best = 0
    weight = n + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight and _s44_outside(word, own):
            best, weight = word, w

    for seed in seeds:
        assert all((row & seed).bit_count() % 2 == 0 for row in checks)
        inspect(seed)

    cols = _jump_columns(checks, n)
    order = list(range(n))
    for _ in range(2):
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

        reduced = []
        for word in nulls:
            w = word.bit_count()
            for row in stabilizers:
                trial = word ^ row
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            inspect(word)
            if _s44_outside(word, own):
                reduced.append(word)

        short = sorted(set(reduced), key=int.bit_count)[:32]
        for i, first in enumerate(short):
            for second in short[:i]:
                inspect(first ^ second)
        tiny = short[:18]
        for i, first in enumerate(tiny):
            for j in range(i):
                pair = first ^ tiny[j]
                for h in range(j):
                    inspect(pair ^ tiny[h])

        for descent in range(3):
            if not best:
                break
            starting = best
            for row in stabilizers:
                inspect(starting ^ row)
            starting = best
            nearby = sorted(
                stabilizers, key=lambda row: (starting ^ row).bit_count()
            )[:48]
            for i, first in enumerate(nearby):
                for second in nearby[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

    if best:
        assert _s44_outside(best, own)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
    return best


def _s44_graph_key(hx, hz, n):
    """Conservative uncoloured Tanner invariant; equality is inconclusive."""
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _jump_columns(rows, n)
    # No group indices, protograph labels or CSS colouring enter this key.
    key = (
        n, len(rows),
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        _tanner_signature(rows, []),
    )
    return sha256(repr(key).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _s44_portfolio(G, count):
    """Pure immutable portfolio of twenty support/parity prescriptions.

    Historical order-110 helpers remain uncalled. Caching changes build
    cost only; no slot can mutate another slot's selected filling.
    """
    q, e, width = G.q, G.e, 3 * G.q
    assert q == 44 and A_SHAPE == B_SHAPE == (2, 3)
    cap = min(9, 12, W_MAX)
    profiles = (
        ((1, 1, 3), (1, 3, 1)),
        ((1, 1, 3), (2, 2, 1)),
        ((1, 2, 2), (2, 1, 2)),
        ((0, 2, 3), (3, 1, 1)),
        ((0, 1, 4), (3, 2, 0)),
    )
    for support in profiles:
        assert all(sum(row) == 5 for row in support)
        assert max(sum(row[j] for row in support) for j in range(3)) <= 4
        assert _no_dead_lines(support, support)
        assert _grid_weights(support, support) == 9 <= cap

    def draw(local, card):
        profile, mode = divmod(card, 4)
        grid = []
        for i, counts in enumerate(profiles[profile]):
            row = []
            for j, terms in enumerate(counts):
                if not terms:
                    row.append([])
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    row.append([anchor])
                    continue

                if mode == 0:
                    even_b, odd_b = 0, 1
                elif mode == 1:
                    even_b, odd_b = 2, 3
                elif mode == 2:
                    even_b = 2 * ((i + j) % 2)
                    odd_b = local.choice((1, 3))
                else:
                    even_b = local.choice((0, 2))
                    odd_b = local.choice((1, 3))
                rotation = (local.randrange(1, 11), even_b)
                cross = (local.randrange(11), odd_b)

                if terms == 2:
                    use_cross = (
                        mode == 1
                        or (mode == 0 and (i + j) % 2 == 0)
                        or (mode == 3 and local.randrange(2) == 0))
                    relative = cross if use_cross else rotation
                    model = [e, relative]
                elif terms == 3:
                    # Noncommuting even/odd ratios, with both quotient
                    # parity lifts explored independently.
                    if (i + j + mode) % 2:
                        model = [e, cross, G.mul(cross, rotation)]
                    else:
                        model = [e, rotation, cross]
                else:
                    # Three even terms and one odd term. Unlike two
                    # equal-coset pairs, these do not automatically make
                    # every complete C11 coset a kernel word.
                    second_a = local.choice(
                        [a for a in range(1, 11) if a != rotation[0]])
                    second = (second_a, even_b)
                    model = [e, rotation, second, cross]
                cell = [G.mul(anchor, g) for g in model]
                assert len({G._key(g) for g in cell}) == terms
                row.append(cell)
            grid.append(row)
        return grid

    bank, fillings = [], set()
    for card in range(20):
        local = random.Random(BASE_SEED ^ (0x442348 + 104729 * card))
        raw = []
        for trial in range(96):
            grid = draw(local, card)
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _s44_component(G, grid)
            if gf2_rank(rows) < 2 * q - 6:
                continue
            stats = _jump_kernel4(rows, width)
            if stats is None:
                continue
            cols, squares = stats
            raw.append((squares, trial, grid, rows, cols))
        raw.sort(key=lambda record: record[:2])

        shortlist = []
        for squares, trial, grid, rows, cols in raw[:16]:
            seed = _s44_orbit_seed(G, rows)
            if seed.bit_count() < 24:
                continue
            reading, word = _jump_probe(local, cols, 4, seed)
            if reading < 24:
                continue
            shortlist.append(dict(
                grid=grid, rows=rows, cols=cols, squares=squares,
                trial=trial, card=card, reading=reading, left=word))
        shortlist.sort(key=lambda record: (
            -record["reading"], record["squares"], record["trial"]))

        # Every publishable constituent receives the same 4+24 passes.
        for record in shortlist[:4]:
            reading, word = _jump_probe(
                local, record["cols"], 24, record["left"])
            if reading < 24:
                continue
            right = 0
            source = word
            while source:
                bit = source & -source
                j, h = divmod(bit.bit_length() - 1, q)
                right |= 1 << (j * q + G.idx(G.inv(G.elements[h])))
                source ^= bit
            right_rows = _s44_component(G, record["grid"], True)
            assert all((row & right).bit_count() % 2 == 0
                       for row in right_rows)
            assert gf2_rank(right_rows) == gf2_rank(record["rows"])
            record["reading"], record["left"] = reading, word
            record["right"] = right
            bank.append(record)

    if len(bank) < 5:
        raise ValueError("too few distance-24-probed saturated constituents")

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            preliminary = (
                min(a["reading"], b["reading"], 28),
                int(a["card"] // 4 >= 3 or b["card"] // 4 >= 3),
                int(a["card"] != b["card"]),
                -a["squares"] - b["squares"])
            plans.append((preliminary, ai, bi))
    plans.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)

    results, graph_keys = [], set()
    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(r[j] for r in sb)
                 for row in sa for j in range(3))
        wz = max(sum(row) + sum(r[j] for r in sa)
                 for row in sb for j in range(3))
        assert max(wx, wz) <= cap
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 572 or not (48 <= k <= 71):
            continue
        if (_jump_kernel4(hx, n) is None
                or _jump_kernel4(hz, n) is None):
            continue
        signature = _s44_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap

        xseeds = [b["right"] << (block * width) for block in range(3)]
        zseeds = []
        for block in range(3):
            word = 0
            source = a["left"]
            while source:
                bit = source & -source
                j, h = divmod(bit.bit_length() - 1, q)
                word |= 1 << ((j * 3 + block) * q + h)
                source ^= bit
            zseeds.append(word)

        local = random.Random(BASE_SEED ^ (0x44C248 + 65537 * ai + bi))
        xword = _s44_logical(hz, hx, n, local, xseeds)
        zword = _s44_logical(hx, hz, n, local, zseeds)
        if not xword or not zword:
            continue
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 24:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        # A capped upper-bound feature prevents very poorly shortened
        # witnesses from overwhelming the measured distance-24 target.
        quality = (
            k * min(reading, 26) ** 2 / n,
            min(reading, 26), preliminary, k)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["reading"], b["reading"]), signature.hex()))
        if len(results) >= max(count, 5) + 12:
            break

    results.sort(key=lambda record: record[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish a complete immutable design on the new shape."""
    catalogue = _s44_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("rank-48 saturated portfolio exhausted")
    quality, A, B, k, reading, cards, components, signature = catalogue[idx]
    return {
        "A": A, "B": B, "k": k, "quality": quality,
        "design_cards": cards, "component_readings": components,
        "quantum_witness_reading": reading,
        "kernel4_excluded": True, "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Every cell of both new shapes is defined without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: The independent-grounded Z11 catalogue supplies twenty inequivalent designs or escapes distance four: its sixty evaluated slots retained five distinct codes, all five measured at distance four.
# REFUTED: Independently optimized C7:C8 degree-three constituents supply five measured readings >=14: all five measured finalists instead had d30s=12, with sector-1 Z families across both A columns and all B columns.
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
