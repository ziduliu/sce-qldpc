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
CONCEPT = ("Metacyclic Z35:Z1, order 35, A=(2,3), B=(2,4): shared degree-three spectral rank deficits and optimized cubic constituents should supply five novel distance-fourteen readings in cell 0.150, exceeding its 28.452 record; distance survival is a measurement hypothesis.")
GROUP = ("Retain the parent's metacyclic construction rule, changing only m,r,s to 35,1,1. Factor sixteen permits orders through 37; choose 35 for its order-seven degree-three factors. The reported C35 cubic bed reached 27.3, and C31 reached 28.452 with distance-fourteen readings. Since characteristic two does not divide 35, shared spectral losses can deliberately increase homology, but verified complete ranks decide.")
SHAPES = ("Redirect this L2 round from cell 0.500 to 0.150 using unequal (2,3)x(2,4) constituents: 105/140 variables and 70 checks each, n=560, kb=70 and guaranteed rate 0.125. A shared degree-three rank-one evaluation predicts k=85 with full augmentation or k=87 with A's augmentation loss. Both rates lie in [0.125,0.175). Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly; cubic columns replace that degree-two mechanism.")
FILLING = ("Optimize ten A and ten B support/gap prescriptions under the same GF8 root constraint, retaining only their exact intended verified constituent ranks. Every column has three terms; A row totals four/five or four/five with reversed allocation, B rows total six, giving HX<=8 and HZ<=9. Exact kernel-four exclusion, verified cyclic cofactor witnesses and equal-budget kernel searches select constituents. Complete lifts enforce actual rate, both budgets, CSS, kernel-four exclusion and nontrivial mixed-sector logical-witness searches. Intrinsic full-Tanner rooted invariants conservatively separate an immutable portfolio. Five [560,87,14] readings would score 30.45; unseen archive novelty and distance-fourteen survival remain unproved.")
FAMILY_NAME = "r015_m35r1s1_23x24_sharedGF8_cubic_witness_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 35

def custom_ops(q):
    # Same metacyclic construction rule; new size factor is sixteen.
    # Z_35 x| Z_1; order seven supplies the shared GF8 factor.
    if not valid_q(q):
        raise ValueError("this metacyclic parameter case has order 35")
    m, r, s = 35, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
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


def _retired_z11_owned_context(rng, G, idx):
    """Select an optimized, intrinsically owned complete filling.

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


def _s35_columns(rows, width):
    """Syndromes in the coordinates of a verified scaffold matrix."""
    result = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            result[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return result


def _s35_rows(G, grid, side):
    """Expose the actual left A or right B action, using a zero dummy."""
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], grid)
    return [row & mask for row in rows]


def _s35_kernel4(rows, width):
    """Exactly exclude every nonzero kernel support of weights 1..4."""
    cols = _s35_columns(rows, width)
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


def _s35_span(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous
    return basis


def _s35_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _s35_probe(rng, rows, width, passes, seed=0, own=None):
    """Search actual kernel/logical witnesses; readings are upper bounds."""
    cols = _s35_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _s35_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for turn in range(passes):
        rng.shuffle(order)
        if own is not None and turn % 2:
            n1 = A_SHAPE[1] * B_SHAPE[1] * 35
            order.sort(key=lambda j: j < n1)
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

        rng.shuffle(kernel)
        for i in range(len(kernel)):
            current = kernel[i]
            current_weight = current.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = current ^ other
                w = trial.bit_count()
                if 0 < w < current_weight:
                    current, current_weight = trial, w
            kernel[i] = current
            inspect(current)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 18:
                    for h in range(j):
                        inspect(pair ^ short[h])
        if own is None and weight <= 10:
            break
        if own is not None and weight <= 10:
            break

    if best:
        if any((row & best).bit_count() & 1 for row in rows):
            raise ValueError("verified witness failed commutation")
        if own is not None and not _s35_outside(best, own):
            raise ValueError("logical probe returned a stabilizer")
    return best


def _s35_cofactor(rows, blocks, q):
    """Cyclic cofactors extracted from the verified column syndromes."""
    from itertools import combinations
    mask = (1 << q) - 1
    cols = _s35_columns(rows, blocks * q)
    polys = [[(cols[j * q] >> (i * q)) & mask
              for j in range(blocks)] for i in range(2)]

    def rotate(poly, shift):
        return poly if not shift else (
            ((poly << shift) | (poly >> (q - shift))) & mask)

    def multiply(a, b):
        result = 0
        while a:
            bit = a & -a
            result ^= rotate(b, bit.bit_length() - 1)
            a ^= bit
        return result

    best = 0
    for js in combinations(range(blocks), 3):
        pieces = [0] * blocks
        for omitted, j in enumerate(js):
            a, b = js[:omitted] + js[omitted + 1:]
            pieces[j] = (multiply(polys[0][a], polys[1][b])
                         ^ multiply(polys[0][b], polys[1][a]))
        word = sum(poly << (j * q) for j, poly in enumerate(pieces))
        if not word:
            continue
        if any((row & word).bit_count() & 1 for row in rows):
            raise ValueError("verified cyclic cofactor failed")
        candidates = [word]
        for shift in range(1, q):
            moved = sum(rotate(poly, shift) << (j * q)
                        for j, poly in enumerate(pieces))
            if word ^ moved:
                candidates.append(word ^ moved)
        candidate = min(candidates, key=int.bit_count)
        if not best or candidate.bit_count() < best.bit_count():
            best = candidate
    return best


def _s35_spectral_draw(rng, G, profile, policy):
    """Equal row evaluations at alpha in GF8, without a private lift.

    GF8 = F2[t]/(t^3+t+1), alpha=t has order seven. Exponents
    differing by seven retain the constraint but vary the C5 direction.
    Only subsequent verified ranks decide which factors actually lose rank.
    """
    def multiply(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 0xB
        return out

    powers = [1]
    for _ in range(6):
        powers.append(multiply(powers[-1], 2))
    logs = {value: i for i, value in enumerate(powers)}
    if len(logs) != 7 or multiply(powers[-1], 2) != 1:
        raise ValueError("GF8 construction failed")

    elements = {g[0]: g for g in G.elements}
    grid = [[[] for _ in profile[0]] for _ in range(2)]
    extent = (7, 14, 21, 28, 34)[policy]
    used = set()
    for j in range(len(profile[0])):
        counts = (profile[0][j], profile[1][j])
        if sum(counts) != 3:
            raise ValueError("spectral prescription is not cubic")
        offsets, values = [], []
        for count in counts:
            if count == 0:
                terms = []
            elif count == 1:
                terms = [0]
            elif count == 2:
                choices = [d for d in range(1, extent + 1)
                           if d % 7 and min(d, 35 - d) not in used]
                if not choices:
                    choices = [d for d in range(1, 35) if d % 7]
                gap = rng.choice(choices)
                used.add(min(gap, 35 - gap))
                terms = [0, gap if rng.randrange(2) else (-gap) % 35]
            else:
                a = rng.randrange(1, 7)
                b = logs[1 ^ powers[a]]
                terms = [7 * rng.randrange(5),
                         a + 7 * rng.randrange(5),
                         b + 7 * rng.randrange(5)]
            value = 0
            for t in terms:
                value ^= powers[t % 7]
            offsets.append(terms)
            values.append(value)

        anchors = [rng.randrange(35), rng.randrange(35)]
        if counts[0] and counts[1]:
            if not values[0] or not values[1]:
                raise ValueError("mixed spectral column vanished")
            residue = (anchors[0] + logs[values[0]]
                       - logs[values[1]]) % 7
            anchors[1] = residue + 7 * rng.randrange(5)
        elif values != [0, 0]:
            raise ValueError("exclusive cubic column did not vanish")

        evaluations = []
        for i in range(2):
            labels = sorted((anchors[i] + t) % 35 for t in offsets[i])
            if len(set(labels)) != counts[i]:
                raise ValueError("duplicate spectral cell term")
            grid[i][j] = [elements[t] for t in labels]
            value = 0
            for t in labels:
                value ^= powers[t % 7]
            evaluations.append(value)
        if evaluations[0] != evaluations[1]:
            raise ValueError("shared spectral equation failed")
    return grid


def _s35_component_bank(G, side):
    """Ten deliberate support/gap cards per side, optimized before pairing."""
    profiles = (
        (((0, 2, 2), (3, 1, 1)),
         ((1, 1, 2), (2, 2, 1)))
        if side == "A" else
        (((1, 1, 2, 2), (2, 2, 1, 1)),
         ((0, 1, 2, 3), (3, 2, 1, 0)))
    )
    bank, signatures = [], set()
    for card in range(10):
        pi, policy = divmod(card, 5)
        profile = profiles[pi]
        augmentation = gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in profile])
        expected = 2 * (G.q - 1) + augmentation - 3
        local = random.Random(
            BASE_SEED ^ (0x3508C0 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        width = len(profile[0]) * G.q
        raw, fillings = [], set()
        for trial in range(128):
            grid = _s35_spectral_draw(local, G, profile, policy)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _s35_rows(G, grid, side)
            if gf2_rank(rows) != expected:
                continue
            if not _s35_kernel4(rows, width):
                continue
            seed = _s35_cofactor(rows, len(profile[0]), G.q)
            if not seed or seed.bit_count() < 12:
                continue
            penalty = sum(
                (shared * (shared - 1) // 2)
                for i, row in enumerate(rows)
                for other in rows[:i]
                for shared in [(row & other).bit_count()])
            raw.append((seed.bit_count(), penalty, trial,
                        grid, rows, seed))

        raw.sort(key=lambda item: (-item[0], item[1], item[2]))
        refined = []
        for _, penalty, trial, grid, rows, seed in raw[:12]:
            witness = _s35_probe(local, rows, width, 4, seed)
            refined.append((witness.bit_count(), penalty, trial,
                            grid, rows, witness))
        refined.sort(key=lambda item: (-item[0], item[1], item[2]))

        kept = 0
        for _, penalty, trial, grid, rows, seed in refined[:8]:
            witness = _s35_probe(local, rows, width, 20, seed)
            if not witness or witness.bit_count() < 12:
                continue
            signature = _tanner_signature(rows, [])
            if signature in signatures:
                continue
            signatures.add(signature)
            bank.append(dict(
                grid=grid, rows=rows, word=witness,
                reading=witness.bit_count(), rank=expected,
                augmentation=augmentation, card=card,
                trial=trial, penalty=penalty))
            kept += 1
            if kept >= 4:
                break
    bank.sort(key=lambda r: (
        -r["reading"], r["penalty"], r["card"], r["trial"]))
    return bank


def _s35_graph_key(G, hx, hz, n):
    """Rooted invariant of the COMPLETE Tanner graph, without CSS colours.

    Unequal keys certify nonisomorphism; equality is rejected conservatively.
    All high-degree qubits are rooted. Abelian simultaneous translation
    makes each sector-two block one orbit, restoring q multiplicities.
    """
    from collections import Counter
    from hashlib import sha256
    rows = list(hx) + list(hz)
    adjacency = [[] for _ in range(n + len(rows))]
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            j = bit.bit_length() - 1
            adjacency[j].append(n + r)
            adjacency[n + r].append(j)
            row ^= bit
    degrees = [len(neighbors) for neighbors in adjacency]
    n1 = A_SHAPE[1] * B_SHAPE[1] * G.q
    high = [j for j in range(n) if degrees[j] >= 10]
    if high != list(range(n1, n)):
        raise ValueError("unexpected intrinsic high-degree root class")

    rooted = []
    for block in range(A_SHAPE[0] * B_SHAPE[0]):
        root = n1 + block * G.q + G.idx(G.e)
        initial = [(int(v >= n), degrees[v], int(v == root))
                   for v in range(len(adjacency))]
        unique = sorted(set(initial))
        palette = {key: i for i, key in enumerate(unique)}
        colors = [palette[key] for key in initial]
        digest = sha256()
        digest.update(repr(tuple(sorted(Counter(initial).items())))
                      .encode("ascii"))
        previous = len(unique)
        for _ in range(12):
            keys = [
                (colors[v],
                 tuple(sorted(colors[w] for w in neighbors)))
                for v, neighbors in enumerate(adjacency)]
            digest.update(repr(tuple(sorted(Counter(keys).items())))
                          .encode("ascii"))
            unique = sorted(set(keys))
            palette = {key: i for i, key in enumerate(unique)}
            colors = [palette[key] for key in keys]
            if len(unique) == previous:
                break
            previous = len(unique)
        rooted.extend([digest.digest()] * G.q)
    intersections = Counter(
        (rows[i] & rows[j]).bit_count()
        for i in range(len(rows)) for j in range(i))
    return (
        n, len(rows),
        tuple(sorted(Counter(degrees[:n]).items())),
        tuple(sorted(Counter(degrees[n:]).items())),
        tuple(sorted(intersections.items())),
        tuple(sorted(rooted)))


def _s35_logical_reading(G, a, b, hx, hz, n, rng):
    """Verified nontrivial witnesses, including mixed-sector searches."""
    q, cb = G.q, B_SHAPE[1]
    reading = n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _s35_span(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        blocks = A_SHAPE[1] if kind == "X" else cb
        seeds = []
        for block in range(blocks):
            if kind == "X":
                word = source << (block * cb * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + block) * q + h)
                    bits ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("verified embedded witness failed")
            if _s35_outside(word, own):
                seeds.append(word)
        if not seeds:
            return None

        reduced = []
        for word in seeds:
            for _ in range(2):
                previous = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:48]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == previous:
                    break
            reduced.append(word)
        seed = min(reduced, key=int.bit_count)
        witness = _s35_probe(rng, checks, n, 2, seed, own)
        if not witness:
            return None
        reading = min(reading, witness.bit_count())
    return reading


@lru_cache(maxsize=1)
def _s35_portfolio(G, count):
    """Immutable jointly screened catalogue; memoization changes only cost."""
    if G.q != 35:
        raise ValueError("shared-GF8 portfolio requires order 35")
    aa = _s35_component_bank(G, "A")
    bb = _s35_component_bank(G, "B")
    if not aa or not bb:
        raise ValueError("shared-factor constituent bank exhausted")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            # Shared degree-three loss contributes fifteen. A's
            # augmentation loss contributes two more when present.
            predicted = 85 + 2 * int(a["augmentation"] == 1)
            weak = min(a["reading"], b["reading"])
            priority = (predicted * weak * weak, weak, predicted,
                        -a["penalty"] - b["penalty"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    cap = min(9, 12, W_MAX)
    results, graph_keys = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        if not _no_dead_lines(sa, sb) or max(wx, wz) > cap:
            continue
        if wx > 8 or wz > 9 or _grid_weights(sa, sb) != max(wx, wz):
            raise ValueError("new-shape row arithmetic disagrees")

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 560 or k not in (85, 87):
            continue
        if not (125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = _s35_graph_key(G, hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        if not (_s35_kernel4(hx, n) and _s35_kernel4(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified shared-factor lift failed CSS")

        local = random.Random(
            BASE_SEED ^ (0x3508CC + 65537 * ai + 1009 * bi))
        reading = _s35_logical_reading(G, a, b, hx, hz, n, local)
        if reading is None or reading < 12:
            continue
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, reading, k, preliminary)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), (a["rank"], b["rank"])))
        if len(results) >= count + 24:
            break
        if tested >= max(320, 8 * count):
            break
    results.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an optimized new-shape design."""
    entries = _s35_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("distinct shared-GF8 target-rate portfolio exhausted")
    quality, A, B, k, reading, cards, ranks = entries[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "logical_witness_upper_bound": reading,
        "design_cards": cards, "component_ranks": ranks,
        "shared_spectral_factor_degree": 3,
        "complete_kernel_exclusion_through": 4,
        "selection_quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell of the immutable new-shape portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: The independent-grounded Z11 catalogue supplies twenty inequivalent designs or escapes distance four: its current sixty-slot evaluation retained five distinct codes, all five measured at distance four.
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
