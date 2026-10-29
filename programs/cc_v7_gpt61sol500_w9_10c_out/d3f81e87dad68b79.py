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
CONCEPT = ("Cyclic C20, order 20, A=B=(2,5): replacing three-matching cores by one internal binomial per row and one matching should supply five novel distance-eight readings in cell 0.300, scoring 180*64/580=19.862 if those upper-bound readings survive measurement.")
GROUP = ("Retain the parent's explicitly implemented cyclic construction rule, changing only its order because size factor 29 excludes C33. Order 20 is the largest admissible cyclic lift and gives n=580. The recorded nonabelian order-20 (2,5)x(2,5) bed reached fitness 15.207 with distance-seven readings; the cyclic order-20 bed is absent from BEDS. This motivates a new core mechanism, not a transferred distance claim.")
SHAPES = ("Change to equal (2,5) constituents with 100 variables and 40 checks each. With q=20, n=580, kb=180 and guaranteed rate 9/29=0.310345 target cell 0.300. Require both constituent ranks 40 and actual complete k=180. Parent weight-fourteen minima mainly occupied sector-1 Acol/Bcol blocks, especially Acol1; slot 4 exposed a weight-thirteen mixed-sector operator touching Arow0/Brow0..1, so complete mixed-sector screening remains necessary.")
FILLING = ("Use column degrees (2,2,2,3,3): an upper internal binomial, a lower internal binomial, one matching, and two oppositely oriented cubic reinforcement columns. Every constituent row totals six, so BOTH CSS maxima are nine, satisfying the stricter scaffold cap as well as twelve. Twenty core-step/reinforcement policies receive independent optimization and equal finalist budgets. Verified constituent matrices exactly exclude kernels through seven using syndrome collisions and verified cyclic translation. Explicit cyclic cofactors supply witnesses, never optimistic sentinels. Complete ranks, CSS, kernel-four exclusion, stabilizer descent and mixed-sector logical probes gate an immutable portfolio. Intrinsic rooted uncoloured Tanner invariants conservatively reject repeats. Quantum distance-eight survival, portfolio yield and novelty against the unseen archive remain hypotheses.")
FAMILY_NAME = "r030_c20_25x25_internalcore_kernel7_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("the resized cyclic group requires order twenty")
    # Same cyclic construction rule; factor 29 requires q <= 20.
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    def idx(g):
        return int(g[0]) % q

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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


def _retired_c11_owned_context(rng, G, idx):
    """Uncalled historical C11 reference; never publishes new cells.

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


def _r15_columns(rows, width):
    out = [0] * width
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _r15_clear_four(rows, width):
    """Exact exclusion of every nonzero kernel support of weight <=4."""
    cols = _r15_columns(rows, width)
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
    return True


def _r15_span(rows):
    pivots = {}
    for original in rows:
        row = original
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    return pivots


def _r15_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _r15_rows(G, grid, side):
    """Expose only the evaluator's verified constituent action."""
    unit = [[[G.e]]]
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    mask = (1 << width) - 1
    return [row & mask for row in rows[:len(grid) * G.q]]


def _r15_cofactor(rows, count, q):
    """Actual cyclic kernel witnesses from verified column syndromes."""
    from itertools import combinations
    cols = _r15_columns(rows, count * q)
    mask = (1 << q) - 1
    polys = [[(cols[j*q] >> (i*q)) & mask
              for j in range(count)] for i in range(2)]

    def rotate(a, shift):
        return a if not shift else (
            ((a << shift) | (a >> (q - shift))) & mask)

    def product(a, b):
        result = 0
        while a:
            bit = a & -a
            result ^= rotate(b, bit.bit_length() - 1)
            a ^= bit
        return result

    best = 0
    for js in combinations(range(count), 3):
        pieces = [0] * count
        for omitted, j in enumerate(js):
            a, b = js[:omitted] + js[omitted + 1:]
            pieces[j] = (product(polys[0][a], polys[1][b])
                         ^ product(polys[0][b], polys[1][a]))
        word = sum(poly << (j*q) for j, poly in enumerate(pieces))
        if not word:
            continue
        if any((row & word).bit_count() & 1 for row in rows):
            raise ValueError("verified cyclic cofactor failed")
        if not best or word.bit_count() < best.bit_count():
            best = word
        for shift in range(1, q):
            moved = sum(rotate(poly, shift) << (j*q)
                        for j, poly in enumerate(pieces))
            trial = word ^ moved
            if trial and trial.bit_count() < best.bit_count():
                best = trial
    return best


def _r15_probe(rng, rows, width, passes, seed=0, own=None):
    """Heuristic search returning an actual kernel or logical witness."""
    cols = _r15_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _r15_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for turn in range(passes):
        rng.shuffle(order)
        if own is not None and turn % 2:
            ra, ca = A_SHAPE
            rb, cb = B_SHAPE
            n1 = width * ca * cb // (ca * cb + ra * rb)
            order.sort(key=lambda j: j < n1)
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
                inspect(word)

        rng.shuffle(basis)
        for _ in range(2):
            for i, current in enumerate(basis):
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

        short = sorted(basis, key=int.bit_count)[:22]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 18:
                    for h in range(j):
                        inspect(pair ^ short[h])
        for _ in range(2):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break
        if weight < 8:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _r15_outside(best, own)
    return best


def _r15_draw(rng, G, side, card):
    """Structural cards alter relative voltages, not merely cell anchors."""
    q, els = G.q, G.elements

    def cell(count, extent):
        if not count:
            return []
        anchor = rng.randrange(q)
        if count == 1:
            return [els[anchor]]
        if count == 2:
            gap = rng.randrange(1, extent + 1)
            if rng.randrange(2):
                gap = -gap
            return [els[anchor], els[(anchor + gap) % q]]
        while True:
            u, v = rng.sample(range(1, q), 2)
            gaps = {min(u, q-u), min(v, q-v),
                    min((u-v) % q, (v-u) % q)}
            if len(gaps) == 3:
                return [els[anchor], els[(anchor+u) % q],
                        els[(anchor+v) % q]]

    if side == "A":
        directions = ((1, 2), (1, 3), (1, 4), (1, 5))
        d0, d1 = directions[card // 3]
        policy = card % 3
        while True:
            u, v = rng.sample(range(1, q), 2)
            residues = len({0, u % 3, v % 3})
            if residues == policy + 1:
                break
        bridge = rng.randrange(q)
        return [
            [[], [G.e, els[3*d0]],
             [els[bridge], els[(bridge + 3*d1) % q]]],
            [[G.e, els[u], els[v]], [G.e], [G.e]],
        ]

    profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (3, 2, 1, 0)),
        ((3, 0, 1, 2), (0, 3, 2, 1)),
    )
    profile = profiles[card // 4]
    extent = (4, 8, 12, 16)[card % 4]
    return [[cell(count, extent) for count in row] for row in profile]


def _r15_bank(G, side):
    """Independent constituent optimization with equal finalist budgets."""
    from collections import Counter
    wanted_rank = 63 if side == "A" else 66
    count = 3 if side == "A" else 4
    width = count * G.q
    records, component_keys = [], set()

    for card in range(12):
        local = random.Random(
            BASE_SEED ^ (0xC330150 + 104729*card
                         + (1000003 if side == "B" else 0)))
        proposals, fillings = [], set()
        for trial in range(128):
            grid = _r15_draw(local, G, side, card)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _r15_rows(G, grid, side)
            if gf2_rank(rows) != wanted_rank:
                continue
            if not _r15_clear_four(rows, width):
                continue
            seed = _r15_cofactor(rows, count, G.q)
            if not seed or seed.bit_count() < 12:
                continue
            cycles = sum(
                (overlap * (overlap - 1) // 2)
                for i, row in enumerate(rows)
                for other in rows[:i]
                for overlap in [(row & other).bit_count()])
            proposals.append((seed.bit_count(), cycles, trial,
                              grid, rows, seed))

        proposals.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, cycles, trial, grid, rows, seed in proposals[:20]:
            word = _r15_probe(local, rows, width, 4, seed)
            if word and word.bit_count() >= 12:
                refined.append((word.bit_count(), cycles, trial,
                                grid, rows, word))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))

        finalists = []
        for _, cycles, trial, grid, rows, seed in refined[:6]:
            word = _r15_probe(local, rows, width, 20, seed)
            if word and word.bit_count() >= 12:
                finalists.append((word.bit_count(), cycles, trial,
                                  grid, rows, word))
        finalists.sort(key=lambda r: (-r[0], r[1], r[2]))

        kept = 0
        for reading, cycles, trial, grid, rows, word in finalists:
            cols = _r15_columns(rows, width)
            key = (_tanner_signature(rows, []),
                   tuple(sorted(Counter(c.bit_count() for c in cols).items())))
            if key in component_keys:
                continue
            component_keys.add(key)
            records.append(dict(
                grid=grid, rows=rows, word=word, reading=reading,
                cycles=cycles, rank=wanted_rank, card=card, trial=trial))
            kept += 1
            if kept == 3:
                break

    records.sort(key=lambda r: (
        -r["reading"], r["cycles"], r["card"], r["trial"]))
    return records


def _r15_logical_reading(G, a, b, hx, hz, n, rng):
    """Explicit nontrivial witnesses, including stabilizer descent."""
    q, best = G.q, n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _r15_span(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        blocks = 3 if kind == "X" else 4
        seeds = []
        for block in range(blocks):
            if kind == "X":
                word = source << (block * 4*q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*4 + block)*q + h)
                    bits ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("verified constituent embedding failed")
            if not _r15_outside(word, own):
                continue
            for _ in range(3):
                starting = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting_pairs = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting_pairs).bit_count() >= 2),
                    key=lambda row: -(row & starting_pairs).bit_count())[:40]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting_pairs ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and _r15_outside(word, own)
            seeds.append(word)
        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        word = _r15_probe(rng, checks, n, 2, seed, own)
        if not word:
            return None
        best = min(best, word.bit_count())
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _r15_portfolio(G, count):
    """Immutable deterministic portfolio; no slot-order dependence."""
    from collections import Counter
    if G.q != 33:
        raise ValueError("subgroup-rank portfolio requires C33")
    aa, bb = _r15_bank(G, "A"), _r15_bank(G, "B")
    if not aa or not bb:
        raise ValueError("screened subgroup-rank constituent bank exhausted")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            priority = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                -a["cycles"] - b["cycles"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    results, graph_keys = [], set()
    tested = high = 0
    cap = min(9, W_MAX)
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert wx <= 8 and wz <= 9
        if max(wx, wz) > cap or not _no_dead_lines(sa, sb):
            continue
        assert _grid_weights(sa, sb) == max(wx, wz)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 528 or k != 72:
            continue
        if not (125*n < 1000*k < 175*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        cols = _r15_columns(hx + hz, n)
        signature = (
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(Counter(r.bit_count() for r in hx + hz).items())),
            _tanner_signature(hx, hz))
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1

        # Complete-matrix gates include words crossing the LP sectors.
        if not (_r15_clear_four(hx, n) and _r15_clear_four(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified C33 lift failed CSS")
        local = random.Random(
            BASE_SEED ^ (0xC33CC150 + 65537*ai + 1009*bi))
        reading = _r15_logical_reading(G, a, b, hx, hz, n, local)
        if reading is not None and reading >= 12:
            screen = (
                ("k", k),
                ("logical_witness_upper_bound", reading),
                ("component_witness_readings", (a["reading"], b["reading"])),
                ("component_ranks", (a["rank"], b["rank"])),
                ("cards", (a["card"], b["card"])),
                ("complete_kernel_exclusion_through", 4),
            )
            quality = (k*reading*reading/n, reading, preliminary, -ai, -bi)
            results.append((quality, freeze(A), freeze(B), screen))
            high += int(reading >= 14)
        if len(results) >= count and high >= min(count, 20):
            break
        if tested >= max(240, 8*count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def _r30_translate(row, width, q):
    """Simultaneous blockwise translation, tested on verified matrices."""
    mask = (1 << q) - 1
    moved = 0
    for block in range(width // q):
        value = (row >> (block * q)) & mask
        value = ((value << 1) | (value >> (q - 1))) & mask
        moved |= value << (block * q)
    return moved


def _r30_key(hx, hz, width, q):
    """Intrinsic rooted invariant of the uncoloured Tanner graph.

    Equal keys are conservatively discarded. Translation reduces the
    root calculations only after it is verified on the actual rows.
    No protograph labels, group coordinates or CSS check colours enter
    the recorded graph features.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _r15_columns(rows, width)
    m = len(rows)

    def bits(word):
        result = []
        while word:
            bit = word & -word
            result.append(bit.bit_length() - 1)
            word ^= bit
        return result

    adjacency = [[m + j for j in bits(row)] for row in rows]
    adjacency += [bits(col) for col in cols]
    highest = max(col.bit_count() for col in cols)
    roots = [j for j, col in enumerate(cols)
             if col.bit_count() == highest]
    symmetric = Counter(
        _r30_translate(row, width, q) for row in rows) == Counter(rows)
    if symmetric:
        roots = [j for j in roots if j % q == 0]

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    initial = [(0, len(ns)) for ns in adjacency[:m]]
    initial += [(1, len(ns)) for ns in adjacency[m:]]
    rooted = []
    for root in roots:
        colors = compress([
            (tag, int(v == m + root))
            for v, tag in enumerate(initial)])
        history = []
        for _ in range(8):
            features = [
                (colors[v], tuple(sorted(colors[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            colors = compress(features)
        digest = sha256(repr(history).encode("ascii")).digest()
        rooted.extend([digest] * (q if symmetric else 1))

    return (
        width, m,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        tuple(sorted(rooted)),
    )


def _r30_clean_seven(rows, width, q):
    """Exactly exclude every nonzero constituent kernel word through seven.

    Collisions among supports of sizes <=3 detect all weights <=6.
    A seven-word must contain an odd number of degree-three columns.
    Verified cyclic translation moves one such variable to an identity
    coordinate, so anchored quartets versus all triples cover weight 7.
    """
    from itertools import combinations
    from collections import Counter

    cols = _r15_columns(rows, width)
    degrees = [col.bit_count() for col in cols]
    if any(degree not in (2, 3) for degree in degrees):
        raise ValueError("kernel-seven screen requires column degrees 2/3")
    if Counter(_r30_translate(row, width, q) for row in rows) != Counter(rows):
        raise ValueError("constituent translation was not verified")
    if any(len(set(degrees[j:j + q])) != 1
           for j in range(0, width, q)):
        raise ValueError("constituent degree orbits changed")

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
    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)

    anchors = [j for j in range(0, width, q) if degrees[j] == 3]
    for anchor in anchors:
        others = [j for j in range(width) if j != anchor]
        first = cols[anchor]
        for a, b, c in combinations(others, 3):
            if (first ^ cols[a] ^ cols[b] ^ cols[c]) in seen:
                return False
    return True


def _r30_draw(local, G, card):
    """Twenty normalized core/reinforcement policies, not output draws."""
    q, els = G.q, G.elements
    core_class, gap_policy = divmod(card, 5)
    units = (1, 3, 7, 9)
    tens = (2, 6)
    upper_pool = tens if core_class in (2, 3) else units
    lower_pool = tens if core_class in (1, 3) else units
    a, b = local.choice(upper_pool), local.choice(lower_pool)
    if local.randrange(2):
        a = (-a) % q
    if local.randrange(2):
        b = (-b) % q

    extent = (3, 5, 7, 9, 19)[gap_policy]
    c, d = local.randrange(1, extent + 1), local.randrange(1, extent + 1)
    if local.randrange(2):
        c = (-c) % q
    if local.randrange(2):
        d = (-d) % q
    u, v = local.randrange(q), local.randrange(q)

    # Column and row gauges fix the core matching and singleton anchors.
    # Genuine relative gaps and reinforcement placements remain variable.
    C = [
        [[G.e, els[a]], [], [G.e], [G.e, els[c]], [G.e]],
        [[], [G.e, els[b]], [G.e], [els[u]],
         [els[v], els[(v + d) % q]]],
    ]
    assert [sum(map(len, row)) for row in C] == [6, 6]
    assert [sum(len(C[i][j]) for i in range(2))
            for j in range(5)] == [2, 2, 2, 3, 3]
    assert all(len(set(cell)) == len(cell) <= 5
               for row in C for cell in row)
    return C


def _r30_bank(G, side):
    """Optimize each of twenty prescriptions before selecting constituents."""
    width = 5 * G.q
    bank, component_keys = [], set()

    for card in range(20):
        raw, filling_keys = [], set()
        for trial in range(192):
            local = random.Random(
                BASE_SEED ^ (0x203007 + 104729 * card + 7919 * trial
                             + 1000003 * int(side == "B")))
            C = _r30_draw(local, G, card)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            rows = _r15_rows(G, C, side)
            if gf2_rank(rows) != 2 * G.q:
                continue
            if not _r15_clear_four(rows, width):
                continue
            seed = _r15_cofactor(rows, 5, G.q)
            if not seed or seed.bit_count() < 8:
                continue
            cycles = sum(
                overlap * (overlap - 1) // 2
                for i, row in enumerate(rows)
                for other in rows[:i]
                for overlap in [(row & other).bit_count()])
            raw.append((cycles, trial, C, rows, seed))

        raw.sort(key=lambda record: record[:2])
        refined = []
        for cycles, trial, C, rows, seed in raw[:24]:
            local = random.Random(
                BASE_SEED ^ (0x2030A7 + 65537 * card + 257 * trial
                             + 1000003 * int(side == "B")))
            word = _r15_probe(local, rows, width, 4, seed)
            if word and word.bit_count() >= 8:
                refined.append((cycles, trial, C, rows, word))
        refined.sort(key=lambda record: record[:2])

        kept = 0
        for cycles, trial, C, rows, seed in refined[:12]:
            if not _r30_clean_seven(rows, width, G.q):
                continue
            local = random.Random(
                BASE_SEED ^ (0x2030B7 + 65537 * card + 257 * trial
                             + 1000003 * int(side == "B")))
            word = _r15_probe(local, rows, width, 20, seed)
            if not word or word.bit_count() < 8:
                raise ValueError("constituent witness contradicts exact screen")
            key = _r30_key(rows, [], width, G.q)
            if key in component_keys:
                continue
            component_keys.add(key)
            bank.append(dict(
                grid=C, word=word, reading=word.bit_count(),
                rank=2 * G.q, cycles=cycles, card=card, trial=trial,
                exact_kernel_exclusion_through=7))
            kept += 1
            if kept >= 4:
                break

    bank.sort(key=lambda record: (
        -min(record["reading"], 8), record["cycles"],
        record["card"], record["trial"]))
    return bank


def _r30_pair_reading(G, a, b, hx, hz, n, local):
    """Explicit nontrivial embedded and mixed-sector logical witnesses."""
    q = G.q
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    reading = n + 1
    for kind, checks, stabilizers, record in (
            ("X", hz, hx, b), ("Z", hx, hz, a)):
        own = _r15_span(stabilizers)
        seeds = []
        for block in range(ca if kind == "X" else cb):
            source = record["word"]
            if kind == "X":
                word = source << (block * cb * q)
            else:
                word = 0
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + block) * q + h)
                    source ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("verified new-shape embedding failed")
            if not _r15_outside(word, own):
                continue

            for _ in range(3):
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
                    key=lambda row: -(row & starting).bit_count())[:32]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if previous == word:
                    break
            assert word and _r15_outside(word, own)
            seeds.append(word)

        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        word = _r15_probe(local, checks, n, 6, seed, own)
        if not word:
            return None
        reading = min(reading, word.bit_count())
    return reading


@lru_cache(maxsize=1)
def _r30_portfolio(G, count):
    """Immutable deterministic portfolio; no recycling on exhaustion."""
    assert G.q == 20 and A_SHAPE == B_SHAPE == (2, 5)
    aa, bb = _r30_bank(G, "A"), _r30_bank(G, "B")
    if not aa or not bb:
        raise ValueError("kernel-seven-clean cyclic constituent bank exhausted")

    local = random.Random(BASE_SEED ^ 0x2030CA7)
    plans = [(ai, bi) for ai in range(len(aa)) for bi in range(len(bb))]
    local.shuffle(plans)
    plans.sort(key=lambda pair: (
        aa[pair[0]]["cycles"] + bb[pair[1]]["cycles"],
        aa[pair[0]]["card"] == bb[pair[1]]["card"],
        -min(aa[pair[0]]["reading"], bb[pair[1]]["reading"], 8)))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    results, graph_keys = [], set()
    cap = min(9, W_MAX)
    tested = 0
    for ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(5))
        assert wx == wz == 9
        assert _grid_weights(sa, sb) == 9
        if max(wx, wz) > cap or not _no_dead_lines(sa, sb):
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if n != 580 or k != 180:
            continue
        assert 275 * n <= 1000 * k < 325 * n
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if not (_r15_clear_four(hx, n) and _r15_clear_four(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified internal-core lift failed CSS")

        signature = _r30_key(hx, hz, n, G.q)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        probe_rng = random.Random(
            BASE_SEED ^ (0x2030CC7 + 65537 * ai + 1009 * bi))
        reading = _r30_pair_reading(G, a, b, hx, hz, n, probe_rng)
        if reading is None or reading < 8:
            continue
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")

        # The internal core provides an explicit constituent ceiling.
        # Failed mixed-sector probes cannot earn an optimistic sentinel.
        capped = min(reading, a["reading"], b["reading"], 8)
        quality = (k * capped * capped / n,
                   -a["cycles"] - b["cycles"], -ai, -bi)
        screen = (
            ("k", k),
            ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("component_ranks", (a["rank"], b["rank"])),
            ("design_cards", (a["card"], b["card"])),
            ("exact_constituent_kernel_exclusion_through", 7),
            ("exact_complete_kernel_exclusion_through", 4),
            ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + max(20, count // 3):
            break
        if tested >= max(640, 16 * count) and len(results) >= count:
            break

    results.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete target-cell internal-core design."""
    portfolio = _r30_portfolio(G, N_SLOTS)
    if not 0 <= idx < len(portfolio):
        raise ValueError("fresh C20 internal-core portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Publish the complete selected cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: Independent grounded constituents and intrinsic ownership bins supply twenty inequivalent designs: the current sixty-slot evaluation retained only five distinct codes, all with measured distance four.
# REFUTED: The C33 subgroup-rank cubic portfolio supplies twenty inequivalent initial designs: the measured current program retained only nine distinct codes across sixty slots.
# REFUTED: Every C33 subgroup-rank finalist preserves its distance-fourteen screen reading: measured slot 4 fell to thirteen and exposed mixed-sector X logicals.
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
