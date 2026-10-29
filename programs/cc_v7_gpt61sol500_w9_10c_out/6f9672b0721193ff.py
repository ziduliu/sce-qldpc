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
CONCEPT = ("Generalized quaternion Q16, order 16, A=B=(2,5): maximum-girth matching cores reinforced by cubic columns should supply five novel distance-eight readings in cell 0.300; this is a falsifiable measurement hypothesis.")
GROUP = ("Jump from cyclic C11 to Q16, with x^8=1, y^2=x^4 and y*x*y^-1=x^-1. These groups are genuinely nonisomorphic. Q16 has order-four/eight directions, permitting matching-core cycles longer than involutions allow. Related quaternion beds measured stronger smaller-rate portfolios than their wide incidence variants; this motivates constituent optimization, not a transferred distance guarantee. Order sixteen gives n=464.")
SHAPES = ("Change to equal 80-variable, 32-check constituents. The size factor is 29, n=464, kb=144 and guaranteed rate 9/29=0.310345. Verified full ranks retain [0.275,0.325). Aim at the relatively weak 0.300 frontier cell instead of the parent's 0.500 cell. Parent weight-four minima broadly occupied sector-1 Acol/Bcol blocks; replacing seven-column incidence constituents by reinforced five-column constituents changes their mechanism.")
FILLING = ("Twenty reinforcement cards cover two supports with row totals six and column totals two/three, giving BOTH CSS maxima nine, also below twelve. Enumerate matching cores through the verified helper and retain maximum exact graph girth. Optimize each card using exact constituent exclusion through four, exact five/six spectra and equally budgeted kernel-witness probes. Pair constituents using verified full ranks, CSS, complete kernel-four rejection and actual non-stabilizer logical probes. Different intrinsic full Tanner fingerprints separate published entries without ownership bins or padding. Lower-reading fallbacks prevent missing-finalist zeros; unknown archive novelty and survival of distance-eight readings remain hypotheses.")
FAMILY_NAME = "r030_quaternion16_25x25_core_cubic_spectrum6_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q16 requires order sixteen")
    # x^8=1, y^2=x^4, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 4*(b // 2)) % 8, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 4*b*d) % 8,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 8, 0) if b == 0 else ((a + 4) % 8, 1)

    def idx(g):
        a, b = red(g)
        return 8*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


def _retired_z11_context(rng, G, idx):
    """Uncalled historical C11 filling; never publishes the new shapes.

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


def _q16_columns(rows, width):
    """Syndromes from verified matrices, never a private lift."""
    columns = [0]*width
    mask = (1 << width)-1
    for r, original in enumerate(rows):
        row = original & mask
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return columns


def _q16_clear4(rows, width):
    """Exactly exclude every nonzero kernel support of weight <=4."""
    cols = _q16_columns(rows, width)
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


def _q16_spectrum(rows, width):
    """After clear4: find weight five, or count every weight-six word.

    The returned floor is exact only through the examined weights.
    A floor of seven does not assert that a weight-seven word exists.
    """
    cols = _q16_columns(rows, width)
    if not _q16_clear4(rows, width):
        return None
    singles = {s: 1 << i for i, s in enumerate(cols)}
    pairs = {}
    for i, first in enumerate(cols):
        for j in range(i):
            pairs[first ^ cols[j]] = (1 << i) | (1 << j)
    triples, collisions, witness = {}, 0, 0
    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            word = (1 << i) | (1 << j)
            for h in range(j):
                syndrome = pair ^ cols[h]
                triple = word | (1 << h)
                old = pairs.get(syndrome)
                if old is not None:
                    candidate = triple ^ old
                    if candidate:
                        return candidate.bit_count(), 1, candidate
                if syndrome == 0 or syndrome in singles:
                    raise ValueError("clear4 and spectrum disagree")
                old = triples.get(syndrome)
                if old is None:
                    triples[syndrome] = (triple, 1)
                else:
                    collisions += old[1]
                    triples[syndrome] = (old[0], old[1]+1)
                    if not witness:
                        witness = triple ^ old[0]
    if collisions:
        if collisions % 10 or witness.bit_count() != 6:
            raise ValueError("inconsistent verified six-word spectrum")
        return 6, collisions//10, witness
    return 7, 0, 0


def _q16_span(rows):
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


def _q16_outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _q16_probe(rows, width, local, passes, stabilizers=None, seeds=()):
    """Carry actual kernel/logical witnesses; high readings are upper bounds."""
    cols = _q16_columns(rows, width)
    own = _q16_span(stabilizers) if stabilizers is not None else None
    best, weight = 0, width+1

    def record(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _q16_outside(word, own):
            return
        best, weight = word, w

    for seed in seeds:
        if seed and all((seed & row).bit_count() % 2 == 0 for row in rows):
            record(seed)

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
        pivots, basis = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                record(word)

        local.shuffle(basis)
        for i, word in enumerate(basis):
            w = word.bit_count()
            for j, other in enumerate(basis):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            basis[i] = word
            record(word)
        for i, first in enumerate(basis):
            for second in basis[:i]:
                record(first ^ second)
        short = sorted(set(basis), key=int.bit_count)[:20]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    record(pair ^ short[h])

        if stabilizers is not None and best:
            for descent in range(2):
                starting = best
                for i, first in enumerate(stabilizers):
                    record(starting ^ first)
                    for second in stabilizers[:i]:
                        record(starting ^ first ^ second)
                if best == starting:
                    break
        if weight <= (5 if stabilizers is not None else 6):
            break

    if not best:
        raise ValueError("verified witness probe found no admissible word")
    if any((best & row).bit_count() % 2 for row in rows):
        raise ValueError("kernel probe produced a noncommuting word")
    if own is not None and not _q16_outside(best, own):
        raise ValueError("logical probe produced a stabilizer")
    return weight, best


def _q16_graph_key(hx, hz, n):
    """Intrinsic complete uncoloured Tanner invariants.

    Unequal keys certify inequivalence; equal keys are rejected
    conservatively. No group labels or CSS colours enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx)+list(hz)
    cols = _q16_columns(rows, n)
    adjacency = [[] for _ in range(n+len(rows))]
    for j, col in enumerate(cols):
        while col:
            bit = col & -col
            r = bit.bit_length()-1
            adjacency[j].append(n+r)
            adjacency[n+r].append(j)
            col ^= bit
    initial = [(int(v >= n), len(neighbors))
               for v, neighbors in enumerate(adjacency)]
    palette = {f: i for i, f in enumerate(sorted(set(initial)))}
    colors = [palette[f] for f in initial]
    history = [tuple(sorted(Counter(initial).items()))]
    for _ in range(6):
        features = [
            (colors[v], tuple(sorted(colors[w] for w in neighbors)))
            for v, neighbors in enumerate(adjacency)]
        history.append(tuple(sorted(Counter(features).items())))
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        following = [palette[f] for f in features]
        if len(set(following)) == len(set(colors)):
            break
        colors = following
    invariant = (
        n, len(rows),
        tuple(history),
        _tanner_signature(hx, hz),
        tuple(sorted(Counter(
            (rows[i] & rows[j]).bit_count()
            for i in range(len(rows)) for j in range(i)).items())))
    return sha256(repr(invariant).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q16_portfolio(G, count):
    """Deterministic jointly screened portfolio; cached grids are immutable."""
    from itertools import combinations_with_replacement

    if G.q != 16:
        raise ValueError("Q16 portfolio requires order sixteen")
    q, width, cap = G.q, 5*G.q, min(9, W_MAX)
    if cap < 9:
        raise ValueError("reinforced constituents require check weight nine")
    e, unit = G.e, [[[G.e]]]
    mask = (1 << width)-1

    def component(grid, right=False):
        if right:
            _, rows, _ = lifted_matrices(G, unit, grid)
        else:
            rows, _, _ = lifted_matrices(G, grid, unit)
        return [row & mask for row in rows[:2*q]]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def order(g):
        value = e
        for t in range(1, q+1):
            value = G.mul(value, g)
            if value == e:
                return t
        raise ValueError("invalid quaternion element order")

    # Exhaustive normalized three-matching cores. All adjacency comes
    # from the exact right-action matrix used for B by the evaluator.
    cores = []
    for g in G.elements:
        if g == e:
            continue
        for h in G.elements:
            if h == e or h == g:
                continue
            core = [[[e], [e], [e]], [[e], [g], [h]]]
            _, lifted, _ = lifted_matrices(G, unit, core)
            cmask = (1 << (3*q))-1
            rows = [r & cmask for r in lifted[:2*q]]
            cols = _q16_columns(rows, 3*q)
            adjacency = [[] for _ in rows]
            edges = set()
            legal = True
            for edge, col in enumerate(cols):
                if col.bit_count() != 2:
                    legal = False
                    break
                bit = col & -col
                u = bit.bit_length()-1
                v = (col ^ bit).bit_length()-1
                key = min(u, v), max(u, v)
                if key in edges:
                    legal = False
                    break
                edges.add(key)
                adjacency[u].append((v, edge))
                adjacency[v].append((u, edge))
            if not legal:
                continue
            girth = 2*q+1
            for root in range(2*q):
                distance, parent_edge = [-1]*(2*q), [-1]*(2*q)
                distance[root] = 0
                queue = [root]
                for u in queue:
                    for v, edge in adjacency[u]:
                        if distance[v] < 0:
                            distance[v] = distance[u]+1
                            parent_edge[v] = edge
                            queue.append(v)
                        elif edge != parent_edge[u] and edge != parent_edge[v]:
                            girth = min(girth, distance[u]+distance[v]+1)
            cores.append((girth, g, h))
    if not cores:
        raise ValueError("no simple verified quaternion matching core")
    best_girth = max(c[0] for c in cores)
    cores = sorted(c for c in cores if c[0] == best_girth)

    # Four direction prescriptions. Outside parity is a proposal feature,
    # not an inequivalence certificate; full graph invariants decide that.
    pools = [
        [g for g in G.elements if g[1] == 0 and order(g) == 4],
        [g for g in G.elements if g[1] == 0 and order(g) == 8],
        [g for g in G.elements if g[1] == 1 and g[0] % 2 == 0],
        [g for g in G.elements if g[1] == 1 and g[0] % 2 == 1],
    ]
    policies = list(combinations_with_replacement(range(4), 2))
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 1, 1, 1, 0), (0, 1, 1, 1, 3)),
    )
    bank, filling_keys = [], set()
    for card in range(20):
        local = random.Random(BASE_SEED ^ (0x161625 + 104729*card))
        profile = card//10
        aclass, bclass = policies[card % 10]
        counts = profiles[profile]
        assert _no_dead_lines(counts, counts)
        assert _grid_weights(counts, counts) == 9
        raw = []

        for trial in range(144):
            _, g, h = local.choice(cores)
            s = local.choice(pools[aclass])
            t = local.choice(pools[bclass])
            if profile == 0:
                grid = [
                    [[e, s], [G.pick(local)], [e], [e], [e]],
                    [[G.pick(local)], [e, t], [e], [g], [h]],
                ]
            else:
                u = local.choice([z for z in G.elements if z not in (e, s)])
                v = local.choice([z for z in G.elements if z not in (e, t)])
                grid = [
                    [[e, s, u], [e], [e], [e], []],
                    [[], [e], [g], [h], [e, t, v]],
                ]
            filling = tuple(tuple(sorted(G.idx(z) for z in cell))
                            for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            rows = component(grid)
            if not _q16_clear4(rows, width):
                continue
            cols = _q16_columns(rows, width)
            squares = sum(
                ((a & b).bit_count()*((a & b).bit_count()-1))//2
                for i, a in enumerate(cols) for b in cols[:i])
            raw.append((squares, trial, grid, rows))
        raw.sort(key=lambda item: item[:2])

        finalists = []
        for squares, trial, grid, rows in raw[:24]:
            spectrum = _q16_spectrum(rows, width)
            if spectrum is None:
                continue
            floor, multiplicity, seed = spectrum
            reading, witness = _q16_probe(
                rows, width, local, 4, seeds=(seed,))
            finalists.append((
                -reading, -floor, multiplicity, squares, trial,
                grid, rows, witness))
        finalists.sort(key=lambda item: item[:5])

        # Equal refinement budgets for every retained constituent.
        for candidate in finalists[:2]:
            _, neg_floor, multiplicity, squares, trial, grid, rows, seed = candidate
            reading, witness = _q16_probe(
                rows, width, local, 16, seeds=(seed,))
            right_rows = component(grid, True)
            inverse_word, bits = 0, witness
            while bits:
                bit = bits & -bits
                j, position = divmod(bit.bit_length()-1, q)
                inverse_word |= 1 << (
                    j*q + G.idx(G.inv(G.elements[position])))
                bits ^= bit
            if any((r & inverse_word).bit_count() % 2 for r in right_rows):
                raise ValueError("verified inversion transfer failed")
            bank.append(dict(
                grid=freeze(grid), reading=reading,
                left_word=witness, right_word=inverse_word,
                floor=-neg_floor, multiplicity=multiplicity,
                card=card, trial=trial, squares=squares))
    if len(bank) < 5:
        raise ValueError("too few optimized quaternion constituents")
    bank.sort(key=lambda r: (
        -r["reading"], -r["floor"], r["multiplicity"],
        r["squares"], r["card"], r["trial"]))

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"]-b["reading"],
         a["multiplicity"]+b["multiplicity"], ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))
    results, graph_keys = [], set()
    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 464 or not (275*n <= 1000*k < 325*n):
            continue
        if max(r.bit_count() for r in hx+hz) > cap:
            continue
        if not (_q16_clear4(hx, n) and _q16_clear4(hz, n)):
            continue
        signature = _q16_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quaternion lift failed CSS")

        xseeds = [b["right_word"] << (j*5*q) for j in range(5)]
        zseeds = []
        for beta in range(5):
            word, bits = 0, a["left_word"]
            while bits:
                bit = bits & -bits
                j, position = divmod(bit.bit_length()-1, q)
                word |= 1 << ((j*5+beta)*q+position)
                bits ^= bit
            zseeds.append(word)
        local = random.Random(
            BASE_SEED ^ (0x16CC30 + 1009*ai + 65537*bi))
        dx, xword = _q16_probe(
            hz, n, local, 2, stabilizers=hx, seeds=xseeds)
        dz, zword = _q16_probe(
            hx, n, local, 2, stabilizers=hz, seeds=zseeds)
        reading = min(dx, dz)
        graph_keys.add(signature)
        quality = k*reading*reading/float(n)
        results.append((
            quality, A, B, k, reading, (a["card"], b["card"]),
            (a["floor"], b["floor"]),
            (xword, zword), signature.hex()))
        if len(results) >= max(count, 5)+12:
            break
    results.sort(key=lambda item: (-item[0], item[-1]))
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication; never pad with another slot's design."""
    portfolio = _q16_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(portfolio)):
        raise ValueError("distinct quaternion reinforcement portfolio exhausted")
    quality, A, B, k, reading, cards, floors, words, signature = portfolio[idx]
    return {
        "A": A, "B": B, "k": k, "selection_score": quality,
        "logical_witness_upper_bound": reading,
        "reinforcement_cards": cards, "constituent_floors": floors,
        "verified_logical_words": words, "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell from the screened quaternion design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: Independent grounded C11 constituents and intrinsic ownership bins supply twenty evaluator-distinct designs or escape weight four: the current sixty-slot evaluation retained five classes, with all five measured distances equal to four.
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
