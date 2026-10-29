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
CONCEPT = ("Dihedral C5 semidirect_-1 C2, order 10, A=B=(3,7): a length-fifteen matching core reinforced by four cubic columns, with exact constituent join screening through seven, should yield five novel distance-eight readings in cell 0.300.")
GROUP = ("Jump from cyclic Z8 to the genuinely nonabelian dihedral group of order ten. The recorded Z5:Z2(s=4), (3,7)x(3,7) bed reached fitness 13.5 with distance-seven readings. This motivates a stronger dependency screen, not an assumed group-distance guarantee.")
SHAPES = ("Equal components have 70 variables and 30 checks. Factor 58 permits q<=10; q=10 gives n=580, kb=160 and guaranteed rate 16/58=0.275862 in cell 0.300. The parent's five distance-six minima occupied sector-1 Acol/Bcol blocks broadly. Replace its pendant/eight-column geometry with a longer three-edge core and four cubic reinforcement blocks.")
FILLING = ("Twenty support/quotient prescriptions use column degrees 2,2,2,3,3,3,3 and six terms per row, so BOTH CSS weight maxima are nine, also below twelve. Verified constituent matrices exclude kernels through five and count sixes; exact graph joins then cover every remaining support through seven with zero/two/four/six cubic variables. Floor-eight constituents precede measured-witness-guided fallbacks. Immutable publication rejects equal full uncoloured Tanner invariants, verifies actual ranks and CSS, excludes complete kernels through four and probes explicit nontrivial logicals. Constituent floors do not certify mixed-sector quantum distance; unseen archive novelty and distance-eight survival remain hypotheses.")
FAMILY_NAME = "r030_dihedral5_q10_37x37_cycle15_join7_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect_-1 C2 requires order ten")
    # x^5=y^2=1, y*x*y^-1=x^-1; (a,b)=x^a*y^b.
    els = [(a, b) for b in range(2) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c)) % 5, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a if b else -a) % 5, b)

    def idx(g):
        a, b = red(g)
        return 5*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _unused_order11_reference(rng, G, idx):
    """Uncalled historical catalogue; never publishes a candidate.

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


def _unused_order8_multisupport_context(rng, G, idx):
    """Pure reconstruction of a screened multisupport catalogue.

    Different intrinsic invariants certify different Tanner graphs.
    Reserved positions prevent retries from recycling another slot.
    """
    from itertools import permutations
    from collections import Counter

    if G.q != 8:
        raise ValueError("multisupport catalogue requires order eight")
    local = random.Random(BASE_SEED ^ 0x386704)
    q, width, cap = G.q, 8 * G.q, min(9, W_MAX)
    mask = (1 << width) - 1
    dummy = [[[G.e]]]

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def pair_table(cols):
        """Exactly exclude kernel weights one through four."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs[syndrome] = (j, i)
        return pairs

    def six_count(rows):
        """Exclude weights <=5, then count every weight-six word."""
        cols = columns(rows, width)
        pairs = pair_table(cols)
        if pairs is None:
            return None
        triples = {}
        collisions = 0
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                for h in range(j):
                    value = syndrome ^ cols[h]
                    if value in pairs:
                        return None
                    previous = triples.get(value, 0)
                    collisions += previous
                    triples[value] = previous + 1
        # Once weights <=5 are excluded, colliding triples are disjoint.
        # Each six-word has ten unordered partitions into two triples.
        assert collisions % 10 == 0
        return collisions // 10

    def support_key(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(8)))
            for order in permutations(range(3)))

    seed_profile = (
        (1, 1, 0, 0, 0, 2, 2, 0),
        (0, 1, 1, 1, 1, 0, 0, 2),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )
    profiles = [seed_profile]
    support_keys = {support_key(seed_profile)}
    double_types = sorted(set(permutations((1, 1, 0)))
                          | set(permutations((2, 0, 0))))
    triple_types = sorted(set(permutations((2, 1, 0)))
                          | {(1, 1, 1)}
                          | set(permutations((3, 0, 0))))
    # Count profiles are deliberate designs, not voltage orbit copies.
    for attempt in range(32768):
        cols = [(1, 0, 0)]
        cols += [local.choice(double_types) for _ in range(4)]
        cols += [local.choice(triple_types) for _ in range(3)]
        profile = tuple(tuple(c[i] for c in cols) for i in range(3))
        if any(sum(row) != 6 for row in profile):
            continue
        key = support_key(profile)
        if key in support_keys:
            continue
        support_keys.add(key)
        profiles.append(profile)
        if len(profiles) >= 18:
            break

    def draw(profile, seeded):
        C = [[[] for _ in range(8)] for _ in range(3)]
        if seeded:
            # Preserve the successful matching core, but change genuine
            # relative binomial gaps and endpoint voltages.
            C[0][0] = [G.e]
            C[0][1] = [G.e]
            C[1][1] = [G.e]
            for j, voltage in enumerate((0, 1, 3), 2):
                C[1][j] = [G.e]
                C[2][j] = [G.elements[voltage]]
            for j, row in ((5, 0), (6, 0), (7, 1)):
                C[row][j] = [G.e, G.elements[local.randrange(1, q)]]
                C[2][j] = [G.elements[local.randrange(q)]]
            return C
        for j in range(8):
            first = True
            for i in range(3):
                count = profile[i][j]
                if not count:
                    continue
                labels = local.sample(range(q), count)
                if first:
                    # A column gauge fixes one term, not relative labels.
                    anchor = labels[0]
                    labels = [(v - anchor) % q for v in labels]
                    first = False
                C[i][j] = [G.elements[v] for v in sorted(labels)]
        return C

    bank, fill_keys, component_keys = [], set(), set()
    for attempt in range(12288):
        # Half the proposals exploit a measured support; the remainder
        # explore different intrinsic hyperedge incidence structures.
        pi = 0 if attempt % 2 == 0 else (
            1 + (attempt // 2) % max(1, len(profiles) - 1))
        pi %= len(profiles)
        C = draw(profiles[pi], pi == 0)
        filling = tuple(tuple(G.idx(g) for g in cell)
                        for row in C for cell in row)
        if filling in fill_keys:
            continue
        fill_keys.add(filling)
        counts = [[len(cell) for cell in row] for row in C]
        if any(sum(row) != 6 for row in counts):
            raise ValueError("constituent row budget changed")
        if not _no_dead_lines(counts, counts):
            continue

        left, _, _ = lifted_matrices(G, C, dummy)
        _, right, _ = lifted_matrices(G, dummy, C)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if gf2_rank(left) != 24 or gf2_rank(right) != 24:
            continue
        nl = six_count(left)
        if nl is None:
            continue
        nr = six_count(right)
        if nr is None:
            continue
        signature = (_tanner_signature(left, []),
                     tuple(sorted((nl, nr))))
        if signature in component_keys:
            continue
        component_keys.add(signature)
        bank.append((max(nl, nr), nl + nr, filling, C, pi))
        if len(bank) >= 32:
            break
    if len(bank) < 6:
        raise ValueError("too few kernel-five-clean multisupport constituents")
    bank.sort(key=lambda entry: entry[:3])

    plans = sorted(
        (int(bank[a][0] != 0 or bank[b][0] != 0),
         max(bank[a][0], bank[b][0]),
         bank[a][1] + bank[b][1], a, b)
        for a in range(len(bank)) for b in range(a, len(bank)))
    wanted = {idx + retry * N_SLOTS for retry in range(4)}
    graph_keys = set()
    accepted = 0
    n1 = 64 * q

    def full_key(hx, hz, n):
        cols = columns(hx + hz, n)
        return (
            tuple(sorted(Counter(r.bit_count()
                                 for r in hx + hz).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            _tanner_signature(hx, hz),
        )

    def mixed_five(cols, pairs):
        # Pure sector-1 words decompose into constituent kernel words,
        # already excluded through five. Every remaining five-word has
        # a triple partition containing a sector-2 qubit.
        for i in range(n1, len(cols)):
            first = cols[i]
            for j in range(i):
                syndrome = first ^ cols[j]
                for h in range(j):
                    if (syndrome ^ cols[h]) in pairs:
                        return True
        return False

    for _, _, _, a, b in plans:
        A, B = bank[a][3], bank[b][3]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if _grid_weights(sa, sb) > cap or not _no_dead_lines(sa, sb):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (325*n <= 1000*k < 375*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = full_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        position = accepted
        accepted += 1
        if position not in wanted:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified multisupport lift failed CSS")

        legal = True
        for checks in (hx, hz):
            cols = columns(checks, n)
            pairs = pair_table(cols)
            if pairs is None or mixed_five(cols, pairs):
                legal = False
                break
        if legal:
            return {
                "A": A, "B": B, "k": k,
                "catalogue_position": position,
                "component_six_counts": (bank[a][0], bank[b][0]),
                "support_cards": (bank[a][4], bank[b][4]),
                "verified_kernel_floor": 6,
            }
    raise ValueError("reserved multisupport designs failed complete screens")


from functools import lru_cache


def _d10_columns(rows, width):
    """Syndromes extracted solely from verified scaffold matrices."""
    cols = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _d10_rows(G, grid, right=False):
    """The zero dummy is a screening device, never an emitted side."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    return [row & mask for row in rows], width


def _d10_pairs(cols):
    """Exact exclusion of all nonzero kernel supports through four."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = {}
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs[syndrome] = (1 << i) | (1 << j)
    return pairs


def _d10_floor(rows, width):
    """Exclude through five; count sixes and retain an actual witness."""
    cols = _d10_columns(rows, width)
    pairs = _d10_pairs(cols)
    if pairs is None:
        return None
    triples, collisions, witness = {}, 0, 0
    for i, first in enumerate(cols):
        for j in range(i):
            partial = first ^ cols[j]
            for h in range(j):
                syndrome = partial ^ cols[h]
                if syndrome in pairs:
                    return None
                word = (1 << i) | (1 << j) | (1 << h)
                previous = triples.get(syndrome)
                if previous is None:
                    triples[syndrome] = (1, word)
                else:
                    collisions += previous[0]
                    if not witness:
                        witness = word ^ previous[1]
                    triples[syndrome] = (previous[0] + 1, previous[1])
    # Smaller words are absent, so matching triples are disjoint.
    assert collisions % 10 == 0
    if witness:
        assert witness.bit_count() == 6
    return (6 if collisions else 7), collisions // 10, witness


def _d10_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _d10_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _d10_probe(rows, width, rng, passes, seed=0, own=None):
    """Return an explicit kernel/logical witness, never a lower bound."""
    cols = _d10_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if own is not None and not _d10_outside(word, own):
            return
        best, weight = word, word.bit_count()

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
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
        for i, word in enumerate(kernel):
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        short = sorted(kernel, key=int.bit_count)[:24]
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
        if own is not None and weight <= 6:
            break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _d10_outside(best, own)
    return best


def _d10_join7(G, rows):
    """Exact constituent exclusion through seven, after floor >=7.

    Core variables have degree two; forty reinforcement variables have
    degree three. A kernel has an even number of cubic variables.
    Regular-action translations let each nonempty cubic support contain
    an identity-coordinate variable in one of its four blocks.
    """
    from itertools import combinations

    q, width = G.q, 7 * G.q
    cols = _d10_columns(rows, width)
    adjacency = [[] for _ in rows]
    for edge, syndrome in enumerate(cols[:3*q]):
        assert syndrome.bit_count() == 2
        bit = syndrome & -syndrome
        a = bit.bit_length() - 1
        b = (syndrome ^ bit).bit_length() - 1
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))

    distances, cycle = [], 0
    for root in range(len(rows)):
        distance = [10000] * len(rows)
        path = [0] * len(rows)
        distance[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if distance[b] == 10000:
                    distance[b] = distance[a] + 1
                    path[b] = path[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = path[a] ^ path[b] ^ (1 << edge)
                    if word and (not cycle or
                                 word.bit_count() < cycle.bit_count()):
                        cycle = word
        distances.append(distance)
    assert cycle and cycle.bit_count() == 15
    cubic = list(range(3*q, width))
    assert all(cols[j].bit_count() == 3 for j in cubic)
    anchors = [j*q + G.idx(G.e) for j in range(3, 7)]
    memo = {}

    def join(boundary, budget):
        if not boundary:
            return 0
        if budget < 0 or boundary.bit_count() > 2 * budget:
            return budget + 1
        key = (boundary, budget)
        if key in memo:
            return memo[key]
        bit = boundary & -boundary
        a = bit.bit_length() - 1
        rest = boundary ^ bit
        choices, best = rest, budget + 1
        while choices:
            other = choices & -choices
            b = other.bit_length() - 1
            d = distances[a][b]
            if d <= budget:
                tail = join(rest ^ other, budget - d)
                if tail <= budget - d:
                    best = min(best, d + tail)
            choices ^= other
        memo[key] = best
        return best

    # Two cubic variables plus at most five graph edges.
    for anchor in anchors:
        for other in cubic:
            if other != anchor:
                if join(cols[anchor] ^ cols[other], 5) <= 5:
                    return 7, cycle

    # Four cubic variables plus at most three graph edges.
    for anchor in anchors:
        others = [j for j in cubic if j != anchor]
        for a, b, c in combinations(others, 3):
            boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
            if boundary.bit_count() <= 6 and join(boundary, 3) <= 3:
                return 7, cycle

    # Six cubic variables plus one graph edge. Pure cubic dependencies
    # through six have already been excluded by the syndrome screen.
    triples = {}
    for a, b, c in combinations(cubic, 3):
        syndrome = cols[a] ^ cols[b] ^ cols[c]
        assert syndrome not in triples
        triples[syndrome] = (1 << a) | (1 << b) | (1 << c)
    for anchor in anchors:
        others = [j for j in cubic if j != anchor]
        for a, b in combinations(others, 2):
            syndrome = cols[anchor] ^ cols[a] ^ cols[b]
            word = (1 << anchor) | (1 << a) | (1 << b)
            for edge, edge_syndrome in enumerate(cols[:3*q]):
                previous = triples.get(syndrome ^ edge_syndrome)
                if previous is not None:
                    witness = word ^ previous ^ (1 << edge)
                    if witness and witness.bit_count() <= 7:
                        assert witness.bit_count() == 7
                        return 7, witness
    # Zero-cubic words are core cycles; eight cubic variables cannot
    # occur in a support of weight at most seven.
    return 8, cycle


def _d10_key(hx, hz, width):
    """Conservative invariant of the full uncoloured Tanner graph.

    Published check degrees are 8/9 and qubit degrees are 4/5/6/12,
    so arbitrary uncoloured isomorphisms preserve the two vertex classes.
    Equal keys are rejected; equality is not an isomorphism certificate.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    cols = _d10_columns(rows, width)
    return (
        width,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        _tanner_signature(list(hx), list(hz)),
    )


@lru_cache(maxsize=1)
def _d10_portfolio(G, count):
    """Deterministic optimization; cached publication data are immutable."""
    from itertools import combinations_with_replacement, permutations

    assert G.q == 10
    q, e, width = G.q, G.e, 7 * G.q
    elements = {tuple(g): g for g in G.elements}
    rotations = [elements[(a, 0)] for a in range(1, 5)]
    cosets = [[elements[(a, b)] for a in range(5)] for b in range(2)]
    types = sorted(
        {(1, 1, 1)}
        | set(permutations((2, 1, 0)))
        | set(permutations((3, 0, 0))))
    canonical = set()
    for columns in combinations_with_replacement(types, 4):
        if any(sum(c[i] for c in columns) != 4 for i in range(3)):
            continue
        key = min(
            tuple(sorted(tuple(c[i] for i in order) for c in columns))
            for order in permutations(range(3)))
        canonical.add(key)
    plain = ((1, 1, 1),) * 4
    profiles = [plain] + sorted(canonical - {plain})

    def invert_coordinates(word):
        result = 0
        while word:
            bit = word & -word
            j, h = divmod(bit.bit_length() - 1, q)
            result |= 1 << (j*q + G.idx(G.inv(G.elements[h])))
            word ^= bit
        return result

    def draw(local, profile, policy, trial):
        C = [[[] for _ in range(7)] for _ in range(3)]
        C[0][0], C[1][0] = [e], [e]
        C[1][1], C[2][1] = [e], [e]
        C[2][2] = [e]
        C[0][2] = [elements[(1 + (trial // 8) % 2, 0)]]
        palette = ((0, 0), (0, 1), (1, 0), (1, 1))
        for j, counts in enumerate(profile, 3):
            first = True
            prescribed = palette[(policy + j) % 4]
            for i, terms in enumerate(counts):
                if not terms:
                    continue
                if first:
                    anchor = e
                    first = False
                else:
                    parity = prescribed[(i - 1) % 2]
                    anchor = local.choice(cosets[parity])
                if terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    cell = [anchor, G.mul(anchor, local.choice(rotations))]
                elif policy == 0:
                    relative = local.sample(rotations, 2)
                    cell = [anchor] + [G.mul(anchor, g) for g in relative]
                else:
                    relative = local.sample(
                        [g for g in G.elements if g != e], 2)
                    cell = [anchor] + [G.mul(anchor, g) for g in relative]
                assert len(set(cell)) == terms
                C[i][j] = cell
        assert all(sum(map(len, row)) == 6 for row in C)
        return C

    bank, component_keys = [], set()
    for card in range(20):
        profile = profiles[(card // 4) % len(profiles)]
        policy = card % 4
        local = random.Random(BASE_SEED ^ (0xD103707 + 104729 * card))
        raw, fillings = [], set()
        for trial in range(128):
            C = draw(local, profile, policy, trial)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows, _ = _d10_rows(G, C)
            if gf2_rank(rows) != 3*q:
                continue
            cols = _d10_columns(rows, width)
            if _d10_pairs(cols) is None:
                continue
            cycles = sum(
                (row & other).bit_count()
                * ((row & other).bit_count() - 1) // 2
                for i, row in enumerate(rows) for other in rows[:i])
            raw.append((cycles, trial, C, rows))
        raw.sort(key=lambda r: r[:2])
        refined = []
        for cycles, trial, C, rows in raw[:24]:
            screened = _d10_floor(rows, width)
            if screened is None:
                continue
            floor, sixes, seed = screened
            if floor == 7:
                floor, seed = _d10_join7(G, rows)
            word = _d10_probe(rows, width, local, 8, seed)
            if not word:
                continue
            assert word.bit_count() >= floor
            right, _ = _d10_rows(G, C, True)
            # Verify the inversion transfer against both actual actions.
            assert sorted(invert_coordinates(r) for r in rows) == sorted(right)
            right_word = invert_coordinates(word)
            assert all((r & right_word).bit_count() % 2 == 0 for r in right)
            refined.append((
                floor, min(word.bit_count(), 10), sixes, cycles,
                trial, C, word, right_word, rows))
        refined.sort(key=lambda r: (-r[0], -r[1], r[2], r[3], r[4]))
        kept = 0
        for record in refined:
            signature = _d10_key(record[8], [], width)
            if signature in component_keys:
                continue
            component_keys.add(signature)
            bank.append((record, card))
            kept += 1
            if kept >= 4:
                break
    if len(bank) < 5:
        raise ValueError("too few independently screened dihedral constituents")

    plans = []
    for ai, (a, acard) in enumerate(bank):
        for bi in range(ai, len(bank)):
            b, bcard = bank[bi]
            priority = (
                min(a[0], b[0]), min(a[1], b[1]),
                -a[2]-b[2], -a[3]-b[3], int(acard != bcard))
            plans.append((priority, ai, bi))
    plans.sort(key=lambda r: (r[0], -r[1], -r[2]), reverse=True)

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(7):
            if kind == "X":
                word = record[7] << (block * width)
            else:
                word, source = 0, record[6]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*7 + block)*q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _d10_outside(word, own):
                continue
            for _ in range(2):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == previous:
                    break
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, graph_keys = [], set()
    tested = 0
    for priority, ai, bi in plans:
        a, acard = bank[ai]
        b, bcard = bank[bi]
        A, B = a[5], b[5]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= min(9, 12, W_MAX)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (275*n <= 1000*k < 325*n):
            continue
        if (_d10_pairs(_d10_columns(hx, n)) is None
                or _d10_pairs(_d10_columns(hz, n)) is None):
            continue
        signature = _d10_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        if not css_ok(hx, hz):
            raise ValueError("verified dihedral lift failed CSS")
        own_x, own_z = _d10_span(hx), _d10_span(hz)
        xseed = embedded(b, "X", hz, hx, own_x)
        zseed = embedded(a, "Z", hx, hz, own_z)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0xD10C00 + 257*ai + bi))
        xword = _d10_probe(hz, n, local, 4, xseed, own_x)
        zword = _d10_probe(hx, n, local, 4, zseed, own_z)
        reading = min(xword.bit_count(), zword.bit_count())
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        assert lifted_k(G, A, B) == k

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * min(reading, 10)**2 / n, priority, reading)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (acard, bcard), (a[0], b[0]), (a[2], b[2])))
        if len(results) >= count + max(10, count // 3):
            break
        if tested >= max(240, 10 * count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure publication of one immutable, fully screened structural design."""
    portfolio = _d10_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct dihedral join-screened portfolio exhausted")
    quality, A, B, k, reading, cards, floors, sixes = portfolio[idx]
    return dict(
        A=A, B=B, screen_k=k, quality=quality,
        logical_witness_upper_bound=reading,
        design_cards=cards, constituent_kernel_floors=floors,
        constituent_six_word_counts=sixes)


def CELL(rng, G, side, i, j, ctx):
    """Copy the complete cell without mutating the cached portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: Independent grounded constituents and intrinsic ownership bins supply twenty inequivalent designs: the current program retained only five distinct codes across sixty slots, all with measured distance four.
# REFUTED: Multisupport Z8 constituent-six prioritization produces distance-seven finalists: all five measured current finalists instead had k=200 and d30s=6.
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
