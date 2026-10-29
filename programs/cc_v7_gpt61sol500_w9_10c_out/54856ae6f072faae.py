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
CONCEPT = ("Dic5, order 20, A=B=(2,5): cubic reinforcements of verified girth-eight matching cores, screened exactly through constituent weight seven, should supply five measured distance-eight codes in cell 0.300; at k=180 these score 19.862, a falsifiable measurement hypothesis.")
GROUP = ("Jump from Z11 to nonisomorphic Dic5. Order-ten rotations and order-four outside elements allow noncommuting matching cores without involutory edges. Related order-20 beds measured fitness 11.2 and 15.2; these motivate stronger screening, not a group-distance guarantee. Order 20 is the largest lift admitted by size factor 29.")
SHAPES = ("Equal (2,5) constituents have 100 variables and 40 checks, n=580, kb=180 and guaranteed rate 9/29=0.310345 in cell 0.300. Verified full ranks retain [0.275,0.325). Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly; both new constituents receive independent reinforcement optimization.")
FILLING = ("Twenty cards combine split/isolated cubic reinforcements with five relative-element policies. Each row contains six terms and each column two or three, giving BOTH CSS maxima nine. All graph data and syndromes come from verified lifts. Exact two/four/six-cubic branches screen every constituent kernel through seven; girth-eight core words supply explicit witnesses. Prefer certified constituent distance eight, with screened weaker fallbacks rather than duplicate padding. Full kernel-four exclusion, ranks, CSS and embedded-witness stabilizer descent gate selection. Different intrinsic full-Tanner fingerprints separate an immutable portfolio; mixed-sector distance and unseen archive novelty remain measurement questions.")
FAMILY_NAME = "r030_dic5_25x25_exact_core8_cubic_join_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic5 has order twenty")
    # x^10=1, y^2=x^5, y*x*y^-1=x^-1; (a,b)=x^a y^b.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        a, b = g
        return ((int(a) + 5*(int(b)//2)) % 10, int(b) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 5*b*d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a + 5) % 10, b)

    def idx(g):
        a, b = red(g)
        return 10*b + a

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


def _retired_z11_owned_context(rng, G, idx):
    """Inactive Z11-specific reference; never fills the new group or shapes.

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


def _d20_columns(rows, width):
    """Column syndromes of the exact scaffold construction."""
    result = [0]*width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            result[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return result


def _d20_component(G, grid, side):
    unit = [[[G.e]]]
    width = len(grid[0])*G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    mask = (1 << width)-1
    return [row & mask for row in rows[:len(grid)*G.q]]


def _d20_core_data(rows, width):
    """Exact graph girth, all-pairs distances and a shortest cycle word."""
    columns = _d20_columns(rows, width)
    adjacency = [[] for _ in rows]
    edges = set()
    for edge, syndrome in enumerate(columns):
        if syndrome.bit_count() != 2:
            raise ValueError("verified matching core is not an incidence graph")
        bit = syndrome & -syndrome
        u = bit.bit_length()-1
        v = (syndrome ^ bit).bit_length()-1
        key = tuple(sorted((u, v)))
        if key in edges:
            return 2, (), 0
        edges.add(key)
        adjacency[u].append((v, edge))
        adjacency[v].append((u, edge))

    girth, witness = width+1, 0
    distances = []
    for root in range(len(rows)):
        distance = [10000]*len(rows)
        paths = [0]*len(rows)
        distance[root] = 0
        queue = [root]
        for u in queue:
            for v, edge in adjacency[u]:
                if distance[v] == 10000:
                    distance[v] = distance[u]+1
                    paths[v] = paths[u] ^ (1 << edge)
                    queue.append(v)
                else:
                    word = paths[u] ^ paths[v] ^ (1 << edge)
                    weight = word.bit_count()
                    if word and weight < girth:
                        girth, witness = weight, word
        distances.append(tuple(distance))
    if witness:
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
    return girth, tuple(distances), witness


@lru_cache(maxsize=1)
def _d20_cores(G):
    """Immutable noncommuting core bank; both actions use verified lifts."""
    def order(g):
        value = G.e
        for t in range(1, G.q+1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid group order")

    orders = {g: order(g) for g in G.elements}
    choices = [g for g in G.elements if orders[g] >= 4]
    representatives = [
        g for g in choices
        if G.idx(g) == min(
            G.idx(G.mul(G.mul(a, g), G.inv(a)))
            for a in G.elements)]
    records = []
    for g in representatives:
        for h in choices:
            if g == h or G.mul(g, h) == G.mul(h, g):
                continue
            core = [
                [[G.e], [G.e], [G.e]],
                [[G.e], [g], [h]],
            ]
            left = _d20_component(G, core, "A")
            right = _d20_component(G, core, "B")
            da = _d20_core_data(left, 3*G.q)
            db = _d20_core_data(right, 3*G.q)
            if min(da[0], db[0]) >= 8 and da[2] and db[2]:
                records.append((g, h, da, db))
    if not records:
        raise ValueError("no verified Dic5 matching core of girth eight")
    best = max(min(r[2][0], r[3][0]) for r in records)
    return tuple(r for r in records if min(r[2][0], r[3][0]) == best)


def _d20_small_distance(cols, distances, girth, q, deep):
    """Exact constituent kernel screen through seven when deep=True.

    Columns 0/1 have degree three; columns 2/3/4 are graph edges.
    A kernel contains an even number of cubic variables. Through
    weight seven its possibilities are therefore 0, 2, 4 or 6 cubics.
    T-join costs are exact minimum matchings in the graph metric.
    Regular-action translates justify identity-coordinate anchors;
    no alternative lift convention is used.
    """
    from itertools import combinations

    cubic = [i for i, c in enumerate(cols) if c.bit_count() == 3]
    graph = [i for i, c in enumerate(cols) if c.bit_count() == 2]
    assert cubic == list(range(2*q))
    assert graph == list(range(2*q, 5*q))
    anchors = (0, q)
    best = min(8, girth)
    memo = {}

    def join(boundary, budget):
        if not boundary:
            return 0
        if budget < 0 or boundary.bit_count() > 2*budget:
            return budget+1
        key = (boundary, budget)
        if key in memo:
            return memo[key]
        bit = boundary & -boundary
        u = bit.bit_length()-1
        rest = boundary ^ bit
        choices = rest
        value = budget+1
        while choices:
            other = choices & -choices
            v = other.bit_length()-1
            distance = distances[u][v]
            if distance <= budget:
                value = min(
                    value,
                    distance + join(rest ^ other, budget-distance))
            choices ^= other
        memo[key] = value
        return value

    for anchor in anchors:
        for other in cubic:
            if other != anchor:
                best = min(
                    best, 2 + join(cols[anchor] ^ cols[other], best-3))
    if not deep or best < 6:
        return best

    for anchor in anchors:
        others = [i for i in cubic if i != anchor]
        for a, b, c in combinations(others, 3):
            boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
            if boundary.bit_count() <= 2*(best-5):
                best = min(best, 4 + join(boundary, best-5))
        if best < 6:
            return best

    # Six cubic variables with zero graph edges: triple collisions.
    # With one graph edge: compare anchored cubic triples to all triples.
    # If a collision has overlapping triples, it is a smaller kernel
    # word already covered by the two/four-cubic branches.
    if best > 6:
        triples = {}
        for a, b, c in combinations(cubic, 3):
            syndrome = cols[a] ^ cols[b] ^ cols[c]
            word = (1 << a) | (1 << b) | (1 << c)
            old = triples.get(syndrome)
            if old is not None:
                best = min(best, (old ^ word).bit_count())
            else:
                triples[syndrome] = word
        if best > 7:
            for anchor in anchors:
                others = [i for i in cubic if i != anchor]
                for a, b in combinations(others, 2):
                    syndrome = cols[anchor] ^ cols[a] ^ cols[b]
                    word = (1 << anchor) | (1 << a) | (1 << b)
                    for edge in graph:
                        old = triples.get(syndrome ^ cols[edge])
                        if old is not None:
                            weight = (old ^ word ^ (1 << edge)).bit_count()
                            best = min(best, weight)
                if best <= 7:
                    break
    return best


def _d20_bank(rng, G, side, profile, policy):
    """Optimize a deliberate reinforcement card, not an emitted draw."""
    from itertools import combinations

    q = G.q
    orders = {}
    for g in G.elements:
        value = G.e
        for t in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = t
                break
    permitted = ((10,), (5,), (4,), (4, 10), (4, 5, 10))[policy]
    ratios = [g for g in G.elements if orders[g] in permitted]
    all_ratios = [g for g in G.elements if orders[g] >= 4]
    triangles = []
    for r in ratios:
        for s in all_ratios:
            if r == s or orders[G.mul(G.inv(r), s)] < 4:
                continue
            commute = G.mul(r, s) == G.mul(s, r)
            if policy == 0 and not commute:
                continue
            if policy in (1, 2) and commute:
                continue
            triangles.append((r, s))
    if not triangles:
        triangles = [(r, s) for r in ratios for s in all_ratios if r != s]

    cores = _d20_cores(G)
    raw, seen = [], set()
    for trial in range(160):
        g, h, da, db = rng.choice(cores)
        data = da if side == "A" else db
        grid = [
            [[], [], [G.e], [G.e], [G.e]],
            [[], [], [G.e], [g], [h]],
        ]
        counts = (
            ((2, 1), (1, 2)),
            ((3, 0), (0, 3)),
        )[profile]
        for i in range(2):
            for j in range(2):
                count = counts[i][j]
                if not count:
                    continue
                first = G.pick(rng)
                if count == 1:
                    cell = [first]
                elif count == 2:
                    cell = [first, G.mul(first, rng.choice(ratios))]
                else:
                    r, s = rng.choice(triangles)
                    cell = [first, G.mul(first, r), G.mul(first, s)]
                assert len(set(cell)) == count
                grid[i][j] = cell

        filling = tuple(tuple(sorted(G.idx(x) for x in cell))
                        for row in grid for cell in row)
        if filling in seen:
            continue
        seen.add(filling)
        rows = _d20_component(G, grid, side)
        cols = _d20_columns(rows, 5*q)
        cheap = _d20_small_distance(cols, data[1], data[0], q, False)
        if cheap < 6:
            continue
        c4 = sum(
            ((a & b).bit_count()*((a & b).bit_count()-1))//2
            for i, a in enumerate(cols) for b in cols[:i])
        raw.append(((-cheap, c4, trial), grid, rows, cols, data))

    raw.sort(key=lambda r: r[0])
    finalists = []
    for cheap, grid, rows, cols, data in raw[:20]:
        floor = _d20_small_distance(cols, data[1], data[0], q, True)
        if floor < 6:
            continue
        witness = data[2] << (2*q)
        assert witness
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
        finalists.append({
            "grid": grid, "floor": floor, "word": witness,
            "upper": witness.bit_count(), "cost": cheap[1],
        })
    finalists.sort(
        key=lambda r: (r["floor"], -r["cost"]), reverse=True)
    return finalists[:4]


def _d20_clear_four(rows, width):
    """Exact exclusion of every kernel support of weight at most four."""
    cols = _d20_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, first in enumerate(cols):
        for other in cols[:i]:
            syndrome = first ^ other
            if syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _d20_signature(hx, hz, width):
    """Intrinsic full-Tanner invariants, deliberately without CSS colours."""
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    cols = _d20_columns(rows, width)
    high = [c for c in cols if c.bit_count() == 12]
    incident = [[] for _ in rows]
    for v, column in enumerate(high):
        while column:
            bit = column & -column
            incident[bit.bit_length()-1].append(v)
            column ^= bit

    adjacency = [Counter() for _ in high]
    for endpoints in incident:
        for i, u in enumerate(endpoints):
            for v in endpoints[:i]:
                adjacency[u][v] += 1
                adjacency[v][u] += 1
    size = len(high)
    power = [[int(u == v) for v in range(size)] for u in range(size)]
    diagonal = [[] for _ in high]
    two = None
    for length in range(1, 9):
        following = [[0]*size for _ in high]
        for u, neighbors in enumerate(adjacency):
            target = following[u]
            for v, multiplicity in neighbors.items():
                for w, value in enumerate(power[v]):
                    target[w] += multiplicity*value
        power = following
        if length == 2:
            two = power
        for u in range(size):
            diagonal[u].append(power[u][u])
    walks = tuple(sorted(
        (tuple(diagonal[u]), tuple(sorted(two[u])))
        for u in range(size)))
    intrinsic = (
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        _tanner_signature(hx, hz), walks)
    return sha256(repr(intrinsic).encode("ascii")).digest()


def _d20_quantum_reading(G, a, b, hx, hz, width):
    """Explicit nontrivial embedded witnesses and stabilizer descent.

    This is an upper reading. Constituent certificates do not certify
    the absence of lighter mixed-sector quantum logical operators.
    """
    q = G.q
    best = width+1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        span = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = span.get(p)
                if old is None:
                    span[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = span.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        seeds = []
        for block in range(5):
            if kind == "X":
                word = b["word"] << (block*5*q)
            else:
                word, source = 0, a["word"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*5+block)*q+h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if nontrivial(word):
                seeds.append(word)
        for word in sorted(seeds, key=int.bit_count)[:3]:
            weight = word.bit_count()
            for _ in range(2):
                starting = word
                for row in stabilizers:
                    trial = starting ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                moves = [row for row in stabilizers
                         if (row & starting).bit_count() >= 2]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and nontrivial(word)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            best = min(best, weight)
    return best


def _d20_design(rng, G, idx, excluded):
    """Select a novel full code from independently optimized constituents."""
    card = idx % 20
    pa, pb = (card//5)//2, (card//5) % 2
    policy = card % 5
    aa = _d20_bank(rng, G, "A", pa, policy)
    bb = _d20_bank(rng, G, "B", pb, (policy+2) % 5)
    if not aa or not bb:
        return None
    proposals = sorted(
        ((min(a["floor"], b["floor"]), -a["cost"]-b["cost"], i, j)
         for i, a in enumerate(aa) for j, b in enumerate(bb)),
        reverse=True)
    winner, quality = None, None
    tested = 0
    for floor, _, i, j in proposals:
        a, b = aa[i], bb[j]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        assert all(sum(row) == 6 for row in sa+sb)
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9 <= min(9, W_MAX)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 580 and k > 0 and 275*n <= 1000*k < 325*n):
            continue
        if not (_d20_clear_four(hx, n) and _d20_clear_four(hz, n)):
            continue
        signature = _d20_signature(hx, hz, n)
        if signature in excluded:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified Dic5 lift failed CSS")
        reading = _d20_quantum_reading(G, a, b, hx, hz, n)
        effective = min(reading, a["upper"], b["upper"])
        candidate_quality = (
            min(floor, effective), k*effective*effective/float(n),
            -a["cost"]-b["cost"], signature)
        if quality is None or candidate_quality > quality:
            quality = candidate_quality
            winner = (A, B, k, a["floor"], b["floor"],
                      effective, signature)
        tested += 1
        if tested >= 5:
            break
    return winner


@lru_cache(maxsize=1)
def _d20_portfolio(G):
    """Immutable indexed portfolio; call order cannot change its designs."""
    entries, excluded = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for idx in range(N_SLOTS):
        entry = None
        for restart in range(3):
            local = random.Random(
                BASE_SEED*10007 + 0xD205258
                + 65537*idx + 1000003*restart)
            result = _d20_design(local, G, idx, excluded)
            if result is None:
                continue
            A, B, k, fa, fb, reading, signature = result
            excluded.add(signature)
            entry = (freeze(A), freeze(B), k, fa, fb, reading)
            break
        entries.append(entry)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed lookup of a deliberate, verified new-group design."""
    entry = _d20_portfolio(G)[idx]
    if entry is None:
        raise ValueError("novel Dic5 reinforced-core card exhausted")
    A, B, k, fa, fb, reading = entry
    return {
        "A": A, "B": B, "k": k, "card": idx % 20,
        "constituent_exact_floors": (fa, fb),
        "embedded_quantum_upper": reading,
    }


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
# REFUTED: Intrinsic ownership bins produce twenty evaluator-distinct Z11 designs: the current sixty-slot evaluation retained only five classes, all with measured distance four.
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
