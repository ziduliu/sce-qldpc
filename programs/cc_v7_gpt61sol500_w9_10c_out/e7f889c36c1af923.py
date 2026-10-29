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
CONCEPT = ("C5 x C2, order 10, A=B=(2,7): jointly varied triangle-free incidence constituents should supply five novel distance-four codes in cell 0.500, replacing the parent's triangle-bearing split B.")
GROUP = ("Jump from the cyclic order-seven group to the direct-product construction C5 x C2. Different orders prove these groups are not relabelings. The order-ten group admits seven-matching graphs with three omitted matchings and internal cycles of lengths five or ten. Its recorded Z10 (2,7)x(2,7) bed scored 4.6; exact new-matrix screening, not an assumed group advantage, is the hypothesis.")
SHAPES = ("Equal (2,7) constituents have 70 variables and 20 checks. Factor 53 allows q<=11; q=10 gives n=530, kb=250 and guaranteed rate 25/53=0.471698. Connected regular pairs are expected near k=262, rate 0.494340; grounded alternatives are admitted only by actual ranks in [0.475,0.525). Parent weight-three minima occupied sector-1 Acol/Bcol blocks broadly. Remove its three-row, triangle-bearing K7 B component.")
FILLING = ("Enumerate regular matching/internal-cycle cards and grounded boundary cards, retaining relative deck arrangements until joint screening. Regular rows total seven with column totals two; boundary rows total at most seven with column totals one or two. BOTH LP maxima are therefore at most nine, also below twelve. Verified constituent and complete matrices reject every kernel support through three. Explicit commuting, non-stabilizer weight-four witnesses certify distance four for admitted codes. Unequal intrinsic full-Tanner intersection/contraction invariants separate published entries. Regular pairs precede grounded fallbacks; catalogue yield and unseen-archive novelty remain measured hypotheses.")
FAMILY_NAME = "r050_c5xc2_q10_27x27_jointdeck_certified_d4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 x C2 requires order ten")
    els = [(a, b) for b in range(2) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + c) % 5, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 5, (-b) % 2)

    def idx(g):
        a, b = red(g)
        return 5*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _voltage_canon(values, q):
    """Remove independent translations and component-row reversals."""
    return min(
        tuple(sorted((sign * x + shift) % q for x in values))
        for sign in (1, -1) for shift in range(q)
    )


def _component_image(desc, scale, q):
    kind, step, values = desc
    values = _voltage_canon([(scale * x) % q for x in values], q)
    if kind:
        step = (scale * step) % q
        step = min(step, q - step)
    return kind, step, values


def _design_catalogue(q):
    """Finite designs, not random draws; quotient explicit orbit copies."""
    from itertools import combinations

    dense = sorted({
        (0, 0, _voltage_canon(values, q))
        for values in combinations(range(q), 7)
    })

    # Each internal column is an 11-cycle. Cross-voltage differences
    # exclude +/-step, so no cross edge completes an internal triangle.
    rings = sorted({
        (1, step, _voltage_canon(
            [(2 * j * step) % q for j in range(5)], q))
        for step in range(1, (q + 1) // 2)
    })
    components = dense + rings
    images = {
        c: tuple(_component_image(c, u, q) for u in range(1, q))
        for c in components
    }
    affine_family = {c: min(images[c]) for c in components}

    pairs = set()
    for ia, a in enumerate(components):
        for b in components[ia:]:
            # Shared group automorphism; independent component gauges;
            # equal shapes also permit exchange of the two sides.
            key = min(
                tuple(sorted((images[a][u], images[b][u])))
                for u in range(q - 1)
            )
            pairs.add(key)

    # Cover intrinsic component-family pairs before spending slots on
    # different relative orientations of the same two families.
    buckets = {}
    for a, b in sorted(pairs):
        family = tuple(sorted((affine_family[a], affine_family[b])))
        buckets.setdefault(family, []).append((a, b))
    keys = sorted(buckets)
    catalogue = []
    for depth in range(max(len(v) for v in buckets.values())):
        for key in keys:
            if depth < len(buckets[key]):
                catalogue.append(buckets[key][depth])
    return catalogue


def _component_cells(G, desc):
    kind, step, values = desc
    e = G.e
    element = {(g[0], g[1]): g for g in G.elements}

    def x(t):
        return element[(t % G.q, 0)]

    if not kind:
        return [
            [[e] for _ in values],
            [[x(t)] for t in values],
        ]
    return [
        [[e, x(step)], []] + [[e] for _ in values],
        [[], [e, x(step)]] + [[x(t)] for t in values],
    ]


def _no_kernel_up_to_three(rows, n):
    """Exact test on the verified lift, not a distance-search proxy."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    signatures = set(columns)
    if 0 in signatures or len(signatures) != n:
        return False
    # With nonzero, distinct columns, a+b=c necessarily uses three
    # distinct qubits. This tests every weight-three kernel vector.
    for i, a in enumerate(columns):
        for j in range(i):
            if (a ^ columns[j]) in signatures:
                return False
    return True


def _three_orbit_canon(desc, q):
    """Quotient common scaling, vertex-layer gauges and equal-degree swap."""
    ds, s01, s02, s12 = desc
    images = []
    orientations = (
        (ds, s01, s02, s12),
        ((ds[1], ds[0], ds[2]),
         tuple((-v) % q for v in s01), s12, s02),
    )
    for steps, a, b, c in orientations:
        for scale in range(1, q):
            dd = tuple(min(scale*d % q, (-scale*d) % q)
                       if d else 0 for d in steps)
            for anchor_a in a:
                for anchor_b in b:
                    aa = tuple(sorted(scale*(v-anchor_a) % q for v in a))
                    bb = tuple(sorted(scale*(v-anchor_b) % q for v in b))
                    cc = tuple(sorted(scale*(v+anchor_a-anchor_b) % q
                                      for v in c))
                    images.append((dd, aa, bb, cc))
    return min(images)


def _three_orbit_cards(q):
    """Triangle-free cards first; subsequent cards change intrinsic graphs."""
    from itertools import combinations, product

    def anchored(count):
        return [(0,) + tail
                for tail in combinations(range(1, q), count - 1)]

    def compatible(labels, d0, d1):
        forbidden = {d % q for d in (d0, -d0, d1, -d1) if d}
        return all((a-b) % q not in forbidden
                   for i, a in enumerate(labels) for b in labels[:i])

    priority = set()
    # Row degrees are (7,7,6). With one internal edge orbit on
    # rows 0 and 1, row 2 may have either zero or one.
    for d0, d1, d2 in product(range(1, 4), range(1, 4), range(4)):
        t2 = int(d2 != 0)
        c01, c02, c12 = 2 + t2, 3 - t2, 3 - t2
        for a in anchored(c01):
            if not compatible(a, d0, d1):
                continue
            for b in anchored(c02):
                if not compatible(b, d0, d2):
                    continue
                forbidden = {(v-u) % q for u in a for v in b}
                for c in combinations(range(q), c12):
                    if forbidden.intersection(c):
                        continue
                    if not compatible(c, d1, d2):
                        continue
                    desc = ((d0, d1, d2), a, b, tuple(c))
                    priority.add(_three_orbit_canon(desc, q))

    for desc in sorted(priority):
        yield desc

    # Step and matching-set mutations retain the same legal counts.
    # These are new graph candidates, not translated saved fillings.
    mutations = set()
    for desc in sorted(priority):
        ds, a, b, c = desc
        for i, old in enumerate(ds):
            if not old:
                continue
            for replacement in range(1, 4):
                changed = list(ds)
                changed[i] = replacement
                mutations.add(_three_orbit_canon(
                    (tuple(changed), a, b, c), q))
        for part in range(1, 4):
            values = desc[part]
            for position in range(len(values)):
                for replacement in range(q):
                    changed = list(values)
                    changed[position] = replacement
                    if len(set(changed)) != len(changed):
                        continue
                    card = list(desc)
                    card[part] = tuple(sorted(changed))
                    mutations.add(_three_orbit_canon(tuple(card), q))
    for desc in sorted(mutations - priority):
        yield desc

    # Additional deliberate three-partite support cards: no internal
    # columns and matching multiplicities (4,3,3), still degrees (7,7,6).
    for a in anchored(4):
        for b in anchored(3):
            for c in combinations(range(q), 3):
                yield ((0, 0, 0), a, b, tuple(c))


def _incidence_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _three_orbit_info(G, B):
    """All graph information comes from a projected VERIFIED scaffold lift."""
    _, hz, _ = lifted_matrices(G, [[[G.e]]], B)
    width = B_SHAPE[1] * G.q
    mask = (1 << width) - 1
    rows = [row & mask for row in hz]
    columns = _incidence_columns(rows, width)
    if (any(c.bit_count() != 2 for c in columns)
            or len(set(columns)) != width):
        return None

    nv = len(rows)
    adj = [0] * nv
    for column in columns:
        low = column & -column
        u = low.bit_length() - 1
        v = (column ^ low).bit_length() - 1
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    reached, frontier = 1, 1
    while frontier:
        following = 0
        while frontier:
            bit = frontier & -frontier
            following |= adj[bit.bit_length() - 1]
            frontier ^= bit
        frontier = following & ~reached
        reached |= frontier
    if reached != (1 << nv) - 1:
        return None

    neighbours = [[v for v in range(nv) if (adj[u] >> v) & 1]
                  for u in range(nv)]
    power = [[(adj[u] >> v) & 1 for v in range(nv)]
             for u in range(nv)]
    diagonal = [[] for _ in range(nv)]
    triangles = 0
    for length in range(2, 9):
        power = [[sum(power[w][v] for w in neighbours[u])
                  for v in range(nv)] for u in range(nv)]
        for u in range(nv):
            diagonal[u].append(power[u][u])
        if length == 3:
            triangles = sum(power[u][u] for u in range(nv)) // 6
    profiles = []
    for u in range(nv):
        pair_profile = tuple(sorted(
            (len(neighbours[v]), (adj[u] >> v) & 1,
             (adj[u] & adj[v]).bit_count())
            for v in range(nv)))
        profiles.append((len(neighbours[u]),
                         tuple(diagonal[u]), pair_profile))
    return triangles, tuple(sorted(profiles))


def _full_tanner_invariant(hx, hz):
    """Different values certify non-isomorphic uncoloured Tanner graphs."""
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    weights = [row.bit_count() for row in rows]
    overlaps = {}
    for i in range(m):
        for j in range(i):
            overlap = (rows[i] & rows[j]).bit_count()
            overlaps[i, j] = overlap
            if overlap in (1, 2):
                relation = one if overlap == 1 else two
                relation[i] |= 1 << j
                relation[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            common = (
                overlaps[i, j],
                (one[i] & one[j]).bit_count(),
                (one[i] & two[j]).bit_count()
                + (two[i] & one[j]).bit_count(),
                (two[i] & two[j]).bit_count())
            ki, kj = (weights[j],) + common, (weights[i],) + common
            profiles[i][ki] = profiles[i].get(ki, 0) + 1
            profiles[j][kj] = profiles[j].get(kj, 0) + 1
    return tuple(sorted(
        (weights[i], tuple(sorted(profiles[i].items())))
        for i in range(m)))


def _retired_order7_context(rng, G, idx):
    """Inactive historical order-seven context; never publishes new cells.

    Candidate generation is followed by intrinsic component pruning,
    ranked structural pairings and complete verified-matrix admission.
    Exhaustion voids a slot rather than recycling a saved design.
    """
    from itertools import combinations, product

    q, cap = G.q, min(9, W_MAX)
    if q != 7:
        raise ValueError("two-constituent catalogue requires order seven")
    elements = {G._key(g): g for g in G.elements}
    local = random.Random(BASE_SEED ^ 0x27310B2)

    def x(t):
        return elements[(t % q, 0)]

    def build_A(upper, lower, values):
        p = len(upper)
        assert len(lower) == p and 2*p + len(values) == 7
        return [
            [[G.e, x(d)] for d in upper]
            + [[G.e] for _ in values] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(v)] for v in values]
            + [[G.e, x(d)] for d in lower],
        ]

    def build_B(steps, s01, s02, s12):
        B = [[], [], []]

        def append(entries):
            for i in range(3):
                B[i].append(entries.get(i, []))

        for i, ds in enumerate(steps):
            for d in ds:
                append({i: [G.e, x(d)]})
        for (i, j), values in zip(
                ((0, 1), (0, 2), (1, 2)), (s01, s02, s12)):
            for v in values:
                append({i: [G.e], j: [x(v)]})
        assert all(len(row) == 10 for row in B)
        assert tuple(sum(map(len, row)) for row in B) == (7, 7, 6)
        return B

    def walk_key(adj):
        """Rooted weighted-walk profiles; an exact permutation invariant."""
        size = len(adj)
        power = [[0]*size for _ in range(size)]
        for u, neighbours in enumerate(adj):
            for v in neighbours:
                power[u][v] += 1
        initial = [row[:] for row in power]
        diagonal = [[] for _ in adj]
        two = None
        for length in range(2, 7):
            power = [
                [sum(power[v][w] for v in adj[u])
                 for w in range(size)]
                for u in range(size)
            ]
            if length == 2:
                two = [row[:] for row in power]
            for u in range(size):
                diagonal[u].append(power[u][u])
        return tuple(sorted(
            (len(adj[u]), tuple(diagonal[u]),
             tuple(sorted(
                 (len(adj[v]), initial[u][v], two[u][v])
                 for v in range(size))))
            for u in range(size)))

    def component_info(C, side):
        """Extract the actual left/right constituent with a zero dummy."""
        width = len(C[0]) * q
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], C)
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        cols = _incidence_columns(rows, width)
        if (any(c.bit_count() != 2 for c in cols)
                or len(set(cols)) != width):
            return None

        adj = [[] for _ in rows]
        masks = [0]*len(rows)
        for col in cols:
            first = col & -col
            u = first.bit_length() - 1
            v = (col ^ first).bit_length() - 1
            adj[u].append(v)
            adj[v].append(u)
            masks[u] |= 1 << v
            masks[v] |= 1 << u

        remaining = set(range(len(rows)))
        components = 0
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            queue = [root]
            for u in queue:
                for v in adj[u]:
                    if v in remaining:
                        remaining.remove(v)
                        queue.append(v)
            components += 1

        triangles = sum(
            (masks[u] & masks[v]).bit_count()
            for u in range(len(rows)) for v in adj[u]) // 6
        squares = sum(
            ((masks[u] & masks[v]).bit_count()
             * ((masks[u] & masks[v]).bit_count() - 1)) // 2
            for u in range(len(rows)) for v in range(u)) // 2
        signature = walk_key(adj)
        return components, triangles, squares, signature

    def filling_key(C):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in C for cell in row)

    def retain(bank, keys, C, side, card):
        filling = filling_key(C)
        if filling in keys["fill"]:
            return
        keys["fill"].add(filling)
        info = component_info(C, side)
        if info is None or (side == "A" and info[0] != 1):
            return
        if info[3] in keys["graph"]:
            return
        keys["graph"].add(info[3])
        bank.append({"cells": C, "info": info, "card": card})

    # A no longer has to be the unique triangle-free K7,7 graph.
    # Internal edges use distinct inverse classes, so no parallel
    # constituent columns are created.
    aa, akeys = [], {"fill": set(), "graph": set()}
    for p in range(4):
        step_sets = list(combinations((1, 2, 3), p))
        count = 7 - 2*p
        for ai, upper in enumerate(step_sets):
            for lower in step_sets[ai:]:
                for tail in combinations(range(1, q), count - 1):
                    values = (0,) + tail
                    C = build_A(upper, lower, values)
                    retain(aa, akeys, C, "A", (upper, lower, values))
    aa.sort(key=lambda a: (a["info"][1], a["info"][2], a["card"]))

    bb, bkeys = [], {"fill": set(), "graph": set()}

    # New rank-deficit prescription: K7,7 on rows 0/1 and K7 on row 2.
    # It has two connected components, but no dead protograph line.
    split_card = (((), (), (1, 2, 3)), tuple(range(q)), (), ())
    retain(bb, bkeys, build_B(*split_card), "B", split_card)

    # References reproduce the TWO shown parent matrices solely for
    # archive exclusion, and also provide B constituents for NEW A pairs.
    parent_cards = (
        (((1,), (1,), ()), (0, 2), (0, 2, 4), (1, 3, 6)),
        (((1,), (1,), (1,)), (0, 2, 4), (0, 2), (1, 4)),
    )
    for card in parent_cards:
        retain(bb, bkeys, build_B(*card), "B", card)

    # Exhaustive legal internal-count profiles; bounded label searches
    # retain different intrinsic graphs rather than twenty raw draws.
    for l0 in range(4):
        for l1 in range(l0, 4):
            for l2 in range(4):
                counts = (4-l0-l1+l2,
                          3-l0+l1-l2,
                          3+l0-l1-l2)
                if min(counts) < 0 or max(counts) > q:
                    continue
                options = [
                    list(combinations((1, 2, 3), number))
                    for number in (l0, l1, l2)
                ]
                for trial in range(32):
                    steps = tuple(local.choice(v) for v in options)
                    values = tuple(tuple(sorted(local.sample(range(q), c)))
                                   for c in counts)
                    card = (steps,) + values
                    retain(bb, bkeys, build_B(*card), "B", card)
    bb.sort(key=lambda b: (
        -b["info"][0], b["info"][1], b["info"][2], b["card"]))

    def full_key(hx, hz, n):
        """Intrinsic full Tanner information, without CSS/protograph colors.

        Incidence pairs have degree-four sector-1 qubits and degree-13/14
        sector-2 qubits. Every check meets two of the latter. Contracting
        those checks gives a weighted graph preserved by every uncoloured
        Tanner isomorphism. Different keys certify nonisomorphism;
        equality is inconclusive and is conservatively discarded.
        """
        rows = hx + hz
        cols = _incidence_columns(rows, n)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() >= 13]
        if len(high) != 6*q:
            raise ValueError("incidence degree projection changed")
        positions = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        adj = [[] for _ in high]
        for row in rows:
            bits = row & high_mask
            if bits.bit_count() != 2:
                raise ValueError("incidence check changed high-degree count")
            first = bits & -bits
            u = positions[first.bit_length() - 1]
            v = positions[(bits ^ first).bit_length() - 1]
            adj[u].append(v)
            adj[v].append(u)
        return _full_tanner_invariant(hx, hz), walk_key(adj)

    parent_A = build_A((), (), tuple(range(q)))
    archive_keys = set()
    for card in parent_cards:
        hx, hz, n = lifted_matrices(G, parent_A, build_B(*card))
        archive_keys.add(full_key(hx, hz, n))

    # Two-component B first: predicted k=266 instead of 259.
    # Afterwards cover increasingly triangle-rich connected pairings.
    plans = sorted(
        (-b["info"][0],
         a["info"][1] + b["info"][1],
         a["info"][2] + b["info"][2], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))

    graph_keys = set()
    accepted = 0
    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["cells"], b["cells"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 532 or not (475*n <= 1000*k < 525*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Complete exact kernel exclusion through weight two, on BOTH
        # full matrices and including mixed-sector supports.
        cx = _incidence_columns(hx, n)
        cz = _incidence_columns(hz, n)
        if (0 in cx or 0 in cz
                or len(set(cx)) != n or len(set(cz)) != n):
            continue

        signature = full_key(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        graph_keys.add(signature)
        if accepted == idx:
            if not css_ok(hx, hz):
                raise ValueError("verified two-constituent lift failed CSS")
            floor = 3
            if a["info"][1] == b["info"][1] == 0:
                if not (_no_kernel_up_to_three(hx, n)
                        and _no_kernel_up_to_three(hz, n)):
                    raise ValueError("triangle-free lift failed kernel-three")
                floor = 4
            return {
                "idx": idx, "cells": {"A": A, "B": B},
                "screen": {
                    "k": k, "excluded_below": floor,
                    "A_triangles": a["info"][1],
                    "B_triangles": b["info"][1],
                    "B_components": b["info"][0],
                    "cards": (a["card"], b["card"]),
                },
            }
        accepted += 1
    raise ValueError("novel two-constituent target-rate catalogue exhausted")


from functools import lru_cache


def _ten_span(rows):
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


def _ten_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _ten_four_witness(checks, stabilizers, width):
    """Find an actual weight-four logical in scaffold-built matrices."""
    cols = _incidence_columns(checks, width)
    own = _ten_span(stabilizers)
    pairs = {}
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            word = (1 << i) | (1 << j)
            old = pairs.get(syndrome)
            if old is None:
                pairs[syndrome] = word
                continue
            difference = word ^ old
            if (difference.bit_count() == 4
                    and _ten_outside(difference, own)):
                assert all((row & difference).bit_count() % 2 == 0
                           for row in checks)
                return difference
    return None


def _ten_graph_key(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    High-degree qubits are identified by degree, not sector labels.
    Their check contraction retains edge multiplicities, check weights
    and grounding markers. Unequal keys certify non-isomorphism;
    equal keys are conservatively discarded.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _incidence_columns(rows, width)
    high = [j for j, col in enumerate(cols) if col.bit_count() >= 12]
    if len(high) != 40:
        raise ValueError("order-ten incidence contraction changed")
    position = {j: i for i, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    size = len(high)
    adjacency = [Counter() for _ in high]
    markers = [Counter() for _ in high]

    for row in rows:
        endpoints = row & mask
        degree = endpoints.bit_count()
        if degree not in (1, 2):
            raise ValueError("check is not an intrinsic edge or ground")
        first = endpoints & -endpoints
        u = position[first.bit_length() - 1]
        weight = row.bit_count()
        if degree == 1:
            markers[u][weight] += 1
            adjacency[u][u] += weight + 1
        else:
            v = position[(endpoints ^ first).bit_length() - 1]
            adjacency[u][v] += weight + 1
            adjacency[v][u] += weight + 1

    labels = [
        (cols[j].bit_count(), tuple(sorted(markers[u].items())))
        for u, j in enumerate(high)]
    power = [[int(u == v) for v in range(size)]
             for u in range(size)]
    walks = [[[] for _ in range(size)] for _ in range(size)]
    for _ in range(8):
        following = [[0] * size for _ in range(size)]
        for u, neighbors in enumerate(adjacency):
            target = following[u]
            for v, multiplicity in neighbors.items():
                for w, value in enumerate(power[v]):
                    target[w] += multiplicity * value
        power = following
        for u in range(size):
            for v in range(size):
                walks[u][v].append(power[u][v])

    profiles = tuple(sorted(
        (labels[u], tuple(walks[u][u]),
         tuple(sorted(
             (labels[v], tuple(walks[v][v]), tuple(walks[u][v]))
             for v in range(size))))
        for u in range(size)))
    intrinsic = (
        width, len(rows),
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        _full_tanner_invariant(list(hx), list(hz)),
        profiles)
    return sha256(repr(intrinsic).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _ten_portfolio(G, count):
    """Pure deterministic catalogue; cached published cells are immutable.

    Structural cards are enumerated, not emitted as independent draws.
    No constituent-isomorphism pruning discards relative deck structure.
    Only complete intrinsic invariants separate published designs.
    """
    from itertools import combinations

    if G.q != 10 or A_SHAPE != (2, 7) or B_SHAPE != (2, 7):
        raise ValueError("joint-deck portfolio requires the order-ten bed")
    q, e, width = G.q, G.e, 7 * G.q
    cap = min(9, W_MAX)
    elements = {G._key(g): g for g in G.elements}

    def z(t):
        # (1,1) generates C5 x C2 and has order ten.
        return elements[(t % 5, t % 2)]

    def translation_key(values):
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    def canonical(card):
        kind, upper, lower, values = card
        forward = translation_key(values)
        if kind:
            return kind, upper, lower, forward
        backward = translation_key(tuple((-v) % q for v in values))
        return min((kind, upper, lower, forward),
                   (kind, lower, upper, backward))

    def build(card):
        kind, upper, lower, values = card
        if kind == 0:
            grid = [
                [[e, z(d)] for d in upper]
                + [[e] for _ in values] + [[] for _ in lower],
                [[] for _ in upper]
                + [[z(v)] for v in values]
                + [[e, z(d)] for d in lower],
            ]
        elif kind == 1:
            # Six matchings and one grounding singleton, rows (7,6).
            grid = [
                [[e] for _ in values] + [[e]],
                [[z(v)] for v in values] + [[]],
            ]
        else:
            # Five matchings, opposite-layer ring and grounding.
            # The singleton layer has no internal edge.
            grid = [
                [[e] for _ in values] + [[], [e]],
                [[z(v)] for v in values] + [[e, z(lower[0])], []],
            ]
        assert all(len(row) == 7 for row in grid)
        assert max(sum(map(len, row)) for row in grid) <= 7
        assert all(1 <= sum(len(grid[i][j]) for i in range(2)) <= 2
                   for j in range(7))
        assert all(len({G._key(g) for g in cell}) == len(cell)
                   for row in grid for cell in row)
        return grid

    def component_rows(grid, right):
        # A zero dummy exposes the exact scaffold constituent action.
        if right:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        else:
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    def component_info(grid, grounded):
        left = component_rows(grid, False)
        right = component_rows(grid, True)
        expected = 2*q if grounded else 2*q - 1
        if gf2_rank(left) != expected or gf2_rank(right) != expected:
            return None
        if not (_no_kernel_up_to_three(left, width)
                and _no_kernel_up_to_three(right, width)):
            return None

        # Exact weight-four multiplicity after exclusion through three.
        cols = _incidence_columns(left, width)
        multiplicities, collisions = {}, 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                old = multiplicities.get(syndrome, 0)
                collisions += old
                multiplicities[syndrome] = old + 1
        assert collisions % 3 == 0
        return collisions // 3

    descriptors = set()
    # Exclude the involution z^5: its internal binomial would produce
    # parallel lifted incidence edges and a weight-two kernel.
    directions = (1, 2, 3, 4)
    for p in range(3):
        steps = list(combinations(directions, p))
        for upper in steps:
            for lower in steps:
                for values in combinations(range(q), 7 - 2*p):
                    descriptors.add(canonical(
                        (0, upper, lower, values)))
    for values in combinations(range(q), 6):
        descriptors.add(canonical((1, (), (), values)))
    for d in directions:
        for values in combinations(range(q), 5):
            descriptors.add(canonical((2, (), (d,), values)))

    regular_buckets = [[], [], []]
    boundaries = []
    for card in sorted(descriptors):
        grid = build(card)
        fours = component_info(grid, card[0] != 0)
        if fours is None:
            continue
        record = (fours, card, grid)
        if card[0] == 0:
            regular_buckets[len(card[1])].append(record)
        else:
            boundaries.append(record)

    for bucket in regular_buckets:
        bucket.sort(key=lambda r: (r[0], r[1]))
    boundaries.sort(key=lambda r: (r[0], r[1]))

    # Interleave genuinely different support profiles. Preserve several
    # normalized deck descriptions even when constituent graphs agree.
    regular = []
    for turn in range(max([len(b) for b in regular_buckets] + [0])):
        for p in (2, 1, 0):
            if turn < len(regular_buckets[p]):
                regular.append(regular_buckets[p][turn])
    regular = regular[:96]
    boundaries = boundaries[:64]
    if not regular:
        raise ValueError("no connected triangle-free order-ten constituents")

    plans = []
    for ai, a in enumerate(regular):
        for bi in range(ai, len(regular)):
            b = regular[bi]
            plans.append((0, max(a[0], b[0]), a[0] + b[0], ai, bi))
    for ai, a in enumerate(regular):
        for bi, b in enumerate(boundaries):
            plans.append((1, max(a[0], b[0]), a[0] + b[0], ai, bi))
    plans.sort()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    keys, entries = set(), []
    for boundary, _, _, ai, bi in plans:
        a = regular[ai]
        b = (boundaries if boundary else regular)[bi]
        A, B = a[2], b[2]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if n != 530 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            continue

        # Complete, exact rejection includes mixed-sector supports.
        if not (_no_kernel_up_to_three(hx, n)
                and _no_kernel_up_to_three(hz, n)):
            continue

        signature = _ten_graph_key(hx, hz, n)
        if signature in keys:
            continue

        witness = _ten_four_witness(hz, hx, n)
        kind = "X"
        if witness is None:
            witness = _ten_four_witness(hx, hz, n)
            kind = "Z"
        if witness is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified order-ten joint deck failed CSS")

        keys.add(signature)
        entries.append((
            k * 16 / n, freeze(A), freeze(B), k,
            kind, witness, (a[1], b[1]), signature.hex()))
        if len(entries) >= count:
            break

    # Never pad exhausted slots with relabelings or lower-distance codes.
    entries.sort(key=lambda r: (-r[0], r[7]))
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of a complete certified-distance-four code."""
    catalogue = _ten_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(catalogue)):
        raise ValueError("novel order-ten distance-four catalogue exhausted")
    quality, A, B, k, kind, witness, cards, signature = catalogue[idx]
    return {
        "cells": {"A": A, "B": B},
        "screen_k": k, "exact_distance": 4,
        "verified_logical_type": kind, "verified_logical": witness,
        "design_cards": cards, "intrinsic_signature": signature,
        "certified_score": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell without resampling or shared-state mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform labeling alone removes the parent's sector-2 bottleneck; its measured top five contained four d30s=4 codes and one d30s=2 code, with minimum logicals exclusively in sector 2.
# REFUTED: Requiring k=277 admits the parent's proposed catalogue into cell 0.500; all 60 evaluated slots instead failed the verified target-rate check and none was measured.
# REFUTED: The four/six-internal-column catalogue supplies five distance-four measurements; its measured top five instead had distances 4,4,4,3,3, and 60 slots collapsed to ten distinct codes.
# REFUTED: Fixed K7,7 A with the triangle-free three-orbit B catalogue supplies five inequivalent distance-four measurements; all 60 evaluated slots collapsed to two measured k=259,d30s=4 codes, giving fitness 3.1156.
# REFUTED: Prioritizing the split K7,7 plus K7 B improves the distance-four leaders: all five current finalists instead measured k=266,d30s=3, and sixty slots retained eleven distinct codes.
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
