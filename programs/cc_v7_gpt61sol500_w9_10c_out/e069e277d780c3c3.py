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
CONCEPT = ("Metacyclic Z5:Z1, action 1, order 5, A=(3,8), B=(4,11): balanced support-and-voltage optimization against exact constituent short-word spectra should supply five novel distance-six-or-better finalists in cell 0.350.")
GROUP = ("Keep the parent's split metacyclic construction rule, GTYPE and A0; resize only m from eight to five, retaining r=1,s=1. The new factor 100 makes q=5 the largest admissible lift. The parent measured five k=200,d30s=6 finalists. Transfer of that distance reading to this untried asymmetric bed is a hypothesis, not a group-distance guarantee.")
SHAPES = ("A=(3,8), B=(4,11) give 40-variable/15-check and 55-variable/20-check constituents, n=500,kb=175 and guaranteed rate 35/100=0.350. Actual verified ranks retain [0.325,0.375). Parent minima were entirely sector-1 Acol/Bcol operators, especially around Acol2/3; four B check-block rows replace its three-row constituent mechanism. Five distance-six readings at the bound would score 12.600.")
FILLING = ("Twenty optimization cards exchange supports and relative voltages while preserving six terms per row and column degrees 1..3, giving BOTH CSS maxima nine. Exact verified constituent spectra prioritize exclusion through five and then fewer six-words. Independently optimized unequal constituents form an immutable portfolio. Complete verified lifts enforce rate, CSS and kernel exclusion through four; a checked cyclic translation automorphism accelerates the remaining mixed-sector five-word test. Intrinsic uncoloured Tanner invariants conservatively discard repeats. Weight-five fallbacks remain available; unseen-archive novelty and distance-six measurement survival remain hypotheses.")
FAMILY_NAME = "r035_m5r1s1_38x411_balanced_exchange_spectrum_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 5

def custom_ops(q):
    # Same split metacyclic rule; size factor 100 requires q <= 5.
    m, r, s = 5, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (4, 11)              # (rb, cb)


def _syndrome_columns(rows, n):
    """Column syndromes of a matrix returned by the verified lift."""
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _short_pairs(cols):
    """Return pair syndromes, or None if a kernel support of size 1..4 exists.

    With distinct nonzero columns, a collision between different
    unordered pairs necessarily has four distinct endpoints.
    """
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = {}
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in singles or s in pairs:
                return None
            pairs[s] = (j, i)
    return pairs


def _five_count(cols, pairs):
    """Exact number of weight-five kernel supports, given no weights 1..4."""
    total = 0
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            for h in range(j):
                pair = pairs.get(s ^ cols[h])
                if pair is not None:
                    a, b = pair
                    if a not in (i, j, h) and b not in (i, j, h):
                        total += 1
    return total // 10


def _triangle_component(rng, G, variant):
    """Each row has six terms; columns have degrees 2,2,2,2,2,2,3,3."""
    els = G.elements
    C = [[[] for _ in range(8)] for _ in range(3)]

    # Base voltage sums avoid triangles; pair voltage differences
    # 1,2,3 avoid the elementary four-cycle relations in Z8.
    edges = ((0, 1, 0), (0, 1, 1),
             (1, 2, 0), (1, 2, 2),
             (2, 0, 1), (2, 0, 4))
    for j, (u, v, shift) in enumerate(edges):
        C[u][j] = [els[0]]
        C[v][j] = [els[shift]]

    # Search genuine relative voltages, not independent row/column gauges.
    # Odd scaling changes the chosen normalization, not the screening.
    scale = (1, 3, 5, 7)[variant % 4]
    for j in (6, 7):
        C[0][j] = [els[0]]
        C[1][j] = [els[(scale * rng.randrange(8)) % 8]]
        C[2][j] = [els[(scale * rng.randrange(8)) % 8]]
    return C


def _components_from_verified(hx, hz, q):
    """Extract the actual left A and right B component matrices."""
    mask = (1 << q) - 1
    ar = []
    for i in range(3):
        for r in range(q):
            row = hx[(i * 8) * q + r]  # beta=0
            compact = 0
            for j in range(8):
                compact |= ((row >> (j * 8 * q)) & mask) << (j * q)
            ar.append(compact)
    # j=0, all B rows, contiguous sector-1 B-column blocks.
    br = [row & ((1 << (8 * q)) - 1) for row in hz[:3 * q]]
    return ar, br


def _inactive_pendant_context(rng, G, idx):
    """Historical reference only; never called by the new publication path."""
    from itertools import combinations

    if G.q != 8:
        raise ValueError("pendant/core profile requires order eight")
    local = random.Random(BASE_SEED ^ 0x3866A)
    width = 8 * G.q
    mask = (1 << width) - 1
    identity = [[[G.e]]]

    def draw_component():
        C = [[[] for _ in range(8)] for _ in range(3)]
        C[0][0] = [G.e]
        C[0][1] = [G.e]
        C[1][1] = [G.e]

        # Three distinct matchings with six distinct directed differences.
        for j, voltage in enumerate((0, 1, 3), 2):
            C[1][j] = [G.e]
            C[2][j] = [G.elements[voltage]]

        # No full three-row columns: all three degree-three columns
        # have a binomial endpoint and a monomial endpoint.
        for j, row in ((5, 0), (6, 0), (7, 1)):
            gap = local.randrange(1, G.q)
            C[row][j] = [G.e, G.elements[gap]]
            C[2][j] = [G.elements[local.randrange(G.q)]]
        return C

    def constituent_rows(C):
        # Identity dummy sides make extraction unambiguous. These remain
        # scaffold-built left/right lifts, not independently coded lifts.
        left, _, _ = lifted_matrices(G, C, identity)
        _, right, _ = lifted_matrices(G, identity, C)
        return ([row & mask for row in left],
                [row & mask for row in right])

    def constituent_sixes(rows):
        """Exclude weights <=5 and count weight-six kernel words exactly."""
        cols = _syndrome_columns(rows, width)
        pairs = _short_pairs(cols)
        if pairs is None:
            return None
        triples = {}
        collisions = 0
        for i, first in enumerate(cols):
            for j in range(i):
                pair = first ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in pairs:
                        return None
                    previous = triples.get(syndrome, 0)
                    collisions += previous
                    triples[syndrome] = previous + 1
        # Smaller words have been excluded, so colliding triples are
        # disjoint. Each six-word has ten unordered triple partitions.
        return collisions // 10

    def graph_signature(rows):
        """An uncoloured check-intersection Tanner-isomorphism invariant.

        Equality is inconclusive; inequality certifies nonisomorphism.
        Conservative deduplication therefore cannot submit orbit copies.
        """
        m = len(rows)
        one, several = [0] * m, [0] * m
        for i in range(m):
            for j in range(i):
                overlap = (rows[i] & rows[j]).bit_count()
                if overlap:
                    adjacency = one if overlap == 1 else several
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        profiles = [{} for _ in rows]
        for i in range(m):
            for j in range(i):
                kind = (((one[i] >> j) & 1)
                        + 2 * ((several[i] >> j) & 1))
                key = (
                    kind,
                    (one[i] & one[j]).bit_count(),
                    (one[i] & several[j]).bit_count()
                    + (several[i] & one[j]).bit_count(),
                    (several[i] & several[j]).bit_count(),
                )
                for profile in (profiles[i], profiles[j]):
                    profile[key] = profile.get(key, 0) + 1
        return tuple(sorted(
            (row.bit_count(), tuple(sorted(profile.items())))
            for row, profile in zip(rows, profiles)))

    # Retain different verified constituent graph invariants, not random
    # gauges of one constituent. Low six-word multiplicity breaks ties.
    components, component_keys = [], set()
    fillings_seen = set()
    for attempt in range(8192):
        C = draw_component()
        fingerprint = tuple(tuple(G.idx(g) for g in cell)
                            for row in C for cell in row)
        if fingerprint in fillings_seen:
            continue
        fillings_seen.add(fingerprint)
        left, right = constituent_rows(C)
        if gf2_rank(left) != 3 * G.q or gf2_rank(right) != 3 * G.q:
            continue
        six_left = constituent_sixes(left)
        if six_left is None:
            continue
        six_right = constituent_sixes(right)
        if six_right is None:
            continue
        signature = graph_signature(left)
        if signature in component_keys:
            continue
        component_keys.add(signature)
        components.append((max(six_left, six_right),
                           six_left + six_right, fingerprint, C))
        if len(components) >= 40:
            break
    if len(components) < 8:
        raise ValueError("too few distinct kernel-six constituent designs")
    components.sort(key=lambda item: item[:3])

    plans = sorted(
        (max(components[a][0], components[b][0]),
         components[a][1] + components[b][1], a, b)
        for a, b in combinations(range(len(components)), 2))

    # Primary and fallback catalogue positions belong to disjoint slots.
    # Rebuilding the same prefix is pure and removes cross-slot state.
    wanted = {idx + N_SLOTS * retry for retry in range(4)}
    last_position = max(wanted)
    graph_keys = set()
    accepted = 0
    fallback = None
    n1 = A_SHAPE[1] * B_SHAPE[1] * G.q

    def mixed_five(cols, pairs):
        """Exact remaining weight-five test after constituent screening.

        A pure sector-1 kernel word decomposes into constituent words,
        already excluded through five. Every other support includes a
        sector-2 qubit and has a triple partition containing that qubit.
        """
        for i in range(n1, len(cols)):
            first = cols[i]
            for j in range(i):
                pair = first ^ cols[j]
                for h in range(j):
                    if (pair ^ cols[h]) in pairs:
                        return True
        return False

    for _, _, a, b in plans:
        A, B = components[a][3], components[b][3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb):
            continue
        if _grid_weights(sa, sb) > min(9, W_MAX):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.325 * n <= k < 0.375 * n):
            continue

        signature = graph_signature(hx + hz)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        position = accepted
        accepted += 1
        if position not in wanted:
            if position >= last_position:
                break
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified catalogue failed CSS commutation")

        legal = True
        floor = 6
        for checks in (hz, hx):
            cols = _syndrome_columns(checks, n)
            pairs = _short_pairs(cols)
            if pairs is None:
                legal = False
                break
            if mixed_five(cols, pairs):
                floor = 5
            del pairs
        if legal:
            result = {"A": A, "B": B, "k": k,
                      "screened_floor": floor, "catalogue": position}
            if floor == 6:
                return result
            if fallback is None:
                fallback = result
        if position >= last_position:
            break

    if fallback is not None:
        return fallback
    raise ValueError("reserved novel catalogue designs failed full screens")


from functools import lru_cache


def _asymmetric_graph_key(rows, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Equality is inconclusive and is discarded conservatively.
    Unequal invariants certify that two Tanner graphs are not isomorphic.
    """
    from collections import Counter
    from hashlib import sha256

    cols = _syndrome_columns(rows, width)
    incidence = []
    for col in cols:
        neighbours = []
        while col:
            bit = col & -col
            neighbours.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(neighbours)

    degrees = [row.bit_count() for row in rows]
    adjacency = [{} for _ in rows]
    masks = [0] * len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = adjacency[j][i] = overlap
                masks[i] |= 1 << j
                masks[j] |= 1 << i

    features = [
        (degrees[i], tuple(sorted(Counter(
            (degrees[j], overlap,
             (masks[i] & masks[j]).bit_count())
            for j, overlap in neighbours.items()).items())))
        for i, neighbours in enumerate(adjacency)
    ]
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]

    for _ in range(3):
        qfeatures = [
            tuple(sorted(colors[r] for r in neighbours))
            for neighbours in incidence
        ]
        qpalette = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        profiles = [[] for _ in rows]
        for neighbours, feature in zip(incidence, qfeatures):
            for r in neighbours:
                profiles[r].append(qpalette[feature])
        refined = [
            (colors[i], tuple(sorted(profiles[i])),
             tuple(sorted((overlap, colors[j])
                          for j, overlap in adjacency[i].items())))
            for i in range(len(rows))
        ]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items())),
        ))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]

    value = (
        width, len(rows),
        tuple(sorted(Counter(degrees).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        history,
    )
    return sha256(repr(value).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _asymmetric_portfolio(G, count):
    """Pure deterministic optimization with immutable publication grids."""
    from collections import Counter

    if G.q != 5:
        raise ValueError("asymmetric exchange bed requires order five")
    q, cap = G.q, min(9, W_MAX)
    dummy = [[[]]]

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    def normalize(C):
        """Remove cyclic column gauges, not relative labels."""
        rows, columns = len(C), len(C[0])
        D = [[list(cell) for cell in row] for row in C]
        for j in range(columns):
            anchor = next(
                G.idx(D[i][j][0]) for i in range(rows) if D[i][j])
            for i in range(rows):
                D[i][j] = [
                    G.elements[t] for t in sorted(
                        (G.idx(g) - anchor) % q for g in D[i][j])
                ]
        return D

    def draw(local, shape, card):
        """Different count pressures, always row-six and column 1..3."""
        rows, columns = shape
        degrees = [1] * columns
        remaining = 6 * rows - columns
        while remaining:
            choices = [j for j in range(columns) if degrees[j] < 3]
            if card % 4 < 2:
                # Concentrated hyperedges versus flatter incidence cards.
                largest = max(degrees[j] for j in choices)
                preferred = [j for j in choices if degrees[j] == largest]
                j = local.choice(preferred)
            else:
                smallest = min(degrees[j] for j in choices)
                preferred = [j for j in choices if degrees[j] == smallest]
                j = local.choice(preferred)
            degrees[j] += 1
            remaining -= 1

        # Allocate complete column term counts to exact row budgets.
        capacities = [6] * rows
        counts = [[0] * columns for _ in range(rows)]
        order = list(range(columns))
        local.shuffle(order)
        for j in order:
            for _ in range(degrees[j]):
                available = [i for i in range(rows) if capacities[i]]
                if card % 5 == 0:
                    available.sort(key=lambda i: -counts[i][j])
                    i = available[0] if local.random() < 0.6 else \
                        local.choice(available)
                else:
                    i = local.choice(available)
                counts[i][j] += 1
                capacities[i] -= 1

        assert capacities == [0] * rows
        C = [
            [[G.elements[t] for t in local.sample(range(q), terms)]
             for terms in row]
            for row in counts
        ]
        return normalize(C)

    def mutate(local, C, probability):
        rows, columns = len(C), len(C[0])
        D = [[list(cell) for cell in row] for row in C]
        if local.random() < probability:
            degrees = [
                sum(len(D[i][j]) for i in range(rows))
                for j in range(columns)
            ]
            moves = [
                (i, a, b)
                for i in range(rows)
                for a in range(columns) for b in range(columns)
                if a != b and D[i][a]
                and degrees[a] > 1 and degrees[b] < 3
            ]
            if moves:
                i, a, b = local.choice(moves)
                del D[i][a][local.randrange(len(D[i][a]))]
                choices = [g for g in G.elements if g not in D[i][b]]
                D[i][b].append(local.choice(choices))
                return normalize(D)

        occupied = [
            (i, j) for i in range(rows) for j in range(columns) if D[i][j]
        ]
        i, j = local.choice(occupied)
        position = local.randrange(len(D[i][j]))
        choices = [g for g in G.elements if g not in D[i][j]]
        D[i][j][position] = local.choice(choices)
        return normalize(D)

    def component_rows(C, side):
        """Expose only the scaffold's actual constituent action."""
        width = len(C[0]) * q
        mask = (1 << width) - 1
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, dummy)
        else:
            _, rows, _ = lifted_matrices(G, dummy, C)
        return [row & mask for row in rows], width

    def spectrum(rows, width):
        """Exact cleanliness through four; exact five/six multiplicities.

        Dirty penalties guide optimization but are not distance readings.
        Triple collisions count six-words exactly after weights <=4
        have been excluded.
        """
        cols = _syndrome_columns(rows, width)
        singles = Counter(cols)
        bad = 10000 * (
            singles.get(0, 0)
            + sum(c * (c - 1) // 2 for c in singles.values()))
        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                bad += 100 * singles.get(syndrome, 0)
                previous = pairs.get(syndrome, 0)
                bad += previous
                pairs[syndrome] = previous + 1
        if bad:
            return 4, bad, 0

        triples = {}
        five = collisions = 0
        for i, first in enumerate(cols):
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    syndrome = partial ^ cols[h]
                    five += int(syndrome in pairs)
                    previous = triples.get(syndrome, 0)
                    collisions += previous
                    triples[syndrome] = previous + 1
        assert five % 10 == 0 and collisions % 10 == 0
        five, six = five // 10, collisions // 10
        return (5 if five else 6 if six else 7), five, six

    def objective(stats, card):
        floor, five, six = stats
        if floor == 4:
            return floor, -five, 0
        if floor == 5:
            return floor, -five, -six
        target = q * (card % 5)
        return floor, -six, -abs(six - target)

    banks = {}
    for side, shape in (("A", A_SHAPE), ("B", B_SHAPE)):
        bank, fillings, graph_keys = [], set(), set()
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0x354115 + 104729 * card
                             + (1000003 if side == "B" else 0)))
            retained = []
            for restart in range(8):
                C = draw(local, shape, card + restart)
                rows, width = component_rows(C, side)
                stats = spectrum(rows, width)
                quality = objective(stats, card)
                best = (quality, C, stats)

                for step in range(144):
                    probability = (0.08, 0.20, 0.35, 0.55)[card // 5]
                    D = mutate(local, C, probability)
                    dr, dw = component_rows(D, side)
                    ds = spectrum(dr, dw)
                    dq = objective(ds, card)
                    if dq > best[0]:
                        best = (dq, D, ds)
                    if dq >= quality or (
                            step % 37 == 36 and ds[0] >= stats[0] - 1):
                        C, stats, quality = D, ds, dq

                _, C, stats = best
                if stats[0] < 5:
                    continue
                filling = freeze(C)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, width = component_rows(C, side)
                signature = _asymmetric_graph_key(rows, width)
                if signature in graph_keys:
                    continue
                graph_keys.add(signature)
                rank = gf2_rank(rows)
                retained.append((
                    stats[0], stats[1], stats[2], rank,
                    card, restart, filling,
                ))

            retained.sort(key=lambda r: (-r[0], r[1], r[2], r[5]))
            bank.extend(retained[:4])
        bank.sort(key=lambda r: (-r[0], r[1], r[2], r[4], r[5]))
        banks[side] = bank

    aa, bb = banks["A"], banks["B"]
    if not aa or not bb:
        raise ValueError("no kernel-four-clean asymmetric constituent bank")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            priority = (
                min(a[0], b[0]), -a[1] - b[1],
                -max(a[2], b[2]), -a[2] - b[2],
                int(a[4] != b[4]),
            )
            plans.append((priority, ai, bi))
    plans.sort(reverse=True)

    n1 = A_SHAPE[1] * B_SHAPE[1] * q

    def translate(word):
        """A qubit-coordinate permutation, not an independently coded lift."""
        moved = 0
        while word:
            bit = word & -word
            block, h = divmod(bit.bit_length() - 1, q)
            moved |= 1 << (block * q + (h + 1) % q)
            word ^= bit
        return moved

    def mixed_five(cols, pairs, n):
        """Exact remaining five-word existence test after verified symmetry.

        Translate a sector-2 endpoint to coordinate zero. Every mixed
        five-support then partitions into an anchored triple and a pair.
        Overlapping matches would yield a kernel word of weight <=4,
        already excluded by _short_pairs.
        """
        for anchor in range(n1, n, q):
            first = cols[anchor]
            for j in range(n):
                if j == anchor:
                    continue
                partial = first ^ cols[j]
                for h in range(j):
                    if h != anchor and (partial ^ cols[h]) in pairs:
                        return True
        return False

    results, graph_keys = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a[6], b[6]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert all(sum(row) == 6 for row in sa + sb)
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 500 and k > 0 and 325 * n <= 1000 * k < 375 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        signature = _asymmetric_graph_key(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1

        floor = 6 if min(a[0], b[0]) >= 6 else 5
        passed = True
        if floor == 6:
            if (set(map(translate, hx)) != set(hx)
                    or set(map(translate, hz)) != set(hz)):
                raise ValueError("verified lift failed cyclic translation test")

        for checks in (hx, hz):
            cols = _syndrome_columns(checks, n)
            pairs = _short_pairs(cols)
            if pairs is None:
                passed = False
                break
            if floor == 6 and mixed_five(cols, pairs, n):
                floor = 5
        if not passed:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified asymmetric lift failed CSS")

        # A pure sector-1 word splits into constituent kernel words.
        # Thus clean constituents plus both mixed tests certify floor six.
        # Constituent seven-cleanliness is only a selection feature.
        quality = (
            floor, k, min(a[0], b[0]),
            -max(a[2], b[2]), -a[2] - b[2],
        )
        results.append((
            quality, A, B, k, floor,
            (a[4], b[4]), (a[2], b[2]),
        ))
        if len(results) >= count + 12:
            break
        if tested >= max(480, 12 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one complete screened asymmetric design without resampling."""
    portfolio = _asymmetric_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("screened asymmetric portfolio exhausted")
    quality, A, B, k, floor, cards, sixes = portfolio[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "verified_kernel_floor": floor,
        "optimization_cards": cards,
        "constituent_six_counts": sixes,
        "selection_quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell from the immutable screened portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Reserved positions separated by the pendant catalogue's Tanner invariants remain evaluator-distinct: sixty evaluated slots retained twenty-six distinct codes and thirty-four iso_dup slots.
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
