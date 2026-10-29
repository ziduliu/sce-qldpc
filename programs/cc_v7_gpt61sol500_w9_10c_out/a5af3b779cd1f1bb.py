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
CONCEPT = ("Cyclic Z8, order 8, A=(3,8), B=(3,8): support-and-voltage exchange optimization against exact short-kernel spectra should exceed fitness 12.329 through larger distance readings or useful rank deficits; novel kernel-five-clean fallbacks are retained.")
GROUP = ("Keep the frozen metacyclic realization Z8:Z1, s=1, q=8. This bed leads cell 0.350 at fitness 12.329, with all five current finalists at k=200,d30s=6. Search a different filling mechanism on this measured bed; neither Z8 nor a large search reading certifies larger distance.")
SHAPES = ("Keep equal (3,8) sides, with 64 variables and 24 checks per constituent. Factor 73 permits q<=8; q=8 gives n=584, kb=200 and guaranteed rate 25/73=0.342466 in cell 0.350. Current minima were entirely sector-1 Acol/Bcol witnesses, particularly around Acol2/3 and several B columns, with no measured Arow/Brow minimum.")
FILLING = ("Twenty optimization cards vary incidence pressure and short-word multiplicity objectives. Support exchanges preserve six terms per row and column degrees 1..3, so BOTH CSS maxima remain nine. Verified constituents are optimized using exact weight-five/six spectra without forcing full rank. A deterministic immutable portfolio pairs independently optimized constituents, measures actual target-band ranks, verifies CSS and excludes complete kernels through five. Full uncoloured Tanner invariants reject repeats and explicitly shown archived fillings conservatively. These gates certify distance at least six, not seven, and cannot establish novelty against the unavailable complete archive.")
FAMILY_NAME = "r035_m8r1s1_38x38_support_exchange_rank_spectrum_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same metacyclic construction, smaller cyclic order for size factor 73.
    m, r, s = 8, 1, 1
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
B_SHAPE = (3, 8)               # (rb, cb)


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


def _pendant_catalogue_reference(rng, G, idx):
    """Uncalled historical implementation; never publishes a candidate."""
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


def _exchange_signature(rows, width):
    """Conservative invariant of the complete uncoloured Tanner graph.

    Equality is inconclusive and is discarded. The evaluator remains
    authoritative, especially for comparisons with the unseen archive.
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

    adjacency = [{} for _ in rows]
    masks = [0] * len(rows)
    degrees = [row.bit_count() for row in rows]
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
        for i, neighbours in enumerate(adjacency)]
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]
    for _ in range(3):
        qfeatures = [tuple(sorted(colors[r] for r in neighbours))
                     for neighbours in incidence]
        qpalette = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        profiles = [[] for _ in rows]
        for neighbours, feature in zip(incidence, qfeatures):
            for r in neighbours:
                profiles[r].append(qpalette[feature])
        refined = [
            (colors[i], tuple(sorted(profiles[i])),
             tuple(sorted((overlap, colors[j])
                          for j, overlap in adjacency[i].items())))
            for i in range(len(rows))]
        history.append((tuple(sorted(Counter(qfeatures).items())),
                        tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return sha256(repr((
        width, len(rows), tuple(sorted(Counter(degrees).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        history)).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _exchange_portfolio(G, count):
    """Pure deterministic optimization with immutable publication data.

    Memoization changes build cost only. No caller can mutate the
    cached grids, and slot RNG state does not influence another slot.
    """
    from collections import Counter

    if G.q != 8:
        raise ValueError("support-exchange portfolio requires order eight")
    q, width = G.q, 8 * G.q
    mask = (1 << width) - 1
    dummy = [[[]]]
    cap = min(9, W_MAX)

    profiles = (
        ((1, 1, 0, 0, 0, 2, 2, 0),
         (0, 1, 1, 1, 1, 0, 0, 2),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        # Two full hyperedges replace two binomial-ended columns.
        ((1, 1, 0, 0, 0, 1, 1, 2),
         (0, 1, 1, 1, 1, 1, 1, 0),
         (0, 0, 1, 1, 1, 1, 1, 1)),
    )

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    def normalize(C):
        """Remove cyclic column gauges, retaining relative voltages."""
        out = [[list(cell) for cell in row] for row in C]
        for j in range(8):
            anchor = next(out[i][j][0][0]
                          for i in range(3) if out[i][j])
            for i in range(3):
                out[i][j] = sorted(
                    [G.elements[(g[0] - anchor) % q]
                     for g in out[i][j]])
        return out

    def draw(local, profile):
        C = [
            [[G.elements[v] for v in local.sample(range(q), terms)]
             for terms in row]
            for row in profile]
        # Optimize the matching bundle inside a known legal voltage class;
        # all hyperedge labels remain independently selected.
        for j, voltage in enumerate((0, 1, 3), 2):
            if profile[1][j] == profile[2][j] == 1:
                C[1][j] = [G.e]
                C[2][j] = [G.elements[voltage]]
        return normalize(C)

    def mutate(local, C, exchange_probability):
        D = [[list(cell) for cell in row] for row in C]
        if local.random() < exchange_probability:
            degrees = [sum(len(D[i][j]) for i in range(3))
                       for j in range(8)]
            moves = [
                (i, a, b)
                for i in range(3) for a in range(8) for b in range(8)
                if a != b and D[i][a] and degrees[a] > 1
                and degrees[b] < 3]
            if moves:
                i, a, b = local.choice(moves)
                del D[i][a][local.randrange(len(D[i][a]))]
                choices = [g for g in G.elements if g not in D[i][b]]
                D[i][b].append(local.choice(choices))
                return normalize(D)
        occupied = [(i, j) for i in range(3) for j in range(8)
                    if D[i][j]]
        i, j = local.choice(occupied)
        t = local.randrange(len(D[i][j]))
        choices = [g for g in G.elements if g not in D[i][j]]
        D[i][j][t] = local.choice(choices)
        return normalize(D)

    def component_rows(C, side):
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, dummy)
        else:
            _, rows, _ = lifted_matrices(G, dummy, C)
        return [row & mask for row in rows]

    def spectrum(rows):
        """Exact cleanliness through four and exact five/six counts.

        The dirty-screen cost is an optimization penalty, not an exact
        distance. Once weights <=4 are excluded, both multiplicities
        below are exact.
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
                old = pairs.get(syndrome, 0)
                bad += old
                pairs[syndrome] = old + 1
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
        assert five % 10 == collisions % 10 == 0
        five, six = five // 10, collisions // 10
        return (5 if five else 6 if six else 7), five, six

    def objective(stats, rank, card):
        floor, five, six = stats
        if floor == 4:
            return floor, -five, 0, 0
        if floor == 5:
            return floor, -five, -six, -rank
        # Different cards deliberately trade incidence exploration,
        # rank pressure and six-word multiplicity, rather than drawing
        # twenty copies of one voltage prescription.
        target = q * (card % 5)
        if card // 5 % 2:
            return floor, -rank, -six, -abs(six - target)
        return floor, -six, -abs(six - target), -rank

    components, component_fillings = [], set()
    for card in range(20):
        local = random.Random(BASE_SEED ^ (0xE83506 + 104729 * card))
        retained = []
        for restart in range(8):
            C = draw(local, profiles[(card + restart) % len(profiles)])
            # Broader support starts are obtained by legal exchanges,
            # not by changing the frozen protograph dimensions.
            if card >= 10:
                for _ in range(4 + restart % 4):
                    C = mutate(local, C, 1.0)
            rows = component_rows(C, "A")
            stats, rank = spectrum(rows), gf2_rank(rows)
            quality = objective(stats, rank, card)
            best = (quality, C, stats, rank)
            for step in range(144):
                probability = (0.08, 0.20, 0.35, 0.55)[card // 5]
                D = mutate(local, C, probability)
                dr = component_rows(D, "A")
                ds, drank = spectrum(dr), gf2_rank(dr)
                dq = objective(ds, drank, card)
                if dq > best[0]:
                    best = (dq, D, ds, drank)
                if dq >= quality or (
                        step % 37 == 36 and ds[0] >= stats[0] - 1):
                    C, stats, rank, quality = D, ds, drank, dq
            _, C, stats, rank = best
            if stats[0] < 6:
                continue
            right = component_rows(C, "B")
            rs = spectrum(right)
            if rs[0] < 6:
                continue
            filling = freeze(C)
            if filling in component_fillings:
                continue
            component_fillings.add(filling)
            retained.append((
                min(stats[0], rs[0]), max(stats[2], rs[2]),
                stats[2] + rs[2], rank + gf2_rank(right),
                card, restart, filling))
        retained.sort(key=lambda r: (-r[0], r[1], r[3], r[5]))
        components.extend(retained[:4])

    if len(components) < 8:
        raise ValueError("exchange search found too few kernel-five-clean constituents")

    # These are archive references ONLY, never candidate outputs.
    # Restrict the exclusion list to explicitly shown pairings rather
    # than claiming access to the complete run archive.
    reference_cards = (
        (5, 6, 5, 2, 4, 7), (6, 1, 5, 2, 7, 7),
        (6, 3, 1, 4, 5, 7), (2, 7, 7, 4, 5, 6),
        (4, 1, 5, 2, 5, 7), (6, 3, 7, 4, 2, 4),
        (1, 3, 3, 6, 7, 5), (2, 3, 4, 6, 5, 2),
        (7, 5, 1, 5, 7, 5), (1, 5, 2, 7, 2, 6),
        (6, 1, 1, 2, 7, 6), (1, 2, 3, 7, 4, 5),
        (7, 3, 5, 5, 7, 7),
    )

    def reference(card):
        C = [[[] for _ in range(8)] for _ in range(3)]
        C[0][0] = C[0][1] = C[1][1] = [G.e]
        for j, voltage in enumerate((0, 1, 3), 2):
            C[1][j], C[2][j] = [G.e], [G.elements[voltage]]
        for t, (j, i) in enumerate(((5, 0), (6, 0), (7, 1))):
            C[i][j] = [G.e, G.elements[card[t]]]
            C[2][j] = [G.elements[card[t + 3]]]
        return C

    references = [reference(card) for card in reference_cards]
    archive_keys = set()
    shown_pairs = (
        (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),
        (0, 4), (1, 4), (2, 4), (2, 5), (3, 4), (4, 5),
        (0, 6), (0, 7), (0, 8), (1, 7), (2, 7), (1, 9),
        (4, 6), (4, 9), (4, 8), (6, 8),
        (10, 10), (10, 11), (10, 12), (11, 11),
    )
    for a, b in shown_pairs:
        hx, hz, n = lifted_matrices(G, references[a], references[b])
        archive_keys.add(_exchange_signature(hx + hz, n))

    plans = []
    for a in range(len(components)):
        for b in range(a, len(components)):
            ca, cb = components[a], components[b]
            pressure = (
                min(ca[0], cb[0]), -ca[3] - cb[3],
                -max(ca[1], cb[1]), -ca[2] - cb[2],
                int(ca[4] != cb[4]))
            plans.append((pressure, a, b))
    plans.sort(reverse=True)

    n1 = 64 * q

    def mixed_five(cols, pairs):
        """Test all remaining five-supports after constituent screens."""
        for i in range(n1, len(cols)):
            first = cols[i]
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    if (partial ^ cols[h]) in pairs:
                        return True
        return False

    results, graph_keys = [], set()
    for pressure, a, b in plans:
        A, B = components[a][6], components[b][6]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 584 and k > 0 and 325*n <= 1000*k < 375*n):
            continue
        signature = _exchange_signature(hx + hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        graph_keys.add(signature)

        passed = True
        for checks in (hx, hz):
            cols = _syndrome_columns(checks, n)
            pairs = _short_pairs(cols)
            if pairs is None or mixed_five(cols, pairs):
                passed = False
                break
        if not passed:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified exchange pair failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= cap
        # Constituent seven-cleanliness is only a selection feature:
        # no claim is made about complete mixed-sector weight six.
        quality = (min(components[a][0], components[b][0]), k,
                   -max(components[a][1], components[b][1]),
                   -components[a][2] - components[b][2])
        results.append((quality, A, B, k,
                        (components[a][4], components[b][4]),
                        (components[a][1], components[b][1])))
        if len(results) >= count + 12:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable, independently optimized complete design."""
    portfolio = _exchange_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel kernel-five-clean exchange portfolio exhausted")
    quality, A, B, k, cards, sixes = portfolio[idx]
    return {"A": A, "B": B, "screen_k": k, "quality": quality,
            "optimization_cards": cards, "component_six_counts": sixes,
            "complete_kernel_floor": 6}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell without post-screen sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Lower six-word multiplicity on the fixed pendant catalogue eliminates its measured weight-six logical family: all five current finalists instead had k=200,d30s=6, with thousands of sector-1 weight-six witnesses.
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
