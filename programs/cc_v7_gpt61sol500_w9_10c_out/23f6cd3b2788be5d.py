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
CONCEPT = ("C3 semidirect C16 with inversion action, order 48, A=(1,2), B=(2,5): witness-guided cubic reinforcement around longer matching cores should supply five fresh measured distance-ten readings in cell 0.250; this is a falsifiable measurement hypothesis.")
GROUP = ("Jump from the order-20 affine group to a nonsplit C8-central extension of S3, isomorphic to C3:C16 with inversion action. Its different order proves it is not a parent relabeling. The recorded order-48 C3:C16 (1,2)x(2,5) bed reached fitness 20.9 with distance-nine readings. Its longer central direction motivates new reinforcement optimization, not a transferred distance guarantee.")
SHAPES = ("Change to A=(1,2), B=(2,5): respectively 96 variables/48 checks and 240 variables/96 checks. Factor twelve permits q<=49; q=48 gives n=576, kb=144 and guaranteed rate 0.250. Target the weaker cell 0.250 instead of the parent's 0.300. Parent weight-seven minima were entirely sector-1 Acol/Bcol witnesses, often involving Acol1/2 and Bcol2/3. Complete verified ranks retain only [0.225,0.275).")
FILLING = ("Twenty central-coordinate, noncommutation and support prescriptions optimize noncommuting A trinomials and crossed-binomial, isolated-trinomial or pendant/cubic B incidence. Every constituent row totals six and every column at most three, so BOTH CSS maxima are nine, also below twelve. Verified kernel-four rejection and attained kernel witnesses guide witness-block label repairs. Full matrices enforce actual rate, CSS, both kernel-four screens and intrinsic Tanner separation, supplemented by canonical certificates when available. Embedded logical witnesses undergo stabilizer descent and mixed-sector probing. Cached outputs are immutable; distance-ten survival and novelty against the unseen archive are not certified.")
FAMILY_NAME = "r025_s3_c8_nonsplit_q48_12x25_cubic_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("the nonsplit S3/C8 central extension has order 48")
    # (a,b,c)=x^a*y^b*z^c, x^3=z^8=1, y^2=z,
    # y*x*y^-1=x^-1, and z central. Thus y has order sixteen.
    # The carry cocycle records y*y=z.
    els = [(a, b, c) for c in range(8)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 8)

    def mul(g, h):
        a, b, c = red(g)
        d, f, t = red(h)
        return ((a + (-1 if b else 1)*d) % 3,
                (b + f) % 2,
                (c + t + (b + f)//2) % 8)

    def inv(g):
        a, b, c = red(g)
        return (((a if b else -a) % 3), b, (-c - b) % 8)

    def idx(g):
        a, b, c = red(g)
        return a + 3*b + 6*c

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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


def _unused_z8_catalogue(rng, G, idx):
    """Uncalled historical order-eight catalogue; never publishes cells."""
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


def _f20_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _f20_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _f20_rows(G, grid, right=False):
    """Expose the exact scaffold action; the zero dummy is never emitted."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    return [row & mask for row in rows], width


def _f20_girth(rows, width):
    """Exact graph girth for a verified degree-two constituent."""
    cols = _syndrome_columns(rows, width)
    adj = [[] for _ in rows]
    edges = set()
    for col in cols:
        if col.bit_count() != 2:
            return 0
        low = col & -col
        a = low.bit_length() - 1
        b = (col ^ low).bit_length() - 1
        edge = (min(a, b), max(a, b))
        if edge in edges:
            return 2
        edges.add(edge)
        adj[a].append(b)
        adj[b].append(a)
    best = len(rows) + 1
    for root in range(len(rows)):
        distance = [-1] * len(rows)
        parent = [-1] * len(rows)
        distance[root] = 0
        queue = [root]
        for u in queue:
            for v in adj[u]:
                if distance[v] < 0:
                    distance[v] = distance[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v:
                    best = min(best, distance[u] + distance[v] + 1)
        if best < 8:
            return best
    return best


def _f20_floor(rows, width):
    """Exclude kernels <=4; then detect five and count six exactly."""
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
                value = pair ^ cols[h]
                if value in pairs:
                    return (5, 0)
                old = triples.get(value, 0)
                collisions += old
                triples[value] = old + 1
    # With smaller words absent, equal-syndrome triples are disjoint.
    assert collisions % 10 == 0
    return (6 if collisions else 7, collisions // 10)


def _f20_key(rows, width):
    """Conservative invariant of the entire uncoloured Tanner graph.

    Different keys separate graph isomorphisms; equality is inconclusive.
    Check degrees 8/9 and qubit degrees <=6 distinguish the vertex classes
    for the full published codes.
    """
    from collections import Counter
    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                overlaps[(i, j)] = c
                masks = one if c == 1 else many
                masks[i] |= 1 << j
                masks[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlaps.get((i, j), 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(rows[j].bit_count(),) + feature] += 1
            profiles[j][(rows[i].bit_count(),) + feature] += 1
    features = [
        (row.bit_count(), tuple(sorted(profile.items())))
        for row, profile in zip(rows, profiles)]
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    columns = _syndrome_columns(rows, width)
    qubits = Counter()
    for col in columns:
        neighborhood = []
        while col:
            bit = col & -col
            neighborhood.append(colors[bit.bit_length() - 1])
            col ^= bit
        qubits[tuple(sorted(neighborhood))] += 1
    return (tuple(sorted(Counter(features).items())),
            tuple(sorted(qubits.items())))


def _f20_probe(rows, width, rng, passes, seed=0, own=None):
    """Actual kernel/logical witnesses; no unresolved distance cap."""
    cols = _syndrome_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _f20_outside(word, own):
            return
        best, weight = word, w

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
        # Elementary basis exchanges preserve the full kernel.
        for i in range(len(kernel)):
            for j in range(len(kernel)):
                if i != j:
                    trial = kernel[i] ^ kernel[j]
                    if 0 < trial.bit_count() < kernel[i].bit_count():
                        kernel[i] = trial
            inspect(kernel[i])
        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if own is not None and weight <= 6:
            break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _f20_outside(best, own)
    return best


@lru_cache(maxsize=1)
def _f20_catalogue(G, count):
    """Deterministic pure portfolio, with memoization only for build cost.

    Cached outputs are immutable. Slot publication cannot alter the
    catalogue or affect another slot's choice.
    """
    from itertools import combinations
    if G.q != 20:
        raise ValueError("affine core-eight portfolio requires order twenty")
    q, e = G.q, G.e

    def order(g):
        power = e
        for n in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return n
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    units = []
    for r, s in combinations(G.elements[1:], 2):
        cell = [e, r, s]
        rows, width = _f20_rows(G, [[cell]])
        if gf2_rank(rows) == q:
            spectrum = tuple(sorted(
                (orders[r], orders[s], orders[G.mul(G.inv(r), s)])))
            units.append((spectrum, cell))
    if not units:
        raise ValueError("no verified unit trinomials")

    cores = []
    for g in G.elements[1:]:
        for h in G.elements[1:]:
            if g == h or G.mul(g, h) == G.mul(h, g):
                continue
            C = [[[e], [e], [e]], [[e], [g], [h]]]
            rows, width = _f20_rows(G, C)
            if _f20_girth(rows, width) >= 8:
                cores.append((g, h, (orders[g], orders[h],
                                    orders[G.mul(G.inv(g), h)])))
    if not cores:
        raise ValueError("no verified girth-eight matching core")

    profiles = (
        ((1, 1, 1, 3, 0), (1, 1, 1, 0, 3)),
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
    )
    spectra = ((5, 5, 5), (4, 4, 5),
               (2, 4, 4), (2, 2, 5), None)
    high = [g for g in G.elements if orders[g] in (4, 5)]
    bank, component_keys = [], set()

    for card in range(20):
        profile = card // 10
        policy = card % 5
        core_kind = (5, 4, 4) if card % 10 < 5 else (4, 4, 5)
        core_pool = [c for c in cores if c[2] == core_kind] or cores
        triple_pool = [cell for spectrum, cell in units
                       if spectra[policy] is None
                       or spectrum == spectra[policy]]
        if not triple_pool:
            triple_pool = [cell for _, cell in units]
        local = random.Random(BASE_SEED ^ (0x205807 + 7919 * card))
        records, fillings = [], set()
        for trial in range(64):
            support = profiles[profile]
            C = [[[] for _ in range(5)] for _ in range(2)]
            for i, row in enumerate(support):
                for j, terms in enumerate(row):
                    anchor = G.pick(local)
                    if terms == 1:
                        C[i][j] = [anchor]
                    elif terms == 2:
                        relative = local.choice(high)
                        C[i][j] = [anchor, G.mul(anchor, relative)]
                    elif terms == 3:
                        C[i][j] = [
                            G.mul(anchor, g)
                            for g in local.choice(triple_pool)]
            g, h, _ = local.choice(core_pool)
            matching = (0, 1, 2) if profile == 0 else (2, 3, 4)
            for j, voltage in zip(matching, (e, g, h)):
                C[0][j], C[1][j] = [e], [voltage]
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            left, width = _f20_rows(G, C)
            right, _ = _f20_rows(G, C, True)
            if gf2_rank(left) != 2*q or gf2_rank(right) != 2*q:
                continue
            quality = _f20_floor(left, width)
            if quality is None:
                continue
            signature = _f20_key(left, width)
            records.append((quality, trial, C, left, right, signature))
        records.sort(key=lambda r: (-r[0][0], r[0][1], r[1]))
        kept = 0
        for quality, trial, C, left, right, signature in records:
            if signature in component_keys:
                continue
            lw = _f20_probe(left, 5*q, local, 12)
            if not lw:
                continue
            rw, source = 0, lw
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length() - 1, q)
                rw |= 1 << (j*q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            # Verify the inversion transfer against the actual right lift.
            assert all((row & rw).bit_count() % 2 == 0 for row in right)
            component_keys.add(signature)
            bank.append((quality[0], quality[1], lw.bit_count(),
                         card, trial, C, lw, rw))
            kept += 1
            if kept >= 3:
                break
    if len(bank) < 5:
        raise ValueError("too few screened affine constituents")
    bank.sort(key=lambda r: (-r[0], -min(r[2], 8), r[1], r[3], r[4]))

    plans = sorted(
        (-min(a[0], b[0]), -min(a[2], b[2], 8),
         a[1] + b[1], ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))
    graph_keys, results = set(), []
    local = random.Random(BASE_SEED ^ 0x20C058)
    tested = 0

    def embedded_seed(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(5):
            if kind == "X":
                word = record[7] << (block * 5*q)
            else:
                word, source = 0, record[6]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _f20_outside(word, own):
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

    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[5], b[5]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(5))
        assert wx == wz == 9
        assert max(wx, wz) <= min(W_MAX, 9, 12)
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (275*n <= 1000*k < 325*n):
            continue
        signature = _f20_key(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified affine lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
        bx, bz = _f20_basis(hx), _f20_basis(hz)
        xseed = embedded_seed(b, "X", hz, hx, bx)
        zseed = embedded_seed(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        xword = _f20_probe(hz, n, local, 3, xseed, bx)
        zword = _f20_probe(hx, n, local, 3, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        frozen_A = tuple(tuple(tuple(cell) for cell in row) for row in A)
        frozen_B = tuple(tuple(tuple(cell) for cell in row) for row in B)
        quality = (k*reading*reading/n, min(a[0], b[0]), -a[1]-b[1])
        results.append((quality, frozen_A, frozen_B, k, reading,
                        (a[3], b[3]), (a[0], b[0])))
        if len(results) >= count:
            break
        if tested >= max(192, 6*count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results)


def _ce48_cost(rows, width):
    """Verified short-incidence-cycle pressure, not a distance bound."""
    squares = 0
    adjacency = [0] * len(rows)
    internal = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            squares += overlap * (overlap - 1) // 2
    for col in _syndrome_columns(rows, width):
        d = col.bit_count()
        internal += d * (d - 1) * (d - 2) // 6
        bits = col
        while bits:
            bit = bits & -bits
            adjacency[bit.bit_length() - 1] |= col ^ bit
            bits ^= bit
    triangles = 0
    for i, neighbors in enumerate(adjacency):
        later = neighbors >> (i + 1)
        while later:
            bit = later & -later
            j = i + bit.bit_length()
            triangles += ((neighbors & adjacency[j]) >> (j + 1)).bit_count()
            later ^= bit
    return 24 * squares + max(0, triangles - internal)


def _ce48_core(rows, width):
    """Exact shortest cycle and connectivity of a verified matching graph."""
    adjacency = [[] for _ in rows]
    endpoints = set()
    for j, col in enumerate(_syndrome_columns(rows, width)):
        if col.bit_count() != 2:
            return None
        bit = col & -col
        a = bit.bit_length() - 1
        b = (col ^ bit).bit_length() - 1
        edge = min(a, b), max(a, b)
        if edge in endpoints:
            return None
        endpoints.add(edge)
        adjacency[a].append((b, j))
        adjacency[b].append((a, j))

    best = 0
    for root in range(len(rows)):
        paths = [None] * len(rows)
        paths[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if paths[b] is None:
                    paths[b] = paths[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = paths[a] ^ paths[b] ^ (1 << edge)
                    if word and (not best or
                                 word.bit_count() < best.bit_count()):
                        best = word
        if len(queue) != len(rows):
            return None
    if not best:
        return None
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best.bit_count(), best


@lru_cache(maxsize=1)
def _ce48_portfolio(G, count):
    """Pure deterministic optimization; only immutable outputs are cached."""
    from collections import Counter
    from hashlib import sha256
    try:
        import pynauty
    except ImportError:
        pynauty = None

    q, e = G.q, G.e
    assert q == 48 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    cap = min(9, 12, W_MAX)
    if cap < 9:
        raise ValueError("central-extension profiles require cap nine")

    a_support = ((3, 3),)
    b_supports = (
        ((1, 1, 1, 2, 1), (1, 1, 1, 1, 2)),
        ((1, 1, 1, 3, 0), (1, 1, 1, 0, 3)),
        ((1, 1, 1, 1, 2), (0, 1, 2, 2, 1)),
    )
    for support in b_supports:
        assert _no_dead_lines(a_support, support)
        assert _grid_weights(a_support, support) == 9

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def filling(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def digest(rows, width):
        return sha256(repr(_f20_key(list(rows), width)).encode("ascii")).digest()

    def certificate(hx, hz, n):
        if pynauty is None:
            return None
        rows = list(hx) + list(hz)
        m = len(rows)
        adjacency = {i: [] for i in range(m + n)}
        for i, row in enumerate(rows):
            while row:
                bit = row & -row
                j = m + bit.bit_length() - 1
                adjacency[i].append(j)
                adjacency[j].append(i)
                row ^= bit
        colors = {}
        for i, neighbors in adjacency.items():
            colors.setdefault(len(neighbors), set()).add(i)
        graph = pynauty.Graph(
            number_of_vertices=m + n, directed=False,
            adjacency_dict=adjacency,
            vertex_coloring=[colors[d] for d in sorted(colors)])
        return pynauty.certificate(graph)

    # Rotation/central generators and odd y-cosets generate the full
    # extension. Screen their ACTUAL matching graphs before using them.
    cores = []
    for a in (1, 2):
        for c in (1, 3, 5, 7):
            g = (a, 0, c)
            for d in range(3):
                for t in range(8):
                    h = (d, 1, t)
                    C = [[[e], [e], [e]], [[e], [g], [h]]]
                    rows, width = _f20_rows(G, C)
                    info = _ce48_core(rows, width)
                    if info is not None and info[0] >= 8:
                        cores.append((info[0], g, h))
    if not cores:
        raise ValueError("no connected matching core with girth at least eight")
    peak = max(record[0] for record in cores)
    preferred = [record for record in cores if record[0] == peak]

    nonidentity = [g for g in G.elements if g != e]

    def triple(local, card, j):
        """Central-coordinate and noncommutation prescriptions."""
        policy = card % 5
        parity = (card // 5 + j) % 2
        for _ in range(128):
            c = local.choice([t for t in range(8) if t % 2 == parity])
            t = local.randrange(8)
            if policy in (0, 2, 4):
                g = (local.choice((1, 2)), 0, c)
                h = (local.randrange(3), 1, t)
            elif policy == 1:
                g = (local.randrange(3), 1, c)
                h = (local.randrange(3), 1, t)
            else:
                g, h = local.choice(nonidentity), local.choice(nonidentity)
            if g == h or G.mul(g, h) == G.mul(h, g):
                continue
            anchor = G.pick(local)
            return [G.mul(anchor, z) for z in (e, g, h)]
        raise ValueError("noncommuting central-coordinate prescription empty")

    def draw(local, side, card):
        if side == "A":
            return [[triple(local, card, j) for j in range(2)]]
        support = b_supports[card % 3]
        C = [[[] for _ in range(5)] for _ in range(2)]
        for i, row in enumerate(support):
            for j, terms in enumerate(row):
                if terms == 1:
                    C[i][j] = [G.pick(local)]
                elif terms == 2:
                    anchor = G.pick(local)
                    pool = [g for g in nonidentity
                            if g[1] == 1 or g[2] % 2]
                    C[i][j] = [anchor, G.mul(anchor, local.choice(pool))]
                elif terms == 3:
                    C[i][j] = triple(local, card + i, j)
        if card % 3 != 2:
            pool = preferred if card % 4 else cores
            _, g, h = local.choice(pool)
            for j, voltage in enumerate((e, g, h)):
                C[0][j], C[1][j] = [e], [voltage]
        return C

    def make_bank(side):
        width = (2 if side == "A" else 5) * q
        checks = (1 if side == "A" else 2) * q
        bank, proposed, multiplicities = [], set(), Counter()
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0xCE4801 + 104729 * card
                             + (1000003 if side == "B" else 0)))
            raw = []
            for trial in range(64):
                C = draw(local, side, card)
                key = filling(C)
                if key in proposed:
                    continue
                proposed.add(key)
                rows, actual_width = _f20_rows(G, C)
                assert actual_width == width
                rank = gf2_rank(rows)
                if rank < checks - 2:
                    continue
                if _short_pairs(_syndrome_columns(rows, width)) is None:
                    continue
                raw.append(dict(
                    grid=C, rows=rows, rank=rank,
                    cost=_ce48_cost(rows, width), card=card, trial=trial))
            raw.sort(key=lambda r: (r["cost"], -r["rank"], r["trial"]))
            shortlist = raw[:6]
            for record in shortlist:
                word = _f20_probe(record["rows"], width, local, 4)
                record["word"] = word
                record["reading"] = word.bit_count() if word else 0
            shortlist = [r for r in shortlist if r["word"]]
            if not shortlist:
                continue
            shortlist.sort(key=lambda r:
                           (-r["reading"], r["cost"], -r["rank"]))
            cursor = shortlist[0]
            repaired = []

            # Change an actual witness block, not a gauge-only orbit.
            # Every trial is lifted and screened again after the edit.
            for trial in range(12):
                C = [[list(cell) for cell in row]
                     for row in cursor["grid"]]
                touched = [j for j in range(len(C[0]))
                           if (cursor["word"] >> (j*q)) & ((1 << q) - 1)]
                j = local.choice(touched)
                endpoints = [i for i in range(len(C)) if C[i][j]]
                i = local.choice(endpoints)
                cell = C[i][j]
                position = local.randrange(len(cell))
                options = [g for g in G.elements if g not in cell]
                cell[position] = local.choice(options)
                key = filling(C)
                if key in proposed:
                    continue
                proposed.add(key)
                rows, _ = _f20_rows(G, C)
                rank = gf2_rank(rows)
                if rank < checks - 2:
                    continue
                if _short_pairs(_syndrome_columns(rows, width)) is None:
                    continue
                word = _f20_probe(rows, width, local, 4)
                if not word:
                    continue
                record = dict(
                    grid=C, rows=rows, rank=rank, word=word,
                    reading=word.bit_count(), cost=_ce48_cost(rows, width),
                    card=card, trial=64 + trial)
                repaired.append(record)
                if (record["reading"], -record["cost"]) > (
                        cursor["reading"], -cursor["cost"]):
                    cursor = record

            finalists = shortlist + repaired
            finalists.sort(key=lambda r:
                           (-r["reading"], r["cost"], -r["rank"], r["trial"]))
            kept = 0
            for record in finalists:
                signature = digest(record["rows"], width)
                if multiplicities[signature] >= 2:
                    continue
                word = _f20_probe(
                    record["rows"], width, local, 20, record["word"])
                if not word:
                    continue
                right, _ = _f20_rows(G, record["grid"], True)
                transferred, source = 0, word
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    transferred |= 1 << (
                        j*q + G.idx(G.inv(G.elements[t])))
                    source ^= bit
                assert all((r & transferred).bit_count() % 2 == 0
                           for r in right)
                assert gf2_rank(right) == record["rank"]
                multiplicities[signature] += 1
                record["grid"] = freeze(record["grid"])
                record["word"], record["right"] = word, transferred
                record["reading"] = word.bit_count()
                bank.append(record)
                kept += 1
                if kept >= 2:
                    break
        bank.sort(key=lambda r:
                  (-r["reading"], r["cost"], r["card"], r["trial"]))
        return bank

    aa, bb = make_bank("A"), make_bank("B")
    if not aa or not bb:
        raise ValueError("central-extension constituent optimization exhausted")

    def descend(word, stabilizers):
        if not word:
            return 0
        for _ in range(3):
            starting = word
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            moves = sorted(
                (row for row in stabilizers
                 if (row & word).bit_count() >= 2),
                key=lambda row: -(row & word).bit_count())[:48]
            starting_pair = word
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    trial = starting_pair ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        return word

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(2 if kind == "X" else 5):
            if kind == "X":
                word = record["right"] << (block * 5*q)
            else:
                word, source = 0, record["word"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _f20_outside(word, own):
                continue
            word = descend(word, stabilizers)
            assert word and _f20_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"], a["cost"] + b["cost"],
         int(a["card"] == b["card"]), ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))
    invariants, certificates, results = set(), set(), []
    tested = 0
    for _, _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sum(sa[i][j] for i in range(1))
                 for row in sb for j in range(2))
        assert max(wx, wz) <= cap and _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or not (225*n <= 1000*k < 275*n):
            continue
        signature = digest(hx + hz, n)
        if signature in invariants:
            continue
        canonical = certificate(hx, hz, n)
        if canonical is not None and canonical in certificates:
            continue
        invariants.add(signature)
        if canonical is not None:
            certificates.add(canonical)
        tested += 1
        if tested > max(320, 6*count):
            break
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap
        bx, bz = _f20_basis(hx), _f20_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0xCE48E1 + 65537*ai + bi))
        xword = _f20_probe(hz, n, local, 4, xseed, bx)
        zword = _f20_probe(hx, n, local, 4, zseed, bz)
        xword, zword = descend(xword, hx), descend(zword, hz)
        assert xword and zword
        assert _f20_outside(xword, bx) and _f20_outside(zword, bz)
        assert all((row & xword).bit_count() % 2 == 0 for row in hz)
        assert all((row & zword).bit_count() % 2 == 0 for row in hx)
        reading = min(xword.bit_count(), zword.bit_count())
        quality = (k * reading * reading / n, reading, k,
                   -a["cost"] - b["cost"])
        results.append((
            quality, A, B, k, reading, (a["card"], b["card"]),
            (a["reading"], b["reading"]),
            (a["rank"], b["rank"])))
        if len(results) >= max(count + 12, 2*count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one optimized immutable design, without post-screen draws."""
    catalogue = _ce48_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("central-extension portfolio exhausted")
    quality, A, B, k, reading, cards, components, ranks = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "constituent_witness_upper_bounds": components,
            "constituent_ranks": ranks, "design_cards": cards,
            "quality": quality, "complete_kernel_exclusion_through": 4}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without modifying cached designs."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Reserved positions separated by the pendant catalogue's constituent/full check-intersection signatures remain evaluator-distinct: the measured sixty-slot table retained only twenty-six distinct codes.
# REFUTED: Twenty affine support/spectrum cards provide broad evaluator-distinct diversity: the current sixty-slot evaluation retained only five classes, all five measured at k=180,d30s=7.
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
