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
CONCEPT = ("Metacyclic C5 semidirect_-1 C2, order 10, A=B=(3,7): verified long-cycle triangle cores with four cubic columns should supply five novel measured distance readings >=8 in cell 0.300; quantum distance eight remains a falsifiable measurement hypothesis.")
GROUP = ("Keep the semidirect-product construction rule, resizing its parameters from (m,r,s)=(5,4,2) to (5,2,4). The new factor 58 makes order twenty invalid; order ten is the largest admissible lift and gives n=580. The parent measured five [580,180,7] codes, whereas the related cyclic order-ten (3,7)x(3,7) bed reached only fitness 10.6. This untried nonabelian bed is not assumed to inherit either distance.")
SHAPES = ("Equal (3,7) constituents have 70 variables and 30 checks. With q=10, n=580, kb=160 and guaranteed rate 8/29=0.275862, they target cell 0.300 near its lower edge; verified full ranks enforce [0.275,0.325). Three check rows and four cubic columns replace the parent's two-row geometry. Parent minima were entirely sector-1 Acol/Bcol witnesses, often involving Acol1/2, with no sector-2 Arow/Brow minima reported.")
FILLING = ("Twenty balanced cubic-incidence/core prescriptions use three degree-two triangle columns and four degree-three columns. Every constituent row totals six, so BOTH LP maxima are nine, also below twelve. Core cycles are screened with the verified lift for girth >=10. Constituent screens exactly exclude kernels through six when attainable; shortlisted clean constituents receive an anchored exhaustive weight-seven test. Explicit left/right kernel and full-code logical witnesses rank candidates. An immutable conservative full-Tanner-invariant gate rejects repeated designs, while complete lifts enforce target-band ranks, CSS and kernel exclusion through four. Lower-floor fallbacks are recorded, and constituent screens do not certify mixed-sector quantum distance.")
FAMILY_NAME = "r030_m5r2s4_q10_37x37_triangle_fourcubic_kernel7_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect_-1 C2 requires order ten")
    # Same semidirect-product rule, resized for factor 58.
    # (a,b)=x^a*y^b; 4^2=1 mod 5.
    m, r, s = 5, 2, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m) * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m) * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 7)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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


@lru_cache(maxsize=1)
def _f20_join_portfolio(G, count):
    """Immutable portfolio of optimized structural prescriptions.

    Every matrix and syndrome used for screening comes from the verified
    scaffold. Memoization changes build cost, not publication order.
    """
    from itertools import combinations

    q, e = G.q, G.e
    if q != 20:
        raise ValueError("affine join portfolio requires order twenty")
    width = 5 * q

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != e]
    ratios = {
        degree: [g for g in nonzero if orders[g] == degree]
        for degree in (2, 4, 5)
    }

    def core_data(g, h):
        """Actual graph, shortest paths and shortest cycle of a core."""
        grid = [[[e], [e], [e]], [[e], [g], [h]]]
        rows, core_width = _f20_rows(G, grid)
        cols = _syndrome_columns(rows, core_width)
        adjacency = [[] for _ in rows]
        endpoints = set()
        for j, syndrome in enumerate(cols):
            if syndrome.bit_count() != 2:
                return None
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            edge = (min(a, b), max(a, b))
            if edge in endpoints:
                return None
            endpoints.add(edge)
            adjacency[a].append((b, j))
            adjacency[b].append((a, j))

        distances, paths = [], []
        cycle = 0
        for root in range(len(rows)):
            dist = [10000] * len(rows)
            path = [0] * len(rows)
            dist[root] = 0
            queue = [root]
            for a in queue:
                for b, edge in adjacency[a]:
                    if dist[b] == 10000:
                        dist[b] = dist[a] + 1
                        path[b] = path[a] ^ (1 << edge)
                        queue.append(b)
                    else:
                        word = path[a] ^ path[b] ^ (1 << edge)
                        if word and (not cycle or
                                     word.bit_count() < cycle.bit_count()):
                            cycle = word
            distances.append(tuple(dist))
            paths.append(tuple(path))
        if not cycle or cycle.bit_count() < 8:
            return None
        assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
        return (g, h, tuple(cols), tuple(distances),
                tuple(paths), cycle)

    cores = [[], []]
    for g, h in combinations(nonzero, 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        data = core_data(g, h)
        if data is not None:
            cores[int(orders[g] == orders[h])].append(data)
    if not any(cores):
        raise ValueError("no verified affine matching core of girth eight")
    for kind in range(2):
        if not cores[kind]:
            cores[kind] = list(cores[1 - kind])

    triangle_spectra = ((5, 5, 5), (4, 4, 5),
                        (2, 4, 4), (2, 2, 5), None)
    triangles = [[] for _ in triangle_spectra]
    for r, s in combinations(nonzero, 2):
        spectrum = tuple(sorted(
            (orders[r], orders[s], orders[G.mul(G.inv(r), s)])))
        for policy, requested in enumerate(triangle_spectra):
            if requested is None or spectrum == requested:
                triangles[policy].append((r, s))
    for policy in range(4):
        if not triangles[policy]:
            triangles[policy] = list(triangles[4])

    def join_screen(rows, data):
        """Exact constituent exclusion through seven, with a witness.

        Core columns have degree two and the other forty columns degree
        three. A kernel contains an even number of cubic variables.
        Thus weights <=7 have zero, two, four or six cubic variables.

        Regular constituent actions let each nonempty cubic support be
        translated to contain an identity-coordinate cubic variable.
        This symmetry is NOT assumed for complete quantum matrices.
        """
        cols = _syndrome_columns(rows, width)
        if tuple(cols[:3*q]) != data[2]:
            raise ValueError("verified reinforcement changed the core")
        if _short_pairs(cols) is None:
            return None
        cubic = list(range(3*q, width))
        if any(cols[j].bit_count() != 3 for j in cubic):
            raise ValueError("join screen requires exactly two cubic blocks")
        anchors = (3*q + G.idx(e), 4*q + G.idx(e))
        distances, paths = data[3], data[4]
        best = data[5]
        memo = {}

        def tjoin(boundary, budget):
            """Exact minimum matching in the graph-distance metric."""
            if not boundary:
                return (0, 0)
            if budget < 0 or boundary.bit_count() > 2 * budget:
                return None
            key = (boundary, budget)
            if key in memo:
                return memo[key]
            bit = boundary & -boundary
            a = bit.bit_length() - 1
            rest = boundary ^ bit
            choices = rest
            answer = None
            while choices:
                other = choices & -choices
                b = other.bit_length() - 1
                distance = distances[a][b]
                if distance <= budget:
                    tail = tjoin(rest ^ other, budget - distance)
                    if tail is not None:
                        cost = distance + tail[0]
                        word = paths[a][b] ^ tail[1]
                        if answer is None or cost < answer[0]:
                            answer = (cost, word)
                choices ^= other
            memo[key] = answer
            return answer

        # Two cubic variables plus at most five core edges.
        for anchor in anchors:
            for other in cubic:
                if other == anchor:
                    continue
                budget = min(5, best.bit_count() - 3)
                result = tjoin(cols[anchor] ^ cols[other], budget)
                if result is not None:
                    word = (1 << anchor) ^ (1 << other) ^ result[1]
                    if word.bit_count() < best.bit_count():
                        best = word
                        if best.bit_count() <= 6:
                            return None

        # Exact pure-cubic dependencies through six.
        triples = {}
        for a, b, c in combinations(cubic, 3):
            syndrome = cols[a] ^ cols[b] ^ cols[c]
            word = (1 << a) | (1 << b) | (1 << c)
            previous = triples.get(syndrome)
            if previous is not None:
                witness = previous ^ word
                assert witness and witness.bit_count() <= 6
                return None
            triples[syndrome] = word

        # Four cubic variables plus at most three core edges.
        for anchor in anchors:
            others = [j for j in cubic if j != anchor]
            for a, b, c in combinations(others, 3):
                boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                budget = min(3, best.bit_count() - 5)
                if boundary.bit_count() > 2 * budget:
                    continue
                result = tjoin(boundary, budget)
                if result is not None:
                    word = ((1 << anchor) | (1 << a)
                            | (1 << b) | (1 << c)) ^ result[1]
                    if word.bit_count() < best.bit_count():
                        best = word
                        if best.bit_count() <= 6:
                            return None

        # All <=6 cases are now excluded. If a seven-word is already
        # known, its existence cannot be undone by more screening.
        if best.bit_count() == 7:
            return (7, best)

        # Six cubic variables plus one core edge. Compare anchored
        # cubic triples with every cubic triple; overlaps are harmless
        # because the resulting actual support is checked explicitly.
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
                            assert all((row & witness).bit_count() % 2 == 0
                                       for row in rows)
                            return (7, witness)

        # Zero-cubic words are core cycles, already excluded through
        # seven. Eight-cubic words cannot have total weight <=7.
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return (8, best)

    def cycle_penalty(rows):
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                shared = (row & other).bit_count()
                total += shared * (shared - 1) // 2
        return total

    bank, component_keys = [], set()
    for card in range(20):
        profile, remainder = divmod(card, 10)
        core_kind, policy = divmod(remainder, 5)
        local = random.Random(BASE_SEED ^ (0x207B91 + 104729 * card))
        records, fillings = [], set()
        for trial in range(160):
            data = local.choice(cores[core_kind])
            g, h = data[:2]
            C = [[[e], [e], [e], [], []],
                 [[e], [g], [h], [], []]]

            if profile == 0:
                # Normalize the monomial endpoint of each cubic column.
                # The remaining anchors are genuine relative voltages.
                if policy == 0:
                    first_pool = second_pool = ratios[2]
                elif policy == 1:
                    first_pool = second_pool = ratios[4]
                elif policy == 2:
                    first_pool = second_pool = ratios[5]
                elif policy == 3:
                    first_pool, second_pool = ratios[2], ratios[4]
                else:
                    first_pool = second_pool = nonzero
                a, b = G.pick(local), G.pick(local)
                r, s = local.choice(first_pool), local.choice(second_pool)
                C[0][3], C[1][3] = [a, G.mul(a, r)], [e]
                C[0][4], C[1][4] = [e], [b, G.mul(b, s)]
            else:
                # No invertibility restriction: the complete verified
                # constituent screen decides whether a triangle is useful.
                r, s = local.choice(triangles[policy])
                u, v = local.choice(triangles[(policy + 1) % 5])
                C[0][3], C[1][4] = [e, r, s], [e, u, v]

            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows, _ = _f20_rows(G, C)
            if gf2_rank(rows) != 2*q:
                continue
            screened = join_screen(rows, data)
            if screened is None:
                continue
            floor, witness = screened
            penalty = cycle_penalty(rows)
            records.append((floor, penalty, trial, C, rows, witness))

        records.sort(key=lambda r: (-r[0], r[1], r[2]))
        kept = 0
        for floor, penalty, trial, C, rows, seed in records:
            signature = _f20_key(rows, width)
            if signature in component_keys:
                continue
            witness = _f20_probe(rows, width, local, 8, seed)
            if not witness or witness.bit_count() < floor:
                raise ValueError("kernel probe contradicted exact join screen")
            right_rows, _ = _f20_rows(G, C, True)
            right_word, source = 0, witness
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length() - 1, q)
                right_word |= 1 << (
                    j*q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            assert all((row & right_word).bit_count() % 2 == 0
                       for row in right_rows)
            assert gf2_rank(right_rows) == 2*q
            component_keys.add(signature)
            bank.append(dict(
                grid=C, floor=floor, penalty=penalty,
                left=witness, right=right_word,
                reading=witness.bit_count(), card=card, trial=trial))
            kept += 1
            if kept >= 5:
                break
    if len(bank) < 5:
        raise ValueError("too few distinct affine join-screened constituents")

    # Reject every pairing of the three constituents shown in the
    # current parent's table. Equal invariants are rejected conservatively.
    parent_components = (
        [
            [[(0, 1), (2, 1)], [(3, 2)], [e], [e], [e]],
            [[(0, 2)], [e, (3, 1)], [e], [(2, 0)], [(1, 3)]],
        ],
        [
            [[(4, 2), (4, 3)], [(4, 2)], [e], [e], [e]],
            [[(2, 0)], [(1, 2), (1, 3)], [e], [(1, 0)], [(1, 1)]],
        ],
        [
            [[(3, 0), (4, 0)], [(2, 1)], [e], [e], [e]],
            [[(1, 2)], [(0, 3), (4, 3)], [e], [(4, 0)], [(4, 3)]],
        ],
    )
    excluded = set()
    for ai, A in enumerate(parent_components):
        for B in parent_components[ai:]:
            hx, hz, n = lifted_matrices(G, A, B)
            excluded.add(_f20_key(hx + hz, n))

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            quality = (
                min(a["floor"], b["floor"]),
                min(a["reading"], b["reading"], 9),
                -a["penalty"] - b["penalty"],
                int(a["card"] != b["card"]))
            plans.append((quality, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def embedded(record, kind, checks, stabilizers, own):
        """Explicit nontrivial witnesses, reduced by stabilizer pairs."""
        best = 0
        for block in range(5):
            if kind == "X":
                word = record["right"] << (block * width)
            else:
                word, source = 0, record["left"]
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
                starting = word
                weight = word.bit_count()
                for i, first in enumerate(stabilizers):
                    trial = starting ^ first
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                    for second in stabilizers[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and _f20_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, tested = [], 0
    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (275*n <= 1000*k < 325*n):
            continue
        signature = _f20_key(hx + hz, n)
        if signature in excluded:
            continue
        excluded.add(signature)
        tested += 1
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified join-selected lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        bx, bz = _f20_basis(hx), _f20_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0x207C00 + 257*ai + bi))
        xword = _f20_probe(hz, n, local, 3, xseed, bx)
        zword = _f20_probe(hx, n, local, 3, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 7:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, preliminary, k)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), (a["floor"], b["floor"])))
        if len(results) >= count:
            break
        if tested >= max(320, 10*count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results)


def _r30_seven_screen(G, rows, width):
    """Exactly test weight seven AFTER exclusion through six.

    Every constituent kernel support has a regular-action translate
    containing an identity-coordinate variable in some block. Every
    seven-support splits into an anchored quartet and a triple.
    The syndromes themselves come only from the verified lift.
    """
    from itertools import combinations

    cols = _syndrome_columns(rows, width)
    triples = {}
    bits = [1 << i for i in range(width)]
    for a, b, c in combinations(range(width), 3):
        syndrome = cols[a] ^ cols[b] ^ cols[c]
        word = bits[a] | bits[b] | bits[c]
        if syndrome in triples:
            raise ValueError("weight-six exclusion contradicted")
        triples[syndrome] = word

    for block in range(width // G.q):
        anchor = block * G.q + G.idx(G.e)
        others = [i for i in range(width) if i != anchor]
        first = cols[anchor]
        for a, b, c in combinations(others, 3):
            old = triples.get(first ^ cols[a] ^ cols[b] ^ cols[c])
            if old is None:
                continue
            word = old ^ bits[anchor] ^ bits[a] ^ bits[b] ^ bits[c]
            if word:
                assert word.bit_count() == 7
                assert all((row & word).bit_count() % 2 == 0
                           for row in rows)
                return 7, word
    return 8, 0


@lru_cache(maxsize=1)
def _r30_components(G):
    """Immutable bank of optimized three-row incidence prescriptions."""
    from itertools import combinations_with_replacement, permutations

    if G.q != 10:
        raise ValueError("three-row triangle bank requires order ten")
    q, e, width = G.q, G.e, 7 * G.q

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid group element order")

    orders = {g: order(g) for g in G.elements}
    cores = [[], []]
    for g in G.elements:
        for h in G.elements:
            C = [[[e], [], [h]],
                 [[e], [e], []],
                 [[], [g], [e]]]
            rows, core_width = _f20_rows(G, C)
            girth = _f20_girth(rows, core_width)
            if girth >= 10:
                kind = int(G.mul(g, h) != G.mul(h, g))
                cores[kind].append((g, h, girth))
    if not any(cores):
        raise ValueError("no verified long-cycle triangle")
    for kind in range(2):
        if not cores[kind]:
            cores[kind] = list(cores[1 - kind])

    cubic_types = sorted(
        {(1, 1, 1)}
        | set(permutations((2, 1, 0)))
        | set(permutations((3, 0, 0))))

    def canonical(columns):
        return min(
            tuple(sorted(tuple(column[i] for i in p)
                         for column in columns))
            for p in permutations(range(3)))

    candidates = {}
    for columns in combinations_with_replacement(cubic_types, 4):
        if any(sum(c[i] for c in columns) != 4 for i in range(3)):
            continue
        key = canonical(columns)
        candidates.setdefault(key, columns)
    profiles = list(candidates.values())
    profiles.sort(key=lambda cs: (
        -sum(c == (1, 1, 1) for c in cs),
        sum(max(c) == 3 for c in cs), canonical(cs)))
    profiles = profiles[:10]
    if not profiles:
        raise ValueError("no balanced four-cubic incidence profile")

    bank, component_keys = [], set()
    for card in range(20):
        profile = profiles[(card // 2) % len(profiles)]
        core_kind = card % 2
        local = random.Random(BASE_SEED ^ (0x307810 + 104729 * card))
        raw, seen = [], set()

        for trial in range(128):
            g, h, girth = local.choice(cores[core_kind])
            C = [[[e], [], [h], [], [], [], []],
                 [[e], [e], [], [], [], [], []],
                 [[], [g], [e], [], [], [], []]]
            for j, counts in enumerate(profile, 3):
                for i, terms in enumerate(counts):
                    if not terms:
                        continue
                    labels = local.sample(G.elements, terms)
                    C[i][j] = labels

            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in seen:
                continue
            seen.add(filling)
            counts = [[len(cell) for cell in row] for row in C]
            assert all(sum(row) == 6 for row in counts)
            assert sorted(sum(counts[i][j] for i in range(3))
                          for j in range(7)) == [2, 2, 2, 3, 3, 3, 3]

            rows, _ = _f20_rows(G, C)
            if gf2_rank(rows) != 3 * q:
                continue
            quality = _f20_floor(rows, width)
            if quality is None:
                continue
            floor, sixes = quality
            penalty = sum(
                overlap * (overlap - 1) // 2
                for i, row in enumerate(rows)
                for other in rows[:i]
                for overlap in [(row & other).bit_count()])
            raw.append((floor, sixes, penalty, trial,
                        C, rows, girth))

        raw.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))
        kept = 0
        for floor, sixes, penalty, trial, C, rows, girth in raw:
            signature = _f20_key(rows, width)
            if signature in component_keys:
                continue
            seed = 0
            if floor == 7:
                floor, seed = _r30_seven_screen(G, rows, width)
            witness = _f20_probe(rows, width, local, 12, seed)
            if not witness:
                continue
            assert witness.bit_count() >= floor

            right_rows, _ = _f20_rows(G, C, True)
            if gf2_rank(right_rows) != 3 * q:
                continue
            right_word, source = 0, witness
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length() - 1, q)
                right_word |= 1 << (
                    j * q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            assert all((row & right_word).bit_count() % 2 == 0
                       for row in right_rows)
            component_keys.add(signature)
            frozen = tuple(tuple(tuple(cell) for cell in row)
                           for row in C)
            bank.append((
                floor, witness.bit_count(), sixes, penalty,
                card, trial, frozen, witness, right_word, girth))
            kept += 1
            if kept >= 5:
                break

    bank.sort(key=lambda r: (
        -r[0], -min(r[1], 10), r[2], r[3], r[4], r[5]))
    if len(bank) < 5:
        raise ValueError("too few screened three-row constituents")
    return tuple(bank)


@lru_cache(maxsize=1)
def _r30_portfolio(G, count):
    """Pure deterministic portfolio using complete evaluator matrices.

    The first twenty accepted entries are measured candidates, not
    twenty group-automorphism images. Further harness slots use later
    invariant-separated entries from the same optimized search.
    """
    bank = _r30_components(G)
    q, width = G.q, 7 * G.q
    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            preliminary = (
                min(a[0], b[0]),
                min(a[1], b[1], 10),
                -a[2] - b[2],
                -a[3] - b[3],
                int(a[4] != b[4]))
            plans.append((preliminary, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(7):
            if kind == "X":
                word = record[8] << (block * width)
            else:
                word, source = 0, record[7]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 7 + block) * q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _f20_outside(word, own):
                continue
            for _ in range(3):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == previous:
                    break
            assert word and _f20_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, graph_keys = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A = [[list(cell) for cell in row] for row in a[6]]
        B = [[list(cell) for cell in row] for row in b[6]]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(sa[i]) + sum(sb[r][j] for r in range(3))
                 for i in range(3) for j in range(7))
        wz = max(sum(sb[r]) + sum(sa[i][j] for i in range(3))
                 for r in range(3) for j in range(7))
        assert wx == wz == 9
        assert max(wx, wz) <= min(9, W_MAX, 12)
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (k > 0 and 275*n <= 1000*k < 325*n):
            continue
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        signature = _f20_key(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        if not css_ok(hx, hz):
            raise ValueError("verified three-row lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)

        bx, bz = _f20_basis(hx), _f20_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(
            BASE_SEED ^ (0x307C00 + 104729 * ai + 7919 * bi))
        xword = _f20_probe(hz, n, local, 4, xseed, bx)
        zword = _f20_probe(hx, n, local, 4, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        tested += 1
        quality = (k * reading * reading / n, preliminary, k)
        results.append((
            quality, a[6], b[6], k, reading,
            (a[4], b[4]), (a[0], b[0]), (a[9], b[9])))

        # Oversample before sorting by actual discovered logical weights.
        if len(results) >= max(count + 20, 2 * count):
            break
        if tested >= max(480, 8 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable, screened three-row design."""
    catalogue = _r30_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("three-row target-rate portfolio exhausted")
    quality, A, B, k, reading, cards, floors, girths = catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k, "quality": quality,
        "logical_witness_upper_bound": reading,
        "constituent_kernel_floors": floors,
        "verified_core_girths": girths,
        "design_cards": cards,
        "below_eight_fallback": reading < 8,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without modifying the immutable portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Reserved positions separated by the pendant catalogue's constituent/full check-intersection signatures remain evaluator-distinct: the measured sixty-slot table retained only twenty-six distinct codes.
# REFUTED: Twenty unit-trinomial/crossed-binomial spectrum cards supply twenty evaluator-distinct affine codes: the current sixty-slot table retained only five classes, all five measured at k=180,d30s=7.
# REFUTED: Exact constituent-seven join screening with broad reinforcement on the affine (2,5)x(2,5) bed supplies five measured readings >=8: all five current finalists instead measured k=180,d30s=7, and sixty slots retained only five classes.
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
