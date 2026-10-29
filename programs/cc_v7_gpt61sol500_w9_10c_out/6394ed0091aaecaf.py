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
CONCEPT = ("Dic9, order 36, A=(2,3), B=(2,4): exact normal-C9 invariant-word screening followed by independently refined saturated constituents should supply five novel distance readings >=15 in cell 0.150; five rank-bound finalists would score 28.125.")
GROUP = ("Jump from cyclic Z8 to Dic9: x^18=1, y^2=x^9, y*x*y^-1=x^-1. The related order-36, centre-two, twelve-class (2,3)x(2,4) bed reached fitness 24.3, while the 0.150 frontier is 25.2. These measurements motivate a better constituent search, not a distance guarantee.")
SHAPES = ("A=(2,3), B=(2,4) give 108-variable/72-check and 144-variable/72-check constituents. Factor 16 permits q<=37; fixed q=36 gives n=576 and kb=72, guaranteed rate 0.125. Verified ranks retain only [0.125,0.175). Parent distance-six minima occupied sector-1 Acol/Bcol blocks broadly; screen both constituents and complete logical witnesses.")
FILLING = ("Four A supports and five B supports define twenty structural pair cards, with two mixed-coset policies per constituent. Every row totals five and every column has at most four terms, so BOTH CSS row-weight sums are <=9, also below twelve. Verified constituent lifts exclude kernels through four and enumerate the complete normal-C9 invariant subspace before equal-budget information-set refinement. Full verified lifts enforce actual rate, CSS, logical-weight-four rejection and unequal uncoloured Tanner invariants. Embedded nontrivial witnesses undergo stabilizer descent and a full-matrix probe. All larger witness readings are upper bounds; archive novelty and distance-fifteen survival remain hypotheses.")
FAMILY_NAME = "r015_dic9_23x24_invariant_independent_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic9 has order 36")
    # x^18=1, y^2=x^9, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(18)]

    def red(g):
        a, b = g
        return ((a + 9 * (b // 2)) % 18, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 9*b*d) % 18,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 18, 0) if b == 0 else ((a + 9) % 18, 1)

    def idx(g):
        a, b = red(g)
        return 18*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def _inactive_z8_catalogue(rng, G, idx):
    """Historical Z8 reference only; the Dic9 entry point never calls this."""
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


def _d9_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _d9_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _d9_probe(rows, width, rng, passes, seed=0, own=None):
    """Return an explicit kernel/logical witness, never a lower bound."""
    cols = _syndrome_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if own is not None and not _d9_outside(word, own):
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

        # Elementary exchanges preserve the complete kernel.
        rng.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            for j in range(len(kernel)):
                if i == j:
                    continue
                trial = word ^ kernel[j]
                if 0 < trial.bit_count() < word.bit_count():
                    word = trial
            kernel[i] = word
            inspect(word)

        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        for _ in range(2):
            if not best:
                break
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break
        if weight <= 8:
            break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _d9_outside(best, own)
    return best


def _d9_component(G, grid, side):
    """Expose only the evaluator's verified left/right constituent."""
    zero = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    return [row & mask for row in rows], width


def _d9_invariant_word(G, cols, nc):
    """Minimum word in the COMPLETE normal-C9 invariant subspace.

    Each block has four C9 cosets. Elimination uses syndromes extracted
    from the verified matrix; every combination of its invariant kernel
    basis is enumerated. This says nothing about non-invariant words.
    """
    orbit_words, orbit_syndromes = [], []
    for j in range(nc):
        for b in range(2):
            for parity in range(2):
                word = syndrome = 0
                for a in range(parity, 18, 2):
                    position = j*G.q + G.idx((a, b))
                    word |= 1 << position
                    syndrome ^= cols[position]
                orbit_words.append(word)
                orbit_syndromes.append(syndrome)

    pivots, basis = {}, []
    for j, syndrome in enumerate(orbit_syndromes):
        word = 1 << j
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

    best, weight = 0, len(orbit_words) + 1
    previous = current = 0
    for number in range(1, 1 << len(basis)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous).bit_length() - 1
        current ^= basis[changed]
        if current.bit_count() < weight:
            best, weight = current, current.bit_count()
        previous = gray
    actual = 0
    while best:
        bit = best & -best
        actual ^= orbit_words[bit.bit_length() - 1]
        best ^= bit
    return actual


def _d9_low4(checks, stabilizers, width):
    """Exact existence test for a nontrivial logical of weight <=4."""
    own = _d9_span(stabilizers)
    cols = _syndrome_columns(checks, width)
    representatives = {0: 0}

    def inspect(syndrome, word):
        old = representatives.get(syndrome)
        if old is None:
            representatives[syndrome] = word
            return False
        return _d9_outside(word ^ old, own)

    for i, syndrome in enumerate(cols):
        if inspect(syndrome, 1 << i):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            if inspect(syndrome ^ cols[j], (1 << i) | (1 << j)):
                return True
    # Every support <=4 splits into two supports <=2. One representative
    # per syndrome suffices because the stabilizer space is linear.
    return False


def _d9_signature(rows, width):
    """Conservative invariant of the complete uncoloured Tanner graph.

    Unequal signatures certify different check/qubit Tanner graphs;
    equality is discarded without claiming a complete isomorphism test.
    """
    from collections import Counter
    from hashlib import sha256

    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0]*m, [0]*m
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency = one if overlap == 1 else many
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            common = (
                (rows[i] & rows[j]).bit_count(),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(weights[j],) + common] += 1
            profiles[j][(weights[i],) + common] += 1
    cols = _syndrome_columns(rows, width)
    invariant = (
        width, m,
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted((weights[i], tuple(sorted(profiles[i].items())))
                     for i in range(m))))
    return sha256(repr(invariant).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _d9_portfolio(G, count):
    """Pure deterministic portfolio; publication data are immutable."""
    if G.q != 36:
        raise ValueError("Dic9 portfolio requires order 36")
    aprofiles = (
        ((1, 2, 2), (2, 1, 2)),
        ((1, 1, 3), (2, 2, 1)),
        ((1, 1, 3), (1, 3, 1)),
        ((1, 2, 2), (1, 2, 2)),
    )
    bprofiles = (
        ((1, 1, 1, 2), (1, 1, 2, 1)),
        ((1, 1, 1, 2), (1, 1, 1, 2)),
        ((1, 1, 3, 0), (1, 1, 0, 3)),
        ((2, 1, 2, 0), (0, 2, 1, 2)),
        ((1, 2, 2, 0), (1, 0, 2, 2)),
    )
    for a in aprofiles:
        for b in bprofiles:
            assert _no_dead_lines(a, b)
            assert all(sum(row) == 5 for row in a + b)
            wx = max(sum(row) + sum(b[i][j] for i in range(2))
                     for row in a for j in range(4))
            wz = max(sum(row) + sum(a[i][j] for i in range(2))
                     for row in b for j in range(3))
            assert max(wx, wz) <= min(9, 12, W_MAX)

    rotations = [(a, 0) for a in range(1, 18) if a not in (3, 6, 9, 12, 15)]
    crosses = [(a, 1) for a in range(18)]

    def draw(profile, policy, local, trial):
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, terms in enumerate(counts):
                if not terms:
                    row.append([])
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    degree = sum(r[j] for r in profile)
                    # A degree-two binomial must not install an isolated
                    # order-four cycle. Cubic/quartic columns can cross.
                    mixed = degree > 2 and (
                        policy == 1 or (i + j + trial) % 2 == 0)
                    relative = local.choice(crosses if mixed else rotations)
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    rotation = local.choice(rotations)
                    cross = local.choice(crosses)
                    if (policy + i + j + trial) % 2:
                        second = G.mul(anchor, cross)
                        cell = [anchor, second, G.mul(second, rotation)]
                    else:
                        cell = [anchor, G.mul(anchor, rotation),
                                G.mul(anchor, cross)]
                assert len(set(cell)) == terms
                row.append(cell)
            grid.append(row)
        return grid

    def bank(side, profiles):
        entries = []
        for profile_number, profile in enumerate(profiles):
            for policy in range(2):
                local = random.Random(
                    BASE_SEED ^ (0xD936 + 7919*profile_number
                                 + 104729*policy + 65537*(side == "B")))
                raw, fillings = [], set()
                for trial in range(128):
                    grid = draw(profile, policy, local, trial)
                    filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                    for row in grid for cell in row)
                    if filling in fillings:
                        continue
                    fillings.add(filling)
                    rows, width = _d9_component(G, grid, side)
                    cols = _syndrome_columns(rows, width)
                    pairs = _short_pairs(cols)
                    if pairs is None:
                        continue
                    del pairs
                    rank = gf2_rank(rows)
                    if rank < 2*G.q - 4:
                        continue
                    invariant_word = _d9_invariant_word(
                        G, cols, len(profile[0]))
                    invariant_reading = (
                        invariant_word.bit_count() if invariant_word else width + 1)
                    cycles = 0
                    for i, row in enumerate(rows):
                        for other in rows[:i]:
                            shared = (row & other).bit_count()
                            cycles += shared*(shared - 1)//2
                    cheap = (int(invariant_reading <= 9),
                             -invariant_reading, cycles, -rank, trial)
                    raw.append((cheap, grid, rows, width, invariant_word))
                raw.sort(key=lambda r: r[0])
                refined = []
                for cheap, grid, rows, width, seed in raw[:12]:
                    word = _d9_probe(rows, width, local, 4, seed)
                    if word:
                        refined.append(dict(
                            grid=grid, rows=rows, width=width, word=word,
                            cycles=cheap[2], profile=profile_number,
                            policy=policy, trial=cheap[-1]))
                refined.sort(key=lambda r: (
                    r["word"].bit_count(), -r["cycles"], -r["trial"]),
                    reverse=True)
                finalists = refined[:4]
                # Every retained finalist receives the same longer budget.
                for record in finalists:
                    record["word"] = _d9_probe(
                        record["rows"], record["width"], local, 24,
                        record["word"])
                finalists.sort(key=lambda r: (
                    r["word"].bit_count(), -r["cycles"], -r["trial"]),
                    reverse=True)
                entries.extend(finalists[:3])
        if len(entries) < 5:
            raise ValueError("too few weight-four-clean Dic9 constituents")
        return entries

    aa, bb = bank("A", aprofiles), bank("B", bprofiles)
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["word"].bit_count(), b["word"].bit_count())
            card = 5*a["profile"] + b["profile"]
            plans.append((-min(weak, 20),
                          a["cycles"] + b["cycles"], card, ai, bi))
    plans.sort()

    def reduce_word(word, stabilizers):
        for _ in range(3):
            starting = word
            ranked = sorted(stabilizers,
                            key=lambda row: (starting ^ row).bit_count())
            for row in ranked:
                trial = starting ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            short = ranked[:40]
            for i, first in enumerate(short):
                for second in short[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        return word

    def quantum_reading(a, b, hx, hz, n, local):
        readings = []
        q, ca, cb = G.q, A_SHAPE[1], B_SHAPE[1]
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = _d9_span(stabilizers)
            seeds = []
            for block in range(ca if kind == "X" else cb):
                if kind == "X":
                    word = b["word"] << (block*cb*q)
                else:
                    word, source = 0, a["word"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j*cb + block)*q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if _d9_outside(word, own):
                    seeds.append(reduce_word(word, stabilizers))
            if not seeds:
                continue
            seed = min(seeds, key=int.bit_count)
            word = _d9_probe(checks, n, local, 1, seed, own)
            word = reduce_word(word, stabilizers)
            assert word and _d9_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            readings.append(word.bit_count())
        return min(readings) if readings else None

    results, graph_keys = [], set()
    local = random.Random(BASE_SEED ^ 0x150D936)
    wanted = count + max(20, count//2)
    for negative_weak, cycles, card, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 576 and k > 0
                and 125*n <= 1000*k < 175*n):
            continue
        if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
            continue
        if (_d9_low4(hz, hx, n) or _d9_low4(hx, hz, n)):
            continue
        signature = _d9_signature(hx + hz, n)
        if signature in graph_keys:
            continue
        reading = quantum_reading(a, b, hx, hz, n, local)
        if reading is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified Dic9 portfolio failed CSS")
        graph_keys.add(signature)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        # Capping the selection reading prevents poorly searched large
        # witnesses from dominating through an unresolved upper bound.
        quality = (k*min(reading, 18)**2/n, min(reading, 18),
                   -negative_weak, -cycles, k)
        results.append((quality, freeze(A), freeze(B), k, reading,
                        card, (a["policy"], b["policy"])))
        if len(results) >= wanted:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup; no slot mutates another slot's immutable design."""
    portfolio = _d9_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct target-rate Dic9 portfolio exhausted")
    quality, A, B, k, reading, card, policies = portfolio[idx]
    return dict(A=A, B=B, screen_k=k, quality=quality,
                logical_witness_upper_bound=reading,
                structural_card=card, coset_policies=policies)


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh list from the selected immutable complete cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Multisupport Z8 six-count prioritization supplies distance-seven finalists: the shown v4 program's five measured finalists all had d30s=6.
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
