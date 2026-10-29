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
CONCEPT = ("S3 x S3, order 36, A=(2,3), B=(2,4): augmentation-rank-one A supports and balanced cubic B supports should supply five novel measured distance readings >=16 and improve the 25.162 frontier in cell 0.150; this is a measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from C19:C6 to the direct product S3 x S3, explicitly implemented as D3 x D3. Its order is 36, centre is trivial and it has nine conjugacy classes; that bed is unlisted in BEDS. The measured C37 bed on these shapes reached five distance-fourteen readings. Two independent noncommuting factors test a different constituent mechanism without assuming distance transfer.")
SHAPES = ("A=(2,3) and B=(2,4) have 108 and 144 variables, each with 72 checks. Factor sixteen permits q<=37; q=36 gives n=576, kb=72 and guaranteed rate 0.125. A has augmentation rank one and B rank two, but only verified full ranks decide publication: require 0.125<k/n<0.175. Parent weight-eighteen families occupied sector-1 Acol/Bcol blocks; its lighter measured witnesses also touched Arow0/Brow0.")
FILLING = ("A counts are (0,2,2)/(3,1,1); two B profiles have row totals six and column totals three. BOTH LP budgets are checked explicitly: HX<=8 and HZ<=9, below nine and twelve. Twenty structural prescriptions vary binomial relative orders, placement and available noncommuting triangle spectra. Verified constituents exclude kernels through four, screen actual normal-C3xC3 orbit witnesses, and receive equal-budget information-set refinement. Full lifts enforce measured rate, CSS and kernel-four exclusion, reject repeated uncoloured Tanner invariants conservatively, and refine embedded plus independent mixed-sector logical witnesses. Cached outputs are immutable; larger witness readings and archive novelty remain measurement questions.")
FAMILY_NAME = "r015_s3xs3_q36_23x24_rankoneA_cubic_mixed_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 x S3 has order 36")
    # (a,b,c,d)=x^a y^b u^c v^d.
    # x^3=u^3=y^2=v^2=1; yxy^-1=x^-1, vuv^-1=u^-1.
    # The two factors commute.
    els = [(a,b,c,d) for d in range(2) for c in range(3)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2,
                int(g[2]) % 3, int(g[3]) % 2)

    def mul(g, h):
        a,b,c,d = red(g)
        aa,bb,cc,dd = red(h)
        return ((a + (-aa if b else aa)) % 3, (b+bb) % 2,
                (c + (-cc if d else cc)) % 3, (d+dd) % 2)

    def inv(g):
        a,b,c,d = red(g)
        return ((a if b else -a) % 3, b,
                (c if d else -c) % 3, d)

    def idx(g):
        a,b,c,d = red(g)
        return a + 3*b + 6*c + 18*d

    return dict(elements=els, ident=(0,0,0,0), mul=mul, inv=inv,
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
    """Historical reference only; the order-114 publisher never calls this."""
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


def _c19_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _c19_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _c19_probe(rows, width, local, passes, seed=0):
    """Find actual kernel witnesses with an equal budget per constituent."""
    cols = _syndrome_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if word and word.bit_count() < weight:
            best, weight = word, word.bit_count()

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
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

        # Elementary exchanges preserve the complete kernel basis.
        local.shuffle(basis)
        for i in range(len(basis)):
            current = basis[i]
            w = current.bit_count()
            for j in range(len(basis)):
                if i != j:
                    trial = current ^ basis[j]
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        current, w = trial, tw
            basis[i] = current
            inspect(current)

        for i, first in enumerate(basis):
            for second in basis[:i]:
                inspect(first ^ second)
        short = sorted(basis, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)

    if not best:
        raise ValueError("constituent kernel unexpectedly empty")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _c19_graph_key(hx, hz, n):
    """Conservative full uncoloured Tanner invariant.

    Published checks have degree nine, whereas qubits have degree six
    or twelve, so graph isomorphisms preserve the check vertex class.
    Equal invariants are inconclusive and are discarded conservatively.
    """
    from collections import Counter

    rows = hx + hz
    cols = _syndrome_columns(rows, n)
    m = len(rows)
    one, many = [0]*m, [0]*m
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
    history = [tuple(sorted(Counter(features).items()))]
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    incidence = []
    for col in cols:
        neighbours = []
        while col:
            bit = col & -col
            neighbours.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(neighbours)

    for _ in range(3):
        qfeatures = [tuple(sorted(colors[r] for r in neighbours))
                     for neighbours in incidence]
        qp = {f: i for i, f in enumerate(sorted(set(qfeatures)))}
        check_profiles = [[] for _ in rows]
        for neighbours, feature in zip(incidence, qfeatures):
            for r in neighbours:
                check_profiles[r].append(qp[feature])
        refined = [(colors[r], tuple(sorted(check_profiles[r])))
                   for r in range(m)]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: i for i, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return tuple(history)


@lru_cache(maxsize=1)
def _c19_catalogue(G, count):
    """Pure deterministic portfolio; memoization changes only build cost."""
    from collections import Counter

    if G.q != 114:
        raise ValueError("cubic portfolio requires order 114")
    q, width = G.q, 2*G.q
    cap = min(9, W_MAX)
    counts = [[3, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 9 <= cap
    unit = [[[G.e]]]
    mask = (1 << width) - 1
    prescriptions = (
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 2), (2, 3), (2, 4), (2, 5), (3, 3),
    )

    def verified_rows(C, side):
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, C)
        return [row & mask for row in rows[:q]]

    def cycle_cost(rows):
        shared = Counter()
        for col in _syndrome_columns(rows, width):
            neighbours = []
            while col:
                bit = col & -col
                neighbours.append(bit.bit_length() - 1)
                col ^= bit
            for i, a in enumerate(neighbours):
                for b in neighbours[:i]:
                    shared[(min(a, b), max(a, b))] += 1
        return sum(c*(c-1)//2 for c in shared.values())

    def constituent_bank(side):
        bank, fillings = [], set()
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0x19C600 + 7919*card
                             + (104729 if side == "B" else 0)))
            bpair = prescriptions[card % 10]
            raw = []
            for trial in range(48):
                row = []
                for j in range(2):
                    b = bpair[(j + int(side == "B")) % 2]
                    rotation = (local.randrange(1, 19), 0)
                    cross = (local.randrange(19), b)
                    if (card//10 + j + int(side == "B")) % 2:
                        cell = [G.e, cross, G.mul(cross, rotation)]
                    else:
                        cell = [G.e, rotation, cross]
                    assert len(set(cell)) == 3
                    row.append(cell)
                C = [row]
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows = verified_rows(C, side)
                if gf2_rank(rows) != q:
                    continue
                raw.append((cycle_cost(rows), trial, C, rows))
            raw.sort(key=lambda r: r[:2])

            shortlisted = []
            for cost, trial, C, rows in raw[:10]:
                if _short_pairs(_syndrome_columns(rows, width)) is None:
                    continue
                witness = _c19_probe(rows, width, local, 6)
                shortlisted.append(
                    (witness.bit_count(), cost, trial, C, rows, witness))
            shortlisted.sort(key=lambda r: (-r[0], r[1], r[2]))

            # Identical refinement budget for every shortlisted finalist.
            for _, cost, trial, C, rows, witness in shortlisted[:2]:
                witness = _c19_probe(rows, width, local, 12, witness)
                bank.append((witness.bit_count(), cost, card, trial,
                             C, witness))
        bank.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))
        if len(bank) < 5:
            raise ValueError("too few cubic constituents passed exact screens")
        return bank

    aa, bb = constituent_bank("A"), constituent_bank("B")
    plans = sorted(
        (-min(a[0], b[0]), -(a[0]+b[0]), a[1]+b[1], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))
    results, graph_keys, filling_keys = [], set(), set()

    def logical_reading(a, b, hx, hz):
        """Actual nontrivial witnesses, including stabilizer-pair descent."""
        best = None
        for kind, checks, stabilizers, source in (
                ("X", hz, hx, b[5]), ("Z", hx, hz, a[5])):
            own = _c19_span(stabilizers)
            for block in range(2):
                if kind == "X":
                    word = source << (block*width)
                else:
                    word, remaining = 0, source
                    while remaining:
                        bit = remaining & -remaining
                        j, h = divmod(bit.bit_length()-1, q)
                        word |= 1 << ((j*2 + block)*q + h)
                        remaining ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not _c19_outside(word, own):
                    continue
                weight = word.bit_count()
                for _ in range(2):
                    starting = word
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
                assert word and _c19_outside(word, own)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = weight if best is None else min(best, weight)
        return best

    # Catalogue separation is global, not just within a slot's search.
    for _, _, _, ai, bi in plans[:1024]:
        a, b = aa[ai], bb[bi]
        A, B = a[4], b[4]
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for C in (A, B) for row in C for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 570 or not (175*n <= 1000*k < 225*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        reading = logical_reading(a, b, hx, hz)
        if reading is None:
            continue
        signature = _c19_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified cubic lift failed CSS")
        graph_keys.add(signature)
        frozen_A = tuple(tuple(tuple(cell) for cell in row) for row in A)
        frozen_B = tuple(tuple(tuple(cell) for cell in row) for row in B)
        quality = (k*reading*reading/n, reading,
                   min(a[0], b[0]), -a[1]-b[1])
        results.append((quality, frozen_A, frozen_B, k, reading,
                        (a[2], b[2]), (a[0], b[0])))
        # Give publication a wider equally screened selection than count.
        if len(results) >= max(count, 5) + 12:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def _s36_component(G, grid, side):
    """Isolate the exact evaluator action with an unpublished zero side."""
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], grid)
    assert len(rows) == len(grid) * G.q
    return [row & mask for row in rows]


def _s36_orbit_seed(G, rows, blocks):
    """Exhaustively search the normal-C3xC3 invariant kernel.

    The four cosets have fixed reflection coordinates b,d. Enumeration
    concerns this invariant subspace only, not the complete kernel.
    """
    width = blocks * G.q
    cols = _syndrome_columns(rows, width)
    pivots, basis = {}, []
    for block in range(blocks):
        for b in range(2):
            for d in range(2):
                word = syndrome = 0
                for a in range(3):
                    for c in range(3):
                        j = block * G.q + G.idx((a, b, c, d))
                        word |= 1 << j
                        syndrome ^= cols[j]
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

    best = previous = word = 0
    for number in range(1, 1 << len(basis)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous).bit_length() - 1
        word ^= basis[changed]
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word
        previous = gray
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _s36_reduce(word, stabilizers):
    """Actual stabilizer descent, preserving the logical class."""
    for _ in range(3):
        previous = word
        for row in stabilizers:
            trial = word ^ row
            if trial.bit_count() < word.bit_count():
                word = trial
        starting = word
        moves = sorted(stabilizers,
                       key=lambda row: (starting ^ row).bit_count())[:40]
        for i, first in enumerate(moves):
            for second in moves[:i]:
                trial = starting ^ first ^ second
                if trial.bit_count() < word.bit_count():
                    word = trial
        if word == previous:
            break
    return word


def _s36_mixed(checks, stabilizers, n, local, seed):
    """Stabilizer-aware information-set search over BOTH LP sectors.

    Returned values are explicit logical witnesses and upper bounds.
    The second information set prioritizes sector two.
    """
    cols = _syndrome_columns(checks, n)
    own = _c19_span(stabilizers)
    best = _s36_reduce(seed, stabilizers)
    assert best and _c19_outside(best, own)
    weight = best.bit_count()

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight and _c19_outside(word, own):
            best, weight = word, w

    for pass_number in range(2):
        order = list(range(n))
        local.shuffle(order)
        if pass_number:
            order.sort(key=lambda j: j < 12 * 36)
        pivots, nulls = {}, []
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
                nulls.append(word)
                inspect(word)

        short = sorted(nulls, key=int.bit_count)[:28]
        reduced = []
        for word in nulls:
            w = word.bit_count()
            for other in short:
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            reduced.append(word)
            inspect(word)
        short = sorted(set(reduced), key=int.bit_count)[:28]
        for i, first in enumerate(short):
            for second in short[:i]:
                inspect(first ^ second)
        tiny = short[:18]
        for i, first in enumerate(tiny):
            for j in range(i):
                pair = first ^ tiny[j]
                for h in range(j):
                    inspect(pair ^ tiny[h])
        best = _s36_reduce(best, stabilizers)
        weight = best.bit_count()

    assert best and _c19_outside(best, own)
    assert all((row & best).bit_count() % 2 == 0 for row in checks)
    return best


@lru_cache(maxsize=1)
def _s36_catalogue(G, count):
    """Pure immutable portfolio of independent structural prescriptions.

    Historical order-114/Z8 catalogues are uncalled. All screening
    matrices, including component matrices, use the verified scaffold.
    """
    from itertools import combinations
    from collections import Counter

    if G.q != 36:
        raise ValueError("direct-product portfolio requires order 36")
    q, e = G.q, G.e
    cap = min(9, 12, W_MAX)
    sa = ((0, 2, 2), (3, 1, 1))
    bprofiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (3, 2, 1, 0)),
    )
    for sb in bprofiles:
        wx = max(map(sum, sa)) + max(
            sum(row[j] for row in sb) for j in range(4))
        wz = max(map(sum, sb)) + max(
            sum(row[j] for row in sa) for j in range(3))
        assert wx == 8 and wz == 9 and max(wx, wz) <= cap
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9

    def element_order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid direct-product element")

    orders = {g: element_order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != e]
    ratios = {
        3: tuple(g for g in nonzero if orders[g] == 3),
        6: tuple(g for g in nonzero if orders[g] == 6),
    }
    triangle_classes = {}
    for g, h in combinations(nonzero, 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        spectrum = tuple(sorted((
            orders[g], orders[h], orders[G.mul(G.inv(g), h)])))
        triangle_classes.setdefault(spectrum, []).append((g, h))
    spectra = tuple(sorted(triangle_classes))
    if not spectra or not ratios[3] or not ratios[6]:
        raise ValueError("direct-product prescriptions absent")

    def draw(local, profile, card, side):
        mode = card % 10
        policies = (
            (3, 3), (6, 6), (3, 6), (6, 3), (None, None),
        )
        policy = policies[mode % 5]
        triangle_kind = (mode // 5 + card // 10
                         + int(side == "B")) % len(spectra)
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
                    degree = policy[(i + j) % 2]
                    pool = (ratios[degree] if degree is not None else
                            ratios[3] + ratios[6])
                    relative = local.choice(pool)
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    g, h = local.choice(
                        triangle_classes[spectra[triangle_kind]])
                    if (i + j + card) % 2:
                        g, h = h, g
                    cell = [anchor, G.mul(anchor, g),
                            G.mul(anchor, h)]
                assert len(set(cell)) == terms
                row.append(cell)
            grid.append(row)
        return grid

    def cycle_cost(rows):
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                shared = (row & other).bit_count()
                total += shared * (shared - 1) // 2
        return total

    def constituent_bank(side):
        bank, fillings = [], set()
        for card in range(20):
            profile = sa if side == "A" else bprofiles[card // 10]
            blocks = len(profile[0])
            width = blocks * q
            local = random.Random(
                BASE_SEED ^ (0x363150 + 104729 * card
                             + 7919 * int(side == "B")))
            raw = []
            for trial in range(72):
                grid = draw(local, profile, card, side)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows = _s36_component(G, grid, side)
                rank = gf2_rank(rows)
                if rank < 2 * q - 8:
                    continue
                if _short_pairs(_syndrome_columns(rows, width)) is None:
                    continue
                seed = _s36_orbit_seed(G, rows, blocks)
                if seed and seed.bit_count() < 12:
                    continue
                raw.append((cycle_cost(rows), trial, grid, rows, seed))
            raw.sort(key=lambda item: item[:2])

            shortlisted = []
            for cost, trial, grid, rows, seed in raw[:10]:
                word = _c19_probe(rows, width, local, 4, seed)
                shortlisted.append(
                    (word.bit_count(), cost, trial, grid, rows, word))
            shortlisted.sort(key=lambda r: (-r[0], r[1], r[2]))

            # Every publishable constituent receives the same 4+12 passes.
            for _, cost, trial, grid, rows, seed in shortlisted[:3]:
                word = _c19_probe(rows, width, local, 12, seed)
                frozen = tuple(tuple(tuple(cell) for cell in row)
                               for row in grid)
                bank.append((word.bit_count(), cost, card, trial,
                             frozen, word))
        bank.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))
        if len(bank) < 5:
            raise ValueError("too few direct-product constituents")
        return bank

    aa, bb = constituent_bank("A"), constituent_bank("B")
    plans = sorted(
        (-min(a[0], b[0]), -(a[0] + b[0]), a[1] + b[1], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))

    def embedded(record, kind, checks, stabilizers):
        own = _c19_span(stabilizers)
        best = 0
        for block in range(3 if kind == "X" else 4):
            if kind == "X":
                word = record[5] << (block * 4 * q)
            else:
                word, source = 0, record[5]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 4 + block) * q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _c19_outside(word, own):
                continue
            word = _s36_reduce(word, stabilizers)
            assert word and _c19_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, graph_keys = [], set()
    wanted = max(count, 5) + 12
    for _, _, _, ai, bi in plans[:2048]:
        a, b = aa[ai], bb[bi]
        A, B = a[4], b[4]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Strict lower endpoint avoids the 0.125 nearest-centre tie.
        if n != 576 or not (125*n < 1000*k < 175*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified direct-product lift failed CSS")

        signature = _c19_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        xs = embedded(b, "X", hz, hx)
        zs = embedded(a, "Z", hx, hz)
        if not xs or not zs:
            continue
        local = random.Random(
            BASE_SEED ^ (0x3631C0 + 65537 * ai + 257 * bi))
        xword = _s36_mixed(hz, hx, n, local, xs)
        zword = _s36_mixed(hx, hz, n, local, zs)
        reading = min(xword.bit_count(), zword.bit_count())
        graph_keys.add(signature)

        # Cap the selection feature, not the reported witness. This
        # prevents poorly shortened large words from dominating selection.
        quality = (k * min(reading, 18) ** 2 / n,
                   min(reading, 18), min(a[0], b[0]),
                   -a[1] - b[1], k)
        results.append((quality, A, B, k, reading,
                        (a[2], b[2]), (a[0], b[0])))
        if len(results) >= wanted:
            break
    results.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish a fresh list view of one immutable screened prescription."""
    catalogue = _s36_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("direct-product target-rate portfolio exhausted")
    quality, A, B, k, reading, cards, constituents = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "component_witness_upper_bounds": constituents,
            "design_cards": cards, "quality": quality}


def CELL(rng, G, side, i, j, ctx):
    """Publish the selected full cell without altering cached designs."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Reserved positions separated by the pendant catalogue's constituent/full check-intersection signatures remain evaluator-distinct: the measured sixty-slot parent retained only twenty-six distinct codes.
# REFUTED: Kernel-four screening plus constituent-witness stabilizer-pair descent ensures that every d2s=18 finalist survives measurement at eighteen: current slots 1,2,9 instead measured 15,17,16, with mixed-sector minimum witnesses.
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
