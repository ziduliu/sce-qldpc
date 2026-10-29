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
CONCEPT = ("C3 direct product Dic3, order 36, A=(2,3), B=(2,4): order-twelve mixed-coset binomials and independently optimized constituents should supply five novel measured readings >=16 in cell 0.150; this is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump to C3 x Dic3, with x^6=1, y^2=x^3, y*x*y^-1=x^-1 and an independent central z of order three. Its centre has order six, whereas the parent's C9:C4 centre has order two, proving nonisomorphism. Order-twelve elements introduce relative orders absent from the parent. The related Dic11 saturated bed reached fitness 46.5, while the current parent measured five [576,76,14] leaders at fitness 25.861; neither establishes transferred distance.")
SHAPES = ("Keep A=(2,3), B=(2,4): constituents have 108 and 144 variables, each with 72 checks. Factor sixteen permits q<=37; fixed q=36 gives n=576,kb=72 and guaranteed rate 1/8. Verified complete ranks require 0.125<k/n<0.175, retaining cell 0.150 without the lower-bound tie. Parent measured weight-fourteen X minima were entirely sector-1 Acol/Bcol witnesses, broadly using all B columns and often Acol1; measured Z witnesses had weights sixteen or eighteen.")
FILLING = ("Twenty structural/coset cards retain balanced cubic columns and saturated five-term-row alternatives, with BOTH LP maxima <=9, also satisfying twelve. Relative binomials now use orders four, six and twelve, with deliberate central-C3 coordinate policies. Verified constituents screen normal-C3^2, both order-three factors, central-C2 and central-C6 invariant witnesses, exclude all kernel supports through six, and receive equal enlarged information-set budgets. Repeated constituent Tanner invariants are discarded before pairing. Full verified lifts enforce actual rate, CSS, kernel-four exclusion, global Tanner-invariant separation and explicit embedded/mixed-sector witnesses with stabilizer descent. Publication is immutable; larger readings and unseen-archive novelty remain hypotheses.")
FAMILY_NAME = "r015_c3xdic3_q36_23x24_order12_subgroup_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C3 direct product Dic3 requires order 36")
    # x^6=1, y^2=x^3, y*x*y^-1=x^-1.
    # z is independent, central, and has order three.
    # Encoding: x^a y^b z^c.
    els = [(a, b, c) for c in range(3)
           for b in range(2) for a in range(6)]

    def red(g):
        a, b, c = map(int, g)
        return ((a + 3 * (b // 2)) % 6, b % 2, c % 3)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d) + 3*b*e) % 6,
                (b + e) % 2, (c + f) % 3)

    def inv(g):
        a, b, c = red(g)
        if b == 0:
            return ((-a) % 6, 0, (-c) % 3)
        return ((a + 3) % 6, 1, (-c) % 3)

    def idx(g):
        a, b, c = red(g)
        return 12*c + 6*b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
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


def _r15_rows(G, grid, side):
    """Extract arbitrary-shaped constituents from the verified scaffold."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    mask = (1 << width) - 1
    assert len(rows) == len(grid) * G.q
    return [row & mask for row in rows], width


def _r15_clear6(G, rows, width):
    """Exact constituent kernel exclusion through six.

    Verified constituent kernels have a regular action transitive within
    each variable block. Any short support can consequently be translated
    to include an identity-coordinate anchor. This is not assumed for a
    full quantum matrix.
    """
    cols = _syndrome_columns(rows, width)
    pairs = _short_pairs(cols)
    if pairs is None:
        return False
    anchored = {}
    for block in range(width // G.q):
        anchor = block * G.q + G.idx(G.e)
        others = [j for j in range(width) if j != anchor]
        for ai, a in enumerate(others):
            value = cols[anchor] ^ cols[a]
            word = (1 << anchor) | (1 << a)
            for b in others[:ai]:
                syndrome = value ^ cols[b]
                support = word | (1 << b)
                # Triple/pair collisions detect all anchored five-words.
                if syndrome in pairs:
                    return False
                old = anchored.get(syndrome)
                if old is not None and old != support:
                    return False
                anchored[syndrome] = support

    # Triple/triple collisions detect every translated six-word.
    for i, first in enumerate(cols):
        for j in range(i):
            value = first ^ cols[j]
            word = (1 << i) | (1 << j)
            for h in range(j):
                old = anchored.get(value ^ cols[h])
                if old is not None and old != (word | (1 << h)):
                    return False
    return True


def _r15_orbit_seed(G, rows, width):
    """Actual kernel witnesses in three subgroup-invariant subspaces.

    Compression uses only syndromes of the verified matrix. Small
    compressed kernels are enumerated exactly; larger ones receive
    basis/pair/triple witness searches, not distance certificates.
    """
    cols = _syndrome_columns(rows, width)
    # Build coset partitions using the new group's actual operations.
    # These are witness subspaces, not a separately implemented lift.
    generators = (
        ((2, 0, 0), (0, 0, 1)),  # normal C3 x C3, order nine
        ((2, 0, 0),),             # the noncentral C3 factor
        ((0, 0, 1),),             # the independent central C3
        ((3, 0, 0),),             # central C2
        ((3, 0, 0), (0, 0, 1)),  # the whole central C6
    )

    def cosets(gs):
        steps = tuple(gs) + tuple(G.inv(g) for g in gs)
        subgroup, frontier = {G.e}, [G.e]
        while frontier:
            first = frontier.pop()
            for step in steps:
                product = G.mul(first, step)
                if product not in subgroup:
                    subgroup.add(product)
                    frontier.append(product)
        remaining = set(G.elements)
        partition = []
        while remaining:
            representative = min(remaining, key=G.idx)
            packet = tuple(sorted(
                (G.mul(representative, h) for h in subgroup),
                key=G.idx))
            assert len(set(packet)) == len(subgroup)
            partition.append(packet)
            remaining.difference_update(packet)
        return tuple(partition)

    packets = tuple(cosets(gs) for gs in generators)
    assert tuple(len(p[0]) for p in packets) == (9, 3, 3, 2, 6)
    best = 0

    def inspect(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

    for partition in packets:
        pivots, kernel = {}, []
        for block in range(width // G.q):
            for packet in partition:
                syndrome = word = 0
                for g in packet:
                    position = block * G.q + G.idx(g)
                    syndrome ^= cols[position]
                    word |= 1 << position
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

        if len(kernel) <= 14:
            previous = word = 0
            for number in range(1, 1 << len(kernel)):
                gray = number ^ (number >> 1)
                changed = (gray ^ previous).bit_length() - 1
                word ^= kernel[changed]
                inspect(word)
                previous = gray
        else:
            for i, first in enumerate(kernel):
                inspect(first)
                for second in kernel[:i]:
                    inspect(first ^ second)
            short = sorted(kernel, key=int.bit_count)[:24]
            for i, first in enumerate(short):
                for j in range(i):
                    pair = first ^ short[j]
                    for h in range(j):
                        inspect(pair ^ short[h])
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _r15_reduce(word, stabilizers):
    """Nonincreasing descent within one actual stabilizer coset."""
    for _ in range(3):
        starting = word
        moves = sorted(stabilizers,
                       key=lambda row: (starting ^ row).bit_count())
        for row in moves:
            trial = starting ^ row
            if trial.bit_count() < word.bit_count():
                word = trial
        short = moves[:28]
        for i, first in enumerate(short):
            for second in short[:i]:
                trial = starting ^ first ^ second
                if trial.bit_count() < word.bit_count():
                    word = trial
        if word == starting:
            break
    return word


def _r15_logical_reading(G, a, b, hx, hz, n, local):
    """Explicit logical witnesses, including independent mixed sectors."""
    q, ca, cb = G.q, A_SHAPE[1], B_SHAPE[1]
    readings = []
    for kind, checks, stabilizers, source in (
            ("X", hz, hx, b["witness"]),
            ("Z", hx, hz, a["witness"])):
        own = _c19_span(stabilizers)
        best, weight = 0, n + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if word and w < weight and _c19_outside(word, own):
                best, weight = word, w

        for block in range(ca if kind == "X" else cb):
            if kind == "X":
                word = source << (block * cb * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + block) * q + t)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if _c19_outside(word, own):
                inspect(_r15_reduce(word, stabilizers))

        cols = _syndrome_columns(checks, n)
        order = list(range(n))
        for turn in range(3):
            local.shuffle(order)
            if turn == 1:
                order.sort(key=lambda j: j < ca * cb * q)
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

            short = sorted(kernel, key=int.bit_count)[:32]
            reduced = []
            for word in kernel:
                for other in short:
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
                for row in stabilizers:
                    trial = word ^ row
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
                inspect(word)
                if _c19_outside(word, own):
                    reduced.append(word)
            short = sorted(set(reduced), key=int.bit_count)[:24]
            for i, first in enumerate(short):
                for j in range(i):
                    pair = first ^ short[j]
                    inspect(pair)
                    for h in range(j):
                        inspect(pair ^ short[h])
            if best:
                inspect(_r15_reduce(best, stabilizers))

        if not best:
            return None
        assert _c19_outside(best, own)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
        readings.append(weight)
    return min(readings)


@lru_cache(maxsize=1)
def _r15_portfolio(G, count):
    """Pure deterministic optimization; published grids are immutable."""
    from collections import Counter
    from hashlib import sha256

    assert G.q == 36 and A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    assert G.e == (0, 0, 0)
    assert all(len(g) == 3 for g in G.elements)
    cap = min(9, 12, W_MAX)
    assert cap >= 9

    # Regime zero uses cubic columns throughout, reallocating term
    # budget toward the longer, higher-rate B constituent.
    # Regime one tests the low-rate bed's saturated reinforcement model.
    profiles = {
        ("A", 0): (
            ((1, 1, 2), (2, 2, 1)),
            ((0, 1, 3), (3, 2, 0)),
            ((0, 2, 2), (3, 1, 1)),
            ((1, 1, 1), (2, 2, 2)),
        ),
        ("B", 0): (
            ((1, 1, 2, 2), (2, 2, 1, 1)),
            ((0, 2, 2, 2), (3, 1, 1, 1)),
            ((0, 1, 2, 3), (3, 2, 1, 0)),
            ((1, 1, 1, 3), (2, 2, 2, 0)),
        ),
        ("A", 1): (
            ((1, 2, 2), (2, 1, 2)),
            ((1, 1, 3), (2, 2, 1)),
            ((1, 1, 3), (1, 3, 1)),
            ((1, 2, 2), (1, 2, 2)),
        ),
        ("B", 1): (
            ((1, 1, 1, 2), (1, 1, 2, 1)),
            ((1, 1, 1, 2), (1, 1, 1, 2)),
            ((1, 1, 0, 3), (1, 1, 3, 0)),
            ((1, 1, 1, 2), (0, 1, 3, 1)),
            ((1, 1, 0, 3), (1, 0, 3, 1)),
        ),
    }
    # Every rotation here has order six. Crosses with nonzero z
    # coordinate have order twelve; the remaining crosses have order four.
    rotations = [g for g in G.elements
                 if g[1] == 0 and g[0] in (1, 5)]
    crosses = [g for g in G.elements if g[1] == 1 and g[2] != 0]
    short_crosses = [g for g in G.elements
                     if g[1] == 1 and g[2] == 0]
    long = rotations + crosses + short_crosses

    def element_order(g):
        product, order = G.e, 0
        while True:
            product = G.mul(product, g)
            order += 1
            if product == G.e:
                return order

    assert all(element_order(g) == 6 for g in rotations)
    assert all(element_order(g) == 12 for g in crosses)
    assert all(element_order(g) == 4 for g in short_crosses)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def cycle_cost(rows):
        cols = _syndrome_columns(rows, max(row.bit_length() for row in rows))
        shared = Counter()
        adjacency = [0] * len(rows)
        internal = 0
        for syndrome in cols:
            degree = syndrome.bit_count()
            internal += degree * (degree - 1) * (degree - 2) // 6
            neighbors = []
            bits = syndrome
            while bits:
                bit = bits & -bits
                i = bit.bit_length() - 1
                neighbors.append(i)
                adjacency[i] |= syndrome ^ bit
                bits ^= bit
            for i, first in enumerate(neighbors):
                for second in neighbors[:i]:
                    shared[(min(first, second), max(first, second))] += 1
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            bits = neighbors >> (i + 1)
            while bits:
                bit = bits & -bits
                j = i + bit.bit_length()
                triangles += ((neighbors & adjacency[j]) >> (j + 1)).bit_count()
                bits ^= bit
        squares = sum(v * (v - 1) // 2 for v in shared.values())
        return 16 * squares + max(0, triangles - internal)

    def draw(local, support, policy, side, trial):
        grid = [[[] for _ in support[0]] for _ in support]
        for j in range(len(support[0])):
            first_endpoint = True
            for i in range(len(support)):
                terms = support[i][j]
                if not terms:
                    continue
                anchor = G.e if first_endpoint else G.pick(local)
                first_endpoint = False
                if terms == 1:
                    grid[i][j] = [anchor]
                    continue
                if terms == 2:
                    if policy == 0:
                        pool = crosses
                    elif policy == 1:
                        pool = rotations if (i + j + trial) % 2 else crosses
                    elif policy == 2:
                        # Compare order-four and order-twelve crossings.
                        pool = (short_crosses if (i + j + trial) % 3 == 0
                                else crosses)
                    elif policy == 3:
                        # Deliberately alternate the independent central
                        # coordinate, rather than applying a global twist.
                        coordinate = 1 + ((i + j + int(side == "B")) % 2)
                        pool = [g for g in crosses if g[2] == coordinate]
                    else:
                        pool = long
                    relative = local.choice(pool)
                    grid[i][j] = [anchor, G.mul(anchor, relative)]
                    continue

                for _ in range(64):
                    if policy == 2:
                        r, s = local.sample(crosses, 2)
                    elif policy == 4:
                        r, s = local.sample(long, 2)
                    else:
                        r, s = local.choice(rotations), local.choice(crosses)
                    if r == s or G.mul(r, s) == G.mul(s, r):
                        continue
                    grid[i][j] = [anchor, G.mul(anchor, r), G.mul(anchor, s)]
                    break
                if not grid[i][j]:
                    return None
        assert all(len(set(cell)) == len(cell)
                   for row in grid for cell in row)
        return grid

    def bank(side):
        entries, fillings, component_keys = [], set(), set()
        for card in range(20):
            regime, subcard = divmod(card, 10)
            options = profiles[(side, regime)]
            pi = (subcard // 2) % len(options)
            support, policy = options[pi], subcard % 5
            local = random.Random(
                BASE_SEED ^ (0xC3D31501 + 104729 * card
                             + 1000003 * int(side == "B")))
            raw = []
            for trial in range(128):
                C = draw(local, support, policy, side, trial)
                if C is None:
                    continue
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for row in C for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, width = _r15_rows(G, C, side)
                if gf2_rank(rows) < 2 * G.q - 8:
                    continue
                if _short_pairs(_syndrome_columns(rows, width)) is None:
                    continue
                seed = _r15_orbit_seed(G, rows, width)
                cheap = (int(bool(seed) and seed.bit_count() < 16),
                         cycle_cost(rows), trial)
                raw.append((cheap, C, rows, width, seed))
            raw.sort(key=lambda r: r[0])

            shortlisted = []
            for cheap, C, rows, width, seed in raw[:20]:
                word = _c19_probe(rows, width, local, 4, seed)
                shortlisted.append((
                    -min(word.bit_count(), 22), cheap,
                    C, rows, width, word))
            shortlisted.sort(key=lambda r: r[:2])

            finalists = []
            for _, cheap, C, rows, width, seed in shortlisted[:8]:
                if not _r15_clear6(G, rows, width):
                    continue
                # Every finalist receives the same deeper budget and
                # retains its existing actual witness.
                word = _c19_probe(rows, width, local, 32, seed)
                assert word.bit_count() >= 7
                key = _c19_graph_key(rows, [], width)
                finalists.append(dict(
                    grid=freeze(C), witness=word,
                    reading=word.bit_count(), cost=cheap[1],
                    card=card, profile=pi, policy=policy, regime=regime,
                    component_key=key))
            finalists.sort(key=lambda r: (
                -min(r["reading"], 22), r["cost"], repr(r["grid"])))

            kept = 0
            for record in finalists:
                key = record["component_key"]
                if key in component_keys:
                    continue
                component_keys.add(key)
                entries.append(record)
                kept += 1
                if kept == 4:
                    break
        if len(entries) < 6:
            raise ValueError("too few distinct product-group constituents")
        return entries

    aa, bb = bank("A"), bank("B")
    plans = []
    pairing = random.Random(BASE_SEED ^ 0xC3D31502)
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            if a["regime"] != b["regime"]:
                continue
            sa = [[len(cell) for cell in row] for row in a["grid"]]
            sb = [[len(cell) for cell in row] for row in b["grid"]]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            prescribed = (
                b["card"] % 10 == (3 * (a["card"] % 10) + 1) % 10)
            priority = (
                min(a["reading"], b["reading"], 22),
                min(a["reading"], 22) + min(b["reading"], 22),
                int(prescribed), -a["cost"] - b["cost"],
                pairing.random())
            plans.append((priority, ai, bi))
    plans.sort(reverse=True)

    results, signatures = [], set()
    wanted = count + max(12, count // 3)
    quota = max(4, (wanted + min(len(aa), len(bb)) - 1)
                // min(len(aa), len(bb)))
    a_uses, b_uses = [0] * len(aa), [0] * len(bb)
    tested = 0
    for preliminary, ai, bi in plans:
        if a_uses[ai] >= quota or b_uses[bi] >= quota:
            continue
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or not (125 * n < 1000 * k < 175 * n):
            continue
        assert k >= 72
        assert max(row.bit_count() for row in hx + hz) <= cap
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue

        # This invariant uses neither CSS colouring nor block/group labels.
        # Equality is inconclusive and is rejected conservatively.
        rows = hx + hz
        signature = sha256(repr((
            n,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in
                                 _syndrome_columns(rows, n)).items())),
            _c19_graph_key(hx, hz, n),
        )).encode("ascii")).digest()
        if signature in signatures:
            continue
        signatures.add(signature)
        if not css_ok(hx, hz):
            raise ValueError("verified resized lift failed CSS")
        tested += 1
        local = random.Random(
            BASE_SEED ^ (0xC3D31503 + 65537 * ai + bi))
        reading = _r15_logical_reading(G, a, b, hx, hz, n, local)
        if reading is not None and reading >= 7:
            bounded = min(reading, 20)
            quality = (k * bounded * bounded / n, bounded,
                       -a["cost"] - b["cost"], preliminary)
            results.append((
                quality, A, B, k, reading,
                (a["card"], b["card"]),
                (a["reading"], b["reading"])))
            a_uses[ai] += 1
            b_uses[bi] += 1
        if len(results) >= wanted or tested >= max(256, 8 * count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    if len(results) < 5:
        raise ValueError("fewer than five target-rate resized designs")
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish a complete immutable product-group target-rate design."""
    assert G.q == 36 and G.e == (0, 0, 0)
    catalogue = _r15_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct resized portfolio exhausted")
    quality, A, B, k, reading, cards, constituents = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "component_witness_upper_bounds": constituents,
            "design_cards": cards, "quality": quality,
            "constituent_kernel_exclusion_through": 6,
            "complete_kernel_exclusion_through": 4}


def CELL(rng, G, side, i, j, ctx):
    """Publish the selected full cell without altering cached designs."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Reserved positions separated by the pendant catalogue's constituent/full check-intersection signatures remain evaluator-distinct: the measured sixty-slot parent retained only twenty-six distinct codes.
# REFUTED: An order-114 cubic d2s reading of eighteen always survives the longer search: measured slots 1,2,9 instead reached d30s=15,17,16, with minimum witnesses touching both LP sectors.
# REFUTED: The C9:C4 subgroup-aware multirow portfolio supplies five measured readings >=16: all five measured leaders instead had n=576,k=76,d30s=14, with minimum X witnesses entirely in sector 1.
# REFUTED: The C9:C4 portfolio's intrinsic-invariant separation supplies twenty evaluator-inequivalent initial slots: slots 0..19 retained twelve classes, and sixty slots retained thirty-nine.
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
