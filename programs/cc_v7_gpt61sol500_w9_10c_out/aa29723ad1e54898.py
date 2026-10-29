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
CONCEPT = ("AGL(1,8) times C2, order 112, A=B=(1,2): compatible quotient rank deficits and jointly screened noncommuting cubic constituents should supply five novel cell-0.200 finalists with mean measured score above 64.8; this is a falsifiable measurement hypothesis.")
GROUP = ("Jump from order-114 C19:C6 to (C2^3:C7) times C2. Multiplication by a primitive element of F8 faithfully acts on its additive group. Different order and noncyclic normal subgroup make this a genuine group change. Related cubic beds measured 56.7, 60.8 and 64.8. BEDS lists no successful evaluation identified as this affine-field group; its unknown order-112 failure is not positive evidence. Characteristic-two rank surplus is tested, not assumed.")
SHAPES = ("Keep equal (1,2) sides, each with 224 variables and 112 checks. Factor five allows q<=119; q=112 gives n=560, kb=112 and guaranteed rate 0.200. Actual ranks retain 112<=k<=125, entirely inside cell 0.200. Parent minima at weights 15,16,17 included mixed-sector operators touching Arow0/Brow0 and several Acol/Bcol blocks; its weight-eighteen families were sector-1 operators. Full logical probing therefore supplements constituent selection.")
FILLING = ("Twenty aligned/heterogeneous quotient-triangle prescriptions optimize genuine F8 and central-coordinate voltages. Every published cell has three distinct elements: BOTH CSS sums are 6+3=9, respecting both nine and twelve. Verified constituents exclude kernels through four, search normal-C2^3 coset witnesses and receive equal information-set budgets. Complete lifts enforce actual rate, CSS and full kernel-four exclusion. Explicit nontrivial embedded and independently searched mixed-sector logicals undergo stabilizer descent. Degree-conditioned full-Tanner invariants conservatively separate immutable portfolio entries. Actual rank surplus is rewarded, while all witness and GPU readings beyond the exact cutoff remain upper bounds; unseen archive novelty is not certified.")
FAMILY_NAME = "r020_agl1_f8_xc2_q112_12x12_quotientdefect_mixed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,8) times C2 requires order 112")
    # F8 = F2[t]/(t^3+t+1), alpha=t has multiplicative order seven.
    # (a,b,c) represents z -> alpha^b*z+a and an independent C2.
    def field_mul(a, b):
        answer = 0
        while b:
            if b & 1:
                answer ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return answer

    table = [[field_mul(a, b) for b in range(8)] for a in range(8)]
    powers = [1]
    for _ in range(6):
        powers.append(table[powers[-1]][2])
    assert table[powers[-1]][2] == 1 and len(set(powers)) == 7
    els = [(a, b, c) for c in range(2)
           for b in range(7) for a in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 7, int(g[2]) % 2)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return (a ^ table[powers[b]][d], (b + e) % 7, c ^ f)

    def inv(g):
        a, b, c = red(g)
        bi = (-b) % 7
        return (table[powers[bi]][a], bi, c)

    def idx(g):
        a, b, c = red(g)
        return 56*c + 8*b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


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


def _c19_probe(rows, width, local, passes, seed=0, own=None,
               stabilizers=None):
    """Find actual kernel/logical witnesses, never a distance certificate."""
    cols = _syndrome_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if own is not None and not _c19_outside(word, own):
            return
        best, weight = word, word.bit_count()

    if seed:
        assert all((row & seed).bit_count() % 2 == 0 for row in rows)
        assert own is None or _c19_outside(seed, own)
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
        if best and stabilizers:
            for _ in range(2):
                previous = best
                for row in stabilizers:
                    inspect(best ^ row)
                starting = best
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (starting ^ row).bit_count())[:32]
                for i, first in enumerate(nearby):
                    for second in nearby[:i]:
                        inspect(starting ^ first ^ second)
                if best == previous:
                    break

    if not best:
        raise ValueError("kernel/logical witness absent")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    assert own is None or _c19_outside(best, own)
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

    # Condition projections on intrinsic qubit degree, distinguishing
    # the two sectors without using their labels or CSS check colours.
    relations = {}
    for neighbours in incidence:
        relation = relations.setdefault(len(neighbours), [0] * m)
        mask = sum(1 << r for r in neighbours)
        for r in neighbours:
            relation[r] |= mask ^ (1 << r)
    projected_profiles = [[] for _ in rows]
    for degree, relation in sorted(relations.items()):
        for i, neighbours in enumerate(relation):
            profile = Counter()
            walk4 = 0
            for j, other in enumerate(relation):
                common = (neighbours & other).bit_count()
                profile[((neighbours >> j) & 1,
                         other.bit_count(), common)] += 1
                walk4 += common * common
            projected_profiles[i].append((
                degree, neighbours.bit_count(), walk4,
                tuple(sorted(profile.items()))))
    return (tuple(history),
            tuple(sorted(tuple(p) for p in projected_profiles)))


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


@lru_cache(maxsize=1)
def _gf8_catalogue(G, count):
    """Immutable portfolio on the affine-field bed.

    Historical Z8/C19 publishers are uncalled. Every screening matrix
    below is obtained from the scaffold's verified construction.
    """
    from collections import Counter

    if G.q != 112:
        raise ValueError("affine-field portfolio requires order 112")
    q, width, n_expected = G.q, 2 * G.q, 5 * G.q
    cap = min(9, 12, W_MAX)
    support = [[3, 3]]
    wx = sum(support[0]) + max(support[0])
    wz = sum(support[0]) + max(support[0])
    assert wx == wz == 9 <= cap
    assert _no_dead_lines(support, support)

    # Some C7 trinomials are units, others vanish in a cubic quotient.
    # Aligned policies test compatible deficits; heterogeneous policies
    # test whether they instead introduce short constituent words.
    models = ((1, 2), (1, 3), (1, 5), (2, 3), (2, 4),
              (2, 6), (3, 4), (3, 5), (3, 6), (4, 6))

    def component(C, side):
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], C)
        assert len(rows) == q
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def draw(local, card):
        kinds = (card % 10,
                 card % 10 if card < 10 else (3 * card + 1) % 10)
        row = []
        for kind in kinds:
            b, d = models[kind]
            for _ in range(64):
                g = (local.randrange(8), b, local.randrange(2))
                h = (local.randrange(8), d, local.randrange(2))
                if G.mul(g, h) == G.mul(h, g):
                    continue
                cell = [G.e, g, h]
                assert len(set(cell)) == 3
                row.append(cell)
                break
            else:
                return None
        # Do not publish a constituent confined to the index-two affine
        # subgroup merely because its duplicate copies look promising.
        if not any(g[2] for cell in row for g in cell):
            return None
        return [row]

    def normal_seed(rows):
        """Actual kernel words constant on normal-C2^3 cosets.

        Enumerate the compressed kernel when its dimension is small.
        This does not enumerate the complete constituent kernel.
        """
        cols = _syndrome_columns(rows, width)
        pivots, basis = {}, []
        for block in range(2):
            for central in range(2):
                for b in range(7):
                    syndrome = word = 0
                    for a in range(8):
                        j = block*q + G.idx((a, b, central))
                        syndrome ^= cols[j]
                        word |= 1 << j
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
        best = (1 << width) - 1
        assert all((row & best).bit_count() % 2 == 0 for row in rows)

        def inspect(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                best = word

        if len(basis) <= 15:
            word = previous = 0
            for number in range(1, 1 << len(basis)):
                gray = number ^ (number >> 1)
                changed = (gray ^ previous).bit_length() - 1
                word ^= basis[changed]
                inspect(word)
                previous = gray
        else:
            for i, first in enumerate(basis):
                inspect(first)
                for j in range(i):
                    pair = first ^ basis[j]
                    inspect(pair)
                    for h in range(j):
                        inspect(pair ^ basis[h])
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return best

    def cycle_cost(rows, cols):
        shared = Counter()
        adjacency = [0] * q
        internal = 0
        for col in cols:
            neighbours = []
            internal += col.bit_count() * (col.bit_count()-1) \
                * (col.bit_count()-2) // 6
            bits = col
            while bits:
                bit = bits & -bits
                r = bit.bit_length() - 1
                neighbours.append(r)
                adjacency[r] |= col ^ bit
                bits ^= bit
            for i, a in enumerate(neighbours):
                for b in neighbours[:i]:
                    shared[(min(a, b), max(a, b))] += 1
        four = sum(c*(c-1)//2 for c in shared.values())
        triangles = 0
        for i, neighbours in enumerate(adjacency):
            bits = neighbours >> (i + 1)
            while bits:
                bit = bits & -bits
                j = bit.bit_length() + i
                triangles += ((neighbours & adjacency[j])
                              >> (j + 1)).bit_count()
                bits ^= bit
        return 16*four + max(0, triangles-internal)

    def bank(side):
        entries, seen = [], set()
        for prescription in range(20):
            card = (prescription if side == "A"
                    else (7*prescription + 2) % 20)
            local = random.Random(
                BASE_SEED ^ (0x8A1120 + 104729*prescription
                             + 7919*int(side == "B")))
            raw = []
            for trial in range(72):
                C = draw(local, card)
                if C is None:
                    continue
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for cell in C[0])
                if filling in seen:
                    continue
                seen.add(filling)
                rows = component(C, side)
                rank = gf2_rank(rows)
                if rank < q - 24:
                    continue
                cols = _syndrome_columns(rows, width)
                if _short_pairs(cols) is None:
                    continue
                cost = cycle_cost(rows, cols)
                raw.append((cost, trial, C, rows, rank))
            raw.sort(key=lambda r: r[:2])

            refined = []
            for cost, trial, C, rows, rank in raw[:12]:
                seed = normal_seed(rows)
                word = _c19_probe(rows, width, local, 6, seed)
                refined.append((word.bit_count(), cost, trial,
                                C, rows, word, rank))
            refined.sort(key=lambda r: (-r[0], r[1], r[2]))

            # Every publishable constituent receives the same 6+12
            # information-set passes and the same normal-subgroup screen.
            for _, cost, trial, C, rows, word, rank in refined[:3]:
                word = _c19_probe(rows, width, local, 12, word)
                entries.append((word.bit_count(), cost, card, trial,
                                C, word, rank))
        entries.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))
        if len(entries) < 5:
            raise ValueError("too few screened affine-field constituents")
        return entries

    aa, bb = bank("A"), bank("B")
    plans = sorted(
        (-min(a[0], b[0], 20), -min(a[0]+b[0], 40),
         a[1]+b[1], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))

    def embedded(source, kind, checks, stabilizers, own):
        best = 0
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
            for _ in range(2):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                starting = word
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (starting ^ row).bit_count())[:32]
                for i, first in enumerate(nearby):
                    for second in nearby[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < word.bit_count():
                            word = trial
                if word == previous:
                    break
            assert word and _c19_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, graph_keys = [], set()
    wanted = max(count, 5) + 24
    for preliminary, total, cost, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a[4], b[4]
        assert all(len(cell) == len(set(cell)) == 3
                   for C in (A, B) for row in C for cell in row)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != n_expected or not (112 <= k <= 125):
            continue
        assert 175*n <= 1000*k < 225*n
        assert max(row.bit_count() for row in hx + hz) == 9 <= cap
        if _short_pairs(_syndrome_columns(hx, n)) is None:
            continue
        if _short_pairs(_syndrome_columns(hz, n)) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified affine-field lift failed CSS")
        signature = _c19_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        own_x, own_z = _c19_span(hx), _c19_span(hz)
        xs = embedded(b[5], "X", hz, hx, own_x)
        zs = embedded(a[5], "Z", hx, hz, own_z)
        if not xs or not zs:
            continue
        local = random.Random(
            BASE_SEED ^ (0x8A11C0 + 65537*ai + 257*bi))
        # These searches use every qubit, including the entire second
        # sector, and reject stabilizer-span words when scoring.
        xword = _c19_probe(hz, n, local, 4, xs, own_x, hx)
        zword = _c19_probe(hx, n, local, 4, zs, own_z, hz)
        reading = min(xword.bit_count(), zword.bit_count())

        def freeze(C):
            return tuple(tuple(tuple(cell) for cell in row) for row in C)

        # A search failure cannot dominate through an arbitrarily large
        # witness. Actual k, not a quotient-derived prediction, is used.
        quality = (k*min(reading, 20)**2/n,
                   min(reading, 20), -cost, k)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a[2], b[2]), (a[0], b[0]), (q-a[6], q-b[6])))
        if len(results) >= wanted:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable jointly screened affine-field design."""
    catalogue = _gf8_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct affine-field portfolio exhausted")
    quality, A, B, k, reading, cards, constituents, deficits = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "component_witness_upper_bounds": constituents,
            "component_rank_deficits": deficits,
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
# REFUTED: The order-114 cubic portfolio's intrinsic-key gate guarantees evaluator-distinct outputs: its first twenty slots included four isomorphism duplicates, and sixty slots retained only forty-two classes.
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
