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
CONCEPT = ("H_3(F3) direct C2, order 54, A=B=(1,3): noncommuting odd-coordinate binomials and screened trinomial reinforcement should produce five novel distance readings at least six in frontier cell 0.400.")
GROUP = ("Replace cyclic Z8 by the nonabelian Heisenberg group over F3 times C2. Noncentral odd-coordinate elements have order six and can be noncommuting, avoiding the automatic commuting-generator square of cyclic binomials. The current 0.400 frontier measured distance five and fitness 10.2; improvement is a falsifiable construction hypothesis, not a transferred group-distance claim.")
SHAPES = ("Equal (1,3) constituents have 162 variables and 54 checks. Order 54 gives n=540, kb=216 and guaranteed rate 0.400. Complete verified ranks retain only [0.375,0.425). The parent's weight-six minima broadly occupied sector-1 Acol/Bcol blocks, especially Acol2/3, without reported sector-2 minima. Replace its pendant components and screen both sectors.")
FILLING = ("A consists of three noncommuting odd-coordinate binomials. B is either another screened binomial component or one trinomial plus two odd-coordinate binomials. Counts (6,6) give both CSS maxima eight; counts (6,7) give both maxima nine, respecting the stricter scaffold budget as well as the assignment cap twelve. Twenty relative-coordinate cards supply candidates rather than orbit copies. Verified constituent kernels are excluded through five, complete kernels through four, and remaining mixed-sector five-words are excluded using a central translation whose action is explicitly verified on both built check sets. Immutable positions and intrinsic full-Tanner signatures reject repeats conservatively. These exact gates certify distance at least six; unseen archive novelty and higher measurement readings remain hypotheses.")
FAMILY_NAME = "r040_heisenberg3_times_c2_13x13_central_kernel5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    # Upper-unitriangular 3x3 matrices over F3, direct product with C2.
    # (a,b,c,t) denotes matrix entries (1,2)=a, (2,3)=b, (1,3)=c.
    if not valid_q(q):
        raise ValueError("H_3(F3) times C2 has order 54")
    els = [(a,b,c,t) for t in range(2) for c in range(3)
           for b in range(3) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3,
                int(g[2]) % 3, int(g[3]) % 2)

    def mul(g, h):
        a,b,c,t = red(g)
        x,y,z,u = red(h)
        return ((a+x) % 3, (b+y) % 3,
                (c+z+a*y) % 3, (t+u) % 2)

    def inv(g):
        a,b,c,t = red(g)
        return ((-a) % 3, (-b) % 3, (-c+a*b) % 3, t)

    def idx(g):
        a,b,c,t = red(g)
        return a + 3*b + 9*c + 27*t

    return dict(elements=els, ident=(0,0,0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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
    """Historical reference only; the new publication path never calls this."""
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


@lru_cache(maxsize=1)
def _tangent_portfolio(G, count):
    """Pure deterministic construction; cached publication grids are immutable."""
    from itertools import combinations_with_replacement
    from collections import Counter
    from hashlib import sha256

    if G.q != 8:
        raise ValueError("tangent portfolio requires the frozen order eight")
    q, width, n1 = G.q, 8 * G.q, 64 * G.q
    cap = min(9, W_MAX)
    mask = (1 << width) - 1
    dummy = [[[]]]

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    def rows_for(C, right=False):
        # All screening matrices use the scaffold's actual conventions.
        if right:
            _, rows, _ = lifted_matrices(G, dummy, C)
        else:
            rows, _, _ = lifted_matrices(G, C, dummy)
        return [row & mask for row in rows]

    def basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = word
                    break
                word ^= previous
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                return True
            word ^= previous
        return False

    def short_words(rows):
        """Generate representatives spanning all constituent kernels <=5.

        Differences within a syndrome class are generated by comparison
        with one representative. Pair representatives cover <=4; triples
        against representatives of weight <=2 then cover every five-word.
        """
        cols = _syndrome_columns(rows, width)
        reps, words = {0: 0}, set()

        def inspect(syndrome, word):
            previous = reps.get(syndrome)
            if previous is None:
                reps[syndrome] = word
            elif word != previous:
                words.add(word ^ previous)

        for j, syndrome in enumerate(cols):
            inspect(syndrome, 1 << j)
        for i, first in enumerate(cols):
            for j in range(i):
                inspect(first ^ cols[j], (1 << i) | (1 << j))

        triples, collisions = {}, 0
        for i, first in enumerate(cols):
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    syndrome = partial ^ cols[h]
                    previous = reps.get(syndrome)
                    if previous is not None:
                        word = ((1 << i) | (1 << j) | (1 << h)) ^ previous
                        if word:
                            words.add(word)
                    old = triples.get(syndrome, 0)
                    collisions += old
                    triples[syndrome] = old + 1
        clean = not words
        sixes = collisions // 10 if clean else None
        if clean:
            assert collisions % 10 == 0
        return tuple(sorted(words, key=lambda w: (w.bit_count(), w))), sixes

    def invert_word(word):
        result = 0
        while word:
            bit = word & -word
            j, h = divmod(bit.bit_length() - 1, q)
            result |= 1 << (j*q + G.idx(G.inv(G.elements[h])))
            word ^= bit
        return result

    # Intrinsic full-Tanner invariant, never a filling-label fingerprint.
    def graph_key(rows, length):
        m = len(rows)
        one, many = [0]*m, [0]*m
        overlaps = {}
        for i in range(m):
            for j in range(i):
                value = (rows[i] & rows[j]).bit_count()
                if value:
                    overlaps[(i, j)] = value
                    relation = one if value == 1 else many
                    relation[i] |= 1 << j
                    relation[j] |= 1 << i
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
        cols = _syndrome_columns(rows, length)
        neighborhoods = Counter()
        for col in cols:
            adjacent = []
            while col:
                bit = col & -col
                adjacent.append(colors[bit.bit_length() - 1])
                col ^= bit
            neighborhoods[tuple(sorted(adjacent))] += 1
        signature = (
            length, m, tuple(sorted(Counter(features).items())),
            tuple(sorted(neighborhoods.items())))
        return sha256(repr(signature).encode("ascii")).digest()

    # Enumerate count multisets, not permutations of one count grid.
    vectors = [
        (a, b, c)
        for a in range(4) for b in range(4) for c in range(4)
        if 1 <= a+b+c <= 3 and a % 2 == b % 2]
    parity_profiles = []
    for columns in combinations_with_replacement(vectors, 8):
        if any(sum(c[i] for c in columns) != 6 for i in range(3)):
            continue
        profile = tuple(tuple(c[i] for c in columns) for i in range(3))
        augmentation = [
            sum((profile[i][j] % 2) << j for j in range(8))
            for i in range(3)]
        if gf2_rank(augmentation) == 2:
            parity_profiles.append(profile)

    bank, component_keys = [], set()

    def retain(C, card, allow_short):
        filling = freeze(C)
        left = rows_for(C)
        rank = gf2_rank(left)
        if allow_short:
            # Small torsion deficits could keep a pair inside cell 0.350.
            if rank not in (22, 23):
                return None
        elif rank != 24:
            return None
        words, sixes = short_words(left)
        if not allow_short and words:
            return None
        if words and words[0].bit_count() <= 3:
            return None
        signature = graph_key(left, width)
        if signature in component_keys:
            return None
        right = rows_for(C, True)
        right_words = tuple(invert_word(w) for w in words)
        if gf2_rank(right) != rank:
            return None
        if any((row & word).bit_count() % 2
               for row in right for word in right_words):
            raise ValueError("verified left/right witness transfer failed")
        component_keys.add(signature)
        record = (rank, len(words), sixes if sixes is not None else 100000,
                  card, filling, words, right_words)
        bank.append(record)
        return record

    # Twenty parity/valuation cards vary both support and voltage policy.
    # These are allowed to fail: publication still requires logical screens.
    for card in range(20):
        if not parity_profiles:
            break
        profile = parity_profiles[card % len(parity_profiles)]
        local = random.Random(BASE_SEED ^ (0x8AD120 + 104729*card))
        kept = 0
        for trial in range(40):
            C = [[[] for _ in range(8)] for _ in range(3)]
            for j in range(8):
                for i in range(3):
                    terms = profile[i][j]
                    if not terms:
                        continue
                    labels = local.sample(range(q), terms)
                    # Policies change genuine relative gaps, not gauges.
                    if terms == 2 and card % 3 != 2:
                        gap = local.choice((1, 3, 5, 7) if card % 3 == 0
                                           else (2, 4, 6))
                        labels[1] = (labels[0] + gap) % q
                    C[i][j] = [G.elements[v] for v in sorted(labels)]
            if retain(C, card, True) is not None:
                kept += 1
                if kept >= 3:
                    break

    # Full-rank fallback cards use correlated endpoint voltages, exhaustive
    # gap choices, and later relative A/B automorphism twists. They are
    # selected independently of the historical parent's catalogue prefix.
    local = random.Random(BASE_SEED ^ 0x8AD660)
    proposals = [(a, b, c, t, policy)
                 for a in range(1, q) for b in range(1, q)
                 for c in range(1, q) for t in range(q)
                 for policy in range(4)]
    local.shuffle(proposals)
    clean_count = 0
    for a, b, c, t, policy in proposals:
        C = [[[] for _ in range(8)] for _ in range(3)]
        C[0][0] = C[0][1] = C[1][1] = [G.e]
        for j, voltage in enumerate((0, 1, 3), 2):
            C[1][j], C[2][j] = [G.e], [G.elements[voltage]]
        endpoints = (
            (t + 2*a + b + policy) % q,
            (3*t + b + 2*c + 2*policy) % q,
            (5*t + a + c + 3*policy) % q)
        for gap, endpoint, (j, i) in zip(
                (a, b, c), endpoints, ((5, 0), (6, 0), (7, 1))):
            C[i][j] = [G.e, G.elements[gap]]
            C[2][j] = [G.elements[endpoint]]
        if retain(C, 20 + policy, False) is not None:
            clean_count += 1
            if clean_count >= max(32, count):
                break
    if clean_count < 8:
        raise ValueError("too few independently screened fallback constituents")

    # Exclusion references ONLY. Block every pairing of the ten
    # constituents explicitly appearing in the current sixty-slot table.
    reference_cards = (
        (5,6,5,2,4,7), (6,1,5,2,7,7), (6,3,1,4,5,7),
        (2,7,7,4,5,6), (4,1,5,2,5,7), (6,3,7,4,2,4),
        (1,3,3,6,7,5), (2,3,4,6,5,2), (7,5,1,5,7,5),
        (1,5,2,7,2,6),
    )

    def reference(card):
        C = [[[] for _ in range(8)] for _ in range(3)]
        C[0][0] = C[0][1] = C[1][1] = [G.e]
        for j, voltage in enumerate((0, 1, 3), 2):
            C[1][j], C[2][j] = [G.e], [G.elements[voltage]]
        for t, (j, i) in enumerate(((5, 0), (6, 0), (7, 1))):
            C[i][j] = [G.e, G.elements[card[t]]]
            C[2][j] = [G.elements[card[t+3]]]
        return C

    references = [reference(card) for card in reference_cards]
    excluded = set()
    for a in range(len(references)):
        for b in range(a, len(references)):
            hx, hz, n = lifted_matrices(G, references[a], references[b])
            excluded.add(graph_key(hx + hz, n))

    def twist(C, u):
        return tuple(tuple(tuple(G.elements[(u*g[0]) % q] for g in cell)
                           for cell in row) for row in C)

    def shift_word(word):
        """One simultaneous group-coordinate translation, not a new lift."""
        shifted = 0
        while word:
            bit = word & -word
            block, h = divmod(bit.bit_length() - 1, q)
            shifted |= 1 << (block*q + (h+1) % q)
            word ^= bit
        return shifted

    def pair_quotient(checks, stabilizers, n):
        """Exactly reject nontrivial logicals of weights <=4."""
        own = basis(stabilizers)
        cols = _syndrome_columns(checks, n)
        reps = {0: 0}

        def inspect(syndrome, word):
            previous = reps.get(syndrome)
            if previous is None:
                reps[syndrome] = word
                return True
            return not outside(word ^ previous, own)

        for j, syndrome in enumerate(cols):
            if not inspect(syndrome, 1 << j):
                return None
        for i, first in enumerate(cols):
            for j in range(i):
                if not inspect(first ^ cols[j], (1 << i) | (1 << j)):
                    return None
        return cols, reps, own

    def pure_five(words, kind, checks, own):
        # Any pure-sector-1 kernel of weight <=5 decomposes into
        # constituent kernels, each itself of weight <=5.
        for source in words:
            for block in range(8):
                if kind == "X":
                    word = source << (block*width)
                else:
                    word, bits = 0, source
                    while bits:
                        bit = bits & -bits
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j*8 + block)*q + h)
                        bits ^= bit
                if any((row & word).bit_count() % 2 for row in checks):
                    raise ValueError("verified constituent embedding failed")
                if outside(word, own):
                    return False
        return True

    def mixed_five(cols, reps, own, n):
        """Exact remaining logical-five screen using verified symmetry.

        Translate one sector-2 qubit to coordinate zero. A five-support
        partitions into an anchored triple and a word of weight <=2.
        Pair-class differences were already proved to be stabilizers.
        """
        for anchor in range(n1, n, q):
            first = cols[anchor]
            anchor_bit = 1 << anchor
            for j in range(n):
                if j == anchor:
                    continue
                partial = first ^ cols[j]
                partial_word = anchor_bit | (1 << j)
                for h in range(j):
                    if h == anchor:
                        continue
                    previous = reps.get(partial ^ cols[h])
                    if previous is not None:
                        word = partial_word ^ (1 << h) ^ previous
                        if outside(word, own):
                            return False
        return True

    # Useful rank deficits precede clean full-rank fallbacks.
    plans = sorted(
        (bank[a][0] + bank[b][0],
         bank[a][1] + bank[b][1],
         max(bank[a][2], bank[b][2]), a, b, u)
        for a in range(len(bank)) for b in range(a, len(bank))
        for u in (1, 3, 5, 7))
    results, seen = [], set()
    for _, _, _, a, b, u in plans:
        A, B = bank[a][4], twist(bank[b][4], u)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (325*n <= 1000*k < 375*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        # Verify the translation automorphism on both built check sets.
        if (set(map(shift_word, hx)) != set(hx)
                or set(map(shift_word, hz)) != set(hz)):
            raise ValueError("built matrices failed cyclic translation test")
        if not css_ok(hx, hz):
            raise ValueError("verified tangent pair failed CSS")

        xdata = pair_quotient(hz, hx, n)
        zdata = pair_quotient(hx, hz, n)
        if xdata is None or zdata is None:
            continue
        # Recompute twisted B representatives using the verified right lift.
        bwords, _ = short_words(rows_for(B, True))
        if not pure_five(bwords, "X", hz, xdata[2]):
            continue
        if not pure_five(bank[a][5], "Z", hx, zdata[2]):
            continue

        signature = graph_key(hx + hz, n)
        if signature in excluded or signature in seen:
            continue
        seen.add(signature)
        if not mixed_five(*xdata, n) or not mixed_five(*zdata, n):
            continue
        results.append((k, freeze(A), freeze(B),
                        (bank[a][3], bank[b][3], u),
                        (bank[a][0], bank[b][0])))
        if len(results) >= count:
            break
    results.sort(key=lambda r: (-r[0], r[3]))
    return tuple(results)


def _heis_signature(rows, width):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive."""
    from collections import Counter
    from hashlib import sha256

    m = len(rows)
    degrees = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            if value:
                overlaps[i, j] = value
                relation = one if value == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlaps.get((i, j), 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(degrees[j],) + feature] += 1
            profiles[j][(degrees[i],) + feature] += 1
    features = [
        (degrees[i], tuple(sorted(profiles[i].items())))
        for i in range(m)]
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    columns = _syndrome_columns(rows, width)
    incidence = []
    for column in columns:
        neighbours = []
        while column:
            bit = column & -column
            neighbours.append(bit.bit_length() - 1)
            column ^= bit
        incidence.append(neighbours)

    history = [tuple(sorted(Counter(features).items()))]
    for _ in range(3):
        qfeatures = [
            tuple(sorted(colors[r] for r in neighbours))
            for neighbours in incidence]
        qpalette = {f: i for i, f in enumerate(sorted(set(qfeatures)))}
        attached = [[] for _ in rows]
        for neighbours, feature in zip(incidence, qfeatures):
            for r in neighbours:
                attached[r].append(qpalette[feature])
        refined = [
            (colors[i], tuple(sorted(attached[i])))
            for i in range(m)]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: i for i, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return sha256(repr((width, m, history)).encode("ascii")).digest()


def _heis_component_rows(G, grid, right=False):
    """Expose only scaffold-built left/right constituent matrices."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    return [row & mask for row in rows]


def _heis_component_clean(rows, width):
    """Exactly exclude constituent kernel words of weights one through five."""
    columns = _syndrome_columns(rows, width)
    pairs = _short_pairs(columns)
    if pairs is None:
        return False

    # A graph-incidence constituent has kernel words equal to unions
    # of cycles. Bipartiteness excludes odd cycles; the pair screen
    # already excluded all cycles of length at most four.
    if all(column.bit_count() == 2 for column in columns):
        adjacency = [[] for _ in rows]
        for column in columns:
            bit = column & -column
            a = bit.bit_length() - 1
            b = (column ^ bit).bit_length() - 1
            adjacency[a].append(b)
            adjacency[b].append(a)
        colors = [-1] * len(rows)
        bipartite = True
        for root in range(len(rows)):
            if colors[root] != -1:
                continue
            colors[root] = 0
            queue = [root]
            for a in queue:
                for b in adjacency[a]:
                    if colors[b] == -1:
                        colors[b] = 1 - colors[a]
                        queue.append(b)
                    elif colors[b] == colors[a]:
                        bipartite = False
        if bipartite:
            return True

    # With weights <=4 excluded, overlapping triple/pair matches
    # cannot occur. Every remaining match is a weight-five word.
    for i, first in enumerate(columns):
        for j in range(i):
            partial = first ^ columns[j]
            for h in range(j):
                if (partial ^ columns[h]) in pairs:
                    return False
    return True


@lru_cache(maxsize=1)
def _heis_portfolio(G, count):
    """Pure deterministic search returning immutable, screened designs."""
    from itertools import combinations

    if G.q != 54:
        raise ValueError("Heisenberg portfolio requires order 54")
    q, width, n1 = G.q, 3 * G.q, 9 * G.q
    cap = min(9, W_MAX)
    local = random.Random(BASE_SEED ^ 0x543136)
    odd = [g for g in G.elements
           if g[3] == 1 and (g[0] or g[1])]
    even = [g for g in G.elements
            if g[3] == 0 and g != G.e]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def noncommuting(g, h):
        return G.mul(g, h) != G.mul(h, g)

    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g))) for g in odd
    })
    bank, component_keys = [], set()
    for triple in combinations(reps, 3):
        generators = [G.elements[t] for t in triple]
        if not all(noncommuting(generators[i], generators[j])
                   for i in range(3) for j in range(i)):
            continue
        A = [[[G.e, g] for g in generators]]
        rows = _heis_component_rows(G, A)
        if gf2_rank(rows) != q - 1:
            continue
        if not _heis_component_clean(rows, width):
            continue
        signature = _heis_signature(rows, width)
        if signature in component_keys:
            continue
        right = _heis_component_rows(G, A, True)
        if not _heis_component_clean(right, width):
            continue
        component_keys.add(signature)
        bank.append(freeze(A))
    if not bank:
        raise ValueError("no connected kernel-five-clean odd Cayley component")

    # z generates the entire centre C3 x C2, of order six.
    z = (0, 0, 1, 1)
    powers, power = [], G.e
    for _ in range(6):
        powers.append(power)
        power = G.mul(power, z)
    assert power == G.e and len(set(powers)) == 6

    unused = set(range(q))
    orbit_reps = []
    while unused:
        root = min(unused)
        orbit = {G.idx(G.mul(G.elements[root], p)) for p in powers}
        assert len(orbit) == 6
        unused.difference_update(orbit)
        orbit_reps.append(root)
    assert len(orbit_reps) == 9

    movement = [G.idx(G.mul(g, z)) for g in G.elements]

    def translate(word):
        out = 0
        while word:
            bit = word & -word
            block, h = divmod(bit.bit_length() - 1, q)
            out |= 1 << (block * q + movement[h])
            word ^= bit
        return out

    def mixed_five(columns, pairs, n):
        """Exact mixed-sector screen after verified central symmetry.

        Translate one sector-2 qubit to its central-orbit representative.
        Every five-support has an anchored triple/pair partition.
        """
        for h0 in orbit_reps:
            anchor = n1 + h0
            first = columns[anchor]
            for j in range(n):
                if j == anchor:
                    continue
                partial = first ^ columns[j]
                for h in range(j):
                    if h != anchor and (partial ^ columns[h]) in pairs:
                        return False
        return True

    results, graph_keys, fillings = [], set(), set()

    def consider(A, B, card, component_passed=False):
        filling = (freeze(A), freeze(B))
        if filling in fillings:
            return False
        fillings.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            return False
        if not component_passed:
            br = _heis_component_rows(G, B, True)
            if not _heis_component_clean(br, width):
                return False

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 540 or not (375 * n <= 1000 * k < 425 * n):
            return False
        if max(row.bit_count() for row in hx + hz) > cap:
            return False

        # Central right multiplication commutes with both regular actions.
        # Check the assertion on the actual matrices before using it.
        if (set(map(translate, hx)) != set(hx)
                or set(map(translate, hz)) != set(hz)):
            raise ValueError("built matrices failed central translation test")
        if not css_ok(hx, hz):
            raise ValueError("verified Heisenberg lift failed CSS")

        data = []
        for checks in (hx, hz):
            columns = _syndrome_columns(checks, n)
            pairs = _short_pairs(columns)
            if pairs is None:
                return False
            data.append((columns, pairs))
        signature = _heis_signature(hx + hz, n)
        if signature in graph_keys:
            return False

        # Pure sector-1 words split into independent constituent kernels,
        # already excluded through five. Only mixed words remain.
        for columns, pairs in data:
            if not mixed_five(columns, pairs, n):
                return False
        graph_keys.add(signature)
        results.append((k, freeze(A), freeze(B), card))
        return True

    # Exhaustive screened incidence pairings precede reinforced cards.
    for a in range(len(bank)):
        for b in range(a, len(bank)):
            consider(bank[a], bank[b], ("odd-incidence", a, b), True)

    # Twenty deliberate relative-coordinate policies. Their outputs must
    # still pass full invariant and exact-kernel gates; labels alone never
    # establish distinctness. Further epochs serve larger harnesses.
    for attempt in range(max(8192, 128 * count)):
        if len(results) >= count + 8:
            break
        card, epoch = attempt % 20, attempt // 20
        A = bank[(epoch + card // 5) % len(bank)]
        s, t = local.sample(odd, 2)
        if not noncommuting(s, t):
            continue

        pattern = card // 5
        gp = odd if pattern in (0, 2) else even
        hp = odd if pattern in (0, 1) else even
        g, h = local.choice(gp), local.choice(hp)
        if g == h or not noncommuting(g, h):
            continue

        # Policies distinguish quotient alignment relative to s,t.
        determinants = (
            (g[0] * s[1] - g[1] * s[0]) % 3,
            (h[0] * t[1] - h[1] * t[0]) % 3)
        policy = card % 5
        if policy == 0 and 0 not in determinants:
            continue
        if policy == 1 and 0 in determinants:
            continue
        if policy == 2 and determinants[0] != determinants[1]:
            continue
        if policy == 3 and determinants[0] == determinants[1]:
            continue

        B = [[[G.e, g, h], [G.e, s], [G.e, t]]]
        consider(A, B, ("reinforced", card, epoch))

    results.sort(key=lambda item: (-item[0], repr(item[3])))
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable Heisenberg design; call order changes nothing."""
    portfolio = _heis_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct kernel-five-clean Heisenberg portfolio exhausted")
    k, A, B, card = portfolio[idx]
    return {"A": A, "B": B, "screen_k": k,
            "design_card": card, "verified_kernel_floor": 6}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without changing the screened portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Lower constituent six-word multiplicity on the fixed full-rank pendant catalogue eliminates its weight-six logical family: all five measured finalists instead had k=200,d30s=6, with thousands of sector-1 weight-six witnesses.
# REFUTED: Reserved positions separated by the pendant catalogue's Tanner invariants remain evaluator-distinct: the sixty-slot evaluation retained twenty-six distinct codes and thirty-four iso_dup slots.
# REFUTED: The augmentation-rank-two tangent publication improved the measured rank or fitness: all five current finalists remained at k=200,d30s=6 and fitness 12.3290.
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
