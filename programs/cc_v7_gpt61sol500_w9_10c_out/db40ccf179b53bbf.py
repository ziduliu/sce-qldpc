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
CONCEPT = ("Quaternion Q8, order 8, A=B=(3,8): grounding a subdivided three-matching core and optimizing its cubic reinforcement should produce five novel measured distance readings >=7 in cell 0.350; complete kernel-five-clean distance-six fallbacks remain available.")
GROUP = ("Jump from cyclic C8 to nonabelian Q8, x^4=1, y^2=x^2, y*x*y^-1=x^-1. Noncommutativity proves this is not a relabeling. Three noncentral inverse classes give noncommuting order-four directions. The shown Q8 pendant catalogue still measured distance six, so this proposal changes its core incidence rather than assuming a quaternion distance advantage.")
SHAPES = ("Keep equal (3,8) constituents, each with 64 variables and 24 checks. Factor 73 permits q<=8; q=8 gives n=584,kb=200 and guaranteed rate 25/73=0.342466 in cell 0.350. Parent minima were entirely sector-1 Acol/Bcol witnesses, particularly around Acol2/3, with no reported sector-2 minimum. Subdividing a core matching across the third row changes those dependencies without changing shapes.")
FILLING = ("Four degree-two columns form two direct matchings and one subdivided matching. One grounding singleton and three cubic columns complete row totals six and column totals <=3, so BOTH CSS maxima are nine, also below twelve. Twenty support/multiplicity cards optimize genuine relative quaternion labels, exact constituent exclusion through five, weight-six counts and actual kernel witnesses. Full verified lifts require k=200, CSS and complete kernel exclusion through five; the mixed-sector test uses central translation only after verifying it on both check sets. Distinct rooted full-Tanner invariants reject submitted orbit copies conservatively. Constituent seven-cleanliness does not exclude mixed-sector six-logicals; larger witness readings and unseen-archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r035_q8_38x38_subdivided_core_cubic_sixescape_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 has order eight")
    # Element x^a*y^b, with x^4=1 and y^2=x^2.
    els = [(a, b) for b in range(2) for a in range(4)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 2*(b // 2)) % 4, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 2*b*d) % 4,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 4, 0) if b == 0 else ((a + 2) % 4, 1)

    def idx(g):
        a, b = red(g)
        return 4*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _sub_basis(rows):
    out = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = out.get(p)
            if previous is None:
                out[p] = word
                break
            word ^= previous
    return out


def _sub_outside(word, own):
    while word:
        p = word.bit_length() - 1
        previous = own.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _sub_probe(rows, width, local, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, never an unresolved cap."""
    cols = _syndrome_columns(rows, width)
    best = seed

    def inspect(word):
        nonlocal best
        if not word or (best and word.bit_count() >= best.bit_count()):
            return
        if own is not None and not _sub_outside(word, own):
            return
        best = word

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
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

        local.shuffle(kernel)
        for _ in range(2):
            for i, current in enumerate(kernel):
                for j, other in enumerate(kernel):
                    if i != j:
                        trial = current ^ other
                        if trial.bit_count() < current.bit_count():
                            current = trial
                kernel[i] = current
                inspect(current)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 16:
                    for h in range(j):
                        inspect(pair ^ short[h])
        for _ in range(2):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _sub_outside(best, own)
    return best


def _sub_spectrum(rows, width):
    """Exact exclusion through five and exact weight-six multiplicity."""
    cols = _syndrome_columns(rows, width)
    pairs = _short_pairs(cols)
    if pairs is None:
        return None
    triples, collisions, witness = {}, 0, 0
    for i, first in enumerate(cols):
        for j in range(i):
            partial = first ^ cols[j]
            for h in range(j):
                syndrome = partial ^ cols[h]
                if syndrome in pairs:
                    return None
                word = (1 << i) | (1 << j) | (1 << h)
                old = triples.get(syndrome)
                if old is None:
                    triples[syndrome] = (1, word)
                else:
                    multiplicity, representative = old
                    collisions += multiplicity
                    triples[syndrome] = (multiplicity + 1, representative)
                    if not witness:
                        witness = word ^ representative
    assert collisions % 10 == 0
    if witness:
        assert witness.bit_count() == 6
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
    return collisions // 10, witness


def _sub_key(rows, width):
    """Rooted invariant of the complete uncoloured Tanner graph.

    Different keys certify nonisomorphism. Equal keys are discarded
    conservatively; this is not a complete archive-isomorphism oracle.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(rows)
    m = len(rows)
    cols = _syndrome_columns(rows, width)

    def positions(word):
        out = []
        while word:
            bit = word & -word
            out.append(bit.bit_length() - 1)
            word ^= bit
        return out

    adjacency = [[m+j for j in positions(row)] for row in rows]
    adjacency += [positions(col) for col in cols]
    degrees = [len(ns) for ns in adjacency]
    roots = [m+j for j, col in enumerate(cols)
             if col.bit_count() == max(c.bit_count() for c in cols)]

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    rooted = []
    for root in roots:
        colors = compress([(degrees[v], int(v == root))
                           for v in range(len(adjacency))])
        history = []
        for _ in range(4):
            features = [
                (colors[v], tuple(sorted(colors[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            colors = compress(features)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    overlaps = Counter(
        (rows[i].bit_count(), rows[j].bit_count(),
         (rows[i] & rows[j]).bit_count())
        for i in range(m) for j in range(i)
        if rows[i].bit_count() == rows[j].bit_count())
    value = (
        width, m, tuple(sorted(Counter(degrees).items())),
        tuple(sorted(rooted)), tuple(sorted(overlaps.items())))
    return sha256(repr(value).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _subdivided_portfolio(G, count):
    """Pure deterministic optimization; publication grids are immutable."""
    from itertools import combinations_with_replacement

    assert G.q == 8 and A_SHAPE == B_SHAPE == (3, 8)
    q, width, n1 = 8, 64, 512
    cap = min(9, 12, W_MAX)
    mask = (1 << width) - 1
    dummy = [[[]]]
    e, g, h = G.e, (1, 0), (0, 1)
    assert G.mul(g, h) != G.mul(h, g)

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    def component_rows(C, right=False):
        if right:
            _, rows, _ = lifted_matrices(G, dummy, C)
        else:
            rows, _, _ = lifted_matrices(G, C, dummy)
        return [row & mask for row in rows]

    # The core row totals are (4,3,2), including the grounding column.
    # Reinforcement therefore has row totals (2,3,4), not the parent's
    # (4,2,3). Cubic columns are enumerated as count multisets.
    types = [(a, b, 3-a-b)
             for a in range(4) for b in range(4-a)]
    profiles = [
        columns
        for columns in combinations_with_replacement(types, 3)
        if tuple(sum(c[i] for c in columns) for i in range(3))
        == (2, 3, 4)]
    profiles.sort(key=lambda cs: (
        -sum(c == (1, 1, 1) for c in cs),
        sum(max(c) == 3 for c in cs), cs))
    assert profiles

    directions = [x for x in G.elements
                  if x != e and G.mul(x, x) != e]

    def draw(local, profile, policy):
        C = [[[] for _ in range(8)] for _ in range(3)]
        C[0][0] = [e]
        C[0][1], C[1][1] = [e], [e]
        C[0][2], C[1][2] = [e], [g]
        C[1][3], C[2][3] = [e], [e]
        C[2][4], C[0][4] = [e], [h]

        for j, counts in enumerate(profile, 5):
            for i, terms in enumerate(counts):
                if terms == 0:
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    gap = local.choice(directions)
                    if policy == 1:
                        gap = local.choice((g, G.inv(g), h, G.inv(h)))
                    cell = [anchor, G.mul(anchor, gap)]
                else:
                    for _ in range(32):
                        cell = local.sample(G.elements, terms)
                        if policy == 3 or all(
                            G.mul(d, d) != e
                            for a in range(terms) for b in range(a)
                            for d in [G.mul(G.inv(cell[a]), cell[b])]):
                            break
                    else:
                        return None
                C[i][j] = sorted(cell, key=G.idx)
        return C

    def inverse_word(word):
        result = 0
        while word:
            bit = word & -word
            block, t = divmod(bit.bit_length() - 1, q)
            result |= 1 << (block*q + G.idx(G.inv(G.elements[t])))
            word ^= bit
        return result

    bank, fillings = [], set()
    # Twenty optimization cards vary reinforcement incidence and gap
    # policy. Each searches many fillings; none emits a single random draw.
    for card in range(20):
        local = random.Random(BASE_SEED ^ (0x8D1C731 + 104729*card))
        profile = profiles[(card // 4) % len(profiles)]
        policy = card % 4
        records = []
        for trial in range(192):
            C = draw(local, profile, policy)
            if C is None:
                continue
            filling = freeze(C)
            if filling in fillings:
                continue
            fillings.add(filling)
            counts = [[len(cell) for cell in row] for row in C]
            assert all(sum(row) == 6 for row in counts)
            assert all(sum(row[j] for row in counts) <= 3
                       for j in range(8))
            assert _no_dead_lines(counts, counts)
            assert _grid_weights(counts, counts) <= cap
            assert all(len(set(cell)) == len(cell) <= 5
                       for row in C for cell in row)

            left = component_rows(C)
            if gf2_rank(left) != 24:
                continue
            info = _sub_spectrum(left, width)
            if info is None:
                continue
            sixes, seed = info
            records.append((int(sixes != 0), sixes, trial,
                            filling, left, seed))

        records.sort(key=lambda r: r[:3])
        refined = []
        # Equal witness-search budgets, including six-word fallbacks.
        for dirty, sixes, trial, C, left, seed in records[:12]:
            word = _sub_probe(left, width, local, 8, seed)
            if not word or word.bit_count() < 6:
                continue
            right = component_rows(C, True)
            transferred = inverse_word(word)
            if gf2_rank(right) != 24:
                continue
            assert all((row & transferred).bit_count() % 2 == 0
                       for row in right)
            refined.append((
                min(word.bit_count(), 8), -sixes, -trial,
                C, word, transferred, sixes, card))
        refined.sort(key=lambda r: r[:3], reverse=True)
        bank.extend(refined[:4])

    if len(bank) < 8:
        raise ValueError("too few screened subdivided-core constituents")

    plans = sorted(
        (-min(a[0], b[0]), a[6] + b[6],
         int(a[7] == b[7]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))

    z = (2, 0)
    central_map = [G.idx(G.mul(z, x)) for x in G.elements]

    def translate(word):
        moved = 0
        while word:
            bit = word & -word
            block, t = divmod(bit.bit_length() - 1, q)
            moved |= 1 << (block*q + central_map[t])
            word ^= bit
        return moved

    # One representative per central-translation orbit in each sector-2
    # block. No noncentral global symmetry is assumed.
    representatives = [t for t in range(q) if t < central_map[t]]

    def mixed_five(cols, pairs, n):
        """Exact remaining kernel-five test after verified central symmetry."""
        for block in range(n1 // q, n // q):
            for t in representatives:
                anchor = block*q + t
                first = cols[anchor]
                for j, second in enumerate(cols):
                    if j == anchor:
                        continue
                    partial = first ^ second
                    for k in range(j):
                        if k != anchor and (partial ^ cols[k]) in pairs:
                            return True
        return False

    def embedded(record, kind, checks, stabilizers, own):
        source = record[5] if kind == "X" else record[4]
        best = 0
        for block in range(8):
            if kind == "X":
                word = source << (block*width)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*8 + block)*q + t)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _sub_outside(word, own):
                continue
            for _ in range(3):
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

    results, graph_keys = [], set()
    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[3], b[3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(8))
        wz = max(sum(row) + sum(sa[i][j] for i in range(3))
                 for row in sb for j in range(8))
        assert wx == wz == 9
        assert _no_dead_lines(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or k != 200:
            continue
        assert 325*n <= 1000*k < 375*n
        if not css_ok(hx, hz):
            raise ValueError("verified subdivided lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= cap

        bx, bz = _sub_basis(hx), _sub_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0x8D2C731 + 65537*ai + bi))
        xword = _sub_probe(hz, n, local, 2, xseed, bx)
        zword = _sub_probe(hx, n, local, 2, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 6:
            continue

        key = _sub_key(hx + hz, n)
        if key in graph_keys:
            continue
        graph_keys.add(key)

        # Verify the only symmetry used by the exhaustive mixed test.
        if (set(map(translate, hx)) != set(hx)
                or set(map(translate, hz)) != set(hz)):
            raise ValueError("built Q8 matrices failed central symmetry")
        clean = True
        for checks in (hx, hz):
            cols = _syndrome_columns(checks, n)
            pairs = _short_pairs(cols)
            if pairs is None or mixed_five(cols, pairs, n):
                clean = False
                break
        if not clean:
            continue

        quality = (reading, min(a[0], b[0]), -a[6]-b[6])
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a[7], b[7]), (a[6], b[6]), key.hex()))
        if len(results) >= count:
            break

    results.sort(key=lambda r: (r[0], r[7]), reverse=True)
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure lookup; exhausted portfolios are never recycled or relabelled."""
    portfolio = _subdivided_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel screened subdivided-core portfolio exhausted")
    quality, A, B, k, reading, cards, sixes, signature = portfolio[idx]
    return {"A": A, "B": B, "screen_k": k,
            "design_cards": cards, "component_six_counts": sixes,
            "logical_witness_upper_bound": reading,
            "verified_complete_kernel_floor": 6,
            "selection_quality": quality, "tanner_signature": signature}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without sampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Singleton-dominated uniform fillings on Z17 (2,5)x(2,6) did not eliminate sector-1 weight-four logicals: all five measured parent codes had d30s=4.
# REFUTED: Weight-five multiplicity targets on fixed six-edge/two-triple Z8 components did not exceed d30s=5: all five measured codes had k=200,d30s=5, and sixty slots collapsed to nineteen classes.
# REFUTED: Lower constituent six-word multiplicity on the fixed full-rank pendant catalogue eliminates its weight-six logical family: all five measured finalists instead had k=200,d30s=6, with thousands of sector-1 weight-six witnesses.
# REFUTED: Reserved positions separated by the pendant catalogue's Tanner invariants remain evaluator-distinct: the sixty-slot evaluation retained twenty-six distinct codes and thirty-four iso_dup slots.
# REFUTED: Augmentation-rank-two tangent proposals supply measured rank-surplus finalists: all five current finalists instead had k=200,k-kb=0,d30s=6; sixty slots retained twelve distinct codes and three archive duplicates.
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
