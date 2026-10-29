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
CONCEPT = ("Dicyclic Dic5, order 20, A=B=(2,5): verified girth-eight matching cores and exact constituent exclusion through seven should supply five novel measured distance-eight leaders in cell 0.300; constituent success does not certify mixed-sector quantum distance.")
GROUP = ("Jump from centre-trivial affine C5:C4 to Dic5: x^10=1, y^2=x^5, y*x*y^-1=x^-1. Its centre has order two, and order-ten rotations replace the affine order-five translation subgroup. The parent measured distance seven throughout its finalists, fitness 15.2070, with extensive isomorphic collapse. Different attachment relations motivate this experiment; no measured Dic5 advantage is assumed.")
SHAPES = ("Retain equal (2,5) constituents, each with 100 variables and 40 checks. Factor 29 allows q<=20; q=20 gives n=580, kb=180 and guaranteed rate 9/29=0.310345 in cell 0.300. Parent minima occupied sector-1 Acol/Bcol blocks, especially Acol1/2, without reported Arow/Brow minima. Verified complete ranks enforce the target cell.")
FILLING = ("Twenty matching-core/support/relative-order prescriptions optimize split-binomial or isolated-trinomial cubic attachments. Every constituent row totals six and column totals two or three: BOTH CSS maxima are nine, below twelve. All screening uses verified lifts. Exact pair/triple and anchored quartet/triple screens return attained constituent witnesses; distance-eight constituents precede explicit six/seven fallbacks. Embedded logical witnesses undergo stabilizer descent and full-kernel probing. Complete ranks, CSS and logical-four gates precede publication. Optional uncoloured Tanner certificates, otherwise unequal intrinsic invariants, separate entries. Shown affine pairings are exclusion references only; unseen archive novelty and quantum distance eight remain measurement hypotheses.")
FAMILY_NAME = "r030_dic5_q20_25x25_attained_kernel8_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic5 has order 20")
    # x^10=1, y^2=x^5, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 5 * (b // 2)) % 10, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 5*b*d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 10, 0) if b == 0 else ((a + 5) % 10, 1)

    def idx(g):
        a, b = red(g)
        return a + 10*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _sd_columns(rows, n):
    """Column syndromes of a supplied, verified binary matrix."""
    columns = [0] * n
    for i, row in enumerate(rows):
        bit = 1 << i
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low
    return columns


def _sd_pairs(columns):
    """None means a classical dependency of weight at most four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = set()
    for i, x in enumerate(columns):
        for y in columns[i + 1:]:
            syndrome = x ^ y
            if syndrome in singles or syndrome in pairs:
                return None
            pairs.add(syndrome)
    return pairs


def _sd_fives(columns, pairs, cap=80):
    """Count triple/pair matches, capped; each weight-five word counts ten."""
    count = 0
    for i in range(len(columns) - 2):
        for j in range(i + 1, len(columns) - 1):
            syndrome = columns[i] ^ columns[j]
            for z in columns[j + 1:]:
                if syndrome ^ z in pairs:
                    count += 1
                    if count >= cap:
                        return cap
    return count


def _sd_basis(rows):
    basis = {}
    for value in rows:
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return basis


def _sd_in_span(value, basis):
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def _sd_no_short_logical(checks, stabilizers, n):
    """Exact rejection of nontrivial logicals of weights one through four.

    Compare syndromes of all supports of weight <=2. A collision gives
    a commuting support of weight <=4; span membership distinguishes
    stabilizers from logicals.
    """
    columns = _sd_columns(checks, n)
    basis = _sd_basis(stabilizers)
    seen = {0: 0}

    def accept(syndrome, support):
        if syndrome in seen:
            return _sd_in_span(support ^ seen[syndrome], basis)
        seen[syndrome] = support
        return True

    for i, syndrome in enumerate(columns):
        if not accept(syndrome, 1 << i):
            return False
    for i, x in enumerate(columns):
        bit = 1 << i
        for j in range(i + 1, n):
            if not accept(x ^ columns[j], bit | (1 << j)):
                return False
    return True


def _sd_components(hx, hz, q):
    """Extract components from verified lifts, not a separate lift model."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    arows = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            arows.append(row)
    bmask = (1 << (cb * q)) - 1
    brows = [hz[a * q + t] & bmask
             for a in range(rb) for t in range(q)]
    return (_sd_columns(arows, ca * q),
            _sd_columns(brows, cb * q))


def _af_clean6(columns, pairs):
    """Exact constituent exclusion through six, given a clean <=4 test."""
    singles = set(columns)
    triples = set()
    for i, first in enumerate(columns):
        for j in range(i):
            pair = first ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if (syndrome == 0 or syndrome in singles
                        or syndrome in pairs or syndrome in triples):
                    return False
                triples.add(syndrome)
    return True


def _af_probe(columns, local, rounds, own=None):
    """Search actual kernel witnesses; twelve is an unresolved cutoff."""
    width = len(columns)
    best = 12
    order = list(range(width))

    def inspect(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if own is not None and _sd_in_span(word, own):
            return
        best = word.bit_count()

    for _ in range(rounds):
        local.shuffle(order)
        pivots, nulls = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, word)
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome:
                nulls.append(word)
                inspect(word)
        for i, word in enumerate(nulls):
            for other in nulls[:i]:
                inspect(word ^ other)
        short = sorted(nulls, key=int.bit_count)[:20]
        for i, word in enumerate(short):
            for j in range(i):
                pair = word ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if best <= 6:
            break
    return best


def _af_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _af_component_rows(G, side, grid):
    """Obtain the constituent from the verified left/right construction."""
    dummy = [[[G.e]]]
    mask = (1 << (5 * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz]


def _af_bank(G, side):
    """Optimize six structural prescriptions, with no cross-slot state."""
    from itertools import combinations_with_replacement

    local = random.Random(BASE_SEED ^ (0xAF520 if side == "A" else 0xAF521))
    spectra = ((2, 2, 5), (2, 4, 4), (4, 4, 5))
    policies = list(combinations_with_replacement(range(3), 2))

    def order(g):
        power = G.e
        for length in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return length
        raise ValueError("broken group order")

    orders = {g: order(g) for g in G.elements}
    nonidentity = [g for g in G.elements if g != G.e]
    triangles = {s: [] for s in spectra}
    cores = []
    for g in nonidentity:
        for h in nonidentity:
            if g == h or G.mul(g, h) == G.mul(h, g):
                continue
            relative = G.mul(G.inv(g), h)
            spectrum = tuple(sorted((orders[g], orders[h], orders[relative])))
            if spectrum in triangles:
                triangles[spectrum].append((g, h))
            if orders[g] == orders[h] == 4 and orders[relative] == 5:
                cores.append((g, h))
    if not cores or any(not triangles[s] for s in spectra):
        raise ValueError("required affine core or triangle spectrum absent")

    buckets = [[] for _ in policies]
    fingerprints = [set() for _ in policies]
    for attempt in range(192):
        policy = attempt % len(policies)
        s0, s1 = policies[policy]
        g, h = local.choice(cores)
        grid = [
            [[G.e], [G.e], [G.e], [], []],
            [[G.e], [g], [h], [], []],
        ]
        for row, col, spectrum_id in ((0, 3, s0), (1, 4, s1)):
            r, s = local.choice(triangles[spectra[spectrum_id]])
            anchor = G.pick(local)
            cell = [anchor, G.mul(anchor, r), G.mul(anchor, s)]
            if len(set(cell)) != 3:
                raise ValueError("triangle contains duplicate elements")
            grid[row][col] = cell

        fingerprint = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
        if fingerprint in fingerprints[policy]:
            continue
        fingerprints[policy].add(fingerprint)
        rows = _af_component_rows(G, side, grid)
        columns = _sd_columns(rows, 5 * G.q)
        pairs = _sd_pairs(columns)
        if pairs is None:
            continue
        clean6 = _af_clean6(columns, pairs)
        reading = _af_probe(columns, local, 3)
        # A failed exact-six test exposes a word <=6, even if probing
        # misses it. Never let that candidate advertise a higher cutoff.
        if not clean6:
            reading = min(reading, 6)
        reading = min(reading, 8)
        cycles = _af_cycles(rows)
        key = (reading, clean6, -cycles)
        buckets[policy].append((key, attempt, grid, policy))
        buckets[policy].sort(key=lambda item: (item[0], -item[1]),
                             reverse=True)
        del buckets[policy][3:]

    bank = [entry for bucket in buckets for entry in bucket]
    if not bank:
        raise ValueError("no affine constituent passed exact weight-four screen")
    return bank


def _af_signature(hx, hz, n):
    """Invariant of the full uncoloured Tanner graph.

    Unequal signatures certify non-isomorphism. Equal signatures are
    rejected conservatively; they are not an isomorphism certificate.
    """
    rows = list(hx) + list(hz)
    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlaps.append((i, j, value))
            if value:
                relation = one if value == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [{} for _ in rows]
    for i, j, value in overlaps:
        key = (value,
               rows[j].bit_count(),
               (one[i] & one[j]).bit_count(),
               (one[i] & many[j]).bit_count()
               + (many[i] & one[j]).bit_count(),
               (many[i] & many[j]).bit_count())
        profiles[i][key] = profiles[i].get(key, 0) + 1
        reverse_key = (value, rows[i].bit_count()) + key[2:]
        profiles[j][reverse_key] = profiles[j].get(reverse_key, 0) + 1

    columns = _sd_columns(rows, n)
    degrees = tuple(sorted(column.bit_count() for column in columns))
    roots = tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profile.items())))
        for i, profile in enumerate(profiles)))
    return degrees, roots


from functools import lru_cache


@lru_cache(maxsize=1)
def _af_endpoint_portfolio(G, count):
    """Pure deterministic search; cached published cells are immutable."""
    from itertools import combinations
    from collections import Counter
    import hashlib

    q, e = G.q, G.e
    assert q == 20
    width = 5*q
    support = ((1, 1, 1, 2, 1), (1, 1, 1, 1, 2))
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) == 9 <= W_MAX

    def digest(hx, hz, n):
        # Hashing preserves the conservative invariant-separation rule:
        # a hash collision discards a candidate, never creates novelty.
        value = _af_signature(hx, hz, n)
        return hashlib.sha256(repr(value).encode("ascii")).digest()

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid affine element order")

    orders = {g: order(g) for g in G.elements}
    ratios = {d: [g for g in G.elements if orders[g] == d]
              for d in (2, 4, 5)}
    nonidentity = [g for g in G.elements if g != e]

    def core(g, h):
        """Exact graph distances and a shortest cycle from verified rows."""
        grid = [[[e], [e], [e]], [[e], [g], [h]]]
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
        mask = (1 << (3*q)) - 1
        rows = [row & mask for row in rows]
        cols = _sd_columns(rows, 3*q)
        adjacency = [[] for _ in rows]
        edges = set()
        for j, syndrome in enumerate(cols):
            if syndrome.bit_count() != 2:
                raise ValueError("core is not a degree-two graph")
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            edge = tuple(sorted((a, b)))
            if edge in edges:
                raise ValueError("parallel matching-core edges")
            edges.add(edge)
            adjacency[a].append((b, j))
            adjacency[b].append((a, j))
        distances, paths, shortest = [], [], 0
        for root in range(len(rows)):
            dist = [10000] * len(rows)
            path = [0] * len(rows)
            dist[root] = 0
            queue = [root]
            for a in queue:
                for b, j in adjacency[a]:
                    if dist[b] == 10000:
                        dist[b] = dist[a] + 1
                        path[b] = path[a] ^ (1 << j)
                        queue.append(b)
                    else:
                        word = path[a] ^ path[b] ^ (1 << j)
                        if word and (not shortest or
                                     word.bit_count() < shortest.bit_count()):
                            shortest = word
            if len(queue) != len(rows):
                raise ValueError("disconnected prescribed matching core")
            distances.append(tuple(dist))
            paths.append(tuple(path))
        if not shortest or shortest.bit_count() != 8:
            raise ValueError("prescribed core does not have girth eight")
        assert all((row & shortest).bit_count() % 2 == 0 for row in rows)
        return g, h, tuple(cols), tuple(distances), tuple(paths), shortest

    # Two relative-order geometries, independently verified rather than
    # declared clean from their group presentation.
    cores = (
        core((1, 0), (0, 1)),
        core((0, 1), (1, 1)),
    )

    def cheap_two(cols, data):
        """Reject an actual <=6 witness using two cubic variables."""
        distances, paths = data[3], data[4]
        memo = {}

        def join(boundary, budget):
            if not boundary:
                return 0
            if budget < 0 or boundary.bit_count() > 2*budget:
                return None
            key = boundary, budget
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
                    tail = join(rest ^ other, budget - distance)
                    if tail is not None:
                        answer = paths[a][b] ^ tail
                        break
                choices ^= other
            memo[key] = answer
            return answer

        for anchor in (3*q, 4*q):
            for other in range(3*q, width):
                if other == anchor:
                    continue
                answer = join(cols[anchor] ^ cols[other], 4)
                if answer is not None:
                    word = (1 << anchor) ^ (1 << other) ^ answer
                    assert 0 < word.bit_count() <= 6
                    return True
        return False

    def profile(rows, data):
        """Exact floor through seven and explicit seven/eight witnesses.

        After excluding <=6, triple syndromes are unique. A seven-word
        has at least two cubic variables: a pure core word is a cycle,
        and the core has girth eight. A regular-action translate puts
        one cubic variable at an identity-coordinate cubic anchor.
        Comparing those anchored quartets with ALL triples is exhaustive
        for existence of a constituent seven-word.

        The exposure count is an anchored construction statistic, not
        an uncoloured Tanner invariant or a quantum-distance certificate.
        """
        cols = _sd_columns(rows, width)
        assert tuple(cols[:3*q]) == data[2]
        assert all(c.bit_count() == 3 for c in cols[3*q:])
        if cheap_two(cols, data):
            return None
        pairs = _sd_pairs(cols)
        if pairs is None:
            return None
        singles = set(cols)
        bits = [1 << j for j in range(width)]
        triples = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = bits[i] | bits[j]
                for h in range(j):
                    value = syndrome ^ cols[h]
                    if (value == 0 or value in singles
                            or value in pairs or value in triples):
                        return None
                    triples[value] = word | bits[h]
        del pairs

        sevens = set()
        witness = data[5]
        cap = 128
        for anchor in (3*q, 4*q):
            others = [j for j in range(width) if j != anchor]
            for a, b, c in combinations(others, 3):
                value = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                previous = triples.get(value)
                if previous is None:
                    continue
                word = (bits[anchor] | bits[a] | bits[b] | bits[c]) ^ previous
                assert word.bit_count() == 7
                sevens.add(word)
                witness = word
                if len(sevens) >= cap:
                    break
            if len(sevens) >= cap:
                break
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
        return (7 if sevens else 8), len(sevens), witness

    pair_orders = ((2, 2), (2, 4), (2, 5),
                   (4, 4), (4, 5), (5, 5))
    relation_pairs = []
    for policy in range(10):
        choices = []
        for r in nonidentity:
            for s in nonidentity:
                commute = G.mul(r, s) == G.mul(s, r)
                if policy < 6:
                    allowed = (orders[r], orders[s]) == pair_orders[policy]
                elif policy == 6:
                    allowed = commute
                elif policy == 7:
                    allowed = not commute
                elif policy == 8:
                    allowed = G.mul(r, r) == G.mul(s, s)
                else:
                    allowed = G.mul(r, r) != G.mul(s, s)
                if allowed:
                    choices.append((r, s))
        if not choices:
            raise ValueError("empty prescribed reinforcement relation")
        relation_pairs.append(tuple(choices))

    records = []
    for card in range(20):
        data = cores[card // 10]
        g, h = data[:2]
        local = random.Random(BASE_SEED ^ (0xA7E020 + 104729*card))
        candidates, fillings = [], set()
        for trial in range(384):
            r, s = local.choice(relation_pairs[card % 10])
            a, b = G.pick(local), G.pick(local)
            C = [
                [[e], [e], [e], [a, G.mul(a, r)], [e]],
                [[e], [g], [h], [e], [b, G.mul(b, s)]],
            ]
            filling = tuple(tuple(sorted(G.idx(z) for z in cell))
                            for row in C for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _af_component_rows(G, "A", C)
            if gf2_rank(rows) != 2*q:
                continue
            screened = profile(rows, data)
            if screened is None:
                continue
            floor, exposure, witness = screened
            quality = (floor, -exposure, -_af_cycles(rows), -trial)
            candidates.append((quality, C, witness, card, trial, rows))
        candidates.sort(key=lambda item: item[0], reverse=True)
        records.extend(candidates[:8])

    records.sort(key=lambda item: (item[0], -item[3]), reverse=True)
    bank, constituent_counts = [], Counter()
    for quality, C, left_word, card, trial, rows in records:
        key = digest(rows, [], width)
        # Retain a few embeddings of the same constituent graph because
        # their LP compatibility need not follow from component isomorphism.
        # Only the complete-code invariant gates final publication.
        if constituent_counts[key] >= 3:
            continue
        right_rows = _af_component_rows(G, "B", C)
        if gf2_rank(right_rows) != 2*q:
            continue
        right_word, source = 0, left_word
        while source:
            bit = source & -source
            j, t = divmod(bit.bit_length() - 1, q)
            right_word |= 1 << (j*q + G.idx(G.inv(G.elements[t])))
            source ^= bit
        assert all((row & right_word).bit_count() % 2 == 0
                   for row in right_rows)
        constituent_counts[key] += 1
        bank.append((quality, C, left_word, right_word, card, trial))
    if len(bank) < 3:
        raise ValueError("too few constituent-seven-clean split designs")

    excluded = set()

    def exclude_pair(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(digest(hx, hz, n))

    # Reconstruct the parent's banks solely as exclusion references.
    # Excluding all their pairings is conservative, including unused ones.
    old_a, old_b = _af_bank(G, "A"), _af_bank(G, "B")
    for a in old_a:
        for b in old_b:
            exclude_pair(a[2], b[2])

    # Shown archived distance-seven constituents; never proposal seeds.
    known = (
        [[[e], [e], [e], [(4, 0), (2, 3)], [e]],
         [[e], [(4, 0)], [(4, 1)], [e], [(3, 2), (1, 3)]]],
        [[[e], [e], [e], [(2, 1), (4, 1)], [e]],
         [[e], [(2, 0)], [(0, 1)], [e], [(2, 2), (3, 2)]]],
        [[[e], [e], [e], [(4, 0), (4, 1)], [e]],
         [[e], [(4, 0)], [(3, 1)], [e], [(3, 2), (1, 3)]]],
        [[[e], [e], [e], [(3, 0), (1, 3)], [e]],
         [[e], [(1, 3)], [(2, 3)], [e], [(0, 1), (1, 1)]]],
        [[[e], [e], [e], [(2, 2), (3, 2)], [e]],
         [[e], [(0, 3)], [(3, 3)], [e], [(0, 1), (2, 1)]]],
        [[[(0, 1), (2, 1)], [(3, 2)], [e], [e], [e]],
         [[(0, 2)], [e, (3, 1)], [e], [(2, 0)], [(1, 3)]]],
        [[[(4, 2), (4, 3)], [(4, 2)], [e], [e], [e]],
         [[(2, 0)], [(1, 2), (1, 3)], [e], [(1, 0)], [(1, 1)]]],
        [[[(3, 0), (4, 0)], [(2, 1)], [e], [e], [e]],
         [[(1, 2)], [(0, 3), (4, 3)], [e], [(4, 0)], [(4, 3)]]],
    )
    for ai, A in enumerate(known):
        for B in known[ai:]:
            exclude_pair(A, B)

    def embedded(record, kind, checks, stabilizers, own):
        """Produce and descend genuine nontrivial logical witnesses."""
        best = 0
        for block in range(5):
            if kind == "X":
                word = record[3] << (block*width)
            else:
                word, source = 0, record[2]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if _sd_in_span(word, own):
                continue
            for _ in range(3):
                starting = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting_pair = word
                moves = [row for row in stabilizers
                         if (row & starting_pair).bit_count() >= 2]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting_pair ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and not _sd_in_span(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            preliminary = (
                min(a[0][0], b[0][0]),
                a[0][1] + b[0][1],
                a[0][2] + b[0][2],
                int(a[4] != b[4]))
            plans.append((preliminary, ai, bi))
    plans.sort(key=lambda item: (item[0], -item[1], -item[2]),
               reverse=True)

    results, tested = [], 0
    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[1], b[1]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or k != 180:
            continue
        signature = digest(hx, hz, n)
        if signature in excluded:
            continue
        excluded.add(signature)
        tested += 1
        if tested > max(640, 12*count):
            break
        if not _sd_no_short_logical(hz, hx, n):
            continue
        if not _sd_no_short_logical(hx, hz, n):
            continue
        bx, bz = _sd_basis(hx), _sd_basis(hz)
        xword = embedded(b, "X", hz, hx, bx)
        zword = embedded(a, "Z", hx, hz, bz)
        if not xword or not zword:
            continue
        local = random.Random(BASE_SEED ^ (0xA7F020 + 257*ai + bi))
        dx = _af_probe(_sd_columns(hz, n), local, 4, bx)
        dz = _af_probe(_sd_columns(hx, n), local, 4, bz)
        # Explicit seeds prevent an unresolved probe cutoff from being
        # presented as an attained witness upper bound.
        reading = min(dx, dz, xword.bit_count(), zword.bit_count())
        if reading < 7:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k*reading*reading/n, preliminary)
        results.append((quality, freeze(A), freeze(B), k, reading,
                        (a[4], b[4]), (a[0][0], b[0][0])))
        if len(results) >= count:
            break
    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results)


@lru_cache(maxsize=1)
def _di_portfolio(G, count):
    """Deterministic search; cached published cells are immutable."""
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    assert G.q == 20 and A_SHAPE == B_SHAPE == (2, 5)
    q, e, width = G.q, G.e, 5*G.q
    cap = min(9, W_MAX)
    if cap < 9:
        raise ValueError("Dic5 reinforcement profiles require cap nine")
    try:
        import pynauty
    except ImportError:
        pynauty = None

    def rows_for(C, right=False):
        # Only scaffold-built regular actions are used.
        dummy = [[[]]]
        if right:
            _, rows, _ = lifted_matrices(G, dummy, C)
        else:
            rows, _, _ = lifted_matrices(G, C, dummy)
        mask = (1 << (len(C[0])*q)) - 1
        return [row & mask for row in rows]

    def graph_key(hx, hz, n):
        rows = list(hx) + list(hz)
        if pynauty is None:
            value = _af_signature(hx, hz, n)
            return sha256(repr(value).encode("ascii")).digest()
        m = len(rows)
        adjacency = {i: [] for i in range(m+n)}
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
            number_of_vertices=m+n, directed=False,
            adjacency_dict=adjacency,
            vertex_coloring=[colors[d] for d in sorted(colors)])
        return pynauty.certificate(graph)

    def element_order(g):
        power = e
        for length in range(1, q+1):
            power = G.mul(power, g)
            if power == e:
                return length
        raise ValueError("invalid Dic5 element order")

    orders = {g: element_order(g) for g in G.elements}
    nonidentity = [g for g in G.elements if g != e]

    def core_data(g, h):
        C = [[[e], [e], [e]], [[e], [g], [h]]]
        rows = rows_for(C)
        cols = _sd_columns(rows, 3*q)
        adjacency = [[] for _ in rows]
        edges = set()
        for j, col in enumerate(cols):
            if col.bit_count() != 2:
                return None
            bit = col & -col
            a = bit.bit_length()-1
            b = (col ^ bit).bit_length()-1
            edge = tuple(sorted((a, b)))
            if edge in edges:
                return None
            edges.add(edge)
            adjacency[a].append((b, j))
            adjacency[b].append((a, j))
        distances, paths, cycle = [], [], 0
        for root in range(len(rows)):
            distance = [10000]*len(rows)
            path = [0]*len(rows)
            distance[root] = 0
            queue = [root]
            for a in queue:
                for b, j in adjacency[a]:
                    if distance[b] == 10000:
                        distance[b] = distance[a]+1
                        path[b] = path[a] ^ (1 << j)
                        queue.append(b)
                    else:
                        word = path[a] ^ path[b] ^ (1 << j)
                        if word and (not cycle or
                                     word.bit_count() < cycle.bit_count()):
                            cycle = word
            if len(queue) != len(rows):
                return None
            distances.append(tuple(distance))
            paths.append(tuple(path))
        assert cycle
        assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
        return (g, h, tuple(cols), tuple(distances), tuple(paths), cycle)

    cores = []
    for g in nonidentity:
        for h in nonidentity:
            if G.mul(g, h) == G.mul(h, g):
                continue
            data = core_data(g, h)
            if data is not None:
                cores.append(data)
    if not cores:
        raise ValueError("no connected verified Dic5 matching core")
    girth = max(data[5].bit_count() for data in cores)
    cores = [data for data in cores if data[5].bit_count() == girth]
    core_pools = [
        [data for data in cores
         if (orders[data[0]] == orders[data[1]]) == bool(kind)]
        for kind in range(2)]
    for kind in range(2):
        if not core_pools[kind]:
            core_pools[kind] = list(cores)

    pair_orders = ((4, 4), (5, 10), (10, 10), (4, 10), None)
    relative_pairs = []
    triangle_pools = []
    for policy in range(5):
        pairs, triangles = [], []
        for r in nonidentity:
            for s in nonidentity:
                if r == s:
                    continue
                requested = pair_orders[policy]
                if requested is None or (orders[r], orders[s]) == requested:
                    pairs.append((r, s))
                relative = G.mul(G.inv(r), s)
                spectrum = (orders[r], orders[s], orders[relative])
                commute = G.mul(r, s) == G.mul(s, r)
                allowed = (
                    (policy == 0 and not commute)
                    or (policy == 1 and commute and min(spectrum) >= 5)
                    or (policy == 2 and min(spectrum) >= 4)
                    or (policy == 3 and 2 in spectrum)
                    or (policy == 4))
                if allowed:
                    triangles.append((r, s))
        relative_pairs.append(pairs or [
            (r, s) for r in nonidentity for s in nonidentity if r != s])
        triangle_pools.append(triangles or [
            (r, s) for r in nonidentity for s in nonidentity if r != s])

    def cheap_two(cols, data):
        """Find actual <=6 words with two cubic variables."""
        distances, paths = data[3], data[4]
        memo = {}

        def join(boundary, budget):
            if not boundary:
                return 0
            if budget < 0 or boundary.bit_count() > 2*budget:
                return None
            key = boundary, budget
            if key in memo:
                return memo[key]
            bit = boundary & -boundary
            a = bit.bit_length()-1
            rest = boundary ^ bit
            choices, answer = rest, None
            while choices:
                other = choices & -choices
                b = other.bit_length()-1
                d = distances[a][b]
                if d <= budget:
                    tail = join(rest ^ other, budget-d)
                    if tail is not None:
                        answer = paths[a][b] ^ tail
                        break
                choices ^= other
            memo[key] = answer
            return answer

        for anchor in (3*q, 4*q):
            for other in range(3*q, width):
                if anchor == other:
                    continue
                answer = join(cols[anchor] ^ cols[other], 4)
                if answer is not None:
                    word = (1 << anchor) ^ (1 << other) ^ answer
                    assert 0 < word.bit_count() <= 6
                    return word
        return 0

    def exact_component(rows, data, fallback=False):
        """Attained constituent distance, with exact screens through seven.

        Every <=6 word splits into supports of size <=3. If these are
        absent, every seven-word contains a cubic variable because the
        degree-two core has girth >=8. A regular-action translate puts
        that variable at one of the two cubic identity-coordinate anchors.
        No regular-action assumption is made for complete LP matrices.
        """
        cols = _sd_columns(rows, width)
        assert tuple(cols[:3*q]) == data[2]
        assert all(col.bit_count() == 3 for col in cols[3*q:])
        short = cheap_two(cols, data)
        if short and not fallback:
            return None
        if _sd_pairs(cols) is None:
            return None
        bits = [1 << j for j in range(width)]
        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                pairs[first ^ cols[j]] = bits[i] | bits[j]
        triples = {}
        best = data[5]
        if short and short.bit_count() < best.bit_count():
            best = short
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                pair_word = bits[i] | bits[j]
                for h in range(j):
                    value = syndrome ^ cols[h]
                    word = pair_word | bits[h]
                    previous = pairs.get(value)
                    if previous is not None:
                        witness = previous ^ word
                        assert witness.bit_count() == 5
                        return None
                    previous = triples.get(value)
                    if previous is not None:
                        witness = previous ^ word
                        assert witness.bit_count() == 6
                        if witness.bit_count() < best.bit_count():
                            best = witness
                    else:
                        triples[value] = word
        if best.bit_count() <= 6:
            assert best.bit_count() == 6
            assert all((row & best).bit_count() % 2 == 0 for row in rows)
            return 6, best

        for anchor in (3*q, 4*q):
            others = [j for j in range(width) if j != anchor]
            for a, b, c in combinations(others, 3):
                previous = triples.get(
                    cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c])
                if previous is not None:
                    witness = previous ^ bits[anchor] ^ bits[a] ^ bits[b] ^ bits[c]
                    assert witness.bit_count() == 7
                    assert all((row & witness).bit_count() % 2 == 0
                               for row in rows)
                    return 7, witness
        assert best.bit_count() == girth
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return girth, best

    records, filling_keys = [], set()
    for card in range(20):
        profile, policy = divmod(card, 10)
        core_kind = policy // 5
        policy %= 5
        local = random.Random(BASE_SEED ^ (0xD1C520 + 104729*card))
        candidates, fallback_candidates = [], []
        for trial in range(192):
            data = local.choice(core_pools[core_kind])
            g, h = data[:2]
            C = [[[e], [e], [e], [], []],
                 [[e], [g], [h], [], []]]
            if profile == 0:
                r, s = local.choice(relative_pairs[policy])
                a, b = G.pick(local), G.pick(local)
                C[0][3], C[1][3] = [a, G.mul(a, r)], [e]
                C[0][4], C[1][4] = [e], [b, G.mul(b, s)]
            else:
                r, s = local.choice(triangle_pools[policy])
                u, v = local.choice(triangle_pools[(policy+2) % 5])
                C[0][3], C[1][4] = [e, r, s], [e, u, v]
            filling = tuple(tuple(sorted(G.idx(z) for z in cell))
                            for row in C for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            rows = rows_for(C)
            if gf2_rank(rows) != 2*q:
                continue
            screened = exact_component(rows, data)
            if screened is None:
                # Reserve a bounded, explicitly weaker fallback pool.
                if len(fallback_candidates) < 8:
                    fallback_candidates.append((trial, C, rows, data))
                continue
            floor, witness = screened
            candidates.append(
                ((floor, -_af_cycles(rows), -trial),
                 C, rows, witness, card))
            if sum(item[0][0] >= 8 for item in candidates) >= 5:
                break
        if len(candidates) < 4:
            for trial, C, rows, data in fallback_candidates:
                screened = exact_component(rows, data, fallback=True)
                if screened is not None:
                    floor, witness = screened
                    candidates.append(
                        ((floor, -_af_cycles(rows), -trial),
                         C, rows, witness, card))
        candidates.sort(key=lambda item: item[0], reverse=True)
        records.extend(candidates[:6])

    records.sort(key=lambda item: (item[0], -item[4]), reverse=True)
    bank, component_counts = [], Counter()
    for quality, C, rows, left_word, card in records:
        signature = graph_key(rows, [], width)
        if component_counts[signature] >= 3:
            continue
        right_rows = rows_for(C, right=True)
        if gf2_rank(right_rows) != 2*q:
            continue
        right_word, source = 0, left_word
        while source:
            bit = source & -source
            j, t = divmod(bit.bit_length()-1, q)
            right_word |= 1 << (j*q + G.idx(G.inv(G.elements[t])))
            source ^= bit
        assert all((row & right_word).bit_count() % 2 == 0
                   for row in right_rows)
        component_counts[signature] += 1
        bank.append((quality, C, left_word, right_word, card))
    if len(bank) < 3:
        raise ValueError("too few attained Dic5 constituent designs")

    # Rebuild shown affine constituents only to exclude their Tanner
    # graphs, even if a different regular presentation realizes one.
    old_elements = [(a, b) for b in range(4) for a in range(5)]
    powers = (1, 2, 4, 3)
    oldG = _G(dict(
        elements=old_elements, ident=(0, 0),
        mul=lambda g, h: ((g[0]+powers[g[1]]*h[0]) % 5,
                          (g[1]+h[1]) % 4),
        inv=lambda g: ((-powers[(-g[1]) % 4]*g[0]) % 5,
                       (-g[1]) % 4)), 20)
    oe = oldG.e
    references = (
        ((1,0), (0,1), ((2,2),(3,2)), ((2,1),(0,2))),
        ((1,0), (0,1), ((3,0),(2,1)), ((3,1),(1,2))),
        ((1,0), (0,1), ((3,0),(2,1)), ((4,1),(1,3))),
        ((1,0), (0,1), ((0,2),(3,2)), ((1,3),(2,3))),
        ((0,1), (1,1), ((2,0),(1,1)), ((4,1),(0,2))),
    )
    known = [
        [[[oe], [oe], [oe], list(a), [oe]],
         [[oe], [g], [h], [oe], list(b)]]
        for g, h, a, b in references]
    excluded = set()
    for ai, A in enumerate(known):
        for B in known[ai:]:
            hx, hz, n = lifted_matrices(oldG, A, B)
            excluded.add(graph_key(hx, hz, n))

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(5):
            if kind == "X":
                word = record[3] << (block*width)
            else:
                word, source = 0, record[2]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*5+block)*q+t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if _sd_in_span(word, own):
                continue
            for _ in range(2):
                starting, weight = word, word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                start = word
                moves = [row for row in stabilizers
                         if (row & start).bit_count() >= 2]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = start ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and not _sd_in_span(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    plans = sorted(
        (-min(a[0][0], b[0][0]), -a[0][0]-b[0][0],
         -a[0][1]-b[0][1], int(a[4] == b[4]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))
    results, tested = [], 0
    for _, _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[1], b[1]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        wx = max(sum(row)+sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row)+sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(5))
        assert wx == wz == 9 <= cap
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 580 or not (275*n <= 1000*k < 325*n):
            continue
        signature = graph_key(hx, hz, n)
        if signature in excluded:
            continue
        excluded.add(signature)
        tested += 1
        if tested > max(240, 6*count):
            break
        if not _sd_no_short_logical(hz, hx, n):
            continue
        if not _sd_no_short_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= cap
        bx, bz = _sd_basis(hx), _sd_basis(hz)
        xword = embedded(b, "X", hz, hx, bx)
        zword = embedded(a, "Z", hx, hz, bz)
        if not xword or not zword:
            continue
        local = random.Random(BASE_SEED ^ (0xD1C580 + 8191*ai + bi))
        dx = _af_probe(_sd_columns(hz, n), local, 4, bx)
        dz = _af_probe(_sd_columns(hx, n), local, 4, bz)
        # An attained seed prevents an unresolved probe cutoff from
        # masquerading as an attained logical upper bound.
        reading = min(dx, dz, xword.bit_count(), zword.bit_count())
        if reading < 6:
            continue

        def freeze(C):
            return tuple(tuple(tuple(cell) for cell in row) for row in C)

        floors = (a[0][0], b[0][0])
        quality = (k*reading*reading/n, min(floors),
                   a[0][1]+b[0][1], int(a[4] != b[4]))
        results.append((quality, freeze(A), freeze(B), k,
                        reading, (a[4], b[4]), floors))
        if len(results) >= max(count+8, 2*count):
            break
    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish a screened Dic5 prescription, never an orbit-copy draw."""
    catalogue = _di_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct attained-witness Dic5 portfolio exhausted")
    quality, A, B, k, reading, cards, floors = catalogue[idx]
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "quality": quality, "design_cards": cards,
            "exact_constituent_distances": floors,
            "logical_witness_upper_bound": reading,
            "logical_floor_certificate": 5,
            "constituent_eight_preferred": min(floors) >= 8}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without changing the immutable selection."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Different cyclic gap/triangle prescriptions guarantee inequivalent codes: the parent produced 60 slots but only 22 distinct Tanner graphs.
# REFUTED: C5xC7 quotient-order/orbit cards supply five inequivalent high-rate finalists: sixty slots collapsed to four distinct codes, all measured at distance three, leaving one zero contribution.
# REFUTED: Noncommuting affine matching cores with the parent's isolated trinomial spectrum bank supply five readings above six: all five measured current finalists had k=180,d30s=6, with weight-six X minima confined to sector 1.
# REFUTED: Split cubic attachments and anchored-seven preference supply five measured affine distance-eight leaders: the current finalists instead measured k=180,d30s=7, with extensive isomorphic collapse and sector-1 minima.
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
