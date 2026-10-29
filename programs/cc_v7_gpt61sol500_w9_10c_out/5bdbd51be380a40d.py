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
CONCEPT = ("C3 semidirect_-1 C16, order 48, A=(1,2), B=(2,5): independently optimized long-order reinforced constituents and complete mixed-sector witness screening should supply five novel measured distance readings >=10 in frontier cell 0.250; this is a falsifiable measurement hypothesis.")
GROUP = ("Keep the split-metacyclic construction rule, resizing its parameters because factor twelve puts the parent's q=55 outside the size window. Use m=3,r=16,s=2, q=48 and n=576; the centre has order eight and there are twenty-four conjugacy classes. The related order-48 centre-eight/24-class bed reached fitness 20.9 on these shapes. That reading motivates this presentation, not a distance guarantee.")
SHAPES = ("Retarget the 0.250 frontier cell rather than resubmit the parent's distance-five binomial mechanism in 0.400. A=(1,2), B=(2,5) give constituent lengths/check counts 96/48 and 240/96, n=576, kb=144 and guaranteed rate 3/12=0.250. The size window permits q<=49. Parent minima occupied sector-1 Acol/Bcol blocks broadly; the earlier weight-four failure also entered Arow0/Brow0. Actual full ranks enforce [0.225,0.275).")
FILLING = ("Twenty support/quotient cards independently optimize A and B. Five B supports include two cubic columns, fewer matching columns with a long-order isolated binomial, and one degree-four alternative paired with A counts (2,3). Other pairings use A=(3,3). Binomial quotients have orders sixteen or twenty-four. Both CSS maxima are <=9. Verified constituent lifts supply exact kernel-four gates, normal-C3 witnesses and equal-budget information-set refinement. Full ranks, complete kernel-four exclusion, intrinsic Tanner invariants and stabilizer-aware mixed-sector witness searches select immutable outputs. Larger witness readings remain upper bounds; the evaluator decides archive novelty.")
FAMILY_NAME = "r025_metacyclic_m3r16s2_12x25_longorder_independent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C3 semidirect C16 requires order 48")
    # Same split-metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    m, r, s = 3, 16, 2
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi] * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _kernel_profile(rows, n):
    """Exact low-weight kernel screen, not a distance-search estimate.

    A reading of five means only that weights one through four were
    excluded. Equal syndromes of words of weight <=2 detect all such
    kernel words. Collision counts are only a secondary selection feature.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= check
            row ^= low
    seen = {0: 0}
    lightest, collisions = 5, 0

    def inspect(syndrome, word):
        nonlocal lightest, collisions
        if syndrome in seen:
            difference = word ^ seen[syndrome]
            if difference:
                lightest = min(lightest, difference.bit_count())
                collisions += 1
        else:
            seen[syndrome] = word

    for j, syndrome in enumerate(columns):
        inspect(syndrome, 1 << j)
    for j, syndrome in enumerate(columns):
        for i in range(j):
            inspect(syndrome ^ columns[i], (1 << j) | (1 << i))
    return lightest, collisions


def _component_rows(G, side, grid):
    """Expose exactly the scaffold's component action for any shape.

    The zero auxiliary cell is never published. Sector 1 contains the
    complete constituent, while every auxiliary-sector term is zero.
    """
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    assert len(rows) == len(grid) * G.q
    assert all(row.bit_length() <= width for row in rows)
    return [row & mask for row in rows]


def _generator_key(G, generators):
    """Quotient known component orbit copies, not arbitrary graph isos.

    Affine-group automorphisms have x->x^u, y->x^v*y. Binomial
    generators can also be inverted and their columns permuted.
    """
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    keys = []
    for u in range(1, 11):
        for v in range(11):
            images = []
            for a, b in generators:
                g = ((u * a + v * sums[b]) % 11, b)
                images.append(min(G.idx(g), G.idx(G.inv(g))))
            keys.append(tuple(sorted(images)))
    return min(keys)


def _component_bank(G):
    """Deterministically optimize different quotient prescriptions."""
    local = random.Random(BASE_SEED ^ 0x550403)
    # Quotient-coordinate profiles distinguish construction prescriptions.
    # Avoid two rotations, which commute and immediately create a 4-word.
    profiles = ((0, 1, 2), (0, 1, 1), (1, 1, 2),
                (1, 2, 2), (0, 2, 2), (1, 2, 3))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    entries, used = [], set()
    for attempt in range(256):
        profile = profiles[attempt % len(profiles)]
        generators = [local.choice(pools[b]) for b in profile]
        undirected = [min(G.idx(g), G.idx(G.inv(g)))
                      for g in generators]
        if len(set(undirected)) != 3:
            continue
        key = _generator_key(G, generators)
        if key in used:
            continue
        used.add(key)
        A = [[[G.e, g] for g in generators]]
        reading, collisions = _kernel_profile(
            _component_rows(G, "A", A), 3 * G.q)
        entries.append(((reading, -collisions), key, A))
    if not entries:
        raise ValueError("component bank unexpectedly empty")
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Prefer every available clean component. If the hypothesis fails,
    # retain the best legal components instead of invalidating all slots.
    clean = [item for item in entries if item[0][0] == 5]
    return (clean if clean else entries[:12])[:16]


def _draw_B(rng, G, attempt, bank):
    """Indexed support/order prescriptions, subsequently screened."""
    if attempt % 4 == 0:
        source = bank[(attempt // 4) % len(bank)][2]
        # Independent samples of quotients, not copies of source labels.
        profile = [cell[1][1] for cell in source[0]]
        cells = []
        for b in profile:
            pool = [g for g in G.elements if g != G.e and g[1] == b]
            cells.append([G.e, rng.choice(pool)])
        return [cells]

    # One degree-three column and two degree-two columns: total seven.
    # Vary the trinomial's quotient-coordinate pattern deliberately.
    patterns = ((0, 1), (0, 2), (1, 1), (1, 2),
                (1, 3), (2, 2), (2, 3), (2, 4))
    b0, b1 = patterns[(attempt // 4) % len(patterns)]
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    g = rng.choice(pools[b0])
    h = rng.choice(pools[b1])
    while h == g:
        h = rng.choice(pools[b1])
    cells = [[G.e, g, h]]
    for j in range(2):
        b = (attempt // (8 + 3 * j) + j) % 5
        cells.append([G.e, rng.choice(pools[b])])
    return [cells]


def _tanner_signature(hx, hz):
    """An uncoloured Tanner-isomorphism invariant, not an isomorphism test.

    Check intersections define two adjacency relations. Each check's
    profile counts common neighbours in both relations for every other
    check. Different signatures certify non-isomorphic Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                adj = one if c == 1 else two
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            kind = ((one[i] >> j) & 1) + 2 * ((two[i] >> j) & 1)
            key = (kind,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & two[j]).bit_count()
                   + (two[i] & one[j]).bit_count(),
                   (two[i] & two[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(tuple(sorted(p.items())) for p in profiles))


def _legacy_c11_cross_cayley_reference(rng, G, idx):
    """Inactive historical reference; never used by the new portfolio."""
    from itertools import combinations, product

    def clean4(rows, width):
        """Exactly exclude all nonzero kernel supports of weights 1..4."""
        cols = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= tag
                row ^= bit
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return False
        pairs = set()
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        # Distinct nonzero columns make a pair-pair collision a
        # four-distinct-qubit support. Pair-single collisions detect three.
        return True

    # A binomial is an undirected Cayley edge. Enumerate ALL triples,
    # rather than sampling a small bank and correlating its two sides.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if G._key(g) != G._key(G.e)
    })
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    automorphisms = []
    for u in range(1, 11):
        for v in range(11):
            image = {}
            for label in reps:
                a, b = G.elements[label]
                g = ((u * a + v * sums[b]) % 11, b)
                image[label] = min(G.idx(g), G.idx(G.inv(g)))
            automorphisms.append(image)

    keys = set()
    for triple in combinations(reps, 3):
        generators = [G.elements[t] for t in triple]
        # Commuting distinct edge generators immediately make a square.
        if any(G.mul(generators[i], generators[j])
               == G.mul(generators[j], generators[i])
               for i in range(3) for j in range(i)):
            continue
        key = min(tuple(sorted(image[t] for t in triple))
                  for image in automorphisms)
        keys.add(key)

    bank = []
    for key in sorted(keys):
        grid = [[[G.e, G.elements[t]] for t in key]]
        rows = _component_rows(G, "A", grid)
        if gf2_rank(rows) != G.q - 1:
            continue
        if not clean4(rows, 3 * G.q):
            continue
        quotient = tuple(sorted(
            min(G.elements[t][1], (-G.elements[t][1]) % 5)
            for t in key))
        bank.append((quotient, key, grid))
    if not bank:
        raise ValueError("no connected kernel-four-clean constituent")

    graph_keys, filling_keys = set(), set()
    accepted = 0

    def consider(A, B, prescription):
        nonlocal accepted
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            return None
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > min(9, W_MAX)
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 375 * n <= 1000 * k < 425 * n):
            return None
        # Constituent cleanliness does not exclude mixed-sector words.
        # Both complete centralizers must pass, including sector 2.
        if not clean4(hx, n) or not clean4(hz, n):
            return None
        signature = _tanner_signature(hx, hz)
        if signature in graph_keys:
            return None
        graph_keys.add(signature)

        if accepted == idx:
            assert n == 550 and css_ok(hx, hz)
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "centralizer_screens": (5, 5),
                    "prescription": prescription}
        accepted += 1
        return None

    # Inner scalings are powers of 3. The multiplier 2 represents the
    # other outer-automorphism coset. Its relative orientation is tested,
    # not assumed to produce a different code.
    plans = []
    for a in range(len(bank)):
        for b in range(a, len(bank)):
            for twist in (1, 2):
                plans.append((bank[a][0] == bank[b][0], a, b, twist))
    plans.sort()
    for _, a, b, twist in plans:
        A = bank[a][2]
        B = [[[
            G.elements[G.idx(((twist * g[0]) % 11, g[1]))]
            for g in cell
        ] for cell in bank[b][2][0]]]
        result = consider(A, B, ("cross-binomial", a, b, twist))
        if result is not None:
            return result

    # Additional structural cards use one noncommuting trinomial and
    # two binomials. Total counts are (6,7), so both weight maxima are 9.
    # Cards prescribe quotient coordinates; intrinsic invariants and
    # exact screens, not random label fingerprints, select the catalogue.
    cards = sorted(
        (b0, b1, c0, c1)
        for b0, b1, c0, c1 in product(range(5), repeat=4)
        if b0 <= b1 and c0 <= c1
        and (b0 or b1) and (c0 or c1))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55C405)
    for attempt in range(4096):
        card = attempt % len(cards)
        b0, b1, c0, c1 = cards[card]
        g, h = local.choice(pools[b0]), local.choice(pools[b1])
        if g == h or G.mul(g, h) == G.mul(h, g):
            continue
        s, t = local.choice(pools[c0]), local.choice(pools[c1])
        if G.mul(s, t) == G.mul(t, s):
            continue
        B = [[[G.e, g, h], [G.e, s], [G.e, t]]]
        if not clean4(_component_rows(G, "B", B), 3 * G.q):
            continue
        a = (attempt // len(cards) + card) % len(bank)
        result = consider(bank[a][2], B, ("trinomial-card", a, card))
        if result is not None:
            return result
    raise ValueError("strict full-kernel-four-clean catalogue exhausted")


def _sg_columns(rows, width):
    """Syndromes of rows supplied by the verified scaffold."""
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _sg_clean4(rows, width):
    """Return columns iff every nonzero kernel support through four is absent."""
    cols = _sg_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = set()
    for j, first in enumerate(cols):
        for i in range(j):
            syndrome = first ^ cols[i]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs.add(syndrome)
    return cols


def _sg_span(rows):
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


def _sg_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        previous = pivots.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _sg_probe(local, cols, passes, seed=0, own=None, stabilizers=()):
    """Search actual kernel/logical witnesses; a reading is an upper bound.

    Supports remain in the supplied matrix's original coordinates.
    Carrying the previous witness makes refinement nonincreasing.
    """
    best = seed
    weight = seed.bit_count() if seed else len(cols) + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if (word and w < weight
                and (own is None or _sg_outside(word, own))):
            best, weight = word, w

    order = list(range(len(cols)))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        local.shuffle(kernel)
        if own is None:
            # Elementary exchanges preserve the constituent kernel span.
            for i in range(len(kernel)):
                word = kernel[i]
                w = word.bit_count()
                for j, other in enumerate(kernel):
                    if i != j:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if 0 < tw < w:
                            word, w = trial, tw
                kernel[i] = word
                inspect(word)

        short = sorted(kernel, key=int.bit_count)
        if own is not None:
            short = short[:80]
        for i, first in enumerate(short):
            for second in short[:i]:
                inspect(first ^ second)
        tiny = short[:24]
        for i, first in enumerate(tiny):
            for j in range(i):
                pair = first ^ tiny[j]
                for h in range(j):
                    inspect(pair ^ tiny[h])

        if best:
            starting = best
            moves = short[:64]
            for i, first in enumerate(moves):
                inspect(starting ^ first)
                for second in moves[:i]:
                    inspect(starting ^ first ^ second)

            if stabilizers:
                starting = best
                relevant = [row for row in stabilizers
                            if (row & starting).bit_count() >= 2]
                for i, first in enumerate(relevant):
                    inspect(starting ^ first)
                    for second in relevant[:i]:
                        inspect(starting ^ first ^ second)
        if weight <= 5:
            break

    if best:
        syndrome = 0
        bits = best
        while bits:
            bit = bits & -bits
            syndrome ^= cols[bit.bit_length() - 1]
            bits ^= bit
        assert syndrome == 0
        assert own is None or _sg_outside(best, own)
    return best


def _sg_normal_seed(G, cols, blocks, local):
    """Probe normal-C3 coset words using actual constituent syndromes.

    No transitivity assumption is made about the complete quantum code.
    """
    words, syndromes = [], []
    for j in range(blocks):
        for b in range(16):
            word = syndrome = 0
            for a in range(3):
                position = j * G.q + G.idx((a, b))
                word |= 1 << position
                syndrome ^= cols[position]
            words.append(word)
            syndromes.append(syndrome)
    compressed = _sg_probe(local, syndromes, 6)
    expanded = 0
    while compressed:
        bit = compressed & -compressed
        expanded ^= words[bit.bit_length() - 1]
        compressed ^= bit
    return expanded


def _sg_graph_key(hx, hz, width):
    """Conservative intrinsic full-Tanner invariant, not a label key."""
    from hashlib import sha256
    rows = hx + hz
    signature = (
        width, len(rows),
        tuple(sorted(row.bit_count() for row in rows)),
        tuple(sorted(col.bit_count()
                     for col in _sg_columns(rows, width))),
        _tanner_signature(hx, hz),
    )
    return sha256(repr(signature).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _sg_portfolio(G, count):
    """Deterministic immutable portfolio; memoization changes only cost."""
    from itertools import product

    assert G.q == 48
    q, e, cap = G.q, G.e, min(9, W_MAX)
    profiles = (
        ((1, 2, 1, 1, 1), (2, 1, 1, 1, 1)),
        ((3, 0, 1, 1, 1), (0, 3, 1, 1, 1)),
        ((2, 2, 1, 1, 0), (1, 1, 1, 1, 2)),
        ((3, 1, 1, 1, 0), (0, 2, 1, 1, 2)),
        ((2, 1, 1, 1, 1), (2, 1, 1, 1, 1)),
    )
    models = (
        ((0, 1), (2, 3)),
        ((2, 5), (4, 7)),
        ((4, 9), (6, 11)),
        ((6, 13), (2, 15)),
    )

    def element_order(g):
        power = e
        for length in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return length
        raise ValueError("invalid element order")

    orders = {g: element_order(g) for g in G.elements}
    long16 = [g for g in G.elements if orders[g] == 16]
    long24 = [g for g in G.elements if orders[g] == 24]
    assert long16 and long24

    def triple(local, pair):
        b0, b1 = pair
        for _ in range(64):
            a = local.randrange(1, 3) if b0 % 2 == 0 else local.randrange(3)
            c = local.randrange(1, 3) if b1 % 2 == 0 else local.randrange(3)
            g, h = (a, b0), (c, b1)
            if (g != e and h != e and g != h
                    and G.mul(g, h) != G.mul(h, g)):
                return [e, g, h]
        return None

    def draw(local, profile, mode, side):
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
                    pool = (long16 if mode == 0 else long24 if mode == 1
                            else long16 + long24)
                    relative = local.choice(pool)
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    pair = models[mode][(i + j + int(side == "B")) % 2]
                    base = triple(local, pair)
                    if base is None:
                        return None
                    cell = [G.mul(anchor, g) for g in base]
                assert len({G._key(g) for g in cell}) == terms
                row.append(cell)
            grid.append(row)
        return grid

    def cycle_cost(rows):
        cost = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                shared = (row & other).bit_count()
                cost += shared * (shared - 1) // 2
        return cost

    def constituents(side):
        bank, component_keys = [], set()
        for card in range(20):
            mode, kind = card % 4, card // 4
            profile = (((2, 3),) if kind == 4 else ((3, 3),)) \
                if side == "A" else profiles[kind]
            local = random.Random(
                BASE_SEED ^ (0x481225 + 104729 * card
                             + 1000003 * int(side == "B")))
            raw, fillings = [], set()
            width = len(profile[0]) * q
            for trial in range(64 if side == "A" else 96):
                grid = draw(local, profile, mode, side)
                if grid is None:
                    continue
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows = _component_rows(G, side, grid)
                rank = gf2_rank(rows)
                if rank < len(profile) * q - 4:
                    continue
                cols = _sg_clean4(rows, width)
                if cols is None:
                    continue
                raw.append(dict(
                    grid=grid, rows=rows, cols=cols, rank=rank,
                    cost=cycle_cost(rows), card=card, trial=trial))
            raw.sort(key=lambda r: (r["cost"], -r["rank"], r["trial"]))

            shortlist = []
            for record in raw[:12]:
                seed = _sg_normal_seed(
                    G, record["cols"], len(profile[0]), local)
                word = _sg_probe(local, record["cols"], 3, seed)
                if not word:
                    continue
                record.update(witness=word, reading=word.bit_count())
                shortlist.append(record)
            shortlist.sort(key=lambda r: (
                -r["reading"], r["cost"], r["trial"]))

            refined = []
            for record in shortlist[:4]:
                word = _sg_probe(
                    local, record["cols"], 12, record["witness"])
                record.update(witness=word, reading=word.bit_count())
                refined.append(record)
            refined.sort(key=lambda r: (
                -r["reading"], r["cost"], r["trial"]))

            kept = 0
            for record in refined:
                key = _sg_graph_key(record["rows"], [], width)
                if key in component_keys:
                    continue
                component_keys.add(key)
                bank.append(record)
                kept += 1
                if kept == 2:
                    break
        return bank

    aa, bb = constituents("A"), constituents("B")
    if len(aa) < 3 or len(bb) < 5:
        raise ValueError("too few independently refined constituents")

    # The shown q=48 frontier filling is a reference only, never an output.
    known_A = [[[(1, 2), (1, 9), (2, 11)],
                [(2, 7), (1, 9), (0, 15)]]]
    known_B = [
        [[e, (2, 1)], [(0, 5)], [e], [e], [e]],
        [[(2, 2)], [(1, 7), (2, 14)], [e], [(0, 5)], [(1, 12)]],
    ]
    khx, khz, kn = lifted_matrices(G, known_A, known_B)
    graph_keys = {_sg_graph_key(khx, khz, kn)}

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"], a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(aa)
        for bi, b in enumerate(bb))

    def logical_reading(a, b, hx, hz, local):
        """Verified embedded and independently generated mixed-sector words."""
        readings = []
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = _sg_span(stabilizers)
            cols = _sg_columns(checks, 12 * q)
            seed = 0
            for block in range(2 if kind == "X" else 5):
                if kind == "X":
                    word = b["witness"] << (block * 5 * q)
                else:
                    word, source = 0, a["witness"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 5 + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if (_sg_outside(word, own)
                        and (not seed or word.bit_count() < seed.bit_count())):
                    seed = word
            word = _sg_probe(
                local, cols, 3, seed, own=own, stabilizers=stabilizers)
            if not word:
                return None
            assert _sg_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            readings.append(word.bit_count())
        return min(readings)

    results, fillings = [], set()
    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        wx = max(sum(sa[0]) + sum(row[j] for row in sb)
                 for j in range(5))
        wz = max(sum(row) + sa[0][j] for row in sb for j in range(2))
        assert max(wx, wz) <= cap

        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or not (225 * n <= 1000 * k < 275 * n):
            continue
        if _sg_clean4(hx, n) is None or _sg_clean4(hz, n) is None:
            continue
        signature = _sg_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        local = random.Random(BASE_SEED ^ (0x48D025 + 1009 * ai + bi))
        reading = logical_reading(a, b, hx, hz, local)
        if reading is None:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, reading, k,
                   -a["cost"] - b["cost"])
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["reading"], b["reading"]), signature.hex()))
        if len(results) >= max(count, 5) + 16:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure publication of an immutable, independently optimized design."""
    catalogue = _sg_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("long-order target-rate portfolio exhausted")
    quality, A, B, k, reading, cards, components, signature = catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k, "quality": quality,
        "design_cards": cards, "component_witness_readings": components,
        "quantum_witness_upper_bound": reading,
        "complete_kernel4_excluded": True,
        "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without modifying the portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent screens and the two-second proxy suffice to protect the Frobenius-bed finalists from weight-four logicals; slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: Cross-Cayley pairing supplies five inequivalent k=226 finalists: the measured parent supplied three k=226 finalists and two k=222 finalists, while slot 3 collapsed to slot 0.
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
