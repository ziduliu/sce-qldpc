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
CONCEPT = ("C11 semidirect C4 with inversion action, order 44, A=B=(2,3): independently optimized saturated constituents, including exclusive four-term cells, should supply five fresh cell-0.100 finalists with measured fitness above 46.5234; this is a falsifiable measurement hypothesis.")
GROUP = ("Retain the parent's cyclic-by-cyclic semidirect construction rule. Factor thirteen requires reducing q from 117; parameters m=11,r=4,s=10 give q=44 and n=572. This group is isomorphic to Dic11: its normal C11 corresponds to the square of the dicyclic rotation, and its C4 generator to the dicyclic cross generator. The shown equivalent bed measured five distance-24 finalists. Neither distance survival nor novelty against its archive is assumed.")
SHAPES = ("Change both sides to (2,3), giving equal 132-variable, 88-check constituents, n=572, kb=44 and guaranteed rate 1/13. The size window permits q<=46; q=44 fits. Complete verified ranks require k>0 and 8*k<n, placing every published code below 0.125. Parent weight-18 families occupied sector-1 Acol/Bcol blocks; weight-15 minima also touched Arow0/Brow0. Two-row constituents replace that sector-1 bottleneck, while complete logical probes retain mixed-sector coverage.")
FILLING = ("Twenty cards combine four saturated supports with concentrated or pervasive mixed-quotient labels. Each component row totals five and every column totals at most four, so BOTH CSS maxima are nine, respecting the stricter cap. The new support ((0,1,4),(2,3,0)) introduces exclusive tetranomials. Verified constituent matrices exclude kernels through four, exactly enumerate normal-C11 invariant kernel witnesses, and receive equal 6+26-pass refinement. Complete lifts enforce actual rate, CSS and kernel-four exclusion; explicit embedded and mixed-sector logical witnesses receive stabilizer-aware probes. A globally reconstructed immutable portfolio conservatively rejects equal uncoloured Tanner invariants, replacing ownership buckets. Larger readings remain upper bounds; unseen archive novelty remains evaluator-tested.")
FAMILY_NAME = "r010_metacyclic11by4_inv_q44_23x23_saturated_tetra_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Same metacyclic rule; the new size factor requires q <= 46.
    # x^m=y^r=1, y*x*y^-1=x^s, with s^r=1 modulo m.
    m, r, s = 11, 4, 10
    if not valid_q(q) or q != m*r:
        raise ValueError("C11 semidirect C4 requires order 44")
    assert pow(s, r, m) == 1
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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
    """Extract the component from the verified construction.

    A one-cell dummy side reduces screening cost without changing the
    left/right action used for the component being extracted.
    """
    dummy = [[[G.e]]]
    mask = (1 << (3 * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx[:G.q]]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz[:G.q]]


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


def _legacy_order55_certificate_reference(rng, G, idx):
    """Inactive order-55 reference; the new NEW_CTX never calls it."""
    from collections import Counter
    from hashlib import sha256

    q = G.q
    assert q == 55
    width = 3 * q
    eid = G.idx(G.e)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def clean_four(rows, n):
        """Exact absence of nonzero kernel words of weights one to four."""
        cols = columns(rows, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return False
        pairs = set()
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        return True

    def five_count(rows):
        """Exact constituent weight-five count, after excluding <=4.

        Unique pair syndromes make every anchor/pair/pair collision
        a five-distinct-coordinate kernel support. A support containing
        an anchor is counted six times. Regular group translations act
        transitively inside each of the three column blocks, so summing
        anchor incidences and multiplying by q counts every support five
        times. Only syndromes from the verified lift are used.
        """
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return None
        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs[syndrome] = (i, j)

        incidences = 0
        for block in range(3):
            anchor = block * q + eid
            hits = 0
            for syndrome, pair in pairs.items():
                if anchor in pair:
                    continue
                other = pairs.get(syndrome ^ cols[anchor])
                if other is not None:
                    assert anchor not in other
                    assert not set(pair).intersection(other)
                    hits += 1
            assert hits % 6 == 0
            incidences += hits // 6
        assert (q * incidences) % 5 == 0
        return q * incidences // 5

    def basic_key(hx, hz):
        signature = _tanner_signature(hx, hz)
        return sha256(repr(signature).encode("ascii")).digest()

    def graph_key(hx, hz, n, basic):
        """Uncoloured full-Tanner invariants, never group-label keys."""
        rows = hx + hz
        degrees = [row.bit_count() for row in rows]
        check_profiles = []
        for i, row in enumerate(rows):
            profile = Counter(
                (degrees[j], (row & other).bit_count())
                for j, other in enumerate(rows))
            check_profiles.append(
                (degrees[i], tuple(sorted(profile.items()))))

        qubit_profiles = Counter()
        for syndrome in columns(rows, n):
            neighbours = []
            bits = syndrome
            while bits:
                bit = bits & -bits
                neighbours.append(degrees[bit.bit_length() - 1])
                bits ^= bit
            qubit_profiles[tuple(sorted(neighbours))] += 1
        signature = (
            basic, tuple(sorted(check_profiles)),
            tuple(sorted(qubit_profiles.items())))
        return sha256(repr(signature).encode("ascii")).digest()

    # Reconstruct the parent's deterministic catalogue, including the
    # sixty-slot evaluation shown in the prompt. Equal invariants are
    # conservatively rejected; this is not access to the entire archive.
    legacy_bank = _component_bank(G)
    archived = set()
    legacy_fillings = set()
    old_rng = random.Random(BASE_SEED ^ 0x550409)
    for attempt in range(4096):
        a_score, _, A = legacy_bank[attempt % len(legacy_bank)]
        B = _draw_B(old_rng, G, attempt, legacy_bank)
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in legacy_fillings:
            continue
        legacy_fillings.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > min(9, W_MAX)
                or not _no_dead_lines(sa, sb)):
            continue
        b_score = _kernel_profile(_component_rows(G, "B", B), width)
        required = min(a_score[0], 5 if attempt < 256 else 4)
        if b_score[0] < required:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (375 * n <= 1000 * k < 425 * n):
            continue
        archived.add(basic_key(hx, hz))
        if len(archived) >= max(60, N_SLOTS):
            break

    bank = []
    for _, key, A in legacy_bank:
        rows = _component_rows(G, "A", A)
        if gf2_rank(rows) != q - 1:
            continue
        five = five_count(rows)
        if five is not None:
            bank.append((five, key, A))
    bank.sort(key=lambda item: (item[0], item[1]))
    if not bank:
        raise ValueError("no connected kernel-four-clean A constituent")

    # Ten trinomial quotient cards times two reinforcement classes.
    # Later epochs change the reinforcement exponent, not just a gauge.
    triangles = (
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 1),
        (1, 2), (1, 3), (1, 4), (2, 2), (2, 3))
    pools = [[g for g in G.elements if g != G.e and g[1] == b]
             for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55F15D)
    graph_keys, filling_keys = set(), set()

    for position in range(idx + 1):
        card, epoch = position % 20, position // 20
        b0, b1 = triangles[card % 10]
        reinforcement = 1 + (card // 10 + 2 * epoch) % 4
        winner = None

        for batch in range(12):
            proposals = []
            for trial in range(64):
                g, h = local.choice(pools[b0]), local.choice(pools[b1])
                if g == h:
                    continue
                rotation = local.choice(pools[0])
                cyclic = local.choice(pools[reinforcement])
                B = [[[G.e, g, h],
                      [G.e, rotation], [G.e, cyclic]]]
                ai = (position + trial + batch) % len(bank)
                afive, _, A = bank[ai]
                filling = tuple(
                    tuple(sorted(G.idx(x) for x in cell))
                    for grid in (A, B) for row in grid for cell in row)
                if filling in filling_keys:
                    continue
                filling_keys.add(filling)

                rows = _component_rows(G, "B", B)
                if gf2_rank(rows) != q:
                    continue
                bfive = five_count(rows)
                if bfive is None:
                    continue
                pressure = 1 + card % 3
                burden = pressure * afive + (4 - pressure) * bfive
                proposals.append(
                    ((burden, max(afive, bfive), afive + bfive, filling),
                     A, B, afive, bfive, cyclic))
                if len(proposals) >= 8:
                    break

            proposals.sort(key=lambda item: item[0])
            for quality, A, B, afive, bfive, cyclic in proposals:
                sa = [[len(cell) for cell in row] for row in A]
                sb = [[len(cell) for cell in row] for row in B]
                assert sa == [[2, 2, 2]] and sb == [[3, 2, 2]]
                assert _no_dead_lines(sa, sb)
                assert _grid_weights(sa, sb) == 9
                if _grid_weights(sa, sb) > min(9, W_MAX):
                    raise ValueError("distance-five cards require cap nine")

                hx, hz, n = lifted_matrices(G, A, B)
                rx, rz = gf2_rank(hx), gf2_rank(hz)
                k = n - rx - rz
                if n != 550 or k != 222:
                    continue
                basic = basic_key(hx, hz)
                if basic in archived:
                    continue
                signature = graph_key(hx, hz, n, basic)
                if signature in graph_keys:
                    continue

                # Full kernels include every mixed-sector support.
                # This is stronger than merely rejecting short logicals.
                if not clean_four(hx, n) or not clean_four(hz, n):
                    continue

                # Bcol2 is the order-five binomial. Its cyclic orbit
                # gives a candidate X logical in each A-column block.
                power = G.e
                orbit = []
                for _ in range(5):
                    orbit.append(G.idx(power))
                    power = G.mul(power, cyclic)
                assert power == G.e and len(set(orbit)) == 5
                witness = None
                for acol in range(3):
                    base = (3 * acol + 2) * q
                    word = sum(1 << (base + t) for t in orbit)
                    if any((row & word).bit_count() & 1 for row in hz):
                        continue
                    if gf2_rank(hx + [word]) == rx + 1:
                        witness = word
                        break
                if witness is None:
                    continue

                assert css_ok(hx, hz)
                assert witness.bit_count() == 5
                assert max(row.bit_count() for row in hx + hz) == 9
                graph_keys.add(signature)
                winner = {
                    "A": A, "B": B, "screen_k": k,
                    "distance_certificate": 5,
                    "X_witness": witness,
                    "constituent_weight_five_counts": (afive, bfive),
                    "card": (card, epoch, reinforcement),
                    "selection_quality": quality[:3],
                    "tanner_signature": signature.hex(),
                }
                break
            if winner is not None:
                break

        if winner is None:
            raise ValueError("fresh certified-distance-five card exhausted")
        if position == idx:
            assert lifted_k(G, winner["A"], winner["B"]) == 222
            return winner

    raise ValueError("unreachable catalogue position")


def _lp_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    return columns


def _lp_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in basis:
                basis[p] = word
                break
            word ^= basis[p]
    return basis


def _lp_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        if p not in basis:
            return True
        word ^= basis[p]
    return False


def _lp_clean_four(rows, width):
    """Exactly exclude all nonzero kernel supports of weights 1..4."""
    columns = _lp_columns(rows, width)
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _lp_component(G, grid, side):
    """Extract only from the scaffold's verified left/right construction."""
    unit = [[[G.e]]]
    width = 2 * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    rows = [row & mask for row in rows[:G.q]]
    assert len(rows) == G.q
    assert all(c.bit_count() == 3
               for c in _lp_columns(rows, width))
    return rows


def _lp_component_cycles(rows, width):
    """Count Tanner four-cycles and, when absent, Tanner six-cycles."""
    count = len(rows)
    adjacency = [0] * count
    fours = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            fours += overlap * (overlap - 1) // 2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    if fours:
        return fours, width * width

    directed = 0
    for i, neighbours in enumerate(adjacency):
        bits = neighbours
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            directed += (neighbours & adjacency[j]).bit_count()
            bits ^= bit
    # Each degree-three variable induces one check-projection triangle.
    # Without four-cycles, every remaining triangle is a Tanner six-cycle.
    sixes = directed // 6 - width
    assert directed % 6 == 0 and sixes >= 0
    return 0, sixes


def _lp_probe(rows, width, rng, passes, own=None, seed=0,
              stabilizers=None):
    """Return an explicit kernel/logical witness, never a lower bound.

    Randomized column elimination constructs original-coordinate kernel
    vectors. When own is supplied, only vectors outside its row span
    are scored. A supplied witness is retained unless a lighter one is
    actually found.
    """
    columns = _lp_columns(rows, width)
    best_word = seed
    best = seed.bit_count() if seed else width + 1
    if seed:
        assert all((row & seed).bit_count() % 2 == 0 for row in rows)
        assert own is None or _lp_outside(seed, own)

    def inspect(word):
        nonlocal best, best_word
        weight = word.bit_count()
        if not word or weight >= best:
            return
        if own is not None and not _lp_outside(word, own):
            return
        best, best_word = weight, word

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, nulls = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
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

        short = sorted(nulls, key=int.bit_count)[:32]
        reduced = []
        for original in nulls:
            word = original
            weight = word.bit_count()
            for other in short:
                candidate = word ^ other
                new_weight = candidate.bit_count()
                if 0 < new_weight < weight:
                    word, weight = candidate, new_weight
            reduced.append(word)
            inspect(word)

        short = sorted(reduced, key=int.bit_count)[:32]
        for i, first in enumerate(short):
            for second in short[:i]:
                inspect(first ^ second)
        tiny = short[:16]
        for i, first in enumerate(tiny):
            for j in range(i):
                pair = first ^ tiny[j]
                for h in range(j):
                    inspect(pair ^ tiny[h])

        if best_word and stabilizers:
            # Actual check rows, not a separately implemented lift.
            for _ in range(2):
                starting = best_word
                for row in stabilizers:
                    inspect(starting ^ row)
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (best_word & row).bit_count(),
                    reverse=True)[:24]
                starting = best_word
                for i, first in enumerate(nearby):
                    for second in nearby[:i]:
                        inspect(starting ^ first ^ second)
                if best_word == starting:
                    break
        if best <= (8 if own is None else 6):
            break

    if best_word:
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        assert own is None or _lp_outside(best_word, own)
    return best, best_word


def _lp_owned_bucket(hx, hz, width):
    """Intrinsic, uncoloured full-Tanner ownership.

    Isomorphic Tanner graphs necessarily have the same signature and
    bucket. Hash collisions conservatively share a bucket; they cannot
    make one graph eligible for two different slots. CSS types and
    protograph/group labels never enter the signature.
    """
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    incidence = []
    shared = Counter()
    projected = [0] * len(rows)
    for column in _lp_columns(rows, width):
        checks = []
        while column:
            bit = column & -column
            checks.append(bit.bit_length() - 1)
            column ^= bit
        incidence.append(checks)
        for i, a in enumerate(checks):
            for b in checks[:i]:
                shared[(min(a, b), max(a, b))] += 1
        # Variable degree is an intrinsic graph property.
        if len(checks) >= 12:
            mask = sum(1 << r for r in checks)
            for r in checks:
                projected[r] |= mask ^ (1 << r)

    adjacency = [{} for _ in rows]
    for (a, b), multiplicity in shared.items():
        adjacency[a][b] = multiplicity
        adjacency[b][a] = multiplicity
    masks = [sum(1 << j for j in neighbours)
             for neighbours in adjacency]
    degrees = [row.bit_count() for row in rows]
    features = []
    for i, neighbours in enumerate(adjacency):
        profile = Counter(
            (degrees[j], multiplicity,
             (masks[i] & masks[j]).bit_count(),
             (projected[i] & projected[j]).bit_count())
            for j, multiplicity in neighbours.items())
        walk4 = sum((masks[i] & other).bit_count() ** 2
                    for other in masks)
        features.append((
            degrees[i], masks[i].bit_count(),
            projected[i].bit_count(),
            tuple(sorted(profile.items())), walk4))

    palette = {feature: color for color, feature
               in enumerate(sorted(set(features)))}
    colors = [palette[feature] for feature in features]
    variable_features = [
        tuple(sorted(colors[r] for r in checks))
        for checks in incidence]
    qpalette = {feature: color for color, feature
                in enumerate(sorted(set(variable_features)))}
    check_profiles = [[] for _ in rows]
    for checks, feature in zip(incidence, variable_features):
        for r in checks:
            check_profiles[r].append(qpalette[feature])
    refined = [
        (colors[r], tuple(sorted(profile)))
        for r, profile in enumerate(check_profiles)]
    signature = (
        width, len(rows),
        tuple(sorted(Counter(len(c) for c in incidence).items())),
        tuple(sorted(Counter(features).items())),
        tuple(sorted(Counter(variable_features).items())),
        tuple(sorted(Counter(refined).items())))
    digest = sha256(repr(signature).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % N_SLOTS


def _inactive_order117_owned_reference(rng, G, idx):
    """Historical order-117 publisher; the new NEW_CTX never calls it."""
    assert G.q == 117 and 0 <= idx < N_SLOTS
    q, width = G.q, 2 * G.q
    cap = min(9, 12, W_MAX)
    support = [[3, 3]]

    # Recheck BOTH LP sums explicitly; do not use the card's looser cap.
    wx = max(sum(support[0]) + support[0][j] for j in range(2))
    wz = max(sum(support[0]) + support[0][j] for j in range(2))
    assert wx == wz == 9 and max(wx, wz) <= cap
    assert _no_dead_lines(support, support)

    # Each pair prescribes the two nonidentity y-coordinates in a cell.
    # The first twenty searches cover four A models by five B models.
    # Central-coordinate models are genuine alternatives, not gauge
    # transforms of an already completed filling.
    models = (
        ((0, 1), (0, 2)),
        ((0, 4), (0, 5)),
        ((1, 2), (1, 4)),
        ((1, 3), (2, 6)),
        ((2, 4), (2, 5)),
        ((1, 5), (2, 7)),
        ((4, 7), (5, 8)),
        ((1, 6), (2, 3)),
    )
    card, epoch = idx % 20, idx // 20
    amodel = (card // 5 + 4 * epoch) % len(models)
    bmodel = (card % 5 + 3 * epoch) % len(models)
    local = random.Random(BASE_SEED ^ 0x117C020 ^ (idx * 1000003))

    def draw_grid(model):
        row = []
        for b0, b1 in models[model]:
            for _ in range(64):
                a = local.randrange(1, 13) if b0 == 0 else local.randrange(13)
                c = local.randrange(1, 13) if b1 == 0 else local.randrange(13)
                g, h = (a, b0), (c, b1)
                if g == G.e or h == G.e or g == h:
                    continue
                if G.mul(g, h) == G.mul(h, g):
                    continue
                row.append([G.e, g, h])
                break
            else:
                return None
        return [row]

    def constituent_pool(side, model):
        raw, seen = [], set()
        for trial in range(128):
            grid = draw_grid(model)
            if grid is None:
                continue
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for cell in grid[0])
            if key in seen:
                continue
            seen.add(key)
            rows = _lp_component(G, grid, side)
            if not _lp_clean_four(rows, width):
                continue
            four, six = _lp_component_cycles(rows, width)
            raw.append((four, six, trial, grid, rows))
            if len(raw) >= 30:
                break
        raw.sort(key=lambda item: item[:3])

        pool = []
        for four, six, trial, grid, rows in raw[:14]:
            reading, witness = _lp_probe(rows, width, local, 6)
            if not witness:
                continue
            pool.append({
                "grid": grid, "rows": rows,
                "reading": reading, "witness": witness,
                "cycles": (four, six), "trial": trial,
            })
        pool.sort(key=lambda record: (
            -record["reading"], record["cycles"], record["trial"]))

        # Every shortlisted constituent receives the same extra budget.
        for record in pool[:8]:
            record["reading"], record["witness"] = _lp_probe(
                record["rows"], width, local, 18,
                seed=record["witness"])
        pool = pool[:8]
        pool.sort(key=lambda record: (
            -record["reading"], record["cycles"], record["trial"]))
        return pool[:6]

    def embedded_seed(record, kind, checks, own):
        """Verify nontrivial embedded constituent words in the full code."""
        best_word = 0
        for block in range(2):
            source = record["witness"]
            if kind == "X":
                word = source << (block * 2 * q)
            else:
                word = 0
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 2 + block) * q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if _lp_outside(word, own):
                if not best_word or word.bit_count() < best_word.bit_count():
                    best_word = word
        return best_word

    def finish(winner):
        hx, hz, n = lifted_matrices(G, winner["A"], winner["B"])
        assert n == 585 and css_ok(hx, hz)
        assert lifted_k(G, winner["A"], winner["B"]) == winner["screen_k"]
        assert max(row.bit_count() for row in hx + hz) <= cap
        return winner

    winner, survivors = None, 0
    fillings = set()
    for batch in range(6):
        apool = constituent_pool("A", amodel)
        bpool = constituent_pool("B", bmodel)
        plans = []
        for ai, a in enumerate(apool):
            for bi, b in enumerate(bpool):
                quality = (
                    min(a["reading"], b["reading"]),
                    a["reading"] + b["reading"],
                    -a["cycles"][0] - b["cycles"][0],
                    -a["cycles"][1] - b["cycles"][1])
                plans.append((quality, ai, bi))
        plans.sort(reverse=True)

        for preliminary, ai, bi in plans:
            a, b = apool[ai], bpool[bi]
            A, B = a["grid"], b["grid"]
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            assert all(len(set(cell)) == 3
                       for grid in (A, B) for row in grid for cell in row)

            hx, hz, n = lifted_matrices(G, A, B)
            if _lp_owned_bucket(hx, hz, n) != idx:
                continue
            bx, bz = _lp_basis(hx), _lp_basis(hz)
            k = n - len(bx) - len(bz)
            assert k == n - gf2_rank(hx) - gf2_rank(hz)
            if n != 585 or not (117 <= k <= 131):
                continue
            if not (_lp_clean_four(hx, n) and _lp_clean_four(hz, n)):
                continue

            sx = embedded_seed(b, "X", hz, bx)
            sz = embedded_seed(a, "Z", hx, bz)
            dx, wx = _lp_probe(
                hz, n, local, 8, own=bx, seed=sx, stabilizers=hx)
            dz, wz = _lp_probe(
                hx, n, local, 8, own=bz, seed=sz, stabilizers=hz)
            available = [(d, word) for d, word in ((dx, wx), (dz, wz))
                         if word]
            if not available:
                continue
            reading = min(d for d, _ in available)
            assert reading >= 5
            quality = (k * reading * reading / float(n),
                       reading, preliminary, k)
            result = {
                "A": A, "B": B, "screen_k": k,
                "card": (amodel, bmodel, epoch),
                "intrinsic_owner": idx,
                "component_witness_bounds": (
                    a["reading"], b["reading"]),
                "logical_witness_upper_bound": reading,
                "verified_logical_words": (wx, wz),
                "exact_kernel_cutoff": 5,
                "selection_quality": quality,
            }
            if winner is None or quality > winner["selection_quality"]:
                winner = result
            survivors += 1
            if survivors >= 3:
                return finish(winner)

        # A larger verified witness is a selection feature, not proof
        # that a lighter logical does not exist.
        if winner is not None and winner["logical_witness_upper_bound"] >= 14:
            return finish(winner)

    if winner is not None:
        return finish(winner)
    raise ValueError("no cubic target-rate survivor in intrinsic owner bucket")


from functools import lru_cache


_LOW_PROFILES = (
    ((1, 2, 2), (2, 1, 2)),
    ((1, 1, 3), (2, 2, 1)),
    ((1, 1, 3), (1, 3, 1)),
    ((0, 1, 4), (2, 3, 0)),
)


def _low_component(G, grid, side):
    """Expose only the exact scaffold constituent action."""
    unit = [[[G.e]]]
    width = 3*G.q
    mask = (1 << width)-1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    rows = [row & mask for row in rows[:2*G.q]]
    assert len(rows) == 2*G.q
    return rows


def _low_normal_seed(G, rows):
    """Exact minimum kernel witness in the normal-C11 invariant subspace.

    This is not an enumeration of the complete constituent kernel.
    Column syndromes and expanded words come from the verified matrix.
    """
    q, width = G.q, 3*G.q
    cols = _lp_columns(rows, width)
    pivots, basis = {}, []
    for block in range(3):
        for b in range(4):
            syndrome = word = 0
            for a in range(11):
                j = block*q+G.idx((a, b))
                syndrome ^= cols[j]
                word |= 1 << j
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)

    best = 0
    word = previous = 0
    assert len(basis) <= 12
    for number in range(1, 1 << len(basis)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous).bit_length()-1
        word ^= basis[changed]
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word
        previous = gray
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _low_cycles(rows, width):
    """General-degree Tanner cycle penalties, not distance bounds."""
    cols = _lp_columns(rows, width)
    adjacency = [0]*len(rows)
    fours = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            fours += overlap*(overlap-1)//2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    if fours:
        return fours, width*width
    directed = 0
    for i, neighbours in enumerate(adjacency):
        bits = neighbours
        while bits:
            bit = bits & -bits
            j = bit.bit_length()-1
            directed += (neighbours & adjacency[j]).bit_count()
            bits ^= bit
    internal = sum(
        c.bit_count()*(c.bit_count()-1)*(c.bit_count()-2)//6
        for c in cols)
    assert directed % 6 == 0
    return 0, max(0, directed//6-internal)


def _low_draw(local, G, profile, mode, trial):
    """Structural quotient policies; anchors alone are not designs."""
    e = G.e
    grid = []
    for i, counts in enumerate(profile):
        row = []
        for j, count in enumerate(counts):
            if not count:
                row.append([])
                continue
            anchor = G.pick(local)
            relative = [e]
            if count == 2:
                mixed = mode or (i+j+trial) % 2
                b = local.choice((1, 3)) if mixed else local.choice((0, 2))
                # Never use the central involution as a binomial ratio.
                a = local.randrange(11) if b & 1 else local.randrange(1, 11)
                relative.append((a, b))
            elif count == 3:
                rotation = (local.randrange(1, 11),
                            0 if mode == 0 else local.choice((0, 2)))
                cross = (local.randrange(11), local.choice((1, 3)))
                if (trial+i+j+mode) % 2:
                    relative += [cross, G.mul(cross, rotation)]
                else:
                    relative += [rotation, cross]
            elif count == 4:
                if mode == 0:
                    rotation = (local.randrange(1, 11), 0)
                    cross = (local.randrange(11), local.choice((1, 3)))
                    relative += [rotation, cross, G.mul(cross, rotation)]
                else:
                    # Four quotient classes, not an anchor image of
                    # a completed two-coset tetranomial.
                    relative += [(local.randrange(11), b) for b in (1, 2, 3)]
            cell = [G.mul(anchor, g) for g in relative]
            assert len(cell) == count and len(set(cell)) == count
            row.append(cell)
        grid.append(row)
    return grid


@lru_cache(maxsize=16)
def _low_bank(G, side, profile_index, mode):
    """Immutable, equally refined finalists for one support/element card."""
    profile = _LOW_PROFILES[profile_index]
    q, width = G.q, 3*G.q
    local = random.Random(
        BASE_SEED ^ (0x44A230 + 104729*profile_index + 7919*mode
                     + 1000003*int(side == "B")))
    raw, fillings = [], set()
    for trial in range(160):
        grid = _low_draw(local, G, profile, mode, trial)
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        rows = _low_component(G, grid, side)
        rank = gf2_rank(rows)
        if rank < 2*q-6:
            continue
        if not _lp_clean_four(rows, width):
            continue
        seed = _low_normal_seed(G, rows)
        four, six = _low_cycles(rows, width)
        orbit_weight = seed.bit_count() if seed else width+1
        # Exact normal-subgroup witnesses of weight 11/22 cannot
        # support the intended distance-24 measurement hypothesis.
        cheap = (int(orbit_weight <= 22),
                 (12+8*mode)*four+six, -orbit_weight, trial)
        raw.append((cheap, grid, rows, seed, rank, (four, six)))
    raw.sort(key=lambda record: record[0])

    shortlisted = []
    for cheap, grid, rows, seed, rank, cycles in raw[:16]:
        reading, word = _lp_probe(
            rows, width, local, 6, seed=seed)
        if not word:
            continue
        shortlisted.append(
            (reading, cycles, cheap[-1], grid, rows, word, rank))
    shortlisted.sort(key=lambda r: (-r[0], r[1], r[2]))

    finalists = []
    for _, cycles, trial, grid, rows, seed, rank in shortlisted[:8]:
        reading, word = _lp_probe(
            rows, width, local, 26, seed=seed)
        if not word or reading < 16:
            continue
        frozen = tuple(tuple(tuple(cell) for cell in row) for row in grid)
        finalists.append((reading, cycles, trial, frozen, word, rank))
    finalists.sort(key=lambda r: (-r[0], r[1], r[2]))
    return tuple(finalists[:6])


def _low_graph_key(hx, hz, width):
    """Conservative uncoloured Tanner invariant, not a complete iso test.

    It includes check weights, degree-conditioned intersections,
    common projected neighbours and qubit neighbourhood colours.
    No element, protograph, sector or CSS-type labels enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    columns = _lp_columns(rows, width)
    incidence = []
    shared = Counter()
    relations = {}
    for column in columns:
        checks = []
        bits = column
        while bits:
            bit = bits & -bits
            checks.append(bit.bit_length()-1)
            bits ^= bit
        incidence.append(checks)
        for i, a in enumerate(checks):
            for b in checks[:i]:
                shared[min(a, b), max(a, b)] += 1
        relation = relations.setdefault(len(checks), [0]*len(rows))
        mask = sum(1 << r for r in checks)
        for r in checks:
            relation[r] |= mask ^ (1 << r)

    neighbours = [{} for _ in rows]
    for (a, b), overlap in shared.items():
        neighbours[a][b] = overlap
        neighbours[b][a] = overlap
    masks = [sum(1 << j for j in ns) for ns in neighbours]
    ordered_relations = tuple(sorted(relations.items()))
    features = []
    for i, ns in enumerate(neighbours):
        profile = Counter(
            (weights[j], overlap,
             (masks[i] & masks[j]).bit_count(),
             tuple((degree, (relation[i] & relation[j]).bit_count())
                   for degree, relation in ordered_relations))
            for j, overlap in ns.items())
        features.append((
            weights[i], masks[i].bit_count(),
            tuple((degree, relation[i].bit_count())
                  for degree, relation in ordered_relations),
            tuple(sorted(profile.items()))))
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]

    for _ in range(2):
        qubits = [tuple(sorted(colors[r] for r in checks))
                  for checks in incidence]
        qp = {f: c for c, f in enumerate(sorted(set(qubits)))}
        check_profiles = [[] for _ in rows]
        for checks, feature in zip(incidence, qubits):
            for r in checks:
                check_profiles[r].append(qp[feature])
        refined = [(colors[r], tuple(sorted(check_profiles[r])))
                   for r in range(len(rows))]
        history.append((
            tuple(sorted(Counter(qubits).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]

    signature = (
        width, len(rows),
        tuple(sorted(Counter(len(c) for c in incidence).items())),
        tuple(history))
    return sha256(repr(signature).encode("ascii")).digest()


def _low_embedded(G, record, kind, checks, stabilizers, own):
    """Actual logical witnesses; verify the complete embedding first."""
    q, width = G.q, 3*G.q
    candidates = []
    for block in range(3):
        source = record[4]
        if kind == "X":
            word = source << (block*width)
        else:
            word = 0
            while source:
                bit = source & -source
                j, h = divmod(bit.bit_length()-1, q)
                word |= 1 << ((j*3+block)*q+h)
                source ^= bit
        assert all((row & word).bit_count() % 2 == 0 for row in checks)
        if not _lp_outside(word, own):
            continue
        for _ in range(3):
            previous = word
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            starting = word
            nearby = sorted(
                stabilizers,
                key=lambda row: (starting ^ row).bit_count())[:40]
            weight = word.bit_count()
            for i, first in enumerate(nearby):
                for second in nearby[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
            if word == previous:
                break
        assert word and _lp_outside(word, own)
        assert all((row & word).bit_count() % 2 == 0 for row in checks)
        candidates.append(word)
    return min(candidates, key=int.bit_count) if candidates else 0


@lru_cache(maxsize=1)
def _low_portfolio(G, count):
    """Pure deterministic portfolio; memoization affects only build cost."""
    assert G.q == 44
    cap = min(9, W_MAX)
    banks = {
        (side, p, mode): _low_bank(G, side, p, mode)
        for side in ("A", "B") for p in range(4) for mode in range(2)
    }
    profile_pairs = tuple(
        (a, b) for a in range(4) for b in range(a, 4))
    excluded, evaluated, entries = set(), {}, []

    def evaluate(pa, ma, ai, pb, mb, bi):
        identity = (pa, ma, ai, pb, mb, bi)
        if identity in evaluated:
            return evaluated[identity]
        evaluated[identity] = None
        a = banks["A", pa, ma][ai]
        b = banks["B", pb, mb][bi]
        A, B = a[3], b[3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row)+sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(3))
        wz = max(sum(row)+sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert wx <= 9 and wz <= 9
        assert _no_dead_lines(sa, sb)
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        bx, bz = _lp_basis(hx), _lp_basis(hz)
        k = n-len(bx)-len(bz)
        if n != 572 or not (k > 0 and 8*k < n):
            return None
        assert k >= 44
        if not (_lp_clean_four(hx, n) and _lp_clean_four(hz, n)):
            return None
        if not css_ok(hx, hz):
            raise ValueError("verified low-rate lift failed CSS")
        assert max(row.bit_count() for row in hx+hz) <= cap

        signature = _low_graph_key(hx, hz, n)
        xs = _low_embedded(G, b, "X", hz, hx, bx)
        zs = _low_embedded(G, a, "Z", hx, hz, bz)
        local = random.Random(
            BASE_SEED ^ (0x4423CC + 65537*pa + 7919*ma + 1009*ai
                         + 104729*pb + 257*mb + bi))
        dx, xword = _lp_probe(
            hz, n, local, 4, own=bx, seed=xs, stabilizers=hx)
        dz, zword = _lp_probe(
            hx, n, local, 4, own=bz, seed=zs, stabilizers=hz)
        if not xword or not zword:
            return None
        reading = min(dx, dz)
        cycles = a[1][0]+b[1][0], a[1][1]+b[1][1]
        result = (
            A, B, k, reading, signature, (a[0], b[0]),
            cycles, (pa, ma, ai, pb, mb, bi), (xword, zword))
        evaluated[identity] = result
        return result

    for idx in range(count):
        card, epoch = idx % 20, idx // 20
        pa, pb = profile_pairs[card//2]
        ma = (card % 2 + epoch) % 2
        mb = (card % 2 + card//2 + epoch//2) % 2
        survivors = []

        # Primary prescription first. Structural alternatives rescue a
        # failed card without returning a relabeling of a prior output.
        alternatives = (
            (pa, ma, pb, mb),
            (pa, 1-ma, pb, mb),
            (pa, ma, pb, 1-mb),
            ((pa+1) % 4, ma, (pb+1) % 4, mb),
        )
        for ap, am, bp, bm in alternatives:
            aa, bb = banks["A", ap, am], banks["B", bp, bm]
            plans = sorted(
                (-min(a[0], b[0]), -a[0]-b[0],
                 a[1][0]+b[1][0], a[1][1]+b[1][1], ai, bi)
                for ai, a in enumerate(aa) for bi, b in enumerate(bb))
            for _, _, _, _, ai, bi in plans:
                record = evaluate(ap, am, ai, bp, bm, bi)
                if record is None or record[4] in excluded:
                    continue
                k, reading, cycles = record[2], record[3], record[6]
                pressure = 1+idx % 5
                # Witness-backed selection only; no claimed distance
                # lower bound beyond the exact short-kernel cutoff.
                quality = (
                    k*min(reading, 30)**2/572,
                    reading, -pressure*cycles[0]-cycles[1], k)
                survivors.append((quality, record))
                if len(survivors) >= 3:
                    break
            if survivors:
                break

        if not survivors:
            entries.append(None)
            continue
        quality, winner = max(survivors, key=lambda item: item[0])
        excluded.add(winner[4])
        entries.append((quality, winner))
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish one complete immutable low-rate design."""
    catalogue = _low_portfolio(G, N_SLOTS)
    entry = catalogue[idx]
    if entry is None:
        raise ValueError("distinct low-rate structural card exhausted")
    quality, record = entry
    A, B, k, reading, signature, components, cycles, card, words = record
    assert 8*k < 572
    return {
        "A": A, "B": B, "screen_k": k,
        "logical_witness_upper_bound": reading,
        "component_witness_upper_bounds": components,
        "design_card": card, "selection_quality": quality,
        "tanner_signature": signature.hex(),
        "verified_logical_words": words,
        "complete_kernel_exclusion_through": 4,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected two-row cell without shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent-first selection with complete short-word screens used only as diagnostics protects all finalists from weight-four logicals: parent slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: The certificate catalogue's Tanner-key gate prevents evaluator isomorphism duplicates: current slots 3 and 4 collapsed to slot 0, and sixty slots retained only twenty distinct codes.
# REFUTED: Quotient cards with intrinsic ownership supply sixty publishable inequivalent order-117 cubic slots: the measured table retained 41 distinct codes, with fourteen isomorphism duplicates and five ownership-bucket failures.
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
