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
CONCEPT = ("Dic5, order 20, A=B=(2,5): independently optimized cubic-reinforced constituents and complete mixed-sector screening should supply five novel distance readings >=8 in cell 0.300; measurement can refute this hypothesis.")
GROUP = ("Jump from C11 semidirect C5 to dicyclic Dic5. The measured Dic11 (2,3)x(2,3) bed reached fitness 46.5, motivating a smaller dicyclic experiment without transferring its distance claims. Dic5 rotations have orders five or ten, and outside elements have order four. Isolated binomials use order-ten rotations; mixed-coset reinforcement is screened rather than assumed safe.")
SHAPES = ("Equal (2,5) constituents have 100 variables and 40 checks. Factor 29 permits q<=20; q=20 gives n=580, kb=180 and guaranteed rate 9/29=0.310345 in cell 0.300, whose frontier currently measures distance seven. Parent weight-five minima occupied sector-1 Acol/Bcol blocks, especially Bcol2 across every A column; an earlier failure also touched Arow0/Brow0. Replace the single-row constituents and screen both complete sectors. Actual ranks retain [0.275,0.325).")
FILLING = ("Twenty cards combine four saturated count profiles with two mixed-coset policies. Every constituent row totals six and every column has at most three terms, so BOTH CSS weight maxima are nine, below twelve as well. Independently optimize verified constituents using exact kernel-four rejection, C5-invariant witnesses, cycle penalties and equally refined information-set probes. Pair finalists, reject complete kernels through four, verify logical witnesses and reduce them with stabilizers. Immutable portfolio entries have distinct intrinsic full-Tanner signatures. No archived filling seeds a candidate; external archive novelty remains evaluator-tested. Readings above the exact cutoff remain upper bounds.")
FAMILY_NAME = "r030_dic5_25x25_cubic_independent_witness_portfolio"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic5 has order 20")
    # x^10=1, y^2=x^5, y*x*y^-1=x^-1.
    # Canonical element (a,b) represents x^a y^b, b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 5 * (b // 2)) % 10, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 5 * b * d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 10, 0) if b == 0 else ((a + 5) % 10, 1)

    def idx(g):
        a, b = red(g)
        return 10 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
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


def _legacy_frobenius_certificate_reference(rng, G, idx):
    """Inactive historical reference; the Dic5 builder never calls this."""
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


def _d5_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _d5_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _d5_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _d5_component(G, grid, side):
    """Extract the exact evaluator component with a unit opposite side."""
    unit = [[[G.e]]]
    width = 5 * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    return [row & mask for row in rows[:2 * G.q]]


def _d5_invariant_seed(G, rows):
    """Find actual words in the normal-C5 invariant kernel.

    Each block has four C5 cosets. Elimination gives the complete
    invariant kernel; enumerate it when small, otherwise inspect its
    basis and pairs. No invariant-subspace result certifies full distance.
    """
    width = 5 * G.q
    cols = _d5_columns(rows, width)
    words, syndromes = [], []
    for block in range(5):
        for b in range(2):
            for parity in range(2):
                word = syndrome = 0
                for a in range(parity, 10, 2):
                    j = block * G.q + G.idx((a, b))
                    word |= 1 << j
                    syndrome ^= cols[j]
                words.append(word)
                syndromes.append(syndrome)

    pivots, nulls = {}, []
    for syndrome, word in zip(syndromes, words):
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

    best = 0

    def inspect(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

    if len(nulls) <= 14:
        previous = word = 0
        for number in range(1, 1 << len(nulls)):
            gray = number ^ (number >> 1)
            changed = (gray ^ previous).bit_length() - 1
            word ^= nulls[changed]
            inspect(word)
            previous = gray
    else:
        for i, word in enumerate(nulls):
            inspect(word)
            for other in nulls[:i]:
                inspect(word ^ other)
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _d5_probe(rng, rows, width, passes, seed=0, stabilizers=None):
    """Return an explicit kernel/logical witness; search is heuristic."""
    cols = _d5_columns(rows, width)
    span = _d5_basis(stabilizers) if stabilizers is not None else None
    best = 0

    def inspect(word):
        nonlocal best
        if not word or (best and word.bit_count() >= best.bit_count()):
            return
        if span is not None and not _d5_outside(word, span):
            return
        best = word

    inspect(seed)
    for _ in range(passes):
        order = list(range(width))
        rng.shuffle(order)
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

        # Elementary exchanges preserve the component kernel basis.
        # Full-code probes instead retain logical labels via exact span tests.
        if span is None:
            rng.shuffle(nulls)
            for i, word in enumerate(nulls):
                weight = word.bit_count()
                for j, other in enumerate(nulls):
                    if i == j:
                        continue
                    trial = word ^ other
                    w = trial.bit_count()
                    if 0 < w < weight:
                        word, weight = trial, w
                nulls[i] = word
                inspect(word)

        short = sorted(nulls, key=int.bit_count)[:32]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])

        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)

        if best and best.bit_count() <= 5:
            break

    if best and stabilizers is not None:
        # Adding checks preserves the logical class and commutation.
        for _ in range(3):
            previous = best
            for row in stabilizers:
                inspect(best ^ row)
            starting = best
            moves = [row for row in stabilizers
                     if (row & starting).bit_count() >= 2]
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    inspect(starting ^ first ^ second)
            if best == previous:
                break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        if span is not None:
            assert _d5_outside(best, span)
    return best


def _d5_cycle_cost(rows, width):
    cols = _d5_columns(rows, width)
    c4 = 0
    for i, first in enumerate(cols):
        for second in cols[:i]:
            overlap = (first & second).bit_count()
            c4 += overlap * (overlap - 1) // 2
    return c4


def _d5_signature(hx, hz, n):
    """Intrinsic uncoloured invariant; equal signatures are merged."""
    from hashlib import sha256
    from collections import Counter
    rows = hx + hz
    value = (
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(
            c.bit_count() for c in _d5_columns(rows, n)).items())),
        _tanner_signature(hx, hz),
    )
    return sha256(repr(value).encode("ascii")).digest()


def _d5_design(rng, G, idx, excluded):
    """Optimize one structural card without using any archived filling."""
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 1, 1, 1, 0), (0, 1, 1, 1, 3)),
        ((2, 2, 1, 1, 0), (1, 0, 1, 1, 3)),
        ((3, 2, 1, 0, 0), (0, 1, 1, 2, 2)),
    )
    pairs = tuple((a, b) for a in range(4) for b in range(a, 4))
    card, epoch = idx % 20, idx // 20
    pa, pb = pairs[card // 2]
    policy = card % 2
    sa, sb = profiles[pa], profiles[pb]
    cap = min(9, W_MAX)
    assert all(sum(row) == 6 for grid in (sa, sb) for row in grid)
    assert _grid_weights(sa, sb) == 9 <= cap
    assert _no_dead_lines(sa, sb)

    # Units modulo ten give exactly order-ten rotation ratios.
    long_rotations = ((1, 0), (3, 0), (7, 0), (9, 0))
    rotations = tuple((a, 0) for a in range(1, 10) if a != 5)

    def draw(profile, side):
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, count in enumerate(counts):
                if not count:
                    row.append([])
                    continue
                first = G.pick(rng)
                if count == 1:
                    cell = [first]
                elif count == 2:
                    degree = profile[0][j] + profile[1][j]
                    isolated = profile[1 - i][j] == 0
                    mixed = (not isolated and
                             (policy == 0 or
                              (i + j + epoch + int(side == "B")) % 2))
                    if mixed:
                        relative = (rng.randrange(10), 1)
                    else:
                        relative = rng.choice(
                            long_rotations if degree == 2 else rotations)
                    cell = [first, G.mul(first, relative)]
                else:
                    rotation = rng.choice(long_rotations)
                    cross = (rng.randrange(10), 1)
                    if (j + policy + epoch + int(side == "B")) % 2:
                        second = G.mul(first, cross)
                        cell = [first, second, G.mul(second, rotation)]
                    else:
                        cell = [first, G.mul(first, rotation),
                                G.mul(first, cross)]
                assert len(set(cell)) == count
                row.append(cell)
            grid.append(row)
        return grid

    def bank(profile, side):
        width = 5 * G.q
        proposals, fills = [], set()
        for trial in range(96):
            grid = draw(profile, side)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
            if key in fills:
                continue
            fills.add(key)
            rows = _d5_component(G, grid, side)
            if _kernel_profile(rows, width)[0] != 5:
                continue
            cost = _d5_cycle_cost(rows, width)
            proposals.append((cost, trial, grid, rows))
        proposals.sort(key=lambda item: item[:2])

        shortlist = []
        for cost, trial, grid, rows in proposals[:16]:
            seed = _d5_invariant_seed(G, rows)
            word = _d5_probe(rng, rows, width, 3, seed)
            if not word:
                continue
            shortlist.append(dict(
                grid=grid, rows=rows, word=word, cost=cost, trial=trial,
                quality=(word.bit_count(), -cost, -trial)))
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        finalists = shortlist[:4]
        for item in finalists:
            item["word"] = _d5_probe(
                rng, item["rows"], width, 18, item["word"])
            item["quality"] = (
                item["word"].bit_count(), -item["cost"], -item["trial"])
        finalists.sort(key=lambda item: item["quality"], reverse=True)
        return finalists[:3]

    aa, bb = bank(sa, "A"), bank(sb, "B")
    proposals = []
    for a in aa:
        for b in bb:
            weak = min(a["word"].bit_count(), b["word"].bit_count())
            proposals.append((
                (weak, -a["cost"] - b["cost"]), a, b))
    proposals.sort(key=lambda item: item[0], reverse=True)
    survivors = []

    for preliminary, a, b in proposals:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (k > 0 and 275 * n <= 1000 * k < 325 * n):
            continue
        signature = _d5_signature(hx, hz, n)
        if signature in excluded:
            continue
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap

        readings = []
        for kind, checks, own, source in (
                ("X", hz, hx, b["word"]),
                ("Z", hx, hz, a["word"])):
            span = _d5_basis(own)
            seed = 0
            for block in range(5):
                if kind == "X":
                    word = source << (block * 5 * G.q)
                else:
                    word = 0
                    bits = source
                    while bits:
                        bit = bits & -bits
                        j, h = divmod(bit.bit_length() - 1, G.q)
                        word |= 1 << ((j * 5 + block) * G.q + h)
                        bits ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if _d5_outside(word, span):
                    if not seed or word.bit_count() < seed.bit_count():
                        seed = word
            witness = _d5_probe(
                rng, checks, n, 3, seed, stabilizers=own)
            if witness:
                readings.append(witness.bit_count())

        # A missing probe is not rewarded with a high sentinel.
        if not readings:
            continue
        reading = min(readings)
        survivors.append(dict(
            A=A, B=B, k=k, signature=signature,
            component_readings=(
                a["word"].bit_count(), b["word"].bit_count()),
            logical_reading=reading,
            quality=(k * reading * reading / float(n),
                     reading, k, preliminary)))
        if len(survivors) >= 3:
            break

    return max(survivors, key=lambda item: item["quality"]) \
        if survivors else None


from functools import lru_cache


@lru_cache(maxsize=1)
def _d5_portfolio(G):
    """Deterministic immutable result; call order changes no filling.

    Memoization avoids repeated optimization. No returned entry or cell
    is mutated, and every slot is selected against the same prefix.
    """
    assert G.q == 20
    entries, excluded = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for idx in range(N_SLOTS):
        entry = None
        for restart in range(3):
            local = random.Random(
                BASE_SEED * 10007 + 0xD520300
                + 65537 * idx + 1000003 * restart)
            winner = _d5_design(local, G, idx, excluded)
            if winner is None:
                continue
            excluded.add(winner["signature"])
            entry = (
                freeze(winner["A"]), freeze(winner["B"]),
                winner["k"], winner["component_readings"],
                winner["logical_reading"])
            break
        entries.append(entry)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of an independently optimized structural card."""
    entry = _d5_portfolio(G)[idx]
    if entry is None:
        raise ValueError("Dic5 target-rate structural card exhausted")
    A, B, k, components, logical = entry
    return {"A": A, "B": B, "screen_k": k,
            "design_card": idx % 20,
            "component_witness_readings": components,
            "logical_witness_reading": logical,
            "complete_kernel_floor": 5}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh list from the immutable selected filling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent-first selection with complete short-word screens used only as diagnostics protects all finalists from weight-four logicals: parent slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
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
