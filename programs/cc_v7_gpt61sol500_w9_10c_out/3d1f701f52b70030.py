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
CONCEPT = ("On C11 semidirect C5 with action 3, order 55, A=(1,3), B=(1,3), admitting constituent dependencies only when complete CSS homology makes them stabilizers may uncover fresh rank-surplus distance-five codes or a certified logical floor of six in cell 0.400.")
GROUP = ("Frozen L1 group and q=55, retained byte-for-byte. This bed reached measured fitness 10.2002; all five current finalists have k=222,d30s=5. Nonidentity orders five and eleven explain candidate cycle witnesses but do not certify quantum distance. Test a broader support and homology model on the successful bed.")
SHAPES = ("Frozen equal (1,3) sides have 165 variables and 55 checks per constituent, n=550, kb=220 and guaranteed rate 0.400; the size window permits q<=59. Current weight-five minima occupy sector-1 Acol/Bcol blocks, with X families concentrated at Bcol2 across all A columns. The earlier weight-four failure also involved Arow0/Brow0. Both complete logical spaces and actual ranks are screened, retaining [0.375,0.425).")
FILLING = ("A rank-relaxed binomial branch precedes twenty optimized support/quotient cards. Cards admit unit-containing and four-/five-term profiles, repeated directions, two order-five reinforcements and commuting C5 trinomials, rather than requiring the parent's connected binomial A and rotation/order-five B reinforcement. Verified full matrices undergo exact stabilizer-aware logical exclusion through four. Explicit five-witnesses or exhaustive triple/pair comparisons distinguish certified floors five and six; actual k times floor squared sets priority. Known archive references and equal intrinsic Tanner invariants are conservatively excluded. Immutable memoization changes build cost only. Strict frontier improvement and novelty against unshown archive entries remain measurement hypotheses.")
FAMILY_NAME = "r040_frobenius11by5_13x13_support_relaxed_homology5_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    # x^11=y^5=1, y*x*y^-1=x^3; 3 has order five modulo 11.
    if not valid_q(q):
        raise ValueError("this affine group has order 55")
    powers = [pow(3, b, 11) for b in range(5)]
    els = [(a, b) for b in range(5) for a in range(11)]

    def red(g):
        return (g[0] % 11, g[1] % 5)

    def mul(g, h):
        return ((g[0] + powers[g[1] % 5] * h[0]) % 11,
                (g[1] + h[1]) % 5)

    def inv(g):
        bi = (-g[1]) % 5
        return ((-powers[bi] * g[0]) % 11, bi)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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


def _legacy_fivecount_reference(rng, G, idx):
    """Inactive historical catalogue; the new publisher does not call it."""
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


from functools import lru_cache


@lru_cache(maxsize=1)
def _support_relaxed_portfolio(G, count):
    """Build an immutable portfolio using only scaffold-produced matrices."""
    from itertools import combinations_with_replacement
    from collections import Counter
    from hashlib import sha256

    q, e = G.q, G.e
    assert q == 55
    cap = min(9, W_MAX)
    local = random.Random(BASE_SEED ^ 0x55D6A3)

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def basis(rows):
        out = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                if p not in out:
                    out[p] = word
                    break
                word ^= out[p]
        return out

    def outside(word, own):
        while word:
            p = word.bit_length() - 1
            old = own.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def quantum_four(checks, stabilizers, width):
        """Exact logical exclusion through four, permitting stabilizers.

        Every support of size <=4 splits into two supports of size <=2.
        For each syndrome, one representative suffices: all collisions
        must differ from that representative by a stabilizer.
        """
        cols = columns(checks, width)
        own = basis(stabilizers)
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            word = 1 << j
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif outside(word ^ old, own):
                return None
        for j, first in enumerate(cols):
            bit = 1 << j
            for i in range(j):
                syndrome = first ^ cols[i]
                word = bit | (1 << i)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif outside(word ^ old, own):
                    return None
        return cols, own, seen

    def has_logical_five(screen):
        """Exhaustive triple/pair test after quantum_four has passed."""
        cols, own, seen = screen
        bits = [1 << j for j in range(len(cols))]
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                word = bits[j] | bits[i]
                for h in range(i):
                    old = seen.get(syndrome ^ cols[h])
                    if old is not None:
                        witness = word ^ bits[h] ^ old
                        if outside(witness, own):
                            assert witness.bit_count() == 5
                            return True
        return False

    def explicit_five(A, B, hx, hz, sx, sz):
        """Test actual binomial-cycle logicals; no presumed embedding."""
        for kind, grid, checks, own in (
                ("X", B, hz, sx[1]), ("Z", A, hx, sz[1])):
            for j, cell in enumerate(grid[0]):
                if len(cell) != 2 or e not in cell:
                    continue
                generator = next(g for g in cell if g != e)
                power, orbit = e, []
                for _ in range(5):
                    orbit.append(G.idx(power))
                    power = G.mul(power, generator)
                if power != e or len(set(orbit)) != 5:
                    continue
                for block in range(3):
                    if kind == "X":
                        word = sum(
                            1 << ((block * 3 + j) * q + t)
                            for t in orbit)
                    else:
                        word = sum(
                            1 << ((j * 3 + block) * q + t)
                            for t in orbit)
                    if (all((row & word).bit_count() % 2 == 0
                            for row in checks)
                            and outside(word, own)):
                        return True
        return False

    def graph_key(hx, hz, width):
        """Intrinsic uncoloured incidence invariants, not label keys.

        Equality is inconclusive and is rejected conservatively.
        Inequality separates check/qubit-preserving Tanner isomorphisms.
        """
        rows = hx + hz
        degrees = [row.bit_count() for row in rows]
        features = []
        for i, row in enumerate(rows):
            profile = Counter(
                (degrees[j], (row & other).bit_count())
                for j, other in enumerate(rows))
            features.append((degrees[i], tuple(sorted(profile.items()))))
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]
        qubits = Counter()
        for syndrome in columns(rows, width):
            neighbours = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbours.append(colors[bit.bit_length() - 1])
                syndrome ^= bit
            qubits[tuple(sorted(neighbours))] += 1
        signature = (
            _tanner_signature(hx, hz),
            tuple(sorted(Counter(features).items())),
            tuple(sorted(qubits.items())))
        return sha256(repr(signature).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def binomials(key):
        return [[[e, G.elements[t]] for t in key]]

    excluded = set()
    # References are used only for exclusion, never published.
    known_binary = (
        (((9, 0), (2, 2), (3, 2)),
         ((10, 0), (8, 2), (3, 2))),
        (((10, 0), (1, 1), (9, 2)),
         ((10, 0), (10, 1), (1, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((1, 0), (0, 2), (5, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((0, 1), (1, 2), (10, 2))),
        (((1, 0), (0, 2), (5, 2)),
         ((0, 1), (1, 2), (10, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((2, 0), (0, 1), (10, 2))),
    )
    for aa, bb in known_binary:
        A, B = [[[e, g] for g in aa]], [[[e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(graph_key(hx, hz, n))

    # Every distinct filling listed in the current parent's table.
    # Packet: A prescription, two trinomial terms, rotation, C5 term.
    aa = (
        ((9, 0), (2, 2), (3, 2)),
        ((10, 0), (1, 1), (9, 2)),
    )
    parent_packets = (
        (0, (1,0), (9,1), (7,0), (7,1)),
        (0, (7,0), (4,2), (10,0), (4,1)),
        (0, (5,0), (0,3), (1,0), (10,1)),
        (0, (7,1), (5,2), (2,0), (9,1)),
        (0, (0,1), (4,3), (5,0), (1,1)),
        (1, (4,2), (5,3), (10,0), (7,1)),
        (0, (2,0), (10,1), (3,0), (4,2)),
        (0, (2,0), (0,2), (1,0), (2,2)),
        (0, (2,0), (9,3), (3,0), (3,2)),
        (1, (0,1), (8,1), (9,0), (5,2)),
        (0, (1,1), (3,2), (7,0), (2,2)),
        (0, (7,1), (9,3), (5,0), (10,2)),
        (1, (0,1), (0,4), (4,0), (2,2)),
        (0, (0,2), (10,3), (10,0), (10,2)),
        (0, (0,1), (4,2), (4,0), (1,3)),
        (0, (3,2), (10,2), (1,0), (8,3)),
        (0, (10,1), (7,3), (7,0), (8,4)),
        (1, (10,0), (4,2), (7,0), (1,2)),
        (1, (5,1), (3,2), (9,0), (8,2)),
        (1, (3,2), (4,3), (6,0), (6,2)),
    )
    for a, g, h, s, t in parent_packets:
        A = [[[e, x] for x in aa[a]]]
        B = [[[e, g, h], [e, s], [e, t]]]
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(graph_key(hx, hz, n))

    filling_keys, reserved = set(), set()
    results = []

    def inspect(A, B, card):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            return None
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > cap
                or not _no_dead_lines(sa, sb)):
            return None
        if any(len({G._key(g) for g in cell}) != len(cell)
               for grid in (A, B) for row in grid for cell in row):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or not (375*n <= 1000*k < 425*n):
            return None
        signature = graph_key(hx, hz, n)
        if signature in excluded or signature in reserved:
            return None

        sx = quantum_four(hz, hx, n)
        if sx is None:
            return None
        sz = quantum_four(hx, hz, n)
        if sz is None:
            return None
        if (explicit_five(A, B, hx, hz, sx, sz)
                or has_logical_five(sx) or has_logical_five(sz)):
            floor = 5
        else:
            floor = 6
        del sx, sz
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap
        reserved.add(signature)
        quality = (k * floor * floor / float(n), floor, k)
        return quality, freeze(A), freeze(B), k, floor, card

    # Include repeated directions and proper-subgroup constituents.
    # These are proposals only: full homology, not component girth,
    # decides whether their short dependencies are harmless.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != e
    })
    orbit_keys = set()
    for anchor in (G.idx((1, 0)), G.idx((0, 1)), G.idx((0, 2))):
        for tail in combinations_with_replacement(reps, 2):
            orbit_keys.add(_generator_key(
                G, [G.elements[t] for t in (anchor,) + tail]))

    bank = []
    for key in sorted(orbit_keys):
        grid = binomials(key)
        rows = _component_rows(G, "A", grid)
        reading, collisions = _kernel_profile(rows, 3*q)
        bank.append((gf2_rank(rows), reading, collisions, key, grid))
    clean = [item for item in bank if item[1] == 5]
    if not clean:
        raise ValueError("no clean constituent for support-relaxed cards")
    clean.sort(key=lambda item: (item[2], item[3]))

    plans = [
        (-(2*q-a[0]-b[0]), -min(a[1], b[1]), ai, bi, scale)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai)
        for scale in (1, 2)
    ]
    plans.sort()
    # Reserve a finite budget for the deliberately rank-relaxed branch.
    for _, _, ai, bi, scale in plans[:160]:
        A = bank[ai][4]
        B = [[[
            ((scale*g[0]) % 11, g[1]) for g in cell
        ] for cell in bank[bi][4][0]]]
        result = inspect(A, B, ("rank-relaxed", ai, bi, scale))
        if result is not None:
            results.append(result)
    results.sort(key=lambda item: item[0], reverse=True)
    results = results[:count]

    pools = [[g for g in G.elements if g != e and g[1] == b]
             for b in range(5)]
    triangles = (
        (0,0), (0,1), (0,2), (1,1), (1,2),
        (1,3), (1,4), (2,2), (2,3), (2,4))
    # Every profile obeys both LP weight inequalities by construction.
    profiles = (
        ((2,2,2), (3,2,2)),
        ((1,3,3), (2,2,2)),
        ((1,2,3), (1,2,3)),
        ((1,1,4), (1,2,2)),
        ((1,1,5), (1,1,2)),
        ((1,1,3), (2,2,2)),
    )

    def draw_grid(counts, position, trial):
        grid = []
        for j, terms in enumerate(counts):
            if terms == 1:
                grid.append([e])
                continue
            if terms == 2:
                # Both reinforcements may have order five; rotation is
                # no longer mandatory. Quotient cards differ intrinsically.
                b = (1 + (position + j + trial//5) % 4
                     if trial % 5 else 0)
                grid.append([e, local.choice(pools[b])])
                continue
            if terms == 3 and trial % 8 == 0:
                # Commuting C5 polynomial, not merely a shifted filling.
                g = local.choice(pools[1 + position % 4])
                exponent = 2 + (position + trial//8) % 3
                h = e
                for _ in range(exponent):
                    h = G.mul(h, g)
                grid.append([e, g, h])
                continue
            if terms == 3:
                b0, b1 = triangles[position % 10]
                g, h = local.choice(pools[b0]), local.choice(pools[b1])
                if g == h:
                    h = local.choice([x for x in pools[b1] if x != g])
                grid.append([e, g, h])
                continue
            # Quartic/quintic cards vary quotient occupancy, not gauges.
            chosen = [e]
            for t in range(terms - 1):
                b = (position + j + t + trial//7) % 5
                options = [g for g in pools[b] if g not in chosen]
                chosen.append(local.choice(options))
            grid.append(chosen)
        return [grid]

    while len(results) < count:
        position = len(results)
        card, epoch = position % 20, position // 20
        survivors = []
        for trial in range(320):
            # Most budget goes to the demonstrated distance-five support.
            # Other profiles test the genuinely new absorption hypothesis.
            policy = 0 if trial % 5 else 1 + (card + trial//5) % 5
            ac, bc = profiles[policy]
            if policy == 0:
                source = clean[(card + epoch + trial//3) % len(clean)]
                A = source[4]
                B = draw_grid(bc, card + 3*epoch, trial)
            else:
                A = draw_grid(ac, card + epoch, trial)
                B = draw_grid(bc, card + 7 + 2*epoch, trial + 1)
            if card >= 10:
                A, B = B, A

            result = inspect(A, B, ("support-card", card, epoch, policy))
            if result is not None:
                survivors.append(result)
                if result[4] >= 6 or len(survivors) >= 3:
                    break
        if not survivors:
            raise ValueError("fresh support-relaxed homology card exhausted")
        # Every survivor was reserved; discarded competitors cannot be
        # recycled as nominally different slots.
        results.append(max(survivors, key=lambda item: item[0]))

    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure publication from immutable, fully screened portfolio entries."""
    quality, A, B, k, floor, card = _support_relaxed_portfolio(G, N_SLOTS)[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "certified_logical_floor": floor,
        "design_card": card, "selection_quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without mutating the cached portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent-first selection with complete short-word screens used only as diagnostics protects all finalists from weight-four logicals: parent slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: The five-count quotient cards and their Tanner-invariant reservations prevent evaluator isomorphic repeats: the measured current sixty-slot table retained only twenty distinct codes, with forty iso_dup slots.
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
