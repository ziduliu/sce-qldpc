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
CONCEPT = ("On C5 semidirect C4 with action 2, order 20, A=B=(2,5), twenty optimized sparse constituent prescriptions and exact kernel screening through six should supply fresh distance-eight measured finalists in cell 0.300.")
GROUP = ("Retain the metacyclic semidirect-product construction family. Factor 29 makes the parent's order 55 inadmissible; m=5,r=4,s=2 gives order 20 and n=580, the largest lift this shape permits. This exact bed measured distance seven and fitness 15.2. Distance eight and archive novelty remain falsifiable hypotheses, not consequences of the group presentation.")
SHAPES = ("Change both sides to (2,5), giving equal 100-variable, 40-check constituents, n=580, kb=180 and guaranteed rate 9/29=0.310345. The size window permits q<=20. Target neighbouring cell 0.300 because this shape accommodates distributed cubic columns under cap nine. Parent weight-five minima occupied all nine sector-1 blocks; earlier failures also touched sector 2. Verified full ranks retain [0.275,0.325).")
FILLING = ("Four row-six, column-at-most-three support arrangements times five element prescriptions define twenty optimization cards per side. Both CSS maxima are nine. Optimize actual verified constituent witnesses, prioritizing exact kernel exclusion through six; weaker constituent screens remain explicitly distinguished. Pair independently optimized left/right constituents, reject complete kernels through four, verify target-band ranks and CSS, and search actual nontrivial mixed-sector witnesses with equal budgets. Intrinsic uncoloured Tanner invariants exclude repeated classes and the shown order-20 frontier reference. Only the complete kernel-four gate certifies a quantum floor of five; larger witness readings are upper bounds.")
FAMILY_NAME = "r030_metacyclic5by4_q20_25x25_kernel6_witnesscards_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Same metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 29 requires q <= 20; 2 has order four modulo 5.
    m, r, s = 5, 4, 2
    if not valid_q(q):
        raise ValueError("this metacyclic group has order 20")
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


def NEW_CTX(rng, G, idx):
    """Pure reconstruction of a fresh, certified-distance-five prefix."""
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
def _rankreserve_dual5_catalogue(G, count):
    """Build once; publish immutable, invariant-distinct complete fillings.

    The earlier NEW_CTX definition is retained as an uncalled reference.
    Rank and homology calculations use only the scaffold's verified lift.
    """
    from collections import Counter
    from hashlib import sha256

    assert G.q == 55
    q, width = G.q, 3 * G.q
    local = random.Random(BASE_SEED ^ 0x55D524)
    cap = min(9, W_MAX)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def logical_four(checks, stabilizers, n):
        """Exactly exclude NONTRIVIAL logical supports of weights 1..4.

        Two words of weight <=2 with equal syndromes must have equal
        residues modulo the stabilizer space. Short stabilizers are legal.
        """
        basis = {}
        for original in stabilizers:
            word = original
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old
        ordered = sorted(basis, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in ordered:
                if (word >> p) & 1:
                    word ^= basis[p]
            residues.append(word)

        cols = columns(checks, n)
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            residue = residues[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = residue
            elif old != residue:
                return False
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                residue = residues[j] ^ residues[i]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = residue
                elif old != residue:
                    return False
        return True

    def signature(hx, hz, n):
        rows = hx + hz
        value = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(
                c.bit_count() for c in columns(rows, n)).items())),
            _tanner_signature(hx, hz),
        )
        # This hashes an intrinsic invariant, not labels or a filling.
        return sha256(repr(value).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    reserved, fillings = set(), set()

    def inspect(A, B, minimum_k, card):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            return None
        fillings.add(filling)

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
        if n != 550 or k < minimum_k:
            return None
        if not (375 * n <= 1000 * k < 425 * n):
            return None
        if not logical_four(hz, hx, n):
            return None
        if not logical_four(hx, hz, n):
            return None

        sig = signature(hx, hz, n)
        if sig in reserved:
            return None
        reserved.add(sig)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap
        # Five is an exact logical lower bound, not a missing-witness cap.
        quality = (k * 25 / float(n), k)
        return quality, (
            freeze(A), freeze(B), k, card, sig.hex())

    # Conservatively exclude the shown archived pure-binomial classes.
    # The evaluator's complete archive remains authoritative.
    known = (
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
    for aa, bb in known:
        A = [[[G.e, g] for g in aa]]
        B = [[[G.e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        reserved.add(signature(hx, hz, n))

    # These alter incidence degrees, not the frozen protograph shapes.
    # Both CSS term-count maxima are independently checked below.
    profiles = (
        ((3, 2, 1), (3, 2, 1)),
        ((3, 3, 1), (2, 2, 1)),
        ((4, 1, 1), (3, 1, 1)),
        ((3, 2, 2), (2, 2, 1)),
    )
    nonidentity = [g for g in G.elements if g != G.e]
    rotations = [g for g in nonidentity if g[1] == 0]
    primary = []

    for pi, (sa, sb) in enumerate(profiles):
        assert _grid_weights([sa], [sb]) <= 9
        if _grid_weights([sa], [sb]) > cap:
            continue
        for trial in range(384):
            mode = trial % 4
            generator = (local.randrange(11), 1)
            subgroup, power = [], G.e
            for _ in range(4):
                power = G.mul(power, generator)
                subgroup.append(power)

            def draw(counts):
                row = []
                for terms in counts:
                    pool = (rotations if mode == 0 else
                            subgroup if mode == 1 else nonidentity)
                    if terms == 1:
                        cell = [G.e]
                    elif mode == 3:
                        # Algebraically related terms can create rank
                        # deficits; complete homology decides legality.
                        g = local.choice(nonidentity)
                        powers, value = [], G.e
                        for _ in range(terms - 1):
                            value = G.mul(value, g)
                            powers.append(value)
                        cell = [G.e] + powers
                    else:
                        cell = [G.e] + local.sample(pool, terms - 1)
                    row.append(cell)
                return [row]

            A, B = draw(sa), draw(sb)
            result = inspect(A, B, 224, ("rank-profile", pi, mode))
            if result is not None:
                primary.append(result)

    primary.sort(key=lambda item: (item[0], item[1][4]), reverse=True)
    entries = [entry for _, entry in primary[:count]]

    # Supplementary cards change the intrinsic reinforcement mechanism:
    # BOTH binomials use order-five quotients and must not commute.
    bank = []
    for score, key, A in _component_bank(G):
        if score[0] != 5:
            continue
        if gf2_rank(_component_rows(G, "A", A)) == q - 1:
            bank.append((key, A))
    bank.sort(key=lambda item: item[0])
    if not bank:
        raise ValueError("no clean connected supplementary constituent")

    pools = [[g for g in nonidentity if g[1] == b]
             for b in range(5)]
    triangles = (
        (0, 0), (0, 1), (0, 2), (1, 1), (1, 2),
        (1, 3), (1, 4), (2, 2), (2, 3), (2, 4))
    reinforcements = ((1, 1), (1, 2), (2, 2),
                      (1, 3), (2, 3), (2, 4))

    while len(entries) < count:
        position = len(entries)
        card, epoch = position % 20, position // 20
        b0, b1 = triangles[card % 10]
        finalists = []

        for trial in range(1024):
            c0, c1 = reinforcements[
                (card // 10 + epoch + trial // 32)
                % len(reinforcements)]
            g, h = local.choice(pools[b0]), local.choice(pools[b1])
            s, t = local.choice(pools[c0]), local.choice(pools[c1])
            if g == h or G.mul(s, t) == G.mul(t, s):
                continue
            B = [[[G.e, g, h], [G.e, s], [G.e, t]]]

            # This is only a cheap proposal filter. Publication still
            # tests both complete, stabilizer-aware logical spaces.
            rows = _component_rows(G, "B", B)
            if _kernel_profile(rows, width)[0] != 5:
                continue
            A = bank[(position + trial // 7) % len(bank)][1]
            result = inspect(
                A, B, 222, ("dual-five", card, epoch, c0, c1))
            if result is None:
                continue
            finalists.append(result)
            if len(finalists) >= 3:
                break

        if not finalists:
            raise ValueError("novel dual-five reinforcement card exhausted")
        # All inspected survivors are reserved, including discarded ones,
        # so a later slot cannot recycle a discarded finalist.
        _, winner = max(
            finalists, key=lambda item: (item[0], item[1][4]))
        entries.append(winner)

    entries.sort(key=lambda entry: (-entry[2], entry[4]))
    return tuple(entries)


def _p25_columns(rows, width):
    """Syndromes of the supplied verified matrix."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _p25_basis(rows):
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


def _p25_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _p25_clean6(rows, width):
    """Exactly exclude nonzero kernel supports of weights one through six.

    Every such support splits into two words of weight at most three.
    Enumerated words are distinct, so a repeated syndrome is decisive.
    """
    cols = _p25_columns(rows, width)
    seen = {0}
    for c in cols:
        if c in seen:
            return False
        seen.add(c)
    for i, c in enumerate(cols):
        for j in range(i):
            s = c ^ cols[j]
            if s in seen:
                return False
            seen.add(s)
    for i, c in enumerate(cols):
        for j in range(i):
            pair = c ^ cols[j]
            for h in range(j):
                s = pair ^ cols[h]
                if s in seen:
                    return False
                seen.add(s)
    return True


def _p25_component(G, grid, side):
    """Expose only the scaffold's exact left or right constituent action."""
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], grid)
    assert len(rows) == len(grid) * G.q
    return [row & mask for row in rows]


def _p25_probe(local, rows, width, passes, seed=0, own=None):
    """Search actual kernel/logical witnesses; high readings are upper bounds."""
    cols = _p25_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _p25_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
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

        local.shuffle(nulls)
        for _ in range(2):
            for i in range(len(nulls)):
                current = nulls[i]
                w = current.bit_count()
                for j, other in enumerate(nulls):
                    if i == j:
                        continue
                    trial = current ^ other
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        current, w = trial, tw
                nulls[i] = current
                inspect(current)

        short = sorted(set(nulls), key=int.bit_count)[:28]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 18:
                    for h in range(j):
                        inspect(pair ^ short[h])
        if best and weight <= 7:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _p25_outside(best, own)
    return best


def _p25_draw(local, G, profile, policy):
    """Five intrinsic label prescriptions, subsequently optimized."""
    grid = [[[] for _ in profile[0]] for _ in profile]
    nonidentity = [g for g in G.elements if g != G.e]
    for i, counts in enumerate(profile):
        for j, count in enumerate(counts):
            if not count:
                continue
            if count == 1:
                # Independent cross voltages, not a completed-code gauge.
                grid[i][j] = [local.choice(G.elements)]
                continue

            for _ in range(64):
                if policy == 0:
                    tail = local.sample(nonidentity, count - 1)
                elif policy == 1:
                    bs = [1, 3] if count == 3 else [1 + 2 * local.randrange(2)]
                    tail = [(local.randrange(5), b) for b in bs]
                elif policy == 2:
                    bs = [0, 1 + local.randrange(3)] if count == 3 else [
                        1 + local.randrange(3)]
                    tail = [(local.randrange(1, 5) if b == 0
                             else local.randrange(5), b) for b in bs]
                elif policy == 3:
                    bs = [1, 2] if count == 3 else [2]
                    tail = [(local.randrange(5), b) for b in bs]
                else:
                    tail = local.sample(nonidentity, count - 1)

                cell = [G.e] + tail
                if len(set(cell)) != count:
                    continue
                if count == 3 and policy != 3:
                    if G.mul(tail[0], tail[1]) == G.mul(tail[1], tail[0]):
                        continue
                if policy == 4:
                    anchor = local.choice(G.elements)
                    cell = [G.mul(anchor, g) for g in cell]
                grid[i][j] = cell
                break
            else:
                return None
    return grid


def _p25_bank(G, side):
    """Twenty optimization cards, not twenty independently emitted draws."""
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 0, 1, 1, 1), (0, 3, 1, 1, 1)),
        ((2, 2, 1, 1, 0), (1, 1, 1, 1, 2)),
        ((0, 2, 2, 1, 1), (2, 0, 1, 2, 1)),
    )
    bank, used = [], set()
    width = 5 * G.q
    for card in range(20):
        pi, policy = divmod(card, 5)
        profile = profiles[pi]
        assert all(sum(row) == 6 for row in profile)
        assert max(sum(profile[i][j] for i in range(2))
                   for j in range(5)) == 3
        local = random.Random(
            BASE_SEED ^ (0x205C80 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        raw, fillings = [], set()

        for trial in range(96):
            grid = _p25_draw(local, G, profile, policy)
            if grid is None:
                continue
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _p25_component(G, grid, side)
            if _kernel_profile(rows, width)[0] != 5:
                continue
            penalty = sum(
                max(0, (rows[i] & rows[j]).bit_count() - 1)
                for i in range(len(rows)) for j in range(i))
            raw.append((penalty, trial, grid, rows))

        raw.sort(key=lambda item: item[:2])
        finalists = []
        for penalty, trial, grid, rows in raw[:14]:
            floor = 7 if _p25_clean6(rows, width) else 5
            word = _p25_probe(local, rows, width, 6)
            if word:
                finalists.append(
                    (word.bit_count(), floor, penalty, trial, grid, rows, word))
        finalists.sort(key=lambda item: (-item[0], -item[1],
                                         item[2], item[3]))

        for _, floor, penalty, trial, grid, rows, seed in finalists[:3]:
            word = _p25_probe(local, rows, width, 24, seed)
            key = (
                tuple(sorted(r.bit_count() for r in rows)),
                tuple(sorted(c.bit_count()
                             for c in _p25_columns(rows, width))),
                _tanner_signature(rows, []))
            if key in used:
                continue
            used.add(key)
            bank.append(dict(
                grid=grid, word=word, reading=word.bit_count(),
                floor=floor, card=card, trial=trial, penalty=penalty))

    bank.sort(key=lambda r: (-r["reading"], -r["floor"],
                            r["penalty"], r["card"], r["trial"]))
    return bank


def _p25_signature(hx, hz, n):
    """Intrinsic uncoloured Tanner features, not element fingerprints."""
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    cols = _p25_columns(rows, n)
    degrees = [row.bit_count() for row in rows]
    qubits = Counter()
    for col in cols:
        neighbours = []
        while col:
            bit = col & -col
            neighbours.append(degrees[bit.bit_length() - 1])
            col ^= bit
        qubits[tuple(sorted(neighbours))] += 1

    value = (
        n, len(rows), tuple(sorted(Counter(degrees).items())),
        tuple(sorted(qubits.items())), _tanner_signature(hx, hz))
    return sha256(repr(value).encode("ascii")).digest()


def _p25_logical_reading(G, a, b, hx, hz, n, local):
    """Verify and shorten embedded and mixed-sector nontrivial witnesses."""
    q = G.q
    overall = n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _p25_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(5):
            if kind == "X":
                word = source << (block * 5 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 5 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _p25_outside(word, own):
                continue

            for _ in range(2):
                starting, weight = word, word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                moves = sorted(
                    (row for row in stabilizers
                     if (row & word).bit_count() >= 2),
                    key=lambda row: -(row & word).bit_count())[:48]
                anchor = word
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = anchor ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and _p25_outside(word, own)
            seeds.append(word)

        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        word = _p25_probe(local, checks, n, 3, seed, own)
        if not word:
            return None
        overall = min(overall, word.bit_count())
    return overall


@lru_cache(maxsize=1)
def _p25_portfolio(G, count):
    """Immutable deterministic portfolio; all matrix screens are verified."""
    assert G.q == 20
    cap = min(9, W_MAX)
    if cap < 9:
        raise ValueError("row-six cubic cards require cap nine")
    aa, bb = _p25_bank(G, "A"), _p25_bank(G, "B")
    if not aa or not bb:
        raise ValueError("no screened order-20 constituent bank")

    reserved = set()
    # Shown frontier reference only; unseen archive novelty is evaluator-tested.
    reference = [
        [[(0, 1), (2, 1)], [(3, 2)], [G.e], [G.e], [G.e]],
        [[(0, 2)], [G.e, (3, 1)], [G.e], [(2, 0)], [(1, 3)]],
    ]
    hx, hz, n = lifted_matrices(G, reference, reference)
    reserved.add(_p25_signature(hx, hz, n))

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -min(a["floor"], b["floor"]),
         a["penalty"] + b["penalty"], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))
    results, fillings = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(5))
        assert wx == wz == 9
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (275 * n <= 1000 * k < 325 * n):
            continue
        sig = _p25_signature(hx, hz, n)
        if sig in reserved:
            continue
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap

        local = random.Random(
            BASE_SEED ^ (0x205D80 + 65537 * ai + 1009 * bi))
        reading = _p25_logical_reading(G, a, b, hx, hz, n, local)
        if reading is None:
            continue
        reserved.add(sig)
        quality = (k * reading * reading / float(n), reading, k,
                   -a["penalty"] - b["penalty"])
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), (a["floor"], b["floor"]),
            (a["reading"], b["reading"]), sig.hex()))
        if len(results) >= count + 40:
            break

    results.sort(key=lambda item: (item[0], item[-1]), reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of an independently optimized two-row filling."""
    portfolio = _p25_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("invariant-distinct order-20 portfolio exhausted")
    quality, A, B, k, reading, cards, floors, components, sig = portfolio[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "logical_floor": 5,
        "logical_witness_upper_bound": reading,
        "design_cards": cards,
        "constituent_kernel_floors": floors,
        "constituent_witness_upper_bounds": components,
        "selection_quality": quality,
        "tanner_signature": sig,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent-first selection with complete short-word screens used only as diagnostics protects all finalists from weight-four logicals: parent slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: Low-weight-five multiplicity selection alone raises this bed's frontier: the current program measured five k=222,d30s=5 finalists and scored 10.0910, below the recorded 10.2002 frontier.
# REFUTED: Rank-reserve term profiles supply k>=224 measured finalists on the order-55 single-row bed: the current table measured five k=222,d30s=5 finalists, leaving fitness unchanged at 10.0910.
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
