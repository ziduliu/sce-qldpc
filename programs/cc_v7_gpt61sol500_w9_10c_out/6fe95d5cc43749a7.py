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
CONCEPT = ("Cyclic C17, order 17, A=B=(3,5): replacing two-row cofactor bottlenecks with equally sized three-row constituents and controlled augmentation deficits should supply five novel measured codes in cell 0.150 with fitness above 28.452; distance-reading survival is falsifiable.")
GROUP = ("Keep GTYPE=custom and the explicitly implemented cyclic construction rule, changing only its order from 31 to 17 because factor 34 cannot fit order 31. Seventeen is the largest admissible lift. Since 2 has order eight modulo 17, F2[C17] has the augmentation factor and two degree-eight factors. Verified constituent ranks require full rank on those nontrivial factors. The related C17 (3,5)x(3,5) bed measured fitness 12.0, so this is a filling improvement on a measured shape, not an untried-bed claim or a transferred distance guarantee.")
SHAPES = ("Equal A=B=(3,5) give n=578, kb=68 and guaranteed rate 2/17=0.117647. Both constituents have 85 variables and 51 checks; the size window permits q<=17. Augmentation ranks one/one with full nontrivial-factor ranks predict k=84, rate 0.145329; one/two predict k=78, and two/two k=74. Complete verified ranks enforce [0.125,0.175). Parent weight-fourteen minima occupied sector-1 Acol0..2/Bcol0..3 broadly, especially Acol1, with none of the reported minima in sector 2. Three check rows replace its two-row cofactors and expand the LP sectors to 25 and nine blocks.")
FILLING = ("Twenty cards per side combine four support profiles with five relative-voltage policies: all-monomial cubic columns, one binomial reinforcement, and two balanced zero/binomial redistributions. Monomial anchors remove common gauges; separated row-pair voltages reduce four-cycle pressure. Verified three-row cofactor witnesses include translated cross-minor sums, followed by equal-budget kernel refinement. Actual augmentation and constituent ranks gate the banks. Complete lifts enforce both row maxima <=9, target rate, CSS and kernel exclusion through four; embedded and mixed-sector logical witnesses rank the portfolio. Rooted uncoloured Tanner refinements conservatively reject equal signatures. Five k84,d14 readings would score 28.4844, just above the parent; larger readings, catalogue yield and unseen archive novelty remain hypotheses.")
FAMILY_NAME = "r015_c17_35x35_augmentation_three_row_crossminor_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same cyclic construction rule and one-coordinate encoding.
    # New shape factor 34 requires q <= 17.
    if not valid_q(q):
        raise ValueError("C17 has order 17")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    def idx(g):
        return int(g[0]) % q

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


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


def _extract_components(G, hx, hz):
    """Both constituent matrices, extracted from the verified full lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    br = [row & ((1 << (cb * q)) - 1)
          for row in hz[:rb * q]]
    return ar, br


def _odd_inverse_classes(G):
    """Six undirected odd-parity generator classes, none involutory."""
    classes = {}
    for g in G.elements:
        if g[0] % 2 == 1:
            key = min(G.idx(g), G.idx(G.inv(g)))
            classes.setdefault(key, g)
    pools = [
        [g for _, g in sorted(classes.items()) if g[1] == b]
        for b in range(2)
    ]
    assert len(pools[0]) == len(pools[1]) == 3
    assert all(G.mul(g, g) != G.e for pool in pools for g in pool)
    return pools


def _draw_incidence(rng, G, attempt, pools):
    """Twenty structural cards, optimized rather than emitted as draws.

    A's graph is bipartite under a mod 2. B varies its internal-edge
    allocation. With internal edges present, cross-matchings have one
    common parity, so the B incidence graph is also bipartite.
    """
    card = attempt % 20
    epoch = attempt // 20
    outside_a = card % 4
    generators = (rng.sample(pools[0], 3 - outside_a)
                  + rng.sample(pools[1], outside_a))
    rng.shuffle(generators)
    A = [[[G.e, g] for g in generators]]

    internal = card // 5
    B = [[], []]
    for row in range(2):
        # Independent inverse classes exclude parallel internal edges.
        chosen = rng.sample(pools[0] + pools[1], internal)
        for g in chosen:
            B[row].append([G.e, g])
            B[1 - row].append([])

    cross_count = 7 - 2 * internal
    if internal:
        parity = epoch % 2
        ratio_pools = [
            [g for g in G.elements
             if g[1] == b and g[0] % 2 == parity]
            for b in range(2)
        ]
        outside_b = (card + epoch) % (cross_count + 1)
    else:
        ratio_pools = [
            [g for g in G.elements if g[1] == b]
            for b in range(2)
        ]
        outside_b = 1 + (card + epoch) % 6

    ratios = (rng.sample(ratio_pools[0], cross_count - outside_b)
              + rng.sample(ratio_pools[1], outside_b))
    rng.shuffle(ratios)
    for g in ratios:
        B[0].append([G.e])
        B[1].append([g])

    assert len(B[0]) == len(B[1]) == 7
    assert all(sum(map(len, row)) == 7 for row in B)
    assert all(len(B[0][j]) + len(B[1][j]) == 2 for j in range(7))
    return A, B


def _no_kernel_through_three(rows, n):
    """Exactly exclude all nonzero kernel words of weights 1, 2 and 3."""
    columns = [0] * n
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, first in enumerate(columns):
        for j in range(i):
            if (first ^ columns[j]) in singles:
                return False
    return True


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
    return tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)))


def _unused_commonC12_reference(rng, G, idx):
    """Uncalled historical subgroup catalogue; never publishes cells.

    The first gate is the proposed rank gain, not an augmentation
    calculation. All component and quantum screens use verified lifts.
    """
    from itertools import combinations

    local = random.Random(BASE_SEED ^ 0xC1225845)
    rotations = {g[0]: g for g in G.elements if g[1] == 0}

    def x(a):
        return rotations[a % 12]

    # These distinct inverse classes give two K6,6 incidence components.
    # Do not vary them by automorphisms and call those new designs.
    A = [[[G.e, x(d)] for d in (1, 3, 5)]]
    sa = [[2, 2, 2]]

    def canon(top, bottom, ratios):
        """Remove known B gauges and common rotation automorphisms.

        Odd A inverse classes are preserved as a set by every unit
        modulo twelve. Internal edges can be reversed independently.
        B-row exchange negates cross voltages and exchanges edge sets.
        Equal keys are rejected; this is not a complete graph-iso test.
        """
        best = None
        for u in (1, 5, 7, 11):
            ta = tuple(sorted(min((u * d) % 12,
                                  (-u * d) % 12) for d in top))
            tb = tuple(sorted(min((u * d) % 12,
                                  (-u * d) % 12) for d in bottom))
            values = [(u * a) % 12 for a in ratios]
            for shift in range(12):
                forward = tuple(sorted((a + shift) % 12
                                       for a in values))
                reverse = tuple(sorted((-a + shift) % 12
                                       for a in values))
                for key in ((ta, tb, forward), (tb, ta, reverse)):
                    if best is None or key < best:
                        best = key
        return best

    # Deliberate orbit representatives, not seven fresh labels per slot.
    dense = sorted({
        canon((), (), values)
        for values in combinations(range(12), 7)
    })
    local.shuffle(dense)

    def draw_card(attempt):
        internal = attempt % 4
        epoch = attempt // 4
        policy = epoch % 6
        if internal == 0:
            return dense[epoch % len(dense)]

        cross_count = 7 - 2 * internal
        if internal == 3:
            # Six internal columns plus one bridge matching.
            top = bottom = (1, 3, 5)
            ratios = (0,)
        elif policy < 3:
            # A sufficient bipartite construction; all these steps
            # have order four or twelve, but labels remain rotations.
            top = tuple(local.sample((1, 3, 5), internal))
            bottom = tuple(local.sample((1, 3, 5), internal))
            parity = policy % 2
            ratios = tuple(local.sample(
                list(range(parity, 12, 2)), cross_count))
        elif policy == 3 and internal == 1:
            # Order-six cycles: this six-voltage set has no difference
            # +/-2, preventing triangles involving an internal edge.
            top = bottom = (2,)
            ratios = tuple(local.sample((0, 1, 4, 5, 8, 9), 5))
        elif policy == 3:
            # Triangle-free but not forced to have the parent's odd
            # grading. The verified screen decides which bridges work.
            top = bottom = (2, 3)
            ratios = tuple(local.sample(range(12), 3))
        else:
            # Broader order-four/six/twelve edge prescriptions.
            # Reject triangles below rather than trusting this proposal.
            top = tuple(local.sample((1, 2, 3, 5), internal))
            bottom = tuple(local.sample((1, 2, 3, 5), internal))
            ratios = tuple(local.sample(range(12), cross_count))
        return canon(top, bottom, ratios)

    def build_B(card):
        top, bottom, ratios = card
        B = [[], []]
        for d in top:
            B[0].append([G.e, x(d)])
            B[1].append([])
        for d in bottom:
            B[0].append([])
            B[1].append([G.e, x(d)])
        for a in ratios:
            B[0].append([G.e])
            B[1].append([x(a)])
        return B

    graph_keys, card_keys = set(), set()
    accepted = 0
    for attempt in range(8192):
        card = draw_card(attempt)
        if card in card_keys:
            continue
        card_keys.add(card)
        B = build_B(card)
        sb = [[len(cell) for cell in row] for row in B]
        assert len(B[0]) == len(B[1]) == 7
        assert all(sum(row) == 7 for row in sb)
        assert all(sb[0][j] + sb[1][j] == 2 for j in range(7))
        assert _no_dead_lines(sa, sb)
        if _grid_weights(sa, sb) > min(9, W_MAX):
            raise ValueError("common-C12 profiles require row cap nine")

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # A failed rank-gain hypothesis must not silently emit k=254.
        if n != 552 or k != 258:
            continue
        assert 425 * n <= 1000 * k < 475 * n

        ar, br = _extract_components(G, hx, hz)
        qa = _kernel_profile(ar, 3 * G.q)
        qb = _kernel_profile(br, 7 * G.q)
        if min(qa[0], qb[0]) < 4:
            continue

        signature = _tanner_signature(hx, hz)
        if signature in graph_keys:
            continue
        # Check mixed-sector words too, not just constituent triangles.
        if not _no_kernel_through_three(hx, n):
            continue
        if not _no_kernel_through_three(hz, n):
            continue
        graph_keys.add(signature)

        if accepted == idx:
            assert css_ok(hx, hz), "verified lift failed CSS"
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) == 9
            assert all(g[1] == 0
                       for grid in (A, B) for row in grid
                       for cell in row for g in cell)
            return {
                "A": A, "B": B, "screen_k": k,
                "component_screens": (qa, qb),
                "centralizer_weights_through_three_excluded": True,
                "design_card": card,
                "rank_gain_above_bound": k - 240,
            }
        accepted += 1

    raise ValueError("distinct k258 common-C12 catalogue exhausted")


def _c31_columns(rows, width):
    """Column syndromes of a supplied verified matrix."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _c31_rows(G, grid, side):
    """Expose exactly the scaffold's left A or right B action."""
    dummy = [[[G.e]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:len(grid) * G.q]]


def _c31_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _c31_outside(word, basis):
    """Exact membership test in the supplied stabilizer span."""
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c31_probe(rng, rows, width, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, never an unresolved cap."""
    cols = _c31_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _c31_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for pass_number in range(passes):
        rng.shuffle(order)
        if own is not None and pass_number % 2:
            # Derive the sector boundary from the supplied complete width.
            ca_cb = A_SHAPE[1] * B_SHAPE[1]
            factor = ca_cb + A_SHAPE[0] * B_SHAPE[0]
            n1 = width * ca_cb // factor
            order.sort(key=lambda j: j < n1)
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

        # Basis exchanges preserve the kernel, including its logical classes.
        rng.shuffle(kernel)
        for _ in range(2):
            for i in range(len(kernel)):
                current = kernel[i]
                w = current.bit_count()
                for j, other in enumerate(kernel):
                    if i == j:
                        continue
                    trial = current ^ other
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        current, w = trial, tw
                kernel[i] = current
                inspect(current)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 18:
                    for h in range(j):
                        inspect(pair ^ short[h])

        if own is None and weight <= 10:
            break
        if own is not None and weight <= 8:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _c31_outside(best, own)
    return best


def _c31_cofactor(rows, count, q):
    """Verified cyclic cofactor witnesses and translated differences.

    Coefficients are extracted from actual column syndromes, not from
    a separately implemented lift. Every resulting word is checked.
    """
    from itertools import combinations
    mask = (1 << q) - 1
    cols = _c31_columns(rows, count * q)
    polys = [[(cols[j * q] >> (i * q)) & mask
              for j in range(count)] for i in range(2)]

    def rotate(poly, shift):
        return poly if not shift else (
            ((poly << shift) | (poly >> (q - shift))) & mask)

    def multiply(a, b):
        result = 0
        while a:
            bit = a & -a
            result ^= rotate(b, bit.bit_length() - 1)
            a ^= bit
        return result

    best = 0

    def inspect(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

    for js in combinations(range(count), 3):
        pieces = [0] * count
        for omitted, j in enumerate(js):
            a, b = js[:omitted] + js[omitted + 1:]
            pieces[j] = (multiply(polys[0][a], polys[1][b])
                         ^ multiply(polys[0][b], polys[1][a]))
        word = sum(poly << (j * q) for j, poly in enumerate(pieces))
        if not word:
            continue
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        inspect(word)
        for shift in range(1, q):
            moved = sum(rotate(poly, shift) << (j * q)
                        for j, poly in enumerate(pieces))
            inspect(word ^ moved)
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _c31_draw(rng, G, profile, policy):
    """Optimize relative gaps rather than emit random gauges as designs."""
    q = G.q
    extent = (3, 5, 9, 13, 15)[policy]
    used = set()
    grid = []
    for counts in profile:
        row = []
        for count in counts:
            if not count:
                row.append([])
                continue
            anchor = rng.randrange(q)
            offsets = [0]
            if count == 2:
                choices = [d for d in range(1, extent + 1)
                           if d not in used]
                gap = rng.choice(choices or list(range(1, extent + 1)))
                used.add(gap)
                offsets.append(gap if rng.randrange(2) else -gap)
            elif count == 3:
                if policy == 0:
                    d = rng.randrange(1, q)
                    offsets = [0, d, 2 * d]
                elif policy in (1, 2, 3):
                    while True:
                        a, b = rng.sample(range(1, q), 2)
                        ds = (a, b, (a - b) % q)
                        if len({min(d, (-d) % q) for d in ds}) == 3:
                            offsets = [0, a, b]
                            break
                else:
                    offsets = [0] + rng.sample(range(1, q), 2)
            cell = [G.elements[(anchor + d) % q] for d in offsets]
            assert len(set(cell)) == count <= 5
            row.append(cell)
        grid.append(row)
    return grid


def _c31_singular_draw(rng, G, profile, policy, root):
    """Sparse evaluation constraints, NOT an alternative matrix lift.

    Use GF32=F2[t]/(t^5+t^2+1). Every nonidentity power of t has
    order 31. Equality of the two polynomial rows at beta imposes
    rank <=1 at that factor; verified constituent ranks test the claim.
    """
    assert G.q == 31 and root in (1, 3)

    def multiply(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 32:
                a ^= 0x25
        return result

    beta = 1
    for _ in range(root):
        beta = multiply(beta, 2)
    powers = [1]
    for _ in range(30):
        powers.append(multiply(powers[-1], beta))
    assert len(set(powers)) == 31
    assert multiply(powers[-1], beta) == 1
    logarithm = {value: exponent for exponent, value in enumerate(powers)}

    extent = (5, 8, 11, 14, 15)[policy]
    used_gaps = set()
    grid = [[[] for _ in profile[0]] for _ in range(2)]
    for j in range(len(profile[0])):
        counts = (profile[0][j], profile[1][j])
        assert sum(counts) == 3
        offsets, values = [], []
        for count in counts:
            if count == 0:
                terms = []
            elif count == 1:
                terms = [0]
            elif count == 2:
                choices = [d for d in range(1, extent + 1)
                           if d not in used_gaps]
                d = rng.choice(choices or list(range(1, extent + 1)))
                used_gaps.add(d)
                terms = [0, d if rng.randrange(2) else (-d) % 31]
            elif count == 3:
                # beta^a + beta^b = 1: three distinct monomials.
                a = rng.randrange(1, 31)
                b = logarithm[1 ^ powers[a]]
                terms = [0, a, b]
                assert len(set(terms)) == 3
            else:
                raise ValueError("spectral card is not cubic")
            value = 0
            for exponent in terms:
                value ^= powers[exponent]
            offsets.append(terms)
            values.append(value)

        anchors = [rng.randrange(31), rng.randrange(31)]
        if counts[0] and counts[1]:
            assert values[0] and values[1]
            anchors[1] = (
                anchors[0] + logarithm[values[0]]
                - logarithm[values[1]]) % 31
        else:
            assert values == [0, 0]

        evaluations = []
        for i in range(2):
            labels = [(anchors[i] + d) % 31 for d in offsets[i]]
            grid[i][j] = [G.elements[label] for label in sorted(labels)]
            assert len(set(labels)) == counts[i]
            value = 0
            for label in labels:
                value ^= powers[label]
            evaluations.append(value)
        assert evaluations[0] == evaluations[1]
    return grid


def _c31_component_bank(G, side):
    """Optimize deliberate full-rank and split-factor constituent cards."""
    if side == "A":
        profiles = (
            ((0, 2, 2), (3, 1, 1)),  # augmentation rank one
            ((1, 1, 2), (2, 2, 1)),  # augmentation rank two
        )
    else:
        profiles = (
            ((1, 1, 2, 2), (2, 2, 1, 1)),
            ((0, 1, 2, 3), (3, 2, 1, 0)),
        )

    bank, component_keys = [], set()
    spectral_cards = 5 * len(profiles)
    fallback_cards = 5 if side == "A" else spectral_cards
    for card in range(spectral_cards + fallback_cards):
        forced = card < spectral_cards
        pi, policy = divmod(card % spectral_cards, 5)
        profile = profiles[pi]
        augmentation = gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in profile
        ])
        expected_rank = 2 * (G.q - 1) + augmentation - (5 if forced else 0)
        local = random.Random(
            BASE_SEED ^ (0xC315F202 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        width = len(profile[0]) * G.q
        raw, fillings = [], set()
        for trial in range(160):
            grid = (_c31_singular_draw(
                local, G, profile, policy, 1 if side == "A" else 3)
                    if forced else _c31_draw(local, G, profile, policy))
            filling = tuple(tuple(sorted(G.idx(g) for g in c))
                            for row in grid for c in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _c31_rows(G, grid, side)
            rank = gf2_rank(rows)
            # Exactly one degree-five loss for spectral cards; no
            # unmeasured assumption about other factors or augmentation.
            if rank != expected_rank:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            seed = _c31_cofactor(rows, len(profile[0]), G.q)
            if not seed or seed.bit_count() < 12:
                continue
            penalty = sum(
                ((rows[i] & rows[j]).bit_count()
                 * ((rows[i] & rows[j]).bit_count() - 1) // 2)
                for i in range(len(rows)) for j in range(i))
            raw.append((seed.bit_count(), penalty, trial,
                        grid, rows, seed, rank))

        raw.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, penalty, trial, grid, rows, seed, rank in raw[:16]:
            word = _c31_probe(local, rows, width, 6, seed)
            refined.append((word.bit_count(), penalty, trial,
                            grid, rows, word, rank))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))

        kept = 0
        for _, penalty, trial, grid, rows, seed, rank in refined[:8]:
            word = _c31_probe(local, rows, width, 32, seed)
            if not word or word.bit_count() < 12:
                continue
            key = _tanner_signature(rows, [])
            if key in component_keys:
                continue
            component_keys.add(key)
            bank.append({
                "grid": grid, "word": word,
                "reading": word.bit_count(), "rank": rank,
                "penalty": penalty, "card": card, "trial": trial,
                "spectral_root": (1 if side == "A" else 3) if forced else None,
                "augmentation_rank": augmentation,
            })
            kept += 1
            if kept == 5:
                break

    bank.sort(key=lambda r: (
        -r["reading"], r["penalty"], r["card"], r["trial"]))
    return bank


def _c31_logical_reading(G, a, b, hx, hz, n, rng):
    """Actual nontrivial witnesses, reduced by legal stabilizer moves."""
    q = G.q
    best = n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _c31_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        blocks = A_SHAPE[1] if kind == "X" else B_SHAPE[1]
        seeds = []
        for block in range(blocks):
            if kind == "X":
                word = source << (block * B_SHAPE[1] * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * B_SHAPE[1] + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if _c31_outside(word, own):
                seeds.append(word)
        if not seeds:
            return None

        reduced = []
        for word in seeds:
            for _ in range(3):
                previous = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:64]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == previous:
                    break
            assert word and _c31_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            reduced.append(word)

        seed = min(reduced, key=int.bit_count)
        word = _c31_probe(rng, checks, n, 2, seed, own)
        if not word:
            return None
        best = min(best, word.bit_count())
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _c31_portfolio(G, count):
    """Deterministic immutable portfolio; slot call order changes nothing."""
    if G.q != 31:
        raise ValueError("C31 cubic portfolio requires order 31")
    aa = _c31_component_bank(G, "A")
    bb = _c31_component_bank(G, "B")
    if not aa or not bb:
        raise ValueError("no screened C31 constituent bank")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["reading"], b["reading"])
            # A and B spectral roots occupy disjoint Frobenius orbits.
            # For these exact-rank cards, their separate homology gains
            # suggest this ordering. Complete verified ranks still decide.
            deficits = (62 - a["rank"], 62 - b["rank"])
            predicted_k = 62 + 2 * deficits[0] + deficits[1]
            priority = (
                predicted_k * weak * weak, weak, predicted_k,
                a["reading"] + b["reading"],
                -a["penalty"] - b["penalty"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    graph_keys, results = set(), []
    cap = min(9, 12, W_MAX)
    tested = 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert wx <= 8 and wz <= 9
        assert _no_dead_lines(sa, sb)
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 496 and k >= 67
                and 125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        columns = _c31_columns(hx + hz, n)
        key = (_tanner_signature(hx, hz),
               tuple(sorted(c.bit_count() for c in columns)))
        if key in graph_keys:
            continue
        graph_keys.add(key)
        tested += 1

        # These are exact complete-kernel gates, including both sectors.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified C31 lift failed CSS")

        local = random.Random(
            BASE_SEED ^ (0xC31CCF202 + 65537 * ai + 1009 * bi))
        reading = _c31_logical_reading(G, a, b, hx, hz, n, local)
        if reading is None or reading < 12:
            continue
        if lifted_k(G, A, B) != k:
            raise ValueError("verified C31 rank helpers disagree")
        quality = (k * reading * reading / n, reading, k, preliminary)
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("design_cards", (a["card"], b["card"])),
            ("component_ranks", (a["rank"], b["rank"])),
            ("spectral_roots", (a["spectral_root"], b["spectral_root"])),
            ("augmentation_ranks", (
                a["augmentation_rank"], b["augmentation_rank"])),
            ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("complete_kernel_exclusion_through", 4),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + 40:
            break
        if tested >= max(480, 12 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def _s35_profiles():
    """Four genuinely different support budgets, not permutation copies."""
    regular = [[1] * 5 for _ in range(3)]
    reinforced = [row[:] for row in regular]
    reinforced[0][0] = 2
    row_balanced = [row[:] for row in regular]
    row_balanced[0][0], row_balanced[0][1] = 0, 2
    column_balanced = [row[:] for row in regular]
    column_balanced[0][0], column_balanced[1][0] = 0, 2
    return tuple(tuple(tuple(row) for row in grid)
                 for grid in (regular, reinforced,
                              row_balanced, column_balanced))


def _s35_draw(rng, G, profile, policy):
    """Gauge-normalized voltages with separated row-pair differences."""
    q = G.q
    quadratic = sorted({a * a % q for a in range(1, q)})
    first_pool = (quadratic if policy == 1 else
                  list(range(1, 10)) if policy == 3 else
                  list(range(1, q)))
    a = [0] + rng.sample(first_pool, 4)
    b, used, differences = [0], {0}, {0}

    for j in range(1, 5):
        pool = (quadratic if policy == 2 else
                list(range(1, 10)) if policy == 3 else
                list(range(1, q)))
        choices = [v for v in pool
                   if v not in used and (v - a[j]) % q not in differences]
        if not choices:
            choices = [v for v in range(1, q)
                       if v not in used
                       and (v - a[j]) % q not in differences]
        if not choices:
            return None
        v = rng.choice(choices)
        b.append(v)
        used.add(v)
        differences.add((v - a[j]) % q)

    anchors = ([0] * 5, a, b)
    extent = (8, 4, 8, 6, 8)[policy]
    grid = []
    for i, counts in enumerate(profile):
        row, gaps = [], set()
        for j, count in enumerate(counts):
            if not count:
                row.append([])
                continue
            anchor = anchors[i][j]
            labels = [anchor]
            if count == 2:
                choices = [d for d in range(1, extent + 1) if d not in gaps]
                gap = rng.choice(choices or list(range(1, extent + 1)))
                gaps.add(gap)
                if policy == 4 or rng.randrange(2):
                    gap = -gap
                labels.append((anchor + gap) % q)
            cell = [G.elements[v] for v in sorted(labels)]
            assert len(set(cell)) == count <= CELL_MAX
            row.append(cell)
        grid.append(row)
    return grid


def _s35_cofactor(rows, count, q):
    """Three-row cyclic minors extracted ONLY from verified syndromes.

    Every nonzero minor word is verified. Translations and cross-minor
    sums are actual kernel witnesses, never distance lower bounds.
    """
    from itertools import combinations, permutations
    mask = (1 << q) - 1
    cols = _c31_columns(rows, count * q)
    polys = [[(cols[j * q] >> (i * q)) & mask
              for j in range(count)] for i in range(3)]

    def rotate(poly, shift):
        return poly if not shift else (
            ((poly << shift) | (poly >> (q - shift))) & mask)

    def multiply(a, b):
        value = 0
        while a:
            bit = a & -a
            value ^= rotate(b, bit.bit_length() - 1)
            a ^= bit
        return value

    minors = {}
    for js in combinations(range(count), 3):
        value = 0
        for perm in permutations(js):
            value ^= multiply(
                multiply(polys[0][perm[0]], polys[1][perm[1]]),
                polys[2][perm[2]])
        minors[js] = value

    orbits, best = [], 0

    def inspect(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

    for js in combinations(range(count), 4):
        pieces = [0] * count
        for omitted, j in enumerate(js):
            pieces[j] = minors[js[:omitted] + js[omitted + 1:]]
        word = sum(poly << (j * q) for j, poly in enumerate(pieces))
        if not word:
            continue
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        orbit = [
            sum(rotate(poly, shift) << (j * q)
                for j, poly in enumerate(pieces))
            for shift in range(q)
        ]
        inspect(word)
        for moved in orbit[1:]:
            inspect(word ^ moved)
        orbits.append(orbit)

    for i, first in enumerate(orbits):
        for second in orbits[:i]:
            for moved in second:
                inspect(first[0] ^ moved)

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _s35_fingerprint(rows, width, q):
    """Rooted uncoloured Tanner invariant, with verified orbit reduction.

    Check nodes have no X/Z labels. Roots are selected by intrinsic
    qubit degrees. Unequal fingerprints imply inequivalence; equality
    is conservatively discarded and is not an isomorphism assertion.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(rows)
    columns = _c31_columns(rows, width)
    m = len(rows)

    def positions(word):
        result = []
        while word:
            bit = word & -word
            result.append(bit.bit_length() - 1)
            word ^= bit
        return result

    adjacency = [[m + j for j in positions(row)] for row in rows]
    adjacency += [positions(column) for column in columns]
    degrees = [column.bit_count() for column in columns]
    highest = max(degrees)
    high = [j for j, degree in enumerate(degrees) if degree == highest]
    mask = (1 << q) - 1

    def translate(row):
        moved = 0
        for block in range(width // q):
            value = (row >> (block * q)) & mask
            value = ((value << 1) | (value >> (q - 1))) & mask
            moved |= value << (block * q)
        return moved

    symmetric = (
        width % q == 0
        and Counter(translate(row) for row in rows) == Counter(rows))
    roots = ([j for j in high if j % q == 0] if symmetric else high)

    def compress(features):
        palette = {feature: i
                   for i, feature in enumerate(sorted(set(features)))}
        return [palette[feature] for feature in features]

    initial = [(0, len(neighbours)) for neighbours in adjacency[:m]]
    initial += [(1, len(neighbours)) for neighbours in adjacency[m:]]
    rooted = []
    for root in roots:
        colours = compress([
            (tag, int(v == m + root)) for v, tag in enumerate(initial)])
        history = []
        for _ in range(7):
            features = [
                (colours[v], tuple(sorted(colours[w] for w in neighbours)))
                for v, neighbours in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            colours = compress(features)
        digest = sha256(repr(history).encode("ascii")).digest()
        rooted.extend([digest] * (q if symmetric else 1))

    return (
        width, m,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(degrees).items())),
        tuple(sorted(rooted)),
    )


def _s35_bank(G, side):
    """Twenty support/geometry prescriptions with equal refinement budgets."""
    profiles = _s35_profiles()
    records, component_keys = [], set()
    width = 5 * G.q

    for card in range(20):
        pi, policy = divmod(card, 5)
        profile = profiles[pi]
        augmentation = gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in profile])
        # ord_17(2)=8. This rank requires both nontrivial factors
        # to retain all three constituent rows.
        expected_rank = 3 * (G.q - 1) + augmentation
        local = random.Random(
            BASE_SEED ^ (0x17351503 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        raw, fillings = [], set()

        for trial in range(128):
            grid = _s35_draw(local, G, profile, policy)
            if grid is None:
                continue
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _c31_rows(G, grid, side)
            rank = gf2_rank(rows)
            if rank != expected_rank:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            seed = _s35_cofactor(rows, 5, G.q)
            if not seed or seed.bit_count() < 10:
                continue
            penalty = sum(
                overlap * (overlap - 1) // 2
                for i, row in enumerate(rows)
                for other in rows[:i]
                for overlap in [(row & other).bit_count()])
            raw.append((seed.bit_count(), penalty, trial,
                        grid, rows, seed, rank))

        raw.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, penalty, trial, grid, rows, seed, rank in raw[:16]:
            word = _c31_probe(local, rows, width, 6, seed)
            if word and word.bit_count() >= 10:
                refined.append((word.bit_count(), penalty, trial,
                                grid, rows, word, rank))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))

        finalists = []
        for _, penalty, trial, grid, rows, seed, rank in refined[:8]:
            word = _c31_probe(local, rows, width, 32, seed)
            if word and word.bit_count() >= 10:
                finalists.append((word.bit_count(), penalty, trial,
                                  grid, rows, word, rank))
        finalists.sort(key=lambda r: (-r[0], r[1], r[2]))

        kept = 0
        for reading, penalty, trial, grid, rows, word, rank in finalists:
            key = _s35_fingerprint(rows, width, G.q)
            if key in component_keys:
                continue
            component_keys.add(key)
            records.append(dict(
                grid=grid, word=word, reading=reading, rank=rank,
                augmentation=augmentation, penalty=penalty,
                card=card, trial=trial))
            kept += 1
            if kept == 5:
                break

    records.sort(key=lambda r: (
        -r["reading"], r["penalty"], r["card"], r["trial"]))
    return records


@lru_cache(maxsize=1)
def _s35_portfolio(G, count):
    """Immutable, complete-matrix-screened three-row portfolio.

    Historical C31 policies remain uncalled. Neither an exhausted
    catalogue nor a failed rate gate is repaired by recycling a code.
    """
    assert G.q == 17 and A_SHAPE == B_SHAPE == (3, 5)
    aa, bb = _s35_bank(G, "A"), _s35_bank(G, "B")
    if not aa or not bb:
        raise ValueError("C17 three-row constituent bank exhausted")

    cap = min(9, W_MAX)
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            sa = [[len(cell) for cell in row] for row in a["grid"]]
            sb = [[len(cell) for cell in row] for row in b["grid"]]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            ar, br = a["augmentation"], b["augmentation"]
            # Only an ordering calculation. Complete lifted ranks
            # determine the published dimension.
            predicted = (
                (G.q - 1) * 4
                + (5 - ar) * (5 - br) + (3 - ar) * (3 - br))
            if not 125 * 578 <= 1000 * predicted < 175 * 578:
                continue
            weak = min(a["reading"], b["reading"])
            priority = (
                predicted * weak * weak, weak, predicted,
                a["reading"] + b["reading"],
                -a["penalty"] - b["penalty"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    graph_keys, results = set(), []
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sum(sa[i][j] for i in range(3))
                 for row in sb for j in range(5))
        assert max(wx, wz) == _grid_weights(sa, sb)
        assert _no_dead_lines(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 578 and k > 0
                and 125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Complete centralizers include sector-2 and mixed-sector words.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified C17 three-row lift failed CSS")

        key = _s35_fingerprint(hx + hz, n, G.q)
        if key in graph_keys:
            continue
        graph_keys.add(key)
        tested += 1

        local = random.Random(
            BASE_SEED ^ (0x1735CC03 + 65537 * ai + 1009 * bi))
        # This existing embedding/descent helper uses dynamic shapes;
        # its information-set sector boundary has been made size-derived.
        reading = _c31_logical_reading(G, a, b, hx, hz, n, local)
        if reading is not None and reading >= 10:
            if lifted_k(G, A, B) != k:
                raise ValueError("verified C17 rank helpers disagree")
            quality = (k * reading * reading / n, reading, k, preliminary)
            screen = (
                ("k", k),
                ("logical_witness_upper_bound", reading),
                ("component_witness_readings",
                 (a["reading"], b["reading"])),
                ("component_ranks", (a["rank"], b["rank"])),
                ("augmentation_ranks",
                 (a["augmentation"], b["augmentation"])),
                ("design_cards", (a["card"], b["card"])),
                ("row_weight_maxima", (wx, wz)),
                ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("complete_kernel_exclusion_through", 4),
            )
            results.append((quality, freeze(A), freeze(B), screen))

        if len(results) >= count + 24:
            break
        if tested >= max(480, 12 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete, novel-signature three-row design."""
    assert G.q == 17 and 0 <= idx < N_SLOTS
    portfolio = _s35_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("target-rate C17 three-row portfolio exhausted")
    A, B, screen = portfolio[idx]
    screen = dict(screen)
    assert 125 * 578 <= 1000 * screen["k"] < 175 * 578
    return {"A": A, "B": B, "screen": screen}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell without resampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty evaluated slots retained only five classes, exhausted after slot 51, and all five measured leaders had d30s=4 with abundant sector-1 witnesses.
# REFUTED: The C31 cubic portfolio's intrinsic-signature filter alone supplies sixty inequivalent designs: the measured parent retained only 28 distinct codes, while its five leaders remained at k=64,d30s=14.
# REFUTED: The split-GF32 C31 portfolio's intrinsic-signature filtering supplies sixty inequivalent designs: the current sixty-slot table retained only 19 classes, with 41 iso_dup statuses and five measured k72,d30s14 leaders.
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
