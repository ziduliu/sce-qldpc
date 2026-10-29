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
CONCEPT = ("C4 wreath C2, order 32, A=(2,3), B=(2,4): optimizing noncommuting constituents and order-eight matching cycles should supply five fresh cell-0.150 codes with measured distance readings at least sixteen and fitness above 28.452; distance survival is a falsifiable hypothesis.")
GROUP = ("Jump from abelian C31 to (C4 x C4) semidirect C2, where C2 swaps the coordinates. This nonabelian order-32 group has centre C4, and no order-32 centre-four bed appears in BEDS. Related order-32 nonabelian beds reached 30.8 on equal shapes but only 21.4 on these shapes, motivating a different group and filling mechanism rather than transferred distance. The binary group algebra of a 2-group is local: augmentation-full-rank cards control constituent rank. Verified complete lifts decide every published k.")
SHAPES = ("Keep A=(2,3), B=(2,4), now q=32: n=512, kb=64 and guaranteed rate 1/8. The shape permits q<=37. Unequal constituents have 96 and 128 variables, each with 64 checks. Parent measured weight-fourteen X/Z minima occupied sector-1 Acol0..2/Bcol0..3, especially Acol1 for X, without reported sector-2 minima. Actual full ranks restrict every published code to [0.125,0.175), targeting cell 0.150.")
FILLING = ("Twenty optimized prescriptions per side vary balanced/exclusive cubic A supports, cubic/irregular B supports, swap-coset allocations and relative-element orders. A rows total four/five, B rows six, A columns three and B columns at most four, so BOTH CSS row-weight sums are at most nine, below twelve. The two-matching B branch fixes relative order eight, excluding its shorter pure matching cycles. Normal-C4-squared and central-C4 invariant kernels supply verified seeds; equal-budget information-set refinement selects constituent banks. Full verified lifts enforce actual rate, CSS, kernel-four exclusion and explicit embedded/mixed-sector logical witnesses. Rooted Tanner invariants use central-orbit reduction only after checking the actual matrix symmetry, conservatively rejecting equivalent outputs. Witness readings are capped at eighteen for ordering, not certified as distances; unseen-archive novelty remains a hypothesis.")
FAMILY_NAME = "r015_c4wrc2_q32_23x24_order8_noncommuting_jointwitness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C4 wreath C2 has order 32")
    # (a,c,b): two C4 coordinates followed by the swapping C2.
    els = [(a, c, b) for b in range(2)
           for a in range(4) for c in range(4)]

    def red(g):
        return (int(g[0]) % 4, int(g[1]) % 4, int(g[2]) % 2)

    def mul(g, h):
        a, c, b = red(g)
        u, v, d = red(h)
        return ((a + (v if b else u)) % 4,
                (c + (u if b else v)) % 4,
                (b + d) % 2)

    def inv(g):
        a, c, b = red(g)
        return ((-c if b else -a) % 4,
                (-a if b else -c) % 4, b)

    def idx(g):
        a, c, b = red(g)
        return 16 * b + 4 * a + c

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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
            # Derive the sector boundary from the complete matrix width.
            ra, ca = A_SHAPE
            rb, cb = B_SHAPE
            n1 = width * ca * cb // (ca * cb + ra * rb)
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

        # Refine the incumbent using this information set's short words.
        # Every move remains inside the verified kernel; quotient tests
        # prevent a logical incumbent from descending into a stabilizer.
        for _ in range(3):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

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


def _w32_order(G, g):
    value = G.e
    for length in range(1, G.q + 1):
        value = G.mul(value, g)
        if value == G.e:
            return length
    raise ValueError("invalid wreath-product element")


def _w32_draw(local, G, side, card):
    """Twenty construction prescriptions, optimized before publication."""
    if side == "A":
        profiles = (
            ((1, 1, 2), (2, 2, 1)),
            ((0, 2, 2), (3, 1, 1)),
        )
        profile = profiles[card // 10]
        mode, variant = divmod(card % 10, 2)
    else:
        profiles = (
            ((1, 1, 2, 2), (2, 2, 1, 1)),
            ((0, 1, 2, 3), (3, 2, 1, 0)),
            ((1, 1, 3, 1), (3, 2, 0, 1)),
            ((1, 3, 1, 1), (3, 1, 1, 1)),
        )
        profile = profiles[card // 5]
        mode, variant = card % 5, (card // 5) % 2

    rotations = [
        g for g in G.elements
        if g[2] == 0 and g[0] != g[1] and _w32_order(G, g) == 4]
    outside8 = [
        g for g in G.elements if g[2] and _w32_order(G, g) == 8]
    outside4 = [
        g for g in G.elements if g[2] and _w32_order(G, g) == 4]
    axes = [
        g for g in rotations
        if g[1 - variant] == 0 and g[variant] % 2]
    broad = rotations + outside8 + outside4
    used = set()

    def choose(pool):
        options = [
            g for g in pool
            if min(G.idx(g), G.idx(G.inv(g))) not in used]
        g = local.choice(options or pool)
        used.add(min(G.idx(g), G.idx(G.inv(g))))
        return g

    grid = []
    for i, counts in enumerate(profile):
        row = []
        for j, count in enumerate(counts):
            if not count:
                row.append([])
                continue
            anchor = G.pick(local)
            if count == 1:
                relative = [G.e]
            elif count == 2:
                if mode == 0:
                    pool = outside8
                elif mode == 1:
                    pool = outside8 if (i + j + variant) % 2 else rotations
                elif mode == 2:
                    pool = outside4 if (i + j) % 2 else outside8
                elif mode == 3:
                    pool = axes if i == variant else outside8
                else:
                    pool = broad
                relative = [G.e, choose(pool)]
            else:
                assert count == 3
                # Odd augmentation makes these trinomials units in
                # F2[G]; noncommutation is an independent design choice.
                for _ in range(128):
                    if mode == 2:
                        g, h = local.choice(outside8), local.choice(outside4)
                    elif mode == 4:
                        g, h = local.sample(broad, 2)
                    else:
                        g = local.choice(axes if mode == 3 else rotations)
                        h = local.choice(outside8)
                    if g != h and G.mul(g, h) != G.mul(h, g):
                        relative = [G.e, g, h]
                        break
                else:
                    raise ValueError("noncommuting trinomial unavailable")
            cell = [G.mul(anchor, g) for g in relative]
            assert len(set(cell)) == count <= 5
            row.append(cell)
        grid.append(row)

    if side == "B" and card // 5 == 3:
        # Two matching columns have a cycle of length 2*ord(g)=16.
        # They are not allowed to inherit an order-two/four ratio.
        g = local.choice(outside8)
        grid[0][2], grid[1][2] = [G.e], [G.e]
        grid[0][3], grid[1][3] = [G.e], [g]
        assert _w32_order(G, g) == 8
    return grid


def _w32_orbit_seed(G, rows, width, local):
    """Actual kernels in two subgroup-invariant subspaces.

    Atom syndromes come from the scaffold constituent, not another lift.
    Both subgroups are normal, so either constituent action preserves
    their coset-constant subspace. The resulting witness is checked.
    """
    cols = _c31_columns(rows, width)
    best = 0
    subgroups = (
        [(a, c, 0) for a in range(4) for c in range(4)],
        [(t, t, 0) for t in range(4)],
    )
    for subgroup in subgroups:
        packets, covered = [], set()
        for g in G.elements:
            if g in covered:
                continue
            packet = {G.mul(g, h) for h in subgroup}
            assert len(packet) == len(subgroup)
            covered.update(packet)
            packets.append(tuple(sorted(packet, key=G.idx)))
        assert len(covered) == G.q

        atoms, syndromes = [], []
        for block in range(width // G.q):
            for packet in packets:
                word = syndrome = 0
                for g in packet:
                    j = block * G.q + G.idx(g)
                    word |= 1 << j
                    syndrome ^= cols[j]
                atoms.append(word)
                syndromes.append(syndrome)

        compressed_rows = [0] * len(rows)
        for j, syndrome in enumerate(syndromes):
            while syndrome:
                bit = syndrome & -syndrome
                compressed_rows[bit.bit_length() - 1] |= 1 << j
                syndrome ^= bit
        compressed = _c31_probe(
            local, compressed_rows, len(atoms), 8)
        if not compressed:
            continue
        word = 0
        while compressed:
            bit = compressed & -compressed
            word ^= atoms[bit.bit_length() - 1]
            compressed ^= bit
        assert word
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        if not best or word.bit_count() < best.bit_count():
            best = word
    return best


def _w32_cost(rows):
    """Short-incidence-cycle burden, not a distance estimate."""
    return sum(
        overlap * (overlap - 1) // 2
        for i, row in enumerate(rows)
        for other in rows[:i]
        for overlap in [(row & other).bit_count()])


def _w32_key(hx, hz, width, G):
    """Intrinsic rooted Tanner invariant; equal keys are inconclusive.

    X/Z checks are pooled. Central orbit reduction is used only after
    verifying that its qubit permutation preserves the actual row
    multiset. Root multiplicities are retained.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _c31_columns(rows, width)
    m, q = len(rows), G.q

    def bits(word):
        result = []
        while word:
            bit = word & -word
            result.append(bit.bit_length() - 1)
            word ^= bit
        return result

    adjacency = [[m + j for j in bits(row)] for row in rows]
    adjacency += [bits(col) for col in cols]
    highest = max(col.bit_count() for col in cols)
    roots = [j for j, col in enumerate(cols)
             if col.bit_count() == highest]

    z = (1, 1, 0)
    permutation = [G.idx(G.mul(g, z)) for g in G.elements]

    def move_index(j):
        block, h = divmod(j, q)
        return block * q + permutation[h]

    def move_word(word):
        moved = 0
        while word:
            bit = word & -word
            moved |= 1 << move_index(bit.bit_length() - 1)
            word ^= bit
        return moved

    symmetric = Counter(map(move_word, rows)) == Counter(rows)
    rooted_plans = []
    if symmetric:
        covered = set()
        root_set = set(roots)
        for root in roots:
            if root in covered:
                continue
            orbit, following = set(), root
            while following not in orbit:
                orbit.add(following)
                following = move_index(following)
            assert orbit <= root_set and len(orbit) == 4
            covered.update(orbit)
            rooted_plans.append((root, len(orbit)))
        assert covered == root_set
    else:
        rooted_plans = [(root, 1) for root in roots]

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    initial = [(0, len(ns)) for ns in adjacency[:m]]
    initial += [(1, len(ns)) for ns in adjacency[m:]]
    rooted = []
    for root, multiplicity in rooted_plans:
        colours = compress([
            (tag, int(v == m + root))
            for v, tag in enumerate(initial)])
        history = []
        for _ in range(5):
            features = [
                (colours[v], tuple(sorted(colours[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            colours = compress(features)
        digest = sha256(repr(history).encode("ascii")).digest()
        rooted.extend([digest] * multiplicity)

    return (
        width,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        tuple(sorted(rooted)),
        _tanner_signature(rows, []),
    )


def _w32_bank(G, side):
    """Equal-budget optimization of twenty relative-element prescriptions."""
    width = (3 if side == "A" else 4) * G.q
    bank, component_keys, fillings = [], set(), set()
    for card in range(20):
        raw = []
        for trial in range(160):
            local = random.Random(
                BASE_SEED ^ (0x432B710 + 104729 * card + 7919 * trial
                             + 1000003 * int(side == "B")))
            grid = _w32_draw(local, G, side, card)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _c31_rows(G, grid, side)
            rank = gf2_rank(rows)
            if rank < 56:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            raw.append((_w32_cost(rows), trial, grid, rows, rank))
        raw.sort(key=lambda r: r[:2])

        refined = []
        for cost, trial, grid, rows, rank in raw[:20]:
            local = random.Random(
                BASE_SEED ^ (0x432B711 + 65537 * card + 257 * trial
                             + 1000003 * int(side == "B")))
            seed = _w32_orbit_seed(G, rows, width, local)
            word = _c31_probe(local, rows, width, 4, seed)
            if word and word.bit_count() >= 12:
                refined.append((min(word.bit_count(), 20), cost, trial,
                                grid, rows, word, rank))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))

        finalists = []
        for _, cost, trial, grid, rows, seed, rank in refined[:8]:
            local = random.Random(
                BASE_SEED ^ (0x432B712 + 65537 * card + 257 * trial
                             + 1000003 * int(side == "B")))
            word = _c31_probe(local, rows, width, 28, seed)
            if word and word.bit_count() >= 12:
                finalists.append((word.bit_count(), cost, trial,
                                  grid, rows, word, rank))
        finalists.sort(key=lambda r: (-min(r[0], 20), r[1], r[2]))

        kept = 0
        for reading, cost, trial, grid, rows, word, rank in finalists:
            key = _w32_key(rows, [], width, G)
            if key in component_keys:
                continue
            component_keys.add(key)
            bank.append(dict(
                grid=grid, word=word, reading=reading, rank=rank,
                penalty=cost, card=card, trial=trial))
            kept += 1
            if kept == 6:
                break
    bank.sort(key=lambda r: (
        -min(r["reading"], 20), r["penalty"], r["card"], r["trial"]))
    return bank


@lru_cache(maxsize=1)
def _w32_portfolio(G, count):
    """Immutable complete-code portfolio, independent of slot call order."""
    assert G.q == 32 and A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    aa, bb = _w32_bank(G, "A"), _w32_bank(G, "B")
    if not aa or not bb:
        raise ValueError("wreath-product constituent bank exhausted")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            # Deficits are only a tie-break here. No cyclic spectral
            # formula is used to predict the nonabelian complete k.
            priority = (
                min(a["reading"], b["reading"], 18),
                min(a["reading"], 20) + min(b["reading"], 20),
                -a["penalty"] - b["penalty"],
                128 - a["rank"] - b["rank"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    cap = min(9, 12, W_MAX)
    results, graph_keys = [], set()
    tested = 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        adegrees = [sum(sa[i][j] for i in range(2)) for j in range(3)]
        bdegrees = [sum(sb[i][j] for i in range(2)) for j in range(4)]
        assert adegrees == [3, 3, 3]
        assert sorted(map(sum, sa)) == [4, 5]
        assert list(map(sum, sb)) == [6, 6]
        assert max(bdegrees) <= 4
        wx = max(sum(row) + degree for row in sa for degree in bdegrees)
        wz = max(sum(row) + degree for row in sb for degree in adegrees)
        assert wx <= 9 and wz == 9
        assert _no_dead_lines(sa, sb)
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            continue
        assert all(len(set(cell)) == len(cell) <= 5
                   for grid in (A, B) for row in grid for cell in row)

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 512 and k >= 64
                and 125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if (_kernel_profile(hx, n)[0] != 5
                or _kernel_profile(hz, n)[0] != 5):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified wreath-product lift failed CSS")

        key = _w32_key(hx, hz, n, G)
        if key in graph_keys:
            continue
        tested += 1
        local = random.Random(
            BASE_SEED ^ (0x432B713 + 65537 * ai + 1009 * bi))
        # Both sources are witnesses of the actual scaffold actions.
        # This helper checks each embedding and its logical quotient.
        reading = _c31_logical_reading(G, a, b, hx, hz, n, local)
        if reading is not None and reading >= 12:
            assert lifted_k(G, A, B) == k
            graph_keys.add(key)
            capped = min(reading, 18)
            quality = (k * capped * capped / n, capped, k, preliminary)
            screen = (
                ("k", k),
                ("logical_witness_upper_bound", reading),
                ("component_witness_readings",
                 (a["reading"], b["reading"])),
                ("component_ranks", (a["rank"], b["rank"])),
                ("design_cards", (a["card"], b["card"])),
                ("B_column_degrees", tuple(bdegrees)),
                ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("complete_kernel_exclusion_through", 4),
            )
            results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + max(32, count // 2):
            break
        if tested >= max(960, 16 * count) and len(results) >= count:
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Publish a complete optimized wreath-product design without recycling."""
    assert G.q == 32 and 0 <= idx < N_SLOTS
    portfolio = _w32_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("fresh target-rate wreath-product portfolio exhausted")
    A, B, screen = portfolio[idx]
    screen = dict(screen)
    assert 125 * 512 <= 1000 * screen["k"] < 175 * 512
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
# REFUTED: Increasing split-GF32 rank deficits preserves weight-fourteen readings: the current k=74 candidates instead had d2s=12, while all five measured leaders remained k=72,d30s=14 and sixty slots retained only nineteen classes.
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
