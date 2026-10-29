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
CONCEPT = ("Quaternion Q8, order 8, A=B=(3,8): optimizing noncommuting grounded constituents for exclusion of weight-six kernel words, then screening actual full-code logical witnesses, should supply five novel finalists exceeding the 0.350 frontier record 12.3290. Distance-six fallbacks preserve a measurable test of the stronger hypothesis.")
GROUP = ("Jump from the order-24 C12:C2(action 5) parent to quaternion Q8, with x^4=1, y^2=x^2 and y*x*y^-1=x^-1. The different order proves this is not a relabeling. Three noncentral inverse classes provide noncommuting order-four directions. The cyclic order-eight (3,8)x(3,8) bed measured distance six; anonymous order-eight beds do not identify this exact group/support combination. No distance guarantee transfers.")
SHAPES = ("Use equal (3,8) constituents, each with 64 variables and 24 checks. Factor 73 permits q<=8, giving n=584, kb=200 and guaranteed rate 25/73=0.342466 in cell 0.350. This targets a weaker frontier cell instead of the parent's 0.450 plateau. Parent weight-four minima broadly occupied sector-1 Acol0..2/Bcol0..6; the new bed changes both constituents and increases sector 2 from 48 to 72 qubits. Verified full-rank constituents and complete ranks require k=200.")
FILLING = ("Twenty canonical count cards have row totals six and column degrees 1,2,2,2,2,3,3,3. Relative quaternion labels avoid involutory binomial gaps and optimize exact constituent exclusion through five plus weight-six counts on BOTH verified actions. Heterogeneous pairs receive verified rank, CSS, complete-kernel-four and explicit non-stabilizer logical-witness screens. Rooted intrinsic high-qubit/corner invariants conservatively reject equivalent submitted graphs. Both CSS maxima are <=9, also below twelve. Larger readings remain upper bounds; distance-seven survival and novelty against the unseen archive are hypotheses.")
FAMILY_NAME = "r035_quaternion8_38x38_grounded_kernel6_logicalportfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 has order eight")
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1; element x^a*y^b.
    els = [(a, b) for b in range(2) for a in range(4)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 2 * (b // 2)) % 4, b % 2)

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


def _unused_common_C12_reference(rng, G, idx):
    """Historical common-C12 policy, retained but never published.

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


from functools import lru_cache


@lru_cache(maxsize=1)
def _relative_support_portfolio(G, count):
    """Pure deterministic portfolio; memoized outputs are immutable.

    Only verified scaffold lifts supply matrices for screening.
    No ownership buckets, recycled entries, or cross-slot mutation.
    """
    from itertools import combinations, combinations_with_replacement
    from collections import Counter
    from hashlib import sha256

    q, e, cap = G.q, G.e, min(9, W_MAX)
    local = random.Random(BASE_SEED ^ 0x248045D4)
    if q != 24:
        raise ValueError("relative subgroup portfolio requires order 24")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def closure(generators):
        reached, pending = {e}, [e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        for first in pending:
            for second in steps:
                product = G.mul(first, second)
                if product not in reached:
                    reached.add(product)
                    pending.append(product)
        return tuple(sorted(reached))

    rotations = closure(((1, 0),))
    outside_h12 = closure(((2, 0), (1, 1)))
    h8 = tuple(closure(((3, 0), (a, 1))) for a in (0, 2, 4))
    assert len(rotations) == len(outside_h12) == 12
    assert all(len(group) == 8 for group in h8)

    directions = tuple(sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != e and G.mul(g, g) != e
    }))

    def component_rows(A):
        rows, _, _ = lifted_matrices(G, A, [[[]]])
        mask = (1 << (3 * q)) - 1
        return [row & mask for row in rows]

    # Quotient prescriptions, not independent column gauges.
    # Do not identify arbitrary constituent graph isomorphisms here:
    # their embeddings can interact differently with the other side.
    bank, a_orbits = [], set()
    for triple in combinations(directions, 3):
        key = min(
            tuple(sorted(
                min(((u*a + v*b) % 12, b),
                    G.inv(((u*a + v*b) % 12, b)))
                for a, b in triple))
            for u in (1, 5, 7, 11) for v in range(0, 12, 2))
        if key in a_orbits:
            continue
        a_orbits.add(key)
        A = [[[e, g] for g in triple]]
        ar = component_rows(A)
        quality = _kernel_profile(ar, 3 * q)
        if quality[0] < 4:
            continue
        bank.append((q - gf2_rank(ar), quality, triple, A))
    bank.sort(key=lambda record: (
        -record[0], record[1][1], record[2]))
    if not bank:
        raise ValueError("no screened binomial quotient family")

    regular = []
    for internal in range(4):
        cols = ((2, 0),) * internal + ((0, 2),) * internal
        cols += ((1, 1),) * (7 - 2 * internal)
        regular.append(cols)

    # Enumerate genuine support multisets rather than permutations of
    # eight hand-picked patterns. Row exchange is removed canonically.
    types = ((0, 1), (1, 0), (0, 2), (1, 1), (2, 0),
             (0, 3), (1, 2), (2, 1), (3, 0))
    mixed = []
    for cols in combinations_with_replacement(types, 7):
        if tuple(sum(c[i] for c in cols) for i in range(2)) != (7, 7):
            continue
        if not any(sum(c) != 2 for c in cols):
            continue
        swapped = tuple(sorted((b, a) for a, b in cols))
        if swapped < cols:
            continue
        # Two pure monomial columns on one row already give a
        # constituent two-word, independent of their voltages.
        if cols.count((1, 0)) > 1 or cols.count((0, 1)) > 1:
            continue
        mixed.append(cols)
    mixed.sort(key=lambda cols: (
        -sum(sum(c) == 3 for c in cols), cols))
    if not mixed:
        raise ValueError("nonregular support catalogue is empty")

    def draw_A(attempt):
        if attempt % 16 != 15:
            record = bank[(attempt // 20) % len(bank)]
            return record[3], record[1], False

        # A different support hypothesis, not a permutation of three
        # binomials. The opposite row budget is reduced accordingly.
        profile = (3, 2, 2) if (attempt // 16) % 2 else (3, 3, 1)
        cells = []
        for terms in profile:
            if terms == 1:
                cell = [e]
            elif terms == 2:
                cell = [e, local.choice(directions)]
            else:
                outside = local.choice(
                    [g for g in G.elements if g[1] == 1])
                other = local.choice(
                    [g for g in G.elements if g not in (e, outside)])
                cell = [e, outside, other]
            cells.append(cell)
        A = [cells]
        return A, _kernel_profile(component_rows(A), 3*q), True

    def draw_B(attempt, experimental):
        card, epoch = attempt % 20, attempt // 20
        allocation, policy = divmod(card, 5)
        if experimental:
            # Row totals six and column totals at most two:
            # A-row seven + B-column two <=9;
            # B-row six + A-column three <=9.
            profile = ((1, 0), (0, 1)) + ((1, 1),) * 5
        elif attempt >= 240 and allocation >= 2:
            profile = mixed[(epoch + 7*card) % len(mixed)]
        else:
            profile = regular[allocation]

        if policy == 0:
            pool = h8[epoch % len(h8)]
        elif policy == 1:
            pool = rotations
        elif policy == 2:
            pool = outside_h12
        else:
            pool = tuple(G.elements)

        gaps = tuple(sorted({
            min(g, G.inv(g)) for g in pool
            if g != e and G.mul(g, g) != e
        }))
        if policy == 4 and not experimental:
            # Reverse the a-parity grading on internal edges and
            # preserve it on relative matching differences.
            gaps = tuple(g for g in directions if g[0] % 2)
            pool = tuple(g for g in G.elements if g[0] % 2 == epoch % 2)

        B = [[[] for _ in range(7)] for _ in range(2)]
        cross = [j for j, c in enumerate(profile) if c == (1, 1)]
        if len(cross) > len(pool):
            return None
        for j, voltage in zip(cross, local.sample(pool, len(cross))):
            B[0][j], B[1][j] = [e], [voltage]

        used = [set(), set()]
        for j, counts in enumerate(profile):
            if j in cross:
                continue
            for i, terms in enumerate(counts):
                if terms == 0:
                    continue
                if terms == 1:
                    B[i][j] = [local.choice(pool)]
                elif terms == 2:
                    options = [g for g in gaps if g not in used[i]]
                    if not options:
                        return None
                    gap = local.choice(options)
                    used[i].add(gap)
                    first = e if sum(counts) == 2 else local.choice(pool)
                    B[i][j] = [first, G.mul(first, gap)]
                else:
                    if len(pool) < terms:
                        return None
                    B[i][j] = local.sample(pool, terms)
        return B, (card, policy, experimental)

    def logical_profile(checks, stabilizers, width):
        """Exact existence of nontrivial logicals through weight four.

        Five means exclusion through four, not a GPU distance estimate.
        The burden is a representative-collision diagnostic, not a
        count of distinct logical operators.
        """
        basis = {}
        for row in stabilizers:
            while row:
                pivot = row.bit_length() - 1
                old = basis.get(pivot)
                if old is None:
                    basis[pivot] = row
                    break
                row ^= old

        def reduce(word):
            remainder = 0
            while word:
                pivot = word.bit_length() - 1
                old = basis.get(pivot)
                if old is None:
                    bit = 1 << pivot
                    remainder |= bit
                    word ^= bit
                else:
                    word ^= old
            return remainder

        cols = columns(checks, width)
        quotients = [reduce(1 << j) for j in range(width)]
        reps = {0: (0, 0)}
        floor, burden = 5, 0
        for j, syndrome in enumerate(cols):
            word, quotient = 1 << j, quotients[j]
            old = reps.get(syndrome)
            if old is None:
                reps[syndrome] = (word, quotient)
            elif old[1] != quotient:
                return (word ^ old[0]).bit_count(), 1

        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                quotient = quotients[i] ^ quotients[j]
                old = reps.get(syndrome)
                if old is None:
                    reps[syndrome] = (word, quotient)
                elif old[1] != quotient:
                    weight = (word ^ old[0]).bit_count()
                    if weight < 4:
                        return weight, burden + 1
                    floor, burden = 4, burden + 1
        return floor, burden

    def fingerprint(hx, hz, width):
        """Intrinsic Tanner invariant, with no CSS or block colouring.

        Root every intrinsically high-degree qubit. Project checks and
        low-qubit corners onto these roots, retaining multiplicities.
        Different keys certify nonisomorphism; equality is inconclusive
        and is rejected conservatively.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, width)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        positions = {j: v for v, j in enumerate(high)}
        ends, tags, check_ends = [], [], []

        for row in rows:
            endpoints = []
            bits = row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in positions:
                    endpoints.append(positions[j])
                bits ^= bit
            endpoints = tuple(sorted(endpoints))
            ends.append(endpoints)
            check_ends.append(endpoints)
            tags.append((0, row.bit_count()))

        for j, syndrome in enumerate(cols):
            if j in positions:
                continue
            endpoints, weights = [], []
            bits = syndrome
            while bits:
                bit = bits & -bits
                r = bit.bit_length() - 1
                endpoints.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                bits ^= bit
            ends.append(tuple(sorted(endpoints)))
            tags.append((1, tuple(sorted(weights))))

        incident = [[] for _ in high]
        for edge, endpoints in enumerate(ends):
            for v in endpoints:
                incident[v].append(edge)

        def compress(features):
            table = {f: i for i, f in enumerate(sorted(set(features)))}
            return [table[f] for f in features]

        roots = []
        for root in range(len(high)):
            colours = [int(v == root) for v in range(len(high))]
            history = []
            for _ in range(3):
                ef = [(tag, tuple(sorted(colours[v] for v in endpoints)))
                      for tag, endpoints in zip(tags, ends)]
                ec = compress(ef)
                vf = [(colours[v], tuple(sorted(ec[t] for t in incident[v])))
                      for v in range(len(high))]
                history.append((tuple(sorted(Counter(ef).items())),
                                tuple(sorted(Counter(vf).items()))))
                colours = compress(vf)
            roots.append(sha256(repr(history).encode("ascii")).digest())
        return (
            width,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(roots)),
            _tanner_signature(hx, hz),
        )

    graph_keys, fillings, results = set(), set(), []

    def incidence(top, bottom, voltages):
        B = [[], []]
        for gap in top:
            B[0].append([e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([e, gap])
        for voltage in voltages:
            B[0].append([e])
            B[1].append([voltage])
        return B

    # Exclusion references only. Never publish an archived anchor.
    references = (
        ((), (), (0, 1, 2, 3, 6, 7, 9)),
        ((1,), (5,), (0, 2, 4, 6, 8)),
        ((1, 3), (3, 5), (0, 2, 6)),
        ((1, 3, 5), (1, 3, 5), (0,)),
        ((1, 3), (2, 3), (0, 4, 8)),
        ((5, 3), (5, 3), (2, 0, 4)),
        ((3, 1, 5), (5, 1, 3), (2,)),
    )
    old_A = [[[e, (d, 0)] for d in (1, 3, 5)]]
    for top, bottom, voltages in references:
        B = incidence(tuple((d, 0) for d in top),
                      tuple((d, 0) for d in bottom),
                      tuple((d, 0) for d in voltages))
        hx, hz, n = lifted_matrices(G, old_A, B)
        graph_keys.add(fingerprint(hx, hz, n))

    additional = (
        (old_A, incidence((), (),
            tuple((d, 0) for d in (6, 1, 3, 10, 5, 2, 0)))),
        (old_A, incidence((), (),
            ((0,0),(4,1),(10,1),(0,1),(6,0),(2,0),(10,0)))),
        ([[[e, (d, 1)] for d in (1, 3, 5)]],
         incidence((), (),
            tuple((d, 0) for d in (8, 9, 2, 6, 5, 10, 11)))),
        ([[[e, (d, 1)] for d in (1, 3, 5)]],
         incidence(((3,1),(1,1)), ((9,1),(11,1)),
                   ((6,0),(2,0),(10,0)))),
    )
    for A, B in additional:
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(fingerprint(hx, hz, n))

    target = max(count, 20) + 12
    for batch in range(72):
        pool = []
        for trial in range(48):
            attempt = 48*batch + trial
            A, qa, experimental = draw_A(attempt)
            proposal = draw_B(attempt, experimental)
            if proposal is None:
                continue
            B, card = proposal
            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            if any(len(set(c)) != len(c) or len(c) > 5
                   for grid in (A, B) for row in grid for c in row):
                raise ValueError("constructed support contains duplicate terms")

            # The entire regular common-C12 mechanism is exhausted in
            # the parent. Do not rediscover its gauges or orbit images.
            if (not experimental
                    and all(sb[0][j] + sb[1][j] == 2 for j in range(7))
                    and all(g[1] == 0 for grid in (A, B)
                            for row in grid for c in row for g in c)):
                continue

            filling = tuple(tuple(sorted(G.idx(g) for g in c))
                            for grid in (A, B) for row in grid for c in row)
            if filling in fillings:
                continue
            fillings.add(filling)

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425*n <= 1000*k < 475*n):
                continue
            if max(r.bit_count() for r in hx + hz) > cap:
                continue
            _, br = _extract_components(G, hx, hz)
            qb = _kernel_profile(br, 7*q)

            # Constituent failures are diagnostics for the experimental
            # supports. Full logical-quotient tests decide publication.
            if not experimental and min(qa[0], qb[0]) < 4:
                continue
            priority = (-k, -(min(qa[0], qb[0])),
                        qa[1] + qb[1], attempt)
            pool.append((priority, A, B, hx, hz, n, k,
                         rx, rz, qa, qb, card))

        pool.sort(key=lambda record: record[0])
        # Keep an independently budgeted experimental lane; a lower-k
        # support hypothesis must not lose before its logical test.
        chosen = pool[:12]
        chosen += [record for record in pool[12:]
                   if record[-1][2]][:2]
        for record in chosen:
            _, A, B, hx, hz, n, k, rx, rz, qa, qb, card = record
            key = fingerprint(hx, hz, n)
            if key in graph_keys:
                continue
            graph_keys.add(key)

            xf = logical_profile(hz, hx, n)
            if xf[0] < 4:
                continue
            zf = logical_profile(hx, hz, n)
            if zf[0] < 4:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified relative lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")

            floor = min(xf[0], zf[0])
            quality = (k*floor*floor, floor, k,
                       -(xf[1] + zf[1]), -qa[1] - qb[1])
            screen = (
                ("k", k),
                ("card", card),
                ("logical_profiles", (xf, zf)),
                ("exact_logical_exclusion_through", floor - 1),
                ("constituent_diagnostics", (qa, qb)),
                ("rank_deficits", (len(hx)-rx, len(hz)-rz)),
            )
            results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= target:
            break

    results.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def _q8_columns(rows, width):
    """Column syndromes of supplied verified matrices."""
    result = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            result[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return result


def _q8_six_info(rows, width):
    """Exclude kernels through five; count all weight-six words.

    A returned zero count certifies exclusion through six for THIS
    constituent only. A nonzero witness is an actual kernel word.
    """
    cols = _q8_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            value = first ^ cols[j]
            if value in singles or value in pairs:
                return None
            pairs.add(value)

    triples = {}
    collisions, witness = 0, 0
    for i, first in enumerate(cols):
        for j in range(i):
            value = first ^ cols[j]
            for h in range(j):
                syndrome = value ^ cols[h]
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
    # Shorter kernels have been excluded, so colliding triples are
    # disjoint; each six-word has ten unordered triple partitions.
    assert collisions % 10 == 0
    if witness:
        assert witness.bit_count() == 6
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
    return collisions // 10, witness


def _q8_probe(rng, rows, width, passes=12):
    """Heuristic constituent search returning an actual kernel witness."""
    cols = _q8_columns(rows, width)
    best = 0

    def inspect(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

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

        rng.shuffle(nulls)
        for i, word in enumerate(nulls):
            for j, other in enumerate(nulls):
                if i != j and (word ^ other).bit_count() < word.bit_count():
                    word ^= other
            nulls[i] = word
            inspect(word)

        short = sorted(nulls, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])
        if best and best.bit_count() <= 7:
            break
    if not best:
        raise ValueError("Q8 constituent witness search returned no word")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _q8_profiles(rng):
    """Twenty deliberate support multisets, modulo row/column gauges."""
    from itertools import permutations

    seed = (
        (1, 1, 0, 0, 0, 2, 2, 0),
        (0, 1, 1, 1, 1, 0, 0, 2),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )
    orders = tuple(permutations(range(3)))

    def canonical(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(8)))
            for order in orders)

    double = sorted(set(permutations((2, 0, 0)))
                    | set(permutations((1, 1, 0))))
    triple = sorted(set(permutations((3, 0, 0)))
                    | set(permutations((2, 1, 0)))
                    | {(1, 1, 1)})
    profiles, keys = [seed], {canonical(seed)}
    for _ in range(65536):
        columns = [(1, 0, 0)]
        columns += [rng.choice(double) for _ in range(4)]
        columns += [rng.choice(triple) for _ in range(3)]
        profile = tuple(tuple(c[i] for c in columns) for i in range(3))
        if any(sum(row) != 6 for row in profile):
            continue
        key = canonical(profile)
        if key in keys:
            continue
        keys.add(key)
        profiles.append(profile)
        if len(profiles) == 20:
            break
    if len(profiles) != 20:
        raise ValueError("Q8 canonical support catalogue incomplete")
    return tuple(profiles)


def _q8_draw(rng, G, profile, seeded, policy):
    """Choose relative noncommuting labels, not completed-code gauges."""
    e = G.e
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != e and G.mul(g, g) != e
    })
    assert len(directions) == 3
    C = [[[] for _ in range(8)] for _ in range(3)]

    if seeded:
        # Preserve the grounded connectivity skeleton, not cyclic labels.
        # Three matching voltages must have non-involutory differences.
        options = []
        for g in G.elements:
            for h in G.elements:
                if g == e or h == e or g == h:
                    continue
                differences = (g, h, G.mul(G.inv(g), h))
                if any(G.mul(d, d) == e for d in differences):
                    continue
                if policy % 2 == 0 and G.mul(g, h) == G.mul(h, g):
                    continue
                options.append((g, h))
        g, h = rng.choice(options)
        C[0][0] = [e]
        C[0][1] = [e]
        C[1][1] = [e]
        for j, voltage in enumerate((e, g, h), 2):
            C[1][j] = [e]
            C[2][j] = [voltage]
        gaps = (rng.sample(directions, 3) if policy < 2 else
                [rng.choice(directions) for _ in range(3)])
        for (j, row), gap in zip(((5, 0), (6, 0), (7, 1)), gaps):
            C[row][j] = [e, gap]
            C[2][j] = [G.pick(rng)]
        return C

    for j in range(8):
        anchor_fixed = False
        for i in range(3):
            count = profile[i][j]
            if not count:
                continue
            if count == 1:
                cell = [e if not anchor_fixed else G.pick(rng)]
            elif count == 2:
                first = e if not anchor_fixed else G.pick(rng)
                gap = rng.choice(directions)
                cell = [first, G.mul(first, gap)]
            else:
                for _ in range(32):
                    cell = rng.sample(G.elements, count)
                    if not anchor_fixed:
                        inverse = G.inv(cell[0])
                        cell = [G.mul(inverse, g) for g in cell]
                    differences = [
                        G.mul(G.inv(cell[a]), cell[b])
                        for a in range(count) for b in range(a)
                    ]
                    if all(G.mul(d, d) != e for d in differences):
                        break
                else:
                    return None
            C[i][j] = cell
            anchor_fixed = True
    return C


def _q8_basis(rows):
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


def _q8_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _q8_logical_reading(G, a, b, hx, hz):
    """Explicit embedded logicals, shortened by legal stabilizer moves.

    The returned reading is an UPPER bound, not a distance certificate.
    Both types use the actual left/right constituent witness.
    """
    q = G.q
    overall = None
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _q8_basis(stabilizers)
        source = b["right_word"] if kind == "X" else a["left_word"]
        best = 0
        for block in range(8):
            if kind == "X":
                word = source << (block * 8 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 8 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _q8_outside(word, own):
                continue

            for _ in range(3):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                starting = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:40]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < word.bit_count():
                            word = trial
                if word == previous:
                    break
            assert word and _q8_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not best or word.bit_count() < best.bit_count():
                best = word
        if not best:
            return None
        overall = (best.bit_count() if overall is None else
                   min(overall, best.bit_count()))
    return overall


def _q8_fingerprint(hx, hz, width):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive.

    Degree-twelve qubits identify sector 2 without group or block labels.
    Rooted projected checks and low-qubit corners retain multiplicity.
    Different keys certify different submitted Tanner graphs.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _q8_columns(rows, width)
    high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
    positions = {j: v for v, j in enumerate(high)}
    endpoints, tags, check_ends = [], [], []
    for row in rows:
        ends = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                ends.append(positions[j])
            bits ^= bit
        ends = tuple(sorted(ends))
        endpoints.append(ends)
        check_ends.append(ends)
        tags.append((0, row.bit_count()))

    for j, syndrome in enumerate(cols):
        if j in positions:
            continue
        ends, weights = [], []
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            ends.extend(check_ends[r])
            weights.append(rows[r].bit_count())
            syndrome ^= bit
        endpoints.append(tuple(sorted(ends)))
        tags.append((1, tuple(sorted(weights))))

    incident = [[] for _ in high]
    for edge, ends in enumerate(endpoints):
        for v in ends:
            incident[v].append(edge)

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(3):
            ef = [(tag, tuple(sorted(colours[v] for v in ends)))
                  for tag, ends in zip(tags, endpoints)]
            ec = compress(ef)
            vf = [(colours[v], tuple(sorted(ec[t] for t in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(ef).items())),
                            tuple(sorted(Counter(vf).items()))))
            colours = compress(vf)
        rooted.append(sha256(repr(history).encode("ascii")).digest())
    return (
        width,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted(rooted)),
        _tanner_signature(hx, hz),
    )


@lru_cache(maxsize=1)
def _q8_portfolio(G, count):
    """Immutable deterministic portfolio, independent of slot call order."""
    if G.q != 8:
        raise ValueError("grounded quaternion portfolio requires Q8")
    local = random.Random(BASE_SEED ^ 0x8356731)
    profiles = _q8_profiles(local)
    cap, width = min(9, W_MAX), 64
    mask = (1 << width) - 1
    bank, filling_keys = [], set()
    bank_target = max(40, 2 * int(count ** 0.5) + 32)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    # Equal initial budgets for twenty actual support prescriptions.
    # Later exploitation changes genuine quaternion relative labels.
    for attempt in range(6144):
        card = attempt % 20
        pi = card if attempt < 1920 else (
            0 if attempt % 3 else 1 + (attempt // 3) % 19)
        policy = (attempt // 20) % 4
        C = _q8_draw(local, G, profiles[pi], pi == 0, policy)
        if C is None:
            continue
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in C for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        counts = [[len(cell) for cell in row] for row in C]
        assert all(sum(row) == 6 for row in counts)
        assert all(sum(row[j] for row in counts) <= 3 for j in range(8))
        assert all(len(set(cell)) == len(cell) <= 5
                   for row in C for cell in row)
        assert _no_dead_lines(counts, counts)
        assert _grid_weights(counts, counts) <= cap

        # Only the verified helper supplies either regular action.
        left, _, _ = lifted_matrices(G, C, [[[]]])
        _, right, _ = lifted_matrices(G, [[[]]], C)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if gf2_rank(left) != 24 or gf2_rank(right) != 24:
            continue
        li = _q8_six_info(left, width)
        if li is None:
            continue
        ri = _q8_six_info(right, width)
        if ri is None:
            continue
        left_word = li[1] or _q8_probe(local, left, width)
        right_word = ri[1] or _q8_probe(local, right, width)
        assert left_word.bit_count() >= 6 and right_word.bit_count() >= 6
        bank.append({
            "grid": C, "left_word": left_word, "right_word": right_word,
            "six_counts": (li[0], ri[0]),
            "readings": (left_word.bit_count(), right_word.bit_count()),
            "card": (pi, policy), "attempt": attempt,
        })
        if attempt >= 1919 and len(bank) >= 3 * bank_target:
            break

    if len(bank) < 8:
        raise ValueError("too few kernel-five-clean quaternion constituents")
    bank.sort(key=lambda r: (
        -min(r["readings"]), max(r["six_counts"]),
        sum(r["six_counts"]), r["attempt"]))
    bank = bank[:bank_target]

    plans = sorted(
        (-min(a["readings"][0], b["readings"][1]),
         max(a["six_counts"][0], b["six_counts"][1]),
         a["six_counts"][0] + b["six_counts"][1], ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank)
        if ai <= bi)

    graph_keys, results = set(), []
    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(8))
        wz = max(sum(row) + sum(sa[i][j] for i in range(3))
                 for row in sb for j in range(8))
        assert wx <= 9 and wz <= 9
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if n != 584 or k != 200:
            continue
        assert 325*n <= 1000*k < 375*n
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Complete matrices include both sectors. These gates are exact
        # through four, but do not claim exclusion of mixed weight five.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        reading = _q8_logical_reading(G, a, b, hx, hz)
        if reading is None or reading < 6:
            continue
        key = _q8_fingerprint(hx, hz, n)
        if key in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quaternion lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified quaternion rank helpers disagree")
        graph_keys.add(key)
        quality = (
            k * reading * reading / n, reading,
            -a["six_counts"][0] - b["six_counts"][1],
            -max(a["six_counts"][0], b["six_counts"][1]),
        )
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("cards", (a["card"], b["card"])),
            ("constituent_six_counts",
             (a["six_counts"][0], b["six_counts"][1])),
            ("constituent_witness_readings",
             (a["readings"][0], b["readings"][1])),
            ("complete_kernel_exclusion_through", 4),
            ("component_kernel_exclusion_through", 5),
            ("LP_rank_deficits", (len(hx)-rx, len(hz)-rz)),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= max(count, 20) + 16:
            break

    results.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup; an exhausted catalogue is never recycled or relabelled."""
    portfolio = _q8_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("inequivalent quaternion portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Publish the complete selected quaternion cell without mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty evaluated slots retained only five downstream classes, exhausted after slot 51, and all five measured codes had d30s=4.
# REFUTED: The order-eight relative-support portfolio escapes weight four on the order-24 bed: all five measured leaders had d30s=4, and sixty evaluated slots retained twelve downstream classes.
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
