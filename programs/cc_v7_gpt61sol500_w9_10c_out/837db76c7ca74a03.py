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
CONCEPT = ("Faithful C11 semidirect_3 C5, order 55, A=B=(1,3): complete kernel-four exclusion plus explicit nontrivial weight-five logicals should supply five fresh k>=226 outputs in cell 0.400, scoring at least 10.2727 and improving its measured 10.2 frontier.")
GROUP = ("Jump from Dic6 to the faithful Frobenius group C11:C5. The action 3 has order five modulo eleven, so the presentation is valid and genuinely different from the order-24 parent. This group's measured (1,3)x(1,3) bed already reached distance five and fitness 10.2. Reuse that algebraic bed with enumerated, exactly screened new fillings, not copied archive labels.")
SHAPES = ("Equal (1,3) constituents have 165 variables and 55 checks. Factor ten permits q<=59; the prescribed q=55 gives n=550, kb=220 and guaranteed rate 0.400. Verified complete ranks require k>=226 and actual rate in [0.375,0.425). Parent weight-four minima broadly occupied sector-1 Acol0..2/Bcol0..6. The new three-column B and full kernel-four gate target those words and mixed-sector alternatives.")
FILLING = ("Enumerate connected inverse-class generator triples modulo all Frobenius automorphisms and independent binomial reversals. Verified constituent lifts exclude kernels through four and supply an actual five-cycle. Pair heterogeneous constituents with both relative outer-automorphism classes; every cell has two distinct terms, giving BOTH CSS row maxima eight. Full uncoloured Tanner invariants conservatively separate outputs and exclude the shown frontier anchor under all faithful quotient encodings. Complete matrices exclude every kernel word through four and verify an embedded weight-five word outside the stabilizer span. Accepted outputs therefore have distance exactly five; evaluator readings remain upper bounds, and novelty against the unseen archive is not certified.")
FAMILY_NAME = "r040_faithful_c11sd3_c5_q55_13x13_exactfive_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("faithful C11:C5 requires order 55")
    # x^11=y^5=1, y*x*y^-1=x^3; 3 has order five modulo 11.
    powers = [pow(3, b, 11) for b in range(5)]
    els = [(a, b) for b in range(5) for a in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 5)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 11, (b + d) % 5)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 5
        return ((-powers[bi] * a) % 11, bi)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _column_syndromes(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _component_collisions(rows, width):
    """None rejects kernels <=3; otherwise count pair collisions at weight 4."""
    columns = _column_syndromes(rows, width)
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return None
    pairs = set()
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return None
            if syndrome in pairs:
                collisions += 1
            else:
                pairs.add(syndrome)
    return collisions


def _a_catalogue(G):
    """Spanning quotient triples, modulo inversion and group automorphisms."""
    from itertools import combinations

    # Inverse representatives of odd-a elements. The original triples
    # have a bipartite grading and contain no involution differences.
    options = ((1, 0), (3, 0), (5, 0),
               (1, 1), (3, 1), (5, 1))
    representatives = {}

    def unoriented(g):
        return min(g, G.inv(g))

    for triple in combinations(options, 3):
        reached = {G.e}
        frontier = [G.e]
        generators = triple + tuple(G.inv(g) for g in triple)
        while frontier:
            g = frontier.pop()
            for h in generators:
                product = G.mul(g, h)
                if product not in reached:
                    reached.add(product)
                    frontier.append(product)
        if len(reached) != G.q:
            continue

        # Every Dic6 automorphism sends x to x^u and y to x^v*y.
        key = min(
            tuple(sorted(unoriented(((u * a + v * b) % 12, b))
                         for a, b in triple))
            for u in (1, 5, 7, 11) for v in range(12)
        )
        representatives.setdefault(key, triple)

    catalogue = []
    for key in sorted(representatives):
        triple = representatives[key]
        A = [[[G.e, g] for g in triple]]
        # This is a projection of the VERIFIED lift with a dummy side.
        hx, _, _ = lifted_matrices(G, A, [[[G.e]]])
        mask = (1 << (3 * G.q)) - 1
        quality = _component_collisions([r & mask for r in hx], 3 * G.q)
        if quality is not None:
            catalogue.append((triple, quality))
    if not catalogue:
        raise ValueError("No spanning A quotient family passed screening")
    return catalogue


def _b_profiles():
    """Seven support cards: both rows sum to 7; every column sums to 1..3."""
    columns = (
        ((1, 1),) * 7,
        ((2, 0), (0, 2)) + ((1, 1),) * 5,
        ((2, 0),) * 2 + ((0, 2),) * 2 + ((1, 1),) * 3,
        ((2, 0),) * 3 + ((0, 2),) * 3 + ((1, 1),),
        ((1, 0), (1, 2)) + ((1, 1),) * 5,
        ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
        ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
    )
    return tuple(tuple(tuple(c[i] for c in profile) for i in range(2))
                 for profile in columns)


def _draw_b(rng, G, profile, mode):
    def cell(count, order_model):
        if count == 0:
            return []
        first = G.pick(rng)
        if count == 1:
            return [first]
        if count == 2:
            if order_model == 0:
                quotient = (rng.choice((1, 5, 7, 11)), 0)  # order 12
            elif order_model == 1:
                quotient = (rng.choice((3, 9)), 0)          # order 4
            else:
                quotient = (rng.randrange(12), 1)          # order 4
            return [first, G.mul(first, quotient)]
        result = [first]
        while len(result) < count:
            g = G.pick(rng)
            if g not in result:
                result.append(g)
        return result

    # Different internal columns can use different quotient-order models.
    return [[cell(profile[i][j], (mode + i + j) % 3)
             for j in range(7)] for i in range(2)]


def _verified_components(hx, hz, q):
    """Project the actual left/right components out of the supplied LP lift."""
    mask = (1 << q) - 1
    arows = []
    for t in range(q):
        source = hx[t]                    # A-row 0, B-column 0
        row = 0
        for j in range(3):
            row |= ((source >> (j * 7 * q)) & mask) << (j * q)
        arows.append(row)
    bmask = (1 << (7 * q)) - 1
    brows = [row & bmask for row in hz[:2 * q]]
    return arows, brows


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


def _no_short_kernel(rows, n):
    """Exactly exclude kernel vectors of weights one, two and three."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    seen = set(columns)
    if 0 in seen or len(seen) != n:
        return False
    for i, c in enumerate(columns):
        for j in range(i):
            if (c ^ columns[j]) in seen:
                return False
    return True


def _unused_dic6_ctx(rng, G, idx):
    """Uncalled historical Dic6 prefix; never publishes on the new bed.

    Different accepted full-Tanner invariants certify non-isomorphic
    submitted graphs. Equal invariants are conservatively discarded.
    Every prefix entry passes the same screens, independently of idx.
    """
    local = random.Random(BASE_SEED ^ 0xD6C258)
    profiles = _b_profiles()

    # +/-1,+/-3,+/-5 are precisely the odd rotations in C12.
    # The verified A incidence lift therefore has two K6,6 components.
    root_A = [[[G.e, (d, 0)] for d in (1, 3, 5)]]
    bank = [(root_A, 2)]
    for triple, _ in _a_catalogue(G):
        A = [[[G.e, g] for g in triple]]
        hx, _, _ = lifted_matrices(G, A, [[[G.e]]])
        mask = (1 << (3 * G.q)) - 1
        ar = [row & mask for row in hx]
        bank.append((A, G.q - gf2_rank(ar)))

    def signature(hx, hz):
        return (tuple(sorted(row.bit_count() for row in hx + hz)),
                _tanner_signature(hx, hz))

    graph_keys = set()

    # These are the three shown regular-B archive anchors with the
    # two-component rotation A topology. Column gauges and binomial
    # reversals normalize their A cells to root_A.
    anchors = (
        (((8,0),(1,1),(1,0),(6,1),(2,1),(8,1),(5,1)),
         ((3,0),(10,0),(10,0),(11,1),(4,1),(2,1),(8,1))),
        (((11,1),(2,0),(3,0),(5,1),(8,1),(4,0),(10,1)),
         ((2,1),(3,0),(5,1),(2,1),(3,1),(9,0),(0,0))),
        (((8,0),(8,0),(7,1),(10,1),(5,1),(10,1),(11,0)),
         ((2,1),(5,1),(3,0),(3,0),(1,1),(0,1),(4,0))),
    )
    for top, bottom in anchors:
        B = [[[g] for g in top], [[g] for g in bottom]]
        hx, hz, _ = lifted_matrices(G, root_A, B)
        graph_keys.add(signature(hx, hz))

    # Five constructive relative-label policies. For policies 0,2,3,4,
    # internal edges reverse a grading and cross-voltage differences
    # preserve it, making the regular B incidence graph bipartite.
    policies = []
    for mode in range(5):
        if mode == 0:
            steps = [(a, 0) for a in range(1, 12, 2)]
            labels = [(a, 0) for a in range(0, 12, 2)]
        elif mode == 1:
            # Also test order-six rotation gaps, absent from the
            # parent's order-12/order-4-only binomial model.
            steps = [(a, 0) for a in (1, 2, 3, 5, 7, 9, 10, 11)]
            labels = [(a, 0) for a in range(12)]
        else:
            def grade(g):
                a, b = g
                return (a if mode == 2 else b if mode == 3
                        else a + b) % 2
            steps = [g for g in G.elements if grade(g) == 1]
            labels = [g for g in G.elements if grade(g) == 0]
        # An internal edge orbit and its inverse are the same orbit.
        steps = sorted({min(g, G.inv(g)) for g in steps})
        policies.append((steps, labels))

    # Four support prescriptions times five relative-label policies.
    # Profiles 1..3 have two, four or six internal-cycle columns.
    # Profile 4 is a one-singleton/one-degree-three alternative.
    plans = tuple((p, mode) for p in (1, 2, 3, 4)
                  for mode in range(5))
    accepted = 0

    for attempt in range(max(4096, 192 * (idx + 1))):
        profile_index, mode = plans[attempt % len(plans)]
        if attempt >= 800:
            # Broaden support prescriptions only after spending a
            # substantial search on the high-rank internally cycled bed.
            profile_index = (profile_index + attempt // 20) % len(profiles)
        family_index = (0 if attempt < 1600 else
                        (attempt // len(plans)) % len(bank))
        A, a_deficit = bank[family_index]
        sb = profiles[profile_index]
        sa = [[2, 2, 2]]
        if not (_no_dead_lines(sa, sb)
                and _grid_weights(sa, sb) <= min(9, W_MAX)):
            raise ValueError("Rank-deficit card violates support legality")

        steps, labels = policies[mode]
        B = [[[] for _ in range(7)] for _ in range(2)]

        if profile_index <= 3:
            # Each column is a simple incidence edge: either one
            # internal binomial or a cross-row matching.
            internal = [[j for j in range(7)
                         if sb[i][j] == 2 and sb[1-i][j] == 0]
                        for i in range(2)]
            if any(len(js) > len(steps) for js in internal):
                continue
            for i in range(2):
                gaps = local.sample(steps, len(internal[i]))
                for j, gap in zip(internal[i], gaps):
                    B[i][j] = [G.e, gap]
            cross = [j for j in range(7)
                     if sb[0][j] == sb[1][j] == 1]
            if len(cross) > len(labels):
                continue
            voltages = local.sample(labels, len(cross))
            for j, voltage in zip(cross, voltages):
                B[0][j], B[1][j] = [G.e], [voltage]
        else:
            # Mixed-degree fallbacks retain the same subgroup/parity
            # prescriptions, rather than reverting to uniform G labels.
            for i in range(2):
                for j, count in enumerate(sb[i]):
                    if count == 0:
                        continue
                    first = local.choice(labels)
                    if count == 1:
                        cell = [first]
                    elif count == 2:
                        gap = local.choice(steps)
                        cell = [first, G.mul(first, gap)]
                    else:
                        cell = local.sample(labels, count)
                    if len({G._key(g) for g in cell}) != count:
                        raise ValueError("Constructed cell contains duplicates")
                    B[i][j] = cell

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _verified_components(hx, hz, G.q)
        qa = _component_collisions(ar, 3 * G.q)
        qb = _component_collisions(br, 7 * G.q)
        if qa is None or qb is None:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        minimum_k = 258 if attempt < 800 else 250 if attempt < 1600 else 245
        if not (n == 552 and k >= minimum_k
                and 425 * n <= 1000 * k < 475 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
            continue

        key = signature(hx, hz)
        if key in graph_keys:
            continue
        # Complete matrices include both sectors. This is an exact
        # exclusion through weight three, not a distance estimate.
        if not (_no_short_kernel(hx, n)
                and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("Verified lift failed CSS commutation")

        graph_keys.add(key)
        if accepted == idx:
            if lifted_k(G, A, B) != k:
                raise ValueError("Verified rank helpers disagree")
            return {"A": A, "B": B, "k": k,
                    "card": (profile_index, mode, family_index),
                    "component_quality": (qa, qb),
                    "A_rank_deficit": a_deficit,
                    "LP_rank_deficits": (
                        len(hx) - gf2_rank(hx),
                        len(hz) - gf2_rank(hz))}
        accepted += 1

    raise ValueError("Fresh rank-deficit catalogue exhausted")


from functools import lru_cache


def _f55_span(rows):
    basis = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            old = basis.get(pivot)
            if old is None:
                basis[pivot] = word
                break
            word ^= old
    return basis


def _f55_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        old = basis.get(pivot)
        if old is None:
            return True
        word ^= old
    return False


def _f55_kernel4(rows, width):
    """Exact exclusion of every nonzero kernel support through four.

    All such supports split into two supports of size at most two.
    Distinct nonzero singles and unique pair syndromes exclude them.
    This is not a budget-limited distance estimate.
    """
    columns = _column_syndromes(rows, width)
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


def _f55_cycle(rows, width):
    """Shortest actual cycle of a verified degree-two constituent."""
    columns = _column_syndromes(rows, width)
    adjacency = [[] for _ in rows]
    edges = set()
    for edge, syndrome in enumerate(columns):
        if syndrome.bit_count() != 2:
            raise ValueError("binomial constituent is not an incidence graph")
        bit = syndrome & -syndrome
        a = bit.bit_length() - 1
        b = (syndrome ^ bit).bit_length() - 1
        endpoints = (min(a, b), max(a, b))
        if endpoints in edges:
            raise ValueError("parallel constituent edges survived kernel4")
        edges.add(endpoints)
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))

    best = 0
    for root in range(len(rows)):
        distance = [-1] * len(rows)
        paths = [0] * len(rows)
        distance[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if distance[b] < 0:
                    distance[b] = distance[a] + 1
                    paths[b] = paths[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = paths[a] ^ paths[b] ^ (1 << edge)
                    if word and (not best or
                                 word.bit_count() < best.bit_count()):
                        best = word
        if best and best.bit_count() == 5:
            break
    if not best:
        raise ValueError("connected six-regular constituent has no cycle")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _f55_rows(G, grid, right=False):
    """Expose exactly the scaffold's component action with a zero dummy."""
    if right:
        _, rows, _ = lifted_matrices(G, [[[]]], grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
    width = 3 * G.q
    mask = (1 << width) - 1
    assert len(rows) == G.q
    return [row & mask for row in rows]


def _f55_key(hx, hz, n):
    """Intrinsic full uncoloured Tanner invariant.

    Checks have degree eight, whereas qubits have degree four or twelve.
    Thus these vertex classes are intrinsic, without X/Z colouring.
    Unequal digests imply unequal invariant tuples; equality is rejected
    conservatively, never treated as proof of graph isomorphism.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _column_syndromes(rows, n)
    qubit_profiles = Counter()
    for column in columns:
        weights = []
        while column:
            bit = column & -column
            weights.append(rows[bit.bit_length() - 1].bit_count())
            column ^= bit
        qubit_profiles[tuple(sorted(weights))] += 1
    signature = (
        n,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(qubit_profiles.items())),
        _tanner_signature(list(hx), list(hz)),
    )
    return sha256(repr(signature).encode("ascii")).digest()


def _f55_five_logical(G, a, b, hx, hz):
    """Exhibit an actual weight-five logical, not just a constituent word."""
    q = G.q
    for kind, checks, stabilizers, source in (
            ("X", hz, hx, b["right"]),
            ("Z", hx, hz, a["left"])):
        basis = _f55_span(stabilizers)
        for block in range(3):
            if kind == "X":
                word = source << (block * 3 * q)
            else:
                word = 0
                bits = source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 3 + block) * q + h)
                    bits ^= bit
            if word.bit_count() != 5:
                raise ValueError("constituent five-cycle weight changed")
            if not all((row & word).bit_count() % 2 == 0
                       for row in checks):
                raise ValueError("verified constituent embedding failed")
            if _f55_outside(word, basis):
                return kind, word
    return None


@lru_cache(maxsize=1)
def _f55_portfolio(G, count):
    """Deterministic immutable portfolio of exactly distance-five codes.

    Enumerated generator prescriptions replace random slot fillings.
    Memoization affects build cost only; outputs do not depend on call
    order, and every returned cell is copied by CELL.
    """
    from itertools import combinations

    q, e = G.q, G.e
    if q != 55:
        raise ValueError("exact-five portfolio requires faithful order 55")
    support = [[2, 2, 2]]
    cap = min(9, 12, W_MAX)
    wx = sum(support[0]) + max(support[0])
    wz = sum(support[0]) + max(support[0])
    assert wx == wz == 8 <= cap
    assert _grid_weights(support, support) == 8
    assert _no_dead_lines(support, support)

    options = sorted({
        min(g, G.inv(g)) for g in G.elements if g != e
    })
    powers = [pow(3, b, 11) for b in range(5)]

    def canonical(triple):
        # All automorphisms: translations a -> u*a+(1-3^b)*v.
        # Independent generator inversions reverse binomial edge orbits.
        return min(
            tuple(sorted(min(
                image, G.inv(image))
                for image in (
                    ((u*a + (1-powers[b])*v) % 11, b)
                    for a, b in triple)))
            for u in range(1, 11) for v in range(11)
        )

    prescriptions = sorted({canonical(t)
                            for t in combinations(options, 3)})
    local = random.Random(BASE_SEED ^ 0xF550405)
    local.shuffle(prescriptions)
    bank = []

    for card, triple in enumerate(prescriptions):
        grid = [[[e, g] for g in triple]]
        rows = _f55_rows(G, grid)
        # Incidence rank q-1 certifies connectedness of this component.
        if gf2_rank(rows) != q - 1:
            continue
        if not _f55_kernel4(rows, 3*q):
            continue
        left = _f55_cycle(rows, 3*q)
        # A six-regular graph on 55 vertices cannot have girth >=6:
        # its bipartite Moore bound would require at least 62 vertices.
        if left.bit_count() != 5:
            raise ValueError("verified girth contradicts the Moore bound")

        right_rows = _f55_rows(G, grid, True)
        right = 0
        bits = left
        while bits:
            bit = bits & -bits
            j, h = divmod(bit.bit_length() - 1, q)
            right |= 1 << (j*q + G.idx(G.inv(G.elements[h])))
            bits ^= bit
        assert all((row & right).bit_count() % 2 == 0
                   for row in right_rows)
        assert gf2_rank(right_rows) == q - 1
        bank.append({
            "grid": grid, "left": left, "right": right,
            "card": card, "triple": triple,
        })

    if len(bank) < 2:
        raise ValueError("too few connected verified girth-five constituents")

    excluded = set()
    # Exclusion references only. The frontier's quotient generator can
    # encode any of the four faithful actions; cover all those encodings.
    for quotient_scale in range(1, 5):
        def image(g):
            return (g[0] % 11, quotient_scale*g[1] % 5)
        old_A = [[[e, image(g)] for g in
                  ((1, 0), (0, 1), (5, 2))]]
        old_B = [[[e, image(g)] for g in
                  ((1, 0), (0, 2), (5, 2))]]
        hx, hz, n = lifted_matrices(G, old_A, old_B)
        excluded.add(_f55_key(hx, hz, n))

    # Inner translation/scaling gauges were removed above. Aut(G)/Inn(G)
    # has two classes, represented by translation-coordinate scales 1,2.
    plans = [(ai, bi, scale)
             for ai in range(len(bank))
             for bi in range(ai, len(bank))
             for scale in (1, 2)]
    local.shuffle(plans)
    plans.sort(key=lambda p: p[0] == p[1])
    results = []

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for ai, bi, scale in plans:
        a = bank[ai]
        original_b = bank[bi]
        B = [[[(scale*g[0] % 11, g[1]) for g in cell]
              for cell in original_b["grid"][0]]]
        b = {"grid": B, "right": 0}
        bits = original_b["right"]
        while bits:
            bit = bits & -bits
            j, h = divmod(bit.bit_length() - 1, q)
            g = G.elements[h]
            b["right"] |= 1 << (
                j*q + G.idx((scale*g[0] % 11, g[1])))
            bits ^= bit
        A = a["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 550 and k >= 226
                and 375*n <= 1000*k < 425*n):
            continue
        if max(row.bit_count() for row in hx + hz) != 8:
            raise ValueError("full row weights disagree with both budgets")
        key = _f55_key(hx, hz, n)
        if key in excluded:
            continue
        if not (_f55_kernel4(hx, n) and _f55_kernel4(hz, n)):
            continue
        logical = _f55_five_logical(G, a, b, hx, hz)
        if logical is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified exact-five matrices failed CSS")
        assert lifted_k(G, A, B) == k
        excluded.add(key)
        results.append((
            k, freeze(A), freeze(B),
            (a["card"], original_b["card"], scale),
            logical[0], logical[1], key.hex(),
            (len(hx)-rx, len(hz)-rz),
        ))
        # Compare a wider screened portfolio before rank-first publication.
        if len(results) >= max(count, 5) + 16:
            break

    results.sort(key=lambda record: (-record[0], record[6]))
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete invariant-separated exact-five design."""
    portfolio = _f55_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("fresh exact-five Frobenius portfolio exhausted")
    k, A, B, cards, kind, word, signature, deficits = portfolio[idx]
    return {
        "A": A, "B": B, "k": k, "design_cards": cards,
        "kernel_weights_1_to_4_excluded": True,
        "logical_five_type": kind, "logical_five_word": word,
        "exact_code_distance": 5, "tanner_signature": signature,
        "LP_rank_deficits": deficits,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell from the immutable screened design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on the spanning-A Dic6 catalogue escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
# REFUTED: The rank-deficit Dic6 prefix supplies twenty inequivalent initial designs or escapes distance four: current slots 0..19 retained seven classes, and all five measured leaders had d30s=4 with abundant sector-1 witnesses.
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
