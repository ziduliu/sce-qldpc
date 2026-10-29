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
CONCEPT = ("Frozen Dic3, order 12, A=(2,6), B=(2,7): exhaustive gauge-quotiented B incidence topologies, separated by their intrinsic degree-13-qubit/weight-eight-check projections, should supply five novel k=258 distance-four finalists and improve fitness above 7.4664.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This bed measured four k=258,d30s=4 finalists and one k=256,d30s=4 finalist. These readings support improving the fifth rank and novel-topology yield on Dic3, not assuming that another element-order policy escapes distance four.")
SHAPES = ("Keep A=(2,6), B=(2,7) byte-for-byte unchanged: n=552, kb=240 and guaranteed rate 20/46=10/23 in cell 0.450. The size window permits q<=13; the frozen group fixes q=12. Unequal constituents have 72 variables/24 checks and 84 variables/24 checks. Parent measured minima occupied sector-1 Acol0..5/Bcol0..6 broadly, with X minima especially frequent at Acol4; none of the measured minima required sector 2. Verified full ranks require k=258, rate 0.46739.")
FILLING = ("A has three outside-class binomials on each isolated row. Exhaustively enumerate B's four balanced internal-edge allocations and five cross-coset-count prescriptions, removing automorphisms, row exchange and matching-anchor gauges before lifting. Every B column has two terms, A rows total six and B rows seven, giving HX=8 and HZ=9. Verified constituents and complete kernels exclude weights 1..3; exact stabilizer-aware screening checks weight four. Recover B's incidence graph intrinsically from weight-eight checks and degree-13 qubits, and conservatively reject repeated rooted graph invariants plus shown archive anchors. Five fresh k=258,d30s=4 codes would score 7.47826; novelty against the unseen archive remains a measurement hypothesis.")
FAMILY_NAME = "r045_dic3_26x27_exhaustive_Bprojection_portfolio_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic3 has order twelve")
    # Same dicyclic rule: x^(2t)=1, y^2=x^t, y*x*y^-1=x^-1.
    # Factor 46 requires q<=13; t=3 gives q=12.
    t, m = 3, 6
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + t * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + t * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + t) % m, 1)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _unused_dic6_reference(rng, G, idx):
    """Uncalled historical policy; never publishes a new-shape candidate.

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


def _newshape_components(G, hx, hz):
    """Extract both constituents from the exact evaluator matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for h in range(q):
            source = hx[i * cb * q + h]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(row)
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _logical_four_profile(checks, stabilizers, n):
    """Exact logical existence through four, modulo the stabilizer span.

    Five means exclusion through four, not a larger-distance estimate.
    The burden counts detected collisions, not distinct operators.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def reduce(word):
        remainder = 0
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                bit = 1 << p
                remainder |= bit
                word ^= bit
            else:
                word ^= old
        return remainder

    columns = _column_syndromes(checks, n)
    # Reduction is linear. Precomputing unit remainders avoids running
    # elimination for every pair collision.
    quotients = [reduce(1 << i) for i in range(n)]
    representatives = {0: (0, 0)}
    floor, burden = 5, 0

    for i, syndrome in enumerate(columns):
        word, quotient = 1 << i, quotients[i]
        old = representatives.get(syndrome)
        if old is None:
            representatives[syndrome] = (word, quotient)
        elif quotient != old[1]:
            return (word ^ old[0]).bit_count(), 1

    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            word = (1 << i) | (1 << j)
            quotient = quotients[i] ^ quotients[j]
            old = representatives.get(syndrome)
            if old is None:
                representatives[syndrome] = (word, quotient)
            elif quotient != old[1]:
                weight = (word ^ old[0]).bit_count()
                if weight < 4:
                    return weight, burden + 1
                floor = 4
                burden += 1
    return floor, burden


def _newshape_signature(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive.

    Root every intrinsically high-degree qubit. Keep projected check
    hyperedges and low-qubit corners, including endpoint multiplicity.
    No group indices, X/Z colours or protograph labels enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _column_syndromes(rows, n)
    high = [j for j, c in enumerate(columns) if c.bit_count() >= 10]
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

    for j, syndrome in enumerate(columns):
        if j in positions:
            continue
        ends, weights = [], []
        bits = syndrome
        while bits:
            bit = bits & -bits
            r = bit.bit_length() - 1
            ends.extend(check_ends[r])
            weights.append(rows[r].bit_count())
            bits ^= bit
        endpoints.append(tuple(sorted(ends)))
        tags.append((1, tuple(sorted(weights))))

    incident = [[] for _ in high]
    for edge, ends in enumerate(endpoints):
        for v in ends:
            incident[v].append(edge)

    def compress(signatures):
        mapping = {s: i for i, s in enumerate(sorted(set(signatures)))}
        return [mapping[s] for s in signatures]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(4):
            es = [(tag, tuple(sorted(colours[v] for v in ends)))
                  for tag, ends in zip(tags, endpoints)]
            ec = compress(es)
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            colours = compress(vs)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    return (
        n,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(rooted)),
        _tanner_signature(hx, hz),
    )


def _newshape_draw(rng, G, attempt):
    """Twenty support prescriptions, optimized rather than emitted draws."""
    card, epoch = attempt % 20, attempt // 20
    ap, bp = divmod(card, 5)
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    })
    outside = [g for g in directions if g[1] == 1]
    rotations = [g for g in G.elements if g[1] == 0]
    all_labels = list(G.elements)

    def incidence(width, internal, policy):
        C = [[], []]
        # Policy 0 is the quotient-b grading. Other policies include
        # rotation order-six gaps and genuinely mixed relative labels.
        steps = outside if policy == 0 else directions
        if internal > len(steps):
            return None
        for i in range(2):
            for gap in rng.sample(steps, internal):
                C[i].append([G.e, gap])
                C[1 - i].append([])
        cross = width - 2 * internal
        labels = rotations if policy == 0 and internal else all_labels
        if cross > len(labels):
            labels = all_labels
        for voltage in rng.sample(labels, cross):
            C[0].append([G.e])
            C[1].append([voltage])
        return C

    A = incidence(6, ap, epoch % 3)
    if A is None:
        return None

    if bp < 4:
        B = incidence(7, bp, (epoch // 3) % 3)
        if B is None:
            return None
    else:
        # Equal row totals seven, with a singleton and degree-three
        # column. These are not regular incidence relabelings.
        profiles = (
            ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
            ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
            ((1, 0), (1, 2)) + ((1, 1),) * 5,
            ((1, 0), (0, 1), (2, 1), (1, 2))
            + ((1, 1),) * 3,
        )
        profile = profiles[epoch % len(profiles)]
        B = [[], []]
        for counts in profile:
            for i, count in enumerate(counts):
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    gap = rng.choice(directions)
                    cell = [first, G.mul(first, gap)]
                else:
                    cell = rng.sample(all_labels, count)
                B[i].append(cell)

    # Broaden both constituents together while preserving the actual
    # budget. Publication still requires the full logical screen.
    if attempt >= 960 and epoch % 4 == 3:
        profiles_a = (
            ((1, 0), (2, 1)) + ((1, 1),) * 3 + ((0, 2),),
            ((1, 0), (1, 2)) + ((1, 1),) * 4,
        )
        profile = profiles_a[(epoch // 4) % 2]
        A = [[rng.sample(all_labels, c[i]) for c in profile]
             for i in range(2)]
        # Removing one term from each B row permits A degree-three
        # columns: max B-row six plus max A-column three is nine.
        B = incidence(7, 0, 1)
        B[0][-2] = []
        B[1][-1] = []

    sa = [[len(c) for c in row] for row in A]
    sb = [[len(c) for c in row] for row in B]
    if not _no_dead_lines(sa, sb):
        return None
    if _grid_weights(sa, sb) > min(9, W_MAX):
        return None
    assert len(A) == 2 and all(len(row) == 6 for row in A)
    assert len(B) == 2 and all(len(row) == 7 for row in B)
    assert all(len(set(c)) == len(c) <= 5
               for grid in (A, B) for row in grid for c in row)
    return A, B, (card, epoch % 3)


def _newshape_archive_keys(G):
    """Shown exclusions only; these fillings are never candidate outputs."""
    def incidence(group, top, bottom, voltages):
        B = [[], []]
        for gap in top:
            B[0].append([group.e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([group.e, gap])
        for voltage in voltages:
            B[0].append([group.e])
            B[1].append([voltage])
        return B

    # Same dicyclic family, old order: references must be lifted with
    # their own group, not interpreted as new-order element labels.
    def old_mul(g, h):
        return ((g[0] + (h[0] if g[1] == 0 else -h[0])
                 + 6 * g[1] * h[1]) % 12,
                (g[1] + h[1]) % 2)

    def old_inv(g):
        return ((-g[0]) % 12, 0) if g[1] == 0 else ((g[0] + 6) % 12, 1)

    old = _G(dict(
        elements=[(a, b) for b in range(2) for a in range(12)],
        ident=(0, 0), mul=old_mul, inv=old_inv), 24)
    old_A = [[[old.e, (d, 0)] for d in (1, 3, 5)]]
    references = (
        (((1,0),), ((5,0),),
         ((6,0),(0,0),(2,0),(4,0),(10,0))),
        (((5,0),(3,0)), ((5,0),(3,0)),
         ((2,0),(0,0),(4,0))),
        (((3,0),(1,0),(5,0)), ((5,0),(1,0),(3,0)), ((2,0),)),
        ((), (), ((4,0),(9,0),(3,0),(8,0),(5,0),(0,0),(6,0))),
        ((), (), ((0,0),(4,1),(10,1),(0,1),(6,0),(2,0),(10,0))),
        ((), (), ((0,0),(1,0),(2,0),(3,0),(6,0),(7,0),(9,0))),
    )
    keys = set()
    for top, bottom, voltages in references:
        B = incidence(old, top, bottom, voltages)
        hx, hz, n = lifted_matrices(old, old_A, B)
        keys.add(_newshape_signature(hx, hz, n))

    # C3 semidirect C4 with inversion is isomorphic to Dic3.
    # Convert the shown order-12 leaders before exclusion.
    def convert(g):
        a, b = g
        return ((2 * a + 3 * (b // 2)) % 6, b % 2)

    A = [
        [[G.e, convert(g)] for g in ((0,1),(2,1),(1,1))]
        + [[], [], []],
        [[], [], []]
        + [[G.e, convert(g)] for g in ((2,1),(1,1),(0,1))],
    ]
    cards = (
        (((0,1),(1,1)), ((2,1),(1,1)), ((1,2),(2,0),(0,2))),
        ((), (), ((2,3),(0,2),(1,3),(2,2),(1,2),(0,0),(2,0))),
        (((1,1),), ((0,1),), ((1,2),(2,2),(2,0),(0,2),(0,0))),
    )
    for top, bottom, voltages in cards:
        B = incidence(G, tuple(map(convert, top)),
                      tuple(map(convert, bottom)),
                      tuple(map(convert, voltages)))
        hx, hz, n = lifted_matrices(G, A, B)
        keys.add(_newshape_signature(hx, hz, n))
    return keys


from functools import lru_cache


@lru_cache(maxsize=1)
def _newshape_portfolio(G):
    """Deterministic immutable portfolio, independent of slot call order."""
    local = random.Random(BASE_SEED ^ 0xD3122675)
    graph_keys = _newshape_archive_keys(G)
    filling_keys, entries = set(), []
    cap = min(9, W_MAX)
    target_count = max(N_SLOTS, 80)

    def freeze(grid):
        return tuple(tuple(tuple(c) for c in row) for row in grid)

    for batch in range(64):
        candidates = []
        for trial in range(48):
            attempt = 48 * batch + trial
            proposal = _newshape_draw(local, G, attempt)
            if proposal is None:
                continue
            A, B, card = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            ar, br = _newshape_components(G, hx, hz)
            qa = _component_collisions(ar, 6 * G.q)
            qb = _component_collisions(br, 7 * G.q)
            # Constituent short words are diagnostics, not automatic
            # logical-distance claims. Complete quotient tests decide.
            diagnostics = (
                int(qa is None) + int(qb is None),
                (qa or 0) + (qb or 0),
            )
            candidates.append(
                ((-k, diagnostics, attempt),
                 A, B, hx, hz, n, k, rx, rz, card, qa, qb))

        candidates.sort(key=lambda record: record[0])
        survivors = []
        for record in candidates[:20]:
            (_, A, B, hx, hz, n, k, rx, rz, card, qa, qb) = record
            xf = _logical_four_profile(hz, hx, n)
            if xf[0] < 4:
                continue
            zf = _logical_four_profile(hx, hz, n)
            if zf[0] < 4:
                continue
            floor = min(xf[0], zf[0])
            quality = (k * floor * floor, floor, k,
                       -(xf[1] + zf[1]))
            survivors.append((quality, record, xf, zf))

        survivors.sort(key=lambda record: record[0], reverse=True)
        for quality, record, xf, zf in survivors[:8]:
            (_, A, B, hx, hz, n, k, rx, rz, card, qa, qb) = record
            key = _newshape_signature(hx, hz, n)
            if key in graph_keys:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified new-shape lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            graph_keys.add(key)
            screen = (
                ("k", k),
                ("card", card),
                ("logical_profiles", (xf, zf)),
                ("component_collisions", (qa, qb)),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("exact_logical_exclusion_through", min(xf[0], zf[0]) - 1),
            )
            entries.append((quality, freeze(A), freeze(B), screen))

        if len(entries) >= target_count:
            break

    entries.sort(key=lambda entry: entry[0], reverse=True)
    # No recycling or relabelled copies to fill an exhausted portfolio.
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:N_SLOTS])


@lru_cache(maxsize=1)
def _dic3_exhaustive_portfolio(G):
    """Immutable exhaustive topology portfolio, not independent slot draws.

    All matrix screens use scaffold lifts. Algebraic tables below only
    quotient known labeling gauges; they do not implement another lift.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations

    q, e = G.q, G.e
    assert q == 12
    cap = min(9, W_MAX)
    outside = ((0, 1), (1, 1), (2, 1))
    A = [
        [[e, g] for g in outside] + [[], [], []],
        [[], [], []] + [[e, g] for g in outside],
    ]
    sa = [[len(cell) for cell in row] for row in A]
    assert tuple(map(sum, sa)) == (6, 6)

    # An order-three internal edge gives a constituent triangle; an
    # involution gives parallel edge columns. Conservatively omit both.
    directions = tuple(G.idx(g) for g in
                       ((1, 0), (0, 1), (1, 1), (2, 1)))
    multiply = [[G.idx(G.mul(g, h)) for h in G.elements]
                for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(e)

    def edge_class(label):
        return min(label, inverse[label])

    actions = [
        tuple(G.idx(((u * a + v * b) % 6, b))
              for a, b in G.elements)
        for u in (1, 5) for v in range(6)
    ]

    def canonical(top, bottom, ratios):
        """Known common automorphisms, B-row exchange and anchor gauges.

        A's three outside inverse classes are preserved as a set.
        Normalizing a cross matching conjugates the bottom internal
        directions. Equal keys are discarded conservatively.
        """
        best = None
        for action in actions:
            ta = tuple(action[g] for g in top)
            tb = tuple(action[g] for g in bottom)
            rr = tuple(action[g] for g in ratios)
            for first, second, values in (
                    (ta, tb, rr),
                    (tb, ta, tuple(inverse[g] for g in rr))):
                left = tuple(sorted(edge_class(g) for g in first))
                for anchor in values:
                    v = inverse[anchor]
                    right = tuple(sorted(
                        edge_class(multiply[multiply[v][g]][anchor])
                        for g in second))
                    bridges = tuple(sorted(multiply[v][g] for g in values))
                    key = (left, right, bridges)
                    if best is None or key < best:
                        best = key
        return best

    def build_B(top, bottom, ratios):
        B = [[], []]
        for gap in top:
            B[0].append([e, G.elements[gap]])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([e, G.elements[gap]])
        for voltage in ratios:
            B[0].append([e])
            B[1].append([G.elements[voltage]])
        return B

    def incidence_graph(rows, width):
        """Graph recovered from VERIFIED degree-two column syndromes."""
        columns = _column_syndromes(rows, width)
        adjacency = [0] * len(rows)
        edges = set()
        for column in columns:
            if column.bit_count() != 2:
                return None
            bit = column & -column
            a = bit.bit_length() - 1
            b = (column ^ bit).bit_length() - 1
            edge = (min(a, b), max(a, b))
            if edge in edges:
                return None
            edges.add(edge)
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a
        return adjacency

    def graph_key(adjacency):
        """Rooted uncoloured graph invariant, not an isomorphism oracle.

        Retain connected components, rooted colour-refinement histories,
        distance shells and twelve closed-walk counts. Inequality
        certifies non-isomorphism; equality is conservatively rejected.
        """
        size = len(adjacency)
        neighbors = [
            tuple(j for j in range(size) if (mask >> j) & 1)
            for mask in adjacency
        ]
        components = []
        remaining = (1 << size) - 1
        while remaining:
            bit = remaining & -remaining
            root = bit.bit_length() - 1
            visited = {root}
            queue = [root]
            for v in queue:
                for w in neighbors[v]:
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
            components.append(tuple(sorted(visited)))
            for v in visited:
                remaining &= ~(1 << v)

        packets = []
        for component in components:
            position = {v: j for j, v in enumerate(component)}
            local_neighbors = [
                tuple(position[w] for w in neighbors[v])
                for v in component
            ]
            m = len(component)
            rooted = []
            for root in range(m):
                colours = [int(v == root) for v in range(m)]
                history = []
                for _ in range(6):
                    features = [
                        (colours[v],
                         tuple(sorted(colours[w]
                                      for w in local_neighbors[v])))
                        for v in range(m)
                    ]
                    history.append(tuple(sorted(Counter(features).items())))
                    palette = {
                        feature: j
                        for j, feature in enumerate(sorted(set(features)))
                    }
                    colours = [palette[feature] for feature in features]

                distances = [-1] * m
                distances[root] = 0
                queue = [root]
                for v in queue:
                    for w in local_neighbors[v]:
                        if distances[w] < 0:
                            distances[w] = distances[v] + 1
                            queue.append(w)
                shells = tuple(sorted(Counter(distances).items()))

                walks = [int(v == root) for v in range(m)]
                closed = []
                for _ in range(12):
                    walks = [
                        sum(walks[w] for w in local_neighbors[v])
                        for v in range(m)
                    ]
                    closed.append(walks[root])
                packet = (tuple(history), shells, tuple(closed))
                rooted.append(sha256(repr(packet).encode("ascii")).digest())
            packets.append((
                m, tuple(sorted(map(len, local_neighbors))),
                tuple(sorted(rooted)),
            ))
        return tuple(sorted(packets))

    def intrinsic_B_projection(hx, hz, n):
        """Recover B copies without group, sector or X/Z labels.

        In this published support family, degree-13 qubits are exactly
        sector 2, and weight-eight checks are exactly HX. Each such
        check has two high-qubit endpoints, producing two B graphs.
        These properties survive arbitrary full Tanner relabeling.
        """
        rows = list(hx) + list(hz)
        columns = _column_syndromes(rows, n)
        high = [j for j, c in enumerate(columns) if c.bit_count() == 13]
        positions = {j: v for v, j in enumerate(high)}
        adjacency = [0] * len(high)
        edges = set()
        for row in rows:
            if row.bit_count() != 8:
                continue
            ends = []
            bits = row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in positions:
                    ends.append(positions[j])
                bits ^= bit
            if len(ends) != 2:
                raise ValueError("intrinsic B projection lost an endpoint")
            a, b = ends
            edge = (min(a, b), max(a, b))
            if edge in edges:
                raise ValueError("intrinsic B projection has parallel edges")
            edges.add(edge)
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a
        assert len(high) == 4 * q
        return graph_key(adjacency)

    legacy_keys = _newshape_archive_keys(G)
    projected_keys = set()

    # Current-program exclusions, in the frozen Dic3 encoding. A column
    # permutation normalizes each shown high-rank A filling to A above.
    # These references are never published as candidates.
    current_cards = (
        ((), (),
         ((0,1),(3,1),(2,0),(1,0),(1,1),(0,0),(4,1))),
        (((1,1),(0,1)), ((2,1),(0,1)),
         ((3,0),(0,0),(4,0))),
        (((1,1),), ((2,1),),
         ((2,0),(0,0),(3,0),(1,0),(5,0))),
        (((2,1),(0,1),(1,1)), ((0,1),(1,1),(2,1)),
         ((5,0),)),
    )
    for top, bottom, ratios in current_cards:
        B = build_B(tuple(G.idx(g) for g in top),
                    tuple(G.idx(g) for g in bottom),
                    tuple(G.idx(g) for g in ratios))
        hx, hz, n = lifted_matrices(G, A, B)
        legacy_keys.add(_newshape_signature(hx, hz, n))
        projected_keys.add(intrinsic_B_projection(hx, hz, n))

    # Exhaustive prescriptions, with one bridge normalized to identity.
    # Four internal allocations cross five quotient-count prescriptions.
    # Randomness only orders exhaustive representatives; it does not
    # produce independently drawn slot fillings.
    plans = [[] for _ in range(20)]
    orbit_keys = set()
    others = tuple(j for j in range(q) if j != identity)
    for internal in range(4):
        cross = 7 - 2 * internal
        edge_sets = tuple(combinations(directions, internal))
        for top in edge_sets:
            for bottom in edge_sets:
                for rest in combinations(others, cross - 1):
                    ratios = (identity,) + rest
                    key = canonical(top, bottom, ratios)
                    if key in orbit_keys:
                        continue
                    orbit_keys.add(key)
                    outside_count = sum(G.elements[g][1] for g in ratios)
                    policy = outside_count % 5
                    plans[5 * internal + policy].append(
                        (top, bottom, ratios))

    local = random.Random(BASE_SEED ^ 0xD3B258E2)
    for pool in plans:
        local.shuffle(pool)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    entries = []
    target = max(N_SLOTS, 20) + 16
    exhausted = max(map(len, plans), default=0)
    for offset in range(exhausted):
        for card, pool in enumerate(plans):
            if offset >= len(pool):
                continue
            top, bottom, ratios = pool[offset]
            B = build_B(top, bottom, ratios)
            sb = [[len(cell) for cell in row] for row in B]
            assert _no_dead_lines(sa, sb)
            assert tuple(map(sum, sb)) == (7, 7)
            assert all(sb[0][j] + sb[1][j] == 2 for j in range(7))
            assert _grid_weights(sa, sb) == 9 <= cap

            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _newshape_components(G, hx, hz)
            qb = _component_collisions(br, 7 * q)
            if qb is None:
                continue
            graph = incidence_graph(br, 7 * q)
            if graph is None:
                continue
            component_key = graph_key(graph)
            if len(component_key) != 1 or component_key[0][0] != 2 * q:
                continue
            expected_projection = tuple(sorted(component_key * 2))
            if expected_projection in projected_keys:
                continue

            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if n != 552 or k != 258:
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue
            if not (_no_short_kernel(hx, n)
                    and _no_short_kernel(hz, n)):
                continue

            projection = intrinsic_B_projection(hx, hz, n)
            if projection != expected_projection:
                raise ValueError("verified B extraction and projection disagree")
            if projection in projected_keys:
                continue
            signature = _newshape_signature(hx, hz, n)
            if signature in legacy_keys:
                projected_keys.add(projection)
                continue

            xf = _logical_four_profile(hz, hx, n)
            zf = _logical_four_profile(hx, hz, n)
            if min(xf[0], zf[0]) < 4:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified exhaustive Dic3 lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")

            projected_keys.add(projection)
            legacy_keys.add(signature)
            screen = (
                ("k", k),
                ("design_card", (card, offset)),
                ("logical_profiles", (xf, zf)),
                ("B_four_collision_diagnostic", qb),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("full_kernel_weights_1_to_3_excluded", True),
                ("intrinsic_B_projection_distinct", True),
            )
            quality = (k * 16, -(xf[1] + zf[1]), -qb)
            entries.append((quality, card, freeze(A), freeze(B), screen))
        if len(entries) >= target:
            break

    # Spread publication across structural prescriptions, rather than
    # letting equal proxy scores select many nearby labeling policies.
    buckets = [[] for _ in range(20)]
    for entry in entries:
        buckets[entry[1]].append(entry)
    for bucket in buckets:
        bucket.sort(key=lambda entry: entry[0], reverse=True)
    ordered = []
    for position in range(max(map(len, buckets), default=0)):
        for bucket in buckets:
            if position < len(bucket):
                _, _, frozen_A, frozen_B, screen = bucket[position]
                ordered.append((frozen_A, frozen_B, screen))
    return tuple(ordered[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure lookup of an exhaustive, projection-distinct screened design."""
    portfolio = _dic3_exhaustive_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("fresh k258 exhaustive Dic3 topologies exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete canonical-topology cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on the spanning-A Dic6 catalogue escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
# REFUTED: The Dic6 rank-deficit prefix supplies twenty inequivalent initial designs and escapes weight four: slots 0..19 retained seven classes, and all five measured leaders had d30s=4 with abundant sector-1 witnesses.
# REFUTED: Rank-first stabilizer-aware Dic3 support batches supply twenty evaluator-distinct designs or escape distance four: the current sixty-slot table retained twelve classes, with all five measured leaders at d30s=4 and the fifth at k=256 rather than 258.
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
