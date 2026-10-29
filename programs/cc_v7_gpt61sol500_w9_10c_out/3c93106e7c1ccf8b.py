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
CONCEPT = ("C11 semidirect C4 with inversion, order 44, A=B=(2,3): joint cubic-constituent and verified-rank optimization should supply five fresh k>=48 codes with distance readings >=24, exceeding the 46.5 incumbent in cell 0.100.")
GROUP = ("Jump from order-12 Dic3 to C11:C4 with inversion, order 44. This different group has a measured (2,3)x(2,3) bed record of 46.5, including k=45 or 48 and d30s=24. Those measurements motivate new relative-label optimization, not a transferred distance certificate.")
SHAPES = ("Equal (2,3) constituents have 132 variables and 88 checks. q=44 gives n=572, kb=44 and guaranteed rate 1/13; the shape permits q<=46. Target cell 0.100 instead of the reported parent's 0.450 plateau. Parent weight-four minima broadly occupied sector-1 Acol/Bcol blocks. Cubic columns replace its incidence components while retaining four sector-2 block pairs. Verified full ranks require k/n<0.125.")
FILLING = ("Optimize twenty prescriptions: four row-five, column-at-most-four supports crossed with five relative quotient patterns. Both CSS weight sums are at most nine, also below twelve. Constituent screens use verified lifts, actual kernel witnesses and equal-budget refinement. Full-code selection uses exact ranks, complete stabilizer-aware logical exclusion through four, embedded nontrivial witnesses and further information-set refinement. Rooted intrinsic uncoloured Tanner signatures conservatively reject repeated outputs and the displayed order-44 frontier representative. Five k=48,d30s=24 readings would score 48.3357; their yield and novelty against the unseen archive remain hypotheses.")
FAMILY_NAME = "r010_c11c4_inversion_23x23_joint_cubic_witness_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C11:C4 with inversion has order 44")
    # x^11=y^4=1; y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(4) for a in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b % 2 == 0 else -c)) % 11,
                (b + d) % 4)

    def inv(g):
        a, b = red(g)
        return ((-a if b % 2 == 0 else a) % 11, (-b) % 4)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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


def _c44_basis(rows):
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


def _c44_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c44_component_rows(G, grid, side):
    """Extract only from the scaffold's verified left/right lift."""
    dummy = [[[G.e]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    mask = (1 << (3 * G.q)) - 1
    return [row & mask for row in rows[:2 * G.q]]


def _c44_probe(rng, rows, width, rounds, seed=0, own=None):
    """Return a verified kernel/logical witness, not a distance floor."""
    columns = _column_syndromes(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1
    order = list(range(width))

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _c44_outside(word, own):
            return
        best, weight = word, w

    for _ in range(rounds):
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

        # Greedy basis exchanges preserve the actual kernel.
        rng.shuffle(nulls)
        for _ in range(2):
            for i, word in enumerate(nulls):
                w = word.bit_count()
                for j, other in enumerate(nulls):
                    if i == j:
                        continue
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w = trial, tw
                nulls[i] = word
                inspect(word)

        short = sorted(nulls, key=int.bit_count)[:20]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 14:
                    for h in range(j):
                        inspect(pair ^ short[h])

        for _ in range(2):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

        if weight <= (12 if own is None else 16):
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _c44_outside(best, own)
    return best


def _c44_draw(rng, G, card):
    """Twenty optimization prescriptions, not twenty published draws."""
    profiles = (
        ((1, 1, 3), (1, 3, 1)),
        ((1, 2, 2), (1, 2, 2)),
        ((0, 2, 3), (2, 2, 1)),
        ((2, 1, 2), (2, 3, 0)),
    )
    profile, policy = divmod(card, 5)
    counts = profiles[profile]
    patterns = ((0, 1), (1, 1), (1, 3), (0, 3), (0, 0))
    pools = [
        [g for g in G.elements if g != G.e and g[1] == b]
        for b in range(4)
    ]
    grid = []
    for i, row in enumerate(counts):
        filled = []
        for j, count in enumerate(row):
            if not count:
                filled.append([])
                continue
            first = G.pick(rng)
            if count == 1:
                cell = [first]
            elif count == 2:
                # Explicit order/quotient pressures vary relative labels.
                b = (policy + i + 2 * j) % 4
                gap = rng.choice(pools[b])
                cell = [first, G.mul(first, gap)]
            else:
                b, d = patterns[policy]
                for _ in range(64):
                    g, h = rng.choice(pools[b]), rng.choice(pools[d])
                    if g == h:
                        continue
                    if policy < 4 and G.mul(g, h) == G.mul(h, g):
                        continue
                    cell = [first, G.mul(first, g), G.mul(first, h)]
                    break
                else:
                    return None
            assert len(set(cell)) == count
            filled.append(cell)
        grid.append(filled)
    assert all(sum(row) == 5 for row in counts)
    assert max(sum(row[j] for row in counts) for j in range(3)) <= 4
    return grid


def _c44_bank(G, side):
    """Equal-budget constituent optimization across all twenty cards."""
    local = random.Random(
        BASE_SEED ^ (0x441323 + 1000003 * int(side == "B")))
    width = 3 * G.q
    buckets = [[] for _ in range(20)]
    fillings = set()
    for attempt in range(1600):
        card = attempt % 20
        grid = _c44_draw(local, G, card)
        if grid is None:
            continue
        key = tuple(tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
        if key in fillings:
            continue
        fillings.add(key)
        rows = _c44_component_rows(G, grid, side)
        rank = gf2_rank(rows)
        if rank < 2 * G.q - 8:
            continue
        if _component_collisions(rows, width) is None:
            continue
        cycles = sum(
            c * (c - 1) // 2
            for i, row in enumerate(rows) for other in rows[:i]
            for c in [(row & other).bit_count()])
        buckets[card].append((cycles, attempt, grid, rows, rank))

    bank = []
    for card, proposals in enumerate(buckets):
        proposals.sort(key=lambda r: (r[0], r[1]))
        rough = []
        for cycles, attempt, grid, rows, rank in proposals[:20]:
            word = _c44_probe(local, rows, width, 4)
            if word and word.bit_count() >= 18:
                rough.append((word.bit_count(), cycles, attempt,
                              grid, rows, rank, word))
        rough.sort(key=lambda r: (-min(r[0], 26), r[1], r[2]))
        refined = []
        for _, cycles, attempt, grid, rows, rank, seed in rough[:6]:
            word = _c44_probe(local, rows, width, 16, seed)
            if word and word.bit_count() >= 18:
                refined.append((word.bit_count(), cycles, attempt,
                                grid, rows, rank, word))
        refined.sort(key=lambda r: (-min(r[0], 26), r[1], r[2]))
        for reading, cycles, attempt, grid, rows, rank, word in refined[:3]:
            frozen = tuple(tuple(tuple(cell) for cell in row) for row in grid)
            bank.append(dict(grid=frozen, rank=rank, word=word,
                             reading=reading, card=card, cycles=cycles))
    bank.sort(key=lambda r: (-min(r["reading"], 26),
                             r["rank"], r["cycles"], r["card"]))
    return bank


def _c44_pair_witness(G, a, b, hx, hz, n, rng):
    """Embed actual constituent witnesses, then search their logical cosets."""
    q, cb = G.q, B_SHAPE[1]
    readings = []
    for kind, checks, stabilizers in (("X", hz, hx), ("Z", hx, hz)):
        own = _c44_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(3):
            if kind == "X":
                word = source << (block * cb * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _c44_outside(word, own):
                continue
            for _ in range(3):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == previous:
                    break
            seeds.append(word)
        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        word = _c44_probe(rng, checks, n, 3, seed, own)
        if not word:
            return None
        readings.append(word.bit_count())
    return min(readings)


@lru_cache(maxsize=1)
def _c44_portfolio(G):
    """Deterministic immutable portfolio; memoization changes cost only."""
    assert G.q == 44 and A_SHAPE == B_SHAPE == (2, 3)
    aa, bb = _c44_bank(G, "A"), _c44_bank(G, "B")
    if not aa or not bb:
        raise ValueError("order-44 constituent bank exhausted")
    cap = min(9, 12, W_MAX)

    # The displayed frontier uses Dic11 coordinates x^a*y^b.
    # x maps to (1,2), y to (0,1) in THIS semidirect presentation.
    def convert(g):
        a, b = g
        return (a % 11, (2 * a + b) % 4)

    reference_A = (
        (((0,1),), ((14,1),), ((18,0),(1,1),(10,1))),
        (((18,1),), ((1,0),(7,1),(14,1)), ((17,0),)),
    )
    reference_B = (
        (((11,0),), ((7,1),), ((21,0),(19,1),(21,1))),
        (((17,0),), ((9,0),(1,1),(21,1)), ((0,0),)),
    )
    def reference(grid):
        return [[[convert(g) for g in cell] for cell in row] for row in grid]
    hx, hz, n = lifted_matrices(G, reference(reference_A),
                               reference(reference_B))
    graph_keys = {_newshape_signature(hx, hz, n)}

    # Interleave twenty support/relative-label pairing prescriptions.
    plans = [[] for _ in range(20)]
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["reading"], b["reading"], 26)
            priority = (weak, 4 * G.q - a["rank"] - b["rank"],
                        -a["cycles"] - b["cycles"], -ai, -bi)
            plans[(a["card"] + 7 * b["card"]) % 20].append(
                (priority, ai, bi))
    for bucket in plans:
        bucket.sort(reverse=True)
    ordered = []
    for turn in range(max(map(len, plans))):
        for card in range(20):
            if turn < len(plans[card]):
                ordered.append(plans[card][turn])

    entries = []
    wanted = max(N_SLOTS + 40, 2 * N_SLOTS)
    tested = 0
    for preliminary, ai, bi in ordered:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(3))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert _no_dead_lines(sa, sb)
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 572 and k >= 44 and 1000 * k < 125 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        tested += 1
        local = random.Random(
            BASE_SEED ^ (0x44CC1323 + 65537 * ai + 1009 * bi))
        reading = _c44_pair_witness(G, a, b, hx, hz, n, local)
        if reading is None or reading < 18:
            continue

        xf = _logical_four_profile(hz, hx, n)
        zf = _logical_four_profile(hx, hz, n)
        if min(xf[0], zf[0]) != 5:
            continue
        signature = _newshape_signature(hx, hz, n)
        if signature in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified order-44 lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified order-44 rank helpers disagree")
        graph_keys.add(signature)

        # Witness weights are upper bounds. Cap optimistic heuristic
        # readings rather than rewarding unresolved high values.
        capped = min(reading, 26)
        quality = (k * capped * capped / n, capped, k, preliminary)
        screen = (
            ("k", k),
            ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("component_ranks", (a["rank"], b["rank"])),
            ("design_cards", (a["card"], b["card"])),
            ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("exact_logical_exclusion_through", 4),
            ("CSS_weight_maxima", (wx, wz)),
        )
        entries.append((quality, A, B, screen))
        if len(entries) >= wanted:
            break
        if tested >= 600 and len(entries) >= N_SLOTS:
            break

    entries.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a jointly optimized order-44 cubic design."""
    portfolio = _c44_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("fresh target-rate order-44 portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell from the screened order-44 portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on the spanning-A Dic6 catalogue escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
# REFUTED: The Dic6 rank-deficit prefix supplies twenty inequivalent initial designs and escapes weight four: slots 0..19 retained seven classes, and all five measured leaders had d30s=4 with abundant sector-1 witnesses.
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
