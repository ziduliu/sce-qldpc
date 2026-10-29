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
CONCEPT = ("Q8=Dic2, order eight, A=B=(3,8): independently optimized saturated constituents with singleton and cubic columns should supply five novel codes with measured distance readings >=7 in cell 0.350, improving its current distance-six portfolio; this is a falsifiable measurement hypothesis.")
GROUP = ("Retain the dicyclic construction rule. Factor 73 excludes order 24 and permits q<=8, so t=2 gives the largest admissible dicyclic lift, Q8. The cyclic order-eight (3,8) bed measured five n=584,k=200,d30s=6 codes. Q8 changes relative constituent connectivity; neither that reading nor element order establishes a transferred distance guarantee.")
SHAPES = ("Use equal A=B=(3,8): each constituent has 64 variables and 24 checks; n=584, kb=200 and guaranteed rate 25/73=0.34247, inside cell 0.350. Parent weight-four minima occupied sector-1 Acol0..2/Bcol0..6 broadly. Three-row constituents and nine sector-2 block pairs replace that bottleneck. Actual verified ranks enforce [0.325,0.375).")
FILLING = ("Twenty cards combine four saturated support prescriptions with five quaternion relative-label policies. Every constituent row has six terms and every column one through three, so BOTH CSS maxima are nine. Verified constituent matrices receive exact kernel screening through six and equal heuristic witness budgets. Independent finalists are paired, tested modulo stabilizers through four, and compared using actual embedded and randomized mixed-sector logical witnesses. Intrinsic uncoloured Tanner invariants conservatively reject repeats and the shown cyclic archive anchor. A high witness reading remains an upper bound; distance-seven survival and unseen-archive novelty are hypotheses.")
FAMILY_NAME = "r035_dic2_q8_38x38_saturated_independent_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic2=Q8 has order eight")
    # Same dicyclic rule: x^(2t)=1, y^2=x^t, y*x*y^-1=x^-1.
    # Factor 73 forces q<=8; t=2 gives q=8.
    t, m = 2, 4
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

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


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


def _unused_dic6_context(rng, G, idx):
    """Historical policy, uncalled on the new group and shapes.

    Reconstruct the same prefix for every slot. No mutable cache, gauge
    lottery, or relaxation of the full-matrix short-word screens.
    """
    from collections import Counter
    from hashlib import sha256

    local = random.Random(BASE_SEED ^ 0xD612B304)
    profiles = _b_profiles()
    cap = min(9, W_MAX)

    # <x^2,y> has order twelve. Its six outside elements form precisely
    # three inverse classes. This is NOT an automorphism of rotation A:
    # automorphisms preserve the characteristic rotation subgroup.
    A = [[[G.e, (a, 1)] for a in (0, 2, 4)]]
    sa = [[2, 2, 2]]
    dummy_hx, _, _ = lifted_matrices(G, A, [[[G.e]]])
    ar = [row & ((1 << (3 * G.q)) - 1) for row in dummy_hx]
    qa = _component_collisions(ar, 3 * G.q)
    if qa is None:
        raise ValueError("Outside-H12 A failed its verified kernel screen")
    a_deficit = G.q - gf2_rank(ar)

    def closure(generators):
        reached, pending = {G.e}, [G.e]
        generators = tuple(generators) + tuple(G.inv(g) for g in generators)
        while pending:
            g = pending.pop()
            for h in generators:
                product = G.mul(g, h)
                if product not in reached:
                    reached.add(product)
                    pending.append(product)
        return sorted(reached)

    shared = closure(((2, 0), (0, 1)))
    alternate = closure(((2, 0), (1, 1)))
    rotations = closure(((1, 0),))
    quaternion = [closure(((3, 0), (a, 1))) for a in range(3)]
    assert len(shared) == len(alternate) == len(rotations) == 12
    assert all(len(pool) == 8 for pool in quaternion)

    def intrinsic_signature(hx, hz, n):
        """Rooted hypergraph invariant of intrinsically identified qubits.

        Degree-13 qubits identify sector 2 in these support profiles.
        Checks become weighted hyperedges on them. Every other qubit
        contributes its incident-check corner, retaining endpoint
        multiplicities. All features survive arbitrary Tanner relabeling.
        Equality is inconclusive and is conservatively rejected.
        """
        rows = list(hx) + list(hz)
        columns = _column_syndromes(rows, n)
        high = [j for j, col in enumerate(columns)
                if col.bit_count() == 13]
        positions = {j: t for t, j in enumerate(high)}
        endpoints, tags = [], []
        check_ends = []

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
            check_ends.append(ends)
            endpoints.append(ends)
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
            # Repeated endpoints are retained in edge signatures;
            # incident multiplicity is retained here as well.
            for v in ends:
                incident[v].append(edge)

        def compress(signatures):
            table = {s: i for i, s in enumerate(sorted(set(signatures)))}
            return [table[s] for s in signatures]

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
            n, tuple(sorted(Counter(row.bit_count() for row in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in columns).items())),
            tuple(sorted(rooted)),
        )

    def signature(hx, hz, n):
        return (intrinsic_signature(hx, hz, n),
                _tanner_signature(hx, hz))

    graph_keys, filling_keys = set(), set()
    old_A = [[[G.e, (d, 0)] for d in (1, 3, 5)]]

    def incidence(top, bottom, voltages):
        B = [[], []]
        for gap in top:
            B[0].append([G.e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([G.e, gap])
        for voltage in voltages:
            B[0].append([G.e])
            B[1].append([voltage])
        return B

    # Exclusion references only: never publish these archived fillings.
    archive_cards = (
        (((1,0),), ((5,0),),
         ((6,0),(0,0),(2,0),(4,0),(10,0))),
        (((5,0),(3,0)), ((5,0),(3,0)),
         ((2,0),(0,0),(4,0))),
        (((3,0),(1,0),(5,0)), ((5,0),(1,0),(3,0)), ((2,0),)),
        ((), (), ((4,0),(9,0),(3,0),(8,0),(5,0),(0,0),(6,0))),
        ((), (), ((0,0),(4,1),(10,1),(0,1),(6,0),(2,0),(10,0))),
        ((), (), ((0,0),(1,0),(2,0),(3,0),(6,0),(7,0),(9,0))),
        (((1,0),), ((5,0),),
         ((0,0),(2,0),(4,0),(6,0),(8,0))),
        (((1,0),(3,0)), ((3,0),(5,0)), ((0,0),(2,0),(6,0))),
        (((1,0),(3,0),(5,0)), ((1,0),(3,0),(5,0)), ((0,0),)),
        (((1,0),(3,0)), ((2,0),(3,0)), ((0,0),(4,0),(8,0))),
    )
    for top, bottom, voltages in archive_cards:
        old_B = incidence(top, bottom, voltages)
        hx, hz, n = lifted_matrices(G, old_A, old_B)
        graph_keys.add(signature(hx, hz, n))

    def draw_B(attempt):
        # Four regular support profiles times five subgroup prescriptions.
        card, epoch = attempt % 20, attempt // 20
        profile_index, mode = card // 5, card % 5
        if attempt >= 1200 and epoch % 3 == 0:
            profile_index = 4 + epoch % 3
        sb = profiles[profile_index]
        if mode == 0:
            labels = shared
        elif mode == 1:
            labels = alternate
        elif mode == 2:
            labels = rotations
        elif mode == 3:
            labels = quaternion[epoch % 3]
        else:
            labels = list(G.elements)

        steps = sorted({
            min(g, G.inv(g)) for g in labels
            if g != G.e and G.mul(g, g) != G.e
        })
        B = [[[] for _ in range(7)] for _ in range(2)]

        if profile_index <= 3:
            internal = [[j for j in range(7)
                         if sb[i][j] == 2 and sb[1-i][j] == 0]
                        for i in range(2)]
            if any(len(js) > len(steps) for js in internal):
                return None
            cross = [j for j in range(7)
                     if sb[0][j] == sb[1][j] == 1]

            # Alternate a sufficient bipartite prescription with broader
            # order-three/six proposals. Verified screening decides the latter.
            graded = bool(internal[0]) and epoch % 3 == 0 and mode in (0, 1)
            edge_pool = ([g for g in steps if g[1] == 1]
                         if graded else steps)
            voltage_pool = ([g for g in labels if g[1] == epoch % 2]
                            if graded else labels)
            if (any(len(js) > len(edge_pool) for js in internal)
                    or len(cross) > len(voltage_pool)):
                return None
            for i in range(2):
                for j, gap in zip(internal[i],
                                  local.sample(edge_pool, len(internal[i]))):
                    B[i][j] = [G.e, gap]
            for j, voltage in zip(cross,
                                  local.sample(voltage_pool, len(cross))):
                B[0][j], B[1][j] = [G.e], [voltage]
        else:
            for i in range(2):
                for j, count in enumerate(sb[i]):
                    if count == 1:
                        B[i][j] = [local.choice(labels)]
                    elif count == 2:
                        first, gap = local.choice(labels), local.choice(steps)
                        B[i][j] = [first, G.mul(first, gap)]
                    elif count:
                        B[i][j] = local.sample(labels, count)

        return B, sb, (profile_index, mode, epoch % 3)

    accepted = 0
    for batch in range(128):
        pool = []
        for trial in range(40):
            attempt = 40 * batch + trial
            proposal = draw_B(attempt)
            if proposal is None:
                continue
            B, sb, card = proposal
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in B for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                raise ValueError("Subgroup card violates support legality")
            if any(len(set(cell)) != len(cell) for row in B for cell in row):
                raise ValueError("Subgroup card contains duplicate elements")

            hx, hz, n = lifted_matrices(G, A, B)
            actual_ar, br = _verified_components(hx, hz, G.q)
            if actual_ar != ar:
                raise ValueError("Verified constituent extraction disagrees")
            qb = _component_collisions(br, 7 * G.q)
            if qb is None:
                continue

            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            # Preserve rank-first selection, but compare optimized pools,
            # rather than immediately publishing the first passing draw.
            priority = (-k, qb, attempt)
            pool.append((priority, B, hx, hz, n, k, rx, rz, qb, card))

        pool.sort(key=lambda record: record[0])
        for _, B, hx, hz, n, k, rx, rz, qb, card in pool:
            if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
                continue
            key = signature(hx, hz, n)
            if key in graph_keys:
                continue
            graph_keys.add(key)

            if accepted == idx:
                if not css_ok(hx, hz):
                    raise ValueError("Verified outside-H12 lift failed CSS")
                if lifted_k(G, A, B) != k:
                    raise ValueError("Verified rank helpers disagree")
                return {
                    "A": A, "B": B, "k": k, "card": card,
                    "component_quality": (qa, qb),
                    "A_rank_deficit": a_deficit,
                    "LP_rank_deficits": (len(hx) - rx, len(hz) - rz),
                    "outside_H12_A": True,
                    "full_kernel_weights_1_to_3_excluded": True,
                }
            accepted += 1

    raise ValueError("Fresh outside-H12 subgroup catalogue exhausted")


def _q8_span(rows):
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


def _q8_component(G, grid, side):
    """Project only from the scaffold's exact left/right construction."""
    dummy = [[[]]]
    width = 8 * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:3 * G.q]]


def _q8_six_profile(rows, width):
    """Exact kernel minimum through six; seven means exclusion through six.

    Publication candidates already exclude weights through four.
    Every returned seed is an actual kernel word.
    """
    from itertools import combinations
    cols = _column_syndromes(rows, width)
    reps = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = reps.get(syndrome)
        if old is not None:
            return (word ^ old).bit_count(), word ^ old, 1
        reps[syndrome] = word
    for i in range(width):
        for j in range(i):
            syndrome = cols[i] ^ cols[j]
            word = (1 << i) | (1 << j)
            old = reps.get(syndrome)
            if old is not None:
                return (word ^ old).bit_count(), word ^ old, 1
            reps[syndrome] = word

    seed, burden = 0, 0
    for i, j, h in combinations(range(width), 3):
        syndrome = cols[i] ^ cols[j] ^ cols[h]
        word = (1 << i) | (1 << j) | (1 << h)
        old = reps.get(syndrome)
        if old is None:
            reps[syndrome] = word
            continue
        difference = word ^ old
        if not difference:
            continue
        weight = difference.bit_count()
        if weight < 6:
            return weight, difference, burden + 1
        if not seed:
            seed = difference
        burden += 1
    return (6 if seed else 7), seed, burden


def _q8_probe(local, rows, width, passes, seed=0, own=None):
    """Heuristic search carrying actual kernel or nontrivial logical words."""
    cols = _column_syndromes(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1
    if best and own is not None and not _q8_outside(best, own):
        best, weight = 0, width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _q8_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        local.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            w = word.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            kernel[i] = word
            inspect(word)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 16:
                    for h in range(j):
                        inspect(pair ^ short[h])

        if best:
            for _ in range(2):
                starting = best
                moves = kernel if own is None else list(rows)
                # Opposite checks are kernel moves only for component
                # searches; full-code stabilizer descent is done below.
                if own is not None:
                    break
                for i, first in enumerate(moves):
                    inspect(starting ^ first)
                    for second in moves[:i]:
                        inspect(starting ^ first ^ second)
                if best == starting:
                    break
        if weight <= 5:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _q8_outside(best, own)
    return best


def _q8_draw(local, G, card):
    """Four support prescriptions times five relative-label prescriptions."""
    family, mode = divmod(card, 5)
    original = (
        (1, 1, 0, 0, 0, 2, 2, 0),
        (0, 1, 1, 1, 1, 0, 0, 2),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )
    if family == 0:
        counts = original
    else:
        degrees = (
            (1, 2, 2, 2, 2, 3, 3, 3),
            (1, 1, 2, 2, 3, 3, 3, 3),
            (1, 1, 1, 3, 3, 3, 3, 3),
        )[family - 1]
        # Exactly six terms per row, with prescribed column degrees.
        tokens = [i for i in range(3) for _ in range(6)]
        local.shuffle(tokens)
        counts = [[0] * 8 for _ in range(3)]
        offset = 0
        for j, degree in enumerate(degrees):
            for i in tokens[offset:offset + degree]:
                counts[i][j] += 1
            offset += degree

    directions = ((1, 0), (0, 1), (1, 1))
    grid = []
    for i in range(3):
        row = []
        for j in range(8):
            count = counts[i][j]
            if not count:
                row.append([])
                continue
            first = G.pick(local)
            if count == 1:
                cell = [first]
            elif count == 2:
                gap = (local.choice(directions) if mode in (0, 4)
                       else directions[mode - 1])
                if local.randrange(2):
                    gap = G.inv(gap)
                cell = [first, G.mul(first, gap)]
            else:
                if mode == 4:
                    # A different cubic triangle: a same-coset pair
                    # and an independently placed outside element.
                    a = local.choice((1, 3))
                    relative = [(a, 0), (local.randrange(4), 1)]
                else:
                    g, h = local.sample(list(directions), 2)
                    g = g if local.randrange(2) else G.inv(g)
                    h = h if local.randrange(2) else G.inv(h)
                    assert G.mul(g, h) != G.mul(h, g)
                    relative = [g, h]
                cell = [first] + [G.mul(first, g) for g in relative]
            assert len(set(cell)) == count <= 3
            row.append(cell)
        grid.append(row)
    assert all(sum(map(len, row)) == 6 for row in grid)
    assert all(1 <= sum(len(grid[i][j]) for i in range(3)) <= 3
               for j in range(8))
    return grid


def _q8_bank(G, side):
    """Independent constituent optimization with equal finalist budgets."""
    bank, fillings = [], set()
    width = 8 * G.q
    for card in range(20):
        local = random.Random(
            BASE_SEED ^ (0x838350 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        raw = []
        for trial in range(256):
            grid = _q8_draw(local, G, card)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _q8_component(G, grid, side)
            collisions = _component_collisions(rows, width)
            if collisions is None or collisions:
                continue
            # Preserve full component rank whenever available.
            rank = gf2_rank(rows)
            cycles = sum(
                (rows[i] & rows[j]).bit_count()
                * ((rows[i] & rows[j]).bit_count() - 1) // 2
                for i in range(len(rows)) for j in range(i))
            raw.append((-rank, cycles, trial, grid, rows))
        raw.sort(key=lambda record: record[:3])

        refined = []
        for neg_rank, cycles, trial, grid, rows in raw[:16]:
            floor, seed, burden = _q8_six_profile(rows, width)
            if floor < 5:
                continue
            word = _q8_probe(local, rows, width, 4, seed)
            if word:
                refined.append(
                    (word.bit_count(), floor, -burden, -cycles,
                     trial, grid, rows, word, -neg_rank))
        refined.sort(key=lambda record: record[:4], reverse=True)

        finalists = []
        for reading, floor, burden, cycles, trial, grid, rows, seed, rank in refined[:4]:
            word = _q8_probe(local, rows, width, 16, seed)
            if word and word.bit_count() >= 5:
                finalists.append(dict(
                    grid=grid, rows=rows, word=word,
                    reading=word.bit_count(), floor=floor,
                    burden=-burden, cycles=-cycles,
                    rank=rank, card=card, trial=trial))
        finalists.sort(key=lambda r: (
            -r["reading"], -r["floor"], r["burden"], r["cycles"]))
        bank.extend(finalists[:3])
    bank.sort(key=lambda r: (
        -r["reading"], -r["floor"], r["burden"],
        r["cycles"], r["card"], r["trial"]))
    return bank


def _q8_low4(checks, stabilizers, width):
    """Exact existence test for nontrivial logicals of weight <=4."""
    own = _q8_span(stabilizers)
    cols = _column_syndromes(checks, width)
    reps = {0: 0}

    def collision(syndrome, word):
        old = reps.get(syndrome)
        if old is None:
            reps[syndrome] = word
            return False
        return _q8_outside(word ^ old, own)

    for i, syndrome in enumerate(cols):
        if collision(syndrome, 1 << i):
            return True
    for i, first in enumerate(cols):
        for j in range(i):
            if collision(first ^ cols[j], (1 << i) | (1 << j)):
                return True
    return False


def _q8_graph_key(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive."""
    from collections import Counter
    from hashlib import sha256
    rows = list(hx) + list(hz)
    cols = _column_syndromes(rows, n)
    corners = []
    for syndrome in cols:
        checks = []
        bits = syndrome
        while bits:
            bit = bits & -bits
            checks.append(bit.bit_length() - 1)
            bits ^= bit
        overlaps = Counter(
            (rows[a] & rows[b]).bit_count()
            for ai, a in enumerate(checks) for b in checks[:ai])
        corners.append((
            len(checks),
            tuple(sorted(rows[r].bit_count() for r in checks)),
            tuple(sorted(overlaps.items()))))
    signature = (
        n, len(rows),
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(corners).items())),
        _tanner_signature(hx, hz),
    )
    return sha256(repr(signature).encode("ascii")).digest()


def _q8_logical_reading(local, G, a, b, hx, hz, n):
    """Explicit logical upper bounds, including randomized mixed sectors."""
    q = G.q
    overall = n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _q8_span(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
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

            # Actual stabilizer moves preserve the logical class.
            for _ in range(2):
                starting = word
                weight = word.bit_count()
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:48]
                for i, first in enumerate(moves):
                    trial = starting ^ first
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            seeds.append(word)

        seed = min(seeds, key=int.bit_count) if seeds else 0
        word = _q8_probe(local, checks, n, 2, seed, own)
        if not word:
            return None
        assert _q8_outside(word, own)
        assert all((row & word).bit_count() % 2 == 0 for row in checks)
        overall = min(overall, word.bit_count())
    return overall


from functools import lru_cache


@lru_cache(maxsize=1)
def _q8_portfolio(G, count):
    """Immutable deterministic portfolio, independent of slot call order."""
    assert G.q == 8
    cap = min(9, W_MAX)
    aa, bb = _q8_bank(G, "A"), _q8_bank(G, "B")
    if not aa or not bb:
        raise ValueError("no small-kernel-screened Q8 constituent bank")

    # Exclusion-only reference for the shown cyclic order-eight leader.
    old = _G(dict(
        elements=list(range(8)), ident=0,
        mul=lambda g, h: (g + h) % 8,
        inv=lambda g: (-g) % 8), 8)
    archive_A = [
        [[0], [0], [], [], [], [0, 5], [0, 6], []],
        [[], [0], [0], [0], [0], [], [], [0, 5]],
        [[], [], [0], [1], [3], [2], [4], [7]],
    ]
    archive_B = [
        [[0], [0], [], [], [], [0, 6], [0, 1], []],
        [[], [0], [0], [0], [0], [], [], [0, 5]],
        [[], [], [0], [1], [3], [2], [7], [7]],
    ]
    ax, az, an = lifted_matrices(old, archive_A, archive_B)
    graph_keys = {_q8_graph_key(ax, az, an)}

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -min(a["floor"], b["floor"]),
         a["burden"] + b["burden"],
         a["cycles"] + b["cycles"], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))
    entries, tested = [], 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for _, _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        assert _no_dead_lines(sa, sb)
        # Six-term rows plus columns of degree at most three.
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(8))
        wz = max(sum(row) + sum(sa[i][j] for i in range(3))
                 for row in sb for j in range(8))
        assert max(wx, wz) == _grid_weights(sa, sb) <= 9
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 584 and k >= 200
                and 325 * n <= 1000 * k < 375 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        key = _q8_graph_key(hx, hz, n)
        if key in graph_keys:
            continue
        graph_keys.add(key)
        tested += 1

        if (_q8_low4(hz, hx, n) or _q8_low4(hx, hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified Q8 saturated lift failed CSS")
        local = random.Random(
            BASE_SEED ^ (0x838CC350 + 65537 * ai + 1009 * bi))
        reading = _q8_logical_reading(local, G, a, b, hx, hz, n)
        if reading is None or reading < 5:
            continue
        if lifted_k(G, A, B) != k:
            raise ValueError("verified Q8 rank helpers disagree")

        quality = (k * reading * reading / n, reading, k,
                   -a["burden"] - b["burden"])
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("component_readings", (a["reading"], b["reading"])),
            ("exact_component_floors_through_six", (a["floor"], b["floor"])),
            ("cards", (a["card"], b["card"])),
            ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("logical_exclusion_through", 4),
        )
        entries.append((quality, freeze(A), freeze(B), screen))
        if len(entries) >= max(count + 20, 80):
            break
        if tested >= max(256, 8 * count):
            break

    entries.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete independently optimized target-rate code."""
    portfolio = _q8_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel target-rate Q8 saturated portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Publish the complete selected cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on the spanning-A Dic6 catalogue escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
# REFUTED: The rotation-A coset-cycle prefix supplies twenty inequivalent initial high-rank designs: current slots 0..19 retained seven classes, and all five measured leaders still had d30s=4 with abundant sector-1 witnesses.
# REFUTED: Replacing rotation A by outside-H12 binomials escapes the distance-four plateau or supplies twenty inequivalent initial designs: the current sixty-slot table retained ten classes, and all five measured leaders had d30s=4 with sector-1 witnesses.
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
