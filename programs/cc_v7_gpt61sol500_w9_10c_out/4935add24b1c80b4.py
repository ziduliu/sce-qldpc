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
CONCEPT = ("Q16 direct-product C3, order 48, A=(1,2), B=(2,5): cubic A constituents paired with reinforced noncommuting matching cores should supply five novel distance readings >=10 and improve the 0.250 frontier; constituent screens do not certify quantum distance ten.")
GROUP = ("Jump from Dic6 to Q16 x C3. Related listed Q16 beds reached fitness 17.7 and 11.2 on other shapes. This direct product has centre order six and 21 conjugacy classes, unlike the listed order-48 beds. The central C3 factor raises some outside element orders from four to twelve and rotation orders from eight to twenty-four. Verified core girth and dependencies decide whether that algebra helps.")
SHAPES = ("A=(1,2), B=(2,5) gives n=576, kb=144 and guaranteed rate 0.250; factor twelve allows q<=49 and the new group uses q=48. Constituents have 96 variables/48 checks and 240 variables/96 checks. The parent's measured weight-four minima broadly occupied sector-1 Acol0..2/Bcol0..6. Cubic A columns and cubic reinforcement of B replace that incidence-only mechanism. Verified full ranks enforce k=144.")
FILLING = ("Twenty prescriptions vary two reinforcement supports, five relative-order policies and two core-girth tiers. A columns have three terms; B has three matching columns and two cubic columns. Both constituent row totals are six and both CSS weight maxima are nine. Exact A screening excludes kernels through seven; exact graph joins give B floors seven or eight. Explicit constituent and non-stabilizer logical probes supply upper bounds used only for selection. Complete matrices exclude kernels through four, satisfy CSS and have conservatively separated intrinsic Tanner invariants. An immutable memoized portfolio changes build cost, not slot choices; longer quantum distance and unseen archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r025_q16xc3_q48_12x25_cubic_exactjoin_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q16 x C3 has order 48")
    # x^8=1, y^2=x^4, y*x*y^-1=x^-1; z is central of order three.
    # Canonical element x^a y^b z^c.
    els = [(a, b, c) for c in range(3)
           for b in range(2) for a in range(8)]

    def red(g):
        a, b, c = g
        return ((a + 4 * (b // 2)) % 8, b % 2, c % 3)

    def mul(g, h):
        a, b, c = red(g)
        d, f, t = red(h)
        return ((a + (d if b == 0 else -d) + 4 * b * f) % 8,
                (b + f) % 2, (c + t) % 3)

    def inv(g):
        a, b, c = red(g)
        return ((-a) % 8, 0, (-c) % 3) if b == 0 else \
               ((a + 4) % 8, 1, (-c) % 3)

    def idx(g):
        a, b, c = red(g)
        return 16 * c + 8 * b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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
    """Uncalled historical outside-H12 catalogue; never publishes cells.

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


from functools import lru_cache


def _q48_rows(G, grid, right=False):
    """Expose a constituent using only the verified scaffold lift."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    return [row & mask for row in rows], width


def _q48_pairs(columns):
    """Exact exclusion of every kernel support of weights one through four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = {}
    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs[syndrome] = (j, i)
    return pairs


def _q48_cubic_clean(rows, width):
    """All columns are cubic: parity excludes odd kernel weights.

    Pair and triple collisions therefore exclude every weight <=7.
    This is a constituent certificate, not a quantum-distance claim.
    """
    from itertools import combinations
    columns = _column_syndromes(rows, width)
    if any(c.bit_count() != 3 for c in columns):
        return False
    if _q48_pairs(columns) is None:
        return False
    triples = set()
    for a, b, c in combinations(range(width), 3):
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        if syndrome in triples:
            return False
        triples.add(syndrome)
    return True


def _q48_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _q48_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _q48_probe(rows, width, rng, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, hence only an upper bound."""
    columns = _column_syndromes(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if not word or word.bit_count() >= weight:
            return
        if own is not None and not _q48_outside(word, own):
            return
        best, weight = word, word.bit_count()

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        short = sorted(kernel, key=int.bit_count)[:32]
        for i, first in enumerate(short):
            for second in short[:i]:
                inspect(first ^ second)
        for i in range(len(short)):
            for other in short:
                trial = short[i] ^ other
                if trial and trial.bit_count() < short[i].bit_count():
                    short[i] = trial
            inspect(short[i])
        short.sort(key=int.bit_count)
        for i in range(min(16, len(short))):
            for j in range(i):
                for h in range(j):
                    inspect(short[i] ^ short[j] ^ short[h])

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _q48_outside(best, own)
    return best


def _q48_core(G, g, h):
    """Verified cubic matching graph, all-pairs paths and shortest cycle."""
    grid = [[[G.e], [G.e], [G.e]],
            [[G.e], [g], [h]]]
    rows, width = _q48_rows(G, grid)
    columns = _column_syndromes(rows, width)
    adjacency = [[] for _ in rows]
    edges = set()
    for j, column in enumerate(columns):
        if column.bit_count() != 2:
            return None
        bit = column & -column
        a = bit.bit_length() - 1
        b = (column ^ bit).bit_length() - 1
        edge = (min(a, b), max(a, b))
        if edge in edges:
            return None
        edges.add(edge)
        adjacency[a].append((b, j))
        adjacency[b].append((a, j))

    distances, paths, cycle = [], [], 0
    for root in range(len(rows)):
        dist, path = [10000] * len(rows), [0] * len(rows)
        dist[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if dist[b] == 10000:
                    dist[b] = dist[a] + 1
                    path[b] = path[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = path[a] ^ path[b] ^ (1 << edge)
                    if word and (not cycle or
                                 word.bit_count() < cycle.bit_count()):
                        cycle = word
        if cycle and cycle.bit_count() < 8:
            return None
        distances.append(tuple(dist))
        paths.append(tuple(path))

    if not cycle:
        return None
    assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
    return (g, h, tuple(columns), tuple(distances),
            tuple(paths), cycle)


def _q48_join(rows, width, G, core):
    """Exact constituent screening through six, then through seven.

    The three core blocks are degree-two edges; the last two blocks
    are cubic. Kernel supports contain an even number of cubic columns.
    Regular constituent symmetry permits identity-coordinate anchors.
    No such symmetry is assumed for the full quantum matrices.
    """
    from itertools import combinations
    q = G.q
    columns = _column_syndromes(rows, width)
    if tuple(columns[:3*q]) != core[2]:
        raise ValueError("reinforcement changed the verified matching core")
    if _q48_pairs(columns) is None:
        return None
    cubic = list(range(3*q, width))
    if any(columns[j].bit_count() != 3 for j in cubic):
        raise ValueError("join model requires cubic reinforcement")
    anchors = (3*q + G.idx(G.e), 4*q + G.idx(G.e))
    distances, paths = core[3], core[4]
    best = core[5]
    memo = {}

    def tjoin(boundary, budget):
        if not boundary:
            return (0, 0)
        if budget < 0 or boundary.bit_count() > 2 * budget:
            return None
        key = (boundary, budget)
        if key in memo:
            return memo[key]
        bit = boundary & -boundary
        a = bit.bit_length() - 1
        rest, choices = boundary ^ bit, boundary ^ bit
        answer = None
        while choices:
            other = choices & -choices
            b = other.bit_length() - 1
            length = distances[a][b]
            if length <= budget:
                tail = tjoin(rest ^ other, budget - length)
                if tail is not None:
                    candidate = (length + tail[0],
                                 paths[a][b] ^ tail[1])
                    if answer is None or candidate[0] < answer[0]:
                        answer = candidate
            choices ^= other
        memo[key] = answer
        return answer

    def retain(word):
        nonlocal best
        if word and word.bit_count() < best.bit_count():
            best = word
        return best.bit_count() >= 7

    # Two cubic variables and at most five graph edges.
    for anchor in anchors:
        for other in cubic:
            if other == anchor:
                continue
            result = tjoin(columns[anchor] ^ columns[other], 5)
            if result is not None:
                word = (1 << anchor) ^ (1 << other) ^ result[1]
                if not retain(word):
                    return None

    # Pure-cubic dependencies through six.
    triples = {}
    for a, b, c in combinations(cubic, 3):
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        word = (1 << a) | (1 << b) | (1 << c)
        if syndrome in triples:
            return None
        triples[syndrome] = word

    # Four cubic variables and at most three graph edges.
    for anchor in anchors:
        others = [j for j in cubic if j != anchor]
        for a, b, c in combinations(others, 3):
            boundary = (columns[anchor] ^ columns[a]
                        ^ columns[b] ^ columns[c])
            budget = 2 if best.bit_count() == 7 else 3
            if boundary.bit_count() > 2 * budget:
                continue
            result = tjoin(boundary, budget)
            if result is not None:
                word = ((1 << anchor) | (1 << a)
                        | (1 << b) | (1 << c)) ^ result[1]
                if not retain(word):
                    return None

    if best.bit_count() == 7:
        return (7, best)

    # Six cubic variables and one graph edge: the only remaining <=7 case.
    for anchor in anchors:
        others = [j for j in cubic if j != anchor]
        for a, b in combinations(others, 2):
            syndrome = columns[anchor] ^ columns[a] ^ columns[b]
            word = (1 << anchor) | (1 << a) | (1 << b)
            for edge, edge_syndrome in enumerate(columns[:3*q]):
                previous = triples.get(syndrome ^ edge_syndrome)
                if previous is not None:
                    witness = word ^ previous ^ (1 << edge)
                    if witness and witness.bit_count() <= 7:
                        assert witness.bit_count() == 7
                        assert all((row & witness).bit_count() % 2 == 0
                                   for row in rows)
                        return (7, witness)

    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return (8, best)


def _q48_key(hx, hz, n):
    """Conservative intrinsic full-Tanner invariant, without X/Z labels."""
    from collections import Counter
    from hashlib import sha256
    rows = list(hx) + list(hz)
    columns = _column_syndromes(rows, n)
    high = [j for j, column in enumerate(columns)
            if column.bit_count() == 12]
    position = {j: v for v, j in enumerate(high)}
    edges, tags = [], []
    incident = [[] for _ in high]
    for row in rows:
        ends = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in position:
                ends.append(position[j])
            bits ^= bit
        edge = len(edges)
        edges.append(tuple(ends))
        tags.append(row.bit_count())
        for v in ends:
            incident[v].append(edge)

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(5):
            es = [(tag, tuple(sorted(colours[v] for v in ends)))
                  for tag, ends in zip(tags, edges)]
            palette = {s: i for i, s in enumerate(sorted(set(es)))}
            ec = [palette[s] for s in es]
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            palette = {s: i for i, s in enumerate(sorted(set(vs)))}
            colours = [palette[s] for s in vs]
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    return (n, tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in columns).items())),
            tuple(sorted(rooted)), _tanner_signature(hx, hz))


@lru_cache(maxsize=1)
def _q48_portfolio(G, count):
    """Pure deterministic portfolio; cached published grids are immutable."""
    from itertools import combinations
    q, e = G.q, G.e
    if q != 48:
        raise ValueError("quaternion-product portfolio requires order 48")
    local = random.Random(BASE_SEED ^ 0x481225)
    nonidentity = [g for g in G.elements if g != e]

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid group element order")

    orders = {g: order(g) for g in G.elements}
    policies = (
        [g for g in nonidentity if orders[g] == 24],
        [g for g in nonidentity if orders[g] == 12],
        [g for g in nonidentity if orders[g] in (8, 24)],
        [g for g in nonidentity if orders[g] in (4, 12)],
        nonidentity,
    )
    policies = [pool or nonidentity for pool in policies]

    # Enumerate genuine matching cores before reinforcement proposals.
    descriptors = []
    for g, h in combinations(nonidentity, 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        data = _q48_core(G, g, h)
        if data is not None:
            descriptors.append((data[5].bit_count(), g, h))
    if not descriptors:
        raise ValueError("no verified matching core of girth at least eight")
    descriptors.sort(reverse=True)
    cores = [_q48_core(G, g, h)
             for _, g, h in descriptors[:64]]
    high_cores = [c for c in cores if c[5].bit_count() >= 10] or cores
    ordinary_cores = [c for c in cores if c[5].bit_count() == 8] or cores

    abank, bbank = [], []
    akeys, bkeys, fillings = set(), set(), set()
    goal = max(24, min(80, 2 * count))

    def inverse_word(word):
        result = 0
        while word:
            bit = word & -word
            j, t = divmod(bit.bit_length() - 1, q)
            result |= 1 << (j*q + G.idx(G.inv(G.elements[t])))
            word ^= bit
        return result

    for attempt in range(640):
        card = attempt % 20
        profile, remainder = divmod(card, 10)
        tier, policy = divmod(remainder, 5)
        pool = policies[policy]
        rng = random.Random(BASE_SEED ^ (0x48A000 + 104729 * attempt))

        if len(abank) < goal:
            A = [[[] for _ in range(2)]]
            for j in range(2):
                options = policies[(policy + j) % 5]
                g, h = rng.sample(options, 2)
                A[0][j] = [e, g, h]
            filling = ("A", tuple(tuple(sorted(G.idx(g) for g in cell))
                                   for cell in A[0]))
            if filling not in fillings:
                fillings.add(filling)
                ar, aw = _q48_rows(G, A)
                if gf2_rank(ar) == q and _q48_cubic_clean(ar, aw):
                    key = _tanner_signature(ar, [])
                    if key not in akeys:
                        witness = _q48_probe(ar, aw, rng, 10)
                        if witness:
                            akeys.add(key)
                            abank.append((witness.bit_count(), card, attempt,
                                          A, witness))

        if len(bbank) < goal:
            core = rng.choice(high_cores if tier else ordinary_cores)
            g, h = core[:2]
            B = [[[e], [e], [e], [], []],
                 [[e], [g], [h], [], []]]
            if profile == 0:
                a, b = G.pick(rng), G.pick(rng)
                r, s = rng.choice(pool), rng.choice(policies[(policy+1) % 5])
                B[0][3], B[1][3] = [a, G.mul(a, r)], [e]
                B[0][4], B[1][4] = [e], [b, G.mul(b, s)]
            else:
                r, s = rng.sample(pool, 2)
                u, v = rng.sample(policies[(policy+1) % 5], 2)
                B[0][3], B[1][4] = [e, r, s], [e, u, v]

            filling = ("B", tuple(tuple(sorted(G.idx(g) for g in cell))
                                   for row in B for cell in row))
            if filling not in fillings:
                fillings.add(filling)
                br, bw = _q48_rows(G, B)
                if gf2_rank(br) == 2*q:
                    screened = _q48_join(br, bw, G, core)
                    if screened is not None:
                        floor, seed = screened
                        key = _tanner_signature(br, [])
                        if key not in bkeys:
                            witness = _q48_probe(br, bw, rng, 10, seed)
                            if witness and witness.bit_count() >= floor:
                                right, _ = _q48_rows(G, B, True)
                                rw = inverse_word(witness)
                                assert all((row & rw).bit_count() % 2 == 0
                                           for row in right)
                                bkeys.add(key)
                                bbank.append((witness.bit_count(), floor,
                                              card, attempt, B, rw))
        if len(abank) >= goal and len(bbank) >= goal:
            break

    if len(abank) < 4 or len(bbank) < 4:
        raise ValueError("too few screened cubic/reinforced constituents")
    abank.sort(key=lambda r: (-r[0], r[1], r[2]))
    bbank.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
    plans = sorted(
        (-min(a[0], b[0]), -b[1], ai, bi)
        for ai, a in enumerate(abank)
        for bi, b in enumerate(bbank))

    def reduce_word(word, stabilizers):
        for _ in range(4):
            start = word
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            if word == start:
                break
        # A bounded exact pair sweep can expose cancellation missed by
        # single-generator descent. Every tested word is still logical.
        start, weight = word, word.bit_count()
        for i, first in enumerate(stabilizers):
            for second in stabilizers[:i]:
                trial = start ^ first ^ second
                if trial.bit_count() < weight:
                    word, weight = trial, trial.bit_count()
        return word

    graph_keys, results = set(), []
    tested = 0
    for _, _, ai, bi in plans:
        a, b = abank[ai], bbank[bi]
        A, B = a[3], b[4]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert all(sum(row) == 6 for row in sa + sb)
        assert _no_dead_lines(sa, sb)
        # HX: A-row six plus B-column at most three.
        # HZ: B-row six plus A-column three.
        assert _grid_weights(sa, sb) == 9
        assert _grid_weights(sa, sb) <= min(9, 12, W_MAX)
        assert all(len(cell) <= 5 and len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or k != 144:
            continue
        if _q48_pairs(_column_syndromes(hx, n)) is None:
            continue
        if _q48_pairs(_column_syndromes(hz, n)) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quaternion-product lift failed CSS")

        bx, bz = _q48_basis(hx), _q48_basis(hz)
        xseed, zseed = 0, 0
        for block in range(2):
            word = b[5] << (block * 5*q)
            assert all((row & word).bit_count() % 2 == 0 for row in hz)
            if _q48_outside(word, bx):
                word = reduce_word(word, hx)
                if not xseed or word.bit_count() < xseed.bit_count():
                    xseed = word
        for block in range(5):
            word, source = 0, a[4]
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length() - 1, q)
                word |= 1 << ((j*5 + block)*q + t)
                source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in hx)
            if _q48_outside(word, bz):
                word = reduce_word(word, hz)
                if not zseed or word.bit_count() < zseed.bit_count():
                    zseed = word
        if not xseed or not zseed:
            continue

        tested += 1
        rng = random.Random(BASE_SEED ^ (0x48C000 + 257*ai + bi))
        xword = _q48_probe(hz, n, rng, 2, xseed, bx)
        zword = _q48_probe(hx, n, rng, 2, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 7:
            continue
        key = _q48_key(hx, hz, n)
        if key in graph_keys:
            continue
        graph_keys.add(key)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, b[1],
                   min(a[0], b[0]), -ai, -bi)
        results.append((quality, freeze(A), freeze(B), k, reading,
                        (a[1], b[2]), (8, b[1])))
        if len(results) >= count:
            break
        if tested >= max(240, 8 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Publish an invariant-separated optimized design without resampling."""
    catalogue = _q48_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("quaternion-product screened portfolio exhausted")
    quality, A, B, k, reading, cards, floors = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "constituent_kernel_floors": floors,
            "design_cards": cards, "selection_quality": quality}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell from the immutable selected filling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on the spanning-A Dic6 catalogue escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
# REFUTED: The rotation-A coset-cycle prefix supplies twenty inequivalent initial high-rank designs: current slots 0..19 retained seven classes, and all five measured leaders still had d30s=4 with abundant sector-1 witnesses.
# REFUTED: Replacing rotation A by the outside-H12 inverse classes escapes the distance-four plateau: the current sixty-slot evaluation retained ten distinct codes, all with d2s=4, and all five measured leaders still had d30s=4.
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
