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
CONCEPT = ("Dic2=Q8, order 8, A=B=(2,7): coupling triangle-free regular incidence constituents and singly grounded constituents should supply five novel distance-four-or-better measured codes in cell 0.500.")
GROUP = ("Keep the dicyclic construction rule, repairing only its parameter from Dic7 to Dic2. Size factor 53 requires q<=11, so order eight is the largest admissible dicyclic lift. The parent measured five distance-three codes and sixteen distinct classes among sixty slots; changing constituent geometry addresses that plateau. This Q8 shape bed is absent from the reported BEDS list.")
SHAPES = ("Equal constituents have 56 variables and 16 checks, n=424 and kb=200, guaranteed rate 25/53=0.471698. Actual rank deficits select cell 0.500: connected regular/regular pairs are expected at k=212, and regular/grounded pairs at k=205, but every output must pass verified actual-rank screening. Parent minima occupied sector-1 Acol/Bcol blocks, especially Bcol0 for X and Acol0 for Z; the new complete-matrix screen excludes commuting weights one through three.")
FILLING = ("Exhaustively enumerate seven-matching and balanced internal-cycle regular cards, plus six-matchings with one grounding singleton. Cells have at most two distinct terms, rows at most seven and columns at most two, giving BOTH CSS maxima nine, also below twelve. Both verified constituent actions must exclude kernel weights one through three. Complete verified lifts must satisfy CSS, the actual target rate and the same exclusion in both sectors. Intrinsic check-intersection and high-degree-qubit joint-walk invariants conservatively separate an immutable catalogue. Prefer rank surplus; larger distance and novelty against the unseen archive remain measurement hypotheses.")
FAMILY_NAME = "r050_dic2_27x27_regular_grounded_kernel3_catalog"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic2=Q8 requires order eight")
    # Same dicyclic rule: x^(2m)=1, y^2=x^m, y*x*y^-1=x^-1.
    # Here m=2; (a,b) represents x^a y^b.
    m = 2
    modulus = 2*m
    els = [(a, b) for b in range(2) for a in range(modulus)]

    def red(g):
        a, b = g
        return ((a + m*(b // 2)) % modulus, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + m*b*d) % modulus,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a + m) % modulus, b)

    def idx(g):
        a, b = red(g)
        return modulus*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


_DIC_PROFILES = (
    (4, 4, 4), (4, 4, 7), (4, 4, 14), (4, 7, 7),
    (4, 7, 14), (4, 14, 14), (7, 7, 7), (7, 7, 14),
    (7, 14, 14), (14, 14, 14),
)
_DIC_PRIMARY = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 4), (0, 9), (1, 5), (1, 8), (2, 3),
    (2, 7), (3, 5), (4, 6), (5, 8), (6, 9),
)
_DIC_CARDS = _DIC_PRIMARY + tuple(
    (a, b) for a in range(10) for b in range(a, 10)
    if (a, b) not in _DIC_PRIMARY
)


def _dic_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid finite-group order")


def _dic_component(rng, G, orders, pools):
    """A fixed monomial pivot plus three prescribed-order binomials."""
    row = [[G.e]]
    used = set()
    for order in orders:
        available = [h for h in pools[order]
                     if h not in used and G.inv(h) not in used]
        if not available:
            raise ValueError("relative-order card exhausted")
        h = rng.choice(available)
        used.add(h)
        g = G.pick(rng)
        row.append([g, G.mul(g, h)])
    return [row]


def _dic_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _dic_excludes_two(rows, n):
    """Complete exclusion of commuting supports of weights one and two."""
    columns = _dic_columns(rows, n)
    return 0 not in columns and len(set(columns)) == n


def _dic_three_witness(G, B):
    """X witness in Acol1: one B-binomial bit and two B-pivot bits.

    B[0][0]=identity. For binomial labels u,v, selecting its column
    at identity has check syndromes at u^-1 and v^-1, canceled by
    those two columns of the monomial pivot.
    """
    q = G.q
    u, v = B[0][1]
    base = 4*q                  # Acol1/Bcol0
    return ((1 << (base + q + G.idx(G.e)))
            | (1 << (base + G.idx(G.inv(u))))
            | (1 << (base + G.idx(G.inv(v)))))


def _extract_components(G, hx, hz):
    """Extract components from the verified lift, not a separate lift."""
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
    # j=0 HZ rows already have consecutive B-column blocks.
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _component_probe(rows, n, rng):
    """Exact search through weight 4, then heuristic kernel probes."""
    from itertools import combinations

    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2

    pairs = set()
    found4 = False
    for i in range(n):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    if found4:
        return 4

    best = n + 1
    for _ in range(4):
        order = list(range(n))
        rng.shuffle(order)
        image = [1 << order[c] for c in range(n)]
        pivots = {}
        for original in rows:
            row = 0
            bits = original
            while bits:
                bit = bits & -bits
                row ^= image[bit.bit_length() - 1]
                bits ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ascending = sorted(pivots)
        basis = []
        for f in range(n):
            if f in pivots:
                continue
            word = 1 << f
            for p in ascending:
                if (pivots[p] & word).bit_count() & 1:
                    word ^= 1 << p
            basis.append(word)

        for word in basis:
            best = min(best, word.bit_count())
        for a, b in combinations(basis, 2):
            best = min(best, (a ^ b).bit_count())
        for a, b, c in combinations(basis, 3):
            best = min(best, (a ^ b ^ c).bit_count())
        if best <= 5:
            break
    return best


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _retired_dic7_context(rng, G, idx):
    """Inactive historical one-row catalogue; never fills the new shapes."""
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def signature(hx, hz, n):
        """Necessary Tanner-isomorphism invariant, deliberately uncoloured.

        Check intersections retain their exact multiplicities. Root
        profiles record common-neighbour counts and distance spheres
        separately for each intersection multiplicity. Different
        signatures prove non-isomorphism; equality proves nothing.
        """
        rows = hx + hz
        m = len(rows)
        relations = [[0] * m for _ in range(10)]
        overlap = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i):
                c = (rows[i] & rows[j]).bit_count()
                overlap[i][j] = overlap[j][i] = c
                if c:
                    relations[c][i] |= 1 << j
                    relations[c][j] |= 1 << i
        active = [r for r in relations[1:] if any(r)]

        profiles = [{} for _ in rows]
        for i in range(m):
            for j in range(i):
                common = []
                for a, relation in enumerate(active):
                    for b in range(a + 1):
                        other = active[b]
                        value = (relation[i] & other[j]).bit_count()
                        if a != b:
                            value += (other[i] & relation[j]).bit_count()
                        common.append(value)
                key = (overlap[i][j], tuple(common))
                for p in (profiles[i], profiles[j]):
                    p[key] = p.get(key, 0) + 1

        roots = []
        for root in range(m):
            spheres = []
            for relation in active:
                visited = frontier = 1 << root
                shell = []
                while frontier:
                    neighbors = 0
                    bits = frontier
                    while bits:
                        bit = bits & -bits
                        neighbors |= relation[bit.bit_length() - 1]
                        bits ^= bit
                    frontier = neighbors & ~visited
                    if frontier:
                        shell.append(frontier.bit_count())
                        visited |= frontier
                spheres.append(tuple(shell))
            roots.append((rows[root].bit_count(),
                          tuple(sorted(profiles[root].items())),
                          tuple(spheres)))
        degrees = [0] * n
        for row in rows:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return (tuple(sorted(degrees)), tuple(sorted(roots)))

    def cells(steps):
        return [[[G.e]] + [[G.e, h] for h in steps]]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    def orbit_key(a, b):
        """Common Dic7 automorphisms, step inversion and side interchange."""
        keys = []
        for u in (1, 3, 5, 9, 11, 13):
            for v in range(14):
                pair = []
                for steps in (a, b):
                    transformed = [
                        ((u * g[0] + v * g[1]) % 14, g[1])
                        for g in steps
                    ]
                    pair.append(tuple(sorted(inverse_class(g)
                                             for g in transformed)))
                keys.append(tuple(sorted(pair)))
        return min(keys)

    # The thirteen distinct parent codes, in the frozen element ordering.
    # Rejecting their signatures also rejects every isomorphic image.
    archived = (
        ((5,25),(1,17),(9,26),(3,22),(7,20),(5,14)),
        ((1,21),(0,14),(21,27),(6,22),(6,21),(14,26)),
        ((1,26),(7,14),(18,19),(10,21),(12,15),(16,19)),
        ((13,25),(15,27),(17,27),(11,23),(19,25),(16,26)),
        ((6,21),(15,21),(8,11),(9,21),(11,13),(2,11)),
        ((24,26),(16,22),(0,4),(15,17),(6,10),(2,10)),
        ((0,8),(0,2),(0,9),(15,25),(3,11),(14,27)),
        ((8,20),(1,17),(13,23),(4,21),(16,20),(16,21)),
        ((13,17),(9,22),(2,18),(26,27),(1,12),(4,9)),
        ((9,18),(5,24),(20,24),(10,19),(0,5),(1,4)),
        ((7,15),(7,19),(13,15),(9,22),(3,22),(15,23)),
        ((12,26),(5,7),(17,25),(2,8),(0,2),(19,23)),
        ((8,18),(0,15),(12,16),(12,23),(11,15),(12,21)),
    )
    blocked = set()
    for record in archived:
        A = [[[G.e]] + [[G.elements[a], G.elements[b]]
                       for a, b in record[:3]]]
        B = [[[G.e]] + [[G.elements[a], G.elements[b]]
                       for a, b in record[3:]]]
        hx, hz, n = lifted_matrices(G, A, B)
        blocked.add(signature(hx, hz, n))

    pools = {
        order: [g for g in G.elements
                if _dic_order(G, g) == order
                and G.idx(g) == inverse_class(g)]
        for order in (4, 7, 14)
    }

    def draw_steps(local, orders):
        steps = []
        for order in orders:
            available = [g for g in pools[order] if g not in steps]
            steps.append(local.choice(available))
        return tuple(sorted(steps, key=G.idx))

    # Every slot rebuilds the same accepted prefix. Thus invariant
    # separation is enforced across slots without mutable shared state.
    local = random.Random(BASE_SEED ^ 0xD1C7C3)
    seen_orbits = set()
    accepted = 0
    for attempt in range(8192):
        pa, pb = _DIC_CARDS[attempt % len(_DIC_CARDS)]
        a = draw_steps(local, _DIC_PROFILES[pa])
        b = draw_steps(local, _DIC_PROFILES[pb])
        orbit = orbit_key(a, b)
        if orbit in seen_orbits:
            continue
        seen_orbits.add(orbit)

        A, B = cells(a), cells(b)
        hx, hz, n = lifted_matrices(G, A, B)
        if not (_dic_excludes_two(hx, n)
                and _dic_excludes_two(hz, n)):
            continue
        graph = signature(hx, hz, n)
        if graph in blocked:
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 9 * G.q:
            continue
        witness = _dic_three_witness(G, B)
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue

        blocked.add(graph)
        if accepted == idx:
            assert css_ok(hx, hz)
            return {"A": A, "B": B, "k": k,
                    "distance_certificate": 3,
                    "card": (pa, pb), "witness": witness}
        accepted += 1

    raise ValueError("new invariant-separated distance-three catalogue exhausted")


from functools import lru_cache


def _q8_clear_three(rows, width):
    """Exact exclusion of all nonzero kernel supports of weights 1..3."""
    cols = _dic_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    for i, first in enumerate(cols):
        for j in range(i):
            if (first ^ cols[j]) in singles:
                return False
    return True


def _q8_graph_key(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Unequal keys certify non-isomorphism; equality is inconclusive.
    No CSS colours, group labels or protograph coordinates enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _dic_columns(rows, width)
    m = len(rows)
    one, many = [0]*m, [0]*m
    overlaps = []
    row_colors = []
    for row in rows:
        degrees = Counter()
        bits = row
        while bits:
            bit = bits & -bits
            degrees[cols[bit.bit_length()-1].bit_count()] += 1
            bits ^= bit
        row_colors.append((row.bit_count(), tuple(sorted(degrees.items()))))

    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps.append((i, j, overlap))
            if overlap:
                relation = one if overlap == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i, j, overlap in overlaps:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(row_colors[j], overlap, common)] += 1
        profiles[j][(row_colors[i], overlap, common)] += 1
    intersections = tuple(sorted(
        (row_colors[i], tuple(sorted(profiles[i].items())))
        for i in range(m)))

    # All sector-one qubits have degree <=4. Sector-two qubits have
    # degree 12..14, so this class is identifiable without block labels.
    high = [j for j, col in enumerate(cols) if col.bit_count() >= 12]
    if len(high) != 32:
        raise ValueError("unexpected intrinsic high-degree sector")
    positions = {j: i for i, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]
    anchors = [Counter() for _ in high]
    edge_colors = [Counter() for _ in high]

    for row in rows:
        bits = row & mask
        endpoints = []
        while bits:
            bit = bits & -bits
            endpoints.append(positions[bit.bit_length()-1])
            bits ^= bit
        weight = row.bit_count()
        if len(endpoints) == 1:
            anchors[endpoints[0]][weight] += 1
        elif len(endpoints) == 2:
            u, v = endpoints
            adjacency[u][v] += 1
            adjacency[v][u] += 1
            edge_colors[u][weight] += 1
            edge_colors[v][weight] += 1
        else:
            raise ValueError("projected check is not an edge or grounding")

    size = len(high)
    colors = [
        (cols[j].bit_count(),
         tuple(sorted(anchors[i].items())),
         tuple(sorted(edge_colors[i].items())))
        for i, j in enumerate(high)]
    walks = [[[] for _ in range(size)] for _ in range(size)]
    power = [[int(i == j) for j in range(size)] for i in range(size)]
    for _ in range(8):
        following = [[0]*size for _ in range(size)]
        for u in range(size):
            target = following[u]
            for v, multiplicity in adjacency[u].items():
                source = power[v]
                for w, value in enumerate(source):
                    target[w] += multiplicity*value
        power = following
        for u in range(size):
            for v in range(size):
                walks[u][v].append(power[u][v])

    joint_walks = tuple(sorted(
        (colors[u], tuple(walks[u][u]),
         tuple(sorted(
             (colors[v], tuple(walks[v][v]), tuple(walks[u][v]))
             for v in range(size))))
        for u in range(size)))
    intrinsic = (
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        intersections, joint_walks)
    return sha256(repr(intrinsic).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q8_portfolio(G):
    """Immutable exhaustive portfolio; memoization does not alter outputs."""
    from itertools import combinations

    if G.q != 8:
        raise ValueError("new shape requires Dic2 of order eight")
    q, width = G.q, 7*G.q
    cap = min(9, W_MAX)
    e = G.e
    dummy = [[[e]]]
    mask = (1 << width)-1

    def fingerprint(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def component_ok(grid, wanted_rank):
        # Obtain BOTH constituent actions from the verified helper.
        left, _, _ = lifted_matrices(G, grid, dummy)
        _, right, _ = lifted_matrices(G, dummy, grid)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        return (
            gf2_rank(left) == wanted_rank
            and gf2_rank(right) == wanted_rank
            and _q8_clear_three(left, width)
            and _q8_clear_three(right, width))

    regular, grounded = [], []
    seen_fillings = set()

    def retain(grid, card, bank, wanted_rank):
        filling = fingerprint(grid)
        if filling in seen_fillings:
            return
        seen_fillings.add(filling)
        counts = [[len(cell) for cell in row] for row in grid]
        if len(grid) != 2 or any(len(row) != 7 for row in grid):
            raise ValueError("constituent card has the wrong shape")
        if max(map(sum, counts)) > 7:
            return
        if any(sum(counts[i][j] for i in range(2)) > 2
               for j in range(7)):
            return
        if not component_ok(grid, wanted_rank):
            return
        bank.append((card, freeze(grid)))

    # Seven distinct cross-matchings: a regular incidence component.
    for labels in combinations(range(q), 7):
        grid = [
            [[e] for _ in labels],
            [[G.elements[t]] for t in labels],
        ]
        retain(grid, ("seven-matchings", labels), regular, 2*q-1)

    # Two internal cycle columns per layer and three cross-matchings.
    # Exhaustive deck descriptions are retained until FULL-code screening.
    directions = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if _dic_order(G, g) == 4
    })
    step_pairs = list(combinations(directions, 2))
    for top in step_pairs:
        for bottom in step_pairs:
            for labels in combinations(range(q), 3):
                grid = [
                    [[e, G.elements[t]] for t in top]
                    + [[e] for _ in labels] + [[] for _ in bottom],
                    [[] for _ in top]
                    + [[G.elements[t]] for t in labels]
                    + [[e, G.elements[t]] for t in bottom],
                ]
                retain(grid, ("internal-cycles", top, bottom, labels),
                       regular, 2*q-1)

    # One grounding singleton, not two. Six cross-matchings give row
    # totals seven and six, column totals two except the singleton.
    # The verified screen checks triangles and grounding interactions.
    for labels in combinations(range(q), 6):
        grid = [
            [[e] for _ in labels] + [[e]],
            [[G.elements[t]] for t in labels] + [[]],
        ]
        retain(grid, ("one-grounding", labels), grounded, 2*q)

    if not regular or not grounded:
        raise ValueError("regular or grounded constituent bank exhausted")

    regular.sort(key=lambda item: item[0])
    grounded.sort(key=lambda item: item[0])
    graph_keys = set()
    entries = []

    def consider(a, b):
        A, B = a[1], b[1]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            return
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 424 or max(row.bit_count() for row in hx+hz) > cap:
            return
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            return

        # This includes mixed-sector words; constituent girth alone
        # would not establish the complete-code floor.
        if not (_q8_clear_three(hx, n)
                and _q8_clear_three(hz, n)):
            return
        signature = _q8_graph_key(hx, hz, n)
        if signature in graph_keys:
            return
        if not css_ok(hx, hz):
            raise ValueError("verified Q8 catalogue failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")
        graph_keys.add(signature)
        entries.append((A, B, k, (a[0], b[0]), signature))

    # Higher-rank-surplus regular pairs first. Grounded pairs supply
    # structurally different fallbacks, still within the requested cell.
    for i, a in enumerate(regular):
        for b in regular[i:]:
            consider(a, b)
            if len(entries) >= N_SLOTS:
                return tuple(sorted(entries, key=lambda item: (-item[2], item[4])))

    for a in regular:
        for b in grounded:
            consider(a, b)
            if len(entries) >= N_SLOTS:
                return tuple(sorted(entries, key=lambda item: (-item[2], item[4])))

    # Never pad the portfolio with isomorphic copies.
    return tuple(sorted(entries, key=lambda item: (-item[2], item[4])))


def NEW_CTX(rng, G, idx):
    """Pure indexed selection of an immutable, full-matrix-screened design."""
    entries = _q8_portfolio(G)
    if not (0 <= idx < len(entries)):
        raise ValueError("distinct target-rate Q8 catalogue exhausted")
    A, B, k, card, signature = entries[idx]
    return {
        "A": A, "B": B, "k": k, "card": card,
        "verified_kernel_floor": 4,
        "intrinsic_signature": signature.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct Dic7 relative-order cards ensure distinct Tanner graphs: the parent produced only thirteen distinct codes among sixty slots, with numerous cross-card isomorphic duplicates.
# REFUTED: The Dic7 archive-cycle catalogue supplies one evaluator-distinct code per slot: the current sixty-slot evaluation retained sixteen classes, and all five measured distances remained three.
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
