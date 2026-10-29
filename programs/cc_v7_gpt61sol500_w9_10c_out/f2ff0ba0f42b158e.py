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
CONCEPT = ("Dic6, order 24, A=(1,3), B=(2,7): replacing regular B incidence constituents by a singleton-grounded boundary should retain five novel k=250 codes with distance readings >=4, restoring a full five-code score above 5.9824 in cell 0.450.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This bed measured distance-four codes, but the current sixty-slot program retained only four distinct codes and scored 5.9824. Retain its verified order-24 A rank-deficit mechanism while changing B's intrinsic boundary structure, rather than pursuing further copies of its k=258 regular graphs.")
SHAPES = ("Keep A=(1,3), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.43478. The size window permits q<=26; the frozen lift uses q=24. Unequal constituents have 72 variables/24 checks and 168 variables/48 checks. Parent minima occupied sector-1 Acol0..2/Bcol0..6 broadly. Rank-22 A and split full-row-rank B predict k=250; complete verified ranks enforce that value, giving rate 0.45290 inside cell 0.450.")
FILLING = ("Twenty cards combine two verified rank-22 binomial A models, two grounded B supports and five relative-voltage policies. B has one singleton plus six matchings, or one singleton plus five matchings and an opposite-row binomial. B row totals are (7,6) or (6,7), and its columns total one or two: HX weights are seven/eight and HZ weights eight/nine. Two monomial pivots make B full-row-rank. Deterministic finite pools minimize constituent four-collision diagnostics; complete verified kernels exclude weights 1..3. Rooted intrinsic high-qubit projections retain low-qubit corners and conservatively reject repeated Tanner invariants. Five novel distance-four measurements would score 7.24638; neither archive novelty nor distance above four is assumed.")
FAMILY_NAME = "r045_dic6_13x27_grounded_splitB_kernel3_portfolio_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if q != 24:
        raise ValueError("Dic6 has order 24")
    # x^12=1, y^2=x^6, y*x*y^-1=x^-1; canonical element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(12)]

    def red(g):
        a, b = g
        return ((a + 6 * (b // 2)) % 12, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 6 * b * d) % 12,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 12, 0) if b == 0 else ((a + 6) % 12, 1)

    def idx(g):
        a, b = red(g)
        return 12 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
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


from functools import lru_cache


@lru_cache(maxsize=1)
def _d6_nonbipartite_portfolio(G, count):
    """Immutable deterministic portfolio; caching changes only build cost.

    Group algebra constructs voltage prescriptions. All accepted ranks,
    component screens and complete quantum screens use verified lifts.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    if q != 24 or cap < 9:
        raise ValueError("Dic6 incidence portfolio requires q=24 and cap nine")

    def closure(generators):
        reached, todo = {e}, [e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        for g in todo:
            for h in steps:
                product = G.mul(g, h)
                if product not in reached:
                    reached.add(product)
                    todo.append(product)
        return tuple(sorted(reached))

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    triples = (
        ((1, 0), (3, 0), (5, 0)),
        ((0, 1), (2, 1), (4, 1)),
        ((1, 1), (3, 1), (5, 1)),
    )
    acells = tuple([[[e, g] for g in triple]] for triple in triples)
    asubgroups = tuple(closure(triple) for triple in triples)
    subgroups = list(asubgroups)
    subgroups.extend(closure(((3, 0), (a, 1))) for a in range(3))
    assert all(len(h) == 12 for h in asubgroups)
    assert all(len(h) == 8 for h in subgroups[3:])

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

    def internal_description(subgroup, gaps):
        """Algebraic construction gate and odd-cycle preference.

        This describes a Cayley graph, not a substitute LP matrix lift.
        Every published constituent is screened again on verified rows.
        """
        directions = set(gaps) | {G.inv(g) for g in gaps}
        if len(directions) != 2 * len(gaps):
            return None
        if any(G.mul(g, h) in directions
               for g in directions for h in directions):
            return None
        colors, nonbipartite = {}, False
        for root in subgroup:
            if root in colors:
                continue
            colors[root] = 0
            todo = [root]
            for g in todo:
                for h in directions:
                    neighbor = G.mul(g, h)
                    if neighbor not in colors:
                        colors[neighbor] = 1 - colors[g]
                        todo.append(neighbor)
                    elif colors[neighbor] == colors[g]:
                        nonbipartite = True
        return directions, int(nonbipartite)

    def cross_cliques(subgroup, top, bottom, size):
        """All compatible voltage sets containing the identity.

        Row gauges can normalize one cross voltage to e; conjugated
        internal directions remain covered by the enumerated subgroup.
        Compatibility excludes algebraic cross/internal triangles.
        """
        td = set(top) | {G.inv(g) for g in top}
        bd = set(bottom) | {G.inv(g) for g in bottom}

        def compatible(v, w):
            return (G.mul(G.inv(v), w) not in td
                    and G.mul(v, G.inv(w)) not in bd)

        options = [g for g in subgroup if g != e and compatible(e, g)]

        def extend(chosen, available):
            need = size - len(chosen)
            if need == 0:
                yield tuple(chosen)
                return
            if len(available) < need:
                return
            for i, g in enumerate(available):
                tail = [h for h in available[i + 1:]
                        if compatible(g, h)]
                yield from extend(chosen + [g], tail)

        yield from extend([e], options)

    def fingerprint(hx, hz, n):
        """Intrinsic rooted projection plus full check intersections.

        Degree-13 qubits intrinsically identify sector 2 for these
        regular profiles. Check weights label its projected edges.
        All roots are retained; no block transitivity is assumed.
        Equality is inconclusive and is rejected conservatively.
        """
        rows = list(hx) + list(hz)
        columns = _column_syndromes(rows, n)
        high = [j for j, col in enumerate(columns)
                if col.bit_count() == 13]
        positions = {j: i for i, j in enumerate(high)}
        adjacency = [Counter() for _ in high]
        for row in rows:
            ends = []
            bits = row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in positions:
                    ends.append(positions[j])
                bits ^= bit
            if len(ends) != 2:
                raise ValueError("regular projection has a non-edge check")
            a, b = ends
            tag = row.bit_count()
            adjacency[a][(b, tag)] += 1
            adjacency[b][(a, tag)] += 1

        def compress(features):
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            return [palette[f] for f in features]

        rooted = []
        for root in range(len(high)):
            colors = [int(i == root) for i in range(len(high))]
            history = []
            for _ in range(6):
                features = [
                    (colors[i], tuple(sorted(
                        (tag, multiplicity, colors[j])
                        for (j, tag), multiplicity in neighbors.items())))
                    for i, neighbors in enumerate(adjacency)
                ]
                history.append(tuple(sorted(Counter(features).items())))
                colors = compress(features)
            rooted.append(sha256(repr(history).encode("ascii")).digest())

        return (
            n,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in columns).items())),
            tuple(sorted(rooted)),
            _tanner_signature(hx, hz),
        )

    # Exclusion references only. These are frozen-Dic6 fillings shown
    # in the current table or the supplied outside-H12 table.
    references = (
        (0, ((1,0),), ((5,0),),
         ((6,0),(0,0),(2,0),(4,0),(10,0))),
        (0, ((5,0),(3,0)), ((5,0),(3,0)),
         ((2,0),(0,0),(4,0))),
        (0, ((3,0),(1,0),(5,0)), ((5,0),(1,0),(3,0)), ((2,0),)),
        (0, (), (), ((4,0),(9,0),(3,0),(8,0),(5,0),(0,0),(6,0))),
        (0, (), (), ((0,0),(4,1),(10,1),(0,1),(6,0),(2,0),(10,0))),
        (0, (), (), ((0,0),(1,0),(2,0),(3,0),(6,0),(7,0),(9,0))),
        (0, ((1,0),), ((5,0),),
         ((0,0),(2,0),(4,0),(6,0),(8,0))),
        (0, ((1,0),(3,0)), ((3,0),(5,0)), ((0,0),(2,0),(6,0))),
        (0, ((1,0),(3,0),(5,0)), ((1,0),(3,0),(5,0)), ((0,0),)),
        (0, ((1,0),(3,0)), ((2,0),(3,0)), ((0,0),(4,0),(8,0))),
        (1, (), (), ((10,1),(8,0),(6,1),(10,0),(2,0),(8,1),(4,0))),
        (1, ((0,1),(4,1)), ((2,1),(0,1)), ((4,0),(6,0),(2,0))),
        (1, ((2,1),), ((2,1),),
         ((6,0),(0,0),(10,0),(2,0),(8,0))),
        (1, ((4,1),(0,1),(2,1)), ((4,1),(2,1),(0,1)), ((4,0),)),
        (1, (), (), ((1,0),(4,0),(5,0),(11,0),(2,0),(3,0),(10,0))),
        (2, ((3,1),(1,1)), ((9,1),(11,1)), ((6,0),(2,0),(10,0))),
    )
    graph_keys = set()
    for ai, top, bottom, voltages in references:
        hx, hz, n = lifted_matrices(
            G, acells[ai], incidence(top, bottom, voltages))
        graph_keys.add(fingerprint(hx, hz, n))

    # Enumerate genuine structural prescriptions. In particular, H12
    # rows may mix x^2 with an outside direction, creating an odd cycle
    # without an internal triangle. The parent's policies cannot do this.
    cards = []
    for si, subgroup in enumerate(subgroups):
        directions = sorted({
            min(g, G.inv(g)) for g in subgroup
            if orders[g] >= 4
        })
        for internal in (2, 1, 0, 3):
            descriptions = []
            for gaps in combinations(directions, internal):
                description = internal_description(subgroup, gaps)
                if description is not None:
                    descriptions.append((gaps, description[1]))
            for top, odd_top in descriptions:
                for bottom, odd_bottom in descriptions:
                    cross_count = 7 - 2 * internal
                    for voltages in cross_cliques(
                            subgroup, top, bottom, cross_count):
                        sixes = sum(orders[g] == 6 for g in top + bottom)
                        priority = (
                            -(odd_top + odd_bottom), -sixes,
                            int(internal != 2), internal, si,
                            top, bottom, voltages)
                        cards.append((priority, si, top, bottom, voltages))
    cards.sort(key=lambda record: record[0])

    def orbit_key(ai, top, bottom, voltages):
        """Remove common Dic6 automorphisms, inversions and B-row exchange."""
        keys = []
        for u in (1, 5, 7, 11):
            for v in range(12):
                def image(g):
                    return ((u * g[0] + v * g[1]) % 12, g[1])

                def undirected(g):
                    h = image(g)
                    return min(G.idx(h), G.idx(G.inv(h)))

                aa = tuple(sorted(undirected(g) for g in triples[ai]))
                tt = tuple(sorted(undirected(g) for g in top))
                bb = tuple(sorted(undirected(g) for g in bottom))
                vv = tuple(sorted(G.idx(image(g)) for g in voltages))
                iv = tuple(sorted(G.idx(G.inv(image(g))) for g in voltages))
                keys.append((aa, tt, bb, vv))
                keys.append((aa, bb, tt, iv))
        return min(keys)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    orbit_keys, results = set(), []
    # High-rank candidates are exhausted before lower-rank fallbacks.
    # A matched normal subgroup is a rank hypothesis, not an acceptance
    # shortcut. Quaternion B proposals test a different component size.
    for tier in range(2):
        for _, si, top, bottom, voltages in cards:
            subgroup = subgroups[si]
            for ai, A in enumerate(acells):
                matched = subgroup == asubgroups[ai] or len(subgroup) == 8
                if matched != (tier == 0):
                    continue
                orbit = orbit_key(ai, top, bottom, voltages)
                if orbit in orbit_keys:
                    continue
                orbit_keys.add(orbit)
                B = incidence(top, bottom, voltages)
                sa = [[2, 2, 2]]
                sb = [[len(cell) for cell in row] for row in B]
                if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                    raise ValueError("enumerated incidence card is illegal")
                assert all(len(set(cell)) == len(cell)
                           for row in B for cell in row)

                hx, hz, n = lifted_matrices(G, A, B)
                rx, rz = gf2_rank(hx), gf2_rank(hz)
                k = n - rx - rz
                minimum_k = 258 if tier == 0 else 250
                if not (n == 552 and k >= minimum_k
                        and 425*n <= 1000*k < 475*n):
                    continue
                ar, br = _verified_components(hx, hz, q)
                qa = _component_collisions(ar, 3*q)
                qb = _component_collisions(br, 7*q)
                if qa is None or qb is None:
                    continue
                if not (_no_short_kernel(hx, n)
                        and _no_short_kernel(hz, n)):
                    continue

                signature = fingerprint(hx, hz, n)
                if signature in graph_keys:
                    continue
                graph_keys.add(signature)
                if max(row.bit_count() for row in hx + hz) > cap:
                    raise ValueError("verified rows exceed support budget")

                results.append((
                    freeze(A), freeze(B), k, (ai, si, len(top), tier),
                    (qa, qb), (len(hx)-rx, len(hz)-rz)))
                if len(results) >= count:
                    return tuple(sorted(results, key=lambda r: -r[2]))

    return tuple(sorted(results, key=lambda r: -r[2]))


@lru_cache(maxsize=1)
def _d6_grounded_portfolio(G, count):
    """Deterministic immutable boundary portfolio.

    The former regular-incidence portfolio is not called. Each position
    optimizes a construction card against all previously accepted graph
    invariants. Memoization affects cost, not the returned fillings.
    """
    from collections import Counter
    from hashlib import sha256

    q, e, cap = G.q, G.e, min(9, W_MAX)
    if q != 24 or cap < 9:
        raise ValueError("grounded Dic6 portfolio requires q=24 and cap nine")

    def closure(generators):
        reached, todo = {e}, [e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        for g in todo:
            for h in steps:
                product = G.mul(g, h)
                if product not in reached:
                    reached.add(product)
                    todo.append(product)
        return tuple(sorted(reached))

    triples = (
        ((1, 0), (3, 0), (5, 0)),
        ((0, 1), (2, 1), (4, 1)),
    )
    subgroups = tuple(closure(gs) for gs in triples)
    third_subgroup = closure(((2, 0), (1, 1)))
    quaternion = tuple(closure(((3, 0), (a, 1))) for a in range(3))
    assert all(len(h) == 12 for h in subgroups)
    assert len(third_subgroup) == 12
    assert all(len(h) == 8 for h in quaternion)

    abank = []
    for generators in triples:
        A = [[[e, g] for g in generators]]
        # Zero auxiliary side exposes precisely the verified A action.
        ar, _, _ = lifted_matrices(G, A, [[[]]])
        qa = _component_collisions(ar, 3 * q)
        if qa is None or gf2_rank(ar) != q - 2:
            raise ValueError("grounded card's A model failed verified screening")
        abank.append((A, ar, qa))

    all_elements = tuple(sorted(G.elements))
    # Reject identity, involutions and order-three internal directions.
    gaps = tuple(sorted({
        min(g, G.inv(g)) for g in all_elements
        if g != e
        and G.mul(g, g) != e
        and G.mul(G.mul(g, g), g) != e
    }))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def graph_key(hx, hz, n):
        """Intrinsic rooted projection retaining low-qubit corners.

        Qubit degrees >=10 identify the high-degree vertices without
        protograph labels. A check projects to an edge or grounding mark,
        coloured by its actual weight. Other qubits retain the multiset
        of endpoints of all their incident checks, including repeated
        endpoints. All roots and all multiplicities are retained.

        Different keys certify non-isomorphic Tanner graphs. Equality
        is inconclusive and is conservatively rejected.
        """
        rows = list(hx) + list(hz)
        cols = _column_syndromes(rows, n)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() >= 10]
        if len(high) != 2 * q:
            raise ValueError("unexpected grounded high-degree projection")
        position = {j: v for v, j in enumerate(high)}
        adjacency = [Counter() for _ in high]
        anchors = [Counter() for _ in high]
        check_ends = []

        for row in rows:
            ends = []
            bits = row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in position:
                    ends.append(position[j])
                bits ^= bit
            ends = tuple(sorted(ends))
            check_ends.append(ends)
            weight = row.bit_count()
            if len(ends) == 1:
                anchors[ends[0]][weight] += 1
            elif len(ends) == 2:
                a, b = ends
                adjacency[a][(b, weight)] += 1
                adjacency[b][(a, weight)] += 1
            else:
                raise ValueError("grounded check projection is not an edge/mark")

        labels = [
            (cols[j].bit_count(), tuple(sorted(anchors[v].items())))
            for v, j in enumerate(high)
        ]
        corners = []
        for j, syndrome in enumerate(cols):
            if j in position:
                continue
            ends, weights = [], []
            bits = syndrome
            while bits:
                bit = bits & -bits
                r = bit.bit_length() - 1
                ends.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                bits ^= bit
            corners.append((
                syndrome.bit_count(), tuple(sorted(weights)),
                tuple(sorted(ends))))

        def compress(features):
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            return [palette[f] for f in features]

        rooted = []
        for root in range(len(high)):
            colors = compress([
                (labels[v], int(v == root)) for v in range(len(high))
            ])
            history = []
            for iteration in range(6):
                features = [
                    (labels[v], colors[v], tuple(sorted(
                        (weight, multiplicity, colors[u])
                        for (u, weight), multiplicity
                        in adjacency[v].items())))
                    for v in range(len(high))
                ]
                history.append(tuple(sorted(Counter(features).items())))
                colors = compress(features)
                if iteration in (2, 5):
                    cf = [
                        (degree, weights,
                         tuple(sorted(colors[v] for v in ends)))
                        for degree, weights, ends in corners
                    ]
                    history.append(tuple(sorted(Counter(cf).items())))
            rooted.append(sha256(repr(tuple(history)).encode("ascii")).digest())

        signature = (
            n,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(rooted)),
        )
        return sha256(repr(signature).encode("ascii")).hexdigest()

    graph_keys, filling_keys, entries = set(), set(), []
    sa = [[2, 2, 2]]

    for slot in range(count):
        card = slot % 20
        amodel = card // 10
        boundary = (card % 10) // 5
        policy = card % 5
        epoch = slot // 20
        A, expected_ar, qa = abank[amodel]
        common = subgroups[amodel]
        complement = tuple(g for g in all_elements if g not in common)
        local = random.Random(
            BASE_SEED * 10007 + 0xD6B0250 + 104729 * slot)

        prescribed = (
            common,
            subgroups[1 - amodel],
            quaternion[epoch % 3],
            third_subgroup,
            all_elements,
        )[policy]
        selected = None

        # First optimize the named subgroup prescription. If it has no
        # new invariant, broaden relative coupling, not A/B gauges.
        for stage, trials in enumerate((48, 64, 96)):
            candidates = []
            cross_count = 6 - boundary
            edge_pool = tuple(g for g in gaps if g in prescribed)
            if stage:
                edge_pool = gaps

            for trial in range(trials):
                if boundary and not edge_pool:
                    continue
                if stage == 0:
                    if len(prescribed) < cross_count:
                        continue
                    voltages = local.sample(prescribed, cross_count)
                elif stage == 1:
                    outside = 1 + (slot + trial) % (cross_count - 1)
                    voltages = (
                        local.sample(common, cross_count - outside)
                        + local.sample(complement, outside))
                else:
                    voltages = local.sample(all_elements, cross_count)

                # The singleton grounds only the top check row. Any
                # internal binomial lies exclusively in the other row.
                B = [[[e]], [[]]]
                gap = None
                if boundary:
                    gap = local.choice(edge_pool)
                    B[0].append([])
                    B[1].append([e, gap])
                for voltage in voltages:
                    B[0].append([e])
                    B[1].append([voltage])

                sb = [[len(cell) for cell in row] for row in B]
                if not (_no_dead_lines(sa, sb)
                        and _grid_weights(sa, sb) <= cap):
                    raise ValueError("grounded boundary card is illegal")
                assert sorted(map(sum, sb)) == [6, 7]
                assert sorted(sb[0][j] + sb[1][j]
                              for j in range(7)) == [1] + [2] * 6
                assert all(len(set(cell)) == len(cell)
                           for grid in (A, B) for row in grid for cell in row)

                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for grid in (A, B) for row in grid for cell in row)
                if filling in filling_keys:
                    continue
                filling_keys.add(filling)

                hx, hz, n = lifted_matrices(G, A, B)
                ar, br = _verified_components(hx, hz, q)
                if ar != expected_ar:
                    raise ValueError("verified A extraction disagrees")
                qb = _component_collisions(br, 7 * q)
                if qb is None:
                    continue

                rx, rz = gf2_rank(hx), gf2_rank(hz)
                k = n - rx - rz
                # Grounding and a cross matching give two monomial
                # pivots. Verify the predicted free B kernel and k.
                if gf2_rank(br) != 2 * q:
                    continue
                if not (n == 552 and k == 250
                        and 425 * n <= 1000 * k < 475 * n):
                    continue
                if max(row.bit_count() for row in hx + hz) > cap:
                    continue

                priority = (qb, trial, filling)
                candidates.append((
                    priority, B, hx, hz, n, k, rx, rz, qb, gap))

            candidates.sort(key=lambda record: record[0])
            # Bounded invariant refinement avoids repeatedly spending
            # the whole budget on an exhausted subgroup orbit.
            for _, B, hx, hz, n, k, rx, rz, qb, gap in candidates[:12]:
                key = graph_key(hx, hz, n)
                if key in graph_keys:
                    continue
                if not (_no_short_kernel(hx, n)
                        and _no_short_kernel(hz, n)):
                    continue
                if not css_ok(hx, hz):
                    raise ValueError("verified grounded lift failed CSS")
                graph_keys.add(key)
                selected = (
                    freeze(A), freeze(B), k,
                    (card, stage, amodel, boundary, policy),
                    (qa, qb), (len(hx) - rx, len(hz) - rz), key)
                break
            if selected is not None:
                break

        # Preserve positions: a failed card cannot recycle another
        # slot's accepted filling or silently return an orbit copy.
        entries.append(selected)

    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a boundary card; no regular-incidence fallback."""
    catalogue = _d6_grounded_portfolio(G, N_SLOTS)
    entry = catalogue[idx]
    if entry is None:
        raise ValueError("distinct grounded boundary card exhausted")
    A, B, k, card, quality, deficits, signature = entry
    hx, hz, n = lifted_matrices(G, A, B)
    if not css_ok(hx, hz):
        raise ValueError("verified grounded lift failed CSS")
    if lifted_k(G, A, B) != k:
        raise ValueError("verified rank helpers disagree")
    assert k == 250 and n == 552
    assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
    return {
        "A": A, "B": B, "k": k, "card": card,
        "component_quality": quality,
        "LP_rank_deficits": deficits,
        "tanner_signature": signature,
        "grounded_unequal_B_rows": True,
        "complete_kernel_weights_1_to_3_excluded": True,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish the complete boundary cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on the spanning-A Dic6 catalogue escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
# REFUTED: The rotation-A coset-cycle prefix supplies twenty inequivalent initial high-rank designs: measured slots 0..19 retained seven classes, and all five measured leaders still had d30s=4 with abundant sector-1 witnesses.
# REFUTED: Exhaustively prescribed non-bipartite clique cards restore five novel high-rank finalists: the measured sixty-slot program retained only four distinct codes, all k=258,d30s=4, and scored 5.9824.
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
