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
CONCEPT = ("Frozen C3 semidirect_2 C4, order 12, A=(2,6), B=(2,7): triangle-free order-six and mixed-parity incidence pairings should supply five fresh distance-four codes with mean k>256.2, improving the measured parent fitness 7.4258 in cell 0.450.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This bed measured k=258,d30s=4, but sixty slots retained only ten classes. Its normal C6 contains an order-six inverse class omitted from the parent's internal-edge policy. Testing that class and mixed-parity cross voltages changes relative incidence structures without assuming an improvement from element orders.")
SHAPES = ("Keep A=(2,6), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.43478. The window permits q<=13 and the frozen group fixes q=12. Components have 72 variables/24 checks and 84 variables/24 checks. Parent minima broadly used sector-1 Acol0..5/Bcol0..6, particularly Acol1 and Acol4; no measured minimum required sector 2. Verified ranks retain [0.425,0.475).")
FILLING = ("Enumerate finite inverse-normalized B incidence cards with balanced internal allocations, admitting order-six gaps and mixed-parity matchings. Pair them with separated outside-binomial A and normal-C6 matching A. Both A rows total six and B rows seven; every column totals two, so HX=8 and HZ=9. Globally prioritize verified k, then fewer constituent four-collisions. Exact complete-kernel exclusion through three and an explicit non-stabilizer four-witness certify distance four for every published entry. Rooted high-qubit hyperedges PLUS low-qubit corners conservatively reject repeats and shown archive anchors. Five fresh k=258 entries would score 7.47826; unseen-archive novelty and catalogue yield remain hypotheses.")
FAMILY_NAME = "r045_m3r4s2_26x27_rooted_incidence_rank_v3_sixcycle_corners"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 46 requires q<=13; 2^4=1 modulo 3.
    m, r, s = 3, 4, 2
    if not valid_q(q):
        raise ValueError("this semidirect product has order 12")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + powers[g[1] % r] * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-powers[bi] * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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
    """Three order-four edge classes and the normal order-six subgroup."""
    steps = sorted({min(g, G.inv(g))
                    for g in G.elements if g[1] % 2})
    subgroup = [g for g in G.elements if g[1] % 2 == 0]
    assert len(steps) == 3 and len(subgroup) == 6
    assert all(G.mul(g, g) != G.e for g in steps)
    return steps, subgroup


def _draw_incidence(rng, G, attempt, pools):
    """Four A prescriptions times five B support prescriptions."""
    steps, subgroup = pools
    card, epoch = attempt % 20, attempt // 20
    amode, bmode = card % 4, card // 4

    def regular(width, internal, labels=None):
        C = [[], []]
        for i in range(2):
            for gap in rng.sample(steps, internal):
                C[i].append([G.e, gap])
                C[1 - i].append([])
        cross = width - 2 * internal
        if labels is None:
            # Odd internal gaps reverse b-parity. Cross differences
            # preserve it, giving a sufficient triangle-free grading.
            pool = subgroup if internal else G.elements
            labels = rng.sample(pool, cross)
        assert len(labels) == cross
        for voltage in labels:
            C[0].append([G.e])
            C[1].append([voltage])
        assert all(len(row) == width for row in C)
        assert all(sum(map(len, row)) == width for row in C)
        return C

    if amode == 0:
        # Six distinct subgroup matchings give two K6,6 components.
        A = regular(6, 0, list(subgroup))
    elif amode == 1:
        outside = [g for g in G.elements if g[1] % 2]
        count = 1 + epoch % 5
        labels = (rng.sample(subgroup, 6 - count)
                  + rng.sample(outside, count))
        A = regular(6, 0, labels)
    elif amode == 2:
        A = regular(6, 1)
    else:
        A = regular(6, 2 + epoch % 2)

    if bmode < 4:
        B = regular(7, bmode)
    else:
        # Row sums seven, column degrees 1,3,2,2,2,2,2.
        counts = ((1, 2, 1, 1, 1, 1, 0),
                  (0, 1, 1, 1, 1, 1, 2))
        B = [[[] for _ in range(7)] for _ in range(2)]
        for i in range(2):
            for j, count in enumerate(counts[i]):
                if count == 1:
                    B[i][j] = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    gap = rng.choice(steps)
                    B[i][j] = [first, G.mul(first, gap)]

    assert all(sum(map(len, row)) == 6 for row in A)
    assert all(sum(map(len, row)) == 7 for row in B)
    assert all(len(A[0][j]) + len(A[1][j]) == 2
               for j in range(6))
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


def _catalogue_invariant(hx, hz, n):
    """Intrinsic rooted invariant, including nonregular hyperedges.

    Degree-13 qubits identify sector 2 without using block labels.
    Check weights colour its projected hyperedges. Root every such
    qubit, rather than assuming a transitive full-code group action.
    Equality is inconclusive and is conservatively rejected.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    high = [j for j, col in enumerate(columns)
            if col.bit_count() == 13]
    positions = {j: t for t, j in enumerate(high)}
    edges, weights = [], []
    incident = [[] for _ in high]
    for row in rows:
        endpoints = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                endpoints.append(positions[j])
            bits ^= bit
        edge = len(edges)
        edges.append(tuple(endpoints))
        weights.append(row.bit_count())
        for v in endpoints:
            incident[v].append(edge)

    def compress(signatures):
        table = {s: i for i, s in enumerate(sorted(set(signatures)))}
        return [table[s] for s in signatures]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(5):
            es = [(w, tuple(sorted(colours[v] for v in endpoints)))
                  for w, endpoints in zip(weights, edges)]
            ec = compress(es)
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            colours = compress(vs)
        rooted.append(sha256(repr(tuple(history)).encode()).digest())
    return (n, tuple(sorted(Counter(weights).items())),
            tuple(sorted(Counter(c.bit_count()
                                 for c in columns).items())),
            tuple(sorted(rooted)))


def _shown_archive_invariants():
    """Only exclusion references; never submit these archived fillings."""
    els = [(a, b) for b in range(2) for a in range(12)]

    def mul(g, h):
        return ((g[0] + pow(5, g[1], 12) * h[0]) % 12,
                (g[1] + h[1]) % 2)

    def inv(g):
        b = (-g[1]) % 2
        return ((-pow(5, b, 12) * g[0]) % 12, b)

    old = _G(dict(elements=els, ident=(0, 0), mul=mul, inv=inv), 24)
    e = old.e

    def incidence(top, bottom, ratios):
        B = [[], []]
        for gap in top:
            B[0].append([e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([e, gap])
        for voltage in ratios:
            B[0].append([e])
            B[1].append([voltage])
        return B

    references = [
        (((3,1),(5,1),(1,1)), (), (),
         ((7,1),(9,1),(10,1),(5,1),(0,0),(3,1),(5,0))),
        (((3,0),(1,0),(5,0)), ((5,1),), ((1,0),),
         ((0,0),(8,1),(6,0),(10,1),(2,0))),
        (((5,1),(1,1),(3,1)), ((1,1),(1,0)), ((3,0),(5,1)),
         ((6,1),(8,1),(10,1))),
        (((3,0),(5,0),(1,0)), ((1,1),(1,0)), ((3,0),(5,0)),
         ((6,1),(10,1),(4,0))),
        (((5,1),(1,1),(3,1)), ((3,1),(1,1),(3,0)),
         ((1,0),(5,1),(3,1)), ((2,1),)),
    ]
    common = (
        ((), (), (0,1,2,3,6,7,9)),
        ((1,), (5,), (0,2,4,6,8)),
        ((1,3), (3,5), (0,2,6)),
        ((1,3,5), (1,3,5), (0,)),
        ((5,1), (5,3), (0,4,8)),
    )
    for top, bottom, ratios in common:
        references.append(
            (((1,0),(3,0),(5,0)),
             tuple((d,0) for d in top),
             tuple((d,0) for d in bottom),
             tuple((d,0) for d in ratios)))

    result = set()
    for generators, top, bottom, ratios in references:
        A = [[[e, g] for g in generators]]
        B = incidence(top, bottom, ratios)
        hx, hz, n = lifted_matrices(old, A, B)
        result.add(_catalogue_invariant(hx, hz, n))
    return result


from functools import lru_cache


@lru_cache(maxsize=1)
def _six_cycle_portfolio(G):
    """Immutable deterministic catalogue; memoization changes cost only.

    No slot mutates this result. All component actions, ranks and
    logical certificates are obtained from the verified construction.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    assert G.q == 12
    cap = min(9, W_MAX)
    if cap < 9:
        raise ValueError("six-cycle incidence requires row cap nine")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def incidence(group, width, top, bottom, voltages):
        C = [[], []]
        for gap in top:
            C[0].append([group.e, gap])
            C[1].append([])
        for gap in bottom:
            C[0].append([])
            C[1].append([group.e, gap])
        for voltage in voltages:
            C[0].append([group.e])
            C[1].append([voltage])
        assert all(len(row) == width for row in C)
        return C

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def component_four(rows, width):
        """Exclude weights <=3; return a verified four-word and burden."""
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return None
        pairs, witness, burden = {}, 0, 0
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles:
                    return None
                word = (1 << i) | (1 << j)
                old = pairs.get(syndrome)
                if old is None:
                    pairs[syndrome] = word
                else:
                    difference = word ^ old
                    assert difference.bit_count() == 4
                    burden += 1
                    if not witness:
                        witness = difference
        return witness, burden

    def graph_key(hx, hz, n):
        """Intrinsic uncoloured invariant including low-qubit corners.

        High qubits are selected by degree, not LP sector labels.
        Check hyperedges and low-qubit incident-check corners retain
        endpoint multiplicity. Different keys imply non-isomorphism;
        equality is inconclusive and is conservatively rejected.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, n)
        high = [j for j, c in enumerate(cols) if c.bit_count() == 13]
        position = {j: v for v, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        ends, tags, check_ends = [], [], []

        for row in rows:
            bits, vertices = row & high_mask, []
            while bits:
                bit = bits & -bits
                vertices.append(position[bit.bit_length() - 1])
                bits ^= bit
            vertices = tuple(sorted(vertices))
            check_ends.append(vertices)
            ends.append(vertices)
            tags.append((0, row.bit_count()))

        for j, syndrome in enumerate(cols):
            if j in position:
                continue
            vertices, weights = [], []
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                vertices.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                syndrome ^= bit
            ends.append(tuple(sorted(vertices)))
            tags.append((1, tuple(sorted(weights))))

        incident = [[] for _ in high]
        for edge, vertices in enumerate(ends):
            for v in vertices:
                incident[v].append(edge)

        def compress(signatures):
            mapping = {s: i for i, s in enumerate(sorted(set(signatures)))}
            return [mapping[s] for s in signatures]

        rooted = []
        for root in range(len(high)):
            colours = [int(v == root) for v in range(len(high))]
            history = []
            for _ in range(4):
                es = [(tag, tuple(sorted(colours[v] for v in vertices)))
                      for tag, vertices in zip(tags, ends)]
                ec = compress(es)
                vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                      for v in range(len(high))]
                history.append((tuple(sorted(Counter(es).items())),
                                tuple(sorted(Counter(vs).items()))))
                colours = compress(vs)
            rooted.append(sha256(repr(history).encode("ascii")).digest())

        value = (
            n, tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(rooted)), _tanner_signature(list(hx), list(hz)),
        )
        return sha256(repr(value).encode("ascii")).digest()

    outside, subgroup = _odd_inverse_classes(G)
    root_A = incidence(G, 6, outside, outside, ())
    normal_A = incidence(G, 6, (), (), tuple(sorted(subgroup)))
    bank = (root_A, normal_A)

    # All non-involutory inverse classes with order at least four.
    # Order-three edges would themselves give forbidden triangles.
    def order(g):
        power = G.e
        for t in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return t
        raise ValueError("invalid group element order")

    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if order(g) >= 4
    })
    assert sorted(order(g) for g in directions) == [4, 4, 4, 6]

    def canonical(top, bottom, voltages):
        """Remove common automorphisms, reversal and B-row exchange.

        This is only a known-orbit quotient. Full Tanner invariants
        perform an additional, conservative equivalence screen.
        """
        images = []
        for u in (1, 2):
            for v in range(3):
                for epsilon in (1, 3):
                    def image(g):
                        a, b = g
                        return ((u * a + v * (b % 2)) % 3,
                                epsilon * b % 4)

                    def gaps(values):
                        return tuple(sorted(
                            min(image(g), G.inv(image(g))) for g in values))

                    ta, tb = gaps(top), gaps(bottom)
                    vv = tuple(sorted(image(g) for g in voltages))
                    images.append((ta, tb, vv))
                    images.append((tb, ta,
                                   tuple(sorted(G.inv(g) for g in vv))))
        return min(images)

    cards = set()
    for internal in range(4):
        choices = list(combinations(directions, internal))
        cross = 7 - 2 * internal
        for top in choices:
            for bottom in choices:
                for voltages in combinations(sorted(G.elements), cross):
                    # The parent's internal-gap policy used only order
                    # four and a common cross-voltage parity. Explore
                    # the missing relative structures first.
                    six = any(order(g) == 6 for g in top + bottom)
                    mixed = len({g[1] % 2 for g in voltages}) > 1
                    if internal and not (six or mixed):
                        continue
                    cards.add(canonical(top, bottom, voltages))

    # Exclusion references only: all ten distinct current-table classes.
    # A-column permutations and binomial reversals do not change these
    # normalized incidence references.
    current_refs = (
        ((outside, outside, ()),
         (((0,1),(1,1)), ((2,1),(1,1)), ((1,2),(2,0),(0,2)))),
        ((outside, outside, ()),
         ((), (), ((2,3),(0,2),(1,3),(2,2),(1,2),(0,0),(2,0)))),
        ((outside, outside, ()),
         (((1,1),), ((0,1),), ((1,2),(2,2),(2,0),(0,2),(0,0)))),
        (((), (), tuple(sorted(subgroup))),
         (((1,1),(2,1)), ((2,1),(0,1)), ((2,2),(1,2),(0,0)))),
        ((((1,1),(2,1)), ((2,1),(1,1)), ((2,2),(2,0))),
         (((2,1),(1,1)), ((2,1),(1,1)), ((0,0),(2,2),(1,0)))),
        ((((2,1),), ((1,1),), ((1,0),(1,2),(2,0),(2,2))),
         (((2,1),(0,1)), ((0,1),(2,1)), ((2,2),(0,0),(1,2)))),
        ((((1,1),(2,1)), ((1,1),(0,1)), ((0,2),(2,2))),
         ((), (), ((0,3),(1,3),(1,1),(1,2),(2,0),(0,2),(2,3)))),
        ((((0,1),), ((2,1),), ((0,2),(0,0),(1,0),(2,2))),
         (((1,1),), ((0,1),), ((1,2),(2,0),(1,0),(2,2),(0,2)))),
        ((((0,1),(2,1)), ((2,1),(1,1)), ((2,0),(1,0))),
         (((2,1),(0,1),(1,1)), ((2,1),(1,1),(0,1)), ((2,2),))),
        ((((2,1),), ((0,1),), ((0,2),(2,0),(1,2),(1,0))),
         (((0,1),(1,1),(2,1)), ((2,1),(1,1),(0,1)), ((0,0),))),
    )
    graph_keys = set()
    for acard, bcard in current_refs:
        A = incidence(G, 6, *acard)
        B = incidence(G, 7, *bcard)
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(graph_key(hx, hz, n))

    # The shown common-C12 order-24 leaders are additional Tanner
    # references, not candidate groups or alternative submitted beds.
    def old_mul(g, h):
        return ((g[0] + pow(5, g[1], 12) * h[0]) % 12,
                (g[1] + h[1]) % 2)

    def old_inv(g):
        b = (-g[1]) % 2
        return ((-pow(5, b, 12) * g[0]) % 12, b)

    old = _G(dict(
        elements=[(a,b) for b in range(2) for a in range(12)],
        ident=(0,0), mul=old_mul, inv=old_inv), 24)
    old_A = [[[old.e, (d,0)] for d in (1,3,5)]]
    old_cards = (
        ((), (), (0,1,2,3,6,7,9)),
        ((1,), (5,), (0,2,4,6,8)),
        ((1,3), (3,5), (0,2,6)),
        ((1,3,5), (1,3,5), (0,)),
        ((1,3), (2,3), (0,4,8)),
    )
    for top, bottom, voltages in old_cards:
        B = incidence(old, 7, tuple((d,0) for d in top),
                      tuple((d,0) for d in bottom),
                      tuple((d,0) for d in voltages))
        hx, hz, n = lifted_matrices(old, old_A, B)
        graph_keys.add(graph_key(hx, hz, n))

    # Independently screen the finite constituent catalogue before
    # spending full-code invariant work on any pairing.
    constituents = []
    for card in sorted(cards):
        B = incidence(G, 7, *card)
        _, hz, _ = lifted_matrices(G, [[[G.e]]], B)
        br = [row & ((1 << (7 * G.q)) - 1) for row in hz[:2 * G.q]]
        info = component_four(br, 7 * G.q)
        if info is None or not info[0]:
            continue
        six = any(order(g) == 6 for g in card[0] + card[1])
        constituents.append((not six, info[1], card, freeze(B), info[0]))
    constituents.sort(key=lambda item: item[:3])

    candidates = []
    for policy, burden, card, B, witness in constituents:
        for ai, A in enumerate(bank):
            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k >= 256
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            priority = (-k, policy, burden, ai, card)
            candidates.append((priority, ai, B, witness, k, rx, rz))
    candidates.sort(key=lambda item: item[0])

    def four_logical(hx, hz, witness, n):
        """Embed an actual B kernel word and verify its logical class."""
        basis = {}
        for word in hx:
            while word:
                p = word.bit_length() - 1
                old_row = basis.get(p)
                if old_row is None:
                    basis[p] = word
                    break
                word ^= old_row

        for block in range(6):
            word = witness << (block * 7 * G.q)
            if word.bit_count() != 4:
                raise ValueError("four-witness embedding changed weight")
            if any((word & row).bit_count() & 1 for row in hz):
                raise ValueError("verified component embedding failed")
            remainder = word
            while remainder:
                p = remainder.bit_length() - 1
                old_row = basis.get(p)
                if old_row is None:
                    return word
                remainder ^= old_row
        return 0

    entries = []
    for priority, ai, B, witness, k, rx, rz in candidates:
        A = bank[ai]
        hx, hz, n = lifted_matrices(G, A, B)
        if not (_no_kernel_through_three(hx, n)
                and _no_kernel_through_three(hz, n)):
            continue
        logical = four_logical(hx, hz, witness, n)
        if not logical:
            continue
        key = graph_key(hx, hz, n)
        if key in graph_keys:
            continue

        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9
        assert all(sum(row) == 6 for row in sa)
        assert all(sum(row) == 7 for row in sb)
        assert all(len(set(c)) == len(c) <= 5
                   for grid in (A, B) for row in grid for c in row)
        assert css_ok(hx, hz)
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) == 9

        graph_keys.add(key)
        screen = (
            ("k", k), ("A_policy", ai), ("B_card", priority[-1]),
            ("component_four_burden", priority[2]),
            ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("exact_distance_certificate", 4),
            ("verified_X_four_witness", logical),
            ("shown_archive_references", len(current_refs) + len(old_cards)),
        )
        entries.append((freeze(A), freeze(B), screen))
        if len(entries) >= N_SLOTS:
            break

    # Exhaustion never triggers recycling, relabeling, or weakened gates.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete, globally ranked incidence design."""
    portfolio = _six_cycle_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("fresh certified-four six-cycle catalogue exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without sampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supply twenty inequivalent initial designs on C12:C2(s=5): slots 0..19 collapsed to four classes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: High-qubit-only rooted incidence screening supplies twenty inequivalent initial C3:C4 designs: current slots 0..19 retained seven classes, sixty slots retained ten, and all five measured leaders still had sector-1 distance-four witnesses.
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
