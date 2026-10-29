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
CONCEPT = ("Dic7, order 28, A=(1,4), B=(1,4): exhaustive joint inverse-class prescriptions, prioritizing proper-subgroup/generating pairings and screened by rooted complete-Tanner refinement, should supply at least five archive-new certified-distance-three finalists in cell 0.550.")
GROUP = ("Keep the frozen Dic7 presentation and order byte-for-byte. Five current finalists measured [476,252,3], while sixty slots retained sixteen distinct classes. Orders 4, 7 and 14 remain available; joint structural selection replaces the current relative-order sampling catalogue.")
SHAPES = ("Keep equal 112-variable, 28-check components: n=476, kb=252 and guaranteed rate 9/17=0.529412 enter cell 0.550. The size window permits q<=35, but the frozen group fixes q=28. Current minimum logicals occupied sector 1, particularly Bcol0 for X and Acol0 for Z. Unit pivots retain exact k=252. The retained profile certifies d=3 and scores 81/17; strict fitness improvement is not promised.")
FILLING = ("Retain [1,2,2,2] on both sides, giving BOTH check maxima nine. Exhaustively enumerate joint inverse-distinct direction triples, prioritize proper-subgroup/generating pairs, and remove common automorphism and side-swap copies. Exclude all 56 Dic7 representatives explicitly reported here using complete uncoloured Tanner incidence with each intrinsic degree-fourteen qubit individualized in turn. Distinct invariant digests separate outputs conservatively. Verified lifts certify ranks, CSS, complete kernel exclusion through weight two and an actual non-stabilizer weight-three witness. An immutable memoized catalogue changes build cost only; unseen archive novelty and finalist yield remain evaluator-tested hypotheses.")
FAMILY_NAME = "r055_dic7_14x14_joint_mixed_fullroot_archive56_certified3_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic7 requires order 28")
    # x^14=1, y^2=x^7, y*x*y^-1=x^-1.
    # (a,b) encodes x^a y^b, with b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        return (g[0] % 14, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else -1) * c + 7*b*d) % 14,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a + 7) % 14, b)

    def idx(g):
        a, b = red(g)
        return 14*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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


def _retired_dic_cycle_context(rng, G, idx):
    """Inactive historical catalogue; never publishes candidate cells."""
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


@lru_cache(maxsize=1)
def _dic_fullroot_portfolio(G, count):
    """Pure deterministic selection, cached only to avoid rebuilding.

    Published grids are immutable. No slot changes another slot's
    prescription, exclusions, or accepted-catalogue position.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations, combinations_with_replacement

    if G.q != 28:
        raise ValueError("frozen Dic7 portfolio requires order twenty-eight")
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def component(labels):
        return [[[G.e]] +
                [[G.e, G.elements[label]] for label in labels]]

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the COMPLETE uncoloured Tanner graph.

        Individualize every degree-fourteen qubit, refine the entire
        incidence graph, and sort the rooted results. No X/Z colours,
        group labels, or protograph coordinates enter the invariant.
        Equal keys are inconclusive and rejected conservatively.
        """
        rows = list(hx) + list(hz)
        m = len(rows)
        cols = _dic_columns(rows, n)
        incidence = [[] for _ in range(m + n)]

        for j, column in enumerate(cols):
            vertex = m + j
            while column:
                bit = column & -column
                check = bit.bit_length() - 1
                incidence[check].append(vertex)
                incidence[vertex].append(check)
                column ^= bit

        degrees = [len(neighbors) for neighbors in incidence]
        roots = [m + j for j, column in enumerate(cols)
                 if column.bit_count() == 14]
        if len(roots) != G.q:
            raise ValueError("unit-binomial intrinsic degree class changed")
        degree_profile = tuple(sorted(Counter(degrees).items()))

        def rooted_digest(root):
            initial = [(degrees[v], int(v == root))
                       for v in range(len(incidence))]
            palette = {feature: c for c, feature in
                       enumerate(sorted(set(initial)))}
            colors = [palette[feature] for feature in initial]
            digest = sha256(repr(
                tuple(sorted(Counter(initial).items()))
            ).encode("ascii"))

            for _ in range(10):
                old_count = len(set(colors))
                features = [
                    (colors[v],
                     tuple(sorted(colors[u] for u in neighbors)))
                    for v, neighbors in enumerate(incidence)
                ]
                digest.update(repr(
                    tuple(sorted(Counter(features).items()))
                ).encode("ascii"))
                palette = {feature: c for c, feature in
                           enumerate(sorted(set(features)))}
                colors = [palette[feature] for feature in features]
                # Refinement includes the old colour and cannot merge
                # classes. A stable class count means stable partition.
                if len(palette) == old_count:
                    break
            return digest.digest()

        rooted = tuple(sorted(rooted_digest(root) for root in roots))
        return sha256(repr(
            (degree_profile, rooted)
        ).encode("ascii")).digest()

    # Exclusion references ONLY: all thirteen distinct direct-parent
    # representatives, with three A pairs followed by three B pairs.
    # The final packet uses the reported pair (11,26), correcting the
    # historical exclusion transcription without changing that ledger.
    packets = (
        (5,25, 1,17, 9,26, 3,22, 7,20, 5,14),
        (1,21, 0,14, 21,27, 6,22, 6,21, 14,26),
        (1,26, 7,14, 18,19, 10,21, 12,15, 16,19),
        (13,25, 15,27, 17,27, 11,23, 19,25, 16,26),
        (6,21, 15,21, 8,11, 9,21, 11,13, 2,11),
        (24,26, 16,22, 0,4, 15,17, 6,10, 2,10),
        (0,8, 0,2, 0,9, 15,25, 3,11, 14,27),
        (8,20, 1,17, 13,23, 4,21, 16,20, 16,21),
        (13,17, 9,22, 2,18, 26,27, 1,12, 4,9),
        (9,18, 5,24, 20,24, 10,19, 0,5, 1,4),
        (7,15, 7,19, 13,15, 9,22, 3,22, 15,23),
        (12,26, 5,7, 17,25, 2,8, 0,2, 19,23),
        (8,18, 0,15, 12,16, 12,23, 11,26, 12,21),
    )
    blocked = set()
    for packet in packets:
        sides = []
        for offset in (0, 6):
            row = [[G.e]]
            for p in range(offset, offset + 6, 2):
                row.append([G.elements[packet[p]],
                            G.elements[packet[p + 1]]])
            sides.append([row])
        hx, hz, n = lifted_matrices(G, sides[0], sides[1])
        blocked.add(graph_key(hx, hz, n))

    # All sixteen current, fourteen Program-3, and thirteen D2
    # representatives. Each packet is three normalized A directions
    # followed by three normalized B directions.
    shown = (
        # Current table.
        (14,19,20, 17,18,19),
        (2,17,18, 4,18,20),
        (3,15,20, 3,14,17),
        (1,6,20, 3,6,19),
        (1,5,18, 1,5,18),
        (3,4,6, 3,4,6),
        (1,3,6, 1,2,3),
        (14,18,20, 2,3,14),
        (17,18,20, 1,3,5),
        (4,18,20, 1,3,15),
        (1,6,20, 3,4,19),
        (1,6,17, 2,4,6),
        (14,18,19, 4,19,20),
        (6,14,20, 4,6,17),
        (14,18,19, 14,16,20),
        (4,18,20, 4,16,18),
        # Shown Program 3.
        (1,17,20, 1,19,20),
        (3,16,19, 4,5,17),
        (1,3,4, 3,4,5),
        (1,18,20, 6,18,20),
        (1,3,6, 3,17,18),
        (1,2,5, 3,6,17),
        (4,17,20, 15,18,19),
        (4,18,20, 6,16,17),
        (2,6,15, 2,17,19),
        (3,4,19, 3,6,20),
        (1,6,19, 3,4,16),
        (5,14,19, 14,19,20),
        (1,3,15, 14,15,17),
        (2,5,15, 2,5,15),
        # Shown D2.
        (1,2,15, 1,14,20),
        (2,16,19, 6,16,19),
        (1,3,4, 4,6,19),
        (2,6,16, 16,18,20),
        (3,18,19, 4,5,15),
        (3,5,19, 5,6,17),
        (3,15,17, 14,15,16),
        (1,16,19, 4,15,17),
        (5,14,18, 5,18,19),
        (5,16,17, 16,18,20),
        (3,4,6, 14,15,18),
        (3,4,18, 5,6,14),
        (5,14,19, 15,17,20),
    )
    for packet in shown:
        hx, hz, n = lifted_matrices(
            G, component(packet[:3]), component(packet[3:]))
        blocked.add(graph_key(hx, hz, n))

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    directions = sorted({
        inverse_class(g) for g in G.elements
        if _dic_order(G, g) in (4, 7, 14)
    })
    triples = list(combinations(directions, 3))

    def subgroup_size(labels):
        reached, frontier = {G.e}, [G.e]
        generators = [G.elements[label] for label in labels]
        while frontier:
            h = frontier.pop()
            for g in generators:
                t = G.mul(h, g)
                if t not in reached:
                    reached.add(t)
                    frontier.append(t)
        return len(reached)

    sizes = [subgroup_size(triple) for triple in triples]
    proper = [i for i, size in enumerate(sizes) if size < G.q]
    full = [i for i, size in enumerate(sizes) if size == G.q]
    mixed = [(a, b) for a in proper for b in full]
    remainder = [
        (a, b)
        for a, b in combinations_with_replacement(range(len(triples)), 2)
        if (sizes[a] == G.q) == (sizes[b] == G.q)
    ]

    # Exhaustive structural prescriptions, not independent slot draws.
    # Mixed constituents get first access to the novel catalogue.
    local = random.Random(BASE_SEED ^ 0xD7F05531)
    local.shuffle(mixed)
    local.shuffle(remainder)

    actions = [(u, v) for u in (1, 3, 5, 9, 11, 13)
               for v in range(14)]
    images = []
    for triple in triples:
        transformed = []
        for u, v in actions:
            labels = []
            for label in triple:
                a, b = G.elements[label]
                labels.append(inverse_class(
                    ((u * a + v * b) % 14, b)))
            transformed.append(tuple(sorted(labels)))
        images.append(tuple(transformed))

    def orbit_key(a, b):
        return min(tuple(sorted((x, y)))
                   for x, y in zip(images[a], images[b]))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    seen_orbits, entries = set(), []
    for ia, ib in mixed + remainder:
        orbit = orbit_key(ia, ib)
        if orbit in seen_orbits:
            continue
        seen_orbits.add(orbit)

        A, B = component(triples[ia]), component(triples[ib])
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 476:
            raise ValueError("frozen bed size changed")
        if max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            continue
        if not (_dic_excludes_two(hx, n)
                and _dic_excludes_two(hz, n)):
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if rx != 112 or rz != 112 or k != 252:
            continue
        witness = _dic_three_witness(G, B)
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue

        signature = graph_key(hx, hz, n)
        if signature in blocked:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified fullroot candidate failed CSS")

        blocked.add(signature)
        entries.append((
            freeze(A), freeze(B), witness,
            (triples[ia], triples[ib]),
            (sizes[ia], sizes[ib]), signature.hex()))
        if len(entries) >= count:
            break

    # Never pad a short catalogue with archived or relabelled copies.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication from the immutable joint catalogue."""
    entries = _dic_fullroot_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("archive-filtered fullroot Dic7 catalogue exhausted")
    A, B, witness, prescription, sizes, signature = entries[idx]
    return {
        "A": A, "B": B, "k": 252,
        "distance_certificate": 3,
        "witness": witness, "prescription": prescription,
        "component_subgroup_orders": sizes,
        "full_tanner_root_signature": signature,
        "reported_archive_references": 56,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without mutating the cached catalogue."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct Dic7 relative-order cards ensure distinct Tanner graphs: the parent produced only thirteen distinct codes among sixty slots, with numerous cross-card isomorphic duplicates.
# REFUTED: The current check-intersection/sphere catalogue supplies twenty evaluator-distinct slots: sixty slots retained sixteen classes and forty-four iso_dup slots; all five measured codes remained k=252,d30s=3.
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
