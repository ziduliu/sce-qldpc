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
CONCEPT = ("Dic7, order 28, shapes (1,4)x(1,4): complete-incidence rooted refinement and joint degree-14 projection refinement should recover at least five archive-new inequivalent codes with certified distance three in cell 0.550.")
GROUP = ("Keep the frozen Dic7 presentation and order. The current five measured finalists all have k=252 and d30s=3, while sixty slots retained fourteen distinct classes. Improve joint catalogue diversity rather than claiming an unsupported group or distance advantage.")
SHAPES = ("Keep equal 112-variable, 28-check components: n=476, kb=252 and guaranteed rate 9/17=0.5294 in cell 0.550. The size window permits q<=35, while this round fixes q=28. Monomial pivots make k=252 exact. Current minima occupied sector-1 Acol/Bcol blocks, especially pivot columns; none were reported in sector 2. Certified distance-three finalists score exactly 81/17, so this revision targets novelty and slot yield, not a strict gain beyond that value.")
FILLING = ("Retain [1,2,2,2] on both sides, giving both check maxima nine. Interleave generated-subgroup pairings, remove common automorphism copies, and exclude reported current and earlier representatives. Strengthen local check-intersection invariants with pair-colour refinement of the intrinsic degree-14 projection and rooted refinement of the COMPLETE uncoloured bipartite incidence graph. Unequal signatures certify inequivalence; equality is discarded conservatively. Verified lifts retain CSS, exact k, complete commuting-weight-one/two exclusion and a non-stabilizer weight-three witness. Memoization only saves deterministic invariant computation; unreported archive novelty remains evaluator-tested.")
FAMILY_NAME = "r055_dic7_14x14_subgroup_joint_incidence_archive_certified3_v4"

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


from functools import lru_cache


@lru_cache(maxsize=1024)
def _dic_joint_incidence_key(hx, hz, q):
    """Deterministic intrinsic invariant of the UNCOLOURED Tanner graph.

    Memoization changes computation cost only, never catalogue ownership.
    Different keys certify non-isomorphism; equal keys are inconclusive.
    """
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    n = 17 * q
    m = len(rows)
    columns = _dic_columns(rows, n)

    # Preserve the parent's complete check-intersection information.
    one, many = [0] * m, [0] * m
    overlaps = []
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
        profiles[i][(rows[j].bit_count(), overlap) + common] += 1
        profiles[j][(rows[i].bit_count(), overlap) + common] += 1
    intersection = tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)
    ))

    # Degree fourteen intrinsically identifies the pivot-sector qubits.
    # Neither CSS colours nor protograph/group labels identify this class.
    high = [j for j, column in enumerate(columns)
            if column.bit_count() == 14]
    if len(high) != q:
        raise ValueError("certified support lost its degree-fourteen class")
    position = {j: i for i, j in enumerate(high)}
    high_mask = sum(1 << j for j in high)
    edges = [[Counter() for _ in high] for _ in high]
    anchors = [Counter() for _ in high]

    for row in rows:
        bits = row & high_mask
        if bits.bit_count() == 1:
            u = position[bits.bit_length() - 1]
            anchors[u][row.bit_count()] += 1
        elif bits.bit_count() == 2:
            first = bits & -bits
            u = position[first.bit_length() - 1]
            v = position[(bits ^ first).bit_length() - 1]
            edges[u][v][row.bit_count()] += 1
            edges[v][u][row.bit_count()] += 1
        else:
            raise ValueError("unexpected intrinsic projected incidence")

    markers = [tuple(sorted(anchor.items())) for anchor in anchors]
    initial = [
        (int(u == v), markers[u], markers[v],
         tuple(sorted(edges[u][v].items())))
        for u in range(q) for v in range(q)
    ]
    palette = {feature: i for i, feature in
               enumerate(sorted(set(initial)))}
    colours = [palette[feature] for feature in initial]
    pair_history = [tuple(sorted(Counter(initial).items()))]

    # Two-dimensional colour refinement keeps correlations between
    # the A and B actions that independent constituent statistics lose.
    for _ in range(4):
        features = []
        for u in range(q):
            for v in range(q):
                routes = Counter(
                    (colours[u * q + w], colours[w * q + v])
                    for w in range(q)
                )
                features.append((
                    colours[u * q + v], tuple(sorted(routes.items()))
                ))
        pair_history.append(tuple(sorted(Counter(features).items())))
        palette = {feature: i for i, feature in
                   enumerate(sorted(set(features)))}
        following = [palette[feature] for feature in features]
        stable = len(set(following)) == len(set(colours))
        colours = following
        if stable:
            break

    # Refine the COMPLETE bipartite graph, rooted separately at every
    # member of the intrinsic high-degree class. Sorting root signatures
    # removes root ordering. Initial colours use degree and root status.
    incidence = [[] for _ in range(m + n)]
    for j, column in enumerate(columns):
        vertex = m + j
        while column:
            bit = column & -column
            check = bit.bit_length() - 1
            incidence[check].append(vertex)
            incidence[vertex].append(check)
            column ^= bit
    degrees = [len(neighbours) for neighbours in incidence]
    rooted = []

    for j in high:
        root = m + j
        initial = [(degree, int(v == root))
                   for v, degree in enumerate(degrees)]
        palette = {feature: i for i, feature in
                   enumerate(sorted(set(initial)))}
        colours = [palette[feature] for feature in initial]
        history = [tuple(sorted(Counter(initial).items()))]
        for _ in range(5):
            features = [
                (colours[v],
                 tuple(sorted(colours[w] for w in neighbours)))
                for v, neighbours in enumerate(incidence)
            ]
            history.append(tuple(sorted(Counter(features).items())))
            palette = {feature: i for i, feature in
                       enumerate(sorted(set(features)))}
            following = [palette[feature] for feature in features]
            stable = len(set(following)) == len(set(colours))
            colours = following
            if stable:
                break
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    description = (
        tuple(sorted(Counter(degrees).items())),
        intersection, tuple(pair_history), tuple(sorted(rooted)),
    )
    return sha256(repr(description).encode("ascii")).digest()


def NEW_CTX(rng, G, idx):
    """Rebuild a pure, subgroup-interleaved archive-filtered prefix.

    Only deterministic invariant values are memoized. Selection has no
    mutable cross-slot state and does not depend on invocation order.
    """
    from itertools import combinations, combinations_with_replacement
    from math import gcd

    assert G.q == 28
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def graph_signature(hx, hz):
        return _dic_joint_incidence_key(tuple(hx), tuple(hz), G.q)

    # All thirteen distinct representatives in the reported parent table.
    # Each packet encodes its six binomial cells by exact element indices:
    # three A pairs followed by three B pairs; both pivots are identity.
    # These are exclusion references only, never candidate outputs.
    archived = (
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
    graph_keys = set()
    for packet in archived:
        sides = []
        for offset in (0, 6):
            row = [[G.e]]
            for p in range(offset, offset + 6, 2):
                row.append([G.elements[packet[p]],
                            G.elements[packet[p + 1]]])
            sides.append([row])
        hx, hz, _ = lifted_matrices(G, sides[0], sides[1])
        graph_keys.add(graph_signature(hx, hz))

    # All fourteen distinct representatives in the CURRENT table, plus
    # normalized references reported in the shown earlier Dic7 catalogues.
    # These are exclusion references only, never candidate seeds.
    normalized_archive = (
        ((1, 17, 20), (1, 19, 20)),
        ((3, 16, 19), (4, 5, 17)),
        ((1, 3, 4), (3, 4, 5)),
        ((1, 18, 20), (6, 18, 20)),
        ((1, 3, 6), (3, 17, 18)),
        ((1, 2, 5), (3, 6, 17)),
        ((4, 17, 20), (15, 18, 19)),
        ((4, 18, 20), (6, 16, 17)),
        ((2, 6, 15), (2, 17, 19)),
        ((3, 4, 19), (3, 6, 20)),
        ((1, 6, 19), (3, 4, 16)),
        ((5, 14, 19), (14, 19, 20)),
        ((1, 3, 15), (14, 15, 17)),
        ((2, 5, 15), (2, 5, 15)),
        ((1, 2, 15), (1, 14, 20)),
        ((2, 16, 19), (6, 16, 19)),
        ((1, 3, 4), (4, 6, 19)),
        ((2, 6, 16), (16, 18, 20)),
        ((3, 18, 19), (4, 5, 15)),
        ((3, 5, 19), (5, 6, 17)),
        ((3, 15, 17), (14, 15, 16)),
        ((1, 16, 19), (4, 15, 17)),
        ((5, 14, 18), (5, 18, 19)),
        ((3, 4, 6), (14, 17, 18)),
        ((5, 14, 19), (15, 17, 20)),
    )
    for a, b in normalized_archive:
        A = [[[G.e]] + [[G.e, G.elements[t]] for t in a]]
        B = [[[G.e]] + [[G.e, G.elements[t]] for t in b]]
        hx, hz, _ = lifted_matrices(G, A, B)
        graph_keys.add(graph_signature(hx, hz))

    # Thirteen inverse-pair directions: six noncentral rotations and
    # seven order-four directions. The involution would create duplicate
    # component columns, and is therefore excluded.
    directions = ([G.elements[a] for a in range(1, 7)]
                  + [G.elements[14 + a] for a in range(7)])
    triples = list(combinations(range(len(directions)), 3))

    def inverse_key(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    # All Dic7 automorphisms x -> x^u, y -> x^v y.
    maps = []
    for u in range(14):
        if gcd(u, 14) != 1:
            continue
        for v in range(14):
            maps.append(tuple(
                inverse_key(((u * a + v * b) % 14, b))
                for a, b in directions
            ))

    def orbit_key(a, b):
        return min(tuple(sorted((
            tuple(sorted(mapping[t] for t in a)),
            tuple(sorted(mapping[t] for t in b))
        ))) for mapping in maps)

    def component(triple):
        return [[[G.e]] +
                [[G.e, directions[t]] for t in triple]]

    def generated_order(triple):
        """Group-theoretic card label, not a Tanner inequivalence claim."""
        reached, frontier = {G.e}, [G.e]
        generators = [directions[t] for t in triple]
        while frontier:
            value = frontier.pop()
            for generator in generators:
                following = G.mul(value, generator)
                if following not in reached:
                    reached.add(following)
                    frontier.append(following)
        return len(reached)

    subgroup_orders = [generated_order(triple) for triple in triples]
    local = random.Random(BASE_SEED ^ 0xD7C1404)
    proposals = list(combinations_with_replacement(range(len(triples)), 2))
    local.shuffle(proposals)
    buckets = {}
    for ia, ib in proposals:
        profile = tuple(sorted((subgroup_orders[ia], subgroup_orders[ib])))
        buckets.setdefault(profile, []).append((ia, ib))

    # Deliberately interleave proper/proper, proper/full and full/full
    # constituent prescriptions before full-graph novelty screening.
    profiles = sorted(buckets, key=lambda p: (
        not (p[0] < G.q == p[1]), p
    ))
    plans = [
        buckets[profile][turn]
        for turn in range(max(map(len, buckets.values())))
        for profile in profiles if turn < len(buckets[profile])
    ]
    orbit_keys = set()
    accepted = 0

    for ia, ib in plans:
        ta, tb = triples[ia], triples[ib]
        orbit = orbit_key(ta, tb)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A, B = component(ta), component(tb)
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 476 or max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            continue
        signature = graph_signature(hx, hz)
        if signature in graph_keys:
            continue
        if not (_dic_excludes_two(hx, n)
                and _dic_excludes_two(hz, n)):
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 9 * G.q or 1000 * k <= 525 * n:
            continue

        # Validate the explicit witness against the evaluator's exact
        # matrices; do not infer quantum distance from a component graph.
        witness = _dic_three_witness(G, B)
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue

        graph_keys.add(signature)
        if accepted == idx:
            if not css_ok(hx, hz):
                raise ValueError("verified lift failed CSS screening")
            return {"A": A, "B": B, "k": k,
                    "distance_certificate": 3,
                    "directions": (ta, tb), "witness": witness,
                    "subgroup_card": tuple(sorted((
                        subgroup_orders[ia], subgroup_orders[ib]))),
                    "joint_incidence_key": signature.hex()}
        accepted += 1

    raise ValueError("archive-disjoint certified catalogue exhausted")


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell from the joint-incidence-screened design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct relative-order cards provide distinct Dic7 codes: the measured sixty-slot parent collapsed to thirteen distinct codes, with multiple primary cards marked iso_dup.
# REFUTED: The local check-intersection direction catalogue supplies twenty evaluator-distinct Dic7 designs: the current sixty-slot evaluation retained fourteen classes and forty-six iso_dup slots, with all five measured distances remaining three.
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
