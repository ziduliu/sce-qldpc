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
CONCEPT = ("On frozen C12 semidirect_5 C2, order 24, A=(1,3), B=(2,7), deliberately coupling rotation and odd-coset A components to different index-two B subgroups should supply five novel k>=256 distance-four finalists, improving fitness above 7.3620 in cell 0.450.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. The parent measured five k=254,d30s=4 codes; the shown common-C12 programme measured k=258,d30s=4 but its codes are archived. The same frozen group has order-12 subgroups defined by even a and even a+b. Their relative left/right component couplings provide an algebraic experiment, not a transferred distance or rank guarantee.")
SHAPES = ("Keep A=(1,3), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.43478, in cell 0.450. The shape permits q<=26 and retains q=24. Constituents have 72 variables/24 checks and 168 variables/48 checks. All measured parent minima occupied sector-1 Acol0..2/Bcol0..6 broadly. Verified full ranks enforce [0.425,0.475), and complete kernel exclusion through weight three protects both sectors.")
FILLING = ("Optimize twenty regular support/subgroup prescriptions, rather than emit twenty independent draws. Two screened A topologies are paired with rotation, even-a, even-a+b and broader B policies; order-six internal edges are now admitted. Both B rows total seven and all columns total two, preserving HX weight eight and HZ weight nine. An immutable rank-prioritized portfolio rejects shown archive anchors and repeated intrinsic rooted contraction, low-qubit corner and check-intersection invariants. Five novel k=256 distance-four readings would score 7.42029; their yield, larger distances and novelty against unseen archive entries remain hypotheses.")
FAMILY_NAME = "r045_m12r2s5_13x27_parity_incidence_rank_catalog_v3_relativeH12"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 23 requires replacing the parent's order 55.
    m, r, s = 12, 2, 5
    if not valid_q(q):
        raise ValueError("this semidirect product has order 24")
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

A_SHAPE = (1, 3)               # (ra, ca)
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
    """Six undirected odd-parity generator classes, none involutory."""
    classes = {}
    for g in G.elements:
        if g[0] % 2 == 1:
            key = min(G.idx(g), G.idx(G.inv(g)))
            classes.setdefault(key, g)
    pools = [
        [g for _, g in sorted(classes.items()) if g[1] == b]
        for b in range(2)
    ]
    assert len(pools[0]) == len(pools[1]) == 3
    assert all(G.mul(g, g) != G.e for pool in pools for g in pool)
    return pools


def _draw_incidence(rng, G, attempt, pools):
    """Twenty structural cards, optimized rather than emitted as draws.

    A's graph is bipartite under a mod 2. B varies its internal-edge
    allocation. With internal edges present, cross-matchings have one
    common parity, so the B incidence graph is also bipartite.
    """
    card = attempt % 20
    epoch = attempt // 20
    outside_a = card % 4
    generators = (rng.sample(pools[0], 3 - outside_a)
                  + rng.sample(pools[1], outside_a))
    rng.shuffle(generators)
    A = [[[G.e, g] for g in generators]]

    internal = card // 5
    B = [[], []]
    for row in range(2):
        # Independent inverse classes exclude parallel internal edges.
        chosen = rng.sample(pools[0] + pools[1], internal)
        for g in chosen:
            B[row].append([G.e, g])
            B[1 - row].append([])

    cross_count = 7 - 2 * internal
    if internal:
        parity = epoch % 2
        ratio_pools = [
            [g for g in G.elements
             if g[1] == b and g[0] % 2 == parity]
            for b in range(2)
        ]
        outside_b = (card + epoch) % (cross_count + 1)
    else:
        ratio_pools = [
            [g for g in G.elements if g[1] == b]
            for b in range(2)
        ]
        outside_b = 1 + (card + epoch) % 6

    ratios = (rng.sample(ratio_pools[0], cross_count - outside_b)
              + rng.sample(ratio_pools[1], outside_b))
    rng.shuffle(ratios)
    for g in ratios:
        B[0].append([G.e])
        B[1].append([g])

    assert len(B[0]) == len(B[1]) == 7
    assert all(sum(map(len, row)) == 7 for row in B)
    assert all(len(B[0][j]) + len(B[1][j]) == 2 for j in range(7))
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


from functools import lru_cache


@lru_cache(maxsize=1)
def _relative_h12_portfolio(G, count):
    """Pure deterministic selection with immutable memoized results.

    Memoization changes build cost only. No slot mutates the portfolio
    or influences which design another slot receives.
    """
    from collections import Counter
    from hashlib import sha256

    local = random.Random(BASE_SEED ^ 0x2412C305)
    cap = min(9, W_MAX)
    if G.q != 24:
        raise ValueError("relative-H12 model requires the frozen order")
    sa = [[2, 2, 2]]
    sb = [[1] * 7, [1] * 7]
    if _grid_weights(sa, sb) > cap:
        raise ValueError("relative-H12 incidence requires cap nine")

    def element(index):
        return G.elements[index]

    def incidence(top, bottom, voltages):
        B = [[], []]
        for g in top:
            B[0].append([G.e, g])
            B[1].append([])
        for g in bottom:
            B[0].append([])
            B[1].append([G.e, g])
        for g in voltages:
            B[0].append([G.e])
            B[1].append([g])
        return B

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def intrinsic_key(hx, hz, n):
        """Uncoloured full-Tanner invariants, not an isomorphism test.

        High qubits are identified by degree. Each check contracts to
        an edge on them. Low-qubit corners retain all incident-check
        endpoints, including multiplicities. Rooted walk features use
        neither group coordinates nor protograph/X/Z labels.
        Equal keys are discarded conservatively.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, n)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        positions = {j: v for v, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        adjacency = [Counter() for _ in high]
        incident_weights = [Counter() for _ in high]
        check_ends = []

        for row in rows:
            bits = row & high_mask
            ends = []
            while bits:
                bit = bits & -bits
                ends.append(positions[bit.bit_length() - 1])
                bits ^= bit
            if len(ends) != 2:
                raise ValueError("regular incidence contraction changed")
            u, v = ends
            check_ends.append((u, v))
            adjacency[u][v] += 1
            adjacency[v][u] += 1
            incident_weights[u][row.bit_count()] += 1
            incident_weights[v][row.bit_count()] += 1

        labels = [
            (cols[j].bit_count(),
             tuple(sorted(incident_weights[v].items())))
            for v, j in enumerate(high)
        ]
        corners = []
        for j, syndrome in enumerate(cols):
            if j in positions:
                continue
            ends, weights = [], []
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                ends.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                syndrome ^= bit
            corners.append((tuple(sorted(weights)), tuple(ends)))

        neighbours = [tuple(a.items()) for a in adjacency]
        rooted = []
        for root in range(len(high)):
            vector = [0] * len(high)
            vector[root] = 1
            diagonal = []
            walks = [[] for _ in high]
            for length in range(1, 13):
                following = [0] * len(high)
                for u, value in enumerate(vector):
                    if value:
                        for v, multiplicity in neighbours[u]:
                            following[v] += value * multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 6:
                    for v, value in enumerate(vector):
                        walks[v].append(value)
            vertex_features = [
                (labels[v], tuple(walks[v])) for v in range(len(high))
            ]
            corner_features = Counter(
                (weights, tuple(sorted(vertex_features[v] for v in ends)))
                for weights, ends in corners
            )
            profile = (
                labels[root], tuple(diagonal),
                tuple(sorted(vertex_features)),
                tuple(sorted(corner_features.items())),
            )
            rooted.append(sha256(repr(profile).encode("ascii")).digest())

        return (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(rooted)),
            _tanner_signature(hx, hz),
        )

    # These are exclusion references only, never candidate outputs.
    # Packets encode A gaps, upper/lower B internal gaps, cross voltages.
    # Element indices are interpreted in THIS frozen group's ordering.
    shown_parent = (
        ((15,17,13), (), (), (19,21,22,17,0,15,5)),
        ((3,1,5), (17,), (1,), (0,20,6,22,2)),
        ((17,13,15), (13,1), (3,17), (18,20,22)),
        ((3,5,1), (13,1), (3,5), (18,22,4)),
        ((17,13,15), (15,13,3), (1,17,15), (14,)),
        ((13,15,17), (5,1), (17,15), (3,11,1)),
        ((3,13,5), (5,), (5,), (16,20,18,22,14)),
        ((15,3,13), (3,5), (1,3), (21,15,17)),
        ((13,15,3), (3,), (5,), (8,10,6,20,14)),
        ((3,5,17), (), (), (15,20,3,16,6,8,14)),
        ((15,3,5), (3,), (13,), (21,23,15,17,3)),
        ((3,17,15), (17,5), (15,13), (13,11,15)),
        ((1,17,3), (13,3,5), (1,13,15), (12,)),
        ((17,15,1), (13,17), (13,3), (0,12,14)),
        ((3,15,13), (5,13), (15,17), (2,6,20)),
        ((1,17,15), (15,1,17), (13,17,3), (6,)),
        ((5,17,3), (3,15,1), (5,15,3), (22,)),
        ((13,5,1), (17,15,13), (15,1,5), (11,)),
        ((5,3,17), (3,5), (3,3), (14,16,4)),
        ((17,3,15), (1,5), (5,17), (2,10,4)),
        ((15,1,3), (17,), (13,), (1,11,3,9,5)),
    )
    shown_common_c12 = (
        ((1,3,5), (), (), (0,1,2,3,6,7,9)),
        ((1,3,5), (1,), (5,), (0,2,4,6,8)),
        ((1,3,5), (1,3), (3,5), (0,2,6)),
        ((1,3,5), (1,3,5), (1,3,5), (0,)),
        ((1,3,5), (1,3), (2,3), (0,4,8)),
    )
    graph_keys = set()
    for gaps, top, bottom, voltages in shown_parent + shown_common_c12:
        A = [[[G.e, element(t)] for t in gaps]]
        B = incidence(tuple(map(element, top)),
                      tuple(map(element, bottom)),
                      tuple(map(element, voltages)))
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(intrinsic_key(hx, hz, n))

    # Both A models are regular incidence lifts. The odd-coset model
    # generates H={a+b even}, not the rotation subgroup.
    bank = []
    for gaps in (((1,0),(3,0),(5,0)),
                 ((1,1),(3,1),(5,1))):
        A = [[[G.e, g] for g in gaps]]
        hx, _, _ = lifted_matrices(G, A, [[[G.e]]])
        ar = [r & ((1 << (3 * G.q)) - 1) for r in hx]
        qa = _kernel_profile(ar, 3 * G.q)
        if qa[0] < 4:
            raise ValueError("prescribed A failed verified screening")
        bank.append((A, qa))

    rotations = [g for g in G.elements if g[1] == 0]
    h12 = [g for g in G.elements if (g[0] + g[1]) % 2 == 0]
    m12 = [g for g in G.elements if g[0] % 2 == 0]
    assert len(rotations) == len(h12) == len(m12) == 12

    def directions(pool):
        return sorted({
            min(g, G.inv(g)) for g in pool
            if g != G.e and G.mul(g, g) != G.e
        })

    def draw(attempt):
        card, epoch = attempt % 20, attempt // 20
        internal, policy = divmod(card, 5)
        ai = (epoch + card) % len(bank)
        A, qa = bank[ai]

        if policy == 0:
            labels = rotations
            steps = [g for g in directions(labels) if g[0] % 2]
            if internal:
                parity = epoch % 2
                labels = [g for g in labels if g[0] % 2 == parity]
        elif policy == 1:
            labels = h12
            steps = [g for g in directions(labels) if g[1] == 1]
            if internal:
                labels = [g for g in labels if g[1] == epoch % 2]
        elif policy == 2:
            labels = m12
            # x^2 has order six. A common grading constructs
            # triangle-free internal cycles without involution gaps.
            steps = [g for g in directions(labels)
                     if g[1] == 0 and (g[0] // 2) % 2]
            if internal:
                labels = [g for g in labels
                          if (g[0] // 2) % 2 == epoch % 2]
        else:
            labels = list(G.elements)
            steps = directions(labels)
            if policy == 3 and internal:
                steps = [g for g in steps if g[0] % 2]
                labels = [g for g in labels if g[0] % 2 == epoch % 2]
            # Policy four is not forced bipartite. Verified constituent
            # and full-matrix screens decide whether it is admissible.

        cross = 7 - 2 * internal
        if internal > len(steps) or cross > len(labels):
            return None
        top = local.sample(steps, internal)
        bottom = local.sample(steps, internal)
        voltages = local.sample(labels, cross)
        B = incidence(top, bottom, voltages)
        return A, B, qa, (ai, internal, policy)

    entries, filling_keys = [], set()
    wanted = max(32, 2 * count)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for batch in range(48):
        candidates = []
        # Three trials per structural prescription, then optimization.
        for trial in range(60):
            attempt = 60 * batch + trial
            proposal = draw(attempt)
            if proposal is None:
                continue
            A, B, qa, card = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            ac = [[len(c) for c in row] for row in A]
            bc = [[len(c) for c in row] for row in B]
            assert _no_dead_lines(ac, bc)
            assert _grid_weights(ac, bc) == 9
            assert all(sum(row) == 7 for row in bc)
            assert all(bc[0][j] + bc[1][j] == 2 for j in range(7))
            assert all(len(set(c)) == len(c) <= 5
                       for grid in (A, B) for row in grid for c in row)

            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _extract_components(G, hx, hz)
            qb = _kernel_profile(br, 7 * G.q)
            if qb[0] < 4:
                continue
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            priority = (-k, qa[1] + qb[1], attempt)
            candidates.append(
                (priority, A, B, hx, hz, n, k, rx, rz, qa, qb, card))

        candidates.sort(key=lambda record: record[0])
        for priority, A, B, hx, hz, n, k, rx, rz, qa, qb, card in candidates:
            key = intrinsic_key(hx, hz, n)
            if key in graph_keys:
                continue
            if not (_no_kernel_through_three(hx, n)
                    and _no_kernel_through_three(hz, n)):
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified relative-H12 lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            assert max(r.bit_count() for r in hx + hz) == 9
            graph_keys.add(key)
            screen = (
                ("k", k),
                ("design_card", card),
                ("component_screens", (qa, qb)),
                ("rank_gain_above_bound", k - 240),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("full_kernel_weights_1_to_3_excluded", True),
                ("shown_archive_references",
                 len(shown_parent) + len(shown_common_c12)),
            )
            entries.append((priority, freeze(A), freeze(B), screen))
        if len(entries) >= wanted:
            break

    # Compare the optimized portfolio globally; do not privilege the
    # first accepted subgroup card or recycle an exhausted position.
    entries.sort(key=lambda record: record[0])
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of an immutable, archive-screened relative-coset design."""
    portfolio = _relative_h12_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel relative-H12 incidence portfolio exhausted")
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
# REFUTED: The rank-first odd-incidence prefix supplies twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 retained four classes, and all five measured leaders had k=254,d30s=4 with sector-1 witnesses.
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
