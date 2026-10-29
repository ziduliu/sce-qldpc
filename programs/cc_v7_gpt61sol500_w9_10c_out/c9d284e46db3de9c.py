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
CONCEPT = ("S4 x C2, order 48, A=B=(1,3): broader trinomial relative-order classes and exact stabilizer-aware absorption of short constituent cycles should supply five fresh codes with logical floor at least five in cell 0.400; higher-rank floor-four designs are reserves, not a promised improvement.")
GROUP = ("Frozen L1 group and order, retained byte-for-byte. This exact bed scored 6.6268: all five measured finalists had d30s=4, and sixty slots retained seventeen distinct codes. The order-55 bed's distance-five results are not transferable evidence for S4 x C2. Test new group-algebra fillings rather than changing the bed.")
SHAPES = ("Frozen equal (1,3) sides give 144-variable, 48-check constituents, n=480, kb=192 and guaranteed rate 0.400. These shapes allow q<=59; this round fixes q=48. All reported parent minima occupied sector-1 Acol/Bcol blocks broadly across the nine blocks. Both complete CSS logical spaces, including Arow0/Brow0, are screened, and actual ranks retain [0.375,0.425).")
FILLING = ("Enumerate binomial component orbits under conjugation, the central parity twist, inversion and column permutation. Enumerate normalized trinomial classes, including involutory relative quotients absent from the parent's generator policies. Twenty triangle/binomial prescriptions optimize actual k and exact nontrivial-logical exclusion through four. Counts remain (2,2,2)/(2,2,2) or (2,2,2)/(3,2,2), giving both weight maxima eight or nine. Prefer certified logical floor five, then larger-k floor-four reserves. Full uncoloured Tanner invariants conservatively separate outputs; reconstructed parent trinomial references are excluded, and pure-binomial reserves require k>=202, absent from the parent's table. Distance beyond the exact cutoff and novelty against unseen archive entries remain evaluator-tested.")
FAMILY_NAME = "r040_s4xc2_13x13_triangle_homology4_rankreserve_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S4 x C2 has order 48")
    from itertools import permutations
    # (p0,p1,p2,p3,b): permutation images and an independent C2 bit.
    perms = list(permutations(range(4)))
    els = [p + (b,) for b in range(2) for p in perms]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(g[:4]) + (g[4] % 2,)

    def mul(g, h):
        g, h = red(g), red(h)
        return tuple(g[h[i]] for i in range(4)) + (g[4] ^ h[4],)

    def inv(g):
        g = red(g)
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + (g[4],)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Pure publication from a deterministic immutable optimized portfolio."""
    portfolio = _s4_homology_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel S4 x C2 homology portfolio exhausted")
    A, B, k, floor, card = portfolio[idx]
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "screen": {"k": k, "certified_logical_floor": floor,
                   "design_card": card},
    }


from functools import lru_cache


@lru_cache(maxsize=1)
def _s4_homology_portfolio(G, count):
    """Memoization changes cost only; every published cell is immutable."""
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    assert G.q == 48
    q, width, e = G.q, 3 * G.q, G.e
    local = random.Random(BASE_SEED ^ 0x48A413)
    mask = (1 << width) - 1

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def component(grid, side):
        # Zero auxiliary cells isolate the VERIFIED constituent action.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        return [row & mask for row in rows[:q]]

    def kernel_feature(rows):
        """A short constituent witness; no quantum-distance claim."""
        cols = columns(rows, width)
        seen = {0: 0}
        floor, witness = 5, 0
        for j, syndrome in enumerate(cols):
            word = 1 << j
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif word != old:
                trial = word ^ old
                if trial.bit_count() < floor:
                    floor, witness = trial.bit_count(), trial
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                word = (1 << j) | (1 << i)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif word != old:
                    trial = word ^ old
                    if trial.bit_count() < floor:
                        floor, witness = trial.bit_count(), trial
                        if floor <= 2:
                            return floor, witness
        return floor, witness

    def logical_floor(checks, own, n):
        """Certify floor four or five, allowing short stabilizers.

        Every support through four splits into two supports through two.
        Equal syndromes must have equal canonical stabilizer residues.
        Values below four only signal rejection, not an exact distance.
        """
        pivots = basis(own)
        ordered = sorted(pivots, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in ordered:
                if (word >> p) & 1:
                    word ^= pivots[p]
            residues.append(word)

        cols = columns(checks, n)
        singles = {0: (0, 0)}
        for j, syndrome in enumerate(cols):
            value = (residues[j], 1 << j)
            old = singles.get(syndrome)
            if old is None:
                singles[syndrome] = value
            elif old[0] != value[0]:
                return 2

        pairs = {}
        four = False
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                residue = residues[j] ^ residues[i]
                old = singles.get(syndrome)
                if old is not None and old[0] != residue:
                    return 3
                old = pairs.get(syndrome)
                if old is None:
                    pairs[syndrome] = residue
                elif old != residue:
                    four = True
        return 4 if four else 5

    def graph_key(hx, hz, n):
        """Intrinsic uncoloured full-Tanner invariants.

        Unequal keys certify nonisomorphism, apart from hash collisions.
        Equal keys are conservatively merged, not declared isomorphic.
        """
        rows = hx + hz
        m = len(rows)
        degrees = [row.bit_count() for row in rows]
        adjacency = [0] * m
        one, many = [0] * m, [0] * m
        overlaps = {}
        for i, row in enumerate(rows):
            for j in range(i):
                shared = (row & rows[j]).bit_count()
                if shared:
                    overlaps[i, j] = shared
                    relation = one if shared == 1 else many
                    relation[i] |= 1 << j
                    relation[j] |= 1 << i
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        profiles = [Counter() for _ in rows]
        for i in range(m):
            for j in range(i):
                feature = (
                    overlaps.get((i, j), 0),
                    (one[i] & one[j]).bit_count(),
                    (one[i] & many[j]).bit_count()
                    + (many[i] & one[j]).bit_count(),
                    (many[i] & many[j]).bit_count())
                profiles[i][(degrees[j],) + feature] += 1
                profiles[j][(degrees[i],) + feature] += 1

        features = [
            (degrees[i], tuple(sorted(profiles[i].items())))
            for i in range(m)]
        palette = {f: j for j, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]
        qubits = Counter()
        for syndrome in columns(rows, n):
            neighborhood = []
            while syndrome:
                bit = syndrome & -syndrome
                neighborhood.append(colors[bit.bit_length() - 1])
                syndrome ^= bit
            qubits[tuple(sorted(neighborhood))] += 1

        spheres = []
        for root in range(m):
            visited = frontier = 1 << root
            sizes = []
            while frontier:
                following = 0
                bits = frontier
                while bits:
                    bit = bits & -bits
                    following |= adjacency[bit.bit_length() - 1]
                    bits ^= bit
                frontier = following & ~visited
                if frontier:
                    sizes.append(frontier.bit_count())
                    visited |= frontier
            spheres.append((degrees[root], tuple(sizes)))

        value = (
            tuple(sorted(features)), tuple(sorted(qubits.items())),
            tuple(sorted(spheres)))
        return sha256(repr(value).encode("ascii")).digest()

    def order(g):
        power = e
        for length in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return length
        raise ValueError("invalid group element order")

    orders = [order(g) for g in G.elements]

    def parity(g):
        return sum(g[i] > g[j]
                   for i in range(4) for j in range(i + 1, 4)) % 2

    # All automorphisms needed here: S4 conjugation and the independent
    # central twist (p,b)->(p,b+sign(p)). This does not edit CUSTOM_GROUP.
    actions = []
    for p in [g for g in G.elements if g[4] == 0]:
        for twist in (0, 1):
            image = []
            for g in G.elements:
                h = G.mul(G.mul(p, g), G.inv(p))
                h = h[:4] + (h[4] ^ (twist * parity(h)),)
                image.append(G.idx(h))
            actions.append(image)

    inverse = [G.idx(G.inv(g)) for g in G.elements]
    directions = sorted({
        min(j, inverse[j]) for j in range(q)
        if orders[j] in (3, 4, 6)
    })
    component_keys = set()
    for triple in combinations(directions, 3):
        key = min(
            tuple(sorted(min(action[j], inverse[action[j]])
                         for j in triple))
            for action in actions)
        component_keys.add(key)

    bank = []
    for key in sorted(component_keys):
        grid = [[[e, G.elements[j]] for j in key]]
        rows = component(grid, "A")
        floor, witness = kernel_feature(rows)
        if floor >= 3:
            bank.append((key, grid, floor, witness, gf2_rank(rows)))
    bank.sort(key=lambda r: (-r[2], -r[4], r[0]))
    if not bank:
        raise ValueError("no admissible binomial constituent")

    # Normalize a triangle at each of its three vertices, then quotient
    # automorphisms. Distinct prototypes are not random anchor images.
    triangle_keys = set()
    nonidentity = [j for j in range(q) if G.elements[j] != e]
    for r, s in combinations(nonidentity, 2):
        cell = [e, G.elements[r], G.elements[s]]
        candidates = []
        for anchor in cell:
            normalized = [
                G.idx(G.mul(G.inv(anchor), g))
                for g in cell if g != anchor]
            candidates.extend(
                tuple(sorted(action[j] for j in normalized))
                for action in actions)
        triangle_keys.add(min(candidates))

    triangles = []
    for key in sorted(triangle_keys):
        r, s = [G.elements[j] for j in key]
        spectrum = tuple(sorted((
            orders[key[0]], orders[key[1]],
            orders[G.idx(G.mul(G.inv(r), s))])))
        rows, _, _ = lifted_matrices(G, [[[e, r, s]]], [[[]]])
        rank = gf2_rank([row & ((1 << q) - 1) for row in rows])
        triangles.append((spectrum, rank, key))
    # Explore involutory triangles first, but retain every triangle class.
    triangles.sort(key=lambda t: (
        -t[0].count(2), t[0], t[1], t[2]))

    excluded = set()
    # The five distinct asymmetric references shown in the parent table.
    # Reproduction is used ONLY for exclusion and never for publication.
    for slot in (16, 17, 18, 19, 57):
        try:
            old = _s4_parent_reference(
                random.Random(BASE_SEED * 10007 + slot), G, slot)
            A, B = old["cells"]["A"], old["cells"]["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            excluded.add(graph_key(hx, hz, n))
        except ValueError:
            pass

    entries, fillings = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def inspect(A, B, afloor, aword, card, pure=False):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            return
        fillings.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb):
            return
        if _grid_weights(sa, sb) > min(9, W_MAX):
            return

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 480 or not (375*n <= 1000*k < 425*n):
            return
        # Parent pure-binomial codes had k<=200. Reserve only the
        # previously unreported higher-rank branch, not its orbit copies.
        if pure and k < 202:
            return

        bx, bz = basis(hx), basis(hz)

        def embedded_upper(word, kind):
            upper = n + 1
            if not word:
                return upper
            for block in range(3):
                if kind == "X":
                    trial = word << (block * width)
                    checks, own = hz, bx
                else:
                    trial, source = 0, word
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        trial |= 1 << ((3*j + block)*q + h)
                        source ^= bit
                    checks, own = hx, bz
                assert all((row & trial).bit_count() % 2 == 0
                           for row in checks)
                if outside(trial, own):
                    upper = min(upper, trial.bit_count())
            return upper

        if embedded_upper(aword, "Z") < 4:
            return
        br = [row & mask for row in hz[:q]]
        bfloor, bword = kernel_feature(br)
        if embedded_upper(bword, "X") < 4:
            return

        fx = logical_floor(hz, hx, n)
        if fx < 4:
            return
        fz = logical_floor(hx, hz, n)
        if fz < 4:
            return
        floor = min(fx, fz)
        signature = graph_key(hx, hz, n)
        if signature in excluded:
            return
        excluded.add(signature)

        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        entry = (freeze(A), freeze(B), k, floor, card)
        quality = (k * floor * floor / n, floor, k)
        entries.append((quality, signature, entry))

    # New rank-surplus reserves. Relative central twists are tested as
    # actual matrices; the Tanner invariant decides whether they differ.
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            for twist in (0, 1):
                B = [[[
                    g[:4] + (g[4] ^ (twist * parity(g)),)
                    for g in cell
                ] for cell in b[1][0]]]
                inspect(a[1], B, a[2], a[3],
                        ("rank-reserve", ai, bi, twist), pure=True)

    binomial_policies = ((4, 4), (4, 6), (6, 6),
                         (3, 4), (3, 6), (3, 3))
    pools = {
        degree: [G.elements[j] for j in range(q) if orders[j] == degree]
        for degree in (3, 4, 6)
    }
    # Twenty intrinsic triangle/order prescriptions. Additional rounds
    # vary their relative coupling and A component, not a global gauge.
    trials = max(96, 3 * count)
    for card in range(20):
        spectrum, triangle_rank, key = triangles[card % len(triangles)]
        o0, o1 = binomial_policies[card % len(binomial_policies)]
        for trial in range(trials):
            action = actions[local.randrange(len(actions))]
            r, s = [G.elements[action[j]] for j in key]
            g, h = local.choice(pools[o0]), local.choice(pools[o1])
            if g == h or g == G.inv(h):
                continue
            B = [[[e, r, s], [e, g], [e, h]]]
            a = bank[(card + trial) % len(bank)]
            inspect(a[1], B, a[2], a[3],
                    ("triangle-card", card, spectrum, triangle_rank,
                     o0, o1, trial))

    # Floors are exact acceptance results, not unresolved probe caps.
    # Sorting by their conservative score favours five-floor candidates
    # before most four-floor reserves; the evaluator measures the result.
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(item[2] for item in entries[:count])


def _s4_parent_reference(rng, G, idx):
    """Inactive parent selector, retained only to reconstruct exclusions."""
    card, epoch = idx % 20, idx // 20
    count_a = card % 4
    reinforced = card >= 16
    count_b = card // 4 if not reinforced else card % 3
    sa = ((2, 2, 2),)
    sb = ((3, 2, 2),) if reinforced else ((2, 2, 2),)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(W_MAX, 9)

    def element_order(g):
        power = G.e
        for order in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return order
        raise ValueError("invalid finite-group element")

    by_order = {order: [] for order in (3, 4, 6)}
    for g in G.elements:
        order = element_order(g)
        if order in by_order:
            by_order[order].append(g)
    central = (0, 1, 2, 3, 1)
    triple_orders = ((3, 4), (3, 6), (4, 6), (6, 6))

    def draw_grid(side, support):
        count4 = count_a if side == "A" else count_b
        grid = [[]]
        previous = []
        for j, count in enumerate(support[0]):
            first = G.pick(rng)
            if count == 2:
                order = 4 if j < count4 else 6
                choices = list(by_order[order])
                # Additional harness slots change an intrinsic
                # commutation preference, not a global relabeling.
                if previous and epoch:
                    commute = epoch % 2 == 0
                    restricted = [
                        h for h in choices
                        if (G.mul(previous[-1], h)
                            == G.mul(h, previous[-1])) == commute
                    ]
                    if restricted:
                        choices = restricted
                ratio = rng.choice(choices)
                previous.append(ratio)
                cell = [first, G.mul(first, ratio)]
            else:
                oa, ob = triple_orders[card % 4]
                r = rng.choice(by_order[oa])
                choices = [
                    s for s in by_order[ob]
                    if G.mul(r, s) != G.mul(s, r)
                    and G.mul(G.inv(r), s) != central
                ]
                s = rng.choice(choices)
                cell = [first, G.mul(first, r), G.mul(first, s)]
            assert len(set(cell)) == count
            grid[0].append(cell)
        return grid

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def component_quality(rows, width):
        """Small-kernel collision count and Tanner 4/6-cycle penalties."""
        cols = columns(rows, width)
        seen = {0}
        bad = 0
        for s in cols:
            bad += int(s in seen)
            seen.add(s)
        c4 = 0
        for i, s in enumerate(cols):
            for t in cols[:i]:
                u = s ^ t
                bad += int(u in seen)
                seen.add(u)
                overlap = (s & t).bit_count()
                c4 += overlap * (overlap - 1) // 2

        # Project onto check nodes. With no 4-cycles, subtracting
        # within-column triangles leaves exactly the Tanner 6-cycles.
        adj = [0] * len(rows)
        internal = 0
        for s in cols:
            degree = s.bit_count()
            internal += degree * (degree - 1) * (degree - 2) // 6
            bits = s
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= s ^ bit
                bits ^= bit
        triangles = 0
        for u, neighbors in enumerate(adj):
            bits = neighbors >> (u + 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() + u
                triangles += ((neighbors & adj[v]) >> (v + 1)).bit_count()
                bits ^= bit
        return bad, c4, max(0, triangles - internal)

    def no_kernel6(rows, width):
        """Exact absence test for constituent kernel words of weight <=6.

        All such supports split into two supports of size <=3.
        Different supports with equal syndromes expose a nonzero word.
        A False result is only a failed screen, not a minimum distance.
        """
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return False
        pairs = set()
        for i, s in enumerate(cols):
            for j in range(i):
                syndrome = s ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        triples = set()
        for i, s in enumerate(cols):
            for j in range(i):
                pair = s ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if (syndrome == 0 or syndrome in singles
                            or syndrome in pairs or syndrome in triples):
                        return False
                    triples.add(syndrome)
        return True

    def low_logical(checks, stabilizers, width, through=4):
        """Exact logical existence test through weight two or four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        syndromes = columns(checks, width)
        representatives = {0: 0}
        for i, syndrome in enumerate(syndromes):
            word = 1 << i
            if syndrome in representatives:
                if nontrivial(word ^ representatives[syndrome]):
                    return True
            else:
                representatives[syndrome] = word
        if through == 2:
            return False
        for i, syndrome in enumerate(syndromes):
            bit = 1 << i
            for j in range(i):
                s = syndrome ^ syndromes[j]
                word = bit | (1 << j)
                if s in representatives:
                    if nontrivial(word ^ representatives[s]):
                        return True
                else:
                    representatives[s] = word
        # All words of weight <=4 split into two words of weight <=2.
        # One representative per syndrome suffices because the
        # stabilizer space is linear.
        return False

    def basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                if p in pivots:
                    word ^= pivots[p]
                else:
                    pivots[p] = word
                    break
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    def logical_probe(checks, stabilizers, width, rounds):
        """Information-set upper bound; every accepted witness is checked."""
        cols = columns(checks, width)
        own = basis(stabilizers)
        order = list(range(width))
        best = width + 1
        for _ in range(rounds):
            rng.shuffle(order)
            pivots, nulls = {}, []
            for j in order:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = (syndrome, word)
                        break
                    old_s, old_w = pivots[p]
                    syndrome ^= old_s
                    word ^= old_w
                if not syndrome:
                    nulls.append(word)
            short = sorted(nulls, key=int.bit_count)[:24]
            for i, word in enumerate(short):
                for candidate in [word] + [
                        word ^ other for other in short[:i]]:
                    weight = candidate.bit_count()
                    if candidate and weight < best and outside(candidate, own):
                        assert all(
                            (row & candidate).bit_count() % 2 == 0
                            for row in checks)
                        best = weight
            if best <= 3:
                break
        return best

    q = G.q
    width = 3 * q
    component_mask = (1 << width) - 1
    unit = [[[G.e]]]

    def component_pool(side, support):
        proposals, seen = [], set()
        for trial in range(48):
            grid = draw_grid(side, support)
            key = tuple(tuple(sorted(cell)) for cell in grid[0])
            if key in seen:
                continue
            seen.add(key)
            if side == "A":
                rows, _, _ = lifted_matrices(G, grid, unit)
            else:
                _, rows, _ = lifted_matrices(G, unit, grid)
            rows = [row & component_mask for row in rows]
            quality = component_quality(rows, width)
            proposals.append((quality, trial, grid, rows))
        proposals.sort(key=lambda item: (item[0], item[1]))
        return proposals[:5]

    best, best_quality = None, None
    for batch in range(3):
        apool = component_pool("A", sa)
        bpool = component_pool("B", sb)
        pairings = sorted(
            ((a[0][0] + b[0][0],
              a[0][1] + b[0][1],
              a[0][2] + b[0][2], ai, bi)
             for ai, a in enumerate(apool)
             for bi, b in enumerate(bpool))
        )
        tested = 0
        for _, _, _, ai, bi in pairings:
            qa, _, A, ar = apool[ai]
            qb, _, B, br = bpool[bi]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 375 * n <= 1000 * k < 425 * n):
                continue
            if (low_logical(hz, hx, n, through=2)
                    or low_logical(hx, hz, n, through=2)):
                continue

            clean4 = not (
                low_logical(hz, hx, n)
                or low_logical(hx, hz, n))
            dx = logical_probe(hz, hx, n, 8)
            dz = logical_probe(hx, hz, n, 8)
            reading = min(dx, dz)
            # An exact failed weight-four screen supplies an upper
            # bound even when the randomized probe misses its witness.
            if not clean4:
                reading = min(reading, 4)
            quality = (
                int(clean4), k * reading * reading / n,
                -(qa[0] + qb[0]), -(qa[1] + qb[1]),
                -(qa[2] + qb[2]),
            )
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best = (A, B, hx, hz, k, clean4, dx, dz)
            tested += 1
            if tested >= 8:
                break
        if best is not None and best[5]:
            break

    if best is None:
        raise ValueError("No target-rate filling excluded weight-one/two logicals")
    A, B, hx, hz, k, clean4, dx, dz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "screen": {"k": k, "no_logical4": clean4,
                   "probe_X": dx, "probe_Z": dz,
                   "selection": best_quality},
    }


def CELL(rng, G, side, i, j, ctx):
    """Copy one full cell without modifying the immutable portfolio."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: The Dic11 parent's 2-second distance reading always survives the longer search: slot 42 fell from 22 to 20, with mixed-sector Z witnesses.
# REFUTED: The S4 x C2 parent's bounded support-card search always emits a valid candidate: fourteen slots failed its weight-four screen, and additional slots collapsed as isomorphic duplicates.
# REFUTED: Order-four/order-six binomial cards with limited noncommuting trinomial reinforcement raise this S4 x C2 (1,3)x(1,3) bed beyond distance reading four: all five measured finalists had d30s=4, with minima entirely in sector 1, and sixty slots retained only seventeen distinct codes.
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
