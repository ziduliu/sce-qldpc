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
CONCEPT = ("C12 semidirect C2 with action 5, order 24, A=(1,3), B=(2,7): exhaustive order-twelve matching orbits and order-eight Sylow-supported B components should supply five novel distance-four finalists with mean k>256, improving fitness above 7.4200 in cell 0.450.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This bed measured k=258,d30s=4. The outside-A closure <x^2,xy> has order twelve, while <x^3,x^a*y> gives order-eight subgroups isomorphic to C4 x C2. Their different component-connectivity prescriptions motivate a rank-yield experiment; verified full ranks, not subgroup orders, determine its benefit.")
SHAPES = ("Keep A=(1,3), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.43478. The shape permits q<=26 and the frozen lift uses q=24. Unequal constituents have 72 variables/24 checks and 168 variables/48 checks. Parent weight-four minima broadly occupied sector-1 Acol0..2/Bcol0..6, especially Acol1 and Bcol5. Actual ranks retain [0.425,0.475); improving novel rank-deficit yield, rather than promising distance above four, is the objective.")
FILLING = ("Exhaust seven-matching sets in both order-twelve A closures modulo explicit B gauges and A-preserving group automorphisms. Enumerate Sylow-supported seven-matchings and four-internal-column B designs, then optimize twenty broader support/quotient prescriptions. A has three binomials; B rows total seven and columns at most three, so BOTH CSS sums stay <=9. A shared immutable portfolio globally prioritizes verified k and excludes complete HX/HZ kernels through weight three. Rooted intrinsic Tanner invariants conservatively reject repeats and shown archive anchors. No ownership lottery discards a promising design. Five fresh k=258,d30s=4 codes would score 7.47826; their yield and novelty against the unseen archive remain hypotheses.")
FAMILY_NAME = "r045_m12r2s5_13x27_logicalquotient_mixeddegree_v4_subgroup_orbits_globalrank"

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
def _subgroup_rank_portfolio(G, count):
    """Deterministic immutable portfolio; slot order affects no selection."""
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    local = random.Random(BASE_SEED ^ 0x24812704)
    q, e = G.q, G.e
    cap = min(9, W_MAX)
    if q != 24 or cap < 9:
        raise ValueError("subgroup portfolio requires the frozen q24/cap9 bed")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def closure(generators):
        reached, pending = {e}, [e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        for first in pending:
            for second in steps:
                product = G.mul(first, second)
                if product not in reached:
                    reached.add(product)
                    pending.append(product)
        return tuple(sorted(reached))

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def graph_key(hx, hz, width):
        """Intrinsic rooted high-qubit/corner invariant, without LP labels.

        Different keys certify Tanner inequivalence. Equal keys are
        discarded conservatively; this is not a complete isomorphism test.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, width)
        high = [j for j, c in enumerate(cols) if c.bit_count() == 13]
        position = {j: v for v, j in enumerate(high)}
        ends, tags, check_ends = [], [], []

        for row in rows:
            endpoints = []
            bits = row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in position:
                    endpoints.append(position[j])
                bits ^= bit
            endpoints = tuple(sorted(endpoints))
            check_ends.append(endpoints)
            ends.append(endpoints)
            tags.append((0, row.bit_count()))

        for j, syndrome in enumerate(cols):
            if j in position:
                continue
            endpoints, weights = [], []
            bits = syndrome
            while bits:
                bit = bits & -bits
                r = bit.bit_length() - 1
                endpoints.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                bits ^= bit
            ends.append(tuple(sorted(endpoints)))
            tags.append((1, tuple(sorted(weights))))

        incident = [[] for _ in high]
        for edge, endpoints in enumerate(ends):
            for v in endpoints:
                incident[v].append(edge)

        def compress(values):
            palette = {v: i for i, v in enumerate(sorted(set(values)))}
            return [palette[v] for v in values]

        roots = []
        for root in range(len(high)):
            colors = [int(v == root) for v in range(len(high))]
            history = []
            for _ in range(4):
                es = [(tag, tuple(sorted(colors[v] for v in ep)))
                      for tag, ep in zip(tags, ends)]
                ec = compress(es)
                vs = [(colors[v],
                       tuple(sorted(ec[t] for t in incident[v])))
                      for v in range(len(high))]
                history.append((tuple(sorted(Counter(es).items())),
                                tuple(sorted(Counter(vs).items()))))
                colors = compress(vs)
            roots.append(sha256(repr(history).encode("ascii")).digest())
        return (
            width,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(roots)),
            _tanner_signature(hx, hz),
        )

    directions = tuple(sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != e and G.mul(g, g) != e
    }))
    bank = []
    for triple in combinations(directions, 3):
        A = [[[e, g] for g in triple]]
        hx, _, _ = lifted_matrices(G, A, [[[e]]])
        ar = [r & ((1 << (3*q)) - 1) for r in hx[:q]]
        qa = _kernel_profile(ar, 3*q)
        if qa[0] < 4:
            continue
        bank.append((q - gf2_rank(ar), qa[1], triple, freeze(A)))
    bank.sort(key=lambda r: (-r[0], r[1], r[2]))
    if not bank:
        raise ValueError("no triangle-free binomial A component")

    root_triples = (
        ((1, 0), (3, 0), (5, 0)),
        ((1, 1), (3, 1), (5, 1)),
    )
    roots = [next(r for r in bank if r[2] == triple)
             for triple in root_triples]
    spanning = [r for r in bank if r[0] == 1] or bank

    def incidence(top, bottom, voltages):
        B = [[], []]
        for g in top:
            B[0].append([e, g])
            B[1].append([])
        for g in bottom:
            B[0].append([])
            B[1].append([e, g])
        for g in voltages:
            B[0].append([e])
            B[1].append([g])
        return B

    def matching_orbits(record):
        """Exhaust all seven-subsets, removing explicit safe orbit copies.

        Left/right B-row gauges preserve A's left action. The three
        automorphism generators below preserve each root A inverse-class
        set. Inversion represents B-row exchange.
        """
        subgroup = closure(record[2])
        labels = tuple(sorted(G.idx(g) for g in subgroup))
        assert len(labels) == 12
        actions = []
        for g in record[2]:
            actions.append({t: G.idx(G.mul(g, G.elements[t]))
                            for t in labels})
            actions.append({t: G.idx(G.mul(G.elements[t], g))
                            for t in labels})
        for u, v in ((5, 0), (7, 0), (1, 2)):
            image = {}
            for t in labels:
                a, b = G.elements[t]
                image[t] = G.idx(((u*a + v*b) % 12, b))
            assert set(image.values()) == set(labels)
            actions.append(image)
        actions.append({t: G.idx(G.inv(G.elements[t])) for t in labels})

        remaining = set(combinations(labels, 7))
        representatives = []
        while remaining:
            first = min(remaining)
            remaining.remove(first)
            representatives.append(first)
            pending = [first]
            for current in pending:
                for action in actions:
                    image = tuple(sorted(action[t] for t in current))
                    if image in remaining:
                        remaining.remove(image)
                        pending.append(image)
        return tuple(representatives)

    # Exhaustive high-rank prescriptions, before random optimization.
    prototypes = []
    for root_index, record in enumerate(roots):
        for labels in matching_orbits(record):
            B = incidence((), (), tuple(G.elements[t] for t in labels))
            prototypes.append((record, B, ("H12-orbit", root_index, labels)))

    sylows = [closure(((3, 0), (a, 1))) for a in range(3)]
    assert all(len(H) == 8 for H in sylows)
    for root_index, record in enumerate(roots):
        for h_index, H in enumerate(sylows):
            for omitted in H:
                voltages = tuple(g for g in H if g != omitted)
                prototypes.append((
                    record, incidence((), (), voltages),
                    ("H8-seven-matchings", root_index, h_index,
                     G.idx(omitted))))
            gaps = tuple(sorted({
                min(g, G.inv(g)) for g in H
                if g != e and G.mul(g, g) != e
            }))
            assert len(gaps) == 2
            for parity in range(2):
                voltages = tuple(g for g in H if g[0] % 2 == parity)
                assert len(voltages) == 4
                for chosen in combinations(voltages, 3):
                    prototypes.append((
                        record, incidence(gaps, gaps, chosen),
                        ("H8-four-internal", root_index, h_index, parity,
                         tuple(G.idx(g) for g in chosen))))

    # Shown archive exclusions only. No excluded filling is published.
    graph_keys = set()
    reference_cards = (
        (0, (), (), ((0,0),(1,0),(2,0),(3,0),(6,0),(7,0),(9,0))),
        (0, ((1,0),), ((5,0),), ((0,0),(2,0),(4,0),(6,0),(8,0))),
        (0, ((1,0),(3,0)), ((3,0),(5,0)), ((0,0),(2,0),(6,0))),
        (0, ((1,0),(3,0),(5,0)), ((1,0),(3,0),(5,0)), ((0,0),)),
        (0, ((1,0),(3,0)), ((2,0),(3,0)), ((0,0),(4,0),(8,0))),
        (1, (), (), ((8,0),(9,0),(2,0),(6,0),(5,0),(10,0),(11,0))),
        (1, ((3,1),(1,1)), ((9,1),(11,1)), ((6,0),(2,0),(10,0))),
    )
    for ai, top, bottom, voltages in reference_cards:
        A = roots[ai][3]
        B = incidence(top, bottom, voltages)
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(graph_key(hx, hz, n))

    odd = tuple(g for g in directions if g[0] % 2)
    fillings, entries = set(), []

    def proposal(attempt):
        card, epoch = attempt % 20, attempt // 20
        amodel, bmodel = divmod(card, 5)
        record = (roots[amodel] if amodel < 2 else
                  spanning[epoch % len(spanning)] if amodel == 2 else
                  bank[epoch % len(bank)])
        H = sylows[epoch % 3]

        if bmodel == 0:
            omitted = H[(epoch // 3) % len(H)]
            B = incidence((), (), tuple(g for g in H if g != omitted))
        elif bmodel == 1:
            gaps = tuple(sorted({
                min(g, G.inv(g)) for g in H
                if g != e and G.mul(g, g) != e
            }))
            pool = [g for g in H if g[0] % 2 == (epoch // 3) % 2]
            B = incidence(gaps, gaps, local.sample(pool, 3))
        elif bmodel == 2:
            pool = (closure(record[2]) if epoch % 2 == 0
                    else tuple(G.elements))
            B = incidence((), (), local.sample(pool, 7))
        elif bmodel == 3:
            internal = 1 + epoch % 3
            top = local.sample(odd, internal)
            bottom = local.sample(odd, internal)
            pool = [g for g in G.elements if g[0] % 2 == epoch % 2]
            B = incidence(top, bottom,
                          local.sample(pool, 7 - 2*internal))
        else:
            # One singleton/cubic redistribution, not an incidence orbit.
            B = [[[local.choice(G.elements)], [e]],
                 [[], local.sample(list(G.elements), 2)]]
            for g in local.sample(list(G.elements), 5):
                B[0].append([e])
                B[1].append([g])
        return record, B, ("optimized-card", card, epoch % 6)

    def prepare(record, B, card, serial):
        A = record[3]
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            return None
        fillings.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            raise ValueError("subgroup proposal violates support budget")
        assert all(len(set(c)) == len(c) <= 5
                   for grid in (A, B) for row in grid for c in row)

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 552 and 425*n <= 1000*k < 475*n):
            return None
        ar, br = _extract_components(G, hx, hz)
        qb = _kernel_profile(br, 7*q)
        if qb[0] < 4:
            return None
        priority = (-k, record[1] + qb[1], serial)
        return (priority, A, B, hx, hz, n, k, rx, rz, card,
                q - gf2_rank(ar), 2*q - gf2_rank(br))

    def retain(candidate):
        if candidate is None:
            return
        (_, A, B, hx, hz, n, k, rx, rz, card, ad, bd) = candidate
        key = graph_key(hx, hz, n)
        if key in graph_keys:
            return
        graph_keys.add(key)
        # Both sectors are included. These are exact absence tests,
        # not unresolved kernel-probe readings or GPU estimates.
        if not (_no_kernel_through_three(hx, n)
                and _no_kernel_through_three(hz, n)):
            return
        if not css_ok(hx, hz) or lifted_k(G, A, B) != k:
            raise ValueError("verified subgroup construction disagrees")
        assert max(r.bit_count() for r in hx + hz) <= cap
        screen = (
            ("k", k), ("card", card),
            ("component_rank_deficits", (ad, bd)),
            ("LP_rank_deficits", (len(hx)-rx, len(hz)-rz)),
            ("complete_kernel_weights_1_to_3_excluded", True),
        )
        entries.append((k, freeze(A), freeze(B), screen))

    candidates = []
    for serial, (record, B, card) in enumerate(prototypes):
        candidate = prepare(record, B, card, serial)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda r: r[0])
    for candidate in candidates:
        retain(candidate)

    # Rank-sort optimized batches; never assign an invariant to a slot
    # before deciding whether it is one of the strongest available codes.
    target = max(count + 20, 40)
    for batch in range(96):
        candidates = []
        for trial in range(40):
            attempt = 40*batch + trial
            record, B, card = proposal(attempt)
            candidate = prepare(record, B, card, len(prototypes) + attempt)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda r: r[0])
        for candidate in candidates[:12]:
            retain(candidate)
        if len(entries) >= target and batch >= 15:
            break

    entries.sort(key=lambda r: (-r[0], repr(r[3])))
    return tuple((A, B, screen) for _, A, B, screen in entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a globally optimized, invariant-separated design."""
    portfolio = _subgroup_rank_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("fresh subgroup rank portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def _unused_owned_reference(rng, G, idx):
    """Uncalled historical ownership policy; never publishes candidates."""
    import hashlib

    local = random.Random(BASE_SEED ^ 0xC1253045 ^ (104729 * idx))
    q = G.q
    sa = [[2, 2, 2]]
    card = idx % 20
    amodel, bmodel = divmod(card, 5)

    # Each column pair is (top count, bottom count). Every profile has
    # seven columns, both row sums seven, and column sums at most three.
    profiles = (
        ((1, 1),) * 7,
        ((2, 0), (0, 2)) + ((1, 1),) * 5,
        ((2, 0),) * 2 + ((0, 2),) * 2 + ((1, 1),) * 3,
        ((2, 0),) * 3 + ((0, 2),) * 3 + ((1, 1),),
        ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
        ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
        ((1, 0), (1, 2)) + ((1, 1),) * 5,
        ((1, 0), (0, 1), (2, 1), (1, 2)) + ((1, 1),) * 3,
    )
    assert all(len(p) == 7 and
               all(sum(c[i] for c in p) == 7 for i in range(2))
               for p in profiles)
    assert min(9, W_MAX) >= 9

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def logical_floor(checks, stabilizers, width):
        """Exact existence screen through four, modulo stabilizers.

        Equal syndromes of supports of size <=2 cover every support
        of size <=4. One representative per syndrome suffices for
        existence: differences that reduce to zero are stabilizers.
        Five means exclusion through four, not a distance estimate.
        The collision burden is a diagnostic, not an operator count.
        """
        span = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p not in span:
                    span[p] = row
                    break
                row ^= span[p]

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = span.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, width)
        reps = {0: 0}
        floor, burden = 5, 0
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = reps.get(syndrome)
            if old is None:
                reps[syndrome] = word
            elif nontrivial(old ^ word):
                return (old ^ word).bit_count(), 0

        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                old = reps.get(syndrome)
                if old is None:
                    reps[syndrome] = word
                    continue
                difference = old ^ word
                if not difference:
                    continue
                weight = difference.bit_count()
                if weight <= 3:
                    if nontrivial(difference):
                        return weight, burden
                elif nontrivial(difference):
                    floor = 4
                    burden += 1
        return floor, burden

    def fingerprint(hx, hz, width):
        """Intrinsic invariant, pooled without X/Z or block labels."""
        rows = hx + hz
        cols = columns(rows, width)
        degrees = [c.bit_count() for c in cols]
        qubit_profiles = []
        for col in cols:
            incident = []
            while col:
                bit = col & -col
                incident.append(rows[bit.bit_length() - 1].bit_count())
                col ^= bit
            qubit_profiles.append(tuple(sorted(incident)))
        return (tuple(sorted(degrees)),
                tuple(sorted(qubit_profiles)),
                _tanner_signature(hx, hz))

    def closure(generators):
        reached, todo = {G.e}, [G.e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        for first in todo:
            for second in steps:
                product = G.mul(first, second)
                if product not in reached:
                    reached.add(product)
                    todo.append(product)
        return tuple(sorted(reached))

    inverse_classes = tuple(sorted({
        min(g, G.inv(g)) for g in G.elements if g != G.e
    }))
    odd_classes = tuple(g for g in inverse_classes if g[0] % 2)
    rotations = tuple(g for g in G.elements if g[1] == 0)
    dic3 = tuple(g for g in G.elements if (g[0] + g[1]) % 2 == 0)

    def draw_A(model):
        if model == 0:
            generators = ((1, 0), (3, 0), (5, 0))
        elif model == 1:
            generators = ((1, 1), (3, 1), (5, 1))
        elif model == 2:
            # Mixed odd classes normally generate the entire group.
            while True:
                generators = tuple(local.sample(odd_classes, 3))
                if len(closure(generators)) == q:
                    break
        else:
            # Rank-rich smaller closures are allowed as hypotheses.
            # Their short words must pass the complete logical test.
            generators = tuple(local.sample(inverse_classes, 3))
        return [[[G.e, g] for g in generators]], closure(generators)

    def draw_B(profile, pool, policy):
        B = [[[] for _ in range(7)] for _ in range(2)]
        nonidentity = [g for g in pool if g != G.e]
        gaps = [g for g in nonidentity if G.mul(g, g) != G.e]
        if not gaps:
            gaps = nonidentity

        cross = [j for j, c in enumerate(profile) if c == (1, 1)]
        voltages = local.sample(pool, len(cross))
        for j, voltage in zip(cross, voltages):
            B[0][j], B[1][j] = [G.e], [voltage]

        used = [set(), set()]
        for j, counts in enumerate(profile):
            if j in cross:
                continue
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                if count == 1:
                    B[i][j] = [local.choice(pool)]
                elif count == 2:
                    options = [g for g in gaps
                               if min(g, G.inv(g)) not in used[i]]
                    gap = local.choice(options or gaps)
                    used[i].add(min(gap, G.inv(gap)))
                    first = G.e if policy % 2 == 0 else local.choice(pool)
                    B[i][j] = [first, G.mul(first, gap)]
                else:
                    # Complete trinomial support, not a binomial gauge.
                    B[i][j] = local.sample(pool, count)
        return B

    # Only exclusions: never emit an anchor or its invariant image.
    root_A = [[[G.e, (d, 0)] for d in (1, 3, 5)]]
    references = (
        ((), (), (0, 1, 2, 3, 6, 7, 9)),
        ((1,), (5,), (0, 2, 4, 6, 8)),
        ((1, 3), (3, 5), (0, 2, 6)),
        ((1, 3, 5), (1, 3, 5), (0,)),
        ((1, 3), (2, 3), (0, 4, 8)),
    )
    archive_keys = set()
    for top, bottom, voltages in references:
        B = [[], []]
        for d in top:
            B[0].append([G.e, (d, 0)])
            B[1].append([])
        for d in bottom:
            B[0].append([])
            B[1].append([G.e, (d, 0)])
        for d in voltages:
            B[0].append([G.e])
            B[1].append([(d, 0)])
        hx, hz, n = lifted_matrices(G, root_A, B)
        archive_keys.add(fingerprint(hx, hz, n))

    best, best_quality = None, None
    fillings, graph_keys = set(), set()
    owned_survivors = 0
    for attempt in range(1536):
        epoch = attempt // 32
        # Initial search honors the twenty prescriptions. Later epochs
        # broaden supports and A families without emitting orbit copies.
        model = amodel if attempt < 512 else (amodel + epoch) % 4
        p = bmodel if bmodel < 4 else 4 + epoch % 4
        if attempt >= 768:
            p = (p + epoch) % len(profiles)
        profile = profiles[p]
        A, subgroup = draw_A(model)

        # Compare common-subgroup, differing-subgroup and full-group
        # couplings; relative label structure, not gauges, changes.
        policies = (subgroup, rotations, dic3, tuple(G.elements))
        pool = policies[epoch % 4]
        if len(pool) < 7:
            pool = tuple(G.elements)
        B = draw_B(profile, pool, epoch)
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= min(9, W_MAX)
        assert all(len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 552 and 425 * n <= 1000 * k < 475 * n):
            continue

        key = fingerprint(hx, hz, n)
        if key in archive_keys or key in graph_keys:
            continue
        graph_keys.add(key)
        digest = hashlib.blake2b(
            repr(key).encode("ascii"), digest_size=16).digest()
        # An isomorphism preserves this invariant and therefore its
        # owner. Different slots cannot submit the same Tanner graph.
        if int.from_bytes(digest, "little") % N_SLOTS != idx:
            continue

        xf = logical_floor(hz, hx, n)
        if xf[0] < 4:
            continue
        zf = logical_floor(hx, hz, n)
        if zf[0] < 4:
            continue
        floor = min(xf[0], zf[0])
        quality = (k * floor * floor, floor, k, -(xf[1] + zf[1]))
        owned_survivors += 1
        if best_quality is None or quality > best_quality:
            ar, br = _extract_components(G, hx, hz)
            best_quality = quality
            best = {
                "A": A, "B": B, "screen_k": k,
                "design_card": (card, model, p, epoch % 4),
                "logical_exclusion_through": floor - 1,
                "logical_collision_diagnostics": (xf, zf),
                "component_diagnostics": (
                    _kernel_profile(ar, 3 * q),
                    _kernel_profile(br, 7 * q)),
                "rank_deficits": (len(hx) - rx, len(hz) - rz),
                "invariant_owner": idx,
            }
        if floor == 5 and k >= 258:
            break
        if owned_survivors >= 12 and attempt >= 768:
            break

    if best is None:
        raise ValueError("no novel owned filling passed logical-quotient screens")
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz), "verified lift failed CSS"
    assert lifted_k(G, best["A"], best["B"]) == best["screen_k"]
    assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell from the immutable subgroup rank portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty slots retained only five downstream classes, exhausted after slot 51, and all five measured codes still had d30s=4.
# REFUTED: Invariant ownership and mixed-degree logical-quotient cards improve this bed's five-code score beyond 7.4780: the measured program scored 7.4200, retained 21 downstream classes from sixty slots, voided three slots, and all five finalists still had d30s=4.
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
