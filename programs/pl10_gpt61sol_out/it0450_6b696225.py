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
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("For (C7 semidirect C3) times C2, order 42, shapes (1,2)x(2,6), a unit A trinomial plus a noncommuting quartet, eight-term B rows and noisy-syndrome holdout selection should reduce collective worse-basis circuit failures beyond the current 6.0611. Measured ASC pL, not sampled distance, decides this hypothesis.")
GROUP = ("Frozen L1: retain exactly the action-2 order-42 group, its central C2 and its encoding. This exact bed reached 6.1594; the current program scored 6.0611. These readings motivate changing filling and decoder exposure, not another group or an asserted causal group advantage.")
SHAPES = ("Frozen (1,2)x(2,6) gives components with 84 variables/42 checks and 252 variables/84 checks. Size factor fourteen permits q<=42, so n=588, kb=168 and guaranteed rate 2/7. Complete verified ranks retain 168<=k<=190 in cell0.300. Parent failures diffusely enriched Acol/Bcol data by about1.1-1.2x and Acol/Brow checks by about1.0-1.1x, establishing no persistent isolated hotspot; collective flips affected all176 logicals.")
FILLING = ("Twenty contracts cross five B support architectures with four binomial-order policies; a deterministic catalogue additionally reserves rooted-invariant-distinct A components for every supplied slot. A counts are(3,4), with a verified unit trinomial, and B row totals are(8,8), with column degrees two/three or two/three/four. HX<=11 and HZ=12; six-round CNOTs are26208,504 more than the parent's25704, so this is a protection transfer, not a gate-count reduction. Helper-derived actions optimize B rank, connectivity and cycles. Exact short-kernel warnings, capped sampled protection and full quantum exclusion through four precede six-round noisy-syndrome BP probes and independent holdouts. The probes include approximate extraction exposure and opposite-check suffix faults but omit ASC ordering and relay/LSD; they are not circuit fitness. Target parent slot37's100/53000 Z failures versus slot2's51/200000, despite equal k176,d2s10,weight12 and gate count; do not deliberately chase rank-deficit ceiling bonuses.")
FAMILY_NAME = "r030_frob21_times_C2_12x26_unitquartet_temporalholdout_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("(C7 semidirect C3) times C2 requires q=42")
    # x^7=y^6=1, y*x*y^-1=x^2.
    # The action has order three; y^3 is central of order two.
    act = (1, 2, 4, 1, 2, 4)
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + act[b] * c) % 7, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-act[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _r030_columns(rows, width):
    """Transpose only rows supplied by the verified construction."""
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= marker
            row ^= bit
    return cols


def _r030_small_kernel(cols, depth=4):
    """Exact nonzero component-kernel existence through depth four/six."""
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return True
        seen.add(syndrome)
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            if value in seen:
                return True
            seen.add(value)
    if depth == 6:
        for i, syndrome in enumerate(cols):
            for j in range(i):
                pair = syndrome ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True
                    seen.add(value)
    return False


def _r030_cycles(rows, width):
    """Exact C4; corrected triangle proxy, exact C6 when C4 is absent."""
    adjacency = [0] * len(rows)
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap * (overlap - 1) // 2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    triangles = 0
    for i in range(len(rows)):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6
                for d in (c.bit_count()
                          for c in _r030_columns(rows, width)))
    return four, max(0, triangles - stars)


def _r030_exposure(cols, q):
    """Lift-zero anchored triple boundaries: a heuristic, not a certificate."""
    loads = []
    width = len(cols)
    for anchor in range(0, width, q):
        total = 0
        for i in range(width):
            if i == anchor:
                continue
            pair = cols[anchor] ^ cols[i]
            for j in range(i):
                if j == anchor:
                    continue
                weight = (pair ^ cols[j]).bit_count()
                if weight <= 5:
                    total += (6 - weight) ** 2
        loads.append(total)
    samples = max(1, (width - 1) * (width - 2) // 2)
    return max(loads) / samples, sum(loads) / (len(loads) * samples)


def _r030_probe(rng, cols, q, trials=24):
    """Verified kernel witnesses; sampled minimum is only an upper bound."""
    width = len(cols)
    order = list(range(width))
    minimum = width + 1
    short = set()

    def offer(word):
        nonlocal minimum
        if not word:
            return
        weight = word.bit_count()
        minimum = min(minimum, weight)
        if weight <= 8 and len(short) < 8192:
            short.add(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                offer(word)
                for other in basis:
                    offer(word ^ other)
                basis.append(word)
        if minimum <= 6:
            break

    mask = (1 << q) - 1
    loads = [sum(bool((word >> (j * q)) & mask) for word in short)
             for j in range(width // q)]
    return minimum, len(short) / width, max(loads, default=0) / width


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def _retired_degree3_CTX(rng, G, idx):
    """Inactive degree-three experiment retained for provenance."""
    import math

    q = G.q
    assert q == 42
    design, mode = idx % 20, idx // 20
    b_model, a_model = divmod(design, 5)

    # Sorted weighted-column architectures, not column-permutation copies.
    # Every column totals three and both rows total nine.
    b_first = (
        (1, 1, 1, 2, 2, 2),
        (0, 1, 2, 2, 2, 2),
        (0, 1, 1, 2, 2, 3),
        (0, 0, 2, 2, 2, 3),
    )[b_model]
    counts = {
        "A": [[3, 3]],
        "B": [list(b_first), [3 - t for t in b_first]],
    }
    assert _grid_weights(counts["A"], counts["B"]) == 12
    assert _no_dead_lines(counts["A"], counts["B"])
    assert all(sum(row) == 9 for row in counts["B"])
    assert all(sum(col) == 3 for col in zip(*counts["B"]))

    # A cards differ in normal-subgroup, central-coset and quotient
    # structure. They are not transforms of a saved filling.
    a_phases = (
        ((0, 1), (0, 1)),
        ((0, 1), (3, 1)),
        ((1, 2), (1, 2)),
        ((0, 1), (0, 2)),
        ((0, 1), (3, 2)),
    )[a_model]

    orders = []
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if G._key(power) == G._key(G.e):
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite-group element order")
    relative = [
        [orders[G.idx(G.mul(G.inv(g), h))] for h in G.elements]
        for g in G.elements
    ]

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if side == "A":
            b, c = a_phases[j]
            if mode % 2:
                b, c = (-b) % 6, (-c) % 6
            origin = G.pick(rng)
            rel = [G.e, (rng.randrange(1, 7), b),
                   (rng.randrange(7), c)]
            return [G.mul(origin, g) for g in rel]
        if not weight:
            return []
        if weight == 1:
            return [G.pick(rng)]
        # Extended harness cards change the binomial order palette.
        # Trinomials always have a realizable non-involution palette.
        palettes = ((3, 6, 7, 14), (6, 7, 14), (6, 14))
        palette = (palettes[mode % 3] if weight == 2
                   else (3, 6, 7, 14))
        for _ in range(64):
            chosen = [rng.randrange(q)]
            for _ in range(1, weight):
                available = [
                    h for h in range(q)
                    if all(relative[g][h] in palette for g in chosen)
                ]
                if not available:
                    break
                chosen.append(rng.choice(available))
            if len(chosen) == weight:
                return [G.elements[t] for t in chosen]
        raise ValueError("relative-order card could not be realized")

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(side, grid):
        # Zero probes preserve exactly the evaluator's lift conventions.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def portfolio(side):
        positions = [(i, j) for i, row in enumerate(counts[side])
                     for j, weight in enumerate(row) if weight]
        pool, signatures = [], set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            if len(pool) >= 8 and stats > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in signatures:
                return
            signatures.add(signature)
            serial += 1
            pool.append((stats, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(3):
            grid = draw_grid(side)
            rows, width = component(side, grid)
            stats = _r030_cycles(rows, width)
            energy = (80 * stats[0] + stats[1] / 8) / q
            remember(grid, rows, stats)
            for step in range(160):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                trial_rows, _ = component(side, grid)
                trial_stats = _r030_cycles(trial_rows, width)
                proposed = (80 * trial_stats[0] + trial_stats[1] / 8) / q
                delta = proposed - energy
                temperature = 8 * (0.98 ** step) + 0.08
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    rows, stats, energy = trial_rows, trial_stats, proposed
                    remember(grid, rows, stats)
                else:
                    grid[i][j] = old

        finalists = []
        for stats, serial, grid, rows in pool:
            width = len(grid[0]) * q
            cols = _r030_columns(rows, width)
            assert all(col.bit_count() == 3 for col in cols)
            disconnected = int(not _connected_checks(rows, [], width))
            bad4 = int(_r030_small_kernel(cols, 4))
            # The shorter A component permits an exact depth-six test.
            # For B, sampled witnesses are warnings, never certificates.
            bad6 = (int(_r030_small_kernel(cols, 6))
                    if side == "A" and not bad4 else bad4)
            distance, short, peak = _r030_probe(rng, cols, q)
            boundary_peak, boundary_mean = _r030_exposure(cols, q)
            item = {
                "grid": grid, "disconnected": disconnected,
                "bad4": bad4, "bad6": bad6,
                "gap": max(0, 8 - distance),
                "short": short, "peak": peak,
                "boundary_peak": boundary_peak,
                "boundary_mean": boundary_mean,
                "four": stats[0] / q, "six": stats[1] / q,
                "deficit": len(rows) - gf2_rank(rows),
            }
            item["key"] = (
                disconnected, bad4, bad6, item["gap"],
                short, peak, boundary_peak, boundary_mean,
                item["four"], item["six"], item["deficit"], serial,
            )
            finalists.append(item)
        finalists.sort(key=lambda item: item["key"])
        return finalists[:6]

    # Each retry keeps the same deliberate count/order contract.
    # No final cells are repaired or redrawn after acceptance.
    for batch in range(3):
        apool, bpool = portfolio("A"), portfolio("B")
        candidates = []
        for a in apool:
            for b in bpool:
                key = (
                    a["disconnected"] + b["disconnected"],
                    a["bad4"] + b["bad4"], a["bad6"],
                    max(a["gap"], b["gap"]),
                    max(a["short"], b["short"]),
                    a["short"] + b["short"],
                    max(a["peak"], b["peak"]),
                    max(a["boundary_peak"], b["boundary_peak"]),
                    a["boundary_mean"] + b["boundary_mean"],
                    max(a["four"], b["four"]),
                    a["four"] + b["four"],
                    a["six"] + b["six"],
                )
                candidates.append((key, a["grid"], b["grid"]))
        candidates.sort(key=lambda item: item[0])

        for key, A, B in candidates:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not 168 <= k <= 190:
                continue
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue
            assert n == 588 and css_ok(hx, hz)
            assert max(row.bit_count() for row in hx) == 9
            assert max(row.bit_count() for row in hz) == 12
            assert 6 * sum(row.bit_count() for row in hx + hz) == 25704
            return {"cells": {"A": A, "B": B},
                    "design": (a_model, b_model, mode),
                    "k": k, "screen": key}

    raise ValueError("No fresh count/order filling passed verified cell-0.300 screening")


def NEW_CTX(rng, G, idx):
    """Unit/quartet protection transfer with temporal-decoder holdouts."""
    import math
    import hashlib
    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 42 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    assert 0 <= idx < N_SLOTS
    mask = (1 << q) - 1

    # Every action is extracted from the verified helper. Assembly is
    # linear composition of these blocks, not a separate lift convention.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][g] = tuple(row & mask for row in hx)
        actions["B"][g] = tuple(row & mask for row in hz)

    def component(grid, side):
        result = []
        for cell_row in grid:
            rows = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    for r, word in enumerate(actions[side][g]):
                        rows[r] ^= word << (j * q)
            result.extend(rows)
        return result

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        assert g in orders

    def relative_order(a, b):
        return orders[G.mul(G.inv(a), b)]

    palettes = {
        t: [g for g in G.elements if orders[g] == t]
        for t in (6, 7, 14)
    }
    assert all(palettes.values())

    # Precompute realizable unit triangles instead of attempting a
    # potentially empty prescribed quotient spectrum.
    nonidentity = [g for g in G.elements if g != G.e]
    units = []
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        cell = [G.e, u, v]
        if any(relative_order(a, b) <= 2
               for a, b in combinations(cell, 2)):
            continue
        rows = component([[cell]], "A")
        if gf2_rank(rows) == q:
            units.append(tuple(cell))
    assert units

    def fingerprint(rows, width):
        """Rooted component invariant, not full-code canonization.

        Check roots form one regular-lift orbit on this one-row side.
        Unequal fingerprints therefore distinguish component graphs.
        Complete Tanner equivalence remains evaluator-tested.
        """
        nr = len(rows)
        cols = _r030_columns(rows, width)
        adjacency = [[] for _ in range(nr + width)]
        for c, syndrome in enumerate(cols):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                syndrome ^= bit
        initial = [(int(v >= nr), len(ns), int(v == 0))
                   for v, ns in enumerate(adjacency)]
        palette = {s: i for i, s in enumerate(sorted(set(initial)))}
        colors = [palette[s] for s in initial]
        digest = hashlib.sha256()
        for _ in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        return digest.hexdigest()

    # Rebuild a deterministic prefix in each pure call. Slot idx owns
    # the idx-th passing component class, not an offset or orbit copy.
    source = random.Random(4207342607)
    bank, labels, identities = [], set(), set()
    for serial in range(60000):
        triangle = list(source.choice(units))
        quartet = [G.e] + source.sample(nonidentity, 3)
        if any(relative_order(a, b) <= 2
               for a, b in combinations(quartet, 2)):
            continue
        A = [[triangle, quartet]]
        label = tuple(tuple(sorted(G.idx(g) for g in cell))
                      for cell in A[0])
        if label in labels:
            continue
        labels.add(label)

        # Cheap algebraic proposal filter; actual cycles are still
        # checked on the verified component action.
        differences = []
        for cell in A[0]:
            differences.extend(
                G.mul(a, G.inv(b))
                for a in cell for b in cell if a != b)
        if len(set(differences)) != len(differences):
            continue

        ar = component(A, "A")
        four, six = _r030_cycles(ar, 2 * q)
        if four or not _connected_checks(ar, [], 2 * q):
            continue
        ac = _r030_columns(ar, 2 * q)
        if _r030_small_kernel(ac, 4):
            continue
        identity = fingerprint(ar, 2 * q)
        if identity in identities:
            continue
        if _r030_small_kernel(ac, 6):
            continue
        identities.add(identity)
        assert gf2_rank(ar) == q
        bank.append((identity, serial, A, ar, six))
        if len(bank) > idx:
            break
    if len(bank) <= idx:
        raise ValueError("unit/quartet component-class catalogue exhausted")
    a_identity, a_serial, A, ar, a_six = bank[idx]
    ac = _r030_columns(ar, 2 * q)
    da, a_short, a_peak = _r030_probe(
        random.Random(4207341001 + idx), ac, q, 40)

    # Five explicit, row/column-inequivalent allocation architectures.
    # Every profile has sixteen terms and row totals eight.
    profiles = (
        ((1,1), (1,1), (1,2), (1,2), (2,1), (2,1)),
        ((1,1), (1,1), (0,3), (2,1), (2,1), (2,1)),
        ((1,1), (1,1), (0,3), (1,2), (2,1), (3,0)),
        ((1,1), (1,1), (1,1), (1,2), (2,1), (2,2)),
        ((2,0), (0,2), (1,1), (1,2), (2,1), (2,2)),
    )
    design, batch = idx % 20, idx // 20
    policy, architecture = divmod(design, 5)
    profile = profiles[architecture]
    sa = [[3, 4]]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert tuple(map(sum, sb)) == (8, 8)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert all(0 <= w <= CELL_MAX for row in sb for w in row)
    binomial_orders = ((7,), (14,), (7,14), (6,7,14))[policy]
    ordinary = [g for t in binomial_orders for g in palettes[t]]
    long = palettes[7] + palettes[14]
    matching = [j for j, column in enumerate(profile)
                if column == (1, 1)]

    def draw_cell(i, j):
        count = sb[i][j]
        if not count:
            return []
        anchor = G.pick(rng)
        if count == 1:
            return [anchor]
        if count == 2:
            palette = long if sum(profile[j]) == 2 else ordinary
            return [anchor, G.mul(anchor, rng.choice(palette))]
        if count == 3:
            return [G.mul(anchor, g) for g in rng.choice(units)]
        raise ValueError("unexpected B cell count")

    def draw_B():
        B = [[draw_cell(i, j) for j in range(6)] for i in range(2)]
        holonomies = []
        for number, j in enumerate(matching):
            if number == 0:
                h = G.e
            else:
                candidates = palettes[14] if number == 1 else long
                legal = [g for g in candidates
                         if all(relative_order(g, old) > 6
                                for old in holonomies)]
                if not legal:
                    return None
                h = rng.choice(legal)
            holonomies.append(h)
            B[0][j], B[1][j] = [G.e], [h]
        return B

    def matching_ok(B):
        hs = [B[1][j][0] for j in matching]
        return all(relative_order(a, b) > 6
                   for a, b in combinations(hs, 2))

    def assess(B):
        if B is None or not matching_ok(B):
            return None
        br = component(B, "B")
        four, six = _r030_cycles(br, 6 * q)
        deficit = 2 * q - gf2_rank(br)
        connected = _connected_checks(br, [], 6 * q)
        energy = (120 * deficit + 100 * int(not connected)
                  + (80 * four + six / 8.0) / q)
        return energy, deficit, four, six, connected, br

    def anchored_clean6(cols):
        """Exact <=6 exclusion using translation-anchored subsets.

        Any kernel support can be translated to contain a lift-zero
        variable. Split it into two subsets of size <=3, with an anchor
        in the first. Compare anchored subsets against all <=3 subsets.
        Store support words so self-comparisons are not false positives.
        This uses the regular action of verified component columns.
        """
        width = len(cols)
        anchored = {}

        def insert(syndrome, word):
            old = anchored.get(syndrome)
            if old is not None and old != word:
                return False
            anchored[syndrome] = word
            return True

        for a in range(0, width, q):
            bit_a = 1 << a
            if not insert(cols[a], bit_a):
                return False
            for b in range(width):
                if b == a:
                    continue
                bit_b = 1 << b
                pair = cols[a] ^ cols[b]
                if not insert(pair, bit_a | bit_b):
                    return False
                for c in range(b):
                    if c == a:
                        continue
                    if not insert(pair ^ cols[c],
                                  bit_a | bit_b | (1 << c)):
                        return False

        old = anchored.get(0)
        if old is not None:
            return False
        bits = [1 << c for c in range(width)]
        for a, syndrome in enumerate(cols):
            old = anchored.get(syndrome)
            if old is not None and old != bits[a]:
                return False
            for b in range(a):
                pair = syndrome ^ cols[b]
                word = bits[a] | bits[b]
                old = anchored.get(pair)
                if old is not None and old != word:
                    return False
                for c in range(b):
                    old = anchored.get(pair ^ cols[c])
                    if old is not None and old != (word | bits[c]):
                        return False
        return True

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round noisy-syndrome BP stress, NOT the ASC experiment.

        The final syndrome is perfect. Exposure rates and opposite-check
        suffix faults are approximate; ASC ordering and relay/LSD are
        omitted. Unresolved BP counts as probe failure, not measured pL.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length() - 1)
                row ^= bit
            return out

        ns = [sites(row) for row in checks]
        hook_ns = [sites(row) for row in stabilizers]
        m, rounds = len(checks), 6
        nv = rounds * (n + m)
        nd = (rounds + 1) * m
        row_width = max(map(len, ns)) + 2
        incidence = np.full((nd, row_width), nv, dtype=np.int32)
        for t in range(rounds + 1):
            for r, row in enumerate(ns):
                position = 0
                if t < rounds:
                    incidence[t*m+r, :len(row)] = [t*n+c for c in row]
                    position = len(row)
                    incidence[t*m+r, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = rounds*n+(t-1)*m+r

        flat = incidence.ravel()
        valid = flat != nv
        ne = len(flat)
        degree = np.bincount(flat[valid], minlength=nv)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (nv + 1, int(degree.max())), ne, dtype=np.int32)
        used = np.zeros(nv, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        sensed = np.asarray([
            c.bit_count() for c in _r030_columns(checks, n)])
        opposite = np.asarray([
            c.bit_count() for c in _r030_columns(stabilizers, n)])
        levels = np.asarray((0.55, 0.80, 1.05))[np.arange(shots) % 3]
        data_mu = 0.002 * (2 + 2*(sensed + opposite)) * 2/3
        measure_mu = np.asarray([
            0.002 * (2 + len(row)) * 2/3 for row in ns])
        data_p = 0.5 * (1 - np.exp(
            -2 * levels[:, None] * data_mu[None, :]))
        measure_p = 0.5 * (1 - np.exp(
            -2 * levels[:, None] * measure_mu[None, :]))

        channel = np.random.default_rng(seed)
        error = np.zeros((shots, nv + 1), dtype=bool)
        for t in range(rounds):
            error[:, t*n:(t+1)*n] = (
                channel.random((shots, n)) < data_p)
            start = rounds*n+t*m
            error[:, start:start+m] = (
                channel.random((shots, m)) < measure_p)
        for row in hook_ns:
            ordering = channel.permutation(row)
            event_p = levels * 0.002 * len(row) * 2/3
            for t in range(rounds):
                active = channel.random(shots) < event_p
                for shot in np.flatnonzero(active):
                    cut = int(channel.integers(1, len(ordering)))
                    error[shot, t*n+ordering[cut:]] ^= True

        marginal_mu = data_mu + 0.002 * opposite * 0.35
        marginal_p = np.clip(0.5 * (1 - np.exp(
            -2 * levels[:, None] * marginal_mu[None, :])), 0.001, 0.25)
        probabilities = np.concatenate(
            ([marginal_p] * rounds
             + [np.clip(measure_p, 0.001, 0.25)] * rounds), axis=1)
        prior = np.full((shots, nv + 1), 30.0)
        prior[:, :nv] = np.log((1 - probabilities) / probabilities)
        target = (
            error[:, flat].reshape(shots, nd, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)
        span = _row_span(stabilizers)
        wrong = late = work = 0

        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, nd, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1 - 1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True) - logs), 1 - 1e-8)
            outgoing = (
                (np.log1p(product) - np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, ne)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, ne + 1))
            padded[:, :ne] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, nv] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, nd, row_width).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                residual = final[shot] ^ true_final[shot]
                word = sum(1 << int(c) for c in np.flatnonzero(residual))
                wrong += int(not _in_row_span(word, span))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, late, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping * messages
                + (1 - damping) * (posterior[:, flat] - outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (wrong + unresolved, unresolved,
                late + unresolved, work + 33*unresolved)

    def quality(x, z):
        return tuple(v for a, b in zip(x, z) for v in (max(a, b), a+b))

    sites = [(i, j) for i in range(2) for j in range(6)
             if sb[i][j]]
    seed = 4207345881 + 1009*idx
    for retry in range(3):
        pool, encountered = [], set()
        serial = 0

        def retain(B, state):
            nonlocal serial
            _, deficit, four, six, connected, br = state
            if not connected:
                return
            label = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
            if label in encountered:
                return
            encountered.add(label)
            serial += 1
            pool.append(((deficit, four, six), serial,
                         clone(B), list(br)))
            pool.sort(key=lambda entry: entry[:2])
            del pool[8:]

        for restart in range(4):
            for attempt in range(32):
                B = draw_B()
                current = assess(B)
                if current is not None:
                    break
            else:
                continue
            retain(B, current)
            for step in range(220):
                fresh = draw_B()
                if fresh is None:
                    continue
                j = rng.randrange(6)
                old = (B[0][j], B[1][j])
                B[0][j], B[1][j] = fresh[0][j], fresh[1][j]
                proposed = assess(B)
                if proposed is None:
                    B[0][j], B[1][j] = old
                    continue
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.981 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if delta < 0 or step % 16 == 0:
                        retain(B, current)
                else:
                    B[0][j], B[1][j] = old
            retain(B, current)

        prepared = []
        for coarse, number, B, br in pool:
            bc = _r030_columns(br, 6 * q)
            if _r030_small_kernel(bc, 4):
                continue
            safe6 = anchored_clean6(bc)
            db, b_short, b_peak = _r030_probe(
                random.Random(4207342001 + idx), bc, q, 32)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 588 or not 168 <= k <= 190:
                continue
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue

            # Confirm cached component assembly against the final lift.
            actual_a = [
                sum(((hx[r] >> (j*6*q)) & mask) << (j*q)
                    for j in range(2)) for r in range(q)]
            actual_b = [row & ((1 << (6*q)) - 1)
                        for row in hz[:2*q]]
            assert actual_a == ar and actual_b == br
            protection = (
                int(not safe6), max(0, 8 - min(da, db)),
                coarse[1], max(a_short, b_short),
                a_short + b_short, max(a_peak, b_peak),
                coarse[2], coarse[0], number,
            )
            prepared.append((protection, B, hx, hz, k, db, safe6))
        prepared.sort(key=lambda entry: entry[0])

        training = []
        for number, entry in enumerate(prepared[:4]):
            protection, B, hx, hz, k, db, safe6 = entry
            x = temporal_probe(hz, hx, 588, seed, 24)
            z = temporal_probe(hx, hz, 588, seed + 1, 24)
            key = protection[:2] + quality(x, z) + protection[2:]
            training.append((key, number, entry, x, z))
        training.sort(key=lambda entry: entry[:2])

        finalists = []
        for _, number, entry, tx, tz in training[:2]:
            protection, B, hx, hz, k, db, safe6 = entry
            x = temporal_probe(hz, hx, 588, seed + 1000003, 48)
            z = temporal_probe(hx, hz, 588, seed + 1000004, 48)
            # Independent holdout leads; training breaks ties.
            key = (protection[:2] + quality(x, z)
                   + quality(tx, tz) + protection[2:] + (number,))
            finalists.append((key, entry, x, z, tx, tz))
        if not finalists:
            continue
        finalists.sort(key=lambda entry: entry[0])
        key, entry, x, z, tx, tz = finalists[0]
        _, B, hx, hz, k, db, safe6 = entry
        assert css_ok(hx, hz)
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx) <= 11
        assert max(row.bit_count() for row in hz) == 12
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert 6 * sum(row.bit_count() for row in hx + hz) == 26208
        assert all(len(cell) == len(set(cell)) <= CELL_MAX
                   for grid in (A, B) for row in grid for cell in row)
        return {
            "cells": {"A": A, "B": B}, "idx": idx, "k": k,
            "contract": (architecture, policy, batch),
            "A_class": (a_identity, a_serial),
            "component_upper_bounds": (da, db),
            "B_clean6": safe6, "A_six_cycles": a_six,
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z),
            "screen": key, "cnots": 26208,
        }

    raise ValueError("no unit/quartet target-cell finalist passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete holdout-selected cell without late changes."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: Even-quotient doubletons at weight ten uniformly retain zero-failure memory: current slot25 had four Z failures in 200000 shots, flipping 47 logical qubits, despite d2s=15 and k=66; four measured siblings had no failures.
# REFUTED: Equal k=176,d2s=10,weight twelve and25704 CNOTs determine reliability: current slots2/37 had worse-basis s=6.437/5.663 and Z failures51/200000 versus100/53000.
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
