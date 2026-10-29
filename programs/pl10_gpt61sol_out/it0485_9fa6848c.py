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
CONCEPT = ("Generalized quaternion Q32, order 32, shapes (1,2)x(2,6): long-ratio unit trinomials, invariant-reserved component geometries and independent noisy-syndrome holdouts should improve worse-basis circuit fitness above the parent's 6.0611 in cell 0.300.")
GROUP = ("Jump from the order-42 (C7:C3)xC2 parent to Q32=<x,y | x^16=1,y^2=x^8,y*x*y^-1=x^-1>. The different orders establish a genuine group change. Over F2, odd-augmentation elements of a finite 2-group algebra are units: every distinct-term trinomial is therefore invertible. BEDS contains no order-32 (1,2)x(2,6) evaluation. Related high-rate order-32 readings around 3.5 do not establish an advantage on this lower-rate bed.")
SHAPES = ("Keep (1,2)x(2,6) for target cell 0.300. At q=32, n=448 and kb=128 give guaranteed rate 2/7; require exact helper-built k=128. Components have 64 variables/32 checks and 192 variables/64 checks. Parent slot37 had 100 Z failures/53000 shots versus 91 X failures/200000, with mild enrichment across Acol0-1/Bcol0-1 and widespread logical flips. Other slots implicated different Acol/Bcol blocks, so no persistent isolated Brow hotspot is assumed.")
FILLING = ("Twenty contracts cross five A triangle-order pairs with four B count architectures; additional slots reserve unused rooted uncolored A-component invariants rather than draw orbit copies. A uses unit noncommuting trinomials with edge spectra (4,4,16), (4,4,8) or (4,4,4). B rows total nine and columns three, using predominantly order-16 doubletons and unit trinomials. HX/HZ weights remain 9/12; extraction uses19584 CNOTs versus25704 solely because q is smaller. All actions, ranks and short-logical classification use verified helpers. Exact component warnings and capped sampled witnesses shortlist fresh B voltages; common-stream two-interval noisy-syndrome BP training and independent holdouts then protect the worse basis. These probes include approximate suffix faults but omit ASC ordering and relay/LSD, and are not circuit pL. Target the parent's collective failures and late settlement, not its distance proxy or rank-excess promotion.")
FAMILY_NAME = "r030_Q32_12x26_unit_longratio_temporalholdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("generalized quaternion Q32 requires q=32")
    # (a,b)=x^a*y^b.
    # x^16=1, y^2=x^8, y*x*y^-1=x^-1.
    # The carry x^8 is central and fixed by inversion.
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        return (int(g[0]) % 16, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 8*b*d) % 16,
                (b+d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 16, 0) if b == 0 else ((a+8) % 16, 1)

    def idx(g):
        a, b = red(g)
        return 16*b+a

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


def _retired_order42_ctx(rng, G, idx):
    """Inactive parent builder retained for provenance; never called."""
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


def _q32_fingerprint(rows, width):
    """Rooted uncolored component invariant, not full-code canonization.

    A single-row regular component has a transitive check orbit.
    Equal invariants are conservatively collapsed; complete LP Tanner
    equivalence remains the evaluator's authoritative test.
    """
    import hashlib
    from collections import Counter

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, syndrome in enumerate(_r030_columns(rows, width)):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            syndrome ^= bit
    initial = [(int(v >= nr), len(ns), int(v == 0))
               for v, ns in enumerate(adjacency)]
    palette = {key: t for t, key in enumerate(sorted(set(initial)))}
    colors = [palette[key] for key in initial]
    digest = hashlib.sha256()
    for _ in range(14):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in ns)))
            for v, ns in enumerate(adjacency)]
        digest.update(repr(sorted(
            Counter(signatures).items())).encode("ascii"))
        palette = {key: t for t, key in enumerate(sorted(set(signatures)))}
        colors = [palette[key] for key in signatures]
    return digest.hexdigest()


def _q32_sample(rows, width, q, seed):
    """Sample genuine kernel words; minima are upper bounds only."""
    source = random.Random(seed)
    cols = _r030_columns(rows, width)
    permutation = list(range(width))
    minimum = width + 1
    short = set()

    def inspect(word):
        nonlocal minimum
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 10 and len(short) < 8192:
                short.add(word)

    for _ in range(16):
        source.shuffle(permutation)
        pivots, basis = {}, []
        for c in permutation:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                inspect(word)

        for sweep in range(2):
            source.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposed = word ^ other
                    inspect(proposed)
                    if proposed and proposed.bit_count() < weight:
                        word, weight = proposed, proposed.bit_count()
                basis[i] = word
                inspect(word)
        if minimum <= 4:
            break

    mask = (1 << q) - 1
    vulnerable = [word for word in short if word.bit_count() <= 8]
    loads = [sum(bool((word >> (j*q)) & mask) for word in vulnerable)
             for j in range(width // q)]
    return minimum, len(vulnerable), max(loads, default=0)


def _q32_temporal_probe(checks, stabilizers, n, seed, shots):
    """Two noisy-syndrome intervals, NOT the ASC memory experiment.

    H*e1+m=s1 and H*(e1+e2)=s2, with a perfect closing syndrome.
    Degree exposure and random opposite-check suffixes are approximate.
    Actual ASC ordering, six separate rounds, relay BP and LSD are
    omitted. Settled residuals are classified using actual stabilizers.
    """
    import numpy as np

    def sites(row):
        result = []
        while row:
            bit = row & -row
            result.append(bit.bit_length() - 1)
            row ^= bit
        return result

    physical = [sites(row) for row in checks]
    opposite = [sites(row) for row in stabilizers]
    m = len(checks)
    width = 2*n + m
    extended = [
        row | (1 << (2*n+r)) for r, row in enumerate(checks)
    ] + [
        row | (row << n) for row in checks
    ]
    neighborhoods = [sites(row) for row in extended]
    incident = [[] for _ in range(width)]
    edge_variables = []
    for row in neighborhoods:
        for c in row:
            incident[c].append(len(edge_variables))
            edge_variables.append(c)
    ne = len(edge_variables)
    maximum = max(map(len, neighborhoods))
    ce = np.full((2*m, maximum), ne, dtype=np.int32)
    cv = np.full((2*m, maximum), width, dtype=np.int32)
    valid = np.zeros((2*m, maximum), dtype=bool)
    offset = 0
    for r, row in enumerate(neighborhoods):
        degree = len(row)
        ce[r, :degree] = np.arange(offset, offset+degree)
        cv[r, :degree] = row
        valid[r, :degree] = True
        offset += degree
    ve = np.full((width, max(map(len, incident))),
                 ne, dtype=np.int32)
    for c, edges in enumerate(incident):
        ve[c, :len(edges)] = edges
    ev = np.asarray(edge_variables, dtype=np.int32)

    sensed = np.asarray([
        value.bit_count() for value in _r030_columns(checks, n)
    ], dtype=float)
    spread = np.asarray([
        value.bit_count() for value in _r030_columns(stabilizers, n)
    ], dtype=float)
    data_mu = 3*0.002*(2+sensed+spread)*2/3
    measure_mu = np.asarray([
        3*0.002*(len(row)+2)*2/3 for row in physical
    ])
    levels = np.asarray((0.55, 0.80, 1.05))[np.arange(shots) % 3]
    data_rates = 0.5*(1-np.exp(
        -2*levels[:, None]*data_mu[None, :]))
    measure_rates = 0.5*(1-np.exp(
        -2*levels[:, None]*measure_mu[None, :]))
    rates = np.concatenate(
        (data_rates, data_rates, measure_rates), axis=1)
    channel = np.random.default_rng(seed)
    errors = np.zeros((shots, width+1), dtype=bool)
    errors[:, :width] = channel.random((shots, width)) < rates

    # These are stress opportunities, not ASC-certified hook faults.
    for row in opposite:
        order = channel.permutation(row)
        event_rates = levels*0.25*3*0.002*len(row)*2/3
        for interval in range(2):
            active = channel.random(shots) < event_rates
            for shot in np.flatnonzero(active):
                cut = int(channel.integers(1, len(order)))
                errors[shot, interval*n+order[cut:]] ^= True

    marginal_mu = data_mu + 0.25*3*0.002*spread*0.35
    marginal = np.clip(0.5*(1-np.exp(
        -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
    prior_rates = np.concatenate(
        (marginal, marginal,
         np.clip(measure_rates, 0.002, 0.25)), axis=1)
    prior = np.log((1-prior_rates)/prior_rates)
    target = np.logical_xor.reduce(
        errors[:, cv] & valid[None, :, :], axis=2)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, ev].copy()
    outgoing = np.zeros((shots, ne+1), dtype=float)
    done = np.zeros(shots, dtype=bool)
    decoded = np.zeros((shots, width), dtype=bool)
    completion = np.full(shots, 41, dtype=np.int32)
    positions = np.arange(maximum)[None, None, :]

    for iteration in range(1, 41):
        padded = np.concatenate(
            (messages, np.full((shots, 1), 1.0e6)), axis=1)
        incoming = padded[:, ce]
        magnitude = np.abs(incoming)
        smallest_at = magnitude.argmin(axis=2)
        smallest = magnitude.min(axis=2)
        second = np.partition(magnitude, 1, axis=2)[:, :, 1]
        signs = np.where(incoming < 0, -1.0, 1.0)
        product = signs.prod(axis=2)*target_sign
        excluded = np.where(
            positions == smallest_at[:, :, None],
            second[:, :, None], smallest[:, :, None])
        update = 0.8*product[:, :, None]*signs*excluded
        outgoing[:, :ne] = update[:, valid]
        posterior = prior + outgoing[:, ve].sum(axis=2)
        hard = posterior < 0
        padded_hard = np.concatenate(
            (hard, np.zeros((shots, 1), dtype=bool)), axis=1)
        syndrome = np.logical_xor.reduce(
            padded_hard[:, cv] & valid[None, :, :], axis=2)
        settled = np.all(syndrome == target, axis=1)
        newly = settled & ~done
        decoded[newly] = hard[newly]
        completion[newly] = iteration
        done |= settled
        if done.all():
            break
        proposal = posterior[:, ev] - outgoing[:, :ne]
        messages = np.clip(
            0.35*messages + 0.65*proposal, -30, 30)

    span = _row_span(stabilizers)
    wrong = residual_weight = 0
    for shot in np.flatnonzero(done):
        residual = (
            errors[shot, :n] ^ errors[shot, n:2*n]
            ^ decoded[shot, :n] ^ decoded[shot, n:2*n])
        word = sum(1 << int(c) for c in np.flatnonzero(residual))
        if not _in_row_span(word, span):
            wrong += 1
            residual_weight += int(residual.sum())
    unresolved = int((~done).sum())
    return (wrong+unresolved, unresolved, wrong,
            residual_weight, int(completion.sum()))


def NEW_CTX(rng, G, idx):
    """Pure Q32 catalogue reservation and fresh voltage/decoder selection."""
    import math

    q = G.q
    assert q == 32 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    design, batch = idx % 20, idx // 20
    requested_model, b_model = divmod(design, 4)
    mask = (1 << q)-1

    # Cache only singleton actions obtained from the verified helper.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        result = []
        for cell_row in grid:
            rows = [0]*q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    for r, word in enumerate(actions[side][G.idx(g)]):
                        rows[r] ^= word << (j*q)
            result.extend(rows)
        return result

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[G.idx(g)] = order
                break
        assert G.idx(g) in orders
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (4, 8, 16)
    }

    # A normal-subgroup edge of order t and an outside element give
    # a noncommuting triangle with edge spectrum (4,4,t).
    triangles = {}
    for wanted in (4, 8, 16):
        book = []
        for a in range(1, 16):
            u = (a, 0)
            if orders[G.idx(u)] != wanted:
                continue
            for b in range(16):
                v = (b, 1)
                assert G.mul(u, v) != G.mul(v, u)
                cell = [G.e, u, v]
                assert len(set(cell)) == 3
                book.append(cell)
        triangles[wanted] = book
    assert all(triangles.values())

    a_cards = ((16, 16), (16, 8), (16, 4), (8, 8), (8, 4))
    source = random.Random(320448731)
    raw = [[] for _ in a_cards]
    seen_labels = set()
    proposal_count = max(12000, 160*N_SLOTS)

    # Rebuilt identically in every pure call. Independent component
    # gauges and column swaps are removed from the proposal labels.
    for serial in range(proposal_count):
        model = serial % len(a_cards)
        first, second = a_cards[model]
        A = [[list(source.choice(triangles[first])),
              list(source.choice(triangles[second]))]]
        label = tuple(sorted(
            tuple(sorted(G.idx(g) for g in cell)) for cell in A[0]))
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ar = component(A, "A")
        if not _connected_checks(ar, [], 2*q):
            continue
        coarse = _r030_cycles(ar, 2*q)
        raw[model].append((coarse, label, serial, A, ar))

    banks = []
    capacity = max(24, N_SLOTS//5+10)
    for model, proposals in enumerate(raw):
        entries, identities = [], set()
        for coarse, label, serial, A, ar in sorted(
                proposals, key=lambda item: item[:3]):
            cols = _r030_columns(ar, 2*q)
            if _r030_small_kernel(cols, 4):
                continue
            identity = _q32_fingerprint(ar, 2*q)
            if identity in identities:
                continue
            identities.add(identity)
            bad6 = int(_r030_small_kernel(cols, 6))
            # Exact short-word warning first; cycles are only heuristics.
            quality = (bad6, coarse[0], coarse[1], identity, serial)
            entries.append((quality, identity, model, A, ar))
            if len(entries) >= capacity:
                break
        entries.sort(key=lambda item: item[0])
        banks.append(entries)

    # Reserve a different invariant for every supplied slot. A card
    # can borrow another unused class rather than emit an orbit copy
    # if its finite preferred catalogue is exhausted.
    reserved = set()
    chosen = None
    for slot in range(idx+1):
        preferred = (slot % 20)//4
        available = [item for item in banks[preferred]
                     if item[1] not in reserved]
        if not available:
            available = [item for bank in banks for item in bank
                         if item[1] not in reserved]
        if not available:
            raise ValueError("unused Q32 A-component classes exhausted")
        chosen = min(available, key=lambda item: item[0])
        reserved.add(chosen[1])

    a_quality, a_identity, actual_model, A, ar = chosen
    A = clone(A)
    assert gf2_rank(ar) == q
    da, na, pa = _q32_sample(
        ar, 2*q, q, 3200101+1009*idx)

    b_first = (
        (1, 1, 1, 2, 2, 2),
        (0, 1, 2, 2, 2, 2),
        (0, 1, 1, 2, 2, 3),
        (0, 0, 2, 2, 2, 3),
    )[b_model]
    sa = [[3, 3]]
    sb = [list(b_first), [3-t for t in b_first]]
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert tuple(map(sum, sb)) == (9, 9)
    assert all(sum(column) == 3 for column in zip(*sb))

    def draw_cell(i, j):
        count = sb[i][j]
        if count == 0:
            return []
        anchor = G.pick(rng)
        if count == 1:
            return [anchor]
        if count == 2:
            # Extended cards change actual internal orders, not gauges.
            desired = 16
            if batch % 3 == 1 and (j+2*i+design) % 4 == 0:
                desired = 8
            elif batch % 3 == 2 and (j+2*i+design) % 5 == 0:
                desired = 4
            delta = rng.choice(palettes[desired])
            return [anchor, G.mul(anchor, delta)]
        wanted = (16, 8, 16)[(i+j+actual_model+batch) % 3]
        return [G.mul(anchor, g)
                for g in rng.choice(triangles[wanted])]

    def draw_B():
        return [[draw_cell(i, j) for j in range(6)]
                for i in range(2)]

    positions = [(i, j) for i in range(2) for j in range(6)
                 if sb[i][j]]
    pool, labels = [], set()
    serial = 0

    def assess(B):
        br = component(B, "B")
        four, six = _r030_cycles(br, 6*q)
        disconnected = int(not _connected_checks(br, [], 6*q))
        deficit = 2*q-gf2_rank(br)
        energy = (1000*deficit + 100*disconnected
                  + (80*four+six/8)/q)
        return energy, deficit, disconnected, four, six, br

    def retain(B, result):
        nonlocal serial
        _, deficit, disconnected, four, six, br = result
        if deficit or disconnected:
            return
        label = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in B for cell in row)
        if label in labels:
            return
        labels.add(label)
        serial += 1
        pool.append(((four, six), serial, clone(B), list(br)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    for restart in range(4):
        B = draw_B()
        current = assess(B)
        retain(B, current)
        for step in range(180):
            i, j = rng.choice(positions)
            old = B[i][j]
            B[i][j] = draw_cell(i, j)
            proposed = assess(B)
            delta = proposed[0]-current[0]
            temperature = 12*(0.98**step)+0.08
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if delta < 0 or step % 12 == 0:
                    retain(B, current)
            else:
                B[i][j] = old
        retain(B, current)

    prepared, fallback = [], None
    for coarse, number, B, expected_b in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 448 or k != 128:
            continue
        if not _connected_checks(hx, hz, n):
            continue
        actual_a = [
            sum(((hx[r] >> (j*6*q)) & mask) << (j*q)
                for j in range(2))
            for r in range(q)]
        actual_b = [
            row & ((1 << (6*q))-1) for row in hz[:2*q]]
        assert actual_a == ar and actual_b == expected_b
        assert css_ok(hx, hz)
        item = (coarse, number, B, hx, hz, k)
        if fallback is None:
            fallback = item
        if _has_short_logical(hx, hz, n):
            continue
        bc = _r030_columns(expected_b, 6*q)
        bad4 = int(_r030_small_kernel(bc, 4))
        db, nb, pb = _q32_sample(
            expected_b, 6*q, q, 3200201+1009*idx)
        boundary_peak, boundary_mean = _r030_exposure(bc, q)
        key = (
            bad4, a_quality[0], max(0, 10-min(da, db)),
            max(na/(2*q), nb/(6*q)),
            na/(2*q)+nb/(6*q),
            max(pa/(2*q), pb/(6*q)),
            boundary_peak, boundary_mean,
            coarse, number)
        prepared.append((key, B, hx, hz, k, db))
    prepared.sort(key=lambda item: item[0])

    def quality(x, z):
        return tuple(value for t in range(len(x))
                     for value in (max(x[t], z[t]), x[t]+z[t]))

    seed = 320448031+1009*idx
    training = []
    for item in prepared[:3]:
        key, B, hx, hz, k, db = item
        x = _q32_temporal_probe(hz, hx, 448, seed, 48)
        z = _q32_temporal_probe(hx, hz, 448, seed+1, 48)
        ranked = key[:3]+quality(x, z)+key[3:]
        training.append((ranked, item, x, z))
    training.sort(key=lambda item: item[0])

    winner = None
    for _, item, tx, tz in training[:2]:
        key, B, hx, hz, k, db = item
        x = _q32_temporal_probe(
            hz, hx, 448, seed+1000003, 96)
        z = _q32_temporal_probe(
            hx, hz, 448, seed+1000004, 96)
        # Independent holdout leads the final decoder comparison.
        ranked = (key[:3]+quality(x, z)
                  +quality(tx, tz)+key[3:])
        if winner is None or ranked < winner[0]:
            winner = ranked, B, hx, hz, k, db, x, z, tx, tz

    safe = winner is not None
    if winner is None:
        # Preserve a legal slot if a finite protection search misses;
        # the evaluator still applies its own distance/measurement tiers.
        if fallback is None:
            for _ in range(32):
                B = draw_B()
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n == 448 and k == 128:
                    fallback = ((10**12,), 0, B, hx, hz, k)
                    break
        if fallback is None:
            raise ValueError("no legal exact-rate Q32 filling")
        coarse, number, B, hx, hz, k = fallback
        winner = ((10**12,), B, hx, hz, k, 0,
                  (), (), (), ())

    key, B, hx, hz, k, db, x, z, tx, tz = winner
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 128
    assert max(row.bit_count() for row in hx) == 9
    assert max(row.bit_count() for row in hz) == 12
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 6*sum(row.bit_count() for row in hx+hz) == 19584
    assert 0.275 <= k/448 < 0.325
    assert all(len(set(cell)) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    return {
        "cells": {"A": A, "B": B}, "k": k,
        "design": (requested_model, actual_model, b_model, batch),
        "A_class": a_identity, "screen": key,
        "component_upper_bounds": (da, db),
        "bp_holdout": (x, z), "bp_training": (tx, tz),
        "small_logical_safe": safe, "cnots": 19584,
    }


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected Q32 cell without late changes."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: Even-quotient doubletons at weight ten uniformly retain zero-failure memory: current slot25 had four Z failures in 200000 shots, flipping 47 logical qubits, despite d2s=15 and k=66; four measured siblings had no failures.
# REFUTED: Involution-free degree-three components and boundary/kernel screening uniformly suppress collective memory failures on the order-42 bed: fitness was 6.0611; slot37 had 100 Z failures/53000 shots and s=5.663, versus slot2's 51 Z and 60 X failures/200000 and s=6.437 at the same n,k,w,d2s,depth and CNOT count.
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
