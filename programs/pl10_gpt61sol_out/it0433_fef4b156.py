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
CONCEPT = ("For C7 semidirect C4 with inversion action, order 28, shapes (2,4)x(2,4), equal regular-doubleton components and joint decoder selection should suppress collective circuit failures relative to the unequal-component parent in target cell 0.250.")
GROUP = ("L2 retains the metacyclic semidirect-product construction rule. The new size factor twenty makes q=42 illegal, so change parameters to m=7,r=4,s=6 and q=28; 6^4=1 mod 7 validates the action. RUN STATE records about 7.1 on this group's (2,4)x(2,4) bed. This supports trying fresh geometry, not attributing reliability to the group alone.")
SHAPES = ("Equal A=(2,4),B=(2,4) components each have 112 variables and 56 checks. At q=28, n=560 and kb=112 give guaranteed rate 0.200; final evaluator-convention ranks must satisfy 0.225<=k/n<0.275. Even-cell rank deficits motivate, but do not replace, this verified rate gate. Parent failure enrichment was diffuse across Acol/Bcol blocks, usually 1.1-1.2x, and did not identify a persistent Acol/Brow defect. Equal components replace the parent's 84/42 and 252/84 asymmetry.")
FILLING = ("Every cell is a distinct-element doubleton: component row weights eight and column weights four give HX and HZ weights twelve. Twenty indexed contracts cross four A relative-order schedules with five B schedules using orders 4,7,14; further slots consume fresh canonical relative geometries, not saved-code relabelings. Minimize repeated internal directions, then optimize independent cell voltages using helper-derived actions, connectivity, ranks and exact Tanner cycles. Refine by exact component-kernel warnings through six, triple boundaries and capped sampled witnesses. Complete verified lifts enforce target rate, CSS, connectivity and prefer no quantum logical through four. Common-noise BP probes classify settled residuals modulo actual stabilizers, include possible opposite-check correlated pairs, and use independent holdouts. They omit ASC scheduling, measurement histories and relay/LSD, so remain pL surrogates. Six-round CNOTs increase from the parent's baseline 25704 to 32256; the falsifiable experiment is stronger balanced protection despite greater gate exposure. Parent slots15/17 shared n,k,w,d2s and depth but scored 5.479/6.277.")
FAMILY_NAME = "r025_C7byC4_s6_24x24_regular4_jointbp_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C4 requires q=28")
    # Same semidirect-product rule, with size-compatible parameters.
    # x^7=y^4=1, y*x*y^-1=x^6; y^2 is central.
    m, r, s = 7, 4, 6
    assert m*r == q and pow(s, r, m) == 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m)*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m)*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _fg_columns(rows, width):
    """Column syndromes of a verified component action."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _fg_cost(rows, width):
    """Connectivity, exact rank deficit, and exact Tanner 4/6 cycles."""
    from itertools import combinations

    cols = _fg_columns(rows, width)
    if not all(cols) or not all(rows):
        return None
    neighbors = [[] for _ in range(width)]
    adj = [dict() for _ in rows]
    for c, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            neighbors[c].append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, b in combinations(neighbors[c], 2):
            adj[a][b] = adj[a].get(b, 0) + 1
            adj[b][a] = adj[b].get(a, 0) + 1

    seen, todo = {0}, [0]
    while todo:
        a = todo.pop()
        for b in adj[a]:
            if b not in seen:
                seen.add(b)
                todo.append(b)
    if len(seen) != len(rows):
        return None

    four = sum(v * (v - 1) // 2
               for a in range(len(rows))
               for b, v in adj[a].items() if a < b)
    six = 0
    for a in range(len(rows)):
        for b, ab in adj[a].items():
            if b <= a:
                continue
            for c, bc in adj[b].items():
                if c > b and c in adj[a]:
                    six += ab * bc * adj[a][c]
    for ns in neighbors:
        for a, b, c in combinations(ns, 3):
            six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
    return len(rows) - gf2_rank(rows), four, six


def _fg_quality(rng, rows, width, q):
    """Exact kernel<=4 exclusion and small-boundary/kernel heuristics.

    Triple boundaries are enumerated, but triple syndromes are not stored:
    this is not an exact exclusion test through weight six.
    Randomized kernel circuits give upper bounds, never pL predictions.
    """
    cols = _fg_columns(rows, width)
    seen = {0}
    for value in cols:
        if value in seen:
            return True, 4, 1.0e9, 1.0e9
        seen.add(value)
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            if syndrome in seen:
                return True, 4, 1.0e9, 1.0e9
            seen.add(syndrome)

    exposure = 0
    loads = [0] * (width // q)
    for i in range(width):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                weight = (pair ^ cols[h]).bit_count()
                if weight < 5:
                    penalty = (5 - weight) ** 2
                    exposure += penalty
                    loads[i // q] += penalty
                    loads[j // q] += penalty
                    loads[h // q] += penalty
    triples = width * (width - 1) * (width - 2) // 6
    boundary = exposure / max(1, triples)
    peak = max(loads) * len(loads) / max(1, 3 * triples)

    order = list(range(width))
    best = width + 1
    for _ in range(28):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = syndrome, word
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome:
                best = min(best, word.bit_count())
                for previous in basis:
                    best = min(best, (word ^ previous).bit_count())
                basis.append(word)
        if best <= 6:
            break
    return False, best, boundary, peak


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


def _retired_q42_ctx(rng, G, idx):
    """Inactive historical search; the shape-aware builder below replaces it."""
    import math

    q = G.q
    design = idx % 20
    mode = idx // 20
    architecture, policy = divmod(design, 5)
    sa = [[3, 3 if mode < 2 else 4]]
    sb = [[2, 2, 2, 1, 1, 1],
          [1, 1, 1, 2, 2, 2]]
    if mode >= 1:
        sb[0][0] = 1
    if mode >= 2:
        sb[1][5] = 1

    wx = sum(sa[0]) + max(sum(row[j] for row in sb)
                          for j in range(B_SHAPE[1]))
    wz = max(map(sum, sb)) + max(sa[0])
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    elements = list(G.elements)
    allowed = (3, 6, 7, 14)

    def element_order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid element order")

    orders = {g: element_order(g) for g in elements}
    by_order = {t: [g for g in elements if orders[g] == t]
                for t in allowed}

    def relative_order(g, h):
        return orders[G.mul(G.inv(g), h)]

    # Unordered edge-order spectra distinguish the internal briefs.
    # Noncommutation is required, not inferred from the spectra.
    triangles = ((3, 3, 7), (6, 6, 7),
                 (3, 6, 14), (3, 6, 6))
    triangle_books = [[] for _ in triangles]
    usable = [g for g in elements if orders[g] in allowed]
    for a, u in enumerate(usable):
        for v in usable[a + 1:]:
            if G.mul(u, v) == G.mul(v, u):
                continue
            spectrum = tuple(sorted((orders[u], orders[v],
                                     relative_order(u, v))))
            if spectrum in triangles:
                triangle_books[triangles.index(spectrum)].append((u, v))
    assert all(triangle_books)
    a_cards = ((0, 1), (1, 2), (0, 2), (3, 3))
    a_contract = a_cards[architecture]
    b_palettes = ((14,), (7,), (6,), (3,), (14, 7, 6, 3))
    palette = b_palettes[policy]

    def draw_cell(side, i, j):
        g = G.pick(rng)
        if side == 0:
            u, v = rng.choice(triangle_books[a_contract[j]])
            normalized = [G.e, u, v]
            if sa[i][j] == 4:
                extensions = [
                    w for w in elements
                    if all(relative_order(t, w) in allowed
                           for t in normalized)
                ]
                normalized.append(rng.choice(extensions))
            return [G.mul(g, t) for t in normalized]
        if sb[i][j] == 1:
            return [g]
        order = palette[(j + 2 * i) % len(palette)]
        return [g, G.mul(g, rng.choice(by_order[order]))]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_grid(side):
        counts = sa if side == 0 else sb
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts)]

    # Cache singleton blocks extracted only from lifted_matrices.
    # No independently implemented regular-representation convention.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][g] = tuple(row & mask for row in hx)
        actions[1][g] = tuple(row & mask for row in hz)

    def component_rows(grid, side):
        result = []
        for grid_row in grid:
            rows = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        rows[r] ^= bits << (j * q)
            result.extend(rows)
        return result

    widths = (2 * q, 6 * q)
    shapes = (A_SHAPE, B_SHAPE)

    def energy(cost):
        deficit, four, six = cost
        return (12 * q * deficit + 80 * four + six) / q

    for batch in range(2):
        pools = [[], []]
        for side in range(2):
            seen_fillings = set()

            def offer(cost, serial, grid, rows):
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen_fillings:
                    return
                seen_fillings.add(signature)
                pools[side].append(
                    (cost, serial, clone(grid), list(rows)))
                pools[side].sort(key=lambda item: (item[0], item[1]))
                del pools[side][5:]

            for restart in range(4):
                for attempt in range(24):
                    grid = draw_grid(side)
                    rows = component_rows(grid, side)
                    current = _fg_cost(rows, widths[side])
                    if current is not None:
                        break
                else:
                    continue
                offer(current, restart * 121, grid, rows)

                for step in range(120):
                    nr, nc = shapes[side]
                    i, j = rng.randrange(nr), rng.randrange(nc)
                    old = grid[i][j]
                    if side == 1 and rng.random() < 0.3:
                        shift = G.pick(rng)
                        proposed_cell = [G.mul(shift, g) for g in old]
                    else:
                        # On single-row A, redraw its internal geometry;
                        # do not spend the search on whole-cell gauges.
                        proposed_cell = draw_cell(side, i, j)
                    assert len(proposed_cell) == len(old)
                    assert len(set(proposed_cell)) == len(old)
                    grid[i][j] = proposed_cell
                    proposed_rows = component_rows(grid, side)
                    proposed = _fg_cost(proposed_rows, widths[side])
                    if proposed is None:
                        grid[i][j] = old
                        continue
                    delta = energy(proposed) - energy(current)
                    temperature = 20 * (0.975 ** step) + 0.15
                    if delta > 0 and rng.random() >= math.exp(
                            -min(700.0, delta / temperature)):
                        grid[i][j] = old
                        continue
                    improved = proposed < current
                    current, rows = proposed, proposed_rows
                    if improved or step % 8 == 0:
                        offer(current, restart * 121 + step + 1,
                              grid, rows)
                offer(current, 1000 + restart, grid, rows)

        if not all(pools):
            continue

        finalists = [[], []]
        for side in range(2):
            for cost, serial, grid, rows in pools[side][:4]:
                bad, distance, boundary, peak = _fg_quality(
                    rng, rows, widths[side], q)
                finalists[side].append(
                    (cost, bad, distance, boundary, peak, serial, grid))

        options = []
        for a in finalists[0]:
            for b in finalists[1]:
                ac, abad, ad, abound, apeak, at, A = a
                bc, bbad, bd, bbound, bpeak, bt, B = b
                # Establish only a coarse sampled protection floor.
                # Above it, prefer smaller boundaries and cycle burden;
                # larger upper bounds are not treated as circuit scores.
                key = (int(abad) + int(bbad),
                       int(min(ad, bd) < 10),
                       max(apeak, bpeak), abound + bbound,
                       ac[1] + bc[1], -min(ad, bd, 12),
                       ac[2] + bc[2], ac[0] + bc[0], at, bt)
                options.append((key, A, B))
        options.sort(key=lambda item: item[0])

        fallback = None
        examined = 0
        for key, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 588 and k > 0 and
                    11 * n <= 40 * k < 13 * n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            if not _connected_checks(hx, hz, n):
                continue

            # Confirm that the cached actions are exactly the components
            # of the submitted complete matrices on both sides.
            ar = [
                sum(((hx[r] >> (j * B_SHAPE[1] * q)) & mask)
                    << (j * q) for j in range(A_SHAPE[1]))
                for r in range(q)
            ]
            br = [row & ((1 << widths[1]) - 1)
                  for row in hz[:B_SHAPE[0] * q]]
            assert ar == component_rows(A, 0)
            assert br == component_rows(B, 1)

            candidate = {"A": A, "B": B, "design": design,
                         "mode": mode, "k": k, "screen": key,
                         "contract": (a_contract, palette),
                         "small_logical_safe": False}
            if fallback is None:
                fallback = candidate
            if not _has_short_logical(hx, hz, n):
                candidate["small_logical_safe"] = True
                return candidate
            examined += 1
            if examined >= 12:
                break

        # A legal target-cell candidate is preferable to voiding a slot
        # solely because a heuristic protection preference was unmet.
        if fallback is not None:
            return fallback
    raise ValueError("no verified nonfaithful-action filling in cell 0.300")


def _shape_six_warning(cols):
    """Exact nonzero component-kernel existence through weight six."""
    seen = {0}
    for value in cols:
        if value in seen:
            return True
        seen.add(value)
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            if syndrome in seen:
                return True
            seen.add(syndrome)
    for i, value in enumerate(cols):
        for j in range(i):
            pair = value ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
    return False


def _shape_bp(checks, stabilizers, n, seed, shots):
    """Common-noise BP surrogate, NOT the scored ASC memory experiment.

    Rates approximate extraction exposure. Opposite-check pairs are
    possible correlated faults, not certified ASC hooks. Unresolved BP
    receives a penalty although relay/LSD might resolve it.
    """
    import numpy as np

    def neighborhoods(rows):
        out = []
        for row in rows:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            out.append(ns)
        return out

    ns = neighborhoods(checks)
    hooks = neighborhoods(stabilizers)
    weight = len(ns[0])
    assert all(len(row) == weight for row in ns)
    incidence = np.asarray(ns, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(ns), len(flat)
    degree = np.bincount(flat, minlength=n)
    assert np.all(degree > 0)
    variable_edges = np.full(
        (n, int(degree.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    opposite = _fg_columns(stabilizers, n)
    exposure = degree + np.asarray(
        [c.bit_count() for c in opposite], dtype=np.int32)
    levels = np.asarray((0.012, 0.020, 0.030))
    rates = np.minimum(
        levels[np.arange(shots) % 3, None]
        * exposure[None, :] / 8.0, 0.20)
    channel = np.random.default_rng(seed)
    error = channel.random((shots, n)) < rates
    for shot in range(shots):
        if channel.random() < 0.35:
            row = hooks[int(channel.integers(len(hooks)))]
            a = int(channel.integers(len(row)))
            b = int(channel.integers(len(row)-1))
            b += b >= a
            error[shot, row[a]] ^= True
            error[shot, row[b]] ^= True

    prior_rates = np.minimum(rates + 0.7/n, 0.25)
    prior = np.log((1-prior_rates)/prior_rates)
    target = (error[:, flat].reshape(shots, nr, weight)
              .sum(axis=2) & 1).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()
    span = _row_span(stabilizers)
    failed = effort = 0

    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logs = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
        products = np.minimum(
            np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
        outgoing = (
            (np.log1p(products)-np.log1p(-products))
            * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]).reshape(count, edges)
        padded = np.zeros((count, edges+1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior+padded[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        syndrome = (hard[:, flat].reshape(count, nr, weight)
                    .sum(axis=2) & 1).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local] ^ hard[local]):
                word |= 1 << int(c)
            failed += int(not _in_row_span(word, span))
            effort += iteration
        keep = ~settled
        if not np.any(keep):
            return failed, 0, effort
        messages = np.clip(
            0.30*messages+0.70*(posterior[:, flat]-outgoing),
            -26, 26)[keep]
        prior, target = prior[keep], target[keep]
        target_sign, error = target_sign[keep], error[keep]

    unresolved = len(error)
    return failed+unresolved, unresolved, effort+33*unresolved


def NEW_CTX(rng, G, idx):
    """Indexed regular-four geometries, verified lifts, joint BP holdouts."""
    import math
    from collections import Counter

    q = G.q
    assert q == 28 and A_SHAPE == B_SHAPE == (2, 4)
    sa = [[2]*4 for _ in range(2)]
    sb = [[2]*4 for _ in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    # HX: A-row eight plus B-column four.
    # HZ: B-row eight plus A-column four.
    assert max(map(sum, sa))+max(map(sum, zip(*sb))) == 12
    assert max(map(sum, sb))+max(map(sum, zip(*sa))) == 12

    els = list(G.elements)
    orders = {}
    for g in els:
        power = G.e
        for t in range(1, q+1):
            power = G.mul(power, g)
            if power == G.e:
                orders[g] = t
                break
        assert g in orders
    palettes = {t: [g for g in els if orders[g] == t]
                for t in (4, 7, 14)}
    assert all(palettes.values())

    def direction(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    capacity = {t: len({direction(g) for g in palette})
                for t, palette in palettes.items()}
    a_cards = (
        (14,14,14,14,14,14,14,14),
        (14,14,14,4,14,14,4,14),
        (14,14,4,4,14,4,14,4),
        (7,14,4,14,14,7,14,4),
    )
    b_cards = (
        (14,14,14,14,14,14,14,14),
        (14,14,14,7,14,14,7,14),
        (14,14,14,4,14,14,4,14),
        (14,14,7,4,14,7,14,4),
        (14,7,4,4,7,14,4,4),
    )

    # Actual presentation automorphisms x->x^u, y->x^v*y^w.
    # Used only to quotient relative-geometry briefs, never as a lift.
    automorphisms = []
    inverse_indices = [G.idx(G.inv(g)) for g in els]
    for u in range(1, 7):
        for v in range(7):
            for w in (1, 3):
                image = tuple(G.idx(
                    ((u*a + (v if b % 2 else 0)) % 7,
                     w*b % 4)) for a, b in els)
                assert len(set(image)) == q
                automorphisms.append(image)

    def geometry_key(grids):
        images = []
        for image in automorphisms:
            sides = []
            for grid in grids:
                pairs = []
                for j in range(4):
                    values = []
                    for i in range(2):
                        t = G.idx(grid[i][j])
                        values.append(min(image[t],
                                          image[inverse_indices[t]]))
                    pairs.append(tuple(values))
                sides.append(min(
                    tuple(sorted(pairs)),
                    tuple(sorted((b, a) for a, b in pairs))))
            images.append(tuple(sorted(sides)))
        return min(images)

    def relative_grid(grng, schedule):
        grid = []
        for i in range(2):
            requested = list(schedule[4*i:4*i+4])
            grng.shuffle(requested)
            for attempt in range(256):
                row = [grng.choice(palettes[t]) for t in requested]
                counts = Counter(requested)
                if all(len({direction(g) for g in row
                            if orders[g] == t}) == min(number, capacity[t])
                       for t, number in counts.items()):
                    grid.append(row)
                    break
            else:
                raise ValueError("minimal-repeat direction brief exhausted")
        return grid

    # Pure deterministic catalogue. Each accepted entry changes relative
    # supports before the independent voltage search. This removes obvious
    # automorphism/permutation copies, not arbitrary Tanner isomorphisms.
    grng = random.Random(280424731)
    book, seen = [], set()
    for proposal in range(40000):
        architecture, policy = divmod(len(book) % 20, 5)
        grids = [relative_grid(grng, a_cards[architecture]),
                 relative_grid(grng, b_cards[policy])]
        identity = geometry_key(grids)
        if identity in seen:
            continue
        seen.add(identity)
        book.append((identity, grids, architecture, policy))
        if len(book) > idx:
            break
    if len(book) <= idx:
        raise ValueError("fresh regular-four relative catalogue exhausted")
    contract, geometry, architecture, policy = book[idx]

    # Cache only singleton blocks returned by the verified helper.
    mask = (1 << q)-1
    actions = [{}, {}]
    for g in els:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][g] = tuple(row & mask for row in hx)
        actions[1][g] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_cell(side, i, j):
        origin = G.pick(rng)
        return [origin, G.mul(origin, geometry[side][i][j])]

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(4)]
                for i in range(2)]

    def component_rows(grid, side):
        out = []
        for grid_row in grid:
            rows = [0]*q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        rows[r] ^= bits << (j*q)
            out.extend(rows)
        return out

    def energy(cost):
        deficit, four, six = cost
        # Augmentation deficits are expected; full CSS ranks decide rate.
        return 8*deficit+(80*four+six/8)/q

    def portfolio(side):
        pool, identities = [], set()

        def offer(grid, rows, cost, serial):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in identities:
                return
            identities.add(signature)
            pool.append((cost, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(4):
            for attempt in range(48):
                grid = draw_grid(side)
                rows = component_rows(grid, side)
                cost = _fg_cost(rows, 4*q)
                if cost is not None:
                    break
            else:
                continue
            offer(grid, rows, cost, restart*1000)
            for step in range(160):
                i, j = rng.randrange(2), rng.randrange(4)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows = component_rows(grid, side)
                proposed = _fg_cost(proposed_rows, 4*q)
                if proposed is None:
                    grid[i][j] = old
                    continue
                delta = energy(proposed)-energy(cost)
                temperature = 10*(0.976**step)+0.10
                if delta > 0 and rng.random() >= math.exp(
                        -min(700.0, delta/temperature)):
                    grid[i][j] = old
                    continue
                rows, cost = proposed_rows, proposed
                if step % 16 == 15:
                    offer(grid, rows, cost, restart*1000+step+1)
            offer(grid, rows, cost, restart*1000+999)

        finalists = []
        for cost, serial, grid, rows in pool[:5]:
            cols = _fg_columns(rows, 4*q)
            assert all(c.bit_count() == 4 for c in cols)
            bad4, distance, boundary, peak = _fg_quality(
                rng, rows, 4*q, q)
            bad6 = bad4 or _shape_six_warning(cols)
            key = (int(bad4), int(bad6), max(0, 12-distance),
                   peak, boundary, cost[1], cost[2], cost[0],
                   -min(distance, 14), serial)
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        return finalists[:3]

    def verified(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 560 and 225*n <= 1000*k < 275*n):
            return None
        if not _connected_checks(hx, hz, n):
            return None
        return hx, hz, n, k

    def finish(candidate):
        A, B = candidate["A"], candidate["B"]
        hx, hz, n, k = verified(A, B)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx) == 12
        assert max(row.bit_count() for row in hz) == 12
        assert 6*sum(row.bit_count() for row in hx+hz) == 32256
        actual_a = [
            sum(((hx[(i*4)*q+r] >> (j*4*q)) & mask) << (j*q)
                for j in range(4))
            for i in range(2) for r in range(q)]
        actual_b = [row & ((1 << (4*q))-1) for row in hz[:2*q]]
        assert actual_a == component_rows(A, 0)
        assert actual_b == component_rows(B, 1)
        assert all(len(cell) == len(set(cell)) == 2
                   for grid in (A, B) for row in grid for cell in row)
        candidate.update(idx=idx, k=k, contract=contract,
                         design=(architecture, policy), cnots=32256)
        return candidate

    def probe_key(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2])

    fallback = None
    for batch in range(2):
        apool, bpool = portfolio(0), portfolio(1)
        options = []
        for ak, A in apool:
            for bk, B in bpool:
                structure = tuple(v for a, b in zip(ak[:-1], bk[:-1])
                                  for v in (max(a, b), a+b))
                options.append((structure, ak[-1], bk[-1], A, B))
        options.sort(key=lambda item: item[:3])
        coarse = []
        seed = 280424919+1009*idx
        for structure, at, bt, A, B in options:
            result = verified(A, B)
            if result is None:
                continue
            hx, hz, n, k = result
            candidate = {"A": A, "B": B, "component_screen": structure,
                         "small_logical_safe": False}
            if fallback is None:
                fallback = candidate
            if _has_short_logical(hx, hz, n):
                continue
            candidate["small_logical_safe"] = True
            x = _shape_bp(hz, hx, n, seed, 48)
            z = _shape_bp(hx, hz, n, seed+1, 48)
            key = probe_key(x, z)+(structure, at, bt)
            coarse.append((key, candidate, hx, hz, x, z))
            if len(coarse) >= 4:
                break
        if coarse:
            coarse.sort(key=lambda item: item[0])
            holdouts = []
            for _, candidate, hx, hz, cx, cz in coarse[:2]:
                x = _shape_bp(hz, hx, 560, seed+1000003, 48)
                z = _shape_bp(hx, hz, 560, seed+1000004, 48)
                total_x = tuple(a+b for a, b in zip(cx, x))
                total_z = tuple(a+b for a, b in zip(cz, z))
                key = probe_key(total_x, total_z)+(
                    candidate["component_screen"],)
                candidate["bp_surrogate"] = (total_x, total_z)
                candidate["joint_screen"] = key
                holdouts.append((key, candidate))
            holdouts.sort(key=lambda item: item[0])
            return finish(holdouts[0][1])

    # Optional protection preferences must not void an otherwise legal
    # target-band slot. Rescue preserves its complete relative contract.
    if fallback is not None:
        return finish(fallback)
    for attempt in range(128):
        A, B = draw_grid(0), draw_grid(1)
        if verified(A, B) is not None:
            return finish({"A": A, "B": B,
                           "component_screen": (10**12, attempt),
                           "small_logical_safe": False})
    raise ValueError("no regular-four filling retained target cell 0.250")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without mutation or late redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance alone raises measured fitness: zero-failure parent slot58 had d2s=22,k=45,s=7.165, below zero-failure slot20 with d2s=20,k=58,s=7.276; their sample ceilings limited both.
# REFUTED: Equal n,k,w,d2s and ASC depth certify comparable circuit reliability: current-parent slots15/17 both had n=588,k=176,w=12,d2s=10,depth=15, but Z failed 100/34750 versus 31/200000 shots and worse-basis s was 5.479 versus 6.277.
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
