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
CONCEPT = ("For generalized quaternion Q32, order 32, shapes (1,3)x(2,5), odd-augmentation unit trinomials and indexed relative-order contracts should improve worse-basis circuit fitness beyond cell 0.350's incumbent 5.0888.")
GROUP = ("Jump from cyclic Z17 to Q32: x^16=1, y^2=x^8, yxy^-1=x^-1. In F2 of a finite 2-group every odd-augmentation element is a unit. This controls component ranks without optimistic distance assumptions. The listed centre-two, eleven-class order-32 bed might be Q32 and scored about 4.4 at these shapes; it cautions against a group-only advantage and motivates a different unit/order filling.")
SHAPES = ("Target cell 0.350 with (1,3)x(2,5): size factor 17, q=32, n=544, kb=192, guaranteed rate 6/17=0.352941. A has 96 variables/32 checks and B 160 variables/64 checks. A contains unit cells and B has full augmentation row rank; require final verified k=192. Parent slot20 had 100/8250 Z failures versus 100/50250 X failures, affecting all 204 logicals with diffuse Acol/Bcol and Acol/Brow enrichment, not a demonstrated isolated hotspot.")
FILLING = ("Twenty contracts cross five distinct A rotation-order architectures with four B degree-three support profiles. Extra batches change B relative differences from even rotations to primitive rotations to mixed-coset order-four elements, not transformations of saved codes. A rows total nine; B rows seven/eight and columns three, so HX=12 and HZ<=11. Six-round CNOT count is 23616 versus 25092, but density per qubit is unchanged. Multistart searches compose only verified singleton actions and screen connectivity, ranks, short kernels, sampled witness concentration and small boundaries. Final helper-built matrices select mixed-cycle burden, verify CSS and k, and prefer exclusion of quantum logicals through four. These are structural screens, not ASC simulations or pL certificates.")
FAMILY_NAME = "r035_Q32_13x25_unit_ordercontracts_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q32 requires order 32")
    # Q32 = Dic8: x^16=1, y^2=x^8, y*x*y^-1=x^-1.
    # Canonical elements are x^a*y^b, b in {0,1}.
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

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _legacy_col3_ctx(rng, G, idx):
    """Inactive parent search, retained as source history; never called."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    # Entries give the upper-row count; the lower-row count is 3-entry.
    # All A rows total 7/8; all B rows total 9/9.
    a_profiles = (
        (2, 2, 1, 1, 1),
        (0, 2, 2, 2, 1),
        (3, 0, 2, 1, 1),
        (3, 0, 0, 2, 2),
        (3, 3, 0, 0, 1),
    )
    b_profiles = (
        (2, 2, 2, 1, 1, 1),
        (0, 2, 2, 2, 2, 1),
        (3, 0, 2, 2, 1, 1),
        (3, 0, 0, 2, 2, 2),
    )
    pa = a_profiles[(idx // 4) % len(a_profiles)]
    pb = b_profiles[idx % len(b_profiles)]
    counts = {
        "A": [list(pa), [3 - x for x in pa]],
        "B": [list(pb), [3 - x for x in pb]],
    }
    assert _grid_weights(counts["A"], counts["B"]) <= W_MAX
    assert _no_dead_lines(counts["A"], counts["B"])

    dummy_a = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    dummy_b = [[[G.e] for _ in range(cb)] for _ in range(rb)]
    mask = (1 << q) - 1

    def draw(side):
        return [[rng.sample(G.elements, t) for t in row]
                for row in counts[side]]

    def component_columns(side, grid):
        # Project actual helper-built matrices; do not substitute a lift.
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, dummy_b)
            cols = [0] * (ca * q)
            for i in range(ra):
                for r in range(q):
                    row = hx[(i * cb) * q + r]  # B-column zero
                    check_bit = 1 << (i * q + r)
                    for j in range(ca):
                        bits = (row >> (j * cb * q)) & mask
                        while bits:
                            bit = bits & -bits
                            cols[j * q + bit.bit_length() - 1] ^= check_bit
                            bits ^= bit
        else:
            _, hz, _ = lifted_matrices(G, dummy_a, grid)
            cols = [0] * (cb * q)
            sector_mask = (1 << (cb * q)) - 1
            for a in range(rb):
                for r in range(q):
                    bits = hz[a * q + r] & sector_mask  # A-column zero
                    check_bit = 1 << (a * q + r)
                    while bits:
                        bit = bits & -bits
                        cols[bit.bit_length() - 1] ^= check_bit
                        bits ^= bit
        assert all(c.bit_count() == 3 for c in cols)
        return cols

    def cycle_data(cols):
        """Exact component four-cycle count and check adjacency."""
        pairs = {}
        adj = [0] * (2 * q)
        conflicts = 0
        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a in range(3):
                u = nodes[a]
                for b in range(a + 1, 3):
                    v = nodes[b]
                    key = (1 << u) | (1 << v)
                    old = pairs.get(key, 0)
                    conflicts += old
                    pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return conflicts, adj

    def kernel_metric(cols, adj):
        """Exact weight-four count; weight-six count when weight four is absent."""
        pair_hist = {}
        collisions = 0
        nc = len(cols)
        for a in range(nc - 1):
            for b in range(a + 1, nc):
                syn = cols[a] ^ cols[b]
                old = pair_hist.get(syn, 0)
                collisions += old
                pair_hist[syn] = old + 1
        # Unique degree-three columns: each weight-four kernel has
        # exactly three unordered pair partitions.
        n4 = collisions // 3

        triangles = 0
        for u in range(len(adj)):
            bits = adj[u] & ~((1 << (u + 1)) - 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() - 1
                triangles += (
                    adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                ).bit_count()
                bits ^= bit
        # With no four-cycles, subtract triangles within one column.
        n6cycles = triangles - nc
        if n4:
            return (n4, 10**9, n6cycles)

        triple_hist = {}
        collisions = 0
        for a in range(nc - 2):
            for b in range(a + 1, nc - 1):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc):
                    syn = ab ^ cols[c]
                    old = triple_hist.get(syn, 0)
                    collisions += old
                    triple_hist[syn] = old + 1
        # Without weight-two/four kernels, colliding triples are disjoint.
        # Each weight-six kernel has ten unordered triple partitions.
        return (0, collisions // 10, n6cycles)

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def optimize(side):
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        current = draw(side)
        cols = component_columns(side, current)
        cost, adj = cycle_data(cols)
        fallback, fallback_cost = current, cost
        best, best_metric = None, None
        tested = set()

        for step in range(1600):
            if cost == 0:
                sig = signature(current)
                if sig not in tested:
                    tested.add(sig)
                    metric = kernel_metric(cols, adj)
                    if best_metric is None or metric < best_metric:
                        best, best_metric = current, metric
                    if len(tested) >= 20 and best_metric[0] == 0:
                        break

            i, j = sites[rng.randrange(len(sites))]
            trial = [list(row) for row in current]
            trial[i][j] = rng.sample(G.elements, counts[side][i][j])
            trial_cols = component_columns(side, trial)
            trial_cost, trial_adj = cycle_data(trial_cols)
            if trial_cost < fallback_cost:
                fallback, fallback_cost = trial, trial_cost

            # Neutral moves explore genuinely different kernels. Rare
            # uphill moves and restarts escape four-cycle local minima.
            if trial_cost <= cost or rng.random() < 0.015:
                current, cols = trial, trial_cols
                cost, adj = trial_cost, trial_adj
            if step % 200 == 199 and cost:
                current = draw(side)
                cols = component_columns(side, current)
                cost, adj = cycle_data(cols)
                if cost < fallback_cost:
                    fallback, fallback_cost = current, cost

        # The fallback remains hard-rule legal even if screening misses
        # a four-cycle-free filling; no invalid slot is silently repaired.
        return best if best is not None else fallback

    cells = {"A": optimize("A"), "B": optimize("B")}
    hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k <= 0 or not css_ok(hx, hz):
        raise ValueError("verified final CSS/rank check failed")
    return {"idx": idx, "cells": cells, "k": k}


def NEW_CTX(rng, G, idx):
    """Build one quaternion unit/order contract using verified actions."""
    import math

    q = G.q
    assert q == 32
    design, batch = idx % 20, idx // 20
    a_model, b_model = divmod(design, 4)
    policy = batch % 3

    # The five cards have different rotation-order multisets:
    # 16/16/16, 16/16/8, 16/8/8, 16/8/4, and 8/4/8.
    # Noncommuting mixed-coset partners accompany every rotation.
    a_steps = (
        (1, 3, 5),
        (1, 3, 2),
        (1, 2, 6),
        (1, 2, 4),
        (2, 4, 6),
    )[a_model]
    first = (
        (2, 2, 1, 1, 1),
        (0, 2, 2, 2, 1),
        (3, 0, 2, 1, 1),
        (3, 0, 0, 2, 2),
    )[b_model]
    supports = {
        "A": [[3, 3, 3]],
        "B": [list(first), [3-t for t in first]],
    }
    sa, sb = supports["A"], supports["B"]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert tuple(map(sum, sb)) == (7, 8)
    assert all(sum(column) == 3 for column in zip(*sb))
    assert gf2_rank([
        sum((t & 1) << j for j, t in enumerate(row)) for row in sb
    ]) == 2

    # These order policies change relative elements, not whole-code gauges.
    palettes = (
        [(a, 0) for a in (2, 4, 6)],       # orders 8,4,8
        [(a, 0) for a in (1, 3, 5, 7)],    # order 16
        [(a, 1) for a in range(8)],        # order 4
    )
    palette = palettes[policy]

    def relative_key(cell):
        h = G.mul(G.inv(cell[0]), cell[1])
        return min(G.idx(h), G.idx(G.inv(h)))

    def draw_cell(side, i, j, forbidden=()):
        count = supports[side][i][j]
        if not count:
            return []
        anchor = G.pick(rng)
        if count == 1:
            return [anchor]
        if count == 2:
            available = [
                h for h in palette
                if min(G.idx(h), G.idx(G.inv(h))) not in forbidden
            ]
            if not available:
                raise ValueError("binomial relative-order palette exhausted")
            return [anchor, G.mul(anchor, rng.choice(available))]

        if side == "A":
            a = a_steps[j]
        else:
            # Pure B trinomials remain noncommuting, including the
            # architecture having two pure columns in one row.
            rotations = (1, 3, 5, 7) if policy == 1 else (2, 6)
            a = rng.choice(rotations)
        b = rng.randrange(16)
        # Extra batches alter the initial relative embedding as well
        # as the B order policy; optimization never transforms a saved code.
        b = (b + 2*(batch // 3)*(j+1)) % 16
        relative = [G.e, (a, 0), (b, 1)]
        assert G.mul(relative[1], relative[2]) != G.mul(
            relative[2], relative[1])
        return [G.mul(anchor, h) for h in relative]

    def draw_grid(side):
        grid = []
        for i, counts in enumerate(supports[side]):
            row = []
            for j, count in enumerate(counts):
                used = {relative_key(c) for c in row if len(c) == 2}
                row.append(draw_cell(side, i, j, used))
            grid.append(row)
        return grid

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    mask = (1 << q)-1
    actions = {"A": [], "B": []}
    for g in G.elements:
        # Exact singleton actions from the supplied helper. No
        # independent implementation of a left or right lift is used.
        ar = lifted_matrices(G, [[[g]]], [[[]]])[0]
        br = lifted_matrices(G, [[[]]], [[[g]]])[1]
        actions["A"].append(tuple(row & mask for row in ar))
        actions["B"].append(tuple(row & mask for row in br))

    def component_rows(side, grid):
        rows = []
        for grid_row in grid:
            out = [0]*q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, bits in enumerate(block):
                        out[r] ^= bits << (j*q)
            rows.extend(out)
        return rows

    def columns_of(rows, width):
        columns = [0]*width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length()-1] |= tag
                row ^= bit
        return columns

    def cycle_data(rows, columns):
        """Exact C4; triangle surrogate for C6 when C4 is nonzero."""
        adjacency = [0]*len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap*(overlap-1)//2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        unseen = (1 << len(rows))-1
        parts = 0
        while unseen:
            parts += 1
            pending = unseen & -unseen
            while pending:
                bit = pending & -pending
                pending ^= bit
                if not unseen & bit:
                    continue
                unseen ^= bit
                pending |= adjacency[bit.bit_length()-1] & unseen

        triangles = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles += (
                    neighbors & adjacency[j] & ((1 << j)-1)
                ).bit_count()
                lower ^= bit
        stars = sum(
            d*(d-1)*(d-2)//6
            for d in (column.bit_count() for column in columns)
        )
        return parts, four, max(0, triangles-stars)

    def short_boundaries(columns):
        """Kernel collision zero-tests through six and triple boundaries.

        All columns have odd degree three, so odd-weight kernel words
        are impossible. Collision counts are exposure proxies; their
        zero values exclude weights two, four and six exactly.
        """
        two = len(columns)-len(set(columns))
        pairs = {}
        four = 0
        for i, value in enumerate(columns):
            for other in columns[:i]:
                syndrome = value ^ other
                four += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0)+1
        if two or four:
            return two, four, 10**9, 10**9

        triples = {}
        six = boundary = 0
        for i, value in enumerate(columns):
            for j in range(i):
                pair = value ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    six += triples.get(syndrome, 0)
                    triples[syndrome] = triples.get(syndrome, 0)+1
                    weight = syndrome.bit_count()
                    if weight < 5:
                        boundary += (5-weight)**2
        return 0, 0, six, boundary

    def sampled_exposure(columns, trials):
        """Actual kernel witnesses; never a distance lower bound."""
        order = list(range(len(columns)))
        minimum = len(columns)+1
        short = set()

        def observe(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 8 and len(short) < 4096:
                    short.add(word)

        for _ in range(trials):
            rng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    observe(word)

            for sweep in range(2):
                rng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        proposal = word ^ other
                        observe(proposal)
                        if proposal and proposal.bit_count() < weight:
                            word = proposal
                            weight = word.bit_count()
                    basis[i] = word
            if minimum <= 4:
                break

        loads = [
            sum(bool((word >> (j*q)) & mask) for word in short)
            for j in range(len(columns)//q)
        ]
        return minimum, len(short), max(loads, default=0)

    def select_component(side):
        width = len(supports[side][0])*q
        sites = [(i, j) for i, row in enumerate(supports[side])
                 for j, count in enumerate(row) if count]
        pool, seen = [], set()

        def analyze(grid):
            rows = component_rows(side, grid)
            columns = columns_of(rows, width)
            assert all(column.bit_count() == 3 for column in columns)
            parts, four, six = cycle_data(rows, columns)
            deficit = len(rows)-gf2_rank(rows)
            energy = (300*(parts-1) + 100*deficit
                      + (100*four+six)/q)
            return energy, (parts-1, deficit, four, six), rows, columns

        def retain(grid, result):
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            if signature in seen:
                return
            seen.add(signature)
            energy, coarse, rows, columns = result
            # Neutral minima retain multiple independently optimized
            # embeddings instead of always retaining the earliest gauge.
            pool.append((coarse, rng.random(), clone(grid),
                         list(rows), list(columns)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(3):
            grid = draw_grid(side)
            current = analyze(grid)
            retain(grid, current)
            local = current[0]
            steps = 320 + 20*(batch % 3)
            for step in range(steps):
                i, j = rng.choice(sites)
                old = grid[i][j]
                forbidden = {
                    relative_key(cell)
                    for k, cell in enumerate(grid[i])
                    if k != j and len(cell) == 2
                }
                grid[i][j] = draw_cell(side, i, j, forbidden)
                proposed = analyze(grid)
                delta = proposed[0]-current[0]
                temperature = 16.0*(0.986**step)+0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current = proposed
                else:
                    grid[i][j] = old
                improved = current[0] < local
                local = min(local, current[0])
                if improved or step % 32 == 31:
                    retain(grid, current)
            retain(grid, current)

        finalists = []
        for coarse, tie, grid, rows, columns in pool[:4]:
            two, four, six, boundary = short_boundaries(columns)
            distance, count, peak = sampled_exposure(
                columns, 40 if side == "A" else 32)
            parts, deficit, c4, c6 = coarse
            # Once the sampled floor is met, prioritize boundaries and
            # collective exposure rather than larger optimistic distances.
            key = (parts, deficit, two, four, six,
                   max(0, 8-distance), boundary/width,
                   count/width, peak/width, c4/q, c6/q)
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        return finalists[:3]

    def full_four(rows):
        return sum(
            overlap*(overlap-1)//2
            for i, row in enumerate(rows)
            for other in rows[:i]
            for overlap in [(row & other).bit_count()]
        )

    def has_short_logical(hx, hz, n):
        """Exact quantum-logical existence test through weight four."""
        for checks, stabilizers in ((hx, hz), (hz, hx)):
            pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length()-1
                    if p in pivots:
                        row ^= pivots[p]
                    else:
                        pivots[p] = row
                        break

            def trivial(word):
                while word:
                    p = word.bit_length()-1
                    if p not in pivots:
                        return False
                    word ^= pivots[p]
                return True

            columns = columns_of(checks, n)
            seen = {0: 0}
            for i, syndrome in enumerate(columns):
                word = 1 << i
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif not trivial(word ^ old):
                    return True
            for i, value in enumerate(columns):
                for j in range(i):
                    syndrome = value ^ columns[j]
                    word = (1 << i) | (1 << j)
                    old = seen.get(syndrome)
                    if old is None:
                        seen[syndrome] = word
                    elif not trivial(word ^ old):
                        return True
        return False

    apool = select_component("A")
    bpool = select_component("B")
    options = []
    for ak, A in apool:
        for bk, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 544 or k != 192:
                continue
            cx, cz = full_four(hx), full_four(hz)
            joint = tuple(
                value for a, b in zip(ak[:9], bk[:9])
                for value in (max(a, b), a+b)
            )
            key = joint + (
                max(cx/len(hx), cz/len(hz)), cx+cz,
                ak[9]+bk[9], ak[10]+bk[10],
            )
            options.append((key, A, B, hx, hz, n, k))
    options.sort(key=lambda item: item[0])
    if not options:
        raise ValueError("no Q32 unit/order pair retained verified k=192")

    selected, safe = options[0], False
    for item in options[:3]:
        _, _, _, hx, hz, n, _ = item
        if not has_short_logical(hx, hz, n):
            selected, safe = item, True
            break

    key, A, B, hx, hz, n, k = selected
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) == 11
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 6*sum(row.bit_count() for row in hx+hz) == 23616
    for side, grid in (("A", A), ("B", B)):
        width = len(grid[0])*q
        expected = (lifted_matrices(G, grid, [[[]]])[0]
                    if side == "A" else
                    lifted_matrices(G, [[[]]], grid)[1])
        keep = (1 << width)-1
        assert component_rows(side, grid) == [
            row & keep for row in expected
        ]
        assert all(len(cell) <= CELL_MAX
                   and len({G._key(g) for g in cell}) == len(cell)
                   for row in grid for cell in row)

    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "contract": (a_model, b_model, policy, batch),
            "screen": key, "small_logical_safe": safe}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases establish balanced memory reliability: current-parent slot20 had 100/8250 failed Z shots versus 100/50250 failed X shots, with diffuse block enrichment and flips across all 204 logical qubits.
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
