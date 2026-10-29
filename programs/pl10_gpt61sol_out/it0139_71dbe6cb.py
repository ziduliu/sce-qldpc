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
CONCEPT = ("For metacyclic Z24:Z2(s=7), order 48, A=(1,2), B=(2,5), heterogeneous binomial orders and selective B-trinomial reinforcement should improve worst-basis circuit reliability when decoder ambiguity, rather than cycles alone, orders the finalists.")
GROUP = ("Frozen under L1. Retain the order-48 bed for comparison with its measured fitness 6.7291. It is C3 x D16: reduction of the rotation exponent modulo three is a quotient homomorphism. Trinomials projecting to a monomial are units because the normal kernel is a 2-group; retain this algebra while changing the labeling architecture.")
SHAPES = ("Frozen (1,2)x(2,5): n=576, kb=144 and guaranteed rate 0.250; the size factor allows q<=49 and this group fixes q=48. Components have 96/240 variables and 48/96 checks. Parent failures span both A columns and several B columns, with mostly weak Acol/Brow check enrichment. Slot19 had 5 Z and 9 X failures, versus slot18's 8 Z and 42 X failures, all at 200000 shots per basis; this does not identify one block to repair.")
FILLING = ("Twenty contracts cross A=(3,4), four legal B profiles and five heterogeneous relative-order portfolios containing orders 8,12,24. Two B profiles introduce a quotient-unit trinomial, one opposite a zero cell. B rows remain <=8 and columns <=4; both CSS weight budgets are <=12, with 27072 or 27936 six-round CNOTs. Retain verified singleton actions, component-equality checks, exact k=144 and CSS validation. Finalists face shared independent/fixed-weight synthetic errors and normalized min-sum component decoding; this optional NumPy screen is an uncalibrated decoder-ambiguity surrogate, not measured circuit pL. Small-boundary exposure precedes cycle tie-breakers. Parent slots14/50/52 all had d2s=12, depth15 and 27072 CNOTs but retained collective failures and different basis imbalances.")
FAMILY_NAME = "r025_m24r2s7_12x25_unit_hybrid_decoder_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # Same metacyclic construction; 7^2 = 1 mod 24.
    # The new factor 12 uses this lift to retain n=576.
    m, r, s = 24, 2, 7
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _design_four_cycles(rows):
    """Count four-cycles within one basis, not across X and Z."""
    seen = {}
    cycles = 0
    for row in rows:
        cols = []
        while row:
            bit = row & -row
            cols.append(bit.bit_length() - 1)
            row ^= bit
        for a in range(len(cols)):
            for b in range(a):
                pair = (cols[b], cols[a])
                previous = seen.get(pair, 0)
                cycles += previous
                seen[pair] = previous + 1
    return cycles


def _design_small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1..4.

    Equal syndromes of supports of size <=2 exhaust this range.
    Reduce their difference against the opposite stabilizer row space.
    Inputs are exclusively the scaffold's verified lifted matrices.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def trivial(vector):
        while vector:
            p = vector.bit_length() - 1
            if p not in pivots:
                return False
            vector ^= pivots[p]
        return True

    syndromes = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] ^= syndrome_bit
            row ^= bit

    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(syndromes):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif not trivial(previous ^ bits[i]):
            return True

    for i in range(n):
        si, vi = syndromes[i], bits[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            vector = vi ^ bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = vector
            elif not trivial(previous ^ vector):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Twenty support/order contracts with shared-noise finalist screening."""
    import math
    from itertools import combinations

    design, batch = idx % 20, idx // 20
    aw = (3, 4)
    bp, mode = divmod(design, 5)
    quart_mode = (mode + batch) % 5
    b_profiles = (
        ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1)),
        ((1, 2), (1, 2), (2, 1), (2, 1), (2, 2)),
        ((1, 3), (2, 1), (2, 1), (2, 1), (1, 2)),
        ((0, 3), (2, 1), (2, 1), (2, 1), (2, 2)),
    )
    sa = [list(aw)]
    sb = [[c[i] for c in b_profiles[bp]] for i in range(2)]
    # The reinforced profiles have row totals 8/8, not 7/9.
    # All column totals are three or four, including the zero/3 column.
    assert max(map(sum, sb)) <= 8
    assert all(3 <= sum(c) <= 4 for c in b_profiles[bp])
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    q = G.q
    mask = (1 << q) - 1
    orders = {
        24: (1, 5, 7, 11, 13, 17, 19, 23),
        12: (2, 10, 14, 22),
        8: (3, 9, 15, 21),
        4: (6, 18),
    }
    tri_order = 4 if batch % 3 == 1 else 8
    # Prescribed heterogeneous portfolios, not uniform-palette redraws.
    # Each first-batch portfolio contains all three rotation orders.
    # Their row-wise order multisets differ on each support profile.
    portfolios = (
        (24, 24, 12, 8, 12),
        (24, 12, 24, 8, 8),
        (24, 8, 12, 24, 12),
        (12, 8, 24, 12, 8),
        (8, 12, 24, 8, 24),
    )
    portfolio = portfolios[mode]
    if batch % 3 == 2:
        order_cycle = {24: 12, 12: 8, 8: 24}
        portfolio = tuple(order_cycle[o] for o in portfolio)
    twist_book = (
        (1, 0, 1, 0, 0),
        (1, 1, 0, 0, 0),
        (1, 0, 0, 1, 1),
        (0, 1, 1, 0, 1),
        (1, 1, 1, 1, 0),
    )
    twists = twist_book[mode]

    # Cache actual singleton actions from the verified helper. All
    # component rows below are XOR-compositions of these exact blocks;
    # no independently implemented left/right permutation convention.
    actions = {
        "A": [{} for _ in range(A_SHAPE[1])],
        "B": [{} for _ in range(B_SHAPE[1])],
    }
    for g in G.elements:
        x, z, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        for side, rows in (("A", x), ("B", z)):
            for j, table in enumerate(actions[side]):
                table[G._key(g)] = tuple(
                    (row & mask) << (j * q) for row in rows)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        result = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    term = actions[side][j][G._key(g)]
                    for t, bits in enumerate(term):
                        out[t] ^= bits
            result.extend(out)
        return result

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def small_kernel(cols):
        """Exact existence test for component kernel weights 1..4."""
        if 0 in cols or len(set(cols)) != len(cols):
            return True
        singles = set(cols)
        pairs = set()
        for i, si in enumerate(cols):
            for j in range(i):
                s = si ^ cols[j]
                if s in singles or s in pairs:
                    return True
                pairs.add(s)
        return False

    def cycle_cost(rows, width, connectivity=False):
        """Exact Tanner four/six cycles, including overlap corrections."""
        cols = columns(rows, width)
        adj = [dict() for _ in rows]
        neighborhoods = []
        for col in cols:
            ns = []
            while col:
                bit = col & -col
                ns.append(bit.bit_length() - 1)
                col ^= bit
            neighborhoods.append(ns)
            for a, b in combinations(ns, 2):
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1
        four = sum(v * (v - 1) // 2
                   for a, neighbors in enumerate(adj)
                   for b, v in neighbors.items() if b > a)
        neighbor_bits = [
            sum(1 << b for b in neighbors) for neighbors in adj
        ]
        six = 0
        for a, neighbors in enumerate(adj):
            for b, ab in neighbors.items():
                if b <= a:
                    continue
                common = (neighbor_bits[a] & neighbor_bits[b]
                          & ~((1 << (b + 1)) - 1))
                while common:
                    bit = common & -common
                    c = bit.bit_length() - 1
                    six += ab * adj[a][c] * adj[b][c]
                    common ^= bit
        for ns in neighborhoods:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        if not connectivity:
            return four, six
        seen, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        return four, six, len(seen) == len(rows)

    def a_cell(j):
        a, b = rng.randrange(24), rng.randrange(2)
        if aw[j] == 3:
            d = rng.choice(orders[tri_order])
            residue = (a + rng.choice((1, 2))) % 3
            choices = [c for c in range(24) if c % 3 == residue]
            if aw == (3, 3) and tri_order == 4:
                # Two distinct reflection parities join the order-24
                # subgraphs that a single order-four trinomial permits.
                choices = [c for c in choices if (c - a) % 2 == j]
            c = rng.choice(choices)
            # First two terms cancel in the C3 quotient. The remaining
            # monomial is a unit; the normal D16 factor is a 2-group.
            return [(a, b), ((a + d) % 24, b), (c, 1 - b)]
        if quart_mode in (2, 4):
            d = rng.choice(orders[8])
            residue = (a + rng.choice((1, 2))) % 3
            long_steps = set(orders[24] + orders[12])
            choices = [
                c for c in range(24)
                if c % 3 == residue
                and (c - a) % 24 in long_steps
                and (c - a - d) % 24 in long_steps
            ]
            c = rng.choice(choices)
            opposite_residue = (-a - c) % 3
            h = 3 * rng.randrange(8) + opposite_residue
            return [(a, b), ((a + d) % 24, b),
                    (c, b), (h, 1 - b)]
        long_order = 24 if quart_mode % 2 == 0 else 12
        d = rng.choice(orders[long_order])
        c = 3 * rng.randrange(8) + (-a - (a + d)) % 3
        e = rng.choice(orders[8])
        # One long-order pair and one order-eight pair; every
        # mixed-coset difference has order six, never order two.
        return [(a, b), ((a + d) % 24, b),
                (c, 1 - b), ((c + e) % 24, 1 - b)]

    def b_cell(i, j, cosets):
        count = sb[i][j]
        if count == 0:
            return []
        a, b = rng.randrange(24), cosets[i][j]
        out = [(a, b)]
        if count == 2:
            order = portfolio[(j + 2 * i) % 5]
            d = rng.choice(orders[order])
            out.append(((a + d) % 24, b))
        elif count == 3:
            # The first pair cancels in C3 and the third term survives.
            # Mixed-coset ratios have nonzero C3 residue, avoiding
            # involutory ratios. This trinomial is a quotient unit.
            d = rng.choice(orders[8])
            residue = (a + rng.choice((1, 2))) % 3
            c = rng.choice([v for v in range(24) if v % 3 == residue])
            out.extend((((a + d) % 24, b), (c, 1 - b)))
        assert len(set(out)) == count
        return out

    def component_pool(side):
        width = (2 if side == "A" else 5) * q
        pool = []
        for restart in range(6):
            if side == "A":
                cosets = None
                grid = [[a_cell(j) for j in range(2)]]
            else:
                first = [rng.randrange(2) for _ in range(5)]
                cosets = [first, [b ^ t for b, t in zip(first, twists)]]
                grid = [[b_cell(i, j, cosets) for j in range(5)]
                        for i in range(2)]
            rows = component(grid, side)
            four, six = cycle_cost(rows, width)
            energy = (200 * four + six) / q
            local_grid, local_energy = clone(grid), energy
            steps = 240 + 20 * (batch % 3)
            for step in range(steps):
                i = 0 if side == "A" else rng.randrange(2)
                j = rng.randrange(2 if side == "A" else 5)
                old = grid[i][j]
                if rng.random() < 0.35:
                    shift = rng.randrange(1, 24)
                    new = [((g[0] + shift) % 24, g[1]) for g in old]
                else:
                    new = (a_cell(j) if side == "A"
                           else b_cell(i, j, cosets))
                grid[i][j] = new
                trial_rows = component(grid, side)
                c4, c6 = cycle_cost(trial_rows, width)
                trial_energy = (200 * c4 + c6) / q
                delta = trial_energy - energy
                temperature = 6.0 * (0.983 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    energy = trial_energy
                    if energy < local_energy or (
                            energy == local_energy and rng.random() < 0.02):
                        local_grid, local_energy = clone(grid), energy
                else:
                    grid[i][j] = old
            rows = component(local_grid, side)
            if gf2_rank(rows) != len(rows):
                continue
            cols = columns(rows, width)
            four, six, connected = cycle_cost(rows, width, True)
            pool.append(dict(grid=local_grid, rows=rows, cols=cols,
                             bad=int(small_kernel(cols)), four=four,
                             six=six, connected=connected))
        pool.sort(key=lambda e: (
            e["bad"], not e["connected"], e["four"], e["six"]))
        if not pool:
            raise ValueError("no full-row-rank component in contract")
        return pool[:5]

    def a_five_six(cols):
        """Exact weight-five/six multiplicity when weights <=4 are absent."""
        pairs = {cols[i] ^ cols[j]
                 for i in range(len(cols)) for j in range(i)}
        triples = {}
        five = six = 0
        for i, si in enumerate(cols):
            for j in range(i):
                pair = si ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    five += int(syndrome in pairs)
                    old = triples.get(syndrome, 0)
                    six += old
                    triples[syndrome] = old + 1
        return five // 10, six // 10

    def anchored_boundaries(cols):
        """Small triple boundaries, sampled uniformly over lift orbits.

        One anchor per protograph column represents its group orbit.
        This is a trapping-set surrogate, not a circuit-failure estimate.
        """
        block_costs = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i, si in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ si
                for j in range(i):
                    if j == anchor:
                        continue
                    w = (pair ^ cols[j]).bit_count()
                    if w <= 4:
                        cost += (5 - w) ** 2
            block_costs.append(cost)
        return max(block_costs), sum(block_costs)

    def kernel_scan(cols, trials):
        """Actual kernel witnesses from random information sets.

        Greedy basis reductions explore further verified kernel words.
        Returned distances remain upper bounds, never certificates.
        """
        order = list(range(len(cols)))
        best = len(cols) + 1
        short = set()
        for trial in range(trials):
            rng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p in pivots:
                        s, v = pivots[p]
                        syndrome ^= s
                        word ^= v
                    else:
                        pivots[p] = (syndrome, word)
                        break
                if not syndrome:
                    basis.append(word)
                    w = word.bit_count()
                    best = min(best, w)
                    if w <= 12:
                        short.add(word)
            for sweep in range(2):
                rng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for j in range(i):
                        v = word ^ basis[j]
                        w = v.bit_count()
                        best = min(best, w)
                        if w <= 12:
                            short.add(v)
                        if w < weight:
                            word, weight = v, w
                    basis[i] = word
            if best <= 6:
                break
        return best, sum(v.bit_count() <= 10 for v in short)

    ap, bpools = component_pool("A"), component_pool("B")

    try:
        import numpy as np
    except ImportError:
        np = None

    noise_books = {}
    if np is not None:
        for side, width in (("A", 2 * q), ("B", 5 * q)):
            errors = np.zeros((width, 256), dtype=np.uint8)
            # Same errors for every finalist on this side. Independent
            # p=.025 patterns probe distributed noise; weights 4..6
            # deliberately expose near-threshold decoder ambiguities.
            # Neither distribution models the ASC circuit exactly.
            for shot in range(160):
                for c in range(width):
                    errors[c, shot] = int(rng.random() < 0.025)
            for shot in range(160, 256):
                weight = 4 + (shot - 160) % 3
                errors[rng.sample(range(width), weight), shot] = 1
            noise_books[side] = errors

    def decoder_probe(rows, side):
        """Shared-noise normalized min-sum on a verified component.

        Counts classical decoding failures, including unsettled shots.
        This is not relay-BP, not stabilizer-aware quantum decoding,
        and not a circuit-level failure-rate measurement.
        """
        if np is None:
            return (0, 0, 0)
        errors = noise_books[side]
        width, shots = errors.shape
        matrix = np.zeros((len(rows), width), dtype=np.uint8)
        edge_vars = []
        row_edges = []
        var_edges = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            edges = []
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                row ^= bit
                edge = len(edge_vars)
                edge_vars.append(c)
                edges.append(edge)
                var_edges[c].append(edge)
                matrix[r, c] = 1
            row_edges.append(edges)

        nedges = len(edge_vars)
        variables = np.asarray(edge_vars, dtype=np.intp)
        rd = max(map(len, row_edges))
        vd = max(map(len, var_edges))
        rmap = np.full((len(rows), rd), nedges, dtype=np.intp)
        vmap = np.full((width, vd), nedges, dtype=np.intp)
        for r, edges in enumerate(row_edges):
            rmap[r, :len(edges)] = edges
        for c, edges in enumerate(var_edges):
            vmap[c, :len(edges)] = edges
        valid = rmap != nedges

        target = (matrix @ errors) & 1
        target_sign = np.where(target != 0, -1.0, 1.0)
        prior = math.log((1.0 - 0.025) / 0.025)
        incoming = np.full((nedges + 1, shots), prior, dtype=float)
        incoming[nedges] = 30.0
        messages = np.zeros_like(incoming)
        final = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        last = np.zeros_like(errors)
        slots = np.arange(rd)[None, :, None]

        for iteration in range(40):
            gathered = incoming[rmap]
            signs = np.where(gathered >= 0, 1.0, -1.0)
            absolute = np.abs(gathered)
            smallest_at = np.argmin(absolute, axis=1)
            first = np.min(absolute, axis=1)
            second = np.partition(absolute, 1, axis=1)[:, 1, :]
            excluded = np.where(
                slots == smallest_at[:, None, :],
                second[:, None, :], first[:, None, :])
            parity = np.prod(signs, axis=1) * target_sign
            outgoing = 0.80 * excluded * signs * parity[:, None, :]
            messages[rmap[valid]] = outgoing[valid]
            messages[nedges] = 0.0

            posterior = prior + messages[vmap].sum(axis=1)
            last = (posterior < 0).astype(np.uint8)
            if iteration % 4 == 3:
                settled = (
                    np.all(((matrix @ last) & 1) == target, axis=0)
                    & active)
                final[:, settled] = last[:, settled]
                active[settled] = False
                if not active.any():
                    break

            extrinsic = posterior[variables] - messages[:nedges]
            incoming[:nedges] = (
                0.20 * incoming[:nedges]
                + 0.80 * np.clip(extrinsic, -20.0, 20.0))
            incoming[nedges] = 30.0

        final[:, active] = last[:, active]
        different = final != errors
        failed = np.any(different, axis=0) | active
        return (int(failed.sum()), int(active.sum()),
                int((different & failed[None, :]).sum()))

    for side, pool in (("A", ap), ("B", bpools)):
        for entry in pool:
            entry["d"], entry["short"] = kernel_scan(
                entry["cols"], 64 if side == "A" else 32)
            entry["traps"] = anchored_boundaries(entry["cols"])
            entry["five_six"] = (
                a_five_six(entry["cols"])
                if side == "A" and not entry["bad"] else (0, 0))
            entry["decoder"] = decoder_probe(entry["rows"], side)

    options = []
    for a in ap:
        for b in bpools:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            # Check that cached singleton composition screens precisely
            # the components present in the final evaluator matrices.
            ar = [(row & mask)
                  | (((row >> (5 * q)) & mask) << q)
                  for row in hx[:q]]
            br = [row & ((1 << (5 * q)) - 1) for row in hz[:2 * q]]
            assert ar == a["rows"] and br == b["rows"]
            cx, cz = _design_four_cycles(hx), _design_four_cycles(hz)
            da, db = a["d"], b["d"]
            ta, tb = a["traps"], b["traps"]
            five, six = a["five_six"]
            pa, pb = a["decoder"], b["decoder"]
            key = (
                a["bad"] + b["bad"],
                int(five > 0) + int(six > 0),
                max(0, 10 - da) + max(0, 10 - db),
                int(not a["connected"]) + int(not b["connected"]),
                max(pa[0], pb[0]), pa[0] + pb[0],
                pa[1] + pb[1], pa[2] + pb[2],
                max(ta[0], tb[0]), ta[1] + tb[1],
                -min(da, db, 12),
                max(cx / len(hx), cz / len(hz)), cx + cz,
                a["short"] + b["short"],
                a["six"] + b["six"], five + six,
            )
            options.append((key, A, B, hx, hz, n))
    options.sort(key=lambda item: item[0])

    fallback = None
    for _, A, B, hx, hz, n in options:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 144:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        candidate = {"idx": idx, "cells": {"A": A, "B": B}}
        if fallback is None:
            fallback = candidate
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        return candidate
    if fallback is not None:
        return fallback
    raise ValueError("no exact-bound filling passed verified rank screen")


def CELL(rng, G, side, i, j, ctx):
    """Emit a fresh copy of the complete algebraically screened cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Smaller ASC depth alone ensures better balanced memory: current slot42 at depth14 failed 91 X and 11 Z shots, while slot5 at depth15 failed 26 X and 17 Z shots, each out of 200000.
# REFUTED: Fixing n,k,d2s,weight,ASC depth and CNOT count fixes memory reliability: parent slots14/50 both had 576,144,12,12,15,27072 but respectively failed 15/26 X shots and 4/12 Z shots per 200000.
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
