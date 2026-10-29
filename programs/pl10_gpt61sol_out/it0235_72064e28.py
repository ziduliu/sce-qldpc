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
CONCEPT = ("For metacyclic Z24:Z2(s=7), order 48, A=(1,2), B=(2,5), noncommuting B doubletons and stronger unit-pentanomial A protection should reduce worse-basis circuit failures at rate 0.250; measured pL can refute this.")
GROUP = ("Frozen byte-for-byte under L1. This exact bed achieved fitness 6.7291. Its normal D16 subgroup has C3 quotient; cells projecting to a quotient monomial are units because the kernel augmentation ideal is nilpotent. This motivates new odd-support cells, not a claimed group advantage.")
SHAPES = ("Frozen (1,2)x(2,5): n=576, kb=144, guaranteed rate 0.250; q<=49 is size-legal and the group fixes q=48. Components have 96 variables/48 checks and 240 variables/96 checks. Parent failures implicated several Acol/Bcol and Acol/Brow blocks, mostly weakly; the maps do not establish one isolated repair target. Exact final ranks require k=144, retaining zero excess over kb.")
FILLING = ("Twenty contracts cross four count architectures with five relative-order policies. B gains mixed-coset doubletons with order-six ratios. A uses a quotient-unit trinomial plus a homogeneous Sidon quartet or a quotient-unit pentanomial. The (3,5) contracts use B rows 7/7 and column totals 2/3, keeping both check budgets <=12; six-round CNOT counts are 27072,27936,28224. Retain multiple component-search checkpoints, cap sampled protection rewards at twelve, and prioritize small-boundary exposure over cycle tie-breakers. Complete verified CSS finalists receive stabilizer-aware normalized min-sum probes with independent errors and correlated opposite-check subsets. This optional NumPy probe is not relay-BP, ASC scheduling or measured circuit pL. It targets collective decoder failures: parent slot18 flipped 1364 logical qubits in 42 X failures, while slot52 was Z-limited. Final matrices require exact rank, CSS and legal weights, and prefer no logical <=4.")
FAMILY_NAME = "r025_m24r2s7_12x25_unit5_mixed6_cssprobe_v4"

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
    """Twenty count/ratio contracts; all optimization state is local."""
    import math
    from itertools import combinations

    design, batch = idx % 20, idx // 20
    architecture, mode = divmod(design, 5)
    contracts = (
        ((3, 4), ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1))),
        ((3, 4), ((1, 2), (1, 2), (2, 1), (2, 1), (2, 2))),
        ((3, 5), ((1, 2), (1, 2), (2, 1), (2, 1), (1, 1))),
        ((3, 5), ((0, 3), (2, 1), (2, 1), (2, 1), (1, 1))),
    )
    aw, profile = contracts[architecture]
    sa = [list(aw)]
    sb = [[c[i] for c in profile] for i in range(2)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert gf2_rank([
        sum((w % 2) << j for j, w in enumerate(row))
        for row in sb
    ]) == 2

    q = G.q
    mask = (1 << q) - 1
    orders = {
        24: (1, 5, 7, 11, 13, 17, 19, 23),
        12: (2, 10, 14, 22),
        8: (3, 9, 15, 21),
        4: (6, 18),
    }
    tri_order = 4 if mode == 4 else 8
    if batch % 3 == 1:
        tri_order = 12 - tri_order
    portfolios = (
        (24, 12, 8),
        (24, 8, 12),
        (12, 24, 8),
        (8, 24, 12),
        (12, 8, 24),
    )
    portfolio = portfolios[(mode + batch) % 5]
    twist_book = (
        (1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (1, 0, 0, 1, 0),
        (0, 1, 0, 1, 0),
        (1, 1, 0, 0, 1),
    )
    twists = twist_book[(mode + batch) % 5]

    sites = [(i, j) for i in range(2) for j in range(5)
             if sb[i][j] == 2]
    start = (mode + 2 * batch) % 5
    sites.sort(key=lambda p: (
        (p[1] - start) % 5, p[0] ^ ((mode + batch) % 2)))
    mixed_count = min(len(sites), (1, 2, 3, 3, 4)[mode])
    mixed_sites = set(sites[:mixed_count])
    mixed_ratios = [(a, 1) for a in range(24) if a % 3]
    # For this frozen law, these ratios have order six, not two.
    for g in mixed_ratios:
        power = G.e
        for t in range(1, 7):
            power = G.mul(power, g)
            if G._key(power) == G._key(G.e):
                assert t == 6
                break
        else:
            raise AssertionError("mixed ratio is not order six")

    # Homogeneous quartets with distinct undirected differences and
    # all internal rotation orders >=8. Quotient translations here
    # remove redundant catalogue entries, not generate extra slots.
    quartet_book = set()
    for tail in combinations(range(1, 24), 3):
        values = (0,) + tail
        differences = [
            (a - b) % 24 for a, b in combinations(values, 2)]
        directions = [min(d, 24 - d) for d in differences]
        if len(set(directions)) != 6:
            continue
        if any(24 // math.gcd(d, 24) < 8 for d in differences):
            continue
        signature = min(
            tuple(sorted((a - origin) % 24 for a in values))
            for origin in values)
        quartet_book.add(signature)
    quartet_book = sorted(quartet_book)
    assert quartet_book

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
        base = G.pick(rng)
        if j == 0:
            d = rng.choice(orders[tri_order])
            c = rng.choice([a for a in range(24) if a % 3])
            relative = [G.e, (d, 0), (c, 1)]
            # The first pair cancels in C3; the third monomial survives.
        elif aw[j] == 4:
            values = rng.choice(quartet_book)
            relative = [(a, 0) for a in values]
            # This entire quartet is homogeneous, unlike the parent's
            # two-coset four-term models. A0 supplies the required unit.
        else:
            first_order = (8, 8, 4, 8, 4)[mode]
            second_order = (8, 4, 8, 8, 4)[mode]
            if batch % 3 == 2:
                first_order, second_order = second_order, first_order
            d = rng.choice(orders[first_order])
            e = rng.choice(orders[second_order])
            residue = 1 + (mode + batch) % 2
            c = rng.choice([a for a in range(24) if a % 3 == residue])
            h = rng.choice([a for a in range(24)
                            if a % 3 == (-residue) % 3])
            relative = [G.e, (d, 0), (c, 0),
                        (h, 1), ((h + e) % 24, 1)]
            # Two pairs cancel in C3, leaving (c,0). Thus the five-term
            # block is a unit, not an assumed odd-augmentation unit in
            # the full non-two-group algebra.
        out = [G.mul(base, g) for g in relative]
        assert len({G._key(g) for g in out}) == aw[j]
        return out

    def b_cell(i, j, cosets):
        count = sb[i][j]
        if count == 0:
            return []
        base = (rng.randrange(24), cosets[i][j])
        if count == 1:
            return [base]
        if count == 2:
            if (i, j) in mixed_sites:
                ratio = rng.choice(mixed_ratios)
            else:
                order = portfolio[(j + 2 * i + batch) % 3]
                ratio = (rng.choice(orders[order]), 0)
            out = [base, G.mul(base, ratio)]
        else:
            # The zero/3 column contains an invertible trinomial.
            # Its first pair cancels in C3 and its third term survives.
            d = rng.choice(orders[tri_order])
            c = rng.choice([a for a in range(24) if a % 3])
            out = [base, G.mul(base, (d, 0)), G.mul(base, (c, 1))]
        assert len({G._key(g) for g in out}) == count
        return out

    def component_pool(side):
        width = (2 if side == "A" else 5) * q
        pool, seen = [], set()

        def fresh():
            if side == "A":
                return [[a_cell(j) for j in range(2)]], None
            first = [rng.randrange(2) for _ in range(5)]
            cosets = [first, [b ^ t for b, t in zip(first, twists)]]
            return [[b_cell(i, j, cosets) for j in range(5)]
                    for i in range(2)], cosets

        def retain(grid):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            rows = component(grid, side)
            if gf2_rank(rows) != len(rows):
                return
            cols = columns(rows, width)
            four, six, connected = cycle_cost(rows, width, True)
            pool.append(dict(
                grid=clone(grid), rows=rows, cols=cols,
                bad=int(small_kernel(cols)), four=four,
                six=six, connected=connected))
            pool.sort(key=lambda e: (
                e["bad"], not e["connected"], e["four"], e["six"]))
            del pool[8:]

        for restart in range(8):
            grid, cosets = fresh()
            rows = component(grid, side)
            four, six = cycle_cost(rows, width)
            energy = (160 * four + six) / q
            local_grid, local_energy = clone(grid), energy
            retain(grid)
            steps = 220 + 20 * (batch % 3)
            for step in range(steps):
                i = 0 if side == "A" else rng.randrange(2)
                j = rng.randrange(2 if side == "A" else 5)
                old = grid[i][j]
                if old and rng.random() < 0.25:
                    shift = rng.randrange(1, 24)
                    new = [((g[0] + shift) % 24, g[1]) for g in old]
                else:
                    new = (a_cell(j) if side == "A"
                           else b_cell(i, j, cosets))
                grid[i][j] = new
                trial_rows = component(grid, side)
                c4, c6 = cycle_cost(trial_rows, width)
                trial_energy = (160 * c4 + c6) / q
                delta = trial_energy - energy
                temperature = 8.0 * (0.984 ** step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    energy = trial_energy
                    if energy < local_energy or (
                            energy == local_energy and rng.random() < 0.02):
                        local_grid, local_energy = clone(grid), energy
                    if step % 55 == 54:
                        retain(grid)
                else:
                    grid[i][j] = old
            retain(local_grid)
            retain(grid)

        # Fresh rank rescue stays inside the same count/ratio contract.
        if not pool:
            for _ in range(96):
                grid, _ = fresh()
                retain(grid)
                if len(pool) >= 4:
                    break
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
    for side, pool in (("A", ap), ("B", bpools)):
        for entry in pool:
            entry["d"], entry["short"] = kernel_scan(
                entry["cols"], 64 if side == "A" else 32)
            entry["traps"] = anchored_boundaries(entry["cols"])
            entry["five_six"] = (
                a_five_six(entry["cols"])
                if side == "A" and not entry["bad"] else (0, 0))

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
            key = (
                a["bad"] + b["bad"],
                int(five > 0) + int(six > 0),
                max(0, 10 - da) + max(0, 10 - db),
                int(not a["connected"]) + int(not b["connected"]),
                max(ta[0], tb[0]), ta[1] + tb[1],
                a["short"] + b["short"],
                -min(da, db, 12),
                max(cx / len(hx), cz / len(hz)), cx + cz,
                a["six"] + b["six"], five + six,
            )
            options.append((key, A, B, hx, hz, n))
    options.sort(key=lambda item: item[0])

    try:
        import numpy as np
    except ImportError:
        np = None

    def css_probe(checks, stabilizers, n, basis):
        """Complete-CSS, stabilizer-aware ambiguity surrogate.

        Uses noiseless syndromes and normalized min-sum, NOT the
        evaluator's ASC circuit or relay-BP cascade. Opposite-check
        subsets model possible correlated ancilla spread without
        claiming knowledge of the measured schedule's hook ordering.
        """
        if np is None:
            return (0, 0, 0)

        shots = 128
        srng = random.Random(870127 + 1009 * idx + 7919 * basis)
        check_cols = columns(checks, n)
        stab_cols = columns(stabilizers, n)
        degrees = np.asarray([
            x.bit_count() + z.bit_count()
            for x, z in zip(check_cols, stab_cols)
        ], dtype=np.float32)
        probabilities = np.clip(0.0023 * degrees, 0.012, 0.035)
        prior = np.log((1.0 - probabilities) / probabilities)[:, None]

        errors = np.zeros((n, shots), dtype=np.uint8)
        for shot in range(shots):
            scale = 0.8 if shot < 40 else (1.4 if shot < 80 else 0.35)
            for c in range(n):
                errors[c, shot] = int(
                    srng.random() < float(probabilities[c]) * scale)
            if shot >= 80:
                row = stabilizers[srng.randrange(len(stabilizers))]
                neighbors = []
                while row:
                    bit = row & -row
                    neighbors.append(bit.bit_length() - 1)
                    row ^= bit
                srng.shuffle(neighbors)
                weight = min(len(neighbors), 2 + (shot - 80) % 3)
                for c in neighbors[:weight]:
                    errors[c, shot] ^= 1

        edge_vars, row_edges = [], []
        var_edges = [[] for _ in range(n)]
        for row in checks:
            edges = []
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                row ^= bit
                edge = len(edge_vars)
                edge_vars.append(c)
                edges.append(edge)
                var_edges[c].append(edge)
            row_edges.append(edges)

        nedges = len(edge_vars)
        variables = np.asarray(edge_vars, dtype=np.intp)
        rd = max(map(len, row_edges))
        vd = max(map(len, var_edges))
        rmap = np.full((len(checks), rd), nedges, dtype=np.intp)
        cmap = np.full((len(checks), rd), n, dtype=np.intp)
        vmap = np.full((n, vd), nedges, dtype=np.intp)
        for r, edges in enumerate(row_edges):
            rmap[r, :len(edges)] = edges
            cmap[r, :len(edges)] = variables[edges]
        for c, edges in enumerate(var_edges):
            vmap[c, :len(edges)] = edges
        valid = rmap != nedges

        padded_errors = np.vstack((
            errors, np.zeros((1, shots), dtype=np.uint8)))
        target = np.bitwise_xor.reduce(padded_errors[cmap], axis=1)
        target_sign = np.where(target != 0, -1.0, 1.0).astype(np.float32)
        incoming = np.empty((nedges + 1, shots), dtype=np.float32)
        incoming[:nedges] = prior[variables]
        incoming[nedges] = 30.0
        messages = np.zeros_like(incoming)
        final = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        effort = 0
        slots = np.arange(rd)[None, :, None]

        for iteration in range(48):
            effort += int(active.sum())
            gathered = incoming[rmap]
            signs = np.where(gathered >= 0, 1.0, -1.0).astype(np.float32)
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
                padded = np.vstack((
                    last, np.zeros((1, shots), dtype=np.uint8)))
                syndrome = np.bitwise_xor.reduce(padded[cmap], axis=1)
                settled = np.all(syndrome == target, axis=0) & active
                final[:, settled] = last[:, settled]
                active[settled] = False
                if not active.any():
                    break

            extrinsic = posterior[variables] - messages[:nedges]
            incoming[:nedges] = (
                0.20 * incoming[:nedges]
                + 0.80 * np.clip(extrinsic, -20.0, 20.0))
            incoming[nedges] = 30.0

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        failed = int(active.sum())
        for shot in range(shots):
            if active[shot]:
                continue
            difference = np.flatnonzero(final[:, shot] ^ errors[:, shot])
            word = sum(1 << int(c) for c in difference)
            # A syndrome-correct answer differing by a stabilizer is
            # successful. Do not count quantum degeneracy as failure.
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    failed += 1
                    break
                word ^= old
        return failed, int(active.sum()), effort

    fallback = None
    finalists = []
    used_a, used_b = {}, {}
    expected_cnots = 6 * q * (
        7 * sum(aw) + 3 * sum(map(sum, sb)))
    for key, A, B, hx, hz, n in options:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or k != 144:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        candidate = {
            "idx": idx, "contract": (architecture, mode, batch),
            "k": k, "cnots": expected_cnots,
            "mixed_sites": tuple(sorted(mixed_sites)),
            "cells": {"A": A, "B": B},
        }
        if fallback is None:
            fallback = candidate

        asign = tuple(tuple(sorted(G.idx(g) for g in cell))
                      for row in A for cell in row)
        bsign = tuple(tuple(sorted(G.idx(g) for g in cell))
                      for row in B for cell in row)
        # Compare more than variations of just one component.
        if used_a.get(asign, 0) >= 2 or used_b.get(bsign, 0) >= 2:
            continue
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        used_a[asign] = used_a.get(asign, 0) + 1
        used_b[bsign] = used_b.get(bsign, 0) + 1
        finalists.append((key, candidate, hx, hz))
        if len(finalists) == 4:
            break

    winner = None
    for structural, candidate, hx, hz in finalists:
        pz = css_probe(hx, hz, 576, 0)
        px = css_probe(hz, hx, 576, 1)
        screen = (
            max(pz[0], px[0]), pz[0] + px[0],
            max(pz[1], px[1]), pz[1] + px[1],
            pz[2] + px[2],
        ) + structural
        candidate["css_probe"] = (pz, px)
        candidate["screen"] = screen
        if winner is None or screen < winner[0]:
            winner = screen, candidate
    if winner is not None:
        return winner[1]
    if fallback is not None:
        return fallback
    raise ValueError("no exact-bound filling passed verified rank screen")


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh copy of the selected complete cell, including zeros."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Smaller ASC depth alone ensures better balanced memory: current slot42 at depth14 failed 91 X and 11 Z shots, while slot5 at depth15 failed 26 X and 17 Z shots, each out of 200000.
# REFUTED: Equal n,k,w,d2s,depth and CNOT count determine circuit reliability: parent slots14/50 both had 576,144,12,12,15,27072, but failed 15/26 X and 4/12 Z shots per 200000.
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
