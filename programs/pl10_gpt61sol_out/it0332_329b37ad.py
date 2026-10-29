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
CONCEPT = ("For C7 semidirect C4 with inversion action, order 28, A=(2,4), B=(2,4), regular doubletons with distinct relative-gap contracts and reduced opposite-check pair ambiguity should improve worse-basis circuit fitness above 7.0617; measured pL can refute this.")
GROUP = ("Frozen byte-for-byte under L1: retain the exact split-metacyclic group m=7,r=4,s=6 and its encoding. This bed already achieves approximately 7.1. Its inversion action satisfies 6^4=1 modulo seven; the controlled comparison changes filling, not group algebra or lift order.")
SHAPES = ("Frozen equal (2,4)x(2,4) components have 112 variables and 56 checks each. At q28, n560 and kb112 give guaranteed rate0.200; the shape permits q<=29. Require actual complete k128, rate0.228571 in cell0.250, rather than accepting arbitrary rank excess. Parent failures implicated several Acol/Bcol blocks, including Bcol3 across slot2's X failures, with mostly mild Acol/Brow check enrichment. The sparse maps do not establish one persistent defect.")
FILLING = ("Regular doubletons throughout give component row degree eight, column degree four, CSS weight twelve and 32256 six-round CNOTs. Twenty deliberate relative-gap contracts come from a deterministic catalogue removing obvious scaling, row/column permutation and component-exchange copies; larger slot budgets extend that catalogue. Rank-two order-four incidence guides the quotient, but helper-built component rank54 and complete k128 are required. Anneal fresh origins and order-four ratios with exact Tanner C4/C6 counts. Screen kernels through six, sampled protection capped at twelve, weighted short-word orbits through fourteen under explicitly verified component symmetries, and anchored triple/four-variable boundaries. Joint opposite-check pair exposure precedes cycle tie-breakers. These pairs indicate possible correlated-fault opportunities, not ASC-certified hooks or calibrated pL. Target fewer collective failures and decoder fall-throughs: parent slot22 had d2s13,s7.071, versus slot4 d2s12,s7.176. Gate count is held fixed, not reduced.")
FAMILY_NAME = "r025_m7r4s6_24x24_regular_gap_orbit_pairambiguity_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 28")
    # Same split-metacyclic construction rule.
    # x^7=y^4=1 and y*x*y^-1=x^6; 6^4=1 modulo seven.
    m, r, s = 7, 4, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def _inactive_parent_ctx(rng, G, idx):
    """Historical parent search; never called on the new shape/group."""
    import math
    from itertools import combinations

    design, batch = idx % 20, idx // 20
    aw = (3, 3) if design < 10 else (3, 4)
    bp = (design // 5) % 2
    mode = design % 5
    b_profiles = (
        ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1)),
        ((1, 2), (1, 2), (2, 1), (2, 1), (2, 2)),
    )
    sa = [list(aw)]
    sb = [[c[i] for c in b_profiles[bp]] for i in range(2)]
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
    tri_order = 4 if mode == 3 else 8
    if batch % 3 == 1:
        tri_order = 12 - tri_order
    palettes = ((24,), (12,), (8,), (24, 8), (24, 12))
    palette = palettes[mode]
    if batch % 3 == 2:
        palette = palette + ((12,) if mode == 2 else (8,))
    twist_book = (
        (1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (1, 0, 0, 1, 0),
        (0, 1, 0, 1, 0),
        (1, 1, 0, 0, 1),
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
        if mode in (2, 4):
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
        long_order = 24 if mode % 2 == 0 else 12
        d = rng.choice(orders[long_order])
        c = 3 * rng.randrange(8) + (-a - (a + d)) % 3
        e = rng.choice(orders[8])
        # One long-order pair and one order-eight pair; every
        # mixed-coset difference has order six, never order two.
        return [(a, b), ((a + d) % 24, b),
                (c, 1 - b), ((c + e) % 24, 1 - b)]

    def b_cell(i, j, cosets):
        a, b = rng.randrange(24), cosets[i][j]
        out = [(a, b)]
        if sb[i][j] == 2:
            order = palette[(j + 2 * i) % len(palette)]
            d = rng.choice(orders[order])
            out.append(((a + d) % 24, b))
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
                max(cx / len(hx), cz / len(hz)), cx + cz,
                -min(da, db, 12),
                max(ta[0], tb[0]), ta[1] + tb[1],
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


def NEW_CTX(rng, G, idx):
    """Even-cell support/order contracts with verified target-band ranks."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 28
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    assert (ra, ca, rb, cb) == (2, 4, 2, 4)
    design, batch = idx % 20, idx // 20
    architecture, mode = divmod(design, 5)

    # The parent's sparse contracts screened at d2s4..9, whereas all
    # measured candidates were regular. Spend every slot on a distinct
    # regular relative geometry, not on zero-block architectures.
    sa = [[2] * 4 for _ in range(2)]
    sb = [[2] * 4 for _ in range(2)]
    supports = {"A": sa, "B": sb}
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert all(sum(row) <= 8 for grid in (sa, sb) for row in grid)
    assert all(sum(column) <= 4
               for grid in (sa, sb) for column in zip(*grid))

    # Actual orders, obtained from the actual group operations.
    by_order = {t: [] for t in (4, 7, 14)}
    for g in G.elements:
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if G._key(value) == G._key(G.e):
                if t in by_order:
                    by_order[t].append(g)
                break
    assert all(by_order.values())

    from itertools import combinations_with_replacement

    def gap_direction(d):
        d %= 7
        return min(d, 7 - d)

    def geometry_key(grids):
        """Conservative relative-gap catalogue key, not a Tanner classifier.

        Even ratios retain their C7 direction and order. Odd ratios
        are recorded only by order: their rotation coordinate can be
        changed by conjugation. Remove independent component scaling
        as well as row/column permutations and component exchange.
        The evaluator still checks arbitrary Tanner equivalence.
        """
        sides = []
        for grid in grids:
            images = []
            for u in range(1, 7):
                columns = []
                for j in range(4):
                    pair = []
                    for i in range(2):
                        order, d = grid[i][j]
                        pair.append((order, gap_direction(u*d))
                                    if order != 4 else (4, 0))
                    columns.append(tuple(pair))
                images.append(tuple(sorted(columns)))
                images.append(tuple(sorted((b, a) for a, b in columns)))
            sides.append(min(images))
        return tuple(sorted(sides))

    # Row briefs specify the number of order-four cells. The first
    # twenty cards cross these briefs and then extend them with fresh
    # relative directions, rather than repeat labeling orbits.
    row_briefs = ((2, 2), (2, 3), (1, 3), (3, 3), (1, 2))
    brief_pairs = list(combinations_with_replacement(range(5), 2))
    crng = random.Random(280420937)
    catalogue, used = [], set()

    def make_geometry(brief):
        grid = []
        odd_masks = []
        for number in row_briefs[brief]:
            odd = set(crng.sample(range(4), number))
            odd_masks.append(sum(1 << j for j in odd))
            directions = iter(crng.sample(range(1, 4), 4-number))
            row = []
            for j in range(4):
                if j in odd:
                    row.append((4, 0))
                else:
                    # No repeated undirected even direction in a row.
                    order = 7 if crng.random() < 0.35 else 14
                    row.append((order, next(directions)))
            grid.append(row)
        if gf2_rank(odd_masks) != 2:
            return None
        return grid

    for attempt in range(50000):
        pa, pb = brief_pairs[(7 * len(catalogue)) % len(brief_pairs)]
        ga, gb = make_geometry(pa), make_geometry(pb)
        if ga is None or gb is None:
            continue
        signature = geometry_key((ga, gb))
        if signature in used:
            continue
        used.add(signature)
        catalogue.append((signature, ga, gb, (pa, pb)))
        if len(catalogue) > idx:
            break
    if len(catalogue) <= idx:
        raise ValueError("distinct regular gap catalogue exhausted")

    geometry_signature, ga, gb, brief = catalogue[idx]
    relative_cards = {"A": ga, "B": gb}
    order_cards = {
        side: {(i, j): grid[i][j][0]
               for i in range(2) for j in range(4)}
        for side, grid in relative_cards.items()
    }
    for grid in relative_cards.values():
        odd_masks = [
            sum((grid[i][j][0] == 4) << j for j in range(4))
            for i in range(2)
        ]
        assert gf2_rank(odd_masks) == 2

    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in hx)
        actions["B"][G._key(g)] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_cell(side, i, j):
        base = G.pick(rng)
        order, direction = relative_cards[side][i][j]
        if order == 4:
            # Change genuine odd-ratio geometry as well as the origin.
            relative = rng.choice(by_order[4])
        else:
            # Preserve the slot's prescribed undirected C7 gap.
            sign = rng.choice((-1, 1))
            relative = ((sign * direction) % 7, 0 if order == 7 else 2)
        cell = [base, G.mul(base, relative)]
        assert len({G._key(g) for g in cell}) == 2
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(4)]
                for i in range(2)]

    def component(grid, side):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows

    def columns(rows):
        out = [0] * (4 * q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def graph_stats(rows):
        """Exact Tanner C4/C6, including repeated-variable corrections."""
        adjacency = [0] * len(rows)
        overlaps = [dict() for _ in rows]
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
                    overlaps[i][j] = overlaps[j][i] = overlap

        six_proxy = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                common = neighbors & adjacency[j] & ((1 << j) - 1)
                ab = overlaps[i][j]
                shared = rows[i] & rows[j]
                while common:
                    third = common & -common
                    h = third.bit_length() - 1
                    ac, bc = overlaps[i][h], overlaps[j][h]
                    reused = (shared & rows[h]).bit_count()
                    six_proxy += (
                        ab*ac*bc - reused*(ab+ac+bc) + 2*reused)
                    common ^= third
                lower ^= bit
        assert six_proxy >= 0

        unseen = (1 << len(rows)) - 1
        parts = 0
        while unseen:
            todo = unseen & -unseen
            unseen ^= todo
            parts += 1
            while todo:
                bit = todo & -todo
                todo ^= bit
                neighbors = adjacency[bit.bit_length() - 1] & unseen
                unseen ^= neighbors
                todo |= neighbors
        return four, six_proxy, parts

    def small_kernel(cols):
        """Exact existence test for nonzero component words through six."""
        seen = {0}
        for syndrome in cols:
            if syndrome in seen:
                return True
            seen.add(syndrome)
        for i, si in enumerate(cols):
            for j in range(i):
                syndrome = si ^ cols[j]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
        for i, si in enumerate(cols):
            for j in range(i):
                pair = si ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return True
                    seen.add(syndrome)
        return False

    def witness_scan(cols, side):
        """Actual sampled kernel words and verified translation orbits.

        Minima are upper bounds. Orbit burdens are sampled ambiguity
        statistics, not exact logical spectra or circuit failure rates.
        """
        srng = random.Random(280913711 + 65537*idx + (side == "B"))
        permutation = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 14 and len(short) < 4096:
                short.add(word)

        for trial in range(72):
            srng.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
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
                    basis.append(word)
                    record(word)
            for sweep in range(2):
                srng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        candidate = word ^ other
                        record(candidate)
                        candidate_weight = candidate.bit_count()
                        if candidate and candidate_weight < weight:
                            word, weight = candidate, candidate_weight
                    basis[i] = word
            if minimum <= 6:
                break

        # These permutations are only candidate symmetries until the
        # entire helper-derived component incidence verifies them.
        permutations = []
        for g in G.elements:
            p = tuple(
                G.idx(G.mul(h, g) if side == "A" else G.mul(g, h))
                for h in G.elements)
            assert len(set(p)) == q
            permutations.append(p)

        def translate(word, p):
            out = 0
            while word:
                bit = word & -word
                c = bit.bit_length() - 1
                out |= 1 << ((c // q)*q + p[c % q])
                word ^= bit
            return out

        for p in permutations:
            for j in range(4):
                for h in range(q):
                    assert cols[j*q+p[h]] == translate(cols[j*q+h], p)

        # Canonicalization works even for non-free support orbits.
        # No division by q or assumption about orbit size is made.
        orbits = {
            min(translate(word, p) for p in permutations)
            for word in short
        }
        low10 = 0
        burden = 0
        loads = [0] * 4
        for word in orbits:
            weight = word.bit_count()
            low10 += int(weight <= 10)
            cost = (16-weight)**2
            burden += cost
            for j in range(4):
                loads[j] += cost * ((word >> (j*q)) & mask).bit_count()
        return minimum, low10, burden, max(loads, default=0)

    def boundaries(cols):
        """Anchored three/four-variable small-syndrome exposure.

        One variable per block anchors the verified transitive action.
        Counts are block exposure statistics, not exact multiplicities
        or certificates of absorbing sets or ASC fault tolerance.
        """
        loads = []
        size = len(cols)
        for anchor in range(0, size, q):
            cost = 0
            ca = cols[anchor]
            allowed = [i for i in range(size) if i != anchor]
            for ai, i in enumerate(allowed):
                pair = ca ^ cols[i]
                for bj in range(ai):
                    j = allowed[bj]
                    triple = pair ^ cols[j]
                    weight = triple.bit_count()
                    if weight <= 6:
                        cost += (8-weight)**2
                    for ck in range(bj):
                        h = allowed[ck]
                        weight = (triple ^ cols[h]).bit_count()
                        if weight <= 6:
                            cost += (8-weight)**2
            loads.append(cost)
        return max(loads), sum(loads)

    def portfolio(side):
        sites = list(order_cards[side])
        pool, seen = [], set()
        expected_deficit = 2

        def analyze(grid):
            rows = component(grid, side)
            four, six, parts = graph_stats(rows)
            energy = (100 * four + six / 8.0) / q + 30 * (parts - 1)
            return energy, rows, four, six, parts

        def retain(grid, result):
            _, rows, four, six, parts = result
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            if signature in seen:
                return
            seen.add(signature)
            deficit = len(rows) - gf2_rank(rows)
            if deficit != expected_deficit:
                return
            coarse = (parts - 1, abs(deficit - expected_deficit),
                      four, six)
            pool.append((coarse, clone(grid), list(rows)))
            pool.sort(key=lambda entry: entry[0])
            del pool[6:]

        for restart in range(4):
            grid = draw_grid(side)
            current = analyze(grid)
            retain(grid, current)
            local_grid, local = clone(grid), current
            for step in range(220 + 20 * (batch % 3)):
                i, j = rng.choice(sites)
                previous = grid[i][j]
                # Redraw the internal ratio as well as its offset:
                # this is not a saved-code gauge transformation.
                grid[i][j] = draw_cell(side, i, j)
                proposed = analyze(grid)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.983 ** step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if current[0] < local[0]:
                        local_grid, local = clone(grid), current
                    if step % 24 == 0:
                        retain(grid, current)
                else:
                    grid[i][j] = previous
            retain(local_grid, local)
            retain(grid, current)

        finalists = []
        for coarse, grid, rows in pool:
            # Confirm cached composition against the verified zero probe.
            if side == "A":
                verified = lifted_matrices(G, grid, [[[]]])[0]
            else:
                verified = lifted_matrices(G, [[[]]], grid)[1]
            width_mask = (1 << (4 * q)) - 1
            assert rows == [row & width_mask for row in verified]
            cols = columns(rows)
            bad = int(small_kernel(cols))
            distance, low10, low12, peak = witness_scan(cols, side)
            boundary_peak, boundary_total = boundaries(cols)
            key = (coarse[0], bad, max(0, 10 - distance),
                   low10, low12, peak, boundary_peak, boundary_total,
                   coarse[2], coarse[3], coarse[1])
            finalists.append(dict(
                grid=grid, rows=rows, key=key, bad=bad,
                parts=coarse[0], d=distance, low10=low10,
                low12=low12, peak=peak, boundary_peak=boundary_peak,
                boundary_total=boundary_total,
                four=coarse[2], six=coarse[3]))
        finalists.sort(key=lambda entry: entry["key"])
        return finalists

    def pair_exposure(checks, opposite, n):
        """Small-syndrome pairs sharing an opposite-basis check.

        An opposite check supplies possible correlated-fault support.
        Without its actual ASC gate ordering these are opportunities,
        not certified hooks. Use only complete verified CSS matrices.
        """
        cols = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                cols[bit.bit_length()-1] |= tag
                row ^= bit
        loads = [0] * (n // q)
        burden = 0
        for row in opposite:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length()-1)
                row ^= bit
            for i, c in enumerate(sites):
                for d in sites[:i]:
                    boundary = (cols[c] ^ cols[d]).bit_count()
                    if boundary <= 8:
                        cost = (10-boundary)**2
                        burden += cost
                        loads[c // q] += cost
                        loads[d // q] += cost
        return burden / len(checks), max(loads, default=0) / q

    apool, bpool = portfolio("A"), portfolio("B")
    proposals = []
    for a in apool:
        for b in bpool:
            da, db = min(a["d"], 12), min(b["d"], 12)
            key = (
                a["parts"] + b["parts"],
                a["bad"] + b["bad"],
                max(0, 10 - da) + max(0, 10 - db),
                max(a["low10"], b["low10"]),
                a["low10"] + b["low10"],
                max(a["low12"], b["low12"]),
                a["low12"] + b["low12"],
                max(a["peak"], b["peak"]),
                max(a["boundary_peak"], b["boundary_peak"]),
                a["boundary_total"] + b["boundary_total"],
                a["four"] + b["four"], a["six"] + b["six"],
                -min(da, db), abs(da - db),
            )
            proposals.append((key, a, b))
    proposals.sort(key=lambda entry: entry[0])

    prepared = []
    for key, a, b in proposals:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 560 or k != 128:
            continue
        ar = [
            sum(((hx[(i * cb) * q + t] >> (j * cb * q)) & mask)
                << (j * q) for j in range(ca))
            for i in range(ra) for t in range(q)
        ]
        br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
        assert ar == a["rows"] and br == b["rows"]
        assert n == 560
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        cx, cz = _design_four_cycles(hx), _design_four_cycles(hz)
        ex, ez = pair_exposure(hx, hz, n), pair_exposure(hz, hx, n)
        # Retain short-word exclusion and a coarse protection floor.
        # Then compare correlated-pair ambiguity before cycle burden.
        quality = key[:3] + (
            max(ex[0], ez[0]), ex[0]+ez[0],
            max(ex[1], ez[1]), ex[1]+ez[1],
        ) + key[3:] + (
            max(cx / len(hx), cz / len(hz)), cx+cz,
        )
        prepared.append((quality, A, B, hx, hz, n, k))
        if len(prepared) == 6:
            break

    # Fresh same-contract rank fallback: never repair emitted cells or
    # assume the augmentation-sector calculation equals the CSS rank.
    if not prepared:
        for trial in range(192):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n == 560 and k == 128:
                prepared.append(((10**12, trial), A, B, hx, hz, n, k))
                break
    if not prepared:
        raise ValueError("even-cell contract missed verified cell 0.250")
    prepared.sort(key=lambda entry: entry[0])

    def emit(entry):
        key, A, B, hx, hz, n, k = entry
        assert css_ok(hx, hz)
        assert n == 560
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        expected = 6 * q * (
            (cb + rb) * sum(map(sum, sa))
            + (ca + ra) * sum(map(sum, sb))
        )
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected
        return {
            "idx": idx, "contract": (brief, geometry_signature),
            "k": k, "screen": key, "cnots": expected,
            "orders": order_cards, "relative_gaps": relative_cards,
            "cells": {"A": A, "B": B},
        }

    for entry in prepared:
        _, _, _, hx, hz, n, _ = entry
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        return emit(entry)
    return emit(prepared[0])


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete gap-contract cell without redraws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Smaller ASC depth alone ensures better balanced memory: current slot42 at depth14 failed 91 X and 11 Z shots, while slot5 at depth15 failed 26 X and 17 Z shots, each out of 200000.
# REFUTED: Equal d2s, k, depth and CNOT count determine circuit reliability: parent slots18/19 both had d2s11,k144,depth15,CNOTs27936, but failed 42/nine X shots per 200000 and scored 6.483/7.005.
# REFUTED: Raising d2s from twelve to thirteen guarantees better memory on this order-28 regular bed: parent slot22 had d2s13,s7.071 and six Z failures/200000, while slot4 had d2s12,s7.176 and four Z failures/200000; both had k128,depth16,CNOTs32256.
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
