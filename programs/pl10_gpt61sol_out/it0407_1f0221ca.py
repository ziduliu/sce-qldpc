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
CONCEPT = ("For cyclic C29, order 29, A=(2,4), B=(2,4), balanced degree-three components with nine-term CSS checks should improve cell 0.200 circuit reliability through reduced extraction exposure and decoding ambiguity; measured pL can refute this.")
GROUP = ("L3 jump from nonabelian C7:C4 to cyclic C29, a genuinely different group. The C29 (2,4)x(2,4) bed reached about 7.2, while C27 mixed singleton/doubleton codes reached about 7.2 in cell 0.200. These readings motivate a fresh geometry-controlled filling, not a group-only performance claim. Order29 is the largest legal lift for size factor twenty, giving n580.")
SHAPES = ("Keep equal (2,4)x(2,4): each component has 116 variables and 58 checks. The guaranteed bound is kb116, rate0.200. Complementary odd/even count rows have full augmentation rank; require actual component ranks58 and complete verified k116, eliminating the parent's deliberate rank excess. Parent failures implicated different Acol/Bcol blocks across slots, including Bcol3 in slot2 and Bcol0 in slot4, with mostly mild Acol/Brow check enrichment. These small samples do not establish a persistent hotspot.")
FILLING = ("All cells are singleton/doubleton with complementary counts [[1,1,2,2],[2,2,1,1]]. Component rows weigh six and columns three, so both CSS check weights are nine. A deterministic catalogue assigns distinct joint relative-gap contracts after conservatively removing independent component multipliers, row/column permutations and component exchange. Fresh inter-row voltages are annealed using helper-derived actions. Exact component exclusion through six, shared-bank normalized-min-sum component probes, capped sampled protection and short-word/block exposure screen alternatives. Complete helper-built ranks, CSS, component equality and legal weights are checked; quantum logical exclusion through four is preferred. Six-round CNOTs25056 replace the parent's measured32256. Target fewer collective failures and cascade fall-throughs: parent slot22 had d2s13 and six Z failures/200000, whereas slot4 had d2s12 and four Z failures. Component probes omit syndrome history, ASC ordering and relay-BP/LSD and are not calibrated circuit pL estimates.")
FAMILY_NAME = "r020_C29_24x24_mixed3_gatelean_gapdecode_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C29 requires order 29")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _retired_even28_ctx(rng, G, idx):
    """Inactive historical order-28 search; never called by this builder."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 28
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    assert (ra, ca, rb, cb) == (2, 4, 2, 4)
    design, batch = idx % 20, idx // 20
    architecture, mode = divmod(design, 5)

    regular = ((2, 2, 2, 2), (2, 2, 2, 2))
    sparse = ((0, 2, 2, 2), (2, 0, 2, 2))
    unequal_rows = ((0, 0, 2, 2), (2, 2, 2, 2))
    contracts = (
        (regular, regular),
        (sparse, regular),
        (sparse, sparse),
        (unequal_rows, regular),
    )
    sa, sb = ([list(row) for row in grid]
              for grid in contracts[architecture])
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

    order_cards = {}
    for side, support in supports.items():
        sites = [(i, j) for i, row in enumerate(support)
                 for j, weight in enumerate(row) if weight]
        degrees = [sum(row[j] for row in support) for j in range(4)]
        weak = {site for site in sites if degrees[site[1]] == 2}
        eligible = [site for site in sites if site not in weak]

        def mask_rank(chosen):
            rows = [
                sum(1 << j for i0, j in chosen if i0 == i)
                for i in range(2)
            ]
            return gf2_rank(rows)

        card = {site: 7 for site in sites}
        if mode == 0:
            # At the C7-trivial sector, order-14 differences give
            # 1+y^2, whereas order-seven differences project to zero.
            # Require independent leading coefficient rows. This is
            # only an architectural rank guide; full ranks are checked.
            number = min(len(sites) - 1,
                         max(len(weak) + 1, 2 + batch % 3))
            book = [
                set(chosen) for chosen in combinations(sites, number)
                if weak.issubset(chosen) and mask_rank(chosen) == 2
            ]
            if not book:
                book = [
                    set(chosen)
                    for number in range(2, len(sites) + 1)
                    for chosen in combinations(sites, number)
                    if weak.issubset(chosen) and mask_rank(chosen) == 2
                ]
            chosen = rng.choice(book)
            for site in chosen:
                card[site] = 14
        else:
            # Order-four ratios have odd y exponent and hence a
            # first-order augmentation zero in F2[C4].
            # Never place them in a degree-two component column:
            # that would create an unavoidable weight-four word.
            wanted = (2, 3, 4, 5)[mode - 1]
            if len(eligible) == 4:
                wanted = 2 if mode in (1, 3) else 3
            wanted = min(wanted, len(eligible))
            book = [
                set(chosen)
                for chosen in combinations(eligible, wanted)
                if mask_rank(chosen) == 2
            ]
            assert book
            odd = rng.choice(book)
            for site in sites:
                if site in odd:
                    card[site] = 4
                elif site in weak:
                    card[site] = 14
                elif mode in (1, 2):
                    card[site] = 14
                else:
                    card[site] = 7

            # Larger harness batches alter the order portfolio itself,
            # not an optimized filling's automorphism/permutation image.
            remaining = [site for site in eligible if site not in odd]
            if batch % 3 == 1:
                for t, site in enumerate(remaining):
                    if t % 2 == 0:
                        card[site] = 7 if mode in (1, 2) else 14
            elif batch % 3 == 2 and remaining:
                site = remaining[(mode + architecture) % len(remaining)]
                card[site] = 7 if card[site] == 14 else 14
        assert all(card[site] == 14 for site in weak)
        order_cards[side] = card

    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in hx)
        actions["B"][G._key(g)] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_cell(side, i, j):
        if supports[side][i][j] == 0:
            return []
        base = G.pick(rng)
        relative = rng.choice(by_order[order_cards[side][i, j]])
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
        """Exact C4 and connectivity; C6 proxy for proposal ordering."""
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
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (
                    neighbors & adjacency[j] & ((1 << j) - 1)
                ).bit_count()
                lower ^= bit
        cols = columns(rows)
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (col.bit_count() for col in cols))
        six_proxy = max(0, triangles - stars)

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
        """Verified sampled kernel words, not a distance certificate."""
        srng = random.Random(912733 + 65537 * idx + (side == "B"))
        permutation = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 12 and len(short) < 4096:
                short.add(word)

        for trial in range(40):
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

        low10 = sum(word.bit_count() <= 10 for word in short)
        loads = [
            sum(bool((word >> (j * q)) & mask) for word in short)
            for j in range(4)
        ]
        return minimum, low10, len(short), max(loads, default=0)

    def boundaries(cols):
        """Orbit-anchored triple exposure, not a trapping-set certificate."""
        loads = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i, si in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ si
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 4:
                        cost += (5 - weight) ** 2
            loads.append(cost)
        return max(loads), sum(loads)

    def portfolio(side):
        sites = list(order_cards[side])
        pool, seen = [], set()
        expected_deficit = 4 if mode == 0 else 2

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
        if not (k > 0 and 225 * n <= 1000 * k < 275 * n):
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
        # A coarse protection floor comes first. Complete-code cycle
        # imbalance is a structural tie-breaker, not measured circuit pL.
        quality = key[:3] + (
            max(cx / len(hx), cz / len(hz)), cx + cz,
        ) + key[3:]
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
            if k > 0 and 225 * n <= 1000 * k < 275 * n:
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
            "idx": idx, "contract": (architecture, mode, batch),
            "k": k, "screen": key, "cnots": expected,
            "orders": order_cards, "cells": {"A": A, "B": B},
        }

    for entry in prepared:
        _, _, _, hx, hz, n, _ = entry
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        return emit(entry)
    return emit(prepared[0])


def NEW_CTX(rng, G, idx):
    """Distinct mixed-degree gap contracts with verified full-rank lifts."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 29 and A_SHAPE == B_SHAPE == (2, 4)
    support = [[1, 1, 2, 2], [2, 2, 1, 1]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) == 9
    assert W_MAX >= 9 and CELL_MAX >= 2
    assert all(sum(row) == 6 for row in support)
    assert all(sum(column) == 3 for column in zip(*support))
    mask = (1 << q) - 1
    width = 4 * q

    def direction(value):
        value %= q
        return min(value, q - value)

    def clean_pair(a, b):
        # Exclude repeated directions and signed three-gap relations
        # within the two doubletons belonging to one component row.
        signed = (a, -a, b, -b)
        return a != b and all(
            (x + y + z) % q
            for x in signed for y in signed for z in signed)

    def side_key(gaps):
        images = []
        for u in range(1, q):
            lower = tuple(sorted(direction(u * d) for d in gaps[:2]))
            upper = tuple(sorted(direction(u * d) for d in gaps[2:]))
            images.append(min((lower, upper), (upper, lower)))
        return min(images)

    # Every slot consumes a new joint gap architecture. Origins are
    # subsequently optimized; no submitted slot is a saved-code image.
    # This key removes obvious gap orbits, not arbitrary Tanner isomorphism.
    crng = random.Random(290603719)
    catalogue, used = [], set()
    for proposal in range(30000):
        ga = crng.sample(range(1, 15), 4)
        gb = crng.sample(range(1, 15), 4)
        if not all(clean_pair(g[0], g[1]) and clean_pair(g[2], g[3])
                   for g in (ga, gb)):
            continue
        ka, kb = side_key(ga), side_key(gb)
        if ka == kb:
            continue
        identity = tuple(sorted((ka, kb)))
        if identity in used:
            continue
        used.add(identity)
        catalogue.append((identity, tuple(ga), tuple(gb)))
        if len(catalogue) > idx:
            break
    if len(catalogue) <= idx:
        raise ValueError("distinct mixed-degree gap catalogue exhausted")
    identity, ga, gb = catalogue[idx]
    gaps = {"A": ga, "B": gb}

    # Cache only singleton blocks returned by the verified helper.
    # No independent left/right multiplication convention is screened.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def draw_cell(side, i, j):
        base = G.pick(rng)
        if support[i][j] == 1:
            return [base]
        return [base, G.mul(base, (gaps[side][j],))]

    def draw(side):
        return [[draw_cell(side, i, j) for j in range(4)]
                for i in range(2)]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows

    def columns(rows):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def cycle_stats(rows):
        """Exact C4 and a proposal-only C6 proxy."""
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
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (
                    neighbors & adjacency[j] & ((1 << j) - 1)
                ).bit_count()
                lower ^= bit
        # Every variable has degree three. Triangle-minus-stars is
        # exact without C4; otherwise it is only a search tie-breaker.
        six = max(0, triangles - width)
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
        energy = (120 * four + six / 6.0) / q + 40 * (parts - 1)
        return energy, four, six, parts

    def short_boundaries(cols):
        """Exact kernel exclusion through six and triple-boundary exposure.

        Distinct subsets of size <=3 exhaust the short-word existence
        test. Boundary loads are heuristics, not absorbing-set certificates.
        """
        seen = {0}
        bad = False
        for syndrome in cols:
            bad |= syndrome in seen
            seen.add(syndrome)
        for i, si in enumerate(cols):
            for j in range(i):
                syndrome = si ^ cols[j]
                bad |= syndrome in seen
                seen.add(syndrome)
        loads = [0] * 4
        total = 0
        for i, si in enumerate(cols):
            for j in range(i):
                pair = si ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    bad |= syndrome in seen
                    seen.add(syndrome)
                    weight = syndrome.bit_count()
                    if weight <= 5:
                        cost = (7 - weight) ** 2
                        total += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[h // q] += cost
        return int(bad), max(loads) / q, total / width

    def sampled_exposure(cols, side):
        """Actual sampled kernel witnesses; minima remain upper bounds."""
        srng = random.Random(290917831 + 1009 * idx + (side == "B"))
        order = list(range(width))
        minimum, short = width + 1, set()

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 14 and len(short) < 2048:
                short.add(word)

        for trial in range(48):
            srng.shuffle(order)
            pivots, basis = {}, []
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
                    basis.append(word)
                    record(word)
            for sweep in range(2):
                srng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        candidate = word ^ other
                        record(candidate)
                        cw = candidate.bit_count()
                        if candidate and cw < weight:
                            word, weight = candidate, cw
                    basis[i] = word
            if minimum <= 6:
                break

        def rotate_blocks(word, shift, blocks):
            return sum(
                (((((word >> (j * q)) & mask) << shift)
                  | (((word >> (j * q)) & mask) >> (q - shift))) & mask)
                << (j * q) for j in range(blocks))

        # Verify the symmetry used only to reduce sampled witness orbits.
        for j in range(4):
            for shift in range(q):
                assert cols[j * q + shift] == rotate_blocks(
                    cols[j * q], shift, 2)
        orbits = {
            min(rotate_blocks(word, shift, 4) for shift in range(q))
            for word in short
        }
        burden = 0
        loads = [0] * 4
        for word in orbits:
            cost = (16 - word.bit_count()) ** 2
            burden += cost
            for j in range(4):
                loads[j] += cost * ((word >> (j * q)) & mask).bit_count()
        return minimum, burden, max(loads), len(orbits)

    try:
        import numpy as np
    except ImportError:
        np = None

    def decoder_probe(rows, side):
        """Fixed-bank classical component benchmark, NOT circuit pL.

        First valid syndrome settlement can be a miscorrection. Count
        both wrong settlements and unresolved patterns. This probe omits
        syndrome history, quantum degeneracy, ASC gates and relay-BP/LSD.
        """
        if np is None:
            return (0, 0, 0)
        shots = 144
        srng = random.Random(290713037 + 1009 * idx + (side == "B"))
        errors = np.zeros((width + 1, shots), dtype=np.uint8)
        for shot in range(shots):
            weight = 3 + shot // 48
            for c in srng.sample(range(width), weight):
                errors[c, shot] = 1

        edge_vars, redges, sites = [], [], []
        vedges = [[] for _ in range(width)]
        for row in rows:
            edges, variables = [], []
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                row ^= bit
                edge = len(edge_vars)
                edge_vars.append(c)
                edges.append(edge)
                variables.append(c)
                vedges[c].append(edge)
            redges.append(edges)
            sites.append(variables)
        ne = len(edge_vars)
        rd, vd = max(map(len, redges)), max(map(len, vedges))
        rmap = np.full((len(rows), rd), ne, dtype=np.intp)
        cmap = np.full((len(rows), rd), width, dtype=np.intp)
        vmap = np.full((width, vd), ne, dtype=np.intp)
        for r, edges in enumerate(redges):
            rmap[r, :len(edges)] = edges
            cmap[r, :len(edges)] = sites[r]
        for c, edges in enumerate(vedges):
            vmap[c, :len(edges)] = edges
        valid = rmap != ne
        edge_vars = np.asarray(edge_vars, dtype=np.intp)

        def syndrome(bits):
            return np.bitwise_xor.reduce(bits[cmap], axis=1)

        target = syndrome(errors)
        target_sign = np.where(target, -1.0, 1.0).astype(np.float32)
        prior = math.log((width - 4.0) / 4.0)
        incoming = np.full((ne + 1, shots), prior, dtype=np.float32)
        incoming[ne] = 30.0
        messages = np.zeros_like(incoming)
        hard = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        slots = np.arange(rd)[None, :, None]
        wrong = late = 0
        for iteration in range(32):
            gathered = incoming[rmap]
            signs = np.where(gathered < 0, -1.0, 1.0)
            magnitude = np.abs(gathered)
            at = np.argmin(magnitude, axis=1)
            first = np.min(magnitude, axis=1)
            second = np.partition(magnitude, 1, axis=1)[:, 1, :]
            excluded = np.where(
                slots == at[:, None, :],
                second[:, None, :], first[:, None, :])
            parity = np.prod(signs, axis=1) * target_sign
            update = 0.80 * excluded * signs * parity[:, None, :]
            messages[rmap[valid]] = update[valid]
            messages[ne] = 0.0
            posterior = prior + messages[vmap].sum(axis=1)
            hard[:width] = posterior < 0
            settled = active & np.all(syndrome(hard) == target, axis=0)
            incorrect = np.any(hard[:width] != errors[:width], axis=0)
            wrong += int(np.count_nonzero(settled & incorrect))
            if iteration >= 12:
                late += int(settled.sum())
            active[settled] = False
            if not active.any():
                break
            proposal = posterior[edge_vars] - messages[:ne]
            incoming[:ne] = (
                0.30 * incoming[:ne]
                + 0.70 * np.clip(proposal, -20.0, 20.0))
            incoming[ne] = 30.0
        remaining = int(active.sum())
        return wrong + remaining, remaining, late

    def portfolio(side):
        pool, seen = [], set()
        serial = 0

        def retain(grid, rows, stats):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            # Full augmentation rank is only a guide; the actual
            # helper-derived component rank is required here.
            if gf2_rank(rows) != 2 * q:
                return
            serial += 1
            coarse = (stats[3] - 1, stats[1], stats[2], serial)
            pool.append((coarse, clone(grid), list(rows)))
            pool.sort(key=lambda entry: entry[0])
            del pool[6:]

        for restart in range(5):
            grid = draw(side)
            rows = component(grid, side)
            current = cycle_stats(rows)
            retain(grid, rows, current)
            local_grid, local_rows, local = clone(grid), list(rows), current
            for step in range(300):
                i, j = rng.randrange(2), rng.randrange(4)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                trial_rows = component(grid, side)
                proposed = cycle_stats(trial_rows)
                delta = proposed[0] - current[0]
                temperature = 10.0 * (0.984 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    rows, current = trial_rows, proposed
                    if current[0] < local[0]:
                        local_grid = clone(grid)
                        local_rows, local = list(rows), current
                    if step % 20 == 0:
                        retain(grid, rows, current)
                else:
                    grid[i][j] = old
            retain(local_grid, local_rows, local)
            retain(grid, rows, current)

        if not pool:
            for trial in range(128):
                grid = draw(side)
                rows = component(grid, side)
                retain(grid, rows, cycle_stats(rows))
                if len(pool) >= 3:
                    break
        if not pool:
            raise ValueError("gap contract missed full component rank")

        finalists = []
        for coarse, grid, rows in pool:
            if side == "A":
                verified = lifted_matrices(G, grid, [[[]]])[0]
            else:
                verified = lifted_matrices(G, [[[]]], grid)[1]
            assert rows == [row & ((1 << width) - 1) for row in verified]
            cols = columns(rows)
            assert all(row.bit_count() == 6 for row in rows)
            assert all(col.bit_count() == 3 for col in cols)
            bad, boundary_peak, boundary_total = short_boundaries(cols)
            distance, burden, peak, multiplicity = sampled_exposure(cols, side)
            bp = decoder_probe(rows, side)
            key = (
                bad, coarse[0], max(0, 10 - distance),
                bp[0], bp[1], bp[2],
                burden, peak, boundary_peak, boundary_total,
                coarse[1], coarse[2], -min(distance, 12))
            finalists.append(dict(
                grid=grid, rows=rows, key=key, bad=bad,
                parts=coarse[0], d=distance, bp=bp,
                burden=burden, peak=peak, multiplicity=multiplicity,
                boundary_peak=boundary_peak, boundary_total=boundary_total,
                four=coarse[1], six=coarse[2]))
        finalists.sort(key=lambda entry: entry["key"])
        return finalists[:4]

    apool, bpool = portfolio("A"), portfolio("B")
    proposals = []
    for a in apool:
        for b in bpool:
            floor_a, floor_b = max(0, 10 - a["d"]), max(0, 10 - b["d"])
            key = (
                max(a["bad"], b["bad"]), a["bad"] + b["bad"],
                a["parts"] + b["parts"],
                max(floor_a, floor_b), floor_a + floor_b,
                max(a["bp"][0], b["bp"][0]), a["bp"][0] + b["bp"][0],
                max(a["bp"][1], b["bp"][1]), a["bp"][1] + b["bp"][1],
                a["bp"][2] + b["bp"][2],
                max(a["burden"], b["burden"]), a["burden"] + b["burden"],
                max(a["peak"], b["peak"]),
                max(a["boundary_peak"], b["boundary_peak"]),
                a["boundary_total"] + b["boundary_total"],
                a["four"] + b["four"], a["six"] + b["six"],
                -min(a["d"], b["d"], 12))
            proposals.append((key, a, b))
    proposals.sort(key=lambda entry: entry[0])

    prepared = []
    for key, a, b in proposals[:12]:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or k != 116:
            continue
        ar = [
            sum(((hx[(i * 4) * q + t] >> (j * 4 * q)) & mask)
                << (j * q) for j in range(4))
            for i in range(2) for t in range(q)
        ]
        br = [row & ((1 << width) - 1) for row in hz[:2 * q]]
        assert ar == a["rows"] and br == b["rows"]
        assert all(row.bit_count() == 9 for row in hx + hz)
        cx, cz = _design_four_cycles(hx), _design_four_cycles(hz)
        quality = key + (
            max(cx / len(hx), cz / len(hz)), cx + cz)
        prepared.append((quality, A, B, hx, hz, a["bp"], b["bp"]))
    if not prepared:
        raise ValueError("no complete verified k116 mixed-degree pair")
    prepared.sort(key=lambda entry: entry[0])

    def emit(entry):
        key, A, B, hx, hz, ap, bp = entry
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert 6 * sum(row.bit_count() for row in hx + hz) == 25056
        assert all(
            len(cell) == support[i][j]
            and len({G._key(g) for g in cell}) == len(cell)
            for grid in (A, B)
            for i, row in enumerate(grid) for j, cell in enumerate(row))
        return {
            "idx": idx, "contract": identity,
            "relative_gaps": gaps, "k": 116, "cnots": 25056,
            "component_probe": (ap, bp), "screen": key,
            "cells": {"A": A, "B": B},
        }

    # A finite structural search is not permitted to void otherwise
    # legal slots merely because its preferred short-logical gate misses.
    for entry in prepared:
        _, _, _, hx, hz, _, _ = entry
        if _design_small_logical(hx, hz, 580):
            continue
        if _design_small_logical(hz, hx, 580):
            continue
        return emit(entry)
    return emit(prepared[0])


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected mixed-degree cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Smaller ASC depth alone ensures better balanced memory: current slot42 at depth14 failed 91 X and 11 Z shots, while slot5 at depth15 failed 26 X and 17 Z shots, each out of 200000.
# REFUTED: Equal d2s, k, depth and CNOT count determine circuit reliability: parent slots18/19 both had d2s11,k144,depth15,CNOTs27936, but failed 42/nine X shots per 200000 and scored 6.483/7.005.
# REFUTED: Raising d2s from twelve to thirteen guarantees better memory on the parent's order-28 regular bed: slot22 had d2s13,s7.071 and six Z failures/200000, while slot4 had d2s12,s7.176 and four Z failures/200000; both had k128,depth16,CNOTs32256.
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
