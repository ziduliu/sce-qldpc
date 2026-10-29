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
CONCEPT = ("For binary octahedral 2O, order 48, A=(1,2), B=(2,5), reinforced B components with unit-triple cells and selectively stronger A columns should reduce collective worst-basis failures and improve circuit fitness above 6.6606; measured pL can refute this.")
GROUP = ("Frozen byte-for-byte under L1. Keep the determinant-cocycle twist of GL(2,3), with its unique involution and normal Q8 subgroup. Odd Q8-supported triples, and triples projecting to one monomial modulo Q8, are units. This exact bed scored 6.6606; related GL(2,3) scored 6.7781. These readings motivate another filling comparison, not a group-only advantage.")
SHAPES = ("Frozen (1,2)x(2,5) for cell 0.250: q48 gives n576, kb144 and guaranteed rate0.250; the size factor permits q<=49. Components have 96 variables/48 checks and 240 variables/96 checks. Current slot18 failed 56 X versus six Z shots, while slot12 failed seven X versus nine Z, each of 200000. Failure maps mildly implicate both A columns and Bcol0/3/4, with generally weak Acol/Brow check enrichment rather than one persistent hotspot. Actual complete ranks retain 144<=k<=158, entirely within this rate cell.")
FILLING = ("Twenty contracts cross four new support profiles with five order policies: A weights3/3 with B totals17 or18, A3/4 with B16, and A4/4 with B14. B triples are units, including pure triple columns; the quartet profile has one split degree-two B column and maximum CSS weight11. Other profiles have maximum weight12. Six-round CNOT budgets are26784,27648,27936,28224: reinforcement is not claimed to reduce all gate exposure. Broaden A partner triangles to include nonunit mixed-quotient geometries, protected by a unit first trinomial where present; even quartets instead require component rank47. Verified singleton actions, short-word screens and complete ranks shortlist fresh fillings. Compare four finalists using a shared-noise 48-shot six-round normalized-min-sum surrogate with measurement errors, opposite-check pairs and ideal final syndrome closure; classify final residuals modulo actual stabilizers. It omits ASC ordering and relay-BP/LSD and is not calibrated pL. Target current X decoder fall-throughs and collective flips: slot18 had only18 found minimum-weight X operators yet56 X failures, versus slot12's at least300 operators and seven failures.")
FAMILY_NAME = "r025_binary_octahedral_q48_12x25_unittriple_reinforced_memoryprobe_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("binary octahedral twist requires order 48")
    # Invertible row-major matrices over F3, with a central cocycle.
    # g*h = (-I)^(epsilon(g)*epsilon(h)) times the usual matrix product,
    # where epsilon is the determinant-sign homomorphism to C2.
    # The cocycle is central and satisfies the cocycle identity.
    els = [(a, b, c, d)
           for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d - b*c) % 3]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 3 for x in g)

    def determinant(g):
        return (g[0]*g[3] - g[1]*g[2]) % 3

    def mul(g, h):
        a, b, c, d = red(g)
        e, f, k, l = red(h)
        sign = 2 if ((a*d-b*c) % 3 == 2
                     and (e*l-f*k) % 3 == 2) else 1
        return ((sign*(a*e+b*k)) % 3,
                (sign*(a*f+b*l)) % 3,
                (sign*(c*e+d*k)) % 3,
                (sign*(c*f+d*l)) % 3)

    def inv(g):
        a, b, c, d = red(g)
        det = (a*d-b*c) % 3
        # Multiply the ordinary inverse by -I in the odd coset.
        factor = (det * (2 if det == 2 else 1)) % 3
        return ((factor*d) % 3, (-factor*b) % 3,
                (-factor*c) % 3, (factor*a) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
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
    """Four exposure/protection architectures times five order policies."""
    import math
    from itertools import combinations

    design, batch = idx % 20, idx // 20
    architecture, mode = divmod(design, 5)
    contracts = (
        ((3, 3), ((1, 2), (1, 2), (2, 1), (2, 2), (3, 1))),
        ((3, 3), ((0, 3), (1, 3), (2, 2), (3, 0), (3, 1))),
        ((3, 4), ((0, 3), (1, 2), (1, 2), (3, 0), (3, 1))),
        ((4, 4), ((0, 3), (1, 1), (1, 2), (2, 1), (3, 0))),
    )
    aw, profile = contracts[architecture]
    sa = [list(aw)]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert sum(map(sum, sb)) == (17, 18, 16, 14)[architecture]
    assert CELL_MAX >= max(max(aw), 3)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    q = G.q
    mask = (1 << q) - 1
    elements = list(G.elements)
    assert q == 48

    def determinant(g):
        return (g[0]*g[3] - g[1]*g[2]) % 3

    def element_order(g):
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if G._key(value) == G._key(G.e):
                return order
        raise ValueError("validated group has no element order")

    orders = {g: element_order(g) for g in elements}
    by_order = {
        order: [g for g in elements if orders[g] == order]
        for order in (4, 6, 8)
    }
    central = (2, 0, 0, 2)
    normal_four = [
        g for g in by_order[4] if determinant(g) == 1
    ]
    quaternion = set([G.e, central] + normal_four)
    assert len(quaternion) == 8
    assert all(G.mul(g, h) in quaternion
               for g in quaternion for h in quaternion)
    assert all(G.mul(G.mul(g, h), G.inv(g)) in quaternion
               for g in elements for h in quaternion)
    # This invariant also distinguishes the group from the parent
    # and from untwisted GL(2,3), rather than merely changing encoding.
    assert sum(order == 2 for order in orders.values()) == 1

    def relative_order(g, h):
        return orders[G.mul(G.inv(g), h)]

    # The first two terms lie in normal Q8. Their images cancel in
    # F2[G/Q8], leaving the third monomial. These trinomials are units.
    # Require noncommutation explicitly, not from element orders alone.
    triangle_books = {}
    for v_order, edge_order in ((8, 4), (4, 8), (4, 4), (6, 6)):
        triangle_books[(v_order, edge_order)] = [
            (u, v)
            for u in normal_four for v in by_order[v_order]
            if v not in quaternion
            and relative_order(u, v) == edge_order
            and G.mul(u, v) != G.mul(v, u)
        ]
    assert all(triangle_books.values())
    first_cards = ((8, 4), (4, 8), (4, 4), (8, 4), (4, 4))
    first_card = first_cards[(mode + batch) % len(first_cards)]
    first_book = triangle_books[first_card]

    q8_pairs = [
        (u, v) for u in normal_four for v in normal_four
        if relative_order(u, v) == 4
        and G.mul(u, v) != G.mul(v, u)
    ]
    assert q8_pairs

    # Each B triple is an actual group-algebra unit. The Q8-contained
    # family has odd augmentation; the other families project to a
    # monomial modulo normal Q8. No unit claim is made for quartets.
    unit_books = (
        triangle_books[(6, 6)],
        triangle_books[(8, 4)],
        triangle_books[(4, 8)],
        triangle_books[(4, 4)],
        q8_pairs,
    )
    mixed_pairs = [
        (u, v) for u in elements for v in elements
        if u not in quaternion and v not in quaternion
        and orders[u] in (3, 4, 6, 8)
        and orders[v] in (3, 4, 6, 8)
        and relative_order(u, v) in (3, 4, 6, 8)
        and G.mul(u, v) != G.mul(v, u)
        and 3 in (orders[u], orders[v], relative_order(u, v))
    ]
    assert mixed_pairs
    if architecture == 0:
        partner_pairs = mixed_pairs
    elif architecture == 1:
        partner_pairs = (
            triangle_books[(8, 4)] + triangle_books[(4, 8)]
            + triangle_books[(4, 4)]
        )
    else:
        partner_pairs = triangle_books[(6, 6)] + mixed_pairs

    partner_book = []
    for u, v in partner_pairs:
        extensions = tuple(
            w for w in elements
            if all(relative_order(t, w) in (3, 4, 6, 8)
                   for t in (G.e, u, v))
        ) if aw[1] == 4 else ()
        if aw[1] == 3 or extensions:
            partner_book.append((u, v, extensions))
    assert partner_book

    first_extensions = {
        (u, v): tuple(
            w for w in elements
            if all(relative_order(t, w) in (3, 4, 6, 8)
                   for t in (G.e, u, v))
        )
        for u, v in first_book
    } if aw[0] == 4 else {}
    if aw[0] == 4:
        first_book = [
            pair for pair in first_book if first_extensions[pair]
        ]
        assert first_book

    # Every portfolio includes all three orders. Their five-entry
    # order multisets differ; they are not uniform-palette redraws.
    portfolios = (
        (8, 8, 8, 6, 4),
        (6, 6, 6, 8, 4),
        (4, 4, 4, 6, 8),
        (8, 8, 6, 6, 4),
        (8, 6, 6, 4, 4),
    )
    portfolio = portfolios[mode]
    if batch % 3 == 1:
        portfolio = portfolio[2:] + portfolio[:2]
    elif batch % 3 == 2:
        portfolio = tuple(reversed(portfolio))
    binomial_sites = [
        (i, j) for i in range(2) for j in range(5)
        if sb[i][j] == 2
    ]
    binomial_number = {site: t for t, site in enumerate(binomial_sites)}
    four_pools = {
        sign: [g for g in by_order[4] if determinant(g) == sign]
        for sign in (1, 2)
    }
    assert all(four_pools.values())

    wx = sum(aw) + max(sum(row[j] for row in sb) for j in range(5))
    wz = max(map(sum, sb)) + max(aw)
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)

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
        # Fix a common term, but redraw internal geometry rather than
        # submitting translations of an already selected component.
        if j == 0:
            u, v = rng.choice(first_book)
            out = [G.e, u, v]
            if aw[j] == 4:
                out.append(rng.choice(first_extensions[u, v]))
        else:
            u, v, extensions = rng.choice(partner_book)
            out = [G.e, u, v]
            if aw[j] == 4:
                out.append(rng.choice(extensions))
        assert len(out) == aw[j]
        assert len({G._key(g) for g in out}) == len(out)
        return out

    def b_cell(i, j):
        count = sb[i][j]
        if count == 0:
            return []
        base = G.pick(rng)
        out = [base]
        if count == 2:
            number = binomial_number[(i, j)]
            order = portfolio[(number + batch) % len(portfolio)]
            if order == 4:
                sign = 1 if (number + mode + batch) % 2 else 2
                relative = rng.choice(four_pools[sign])
            else:
                relative = rng.choice(by_order[order])
            out.append(G.mul(base, relative))
            assert relative_order(out[0], out[1]) == order
        elif count == 3:
            # Change triple quotient geometry across the five policies,
            # including when a profile has only two binomial positions.
            book = unit_books[(mode + 2*i + j + batch) % 5]
            u, v = rng.choice(book)
            out.extend((G.mul(base, u), G.mul(base, v)))
        assert len(out) == count
        assert len({G._key(g) for g in out}) == len(out)
        return out

    def component_pool(side):
        width = (2 if side == "A" else 5) * q
        pool, seen = [], set()

        def draw():
            if side == "A":
                return [[a_cell(j) for j in range(2)]]
            return [[b_cell(i, j) for j in range(5)]
                    for i in range(2)]

        def analyze(grid):
            rows = component(grid, side)
            four, six, connected = cycle_cost(rows, width, True)
            deficit = len(rows) - gf2_rank(rows)
            expected = int(side == "A" and aw == (4, 4))
            energy = ((200 * four + six) / q
                      + 24 * abs(deficit - expected)
                      + 40 * int(not connected))
            return energy, rows, four, six, connected, deficit

        def offer(grid, state):
            _, rows, four, six, connected, deficit = state
            expected = int(side == "A" and aw == (4, 4))
            # Two even A cells cannot have rank48: retain rank47,
            # and let the complete verified CSS ranks determine k.
            if deficit != expected:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            if signature in seen:
                return
            seen.add(signature)
            key = (not connected, four, six)
            pool.append((key, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[0])
            del pool[8:]

        for restart in range(6):
            grid = draw()
            current = analyze(grid)
            offer(grid, current)
            steps = 240 + 20 * (batch % 3)
            for step in range(steps):
                i = 0 if side == "A" else rng.randrange(2)
                j = rng.randrange(2 if side == "A" else 5)
                old = grid[i][j]
                if side == "B" and rng.random() < 0.35:
                    # Preserve internal ratio orders while changing
                    # inter-cell voltages on the two-row component.
                    shift = G.pick(rng)
                    new = [G.mul(shift, g) for g in old]
                else:
                    new = a_cell(j) if side == "A" else b_cell(i, j)
                assert len(new) == len(old)
                assert len({G._key(g) for g in new}) == len(new)
                grid[i][j] = new
                proposed = analyze(grid)
                delta = proposed[0] - current[0]
                temperature = 8.0 * (0.983 ** step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if step % 16 == 0:
                        offer(grid, current)
                else:
                    grid[i][j] = old
            offer(grid, current)

        # A heuristic optimum must not void a rank-legal contract.
        if not pool:
            for attempt in range(128):
                grid = draw()
                offer(grid, analyze(grid))
                if len(pool) >= 4:
                    break
        if not pool:
            raise ValueError("no full-row-rank component in contract")

        finalists = []
        for coarse, grid, rows in pool:
            cols = columns(rows, width)
            finalists.append(dict(
                grid=grid, rows=rows, cols=cols,
                bad=int(small_kernel(cols)),
                four=coarse[1], six=coarse[2],
                connected=not coarse[0],
            ))
        finalists.sort(key=lambda e: (
            e["bad"], not e["connected"], e["four"], e["six"]))
        return finalists[:5]

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

    def pair_exposure(checks, other, n):
        """Degree-weighted pair boundaries on the verified full code.

        Shared opposite checks indicate possible correlated-fault
        opportunities, not hooks certified for the ASC gate ordering.
        This structural statistic is not a circuit-failure estimate.
        """
        cols = columns(checks, n)
        mate = columns(other, n)
        degrees = [
            a.bit_count() + b.bit_count()
            for a, b in zip(cols, mate)
        ]
        burden = 0
        for i, syndrome in enumerate(cols):
            for j in range(i):
                boundary = (syndrome ^ cols[j]).bit_count()
                if boundary <= 5:
                    shared = (mate[i] & mate[j]).bit_count()
                    burden += (degrees[i] * degrees[j]
                               * (6 - boundary) ** 2 * (1 + shared))
        return burden / len(checks)

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
            ex = pair_exposure(hx, hz, n)
            ez = pair_exposure(hz, hx, n)
            key = (
                a["bad"] + b["bad"],
                int(five > 0) + int(six > 0),
                max(0, 10 - da) + max(0, 10 - db),
                int(not a["connected"]) + int(not b["connected"]),
                # Match coarse protection first; then compare broader
                # fault opportunities rather than uncapped distance.
                max(ex, ez), ex + ez,
                max(ta[0], tb[0]), ta[1] + tb[1],
                max(cx / len(hx), cz / len(hz)), cx + cz,
                -min(da, db, 12),
                a["short"] + b["short"],
                a["six"] + b["six"], five + six,
            )
            options.append((key, A, B, hx, hz, n))
    options.sort(key=lambda item: item[0])

    verified = []
    for key, A, B, hx, hz, n in options:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 576 and 144 <= k <= 158):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        expected_cnots = 6*q*(
            7*sum(aw) + 3*sum(map(sum, sb)))
        assert 6*sum(row.bit_count() for row in hx + hz) == expected_cnots
        verified.append((key, A, B, hx, hz, n, k))
    if not verified:
        raise ValueError("no reinforced filling retained cell0.250")

    # Cover different component choices in the small joint shortlist.
    # These are fresh optimized components, not relabeled saved codes.
    def grid_key(grid):
        return tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in grid for cell in row
        )

    chosen = [verified[0]]
    seen_a = {grid_key(verified[0][1])}
    seen_b = {grid_key(verified[0][2])}
    for coordinate, seen in ((1, seen_a), (2, seen_b)):
        for entry in verified[1:]:
            signature = grid_key(entry[coordinate])
            if signature not in seen:
                if not any(entry is old for old in chosen):
                    chosen.append(entry)
                    seen_a.add(grid_key(entry[1]))
                    seen_b.add(grid_key(entry[2]))
                break
    for entry in verified:
        if len(chosen) >= 4:
            break
        if not any(entry is old for old in chosen):
            chosen.append(entry)

    safe = [
        entry for entry in chosen
        if not _design_small_logical(entry[3], entry[4], entry[5])
        and not _design_small_logical(entry[4], entry[3], entry[5])
    ]
    prepared = safe if safe else chosen

    try:
        import numpy as np
    except ImportError:
        np = None

    def memory_probe(checks, stabilizers, basis):
        """Synthetic six-round memory, NOT the ASC/relay-BP experiment.

        Verified checks define the spacetime syndrome-difference matrix.
        Data increments and measurement errors are independent channel
        terms, with additional unscheduled opposite-check pair faults.
        An ideal closing syndrome gives the final temporal boundary.
        Normalized min-sum residuals are classified modulo stabilizers.
        """
        if np is None:
            return (0, 0, 0)
        rounds, n, shots = 6, 576, 48
        m = len(checks)
        nv = rounds*(n + m)
        srng = np.random.default_rng(481027 + 65537*idx + 4099*basis)

        data_degree = np.zeros(n, dtype=np.float32)
        check_sites = []
        for row in checks:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length() - 1)
                row ^= bit
            check_sites.append(sites)
            data_degree[sites] += 1

        mate_sites = []
        for row in stabilizers:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length() - 1)
                row ^= bit
            mate_sites.append(sites)
            data_degree[sites] += 1

        # Common random uniforms compare different finalists in a slot.
        # Degree-scaled probabilities penalize extra gate opportunities,
        # but are a deliberately synthetic channel, not fitted ASC noise.
        level = np.where(np.arange(shots) < 32, 0.014, 0.022)
        data_p = np.clip(
            0.001 + data_degree[:, None]*level[None, :]/8.0,
            0.001, 0.10)
        row_degree = np.asarray(
            [len(sites) for sites in check_sites], dtype=np.float32)
        meas_level = np.where(np.arange(shots) < 32, 0.006, 0.010)
        meas_p = np.clip(
            row_degree[:, None]*meas_level[None, :]/10.0,
            0.001, 0.10)
        probabilities = np.concatenate((
            np.tile(data_p, (rounds, 1)),
            np.tile(meas_p, (rounds, 1))), axis=0).astype(np.float32)
        errors = np.zeros((nv + 1, shots), dtype=np.uint8)
        errors[:nv] = srng.random((nv, shots)) < probabilities

        pair_rng = random.Random(482011 + 65537*idx + 4099*basis)
        for t in range(rounds):
            for shot in range(shots):
                chance = 0.0015 if shot < 32 else 0.003
                for sites in mate_sites:
                    if pair_rng.random() < chance:
                        a, b = pair_rng.sample(sites, 2)
                        errors[t*n + a, shot] ^= 1
                        errors[t*n + b, shot] ^= 1

        # At time t, a measurement error appears in differences t,t+1.
        # The last difference contains only the final measurement error;
        # data readout supplies a noiseless final syndrome closure.
        rsites = []
        for t in range(rounds + 1):
            for r, sites in enumerate(check_sites):
                row_sites = [t*n + c for c in sites] if t < rounds else []
                if t < rounds:
                    row_sites.append(rounds*n + t*m + r)
                if t:
                    row_sites.append(rounds*n + (t - 1)*m + r)
                rsites.append(row_sites)

        variables = []
        redges = []
        vedges = [[] for _ in range(nv)]
        for sites in rsites:
            edges = []
            for c in sites:
                edge = len(variables)
                variables.append(c)
                edges.append(edge)
                vedges[c].append(edge)
            redges.append(edges)
        ne = len(variables)
        rd = max(map(len, redges))
        vd = max(map(len, vedges))
        rmap = np.full((len(redges), rd), ne, dtype=np.intp)
        cmap = np.full((len(redges), rd), nv, dtype=np.intp)
        vmap = np.full((nv, vd), ne, dtype=np.intp)
        for r, edges in enumerate(redges):
            rmap[r, :len(edges)] = edges
            cmap[r, :len(edges)] = rsites[r]
        for c, edges in enumerate(vedges):
            vmap[c, :len(edges)] = edges
        valid = rmap != ne
        variables = np.asarray(variables, dtype=np.intp)

        def syndrome(bits):
            return np.bitwise_xor.reduce(bits[cmap], axis=1)

        target = syndrome(errors)
        target_sign = np.where(target, -1.0, 1.0).astype(np.float32)
        prior = np.log((1.0 - probabilities)/probabilities)
        incoming = np.zeros((ne + 1, shots), dtype=np.float32)
        incoming[:ne] = prior[variables]
        incoming[ne] = 40.0
        messages = np.zeros_like(incoming)
        final = np.zeros_like(errors)
        hard = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        positions = np.arange(rd)[None, :, None]
        late = 0

        for iteration in range(48):
            gathered = incoming[rmap]
            signs = np.where(gathered < 0, -1.0, 1.0)
            magnitude = np.abs(gathered)
            at = np.argmin(magnitude, axis=1)
            first = np.min(magnitude, axis=1)
            second = np.partition(magnitude, 1, axis=1)[:, 1, :]
            excluded = np.where(
                positions == at[:, None, :],
                second[:, None, :], first[:, None, :])
            parity = np.prod(signs, axis=1)*target_sign
            update = 0.80*excluded*signs*parity[:, None, :]
            messages[rmap[valid]] = update[valid]
            messages[ne] = 0.0
            posterior = prior + messages[vmap].sum(axis=1)
            hard[:nv] = posterior < 0
            if iteration % 4 == 3:
                settled = (
                    np.all(syndrome(hard) == target, axis=0) & active)
                final[:, settled] = hard[:, settled]
                if iteration >= 15:
                    late += int(settled.sum())
                active[settled] = False
                if not active.any():
                    break
            proposal = posterior[variables] - messages[:ne]
            incoming[:ne] = (
                0.25*incoming[:ne]
                + 0.75*np.clip(proposal, -24.0, 24.0))
            incoming[ne] = 40.0
        final[:, active] = hard[:, active]

        # Only the accumulated data residual determines the memory
        # logical class. Measurement-error estimates are not logicals.
        residual = np.bitwise_xor.reduce(
            (errors[:rounds*n] ^ final[:rounds*n]).reshape(
                rounds, n, shots), axis=0)
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        packed = np.packbits(
            residual.T, axis=1, bitorder="little")
        failures = 0
        for shot in range(shots):
            word = int.from_bytes(packed[shot].tobytes(), "little")
            while word:
                old = pivots.get(word.bit_length() - 1)
                if old is None:
                    break
                word ^= old
            failures += int(bool(active[shot]) or bool(word))
        return failures, int(active.sum()), late

    best = None
    for key, A, B, hx, hz, n, k in prepared:
        pz = memory_probe(hx, hz, 0)
        px = memory_probe(hz, hx, 1)
        # Preserve short-word protection and connectivity priorities.
        # Above them, compare synthetic decoding before further static
        # exposure/witness tie-breakers. The evaluator alone supplies pL.
        quality = key[:4] + (
            max(px[0], pz[0]), px[0] + pz[0],
            max(px[1], pz[1]), px[1] + pz[1],
            px[2] + pz[2],
        ) + key[4:]
        candidate = {
            "idx": idx, "contract": (architecture, mode, batch),
            "k": k, "screen": quality,
            "synthetic_memory": {"X": px, "Z": pz},
            "cnots": 6*sum(row.bit_count() for row in hx + hz),
            "cells": {"A": A, "B": B},
        }
        if best is None or quality < best[0]:
            best = quality, candidate
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Emit a fresh copy of the complete algebraically screened cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Smaller ASC depth alone ensures better balanced memory: current slot42 at depth14 failed 91 X and 11 Z shots, while slot5 at depth15 failed 26 X and 17 Z shots, each out of 200000.
# REFUTED: Equal n,k,w,d2s,ASC depth and CNOT count determine circuit reliability: parent slots14/50 both had 576,144,12,12,15,27072, but scored 6.844/6.657 with 15/26 X failures and 4/12 Z failures per 200000 shots.
# REFUTED: Fewer found minimum-weight X operators necessarily improve X memory: current slot18 had18 found weight12 X operators and56 X failures, while slot12 had at least300 such operators and seven X failures, each out of200000 shots.
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
