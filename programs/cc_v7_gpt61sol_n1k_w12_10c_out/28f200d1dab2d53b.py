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
CONCEPT = ("On the modular group C16 semidirect_9 C2, order 32, A=(2,7), B=(2,6), mixed-column A and exclusively cubic B components with witness-directed repair should yield five new distance readings >=8 and improve the 0.450 incumbent's fitness 21.304. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from Frobenius21 x C4 to the nonabelian modular group of order 32: b*a*b^-1=a^9 and 9^2=1 mod 16. Odd rotation exponents provide order-sixteen elements, including outside C16. The cyclic q=32 bed on these shapes reached five readings of seven; noncommuting voltages test whether that plateau can be broken. This group-and-shape bed is absent from BEDS.")
SHAPES = ("A=(2,7), B=(2,6), q=32: n=1472, k>=640, guaranteed rate 20/46=0.434783 in cell 0.450. Components have 224 and 192 variables with 64 checks each. Parent weight-six minima were entirely sector one, especially Acol1/2 across Bcol0/1/2. Replace that one-row backbone; retain only verified full-row-rank components and exact full-code rates in [0.425,0.475).")
FILLING = ("Twenty cards cross five B cubic-concentration profiles with four A support/order models. A has three degree-two and four degree-three columns; B has six degree-three columns. Every component row has nine terms, so both CSS caps are 9+3=12. Cache only scaffold-built monomial blocks, optimize actual component four-cycle/projection geometry, exclude component kernels through five when feasible, and repair verified words found by bounded parity completion. Independent elimination searches challenge retained candidates. Full CSS, ranks, weights and rate gate final cells. A budget-exhausted search is not an exclusion certificate, and classical component readings do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r045_Modular32_27x26_cubicB_witness8_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # Modular group: a^16=b^2=1, b*a*b^-1=a^9.
    # (x,y) denotes a^x b^y; 9^2=1 modulo sixteen.
    if not valid_q(q):
        raise ValueError("C16:C2(action 9) requires order 32")
    action = (1, 9)
    els = [(x, y) for y in range(2) for x in range(16)]

    def red(g):
        return (g[0] % 16, g[1] % 2)

    def mul(g, h):
        x, y = red(g)
        u, v = red(h)
        return ((x + action[y]*u) % 16, (y+v) % 2)

    def inv(g):
        x, y = red(g)
        return ((-action[y]*x) % 16, y)

    def idx(g):
        x, y = red(g)
        return 16*y+x

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


# Six triple columns, written column-major. Every profile sums to
# (7,5,6), completing the anchored skeleton's row totals to (9,9,9).
_ANCHOR_TRIPLES = (
    ((2,0,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
    ((2,1,0), (1,0,2), (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
    ((2,0,1), (1,2,0), (1,0,2), (1,1,1), (1,1,1), (1,1,1)),
    ((2,0,1), (2,1,0), (0,2,1), (1,0,2), (1,1,1), (1,1,1)),
    ((2,0,1), (2,0,1), (1,2,0), (1,1,1), (1,1,1), (0,1,2)),
)
_ANCHOR_FRAMES = ((0,1,3), (0,1,4), (0,1,5), (0,2,5))


def _anchor_columns(G, cells, side):
    """Obtain component columns only through the verified lift helper.

    The empty dummy side contributes no sector-two entries. Its dead
    lines are used only in this extraction call, never in a submitted code.
    """
    width = len(cells[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, cells, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], cells)
    mask = (1 << width) - 1
    cols = [0] * width
    for r, raw in enumerate(rows):
        raw &= mask
        while raw:
            bit = raw & -raw
            cols[bit.bit_length() - 1] |= 1 << r
            raw ^= bit
    return cols


def _anchor_pairs(cols):
    """Exact exclusion of all component kernels of weights one through four."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = set()
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs.add(syndrome)
    return pairs


def _anchor_spectrum(cols, pairs, limit5=None):
    """Exact weight-five/six counts, conditional on the kernel-four screen."""
    triples = {}
    hits5 = hits6 = 0
    limit = None if limit5 is None else 10 * limit5
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for k in range(j):
                syndrome = pair ^ cols[k]
                if syndrome in pairs:
                    hits5 += 1
                    if limit is not None and hits5 > limit:
                        return None
                old = triples.get(syndrome, 0)
                hits6 += old
                triples[syndrome] = old + 1

    # Overlapping matching subsets would give a kernel of weight <=4,
    # already excluded. A five-word has ten pair/triple partitions;
    # a six-word has ten unordered triple/triple partitions.
    assert hits5 % 10 == 0 and hits6 % 10 == 0
    check_pairs = {}
    four = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, u in enumerate(checks):
            for v in checks[:a]:
                key = (v, u)
                old = check_pairs.get(key, 0)
                four += old
                check_pairs[key] = old + 1
    return hits5 // 10, hits6 // 10, four


def _anchor_seed(G, frame):
    """A singleton, a branch, and three Sidon-voltage junction columns."""
    cells = [[[] for _ in range(5)] for _ in range(3)]
    cells[0][0] = [G.e]
    cells[0][1] = [G.e]
    cells[1][1] = [G.e]
    for j, voltage in enumerate(frame, 2):
        cells[1][j] = [G.e]
        cells[2][j] = [(voltage, 0)]
    return cells


def _anchor_component(rng, G, side, profile, frame, emphasis):
    """Greedy kernel-four exclusion, followed by exact spectrum selection."""
    counts = _ANCHOR_TRIPLES[profile]
    seed = _anchor_seed(G, _ANCHOR_FRAMES[frame])
    assert _anchor_pairs(_anchor_columns(G, seed, side)) is not None
    best = None
    best_key = None
    completed = 0

    def key_of(spectrum):
        n5, n6, four = spectrum
        # The first objective always minimizes actual weight-five words.
        # Cards differ in the next objective, not just in a group gauge.
        return (n5, n6, four) if emphasis == 0 else (n5, four, n6)

    for attempt in range(40):
        cells = [[list(cell) for cell in row] for row in seed]
        good = True
        for column_counts in counts:
            accepted = False
            for trial in range(96):
                column = [rng.sample(G.elements, count)
                          for count in column_counts]
                candidate = [row + [column[i]]
                             for i, row in enumerate(cells)]
                cols = _anchor_columns(G, candidate, side)
                pairs = _anchor_pairs(cols)
                if pairs is None:
                    continue
                cells = candidate
                accepted = True
                break
            if not accepted:
                good = False
                break
        if not good:
            continue
        completed += 1
        spectrum = _anchor_spectrum(
            cols, pairs, None if best_key is None else best_key[0])
        if spectrum is not None:
            key = key_of(spectrum)
            if best_key is None or key < best_key:
                best, best_key = cells, key
        if completed >= 8:
            break

    if best is None:
        raise ValueError("anchored component could not exclude kernels <=4")

    # Change actual labels, including binomial differences, while retaining
    # this card's support. Every accepted move is a strict improvement.
    occupied = [(i, j) for i in range(3) for j in range(5, 11)
                if best[i][j]]
    for move in range(16):
        i, j = occupied[rng.randrange(len(occupied))]
        candidate = [[list(cell) for cell in row] for row in best]
        candidate[i][j] = rng.sample(G.elements, len(best[i][j]))
        cols = _anchor_columns(G, candidate, side)
        pairs = _anchor_pairs(cols)
        if pairs is None:
            continue
        spectrum = _anchor_spectrum(cols, pairs, best_key[0])
        if spectrum is None:
            continue
        key = key_of(spectrum)
        if key < best_key:
            best, best_key = candidate, key
    return best, best_key


# Twenty trinomial construction recipes, not automorphism-generated slots.
# First coordinates here are C3 cosets; second profiles are C4 cosets.
_BP_FRAMES = ((0,0,0), (0,0,1), (0,1,1), (0,1,2), (0,2,2))
_BP_TRIANGLES = ((0,0,0), (0,0,2), (0,1,2), (0,1,3))


def _bp_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _bp_backbone(rng, G, frame):
    """Require connectedness and no nonbacktracking length-two collision.

    Odd C4 coordinates make the check graph bipartite. Distinct signed
    generators exclude doubled edges; unique reduced two-step products
    exclude its four-cycles. Final component screens use verified lifts.
    """
    def legal(gs):
        ns = gs + [G.inv(g) for g in gs]
        keys = {G._key(g) for g in ns}
        if len(keys) != 6 or G._key(G.e) in keys:
            return False
        products = set()
        for g in ns:
            inverse = G._key(G.inv(g))
            for h in ns:
                if G._key(h) == inverse:
                    continue
                p = G._key(G.mul(g, h))
                if p in products or p == G._key(G.e):
                    return False
                products.add(p)
        reached = {G._key(G.e)}
        todo = [G.e]
        while todo:
            g = todo.pop()
            for h in ns:
                p = G.mul(g, h)
                key = G._key(p)
                if key not in reached:
                    reached.add(key)
                    todo.append(p)
        return len(reached) == G.q

    for trial in range(128):
        gs = [(rng.randrange(1, 7), 0, rng.choice((1, 3))),
              (rng.randrange(7), 1, rng.choice((1, 3))),
              (rng.randrange(7), 2, rng.choice((1, 3)))]
        if legal(gs):
            return 0, gs

    # Explicit legal seed, independently checked rather than trusted.
    gs = [(1, 0, 1), (0, 1, 1), (4, 2, 1)]
    if not legal(gs):
        raise ValueError("explicit Frobenius21 x C4 backbone failed")
    return 0, gs


def _bp_record(G, cells, side):
    """All matrix syndromes use the scaffold's verified conventions."""
    if side == "A":
        rows, _, _ = lifted_matrices(G, [cells], [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], [cells])
    width = 4 * G.q
    mask = (1 << width) - 1
    rows = [row & mask for row in rows]
    columns = [0] * width
    incidence = [[] for _ in rows]
    for r, row in enumerate(rows):
        for v in _bp_bits(row):
            columns[v] |= 1 << r
            incidence[r].append(v)

    adjacency = [set() for _ in columns]
    for variables in incidence:
        for v in variables:
            adjacency[v].update(u for u in variables if u != v)
    return rows, columns, adjacency


def _bp_small_floor(columns, adjacency):
    """Exact exclusion through five; capped counts of actual five-words.

    Degrees are two or three. A five-word must contain an even number
    of degree-three variables. Five degree-two variables cannot form a
    kernel in the screened bipartite backbone. Every remaining five-word
    therefore has a partition containing two degree-three variables and
    one degree-two variable on its triple side.
    """
    width = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1, 0
    if len(singles) != width:
        return 2, 0
    pairs = {}
    bits = [1 << v for v in range(width)]
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles:
                return 3, 0
            if syndrome in pairs:
                return 4, 0
            pairs[syndrome] = bits[a] | bits[b]

    doubles = [v for v, x in enumerate(columns) if x.bit_count() == 2]
    triples = [v for v, x in enumerate(columns) if x.bit_count() == 3]
    assert len(doubles) + len(triples) == width
    incidence = {}
    for v in doubles:
        for r in _bp_bits(columns[v]):
            incidence.setdefault(r, []).append(v)

    found = set()
    for t, a in enumerate(triples):
        for b in triples[:t]:
            syndrome = columns[a] ^ columns[b]
            used = bits[a] | bits[b]
            if syndrome.bit_count() == 6:
                # A disjoint double would leave weight eight, larger
                # than every possible pair syndrome (at most six).
                candidates = set()
                for r in _bp_bits(syndrome):
                    candidates.update(incidence.get(r, ()))
            else:
                candidates = doubles
            for c in candidates:
                previous = pairs.get(syndrome ^ columns[c])
                if previous is None or previous & (used | bits[c]):
                    continue
                word = previous | used | bits[c]
                assert word.bit_count() == 5
                found.add(word)
                if len(found) >= 256:
                    return 5, 256
    return (5, len(found)) if found else (6, 0)


def _bp_reading(rng, columns, trials):
    """Genuine classical kernel witnesses; sampled upper readings only."""
    width = len(columns)
    order = list(range(width))
    best = width + 1
    for trial in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for v in order:
            syndrome, word = columns[v], 1 << v
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        words.sort(key=int.bit_count)
        for a, word in enumerate(words[:20]):
            for other in words:
                candidate = word ^ other
                if candidate:
                    best = min(best, candidate.bit_count())
        if best <= 5:
            break
    return best


def _bp_cycles(columns):
    """Exact Tanner four-cycle count, used only as a secondary objective."""
    pairs = {}
    count = 0
    for syndrome in columns:
        checks = list(_bp_bits(syndrome))
        for a, u in enumerate(checks):
            for v in checks[:a]:
                key = (v, u)
                count += pairs.get(key, 0)
                pairs[key] = pairs.get(key, 0) + 1
    return count


def _bp_component_pool(rng, G, side, frame, mode):
    """Optimize actual trinomial geometry within one of twenty recipes."""
    bs = _BP_FRAMES[frame]
    zs = _BP_TRIANGLES[mode]
    pool = []

    def draw_tri():
        tri = []
        for b, z in zip(bs, zs):
            available = [(a, b, z) for a in range(7)
                         if (a, b, z) not in tri]
            tri.append(rng.choice(available))
        return tri

    def inspect(cells):
        rows, columns, adjacency = _bp_record(G, cells, side)
        if gf2_rank(rows) != G.q:
            return None
        floor, n5 = _bp_small_floor(columns, adjacency)
        return {"cells": cells, "columns": columns, "floor": floor,
                "n5": n5, "cycles": _bp_cycles(columns),
                "reading": len(columns) + 1}

    def geometry(item):
        return (item["floor"], -item["n5"], -item["cycles"])

    def remember(item):
        if item is None:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in item["cells"])
        if any(signature == old["signature"] for old in pool):
            return
        item["signature"] = signature
        pool.append(item)
        pool.sort(key=geometry, reverse=True)
        del pool[6:]

    for attempt in range(48):
        _, gs = _bp_backbone(rng, G, frame)
        cells = [[G.e, g] for g in gs] + [draw_tri()]
        remember(inspect(cells))
        if len(pool) >= 6 and sum(x["floor"] == 6 for x in pool) >= 3:
            break
    if not pool:
        raise ValueError("no full-rank direct-product component")

    # Repair actual exponent geometry, preserving the card's quotient
    # profile and the already legal binomial backbone.
    current = pool[0]
    for move in range(16):
        cells = [list(cell) for cell in current["cells"]]
        if move % 4 == 0:
            cells[3] = draw_tri()
        else:
            t = rng.randrange(3)
            old = cells[3][t]
            available = [(a, old[1], old[2]) for a in range(7)
                         if (a, old[1], old[2]) not in cells[3]]
            if not available:
                continue
            cells[3][t] = rng.choice(available)
        item = inspect(cells)
        remember(item)
        if item is not None and geometry(item) >= geometry(current):
            current = item

    finalists = sorted(pool, key=geometry, reverse=True)[:4]
    for item in finalists:
        item["reading"] = _bp_reading(rng, item["columns"], 16)
        item["key"] = (item["floor"], -item["n5"],
                       item["reading"], -item["cycles"])
    finalists.sort(key=lambda item: item["key"], reverse=True)

    # Confirm several independent candidates, not one lucky reading.
    for item in finalists[:3]:
        item["reading"] = min(
            item["reading"], _bp_reading(rng, item["columns"], 48))
        item["key"] = (item["floor"], -item["n5"],
                       item["reading"], -item["cycles"])
    finalists.sort(key=lambda item: item["key"], reverse=True)
    return finalists[:3]


def _bp_full_kernel4_free(rows, width):
    """Exclude every complete-check-matrix kernel word of weight <=4.

    This is stronger than checking only components. It does not exclude
    weight-five logicals or certify the proposed distance-six target.
    """
    columns = [0] * width
    for r, row in enumerate(rows):
        for v in _bp_bits(row):
            columns[v] |= 1 << r
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def NEW_CTX(rng, G, idx):
    """Twenty modular-group topology cards with verified witness repair."""
    from itertools import product
    from math import exp

    q = G.q
    assert q == 32
    card, cohort = idx % 20, idx // 20
    concentration, mode = card // 4, card % 4
    bits = _bp_bits
    mask = (1 << q) - 1

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)
    orders = []
    for g in range(q):
        h = identity
        for order in range(1, q + 1):
            h = table[h][g]
            if h == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid modular-group element order")
    relative = [[orders[table[inverses[g]][h]] for h in range(q)]
                for g in range(q)]

    # These are authoritative monomial matrices, not a handwritten lift.
    hx, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, hz, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    blocks = {
        "A": [[(row >> (j*q)) & mask for row in hx]
              for j in range(q)],
        "B": [[(row >> (j*q)) & mask for row in hz]
              for j in range(q)],
    }

    tail_profiles = (
        ((2,1), (2,1), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (1,2)),
        ((3,0), (3,0), (0,3), (0,3)),
        ((3,0), (1,2), (1,2), (1,2)),
    )
    acols = ((2,0), (0,2), (1,1)) + tail_profiles[mode]
    sa = [[column[i] for column in acols] for i in range(2)]

    # Every B column is cubic. Concentration changes actual zero blocks.
    options = [(0, 3) if j < concentration else (1, 2)
               for j in range(6)]
    allocations = [p for p in product(*options) if sum(p) == 9]
    assert allocations
    upper = rng.choice(allocations)
    sb = [list(upper), [3-x for x in upper]]
    assert all(sum(row) == 9 for row in sa + sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    penalties = (
        {2: 1000, 4: 12, 8: 2, 16: 0},
        {2: 1000, 4: 8, 8: 0, 16: 1},
        {2: 1000, 4: 16, 8: 1, 16: 0},
        {2: 1000, 4: 3, 8: 0, 16: 2},
    )
    long_ratios = [g for g in range(q)
                   if orders[g] in ((16,) if mode in (0, 2)
                                    else (8, 16))]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_cell(count):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            return [root, table[root][rng.choice(long_ratios)]]
        assert count == 3
        best = None
        for _ in range(12):
            cell = rng.sample(range(q), 3)
            cost = sum(penalties[mode].get(relative[g][h], 1000)
                       for i, g in enumerate(cell) for h in cell[:i])
            if best is None or cost < best[0]:
                best = cost, cell
        return best[1]

    def fresh(support, side):
        grid = [[draw_cell(count) for count in row] for row in support]
        if side == "A":
            gap = (1, 7, 9, 15)[mode]
            grid[0][0] = [identity, G.idx((1, 0))]
            grid[1][1] = [identity, G.idx((gap, 1))]
            grid[0][2] = [identity]
            grid[1][2] = [identity]
        return grid

    def inspect(grid, side):
        nc = len(grid[0])
        rows = [0] * (2*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                part = [0] * q
                for g in cell:
                    monomial = blocks[side][g]
                    for r in range(q):
                        part[r] ^= monomial[r]
                for r in range(q):
                    rows[i*q+r] |= part[r] << (j*q)
        columns = [0] * (nc*q)
        adjacency = [0] * (2*q)
        forced = 0
        for r, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << r
        for column in columns:
            degree = column.bit_count()
            forced += degree*(degree-1)*(degree-2)//6
            for r in bits(column):
                adjacency[r] |= column ^ (1 << r)
        four = 0
        for r, row in enumerate(rows):
            for other in rows[:r]:
                overlap = (row & other).bit_count()
                four += overlap*(overlap-1)//2
        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u+1))):
                triangles += (
                    neighbors & adjacency[v] & (-1 << (v+1))
                ).bit_count()
        excess = max(0, triangles-forced)
        peak = max(x.bit_count() for x in adjacency)
        # Projection excess is a heuristic when four-cycles are present.
        cost = 900*four + (1+mode)*excess + mode*peak
        return {"grid": copy_grid(grid), "rows": rows,
                "columns": columns, "cost": cost,
                "cycles": (four, excess), "words": set(),
                "observed": len(columns)+1}

    def syndrome(word, columns):
        value = 0
        for v in bits(word):
            value ^= columns[v]
        return value

    def challenge(rec, budget, trials):
        """Exact <=5 screening plus bounded searches for actual <=8 words."""
        columns = rec["columns"]
        width = len(columns)
        singles, pairs = {}, {}
        short = None

        def note(word):
            if word and syndrome(word, columns) == 0:
                weight = word.bit_count()
                rec["observed"] = min(rec["observed"], weight)
                if weight <= 8 and len(rec["words"]) < 16:
                    rec["words"].add(word)

        for v, value in enumerate(columns):
            word = 1 << v
            if value == 0:
                short = word
                break
            previous = singles.get(value)
            if previous is not None:
                short = previous ^ word
                break
            singles[value] = word
        if short is None:
            for a, value in enumerate(columns):
                for b in range(a):
                    key = value ^ columns[b]
                    word = (1 << a) | (1 << b)
                    previous = singles.get(key)
                    if previous is None:
                        previous = pairs.get(key)
                    if previous is not None:
                        short = previous ^ word
                        break
                    pairs[key] = word
                if short is not None:
                    break

        # Regular symmetry is used only for classical components.
        if short is None:
            for root in range(0, width, q):
                cr = columns[root]
                for a, value in enumerate(columns):
                    if a == root:
                        continue
                    partial = cr ^ value
                    used = (1 << root) | (1 << a)
                    for b in range(a):
                        if b == root:
                            continue
                        previous = pairs.get(partial ^ columns[b])
                        word = used | (1 << b)
                        if previous is not None and not (previous & word):
                            short = previous | word
                            break
                    if short is not None:
                        break
                if short is not None:
                    break

        if short is not None:
            note(short)
            rec["screen_cap"] = short.bit_count()
            rec["d"] = min(rec["observed"], _bp_reading(
                rng, columns, min(trials, 8)))
            return

        rec["screen_cap"] = 6  # Exact absence through five, component only.
        incidence = [0] * len(rec["rows"])
        for v, column in enumerate(columns):
            for r in bits(column):
                incidence[r] |= 1 << v
        roots = list(range(0, width, q))
        rng.shuffle(roots)
        priorities = [rng.random() for _ in columns]
        all_variables = (1 << width)-1

        for root in roots:
            nodes = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if nodes[0] <= 0:
                    return
                nodes[0] -= 1
                if value == 0:
                    note(word)
                    return
                left = min(8, rec["observed"]) - word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    note(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(word | pair)
                left = min(8, rec["observed"]) - word.bit_count()
                if left <= 2:
                    return
                chosen = None
                smallest = width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if count == 0:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                candidates = list(bits(chosen))
                candidates.sort(key=lambda v: (
                    -(columns[v] & value).bit_count(), priorities[v]))
                live = allowed
                for v in candidates:
                    live &= ~(1 << v)
                    visit(value ^ columns[v], word | (1 << v), live)
                    if nodes[0] <= 0:
                        break

            allowed = all_variables & ~((1 << root)-1)
            visit(columns[root], 1 << root, allowed & ~(1 << root))
            if rec["observed"] <= 6:
                break
        rec["d"] = min(rec["observed"],
                       _bp_reading(rng, columns, trials))
        # A missed bounded witness is never converted into a certified floor.

    def component_pool(side, support):
        mutable = [(i, j) for i in range(2)
                   for j in range(len(support[0]))
                   if support[i][j] and (side == "B" or j >= 3)]
        bank = {}
        known = set()

        def signature(rec):
            return tuple(tuple(sorted(cell))
                         for row in rec["grid"] for cell in row)

        def remember(rec):
            if gf2_rank(rec["rows"]) != 2*q:
                return
            bank[signature(rec)] = rec
            if len(bank) > 6:
                worst = max(bank, key=lambda key: bank[key]["cost"])
                del bank[worst]

        for restart in range(4):
            current = inspect(fresh(support, side), side)
            remember(current)
            for step in range(96):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(mutable)
                grid[i][j] = draw_cell(support[i][j])
                proposed = inspect(grid, side)
                delta = proposed["cost"]-current["cost"]
                temperature = q*(8*(1-step/96.0)**2 + 0.1)
                if delta <= 0 or rng.random() < exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(current)
        if not bank:
            raise ValueError("no full-row-rank modular component")

        records = list(bank.values())
        for rec in records:
            challenge(rec, 7000, 8)
            known.update(rec["words"])

        def merit(rec):
            return (rec["screen_cap"], rec["d"], -rec["cost"])

        records.sort(key=merit, reverse=True)
        current = records[0]
        keepers = records[:4]

        def repair_cost(rec):
            return rec["cost"] + 4000*sum(
                syndrome(word, rec["columns"]) == 0 for word in known)

        for step in range(24):
            grid = copy_grid(current["grid"])
            active = [word for word in known
                      if syndrome(word, current["columns"]) == 0]
            choices = mutable
            if active and step % 3 != 2:
                occupied = {v // q for v in bits(rng.choice(active))}
                targeted = [(i, j) for i, j in mutable if j in occupied]
                if targeted:
                    choices = targeted
            i, j = rng.choice(choices)
            grid[i][j] = draw_cell(support[i][j])
            proposed = inspect(grid, side)
            if gf2_rank(proposed["rows"]) != 2*q:
                continue
            delta = repair_cost(proposed)-repair_cost(current)
            temperature = 80*(1-step/24.0)**2 + 0.2
            if delta > 0 and rng.random() >= exp(
                    min(0.0, -delta/temperature)):
                continue
            challenge(proposed, 5000, 6)
            known.update(proposed["words"])
            if merit(proposed) >= merit(current) or (
                    proposed["screen_cap"] == current["screen_cap"]
                    and proposed["d"] == current["d"]):
                current = proposed
            keepers.append(proposed)
            keepers.sort(key=merit, reverse=True)
            del keepers[6:]

        final, seen = [], set()
        for rec in sorted(keepers + [current], key=merit, reverse=True):
            key = signature(rec)
            if key in seen:
                continue
            seen.add(key)
            previous = rec["d"]
            challenge(rec, 30000, 48)
            rec["d"] = min(previous, rec["d"])
            assert all(syndrome(word, rec["columns"]) == 0
                       for word in rec["words"])
            final.append(rec)
            if len(final) == 3:
                break
        final.sort(key=merit, reverse=True)
        return final

    aa = component_pool("A", sa)
    bb = component_pool("B", sb)
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        min(ab[0]["screen_cap"], ab[1]["screen_cap"]),
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"] + ab[1]["d"],
        -ab[0]["cost"] - ab[1]["cost"]), reverse=True)

    for a, b in proposals:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1472 and k >= 640
                and 425*n <= 1000*k < 475*n):
            continue

        # Check cached component assembly against the complete verified lift.
        actual_a = []
        for i in range(2):
            for r in range(q):
                source = hx[(i*6)*q+r]
                actual_a.append(sum(
                    ((source >> (j*6*q)) & mask) << (j*q)
                    for j in range(7)))
        actual_b = [hz[alpha*q+r] & ((1 << (6*q))-1)
                    for alpha in range(2) for r in range(q)]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)
        return {"A": A, "B": B, "k": k,
                "card": (concentration, mode, cohort),
                "component_readings": (a["d"], b["d"]),
                "component_screen_caps": (
                    a["screen_cap"], b["screen_cap"]),
                "component_cycles": (a["cycles"], b["cycles"])}
    raise ValueError("modular cubic-component pairs missed cell 0.450")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored new-shape cell without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
# REFUTED: C7:C12(s=3) odd-quotient backbones and kernel-five selection produced five readings >=6; all five measured codes instead returned weight four with abundant sector-one X and Z minima.
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
