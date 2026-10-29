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
CONCEPT = ("For SL(2,5), order 120, A=(1,2), B=(2,5), long-order relative labels and independently screened components should exceed five-code fitness 52.4272 in rate cell 0.250. A non-improving measured fitness refutes this hypothesis; distance readings remain provisional upper bounds.")
GROUP = ("Jump from cyclic Z29 to the determinant-one matrix group SL(2,5). Its order is 120, centre has order two, and it has nine conjugacy classes. The run's order-120 centre-two nine-class bed scored 64.8 on (1,2)x(3,4); these shared structural invariants motivate transfer to this untried shape bed, without establishing a distance improvement.")
SHAPES = ("A=(1,2), B=(2,5), q=120 give n=1440, k>=360 and guaranteed rate 3/12=0.250. The unrestricted shape permits q<=124; this group supplies q=120. Components have 240 variables/120 checks and 600 variables/240 checks. Parent measured minima occupied only sector-1 Acol/Bcol blocks, with frequent Bcol3/4 use; both component kernels are replaced and exact full ranks retain [0.225,0.275).")
FILLING = ("Twenty cards cross five density allocations with four prescribed binomial relative-order models: ten, six, five, or mixed six/ten. All component columns have degree at least three. Explicit margins give both quantum row caps <=12. Select independent components using verified lifts, exact cycles and repeated genuine kernel witnesses; repair cells touched by short witnesses. Pair finalists using exact full ranks and sampled actual logicals distinguished from the ORIGINAL stabilizer spans. No cyclic voltage or minor screen is used; none of these searches certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_SL2F5_12x25_relativeorder_verified_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    from itertools import product
    if not valid_q(q):
        raise ValueError("SL(2,5) requires order 120")
    els = [g for g in product(range(5), repeat=4)
           if (g[0]*g[3] - g[1]*g[2]) % 5 == 1]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % 5 for x in g)

    def mul(g, h):
        a, b, c, d = g
        e, f, u, v = h
        return ((a*e+b*u) % 5, (a*f+b*v) % 5,
                (c*e+d*u) % 5, (c*f+d*v) % 5)

    def inv(g):
        a, b, c, d = g
        return (d, (-b) % 5, (-c) % 5, a)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _balanced_support(rng, columns, zeros):
    """Exact margins; each column touches at least two base checks."""
    from functools import lru_cache
    options = []
    for total in columns:
        opts = [
            (a, b, total-a-b)
            for a in range(4) for b in range(4)
            if 0 <= total-a-b <= 3
            and sum(v > 0 for v in (a, b, total-a-b)) >= 2
        ]
        rng.shuffle(opts)
        options.append(opts)

    @lru_cache(None)
    def feasible(j, a, b, c, z):
        if min(a, b, c, z) < 0:
            return False
        if j == len(columns):
            return (a, b, c, z) == (0, 0, 0, 0)
        if a+b+c != sum(columns[j:]) or z > len(columns)-j:
            return False
        for v in options[j]:
            dz = sum(x == 0 for x in v)
            if feasible(j+1, a-v[0], b-v[1], c-v[2], z-dz):
                return True
        return False

    state = (8, 8, 8, zeros)
    if not feasible(0, *state):
        raise ValueError("infeasible deliberate support profile")
    grid = [[], [], []]
    for j in range(len(columns)):
        choices = []
        for v in options[j]:
            nxt = tuple(state[t]-v[t] for t in range(3))
            nz = state[3]-sum(x == 0 for x in v)
            if feasible(j+1, *nxt, nz):
                choices.append((v, nxt+(nz,)))
        v, state = choices[rng.randrange(len(choices))]
        for i in range(3):
            grid[i].append(v[i])
    return grid


def _count_minor_floor(grid):
    """Term-count bound used only to select a support topology."""
    from itertools import combinations, permutations
    det = {}
    for cols in combinations(range(len(grid[0])), 3):
        det[cols] = sum(
            grid[0][cols[p[0]]] *
            grid[1][cols[p[1]]] *
            grid[2][cols[p[2]]]
            for p in permutations(range(3))
        )
    return min(
        sum(det[tuple(c for c in cols if c != omitted)]
            for omitted in cols)
        for cols in combinations(range(len(grid[0])), 4)
    )


def _voltage_cost(cells, q):
    """Algebraic repeated-difference objective, not a substitute lift."""
    score = 0
    for i in range(3):
        counts = {}
        for cell in cells[i]:
            for u in range(len(cell)):
                for v in range(u):
                    d = (cell[u][0]-cell[v][0]) % q
                    d = min(d, q-d)
                    score += counts.get(d, 0)
                    counts[d] = counts.get(d, 0)+1
        for k in range(i):
            counts = {}
            for j in range(len(cells[0])):
                for g in cells[i][j]:
                    for h in cells[k][j]:
                        d = (g[0]-h[0]) % q
                        score += counts.get(d, 0)
                        counts[d] = counts.get(d, 0)+1
    return score


def _optimized_labels(rng, G, support):
    cells = [
        [rng.sample(G.elements, count) for count in row]
        for row in support
    ]
    places = [
        (i, j, t)
        for i, row in enumerate(cells)
        for j, cell in enumerate(row)
        for t in range(len(cell))
    ]
    cost = _voltage_cost(cells, G.q)
    best_cost = cost
    best = [[list(c) for c in row] for row in cells]
    for step in range(360):
        if best_cost == 0:
            break
        i, j, t = places[rng.randrange(len(places))]
        old = cells[i][j][t]
        proposal = G.pick(rng)
        if proposal in cells[i][j]:
            continue
        cells[i][j][t] = proposal
        new_cost = _voltage_cost(cells, G.q)
        neutral = 0.35 if step < 240 else 0.08
        accept = (
            new_cost < cost
            or (new_cost == cost and rng.random() < neutral)
            or (step < 180 and new_cost == cost+1
                and rng.random() < 0.025)
        )
        if accept:
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best = [[list(c) for c in row] for row in cells]
        else:
            cells[i][j][t] = old
    return best, best_cost


def _component_rows(hx, hz, q):
    """Extract components from the VERIFIED full quantum construction."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i*cb)*q+t]
            row = 0
            for j in range(ca):
                row |= ((source >> (j*cb*q)) & mask) << (j*q)
            ar.append(row)
    br = [
        hz[a*q+t] & ((1 << (cb*q))-1)
        for a in range(rb) for t in range(q)
    ]
    return ar, br


def _minor_reading(rows, cells, q, inverse=False):
    """Verified nonzero classical kernel words; not quantum-distance proofs."""
    from itertools import combinations, permutations
    mask = (1 << q)-1

    def multiply(a, b):
        out = 0
        while a:
            bit = a & -a
            shift = bit.bit_length()-1
            out ^= ((b << shift) | (b >> (q-shift))) & mask
            a ^= bit
        return out

    polys = []
    for row in cells:
        prow = []
        for cell in row:
            p = 0
            for g in cell:
                exponent = (-g[0] if inverse else g[0]) % q
                p ^= 1 << exponent
            prow.append(p)
        polys.append(prow)

    minors = {}
    width = len(cells[0])
    for cols in combinations(range(width), 3):
        p = 0
        for perm in permutations(range(3)):
            p ^= multiply(
                multiply(polys[0][cols[perm[0]]],
                         polys[1][cols[perm[1]]]),
                polys[2][cols[perm[2]]]
            )
        minors[cols] = p

    best = width*q+1
    for cols in combinations(range(width), 4):
        word = 0
        for omitted in cols:
            minor = minors[tuple(c for c in cols if c != omitted)]
            word |= minor << (omitted*q)
        if word and all((word & row).bit_count() % 2 == 0
                        for row in rows):
            best = min(best, word.bit_count())
    return best


def _information_set_reading(rows, width, rng, trials):
    """Search genuine component-kernel vectors in random column orders."""
    best = width+1
    for _ in range(trials):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapping = [1 << p for p in permutation]
        pivots = {}
        for source in rows:
            row = 0
            while source:
                bit = source & -source
                row ^= mapping[bit.bit_length()-1]
                source ^= bit
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ordered = sorted(pivots)
        for at, p in enumerate(ordered):
            bit = 1 << p
            reduced = pivots[p]
            for higher in ordered[at+1:]:
                if pivots[higher] & bit:
                    pivots[higher] ^= reduced

        words = []
        for f in range(width):
            if f in pivots:
                continue
            bit = 1 << f
            word = bit
            for p in ordered:
                if pivots[p] & bit:
                    word |= 1 << p
            best = min(best, word.bit_count())
            words.append(word)

        words.sort(key=int.bit_count)
        shortlist = words[:18]
        for i, word in enumerate(shortlist):
            for other in shortlist[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 8:
            break
    return best


def _retired_cyclic_ctx(rng, G, idx):
    """Inactive historical implementation; not used on the matrix-group bed."""
    from functools import lru_cache

    def support(columns, zeros, row_sum):
        """Exact row/column margins; no column uses only one base row."""
        options = []
        for total in columns:
            opts = [
                (a, b, total-a-b)
                for a in range(4) for b in range(4)
                if 0 <= total-a-b <= 3
                and sum(x > 0 for x in (a, b, total-a-b)) >= 2
            ]
            rng.shuffle(opts)
            options.append(opts)

        @lru_cache(None)
        def feasible(j, a, b, c, z):
            if min(a, b, c, z) < 0:
                return False
            if j == len(columns):
                return (a, b, c, z) == (0, 0, 0, 0)
            if a+b+c != sum(columns[j:]) or z > len(columns)-j:
                return False
            return any(
                feasible(j+1, a-v[0], b-v[1], c-v[2],
                         z-sum(x == 0 for x in v))
                for v in options[j]
            )

        state = (row_sum, row_sum, row_sum, zeros)
        if not feasible(0, *state):
            raise ValueError("infeasible reallocated support profile")
        grid = [[], [], []]
        for j in range(len(columns)):
            choices = []
            for v in options[j]:
                nxt = tuple(state[t]-v[t] for t in range(3))
                nxt += (state[3]-sum(x == 0 for x in v),)
                if feasible(j+1, *nxt):
                    choices.append((v, nxt))
            v, state = choices[rng.randrange(len(choices))]
            for i in range(3):
                grid[i].append(v[i])
        return grid

    def kernel_reading(rows, width, trials):
        """Genuine null words: random systematic bases, pairs and triples.

        Syndrome columns come exclusively from the verified component.
        Searching these words is not exhaustive and is not a quantum
        distance certificate.
        """
        columns = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                columns[bit.bit_length()-1] |= 1 << r
                source ^= bit

        best, witness = width+1, None
        order = list(range(width))
        for _ in range(trials):
            rng.shuffle(order)
            pivots, words = {}, []
            for j in order:
                syndrome, word = columns[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length()-1
                    entry = pivots.get(p)
                    if entry is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= entry[0]
                    word ^= entry[1]
                if not syndrome:
                    words.append(word)
                    weight = word.bit_count()
                    if weight < best:
                        best, witness = weight, word

            # Unlike the parent, examine every basis pair, not just
            # pairs among the eighteen shortest individual words.
            words.sort(key=int.bit_count)
            for i, word in enumerate(words):
                for j in range(i):
                    candidate = word ^ words[j]
                    weight = candidate.bit_count()
                    if weight < best:
                        best, witness = weight, candidate

            short = words[:16]
            for i, word in enumerate(short):
                for j in range(i):
                    pair = word ^ short[j]
                    for t in range(j):
                        candidate = pair ^ short[t]
                        weight = candidate.bit_count()
                        if weight < best:
                            best, witness = weight, candidate
            if best <= 10:
                break

        if witness is None or not witness:
            raise ValueError("component search produced no nonzero witness")
        assert all((witness & row).bit_count() % 2 == 0 for row in rows)
        return best

    def quality(item):
        return (min(item["da"], item["db"]),
                item["da"]+item["db"], -item["cost"])

    def refine(item, trials_a, trials_b):
        # Accumulate actual readings separately; never substitute the
        # combined minimum for both components as the parent did.
        item["da"] = min(item["da"], kernel_reading(
            item["ar"], A_SHAPE[1]*G.q, trials_a))
        item["db"] = min(item["db"], kernel_reading(
            item["br"], B_SHAPE[1]*G.q, trials_b))

    design = idx % 20
    # One zero cannot balance an otherwise monomial degree-three A:
    # its single 2/1/0 column would leave unequal row totals.
    zero_a = (0, 2, 3, 4, 5)[design % 5]
    zero_b = design // 5
    degree_five = (idx // 20) % 3
    b_columns = ([3]*(1+degree_five)
                 + [4]*(6-2*degree_five) + [5]*degree_five)

    pool = []
    for topology in range(3):
        sa = max(
            (support([3]*6, zero_a, 6) for _ in range(4)),
            key=_count_minor_floor
        )
        sb = max(
            (support(b_columns, zero_b, 9) for _ in range(4)),
            key=_count_minor_floor
        )
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        for attempt in range(6):
            A, cost_a = _optimized_labels(rng, G, sa)
            B, cost_b = _optimized_labels(rng, G, sb)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _component_rows(hx, hz, G.q)
            da = min(_minor_reading(ar, A, G.q),
                     kernel_reading(ar, A_SHAPE[1]*G.q, 8))
            db = min(_minor_reading(br, B, G.q, inverse=True),
                     kernel_reading(br, B_SHAPE[1]*G.q, 12))
            pool.append({
                "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
                "ar": ar, "br": br, "da": da, "db": db,
                "cost": cost_a+cost_b
            })

    finalists = []
    for item in sorted(pool, key=quality, reverse=True):
        k = item["n"]-gf2_rank(item["hx"])-gf2_rank(item["hz"])
        if not (k > 0 and 225*item["n"] <= 1000*k < 275*item["n"]):
            continue
        item["k"] = k
        refine(item, 48, 80)
        finalists.append(item)
        if len(finalists) == 4:
            break
    if not finalists:
        raise ValueError("no candidate passed the actual lifted-rate gate")

    finalists.sort(key=quality, reverse=True)
    deep = finalists[:2]
    for item in deep:
        refine(item, 128, 192)
    chosen = max(
        deep,
        key=lambda item: (
            item["k"]*min(item["da"], item["db"])**2,
            min(item["da"], item["db"]),
            item["da"]+item["db"], -item["cost"]
        )
    )
    assert css_ok(chosen["hx"], chosen["hz"])
    return {"idx": idx, "cells": {"A": chosen["A"], "B": chosen["B"]},
            "k": chosen["k"],
            "component_readings": (chosen["da"], chosen["db"])}


# New construction does not call the historical cyclic-only helpers above.
_SL_CARDS = (
    ((3,3), (6,3,3,3,3), (9,9)),
    ((3,3), (5,4,3,3,3), (9,9)),
    ((3,3), (4,4,4,3,3), (9,9)),
    ((3,4), (4,3,3,3,3), (8,8)),
    ((4,4), (4,3,3,3,3), (8,8)),
)


def _sl_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length()-1
        word ^= bit


def _sl_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _sl_bits(row):
            columns[c] |= 1 << r
    return columns


def _sl_span(rows):
    pivots = {}
    for source in rows:
        row = source
        while row:
            p = row.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    return pivots


def _sl_remainder(word, pivots):
    """Canonical linear remainder modulo the ORIGINAL row span."""
    result = 0
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            bit = 1 << p
            result |= bit
            word ^= bit
        else:
            word ^= old
    return result


def _sl_probe(columns, seed, trials, stabilizers=None):
    """Actual kernel words; with stabilizers, retain only nontrivial logicals."""
    local = random.Random(seed)
    width = len(columns)
    span = None if stabilizers is None else _sl_span(stabilizers)
    best, witness = width+1, None

    def eligible(word):
        return bool(word) and (
            span is None or _sl_remainder(word, span) != 0)

    def note(word):
        nonlocal best, witness
        if word and word.bit_count() < best and eligible(word):
            best, witness = word.bit_count(), word

    order = list(range(width))
    for _ in range(trials):
        local.shuffle(order)
        pivots, circuits = {}, []
        for c in order:
            value, word = columns[c], 1 << c
            while value:
                p = value.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (value, word)
                    break
                value ^= previous[0]
                word ^= previous[1]
            if value == 0:
                circuits.append(word)
                note(word)

        circuits.sort(key=int.bit_count)
        light = circuits[:16]
        for a in light:
            for b in circuits:
                if a != b:
                    note(a ^ b)
        for _ in range(min(96, len(circuits)*2)):
            if len(circuits) >= 3:
                a, b, c = local.sample(circuits, 3)
                note(a ^ b ^ c)

        reducers = (list(stabilizers) if stabilizers is not None
                    else list(circuits))
        local.shuffle(reducers)
        seeds = [word for word in light if eligible(word)][:3]
        if witness is not None:
            seeds.append(witness)
        for word in seeds:
            weight = word.bit_count()
            for sweep in range(2):
                changed = False
                for other in reducers:
                    proposal = word ^ other
                    if proposal and proposal.bit_count() < weight:
                        word, weight = proposal, proposal.bit_count()
                        changed = True
                if not changed:
                    break
            note(word)
        if best <= 8:
            break

    if witness is not None:
        syndrome = 0
        for c in _sl_bits(witness):
            syndrome ^= columns[c]
        assert syndrome == 0
        assert eligible(witness)
    return best, witness


def _sl_geometry(rows, columns):
    """Exact component C4/C6 counts, with repeated variables excluded."""
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    four = six = 0
    for a in range(h):
        for b in range(a+1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                four += overlap*(overlap-1)//2
    for a in range(h):
        for t, b in enumerate(upper[a]):
            for c in upper[a][t+1:]:
                bc = common[b][c]
                if bc:
                    ab, ac = common[a][b], common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    six += ab*ac*bc - triple*(ab+ac+bc-2)

    adjacency = [0] * h
    for value in columns:
        for r in _sl_bits(value):
            adjacency[r] |= value ^ (1 << r)
    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _sl_bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    connected = visited == (1 << h)-1
    return four, six, connected


def _sl_record(G, grid, side):
    """Use only the scaffold's verified lift, including its side conventions."""
    unit = [[[G.e]]]
    width = len(grid[0])*G.q
    mask = (1 << width)-1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, unit)
        rows = [row & mask for row in hx]
    else:
        _, hz, _ = lifted_matrices(G, unit, grid)
        rows = [row & mask for row in hz]
    columns = _sl_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    rank = gf2_rank(rows)
    if rank < len(rows)-12:
        return None
    four, six, connected = _sl_geometry(rows, columns)
    if not connected:
        return None
    return {"grid": grid, "rows": rows, "cols": columns,
            "rank": rank, "cycles": (four, six)}


def NEW_CTX(rng, G, idx):
    """Twenty prescribed margin/order designs with independent retesting."""
    from functools import lru_cache

    card = idx % 20
    density, model = card % 5, card // 5
    cohort = idx // 20
    counts_a, degrees_b, targets_b = _SL_CARDS[density]
    q = G.q
    assert q == 120

    # All B columns touch both base checks. A margins are explicit.
    options = [
        tuple(a for a in range(1, 4) if 1 <= degree-a <= 3)
        for degree in degrees_b
    ]

    @lru_cache(None)
    def feasible(j, remaining):
        if j == len(degrees_b):
            return remaining == 0
        return any(feasible(j+1, remaining-a) for a in options[j])

    remaining = targets_b[0]
    sb = [[], []]
    for j, degree in enumerate(degrees_b):
        choices = [a for a in options[j]
                   if feasible(j+1, remaining-a)]
        a = rng.choice(choices)
        sb[0].append(a)
        sb[1].append(degree-a)
        remaining -= a
    sa = [list(counts_a)]
    assert tuple(map(sum, sb)) == targets_b
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid matrix-group element order")

    def draw_cell(count, i, j, side):
        if count == 1:
            return [G.pick(rng)]
        desired = (10, 6, 5)[model] if model < 3 else (
            6 if (i+j+int(side == "B")+cohort) % 2 else 10)
        for _ in range(2048):
            cell = rng.sample(G.elements, count)
            relatives = [
                G.mul(G.inv(cell[a]), cell[b])
                for a in range(count) for b in range(a)
            ]
            relative_orders = [orders[g] for g in relatives]
            # Exclude the central involution, which makes binomial
            # blocks split into length-two cycles.
            if min(relative_orders) <= 2:
                continue
            if count == 2:
                if relative_orders[0] == desired:
                    return cell
                continue
            if desired not in relative_orders:
                continue
            roots = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if any(G.mul(a, b) != G.mul(b, a)
                   for a in roots for b in roots):
                return cell
        raise ValueError("relative-order cell recipe exhausted")

    def fresh(support, side):
        return [[draw_cell(count, i, j, side)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    diagnostic_seed = rng.randrange(1 << 60)

    def merit(record):
        four, six = record["cycles"]
        return (record["d"], -four, -six, record["rank"])

    def component_pool(support, side):
        seed = diagnostic_seed + (0 if side == "A" else 100003)
        records = []
        signatures = set()
        for trial in range(24):
            grid = fresh(support, side)
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                continue
            signatures.add(signature)
            record = _sl_record(G, grid, side)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["cycles"][0], r["cycles"][1], -r["rank"]))
        screened = []
        for record in records[:6]:
            record["d"], record["word"] = _sl_probe(
                record["cols"], seed, 8)
            screened.append(record)
        if not screened:
            return []

        # Repair actual witness locations while preserving the card's
        # margins and relative-order conditions.
        current = max(screened, key=merit)
        for step in range(6):
            word = current["word"]
            active = [c//q for c in _sl_bits(word)] if word else []
            j = rng.choice(active) if active else rng.randrange(len(support[0]))
            occupied = [i for i in range(len(support)) if support[i][j]]
            i = rng.choice(occupied)
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            grid[i][j] = draw_cell(support[i][j], i, j, side)
            record = _sl_record(G, grid, side)
            if record is None:
                continue
            record["d"], record["word"] = _sl_probe(
                record["cols"], seed, 8)
            screened.append(record)
            if merit(record) > merit(current):
                current = record

        finalists = sorted(screened, key=merit, reverse=True)[:3]
        for record in finalists:
            reading, word = _sl_probe(record["cols"], seed+700001, 32)
            if reading < record["d"]:
                record["d"], record["word"] = reading, word
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    def quantum_probe(hx, hz, n, seed, trials):
        reading = n+1
        weights = []
        for t, (opposite, stabilizers) in enumerate(((hz, hx), (hx, hz))):
            d, word = _sl_probe(
                _sl_columns(opposite, n), seed+100003*t,
                trials, stabilizers)
            if word is not None:
                assert all((word & row).bit_count() % 2 == 0
                           for row in opposite)
                assert _sl_remainder(word, _sl_span(stabilizers)) != 0
                reading = min(reading, d)
                weights.append(d)
        return reading, tuple(weights)

    for restart in range(3):
        aa = component_pool(sa, "A")
        bb = component_pool(sb, "B")
        pairs = []
        for a in aa:
            for b in bb:
                hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 1440 and k >= 360
                        and 225*n <= 1000*k < 275*n):
                    continue
                key = (min(a["d"], b["d"]), a["d"]+b["d"],
                       -sum(a["cycles"])-sum(b["cycles"]))
                pairs.append({"A": a["grid"], "B": b["grid"],
                              "hx": hx, "hz": hz, "n": n, "k": k,
                              "component_d": (a["d"], b["d"]),
                              "key": key})
        pairs.sort(key=lambda r: r["key"], reverse=True)
        finalists = pairs[:2]
        if not finalists:
            continue
        for record in finalists:
            d1, w1 = quantum_probe(
                record["hx"], record["hz"], record["n"],
                diagnostic_seed+900001, 5)
            d2, w2 = quantum_probe(
                record["hx"], record["hz"], record["n"],
                diagnostic_seed+1900001, 9)
            record["logical_d"] = min(d1, d2)
            record["witnesses"] = w1+w2

        def final_merit(record):
            da, db = record["component_d"]
            reading = min(da, db, record["logical_d"])
            return (record["k"]*reading*reading, reading,
                    min(da, db), record["key"][2])

        chosen = max(finalists, key=final_merit)
        hx, hz = chosen["hx"], chosen["hz"]
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)
        return {"idx": idx, "card": (density, model, cohort),
                "cells": {"A": chosen["A"], "B": chosen["B"]},
                "k": chosen["k"],
                "component_readings": chosen["component_d"],
                "logical_reading": chosen["logical_d"],
                "verified_witness_weights": chosen["witnesses"]}
    raise ValueError("no matrix-group pair passed the exact rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete verified filling without post-screen draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Reallocated-kernel-pairs-v2 would yield five selected readings >=16 on Z29 (3,6)x(3,7); measured winners instead read 16,16,14,14,14, with all minimum witnesses in sector 1.
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
