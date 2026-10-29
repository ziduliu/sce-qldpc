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
CONCEPT = ("Hypothesis: on split-metacyclic C37:C3(s=10), order 111, A=(1,2), B=(2,5), shared-C3-annihilator quartic A operators produce five new codes in cell 0.350 with measured readings >=14, exceeding fitness 50.8. Exact ranks decide the rate; finite distance readings remain provisional upper bounds.")
GROUP = ("Preserve the split-metacyclic construction rule and resize only its parameters to m=37,r=3,s=10,q=111. The action is valid because 10^3=1 modulo 37. This exact group reached fitness 103.0 on ordinary (1,2)x(2,5) fillings, and stronger readings on its low-rate beds. Those measurements motivate a new constrained filling, not a distance-transfer claim.")
SHAPES = ("Change B from (1,3) to (2,5), retaining A=(1,2). Factor twelve gives n=1332, k>=333 and guaranteed rate 0.250; q_max=124. Components have 222 variables/111 checks and 555 variables/222 checks. Require A rank 74 with its explicit 37-dimensional common left-annihilator space and B rank 222; the complete native ranks must place every submitted code in [0.325,0.375). Parent weight-sixteen X minima occupied single-Acol sector-one strips across all three Bcols. Replace that B kernel with two check-block rows and five columns.")
FILLING = ("Five A/quotient policies cross four B support profiles. A cells [e,y,h,y^2*h] have four distinct terms and share the check functionals constant on left-C3 orbits; verify every functional against authoritative columns and require actual A rank 74. B row totals are seven/eight or eight/eight, column totals at most four, giving both CSS maxima <=12. Fresh long-order binomials and noncommuting Sidon larger B cells replace the archived cubic model. Scaffold-derived components undergo exact ranks and kernel-four exclusion, accumulated C3-fixed/unrestricted witness searches, actual witness-directed repairs and independent finalist challenges. Pairings use native k and genuine logical strips modulo original stabilizer spans. Neither component screening nor a search miss certifies quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_m37r3s10_12x25_commonC3_quartic_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # Same split-metacyclic rule; resized for the factor-twelve shape.
    if not valid_q(q):
        raise ValueError("C37:C3 requires order 111")
    m, r, s = 37, 3, 10
    powers = tuple(pow(s, b, m) for b in range(r))
    assert pow(s, r, m) == 1 and powers[1] != 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _retired_reallocated_ctx(rng, G, idx):
    """Inactive historical three-row experiment; not called by NEW_CTX."""
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


def NEW_CTX(rng, G, idx):
    """Twenty shape/rank-gain cards; every binary diagnostic is authoritative."""
    q = G.q
    assert q == 111 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    model, topology = card % 5, card // 5
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0x373125 + 1000003*idx))
    mask = (1 << q)-1
    sa = [[4, 4]]
    bplans = (
        ((1,1,1,2,2), (2,2,2,1,1)),
        ((0,1,2,2,2), (3,2,1,1,1)),
        ((1,1,2,2,2), (2,2,1,1,2)),
        ((1,1,2,2,2), (1,2,1,2,2)),
    )
    sb = [list(row) for row in bplans[topology]]
    assert _grid_weights(sa, sb) <= 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident, y = G.idx(G.e), G.idx((0, 1))
    yy = table[y][y]
    subgroup = (ident, y, yy)

    # Both action conventions are supplied by the fixed construction.
    gallery = [[[g] for g in G.elements]]
    xx, _, _ = lifted_matrices(G, gallery, [[[]]])
    _, zz, _ = lifted_matrices(G, [[[]]], gallery)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz]
              for g in range(q)],
    }
    del xx, zz

    def partition(side):
        remaining, groups = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {table[root][h] if side == "A"
                     else table[h][root] for h in subgroup}
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            groups.append(sum(1 << c for c in orbit))
        return groups

    # A columns commute with right translations; B with left translations.
    fixed_masks = {
        side: [word << (j*q) for j in range(count)
               for word in partition(side)]
        for side, count in (("A", 2), ("B", 5))
    }
    # Left-C3 check functionals explicitly annihilate every proposed A.
    annihilators = partition("B")
    assert len(annihilators) == 37

    def quartic(a, b):
        h = G.idx((a, b))
        cell = [ident, y, h, table[yy][h]]
        assert len(set(cell)) == 4
        return cell

    def sidon(cell):
        left = {table[inverse[a]][b] for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]] for a in cell for b in cell if a != b}
        return len(left) == len(right) == len(cell)*(len(cell)-1)

    def draw_b(i, j):
        count = sb[i][j]
        if count < 2:
            return rng.sample(range(q), count)
        if count == 2:
            b = rng.randrange(3)
            return [G.idx((a, b)) for a in rng.sample(range(37), 2)]
        mixed = ((1,1,1), (2,1,0))[
            (model + (i+j+cohort if model == 2 else 0)) % 2]
        occupancy = mixed if count == 3 else (
            (2,1,1) if model < 3 else (2,2,0))
        for _ in range(1024):
            fibers = rng.sample(range(3), 3)
            cell = [G.idx((a, b))
                    for b, number in zip(fibers, occupancy)
                    for a in rng.sample(range(37), number)]
            if not sidon(cell):
                continue
            relatives = [table[inverse[cell[0]]][g] for g in cell[1:]]
            if any(table[a][b] != table[b][a]
                   for t, a in enumerate(relatives) for b in relatives[:t]):
                return cell
        raise ValueError("new B occupancy policy exhausted")

    def fresh_b():
        return [[draw_b(i, j) for j in range(5)] for i in range(2)]

    def inspect(grid, side):
        height, width = len(grid)*q, len(grid[0])*q
        rows = [0]*height
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, word in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= word << (j*q)
        rank = gf2_rank(rows)
        if rank != (74 if side == "A" else 222):
            return None
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if side == "A":
            assert all((word & col).bit_count() % 2 == 0
                       for word in annihilators for col in columns)
        singles = set(columns)
        if 0 in singles or len(singles) != width:
            return None
        pairs = set()
        for at, col in enumerate(columns):
            for other in columns[:at]:
                value = col ^ other
                if value in singles or value in pairs:
                    return None
                pairs.add(value)
        uses, four, forced = {}, 0, 0
        adjacency = [0]*height
        for col in columns:
            ns = list(bits(col))
            d = len(ns)
            forced += d*(d-1)*(d-2)//6
            for at, a in enumerate(ns):
                adjacency[a] |= col ^ (1 << a)
                for b in ns[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        triangles = sum(
            (neighbors & adjacency[b] & (-1 << (b+1))).bit_count()
            for a, neighbors in enumerate(adjacency)
            for b in bits(neighbors & (-1 << (a+1))))
        excess = max(0, triangles-forced)
        return dict(grid=[[list(c) for c in row] for row in grid],
                    rows=rows, columns=columns, rank=rank,
                    cycles=(four, excess),
                    cost=(600+200*model)*four+excess,
                    d=width+1, words=[], fixed_words=[],
                    signature=tuple(tuple(sorted(c))
                                    for row in grid for c in row))

    def search(columns, diagnostic, trials, initial=()):
        """Persistent genuine null words; finite search misses prove nothing."""
        order = list(range(len(columns)))
        beam = sorted(set(initial), key=lambda w: (w.bit_count(), w))[:20]
        for word in beam:
            assert word and syndrome(columns, word) == 0
        for turn in range(trials):
            diagnostic.shuffle(order)
            if beam and turn % 4 == 3:
                active = list(bits(diagnostic.choice(beam[:5])))
                preferred = set(diagnostic.sample(
                    active, max(1, len(active)//2)))
                order.sort(key=lambda c: c not in preferred)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            if not basis:
                continue
            light = sorted(set(beam+basis),
                           key=lambda w: (w.bit_count(), w))[:14]
            grown = list(light)+beam
            threshold = light[0].bit_count()+6
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= threshold:
                        grown.append(candidate)
                if len(grown) > 256:
                    grown = sorted(set(grown),
                                   key=lambda w: (w.bit_count(), w))[:32]
            moves = basis+beam
            diagnostic.shuffle(moves)
            for initial_word in light[:5]:
                word = initial_word
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                grown.append(word)
            beam = sorted(set(grown),
                          key=lambda w: (w.bit_count(), w))[:20]
            if beam and beam[0].bit_count() <= 6:
                break
        return beam

    def query(rec, side, trials, turn):
        offset = 100003*int(side == "B")
        groups = fixed_masks[side]
        if "fixed_columns" not in rec:
            rec["fixed_columns"] = [
                syndrome(rec["columns"], word) for word in groups]
        fixed = search(
            rec["fixed_columns"], random.Random(seed+offset+1700003*turn),
            max(4, min(20, trials)), rec["fixed_words"])
        rec["fixed_words"] = fixed
        actuals = []
        for pattern in fixed[:8]:
            actual = 0
            for c in bits(pattern):
                actual ^= groups[c]
            assert actual.bit_count() == 3*pattern.bit_count()
            assert syndrome(rec["columns"], actual) == 0
            actuals.append(actual)
        words = search(
            rec["columns"], random.Random(seed+offset+700001*turn),
            trials, rec["words"]+actuals)
        assert words
        rec["words"] = words
        rec["d"] = min(rec["d"], words[0].bit_count())
        rec["fixed_d"] = min(
            (w.bit_count() for w in actuals),
            default=len(rec["columns"])+1)
        assert all(syndrome(rec["columns"], w) == 0 for w in words[:4])

    def merit(rec):
        return (min(20, rec["d"]), -rec["cost"],
                rec["d"], rec["rank"])

    def pool(side):
        records, seen = [], set()
        def add(grid):
            signature = tuple(tuple(sorted(c)) for row in grid for c in row)
            if signature in seen:
                return None
            seen.add(signature)
            rec = inspect(grid, side)
            if rec is not None:
                records.append(rec)
            return rec

        if side == "A":
            policies = (
                ((0,0),), ((0,2),), ((2,2),),
                ((0,0),(0,2)), ((0,2),(2,2)),
            )[model]
            # Enumerate genuinely different relative couplings. Normalize
            # common normal scaling and elementary ratio symmetries before
            # screening; do not populate the pool with their orbit copies.
            catalog = set()
            for b1, b2 in policies:
                for t in range(2, 37):
                    images = {
                        sign*pow(10, b, 37)*u % 37
                        for sign in (-1, 1) for b in range(3)
                        for u in (t, pow(t, -1, 37))}
                    representative = min(images)
                    if representative != 1:
                        catalog.add((b1, b2, representative))
            for b1, b2, t in sorted(catalog):
                add([[quartic(1, b1), quartic(t, b2)]])
        else:
            for _ in range(48):
                add(fresh_b())
        records.sort(key=lambda rec: rec["cost"])
        selected = records[:8]
        if not selected:
            raise ValueError("empty exact-rank common-C3 component pool")
        for rec in selected:
            query(rec, side, 10, 1)
        selected.sort(key=merit, reverse=True)
        if side == "B":
            current = selected[0]
            for step in range(16):
                grid = [[list(c) for c in row] for row in current["grid"]]
                word = rng.choice(current["words"][:6])
                j = rng.choice(sorted({c//q for c in bits(word)}))
                i = rng.choice([r for r in range(2) if sb[r][j]])
                grid[i][j] = draw_b(i, j)
                rec = add(grid)
                if rec is None:
                    continue
                rec["words"] = [w for w in current["words"]
                                if syndrome(rec["columns"], w) == 0]
                query(rec, side, 8, 20+step)
                selected.append(rec)
                if merit(rec) >= merit(current):
                    current = rec
        finalists = sorted(selected, key=merit, reverse=True)[:3]
        for rec in finalists:
            query(rec, side, 64+8*(cohort % 3), 101)
            query(rec, side, 32, 211)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logicals(a, b, hx, hz, n):
        """Actual nontrivial strips, reduced by ORIGINAL stabilizers."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in rec["words"][:6]:
                for location in range(5 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j+location)*q) for j in range(2))
                    else:
                        actual = word << (5*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            reducers = list(stabilizers)
            random.Random(seed+3100009+int(kind == "X")).shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:5]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers, key=lambda r: (r & actual).bit_count(),
                        reverse=True)[:24]
                    best_word = actual
                    for at, first in enumerate(nearby):
                        for second in nearby[:at]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < best_word.bit_count():
                                best_word = proposed
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool("A"), pool("B")
    best = None
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1332 and k >= 333
                    and 325*n <= 1000*k < 375*n):
                continue
            ar, br = _component_rows(hx, hz, q)
            assert ar == a["rows"] and br == b["rows"]
            assert [[len(c) for c in row] for row in A] == sa
            assert [[len(c) for c in row] for row in B] == sb
            assert max(row.bit_count() for row in hx+hz) <= 12
            logical_readings = logicals(a, b, hx, hz, n)
            detected = min(18, a["d"], b["d"], *logical_readings)
            key = (k*detected*detected, detected,
                   a["d"]+b["d"], -a["cost"]-b["cost"])
            answer = dict(
                idx=idx, cells={"A": A, "B": B}, k=k,
                card=(model, topology, cohort), target_cell=0.350,
                supports=(sa, sb), component_ranks=(a["rank"], b["rank"]),
                common_left_dimension=len(annihilators),
                component_upper_readings=(a["d"], b["d"]),
                fixed3_upper_readings=(a["fixed_d"], b["fixed_d"]),
                verified_logical_upper_readings=logical_readings,
                component_cycles=(a["cycles"], b["cycles"]),
                pairing_quality=key)
            if best is None or key > best[0]:
                best = (key, answer, hx, hz)
    if best is None:
        raise ValueError("common-C3 quartic pairings missed cell 0.350")
    _, answer, hx, hz = best
    assert css_ok(hx, hz)
    return answer


def _frob205_commonleft_reference_ctx(rng, G, idx):
    """Inactive order-205 reference; NEW_CTX above defines the new shape."""
    from math import exp

    q = G.q
    assert q == 205 and A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    card, cohort = idx % 20, idx // 20
    model, objective = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]

    # Every binary block comes from the authoritative construction.
    monomials = [[[g] for g in G.elements]]
    source, _, _ = lifted_matrices(G, monomials, [[[]]])
    mask = (1 << q)-1
    ablocks = [[(row >> (g*q)) & mask for row in source]
               for g in range(q)]
    _, source, _ = lifted_matrices(G, [[[]]], monomials)
    bblocks = [[(row >> (g*q)) & mask for row in source]
               for g in range(q)]
    del source

    # Regular component symmetries: A commutes with right multiplication;
    # B commutes with left multiplication. These are coordinate partitions,
    # not separately implemented lifts or submitted symmetry images.
    subgroup = [G.idx((0, b)) for b in range(5)]
    fixed_masks = {}
    for side, count in (("A", 2), ("B", 3)):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == "A" else table[h][root]
                for h in subgroup
            }
            assert len(orbit) == 5 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        fixed_masks[side] = [
            sum(1 << (j*q+c) for c in orbit)
            for j in range(count) for orbit in orbits
        ]

    profiles = (
        (0, 0, 1), (0, 1, 1), (0, 1, 2),
        (0, 1, 3), (0, 2, 4),
    )

    # A-left row operators preserve right-C5-fixed functionals.
    # Form the restriction solely by summing authoritative monomial rows.
    normal_mask = (1 << 41)-1
    ident = G.idx(G.e)

    def expand_functional(word):
        return sum(word << (41*b) for b in range(5))

    restricted = []
    for g in range(q):
        images = []
        for a in range(41):
            full = 0
            for b in range(5):
                full ^= ablocks[g][G.idx((a, b))]
            image = full & normal_mask
            assert full == expand_functional(image)
            images.append(image)
        restricted.append(images)

    def left_dependencies(images):
        """Exact coefficient dependencies of scaffold-derived row images."""
        pivots, basis = {}, []
        for r, value in enumerate(images):
            word = 1 << r
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    memberships = {}
    for g in range(q):
        if g == ident:
            continue
        for h in range(g):
            if h == ident:
                continue
            cell = tuple(sorted((ident, g, h)))

            # Multiplying every term on the right is a column gauge.
            # Keep one representative, not several copies of one operator.
            representative = min(
                tuple(sorted(table[x][inverse[root]] for x in cell))
                for root in cell)
            if cell != representative:
                continue

            left = {table[inverse[x]][y]
                    for x in cell for y in cell if x != y}
            right = {table[x][inverse[y]]
                     for x in cell for y in cell if x != y}
            if len(left) != 6 or len(right) != 6:
                continue
            u = table[inverse[cell[0]]][cell[1]]
            v = table[inverse[cell[0]]][cell[2]]
            if table[u][v] == table[v][u]:
                continue

            images = [
                restricted[cell[0]][a] ^ restricted[cell[1]][a]
                ^ restricted[cell[2]][a] for a in range(41)]
            basis = left_dependencies(images)
            # This is a proposal restriction, not an exclusion claim.
            # Four restricted dependencies target a twenty-dimensional
            # regular-module defect, independently checked below.
            if len(basis) != 4:
                continue
            values = [0]
            for word in basis:
                values += [old ^ word for old in values]
            for functional in values[1:]:
                value = 0
                for a in bits(functional):
                    value ^= images[a]
                assert value == 0
                memberships.setdefault(functional, []).append(cell)

    shared_catalog = [
        (functional, tuple(sorted(set(cells))))
        for functional, cells in memberships.items()
        if len(set(cells)) >= 2
    ]

    def catalog_key(item):
        functional, cells = item
        weight = functional.bit_count()
        quotient_shapes = set()
        for cell in cells:
            bs = [G.elements[g][1] for g in cell]
            quotient_shapes.add(min(
                tuple(sorted((b-phase) % 5 for b in bs))
                for phase in range(5)))
        preference = (
            (-len(cells), weight),
            (weight, -len(cells)),
            (abs(weight-20), -len(cells)),
            (-weight, -len(cells)),
            (-len(quotient_shapes), -len(cells)),
        )[model]
        return preference + (functional,)

    shared_catalog.sort(key=catalog_key)
    a_frames = {}

    def profile_for(side, j):
        return (model + objective*j
                + int(side == "B")*(objective+1) + cohort) % 5

    def draw_cell(side, j):
        profile = profiles[profile_for(side, j)]
        for _ in range(1024):
            phase = rng.randrange(5)
            native = [(rng.randrange(41), (b+phase) % 5)
                      for b in profile]
            cell = [G.idx(g) for g in native]
            if len(set(cell)) != 3:
                continue
            left = {table[inverse[a]][b]
                    for a in cell for b in cell if a != b}
            right = {table[a][inverse[b]]
                     for a in cell for b in cell if a != b}
            if len(left) != 6 or len(right) != 6:
                continue
            u = table[inverse[cell[0]]][cell[1]]
            v = table[inverse[cell[0]]][cell[2]]
            if table[u][v] == table[v][u]:
                continue
            return cell
        raise ValueError("noncommuting trinomial profile exhausted")

    def fresh_component(side, trial):
        count = 2 if side == "A" else 3
        # Explicit full-rank controls compete with common-module joins.
        # An empty catalog is not silently interpreted as an exclusion.
        if side != "A" or not shared_catalog or trial % 8 == 7:
            return [draw_cell(side, j) for j in range(count)]
        limit = min(256, len(shared_catalog))
        turn = (31*objective + 43*cohort + 5*trial + model) % limit
        functional, options = shared_catalog[turn]
        cells = [list(cell) for cell in rng.sample(options, 2)]
        signature = tuple(tuple(sorted(cell)) for cell in cells)
        a_frames[signature] = (functional, options)
        return cells

    def replacement_cell(side, j, current):
        frame = current["frame"]
        if side != "A" or frame is None:
            return draw_cell(side, j)
        functional, options = frame
        other = tuple(sorted(current["cells"][1-j]))
        old = tuple(sorted(current["cells"][j]))
        available = [cell for cell in options
                     if cell != other and cell != old]
        if not available:
            available = [cell for cell in options if cell != other]
        cells = [list(cell) for cell in current["cells"]]
        cells[j] = list(rng.choice(available))
        signature = tuple(tuple(sorted(cell)) for cell in cells)
        a_frames[signature] = frame
        return cells[j]

    def inspect(cells, side):
        """GF2-linear sums of scaffold-built monomial blocks."""
        blocks = ablocks if side == "A" else bblocks
        width = len(cells)*q
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, word in enumerate(blocks[g]):
                    rows[r] ^= word << (j*q)
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        assert all(c.bit_count() == 3 for c in columns)

        adjacency = [0]*q
        uses, four = {}, 0
        for column in columns:
            checks = list(bits(column))
            for t, a in enumerate(checks):
                adjacency[a] |= column ^ (1 << a)
                for b in checks[:t]:
                    key = (b, a)
                    old = uses.get(key, 0)
                    four += old
                    uses[key] = old+1

        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        excess = max(0, triangles-width)
        peak = max(map(int.bit_count, adjacency))
        bad = width-len(set(columns))
        extra = (excess, excess+4*peak,
                 2*excess+peak, excess+12*peak)[objective]
        cost = 1000000*bad + (700+300*objective)*four + extra
        signature = tuple(tuple(sorted(c)) for c in cells)
        frame = a_frames.get(signature) if side == "A" else None
        rank = gf2_rank(rows)
        wanted = q-20 if frame is not None else q
        rank_valid = rank == wanted
        if frame is not None:
            expanded = expand_functional(frame[0])
            assert expanded
            assert all((expanded & column).bit_count() % 2 == 0
                       for column in columns)
        return dict(
            cells=[list(c) for c in cells], rows=rows, columns=columns,
            adjacency=adjacency, cycles=(four, excess, peak), cost=cost,
            signature=signature, frame=frame, rank=rank,
            rank_valid=rank_valid,
        )

    def prepare(rec, side):
        """Exact rank, expanded annihilator and kernel-through-four gates."""
        if not rec["rank_valid"]:
            return False
        if rec["frame"] is not None:
            assert side == "A" and rec["rank"] == q-20
            expanded = expand_functional(rec["frame"][0])
            assert all((column & expanded).bit_count() % 2 == 0
                       for column in rec["columns"])
        elif rec["rank"] != q:
            return False
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in bits(frontier):
                neighbors |= rec["adjacency"][r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << q)-1:
            return False

        columns = rec["columns"]
        singles, pairs = {}, {}
        incidence = [0]*q
        for c, value in enumerate(columns):
            if not value or value in singles:
                return False
            word = 1 << c
            singles[value] = word
            for r in bits(value):
                incidence[r] |= word
        for a, value in enumerate(columns):
            for b in range(a):
                key = value ^ columns[b]
                if key in singles or key in pairs:
                    return False
                pairs[key] = (1 << a) | (1 << b)
        # Odd column degrees additionally rule out every odd-weight kernel.
        rec.update(singles=singles, pairs=pairs, incidence=incidence,
                   words=set(), d=len(columns)+1, fixed_bank=[])
        return True

    def note(rec, word):
        if not word:
            return
        weight = word.bit_count()
        rec["d"] = min(rec["d"], weight)
        if weight <= rec["d"]+6:
            rec["words"].add(word)
            if len(rec["words"]) > 96:
                rec["words"] = set(sorted(
                    rec["words"], key=lambda w: (w.bit_count(), w))[:32])

    def completion(rec, budget, offset, cap=18):
        """Budgeted parity completion; a budget stop proves no exclusion."""
        diagnostic = random.Random(seed+offset)
        columns = rec["columns"]
        width = len(columns)
        priorities = [diagnostic.random() for _ in columns]
        roots = list(range(0, width, q))
        diagnostic.shuffle(roots)
        all_variables = (1 << width)-1

        for root in roots:
            remaining = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    note(rec, word)
                    return
                left = min(cap, rec["d"])-word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = rec["singles"].get(value, 0)
                if one and one & allowed == one:
                    note(rec, word | one)
                pair = rec["pairs"].get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(rec, word | pair)
                if left <= 2:
                    return

                chosen, smallest = 0, width+1
                for r in bits(value):
                    available = rec["incidence"][r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                choices = list(bits(chosen))
                choices.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in choices:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            allowed = all_variables & ~((1 << (root+1))-1)
            visit(columns[root], 1 << root, allowed)

    def null_search(columns, trials, diagnostic, initial=()):
        """Actual null words from circuit cancellation and descent."""
        order = list(range(len(columns)))
        retained = sorted(set(initial), key=int.bit_count)[:24]
        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            if not basis:
                continue

            light = sorted(set(retained+basis), key=int.bit_count)[:20]
            additions = []
            threshold = light[0].bit_count()+4
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= threshold:
                        additions.append(candidate)
            moves = retained+basis
            diagnostic.shuffle(moves)
            for initial_word in light[:6]:
                word = initial_word
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                additions.append(word)
            retained = sorted(set(retained+light+additions),
                              key=lambda w: (w.bit_count(), w))[:24]
            if retained and retained[0].bit_count() <= 8:
                break
        return retained

    def challenge(rec, side, trials, offset):
        """Retain unrestricted and C5-fixed genuine component witnesses."""
        masks = fixed_masks[side]
        if "fixed_columns" not in rec:
            rec["fixed_columns"] = [
                syndrome(rec["columns"], word) for word in masks
            ]
        fixed = null_search(
            rec["fixed_columns"], max(4, min(12, trials)),
            random.Random(seed+offset+5000011), rec["fixed_bank"])
        rec["fixed_bank"] = fixed
        for word in fixed:
            actual = 0
            for c in bits(word):
                actual ^= masks[c]
            assert actual.bit_count() == 5*word.bit_count()
            assert syndrome(rec["columns"], actual) == 0
            note(rec, actual)

        found = null_search(
            rec["columns"], trials, random.Random(seed+offset),
            rec["words"])
        for word in found:
            note(rec, word)
        rec["words"] = set(sorted(
            rec["words"], key=lambda w: (w.bit_count(), w))[:32])
        assert rec["words"]
        for word in sorted(rec["words"], key=int.bit_count)[:4]:
            assert syndrome(rec["columns"], word) == 0
        rec["fixed_d"] = min(
            (5*w.bit_count() for w in fixed),
            default=len(rec["columns"])+1)

    def merit(rec, side):
        multiplicity = sum(w.bit_count() == rec["d"]
                           for w in rec["words"])
        # The parent repeatedly measured sixteen. Once that diagnostic
        # threshold ties, prefer verified dimension gain to a more
        # optimistic component reading. This is not a distance lower bound.
        defect = q-rec["rank"] if side == "A" else 0
        cap = 32 if side == "A" else 24
        extra = rec["fixed_d"] if objective == 3 else 0
        return (min(16, rec["d"]), defect, min(cap, rec["d"]),
                -multiplicity, extra, -rec["cost"])

    def pool(side):
        count = 2 if side == "A" else 3
        offset = 100003 if side == "A" else 200003
        bank, seen = {}, set()

        def initial_key(rec):
            return (not rec["rank_valid"],
                    side == "A" and rec["frame"] is None,
                    rec["cost"])

        def remember(rec):
            signature = rec["signature"]
            if signature not in bank:
                bank[signature] = rec
            if len(bank) > 10:
                worst = max(bank, key=lambda key: initial_key(bank[key]))
                del bank[worst]

        starts = []
        for trial in range(32):
            rec = inspect(fresh_component(side, trial), side)
            starts.append(rec)
            remember(rec)
        starts.sort(key=initial_key)

        for initial in starts[:2]:
            current = initial
            for step in range(72):
                cells = [list(c) for c in current["cells"]]
                j = rng.randrange(count)
                cells[j] = replacement_cell(side, j, current)
                proposed = inspect(cells, side)
                if not proposed["rank_valid"]:
                    continue
                delta = proposed["cost"]-current["cost"]
                temperature = q*(6*(1-step/72.0)**2+0.1)
                if delta <= 0 or rng.random() < exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(proposed)

        selected = []
        for rec in sorted(bank.values(), key=initial_key):
            seen.add(rec["signature"])
            if not prepare(rec, side):
                continue
            completion(rec, 5000, offset+17)
            challenge(rec, side, 8, offset)
            selected.append(rec)
            if len(selected) >= 5:
                break
        if not selected:
            raise ValueError("no full-rank kernel-four-free trinomial component")

        selected.sort(key=lambda rec: merit(rec, side), reverse=True)
        current = selected[0]
        keepers = selected[:3]
        historical = set().union(*(rec["words"] for rec in selected))

        # Repair actual cells used by a retained witness, not group gauges.
        for step in range(12):
            proposals = []
            active = sorted(current["words"], key=int.bit_count)[:8]
            for _ in range(2):
                cells = [list(c) for c in current["cells"]]
                word = rng.choice(active)
                occupied = sorted({c//q for c in bits(word)})
                j = rng.choice(occupied)
                cells[j] = replacement_cell(side, j, current)
                rec = inspect(cells, side)
                if not rec["rank_valid"] or rec["signature"] in seen:
                    continue
                seen.add(rec["signature"])
                if rec["cycles"][0] > current["cycles"][0]+q:
                    continue
                survived = sum(
                    syndrome(rec["columns"], w) == 0
                    for w in historical)
                proposals.append((survived, rec["cost"], rec))
            proposals.sort(key=lambda item: item[:2])

            for _, _, rec in proposals:
                if not prepare(rec, side):
                    continue
                for known in historical:
                    if syndrome(rec["columns"], known) == 0:
                        note(rec, known)
                completion(rec, 3500, offset+700001+7919*step)
                challenge(rec, side, 6, offset+1000003+7919*step)
                historical.update(rec["words"])
                if len(historical) > 80:
                    historical = set(sorted(
                        historical, key=int.bit_count)[:80])
                keepers.append(rec)
                if merit(rec, side) > merit(current, side):
                    current = rec
                keepers.sort(key=lambda r: merit(r, side), reverse=True)
                del keepers[6:]

        unique = {rec["signature"]: rec for rec in keepers}
        finalists = sorted(
            unique.values(), key=lambda r: merit(r, side),
            reverse=True)[:3]
        for rec in finalists:
            completion(rec, 24000, offset+1700021)
            challenge(rec, side, 48, offset+1900003)
            # Independent retesting retains every earlier binding witness.
            completion(rec, 16000, offset+2700011)
            challenge(rec, side, 24, offset+2900009)
        finalists.sort(key=lambda r: merit(r, side), reverse=True)
        return finalists[:2]

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_challenge(a, b, hx, hz, n):
        """Genuine logical upper readings in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(
                rec["words"], key=lambda w: (w.bit_count(), w))[:6]
            candidates = set()
            for word in words:
                assert syndrome(rec["columns"], word) == 0
                for fixed in range(3 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j+fixed)*q) for j in range(2))
                    else:
                        actual = word << (3*fixed*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            reading = n+1
            diagnostic = random.Random(
                seed+(4100011 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            diagnostic.shuffle(reducers)
            for actual in sorted(
                    candidates, key=lambda w: (w.bit_count(), w))[:4]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best_word, best_weight = actual, actual.bit_count()
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word, best_weight = proposed, weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool("A"), pool("B")
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        min(16, ab[0]["d"], ab[1]["d"]),
        q-ab[0]["rank"],
        min(32, ab[0]["d"])+min(24, ab[1]["d"]),
        -ab[0]["cost"]-ab[1]["cost"]), reverse=True)

    best = None
    for a, b in proposals:
        A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
        B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1435 and k >= 410
                and 275*n <= 1000*k < 325*n):
            continue

        # Check every assembly against the complete native construction.
        actual_a = [
            sum(((row >> (3*j*q)) & mask) << (j*q)
                for j in range(2))
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << (3*q))-1) for row in hz[:q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert [[len(c) for c in row] for row in A] == sa
        assert [[len(c) for c in row] for row in B] == sb
        assert all(len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        assert max(row.bit_count() for row in hx) == 9
        assert max(row.bit_count() for row in hz) == 12

        logicals = logical_challenge(a, b, hx, hz, n)
        detected = min(16, a["d"], b["d"], *logicals)
        quality = (
            k*detected*detected, detected,
            min(32, a["d"])+min(24, b["d"]),
            -a["cost"]-b["cost"])
        answer = {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (model, objective, cohort),
            "component_upper_readings": (a["d"], b["d"]),
            "fixed5_upper_readings": (a["fixed_d"], b["fixed_d"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "component_ranks": (a["rank"], b["rank"]),
            "common_left_functional": (
                a["frame"][0] if a["frame"] is not None else None),
            "shared_catalog_size": len(shared_catalog),
            "verified_logical_upper_readings": logicals,
            "pairing_quality": quality,
        }
        if best is None or quality > best[0]:
            best = (quality, answer, hx, hz)

    if best is None:
        raise ValueError("common-left Frobenius-205 pairings missed cell 0.300")
    quality, answer, hx, hz = best
    assert css_ok(hx, hz)
    return answer


def CELL(rng, G, side, i, j, ctx):
    """Return the complete shape/rank-gain selected cell without new draws."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == ctx["supports"][int(side == "B")][i][j]
    assert len(cell) <= 5 and len(set(cell)) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Row-six/row-nine density reallocation and deeper kernel-pair screening yield five selected readings >=16; the measured current program returned 14,14,16,14,16, all with sector-1 minimum witnesses.
# REFUTED: Order-84 quotient-card six-witness selection breaks the high-rate distance-six plateau; all five measured parent codes returned six, k=756, fitness 19.059, and abundant sector-one minima.
# REFUTED: Full-rank cubic components with unrestricted/C5-fixed witness repair produce five readings >=18 on C41:C5 (1,2)x(1,3); all five measured winners instead retained k=410, reading sixteen and sector-one X minima.
# REFUTED: Common-left cubic proposal screening on C41:C5 yields five measured k=450 codes; all five reported current winners instead had k=410 and reading sixteen.
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
