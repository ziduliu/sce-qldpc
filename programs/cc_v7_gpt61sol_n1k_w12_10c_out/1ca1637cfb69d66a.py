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
CONCEPT = ("On S4 x C3, order 72, A=(1,3), B=(2,5), shared central-coordinate prescriptions and entirely degree-three components should produce five new readings >=12 with 434<=k<=458, improving cell 0.350 beyond 50.824. Measured failure refutes this hypothesis; finite searches certify no distance.")
GROUP = ("Jump from the order-84 metacyclic group to the direct product of permutation S4 with central C3, order 72, centre three and fifteen conjugacy classes. This group/shape bed is absent from the displayed BEDS. Central coordinates (0,1,2) propose common annihilators on A; authoritative binary ranks test the proposal. Related cubic-component beds reached readings twelve, but those provisional upper bounds establish no distance transfer.")
SHAPES = ("Change to A=(1,3), B=(2,5): q=72 gives n=1224, k>=432 and guaranteed rate 6/17=0.352941 in cell 0.350; q_max=88. Components have 216 variables/72 checks and 360 variables/144 checks, nominal rates 2/3 and 3/5. Parent weight-six minima occupied sector-one Acol1/2 across Bcol0..2. Replace both supports and remove every degree-two B column. Actual full ranks enforce 434<=k<=458.")
FILLING = ("Twenty cards cross five A permutation preferences with four B support topologies. A has three central-charge trinomials; every B column has degree three and its row totals are seven/eight. HX weight is twelve and HZ at most eleven. Native bidirectional Sidon proposals, authoritative component geometry, exact kernel-through-four rejection, localized enumeration, C2/C3/V4-fixed restrictions and independent retained-witness searches challenge both components. Fresh unrestricted A alternatives preserve the central prescription when preferred catalogs fail. Select complete pairings by actual k and original-span-verified logical strips reduced with original stabilizers, capping selection readings at twelve. Final native assembly, distinctness, weights, rate and CSS are binding; no finite search certifies distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_S4xC3_13x25_charge3_allcubic_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 72

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("S4 x C3 requires order 72")
    # (p0,p1,p2,p3,z): permutation p, and independent central C3.
    els = [p + (z,) for z in range(3)
           for p in permutations(range(4))]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) % 4 for i in range(4)) + (int(g[4]) % 3,)

    def mul(g, h):
        g, h = red(g), red(h)
        return tuple(g[h[i]] for i in range(4)) + ((g[4]+h[4]) % 3,)

    def inv(g):
        g = red(g)
        p = g[:4]
        return tuple(p.index(i) for i in range(4)) + ((-g[4]) % 3,)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0,1,2,3,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
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
    """Twenty S4 x C3 charge/support cards with binding native-rank tests."""
    import math
    q = G.q
    assert q == 72 and A_SHAPE == (1,3) and B_SHAPE == (2,5)
    card, cohort = idx % 20, idx // 20
    model, topology = divmod(card, 4)
    seed = rng.randrange(1 << 62) ^ (1000003*idx + 0x743350)
    mask = (1 << q)-1
    table = [[G.idx(G.mul(g,h)) for h in G.elements] for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

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

    def columns_of(rows, width):
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return columns

    class Probe:
        def __init__(self):
            self.q, self.e, self.elements = q, ident, list(range(q))
        def idx(self, g):
            return g
        def mul(self, g, h):
            return table[g][h]
        def inv(self, g):
            return inverse[g]

    P = Probe()
    blocks = {"A": [], "B": []}
    # Cache authoritative monomial blocks in small batches.
    for start in range(0, q, 12):
        ids = list(range(start, min(q,start+12)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            for side, rows in (("A",xx), ("B",zz)):
                blocks[side].append([(row >> (j*q)) & mask for row in rows])

    bprofiles = (
        ((1,1,1,2,2), (2,2,2,1,1)),
        ((3,1,1,1,1), (0,2,2,2,2)),
        ((3,0,1,1,2), (0,3,2,2,1)),
        ((0,0,2,2,3), (3,3,1,1,0)),
    )
    supports = {"A": [[3,3,3]],
                "B": [list(row) for row in bprofiles[topology]]}
    assert [sum(row) for row in supports["B"]] == [7,8]
    assert [sum(row[j] for row in supports["B"]) for j in range(5)] == [3]*5
    assert _grid_weights(supports["A"], supports["B"]) == 12 <= W_MAX
    assert _no_dead_lines(supports["A"], supports["B"])

    permutations = [g[:4] for g in G.elements[:24]]
    parity = {p: sum(p[i] > p[j] for i in range(4)
                    for j in range(i+1,4)) % 2 for p in permutations}
    even = [p for p in permutations if not parity[p]]
    odd = [p for p in permutations if parity[p]]
    four_cycles = []
    for p in permutations:
        visited, at = set(), 0
        while at not in visited:
            visited.add(at)
            at = p[at]
        if len(visited) == 4:
            four_cycles.append(p)

    def sidon(ids):
        if len(set(ids)) != len(ids):
            return False
        left = {table[inverse[a]][b] for a in ids for b in ids if a != b}
        right = {table[a][inverse[b]] for a in ids for b in ids if a != b}
        return len(left) == len(right) == len(ids)*(len(ids)-1)

    def fresh_a(turn):
        preference = model if turn < 48 else 0
        pools = ((permutations,permutations), (even,even),
                 (odd,odd), (odd,even), (four_cycles,four_cycles))
        first, second = pools[preference]
        for _ in range(256):
            a, b = rng.choice(first), rng.choice(second)
            ids = [ident, G.idx(a+(1,)), G.idx(b+(2,))]
            if (sidon(ids) and
                    table[ids[1]][ids[2]] != table[ids[2]][ids[1]]):
                return ids
        # Still a fresh central-charge proposal, not an archived fallback.
        for _ in range(1024):
            ids = [ident, G.idx(rng.choice(permutations)+(1,)),
                   G.idx(rng.choice(permutations)+(2,))]
            if sidon(ids):
                return ids
        raise ValueError("central-charge cubic catalog exhausted")

    def fresh_cell(count):
        for _ in range(256):
            ids = rng.sample(range(q), count)
            if sidon(ids):
                return ids
        raise ValueError("native Sidon cell exhausted")

    def fresh(side, turn):
        return [[fresh_a(turn) if side == "A" else fresh_cell(count)
                 for count in row] for row in supports[side]]

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        columns = columns_of(rows, nc*q)
        assert all(c.bit_count() == 3 for c in columns)
        defect = nr*q-gf2_rank(rows)
        if side == "A" and not 2 <= defect <= 8:
            return None
        uses, four = {}, 0
        for value in columns:
            ns = list(bits(value))
            for at, a in enumerate(ns):
                for b in ns[:at]:
                    key = b,a
                    four += uses.get(key,0)
                    uses[key] = uses.get(key,0)+1
        return dict(grid=[[list(c) for c in row] for row in grid],
                    rows=rows, columns=columns, defect=defect,
                    four=four, words=[], reading=nc*q+1)

    def free4(columns):
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return False
        pairs = set()
        for i, value in enumerate(columns):
            for other in columns[:i]:
                combined = value ^ other
                if combined in singles or combined in pairs:
                    return False
                pairs.add(combined)
        return True

    # Restriction masks are coordinate orbits of VALID component actions.
    subgroups = (
        (ident, G.idx((1,0,2,3,0))),
        (ident, G.idx((0,1,2,3,1)), G.idx((0,1,2,3,2))),
        tuple(G.idx(p+(0,)) for p in
              ((0,1,2,3),(1,0,3,2),(2,3,0,1),(3,2,1,0))),
    )
    partitions = {"A": [], "B": []}
    for side, nc in (("A",3), ("B",5)):
        for subgroup in subgroups:
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {table[root][h] if side == "A" else table[h][root]
                         for h in subgroup}
                assert len(orbit) == len(subgroup) and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(orbit)
            partitions[side].append([
                sum(1 << (j*q+c) for c in orbit)
                for j in range(nc) for orbit in orbits])

    def dependencies(columns):
        pivots, basis = {}, []
        for c, value in enumerate(columns):
            word = 1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = value,word
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    def search(columns, trials, local, initial=()):
        """Retain genuine circuits; independent calls can only lower readings."""
        retained = set(initial)
        order = list(range(len(columns)))
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = value,word
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            light = sorted(set(basis) | retained, key=int.bit_count)[:16]
            grown = set(light) | retained
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= 24:
                        grown.add(candidate)
            moves = basis + list(retained)
            local.shuffle(moves)
            for word in light[:4]:
                for sweep in range(2):
                    before = word.bit_count()
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    grown.add(word)
                    if word.bit_count() == before:
                        break
            retained = set(sorted(grown,
                key=lambda w: (w.bit_count(),w))[:20])
        assert retained and all(syndrome(columns,w) == 0 for w in retained)
        return sorted(retained, key=lambda w: (w.bit_count(),w))

    def diagnose(record, side, trials, turn):
        local = random.Random(seed+100003*int(side == "B")+700001*turn)
        words = list(record["words"])
        for groups in partitions[side]:
            compressed = [syndrome(record["columns"], group) for group in groups]
            found = search(compressed, max(3,trials//4), local)
            for word in found[:4]:
                actual = 0
                for c in bits(word):
                    actual ^= groups[c]
                assert actual and syndrome(record["columns"],actual) == 0
                words.append(actual)
        # Exact single-column enumeration when its nullity is small.
        if not record.get("localized",False):
            for j in range(len(record["grid"][0])):
                block = record["columns"][j*q:(j+1)*q]
                basis = dependencies(block)
                if len(basis) <= 12:
                    word, best = 0, None
                    for t in range(1,1 << len(basis)):
                        word ^= basis[(t & -t).bit_length()-1]
                        if best is None or word.bit_count() < best.bit_count():
                            best = word
                    if best is not None:
                        words.append(best << (j*q))
                else:
                    words.extend(w << (j*q) for w in
                                 sorted(basis,key=int.bit_count)[:4])
            record["localized"] = True
        record["words"] = search(record["columns"],trials,local,words)
        record["reading"] = min(record["reading"],
                                record["words"][0].bit_count())

    def merit(record, side):
        return (min(12,record["reading"]),
                record["defect"] if side == "A" else -record["defect"],
                -record["four"], record["reading"])

    def pool(side):
        records, seen = [], set()
        def remember(grid):
            signature = tuple(tuple(sorted(cell))
                              for row in grid for cell in row)
            if signature in seen:
                return None
            seen.add(signature)
            rec = inspect(grid,side)
            if rec is not None:
                records.append(rec)
            return rec
        for turn in range(80):
            remember(fresh(side,turn))
        if not records:
            raise ValueError("empty central-charge geometry portfolio")
        records.sort(key=lambda r: (r["four"],
            -r["defect"] if side == "A" else r["defect"]))
        positions = [(i,j) for i,row in enumerate(supports[side])
                     for j,count in enumerate(row) if count]
        for chain, initial in enumerate(records[:2]):
            current = initial
            for step in range(40):
                grid = [[list(c) for c in row] for row in current["grid"]]
                i,j = rng.choice(positions)
                grid[i][j] = (fresh_a(step) if side == "A" else
                              fresh_cell(supports[side][i][j]))
                proposed = remember(grid)
                if proposed is None:
                    continue
                delta = proposed["four"]-current["four"]
                temperature = q*(1-step/40)**2+1
                if delta <= 0 or rng.random() < math.exp(-delta/temperature):
                    current = proposed
        records.sort(key=lambda r: (r["four"],
            -r["defect"] if side == "A" else r["defect"]))
        screened = []
        for rec in records:
            if not free4(rec["columns"]):
                continue
            # Odd degree excludes odd-weight component kernels as well.
            diagnose(rec,side,8,1+len(screened))
            screened.append(rec)
            if len(screened) >= 12:
                break
        if not screened:
            raise ValueError("no all-cubic component passed kernel-four gates")
        finalists = sorted(screened,key=lambda r: merit(r,side),reverse=True)[:3]
        for at,rec in enumerate(finalists):
            diagnose(rec,side,24,101+at)
            diagnose(rec,side,40+4*(cohort % 3),201+at)
        finalists.sort(key=lambda r: merit(r,side),reverse=True)
        return finalists

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

    def outside(word,pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_reading(a,b,hx,hz,n):
        """Only original-span-verified logicals supply quantum upper readings."""
        reading = n+1
        for side,rec,opposite,stabilizers in (
                ("A",a,hx,hz), ("B",b,hz,hx)):
            pivots, candidates = span(stabilizers), set()
            for word in rec["words"][:6]:
                for fixed in range(5 if side == "A" else 3):
                    actual = (sum(
                        ((word >> (j*q)) & mask) << ((5*j+fixed)*q)
                        for j in range(3)) if side == "A"
                        else word << (fixed*5*q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual,pivots):
                        candidates.add(actual)
            for actual in sorted(candidates,key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:20]
                    winner = actual
                    for i,first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual,pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading,actual.bit_count())
        return reading

    aa,bb = pool("A"),pool("B")
    best = None
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in c] for c in row] for row in a["grid"]]
            B = [[[G.elements[g] for g in c] for c in row] for row in b["grid"]]
            hx,hz,n = lifted_matrices(G,A,B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            assert n == 1224 and k >= 432
            if not 434 <= k <= 458:
                continue
            ar,br = _component_rows(hx,hz,q)
            assert ar == a["rows"] and br == b["rows"]
            dq = logical_reading(a,b,hx,hz,n)
            detected = min(12,a["reading"],b["reading"],dq)
            quality = (k*detected*detected,detected,
                       -a["four"]-b["four"])
            if best is None or quality > best[0]:
                best = quality,A,B,hx,hz,k,a,b,dq
    if best is None:
        raise ValueError("all-cubic pairings missed the actual 0.350 rank gate")
    quality,A,B,hx,hz,k,a,b,dq = best
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert css_ok(hx,hz)
    return dict(idx=idx,cells={"A":A,"B":B},k=k,
                card=(model,topology,cohort),target_cell=0.350,
                component_defects=(a["defect"],b["defect"]),
                component_readings=(a["reading"],b["reading"]),
                component_four_cycles=(a["four"],b["four"]),
                verified_logical_upper_reading=None if dq > 1224 else dq,
                selection_reading_cap=12,quality=quality)


def _retired_subgroup_ctx(rng, G, idx):
    """Inactive order-84 subgroup portfolio; not called on S4 x C3."""
    q, width = G.q, 4*G.q
    assert q == 84 and A_SHAPE == B_SHAPE == (1, 4)
    card, cohort = idx % 20, idx // 20
    model, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    plans = (
        ((2,2,2,3), (2,2,2,3)),
        ((2,2,3,3), (2,2,2,2)),
        ((2,2,2,4), (2,2,2,2)),
        ((2,2,2,2), (2,2,2,2)),
        ((1,2,3,3), (2,2,2,3)),
    )
    for sa, sb in plans:
        assert _grid_weights([list(sa)], [list(sb)]) <= min(12, W_MAX)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    # Cache only authoritative monomial blocks; assembly is GF2 linearity.
    gallery = [[[g] for g in G.elements]]
    xx, _, _ = lifted_matrices(G, gallery, [[[]]])
    _, zz, _ = lifted_matrices(G, [[[]]], gallery)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx] for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz] for g in range(q)],
    }
    del xx, zz

    def columns_of(rows, n):
        columns = [0]*n
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return columns

    def component(cells, side):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, value in enumerate(blocks[side][G.idx(g)]):
                    rows[r] ^= value << (j*q)
        return rows, columns_of(rows, width)

    def syndrome(columns, word):
        result = 0
        for c in bits(word):
            result ^= columns[c]
        return result

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

    def residue(word, pivots):
        answer = 0
        while word:
            lead = word.bit_length()-1
            old = pivots.get(lead)
            if old is None:
                bit = 1 << lead
                answer ^= bit
                word ^= bit
            else:
                word ^= old
        return answer

    def four_cost(columns):
        uses, cost = {}, 0
        for value in columns:
            ns = list(bits(value))
            for i, a in enumerate(ns):
                for b in ns[:i]:
                    key = (b, a)
                    cost += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        return cost

    def legal_bins(ids, connected):
        signed = ids + [inverse[g] for g in ids]
        if len(set(signed)) != 2*len(ids) or ident in signed:
            return False
        products = set()
        for g in signed:
            for h in signed:
                if h == inverse[g]:
                    continue
                value = table[g][h]
                if value == ident or value in products:
                    return False
                products.add(value)
        if not connected:
            return True
        reached, pending = {ident}, [ident]
        for value in pending:
            for g in signed:
                image = table[value][g]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    def draw_bins(count, turn):
        # Models 2/3 propose proper NONABELIAN subgroups, not the
        # charge-zero commuting subgroup of the earlier failed model.
        coordinate_sets = (
            (1,3,5,7,9,11), (0,1,2,5,7,10),
            (0,2,4,6,8,10), (0,4,8), tuple(range(12)),
        )
        late = turn >= 48
        coordinates = coordinate_sets[0 if late else model]
        connected = late or model == 0
        for _ in range(96):
            ids = [G.idx((rng.randrange(7), rng.choice(coordinates)))
                   for _ in range(count)]
            if legal_bins(ids, connected):
                return ids
        # A fresh, independently drawn odd backbone remains available.
        # No archived cells or previous context are recovered.
        for _ in range(768):
            ids = [G.idx((rng.randrange(7),
                          rng.choice((1,3,5,7,9,11))))
                   for _ in range(count)]
            if legal_bins(ids, count == 3):
                return ids
        # Some alternative degree-eight graphs cannot pass this C4 gate.
        # They remain legal proposals and face the actual quotient gates.
        return rng.sample([g for g in range(q) if g != ident], count)

    def dense(count, gs, turn):
        for _ in range(128):
            if count == 3 and len(gs) >= 3 and policy < 2:
                a, b, c = gs[:3]
                if policy == 0:
                    ids = [ident, table[a][b], table[b][c]]
                else:
                    ids = [ident, table[a][inverse[b]], table[c][a]]
                # Change one actual relative term, not a common gauge.
                if turn % 3:
                    ids[1 + turn % 2] = rng.randrange(q)
            elif count == 3 and policy == 2:
                phase = rng.randrange(12)
                ids = [G.idx((rng.randrange(7),
                              (phase+b+3*rng.randrange(4)) % 12))
                       for b in range(3)]
            else:
                ids = [ident] + rng.sample(
                    [g for g in range(q) if g != ident], count-1)
            if len(set(ids)) != count:
                continue
            left = {table[inverse[g]][h]
                    for g in ids for h in ids if g != h}
            right = {table[g][inverse[h]]
                     for g in ids for h in ids if g != h}
            if len(left) == len(right) == count*(count-1):
                return [G.elements[g] for g in ids]
        return rng.sample(G.elements, count)

    def fresh(counts, turn):
        gs = draw_bins(sum(c == 2 for c in counts), turn)
        cells, bi = [], 0
        for count in counts:
            if count == 1:
                cells.append([G.pick(rng)])
            elif count == 2:
                cells.append([G.e, G.elements[gs[bi]]])
                bi += 1
            else:
                cells.append(dense(count, gs, turn+len(cells)))
        return cells

    def small_words(columns, cap=32):
        """Exact absence test through five; positive witness banks are capped.

        The regular COMPONENT action permits one root per column block.
        Pair lists retain collisions, so rank-deficient proposals do not
        silently lose witnesses through representative replacement.
        """
        singles, pairs, found = {0: [0]}, {}, set()
        for c, value in enumerate(columns):
            word = 1 << c
            for old in singles.get(value, ()):
                if word ^ old:
                    found.add(word ^ old)
            singles.setdefault(value, []).append(word)
        for a, value in enumerate(columns):
            for b in range(a):
                key = value ^ columns[b]
                word = (1 << a) | (1 << b)
                for old in singles.get(key, ()):
                    if word ^ old:
                        found.add(word ^ old)
                for old in pairs.get(key, ()):
                    if word ^ old:
                        found.add(word ^ old)
                pairs.setdefault(key, []).append(word)
                if len(found) >= cap:
                    return found, pairs, False
        for root in range(0, width, q):
            for a, value in enumerate(columns):
                if a == root:
                    continue
                prefix = columns[root] ^ value
                used = (1 << root) | (1 << a)
                for b in range(a):
                    if b == root:
                        continue
                    selected = used | (1 << b)
                    for old in pairs.get(prefix ^ columns[b], ()):
                        word = old ^ selected
                        if word and word.bit_count() <= 5:
                            found.add(word)
                            if len(found) >= cap:
                                return found, pairs, False
        return found, pairs, not found

    def short_completion(record, budget=24000):
        """Budgeted rooted search through six; exhaustion proves nothing."""
        columns = record["columns"]
        singles, pairs, incidence = {}, {}, [0]*q
        for c, value in enumerate(columns):
            singles.setdefault(value, []).append(1 << c)
            for r in bits(value):
                incidence[r] |= 1 << c
            for d in range(c):
                pairs.setdefault(value ^ columns[d], []).append(
                    (1 << c) | (1 << d))
        found = set(record["words"])
        all_variables = (1 << width)-1
        degree = max(c.bit_count() for c in columns)
        for root in range(0, width, q):
            nodes = [budget]
            def visit(value, word, allowed):
                if nodes[0] <= 0 or len(found) >= 256:
                    return
                nodes[0] -= 1
                if not value:
                    if word:
                        found.add(word)
                    return
                left = 6-word.bit_count()
                if left <= 0 or value.bit_count() > degree*left:
                    return
                for extra in singles.get(value, ()):
                    if extra & allowed == extra:
                        found.add(word | extra)
                if left >= 2:
                    for extra in pairs.get(value, ()):
                        if extra & allowed == extra:
                            found.add(word | extra)
                if left <= 2:
                    return
                choices, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    size = available.bit_count()
                    if not size:
                        return
                    if size < smallest:
                        choices, smallest = available, size
                order = sorted(bits(choices), key=lambda c: (
                    -(columns[c] & value).bit_count(), c))
                live = allowed
                for c in order:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if nodes[0] <= 0 or len(found) >= 256:
                        break
            visit(columns[root], 1 << root,
                  all_variables ^ (1 << root))
        assert all(syndrome(columns, w) == 0 for w in found)
        record["words"] = sorted(found, key=lambda w: (w.bit_count(), w))[:64]

    def kernel_probe(columns, local, trials, initial=(), keep=48):
        retained = set(initial)
        order = list(range(len(columns)))
        for _ in range(trials):
            local.shuffle(order)
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
            light = sorted(set(basis) | retained, key=int.bit_count)[:16]
            grown = set(light) | retained
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= 12:
                        grown.add(candidate)
            retained = set(sorted(grown,
                key=lambda w: (w.bit_count(), w))[:keep])
        assert retained and all(syndrome(columns, w) == 0 for w in retained)
        return sorted(retained, key=lambda w: (w.bit_count(), w))

    def pool(side, counts):
        good, deficient, seen = [], [], set()
        baseline = counts == (2,2,2,3)
        for turn in range(72 if baseline else 24):
            cells = fresh(counts, turn)
            signature = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell)) for cell in cells))
            if signature in seen:
                continue
            seen.add(signature)
            rows, columns = component(cells, side)
            rank = gf2_rank(rows)
            words, _, free5 = small_words(columns)
            if rank == q and not free5:
                continue
            record = dict(cells=cells, rows=rows, columns=columns,
                          rank=rank, free5=free5, words=list(words),
                          cycles=four_cost(columns), signature=signature)
            if rank == q:
                good.append(record)
                good.sort(key=lambda r: (r["cycles"], r["signature"]))
                del good[4:]
            else:
                deficient.append(record)
                deficient.sort(key=lambda r: (
                    -int(r["free5"]), r["rank"], r["cycles"]))
                del deficient[3:]
            if baseline and turn >= 55 and len(good) >= 3:
                break
        finalists = good[:2] + deficient[:2]
        for at, record in enumerate(finalists):
            short_completion(record)
            record["words"] = kernel_probe(
                record["columns"],
                random.Random(seed + 100003*int(side == "B")
                              + 700001*at + sum(counts)),
                16, record["words"])
        return finalists

    def quantum_reading(hx, hz, a, b, n, deeper=False):
        """Only original-span-verified nontrivial words lower the reading."""
        reading, hits = n+1, 0
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            columns = columns_of(opposite, n)
            pivots = span(stabilizers)
            # Linear quotient residues accelerate repeated membership tests.
            quotients = [residue(1 << c, pivots) for c in range(n)]
            seeds, observed = [], set()
            for at, word in enumerate(record["words"]):
                translations = range(q) if at < 12 else (ident,)
                for t in translations:
                    moved = 0
                    for c in bits(word):
                        block, h = divmod(c, q)
                        h = table[h][t] if side == "A" else table[t][h]
                        moved |= 1 << (block*q+h)
                    for fixed in range(4):
                        if side == "A":
                            actual = sum(
                                ((moved >> (j*q)) & mask)
                                << ((4*j+fixed)*q) for j in range(4))
                        else:
                            actual = moved << (fixed*width)
                        if actual in observed:
                            continue
                        observed.add(actual)
                        assert syndrome(columns, actual) == 0
                        if syndrome(quotients, actual):
                            reading = min(reading, actual.bit_count())
                            hits += int(actual.bit_count() <= 6)
                            seeds.append(actual)
                if reading <= 4:
                    break
            if deeper and reading >= 6:
                words = kernel_probe(
                    columns,
                    random.Random(seed + 2700011
                                  + 100003*int(side == "B")),
                    8, sorted(set(seeds), key=int.bit_count)[:24], 64)
                for word in words:
                    if syndrome(quotients, word):
                        reading = min(reading, word.bit_count())
                        hits += int(word.bit_count() <= 6)
        return reading, hits

    def logical4_free(opposite, stabilizers, n):
        """Exact logical exclusion through four; short stabilizers are allowed."""
        columns = columns_of(opposite, n)
        pivots = span(stabilizers)
        singles = {0: 0}
        for c, value in enumerate(columns):
            word = 1 << c
            old = singles.get(value)
            if old is not None:
                if residue(word ^ old, pivots):
                    return False
            else:
                singles[value] = word
        pairs = {}
        for a, value in enumerate(columns):
            for b in range(a):
                key = value ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(key)
                if old is not None and residue(word ^ old, pivots):
                    return False
                old = pairs.get(key)
                if old is not None:
                    if residue(word ^ old, pivots):
                        return False
                else:
                    pairs[key] = word
        return True

    cache, proposals = {}, []
    for plan_id, (sa, sb) in enumerate(plans):
        if plan_id in (1,2,4) and (model+policy+cohort) % 2:
            sa, sb = sb, sa
        for side, counts in (("A", sa), ("B", sb)):
            key = (side, counts)
            if key not in cache:
                cache[key] = pool(side, counts)
        aa, bb = cache[("A", sa)], cache[("B", sb)]
        pairs = [(a,b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            ab[0]["free5"] and ab[1]["free5"],
            2*q-ab[0]["rank"]-ab[1]["rank"],
            -ab[0]["cycles"]-ab[1]["cycles"]), reverse=True)
        # Both full-rank and deficient mechanisms receive quotient testing.
        chosen = pairs[:2]
        deficient = [ab for ab in pairs
                     if ab[0]["rank"] < q or ab[1]["rank"] < q]
        for pair in deficient[:2]:
            if all(pair[0] is not a or pair[1] is not b for a,b in chosen):
                chosen.append(pair)
        for a, b in chosen:
            A, B = [a["cells"]], [b["cells"]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            assert n == 1428
            if not (k >= 756 and 40*k > 21*n):
                continue
            dq, hits = quantum_reading(hx, hz, a, b, n)
            d = 6 if dq > n else min(8, dq)
            merit = (k*d*d, d, -hits,
                     -a["cycles"]-b["cycles"])
            proposals.append(dict(
                A=A, B=B, hx=hx, hz=hz, k=k, a=a, b=b,
                dq=dq, hits=hits, merit=merit, plan=plan_id))
    proposals.sort(key=lambda r: r["merit"], reverse=True)

    # Independent complete-matrix challenges cannot erase earlier witnesses.
    for record in proposals[:4]:
        dq, hits = quantum_reading(
            record["hx"], record["hz"], record["a"], record["b"],
            1428, deeper=True)
        record["dq"] = min(record["dq"], dq)
        d = 6 if record["dq"] > 1428 else min(8, record["dq"])
        record["merit"] = (
            record["k"]*d*d, d, -max(record["hits"], hits),
            -record["a"]["cycles"]-record["b"]["cycles"])
    proposals.sort(key=lambda r: r["merit"], reverse=True)

    for record in proposals:
        hx, hz = record["hx"], record["hz"]
        if not logical4_free(hx, hz, 1428):
            continue
        if not logical4_free(hz, hx, 1428):
            continue
        A, B = record["A"], record["B"]
        sa = [[len(cell) for cell in A[0]]]
        sb = [[len(cell) for cell in B[0]]]
        assert _grid_weights(sa, sb) <= min(12, W_MAX)
        assert _no_dead_lines(sa, sb)
        assert all(len(cell) == len({G._key(g) for g in cell})
                   for grid in (A,B) for row in grid for cell in row)
        ar, br = _component_rows(hx, hz, q)
        assert ar == record["a"]["rows"] and br == record["b"]["rows"]
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": record["k"],
            "card": (model, policy, cohort, record["plan"]),
            "support": (sa, sb), "quality": record["merit"],
            "component_ranks": (
                record["a"]["rank"], record["b"]["rank"]),
            "component_kernel5_free": (
                record["a"]["free5"], record["b"]["free5"]),
            "verified_quantum_upper_reading": (
                None if record["dq"] > 1428 else record["dq"]),
            "screen": "original-span logical-through-four exclusion",
        }
    raise ValueError("new subgroup/support proposals failed quotient gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete joint quotient-selected cell without new draws."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) <= 5
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Row-six/row-nine density reallocation and deeper kernel-pair screening yield five selected readings >=16; the measured current program returned 14,14,16,14,16, all with sector-1 minimum witnesses.
# REFUTED: Full-rank component kernel-five screening and classical six-witness reduction improve this C7:C12(s=2) (1,4)x(1,4) bed beyond 19.059; all five measured parent codes instead retained k=756 and distance six, with abundant sector-one minima.
# REFUTED: Proper-subgroup/support selection improves the shown order-84 high-rate program beyond 19.059; its five measured codes remained k=756 with distance-six sector-one minima.
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
