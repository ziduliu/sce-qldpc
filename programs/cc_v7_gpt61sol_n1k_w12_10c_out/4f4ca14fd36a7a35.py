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
CONCEPT = ("On the extension-field affine group (F16,+):C5, order 80, shapes (1,3)x(2,5), independently optimized degree-three components without standalone binomial columns should exceed five-code fitness 35.5 in cell 0.350. Non-improvement refutes this experiment; distance readings remain provisional upper bounds.")
GROUP = ("Jump from C67:C3 to the additive binary field F16 semidirect C5. Multiplication by an order-five field element acts irreducibly on the four-dimensional translation group. This order-80 extension-field bed is absent from BEDS. The order-42 affine bed reached 35.5 in cell 0.350, motivating a related construction without assuming distance transfer.")
SHAPES = ("A=(1,3), B=(2,5), q=80 give n=1360 and k>=480, guaranteed rate 6/17=0.352941. Components have 240 variables/80 checks and 400 variables/160 checks, with nominal rates 2/3 and 3/5. Parent weight-14..16 X minima occupied sector-1 strips, prominently Acol1 across all Bcols. Replace both component geometries and retain exact full rates in [0.325,0.375).")
FILLING = ("A has three distinct-label trinomials; every B column totals three terms. Five B margin profiles cross four quotient-label/cycle objectives to form twenty deliberate cards. B row totals are 7/8 or 6/9, giving HX=9+3=12 and HZ<=9+3=12. Avoid involutory within-cell ratios; trinomial relatives generate the full affine group. Assemble screening components only from verified monomial lifts. Optimize actual four/six cycles, screen localized and systematic kernel witnesses, repair blocks used by surviving words, and independently deepen finalists. Exact full ranks, component extraction, CSS and both weight sums gate submission. These screens do not certify quantum distance or complete Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_F16add_C5_13x25_degree3_witness_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 80

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("the F16 additive-by-C5 group requires order 80")

    # F16 = F2[z]/(z^4+z+1). The field element z^3 has order five.
    # (a,b) acts as t -> a + (z^3)^b*t, with addition implemented by XOR.
    def fmul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 16:
                a ^= 19
        return result

    action = (1, 8, 12, 10, 15)
    assert all(fmul(action[i], action[j]) == action[(i+j) % 5]
               for i in range(5) for j in range(5))
    els = [(a, b) for b in range(5) for a in range(16)]

    def red(g):
        return (int(g[0]) % 16, int(g[1]) % 5)

    def mul(g, h):
        return (g[0] ^ fmul(action[g[1]], h[0]),
                (g[1] + h[1]) % 5)

    def inv(g):
        b = (-g[1]) % 5
        return (fmul(action[b], g[0]), b)

    def idx(g):
        a, b = red(g)
        return 16*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _legacy_normalunit_ctx(rng, G, idx):
    """Unused historical proposal; the new extension-field NEW_CTX is below."""
    from itertools import combinations

    # Kinds: 0 = one coset; 1 = 1+1+1; 2 = 2+1+0.
    # At most one one-coset cell per side avoids the obvious
    # commuting normal-polynomial pair. B always has exactly one.
    profiles = (
        ((2, 2), (0, 2, 2)),
        ((1, 2), (0, 2, 2)),
        ((2, 2), (0, 1, 2)),
        ((0, 2), (0, 2, 2)),
    )
    occupancy = ((3, 0, 0), (1, 1, 1), (2, 1, 0))
    card, cohort = idx % 20, idx // 20
    aplan, bplan = profiles[card % 4]
    plans = {"A": aplan, "B": bplan}
    acounts, bcounts = (3, 3), (3, 3, 3)
    q = G.q
    separation = 1 + 3 * (cohort % 3)
    sa, sb = [list(acounts)], [list(bcounts)]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    def affine_class(t):
        """Unordered affine class of {0,1,t} in the normal subgroup."""
        it = pow(t, -1, 67)
        io = pow((1-t) % 67, -1, 67)
        return min(t, (1-t) % 67, it, io,
                   ((t-1)*it) % 67, (t*io) % 67)

    # Exclude repeated-difference progressions and the class of
    # {1,29,37}, whose polynomial can commute with the order-three action.
    excluded = {affine_class(2), affine_class(30)}
    classes = sorted(
        {affine_class(t) for t in range(2, 67)} - excluded)
    assert len(classes) >= 5
    triangle_at = card // 4 + 5*cohort
    triangle = {
        "B": classes[triangle_at % len(classes)],
        "A": classes[(triangle_at+1) % len(classes)],
    }

    def full_relative(cell):
        """For this Frobenius group, noncommuting relative generators
        generate all 201 elements: proper subgroups lie in C67 or C3."""
        if len(set(cell)) != 3:
            return False
        u = G.mul(G.inv(cell[0]), cell[1])
        v = G.mul(G.inv(cell[0]), cell[2])
        return G.mul(u, v) != G.mul(v, u)

    def draw_cell(side, j, count):
        assert count == 3
        kind = plans[side][j]
        for attempt in range(1000):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            legal = True
            for b, size in zip(fibers, occupancy[kind]):
                if not size:
                    continue
                if kind == 0:
                    origin = rng.randrange(67)
                    scale = rng.randrange(1, 67)
                    rotations = [
                        origin, (origin+scale) % 67,
                        (origin+triangle[side]*scale) % 67,
                    ]
                else:
                    rotations = rng.sample(range(67), size)
                differences = [(a-c) % 67 for a in rotations
                               for c in rotations if a != c]
                if (len(set(differences)) != len(differences)
                        or any(min(d, 67-d) < separation
                               for d in differences)):
                    legal = False
                    break
                cell.extend((a, b) for a in rotations)
            if not legal or len(cell) != 3 or len(set(cell)) != 3:
                continue
            if kind != 0 and not full_relative(cell):
                continue
            return cell
        raise ValueError("normal-unit trinomial geometry could not be filled")

    def component(side, cells):
        # Empty one-cell dummy side is only an extraction device.
        # The component is produced by the scaffold, not a separate lift.
        width = len(cells) * q
        mask = (1 << width)-1
        if side == "A":
            hx, _, _ = lifted_matrices(G, [cells], [[[]]])
            rows = [row & mask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, [[[]]], [cells])
            rows = [row & mask for row in hz]
        columns = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                columns[bit.bit_length()-1] |= 1 << r
                source ^= bit
        return rows, columns

    def geometry(rows, columns):
        """Exact C4 count; projection excess is a secondary heuristic."""
        width = len(columns)
        bad = sum(c == 0 for c in columns)
        bad += width-len(set(columns))
        adjacency = [0] * len(rows)
        four = 0
        for a, row in enumerate(rows):
            for b in range(a):
                common = (row & rows[b]).bit_count()
                four += common*(common-1)//2
                if common:
                    adjacency[a] |= 1 << b
                    adjacency[b] |= 1 << a
        reached = frontier = 1
        while frontier:
            neighbors = 0
            while frontier:
                bit = frontier & -frontier
                neighbors |= adjacency[bit.bit_length()-1]
                frontier ^= bit
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << len(rows))-1:
            bad += len(rows)-reached.bit_count()
        forced = sum(c.bit_count()*(c.bit_count()-1)*
                     (c.bit_count()-2)//6 for c in columns)
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            upper = neighbors & (-1 << (a+1))
            while upper:
                bit = upper & -upper
                b = bit.bit_length()-1
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
                upper ^= bit
        excess = max(0, triangles-forced)
        return bad, four, excess

    def syndrome(columns, word):
        value = 0
        while word:
            bit = word & -word
            value ^= columns[bit.bit_length()-1]
            word ^= bit
        return value

    def kernel_search(rows, columns, seed, trials):
        """Actual null words; high readings do not prove large distance."""
        local = random.Random(seed)
        width = len(columns)
        best, witness = width+1, None

        def note(word):
            nonlocal best, witness
            if word and word.bit_count() < best:
                best, witness = word.bit_count(), word

        # Cheap deterministic candidates for commuting-block circuits.
        # Outputs of verified component columns are inserted into the
        # opposite variable blocks, then their syndrome is checked.
        # This is not an exhaustive weight-six search.
        for j, k in combinations(range(width // q), 2):
            for t in range(q):
                word = ((columns[k*q+t] << (j*q))
                        ^ (columns[j*q+t] << (k*q)))
                if (word and word.bit_count() < best
                        and syndrome(columns, word) == 0):
                    note(word)

        order = list(range(width))
        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
            for j in order:
                value, word = columns[j], 1 << j
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    note(word)
                    words.append(word)
            words.sort(key=int.bit_count)
            # Every basis pair, rather than pairs of only short basis words.
            for i, word in enumerate(words):
                for j in range(i):
                    note(word ^ words[j])
            for a, b, c in combinations(words[:12], 3):
                note(a ^ b ^ c)
            if best <= 8:
                break
        if witness is None:
            raise ValueError("no component kernel witness")
        assert syndrome(columns, witness) == 0
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
        return best, witness

    def no_kernel4(columns):
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return False
        pairs = set()
        for i, a in enumerate(columns):
            for j in range(i):
                value = a ^ columns[j]
                if value in singles or value in pairs:
                    return False
                pairs.add(value)
        return True

    def optimize(side, counts, restart):
        seed = (BASE_SEED + 1009*idx + 700001*restart
                + (0 if side == "A" else 300007))
        bank = set()

        def inspect(cells):
            rows, columns = component(side, cells)
            stats = geometry(rows, columns)
            mask = (1 << q)-1
            block_ranks = tuple(
                gf2_rank([(row >> (j*q)) & mask for row in rows])
                for j in range(len(cells))
            )
            bad = stats[0]
            bad += sum(max(0, q-4-rank) for rank in block_ranks)
            bad += sum(q-block_ranks[j]
                       for j, kind in enumerate(plans[side]) if kind == 0)
            stats = (bad, stats[1], stats[2])
            return {"cells": cells, "rows": rows, "cols": columns,
                    "stats": stats, "block_ranks": block_ranks,
                    "upper": len(columns)+1, "witness": None}

        def rating(record):
            reading = record["upper"]
            for word in bank:
                if word.bit_count() < reading:
                    if syndrome(record["cols"], word) == 0:
                        reading = word.bit_count()
            deficit = sum(q-rank for rank in record["block_ranks"])
            return (reading, -record["stats"][0],
                    -record["stats"][1],
                    -record["stats"][2]-q*deficit)

        def probe(record, trials, offset):
            reading, word = kernel_search(
                record["rows"], record["cols"], seed+offset, trials)
            record["upper"] = min(record["upper"], reading)
            record["witness"] = word
            bank.add(word)

        starts = []
        for _ in range(64 if side == "B" else 32):
            cells = [draw_cell(side, j, count)
                     for j, count in enumerate(counts)]
            record = inspect(cells)
            if record["stats"][0] == 0:
                starts.append(record)
        if not starts:
            return []
        starts.sort(key=lambda record: record["stats"])
        pool = starts[:8 if side == "B" else 6]
        for record in pool:
            probe(record, 8 if side == "B" else 12, 0)
        for record in sorted(pool, key=rating, reverse=True)[:2]:
            probe(record, 32, 1)
        current = max(pool, key=rating)

        # Spend more mutations on the longer component carrying the
        # parent's light X family. Preserve its occupancy and triangle card.
        for step in range(20 if side == "B" else 10):
            surviving = [w for w in bank
                         if syndrome(current["cols"], w) == 0]
            active = []
            if surviving:
                word = min(surviving, key=int.bit_count)
                while word:
                    bit = word & -word
                    active.append((bit.bit_length()-1)//q)
                    word ^= bit
            j = (rng.choice(active) if active and rng.random() < 0.8
                 else rng.randrange(len(counts)))
            cells = [list(cell) for cell in current["cells"]]
            kind = plans[side][j]
            if kind == 0 or rng.random() < 0.30:
                # The one-coset cell retains its unordered affine class.
                cells[j] = draw_cell(side, j, counts[j])
            else:
                # Usually retain two labels and alter one exponent.
                # This preserves accumulated component geometry rather
                # than discarding an entire useful mixed trinomial.
                t = rng.randrange(3)
                old = cells[j][t]
                cells[j][t] = (rng.randrange(67), old[1])
                if cells[j][t] == old or not full_relative(cells[j]):
                    continue
            proposed = inspect(cells)
            if proposed["stats"][0]:
                continue
            if proposed["stats"][1] > current["stats"][1] + 2*q:
                continue
            probe(proposed, 8 if side == "B" else 12, 0)
            pk, ck = rating(proposed), rating(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.12):
                current = proposed
                pool.append(proposed)

        finalists = sorted(pool, key=rating, reverse=True)[:3]
        screened = []
        for record in finalists:
            if not no_kernel4(record["cols"]):
                continue
            probe(record, 64, 2)
            screened.append(record)
        screened.sort(key=rating, reverse=True)
        screened = screened[:2]
        for record in screened:
            probe(record, 128, 3)
        screened.sort(key=rating, reverse=True)
        # Both returned components received independent deep confirmation;
        # an optimistic third candidate cannot bypass the longer probe.
        return [(rating(record), record) for record in screened]

    for restart in range(3):
        apool = optimize("A", acounts, restart)
        bpool = optimize("B", bcounts, restart)
        proposals = []
        for ak, ar in apool:
            for bk, br in bpool:
                key = (min(ak[0], bk[0]), ak[0]+bk[0],
                       ak[2]+bk[2], ak[3]+bk[3])
                proposals.append((key, ar, br, ak[0], bk[0]))
        proposals.sort(key=lambda item: item[0], reverse=True)
        for quality, ar, br, da, db in proposals:
            A, B = [ar["cells"]], [br["cells"]]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 1407
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 402 and 275*n <= 1000*k < 325*n):
                continue
            # Check that dummy-side extraction screens the final components.
            actual_a, actual_b = _component_rows(hx, hz, q)
            assert actual_a == ar["rows"] and actual_b == br["rows"]
            assert max(row.bit_count() for row in hx+hz) <= 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            return {"idx": idx, "card": card,
                    "cells": {"A": A, "B": B}, "k": k,
                    "occupancy_profiles": (aplan, bplan),
                    "triangle_classes": triangle,
                    "verified_cell_ranks": (
                        ar["block_ranks"], br["block_ranks"]),
                    "component_readings": (da, db), "quality": quality}
    raise ValueError("no new-shape filling passed the exact target-rate gate")


# Upper-row counts; every B column has total degree three.
# These are distinct weighted support profiles, not column permutations.
_F16_B_PROFILES = (
    (1, 1, 1, 2, 2),
    (0, 1, 2, 2, 2),
    (1, 1, 1, 1, 3),
    (0, 1, 1, 2, 3),
    (1, 1, 1, 1, 2),
)


def _f16_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length()-1
        word ^= bit


def _f16_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _f16_bits(row):
            columns[c] |= 1 << r
    return columns


def _f16_syndrome(columns, word):
    value = 0
    for c in _f16_bits(word):
        value ^= columns[c]
    return value


def _f16_dependencies(columns, order):
    """Nonzero dependencies in the original column coordinates."""
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
    return basis


def _f16_local_words(columns, q):
    """Localized genuine kernels; exhaustive only when nullity <=12."""
    width = len(columns)
    reading, retained = width+1, []

    def note(word):
        nonlocal reading
        if word:
            weight = word.bit_count()
            if weight < reading:
                reading = weight
            retained.append(word)

    for j in range(width // q):
        restricted = columns[j*q:(j+1)*q]
        basis = _f16_dependencies(restricted, range(q))
        if not basis:
            continue
        if len(basis) <= 12:
            word = 0
            best, witness = q+1, None
            # Consecutive Gray-code differences toggle bit v2(t).
            for t in range(1, 1 << len(basis)):
                word ^= basis[(t & -t).bit_length()-1]
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word
            note(witness << (j*q))
        else:
            for i, word in enumerate(basis):
                note(word << (j*q))
                for other in basis[:i]:
                    note((word ^ other) << (j*q))

    retained = sorted(set(retained), key=int.bit_count)[:8]
    assert all(_f16_syndrome(columns, word) == 0 for word in retained)
    return reading, retained


def _f16_kernel_probe(columns, seed, trials, deep=False):
    """Sample actual kernel words; a high reading is not a certificate."""
    from itertools import combinations

    diagnostic = random.Random(seed)
    width = len(columns)
    reading, witness = width+1, None
    retained = []
    order = list(range(width))

    def note(word):
        nonlocal reading, witness
        if word:
            weight = word.bit_count()
            if weight < reading:
                reading, witness = weight, word

    for _ in range(trials):
        diagnostic.shuffle(order)
        basis = _f16_dependencies(columns, order)
        if not basis:
            raise ValueError("degree-three component unexpectedly has no kernel")
        basis.sort(key=int.bit_count)
        for word in basis:
            note(word)

        anchors = len(basis) if deep else min(24, len(basis))
        additions = []
        for i, word in enumerate(basis[:anchors]):
            for other in basis[i+1:]:
                candidate = word ^ other
                note(candidate)
                if candidate and candidate.bit_count() <= reading+4:
                    additions.append(candidate)

        light = sorted(set(retained + basis + additions),
                       key=int.bit_count)[:16]
        for a, b, c in combinations(light[:10], 3):
            note(a ^ b ^ c)

        moves = basis + retained
        diagnostic.shuffle(moves)
        for initial in light[:4]:
            word, weight = initial, initial.bit_count()
            for sweep in range(3 if deep else 2):
                changed = False
                for move in moves:
                    candidate = word ^ move
                    if candidate and candidate.bit_count() < weight:
                        word, weight = candidate, candidate.bit_count()
                        changed = True
                note(word)
                if not changed:
                    break
            additions.append(word)

        if witness is not None:
            additions.append(witness)
        retained = sorted(set(retained + light + additions),
                          key=int.bit_count)[:24]
        if reading <= 6:
            break

    assert witness is not None
    words = sorted(set(retained + [witness]), key=int.bit_count)[:8]
    assert all(_f16_syndrome(columns, word) == 0 for word in words)
    return reading, words


def _f16_no_kernel4(columns):
    """Exact component exclusion through weight four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    pairs = set()
    for i, a in enumerate(columns):
        for b in columns[:i]:
            value = a ^ b
            if value in singles or value in pairs:
                return False
            pairs.add(value)
    return True


def _f16_geometry(rows, width):
    """Exact simple Tanner C4/C6 counts and a projection-C4 heuristic."""
    columns = _f16_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    if any(column.bit_count() != 3 for column in columns):
        raise ValueError("component column degree is not three")

    common = [dict() for _ in rows]
    adjacency = [0] * len(rows)
    four = 0
    for v, column in enumerate(columns):
        checks = list(_f16_bits(column))
        for at, a in enumerate(checks):
            adjacency[a] |= column ^ (1 << a)
            for b in checks[at+1:]:
                previous = common[a].get(b, 0)
                four += previous.bit_count()
                common[a][b] = previous | (1 << v)

    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _f16_bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows))-1:
        return None

    six = 0
    for a, neighbors in enumerate(common):
        ns = sorted(neighbors)
        for at, b in enumerate(ns):
            ab = neighbors[b]
            nab = ab.bit_count()
            for c in ns[at+1:]:
                bc = common[b].get(c, 0)
                if not bc:
                    continue
                ac = neighbors[c]
                nac, nbc = ac.bit_count(), bc.bit_count()
                reused = (ab & ac & bc).bit_count()
                six += nab*nac*nbc - reused*(nab+nac+nbc-2)

    projected = 0
    for a, neighbors in enumerate(adjacency):
        for b in range(a):
            overlap = (neighbors & adjacency[b]).bit_count()
            projected += overlap*(overlap-1)//2
    projected //= 2
    return (four, six, projected), columns


def NEW_CTX(rng, G, idx):
    """Twenty new affine-field support/objective cards with witness repair."""
    import math

    q = G.q
    assert q == 80
    assert A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    profile, mode = card // 4, card % 4
    upper = _F16_B_PROFILES[profile]
    supports = {
        "A": [[3, 3, 3]],
        "B": [list(upper), [3-v for v in upper]],
    }
    assert _grid_weights(supports["A"], supports["B"]) == 12 <= W_MAX
    assert _no_dead_lines(supports["A"], supports["B"])
    assert max(map(sum, supports["B"])) <= 9
    assert all(sum(row[j] for row in supports["B"]) == 3
               for j in range(5))
    seed = rng.randrange(1 << 62)

    # Cache native group operations only. The probe has the same element
    # indexing and exact operation table as the validated original group.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)
    assert all(G.idx(g) == t for t, g in enumerate(G.elements))
    assert all(g == (t % 16, t // 16)
               for t, g in enumerate(G.elements))

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe_group = Probe()
    mask = (1 << q)-1

    # Every monomial screening block comes from lifted_matrices.
    # XOR and disjoint column shifts below use only GF2 linearity.
    raw, _, _ = lifted_matrices(
        probe_group, [[[g] for g in range(q)]], [[[]]])
    left = [
        tuple((row >> (j*q)) & mask for row in raw)
        for j in range(q)
    ]
    _, raw, _ = lifted_matrices(
        probe_group, [[[]]], [[[g] for g in range(q)]])
    right = [
        tuple((row >> (j*q)) & mask for row in raw)
        for j in range(q)
    ]
    atoms = {"A": left, "B": right}
    del raw

    def whole_sidon(cell):
        if len(set(cell)) != len(cell):
            return False
        if len(cell) < 2:
            return True
        # All relative ratios must be non-involutory. Here that means
        # distinct complement coordinates inside each nonzero cell.
        if len({g // 16 for g in cell}) != len(cell):
            return False
        differences_left = {
            table[inverses[a]][b]
            for a in cell for b in cell if a != b
        }
        differences_right = {
            table[a][inverses[b]]
            for a in cell for b in cell if a != b
        }
        expected = len(cell)*(len(cell)-1)
        return len(differences_left) == len(differences_right) == expected

    def full_relative(cell):
        u = table[inverses[cell[0]]][cell[1]]
        v = table[inverses[cell[0]]][cell[2]]
        # The irreducible C5 action implies that noncommuting relatives
        # generate the entire affine group. Final trinomials are also
        # checked by explicit closure below.
        return table[u][v] != table[v][u]

    triangles = ((0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 1, 4))

    def draw_cell(side, i, j, count):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        for attempt in range(512):
            if count == 2 or mode == 2:
                fibers = rng.sample(range(5), count)
            else:
                base = triangles[(mode+j+2*i+cohort
                                  +int(side == "B")) % 4]
                shift = rng.randrange(5)
                fibers = [(b+shift) % 5 for b in base]
            cell = [16*b+rng.randrange(16) for b in fibers]
            if not whole_sidon(cell):
                continue
            if count == 3 and not full_relative(cell):
                continue
            return cell
        raise ValueError("extension-field relative-label sampler failed")

    def fresh(side):
        return [[draw_cell(side, i, j, count)
                 for j, count in enumerate(row)]
                for i, row in enumerate(supports[side])]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    def component(side, grid):
        width = len(grid[0])*q
        rows = [0] * (len(grid)*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atoms[side][g]
                    for r, word in enumerate(block):
                        rows[i*q+r] ^= word << (j*q)
        return rows, width

    def inspect(side, grid):
        rows, width = component(side, grid)
        inspected = _f16_geometry(rows, width)
        if inspected is None:
            return None
        cycles, columns = inspected
        four, six, projected = cycles
        cost = (100000*four
                + (16, 2, 8, 32)[mode]*six
                + (0, 8, 1, 2)[mode]*projected)
        return dict(
            grid=copy_grid(grid), rows=rows, width=width,
            columns=columns, cycles=cycles, cost=cost,
            signature=signature(grid), d=width+1,
        )

    def mutate(side, record, active=None, turn=0):
        grid = copy_grid(record["grid"])
        places = [
            (i, j) for i, row in enumerate(grid)
            for j, cell in enumerate(row) if cell
        ]
        if active and rng.random() < 0.85:
            candidates = [(i, j) for i, j in places if j in active]
            place = rng.choice(candidates or places)
        else:
            place = rng.choice(places)
        i, j = place
        old = grid[i][j]
        if turn % 5 == 0 or len(old) == 1 or rng.random() < 0.25:
            new = draw_cell(side, i, j, len(old))
        else:
            new = list(old)
            t = rng.randrange(len(new))
            new[t] = 16*(old[t] // 16)+rng.randrange(16)
            if not whole_sidon(new):
                return None
            if len(new) == 3 and not full_relative(new):
                return None
        if set(new) == set(old):
            return None
        grid[i][j] = new
        return inspect(side, grid)

    def optimize(side, restart):
        offset = 0 if side == "A" else 100003
        diagnostic_seed = seed+offset+700001*restart
        keepers, bank = {}, set()

        def retain(record):
            if record is None:
                return
            key = record["signature"]
            old = keepers.get(key)
            if old is None or record["cost"] < old["cost"]:
                keepers[key] = record
            if len(keepers) > 10:
                worst = max(keepers, key=lambda t: keepers[t]["cost"])
                del keepers[worst]

        def merit(record):
            for word in bank:
                if word.bit_count() < record["d"]:
                    if not _f16_syndrome(record["columns"], word):
                        record["d"] = word.bit_count()
            return (
                record.get("free4", False), record["d"],
                -record["cost"], record.get("rank", 0)
            )

        def probe(record, trials, turn, deep=False):
            reading, words = _f16_kernel_probe(
                record["columns"], diagnostic_seed+1000003*turn,
                trials, deep)
            record["d"] = min(record["d"], reading)
            bank.update(words)
            if len(bank) > 96:
                kept = sorted(bank, key=int.bit_count)[:64]
                bank.clear()
                bank.update(kept)
            record["words"] = words

        starts = []
        for _ in range(28 if side == "A" else 40):
            record = inspect(side, fresh(side))
            if record is not None:
                starts.append(record)
                retain(record)
        if not starts:
            return []
        starts.sort(key=lambda r: r["cost"])

        steps = 64 if side == "A" else 96
        for initial in starts[:2]:
            current = initial
            for turn in range(steps):
                proposed = mutate(side, current, turn=turn)
                if proposed is None:
                    continue
                delta = proposed["cost"]-current["cost"]
                temperature = q*(20*(1-turn/steps)**2+0.25)
                accept = delta <= 0
                if not accept:
                    accept = rng.random() < math.exp(-delta/temperature)
                if accept:
                    current = proposed
                    retain(current)

        pool = []
        ordered = sorted(keepers.values(), key=lambda r: r["cost"])
        for record in ordered:
            record["free4"] = _f16_no_kernel4(record["columns"])
            if not record["free4"]:
                continue
            record["rank"] = gf2_rank(record["rows"])
            local, words = _f16_local_words(record["columns"], q)
            record["d"] = min(record["d"], local)
            bank.update(words)
            probe(record, 8, 0)
            pool.append(record)
            if len(pool) == 6:
                break

        # The stronger classical preferences never silently alter a
        # support card. A legal fallback is still screened and measured.
        if not pool:
            for record in ordered[:3]:
                record["free4"] = _f16_no_kernel4(record["columns"])
                record["rank"] = gf2_rank(record["rows"])
                local, words = _f16_local_words(record["columns"], q)
                record["d"] = min(record["d"], local)
                bank.update(words)
                probe(record, 8, 0)
                pool.append(record)

        current = max(pool, key=merit)
        for turn in range(12 if side == "A" else 20):
            surviving = [
                word for word in bank
                if not _f16_syndrome(current["columns"], word)
            ]
            active = None
            if surviving:
                word = min(surviving, key=int.bit_count)
                active = {c // q for c in _f16_bits(word)}
            proposed = mutate(side, current, active, turn)
            if proposed is None:
                continue
            if proposed["cycles"][0] > current["cycles"][0]+q:
                continue
            proposed["free4"] = _f16_no_kernel4(proposed["columns"])
            if current["free4"] and not proposed["free4"]:
                continue
            proposed["rank"] = gf2_rank(proposed["rows"])
            local, words = _f16_local_words(proposed["columns"], q)
            proposed["d"] = min(proposed["d"], local)
            bank.update(words)
            probe(proposed, 8, 1+turn)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (
                    pk[:2] == ck[:2]
                    and proposed["cost"] <= current["cost"]+32*q
                    and rng.random() < 0.12):
                current = proposed
                pool.append(proposed)

        unique = {}
        for record in pool:
            unique[record["signature"]] = record
        first = sorted(unique.values(), key=merit, reverse=True)[:3]
        for record in first:
            probe(record, 24, 101, deep=True)
        finalists = sorted(first, key=merit, reverse=True)[:2]
        for record in finalists:
            # Independent confirmation only lowers the retained reading.
            probe(record, 48 if side == "A" else 64, 211, deep=True)
        finalists.sort(key=merit, reverse=True)
        return finalists

    def assert_full_generation(cell):
        if len(cell) != 3:
            return
        u = table[inverses[cell[0]]][cell[1]]
        v = table[inverses[cell[0]]][cell[2]]
        reached, todo = {identity}, [identity]
        for g in todo:
            for generator in (u, v):
                h = table[g][generator]
                if h not in reached:
                    reached.add(h)
                    todo.append(h)
        assert len(reached) == q

    for restart in range(2):
        aa, bb = optimize("A", restart), optimize("B", restart)
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            ab[0]["free4"] and ab[1]["free4"],
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"]+ab[1]["d"],
            -ab[0]["cost"]-ab[1]["cost"]),
            reverse=True)

        for a, b in pairs:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 17*q == 1360
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 6*q and 325*n <= 1000*k < 375*n):
                continue

            # Check cached GF2 assembly against the final authoritative lift.
            actual_a, actual_b = _component_rows(hx, hz, q)
            assert actual_a == a["rows"] and actual_b == b["rows"]
            for grid in (a["grid"], b["grid"]):
                for row in grid:
                    for cell in row:
                        assert whole_sidon(cell)
                        assert_full_generation(cell)
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert sa == supports["A"] and sb == supports["B"]
            assert _grid_weights(sa, sb) == 12 <= W_MAX
            assert _no_dead_lines(sa, sb)
            assert max(row.bit_count() for row in hx+hz) <= 12
            assert css_ok(hx, hz)
            return {
                "idx": idx, "card": (profile, mode, cohort),
                "cells": {"A": A, "B": B}, "k": k,
                "component_readings": (a["d"], b["d"]),
                "component_ranks": (a["rank"], b["rank"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "kernel4_free": (a["free4"], b["free4"]),
            }
    raise ValueError("affine-field component pairs missed cell 0.350")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without post-screen draws."""
    cell = list(ctx["cells"][side][i][j])
    assert 0 <= len(cell) <= 3
    assert len({G._key(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Reallocated-kernel-pairs-v2 would yield five selected readings >=16 on Z29 (3,6)x(3,7); measured readings were 16,16,14,14,14, with abundant sector-1 minima.
# REFUTED: The long-order B-binomial principal allocation preserves trinomial-quality readings on Z67:Z3 (1,2)x(1,3): every principal card screened at <=10, while all five measured winners used only trinomials and read 12,14,14,12,14.
# REFUTED: A verified normal-unit B trinomial raises every selected sector-1 X reading above 14 on this bed: the current measured readings were 16,14,14,14,14, and all listed X minima remained in sector 1.
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
