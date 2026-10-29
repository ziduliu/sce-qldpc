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
CONCEPT = ("C67 semidirect C3 with action 29, order 201, shapes (1,2)x(1,3): coupled normal-triangle/mixed-gap designs challenged by actual component parity completions through weight fourteen should raise cell 0.300's five-code fitness above 59.4286. Failure to improve measured fitness refutes the prediction; readings remain provisional upper bounds.")
GROUP = ("Keep the frozen split-metacyclic group m=67,r=3,s=29, order 201. This bed leads cell 0.300 at 59.4286, with selected readings 16,14,14,14,14 and k=402 throughout. Normal ratios have order 67, while cross-coset ratios have order three. Use these algebraic distinctions to prescribe relative geometry, not to infer distance.")
SHAPES = ("Keep frozen A=(1,2), B=(1,3), q=201: n=1407, k>=402, guaranteed rate 2/7 and target cell 0.300. Components have 402 and 603 variables with 201 checks each, nominal rates one-half and two-thirds. Parent minimum X witnesses occupied sector-1 Acol/Bcol blocks, predominantly Acol1 across all B columns; Z minima were heavier at 18..22. Emphasize B and retain actual full rates in [0.275,0.325).")
FILLING = ("All cells remain distinct-label trinomials, giving HX weight nine and HZ weight twelve. Twenty cards couple a correctly classified non-progression normal triangle to two prescribed, sign-distinct mixed normal gaps outside its order-three action orbits. B has one verified-invertible normal cell and two full-generating mixed cells; A has two full-generating mixed cells. Cache only scaffold-built monomial summands. Genuine elimination witnesses and budgeted anchored parity completions through fourteen guide mixed-cell replacements; independently challenge both retained finalists. Full ranks, component equality, kernel-four exclusion, weights and CSS gate outputs. An exhausted search proves no exclusion, and component readings are not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r030_m67r3s29_12x13_coupled_gap_parity14_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 201

def custom_ops(q):
    # Same split-metacyclic rule, resized for size factor seven.
    # 29 has order three modulo 67.
    m, r, s = 67, 3, 29
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
B_SHAPE = (1, 3)               # (rb, cb)


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


def NEW_CTX(rng, G, idx):
    """Twenty coupled relative geometries, challenged in verified kernels."""
    from itertools import combinations

    card, cohort = idx % 20, idx // 20
    aplan, bplan = (2, 2), (0, 2, 2)
    plans = {"A": aplan, "B": bplan}
    acounts, bcounts = (3, 3), (3, 3, 3)
    q = G.q
    assert q == 201
    sa, sb = [list(acounts)], [list(bcounts)]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    def affine_class(t):
        """The six actual unordered affine images of {0,1,t}."""
        it = pow(t, -1, 67)
        io = pow((1-t) % 67, -1, 67)
        return min(t, (1-t) % 67, it, io,
                   ((t-1)*it) % 67, (-t*io) % 67)

    excluded = {affine_class(2), affine_class(30)}
    classes = sorted(
        {affine_class(t) for t in range(2, 67)} - excluded)
    assert len(classes) >= 10
    t = classes[(card//4 + 5*cohort) % len(classes)]
    normal_cell = [G.e, (1, 0), (t, 0)]
    normal_differences = [
        (a-b) % 67 for a in (0, 1, t) for b in (0, 1, t)
        if a != b
    ]
    assert len(set(normal_differences)) == 6

    # Coupling is relative to the normalized normal triangle, not
    # merely a new coset name or a common scalar applied to an old code.
    units = (1, 29, 37, 66, 38, 30)
    forbidden = {
        d*u % 67 for d in normal_differences for u in units
    }
    gap_pairs = [
        (a, b) for a in range(1, 34) for b in range(a+1, 34)
        if a not in forbidden and b not in forbidden
        and b*pow(a, -1, 67) % 67 not in units
    ]
    assert len(gap_pairs) >= 4
    pair_at = ((card % 4)*len(gap_pairs)//4 + 7*cohort) % len(gap_pairs)
    mixed_gaps = gap_pairs[pair_at]
    directions = {
        "A": (1, 2) if (card+cohort) % 2 else (1, 1),
        "B": ((1, 2), (1, 1), (2, 2))[cohort % 3],
    }
    triangle = {"B": t, "A": None}

    def full_relative(cell):
        """Noncommuting relatives generate this entire Frobenius group."""
        if len(set(cell)) != 3:
            return False
        u = G.mul(G.inv(cell[0]), cell[1])
        v = G.mul(G.inv(cell[0]), cell[2])
        return G.mul(u, v) != G.mul(v, u)

    def draw_cell(side, j, count):
        assert count == 3
        if side == "B" and j == 0:
            return list(normal_cell)
        if side == "B":
            d = mixed_gaps[j-1]
            direction = directions[side][j-1]
        else:
            d = rng.randrange(1, 67)
            direction = directions[side][j]
        cell = [G.e, (d, 0), (rng.randrange(67), direction)]
        assert len(set(cell)) == 3 and full_relative(cell)
        return cell

    # Every cached summand is constructed by the authoritative helper.
    # XOR linearity and disjoint column concatenation do not implement
    # an alternative left/right lift.
    mask = (1 << q)-1
    raw_a, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, raw_b, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    atoms = {
        "A": [tuple((row >> (j*q)) & mask for row in raw_a)
              for j in range(q)],
        "B": [tuple((row >> (j*q)) & mask for row in raw_b)
              for j in range(q)],
    }
    del raw_a, raw_b

    def component(side, cells):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                block = atoms[side][G.idx(g)]
                for r, value in enumerate(block):
                    rows[r] ^= value << (j*q)
        width = len(cells)*q
        columns = [0]*width
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

    def parity_challenge(record, side, budget, phase):
        """Budgeted anchored parity completions in the actual component.

        Regular component symmetry permits one root per column block.
        Searches return genuine words only. A node budget, an early
        witness stop, or an unresolved cap establishes NO exclusion.
        """
        columns = record["cols"]
        width = len(columns)
        cap = 14 if side == "B" else 12
        best = record["upper"]
        found = record.setdefault("parity_words", set())
        if found:
            found_min = min(w.bit_count() for w in found)
            best = min(best, found_min)
            found.intersection_update(
                {w for w in found if w.bit_count() == found_min})

        def note(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            if weight > cap:
                return
            if weight < best:
                best = weight
                found.clear()
            if weight == best:
                found.add(word)

        singles = {value: 1 << c for c, value in enumerate(columns)}
        pairs = {}
        for i, value in enumerate(columns):
            for j in range(i):
                signature = value ^ columns[j]
                word = (1 << i) | (1 << j)
                old = singles.get(signature)
                if old is not None:
                    note(word ^ old)
                old = pairs.get(signature)
                if old is not None:
                    note(word ^ old)
                else:
                    pairs[signature] = word
            if best <= 4:
                break

        diagnostic = random.Random(
            BASE_SEED + 1009*idx + 1900003*phase
            + (300007 if side == "B" else 0))
        priority = [diagnostic.random() for _ in columns]
        roots = list(range(0, width, q))
        diagnostic.shuffle(roots)
        all_variables = (1 << width)-1
        nodes = 0
        exhausted = False

        if best > 4:
            for root in roots:
                remaining = [max(1, budget//len(roots))]
                stopped = [False]

                def visit(value, word, allowed):
                    nonlocal nodes
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    nodes += 1
                    if value == 0:
                        note(word)
                        return
                    limit = min(cap, best)
                    left = limit-word.bit_count()
                    # Cubic columns force completion-size parity.
                    if (left ^ value.bit_count()) & 1:
                        left -= 1
                    if left <= 0 or value.bit_count() > 3*left:
                        return

                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        note(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        note(word | pair)
                    if best <= 12 or len(found) >= 24:
                        stopped[0] = True
                        remaining[0] = 0
                        return
                    left = min(cap, best)-word.bit_count()
                    if left <= 2:
                        return

                    # Every completion must meet each odd check.
                    # Branch exclusions retain coverage while avoiding
                    # repeatedly choosing the same first completion.
                    odd = value
                    choices = 0
                    smallest = width+1
                    while odd:
                        bit = odd & -odd
                        r = bit.bit_length()-1
                        available = record["rows"][r] & allowed
                        count = available.bit_count()
                        if count == 0:
                            return
                        if count < smallest:
                            choices, smallest = available, count
                        odd ^= bit
                    candidates = []
                    while choices:
                        bit = choices & -choices
                        candidates.append(bit.bit_length()-1)
                        choices ^= bit
                    candidates.sort(key=lambda c: (
                        -(columns[c] & value).bit_count(), priority[c]))
                    live = allowed
                    for c in candidates:
                        bit = 1 << c
                        live &= ~bit
                        visit(value ^ columns[c], word | bit, live)
                        if remaining[0] <= 0:
                            break

                # Earlier column blocks are forbidden, but no arbitrary
                # coordinate ordering within the root block is assumed.
                allowed = all_variables & ~((1 << root)-1)
                allowed &= ~(1 << root)
                visit(columns[root], 1 << root, allowed)
                exhausted |= remaining[0] <= 0 or stopped[0]
                if best <= 12 or len(found) >= 24:
                    exhausted = True
                    break

        record["upper"] = min(record["upper"], best)
        record["parity_nodes"] = record.get("parity_nodes", 0)+nodes
        record["parity_exhausted"] = (
            record.get("parity_exhausted", False) or exhausted)
        for word in list(found)[:4]:
            assert syndrome(columns, word) == 0
            assert all((row & word).bit_count() % 2 == 0
                       for row in record["rows"])
        return found

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
                    -record["stats"][2]-q*deficit,
                    -len(record.get("parity_words", ())))

        def probe(record, trials, offset):
            reading, word = kernel_search(
                record["rows"], record["cols"], seed+offset, trials)
            record["upper"] = min(record["upper"], reading)
            record["witness"] = word
            bank.add(word)

        starts = []
        for _ in range(72 if side == "B" else 24):
            cells = [draw_cell(side, j, count)
                     for j, count in enumerate(counts)]
            record = inspect(cells)
            if record["stats"][0] == 0:
                starts.append(record)
        if not starts:
            return []
        starts.sort(key=lambda record: record["stats"])
        pool = starts[:12 if side == "B" else 6]
        for record in pool:
            probe(record, 8 if side == "B" else 12, 0)
        for record in sorted(pool, key=rating, reverse=True)[:2]:
            probe(record, 32, 1)
        current = max(pool, key=rating)

        # B's normal triangle and mixed gaps are the card's prescription.
        # Repair relative singleton coordinates, not gauge/orbit images.
        mutable = (1, 2) if side == "B" else (0, 1)
        for step in range(36 if side == "B" else 10):
            surviving = [w for w in bank
                         if syndrome(current["cols"], w) == 0]
            active = []
            if surviving:
                word = min(surviving, key=int.bit_count)
                while word:
                    bit = word & -word
                    j = (bit.bit_length()-1)//q
                    if j in mutable:
                        active.append(j)
                    word ^= bit
            j = (rng.choice(active) if active and rng.random() < 0.85
                 else rng.choice(mutable))
            cells = [list(cell) for cell in current["cells"]]
            if rng.random() < 0.30:
                cells[j] = draw_cell(side, j, counts[j])
            else:
                term = 2 if side == "B" else rng.randrange(3)
                old = cells[j][term]
                cells[j][term] = (rng.randrange(67), old[1])
                if cells[j][term] == old or not full_relative(cells[j]):
                    continue
            proposed = inspect(cells)
            if proposed["stats"][0]:
                continue
            if proposed["stats"][1] > current["stats"][1] + 2*q:
                continue
            probe(proposed, 8 if side == "B" else 12, 0)
            if side == "B" and step % 3 == 0:
                bank.update(parity_challenge(
                    proposed, side, 6000, 17+step))
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
            bank.update(parity_challenge(
                record, side, 12000 if side == "B" else 6000, 1009))
            screened.append(record)
        screened.sort(key=rating, reverse=True)
        screened = screened[:2]
        for record in screened:
            probe(record, 128, 3)
            bank.update(parity_challenge(
                record, side, 100000 if side == "B" else 18000, 700001))
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
                    "coupled_mixed_gaps": mixed_gaps,
                    "quotient_directions": directions,
                    "parity_search_nodes": (
                        ar.get("parity_nodes", 0),
                        br.get("parity_nodes", 0)),
                    "parity_search_exhausted": (
                        ar.get("parity_exhausted", False),
                        br.get("parity_exhausted", False)),
                    "verified_cell_ranks": (
                        ar["block_ranks"], br["block_ranks"]),
                    "component_readings": (da, db), "quality": quality}
    raise ValueError("no new-shape filling passed the exact target-rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete challenged trinomial without subsequent draws."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == 3 and len({G._key(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Reallocated-kernel-pairs-v2 would yield five selected readings >=16 on Z29 (3,6)x(3,7); measured readings were 16,16,14,14,14, with abundant sector-1 minima.
# REFUTED: The long-order B-binomial principal allocation preserves trinomial-quality readings on Z67:Z3 (1,2)x(1,3): every principal card screened at <=10, while all five measured winners used only trinomials and read 12,14,14,12,14.
# REFUTED: All five normal-unit affine-triangle prescriptions admit distinct-difference fillings; parent slots 0..3 instead failed labeling before producing a code.
# REFUTED: Normal-unit trinomial selection raises every selected X reading above fourteen; parent selected readings were 16,14,14,14,14, with the weight-fourteen minima entirely in sector 1.
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
