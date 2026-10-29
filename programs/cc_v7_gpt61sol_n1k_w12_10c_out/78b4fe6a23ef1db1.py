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
CONCEPT = ("On Frobenius C53:C4(s=23), order 212, A=(1,2), B=(1,3), all-cubic cells with prescribed relative-order/translation-ratio compositions and independently challenged component witnesses should produce five new measured readings >=18, exceeding the 0.300 incumbent. At the rank bound this scores 92.5714. Failure refutes this measurement target; no distance certificate is claimed.")
GROUP = ("Jump to the faithful order-four Frobenius action: 23^2=-1 and 23^4=1 modulo 53. The group has trivial centre and differs genuinely from the order-84 parent. This centre-one order-212 bed is absent from BEDS; the listed order-212 bed has centre two. Successful order-200/210 cubic beds motivate, but do not establish, distance transfer. q=212 gives n=1484.")
SHAPES = ("Change to A=(1,2), B=(1,3): n=1484, k>=424 and guaranteed rate 2/7=0.285714 target cell 0.300; q_max=214. Components have 424/636 variables and 212 checks, nominal rates 1/2 and 2/3. Parent weight-six minima were entirely sector one, especially Acol1/2 across Bcol0..2. Remove their degree-two backbone and enforce the actual lifted-rate gate.")
FILLING = ("All five cells are cubic: HX weight nine and HZ weight twelve. Twenty cards cross five A relative-order compositions with four B normal-translation-ratio compositions. Native operations propose noncommuting triples; scaffold-derived monomial blocks supply all binary matrices. Component kernel-four exclusion also excludes odd-weight kernels by cubic parity. Geometry selection, witness-directed replacement, independent information sets and budgeted parity completion challenge readings. Final native pairings challenge localized logicals modulo original stabilizer spans, verify full ranks, kernel-four exclusion, weights and CSS, and cap selection readings at eighteen. Neither finite searches nor component exclusions certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r030_Frob53C4_12x13_cubic_ratio_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 212

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius C53:C4 requires order 212")
    # (a,b) = x^a y^b, with y*x*y^-1 = x^23.
    action = (1, 23, 52, 30)
    assert 23*23 % 53 == 52 and pow(23, 4, 53) == 1
    els = [(a,b) for b in range(4) for a in range(53)]

    def red(g):
        return (int(g[0]) % 53, int(g[1]) % 4)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + action[b]*c) % 53, (b+d) % 4)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 4
        return ((-action[bi]*a) % 53, bi)

    def idx(g):
        a,b = red(g)
        return 53*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _retired_order84_highrate_ctx(rng, G, idx):
    """Inactive order-84 experiment; never called on the new group."""
    q = G.q
    assert q == 84 and A_SHAPE == B_SHAPE == (1, 4)
    card, cohort = idx % 20, idx // 20
    model, objective = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    profiles = (
        (0, 0, 1), (0, 1, 2), (0, 1, 4),
        (0, 2, 5), (0, 3, 7),
    )
    support = [[2, 2, 2, 3]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    # Cache group operations, not a separately implemented binary lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    # Both monomial conventions come exclusively from lifted_matrices.
    monomials = [[[g] for g in G.elements]]
    xx, _, _ = lifted_matrices(G, monomials, [[[]]])
    _, zz, _ = lifted_matrices(G, [[[]]], monomials)
    mask = (1 << q)-1
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz]
              for g in range(q)],
    }

    def component(cells, side):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                block = blocks[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        columns = [0]*(4*q)
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return rows, columns

    def legal_backbone(gs):
        signed = gs + [inverse[g] for g in gs]
        if len(set(signed)) != 6 or ident in signed:
            return False
        # Unique reduced two-step products exclude backbone four-cycles.
        products = set()
        for g in signed:
            for h in signed:
                if h == inverse[g]:
                    continue
                value = table[g][h]
                if value == ident or value in products:
                    return False
                products.add(value)
        reached, pending = {ident}, [ident]
        for value in pending:
            for g in signed:
                image = table[value][g]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    def fresh_backbone():
        # Odd C12 coordinates make the verified degree-two check graph
        # bipartite. Coordinates and normal labels are independently fresh.
        for _ in range(768):
            gs = [
                G.idx((rng.randrange(7), rng.choice((1, 3, 5, 7, 9, 11))))
                for _ in range(3)
            ]
            if legal_backbone(gs):
                return [[G.e, G.elements[g]] for g in gs]
        # Checked fallback only; trinomial supports remain independently new.
        gs = [G.idx(g) for g in ((1, 9), (0, 1), (4, 5))]
        if not legal_backbone(gs):
            raise ValueError("validated metacyclic backbone unavailable")
        return [[G.e, G.elements[g]] for g in gs]

    def fresh_triangle(profile):
        for _ in range(256):
            phase = rng.randrange(12)
            cell = [(rng.randrange(7), (b+phase) % 12)
                    for b in profile]
            if len(set(cell)) != 3:
                continue
            ids = [G.idx(g) for g in cell]
            left = {table[inverse[g]][h]
                    for g in ids for h in ids if g != h}
            right = {table[g][inverse[h]]
                     for g in ids for h in ids if g != h}
            if len(left) == len(right) == 6:
                return cell
        raise ValueError("trinomial quotient recipe exhausted")

    def small_words(columns, cap=128):
        """Exact exclusion through five; positive five-counts are capped.

        After the <=4 screen, every five-word containing trinomials has
        a partition with two degree-three variables and one degree-two
        variable on its triple side. Pure-binomial five-words are absent
        because the checked backbone is bipartite.
        """
        width = len(columns)
        singles = set(columns)
        if 0 in singles or len(singles) != width:
            return None
        vb = [1 << c for c in range(width)]
        pairs = {}
        for a, x in enumerate(columns):
            for b in range(a):
                value = x ^ columns[b]
                if value in singles or value in pairs:
                    return None
                pairs[value] = vb[a] | vb[b]

        doubles = [c for c, x in enumerate(columns)
                   if x.bit_count() == 2]
        triples = [c for c, x in enumerate(columns)
                   if x.bit_count() == 3]
        assert len(doubles) == 3*q and len(triples) == q
        incidence = {}
        for c in doubles:
            for r in bits(columns[c]):
                incidence.setdefault(r, []).append(c)

        found = set()
        for at, a in enumerate(triples):
            for b in triples[:at]:
                value = columns[a] ^ columns[b]
                used = vb[a] | vb[b]
                if value.bit_count() == 6:
                    choices = set()
                    for r in bits(value):
                        choices.update(incidence.get(r, ()))
                else:
                    choices = doubles
                for c in choices:
                    previous = pairs.get(value ^ columns[c])
                    occupied = used | vb[c]
                    if previous is None or previous & occupied:
                        continue
                    word = previous | occupied
                    assert word.bit_count() == 5
                    found.add(word)
                    if len(found) >= cap:
                        return cap
        return len(found)

    def cycle_counts(columns):
        """Actual Tanner C4 count and actual binomial six-cycle count."""
        uses, four = {}, 0
        adjacency = [[] for _ in range(q)]
        for variable, column in enumerate(columns):
            ns = list(bits(column))
            for at, a in enumerate(ns):
                for b in ns[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
            if len(ns) == 2:
                a, b = ns
                adjacency[a].append((b, variable))
                adjacency[b].append((a, variable))

        # The regular group action is transitive on check vertices.
        paths, rooted = {}, 0
        root = 0
        for a, e1 in adjacency[root]:
            for b, e2 in adjacency[a]:
                if b == root or e2 == e1:
                    continue
                for end, e3 in adjacency[b]:
                    if end in (root, a) or e3 in (e1, e2):
                        continue
                    previous = paths.setdefault(end, [])
                    for c, d, f1, f2, f3 in previous:
                        if ({a, b}.isdisjoint((c, d))
                                and {e1, e2, e3}.isdisjoint((f1, f2, f3))):
                            rooted += 1
                    previous.append((a, b, e1, e2, e3))
        assert q*rooted % 6 == 0
        return four, q*rooted//6

    def kernel_probe(record, trials, offset):
        """Genuine kernel witnesses; independent readings only decrease."""
        diagnostic = random.Random(seed+offset)
        columns = record["columns"]
        order = list(range(len(columns)))
        reading = record.get("reading", len(columns)+1)
        found = set()
        for _ in range(trials):
            diagnostic.shuffle(order)
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
            light = sorted(basis, key=int.bit_count)[:20]
            for word in light:
                if word:
                    reading = min(reading, word.bit_count())
                    if word.bit_count() <= 6:
                        found.add(word)
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())
                        if candidate.bit_count() <= 6:
                            found.add(candidate)
        for word in found:
            assert all((word & row).bit_count() % 2 == 0
                       for row in record["rows"])
        record["reading"] = reading
        record["sampled_six"] = (
            record.get("sampled_six", set()) | found)

    def geometry(record):
        four, six = record["cycles"]
        if objective == 0:
            secondary = (six, four)
        elif objective == 1:
            secondary = (four, six)
        elif objective == 2:
            secondary = (six+2*four, six)
        else:
            secondary = (six+four, four)
        return (-record["five"], tuple(-x for x in secondary))

    def select(side, profile_id):
        profile = profiles[profile_id]
        pool, seen = [], set()

        def inspect(cells):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in cells)
            if signature in seen:
                return None
            seen.add(signature)
            rows, columns = component(cells, side)
            five = small_words(columns)
            if five is None or gf2_rank(rows) != q:
                return None
            return {
                "cells": cells, "rows": rows, "columns": columns,
                "five": five, "cycles": cycle_counts(columns),
                "signature": signature,
            }

        for trial in range(48):
            cells = fresh_backbone() + [fresh_triangle(profile)]
            record = inspect(cells)
            if record is not None:
                pool.append(record)
                pool.sort(key=geometry, reverse=True)
                del pool[6:]
            if len(pool) == 6 and sum(r["five"] == 0 for r in pool) >= 3:
                break
        if not pool:
            raise ValueError("empty authoritative high-rate component pool")

        current = max(pool, key=geometry)
        for move in range(12):
            cells = [list(cell) for cell in current["cells"]]
            if move % 4 == 3:
                cells[:3] = fresh_backbone()
            cells[3] = fresh_triangle(profile)
            proposed = inspect(cells)
            if proposed is None:
                continue
            pool.append(proposed)
            pool.sort(key=geometry, reverse=True)
            del pool[6:]
            if geometry(proposed) > geometry(current):
                current = proposed

        finalists = sorted(pool, key=geometry, reverse=True)[:3]
        side_offset = 0 if side == "A" else 100003
        for record in finalists:
            kernel_probe(record, 8, side_offset)
            kernel_probe(record, 24+4*(cohort % 3),
                         side_offset+700001)
        finalists.sort(key=lambda r: (
            geometry(r), r["reading"], -len(r["sampled_six"])),
            reverse=True)
        return finalists[:2]

    aa = select("A", model)
    bb = select("B", (2*model+objective+1+cohort) % 5)
    choices = [(a, b) for a in aa for b in bb]
    choices.sort(key=lambda ab: (
        -ab[0]["five"]-ab[1]["five"],
        min(ab[0]["reading"], ab[1]["reading"]),
        -ab[0]["cycles"][1]-ab[1]["cycles"][1],
        -len(ab[0]["sampled_six"])-len(ab[1]["sampled_six"]),
        -ab[0]["cycles"][0]-ab[1]["cycles"][0]),
        reverse=True)

    for a, b in choices:
        A, B = [a["cells"]], [b["cells"]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 17*q == 1428
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= 9*q and 40*k > 21*n):
            continue
        actual_a = [
            sum(((row >> (4*j*q)) & mask) << (j*q)
                for j in range(4))
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << (4*q))-1) for row in hz[:q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert max(row.bit_count() for row in hx+hz) == 12
        assert css_ok(hx, hz)
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (model, objective, cohort),
            "component_five_counts": (a["five"], b["five"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "component_readings": (a["reading"], b["reading"]),
        }
    raise ValueError("screened pairings missed the open high-rate cell")


def _f53_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length()-1
        word ^= bit


def _f53_columns(rows, width):
    columns = [0]*width
    for r, row in enumerate(rows):
        for c in _f53_bits(row):
            columns[c] |= 1 << r
    return columns


def _f53_no4(columns):
    """Exact exclusion of nonzero classical kernel words through four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = singles.copy()
    for a, value in enumerate(columns):
        for b in range(a):
            combined = value ^ columns[b]
            if combined in seen:
                return False
            seen.add(combined)
    return True


def _f53_syndrome(columns, word):
    answer = 0
    for c in _f53_bits(word):
        answer ^= columns[c]
    return answer


def _f53_probe(record, seed, trials):
    """Retain genuine kernel witnesses across fresh information sets."""
    local = random.Random(seed)
    columns = record["columns"]
    order = list(range(len(columns)))
    retained = set(record.get("words", ()))
    best = min(retained, key=int.bit_count) if retained else None

    def observe(word):
        nonlocal best
        if word and (best is None or
                     word.bit_count() < best.bit_count()):
            best = word

    for word in retained:
        assert _f53_syndrome(columns, word) == 0
    for turn in range(trials):
        local.shuffle(order)
        if retained and turn % 5 == 4:
            anchor = local.choice(sorted(retained, key=int.bit_count)[:8])
            active = list(_f53_bits(anchor))
            front = local.sample(active, max(1, len(active)//2))
            chosen = set(front)
            order = front + [c for c in order if c not in chosen]

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
                observe(word)

        light = sorted(set(basis) | retained,
                       key=int.bit_count)[:12]
        grown = list(light)
        for word in light:
            for other in basis:
                proposed = word ^ other
                observe(proposed)
                if proposed and proposed.bit_count() <= best.bit_count()+4:
                    grown.append(proposed)
            if len(grown) > 192:
                grown = sorted(set(grown), key=int.bit_count)[:32]

        moves = basis + list(retained)
        local.shuffle(moves)
        for initial in light[:4]:
            word = initial
            for sweep in range(2):
                changed = False
                for other in moves:
                    proposed = word ^ other
                    if proposed and proposed.bit_count() < word.bit_count():
                        word, changed = proposed, True
                observe(word)
                if not changed:
                    break
            grown.append(word)

        retained = set(sorted(set(grown) | retained | {best},
                              key=int.bit_count)[:24])
        if best.bit_count() <= 8:
            break

    assert best and _f53_syndrome(columns, best) == 0
    record["words"] = sorted(retained | {best},
                             key=lambda w: (w.bit_count(), w))[:24]
    record["reading"] = min(record.get("reading", len(columns)+1),
                            best.bit_count())
    return best


def NEW_CTX(rng, G, idx):
    """Twenty cubic composition cards on the new faithful Frobenius bed."""
    q = G.q
    assert q == 212 and A_SHAPE == (1,2) and B_SHAPE == (1,3)
    card, cohort = idx % 20, idx // 20
    ap, bp = divmod(card, 4)
    seed = rng.randrange(1 << 62) ^ (1000003*(idx+1) + 0x53C41213)
    local = random.Random(seed)
    mask = (1 << q)-1
    ident = G.idx(G.e)
    sa, sb = [[3,3]], [[3,3,3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert sum(sa[0])+max(sb[0]) == 9
    assert sum(sb[0])+max(sa[0]) == 12
    assert _no_dead_lines(sa, sb)

    # Types have relative-order multisets (2,4,4), (4,4,53),
    # and (2,2,53), respectively. These are not common relabelings.
    a_profiles = ((0,0), (0,1), (1,1), (0,2), (1,2))
    # Distinct normal-translation ratios, modulo a common scaling/sign,
    # prescribe relations among three B cells rather than orbit copies.
    b_ratios = ((1,2,3), (1,2,5), (1,3,7), (1,4,9))
    a_plan = a_profiles[ap]
    b_plan = b_ratios[bp]

    table = [[G.idx(G.mul(g,h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        # Cache native operations only; binary actions still come from
        # the scaffold's verified constructor.
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverse[g]

    P = Probe()
    blocks = {"A": [], "B": []}
    # Batches avoid constructing a huge unused dummy-sector matrix.
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start+16)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    def fresh_cell(side, j):
        if side == "B":
            cell = [ident, G.idx((b_plan[j],0)),
                    G.idx((local.randrange(53),1))]
        else:
            kind = a_plan[j]
            if kind == 0:
                for attempt in range(128):
                    u = G.idx((local.randrange(53),1))
                    v = G.idx((local.randrange(53),2))
                    if table[u][v] != table[v][u]:
                        break
                else:
                    raise ValueError("noncommuting C4/C2 cubic exhausted")
                cell = [ident,u,v]
            else:
                cell = [ident, G.idx((local.randrange(1,53),0)),
                        G.idx((local.randrange(53),1 if kind == 1 else 2))]
        assert len(set(cell)) == 3
        assert table[cell[1]][cell[2]] != table[cell[2]][cell[1]]
        return cell

    def fresh(side):
        return [fresh_cell(side,j)
                for j in range(A_SHAPE[1] if side == "A" else B_SHAPE[1])]

    def signature(cells):
        # Internal gauge/permutation deduplication only. The evaluator
        # remains responsible for arbitrary full Tanner isomorphisms.
        normalized = []
        for cell in cells:
            normalized.append(min(
                tuple(sorted(table[g][inverse[root]] for g in cell))
                for root in cell))
        return tuple(sorted(normalized))

    def inspect(cells, side):
        width = len(cells)*q
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                action = blocks[side][g]
                for r, value in enumerate(action):
                    rows[r] ^= value << (j*q)
        columns = _f53_columns(rows, width)
        assert all(c.bit_count() == 3 for c in columns)

        uses, four = {}, 0
        adjacency = [0]*q
        for column in columns:
            checks = list(_f53_bits(column))
            for at, a in enumerate(checks):
                adjacency[a] |= column ^ (1 << a)
                for b in checks[:at]:
                    key = (b,a)
                    four += uses.get(key,0)
                    uses[key] = uses.get(key,0)+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in _f53_bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        # Projection excess is a geometry heuristic, not a distance.
        excess = max(0, triangles-width)
        return dict(cells=[list(c) for c in cells], rows=rows,
                    columns=columns, cycles=(four,excess),
                    signature=signature(cells), side=side)

    def parity_completion(record, offset, budget):
        """Budgeted circuit challenge through sixteen; no exclusion claim."""
        columns = record["columns"]
        width = len(columns)
        best = min(record["words"], key=int.bit_count)
        cap = min(16, best.bit_count()-2)
        if cap < 6:
            return
        diagnostic = random.Random(seed+offset)
        singles = {value: 1 << c for c,value in enumerate(columns)}
        pairs = {}
        incidence = [0]*q
        for c,value in enumerate(columns):
            for r in _f53_bits(value):
                incidence[r] |= 1 << c
            for d in range(c):
                pairs[value ^ columns[d]] = (1 << c) | (1 << d)

        priorities = [diagnostic.random() for _ in columns]
        roots = list(range(0,width,q))
        diagnostic.shuffle(roots)
        all_variables = (1 << width)-1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                assert _f53_syndrome(columns,word) == 0
                best = word

        for root in roots:
            remaining = [max(1,budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    observe(word)
                    return
                left = min(cap,best.bit_count()-2)-word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value,0)
                if one and one & allowed == one:
                    observe(word | one)
                pair = pairs.get(value,0)
                if left >= 2 and pair and pair & allowed == pair:
                    observe(word | pair)
                left = min(cap,best.bit_count()-2)-word.bit_count()
                if left <= 2:
                    return

                chosen, smallest = 0,width+1
                for r in _f53_bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen,smallest = available,count
                choices = list(_f53_bits(chosen))
                choices.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(),priorities[c]))
                live = allowed
                for c in choices:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c],word | bit,live)
                    if remaining[0] <= 0:
                        break

            # Valid classical regular symmetry supplies one root per block.
            visit(columns[root],1 << root,
                  all_variables ^ (1 << root))
            if best.bit_count() <= 8:
                break
        record["words"] = sorted(set(record["words"]) | {best},
                                 key=int.bit_count)[:24]
        record["reading"] = min(record["reading"],best.bit_count())

    def component_pool(side):
        offset = 100003*int(side == "B")
        seen, records, known = set(),[],set()

        def geometry(record):
            four, excess = record["cycles"]
            return ((24,48,12)[cohort % 3]*four+excess, four, excess)

        def merit(record):
            # Every transferred witness is checked against THIS matrix.
            surviving = [word for word in known
                         if _f53_syndrome(record["columns"],word) == 0]
            if surviving:
                record["words"] = sorted(
                    set(record.get("words",())) | set(surviving),
                    key=int.bit_count)[:24]
                record["reading"] = min(record["reading"],
                                       min(map(int.bit_count,surviving)))
            return (min(18,record["reading"]),
                    record["reading"],
                    tuple(-x for x in geometry(record)))

        proposals = []
        for attempt in range(64):
            record = inspect(fresh(side),side)
            sig = record["signature"]
            if sig in seen:
                continue
            seen.add(sig)
            proposals.append(record)
        proposals.sort(key=geometry)

        for at,record in enumerate(proposals):
            if not _f53_no4(record["columns"]):
                continue
            # Cubic parity prohibits odd-weight kernel words, so this
            # additionally excludes weights one through five.
            record["rank"] = gf2_rank(record["rows"])
            _f53_probe(record,seed+offset+1009*at,6)
            records.append(record)
            known.update(w for w in record["words"]
                         if w.bit_count() <= 18)
            if len(records) >= 8:
                break
        if not records:
            raise ValueError("new cubic card missed component kernel gates")

        current = max(records,key=merit)
        for step in range(14):
            cells = [list(c) for c in current["cells"]]
            word = local.choice(current["words"][:6])
            blocks_used = [c//q for c in _f53_bits(word)]
            j = local.choice(blocks_used) if blocks_used else local.randrange(len(cells))
            cells[j] = fresh_cell(side,j)
            proposed = inspect(cells,side)
            sig = proposed["signature"]
            if sig in seen:
                continue
            seen.add(sig)
            if not _f53_no4(proposed["columns"]):
                continue
            proposed["rank"] = gf2_rank(proposed["rows"])
            proposed["words"] = [
                w for w in current["words"]
                if _f53_syndrome(proposed["columns"],w) == 0]
            _f53_probe(proposed,seed+offset+300007+1013*step,6)
            records.append(proposed)
            known.update(w for w in proposed["words"]
                         if w.bit_count() <= 18)
            if merit(proposed) >= merit(current):
                current = proposed

        finalists = sorted(records,key=merit,reverse=True)[:3]
        for at,record in enumerate(finalists):
            _f53_probe(record,seed+offset+700001+1013*at,24)
            parity_completion(record,offset+1700021+1013*at,12000)
        finalists.sort(key=merit,reverse=True)
        finalists = finalists[:2]
        for at,record in enumerate(finalists):
            # Fresh independent orders retain all earlier witnesses.
            _f53_probe(record,seed+offset+2700011+1019*at,64)
            parity_completion(record,offset+3700033+1019*at,24000)
            assert all(_f53_syndrome(record["columns"],w) == 0
                       for w in record["words"])
        finalists.sort(key=merit,reverse=True)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                previous = pivots.get(lead)
                if previous is None:
                    pivots[lead] = word
                    break
                word ^= previous
        return pivots

    def outside(word,pivots):
        while word:
            previous = pivots.get(word.bit_length()-1)
            if previous is None:
                return True
            word ^= previous
        return False

    def localized_readings(a,b,hx,hz,n):
        """Challenge actual logicals modulo ORIGINAL stabilizer spans."""
        answer = []
        for side,record,opposite,stabilizers in (
                ("A",a,hx,hz),("B",b,hz,hx)):
            columns = _f53_columns(opposite,n)
            pivots = span(stabilizers)
            candidates = set()
            for word in record["words"][:4]:
                locations = 3 if side == "A" else 2
                for fixed in range(locations):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j+fixed)*q) for j in range(2))
                    else:
                        actual = word << (fixed*3*q)
                    assert _f53_syndrome(columns,actual) == 0
                    if outside(actual,pivots):
                        candidates.add(actual)

            reading = n+1
            for actual in sorted(candidates,key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:28]
                    best_word,best_weight = actual,actual.bit_count()
                    for at,first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:at]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word,best_weight = proposed,weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual,pivots)
                assert _f53_syndrome(columns,actual) == 0
                reading = min(reading,actual.bit_count())
            answer.append(reading)
        return tuple(answer)

    aa,bb = component_pool("A"),component_pool("B")
    survivors = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
            hx,hz,n = lifted_matrices(G,A,B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            assert n == 1484 and k >= 424
            if not (275*n <= 1000*k < 325*n):
                continue
            ar,br = _component_rows(hx,hz,q)
            assert ar == a["rows"] and br == b["rows"]
            logicals = localized_readings(a,b,hx,hz,n)
            detected = min(18,a["reading"],b["reading"],*logicals)
            merit = (
                k*detected*detected,detected,
                a["reading"]+b["reading"],
                -a["cycles"][0]-b["cycles"][0],
                -a["cycles"][1]-b["cycles"][1])
            survivors.append((merit,A,B,hx,hz,k,a,b,logicals))

    survivors.sort(key=lambda item:item[0],reverse=True)
    for merit,A,B,hx,hz,k,a,b,logicals in survivors:
        if not _f53_no4(_f53_columns(hx,1484)):
            continue
        if not _f53_no4(_f53_columns(hz,1484)):
            continue
        assert max(row.bit_count() for row in hx) == 9
        assert max(row.bit_count() for row in hz) == 12
        assert css_ok(hx,hz)
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A,B) for row in grid for cell in row)
        return dict(
            idx=idx,cells={"A":A,"B":B},k=k,
            card=(ap,bp,cohort),target_cell=0.300,
            A_relative_order_composition=a_plan,
            B_normal_translation_ratios=b_plan,
            component_ranks=(a["rank"],b["rank"]),
            component_readings=(a["reading"],b["reading"]),
            component_cycles=(a["cycles"],b["cycles"]),
            verified_localized_logical_readings=logicals,
            conservative_selection=merit,
            screen="authoritative cubic kernels <=5 excluded; full kernels <=4 excluded")
    raise ValueError("new cubic pairings missed full kernel/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the stored complete cubic cell, without orbit generation."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Row-six/row-nine density reallocation and deeper kernel-pair screening yield five selected readings >=16; the measured current program returned 14,14,16,14,16, all with sector-1 minimum witnesses.
# REFUTED: Reducing genuine six-witness multiplicity on C7:C12(s=2) (1,4)x(1,4) raises fitness above 19.059; all five measured parent codes instead retained k=756 and distance six, with abundant sector-one minima.
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
