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
CONCEPT = ("Hypothesis: on frozen split-metacyclic C7:C12(s=2), order 84, A=(1,3), B=(2,5), jointly four-cycle-free A trinomials, actual eight-cycle selection and verified C2/C3-fixed witness challenges produce five new measured readings >=12 in cell 0.350. Five readings twelve at k=504 score 50.8235, above the incumbent 47.802. This is a falsifiable measurement target, not a distance certificate.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged: m=7,r=12,s=2 and q=84. The incumbent measured four selected readings twelve and one ten, with abundant sector-one minima. Retain this feasible bed rather than assume a group change transfers distance. Its central C2 and complement C3 supply restricted witness challenges through sums of authoritative component columns; all search readings remain provisional upper bounds.")
SHAPES = ("Keep A=(1,3), B=(2,5) byte-for-byte unchanged. q=84 gives n=1428, k>=504 and guaranteed rate 6/17=0.352941 in cell 0.350; q_max=88. Unequal components have 252 variables/84 checks and 420 variables/168 checks, nominal rates 2/3 and 3/5. Parent minima were entirely sector one: Z words spanned Acol0..2, particularly at Bcol2..4; slot 58 also exposed X words at Bcol2/4. Strengthen the A/Z bottleneck while retaining the lengths and exact measured-rate gate.")
FILLING = ("Keep A counts (3,3,3), B column degrees (2,3,3,3,3) and B row totals seven: HX<=12 and HZ<=10. Twenty cards cross five B concentration budgets with four complete relative-C12 A profile mixtures. A cells have bidirectional distinct differences and noncommuting relative generators; their right-difference sets must be mutually disjoint, independently checked as zero actual A Tanner C4. Actual simple C8 counts augment C6 geometry. Exact central-C2-fixed searches through five compressed variables and complement-C3-fixed searches through four return expanded syndrome-verified witnesses through weights ten and twelve. C3 weight-twelve multiplicity is a secondary objective, never a distance certificate. Wider A portfolios, profile-preserving repair and independent challenges retain all binding readings. Final native-lift agreement, actual ranks, weights and CSS remain mandatory.")
HOW_AB = FILLING
FAMILY_NAME = "r035_m7r12s2_13x25_jointzero4_C8_C2C3_witness_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Same split-metacyclic rule, resized for the factor-seventeen shape.
    # The action has order three: 2^3=1 modulo 7.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 84")
    m, r, s = 7, 12, 2
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
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


def _retired_highrate_ctx(rng, G, idx):
    """Inactive historical high-rate experiment; not called by NEW_CTX."""
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


def NEW_CTX(rng, G, idx):
    """Twenty jointly constrained A/support cards with binding challenges."""
    from functools import lru_cache
    from math import exp

    q = G.q
    assert q == 84 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    zero_budget, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    ident = G.idx(G.e)
    # Complete relative C12 profiles, not merely occupancies modulo three.
    # Independent normal coordinates change matrices, not just cell gauges.
    a_plans = (
        ((0, 0, 1), (0, 1, 2), (0, 1, 6)),
        ((0, 0, 1), (0, 1, 3), (0, 1, 3)),
        ((0, 1, 2), (0, 1, 6), (0, 1, 6)),
        ((0, 0, 1), (0, 0, 1), (0, 1, 3)),
    )

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Native operation caching only. Binary blocks come exclusively from
    # the authoritative construction helper, with its original indexing.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    monomials = [[[g] for g in range(q)]]
    xx, _, _ = lifted_matrices(probe, monomials, [[[]]])
    _, zz, _ = lifted_matrices(probe, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz]
              for g in range(q)],
    }
    del xx, zz

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid native group order")

    def support_b():
        degrees = (2, 3, 3, 3, 3)

        @lru_cache(None)
        def feasible(j, upper, zeros):
            if upper < 0 or zeros < 0:
                return False
            if j == len(degrees):
                return upper == zeros == 0
            degree = degrees[j]
            return any(
                feasible(j+1, upper-a,
                         zeros-int(a == 0 or a == degree))
                for a in range(degree+1)
            )

        assert feasible(0, 7, zero_budget)
        upper, zeros = 7, zero_budget
        grid = [[], []]
        for j, degree in enumerate(degrees):
            choices = [
                a for a in range(degree+1)
                if feasible(j+1, upper-a,
                            zeros-int(a == 0 or a == degree))
            ]
            a = rng.choice(choices)
            grid[0].append(a)
            grid[1].append(degree-a)
            upper -= a
            zeros -= int(a == 0 or a == degree)
        assert [sum(row) for row in grid] == [7, 7]
        assert _grid_weights([[3, 3, 3]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[3, 3, 3]], grid)
        return grid

    def cell_quality(cell, mode):
        left, right, relative = set(), set(), []
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
                    relative.append(orders[table[inverses[a]][b]])
        if len(left) != len(cell)*(len(cell)-1):
            return None
        if len(right) != len(left) or any(o <= 3 for o in relative):
            return None
        targets = (
            {7, 12, 28}, {12, 28}, {7, 14, 28}, {4, 7, 12, 28}
        )
        return sum(o not in targets[mode] for o in relative)

    def fresh_cell(count, mode, bridge=False):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            wanted = (
                (7, 12, 28), (12, 28), (7, 14, 28), (4, 7, 12, 28)
            )[mode]
            eligible = [g for g in range(q) if orders[g] in wanted]
            root = rng.randrange(q)
            return [root, table[root][rng.choice(eligible)]]

        assert count == 3
        patterns = ((0, 1, 2), (0, 0, 1), (0, 1, 1), None)
        # bridge is now the one-based A-column index.
        pattern = a_plans[policy][bridge-1] if bridge else patterns[mode]
        best = None
        for attempt in range(192 if bridge else 96):
            if pattern is None:
                cell = rng.sample(range(q), 3)
            else:
                phase = rng.randrange(12 if bridge else 3)
                native = [
                    (rng.randrange(7),
                     (b+phase) % 12 if bridge else
                     (b+phase) % 3 + 3*rng.randrange(4))
                    for b in pattern
                ]
                if len(set(native)) != 3:
                    continue
                cell = [G.idx(g) for g in native]
            quality = cell_quality(cell, mode)
            if quality is None:
                continue
            if bridge:
                u = table[inverses[cell[0]]][cell[1]]
                v = table[inverses[cell[0]]][cell[2]]
                if table[u][v] == table[v][u]:
                    continue
            if best is None or quality < best[0]:
                best = quality, cell
                if quality == 0:
                    break
        if best is not None:
            return best[1]
        if bridge:
            raise ValueError("noncommuting full-C12 A profile exhausted")
        # B retains a legal broad fallback; actual matrices decide quality.
        return rng.sample(range(q), 3)

    def right_differences(cell):
        return {table[a][inverses[b]]
                for a in cell for b in cell if a != b}

    def fresh(side):
        mode = (policy + int(side == "B") + cohort) % 4
        if side == "A":
            # For the authoritative left lift, disjoint right differences
            # exclude cross-column C4. record verifies the actual count.
            for attempt in range(1024):
                if attempt % 32 == 0:
                    first = fresh_cell(3, mode, 1)
                    used_first = right_differences(first)
                second = fresh_cell(3, mode, 2)
                used_second = right_differences(second)
                if not used_first.isdisjoint(used_second):
                    continue
                third = fresh_cell(3, mode, 3)
                used_third = right_differences(third)
                if (used_first | used_second).isdisjoint(used_third):
                    return [[first, second, third]]
            raise ValueError("jointly four-cycle-free A profile exhausted")
        support = support_b()
        return [
            [fresh_cell(count, mode) for count in row]
            for row in support
        ]

    def record(grid, side):
        """GF2-linear superposition of scaffold-built monomial blocks."""
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = blocks[side][g]
                    for t, value in enumerate(block):
                        rows[i*q+t] ^= value << (j*q)

        columns = [0]*(nc*q)
        for r, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << r

        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for column in columns:
            checks = list(bits(column))
            degree = len(checks)
            forced += degree*(degree-1)*(degree-2)//6
            for at, a in enumerate(checks):
                adjacency[a] |= column ^ (1 << a)
                for b in checks[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        # Projection excess is only a geometry heuristic when C4 exist.
        excess = max(0, triangles-forced)
        peak = max((v.bit_count() for v in adjacency), default=0)
        eight = 0
        if side == "A" and four == 0:
            # Exact simple Tanner C8: pair internally disjoint length-four
            # paths. Regular component symmetry is transitive per block.
            checks = [list(bits(row)) for row in rows]
            neighbors = [list(bits(column)) for column in columns]
            rooted = 0
            for root in range(0, nc*q, q):
                paths = {}
                for c1 in neighbors[root]:
                    for middle in checks[c1]:
                        if middle == root:
                            continue
                        for c2 in neighbors[middle]:
                            if c2 == c1:
                                continue
                            for end in checks[c2]:
                                if end == root or end == middle:
                                    continue
                                previous = paths.setdefault(end, [])
                                for d1, other, d2 in previous:
                                    if (other != middle and c1 != d1
                                            and c1 != d2 and c2 != d1
                                            and c2 != d2):
                                        rooted += 1
                                previous.append((c1, middle, c2))
            assert q*rooted % 4 == 0
            eight = q*rooted//4
        if side == "A":
            base = (1000000*four
                    + (60, 120, 240, 480)[policy]*excess + eight)
        else:
            base = (500*(1+policy)*four + excess
                    + (0, 2*peak, excess, 6*peak)[policy])
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cycles=(four, excess, peak, eight), base=base)

    def syndrome(columns, word):
        value = 0
        for v in bits(word):
            value ^= columns[v]
        return value

    def short_word(columns):
        """Return a real witness, or exactly exclude kernel weights <=4."""
        singles = {}
        for v, column in enumerate(columns):
            bit = 1 << v
            if not column:
                return bit
            previous = singles.get(column)
            if previous is not None:
                return bit ^ previous
            singles[column] = bit
        pairs = {}
        for a, column in enumerate(columns):
            for b in range(a):
                value = column ^ columns[b]
                word = (1 << a) | (1 << b)
                previous = singles.get(value)
                if previous is None:
                    previous = pairs.get(value)
                if previous is not None:
                    return word ^ previous
                pairs[value] = word
        return None

    # Restrictions use sums of authoritative columns, never a new lift.
    # A-left components commute with right multiplication.
    fixed_orbits = {}
    for degree, subgroup in (
            (2, [ident, G.idx((0, 6))]),
            (3, [G.idx((0, b)) for b in (0, 4, 8)])):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {table[root][h] for h in subgroup}
            assert len(orbit) == degree and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        fixed_orbits[degree] = orbits

    def fixed_challenges(rec):
        """Exact restricted challenges, not unrestricted distance claims.

        C2: all compressed words through five, expanded weight <=10.
        C3: all compressed words through four, expanded weight <=12.
        Count every C3 weight-four word when no lighter one exists.
        """
        if "fixed_challenge_words" in rec:
            return rec["fixed_challenge_words"]
        assert len(rec["rows"]) == q
        actual_words = []
        for degree in (2, 3):
            expansions = [
                sum(1 << (j*q+t) for t in orbit)
                for j in range(3) for orbit in fixed_orbits[degree]
            ]
            columns = [
                syndrome(rec["columns"], expansion)
                for expansion in expansions
            ]
            word = short_word(columns)
            count12 = 0
            if degree == 3 and word is not None:
                if word.bit_count() < 4:
                    count12 = 1000000
                else:
                    # With <=3 excluded, pair collisions are disjoint
                    # weight-four words. Deduplicate their three partitions.
                    pairs, found = {}, set()
                    for a, value in enumerate(columns):
                        for b in range(a):
                            key = value ^ columns[b]
                            pair = (1 << a) | (1 << b)
                            previous = pairs.setdefault(key, [])
                            for old in previous:
                                candidate = old ^ pair
                                assert candidate.bit_count() == 4
                                found.add(candidate)
                            previous.append(pair)
                    count12 = len(found)
                    assert found
                    word = min(found)
            elif degree == 2 and word is None:
                # <=4 are excluded, so each pair syndrome is unique.
                pairs = {
                    columns[a] ^ columns[b]: (1 << a) | (1 << b)
                    for a in range(len(columns)) for b in range(a)
                }
                for a, value in enumerate(columns):
                    for b in range(a):
                        partial = value ^ columns[b]
                        for c in range(b):
                            old = pairs.get(partial ^ columns[c])
                            if old is not None:
                                triple = (1 << a) | (1 << b) | (1 << c)
                                assert not old & triple
                                word = old | triple
                                assert word.bit_count() == 5
                                break
                        if word is not None:
                            break
                    if word is not None:
                        break
            if degree == 3:
                rec["C3_fixed_weight12_count"] = count12
            if word is not None:
                actual = 0
                for c in bits(word):
                    actual ^= expansions[c]
                assert actual.bit_count() == degree*word.bit_count()
                assert syndrome(rec["columns"], actual) == 0
                actual_words.append(actual)
        rec["fixed_challenge_words"] = actual_words
        return actual_words

    def kernel_probe(rec, trials, diagnostic_seed):
        """Binding genuine witnesses, including exact restricted challenges."""
        diagnostic = random.Random(diagnostic_seed)
        columns = rec["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(rec.get("words", ()))
        if len(rec["rows"]) == q:
            retained.extend(fixed_challenges(rec))
        witness = short_word(columns)

        def observe(word):
            nonlocal witness
            if word and (witness is None or
                         word.bit_count() < witness.bit_count()):
                witness = word

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
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
                    observe(word)
            light = sorted(set(retained+basis),
                           key=int.bit_count)[:16]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= 14:
                        additions.append(candidate)
            moves = retained+basis
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for sweep in range(2):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
                additions.append(word)
            retained = sorted(set(retained+light+additions),
                              key=int.bit_count)[:32]
            if witness is not None and witness.bit_count() <= 4:
                break

        assert witness is not None
        # Budget-limited parity completion challenges elimination readings.
        # A budget stop is never interpreted as absence of lighter words.
        cap = min(12, witness.bit_count()-1)
        if cap >= 5:
            singles, pairs = {}, {}
            incidence = [0]*len(rec["rows"])
            for v, column in enumerate(columns):
                singles.setdefault(column, 1 << v)
                for r in bits(column):
                    incidence[r] |= 1 << v
                for u in range(v):
                    pairs.setdefault(column ^ columns[u],
                                     (1 << v) | (1 << u))
            roots = list(range(0, width, q))
            diagnostic.shuffle(roots)
            priorities = [diagnostic.random() for _ in columns]
            all_variables = (1 << width)-1
            budget = (min(160000, 4000*trials)
                      if len(rec["rows"]) == q else
                      min(48000, 1200*trials))

            for root in roots:
                remaining = [max(1, budget//len(roots))]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    if not value:
                        observe(word)
                        return
                    left = min(cap, witness.bit_count()-1)-word.bit_count()
                    if left <= 0 or value.bit_count() > 3*left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        observe(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        observe(word | pair)
                    left = min(cap, witness.bit_count()-1)-word.bit_count()
                    if left <= 2:
                        return
                    chosen, smallest = 0, width+1
                    for r in bits(value):
                        available = incidence[r] & allowed
                        count = available.bit_count()
                        if not count:
                            return
                        if count < smallest:
                            chosen, smallest = available, count
                    candidates = list(bits(chosen))
                    candidates.sort(key=lambda v: (
                        -(columns[v] & value).bit_count(), priorities[v]))
                    live = allowed
                    for v in candidates:
                        bit = 1 << v
                        live &= ~bit
                        visit(value ^ columns[v], word | bit, live)
                        if remaining[0] <= 0:
                            break

                visit(columns[root], 1 << root,
                      all_variables & ~((1 << (root+1))-1))
                if witness.bit_count() <= 4:
                    break

        assert witness and syndrome(columns, witness) == 0
        rec["words"] = sorted(set(retained+[witness]),
                              key=int.bit_count)[:32]
        rec["d"] = min(rec.get("d", width+1), witness.bit_count())
        return witness

    def component_pool(side):
        known, screened = [], []
        offset = 100003*int(side == "B")
        signatures = set()

        def loss(rec):
            penalty = 0
            for word in known:
                if syndrome(rec["columns"], word) == 0:
                    penalty += 4000*max(1, 13-word.bit_count())**2
            return rec["base"]+penalty

        def optimize(grid, steps):
            current = record(grid, side)
            best = current
            current_cost = best_cost = loss(current)
            mutable = [(i, j) for i, row in enumerate(grid)
                       for j, cell in enumerate(row) if cell]
            mode = (policy+int(side == "B")+cohort) % 4
            for step in range(steps):
                positions = mutable
                active = [w for w in known
                          if syndrome(current["columns"], w) == 0]
                if active and step % 3 != 2:
                    occupied = {v//q for v in bits(rng.choice(active))}
                    targeted = [p for p in mutable if p[1] in occupied]
                    if targeted:
                        positions = targeted
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                proposal = None
                if len(old) >= 2 and step % 3:
                    proposal = list(old)
                    t = rng.randrange(len(old))
                    available = [
                        g for g in range(q) if g not in old
                        and (side != "A" or
                             G.elements[g][1] == G.elements[old[t]][1])
                    ]
                    proposal[t] = rng.choice(available)
                    if cell_quality(proposal, mode) is None:
                        proposal = None
                    elif side == "A":
                        u = table[inverses[proposal[0]]][proposal[1]]
                        v = table[inverses[proposal[0]]][proposal[2]]
                        if table[u][v] == table[v][u]:
                            proposal = None
                grid[i][j] = (
                    proposal if proposal is not None else
                    fresh_cell(len(old), mode, j+1 if side == "A" else 0))
                trial = record(grid, side)
                if side == "A" and trial["cycles"][0]:
                    grid[i][j] = old
                    continue
                trial_cost = loss(trial)
                temperature = 70*(1-step/steps)**2+0.2
                if trial_cost <= current_cost or rng.random() < exp(
                        min(0.0, (current_cost-trial_cost)/temperature)):
                    current, current_cost = trial, trial_cost
                    if trial_cost < best_cost:
                        best, best_cost = trial, trial_cost
                else:
                    grid[i][j] = old
            return best

        def merit(rec):
            return (
                rec["free4"], rec["d"],
                -rec.get("C3_fixed_weight12_count", 0),
                -loss(rec), rec["rank"])

        stages = 3 if side == "A" else 2
        pool = [
            optimize(fresh(side), 48)
            for _ in range(8 if side == "A" else 4)
        ]
        for stage in range(stages):
            tested = []
            for rec in sorted(pool, key=loss):
                signature = tuple(tuple(sorted(cell))
                                  for row in rec["grid"] for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                if side == "A":
                    assert rec["cycles"][0] == 0
                word = short_word(rec["columns"])
                rec["free4"] = word is None
                if word is not None:
                    rec["words"] = [word]
                rec["rank"] = gf2_rank(rec["rows"])
                word = kernel_probe(
                    rec, 8, seed+offset+1009*stage)
                tested.append(rec)
                screened.append(rec)
                # Restricted witnesses are binding repair targets too.
                learned = [word] + rec["words"][:4]
                for other in learned:
                    if other.bit_count() <= 12 and other not in known:
                        known.append(other)
                known.sort(key=lambda w: (w.bit_count(), w))
                del known[48:]
            tested.sort(key=merit, reverse=True)
            if stage < stages-1:
                seeds = tested[:2] if tested else pool[:2]
                pool = [
                    optimize(copy_grid(r["grid"]), 80) for r in seeds
                ] + [
                    optimize(fresh(side), 48)
                    for _ in range(4 if side == "A" else 2)
                ]

        if not screened:
            raise ValueError("empty jointly constrained component portfolio")
        screened.sort(key=merit, reverse=True)
        finalists = screened[:5 if side == "A" else 3]
        for rec in finalists:
            kernel_probe(rec, 32, seed+offset+700001)
        finalists.sort(key=merit, reverse=True)
        # Every returned A candidate receives independent deeper testing.
        for rec in (finalists if side == "A" else finalists[:2]):
            kernel_probe(rec, 64, seed+offset+1700021)
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

    aa, bb = component_pool("A"), component_pool("B")
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        ab[0]["free4"] and ab[1]["free4"],
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["base"]-ab[1]["base"]), reverse=True)
    best = None
    for a, b in proposals:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1428 and k >= 504
                and 325*n <= 1000*k < 375*n):
            continue
        detected = min(14, a["d"], b["d"])
        key = (a["free4"] and b["free4"],
               k*detected*detected, detected,
               -a["C3_fixed_weight12_count"],
               a["d"]+b["d"], -a["base"]-b["base"])
        if best is None or key > best[0]:
            best = key, A, B, hx, hz, k, a, b
    if best is None:
        raise ValueError("cubic pairings missed the actual 0.350 rate gate")

    quality, A, B, hx, hz, k, a, b = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert sa == [[3, 3, 3]]
    assert [sum(row) for row in sb] == [7, 7]
    assert [sum(row[j] for row in sb) for j in range(5)] == [2, 3, 3, 3, 3]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    authoritative_a = [
        sum(((hx[t] >> (j*5*q)) & mask) << (j*q)
            for j in range(3)) for t in range(q)]
    authoritative_b = [
        row & ((1 << (5*q))-1) for row in hz[:2*q]]
    assert authoritative_a == a["rows"]
    assert authoritative_b == b["rows"]
    assert a["cycles"][0] == 0
    for word in a["fixed_challenge_words"]:
        assert syndrome(authoritative_a and a["columns"], word) == 0
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert css_ok(hx, hz)
    return {
        "idx": idx, "cells": {"A": A, "B": B}, "k": k,
        "card": (zero_budget, policy, cohort), "quality": quality,
        "component_readings": (a["d"], b["d"]),
        "component_kernel4_free": (a["free4"], b["free4"]),
        "component_ranks": (a["rank"], b["rank"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "A_relative_C12_profiles": a_plans[policy],
        "A_C3_fixed_weight12_count": a["C3_fixed_weight12_count"],
        "A_expanded_fixed_witness_weights": [
            word.bit_count() for word in a["fixed_challenge_words"]],
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly constrained selected cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Row-six/row-nine density reallocation and deeper kernel-pair screening yield five selected readings >=16; the measured current program returned 14,14,16,14,16, all with sector-1 minimum witnesses.
# REFUTED: Reducing sampled six-witness multiplicity on C7:C12(s=2) (1,4)x(1,4) raises the measured score above 19.059; all five measured codes instead retained distance six, k=756, and abundant sector-one minima.
# REFUTED: Mostly cubic columns and individual component-witness repair yield five selected readings >=11 on this bed; the incumbent instead measured 10,12,12,12,12, with every reported minimum witness in sector one.
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
