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
CONCEPT = ("On split-metacyclic C7:C21(s=2), order 147, A=B=(1,3), primitive-complement trinomials and independently challenged component kernels should produce five new readings >=14 and improve cell 0.400 beyond 59.451. Failure refutes this measurement hypothesis; finite searches certify no quantum distance.")
GROUP = ("Keep the split-metacyclic construction rule, resizing only its parameters to m=7,r=21,s=2. The action satisfies 2^21=1 modulo 7. Order 84 would give n=840 after the shape change; q=147 gives n=1470 and is the largest multiple of twenty-one below the factor-ten size ceiling. This group is isomorphic to Frobenius21 x C7. The related order-84 cubic bed reached distance twelve, whereas the current binomial-heavy bed repeatedly measured six; neither reading establishes distance transfer.")
SHAPES = ("Change both sides to (1,3). Equal components have 441 variables and 147 checks, nominal rate two-thirds; n=1470 and k>=588 guarantee rate 0.400, targeting that frontier rather than the requested 0.550 plateau. Parent minima were entirely sector one, prominently Acol1/2 across Bcol0..2. Every new component column has degree three, removing those binomial cycle supports. Actual full ranks enforce the 0.400 cell.")
FILLING = ("All six cells are trinomials: component row totals nine and both CSS row maxima twelve. Twenty cards cross five primitive C21 quotient prescriptions with four joint three-column plans. Balanced C3 charges propose two common quotient annihilators, verified on authoritative columns. Exact component kernel-four rejection and odd column degree exclude weights through five. Localized kernels, C3-fixed and two C7-fixed restrictions, retained cancellation witnesses, actual normal-label repairs and independent deep challenges screen every finalist. Actual sector-one logical strips modulo original stabilizer spans and full k select pairings with readings capped at fourteen. Full assembly agreement, kernel-four, weight, rate and CSS checks remain binding; neither restricted searches nor sampled readings certify quantum distance or complete Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r040_m7r21s2_13x13_primitive_cubic_restricted_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 147

def custom_ops(q):
    # Same split-metacyclic construction rule; new factor-ten size law.
    # The action has order three and 2^21=1 modulo seven.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 147")
    m, r, s = 7, 21, 2
    assert pow(s, r, m) == 1
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


def _retired_binomial_quotient_ctx(rng, G, idx):
    """Inactive order-84 experiment; never called by the new generator."""
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
    """Equal cubic components; only authoritative blocks supply matrices."""
    q = G.q
    assert q == 147 and A_SHAPE == B_SHAPE == (1, 3)
    card, cohort = idx % 20, idx // 20
    model, mode = divmod(card, 4)
    seed = rng.randrange(1 << 62) ^ (15485863*(idx+1) + 0x147313)
    local = random.Random(seed)
    mask = (1 << q)-1
    width = 3*q
    support = [[3, 3, 3]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class Probe:
        # Native operation cache, not an independently implemented lift.
        def __init__(self):
            self.q, self.e = q, ident
            self.elements = list(range(q))
        def idx(self, g):
            return g
        def mul(self, g, h):
            return table[g][h]
        def inv(self, g):
            return inverse[g]

    P = Probe()
    blocks = {"A": [], "B": []}
    for start in range(0, q, 14):
        ids = list(range(start, min(start+14, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            for side, rows in (("A", xx), ("B", zz)):
                block = [(row >> (j*q)) & mask for row in rows]
                assert all(w.bit_count() == 1 for w in block)
                blocks[side].append(block)

    def columns_of(rows, n):
        columns = [0]*n
        for r, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << r
        return columns

    def syndrome(columns, word):
        answer = 0
        for v in bits(word):
            answer ^= columns[v]
        return answer

    def dependencies(columns, order):
        pivots, basis = {}, []
        for v in order:
            value, word = columns[v], 1 << v
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

    def kernel4_free(columns):
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return False
        pairs = set()
        for i, value in enumerate(columns):
            for j in range(i):
                combined = value ^ columns[j]
                if combined in singles or combined in pairs:
                    return False
                pairs.add(combined)
        return True

    def geometry(columns):
        uses, four = {}, 0
        adjacency = [0]*q
        for column in columns:
            checks = list(bits(column))
            for r in checks:
                adjacency[r] |= column ^ (1 << r)
            for i, r in enumerate(checks):
                for t in checks[:i]:
                    key = (t, r)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        triangles = 0
        for r, neighbors in enumerate(adjacency):
            for t in bits(neighbors & (-1 << (r+1))):
                triangles += (neighbors & adjacency[t]
                              & (-1 << (t+1))).bit_count()
        # Projection excess is a heuristic, not an exact C6 count.
        return four, max(0, triangles-len(columns))

    def component(cells, side):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                block = blocks[side][g]
                for r, value in enumerate(block):
                    rows[r] ^= value << (j*q)
        return rows, columns_of(rows, width)

    # Every nonzero relative complement coordinate here is a unit mod 21.
    # Every profile meets all three C3 charges, proposing a common
    # two-dimensional quotient annihilator, verified below.
    profiles = ((1,2), (1,5), (2,4), (2,10), (4,8))
    plans = ((0,0,0), (0,1,2), (0,2,4), (0,1,4))
    libraries = []
    for u, v in profiles:
        assert {0, u % 3, v % 3} == {0, 1, 2}
        assert all(t % 3 and t % 7 for t in (u, v, (v-u) % 21))
        options = []
        for a in range(7):
            for b in range(7):
                g, h = G.idx((a,u)), G.idx((b,v))
                if table[g][h] == table[h][g]:
                    continue
                cell = (ident, g, h)
                left = {table[inverse[x]][y]
                        for x in cell for y in cell if x != y}
                right = {table[x][inverse[y]]
                         for x in cell for y in cell if x != y}
                if len(left) == len(right) == 6:
                    options.append(cell)
        assert options
        libraries.append(options)

    annihilators = (
        sum(1 << r for r, g in enumerate(G.elements)
            if g[1] % 3 in (0,1)),
        sum(1 << r for r, g in enumerate(G.elements)
            if g[1] % 3 in (1,2)),
    )
    assert gf2_rank(annihilators) == 2

    # Restrictions use valid regular COMPONENT actions. They are not
    # assumed to be symmetries of the complete nonabelian quantum code.
    generators = (
        (3, G.idx((0,7))),
        (7, G.idx((1,0))),
        (7, G.idx((0,3))),
    )
    partitions = {"A": [], "B": []}
    for order, generator in generators:
        subgroup, value = [], ident
        for _ in range(order):
            subgroup.append(value)
            value = table[value][generator]
        assert value == ident and len(set(subgroup)) == order
        for side in ("A", "B"):
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {table[root][h] if side == "A"
                         else table[h][root] for h in subgroup}
                assert len(orbit) == order and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(tuple(sorted(orbit)))
            groups = [sum(1 << (j*q+c) for c in orbit)
                      for j in range(3) for orbit in orbits]
            partitions[side].append((order, groups))

    def expand(word, groups):
        answer = 0
        for v in bits(word):
            answer ^= groups[v]
        return answer

    def search(columns, trials, diagnostic, initial=()):
        """Accumulated genuine kernel words; never an absence certificate."""
        beam = sorted({w for w in initial if w},
                      key=lambda w: (w.bit_count(), w))[:24]
        for word in beam:
            assert syndrome(columns, word) == 0
        best = min(beam, key=int.bit_count) if beam else None
        order = list(range(len(columns)))

        def note(word, grown):
            nonlocal best
            if not word:
                return
            if best is None or word.bit_count() < best.bit_count():
                best = word
            if word.bit_count() <= best.bit_count()+6:
                grown.append(word)

        for turn in range(trials):
            diagnostic.shuffle(order)
            if beam and turn % 4 == 3:
                active = list(bits(diagnostic.choice(beam[:6])))
                preferred = set(diagnostic.sample(
                    active, max(1, len(active)//2)))
                order.sort(key=lambda v: v not in preferred)
            basis = dependencies(columns, order)
            grown = list(beam)
            for word in basis:
                note(word, grown)
            light = sorted(set(basis+beam),
                           key=lambda w: (w.bit_count(), w))[:16]
            moves = basis+beam
            for first in light:
                for second in moves:
                    note(first ^ second, grown)
                if len(grown) > 192:
                    grown = sorted(set(grown),
                                   key=int.bit_count)[:32]
            shortened = sorted(set(grown),
                               key=int.bit_count)[:24]
            for i, first in enumerate(shortened):
                for second in shortened[:i]:
                    note(first ^ second, grown)
            diagnostic.shuffle(moves)
            for word in light[:4]:
                for _ in range(2):
                    before = word.bit_count()
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    note(word, grown)
                    if word.bit_count() == before:
                        break
            beam = sorted({w for w in grown if w},
                          key=lambda w: (w.bit_count(), w))[:24]
            if best is not None:
                beam = sorted(set(beam+[best]),
                              key=lambda w: (w.bit_count(), w))[:24]
            if best is not None and best.bit_count() <= 6:
                break
        if best is None:
            return []
        assert syndrome(columns, best) == 0
        return sorted(set(beam+[best]),
                      key=lambda w: (w.bit_count(), w))[:24]

    def localized(columns):
        """Exact localized enumeration when nullity <=12; witnesses otherwise."""
        words = []
        for j in range(3):
            part = columns[j*q:(j+1)*q]
            basis = dependencies(part, range(q))
            if not basis:
                continue
            best = min(basis, key=int.bit_count)
            if len(basis) <= 12:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length()-1]
                    if word and word.bit_count() < best.bit_count():
                        best = word
            else:
                for i, first in enumerate(basis):
                    for second in basis[:i]:
                        word = first ^ second
                        if word and word.bit_count() < best.bit_count():
                            best = word
            assert syndrome(part, best) == 0
            words.append(best << (j*q))
        return words

    def inspect(cells, side, plan):
        rows, columns = component(cells, side)
        defect = q-gf2_rank(rows)
        if not 2 <= defect <= 6:
            return None
        assert all((functional & column).bit_count() % 2 == 0
                   for functional in annihilators for column in columns)
        if not kernel4_free(columns):
            return None
        assert all(column.bit_count() == 3 for column in columns)
        # Odd column degree excludes every odd-weight kernel word.
        words = localized(columns)
        fixed = []
        for order, groups in partitions[side]:
            fixed.append((order, groups,
                          [syndrome(columns, g) for g in groups], []))
        return dict(cells=[list(c) for c in cells], rows=rows,
                    columns=columns, defect=defect,
                    cycles=geometry(columns), plan=plan,
                    words=words, fixed=fixed, reading=width+1)

    def probe(record, side, trials, turn):
        offset = 100003*int(side == "B")
        retained = list(record["words"])
        refreshed = []
        for at, (order, groups, compressed, old) in enumerate(record["fixed"]):
            diagnostic = random.Random(seed+offset+700001*turn+1013*at)
            beam = search(compressed, max(4, trials//4), diagnostic, old)
            for source in beam[:6]:
                actual = expand(source, groups)
                assert actual.bit_count() == order*source.bit_count()
                assert syndrome(record["columns"], actual) == 0
                retained.append(actual)
            refreshed.append((order, groups, compressed, beam))
        record["fixed"] = refreshed
        beam = search(record["columns"], trials,
                      random.Random(seed+offset+1700003*turn), retained)
        record["words"] = sorted(set(retained+beam),
                                key=lambda w: (w.bit_count(), w))[:24]
        assert record["words"]
        assert all(syndrome(record["columns"], w) == 0
                   for w in record["words"])
        record["reading"] = min(
            record["reading"], record["words"][0].bit_count())

    def merit(record):
        four, excess = record["cycles"]
        secondary = ((-four, -excess, record["defect"]),
                     (-excess, -four, record["defect"]),
                     (record["defect"], -four, -excess),
                     (-four-excess, record["defect"], -four))[mode]
        return (min(14, record["reading"]),) + secondary + (
            min(20, record["reading"]),)

    def pool(side):
        shift = model if side == "A" else (2*model+mode+1+cohort) % 5
        recipe = plans[mode if side == "A" else (mode+model+1) % 4]
        plan = tuple((shift+p+cohort) % 5 for p in recipe)
        records, seen = [], set()

        def remember(cells):
            # Remove column permutations inside the proposal bank only.
            signature = tuple(sorted(tuple(sorted(c)) for c in cells))
            if signature in seen:
                return None
            seen.add(signature)
            return inspect(cells, side, plan)

        for attempt in range(64):
            cells = [list(local.choice(libraries[p])) for p in plan]
            record = remember(cells)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["cycles"], -r["defect"]))
        selected = records[:8]
        if not selected:
            raise ValueError("primitive cubic card failed component gates")
        for at, record in enumerate(selected):
            probe(record, side, 8, 1+at)
        selected.sort(key=merit, reverse=True)

        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(12):
                cells = [list(c) for c in current["cells"]]
                witness = local.choice(current["words"][:6])
                active = sorted({v//q for v in bits(witness)})
                j = local.choice(active or list(range(3)))
                cells[j] = list(local.choice(libraries[plan[j]]))
                proposed = remember(cells)
                if proposed is None:
                    continue
                proposed["words"].extend(
                    word for word in current["words"]
                    if syndrome(proposed["columns"], word) == 0)
                probe(proposed, side, 6, 101+20*chain+step)
                selected.append(proposed)
                if merit(proposed) >= merit(current):
                    current = proposed

        finalists = sorted(selected, key=merit, reverse=True)[:3]
        # Deep-test ALL finalists. Earlier witnesses are never discarded.
        for at, record in enumerate(finalists):
            probe(record, side, 32, 301+at)
            probe(record, side, 64+8*(cohort % 3), 401+at)
        finalists.sort(key=merit, reverse=True)
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

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(a, b, hx, hz, n):
        """Actual strip logicals in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in record["words"][:6]:
                for fixed in range(3):
                    actual = (sum(
                        ((word >> (j*q)) & mask) << ((3*j+fixed)*q)
                        for j in range(3)) if kind == "Z"
                        else word << (3*fixed*q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers, key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    winner = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool("A"), pool("B")
    proposals = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            assert n == 1470 and k >= 588
            if not (375*n <= 1000*k < 425*n):
                continue
            actual_a = [
                sum(((hx[r] >> (3*j*q)) & mask) << (j*q)
                    for j in range(3)) for r in range(q)]
            actual_b = [row & ((1 << width)-1) for row in hz[:q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logicals = logical_readings(a, b, hx, hz, n)
            # Missing quantum witnesses never earn an optimistic score.
            witnessed = min(logicals)
            detected = min(14, a["reading"], b["reading"],
                           witnessed if witnessed <= n else 12)
            key = (k*detected*detected, detected,
                   -a["cycles"][0]-b["cycles"][0],
                   -a["cycles"][1]-b["cycles"][1])
            proposals.append((key, A, B, hx, hz, k, a, b, logicals))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for key, A, B, hx, hz, k, a, b, logicals in proposals:
        # No full-code regular symmetry is assumed in these gates.
        if not kernel4_free(columns_of(hx, 1470)):
            continue
        if not kernel4_free(columns_of(hz, 1470)):
            continue
        assert max(row.bit_count() for row in hx+hz) == 12
        assert css_ok(hx, hz)
        return dict(
            idx=idx, cells={"A": A, "B": B}, k=k,
            card=(model, mode, cohort),
            quotient_plans=(a["plan"], b["plan"]),
            component_defects=(a["defect"], b["defect"]),
            component_readings=(a["reading"], b["reading"]),
            component_geometry=(a["cycles"], b["cycles"]),
            verified_strip_logical_readings=logicals,
            pairing_merit=key, target_cell=0.400)
    raise ValueError("primitive cubic pairs missed full validity/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cubic cell without additional draws."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == 3 and len({G._key(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Row-eight balanced voltage/minor/information-set screening would yield five readings above 14: the measured parent instead returned 15,15,14,14,14, with lighter X witnesses confined to sector 1.
# REFUTED: Row-six/row-nine density reallocation and deeper kernel-pair screening yield five selected readings >=16; the measured current program returned 14,14,16,14,16, all with sector-1 minimum witnesses.
# REFUTED: Component kernel-five exclusion and six-witness multiplicity selection lift the order-84 (1,4)x(1,4) bed above its distance-six plateau; all five measured current codes instead remained k=756 and distance six, with abundant sector-one minima.
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
