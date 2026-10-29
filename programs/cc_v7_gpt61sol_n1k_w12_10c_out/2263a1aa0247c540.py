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
CONCEPT = ("Hypothesis: split-metacyclic C61:C3(s=13), order 183, A=(1,2), B=(2,3), with whole-cell Sidon supports, mixed B ratio orders and witness-directed repair improves five-code fitness above 213.032 in cell 0.150. Search readings remain provisional upper bounds.")
GROUP = ("Keep GTYPE custom and the parent's split-metacyclic multiplication/inverse construction formula, resizing only m,r,s to 61,3,13 as permitted by the L2 card. Here 13^3=1 mod 61 and the action has exact order three. This destination bed measured fitness 213.0, versus 24.0532 on the parent's order-30 bed. Order 183 is the largest prime-normal order-three lift below factor-eight q_max=187; no distance-transfer theorem is asserted.")
SHAPES = ("Change A to (1,2) and B to (2,3): n=8*183=1464, k>=183 and guaranteed rate 1/8 target cell 0.150. Components have 366 variables/183 checks and 549 variables/366 checks. Parent minimum witnesses were entirely sector-1 Acol/Bcol operators, commonly involving Acol3 across several Bcols. Dense A cells and a larger lift replace those cubic-column kernels. Exact full ranks require 0.125<=k/n<0.175 for every returned filling.")
FILLING = ("Twenty cards cross four A fiber-occupancy prescriptions with five cubic B topologies. A counts (4,5) give row total nine; B column totals three and row totals at most seven give HX=12 and HZ<=12. Whole-cell directed differences must be distinct in BOTH orientations, and larger cells must have noncommuting relative generators. B binomials deliberately mix order-61 rotation ratios with order-three quotient ratios, unlike the shown destination model. Verified scaffold monomial blocks supply all component syndromes. Exact kernel-four screening, retained-witness cancellation, implicated-cell exponent repair and independent finalist challenges select candidates. Original stabilizer spans verify and reduce localized quantum witnesses before pairing. Exact ranks, native-lift agreement, weights and CSS gate submissions; no heuristic certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m61r3s13_12x23_whole_sidon_ratio61_logicalrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 183

def custom_ops(q):
    # Same split-metacyclic construction formula; only parameters change.
    # C61 x| C3: y x y^-1 = x^13; 13^3 = 1 mod 61.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 183")
    m, r, s = 61, 3, 13
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
B_SHAPE = (2, 3)               # (rb, cb)


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _cubic_support(rng, cols, concentrated, target):
    """Select a balanced prescription with exactly three terms per column."""
    import itertools
    order = list(range(cols))
    rng.shuffle(order)
    sparse = set(order[:concentrated])
    options = [(0, 3) if j in sparse else (1, 2)
               for j in range(cols)]
    choices = [v for v in itertools.product(*options) if sum(v) == target]
    first = list(rng.choice(choices))
    grid = [first, [3 - v for v in first]]
    assert all(sum(row) > 0 for row in grid)
    return grid


def _coset_pattern(rng, supports):
    """Two split columns carry different coset holonomies."""
    cols = len(supports[0])
    bits = [[rng.randrange(2) for _ in range(cols)] for _ in range(2)]
    split = [j for j in range(cols)
             if supports[0][j] and supports[1][j]]
    assert len(split) >= 2
    j, k = split[:2]
    bits[0][j] = bits[1][j] = 0
    bits[0][k], bits[1][k] = 0, 1
    return bits


def _legal_exponents(exponents, m):
    import math
    if len(set(exponents)) != len(exponents):
        return False
    if len(exponents) < 2:
        return True
    gaps = []
    divisor = m
    for a in range(len(exponents)):
        for b in range(a + 1, len(exponents)):
            d = (exponents[a] - exponents[b]) % m
            if 2 * d == m:
                return False
            gaps.append(min(d, m - d))
            divisor = math.gcd(divisor, d)
    # Distinct signed difference classes prevent internal four-cycles.
    return divisor == 1 and len(set(gaps)) == len(gaps)


def _new_coset_cell(rng, m, count, bit):
    if count == 0:
        return []
    while True:
        exponents = rng.sample(range(m), count)
        if _legal_exponents(exponents, m):
            return [(a, bit) for a in exponents]


def _initial_cubic_cells(rng, G, supports, bits):
    m = G.q // 2
    return [[_new_coset_cell(rng, m, count, bits[i][j])
             for j, count in enumerate(row)]
            for i, row in enumerate(supports)]


def _component_rows(hx, hz, q, side):
    """Extract the actual left/right components from verified matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "B":
        mask = (1 << (cb * q)) - 1
        return [hz[a * q + r] & mask
                for a in range(rb) for r in range(q)]
    mask = (1 << q) - 1
    rows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            rows.append(row)
    return rows


def _cycle_cost(rows, penalty):
    """Count component four/six cycles, excluding reused variable nodes."""
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    c4 = 0
    for a in range(h):
        for b in range(a + 1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                c4 += overlap * (overlap - 1) // 2
    c6 = 0
    for a in range(h):
        ns = upper[a]
        for t, b in enumerate(ns):
            ab = common[a][b]
            for c in ns[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    c6 += ab * ac * bc - triple * (ab + ac + bc - 2)
    return penalty * c4 + c6


def _verified_costs(G, A, B, penalty, side=None):
    hx, hz, _ = lifted_matrices(G, A, B)
    sides = (side,) if side is not None else ("A", "B")
    costs = [_cycle_cost(_component_rows(hx, hz, G.q, s), penalty)
             for s in sides]
    return costs[0] if side is not None else costs


def _small_kernel_penalty(rows, width):
    """Exact <=4 veto plus a connected-triple weight-six witness screen."""
    import itertools
    columns = [0] * width
    adjacency = [set() for _ in range(width)]
    for r, row in enumerate(rows):
        remaining = row
        incident = []
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            columns[j] |= 1 << r
            incident.append(j)
            remaining ^= bit
        for j in incident:
            adjacency[j].update(k for k in incident if k != j)

    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return 1000000

    pairs = set()
    short = 0
    for a in range(width):
        for b in range(a + 1, width):
            syndrome = columns[a] ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                short += 1
            pairs.add(syndrome)
    if short:
        return 100000 + short

    # Enumerate triples connected through shared checks. A collision
    # verifies a six-column kernel word; this is not an exhaustive
    # general weight-six search.
    triples = set()
    for b in range(width):
        for a, c in itertools.combinations(sorted(adjacency[b]), 2):
            triples.add(tuple(sorted((a, b, c))))
    seen = {}
    six = 0
    for a, b, c in sorted(triples):
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        mask = (1 << a) | (1 << b) | (1 << c)
        old = seen.get(syndrome)
        if old is not None:
            if (old ^ mask).bit_count() == 6:
                six += 1
        else:
            seen[syndrome] = mask
    return six


def _optimize_labels(rng, G, A, B, penalty):
    import math
    costs = _verified_costs(G, A, B, penalty)
    current = sum(costs)
    best = current
    best_A, best_B = _copy_cells(A), _copy_cells(B)
    positions = [[(i, j) for i, row in enumerate(grid)
                  for j, cell in enumerate(row) if cell]
                 for grid in (A, B)]
    m = G.q // 2
    steps = 240
    for step in range(steps):
        which = rng.randrange(2)
        grid = A if which == 0 else B
        i, j = rng.choice(positions[which])
        old = list(grid[i][j])
        bit = old[0][1]
        if step % 9 == 0:
            new = _new_coset_cell(rng, m, len(old), bit)
        else:
            new = None
            for _ in range(40):
                exponents = [g[0] for g in old]
                exponents[rng.randrange(len(old))] = rng.randrange(m)
                if _legal_exponents(exponents, m):
                    new = [(a, bit) for a in exponents]
                    break
            if new is None:
                continue
        grid[i][j] = new
        side = ("A", "B")[which]
        proposed_cost = _verified_costs(G, A, B, penalty, side)
        proposed = current - costs[which] + proposed_cost
        temperature = G.q * (18.0 * (1.0 - step / steps) ** 2 + 0.25)
        accept = proposed <= current
        if not accept:
            accept = rng.random() < math.exp((current - proposed) /
                                             temperature)
        if accept:
            current = proposed
            costs[which] = proposed_cost
            if current < best:
                best = current
                best_A, best_B = _copy_cells(A), _copy_cells(B)
        else:
            grid[i][j] = old
    return best_A, best_B, best


def _shape48_ctx(rng, G, idx):
    """Twenty new topology cards with independently challenged witnesses."""
    import itertools
    import math

    q = G.q
    assert q == 30
    m = q // 2
    card, cohort = idx % 20, idx // 20
    ap, bp = divmod(card, 5)
    diagnostic_seed = rng.randrange(1 << 62)

    # Balanced substitutions for columns (1,1,1).
    # Each prescription contributes equally to all three A row totals.
    pair = [(2, 1, 0), (0, 1, 2)]
    triple = [(2, 1, 0), (0, 2, 1), (1, 0, 2)]
    prescriptions = ([], pair, triple, pair + pair)
    columns = list(prescriptions[ap])
    columns += [(1, 1, 1)] * (7 - len(columns))
    rng.shuffle(columns)
    row_order = list(range(3))
    rng.shuffle(row_order)
    sa = [[column[i] for column in columns] for i in row_order]
    sb = _cubic_support(rng, 6, bp, 9)

    assert all(sum(row) == 7 for row in sa)
    assert all(sum(row) == 9 for row in sb)
    assert all(sum(row[j] for row in sa) == 3 for j in range(7))
    assert all(sum(row[j] for row in sb) == 3 for j in range(6))
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    # Cache blocks obtained from the VERIFIED construction. Component
    # assembly below uses only GF2 linearity of these exact blocks.
    mask = (1 << q) - 1
    all_cells = [[[g] for g in G.elements]]
    blocks = {}
    source, _, _ = lifted_matrices(G, all_cells, [[[]]])
    blocks["A"] = [
        [(row >> (g * q)) & mask for row in source]
        for g in range(q)
    ]
    _, source, _ = lifted_matrices(G, [[[]]], all_cells)
    blocks["B"] = [
        [(row >> (g * q)) & mask for row in source]
        for g in range(q)
    ]

    def draw(count, bit=None):
        if bit is None:
            bit = rng.randrange(2)
        return _new_coset_cell(rng, m, count, bit)

    def fresh(support):
        nr, nc = len(support), len(support[0])
        quotient = [[rng.randrange(2) for _ in range(nc)]
                    for _ in range(nr)]
        split = [j for j in range(nc)
                 if support[0][j] and support[1][j]]
        if len(split) >= 2:
            j, k = split[:2]
            quotient[0][j] = quotient[1][j] = 0
            quotient[0][k], quotient[1][k] = 0, 1
        return [[draw(count, quotient[i][j])
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def inspect(grid, side):
        rows = []
        width = len(grid[0]) * q
        monomials = blocks[side]
        for cells in grid:
            indexed = [[G.idx(g) for g in cell] for cell in cells]
            for r in range(q):
                row = 0
                for j, ids in enumerate(indexed):
                    block = 0
                    for g in ids:
                        block ^= monomials[g][r]
                    row |= block << (j * q)
                rows.append(row)
        columns = [0] * width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        assert all(c.bit_count() == 3 for c in columns)
        bad = sum(c == 0 for c in columns)
        bad += width - len(set(columns))
        penalty = 400 + 80 * ((card + cohort) % 4)
        cost = 1000000 * bad + _cycle_cost(rows, penalty)
        return {"grid": grid, "rows": rows, "columns": columns,
                "cost": cost, "reading": width + 1, "word": None}

    def retain(rec, word):
        if word and word.bit_count() < rec["reading"]:
            rec["reading"] = word.bit_count()
            rec["word"] = word

    def circuit_probe(rec, trials, seed):
        """Actual component null words; every reading is an upper bound."""
        local = random.Random(seed)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = []
        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if value:
                    continue
                for sweep in range(2):
                    changed = False
                    for other in retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    if not changed:
                        break
                retain(rec, word)
                words.append(word)
            light = sorted(words, key=int.bit_count)[:20]
            for word in light:
                for other in words + retained:
                    retain(rec, word ^ other)
            for initial in light[:6]:
                word = initial
                for sweep in range(2):
                    changed = False
                    for other in words + retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    retain(rec, word)
                    if not changed:
                        break
            retained = sorted(set(retained + light),
                              key=int.bit_count)[:28]
            if rec["reading"] <= 4:
                break

    def parity_probe(rec, budget, cap=12):
        """Exact <=4 collisions and budgeted anchored parity completion.

        Regular component symmetry supplies one anchor per column block.
        Budget exhaustion is NOT an exclusion certificate.
        """
        columns = rec["columns"]
        width = len(columns)
        singles, pairs = {}, {}
        incidence = [0] * len(rec["rows"])
        for c, value in enumerate(columns):
            word = 1 << c
            if not value:
                retain(rec, word)
            previous = singles.get(value)
            if previous is not None:
                retain(rec, previous ^ word)
            else:
                singles[value] = word
            for r in bits(value):
                incidence[r] |= word
        for a, value in enumerate(columns):
            for b in range(a):
                key = value ^ columns[b]
                word = (1 << a) | (1 << b)
                previous = singles.get(key)
                if previous is not None:
                    retain(rec, previous ^ word)
                previous = pairs.get(key)
                if previous is not None:
                    retain(rec, previous ^ word)
                else:
                    pairs[key] = word
        if rec["reading"] <= 4:
            return

        priorities = [rng.random() for _ in columns]
        roots = list(range(0, width, q))
        rng.shuffle(roots)
        all_variables = (1 << width) - 1

        for root in roots:
            nodes = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if nodes[0] <= 0:
                    return
                nodes[0] -= 1
                if not value:
                    retain(rec, word)
                    return
                left = min(cap, rec["reading"] - 1) - word.bit_count()
                if left <= 0 or value.bit_count() > 3 * left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    retain(rec, word | one)
                pair_word = pairs.get(value, 0)
                if left >= 2 and pair_word and pair_word & allowed == pair_word:
                    retain(rec, word | pair_word)
                left = min(cap, rec["reading"] - 1) - word.bit_count()
                if left <= 2:
                    return

                choices, smallest = None, width + 1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        choices, smallest = available, count
                order = list(bits(choices))
                order.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in order:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if nodes[0] <= 0:
                        break

            allowed = all_variables & ~((1 << (root + 1)) - 1)
            visit(columns[root], 1 << root, allowed)

    def merit(rec):
        # Capping the heuristic prevents a very optimistic component
        # reading from overwhelming the independent challenge stages.
        return (min(18, rec["reading"]), -rec["cost"])

    def signature(rec):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in rec["grid"] for cell in row)

    def component_pool(support, side):
        seed = diagnostic_seed + 100003 * int(side == "B")
        bank = {}

        def remember(rec):
            key = signature(rec)
            bank[key] = rec
            if len(bank) > 5:
                worst = max(bank, key=lambda k: bank[k]["cost"])
                del bank[worst]

        starts = [inspect(fresh(support), side) for _ in range(24)]
        starts.sort(key=lambda rec: rec["cost"])
        positions = [(i, j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]

        for current in starts[:2]:
            remember(current)
            for step in range(64):
                grid = _copy_cells(current["grid"])
                i, j = rng.choice(positions)
                grid[i][j] = draw(support[i][j])
                proposed = inspect(grid, side)
                delta = proposed["cost"] - current["cost"]
                temperature = q * (8 * (1 - step / 64.0) ** 2 + 0.15)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta / temperature)):
                    current = proposed
                    remember(current)

        survivors = list(bank.values())
        for rec in survivors:
            circuit_probe(rec, 16, seed)
            parity_probe(rec, 6000)
        survivors.sort(key=merit, reverse=True)
        current = survivors[0]
        keepers = survivors[:3]

        # Redraw the actual column blocks supporting a retained null word.
        # This changes relative voltages, not just a gauge labeling.
        for step in range(16):
            proposals = []
            for _ in range(2):
                grid = _copy_cells(current["grid"])
                if current["word"] is not None:
                    used = sorted({c // q for c in bits(current["word"])})
                    j = rng.choice(used)
                else:
                    j = rng.randrange(len(support[0]))
                for i in range(len(support)):
                    grid[i][j] = draw(support[i][j])
                proposed = inspect(grid, side)
                if proposed["cost"] > current["cost"] + 1200 * q:
                    continue
                circuit_probe(proposed, 8, seed + 700001)
                parity_probe(proposed, 3000)
                proposals.append(proposed)
            if not proposals:
                continue
            proposed = max(proposals, key=merit)
            if merit(proposed) >= merit(current) or (
                    merit(proposed)[0] == merit(current)[0]
                    and rng.random() < 0.10 * (1 - step / 16.0)):
                current = proposed
            keepers.extend(proposals)
            keepers.sort(key=merit, reverse=True)
            del keepers[6:]

        finalists, seen = [], set()
        for rec in sorted(keepers + [current], key=merit, reverse=True):
            key = signature(rec)
            if key in seen:
                continue
            seen.add(key)
            # Every returned finalist receives the independent deeper
            # challenge, so an unchallenged optimistic record cannot win.
            circuit_probe(rec, 96, seed + 1700003)
            parity_probe(rec, 18000)
            rec["rank"] = gf2_rank(rec["rows"])
            if rec["word"] is not None:
                syndrome = 0
                for c in bits(rec["word"]):
                    syndrome ^= rec["columns"][c]
                assert syndrome == 0
            finalists.append(rec)
            if len(finalists) == 3:
                break
        finalists.sort(key=lambda rec: (
            merit(rec), rec["rank"]), reverse=True)
        return finalists[:2]

    aa = component_pool(sa, "A")
    bb = component_pool(sb, "B")
    proposals = sorted(
        ((a, b) for a in aa for b in bb),
        key=lambda ab: (
            min(ab[0]["reading"], ab[1]["reading"]),
            ab[0]["reading"] + ab[1]["reading"],
            -ab[0]["cost"] - ab[1]["cost"]),
        reverse=True)

    best = None
    for a, b in proposals:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1440 and k >= 480
                and 325 * n <= 1000 * k < 375 * n):
            continue
        # Confirm cached verified-block assembly against the exact
        # components of the final full quantum construction.
        assert _component_rows(hx, hz, q, "A") == a["rows"]
        assert _component_rows(hx, hz, q, "B") == b["rows"]
        reading = min(18, a["reading"], b["reading"])
        key = (k * reading * reading, reading,
               -a["cost"] - b["cost"])
        if best is None or key > best[0]:
            best = (key, A, B, hx, hz, k,
                    (a["reading"], b["reading"]))
    if best is None:
        raise ValueError("new three-row cubic pair missed cell 0.350")

    key, A, B, hx, hz, k, readings = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert all(len({G.idx(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "card": (ap, bp, cohort),
            "cells": {"A": A, "B": B}, "screen_k": k,
            "component_readings": readings, "quality": key}


def _mp_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _mp_syndrome(columns, word):
    result = 0
    for v in _mp_bits(word):
        result ^= columns[v]
    return result


def _mp_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _mp_nontrivial(word, pivots):
    """Exact membership test in the original stabilizer span."""
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def NEW_CTX(rng, G, idx):
    """New low-rate cards, repaired and independently challenged."""
    q = G.q
    assert q == 183
    card, cohort = idx % 20, idx // 20
    mode, topology = card % 4, card // 4
    seed = rng.randrange(1 << 62)
    sa = [[4, 5]]
    bplans = (
        ((1, 1, 2), (2, 2, 1)),
        ((1, 1, 1), (2, 2, 2)),
        ((0, 1, 2), (3, 2, 1)),
        ((0, 1, 1), (3, 2, 2)),
        ((0, 2, 2), (3, 1, 1)),
    )
    sb = [list(row) for row in bplans[topology]]
    amodels = (
        ((2, 1, 1), (2, 2, 1)),
        ((2, 2, 0), (2, 2, 1)),
        ((2, 1, 1), (3, 2, 0)),
        ((3, 1, 0), (3, 1, 1)),
    )
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache native operations, never an independently implemented lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

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

    P = Probe()
    mask = (1 << q) - 1
    monomials = [[[g] for g in range(q)]]
    source, _, _ = lifted_matrices(P, monomials, [[[]]])
    ablocks = [[(row >> (g*q)) & mask for row in source]
               for g in range(q)]
    _, source, _ = lifted_matrices(P, [[[]]], monomials)
    bblocks = [[(row >> (g*q)) & mask for row in source]
               for g in range(q)]

    def whole_sidon(cell):
        if len(cell) != len(set(cell)):
            return False
        if len(cell) < 2:
            return True
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(cell)*(len(cell)-1)
        if len(left) != wanted or len(right) != wanted:
            return False
        if len(cell) >= 3:
            ratios = [table[inverses[cell[0]]][g] for g in cell[1:]]
            if not any(table[a][b] != table[b][a]
                       for t, a in enumerate(ratios) for b in ratios[:t]):
                return False
        return True

    def draw(occupancy):
        count = sum(occupancy)
        if not count:
            return []
        for _ in range(1024):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, size in zip(fibers, occupancy):
                aa = rng.sample(range(61), size)
                cell.extend(G.idx((a, b)) for a in aa)
            if whole_sidon(cell):
                return cell
        raise ValueError("whole-cell Sidon occupancy failed")

    def cell_for(side, i, j):
        if side == "A":
            return draw(amodels[mode][j])
        count = sb[i][j]
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            # Homogeneous pairs have exact ratio order 61. Mixed pairs
            # have exact ratio order three. These are not gauge images.
            homogeneous = (mode + i + 2*j + cohort) % 4 == 0
            return draw((2, 0, 0) if homogeneous else (1, 1, 0))
        return draw((1, 1, 1))

    def fresh(side):
        support = sa if side == "A" else sb
        return [[cell_for(side, i, j) for j in range(len(row))]
                for i, row in enumerate(support)]

    def inspect(grid, side):
        """GF2 sums of authoritative scaffold monomial blocks."""
        blocks = ablocks if side == "A" else bblocks
        width = len(grid[0])*q
        rows = []
        for cells in grid:
            for r in range(q):
                row = 0
                for j, cell in enumerate(cells):
                    block = 0
                    for g in cell:
                        block ^= blocks[g][r]
                    row |= block << (j*q)
                rows.append(row)
        columns = [0]*width
        for r, row in enumerate(rows):
            for v in _mp_bits(row):
                columns[v] |= 1 << r
        if not all(columns) or len(set(columns)) != width:
            return None

        adjacency = [0]*len(rows)
        pairs = {}
        four = forced = 0
        for column in columns:
            ns = list(_mp_bits(column))
            d = len(ns)
            forced += d*(d-1)*(d-2)//6
            for t, a in enumerate(ns):
                adjacency[a] |= column ^ (1 << a)
                for b in ns[:t]:
                    key = (b, a)
                    old = pairs.get(key, 0)
                    four += old
                    pairs[key] = old + 1
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in _mp_bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << len(rows))-1:
            return None

        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in _mp_bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        excess = max(0, triangles-forced)
        cost = (400, 700, 1100, 1600)[mode]*four + excess
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(grid=_copy_cells(grid), rows=rows, columns=columns,
                    cost=cost, cycles=(four, excess), signature=signature,
                    d=width+1, words=[])

    def note(rec, word):
        if not word:
            return
        weight = word.bit_count()
        rec["d"] = min(rec["d"], weight)
        if weight <= rec["d"] + 6:
            rec["words"].append(word)
            if len(rec["words"]) > 80:
                rec["words"] = sorted(set(rec["words"]),
                                      key=int.bit_count)[:20]

    def screen(rec, side):
        """Exact ranks and exclusion of all component words through four."""
        rank = gf2_rank(rec["rows"])
        if rank < len(rec["rows"]) - (3 if side == "A" else 6):
            return False
        if side == "A":
            if min(gf2_rank([row & mask for row in rec["rows"]]),
                   gf2_rank([row >> q for row in rec["rows"]])) < q-5:
                return False
        singles = set(rec["columns"])
        seen = set()
        for i, col in enumerate(rec["columns"]):
            for other in rec["columns"][:i]:
                value = col ^ other
                if value in singles or value in seen:
                    return False
                seen.add(value)
        rec["rank"] = rank
        return True

    def query(rec, side, trials, offset):
        """Genuine component witnesses, retained across independent probes."""
        diagnostic = random.Random(seed + offset)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = list(rec["words"])
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
                    note(rec, word)
            if not basis:
                raise ValueError("component unexpectedly has no kernel")
            light = sorted(set(retained+basis), key=int.bit_count)[:16]
            for t, word in enumerate(basis):
                # A's half-rate kernel receives complete circuit-pair
                # cancellation, rather than short-vs-short cancellation.
                others = basis[:t] if side == "A" else light
                for other in others:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= rec["d"]+2:
                        note(rec, candidate)
            moves = retained+basis
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(rec, word)
            rec["words"] = sorted(set(rec["words"]),
                                  key=int.bit_count)[:20]
            retained = rec["words"]
            if rec["d"] <= 8:
                break
        for word in rec["words"][:3]:
            assert _mp_syndrome(columns, word) == 0

    def merit(rec):
        multiplicity = sum(w.bit_count() == rec["d"] for w in rec["words"])
        return (rec["d"], -multiplicity, -rec["cost"], rec["rank"])

    def pool(side):
        offset = 100003 if side == "A" else 200003
        records, signatures = [], set()
        for _ in range(88 if side == "A" else 56):
            rec = inspect(fresh(side), side)
            if rec is None or rec["signature"] in signatures:
                continue
            signatures.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda r: r["cost"])
        survivors = []
        for rec in records:
            if not screen(rec, side):
                continue
            query(rec, side, 6, offset)
            survivors.append(rec)
            if len(survivors) == (8 if side == "A" else 5):
                break
        if not survivors:
            raise ValueError("no connected Sidon component passed exact gates")
        survivors.sort(key=merit, reverse=True)
        finalists = survivors[:3]
        for rec in finalists:
            query(rec, side, 20, offset+700001)
        finalists.sort(key=merit, reverse=True)
        current = finalists[0]
        keepers = finalists[:]

        for step in range(18 if side == "A" else 10):
            grid = _copy_cells(current["grid"])
            light = current["words"][:8]
            word = rng.choice(light)
            occupied = sorted({v//q for v in _mp_bits(word)})
            j = rng.choice(occupied)
            active = [i for i in range(len(grid)) if grid[i][j]]
            i = rng.choice(active)
            if step % 5 == 0:
                grid[i][j] = cell_for(side, i, j)
            else:
                t = rng.randrange(len(grid[i][j]))
                old = G.elements[grid[i][j][t]]
                options = [G.idx((a, old[1])) for a in range(61)
                           if G.idx((a, old[1])) not in grid[i][j]]
                grid[i][j][t] = rng.choice(options)
                if not whole_sidon(grid[i][j]):
                    continue
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in signatures:
                continue
            signatures.add(rec["signature"])
            if not screen(rec, side):
                continue
            # Historical words matter only if they remain genuine kernels.
            for known in current["words"]:
                if _mp_syndrome(rec["columns"], known) == 0:
                    note(rec, known)
            query(rec, side, 6, offset+1000003+step*7919)
            if merit(rec) <= merit(current):
                continue
            query(rec, side, 20, offset+1300007+step*7919)
            if merit(rec) > merit(current):
                current = rec
                keepers.append(rec)

        unique = {rec["signature"]: rec for rec in keepers}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:3]
        for rec in finalists:
            query(rec, side, 48 if side == "A" else 28, offset+2700011)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    aa, bb = pool("A"), pool("B")

    def localized(hx, hz, a, b, n):
        """Verified nontrivial logicals, reduced with ORIGINAL stabilizers."""
        readings = []
        for kind, opposite, stabilizers, rec in (
                ("Z", hx, hz, a), ("X", hz, hx, b)):
            span = _mp_span(stabilizers)
            diagnostic = random.Random(seed + (3100009 if kind == "Z"
                                               else 4100011))
            candidates = []
            for word in rec["words"][:5]:
                for block in range(3 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((3*j+block)*q)
                            for j in range(2))
                    else:
                        actual = word << (3*block*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if _mp_nontrivial(actual, span):
                        candidates.append(actual)
            found = n+1
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                neighborhood = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:40]
                for _ in range(160):
                    first, second = diagnostic.sample(neighborhood, 2)
                    candidate = actual ^ first ^ second
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                assert actual and _mp_nontrivial(actual, span)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                found = min(found, actual.bit_count())
            readings.append(found)
        return tuple(readings)

    best = None
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1464 and k >= 183
                    and 125*n <= 1000*k < 175*n):
                continue
            assert _component_rows(hx, hz, q, "A") == a["rows"]
            assert _component_rows(hx, hz, q, "B") == b["rows"]
            logicals = localized(hx, hz, a, b, n)
            detected = min(a["d"], b["d"], *logicals)
            key = (detected, k*detected*detected,
                   a["d"]+b["d"], -a["cost"]-b["cost"])
            if best is None or key > best[0]:
                best = (key, A, B, hx, hz, k,
                        (a["d"], b["d"]), logicals)
    if best is None:
        raise ValueError("Sidon repaired pairs missed cell 0.150")
    key, A, B, cached_hx, cached_hz, k, components, logicals = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1464
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "card": (mode, topology, cohort),
            "cells": {"A": A, "B": B}, "screen_k": k,
            "component_readings": components,
            "verified_localized_logical_readings": logicals,
            "quality": key}


def _legacy_cubic_ctx(rng, G, idx):
    """Unused historical two-row optimizer; NEW_CTX above replaces it."""
    design = idx % 20
    sa = _cubic_support(rng, A_SHAPE[1], design % 4, 7)
    sb = _cubic_support(rng, B_SHAPE[1], design // 4, 9)
    assert _grid_weights(sa, sb) == 12
    bits_a = _coset_pattern(rng, sa)
    bits_b = _coset_pattern(rng, sb)
    penalty = 320 + 40 * (idx // 20)
    fallback = None

    for attempt in range(5):
        starts = []
        for _ in range(2):
            A = _initial_cubic_cells(rng, G, sa, bits_a)
            B = _initial_cubic_cells(rng, G, sb, bits_b)
            cost = sum(_verified_costs(G, A, B, penalty))
            starts.append((cost, A, B))
        _, A, B = min(starts, key=lambda item: item[0])
        A, B, cost = _optimize_labels(rng, G, A, B, penalty)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.325 * n <= k < 0.375 * n):
            continue
        ar = _component_rows(hx, hz, G.q, "A")
        br = _component_rows(hx, hz, G.q, "B")
        short = (_small_kernel_penalty(ar, A_SHAPE[1] * G.q) +
                 _small_kernel_penalty(br, B_SHAPE[1] * G.q))
        key = (short, cost)
        if fallback is None or key < fallback[0]:
            fallback = (key, A, B, k)
        if short == 0:
            break

    if fallback is None:
        raise ValueError("no exact-rank candidate in target rate band")
    key, A, B, k = fallback
    hx, hz, _ = lifted_matrices(G, A, B)
    if not css_ok(hx, hz):
        raise ValueError("verified construction failed CSS check")
    assert max(row.bit_count() for row in hx + hz) <= 12
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen_k": k, "structural_penalty": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the optimized full cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling and uniform labels alone suppress very short logicals in the order-52 (2,6)x(2,4) bed: all five measured parent codes had distance 4, with many sector-1 witnesses.
# REFUTED: Adding a third cubic A check-block row on C15:C2 (3,7)x(2,6) produces five measured distances above ten: the measured selections instead returned 8,8,10,8,8, with minimum witnesses entirely in sector 1 and fitness 24.0532.
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
