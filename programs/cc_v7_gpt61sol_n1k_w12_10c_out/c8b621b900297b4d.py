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
CONCEPT = ("On cyclic metacyclic Z29:Z1, order 29, shapes (3,6)x(3,7), B-heavy margin cards and witness-directed label mutations should improve the parent's measured five-code fitness 49.6736 in cell 0.250. A non-improving measured fitness refutes this hypothesis.")
GROUP = ("Frozen metacyclic cyclic specialization m=29,r=1,s=1, q=29. This exact bed has measured programs scoring 46.2 and 49.7; retain the bed and change only its filling. Neither score certifies the reported distances.")
SHAPES = ("Frozen A=(3,6), B=(3,7): n=1479, k>=348, guaranteed rate 12/51=0.235294. Component lengths are 174 and 203 with 87 checks each. Parent minimum X witnesses of weight 14 or 15 occupied only sector-1 Acol/Bcol blocks, prominently Acol2..4 and many B columns; strengthen B while explicitly screening the cheaper A component.")
FILLING = ("Twenty distinct margin cards principally trade A-row weight eight for six and B-row weight eight for nine, maintaining both quantum row caps <=12; four cards preserve stronger A margins as controls. Optimize voltages, then mutate labels using actual surviving component-kernel witnesses from verified lifts. Deep finalist searches retain independent A/B readings instead of clamping both to the initial bottleneck. Exact full-code ranks enforce [0.225,0.275); all witness readings remain upper bounds, not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r025_m29r1s1_36x37_Bheavy_witness_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Same metacyclic rule, resized cyclic specialization.
    m, r, s = 29, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 6)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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
    """Rebudget the components and optimize genuine kernel witnesses."""
    from functools import lru_cache
    from itertools import combinations

    card = idx % 20
    cohort = (idx // 20) % 3
    q = G.q
    mask = (1 << q) - 1

    # Distinct column-degree/row-margin cards, not relabeling images.
    # The first sixteen strengthen B, the observed X-witness bottleneck.
    b_cards = (
        (4,4,4,4,4,4,3),
        (5,4,4,4,4,3,3),
        (5,5,4,4,3,3,3),
        (5,5,5,3,3,3,3),
        (6,4,4,4,3,3,3),
        (6,5,4,3,3,3,3),
        (6,6,3,3,3,3,3),
        (4,4,4,4,4,3,3),
        (5,4,4,4,3,3,3),
        (5,5,4,3,3,3,3),
        (6,4,4,3,3,3,3),
        (6,5,3,3,3,3,3),
        (4,4,4,4,3,3,3),
        (5,4,4,3,3,3,3),
        (5,5,3,3,3,3,3),
        (6,4,3,3,3,3,3),
    )
    if card < 16:
        ac = (3,) * 6
        at = (6, 6, 6)
        bc = b_cards[card]
        total = sum(bc)
        bt = tuple(total // 3 + int(i < total % 3)
                   for i in range(3))
    else:
        controls = (
            ((4,4,4,3,3,3), (7,7,7),
             (5,4,3,3,3,3,3), (8,8,8)),
            ((4,4,3,3,3,3), (7,7,6),
             (5,4,3,3,3,3,3), (8,8,8)),
            ((4,4,4,4,4,4), (8,8,8),
             (4,4,4,3,3,3,3), (8,8,8)),
            ((4,4,4,4,3,3), (8,7,7),
             (4,4,4,3,3,3,3), (8,8,8)),
        )
        ac, at, bc, bt = controls[card - 16]

    def margin_support(degrees, targets, zeros):
        """Exact margins, legal cells, and at least two checks per column."""
        options = []
        for degree in degrees:
            choices = [
                (a, b, degree-a-b)
                for a in range(6) for b in range(6)
                if 0 <= degree-a-b <= 5
                and sum(x > 0 for x in (a,b,degree-a-b)) >= 2
                and (zeros > 0 or min(a,b,degree-a-b) > 0)
            ]
            rng.shuffle(choices)
            options.append(choices)

        @lru_cache(None)
        def feasible(j, a, b, c, z):
            if min(a,b,c,z) < 0:
                return False
            if j == len(degrees):
                return (a,b,c,z) == (0,0,0,0)
            if a+b+c != sum(degrees[j:]) or z > len(degrees)-j:
                return False
            return any(
                feasible(j+1, a-v[0], b-v[1], c-v[2],
                         z-sum(x == 0 for x in v))
                for v in options[j]
            )

        state = tuple(targets) + (zeros,)
        if not feasible(0, *state):
            raise ValueError("infeasible margin card")
        result = [[], [], []]
        for j in range(len(degrees)):
            choices = []
            for v in options[j]:
                nxt = tuple(state[i]-v[i] for i in range(3))
                nxt += (state[3]-sum(x == 0 for x in v),)
                if feasible(j+1, *nxt):
                    choices.append((v, nxt))
            v, state = choices[rng.randrange(len(choices))]
            for i in range(3):
                result[i].append(v[i])
        return result

    # Additional harness cohorts change B's cell multiplicities too.
    # Monomial A cards deliberately retain all three base-check incidences.
    sa = max((margin_support(ac, at, 0) for _ in range(4)),
             key=_count_minor_floor)
    sb = max((margin_support(bc, bt, cohort) for _ in range(4)),
             key=_count_minor_floor)
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def component_rows(cells, side):
        """Extract only from the scaffold's verified construction."""
        empty_a = [[[] for _ in range(6)] for _ in range(3)]
        empty_b = [[[] for _ in range(7)] for _ in range(3)]
        hx, hz, _ = lifted_matrices(
            G, cells if side == "A" else empty_a,
            cells if side == "B" else empty_b)
        ar, br = _component_rows(hx, hz, q)
        return ar if side == "A" else br

    def syndrome_columns(rows, width):
        columns = [0] * width
        for check, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length()-1] |= 1 << check
                row ^= bit
        return columns

    def syndrome(columns, word):
        value = 0
        while word:
            bit = word & -word
            value ^= columns[bit.bit_length()-1]
            word ^= bit
        return value

    def kernel_search(columns, checks, seed, trials):
        """Random systematic kernels, plus short pair/triple combinations.

        Every returned word is a genuine kernel vector of these columns.
        Finding no lighter word is only a budget-limited upper-bound reading.
        """
        local = random.Random(seed)
        width = len(columns)
        best = width + 1
        witnesses = {}

        def note(word):
            nonlocal best, witnesses
            if not word:
                return
            weight = word.bit_count()
            if weight < best:
                best = weight
                witnesses = {
                    w: wt for w, wt in witnesses.items()
                    if wt <= best + 2
                }
            if weight <= best + 2:
                witnesses[word] = weight
                if len(witnesses) > 32:
                    witnesses = dict(sorted(
                        witnesses.items(), key=lambda item: item[1])[:16])

        for _ in range(trials):
            order = list(range(width))
            local.shuffle(order)
            pivots = [None] * checks
            words = []
            for column in order:
                value = columns[column]
                word = 1 << column
                while value:
                    p = value.bit_length()-1
                    previous = pivots[p]
                    if previous is None:
                        pivots[p] = (value, word)
                        break
                    value ^= previous[0]
                    word ^= previous[1]
                if value == 0:
                    note(word)
                    words.append(word)

            words.sort(key=int.bit_count)
            short = words[:24]
            for i, word in enumerate(short):
                for other in short[:i]:
                    note(word ^ other)
            for a, b, c in combinations(short[:8], 3):
                note(a ^ b ^ c)
            if best <= 10:
                break
        found = sorted(witnesses, key=int.bit_count)[:8]
        assert all(syndrome(columns, word) == 0 for word in found)
        return best, found

    def optimize_component(support, side):
        width = len(support[0]) * q
        seed = BASE_SEED + 1009*idx + (0 if side == "A" else 500009)
        bank = set()

        def canonical_word(word):
            # Quotient witness translations, not candidate labelings.
            blocks = [(word >> (j*q)) & mask
                      for j in range(width // q)]
            return min(
                sum((((block << shift) |
                      (block >> (q-shift))) & mask) << (j*q)
                    for j, block in enumerate(blocks))
                for shift in range(q)
            )

        def remember(words):
            nonlocal bank
            bank.update(canonical_word(word) for word in words
                        if word.bit_count() <= 24)
            if len(bank) > 128:
                bank = set(sorted(bank, key=int.bit_count)[:128])

        def rating(record):
            reading = record["upper"]
            for word in bank:
                weight = word.bit_count()
                if weight < reading and syndrome(record["cols"], word) == 0:
                    reading = weight
            return (reading, -record["cost"], record["minor"])

        def probe(cells):
            rows = component_rows(cells, side)
            columns = syndrome_columns(rows, width)
            minor = _minor_reading(
                rows, cells, q, inverse=(side == "B"))
            reading, words = kernel_search(
                columns, len(rows), seed, 24)
            remember(words)
            return {
                "cells": cells, "rows": rows, "cols": columns,
                "minor": minor, "upper": min(minor, reading),
                "cost": _voltage_cost(cells, q), "words": words,
            }

        pool = []
        for _ in range(8):
            cells, _ = _optimized_labels(rng, G, support)
            pool.append(probe(cells))

        # Independent deeper probes reduce dependence on one favorable draw.
        for record in sorted(pool, key=rating, reverse=True)[:3]:
            reading, words = kernel_search(
                record["cols"], len(record["rows"]), seed+1, 96)
            record["upper"] = min(record["upper"], reading)
            record["words"] = words
            remember(words)
        current = max(pool, key=rating)

        for step in range(20):
            surviving = [
                word for word in bank
                if syndrome(current["cols"], word) == 0
            ]
            witness = min(surviving, key=int.bit_count) if surviving else None
            active = []
            if witness:
                while witness:
                    bit = witness & -witness
                    active.append((bit.bit_length()-1) // q)
                    witness ^= bit

            candidate = [[list(cell) for cell in row]
                         for row in current["cells"]]
            for _ in range(2 if step % 6 == 0 else 1):
                j = (active[rng.randrange(len(active))]
                     if active and rng.random() < 0.8
                     else rng.randrange(len(support[0])))
                occupied = [i for i in range(3) if candidate[i][j]]
                i = occupied[rng.randrange(len(occupied))]
                cell = candidate[i][j]
                t = rng.randrange(len(cell))
                element = G.pick(rng)
                while element in cell:
                    element = G.pick(rng)
                cell[t] = element

            # Allow a small cycle trade, but not uncontrolled cycle damage.
            if _voltage_cost(candidate, q) > current["cost"] + 2:
                continue
            proposed = probe(candidate)
            pk, ck = rating(proposed), rating(current)
            accept = pk > ck
            if not accept and pk[0] == ck[0]:
                accept = (proposed["cost"] <= current["cost"] + 1
                          and rng.random() < 0.15)
            if accept:
                current = proposed
                pool.append(proposed)
                pool = sorted(pool, key=rating, reverse=True)[:6]

        finalists = sorted(pool, key=rating, reverse=True)[:2]
        for record in finalists:
            reading, words = kernel_search(
                record["cols"], len(record["rows"]), seed+2, 256)
            record["upper"] = min(record["upper"], reading)
            remember(words)
        chosen = max(finalists, key=rating)
        return chosen["cells"], rating(chosen)[0], chosen["cost"]

    for restart in range(3):
        A, da, cost_a = optimize_component(sa, "A")
        B, db, cost_b = optimize_component(sb, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 225*n <= 1000*k < 275*n):
            continue
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {
            "idx": idx, "card": card, "cells": {"A": A, "B": B},
            "k": k, "component_readings": (da, db),
            "voltage_costs": (cost_a, cost_b),
        }
    raise ValueError("no optimized filling passed the exact lifted-rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell, without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
# REFUTED: Balanced-kernel-v1 would yield five readings above 14 on Z29 (3,6)x(3,7); its measured top five instead read 15,15,14,14,14, with minimum X witnesses confined to sector 1.
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
