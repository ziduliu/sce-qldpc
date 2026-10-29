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
CONCEPT = ("For cyclic C29, order 29, shapes (2,4)x(2,4), sum-free binomial directions and independently optimized inter-row voltages should improve measured circuit reliability in cell 0.250 despite weight-twelve extraction.")
GROUP = ("Jump from C17 to the genuinely different C29. The C29 (2,4)x(2,4) bed has reached about 7.2, motivating a new geometry-controlled filling without attributing that performance to the group alone. Order 29 is the largest legal lift for size factor twenty and gives n=580.")
SHAPES = ("Equal (2,4) components have 116 variables and 58 checks. The guaranteed bound is kb=116, rate 0.200. Zero augmentation and full nontrivial-character ranks predict component rank 56 and complete k=132; require both through verified lifts, giving rate 132/580=0.227586 in cell 0.250. Parent failures implicated Bcol4 in slots8/36 and Acol0/Bcol2, Acol1/Bcol1 and Arow1/Brow2 in slot53, but sparse failure maps do not establish a persistent hotspot.")
FILLING = ("Every cell is a distinct-element doubleton: component row degree eight and column degree four give both CSS row-weight sums twelve. A deterministic catalogue supplies distinct joint gap architectures after removing common multipliers, row/column permutations, independent cell origins and component exchange. Signed three-gap sums are excluded within each component row. Anneal origins without changing prescribed gaps, using helper-derived actions checked against the final full lift. Screen exact kernels through six, sampled protection capped at sixteen, weighted short-witness concentration and triple boundaries. Require full k=132 and 33408 six-round CNOTs. This increases extraction work over the parent's 28560-29376 measured budgets and tests whether stronger regular protection compensates: slot53 had eight X failures and 230 logical flips despite d2s=14, versus one failure per basis in slot10 at d2s=13. Screens are hypotheses about circuit pL, not calibrated failure estimates or ASC hook certificates.")
FAMILY_NAME = "r025_C29_24x24_sumfree_regular4_voltage_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C29 requires order 29")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Canonical regular-four gap contracts, then fresh voltage search."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 29
    assert A_SHAPE == B_SHAPE == (2, 4)
    sa = [[2] * 4 for _ in range(2)]
    sb = [[2] * 4 for _ in range(2)]

    # HX: eight A terms plus four B terms.
    # HZ: eight B terms plus four A terms.
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert CELL_MAX >= 2

    def direction(value):
        value %= q
        return min(value, q - value)

    def row_clean(gaps):
        signed = tuple(v for gap in gaps for v in (gap, -gap))
        # Repeated directions are included in this test. It excludes
        # same-check-orbit triangles, not all component six-cycles.
        return all((a + b + c) % q
                   for a in signed for b in signed for c in signed)

    row_book = [
        gaps for gaps in combinations(range(1, (q + 1) // 2), 4)
        if row_clean(gaps)
    ]
    assert len(row_book) >= 4

    def geometry_key(grids):
        """Remove obvious joint gap orbits, not arbitrary Tanner equivalence."""
        images = []
        for u in range(1, (q + 1) // 2):
            sides = []
            for grid in grids:
                pairs = [
                    (direction(u * grid[0][j]),
                     direction(u * grid[1][j]))
                    for j in range(4)
                ]
                forward = tuple(sorted(pairs))
                reverse = tuple(sorted((b, a) for a, b in pairs))
                sides.append(min(forward, reverse))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Pure deterministic catalogue: slots differ in prescribed relative
    # geometry before any voltage search. Extra slots consume new cards.
    # No earlier selected code is transformed into another submitted slot.
    catalogue_rng = random.Random(290419731)
    book, seen = [], set()
    for attempt in range(10000):
        grids = []
        for side in range(2):
            upper = list(catalogue_rng.choice(row_book))
            lower = list(catalogue_rng.choice(row_book))
            for shuffle in range(64):
                catalogue_rng.shuffle(lower)
                if all(a != b for a, b in zip(upper, lower)):
                    break
            else:
                break
            grids.append([upper, list(lower)])
        if len(grids) != 2:
            continue
        signature = geometry_key(grids)
        if signature in seen:
            continue
        seen.add(signature)
        book.append((signature, grids))
        if len(book) > idx:
            break
    if len(book) <= idx:
        raise ValueError("No unused sum-free joint geometry contract")

    contract, grids = book[idx]
    geometry = dict(zip(("A", "B"), grids))
    for grid in geometry.values():
        assert all(row_clean(row) for row in grid)
        assert all(len(set(row)) == 4 for row in grid)
        assert all(grid[0][j] != grid[1][j] for j in range(4))

    def draw_cell(side, i, j):
        origin = G.pick(rng)
        gap = geometry[side][i][j]
        return [origin, G.mul(origin, (gap,))]

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(4)]
                for i in range(2)]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only singleton blocks returned by the verified construction.
    # Final component projections are independently checked below.
    actions = {"A": {}, "B": {}}
    block_mask = (1 << q) - 1
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz)

    def component(grid, side):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows, len(grid[0]) * q

    def columns(rows, nc):
        out = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def cycle_energy(rows, nc):
        nr = len(rows)
        adj = [0] * nr
        four = 0
        for i in range(nr):
            for j in range(i):
                common = (rows[i] & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adj[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adj[i] & adj[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = 0
        for col in columns(rows, nc):
            d = col.bit_count()
            stars += d * (d - 1) * (d - 2) // 6
        # Exact six-cycle count without four-cycles; otherwise a heuristic.
        six = max(0, triangles - stars)
        return (80.0 * four + six / 4.0) / q, four, six

    def boundary_scan(cols):
        """Exact existence test for kernel words of weights 1..6.

        Collisions between subsets of size at most three cover all six
        weights. Boundary statistics are only trapping-set heuristics.
        """
        nc = len(cols)
        seen = {0: 0}
        bad = False
        risk = 0
        loads = [0] * (nc // q)

        def record(syndrome, word):
            nonlocal bad
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif old != word:
                bad = True

        for i, col in enumerate(cols):
            record(col, 1 << i)
        for i in range(nc):
            for j in range(i):
                record(cols[i] ^ cols[j], (1 << i) | (1 << j))
        for i in range(nc - 2):
            for j in range(i + 1, nc - 1):
                pair = cols[i] ^ cols[j]
                word = (1 << i) | (1 << j)
                for t in range(j + 1, nc):
                    syndrome = pair ^ cols[t]
                    record(syndrome, word | (1 << t))
                    weight = syndrome.bit_count()
                    if weight <= 6:
                        cost = (8 - weight) ** 2
                        risk += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[t // q] += cost
        return int(bad), risk / nc, max(loads) / q

    def kernel_exposure(cols, trials, side):
        """Sample actual kernel words; protection remains an upper bound.

        Common permutation streams compare finalists in a slot.
        Fundamental words, pair XORs and bounded descent expose ambiguity;
        no extra protection reward is given above weight sixteen.
        """
        srng = random.Random(290791 + 1009 * idx + int(side == "B"))
        nc = len(cols)
        order = list(range(nc))
        distance = nc + 1
        small = set()

        def offer(word):
            nonlocal distance
            if not word:
                return
            weight = word.bit_count()
            distance = min(distance, weight)
            if weight <= 16:
                small.add(word)

        for trial in range(trials):
            srng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)

            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)

            for word in sorted(basis, key=int.bit_count)[:6]:
                for step in range(3):
                    best, best_weight = word, word.bit_count()
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best_weight:
                            best, best_weight = candidate, candidate.bit_count()
                    if best == word:
                        break
                    word = best
                    offer(word)
            if distance <= 6:
                break

        # Quotient only the verified cyclic translation action on words.
        # This reduces uneven sampling of the same short-word orbit.
        mask = (1 << q) - 1
        for j in range(nc // q):
            seed = cols[j * q]
            for shift in range(q):
                translated = 0
                for r in range(2):
                    block = (seed >> (r * q)) & mask
                    translated |= (
                        ((block << shift) | (block >> (q - shift))) & mask
                    ) << (r * q)
                assert cols[j * q + shift] == translated

        orbits = set()
        for word in small:
            blocks = [(word >> (j * q)) & mask for j in range(nc // q)]
            images = []
            for shift in range(q):
                images.append(sum(
                    (((block << shift) | (block >> (q - shift))) & mask)
                    << (j * q) for j, block in enumerate(blocks)))
            orbits.add(min(images))

        loads = [0] * (nc // q)
        burden = 0
        for word in orbits:
            cost = (18 - word.bit_count()) ** 2
            burden += cost
            for j in range(len(loads)):
                loads[j] += cost * ((word >> (j * q)) & mask).bit_count()
        return distance, burden, max(loads)

    def design_component(supports, side):
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        locals_found = []
        for restart in range(6):
            grid = draw_grid(side)
            rows, nc = component(grid, side)
            current = cycle_energy(rows, nc)
            local = copy_grid(grid)
            local_stats = current

            for step in range(260):
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                # Change the whole-cell origin, preserving its gap.
                grid[i][j] = draw_cell(side, i, j)
                trial_rows, _ = component(grid, side)
                proposed = cycle_energy(trial_rows, nc)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.985 ** step) + 0.08
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    current = proposed
                    if (current[0] < local_stats[0] or
                            (current[0] == local_stats[0]
                             and rng.random() < 0.06)):
                        local = copy_grid(grid)
                        local_stats = current
                else:
                    grid[i][j] = old

            rows, nc = component(local, side)
            # Zero augmentation forces two missing ranks. Since
            # ord_29(2)=28, reject any additional character collapse.
            if gf2_rank(rows) != 56:
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found[:4]:
            cols = columns(rows, nc)
            assert all(col.bit_count() == 4 for col in cols)
            bad, risk, peak = boundary_scan(cols)
            distance, low, word_peak = kernel_exposure(cols, 48, side)
            key = (bad, max(0, 16 - distance),
                   low, word_peak, peak, risk, stats[1], stats[2])
            finalists.append({
                "grid": grid, "key": key, "d": distance,
                "low": low, "word_peak": word_peak,
                "bad": bad, "risk": risk,
                "peak": peak, "four": stats[1], "six": stats[2],
            })
        finalists.sort(key=lambda item: item["key"])
        if not finalists:
            raise ValueError("No rank-safe component filling")
        return finalists

    apool = design_component(sa, "A")
    bpool = design_component(sb, "B")
    winner = None
    winning_key = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # The augmentation prediction is never substituted for the
            # evaluator-convention ranks of the complete CSS matrices.
            if n != 580 or k != 132:
                continue
            da, db = max(0, 16 - a["d"]), max(0, 16 - b["d"])
            key = (
                max(a["bad"], b["bad"]),
                max(da, db), da + db,
                max(a["low"], b["low"]),
                a["low"] + b["low"],
                max(a["word_peak"], b["word_peak"]),
                a["word_peak"] + b["word_peak"],
                max(a["peak"], b["peak"]),
                max(a["risk"], b["risk"]),
                a["risk"] + b["risk"],
                max(a["four"], b["four"]),
                a["four"] + b["four"],
                a["six"] + b["six"],
            )
            if winning_key is None or key < winning_key:
                winning_key = key
                winner = (A, B, hx, hz)

    if winner is None:
        raise ValueError("No finalist retained verified k=132 in cell 0.250")
    A, B, hx, hz = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 33408
    assert 40 * 132 > 9 * 580 and 40 * 132 < 11 * 580

    # Check cached helper-derived actions against the final complete lift.
    actual_a = []
    for i in range(2):
        for t in range(q):
            row = hx[(i * 4) * q + t]
            actual_a.append(sum(
                ((row >> (j * 4 * q)) & block_mask) << (j * q)
                for j in range(4)))
    actual_b = [
        row & ((1 << (4 * q)) - 1) for row in hz[:2 * q]
    ]
    assert component(A, "A")[0] == actual_a
    assert component(B, "B")[0] == actual_b
    assert all(len(cell) == 2 and
               len({G._key(g) for g in cell}) == 2
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "architecture": contract, "k": 132,
            "screen": winning_key, "cnots": 33408,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without subsequent random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: Increasing d2s guarantees better memory on the parent C17 bed: slot53 had d2s=14, eight X failures/200000 and s=6.806, while slot10 had d2s=13, one failure per basis and s=7.258.
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
