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
CONCEPT = ("For cyclic C17, order 17, shapes (3,5)x(3,5), augmentation ranks two and three should give verified k=70 at n=578 while stronger low-rate components and lighter extraction improve cell 0.100's worse-basis circuit reliability.")
GROUP = ("Preserve the cyclic construction rule and one-coordinate encoding under L2. The new size factor 34 makes the parent's order 29 illegal, so resize to order 17 and n=578. RUN STATE reports approximately 7.3 on C17 (3,5)x(3,5), motivating this bed without establishing an advantage for these new augmentation-rank contracts.")
SHAPES = ("Equal-length (3,5) components each have 85 variables and 51 checks. The guaranteed quantum bound is kb=68, rate 0.11765. Require helper-built component ranks 50 and 51 and complete k=70, giving rate 70/578=0.12111 in target cell 0.100. Parent failures span changing Acol/Bcol data blocks, with mostly weak Acol/Brow check enrichment; sparse maps and widespread logical flips do not identify one persistent repair hotspot.")
FILLING = ("Twenty support-rank contracts cross five rank-two A augmentation architectures with four rank-three B architectures. Primary cells contain one or two distinct elements; row totals <=7 and column totals <=5 keep BOTH CSS check-weight sums <=12. Extra harness slots consume further row/column-canonical support pairs, not transformed saved codes. Fresh whole-cell annealing changes internal differences as well as origins, using verified singleton actions and final component-equality checks. Retain exact kernel detection through six, sampled protection capped at sixteen, weighted short-witness concentration and triple boundaries. Full ranks enforce k=70 and CSS. Primary six-round CNOT budgets are 26928..31824 versus the parent's 33408. Target its 2-10 Z failures/200000 and broad correlated logical flips; structural screens remain uncalibrated pL hypotheses, not ASC hook certificates.")
FAMILY_NAME = "r010_C17_35x35_rank23_gatelean_wholecell_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C17 requires order 17")
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

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Augmentation-rank contracts, fresh labels and exact target ranks."""
    import math
    from itertools import permutations

    q = G.q
    assert q == 17
    assert A_SHAPE == B_SHAPE == (3, 5)
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    def augmentation_rank(grid):
        return gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in grid
        ])

    def support_key(grid):
        """Canonical counts under row/column permutations only.

        Different keys prevent obvious support-orbit copies. They are
        not a claim of arbitrary lifted-Tanner nonisomorphism.
        """
        return min(tuple(sorted(
            tuple(grid[i][j] for i in order) for j in range(5)))
            for order in permutations(range(3)))

    def legal_support(grid, rank):
        return (
            all(4 <= sum(row) <= 7 for row in grid)
            and all(3 <= sum(column) <= 5 for column in zip(*grid))
            and all(0 <= count <= 2 for row in grid for count in row)
            and augmentation_rank(grid) == rank
        )

    # Five count architectures with augmentation rank exactly two:
    # one added term; two in one row; two in one column;
    # a shared two-row rectangle; a five-column parity dependence.
    aprofiles = [
        ((2, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((2, 2, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((2, 1, 1, 1, 1), (2, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((2, 2, 1, 1, 1), (2, 2, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 1), (1, 1, 1, 1, 2)),
    ]
    # Independent parity perturbations give augmentation rank three.
    bprofiles = [
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((2, 2, 1, 1, 1), (1, 1, 2, 1, 1), (1, 1, 1, 1, 1)),
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 1), (1, 1, 1, 1, 1)),
        ((2, 2, 1, 1, 1), (1, 2, 2, 1, 1), (1, 1, 1, 1, 1)),
    ]
    assert all(legal_support(grid, 2) for grid in aprofiles)
    assert all(legal_support(grid, 3) for grid in bprofiles)
    assert len({support_key(grid) for grid in aprofiles}) == 5
    assert len({support_key(grid) for grid in bprofiles}) == 4
    briefs = [(a, b) for a in range(5) for b in range(4)]

    if idx >= 20:
        # Deterministic architecture extension for larger harness budgets.
        # Zero/doubleton redistributions preserve live column protection;
        # no optimized filling from another slot is reused or transformed.
        for profiles, rank in ((aprofiles, 2), (bprofiles, 3)):
            crng = random.Random(17035010 + rank)
            seen = {support_key(grid) for grid in profiles}
            for attempt in range(12000):
                grid = [[1] * 5 for _ in range(3)]
                sites = crng.sample(range(15), crng.randrange(2, 8))
                for site in sites:
                    i, j = divmod(site, 5)
                    grid[i][j] = 2 if crng.random() < 0.80 else 0
                if not legal_support(grid, rank):
                    continue
                signature = support_key(grid)
                if signature in seen:
                    continue
                seen.add(signature)
                profiles.append(tuple(tuple(row) for row in grid))
                if len(profiles) >= 14:
                    break
        used = set(briefs)
        extras = [
            (a, b) for a in range(len(aprofiles))
            for b in range(len(bprofiles)) if (a, b) not in used
        ]
        extras.sort(key=lambda pair: (
            sum(map(sum, aprofiles[pair[0]]))
            + sum(map(sum, bprofiles[pair[1]])), pair))
        briefs.extend(extras)

    if idx >= len(briefs):
        raise ValueError("Unused augmentation-rank architecture book exhausted")
    pa, pb = briefs[idx]
    sa = [list(row) for row in aprofiles[pa]]
    sb = [list(row) for row in bprofiles[pb]]
    counts = {"A": sa, "B": sb}
    contract = (support_key(sa), support_key(sb))

    # Check BOTH extraction budgets, not just total gate count.
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert CELL_MAX >= 2
    expected_cnots = 6 * q * (
        (cb + rb) * sum(map(sum, sa))
        + (ca + ra) * sum(map(sum, sb))
    )

    def draw_cell(side, i, j):
        # Whole-cell redraws change internal differences as well as
        # origins. Sampling without replacement prevents F2 cancellation.
        return rng.sample(G.elements, counts[side][i][j])

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(5)]
                for i in range(3)]

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
        srng = random.Random(170791 + 1009 * idx + int(side == "B"))
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
                for r in range(3):
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
                # Preserve counts and parity, but redraw the entire cell.
                # Internal ratios are mutable, not just voltage origins.
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
            # Augmentation ranks two/three imply deficits one/zero.
            # Require the actual verified component ranks, rejecting
            # any additional collapse in nontrivial cyclic sectors.
            deficit = 1 if side == "A" else 0
            if gf2_rank(rows) != len(rows) - deficit:
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found[:4]:
            cols = columns(rows, nc)
            assert all(3 <= col.bit_count() <= 5 for col in cols)
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
            if n != 578 or k != 70:
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
        raise ValueError("No finalist retained verified k=70 in cell 0.100")
    A, B, hx, hz = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == max(wx, wz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert sum(row.bit_count() for row in hx + hz) * 6 == expected_cnots
    assert 0 < 8 * 70 < 578

    # Check cached singleton composition against the final complete lift.
    # Project exactly the evaluator's component blocks and column order.
    actual_a = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            actual_a.append(sum(
                ((row >> (j * cb * q)) & block_mask) << (j * q)
                for j in range(ca)))
    actual_b = [
        row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]
    ]
    assert component(A, "A")[0] == actual_a
    assert component(B, "B")[0] == actual_b
    assert gf2_rank(actual_a) == 50
    assert gf2_rank(actual_b) == 51
    assert augmentation_rank(sa) == 2
    assert augmentation_rank(sb) == 3
    assert all(
        len(cell) == counts[side][i][j]
        and len({G._key(g) for g in cell}) == len(cell)
        for side, grid in (("A", A), ("B", B))
        for i, row in enumerate(grid) for j, cell in enumerate(row)
    )
    return {"idx": idx, "architecture": contract, "k": 70,
            "support_pair": (pa, pb), "augmentation_ranks": (2, 3),
            "screen": winning_key, "cnots": expected_cnots,
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
# REFUTED: Sum-free regular-four C29 directions suffice to saturate the memory score: none of the five measured parent codes reached its 7.63 ceiling, and their Z bases failed 2-10 times per 200000 shots.
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
