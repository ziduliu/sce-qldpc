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
CONCEPT = ("For nonabelian C4 semidirect C4 with inversion action, order16, shapes(3,5)x(3,5), rank-one augmentation on both sides, compensated reinforcement and common-noise decoder screening should improve worse-basis circuit fitness at verified80<=k<=94 in cell0.150.")
GROUP = ("Jump from C17 to x^4=y^4=1, y*x*y^-1=x^-1, a genuinely different nonabelian order16 group with centre four and ten conjugacy classes. RUN STATE includes an invariant-matched order16 (3,5)x(3,5) bed reaching about7.3; these invariants do not uniquely identify its archived group. The algebra and related reading motivate this experiment, not a proven circuit advantage. Exclude involution differences inside doubletons.")
SHAPES = ("Retain equal(3,5) shapes: each component now has80 variables and48 checks, n=544 and kb=64, guaranteed rate2/17. Require actual full-CSS80<=k<=94, rates0.14706..0.17279, entirely inside cell0.150; component rank deficits only shortlist candidates. Parent failures implicated changing Acol/Bcol blocks, including Bcol4 in slot0 and Bcol2 in slots3/4, and several Acol/Brow check blocks. One or two failures per basis do not establish a persistent repair hotspot.")
FILLING = ("Twenty primary unordered count contracts combine five rank-one augmentation profiles with parity-preserving trinomial reinforcement. Columns total3..5, rows total4..7, cells contain0..3 distinct elements and both CSS row-weight sums are<=12. Primary six-round CNOT budgets are23040..29184; extra canonical count pairs extend the book to larger harness budgets, at most32256 CNOTs. Doubletons have order-four relative elements; trinomials have noncommuting relatives and no involution pair difference. Anneal fresh labels with verified singleton actions, screen exact component kernels through six, capped sampled witnesses and balanced triple boundaries, then use common-noise degree-weighted normalized-min-sum component stress. This is not relay-BP, ASC scheduling or measured circuit pL. Full ranks enforce the target band; prefer actual quantum-logical exclusion through four and verify CSS, weights and component projections. Target the parent's rare collective failures and its k=70 ceiling, not distance as fitness.")
FAMILY_NAME = "r015_C4sdC4_inversion_35x35_rank11_order4_decoderstress_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C4 semidirect C4 requires order16")
    # x^4=y^4=1, y*x*y^-1=x^-1.
    # (a,b) denotes x^a*y^b; inversion defines a C4 action.
    els = [(a, b) for b in range(4) for a in range(4)]

    def red(g):
        return (int(g[0]) % 4, int(g[1]) % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 4, (b + d) % 4)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        return ((a if b & 1 else -a) % 4, bi)

    def idx(g):
        a, b = red(g)
        return 4*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _retired_C17_CTX(rng, G, idx):
    """Inactive historical C17 search; never called by the new builder."""
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


def NEW_CTX(rng, G, idx):
    """Rank-one count cards and fresh nonabelian decoder-screened labels."""
    import math
    from itertools import permutations

    q = G.q
    assert q == 16 and A_SHAPE == B_SHAPE == (3, 5)
    width, mask = 5*q, (1 << q)-1

    def aug(support):
        return gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support])

    def count_key(support):
        return min(
            tuple(sorted(tuple(support[i][j] for i in p)
                         for j in range(5)))
            for p in permutations(range(3)))

    def base(odd, zero_rows):
        support = [[1 if j < odd else 2 for j in range(5)]
                   for _ in range(3)]
        for j, i in enumerate(zero_rows, odd):
            support[i][j] = 0
        return support

    profiles = [
        base(5, ()),
        base(4, (0,)),
        base(3, (0, 1)),
        base(2, (0, 1, 2)),
        base(1, (0, 0, 1, 2)),
    ]
    reinforcements = (
        (0, ((0, 0),)),
        (0, ((0, 0), (1, 1))),
        (0, ((0, 0), (1, 1), (2, 2))),
        (1, ((0, 0),)),
        (2, ((0, 0),)),
        (2, ((0, 0), (1, 1))),
        (4, ((0, 0),)),
    )
    for parent, sites in reinforcements:
        support = [list(row) for row in profiles[parent]]
        for i, j in sites:
            assert support[i][j] == 1
            support[i][j] = 3
        profiles.append(support)

    assert len({count_key(s) for s in profiles}) == len(profiles)
    for s in profiles:
        assert aug(s) == 1
        assert all(4 <= sum(row) <= 7 for row in s)
        assert all(3 <= sum(col) <= 5 for col in zip(*s))

    briefs = [(a, b) for a in range(5) for b in range(a, 5)]
    briefs += [(0, 5), (1, 5), (0, 8), (2, 9), (5, 6)]
    used = set(briefs)
    extras = [(a, b) for a in range(len(profiles))
              for b in range(a, len(profiles))
              if (a, b) not in used
              and _grid_weights(profiles[a], profiles[b]) <= W_MAX]
    extras.sort(key=lambda p: (
        sum(map(sum, profiles[p[0]])) + sum(map(sum, profiles[p[1]])),
        p))
    briefs.extend(extras)
    if idx >= len(briefs):
        raise ValueError("unused rank-one count-card book exhausted")
    pa, pb = briefs[idx]
    sa, sb = profiles[pa], profiles[pb]
    supports = {"A": sa, "B": sb}
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert _no_dead_lines(sa, sb)
    assert max(wx, wz) == _grid_weights(sa, sb) <= min(12, W_MAX)
    assert CELL_MAX >= 3
    expected_cnots = 48*q*(sum(map(sum, sa)) + sum(map(sum, sb)))

    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, q+1):
            power = G.mul(power, g)
            if power == G.e:
                orders[G.idx(g)] = order
                break
    long_elements = [g for g in G.elements if orders[G.idx(g)] == 4]
    assert long_elements

    def draw_cell(count):
        if count == 0:
            return []
        anchor = G.pick(rng)
        if count == 1:
            return [anchor]
        u = rng.choice(long_elements)
        if count == 2:
            return [anchor, G.mul(anchor, u)]
        for _ in range(256):
            v = rng.choice(long_elements)
            relative = G.mul(G.inv(u), v)
            if (orders[G.idx(relative)] == 4
                    and G.mul(u, v) != G.mul(v, u)):
                return [anchor, G.mul(anchor, u), G.mul(anchor, v)]
        raise ValueError("noncommuting trinomial was not realized")

    def draw(side):
        return [[draw_cell(w) for w in row] for row in supports[side]]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Every cached singleton action comes from the verified construction.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for row in grid:
            out = [0]*q
            for j, cell in enumerate(row):
                for g in cell:
                    for r, bits in enumerate(actions[side][G.idx(g)]):
                        out[r] ^= bits << (j*q)
            rows.extend(out)
        return rows

    def columns(rows, nc):
        out = [0]*nc
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def cycles(rows):
        adjacency = [0]*len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common*(common-1)//2
                if common:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(len(rows)):
            lower = adjacency[i] & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j)-1)).bit_count()
                lower ^= bit
        stars = sum(d*(d-1)*(d-2)//6
                    for d in (v.bit_count() for v in columns(rows, width)))
        # The second statistic is a proposal heuristic when C4 is nonzero.
        return four, max(0, triangles-stars)

    def boundary(cols):
        """Exact kernel existence through six; separate boundary surrogate."""
        seen = {0}
        bad = False
        loads, total = [0]*5, 0
        for value in cols:
            bad |= value in seen
            seen.add(value)
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                bad |= syndrome in seen
                seen.add(syndrome)
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    bad |= syndrome in seen
                    seen.add(syndrome)
                    weight = syndrome.bit_count()
                    if weight <= 4:
                        cost = (6-weight)**2
                        total += cost
                        for block in {i//q, j//q, h//q}:
                            loads[block] += cost
        return int(bad), max(loads), total

    # A-left is equivariant under right translations; B-right under left.
    # Check these symmetries against actual component syndromes before use.
    translations = {}
    for side in ("A", "B"):
        translations[side] = [
            tuple(G.idx(G.mul(g, t) if side == "A" else G.mul(t, g))
                  for g in G.elements)
            for t in G.elements]

    def translate(word, permutation):
        out = 0
        while word:
            bit = word & -word
            position = bit.bit_length()-1
            block, local = divmod(position, q)
            out |= 1 << (block*q + permutation[local])
            word ^= bit
        return out

    def sampled(cols, side):
        """Actual kernel witnesses; minimum is only a sampled upper bound."""
        srng = random.Random(164411 + 1009*idx + int(side == "B"))
        order = list(range(width))
        minimum, short = width+1, set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 14 and len(short) < 4096:
                    short.add(word)

        for _ in range(32):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    offer(word)
                    for other in basis:
                        offer(word ^ other)
                    basis.append(word)
            for word in sorted(basis, key=int.bit_count)[:6]:
                for step in range(3):
                    best = word
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best.bit_count():
                            best = candidate
                    if best == word:
                        break
                    word = best
                    offer(word)
            if minimum <= 4:
                break

        for permutation in translations[side]:
            for j in range(5):
                for c in range(q):
                    assert cols[j*q + permutation[c]] == translate(
                        cols[j*q+c], permutation)
        orbits = {min(translate(word, p) for p in translations[side])
                  for word in short}
        loads, burden = [0]*5, 0
        for word in orbits:
            cost = (16-word.bit_count())**2
            burden += cost
            for j in range(5):
                loads[j] += cost*((word >> (j*q)) & mask).bit_count()
        return minimum, burden, max(loads)

    def stress(rows, side):
        """Common-noise classical decoding surrogate, NOT circuit pL."""
        import numpy as np

        nr = len(rows)
        ne = sum(row.bit_count() for row in rows)
        maximum = max(row.bit_count() for row in rows)
        ce = np.full((nr, maximum), ne, dtype=np.int32)
        variable_lists = [[] for _ in range(width)]
        ev, locations = [], []
        matrix = np.zeros((nr, width), dtype=np.uint8)
        for r, row in enumerate(rows):
            position = 0
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                edge = len(ev)
                ce[r, position] = edge
                variable_lists[c].append(edge)
                ev.append(c)
                locations.append(r*maximum + position)
                matrix[r, c] = 1
                position += 1
                row ^= bit
        degree = np.asarray(list(map(len, variable_lists)), dtype=float)
        ve = np.full((width, int(degree.max())), ne, dtype=np.int32)
        for c, edges in enumerate(variable_lists):
            ve[c, :len(edges)] = edges
        ev = np.asarray(ev, dtype=np.int32)
        locations = np.asarray(locations, dtype=np.int32)
        positions = np.arange(maximum)[None, None, :]
        shots = 128
        channel = np.random.default_rng(
            16441991 + 1013*idx + int(side == "B"))
        uniforms = channel.random((shots, width))
        results = []
        for probability in (0.030, 0.050):
            rates = probability*degree/3.0
            magnitude = np.log((1-rates)/rates)
            prior = np.where(uniforms < rates, -magnitude, magnitude)
            messages = prior[:, ev].copy()
            done = np.zeros(shots, dtype=bool)
            wrong = np.zeros(shots, dtype=bool)
            effort = 0
            for iteration in range(1, 33):
                padded = np.concatenate(
                    (messages, np.full((shots, 1), 1.0e6)), axis=1)
                incoming = padded[:, ce]
                absolute = np.abs(incoming)
                where = absolute.argmin(axis=2)
                first = absolute.min(axis=2)
                second = np.partition(absolute, 1, axis=2)[:, :, 1]
                signs = np.where(incoming >= 0, 1.0, -1.0)
                excluded = np.where(
                    positions == where[:, :, None],
                    second[:, :, None], first[:, :, None])
                outgoing = 0.8*signs.prod(axis=2)[:, :, None]*signs*excluded
                check_messages = outgoing.reshape(shots, -1)[:, locations]
                padded_checks = np.concatenate(
                    (check_messages, np.zeros((shots, 1))), axis=1)
                posterior = prior + padded_checks[:, ve].sum(axis=2)
                messages = np.clip(
                    0.25*messages
                    + 0.75*(posterior[:, ev]-check_messages), -30, 30)
                if iteration % 4 == 0:
                    hard = (posterior < 0).astype(np.uint8)
                    resolved = (~done) & ~((hard @ matrix.T) & 1).any(axis=1)
                    wrong[resolved] = hard[resolved].any(axis=1)
                    effort += iteration*int(resolved.sum())
                    done |= resolved
                    if done.all():
                        break
            remaining = int((~done).sum())
            results.append((int((wrong | ~done).sum()),
                            remaining, effort+33*remaining))
        return (results[0][0], results[1][0],
                results[0][1]+results[1][1],
                results[0][2]+results[1][2])

    def portfolio(side):
        sites = [(i, j) for i, row in enumerate(supports[side])
                 for j, w in enumerate(row) if w]
        buckets, seen = {}, set()
        serial = 0

        def remember(grid, rows, cost):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            deficit = len(rows)-gf2_rank(rows)
            if not 2 <= deficit <= 6:
                return
            serial += 1
            bucket = buckets.setdefault(deficit, [])
            bucket.append((cost, serial, clone(grid), list(rows)))
            bucket.sort(key=lambda item: item[:2])
            del bucket[2:]

        def energy(cost):
            return (96*cost[0] + cost[1]/6.0)/q

        for restart in range(6):
            grid = draw(side)
            rows = component(grid, side)
            current = cycles(rows)
            remember(grid, rows, current)
            local_grid, local_rows, local_cost = clone(grid), rows, current
            for step in range(220):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(supports[side][i][j])
                trial_rows = component(grid, side)
                proposed = cycles(trial_rows)
                delta = energy(proposed)-energy(current)
                temperature = 12.0*(0.98**step)+0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    rows, current = trial_rows, proposed
                    if current < local_cost:
                        local_grid = clone(grid)
                        local_rows, local_cost = list(rows), current
                    if step % 32 == 0:
                        remember(grid, rows, current)
                else:
                    grid[i][j] = old
            remember(grid, rows, current)
            remember(local_grid, local_rows, local_cost)

        if not buckets:
            for _ in range(64):
                grid = draw(side)
                rows = component(grid, side)
                remember(grid, rows, cycles(rows))
        finalists = []
        for deficit in sorted(buckets):
            for cost, number, grid, rows in buckets[deficit]:
                cols = columns(rows, width)
                bad, peak, total = boundary(cols)
                distance, burden, word_peak = sampled(cols, side)
                decoder = stress(rows, side)
                key = ((bad, max(0, 12-distance)) + decoder
                       + (peak, total, burden, word_peak, cost[0], cost[1]))
                finalists.append((key, number, grid, rows, deficit))
        finalists.sort(key=lambda item: item[:2])
        return finalists

    def logical4(checks, stabilizers, n):
        """Exact non-stabilizer kernel existence through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, value in enumerate(cols):
            word = 1 << i
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif nontrivial(word ^ old):
                return True
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    def emit(option, short):
        key, A, B, hx, hz, n, k, ar, br = option
        assert n == 544 and 80 <= k <= 94
        assert 125*n < 1000*k < 175*n
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) == max(wx, wz)
        assert max(wx, wz) <= W_MAX
        assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
        actual_a = [
            sum(((hx[(i*5)*q+r] >> (j*5*q)) & mask) << (j*q)
                for j in range(5))
            for i in range(3) for r in range(q)]
        actual_b = [row & ((1 << width)-1) for row in hz[:3*q]]
        assert ar == actual_a and br == actual_b
        for side, grid in (("A", A), ("B", B)):
            for i, row in enumerate(grid):
                for j, cell in enumerate(row):
                    assert len(cell) == supports[side][i][j]
                    assert len(set(cell)) == len(cell) <= CELL_MAX
        return {"idx": idx, "architecture": (pa, pb), "k": k,
                "augmentation_ranks": (1, 1), "screen": key,
                "short_logical_warning": short,
                "cnots": expected_cnots, "cells": {"A": A, "B": B}}

    fallback = None
    for batch in range(3):
        apool, bpool = portfolio("A"), portfolio("B")
        options = []
        for ak, at, A, ar, ad in apool:
            for bk, bt, B, br, bd in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 544 or not 80 <= k <= 94:
                    continue
                paired = tuple(v for a, b in zip(ak, bk)
                               for v in (max(a, b), a+b))
                key = paired + (-k, at, bt)
                options.append((key, A, B, hx, hz, n, k, ar, br))
        options.sort(key=lambda item: item[0])
        for option in options[:8]:
            _, _, _, hx, hz, n, _, _, _ = option
            short = logical4(hx, hz, n) or logical4(hz, hx, n)
            if not short:
                return emit(option, False)
            if fallback is None or option[0] < fallback[0]:
                fallback = option

    # A finite surrogate search must not void an otherwise legal code.
    # A short-logical fallback remains explicit and never outranks a
    # clean finalist. All hard legality and full-rank checks still apply.
    if fallback is not None:
        return emit(fallback, True)
    raise ValueError("no rank-one nonabelian filling retained cell0.150")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened nonabelian cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: Increasing d2s guarantees better memory on the parent C17 bed: slot53 had d2s=14, eight X failures/200000 and s=6.806, while slot10 had d2s=13, one failure per basis and s=7.258.
# REFUTED: Sum-free regular-four C29 directions suffice to saturate the memory score: none of the five measured parent codes reached its 7.63 ceiling, and their Z bases failed 2-10 times per 200000 shots.
# REFUTED: k=70 and d2s=14 certify zero-failure C17 memory: current slot0 had one X failure and s=7.178, while slot1 had no failures and s=7.357; slot4 had two Z failures flipping58 logicals.
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
