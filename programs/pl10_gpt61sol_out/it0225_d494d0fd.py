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
CONCEPT = ("Cyclic Z11, order 11, shapes (4,6)x(4,6): regular degree-four columns, equal four-row components and weight-ten extraction should improve worse-basis circuit fitness beyond the parent's 7.1110 by reducing extraction exposure without weakening sector-one degrees.")
GROUP = ("Preserve the metacyclic construction rule m=1,s=1; adjust only r from 17 to 11. The new factor 52 would give n=884 at order 17; its largest legal lift is order 11, giving n=572. ord_11(2)=10 makes the nontrivial binary cyclotomic factor irreducible. This exact bed is absent from BEDS; related Z17 equal-component beds reached about 7.2, but neither that reading nor irreducibility establishes a circuit advantage.")
SHAPES = ("Equal (4,6) components each have 66 variables and 44 checks. The guaranteed quantum bound is kb=44, rate 1/13=0.07692. Augmentation ranks zero/one and full nontrivial-character ranks predict full k=74,82,92, giving rates 0.12937,0.14336,0.16084 entirely inside cell 0.150; verified component and complete CSS ranks enforce these values. Component dimensions are 25 or 26. Parent failures implicated multiple Acol/Bcol and Arow/Brow blocks, including Bcol3 in slot3 and Bcol2 in slot33, without a persistent Brow hotspot; slot33 also had mixed-sector weight-ten Z witnesses.")
FILLING = ("Twenty unordered canonical support pairs use six connected count profiles: monomial columns and even doubleton-pair columns, with every column total four and every row total six. Cells have 0-2 distinct terms. Extra slots select fresh canonical joint internal geometries, not transformed saved codes. Geometry-preserving voltage annealing retains verified-lift kernel detection through six, capped sampled protection at sixteen, translation-orbit multiplicity and triple boundaries. Broaden full-code opposite-check pair-boundary screening through syndrome weight six. Require component ranks 40/41 and predicted full k=74/82/92. Both CSS check families weigh ten and extraction uses 31680 CNOTs, versus the parent's weight eleven, depth fourteen and 32640 CNOTs. Target its 0-3 failed shots per measured basis in 200000 and collective logical flips while retaining sector-one degree eight. These screens are circuit-pL hypotheses, not distance certificates, ASC hook certificates or measured failure-rate estimates.")
FAMILY_NAME = "r015_m1r11s1_46x46_regular4_gatelean_augmentation_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same metacyclic rule: Z_1 x| Z_11 with trivial action.
    # Factor 52 gives n=884 at order 17 and n=624 at order 12.
    assert q == 11
    m, r, s = 1, 11, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (4, 6)               # (ra, ca)
B_SHAPE = (4, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Four-row regular-column contracts with two augmentation strata."""
    import math
    from itertools import combinations_with_replacement, permutations

    q = G.q
    assert q == 11
    assert A_SHAPE == B_SHAPE == (4, 6)
    height, width = A_SHAPE
    row_permutations = tuple(permutations(range(height)))

    # Odd columns are monomial in every row. Even columns put two
    # terms in each of two rows. Both have total degree four.
    column_types = [(1, 1, 1, 1)]
    column_types.extend(
        tuple(2 if i in (a, b) else 0 for i in range(height))
        for a in range(height) for b in range(a + 1, height)
    )
    profile_keys = set()
    for entries in combinations_with_replacement(column_types, width):
        totals = [sum(column[i] for column in entries)
                  for i in range(height)]
        if totals != [6] * height:
            continue

        # Exclude the disconnected all-even triple-parallel profile.
        # Actual lifted connectivity is not inferred from this test.
        reached, todo = {0}, [0]
        while todo:
            i = todo.pop()
            for column in entries:
                if not column[i]:
                    continue
                for j in range(height):
                    if column[j] and j not in reached:
                        reached.add(j)
                        todo.append(j)
        if len(reached) != height:
            continue

        key = min(
            tuple(sorted(tuple(column[i] for i in p)
                         for column in entries))
            for p in row_permutations
        )
        profile_keys.add(key)

    profiles = [
        [[column[i] for column in key] for i in range(height)]
        for key in sorted(profile_keys)
    ]
    profiles.sort(key=lambda grid: (
        sum(w == 0 for row in grid for w in row),
        tuple(tuple(row) for row in grid),
    ))
    assert len(profiles) == 6
    assert all(w == 1 for row in profiles[0] for w in row)

    # Six canonical profiles give 21 unordered pairs. Omit the entirely
    # monomial pair: every submitted card has a nonempty internal
    # geometry contract, including additional harness phases.
    briefs = [(a, b) for a in range(len(profiles))
              for b in range(a, len(profiles)) if (a, b) != (0, 0)]
    briefs.sort(key=lambda pair: (
        sum(w == 0 for p in pair for row in profiles[p] for w in row),
        pair,
    ))
    assert len(briefs) == 20
    geometry_number, architecture = divmod(idx, len(briefs))
    pa, pb = briefs[architecture]
    sa = [list(row) for row in profiles[pa]]
    sb = [list(row) for row in profiles[pb]]
    supports_by_side = {"A": sa, "B": sb}

    # HX: A-row six plus B-column four.
    # HZ: B-row six plus A-column four.
    assert _grid_weights(sa, sb) == 10
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    augmentation = {}
    for side, support in supports_by_side.items():
        assert list(map(sum, support)) == [6] * height
        assert all(sum(row[j] for row in support) == 4
                   for j in range(width))
        assert all(0 <= w <= min(2, CELL_MAX)
                   for row in support for w in row)
        augmentation[side] = gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support])
        assert augmentation[side] in (0, 1)

    aa, ab = augmentation["A"], augmentation["B"]
    # Semisimple cyclic prediction, enforced later by complete ranks.
    desired_k = ((q - 1) * (width - height) ** 2
                 + (width - aa) * (width - ab)
                 + (height - aa) * (height - ab))
    assert desired_k in (74, 82, 92)
    assert 8 * desired_k >= 572 and 40 * desired_k < 7 * 572

    def normalized(values, multiplier):
        if not values:
            return ()
        image = [(multiplier * value) % q for value in values]
        return min(tuple(sorted((value - origin) % q for value in image))
                   for origin in image)

    def geometry_key(grids):
        """Remove obvious geometry orbits, not arbitrary Tanner equivalence."""
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                relative = [[normalized(cell, u) for cell in row]
                            for row in grid]
                sides.append(min(
                    tuple(sorted(tuple(relative[i][j] for i in p)
                                 for j in range(len(grid[0]))))
                    for p in row_permutations
                ))
            images.append(tuple(sorted(sides)))
        return min(images)

    geometry_rng = random.Random(110463 + 1009 * architecture)

    def relative_grid(support):
        grid = []
        for counts in support:
            used_differences = set()
            row = []
            for weight in counts:
                for attempt in range(128):
                    values = normalized(
                        geometry_rng.sample(range(q), weight), 1)
                    differences = [
                        (a - b) % q for a in values for b in values if a != b
                    ]
                    if len(set(differences)) != len(differences):
                        continue
                    if used_differences.isdisjoint(differences):
                        break
                else:
                    return None
                used_differences.update(differences)
                row.append(values)
            grid.append(row)
        return grid

    # Each extra harness phase gets a genuinely new joint internal
    # geometry contract. No selected matrices are transformed or reused.
    book, seen_geometries = [], set()
    for attempt in range(10000):
        grids = [relative_grid(sa), relative_grid(sb)]
        if any(grid is None for grid in grids):
            continue
        key = geometry_key(grids)
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
        book.append(grids)
        if len(book) > geometry_number:
            break
    if len(book) <= geometry_number:
        raise ValueError("No unused regular-four geometry contract")
    geometry = dict(zip(("A", "B"), book[geometry_number]))

    def draw_grid(side):
        grid = []
        for i, counts in enumerate(supports_by_side[side]):
            row = []
            for j, weight in enumerate(counts):
                origin = G.pick(rng)
                cell = [G.mul(origin, (0, delta))
                        for delta in geometry[side][i][j]]
                assert len(cell) == weight
                assert len({G._key(g) for g in cell}) == weight
                row.append(cell)
            grid.append(row)
        return grid

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        # Use the verified construction with an identity companion.
        # The first sector is precisely the required component matrix.
        identity = [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, identity)
        else:
            _, rows, _ = lifted_matrices(G, identity, grid)
        nc = len(grid[0]) * q
        mask = (1 << nc) - 1
        return [row & mask for row in rows], nc

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
        """Sample actual kernel words and count their translation orbits.

        Fundamental words, pair XORs and bounded greedy descent remain
        upper-bound searches, not protection certificates. Finalists use
        a common permutation stream to reduce screening noise.
        """
        srng = random.Random(917031 + 1009 * idx + int(side == "B"))
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

            # Descend only through kernel-vector additions. Zero is
            # excluded: the objective concerns nonzero component words.
            for word in sorted(basis, key=int.bit_count)[:8]:
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

        mask = (1 << q) - 1
        orbits = set()
        for word in small:
            blocks = [(word >> (j * q)) & mask for j in range(nc // q)]
            representative = None
            for shift in range(q):
                image = 0
                for j, block in enumerate(blocks):
                    rotated = ((block << shift) |
                               (block >> (q - shift))) & mask
                    image |= rotated << (j * q)
                if representative is None or image < representative:
                    representative = image
            orbits.add(representative)

        loads = [0] * (nc // q)
        low14 = 0
        for word in orbits:
            low14 += int(word.bit_count() <= 14)
            for j in range(len(loads)):
                loads[j] += ((word >> (j * q)) & mask).bit_count()
        return distance, low14, len(orbits), max(loads)

    def pair_exposure(hx, hz, n):
        """Full-matrix small boundaries among opposite-check neighbors.

        These pairs are possible correlated-fault opportunities, not
        hooks certified by the ASC schedule. Normalize block concentration
        by its share of all inspected pair opportunities.
        """
        rates, peaks = [], []
        for checks, opposite in ((hx, hz), (hz, hx)):
            cols = columns(checks, n)
            burden = [0] * (n // q)
            opportunities = [0] * (n // q)
            total = count = 0
            for row in opposite:
                neighbors = []
                while row:
                    bit = row & -row
                    neighbors.append(bit.bit_length() - 1)
                    row ^= bit
                for t, i in enumerate(neighbors):
                    for j in neighbors[:t]:
                        count += 1
                        bi, bj = i // q, j // q
                        opportunities[bi] += 1
                        opportunities[bj] += 1
                        boundary = (cols[i] ^ cols[j]).bit_count()
                        if boundary <= 6:
                            cost = (7 - boundary) ** 2
                            total += cost
                            burden[bi] += cost
                            burden[bj] += cost
            rates.append(total / max(1, count))
            peaks.append(max(
                b / max(1, o) for b, o in zip(burden, opportunities)))
        return max(rates), sum(rates), max(peaks)

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

            for step in range(280):
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                shift = G.elements[rng.randrange(1, q)]
                # Preserve all relative gaps, including trinomial
                # geometry, while changing inter-cell voltages.
                grid[i][j] = [G.mul(shift, g) for g in old]
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
            # Verify full rank on the irreducible nontrivial order-11
            # factor and exactly the prescribed augmentation deficit.
            expected_rank = len(rows) - (
                len(supports) - augmentation[side])
            if gf2_rank(rows) != expected_rank:
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found:
            cols = columns(rows, nc)
            assert all(col.bit_count() == 4 for col in cols)
            bad, risk, peak = boundary_scan(cols)
            distance, low14, low, word_peak = kernel_exposure(cols, 64, side)
            # Above sixteen there is no extra sampled-distance reward.
            # Concentration and ambiguity, not an optimistic upper bound,
            # decide between candidates meeting that protection brief.
            key = (
                bad, max(0, 16 - distance),
                low14, word_peak, low, peak, risk,
                stats[1], stats[2],
            )
            finalists.append({
                "grid": grid, "key": key, "d": distance,
                "low14": low14, "low": low, "word_peak": word_peak,
                "bad": bad, "risk": risk, "peak": peak,
                "four": stats[1], "six": stats[2],
            })
        finalists.sort(key=lambda item: item["key"])
        if not finalists:
            raise ValueError("No rank-safe four-row regular-four component")
        return finalists[:4]

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
            if n != 572 or k != desired_k:
                continue
            exposure, exposure_sum, exposure_peak = pair_exposure(hx, hz, n)
            da, db = max(0, 16 - a["d"]), max(0, 16 - b["d"])
            key = (
                max(a["bad"], b["bad"]),
                max(da, db), da + db,
                max(a["low14"], b["low14"]),
                a["low14"] + b["low14"],
                exposure_peak, exposure, exposure_sum,
                max(a["word_peak"], b["word_peak"]),
                max(a["low"], b["low"]), a["low"] + b["low"],
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
        raise ValueError("No finalist retained the 0.150 rate band")
    A, B, hx, hz = winner
    assert css_ok(hx, hz)
    assert all(row.bit_count() == 10 for row in hx + hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 31680
    assert 8 * desired_k >= 572 and 40 * desired_k < 7 * 572
    return {"idx": idx, "architecture": (pa, pb, geometry_number),
            "augmentation": (aa, ab),
            "k": desired_k, "screen": winning_key, "cnots": 31680,
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
# REFUTED: Increasing d2s guarantees better memory on this Z17 bed: parent slot53 had d2s=14, eight X failures/200000 and s=6.806, while slot10 had d2s=13, one failure per basis and s=7.258.
# REFUTED: A two-second d2s=15 excludes lower-weight mixed-sector logicals: measured parent slot33 had d2s=15, but its 30-second search verified four mixed-sector Z-type operators of weight ten.
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
