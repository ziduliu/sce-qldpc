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
CONCEPT = ("For A4 x C3, order 36, shapes (2,3)x(2,3), equal degree-four components and weight-ten extraction should improve worse-basis circuit fitness above the parent's 7.2247 in rate cell 0.150.")
GROUP = ("Keep the exact A4 x C3 construction, order 36 and encoding. This group already produced fitness 7.2247 with rare measured failures. The new factor thirteen gives n=468 and allows q<=46, so the existing order remains legal without changing the group. The experiment isolates shapes and filling rather than claiming a new group advantage.")
SHAPES = ("Equal (2,3) components each have 108 variables and 72 checks, hence dimension at least 36. n=468 and kb=36 give guaranteed rate 1/13, below the target band; only verified complete-code ranks admitting 59<=k<=81 place a filling in cell 0.150. B changes from rate-bound one-half to one-third. Parent measured X bases had zero failures, while Z bases had zero, one, one, two and four failures in 200000 shots. Enrichment involved different Acol/Bcol data and Acol/Brow check blocks, without a persistent hotspot; preserve two-row A protection while reducing extraction exposure and balancing the components.")
FILLING = ("Twenty canonical pairs of relative-order grids prescribe doubleton orders three/six; additional harness slots use further canonical pairs, never transformations of selected fillings. Every component row has six terms and every column four, so both CSS check weights are ten and six-round extraction uses 25920 CNOTs, versus the parent's measured 30672-32832. Optimize fresh voltages using only helper-derived actions, retaining component rank strata before full-k selection. Exact component exclusion through six and quantum logical exclusion through four accompany capped sampled protection at sixteen, pair-XOR/greedy kernel witnesses and block-concentration screening. This targets the parent's collective Z flips, including 165 flips on 80 of 88 logicals in slot32, while reducing gate exposure. These structural screens are hypotheses about circuit pL, not distance or reliability certificates.")
FAMILY_NAME = "r015_A4xC3_23x23_regular4_orderpair_gatelean_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C3 requires order 36")
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(3)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 3,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + ((g[4] + h[4]) % 3,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % 3,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _designed_cell(rng, G, count, target_order, mode, orders):
    """Avoid order-two differences; prescribe binomial cycle lengths."""
    for _ in range(512):
        cell = rng.sample(G.elements, count)
        # Distinct projected permutations prevent pure-central pairs.
        if len({tuple(g[:4]) for g in cell}) != count:
            continue
        relative = [
            orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))]
            for a in range(count) for b in range(a + 1, count)
        ]
        if min(relative) < 3:
            continue
        if count == 2:
            if relative[0] != target_order:
                continue
        else:
            central_span = len({g[4] for g in cell})
            wanted = 3 if mode == 0 else 2
            if central_span != wanted:
                continue
        return cell
    raise ValueError("could not realize cell design")


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Twenty multi-row contracts, screened on evaluator-convention lifts."""
    import math
    from itertools import combinations

    from itertools import product, permutations

    q, e = G.q, G.e
    assert q == 36
    assert A_SHAPE == B_SHAPE == (2, 3)
    design, batch = idx, idx // 20
    sa = [[2, 2, 2], [2, 2, 2]]
    sb = [[2, 2, 2], [2, 2, 2]]
    counts = (sa, sb)
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert wx == wz == _grid_weights(sa, sb) == 10
    assert 10 <= W_MAX
    assert _no_dead_lines(sa, sb)

    orders = _orders(G)
    six = [g for g in G.elements if orders[G.idx(g)] == 6]
    three = [g for g in G.elements
             if orders[G.idx(g)] == 3 and tuple(g[:4]) != tuple(e[:4])]
    assert six and three
    palettes = {3: three, 6: six}

    # Canonicalize the order contracts, not completed code matrices.
    # Orders three and six are intrinsic labels and are never exchanged
    # by a group automorphism. Remove obvious row/column permutations
    # and the exchange of the now equal-shaped components.
    column_permutations = tuple(permutations(range(3)))

    def canonical_orders(values):
        return min(
            tuple(values[3 * (i ^ swap) + j]
                  for i in range(2) for j in p)
            for swap in (0, 1) for p in column_permutations
        )

    profiles = sorted({
        canonical_orders(values)
        for values in product((3, 6), repeat=6)
    }, key=lambda values: (
        values.count(3),
        abs(values[:3].count(3) - values[3:].count(3)),
        values,
    ))
    briefs = [(a, b) for a in range(len(profiles))
              for b in range(a, len(profiles))]
    briefs.sort(key=lambda pair: (
        sum(profiles[p].count(3) for p in pair),
        abs(profiles[pair[0]].count(3)
            - profiles[pair[1]].count(3)),
        pair,
    ))
    assert len(briefs) >= 20
    if idx >= len(briefs):
        raise ValueError("No unused canonical order-pair contract")
    pa, pb = briefs[idx]
    a_orders = tuple(tuple(profiles[pa][3*i:3*i+3])
                     for i in range(2))
    b_orders = tuple(tuple(profiles[pb][3*i:3*i+3])
                     for i in range(2))
    order_grids = (a_orders, b_orders)

    def draw_cell(side, i, j):
        base = G.pick(rng)
        delta = rng.choice(palettes[order_grids[side][i][j]])
        cell = [base, G.mul(base, delta)]
        assert len({G._key(g) for g in cell}) == 2
        return cell

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    # These cached blocks are extracted from the verified helper.
    # No separately implemented group action is used for screening.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G.idx(g)] = tuple(row & mask for row in hx)
        actions[1][G.idx(g)] = tuple(row & mask for row in hz)

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
        return rows

    def geometry(rows, width):
        """Exact Tanner four/six cycles and check-graph connectivity."""
        cols = columns(rows, width)
        adjacency = [dict() for _ in rows]
        neighborhoods = []
        for col in cols:
            ns = []
            while col:
                bit = col & -col
                ns.append(bit.bit_length() - 1)
                col ^= bit
            neighborhoods.append(ns)
            for a, b in combinations(ns, 2):
                adjacency[a][b] = adjacency[a].get(b, 0) + 1
                adjacency[b][a] = adjacency[b].get(a, 0) + 1
        reached, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adjacency[a]:
                if b not in reached:
                    reached.add(b)
                    todo.append(b)
        four = sum(v * (v - 1) // 2
                   for a, neighbors in enumerate(adjacency)
                   for b, v in neighbors.items() if a < b)
        six_cycles = 0
        for a, neighbors in enumerate(adjacency):
            for b, ab in neighbors.items():
                if b <= a:
                    continue
                for c, bc in adjacency[b].items():
                    if c > b and c in neighbors:
                        six_cycles += ab * bc * neighbors[c]
        # Correct triangles that reuse one variable on two/three edges.
        for ns in neighborhoods:
            for a, b, c in combinations(ns, 3):
                six_cycles -= (adjacency[a][b] + adjacency[a][c]
                               + adjacency[b][c] - 2)
        duplicates = width - len(set(cols)) + int(0 in cols)
        return (len(rows) - len(reached), duplicates,
                four, six_cycles), cols

    def short4(cols):
        if 0 in cols or len(set(cols)) != len(cols):
            return (1, 10**9, 10**9)
        singles, pairs = set(cols), {}
        three_words = four_words = 0
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                three_words += value in singles
                four_words += pairs.get(value, 0)
                pairs[value] = pairs.get(value, 0) + 1
        assert three_words % 3 == 0 and four_words % 3 == 0
        return (0, three_words // 3, four_words // 3)

    def short6_boundary(cols):
        """Exact <=6 existence; triple boundary is an exposure surrogate."""
        seen = {0}
        for x in cols:
            if x in seen:
                return True, 10**9
            seen.add(x)
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                if value in seen:
                    return True, 10**9
                seen.add(value)
        boundary = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True, 10**9
                    seen.add(value)
                    weight = value.bit_count()
                    if weight <= 4:
                        boundary += (5 - weight) ** 2
        return False, boundary / len(cols)

    def sampled_protection(cols, side):
        """Actual kernel witnesses; returned minimum is only an upper bound.

        Compare finalists with a common permutation stream. Pair XORs
        and greedy kernel additions expose collective ambiguities missed
        by inspecting fundamental words alone.
        """
        srng = random.Random(836729 + 1009 * idx + side)
        order = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()

        def offer(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 16 and len(short) < 8192:
                short.add(word)

        for trial in range(64):
            srng.shuffle(order)
            pivots, basis = {}, []
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

            for word in sorted(basis, key=int.bit_count)[:8]:
                for sweep in range(3):
                    best, weight = word, word.bit_count()
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            best, weight = candidate, candidate.bit_count()
                    if best == word:
                        break
                    word = best
                    offer(word)
            if minimum <= 6:
                break

        block_mask = (1 << q) - 1
        loads = [0] * (len(cols) // q)
        low12 = 0
        for word in short:
            low12 += int(word.bit_count() <= 12)
            for j in range(len(loads)):
                loads[j] += ((word >> (j * q)) & block_mask).bit_count()
        return minimum, low12, len(short), max(loads, default=0)

    def optimize(side, extra):
        width = len(counts[side][0]) * q
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, count in enumerate(row) if count]
        strata, seen = {}, set()
        serial = 0

        def offer(grid, rows, cost, cols):
            nonlocal serial
            if cost[0] or cost[1]:
                return
            sig = signature(grid)
            if sig in seen:
                return
            seen.add(sig)
            low = short4(cols)
            deficit = len(rows) - gf2_rank(rows)
            serial += 1
            key = low + (cost[2], cost[3])
            entry = (key, serial, clone(grid), list(rows), list(cols), deficit)
            bucket = strata.setdefault(deficit, [])
            bucket.append(entry)
            bucket.sort(key=lambda item: item[:2])
            del bucket[2:]

        def energy(cost):
            return (5000 * cost[0] + 2000 * cost[1]
                    + 160 * cost[2] + cost[3]) / q

        for restart in range(3):
            grid = [[draw_cell(side, i, j) for j in range(len(row))]
                    for i, row in enumerate(counts[side])]
            rows = component(grid, side)
            cost, cols = geometry(rows, width)
            offer(grid, rows, cost, cols)
            for step in range(220 + extra):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows = component(grid, side)
                proposed, proposed_cols = geometry(proposed_rows, width)
                delta = energy(proposed) - energy(cost)
                temperature = 15.0 * (0.978 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, rows, cost, cols = (
                        grid, proposed_rows, proposed, proposed_cols)
                    if step % 10 == 0 or delta < 0:
                        offer(grid, rows, cost, cols)
                else:
                    grid[i][j] = old
            offer(grid, rows, cost, cols)

        # Retain rank diversity because only complete CSS ranks decide
        # which cell a pair occupies. Component rank is not substituted.
        candidates = [bucket[0] for bucket in strata.values()]
        candidates.sort(key=lambda item: item[:2])
        candidates = candidates[:5]
        if len(candidates) < 5:
            remaining = [entry for bucket in strata.values()
                         for entry in bucket[1:]]
            remaining.sort(key=lambda item: item[:2])
            candidates.extend(remaining[:5 - len(candidates)])

        refined = []
        for coarse, serial, grid, rows, cols, deficit in candidates:
            bad, boundary = short6_boundary(cols)
            d, low12, low16, peak = sampled_protection(cols, side)
            # No reward for optimistic sampled distances above sixteen.
            # Once that brief is met, reduce short-word multiplicity and
            # block concentration before further cycle minimization.
            key = (coarse[0], coarse[1], coarse[2], int(bad),
                   max(0, 16 - d), low12, peak, low16,
                   boundary, coarse[3], coarse[4])
            refined.append((key, serial, grid, rows, deficit))
        refined.sort(key=lambda item: item[:2])
        return refined

    def small_logical(checks, stabilizers, n):
        """Exact existence of a quantum logical of weight at most four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    for attempt in range(5):
        apool = optimize(0, 40 * attempt)
        bpool = optimize(1, 40 * attempt)
        options = []
        for ak, at, A, ar, ad in apool:
            for bk, bt, B, br, bd in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 468 or not 59 <= k <= 81:
                    continue
                # Protection rewards saturate. After the floor, favor
                # balanced neighborhoods and extraction burden, not
                # arbitrarily large sampled distance upper bounds.
                paired = tuple(v for t in range(len(ak))
                               for v in (max(ak[t], bk[t]), ak[t] + bk[t]))
                edges = sum(row.bit_count() for row in hx + hz)
                key = paired + (edges, -k, at, bt)
                options.append((key, A, B, hx, hz, k, ar, br))
        options.sort(key=lambda item: item[0])
        for key, A, B, hx, hz, k, ar, br in options:
            if small_logical(hx, hz, 468) or small_logical(hz, hx, 468):
                continue
            assert css_ok(hx, hz)
            assert all(row.bit_count() == 10 for row in hx + hz)
            assert 10 <= W_MAX
            assert 6 * sum(row.bit_count() for row in hx + hz) == 25920

            # Confirm both cached conventions against the complete code.
            actual_a = []
            for i in range(A_SHAPE[0]):
                for t in range(q):
                    row = hx[(i * B_SHAPE[1]) * q + t]
                    actual_a.append(sum(
                        ((row >> (j * B_SHAPE[1] * q)) & mask) << (j * q)
                        for j in range(A_SHAPE[1])))
            actual_b = [row & ((1 << (B_SHAPE[1] * q)) - 1)
                        for row in hz[:B_SHAPE[0] * q]]
            assert actual_a == ar and actual_b == br
            assert all(c.bit_count() == 4
                       for rows in (ar, br)
                       for c in columns(rows, 3 * q))
            assert 8 * k >= 468 and 40 * k < 7 * 468
            return {"idx": idx, "design": design, "batch": batch,
                    "k": k, "screen": key, "cnots": 25920,
                    "contract": (pa, pb, a_orders, b_orders),
                    "cells": {"A": A, "B": B}}
    raise ValueError("no short-logical-safe filling in rate cell 0.150")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory differences and minimizing four-cycles suffices for reliable r045 memory: fitness was 3.2178, measured Z failures were 112..137/250 shots, and every measured code had weight-three X-type logical witnesses.
# REFUTED: One-row syndrome-budget and exact short-logical selection suffices for reliable A4 x C3 memory: fitness was 3.3437; measured codes retained weight-three Z witnesses and failed 113..186 X shots in 250..500 despite predominantly stage-1 settlement.
# REFUTED: Equal d2s=14 certifies equal circuit reliability on A4 x C3: current slots32/46 failed four/one Z shots out of 200000 and scored 7.013/7.283, while both X bases had zero failures.
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
