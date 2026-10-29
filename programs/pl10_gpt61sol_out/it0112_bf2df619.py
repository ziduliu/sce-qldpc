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
CONCEPT = ("For dicyclic Dic10, order 40, shapes (1,2)x(2,6), mixed-coset A cells and order-controlled B cells should raise the 0.300 frontier above 5.8 by reducing collective circuit failures; the measured memory experiment must test this.")
GROUP = ("Jump from the order-28 dihedral parent to Dic10: x^20=1, y^2=x^10, y*x*y^-1=x^-1. This order-40 group has a unique involution. Shape factor 14 permits q<=42, making 40 the largest dicyclic lift order. The related Dic7 equal-component bed reached about 6.7 and the nonabelian order-48 (1,2)x(2,5) bed reached about 6.5; neither reading establishes an advantage for this untried bed.")
SHAPES = ("Target the weaker 0.300 frontier with (1,2)x(2,6): n=560, kb=160, guaranteed rate 2/7=0.285714. A has 40 checks on 80 variables; B has 80 checks on 240 variables. Full verified ranks require 0.275<=k/n<0.325. Parent slot4's X failures were mildly enriched across Acol3/Bcol3 and several Bcol2 blocks; slot2 implicated Acol1/Bcol2 in X. These small-sample maps motivate broad component changes, not a claimed localized cause.")
FILLING = ("Twenty deliberate briefs cross four weighted-support architectures with five binomial relative-order policies: 20, 10, 5, 4, and a spatial mixture. A uses mixed-coset three/four-term cells; B columns have degree two or three, including controlled zero/trinomial alternatives. A row totals are seven/eight, B row totals eight, hence HX weight<=11 and HZ weight<=12. Within-cell involution differences are excluded. Verified component lifts guide annealing, exact kernel-weight<=6 tests, triple-boundary penalties and capped sampled protection floors of ten for A and eight for B. Full matrices screen rate, CSS and logical weight<=4, with a rate-legal fallback. Six-round CNOT counts are 24960 or 26880, an explicit noise-cost tradeoff rather than a promised improvement over the parent's 24192. Parent slot4 had eight X failures flipping 344 logicals, worst #47 in seven shots, despite w=9,d2s=12; this is a correlated-failure objective, not distance fitness.")
FAMILY_NAME = "r030_Dic10_12x26_mixedcoset_orderbook_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic10 requires order 40")
    # Normal form x^a*y^b: x^20=1, y^2=x^10, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(20)]

    def red(g):
        b = int(g[1])
        return ((int(g[0]) + 10 * (b // 2)) % 20, b % 2)

    def idx(g):
        a, b = red(g)
        return a + 20 * b

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 10 * b * d) % 20,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 20, 0) if b == 0 else ((a + 10) % 20, 1)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize one Dic10 support/order brief without shared mutable state."""
    import math
    from itertools import combinations

    design = idx % 20
    mode = (idx // 20) % 3
    architecture, order_design = divmod(design, 5)

    b_profiles = (
        ((1, 1, 2, 2, 1, 1), (2, 2, 1, 1, 1, 1)),
        ((0, 1, 2, 3, 1, 1), (3, 2, 1, 0, 1, 1)),
        ((1, 1, 2, 2, 0, 2), (2, 2, 1, 1, 2, 0)),
    )
    sa = [[3, 4]] if architecture == 3 else [[4, 4]]
    sb = [list(row) for row in b_profiles[
        architecture if architecture < 3 else 0]]

    # Check both sums explicitly. No submitted slot is repaired.
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert wx <= min(11, W_MAX)
    assert wz <= min(12, W_MAX)
    assert _grid_weights(sa, sb) == max(wx, wz)
    assert _no_dead_lines(sa, sb)

    # Rotational differences have orders 20, 10 or 5, respectively.
    # Every outside-coset element has order four in Dic10.
    rotation_steps = (
        (1, 3, 7, 9, 11, 13, 17, 19),
        (2, 6, 14, 18),
        (4, 8, 12, 16),
    )
    central = (10, 0)

    def draw_cell(count, side, i, j):
        if count == 0:
            return []
        base = G.pick(rng)
        if count == 1:
            return [base]
        if count == 2:
            tag = order_design
            if tag == 4:
                tag = (i + j + mode) % 4
            relative = ((rng.choice(rotation_steps[tag]), 0)
                        if tag < 3 else (rng.randrange(20), 1))
            return [base, G.mul(base, relative)]

        # Larger cells contain both cosets. Balanced 2+2 and skew 3+1
        # A policies test quotient structure without assuming its rank.
        if count == 4:
            bs = ([0, 0, 1, 1] if (order_design + mode) % 2 == 0
                  else [0, 0, 0, 1])
        else:
            bs = [0, 0, 1]
        cell = [base]
        for b in bs[1:]:
            while True:
                g = G.mul(base, (rng.randrange(20), b))
                if g in cell:
                    continue
                if any(G.mul(G.inv(h), g) == central for h in cell):
                    continue
                cell.append(g)
                break
        assert len(cell) == count and len(set(cell)) == count
        return cell

    def labels(counts, side):
        return [[draw_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(counts)]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    q = G.q

    def component_rows(grid, side):
        # Project the actual helper-built matrices for each unequal side.
        # A one-cell partner makes sector one the desired component.
        unit = [[[G.e]]]
        if side == 0:
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = hx
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            rows = hz
        width = (A_SHAPE, B_SHAPE)[side][1] * q
        component_mask = (1 << width) - 1
        return [row & component_mask for row in rows]

    def cycle_cost(rows):
        """Exact component 4/6-cycle counts and a rank-deficit penalty."""
        m = len(rows)
        # Every component column is live, so this is its exact width.
        width = max(row.bit_length() for row in rows)
        columns = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1].append(r)
                row ^= bit

        adj = [dict() for _ in range(m)]
        for ns in columns:
            for a, b in combinations(ns, 2):
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1

        seen, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        if len(seen) != m:
            return None

        four = sum(v * (v - 1) // 2
                   for a in range(m) for b, v in adj[a].items() if a < b)
        six = 0
        for a in range(m):
            for b, ab in adj[a].items():
                if b <= a:
                    continue
                for c, bc in adj[b].items():
                    if c > b and c in adj[a]:
                        six += ab * bc * adj[a][c]
        # Remove triangles whose check-pair edges reuse a variable.
        for ns in columns:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        deficit = m - gf2_rank(rows)
        return (four, six + q * deficit, deficit)

    pools = [[], []]

    def energy(cost):
        four, six_with_deficit, deficit = cost
        return (100 * four + six_with_deficit + 4 * q * deficit) / q

    for side, counts in enumerate((sa, sb)):
        seen_fillings = set()

        def offer(cost, serial, grid, rows):
            signature = tuple(tuple(sorted(G._key(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen_fillings:
                return
            seen_fillings.add(signature)
            pools[side].append((cost, serial, clone(grid), list(rows)))
            pools[side].sort(key=lambda item: (item[0], item[1]))
            del pools[side][4:]

        # Optimize voltages inside the architecture, rather than merely
        # increasing the number of independent random draws.
        for restart in range(4):
            for attempt in range(16):
                grid = labels(counts, side)
                rows = component_rows(grid, side)
                current = cycle_cost(rows)
                if current is not None:
                    break
            else:
                continue

            best = current
            best_grid, best_rows = clone(grid), list(rows)
            offer(current, restart * 161, grid, rows)
            for step in range(160):
                i = rng.randrange(len(counts))
                j = rng.randrange(len(counts[0]))
                old = grid[i][j]
                if not old:
                    continue
                if rng.random() < 0.70:
                    # Preserve internal relative orders while changing
                    # inter-cell voltages; this is a search mutation,
                    # not a submitted orbit copy of a saved filling.
                    shift = G.pick(rng)
                    proposed_cell = [G.mul(shift, g) for g in old]
                else:
                    proposed_cell = draw_cell(len(old), side, i, j)
                grid[i][j] = proposed_cell

                proposed_rows = component_rows(grid, side)
                proposed = cycle_cost(proposed_rows)
                if proposed is None:
                    grid[i][j] = old
                    continue
                difference = energy(proposed) - energy(current)
                temperature = 24.0 * (0.975 ** step) + 0.15
                accept = (difference <= 0 or
                          rng.random() < math.exp(
                              -min(700.0, difference / temperature)))
                if not accept:
                    grid[i][j] = old
                    continue

                current, rows = proposed, proposed_rows
                improved = current < best
                if improved:
                    best = current
                    best_grid, best_rows = clone(grid), list(rows)
                if improved or step % 8 == 0:
                    offer(current, restart * 161 + step + 1, grid, rows)
            offer(best, 1000 + restart, best_grid, best_rows)

    if not all(pools):
        raise ValueError("no connected low-gate component candidate")

    def kernel_upper_bound(rows):
        """Sample kernel words using randomized information sets.

        This is only an upper-bound heuristic, not a distance certificate
        and not an estimate of the scored circuit-level failure rate.
        """
        n = max(row.bit_length() for row in rows)
        best = n
        for _ in range(64):
            mat = list(rows)
            order = list(range(n))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                found = next((r for r in range(rank, len(mat))
                              if mat[r] & bit), None)
                if found is None:
                    continue
                mat[rank], mat[found] = mat[found], mat[rank]
                pivot_row = mat[rank]
                for r in range(len(mat)):
                    if r != rank and mat[r] & bit:
                        mat[r] ^= pivot_row
                pivots.append(c)
                rank += 1
                if rank == len(mat):
                    break

            occupied = set(pivots)
            basis = []
            for c in order:
                if c in occupied:
                    continue
                bit = 1 << c
                word = bit
                for r, p in enumerate(pivots):
                    if mat[r] & bit:
                        word ^= 1 << p
                basis.append(word)
                best = min(best, word.bit_count())

            for i, x in enumerate(basis):
                for j in range(i):
                    best = min(best, (x ^ basis[j]).bit_count())
            if best <= 8:
                break
        return best

    def columns_of(rows, ncols):
        columns = [0] * ncols
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return columns

    def small_kernel_and_boundary(rows):
        # Equal syndromes of distinct subsets of size <=3 exactly
        # detect a nonzero component kernel word of weight <=6.
        width = max(row.bit_length() for row in rows)
        columns = columns_of(rows, width)
        seen = {0}
        for value in columns:
            if value in seen:
                return True, 0
            seen.add(value)
        for i, value in enumerate(columns):
            for j in range(i):
                syndrome = value ^ columns[j]
                if syndrome in seen:
                    return True, 0
                seen.add(syndrome)

        boundary = 0
        for i, value in enumerate(columns):
            for j in range(i):
                pair = value ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    if syndrome in seen:
                        return True, boundary
                    seen.add(syndrome)
                    weight = syndrome.bit_count()
                    if weight < 5:
                        boundary += (5 - weight) ** 2
        return False, boundary

    def has_small_logical(hx, hz, n):
        # Exact logical-weight<=4 test on the full verified matrices.
        # One representative per syndrome suffices modulo stabilizers.
        for checks, stabilizers in ((hx, hz), (hz, hx)):
            pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    if p in pivots:
                        row ^= pivots[p]
                    else:
                        pivots[p] = row
                        break

            def trivial(word):
                while word:
                    p = word.bit_length() - 1
                    if p not in pivots:
                        return False
                    word ^= pivots[p]
                return True

            columns = columns_of(checks, n)
            bits = [1 << i for i in range(n)]
            seen = {0: 0}
            for i, syndrome in enumerate(columns):
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = bits[i]
                elif not trivial(previous ^ bits[i]):
                    return True
            for i, syndrome_i in enumerate(columns):
                for j in range(i):
                    syndrome = syndrome_i ^ columns[j]
                    word = bits[i] ^ bits[j]
                    previous = seen.get(syndrome)
                    if previous is None:
                        seen[syndrome] = word
                    elif not trivial(previous ^ word):
                        return True
        return False

    finalists = []
    for pool in pools:
        entries = []
        for cost, trial, grid, rows in pool:
            bad, boundary = small_kernel_and_boundary(rows)
            d = 6 if bad else kernel_upper_bound(rows)
            entries.append((cost, d, bad, boundary, trial, grid))
        finalists.append(entries)

    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, ad, abad, abound, at, ag = a
            bc, bd, bbad, bbound, bt, bg = b
            da, db = min(ad, 12), min(bd, 12)
            # A coarse protection floor comes first. Above that floor,
            # favor graph expansion and short-cycle burden, not an
            # indefinitely increasing sampled distance upper bound.
            # Unequal components have different protection floors.
            # Normalize boundary exposure by their numbers of columns;
            # retain only capped sampled-distance preferences.
            an = abound / A_SHAPE[1]
            bn = bbound / B_SHAPE[1]
            key = (int(abad or bbad), int(ad < 10 or bd < 8),
                   ac[0] + bc[0],
                   max(an, bn), an + bn,
                   ac[1] + bc[1], -min(da, db),
                   abs(da - db), ac[2] + bc[2], at, bt)
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    fallback = None
    examined = 0
    for key, a, b in options:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 560 and 0.275 <= k / n < 0.325):
            continue
        if not css_ok(hx, hz):
            continue
        candidate = {"idx": idx, "design": design, "mode": mode,
                     "cells": {"A": a, "B": b}, "k": k,
                     "screen": key, "small_logical_safe": False}
        if fallback is None:
            fallback = candidate
        if not has_small_logical(hx, hz, n):
            candidate["small_logical_safe"] = True
            return candidate
        examined += 1
        if examined >= 12:
            break

    if fallback is not None:
        return fallback
    raise ValueError("no Dic10 pair has verified rate in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without drawing new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s>=8 guarantees basis-balanced memory performance; seed slot19 had X failures 100/78000 versus Z 29/200000, with scores 5.634 versus 6.509.
# REFUTED: Equal d2s=12 implies equal circuit reliability: parent slot16 scored 6.389 with 42 Z failures and 145 Z LSD resolutions, while slot21 scored 6.931 with 8 Z failures and 28 Z LSD resolutions.
# REFUTED: Check weight nine and d2s=12 ensure balanced zero-failure memory: current slot4 had one Z failure but eight X failures in 200000 shots, flipping 344 X-basis logical qubits in total.
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
