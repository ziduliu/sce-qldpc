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
CONCEPT = ("For dihedral Z14:Z2(s=13), order 28, shapes (2,4)x(2,4), check-weight-nine/ten supports with independently controlled order-7/order-14 binomials and small-boundary optimization should improve mean worse-basis circuit fitness above 6.5820.")
GROUP = ("Frozen under L1: retain the measured order-28 bed, whose fitness is 6.5820. The nearby cyclic order-29 (2,4)x(2,4) bed reaches about 7.2, motivating refinement of this shape's filling without establishing a group advantage.")
SHAPES = ("Frozen (2,4)x(2,4): n=560, guaranteed k>=112 and rate 0.200; q<=29 is size-legal but the unchanged group fixes q=28. Both components have 56 checks and 112 variables. Parent slot47's X map implicated Acol0-3/Bcol0 at 1.2x, and slot2's Z map also implicated Bcol0; these diffuse maps favor component-wide changes.")
FILLING = ("Twenty support/order architectures cross four low-gate support pairs with five relative-order mixtures. Every check has weight <=10; only one architecture permits a degree-four component column. Primary slots use two reflection-holonomy columns. Verified component lifts guide label annealing, exact kernel-weight<=6 tests, triple-boundary penalties and a coarse sampled protection floor. Full verified matrices screen k in [112,124], CSS and logical weight<=4, retaining a rate-legal fallback if optional logical screening finds no passing pair. This targets correlated failures and decoder difficulty: slot16 had d2s=12 but 42 Z failures and 145 Z LSD resolutions, versus slot21's 8 failures and 28 LSD resolutions. Distance is not the scored objective.")
FAMILY_NAME = "r020_m14r2s13_24x24_lowgate_mixedorder_anneal_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # metacyclic Z_14 x| Z_2: y x y^-1 = x^13 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 14, 2, 13
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize an indexed architecture, without shared mutable state."""
    import math
    from itertools import combinations

    # Counts of columns (1,2), (2,1), (2,2), respectively.
    # Totals are 12, 12 and 13. The degree-four profile only pairs
    # with the balanced, row-weight-six profile.
    profiles = ((2, 2, 0), (3, 1, 0), (2, 1, 1))
    support_pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    order_pairs = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
    design = idx % 20
    mode = (idx // 20) % 3
    architecture, order_design = divmod(design, 5)
    pa, pb = support_pairs[architecture]
    even_counts = order_pairs[order_design]

    def support(p):
        a, b, c = profiles[p]
        columns = [(1, 2)] * a + [(2, 1)] * b + [(2, 2)] * c
        return [[v[i] for v in columns] for i in range(2)]

    sa, sb = support(pa), support(pb)
    assert _grid_weights(sa, sb) <= min(10, W_MAX)
    assert _no_dead_lines(sa, sb)

    palettes = []
    twists = []
    for side, counts in enumerate((sa, sb)):
        doubles = [(i, j) for i in range(2) for j in range(4)
                   if counts[i][j] == 2]
        offset = (order_design + side + 2 * mode) % len(doubles)
        even = {doubles[(offset + t) % len(doubles)]
                for t in range(even_counts[side])}
        palettes.append({
            (i, j): ((2, 4, 6) if (i, j) in even else (1, 3, 5))
            for i, j in doubles
        })
        first = (order_design + side) % 4
        if mode == 1:
            twist = {first}
        else:
            gap = 1 + side if mode == 0 else 2 + side
            twist = {first, (first + gap) % 4}
        twists.append(twist)

    def labels(counts, side):
        first_coset = [rng.randrange(2) for _ in range(4)]
        grid = []
        for i in range(2):
            row = []
            for j in range(4):
                b = first_coset[j] ^ int(i == 1 and j in twists[side])
                a = rng.randrange(14)
                cell = [(a, b)]
                if counts[i][j] == 2:
                    delta = rng.choice(palettes[side][i, j])
                    cell.append(((a + delta) % 14, b))
                row.append(cell)
            grid.append(row)
        return grid

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    q = G.q
    component_mask = (1 << (4 * q)) - 1

    def component_rows(grid, side):
        # A unit partner makes sector one exactly the desired component.
        # Both left and right actions still come from the verified helper.
        unit = [[[G.e]]]
        if side == 0:
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = hx
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            rows = hz
        return [row & component_mask for row in rows]

    def cycle_cost(rows):
        """Exact component 4/6-cycle counts and a rank-deficit penalty."""
        m = len(rows)
        columns = [[] for _ in range(4 * q)]
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
            del pools[side][8:]

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
                i, j = rng.randrange(2), rng.randrange(4)
                old = grid[i][j]
                b = old[0][1]
                a = rng.randrange(14)
                proposed_cell = [(a, b)]
                if len(old) == 2:
                    if rng.random() < 0.65:
                        delta = (old[1][0] - old[0][0]) % 14
                    else:
                        delta = rng.choice(palettes[side][i, j])
                    proposed_cell.append(((a + delta) % 14, b))
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
        n = 4 * q
        best = n
        for _ in range(192):
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
        columns = columns_of(rows, 4 * q)
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
            key = (int(abad or bbad), int(min(da, db) < 11),
                   ac[0] + bc[0],
                   max(abound, bbound), abound + bbound,
                   ac[1] + bc[1], -min(da, db),
                   abs(da - db), ac[2] + bc[2], at, bt)
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    fallback = None
    examined = 0
    for key, a, b in options:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (112 <= k <= 124):
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
    raise ValueError("no low-gate pair has verified rate in cell 0.200")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without drawing new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s>=8 guarantees basis-balanced memory performance; seed slot19 had X failures 100/78000 versus Z 29/200000, with scores 5.634 versus 6.509.
# REFUTED: Equal d2s=12 implies equal circuit reliability: parent slot16 scored 6.389 with 42 Z failures and 145 Z LSD resolutions, while slot21 scored 6.931 with 8 Z failures and 28 Z LSD resolutions.
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
