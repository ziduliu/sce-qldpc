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
CONCEPT = ("For nonabelian Frobenius C13:C3, order 39, shapes (1,2)x(2,6), noncommuting small-cell supports with screened collective syndrome ambiguity should raise cell 0.300's mean worse-basis circuit fitness above its approximately 5.8 frontier record.")
GROUP = ("Jump from cyclic C29 to C13:C3 with action x->x^3. Since 3^3=1 modulo 13, this is a valid order-39 semidirect product, not a relabeling of C29. It has no involutions. The existing order-39 (2,3)x(2,3) bed reached approximately 7.1, motivating this untried shape bed without asserting a group-only advantage.")
SHAPES = ("Target cell 0.300 with (1,2)x(2,6): n=546, kb=156 and guaranteed rate 2/7. A has 39 checks on 78 variables; B has 78 checks on 234 variables. Require verified 156<=k<=177, entirely inside the target cell. Parent failures implicated changing Acol/Bcol data blocks and Acol/Brow checks, with only 1-5 failures per measured basis; these maps do not justify a unique hotspot repair.")
FILLING = ("Twenty contracts cross five support pairs with four rotation-ratio and quotient-order policies. A uses noncommuting trinomials or quadrinomials; B uses degree-three columns or controlled degree-two/three/four mixtures. Both check-weight sums are <=12; six-round CNOT counts are 23868, 24336 or 26208. Optimize singleton actions extracted from verified lifts, then prefer exact absence of component kernel words <=6, capped sampled protection and small triple boundaries. Complete verified matrices enforce rate and CSS and prefer no quantum logical <=4, with a rate-legal fallback. Parent slot1 had one failure per basis and s=7.398, whereas slot7 had five Z failures and s=7.077 despite equal d2s=12: collective ambiguity, not upper-bound distance alone, is the hypothesis.")
FAMILY_NAME = "r030_Frob39_12x26_noncommuting_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 semidirect C3 requires order 39")
    # x^13=y^3=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    els = [(a, b) for b in range(3) for a in range(13)]

    def red(g):
        return (int(g[0]) % 13, int(g[1]) % 3)

    def idx(g):
        a, b = red(g)
        return a + 13*b

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(3, b, 13)*c) % 13, (b + d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-pow(3, bi, 13)*a) % 13, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize an indexed count/order contract; all search state is local."""
    from itertools import combinations

    q = G.q
    assert q == 39
    design, batch = idx % 20, idx // 20
    architecture, card = divmod(design, 4)
    a_profiles = ((3, 3), (3, 3), (3, 4), (4, 4), (4, 4))
    b_profiles = (
        ((1, 1, 1, 2, 2, 2), (2, 2, 2, 1, 1, 1)),
        ((0, 1, 1, 2, 2, 3), (3, 2, 2, 1, 1, 0)),
        ((1, 1, 1, 1, 2, 2), (1, 1, 2, 2, 1, 1)),
        ((1, 1, 1, 1, 2, 2), (1, 1, 2, 2, 1, 1)),
        ((1, 1, 1, 1, 2, 2), (1, 1, 1, 2, 1, 2)),
    )
    sa = [list(a_profiles[architecture])]
    sb = [list(row) for row in b_profiles[architecture]]
    counts = (sa, sb)
    shapes = (A_SHAPE, B_SHAPE)
    widths = tuple(shape[1] * q for shape in shapes)
    mask = (1 << q) - 1

    # Check BOTH budgets explicitly.
    wx = sum(sa[0]) + max(sum(row[j] for row in sb)
                         for j in range(B_SHAPE[1]))
    wz = max(map(sum, sb)) + max(sa[0])
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    # Distinct rotation-ratio/order briefs, not transformations of a
    # previously selected filling. Extra batches alter these briefs.
    ratios = (1, 2, 3, 5)
    ratio = ratios[card] * (1 + 2*batch) % 13
    if not ratio:
        ratio = ratios[card]
    wanted = (architecture, card, batch, ratio)

    def translate(cell):
        origin = G.pick(rng)
        return [G.mul(origin, g) for g in cell]

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if not weight:
            return []
        if weight == 1:
            return [G.pick(rng)]
        if side == 0:
            d = 1 if j == 0 else ratio
            b = 1 if j == 0 or card % 2 == 0 else 2
            t = rng.randrange(13)
            cell = [G.e, (d, 0), (t, b)]
            if weight == 4:
                if card in (0, 3):
                    u = rng.choice([u for u in range(13) if u != t])
                    cell.append((u, b))
                elif card == 1:
                    cell.append((rng.randrange(13), 3 - b))
                else:
                    u = rng.choice([u for u in range(13)
                                    if u not in (0, d)])
                    cell.append((u, 0))
            return translate(cell)

        if weight == 2:
            if card == 0:
                rotation = True
            elif card == 1:
                rotation = False
            elif card == 2:
                rotation = (i + j + batch) % 2 == 0
            else:
                rotation = (j + batch) % 3 == 0
            relative = ((rng.randrange(1, 13), 0) if rotation else
                        (rng.randrange(13), rng.choice((1, 2))))
            return translate([G.e, relative])

        # B trinomials contain a rotation and a complementary element.
        d = rng.randrange(1, 13)
        b = 1 + (i + j + card + batch) % 2
        return translate([G.e, (d, 0), (rng.randrange(13), b)])

    def labels(s, side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(s)]

    # Cache ONLY singleton blocks extracted from the verified helper.
    # Subsequent XOR composition is exact linearity of those blocks.
    actions = ({}, {})
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions[0][G.idx(g)] = tuple(row & mask for row in hx)
        actions[1][G.idx(g)] = tuple(row & mask for row in hz)

    def projected(side, grid):
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

    def components(hx, hz):
        # Independently verify cached projections against the FINAL lift.
        ra, ca = A_SHAPE
        rb, cb = B_SHAPE
        ar = []
        for i in range(ra):
            for r in range(q):
                source = hx[(i * cb) * q + r]
                compact = 0
                for j in range(ca):
                    compact |= ((source >> (j * cb * q)) & mask) << (j * q)
                ar.append(compact)
        bmask = (1 << widths[1]) - 1
        br = [row & bmask for row in hz[:rb * q]]
        return ar, br

    def cycle_cost(rows):
        """Exact component 4/6-cycle counts and a rank-deficit penalty."""
        m = len(rows)
        columns = [[] for _ in range(max(row.bit_length() for row in rows))]
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
    for side, support in enumerate(counts):
        seen = set()
        for trial in range(112):
            grid = labels(support, side)
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                continue
            seen.add(signature)
            rows = projected(side, grid)
            cost = cycle_cost(rows)
            if cost is not None:
                pools[side].append((cost, trial, grid, rows))
                pools[side].sort(key=lambda item: (item[0], item[1]))
                del pools[side][4:]

    if not all(pools):
        raise ValueError("no connected component in count/order contract")

    def kernel_upper_bound(rows, width, trials):
        """Sample kernel words using randomized information sets.

        This is only an upper-bound heuristic, not a distance certificate
        and not an estimate of the scored circuit-level failure rate.
        """
        n = width
        best = n
        for _ in range(trials):
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
            if best <= 6:
                break
        return best

    def small_kernel_word(rows, width):
        """Exact kernel exclusion through six and triple-boundary exposure.

        Boundary exposure is a structural surrogate, not measured pL.
        Normalize totals because the two components have unequal lengths.
        """
        columns = [0] * width
        for r, row in enumerate(rows):
            check_bit = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= check_bit
                row ^= bit
        seen = {0}
        for value in columns:
            if value in seen:
                return True, 10**9, 10**9
            seen.add(value)
        for i, x in enumerate(columns):
            for j in range(i):
                value = x ^ columns[j]
                if value in seen:
                    return True, 10**9, 10**9
                seen.add(value)

        exposure = 0
        loads = [0] * (width // q)
        for i, x in enumerate(columns):
            for j in range(i):
                pair = x ^ columns[j]
                for h in range(j):
                    value = pair ^ columns[h]
                    if value in seen:
                        return True, 10**9, 10**9
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 5:
                        penalty = (5 - weight) ** 2
                        exposure += penalty
                        for block in {i // q, j // q, h // q}:
                            loads[block] += penalty
        return False, exposure / width, max(loads) / q

    finalists = []
    for side, pool in enumerate(pools):
        entries = []
        nr, nc = shapes[side]
        sites = [(i, j) for i in range(nr) for j in range(nc)
                 if counts[side][i][j]]
        for cost, trial, grid, rows in pool:
            # Change voltages or redraw within the indexed order contract.
            # Do not use the parent's additive cyclic label operation.
            for step in range(144):
                proposal = [[list(cell) for cell in row] for row in grid]
                i, j = rng.choice(sites)
                if rng.random() < 0.55:
                    shift = G.pick(rng)
                    proposal[i][j] = [G.mul(shift, g)
                                      for g in proposal[i][j]]
                else:
                    proposal[i][j] = draw_cell(side, i, j)
                proposed_rows = projected(side, proposal)
                proposed_cost = cycle_cost(proposed_rows)
                if proposed_cost is not None and proposed_cost <= cost:
                    grid, rows, cost = proposal, proposed_rows, proposed_cost
            bad, boundary, peak = small_kernel_word(rows, widths[side])
            d = (6 if bad else kernel_upper_bound(
                rows, widths[side], 48 if side == 0 else 24))
            entries.append((cost, d, int(bad), boundary, peak, trial, grid))
        finalists.append(entries)

    # Unequal-rate components receive separate coarse protection floors.
    # Above those floors, prioritize ambiguity and extraction structure,
    # not an indefinitely increasing sampled distance upper bound.
    a_floor = 10 if max(sa[0]) == 3 else 12
    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, ad, abad, abound, apeak, at, ag = a
            bc, bd, bbad, bbound, bpeak, bt, bg = b
            da, db = min(ad, 12), min(bd, 10)
            key = (abad + bbad,
                   int(ad < a_floor) + int(bd < 8),
                   ac[0] + bc[0],
                   max(apeak, bpeak), abound + bbound,
                   -(da + db),
                   ac[1] + bc[1], ac[2] + bc[2], at, bt)
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    def short_logical(checks, stabilizers, n):
        """Exact existence of a non-stabilizer logical of weight <=4."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        # Linear quotient representatives modulo the stabilizer span.
        residues = [1 << c for c in range(n)]
        for p in sorted(pivots):
            tail = pivots[p] ^ (1 << p)
            value = 0
            while tail:
                bit = tail & -tail
                value ^= residues[bit.bit_length() - 1]
                tail ^= bit
            residues[p] = value

        columns = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        seen = {0: 0}
        for i, syndrome in enumerate(columns):
            value = residues[i]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
        for i, x in enumerate(columns):
            for j in range(i):
                syndrome = x ^ columns[j]
                value = residues[i] ^ residues[j]
                old = seen.get(syndrome)
                if old is not None and old != value:
                    return True
                seen[syndrome] = value
        return False

    fallback = None
    examined = 0
    for key, a, b in options:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 546 and 156 <= k <= 177):
            continue
        if not css_ok(hx, hz):
            continue
        ar, br = components(hx, hz)
        assert ar == projected(0, a) and br == projected(1, b)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        candidate = {
            "idx": idx, "cells": {"A": a, "B": b}, "k": k,
            "architecture": architecture, "geometry": wanted,
            "screen": key, "small_kernel_safe": key[0] == 0,
            "small_logical_safe": False}
        if fallback is None:
            fallback = candidate
        if (not short_logical(hx, hz, n)
                and not short_logical(hz, hx, n)):
            candidate["small_logical_safe"] = True
            return candidate
        examined += 1
        if examined >= 8:
            break
    if fallback is not None:
        return fallback
    raise ValueError("no screened Frobenius pair remains in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without drawing new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s>=8 guarantees basis-balanced memory performance; seed slot19 had X failures 100/78000 versus Z 29/200000, with scores 5.634 versus 6.509.
# REFUTED: Equal d2s and larger k guarantee no worse circuit score: current-parent slot16 had d2s=12,k=116,s=6.389, versus slot21 with d2s=12,k=112,s=6.931.
# REFUTED: Canonical internal-step signatures alone ensure distinct Tanner codes: the C29 parent retained only 25 distinct codes from 60 slots, including slot2 marked iso_dup_of_1.
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
