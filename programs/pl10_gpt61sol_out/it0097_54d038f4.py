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
CONCEPT = ("For metacyclic Z12:Z2(s=7), order 24, shapes (2,4)x(2,5), degree-three/four component columns and lower extraction load should improve mean worse-basis circuit fitness above the parent's 6.2461.")
GROUP = ("Keep the metacyclic construction rule and action 7. The new factor 24 requires reducing the lift: m=12,r=2 gives q=24 and 7^2=1 mod 12. This group/shape bed is untried. Related low-load two-row beds score 7.1168 and 7.1770, motivating the geometry without establishing an advantage for this group.")
SHAPES = ("Target cell 0.250 with (2,4)x(2,5): n=576, k>=144 and guaranteed rate 0.250. Components have 96 variables/48 checks and 120 variables/48 checks. Parent witnesses occupied sector 1 and its failure maps were diffuse across Acol/Bcol blocks, occasionally enriched 1.2-1.4x; replace the single-row component rather than patch one purported hotspot. Verified final ranks retain every slot in [0.225,0.275).")
FILLING = ("Twenty support/order architectures cross four degree-three/four support pairs with five binomial-order briefs; extra batches vary reflection holonomy. Same-coset binomials avoid short mixed-coset relative elements. Both check-weight budgets are <=12, with 17280-17856 six-round CNOTs versus the parent's measured 25920-29952. Anneal verified component voltages, prefer exact absence of kernel words <=6, penalize small triple boundaries, and use capped sampled kernel upper bounds only as secondary heuristics. Full verified matrices gate rate, CSS and optionally logical weight<=4. Parent slot42 had 91 X versus 11 Z failures despite d2s=10; balanced component protection and reduced exposure, not distance alone, are the hypotheses.")
FAMILY_NAME = "r025_m12r2s7_24x25_lowload_coset_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Same metacyclic construction and action; 7^2 = 1 mod 12.
    # The new factor 24 requires q=24 to retain legal n=576.
    m, r, s = 12, 2, 7
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
B_SHAPE = (2, 5)               # (rb, cb)


def _design_four_cycles(rows):
    """Count four-cycles within one basis, not across X and Z."""
    seen = {}
    cycles = 0
    for row in rows:
        cols = []
        while row:
            bit = row & -row
            cols.append(bit.bit_length() - 1)
            row ^= bit
        for a in range(len(cols)):
            for b in range(a):
                pair = (cols[b], cols[a])
                previous = seen.get(pair, 0)
                cycles += previous
                seen[pair] = previous + 1
    return cycles


def _design_small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1..4.

    Equal syndromes of supports of size <=2 exhaust this range.
    Reduce their difference against the opposite stabilizer row space.
    Inputs are exclusively the scaffold's verified lifted matrices.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def trivial(vector):
        while vector:
            p = vector.bit_length() - 1
            if p not in pivots:
                return False
            vector ^= pivots[p]
        return True

    syndromes = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] ^= syndrome_bit
            row ^= bit

    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(syndromes):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif not trivial(previous ^ bits[i]):
            return True

    for i in range(n):
        si, vi = syndromes[i], bits[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            vector = vi ^ bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = vector
            elif not trivial(previous ^ vector):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Optimize an indexed support/order/holonomy architecture."""
    import math
    from itertools import combinations

    a_profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),  # row totals 6/6
        ((1, 2, 2, 2), (2, 1, 1, 1)),  # row totals 7/5
    )
    b_profiles = (
        ((1, 1, 2, 2, 1), (2, 2, 1, 1, 2)),  # 7/8
        ((1, 1, 2, 2, 2), (2, 2, 1, 1, 2)),  # 8/8, one degree 4
        ((1, 2, 2, 2, 2), (2, 1, 1, 1, 1)),  # 9/6
    )
    support_pairs = ((0, 0), (1, 0), (0, 1), (0, 2))
    # Rotation differences 1,5 have order 12; 2,3,4 have orders 6,4,3.
    # Distinct directions within each row avoid repeated-binomial
    # directions. These briefs change relative-order portfolios,
    # rather than apply group automorphisms to a single filling.
    order_briefs = (
        (1, 5, 2, 3),
        (1, 2, 5, 3),
        (1, 3, 5, 2),
        (2, 3, 1, 5),
        (1, 4, 5, 2),
    )
    design = idx % 20
    architecture, brief = divmod(design, 5)
    mode = (idx // 20) % 3
    pa, pb = support_pairs[architecture]
    sa, sb = a_profiles[pa], b_profiles[pb]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    q = G.q
    priorities = order_briefs[brief]
    steps = []
    twists = []
    for side, counts in enumerate((sa, sb)):
        width = len(counts[0])
        grid = [[0] * width for _ in range(2)]
        for i in range(2):
            sites = [j for j in range(width) if counts[i][j] == 2]
            directions = list(priorities[:len(sites)])
            if i:
                directions.reverse()
            for j, delta in zip(sites, directions):
                grid[i][j] = delta
        steps.append(grid)
        # Additional harness batches get different holonomy briefs.
        ntwist = 1 if ((mode == 1 and side == 0) or
                      (mode == 2 and side == 1)) else 2
        start = (brief + side) % width
        twists.append({(start + t) % width for t in range(ntwist)})

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def labels(counts, side):
        first_coset = [rng.randrange(2) for _ in counts[0]]
        grid = []
        for i, row in enumerate(counts):
            out = []
            for j, count in enumerate(row):
                b = first_coset[j] ^ int(i == 1 and j in twists[side])
                a = rng.randrange(12)
                cell = [(a, b)]
                if count == 2:
                    cell.append(((a + steps[side][i][j]) % 12, b))
                out.append(cell)
            grid.append(out)
        return grid

    def component_rows(grid, side):
        # A unit partner isolates the component in sector one.
        # The helper supplies the exact left/right lift convention.
        unit = [[[G.e]]]
        if side == 0:
            rows, _, _ = lifted_matrices(G, grid, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
        mask = (1 << (len(grid[0]) * q)) - 1
        return [row & mask for row in rows]

    def columns_of(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return columns

    def cycle_cost(rows, width):
        columns = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1].append(r)
                row ^= bit
        adj = [dict() for _ in rows]
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
        if len(seen) != len(rows):
            return None

        four = sum(v * (v - 1) // 2
                   for a in range(len(rows))
                   for b, v in adj[a].items() if a < b)
        six = 0
        for a in range(len(rows)):
            for b, ab in adj[a].items():
                if b <= a:
                    continue
                for c, bc in adj[b].items():
                    if c > b and c in adj[a]:
                        six += ab * bc * adj[a][c]
        # Remove triangles that reuse a variable.
        for ns in columns:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        deficit = len(rows) - gf2_rank(rows)
        return four, six + 4 * q * deficit, deficit

    def energy(cost):
        return (60 * cost[0] + cost[1] + 2 * q * cost[2]) / q

    pools = [[], []]
    for side, counts in enumerate((sa, sb)):
        width = len(counts[0]) * q
        signatures = set()

        def offer(cost, serial, grid, rows):
            signature = tuple(tuple(sorted(G._key(g) for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return
            signatures.add(signature)
            pools[side].append((cost, serial, clone(grid), list(rows)))
            pools[side].sort(key=lambda item: (item[0], item[1]))
            del pools[side][5:]

        for restart in range(3):
            for _ in range(24):
                grid = labels(counts, side)
                rows = component_rows(grid, side)
                current = cycle_cost(rows, width)
                if current is not None:
                    break
            else:
                continue
            offer(current, restart * 1000, grid, rows)
            for step in range(128):
                i = rng.randrange(2)
                j = rng.randrange(len(counts[0]))
                old = grid[i][j]
                shift = rng.randrange(1, 12)
                # Preserve the slot's support, cosets and internal steps;
                # change inter-cell voltages, not the design brief.
                grid[i][j] = [((g[0] + shift) % 12, g[1]) for g in old]
                proposed_rows = component_rows(grid, side)
                proposed = cycle_cost(proposed_rows, width)
                if proposed is None:
                    grid[i][j] = old
                    continue
                delta = energy(proposed) - energy(current)
                temperature = 18.0 * (0.972 ** step) + 0.12
                if delta > 0 and rng.random() >= math.exp(
                        -min(700.0, delta / temperature)):
                    grid[i][j] = old
                    continue
                current, rows = proposed, proposed_rows
                offer(current, restart * 1000 + step + 1, grid, rows)

    if not all(pools):
        raise ValueError("no connected component in the indexed brief")

    def small_kernel_boundary(rows, width):
        """Exact kernel-weight<=6 existence and a boundary heuristic.

        Distinct supports of size <=3 with equal syndromes certify
        a nonzero kernel word of weight <=6. This is a component
        test, not a stabilizer-aware quantum-distance calculation.
        """
        columns = columns_of(rows, width)
        seen = {0}
        for value in columns:
            if value in seen:
                return True, 10**9
            seen.add(value)
        for i, x in enumerate(columns):
            for j in range(i):
                value = x ^ columns[j]
                if value in seen:
                    return True, 10**9
                seen.add(value)
        boundary = 0
        for i, x in enumerate(columns):
            for j in range(i):
                pair = x ^ columns[j]
                for h in range(j):
                    value = pair ^ columns[h]
                    if value in seen:
                        return True, 10**9
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 4:
                        boundary += (4 - weight) ** 2
        return False, boundary / width

    def sampled_kernel_bound(rows, width):
        """Random-information-set upper bound, never a certificate."""
        best = width
        for _ in range(32):
            mat = list(rows)
            order = list(range(width))
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
            for i, word in enumerate(basis):
                for previous in basis[:i]:
                    best = min(best, (word ^ previous).bit_count())
            if best <= 8:
                break
        return best

    finalists = []
    for side, pool in enumerate(pools):
        entries = []
        width = (A_SHAPE[1] if side == 0 else B_SHAPE[1]) * q
        for cost, serial, grid, rows in pool:
            bad, boundary = small_kernel_boundary(rows, width)
            distance = 6 if bad else sampled_kernel_bound(rows, width)
            entries.append((cost, serial, grid, bad, boundary, distance))
        finalists.append(entries)

    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, at, ag, abad, abound, ad = a
            bc, bt, bg, bbad, bbound, bd = b
            da, db = min(ad, 10), min(bd, 10)
            # A capped, sampled protection preference precedes boundary
            # and cycle tie-breakers. Neither it nor the boundary cost
            # is asserted to be the scored circuit-level pL.
            key = (int(abad) + int(bbad), int(min(da, db) < 10),
                   max(abound, bbound), abound + bbound,
                   ac[0] + bc[0], -min(da, db), abs(da - db),
                   ac[1] + bc[1], ac[2] + bc[2], at, bt)
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    fallback = None
    examined = 0
    for key, A, B in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (225 * n <= 1000 * k < 275 * n):
            continue
        if not css_ok(hx, hz):
            continue
        if max(row.bit_count() for row in hx + hz) > W_MAX:
            continue
        candidate = {"idx": idx, "cells": {"A": A, "B": B},
                     "architecture": architecture, "brief": brief,
                     "holonomy_mode": mode, "k": k, "screen": key}
        if fallback is None:
            fallback = candidate
        if (not _design_small_logical(hx, hz, n) and
                not _design_small_logical(hz, hx, n)):
            return candidate
        examined += 1
        if examined >= 8:
            break
    if fallback is not None:
        return fallback
    raise ValueError("no verified pair remains in rate cell 0.250")


def CELL(rng, G, side, i, j, ctx):
    """Return distinct elements of the selected complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: d2s=10 establishes basis-balanced circuit protection: current-parent slot42 had 91 X failures versus 11 Z failures in 200000 shots per basis.
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
