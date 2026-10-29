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
CONCEPT = ("C7 semidirect C6 with action 2, order 42, shapes (1,2)x(2,6): noncommuting A trinomials and balanced B columns should improve cell 0.300 beyond 5.7715 by reducing short-word and small-boundary exposure without increasing extraction load.")
GROUP = ("Jump to the nonfaithful action-2 group, isomorphic to (C7 semidirect C3) x C2. Its central C2 distinguishes it from the parent's centre-one faithful action-3 group. This order-42 shape bed is absent from BEDS. The related parent reached 7.1776 with zero observed failures in all five measured codes; this motivates retaining a substantial nonabelian lift, not assuming a group-only advantage.")
SHAPES = ("A=(1,2), B=(2,6) give n=588, kb=168 and guaranteed rate 2/7 in cell 0.300. A has 84 variables/42 checks and B has 252 variables/84 checks. Full verified ranks enforce 0.275<=k/n<0.325. The parent's measured codes have no failed-shot block attribution; their short witnesses mostly occupied Acol/Bcol sector one, so there is no demonstrated Acol/Brow hotspot to repair.")
FILLING = ("Twenty contracts cross four noncommuting A triangle pairs with five B binomial-order policies, using relative orders 3,6,7,14 and excluding involutory differences. Baseline A weights are (3,3), B rows have nine terms and columns three: HX weight nine, HZ weight twelve, and 25704 six-round CNOTs. Extra slots remove one B doubleton or use A=(3,4) with B row totals eight. Both budgets are checked explicitly. Helper-derived component actions optimize rank, connectivity and exact short cycles; finalists screen component kernels through weight four, normalized triple boundaries and capped sampled kernel witnesses. Complete helper-built CSS matrices determine rate and prefer no quantum logical through weight four. These are pL hypotheses, not circuit measurements: the leading 0.300 bed still reached the failed-shot cap with widespread collective flips.")
FAMILY_NAME = "r030_C7byC6_s2_12x26_noncomm_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this nonfaithful C7 semidirect C6 requires q=42")
    # x^7=y^6=1, y*x*y^-1=x^2.
    # The action has order three; y^3 generates the central C2.
    act = (1, 2, 4, 1, 2, 4)
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + act[b] * c) % 7, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-act[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _fg_columns(rows, width):
    """Column syndromes of a verified component action."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _fg_cost(rows, width):
    """Connectivity, exact rank deficit, and exact Tanner 4/6 cycles."""
    from itertools import combinations

    cols = _fg_columns(rows, width)
    if not all(cols) or not all(rows):
        return None
    neighbors = [[] for _ in range(width)]
    adj = [dict() for _ in rows]
    for c, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            neighbors[c].append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, b in combinations(neighbors[c], 2):
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
    for ns in neighbors:
        for a, b, c in combinations(ns, 3):
            six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
    return len(rows) - gf2_rank(rows), four, six


def _fg_quality(rng, rows, width, q):
    """Exact kernel<=4 exclusion and small-boundary/kernel heuristics.

    Triple boundaries are enumerated, but triple syndromes are not stored:
    this is not an exact exclusion test through weight six.
    Randomized kernel circuits give upper bounds, never pL predictions.
    """
    cols = _fg_columns(rows, width)
    seen = {0}
    for value in cols:
        if value in seen:
            return True, 4, 1.0e9, 1.0e9
        seen.add(value)
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            if syndrome in seen:
                return True, 4, 1.0e9, 1.0e9
            seen.add(syndrome)

    exposure = 0
    loads = [0] * (width // q)
    for i in range(width):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                weight = (pair ^ cols[h]).bit_count()
                if weight < 5:
                    penalty = (5 - weight) ** 2
                    exposure += penalty
                    loads[i // q] += penalty
                    loads[j // q] += penalty
                    loads[h // q] += penalty
    triples = width * (width - 1) * (width - 2) // 6
    boundary = exposure / max(1, triples)
    peak = max(loads) * len(loads) / max(1, 3 * triples)

    order = list(range(width))
    best = width + 1
    for _ in range(28):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = syndrome, word
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome:
                best = min(best, word.bit_count())
                for previous in basis:
                    best = min(best, (word ^ previous).bit_count())
                basis.append(word)
        if best <= 6:
            break
    return False, best, boundary, peak


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def NEW_CTX(rng, G, idx):
    """Optimize twenty noncommutative contracts using verified actions."""
    import math

    q = G.q
    design = idx % 20
    mode = idx // 20
    architecture, policy = divmod(design, 5)
    sa = [[3, 3 if mode < 2 else 4]]
    sb = [[2, 2, 2, 1, 1, 1],
          [1, 1, 1, 2, 2, 2]]
    if mode >= 1:
        sb[0][0] = 1
    if mode >= 2:
        sb[1][5] = 1

    wx = sum(sa[0]) + max(sum(row[j] for row in sb)
                          for j in range(B_SHAPE[1]))
    wz = max(map(sum, sb)) + max(sa[0])
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    elements = list(G.elements)
    allowed = (3, 6, 7, 14)

    def element_order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid element order")

    orders = {g: element_order(g) for g in elements}
    by_order = {t: [g for g in elements if orders[g] == t]
                for t in allowed}

    def relative_order(g, h):
        return orders[G.mul(G.inv(g), h)]

    # Unordered edge-order spectra distinguish the internal briefs.
    # Noncommutation is required, not inferred from the spectra.
    triangles = ((3, 3, 7), (6, 6, 7),
                 (3, 6, 14), (3, 6, 6))
    triangle_books = [[] for _ in triangles]
    usable = [g for g in elements if orders[g] in allowed]
    for a, u in enumerate(usable):
        for v in usable[a + 1:]:
            if G.mul(u, v) == G.mul(v, u):
                continue
            spectrum = tuple(sorted((orders[u], orders[v],
                                     relative_order(u, v))))
            if spectrum in triangles:
                triangle_books[triangles.index(spectrum)].append((u, v))
    assert all(triangle_books)
    a_cards = ((0, 1), (1, 2), (0, 2), (3, 3))
    a_contract = a_cards[architecture]
    b_palettes = ((14,), (7,), (6,), (3,), (14, 7, 6, 3))
    palette = b_palettes[policy]

    def draw_cell(side, i, j):
        g = G.pick(rng)
        if side == 0:
            u, v = rng.choice(triangle_books[a_contract[j]])
            normalized = [G.e, u, v]
            if sa[i][j] == 4:
                extensions = [
                    w for w in elements
                    if all(relative_order(t, w) in allowed
                           for t in normalized)
                ]
                normalized.append(rng.choice(extensions))
            return [G.mul(g, t) for t in normalized]
        if sb[i][j] == 1:
            return [g]
        order = palette[(j + 2 * i) % len(palette)]
        return [g, G.mul(g, rng.choice(by_order[order]))]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_grid(side):
        counts = sa if side == 0 else sb
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts)]

    # Cache singleton blocks extracted only from lifted_matrices.
    # No independently implemented regular-representation convention.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][g] = tuple(row & mask for row in hx)
        actions[1][g] = tuple(row & mask for row in hz)

    def component_rows(grid, side):
        result = []
        for grid_row in grid:
            rows = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        rows[r] ^= bits << (j * q)
            result.extend(rows)
        return result

    widths = (2 * q, 6 * q)
    shapes = (A_SHAPE, B_SHAPE)

    def energy(cost):
        deficit, four, six = cost
        return (12 * q * deficit + 80 * four + six) / q

    for batch in range(2):
        pools = [[], []]
        for side in range(2):
            seen_fillings = set()

            def offer(cost, serial, grid, rows):
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen_fillings:
                    return
                seen_fillings.add(signature)
                pools[side].append(
                    (cost, serial, clone(grid), list(rows)))
                pools[side].sort(key=lambda item: (item[0], item[1]))
                del pools[side][5:]

            for restart in range(4):
                for attempt in range(24):
                    grid = draw_grid(side)
                    rows = component_rows(grid, side)
                    current = _fg_cost(rows, widths[side])
                    if current is not None:
                        break
                else:
                    continue
                offer(current, restart * 121, grid, rows)

                for step in range(120):
                    nr, nc = shapes[side]
                    i, j = rng.randrange(nr), rng.randrange(nc)
                    old = grid[i][j]
                    if side == 1 and rng.random() < 0.3:
                        shift = G.pick(rng)
                        proposed_cell = [G.mul(shift, g) for g in old]
                    else:
                        # On single-row A, redraw its internal geometry;
                        # do not spend the search on whole-cell gauges.
                        proposed_cell = draw_cell(side, i, j)
                    assert len(proposed_cell) == len(old)
                    assert len(set(proposed_cell)) == len(old)
                    grid[i][j] = proposed_cell
                    proposed_rows = component_rows(grid, side)
                    proposed = _fg_cost(proposed_rows, widths[side])
                    if proposed is None:
                        grid[i][j] = old
                        continue
                    delta = energy(proposed) - energy(current)
                    temperature = 20 * (0.975 ** step) + 0.15
                    if delta > 0 and rng.random() >= math.exp(
                            -min(700.0, delta / temperature)):
                        grid[i][j] = old
                        continue
                    improved = proposed < current
                    current, rows = proposed, proposed_rows
                    if improved or step % 8 == 0:
                        offer(current, restart * 121 + step + 1,
                              grid, rows)
                offer(current, 1000 + restart, grid, rows)

        if not all(pools):
            continue

        finalists = [[], []]
        for side in range(2):
            for cost, serial, grid, rows in pools[side][:4]:
                bad, distance, boundary, peak = _fg_quality(
                    rng, rows, widths[side], q)
                finalists[side].append(
                    (cost, bad, distance, boundary, peak, serial, grid))

        options = []
        for a in finalists[0]:
            for b in finalists[1]:
                ac, abad, ad, abound, apeak, at, A = a
                bc, bbad, bd, bbound, bpeak, bt, B = b
                # Establish only a coarse sampled protection floor.
                # Above it, prefer smaller boundaries and cycle burden;
                # larger upper bounds are not treated as circuit scores.
                key = (int(abad) + int(bbad),
                       int(min(ad, bd) < 10),
                       max(apeak, bpeak), abound + bbound,
                       ac[1] + bc[1], -min(ad, bd, 12),
                       ac[2] + bc[2], ac[0] + bc[0], at, bt)
                options.append((key, A, B))
        options.sort(key=lambda item: item[0])

        fallback = None
        examined = 0
        for key, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 588 and k > 0 and
                    11 * n <= 40 * k < 13 * n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            if not _connected_checks(hx, hz, n):
                continue

            # Confirm that the cached actions are exactly the components
            # of the submitted complete matrices on both sides.
            ar = [
                sum(((hx[r] >> (j * B_SHAPE[1] * q)) & mask)
                    << (j * q) for j in range(A_SHAPE[1]))
                for r in range(q)
            ]
            br = [row & ((1 << widths[1]) - 1)
                  for row in hz[:B_SHAPE[0] * q]]
            assert ar == component_rows(A, 0)
            assert br == component_rows(B, 1)

            candidate = {"A": A, "B": B, "design": design,
                         "mode": mode, "k": k, "screen": key,
                         "contract": (a_contract, palette),
                         "small_logical_safe": False}
            if fallback is None:
                fallback = candidate
            if not _has_short_logical(hx, hz, n):
                candidate["small_logical_safe"] = True
                return candidate
            examined += 1
            if examined >= 12:
                break

        # A legal target-cell candidate is preferable to voiding a slot
        # solely because a heuristic protection preference was unmet.
        if fallback is not None:
            return fallback
    raise ValueError("no verified nonfaithful-action filling in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; no subsequent redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance alone raises measured fitness: zero-failure parent slot58 had d2s=22,k=45,s=7.165, below zero-failure slot20 with d2s=20,k=58,s=7.276; their sample ceilings limited both.
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
