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
CONCEPT = ("Metacyclic Z10:Z2 with inversion action s=9, order 20, A=(3,8), B=(3,8): balanced three-check component columns and repeated verified kernel screening should raise the five measured distances above the parent's ten in cell 0.350.")
GROUP = ("Keep the inversion-action metacyclic construction rule and r=2; resize m=22 to m=10 and s=21 to s=9. Factor 73 permits largest even order q=20, giving n=1460. The parent D44 bed returned five distances of ten; this smaller lift with larger components is an untested shape tradeoff, not an established group improvement.")
SHAPES = ("Equal A=(3,8), B=(3,8) give n=73*20=1460, k>=20*5*5=500 and guaranteed rate 25/73=0.342466, targeting cell 0.350. Both components have 160 variables and 60 checks. Parent weight-ten X/Z minimum witnesses occupied sector-1 Acol/Bcol blocks across both components. Three-row components replace both kernels; exact full ranks retain [0.325,0.375).")
FILLING = ("Twenty support cards combine 0,2,3,4,5 balanced (2,1,0) columns with ordinary (1,1,1) columns, plus five one-sided reinforcement cards. Base rows have eight terms and columns three; reinforcement adds one term per row in distinct columns, giving rows nine and columns at most four. The opposite side stays at rows eight/columns three, so both CSS weight maxima are <=12. Fresh labels use order-ten same-coset binomial ratios. Independent verified component pools are selected by cycles, repeated randomized kernel searches and detected short words, then full ranks and CSS are checked. All sampled readings remain upper bounds.")
HOW_AB = FILLING
FAMILY_NAME = "r035_m10r2s9_38x38_balanced_kernel_pools"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Same inversion-action metacyclic rule, resized for factor 73.
    # Z_10 x| Z_2: y x y^-1 = x^9; 9^2 = 1 mod 10.
    m, r, s = 10, 2, 9
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _cubic_support(rng, cols, concentrated, target):
    """Select a balanced prescription with exactly three terms per column."""
    import itertools
    order = list(range(cols))
    rng.shuffle(order)
    sparse = set(order[:concentrated])
    options = [(0, 3) if j in sparse else (1, 2)
               for j in range(cols)]
    choices = [v for v in itertools.product(*options) if sum(v) == target]
    first = list(rng.choice(choices))
    grid = [first, [3 - v for v in first]]
    assert all(sum(row) > 0 for row in grid)
    return grid


def _coset_pattern(rng, supports):
    """Two split columns carry different coset holonomies."""
    cols = len(supports[0])
    bits = [[rng.randrange(2) for _ in range(cols)] for _ in range(2)]
    split = [j for j in range(cols)
             if supports[0][j] and supports[1][j]]
    assert len(split) >= 2
    j, k = split[:2]
    bits[0][j] = bits[1][j] = 0
    bits[0][k], bits[1][k] = 0, 1
    return bits


def _legal_exponents(exponents, m):
    import math
    if len(set(exponents)) != len(exponents):
        return False
    if len(exponents) < 2:
        return True
    gaps = []
    divisor = m
    for a in range(len(exponents)):
        for b in range(a + 1, len(exponents)):
            d = (exponents[a] - exponents[b]) % m
            if 2 * d == m:
                return False
            gaps.append(min(d, m - d))
            divisor = math.gcd(divisor, d)
    # Distinct signed difference classes prevent internal four-cycles.
    return divisor == 1 and len(set(gaps)) == len(gaps)


def _new_coset_cell(rng, m, count, bit):
    if count == 0:
        return []
    while True:
        exponents = rng.sample(range(m), count)
        if _legal_exponents(exponents, m):
            return [(a, bit) for a in exponents]


def _initial_cubic_cells(rng, G, supports, bits):
    m = G.q // 2
    return [[_new_coset_cell(rng, m, count, bits[i][j])
             for j, count in enumerate(row)]
            for i, row in enumerate(supports)]


def _component_rows(hx, hz, q, side):
    """Extract the actual left/right components from verified matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "B":
        mask = (1 << (cb * q)) - 1
        return [hz[a * q + r] & mask
                for a in range(rb) for r in range(q)]
    mask = (1 << q) - 1
    rows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            rows.append(row)
    return rows


def _cycle_cost(rows, penalty):
    """Count component four/six cycles, excluding reused variable nodes."""
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    c4 = 0
    for a in range(h):
        for b in range(a + 1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                c4 += overlap * (overlap - 1) // 2
    c6 = 0
    for a in range(h):
        ns = upper[a]
        for t, b in enumerate(ns):
            ab = common[a][b]
            for c in ns[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    c6 += ab * ac * bc - triple * (ab + ac + bc - 2)
    return penalty * c4 + c6


def _verified_costs(G, A, B, penalty, side=None):
    hx, hz, _ = lifted_matrices(G, A, B)
    sides = (side,) if side is not None else ("A", "B")
    costs = [_cycle_cost(_component_rows(hx, hz, G.q, s), penalty)
             for s in sides]
    return costs[0] if side is not None else costs


def _small_kernel_penalty(rows, width):
    """Exact <=4 veto plus a connected-triple weight-six witness screen."""
    import itertools
    columns = [0] * width
    adjacency = [set() for _ in range(width)]
    for r, row in enumerate(rows):
        remaining = row
        incident = []
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            columns[j] |= 1 << r
            incident.append(j)
            remaining ^= bit
        for j in incident:
            adjacency[j].update(k for k in incident if k != j)

    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return 1000000

    pairs = set()
    short = 0
    for a in range(width):
        for b in range(a + 1, width):
            syndrome = columns[a] ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                short += 1
            pairs.add(syndrome)
    if short:
        return 100000 + short

    # Enumerate triples connected through shared checks. A collision
    # verifies a six-column kernel word; this is not an exhaustive
    # general weight-six search.
    triples = set()
    for b in range(width):
        for a, c in itertools.combinations(sorted(adjacency[b]), 2):
            triples.add(tuple(sorted((a, b, c))))
    seen = {}
    six = 0
    for a, b, c in sorted(triples):
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        mask = (1 << a) | (1 << b) | (1 << c)
        old = seen.get(syndrome)
        if old is not None:
            if (old ^ mask).bit_count() == 6:
                six += 1
        else:
            seen[syndrome] = mask
    return six


def _optimize_labels(rng, G, A, B, penalty):
    import math
    costs = _verified_costs(G, A, B, penalty)
    current = sum(costs)
    best = current
    best_A, best_B = _copy_cells(A), _copy_cells(B)
    positions = [[(i, j) for i, row in enumerate(grid)
                  for j, cell in enumerate(row) if cell]
                 for grid in (A, B)]
    m = G.q // 2
    steps = 240
    for step in range(steps):
        which = rng.randrange(2)
        grid = A if which == 0 else B
        i, j = rng.choice(positions[which])
        old = list(grid[i][j])
        bit = old[0][1]
        if step % 9 == 0:
            new = _new_coset_cell(rng, m, len(old), bit)
        else:
            new = None
            for _ in range(40):
                exponents = [g[0] for g in old]
                exponents[rng.randrange(len(old))] = rng.randrange(m)
                if _legal_exponents(exponents, m):
                    new = [(a, bit) for a in exponents]
                    break
            if new is None:
                continue
        grid[i][j] = new
        side = ("A", "B")[which]
        proposed_cost = _verified_costs(G, A, B, penalty, side)
        proposed = current - costs[which] + proposed_cost
        temperature = G.q * (18.0 * (1.0 - step / steps) ** 2 + 0.25)
        accept = proposed <= current
        if not accept:
            accept = rng.random() < math.exp((current - proposed) /
                                             temperature)
        if accept:
            current = proposed
            costs[which] = proposed_cost
            if current < best:
                best = current
                best_A, best_B = _copy_cells(A), _copy_cells(B)
        else:
            grid[i][j] = old
    return best_A, best_B, best


def NEW_CTX(rng, G, idx):
    """Twenty support cards with independently screened component pools."""
    import itertools

    levels = (0, 2, 3, 4, 5)
    plans = [(a, b, 0)
             for p, a in enumerate(levels) for b in levels[p:]]
    plans += [(a, 0, 3) for a in levels]
    ta, tb, extra = plans[idx % 20]
    cohort = (idx // 20) % 3
    bias = (0.50, 0.25, 0.75)[cohort]
    penalty = (256, 384, 512)[cohort]
    q = G.q
    width = 8 * q

    def support(concentrated, reinforced):
        """Rows eight, columns three, before distinct-column extras."""
        options = tuple(itertools.permutations((2, 1, 0)))
        chosen = None
        for _ in range(512):
            trial = [rng.choice(options) for _ in range(concentrated)]
            if all(sum(col[i] for col in trial) == concentrated
                   for i in range(3)):
                chosen = trial
                break
        if chosen is None:
            # Balanced deterministic fallback, not a repair of final cells.
            chosen = []
            left = concentrated
            if left % 2:
                chosen.extend(((2,1,0), (0,2,1), (1,0,2)))
                left -= 3
            for _ in range(left // 2):
                chosen.extend(((2,1,0), (0,1,2)))
        columns = chosen + [(1,1,1)] * (8 - concentrated)
        rng.shuffle(columns)
        row_order = list(range(3))
        rng.shuffle(row_order)
        result = [[col[i] for col in columns] for i in row_order]
        assert all(sum(row) == 8 for row in result)

        used = set()
        rows = list(range(3))
        rng.shuffle(rows)
        for i in rows[:reinforced]:
            available = [j for j in range(8)
                         if j not in used and result[i][j] <= 1]
            j = rng.choice(available)
            result[i][j] += 1
            used.add(j)
        return result

    sa = support(ta, extra)
    sb = support(tb, 0)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    def draw_grid(sup, favored):
        result = []
        for row in sup:
            cells = []
            for count in row:
                if not count:
                    cells.append([])
                    continue
                bit = int(rng.random() < favored)
                # Distinct binomial labels have rotation ratio of order ten.
                cells.append(_new_coset_cell(rng, q // 2, count, bit))
            result.append(cells)
        return result

    def component(grid, side):
        """Identity probes extract the scaffold's exact component action."""
        unit = [[[G.e]]]
        if side == "A":
            xx, _, _ = lifted_matrices(G, grid, unit)
            source = xx
        else:
            _, zz, _ = lifted_matrices(G, unit, grid)
            source = zz
        mask = (1 << width) - 1
        rows = [row & mask for row in source]
        columns = [0] * width
        for r, word in enumerate(rows):
            while word:
                bit = word & -word
                columns[bit.bit_length() - 1] |= 1 << r
                word ^= bit
        rank = gf2_rank(rows)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cost": _cycle_cost(rows, penalty)
        }

    def kernel_reading(columns, trials):
        """Search actual kernel circuits and their pairwise cancellations."""
        order = list(range(len(columns)))
        best = len(columns) + 1
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            words = []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    words.append(word)
                    best = min(best, word.bit_count())
            light = sorted(words, key=int.bit_count)[:8]
            for a in light:
                for b in words:
                    if a != b:
                        best = min(best, (a ^ b).bit_count())
            if best <= 4:
                break
        return best

    def pool(sup, side, favored):
        records = []
        for _ in range(64):
            record = component(draw_grid(sup, favored), side)
            cols = record["columns"]
            if (0 in cols or len(set(cols)) != width
                    or record["rank"] < 3 * q - 10):
                continue
            records.append(record)
        if not records:
            raise ValueError("no nondegenerate component in fresh pool")

        records.sort(key=lambda rec: (rec["cost"], -rec["rank"]))
        records = records[:10]
        for record in records:
            record["d"] = kernel_reading(record["columns"], 18)
        records.sort(
            key=lambda rec: (rec["d"], -rec["cost"], rec["rank"]),
            reverse=True)

        finalists = records[:4]
        for record in finalists:
            # Independent extra elimination orders challenge high readings.
            record["d"] = min(
                record["d"], kernel_reading(record["columns"], 72))
            short = _small_kernel_penalty(record["rows"], width)
            record["short"] = short
            if short >= 100000:
                record["d"] = min(record["d"], 4)
            elif short:
                record["d"] = min(record["d"], 6)
        finalists.sort(
            key=lambda rec: (
                rec["d"], -rec["short"], -rec["cost"], rec["rank"]),
            reverse=True)
        return finalists[:2]

    # No group automorphism or permutation orbit is used to make slots:
    # each support card independently constructs and selects fresh voltages.
    for attempt in range(2):
        aa = pool(sa, "A", bias)
        bb = pool(sb, "B", 1.0 - bias)
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(
            key=lambda ab: (
                min(ab[0]["d"], ab[1]["d"]),
                ab[0]["d"] + ab[1]["d"],
                -ab[0]["short"] - ab[1]["short"],
                -ab[0]["cost"] - ab[1]["cost"]),
            reverse=True)
        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1460 and k > 0
                    and 325 * n <= 1000 * k < 375 * n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert _grid_weights(sa, sb) <= W_MAX
            return {
                "idx": idx, "card": (ta, tb, extra, cohort),
                "cells": {"A": A, "B": B}, "screen_k": k,
                "component_readings": (a["d"], b["d"]),
                "structural_penalty": (a["short"], b["short"])
            }
    raise ValueError("fresh component pools missed the exact rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the optimized full cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling and uniform labels alone suppress very short logicals in the order-52 (2,6)x(2,4) bed: all five measured parent codes had distance 4, with many sector-1 witnesses.
# REFUTED: The D44 cubic-coset filling's five measured selections have distance above ten; all five instead returned ten, with abundant sector-1 minimum witnesses.
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
