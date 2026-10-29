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
CONCEPT = ("C13 semidirect C6 with action s=4, order 78, A=(1,3), B=(2,5): order-thirteen A-binomial ratios and degree-four reinforcement of B should produce five measured distance readings above ten in cell 0.350.")
GROUP = ("Jump from abelian Z2 x Z22 to the nonabelian split-metacyclic C13:C6, s=4 of exact order six modulo thirteen. This order-78 bed is absent from BEDS. Successful related split-metacyclic beds motivate the experiment without establishing distance transfer. Fixed q=78 gives n=1326; the normal subgroup supplies order-thirteen ratios rather than short affine annihilators.")
SHAPES = ("A=(1,3), B=(2,5): n=17*78=1326, k>=78*2*3=468, guaranteed rate 6/17=0.352941 targets cell 0.350. Components have 234 variables/78 checks and 390 variables/156 checks. Parent measured minima were weight ten, entirely sector-1 Acol/Bcol operators, often emphasizing Acol2/3. Shortening B permits three degree-four columns. Exact full ranks retain [0.325,0.375).")
FILLING = ("A counts (3,3,2) give row total eight; five B support profiles have column totals at most four and row totals at most nine, hence HX<=8+4=12 and HZ<=9+3=12. Four relative-label prescriptions cross those profiles into twenty designs. A's binomial always has order-thirteen ratio, avoiding immediate weight-three/six annihilators. Distinct directed ratios exclude internal four-cycles. Cached monomial blocks come exclusively from verified lifts. Optimize actual cycles, screen kernels through four, redraw columns implicated by genuine kernel witnesses, and independently deepen every finalist. Exact ranks and CSS gate pairing; component readings remain provisional upper bounds, not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r035_C13C6s4_13x25_reinforced_kernel_retest_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 78

def custom_ops(q):
    # (a,b) = x^a y^b, x^13=y^6=1, yxy^-1=x^4.
    # 4^3=-1 mod 13 and 4 has exact order six.
    if not valid_q(q):
        raise ValueError("C13:C6 requires order 78")
    m, r, s = 13, 6, 4
    powers = tuple(pow(s, b, m) for b in range(r))
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty reinforced support/relative-label cards on verified blocks."""
    import math

    q = G.q
    assert q == 78
    card, cohort = idx % 20, idx // 20
    topology, model = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    sa = [[3, 3, 2]]
    bplans = (
        ((2, 2, 2, 1, 2), (2, 2, 2, 2, 1)),
        ((3, 3, 1, 1, 1), (1, 1, 3, 2, 2)),
        ((4, 2, 1, 1, 1), (0, 2, 3, 2, 2)),
        ((2, 2, 1, 1, 2), (2, 2, 2, 2, 1)),
        ((4, 0, 2, 1, 2), (0, 4, 2, 2, 1)),
    )
    sb = [list(row) for row in bplans[topology]]
    assert max(map(sum, sb)) <= 9
    assert max(sum(row[j] for row in sb) for j in range(5)) <= 4
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache native operations only. Authoritative monomial blocks are
    # obtained below from lifted_matrices, not from a separate lift law.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    orders = []
    for g in range(q):
        value, order = ident, 0
        while True:
            value = table[value][g]
            order += 1
            if value == ident:
                break
            assert order <= q
        orders.append(order)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    monomials = {}
    all_cells = [[[g] for g in range(q)]]
    for side in ("A", "B"):
        if side == "A":
            source, _, _ = lifted_matrices(probe, all_cells, [[[]]])
        else:
            _, source, _ = lifted_matrices(probe, [[[]]], all_cells)
        monomials[side] = [
            [(row >> (g*q)) & mask for row in source]
            for g in range(q)
        ]

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

    def draw(count, side, j):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if side == "A":
            if count == 2:
                # In a one-row component a ratio of order L supplies
                # an immediate weight-L single-column annihilator.
                bs = (0, 0)
            else:
                prescriptions = (
                    ((0, 0, 1), (0, 0, 1)),
                    ((0, 0, 1), (0, 0, 2)),
                    ((0, 1, 2), (0, 0, 1)),
                    ((0, 0, 1), (0, 0, 5)),
                )
                bs = prescriptions[model][j]
        elif count == 2:
            normal = model == 0 or (
                model in (1, 3) and rng.random() < 0.5)
            bs = (0, 0) if normal else (0, 1)
        elif count == 3:
            bs = ((0, 0, 1), (0, 1, 2),
                  (0, 0, 5), (0, 2, 4))[model]
        else:
            assert count == 4
            bs = ((0, 0, 1, 2), (0, 1, 1, 2),
                  (0, 0, 4, 5), (0, 0, 2, 4))[model]

        for _ in range(2000):
            relative = [13*b+rng.randrange(13) for b in bs]
            if len(set(relative)) != count:
                continue
            ratios = [table[inverses[g]][h]
                      for g in relative for h in relative if g != h]
            # Repeated directed differences or involution ratios
            # produce internal four-cycles in a regular monomial block.
            if len(set(ratios)) != len(ratios):
                continue
            if any(orders[g] == 2 for g in ratios):
                continue
            if count >= 3:
                x = table[inverses[relative[0]]][relative[1]]
                y = table[inverses[relative[0]]][relative[2]]
                if table[x][y] == table[y][x]:
                    continue
            root = rng.randrange(q)
            cell = [table[root][g] for g in relative]
            if side == "A" and count == 2:
                assert orders[table[inverses[cell[0]]][cell[1]]] == 13
            return cell
        raise ValueError("relative-label prescription could not be filled")

    def fresh(support, side):
        return [[draw(count, side, j) for j, count in enumerate(row)]
                for row in support]

    def inspect(grid, side):
        """GF2-linear assembly of blocks supplied by verified lifts."""
        width = len(grid[0])*q
        blocks = monomials[side]
        rows = []
        for cells in grid:
            for r in range(q):
                row = 0
                for j, cell in enumerate(cells):
                    block = 0
                    for g in cell:
                        block ^= blocks[g][r]
                    row |= block << (j*q)
                rows.append(row)
        cols = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                cols[c] |= 1 << r

        bad = sum(c == 0 for c in cols) + width-len(set(cols))
        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for value in cols:
            neighbors = list(bits(value))
            degree = len(neighbors)
            forced += degree*(degree-1)*(degree-2)//6
            for t, a in enumerate(neighbors):
                adjacency[a] |= value ^ (1 << a)
                for b in neighbors[:t]:
                    key = (b, a)
                    previous = uses.get(key, 0)
                    four += previous
                    uses[key] = previous+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        excess = max(0, triangles-forced)
        cost = 1000000*bad + (400+40*(cohort % 3))*four + excess
        return {"grid": grid, "rows": rows, "cols": cols,
                "cost": cost, "cycles": (four, excess),
                "d": width+1, "word": None, "side": side}

    def syndrome(cols, word):
        value = 0
        for c in bits(word):
            value ^= cols[c]
        return value

    def observe(rec, word):
        if word and word.bit_count() < rec["d"]:
            rec["d"], rec["word"] = word.bit_count(), word

    def short4(rec):
        """Every collision retained here is an actual kernel word."""
        singles, pairs = {}, {}
        hits = 0
        for c, value in enumerate(rec["cols"]):
            word = 1 << c
            if not value:
                observe(rec, word)
                hits += 1
            old = singles.get(value)
            if old is not None:
                observe(rec, old ^ word)
                hits += 1
            else:
                singles[value] = word
        for a, value in enumerate(rec["cols"]):
            for b in range(a):
                key = value ^ rec["cols"][b]
                word = (1 << a) | (1 << b)
                old = singles.get(key)
                if old is not None:
                    observe(rec, old ^ word)
                    hits += 1
                old = pairs.get(key)
                if old is not None:
                    observe(rec, old ^ word)
                    hits += 1
                else:
                    pairs[key] = word
        rec["short"] = hits

    def kernel_probe(rec, trials, offset):
        """Repeated genuine classical witnesses; absence is not certified."""
        local = random.Random(seed+offset+100003*int(rec["side"] == "B"))
        cols = rec["cols"]
        if rec["side"] == "A":
            # This known weight-thirteen annihilator prevents optimistic
            # diagnostics from overlooking the prescribed binomial.
            word = ((1 << 13)-1) << (2*q)
            assert syndrome(cols, word) == 0
            observe(rec, word)
        order = list(range(len(cols)))
        retained = []
        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
            for c in order:
                value, word = cols[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    observe(rec, word)
                    words.append(word)
            light = sorted(words, key=int.bit_count)[:18]
            for word in light:
                for other in words + retained:
                    observe(rec, word ^ other)
            descent = words + retained
            local.shuffle(descent)
            for initial in light[:6]:
                word = initial
                for sweep in range(2):
                    changed = False
                    for other in descent:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    observe(rec, word)
                    if not changed:
                        break
            retained = sorted(set(retained+light), key=int.bit_count)[:24]
            if rec["d"] <= 4:
                break
        assert rec["word"] is not None
        assert syndrome(cols, rec["word"]) == 0

    def merit(rec):
        return (-rec["short"], min(13, rec["d"]), -rec["cost"])

    def pool(support, side):
        bank = {}

        def remember(rec):
            key = signature(rec["grid"])
            bank[key] = rec
            if len(bank) > 6:
                worst = max(bank, key=lambda k: bank[k]["cost"])
                del bank[worst]

        starts = [inspect(fresh(support, side), side) for _ in range(28)]
        starts.sort(key=lambda rec: rec["cost"])
        for rec in starts[:6]:
            remember(rec)
        occupied = [(i, j) for i, row in enumerate(support)
                    for j, count in enumerate(row) if count]
        for current in starts[:2]:
            for step in range(90):
                grid = copy(current["grid"])
                i, j = rng.choice(occupied)
                grid[i][j] = draw(support[i][j], side, j)
                proposed = inspect(grid, side)
                delta = proposed["cost"]-current["cost"]
                temperature = q*(8*(1-step/90.0)**2+0.15)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(current)

        records = list(bank.values())
        for rec in records:
            short4(rec)
            kernel_probe(rec, 12, 0)
        records.sort(key=merit, reverse=True)
        current = records[0]
        keepers = records[:3]

        # Redraw relative voltages in an implicated column, rather than
        # submitting a gauge, group-automorphism or permutation image.
        for step in range(10):
            grid = copy(current["grid"])
            occupied_blocks = sorted({
                c // q for c in bits(current["word"])
            })
            j = rng.choice(occupied_blocks)
            for i in range(len(support)):
                grid[i][j] = draw(support[i][j], side, j)
            proposed = inspect(grid, side)
            if proposed["cycles"][0] > current["cycles"][0]+q:
                continue
            short4(proposed)
            kernel_probe(proposed, 12, 0)
            if merit(proposed) >= merit(current):
                current = proposed
            keepers.append(proposed)
            keepers.sort(key=merit, reverse=True)
            del keepers[6:]

        final, seen = [], set()
        for rec in sorted(keepers+[current], key=merit, reverse=True):
            key = signature(rec["grid"])
            if key in seen:
                continue
            seen.add(key)
            # Every returned finalist gets independent longer challenges.
            kernel_probe(rec, 48, 700001)
            kernel_probe(rec, 32, 1700003)
            rec["rank"] = gf2_rank(rec["rows"])
            final.append(rec)
            if len(final) == 4:
                break
        final.sort(key=lambda rec: (
            merit(rec), rec["rank"]), reverse=True)
        return final[:3]

    aa, bb = pool(sa, "A"), pool(sb, "B")
    best = None
    for a in aa:
        for b in bb:
            hx, hz, n = lifted_matrices(probe, a["grid"], b["grid"])
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1326 and k >= 468
                    and 325*n <= 1000*k < 375*n):
                continue
            # Check cached block assembly against the complete LP lift.
            ar = [
                sum(((hx[r] >> (j*5*q)) & mask) << (j*q)
                    for j in range(3))
                for r in range(q)
            ]
            br = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
            assert ar == a["rows"] and br == b["rows"]
            reading = min(13, a["d"], b["d"])
            key = (-a["short"]-b["short"], k*reading*reading,
                   min(26, a["d"]+b["d"]), -a["cost"]-b["cost"])
            if best is None or key > best[0]:
                best = (key, a, b, hx, hz, k)

    if best is None:
        raise ValueError("reinforced order-78 pools missed cell 0.350")
    key, a, b, cached_hx, cached_hz, k = best
    A = [[[G.elements[g] for g in cell] for cell in row]
         for row in a["grid"]]
    B = [[[G.elements[g] for g in cell] for cell in row]
         for row in b["grid"]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert n-gf2_rank(hx)-gf2_rank(hz) == k
    assert _grid_weights(sa, sb) <= W_MAX and _no_dead_lines(sa, sb)
    return {"idx": idx, "card": (topology, model, cohort),
            "cells": {"A": A, "B": B}, "k": k, "quality": key,
            "component_readings": (a["d"], b["d"]),
            "component_cycles": (a["cycles"], b["cycles"])}


def CELL(rng, G, side, i, j, ctx):
    """Return the screened full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled positive-cell supports with uniform labels suppress sector-1 weight-6 logicals on this bed; all five measured parent codes instead have d=6 and sector-1 minimum witnesses.
# REFUTED: Balanced abelian cubic columns with component weight-six screening suffice to break the distance-ten plateau and supply a novel code per slot; all five current measured readings were ten, and 31 of 60 slots were isomorphic duplicates.
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
