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
CONCEPT = ("Dic11, order 44, with A=(1,2) and B=(3,5): mixed-coset, degree-distinguished fillings selected against verified component logical witnesses should yield five measured d30s readings >=14 in cell 0.150.")
GROUP = ("Jump from cyclic Z23 to nonabelian Dic11. Its recorded (2,3)x(2,3) bed reaches fitness 51.9, motivating this untried shape bed without assuming transfer. Dic11 has nonrotation elements of order four and a unique involution whose relative cell differences can be excluded.")
SHAPES = ("Target 0.150: n=13*44=572, guaranteed k>=88 and rate 2/13; require 88<=k<=100 by verified ranks. A has 88 component variables/44 checks and B has 220/132. Parent minima were sector-1 Z witnesses of weight 12-13 spread across all A columns, particularly Bcol1-3, so screen both component kernels.")
FILLING = ("Twenty enumerated support plans have distinct combined Tanner degree fingerprints. A cells contain 3..5 terms; B cells contain 1..2. Explicit row/column budgets bound both check weights by 12. Mixed-coset cells exclude involution-separated terms; bounded label optimization uses verified component four-cycles, exhaustive A kernel screening through weight six, exact target-rate gating, and stabilizer-tested logical probes with kernel descent. Witness readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r015_dic11_12x35_degree_logical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic11 requires order 44")
    # x^22=1, y^2=x^11, y*x*y^-1=x^-1; normal form x^a y^b.
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((a + 11 * (b // 2)) % 22, b % 2)

    def idx(g):
        a, b = red(g)
        return a + 22 * b

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11 * b * d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _d11_support_plans():
    """Degree fingerprints distinguish plans even under arbitrary relabeling."""
    from itertools import combinations_with_replacement
    choices = {}
    for ap in ((4, 4), (3, 5), (3, 4), (3, 3)):
        ar = sum(ap)
        for masks in combinations_with_replacement(range(8), 5):
            sb = tuple(tuple(1 + ((m >> i) & 1) for m in masks)
                       for i in range(3))
            br = tuple(sum(row) for row in sb)
            bc = tuple(3 + m.bit_count() for m in masks)
            if ar + max(bc) > 12 or max(ap) + max(br) > 12:
                continue
            # Each listed degree occurs q times in the actual Tanner graph.
            checks = tuple(sorted(
                [ar + c for c in bc]
                + [r + a for a in ap for r in br]))
            qubits = tuple(sorted(
                [a + c for a in ap for c in bc]
                + [ar + r for r in br]))
            fingerprint = (checks, qubits)
            priority = (min(ap), min(bc), ar + sum(br),
                        -max(br), -abs(ap[0] - ap[1]))
            entry = (priority, ap, sb)
            old = choices.get(fingerprint)
            if old is None or entry > old:
                choices[fingerprint] = entry
    ordered = sorted(choices.values(), reverse=True)
    if len(ordered) < 20:
        raise ValueError("fewer than twenty degree-distinguished plans")
    return tuple((ap, sb) for _, ap, sb in ordered[:20])


_D11_PLANS = _d11_support_plans()


def NEW_CTX(rng, G, idx):
    """Degree-distinguished architecture, then bounded label optimization.
    Every matrix used for screening comes from lifted_matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ap, bp = _D11_PLANS[idx % 20]
    sa = [list(ap)]
    sb = [list(row) for row in bp]
    assert q == 44
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    involution = (11, 0)
    pools = [[g for g in G.elements if g[1] == b] for b in (0, 1)]

    def draw_cell(count):
        if count == 1:
            return [G.pick(rng)]
        # Both cosets are present; independent samples are not orbit copies.
        while True:
            nr = rng.randrange(1, count)
            cell = (rng.sample(pools[0], count - nr)
                    + rng.sample(pools[1], nr))
            if all(G.mul(G.inv(g), h) != involution
                   for p, g in enumerate(cell) for h in cell[:p]):
                rng.shuffle(cell)
                return cell

    def columns(rows, length):
        out = [0] * length
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def components(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        # Extract beta=0 from HX and j=0 from HZ, preserving the
        # verified lift's actions. These are the two component matrices.
        ar = []
        for i in range(ra):
            for r in range(q):
                row = hx[(i * cb) * q + r]
                compact = 0
                for j in range(ca):
                    compact |= ((row >> (j * cb * q)) & mask) << (j * q)
                ar.append(compact)
        bm = (1 << (cb * q)) - 1
        br = [row & bm for row in hz[:rb * q]]
        return hx, hz, n, columns(ar, ca * q), columns(br, cb * q)

    def four_cycles(cols):
        total = 0
        for i, x in enumerate(cols):
            for j in range(i):
                t = (x & cols[j]).bit_count()
                total += t * (t - 1) // 2
        return total

    def short_kernel(cols):
        """Return a kernel witness of weight <=6, or None.
        Collisions among all subsets of size <=3 cover every such word."""
        seen = {0: 0}

        def insert(s, v):
            old = seen.get(s)
            if old is not None:
                return old ^ v
            seen[s] = v
            return None

        length = len(cols)
        for i in range(length):
            hit = insert(cols[i], 1 << i)
            if hit is not None:
                return hit
        for i in range(length):
            for j in range(i):
                hit = insert(cols[i] ^ cols[j], (1 << i) | (1 << j))
                if hit is not None:
                    return hit
        for i in range(length):
            for j in range(i):
                s = cols[i] ^ cols[j]
                v = (1 << i) | (1 << j)
                for h in range(j):
                    hit = insert(s ^ cols[h], v | (1 << h))
                    if hit is not None:
                        return hit
        return None

    def row_basis(rows):
        basis = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = row
                    break
                row ^= old
        return basis

    def outside_span(v, basis):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    def logical_estimate(cols, opposite_basis, side):
        """Sample component kernel words, rejecting LP stabilizers.
        The returned positive weight is an upper bound from actual witnesses."""
        best = [ca * cb * q + ra * rb * q + 1]

        def consider(v):
            w = v.bit_count()
            if not v or w >= best[0]:
                return
            if side == "A":
                for beta in range(cb):
                    full = 0
                    for j in range(ca):
                        full |= ((v >> (j * q)) & mask) << (
                            (j * cb + beta) * q)
                    if outside_span(full, opposite_basis):
                        best[0] = w
                        return
            else:
                for j in range(ca):
                    full = v << (j * cb * q)
                    if outside_span(full, opposite_basis):
                        best[0] = w
                        return

        order = list(range(len(cols)))
        rounds = 18 if side == "A" else 10
        for _ in range(rounds):
            rng.shuffle(order)
            pivots = {}
            words = []
            for i in order:
                s, v = cols[i], 1 << i
                while s:
                    p = s.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (s, v)
                        break
                    s ^= old[0]
                    v ^= old[1]
                if not s:
                    words.append(v)
                    consider(v)
            words.sort(key=int.bit_count)
            for i, x in enumerate(words):
                for j in range(i):
                    consider(x ^ words[j])
            small = words[:12]
            for i, x in enumerate(small):
                for j in range(i):
                    xy = x ^ small[j]
                    for h in range(j):
                        consider(xy ^ small[h])

            # XOR descent stays in the exact component kernel. Membership
            # in the actual stabilizer span is still checked by consider.
            for start in words[:8]:
                v = start
                for _ in range(5):
                    improved = v
                    weight = v.bit_count()
                    for other in words:
                        trial = v ^ other
                        tw = trial.bit_count()
                        if trial and tw < weight:
                            improved, weight = trial, tw
                    if improved == v:
                        break
                    v = improved
                    consider(v)
            if best[0] <= 6:
                break
        return best[0] if best[0] <= ca * cb * q + ra * rb * q else 0

    targets = {
        "A": [(i, j) for i in range(ra) for j in range(ca) if sa[i][j]],
        "B": [(i, j) for i in range(rb) for j in range(cb) if sb[i][j]],
    }
    winner = None
    reserve = None
    accepted = 0

    for attempt in range(20):
        A = [[draw_cell(count) for count in row] for row in sa]
        B = [[draw_cell(count) for count in row] for row in sb]
        cells = {"A": A, "B": B}
        data = components(A, B)
        cycle = [four_cycles(data[3]), four_cycles(data[4])]

        # Slightly favor A, but preserve both cosets and the involution
        # exclusion throughout optimization. Counts cannot drift.
        for step in range(18):
            side = "A" if step % 3 != 2 else "B"
            which = 0 if side == "A" else 1
            i, j = rng.choice(targets[side])
            cell = cells[side][i][j]
            t = rng.randrange(len(cell))
            old = cell[t]
            others = cell[:t] + cell[t + 1:]
            allowed = [
                g for g in G.elements
                if g not in cell
                and all(G.mul(G.inv(h), g) != involution for h in others)
                and (len(cell) == 1
                     or len({h[1] for h in others} | {g[1]}) == 2)]
            if not allowed:
                continue
            cell[t] = rng.choice(allowed)
            trial = components(A, B)
            cost = four_cycles(trial[3 + which])
            if cost < cycle[which] or (
                    cost == cycle[which] and rng.random() < 0.6):
                data = trial
                cycle[which] = cost
            else:
                cell[t] = old

        hx, hz, n, ac, bc = data
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # kb=88; 100/572 < .175 keeps every admitted code in cell 0.150.
        if n != 572 or not 88 <= k <= 100:
            continue

        short = short_kernel(ac)
        short_weight = short.bit_count() if short is not None else 7
        reserve_key = (short_weight, -cycle[0], -cycle[1])
        if reserve is None or reserve_key > reserve[0]:
            reserve = (reserve_key, A, B)
        if short is not None:
            continue

        dz = logical_estimate(ac, row_basis(hz), "A")
        dx = logical_estimate(bc, row_basis(hx), "B")
        if not dz or not dx:
            continue
        d = min(dx, dz)
        key = (k * d * d / n, d, dx + dz, -sum(cycle))
        if winner is None or key > winner[0]:
            winner = (key, A, B)
        accepted += 1
        if accepted >= 6:
            break

    chosen = winner if winner is not None else reserve
    if chosen is None:
        raise ValueError("no positive-k candidate in the requested rate band")
    A, B = chosen[1], chosen[2]
    hx, hz, _ = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled, predominantly singleton A fillings reliably suppress weight-six Z witnesses; rejected by the parent: four of five measured codes have Z weight 6, while X witnesses weigh 16-20.
# REFUTED: Excluding A-kernel words through weight six establishes d30s>=14; the current Z23 parent still measured four distances of 12 and one of 13.
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
