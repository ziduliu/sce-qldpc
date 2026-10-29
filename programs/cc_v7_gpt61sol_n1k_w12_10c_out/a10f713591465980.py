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
CONCEPT = ("For Z19:Z3(s=7), q=57, shapes (4,2)x(5,3), degree-at-least-three sector-2 variables and exact-lift label screening should raise the parent's five-code X bottleneck beyond weights 6-13 while retaining the 0.150 rate cell.")
GROUP = ("Frozen Z19:Z3(s=7), order 57: this bed scored 15.7, while the same group on (3,4)x(3,4) scored 105.2. Those provisional upper-bound readings motivate testing filling, not assuming group inadequacy.")
SHAPES = ("Frozen (4,2)x(5,3): n=26*57=1482, q_max=57, and k>=4*57=228, rate>=0.15385. Transposed component checks are 2x4 and 3x5, with variable lengths 228 and 285 and nominal rates 1/2 and 2/5. Parent minimum X witnesses occupied sector-2 Arow/Brow blocks, including all twenty blocks at weight 13 in slot 43.")
FILLING = ("Twenty explicit degree-budget profiles replace rolling. All A and B variable degrees are at least three. Four coset policies, exact-lift four-cycle optimization, rank screening, and randomized component-kernel witness screening select new labels. Component witnesses are heuristic upper bounds, not certified logical distances; prefer balanced witnesses and k near the bound.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m19r3s7_42x53_degree_girth_isd"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    # metacyclic Z_19 x| Z_3: y x y^-1 = x^7 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 19, 3, 7
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (4, 2)               # (ra, ca)
B_SHAPE = (5, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Deliberate support portfolios; all screening uses verified lifts."""
    import math

    # First eight profiles prioritize A, the parent's X bottleneck.
    # The remaining profiles spend more budget on B or redistribute
    # A degrees. No profile is produced by permuting another labeling.
    a_profiles = [
        [[2,2],[2,2],[2,3],[3,2]],
        [[1,3],[3,1],[2,3],[3,2]],
        [[1,4],[4,1],[1,3],[3,1]],
        [[1,4],[3,2],[2,2],[3,1]],
        [[2,3],[2,3],[2,2],[3,1]],
        [[1,5],[4,1],[2,2],[2,1]],
        [[1,3],[3,1],[2,2],[3,2]],
        [[1,4],[3,1],[2,2],[3,1]],
        [[1,3],[3,1],[1,3],[3,1]],
        [[1,3],[3,1],[2,2],[2,2]],
        [[2,2],[2,2],[2,2],[2,2]],
        [[1,4],[3,1],[2,2],[2,1]],
        [[1,4],[4,1],[1,2],[2,1]],
        [[1,5],[3,1],[2,1],[2,1]],
        [[1,2],[2,1],[2,2],[2,2]],
        [[1,2],[2,1],[1,3],[3,1]],
        [[1,2],[1,2],[2,2],[3,1]],
        [[1,2],[2,1],[1,2],[3,2]],
        [[1,3],[3,1],[2,2],[2,1]],
        [[1,4],[3,1],[2,1],[2,1]],
    ]
    p = idx % len(a_profiles)
    sa = [row[:] for row in a_profiles[p]]
    if p < 8:
        sb = [[1,1,1] for _ in range(5)]
    elif p == 13:
        sb = [[2,1,1],[1,2,1],[1,1,2],[1,1,1],[1,1,1]]
    elif 14 <= p <= 16:
        sb = [[2,2,1],[1,2,2],[2,1,2],[2,2,1],[1,1,2]]
    elif p == 17:
        sb = [[2,2,1],[1,2,2],[2,1,1],[1,1,1],[1,1,2]]
    else:
        sb = [[2 if j == (i+p) % 3 else 1 for j in range(3)]
              for i in range(5)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    counts = {"A": sa, "B": sb}

    # Different policies change actual within-cell coset structure.
    # Even homogeneous pairs are confined to alternating cells, so
    # a two-cell A row never has both pairs homogeneous under mode 3.
    mode = (p + idx // 20) % 4
    pools = {None: list(G.elements)}
    for b in range(3):
        pools[b] = [g for g in G.elements if g[1] == b]
    locks = {}
    positions = []
    for side in ("A", "B"):
        locks[side] = []
        for i, row in enumerate(counts[side]):
            lr = []
            for j, count in enumerate(row):
                phase = rng.randrange(3)
                if mode == 0:
                    bs = [None] * count
                elif mode == 2 and count >= 3 and count % 2:
                    bs = [phase] * count
                elif mode == 3 and count == 2 and (i+j) % 2 == 0:
                    bs = [phase] * count
                else:
                    bs = [(phase+t) % 3 for t in range(count)]
                lr.append(bs)
                positions.extend((side, i, j, t) for t in range(count))
            locks[side].append(lr)

    def fresh():
        cells = {}
        for side in ("A", "B"):
            grid = []
            for row in locks[side]:
                rr = []
                for bs in row:
                    cell = []
                    for b in bs:
                        g = rng.choice(pools[b])
                        while g in cell:
                            g = rng.choice(pools[b])
                        cell.append(g)
                    rr.append(cell)
                grid.append(rr)
            cells[side] = grid
        return cells

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    n1 = ca * cb * q
    mask = (1 << q) - 1

    def components(hx, hz):
        # Extract A^T and B^T component checks from the VERIFIED
        # sector-2 blocks. This is projection, not a separate lift.
        ar = []
        for j in range(ca):
            for r in range(q):
                z = hz[(j*rb)*q+r] >> n1
                v = 0
                for i in range(ra):
                    v |= ((z >> (i*rb*q)) & mask) << (i*q)
                ar.append(v)
        br = [x >> n1 for x in hx[:cb*q]]
        return ar, br

    def four_cycles(rows):
        total = 0
        for i, x in enumerate(rows):
            for y in rows[:i]:
                t = (x & y).bit_count()
                total += t * (t-1) // 2
        return total

    def inspect(cells):
        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        ar, br = components(hx, hz)
        cost = 3 * four_cycles(ar) + 2 * four_cycles(br)
        sig = tuple(tuple(sorted(G.idx(g) for g in cell))
                    for side in ("A", "B")
                    for row in cells[side] for cell in row)
        return (cost, sig, cells, hx, hz, n, ar, br)

    elite = []

    def remember(item):
        if any(item[1] == old[1] for old in elite):
            return
        elite.append(item)
        elite.sort(key=lambda x: x[0])
        del elite[5:]

    for _ in range(12):
        remember(inspect(fresh()))
    current = elite[0]

    # Local label optimization preserves each deliberate support and
    # its coset policy; it does not submit orbit/permutation copies.
    for step in range(96):
        cells = {side: [[cell[:] for cell in row]
                        for row in current[2][side]]
                 for side in ("A", "B")}
        for _ in range(2 if step % 7 == 0 else 1):
            side, i, j, t = rng.choice(positions)
            cell = cells[side][i][j]
            b = locks[side][i][j][t]
            choices = [g for g in pools[b] if g not in cell]
            cell[t] = rng.choice(choices)
        proposal = inspect(cells)
        remember(proposal)
        delta = proposal[0] - current[0]
        temperature = 2.0 * (1.0-step/96.0) + 0.05
        if delta <= 0 or rng.random() < math.exp(
                -delta / (q * temperature)):
            current = proposal

    def kernel_witness(rows, width):
        # Nullspace vectors of the exact projected component checks.
        # Random systematic generator bases and their pair sums yield
        # witness UPPER bounds only; no distance certification.
        pivots = {}
        for row in rows:
            while row:
                c = row.bit_length() - 1
                if c in pivots:
                    row ^= pivots[c]
                else:
                    pivots[c] = row
                    break
        equations = sorted(pivots.items())
        basis = []
        for c in range(width):
            if c in pivots:
                continue
            v = 1 << c
            for lead, row in equations:
                if (row & v).bit_count() & 1:
                    v ^= 1 << lead
            basis.append(v)
        if not basis:
            return width + 1
        best = min(v.bit_count() for v in basis)
        for _ in range(12):
            work = basis[:]
            columns = list(range(width))
            rng.shuffle(columns)
            used = 0
            for c in columns:
                bit = 1 << c
                pivot = next((i for i in range(used, len(work))
                              if work[i] & bit), None)
                if pivot is None:
                    continue
                work[used], work[pivot] = work[pivot], work[used]
                v = work[used]
                for i in range(len(work)):
                    if i != used and work[i] & bit:
                        work[i] ^= v
                used += 1
                if used == len(work):
                    break
            best = min(best, min(v.bit_count() for v in work))
            for i, v in enumerate(work):
                for u in work[:i]:
                    best = min(best, (v ^ u).bit_count())
            if best <= 8:
                break
        return best

    ranked = []
    for item in elite:
        cost, sig, cells, hx, hz, n, ar, br = item
        k = n - gf2_rank(hx) - gf2_rank(hz)
        ranked.append((k, item))
    # Favor the intended rate band, not rank-deficit inflation.
    eligible = [(k, item) for k, item in ranked if 228 <= k <= 252]
    if not eligible:
        eligible = [min(ranked, key=lambda x: abs(x[0]-228))]

    chosen = None
    best_score = None
    for k, item in eligible:
        da = kernel_witness(item[6], ra*q)
        db = kernel_witness(item[7], rb*q)
        score = (k * min(da, db)**2 / item[5],
                 da + db, -item[0])
        if best_score is None or score > best_score:
            best_score = score
            chosen = item[2]
    return {"idx": idx, "cells": chosen}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the jointly designed and screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
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
