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
CONCEPT = ("Metacyclic Z1:Z29, s=1, order 29, shapes (3,6)x(3,7): three-row components with column degrees three/four and verified lifted short-cycle screening should suppress the parent's weight-8 sector-1 logicals and improve five-code fitness.")
GROUP = ("Keep the parent's cyclic metacyclic construction rule and trivial action; resize only r from 62 to 29 for the new shape. Factor 51 gives n=1479 at q=29. This bed is untested; no distance improvement is asserted as established.")
SHAPES = ("A=(3,6), B=(3,7) give k>=29*3*4=348 and guaranteed rate 12/51=0.235294, targeting cell 0.250. Component lengths are 174 and 203 with three check-block rows, replacing the parent's two-row components. Parent measured minimum logicals were confined to sector-1 Acol/Bcol blocks. Screen exact lifted k to remain in the target band.")
FILLING = ("Twenty density profiles: 2..6 doubled A cells crossed with 0..3 doubled B cells, at most two extras per A row, one per B row, and one per column. A-row/B-row maxima are eight and column maxima four, so both CSS check-weight maxima are <=12. Optimize labels against four/six cycles extracted from verified lifted matrices; reject component kernel words of weight <=4 and out-of-band exact k. These screens are not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r025_m1r29s1_36x37_degree34_cycle_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Same cyclic metacyclic construction rule; resized for factor 51.
    # Z_1 x| Z_29 with trivial action.
    m, r, s = 1, 29, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 6)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def _copy_cells(cells):
    return [[list(cell) for cell in row] for row in cells]


def _component_rows(hx, hz, q, side):
    """Extract components from VERIFIED matrices, without a separate lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "B":
        # j=0 HZ rows; retain only their sector-1 B blocks.
        mask = (1 << (cb * q)) - 1
        return [hz[a * q + r] & mask
                for a in range(rb) for r in range(q)]
    # beta=0 HX rows; compact the selected A-column blocks.
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
    """Exact four/six-cycle counts of the extracted bipartite component."""
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
        neighbors = upper[a]
        for t, b in enumerate(neighbors):
            ab = common[a][b]
            for c in neighbors[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    # Exclude triangles that reuse a variable node.
                    c6 += ab * ac * bc - triple * (ab + ac + bc - 2)
    return penalty * c4 + c6


def _verified_costs(G, A, B, penalty, side=None):
    hx, hz, _ = lifted_matrices(G, A, B)
    if side is not None:
        return _cycle_cost(_component_rows(hx, hz, G.q, side), penalty)
    return [_cycle_cost(_component_rows(hx, hz, G.q, s), penalty)
            for s in ("A", "B")]


def _tiny_component_kernel(rows, width):
    """Exact detection of component kernel words of weight at most four."""
    columns = [0] * width
    for r, row in enumerate(rows):
        remaining = row
        while remaining:
            bit = remaining & -remaining
            columns[bit.bit_length() - 1] |= 1 << r
            remaining ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return True
    pairs = set()
    for a in range(width):
        for b in range(a + 1, width):
            syndrome = columns[a] ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                return True
            pairs.add(syndrome)
    return False


def _density_profile(rng, idx):
    """Twenty genuinely different density prescriptions, not orbit copies."""
    design = idx % 20
    extra_a = 2 + design % 5
    extra_b = design // 5
    sa = [[1] * A_SHAPE[1] for _ in range(A_SHAPE[0])]
    sb = [[1] * B_SHAPE[1] for _ in range(B_SHAPE[0])]
    a_columns = list(range(A_SHAPE[1]))
    b_columns = list(range(B_SHAPE[1]))
    a_rows = list(range(A_SHAPE[0]))
    b_rows = list(range(B_SHAPE[0]))
    rng.shuffle(a_columns)
    rng.shuffle(b_columns)
    rng.shuffle(a_rows)
    rng.shuffle(b_rows)
    for t in range(extra_a):
        sa[a_rows[t % 3]][a_columns[t]] += 1
    for t in range(extra_b):
        sb[b_rows[t]][b_columns[t]] += 1
    assert _grid_weights(sa, sb) <= 12
    return sa, sb


def _initial_cells(rng, G, supports):
    return [[rng.sample(G.elements, count) for count in row]
            for row in supports]


def NEW_CTX(rng, G, idx):
    """Design supports, optimize labels, then screen the exact built code."""
    import math

    sa, sb = _density_profile(rng, idx)
    penalty = 320 + 20 * (idx % 4)
    for attempt in range(4):
        # Two starts compete within this slot's prescribed density profile.
        starts = []
        for _ in range(2):
            A = _initial_cells(rng, G, sa)
            B = _initial_cells(rng, G, sb)
            costs = _verified_costs(G, A, B, penalty)
            starts.append((sum(costs), A, B, costs))
        _, A, B, costs = min(starts, key=lambda item: item[0])
        current = sum(costs)
        best_score = current
        best_A, best_B = _copy_cells(A), _copy_cells(B)

        for step in range(260):
            which = rng.randrange(2)
            side = ("A", "B")[which]
            grid = A if which == 0 else B
            changes = []
            # Occasional coordinated moves help escape single-label plateaus.
            moves = 2 if step % 7 == 0 else 1
            for _ in range(moves):
                i = rng.randrange(len(grid))
                j = rng.randrange(len(grid[0]))
                cell = grid[i][j]
                t = rng.randrange(len(cell))
                old = cell[t]
                new = G.pick(rng)
                while any(G._key(new) == G._key(g) for g in cell):
                    new = G.pick(rng)
                changes.append((cell, t, old))
                cell[t] = new

            proposed_cost = _verified_costs(G, A, B, penalty, side)
            proposed = current - costs[which] + proposed_cost
            temperature = G.q * (20.0 * (1.0 - step / 260.0) ** 2 + 0.35)
            accept = proposed <= current
            if not accept:
                accept = rng.random() < math.exp((current - proposed) /
                                                 temperature)
            if accept:
                current = proposed
                costs[which] = proposed_cost
                if current < best_score:
                    best_score = current
                    best_A, best_B = _copy_cells(A), _copy_cells(B)
            else:
                for cell, t, old in reversed(changes):
                    cell[t] = old

        A, B = best_A, best_B
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.225 * n <= k < 0.275 * n):
            continue
        ar = _component_rows(hx, hz, G.q, "A")
        br = _component_rows(hx, hz, G.q, "B")
        if _tiny_component_kernel(ar, A_SHAPE[1] * G.q):
            continue
        if _tiny_component_kernel(br, B_SHAPE[1] * G.q):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS check")
        return {"idx": idx, "cells": {"A": A, "B": B},
                "screen_k": k, "cycle_cost": best_score}
    raise ValueError("no in-band candidate passed the component screen")


def CELL(rng, G, side, i, j, ctx):
    """Return the already optimized full cell, with distinct elements."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling alone excludes very short logicals: the parent had d2s=2 slots and measured top-five distances 10,8,8,8,8, with minimum witnesses confined to sector 1.
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
