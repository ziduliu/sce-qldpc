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
CONCEPT = ("Test whether the generalized quaternion group Q16, of order 16, changes measured circuit performance with shapes (1,3)x(2,7), while retaining the parent's screened support rule.")
GROUP = ("Use Q16, genuinely different from the parent's order-24 binary tetrahedral group. The RUN STATE includes order-16 beds at other shapes, but no Q16 bed at these shapes; those other beds are not evidence of a Q16 advantage. This is a test of a distinct nonabelian regular action.")
SHAPES = ("Keep (1,3)x(2,7): the guaranteed rate is 10/23 ~= 0.435, and order 16 gives n=(3*7+1*2)*16=368, within the size window. The component codes have parameters [3,2] and [7,5]. Parent failures were spread across data blocks with mild enrichment, while measured codes included weight-4 logical witnesses; retaining the shapes isolates the group change.")
FILLING = ("Keep the parent's slot-stratified, up-to-three-term supports and exact lifted-matrix screening for short logicals and 4-cycles. The measured parent codes still had frequent failures, especially in the X basis, despite that screening. This test changes the group action without claiming that the screen directly optimizes pL.")
FAMILY_NAME = "generalized_quaternion_q16_13x27_stratified_w3_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Generalized quaternion group Q16:
    # a^8=1, x^2=a^4, and x*a*x^-1=a^-1.
    # Elements are (i,b), denoting a^i*x^b.
    els = [(i, b) for b in range(2) for i in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 2)

    def mul(g, h):
        i, b = red(g)
        j, d = red(h)
        return ((i + (-j if b else j) + (4 if b and d else 0)) % 8,
                (b + d) % 2)

    def inv(g):
        i, b = red(g)
        return ((-i) % 8, 0) if b == 0 else ((i + 4) % 8, 1)

    def idx(g):
        i, b = red(g)
        return 8 * b + i

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search sparse legal supports and labels using the verified lift."""
    best = None

    def span_data(check, stabilizers, n):
        pivots = {}
        for row in stabilizers:
            v = row
            while v:
                p = v.bit_length() - 1
                if p in pivots:
                    v ^= pivots[p]
                else:
                    pivots[p] = v
                    break

        def in_span(v):
            while v:
                p = v.bit_length() - 1
                if p not in pivots:
                    return False
                v ^= pivots[p]
            return True

        cols = [0] * n
        for r, row in enumerate(check):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return in_span, cols

    def short12(check, stabilizers, n):
        in_span, cols = span_data(check, stabilizers, n)
        w1 = sum(1 for j, s in enumerate(cols)
                 if s == 0 and not in_span(1 << j))
        buckets = {}
        for j, s in enumerate(cols):
            buckets.setdefault(s, []).append(j)
        w2 = 0
        for ids in buckets.values():
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if not in_span((1 << ids[a]) | (1 << ids[b])):
                        w2 += 1
        return w1, w2

    def short3(check, stabilizers, n, cap=512):
        in_span, cols = span_data(check, stabilizers, n)
        buckets = {}
        for j, s in enumerate(cols):
            buckets.setdefault(s, []).append(j)
        count = 0
        for i in range(n - 2):
            bi = 1 << i
            for j in range(i + 1, n - 1):
                pair = bi | (1 << j)
                target = cols[i] ^ cols[j]
                for k in buckets.get(target, ()):
                    if k > j and not in_span(pair | (1 << k)):
                        count += 1
                        if count >= cap:
                            return cap
        return count

    def four_cycles(rows):
        total = 0
        for i, x in enumerate(rows):
            for y in rows[:i]:
                c = _popcount(x & y)
                total += c * (c - 1) // 2
        return total

    # Stratify support density by slot; trials perturb its term budget and
    # placement. Every cell starts nonzero, and each accepted increment
    # preserves the exact evaluator's row-weight cap.
    candidates = []
    slot_extra = (idx * 7) % 11
    for trial in range(36):
        sa = [[1] * A_SHAPE[1] for _ in range(A_SHAPE[0])]
        sb = [[1] * B_SHAPE[1] for _ in range(B_SHAPE[0])]
        extra = max(0, min(10, slot_extra + trial % 5 - 2))
        placed = 0
        for _ in range(extra):
            choices = []
            for side, grid in (("A", sa), ("B", sb)):
                for i in range(len(grid)):
                    for j in range(len(grid[i])):
                        if grid[i][j] >= 3:
                            continue
                        grid[i][j] += 1
                        if _grid_weights(sa, sb) <= min(12, W_MAX):
                            choices.append((side, i, j))
                        grid[i][j] -= 1
            if not choices:
                break
            side, i, j = rng.choice(choices)
            (sa if side == "A" else sb)[i][j] += 1
            placed += 1
        if placed != extra or not _no_dead_lines(sa, sb):
            continue

        A = [[rng.sample(G.elements, sa[i][j])
              for j in range(A_SHAPE[1])] for i in range(A_SHAPE[0])]
        B = [[rng.sample(G.elements, sb[i][j])
              for j in range(B_SHAPE[1])] for i in range(B_SHAPE[0])]
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue

        x1, x2 = short12(hz, hx, n)
        z1, z2 = short12(hx, hz, n)
        c4 = four_cycles(hx) + four_cycles(hz)
        gates = sum(_popcount(row) for row in hx + hz)
        row_weight = _grid_weights(sa, sb)
        terms = sum(map(sum, sa)) + sum(map(sum, sb))
        prekey = (x1 + z1, x2 + z2, gates, c4,
                  abs(k - 245), row_weight, terms, rng.random())
        candidates.append((prekey, A, B, hx, hz, n, k, gates, c4,
                           row_weight, terms))

    if not candidates:
        raise ValueError("could not construct a positive-k legal filling")

    # Weight-three logicals were still present in the parent's measured
    # codes, so evaluate them on the strongest inexpensive pre-screened set.
    candidates.sort(key=lambda item: item[0])
    best = None
    for prekey, A, B, hx, hz, n, k, gates, c4, row_weight, terms \
            in candidates[:10]:
        x3 = short3(hz, hx, n)
        z3 = short3(hx, hz, n)
        key = (prekey[0], prekey[1], x3 + z3, gates, c4,
               row_weight, abs(k - 245), terms, rng.random())
        if best is None or key < best[0]:
            best = (key, {"A": A, "B": B})

    return {"idx": idx, "cells": best[1]}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell for this slot."""
    return ctx["cells"][side][i][j]

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: unconstrained rolled fillings at row weight 12 did not avoid weight-2 logicals or frequent measured X-basis failures in the reported slots.
# Refuted: screening out only weight-1/2 logicals was not sufficient for low measured pL; selected codes still had weight-3 logical witnesses and frequent circuit failures.
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
