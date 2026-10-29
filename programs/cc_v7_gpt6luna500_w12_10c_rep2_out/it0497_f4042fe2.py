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
CONCEPT = ("For nonabelian C13 semidirect C9 of order 117, with yxy^-1=x^3, on A=(1,2), B=(1,2), screened quadratic fiber labelings will improve the five-code fitness in the 0.200 rate cell.")
GROUP = ("Jump from the parent's C46 to C13 semidirect C9, yxy^-1=x^3. RUN STATE reports fitness 111.4 for the Z13:Z9(s=3), q=117, (1,2)x(1,2) bed; its frontier codes have k=121 and measured distances 23-24.")
SHAPES = ("Use (1,2)x(1,2): the rank-count bound gives k>=117 and rate >=117/585=0.200, and q=117 gives n=585. Equal one-row, two-column protographs give matching component profiles on the X and Z sides, with sector sizes 468 and 117. The parent's lightest logicals touched all nine Acol/Bcol blocks; this compact shape has four such blocks.")
FILLING = ("Put four distinct elements in each cell using quadratic sequences in both coordinates. Require the four cells together to cover all nine C9 fibers, with no fiber appearing more than three times. Each check-row sum is at most 12. Screen verified lifts for CSS, positive k, logicals through weight 4, and 4-cycles; this tests whether structured fiber labels improve on the parent's weight-22/23 witnesses.")
FAMILY_NAME = "r020_m13r9s3_12x12_quadratic_fiber_l4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 117

def custom_ops(q):
    # C13 semidirect C9: y x y^-1 = x^3.
    m, r, s = 13, 9, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        return (g[0] % m) + m * (g[1] % r)

    def red(g):
        return (g[0] % m, g[1] % r)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Screen quadratic C9-fiber labelings with the verified lifted matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    def make_candidate():
        # Each of the four cells follows its own quadratic sequence in b.
        # Reject repeated fibers within a cell and unbalanced global coverage.
        for _ in range(100):
            cells = {"A": [], "B": []}
            all_b = []
            for side in ("A", "B"):
                grid = []
                for _j in range(2):
                    for _attempt in range(80):
                        b0 = rng.randrange(9)
                        db = rng.randrange(1, 9)
                        qb = rng.randrange(9)
                        bs = [(b0 + db*t + qb*t*(t-1)//2) % 9
                              for t in range(4)]
                        if len(set(bs)) == 4:
                            break
                    else:
                        break

                    a0 = rng.randrange(13)
                    da = rng.randrange(1, 13)
                    qa = rng.randrange(13)
                    cell = []
                    for t, b in enumerate(bs):
                        a = (a0 + da*t + qa*t*(t-1)//2) % 13
                        cell.append((a, b))
                    grid.append(cell)
                    all_b.extend(bs)
                if len(grid) != 2:
                    break
                cells[side].append(grid)

            if len(cells["A"]) != 1 or len(cells["B"]) != 1:
                continue
            freq = [all_b.count(b) for b in range(9)]
            if min(freq) >= 1 and max(freq) <= 3:
                return cells
        return None

    def basis(rows):
        piv = {}
        for v in rows:
            while v:
                p = v.bit_length() - 1
                if p in piv:
                    v ^= piv[p]
                else:
                    piv[p] = v
                    break
        return piv

    def in_span(v, piv):
        while v:
            p = v.bit_length() - 1
            if p not in piv:
                return False
            v ^= piv[p]
        return True

    def logical_floor(checks, stabilizers, n):
        piv = basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        by_sig = {}
        for i, sig in enumerate(cols):
            if sig == 0 and not in_span(1 << i, piv):
                return 1
            for j in by_sig.get(sig, ()):
                if not in_span((1 << i) | (1 << j), piv):
                    return 2
            by_sig.setdefault(sig, []).append(i)

        for i in range(n):
            for j in range(i + 1, n):
                for k in by_sig.get(cols[i] ^ cols[j], ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k), piv):
                        return 3
        return 4

    def has_weight4(checks, stabilizers, n):
        piv = basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                sig = cols[i] ^ cols[j]
                old = pairs.get(sig, ())
                for a, b in old:
                    if a != i and a != j and b != i and b != j:
                        mask = (1 << a) | (1 << b) | (1 << i) | (1 << j)
                        if not in_span(mask, piv):
                            return True
                pairs.setdefault(sig, []).append((i, j))
        return False

    def four_cycles(rows):
        seen = {}
        total = 0
        for row in rows:
            bits = []
            while row:
                bit = row & -row
                bits.append(bit.bit_length() - 1)
                row ^= bit
            for a in range(len(bits)):
                for b in range(a):
                    pair = (bits[b], bits[a])
                    old = seen.get(pair, 0)
                    total += old
                    seen[pair] = old + 1
        return total

    candidates = []
    for _trial in range(10):
        cells = make_candidate()
        if cells is None:
            continue
        A, B = cells["A"], cells["B"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > W_MAX or
                not _no_dead_lines(sa, sb)):
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        dx = logical_floor(hz, hx, n)
        dz = logical_floor(hx, hz, n)
        c4 = four_cycles(hx) + four_cycles(hz)
        candidates.append({"cells": cells, "hx": hx, "hz": hz, "n": n,
                           "k": k, "dx": dx, "dz": dz, "c4": c4,
                           "tie": rng.random()})

    if not candidates:
        raise ValueError("no legal positive-k screened filling")

    def in_rate_cell(c):
        rate = c["k"] / c["n"]
        return 0.175 <= rate <= 0.225

    finalists = sorted(
        candidates,
        key=lambda c: (int(in_rate_cell(c)), min(c["dx"], c["dz"]),
                      c["dx"] + c["dz"], c["k"], -c["c4"], c["tie"]),
        reverse=True
    )[:3]

    for c in finalists:
        c["dx_est"], c["dz_est"] = c["dx"], c["dz"]
        if c["dx"] == 4:
            c["dx_est"] = 4 if has_weight4(c["hz"], c["hx"], c["n"]) else 5
        if c["dz"] == 4:
            c["dz_est"] = 4 if has_weight4(c["hx"], c["hz"], c["n"]) else 5

    best = max(finalists, key=lambda c: (
        int(in_rate_cell(c)),
        min(c["dx_est"], c["dz_est"]),
        c["k"] * min(c["dx_est"], c["dz_est"]) ** 2,
        c["dx_est"] + c["dz_est"], -c["c4"], c["tie"]
    ))
    return {"idx": idx, "cells": best["cells"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the full quadratic-sequence cell selected by NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# The measured (1,2)x(4,5) parent produced weight-2 sector-1 Z logicals across all Acol0-1/Bcol0-4 blocks in every measured slot; narrowing A alone did not remove those witnesses.
# The order-46 diverse-difference family scored 50.1428 below its 52.0886 parent; its difference-class rule did not improve the five-code mean.
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
