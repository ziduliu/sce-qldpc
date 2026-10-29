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
CONCEPT = ("Falsifiable: for the dihedral group D23 of order 46 on A=(2,3), B=(2,3), balanced supports with rotation/reflection offsets selected by logical screening through weight 4 will raise mean fitness above 34.1076 in the 0.100 rate cell.")
GROUP = ("Keep D23 of order 46 as required by this L1 round. The parent’s order-46 D23-type bed on these shapes scored 34.1; the related cyclic order-46 bed scored 52.1, motivating a filling search at this lift order without changing the group.")
SHAPES = ("Keep (2,3)x(2,3): the guaranteed rate is 46/598, about 0.077, with q=46 and n=598. Equal shapes give equal-length nominal [3,1] component profiles on the X and Z sides. The parent's lightest measured logicals were in sector 1 and appeared across the Acol/Bcol blocks, so retain the full 3x3 block layout.")
FILLING = ("Compare an all-two-term support profile with profiles whose rows are permutations of (1,2,3); both satisfy the row-weight cap. Label each cell by a random base times distinct relative offsets, balanced between rotations and reflections. Screen verified lifts for CSS, positive k, logicals through weight 3, and 4-cycles; check weight 4 for the two best candidates. This targets the parent's measured weight-20-to-24 logicals and often-boundary k without treating the screen as a distance guarantee.")
FAMILY_NAME = "r010_D23_q46_23x23_balanced_offset_l4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # Dihedral group D23: x^23 = y^2 = 1 and yxy^-1 = x^-1.
    m = 23
    els = [(a, b) for b in range(2) for a in range(m)]

    def mul(g, h):
        sign = 1 if g[1] == 0 else -1
        return ((g[0] + sign * h[0]) % m, (g[1] + h[1]) % 2)

    def inv(g):
        sign = 1 if g[1] == 0 else -1
        return ((-sign * g[0]) % m, g[1])

    def idx(g):
        return g[0] + m * g[1]

    def red(g):
        return (g[0] % m, g[1] % 2)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Select a screened, structured filling for this slot."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    perms = ((1, 2, 3), (1, 3, 2), (2, 1, 3),
             (2, 3, 1), (3, 1, 2), (3, 2, 1))
    rotations = [g for g in G.elements if g[1] == 0 and g != G.e]
    reflections = [g for g in G.elements if g[1] == 1]

    def support_grids(trial):
        if trial == 0:
            return ([[2] * ca for _ in range(ra)],
                    [[2] * cb for _ in range(rb)])
        sa = [list(perms[(idx + trial + 2 * i) % 6]) for i in range(ra)]
        sb = [list(perms[(idx + 2 * trial + 3 + 2 * a) % 6])
              for a in range(rb)]
        return sa, sb

    def make_cells(grid, side, trial):
        used = set()
        out = []
        side_bit = int(side == "B")
        cell_no = 0
        for row in grid:
            rr = []
            for count in row:
                offsets = []
                for t in range(count - 1):
                    want_reflection = (idx + trial + side_bit + cell_no + t) % 2
                    pool = reflections if want_reflection else rotations
                    choices = [h for h in pool if G.idx(h) not in used]
                    if not choices:
                        pool = rotations if want_reflection else reflections
                        choices = [h for h in pool if G.idx(h) not in used]
                    h = rng.choice(choices)
                    used.add(G.idx(h))
                    offsets.append(h)
                base = G.pick(rng)
                rr.append([base] + [G.mul(base, h) for h in offsets])
                cell_no += 1
            out.append(rr)
        return out

    def row_basis(rows):
        piv = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = piv.get(p)
                if old is None:
                    piv[p] = row
                    break
                row ^= old
        return piv

    def in_span(v, piv):
        while v:
            p = v.bit_length() - 1
            old = piv.get(p)
            if old is None:
                return False
            v ^= old
        return True

    def logical_floor(checks, stabilizers, n):
        piv = row_basis(stabilizers)
        synd = [0] * n
        for r, row in enumerate(checks):
            bit = 1 << r
            while row:
                low = row & -row
                synd[low.bit_length() - 1] |= bit
                row ^= low

        by_sig = {}
        for i, sig in enumerate(synd):
            if sig == 0 and not in_span(1 << i, piv):
                return 1
            ids = by_sig.setdefault(sig, [])
            for j in ids:
                if not in_span((1 << i) | (1 << j), piv):
                    return 2
            ids.append(i)

        for i in range(n):
            for j in range(i + 1, n):
                for k in by_sig.get(synd[i] ^ synd[j], ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k), piv):
                        return 3
        return 4

    def has_weight4_logical(checks, stabilizers, n):
        piv = row_basis(stabilizers)
        synd = [0] * n
        for r, row in enumerate(checks):
            bit = 1 << r
            while row:
                low = row & -row
                synd[low.bit_length() - 1] |= bit
                row ^= low

        pairs = {}
        for i in range(n):
            bi = 1 << i
            for j in range(i + 1, n):
                sig = synd[i] ^ synd[j]
                old = pairs.get(sig)
                if old is None:
                    pairs[sig] = [(i, j)]
                    continue
                for a, b in old:
                    if a != i and a != j and b != i and b != j:
                        v = bi | (1 << j) | (1 << a) | (1 << b)
                        if not in_span(v, piv):
                            return True
                old.append((i, j))
        return False

    def four_cycles(rows):
        seen = {}
        total = 0
        for row in rows:
            bits = []
            while row:
                low = row & -row
                bits.append(low.bit_length() - 1)
                row ^= low
            for x in range(len(bits)):
                for y in range(x):
                    pair = (bits[y], bits[x])
                    old = seen.get(pair, 0)
                    total += old
                    seen[pair] = old + 1
        return total

    candidates = []
    for trial in range(10):
        sa, sb = support_grids(trial)
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            continue
        A = make_cells(sa, "A", trial)
        B = make_cells(sb, "B", trial)
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        dx = logical_floor(hz, hx, n)
        dz = logical_floor(hx, hz, n)
        cycles = four_cycles(hx) + four_cycles(hz)
        d = min(dx, dz)
        candidates.append({
            "cells": {"A": A, "B": B}, "hx": hx, "hz": hz,
            "n": n, "k": k, "dx": dx, "dz": dz, "cycles": cycles,
            "tie": rng.random(),
            "quick": (k * d * d, d, k, -cycles)
        })

    if not candidates:
        raise ValueError("no legal positive-k screened filling")

    finalists = sorted(candidates, key=lambda c: c["quick"], reverse=True)[:2]
    for c in finalists:
        if c["dx"] == 4:
            c["dx"] = 4 if has_weight4_logical(c["hz"], c["hx"], c["n"]) else 5
        if c["dz"] == 4:
            c["dz"] = 4 if has_weight4_logical(c["hx"], c["hz"], c["n"]) else 5

    best = max(finalists, key=lambda c: (
        c["k"] * min(c["dx"], c["dz"]) ** 2,
        min(c["dx"], c["dz"]), c["k"], -c["cycles"], c["tie"]))
    return {"idx": idx, "cells": best["cells"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell for this slot."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
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
