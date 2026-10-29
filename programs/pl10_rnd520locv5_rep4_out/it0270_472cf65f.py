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
W_MAX = min(int(os.environ.get("CM_WMAX", "12")), 12)  # row-weight cap drawn from [9, 10, 11, 12] (random arm)
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 975348679


# EVOLVE-BLOCK-START
CONCEPT = ("Random-mutation arm (local moves), not designed: m6r2s1 of order 12, shapes (2,6)x(2,7), row-weight cap 12; this child was drawn at card level 1.")
GROUP = ("Drawn at random by the random-mutation arm (rnd_mutator.py v5, local moves); no reason.")
SHAPES = ("Drawn at random by the random-mutation arm (rnd_mutator.py v5, local moves); no reason.")
FILLING = ("NEW_CTX rolls a legal support grid per slot via roll_supports; CELL fills each cell with distinct uniform elements. Slots differ in counts and elements alike.")
FAMILY_NAME = "rndloc_m6r2s1_2-6x2-7_w12"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    # metacyclic Z_6 x| Z_2: y x y^-1 = x^1 (s^r = 1 mod m);
    # drawn by the random-mutation arm (rnd_mutator.py), not designed
    m, r, s = 6, 2, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Per-slot shared state, built once before the cells are filled.
    roll_supports(rng) (scaffold helper) returns a random weight-legal
    (supports_A, supports_B) pair with every cell >= 1 term (zero
    blocks are legal per slot but the roller never emits them); replace
    or post-process it freely - the scaffold re-validates every slot."""
    sa, sb = roll_supports(rng)
    return {"idx": idx, "sup": {"A": sa, "B": sb}}


def CELL(rng, G, side, i, j, ctx):
    """Return the FULL cell (i, j) of side "A"/"B": a list of 0..5
    DISTINCT group elements (empty list = zero block). G.pick(rng)
    draws a uniform element; G.mul/G.inv/G.e/G.elements are the group."""
    cell = []
    for _ in range(ctx["sup"][side][i][j]):
        g = G.pick(rng)
        while any(G._key(g) == G._key(h) for h in cell):
            g = G.pick(rng)
        cell.append(g)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# ---- local moves (random-mutation arm, RND_LOCAL=1; inserted once by rnd_mutator.py) ----
# LOCAL_BASE: the cells of the parent's best measured code in this program's element encoding,
# or None. With a base of shapes A_SHAPE x B_SHAPE every slot is that code after k local edits,
# k uniform in LOCAL_K. One edit: a cell uniform over all cells of A and B, an operation uniform
# over replace / add / remove, the element removed or replaced uniform over the cell, the new
# element uniform over the group elements not in the cell. An edit that breaks a slot rule
# (0..CELL_MAX distinct elements per cell, row weight <= W_MAX, no all-zero protograph row or
# column) or does not take the slot one edit further from the base is redrawn; a slot whose drawn
# number of edits cannot be completed within LOCAL_TRIES redraws keeps the edits it has (k_done in
# LOCAL_LOG), and only a slot with no legal edit at all is voided. A replacement
# that only renames qubits and checks is redrawn, and a slot whose code is the base with renamed
# qubits and checks is redrawn as a whole (_rndloc_renamed). All draws use the slot's rng.
# Without a base the program's own NEW_CTX / CELL, kept here, fill every slot.
LOCAL_BASE = {"A":[[[[4,1]],[[4,0]],[[0,0]],[[2,0]],[[2,0]],[[1,1]]],[[[1,1]],[[3,1]],[[2,1]],[[1,0],[3,1]],[[2,0]],[[5,0]]]],"B":[[[[0,1]],[[5,1]],[[2,0]],[[3,0]],[[4,0]],[[0,1]],[[3,1]]],[[[5,1]],[[5,0],[5,1]],[[5,0]],[[5,0]],[[5,1]],[[4,1]],[[2,0],[3,0]]]]}
LOCAL_K = (1, 2, 3)
LOCAL_TRIES = 1000
LOCAL_RENAME_BUDGET = 200000
LOCAL_LOG = []
_rndloc_own_NEW_CTX = NEW_CTX
_rndloc_own_CELL = CELL
_rndloc_tabs = {}


class _RndlocCtx(dict):
    pass


def _rndloc_tables(G):
    t = _rndloc_tabs.get(id(G))
    if t is None:
        els = G.elements
        t = ([[G.idx(G.mul(a, b)) for b in els] for a in els], [G.idx(G.inv(a)) for a in els])
        _rndloc_tabs[id(G)] = t
    return t


def _rndloc_renamed(G, M1, M2):
    """True when M2[r][c] = {x_r * g * y_c : g in M1[r][c]} for all cells, for some group elements
    x_r (one per protograph row) and y_c (one per protograph column). Under the lift of
    lifted_matrices this holds exactly when M2 is M1 up to a renaming of qubits and checks
    within their blocks, so A or B can be replaced by M2 without changing the code. None when the
    search takes more than LOCAL_RENAME_BUDGET cell checks."""
    mul, inv = _rndloc_tables(G)
    R, C = len(M1), len(M1[0])
    S1 = [[[G.idx(g) for g in cell] for cell in row] for row in M1]
    S2 = [[frozenset(G.idx(g) for g in cell) for cell in row] for row in M2]
    if any(len(set(S1[r][c])) != len(S2[r][c]) for r in range(R) for c in range(C)):
        return False
    nb = [[] for _ in range(R + C)]          # vertices: rows 0..R-1, columns R..R+C-1
    for r in range(R):
        for c in range(C):
            if S1[r][c]:
                nb[r].append(R + c)
                nb[R + c].append(r)
    val = [None] * (R + C)
    work = [0]

    def holds(r, c):
        work[0] += 1
        x, y = val[r], val[R + c]
        return frozenset(mul[mul[x][g]][y] for g in S1[r][c]) == S2[r][c]

    def options(v):
        for w in nb[v]:
            if val[w] is not None:
                if v < R:                    # row v from column w: x = b * (g0 * y)^-1
                    g0, y = S1[v][w - R][0], val[w]
                    return [mul[b][inv[mul[g0][y]]] for b in S2[v][w - R]]
                g0, x = S1[w][v - R][0], val[w]   # column v from row w: y = (x * g0)^-1 * b
                return [mul[inv[mul[x][g0]]][b] for b in S2[w][v - R]]
        return range(len(inv))

    def solve(order, i):
        if i == len(order):
            return True
        if work[0] > LOCAL_RENAME_BUDGET:
            return None
        v = order[i]
        for a in options(v):
            val[v] = a
            if all(val[w] is None or (holds(v, w - R) if v < R else holds(w, v - R)) for w in nb[v]):
                got = solve(order, i + 1)
                if got is not False:
                    return got
        val[v] = None
        return False

    seen = set()
    for v0 in range(R + C):
        if v0 in seen or not nb[v0]:
            continue
        order = [v0]
        seen.add(v0)
        for v in order:
            for w in nb[v]:
                if w not in seen:
                    seen.add(w)
                    order.append(w)
        got = solve(order, 0)
        if got is not True:
            return got
    return True


def _rndloc_key(x):
    return tuple(_rndloc_key(y) for y in x) if isinstance(x, (list, tuple)) else x


def _rndloc_dist(G, base_cell, cell):
    b = set(G._key(g) for g in base_cell)
    c = set(G._key(g) for g in cell)
    return max(len(b - c), len(c - b))


def _rndloc_base(G):
    """LOCAL_BASE with G's own element objects; None without a base or when its grid shapes are
    not A_SHAPE / B_SHAPE; an element outside G is an error (every slot is voided)."""
    if LOCAL_BASE is None:
        return None
    for side, (rows, cols) in (("A", A_SHAPE), ("B", B_SHAPE)):
        grid = LOCAL_BASE.get(side)
        if not isinstance(grid, list) or len(grid) != rows or any(len(r) != cols for r in grid):
            return None
    els = {_rndloc_key(g): g for g in G.elements}
    out = {}
    for side in ("A", "B"):
        out[side] = []
        for row in LOCAL_BASE[side]:
            out[side].append([])
            for cell in row:
                miss = [x for x in cell if _rndloc_key(x) not in els]
                if miss:
                    raise ValueError("LOCAL_BASE element %r is not in the group" % (miss[0],))
                out[side][-1].append([els[_rndloc_key(x)] for x in cell])
    return out


def NEW_CTX(rng, G, idx):
    base = _rndloc_base(G)
    if base is None:
        return _rndloc_own_NEW_CTX(rng, G, idx)
    where = [(s, i, j) for s in ("A", "B") for i in range(len(base[s])) for j in range(len(base[s][i]))]
    k = rng.choice(LOCAL_K)
    redrawn, unchecked = {}, 0
    while True:
        cells = {s: [[list(c) for c in row] for row in base[s]] for s in ("A", "B")}
        sup = {s: [[len(c) for c in row] for row in cells[s]] for s in ("A", "B")}
        ops, at = [], []
        while len(ops) < k:
            if sum(redrawn.values()) >= LOCAL_TRIES:
                if ops:
                    break                # fewer edits than drawn: the slot keeps the edits it has (k_done < k)
                raise ValueError("local move: no legal edit in %d draws" % LOCAL_TRIES)
            s, i, j = rng.choice(where)
            op = rng.choice(("replace", "add", "remove"))
            cell = cells[s][i][j]
            keys = set(G._key(g) for g in cell)
            free = [g for g in G.elements if G._key(g) not in keys]
            new, why = list(cell), None
            if op != "add" and not cell:
                why = "empty cell"
            elif op == "add" and len(cell) >= CELL_MAX:
                why = "full cell"
            elif op != "remove" and not free:
                why = "no free element"
            elif op == "replace":
                new[rng.randrange(len(new))] = rng.choice(free)
            elif op == "add":
                new.append(rng.choice(free))
            else:
                del new[rng.randrange(len(new))]
            if why is None:
                sup[s][i][j] = len(new)
                if _grid_weights(sup["A"], sup["B"]) > W_MAX:
                    why = "weight cap"
                elif not _no_dead_lines(sup["A"], sup["B"]):
                    why = "dead line"
                elif _rndloc_dist(G, base[s][i][j], new) != _rndloc_dist(G, base[s][i][j], cell) + 1:
                    why = "not one edit further"
                elif op == "replace":
                    trial = [list(row) for row in cells[s]]
                    trial[i][j] = new
                    same = _rndloc_renamed(G, cells[s], trial)
                    if same:
                        why = "renames qubits only"
                    elif same is None:
                        unchecked += 1
                if why is not None:
                    sup[s][i][j] = len(cell)
            if why is None:
                cells[s][i][j] = new
                ops.append(op)
                at.append([s, i, j])
            else:
                redrawn[op + ": " + why] = redrawn.get(op + ": " + why, 0) + 1
        same = [_rndloc_renamed(G, base[s], cells[s]) for s in ("A", "B")]
        if None in same:
            unchecked += 1
        if not all(same):
            break
        redrawn["slot: base with renamed qubits"] = redrawn.get("slot: base with renamed qubits", 0) + 1
        if sum(redrawn.values()) >= LOCAL_TRIES:
            raise ValueError("local move: no legal edit in %d draws" % LOCAL_TRIES)
    rec = {"idx": idx, "k": k, "k_done": len(ops), "ops": ops, "at": at, "redrawn": redrawn}
    if unchecked:
        rec["rename_check_over_budget"] = unchecked
    LOCAL_LOG.append(rec)
    return _RndlocCtx(cells=cells)


def CELL(rng, G, side, i, j, ctx):
    if isinstance(ctx, _RndlocCtx):
        return list(ctx["cells"][side][i][j])
    return _rndloc_own_CELL(rng, G, side, i, j, ctx)
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
