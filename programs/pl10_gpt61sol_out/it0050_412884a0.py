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
CONCEPT = ("GL(2,3), order 48, shapes (1,3)x(1,3): connected degree-three components with relative orders 4,6,8 and screened small-logical multiplicity should improve worse-basis circuit reliability over the parent's weight-three-limited D14 construction.")
GROUP = ("New invertible-matrix construction over F3, not a metacyclic parameter change: |GL(2,3)|=(9-1)(9-3)=48, genuinely different from D14. This matrix-group bed is absent from BEDS. Strong nonabelian lower-rate beds motivate the experiment but do not establish a high-rate group advantage.")
SHAPES = ("Target cell 0.400: equal 1x3 components have 48 checks on 144 variables each, n=480 and guaranteed k>=192, rate 0.400. Verified ranks require 0.375<=k/n<0.425. Parent weight-three witnesses spanned many sector-1 Acol/Bcol blocks; failure enrichment was mostly 1.0-1.1x, supporting a global component change rather than local reinforcement.")
FILLING = ("All six cells are distinct-element trinomials. Twenty explicit pairs of relative-order profiles distinguish the design briefs. Independently screen components for rank, connectivity, small kernel supports and four/six cycles, then compare randomized kernel upper bounds. Finalists undergo actual quantum-logical, rank and CSS screening using verified lifts. Both check-weight sums equal 12, with 20736 six-round CNOTs. Stronger protection must outweigh this increased extraction work; these screens are pL hypotheses, not measured circuit scores.")
FAMILY_NAME = "r040_GL2F3_13x13_triple_ordercycle_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    from itertools import product
    if not valid_q(q):
        raise ValueError("GL(2,3) requires order 48")
    # (a,b,c,d) encodes the matrix [[a,b],[c,d]] over F3.
    els = [g for g in product(range(3), repeat=4)
           if (g[0]*g[3] - g[1]*g[2]) % 3]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 3 for x in g)

    def mul(g, h):
        a, b, c, d = red(g)
        e, f, u, v = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a, b, c, d = red(g)
        det = (a*d-b*c) % 3
        if not det:
            raise ValueError("singular matrix")
        # Each nonzero element of F3 is its own multiplicative inverse.
        return ((det*d) % 3, (-det*b) % 3,
                (-det*c) % 3, (det*a) % 3)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _short_logical_counts(checks, stabilizers, n):
    """Exact counts of nontrivial logical supports of weights 1..4.
    Matrices come exclusively from the verified scaffold lift."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(v):
        while v:
            p = v.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            v ^= old
        return False

    columns = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= syndrome_bit
            row ^= bit

    singles = {}
    counts = [0, 0, 0, 0]
    bits = [1 << a for a in range(n)]
    for a, syndrome in enumerate(columns):
        singles.setdefault(syndrome, []).append(a)
        if syndrome == 0 and nontrivial(bits[a]):
            counts[0] += 1

    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            syndrome = columns[a] ^ columns[b]
            support = bits[a] | bits[b]
            if syndrome == 0 and nontrivial(support):
                counts[1] += 1

            # a < b < c counts each weight-three support once.
            for c in singles.get(syndrome, ()):
                if c > b and nontrivial(support | bits[c]):
                    counts[2] += 1

            bucket = pairs.get(syndrome)
            if bucket is None:
                pairs[syndrome] = [(a, b, support)]
            else:
                # c < d < a < b is the unique ordered partition
                # used to count each weight-four support once.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        counts[3] += 1
                bucket.append((a, b, support))
    return tuple(counts)


def _matrix_triple_pools(G):
    """Classify complete cells by their three relative-element orders."""
    orders = {}
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                orders[G._key(g)] = t
                break
        else:
            raise ValueError("element order missing")

    pools = {}
    nonidentity = [g for g in G.elements
                   if G._key(g) != G._key(G.e)]
    for i, u in enumerate(nonidentity):
        for v in nonidentity[i + 1:]:
            pattern = tuple(sorted((
                orders[G._key(u)], orders[G._key(v)],
                orders[G._key(G.mul(G.inv(u), v))])))
            if min(pattern) >= 4:
                pools.setdefault(pattern, []).append((u, v))
    return pools


def _matrix_component(G, side, grid):
    """Extract only through the evaluator's verified lift conventions."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (3 * G.q)) - 1
    return [row & mask for row in rows]


def _matrix_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _matrix_geometry(rows, n):
    """Component connectivity and exact Tanner four/six-cycle counts."""
    nr = len(rows)
    intersections = {}
    four = six = 0
    for i in range(nr):
        for j in range(i + 1, nr):
            common = rows[i] & rows[j]
            degree = common.bit_count()
            if degree:
                intersections[i, j] = (degree, common)
                four += degree * (degree - 1) // 2
    for (i, j), (a, ij) in intersections.items():
        for k in range(j + 1, nr):
            ik = intersections.get((i, k))
            jk = intersections.get((j, k))
            if ik is not None and jk is not None:
                b, c = ik[0], jk[0]
                common = (ij & rows[k]).bit_count()
                six += a*b*c - common*(a+b+c) + 2*common

    columns = _matrix_columns(rows, n)
    reached = 1
    while True:
        previous = reached
        for column in columns:
            if column & reached:
                reached |= column
        if reached == previous:
            break
    connected = reached.bit_count() == nr and all(columns)
    return int(not connected), four, six


def _matrix_kernel_upper(rng, rows, n, trials):
    """Random fundamental kernel circuits: upper bounds, not certificates."""
    columns = _matrix_columns(rows, n)
    order = list(range(n))
    best = n + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
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
            if not syndrome and word:
                best = min(best, word.bit_count())
    return best


def _matrix_design_component(rng, G, side, brief, pools):
    """Independent voltage search within one algebraic design brief."""
    n = 3 * G.q
    candidates = []
    seen = set()
    for attempt in range(48):
        cells = []
        for pattern in brief:
            u, v = rng.choice(pools[pattern])
            base = G.pick(rng)
            cells.append([base, G.mul(base, u), G.mul(base, v)])
        grid = [cells]
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in cells)
        if signature in seen:
            continue
        seen.add(signature)
        rows = _matrix_component(G, side, grid)
        deficit = G.q - gf2_rank(rows)
        low = _short_logical_counts(rows, [], n)
        disconnected, four, six = _matrix_geometry(rows, n)
        key = (deficit, disconnected) + low + (four, six)
        candidates.append((key, grid, rows))

    candidates.sort(key=lambda item: item[0])
    finalists = []
    for key, grid, rows in candidates[:8]:
        upper = _matrix_kernel_upper(rng, rows, n, 32)
        # Rank, connectivity and small words remain ahead of this estimate.
        final_key = key[:6] + (-upper,) + key[6:]
        finalists.append((final_key, grid))
    finalists.sort(key=lambda item: item[0])
    return finalists[:3]


def NEW_CTX(rng, G, idx):
    """Twenty explicit order-profile pairs, followed by actual LP screening."""
    pools = _matrix_triple_pools(G)
    R, Q, C = (4,4,4), (4,6,6), (6,6,6)
    O, E = (4,8,8), (6,8,8)
    profiles = (
        (O,O,O), (O,O,E), (O,E,E), (E,E,E),
        (Q,O,E), (C,O,E), (R,O,E),
        (Q,E,E), (C,E,E), (R,E,E),
    )
    pairs = (
        (0,0), (0,1), (1,1), (0,2), (1,2),
        (2,2), (0,3), (1,3), (2,3), (3,3),
        (0,4), (1,4), (2,4), (0,5), (1,5),
        (2,5), (4,4), (4,5), (5,5), (0,6),
    )
    pa, pb = pairs[idx % len(pairs)]
    ac, bc = profiles[pa], profiles[pb]
    if any(pattern not in pools for pattern in ac + bc):
        raise ValueError("relative-order profile unavailable in GL(2,3)")
    assert _grid_weights([[3,3,3]], [[3,3,3]]) == 12

    for batch in range(3):
        aa = _matrix_design_component(rng, G, "A", ac, pools)
        bb = _matrix_design_component(rng, G, "B", bc, pools)
        combinations = []
        for ka, A in aa:
            for kb, B in bb:
                joint = (tuple(max(a,b) for a,b in zip(ka,kb)) +
                         tuple(a+b for a,b in zip(ka,kb)))
                combinations.append((joint, A, B))
        combinations.sort(key=lambda item: item[0])

        eligible = []
        for joint, A, B in combinations:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (375*n <= 1000*k < 425*n):
                continue
            eligible.append((joint, A, B, hx, hz, n, k))
            if len(eligible) >= 3:
                break

        best = None
        for joint, A, B, hx, hz, n, k in eligible:
            x = _short_logical_counts(hz, hx, n)
            z = _short_logical_counts(hx, hz, n)
            key = (tuple(max(a,b) for a,b in zip(x,z)) +
                   tuple(a+b for a,b in zip(x,z)) +
                   joint + (k - 4*G.q,))
            if best is None or key < best[0]:
                best = (key, A, B, hx, hz)
        if best is not None:
            key, A, B, hx, hz = best
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"idx": idx, "brief": (pa,pb),
                    "cells": {"A": A, "B": B}, "screen": key}
    raise ValueError("no target-rate matrix-group filling")


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Gate reduction plus short-logical multiplicity optimization on D14 (1,4)x(1,4) did not improve memory: fitness fell from 3.4511 to 2.9485, all distinct slots had d2s=3, and measured failures remained frequent despite mostly first-pass settlement.
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
