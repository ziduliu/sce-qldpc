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
CONCEPT = ("Cyclic C23, order 23, A=(1,4), B=(2,6): triangle-free binomial directions separated from B within-cell differences should improve worse-basis circuit reliability over the parent's 3.4843.")
GROUP = ("Jump from A4 x C3 to C23, a genuinely different group. C23 multirow beds reached about 6.9 and the C59 single-row bed reached 4.6; these motivate, but do not validate, this untried bed. Prime order gives every nonzero binomial difference order 23. The shape factor 26 admits q=23 and n=598.")
SHAPES = ("A=(1,4), B=(2,6) gives component dimensions 23x92 and 46x138, n=598, kb=276 and guaranteed rate 6/13 in cell 0.450. Require verified k=276 for odd A profiles or k=280 for all-binomial A profiles. Parent failures involved many sector-one Acol/Bcol blocks with roughly 1.0-1.1x enrichment, motivating a global component change rather than an unsupported hotspot repair.")
FILLING = ("Four support architectures receive deterministic libraries of graph-invariant-distinct outcomes: A=(2,2,2,3) or (2,2,2,2), and two-row B monomial/doubleton profiles with row totals eight or nine. Construct component-four-cycle-free labels, forbid A binomial triangles, and exclude A binomial directions from B doubleton differences. Verified full-code screens exclude logicals through weight three and minimize weight-four logical pair collisions with possible opposite-check hooks. Full-code cycle and degree signatures prevent library orbit copies. Both check-weight sums are <=12. Six-round CNOT counts are 21252-22356, an explicit cost increase traded against removing the parent's weight-three mechanism; only circuit measurement establishes the benefit.")
FAMILY_NAME = "r045_C23_14x26_separated_hook_library_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C23 requires order 23")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _lp_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _lp_component(G, side, grid):
    """Use an opposite zero-block probe of the verified construction."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _lp_kernel(rows, width):
    """Exact component weight-three/four counts for unique columns."""
    cols = _lp_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = {}
    three = four = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            syndrome = x ^ y
            three += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            four += old
            pairs[syndrome] = old + 1
    assert three % 3 == four % 3 == 0
    return three // 3, four // 3


def _lp_cycle_signature(rows):
    """Exact Tanner four/six cycles and relabeling-invariant degrees."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    adj = [0] * m
    hist = [0] * 13
    four = six = 0
    for i, row in enumerate(rows):
        for j in range(i):
            t = (row & rows[j]).bit_count()
            hist[t] += 1
            four += t * (t - 1) // 2
            if t:
                overlap[i][j] = overlap[j][i] = t
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    for i in range(m):
        earlier = adj[i] & ((1 << i) - 1)
        while earlier:
            bit = earlier & -earlier
            j = bit.bit_length() - 1
            common = adj[i] & adj[j] & ((1 << j) - 1)
            ij = rows[i] & rows[j]
            x = overlap[i][j]
            while common:
                low = common & -common
                h = low.bit_length() - 1
                y, z = overlap[j][h], overlap[i][h]
                t = (ij & rows[h]).bit_count()
                six += x * y * z - t * (x + y + z) + 2 * t
                common ^= low
            earlier ^= bit
    degrees = tuple(sorted(row.bit_count() for row in rows))
    return four, six, tuple(hist), degrees


def _lp_logical_profile(checks, stabilizers, n):
    """Exact short logical counts and possible hook-pair collisions.

    Unique nonzero columns exclude weights one and two. Linear
    residues distinguish logical differences from stabilizers.
    Hook counts describe shared opposite checks, not ASC fault proofs.
    """
    cols = _lp_columns(checks, n)
    if 0 in cols or len(set(cols)) != n:
        return None
    co = _lp_columns(stabilizers, n)
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    bits = [1 << i for i in range(n)]
    residuals = list(bits)
    for p in sorted(pivots):
        row = pivots[p] ^ bits[p]
        residue = 0
        while row:
            bit = row & -row
            residue ^= residuals[bit.bit_length() - 1]
            row ^= bit
        residuals[p] = residue

    singles = dict(zip(cols, residuals))
    pairs = {}
    three = four = double_hook = exposure = 0
    for i, x in enumerate(cols):
        ri, ci = residuals[i], co[i]
        for j in range(i):
            syndrome = x ^ cols[j]
            label = ri ^ residuals[j]
            third = singles.get(syndrome)
            if third is not None and label != third:
                three += 1
            hook = (ci & co[j]).bit_count()

            old = pairs.get(syndrome)
            if old is None:
                pairs[syndrome] = [1, hook, {label: [1, hook]}]
                continue
            same = old[2].get(label)
            same_count, same_hooks = same if same else (0, 0)
            different = old[0] - same_count
            different_hooks = old[1] - same_hooks
            four += different
            double_hook += hook * different_hooks
            exposure += hook * different + different_hooks
            old[0] += 1
            old[1] += hook
            if same is None:
                old[2][label] = [1, hook]
            else:
                same[0] += 1
                same[1] += hook
    assert three % 3 == four % 3 == 0
    return three // 3, four // 3, double_hook, exposure


def NEW_CTX(rng, G, idx):
    """Four architectures, five invariant-distinct designs each.

    A deterministic local library makes idx choose an optimized design,
    not another draw or a relabeling. Larger harness budgets select
    further invariant-distinct entries from the same architectures.
    """
    architecture, choice = divmod(idx % 20, 5)
    choice += 5 * (idx // 20)
    wanted = 5 * ((N_SLOTS + 19) // 20)
    rr = random.Random(734923109 + 104729 * architecture)
    q = G.q

    pa = (2, 2, 2, 3) if architecture < 2 else (2, 2, 2, 2)
    sb = [[1] * 6 for _ in range(2)]
    for j in ((0, 1) if architecture == 1 else (0, 1, 2)):
        sb[0][j] = 2
    for j in ((2, 3, 4) if architecture == 3 else (3, 4, 5)):
        sb[1][j] = 2
    sa = [list(pa)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert max(sum(row) for row in sa) + max(
        sum(row[j] for row in sb) for j in range(6)) <= 12
    assert max(sum(row) for row in sb) + max(pa) <= 12
    expected_k = 12 * q if architecture < 2 else 12 * q + 4

    def undirected(d):
        d %= q
        return min(d, q - d)

    def draw_a():
        for _ in range(1024):
            nb = pa.count(2)
            ds = [1] + rr.sample(range(2, (q + 1) // 2), nb - 1)
            signed = {s * d % q for d in ds for s in (-1, 1)}
            if any((x + y) % q in signed
                   for x in signed for y in signed):
                continue
            row = [[(0,), (d,)] for d in ds]
            if pa[-1] == 3:
                a, b = rr.sample(range(1, q), 2)
                internal = {undirected(a), undirected(b),
                            undirected(b - a)}
                if len(internal) != 3 or internal.intersection(ds):
                    continue
                row.append([(0,), (a,), (b,)])
            return [row], set(ds)
        raise ValueError("no triangle-free A support")

    def draw_b(forbidden):
        """Construct disjoint within-row and cross-row differences.

        This is a proposal rule for C23. The actual component matrices
        below certify the cycle, rank and short-kernel consequences.
        """
        B = [[], []]
        within = [set(), set()]
        cross = set()
        for j in range(6):
            for _ in range(512):
                deltas = []
                for i in range(2):
                    if sb[i][j] == 1:
                        deltas.append(0)
                    else:
                        allowed = [d for d in range(1, (q + 1) // 2)
                                   if d not in forbidden
                                   and d not in within[i]]
                        if not allowed:
                            break
                        deltas.append(rr.choice(allowed) *
                                      rr.choice((-1, 1)) % q)
                if len(deltas) != 2:
                    continue
                v = rr.randrange(q)
                upper = [0] if sb[0][j] == 1 else [0, deltas[0]]
                lower = [v] if sb[1][j] == 1 else [
                    v, (v + deltas[1]) % q]
                differences = [(a - b) % q for a in upper for b in lower]
                if len(set(differences)) != len(differences):
                    continue
                if cross.intersection(differences):
                    continue
                cross.update(differences)
                for i, values in enumerate((upper, lower)):
                    B[i].append([(a,) for a in values])
                    if len(values) == 2:
                        within[i].add(undirected(deltas[i]))
                break
            else:
                raise ValueError("cross-row difference construction stalled")
        return B

    library = {}
    exact_seen = set()
    for batch in range(4):
        pool = []
        for trial in range(96):
            try:
                A, forbidden = draw_a()
                B = draw_b(forbidden)
            except ValueError:
                continue
            identity = tuple(
                tuple(sorted(g[0] for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if identity in exact_seen:
                continue
            exact_seen.add(identity)

            ar, aw = _lp_component(G, "A", A)
            br, bw = _lp_component(G, "B", B)
            if gf2_rank(ar) != q - int(architecture >= 2):
                continue
            if gf2_rank(br) != 2 * q:
                continue
            ak, bk = _lp_kernel(ar, aw), _lp_kernel(br, bw)
            if ak is None or bk is None or ak[0] or bk[0]:
                continue
            ac, bc = _lp_cycle_signature(ar), _lp_cycle_signature(br)
            if ac[0] or bc[0]:
                continue
            coarse = (max(ak[1], bk[1]), ak[1] + bk[1],
                      max(ac[1], bc[1]), ac[1] + bc[1])
            pool.append((coarse, identity, A, B))
        pool.sort(key=lambda item: item[:2])

        for coarse, identity, A, B in pool[:40]:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 598 or k != expected_k:
                continue
            x = _lp_logical_profile(hx, hz, n)
            z = _lp_logical_profile(hz, hx, n)
            if x is None or z is None or x[0] or z[0]:
                continue
            cx, cz = _lp_cycle_signature(hx), _lp_cycle_signature(hz)
            # Different signatures prove these Tanner pairs are not
            # relabelings. Sorting also accounts for an X/Z swap.
            invariant = tuple(sorted(((x, cx), (z, cz))))
            key = (max(x[2], z[2]), x[2] + z[2],
                   max(x[3], z[3]), x[3] + z[3],
                   max(x[1], z[1]), x[1] + z[1],
                   max(cx[0] / len(hx), cz[0] / len(hz)),
                   cx[0] + cz[0],
                   max(cx[1] / len(hx), cz[1] / len(hz)),
                   cx[1] + cz[1], coarse)
            old = library.get(invariant)
            candidate = (key, identity, A, B, hx, hz, k)
            if old is None or candidate[:2] < old[:2]:
                library[invariant] = candidate
        if len(library) >= wanted:
            break

    if len(library) < wanted:
        raise ValueError("insufficient invariant-distinct screened designs")
    ordered = sorted(library.values(), key=lambda item: item[:2])
    key, identity, A, B, hx, hz, k = ordered[choice]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    expected_edges = q * (8 * sum(pa) + 5 * sum(map(sum, sb)))
    assert sum(row.bit_count() for row in hx + hz) == expected_edges
    return {"idx": idx, "architecture": architecture, "k": k,
            "screen": key, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the full selected cell without late resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing four-cycles did not ensure reliable A4 x C3 memory: fitness 3.2178, measured Z failures 112-137/250, and 72-216 weight-three X witnesses.
# REFUTED: Reducing this A4 x C3 bed to 36 weight-three X logicals made failures rare: fitness remained 3.4843 and measured Z failures were 140-176/500 shots despite predominantly first-stage settlement.
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
