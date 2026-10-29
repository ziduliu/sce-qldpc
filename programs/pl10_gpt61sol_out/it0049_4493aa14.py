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
CONCEPT = ("Dic7, order 28, shapes (2,4)x(2,4): balanced degree-three/four components and voltage assignments excluding central-involution differences should improve worse-basis circuit reliability over rolled supports in cell 0.200.")
GROUP = ("Dic7 is nonabelian, unlike the parent's Z23. This bed is untried; the related Dic11 (2,3)x(2,3) bed scored 7.1162, motivating a family test rather than proving a group effect. Dic7 has one involution, so intra-cell involution differences can be excluded explicitly.")
SHAPES = ("Equal (2,4) components each have 112 variables and 56 checks. At q=28, n=560 and kb=112, guaranteed rate 0.200. Verified full ranks retain 112<=k<=125. Parent X failures span Acol0/2 and multiple B-column/B-row blocks with mostly 1.1x enrichment, motivating broad reinforcement rather than an isolated-block patch.")
FILLING = ("Twenty explicit support-pair designs have component column degrees three/four and row degrees six/eight, giving both CSS check-weight sums <=12. Exclude antipodal label pairs; optimize component short cycles; then screen verified full matrices for rate, connectivity and logicals of weight <=4. Sampled kernel witnesses and small-syndrome triples refine selection, but are not distance certificates or circuit scores. Targets are the parent's widespread weight-six witnesses and X failures, including 102/1500 in slot 4 versus only 1/200000 Z failures.")
FAMILY_NAME = "r020_dic7_24x24_antipodal_balanced_traps_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic7 requires order 28")
    # x^14=1, y^2=x^7, y*x*y^-1=x^-1.
    # (a,b) encodes x^a*y^b.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        return (int(g[0]) % 14, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 7*b*d) % 14,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 14, 0) if b == 0 else ((a + 7) % 14, 1)

    def idx(g):
        a, b = red(g)
        return 14*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _d7_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _d7_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _d7_cycles(rows, n):
    """Exact four-cycles; six-cycle heuristic when four-cycles exist."""
    adjacency = [0] * len(rows)
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            common = (row & rows[j]).bit_count()
            four += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    triangles = 0
    for i, neighbors in enumerate(adjacency):
        lower = neighbors & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (neighbors & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = 0
    for col in _d7_columns(rows, n):
        d = col.bit_count()
        stars += d * (d - 1) * (d - 2) // 6
    return four, max(0, triangles - stars)


def _d7_small(cols, stabilizers=None):
    """Exact existence test through weight four, modulo a supplied span."""
    pivots = {}
    for row in stabilizers or []:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if nontrivial(word ^ seen[syndrome]):
                return True
        else:
            seen[syndrome] = word
    for i, si in enumerate(cols):
        for j in range(i):
            syndrome = si ^ cols[j]
            word = (1 << i) | (1 << j)
            if syndrome in seen:
                if nontrivial(word ^ seen[syndrome]):
                    return True
            else:
                seen[syndrome] = word
    return False


def _d7_kernel_estimate(rng, cols):
    """Upper bound from actual randomized fundamental kernel words."""
    order = list(range(len(cols)))
    best = len(cols) + 1
    for _ in range(48):
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in pivots:
                    s, w = pivots[p]
                    syndrome ^= s
                    word ^= w
                else:
                    pivots[p] = (syndrome, word)
                    break
            if not syndrome and word:
                best = min(best, word.bit_count())
    return best


def _d7_traps(cols):
    """Small triple boundaries and repeated triple syndromes."""
    boundary = repeated = 0
    seen = {}
    for i in range(len(cols) - 2):
        for j in range(i + 1, len(cols) - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, len(cols)):
                syndrome = pair ^ cols[k]
                w = syndrome.bit_count()
                if w < 5:
                    boundary += (5 - w) ** 2
                old = seen.get(syndrome, 0)
                repeated += old
                seen[syndrome] = old + 1
    return boundary, repeated


def _d7_connected(hx, hz, n):
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        bit = row & -row
        first = bit.bit_length() - 1
        row ^= bit
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            parent[root(c)] = root(first)
            row ^= bit
    return len({root(c) for c in range(n)}) == 1


def _d7_component_pool(rng, G, supports, side, mode):
    """Independent label optimization inside one fixed support design."""
    import math

    nc = 4 * G.q
    identity = [[[G.e]]]
    central = (7, 0)

    def legal(cell):
        occupied = set(cell)
        return (len(occupied) == len(cell)
                and all(G.mul(g, central) not in occupied for g in cell))

    def component(grid):
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, identity)
        else:
            _, rows, _ = lifted_matrices(G, identity, grid)
        mask = (1 << nc) - 1
        return [row & mask for row in rows]

    def draw():
        grid = []
        for support_row in supports:
            row = []
            for w in support_row:
                if mode == 1 and w > 1:
                    b = rng.randrange(2)
                    bs = [b, 1 - b] + [b] * (w - 2)
                elif mode == 2:
                    bs = [rng.randrange(2)] * w
                else:
                    bs = [rng.randrange(2) for _ in range(w)]
                cell = []
                for b in bs:
                    while True:
                        g = (rng.randrange(14), b)
                        if legal(cell + [g]):
                            cell.append(g)
                            break
                row.append(cell)
            grid.append(row)
        return grid

    pool = []
    for restart in range(5):
        grid = draw()
        rows = component(grid)
        four, six = _d7_cycles(rows, nc)
        energy = 100 * four + six
        best_energy = energy
        best_grid = _d7_copy(grid)

        for step in range(200):
            i, j = rng.randrange(2), rng.randrange(4)
            cell = grid[i][j]
            t = rng.randrange(len(cell))
            old = cell[t]
            new = ((rng.randrange(14), old[1]) if mode
                   else G.pick(rng))
            cell[t] = new
            if not legal(cell):
                cell[t] = old
                continue
            trial_rows = component(grid)
            c4, c6 = _d7_cycles(trial_rows, nc)
            trial_energy = 100 * c4 + c6
            temperature = G.q * (8.0 * (0.975 ** step) + 0.15)
            delta = trial_energy - energy
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                energy = trial_energy
                if energy < best_energy or (
                        energy == best_energy and rng.random() < 0.04):
                    best_energy = energy
                    best_grid = _d7_copy(grid)
            else:
                cell[t] = old

        rows = component(best_grid)
        cols = _d7_columns(rows, nc)
        four, six = _d7_cycles(rows, nc)
        bad = int(_d7_small(cols))
        d = _d7_kernel_estimate(rng, cols)
        pool.append(dict(grid=best_grid, cols=cols, bad=bad,
                         d=d, four=four, six=six))
    pool.sort(key=lambda a: (a["bad"], -min(a["d"], 14),
                             a["four"], a["six"]))
    return pool[:4]


def NEW_CTX(rng, G, idx):
    """Twenty architectures; extra harness slots change coset policy."""
    profiles = (
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((1, 2, 2, 2), (2, 2, 2, 2)),
        ((1, 1, 2, 2), (2, 2, 2, 2)),
        ((1, 2, 2, 2), (2, 1, 2, 2)),
        ((1, 1, 2, 2), (2, 2, 1, 2)),
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((1, 1, 3, 2), (2, 2, 1, 2)),
        ((1, 1, 3, 3), (3, 3, 1, 1)),
    )
    pairs = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 2), (2, 3), (3, 3),
        (0, 6), (1, 6), (3, 6), (6, 6), (0, 7), (7, 7),
    )
    pa, pb = pairs[idx % 20]
    sa, sb = profiles[pa], profiles[pb]
    mode = (idx // 20) % 3
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    for batch in range(3):
        ap = _d7_component_pool(rng, G, sa, "A", mode)
        bp = _d7_component_pool(rng, G, sb, "B", mode)
        eligible = []
        for a in ap:
            for b in bp:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 560 and 112 <= k <= 125):
                    continue
                key = (
                    a["bad"] + b["bad"],
                    -min(a["d"], b["d"], 14),
                    max(a["four"], b["four"]),
                    a["four"] + b["four"],
                    max(a["six"], b["six"]),
                    a["six"] + b["six"],
                    k,
                )
                eligible.append((key, a, b, hx, hz, n))
        eligible.sort(key=lambda item: item[0])

        winner = None
        winner_key = None
        trap_cache = {}
        for base, a, b, hx, hz, n in eligible[:8]:
            if not _d7_connected(hx, hz, n):
                continue
            if _d7_small(_d7_columns(hx, n), hz):
                continue
            if _d7_small(_d7_columns(hz, n), hx):
                continue
            for candidate in (a, b):
                tag = id(candidate)
                if tag not in trap_cache:
                    trap_cache[tag] = _d7_traps(candidate["cols"])
            ta, tb = trap_cache[id(a)], trap_cache[id(b)]
            key = (base[:2], base[2:4],
                   max(ta[0], tb[0]), ta[0] + tb[0],
                   max(ta[1], tb[1]), ta[1] + tb[1], base[4:])
            if winner_key is None or key < winner_key:
                winner_key = key
                winner = (a["grid"], b["grid"], hx, hz)

        if winner is not None:
            A, B, hx, hz = winner
            assert css_ok(hx, hz)
            return {"idx": idx, "cells": {"A": A, "B": B}}
    raise ValueError("No connected Dic7 filling passed target-rate/logical screens")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened cell without subsequent random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform Z23 labels ensure basis-balanced memory: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six Z witnesses.
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
