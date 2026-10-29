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
CONCEPT = ("S3 x C7, order 42, A=(1,3), B=(1,4): three-term A cells and three long-order B doubletons plus one trinomial should reduce short-logical multiplicity and improve circuit reliability in frontier cell 0.450.")
GROUP = ("Jump from Dic11 to the untried direct product S3 x C7, whose centre has order seven. Noncommuting displacements of orders 14 and 21 avoid the order-three doubletons present in the shown A4 x C3 bed, which scored 3.4843 with weight-three witnesses and frequent failures. This is an algebraic hypothesis, not a measured group-only advantage.")
SHAPES = ("A=(1,3), B=(1,4) gives component lengths 126 and 168 with 42 checks each, n=546 and kb=252, rate 6/13=0.46154. Exact lifted ranks enforce 0.425<=k/n<0.475. Current-parent failures were diffuse: slot 33 had 19 Z failures and slot 43 had six X failures, with Bcol1 enrichment up to 1.3x; change both component dimensions rather than claim a unique bad block.")
FILLING = ("A has three trinomials; B has three doubletons and one trinomial. Both side totals are nine: HX weights 11-12, HZ weight 12, and 15876 six-round CNOTs. Twenty deliberate briefs prescribe invariant trinomial projection types; normalized cells remove translation-only search. B doubletons use two distinct S3 reflections with order-14 displacements and one order-21 rotation. Verified lifts screen short component words, cycles, exact rate and quantum logicals through weight four; finalists minimize weight-five/six multiplicities. These proxies test pL improvement, not distance as fitness.")
FAMILY_NAME = "r045_S3xC7_13x14_longratio_multiplicity_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 x C7 requires order 42")
    # (a,b,c)=x^a*y^b*z^c:
    # x^3=y^2=z^7=1, y*x*y^-1=x^-1, z central.
    els = [(a, b, c) for c in range(7)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 7)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d)) % 3,
                (b + e) % 2, (c + f) % 7)

    def inv(g):
        a, b, c = red(g)
        return ((-a if b == 0 else a) % 3, b, (-c) % 7)

    def idx(g):
        a, b, c = red(g)
        return 6*c + 3*b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _projection_brief(idx):
    """Twenty initial architectures, with additional briefs for larger budgets.

    Trinomial types have projected pair-order multisets:
    0: (3,3,3), 1: (2,2,3), 2: (1,2,2),
    3: (1,3,3), 4: (1,1,1).
    These distinguish architectures beyond coordinate translations.
    """
    from itertools import combinations_with_replacement
    primary = list(combinations_with_replacement(range(3), 3))
    extra = [p for p in combinations_with_replacement(range(5), 3)
             if p not in primary]
    bank = [(p, b) for p in primary + extra for b in (1, 2)]
    if idx >= len(bank):
        raise ValueError("slot budget exceeds deliberate projection briefs")
    return bank[idx]


def _trinomial(rng, G, kind):
    """Identity-normalized cell; every pair has nonzero C7 displacement."""
    c, d = rng.sample(range(1, 7), 2)
    if kind == 0:
        a = rng.choice((1, 2))
        u, v = (a, 0), ((-a) % 3, 0)
    elif kind == 1:
        a, b = rng.sample(range(3), 2)
        u, v = (a, 1), (b, 1)
    elif kind == 2:
        a = rng.randrange(3)
        u = v = (a, 1)
    elif kind == 3:
        a = rng.choice((1, 2))
        u = v = (a, 0)
    else:
        u = v = (0, 0)
    return [G.e, u + (c,), v + (d,)]


def _draw_design(rng, G, brief):
    kinds, bkind = brief
    A = [[_trinomial(rng, G, kind) for kind in kinds]]
    # Two different projected reflections and one nonidentity rotation.
    # Their nonzero central coordinates give orders 14,14,21.
    r, s = rng.sample(range(3), 2)
    cs = rng.sample(range(1, 7), 3)
    B = [[[G.e, (r, 1, cs[0])],
          [G.e, (s, 1, cs[1])],
          [G.e, (rng.choice((1, 2)), 0, cs[2])],
          _trinomial(rng, G, bkind)]]
    return A, B


def _component_rows(hx, hz, q):
    """Project components only from the scaffold's verified matrices."""
    mask = (1 << q) - 1
    ar = []
    for row in hx[:q]:             # B-column zero, A-row zero
        packed = 0
        for j in range(3):
            packed |= ((row >> (4*j*q)) & mask) << (j*q)
        ar.append(packed)
    bmask = (1 << (4*q)) - 1
    br = [row & bmask for row in hz[:q]]
    return ar, br


def _columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _short_statistics(rows, n):
    """Short component kernels and four-cycles, not a pL estimate."""
    cols = _columns(rows, n)
    singles = set(cols)
    duplicates = n - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
    c4 = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            c4 += overlap*(overlap-1)//2
    # Multiplicity normalizations are exact when columns are unique
    # and nonzero; otherwise duplicates take priority.
    return (duplicates, triples, collisions), c4, cols, pairs


def _long_statistics(cols, pairs, short):
    """Weight-five/six collision counts and small triple boundaries.

    With no kernels of weights <=4, pair/triple collisions count
    weight-five words ten times, and disjoint triple collisions count
    weight-six words ten times. Otherwise these remain heuristics.
    """
    seen = {}
    five = six = boundary = 0
    for i in range(len(cols)):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                five += pairs.get(syndrome, 0)
                old = seen.get(syndrome, 0)
                six += old
                seen[syndrome] = old + 1
                w = syndrome.bit_count()
                if w < 3:
                    boundary += (3-w)**2
    if not any(short):
        five //= 10
        six //= 10
    return five, six, boundary


def _pivots(rows):
    piv = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            if p in piv:
                row ^= piv[p]
            else:
                piv[p] = row
                break
    return piv


def _nontrivial(word, piv):
    while word:
        p = word.bit_length()-1
        if p not in piv:
            return True
        word ^= piv[p]
    return False


def _quantum_low4(hx, hz, n):
    """Exact stabilizer-aware existence test through weight four."""
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        cols = _columns(checks, n)
        piv = _pivots(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            word = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _nontrivial(word ^ old, piv):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = cols[i] ^ cols[j]
                word = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif _nontrivial(word ^ old, piv):
                    return True
    return False


def NEW_CTX(rng, G, idx):
    """Optimize independent voltages for an invariant projection brief."""
    brief = _projection_brief(idx)
    sa, sb = [[3, 3, 3]], [[2, 2, 2, 3]]
    assert _grid_weights(sa, sb) <= min(W_MAX, 12)
    assert _no_dead_lines(sa, sb)
    q = G.q
    pool = []

    # Screening uses actual helper-built components. No custom lift.
    for trial in range(96):
        A, B = _draw_design(rng, G, brief)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _component_rows(hx, hz, q)
        ast = _short_statistics(ar, 3*q)
        bst = _short_statistics(br, 4*q)
        short = tuple(v for t in range(3)
                      for v in (max(ast[0][t], bst[0][t]),
                                ast[0][t] + bst[0][t]))
        cycles = (max(ast[1], bst[1]), ast[1] + bst[1])
        key = short + cycles
        pool.append((key, trial, A, B, hx, hz, n, ast, bst))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[12:]

    eligible = []
    for item in pool:
        key, trial, A, B, hx, hz, n, ast, bst = item
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 425*n <= 1000*k < 475*n):
            continue
        eligible.append((item, k))
        if len(eligible) == 4:
            break

    # A missed optional neighborhood screen must not silently change
    # the architecture. Additional independent labels retain the brief.
    if not eligible:
        for trial in range(96, 224):
            A, B = _draw_design(rng, G, brief)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 425*n <= 1000*k < 475*n):
                continue
            ar, br = _component_rows(hx, hz, q)
            ast = _short_statistics(ar, 3*q)
            bst = _short_statistics(br, 4*q)
            key = tuple(v for t in range(3)
                        for v in (max(ast[0][t], bst[0][t]),
                                  ast[0][t] + bst[0][t]))
            key += (max(ast[1], bst[1]), ast[1] + bst[1])
            eligible.append(((key, trial, A, B, hx, hz, n, ast, bst), k))
            if len(eligible) == 4:
                break
    if not eligible:
        raise ValueError("no exact-rank candidate in cell 0.450")

    best = None
    for item, k in eligible:
        key, trial, A, B, hx, hz, n, ast, bst = item
        la = _long_statistics(ast[2], ast[3], ast[0])
        lb = _long_statistics(bst[2], bst[3], bst[0])
        multiplicities = tuple(v for t in range(2)
                               for v in (max(la[t], lb[t]), la[t] + lb[t]))
        boundary = (max(la[2], lb[2]), la[2] + lb[2])
        quantum_bad = int(_quantum_low4(hx, hz, n))
        # Quantum protection first; then short-word multiplicities.
        # Gate exposure is fixed across this entire program.
        score = (quantum_bad,) + key[:6] + multiplicities
        score += boundary + key[6:] + (-k, trial)
        if best is None or score < best[0]:
            best = (score, A, B, hx, hz, k)

    score, A, B, hx, hz, k = best
    assert css_ok(hx, hz)
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "brief": brief, "k": k, "screen": score,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened cell without late randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-12 rolling alone removes the D14 (6,2)x(6,2) sector-2 bottleneck: parent measurements instead found weight-four logicals in both bases and worse-basis scores 3.30-3.59.
# REFUTED: Higher screened distance guarantees better Dic11 circuit reliability: current-parent slot 33 had d2s=26 and 19 Z failures, whereas slot 6 had d2s=22 and zero failures in both bases.
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
