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
CONCEPT = ("Metacyclic Z2:Z2(s=1), order 4, shapes (3,11)x(3,11): balanced degree-two/three columns and three-row component geometry should reduce collective circuit failures by reducing short-logical multiplicity in both bases.")
GROUP = ("Keep the metacyclic construction rule. The new size factor 130 makes the parent's q=14 inadmissible; m=2,r=2,s=1 gives q=4 and n=520. This abelian specialization and shape bed are untried. Its purpose is a legal three-row geometry experiment, not a claimed measured group advantage.")
SHAPES = ("Target cell 0.500: equal (3,11) components have 12 checks and 44 variables each; n=520 and kb=256 give guaranteed rate 64/130=0.49231. Verified full ranks keep submitted codes in [0.475,0.525). Parent failures affected all 213 logical qubits, with diffuse approximately 1.1x Acol/Bcol enrichment and thousands of weight-four sector-1 witnesses, motivating protection of both components rather than one unproven Brow hotspot.")
FILLING = ("Twenty unordered support-model pairs have six degree-two and five degree-three columns, nine terms per component row, and column sums two or three: both check weights are <=12 and the six-round CNOT count is 18144. Triangle-free degree-two backbones initialize independent verified-component multistart searches. Local refinement prioritizes weight-one/two/three words, then weight-four multiplicity and cycles; final pairings use the stabilizer-aware full-CSS hook screen. Zero weight-three logicals are preferred, with a rank-legal weaker fallback instead of voiding a slot. The extra gates versus the parent's 13440 test protection against its 107-179 failures per 250 shots; these screens remain circuit-risk surrogates, not pL measurements.")
FAMILY_NAME = "r050_m2r2s1_311x311_balanced_three_row_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 4

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("These metacyclic parameters require q=4")
    # Same metacyclic construction; resized for the factor-130 shape.
    # x^2=y^2=1 and y*x*y^-1=x, since 1^2=1 modulo 2.
    m, r, s = 2, 2, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 11)              # (ra, ca)
B_SHAPE = (3, 11)              # (rb, cb)


def _r050_columns(rows, n):
    """Column syndrome signatures of an already verified lifted matrix."""
    columns = [0] * n
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _r050_small_profile(checks, stabilizers, n):
    """Reject logicals of weight <=3; count weight-four fault opportunities.

    A shared stabilizer check makes a pair a possible ancilla-spread pair.
    Two such pairs forming a logical are a two-fault opportunity, without
    asserting that the unknown ASC ordering actually realizes that hook.
    Each quartet is counted once per disjoint pair partition.
    """
    syndrome = _r050_columns(checks, n)
    shared = _r050_columns(stabilizers, n)

    # Row-space membership only. Lifts and reported k use scaffold helpers.
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break

    def in_span(word):
        while word:
            p = word.bit_length() - 1
            pivot = basis.get(p)
            if pivot is None:
                return False
            word ^= pivot
        return True

    bits = [1 << i for i in range(n)]
    singles = {}
    for i, s in enumerate(syndrome):
        if s == 0 and not in_span(bits[i]):
            return None
        singles.setdefault(s, []).append(bits[i])

    pairs = {}
    hook_count = 0
    logical_four = 0
    hits = [0] * n
    for i in range(n):
        si, bi = syndrome[i], bits[i]
        for j in range(i + 1, n):
            s = si ^ syndrome[j]
            mask = bi | bits[j]
            if s == 0 and not in_span(mask):
                return None
            for third in singles.get(s, ()):
                if not (mask & third) and not in_span(mask | third):
                    return None

            hook = bool(shared[i] & shared[j])
            entry = (mask, hook)
            old = pairs.get(s)
            if old is None:
                pairs[s] = entry
                continue

            bucket = old if isinstance(old, list) else (old,)
            for other_mask, other_hook in bucket:
                if mask & other_mask:
                    continue
                quartet = mask | other_mask
                if in_span(quartet):
                    continue
                logical_four += 1
                if hook and other_hook:
                    hook_count += 1
                    word = quartet
                    while word:
                        bit = word & -word
                        hits[bit.bit_length() - 1] += 1
                        word ^= bit
            if isinstance(old, list):
                old.append(entry)
            else:
                pairs[s] = [old, entry]

    return hook_count, logical_four, max(hits)


# Five odd-degree columns in each model. Every model sums to (5,5,5).
# Models 3 and 4 have the same number of binomial cells but different
# distributions of their singleton partners among the check blocks.
_R050_ODD_MODELS = (
    ((1,1,1),) * 5,
    ((2,1,0), (0,1,2)) + ((1,1,1),) * 3,
    ((2,1,0), (0,2,1), (1,0,2)) + ((1,1,1),) * 2,
    ((2,1,0), (0,1,2), (2,1,0), (0,1,2), (1,1,1)),
    ((2,1,0), (0,1,2), (2,0,1), (0,2,1), (1,1,1)),
    ((2,1,0), (0,2,1), (1,0,2), (2,1,0), (0,1,2)),
)
_R050_MODEL_PAIRS = tuple(
    (a, b) for a in range(6) for b in range(a, 6)
    if (a, b) != (5, 5)
)


def _r050_component(G, side, grid):
    """Use a zero opposite-side probe in the verified construction."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (11 * G.q)) - 1
    return [row & mask for row in rows]


def _r050_component_score(rows, width):
    """Classical short-word counts, not quantum distance or circuit pL.

    With nonzero distinct columns, each triple and quartet contributes
    three pair partitions. Earlier penalties dominate when duplicates
    would make those normalizations inexact.
    """
    cols = _r050_columns(rows, width)
    singles = {}
    one = two = 0
    for s in cols:
        one += int(s == 0)
        two += singles.get(s, 0)
        singles[s] = singles.get(s, 0) + 1

    pairs = {}
    three = four = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            s = x ^ y
            three += singles.get(s, 0)
            four += pairs.get(s, 0)
            pairs[s] = pairs.get(s, 0) + 1

    cycles = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return one + two, three // 3, four // 3, cycles


def _r050_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _r050_cells(rng, G, model):
    """Balanced support architecture with a triangle-free edge backbone."""
    nonidentity = [g for g in G.elements if G._key(g) != G._key(G.e)]
    h = rng.choice(nonidentity)
    v = rng.choice([g for g in nonidentity if G._key(g) != G._key(h)])

    # In C2 x C2, {e,h}+{e,h}={e,h}; {v,vh} is the other coset.
    # Thus the six degree-two columns have no component triangles.
    grid = [[[] for _ in range(11)] for _ in range(3)]
    edge_types = (
        ((0, 1), (G.e, h)),
        ((1, 2), (G.e, h)),
        ((0, 2), (v, G.mul(v, h))),
    )
    j = 0
    for (a, b), voltages in edge_types:
        for voltage in voltages:
            grid[a][j] = [G.e]
            grid[b][j] = [voltage]
            j += 1

    for counts in _R050_ODD_MODELS[model]:
        if counts == (1, 1, 1):
            grid[0][j] = [G.e]
            grid[1][j] = [G.pick(rng)]
            grid[2][j] = [G.pick(rng)]
        else:
            double = counts.index(2)
            single = counts.index(1)
            grid[double][j] = [G.e, rng.choice(nonidentity)]
            grid[single][j] = [G.pick(rng)]
        j += 1

    assert j == 11
    assert [sum(len(c) for c in row) for row in grid] == [9, 9, 9]
    return grid


def _r050_component_pool(rng, G, side, model, batch):
    """Multistart label refinement within one deliberate support model."""
    import math

    pool = []
    signatures = set()
    width = 11 * G.q

    def energy(key):
        return 100000 * key[0] + 2000 * key[1] + key[2] + 0.2 * key[3]

    def remember(grid, key):
        if key[0]:
            return
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in grid for cell in row)
        if signature in signatures:
            return
        signatures.add(signature)
        pool.append((key, _r050_copy(grid)))
        pool.sort(key=lambda item: item[0])
        del pool[5:]

    for restart in range(6):
        grid = _r050_cells(rng, G, model)
        positions = [(i, j) for i in range(3) for j in range(11)
                     if grid[i][j]]
        key = _r050_component_score(_r050_component(G, side, grid), width)
        cost = energy(key)
        remember(grid, key)

        for step in range(240):
            i, j = rng.choice(positions)
            cell = grid[i][j]
            position = rng.randrange(len(cell))
            old = cell[position]
            alternatives = [g for g in G.elements
                            if all(G._key(g) != G._key(x) for x in cell)]
            cell[position] = rng.choice(alternatives)
            trial = _r050_component_score(
                _r050_component(G, side, grid), width)
            new_cost = energy(trial)
            temperature = (12.0 + 2.0 * (batch % 3)) * (
                1.0 - step / 240.0) + 0.25
            delta = new_cost - cost
            if delta <= 0 or rng.random() < math.exp(
                    -min(delta / temperature, 700.0)):
                key, cost = trial, new_cost
                remember(grid, key)
            else:
                cell[position] = old

    if not pool:
        raise ValueError("component search retained zero/parallel columns")
    return pool


def NEW_CTX(rng, G, idx):
    """Optimize both components, then evaluate their actual CSS pairing."""
    assert len(_R050_MODEL_PAIRS) == 20
    design = idx % len(_R050_MODEL_PAIRS)
    batch = idx // len(_R050_MODEL_PAIRS)
    ma, mb = _R050_MODEL_PAIRS[design]

    ap = _r050_component_pool(rng, G, "A", ma, batch)
    bp = _r050_component_pool(rng, G, "B", mb, batch)

    options = []
    for ai, (ak, a) in enumerate(ap):
        for bi, (bk, b) in enumerate(bp):
            coarse = tuple(v for t in range(4)
                           for v in (max(ak[t], bk[t]), ak[t] + bk[t]))
            options.append((coarse, ai, bi, a, b))
    options.sort(key=lambda item: item[:3])

    best = None
    best_score = None
    # Expensive stabilizer-aware screening is reserved for six pairings.
    for coarse, ai, bi, a, b in options[:6]:
        sa = [[len(cell) for cell in row] for row in a]
        sb = [[len(cell) for cell in row] for row in b]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= W_MAX

        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 19 * n <= 40 * k < 21 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > W_MAX:
            continue

        # The strict existing screen returns None for any nontrivial
        # logical through weight three. Prefer its passing pairings;
        # never manufacture a passing result for a weaker fallback.
        pz = _r050_small_profile(hx, hz, n)
        px = _r050_small_profile(hz, hx, n)
        if px is not None and pz is not None:
            score = (
                0,
                max(px[0], pz[0]), px[0] + pz[0],
                max(px[1], pz[1]), px[1] + pz[1],
                max(px[2], pz[2]),
            ) + coarse
        else:
            # These designs remain eligible for the evaluator's screen,
            # but lose to every fully screened pairing in this slot.
            score = (1,) + coarse

        if best_score is None or score < best_score:
            best_score = score
            best = (a, b, hx, hz, k)

    if best is None:
        raise ValueError("no verified pairing remained in the 0.500 cell")
    a, b, hx, hz, k = best
    assert css_ok(hx, hz)
    return {"idx": idx, "design": design, "k": k,
            "cells": {"A": a, "B": b}, "screen": best_score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete pre-screened cell for either new shape."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap compliance with rolled labels is sufficient for low circuit failure in the q14 (6,2)x(6,2) bed; measured s=3.30-3.59, basis failures up to 113/250, and numerous weight-four sector-2 logicals contradict it.
# REFUTED: The systematic [1,3,3,3] A row plus degree-two B hook screen sufficiently suppresses collective circuit failures: fitness was 3.1012, measured codes failed 107-179/250 shots, all 213 logicals were affected, and weight-four witnesses remained abundant.
# REFUTED: Twenty parent structural plans guarantee twenty inequivalent codes: only 14 distinct fillings survived across 60 slots, with numerous isomorphic duplicates.
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
