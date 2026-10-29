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
CONCEPT = ("Circuit-oriented r050 design: metacyclic Z7:Z2 with s=6, q=14, shapes (1,4)x(2,7). A monomial pivot plus three trinomials and degree-two B columns should reduce circuit failures through fewer extraction gates and fewer two-fault opportunities for weight-four logicals.")
GROUP = ("Keep the parent's order-14 metacyclic group. Its fitness 3.4511, diffuse 1.0-1.2x failure enrichment, and numerous weight-four sector-2 witnesses motivate a component-geometry experiment rather than establish a group defect.")
SHAPES = ("Target cell 0.500 with (1,4)x(2,7): n=420 and guaranteed k>=210, rate>=0.500. A is a length-four single-check-block component and B a length-seven two-check-block component. This replaces the parent's tall length-six components and moves the component-kernel sector toward Acol/Bcol; exact lifted ranks keep submitted slots below rate 0.525.")
FILLING = ("A has term counts [1,3,3,3]. B has two terms per column and seven per row, giving HX weight 12, HZ weights 8 or 10, and 13440 circuit CNOTs versus the parent's measured 18816-19488. Twenty structural plans vary pure/mixed trinomials, B coset balance, and paired zero cells. Each searches 24 fillings using verified lifts, rejects non-stabilizer logicals of weight <=3, and minimizes shared-check two-pair opportunities and weight-four logical multiplicity across both bases. This targets syndrome-valid logical slips despite early BP settlement; the hook count is a surrogate, not an ASC simulation.")
FAMILY_NAME = "r050_m7r2s6_14x27_hook_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # metacyclic Z_7 x| Z_2: y x y^-1 = x^6 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 7, 2, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _r050_cells(rng, G, idx):
    """Twelve dense-B plans and eight paired-zero-B plans, not orbit copies."""
    plan = idx % 20
    paired = plan >= 12
    if paired:
        pure = (plan - 12) % 3
        kind = (plan - 12) // 3
    else:
        pure = plan // 3
        kind = plan % 3

    # Pure rotational triples: distinct translation classes. AP triples
    # have three classes; non-AP triples have two. Alternation needs <=2.
    pools = {
        0: [[0, d, (2 * d) % 7] for d in (1, 2, 3)],
        1: [[0, 1, 3], [0, 1, 5]],
    }
    for pool in pools.values():
        rng.shuffle(pool)

    arow = [[G.e]]
    for t in range(3):
        if t < pure:
            offsets = pools[(kind + t) % 2].pop()
            cell = [(x, 0) for x in offsets]
        else:
            # Two rotations and one reflection; independent relative
            # phases, not a global automorphism of one fixed labeling.
            d = rng.randrange(1, 7)
            v = rng.randrange(7)
            cell = [G.e, (d, 0), (v, 1)]
        arow.append(cell)

    b = [[[] for _ in range(7)] for _ in range(2)]
    if not paired:
        reflections = 3 - kind
        rotations = rng.sample([(x, 0) for x in range(7)],
                               7 - reflections)
        reflected = rng.sample([(x, 1) for x in range(7)],
                               reflections)
        voltages = sorted(rotations + reflected, key=G.idx)
        for j, g in enumerate(voltages):
            b[0][j] = [G.e]
            b[1][j] = [g]
    else:
        # One (2,0) column and one (0,2) column preserve both row totals.
        # Their rotation ratio has order seven, avoiding doubled columns.
        d = rng.randrange(1, 7)
        step = (3 * d) % 7
        b[0][0] = [G.e, (step, 0)]
        b[1][1] = [G.e, (step, 0)]

        # Cross-row rotation differences use only +/-d and +/-2d,
        # not the within-row step +/-3d: no component triangles.
        nr = 3 if kind % 2 == 0 else 2
        u, v = rng.randrange(7), rng.randrange(7)
        voltages = [((u + h * d) % 7, 0) for h in range(nr)]
        voltages += [((v + h * d) % 7, 1)
                     for h in range(5 - nr)]
        for j, g in enumerate(sorted(voltages, key=G.idx), 2):
            b[0][j] = [G.e]
            b[1][j] = [g]

    return [arow], b


def NEW_CTX(rng, G, idx):
    """Optimize the slot's structural plan using the evaluator's exact lift."""
    best = None
    best_score = None
    seen = set()
    for _ in range(24):
        a, b = _r050_cells(rng, G, idx)
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (a, b) for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)

        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (475 * n <= 1000 * k < 525 * n):
            continue
        if max(_popcount(row) for row in hx + hz) > W_MAX:
            continue

        # Z errors have HX syndromes and HZ ancilla-spread opportunities;
        # X errors have HZ syndromes and HX spread opportunities.
        pz = _r050_small_profile(hx, hz, n)
        if pz is None:
            continue
        px = _r050_small_profile(hz, hx, n)
        if px is None:
            continue
        score = (
            max(px[0], pz[0]), px[0] + pz[0],
            max(px[1], pz[1]), px[1] + pz[1],
            max(px[2], pz[2]),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = (a, b, hx, hz)

    if best is None:
        raise ValueError("r050 plan found no rate-legal code without small logicals")
    a, b, hx, hz = best
    if not css_ok(hx, hz):
        raise ValueError("verified lift failed CSS check")
    return {"idx": idx, "cells": {"A": a, "B": b},
            "screen": best_score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete pre-screened cell for either new shape."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap compliance with rolled labels is sufficient for low circuit failure in the q14 (6,2)x(6,2) bed; measured s=3.30-3.59, basis failures up to 113/250, and numerous weight-four sector-2 logicals contradict it.
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
