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
CONCEPT = ("On AGL(1,9), order 72, shapes (1,2)x(1,3), mixed-coset trinomials without involution-separated terms, selected against short verified kernels, should raise five-code fitness above the 0.300 cell record of 23.0.")
GROUP = ("Untried finite-field affine family: AGL(1,9) has noncyclic translation subgroup C3xC3, unlike the parent's cyclic Z21. Nonabelian single-row beds reached fitness 81.1 at q=111 and 25.6 at q=58; these motivate, but do not validate, this experiment. The prescribed order 72 gives n=504.")
SHAPES = ("Target 0.300: A=(1,2), B=(1,3), n=7*72=504, guaranteed k>=144 and rate 2/7. Components have 144 variables/72 checks and 216 variables/72 checks. Parent distance-six witnesses were entirely sector-one, often involving Acol1/Acol2 and spread across B columns; dense single-row components replace the weak singleton-heavy columns.")
FILLING = ("Twenty primary cards cross five A coset architectures with four B architectures. Primary cells are trinomials: HX weight 9, HZ weight 12. Optimize 32 labelings per card using verified lifts, exact target-band ranks, kernel collisions through weight four, sampled component kernel witnesses and four-cycle counts; rescreen four finalists more deeply, including full-check short kernels. Extra slots use two asymmetric legal budgets. Kernel readings are upper-bound heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r030_agl1_f9_12x13_coset_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 72

def custom_ops(q):
    # AGL(1,F9): z -> alpha^b*z + u + v*t.
    # F9 = F3[t]/(t^2+1); alpha=1+t has order eight.
    if q != 72:
        raise ValueError("AGL(1,9) has order 72")

    def fm(x, y):
        u, v = x
        a, b = y
        return ((u*a + 2*v*b) % 3, (u*b + v*a) % 3)

    powers = [(1, 0)]
    for _ in range(7):
        powers.append(fm(powers[-1], (1, 1)))
    els = [(u, v, b) for b in range(8)
           for v in range(3) for u in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 3, g[2] % 8)

    def mul(g, h):
        u, v, b = red(g)
        a, c, d = red(h)
        x, y = fm(powers[b], (a, c))
        return ((u+x) % 3, (v+y) % 3, (b+d) % 8)

    def inv(g):
        u, v, b = red(g)
        bi = (-b) % 8
        x, y = fm(powers[bi], ((-u) % 3, (-v) % 3))
        return (x, y, bi)

    def idx(g):
        u, v, b = red(g)
        return 9*b + 3*v + u

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Relative complement exponents distinguish translation/order-eight,
# order-eight/order-four, and translation/order-four cell architectures.
_A_CARDS = ((0, 0), (0, 1), (1, 1), (0, 2), (2, 1))
_B_CARDS = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))


def _affine_cell(rng, G, pools, weight, kind):
    """Translate a prescribed relative-coset architecture."""
    if weight == 2:
        exponents = ((1,), (2,), (2,))[kind]
    elif weight == 3:
        exponents = ((0, 1), (1, 2), (0, 2))[kind]
    elif weight == 4:
        exponents = ((0, 1, 2), (1, 1, 2), (0, 2, 3))[kind]
    else:
        raise ValueError("unsupported affine support weight")
    relative = [G.e]
    for b in exponents:
        choices = [g for g in pools[b] if g not in relative]
        relative.append(rng.choice(choices))
    # No exponent difference is four, and nonzero translations have
    # order three. Thus no pair is separated by an involution.
    base = G.pick(rng)
    return [G.mul(base, g) for g in relative]


def _affine_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= tag
            row ^= bit
    return columns


def _affine_components(hx, hz, q):
    """Extract components only from the scaffold's verified LP lift."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    ar = []
    for source in hx[:q]:
        packed = 0
        for j in range(ca):
            packed |= ((source >> (j*cb*q)) & mask) << (j*q)
        ar.append(packed)
    bmask = (1 << (cb*q)) - 1
    br = [source & bmask for source in hz[:q]]
    return (ar, br, _affine_columns(ar, ca*q),
            _affine_columns(br, cb*q))


def _affine_short4(columns):
    """Return a witnessed kernel weight <=4, or None; exhaustive."""
    seen = {0: 0}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        old = seen.get(syndrome)
        if old is not None:
            return (old ^ word).bit_count()
        seen[syndrome] = word
    for i, syndrome in enumerate(columns):
        word = 1 << i
        for j in range(i):
            s = syndrome ^ columns[j]
            v = word | (1 << j)
            old = seen.get(s)
            if old is not None:
                return (old ^ v).bit_count()
            seen[s] = v
    return None


def _affine_probe(columns, rng, trials):
    """Sample actual classical kernel words, not distance certificates."""
    width = len(columns)
    best = width + 1
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        for i, word in enumerate(words):
            for other in words[:i]:
                best = min(best, (word ^ other).bit_count())
        seeds = sorted(words, key=int.bit_count)[:6]
        for word in seeds:
            weight = word.bit_count()
            for _ in range(3):
                changed = False
                for other in words:
                    candidate = word ^ other
                    w = candidate.bit_count()
                    if candidate and w < weight:
                        word, weight = candidate, w
                        changed = True
                if not changed:
                    break
            best = min(best, weight)
        if best <= 4:
            break
    return best


def _affine_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap-1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Optimize a deliberate support/coset card; retain legal fallbacks."""
    card = idx % 20
    akinds = _A_CARDS[card // 4]
    bkinds = _B_CARDS[card % 4]
    budget = (idx // 20) % 3
    if budget == 0:
        aw, bw = (3, 3), (3, 3, 3)
    elif budget == 1:
        aw, bw = (4, 4), (2, 3, 3)
    else:
        aw, bw = (3, 3), (2, 3, 4)

    # Primary HX/HZ maxima: 9/12.
    # Additional budgets: 11/12 and 10/12.
    sa, sb = [list(aw)], [list(bw)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    pools = [[g for g in G.elements if g[2] == b] for b in range(8)]
    kb = 2 * G.q
    finalists = []

    for attempt in range(32):
        A = [[_affine_cell(rng, G, pools, aw[j], akinds[j])
              for j in range(2)]]
        B = [[_affine_cell(rng, G, pools, bw[j], bkinds[j])
              for j in range(3)]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (275*n <= 1000*k < 325*n):
            continue

        ar, br, ac, bc = _affine_components(hx, hz, G.q)
        short_a = _affine_short4(ac)
        short_b = _affine_short4(bc)
        clean = short_a is None and short_b is None
        trials = 8 if clean else 2
        da = _affine_probe(ac, rng, trials)
        db = _affine_probe(bc, rng, trials)
        if short_a is not None:
            da = min(da, short_a)
        if short_b is not None:
            db = min(db, short_b)
        cycles = _affine_cycles(ar) + _affine_cycles(br)
        key = (int(clean), min(da, db), da+db, -cycles, -(k-kb))
        finalists.append((key, A, B, hx, hz, ac, bc,
                          da, db, cycles, k, n))
        finalists.sort(key=lambda item: item[0], reverse=True)
        del finalists[4:]

    if not finalists:
        raise ValueError("no exact-rank filling in target rate cell")

    winner = None
    for _, A, B, hx, hz, ac, bc, da, db, cycles, k, n in finalists:
        da = min(da, _affine_probe(ac, rng, 48))
        db = min(db, _affine_probe(bc, rng, 48))
        sx = _affine_short4(_affine_columns(hx, n))
        sz = _affine_short4(_affine_columns(hz, n))
        full_clean = sx is None and sz is None
        reading = min(da, db)
        if sx is not None:
            reading = min(reading, sx)
        if sz is not None:
            reading = min(reading, sz)
        # Full-kernel rejection is conservative: it is not a logical test.
        key = (int(full_clean), reading, da+db, -cycles, -(k-kb))
        if winner is None or key > winner[0]:
            winner = (key, A, B, hx, hz, k)

    key, A, B, hx, hz, k = winner
    assert k > 0 and css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": key, "k": k}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled singleton-heavy supports with uniform Z21 labels suffice to obtain a top-five distance reading above six on (2,4)x(2,6); all five measured parent readings were six, with fitness 10.482.
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
