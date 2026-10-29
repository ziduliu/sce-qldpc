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
CONCEPT = ("Target cell 0.300 with the affine group AGL(1,5), order 20, shapes (2,4)x(2,6): balanced component degrees and exclusion of involutory doubleton ratios should improve worse-basis circuit performance over the destination frontier's rolled cyclic filling.")
GROUP = ("AGL(1,5) consists of invertible affine maps over F5 and is genuinely different from the parent's order-21 group. Its trivial centre is shared by the successful order-21, order-39 and order-42 beds, but those readings do not isolate a group effect. Ratios of slopes 2 or 3 have order four; nonidentity translations have order five. This order-20 bed is absent from RUN STATE.")
SHAPES = ("Shapes (2,4)x(2,6) give 80- and 120-variable components, n=28*20=560 and guaranteed k>=160, rate 2/7. Verified lifted ranks require 0.275<=k/n<0.325. The destination 0.300 frontier scores only 4.2. The parent's five measured codes had zero failures in both bases, so no Acol/Brow failure hotspot is established; this is a frontier-directed redesign, not a local failure-map repair.")
FILLING = ("Twenty explicit architectures cross three degree-three A profiles of different row balance with six B profiles, plus two degree-four A designs. Every B column has degree two to four and at most three have degree two. Both CSS row-weight sums are <=12. Doubletons use distinct labels with relative order four or five. Search verified lifts for connected components without kernel words of weight <=4, exact target-band ranks and low four-cycle counts; refine finalists by small-boundary triples and triple-syndrome collisions. Additional harness slots change the ratio-order model rather than deliberately relabeling a previous code. Circuit pL, not these screening statistics, decides success.")
FAMILY_NAME = "r030_agl1_f5_24x26_order45_balanced"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,5) requires q=20")
    # (u,v) represents z -> u*z+v over F5, with u nonzero.
    # Multiplication is composition of affine maps.
    els = [(u, v) for u in range(1, 5) for v in range(5)]

    def red(g):
        u, v = int(g[0]) % 5, int(g[1]) % 5
        if not u:
            raise ValueError("An affine group element needs nonzero slope")
        return (u, v)

    def mul(g, h):
        return ((g[0] * h[0]) % 5,
                (g[1] + g[0] * h[1]) % 5)

    def inv(g):
        ui = pow(g[0], 3, 5)
        return (ui, (-ui * g[1]) % 5)

    def idx(g):
        u, v = red(g)
        return 5 * (u - 1) + v

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _affine_columns(hx, hz, q):
    """Project component columns from the VERIFIED lifted matrices.

    A: sector 1 of X checks at B-column zero.
    B: sector 1 of Z checks at A-column zero.
    No independent left/right-lift implementation is used.
    """
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    ac = [0] * (ca * q)
    bc = [0] * (cb * q)
    amask = sum(((1 << q) - 1) << (j * cb * q)
                for j in range(ca))
    bmask = (1 << (cb * q)) - 1
    for i in range(ra):
        for r in range(q):
            bits = hx[(i * cb) * q + r] & amask
            check = 1 << (i * q + r)
            while bits:
                low = bits & -bits
                c = low.bit_length() - 1
                ac[(c // (cb * q)) * q + c % q] |= check
                bits ^= low
    for a in range(rb):
        for r in range(q):
            bits = hz[a * q + r] & bmask
            check = 1 << (a * q + r)
            while bits:
                low = bits & -bits
                bc[low.bit_length() - 1] |= check
                bits ^= low
    return ac, bc


def _affine_stats(cols, nchecks):
    """Exact short-word counts, Tanner four-cycles and connectivity."""
    parent = list(range(nchecks))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checkpairs = {}
    c4 = 0
    for col in cols:
        rows = []
        bits = col
        while bits:
            low = bits & -bits
            rows.append(low.bit_length() - 1)
            bits ^= low
        if rows:
            for r in rows[1:]:
                parent[root(r)] = root(rows[0])
        for t, r in enumerate(rows):
            for s in rows[t + 1:]:
                pair = (r, s)
                old = checkpairs.get(pair, 0)
                c4 += old
                checkpairs[pair] = old + 1

    colset = set(cols)
    repeated = len(cols) - len(colset)
    pairseen = {}
    words3 = words4 = 0
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            syndrome = x ^ y
            words3 += int(syndrome in colset)
            old = pairseen.get(syndrome, 0)
            words4 += old
            pairseen[syndrome] = old + 1
    components = len({root(r) for r in range(nchecks)})
    return (repeated, words3, words4), c4, components


def _affine_traps(cols):
    """Small boundaries of three-variable sets; repeated triple syndromes.

    The first statistic is a trapping-set heuristic. The second also
    detects weight-six component words when shorter words are absent.
    Neither statistic is asserted to predict circuit pL exactly.
    """
    boundary = repeated = 0
    seen = {}
    count = len(cols)
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, count):
                syndrome = pair ^ cols[k]
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
                old = seen.get(syndrome, 0)
                repeated += old
                seen[syndrome] = old + 1
    return boundary, repeated


def _affine_supports(idx):
    """Twenty explicit architectures with distinct degree/balance profiles."""
    ap = (
        ((2, 2, 1, 1), (1, 1, 2, 2)),  # columns 3; rows 6,6
        ((2, 2, 2, 1), (1, 1, 1, 2)),  # columns 3; rows 7,5
        ((2, 2, 2, 2), (1, 1, 1, 1)),  # columns 3; rows 8,4
        ((2, 2, 2, 2), (2, 2, 2, 2)),  # columns 4; rows 8,8
    )
    bp = (
        ((2, 2, 2, 1, 1, 1), (1, 1, 1, 2, 2, 2)),
        ((2, 2, 2, 1, 1, 1), (1, 1, 1, 2, 2, 1)),
        ((2, 2, 1, 1, 1, 1), (1, 1, 2, 2, 1, 1)),
        ((2, 2, 2, 1, 1, 1), (1, 1, 1, 2, 1, 1)),
        ((2, 2, 1, 1, 1, 1), (1, 1, 2, 1, 1, 1)),
        ((2, 2, 1, 1, 1, 1), (2, 1, 2, 1, 1, 1)),
    )
    pairs = tuple((a, b) for a in range(3) for b in range(6))
    pairs += ((3, 2), (3, 4))
    a, b = pairs[idx % len(pairs)]
    return [list(row) for row in ap[a]], [list(row) for row in bp[b]]


def _affine_grid(rng, G, counts, mode, offset):
    """Doubleton ratios have order four or five, never order two."""
    translations = [(1, v) for v in range(1, 5)]
    quartic = [(u, v) for u in (2, 3) for v in range(5)]
    all_ratios = translations + quartic
    grid = []
    for i, row in enumerate(counts):
        out = []
        for j, weight in enumerate(row):
            g = G.pick(rng)
            cell = [g]
            if weight == 2:
                if mode == 1:
                    ratios = quartic
                elif mode == 2:
                    ratios = (translations if (i + j + offset) % 3
                              else quartic)
                else:
                    ratios = all_ratios
                ratio = ratios[rng.randrange(len(ratios))]
                cell.append(G.mul(g, ratio))
                assert G._key(cell[0]) != G._key(cell[1])
            assert len(cell) == weight
            out.append(cell)
        grid.append(out)
    return grid


def NEW_CTX(rng, G, idx):
    """Search each architecture independently using only verified lifts."""
    sa, sb = _affine_supports(idx)
    assert _no_dead_lines(sa, sb)
    # HX: max A-row + max B-column.
    # HZ: max B-row + max A-column.
    assert _grid_weights(sa, sb) <= 12
    assert min(sum(sa[i][j] for i in range(2))
               for j in range(4)) >= 3
    assert min(sum(sb[i][j] for i in range(2))
               for j in range(6)) >= 2

    # Additional slots change the order model: unrestricted orders 4/5,
    # order 4 only, or a block-dependent translation-heavy design.
    mode = (idx // 20) % 3
    shortlist = []
    for batch in range(3):
        for trial in range(192):
            A = _affine_grid(rng, G, sa, mode, idx)
            B = _affine_grid(rng, G, sb, mode, idx + 1)
            hx, hz, n = lifted_matrices(G, A, B)
            ac, bc = _affine_columns(hx, hz, G.q)
            ast = _affine_stats(ac, A_SHAPE[0] * G.q)
            bst = _affine_stats(bc, B_SHAPE[0] * G.q)
            if ast[2] != 1 or bst[2] != 1:
                continue
            # These are component kernel tests, not asserted quantum
            # distance bounds or substitutes for circuit measurement.
            if any(ast[0]) or any(bst[0]):
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 11 * n <= 40 * k < 13 * n):
                continue
            base = (max(ast[1], bst[1]), ast[1] + bst[1])
            serial = batch * 192 + trial
            shortlist.append((base, -k, serial, A, B, ac, bc, hx, hz))
            shortlist.sort(key=lambda item: item[:3])
            del shortlist[6:]
        if shortlist:
            break

    if not shortlist:
        raise ValueError("No connected short-word-free filling in cell 0.300")

    winner = None
    winning_key = None
    for base, negk, serial, A, B, ac, bc, hx, hz in shortlist:
        at = _affine_traps(ac)
        bt = _affine_traps(bc)
        boundary = (max(at[0], bt[0]), at[0] + bt[0])
        sixwords = (max(at[1], bt[1]), at[1] + bt[1])
        refinement = ((sixwords, boundary) if mode == 1
                      else (boundary, sixwords))
        key = (base, refinement, negk, serial)
        if winning_key is None or key < winning_key:
            winning_key = key
            winner = (A, B, hx, hz)

    A, B, hx, hz = winner
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": winning_key}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, with no late label changes."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's large q=58 single-row uniform-label seed reliably suppresses collective circuit failures; measured worst-basis s=3.17-3.28, widespread logical flips and many sector-1 weight-four witnesses contradict this.
# REFUTED: Increasing d2s alone increases measured s: current parent slot 16 had d2s=21 and s=6.874, below slot 28 with d2s=18 and s=6.989; both had zero observed failures and saturated at their k-dependent ceilings.
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
