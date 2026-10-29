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
CONCEPT = ("Test whether balanced three-/four-term cells on the Frobenius group C37:C3, q=111, shapes (1,2)x(1,2), screened against component logicals, raise the top-five measured distance floor above the parent's 7.")
GROUP = ("C37:C3 uses the order-three action x -> x^10, since 10^3 = 1 mod 37. Its odd order removes involutions and gives n=555. Strong nonabelian q=39/q=44 low-rate beds motivate this untested bed, without predicting its distance.")
SHAPES = ("Keep (1,2)x(1,2), with equal 111-check/222-variable components and guaranteed k>=111, rate 0.200. Parent minimum witnesses were sector-1 operators spanning the four Acol/Bcol blocks; screen both corresponding component kernels.")
FILLING = ("Twenty indexed count/coset plans use only three or four distinct terms per cell, with both check-weight sums <=12. Each searches twelve admissible labelings using verified lifted matrices, rank-based rate gating, exact weight<=4 component witnesses, and randomized kernel descent with stabilizer-membership tests. Witness readings are upper bounds, not certificates.")
FAMILY_NAME = "r020_frob37c3_12x12_balanced_kernel"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C37:C3 requires order 111")
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    def mul(g, h):
        return ((g[0] + pow(s, g[1] % r, m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _fp_basis(rows):
    pivots = {}
    for v in rows:
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    return pivots


def _fp_reduce(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return v
        v ^= pivots[p]
    return 0


def _fp_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _fp_cycles(columns):
    total = 0
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            overlap = (a & b).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _fp_witness(rng, columns, embed, stabilizers):
    """Search component kernels; accept only non-stabilizer embeddings.
    All parity columns originate in lifted_matrices, not a second lift.
    The returned weight is an upper bound from found witnesses."""
    width = len(columns)
    best = width + 1

    def observe(word):
        nonlocal best
        weight = word.bit_count()
        if word and weight < best:
            if _fp_reduce(embed(word), stabilizers):
                best = weight

    # Exhaustive syndrome collisions cover component weights 1..4.
    singles = {}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        if syndrome == 0:
            observe(word)
        if syndrome in singles:
            observe(word ^ singles[syndrome])
        else:
            singles[syndrome] = word
    pairs = {}
    for i in range(width):
        for j in range(i + 1, width):
            syndrome = columns[i] ^ columns[j]
            word = (1 << i) | (1 << j)
            if syndrome in singles:
                observe(word ^ singles[syndrome])
            if syndrome in pairs:
                observe(word ^ pairs[syndrome])
            else:
                pairs[syndrome] = word
    if best <= 4:
        return best

    # Random information sets produce exact nullvectors. Greedy XOR
    # descent stays in the kernel and never accepts the zero vector.
    for _ in range(8):
        order = list(range(width))
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome = columns[c]
            word = 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in pivots:
                    old_syndrome, old_word = pivots[p]
                    syndrome ^= old_syndrome
                    word ^= old_word
                else:
                    pivots[p] = (syndrome, word)
                    break
            if syndrome == 0:
                words.append(word)
        words.sort(key=int.bit_count)
        for start in words[:8]:
            observe(start)
            word = start
            for _ in range(5):
                improved = word
                weight = word.bit_count()
                for other in words:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        improved, weight = trial, tw
                if improved == word:
                    break
                word = improved
                observe(word)
    return best


def _fp_draw_cells(rng, idx):
    # Five weight profiles x four quotient-support models = twenty
    # deliberately different design constraints, not automorphism copies.
    profiles = (
        (3, 4, 3, 4),
        (3, 4, 4, 4),
        (4, 4, 3, 4),
        (4, 4, 4, 4),
        (3, 3, 4, 4),
    )
    plan = idx % 20
    profile, mode = plan // 4, plan % 4
    flat = []
    for c, count in enumerate(profiles[profile]):
        if mode == 0:
            cosets = [0, 1, 2]
            if count == 4:
                cosets.append((c + profile) % 3)
        elif mode == 1:
            cosets = [0, 0, 1] if count == 3 else [0, 0, 1, 2]
        elif mode == 2:
            cosets = [0, 1, 1] if count == 3 else [0, 0, 1, 1]
        else:
            cosets = [0, 1, 2] if count == 3 else [0, 0, 0, 1]
        shift = (c + profile) % 3
        cosets = [(b + shift) % 3 for b in cosets]
        cell = []
        for b in range(3):
            amount = cosets.count(b)
            cell.extend((a, b) for a in rng.sample(range(37), amount))
        flat.append(cell)
    return [flat[:2]], [flat[2:]]


def NEW_CTX(rng, G, idx):
    """Optimize labels within the slot's fixed design constraints."""
    q = G.q
    mask = (1 << q) - 1
    best = None
    best_score = None
    accepted = 0

    for _ in range(96):
        A, B = _fp_draw_cells(rng, idx)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if _grid_weights(sa, sb) > W_MAX:
            raise ValueError("deliberate support profile exceeds weight cap")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0 or not (0.175 <= k / n < 0.225):
            continue

        # Extract the two sector-1 component parity matrices directly
        # from the verified construction, preserving its lift actions.
        ar = [(row & mask) | (((row >> (2 * q)) & mask) << q)
              for row in hx[:q]]
        br = [row & ((1 << (2 * q)) - 1) for row in hz[:q]]
        ac = _fp_columns(ar, 2 * q)
        bc = _fp_columns(br, 2 * q)

        # A-kernel words are Z candidates at Bcol0; B-kernel words
        # are X candidates at Acol0. Test against the actual gauge spans.
        embed_a = lambda word: ((word & mask) |
                                ((word >> q) << (2 * q)))
        embed_b = lambda word: word
        da = _fp_witness(rng, ac, embed_a, _fp_basis(hz))
        db = _fp_witness(rng, bc, embed_b, _fp_basis(hx))
        found = min(da, db, 24)
        cycles = _fp_cycles(ac) + _fp_cycles(bc)
        score = (k * found * found / n,
                 min(da, db), da + db, -cycles, -(k - q))
        if best_score is None or score > best_score:
            best_score = score
            best = (A, B, hx, hz)
        accepted += 1
        if accepted == 12:
            break

    if best is None:
        raise ValueError("no candidate passed the target-rate screen")
    A, B, hx, hz = best
    if not css_ok(hx, hz):
        raise ValueError("verified lift failed CSS commutation")
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional random filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: uniform labels with legal rolled counts alone exclude light logicals; the parent found many weight-2 witnesses and a measured top-five witness of weight 7.
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
