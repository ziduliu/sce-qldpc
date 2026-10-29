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
CONCEPT = ("C3 semidirect C8 with inversion action, order 24, shapes (1,3)x(2,7): noncommuting trinomial A cells and balanced B-column weights should suppress short sector-1 component witnesses and fill the empty rate-0.450 cell.")
GROUP = ("A genuinely different group from C23. Elements of orders 6, 8 and 12 allow long binomial cycles outside a purely abelian labeling model. Strong measured nonabelian q=39 and q=44 beds motivate this test, but do not establish its distance.")
SHAPES = ("A=(1,3), B=(2,7), q=24 give n=552 and k>=240, guaranteed rate 10/23. Components have 72 columns/24 checks and 168 columns/48 checks. Parent minimum logical families were predominantly sector-1 operators across Acol/Bcol blocks; verified lifted ranks enforce 0.425<=k/n<0.475.")
FILLING = ("Twenty deliberate designs combine five B support geometries and four relative-element models. A cells have three terms; B rows each have nine terms and columns have two or three, so both row caps are at most 12. Screen actual lifted ranks, exhaustively detect component kernel witnesses through weight four, and shortlist by four-cycles before randomized kernel probes. Component results are not quantum-distance certificates.")
FAMILY_NAME = "r045_C3rtC8_inv_13x27_shortkernel"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # x^3=y^8=1, y*x*y^-1=x^-1.
    # The action has order two and therefore factors through C8.
    if not valid_q(q):
        raise ValueError("this group has order 24")
    els = [(a, b) for b in range(8) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 8)

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % 3,
                (g[1] + h[1]) % 8)

    def inv(g):
        bi = (-g[1]) % 8
        sign = -1 if bi % 2 else 1
        return ((-sign * g[0]) % 3, bi)

    def idx(g):
        a, b = red(g)
        return 3 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)

# Distinct support geometries, not column-permuted copies:
# their multisets of column splits differ.
_B_SUPPORTS = (
    ((2, 2, 1, 1, 1, 1, 1), (1, 1, 2, 2, 1, 1, 1)),
    ((3, 0, 2, 1, 1, 1, 1), (0, 3, 1, 2, 1, 1, 1)),
    ((3, 2, 1, 1, 1, 1, 0), (0, 1, 2, 1, 1, 1, 3)),
    ((3, 3, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 3, 3)),
    ((2, 2, 2, 0, 1, 1, 1), (1, 1, 0, 3, 2, 1, 1)),
)
# A twenty-design factorial experiment: support geometry x element model.
_DESIGNS = tuple((p, m) for p in range(5) for m in range(4))
_TRI_QUOTIENTS = ((1, 2), (1, 3), (2, 5), (3, 4))


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _component_probe(cols, rng, trials=56):
    """Find classical kernel witnesses; absence of small ones is not proof."""
    width = len(cols)
    best = width + 1
    for _ in range(trials):
        order = list(range(width))
        rng.shuffle(order)
        pivots = {}
        deps = []
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, support)
                    break
                old_syndrome, old_support = pivots[p]
                syndrome ^= old_syndrome
                support ^= old_support
            if not syndrome:
                deps.append(support)
                best = min(best, support.bit_count())

        # Elementary basis operations preserve nonzero kernel witnesses.
        for _ in range(3):
            changed = False
            for a in range(len(deps)):
                v = deps[a]
                weight = v.bit_count()
                for b in range(len(deps)):
                    if a == b:
                        continue
                    z = v ^ deps[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                deps[a] = v
                best = min(best, weight)
            if not changed:
                break
    return best


def _element_order(G, g):
    h = G.e
    for t in range(1, G.q + 1):
        h = G.mul(h, g)
        if G._key(h) == G._key(G.e):
            return t
    raise ValueError("invalid finite-group element")


def _designed_cell(rng, G, count, mode, serial):
    """Distinct terms with controlled relative orders and quotient support."""
    if count == 0:
        return []
    anchor = G.pick(rng)
    if count == 1:
        return [anchor]
    if count == 2:
        # Four genuinely different order profiles, rather than automorphism
        # copies of one labeling. Orders are invariant under automorphisms.
        wanted = ((8,), (12,), (8, 12), (6, 8, 12))[mode]
        order = wanted[serial % len(wanted)]
        choices = [g for g in G.elements if _element_order(G, g) == order]
        x = rng.choice(choices)
        return [anchor, G.mul(anchor, x)]

    u, v = _TRI_QUOTIENTS[mode]
    choices = [
        ((a, u), (b, v))
        for a in (1, 2) for b in (1, 2)
        if G._key(G.mul((a, u), (b, v))) !=
           G._key(G.mul((b, v), (a, u)))
    ]
    x, y = rng.choice(choices)
    cell = [anchor, G.mul(anchor, x), G.mul(anchor, y)]
    assert len({G._key(g) for g in cell}) == 3
    return cell


def _small_kernel_weight(cols):
    """Exact detection through weight four; zero means none was found."""
    singles = {}
    for i, s in enumerate(cols):
        if s == 0:
            return 1
        if s in singles:
            return 2
        singles[s] = i

    pairs = {}
    found_four = False
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            if s in singles:
                return 3
            support = (1 << i) | (1 << j)
            if s in pairs:
                # With distinct nonzero columns, distinct colliding pairs
                # cannot overlap: their XOR is a weight-four kernel vector.
                if (support ^ pairs[s]).bit_count() == 4:
                    found_four = True
            else:
                pairs[s] = support
    return 4 if found_four else 0


def NEW_CTX(rng, G, idx):
    """Optimize labels within one of twenty deliberate structural designs."""
    profile, mode = _DESIGNS[idx % len(_DESIGNS)]
    sa = [[3, 3, 3]]
    sb = [list(row) for row in _B_SUPPORTS[profile]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    q = G.q
    mask = (1 << q) - 1
    pool = []

    for attempt in range(192):
        A = [[_designed_cell(rng, G, 3, mode, j)
              for j in range(ca)]]
        B = [[_designed_cell(rng, G, sb[i][j], mode, i * cb + j)
              for j in range(cb)] for i in range(rb)]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 17 * n <= 40 * k < 19 * n):
            continue

        # Components are extracted exclusively from the verified lift.
        arows = [
            sum(((row >> (j * cb * q)) & mask) << (j * q)
                for j in range(ca))
            for row in hx[:q]
        ]
        brows = [row & ((1 << (cb * q)) - 1)
                 for row in hz[:rb * q]]
        ac = _component_columns(arows, ca * q)
        bc = _component_columns(brows, cb * q)
        wa = _small_kernel_weight(ac)
        wb = _small_kernel_weight(bc)
        cycles = (_component_four_cycles(ac),
                  _component_four_cycles(bc))

        bad = int(bool(wa)) + int(bool(wb))
        tiny = int(0 < wa <= 3) + int(0 < wb <= 3)
        pool.append((tiny, bad, max(cycles), sum(cycles), attempt,
                     k, A, B, ac, bc, wa, wb))
        # Continue longer when every retained candidate has a short witness.
        if len(pool) >= 48 and any(p[1] == 0 for p in pool):
            break

    if not pool:
        raise ValueError("no positive-k candidate in rate cell 0.450")
    pool.sort(key=lambda x: x[:5])
    best = None
    best_score = None

    for candidate in pool[:5]:
        tiny, bad, worst_cycles, total_cycles, attempt, k, A, B, ac, bc, wa, wb = candidate
        da = _component_probe(ac, rng, trials=12)
        db = _component_probe(bc, rng, trials=12)
        if wa:
            da = min(da, wa)
        if wb:
            db = min(db, wb)
        d = min(da, db)
        score = (k * d * d, d, da + db,
                 -tiny, -bad, -worst_cycles, -total_cycles)
        if best_score is None or score > best_score:
            best_score = score
            best = (A, B)

    A, B = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert 0 < k and 17 * n <= 40 * k < 19 * n
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# Refuted by current slots 0/30/34: balanced C23 component-kernel screening does not eliminate sector-1 minimum logical families; measured witnesses remain at weights 14-16.
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
