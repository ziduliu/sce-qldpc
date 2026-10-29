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
CONCEPT = ("Metacyclic Z8:Z2 with inversion s=7, q=16, shapes (2,5)x(2,6): column-weight-three components and screened short syndrome collisions should reduce circuit failures relative to the rolled single-row parent.")
GROUP = ("Keep the metacyclic construction and inversion action. The related q=28 (2,4)x(2,4) bed scored 5.766; this motivates testing, not a performance guarantee. For size factor 34, q=16 is the largest compatible order with r=2 and inversion.")
SHAPES = ("Target cell 0.350: n=34*16=544 and kb=12*16=192, rate 12/34=0.352941. A and B have component lengths 80 and 96 and 32 checks each. Two-row constraints replace the parent's single-row components. Parent failure enrichments of only 1.0-1.1 across Acol/Bcol blocks motivate global strengthening rather than targeting one block.")
FILLING = ("Twenty deliberate support profiles combine five A partitions with four B partitions; every component column has three terms. A row totals are 7/8 or 6/9, B row totals are 9/9: HX <=12 and HZ=12. Multi-term cells use one rotation coset with distinct ordered differences. Each slot searches 96 labelings using verified lifts, retains six, screens exact k=192, and minimizes short full-code and component syndrome collisions plus component cycles. These screens are hypotheses for lower circuit failure, not the fitness.")
FAMILY_NAME = "r035_m8r2s7_25x26_col3_collision"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Same metacyclic rule, resized for the new shape:
    # Z_8 x| Z_2, y x y^-1 = x^7, with 7^2 = 1 mod 8.
    m, r, s = 8, 2, 7
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _balanced_scores(a, b):
    """Worst basis/component first, then total multiplicity."""
    return tuple(v for x, y in zip(a, b) for v in (max(x, y), x + y))


def _syndrome_columns(rows, n):
    """Columns of the supplied, verified binary matrix."""
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _pair_collisions(cols):
    """Weight-1/2 witnesses, then pair-syndrome collisions.

    When there are no weight-1/2 witnesses, a collision between
    different pairs gives a weight-four zero-syndrome word.
    """
    freq = {}
    for s in cols:
        freq[s] = freq.get(s, 0) + 1
    twos = sum(v * (v - 1) // 2 for v in freq.values())
    pairs = {}
    fours = 0
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            old = pairs.get(s, 0)
            fours += old
            pairs[s] = old + 1
    return freq.get(0, 0), twos, fours


def _triple_collisions(cols):
    """Six-term collision proxy; exact disjointness follows if d>4."""
    seen = {}
    total = 0
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for k in range(j):
                s = xy ^ cols[k]
                old = seen.get(s, 0)
                total += old
                seen[s] = old + 1
    return total


def _four_cycles(rows):
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            t = (x & rows[j]).bit_count()
            total += t * (t - 1) // 2
    return total


def _six_cycles(rows):
    """Count check-triangle cycles with three distinct variable nodes."""
    total = 0
    for i in range(2, len(rows)):
        x = rows[i]
        for j in range(1, i):
            y = rows[j]
            xy = (x & y).bit_count()
            if not xy:
                continue
            for k in range(j):
                z = rows[k]
                xz = (x & z).bit_count()
                yz = (y & z).bit_count()
                t = (x & y & z).bit_count()
                total += xy * xz * yz - t * (xy + xz + yz) + 2 * t
    return total


def _verified_components(G, hx, hz):
    """Extract component matrices from the scaffold's actual lifts."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]   # beta=0
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask          # j=0
          for a in range(rb) for t in range(q)]
    return ar, br


def _profile_counts(rng, upper):
    """Column permutations are search freedom, not separate designs."""
    upper = list(upper)
    rng.shuffle(upper)
    counts = [upper, [3 - x for x in upper]]
    if rng.randrange(2):
        counts.reverse()
    return counts


def _coset_cells(rng, G, counts, triples):
    """No mixed-coset binomials with reflection-ratio involutions."""
    m = G.q // 2
    cells = []
    for row in counts:
        out = []
        for weight in row:
            if weight == 0:
                out.append([])
                continue
            g = G.pick(rng)
            if weight == 1:
                offsets = (0,)
            elif weight == 2:
                # Odd rotation difference has maximum order eight.
                offsets = (0, rng.randrange(1, m, 2))
            else:
                offsets = triples[rng.randrange(len(triples))]
            out.append([G.mul(g, (d, 0)) for d in offsets])
        cells.append(out)
    return cells


def NEW_CTX(rng, G, idx):
    """Optimize one of twenty genuinely different support profiles."""
    aprofiles = (
        (1, 1, 1, 2, 2),       # dense, row totals 7/8
        (1, 1, 1, 1, 2),       # dense, row totals 6/9
        (0, 1, 2, 2, 2),       # one concentrated column, 7/8
        (0, 1, 1, 2, 2),       # one concentrated column, 6/9
        (0, 1, 1, 2, 3),       # opposite concentrated columns, 7/8
    )
    bprofiles = (
        (1, 1, 1, 2, 2, 2),
        (0, 1, 2, 2, 2, 2),
        (0, 1, 1, 2, 2, 3),
        (0, 0, 2, 2, 2, 3),
    )
    ap = aprofiles[(idx // 4) % len(aprofiles)]
    bp = bprofiles[idx % len(bprofiles)]
    m = G.q // 2
    triples = []
    for a in range(1, m):
        for b in range(a + 1, m):
            offsets = (0, a, b)
            diffs = [(u - v) % m for u in offsets
                     for v in offsets if u != v]
            if len(set(diffs)) == 6:
                triples.append(offsets)
    assert triples

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = G.q * (ca - ra) * (cb - rb)
    pool = []
    for attempt in range(96):
        sa = _profile_counts(rng, ap)
        sb = _profile_counts(rng, bp)
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= W_MAX
        A = _coset_cells(rng, G, sa, triples)
        B = _coset_cells(rng, G, sb, triples)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _verified_components(G, hx, hz)
        ac = _syndrome_columns(ar, ca * G.q)
        bc = _syndrome_columns(br, cb * G.q)
        pa, pb = _pair_collisions(ac), _pair_collisions(bc)
        c4a, c4b = _four_cycles(ar), _four_cycles(br)
        quick = (_balanced_scores(pa, pb)
                 + _balanced_scores((c4a,), (c4b,))
                 + (rng.random(),))
        if len(pool) == 6 and quick >= pool[-1][0]:
            continue
        # Use the evaluator's actual rank, not a component-rank guess.
        if n - gf2_rank(hx) - gf2_rank(hz) != kb:
            continue
        pool.append((quick, A, B, hx, hz, n, ar, br,
                     ac, bc, pa, pb, c4a, c4b))
        pool.sort(key=lambda item: item[0])
        pool = pool[:6]

    if not pool:
        raise ValueError("no exact-rate candidate passed verified ranks")

    best = None
    for item in pool:
        (_, A, B, hx, hz, n, ar, br,
         ac, bc, pa, pb, c4a, c4b) = item
        # Full-code checks catch small words missed by component screens.
        px = _pair_collisions(_syndrome_columns(hx, n))
        pz = _pair_collisions(_syndrome_columns(hz, n))
        t6a, t6b = _triple_collisions(ac), _triple_collisions(bc)
        c6a, c6b = _six_cycles(ar), _six_cycles(br)
        key = (_balanced_scores(px, pz)
               + _balanced_scores(pa, pb)
               + _balanced_scores((t6a,), (t6b,))
               + _balanced_scores((c4a,), (c4b,))
               + _balanced_scores((c6a,), (c6b,)))
        if best is None or key < best[0]:
            best = key, A, B, hx, hz

    _, A, B, hx, hz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return {"cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """All cells were selected together; return a fresh full-cell list."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal uniform rolling alone gives robust memory in the q58 (1,3)x(1,3) bed: fitness 3.2187, measured weight-four logicals, and worse-basis failed-shot fractions about 0.48-0.576 contradict that hypothesis.
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
