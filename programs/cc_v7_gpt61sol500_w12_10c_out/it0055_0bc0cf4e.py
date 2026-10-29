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
CONCEPT = ("For metacyclic Z4:Z2(s=3), order 8, shapes (2,7)x(2,8), move the connected monomial component to A and make B full-rank: verified k=246 with no commuting words below weight four should improve the parent's k=245, distance-four leaders.")
GROUP = ("Frozen L1 group and order. This bed measured fitness 8.167 with five distance-four readings, so test rank allocation and new fillings without claiming a group-driven distance advantage. Order 8 remains unchanged.")
SHAPES = ("Frozen (2,7)x(2,8): n=480, kb=240 and guaranteed rate 0.500; component matrices have 16 checks and lengths 56 and 64. Parent minima were entirely sector-1 Acol/Bcol operators spread broadly across both components. A connected monomial A has rank 15; full-rank B and exact CSS ranks are required to give k=246, still in cell 0.500.")
FILLING = ("Twenty cards cross ten B support architectures with two relative-element models. A has seven distinct monomial voltages. B bottom counts are odd, and asymmetric top parity gives augmentation rank two. Search verified lifts, require k=246 and exclude every commuting word of weights 1-3. Retain permutation-invariant-distinct syndrome profiles and minimize exact weight-four kernel multiplicity. Missing weight-four words are not assumed, and evaluator readings remain upper bounds.")
FAMILY_NAME = "r050_m4r2s3_27x28_rankshift_invariant4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same metacyclic rule; q=8 is the resized inversion-action lift.
    # y x y^-1 = x^3, and 3^2 = 1 mod 4.
    m, r, s = 4, 2, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)


from functools import lru_cache

# Replace the listed prefix columns of an eight-column (1,1) B.
# Bottom counts remain odd. Every profile has augmentation rank two,
# row totals <=10 and column totals <=5.
_RANKSHIFT_PLANS = (
    ((2, 1),),
    ((0, 3),),
    ((2, 3),),
    ((2, 1), (2, 1)),
    ((2, 1), (0, 3)),
    ((2, 1), (1, 3)),
    ((0, 3), (3, 1)),
    ((0, 1), (2, 1), (2, 1)),
    ((0, 1), (2, 3)),
    ((0, 1), (2, 1), (1, 3)),
)


def _rankshift_cell(rng, G, size, model):
    """Different relative-element orders, not translated orbit copies."""
    if size == 0:
        return []
    if size == 1:
        return [G.pick(rng)]
    base = G.pick(rng)
    if size == 2:
        # Rotation differences have order four; reflection differences
        # have order two. The two models change actual incidence cycles.
        difference = ((rng.choice((1, 3)), 0) if model == 0
                      else (rng.randrange(4), 1))
        relative = [G.e, difference]
    elif model == 0:
        relative = [(a, 0) for a in rng.sample(range(4), 3)]
    else:
        relative = [G.e, (rng.choice((1, 3)), 0),
                    (rng.randrange(4), 1)]
    cell = [G.mul(base, g) for g in relative]
    assert len({G._key(g) for g in cell}) == size
    return cell


def _rankshift_B(rng, G, counts, model):
    B = [[None] * 8 for _ in range(2)]
    relatives = list(G.elements)
    rng.shuffle(relatives)
    mono = [j for j, pair in enumerate(counts) if pair == (1, 1)]
    for j, relative in zip(mono, relatives):
        B[0][j], B[1][j] = [G.e], [G.inv(relative)]
    for j, pair in enumerate(counts):
        if pair == (1, 1):
            continue
        for i in range(2):
            B[i][j] = _rankshift_cell(rng, G, pair[i], model)
        # Remove the common right column factor instead of spending
        # candidate diversity on a monomial gauge.
        anchor = (B[0][j] or B[1][j])[0]
        inverse = G.inv(anchor)
        for i in range(2):
            B[i][j] = [G.mul(g, inverse) for g in B[i][j]]
    return B


def _rankshift_profile(rows, n):
    """Exact exclusion through weight three and weight-four census.

    After nonzero distinct singles and absence of single/pair collisions,
    equal pair syndromes have disjoint supports. Each weight-four kernel
    support contributes exactly three collisions. Kernel words need not
    be logical operators.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return None
            previous = pairs.get(syndrome, 0)
            collisions += previous
            pairs[syndrome] = previous + 1
    histogram = {}
    for syndrome, count in pairs.items():
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in columns))


def _rankshift_fingerprint(hx, hz, px, pz, k):
    """Invariant under qubit/check permutations and global X/Z exchange.

    Differing fingerprints cannot be orbit copies under those actions.
    Equal fingerprints are conservatively discarded, even when the
    underlying codes might actually differ.
    """
    types = tuple(sorted((
        (px[0], px[1], tuple(sorted(r.bit_count() for r in hx))),
        (pz[0], pz[1], tuple(sorted(r.bit_count() for r in hz))),
    )))
    joint = tuple(sorted(zip(px[2], pz[2])))
    swapped = tuple(sorted(zip(pz[2], px[2])))
    return k, types, min(joint, swapped)


@lru_cache(maxsize=1)
def _rankshift_bank(G):
    """Optimize twenty structural cards once; assign distinct outputs."""
    # Seven distinct nonidentity relative voltages connect A's two
    # lifted check blocks. No top-label gauge randomness is retained.
    voltage = [g for g in G.elements if G._key(g) != G._key(G.e)]
    A = [[[G.e] for _ in range(7)],
         [[g] for g in voltage]]
    sa = [[1] * 7 for _ in range(2)]
    keep = {}
    trials = max(24, N_SLOTS)

    for card in range(20):
        profile, model = divmod(card, 2)
        prefix = _RANKSHIFT_PLANS[profile]
        counts = prefix + ((1, 1),) * (8 - len(prefix))
        sb = [[pair[i] for pair in counts] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        local = random.Random(BASE_SEED * 10009 + 38000 + card)

        for attempt in range(trials):
            B = _rankshift_B(local, G, counts, model)
            hx, hz, n = lifted_matrices(G, A, B)

            # Read both component matrices from the verified construction.
            mask = (1 << G.q) - 1
            ar = []
            for i in range(2):
                for r in range(G.q):
                    row = hx[(i * 8) * G.q + r]
                    compact = 0
                    for j in range(7):
                        compact |= ((row >> (j * 8 * G.q)) & mask) << (
                            j * G.q)
                    ar.append(compact)
            bm = (1 << (8 * G.q)) - 1
            br = [row & bm for row in hz[:2 * G.q]]
            if gf2_rank(ar) != 15 or gf2_rank(br) != 16:
                continue
            # Cheap component rejection before the complete CSS census.
            if _rankshift_profile(br, 8 * G.q) is None:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != 246 or not (475 * n <= 1000 * k < 525 * n):
                continue
            px = _rankshift_profile(hx, n)
            if px is None:
                continue
            pz = _rankshift_profile(hz, n)
            if pz is None:
                continue

            fingerprint = _rankshift_fingerprint(hx, hz, px, pz, k)
            objective = (max(px[0], pz[0]), px[0] + pz[0],
                         card, attempt)
            previous = keep.get(fingerprint)
            if previous is None or objective < previous[0]:
                keep[fingerprint] = (objective, A, B)

    selected = sorted(keep.values(), key=lambda item: item[0])[:N_SLOTS]
    if len(selected) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct rankshift codes")
    bank = []
    for objective, A, B in selected:
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        assert n - gf2_rank(hx) - gf2_rank(hz) == 246
        bank.append((A, B, objective[:2]))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Use a different invariant-screened design, never an orbit fallback."""
    bank = _rankshift_bank(G)
    if idx >= len(bank):
        raise ValueError("no further invariant-distinct rankshift design")
    A, B, screen = bank[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "kernel4_profile": screen}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial voltages plus extra A terms on D8 (2,7)x(2,8) raised the measured distance above four; all five measured parent codes had d30s=4, with thousands of sector-1 weight-four witnesses.
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
