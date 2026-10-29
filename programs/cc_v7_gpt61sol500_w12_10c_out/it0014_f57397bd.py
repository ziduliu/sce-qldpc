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
CONCEPT = ("Metacyclic Z7:Z2, inversion s=6, q=14, shapes (2,4)x(2,5): balanced column degrees and short-component-kernel screening should improve on the parent's persistent measured weight-four logicals.")
GROUP = ("Keep the exact parent group and order. The related inversion-action q=28 bed reached fitness 15.9; test whether denser components improve this family before changing the group.")
SHAPES = ("Target cell 0.250: n=24*14=336 and k>=14*(4-2)*(5-2)=84. Components have lengths 56 and 70 with 28 check rows each. Parent weight-four witnesses occupied sector-two Arow/Brow blocks; the new sector sizes are 280 and 56. Exact lifted ranks screen every retained filling into [0.225,0.275).")
FILLING = ("Five explicit support profiles crossed with four coset-label strategies define twenty design tasks. Each task optimizes candidate fillings using verified lifts, exact k, component kernel collision tests through weight seven, full commuting-vector tests through weight four, randomized kernel probes and a four-cycle tie-breaker. Prefer passing candidates; retain the best screened fallback if none passes. These screens do not certify quantum distance.")
FAMILY_NAME = "r025_m7r2s6_24x25_balanced_kernelopt"

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

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# (max HX weight, max HZ weight): (11,12), (11,12),
# (10,12), (11,12), (12,12). Every cell is nonzero.
_SUPPORT_PROFILES = (
    (
        ((1,2,2,2), (2,1,2,2)),
        ((1,2,1,2,2), (2,1,2,1,2)),
    ),
    (
        ((1,2,1,2), (2,1,2,2)),
        ((1,2,1,2,2), (2,1,2,1,2)),
    ),
    (
        ((1,2,1,2), (2,1,2,1)),
        ((2,2,2,1,2), (2,2,2,2,1)),
    ),
    (
        ((1,2,1,2), (2,1,2,1)),
        ((2,2,1,2,2), (3,2,2,1,1)),
    ),
    (
        ((1,2,1,2), (2,1,2,1)),
        ((3,1,2,1,2), (3,2,1,2,1)),
    ),
)


def _syndrome_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _short_kernel_witness(cols, cutoff):
    """Return a witnessed kernel weight <= cutoff, or None.
    Meet in the middle is exhaustive through cutoff 4 or 7.
    These are kernel tests, not tests for nontrivial CSS logicals."""
    n = len(cols)
    seen = {0: 0}

    def insert(syn, word):
        old = seen.get(syn)
        if old is not None:
            delta = old ^ word
            if delta:
                return delta.bit_count()
        else:
            seen[syn] = word
        return None

    for i, syn in enumerate(cols):
        w = insert(syn, 1 << i)
        if w is not None:
            return w
    for i in range(n):
        si, bi = cols[i], 1 << i
        for j in range(i + 1, n):
            w = insert(si ^ cols[j], bi | (1 << j))
            if w is not None:
                return w
    if cutoff <= 4:
        return None

    for i in range(n):
        si, bi = cols[i], 1 << i
        for j in range(i + 1, n):
            sij, bij = si ^ cols[j], bi | (1 << j)
            for k in range(j + 1, n):
                w = insert(sij ^ cols[k], bij | (1 << k))
                if w is not None:
                    return w

    # If every column has odd weight, every kernel word is even.
    # Having excluded weights <=6, there cannot be a weight-seven word.
    if all(s.bit_count() & 1 for s in cols):
        return None
    for i in range(n):
        si = cols[i]
        for j in range(i + 1, n):
            sij = si ^ cols[j]
            for k in range(j + 1, n):
                sijk = sij ^ cols[k]
                for l in range(k + 1, n):
                    old = seen.get(sijk ^ cols[l])
                    if old is not None:
                        word = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        delta = old ^ word
                        if delta:
                            return delta.bit_count()
    return None


def _kernel_probe(rng, cols, trials):
    """Find actual component kernel words by random elimination orders."""
    order = list(range(len(cols)))
    best = len(cols) + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syn, word = cols[j], 1 << j
            while syn:
                p = syn.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syn, word)
                    break
                ps, pw = pivots[p]
                syn ^= ps
                word ^= pw
            if not syn:
                words.append(word)
                best = min(best, word.bit_count())
        if words:
            # Reduce several seeds by the same valid kernel basis.
            seeds = sorted(words, key=int.bit_count)[:4]
            for word in seeds:
                weight = word.bit_count()
                for _ in range(3):
                    changed = False
                    for other in words:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if trial and tw < weight:
                            word, weight = trial, tw
                            changed = True
                    if not changed:
                        break
                best = min(best, weight)
    return best


def _component_rows(hx, hz, q):
    """Extract actual sector-one component matrices from verified lifts."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for h in range(q):
            row = hx[(i * cb) * q + h]
            packed = 0
            for j in range(ca):
                packed |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(packed)
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + h] & bmask
          for a in range(rb) for h in range(q)]
    return ar, br


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            shared = (row & other).bit_count()
            total += shared * (shared - 1) // 2
    return total


def _fill_design(rng, G, supports, mode, side):
    pools = (
        [g for g in G.elements if g[1] == 0],
        [g for g in G.elements if g[1] == 1],
    )
    grid = []
    for i, row in enumerate(supports):
        out = []
        for j, count in enumerate(row):
            phase = (i + j + (side == "B")) & 1
            if count == 1:
                cell = [G.pick(rng)]
            else:
                mixed = (
                    mode == 0
                    or (mode == 1 and phase == 0)
                    or (mode == 2 and (i + 2*j + (side == "B")) % 3 != 0)
                )
                if mode == 3:
                    cell = rng.sample(G.elements, count)
                elif mixed:
                    nr = (count + phase) // 2
                    cell = (rng.sample(pools[0], nr)
                            + rng.sample(pools[1], count - nr))
                else:
                    cell = rng.sample(pools[phase], count)
            out.append(cell)
        grid.append(out)
    return grid


def NEW_CTX(rng, G, idx):
    """Optimize one of twenty support/coset design tasks."""
    profile = idx % 5
    mode = (idx // 5) % 4
    sa, sb = _SUPPORT_PROFILES[profile]
    assert _grid_weights(sa, sb) <= W_MAX
    best = None
    best_key = None

    for attempt in range(48):
        # Optimize at least 24 fillings; extend difficult tasks to 48.
        if attempt >= 24 and best_key is not None and best_key[0]:
            break
        A = _fill_design(rng, G, sa, mode, "A")
        B = _fill_design(rng, G, sb, mode, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.225 <= k / n < 0.275):
            continue

        ar, br = _component_rows(hx, hz, G.q)
        ac = _syndrome_columns(ar, A_SHAPE[1] * G.q)
        bc = _syndrome_columns(br, B_SHAPE[1] * G.q)
        aw = _short_kernel_witness(ac, 7)
        bw = _short_kernel_witness(bc, 7)
        components_pass = aw is None and bw is None
        if not components_pass and best_key is not None and best_key[0]:
            continue

        trials = 96 if components_pass else 16
        da = _kernel_probe(rng, ac, trials)
        db = _kernel_probe(rng, bc, trials)
        if aw is not None:
            da = min(da, aw)
        if bw is not None:
            db = min(db, bw)

        quantum_pass = False
        if components_pass:
            quantum_pass = (
                _short_kernel_witness(_syndrome_columns(hx, n), 4) is None
                and _short_kernel_witness(_syndrome_columns(hz, n), 4) is None
            )
        cycles = _four_cycles(ar) + _four_cycles(br)
        key = (int(components_pass), int(quantum_pass),
               min(da, db), da + db, -cycles)
        if best_key is None or key > best_key:
            best_key = key
            best = {"idx": idx, "cells": {"A": A, "B": B},
                    "hx": hx, "hz": hz}

    if best is None:
        raise ValueError("no filling passed the exact target-rate screen")
    assert css_ok(best["hx"], best["hz"])
    return {"idx": idx, "cells": best["cells"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete, already optimized cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
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
