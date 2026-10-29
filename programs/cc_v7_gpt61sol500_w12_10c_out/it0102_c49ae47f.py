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
CONCEPT = ("Inversion-metacyclic Z9:Z2, order 18, shapes (1,4)x(2,7): twenty trinomial/graph recipes screened for distinct complete-matrix incidence invariants should supply five measured distance-four codes instead of the parent's two.")
GROUP = ("Frozen L1 group and order. This exact bed measured k=273 and d30s=4, but only two distinct codes survived sixty slots. Test genuine filling diversity rather than claim a group-driven distance improvement.")
SHAPES = ("Frozen (1,4)x(2,7): n=540, kb=270 and guaranteed rate 0.500. Component lengths/check counts are 72/18 and 126/36. Parent minima occupied sector-1 Acol/Bcol blocks broadly, with Z witnesses especially involving Acol0 and Acol3. Verified ranks keep every emitted code in [0.475,0.525).")
FILLING = ("Twenty recipes cross zero through three pure-rotation A trinomials with five B graph architectures: seven general monomial columns, one rotation-binomial/zero replacement, two such replacements with equal or unequal rotation steps, and seven rotation-only monomial columns. A retains counts [3,3,3,1]; every B column has two terms, so HX=12 and HZ<=11. Optimize verified component quartet counts, exclude complete commuting words through weight three, and discard equal permutation-invariant fingerprints. Rotation-only B also tests k=276. Quartet counts are kernel statistics, not logical-distance estimates.")
FAMILY_NAME = "r050_m9r2s8_14x27_oddanchor_incidencecards_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 18

def custom_ops(q):
    # Same inversion-metacyclic rule: y x y^-1 = x^-1.
    # Factor 30 requires q<20; take the largest even order, 18.
    m, r, s = 9, 2, 8
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


from functools import lru_cache


def _inc_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _inc_profile(columns, detailed=False):
    """Exactly exclude kernel words through three; census weight four.

    Once singleton syndromes are distinct and nonzero, equal pair
    syndromes have disjoint supports. Every quartet contributes three
    pair collisions. Kernel words need not be logical operators.
    """
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return None
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
    if not detailed:
        return collisions // 3
    histogram = {}
    for syndrome, count in pairs.items():
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in columns))


def _inc_components(hx, hz, q):
    """Extract components using only the scaffold's verified actions."""
    mask = (1 << q) - 1
    ar = [
        sum(((row >> (j * 7 * q)) & mask) << (j * q)
            for j in range(4))
        for row in hx[:q]
    ]
    br = [row & ((1 << (7 * q)) - 1) for row in hz[:2 * q]]
    return ar, br


def _inc_fingerprint(hx, hz, px, pz, k):
    """Invariant under qubit/check permutations and global X/Z exchange.

    Different fingerprints exclude those Tanner isomorphisms.
    Equal fingerprints are conservatively discarded.
    """
    types = tuple(sorted((
        (px[0], px[1], tuple(sorted(r.bit_count() for r in hx))),
        (pz[0], pz[1], tuple(sorted(r.bit_count() for r in hz))),
    )))
    joint = tuple(sorted(zip(px[2], pz[2])))
    swapped = tuple(sorted(zip(pz[2], px[2])))
    return k, types, min(joint, swapped)


def _inc_draw(rng, G, pure_count, graph_model):
    """Change internal triple geometry and actual B support architecture."""
    A = [[]]
    for j in range(3):
        if j < pure_count:
            # Normalization removes a common cell factor; the two
            # remaining differences are independently selected.
            u, v = rng.sample(range(1, 9), 2)
            cell = [G.e, (u, 0), (v, 0)]
        else:
            # Include both order-nine and order-three rotation pairs.
            cell = [G.e, (rng.randrange(1, 9), 0),
                    (rng.randrange(9), 1)]
        A[0].append(cell)
    A[0].append([G.e])

    B = [[None] * 7 for _ in range(2)]
    first_mono = 0
    units = (1, 2, 4, 5, 7, 8)
    if graph_model in (1, 2, 3):
        d = rng.choice(units)
        B[0][0], B[1][0] = [G.e, (d, 0)], []
        first_mono = 1
        if graph_model in (2, 3):
            if graph_model == 2:
                e = d
            else:
                # Distinct unsigned steps give different internal cycles.
                e = rng.choice([x for x in units
                                if min(x, 9-x) != min(d, 9-d)])
            B[0][1], B[1][1] = [], [G.e, (e, 0)]
            first_mono = 2

    pool = ([g for g in G.elements if g[1] == 0]
            if graph_model == 4 else list(G.elements))
    relatives = rng.sample(pool, 7 - first_mono)
    for j, relative in zip(range(first_mono, 7), relatives):
        B[0][j], B[1][j] = [G.e], [G.inv(relative)]
    return A, B


@lru_cache(maxsize=1)
def _inc_bank(G):
    """Optimize twenty architectures; never fill a slot with an orbit copy."""
    assert G.q == 18
    fingerprints = set()
    bank = []
    for slot in range(N_SLOTS):
        card = slot % 20
        pure_count, graph_model = divmod(card, 5)
        local = random.Random(BASE_SEED * 10009 + 9182700 + slot)
        chosen = None

        for batch in range(2):
            pool = []
            exact_seen = set()
            quota = 64 + 16 * batch + 8 * (slot // 20)
            for attempt in range(quota):
                A, B = _inc_draw(local, G, pure_count, graph_model)
                key = tuple(tuple(sorted(cell))
                            for grid in (A, B)
                            for row in grid for cell in row)
                if key in exact_seen:
                    continue
                exact_seen.add(key)
                sa = [[len(cell) for cell in row] for row in A]
                sb = [[len(cell) for cell in row] for row in B]
                assert _grid_weights(sa, sb) <= W_MAX
                assert _no_dead_lines(sa, sb)
                assert all(len(cell) == len(set(cell))
                           for grid in (A, B)
                           for row in grid for cell in row)

                hx, hz, n = lifted_matrices(G, A, B)
                ar, br = _inc_components(hx, hz, G.q)
                # A's monomial anchor is surjective. B is a degree-two
                # incidence matrix; screen its actual extracted rank.
                rank_b = gf2_rank(br)
                if rank_b < 34:
                    continue
                pa = _inc_profile(_inc_columns(ar, 4 * G.q))
                if pa is None:
                    continue
                pb = _inc_profile(_inc_columns(br, 7 * G.q))
                if pb is None:
                    continue
                objective = (max(pa, pb), pa + pb, -rank_b, attempt)
                pool.append((objective, A, B, hx, hz, n))
                pool.sort(key=lambda item: item[0])
                del pool[8:]

            winner = None
            for objective, A, B, hx, hz, n in pool:
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (475 * n <= 1000 * k < 525 * n):
                    continue
                px = _inc_profile(_inc_columns(hx, n), detailed=True)
                if px is None:
                    continue
                pz = _inc_profile(_inc_columns(hz, n), detailed=True)
                if pz is None:
                    continue
                fingerprint = _inc_fingerprint(hx, hz, px, pz, k)
                if fingerprint in fingerprints:
                    continue
                # No reward for an unverified distance above four.
                # Prefer more k, then fewer complete kernel quartets.
                final_key = (-k, max(px[0], pz[0]),
                             px[0] + pz[0], objective)
                if winner is None or final_key < winner[0]:
                    winner = (final_key, A, B, hx, hz, k, fingerprint)

            if winner is not None:
                score, A, B, hx, hz, k, fingerprint = winner
                assert css_ok(hx, hz)
                fingerprints.add(fingerprint)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row) for row in A),
                    tuple(tuple(tuple(cell) for cell in row) for row in B),
                    k, score,
                )
                break
        # A failed recipe is voided, never replaced by a repeated code.
        bank.append(chosen)

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five incidence-distinct screened codes")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign this slot's independently optimized structural recipe."""
    item = _inc_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh incidence-distinct filling for this recipe")
    A, B, k, score = item
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection_score": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further draws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform labels with rolled counts at q=58, (1,3)x(1,3), reliably avoid weight<=3 logicals: many parent slots had d2s=2 or 3; its measured top-five all had d30s=4.
# REFUTED: Voltage-autocorrelation classes and reflection-offset optimization ensure five distinct D18 (1,4)x(2,7) codes: sixty parent slots collapsed to two distinct codes, both with d30s=4.
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
