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
CONCEPT = ("For metacyclic Z37:Z3, s=10, order 111, shapes (1,2)x(3,5), twenty degree-controlled support designs screened on the verified lifts for component kernels of weight <=4 should improve the parent's five-code fitness; this is not a distance guarantee.")
GROUP = ("Retain the metacyclic construction rule. Parameters m=37,r=3,s=10 satisfy s^r=1 mod m and give q=111, allowing n=1443 on the new shapes. The larger normal subgroup is an untested lift-size hypothesis.")
SHAPES = ("A=(1,2), B=(3,5): n=13*111=1443 and k>=2*111=222, guaranteed rate 2/13. Component lengths/check counts are 222/111 and 555/333. Parent measured X witnesses of weight 6 in slots 2 and 9 occupied sector-2 Arow/Brow blocks; wider sides change these component kernels and sector balance.")
FILLING = ("Twenty explicit support profiles combine A degrees (3,3),(3,4),(4,4),(3,5) with five B topologies each. Both check-weight sums are <=12. Fresh mixed-coset labels are selected using actual-lift component weight-2/3/4 kernel counts and four-cycles; exact lifted ranks enforce the target rate band.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m37r3s10_12x35_kernel4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # Same metacyclic rule: 10^3 = 1 mod 37, with action of order 3.
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


# Four component-degree budgets, each with five deliberate B topologies.
# Lists are read-only templates; every candidate gets fresh cell lists.
_A_DEGREES = ((3, 3), (3, 4), (4, 4), (3, 5))
_B_33 = (
    ((1,2,2,2,2), (2,1,2,2,2), (2,2,1,2,2)),
    ((3,1,1,2,2), (1,2,2,2,2), (2,2,2,1,2)),
    ((3,1,1,2,2), (1,3,2,1,2), (1,1,3,2,2)),
    ((3,0,2,2,2), (1,3,2,1,2), (2,2,1,3,1)),
    ((3,3,1,1,1), (1,1,3,3,1), (2,2,2,1,2)),
)
_B_34 = (
    ((1,1,2,2,2), (2,2,1,1,2), (2,2,2,1,1)),
    ((3,1,1,1,2), (1,2,2,2,1), (1,2,2,1,2)),
    ((3,1,1,1,2), (1,3,1,2,1), (1,1,2,2,2)),
    ((3,1,1,1,2), (1,3,1,2,1), (1,1,3,1,2)),
    ((3,0,2,1,2), (1,3,1,2,1), (1,2,2,1,2)),
)
_B_COL4 = (
    ((2,2,1,1,1), (1,1,2,2,1), (1,1,1,1,2)),
    ((3,1,1,1,1), (0,2,2,2,1), (1,1,1,1,2)),
    ((3,1,1,1,1), (0,2,3,2,0), (1,1,0,1,3)),
    ((2,2,2,1,0), (2,0,1,2,2), (0,2,1,1,2)),
    ((3,2,0,1,1), (1,0,2,3,1), (0,2,2,0,2)),
)
_B_DESIGNS = (_B_33, _B_34, _B_COL4, _B_COL4)


def _mixed_cell(rng, G, count, phase):
    """Distinct labels; multi-term cells span at least two quotient cosets."""
    if count == 0:
        return []
    patterns = {
        1: (0,),
        2: (0, 1),
        3: (0, 0, 1),
        4: (0, 0, 1, 2),
        5: (0, 0, 1, 1, 2),
    }
    r = max(g[1] for g in G.elements) + 1
    shift = (phase + rng.randrange(r)) % r
    bs = [(b + shift) % r for b in patterns[count]]
    cell = []
    for b in sorted(set(bs)):
        pool = [g for g in G.elements if g[1] == b]
        cell.extend(rng.sample(pool, bs.count(b)))
    return cell


def _component_stats(rows):
    """Count small classical kernels and four-cycles, not CSS distance."""
    columns = {}
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            pos = bit.bit_length() - 1
            columns[pos] = columns.get(pos, 0) | (1 << check)
            row ^= bit
    syndromes = list(columns.values())
    singles = {}
    n2 = 0
    four_cycles = 0
    check_pairs = {}
    for syndrome in syndromes:
        n2 += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
        neighbors = []
        x = syndrome
        while x:
            bit = x & -x
            neighbors.append(bit.bit_length() - 1)
            x ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                pair = (neighbors[b], neighbors[a])
                previous = check_pairs.get(pair, 0)
                four_cycles += previous
                check_pairs[pair] = previous + 1

    # With distinct nonzero columns, pair collisions count weight-4
    # kernels three times; membership in singles counts weight-3
    # kernels three times. Duplicate columns are separately penalized.
    pairs = {}
    n3 = 0
    n4 = 0
    for a, x in enumerate(syndromes):
        for b in range(a):
            syndrome = x ^ syndromes[b]
            n3 += singles.get(syndrome, 0)
            previous = pairs.get(syndrome, 0)
            n4 += previous
            pairs[syndrome] = previous + 1
    bad = 10000 * n2 + 100 * (n3 // 3) + n4 // 3
    return bad, four_cycles, len(syndromes)


def NEW_CTX(rng, G, idx):
    """Optimize labels on a specified support design, using verified lifts."""
    degree = idx % 4
    topology = (idx // 4) % 5
    sa = [list(_A_DEGREES[degree])]
    sb = [list(row) for row in _B_DESIGNS[degree][topology]]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    q = G.q
    n1 = ca * cb * q
    n = n1 + ra * rb * q
    kb = q * (ca - ra) * (cb - rb)
    # Select beta=0 A columns from HX and j=0 B columns from HZ.
    # These are restrictions of the scaffold's verified matrices;
    # no separate left/right lift convention is implemented here.
    mask_a = sum(((1 << q) - 1) << (j * cb * q)
                 for j in range(ca))
    mask_b = (1 << n1) - 1

    best = None
    best_key = None
    for attempt in range(24):
        cells = {}
        for side, support in (("A", sa), ("B", sb)):
            offset = 1 if side == "B" else 0
            cells[side] = [
                [_mixed_cell(rng, G, count,
                             idx + 2*i + j + offset)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)
            ]
        hx, hz, built_n = lifted_matrices(G, cells["A"], cells["B"])
        k = built_n - gf2_rank(hx) - gf2_rank(hz)
        if built_n != n or not (8*k >= n and 40*k < 7*n):
            continue

        a_rows = [hx[(i*cb)*q + t] & mask_a
                  for i in range(ra) for t in range(q)]
        b_rows = [hz[alpha*q + t] & mask_b
                  for alpha in range(rb) for t in range(q)]
        a_bad, a_cycles, a_size = _component_stats(a_rows)
        b_bad, b_cycles, b_size = _component_stats(b_rows)
        assert a_size == ca*q and b_size == cb*q
        ac = a_cycles / a_size
        bc = b_cycles / b_size
        # Extra harness slots use distinct screening emphases as well
        # as fresh optimized labels, rather than automorphism images.
        if (idx // 20) % 2:
            cycle_key = (ac + bc, max(ac, bc))
        else:
            cycle_key = (max(ac, bc), ac + bc)
        key = (a_bad + b_bad, max(a_bad, b_bad),
               cycle_key, k - kb)
        if best_key is None or key < best_key:
            best_key = key
            best = cells

    if best is None:
        raise ValueError("No screened labeling remained in the target rate band")
    hx, hz, built_n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    return {"idx": idx, "cells": best}


def CELL(rng, G, side, i, j, ctx):
    """Return the preselected full cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent measurements: rolled positive-cell supports on Z19:Z3 (4,2)x(5,3) reliably avoid small sector-2 logicals; measured slots 2 and 9 both contain X logicals of weight 6.
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
