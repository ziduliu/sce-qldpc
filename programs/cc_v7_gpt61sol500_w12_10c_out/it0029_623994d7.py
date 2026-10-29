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
CONCEPT = ("For cyclic Z13, encoded as metacyclic Z13:Z1 with s=1, q=13, shapes (2,5)x(3,8), degree-three-or-four component columns and screening for kernel words through weight six should improve the parent's sector-1 distance-five bottleneck; this is a falsifiable heuristic, not a distance proof.")
GROUP = ("Keep the parent's cyclic trivial-action metacyclic family. The new size factor is 46, so q=13 is the largest compatible cyclic order and n=598. The measured Z23 shape move improved 6.7 to 22.9, motivating a geometry change before a group-family change.")
SHAPES = ("A=(2,5), B=(3,8) give kb=195 and guaranteed rate 15/46=0.32609 in cell 0.350. Components have lengths 65 and 104 with 26 and 39 checks. Parent distance-five witnesses were exclusively in sector 1 across Acol/Bcol blocks; adding a third B row changes that bottleneck.")
FILLING = ("Controlled column-degree-three/four profiles replace rolled counts; balanced zero/double substitutions vary incidence structure. Both HX and HZ row sums are at most 12. Independently optimize labels against component four-cycles, then screen component kernel words through weight six using restrictions of verified lifted_matrices. Check assembled k and CSS; component screening does not certify quantum distance.")
FAMILY_NAME = "r035_m13r1s1_25x38_degree_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Same trivial-action metacyclic construction, resized for n=46*13.
    m, r, s = 13, 1, 1
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
B_SHAPE = (3, 8)               # (rb, cb)


def _component_columns(G, side, cells):
    """Extract component syndromes from the verified LP construction."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "A":
        zero = [[[] for _ in range(cb)] for _ in range(rb)]
        hx, _, _ = lifted_matrices(G, cells, zero)
        rows = [hx[(i * cb) * q + t]
                for i in range(ra) for t in range(q)]
        positions = [j * cb * q + t
                     for j in range(ca) for t in range(q)]
    else:
        zero = [[[] for _ in range(ca)] for _ in range(ra)]
        _, hz, _ = lifted_matrices(G, zero, cells)
        rows = hz[:rb * q]
        positions = range(cb * q)
    return [sum(((row >> p) & 1) << r
                for r, row in enumerate(rows)) for p in positions]


def _four_cycle_penalty(cols, q):
    pairs = {}
    collisions = 0
    for syndrome in cols:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                key = (neighbors[b], neighbors[a])
                collisions += pairs.get(key, 0)
                pairs[key] = pairs.get(key, 0) + 1
    return collisions / float(q)


def _kernel_score(cols):
    """Detect all possible kernel weights <=6 by subset collisions.

    A reading of 7 means no such word was detected by this exhaustive
    component test; it says nothing conclusive about mixed-sector logicals.
    """
    seen = {0: 0}
    best, hits = 7, 0

    def record(syndrome, mask):
        nonlocal best, hits
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = mask
        else:
            difference = previous ^ mask
            if difference:
                weight = difference.bit_count()
                if weight <= 6:
                    best = min(best, weight)
                    hits += 1
            if mask.bit_count() < previous.bit_count():
                seen[syndrome] = mask

    bits = [1 << i for i in range(len(cols))]
    for i, ci in enumerate(cols):
        record(ci, bits[i])
    for i, ci in enumerate(cols):
        for j in range(i):
            record(ci ^ cols[j], bits[i] | bits[j])
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            mask = bits[i] | bits[j]
            for t in range(j):
                record(syndrome ^ cols[t], mask | bits[t])
    return best, -hits


def _copy_cells(cells):
    return [[list(cell) for cell in row] for row in cells]


def _design_component(rng, G, side, counts, idx):
    """Multiple optimized proposals, selected by an explicit criterion."""
    import math
    locations = [(i, j) for i, row in enumerate(counts)
                 for j, count in enumerate(row) if count]
    selected = None
    selected_key = None
    for restart in range(12 + (idx % 3)):
        cells = [[rng.sample(G.elements, count) for count in row]
                 for row in counts]
        columns = _component_columns(G, side, cells)
        penalty = _four_cycle_penalty(columns, G.q)
        local_best = _copy_cells(cells)
        local_penalty = penalty
        for step in range(220):
            i, j = locations[rng.randrange(len(locations))]
            t = rng.randrange(len(cells[i][j]))
            old = cells[i][j][t]
            replacement = G.pick(rng)
            while replacement in cells[i][j]:
                replacement = G.pick(rng)
            cells[i][j][t] = replacement
            proposal_columns = _component_columns(G, side, cells)
            proposal_penalty = _four_cycle_penalty(proposal_columns, G.q)
            temperature = 1.4 * (1.0 - step / 220.0) + 0.08
            delta = proposal_penalty - penalty
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                penalty = proposal_penalty
                if penalty <= local_penalty:
                    local_penalty = penalty
                    local_best = _copy_cells(cells)
            else:
                cells[i][j][t] = old
        columns = _component_columns(G, side, local_best)
        distance, negative_hits = _kernel_score(columns)
        key = (distance, negative_hits, -local_penalty)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected = local_best
        if distance == 7:
            break
    return selected


def _support_design(rng, idx):
    """Four support classes with slot-specific incidence perturbations."""
    mode = idx % 4
    sa = [[1] * 5 for _ in range(2)]
    order = rng.sample(range(5), 5)
    if mode == 2:
        # One degree-four A column, four degree-three columns: rows 8,8.
        sa[0][order[0]] = sa[1][order[0]] = 2
        for t, j in enumerate(order[1:]):
            sa[t % 2][j] = 2
    else:
        # All A columns degree three: rows 8,7.
        for t, j in enumerate(order):
            sa[t % 2][j] = 2

    sb = [[1] * 8 for _ in range(3)]
    # Balanced transfers keep every B row at eight and every column
    # at three, but introduce genuinely different zero/double incidence.
    transfers = (idx // 4) % 3
    available = rng.sample(range(8), 8)
    for cycle in range(transfers):
        for t in range(3):
            j = available[3 * cycle + t]
            source = (t + cycle) % 3
            destination = (source + 1) % 3
            sb[source][j] -= 1
            sb[destination][j] += 1
    if mode in (0, 1):
        # At most one extra per row and column: rows <=9, columns <=4.
        extra = 3 if mode == 0 else 2
        for row, j in zip(rng.sample(range(3), extra),
                          rng.sample(range(8), extra)):
            sb[row][j] += 1

    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def NEW_CTX(rng, G, idx):
    """Return an optimized, fully specified design rather than a draw."""
    fallback = None
    fallback_key = None
    for attempt in range(3):
        sa, sb = _support_design(rng, idx)
        A = _design_component(rng, G, "A", sa, idx)
        B = _design_component(rng, G, "B", sb, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= 12
        ctx = {"idx": idx, "cells": {"A": A, "B": B}}
        if k > 0 and 0.325 <= k / n < 0.375:
            return ctx
        key = (k > 0, -abs(k / n - 0.350))
        if fallback_key is None or key > fallback_key:
            fallback_key, fallback = key, ctx
    return fallback


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; never draw new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the row-weight cap alone excludes weight-five logicals; all five measured parent codes had w=12 and d30s=5, with sector-1 witnesses.
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
