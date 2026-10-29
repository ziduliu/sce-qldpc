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
CONCEPT = ("For cyclic C13, q=13, shapes (2,6)x(2,6), balanced degree-three components with complete cross-row voltage coverage and screened small kernels should improve circuit memory over the parent's weight-four-logical bed; circuit measurement can refute this.")
GROUP = ("C13 is genuinely different from the parent's order-14 dihedral group and this bed is untested. Its twelve nonzero residues exactly accommodate the twelve cross-row differences of six doubled/singleton columns; n=520.")
SHAPES = ("Both lifted components have 78 variables and 26 checks, column weight 3 and row weight 9. Both CSS row-weight sums are 9+3=12. The guaranteed and screened rate is 208/520=0.400. Parent witnesses lie in the large Arow/Brow sector; here sector 1 has 468 qubits and sector 2 only 52, although the latter has higher gate degree.")
FILLING = ("Twenty indexed step-geometry constraints, canonical under cyclic automorphisms and row exchange, guide separate A/B searches rather than orbit copies. Cover every nonzero cross-row difference once and use three distinct doubled-cell steps per row, eliminating component Tanner 4-cycles. Reject component weight-four kernels and minimize weight-six kernel multiplicity. Verified full-matrix screening requires k=208, CSS commutation and no logical of weight <=4. This addresses the parent's s=3.30-3.59 and frequent broad failures; the larger CNOT budget remains a circuit-level risk.")
FAMILY_NAME = "r040_C13_26x26_paircover_screened"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 requires q=13")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def idx(g):
        return red(g)[0]

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _pc_cover(rng, geometry):
    """Exact-cover six prescribed steps on the twelve nonzero residues."""
    steps = list(geometry[0]) + list(geometry[1])
    options = {}
    for d in set(steps):
        options[d] = [
            (a, b, (1 << a) | (1 << b))
            for a in range(1, 13) for b in range(a + 1, 13)
            if min((b - a) % 13, (a - b) % 13) == d
        ]
    answer = [None] * 6

    def visit(todo, free):
        if not todo:
            return True
        choices = []
        for slot in todo:
            legal = [edge for edge in options[steps[slot]]
                     if edge[2] & free == edge[2]]
            if not legal:
                return False
            choices.append((len(legal), slot, legal))
        _, slot, legal = min(choices, key=lambda item: item[:2])
        rng.shuffle(legal)
        rest = tuple(s for s in todo if s != slot)
        for a, b, mask in legal:
            answer[slot] = (a, b)
            if visit(rest, free ^ mask):
                return True
        answer[slot] = None
        return False

    if visit(tuple(range(6)), (1 << 13) - 2):
        return answer
    return None


def _pc_book(rng):
    """Twenty feasible geometry orbits, not multiplier/row-swap copies."""
    from itertools import combinations
    triples = list(combinations(range(1, 7), 3))
    representatives = set()
    for a in triples:
        for b in triples:
            images = []
            for u in range(1, 13):
                image = []
                for triple in (a, b):
                    image.append(tuple(sorted(
                        min((u * d) % 13, 13 - (u * d) % 13)
                        for d in triple)))
                images.append(tuple(sorted(image)))
            representatives.add(min(images))
    book = []
    for geometry in sorted(representatives):
        if _pc_cover(rng, geometry) is not None:
            book.append(geometry)
            if len(book) == 20:
                return book
    raise ValueError("insufficient feasible voltage geometries")


def _pc_grid(pairs, G):
    """Each column has weights (2,1) or (1,2); each row sums to nine."""
    grid = [[None] * 6 for _ in range(2)]
    for j, (a, b) in enumerate(pairs):
        if j < 3:
            grid[0][j] = [(a,), (b,)]
            grid[1][j] = [G.e]
        else:
            grid[0][j] = [G.e]
            grid[1][j] = [((-a) % 13,), ((-b) % 13,)]
    return grid


def _pc_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _pc_four(columns):
    """Exact component weight-four-kernel test."""
    seen = {}
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in seen:
                return True
            seen[syndrome] = (i, j)
    return False


def _pc_six(columns, limit):
    """Count weight-six kernels after weight-four kernels are excluded.

    Each six-set has ten unordered partitions into two triples.
    Early termination cannot discard an improvement.
    """
    counts = {}
    collisions = 0
    bound = None if limit is None else 10 * limit
    for i in range(len(columns)):
        for j in range(i):
            pair = columns[i] ^ columns[j]
            for k in range(j):
                syndrome = pair ^ columns[k]
                previous = counts.get(syndrome, 0)
                collisions += previous
                if bound is not None and collisions > bound:
                    return None
                counts[syndrome] = previous + 1
    return collisions // 10


def _pc_cycles6(columns, m):
    """Tanner six-cycles from the check-intersection graph."""
    adjacency = [0] * m
    for column in columns:
        neighbors = []
        while column:
            bit = column & -column
            neighbors.append(bit.bit_length() - 1)
            column ^= bit
        for i in neighbors:
            for j in neighbors:
                if i != j:
                    adjacency[i] |= 1 << j
    triangles = 0
    for i in range(m):
        for j in range(i + 1, m):
            if (adjacency[i] >> j) & 1:
                triangles += (
                    adjacency[i] & adjacency[j] &
                    ~((1 << (j + 1)) - 1)
                ).bit_count()
    # Every weight-three column supplies one non-Tanner triangle.
    return triangles - len(columns)


def _pc_component_rows(G, grid, side):
    """Use verified lift conventions, including B's right action."""
    unit = [[[G.e]]]
    if side == "A":
        hx, hz, n = lifted_matrices(G, grid, unit)
        rows = hx
    else:
        hx, hz, n = lifted_matrices(G, unit, grid)
        rows = hz
    mask = (1 << (6 * G.q)) - 1
    return [row & mask for row in rows]


def _pc_select(rng, G, side, book, design):
    """Search an indexed geometry; broaden only if it has no eligible code."""
    best = None
    best_score = None
    seen = set()
    for trial in range(384):
        if trial and trial % 96 == 0 and best is not None:
            break
        geometry = book[(design + trial // 96) % len(book)]
        pairs = _pc_cover(rng, geometry)
        if pairs is None:
            continue
        grid = _pc_grid(pairs, G)
        key = tuple(tuple(sorted(cell)) for row in grid for cell in row)
        if key in seen:
            continue
        seen.add(key)
        rows = _pc_component_rows(G, grid, side)
        if gf2_rank(rows) != 2 * G.q:
            continue
        columns = _pc_columns(rows, 6 * G.q)
        if _pc_four(columns):
            continue
        six = _pc_six(columns, None if best_score is None else best_score[0])
        if six is None:
            continue
        score = (six, _pc_cycles6(columns, 2 * G.q))
        if best_score is None or score < best_score:
            best, best_score = grid, score
        # With this geometry, the six-cycle total is fixed by its steps.
        # Zero weight-six kernels is therefore the primary optimum.
        if six == 0:
            return grid
    if best is None:
        raise ValueError("no full-rank component without weight-four kernels")
    return best


def _pc_basis(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _pc_in_span(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return False
        word ^= pivots[p]
    return True


def _pc_no_small_logicals(checks, stabilizers, n):
    """Exact weight <=4 logical exclusion on the verified CSS matrices."""
    columns = _pc_columns(checks, n)
    pivots = _pc_basis(stabilizers)

    def logical(word):
        return bool(word) and not _pc_in_span(word, pivots)

    singles = {}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        if syndrome == 0 and logical(word):
            return False
        if syndrome in singles and logical(word ^ singles[syndrome]):
            return False
        singles.setdefault(syndrome, word)
    pairs = {}
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            word = (1 << i) | (1 << j)
            if syndrome == 0 and logical(word):
                return False
            if syndrome in singles and logical(word ^ singles[syndrome]):
                return False
            if syndrome in pairs and logical(word ^ pairs[syndrome]):
                return False
            pairs.setdefault(syndrome, word)
    return True


def NEW_CTX(rng, G, idx):
    """Build and screen the entire filling before CELL exposes any cells."""
    book = _pc_book(rng)
    design_a = idx % 20
    design_b = (idx + 7) % 20
    for attempt in range(6):
        A = _pc_select(rng, G, "A", book, design_a)
        B = _pc_select(rng, G, "B", book, design_b)
        hx, hz, n = lifted_matrices(G, A, B)
        if n - gf2_rank(hx) - gf2_rank(hz) != 208:
            continue
        if not css_ok(hx, hz):
            continue
        if not _pc_no_small_logicals(hx, hz, n):
            continue
        if not _pc_no_small_logicals(hz, hx, n):
            continue
        return {"idx": idx, "cells": {"A": A, "B": B}}
    raise ValueError("no filling passed verified full-CSS screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; never resample its labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap legality plus unconstrained rolled labels suffices for robust r040 memory; the parent instead measured five weight-four-logical codes with s=3.30-3.59 and frequent failures.
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
