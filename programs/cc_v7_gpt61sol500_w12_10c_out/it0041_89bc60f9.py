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
CONCEPT = ("For C13 semidirect C6 with action 4, order 78, shapes (1,2)x(1,3), dense quotient-mixed supports selected against short logical witnesses should raise five-code distance readings above six and beat the rate-0.300 frontier fitness 19.3.")
GROUP = ("The nonabelian order-39 (2,3)x(2,3) bed reached 44.8, motivating another cyclic-action nonabelian bed without assuming performance transfers. Action 4 has order six modulo 13; the new order-78 group is genuinely different from the cyclic order-21 parent.")
SHAPES = ("Target cell 0.300: n=7*78=546, k>=2*78=156, guaranteed rate 2/7. A and B have 156 and 234 component variables with 78 checks each. Parent weight-six minima occupied sector-one Acol/Bcol blocks, frequently Acol1/2; screen both new components and retain exact rates in [0.275,0.325).")
FILLING = ("Five prescribed count profiles crossed with four quotient-composition rules define twenty optimization tasks. Cells contain 2..4 distinct terms; each component has at most one binomial, with relative label of order 13. Both row-weight sums are <=12. Verified lifts provide ranks, component syndromes and stabilizer membership. Finalists undergo deeper dependency searches and exhaustive nontrivial-logical screening through weight four. Witness readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r030_c13sc6_a4_12x13_dense_logical20"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 78

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 semidirect C6 requires order 78")
    m, r, s = 13, 6, 4
    els = [(a, b) for b in range(r) for a in range(m)]
    actions = [pow(s, b, m) for b in range(r)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + actions[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-actions[bi] * a) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Five count architectures; maximum (HX,HZ) weights are respectively
# (9,12), (10,12), (11,12), (10,12), (9,12).
_DENSE_PROFILES = (
    ((3,3), (3,3,3)),
    ((3,4), (3,3,2)),
    ((4,4), (3,3,2)),
    ((3,3), (4,3,2)),
    ((2,4), (3,3,2)),
)


def _dense_cell(rng, G, count, style, ordinal):
    """Quotient architecture plus independently optimized rotation labels."""
    if count == 2:
        # Exactly one binomial per component at most. Its relative
        # group element has order 13, rather than order two or three.
        x = G.pick(rng)
        y = G.mul(x, (rng.randrange(1, 13), 0))
        return [x, y]

    patterns = (
        {3: (0,1,2), 4: (0,1,2,3)},
        {3: (0,0,1), 4: (0,0,1,1)},
        {3: (0,1,3), 4: (0,1,3,5)},
        {3: (0,1,4), 4: (0,1,2,4)},
    )
    pattern = patterns[style][count]
    # An independent coset offset is a gauge freedom, not a new design;
    # the relative patterns and optimized labels distinguish the tasks.
    shift = rng.randrange(6)
    cell = []
    for b in pattern:
        b = (b + shift + ordinal) % 6
        choices = [g for g in G.elements if g[1] == b and g not in cell]
        cell.append(rng.choice(choices))
    return cell


def _dense_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _dense_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _dense_columns(rows, length):
    columns = [0] * length
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    return columns


def _dense_components(hx, hz, q):
    """Extract components only from the scaffold's verified matrices."""
    mask = (1 << q) - 1
    ar = [(row & mask) | (((row >> (3*q)) & mask) << q)
          for row in hx[:q]]
    br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
    return (_dense_columns(ar, 2*q),
            _dense_columns(br, 3*q))


def _dense_scatter(word, q, side):
    if side == "A":
        mask = (1 << q) - 1
        return (word & mask) | ((word >> q) << (3*q))
    return word


def _dense_short_logical(columns, basis, scatter):
    """Exhaustive nontrivial-logical test through weight four.

    All subsets of size at most two are grouped by syndrome.
    A collision gives a commuting word; stabilizer membership decides
    whether it is a logical. One representative per syndrome suffices:
    if all differences from it are stabilizers, every pair difference is.
    """
    seen = {0: 0}

    def insert(syndrome, word):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
            return None
        delta = old ^ word
        if delta and _dense_outside(scatter(delta), basis):
            return delta.bit_count()
        return None

    for i, syndrome in enumerate(columns):
        hit = insert(syndrome, 1 << i)
        if hit is not None:
            return hit
    for i, syndrome in enumerate(columns):
        bit = 1 << i
        for j in range(i):
            hit = insert(syndrome ^ columns[j], bit | (1 << j))
            if hit is not None:
                return hit
    return None


def _dense_cycles(columns):
    pairs = {}
    for column in columns:
        neighbors = []
        while column:
            bit = column & -column
            neighbors.append(bit.bit_length() - 1)
            column ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b, a)
                pairs[key] = pairs.get(key, 0) + 1
    return sum(v * (v - 1) // 2 for v in pairs.values())


def _dense_witness_search(rng, columns, basis, scatter, trials, best):
    """Information-set dependencies, pair sums and kernel descent.

    Only an actual nonzero word outside the stabilizer span lowers best.
    Failure to find a word is never treated as a positive distance result.
    """
    order = list(range(len(columns)))

    def consider(word):
        nonlocal best
        weight = word.bit_count()
        if word and weight < best and _dense_outside(scatter(word), basis):
            best = weight

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                consider(word)

        short = sorted(words, key=int.bit_count)[:20]
        for i, word in enumerate(short):
            for other in short[:i]:
                consider(word ^ other)

        for seed in short[:4]:
            word = seed
            for _ in range(2):
                weight = word.bit_count()
                chosen = word
                for other in words:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        chosen, weight = trial, tw
                if chosen == word:
                    break
                word = chosen
                consider(word)
        if best <= 4:
            break
    return best


def _dense_improve(rng, candidate, q, trials):
    for side, cols_key, basis_key, distance_key in (
            ("A", "ac", "zs", "da"), ("B", "bc", "xs", "db")):
        scatter = lambda word, side=side: _dense_scatter(word, q, side)
        candidate[distance_key] = _dense_witness_search(
            rng, candidate[cols_key], candidate[basis_key], scatter,
            trials, candidate[distance_key])


def _dense_score(candidate):
    d = min(candidate["da"], candidate["db"], candidate["full"])
    return (candidate["k"] * d * d / candidate["n"],
            min(candidate["da"], candidate["db"]),
            candidate["da"] + candidate["db"],
            -candidate["cycles"])


def NEW_CTX(rng, G, idx):
    """Optimize one prescribed count/quotient task, not an orbit copy."""
    task = idx % 20
    aw, bw = _DENSE_PROFILES[task % 5]
    style = task // 5
    sa, sb = [list(aw)], [list(bw)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    q = G.q
    pool = []
    reserve = None

    for attempt in range(48):
        A = [[_dense_cell(rng, G, t, style, j)
              for j, t in enumerate(aw)]]
        B = [[_dense_cell(rng, G, t, style, j + 2)
              for j, t in enumerate(bw)]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        if reserve is None or k < reserve["k"]:
            reserve = {"A": A, "B": B, "hx": hx, "hz": hz, "k": k}
        if not 275*n <= 1000*k < 325*n:
            continue

        ac, bc = _dense_components(hx, hz, q)
        candidate = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ac": ac, "bc": bc,
            "xs": _dense_basis(hx), "zs": _dense_basis(hz),
            "da": 2*q + 1, "db": 3*q + 1, "full": n + 1,
            "cycles": _dense_cycles(ac) + _dense_cycles(bc),
        }
        for side, cols_key, basis_key, distance_key in (
                ("A", "ac", "zs", "da"), ("B", "bc", "xs", "db")):
            scatter = lambda word, side=side: _dense_scatter(word, q, side)
            hit = _dense_short_logical(
                candidate[cols_key], candidate[basis_key], scatter)
            if hit is not None:
                candidate[distance_key] = hit

        _dense_improve(rng, candidate, q, 12)
        # Require witnessed nontrivial component logicals on both sides.
        if candidate["da"] <= 2*q and candidate["db"] <= 3*q:
            pool.append(candidate)
        if len(pool) >= 16:
            break

    if pool:
        finalists = sorted(pool, key=_dense_score, reverse=True)[:3]
        for candidate in finalists:
            # Include both LP sectors in the exact weight-four screen.
            for rows_key, basis_key in (("hx", "zs"), ("hz", "xs")):
                hit = _dense_short_logical(
                    _dense_columns(candidate[rows_key], candidate["n"]),
                    candidate[basis_key], lambda word: word)
                if hit is not None:
                    candidate["full"] = min(candidate["full"], hit)
            _dense_improve(rng, candidate, q, 32)
        for candidate in finalists:
            if candidate["full"] > 4:
                _dense_improve(rng, candidate, q, 64)
        winner = max(finalists, key=_dense_score)
    else:
        # Preserve a legal complete filling if the bounded heuristic search
        # fails. It receives no invented witness score.
        if reserve is None:
            raise ValueError("no positive-k dense filling")
        winner = reserve

    assert css_ok(winner["hx"], winner["hz"])
    return {"idx": idx, "cells": {"A": winner["A"], "B": winner["B"]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A legal weight-12 budget with rolled singleton-heavy supports reliably suppresses short logicals; many parent slots have d2s=2 and all five measured minima are weight six.
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
