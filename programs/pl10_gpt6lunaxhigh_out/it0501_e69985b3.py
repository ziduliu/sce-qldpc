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
CONCEPT = ("Falsifiable: for dihedral D18 of order 18 on shapes (2,5)x(2,5), 24-26-term fillings using order-9 rotations and sparse reflection labels, selected with lifted-cycle screening, will raise the five-code mean measured score above 4.5362.")
GROUP = ("Keep dihedral D18 of order 18 as required by this L1 round. RUN STATE reports a best of 4.8 for the q=18, (2,5)x(2,5) bed, while the current five measured slots score 4.26-4.90. This tests a changed filling within the same group and order.")
SHAPES = ("Keep A=(2,5), B=(2,5): q=18 gives n=(25+4)*18=522 and the rank-count bound k>=18*(5-2)*(5-2)=162, or rate 0.3103. Both component codes have length 90 and nominal dimension 54. Current failure maps show modest, varying enrichments rather than a persistent hotspot: examples include Acol2/Bcol1 in slot 0 and Acol4/Bcol3 in slot 1. Slot 11 also had 126 second-stage and 26 final-stage X-basis decodes. The new filling tests fewer terms and a different label pattern rather than targeting one block.")
FILLING = ("Use 24-26 terms, with one reflection singleton per protograph row and order-9 rotations elsewhere; doubled cells contain two distinct order-9 rotations. Exclude the identity and order-3 rotations. Screen 24 candidates per slot with exact lifted matrices, considering logicals through weight 4, 4-cycles, sampled 6-cycles, and degree balance. These term counts correspond to 18,144-19,656 CNOTs over six rounds. Fewer terms reduce gate exposure, while cycle screening tests whether the changed labels reduce difficult decoder cases; distance remains a selection aid, not the objective.")
FAMILY_NAME = "r030_D18_q18_25x25_rotgen_sparse24_26_s6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 18

def custom_ops(q):
    # Dihedral group of order 18:
    # r^9=s^2=1 and s*r*s=r^-1; elements are r^a*s^b.
    els = [(a, b) for a in range(9) for b in range(2)]

    def red(g):
        return (int(g[0]) % 9, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-1 if b else 1) * c) % 9, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a if b else -a) % 9, b)

    def idx(g):
        a, b = red(g)
        return 2 * a + b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search sparse, class-balanced fillings using the verified lifts."""
    total_extra = 4 + idx % 3
    profiles = {
        4: (((1, 1), (1, 1)),),
        5: (((1, 1), (1, 2)), ((1, 2), (1, 1))),
        6: (((1, 2), (1, 2)),),
    }[total_extra]
    extra_a, extra_b = profiles[(idx // 3) % len(profiles)]

    def support(cols, extras, trial, phase):
        grid = [[1] * cols for _ in range(2)]
        occupied = set()
        for r, extra in enumerate(extras):
            offset = (idx * (r + 1) + trial * (r + 2) + phase) % cols
            available = [(offset + s) % cols for s in range(cols)
                         if (offset + s) % cols not in occupied]
            for j in rng.sample(available, extra):
                grid[r][j] = 2
                occupied.add(j)
        return grid

    def fill(supports, trial, phase):
        # Use order-9 rotations (r^a with gcd(a,9)=1), never identity
        # or order-3 rotations. One distinct reflection singleton per row.
        rotations = [g for g in G.elements
                     if int(g[1]) % 2 == 0
                     and int(g[0]) % 9 in (1, 2, 4, 5, 7, 8)]
        reflections = [g for g in G.elements if int(g[1]) % 2 == 1]
        used_reflections = set()
        out = []
        for i, row in enumerate(supports):
            doubles = [j for j, count in enumerate(row) if count == 2]
            singles = [j for j, count in enumerate(row) if count == 1]
            reflection_j = singles[(idx + trial + phase + i) % len(singles)]
            rotation_pool = list(rotations)
            rng.shuffle(rotation_pool)
            filled = []
            for j, count in enumerate(row):
                if count == 2:
                    cell = [rotation_pool.pop(), rotation_pool.pop()]
                elif j == reflection_j:
                    choices = [g for g in reflections
                               if G._key(g) not in used_reflections]
                    g = rng.choice(choices)
                    used_reflections.add(G._key(g))
                    cell = [g]
                else:
                    cell = [rotation_pool.pop()]
                filled.append(cell)
            out.append(filled)
        return out

    def four_cycles(rows):
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                common = _popcount(row & other)
                total += common * (common - 1) // 2
        return total

    def sampled_six_cycles(rows, samples):
        total = 0
        for i, j, k in samples:
            x, y, z = rows[i], rows[j], rows[k]
            a = _popcount(x & y)
            b = _popcount(y & z)
            c = _popcount(z & x)
            common = _popcount(x & y & z)
            total += a * b * c - common * (a + b + c) + 2 * common
        m = len(rows)
        return total * (m * (m - 1) * (m - 2) // 6) / len(samples)

    def degree_square(hx, hz, n):
        degrees = [0] * n
        for row in hx + hz:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return sum(d * d for d in degrees)

    def logical_floor(checks, stabilizers, n):
        """Find a non-stabilizer logical through weight 4; cap conservatively."""
        pivots = {}
        for row in stabilizers:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in pivots:
                    x ^= pivots[p]
                else:
                    pivots[p] = x
                    break

        def in_span(x):
            while x:
                p = x.bit_length() - 1
                if p not in pivots:
                    return False
                x ^= pivots[p]
            return True

        sig = [0] * n
        for ri, row in enumerate(checks):
            bit = 1 << ri
            while row:
                b = row & -row
                sig[b.bit_length() - 1] |= bit
                row ^= b

        buckets = {}
        for i, s in enumerate(sig):
            buckets.setdefault(s, []).append(i)

        for i, s in enumerate(sig):
            if s == 0 and not in_span(1 << i):
                return 1
        for ids in buckets.values():
            for u in range(len(ids)):
                for v in range(u + 1, len(ids)):
                    if not in_span((1 << ids[u]) | (1 << ids[v])):
                        return 2
        for i in range(n):
            for j in range(i + 1, n):
                target = sig[i] ^ sig[j]
                for k in buckets.get(target, ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k)):
                        return 3

        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                pairs.setdefault(sig[i] ^ sig[j], []).append((i, j))
        work = 0
        for ps in pairs.values():
            for u, (i, j) in enumerate(ps):
                for k, ell in ps[u + 1:]:
                    work += 1
                    if work > 100000:
                        return 4
                    if i == k or i == ell or j == k or j == ell:
                        continue
                    if not in_span((1 << i) | (1 << j) | (1 << k) | (1 << ell)):
                        return 4
        return 5

    def screen_cycles(candidates):
        candidates.sort(key=lambda x: x[0])
        return candidates[:8]

    tri_hx = [rng.sample(range(A_SHAPE[0] * B_SHAPE[1] * G.q), 3)
              for _ in range(256)]
    tri_hz = [rng.sample(range(A_SHAPE[1] * B_SHAPE[0] * G.q), 3)
              for _ in range(256)]
    pool = []
    for trial in range(24):
        sa = support(A_SHAPE[1], extra_a, trial, 0)
        sb = support(B_SHAPE[1], extra_b, trial, 1)
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            continue
        A = fill(sa, trial, 0)
        B = fill(sb, trial, 1)
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue

        cycles = four_cycles(hx) + four_cycles(hz)
        six = (sampled_six_cycles(hx, tri_hx)
               + sampled_six_cycles(hz, tri_hz))
        balance = degree_square(hx, hz, n)
        max_check_weight = max((_popcount(row) for row in hx + hz), default=0)
        terms = sum(map(sum, sa)) + sum(map(sum, sb))
        cheap_key = (cycles, six, balance, max_check_weight, rng.random())
        pool.append((cheap_key, {"A": A, "B": B}, hx, hz, n,
                     cycles, six, balance, max_check_weight, terms))
        pool = screen_cycles(pool)

    best_key = None
    best_cells = None
    for _, cells, hx, hz, n, cycles, six, balance, max_check_weight, terms in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        dx = logical_floor(hz, hx, n)
        dz = logical_floor(hx, hz, n)
        in_band = int(0.275 <= k / n < 0.325)
        key = (in_band, min(dx, dz), dx + dz, -cycles, -six, -terms,
               -balance, -max_check_weight, -abs(k - 162), rng.random())
        if best_key is None or key > best_key:
            best_key, best_cells = key, cells

    if best_cells is None:
        raise ValueError("no legal positive-k filling candidate")
    return {"idx": idx, "cells": best_cells}


def CELL(rng, G, side, i, j, ctx):
    """Return the preselected full cell of distinct group elements."""
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
