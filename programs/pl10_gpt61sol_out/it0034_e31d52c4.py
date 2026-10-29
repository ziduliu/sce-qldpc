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
CONCEPT = ("C7 semidirect C6 with action 3, order 42, shapes (2,3)x(2,3): equal-coset and two-step-coset doubletons should produce fresh k=66..68 codes retaining zero-failure circuit performance at check weight ten. This tests a higher measurement ceiling without increasing gate count.")
GROUP = ("Keep the frozen order-42 Frobenius group: this measured bed scored 7.1776 and all five measured codes had zero failures in both bases. The evidence supports refining the filling, not claiming that another group is needed.")
SHAPES = ("Keep equal (2,3) components, each with 126 variables and 84 checks: n=546 and guaranteed k>=42, rate 1/13. Exact lifted ranks require k=66..68, still in the open-ended 0.100 cell. The parent has no failed-shot block attribution; its minimum-weight witnesses predominantly occupied sector-one Acol/Bcol blocks.")
FILLING = ("Every cell contains two distinct terms, giving both check weights ten and 30240 CNOTs over six rounds. Twenty explicit same-coset-mask pairs define architectures; remaining cells span coset gap two. All quotient cells vanish under C6->C2, deliberately increasing rank deficits. Verified lifts screen k, connectivity, CSS and logical weight<=4; component screens reject kernel words<=6 and penalize small-boundary triples. Parent scores saturated, while slot 37 still needed 23 Z-basis LSD resolutions: preserve reliability while raising k, rather than treating distance as fitness.")
FAMILY_NAME = "r010_frobenius7by6_23x23_evenquotient_boundary_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if q != 42:
        raise ValueError("C7 semidirect C6 requires q=42")
    # (a,b)=x^a y^b, y*x*y^-1=x^3; 3 has order 6 modulo 7.
    act = (1, 3, 2, 6, 4, 5)
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        return ((g[0] + act[g[1] % 6] * h[0]) % 7,
                (g[1] + h[1]) % 6)

    def inv(g):
        bi = (-g[1]) % 6
        return ((-act[bi] * g[0]) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _design_supports(idx):
    """Twenty different support designs, not permutation-orbit copies."""
    profiles = (
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 2, 2, 2),
        (1, 2, 3, 2, 2, 1),
        (1, 2, 3, 3, 2, 1),
        (1, 3, 2, 2, 1, 3),
        (3, 2, 2, 1, 2, 2),
        (3, 3, 1, 1, 1, 3),
        (3, 2, 2, 1, 3, 1),
        (3, 2, 1, 2, 2, 2),
        (3, 2, 2, 2, 2, 2),
        (3, 3, 1, 2, 1, 3),
        (3, 2, 2, 1, 2, 3),
    )
    pairs = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (0, 10), (0, 11), (1, 9), (2, 9), (3, 3),
        (3, 4), (3, 9), (4, 4), (4, 9), (9, 9),
    )
    a, b = pairs[idx % len(pairs)]
    return ([list(profiles[a][:3]), list(profiles[a][3:])],
            [list(profiles[b][:3]), list(profiles[b][3:])])


def _adjacent_coset_grid(rng, supports):
    """Every intra-cell ratio has order 6 or 7."""
    grid = []
    for row in supports:
        out = []
        for weight in row:
            t = rng.randrange(6)
            bs = [t] if weight == 1 else [t, (t + 1) % 6]
            while len(bs) < weight:
                bs.append((t + rng.randrange(2)) % 6)
            cell = []
            for b in bs:
                g = (rng.randrange(7), b)
                while g in cell:
                    g = (rng.randrange(7), b)
                cell.append(g)
            out.append(cell)
        grid.append(out)
    return grid


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
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


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def NEW_CTX(rng, G, idx):
    """Build a quotient-deficit architecture and screen fresh voltages.

    All projections below come from the verified construction helper.
    No independently implemented left/right lift is used.
    """
    supports = [[2, 2, 2], [2, 2, 2]]
    assert _grid_weights(supports, supports) <= 10
    assert _no_dead_lines(supports, supports)

    # Structurally different placements of quotient-zero cells.
    # No mask erases a whole quotient row or column.
    masks = (
        (),
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 2)),
    )
    pairs = [(a, b) for a in range(len(masks))
             for b in range(a, len(masks))]
    pairs.sort(key=lambda ab: (len(masks[ab[0]]) + len(masks[ab[1]]),
                               abs(ab[0] - ab[1]), ab))
    pairs = pairs[:20]
    design = idx % len(pairs)
    mode = (idx // len(pairs)) % 3
    pa, pb = pairs[design]

    def grid(mask, side):
        same = set(mask)
        result = []
        for i in range(2):
            row = []
            for j in range(3):
                t = rng.randrange(6)
                # Extra harness slots test constrained parity holonomy,
                # not automorphism images of an existing labeling.
                if mode == 1:
                    parity = ((i * j) if side == "A"
                              else i * (j + 1)) % 2
                    t = 2 * rng.randrange(3) + parity
                elif mode == 2:
                    parity = ((i * (j + 1)) if side == "A"
                              else i * j) % 2
                    t = 2 * rng.randrange(3) + parity
                a = rng.randrange(7)
                if (i, j) in same:
                    # Nonidentity ratio in C7: order seven.
                    c = (a + rng.randrange(1, 7)) % 7
                    cell = [(a, t), (c, t)]
                else:
                    # Coset gap two: ratio has order three.
                    cell = [(a, t), (rng.randrange(7), (t + 2) % 6)]
                row.append(cell)
            result.append(row)
        return result

    def components(hx, hz):
        q = G.q
        mask = (1 << q) - 1
        ar = []
        for i in range(2):
            for r in range(q):
                row = hx[(i * 3) * q + r]
                packed = 0
                for j in range(3):
                    packed |= ((row >> (j * 3 * q)) & mask) << (j * q)
                ar.append(packed)
        bmask = (1 << (3 * q)) - 1
        br = [hz[a * q + r] & bmask
              for a in range(2) for r in range(q)]
        return ar, br

    def columns(rows):
        cols = [0] * (3 * G.q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return cols

    def short_and_boundary(cols):
        """Exact kernel-weight<=6 test plus triple-boundary heuristic."""
        seen = {0}
        for value in cols:
            if value in seen:
                return True, 0
            seen.add(value)
        for i in range(len(cols)):
            for j in range(i):
                value = cols[i] ^ cols[j]
                if value in seen:
                    return True, 0
                seen.add(value)
        boundary = 0
        for i in range(len(cols)):
            for j in range(i):
                pair = cols[i] ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True, 0
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 6:
                        boundary += (6 - weight) ** 2
        return False, boundary

    def kernel_estimate(cols, trials):
        """Randomized fundamental circuits: an upper bound, not pL."""
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = (syndrome, word)
                        break
                    old_s, old_w = pivots[p]
                    syndrome ^= old_s
                    word ^= old_w
                if not syndrome and word:
                    best = min(best, word.bit_count())
        return best

    for batch in range(3):
        pool = []
        for trial in range(128):
            A, B = grid(masks[pa], "A"), grid(masks[pb], "B")
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = components(hx, hz)
            ca, cb = _four_cycles(ar), _four_cycles(br)
            cycle_key = (max(ca, cb), ca + cb)
            pool.append((cycle_key, trial, A, B, hx, hz, n, ar, br))
        pool.sort(key=lambda item: (item[0], item[1]))

        # Rank checks use the actual CSS matrices. The quotient argument
        # motivates the target but never substitutes for measured ranks.
        eligible = []
        for item in pool:
            cycle_key, trial, A, B, hx, hz, n, ar, br = item
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (66 <= k <= 68 and 8 * k < n):
                continue
            if not _connected_checks(hx, hz, n):
                continue
            eligible.append((item, k))
            if len(eligible) == 12:
                break

        finalists = []
        for item, k in eligible:
            cycle_key, trial, A, B, hx, hz, n, ar, br = item
            ac, bc = columns(ar), columns(br)
            bad_a, boundary_a = short_and_boundary(ac)
            if bad_a:
                continue
            bad_b, boundary_b = short_and_boundary(bc)
            if bad_b:
                continue
            da = kernel_estimate(ac, 48)
            db = kernel_estimate(bc, 48)
            boundary = (max(boundary_a, boundary_b),
                        boundary_a + boundary_b)
            distance = min(da, db)
            # A coarse protection floor precedes the ceiling benefit.
            # Above it, prefer higher k and then smaller trapping boundaries.
            if mode == 2:
                refinement = (-distance, boundary, cycle_key)
            else:
                refinement = (boundary, -distance, cycle_key)
            key = (int(distance < 18), -k, refinement,
                   abs(da - db), trial)
            finalists.append((key, item, k))

        finalists.sort(key=lambda entry: entry[0])
        for key, item, k in finalists:
            _, _, A, B, hx, hz, n, _, _ = item
            if _has_short_logical(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            return {"cells": {"A": A, "B": B}, "design": design,
                    "mode": mode, "k": k, "screen": key}

    raise ValueError("No connected k=66..68 filling passed kernel screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete quotient-deficit cell without late randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
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
