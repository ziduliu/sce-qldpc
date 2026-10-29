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
CONCEPT = ("S3 direct product C7, order 42, shapes (1,3)x(2,4): degree-three component columns and long-order intra-cell ratios should improve worst-basis circuit reliability in cell 0.300. Test against that cell's 4.171 incumbent, not the saturated low-rate parent's score.")
GROUP = ("The untried S3 x C7 bed has centre order seven, unlike the parent's centreless C7 semidirect C6. Distinct C7 coordinates force intra-cell ratios to have order 7, 14 or 21. The run's successful nonabelian multi-row beds motivate testing this group, but do not establish a direct-product advantage.")
SHAPES = ("At q=42, (1,3)x(2,4) gives n=588 and guaranteed k>=168, rate 2/7 in cell 0.300. A has 126 variables/42 component checks; B has 168 variables/84 checks. Both have column degree three. The current parent had no failures to attribute; its sector-one witnesses spanned Acol/Bcol blocks. Exact lifted ranks enforce 0.275<=k/n<0.325.")
FILLING = ("Twenty architectures combine ten B support profiles with mixed or pure S3-coset cells. A row weight is nine; every B column totals three, and B rows total four through eight, hence HX weight is twelve and HZ weight at most eleven. Six-round CNOT count is 25704. Independently search labels using verified lifts, short component words, small syndrome boundaries and worst-side balance; reject disconnected codes and actual logicals of weight <=4. This targets the 0.300 incumbent's collective failures: slot 23 had 104/2250 Z failures, and slot 25 had 110/1000 X failures. Circuit pL remains the scored test.")
FAMILY_NAME = "r030_s3_times_c7_13x24_degree3_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 direct product C7 requires q=42")
    # (a,b,c) = r^a s^b z^c:
    # r^3=s^2=z^7=1, s*r*s=r^-1, and z is central.
    els = [(a, b, c) for c in range(7)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 7)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d)) % 3,
                (b + e) % 2, (c + f) % 7)

    def inv(g):
        a, b, c = red(g)
        return ((-a if b == 0 else a) % 3, b, (-c) % 7)

    def idx(g):
        a, b, c = red(g)
        return 6 * c + 3 * b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _design_supports(idx):
    """Ten incidence profiles, each tested with two coset architectures."""
    profiles = (
        (1, 1, 2, 2),
        (0, 1, 2, 3),
        (1, 1, 1, 3),
        (0, 1, 2, 2),
        (0, 1, 1, 3),
        (1, 1, 1, 2),
        (0, 0, 2, 3),
        (0, 1, 1, 2),
        (0, 0, 2, 2),
        (1, 1, 1, 1),
    )
    first = profiles[idx % len(profiles)]
    sa = [[3, 3, 3]]
    sb = [list(first), [3 - v for v in first]]
    assert all(sum(sb[i][j] for i in range(2)) == 3
               for j in range(4))
    return sa, sb


def _product_grid(rng, supports, mixed):
    """Distinct central coordinates exclude intra-cell order 2 or 3.

    Mixed cells cross the reflection quotient; pure cells lie in one
    coset of the normal C3 x C7 subgroup. These change cell structure,
    rather than applying a global automorphism to another filling.
    """
    grid = []
    for row in supports:
        out = []
        for weight in row:
            aa = rng.sample(range(3), weight)
            cc = rng.sample(range(7), weight)
            if mixed and weight >= 2:
                bb = [0, 1] + [rng.randrange(2)
                               for _ in range(weight - 2)]
                rng.shuffle(bb)
            else:
                b = rng.randrange(2)
                bb = [b] * weight
            out.append([(a, b, c) for a, b, c in zip(aa, bb, cc)])
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


def _product_components(hx, hz, q):
    """Extract both components from the verified full CSS lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _product_features(rows, ncols):
    """Column syndromes, exact kernel-word<=4 test and four-cycles."""
    cols = [0] * ncols
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    seen = {0}
    bad = False
    for syndrome in cols:
        if syndrome in seen:
            bad = True
        seen.add(syndrome)
    if not bad:
        for i, x in enumerate(cols):
            for y in cols[:i]:
                syndrome = x ^ y
                if syndrome in seen:
                    bad = True
                    break
                seen.add(syndrome)
            if bad:
                break
    return cols, int(bad), _four_cycles(rows)


def _product_kernel_bound(rng, cols, trials):
    """Random fundamental circuits: a component distance upper bound."""
    order = list(range(len(cols)))
    best = len(cols) + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in pivots:
                    s, w = pivots[p]
                    syndrome ^= s
                    word ^= w
                else:
                    pivots[p] = (syndrome, word)
                    break
            if not syndrome and word:
                best = min(best, word.bit_count())
    return best


def _product_boundaries(cols):
    """Triple collisions and small syndrome boundaries.

    When words<=4 are absent, different triples with equal syndromes
    are disjoint and witness weight-six component words. Small nonzero
    boundaries are a decoder-trapping heuristic, not a circuit score.
    """
    seen = {}
    collisions = boundary = 0
    count = len(cols)
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, count):
                syndrome = pair ^ cols[k]
                old = seen.get(syndrome, 0)
                collisions += old
                seen[syndrome] = old + 1
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
    return collisions, boundary


def NEW_CTX(rng, G, idx):
    """Search each incidence/coset architecture independently."""
    sa, sb = _design_supports(idx)
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    mixed = ((idx // 10) % 2 == 0)
    # Extra harness slots search fresh labels with a changed refinement
    # priority; they never replay, permute or automorph a previous code.
    mode = (idx // 20) % 3

    for batch in range(3):
        pool = []
        for trial in range(48):
            A = _product_grid(rng, sa, mixed)
            B = _product_grid(rng, sb, mixed)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k <= 0 or 40 * k < 11 * n or 40 * k >= 13 * n:
                continue
            ar, br = _product_components(hx, hz, G.q)
            ac, abad, a4 = _product_features(ar, A_SHAPE[1] * G.q)
            bc, bbad, b4 = _product_features(br, B_SHAPE[1] * G.q)
            base = (abad + bbad, max(a4, b4), a4 + b4)
            pool.append((base, trial, A, B, hx, hz, n, ac, bc, k))
        pool.sort(key=lambda item: (item[0], item[1]))

        preliminary = []
        for item in pool[:8]:
            base, trial, A, B, hx, hz, n, ac, bc, k = item
            da = _product_kernel_bound(rng, ac, 32)
            db = _product_kernel_bound(rng, bc, 32)
            key = (base[0], -min(da, db), -(da + db), base[1:], trial)
            preliminary.append((key, da, db, item))
        preliminary.sort(key=lambda entry: entry[0])

        finalists = []
        for _, da, db, item in preliminary[:3]:
            base, trial, A, B, hx, hz, n, ac, bc, k = item
            at = _product_boundaries(ac)
            bt = _product_boundaries(bc)
            six = (max(at[0], bt[0]), at[0] + bt[0])
            boundary = (max(at[1], bt[1]), at[1] + bt[1])
            refinement = ((boundary, six) if mode == 1
                          else (six, boundary))
            key = (base[0], refinement, -min(da, db),
                   -(da + db), base[1:], trial)
            finalists.append((key, item))
        finalists.sort(key=lambda entry: entry[0])

        for key, item in finalists:
            _, _, A, B, hx, hz, n, _, _, k = item
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            return {"A": A, "B": B, "design": idx, "k": k}
    raise ValueError("No connected target-rate filling passed logical screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; no subsequent redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Nominal rate 2/7 and legal weight-12 rolled C21 supports suffice to suppress collective failures: the measured (2,4)x(2,6) incumbent scored 3.87-4.42 per code, including 110 failed X shots/1000 and 5322 logical flips in slot 25.
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
