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
CONCEPT = ("Metacyclic C9 semidirect C4 with action 8, order 36, shapes (1,3)x(1,5): triangle-free doubleton components and reduced stabilizer-aware fault-pair ambiguity should improve worse-basis circuit reliability over the parent's small-lift wide components in cell 0.500.")
GROUP = ("Retain the metacyclic construction rule and resize its parameters to m=9,r=4,s=8. The new size factor 16 permits q=36 and n=576; 8^4=1 modulo 9 validates the action. Odd-coset directions have order four, and selected even-coset directions have order six or eighteen. This supplies noncommuting noninvolutory doubletons; it is an algebraic opportunity, not a measured group advantage.")
SHAPES = ("New (1,3)x(1,5) components have 36 checks each and 108/180 variables. n=576 and kb=288 give guaranteed rate 0.500; verified full ranks enforce 0.475<=k/n<0.525. Parent measured failures affected all 240 logicals with diffuse 1.1-1.2x Acol/Bcol enrichment and no stable Brow hotspot, motivating a global structural change. Uniform doubletons give HX weight eight and HZ weight twelve.")
FILLING = ("Twenty cards cross five A direction-order profiles with four B profiles; further harness batches introduce order-six even-coset directions. Every cell is a doubleton, with 16416 six-round CNOTs at n=576. Search normalized relative directions using verified lifts, reject component weight-two/three kernel words, and shortlist by weight-four multiplicity and cycles. Final full-CSS quotient screening minimizes worse-basis logical pair ambiguities and shared-check hook exposure. This targets the parent's 117-147 worse-basis failures per 250 shots and ubiquitous weight-four witnesses; the hook cost is not an ASC-order certificate or measured pL.")
FAMILY_NAME = "r050_m9r4s8_13x15_doubleton_quotientbudget"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires q=36")
    # Same split metacyclic rule: x^9=y^4=1, y*x*y^-1=x^8.
    m, r, s = 9, 4, 8
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


# A profile is (number of order-four directions,
#               number of order-six even-coset directions).
# Remaining directions have order eighteen.
_A_DIRECTION_PROFILES = (
    (1, 0), (1, 1), (2, 0), (2, 1), (3, 0),
)
_B_DIRECTION_BATCHES = (
    ((2, 0), (3, 0), (4, 0), (5, 0)),
    ((1, 1), (2, 1), (3, 1), (4, 1)),
    ((2, 0), (2, 1), (3, 0), (3, 1)),
)


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
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


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def NEW_CTX(rng, G, idx):
    """Indexed direction-order contracts, then verified CSS screening."""
    design, batch = idx % 20, idx // 20
    ap = _A_DIRECTION_PROFILES[design // 4]
    bp = _B_DIRECTION_BATCHES[batch % len(_B_DIRECTION_BATCHES)][design % 4]
    sa, sb = [[2] * A_SHAPE[1]], [[2] * B_SHAPE[1]]
    assert G.q == 36
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    def draw(width, profile):
        odd, short = profile
        long = width - odd - short
        # Inversion identifies +/-a for even-coset directions.
        # (a,1)^2=(0,2), hence these directions have order four.
        directions = [(a, 1) for a in rng.sample(range(9), odd)]
        directions += [(3, 2)] * short
        directions += [(a, 2) for a in rng.sample((1, 2, 4), long)]
        assert len(directions) == width
        assert len({G._key(g) for g in directions}) == width
        rng.shuffle(directions)
        # Origins of single-row cells are column gauges. Do not spend
        # the search on random translations of an unchanged direction.
        return [[[G.e, g] for g in directions]]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(cell[1]) for cell in grid[0]))
                     for grid in (A, B))

    def connected(rows, width):
        """Connectivity of the extracted component Tanner graph."""
        neighbors = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                neighbors[bit.bit_length() - 1].append(r)
                row ^= bit
        adjacency = [set() for _ in rows]
        for checks in neighbors:
            for r in checks:
                adjacency[r].update(checks)
        seen, todo = {0}, [0]
        while todo:
            r = todo.pop()
            for t in adjacency[r]:
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return len(seen) == len(rows)

    def assess(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 576
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 19 * n <= 40 * k < 21 * n):
            return None
        ar, br = _component_rows(hx, hz, G.q)
        ast = _kernel_statistics(ar, A_SHAPE[1] * G.q)
        bst = _kernel_statistics(br, B_SHAPE[1] * G.q)
        # Distinct nonzero columns and no triples establish component
        # kernel exclusion through weight three, not quantum distance.
        if any(ast[:2]) or any(bst[:2]):
            return None
        if not connected(ar, A_SHAPE[1] * G.q):
            return None
        if not connected(br, B_SHAPE[1] * G.q):
            return None
        # Account for the unequal numbers of repeated component blocks.
        az, bx = B_SHAPE[1] * ast[2], A_SHAPE[1] * bst[2]
        fa, fb = _four_cycles(ar), _four_cycles(br)
        key = (max(az, bx), az + bx, fa + fb,
               max(ast[3], bst[3]), ast[3] + bst[3])
        return key, hx, hz, k

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def quotient_columns(rows, width):
        """Canonical linear quotient by the VERIFIED stabilizer span."""
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        # Each pivot row's tail uses smaller columns. Process ascending.
        residual = [0] * width
        for c in range(width):
            pivot = pivots.get(c)
            if pivot is None:
                residual[c] = 1 << c
                continue
            tail, value = pivot ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def fault_pair_profile(checks, stabilizers, width):
        """Stabilizer-aware <=4 logical ambiguity and hook surrogates.

        A shared opposite check indicates a possible spread pair, not
        a guarantee that the final ASC ordering realizes that hook.
        """
        syndrome = columns(checks, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        spread = columns(stabilizers, width)
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        degrees = [x.bit_count() + z.bit_count()
                   for x, z in zip(syndrome, spread)]
        buckets = {}
        double_hook = hook_exposure = weighted_pairs = logical_pairs = 0

        for i in range(width):
            si, ri, hi = syndrome[i], residual[i], spread[i]
            for j in range(i):
                s = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(s)
                if third is not None and third != label:
                    return None
                hook = (hi & spread[j]).bit_count()
                exposure = degrees[i] * degrees[j]
                old = buckets.get(s)
                if old is None:
                    buckets[s] = [1, hook, exposure,
                                  {label: [1, hook, exposure]}]
                    continue
                same = old[3].get(label)
                nc, nh, ne = same if same is not None else (0, 0, 0)
                different = old[0] - nc
                different_hooks = old[1] - nh
                different_exposure = old[2] - ne
                logical_pairs += different
                double_hook += hook * different_hooks
                hook_exposure += hook * different + different_hooks
                weighted_pairs += exposure * different + different_exposure
                old[0] += 1
                old[1] += hook
                old[2] += exposure
                if same is None:
                    old[3][label] = [1, hook, exposure]
                else:
                    same[0] += 1
                    same[1] += hook
                    same[2] += exposure
        # With distinct nonzero syndrome columns, colliding distinct
        # pairs are disjoint. Each quartet has three pair partitions.
        assert logical_pairs % 3 == 0
        return double_hook, hook_exposure, weighted_pairs, logical_pairs // 3

    pool, seen = [], set()
    # Rejection handles triangle-producing directions before submission;
    # it never repairs an illegal evaluated slot.
    for trial in range(640):
        A = draw(A_SHAPE[1], ap)
        B = draw(B_SHAPE[1], bp)
        sig = signature(A, B)
        if sig in seen:
            continue
        seen.add(sig)
        result = assess(A, B)
        if result is None:
            continue
        key, hx, hz, k = result
        pool.append((key, trial, A, B, hx, hz, k))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[6:]
        if trial >= 191 and len(pool) == 6:
            break

    best = None
    for coarse, trial, A, B, hx, hz, k in pool:
        px = fault_pair_profile(hz, hx, 576)
        pz = fault_pair_profile(hx, hz, 576)
        if px is None or pz is None:
            continue
        # First reduce possible two-hook ambiguities, then one-hook
        # exposure and gate-weighted two-data-fault ambiguities.
        key = tuple(v for t in range(4)
                    for v in (max(px[t], pz[t]), px[t] + pz[t]))
        key += coarse
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, px, pz)

    if best is None:
        raise ValueError("No connected target-rate contract passed <=3 screening")
    key, A, B, hx, hz, k, px, pz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx) == 8
    assert max(row.bit_count() for row in hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 16416
    return {"A": A, "B": B, "design": design, "batch": batch,
            "direction_contract": (ap, bp), "k": k,
            "fault_pair_profile": (px, pz), "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the already screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Dispersed doubletons and component short-word/cycle optimization sufficiently suppress collective D8 high-rate failures: fitness was 3.2454, measured worse-basis failures were 117-147/250, and every measured code retained abundant weight-four logicals.
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
