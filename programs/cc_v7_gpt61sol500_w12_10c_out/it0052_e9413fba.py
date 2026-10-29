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
CONCEPT = ("For affine C5 semidirect C4 with action 2, order 20, shapes (2,5)x(2,5), degree-three/four components selected against short kernel witnesses should yield five measured distance readings averaging at least 10 in cell 0.300.")
GROUP = ("A genuinely nonabelian group unlike the parent's C11 x C2 x C2: 2 has order four modulo five. This order-20 bed is absent from BEDS. The cyclic order-19 bed on these shapes scored 23.0 with measured distances 8..9; a benefit from noncommuting labels is experimental.")
SHAPES = ("Equal (2,5) sides give 100 component variables and 40 checks, n=580, and guaranteed k>=180, rate 9/29. Verified ranks retain rates below 0.325. Parent minima occupied sector 1; slot 34's weight-22 X witnesses used Acol0/Bcol1 and Acol1/Bcol1, motivating symmetric component-kernel screening.")
FILLING = ("Twenty canonical support pairs have component row sums seven or eight and column sums three or four, hence both CSS row-weight sums <=12. Optimize independent labels, exclude involution differences in binomials, screen component kernels exhaustively through weight six, refine randomized kernel witnesses, and prefer full-code exclusion of logicals through weight four. Screening is not a quantum-distance certificate.")
FAMILY_NAME = "r030_affine5x4_25x25_degree34_kernel6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Affine C5 semidirect C4 requires q=20")
    # x^5=y^4=1, y*x*y^-1=x^2.
    # Normal form x^a y^b; 2^4=1 modulo five.
    els = [(a, b) for b in range(4) for a in range(5)]
    action = (1, 2, 4, 3)

    def red(g):
        return (g[0] % 5, g[1] % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b] * c) % 5, (b + d) % 4)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        return ((-action[bi] * a) % 5, bi)

    def idx(g):
        a, b = red(g)
        return a + 5 * b

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _affine_support_designs():
    """Remove row/column permutation copies of component count grids."""
    from itertools import combinations_with_replacement
    options = [(a, b) for a in range(5) for b in range(5)
               if 3 <= a + b <= 4]
    representatives = set()
    for columns in combinations_with_replacement(options, 5):
        totals = tuple(sum(c[r] for c in columns) for r in range(2))
        if not all(7 <= w <= 8 for w in totals):
            continue
        canonical = min(
            tuple(columns),
            tuple(sorted((b, a) for a, b in columns)))
        # Require full rank of the binary augmentation matrix.
        p0 = sum((a & 1) << j for j, (a, b) in enumerate(canonical))
        p1 = sum((b & 1) << j for j, (a, b) in enumerate(canonical))
        if not p0 or not p1 or p0 == p1:
            continue
        representatives.add(canonical)

    profiles = [
        (tuple(a for a, b in columns), tuple(b for a, b in columns))
        for columns in sorted(representatives)]
    pairs = [(a, b) for i, a in enumerate(profiles)
             for b in profiles[i:]]

    def priority(pair):
        entries = [w for profile in pair for row in profile for w in row]
        return (entries.count(0), -sum(entries),
                entries.count(1), pair)

    pairs.sort(key=priority)
    if len(pairs) < 20:
        raise ValueError("fewer than twenty canonical support pairs")
    return tuple(pairs[:20])


def _column_syndromes(rows, n):
    """Columns of the verified matrix, represented as syndrome integers."""
    cols = [0] * n
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= marker
            row ^= bit
    return cols


def _four_cycle_count(cols):
    """Exact Tanner four-cycle count from pairs of incident checks."""
    overlaps = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for j, b in enumerate(checks):
            for a in checks[:j]:
                key = (a, b)
                old = overlaps.get(key, 0)
                cycles += old
                overlaps[key] = old + 1
    return cycles


def _has_light_logical(cols, stabilizer_rows):
    """Exhaustively detect nontrivial kernel words of weights 1..4.

    Collisions among supports of sizes 0, 1 and 2 cover every such
    kernel word. Reduction modulo the stabilizer span distinguishes
    logical operators from stabilizers.
    """
    basis = {}
    for row in stabilizer_rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    n = len(cols)
    bits = [1 << j for j in range(n)]
    seen = {0: 0}
    for j, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is not None:
            if nontrivial(bits[j] ^ old):
                return True
        else:
            seen[syndrome] = bits[j]

    for j in range(n):
        sj, bj = cols[j], bits[j]
        for i in range(j):
            syndrome = sj ^ cols[i]
            word = bj ^ bits[i]
            old = seen.get(syndrome)
            if old is not None:
                if nontrivial(word ^ old):
                    return True
            else:
                seen[syndrome] = word
    return False


def _affine_components(hx, hz, q):
    """Extract both component actions from the VERIFIED full lift."""
    mask = (1 << q) - 1
    ar = []
    for i in range(2):
        for t in range(q):
            source = hx[i * 5 * q + t]
            packed = 0
            for j in range(5):
                packed |= ((source >> (j * 5 * q)) & mask) << (j * q)
            ar.append(packed)
    bm = (1 << (5 * q)) - 1
    br = [row & bm for row in hz[:2 * q]]
    return (_column_syndromes(ar, 5 * q),
            _column_syndromes(br, 5 * q))


def _affine_kernel_floor(cols):
    """Exact smallest component kernel weight through six; else flag seven.

    This concerns classical kernels, not their quantum logical quotient.
    A seven flag says only that no kernel word of weight <=6 exists.
    """
    if 0 in cols:
        return 1
    singles = set(cols)
    if len(singles) != len(cols):
        return 2

    pairs = set()
    found_four = False
    for i, x in enumerate(cols):
        for y in cols[:i]:
            s = x ^ y
            if s in singles:
                return 3
            if s in pairs:
                found_four = True
            pairs.add(s)
    if found_four:
        return 4

    triples = set()
    found_six = False
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for h in range(j):
                s = xy ^ cols[h]
                if s in pairs:
                    return 5
                if s in triples:
                    found_six = True
                triples.add(s)
    return 6 if found_six else 7


def _affine_kernel_probe(cols, rng, trials):
    """Sample actual nonzero classical kernel witnesses, not certificates."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syndrome, word = cols[j], 1 << j
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
                best = min(best, word.bit_count())

        # Elementary basis operations preserve independence and kernel membership.
        for sweep in range(3):
            changed = False
            rng.shuffle(words)
            for i in range(len(words)):
                v = words[i]
                weight = v.bit_count()
                for j, u in enumerate(words):
                    if i == j:
                        continue
                    z = v ^ u
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                words[i] = v
                best = min(best, weight)
            if not changed:
                break
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Optimize one canonical count architecture without emitting orbit copies."""
    designs = _affine_support_designs()
    sa, sb = designs[idx % len(designs)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def draw_cell(weight):
        if weight == 0:
            return []
        while True:
            cell = rng.sample(G.elements, weight)
            if weight == 2:
                d = G.mul(G.inv(cell[0]), cell[1])
                # Involutory binomial differences give repeated lift columns.
                if G._key(G.mul(d, d)) == G._key(G.e):
                    continue
            return cell

    def fill(profile):
        return [[draw_cell(weight) for weight in row] for row in profile]

    pool = []
    quota = 48 + 8 * (idx // 20)
    for attempt in range(3 * quota):
        A, B = fill(sa), fill(sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Guaranteed rate is 9/29; enforce the upper edge of cell 0.300.
        if not (180 <= k and 40 * k < 13 * n):
            continue
        ac, bc = _affine_components(hx, hz, G.q)
        fa, fb = _four_cycle_count(ac), _four_cycle_count(bc)
        pool.append({
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ac": ac, "bc": bc, "cycles": (max(fa, fb), fa + fb),
            "attempt": attempt,
        })
        if len(pool) >= quota:
            break

    if not pool:
        raise ValueError("no labeling passed the exact 0.300 rate gate")
    pool.sort(key=lambda item: (item["cycles"], item["attempt"]))

    # Exhaustive short-word tests are inexpensive on 100-column components.
    shortlist = []
    for item in pool[:16]:
        item["floor_a"] = _affine_kernel_floor(item["ac"])
        item["floor_b"] = _affine_kernel_floor(item["bc"])
        shortlist.append(item)
    shortlist.sort(key=lambda item: (
        min(item["floor_a"], item["floor_b"]),
        item["floor_a"] + item["floor_b"],
        -item["cycles"][0], -item["cycles"][1]), reverse=True)
    shortlist = shortlist[:8]

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"] * d * d, d, item["da"] + item["db"],
                -item["cycles"][0], -item["cycles"][1])

    for item in shortlist:
        item["da"] = _affine_kernel_probe(item["ac"], rng, 12)
        item["db"] = _affine_kernel_probe(item["bc"], rng, 12)
    shortlist.sort(key=merit, reverse=True)
    finalists = shortlist[:4]

    # Longer independent probes reduce selection on a lucky short search.
    for item in finalists:
        item["da"] = min(item["da"],
                         _affine_kernel_probe(item["ac"], rng, 48))
        item["db"] = min(item["db"],
                         _affine_kernel_probe(item["bc"], rng, 48))
    finalists.sort(key=merit, reverse=True)

    chosen = finalists[0]
    for item in finalists:
        cx = _column_syndromes(item["hx"], item["n"])
        cz = _column_syndromes(item["hz"], item["n"])
        if _has_light_logical(cx, item["hz"]):
            continue
        if _has_light_logical(cz, item["hx"]):
            continue
        chosen = item
        break

    assert css_ok(chosen["hx"], chosen["hz"])
    return {"A": chosen["A"], "B": chosen["B"],
            "component_readings": (chosen["da"], chosen["db"])}


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the weight cap with rolled uniform fillings alone does not eliminate weight-4 logicals: all five measured parent codes had d30s=4, with minimum operators in sector 2.
# REFUTED: Extra k above the rank bound guarantees no distance penalty: current-parent slot 34 had k-kb=5 and d30s=22, versus slots 35/55/56 with k-kb=0 and d30s=26.
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
