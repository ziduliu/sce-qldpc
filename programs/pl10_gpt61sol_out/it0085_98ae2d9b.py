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
CONCEPT = ("C7 semidirect C6 with action 3, order 42, shapes (1,2)x(4,5): degree-four component columns and controlled rank deficits should preserve near-zero circuit failures while increasing measured k toward 73 in the 0.100 cell.")
GROUP = ("Keep the parent's order-42 group and encoding unchanged: all five measured parent codes had zero failures in both bases and fitness 7.1776. The normal C7 subgroup allows the rotation-only B component to be screened through exactly enumerable 35-variable factors; this is an algebraic screening opportunity, not evidence of a circuit advantage.")
SHAPES = ("A=(1,2), B=(4,5) gives n=14*42=588 and guaranteed k>=42, rate 1/14. Component lengths/check counts are 84/42 and 210/168. Target exact k=67..73, always below rate 0.125. The parent has no failed-shot block attribution; its minimum-weight witnesses predominantly occupied sector-one Acol/Bcol blocks. The asymmetric bed tests ceiling utilization rather than claiming an isolated block repair.")
FILLING = ("A cells are four distinct terms with canonical C6-count architectures whose joint quotient gcd is 1+y. B cells are singleton rotations with explicit normalized holonomy anchors. Both component column degrees are four; HX weight is 12 and HZ weight is 9, giving 33264 six-round CNOTs, above the parent's 30240. Optimize exact B-factor distance and multiplicities, screen A ranks 35 or 41, exclude component words through weight six, and cap sampled protection rewards. Final verified lifts enforce k=67..73, connectivity, absence of quantum logicals through weight four, and CSS. Preserving zero failures despite the extra gate load is the falsifiable hypothesis; distance is not fitness.")
FAMILY_NAME = "r010_frobenius7by6_12x45_rankwindow_factorbook_v1"

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

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (4, 5)               # (rb, cb)


def _shape_architecture(idx):
    """Canonical coset-count pairs, not images of a saved filling."""
    from itertools import combinations_with_replacement

    def remainder(a, b):
        while a and a.bit_length() >= b.bit_length():
            a ^= b << (a.bit_length() - b.bit_length())
        return a

    def gcd(a, b):
        while b:
            a, b = b, remainder(a, b)
        return a

    def polynomial(profile):
        return sum((v & 1) << j for j, v in enumerate(profile))

    profiles = set()
    for terms in combinations_with_replacement(range(6), 4):
        profile = tuple(terms.count(j) for j in range(6))
        if max(profile) > 3:
            continue
        # Conservatively quotient independent cell translations and
        # inversion of C6. Tanner-isomorphism deduplication remains final.
        images = [tuple(profile[(t + sign*j) % 6] for j in range(6))
                  for t in range(6) for sign in (-1, 1)]
        profiles.add(min(images))

    ordered = sorted(profiles)
    book = []
    for a, pa in enumerate(ordered):
        for pb in ordered[a:]:
            # y^6+1=0b1000001. Joint gcd 1+y gives quotient row rank five.
            if gcd(gcd(polynomial(pa), polynomial(pb)), 65) == 3:
                book.append((pa, pb))
    book.sort(key=lambda ab: (
        max(max(ab[0]), max(ab[1])),
        sum(v*v for profile in ab for v in profile), ab))
    assert len(book) >= 20
    return book[idx % len(book)], idx // len(book)


def _shape_draw_A(rng, architecture):
    result = []
    for profile in architecture:
        shift = rng.randrange(6)
        cell = []
        for b, count in enumerate(profile):
            cell.extend((a, (b + shift) % 6)
                        for a in rng.sample(range(7), count))
        assert len(cell) == len(set(cell)) == 4
        result.append(cell)
    return [result]


def _shape_draw_B(rng, idx):
    """Normalized C7 voltages with a deliberate rectangle-holonomy ratio."""
    B = [[[(0, 0)] for _ in range(5)] for _ in range(4)]
    for i in range(1, 4):
        for j in range(1, 5):
            B[i][j] = [(rng.randrange(7), 0)]
    B[1][1] = [(1, 0)]
    B[1][2] = [(1 + idx % 6, 0)]
    return B


def _shape_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _shape_kernel_basis(rows, width):
    """Fundamental kernel basis from verified component rows."""
    pivots, basis = {}, []
    for c, syndrome in enumerate(_shape_columns(rows, width)):
        word = 1 << c
        while syndrome:
            p = syndrome.bit_length() - 1
            if p not in pivots:
                pivots[p] = (syndrome, word)
                break
            old_s, old_w = pivots[p]
            syndrome ^= old_s
            word ^= old_w
        if not syndrome:
            basis.append(word)
    return basis


def _shape_exact_factor(rows):
    """Complete weight spectrum of the 35-variable, dimension-ten factor."""
    basis = _shape_kernel_basis(rows, 35)
    if len(basis) != 10:
        return None
    spectrum = [0] * 36
    word = 0
    for t in range(1, 1 << len(basis)):
        word ^= basis[(t & -t).bit_length() - 1]
        spectrum[word.bit_count()] += 1
    distance = next(w for w in range(1, 36) if spectrum[w])
    return distance, tuple(spectrum[1:15])


def _shape_short_boundary(rows, width):
    """Exact kernel-word exclusion through six, plus a boundary heuristic."""
    cols = _shape_columns(rows, width)
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return True, 0
        seen.add(syndrome)
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in seen:
                return True, 0
            seen.add(syndrome)
    boundary = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return True, 0
                seen.add(syndrome)
                weight = syndrome.bit_count()
                if weight <= 4:
                    boundary += (6 - weight) ** 2
    return False, boundary


def _shape_kernel_upper(rng, rows, width, trials=40):
    """Sampled upper bound only; it is neither a certificate nor circuit pL."""
    cols = _shape_columns(rows, width)
    order = list(range(width))
    best = width + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
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
            if not syndrome:
                basis.append(word)
                best = min(best, word.bit_count())
        for i, word in enumerate(basis):
            for other in basis[:i]:
                best = min(best, (word ^ other).bit_count())
    return best


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
    """Optimize unequal components, then verify the complete CSS code."""
    assert G.q == 42
    sa, sb = [[4, 4]], [[1] * 5 for _ in range(4)]
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    architecture, mode = _shape_architecture(idx)
    q = G.q

    def A_rows(A):
        # A zero opposite cell exposes precisely the scaffold's A-left
        # component. This helper lift is not submitted as a candidate.
        hx, _, _ = lifted_matrices(G, A, [[[]]])
        return hx

    def B_factor(B):
        # Obtain B-right from the verified lift with a unit A.
        # Rotation labels preserve each C6 coset. Restrict to coset zero,
        # giving 28 checks and 35 variables, not an independent lift.
        unit_A = [[[G.e], [G.e]]]
        _, hz, _ = lifted_matrices(G, unit_A, B)
        mask = (1 << 7) - 1
        rows = []
        for alpha in range(4):
            for r in range(7):
                row = hz[alpha * q + r]
                rows.append(sum(
                    ((row >> (j * q)) & mask) << (7 * j)
                    for j in range(5)))
        return rows

    # Exactly enumerate B factors. Rank 25 leaves dimension ten;
    # the six rotation-only factors have equivalent weight spectra.
    bpool = []
    for trial in range(160):
        B = _shape_draw_B(rng, idx)
        br = B_factor(B)
        exact = _shape_exact_factor(br)
        if exact is None:
            continue
        db, spectrum = exact
        if db < 10:
            continue
        cycles = _four_cycles(br)
        key = (-db, spectrum, cycles, trial)
        bpool.append((key, B, db, cycles))
        bpool.sort(key=lambda item: item[0])
        del bpool[4:]
    if not bpool:
        raise ValueError("no dimension-ten B factor passed exact screening")

    # Rank 41 is the quotient-deficit baseline. Rank 35 tests one
    # additional six-dimensional deficit. These are component screens:
    # final k is always calculated from the complete verified matrices.
    apools = {35: [], 41: []}
    for trial in range(768):
        A = _shape_draw_A(rng, architecture)
        ar = A_rows(A)
        rank = gf2_rank(ar)
        if rank not in apools:
            continue
        cycles = _four_cycles(ar)
        apools[rank].append((cycles, trial, A, ar))
        apools[rank].sort(key=lambda item: (item[0], item[1]))
        del apools[rank][8:]
        if trial >= 383 and len(apools[35]) == 8:
            break

    afinal = []
    for rank, pool in apools.items():
        for cycles, trial, A, ar in pool:
            bad, boundary = _shape_short_boundary(ar, 2 * q)
            if bad:
                continue
            da = _shape_kernel_upper(rng, ar, 2 * q)
            afinal.append((rank, A, da, boundary, cycles, trial))
    if not afinal:
        raise ValueError("no A architecture passed the weight-six screen")

    options = []
    for rank, A, da, boundary, acycles, atrial in afinal:
        for bkey, B, db, bcycles in bpool:
            protection = min(da, db)
            # A protection floor comes before a higher measurement ceiling.
            # Above it, seek the rank-35 architecture, then cap optimistic
            # sampled-distance rewards and reduce boundary exposure.
            if mode % 2:
                refinement = (boundary, -min(protection, 14),
                              acycles + 6 * bcycles)
            else:
                refinement = (-min(protection, 14), boundary,
                              acycles + 6 * bcycles)
            key = (int(protection < 12), int(rank != 35),
                   refinement, abs(da - db), atrial, bkey)
            options.append((key, A, B, rank, da, db))
    options.sort(key=lambda item: item[0])

    for key, A, B, rank, da, db in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 588 and 67 <= k <= 73 and 8 * k < n):
            continue
        if not _connected_checks(hx, hz, n):
            continue
        if _has_short_logical(hx, hz, n):
            continue
        assert max(row.bit_count() for row in hx) == 12
        assert max(row.bit_count() for row in hz) == 9
        assert css_ok(hx, hz)
        return {"A": A, "B": B, "design": idx,
                "architecture": architecture, "k": k,
                "A_rank": rank, "component_bounds": (da, db),
                "screen": key}
    raise ValueError("no screened complete code retained k=67..73")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without late randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily increases this bed's circuit score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; both were limited by their sample ceilings.
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
