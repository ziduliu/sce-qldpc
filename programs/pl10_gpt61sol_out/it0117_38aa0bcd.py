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
CONCEPT = ("C7 semidirect C6 with action 3, order 42, shapes (1,2)x(4,5): unit-trinomial A cells and a rank-one C6 quotient architecture in B should produce fresh k=60..72 codes retaining low worse-basis circuit failure rates while increasing the parent's measurement ceiling.")
GROUP = ("Keep the parent's order-42 group unchanged: all five measured parent codes had zero failures in both bases, at fitness 7.1776. This supports retaining the group for a shape experiment, not claiming a group-specific advantage. The new factor 14 gives legal n=588 at the same order.")
SHAPES = ("A=(1,2), B=(4,5) give components with 84 variables/42 checks and 210 variables/168 checks; n=588, kb=42 and guaranteed rate 1/14. Verified full ranks require 60<=k<=72 and 8*k<n. Parent failures have no block attribution because none occurred; its minimum-weight witnesses predominantly occupied sector-one Acol/Bcol blocks.")
FILLING = ("A contains two individually verified unit trinomials. B has one mixed-coset trinomial per column and singleton cells elsewhere; its C6 projection is an outer product of monomial row and column phases, of binary rank six. Support-and-relative-order cards distinguish designs before fresh voltage optimization. A row weight is six and column weight three; B column weight is six and row weights are at most nine, so BOTH CSS check-weight bounds are twelve. Verified component actions screen ranks, short kernels and small boundaries; final full lifts screen actual k, connectivity, logicals<=4 and CSS. Sampled kernel bounds are capped at fourteen and are not pL predictions. Parent scores saturated, while slot37 required 23 Z-basis LSD resolutions; the hypothesis is increased ceiling with controlled ambiguity, not distance-only improvement.")
FAMILY_NAME = "r010_frobenius7by6_12x45_unit_quotientrank1_v1"

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


def _design_supports(idx):
    """Canonical support/relative-order cards, not saved-code relabelings."""
    layouts = ((0, 0, 1, 1, 2), (0, 0, 1, 2, 3))
    cards = []
    for layout in layouts:
        seen = set()
        for mask in range(32):
            gaps = tuple(2 if (mask >> j) & 1 else 1
                         for j in range(5))
            # Columns can permute, and rows with identical incidence
            # counts can exchange. Gap one and gap two have different
            # mixed-coset relative orders: six and three.
            signature = tuple(sorted(
                (sum(r == i for r in layout),
                 sum(r == i and gaps[j] == 2
                     for j, r in enumerate(layout)))
                for i in range(4)))
            if signature in seen:
                continue
            seen.add(signature)
            for agaps in ((1, 1), (1, 2), (2, 2)):
                cards.append((layout, agaps, gaps))
    assert len(cards) >= N_SLOTS
    card = cards[idx * len(cards) // max(1, N_SLOTS)]
    layout, agaps, bgaps = card
    sa = [[3, 3]]
    sb = [[1 + 2 * int(layout[j] == i) for j in range(5)]
          for i in range(4)]
    return sa, sb, agaps, bgaps, card


def _wide_trinomial(rng, b, gap):
    """A same-coset pair plus a distinct-coset singleton."""
    a = rng.randrange(7)
    c = (a + rng.randrange(1, 7)) % 7
    return [(a, b), (c, b),
            (rng.randrange(7), (b + gap) % 6)]


def _wide_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _wide_kernel4(cols):
    """Exact existence of a nonzero component kernel word of weight <=4."""
    seen = {0}
    for value in cols:
        if value in seen:
            return True
        seen.add(value)
    for i, x in enumerate(cols):
        for j in range(i):
            value = x ^ cols[j]
            if value in seen:
                return True
            seen.add(value)
    return False


def _wide_boundaries(cols, q):
    """Worst-block anchored triple boundaries: a trapping-set surrogate."""
    costs = []
    for anchor in range(0, len(cols), q):
        cost = 0
        for i, x in enumerate(cols):
            if i == anchor:
                continue
            pair = cols[anchor] ^ x
            for j in range(i):
                if j == anchor:
                    continue
                weight = (pair ^ cols[j]).bit_count()
                if weight <= 4:
                    cost += (5 - weight) ** 2
        costs.append(cost)
    return max(costs), sum(costs)


def _wide_kernel_estimate(rng, cols, trials):
    """Verified kernel witnesses from random information sets; upper bounds."""
    order = list(range(len(cols)))
    best = len(cols) + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in pivots:
                    s, w = pivots[p]
                    syndrome ^= s
                    word ^= w
                else:
                    pivots[p] = syndrome, word
                    break
            if not syndrome:
                best = min(best, word.bit_count())
                basis.append(word)
        # Greedy reductions remain genuine kernel words.
        for i, word in enumerate(basis):
            weight = word.bit_count()
            for other in basis[:i]:
                reduced = word ^ other
                w = reduced.bit_count()
                best = min(best, w)
                if w < weight:
                    word, weight = reduced, w
            basis[i] = word
        if best <= 6:
            break
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
    """Optimize unequal components within a deliberate quotient contract."""
    sa, sb, agaps, bgaps, card = _design_supports(idx)
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    q = G.q
    mask = (1 << q) - 1

    # Cache singleton blocks from the verified helper, not a separately
    # implemented permutation lift. Final full matrices must reproduce
    # these screened component rows exactly.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        x, z, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in x)
        actions["B"][G._key(g)] = tuple(row & mask for row in z)

    def component(grid, side):
        result = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    term = actions[side][G._key(g)]
                    shift = j * q
                    for t, bits in enumerate(term):
                        out[t] ^= bits << shift
            result.extend(out)
        return result

    def refine(pool, width):
        entries = []
        for four, trial, grid, rows, k in pool:
            if not _connected_checks(rows, [], width):
                continue
            cols = _wide_columns(rows, width)
            bad = int(_wide_kernel4(cols))
            boundary = _wide_boundaries(cols, q)
            d = _wide_kernel_estimate(rng, cols, 40)
            key = (bad, int(d < 14), -min(d, 14),
                   boundary, four, trial)
            entries.append(dict(key=key, grid=grid, rows=rows,
                                bad=bad, boundary=boundary,
                                d=d, four=four, k=k))
        entries.sort(key=lambda entry: entry["key"])
        return entries

    apool = []
    for trial in range(256):
        A = [[_wide_trinomial(rng, rng.randrange(6), gap)
              for gap in agaps]]
        # One unit is enough for the module argument; require both as a
        # stronger filling contract. Unit status is measured on the lift.
        if any(gf2_rank(component([[cell]], "A")) != q
               for cell in A[0]):
            continue
        ar = component(A, "A")
        four = _four_cycles(ar)
        apool.append((four, trial, A, ar, 0))
        apool.sort(key=lambda item: (item[0], item[1]))
        del apool[6:]
    if not apool:
        raise ValueError("no unit-trinomial A pair in contract")
    afinal = refine(apool, 2 * q)
    if not afinal:
        raise ValueError("no connected A component")

    def draw_B():
        up = [rng.randrange(6) for _ in range(4)]
        vp = [rng.randrange(6) for _ in range(5)]
        B = [[[] for _ in range(5)] for _ in range(4)]
        for i in range(4):
            for j in range(5):
                b = (up[i] + vp[j]) % 6
                base = (rng.randrange(7), b)
                if sb[i][j] == 1:
                    B[i][j] = [base]
                else:
                    # The extra same-coset pair cancels under C7->1,
                    # leaving precisely the monomial y^(up[i]+vp[j]).
                    c = rng.randrange(7)
                    d = (c + rng.randrange(1, 7)) % 7
                    t = (b + bgaps[j]) % 6
                    B[i][j] = [base, (c, t), (d, t)]
        return B

    # A unit implies a free rank-one kernel module. This motivates
    # k=5*q-rank(B), but the full verified HX/HZ ranks remain decisive.
    # Keep the k=60 fallback rather than void a legal card when the
    # rarer extra rank deficits are absent.
    bpools = {60: [], 66: [], 72: []}
    high_count = 0
    for trial in range(2048):
        B = draw_B()
        br = component(B, "B")
        suggested_k = 5 * q - gf2_rank(br)
        if suggested_k not in bpools:
            continue
        if suggested_k == 60 and trial >= 128:
            continue
        if suggested_k > 60:
            high_count += 1
        four = _four_cycles(br)
        pool = bpools[suggested_k]
        pool.append((four, trial, B, br, suggested_k))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[4:]
        if trial >= 511 and high_count >= 12:
            break

    bfinal = []
    for k in (72, 66, 60):
        bfinal.extend(refine(bpools[k], 5 * q))
    if not bfinal:
        raise ValueError("no connected B component in target rank window")

    options = []
    for a in afinal:
        for b in bfinal:
            distance = min(a["d"], b["d"])
            ba, bb = a["boundary"], b["boundary"]
            # Once the coarse protection floor is met, seek the higher
            # sampling ceiling, then less worst-component ambiguity.
            # These are structural hypotheses, not measured pL.
            key = (a["bad"] + b["bad"], int(distance < 14),
                   -b["k"], max(ba[0], bb[0]), ba[1] + bb[1],
                   max(a["four"], b["four"]),
                   a["four"] + b["four"], -min(distance, 14))
            options.append((key, a, b))
    options.sort(key=lambda entry: entry[0])

    for key, a, b in options:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        # Compare the actual evaluator-convention components with the
        # cached singleton composition used throughout the search.
        ar = [row & ((1 << (2 * q)) - 1)
              for row in lifted_matrices(G, A, [[[]]])[0]]
        br = [row & ((1 << (5 * q)) - 1)
              for row in hz[:4 * q]]
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (60 <= k <= 72 and 8 * k < n):
            continue
        if not _connected_checks(hx, hz, n):
            continue
        if _has_short_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        assert n == 588
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        return {"A": A, "B": B, "design": idx, "card": card,
                "k": k, "screen": key}

    raise ValueError("no full verified low-rate candidate passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; no subsequent redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger d2s necessarily improves this bed's measured score: zero-failure slot58 had d2s=22,k=45,s=7.165, below zero-failure slot20 with d2s=20,k=58,s=7.276; both were limited by their sampling ceilings.
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
