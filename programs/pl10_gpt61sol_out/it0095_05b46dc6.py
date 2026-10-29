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
CONCEPT = ("C13 semidirect C6, action 4, order 78, shapes (1,2)x(1,3): connected, gauge-fixed components with mostly three-term cells and balanced short-boundary screening should improve worse-basis circuit fitness beyond the 0.300 incumbent's 5.6080.")
GROUP = ("New order-78 Frobenius bed, absent from BEDS and genuinely different from Dic11: 4^3=-1 and 4^6=1 modulo 13. Related order-39 and order-42 Frobenius multi-row beds achieved reliable circuits, motivating this larger-lift experiment without establishing a single-row group advantage.")
SHAPES = ("A=(1,2), B=(1,3) have 156 and 234 variables, respectively, with 78 checks each. n=546 and kb=156 give guaranteed rate 2/7=0.285714 in cell 0.300; the shape permits q<=85 and this group fixes q=78. Verified full ranks retain 0.275<=k/n<0.325. The saturated parent has no failed-shot attribution. The 0.300 incumbent had broad collective failures across all 180 logicals, with mostly 1.1x Acol/Bcol enrichment rather than an isolated Brow hotspot.")
FILLING = ("Five support signatures cross canonical coset-incidence cards, not saved-code orbit copies. Every component has a mixed-coset anchor whose support differences generate the group; fix the A anchor to remove automorphism redundancy and optimize the remaining labels. All cells contain 2..4 distinct terms; both row-weight sums are <=12. Verified lifts screen actual rate, connectivity, component words<=4, cycles, sampled protection floors, normalized triple boundaries and actual quantum logicals<=4, with a legal fallback for optional protection screens. The all-trinomial signature uses 23868 six-round CNOTs versus the 0.300 incumbent's 25200. These are pL hypotheses targeting its 100/53000 Z and 100/65500 X failures and thousands of collective flips, not distance-based fitness claims.")
FAMILY_NAME = "r030_frob13by6_12x13_anchor_coset_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 78

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 semidirect C6 requires q=78")
    # x^13=y^6=1, y*x*y^-1=x^4.
    # 4 has order six modulo 13; (a,b) denotes x^a*y^b.
    act = (1, 4, 3, 12, 9, 10)
    els = [(a, b) for b in range(6) for a in range(13)]

    def red(g):
        return (int(g[0]) % 13, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + act[b] * c) % 13, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-act[bi] * a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _dv_columns(rows, n):
    """Column syndromes of an already verified lifted matrix."""
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _dv_pivots(rows):
    piv = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in piv:
                row ^= piv[p]
            else:
                piv[p] = row
                break
    return piv


def _dv_remainder(word, piv):
    while word:
        p = word.bit_length() - 1
        if p not in piv:
            return word
        word ^= piv[p]
    return 0


def _dv_low4(rows, stabilizers, n):
    """Exact existence test for a nontrivial kernel word of weight <=4."""
    cols = _dv_columns(rows, n)
    piv = _dv_pivots(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if _dv_remainder(word ^ seen[syndrome], piv):
                return True
        else:
            seen[syndrome] = word
    for i in range(n):
        si = cols[i]
        wi = 1 << i
        for j in range(i):
            syndrome = si ^ cols[j]
            word = wi | (1 << j)
            if syndrome in seen:
                if _dv_remainder(word ^ seen[syndrome], piv):
                    return True
            else:
                seen[syndrome] = word
    return False


def _dv_components(hx, hz, q):
    """Extract both components exclusively from the verified CSS lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    arows = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            packed = 0
            for j in range(ca):
                packed |= ((row >> (j * cb * q)) & mask) << (j * q)
            arows.append(packed)
    # A-column zero in HZ contains contiguous B-right component blocks.
    bmask = (1 << (cb * q)) - 1
    brows = [hz[a * q + r] & bmask
             for a in range(rb) for r in range(q)]
    return arows, brows


def _dv_cycles(rows, n):
    """Four-cycle count and a row-intersection triangle heuristic."""
    nr = len(rows)
    adjacency = [0] * nr
    c4 = 0
    for i in range(nr):
        for j in range(i):
            common = (rows[i] & rows[j]).bit_count()
            c4 += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    triangles = 0
    for i in range(nr):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = 0
    for col in _dv_columns(rows, n):
        degree = col.bit_count()
        stars += degree * (degree - 1) * (degree - 2) // 6
    # Exact six-cycle count when intersections are simple; heuristic otherwise.
    return c4, max(0, triangles - stars)


def _dv_kernel_estimate(rng, rows, n, trials):
    """Upper bound from randomized fundamental kernel circuits."""
    cols = _dv_columns(rows, n)
    order = list(range(n))
    best = n + 1
    for _ in range(trials):
        rng.shuffle(order)
        piv = {}
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in piv:
                    s, w = piv[p]
                    syndrome ^= s
                    word ^= w
                else:
                    piv[p] = (syndrome, word)
                    break
            if not syndrome and word:
                best = min(best, word.bit_count())
    return best


def NEW_CTX(rng, G, idx):
    """Optimize an anchor-connected support/coset card for cell 0.300."""
    from itertools import product

    q = G.q
    an, bn = A_SHAPE[1] * q, B_SHAPE[1] * q

    # Distinct component degree signatures. The two independent caps are:
    # sum(A) + max(B) <= 12 and sum(B) + max(A) <= 12.
    profiles = (
        ((3, 3), (3, 3, 3)),
        ((3, 3), (2, 3, 3)),
        ((3, 4), (2, 3, 3)),
        ((3, 3), (2, 3, 4)),
        ((4, 4), (2, 2, 4)),
    )

    # Coset words specify full cells, not random support counts.
    # No word has a coset gap three, which would introduce an
    # involution ratio. Repeated cosets still use DISTINCT elements.
    # The alternatives have different intra-cell ratio-order profiles.
    templates = {
        2: ((0, 0), (0, 1)),
        3: ((0, 0, 1), (0, 1, 2),
            (0, 0, 2), (0, 0, 0)),
        4: ((0, 0, 1, 1), (0, 0, 0, 1),
            (0, 0, 1, 2), (0, 0, 0, 2)),
    }

    def deck_book(weights):
        # Reserve one cell with a nonzero normal-subgroup difference
        # and a coset-one difference. Together these generate all G.
        anchor = next(j for j, w in enumerate(weights) if w >= 3)
        alphabets = [
            (templates[w][0],) if j == anchor else templates[w]
            for j, w in enumerate(weights)]
        # Column permutations are gauges, not additional design cards.
        return sorted({tuple(sorted(deck))
                       for deck in product(*alphabets)})

    profile = idx % len(profiles)
    case = idx // len(profiles)
    wa, wb = profiles[profile]
    cards = sorted(product(deck_book(wa), deck_book(wb)))
    quota = max(1, (N_SLOTS - 1 - profile) // len(profiles) + 1)
    # The default 20 slots and the 60-slot feedback budget both use
    # different cards within each support signature.
    card_number = min(len(cards) - 1, case * len(cards) // quota)
    adeck, bdeck = cards[card_number]
    sa = [[len(word) for word in adeck]]
    sb = [[len(word) for word in bdeck]]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    def draw_side(deck, fix_anchor):
        anchor = next(
            j for j, word in enumerate(deck)
            if len(word) >= 3 and word == templates[len(word)][0])
        cells = []
        for j, word in enumerate(deck):
            if fix_anchor and j == anchor:
                # x -> x^u and y -> x^v*y normalize this generating
                # pair. Do not waste label proposals on those images.
                cell = [(0, 0), (1, 0), (0, 1)]
                if len(word) == 4:
                    cell.append((2, 1))
            else:
                # Independent column gauges fix one term to identity.
                cell = [(0, 0)]
                for b in word[1:]:
                    g = (rng.randrange(13), b)
                    while g in cell:
                        g = (rng.randrange(13), b)
                    cell.append(g)
            cells.append(cell)
        return [cells]

    def connected(rows, width):
        """Connectivity of the actual component/check Tanner graph."""
        cols = _dv_columns(rows, width)
        if any(value == 0 for value in cols):
            return False
        parent = list(range(len(rows)))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for value in cols:
            first = value & -value
            a = first.bit_length() - 1
            value ^= first
            while value:
                bit = value & -value
                b = bit.bit_length() - 1
                parent[root(b)] = root(a)
                value ^= bit
        return len({root(r) for r in range(len(rows))}) == 1

    def boundary_exposure(rows, width):
        """Exact small-triple-boundary heuristic, normalized per variable.

        This is neither a circuit simulation nor a distance certificate.
        All columns come from the verified construction helper.
        """
        cols = _dv_columns(rows, width)
        exposure = 0
        for i in range(width - 2):
            x = cols[i]
            for j in range(i + 1, width - 1):
                pair = x ^ cols[j]
                for h in range(j + 1, width):
                    weight = (pair ^ cols[h]).bit_count()
                    if weight <= 4:
                        exposure += (6 - weight) ** 2
        return exposure / width

    eligible = []
    for batch in range(2):
        pool = []
        for attempt in range(96):
            A = draw_side(adeck, True)
            B = draw_side(bdeck, False)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _dv_components(hx, hz, q)
            a4, a6 = _dv_cycles(ar, an)
            b4, b6 = _dv_cycles(br, bn)
            coarse = (max(a4, b4), a4 + b4,
                      max(a6, b6), a6 + b6)
            item = (coarse, attempt, A, B, hx, hz, n, ar, br)
            pool.append(item)
            pool.sort(key=lambda entry: (entry[0], entry[1]))
            del pool[32:]

        # Rank decisions use the complete evaluator-convention lift.
        # No component-rank or representation-theory formula replaces k.
        for item in pool:
            coarse, attempt, A, B, hx, hz, n, ar, br = item
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 275*n <= 1000*k < 325*n):
                continue
            if not connected(ar, an) or not connected(br, bn):
                continue
            bad_a = int(_dv_low4(ar, [], an))
            bad_b = int(_dv_low4(br, [], bn))
            protection = (max(bad_a, bad_b), bad_a + bad_b)
            eligible.append((protection, coarse, attempt, item, k))
            if len(eligible) >= 12:
                break
        if eligible:
            break

    if not eligible:
        raise ValueError("no connected filling retained cell 0.300")

    eligible.sort(key=lambda entry: entry[:3])
    finalists = []
    for protection, coarse, attempt, item, k in eligible[:4]:
        _, _, A, B, hx, hz, n, ar, br = item
        da = _dv_kernel_estimate(rng, ar, an, 32)
        db = _dv_kernel_estimate(rng, br, bn, 32)
        ba = boundary_exposure(ar, an)
        bb = boundary_exposure(br, bn)
        distance = min(da, db)
        # Remove witnessed weak protection first. Above the modest
        # sampled floor, prioritize normalized trapping boundaries.
        # Large sampled upper bounds are capped rather than chased.
        key = (protection, int(distance < 8),
               max(ba, bb), ba + bb, coarse,
               -min(distance, 12), abs(da - db), -k, attempt)
        finalists.append((key, item, k))
    finalists.sort(key=lambda entry: entry[0])

    # Optional short-logical screening must not void a legal architecture.
    # Prefer passing candidates; otherwise retain the best rate-legal one.
    chosen = finalists[0]
    short_warning = True
    for entry in finalists:
        key, item, k = entry
        _, _, A, B, hx, hz, n, _, _ = item
        if _dv_low4(hx, hz, n) or _dv_low4(hz, hx, n):
            continue
        if not connected(hx + hz, n):
            continue
        chosen = entry
        short_warning = False
        break

    key, item, k = chosen
    _, _, A, B, hx, hz, n, _, _ = item
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 275*n <= 1000*k < 325*n
    return {"idx": idx, "profile": profile, "card": card_number,
            "architecture": (adeck, bdeck), "k": k,
            "screen": key, "short_warning": short_warning,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected complete 2..4-term cell without late redraws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal uniform D7 labels on (6,2)x(6,2) do not suffice for reliable memory: measured s=3.30-3.59, abundant weight-four sector-two logicals, and roughly 10-45 percent failed-shot rates.
# REFUTED: d2s=26 guarantees ceiling-level circuit performance on this Dic11 bed: parent slot 34 had two Z failures, 38 collective logical flips and s=6.864, while slot 26 with the same d2s had no failures.
# REFUTED: Canonical ratio architectures alone prevent Tanner-equivalent slots: the Dic11 parent produced only 28 distinct codes from 60 slots, with 32 isomorphic duplicates.
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
