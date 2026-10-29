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
CONCEPT = ("Cyclic C29, order 29, shapes (2,4)x(2,4): canonical binomial-ratio architectures with reduced short-word multiplicity and small boundaries should improve worse-basis circuit fitness beyond the current 0.200 frontier while keeping check weight nine.")
GROUP = ("Jump from nonabelian C7:C3 to cyclic C29, a genuinely different group and construction rule. The recorded C29 (2,4)x(2,4) bed reached about 7.2, and related C27 degree-three components scored 7.2133. These support testing this load profile, not claiming a causal group advantage. Every nonidentity C29 difference has order 29.")
SHAPES = ("Equal (2,4) components each have 116 variables and 58 checks. n=580 and kb=116 give guaranteed rate 0.200; verified complete ranks require 116<=k<=128, remaining in that cell. The size factor permits q<=29. Parent measured codes had zero failures and no Acol/Brow failure attribution, so there is no demonstrated hotspot to repair; its best measured k=30 code instead saturated at s=6.989.")
FILLING = ("Both sides use row counts (1,1,2,2)/(2,2,1,1): component rows have six terms, columns three, both CSS weights nine, and six-round CNOT count 25056. Each slot receives a different canonical unordered pair of binomial-step architectures, quotienting common multipliers, doubleton reversal and support-preserving permutations. Fresh gauge-normalized holonomies compete through verified component connectivity, ranks, exact short-word collisions, weight-six exposure, sampled short-word multiplicity and triple boundaries. Final complete lifts screen rate, CSS and quantum logicals<=4. Parent slots 11 and 28 both had zero failures but different sampling ceilings; the intended gain is reliable higher-k memory, not a larger distance upper bound.")
FAMILY_NAME = "r020_C29_24x24_canonical_ratio_pair_ambiguity_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C29 requires order 29")
    # Additive cyclic group, not the parent's semidirect product.
    els = [(a,) for a in range(29)]

    def red(g):
        return (int(g[0]) % 29,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 29,)

    def inv(g):
        return ((-int(g[0])) % 29,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _f21_columns(hx, hz, q):
    """Project component columns from the VERIFIED lifted matrices.

    A: sector 1 of X checks at B-column zero.
    B: sector 1 of Z checks at A-column zero.
    No independent left/right-lift implementation is used.
    """
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    ac = [0] * (ca * q)
    bc = [0] * (cb * q)
    amask = sum(((1 << q) - 1) << (j * cb * q)
                for j in range(ca))
    bmask = (1 << (cb * q)) - 1
    for i in range(ra):
        for r in range(q):
            bits = hx[(i * cb) * q + r] & amask
            check = 1 << (i * q + r)
            while bits:
                low = bits & -bits
                c = low.bit_length() - 1
                ac[(c // (cb * q)) * q + c % q] |= check
                bits ^= low
    for a in range(rb):
        for r in range(q):
            bits = hz[a * q + r] & bmask
            check = 1 << (a * q + r)
            while bits:
                low = bits & -bits
                bc[low.bit_length() - 1] |= check
                bits ^= low
    return ac, bc


def _f21_stats(cols, nchecks):
    """Exact short-word counts, Tanner four-cycles and connectivity."""
    parent = list(range(nchecks))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checkpairs = {}
    c4 = 0
    for col in cols:
        rows = []
        bits = col
        while bits:
            low = bits & -bits
            rows.append(low.bit_length() - 1)
            bits ^= low
        if rows:
            for r in rows[1:]:
                parent[root(r)] = root(rows[0])
        for t, r in enumerate(rows):
            for s in rows[t + 1:]:
                pair = (r, s)
                old = checkpairs.get(pair, 0)
                c4 += old
                checkpairs[pair] = old + 1

    colset = set(cols)
    repeated = len(cols) - len(colset)
    pairseen = {}
    words3 = words4 = 0
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            syndrome = x ^ y
            words3 += int(syndrome in colset)
            old = pairseen.get(syndrome, 0)
            words4 += old
            pairseen[syndrome] = old + 1
    components = len({root(r) for r in range(nchecks)})
    return (repeated, words3, words4), c4, components


def _f21_traps(cols):
    """Small boundaries of three-variable sets; repeated triple syndromes.

    The first statistic is a trapping-set heuristic. The second also
    detects weight-six component words when shorter words are absent.
    Neither statistic is asserted to predict circuit pL exactly.
    """
    boundary = repeated = 0
    seen = {}
    count = len(cols)
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, count):
                syndrome = pair ^ cols[k]
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
                old = seen.get(syndrome, 0)
                repeated += old
                seen[syndrome] = old + 1
    return boundary, repeated


def _c29_architectures(idx):
    """Canonical ratio contracts, independent of the voltage-search RNG.

    Each row has two binomials. Quotient their undirected step pairs by
    common C29 multipliers, row exchange and independent pair exchange.
    This removes obvious orbit copies, not arbitrary Tanner isomorphisms.
    """
    from itertools import combinations, combinations_with_replacement

    pairs = list(combinations(range(1, 10), 2))
    representatives = set()
    for first, second in combinations_with_replacement(pairs, 2):
        best = None
        for u in range(1, 29):
            image = tuple(sorted(
                tuple(sorted(min((u * d) % 29, (-u * d) % 29)
                             for d in pair))
                for pair in (first, second)))
            if best is None or image < best:
                best = image
        representatives.add(best)

    book = sorted(representatives)
    count = len(book)
    total = count * (count + 1) // 2
    assert total >= max(20, N_SLOTS)
    position = idx * total // max(20, N_SLOTS)
    # Index unordered pairs without constructing a quadratic-size list.
    for a in range(count):
        width = count - a
        if position < width:
            return book[a], book[a + position]
        position -= width
    raise ValueError("ratio-contract index out of range")


def _c29_rows(G, side, grid):
    """Extract components using the scaffold's lift, including its actions."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (4 * G.q)) - 1
    return [row & mask for row in rows]


def _c29_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _c29_sampled_words(rng, cols, q):
    """Actual kernel witnesses; minimum remains only an upper bound.

    Also record sampled weight<=10 multiplicity and block concentration.
    These statistics are warnings about ambiguity, not measured pL.
    """
    order = list(range(len(cols)))
    best = len(cols) + 1
    short = set()

    def offer(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        best = min(best, weight)
        if weight <= 10:
            short.add(word)

    for _ in range(48):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                offer(word)

        # Basis operations preserve membership in this verified kernel.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for j in range(i):
                    proposal = word ^ basis[j]
                    offer(proposal)
                    pw = proposal.bit_count()
                    if proposal and pw < weight:
                        word, weight = proposal, pw
                basis[i] = word

    mask = (1 << q) - 1
    loads = [sum(bool((word >> (j * q)) & mask) for word in short)
             for j in range(4)]
    return best, len(short), max(loads)


def _c29_low4(checks, stabilizers, width):
    """Exact existence of a non-stabilizer kernel word through weight four."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    cols = _c29_columns(checks, width)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
        elif nontrivial(word ^ previous):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif nontrivial(word ^ previous):
                return True
    return False


def _c29_select(rng, G, side, architecture):
    """Search holonomies while preserving the slot's ratio architecture."""
    q = G.q
    width = 4 * q
    supports = ((1, 1, 2, 2), (2, 2, 1, 1))

    def draw():
        grid = [[], []]
        for i in range(2):
            binomial = 0
            for j in range(4):
                # Fix top-row cell anchors and one bottom-row anchor.
                # These are whole-row/column gauges in the cyclic lift.
                origin = 0 if i == 0 or j == 0 else rng.randrange(q)
                cell = [(origin,)]
                if supports[i][j] == 2:
                    step = architecture[i][binomial]
                    binomial += 1
                    if rng.randrange(2):
                        step = -step
                    cell.append(((origin + step) % q,))
                grid[i].append(cell)
        return grid

    for batch in range(3):
        pool = []
        for trial in range(160):
            grid = draw()
            rows = _c29_rows(G, side, grid)
            deficit = 2 * q - gf2_rank(rows)
            if deficit > 2:
                continue
            cols = _c29_columns(rows, width)
            assert all(col.bit_count() == 3 for col in cols)
            short, four, components = _f21_stats(cols, 2 * q)
            if components != 1 or any(short):
                continue
            coarse = (four, deficit)
            pool.append((coarse, trial, grid, rows, cols))
            pool.sort(key=lambda item: (item[0], item[1]))
            del pool[6:]

        finalists = []
        for coarse, trial, grid, rows, cols in pool:
            boundary, six = _f21_traps(cols)
            distance, multiplicity, peak = _c29_sampled_words(rng, cols, q)
            # First warn about witnessed short kernels. Above twelve,
            # prefer less ambiguity rather than a larger upper bound.
            key = (int(distance < 12), six, multiplicity, peak,
                   boundary, coarse[0], coarse[1],
                   -min(distance, 12))
            finalists.append((key, trial, grid, rows, cols))
        finalists.sort(key=lambda item: (item[0], item[1]))
        if finalists:
            return finalists[:3]
    raise ValueError("no connected ratio-contract component passed screening")


def NEW_CTX(rng, G, idx):
    """Construct one fresh canonical ratio contract in rate cell 0.200."""
    supports = [[1, 1, 2, 2], [2, 2, 1, 1]]
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 9 <= W_MAX
    assert all(sum(row) == 6 for row in supports)
    assert all(sum(column) == 3 for column in zip(*supports))

    architecture_a, architecture_b = _c29_architectures(idx)
    for attempt in range(3):
        aa = _c29_select(rng, G, "A", architecture_a)
        bb = _c29_select(rng, G, "B", architecture_b)
        options = []
        for ka, ta, A, ar, ac in aa:
            for kb, tb, B, br, bc in bb:
                joint = tuple(value for a, b in zip(ka, kb)
                              for value in (max(a, b), a + b))
                options.append((joint, ta, tb, A, B, ac, bc))
        options.sort(key=lambda item: item[:3])

        eligible = []
        for key, ta, tb, A, B, ac, bc in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 580 or not 116 <= k <= 128:
                continue
            # Confirm that the screened components occur unchanged in
            # the complete evaluator-convention CSS matrices.
            actual_a, actual_b = _f21_columns(hx, hz, G.q)
            assert actual_a == ac and actual_b == bc
            eligible.append((key, -k, ta, tb, A, B, hx, hz))
        eligible.sort(key=lambda item: item[:4])

        for key, negative_k, ta, tb, A, B, hx, hz in eligible:
            if _c29_low4(hx, hz, 580) or _c29_low4(hz, hx, 580):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 9
            assert 6 * sum(row.bit_count() for row in hx + hz) == 25056
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "architectures": (architecture_a, architecture_b),
                    "k": -negative_k, "screen": key}

    raise ValueError("no complete C29 contract passed target-rate screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened cell without subsequent relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's large q=58 single-row uniform-label seed reliably suppresses collective circuit failures; measured worst-basis s=3.17-3.28, widespread logical flips and many sector-1 weight-four witnesses contradict this.
# REFUTED: Screened distance alone orders measured fitness: current slot11 had d2s=20,k=24,s=6.893 while slot28 had d2s=18,k=30,s=6.989; both had zero failures in both bases and were limited by different sampling ceilings.
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
