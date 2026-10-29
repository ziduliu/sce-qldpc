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
CONCEPT = ("Frobenius C7 semidirect C3 with action 2, order 21, shapes (2,4)x(2,4): degree-three/four components with order-seven binomial differences and screened small boundaries should retain rare circuit failures and exceed fitness 7.177 in the 0.200 rate cell.")
GROUP = ("Keep the exact order-21 group: all five measured parent codes had zero failures in both bases, with fitness 6.9109 limited by their k-dependent sample ceilings. Order 21 still fits the new shapes, giving n=420, so no group or order change is needed. Same-coset doubletons have order-seven relative differences; their circuit advantage is a hypothesis.")
SHAPES = ("Equal (2,4) components each have 84 variables and 42 checks. The size factor is 20, n=420 and guaranteed k>=84, rate 0.200. Verified final ranks keep every retained code below rate 0.225. Parent failure maps have no failed shots to attribute; its minimum-weight witnesses predominantly occupied sector-one Acol/Bcol blocks, motivating degree-at-least-three protection across every component column.")
FILLING = ("Twenty briefs cross four support architectures with five canonical internal-step geometries; extra harness slots extend the geometry book. Positive singleton/doubleton cells give component column weights 3-4 and row weights 6-7, hence both CSS check weights <=11 and six-round CNOT counts 18144-19656. Same-coset order-seven doubletons replace unrestricted labels. Optimize the two verified component projections independently, screen short kernels, and refine small boundaries with sampled-distance rewards capped at twelve. Complete verified lifts enforce the target rate, reject quantum logicals through weight four and check CSS. These surrogates test retention of reliability at higher k and lower extraction load, not distance as fitness.")
FAMILY_NAME = "r020_frobenius21_24x24_lowload_rotationbook_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 21

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius group requires q=21")
    # x^7=y^3=1, y*x*y^-1=x^2; 2^3=1 modulo 7.
    els = [(a, b) for b in range(3) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 3)

    def mul(g, h):
        return ((g[0] + pow(2, g[1], 7) * h[0]) % 7,
                (g[1] + h[1]) % 3)

    def inv(g):
        b = (-g[1]) % 3
        return ((-pow(2, b, 7) * g[0]) % 7, b)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


def _f21_small6(cols):
    """Exact existence of a nonzero component kernel word through weight six."""
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
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                if value in seen:
                    return True
                seen.add(value)
    return False


def _f21_kernel_estimate(rng, cols, trials=48):
    """Sample fundamental circuits and their pairs: an upper bound, not pL."""
    order = list(range(len(cols)))
    best = len(cols) + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        circuits = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                best = min(best, word.bit_count())
                for previous in circuits:
                    best = min(best, (previous ^ word).bit_count())
                circuits.append(word)
        if best <= 6:
            break
    return best


def _f21_quantum_low4(checks, stabilizers, n):
    """Exact stabilizer-aware logical existence test through weight four."""
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

    cols = [0] * n
    for r, row in enumerate(checks):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    bits = [1 << c for c in range(n)]
    seen = {0: 0}
    for c, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[c]
        elif nontrivial(old ^ bits[c]):
            return True
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif nontrivial(old ^ word):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """A deliberate support/ratio brief, followed by fresh voltage optimization."""
    profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((1, 2, 2, 2), (2, 2, 1, 1)),
        ((1, 2, 2, 2), (2, 2, 2, 1)),
    )
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    architecture = (idx // 5) % len(pairs)
    pa, pb = pairs[architecture]
    sa = [list(row) for row in profiles[pa]]
    sb = [list(row) for row in profiles[pb]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    q = G.q
    assert q == 21
    nchecks = 2 * q

    def direction(x):
        x %= 7
        return min(x, 7 - x)

    def geometry_key(grids):
        """Remove obvious global automorphism and component permutation copies.

        This is not a complete Tanner-isomorphism classification.
        The evaluator's graph-isomorphism deduplication remains authoritative.
        """
        images = []
        for u in range(1, 7):
            sides = []
            for counts, steps in zip((sa, sb), grids):
                columns = [
                    (counts[0][j], direction(u * steps[0][j]),
                     counts[1][j], direction(u * steps[1][j]))
                    for j in range(4)
                ]
                normal = tuple(sorted(columns))
                swapped = tuple(sorted(
                    (c, d, a, b) for a, b, c, d in columns))
                sides.append(min(normal, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Five canonical step geometries per support architecture give twenty
    # distinct construction briefs. Extra harness batches use further
    # geometries, not saved-code translations or automorphism images.
    book_rng = random.Random(210024 + 1009 * architecture)
    wanted = 5 * (idx // 20) + idx % 5
    seen_geometries = set()
    geometry = None
    for attempt in range(30000):
        grids = []
        for counts in (sa, sb):
            grids.append([
                [book_rng.randrange(1, 4) if counts[i][j] == 2 else 0
                 for j in range(4)]
                for i in range(2)
            ])
        signature = geometry_key(grids)
        if signature in seen_geometries:
            continue
        seen_geometries.add(signature)
        if len(seen_geometries) == wanted + 1:
            geometry = grids
            break
    if geometry is None:
        raise ValueError("insufficient canonical order-seven geometries")

    def draw(counts, side):
        grid = []
        for i in range(2):
            row = []
            for j in range(4):
                a, b = rng.randrange(7), rng.randrange(3)
                cell = [(a, b)]
                if counts[i][j] == 2:
                    delta = geometry[side][i][j]
                    cell.append(((a + delta) % 7, b))
                row.append(cell)
            grid.append(row)
        return grid

    def cycle_cost(cols):
        short, four, connected = _f21_stats(cols, nchecks)
        if connected != 1:
            return None
        # All column degrees are positive. This excludes component
        # kernels of weights two through four before cycle refinement.
        return short, four

    unit = [[[G.e] for _ in range(4)] for _ in range(2)]

    def project(side, grid):
        # Only verified lifts supply the left/right component conventions.
        if side == 0:
            hx, hz, _ = lifted_matrices(G, grid, unit)
        else:
            hx, hz, _ = lifted_matrices(G, unit, grid)
        return _f21_columns(hx, hz, q)[side]

    pools = [[], []]
    for trial in range(160):
        A, B = draw(sa, 0), draw(sb, 1)
        hx, hz, _ = lifted_matrices(G, A, B)
        ac, bc = _f21_columns(hx, hz, q)
        for side, grid, cols in ((0, A, ac), (1, B, bc)):
            cost = cycle_cost(cols)
            if cost is None:
                continue
            pools[side].append((cost, trial, grid, cols))
            pools[side].sort(key=lambda entry: (entry[0], entry[1]))
            del pools[side][6:]

    if not all(pools):
        raise ValueError("no connected component voltage candidates")

    finalists = []
    for side, pool in enumerate(pools):
        entries = []
        for cost, trial, grid, cols in pool:
            # Change one cell's inter-cell voltage, preserving its term
            # count and prescribed internal difference. This is not a
            # whole-code relabeling.
            for step in range(32):
                proposal = [[list(cell) for cell in row] for row in grid]
                i, j = rng.randrange(2), rng.randrange(4)
                da, db = rng.randrange(7), rng.randrange(3)
                proposal[i][j] = [
                    ((g[0] + da) % 7, (g[1] + db) % 3)
                    for g in proposal[i][j]
                ]
                proposed_cols = project(side, proposal)
                proposed_cost = cycle_cost(proposed_cols)
                if proposed_cost is not None and proposed_cost <= cost:
                    grid, cols, cost = proposal, proposed_cols, proposed_cost

            bad6 = int(_f21_small6(cols))
            distance = _f21_kernel_estimate(rng, cols)
            boundary, repeated = _f21_traps(cols)
            entries.append((cost, bad6, distance, boundary,
                            repeated, trial, grid))
        finalists.append(entries)

    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, abad, ad, ab, ar, at, ag = a
            bc, bbad, bd, bb, br, bt, bg = b
            # A protection floor precedes boundary/load refinement.
            # Sampled distance has no reward above twelve.
            capped_a, capped_b = min(ad, 12), min(bd, 12)
            key = (
                max(ac[0], bc[0]),
                tuple(x + y for x, y in zip(ac[0], bc[0])),
                abad + bbad,
                -min(capped_a, capped_b),
                max(ab, bb), ab + bb,
                max(ar, br), ar + br,
                max(ac[1], bc[1]), ac[1] + bc[1],
                abs(capped_a - capped_b), at, bt,
            )
            options.append((key, ag, bg))
    options.sort(key=lambda entry: entry[0])

    for key, A, B in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Enforce the measured rate band per code, not merely its mean.
        if not (175 * n <= 1000 * k < 225 * n):
            continue
        if _f21_quantum_low4(hx, hz, n):
            continue
        if _f21_quantum_low4(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= 11
        return {"idx": idx, "architecture": architecture,
                "geometry": wanted, "k": k, "screen": key,
                "cells": {"A": A, "B": B}}

    raise ValueError("no screened filling retained the 0.200 rate cell")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, with no late label changes."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's large q=58 single-row uniform-label seed reliably suppresses collective circuit failures; measured worst-basis s=3.17-3.28, widespread logical flips and many sector-1 weight-four witnesses contradict this.
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
