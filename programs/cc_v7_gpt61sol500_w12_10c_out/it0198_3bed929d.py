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
CONCEPT = ("For cyclic Z17 (metacyclic Z17:Z1), order 17, shapes (2,5)x(2,6), a rank-33 cubic A component and a rank-33 even-column B component should yield verified k=213 while retaining five d30s>=8 readings, raising fitness above 23.031; this is a falsifiable measured-score hypothesis, not a distance certificate.")
GROUP = ("Frozen L1 group, kept byte-for-byte. This exact Z17 bed measures five n=578, k=208, d30s=8 leaders. The previous controlled A deficit improved fitness without increasing distance; test a second verified trivial-sector deficit rather than assuming deeper search raises the distance plateau.")
SHAPES = ("Frozen (2,5)x(2,6) for cell 0.350: n=578, guaranteed k>=204 and rate 12/34=0.35294; q=17 is the largest admissible lift. Components have 85 and 102 variables with 34 checks each. Parent minima occupied only sector-1 Acol/Bcol blocks, emphasizing Acol2 for X and Bcol2/3 for Z. Require assembled k=213, rate 0.36851, still inside cell 0.350.")
FILLING = ("Keep A's four (2,1) columns and one (0,3) column. Replace B by (2,0),(0,2),(1,1),(3,1),(1,3),(2,2): three degree-two and three degree-four columns, row totals 9/9, and augmentation rank one. Require both verified component ranks 33 and assembled k=213; both CSS row-weight maxima are twelve. Twenty cards cross five A gap geometries with four relative pure-binomial B gaps. Difference packing and conflict repairs now screen odd B words too, strongly penalize weight seven, and exhaustively screen finalists through eight. Retry a B search whose finalist still has a shorter witness. Five distance-eight readings would score 23.585; component screening is not quantum-distance certification.")
FAMILY_NAME = "r035_m17r1s1_25x26_dualdefect_evenB_minors8_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # metacyclic Z_17 x| Z_1: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 17, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize dual-deficit fillings using the evaluator's component lifts."""
    import math
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = q * (ca - ra) * (cb - rb)
    mask = (1 << q) - 1
    assert q == 17

    from itertools import combinations

    # A retains the successful cubic, even-top-row architecture.
    acounts = ((2, 1),) * 4 + ((0, 3),)
    # B's parity columns are either (0,0) or (1,1), giving one
    # augmentation dependency without an all-zero protograph line.
    # Each degree-two incidence type occurs only once: repeating a
    # pure-row binomial or using three mixed binomials creates short
    # polynomial relations that voltage optimization cannot remove.
    bcounts = ((2, 0), (0, 2), (1, 1),
               (3, 1), (1, 3), (2, 2))
    card = idx % 20
    sa = [[c[i] for c in acounts] for i in range(2)]
    sb = [[c[i] for c in bcounts] for i in range(2)]
    assert tuple(map(sum, sa)) == (8, 7)
    assert tuple(map(sum, sb)) == (9, 9)
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(gf2_rank([(a & 1) | ((b & 1) << 1)
                        for a, b in counts]) == 1
               for counts in (acounts, bcounts))

    # If all nontrivial cyclic factors retain full component rank,
    # the trivial factor has k=(5-1)*(6-1)+(2-1)*(2-1)=21,
    # rather than the rank-count value 12. Full matrices verify this.
    target_k = kb + (ca - ra) + (cb - rb) + 2
    assert target_k == 213

    geometries = set()
    for ds in combinations(range(1, (q + 1) // 2), 4):
        representative = min(
            tuple(sorted(min((u * d) % q, q - (u * d) % q)
                         for d in ds))
            for u in range(1, q))
        geometries.add(representative)

    def gap_triangles(ds):
        return sum(
            any((a + sign * b) % q in (c, q - c)
                for sign in (-1, 1))
            for a, b, c in combinations(ds, 3))

    gap_sets = sorted(geometries, key=lambda ds: (gap_triangles(ds), ds))
    assert len(gap_sets) >= 5
    # Five A gap orbits cross four relative B pure-binomial gaps.
    # These are structural constraints, not common multiplier images.
    geometry = (card // 4 + 5 * (idx // 20)) % len(gap_sets)
    fixed_gaps = {
        "A": {(0, j): gap_sets[geometry][j] for j in range(4)},
        "B": {(0, 0): 1, (1, 1): (2, 3, 4, 5)[card % 4]},
    }

    def rotate(x, t):
        return ((x >> t) | (x << (q - t))) & mask

    # Precompute the unique orientation of every nonconstant 17-bit
    # necklace. This relabels verified syndromes; it does not build a lift.
    shifts = [-1] * (1 << q)
    canonical = [0] * (1 << q)
    shifts[0] = shifts[mask] = 0
    canonical[mask] = mask
    for x in range(1, mask):
        if shifts[x] >= 0:
            continue
        z = min(rotate(x, t) for t in range(q))
        for t in range(q):
            y = rotate(z, t)
            shifts[y] = (-t) % q
            canonical[y] = z

    def normalize_syndrome(s):
        lo, hi = s & mask, s >> q
        t = shifts[lo] if lo not in (0, mask) else shifts[hi]
        return rotate(lo, t) | (rotate(hi, t) << q), t

    def normalize_word(word, width):
        # A nonzero word of weight <=8 cannot contain a constant
        # nonzero 17-bit block, so this orientation is unique.
        top = (word.bit_length() - 1) // q
        t = shifts[(word >> (top * q)) & mask]
        return sum(rotate((word >> (j * q)) & mask, t) << (j * q)
                   for j in range(width))

    def pair_data(x, y):
        local = []
        for vals in (x, y):
            ds = []
            for a in range(len(vals)):
                for b in range(a):
                    d = (vals[a] - vals[b]) % q
                    ds.append(min(d, q - d))
            if len(ds) != len(set(ds)):
                return None
            local.append(set(ds))
        cross = [(a - b) % q for a in x for b in y]
        if len(cross) != len(set(cross)):
            return None
        return local, set(cross)

    def pack(counts, side, old=None, replace=None):
        """Pack a new component or replace selected voltage columns."""
        width = len(counts)
        replace = set(range(width)) if replace is None else set(replace)
        restarts = 40 if old is None else 3
        for _ in range(restarts):
            used = [set(), set()]
            cross_used = set()
            out = [None] * width
            valid = True
            if old is not None:
                for j in range(width):
                    if j in replace:
                        continue
                    x = [g[0] for g in old[0][j]]
                    y = [g[0] for g in old[1][j]]
                    data = pair_data(x, y)
                    if data is None:
                        valid = False
                        break
                    local, cross = data
                    if (local[0] & used[0] or local[1] & used[1]
                            or cross & cross_used):
                        valid = False
                        break
                    used[0].update(local[0])
                    used[1].update(local[1])
                    cross_used.update(cross)
                    out[j] = (x, y)
            if not valid:
                continue
            order = list(replace)
            rng.shuffle(order)
            order.sort(key=lambda j: -max(counts[j]))
            for j in order:
                chosen = None
                for attempt in range(160):
                    vals = [rng.sample(range(q), count)
                            for count in counts[j]]
                    for i in range(2):
                        gap = fixed_gaps[side].get((i, j))
                        if gap is not None:
                            base = rng.randrange(q)
                            vals[i] = [base, (base + gap) % q]
                    anchor = vals[0][0] if vals[0] else vals[1][0]
                    x, y = [sorted((v - anchor) % q for v in row)
                            for row in vals]
                    data = pair_data(x, y)
                    if data is None:
                        continue
                    local, cross = data
                    if (local[0] & used[0] or local[1] & used[1]
                            or cross & cross_used):
                        continue
                    used[0].update(local[0])
                    used[1].update(local[1])
                    cross_used.update(cross)
                    chosen = (x, y)
                    break
                if chosen is None:
                    valid = False
                    break
                out[j] = chosen
            if valid:
                return [[[G.elements[v] for v in out[j][i]]
                         for j in range(width)] for i in range(2)]
        return None

    zero_A = [[[] for _ in range(ca)] for _ in range(ra)]
    zero_B = [[[] for _ in range(cb)] for _ in range(rb)]

    def component_columns(side, cells):
        """Extract the component solely from the verified construction."""
        if side == "A":
            hx, _, _ = lifted_matrices(G, cells, zero_B)
            width = ca
            rows = []
            for i in range(ra):
                for t in range(q):
                    source = hx[(i * cb) * q + t]
                    rows.append(sum(
                        ((source >> (j * cb * q)) & mask) << (j * q)
                        for j in range(ca)))
        else:
            _, hz, _ = lifted_matrices(G, zero_A, cells)
            width = cb
            rows = [r & ((1 << (width * q)) - 1)
                    for r in hz[:rb * q]]
        cols = [0] * (width * q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Explicitly validate the translation used by orbit screening.
        for j in range(width):
            s = cols[j * q]
            for t in range(q):
                expected = (rotate(s & mask, (-t) % q)
                            | (rotate(s >> q, (-t) % q) << q))
                assert cols[j * q + t] == expected
        counts = acounts if side == "A" else bcounts
        assert all(cols[j * q + t].bit_count() == sum(counts[j])
                   for j in range(width) for t in range(q))
        return cols

    def shifted_bits(width):
        return [[1 << (j * q + (t - shift) % q)
                 for j in range(width) for t in range(q)]
                for shift in range(q)]

    def short_six(cols, bits, width):
        """Exhaustive orbit-reduced rejection of component words <=6."""
        seen = {0: 0}

        def insert(s, positions):
            key, shift = normalize_syndrome(s)
            word = 0
            for p in positions:
                word |= bits[shift][p]
            previous = seen.get(key)
            if previous is None:
                seen[key] = word
            elif previous != word:
                difference = previous ^ word
                return difference.bit_count(), difference
            return None

        length = len(cols)
        # Every support orbit has a representative whose first occupied
        # protograph block contains position zero.
        for a in range(0, length, q):
            hit = insert(cols[a], (a,))
            if hit:
                return hit
        for a in range(0, length, q):
            for b in range(a + 1, length):
                hit = insert(cols[a] ^ cols[b], (a, b))
                if hit:
                    return hit
        for a in range(0, length, q):
            for b in range(a + 1, length - 1):
                s = cols[a] ^ cols[b]
                for c in range(b + 1, length):
                    hit = insert(s ^ cols[c], (a, b, c))
                    if hit:
                        return hit
        # A's cubic columns exclude odd kernel weights. B's even
        # columns do not: its passed-through-six flag is only seven.
        return (8 if all(c.bit_count() & 1 for c in cols) else 7), 0

    def eight_words(cols, bits, width, exhaustive=False):
        """Find weight-seven/eight component kernel orbits.

        Triple/quartet collisions detect seven; quartet/quartet
        collisions detect eight. The caller has excluded words <=6.
        Finalist screening removes the fast syndrome-weight cutoff.
        """
        seen = {}
        multiple = {}
        words = set()
        length = len(cols)
        odd_columns = all(c.bit_count() & 1 for c in cols)
        cutoff = 6 if odd_columns else 8

        def insert(s, positions):
            if not exhaustive and s.bit_count() > cutoff:
                return
            key, shift = normalize_syndrome(s)
            row = bits[shift]
            word = 0
            for p in positions:
                word |= row[p]
            previous = seen.get(key)
            if previous is None:
                seen[key] = word
                return
            if previous == word:
                return
            bucket = multiple.get(key)
            if bucket is None:
                bucket = {previous}
                multiple[key] = bucket
            if word in bucket:
                return
            for other in bucket:
                difference = word ^ other
                weight = difference.bit_count()
                assert weight in (7, 8), "short-six screening mismatch"
                words.add(normalize_word(difference, width))
            bucket.add(word)

        # Every support orbit has a representative with position zero
        # in its first occupied block. Rotate syndrome and support
        # together using the explicitly verified component action.
        if not odd_columns:
            for a in range(0, length, q):
                for b in range(a + 1, length - 1):
                    ab = cols[a] ^ cols[b]
                    for c in range(b + 1, length):
                        insert(ab ^ cols[c], (a, b, c))
        for a in range(0, length, q):
            for b in range(a + 1, length - 2):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, length - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, length):
                        insert(abc ^ cols[d], (a, b, c, d))
        return words

    def optimize(side, counts, budget):
        width = len(counts)
        bits = shifted_bits(width)
        current = None
        current_cost = None
        current_words = ()
        finalists = []
        signatures = set()

        def evaluate(grid):
            cols = component_columns(side, grid)
            # Both components must have precisely one dependency.
            # Additional nontrivial-factor deficits are rejected.
            expected_rank = 2 * q - 1
            if gf2_rank(cols) != expected_rank:
                return None
            floor, witness = short_six(cols, bits, width)
            if floor < 7:
                return (1000000 + 100000 * (7 - floor),
                        (witness,), cols, floor)
            words = eight_words(cols, bits, width)
            seven = sum(word.bit_count() == 7 for word in words)
            eight = len(words) - seven
            cost = 100000 * int(seven > 0) + 1000 * seven + eight
            return cost, tuple(words), cols, floor

        def retain(grid, value):
            signature = tuple(tuple(sorted(g[0] for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return
            signatures.add(signature)
            finalists.append((value[0], signature, grid, value))
            finalists.sort(key=lambda item: (item[0], item[1]))
            del finalists[4:]

        for step in range(budget):
            if current is None or step % 17 == 0:
                trial = pack(counts, side)
            else:
                if current_words:
                    witness = rng.choice(current_words)
                    implicated = [
                        j for j in range(width)
                        if (witness >> (j * q)) & mask]
                    j = rng.choice(implicated)
                else:
                    j = rng.randrange(width)
                replacement = {j}
                if step % 7 == 0:
                    replacement.add(rng.randrange(width))
                trial = pack(counts, side, current, replacement)
            if trial is None:
                continue
            value = evaluate(trial)
            if value is None:
                continue
            retain(trial, value)
            cost, words, _, _ = value
            temperature = 2.0 * (1.0 - step / float(budget)) + 0.12
            if (current is None or cost <= current_cost
                    or rng.random() < math.exp(
                        max(-700.0, (current_cost - cost) / temperature))):
                current, current_cost, current_words = trial, cost, words

        if not finalists:
            # Rank failure is not repaired by changing the support card.
            for _ in range(80):
                trial = pack(counts, side)
                if trial is None:
                    continue
                value = evaluate(trial)
                if value is not None:
                    retain(trial, value)
                    break
        if not finalists:
            raise ValueError("no full-rank voltage assignment for support card")

        winner = None
        for _, _, grid, value in finalists:
            _, witnesses, cols, floor = value
            if floor < 7:
                key = (floor, -len(witnesses), -len(witnesses))
                exact_words = witnesses
            else:
                exact_words = eight_words(
                    cols, bits, width, exhaustive=True)
                default_floor = 10 if side == "A" else 9
                floor = min((word.bit_count() for word in exact_words),
                            default=default_floor)
                minimum_load = sum(word.bit_count() == floor
                                   for word in exact_words)
                # These floors concern only component kernels.
                # They do not certify the quantum LP distance.
                key = (floor, -minimum_load, -len(exact_words))
            if winner is None or key > winner[0]:
                winner = (key, grid, exact_words)
        return winner[1], winner[0]

    # B's new even-column architecture receives the deeper search.
    A, pa = optimize("A", acounts, 80 + 8 * (idx % 3))
    B, pb = optimize("B", bcounts, 128 + 8 * (idx % 3))
    if pb[0] < 8:
        retry_B, retry_pb = optimize("B", bcounts, 192)
        if retry_pb > pb:
            B, pb = retry_B, retry_pb
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    # Final acceptance uses the evaluator's exact matrices and ranks;
    # the augmentation argument alone is not an acceptance criterion.
    assert k == target_k
    assert 325 * n <= 1000 * k < 375 * n
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return {"idx": idx, "cells": {"A": A, "B": B},
            "component_screen": (pa, pb), "gap_geometry": geometry,
            "k": k, "rank_surplus": k - kb}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete dual-deficit screened cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Difference packing and component screening only through weight six do not remove the weight-eight plateau on this bed: all five measured current-parent codes have d30s=8, with minimum witnesses exclusively in sector 1.
# REFUTED: Conflict-directed orbit-eight optimization raises the five measured readings above eight on the full-rank cubic Z17 bed: all five measured v2 leaders instead retained k=204 and d30s=8, leaving fitness at 22.588.
# REFUTED: The mixed234_orbit78 filling improves fitness over 23.031 on this bed: its measured fitness was 17.5652, with weight-seven X witnesses among its leaders; merely adding quartic A columns did not preserve the distance-eight plateau.
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
