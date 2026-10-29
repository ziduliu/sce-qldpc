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
CONCEPT = ("Semidihedral SD16, order 16, shapes (2,5)x(2,6): independently screened components with even-augmentation rows on both sides should supply five verified k=201 codes retaining d30s>=8 and exceed fitness 23.5290 in cell 0.350. This is a falsifiable measured-score hypothesis, not a distance certificate.")
GROUP = ("L3 jump from Q16 to SD16: x^8=y^2=1 and y*x*y^-1=x^3. SD16 has noncentral involutions, unlike Q16's unique involution, so this is not a relabeling. No SD16 bed is listed. The parent's clean-six Q16 program reached five [544,200,8] readings; test transfer of its successful exclusion strategy with a different multiplication law and dependencies on both components.")
SHAPES = ("Keep (2,5)x(2,6) for cell 0.350: n=544, kb=192, guaranteed rate 12/34 and q_max=17; SD16 fixes q=16. Components have 80/32 and 96/32 variables/checks. Parent measured minima were exclusively sector-one Acol/Bcol witnesses across both components, not Arow/Brow witnesses. Verified k=201 gives rate 0.369485 and score 23.647059 at distance eight; gate actual assembled rates in [0.325,0.375).")
FILLING = ("Twenty cards cross five fresh A relative geometries with three cubic B profiles and one even-row B policy. The latter uses either four (2,1), one (0,3), one (0,2) columns, or three (2,1), two (0,3), one (2,0); its isolated binomial has relative order eight. A rows total 8/7; B rows total 9/9 or 8/9, giving HX<=11 and HZ<=12. Require measured component ranks 31/32, 30/32, or 31/31 as prescribed, then compute full k without assuming a surplus formula. Exclude component kernels through six before shortlisting and through seven on even-degree finalists. Independently probe actual full-code logicals, verify CSS and ranks, and reject equal uncoloured Tanner invariants. All found-witness readings remain upper bounds.")
FAMILY_NAME = "r035_SD16_25x26_dual_evenrows_clean7_rank201_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # SD16: (a,b)=x^a*y^b, x^8=y^2=1, y*x*y^-1=x^3.
    if not valid_q(q):
        raise ValueError("SD16 requires order 16")
    els = [(a,b) for b in range(2) for a in range(8)]

    def red(g):
        return (g[0] % 8, g[1] % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + (1 if b == 0 else 3)*c) % 8,
                (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-(1 if b == 0 else 3)*a) % 8, b)

    def idx(g):
        a,b = red(g)
        return 8*b+a

    return dict(elements=els, ident=(0,0), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _retired_z17_context(rng, G, idx):
    """Historical cyclic filling; the Q16 builder never calls this."""
    import math
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = q * (ca - ra) * (cb - rb)
    mask = (1 << q) - 1
    assert q == 17

    from itertools import combinations

    # All component columns remain cubic. The sum of A's seventeen
    # top-row checks is zero because every top cell has even weight.
    # Keep exactly one bottom-only triple: two such columns would have
    # the unavoidable commuting-polynomial kernel word of weight six.
    acounts = ((2, 1),) * 4 + ((0, 3),)
    bprofiles = (
        ((2, 1),) * 3 + ((1, 2),) * 3,
        ((3, 0), (2, 1)) + ((1, 2),) * 4,
        ((3, 0), (0, 3)) + ((2, 1),) * 2 + ((1, 2),) * 2,
    )
    card = idx % 20
    bcounts = bprofiles[card % 3]
    sa = [[c[i] for c in acounts] for i in range(2)]
    sb = [[c[i] for c in bcounts] for i in range(2)]
    target_k = kb + (cb - rb)
    assert target_k == 208
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Canonical four-gap sets under every automorphism x -> u*x of
    # Z17, including unsigned binomial orientation and column sorting.
    # Different representatives are not multiplier/permutation copies.
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
    assert len(gap_sets) >= 7
    # The first twenty cards use seven distinct gap orbits crossed
    # with three B incidence profiles, omitting the twenty-first pair.
    # Extra slots independently optimize further orbit/profile cards.
    geometry = (card // 3 + 7 * (idx // 20)) % len(gap_sets)
    fixed_row = 0
    fixed_gaps = {j: gap_sets[geometry][j] for j in range(4)}

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
                    if side == "A" and j in fixed_gaps:
                        base = rng.randrange(q)
                        vals[fixed_row] = [
                            base, (base + fixed_gaps[j]) % q]
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
        assert all(c.bit_count() == 3 for c in cols)
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
        # Odd column degree excludes odd-weight kernel words.
        return 8, 0

    def eight_words(cols, bits, width, exhaustive=False):
        """Find weight-eight kernel orbits by quartet-syndrome collisions.

        The fast objective retains syndromes of weight <=6 only.
        exhaustive=True removes that restriction and covers every
        weight-eight word, provided short_six has already passed.
        """
        seen = {}
        multiple = {}
        words = set()
        length = len(cols)
        for a in range(0, length, q):
            for b in range(a + 1, length - 2):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, length - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, length):
                        s = abc ^ cols[d]
                        if not exhaustive and s.bit_count() > 6:
                            continue
                        key, shift = normalize_syndrome(s)
                        row = bits[shift]
                        word = row[a] | row[b] | row[c] | row[d]
                        previous = seen.get(key)
                        if previous is None:
                            seen[key] = word
                            continue
                        if previous == word:
                            continue
                        bucket = multiple.get(key)
                        if bucket is None:
                            bucket = {previous}
                            multiple[key] = bucket
                        if word in bucket:
                            continue
                        for other in bucket:
                            difference = word ^ other
                            weight = difference.bit_count()
                            assert weight == 8, "short-six screening mismatch"
                            words.add(normalize_word(difference, width))
                        bucket.add(word)
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
            # Accept exactly the intended trivial-sector dependency,
            # not an uncontrolled deficit in a nontrivial cyclic factor.
            expected_rank = 2 * q - int(side == "A")
            if gf2_rank(cols) != expected_rank:
                return None
            floor, witness = short_six(cols, bits, width)
            if floor < 8:
                return (10000 + 1000 * (8 - floor),
                        (witness,), cols, floor)
            words = eight_words(cols, bits, width)
            return len(words), tuple(words), cols, 8

        def retain(grid, value):
            signature = tuple(tuple(sorted(g[0] for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return
            signatures.add(signature)
            finalists.append((value[0], signature, grid, value))
            finalists.sort(key=lambda item: (item[0], item[1]))
            del finalists[3:]

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
            if floor < 8:
                key = (floor, -len(witnesses))
                exact_words = witnesses
            else:
                exact_words = eight_words(
                    cols, bits, width, exhaustive=True)
                # No words <=8, plus odd-degree parity, gives a COMPONENT
                # floor of ten. This does not certify LP quantum distance.
                key = (8 if exact_words else 10, -len(exact_words))
            if winner is None or key > winner[0]:
                winner = (key, grid, exact_words)
        return winner[1], winner[0]

    # The longer B component receives the larger conflict-repair budget.
    A, pa = optimize("A", acounts, 64 + 8 * (idx % 3))
    B, pb = optimize("B", bcounts, 96 + 8 * (idx % 3))
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


from functools import lru_cache


_Q16_A_DIFFERENCES = (
    ((1,0), (3,0), (0,1), (1,1)),
    ((1,0), (2,0), (0,1), (1,1)),
    ((1,0), (0,1), (1,1), (2,1)),
    ((1,0), (0,1), (2,1), (4,1)),
    ((1,0), (2,0), (3,0), (0,1)),
)
_Q16_B_COUNTS = (
    ((2,1),)*3 + ((1,2),)*3,
    ((3,0), (2,1)) + ((1,2),)*4,
    ((3,0), (0,3)) + ((2,1),)*2 + ((1,2),)*2,
    ((2,1),)*4 + ((0,3), (0,2)),
)


def _q16_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _q16_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _q16_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _q16_four(cols, stabilizers=()):
    """Exact existence screening through four, modulo supplied stabilizers.

    Five is a no-witness sentinel, not a distance reading. The collision
    count is a selection penalty, not a census of distinct supports.
    """
    basis = _q16_basis(stabilizers)
    seen = {0: 0}
    best, hits = 5, 0

    def insert(s, word):
        nonlocal best, hits
        old = seen.get(s)
        if old is None:
            seen[s] = word
        else:
            v = old ^ word
            if v and _q16_outside(v, basis):
                best = min(best, v.bit_count())
                hits += 1

    bits = [1 << i for i in range(len(cols))]
    for i, s in enumerate(cols):
        insert(s, bits[i])
    for i, s in enumerate(cols):
        for j in range(i):
            insert(s ^ cols[j], bits[i] | bits[j])
    return best, hits


def _q16_six(cols):
    """Exact exclusion through six and six-word count after exclusion five."""
    flag, hits = _q16_four(cols)
    if flag < 5:
        return flag, hits

    low = set(cols)
    for i, s in enumerate(cols):
        for j in range(i):
            low.add(s ^ cols[j])
    low.add(0)
    triples, collisions = {}, 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                s = pair ^ cols[h]
                if s in low:
                    # We already excluded every weight <=4.
                    return 5, 1
                old = triples.get(s, 0)
                collisions += old
                triples[s] = old + 1
    assert collisions % 10 == 0
    return (6, collisions//10) if collisions else (7, 0)


def _q16_cycles(cols):
    seen, total = {}, 0
    for s in cols:
        ids = []
        while s:
            bit = s & -s
            ids.append(bit.bit_length()-1)
            s ^= bit
        for i, a in enumerate(ids):
            for b in ids[:i]:
                key = (b,a)
                old = seen.get(key, 0)
                total += old
                seen[key] = old + 1
    return total


def _q16_component(G, side, grid):
    """Extract the actual action using a verified zero companion.

    A zero 1x1 companion removes the other sector from these rows.
    No left/right regular-representation matrix is implemented here.
    """
    width = len(grid[0])*G.q
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, [[[]]])
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, [[[]]], grid)
        rows = hz
    mask = (1 << width)-1
    return _q16_columns([row & mask for row in rows], width)


def _q16_norm(cols, G):
    """Columns for disjoint central-involution pairs.

    A restricted coefficient word of weight t expands to a genuine
    component kernel word of physical weight 2*t.
    """
    q = G.q
    z = (4,0)
    remaining = set(range(q))
    pairs = []
    while remaining:
        a = min(remaining)
        b = G.idx(G.mul(G.elements[a], z))
        assert a != b and b in remaining
        pairs.append((a,b))
        remaining.remove(a)
        remaining.remove(b)
    return [cols[block*q+a] ^ cols[block*q+b]
            for block in range(len(cols)//q) for a,b in pairs]


def _q16_probe(cols, rng, trials, stabilizers=()):
    """Weights of actual found kernel/logical witnesses; upper bounds only."""
    basis = _q16_basis(stabilizers)
    order = list(range(len(cols)))
    best = len(cols)+1

    def consider(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if not _q16_outside(word, basis):
            return
        # Adding stabilizers preserves the nonzero logical class.
        if stabilizers:
            for _ in range(2):
                changed = False
                for row in stabilizers:
                    v = word ^ row
                    if v.bit_count() < word.bit_count():
                        word = v
                        changed = True
                if not changed:
                    break
        best = min(best, word.bit_count())

    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            s, word = cols[c], 1 << c
            while s:
                p = s.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s,word)
                    break
                s ^= old[0]
                word ^= old[1]
            if not s:
                dependencies.append(word)
                consider(word)
        words = sorted(dependencies, key=int.bit_count)[:24]
        for _ in range(2):
            for i, word in enumerate(words):
                for j, other in enumerate(words):
                    if i == j:
                        continue
                    v = word ^ other
                    if v and v.bit_count() < word.bit_count():
                        word = v
                words[i] = word
                consider(word)
        for i, word in enumerate(words):
            for other in words[:i]:
                consider(word ^ other)
        small = words[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    consider(pair ^ small[h])
        if best <= 4:
            break
    return best


def _q16_cell(rng, G, count, long_pair=False):
    """Distinct labels, with no internal central-involution difference."""
    for _ in range(128):
        cell = rng.sample(G.elements, count)
        if any(G.mul(G.inv(g), h) == (4,0)
               for i,g in enumerate(cell) for h in cell[:i]):
            continue
        if long_pair and count == 2:
            d = G.mul(G.inv(cell[0]), cell[1])
            # A rotation with odd exponent has order eight.
            if d[1] or not (d[0] & 1):
                continue
        return cell
    raise ValueError("Q16 cell constraints exhausted")


def _q16_draw(rng, G, side, architecture, counts):
    if side == "A":
        differences = _Q16_A_DIFFERENCES[architecture]
        # Anchoring these cells is a gauge choice within a fresh design,
        # not transporting a previously emitted labeling.
        return [
            [[G.e,d] for d in differences] + [[]],
            [[G.pick(rng)] for _ in differences]
                + [_q16_cell(rng, G, 3)],
        ]
    return [
        [_q16_cell(rng, G, c,
                   long_pair=(i == 1 and j == 5 and counts[0][5] == 0))
         for j,c in enumerate(row)]
        for i,row in enumerate(counts)
    ]


def _q16_pool(rng, G, side, architecture, counts, expected_rank):
    """Independent component optimization for this card's prescribed policy."""
    pool, seen = [], set()
    quota = 176 if side == "A" else 224
    for attempt in range(quota):
        grid = _q16_draw(rng, G, side, architecture, counts)
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        cols = _q16_component(G, side, grid)
        if gf2_rank(cols) != expected_rank:
            continue
        ordinary, hits = _q16_four(cols)
        if ordinary != 5:
            continue
        norm = _q16_norm(cols, G)
        nf, nh = _q16_four(norm)
        cycles = _q16_cycles(cols)
        key = (nf, -nh, -cycles)
        pool.append((key, attempt, grid, cols, norm))
        pool.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        del pool[8:]

    ranked = []
    for key, attempt, grid, cols, norm in pool:
        floor, six = _q16_six(cols)
        if floor <= 5:
            continue
        reading = _q16_probe(cols, rng, 48)
        nf, nh = _q16_four(norm)
        if nf <= 4:
            reading = min(reading, 2*nf)
        # A fresh restricted search can only lower the earlier reading.
        restricted = _q16_probe(norm, rng, 24)
        if restricted <= len(norm):
            reading = min(reading, 2*restricted)
        merit = (reading, floor, -six, nf, -nh, key[2], -attempt)
        ranked.append((merit, grid))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:4]


def _q16_fingerprint(hx, hz, n, k):
    """Uncoloured Tanner invariants, including weighted closed walks.

    Different fingerprints certify nonisomorphism. Equal fingerprints
    are discarded conservatively; they do not prove isomorphism.
    """
    rows = hx+hz
    adjacency = [{} for _ in rows]
    hist = {}
    for i,row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            hist[overlap] = hist.get(overlap, 0)+1
            if overlap:
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap
    triangles, fourth = 0, 0
    for i, neighbors in enumerate(adjacency):
        walks = {}
        for j, ij in neighbors.items():
            for h, jh in adjacency[j].items():
                walks[h] = walks.get(h, 0)+ij*jh
        triangles += sum(v*walks.get(j, 0)
                         for j,v in neighbors.items())
        fourth += sum(v*v for v in walks.values())
    cols = _q16_columns(rows, n)
    return (n, k, tuple(sorted(r.bit_count() for r in rows)),
            tuple(sorted(c.bit_count() for c in cols)),
            tuple(sorted(hist.items())), triangles, fourth)


@lru_cache(maxsize=1)
def _q16_bank(G):
    """Deterministic structural cards, screened before component shortlisting."""
    assert G.q == 16
    q = G.q
    sa = [[2,2,2,2,0], [1,1,1,1,3]]

    # Inverse-pair geometries absent from the parent's principal cards.
    # The last geometry generates <x^2,y>, an index-two Q8 subgroup.
    differences = (
        ((1,0), (3,0), (0,1), (2,1)),
        ((1,0), (2,0), (0,1), (2,1)),
        ((2,0), (0,1), (1,1), (2,1)),
        ((0,1), (1,1), (2,1), (3,1)),
        ((2,0), (0,1), (2,1), (0,1)),
    )
    bprofiles = (
        ((2,1),)*3 + ((1,2),)*3,
        ((3,0), (2,1)) + ((1,2),)*4,
        ((3,0), (0,3)) + ((2,1),)*2 + ((1,2),)*2,
        ((3,0), (3,0), (0,3), (0,3), (2,1), (1,2)),
    )

    def draw(rng, side, architecture, counts, batch):
        if side == "A":
            ds = differences[architecture]
            if architecture == 4 and batch % 2:
                # A genuinely different repeated inverse-pair geometry,
                # not an automorphism image of the repeated outer pair.
                ds = ((2,0), (2,0), (0,1), (2,1))
            return [
                [[G.e,d] for d in ds] + [[]],
                [[G.pick(rng)] for _ in ds]
                    + [_q16_cell(rng, G, 3)],
            ]
        return [[_q16_cell(rng, G, count) for count in row]
                for row in counts]

    def clean_six(cols):
        """Bounded exact constituent decoding after exclusion through four.

        All columns are cubic. A complete singleton/pair table terminates
        branching on an unsatisfied check. There is no node-budget cutoff.
        Each component regular action is transitive inside a variable
        block, so coordinate-zero anchors cover all constituent word
        orbits. No full-code transitivity is assumed.
        """
        assert all(s.bit_count() == 3 for s in cols)
        bits = [1 << j for j in range(len(cols))]
        tails = {0: 0}
        incidence = [[] for _ in range(2*q)]
        for j, syndrome in enumerate(cols):
            assert syndrome not in tails
            tails[syndrome] = bits[j]
            value = syndrome
            while value:
                bit = value & -value
                incidence[bit.bit_length()-1].append(j)
                value ^= bit
        for j, syndrome in enumerate(cols):
            for h in range(j):
                value = syndrome ^ cols[h]
                assert value not in tails
                tails[value] = bits[j] | bits[h]

        def decode(target, forbidden):
            def visit(value, remaining, word):
                if not value:
                    return word
                if value.bit_count() > 3*remaining:
                    return None
                if remaining <= 2:
                    tail = tails.get(value)
                    if (tail is not None
                            and tail.bit_count() <= remaining
                            and not (tail & (word | forbidden))):
                        return word | tail
                    return None
                check = (value & -value).bit_length()-1
                choices = sorted(
                    incidence[check],
                    key=lambda j: -(cols[j] & value).bit_count())
                for j in choices:
                    if bits[j] & (word | forbidden):
                        continue
                    hit = visit(value ^ cols[j], remaining-1,
                                word | bits[j])
                    if hit is not None:
                        return hit
                return None
            return visit(target, 5, 0)

        for anchor in range(0, len(cols), q):
            tail = decode(cols[anchor], bits[anchor])
            if tail is not None:
                witness = tail | bits[anchor]
                syndrome = 0
                value = witness
                while value:
                    bit = value & -value
                    syndrome ^= cols[bit.bit_length()-1]
                    value ^= bit
                assert syndrome == 0 and witness.bit_count() <= 6
                return False
        return True

    def pool(rng, side, architecture, counts, expected_rank,
             retry, batch):
        # Crucially, ordinary short words are screened on EVERY proposal,
        # not only on a central/cycle-selected shortlist.
        quota = (1536 if side == "A" and architecture == 4 else 768)
        quota += retry*(quota//2)
        retained, seen = [], set()
        admitted = 0
        for attempt in range(quota):
            grid = draw(rng, side, architecture, counts, batch)
            signature = tuple(tuple(sorted(cell))
                              for row in grid for cell in row)
            if signature in seen:
                continue
            seen.add(signature)
            cols = _q16_component(G, side, grid)
            if gf2_rank(cols) != expected_rank:
                continue
            if _q16_four(cols)[0] != 5:
                continue
            if not clean_six(cols):
                continue

            norm = _q16_norm(cols, G)
            nf, nh = _q16_four(norm)
            # A central-invariant witness below physical weight eight
            # would contradict the ordinary screening just performed.
            assert nf >= 4
            cycles = _q16_cycles(cols)
            key = (nf, -nh, -cycles)
            retained.append((key, attempt, grid, cols, norm))
            retained.sort(
                key=lambda item: (item[0], -item[1]), reverse=True)
            del retained[8:]
            admitted += 1
            if admitted >= 12:
                break

        ranked = []
        for key, attempt, grid, cols, norm in retained:
            # Independent, non-orbit triple enumeration confirms the
            # decoder's exclusion on every retained component.
            floor, hits = _q16_six(cols)
            if floor != 7:
                continue
            reading = _q16_probe(cols, rng, 64)
            nf, nh = _q16_four(norm)
            if nf == 4:
                reading = min(reading, 8)
            restricted = _q16_probe(norm, rng, 32)
            if restricted <= len(norm):
                reading = min(reading, 2*restricted)
            assert reading >= 8
            merit = (reading, 8, key[2], nf, -nh, -attempt)
            ranked.append((merit, grid))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:4]

    bank, fingerprints = [], set()
    for slot in range(N_SLOTS):
        architecture, profile = divmod(slot % 20, 4)
        batch = slot//20
        bc = bprofiles[profile]
        sb = [[c[i] for c in bc] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(20261119*100003 + slot*104729)
        chosen = None

        for retry in range(2):
            ap = pool(rng, "A", architecture, sa,
                      30 if architecture == 4 else 31, retry, batch)
            bp = pool(rng, "B", architecture, sb, 32, retry, batch)
            candidates = []
            for am,A in ap:
                for bm,B in bp:
                    hx,hz,n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (325*n <= 1000*k < 375*n):
                        continue
                    d = min(am[0], bm[0])
                    score = (k*d*d, d, am[1]+bm[1],
                             am[2]+bm[2])
                    candidates.append((score,A,B,hx,hz,n,k,d))
            candidates.sort(key=lambda item: item[0], reverse=True)

            finalists = []
            for candidate in candidates[:6]:
                _,A,B,hx,hz,n,k,d = candidate
                xc = _q16_columns(hx, n)
                zc = _q16_columns(hz, n)
                if _q16_four(xc, hz)[0] != 5:
                    continue
                if _q16_four(zc, hx)[0] != 5:
                    continue
                dx = _q16_probe(zc, rng, 24, hx)
                dz = _q16_probe(xc, rng, 24, hz)
                limiting = min(d,dx,dz)
                score = (k*limiting*limiting, limiting,
                         min(dx,dz), -abs(dx-dz))
                finalists.append((score,candidate,dx,dz))

            # Equal fresh full-code budgets; never forget an earlier
            # lighter logical witness when refining a candidate.
            refined = []
            for _,candidate,dx,dz in finalists:
                _,A,B,hx,hz,n,k,d = candidate
                dx = min(dx, _q16_probe(
                    _q16_columns(hz, n), rng, 48, hx))
                dz = min(dz, _q16_probe(
                    _q16_columns(hx, n), rng, 48, hz))
                limiting = min(d,dx,dz)
                score = (k*limiting*limiting, limiting,
                         min(dx,dz), -abs(dx-dz))
                refined.append((score,candidate))
            refined.sort(key=lambda item: item[0], reverse=True)

            for score,candidate in refined:
                _,A,B,hx,hz,n,k,d = candidate
                fingerprint = _q16_fingerprint(hx,hz,n,k)
                if fingerprint in fingerprints:
                    continue
                assert n == 544 and k > 0
                assert css_ok(hx,hz)
                assert max(r.bit_count() for r in hx+hz) <= W_MAX
                assert all(len(cell) == len(set(cell))
                           for grid in (A,B) for row in grid for cell in row)
                fingerprints.add(fingerprint)
                chosen = (A,B,k,score)
                break
            if chosen is not None:
                break
        # An unsuccessful structural hypothesis voids its own slot only.
        bank.append(chosen)
    return tuple(bank)


# Historical Q16 and Z17 builders above are inactive. Only the SD16 bank
# below supplies NEW_CTX. Generic syndrome/basis/probe helpers are reused.
_SD_A_GEOMETRIES = (
    ((1,0), (3,0), (1,1), (3,1)),
    ((1,0), (2,0), (1,1), (5,1)),
    ((1,0), (1,1), (3,1), (5,1)),
    ((3,0), (1,1), (5,1), (7,1)),
    ((1,0), (2,0), (1,1), (1,1)),
)
_SD_CUBIC_B = (
    ((2,1),)*3 + ((1,2),)*3,
    ((3,0), (2,1)) + ((1,2),)*4,
    ((3,0), (0,3)) + ((2,1),)*2 + ((1,2),)*2,
)


def _sd_cell(local, G, count):
    """Fresh distinct terms; exclude only the central involution ratio.

    Noncentral involution ratios are allowed in trinomials. Excluding
    every involution ratio would unnecessarily confine these cells to
    smaller subgroups of SD16.
    """
    if count <= 1:
        return [] if count == 0 else [G.pick(local)]
    for _ in range(192):
        cell = local.sample(G.elements, count)
        if any(G.mul(G.inv(g), h) == (4,0)
               for i,g in enumerate(cell) for h in cell[:i]):
            continue
        if count >= 3:
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if not any(G.mul(g,h) != G.mul(h,g)
                       for i,g in enumerate(relatives)
                       for h in relatives[:i]):
                continue
        return cell
    raise ValueError("SD16 distinct noncommuting cell recipe exhausted")


def _sd_counts(architecture, profile):
    if profile < 3:
        return _SD_CUBIC_B[profile]
    if architecture < 2:
        return ((2,1),)*4 + ((0,3), (0,2))
    # Move the single even-degree column to the even row instead.
    # Its pure-row order-eight binomial supplies a real weight-eight
    # component witness, rather than an optimistic no-witness sentinel.
    return ((2,1),)*3 + ((0,3),)*2 + ((2,0),)


def _sd_draw(local, G, side, architecture, profile, counts):
    if side == "A":
        differences = _SD_A_GEOMETRIES[architecture]
        if architecture == 4 and profile < 3:
            # Generate <x^2,xy>, an index-two Q8 subgroup of SD16.
            differences = ((2,0), (1,1), (3,1), (1,1))
        return [
            [[G.e,d] for d in differences] + [[]],
            [[G.pick(local)] for _ in differences]
                + [_sd_cell(local, G, 3)],
        ]

    if profile < 3:
        return [[_sd_cell(local, G, count) for count in row]
                for row in counts]

    # These relative differences generate SD16, not merely its rotations.
    # Independent anchors and bottom labels change actual voltages.
    directions = (
        ((1,0), (3,0), (1,1), (3,1)),
        ((1,0), (2,0), (1,1), (5,1)),
        ((1,0), (1,1), (3,1), (3,0)),
        ((3,0), (1,1), (5,1), (1,0)),
        ((1,0), (2,0), (7,1), (3,0)),
    )[architecture]
    out = [[], []]
    active = 0
    for j, (top, bottom) in enumerate(zip(*counts)):
        if top == 2:
            anchor = G.pick(local)
            if bottom == 0:
                difference = (local.choice((1,3,5,7)), 0)
            else:
                difference = directions[active]
                active += 1
            out[0].append([anchor, G.mul(anchor, difference)])
        else:
            out[0].append([])
        if bottom == 2:
            anchor = G.pick(local)
            difference = (local.choice((1,3,5,7)), 0)
            out[1].append([anchor, G.mul(anchor, difference)])
        else:
            out[1].append(_sd_cell(local, G, bottom))
    return out


def _sd_clean(cols, q, tail_limit):
    """Exact anchored component exclusion using verified syndromes.

    The caller has already excluded kernels through four, so every
    singleton/pair syndrome has a unique support. Branching has no
    node-budget cutoff. Transitivity is used only within each component
    variable block, never asserted for the assembled quantum code.
    """
    bits = [1 << j for j in range(len(cols))]
    tails = {0: 0}
    incidence = [[] for _ in range(2*q)]
    maximum = max(col.bit_count() for col in cols)
    for j, syndrome in enumerate(cols):
        assert syndrome not in tails
        tails[syndrome] = bits[j]
        value = syndrome
        while value:
            bit = value & -value
            incidence[bit.bit_length()-1].append(j)
            value ^= bit
    for j, syndrome in enumerate(cols):
        for h in range(j):
            value = syndrome ^ cols[h]
            assert value not in tails
            tails[value] = bits[j] | bits[h]

    def decode(target, forbidden):
        def visit(value, remaining, word):
            if not value:
                return word
            if value.bit_count() > maximum*remaining:
                return None
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not (tail & (word | forbidden))):
                    return word | tail
                return None
            check = (value & -value).bit_length()-1
            choices = sorted(
                incidence[check],
                key=lambda j: -(cols[j] & value).bit_count())
            for j in choices:
                if bits[j] & (word | forbidden):
                    continue
                found = visit(value ^ cols[j], remaining-1,
                              word | bits[j])
                if found is not None:
                    return found
            return None
        return visit(target, tail_limit, 0)

    for anchor in range(0, len(cols), q):
        tail = decode(cols[anchor], bits[anchor])
        if tail is not None:
            word = tail | bits[anchor]
            syndrome, value = 0, word
            while value:
                bit = value & -value
                syndrome ^= cols[bit.bit_length()-1]
                value ^= bit
            assert syndrome == 0
            assert word.bit_count() <= tail_limit+1
            return False
    return True


def _sd_pool(local, G, side, architecture, profile, counts, rank, retry):
    """Screen before shortlisting, then independently deepen finalists."""
    retained, seen = [], set()
    quota = (1024 if profile == 3 and side == "B" else 640)
    quota += retry*(quota//2)
    admitted = 0
    for attempt in range(quota):
        grid = _sd_draw(local, G, side, architecture, profile, counts)
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        # This helper extracts the action through lifted_matrices with a
        # zero companion; its name is historical, not a Q16 group law.
        cols = _q16_component(G, side, grid)
        if gf2_rank(cols) != rank:
            continue
        if _q16_four(cols)[0] != 5:
            continue
        if not _sd_clean(cols, G.q, 5):
            continue
        cycles = _q16_cycles(cols)
        reading = _q16_probe(cols, local, 4)
        retained.append((reading, -cycles, -attempt, grid, cols))
        retained.sort(key=lambda x: x[:3], reverse=True)
        del retained[8:]
        admitted += 1
        if admitted >= 20:
            break

    finalists = []
    for reading, negative_cycles, attempt, grid, cols in retained:
        # A separate triple enumeration confirms exclusion through six.
        if _q16_six(cols)[0] != 7:
            continue
        if any(col.bit_count() % 2 == 0 for col in cols):
            if not _sd_clean(cols, G.q, 6):
                continue
        reading = min(reading, _q16_probe(cols, local, 32))
        assert reading >= 8
        finalists.append(dict(
            grid=grid, cols=cols, d=reading,
            cycles=-negative_cycles, attempt=-attempt))
    finalists.sort(key=lambda item: (
        min(item["d"], 10), -item["cycles"], -item["attempt"]),
        reverse=True)
    finalists = finalists[:4]
    for item in finalists:
        item["d"] = min(item["d"],
                        _q16_probe(item["cols"], local, 64))
    finalists.sort(key=lambda item: (
        min(item["d"], 10), -item["cycles"], -item["attempt"]),
        reverse=True)
    return finalists[:3]


def _sd_signature(hx, hz, n):
    """Reuse the uncoloured overlap invariants, deliberately omitting k.

    k depends on the X/Z partition and is not used as an uncoloured
    Tanner-graph invariant. Unequal remaining signatures certify that
    emitted codes are not check/qubit permutation copies.
    """
    fingerprint = _q16_fingerprint(hx, hz, n, 0)
    return (fingerprint[0],) + fingerprint[2:]


@lru_cache(maxsize=1)
def _sd_bank(G):
    """Twenty independent policies; unsuccessful cards have no orbit fallback."""
    assert G.q == 16
    assert A_SHAPE == (2,5) and B_SHAPE == (2,6)
    sa = [[2,2,2,2,0], [1,1,1,1,3]]
    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, profile = divmod(slot % 20, 4)
        bc = _sd_counts(architecture, profile)
        sb = [[c[i] for c in bc] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        arank = 30 if architecture == 4 and profile < 3 else 31
        brank = 31 if profile == 3 else 32
        chosen = None

        for retry in range(2):
            local = random.Random(
                BASE_SEED*100003 + 1616201
                + 104729*slot + 7919*retry)
            ap = _sd_pool(local, G, "A", architecture, profile,
                          sa, arank, retry)
            bp = _sd_pool(local, G, "B", architecture, profile,
                          sb, brank, retry)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (325*n <= 1000*k < 375*n):
                        continue
                    d = min(a["d"], b["d"], 10)
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, d=d,
                        cycles=a["cycles"]+b["cycles"],
                        key=(k*d*d, d, -a["cycles"]-b["cycles"])))
            proposals.sort(key=lambda item: item["key"], reverse=True)

            finalists = []
            for item in proposals[:5]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                xc, zc = _q16_columns(hx, n), _q16_columns(hz, n)
                if _q16_four(xc, hz)[0] != 5:
                    continue
                if _q16_four(zc, hx)[0] != 5:
                    continue
                signature = _sd_signature(hx, hz, n)
                if signature in emitted:
                    continue
                dx = _q16_probe(zc, local, 16, hx)
                dz = _q16_probe(xc, local, 16, hz)
                item.update(xc=xc, zc=zc, dx=dx, dz=dz,
                            signature=signature)
                finalists.append(item)

            # Every joint finalist receives the same fresh logical budget.
            # Earlier lighter witnesses are retained in the scalar minima.
            for item in finalists:
                item["dx"] = min(item["dx"], _q16_probe(
                    item["zc"], local, 48, item["hx"]))
                item["dz"] = min(item["dz"], _q16_probe(
                    item["xc"], local, 48, item["hz"]))
                d = min(item["d"], item["dx"], item["dz"])
                item["key"] = (
                    item["k"]*d*d, d,
                    min(item["dx"], item["dz"]), -item["cycles"])
            finalists.sort(key=lambda item: item["key"], reverse=True)

            for item in finalists:
                hx, hz, n, k = (
                    item["hx"], item["hz"], item["n"], item["k"])
                assert n == 544 and k > 0
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                assert all(len(cell) == len(set(cell)) <= 5
                           for grid in (item["A"], item["B"])
                           for row in grid for cell in row)
                assert 325*n <= 1000*k < 375*n
                emitted.add(item["signature"])
                frozen = tuple(
                    tuple(tuple(tuple(cell) for cell in row) for row in grid)
                    for grid in (item["A"], item["B"]))
                chosen = (frozen[0], frozen[1], k, item["key"])
                break
            if chosen is not None:
                break
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this slot's fresh SD16 design, not a retired Q16 filling."""
    selected = _sd_bank(G)[idx]
    if selected is None:
        raise ValueError("SD16 card found no fresh clean-seven target-rate code")
    A, B, k, score = selected
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection": score}


def CELL(rng, G, side, i, j, ctx):
    """Return all screened SD16 terms without resampling or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Difference packing and component screening only through weight six do not remove the weight-eight plateau on this bed: all five measured current-parent codes have d30s=8, with minimum witnesses exclusively in sector 1.
# REFUTED: Conflict-directed orbit-eight optimization raises the five measured readings above eight on the full-rank cubic Z17 bed: all five measured v2 leaders instead retained k=204 and d30s=8, leaving fitness at 22.588.
# REFUTED: Generating Q16 differences and central-first component shortlisting improve five-code fitness above 23.031: the measured program scored 17.0062, with three weight-six leaders and only two weight-eight leaders.
# REFUTED: All four Q16 cubic B incidence policies furnish productive clean-six slots: every current-program slot with idx%4==3 was voided for failure to find a fresh target-rate code.
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
