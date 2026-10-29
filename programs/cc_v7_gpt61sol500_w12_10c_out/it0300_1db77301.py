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
CONCEPT = ("Semidihedral SD32, order 32, shapes (1,3)x(2,5): noncommuting order-eight subgroup architectures, screened before shortlisting, should supply five fresh verified k=200 codes with d30s above eight and improve cell 0.350 beyond 23.529. This is a falsifiable measured-score hypothesis, not a distance certificate.")
GROUP = ("L3 jump from C17 to SD32: x^16=y^2=1 and y*x*y^-1=x^7. Its noncentral involutions distinguish it from the related Q32 presentation as well as from the parent. Related Q32 cubic beds measured five [544,200,8] leaders; use that evidence to test a different multiplication law, not to assert transferred distance. Odd-augmentation elements of this finite 2-group algebra are units.")
SHAPES = ("Target 0.350 with A=(1,3), B=(2,5): n=17*32=544, kb=6*32=192 and guaranteed rate 6/17. The size window permits q<=35; this family fixes q=32. Components have 96/32 and 160/64 variables/checks. Require verified component ranks 32/60 and assembled k=200, rate 0.367647. Parent weight-seven minima occupied sector-one Acol/Bcol blocks, particularly Acol1/2; replace both component geometries.")
FILLING = ("Twenty cards cross five A rotation-pair recipes with four B generator architectures: normal C8, two Q8 recipes and a D8 recipe. Every component column is cubic. B has four top binomials, four bottom monomials and one bottom-only trinomial, with row totals 8/7. Order-eight subgroup generation targets four B dependencies, but actual ranks gate acceptance. Compile cell contributions only from verified lifts; exclude component kernels through six before shortlisting, exhaustively test finalists through eight, independently deepen witness searches and screen both full logical quotients with stabilizer dressings. HX=12 and HZ<=11. Unequal uncoloured Tanner invariants separate emitted designs; equal fingerprints are conservatively rejected.")
FAMILY_NAME = "r035_SD32_13x25_noncommuting_subgroup8_preclean6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # SD32: (a,b)=x^a y^b, x^16=y^2=1, y*x*y^-1=x^7.
    if not valid_q(q):
        raise ValueError("SD32 requires order 32")
    els = [(a,b) for b in range(2) for a in range(16)]

    def red(g):
        return (g[0] % 16, g[1] % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + (1 if b == 0 else 7)*c) % 16, (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-(1 if b == 0 else 7)*a) % 16, b)

    def idx(g):
        a,b = red(g)
        return 16*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _retired_z17_defect_context(rng, G, idx):
    """Inactive historical optimizer; the SD32 builder never calls this."""
    import math
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = q * (ca - ra) * (cb - rb)
    mask = (1 << q) - 1
    assert q == 17

    from itertools import combinations, combinations_with_replacement

    # Preserve A's measured rank-surplus architecture, but do not remove
    # two B terms to pay for higher A degrees, as the mixed234 trial did.
    acounts = ((2, 1),) * 4 + ((0, 3),)

    # B keeps eighteen terms and row totals 9/9. Its distinguished
    # degree-two and degree-four columns cannot be interchanged with
    # cubic columns by a protograph permutation.
    bprofiles = []
    for small in range(3):
        for large in range(1, 4):
            for middle in combinations_with_replacement(range(4), 4):
                if small + large + sum(middle) != 9:
                    continue
                if middle.count(0) > 1 or middle.count(3) > 1:
                    continue
                # Pure same-row columns give short commuting-polynomial
                # relations regardless of their voltage assignments.
                if (small == 0 and 0 in middle
                        or small == 2 and 3 in middle):
                    continue
                counts = ((small, 2-small), (large, 4-large)) + tuple(
                    (a, 3-a) for a in middle)
                image = ((2-small, small), (4-large, large)) + tuple(
                    (3-a, a) for a in reversed(middle))
                if counts <= image:
                    bprofiles.append(counts)
    bprofiles.sort(key=lambda counts: (
        sum(a == 0 or b == 0 for a, b in counts),
        max(max(c) for c in counts), counts))
    assert len(bprofiles) >= 10

    card, batch = idx % 20, idx // 20
    profile = (card // 2 + 10 * batch) % len(bprofiles)
    bcounts = bprofiles[profile]
    # These change relative internal geometry, not a common multiplier
    # or translation of one completed labeling.
    defect_gap = (3, 4, 5, 6, 7, 8)[(card % 2 + 2 * batch) % 6]
    sa = [[c[i] for c in acounts] for i in range(2)]
    sb = [[c[i] for c in bcounts] for i in range(2)]
    assert tuple(map(sum, sa)) == (8, 7)
    assert tuple(map(sum, sb)) == (9, 9)
    assert tuple(map(sum, bcounts)) == (2, 4, 3, 3, 3, 3)
    assert gf2_rank([(a & 1) | ((b & 1) << 1)
                     for a, b in bcounts]) == 2
    target_k = kb + (cb - rb)
    assert target_k == 208
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

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
    geometry = (card // 2 + 3 * batch) % len(gap_sets)
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
                    if side == "B" and j == 1:
                        # The quartic column has counts (1,3), (2,2),
                        # or (3,1). Fix a genuine relative gap recipe,
                        # leaving its inter-row displacement free.
                        for i, count in enumerate(counts[j]):
                            base = rng.randrange(q)
                            if count == 2:
                                gap = 1 if i == 0 else defect_gap
                                vals[i] = [base, (base + gap) % q]
                            elif count == 3:
                                vals[i] = [
                                    base, (base + 1) % q,
                                    (base + defect_gap) % q]
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
        # Only A retains the odd-column exclusion of weight seven.
        # For mixed-degree B, seven is merely the next untested weight.
        odd_columns = all(c.bit_count() & 1 for c in cols)
        return (8 if odd_columns else 7), 0

    def eight_words(cols, bits, width, exhaustive=False):
        """Find component words of weights seven and eight.

        Triple/quartet collisions cover seven, and quartet/quartet
        collisions cover eight. Every word through six was excluded
        by short_six. Fast selection restricts syndrome weight; exact
        finalist screening removes that restriction.
        """
        seen, multiple, words = {}, {}, set()
        length = len(cols)
        odd_columns = all(c.bit_count() & 1 for c in cols)

        def insert(s, positions):
            if not exhaustive and s.bit_count() > 6:
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

        # A's odd column degrees make this triple table unnecessary.
        # B must explicitly test odd words rather than inherit that flag.
        if not odd_columns:
            for a in range(0, length, q):
                for b in range(a + 1, length - 1):
                    ab = cols[a] ^ cols[b]
                    for c in range(b + 1, length):
                        insert(ab ^ cols[c], (a, b, c))

        # Each constituent support orbit has a representative containing
        # coordinate zero in its first occupied protograph block.
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
            # Accept exactly the intended trivial-sector dependency,
            # not an uncontrolled deficit in a nontrivial cyclic factor.
            expected_rank = 2 * q - int(side == "A")
            if gf2_rank(cols) != expected_rank:
                return None
            floor, witness = short_six(cols, bits, width)
            if floor < 7:
                return (1000000 + 100000 * (7 - floor),
                        (witness,), cols, floor)
            words = eight_words(cols, bits, width)
            seven = sum(word.bit_count() == 7 for word in words)
            # First eliminate odd weight-seven witnesses, then reduce
            # eight-word load. Neither count is a quantum-distance claim.
            cost = 10000 * seven + len(words)
            return cost, tuple(words), cols, floor

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
            if floor < 7:
                exact_words = witnesses
                key = (floor, -len(witnesses), -len(witnesses))
            else:
                exact_words = eight_words(
                    cols, bits, width, exhaustive=True)
                odd_columns = all(c.bit_count() & 1 for c in cols)
                # Nine excludes component words through eight.
                # Only odd-column A may upgrade that exclusion to ten.
                floor = min(
                    (word.bit_count() for word in exact_words),
                    default=(10 if odd_columns else 9))
                minimum_load = sum(
                    word.bit_count() == floor for word in exact_words)
                key = (floor, -minimum_load, -len(exact_words))
            if winner is None or key > winner[0]:
                winner = (key, grid, exact_words)
        return winner[1], winner[0]

    # Seek an A floor above eight, and give the new mixed-parity B
    # architecture additional repairs rather than accepting its first
    # weight-seven local optimum.
    A, pa = optimize("A", acounts, 112 + 8 * (idx % 3))
    B, pb = optimize("B", bcounts, 160 + 8 * (idx % 3))
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    # Final acceptance uses the evaluator's exact matrices and ranks;
    # the augmentation argument alone is not an acceptance criterion.
    assert k == target_k
    assert 325 * n <= 1000 * k < 375 * n
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return {"idx": idx, "cells": {"A": A, "B": B},
            "component_screen": (pa, pb),
            "gap_geometry": (geometry, profile, defect_gap),
            "k": k, "rank_surplus": k - kb}


from functools import lru_cache
from collections import Counter

_SD32_A_RECIPES = (
    (1, 1, 1), (1, 1, 2), (1, 2, 2),
    (1, 2, 4), (1, 3, 5),
)
_SD32_B_DIRECTIONS = (
    ((2,0), (6,0), (2,0), (6,0)),
    ((4,0), (1,1), (5,1), (1,1)),
    ((4,0), (1,1), (5,1), (4,0)),
    ((4,0), (0,1), (4,1), (0,1)),
)
_SD32_SA = [[3, 3, 3]]
_SD32_SB = [[2, 2, 2, 2, 0], [1, 1, 1, 1, 3]]


def _sd32_columns(rows, width):
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return cols


def _sd32_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _sd32_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


@lru_cache(maxsize=1)
def _sd32_atlas(G):
    """Compile unit contributions from verified zero-companion lifts."""
    q = G.q
    mask = (1 << q)-1
    out = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g]]]
            zero = [[[]]]
            hx, hz, _ = lifted_matrices(
                G, unit if side == "A" else zero,
                zero if side == "A" else unit)
            rows = hx if side == "A" else hz
            blocks.append(tuple(_sd32_columns(
                [row & mask for row in rows], q)))
        out[side] = tuple(blocks)
    return out


def _sd32_component(G, atlas, side, grid):
    """Superpose scaffold-generated blocks, not a handwritten lift."""
    q = G.q
    cols = [0]*(len(grid[0])*q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                block = atlas[side][G.idx(g)]
                for t, syndrome in enumerate(block):
                    cols[j*q+t] ^= syndrome << (i*q)
    return cols


def _sd32_verified_component(G, side, grid):
    width = len(grid[0])*G.q
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, [[[]]])
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, [[[]]], grid)
        rows = hz
    mask = (1 << width)-1
    return _sd32_columns([row & mask for row in rows], width)


def _sd32_short4(cols, stabilizers=()):
    """Exact kernel/logical existence test through four."""
    basis = _sd32_basis(stabilizers)
    seen = {0: 0}
    bits = [1 << j for j in range(len(cols))]
    for j, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[j]
        elif _sd32_outside(bits[j] ^ old, basis):
            return False
    for j, syndrome in enumerate(cols):
        for h in range(j):
            value = syndrome ^ cols[h]
            word = bits[j] | bits[h]
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _sd32_outside(word ^ old, basis):
                return False
    return True


def _sd32_decoder(cols, height):
    """Exact bounded constituent decoding after exclusion through four."""
    bits = [1 << j for j in range(len(cols))]
    incidence = [[] for _ in range(height)]
    tails = {0: 0}
    for j, syndrome in enumerate(cols):
        assert syndrome.bit_count() == 3
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

    def decode(target, limit, forbidden):
        def visit(value, remaining, word):
            if not value:
                return word
            if value.bit_count() > 3*remaining:
                return None
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
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
        return visit(target, limit, 0)
    return decode


def _sd32_bounded(cols, q, height, bound):
    """Exhaust anchored constituent words up to bound, without a node cap.

    Each constituent regular action is transitive inside a variable
    block. This reduction is not asserted for the complete LP code.
    Return an actual witness or zero, never a sampled distance.
    """
    decode = _sd32_decoder(cols, height)
    for anchor in range(0, len(cols), q):
        completion = decode(cols[anchor], bound-1, 1 << anchor)
        if completion is not None:
            word = completion | (1 << anchor)
            syndrome, value = 0, word
            while value:
                bit = value & -value
                syndrome ^= cols[bit.bit_length()-1]
                value ^= bit
            assert syndrome == 0 and word.bit_count() <= bound
            return word
    return 0


def _sd32_cycles(cols):
    seen, total = {}, 0
    for syndrome in cols:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length()-1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b,a)
                old = seen.get(key, 0)
                total += old
                seen[key] = old+1
    return total


def _sd32_probe(cols, rng, trials, seeds=(), stabilizers=(),
                initial=None):
    """Actual kernel/logical witnesses, retaining earlier lighter readings."""
    basis = _sd32_basis(stabilizers)
    order = list(range(len(cols)))
    best = len(cols)+1 if initial is None else initial
    memory = sorted({w for w in seeds if w},
                    key=lambda w: (w.bit_count(), w))[:32]

    def observe(word):
        nonlocal best
        if (word and word.bit_count() < best
                and _sd32_outside(word, basis)):
            best = word.bit_count()

    for word in memory:
        value, syndrome = word, 0
        while value:
            bit = value & -value
            syndrome ^= cols[bit.bit_length()-1]
            value ^= bit
        assert syndrome == 0
        observe(word)

    for trial in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                observe(word)

        for sweep in range(2):
            reducers = sorted(set(words+memory),
                              key=lambda w: (w.bit_count(), w))[:28]
            for i, word in enumerate(words):
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                words[i] = word
                observe(word)

        short = sorted(set(words+memory),
                       key=lambda w: (w.bit_count(), w))[:32]
        discoveries = set(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    observe(combined)
                    if combined.bit_count() <= best+4:
                        discoveries.add(combined)
        if trial % 4 == 0:
            for i, word in enumerate(short[:8]):
                for j in range(i):
                    for h in range(j):
                        combined = word ^ short[j] ^ short[h]
                        if combined:
                            observe(combined)
                            if combined.bit_count() <= best+4:
                                discoveries.add(combined)

        if stabilizers:
            logicals = [w for w in discoveries
                        if _sd32_outside(w, basis)]
            for word in sorted(logicals, key=int.bit_count)[:3]:
                for sweep in range(2):
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda item: item[0])[:12]
                    chosen, weight = word, word.bit_count()
                    for _, row in nearby:
                        combined = word ^ row
                        if combined.bit_count() < weight:
                            chosen, weight = combined, combined.bit_count()
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            combined = word ^ row ^ other
                            if combined.bit_count() < weight:
                                chosen, weight = combined, combined.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)

        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        if best <= 6:
            break
    return best, tuple(memory)


def _sd32_triple(rng, G, step=None):
    """Fresh mixed-coset, noncommuting trinomials."""
    for attempt in range(256):
        if step is None:
            cell = rng.sample(G.elements, 3)
        else:
            a, b = rng.randrange(16), rng.randrange(2)
            cell = [(a,b), ((a+step) % 16,b),
                    (rng.randrange(16),1-b)]
        if any(G.mul(G.inv(g), h) == (8,0)
               for i,g in enumerate(cell) for h in cell[:i]):
            continue
        relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
        if G.mul(relatives[0], relatives[1]) == G.mul(
                relatives[1], relatives[0]):
            continue
        assert len(set(cell)) == 3
        return cell
    raise ValueError("SD32 noncommuting trinomial recipe exhausted")


def _sd32_draw(rng, G, side, recipe, mode):
    if side == "A":
        return [[_sd32_triple(rng, G, step) for step in recipe]]
    top, bottom = [], []
    for direction in _SD32_B_DIRECTIONS[mode]:
        anchor = G.pick(rng)
        # Fix g1*g0^-1=direction. Independent anchors therefore preserve
        # the prescribed check-orbit subgroup even when it is nonnormal.
        top.append([anchor, G.mul(direction, anchor)])
        bottom.append([G.pick(rng)])
    top.append([])
    bottom.append(_sd32_triple(rng, G))
    return [top, bottom]


def _sd32_pool(rng, G, atlas, side, recipe, mode, retry):
    """Exclude short words on every proposal, before heuristic shortlisting."""
    height = G.q if side == "A" else 2*G.q
    required_rank = 32 if side == "A" else 60
    quota = (160 if side == "A" else 224)*(retry+1)
    pool, seen = [], set()
    admitted = 0

    def admit(grid):
        nonlocal admitted
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        cols = _sd32_component(G, atlas, side, grid)
        assert all(c.bit_count() == 3 for c in cols)
        if gf2_rank(cols) != required_rank:
            return
        if not _sd32_short4(cols):
            return
        if _sd32_bounded(cols, G.q, height, 6):
            return
        reading, words = _sd32_probe(cols, rng, 3)
        assert reading >= 8
        pool.append(dict(
            grid=grid, cols=cols, d=reading, words=words,
            cycles=_sd32_cycles(cols), clean8=False))
        pool.sort(key=lambda item: (
            min(item["d"], 12), -item["cycles"]), reverse=True)
        del pool[6:]
        admitted += 1

    for attempt in range(quota):
        admit(_sd32_draw(rng, G, side, recipe, mode))
        if admitted >= 20:
            break

    for step in range(32):
        if not pool:
            admit(_sd32_draw(rng, G, side, recipe, mode))
            continue
        parent = rng.choice(pool[:min(3, len(pool))])
        grid = [[list(cell) for cell in row] for row in parent["grid"]]
        proposal = _sd32_draw(rng, G, side, recipe, mode)
        qmask = (1 << G.q)-1
        pressure = [
            1+sum(((w >> (j*G.q)) & qmask).bit_count()
                  for w in parent["words"][:8])
            for j in range(len(grid[0]))]
        j = rng.choices(range(len(grid[0])), weights=pressure, k=1)[0]
        for i in range(len(grid)):
            grid[i][j] = proposal[i][j]
        admit(grid)

    for item in pool:
        d, words = _sd32_probe(
            item["cols"], rng, 16, item["words"], initial=item["d"])
        item["d"], item["words"] = min(item["d"], d), words
    pool.sort(key=lambda item: (
        min(item["d"], 12), -item["cycles"]), reverse=True)
    finalists = pool[:3]

    for item in finalists:
        witness = _sd32_bounded(item["cols"], G.q, height, 8)
        item["clean8"] = not bool(witness)
        seeds = item["words"]+((witness,) if witness else ())
        initial = min(item["d"], witness.bit_count()) if witness else item["d"]
        d, words = _sd32_probe(
            item["cols"], rng, 48, seeds, initial=initial)
        item["d"], item["words"] = min(initial, d), words
    finalists.sort(key=lambda item: (
        min(item["d"], 12), item["clean8"], -item["cycles"]),
        reverse=True)
    return finalists


def _sd32_embeddings(words, side, q):
    mask = (1 << q)-1
    if side == "B":
        return [word << (j*5*q)
                for word in words[:12] for j in range(3)]
    return [
        sum(((word >> (j*q)) & mask) << ((j*5+beta)*q)
            for j in range(3))
        for word in words[:12] for beta in range(5)]


def _sd32_fingerprint(hx, hz, n):
    """Uncoloured incidence invariants; inequality proves nonisomorphism."""
    rows = hx+hz
    weights = [row.bit_count() for row in rows]
    adjacency = [0]*len(rows)
    overlaps = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            if value:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlaps[i][j] = overlaps[j][i] = value
    profiles = []
    for i, neighbors in enumerate(overlaps):
        histogram = Counter(
            (weights[j], value,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, value in neighbors.items())
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    degrees = Counter(c.bit_count() for c in _sd32_columns(rows, n))
    return (n, tuple(sorted(degrees.items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _sd32_bank(G):
    """Optimize independent structural cards; never emit an orbit fallback."""
    assert G.q == 32
    assert A_SHAPE == (1,3) and B_SHAPE == (2,5)
    assert _grid_weights(_SD32_SA, _SD32_SB) == 12
    assert _no_dead_lines(_SD32_SA, _SD32_SB)

    # Verify each prescribed subgroup directly rather than relying on names.
    for directions in _SD32_B_DIRECTIONS:
        members = {G.e}
        moves = list(directions)+[G.inv(g) for g in directions]
        pending = [G.e]
        while pending:
            g = pending.pop()
            for h in moves:
                v = G.mul(g,h)
                if v not in members:
                    members.add(v)
                    pending.append(v)
        assert len(members) == 8

    atlas = _sd32_atlas(G)
    bank, emitted = [], set()
    q, mask = G.q, (1 << G.q)-1

    for slot in range(N_SLOTS):
        recipe = _SD32_A_RECIPES[(slot % 20)//4]
        mode = slot % 4
        rng = random.Random(
            BASE_SEED*100003+320135200+104729*slot)
        winner = None

        for retry in range(2):
            ap = _sd32_pool(rng, G, atlas, "A", recipe, mode, retry)
            bp = _sd32_pool(rng, G, atlas, "B", recipe, mode, retry)
            proposals = [
                ((min(a["d"], b["d"], 12),
                  int(a["clean8"])+int(b["clean8"]),
                  -a["cycles"]-b["cycles"]), a, b)
                for a in ap for b in bp]
            proposals.sort(key=lambda item: item[0], reverse=True)
            finalists = []

            for _, a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 544 or k != 200:
                    continue
                assert _sd32_verified_component(G, "A", A) == a["cols"]
                assert _sd32_verified_component(G, "B", B) == b["cols"]

                # Also confirm the extracted components inside full matrices.
                ar = [
                    sum(((row >> (j*5*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
                assert _sd32_columns(ar, 3*q) == a["cols"]
                assert _sd32_columns(br, 5*q) == b["cols"]

                fingerprint = _sd32_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                xc, zc = _sd32_columns(hz, n), _sd32_columns(hx, n)
                if not _sd32_short4(xc, hx):
                    continue
                if not _sd32_short4(zc, hz):
                    continue

                dx, xwords = _sd32_probe(
                    xc, rng, 10,
                    _sd32_embeddings(b["words"], "B", q),
                    stabilizers=hx)
                dz, zwords = _sd32_probe(
                    zc, rng, 10,
                    _sd32_embeddings(a["words"], "A", q),
                    stabilizers=hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    xc=xc, zc=zc, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    da=a["d"], db=b["d"],
                    clean8=int(a["clean8"])+int(b["clean8"]),
                    cycles=a["cycles"]+b["cycles"],
                    fingerprint=fingerprint))
                if len(finalists) == 3:
                    break

            for item in finalists:
                dx, _ = _sd32_probe(
                    item["xc"], rng, 32, item["xwords"],
                    stabilizers=item["hx"], initial=item["dx"])
                dz, _ = _sd32_probe(
                    item["zc"], rng, 32, item["zwords"],
                    stabilizers=item["hz"], initial=item["dz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                d = min(item["da"], item["db"],
                        item["dx"], item["dz"], 12)
                item["key"] = (
                    k*d*d/n, d, item["clean8"], -item["cycles"])
                if winner is None or item["key"] > winner["key"]:
                    winner = item
            if winner is not None and winner["key"][1] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["k"] == 200
        assert 325*winner["n"] <= 1000*winner["k"] < 375*winner["n"]
        assert max(row.bit_count()
                   for row in winner["hx"]+winner["hz"]) <= W_MAX
        assert [[len(c) for c in row] for row in A] == _SD32_SA
        assert [[len(c) for c in row] for row in B] == _SD32_SB
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (A,B) for row in grid for c in row)
        emitted.add(winner["fingerprint"])
        frozen = tuple(
            tuple(tuple(tuple(cell) for cell in row) for row in grid)
            for grid in (A,B))
        bank.append(dict(
            cells={"A": frozen[0], "B": frozen[1]},
            k=200, card=(recipe, mode), selection=winner["key"]))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the slot's fresh SD32 filling, not the retired cyclic design."""
    selected = _sd32_bank(G)[idx]
    if selected is None:
        raise ValueError("SD32 card found no fresh clean-six rank-200 code")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened SD32 cell without additional draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Difference packing and component screening only through weight six do not remove the weight-eight plateau on this bed: all five measured current-parent codes have d30s=8, with minimum witnesses exclusively in sector 1.
# REFUTED: Conflict-directed orbit-eight optimization raises the five measured readings above eight on the full-rank cubic Z17 bed: all five measured v2 leaders instead retained k=204 and d30s=8, leaving fitness at 22.588.
# REFUTED: Cubic rank-surplus selection alone removes the distance-eight bottleneck: all five measured v3 leaders retained k=208 and d30s=8, with 6319-7109 verified weight-eight operators per leader.
# REFUTED: Reducing B to sixteen terms with two complementary pure binomials while increasing A degrees raises this bed above eight: the shown mixed234 trial measured five distance-seven leaders and fitness 17.5652.
# REFUTED: Keeping eighteen B terms with a balanced degree-two/four defect pair raises five C17 readings above eight: all five measured v5 leaders instead had k=208 and d30s=7, lowering fitness to 17.6330.
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
