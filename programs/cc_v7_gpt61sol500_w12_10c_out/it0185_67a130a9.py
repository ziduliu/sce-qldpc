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
CONCEPT = ("C7 semidirect C12 with action 2, order 84, shapes (1,2)x(1,3): noncommuting trinomials selected against independently searched component and dressed logical witnesses should exceed the cell-0.300 frontier fitness 38.0. This is a measured-score hypothesis, not a distance certificate.")
GROUP = ("L3 jump from Z17 to C7:C12, y*x*y^-1=x^2. Since 2^3=1 modulo 7, the order-84 presentation is valid. Its centre C4 distinguishes it from the recorded order-84 centre-C2 beds. The related order-63 centre-three bed reached 29.8 and the order-84 matched-shape frontier reached 38.0; transfer to this group is unproved.")
SHAPES = ("Target cell 0.300 with (1,2)x(1,3): n=7*84=588, kb=2*84=168 and guaranteed rate 2/7. Components have 168 and 252 variables with 84 checks each; the size window permits q<=85. Parent weight-eight minima were exclusively sector-one Acol/Bcol witnesses, often emphasizing Acol2 and Bcol2. Replace both component geometries and gate every emitted code by verified 0.275<=k/n<0.325.")
FILLING = ("Twenty policies cross five quotient-C12 occupancy architectures with four rotation-residue models. All five cells contain three distinct terms, so HX weight is nine and HZ weight twelve. Reject commuting relative generators and internal ratios of order two or three. Exact component exclusion through four precedes fresh-label selection and adaptive redraws. Independently deepen component witnesses, validate actual logical embeddings and stabilizer dressings, and require exact full-code logical-four exclusion. A deterministic bank accepts only unequal uncoloured incidence fingerprints. Sampled witness weights are upper bounds, never distance certificates.")
FAMILY_NAME = "r030_C7semiC12_action2_12x13_trinomial_dressed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # x^7=y^12=1, y*x*y^-1=x^2; 2^3=1 modulo 7.
    if not valid_q(q):
        raise ValueError("C7:C12 requires order 84")
    els = [(a, b) for b in range(12) for a in range(7)]
    action = [pow(2, b, 7) for b in range(12)]

    def red(g):
        return (g[0] % 7, g[1] % 12)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 7, (b+d) % 12)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 12
        return ((-action[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return a + 7*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _retired_z17_context(rng, G, idx):
    """Inactive historical optimizer; the new group never calls this."""
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
from collections import Counter


_JUMP_PATTERNS = (
    (0, 1, 2), (0, 1, 3), (0, 0, 1),
    (0, 2, 3), (0, 1, 6),
)
# Each tuple prescribes the two A cells followed by the three B cells.
_JUMP_CARDS = (
    (0, 0, 0, 0, 0),
    (0, 1, 2, 3, 4),
    (2, 3, 0, 4, 1),
    (4, 2, 1, 0, 3),
    (1, 4, 3, 2, 0),
)


def _jump_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _jump_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            previous = basis.get(p)
            if previous is None:
                basis[p] = row
                break
            row ^= previous
    return basis


def _jump_outside(word, basis):
    while word:
        p = word.bit_length()-1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _jump_small(cols, stabilizers=()):
    """Exact kernel/logical existence test through weight four.

    One representative per syndrome suffices: when every collision
    difference lies in the supplied span, all representatives of that
    syndrome have the same logical class.
    """
    span = _jump_basis(stabilizers)
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
        elif _jump_outside(word ^ previous, span):
            return True
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif _jump_outside(word ^ previous, span):
                return True
    return False


def _jump_components(hx, hz, q):
    """Read both actions from the VERIFIED full LP matrices."""
    mask = (1 << q)-1
    ar = [
        (row & mask) | (((row >> (3*q)) & mask) << q)
        for row in hx[:q]
    ]
    br = [row & ((1 << (3*q))-1) for row in hz[:q]]
    return (_jump_columns(ar, 2*q), _jump_columns(br, 3*q))


def _jump_cycles(cols):
    pairs, total = {}, 0
    for syndrome in cols:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length()-1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                pair = (b, a)
                previous = pairs.get(pair, 0)
                total += previous
                pairs[pair] = previous + 1
    return total


def _jump_fiber_seeds(cols, G):
    """True kernel witnesses restricted to normal-C7 fiber indicators."""
    q = G.q
    fibers = [
        sum(1 << G.idx(g) for g in G.elements if g[1] == b)
        for b in range(12)
    ]
    expanded = [
        fiber << (j*q)
        for j in range(len(cols)//q) for fiber in fibers
    ]
    pivots, dependencies = {}, []
    for c, full in enumerate(expanded):
        syndrome, remaining = 0, full
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        word = 1 << c
        while syndrome:
            p = syndrome.bit_length()-1
            previous = pivots.get(p)
            if previous is None:
                pivots[p] = (syndrome, word)
                break
            syndrome ^= previous[0]
            word ^= previous[1]
        if not syndrome:
            witness = 0
            while word:
                bit = word & -word
                witness ^= expanded[bit.bit_length()-1]
                word ^= bit
            dependencies.append(witness)

    seeds = set(dependencies)
    for i, word in enumerate(dependencies):
        seeds.update(word ^ other for other in dependencies[:i])
    seeds.discard(0)
    return sorted(seeds, key=lambda v: (v.bit_count(), v))[:24]


def _jump_probe(cols, seeds, rng, trials):
    """Search actual nonzero kernel words; a high reading is not a bound."""
    width = len(cols)
    memory = sorted(set(seeds), key=lambda v: (v.bit_count(), v))[:24]
    best = min((v.bit_count() for v in memory), default=width+1)
    order = list(range(width))

    for trial in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        # Shorten true dependencies without leaving the component kernel.
        for sweep in range(1 + int(trials >= 12)):
            reducers = sorted(set(words+memory), key=int.bit_count)[:28]
            rng.shuffle(reducers)
            for i, word in enumerate(words):
                weight = word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                words[i] = word
                best = min(best, weight)

        short = sorted(set(words+memory),
                       key=lambda v: (v.bit_count(), v))[:40]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    weight = candidate.bit_count()
                    best = min(best, weight)
                    if weight <= best+4:
                        discoveries.append(candidate)
        if trials >= 12 and trial % 4 == 0:
            small = short[:10]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for h in range(j):
                        candidate = pair ^ small[h]
                        if candidate:
                            best = min(best, candidate.bit_count())
                            discoveries.append(candidate)
        memory = sorted(set(memory+discoveries),
                        key=lambda v: (v.bit_count(), v))[:24]
        if best <= 6:
            break

    seeds[:] = memory
    return best


def _jump_dressed(item, q):
    """Check actual logical classes before one- and two-check dressings."""
    hx, hz, n = item["hx"], item["hz"], item["n"]
    mask = (1 << q)-1
    best = n+1
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        span = _jump_basis(stabilizers)
        for seed in sorted(set(seeds), key=int.bit_count)[:8]:
            for fixed in range(3 if side == "A" else 2):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((3*j+fixed)*q) for j in range(2))
                else:
                    word = seed << (fixed*3*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    raise ValueError("verified component embedding failed")
                if not _jump_outside(word, span):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        candidate = word ^ row
                        cw = candidate.bit_count()
                        nearby.append((cw, row))
                        if 0 < cw < weight:
                            chosen, weight = candidate, cw
                    nearby.sort(key=lambda entry: entry[0])
                    reducers = [row for _, row in nearby[:12]]
                    for i, row in enumerate(reducers):
                        for other in reducers[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if 0 < cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
    return best


def _jump_fingerprint(hx, hz, n):
    """Uncoloured incidence invariants; inequality proves nonisomorphism.

    Neither labels, protograph positions, nor X/Z colouring enter this
    signature. Equal signatures are rejected conservatively.
    """
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            incidents[bit.bit_length()-1].append(r)
            row ^= bit
    adjacency = [Counter() for _ in rows]
    for neighbors in incidents:
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = {}
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] = square.get(t, 0) + multiplicity*other
        triangles = sum(neighbors.get(t, 0)*value
                        for t, value in square.items())
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0), triangles,
            sum(value*value for value in square.values()),
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (n, tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _jump_bank(G):
    """Pure deterministic optimization of incidence-distinct design cards."""
    assert G.q == 84
    q = G.q
    assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
    assert _no_dead_lines([[3, 3]], [[3, 3, 3]])
    bank, emitted = [], set()

    def merit(item):
        # A cap prevents especially unsuccessful searches from dominating
        # selection. Independent rescreening can only lower each reading.
        d = min(item["da"], item["db"], item.get("logical", 1000), 18)
        return (item["k"]*d*d/item["n"], d,
                min(item["da"], 24)+min(item["db"], 24),
                -item["cycles"])

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        policy = _JUMP_CARDS[architecture]
        rng = random.Random(84012371 + 100003*slot)

        def cell(pattern):
            for retry in range(192):
                phase = rng.randrange(12)
                direction = rng.choice((1, 5, 7, 11))
                bs = [(phase+direction*b) % 12
                      for b in _JUMP_PATTERNS[pattern]]
                if mode == 0:
                    aa = rng.sample(range(7), 3)
                elif mode == 1:
                    base, step = rng.randrange(7), rng.randrange(1, 7)
                    aa = [(base+t*step) % 7 for t in range(3)]
                elif mode == 2:
                    base = rng.randrange(7)
                    aa = [base, base, rng.choice(
                        [a for a in range(7) if a != base])]
                    rng.shuffle(aa)
                else:
                    aa = [rng.randrange(7) for _ in range(3)]
                result = list(zip(aa, bs))
                if len(set(result)) != 3:
                    continue
                relatives = [G.mul(G.inv(result[0]), g)
                             for g in result[1:]]
                if G.mul(relatives[0], relatives[1]) == G.mul(
                        relatives[1], relatives[0]):
                    continue
                legal = True
                for i, g in enumerate(result):
                    for h in result[:i]:
                        relative = G.mul(G.inv(g), h)
                        square = G.mul(relative, relative)
                        if square == G.e or G.mul(square, relative) == G.e:
                            legal = False
                            break
                    if not legal:
                        break
                if legal:
                    return result
            raise ValueError("trinomial policy exhausted")

        def proposal():
            cells = [cell(pattern) for pattern in policy]
            return [[cells[0], cells[1]]], [[cells[2], cells[3], cells[4]]]

        finalists, labels_seen = [], set()
        chosen = None

        def admit(A, B):
            labels = tuple(tuple(sorted(c))
                           for grid in (A, B) for row in grid for c in row)
            if labels in labels_seen:
                return
            labels_seen.add(labels)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 588 and 275*n <= 1000*k < 325*n):
                return
            ac, bc = _jump_components(hx, hz, q)
            if _jump_small(ac) or _jump_small(bc):
                return
            assert all(col.bit_count() == 3 for col in ac+bc)
            aw, bw = _jump_fiber_seeds(ac, G), _jump_fiber_seeds(bc, G)
            item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        ac=ac, bc=bc, aw=aw, bw=bw,
                        da=_jump_probe(ac, aw, rng, 3),
                        db=_jump_probe(bc, bw, rng, 4),
                        cycles=_jump_cycles(ac)+_jump_cycles(bc))
            finalists.append(item)
            finalists.sort(key=merit, reverse=True)
            del finalists[8:]

        for batch in range(3):
            for attempt in range(48):
                admit(*proposal())

            # Change actual voltages in the limiting component, retaining
            # the card's quotient architecture instead of relabeling it.
            for step in range(24):
                if not finalists:
                    admit(*proposal())
                    continue
                parent = finalists[rng.randrange(min(4, len(finalists)))]
                A = [[list(c) for c in row] for row in parent["A"]]
                B = [[list(c) for c in row] for row in parent["B"]]
                side = ("B" if parent["db"] <= parent["da"] else "A")
                if step % 5 == 0:
                    side = rng.choice(("A", "B"))
                width = 2 if side == "A" else 3
                j = rng.randrange(width)
                grid = A if side == "A" else B
                grid[0][j] = cell(policy[j if side == "A" else j+2])
                if step % 8 == 7:
                    other = (j+1) % width
                    grid[0][other] = cell(
                        policy[other if side == "A" else other+2])
                admit(A, B)

            for item in finalists:
                item["da"] = min(item["da"],
                    _jump_probe(item["ac"], item["aw"], rng, 12))
                item["db"] = min(item["db"],
                    _jump_probe(item["bc"], item["bw"], rng, 20))
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:4]

            for item in finalists:
                item["da"] = min(item["da"],
                    _jump_probe(item["ac"], item["aw"], rng, 28))
                item["db"] = min(item["db"],
                    _jump_probe(item["bc"], item["bw"], rng, 44))
                item["logical"] = _jump_dressed(item, q)

            for item in sorted(finalists, key=merit, reverse=True):
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if item["logical"] > n:
                    continue
                if _jump_small(_jump_columns(hx, n), hz):
                    continue
                if _jump_small(_jump_columns(hz, n), hx):
                    continue
                fingerprint = _jump_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) == 12
                assert item["k"] >= 168
                assert all(len(c) == len(set(c)) == 3
                           for grid in (item["A"], item["B"])
                           for row in grid for c in row)
                emitted.add(fingerprint)
                chosen = (item["A"], item["B"], item["k"], merit(item))
                break
            if chosen is not None:
                break
        # Failed cards void only their own slot, never replay an old code.
        bank.append(chosen)

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this new group's independently optimized complete filling."""
    selected = _jump_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh target-rate trinomial code passed screening")
    A, B, k, score = selected
    return {"idx": idx,
            "cells": {
                "A": [[list(c) for c in row] for row in A],
                "B": [[list(c) for c in row] for row in B],
            },
            "k": k, "selection": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without resampling or cyclic packing."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Difference packing and component screening only through weight six do not remove the weight-eight plateau on this bed: all five measured current-parent codes have d30s=8, with minimum witnesses exclusively in sector 1.
# REFUTED: Conflict-directed orbit-eight optimization raises the five measured readings above eight on the full-rank cubic Z17 bed: all five measured v2 leaders instead retained k=204 and d30s=8, leaving fitness at 22.588.
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
