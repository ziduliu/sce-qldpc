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
CONCEPT = ("C43, cyclic order 43, shapes (2,3)x(2,3): degree-distinguished supports, difference-packed labels and cofactor-aware witness repairs should exceed the 0.100 cell record 60.3346 while retaining k/n<0.125.")
GROUP = ("Jump from nonabelian C13:C3 to genuinely different cyclic C43. This bed is absent from BEDS. C46 reached 60.3 versus the current parent's 44.7624, motivating another abelian comparison, not proving superiority. Prime order removes proper nontrivial subgroup norms, but commutativity supplies cofactor witnesses; screen those explicitly using verified component syndromes.")
SHAPES = ("Keep equal (2,3) sides for cell 0.100: n=559, guaranteed k>=43 and rate>=1/13, with two 129-variable/86-check components. These shapes permit q<=46; C43 fixes q=43. All measured parent weight-24 minima occupied sector-1 Acol/Bcol blocks, particularly Acol1/2 and Bcol1/2, not Arow/Brow. Exact assembled ranks enforce 8*k<n.")
FILLING = ("Use canonical support pairs with different uncoloured check/qubit degree signatures. Transport positive cell sizes through five terms while preserving every row and column total. Pack internal and cross-row C43 differences, seed syndrome-verified abelian cofactors, retain randomized kernel words, recombine verified translations, and repair witness-heavy columns. Independently deepen component finalists, measure actual full ranks, and search genuine logical representatives plus single/pair stabilizer dressings. Component merit remains a heuristic; found logical weights are upper bounds, never distance certificates.")
FAMILY_NAME = "r010_C43_23x23_prime_sidon_cofactor_orbit_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 43

def custom_ops(q):
    # Additive cyclic group C43, encoded by one-tuples.
    if not valid_q(q):
        raise ValueError("C43 has order 43")
    els = [(a,) for a in range(43)]

    def red(g):
        return (g[0] % 43,)

    def mul(g, h):
        return ((g[0] + h[0]) % 43,)

    def inv(g):
        return ((-g[0]) % 43,)

    def idx(g):
        return g[0] % 43

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _support_designs():
    """Different degree signatures certify different Tanner graphs.

    Augmentation ranks interleave distinct constructive support policies.
    They are not substitutes for the assembled code's measured rank.
    """
    from itertools import product, permutations
    perms = tuple(permutations(range(3)))
    profiles = []
    for t in product(range(1, 5), repeat=6):
        rows = (sum(t[:3]), sum(t[3:]))
        cols = tuple(t[j] + t[j+3] for j in range(3))
        if min(rows) < 5 or max(rows) > 8:
            continue
        if min(cols) < 3 or max(cols) > 6:
            continue
        images = [
            tuple(t[3*r+c] for r in order for c in perm)
            for order in ((0, 1), (1, 0)) for perm in perms]
        if t != min(images):
            continue
        p0 = sum((t[j] & 1) << j for j in range(3))
        p1 = sum((t[j+3] & 1) << j for j in range(3))
        rank = int(bool(p0 or p1))
        rank += int(bool(p0 and p1 and p0 != p1))
        profiles.append((t, rows, cols, rank))

    buckets = {}
    for a, (ta, ar, ac, ka) in enumerate(profiles):
        for tb, br, bc, kb in profiles[a:]:
            if max(ar)+max(bc) > 12 or max(br)+max(ac) > 12:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc] +
                [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc] +
                [x+y for x in ar for y in br]))
            priority = (min(ac+bc), min(ar+br), sum(ta)+sum(tb),
                        -max(ac+bc), -max(ar+br))
            buckets.setdefault(tuple(sorted((ka, kb))), []).append(
                (priority, (checks, qubits), ta, tb))
    for bucket in buckets.values():
        bucket.sort(reverse=True)

    strata = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
    positions = {s: 0 for s in strata}
    seen, designs = set(), []
    while True:
        added = False
        for stratum in strata:
            bucket = buckets.get(stratum, ())
            while positions[stratum] < len(bucket):
                _, fingerprint, ta, tb = bucket[positions[stratum]]
                positions[stratum] += 1
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                designs.append((ta, tb))
                added = True
                break
        if not added:
            break
    if len(designs) < 20:
        raise ValueError("fewer than twenty degree-distinguished designs")
    return tuple(designs)


_SUPPORT_DESIGNS = _support_designs()


def _transport_counts(rng, counts):
    """Preserve row/column totals, hence both weight caps and signatures."""
    from itertools import product
    counts = tuple(counts)
    columns = tuple(counts[j]+counts[j+3] for j in range(3))
    variants = []
    for top in product(range(1, 6), repeat=3):
        bottom = tuple(columns[j]-top[j] for j in range(3))
        if sum(top) == sum(counts[:3]) and all(1 <= w <= 5 for w in bottom):
            variants.append(tuple(top)+bottom)
    return rng.choice(variants)


def _packed_grid(rng, G, counts, policy, transport=True):
    """Pack group differences; subsequent cycle measurements use the lift."""
    from collections import Counter
    if transport:
        counts = _transport_counts(rng, counts)
    internal_weight, row_weight, cross_weight = (
        (4, 2, 4), (6, 4, 1), (2, 4, 6), (6, 1, 3))[policy % 4]
    grid = [[None]*3 for _ in range(2)]
    row_used = [Counter(), Counter()]
    cross_used = Counter()
    for i in range(2):
        for j in range(3):
            count = counts[3*i+j]
            winner = None
            for attempt in range(12):
                cell = rng.sample(G.elements, count)
                differences = [
                    (g[0]-h[0]) % G.q
                    for g in cell for h in cell if g != h]
                cross = ([] if i == 0 else [
                    (g[0]-h[0]) % G.q
                    for g in grid[0][j] for h in cell])
                penalty = (
                    internal_weight*(len(differences)-len(set(differences)))
                    + row_weight*sum(row_used[i][d] for d in differences)
                    + cross_weight*(len(cross)-len(set(cross))
                                    + sum(cross_used[d] for d in cross)))
                key = (penalty, rng.random())
                if winner is None or key < winner[0]:
                    winner = (key, cell, differences, cross)
                if penalty == 0:
                    break
            _, cell, differences, cross = winner
            grid[i][j] = cell
            row_used[i].update(differences)
            cross_used.update(cross)
    return grid


def _component_rows(hx, hz, q):
    """Restrict the VERIFIED matrices to two sector-1 component slices.

    A slice: B-column zero, all A-columns, checked by HX.
    B slice: A-column zero, all B-columns, checked by HZ.
    No independently implemented left/right lift is used.
    """
    low = (1 << q) - 1
    ar = []
    for row in hx:
        v = 0
        for j in range(3):
            v |= ((row >> (3 * j * q)) & low) << (j * q)
        if v:
            ar.append(v)
    mask = (1 << (3 * q)) - 1
    br = [row & mask for row in hz if row & mask]
    return ar, br


def _syndrome_columns(rows, length):
    cols = [0] * length
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= bit
            row ^= lo
    return cols


def _four_cycles(cols):
    """Exact same-type four-cycle count in this restricted component."""
    seen = {}
    total = 0
    for syn in cols:
        active = []
        while syn:
            lo = syn & -syn
            active.append(lo.bit_length() - 1)
            syn ^= lo
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                key = (a, b)
                old = seen.get(key, 0)
                total += old
                seen[key] = old + 1
    return total


def _c43_syndrome(cols, word):
    syndrome = 0
    while word:
        bit = word & -word
        syndrome ^= cols[bit.bit_length()-1]
        word ^= bit
    return syndrome


def _c43_kernel_words(rng, cols, trials, seeds=()):
    """Retain actual nonzero kernels; no high reading certifies distance."""
    retained = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(len(cols)))
    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)

        words = sorted(set(dependencies+retained),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(words)
        for sweep in range(2):
            reducers = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
            discoveries.update(words)

        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:24]
        for a, word in enumerate(short):
            for other in short[:a]:
                combined = word ^ other
                if combined:
                    discoveries.add(combined)
        if trial % 4 == 0:
            for a, word in enumerate(short[:8]):
                for b in range(a):
                    for c in range(b):
                        combined = word ^ short[b] ^ short[c]
                        if combined:
                            discoveries.add(combined)
        retained = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:32]
        if retained and retained[0].bit_count() <= 4:
            break
    return tuple(retained)


@__import__("functools").lru_cache(maxsize=1)
def _c43_atlas(G):
    """Compile unit-cell actions exclusively from VERIFIED scaffold lifts."""
    assert G.q == 43
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[] for _ in range(3)] for _ in range(2)]
            B = [[[] for _ in range(3)] for _ in range(2)]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = _component_rows(hx, hz, G.q)
            rows = ar if side == "A" else br
            block = tuple(_syndrome_columns(rows, G.q))
            assert len(rows) == G.q
            assert all(column.bit_count() == 1 for column in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)
    return atlas


def _c43_component(G, grid, side):
    """F2-linear superposition of scaffold-generated unit-cell actions."""
    q = G.q
    atlas = _c43_atlas(G)[side]
    cols = [0]*(3*q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                block = atlas[G.idx(g)]
                for h, syndrome in enumerate(block):
                    cols[j*q+h] ^= syndrome << (i*q)
    return cols


def _c43_cofactors(G, grid, cols, side):
    """Commutative minors construct seeds, not an alternative matrix lift."""
    entries = [[set(G.inv(g) if side == "B" else g for g in cell)
                for cell in row] for row in grid]

    def product(left, right):
        result = set()
        for g in left:
            for h in right:
                term = G.mul(g, h)
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
        return result

    minors = []
    for omitted in range(3):
        s, t = [j for j in range(3) if j != omitted]
        minors.append(product(entries[0][s], entries[1][t])
                      ^ product(entries[0][t], entries[1][s]))
    seeds = []
    for shift in G.elements[::7]:
        word = 0
        for j, minor in enumerate(minors):
            for g in minor:
                word ^= 1 << (j*G.q+G.idx(G.mul(g, shift)))
        if word and _c43_syndrome(cols, word) == 0:
            seeds.append(word)

    # Also test all nonzero combinations of constant block indicators.
    mask = (1 << G.q)-1
    for blocks in range(1, 8):
        word = sum(mask << (j*G.q)
                   for j in range(3) if blocks & (1 << j))
        if _c43_syndrome(cols, word) == 0:
            seeds.append(word)
    return tuple(seeds)


def _c43_orbit_words(cols, words, q):
    """Translate supports, verify syndromes, then shorten orbit pairs."""
    mask = (1 << q)-1
    cache = {}

    def images(word):
        if word in cache:
            return cache[word]
        valid = []
        for shift in range(q):
            full = 0
            for j in range(3):
                block = (word >> (j*q)) & mask
                # This rotates C43 support labels, not parity-matrix rows.
                rotated = ((block << shift)
                           | (block >> (q-shift))) & mask
                full |= rotated << (j*q)
            if full and _c43_syndrome(cols, full) == 0:
                valid.append(full)
        assert valid, "identity translate must preserve a kernel witness"
        result = tuple(set(valid))
        cache[word] = result
        return result

    parents = sorted({min(images(w)) for w in words},
                     key=lambda w: (w.bit_count(), w))[:16]
    if not parents:
        return ()
    translated = set()
    for word in parents:
        translated.update(images(word))
    discoveries = set(parents)
    ceiling = parents[0].bit_count()+4
    for word in parents:
        for other in translated:
            combined = word ^ other
            if combined and combined.bit_count() <= ceiling:
                discoveries.add(combined)
    representatives = {
        min(images(w)) for w in sorted(
            discoveries, key=lambda w: (w.bit_count(), w))[:64]}
    return tuple(sorted(representatives,
                        key=lambda w: (w.bit_count(), w))[:32])


def _c43_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _c43_outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c43_logical_reading(rng, checks, stabilizers, n, seeds, trials):
    """Actual non-stabilizer witnesses, including mixed-sector dressings."""
    span = _c43_basis(stabilizers)
    cols = _syndrome_columns(checks, n)
    best = n+1
    logicals = set()

    def observe(word):
        nonlocal best
        if word and _c43_outside(word, span):
            best = min(best, word.bit_count())
            logicals.add(word)

    for word in seeds:
        if _c43_syndrome(cols, word) == 0:
            observe(word)

    order = list(range(n))
    memory = []
    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                observe(word)

        short = sorted(set(dependencies+memory),
                       key=lambda w: (w.bit_count(), w))[:32]
        discoveries = set(short)
        for a, word in enumerate(short):
            for other in short[:a]:
                combined = word ^ other
                if combined:
                    observe(combined)
                    discoveries.add(combined)

        # A stabilizer addition preserves the seed's nonzero logical class.
        for word in sorted(logicals,
                           key=lambda w: (w.bit_count(), w))[:6]:
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda entry: entry[0])
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                moves = [row for _, row in nearby[:12]]
                for a, row in enumerate(moves):
                    for other in moves[:a]:
                        combined = word ^ row ^ other
                        cw = combined.bit_count()
                        if cw < weight:
                            chosen, weight = combined, cw
                if chosen == word:
                    break
                word = chosen
                observe(word)
            discoveries.add(word)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        logicals = set(sorted(logicals,
                              key=lambda w: (w.bit_count(), w))[:24])
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """One degree-distinguished task, independently optimized on both sides."""
    assert G.q == 43
    q, mask = G.q, (1 << G.q)-1
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa = [list(ta[:3]), list(ta[3:])]
    sb = [list(tb[:3]), list(tb[3:])]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def component_key(item):
        d = min(item["d"], 36)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"], -item["rank"])

    def component_pool(side, counts):
        strata, seen = {}, set()

        def admit(grid):
            labels = tuple(tuple(sorted(cell))
                           for row in grid for cell in row)
            if labels in seen:
                return
            seen.add(labels)
            cols = _c43_component(G, grid, side)
            seeds = _c43_cofactors(G, grid, cols, side)
            words = _c43_kernel_words(rng, cols, 2, seeds)
            if not words:
                return
            rank = gf2_rank(cols)
            item = dict(
                grid=grid, cols=cols, words=words,
                d=words[0].bit_count(), rank=rank,
                cycles=_four_cycles(cols))
            bucket = strata.setdefault(rank, [])
            bucket.append(item)
            bucket.sort(key=component_key, reverse=True)
            del bucket[3:]

        for attempt in range(48):
            policy = (idx+attempt//12+int(side == "B")) % 4
            admit(_packed_grid(rng, G, counts, policy))

        for step in range(24):
            available = [item for bucket in strata.values() for item in bucket]
            if not available:
                admit(_packed_grid(rng, G, counts, step % 4))
                continue
            rank = rng.choice(sorted(strata))
            parent = rng.choice(strata[rank])
            current = tuple(len(cell)
                            for row in parent["grid"] for cell in row)
            policy = (idx+step+int(side == "B")) % 4
            if step % 8 == 7:
                # Whole-component count transport preserves its margins.
                admit(_packed_grid(rng, G, current, policy))
                continue
            grid = [[list(cell) for cell in row]
                    for row in parent["grid"]]
            proposal = _packed_grid(
                rng, G, current, policy, transport=False)
            pressure = [
                1 + sum(((word >> (j*q)) & mask).bit_count()
                        for word in parent["words"][:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            i = rng.randrange(2)
            grid[i][j] = proposal[i][j]
            if step % 6 == 5:
                other = (j+1+rng.randrange(2)) % 3
                grid[1-i][other] = proposal[1-i][other]
            admit(grid)

        # Preserve component-rank diversity before filling spare positions.
        first = sorted((bucket[0] for bucket in strata.values()),
                       key=component_key, reverse=True)
        pool = first[:6]
        if len(pool) < 6:
            extras = sorted(
                (item for bucket in strata.values() for item in bucket[1:]),
                key=component_key, reverse=True)
            pool.extend(extras[:6-len(pool)])
        for item in pool:
            words = _c43_kernel_words(
                rng, item["cols"], 32, item["words"])
            words = _c43_orbit_words(item["cols"], words, q)
            words = _c43_kernel_words(rng, item["cols"], 24, words)
            item["words"] = words
            item["d"] = min(item["d"], words[0].bit_count())
        pool.sort(key=component_key, reverse=True)
        return pool

    ap, bp = component_pool("A", ta), component_pool("B", tb)
    proposals = []
    for ai in ap:
        for bi in bp:
            A, B = ai["grid"], bi["grid"]
            ca = [[len(cell) for cell in row] for row in A]
            cb = [[len(cell) for cell in row] for row in B]
            assert _grid_weights(ca, cb) <= 12
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (q <= k and 8*k < n):
                continue

            # Confirm the cached action compiler against this full lift.
            ar, br = _component_rows(hx, hz, q)
            assert _syndrome_columns(ar, 3*q) == ai["cols"]
            assert _syndrome_columns(br, 3*q) == bi["cols"]
            d = min(ai["d"], bi["d"], 36)
            load = sum(w.bit_count() <= d+1
                       for component in (ai, bi)
                       for w in component["words"])
            proposals.append(dict(
                A=A, B=B, ai=ai, bi=bi, hx=hx, hz=hz, n=n, k=k,
                component_d=d, cycles=ai["cycles"]+bi["cycles"],
                key=(k*d*d, d, -load, -ai["cycles"]-bi["cycles"])))

    if not proposals:
        raise ValueError("no assembled rank-legal C43 filling")
    proposals.sort(key=lambda item: item["key"], reverse=True)
    finalists = proposals[:4]

    for item in finalists:
        a_seeds = [
            sum(((word >> (j*q)) & mask) << ((3*j+fixed)*q)
                for j in range(3))
            for word in item["ai"]["words"][:16]
            for fixed in range(3)]
        b_seeds = [
            word << (fixed*3*q)
            for word in item["bi"]["words"][:16]
            for fixed in range(3)]
        # Every finalist receives the same fresh full-code search budget.
        dz = _c43_logical_reading(
            rng, item["hx"], item["hz"], item["n"], a_seeds, 6)
        dx = _c43_logical_reading(
            rng, item["hz"], item["hx"], item["n"], b_seeds, 6)
        item["logical"] = min(dx, dz)

    def merit(item):
        # Classical kernel words can embed as stabilizers: retain their
        # readings only as conservative selection heuristics.
        d = min(item["component_d"], item["logical"], 36)
        return (item["k"]*d*d/item["n"], d,
                item["logical"], -item["cycles"])

    winner = max(finalists, key=merit)
    A, B = winner["A"], winner["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n-gf2_rank(hx)-gf2_rank(hz)
    assert n == 559 and q <= k and 8*k < n
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    ca = [[len(cell) for cell in row] for row in A]
    cb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(ca, cb) <= 12 and _no_dead_lines(ca, cb)
    assert all(len(cell) == len(set(cell)) <= 5
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection": merit(winner)}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell, with no subsequent redraw."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: High check weight alone excludes light logicals; parent slots 24, 46 and 48 had w=12 but d2s=2.
# REFUTED: Single-C3-coset cell concentration is competitive with mixed-coset filling on C13:C3 at (2,3)x(2,3); all parent mode-one slots had d2s<=13, whereas the five measured mixed-coset leaders retained d30s=24.
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
