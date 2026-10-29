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
CONCEPT = ("For abelian C9xC3, order 27, shapes (2,4)x(2,4), distinct order-controlled relative supports and balanced ambiguity screening should improve mean worse-basis circuit fitness above 7.1491. The test concerns circuit failures, not distance as fitness.")
GROUP = ("Jump from nonabelian C7:C4 to the direct product C9xC3, an unlisted group/shape bed. Related cyclic C27 reached 7.2133 and the listed nonabelian order-27 bed about 7.0; these do not establish an abelian advantage. C9xC3 has no involutions, permits order-three/order-nine relative supports, and gives n=540.")
SHAPES = ("Retain equal (2,4) components with 108 variables and 54 checks each. At q=27, n=540 and kb=108 give guaranteed rate 0.200; verified full ranks require 108<=k<122. Component rows total six and columns three, giving both CSS check-weight sums nine. Parent failure maps implicated several sector-one Acol/Bcol blocks, including Bcol1, but only 0-8 failures per measured basis do not establish an isolated defective block.")
FILLING = ("Twenty briefs cross four balanced support pairs with five allocations of order-three doubletons; remaining doubletons have order nine. A deterministic relative-support book removes obvious group-automorphism, row/column and side-exchange copies, without claiming arbitrary Tanner nonisomorphism. Repeated ordered differences within a row are forbidden; trinomials must generate C9xC3. Translate complete cells while preserving each brief, then use verified singleton actions to screen connectivity, ranks, kernels through four, exact six-word multiplicity and sampled ambiguity. Full verified matrices screen CSS, target rate and logicals through four. Check weight remains nine and six-round CNOT count falls from 24192 to 23328. Parent slot1 had eight X failures and 282 logical flips, versus slot0's zero X failures at the same n,k,w,d2s,depth and CNOT count; collective ambiguity and extraction exposure, not a larger distance upper bound, are the hypotheses.")
FAMILY_NAME = "r020_C9xC3_24x24_relativebook_balanced9_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 27

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C9xC3 requires q=27")
    # Direct product: additive coordinates modulo nine and three.
    els = [(a, b) for b in range(3) for a in range(9)]

    def red(g):
        return (int(g[0]) % 9, int(g[1]) % 3)

    def mul(g, h):
        return ((g[0] + h[0]) % 9, (g[1] + h[1]) % 3)

    def inv(g):
        return ((-g[0]) % 9, (-g[1]) % 3)

    def idx(g):
        a, b = red(g)
        return 9 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _legacy_unused_D29_ctx(rng, G, idx):
    """Retained historical helper; current slots never call this function."""
    from itertools import combinations
    from math import cos, sin, pi, hypot

    q, m = G.q, 29
    nc = 3 * q
    mask = (1 << m) - 1
    cmask = (1 << nc) - 1
    unit = [[[G.e]]]
    els = G.elements
    mul = [[G.idx(G.mul(g, h)) for h in els] for g in els]
    inv = [G.idx(G.inv(g)) for g in els]

    # Rotation-step triples modulo all multipliers and column exchanges.
    # The quotient has 26 geometries; slots choose different geometry
    # pairs, not automorphism images of one saved filling.
    representatives = set()
    for ds in combinations(range(1, 15), 3):
        images = []
        for u in range(1, m):
            images.append(tuple(sorted(
                min((u * d) % m, m - (u * d) % m) for d in ds)))
        representatives.add(min(images))
    book = sorted(representatives)
    a = idx % len(book)
    b = (a + 7 + idx // len(book)) % len(book)

    # One representative of every triple of component variables under
    # common right multiplication. A triple has trivial stabilizer:
    # its size is three and three does not divide |D29|=58.
    triple_orbits = []
    for block in range(3):
        anchor = block * q
        for u in range(anchor + 1, nc):
            for v in range(u + 1, nc):
                triple = (anchor, u, v)
                canonical = True
                for x in triple:
                    if x // q != block:
                        continue
                    h = inv[x % q]
                    image = tuple(sorted(
                        (y // q) * q + mul[y % q][h] for y in triple))
                    if image < triple:
                        canonical = False
                        break
                if canonical:
                    triple_orbits.append(triple)
    assert len(triple_orbits) * q == nc * (nc - 1) * (nc - 2) // 6

    trig = [
        ([cos(2 * pi * k * t / m) for t in range(m)],
         [sin(2 * pi * k * t / m) for t in range(m)])
        for k in range(1, 15)
    ]

    def rotate(word, shift):
        shift %= m
        return ((word << shift) | (word >> (m - shift))) & mask

    def action(word, h):
        """Permute check labels by right multiplication; not a lift."""
        lo, hi = word & mask, word >> m
        aa, bb = h
        lo, hi = rotate(lo, aa), rotate(hi, -aa)
        if bb:
            lo, hi = hi, lo
        return lo | (hi << m)

    def canonical_syndrome(word):
        """Normalize at each occupied check; take the smallest image.

        An odd syndrome of weight <=9 cannot have a nontrivial
        stabilizer in D29. Its orbit therefore contains exactly 58
        syndromes, just like an orbit of variable triples.
        """
        lo, hi = word & mask, word >> m
        answer = None
        occupied = lo
        while occupied:
            bit = occupied & -occupied
            t = bit.bit_length() - 1
            image = rotate(lo, -t) | (rotate(hi, t) << m)
            if answer is None or image < answer:
                answer = image
            occupied ^= bit
        occupied = hi
        while occupied:
            bit = occupied & -occupied
            t = bit.bit_length() - 1
            image = rotate(hi, -t) | (rotate(lo, t) << m)
            if answer is None or image < answer:
                answer = image
            occupied ^= bit
        return answer

    def columns(rows, size):
        result = [0] * size
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def component(grid, side):
        """Extract the actual component from the verified scaffold lift."""
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = [row & cmask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            source = [row & cmask for row in hz]
            # Invert both variable and check group coordinates. This
            # permutes the verified right-action component into the
            # left-action coordinates used by the orbit census.
            rows = []
            for r in range(q):
                word = source[inv[r]]
                permuted = 0
                while word:
                    bit = word & -word
                    c = bit.bit_length() - 1
                    permuted |= 1 << ((c // q) * q + inv[c % q])
                    word ^= bit
                rows.append(permuted)
        return rows, columns(rows, nc)

    def no_four(cols):
        # All columns have odd weight three, excluding odd kernels.
        if len(set(cols)) != nc or any(x.bit_count() != 3 for x in cols):
            return False
        seen = set()
        for i, x in enumerate(cols):
            for y in cols[:i]:
                syndrome = x ^ y
                if syndrome == 0 or syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def graph_statistics(rows):
        """Actual check overlaps, Tanner cycles and nontrivial spectrum."""
        overlap = [[0] * q for _ in range(q)]
        four = squares = triangles = 0
        for i in range(q):
            for j in range(i):
                w = (rows[i] & rows[j]).bit_count()
                overlap[i][j] = overlap[j][i] = w
                four += w * (w - 1) // 2
                squares += w * w
        for i in range(q):
            for j in range(i + 1, q):
                ij = overlap[i][j]
                if ij:
                    for k in range(j + 1, q):
                        triangles += ij * overlap[i][k] * overlap[j][k]
        # Each degree-three variable supplies one non-Tanner triangle.
        six = triangles - squares + 2 * nc
        assert six >= 0

        # Fourier blocks of this verified D29-equivariant overlap graph.
        # The trivial degree eigenvalue is excluded; retain the sign
        # representation and both eigenvalues of every 2D block.
        weights = overlap[0]
        rotation_mass = sum(weights[:m])
        reflection_mass = sum(weights[m:])
        radius = abs(rotation_mass - reflection_mass)
        for cs, ss in trig:
            diagonal = sum(weights[t] * cs[t] for t in range(m))
            real = sum(weights[m + t] * cs[t] for t in range(m))
            imag = sum(weights[m + t] * ss[t] for t in range(m))
            radius = max(radius, abs(diagonal) + hypot(real, imag))
        return four, six, round(radius, 9)

    def six_kernels(cols, limit):
        """Exact weight-six count, after excluding smaller kernels.

        Equal canonical triple syndromes correspond to 58 ordinary
        syndrome collisions. Every six-set has ten unordered triple
        partitions. Orbit counting avoids enumerating 862924 triples.
        """
        buckets = {}
        collisions = 0
        boundary = [0] * 10
        for i, j, k in triple_orbits:
            syndrome = cols[i] ^ cols[j] ^ cols[k]
            weight = syndrome.bit_count()
            assert weight in (1, 3, 5, 7, 9)
            boundary[weight] += 1
            key = canonical_syndrome(syndrome)
            old = buckets.get(key, 0)
            collisions += old
            if limit is not None and collisions > limit:
                return None
            buckets[key] = old + 1
        assert (q * collisions) % 10 == 0
        return collisions, q * collisions // 10, tuple(boundary)

    def select(side, geometry):
        # A common reflection translation is gauge: fix the first t=0.
        # Search the remaining offsets, not translated copies.
        parameters = list(combinations(range(m), 2))
        parameters += [(v, u) for u, v in parameters]
        parameters += [(t, t) for t in range(m)]
        rng.shuffle(parameters)
        pool = []
        for t1, t2 in parameters:
            ts = (0, t1, t2)
            # For {1,x^d,x^t y}, the two involutory check differences
            # are x^t y and x^(t+d) y.
            reflection_differences = {
                value for d, t in zip(geometry, ts)
                for value in (t, (t + d) % m)
            }
            if len(reflection_differences) != 6:
                continue
            grid = [[
                [G.e, (d, 0), (t, 1)]
                for d, t in zip(geometry, ts)
            ]]
            rows, cols = component(grid, side)
            if gf2_rank(rows) != q or not no_four(cols):
                continue
            four, cycles6, radius = graph_statistics(rows)
            # Two forced involutory cycles per cell, with no additional
            # overlap: 3 cells * 2 * (q/2) = 3q.
            if four != 3 * q:
                continue
            pool.append(((radius, cycles6, ts), grid, cols))
            if len(pool) >= 160:
                break
        if not pool:
            raise ValueError("no minimally overlapping mixed component")
        pool.sort(key=lambda item: item[0])

        best = None
        best_key = None
        for cheap, grid, cols in pool[:32]:
            # Confirm the orbit action against columns of the verified
            # lift, including the permuted right-action B component.
            for h in ((1, 0), (0, 1)):
                hidx = G.idx(h)
                for block in range(3):
                    for c in range(q):
                        assert action(cols[block * q + c], h) == (
                            cols[block * q + mul[c][hidx]])
            census = six_kernels(
                cols, None if best_key is None else best_key[0])
            if census is None:
                continue
            collisions, words6, boundary = census
            key = (collisions, cheap[0], cheap[1],
                   boundary[1], boundary[3], cheap[2])
            if best_key is None or key < best_key:
                best_key = key
                best = (grid, words6, cheap[0], cheap[1])
            # Pool order protects expansion among the screened
            # zero-weight-six components.
            if collisions == 0:
                break
        if best is None:
            raise ValueError("component orbit screening found no candidate")
        return best

    aa = select("A", book[a])
    bb = select("B", book[b])
    A, B = aa[0], bb[0]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k != 232 or not css_ok(hx, hz):
        raise ValueError("verified complete-code rank or CSS check failed")
    assert max(row.bit_count() for row in hx + hz) == 12
    return {"idx": idx, "cells": {"A": A, "B": B},
            "geometry": (book[a], book[b]),
            "component_screen": (aa[1:], bb[1:])}


def _legacy_unused_r040_ctx(rng, G, idx):
    """Historical asymmetric search; the balanced-shape slots never call it."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 28 and A_SHAPE == (1, 3) and B_SHAPE == (2, 6)
    design, batch = idx % 20, idx // 20
    amodel, bmodel = divmod(design, 5)
    P, Q = (4, 4, 7), (4, 4, 14)
    R, T = (7, 7, 7), (7, 14, 14)
    profile_batches = (
        ((P, P, P), (Q, Q, Q), (P, P, Q), (P, Q, T)),
        ((P, Q, Q), (P, P, T), (Q, Q, T), (P, Q, R)),
        ((P, P, R), (Q, Q, R), (P, T, T), (Q, T, T)),
    )
    abrief = profile_batches[batch % 3][amodel]
    sa = [[3, 3, 3]]
    if bmodel == 4:
        sb = [[3, 2, 2, 0, 1, 1],
              [0, 1, 1, 3, 2, 2]]
    else:
        sb = [[2, 2, 2, 1, 1, 1],
              [1, 1, 1, 2, 2, 2]]
    counts = {"A": sa, "B": sb}
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert all(sum(row) == 9 for grid in counts.values() for row in grid)
    assert all(sum(row[j] for row in sb) == 3 for j in range(6))

    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if G._key(power) == G._key(G.e):
                orders[G._key(g)] = order
                break
        else:
            raise ValueError("element order missing")
    relatives = {
        order: [g for g in G.elements if orders[G._key(g)] == order]
        for order in (4, 7, 14)
    }
    triple_pools = {}
    nonidentity = [g for g in G.elements if G._key(g) != G._key(G.e)]
    for u, v in combinations(nonidentity, 2):
        pattern = tuple(sorted((
            orders[G._key(u)], orders[G._key(v)],
            orders[G._key(G.mul(G.inv(u), v))])))
        if pattern in (P, Q, R, T):
            triple_pools.setdefault(pattern, []).append((u, v))
    assert all(triple_pools.get(pattern) for pattern in (P, Q, R, T))

    # These are exact evaluator actions, not a separately implemented lift.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in hx)
        actions["B"][G._key(g)] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if not weight:
            return []
        base = G.pick(rng)
        if weight == 1:
            return [base]
        if weight == 3:
            pattern = (abrief[j] if side == "A"
                       else (P if (i + batch) % 2 == 0 else Q))
            u, v = rng.choice(triple_pools[pattern])
            relative = (G.e, u, v)
        else:
            if bmodel == 0:
                order = 14 if batch % 3 == 0 or i == 0 else 7
            elif bmodel == 1:
                order = 7 if batch % 3 != 2 or i == 0 else 14
            elif bmodel == 2:
                order = 4
            elif bmodel == 3:
                order = (4, 7, 14)[(j + 2 * i + batch) % 3]
            else:
                order = (7, 14)[(j + i + batch) % 2]
            relative = (G.e, rng.choice(relatives[order]))
        cell = [G.mul(base, g) for g in relative]
        assert len({G._key(g) for g in cell}) == weight
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def component(side, grid):
        result = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                shift = j * q
                for g in cell:
                    for t, word in enumerate(actions[side][G._key(g)]):
                        out[t] ^= word << shift
            result.extend(out)
        return result

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def four_cycles(rows):
        return sum(c * (c - 1) // 2
                   for i, row in enumerate(rows) for other in rows[:i]
                   for c in [(row & other).bit_count()])

    def proposal_statistics(rows, width):
        """Exact C4; check-graph triangle surrogate when C4 is nonzero."""
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                shared = (row & rows[j]).bit_count()
                four += shared * (shared - 1) // 2
                if shared:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (neighbors & adjacency[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        supplied = sum(d * (d - 1) * (d - 2) // 6
                       for d in (c.bit_count() for c in columns(rows, width)))
        return four, max(0, triangles - supplied)

    def connected(cols, nr):
        reached = 1
        while True:
            old = reached
            for col in cols:
                if col & reached:
                    reached |= col
            if reached == old:
                break
        return reached.bit_count() == nr and all(cols)

    def small_kernels(cols):
        """Odd column weights exclude odd kernels; count pair collisions."""
        two = len(cols) - len(set(cols))
        pairs = {}
        four = 0
        for i, x in enumerate(cols):
            for y in cols[:i]:
                syndrome = x ^ y
                previous = pairs.get(syndrome, 0)
                four += previous
                pairs[syndrome] = previous + 1
        return two, four

    def six_kernels(cols):
        """Exact weight-six multiplicity, only after weights <=4 exclusion."""
        syndromes = {}
        collisions = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    previous = syndromes.get(syndrome, 0)
                    collisions += previous
                    syndromes[syndrome] = previous + 1
        assert collisions % 10 == 0
        return collisions // 10

    def boundary_exposure(cols):
        """Anchored triple-boundary exposure, not a trapping certificate."""
        loads = []
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
                    if weight <= 3:
                        cost += (4 - weight) ** 2
            loads.append(cost)
        return max(loads), sum(loads)

    def sampled_words(cols, side):
        """Fundamental and reduced kernel words: upper bounds only."""
        srng = random.Random(193771 + 1009 * idx + (side == "B"))
        permutation = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()
        for trial in range(16):
            srng.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome and word:
                    basis.append(word)
            srng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    short.add(word)
                for other in basis[:i]:
                    candidate = word ^ other
                    if not candidate:
                        continue
                    w = candidate.bit_count()
                    minimum = min(minimum, w)
                    if w <= 10 and len(short) < 8192:
                        short.add(candidate)
                    if w < weight:
                        word, weight = candidate, w
                basis[i] = word
        return minimum, len(short)

    def portfolio(side):
        support = counts[side]
        width = len(support[0]) * q
        positions = [(i, j) for i, row in enumerate(support)
                     for j, weight in enumerate(row) if weight]
        pool, seen = [], set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            if len(pool) >= 6 and stats > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((stats, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(2):
            grid = draw_grid(side)
            rows = component(side, grid)
            stats = proposal_statistics(rows, width)
            energy = (80 * stats[0] + stats[1] / 8) / q
            remember(grid, rows, stats)
            for step in range(180):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows = component(side, grid)
                proposed = proposal_statistics(proposed_rows, width)
                new_energy = (80 * proposed[0] + proposed[1] / 8) / q
                temperature = 5 * (0.978 ** step) + 0.06
                delta = new_energy - energy
                if delta <= 0 or rng.random() < math.exp(
                        -min(700, delta / temperature)):
                    rows, stats, energy = proposed_rows, proposed, new_energy
                    remember(grid, rows, stats)
                else:
                    grid[i][j] = old

        entries = []
        for stats, serial, grid, rows in pool[:4]:
            cols = columns(rows, width)
            assert all(c.bit_count() == 3 for c in cols)
            two, four = small_kernels(cols)
            words6 = six_kernels(cols) if not (two or four) else 1 << 60
            distance, short = sampled_words(cols, side)
            peak, total = boundary_exposure(cols)
            deficit = len(rows) - gf2_rank(rows)
            disconnected = int(not connected(cols, len(rows)))
            key = (deficit, disconnected, two, four, words6,
                   max(0, 8 - distance), peak, total, short,
                   stats[0], stats[1])
            entries.append({
                "key": key, "grid": grid, "rows": rows,
                "bad": two + four, "six": words6,
                "distance": distance, "short": short,
                "boundary": (peak, total), "disconnected": disconnected,
            })
        entries.sort(key=lambda item: item["key"])
        if not entries:
            raise ValueError("empty component portfolio")
        return entries[:3]

    def no_small_logical(checks, stabilizers, n):
        """Exact exclusion through weight four modulo the stabilizer span."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p not in pivots:
                    pivots[p] = row
                    break
                row ^= pivots[p]

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is not None and nontrivial(word ^ previous):
                return False
            seen.setdefault(syndrome, word)
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is not None and nontrivial(word ^ previous):
                    return False
                seen.setdefault(syndrome, word)
        return True

    for attempt in range(2):
        apool, bpool = portfolio("A"), portfolio("B")
        options = []
        for a in apool:
            for b in bpool:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 560 and 375 * n <= 1000 * k < 425 * n):
                    continue
                fx, fz = four_cycles(hx), four_cycles(hz)
                # Scale component witnesses by their sector-1 repetitions.
                wa, wb = 6 * a["six"], 3 * b["six"]
                pa, ta = a["boundary"]
                pb, tb = b["boundary"]
                key = (
                    a["bad"] + b["bad"],
                    a["disconnected"] + b["disconnected"],
                    max(wa, wb), wa + wb,
                    max(0, 8 - a["distance"]) + max(0, 8 - b["distance"]),
                    max(pa, pb), ta + tb,
                    max(fx, fz), fx + fz,
                    a["short"] + b["short"], k - 224,
                )
                options.append((key, A, B, hx, hz, n, k, a, b))
        options.sort(key=lambda item: item[0])
        for key, A, B, hx, hz, n, k, a, b in options:
            if not no_small_logical(hx, hz, n):
                continue
            if not no_small_logical(hz, hx, n):
                continue
            # Confirm cached composition against complete verified probes.
            unit = [[[G.e]]]
            ar = lifted_matrices(G, A, unit)[0]
            br = lifted_matrices(G, unit, B)[1]
            assert [r & ((1 << (3 * q)) - 1) for r in ar] == a["rows"]
            assert [r & ((1 << (6 * q)) - 1) for r in br] == b["rows"]
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
            assert 6 * sum(row.bit_count() for row in hx + hz) == 24192
            return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
                    "contract": (abrief, bmodel, batch),
                    "screen": key}
    raise ValueError("contract missed verified target-rate CSS screening")


def NEW_CTX(rng, G, idx):
    """Indexed C9xC3 relative geometries; search both verified components."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 27 and A_SHAPE == B_SHAPE == (2, 4)
    design, batch = idx % 20, idx // 20
    architecture, policy = divmod(design, 5)
    profiles = (
        ((2, 2, 1, 1), (1, 1, 2, 2)),
        ((3, 0, 2, 1), (0, 3, 1, 2)),
        ((0, 2, 2, 2), (3, 1, 1, 1)),
    )
    pair_cards = ((0, 0), (0, 1), (1, 1), (0, 2))
    ia, ib = pair_cards[architecture]
    counts = {
        "A": [list(row) for row in profiles[ia]],
        "B": [list(row) for row in profiles[ib]],
    }
    assert _no_dead_lines(counts["A"], counts["B"])
    assert _grid_weights(counts["A"], counts["B"]) == 9
    assert 9 <= W_MAX
    for side, profile in (("A", ia), ("B", ib)):
        support = counts[side]
        assert all(sum(row) == 6 for row in support)
        assert all(sum(row[j] for row in support) == 3
                   for j in range(4))
        assert all(0 <= t <= min(3, CELL_MAX)
                   for row in support for t in row)
        # Profile two has augmentation rank ONE, not two.
        # Permit that known deficit; measure the complete-code k.
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in support
        ]
        assert gf2_rank(augmentation) == (1 if profile == 2 else 2)

    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if G._key(power) == G._key(G.e):
                orders[G._key(g)] = order
                break
        else:
            raise ValueError("element order missing")
    palettes = {
        order: [g for g in G.elements if orders[G._key(g)] == order]
        for order in (3, 9)
    }
    P, Q = (3, 9, 9), (9, 9, 9)
    triples = {P: [], Q: []}
    nonidentity = [g for g in G.elements
                   if G._key(g) != G._key(G.e)]
    for u, v in combinations(nonidentity, 2):
        signature = tuple(sorted((
            orders[G._key(u)], orders[G._key(v)],
            orders[G._key(G.mul(G.inv(u), v))],
        )))
        # Independence in G/3G proves that u,v generate C9xC3.
        determinant = (u[0] * v[1] - v[0] * u[1]) % 3
        if signature in triples and determinant:
            triples[signature].append((u, v))
    assert all(palettes.values()) and all(triples.values())

    # Exact linear composition of scaffold-built singleton actions.
    # No independent left/right lift implementation is used.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    unit = [[[G.e]]]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], unit)
        _, hz, _ = lifted_matrices(G, unit, [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in hx)
        actions["B"][G._key(g)] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Five deliberate order allocations, not five draws of one brief.
    # Counts are numbers of order-three doubletons in A and B.
    order_cards = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))

    # Every automorphism of C9xC3 has this triangular form on G/3G.
    # The image of the order-three generator has first coordinate 3v.
    automorphisms = []
    for u in (1, 2, 4, 5, 7, 8):
        for v in range(3):
            for w in range(3):
                for z in (1, 2):
                    automorphisms.append({
                        g: ((u * g[0] + 3 * v * g[1]) % 9,
                            (w * g[0] + z * g[1]) % 3)
                        for g in G.elements
                    })

    def normalized(cell, image):
        if not cell:
            return ()
        values = [image[g] for g in cell]
        return min(
            tuple(sorted(((a - origin[0]) % 9,
                          (b - origin[1]) % 3)
                         for a, b in values))
            for origin in values
        )

    def geometry_key(grids):
        """Conservative internal-geometry quotient, not Tanner classification."""
        images = []
        for automorphism in automorphisms:
            sides = []
            for grid in grids:
                columns = [
                    (normalized(grid[0][j], automorphism),
                     normalized(grid[1][j], automorphism))
                    for j in range(4)
                ]
                direct = tuple(sorted(columns))
                exchanged = tuple(sorted((b, a) for a, b in columns))
                sides.append(min(direct, exchanged))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Reconstruct an indexed book locally: no shared mutable cache.
    # Larger harness batches extend the book with fresh internal geometry.
    geometry_rng = random.Random(927311 + 1571 * architecture)
    wanted = 5 * batch + policy
    book, seen_geometries = [], set()
    for proposal in range(20000):
        card = len(book) % 5
        generation = len(book) // 5
        grids = []
        legal = True
        for side_number, side in enumerate(("A", "B")):
            support = counts[side]
            doubles = [(i, j) for i in range(2) for j in range(4)
                       if support[i][j] == 2]
            charged = set(geometry_rng.sample(
                doubles, order_cards[card][side_number]))
            grid = []
            for i in range(2):
                row, row_differences = [], set()
                for j in range(4):
                    weight = support[i][j]
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [G.e]
                    elif weight == 2:
                        order = 3 if (i, j) in charged else 9
                        cell = [G.e, geometry_rng.choice(palettes[order])]
                    else:
                        pattern = P if (
                            i + j + card + generation + side_number
                        ) % 2 == 0 else Q
                        u, v = geometry_rng.choice(triples[pattern])
                        cell = [G.e, u, v]

                    assert len(set(cell)) == weight
                    for x in cell:
                        for y in cell:
                            if x == y:
                                continue
                            difference = G.mul(G.inv(x), y)
                            if difference in row_differences:
                                legal = False
                            row_differences.add(difference)
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        if not legal:
            continue
        signature = geometry_key(grids)
        if signature in seen_geometries:
            continue
        seen_geometries.add(signature)
        book.append(grids)
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("C9xC3 relative-support book exhausted")
    selected_geometry = book[wanted]
    geometry = {
        "A": selected_geometry[0],
        "B": selected_geometry[1],
    }

    def draw_cell(side, i, j):
        # Preserve the slot's entire internal geometry during optimization.
        # Cell translations change inter-cell voltages, not term counts.
        base = G.pick(rng)
        cell = [G.mul(base, g) for g in geometry[side][i][j]]
        assert len(cell) == counts[side][i][j]
        assert len({G._key(g) for g in cell}) == len(cell)
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(4)]
                for i in range(2)]

    def component(side, grid):
        rows = []
        for row in grid:
            packed = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    for r, word in enumerate(actions[side][G._key(g)]):
                        packed[r] ^= word << (j * q)
            rows.extend(packed)
        return rows

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def graph_stats(rows):
        """Exact C4; triangle statistic is exact C6 when C4 is zero."""
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (neighbors & adjacency[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        # Every degree-three variable supplies one non-Tanner triangle.
        return four, max(0, triangles - 4 * q)

    def connected(cols, nr):
        reached = 1
        while True:
            previous = reached
            for col in cols:
                if col & reached:
                    reached |= col
            if reached == previous:
                return reached.bit_count() == nr and all(cols)

    def clean_four(cols):
        """Exact exclusion through weight four for degree-three columns."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return False
        pairs = set()
        for i, x in enumerate(cols):
            for y in cols[:i]:
                syndrome = x ^ y
                if syndrome == 0 or syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        return True

    def six_and_boundary(cols):
        """Exact six-word count after clean_four; boundary is a heuristic."""
        histogram = {}
        collisions = boundary = one = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    old = histogram.get(syndrome, 0)
                    collisions += old
                    histogram[syndrome] = old + 1
                    weight = syndrome.bit_count()
                    if weight <= 3:
                        boundary += (4 - weight) ** 2
                        one += int(weight == 1)
        # No order-three orbit assumption: enumerate every triple.
        assert collisions % 10 == 0
        return collisions // 10, one, boundary

    def sampled_words(cols, side):
        """Random fundamental/reduced kernel witnesses, not certified distance."""
        probe = random.Random(742031 + 1013 * idx + (side == "B"))
        permutation = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()
        for trial in range(24):
            probe.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome and word:
                    basis.append(word)
            probe.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    short.add(word)
                for other in basis[:i]:
                    candidate = word ^ other
                    if not candidate:
                        continue
                    w = candidate.bit_count()
                    minimum = min(minimum, w)
                    if w <= 10 and len(short) < 16384:
                        short.add(candidate)
                    if w < weight:
                        word, weight = candidate, w
                basis[i] = word
        return minimum, len(short)

    def portfolio(side):
        """Anneal fresh voltages within the slot's explicit support/order brief."""
        positions = [(i, j) for i, row in enumerate(counts[side])
                     for j, weight in enumerate(row) if weight]
        pool, seen = [], set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            if len(pool) >= 16 and stats > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((stats, serial, clone(grid), list(rows)))
            pool.sort(key=lambda entry: entry[:2])
            del pool[16:]

        for restart in range(2):
            grid = draw_grid(side)
            rows = component(side, grid)
            stats = graph_stats(rows)
            energy = (64 * stats[0] + stats[1] / 6) / q
            remember(grid, rows, stats)
            for step in range(360):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows = component(side, grid)
                proposed = graph_stats(proposed_rows)
                proposed_energy = (64 * proposed[0] + proposed[1] / 6) / q
                temperature = 3.0 * (0.984 ** step) + 0.08
                difference = proposed_energy - energy
                if difference <= 0 or rng.random() < math.exp(
                        -min(700, difference / temperature)):
                    rows, stats, energy = (
                        proposed_rows, proposed, proposed_energy)
                    remember(grid, rows, stats)
                else:
                    grid[i][j] = old

        eligible = []
        for stats, serial, grid, rows in pool:
            deficit = 2 * q - gf2_rank(rows)
            if deficit > 4:
                continue
            cols = columns(rows, 4 * q)
            assert all(c.bit_count() == 3 for c in cols)
            if not connected(cols, 2 * q) or not clean_four(cols):
                continue
            six, one, boundary = six_and_boundary(cols)
            preliminary = (deficit, six, one, boundary, stats[0], stats[1])
            eligible.append((preliminary, serial, grid, rows, cols))
        eligible.sort(key=lambda entry: entry[:2])

        refined = []
        for preliminary, serial, grid, rows, cols in eligible[:8]:
            distance, short = sampled_words(cols, side)
            deficit, six, one, boundary, four, cycles = preliminary
            # Sampled protection saturates at twelve. After that,
            # compare ambiguity exposure rather than chase optimistic d.
            quality = (deficit, int(six != 0), max(0, 12 - distance),
                       one, boundary, short, six, four, cycles)
            refined.append({
                "quality": quality, "grid": grid, "rows": rows,
                "distance": distance, "short": short,
                "six": six, "one": one, "boundary": boundary,
                "deficit": deficit, "four": four, "cycles": cycles,
            })
        refined.sort(key=lambda entry: entry["quality"])
        return refined[:3]

    def no_small_logical(checks, stabilizers, n):
        """Exact logical exclusion through four modulo actual stabilizers."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p not in pivots:
                    pivots[p] = row
                    break
                row ^= pivots[p]

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is not None and nontrivial(word ^ previous):
                return False
            seen.setdefault(syndrome, word)
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is not None and nontrivial(word ^ previous):
                    return False
                seen.setdefault(syndrome, word)
        return True

    for attempt in range(3):
        aa, bb = portfolio("A"), portfolio("B")
        if not aa or not bb:
            continue
        options = []
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 540 or not 108 <= k < 122:
                    continue
                key = (
                    max(a["deficit"], b["deficit"]),
                    a["deficit"] + b["deficit"],
                    int(bool(a["six"] or b["six"])),
                    max(0, 12 - min(a["distance"], b["distance"])),
                    max(a["one"], b["one"]), a["one"] + b["one"],
                    max(a["boundary"], b["boundary"]),
                    a["boundary"] + b["boundary"],
                    a["short"] + b["short"],
                    max(a["six"], b["six"]), a["six"] + b["six"],
                    max(a["four"], b["four"]), a["four"] + b["four"],
                    a["cycles"] + b["cycles"], k - 108,
                )
                options.append((key, A, B, hx, hz, n, k, a, b))
        options.sort(key=lambda entry: entry[0])

        for key, A, B, hx, hz, n, k, a, b in options:
            if not no_small_logical(hx, hz, n):
                continue
            if not no_small_logical(hz, hx, n):
                continue
            # Confirm composed singleton actions against complete,
            # evaluator-convention component probes.
            ar = lifted_matrices(G, A, unit)[0]
            br = lifted_matrices(G, unit, B)[1]
            component_mask = (1 << (4 * q)) - 1
            assert [row & component_mask for row in ar] == a["rows"]
            assert [row & component_mask for row in br] == b["rows"]
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 9
            assert 9 <= min(12, W_MAX)
            assert n == 540 and 100 <= n < 600
            assert 6 * sum(row.bit_count() for row in hx + hz) == 23328
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            return {
                "idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "contract": (ia, ib, order_cards[policy], batch),
                "relative_geometry": selected_geometry,
                "screen": key,
                "component_witnesses": (
                    (a["distance"], a["short"]),
                    (b["distance"], b["short"])),
            }
    raise ValueError("C9xC3 contract missed verified rate-0.200 screening")


def CELL(rng, G, side, i, j, ctx):
    """Expose the complete screened cell; never transform a saved design."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone give reliable r040f memory: parent fitness 3.2187, all five measured codes hit a failed-shot cap, with weight-4 sector-1 witnesses and broadly distributed logical flips.
# REFUTED: Balanced collision screening plus d2s=8 certifies robust circuit memory: current slot 0 still had 105/3500 X failures and slot 21 had 106/2500 Z failures, with logical flips spanning all 232 logical qubits.
# REFUTED: Minimal mixed-trinomial overlap, weight-six-kernel screening and improved component expansion suffice to avoid failed-shot caps: all five measured order-58 parent codes still hit those caps in both bases, with s=4.35..4.66 and broad logical flips despite d2s=8.
# REFUTED: Non-involutory C7:C4 ratios and balanced degree-three ambiguity on (1,3)x(2,6) suffice for balanced memory: fitness was 4.3111; every measured X basis reached the failed-shot cap in 1750-2500 shots, with flips across all 224 logicals and diffuse block enrichment.
# REFUTED: Equal n,k,w,d2s,ASC depth and CNOT count establish equal circuit reliability: current parent slots0 and1 both had (560,112,9,12,12,24192), but X failures were 0 versus 8 per 200000 shots and worse-basis scores were 7.382 versus 6.931.
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
