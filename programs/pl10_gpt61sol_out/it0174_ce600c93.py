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
CONCEPT = ("Cyclic C17, order 17, shapes (1,4)x(2,8): replacing the second single-row component with a balanced two-row component should improve worse-basis circuit fitness beyond 3.5010 by reducing short-logical multiplicity without increasing extraction work.")
GROUP = ("Keep the additive cyclic construction rule and one-coordinate encoding. The new size factor 34 makes order 35 illegal; order 17 is the largest compatible lift and gives n=578. This bed is absent from BEDS. Prime-order binomial ratios connect their lift orbits; the order change is required by size, not supported as a group-only performance improvement.")
SHAPES = ("Target cell 0.550 with A=(1,4), B=(2,8): n=578 and kb=306 guarantee rate 9/17. A has 17 checks on 68 variables; B has 34 checks on 136 variables. Require verified k=306. Parent weight-four witnesses occupied many sector-one Acol/Bcol blocks, and failure enrichment was mostly 1.0-1.1x without a persistent Acol/Brow hotspot, motivating global component restructuring.")
FILLING = ("A has three doubletons and one trinomial with disjoint internal directions and no binomial triangles. Five B architectures have row totals nine and column totals two or three, including cross-row monomials and zero/doubleton/trinomial columns. Both CSS weights are <=12, with 18360 six-round CNOTs versus 18900. Canonical A geometries and independently searched uncolored B component classes supply structural cards; reserve unused component-fingerprint pairs across slots. Verified full lifts rank finalists by worse-basis stabilizer-quotient short-logical and possible hook collisions. Target the parent's 105-109 failures per 250 shots and flips across all 315 logicals. Fingerprints are conservative invariants, and shared-check hooks are not ASC-ordering certificates or measured pL.")
FAMILY_NAME = "r055_c17_14x28_balanced_componentclass_hook_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This size-34 cyclic bed requires q=17")
    # Same additive cyclic construction and one-coordinate encoding.
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)


def _short_logical_counts(checks, stabilizers, n):
    """Exact counts of nontrivial logical supports of weights 1..4.
    Matrices come exclusively from the verified scaffold lift."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(v):
        while v:
            p = v.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            v ^= old
        return False

    columns = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= syndrome_bit
            row ^= bit

    singles = {}
    counts = [0, 0, 0, 0]
    bits = [1 << a for a in range(n)]
    for a, syndrome in enumerate(columns):
        singles.setdefault(syndrome, []).append(a)
        if syndrome == 0 and nontrivial(bits[a]):
            counts[0] += 1

    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            syndrome = columns[a] ^ columns[b]
            support = bits[a] | bits[b]
            if syndrome == 0 and nontrivial(support):
                counts[1] += 1

            # a < b < c counts each weight-three support once.
            for c in singles.get(syndrome, ()):
                if c > b and nontrivial(support | bits[c]):
                    counts[2] += 1

            bucket = pairs.get(syndrome)
            if bucket is None:
                pairs[syndrome] = [(a, b, support)]
            else:
                # c < d < a < b is the unique ordered partition
                # used to count each weight-four support once.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        counts[3] += 1
                bucket.append((a, b, support))
    return tuple(counts)


def _designed_side(rng, G, counts, side, model):
    """Singleton-free supports with four distinct ratio-order policies.

    For a single-row cyclic component, translating a whole cell is a
    column-block gauge. Fix its first term to zero instead of spending
    proposals on those relabelings.
    """
    pools = {
        35: [d for d in range(1, 18) if d % 5 and d % 7],
        7: [5, 10, 15],
        5: [7, 14],
    }
    nb = counts.count(2)
    signatures = (
        [35] * nb,
        [7] + [35] * (nb - 1),
        [5] + [35] * (nb - 1),
        [7, 5] + [35] * (nb - 2),
    )
    orders = signatures[model]
    differences = []
    for order in (35, 7, 5):
        differences.extend(rng.sample(pools[order], orders.count(order)))
    rng.shuffle(differences)
    cells = []
    t = 0
    for weight in counts:
        assert 2 <= weight <= 4
        if weight == 2:
            # d in 1..17 identifies an undirected C35 edge type.
            cell = [G.e, (differences[t],)]
            t += 1
        else:
            cell = [G.e] + rng.sample(G.elements[1:], weight - 1)
        cells.append(cell)
    return [cells]


def _c35_components(hx, hz, q):
    """Extract both components exclusively from the verified CSS lift."""
    mask = (1 << q) - 1
    # X checks at B-column zero contain the A-left component.
    ar = [
        sum(((hx[t] >> (j * 4 * q)) & mask) << (j * q)
            for j in range(4))
        for t in range(q)
    ]
    # Z checks at A-column zero contain the B-right component.
    bmask = (1 << (4 * q)) - 1
    br = [hz[t] & bmask for t in range(q)]
    return ar, br


def _retired_c35_ctx(rng, G, idx):
    """Unused previous-bed search retained for provenance."""
    import math

    q = G.q
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) <= W_MAX

    # A trinomial's three undirected differences have respectively
    # this many order-seven and order-five entries. These four types
    # are invariant under translations and C35 automorphisms.
    triple_types = ((0, 0), (1, 0), (0, 1), (1, 1))
    type_pairs = (
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
        (1, 2), (1, 3), (2, 2), (2, 3), (3, 3),
    )
    binomial_signatures = (
        (35, 35, 35),
        (7, 35, 35),
        (5, 35, 35),
        (7, 5, 35),
    )
    # The first two policies give twenty deliberate cards. Additional
    # policies accommodate the larger slot count used in the feedback.
    policies = (
        (0, 0), (1, 2), (1, 1), (2, 2), (3, 0),
        (3, 3), (0, 1), (0, 2), (1, 3), (2, 3),
    )
    phase, design = divmod(idx, len(type_pairs))
    if phase >= len(policies):
        raise ValueError("no unused difference-order card for this slot")
    ta, tb = type_pairs[design]
    pa, pb = policies[phase]
    units = [u for u in range(1, q) if math.gcd(u, q) == 1]
    order_pools = {
        order: [d for d in range(1, (q + 1) // 2)
                if q // math.gcd(d, q) == order]
        for order in (35, 7, 5)
    }

    def undirected(d):
        d %= q
        return min(d, q - d)

    def draw(triple_type, signature):
        """Normalize gauges and construct a four-cycle-free component."""
        wanted = triple_types[triple_type]
        for _ in range(1200):
            a, b = rng.sample(range(1, q), 2)
            internal = {undirected(a), undirected(b),
                        undirected(b - a)}
            # Repeated internal differences would give two translated
            # trinomials a two-check overlap.
            if len(internal) != 3:
                continue
            orders = [q // math.gcd(d, q) for d in internal]
            if (orders.count(7), orders.count(5)) != wanted:
                continue

            differences = []
            possible = True
            for order in (35, 7, 5):
                available = [d for d in order_pools[order]
                             if d not in internal]
                number = signature.count(order)
                if len(available) < number:
                    possible = False
                    break
                differences.extend(rng.sample(available, number))
            if not possible:
                continue

            signed = {s * d % q for d in differences for s in (-1, 1)}
            if any((x + y) % q in signed
                   for x in signed for y in signed):
                continue

            # Distinct binomial differences, disjoint from all trinomial
            # differences, prevent component Tanner four-cycles.
            rng.shuffle(differences)
            return [[[G.e, (d,)] for d in differences] +
                    [[G.e, (a,), (b,)]]]
        raise ValueError("could not realize the difference-order card")

    def canonical(A, B):
        """Discard gauge/automorphism copies within this slot's search."""
        best = None
        for u in units:
            sides = []
            for grid in (A, B):
                cells = []
                for cell in grid[0]:
                    values = [u * g[0] % q for g in cell]
                    normalized = min(
                        tuple(sorted((v - anchor) % q for v in values))
                        for anchor in values)
                    cells.append(normalized)
                sides.append(tuple(sorted(cells)))
            key = tuple(sorted(sides))
            if best is None or key < best:
                best = key
        return best

    def four_cycles(rows):
        return sum(c * (c - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for c in [(row & other).bit_count()])

    pool = []
    seen = set()
    for attempt in range(72):
        A = draw(ta, binomial_signatures[pa])
        B = draw(tb, binomial_signatures[pb])
        signature = canonical(A, B)
        if signature in seen:
            continue
        seen.add(signature)

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _c35_components(hx, hz, q)
        if four_cycles(ar) or four_cycles(br):
            continue
        if gf2_rank(ar) != q or gf2_rank(br) != q:
            continue
        a = _short_logical_counts(ar, [], 4 * q)
        b = _short_logical_counts(br, [], 4 * q)
        if any(a[:3]) or any(b[:3]):
            continue
        key = (max(a[3], b[3]), a[3] + b[3])
        pool.append((key, attempt, A, B, hx, hz, n))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[8:]

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def quotient_columns(stabilizers, n):
        """Canonical linear remainders modulo the verified row space."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        order = sorted(pivots, reverse=True)
        residuals = []
        for c in range(n):
            word = 1 << c
            for p in order:
                if (word >> p) & 1:
                    word ^= pivots[p]
            residuals.append(word)
        return residuals

    def hook_profile(checks, stabilizers, n):
        """Exact logical pair collisions with a shared-check hook surrogate.

        A pair's hook weight is its number of shared opposite checks.
        This deliberately does not claim that ASC realizes every pair.
        Collision classes distinguish logical differences from stabilizers.
        """
        syndrome = columns(checks, n)
        if 0 in syndrome or len(set(syndrome)) != n:
            return None
        spread = columns(stabilizers, n)
        residual = quotient_columns(stabilizers, n)
        singles = dict(zip(syndrome, residual))
        pairs = {}
        three = four = double_hook = hook_exposure = 0

        for i in range(n):
            si, ri, hi = syndrome[i], residual[i], spread[i]
            for j in range(i):
                s = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(s)
                if third is not None and label != third:
                    three += 1
                hook = (hi & spread[j]).bit_count()

                old = pairs.get(s)
                if old is None:
                    pairs[s] = [1, hook, {label: [1, hook]}]
                    continue
                same = old[2].get(label)
                same_count, same_hooks = same if same else (0, 0)
                different = old[0] - same_count
                different_hooks = old[1] - same_hooks

                four += different
                double_hook += hook * different_hooks
                hook_exposure += hook * different + different_hooks
                old[0] += 1
                old[1] += hook
                if same is None:
                    old[2][label] = [1, hook]
                else:
                    same[0] += 1
                    same[1] += hook

        # Every triple and quadruple has three pair representations.
        # Hook scores retain partitions because they describe fault pairs.
        return three // 3, double_hook, hook_exposure, four // 3

    best = None
    for coarse, attempt, A, B, hx, hz, n in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 9 * q:
            continue
        px = hook_profile(hz, hx, n)
        pz = hook_profile(hx, hz, n)
        if px is None or pz is None:
            continue

        # Protect both bases. Possible two-hook logical collisions precede
        # one-hook exposure and unweighted weight-four multiplicity.
        key = tuple(v for t in range(4)
                    for v in (max(px[t], pz[t]), px[t] + pz[t]))
        key += (coarse, attempt)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, px, pz)

    if best is None:
        raise ValueError("no balanced full-rank hook-screened filling")
    key, A, B, hx, hz, k, px, pz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return {"idx": idx, "case": (design, phase), "k": k,
            "screen": key, "hook_profile": (px, pz),
            "cells": {"A": A, "B": B}}


def _s17_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= marker
            row ^= bit
    return out


def _s17_four_cycles(rows):
    return sum(c * (c - 1) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for c in [(row & other).bit_count()])


def _s17_fingerprint(rows, width, q):
    """Uncolored rooted Tanner invariant, not full-code canonization."""
    from collections import Counter
    import hashlib

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, syndrome in enumerate(_s17_columns(rows, width)):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            syndrome ^= bit

    images = []
    # Lift translations are transitive within each check-row orbit.
    # Taking the minimum permits an isomorphism to exchange those orbits.
    for root in range(0, nr, q):
        colors = [0] * nr + [1] * width
        colors[root] = 2
        digest = hashlib.sha256()
        for _ in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(len(adjacency))
            ]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        images.append(digest.hexdigest())
    return min(images)


def _s17_hook_profile(checks, stabilizers, width):
    """Exact short-logical collisions with an uncertified hook surrogate."""
    syndrome = _s17_columns(checks, width)
    if 0 in syndrome or len(set(syndrome)) != width:
        return None
    spread = _s17_columns(stabilizers, width)

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    # Linear canonical remainders modulo the actual stabilizer span.
    residual = [0] * width
    for c in range(width):
        row = pivots.get(c)
        if row is None:
            residual[c] = 1 << c
            continue
        tail, value = row ^ (1 << c), 0
        while tail:
            bit = tail & -tail
            value ^= residual[bit.bit_length() - 1]
            tail ^= bit
        residual[c] = value

    singles = dict(zip(syndrome, residual))
    pairs = {}
    three = four = double_hook = exposure = 0
    for i, si in enumerate(syndrome):
        ri, hi = residual[i], spread[i]
        for j in range(i):
            s, label = si ^ syndrome[j], ri ^ residual[j]
            third = singles.get(s)
            if third is not None and third != label:
                three += 1
            hook = (hi & spread[j]).bit_count()
            old = pairs.get(s)
            if old is None:
                pairs[s] = [1, hook, {label: [1, hook]}]
                continue

            same = old[2].get(label)
            count, hooks = same if same else (0, 0)
            different = old[0] - count
            different_hooks = old[1] - hooks
            four += different
            double_hook += hook * different_hooks
            exposure += hook * different + different_hooks
            old[0] += 1
            old[1] += hook
            if same is None:
                old[2][label] = [1, hook]
            else:
                same[0] += 1
                same[1] += hook

    return three // 3, double_hook, exposure, four // 3


def NEW_CTX(rng, G, idx):
    """Pure component-class catalogue for the new unequal shapes."""
    from itertools import combinations

    q = G.q
    assert q == 17
    mask = (1 << q) - 1

    # Cache singleton actions ONLY from the verified construction.
    actions = [{}, {}]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G.idx(g)] = tuple(row & mask for row in hx)
        actions[1][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for cells in grid:
            out = [0] * q
            for j, cell in enumerate(cells):
                for g in cell:
                    for t, word in enumerate(actions[side][G.idx(g)]):
                        out[t] ^= word << (j * q)
            rows.extend(out)
        return rows

    def direction(d):
        d %= q
        return min(d, q - d)

    def directions(cell):
        return {direction(a - b) for a, b in combinations(cell, 2)}

    def normalized(cell, multiplier=1):
        values = [multiplier * a % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def canonical(cells):
        # Quotient actual cyclic automorphisms, cell origins and
        # component-column permutations, not arbitrary Tanner equivalence.
        return min(tuple(sorted(normalized(cell, u) for cell in cells))
                   for u in range(1, q))

    def quality(rows, width):
        if gf2_rank(rows) != len(rows):
            return None
        short = _short_logical_counts(rows, [], width)
        if any(short[:3]):
            return None
        cols = _s17_columns(rows, width)
        boundary = sum(
            (3 - w) ** 2
            for i, x in enumerate(cols) for y in cols[:i]
            for w in [(x ^ y).bit_count()] if w < 3)
        return (short[3] / width,
                _s17_four_cycles(rows) / len(rows),
                boundary / width)

    # A: three distinct binomial directions disjoint from the
    # trinomial's three directions, with no binomial triangles.
    triples = sorted({
        normalized((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if len(directions((0, a, b))) == 3
    })
    geometries = set()
    all_directions = set(range(1, (q + 1) // 2))
    for triple in triples:
        available = sorted(all_directions - directions(triple))
        for ds in combinations(available, 3):
            signed = {s * d % q for d in ds for s in (-1, 1)}
            if any((a + b) % q in signed
                   for a in signed for b in signed):
                continue
            geometries.add(canonical(
                tuple((0, d) for d in ds) + (triple,)))

    abook = {}
    for geometry in sorted(geometries):
        grid = [[[(a,) for a in cell] for cell in geometry]]
        rows = component(grid, 0)
        cost = quality(rows, 4 * q)
        if cost is None:
            continue
        identity = _s17_fingerprint(rows, 4 * q, q)
        item = (cost, identity, geometry, grid)
        old = abook.get(identity)
        if old is None or (cost, geometry) < (old[0], old[2]):
            abook[identity] = item
    abook = sorted(abook.values(), key=lambda item: item[:3])
    if not abook:
        raise ValueError("no full-rank short-word-screened A geometry")

    # Every B profile has exactly eight columns, eighteen terms,
    # row totals (9,9), and column totals two or three.
    mono = ((1, 1),)
    profiles = (
        mono * 6 + ((2, 1), (1, 2)),
        mono * 6 + ((3, 0), (0, 3)),
        mono * 5 + ((2, 0), (1, 2), (1, 2)),
        mono * 4 + ((2, 0), (0, 2), (2, 1), (1, 2)),
        mono * 5 + ((3, 0), (0, 2), (1, 2)),
    )
    sa = [[2, 2, 2, 3]]
    for profile in profiles:
        sb = [[column[i] for column in profile] for i in range(2)]
        assert len(profile) == 8
        assert tuple(map(sum, sb)) == (9, 9)
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 12
        assert sum(sa[0]) + max(map(sum, profile)) == 12
        assert max(map(sum, sb)) + max(sa[0]) == 12

    nonidentity = [g for g in G.elements if g != G.e]

    def draw_B(crng, profile):
        grid = [[], []]
        voltages = iter(crng.sample(
            G.elements, profile.count((1, 1))))
        for a, b in profile:
            if (a, b) == (1, 1):
                cells = ([G.e], [next(voltages)])
            elif not a or not b:
                weight = a or b
                cell = [G.e] + crng.sample(nonidentity, weight - 1)
                cells = (cell, []) if a else ([], cell)
            else:
                cells = (
                    [G.e] + crng.sample(nonidentity, a - 1),
                    crng.sample(G.elements, b),
                )
            grid[0].append(list(cells[0]))
            grid[1].append(list(cells[1]))
        return grid

    bbooks = []
    for architecture, profile in enumerate(profiles):
        # The same deterministic catalogue is rebuilt in each pure call.
        crng = random.Random(1704283 + 1009 * architecture)
        unique = {}
        for trial in range(160):
            grid = draw_B(crng, profile)
            rows = component(grid, 1)
            cost = quality(rows, 8 * q)
            if cost is None:
                continue
            identity = _s17_fingerprint(rows, 8 * q, q)
            labeling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            item = (cost, identity, labeling, grid)
            old = unique.get(identity)
            if old is None or (cost, labeling) < (old[0], old[2]):
                unique[identity] = item
        book = sorted(unique.values(), key=lambda item: item[:3])
        if not book:
            raise ValueError("no full-rank short-word-screened B class")
        bbooks.append(book[:16])

    # Twenty initial cards: four A classes crossed with five B
    # architectures. If refinement merges A classes, use the available
    # classes and extend through unused B classes rather than orbit copies.
    initial_a = min(4, len(abook))
    cards = [(a, p) for a in range(initial_a) for p in range(5)]
    catalogues = []
    for a, p in cards:
        aa = abook[a]
        catalogue = []
        for bb in bbooks[p]:
            identity = (aa[1], bb[1])
            coarse = tuple(
                value for t in range(3)
                for value in (max(aa[0][t], bb[0][t]),
                              aa[0][t] + bb[0][t]))
            catalogue.append((coarse, identity, aa[3], bb[3]))
        catalogues.append(sorted(catalogue, key=lambda item: item[:2]))

    # Prefix reservation is deterministic and local: no mutable cache
    # and no transformation of an earlier slot's filling.
    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx + 1):
        card = slot % len(cards)
        available = [item for item in catalogues[card]
                     if item[1] not in reserved]
        if not available:
            alternatives = [
                (item[0], item[1], c, item)
                for c, catalogue in enumerate(catalogues)
                for item in catalogue if item[1] not in reserved
            ]
            if not alternatives:
                raise ValueError("unused component-class pairs exhausted")
            alternatives.sort(key=lambda item: item[:3])
            card = alternatives[0][2]
            available = [item for item in catalogues[card]
                         if item[1] not in reserved]
        chosen = available[:3]
        actual_card = card
        reserved.update(item[1] for item in chosen)

    best = None
    for coarse, identity, A, B in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 578 or k != 306:
            continue

        # Certify both cached component conventions against the exact
        # final evaluator-convention matrices.
        ar = [
            sum(((hx[t] >> (j * 8 * q)) & mask) << (j * q)
                for j in range(4))
            for t in range(q)
        ]
        br = [row & ((1 << (8 * q)) - 1)
              for row in hz[:2 * q]]
        assert ar == component(A, 0)
        assert br == component(B, 1)

        px = _s17_hook_profile(hz, hx, n)
        pz = _s17_hook_profile(hx, hz, n)
        if px is None or pz is None or px[0] or pz[0]:
            continue
        key = tuple(
            value for t in range(4)
            for value in (max(px[t], pz[t]), px[t] + pz[t]))
        key += (coarse, identity)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, px, pz)

    if best is None:
        raise ValueError("no exact-bound joint hook-screened finalist")
    key, A, B, hx, hz, px, pz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert sum(row.bit_count() for row in hx + hz) * 6 == 18360
    assert css_ok(hx, hz)
    return {"idx": idx, "case": cards[actual_card], "k": 306,
            "screen": key, "hook_profile": (px, pz),
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return every cell of the new shapes without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Excluding weight-three logicals makes one-sided four-term reinforcement sufficient for better memory: C35 slot 12 still failed 203/250 X shots with d2s=4 and s=3.065, versus all-doubleton slot 0's 154/250 and s=3.275; labels also changed, so this does not isolate term count.
# REFUTED: Four-cycle-free balanced C35 components plus shared-check hook screening certify rare collective circuit failures: the current parent improved to 3.5010 but slot 0 still failed 109/250 Z and 105/250 X shots, affecting all 315 logicals in each basis.
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
