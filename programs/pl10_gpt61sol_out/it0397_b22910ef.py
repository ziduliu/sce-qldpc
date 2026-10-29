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
CONCEPT = ("For generalized quaternion Q16, order 16, shapes (1,4)x(2,7), replacing one A trinomial by a binomial and strengthening one to four B columns should improve measured fitness above 3.2584 by reducing collective Z-memory failures.")
GROUP = ("Frozen under L1: retain Q16, order 16, multiplication and encoding byte-for-byte. This exact bed scored 3.2584. Odd-augmentation elements are units in the finite 2-group algebra, supporting full-rank components containing trinomials; complete helper-built CSS ranks remain the acceptance test, not an assumed consequence of component ranks.")
SHAPES = ("Frozen (1,4)x(2,7): the size factor thirty permits q<=19, while the fixed q=16 gives n=480, kb=240 and guaranteed rate 0.500. Unequal components have 64 variables/16 checks and 112 variables/32 checks. Require verified k=240, retaining cell 0.500 without chasing rank deficits. Parent failures collectively affected essentially all 243 logicals, with diffuse 1.0-1.2x Acol/Bcol data and Acol/Brow check enrichment rather than a persistent local repair site.")
FILLING = ("Twenty contracts cross four B-strengthening levels with five count architectures and invariant-distinct A geometries; further batches consume additional A geometries. A has weights [2,2,2,3]. B has one to four degree-three columns and otherwise degree-two columns, balanced row totals and full augmentation rank. Helper-derived actions guide component annealing and exclusion of kernels through three. Full verified CSS finalists are selected by stabilizer-classified weighted fault ambiguity, common-stream noisy-data BP probes and independent holdouts. These probes omit ASC ordering, measurement histories and relay/LSD and are not calibrated pL measurements. Both row weights remain <=12; six-round CNOT budgets are 14976,15456,15936 or16416 versus the parent's 15360. Target the parent's 122-141 Z failures/250 and widespread logical flips, not distance alone.")
FAMILY_NAME = "r050_Q16_14x27_strengthenedB_decoder_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q16 requires order 16")
    # Canonical x^a*y^b: x^8=1, y^2=x^4, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 4*b*d) % 8,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 8, 0) if b == 0 else ((a + 4) % 8, 1)

    def idx(g):
        a, b = red(g)
        return a + 8*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Four graphic B architectures, all with seven degree-two columns.
# Row totals are 7/7 or 8/6; zeros are prescribed, not repairs.
_B_PROFILES = (
    ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(0,2),(1,1),(1,1),(1,1),(1,1),(1,1)),
    ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
)


def _kernel56(rows, ncols):
    """Exact weight-five/six counts when kernel weights <=4 are absent.

    Pair/triple syndrome matches count each weight-five word ten times.
    Distinct triple collisions count each weight-six word ten times.
    Smaller-word exclusion makes these colliding supports disjoint.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            s = x ^ y
            pairs[s] = pairs.get(s, 0) + 1
    triples = {}
    five = six = 0
    for i in range(ncols):
        for j in range(i):
            s = columns[i] ^ columns[j]
            for h in range(j):
                value = s ^ columns[h]
                five += pairs.get(value, 0)
                six += triples.get(value, 0)
                triples[value] = triples.get(value, 0) + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def _retired_quaternion_graph_CTX(rng, G, idx):
    """Inactive previous filling, retained only for provenance."""
    from itertools import combinations
    from math import exp

    q = G.q
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    assert q == 16

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        # Zero partners are extraction probes, not submitted codes.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    # Automorphisms x->x^u, y->x^v*y, u odd. Independent right
    # translations are A-column gauges. Canonicalization removes these
    # obvious copies; arbitrary Tanner isomorphism remains downstream.
    automorphisms = [
        tuple(G.idx(((u*a + v*b) % 8, b))
              for a, b in G.elements)
        for u in (1, 3, 5, 7) for v in range(8)
    ]

    def normal(cell, transform):
        values = [G.elements[transform[G.idx(g)]] for g in cell]
        return min(tuple(sorted(
            G.idx(G.mul(g, G.inv(origin))) for g in values))
                   for origin in values)

    triangles = [
        (G.e, (d, 0), (a, 1))
        for d in (1, 2, 3, 5, 6, 7) for a in range(8)
    ]
    binomial_pairs = (
        ((G.e, (1, 0)), (G.e, (3, 0))),
        ((G.e, (1, 0)), (G.e, (0, 1))),
        ((G.e, (0, 1)), (G.e, (1, 1))),
    )
    all_cells = set(triangles)
    for pair in binomial_pairs:
        all_cells.update(pair)
    normal_forms = {
        cell: tuple(normal(cell, transform)
                    for transform in automorphisms)
        for cell in all_cells
    }

    book, seen_geometry = [], set()
    for bins in binomial_pairs:
        for first, second in combinations(triangles, 2):
            cells = bins + (first, second)
            signature = min(
                tuple(sorted(normal_forms[cell][t] for cell in cells))
                for t in range(len(automorphisms))
            )
            if signature in seen_geometry:
                continue
            seen_geometry.add(signature)
            grid = [[[G.elements[g] for g in cell]
                     for cell in signature]]
            rows, width = component(grid, "A")
            stats = _kernel_statistics(rows, width)
            if stats[0] or stats[1] or gf2_rank(rows) != q:
                continue
            key = (stats[2], stats[3], _four_cycles(rows),
                   _six_cycles(rows), signature)
            book.append((key, grid))
    book.sort(key=lambda item: item[0])
    wanted = 5 * batch + marker
    if wanted >= len(book):
        raise ValueError("Canonical quaternion A contract book exhausted")
    akey, A = book[wanted]

    profile = _B_PROFILES[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2, 2, 3, 3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    assert all(sum(p) == 2 for p in profile)
    assert max(map(sum, sb)) <= 8

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [(1, j) for j in cross[1:]]
    outside_count = 1 + (marker + batch) % (len(cross) - 1)
    outside_sites = set(cross[1:1 + outside_count])
    rotations = [(a, 0) for a in range(1, 8)]
    outside = [(a, 1) for a in range(8)]
    noncentral = rotations[:3] + rotations[4:] + outside
    internal_palette = (
        noncentral,
        [(a, 0) for a in (1, 3, 5, 7)],
        [(a, 0) for a in (1, 3, 5, 7)] + outside,
    )[batch % 3]

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for i, j in internal:
            grid[i][j] = [G.e, rng.choice(internal_palette)]
        rv = rng.sample(rotations, len(cross) - 1 - outside_count)
        ov = rng.sample(outside, outside_count)
        for j in cross:
            grid[0][j] = [G.e]
            g = (G.e if j == cross[0] else
                 ov.pop() if j in outside_sites else rv.pop())
            grid[1][j] = [g]
        return grid

    def assess_B(grid):
        rows, width = component(grid, "B")
        stats = _kernel_statistics(rows, width)
        # B is an incidence matrix. These are parallel-edge, triangle
        # and four-cycle statistics on the actual helper-built graph.
        energy = (1000 * stats[0] + 100 * stats[1] + stats[2]) / q
        return energy, stats, rows

    pool, seen_B = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != 2*q - 1:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen_B:
            return
        seen_B.add(signature)
        serial += 1
        key = (stats[2], stats[3])
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[6:]

    for restart in range(4):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(220):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [G.e, rng.choice(internal_palette)]
            else:
                palette = outside if j in outside_sites else rotations
                occupied = {proposal[1][c][0] for c in cross if c != j}
                choices = [g for g in palette if g not in occupied]
                if not choices:
                    continue
                proposal[i][j] = [rng.choice(choices)]
            proposed, ps, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 4.0 * (0.982 ** step) + 0.04
            if delta <= 0 or rng.random() < exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No connected triangle-free quaternion B contract")

    def quotient_columns(stabilizers, width):
        # Linear canonical remainders modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
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
        return residual

    def logical_risk(checks, stabilizers, width):
        """Exact short-logical classification; hook costs are surrogates.

        A shared opposite-basis check marks possible ancilla spread,
        not a claim about the ordering of the measured ASC circuit.
        """
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        exposure = [a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]
        buckets = {}
        hooks = double_hooks = weighted = partitions = 0

        for i, si in enumerate(syndrome):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                value = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(value)
                if third is not None and third != label:
                    return None
                h = (hi & spread[j]).bit_count()
                w = exposure[i] * exposure[j]
                bucket = buckets.get(value)
                if bucket is None:
                    buckets[value] = [1, h, w, {label: [1, h, w]}]
                    continue
                same = bucket[3].get(label)
                sc, sh, sw = same if same is not None else (0, 0, 0)
                dc = bucket[0] - sc
                dh = bucket[1] - sh
                dw = bucket[2] - sw
                partitions += dc
                hooks += h * dc + dh
                double_hooks += h * dh
                weighted += w * dc + dw
                bucket[0] += 1
                bucket[1] += h
                bucket[2] += w
                if same is None:
                    bucket[3][label] = [1, h, w]
                else:
                    same[0] += 1
                    same[1] += h
                    same[2] += w
        assert partitions % 3 == 0
        return hooks, double_hooks, weighted, partitions // 3

    winner = None
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 480 or k != 243:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        score = tuple(value for t in range(4)
                      for value in (max(px[t], pz[t]), px[t] + pz[t]))
        score += coarse
        if winner is None or score < winner[0]:
            winner = (score, B, hx, hz, px, pz)

    if winner is None:
        raise ValueError("No k=243 quaternion CSS filling passed screening")
    score, B, hx, hz, px, pz = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    probe_a, _ = component(A, "A")
    probe_b, _ = component(B, "B")
    assert actual_a == probe_a and actual_b == probe_b
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert 6 * sum(row.bit_count() for row in hx + hz) == 15360
    assert 19 * 480 <= 40 * 243 < 21 * 480
    return {"A": A, "B": B, "design": (architecture, wanted),
            "voltage_contract": (outside_count, batch % 3),
            "k": 243, "logical_risk": (px, pz),
            "A_contract": akey, "cycle_score": score}


def NEW_CTX(rng, G, idx):
    """Degree contracts, verified components and decoder-classified selection."""
    import hashlib
    import math
    import random
    from collections import Counter
    from itertools import combinations, combinations_with_replacement

    q = G.q
    assert q == 16
    brief, batch = idx % 20, idx // 20
    extra, architecture = divmod(brief, 5)
    extra += 1
    wanted = 5 * batch + architecture
    mask = (1 << q) - 1

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    # Cache only singleton blocks returned by the verified helper.
    # Final component projections are independently checked below.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz1)

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        rows[i*q+t] ^= bits << (j*q)
        return rows, len(grid[0]) * q

    def connected(rows, width):
        adjacency = [0] * len(rows)
        for value in columns(rows, width):
            sites = []
            while value:
                bit = value & -value
                sites.append(bit.bit_length() - 1)
                value ^= bit
            for r in sites:
                for s in sites:
                    adjacency[r] |= 1 << s
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length() - 1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        return reached == (1 << len(rows)) - 1

    def fingerprint(rows, width):
        """Rooted uncolored invariant, not complete Tanner canonization.

        Regular translations are transitive within each check-row orbit.
        Equal fingerprints are conservatively discarded.
        """
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, value in enumerate(columns(rows, width)):
            while value:
                bit = value & -value
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                value ^= bit
        images = []
        for root in range(0, nr, q):
            colors = [0] * nr + [1] * width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(12):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr + width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            images.append(digest.hexdigest())
        return min(images)

    # A geometry is three binomial directions and one trinomial.
    # Quotient common automorphisms, independent column gauges and
    # column permutations before inspecting uncolored component classes.
    autos = [
        tuple(G.idx(((u*a + v*b) % 8, b))
              for a, b in G.elements)
        for u in (1, 3, 5, 7) for v in range(8)
    ]
    directions = [
        g for g in G.elements
        if g != G.e and G.idx(g) < G.idx(G.inv(g))
    ]
    bins = [(G.e, g) for g in directions]
    triples = [
        (G.e, a, b)
        for a, b in combinations(
            [g for g in G.elements if g != G.e], 2)
    ]

    def normal(cell, aut):
        values = [G.elements[aut[G.idx(g)]] for g in cell]
        return min(tuple(sorted(
            G.idx(G.mul(g, G.inv(origin))) for g in values))
                   for origin in values)

    forms = {
        cell: tuple(normal(cell, aut) for aut in autos)
        for cell in bins + triples
    }
    raw_A, seen_A = [], set()
    for three_bins in combinations(bins, 3):
        for triple in triples:
            cells = three_bins + (triple,)
            signature = min(
                tuple(sorted(forms[cell][t] for cell in cells))
                for t in range(len(autos)))
            if signature in seen_A:
                continue
            seen_A.add(signature)
            A = [[[G.elements[g] for g in cell]
                  for cell in signature]]
            ar, width = component(A, "A")
            stats = _kernel_statistics(ar, width)
            if any(stats[:2]) or gf2_rank(ar) != q:
                continue
            if not connected(ar, width):
                continue
            key = (stats[2], stats[3], _four_cycles(ar),
                   _six_cycles(ar), signature)
            raw_A.append((key, A, ar))

    bank, classes = [], set()
    for key, A, ar in sorted(raw_A, key=lambda item: item[0]):
        identity = fingerprint(ar, 4*q)
        if identity in classes:
            continue
        classes.add(identity)
        bank.append((key, identity, A, ar))
        if len(bank) >= max(24, wanted + 8):
            break
    if wanted >= len(bank):
        raise ValueError("Distinct nine-term quaternion A bank exhausted")
    akey, aid, A, ar = bank[wanted]
    sa = [[len(cell) for cell in A[0]]]
    assert sorted(sa[0]) == [2, 2, 2, 3]

    # Enumerate balanced weighted support geometries, not random count
    # grids. Select five different zero/concentration architectures.
    degree_two = ((1,1), (2,0), (0,2))
    degree_three = ((2,1), (1,2), (3,0), (0,3))
    support_classes = {}
    for light in combinations_with_replacement(degree_two, 7-extra):
        for strong in combinations_with_replacement(degree_three, extra):
            cols = light + strong
            totals = tuple(sum(p[i] for p in cols) for i in range(2))
            if abs(totals[0] - totals[1]) > 1:
                continue
            signature = min(
                tuple(sorted(cols)),
                tuple(sorted((b, a) for a, b in cols)))
            sb = [[p[i] for p in signature] for i in range(2)]
            augmentation = [
                sum((w & 1) << j for j, w in enumerate(row))
                for row in sb]
            if gf2_rank(augmentation) != 2:
                continue
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > W_MAX:
                continue
            zeros = sum(0 in p for p in signature)
            concentrated = sum(3 in p for p in signature)
            category = (zeros, concentrated)
            old = support_classes.get(category)
            if old is None or signature < old:
                support_classes[category] = signature
    support_book = sorted(support_classes.items())
    assert len(support_book) >= 5
    support_category, profile = support_book[architecture][0], (
        support_book[architecture][1])
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sum(map(sum, sb)) == 14 + extra
    assert sorted(map(sum, zip(*sb))) == [2]*(7-extra) + [3]*extra
    assert _grid_weights(sa, sb) <= W_MAX
    expected_cnots = 6*q*(9*9 + 5*(14+extra))
    assert expected_cnots in (14976, 15456, 15936, 16416)

    anchors = [
        next(i for i in range(2) if sb[i][j])
        for j in range(7)]
    mutable = [
        (i, j) for i in range(2) for j in range(7)
        if sb[i][j] and not (i == anchors[j] and sb[i][j] == 1)
    ]
    nonidentity = [g for g in G.elements if g != G.e]
    noninvolutions = [
        g for g in nonidentity if G.mul(g, g) != G.e
    ]

    def new_cell(i, j):
        count = sb[i][j]
        if i == anchors[j]:
            if count == 2:
                return [G.e, rng.choice(noninvolutions)]
            return [G.e] + rng.sample(nonidentity, count-1)
        return rng.sample(G.elements, count)

    def draw_B():
        return [[new_cell(i, j) if sb[i][j] else []
                 for j in range(7)] for i in range(2)]

    def assess_B(B):
        br, width = component(B, "B")
        stats = _kernel_statistics(br, width)
        disconnected = int(not connected(br, width))
        deficit = 2*q - gf2_rank(br)
        energy = (
            1000*stats[0] + 100*stats[1] + stats[2]
            + 200*disconnected + 500*deficit) / q
        return energy, stats, disconnected, deficit, br

    pool, seen_labels = [], set()
    serial = 0

    def remember(B, stats, disconnected, deficit, br):
        nonlocal serial
        if any(stats[:2]) or disconnected or deficit:
            return
        key = (stats[2], stats[3], _four_cycles(br))
        if len(pool) == 10 and key > pool[-1][0]:
            return
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in B for cell in row)
        if signature in seen_labels:
            return
        seen_labels.add(signature)
        serial += 1
        pool.append((key, serial, clone(B), list(br)))
        pool.sort(key=lambda item: item[:2])
        del pool[10:]

    for restart in range(5):
        B = draw_B()
        energy, stats, disconnected, deficit, br = assess_B(B)
        remember(B, stats, disconnected, deficit, br)
        for step in range(400):
            proposal = clone(B)
            i, j = rng.choice(mutable)
            proposal[i][j] = new_cell(i, j)
            proposed, ps, pc, pd, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 8.0*(0.988**step) + 0.04
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                B, energy = proposal, proposed
                stats, disconnected, deficit, br = ps, pc, pd, pr
                remember(B, stats, disconnected, deficit, br)

    if not pool:
        # Fresh complete proposals keep the same count contract.
        # No invalid submitted slot is repaired.
        for trial in range(256):
            B = draw_B()
            _, stats, disconnected, deficit, br = assess_B(B)
            remember(B, stats, disconnected, deficit, br)
    if not pool:
        raise ValueError("No strengthened B passed its component contract")

    def quotient(stabilizers, width):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residue = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                residue[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residue[bit.bit_length() - 1]
                tail ^= bit
            residue[c] = value
        return residue

    def fault_mass(checks, stabilizers, width):
        """Stabilizer-classified ambiguity, not a circuit failure estimate."""
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        residue = quotient(stabilizers, width)
        singles = dict(zip(syndrome, residue))
        exposure = [1 + a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]
        buckets = {}

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0) + mass

        for i, s in enumerate(syndrome):
            add(s, residue[i], 512*exposure[i])
        for i, si in enumerate(syndrome):
            for j in range(i):
                s = si ^ syndrome[j]
                label = residue[i] ^ residue[j]
                third = singles.get(s)
                if third is not None and third != label:
                    return None
                # Shared checks are possible spread sites, not
                # certified hooks of the actual ASC ordering.
                mass = exposure[i]*exposure[j]
                mass += 16*(spread[i] & spread[j]).bit_count()
                add(s, label, mass)
        loss = peak = ambiguous = 0
        spectrum = Counter()
        for classes in buckets.values():
            if len(classes) > 1:
                error = sum(classes.values()) - max(classes.values())
                loss += error
                peak = max(peak, error)
                ambiguous += 1
                spectrum[error] += 1
        return loss, peak, ambiguous, tuple(sorted(spectrum.items()))

    def probe(checks, stabilizers, width, seed, shots):
        """Noisy-data BP surrogate, NOT the ASC memory experiment.

        Classify settled residuals modulo verified stabilizers.
        Measurement histories, ASC ordering and relay/LSD are absent.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length() - 1)
                row ^= bit
            return out

        ns = [sites(row) for row in checks]
        opposite = [sites(row) for row in stabilizers]
        incident = [[] for _ in range(width)]
        edge_variables = []
        for row in ns:
            for c in row:
                incident[c].append(len(edge_variables))
                edge_variables.append(c)
        ne = len(edge_variables)
        max_check = max(map(len, ns))
        check_edges = np.full(
            (len(ns), max_check), ne, dtype=np.int32)
        check_sites = np.full(
            (len(ns), max_check), width, dtype=np.int32)
        valid = np.zeros((len(ns), max_check), dtype=bool)
        offset = 0
        for r, row in enumerate(ns):
            size = len(row)
            check_edges[r, :size] = np.arange(offset, offset+size)
            check_sites[r, :size] = row
            valid[r, :size] = True
            offset += size
        variable_edges = np.full(
            (width, max(map(len, incident))), ne, dtype=np.int32)
        for c, edges in enumerate(incident):
            variable_edges[c, :len(edges)] = edges
        edge_variables = np.asarray(edge_variables, dtype=np.int32)

        exposure = np.asarray([
            a.bit_count() + b.bit_count()
            for a, b in zip(columns(checks, width),
                            columns(stabilizers, width))], dtype=float)
        levels = np.asarray((0.040, 0.055, 0.070))
        rates = np.minimum(
            0.24, levels[np.arange(shots) % 3, None]
            * exposure[None, :] / 5.0)
        channel = np.random.default_rng(seed)
        errors = np.zeros((shots, width+1), dtype=bool)
        errors[:, :width] = (
            channel.random((shots, width)) < rates)
        triggers = channel.random((shots, len(opposite))) < 0.003
        for shot, r in zip(*np.nonzero(triggers)):
            row = opposite[int(r)]
            if len(row) >= 2:
                a, b = channel.choice(len(row), 2, replace=False)
                errors[shot, row[int(a)]] ^= True
                errors[shot, row[int(b)]] ^= True

        target = np.logical_xor.reduce(
            errors[:, check_sites], axis=2)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.log((1-rates)/rates)
        messages = prior[:, edge_variables].copy()
        outgoing = np.zeros((shots, ne+1), dtype=float)
        done = np.zeros(shots, dtype=bool)
        decoded = np.zeros((shots, width), dtype=bool)
        completion = np.full(shots, 33, dtype=np.int32)
        positions = np.arange(max_check)[None, None, :]

        for iteration in range(32):
            padded = np.concatenate(
                (messages, np.zeros((shots, 1))), axis=1)
            shaped = padded[:, check_edges]
            magnitude = np.where(
                valid[None, :, :], np.abs(shaped), np.inf)
            signs = np.where(shaped < 0, -1.0, 1.0)
            product = signs.prod(axis=2) * target_sign
            minimum_at = np.argmin(magnitude, axis=2)
            smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
            size = np.where(
                positions == minimum_at[:, :, None],
                smallest[:, :, 1, None], smallest[:, :, 0, None])
            update = 0.80*product[:, :, None]*signs*size
            update = np.where(valid[None, :, :], update, 0.0)
            outgoing[:, :ne] = update[:, valid]
            posterior = prior + outgoing[:, variable_edges].sum(axis=2)
            hard = posterior < 0
            hard_padded = np.concatenate(
                (hard, np.zeros((shots, 1), dtype=bool)), axis=1)
            parity = np.logical_xor.reduce(
                hard_padded[:, check_sites], axis=2)
            settled = np.all(parity == target, axis=1)
            newly = settled & ~done
            decoded[newly] = hard[newly]
            completion[newly] = iteration + 1
            done |= settled
            if np.all(done):
                break
            proposal = posterior[:, edge_variables] - outgoing[:, :ne]
            messages = (
                0.30*messages + 0.70*np.clip(proposal, -24, 24))

        residue = quotient(stabilizers, width)
        failed = ~done
        difference_weight = 0
        for shot in np.flatnonzero(done):
            difference = decoded[shot] ^ errors[shot, :width]
            label = 0
            for c in np.flatnonzero(difference):
                label ^= residue[int(c)]
            if label:
                failed[shot] = True
                difference_weight += int(difference.sum())
        return (int(failed.sum()), int((~done).sum()),
                difference_weight, int(completion.sum()))

    # Compare genuinely different B component classes, not several
    # labeling images of one structural minimum.
    options, bclasses, fault_classes = [], set(), set()
    for coarse, serial, B, br in pool:
        bid = fingerprint(br, 7*q)
        if bid in bclasses:
            continue
        bclasses.add(bid)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 480 or k != 240:
            continue
        # X errors in Z memory are sensed by HZ.
        zcost = fault_mass(hz, hx, n)
        xcost = fault_mass(hx, hz, n)
        if zcost is None or xcost is None:
            continue
        invariant = tuple(sorted((zcost, xcost)))
        if invariant in fault_classes:
            continue
        fault_classes.add(invariant)
        structural = (
            max(zcost[0], xcost[0]), zcost[0]+xcost[0],
            zcost[0], max(zcost[1], xcost[1]),
            zcost[1]+xcost[1], zcost[2]+xcost[2], coarse)
        options.append((
            structural, serial, B, br, bid, hx, hz, zcost, xcost))
        if len(options) == 4:
            break
    if not options:
        raise ValueError("No k=240 strengthened-B CSS finalist")
    options.sort(key=lambda item: item[:2])

    def quality(z, x):
        return (max(z[0], x[0]), z[0]+x[0],
                max(z[1], x[1]), z[1]+x[1],
                z[0], z[2]+x[2], z[3]+x[3])

    # Common streams compare alternatives; independent holdouts
    # choose the winner instead of reusing training failures.
    seed = 160427903 + 1009*idx
    training = []
    for number, option in enumerate(options):
        structural, serial, B, br, bid, hx, hz, zcost, xcost = option
        z = probe(hz, hx, 480, seed, 32)
        x = probe(hx, hz, 480, seed+1, 32)
        training.append((
            quality(z, x)+structural+(number,), number, option, z, x))
    training.sort(key=lambda item: item[:2])
    finalists = []
    for _, number, option, tz, tx in training[:2]:
        structural, serial, B, br, bid, hx, hz, zcost, xcost = option
        z = probe(hz, hx, 480, seed+1000003, 96)
        x = probe(hx, hz, 480, seed+1000004, 96)
        score = quality(z, x)+quality(tz, tx)+structural+(number,)
        finalists.append((score, option, z, x))
    finalists.sort(key=lambda item: item[0])
    score, option, z, x = finalists[0]
    structural, serial, B, br, bid, hx, hz, zcost, xcost = option

    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == ar and actual_b == br
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 240
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
    assert _no_dead_lines(sa, sb)
    assert all(len(cell) <= CELL_MAX and
               len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": 240,
            "design": (extra, architecture, wanted),
            "support_category": support_category,
            "component_classes": (aid, bid),
            "A_contract": akey, "fault_mass": (zcost, xcost),
            "bp_holdout": (z, x), "selection_score": score,
            "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified degree-contract cell without redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: A<=3/B<=4 component exclusion balances C19 (1,4)x(2,7) circuit memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored 3.1933 despite substantially lower Z failed-shot rates.
# REFUTED: Noncommuting Q16 A geometries, triangle-free graphic B and shared-check quartet minimization suffice for rare high-rate memory failures: fitness was 3.2584, measured Z failures were 122..141/250, and collective flips affected essentially all 243 logicals despite first-pass Z settlement.
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
