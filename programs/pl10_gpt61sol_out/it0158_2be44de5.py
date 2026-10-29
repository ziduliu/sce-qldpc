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
CONCEPT = ("Metacyclic C7 semidirect C4 with inversion action, order 28, shapes (1,4)x(1,5): reducing extraction exposure while screening both bases for truncated fault-channel ambiguity should improve circuit fitness beyond the parent's 3.4785.")
GROUP = ("Keep the metacyclic construction rule, resizing its parameters to m=7,r=4,s=6. Since 6^4=1 modulo 7, the action is valid. The new factor 21 excludes order 32; order 28 gives n=588. Related order-28 metacyclic beds establish feasibility but do not establish a high-rate circuit advantage.")
SHAPES = ("Unequal 1x4 and 1x5 components have 112 and 140 variables with 28 checks each. n=588 and kb=336 guarantee rate 12/21=0.571429 in open-ended cell 0.550. Connected all-doubleton components are required to give verified k=345, nine above the bound. Parent failures spanned Acol/Bcol data and Acol/Brow checks with only about 1.0-1.1x enrichment, so both components are protected rather than repairing an unproven hotspot.")
FILLING = ("All cells are doubletons with non-involutory relative labels. Component row totals eight and ten give HX weight ten, HZ weight twelve, and 16464 six-round CNOTs versus the parent's 17280. Enumerate connected verified component graphs, conservatively collapse equal uncolored rooted fingerprints, and assign distinct ordered component-structure pairs to slots. Joint finalists quotient simultaneous group automorphisms and minimize worse-basis stabilizer-quotient fault ambiguity with gate-degree exposure and normalized shared-check spread masses. Exact short-logical counts and ranks remain screens, not fitness. This targets parent slot 0's 171/500 Z and 188/500 X failures, including 95 flips of its worst X logical; measurement faults and ASC ordering remain unmodeled.")
FAMILY_NAME = "r055_m7r4s6_14x15_gatelean_structural_fault_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C4 with inversion requires q=28")
    # Same metacyclic rule: x^7=y^4=1, y*x*y^-1=x^6.
    m, r, s = 7, 4, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


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


def _single_row_components(hx, hz, q):
    """Project components only from the scaffold's verified lift."""
    mask = (1 << q) - 1
    ar = [
        sum(((hx[t] >> (j * 4 * q)) & mask) << (j * q)
            for j in range(4))
        for t in range(q)
    ]
    br = [hz[t] & ((1 << (4 * q)) - 1) for t in range(q)]
    return ar, br


def _binary_brief(rows, q):
    """Colored weight-four words and uncolored graph invariants.

    A binary component is a graph incidence matrix. Everything here
    is extracted from verified rows, not from a separate group lift.
    Repeated-cell pairs are only a possible hook-risk surrogate.
    """
    columns = [0] * (4 * q)
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    if (len(set(columns)) != len(columns) or
            any(s.bit_count() != 2 for s in columns)):
        return None
    if gf2_rank(rows) != q - 1:
        return None

    adjacency = [0] * q
    for syndrome in columns:
        low = syndrome & -syndrome
        a = low.bit_length() - 1
        b = (syndrome ^ low).bit_length() - 1
        adjacency[a] |= 1 << b
        adjacency[b] |= 1 << a

    # Each triangle is counted on its three edges.
    triangles = 0
    for a, neighbors in enumerate(adjacency):
        later = neighbors >> (a + 1)
        while later:
            bit = later & -later
            b = a + 1 + bit.bit_length() - 1
            triangles += (neighbors & adjacency[b]).bit_count()
            later ^= bit
    if triangles:
        return None

    patterns = ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
    words = {p: 0 for p in patterns}
    pairs = {}
    for a in range(len(columns)):
        for b in range(a + 1, len(columns)):
            syndrome = columns[a] ^ columns[b]
            bucket = pairs.setdefault(syndrome, [])
            for c, d in bucket:
                if d >= a:
                    continue
                colors = {}
                for t in (c, d, a, b):
                    color = t // q
                    colors[color] = colors.get(color, 0) + 1
                partition = tuple(sorted(colors.values(), reverse=True))
                words[partition] += 1
            bucket.append((a, b))

    counts = tuple(words[p] for p in patterns)
    total = sum(counts)
    repeated_pairs = sum(
        number * sum(t * (t - 1) // 2 for t in partition)
        for partition, number in zip(patterns, counts)
    )

    seen = frontier = 1
    shells = []
    while frontier:
        shells.append(frontier.bit_count())
        next_frontier = 0
        bits = frontier
        while bits:
            bit = bits & -bits
            next_frontier |= adjacency[bit.bit_length() - 1]
            bits ^= bit
        next_frontier &= ~seen
        seen |= next_frontier
        frontier = next_frontier
    assert seen == (1 << q) - 1

    neighbor_lists = []
    for neighbors in adjacency:
        ids = []
        while neighbors:
            bit = neighbors & -neighbors
            ids.append(bit.bit_length() - 1)
            neighbors ^= bit
        neighbor_lists.append(ids)
    walks = [1] + [0] * (q - 1)
    returns = []
    for length in range(1, 9):
        walks = [sum(walks[t] for t in ids) for ids in neighbor_lists]
        if length >= 4:
            returns.append(walks[0])

    signature = (counts, tuple(shells), tuple(returns))
    # Heuristic weighting, not a calibrated circuit failure probability.
    risk = 2 * total + 3 * repeated_pairs
    return risk, total, repeated_pairs, signature


def _generator_book(G):
    """Enumerate component briefs, quotienting actual automorphisms.

    No slot is made by applying an automorphism to a saved filling.
    Different signatures are conservative design distinctions; the
    evaluator still performs its stronger full Tanner-isomorphism test.
    """
    from itertools import combinations

    q, els, e = G.q, G.elements, G.e

    def powers(g, count):
        out = [e]
        for _ in range(count):
            out.append(G.mul(out[-1], g))
        return out

    inverse_class = [
        min(t, G.idx(G.inv(g))) for t, g in enumerate(els)
    ]
    generators = [
        t for t, g in enumerate(els)
        if inverse_class[t] == t and G.mul(g, g) != e
    ]

    # Every automorphism is determined by images of x and y. Enumerate
    # homomorphisms satisfying the presentation, then require bijectivity.
    automorphisms = set()
    for x in els:
        xp = powers(x, 8)
        if xp[8] != e or xp[4] == e:
            continue
        for y in els:
            yp = powers(y, 4)
            if yp[4] != e:
                continue
            if G.mul(G.mul(y, x), G.inv(y)) != xp[5]:
                continue
            image = tuple(
                G.idx(G.mul(xp[a], yp[b]))
                for b in range(4) for a in range(8)
            )
            if len(set(image)) == q:
                automorphisms.add(image)
    assert automorphisms
    automorphisms = sorted(automorphisms)

    seen_orbits = set()
    briefs = {}
    for quad in combinations(generators, 4):
        if quad in seen_orbits:
            continue
        orbit = {
            tuple(sorted(inverse_class[permutation[t]] for t in quad))
            for permutation in automorphisms
        }
        seen_orbits.update(orbit)
        canonical = min(orbit)
        A = [[[e, els[t]] for t in canonical]]
        # The zero opposite component is a probe, never a submitted code.
        rows = lifted_matrices(G, A, [[[]]])[0]
        brief = _binary_brief(rows, q)
        if brief is None:
            continue
        entry = brief + (canonical,)
        signature = brief[3]
        old = briefs.get(signature)
        if old is None or entry < old:
            briefs[signature] = entry
    book = sorted(briefs.values())
    if not book:
        raise ValueError("no connected triangle-free binary brief")
    return book


def _same_basis_four(rows):
    return sum(
        overlap * (overlap - 1) // 2
        for i, row in enumerate(rows)
        for other in rows[:i]
        for overlap in [(row & other).bit_count()]
    )


def _retired_balanced_ctx(rng, G, idx):
    """Deterministic structural cards and verified two-basis hook screening.

    All catalogue state is local. Rebuilding the same prefix reserves
    distinct component-fingerprint pairs without mutable cross-slot state.
    """
    from itertools import combinations
    from collections import Counter
    import hashlib

    q = G.q
    counts = [[2, 2, 2, 3]]
    assert _grid_weights(counts, counts) <= min(12, W_MAX)

    # The generated subgroups are respectively nonabelian order 32,
    # C4 x C4 of order 16, C8 x C2 of order 16, and cyclic order 8.
    triples = (
        (G.e, (1, 0), (0, 1)),
        (G.e, (2, 0), (0, 1)),
        (G.e, (1, 0), (2, 2)),
        (G.e, (1, 0), (3, 0)),
    )
    type_pairs = (
        (0, 0), (0, 1), (0, 2), (1, 2), (1, 1),
        (0, 3), (2, 2), (1, 3), (2, 3), (3, 3),
    )
    order_profiles = ((4, 8, 8), (8, 8, 8))

    def element_order(g):
        word = G.e
        for power in range(1, q + 1):
            word = G.mul(word, g)
            if word == G.e:
                return power
        raise ValueError("invalid finite-group element")

    representatives = [
        t for t, g in enumerate(G.elements)
        if t < G.idx(G.inv(g)) and element_order(g) in (4, 8)
    ]
    orders = {t: element_order(G.elements[t])
              for t in representatives}

    def component_rows(grid):
        # The zero opposite side is only a probe, never a submitted code.
        rows = lifted_matrices(G, grid, [[[]]])[0]
        mask = (1 << (4 * q)) - 1
        return [row & mask for row in rows]

    def column_syndromes(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def rooted_fingerprint(rows):
        """Uncolored component Tanner refinement with one rooted check.

        Lift translations act transitively on component checks. Thus an
        isomorphism can be rooted at check zero. Equal hashes are discarded
        conservatively; unequal hashes are not a full-code iso certificate.
        """
        width = 4 * q
        adjacency = [[] for _ in range(q + width)]
        columns = column_syndromes(rows, width)
        for c, syndrome in enumerate(columns):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(q + c)
                adjacency[q + c].append(r)
                syndrome ^= bit
        colors = ([0] * q +
                  [1 if len(adjacency[q + c]) == 2 else 2
                   for c in range(width)])
        colors[0] = 3
        digest = hashlib.sha256()
        for _ in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(len(adjacency))
            ]
            histogram = sorted(Counter(signatures).items())
            digest.update(repr(histogram).encode("ascii"))
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            refined = [palette[s] for s in signatures]
            if refined == colors:
                break
            colors = refined
        return digest.hexdigest()

    def six_cycles(rows):
        # Exact here because component Tanner four-cycles are excluded.
        adjacency = [0] * q
        for i, row in enumerate(rows):
            for j in range(i):
                if row & rows[j]:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(q):
            for j in range(i + 1, q):
                if (adjacency[i] >> j) & 1:
                    triangles += (
                        (adjacency[i] & adjacency[j]) >> (j + 1)
                    ).bit_count()
        return triangles - q  # one star triangle per trinomial column

    books = {}
    for geometry in range(len(triples)):
        for policy, wanted in enumerate(order_profiles):
            unique = {}
            for generators in combinations(representatives, 3):
                if tuple(sorted(orders[t] for t in generators)) != wanted:
                    continue
                grid = [[[G.e, G.elements[t]] for t in generators] +
                        [list(triples[geometry])]]
                rows = component_rows(grid)
                if _same_basis_four(rows) or gf2_rank(rows) != q:
                    continue
                short = _short_logical_counts(rows, [], 4 * q)
                if any(short[:3]):
                    continue
                fingerprint = rooted_fingerprint(rows)
                quality = (short[3], six_cycles(rows))
                labeling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for cell in grid[0])
                entry = (quality, fingerprint, labeling, grid)
                old = unique.get(fingerprint)
                if old is None or (quality, labeling) < (old[0], old[2]):
                    unique[fingerprint] = entry
            book = sorted(unique.values(),
                          key=lambda e: (e[0], e[1], e[2]))
            if not book:
                raise ValueError("no screened component for structural card")
            books[geometry, policy] = book[:8]

    # Pair fingerprints are unordered: A/B exchange must not consume slots.
    catalogues = []
    for policy in range(len(order_profiles)):
        for ta, tb in type_pairs:
            candidates = {}
            for a in books[ta, policy]:
                for b in books[tb, policy]:
                    identity = tuple(sorted((a[1], b[1])))
                    coarse = (
                        max(a[0][0], b[0][0]), a[0][0] + b[0][0],
                        max(a[0][1], b[0][1]), a[0][1] + b[0][1],
                    )
                    item = (coarse, identity, a[3], b[3])
                    old = candidates.get(identity)
                    if old is None or coarse < old[0]:
                        candidates[identity] = item
            catalogues.append(sorted(
                candidates.values(), key=lambda e: (e[0], e[1])))

    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx + 1):
        card = slot % len(catalogues)
        available = [item for item in catalogues[card]
                     if item[1] not in reserved]
        if not available:
            # Extra harness slots extend the structural catalogue rather
            # than reproduce a labeling or an already reserved orbit.
            alternatives = [
                (item[0], item[1], other, item)
                for other, catalogue in enumerate(catalogues)
                for item in catalogue if item[1] not in reserved
            ]
            alternatives.sort(key=lambda e: e[:3])
            if not alternatives:
                raise ValueError("unused component-pair catalogue exhausted")
            card = alternatives[0][2]
            available = [item for item in catalogues[card]
                         if item[1] not in reserved]
        chosen = available[:4]
        actual_card = card
        reserved.update(item[1] for item in chosen)

    def quotient_columns(stabilizers, width):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        positions = sorted(pivots, reverse=True)
        residuals = []
        for c in range(width):
            word = 1 << c
            for p in positions:
                if (word >> p) & 1:
                    word ^= pivots[p]
            residuals.append(word)
        return residuals

    def hook_profile(checks, stabilizers, width):
        """Exact logical pair collisions; shared-check hooks are surrogates.

        Quotient labels separate stabilizers from logical differences.
        No assumption is made about the ASC CNOT order realizing a pair.
        """
        syndrome = column_syndromes(checks, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        spread = column_syndromes(stabilizers, width)
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        pairs = {}
        three = four = double_hook = hook_exposure = 0

        for i in range(width):
            si, ri, hi = syndrome[i], residual[i], spread[i]
            for j in range(i):
                s = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(s)
                if third is not None and third != label:
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

        return three // 3, double_hook, hook_exposure, four // 3

    best = None
    for coarse, identity, A, B in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 9 * q:
            continue
        ar, br = _single_row_components(hx, hz, q)
        if gf2_rank(ar) != q or gf2_rank(br) != q:
            continue
        if any(_short_logical_counts(br, [], 4 * q)[:3]):
            continue
        px = hook_profile(hz, hx, n)
        pz = hook_profile(hx, hz, n)
        if px is None or pz is None or px[0] or pz[0]:
            continue

        # Prioritize possible two-hook logical collisions, then one-hook
        # exposure, then ordinary multiplicity. Protect the worse basis.
        key = tuple(v for t in range(4)
                    for v in (max(px[t], pz[t]), px[t] + pz[t]))
        key += (coarse, identity)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, px, pz)

    if best is None:
        raise ValueError("no full-rank balanced hook-screened finalist")
    key, A, B, hx, hz, k, px, pz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert css_ok(hx, hz)
    return {"idx": idx, "case": actual_card, "k": k,
            "screen": key, "hook_profile": (px, pz),
            "cells": {"A": A, "B": B}}


def _wide_columns(rows, width):
    """Column syndromes of verified matrices."""
    out = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= marker
            row ^= bit
    return out


def _wide_quotient(rows, width):
    """Linear, fully reduced representatives modulo the stabilizer span."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                pivots[p] = row
                break
            row ^= previous

    # Each pivot row has only lower-index terms besides its pivot.
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


def _wide_fault_profile(checks, stabilizers, width):
    """Truncated-channel ambiguity, not measured pL or an ASC simulation.

    Logical classes use the actual stabilizer quotient. Spread masses
    are distributed over all pairs of an opposite check, rather than
    assigning every pair a full ancilla-fault probability.
    """
    syndrome = _wide_columns(checks, width)
    spread = _wide_columns(stabilizers, width)
    residual = _wide_quotient(stabilizers, width)
    if 0 in syndrome or len(set(syndrome)) != width:
        return None

    degrees = [1 + a.bit_count() + b.bit_count()
               for a, b in zip(syndrome, spread)]
    weights = {row.bit_count() for row in stabilizers}
    if len(weights) != 1:
        raise ValueError("all-doubleton bed unexpectedly has irregular checks")
    weight = next(iter(weights))
    spread_mass = 4096 // (weight * (weight - 1) // 2)
    buckets = {0: {0: 100000000}}

    def add(s, label, mass):
        classes = buckets.setdefault(s, {})
        classes[label] = classes.get(label, 0) + mass

    for i, s in enumerate(syndrome):
        add(s, residual[i], 2000 * degrees[i])

    for i, si in enumerate(syndrome):
        ri, hi = residual[i], spread[i]
        for j in range(i):
            s = si ^ syndrome[j]
            label = ri ^ residual[j]
            if s == 0 and label:
                return None
            mass = degrees[i] * degrees[j]
            mass += spread_mass * (hi & spread[j]).bit_count()
            add(s, label, mass)

    loss = worst = ambiguous = 0
    spectrum = {}
    for classes in buckets.values():
        if len(classes) > 1:
            error = sum(classes.values()) - max(classes.values())
            loss += error
            worst = max(worst, error)
            ambiguous += 1
            spectrum[error] = spectrum.get(error, 0) + 1
    return loss, worst, ambiguous, tuple(sorted(spectrum.items()))


def NEW_CTX(rng, G, idx):
    """Enumerated unequal component structures, then joint fault screening.

    Catalogue state is local and deterministic. Slot identity selects a
    different structural pair, not another random draw of one optimum.
    """
    from itertools import combinations
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 28
    sa, sb = [[2] * 4], [[2] * 5]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)

    # Cache singleton actions exclusively from scaffold probes.
    # XOR composition below does not introduce a separate lift convention.
    actions = {"A": [], "B": []}
    for g in G.elements:
        x = lifted_matrices(G, [[[g]]], [[[]]])[0]
        z = lifted_matrices(G, [[[]]], [[[g]]])[1]
        mask = (1 << q) - 1
        actions["A"].append(tuple(row & mask for row in x))
        actions["B"].append(tuple(row & mask for row in z))
    identity = G.idx(G.e)

    def component(generators, side):
        rows = [0] * q
        for j, number in enumerate(generators):
            first = actions[side][identity]
            second = actions[side][number]
            for t in range(q):
                rows[t] ^= (first[t] ^ second[t]) << (j * q)
        return rows

    def powers(g, count):
        out = [G.e]
        for _ in range(count):
            out.append(G.mul(out[-1], g))
        return out

    inverse_class = [
        min(t, G.idx(G.inv(g))) for t, g in enumerate(G.elements)
    ]
    representatives = [
        t for t, g in enumerate(G.elements)
        if inverse_class[t] == t and G.mul(g, g) != G.e
    ]

    # Genuine presentation automorphisms, used only to remove duplicate
    # joint proposals. Applying one to BOTH sides is not a new design.
    automorphisms = set()
    for x in G.elements:
        xp = powers(x, 7)
        if x == G.e or xp[7] != G.e:
            continue
        for y in G.elements:
            yp = powers(y, 4)
            if yp[4] != G.e or yp[2] == G.e:
                continue
            if G.mul(G.mul(y, x), G.inv(y)) != xp[6]:
                continue
            image = tuple(
                G.idx(G.mul(xp[a], yp[b]))
                for b in range(4) for a in range(7)
            )
            if len(set(image)) == q:
                automorphisms.add(image)
    automorphisms = sorted(automorphisms)
    assert automorphisms

    def graph_brief(rows, width):
        columns = _wide_columns(rows, width)
        if (len(set(columns)) != width or
                any(s.bit_count() != 2 for s in columns)):
            return None
        adjacency = [0] * q
        for s in columns:
            low = s & -s
            a = low.bit_length() - 1
            b = (s ^ low).bit_length() - 1
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a

        seen = frontier = 1
        while frontier:
            neighbors = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                neighbors |= adjacency[bit.bit_length() - 1]
                bits ^= bit
            frontier = neighbors & ~seen
            seen |= frontier
        if seen != (1 << q) - 1:
            return None

        triangle_sum = four_sum = 0
        neighbors = []
        for a in range(q):
            neighbors.append([
                b for b in range(q) if (adjacency[a] >> b) & 1
            ])
            for b in range(a):
                common = (adjacency[a] & adjacency[b]).bit_count()
                four_sum += common * (common - 1) // 2
                if (adjacency[a] >> b) & 1:
                    triangle_sum += common
        triangles, fours = triangle_sum // 3, four_sum // 2

        # A component is a vertex-transitive incidence graph. Rooting a
        # check is therefore invariant; degree-two data vertices can be
        # suppressed without losing the component Tanner graph.
        colors = [0] * q
        colors[0] = 1
        digest = hashlib.sha256()
        for _ in range(16):
            signatures = [
                (colors[a], tuple(sorted(colors[b] for b in neighbors[a])))
                for a in range(q)
            ]
            digest.update(repr(sorted(Counter(signatures).items())).encode())
            palette = {s: t for t, s in enumerate(sorted(set(signatures)))}
            refined = [palette[s] for s in signatures]
            if refined == colors:
                break
            colors = refined
        return (triangles, fours), digest.hexdigest()

    def book(length):
        classes = {}
        for generators in combinations(representatives, length):
            rows = component(generators, "A")
            brief = graph_brief(rows, length * q)
            if brief is None:
                continue
            quality, fingerprint = brief
            entry = classes.get(fingerprint)
            if entry is None:
                classes[fingerprint] = [quality, fingerprint, [generators]]
            elif len(entry[2]) < 8:
                entry[2].append(generators)
        entries = sorted(classes.values(), key=lambda e: (e[0], e[1]))
        if not entries:
            raise ValueError("no connected component structures")
        return entries

    abook, bbook = book(4), book(5)
    catalogue = []
    for a, aa in enumerate(abook):
        for b, bb in enumerate(bbook):
            ta, fa = aa[0]
            tb, fb = bb[0]
            # Component triangles screen d=3 vulnerabilities first.
            # Four-cycle multiplicity is only a proposal-order surrogate.
            coarse = (
                int(bool(ta or tb)),
                max(ta, tb), 5 * ta + 4 * tb,
                max(fa, fb), 5 * fa + 4 * fb,
            )
            catalogue.append((coarse, aa[1], bb[1], a, b))
    catalogue.sort()
    if idx >= len(catalogue):
        raise ValueError("distinct component-structure catalogue exhausted")
    coarse, afp, bfp, a, b = catalogue[idx]

    def joint_identity(ag, bg):
        return min(
            (
                tuple(sorted(inverse_class[p[t]] for t in ag)),
                tuple(sorted(inverse_class[p[t]] for t in bg)),
            )
            for p in automorphisms
        )

    def cells(generators):
        return [[[G.e, G.elements[t]] for t in generators]]

    pool = []
    seen = set()
    for ag in abook[a][2]:
        for bg in bbook[b][2]:
            signature = joint_identity(ag, bg)
            if signature in seen:
                continue
            seen.add(signature)
            A, B = cells(ag), cells(bg)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 588 or k != 345:
                continue

            mask = (1 << q) - 1
            ar = [
                sum(((hx[t] >> (j * 5 * q)) & mask) << (j * q)
                    for j in range(4))
                for t in range(q)
            ]
            br = [row & ((1 << (5 * q)) - 1) for row in hz[:q]]
            assert ar == component(ag, "A")
            assert br == component(bg, "B")
            assert gf2_rank(ar) == q - 1
            assert gf2_rank(br) == q - 1

            cx, cz = _same_basis_four(hx), _same_basis_four(hz)
            proposal = (max(cx, cz), cx + cz, signature)
            pool.append((proposal, A, B, hx, hz, n, k))
    pool.sort(key=lambda e: e[0])

    best = None
    for proposal, A, B, hx, hz, n, k in pool[:4]:
        px = _wide_fault_profile(hz, hx, n)
        pz = _wide_fault_profile(hx, hz, n)
        if px is None or pz is None:
            continue
        x = _short_logical_counts(hz, hx, n)
        z = _short_logical_counts(hx, hz, n)
        if any(x[:2]) or any(z[:2]):
            continue
        if coarse[0] == 0 and (x[2] or z[2]):
            continue

        # The fault ambiguity, not an optimistic distance estimate,
        # orders finalists once immediate short vulnerabilities are screened.
        key = (
            max(px[0], pz[0]), px[0] + pz[0],
            max(px[1], pz[1]), px[1] + pz[1],
            max(px[2], pz[2]), px[2] + pz[2],
            max(x[3], z[3]), x[3] + z[3], proposal,
        )
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, px, pz)

    if best is None:
        raise ValueError("no verified unequal-component fault finalist")
    key, A, B, hx, hz, k, px, pz = best
    assert css_ok(hx, hz)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert {row.bit_count() for row in hx} == {10}
    assert {row.bit_count() for row in hz} == {12}
    assert sum(row.bit_count() for row in hx + hz) * 6 == 16464
    return {
        "idx": idx, "case": (afp, bfp), "k": k,
        "screen": key, "fault_profile": (px, pz),
        "cells": {"A": A, "B": B},
    }


def CELL(rng, G, side, i, j, ctx):
    """Every cell of the new unequal shapes is precomputed and distinct."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Singleton-free distance-four screening is sufficient for rare collective C35 circuit failures: slot 0 X failed 154/250 shots, flipping 8221 logical qubits across all 323, despite 250/0/0 settlement; the witness search found 3933 weight-four operators.
# REFUTED: Distinct colored component-brief pairs distinguish full Tanner codes: the order-32 parent submitted 60 slots but produced only 13 distinct codes, including slot 2 isomorphic to slot 0; its five measured codes still scored only 3.34-3.41.
# REFUTED: Excluding logicals through weight three and prioritizing possible two-hook collisions suffices for rare collective circuit failures: the balanced order-32 parent slot 0 still failed 171/500 Z and 188/500 X shots, with failures reaching all 288 logicals.
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
