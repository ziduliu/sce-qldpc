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
CONCEPT = ("C4 semidirect C2 with inversion action, order 8, shapes (2,8)x(2,8): even-degree quartic reinforcement beside complete matching components should improve worse-basis ASC circuit fitness above 3.4785 without increasing six-round extraction work.")
GROUP = ("Retain the split metacyclic construction family, changing parameters only because the new shape factor 68 requires q<=8. Use m=4,r=2,s=3: 3^2=1 modulo 4 gives the valid order-eight dihedral group. BEDS reports an order-eight (2,8)x(2,8) bed near 3.2, so this does not establish a group advantage. Unlike odd-augmentation unit reinforcement, even column degrees retain a component parity dependency; complete verified ranks decide the resulting excess k.")
SHAPES = ("Change both sides to (2,8), with 16 checks and 64 variables per component at q=8. n=544 and kb=288 guarantee rate 9/17 in target cell 0.550 regardless of rank deficits. Require connected even-degree component rank 15, and measure complete k rather than assuming a tensor-product rank formula. Parent failures affected essentially all 288 logicals and many Acol/Bcol blocks, while Acol/Brow check enrichment remained about 1.0-1.1x; there is no persistent hotspot supporting a local repair.")
FILLING = ("A has eight matching columns and row totals eight. B has six or seven matching columns plus one or two degree-four columns, with row totals at most ten. Thus HX sums are <=8+4=12 and HZ sums <=10+2=12; six-round CNOTs are 16320 or 17280 versus the parent's 17280. Twenty contracts vary reinforcement architecture, relative-order geometry and matching exclusions. Verified component lifts exclude kernels through three and require rank 15; uncolored rooted fingerprints conservatively collapse proposals. Pure prefix reservation supplies fresh component classes, including extra slots. Two complete finalists are compared by worse-basis nontrivial pair collisions involving possible shared-check spread, modulo actual stabilizers. Target the parent's 106-188 failures in measured 250-500-shot samples and broad collective flips. Shared checks are not ASC-certified hooks, and this screen omits extraction ordering, measurement histories and relay/LSD; measured circuit pL remains decisive.")
FAMILY_NAME = "r055_metacyclic_D8_28x28_evenquartic_matching_hook_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C4 semidirect C2 with action 3 requires q=8")
    # Same split metacyclic rule: x^4=y^2=1, y*x*y^-1=x^3.
    m, r, s = 4, 2, 3
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

A_SHAPE = (2, 8)               # (ra, ca)
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


def _retired_single_row_CTX(rng, G, idx):
    """Inactive previous-shape catalogue, retained for provenance.

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


def _even_columns(rows, width):
    out = [0] * width
    for r, word in enumerate(rows):
        marker = 1 << r
        while word:
            bit = word & -word
            out[bit.bit_length() - 1] |= marker
            word ^= bit
    return out


def _even_fingerprint(rows, width, q, reinforced):
    """Uncolored component invariant, not full-code canonization.

    All degree-four variables are in the reinforced columns. Component
    translations are transitive within each such column, so taking the
    minimum over one representative per reinforced column covers their
    root orbits. Equal hashes are conservatively discarded.
    """
    from collections import Counter
    import hashlib

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, syndrome in enumerate(_even_columns(rows, width)):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            syndrome ^= bit

    fingerprints = []
    for j in reinforced:
        colors = [0] * nr + [1] * width
        colors[nr + j*q] = 2
        digest = hashlib.sha256()
        for _ in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(len(adjacency))]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        fingerprints.append(digest.hexdigest())
    return min(fingerprints)


def _even_quotient(stabilizers, width):
    """Linear remainders modulo the actual verified stabilizer span."""
    pivots = {}
    for word in stabilizers:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old

    result = [0] * width
    for c in range(width):
        row = pivots.get(c)
        if row is None:
            result[c] = 1 << c
            continue
        tail, value = row ^ (1 << c), 0
        while tail:
            bit = tail & -tail
            value ^= result[bit.bit_length() - 1]
            tail ^= bit
        result[c] = value
    return result


def _even_hook_profile(checks, stabilizers, width):
    """Exact logical collision classification; hook weights are surrogates."""
    syndrome = _even_columns(checks, width)
    if 0 in syndrome or len(set(syndrome)) != width:
        return None
    spread = _even_columns(stabilizers, width)
    residue = _even_quotient(stabilizers, width)
    singles = dict(zip(syndrome, residue))
    pairs = {}
    three = four = double_hook = one_hook = 0

    for i, si in enumerate(syndrome):
        ri, hi = residue[i], spread[i]
        for j in range(i):
            s = si ^ syndrome[j]
            label = ri ^ residue[j]
            if s == 0 and label:
                return None
            third = singles.get(s)
            if third is not None and third != label:
                three += 1
            hook = (hi & spread[j]).bit_count()
            old = pairs.get(s)
            if old is None:
                pairs[s] = [1, hook, {label: [1, hook]}]
                continue
            same = old[2].get(label)
            sc, sh = same if same is not None else (0, 0)
            different = old[0] - sc
            different_hooks = old[1] - sh
            four += different
            double_hook += hook * different_hooks
            one_hook += hook * different + different_hooks
            old[0] += 1
            old[1] += hook
            if same is None:
                old[2][label] = [1, hook]
            else:
                same[0] += 1
                same[1] += hook

    assert three % 3 == 0 and four % 3 == 0
    return three // 3, double_hook, one_hook, four // 3


def NEW_CTX(rng, G, idx):
    """Pure even-degree architecture catalogue and verified joint selection."""
    from itertools import combinations

    q = G.q
    assert q == 8 and A_SHAPE == B_SHAPE == (2, 8)
    width = 8*q
    mask = (1 << q) - 1
    els = G.elements

    def order(g):
        word = G.e
        for t in range(1, q + 1):
            word = G.mul(word, g)
            if word == G.e:
                return t
        raise ValueError("invalid metacyclic element")

    orders = [order(g) for g in els]
    nonidentity = [g for g in els if g != G.e]
    triangle_types = ((2, 4, 4), (2, 2, 2), (2, 2, 4))
    triangles = [[] for _ in triangle_types]
    for u, v in combinations(nonidentity, 2):
        relative = G.mul(G.inv(u), v)
        signature = tuple(sorted((
            orders[G.idx(u)], orders[G.idx(v)],
            orders[G.idx(relative)])))
        for geometry, wanted in enumerate(triangle_types):
            if signature == wanted:
                triangles[geometry].append([G.e, u, v])

    # Pure degree-four cells are separated by their number of order-four
    # relative pairs. Both classes have non-subgroup examples; actual
    # duplicate columns and short kernels are still screened below.
    quartets = [[], []]
    for terms in combinations(nonidentity, 3):
        cell = [G.e] + list(terms)
        long_pairs = sum(
            orders[G.idx(G.mul(G.inv(a), b))] == 4
            for a, b in combinations(cell, 2))
        if long_pairs in (1, 2):
            quartets[long_pairs - 1].append(cell)
    assert all(triangles) and all(quartets)

    missing_pairs = [[], []]
    for a, b in combinations(els, 2):
        nc = G.mul(a, b) != G.mul(b, a)
        missing_pairs[int(nc)].append((a, b))
    assert all(missing_pairs)

    # Verified singleton blocks only; no independently implemented lift.
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(els):
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][t] = tuple(word & mask for word in hx)
        actions["B"][t] = tuple(word & mask for word in hz)

    def component(grid, side):
        rows = [0] * (2*q)
        for i in range(2):
            for j, cell in enumerate(grid[i]):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, word in enumerate(block):
                        rows[i*q+t] ^= word << (j*q)
        return rows

    def translated(crng, relative):
        anchor = G.pick(crng)
        return [G.mul(anchor, g) for g in relative]

    # A is the complete bipartite matching component. B is not obtained
    # by transforming A or a previously submitted B.
    A = [[[G.e] for _ in range(8)],
         [[g] for g in els]]
    ar = component(A, "A")
    assert gf2_rank(ar) == 2*q - 1
    assert all(c.bit_count() == 2
               for c in _even_columns(ar, width))

    # 12 single-quartic, 6 balanced mixed-quartic, 2 pure-quartic briefs.
    cards = [(0, geometry, policy)
             for geometry in range(3) for policy in range(4)]
    cards += [(1, geometry, policy)
              for geometry in range(3) for policy in range(2)]
    cards += [(2, geometry, 0) for geometry in range(2)]
    assert len(cards) == 20

    def draw(crng, card):
        architecture, geometry, policy = card
        B = [[[] for _ in range(8)] for _ in range(2)]
        matching = 7 if architecture == 0 else 6

        if architecture == 0:
            missing_order = (1, 2, 4, None)[policy]
            choices = [g for t, g in enumerate(els)
                       if missing_order is None
                       or orders[t] == missing_order]
            excluded = {G._key(crng.choice(choices))}
        else:
            pair = crng.choice(
                missing_pairs[policy] if architecture == 1
                else missing_pairs[geometry])
            excluded = {G._key(g) for g in pair}
        voltages = [g for g in els if G._key(g) not in excluded]
        assert len(voltages) == matching
        for j, voltage in enumerate(voltages):
            B[0][j], B[1][j] = [G.e], [voltage]

        if architecture == 0:
            B[0][7] = translated(crng, crng.choice(triangles[geometry]))
            B[1][7] = [G.pick(crng)]
        elif architecture == 1:
            B[0][6] = translated(crng, crng.choice(triangles[geometry]))
            B[1][6] = [G.pick(crng)]
            B[0][7] = [G.pick(crng)]
            B[1][7] = translated(
                crng, crng.choice(triangles[(geometry + 1) % 3]))
        else:
            B[0][6] = translated(crng, crng.choice(quartets[geometry]))
            B[1][7] = translated(
                crng, crng.choice(quartets[1 - geometry]))

        sb = [[len(cell) for cell in row] for row in B]
        sa = [[1] * 8 for _ in range(2)]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= min(12, W_MAX)
        assert max(map(sum, sb)) <= 10
        assert all(sum(column) in (2, 4) for column in zip(*sb))
        return B

    books = []
    for number, card in enumerate(cards):
        crng = random.Random(828055103 + 104729*number)
        raw = {}
        reinforced = (7,) if card[0] == 0 else (6, 7)
        for trial in range(128):
            B = draw(crng, card)
            labeling = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for row in B for cell in row)
            if labeling in raw:
                continue
            rows = component(B, "B")
            if gf2_rank(rows) != 2*q - 1:
                continue
            short = _short_logical_counts(rows, [], width)
            if any(short[:3]):
                continue
            quality = (short[3], _same_basis_four(rows))
            raw[labeling] = (quality, labeling, B, rows)

        unique = {}
        for quality, labeling, B, rows in sorted(
                raw.values(), key=lambda entry: entry[:2]):
            identity = _even_fingerprint(
                rows, width, q, reinforced)
            entry = (quality, identity, labeling, B)
            old = unique.get(identity)
            if old is None or (quality, labeling) < (old[0], old[2]):
                unique[identity] = entry
            if len(unique) >= 32:
                break
        books.append(sorted(unique.values(), key=lambda entry: entry[:3]))

    # Reconstruct the same reservation prefix in each pure call.
    # Component fingerprints are conservative distinctions, not a
    # certificate against arbitrary full quantum Tanner isomorphisms.
    reserved = set()
    chosen = None
    for slot in range(idx + 1):
        selected = []
        preferred = slot % len(cards)
        visit = [preferred] + [
            c for c in range(len(cards)) if c != preferred]
        for card_number in visit:
            for entry in books[card_number]:
                if entry[1] in reserved:
                    continue
                reserved.add(entry[1])
                selected.append((card_number, entry))
                if len(selected) == 2:
                    break
            if len(selected) == 2:
                break
        if not selected:
            raise ValueError("fresh even-quartic component classes exhausted")
        chosen = selected

    best = None
    for card_number, (coarse, identity, labeling, B) in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k < 288:
            continue

        # Confirm cached components in the actual evaluator column order.
        actual_a = [
            sum(((hx[(i*8)*q+t] >> (j*8*q)) & mask) << (j*q)
                for j in range(8))
            for i in range(2) for t in range(q)]
        actual_b = [word & ((1 << width)-1) for word in hz[:2*q]]
        assert actual_a == ar
        assert actual_b == component(B, "B")

        px = _even_hook_profile(hz, hx, n)
        pz = _even_hook_profile(hx, hz, n)
        if px is None or pz is None or px[0] or pz[0]:
            continue
        cnots = 6 * sum(word.bit_count() for word in hx + hz)

        # Protect the worse basis's possible correlated logical faults.
        # Normalization follows the per-logical objective only as a
        # surrogate; none of these quantities is measured circuit pL.
        key = tuple(
            value for t in range(1, 4)
            for value in (max(px[t], pz[t]) / k,
                          (px[t] + pz[t]) / k))
        key += (cnots, coarse, identity)
        if best is None or key < best[0]:
            best = (key, B, hx, hz, k, cnots, card_number, identity, px, pz)

    if best is None:
        raise ValueError("no verified even-quartic target-rate finalist")
    key, B, hx, hz, k, cnots, card_number, identity, px, pz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert all(len({G._key(g) for g in cell}) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    assert max(word.bit_count() for word in hx + hz) <= min(12, W_MAX)
    assert cnots in (16320, 17280)
    assert 40*k > 21*544
    assert css_ok(hx, hz)
    return {"idx": idx, "case": cards[card_number], "k": k,
            "screen": key, "component_class": identity,
            "hook_profile": (px, pz), "cnots": cnots,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return every complete cell of the selected two-row architecture."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Singleton-free distance-four screening is sufficient for rare collective C35 circuit failures: slot 0 X failed 154/250 shots, flipping 8221 logical qubits across all 323, despite 250/0/0 settlement; the witness search found 3933 weight-four operators.
# REFUTED: Distinct colored component-brief pairs distinguish full Tanner codes: the order-32 parent submitted 60 slots but produced only 13 distinct codes, including slot 2 isomorphic to slot 0; its five measured codes still scored only 3.34-3.41.
# REFUTED: Balanced unit-reinforced C8:C4(action 5) components and possible-two-hook screening suffice for rare collective memory failures: fitness was 3.4785, measured samples had 106-188 failures in 250-500 shots, essentially all 288 logicals were affected, and both bases retained abundant weight-four witnesses.
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
