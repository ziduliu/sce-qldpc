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
CONCEPT = ("S3 direct product C4, order 24, shapes (1,4)x(1,5): connected triangle-free all-doubleton components should improve worse-basis ASC circuit fitness above 3.4052 by reducing extraction gates without introducing weight-three component kernels.")
GROUP = ("Preserve the S3 x Cm direct-product construction rule. The new size factor twenty-one excludes the parent's q=30 because n would be 630; choose q=24, m=4, giving n=504. Order-four and order-twelve ratios permit triangle-free binary components. This order change is required by the shape experiment and is not evidence of a measured group advantage.")
SHAPES = ("Use unequal (1,4) and (1,5) components with 96/120 variables and 24 checks each. kb=288 guarantees rate 4/7 in the open-ended cell 0.550. Connected all-doubleton components predict complete k=297 through augmentation deficits; require the actual scaffold ranks to confirm it. Parent failures affected essentially all 270 logicals, with diffuse Acol/Bcol data and Acol/Brow check enrichment about 1.0-1.1x, rather than a persistent local hotspot.")
FILLING = ("Prioritize four A doubletons and five B doubletons: row totals eight/ten, HX weight ten, HZ weight twelve, and 14112 six-round CNOTs versus 16200. Enumerate distinct relative-support architectures under simultaneous actual group automorphisms and column gauges; require connected binary components and exclude component kernels through weight three. Favor fewer component quartets, but do not treat this as measured pL. Full-Tanner walk invariants over every high-degree data root conservatively discard equivalent designs without assuming root transitivity. If this binary book is exhausted, distinct A=[2,2,2,3], B=[1,2,2,2,2] contracts extend the portfolio at 14256 CNOTs; their lower protection is explicitly not a reliability claim. Target the parent's 100-128 failures in measured 250-shot samples and widespread collective flips by reducing overall extraction work. Screens omit ASC ordering and relay/LSD decoding.")
FAMILY_NAME = "r055_s3xc4_14x15_binary_lowload_walkdistinct_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 direct product C4 requires q=24")
    # Same direct-product rule, with central cyclic parameter m=4.
    # x^3=y^2=z^4=1, y*x*y^-1=x^-1, z central.
    els = [(a, b, c) for c in range(4)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 4)

    def mul(g, h):
        a, b, c = red(g)
        d, f, t = red(h)
        return ((a + (-d if b else d)) % 3,
                (b + f) % 2, (c + t) % 4)

    def inv(g):
        a, b, c = red(g)
        return ((a if b else -a) % 3, b, (-c) % 4)

    def idx(g):
        a, b, c = red(g)
        return 6 * c + 3 * b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
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
    """Project unequal single-row components from the verified lift."""
    assert A_SHAPE[0] == B_SHAPE[0] == 1
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    ar = [
        sum(((hx[t] >> (j * cb * q)) & mask) << (j * q)
            for j in range(ca))
        for t in range(q)
    ]
    br = [hz[t] & ((1 << (cb * q)) - 1) for t in range(q)]
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


def _retired_s3xc5_ctx(rng, G, idx):
    """Inactive parent catalogue retained for provenance; never called."""
    from itertools import combinations
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 30
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) <= min(12, W_MAX)
    width = 4 * q
    mask = (1 << q) - 1
    els = G.elements

    def order(g):
        word = G.e
        for power in range(1, q + 1):
            word = G.mul(word, g)
            if word == G.e:
                return power
        raise ValueError("invalid element order")

    orders = [order(g) for g in els]
    representatives = [
        t for t, g in enumerate(els)
        if t < G.idx(G.inv(g)) and orders[t] in (5, 10, 15)
    ]
    bin_profiles = (
        (10, 10, 15),
        (10, 15, 15),
        (5, 10, 15),
        (10, 10, 10),
    )
    triple_types = (
        (5, 10, 10),
        (5, 15, 15),
        (10, 10, 15),
        (3, 10, 10),
        (3, 15, 15),
    )
    bin_books = [
        [v for v in combinations(representatives, 3)
         if tuple(sorted(orders[t] for t in v)) == profile]
        for profile in bin_profiles
    ]
    tri_books = {geometry: [] for geometry in triple_types}
    eidx = G.idx(G.e)
    nonidentity = [t for t in range(q) if t != eidx]
    for u, v in combinations(nonidentity, 2):
        relative = G.idx(G.mul(G.inv(els[u]), els[v]))
        geometry = tuple(sorted((orders[u], orders[v], orders[relative])))
        if geometry in tri_books:
            tri_books[geometry].append((u, v))
    assert all(bin_books) and all(tri_books.values())

    # Cache only singleton blocks obtained from the verified construction.
    # Combining these blocks by XOR is not an alternative group lift.
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(els):
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][t] = tuple(row & mask for row in hx)
        actions["B"][t] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j * q)
        return rows

    def columns(rows, nc):
        out = [0] * nc
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def brief(rows):
        """Exact short component words; weighted ambiguity is a surrogate."""
        cs = columns(rows, width)
        if 0 in cs or len(set(cs)) != width or gf2_rank(rows) != q:
            return None
        singles = set(cs)
        degrees = [s.bit_count() for s in cs]
        pairs = {}
        triples = collisions = boundary = 0
        for i, si in enumerate(cs):
            for j in range(i):
                s = si ^ cs[j]
                triples += int(s in singles)
                mass = degrees[i] * degrees[j]
                old = pairs.get(s)
                if old is None:
                    pairs[s] = [1, mass, mass]
                else:
                    collisions += old[0]
                    old[0] += 1
                    old[1] += mass
                    old[2] = max(old[2], mass)
                w = s.bit_count()
                if w <= 2:
                    boundary += (3 - w) ** 2
        ambiguity = sum(total - largest
                        for number, total, largest in pairs.values())
        assert triples % 3 == 0 and collisions % 3 == 0
        return (triples // 3, ambiguity, collisions // 3,
                _same_basis_four(rows), boundary)

    def fingerprint(rows):
        """Rooted uncolored Tanner invariant, not full-code canonization.

        Component checks form one regular-action orbit, so rooting at
        check zero is invariant up to component isomorphism.
        Equal fingerprints are discarded conservatively.
        """
        adjacency = [[] for _ in range(q + width)]
        for c, syndrome in enumerate(columns(rows, width)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(q + c)
                adjacency[q + c].append(r)
                syndrome ^= bit
        colors = [0] * q + [
            1 if len(adjacency[q + c]) == 2 else 2
            for c in range(width)
        ]
        colors[0] = 3
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
        return digest.hexdigest()

    # Fixed streams make the same catalogue and reservation prefix in
    # every call. No mutable cross-slot state or saved-code orbit draws.
    books = []
    for geometry in range(len(triple_types)):
        for policy in range(len(bin_profiles)):
            card = 4 * geometry + policy
            crng = random.Random(305509 + 104729 * card)
            raw = {}
            for trial in range(96):
                generators = crng.choice(bin_books[policy])
                u, v = crng.choice(tri_books[triple_types[geometry]])
                grid = [[[G.e, els[t]] for t in generators] +
                        [[G.e, els[u], els[v]]]]
                labeling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for cell in grid[0])
                if labeling in raw:
                    continue
                rows = component(grid, "A")
                quality = brief(rows)
                if quality is not None:
                    raw[labeling] = (quality, labeling, grid, rows)
            unique = {}
            for quality, labeling, grid, rows in sorted(
                    raw.values(), key=lambda entry: entry[:2]):
                identity = fingerprint(rows)
                if identity not in unique:
                    unique[identity] = (quality, identity, labeling, grid)
                if len(unique) >= 10:
                    break
            book = sorted(unique.values(), key=lambda entry: entry[:3])
            if not book:
                raise ValueError("no full-rank component in structural card")
            books.append(book)

    catalogues = []
    for card in range(20):
        geometry, policy = divmod(card, 4)
        opposite = (4 * ((geometry + 1 + policy // 2) % 5) +
                    (policy + 1) % 4)
        candidates = {}
        for a in books[card]:
            for b in books[opposite]:
                identity = tuple(sorted((a[1], b[1])))
                quality = tuple(
                    value for t in range(5)
                    for value in (max(a[0][t], b[0][t]),
                                  a[0][t] + b[0][t]))
                entry = (quality, identity, a[3], b[3])
                old = candidates.get(identity)
                if old is None or quality < old[0]:
                    candidates[identity] = entry
        catalogues.append(sorted(
            candidates.values(), key=lambda entry: entry[:2]))

    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx + 1):
        card = slot % 20
        available = [entry for entry in catalogues[card]
                     if entry[1] not in reserved]
        if not available:
            alternatives = [
                (entry[0], entry[1], c)
                for c, catalogue in enumerate(catalogues)
                for entry in catalogue if entry[1] not in reserved
            ]
            if not alternatives:
                raise ValueError("unused component-class pairs exhausted")
            card = min(alternatives)[2]
            available = [entry for entry in catalogues[card]
                         if entry[1] not in reserved]
        chosen = available[:4]
        actual_card = card
        # Reserve every finalist, not just the eventual winner.
        reserved.update(entry[1] for entry in chosen)

    def quotient_columns(rows, nc):
        """Linear canonical remainders modulo the actual stabilizer span."""
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        residual = [0] * nc
        for c in range(nc):
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

    def fault_profile(checks, stabilizers, nc):
        """Truncated-channel ML ambiguity, not an ASC circuit simulation.

        Quotient labels distinguish logical differences from stabilizers.
        Shared opposite checks supply a possible spread contribution.
        The integer masses are heuristic; measurement faults and actual
        CNOT suffixes are deliberately not claimed to be represented.
        """
        syndrome = columns(checks, nc)
        spread = columns(stabilizers, nc)
        residual = quotient_columns(stabilizers, nc)
        if 0 in syndrome or len(set(syndrome)) != nc:
            return None
        rates = [1 + s.bit_count() + h.bit_count()
                 for s, h in zip(syndrome, spread)]
        buckets = {0: {0: 100000000}}
        singles = dict(zip(syndrome, residual))
        triples = 0

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0) + mass

        for i, s in enumerate(syndrome):
            add(s, residual[i], 500 * rates[i])
        for i, si in enumerate(syndrome):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                s = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(s)
                if third is not None and third != label:
                    triples += 1
                if s == 0 and label:
                    return None
                mass = rates[i] * rates[j]
                mass += 128 * (hi & spread[j]).bit_count()
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
        return (triples // 3, loss, worst, ambiguous,
                tuple(sorted(spectrum.items())))

    best = None
    for coarse, identity, A, B in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 510 or k != 270:
            continue
        ar, br = _single_row_components(hx, hz, q)
        assert ar == component(A, "A")
        assert br == component(B, "B")
        if brief(ar) is None or brief(br) is None:
            continue
        px = fault_profile(hz, hx, n)
        pz = fault_profile(hx, hz, n)
        if px is None or pz is None:
            continue
        # Exclude low-order logical ambiguity first; then reduce the
        # worse basis's weighted channel loss, not sampled distance.
        key = tuple(
            value for t in range(4)
            for value in (max(px[t], pz[t]), px[t] + pz[t]))
        key += (coarse, identity)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, px, pz)

    if best is None:
        raise ValueError("no target-rate fault-screened finalist")
    key, A, B, hx, hz, k, px, pz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert sum(row.bit_count() for row in hx + hz) * 6 == 16200
    assert css_ok(hx, hz)
    return {"idx": idx, "case": actual_card, "k": k,
            "screen": key, "fault_profile": (px, pz),
            "cells": {"A": A, "B": B}}


def _lowload_walk_identity(hx, hz, n):
    """Uncolored full-Tanner invariant, not a distance or circuit score.

    Use EVERY degree-eighteen data root; no nonabelian root-transitivity
    assumption is made. Unequal invariants certify nonisomorphism.
    Equal invariants are conservatively discarded.

    Integer walk moments are computed modulo 2^64. This remains a
    permutation invariant even when a higher moment wraps.
    """
    import hashlib
    import numpy as np
    from collections import Counter

    checks = hx + hz
    nr = len(checks)
    nv = nr + n
    adjacency = [[] for _ in range(nv)]
    for r, row in enumerate(checks):
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            row ^= bit

    degrees = [len(neighbors) for neighbors in adjacency]
    roots = [nr + c for c in range(n)
             if degrees[nr + c] == 18]
    assert len(roots) == 24
    assert max(degrees[:nr]) <= 12
    histogram = tuple(sorted(Counter(degrees).items()))
    partitions = [
        np.asarray([v for v, degree in enumerate(degrees) if degree == d],
                   dtype=np.int32)
        for d, number in histogram
    ]
    neighbors = np.full(
        (nv, max(degrees)), nv, dtype=np.int32)
    for v, adjacent in enumerate(adjacency):
        neighbors[v, :len(adjacent)] = adjacent

    walks = np.zeros((len(roots), nv + 1), dtype=np.uint64)
    walks[np.arange(len(roots)), roots] = 1
    features = [[] for _ in roots]
    for length in range(1, 11):
        following = walks[:, neighbors].sum(axis=2, dtype=np.uint64)
        walks[:, :nv] = following
        walks[:, nv] = 0
        if length not in (4, 6, 8, 10):
            continue
        summaries = []
        for part in partitions:
            values = walks[:, part]
            summaries.append((
                (values * values).sum(axis=1, dtype=np.uint64),
                values.max(axis=1)))
        for r, root in enumerate(roots):
            features[r].append(int(walks[r, root]))
            for squared, maximum in summaries:
                features[r].extend((int(squared[r]), int(maximum[r])))

    signature = (histogram, tuple(sorted(map(tuple, features))))
    return hashlib.sha256(repr(signature).encode("ascii")).hexdigest()


def NEW_CTX(rng, G, idx):
    """Pure unequal-shape architecture book with full-graph deduplication."""
    from itertools import combinations

    q = G.q
    assert q == 24
    assert A_SHAPE == (1, 4) and B_SHAPE == (1, 5)
    els = G.elements
    mask = (1 << q) - 1
    needed = max(20, N_SLOTS)

    # Cache ONLY singleton blocks produced by the verified construction.
    actions = {"A": [], "B": []}
    for g in els:
        ax, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, bz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"].append(tuple(row & mask for row in ax))
        actions["B"].append(tuple(row & mask for row in bz))

    def component(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j * q)
        return rows

    def order(g):
        value = G.e
        for power in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return power
        raise ValueError("invalid S3 x C4 element")

    orders = [order(g) for g in els]
    directions = [
        t for t, g in enumerate(els)
        if t < G.idx(G.inv(g)) and orders[t] >= 4
    ]
    assert len(directions) == 7

    # Actual direct-product automorphisms. The parity-to-central
    # homomorphism is included, not just S3 conjugation and z inversion.
    automorphisms = []
    for u in (1, 2):
        for v in range(3):
            for s in (1, 3):
                for twist in (0, 1):
                    image = tuple(G.idx(
                        ((u*a + v*b) % 3, b,
                         (s*c + 2*twist*b) % 4))
                        for a, b, c in els)
                    assert len(set(image)) == q
                    automorphisms.append(image)

    forms = {}

    def cell_forms(side, cell):
        labeling = tuple(sorted(G.idx(g) for g in cell))
        key = side, labeling
        if key in forms:
            return forms[key]
        result = []
        for image in automorphisms:
            values = [els[image[t]] for t in labeling]
            normalized = []
            for anchor in values:
                inverse = G.inv(anchor)
                if side == "A":
                    # Right whole-cell multiplication is an A-column gauge.
                    normalized.append(tuple(sorted(
                        G.idx(G.mul(g, inverse)) for g in values)))
                else:
                    # Left whole-cell multiplication is a B-column gauge.
                    normalized.append(tuple(sorted(
                        G.idx(G.mul(inverse, g)) for g in values)))
            result.append(min(normalized))
        forms[key] = tuple(result)
        return forms[key]

    def architecture(A, B):
        sides = [[cell_forms(side, cell) for cell in grid[0]]
                 for side, grid in (("A", A), ("B", B))]
        # Different component lengths keep the two sides distinguished.
        return min(
            tuple(tuple(sorted(cell[t] for cell in side))
                  for side in sides)
            for t in range(len(automorphisms)))

    def binary_book(length, side):
        result = []
        for generators in combinations(directions, length):
            grid = [[[G.e, els[t]] for t in generators]]
            rows = component(grid, side)
            if gf2_rank(rows) != q - 1:
                continue
            short = _short_logical_counts(rows, [], length * q)
            if any(short[:3]):
                continue
            result.append((short[3], generators, grid, rows))
        return sorted(result, key=lambda item: item[:2])

    abook = binary_book(4, "A")
    bbook = binary_book(5, "B")
    assert abook and bbook

    raw = []
    for a in abook:
        for b in bbook:
            coarse = (max(a[0], b[0]), a[0] + b[0])
            raw.append((coarse, a[1], b[1], a[2], b[2]))
    raw.sort(key=lambda item: item[:3])

    bank = []
    seen_architectures = set()
    seen_graphs = set()

    def offer(entries, mode):
        for coarse, alabel, blabel, A, B in entries:
            geometry = architecture(A, B)
            if geometry in seen_architectures:
                continue
            seen_architectures.add(geometry)
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) <= min(12, W_MAX)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 504 or (mode == 0 and k != 297):
                continue
            if k < 288:
                continue
            graph = _lowload_walk_identity(hx, hz, n)
            if graph in seen_graphs:
                continue
            seen_graphs.add(graph)
            bank.append((mode, coarse, geometry, graph, A, B, k))
            if len(bank) >= needed:
                return

    # Leading contracts all retain component protection through weight three.
    offer(raw, 0)

    if len(bank) < needed:
        # Extend the deliberate catalogue, not an orbit of a binary code.
        # These legal lower-protection contracts have 9+2 and 9+3 budgets.
        # They are placed AFTER every invariant-distinct binary proposal.
        crng = random.Random(240415701)
        triples = list(combinations(
            [t for t, g in enumerate(els) if g != G.e], 2))
        ternary = {}
        for trial in range(1600):
            generators = crng.sample(directions, 3)
            generators.sort()
            u, v = crng.choice(triples)
            labeling = tuple(generators) + (u, v)
            if labeling in ternary:
                continue
            A = [[[G.e, els[t]] for t in generators]
                 + [[G.e, els[u], els[v]]]]
            ar = component(A, "A")
            if gf2_rank(ar) != q:
                continue
            short = _short_logical_counts(ar, [], 4 * q)
            if any(short[:2]):
                continue
            ternary[labeling] = (short, labeling, A)

        candidates = sorted(
            ternary.values(),
            key=lambda item: (item[0][2], item[0][3], item[1]))[:64]
        extension = []
        for short, labeling, A in candidates:
            for bw, generators, grid, rows in abook:
                B = [[[G.e]] + [list(cell) for cell in grid[0]]]
                br = component(B, "B")
                bshort = _short_logical_counts(br, [], 5 * q)
                coarse = (
                    max(short[2], bshort[2]), short[2] + bshort[2],
                    max(short[3], bshort[3]), short[3] + bshort[3])
                extension.append((coarse, labeling, generators, A, B))
        extension.sort(key=lambda item: item[:3])
        offer(extension, 1)

    if idx >= len(bank):
        raise ValueError("fresh unequal-shape Tanner architectures exhausted")
    mode, coarse, geometry, graph, A, B, k = bank[idx]
    hx, hz, n = lifted_matrices(G, A, B)
    ar, br = _single_row_components(hx, hz, q)
    assert ar == component(A, "A")
    assert br == component(B, "B")
    assert k == n - gf2_rank(hx) - gf2_rank(hz)
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert all(len(cell) <= CELL_MAX and
               len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    cnots = 6 * sum(row.bit_count() for row in hx + hz)
    assert cnots == (14112 if mode == 0 else 14256)
    assert k / n > 0.525
    return {"idx": idx, "case": (mode, geometry), "k": k,
            "screen": coarse, "full_graph": graph, "cnots": cnots,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return every complete cell of the selected unequal-shape contract."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Singleton-free distance-four screening is sufficient for rare collective C35 circuit failures: slot 0 X failed 154/250 shots, flipping 8221 logical qubits across all 323, despite 250/0/0 settlement; the witness search found 3933 weight-four operators.
# REFUTED: Colored component-brief distinctions prevent Tanner duplication: the order-32 parent supplied 60 slots but only 13 distinct codes; slot 2 collapsed to slot 0.
# REFUTED: Full-rank balanced S3 x C5 components plus weighted truncated-fault selection suffice for rare collective circuit failures: fitness was 3.4052, measured 250-shot samples had 100-128 failures, abundant weight-four logicals remained, and failures reached essentially all 270 logicals despite zero excess dimension.
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
