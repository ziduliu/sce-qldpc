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
CONCEPT = ("Generalized quaternion Q32, order 32, shapes (1,4)x(1,4): asymmetric (2,2,3,3) and (2,2,2,2) components selected by noisy-syndrome logical-failure holdouts should improve worse-basis ASC fitness above 3.5122 at unchanged total extraction load.")
GROUP = ("Frozen L1: keep Q32, order 32 and its encoding byte-for-byte. This exact bed measured 3.5122. Odd-augmentation trinomials are units in the characteristic-two algebra of this finite 2-group, supplying full-rank A cells; a connected binary incidence component B has rank 31. These facts predict k=291 after splitting the surjective A map, but complete scaffold ranks, not the prediction, decide acceptance.")
SHAPES = ("Frozen equal (1,4) components have 32 checks and 128 variables each. n=544 and kb=288 guarantee rate 9/17 in cell 0.550; require complete verified k=291, three above the bound. The size factor seventeen permits q<=35 and the fixed group uses q=32. Parent failures were diffuse across Acol/Bcol data and Acol/Brow checks, mostly at 1.0-1.1x enrichment. Slot 3 Z failed 116/250 shots; slot 1's worst X logical flipped 76/500 times. These readings do not establish a persistent block hotspot.")
FILLING = ("A receives (2,2,3,3), B receives (2,2,2,2): component row totals ten/eight give maximum X/Z check weights twelve/eleven and preserve 17280 six-round CNOTs. Twenty contracts cross five pairs of trinomial relative-order geometries with four connected, triangle-free binary B order profiles. A uses one long rotational binomial, one quaternion binomial and two independently varied trinomials. Verified singleton actions screen component cycles, short kernels and ranks. Pure prefix reservation collapses full Tanner fingerprints rooted at EVERY maximum-degree data vertex, without assuming nonabelian full-code root transitivity. Two finalists undergo common-stream two-interval noisy-syndrome BP and independent holdouts; actual stabilizer membership classifies settled residuals. Target fewer failed shots and collective flips, not larger distance alone. Degree-weighted noise and randomly ordered opposite-check suffixes are uncalibrated surrogates: actual ASC ordering, six-round histories and relay/LSD are omitted. The new check-degree histogram differs from every symmetric parent slot, but the evaluator remains authoritative for the complete archive.")
FAMILY_NAME = "r055_Q32_14x14_asymmetric_odd2_even4_noisysyndrome_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("generalized quaternion Q32 requires q=32")
    # (a,b)=x^a*y^b, x^16=1, y^2=x^8, y*x*y^-1=x^-1.
    # The carry for multiplying two y-coset elements is central.
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        return (int(g[0]) % 16, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + 8*b*d) % 16,
                (b+d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a+8) % 16, 1) if b else ((-a) % 16, 0)

    def idx(g):
        a, b = red(g)
        return 16*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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


def _retired_s3_ctx(rng, G, idx):
    """Inactive S3 x C5 catalogue retained for provenance; never called."""
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


def _q32_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= marker
            row ^= bit
    return out


def _q32_coarse(rows, width):
    """Cheap exact cycle and connectivity tests on verified components.

    With Tanner C4 excluded, a weight-three kernel can only be a
    triangle of degree-two columns: two degree-three columns would
    have to share two checks, which would already violate C4.
    """
    q = len(rows)
    cs = _q32_columns(rows, width)
    if 0 in cs or len(set(cs)) != width:
        return None

    adjacency = [0] * q
    binary = [0] * q
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap > 1:
                return None
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    for syndrome in cs:
        if syndrome.bit_count() == 2:
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            binary[a] |= 1 << b
            binary[b] |= 1 << a

    def triangles(graph):
        total = 0
        for i, neighbors in enumerate(graph):
            lower = neighbors & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                total += (neighbors & graph[j]
                          & ((1 << j) - 1)).bit_count()
                lower ^= bit
        return total

    if triangles(binary):
        return None
    reached = frontier = 1
    while frontier:
        following = 0
        while frontier:
            bit = frontier & -frontier
            following |= adjacency[bit.bit_length() - 1]
            frontier ^= bit
        frontier = following & ~reached
        reached |= frontier
    if reached != (1 << q) - 1:
        return None

    stars = sum(d * (d-1) * (d-2) // 6
                for d in (s.bit_count() for s in cs))
    six = triangles(adjacency) - stars
    assert six >= 0
    return six


def _q32_fingerprint(rows, width):
    """Rooted uncolored component invariant, not full-code canonization.

    Single-row component checks form one regular-action orbit. Equal
    hashes are conservatively collapsed; no saved code is transformed
    into another submitted slot.
    """
    from collections import Counter
    import hashlib

    q = len(rows)
    adjacency = [[] for _ in range(q + width)]
    for c, syndrome in enumerate(_q32_columns(rows, width)):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            adjacency[r].append(q+c)
            adjacency[q+c].append(r)
            syndrome ^= bit
    colors = [0] * q + [1] * width
    colors[0] = 2
    digest = hashlib.sha256()
    for _ in range(12):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in adjacency[v])))
            for v in range(q + width)]
        digest.update(repr(sorted(
            Counter(signatures).items())).encode("ascii"))
        palette = {s: t for t, s in enumerate(
            sorted(set(signatures)))}
        colors = [palette[s] for s in signatures]
    return digest.hexdigest()


def _q32_fault_profile(checks, stabilizers, n):
    """Exact logical collision classification with uncalibrated masses.

    Shared opposite checks mark possible spread pairs, not actual ASC
    suffixes. Quotient labels distinguish stabilizers from logicals.
    """
    cs = _q32_columns(checks, n)
    spread = _q32_columns(stabilizers, n)
    if 0 in cs or len(set(cs)) != n:
        return None

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    # Each pivot's tail only uses lower coordinates, permitting a
    # linear-time dynamic construction of all quotient columns.
    residue = [0] * n
    for c in range(n):
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

    singles = dict(zip(cs, residue))
    exposure = [1 + s.bit_count() + h.bit_count()
                for s, h in zip(cs, spread)]
    buckets = {}
    channel = {0: {0: 100000000}}
    three = double_hook = hook_exposure = partitions = 0

    def add(s, label, mass):
        classes = channel.setdefault(s, {})
        classes[label] = classes.get(label, 0) + mass

    for i, s in enumerate(cs):
        add(s, residue[i], 500 * exposure[i])

    for i, si in enumerate(cs):
        ri, hi = residue[i], spread[i]
        for j in range(i):
            s = si ^ cs[j]
            label = ri ^ residue[j]
            third = singles.get(s)
            if third is not None and third != label:
                three += 1
            if s == 0 and label:
                return None

            hook = (hi & spread[j]).bit_count()
            add(s, label, exposure[i] * exposure[j] + 128 * hook)
            old = buckets.get(s)
            if old is None:
                buckets[s] = [1, hook, {label: [1, hook]}]
                continue

            same = old[2].get(label)
            sc, sh = same if same is not None else (0, 0)
            different = old[0] - sc
            different_hooks = old[1] - sh
            partitions += different
            double_hook += hook * different_hooks
            hook_exposure += hook * different + different_hooks
            old[0] += 1
            old[1] += hook
            if same is None:
                old[2][label] = [1, hook]
            else:
                same[0] += 1
                same[1] += hook

    loss = worst = ambiguous = 0
    spectrum = {}
    for classes in channel.values():
        if len(classes) > 1:
            error = sum(classes.values()) - max(classes.values())
            loss += error
            worst = max(worst, error)
            ambiguous += 1
            spectrum[error] = spectrum.get(error, 0) + 1
    assert three % 3 == 0 and partitions % 3 == 0
    return (three // 3, double_hook, loss, worst,
            hook_exposure, partitions // 3, ambiguous,
            tuple(sorted(spectrum.items())))


def _retired_q32_symmetric_ctx(rng, G, idx):
    """Inactive symmetric quaternion catalogue retained for provenance."""
    from itertools import combinations

    q = G.q
    assert q == 32 and A_SHAPE == B_SHAPE == (1, 4)
    width = 4*q
    mask = (1 << q)-1
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 12
    assert 9 + 3 <= min(12, W_MAX)

    els = G.elements

    def order(g):
        value = G.e
        for power in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                return power
        raise ValueError("invalid quaternion element")

    orders = [order(g) for g in els]
    representatives = [
        t for t, g in enumerate(els)
        if t < G.idx(G.inv(g)) and orders[t] in (4, 8, 16)]
    profiles = (
        (4, 16, 16), (4, 8, 16), (8, 16, 16),
        (4, 4, 16), (4, 8, 8))
    geometries = (
        (4, 4, 16), (4, 4, 8), (4, 4, 4),
        (8, 16, 16), (4, 16, 16))
    binary_books = [
        [v for v in combinations(representatives, 3)
         if tuple(sorted(orders[t] for t in v)) == profile]
        for profile in profiles]
    triangle_books = [[] for _ in geometries]
    nonidentity = [t for t, g in enumerate(els) if g != G.e]
    for u, v in combinations(nonidentity, 2):
        relative = G.idx(G.mul(G.inv(els[u]), els[v]))
        signature = tuple(sorted(
            (orders[u], orders[v], orders[relative])))
        for geometry, wanted in enumerate(geometries):
            if signature == wanted:
                triangle_books[geometry].append((u, v))
    assert all(binary_books) and all(triangle_books)

    # Cyclic triples with exclusively cyclic binary directions would
    # disconnect the component. Exclude those contracts deliberately.
    cards = [(g, p) for g in range(3) for p in range(5)]
    cards += [(3, p) for p in (0, 1, 3)]
    cards += [(4, p) for p in (0, 1)]
    assert len(cards) == 20

    # Batch singleton probes cache ONLY the scaffold's actual blocks.
    # No hand-written regular-representation convention is substituted.
    probe = [[[g] for g in els]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {"A": [], "B": []}
    for t in range(q):
        actions["A"].append(tuple(
            (row >> (t*q)) & mask for row in ax))
        actions["B"].append(tuple(
            (row >> (t*q)) & mask for row in bz))

    def component(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    books = []
    for card, (geometry, policy) in enumerate(cards):
        crng = random.Random(32160517 + 104729*card)
        raw = {}
        for trial in range(320):
            generators = crng.choice(binary_books[policy])
            u, v = crng.choice(triangle_books[geometry])
            grid = [[[G.e, els[t]] for t in generators]
                    + [[G.e, els[u], els[v]]]]
            labeling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for cell in grid[0])
            if labeling in raw:
                continue
            rows = component(grid, "A")
            six = _q32_coarse(rows, width)
            if six is not None:
                raw[labeling] = (six, labeling, grid, rows)

        unique = {}
        # Short-word work is reserved for a cycle-screened shortlist.
        proposals = sorted(raw.values(), key=lambda item: item[:2])
        for six, labeling, grid, rows in proposals[:48]:
            short = _short_logical_counts(rows, [], width)
            if any(short[:3]) or gf2_rank(rows) != q:
                continue
            identity = _q32_fingerprint(rows, width)
            quality = (short[3], six)
            entry = (quality, identity, labeling, grid)
            old = unique.get(identity)
            if old is None or (quality, labeling) < (old[0], old[2]):
                unique[identity] = entry
        books.append(sorted(
            unique.values(), key=lambda item: item[:3])[:14])

    if sum(bool(book) for book in books) < 2:
        raise ValueError("insufficient screened quaternion contracts")

    catalogues = []
    for card in range(len(cards)):
        opposite = (card + 7 + card % 3) % len(cards)
        candidates = {}
        for a in books[card]:
            for b in books[opposite]:
                identity = tuple(sorted((a[1], b[1])))
                coarse = (
                    max(a[0][0], b[0][0]), a[0][0]+b[0][0],
                    max(a[0][1], b[0][1]), a[0][1]+b[0][1])
                item = (coarse, identity, a[3], b[3])
                old = candidates.get(identity)
                if old is None or coarse < old[0]:
                    candidates[identity] = item
        catalogues.append(sorted(
            candidates.values(), key=lambda item: item[:2]))

    # Every pure call reconstructs the same reservation prefix. Extra
    # harness slots consume new class pairs rather than orbit images.
    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx+1):
        card = slot % len(cards)
        available = [item for item in catalogues[card]
                     if item[1] not in reserved]
        if not available:
            alternatives = [
                (item[0], item[1], other)
                for other, catalogue in enumerate(catalogues)
                for item in catalogue if item[1] not in reserved]
            if not alternatives:
                raise ValueError("fresh quaternion class pairs exhausted")
            card = min(alternatives)[2]
            available = [item for item in catalogues[card]
                         if item[1] not in reserved]
        chosen = available[:3]
        actual_card = card
        reserved.update(item[1] for item in chosen)

    best = None
    seen_profiles = set()
    for coarse, identity, A, B in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k != 288:
            continue
        ar, br = _single_row_components(hx, hz, q)
        assert ar == component(A, "A")
        assert br == component(B, "B")
        if _q32_coarse(br, width) is None:
            continue

        px = _q32_fault_profile(hz, hx, n)
        pz = _q32_fault_profile(hx, hz, n)
        if px is None or pz is None or px[0] or pz[0]:
            continue
        invariant = tuple(sorted((px, pz)))
        if invariant in seen_profiles:
            continue
        seen_profiles.add(invariant)

        # Possible two-hook collisions precede channel loss. Ordinary
        # quartet multiplicity is only a later structural tie-break.
        key = tuple(value for t in range(7)
                    for value in (max(px[t], pz[t]), px[t]+pz[t]))
        key += (coarse, identity)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, px, pz, invariant)

    if best is None:
        raise ValueError("no exact-rate quaternion hook-screened finalist")
    key, A, B, hx, hz, px, pz, invariant = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert all(len(set(cell)) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    assert max(row.bit_count() for row in hx+hz) == 12
    assert 6 * sum(row.bit_count() for row in hx+hz) == 17280
    assert css_ok(hx, hz)
    return {"idx": idx, "case": cards[actual_card], "k": 288,
            "screen": key, "fault_profile": (px, pz),
            "invariant": invariant, "cells": {"A": A, "B": B}}


def NEW_CTX(rng, G, idx):
    """Fresh asymmetric contracts and noisy-syndrome holdout selection."""
    from itertools import combinations
    from collections import Counter
    import hashlib
    import numpy as np

    q = G.q
    assert q == 32 and A_SHAPE == B_SHAPE == (1, 4)
    width, mask = 4*q, (1 << q)-1
    sa, sb = [[2, 2, 3, 3]], [[2, 2, 2, 2]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert 12 <= W_MAX
    els = G.elements

    def order(g):
        value = G.e
        for power in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                return power
        raise ValueError("invalid quaternion element")

    orders = [order(g) for g in els]
    representatives = [
        t for t, g in enumerate(els)
        if t < G.idx(G.inv(g)) and orders[t] in (4, 8, 16)]
    rotations = {
        length: [t for t in representatives
                 if els[t][1] == 0 and orders[t] == length]
        for length in (8, 16)}
    quaternion_directions = [
        t for t in representatives if els[t][1] == 1]

    geometries = (
        (4, 4, 16), (4, 4, 8), (4, 4, 4),
        (8, 16, 16), (4, 16, 16))
    triangle_books = [[] for _ in geometries]
    nonidentity = [t for t, g in enumerate(els) if g != G.e]
    for u, v in combinations(nonidentity, 2):
        relative = G.idx(G.mul(G.inv(els[u]), els[v]))
        signature = tuple(sorted(
            (orders[u], orders[v], orders[relative])))
        for geometry, wanted in enumerate(geometries):
            if signature == wanted:
                triangle_books[geometry].append((u, v))
    assert all(triangle_books)

    # Avoid pairing two geometries that necessarily repeat x^4
    # as an internal direction. Those would force component C4.
    a_contracts = ((0, 0), (0, 1), (0, 3), (1, 3), (2, 3))
    b_profiles = (
        (4, 16, 16, 16),
        (4, 4, 16, 16),
        (4, 4, 8, 16),
        (4, 4, 4, 16))
    cards = [(a, b) for a in range(5) for b in range(4)]
    assert len(cards) == 20

    # Cache only blocks returned by the verified construction.
    probe = [[[g] for g in els]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {
        "A": [tuple((row >> (t*q)) & mask for row in ax)
              for t in range(q)],
        "B": [tuple((row >> (t*q)) & mask for row in bz)
              for t in range(q)]}

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    def labeling(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for cell in grid[0])

    def words4(rows):
        """Exact classical weight-four count after coarse exclusion <=3.

        Distinct columns make matching pairs disjoint. Each four-set
        has three pair partitions. No distance estimate is substituted.
        """
        columns = _q32_columns(rows, width)
        singles = set(columns)
        assert 0 not in singles and len(singles) == width
        pairs = {}
        triples = collisions = 0
        for i, x in enumerate(columns):
            for y in columns[:i]:
                syndrome = x ^ y
                triples += int(syndrome in singles)
                previous = pairs.get(syndrome, 0)
                collisions += previous
                pairs[syndrome] = previous + 1
        assert triples == 0 and collisions % 3 == 0
        return collisions // 3

    a_books = []
    for card, (first, second) in enumerate(a_contracts):
        crng = random.Random(320291701 + 104729*card)
        length = 8 if card in (1, 4) else 16
        raw = {}
        for trial in range(640):
            rotation = crng.choice(rotations[length])
            direction = crng.choice(quaternion_directions)
            u, v = crng.choice(triangle_books[first])
            x, y = crng.choice(triangle_books[second])
            grid = [[[G.e, els[rotation]],
                     [G.e, els[direction]],
                     [G.e, els[u], els[v]],
                     [G.e, els[x], els[y]]]]
            key = labeling(grid)
            if key in raw:
                continue
            rows = component(grid, "A")
            six = _q32_coarse(rows, width)
            if six is not None and gf2_rank(rows) == q:
                raw[key] = (six, key, grid, rows)

        unique = {}
        # Refine a cycle-screened bank, not hundreds of orbit copies
        # of the single lowest-energy labeling.
        for six, key, grid, rows in sorted(
                raw.values(), key=lambda item: item[:2])[:96]:
            quality = (words4(rows), six)
            identity = _q32_fingerprint(rows, width)
            entry = (quality, identity, key, grid)
            old = unique.get(identity)
            if old is None or (quality, key) < (old[0], old[2]):
                unique[identity] = entry
        a_books.append(sorted(
            unique.values(), key=lambda item: item[:3])[:16])

    b_books = []
    for profile in b_profiles:
        raw = []
        for generators in combinations(representatives, 4):
            if tuple(sorted(orders[t] for t in generators)) != profile:
                continue
            grid = [[[G.e, els[t]] for t in generators]]
            rows = component(grid, "B")
            six = _q32_coarse(rows, width)
            if six is None or gf2_rank(rows) != q-1:
                continue
            quality = (words4(rows), six)
            raw.append((quality, labeling(grid), grid, rows))
        unique = {}
        for quality, key, grid, rows in sorted(
                raw, key=lambda item: item[:2]):
            identity = _q32_fingerprint(rows, width)
            if identity not in unique:
                unique[identity] = (quality, identity, key, grid)
            if len(unique) >= 16:
                break
        b_books.append(sorted(
            unique.values(), key=lambda item: item[:3]))

    if not any(a_books) or not any(b_books):
        raise ValueError("no screened asymmetric component contracts")

    catalogues = []
    for acard, bcard in cards:
        entries = {}
        for a in a_books[acard]:
            for b in b_books[bcard]:
                identity = tuple(sorted((a[1], b[1])))
                coarse = (
                    max(a[0][0], b[0][0]), a[0][0]+b[0][0],
                    max(a[0][1], b[0][1]), a[0][1]+b[0][1])
                entry = (coarse, identity, a[3], b[3])
                old = entries.get(identity)
                if old is None or coarse < old[0]:
                    entries[identity] = entry
        catalogues.append(sorted(
            entries.values(), key=lambda item: item[:2]))

    def full_fingerprint(hx, hz, n):
        """Invariant rooted at every structurally eligible data vertex.

        Do NOT assume the nonabelian full Tanner graph is transitive
        on sector two. X/Z colors are merged to allow their exchange.
        Integer multiset hashes accelerate refinement; equal hashes
        are discarded conservatively, not treated as an iso proof.
        """
        checks = hx + hz
        nr, nv = len(checks), len(checks)+n
        neighbors = [[] for _ in range(nv)]
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                neighbors[r].append(nr+c)
                neighbors[nr+c].append(r)
                row ^= bit
        data_degrees = [len(neighbors[nr+c]) for c in range(n)]
        maximum = max(data_degrees)
        roots = [nr+c for c, d in enumerate(data_degrees)
                 if d == maximum]
        assert maximum == 18 and len(roots) == q
        padded = np.full(
            (nv+1, max(map(len, neighbors))), nv, dtype=np.int32)
        for v, ns in enumerate(neighbors):
            padded[v, :len(ns)] = ns
        degrees = np.asarray(
            [len(ns) for ns in neighbors]+[0], dtype=np.uint64)
        colors = np.zeros((len(roots), nv+1), dtype=np.uint64)
        colors[:, :nr] = 1
        colors[:, nr:nv] = 2
        colors[np.arange(len(roots)), roots] = 4
        c1 = np.uint64(0x9E3779B185EBCA87)
        c2 = np.uint64(0xC2B2AE3D27D4EB4F)
        c3 = np.uint64(0x165667B19E3779F9)

        def mix(values):
            values = values ^ (values >> 30)
            values = values * np.uint64(0xBF58476D1CE4E5B9)
            values = values ^ (values >> 27)
            values = values * np.uint64(0x94D049BB133111EB)
            return values ^ (values >> 31)

        for _ in range(12):
            messages = mix(colors[:, padded])
            first = messages.sum(axis=2, dtype=np.uint64)
            second = (messages*messages).sum(axis=2, dtype=np.uint64)
            colors = mix(colors*c1 ^ first
                         ^ second*c2 ^ degrees[None, :]*c3)
            colors[:, nv] = 0
        descriptions = sorted(
            row.tobytes() for row in np.sort(colors[:, :nv], axis=1))
        digest = hashlib.sha256()
        for description in descriptions:
            digest.update(description)
        return digest.hexdigest()

    # Reconstruct the same prefix in every pure call. Reserve both
    # finalists, including the one not emitted. Additional slots
    # consume new full-graph invariant classes, not fresh orbit draws.
    reserved_pairs, reserved_graphs = set(), set()
    cursors = [0]*len(cards)
    chosen = None
    for slot in range(idx+1):
        selected = []
        preferred = slot % len(cards)
        for offset in range(len(cards)):
            card = (preferred+offset) % len(cards)
            catalogue = catalogues[card]
            while cursors[card] < len(catalogue) and len(selected) < 2:
                coarse, identity, A, B = catalogue[cursors[card]]
                cursors[card] += 1
                if identity in reserved_pairs:
                    continue
                reserved_pairs.add(identity)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 544 or k != 291:
                    continue
                graph = full_fingerprint(hx, hz, n)
                if graph in reserved_graphs:
                    continue
                reserved_graphs.add(graph)
                selected.append(
                    (coarse, graph, card, A, B, hx, hz))
            if len(selected) == 2:
                break
        if not selected:
            raise ValueError("fresh asymmetric full-Tanner book exhausted")
        chosen = selected

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two-interval noisy-syndrome BP, NOT ASC circuit measurement.

        H e1 + m = s1; H(e1+e2) = s2, with perfect closing syndrome.
        Gate-degree exposure and randomly ordered proper suffix faults
        approximate extraction noise. Actual ASC schedules, six-round
        histories and relay/LSD decoding are not represented.
        """
        def sites(row):
            result = []
            while row:
                bit = row & -row
                result.append(bit.bit_length()-1)
                row ^= bit
            return result

        m, variables = len(checks), 2*n+len(checks)
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        neighborhoods = [sites(row) for row in augmented]
        row_width = max(map(len, neighborhoods))
        incidence = np.full(
            (2*m, row_width), variables, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            incidence[r, :len(ns)] = ns
        flat = incidence.ravel()
        valid = flat != variables
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (variables+1, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        sensed = np.asarray([
            c.bit_count() for c in _q32_columns(checks, n)])
        opposite = np.asarray([
            c.bit_count() for c in _q32_columns(stabilizers, n)])
        levels = np.asarray((0.65, 1.0, 1.35))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+sensed+opposite)*2/3
        measure_mu = np.asarray([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measure_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < data_p
        error[:, n:2*n] = channel.random((shots, n)) < data_p
        error[:, 2*n:variables] = (
            channel.random((shots, m)) < measure_p)

        for row in stabilizers:
            ns = sites(row)
            ordering = channel.permutation(ns)
            event_p = np.minimum(
                levels*3*0.002*len(ns)*2/3, 0.25)
            for interval in range(2):
                active = channel.random(shots) < event_p
                for shot in np.flatnonzero(active):
                    cut = int(channel.integers(1, len(ordering)))
                    error[shot, interval*n+ordering[cut:]] ^= True

        # Heuristic marginal allowance for the correlated suffixes.
        marginal_mu = data_mu + 3*0.002*opposite*0.35
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.30)
        probabilities = np.concatenate(
            (marginal_p, marginal_p,
             np.clip(measure_p, 0.002, 0.30)), axis=1)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log(
            (1-probabilities)/probabilities)
        target = (error[:, flat].reshape(
            shots, 2*m, row_width).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = error[:, :n] ^ error[:, n:2*n]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        failed = residual_weight = effort = 0
        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, 2*m, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs),
                1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, variables] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, 2*m, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            final = hard[:, :n] ^ hard[:, n:2*n]
            for local in np.flatnonzero(settled):
                difference = final[local] ^ true_final[local]
                word = 0
                for c in np.flatnonzero(difference):
                    word |= 1 << int(c)
                if nontrivial(word):
                    failed += 1
                    residual_weight += int(np.count_nonzero(difference))
                effort += iteration
            keep = ~settled
            if not np.any(keep):
                return failed, 0, residual_weight, effort
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (failed+unresolved, unresolved,
                residual_weight, effort+33*unresolved)

    def probe_quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2], x[3]+z[3])

    seed = 320291907 + 1009*idx
    finalists = []
    for coarse, graph, card, A, B, hx, hz in chosen:
        ar, br = _single_row_components(hx, hz, q)
        assert ar == component(A, "A")
        assert br == component(B, "B")
        assert gf2_rank(ar) == q and gf2_rank(br) == q-1
        assert _q32_coarse(ar, width) is not None
        assert _q32_coarse(br, width) is not None

        tx = noisy_probe(hz, hx, 544, seed, 48)
        tz = noisy_probe(hx, hz, 544, seed+1, 48)
        x = noisy_probe(hz, hx, 544, seed+1000003, 96)
        z = noisy_probe(hx, hz, 544, seed+1000004, 96)
        # Independent holdout leads. Training and short-word counts
        # only break ties; neither is the evaluator's circuit score.
        key = (probe_quality(x, z) + probe_quality(tx, tz)
               + coarse + (graph,))
        finalists.append(
            (key, A, B, hx, hz, card, graph, tx, tz, x, z))

    finalists.sort(key=lambda item: item[0])
    key, A, B, hx, hz, card, graph, tx, tz, x, z = finalists[0]
    actual_sa = [[len(cell) for cell in row] for row in A]
    actual_sb = [[len(cell) for cell in row] for row in B]
    assert actual_sa == sa and actual_sb == sb
    assert _no_dead_lines(actual_sa, actual_sb)
    assert _grid_weights(actual_sa, actual_sb) <= W_MAX
    assert all(len(set(cell)) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    assert Counter(row.bit_count() for row in hx+hz) == {
        10: 64, 11: 64, 12: 128}
    assert 6*sum(row.bit_count() for row in hx+hz) == 17280
    assert 544-gf2_rank(hx)-gf2_rank(hz) == 291
    assert css_ok(hx, hz)
    return {"idx": idx, "case": cards[card], "k": 291,
            "screen": key, "full_graph": graph,
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z),
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete asymmetric holdout-selected cell without draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Singleton-free distance-four screening is sufficient for rare collective C35 circuit failures: slot 0 X failed 154/250 shots, flipping 8221 logical qubits across all 323, despite 250/0/0 settlement; the witness search found 3933 weight-four operators.
# REFUTED: Colored component-brief distinctions prevent Tanner duplication: the order-32 parent supplied 60 slots but only 13 distinct codes; slot 2 collapsed to slot 0.
# REFUTED: Full-rank balanced S3 x C5 components plus weighted truncated-fault selection suffice for rare collective circuit failures: fitness was 3.4052, measured 250-shot samples had 100-128 failures, all measured codes retained abundant weight-four logicals, and failures reached essentially all 270 logicals despite zero excess dimension.
# REFUTED: Reserving distinct rooted-component fingerprint pairs prevents full quaternion Tanner duplication: the symmetric Q32 program supplied 60 slots but only 42 distinct codes, including slot 2 isomorphic to slot 1.
# REFUTED: Symmetric Q32 two-hook and truncated-channel selection ensures rare collective circuit failures: fitness was 3.5122, slot 3 Z failed 116/250 shots, and every measured code retained abundant weight-four logicals with broadly distributed flips.
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
