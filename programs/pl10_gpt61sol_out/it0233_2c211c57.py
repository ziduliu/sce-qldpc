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
CONCEPT = ("S3 direct product C5, order 30, shapes (1,4)x(1,4): selecting balanced components by two-basis BP logical failures under a gate-degree-weighted data-and-spread channel should improve measured circuit fitness above 3.4052.")
GROUP = ("Frozen byte-for-byte for L1: retain S3 x C5, order 30. This exact bed scored 3.4052 and produced valid exact-bound codes. Its frequent collective failures motivate a filling experiment, not an unsupported group-change hypothesis.")
SHAPES = ("Frozen equal 1x4 components have 30 checks and 120 variables each. n=510 and kb=270 guarantee rate 9/17 in cell 0.550; require actual k=270 and zero k-kb. Parent failures reached essentially every logical and were diffuse across Acol/Bcol data and Acol/Brow checks, mostly at 1.0-1.1x enrichment. There is no persistent block hotspot to repair.")
FILLING = ("Keep [2,2,2,3] on both sides: row total nine, both CSS weight sums <=12, and 16200 six-round CNOTs. Reconstruct the parent's component catalogue only to exclude all of its candidate fingerprint pairs. A new deterministic catalogue prioritizes component four-cycle avoidance before pair ambiguity. Reserve unused unordered pairs across slots, then compare joint finalists by normalized-min-sum BP logical failures under degree-weighted independent faults and possible opposite-check spread pairs. Successful corrections are classified modulo the verified stabilizer span; unresolved syndromes count as screen failures. Target the parent's 100-128 failures per 250-shot samples and broad logical flips, including logical #223 at 74/500. This small noisy-data experiment omits measurement faults and ASC ordering; it is an uncalibrated decoder-aware surrogate, not circuit pL.")
FAMILY_NAME = "r055_s3xc5_14x14_novelpair_bpchannel_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 30

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 direct product C5 requires q=30")
    # (a,b,c) = x^a y^b z^c:
    # x^3=y^2=z^5=1, y*x*y^-1=x^-1, z central.
    els = [(a, b, c) for c in range(5)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 5)

    def mul(g, h):
        a, b, c = red(g)
        d, f, t = red(h)
        return ((a + (-d if b else d)) % 3,
                (b + f) % 2, (c + t) % 5)

    def inv(g):
        a, b, c = red(g)
        return ((a if b else -a) % 3, b, (-c) % 5)

    def idx(g):
        a, b, c = red(g)
        return 6 * c + 3 * b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
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


def NEW_CTX(rng, G, idx):
    """Pure novel-pair catalogue followed by decoder-aware fault screening."""
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

    def make_books(seed, retained, new_policy):
        """Rebuilt locally; no mutable cache or saved-code transformations."""
        result = []
        for geometry in range(len(triple_types)):
            for policy in range(len(bin_profiles)):
                card = 4 * geometry + policy
                crng = random.Random(seed + 104729 * card)
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
                    if quality is None:
                        continue
                    if new_policy:
                        # After weight-three protection, avoid repeated
                        # check neighborhoods before optimizing pair ML
                        # ambiguity. BP need not rank these like ML.
                        quality = (quality[0], quality[3], quality[1],
                                   quality[2], quality[4])
                    raw[labeling] = (quality, labeling, grid, rows)

                unique = {}
                for quality, labeling, grid, rows in sorted(
                        raw.values(), key=lambda entry: entry[:2]):
                    identity = fingerprint(rows)
                    if identity not in unique:
                        unique[identity] = (
                            quality, identity, labeling, grid)
                    if len(unique) >= retained:
                        break
                result.append(sorted(
                    unique.values(), key=lambda entry: entry[:3]))
        return result

    # Reconstruct the exact parent's proposal book as a rejection set,
    # not as a source of submitted matrices. Excluding its entire pair
    # catalogue is deliberately more conservative than excluding only
    # its 57 submitted codes.
    previous_books = make_books(305509, 10, False)
    previous_pairs = set()
    for card in range(20):
        geometry, policy = divmod(card, 4)
        opposite = (4 * ((geometry + 1 + policy // 2) % 5) +
                    (policy + 1) % 4)
        for a in previous_books[card]:
            for b in previous_books[opposite]:
                previous_pairs.add(tuple(sorted((a[1], b[1]))))

    books = make_books(305509 + 10000019, 18, True)
    catalogues = []
    for card in range(20):
        geometry, policy = divmod(card, 4)
        opposite = (4 * ((geometry + 1 + policy // 2) % 5) +
                    (policy + 1) % 4)
        candidates = {}
        for a in books[card]:
            for b in books[opposite]:
                identity = tuple(sorted((a[1], b[1])))
                if identity in previous_pairs:
                    continue
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

    def bp_channel(checks, stabilizers, nc):
        """Small decoder experiment, NOT the evaluator's memory circuit.

        Uses only verified matrix incidences. Independent data faults
        scale with gate degree; possible spread pairs share an opposite
        check. Measurement faults and ASC gate suffixes are not modeled.
        Common fixed streams reduce comparison noise across finalists.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length() - 1)
                row ^= bit
            return out

        neighborhoods = [sites(row) for row in checks]
        opposite = [sites(row) for row in stabilizers]
        nr = len(neighborhoods)
        edge_variables = []
        incident = [[] for _ in range(nc)]
        for neighbors in neighborhoods:
            for c in neighbors:
                edge = len(edge_variables)
                edge_variables.append(c)
                incident[c].append(edge)

        ne = len(edge_variables)
        max_check = max(map(len, neighborhoods))
        check_edges = np.full((nr, max_check), ne, dtype=np.int32)
        check_sites = np.zeros((nr, max_check), dtype=np.int32)
        valid = np.zeros((nr, max_check), dtype=bool)
        offset = 0
        for r, neighbors in enumerate(neighborhoods):
            degree = len(neighbors)
            check_edges[r, :degree] = np.arange(offset, offset + degree)
            check_sites[r, :degree] = neighbors
            valid[r, :degree] = True
            offset += degree

        max_variable = max(map(len, incident))
        variable_edges = np.full(
            (nc, max_variable), ne, dtype=np.int32)
        for c, edges in enumerate(incident):
            variable_edges[c, :len(edges)] = edges
        edge_variables = np.asarray(edge_variables, dtype=np.int32)

        sensed_degree = np.asarray(
            [len(edges) for edges in incident], dtype=float)
        spread_degree = np.asarray(
            [s.bit_count() for s in columns(stabilizers, nc)],
            dtype=float)
        total_degree = sensed_degree + spread_degree

        shots = 48
        # Three effective data-noise levels, sixteen common shots each.
        base = np.repeat(np.asarray([0.012, 0.024, 0.040]), 16)
        independent = np.minimum(
            0.22, base[:, None] * total_degree[None, :] / 5.0)
        crng = np.random.RandomState(30553071)
        errors = crng.random_sample((shots, nc)) < independent

        # A pessimistic pair-spread channel, not a claim that ASC realizes
        # every pair. Distinct endpoints ensure no duplicate fault term.
        trigger = crng.random_sample((shots, len(opposite))) < 0.006
        for shot, r in zip(*np.nonzero(trigger)):
            neighbors = opposite[int(r)]
            if len(neighbors) >= 2:
                a, b = crng.choice(len(neighbors), 2, replace=False)
                errors[shot, neighbors[int(a)]] ^= True
                errors[shot, neighbors[int(b)]] ^= True

        mean_weight = sum(map(len, opposite)) / len(opposite)
        prior = np.minimum(
            0.24, independent +
            0.012 * spread_degree[None, :] / mean_weight)
        llr = np.log((1.0 - prior) / prior)
        target = np.logical_xor.reduce(
            errors[:, check_sites] & valid[None, :, :], axis=2)
        target_sign = np.where(target, -1.0, 1.0)

        v2c = llr[:, edge_variables].copy()
        c2v = np.zeros((shots, ne + 1), dtype=float)
        done = np.zeros(shots, dtype=bool)
        decoded = np.zeros((shots, nc), dtype=bool)
        completion = np.full(shots, 33, dtype=np.int32)
        positions = np.arange(max_check)[None, None, :]

        for iteration in range(32):
            padded = np.concatenate(
                (v2c, np.zeros((shots, 1))), axis=1)
            messages = padded[:, check_edges]
            magnitudes = np.where(
                valid[None, :, :], np.abs(messages), np.inf)
            signs = np.where(messages < 0.0, -1.0, 1.0)
            product = np.prod(signs, axis=2) * target_sign
            minimum_at = np.argmin(magnitudes, axis=2)
            smallest = np.partition(magnitudes, 1, axis=2)[:, :, :2]
            outgoing_size = np.where(
                positions == minimum_at[:, :, None],
                smallest[:, :, 1, None],
                smallest[:, :, 0, None])
            outgoing = 0.80 * product[:, :, None] * signs * outgoing_size
            outgoing = np.where(valid[None, :, :], outgoing, 0.0)
            c2v[:, :ne] = outgoing[:, valid]

            posterior = llr + c2v[:, variable_edges].sum(axis=2)
            hard = posterior < 0.0
            parity = np.logical_xor.reduce(
                hard[:, check_sites] & valid[None, :, :], axis=2)
            settled = ~np.any(parity != target, axis=1)
            newly = settled & ~done
            decoded[newly] = hard[newly]
            completion[newly] = iteration + 1
            done |= settled
            if np.all(done):
                break

            proposal = posterior[:, edge_variables] - c2v[:, :ne]
            v2c = 0.25 * v2c + 0.75 * np.clip(proposal, -24.0, 24.0)

        residual = quotient_columns(stabilizers, nc)
        failed = ~done
        logical_support = 0
        for shot in np.flatnonzero(done):
            difference = decoded[shot] ^ errors[shot]
            label = 0
            for c in np.flatnonzero(difference):
                label ^= residual[int(c)]
            if label:
                failed[shot] = True
                logical_support += int(np.count_nonzero(difference))

        # Failure, not decoder settlement alone, determines the primary
        # screen. Support and iterations are merely tie-break surrogates.
        return (int(np.count_nonzero(failed)),
                int(np.count_nonzero(~done)),
                logical_support,
                int(completion.sum()))

    best = None
    for coarse, identity, A, B in chosen:
        assert identity not in previous_pairs
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
        if px is None or pz is None or px[0] or pz[0]:
            continue

        bx = bp_channel(hz, hx, n)
        bz = bp_channel(hx, hz, n)
        # Protect the worse basis's actual surrogate-decoder failures.
        # The old ML channel is retained only as a later tie-break.
        key = tuple(
            value for t in range(4)
            for value in (max(bx[t], bz[t]), bx[t] + bz[t]))
        key += tuple(
            value for t in range(1, 4)
            for value in (max(px[t], pz[t]), px[t] + pz[t]))
        key += (coarse, identity)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, k, px, pz, bx, bz)

    if best is None:
        raise ValueError("no novel exact-rate decoder-screened finalist")
    key, A, B, hx, hz, k, px, pz, bx, bz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert sum(row.bit_count() for row in hx + hz) * 6 == 16200
    assert css_ok(hx, hz)
    return {"idx": idx, "case": actual_card, "k": k,
            "screen": key, "fault_profile": (px, pz),
            "bp_channel": (bx, bz),
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: Reducing D7 to (1,4)x(1,4) and lowering total gate count improved circuit reliability; fitness instead fell from 3.4511 to 2.9485, every measured code had d2s=3, and slot 2 failed 114/250 shots in each basis.
# REFUTED: Singleton-free distance-four screening is sufficient for rare collective C35 circuit failures: slot 0 X failed 154/250 shots, flipping 8221 logical qubits across all 323, despite 250/0/0 settlement; the witness search found 3933 weight-four operators.
# REFUTED: Colored component-brief distinctions prevent Tanner duplication: the order-32 parent supplied 60 slots but only 13 distinct codes; slot 2 collapsed to slot 0.
# REFUTED: Gate- and shared-check-weighted pair ambiguity screening suffices for rare collective S3 x C5 memory failures: fitness was 3.4052; slot 1 Z failed 128/250 shots across all 270 logicals, and slot 0 Z flipped logical #223 in 74/500 shots.
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
