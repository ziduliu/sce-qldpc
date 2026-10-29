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
CONCEPT = ("C8 semidirect C4 with action 5, order 32, shapes (1,4)x(1,4): balanced [2,2,2,3] components selected against six-round temporal logical failures should improve worse-basis ASC circuit fitness beyond 3.3714.")
GROUP = ("Frozen byte-for-byte for L1. This exact group/shape parent scored 3.3714, and the shown same-group balanced experiment reached 3.4785; these readings motivate a filling experiment, not a group advantage. Odd-augmentation elements of the characteristic-two group algebra of this finite 2-group are units, so a trinomial removes the all-binomial augmentation deficit.")
SHAPES = ("Frozen equal (1,4) components each have 32 checks and 128 variables. n=544 and kb=288 guarantee rate 9/17 in target cell 0.550; the shape allows q<=35 and the frozen group uses q=32. Require exact verified k=288 rather than the parent's 296. Parent failures and weight-four witnesses spread across Acol/Bcol data blocks, with data and Acol/Brow check enrichment mostly 1.0-1.1x. Slot0's worst X logical flipped 71/250 times; no persistent hotspot justifies a local repair.")
FILLING = ("Both sides use [2,2,2,3]: row totals nine, maximum CSS weight twelve, and 17280 six-round CNOTs versus 15360. Twenty contracts pair trinomial relative-order/subgroup geometries with four binomial-order profiles; vary the trinomial within every contract rather than reuse four fixed representatives. Exact helper-derived component actions screen connectivity, rank, four-cycles and kernels through weight three. Pure prefix reservation discards equal complete-code closed-walk invariants, including equivalences missed by colored component briefs. Two finalists receive common-stream six-round noisy-data/noisy-measurement BP probes and independent holdout probes, classifying settled final residuals modulo actual stabilizers. Target the parent's 102-130 failed shots per 250-shot samples and widespread collective flips, not distance or settlement alone. The temporal channel omits ASC ordering, gate propagation and relay/LSD decoding and is not measured pL; invariant equality is conservatively discarded rather than claimed as complete canonization.")
FAMILY_NAME = "r055_c8sd5c4_14x14_variabletri_fullwalk_temporal_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C8 semidirect C4 with action 5 requires q=32")
    # x^8=y^4=1, y*x*y^-1=x^5; elements are x^a*y^b.
    m, r, s = 8, 4, 5
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
    """Pure variable-trinomial catalogue and temporal logical screening."""
    from itertools import combinations, combinations_with_replacement
    from collections import Counter
    import hashlib
    import numpy as np

    q, els = G.q, G.elements
    assert q == 32 and A_SHAPE == B_SHAPE == (1, 4)
    width, mask = 4*q, (1 << q)-1
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 12 <= W_MAX
    ei = G.idx(G.e)

    def sites(word):
        out = []
        while word:
            bit = word & -word
            out.append(bit.bit_length()-1)
            word ^= bit
        return out

    def columns(rows, size):
        out = [0]*size
        for r, row in enumerate(rows):
            for c in sites(row):
                out[c] |= 1 << r
        return out

    # Cache exact singleton blocks from the verified construction.
    # Subsequent XOR assembly uses these blocks, never a separate lift.
    probe = [[[g] for g in els]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {
        "A": [tuple((row >> (t*q)) & mask for row in ax)
              for t in range(q)],
        "B": [tuple((row >> (t*q)) & mask for row in bz)
              for t in range(q)],
    }

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, word in enumerate(actions[side][G.idx(g)]):
                    rows[r] ^= word << (j*q)
        return rows

    def order(g):
        word = G.e
        for t in range(1, q+1):
            word = G.mul(word, g)
            if word == G.e:
                return t
        raise ValueError("invalid element order")

    orders = [order(g) for g in els]
    representatives = [
        t for t, g in enumerate(els)
        if t < G.idx(G.inv(g)) and orders[t] in (4, 8)
    ]
    profiles = ((4, 4, 4), (4, 4, 8),
                (4, 8, 8), (8, 8, 8))
    binary_books = [
        [triple for triple in combinations(representatives, 3)
         if tuple(sorted(orders[t] for t in triple)) == profile]
        for profile in profiles
    ]

    def subgroup_size(u, v):
        reached, todo = {ei}, [ei]
        while todo:
            a = todo.pop()
            for b in (u, v):
                c = G.idx(G.mul(els[a], els[b]))
                if c not in reached:
                    reached.add(c)
                    todo.append(c)
        return len(reached)

    # Enumerate all normalized trinomial supports, not four saved triples.
    triangle_books = {}
    seen_triangles = set()
    for u, v in combinations(
            [t for t in range(q) if t != ei], 2):
        cell = (G.e, els[u], els[v])
        normalized = min(
            tuple(sorted(G.idx(G.mul(g, G.inv(anchor)))
                         for g in cell))
            for anchor in cell)
        if normalized in seen_triangles:
            continue
        seen_triangles.add(normalized)
        tri = [els[t] for t in normalized]
        rows = component([[[], [], [], tri]], "A")
        if _same_basis_four(rows):
            continue
        a, b = [t for t in normalized if t != ei]
        relative = G.idx(G.mul(G.inv(els[a]), els[b]))
        signature = (
            tuple(sorted((orders[a], orders[b], orders[relative]))),
            subgroup_size(a, b),
            G.mul(els[a], els[b]) != G.mul(els[b], els[a]),
        )
        triangle_books.setdefault(signature, []).append(tri)

    geometries = sorted(triangle_books)
    if not geometries:
        raise ValueError("no four-cycle-free trinomial geometry")

    def assess(rows):
        adjacency = [0]*q
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap > 1:
                    return None
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        reached = frontier = 1
        while frontier:
            following = 0
            for r in sites(frontier):
                following |= adjacency[r]
            frontier = following & ~reached
            reached |= frontier
        if reached != mask or gf2_rank(rows) != q:
            return None
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return None
        pairs = {}
        collisions = 0
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                if value in singles:
                    return None
                old = pairs.get(value, 0)
                collisions += old
                pairs[value] = old+1
        assert collisions % 3 == 0
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            for j in sites(neighbors >> (i+1)):
                j += i+1
                triangles += (
                    (neighbors & adjacency[j]) >> (j+1)
                ).bit_count()
        # One variable-star triangle per lifted trinomial column.
        return collisions//3, triangles-q

    def component_fingerprint(rows):
        cols = columns(rows, width)
        adjacency = [[] for _ in range(q+width)]
        for c, word in enumerate(cols):
            for r in sites(word):
                adjacency[r].append(q+c)
                adjacency[q+c].append(r)
        colors = [0]*q + [word.bit_count() for word in cols]
        colors[0] = 4
        digest = hashlib.sha256()
        for _ in range(10):
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

    contracts = [(g, p) for g in range(len(geometries))
                 for p in range(len(profiles)) if binary_books[p]]
    books = []
    for number, (geometry, policy) in enumerate(contracts):
        crng = random.Random(325031701 + 104729*number)
        raw = {}
        for trial in range(128):
            generators = crng.choice(binary_books[policy])
            tri = crng.choice(triangle_books[geometries[geometry]])
            grid = [[[G.e, els[t]] for t in generators] + [list(tri)]]
            labeling = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for cell in grid[0])
            if labeling in raw:
                continue
            rows = component(grid, "A")
            quality = assess(rows)
            if quality is not None:
                raw[labeling] = (quality, labeling, grid, rows)
        unique = {}
        for quality, labeling, grid, rows in sorted(
                raw.values(), key=lambda item: item[:2]):
            identity = component_fingerprint(rows)
            if identity not in unique:
                unique[identity] = (quality, identity, grid)
            if len(unique) >= 10:
                break
        books.append(sorted(unique.values(), key=lambda item: item[:2]))

    live = [t for t, book in enumerate(books) if book]
    architecture_pairs = list(combinations_with_replacement(live, 2))
    if len(architecture_pairs) < 20:
        raise ValueError("fewer than twenty live architecture contracts")
    cards = tuple(architecture_pairs[
        t*len(architecture_pairs)//20] for t in range(20))
    assert len(set(cards)) == 20

    catalogues = []
    for ca, cb in cards:
        unique = {}
        for a in books[ca]:
            for b in books[cb]:
                identity = tuple(sorted((a[1], b[1])))
                coarse = (
                    max(a[0][0], b[0][0]), a[0][0]+b[0][0],
                    max(a[0][1], b[0][1]), a[0][1]+b[0][1])
                item = (coarse, identity, a[2], b[2])
                old = unique.get(identity)
                if old is None or coarse < old[0]:
                    unique[identity] = item
        catalogues.append(sorted(unique.values(),
                                 key=lambda item: item[:2]))

    def full_invariant(hx, hz):
        """Uncolored complete-code closed-walk invariant.

        Includes all checks without distinguishing X from Z. Arithmetic
        is exact modulo 2^64. Isomorphic Tanner graphs necessarily agree;
        equality is conservatively rejected, not treated as an iso proof.
        """
        rows = hx+hz
        m = len(rows)
        gram = np.empty((m, m), dtype=np.uint64)
        for i, row in enumerate(rows):
            for j in range(i+1):
                value = (row & rows[j]).bit_count()
                gram[i, j] = gram[j, i] = value
        profiles = [[int(gram[i, i])] for i in range(m)]
        power = gram.copy()
        for _ in range(5):
            power = power @ gram
            for i, value in enumerate(np.diag(power)):
                profiles[i].append(int(value))
        return tuple(sorted(Counter(
            tuple(profile) for profile in profiles).items()))

    # Rebuild the same reservation prefix in every pure call. Reserve
    # both finalists, so even a rejected finalist cannot reappear later.
    reserved_pairs, reserved_graphs = set(), set()
    cursors = [0]*len(cards)
    chosen = None
    for slot in range(idx+1):
        selected = []
        preferred = slot % len(cards)
        search_order = [preferred] + [
            c for c in range(len(cards)) if c != preferred]
        for card in search_order:
            catalogue = catalogues[card]
            while cursors[card] < len(catalogue) and len(selected) < 2:
                coarse, identity, A, B = catalogue[cursors[card]]
                cursors[card] += 1
                if identity in reserved_pairs:
                    continue
                reserved_pairs.add(identity)
                hx, hz, n = lifted_matrices(G, A, B)
                if n != 544 or n-gf2_rank(hx)-gf2_rank(hz) != 288:
                    continue
                ar, br = _single_row_components(hx, hz, q)
                assert ar == component(A, "A")
                assert br == component(B, "B")
                if assess(br) is None:
                    continue
                invariant = full_invariant(hx, hz)
                if invariant in reserved_graphs:
                    continue
                reserved_graphs.add(invariant)
                selected.append((coarse, card, A, B, hx, hz, invariant))
            if len(selected) == 2:
                break
        if not selected:
            raise ValueError("fresh full-code invariant catalogue exhausted")
        chosen = selected

    def temporal_probe(checks, stabilizers, seed, shots):
        """Six-round phenomenological BP, not the ASC gate circuit."""
        n, rounds = 544, 6
        neighbors = [sites(row) for row in checks]
        m = len(checks)
        row_width = max(map(len, neighbors))+2
        variables = rounds*(n+m)
        dummy = variables
        incidence = np.full(((rounds+1)*m, row_width),
                            dummy, dtype=np.int32)
        for t in range(rounds+1):
            for r, ns in enumerate(neighbors):
                position = 0
                if t < rounds:
                    incidence[t*m+r, :len(ns)] = [
                        t*n+c for c in ns]
                    position = len(ns)
                    incidence[t*m+r, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = rounds*n+(t-1)*m+r
        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degrees = np.bincount(flat[valid], minlength=variables)
        variable_edges = np.full(
            (variables+1, int(degrees.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        sensed = np.asarray([x.bit_count()
                             for x in columns(checks, n)])
        spread = np.asarray([x.bit_count()
                             for x in columns(stabilizers, n)])
        data_rates = 0.002*(2+sensed+spread)
        measurement_rates = np.asarray([
            0.002*(2+len(ns))*2/3 for ns in neighbors])
        base = np.concatenate((np.tile(data_rates, rounds),
                               np.tile(measurement_rates, rounds)))
        levels = np.asarray((0.8, 1.0, 1.3))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None]*base[None, :], 0.20)
        channel = np.random.default_rng(seed)
        errors = np.zeros((shots, variables+1), dtype=bool)
        errors[:, :variables] = channel.random(
            (shots, variables)) < rates
        target = np.logical_xor.reduce(
            errors[:, flat].reshape(
                shots, (rounds+1)*m, row_width), axis=2)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log((1-rates)/rates)
        messages = prior[:, flat].copy()
        true_final = np.logical_xor.reduce(
            errors[:, :rounds*n].reshape(shots, rounds, n), axis=1)

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        failures = iterations = residual_weight = 0
        positions = np.arange(row_width)[None, None, :]
        for iteration in range(1, 25):
            count = len(prior)
            shaped = messages.reshape(
                count, (rounds+1)*m, row_width)
            magnitudes = np.abs(shaped)
            signs = np.where(shaped < 0, -1.0, 1.0)
            minimum_at = np.argmin(magnitudes, axis=2)
            smallest = np.partition(magnitudes, 1, axis=2)[:, :, :2]
            magnitude = np.where(
                positions == minimum_at[:, :, None],
                smallest[:, :, 1, None], smallest[:, :, 0, None])
            outgoing = (
                0.80*magnitude*signs
                * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = np.logical_xor.reduce(
                hard[:, flat].reshape(
                    count, (rounds+1)*m, row_width), axis=2)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                residual = decoded_final[shot] ^ true_final[shot]
                word = sum(1 << int(c) for c in np.flatnonzero(residual))
                if nontrivial(word):
                    failures += 1
                    residual_weight += int(np.count_nonzero(residual))
                iterations += iteration
            keep = ~settled
            if not np.any(keep):
                return failures, 0, residual_weight, iterations
            messages = np.clip(
                0.30*messages + 0.70*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior, target = prior[keep], target[keep]
            target_sign, true_final = target_sign[keep], true_final[keep]
        unresolved = len(prior)
        return (failures+unresolved, unresolved, residual_weight,
                iterations+25*unresolved)

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                x[2]+z[2], x[3]+z[3])

    finalists = []
    seed = 325031911+1009*idx
    for coarse, card, A, B, hx, hz, invariant in chosen:
        assert css_ok(hx, hz)
        x = temporal_probe(hz, hx, seed, 16)
        z = temporal_probe(hx, hz, seed+1, 16)
        # Independent holdout streams, common across the finalists.
        xx = temporal_probe(hz, hx, seed+1000003, 24)
        zz = temporal_probe(hx, hz, seed+1000004, 24)
        total_x = tuple(a+b for a, b in zip(x, xx))
        total_z = tuple(a+b for a, b in zip(z, zz))
        key = quality(total_x, total_z)+coarse+(invariant,)
        finalists.append((key, card, A, B, hx, hz, total_x, total_z))
    finalists.sort(key=lambda item: item[0])
    key, card, A, B, hx, hz, x, z = finalists[0]
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert all(len({G._key(g) for g in cell}) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    assert max(row.bit_count() for row in hx+hz) == 12
    assert 6*sum(row.bit_count() for row in hx+hz) == 17280
    assert 544-gf2_rank(hx)-gf2_rank(hz) == 288
    assert css_ok(hx, hz)
    return {"idx": idx, "case": cards[card], "k": 288,
            "screen": key, "temporal_surrogate": (x, z),
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
# REFUTED: Colored short-word and graph-walk component briefs provide a distinct, reliable all-binomial portfolio: the current parent produced only 13 distinct codes from 60 slots, scored 3.3714, and measured samples retained 102-130 failures per 250 shots with widespread weight-four logicals.
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
