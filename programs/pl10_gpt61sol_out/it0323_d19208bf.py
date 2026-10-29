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
CONCEPT = ("For cyclic C25, order 25, shapes A=(1,3), B=(2,7), long-order Sidon supports and reduced weight-six component multiplicity should raise worse-basis circuit fitness beyond the 0.450 frontier's 4.1155.")
GROUP = ("L3 jump from SL(2,3) to cyclic C25. BEDS reports about 6.9 for C25 on (2,4)x(2,4), but this high-rate bed is untried. Order-25 relative directions replace the parent's order-three/four/six cycles. Commutativity can introduce weight-six component identities; the experiment controls their multiplicity rather than claiming a larger distance. Neither the related-bed reading nor the group law establishes circuit reliability.")
SHAPES = ("Keep (1,3)x(2,7) for cell 0.450. The size factor 23 permits q<=26; q=25 gives n=575, kb=250 and guaranteed rate 10/23=0.43478. Components have 75 variables/25 checks and 175 variables/50 checks. Require verified k=250. Parent weight-six witnesses and failures spread across sector-one Acol/Bcol blocks, with data enrichment 1.1-1.2x and no persistent Brow hotspot, motivating a global relative-support change.")
FILLING = ("Twenty contracts cross five seventeen-term B architectures with four indexed affine-canonical A geometries; additional batches consume further geometries rather than labeling orbits. A trinomials are Sidon with order-25 differences and at least eight distinct undirected directions across the row. B row totals remain (9,8), internal differences have order 25, and matching holonomies are Sidon with order-25 pairwise differences. Verified lifts screen component ranks and kernels, then exact weight-five/six multiplicities; complete matrices screen exact k, CSS and short logicals. Selection replaces the unsuccessful temporal-BP probes with balanced multiplicity and opposite-check-weighted pair-boundary exposure, an uncalibrated circuit-risk surrogate. Both CSS weight sums are <=12. Six-round CNOTs are 22350 versus 21456 solely because q increases; gates per qubit are unchanged. Target the parent's 108..128 Z failures in 750..1250 shots and collective flips across all 240 logicals; only ASC measurements confirm improvement.")
FAMILY_NAME = "r045_C25_q25_13x27_longorder_sidon_multiplicity_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 25

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C25 requires order 25")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _component_spectrum56(cols, short):
    """Exact weight-five/six counts after exclusion through weight four.

    Unique nonzero columns and zero weight-three/four counts ensure
    matched supports are disjoint. Each weight-five word has ten
    pair/triple partitions; each weight-six word has ten unordered
    triple/triple partitions. Boundary exposure is only a BP surrogate.
    """
    if short[0] or short[1]:
        return (10**12, 10**12, 10**12)
    pairs = {}
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            pairs[value] = pairs.get(value, 0) + 1
    triples = {}
    five = six = boundary = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                five += pairs.get(value, 0)
                old = triples.get(value, 0)
                six += old
                triples[value] = old + 1
                weight = value.bit_count()
                if weight <= 3:
                    boundary += (4 - weight) ** 2
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10, boundary


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Indexed long-order geometries and independently optimized B voltages."""
    from itertools import combinations

    q = G.q
    assert q == 25
    architecture = (idx // 4) % 5
    mode = idx % 4
    extension = idx // 20
    sa = [[3, 3, 3]]
    column_profiles = (
        ((2, 1), (2, 1), (1, 2), (1, 1),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (1, 2), (1, 2), (1, 1),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (0, 3), (2, 1), (1, 1),
         (1, 1), (1, 1), (1, 1)),
        ((2, 1), (2, 1), (1, 2), (2, 0),
         (0, 2), (1, 1), (1, 1)),
        ((3, 0), (1, 2), (1, 2), (2, 0),
         (0, 2), (1, 1), (1, 1)),
    )
    profile = column_profiles[architecture]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert tuple(map(sum, sb)) == (9, 8)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(map(sum, sa)) + max(map(sum, profile)) == 12
    assert max(map(sum, sb)) + max(map(sum, zip(*sa))) == 12

    orders = _orders(G)
    long_elements = [g for g in G.elements if orders[G.idx(g)] == 25]
    multipliers = [g[0] for g in long_elements]
    assert len(long_elements) == 20

    def direction(value):
        value %= q
        return min(value, q - value)

    def directions(values):
        return {direction(a - b) for a, b in combinations(values, 2)}

    def normalized(values, multiplier=1):
        values = [multiplier*a % q for a in values]
        return min(tuple(sorted((a-origin) % q for a in values))
                   for origin in values)

    triangle_catalog = sorted({
        normalized((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if a % 5 and b % 5 and (a-b) % 5
        and len(directions((0, a, b))) == 3
    })
    assert triangle_catalog

    def geometry_key(cells):
        # Quotient genuine cyclic automorphisms, A-column permutations
        # and independent one-row component column translations.
        # This is not a complete full-code Tanner-isomorphism test.
        return min(tuple(sorted(normalized(cell, u) for cell in cells))
                   for u in multipliers)

    def indexed_A():
        # Each support architecture consumes a deterministic book of
        # different relative structures, not random labeling orbits.
        grng = random.Random(250451 + 1009*architecture)
        wanted = mode + 4*extension
        seen = set()
        accepted = 0
        for attempt in range(40000):
            relative = grng.sample(triangle_catalog, 3)
            if len(set().union(*(directions(c) for c in relative))) < 8:
                continue
            signature = geometry_key(relative)
            if signature in seen:
                continue
            seen.add(signature)
            A = [[[(a,) for a in cell] for cell in signature]]
            # The zero-partner probe uses exactly the verified left lift.
            ar = lifted_matrices(G, A, [[[]]])[0]
            if gf2_rank(ar) != q:
                continue
            stats = kernel_stats(columns(ar, 3*q))
            if stats is None or stats[0] or stats[1]:
                continue
            if accepted == wanted:
                return A, signature
            accepted += 1
        raise ValueError("indexed long-order A geometry book exhausted")

    def draw_grid(side, support):
        if side == "A":
            return [[list(cell) for cell in a_geometry[0]]]

        pair_sites = [j for j, column in enumerate(profile)
                      if column == (1, 1)]
        holonomies = [G.e]
        used_matching_directions = set()
        while len(holonomies) < len(pair_sites):
            choices = []
            for g in long_elements:
                new = [
                    direction(g[0] - h[0]) for h in holonomies
                ]
                if any(orders[G.idx(G.mul(G.inv(h), g))] != 25
                       for h in holonomies):
                    continue
                if len(set(new)) != len(new):
                    continue
                if set(new) & used_matching_directions:
                    continue
                choices.append(g)
            if not choices:
                raise ValueError("long-order Sidon matching brief exhausted")
            chosen = rng.choice(choices)
            used_matching_directions.update(
                direction(chosen[0] - h[0]) for h in holonomies)
            holonomies.append(chosen)
        paired = dict(zip(pair_sites, holonomies))

        grid = []
        for i, row_counts in enumerate(support):
            row = []
            used = set()
            for j, count in enumerate(row_counts):
                if j in paired:
                    cell = [G.e] if i == 0 else [G.inv(paired[j])]
                elif count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                else:
                    if count == 2:
                        choices = [
                            (0, g[0]) for g in long_elements
                            if direction(g[0]) not in used
                        ]
                    else:
                        choices = [
                            relative for relative in triangle_catalog
                            if not (directions(relative) & used)
                        ]
                    if not choices:
                        raise ValueError("long-order row-direction brief exhausted")
                    relative = rng.choice(choices)
                    used.update(directions(relative))
                    base = G.pick(rng)
                    cell = [G.mul(base, (a,)) for a in relative]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                assert all(
                    orders[G.idx(G.mul(G.inv(a), b))] == 25
                    for a, b in combinations(cell, 2))
                row.append(cell)
            grid.append(row)
        return grid

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def components(hx, hz):
        # Extract exclusively from the helper-built complete matrices.
        # Bcol0 X checks contain the left-lifted A component.
        mask = (1 << q) - 1
        ar = []
        for row in hx[:q]:
            compact = 0
            for j in range(A_SHAPE[1]):
                base = j * B_SHAPE[1] * q
                compact |= ((row >> base) & mask) << (j * q)
            ar.append(compact)
        # Acol0 Z checks contain BOTH right-lifted B check rows.
        width = B_SHAPE[1] * q
        br = [row & ((1 << width) - 1)
              for row in hz[:B_SHAPE[0] * q]]
        return ar, br

    def kernel_stats(cols):
        """Exact weight-three/four kernel counts for unique columns."""
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        singles = set(cols)
        pairs = {}
        triples = collisions = cycles = 0
        weights = [c.bit_count() for c in cols]
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                triples += int(s in singles)
                old = pairs.get(s, 0)
                collisions += old
                pairs[s] = old + 1
                overlap = (weights[i] + weights[j] - s.bit_count()) // 2
                cycles += overlap * (overlap - 1) // 2
        assert triples % 3 == 0 and collisions % 3 == 0
        return triples // 3, collisions // 3, cycles

    def logical_stats(checks, stabilizers, n):
        """Exact logical counts and shared-check hook opportunities.

        Unique nonzero columns exclude weights one and two. Quotient
        residues classify collisions modulo the stabilizer row span.
        Hook statistics count possible fault-pair partitions, not
        ASC-certified faults or an estimated circuit failure rate.
        """
        cols = columns(checks, n)
        if 0 in cols or len(set(cols)) != n:
            return None
        spread = columns(stabilizers, n)
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items(), reverse=True)
        residuals = []
        for i in range(n):
            v = 1 << i
            for p, row in ordered:
                if (v >> p) & 1:
                    v ^= row
            residuals.append(v)

        singles = {s: residuals[i] for i, s in enumerate(cols)}
        totals = {}
        classes = {}
        triples = fours = double_hook = single_hook = 0
        for i, x in enumerate(cols):
            ri = residuals[i]
            hi = spread[i]
            for j in range(i):
                s = x ^ cols[j]
                residue = ri ^ residuals[j]
                single = singles.get(s)
                if single is not None and residue != single:
                    triples += 1
                hook = (hi & spread[j]).bit_count()
                key = (s, residue)
                nt, ht = totals.get(s, (0, 0))
                nc, hc = classes.get(key, (0, 0))
                different = nt - nc
                different_hooks = ht - hc
                fours += different
                double_hook += hook * different_hooks
                single_hook += hook * different + different_hooks
                totals[s] = (nt + 1, ht + hook)
                classes[key] = (nc + 1, hc + hook)
        assert triples % 3 == 0 and fours % 3 == 0
        return triples // 3, fours // 3, double_hook, single_hook

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round BP surrogate, not ASC or relay/LSD simulation.

        The final syndrome is perfect. Opposite-check pairs approximate
        correlated extraction faults without asserting an ASC ordering.
        Unresolved BP shots are conservatively counted as probe failures.
        """
        import numpy as np

        def positions(word):
            out = []
            while word:
                bit = word & -word
                out.append(bit.bit_length() - 1)
                word ^= bit
            return out

        neighbors = [positions(row) for row in checks]
        spread = [np.asarray(positions(row), dtype=np.int32)
                  for row in stabilizers]
        m, rounds = len(checks), 6
        variables = rounds * (n + m)
        dummy = variables
        detectors = (rounds + 1) * m
        row_width = max(map(len, neighbors)) + 2
        incidence = np.full((detectors, row_width), dummy,
                            dtype=np.int32)
        for t in range(rounds + 1):
            for r, ns in enumerate(neighbors):
                p = 0
                if t < rounds:
                    incidence[t*m+r, :len(ns)] = t*n + np.asarray(ns)
                    p = len(ns)
                    incidence[t*m+r, p] = rounds*n + t*m + r
                    p += 1
                if t:
                    incidence[t*m+r, p] = rounds*n + (t-1)*m + r

        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (variables + 1, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        dc = np.asarray([v.bit_count() for v in columns(checks, n)])
        oc = np.asarray([v.bit_count() for v in columns(stabilizers, n)])
        data_rate = 0.002 * (2 + dc + oc)
        measurement_rate = np.asarray(
            [0.002 * (2 + len(ns)) * 2/3 for ns in neighbors])
        levels = np.asarray((0.75, 1.0, 1.25))[np.arange(shots) % 3]
        independent = np.concatenate((
            np.tile(data_rate, rounds),
            np.tile(measurement_rate, rounds)))
        # The prior includes an approximate marginal contribution from
        # the explicitly injected opposite-check correlated pairs.
        marginal = np.concatenate((
            np.tile(data_rate + 0.003*oc, rounds),
            np.tile(measurement_rate, rounds)))
        physical = np.minimum(levels[:, None]*independent[None, :], 0.25)
        rates = np.minimum(levels[:, None]*marginal[None, :], 0.25)
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables + 1), dtype=bool)
        error[:, :variables] = channel.random(
            (shots, variables)) < physical

        # Fixed-size random streams remain common across finalists.
        shape = (shots, rounds, len(spread))
        active_uniform = channel.random(shape)
        first_uniform = channel.random(shape)
        second_uniform = channel.random(shape)
        for t in range(rounds):
            for r, ns in enumerate(spread):
                if len(ns) < 2:
                    continue
                probability = levels * 0.002*(2 + len(ns))*2/3
                active = np.flatnonzero(
                    active_uniform[:, t, r] < probability)
                a = (first_uniform[active, t, r]*len(ns)).astype(np.int32)
                b = (second_uniform[active, t, r]
                     *(len(ns)-1)).astype(np.int32)
                b += b >= a
                error[active, t*n + ns[a]] ^= True
                error[active, t*n + ns[b]] ^= True

        target = (
            error[:, flat].reshape(shots, detectors, row_width)
            .sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)
        prior = np.empty((shots, variables + 1), dtype=np.float64)
        prior[:, :variables] = np.log((1-rates)/rates)
        prior[:, dummy] = 30.0
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        failures = iteration_sum = 0
        for iteration in range(1, 37):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            products = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(products)-np.log1p(-products))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges + 1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, detectors, row_width)
                .sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(decoded[local] ^ true_final[local]):
                    word |= 1 << int(c)
                failures += int(nontrivial(word))
                iteration_sum += iteration
            keep = ~settled
            if not np.any(keep):
                return failures, 0, iteration_sum
            damping = 0.65 if iteration <= 18 else 0.85
            messages = np.clip(
                (1-damping)*messages
                + damping*(posterior[:, flat]-outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (failures + unresolved, unresolved,
                iteration_sum + 36*unresolved)

    def probe_quality(x, z):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1], x[2] + z[2])

    a_geometry, a_signature = indexed_A()
    shortlist = []
    exact_seen = set()
    for trial in range(320):
        try:
            A = draw_grid("A", sa)
            B = draw_grid("B", sb)
        except ValueError:
            continue
        identity = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row
        )
        if identity in exact_seen:
            continue
        exact_seen.add(identity)

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = components(hx, hz)
        if gf2_rank(ar) != A_SHAPE[0] * q:
            continue
        if gf2_rank(br) != B_SHAPE[0] * q:
            continue
        ast = kernel_stats(columns(ar, A_SHAPE[1] * q))
        bst = kernel_stats(columns(br, B_SHAPE[1] * q))
        if ast is None or bst is None:
            continue

        # Component words can replicate across opposite-side columns.
        # This coarse weighting is only a shortlisting heuristic.
        a3, b3 = B_SHAPE[1] * ast[0], A_SHAPE[1] * bst[0]
        a4, b4 = B_SHAPE[1] * ast[1], A_SHAPE[1] * bst[1]
        ca, cb = ast[2] / len(ar), bst[2] / len(br)
        key = (max(a3, b3), a3 + b3,
               max(a4, b4), a4 + b4,
               max(ca, cb), ca + cb)
        shortlist.append((key, identity, A, B, hx, hz, n))
        shortlist.sort(key=lambda item: item[:2])
        del shortlist[12:]

    options = []
    for component_key, identity, A, B, hx, hz, n in shortlist:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 10 * q or not (425 * n <= 1000 * k < 475 * n):
            continue
        ar, br = components(hx, hz)
        acols = columns(ar, A_SHAPE[1] * q)
        bcols = columns(br, B_SHAPE[1] * q)
        ast, bst = kernel_stats(acols), kernel_stats(bcols)
        if ast is None or bst is None:
            continue
        asp = _component_spectrum56(acols, ast)
        bsp = _component_spectrum56(bcols, bst)
        lx = logical_stats(hx, hz, n)
        lz = logical_stats(hz, hx, n)
        if lx is None or lz is None:
            continue
        fx = _same_basis_cycles(hx)
        fz = _same_basis_cycles(hz)
        # Exact component counts are NOT an exact quantum logical
        # census. Replication weights and triple boundaries remain
        # uncalibrated circuit-risk surrogates.
        a5, b5 = B_SHAPE[1] * asp[0], A_SHAPE[1] * bsp[0]
        a6, b6 = B_SHAPE[1] * asp[1], A_SHAPE[1] * bsp[1]
        ba, bb = asp[2] / len(ar), bsp[2] / len(br)
        # Commutativity makes weight-six component identities possible.
        # Do not reward an unattainable distance threshold: compare
        # their replication-weighted multiplicities before boundaries.
        # These are component counts, not a full logical census.
        key = (max(lx[0], lz[0]), lx[0] + lz[0],
               max(lx[1], lz[1]), lx[1] + lz[1],
               max(ast[0], bst[0]), ast[0] + bst[0],
               max(ast[1], bst[1]), ast[1] + bst[1],
               max(a5, b5), a5 + b5,
               max(a6, b6), a6 + b6,
               max(ba, bb), ba + bb,
               max(lx[2], lz[2]), lx[2] + lz[2],
               max(lx[3], lz[3]), lx[3] + lz[3],
               max(fx / len(hx), fz / len(hz)),
               fx + fz, component_key, identity)
        options.append((key, len(options), A, B, hx, hz, k, lx, lz))

    if not options:
        raise ValueError("no full-rank two-check filling passed screening")
    options.sort(key=lambda item: item[:2])

    def pair_exposure(checks, other, n):
        """Gate-degree-weighted low pair boundaries on verified matrices.

        Shared opposite checks mark possible correlated extraction
        faults, not hooks certified for an ASC gate ordering.
        This statistic is not a measured or calibrated pL estimate.
        """
        cs = columns(checks, n)
        mate = columns(other, n)
        degrees = [a.bit_count() + b.bit_count()
                   for a, b in zip(cs, mate)]
        burden = 0
        for i, syndrome in enumerate(cs):
            for j in range(i):
                boundary = (syndrome ^ cs[j]).bit_count()
                if boundary <= 4:
                    shared = (mate[i] & mate[j]).bit_count()
                    burden += (degrees[i] * degrees[j]
                               * (5-boundary)**2 * (1+shared))
        return burden / len(checks)

    # The measured parent refuted superiority of the lighter temporal
    # probe model. Its historical helper remains inactive above.
    # Short protection and component multiplicity precede exposure.
    finalists = []
    for option in options[:6]:
        key, number, A, B, hx, hz, k, lx, lz = option
        ex = pair_exposure(hx, hz, 23*q)
        ez = pair_exposure(hz, hx, 23*q)
        score = (key[:12] + (max(ex, ez), ex+ez)
                 + key[12:] + (number,))
        finalists.append((score, number, option, ex, ez))
    finalists.sort(key=lambda item: item[:2])
    best_key, _, best, exposure_x, exposure_z = finalists[0]
    _, _, A, B, hx, hz, k, lx, lz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    edges = sum(row.bit_count() for row in hx + hz)
    expected = q * (
        (B_SHAPE[1] + B_SHAPE[0]) * sum(map(sum, sa))
        + (A_SHAPE[1] + A_SHAPE[0]) * sum(map(sum, sb)))
    assert edges == expected
    assert 23*q == 575 and 100 <= 23*q < 600
    assert 6 * edges == 22350
    assert k == 250 and lifted_k(G, A, B) == 250
    assert 0.425 <= k/(23*q) < 0.475
    assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "architecture": architecture, "mode": mode,
            "extension": extension, "k": k, "screen": best_key,
            "relative_A_geometry": a_signature,
            "pair_exposure_surrogate": (exposure_x, exposure_z),
            "logical_profile": (lx, lz), "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing four-cycles alone did not protect this A4 x C3 bed: fitness 3.2178, Z failures 112..137/250, and widespread weight-three sector-1 witnesses despite mostly stage-1 settlement.
# REFUTED: Low gate count and connected binomial A suffice for reliable balanced memory: the triangle-transfer parent scored 3.4383, retained 48 weight-three Z witnesses, and failed 108/250 to 181/500 X shots despite first-stage settlement.
# REFUTED: Unit trinomials, full component ranks and exact weight-three/four hook screening suffice for rare collective failures on A4 x C2 (1,3)x(2,7): fitness was 3.9832; measured slots reached failed-shot caps, retained abundant sector-one weight-five/six witnesses, and failures touched all 240 logicals.
# REFUTED: Quaternion-quotient unit trinomials and component weight-six screening suffice for rare collective SL(2,3) memory failures: fitness was 4.0793; every measured code reached failed-shot caps, retained abundant sector-one weight-six witnesses, and failures collectively touched all 240 logicals.
# REFUTED: Reducing B from eighteen to seventeen terms and selecting with temporal-BP holdouts improves this SL(2,3) high-rate bed: fitness fell from 4.0793 to 3.9373; all measured codes reached failed-shot caps and retained abundant weight-six witnesses and collective flips across all 240 logicals.
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
