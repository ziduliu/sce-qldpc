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
CONCEPT = ("For C7 semidirect C6 with action 2, order 42, shapes (1,2)x(2,6), a unit A trinomial plus an involution-free quadrinomial should reduce collective Z failures when B's two lighter matching columns have order-14 holonomy and verified components exclude kernels through six. Worse-basis ASC fitness must test the principle.")
GROUP = ("Frozen L1: retain the exact validated nonfaithful action-2 group and encoding, isomorphic to (C7 semidirect C3) x C2. This bed has reached about 6.2; the current program scores 6.0832. Order-14 elements permit a length-28 cycle in B's degree-two subgraph. This is an algebraic mechanism for a filling experiment, not evidence of a group advantage.")
SHAPES = ("Frozen A=(1,2), B=(2,6) give n=588, kb=168 and guaranteed rate 2/7 in cell 0.300; the size factor fourteen permits q<=42. Components have 84 variables/42 checks and 252 variables/84 checks. Require complete verified k=168. Parent failures collectively touched almost every logical and diffusely enriched Acol/Bcol blocks by about 1.1-1.2x; Acol/Brow check enrichment was near one, so no persistent localized repair or transferable worst-logical index is assumed.")
FILLING = ("Twenty contracts cross four B weighted-support architectures with five indexed A relative geometries and B order policies. A counts are (3,4); B has four degree-three columns, two degree-two matching columns and row totals eight. Matching holonomy has exact order fourteen, giving length-28 cycles in that subgraph. HX/HZ weights are 10/12 and six-round extraction uses 26208 CNOTs, 504 above the parent's measured 25704: this reallocates protection at nearly unchanged gate count, not a gate reduction. A trinomials and isolated B trinomials are helper-verified units. An automorphism/gauge-canonical catalogue removes obvious relative-support copies; the evaluator remains authoritative for Tanner isomorphism. Component searches refine exact kernel exclusion through six, boundaries and capped sampled witnesses. Full verified matrices require k=168, connectivity and CSS, prefer no logical through four, then undergo common-stream six-round noisy-syndrome BP probes and independent holdouts. Probes omit actual ASC ordering and relay/LSD and are not measured pL. Target slot15's 100/34750 Z failures, widespread flips and fallback, rather than its favorable k=176,d2s=10 proxy.")
FAMILY_NAME = "r030_C7byC6_s2_12x26_unit34_longmatching_temporal_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this nonfaithful C7 semidirect C6 requires q=42")
    # x^7=y^6=1, y*x*y^-1=x^2.
    # The action has order three; y^3 generates the central C2.
    act = (1, 2, 4, 1, 2, 4)
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + act[b] * c) % 7, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-act[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _fg_columns(rows, width):
    """Column syndromes of a verified component action."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _fg_cost(rows, width):
    """Connectivity, exact rank deficit, and exact Tanner 4/6 cycles."""
    from itertools import combinations

    cols = _fg_columns(rows, width)
    if not all(cols) or not all(rows):
        return None
    neighbors = [[] for _ in range(width)]
    adj = [dict() for _ in rows]
    for c, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            neighbors[c].append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, b in combinations(neighbors[c], 2):
            adj[a][b] = adj[a].get(b, 0) + 1
            adj[b][a] = adj[b].get(a, 0) + 1

    seen, todo = {0}, [0]
    while todo:
        a = todo.pop()
        for b in adj[a]:
            if b not in seen:
                seen.add(b)
                todo.append(b)
    if len(seen) != len(rows):
        return None

    four = sum(v * (v - 1) // 2
               for a in range(len(rows))
               for b, v in adj[a].items() if a < b)
    six = 0
    for a in range(len(rows)):
        for b, ab in adj[a].items():
            if b <= a:
                continue
            for c, bc in adj[b].items():
                if c > b and c in adj[a]:
                    six += ab * bc * adj[a][c]
    for ns in neighbors:
        for a, b, c in combinations(ns, 3):
            six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
    return len(rows) - gf2_rank(rows), four, six


def _fg_quality(rng, rows, width, q):
    """Exact kernel<=4 exclusion and small-boundary/kernel heuristics.

    Triple boundaries are enumerated, but triple syndromes are not stored:
    this is not an exact exclusion test through weight six.
    Randomized kernel circuits give upper bounds, never pL predictions.
    """
    cols = _fg_columns(rows, width)
    seen = {0}
    for value in cols:
        if value in seen:
            return True, 4, 1.0e9, 1.0e9
        seen.add(value)
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            if syndrome in seen:
                return True, 4, 1.0e9, 1.0e9
            seen.add(syndrome)

    exposure = 0
    loads = [0] * (width // q)
    for i in range(width):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                weight = (pair ^ cols[h]).bit_count()
                if weight < 5:
                    penalty = (5 - weight) ** 2
                    exposure += penalty
                    loads[i // q] += penalty
                    loads[j // q] += penalty
                    loads[h // q] += penalty
    triples = width * (width - 1) * (width - 2) // 6
    boundary = exposure / max(1, triples)
    peak = max(loads) * len(loads) / max(1, 3 * triples)

    order = list(range(width))
    best = width + 1
    for _ in range(28):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = syndrome, word
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome:
                best = min(best, word.bit_count())
                for previous in basis:
                    best = min(best, (word ^ previous).bit_count())
                basis.append(word)
        if best <= 6:
            break
    return False, best, boundary, peak


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def NEW_CTX(rng, G, idx):
    """Unit/quarter support contracts, long matching cycles and joint probes."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 42 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    design, mode = idx % 20, idx // 20
    architecture, policy = divmod(design, 5)
    strong = (
        ((2,1), (2,1), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (1,2)),
        ((3,0), (3,0), (0,3), (0,3)),
        ((3,0), (1,2), (1,2), (1,2)),
    )[architecture]
    profile = strong + ((1,1), (1,1))
    sa = [[3, 4]]
    sb = [[column[i] for column in profile] for i in range(2)]
    wx = sum(sa[0]) + max(map(sum, profile))
    wz = max(map(sum, sb)) + max(sa[0])
    assert (wx, wz) == (10, 12)
    assert tuple(map(sum, sb)) == (8, 8)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    elements = list(G.elements)
    allowed = (3, 6, 7, 14)

    def element_order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid element order")

    orders = {g: element_order(g) for g in elements}
    by_order = {t: [g for g in elements if orders[g] == t]
                for t in allowed}

    def relative_order(g, h):
        return orders[G.mul(G.inv(g), h)]

    # Enumerate the feasible unit palette, rather than imposing a
    # triangle spectrum that might contain no invertible realization.
    unit_books = {}
    usable = [g for g in elements if orders[g] in allowed]
    for u, v in combinations(usable, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        spectrum = tuple(sorted((orders[u], orders[v],
                                relative_order(u, v))))
        if any(t not in allowed for t in spectrum):
            continue
        template = [G.e, u, v]
        rows = lifted_matrices(G, [[template]], [[[]]])[0]
        if gf2_rank(rows) == q:
            unit_books.setdefault(spectrum, []).append(template)
    unit_types = sorted(unit_books)
    assert unit_types

    # Actual automorphisms x->x^u, y->x^c*y. The partial sums of
    # 1,2,4 repeat modulo seven because the action has order three.
    partial = (0, 1, 3, 0, 1, 3)
    automorphisms = [
        tuple(G.idx(((u*a+c*partial[b]) % 7, b))
              for a, b in elements)
        for u in range(1, 7) for c in range(7)
    ]
    assert all(len(set(image)) == q for image in automorphisms)

    def image_cell(cell, image):
        return [elements[image[G.idx(g)]] for g in cell]

    def relative_key(cell):
        """Remove cell-origin/automorphism copies of relative supports.

        This is deliberately not claimed to classify arbitrary full
        quantum Tanner-graph isomorphisms.
        """
        images = []
        for image in automorphisms:
            values = image_cell(cell, image)
            for origin in values:
                inverse = G.inv(origin)
                images.append(tuple(sorted(
                    G.idx(G.mul(g, inverse)) for g in values)))
        return min(images)

    grng = random.Random(42034019 + 1009*architecture)
    wanted = policy + 5*mode
    book, seen_geometry = [], set()
    for proposal in range(24000):
        card = len(book) % 5
        kind = unit_types[(card + architecture) % len(unit_types)]
        tri = list(grng.choice(unit_books[kind]))
        quartet = [G.e] + grng.sample(usable, 3)
        spectrum = tuple(sorted(relative_order(g, h)
                                for g, h in combinations(quartet, 2)))
        if any(t not in allowed for t in spectrum):
            continue
        n3 = spectrum.count(3)
        requested = (0, 0, 1, 2, 2)[card]
        if proposal < 16000:
            if n3 != requested:
                continue
        elif n3 > 2:
            continue
        if not any(G.mul(g, h) != G.mul(h, g)
                   for g, h in combinations(quartet[1:], 2)):
            continue
        signature = (relative_key(tri), relative_key(quartet))
        if signature in seen_geometry:
            continue
        seen_geometry.add(signature)
        book.append((tri, quartet, signature, spectrum))
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("fresh unit/quadrinomial relative catalogue exhausted")
    tri, quartet, a_contract, quartet_spectrum = book[wanted]
    a_geometry = (tri, quartet)
    b_palettes = ((14,), (7,14), (14,6),
                  (14,14,7), (14,14,6))
    palette = b_palettes[policy]

    def draw_cell(side, i, j):
        if side == 0:
            # Independent relative orientations change the joint
            # component; no selected complete code is relabeled.
            image = rng.choice(automorphisms)
            return image_cell(a_geometry[j], image)

        count = sb[i][j]
        if count == 0:
            return []
        # Two degree-two matching columns: the first is a gauge-fixed
        # identity matching; the second has holonomy of order fourteen.
        # Their isolated degree-two subgraph therefore has girth 28.
        if j == 4 or (j == 5 and i == 0):
            return [G.e]
        if j == 5:
            return [rng.choice(by_order[14])]

        base = G.pick(rng)
        if count == 1:
            return [base]
        if count == 2:
            t = palette[(j + 2*i + mode) % len(palette)]
            return [base, G.mul(base, rng.choice(by_order[t]))]
        kind = unit_types[(policy+i+j+mode) % len(unit_types)]
        template = rng.choice(unit_books[kind])
        image = rng.choice(automorphisms)
        return [G.mul(base, g) for g in image_cell(template, image)]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw_grid(side):
        counts = sa if side == 0 else sb
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts)]

    # Cache singleton blocks extracted only from lifted_matrices.
    # No independently implemented regular-representation convention.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][g] = tuple(row & mask for row in hx)
        actions[1][g] = tuple(row & mask for row in hz)

    def component_rows(grid, side):
        result = []
        for grid_row in grid:
            rows = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        rows[r] ^= bits << (j * q)
            result.extend(rows)
        return result

    widths = (2 * q, 6 * q)
    shapes = (A_SHAPE, B_SHAPE)

    def energy(cost):
        deficit, four, six = cost
        return (12 * q * deficit + 80 * four + six) / q

    def refine(rows, width, side):
        """Exact component exclusion through six, then witness heuristics.

        Equal syndromes of distinct subsets of size <=3 exhaust kernel
        weights <=6. Passing this window is not a distance certificate.
        """
        cols = _fg_columns(rows, width)
        seen = {0}
        for value in cols:
            if value in seen:
                return True, 2, 1.e9, 1.e9, 1.e9, 1.e9
            seen.add(value)
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                if syndrome in seen:
                    return True, 4, 1.e9, 1.e9, 1.e9, 1.e9
                seen.add(syndrome)

        exposure = 0
        loads = [0] * (width // q)
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return True, 6, 1.e9, 1.e9, 1.e9, 1.e9
                    seen.add(syndrome)
                    weight = syndrome.bit_count()
                    if weight < 5:
                        penalty = (5-weight)**2
                        exposure += penalty
                        for c in (i, j, h):
                            loads[c // q] += penalty
        del seen
        triples = width*(width-1)*(width-2)//6
        boundary = exposure / max(1, triples)
        peak = max(loads)*len(loads) / max(1, 3*triples)

        # Common permutation streams compare candidates within a slot.
        probe = random.Random(420734 + 1009*idx + side)
        permutation = list(range(width))
        minimum = width + 1
        short = set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10 and len(short) < 8192:
                    short.add(word)

        for trial in range(36 if side == 0 else 24):
            probe.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)
            probe.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for previous in basis[:i]:
                    candidate = word ^ previous
                    offer(candidate)
                    cw = candidate.bit_count()
                    if candidate and cw < weight:
                        word, weight = candidate, cw
                basis[i] = word
            if minimum <= 6:
                break
        block_mask = (1 << q)-1
        short_loads = [
            sum(bool((word >> (j*q)) & block_mask) for word in short)
            for j in range(width // q)
        ]
        return (False, minimum, boundary, peak,
                len(short)/width, max(short_loads, default=0)/q)

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round phenomenological BP, NOT the scored ASC experiment.

        Noisy increments and check measurements are decoded jointly.
        Closing syndrome is perfect. Degree-weighted rates and random
        opposite-check pairs approximate exposure, not ASC-certified
        hooks. Unresolved BP is penalized even if relay/LSD could fix it.
        """
        import numpy as np

        def neighbors(rows):
            result = []
            for row in rows:
                ns = []
                while row:
                    bit = row & -row
                    ns.append(bit.bit_length()-1)
                    row ^= bit
                result.append(ns)
            return result

        ns = neighbors(checks)
        hooks = neighbors(stabilizers)
        m, rounds = len(checks), 6
        variables = rounds*(n+m)
        dummy = variables
        detectors = (rounds+1)*m
        row_width = max(map(len, ns))+2
        incidence = np.full((detectors, row_width), dummy,
                            dtype=np.int32)
        for t in range(rounds+1):
            for r, row in enumerate(ns):
                position = 0
                if t < rounds:
                    incidence[t*m+r, :len(row)] = [t*n+c for c in row]
                    position = len(row)
                    incidence[t*m+r, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = rounds*n+(t-1)*m+r

        flat = incidence.ravel()
        valid = flat != dummy
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

        gate_degree = np.asarray([
            x.bit_count()+z.bit_count()
            for x, z in zip(_fg_columns(checks, n),
                            _fg_columns(stabilizers, n))
        ], dtype=np.float64)
        data_rate = np.minimum(0.002*(2+2*gate_degree), 0.12)
        measurement_rate = np.asarray([
            0.002*(2+len(row))*2/3 for row in ns])
        base = np.concatenate((np.tile(data_rate, rounds),
                               np.tile(measurement_rate, rounds)))
        levels = np.asarray((0.60, 0.80, 1.00))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None]*base[None, :], 0.25)
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :variables] = (
            channel.random((shots, variables)) < rates)
        for t in range(rounds):
            chosen = channel.integers(len(hooks), size=shots)
            active = channel.random(shots) < 0.30
            for shot in np.flatnonzero(active):
                row = hooks[int(chosen[shot])]
                a = int(channel.integers(len(row)))
                b = int(channel.integers(len(row)-1))
                b += b >= a
                error[shot, t*n+row[a]] ^= True
                error[shot, t*n+row[b]] ^= True

        target = (error[:, flat].reshape(
            shots, detectors, row_width).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.empty((shots, variables+1), dtype=np.float64)
        prior[:, :variables] = np.log((1-rates)/rates)
        prior[:, dummy] = 30.0
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)
        span = _row_span(stabilizers)

        failures = effort = 0
        for iteration in range(1, 25):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1.e-12, 1-1.e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1.e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, detectors, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                residual = decoded_final[shot] ^ true_final[shot]
                word = 0
                for c in np.flatnonzero(residual):
                    word |= 1 << int(c)
                failures += int(not _in_row_span(word, span))
                effort += iteration
            keep = ~settled
            if not np.any(keep):
                return failures, 0, effort
            messages = np.clip(
                0.30*messages+0.70*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior, target = prior[keep], target[keep]
            target_sign, true_final = target_sign[keep], true_final[keep]

        unresolved = len(prior)
        return failures+unresolved, unresolved, effort+25*unresolved

    def probe_quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2])

    for batch in range(2):
        pools = [[], []]
        for side in range(2):
            seen_fillings = set()

            def offer(cost, serial, grid, rows):
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen_fillings:
                    return
                seen_fillings.add(signature)
                pools[side].append(
                    (cost, serial, clone(grid), list(rows)))
                pools[side].sort(key=lambda item: (item[0], item[1]))
                del pools[side][5:]

            for restart in range(4):
                for attempt in range(24):
                    grid = draw_grid(side)
                    rows = component_rows(grid, side)
                    current = _fg_cost(rows, widths[side])
                    if current is not None:
                        break
                else:
                    continue
                offer(current, restart * 121, grid, rows)

                for step in range(120):
                    nr, nc = shapes[side]
                    i, j = rng.randrange(nr), rng.randrange(nc)
                    old = grid[i][j]
                    if side == 1 and rng.random() < 0.3:
                        shift = G.pick(rng)
                        proposed_cell = [G.mul(shift, g) for g in old]
                    else:
                        # On single-row A, redraw its internal geometry;
                        # do not spend the search on whole-cell gauges.
                        proposed_cell = draw_cell(side, i, j)
                    assert len(proposed_cell) == len(old)
                    assert len(set(proposed_cell)) == len(old)
                    grid[i][j] = proposed_cell
                    proposed_rows = component_rows(grid, side)
                    proposed = _fg_cost(proposed_rows, widths[side])
                    if proposed is None:
                        grid[i][j] = old
                        continue
                    delta = energy(proposed) - energy(current)
                    temperature = 20 * (0.975 ** step) + 0.15
                    if delta > 0 and rng.random() >= math.exp(
                            -min(700.0, delta / temperature)):
                        grid[i][j] = old
                        continue
                    improved = proposed < current
                    current, rows = proposed, proposed_rows
                    if improved or step % 8 == 0:
                        offer(current, restart * 121 + step + 1,
                              grid, rows)
                offer(current, 1000 + restart, grid, rows)

        if not all(pools):
            continue

        finalists = [[], []]
        for side in range(2):
            for cost, serial, grid, rows in pools[side][:4]:
                bad, distance, boundary, peak, low, word_peak = refine(
                    rows, widths[side], side)
                finalists[side].append(
                    (cost, bad, distance, boundary, peak,
                     low, word_peak, serial, grid))

        options = []
        for a in finalists[0]:
            for b in finalists[1]:
                ac, abad, ad, abound, apeak, alow, awp, at, A = a
                bc, bbad, bd, bbound, bpeak, blow, bwp, bt, B = b
                # The protection targets are capped sampled upper bounds,
                # not evidence that larger reported distance improves pL.
                ga, gb = max(0, 12-ad), max(0, 10-bd)
                key = (int(abad)+int(bbad),
                       max(ga, gb), ga+gb, ac[0]+bc[0],
                       max(alow, blow), alow+blow, max(awp, bwp),
                       max(apeak, bpeak), abound+bbound,
                       ac[1]+bc[1], ac[2]+bc[2], at, bt)
                options.append((key, A, B))
        options.sort(key=lambda item: item[0])

        fallback = None
        verified = []
        examined = 0
        for key, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            # Hold rate fixed instead of letting rank surplus push a
            # weak code ahead of a stronger one in the proxy screen.
            if n != 588 or k != 168:
                continue
            if not _connected_checks(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx) == 10
            assert max(row.bit_count() for row in hz) == 12 <= W_MAX
            assert 6*sum(row.bit_count() for row in hx+hz) == 26208

            # Independently confirm cached helper contributions against
            # the complete matrices actually submitted.
            ar = [
                sum(((hx[r] >> (j * B_SHAPE[1] * q)) & mask)
                    << (j * q) for j in range(A_SHAPE[1]))
                for r in range(q)
            ]
            br = [row & ((1 << widths[1])-1)
                  for row in hz[:B_SHAPE[0]*q]]
            assert ar == component_rows(A, 0)
            assert br == component_rows(B, 1)
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            holonomy = G.mul(G.inv(B[1][4][0]), B[1][5][0])
            assert orders[holonomy] == 14
            assert A[0][0] and len(A[0][0]) == 3
            assert len(A[0][1]) == 4

            candidate = {
                "A": A, "B": B, "design": design, "mode": mode,
                "k": k, "screen": key, "geometry": wanted,
                "contract": (architecture, a_contract, palette),
                "quartet_spectrum": quartet_spectrum,
                "matching_order": 14, "cnots": 26208,
                "small_logical_safe": False,
            }
            if fallback is None:
                fallback = candidate
            examined += 1
            if not _has_short_logical(hx, hz, n):
                candidate["small_logical_safe"] = True
                verified.append((key, candidate, hx, hz))
            if len(verified) >= 3 or examined >= 12:
                break

        if verified:
            seed = 420588031 + 1009*idx
            coarse = []
            for number, (structure, candidate, hx, hz) in enumerate(verified):
                x = temporal_probe(hz, hx, 588, seed, 16)
                z = temporal_probe(hx, hz, 588, seed+1, 16)
                score = probe_quality(x, z)+(structure, number)
                coarse.append((score, number, structure,
                               candidate, hx, hz, x, z))
            coarse.sort(key=lambda item: item[:2])

            held_out = []
            for _, number, structure, candidate, hx, hz, cx, cz in coarse[:2]:
                # Independent holdouts, common across retained candidates.
                x = temporal_probe(hz, hx, 588, seed+1000003, 32)
                z = temporal_probe(hx, hz, 588, seed+1000004, 32)
                total_x = tuple(a+b for a, b in zip(cx, x))
                total_z = tuple(a+b for a, b in zip(cz, z))
                score = probe_quality(total_x, total_z)+(structure, number)
                candidate["temporal_bp_surrogate"] = (total_x, total_z)
                candidate["joint_screen"] = score
                held_out.append((score, candidate))
            held_out.sort(key=lambda item: item[0])
            return held_out[0][1]

        # Missing an optional protection preference must not discard a
        # legal connected exact-rate filling.
        if fallback is not None:
            return fallback
    raise ValueError("no verified unit34/long-matching filling with k=168")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell; no subsequent redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance alone raises measured fitness: zero-failure parent slot58 had d2s=22,k=45,s=7.165, below zero-failure slot20 with d2s=20,k=58,s=7.276; their sample ceilings limited both.
# REFUTED: Equal d2s=10, weight twelve and ASC depth fifteen ensure similar reliability on this bed: current slots15/17 scored 5.479/6.277, with slot15 reaching 100 Z failures in 34750 shots while slot17 had 31 Z failures in 200000.
# REFUTED: The parent's twenty triangle/order briefs reliably avoid Tanner-equivalent slots: the current sixty-slot evaluation retained forty distinct codes and collapsed twenty as isomorphic duplicates.
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
