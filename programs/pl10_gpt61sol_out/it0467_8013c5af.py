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
CONCEPT = ("For cyclic C29, order 29, shapes (2,4)x(2,4), full-doubleton components with controlled relative-gap overlap and fewer verified BP miscorrections should improve the approximately 7.1 incumbent in cell 0.250. Increased extraction exposure makes this a falsifiable protection-versus-gate-cost experiment.")
GROUP = ("Jump from nonabelian C9 semidirect C3 to genuinely different cyclic C29. BEDS records approximately 7.3 for C29 on these shapes, and FRONTIER records approximately 7.1 for its full-doubleton 0.250 filling. These motivate a fresh decoder-aware geometry experiment, not a group-only advantage. Order29 is the largest legal lift, giving n580.")
SHAPES = ("Retain equal (2,4) components, now with 58 checks and 116 variables each. The guaranteed bound is kb116, rate0.200. All-even cell augmentation motivates additional logicals, but require complete helper-built n580,k132, rate0.227586 in cell0.250. Parent measured failures implicated different Acol/Bcol blocks across slots, including Bcol3 and Bcol0, and several Acol/Brow checks; the small samples do not establish a persistent hotspot.")
FILLING = ("Twenty indexed contracts cross four cross-row direction-overlap portfolios with five canonical relative-gap arrangements; extra harness batches extend each catalogue. Every cell is a distinct-element doubleton. Component rows weigh eight and columns four, so both CSS check weights are twelve and extraction uses33408 CNOTs, versus23328 in the parent. Canonical relative keys remove obvious group/permutation images, not arbitrary Tanner isomorphisms. Verified actions guide offset search; exact component words through six, capped sampled protection and translation-orbit concentration screen finalists. Complete ranks, connectivity, CSS and quantum logicals through four are checked. Common-stream complete-code BP probes classify settled residuals modulo actual stabilizers and select against worse-basis collective ambiguity. Parent slot0 had ten Z failures/200000 and348 logical flips, while slot21 had two failures in each basis; equal w9,d2s12 did not determine reliability. Probes omit syndrome histories, ASC ordering and relay/LSD and are not calibrated circuit pL.")
FAMILY_NAME = "r025_C29_24x24_doubleton_overlap_jointdecode_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C29 requires q=29")
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

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _columns(rows, width):
    result = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            result[bit.bit_length() - 1] |= tag
            row ^= bit
    return result


def _rows_from_columns(columns, height):
    rows = [0] * height
    for c, syndrome in enumerate(columns):
        tag = 1 << c
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length() - 1] |= tag
            syndrome ^= bit
    return rows


def _component_cycles(columns):
    """Four-cycles from shared check pairs; also return check adjacency."""
    pairs = {}
    adjacency = {}
    cycles = 0
    for syndrome in columns:
        nodes = []
        while syndrome:
            bit = syndrome & -syndrome
            nodes.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, u in enumerate(nodes):
            for v in nodes[:a]:
                pair = (v, u)
                old = pairs.get(pair, 0)
                cycles += old
                pairs[pair] = old + 1
                adjacency[u] = adjacency.get(u, 0) | (1 << v)
                adjacency[v] = adjacency.get(v, 0) | (1 << u)
    return cycles, adjacency


def _component_summary(columns, adjacency):
    """Exact weight-four kernels when columns are distinct and nonzero."""
    duplicates = len(columns) - len(set(columns)) + int(0 in columns)
    pairs = {}
    collisions = 0
    for a, x in enumerate(columns):
        for y in columns[:a]:
            syndrome = x ^ y
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1

    triangles = 0
    for u, neighbors in adjacency.items():
        higher = neighbors & ~((1 << (u + 1)) - 1)
        while higher:
            bit = higher & -higher
            v = bit.bit_length() - 1
            triangles += (
                neighbors & adjacency.get(v, 0)
                & ~((1 << (v + 1)) - 1)
            ).bit_count()
            higher ^= bit
    # Exact Tanner six-cycles when four-cycles are absent; otherwise
    # this is only a secondary adjacency statistic.
    six_cycles = max(0, triangles - len(columns))
    return duplicates, collisions // 3, six_cycles


def _six_kernel_count(columns):
    """Exact weight-six count, called only without weight-two/four words."""
    histogram = {}
    collisions = 0
    width = len(columns)
    for a in range(width - 2):
        x = columns[a]
        for b in range(a + 1, width - 1):
            ab = x ^ columns[b]
            for c in range(b + 1, width):
                syndrome = ab ^ columns[c]
                old = histogram.get(syndrome, 0)
                collisions += old
                histogram[syndrome] = old + 1
    # Colliding triples are disjoint under the precondition.
    return collisions // 10


def _small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    # Linear quotient representatives of unit vectors. Each pivot row
    # only uses lower-numbered coordinates, so ascending substitution
    # computes their full stabilizer-row-space remainders.
    residues = [1 << c for c in range(n)]
    for p in sorted(pivots):
        tail = pivots[p] ^ (1 << p)
        value = 0
        while tail:
            bit = tail & -tail
            value ^= residues[bit.bit_length() - 1]
            tail ^= bit
        residues[p] = value

    columns = _columns(checks, n)
    seen = {0: 0}
    for c, syndrome in enumerate(columns):
        value = residues[c]
        old = seen.get(syndrome)
        if old is not None and old != value:
            return True
        seen[syndrome] = value

    for a, x in enumerate(columns):
        ra = residues[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            value = ra ^ residues[b]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
    return False


def _jump_bp_probe(checks, stabilizers, n, seed):
    """Complete-code ambiguity surrogate, not the measured ASC circuit.

    Degree-weighted independent errors and opposite-check pairs provide
    a common stress bank. Pairs are not certified scheduled ASC hooks.
    Settled residuals are classified modulo the actual stabilizer span.
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
    ev, redges = [], []
    vedges = [[] for _ in range(n)]
    for row in neighborhoods:
        edges = []
        for c in row:
            edge = len(ev)
            ev.append(c)
            edges.append(edge)
            vedges[c].append(edge)
        redges.append(edges)

    ne, shots = len(ev), 96
    ce = np.full((len(checks), max(map(len, redges))), ne,
                 dtype=np.int32)
    ve = np.full((n, max(map(len, vedges))), ne, dtype=np.int32)
    for r, edges in enumerate(redges):
        ce[r, :len(edges)] = edges
    for c, edges in enumerate(vedges):
        ve[c, :len(edges)] = edges
    valid = ce != ne
    ev = np.asarray(ev, dtype=np.int32)
    cv = np.concatenate((ev, np.asarray([n], dtype=np.int32)))[ce]

    degree = np.asarray([len(edges) for edges in vedges])
    other_degree = np.asarray(
        [value.bit_count() for value in _columns(stabilizers, n)])
    channel = np.random.default_rng(seed)
    levels = np.asarray((0.010, 0.017, 0.026))
    rates = np.minimum(
        0.20, (degree + other_degree)[:, None] / 8.0
        * levels[np.arange(shots) % 3][None, :])
    errors = np.zeros((n + 1, shots), dtype=np.uint8)
    errors[:n] = channel.random((n, shots)) < rates
    for shot in range(shots):
        if channel.random() < 0.40:
            row = opposite[int(channel.integers(len(opposite)))]
            for endpoint in channel.choice(len(row), 2, replace=False):
                errors[row[int(endpoint)], shot] ^= 1

    target = np.bitwise_xor.reduce(errors[cv], axis=1)
    target_sign = np.where(target, -1.0, 1.0)
    prior = np.log((1.0 - rates) / rates)
    incoming = np.zeros((ne + 1, shots), dtype=np.float64)
    incoming[:ne] = prior[ev]
    incoming[ne] = 30.0
    messages = np.zeros_like(incoming)
    hard = np.zeros_like(errors)
    decoded = np.zeros((n, shots), dtype=np.uint8)
    active = np.ones(shots, dtype=bool)
    slots = np.arange(ce.shape[1])[None, :, None]
    work = 0

    for iteration in range(36):
        work += int(active.sum())
        gathered = incoming[ce]
        signs = np.where(gathered < 0, -1.0, 1.0)
        magnitude = np.abs(gathered)
        at = np.argmin(magnitude, axis=1)
        first = np.min(magnitude, axis=1)
        second = np.partition(magnitude, 1, axis=1)[:, 1, :]
        excluded = np.where(
            slots == at[:, None, :],
            second[:, None, :], first[:, None, :])
        parity = np.prod(signs, axis=1) * target_sign
        outgoing = 0.80 * excluded * signs * parity[:, None, :]
        messages[ce[valid]] = outgoing[valid]
        messages[ne] = 0.0
        posterior = prior + messages[ve].sum(axis=1)
        hard[:n] = posterior < 0
        syndrome = np.bitwise_xor.reduce(hard[cv], axis=1)
        settled = active & np.all(syndrome == target, axis=0)
        decoded[:, settled] = hard[:n, settled]
        active[settled] = False
        if not active.any():
            break
        damping = 0.30 if iteration < 12 else 0.50
        proposal = posterior[ev] - messages[:ne]
        incoming[:ne] = (
            damping * incoming[:ne]
            + (1.0 - damping) * np.clip(proposal, -24.0, 24.0))
        incoming[ne] = 30.0

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    wrong = 0
    for shot in np.flatnonzero(~active):
        word = 0
        for c in np.flatnonzero(errors[:n, shot] ^ decoded[:, shot]):
            word |= 1 << int(c)
        while word:
            old = pivots.get(word.bit_length() - 1)
            if old is None:
                wrong += 1
                break
            word ^= old
    unresolved = int(active.sum())
    return wrong + unresolved, wrong, unresolved, work


def NEW_CTX(rng, G, idx):
    """Indexed doubleton-overlap contracts and exact-rate joint screening."""
    import math

    q = G.q
    assert q == 29 and A_SHAPE == B_SHAPE == (2, 4)
    counts = {
        "A": [[2] * 4 for _ in range(2)],
        "B": [[2] * 4 for _ in range(2)],
    }
    assert _no_dead_lines(counts["A"], counts["B"])
    assert _grid_weights(counts["A"], counts["B"]) == 12 <= W_MAX
    for support in counts.values():
        assert all(sum(row) == 8 for row in support)
        assert all(sum(column) == 4 for column in zip(*support))
    # Independently check both CSS budgets.
    assert max(map(sum, counts["A"])) + max(
        map(sum, zip(*counts["B"]))) == 12
    assert max(map(sum, counts["B"])) + max(
        map(sum, zip(*counts["A"]))) == 12

    elements = list(G.elements)
    architecture = (idx // 5) % 4
    overlap_cards = ((0, 0), (0, 1), (1, 1), (1, 2))

    def direction(d):
        d %= q
        return min(d, q - d)

    def geometry_key(grids):
        # Origins, common C29 automorphisms, row/column permutations and
        # side exchange are not separate contracts. This is not complete
        # classification of arbitrary quantum Tanner isomorphisms.
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                columns = [
                    (direction(u * grid[0][j][1][0]),
                     direction(u * grid[1][j][1][0]))
                    for j in range(4)]
                sides.append(min(tuple(sorted(columns)),
                                 tuple(sorted((b, a) for a, b in columns))))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Four overlap portfolios, five distinct canonical ratio arrangements
    # each. Later batches consume further arrangements, not saved images.
    grng = random.Random(291250731 + 1009 * architecture)
    wanted = 5 * (idx // 20) + idx % 5
    book, seen_geometry = [], set()
    directions = list(range(1, 15))
    for attempt in range(40000):
        grids = []
        for overlap in overlap_cards[architecture]:
            upper = grng.sample(directions, 4)
            lower = grng.sample(upper, overlap)
            lower += grng.sample(
                [d for d in directions if d not in upper], 4 - overlap)
            grng.shuffle(lower)
            assert len(set(upper)) == len(set(lower)) == 4
            assert len(set(upper) & set(lower)) == overlap
            grids.append([
                [[G.e, (d,)] for d in upper],
                [[G.e, (d,)] for d in lower],
            ])
        signature = geometry_key(grids)
        if signature in seen_geometry:
            continue
        seen_geometry.add(signature)
        book.append(grids)
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("C29 doubleton overlap catalogue exhausted")
    geometry = dict(zip(("A", "B"), book[wanted]))

    # Singleton contributions come exclusively from the verified helper.
    mask = (1 << q) - 1
    templates = {"A": [], "B": []}
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        templates["A"].append(_columns([row & mask for row in hx], q))
        templates["B"].append(_columns([row & mask for row in hz], q))

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component_columns(side, grid):
        result = [0] * (4 * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = templates[side][G.idx(g)]
                    for t, syndrome in enumerate(block):
                        result[j * q + t] ^= syndrome << (i * q)
        assert all(s.bit_count() == 4 for s in result)
        return result

    def connected_component(adjacency):
        seen, todo = 0, 1
        while todo:
            bit = todo & -todo
            todo ^= bit
            if seen & bit:
                continue
            seen |= bit
            todo |= adjacency.get(bit.bit_length() - 1, 0) & ~seen
        return seen.bit_count() == 2 * q

    def sampled_kernel(columns, rows):
        """Common-stream kernel witnesses; upper bounds, not circuit pL."""
        srng = random.Random(291251331 + 1009 * idx)
        order = list(range(len(columns)))
        best = len(columns) + 1
        short = set()

        def offer(word):
            nonlocal best
            if word:
                weight = word.bit_count()
                best = min(best, weight)
                if weight <= 16 and len(short) < 2048:
                    short.add(word)

        for _ in range(40):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome and word:
                    offer(word)
                    basis.append(word)
            srng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for previous in basis[:i]:
                    proposal = word ^ previous
                    offer(proposal)
                    pw = proposal.bit_count()
                    if proposal and pw < weight:
                        word, weight = proposal, pw
                basis[i] = word
            if best <= 6:
                break

        assert all(all((row & word).bit_count() % 2 == 0 for row in rows)
                   for word in short)

        def rotate(word, shift, blocks):
            return sum(
                (((((word >> (j * q)) & mask) << shift)
                  | (((word >> (j * q)) & mask) >> (q - shift))) & mask)
                << (j * q) for j in range(blocks))

        # Verify the symmetry used to classify witnesses. It never
        # replaces a submitted filling with a translated code.
        for j in range(4):
            for shift in range(q):
                assert columns[j * q + shift] == rotate(
                    columns[j * q], shift, 2)
        orbits = {
            min(rotate(word, shift, 4) for shift in range(q))
            for word in short
        }
        loads = [
            sum(((word >> (j * q)) & mask).bit_count() for word in orbits)
            for j in range(4)]
        return best, len(orbits), max(loads, default=0)

    def quality(key, columns):
        duplicates, n4, four, six_graph, deficit = key
        singles = set(columns)
        pairs = set()
        triples = 0
        for a, x in enumerate(columns):
            for y in columns[:a]:
                syndrome = x ^ y
                triples += int(syndrome in singles)
                pairs.add(syndrome)
        # Even-degree columns do NOT exclude odd-weight kernel words.
        # Explicitly test weight three before applying collision counts.
        if duplicates or n4 or triples:
            return (1, 1, 10**9, 10**9, 10**9,
                    10**9, 10**9, four, six_graph, deficit)

        histogram = {}
        five = collisions = boundary = 0
        loads = [0] * 4
        for a in range(len(columns)):
            for b in range(a):
                pair = columns[a] ^ columns[b]
                for c in range(b):
                    syndrome = pair ^ columns[c]
                    five += int(syndrome in pairs)
                    collisions += histogram.get(syndrome, 0)
                    histogram[syndrome] = histogram.get(syndrome, 0) + 1
                    weight = syndrome.bit_count()
                    if weight < 8:
                        penalty = (8 - weight) ** 2
                        boundary += penalty
                        for site in (a, b, c):
                            loads[site // q] += penalty
        # Without kernel words <=4, these are disjoint support partitions.
        assert five % 10 == 0 and collisions % 10 == 0
        n5, n6 = five // 10, collisions // 10
        del histogram
        rows = _rows_from_columns(columns, 2 * q)
        distance, short, peak = sampled_kernel(columns, rows)
        # Sixteen is a coarse sampled floor, never a distance certificate.
        # Upper bounds above it receive no extra reward.
        return (int(bool(n5 or n6)), int(distance < 16), n5 + n6,
                short, peak, max(loads), boundary,
                four, six_graph, deficit)

    def optimize(side):
        support = counts[side]
        # A lone nonzero cell in a column has only an origin gauge;
        # spend offset mutations on columns incident to both row orbits.
        sites = [(i, j) for i in range(2) for j in range(4)
                 if support[i][j] and support[1 - i][j]]
        pool, seen = [], set()

        def remember(grid, columns, four, adjacency, serial):
            if not connected_component(adjacency):
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            rows = _rows_from_columns(columns, 2 * q)
            deficit = len(rows) - gf2_rank(rows)
            # The two all-even augmentation rows force at least two
            # component deficits. Require no additional deficit here;
            # complete verified CSS ranks still decide the emitted rate.
            if deficit > 2:
                return
            duplicates, n4, six_graph = _component_summary(
                columns, adjacency)
            key = (duplicates, n4, four, six_graph, deficit)
            pool.append((key, serial, clone(grid), list(columns)))
            pool.sort(key=lambda entry: (entry[0], entry[1]))
            del pool[6:]

        for restart in range(4):
            current = []
            for row in geometry[side]:
                out = []
                for cell in row:
                    origin = G.pick(rng)
                    out.append([G.mul(origin, g) for g in cell])
                current.append(out)
            columns = component_columns(side, current)
            cost, adjacency = _component_cycles(columns)
            remember(current, columns, cost, adjacency, restart * 241)
            for step in range(240):
                i, j = rng.choice(sites)
                trial = clone(current)
                shift = G.pick(rng)
                trial[i][j] = [G.mul(shift, g) for g in trial[i][j]]
                tc = component_columns(side, trial)
                trial_cost, trial_adjacency = _component_cycles(tc)
                delta = (trial_cost - cost) / q
                temperature = 3.0 * (0.98 ** step) + 0.04
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current, columns = trial, tc
                    cost, adjacency = trial_cost, trial_adjacency
                if step % 12 == 0 and (cost == 0 or step % 60 == 0):
                    remember(current, columns, cost, adjacency,
                             restart * 241 + step + 1)
            remember(current, columns, cost, adjacency, 2000 + restart)

        if not pool:
            raise ValueError("no connected small-deficit component")
        finalists = [
            (quality(key, columns), serial, grid)
            for key, serial, grid, columns in pool
        ]
        finalists.sort(key=lambda entry: (entry[0], entry[1]))
        return finalists

    def full_connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            bit = row & -row
            anchor = bit.bit_length() - 1
            row ^= bit
            while row:
                bit = row & -row
                a, b = root(anchor), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(a) for a in range(width)}) == 1

    for batch in range(2):
        a_pool, b_pool = optimize("A"), optimize("B")
        options = []
        for aq, at, A in a_pool:
            for bq, bt, B in b_pool:
                score = tuple(value for a, b in zip(aq, bq)
                              for value in (max(a, b), a + b))
                options.append((score, at, bt, A, B))
        options.sort(key=lambda entry: entry[:3])

        prepared, seen_fillings = [], set()
        for score, at, bt, A, B in options:
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if signature in seen_fillings:
                continue
            seen_fillings.add(signature)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 580 or k != 132:
                continue
            if not full_connected(hx + hz, n):
                continue

            # Confirm the cached component action against the complete
            # code, rather than relying on an independent lift convention.
            ar = [
                sum(((hx[(i * 4) * q + t] >> (j * 4 * q)) & mask)
                    << (j * q) for j in range(4))
                for i in range(2) for t in range(q)
            ]
            br = [row & ((1 << (4 * q)) - 1) for row in hz[:2 * q]]
            assert ar == _rows_from_columns(
                component_columns("A", A), 2 * q)
            assert br == _rows_from_columns(
                component_columns("B", B), 2 * q)
            assert css_ok(hx, hz)
            assert all(row.bit_count() == 12 for row in hx + hz)
            assert 6 * sum(row.bit_count() for row in hx + hz) == 33408
            assert 225 * n <= 1000 * k < 275 * n
            assert all(
                len(cell) == 2 and len(set(cell)) == 2
                for grid in (A, B) for row in grid for cell in row)
            safe = not (_small_logical(hx, hz, n)
                        or _small_logical(hz, hx, n))
            candidate = {
                "A": A, "B": B, "design": idx % 20,
                "geometry": wanted, "overlap": overlap_cards[architecture],
                "k": k, "component_screen": score,
                "small_logical_safe": safe, "cnots": 33408,
            }
            prepared.append((score, candidate, hx, hz))
            if len(prepared) == 4:
                break

        if not prepared:
            continue

        winner = None
        seed = 291252503 + 1009 * idx
        for score, candidate, hx, hz in prepared:
            x = _jump_bp_probe(hz, hx, 580, seed)
            z = _jump_bp_probe(hx, hz, 580, seed + 1)
            decoder_key = (
                max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                max(x[2], z[2]), x[2] + z[2],
                x[3] + z[3])
            quality_key = (
                int(not candidate["small_logical_safe"]),
            ) + decoder_key + score
            if winner is None or quality_key < winner[0]:
                candidate["bp_surrogate"] = (x, z)
                candidate["selection_score"] = quality_key
                winner = quality_key, candidate
        return winner[1]
    raise ValueError("no verified C29 k132 filling in cell 0.250")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Distinct rotational A-binomial steps and raw syndrome-ambiguity screening sufficiently protect the D14 (1,4)x(2,7) bed: fitness was 3.2219, measured slots retained abundant weight-three sector-1 logicals, and X failures were 108-139 per 250 shots.
# REFUTED: The full-row-rank component walk produces a usable filling for every support card: current slots1,4,5 and many others were voided because no full-row-rank filling was found.
# REFUTED: Equal n=560,k=112,d2s=10 with component-kernel screening ensures basis-balanced reliability: slot56 had 43 X failures versus one Z failure/200000 and s=6.365, while slot0 scored 7.118.
# REFUTED: Canonical order-nine relative-support keys suffice for Tanner-distinct submissions: current slots1 and3 were iso_dup_of_0 and iso_dup_of_2 despite their indexed geometry contracts.
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
