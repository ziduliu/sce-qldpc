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
CONCEPT = ("C17, expressed by the frozen metacyclic m=17,r=1,s=1 rule, order 17, shapes (3,5)x(3,5): gate-neutral trinomial rewiring in B plus matched-noise component decoder screening should improve worse-basis memory fitness at verified k=72 and check weight at most nine.")
GROUP = ("Keep the exact group and order byte-for-byte under L1. This order-17 (3,5)x(3,5) bed measured 7.2396; current slots0 and44 saturated at s=7.370 in both bases. These readings support a filling refinement, not a causal group advantage.")
SHAPES = ("Frozen equal (3,5) components each have 85 variables and 51 checks. Size factor34 permits q<=17; the unchanged group gives n=578, kb=68 and guaranteed rate2/17. Verified ranks require k=72, rate0.124567 in cell0.100. Current X failures enriched Acol4/Bcol3 in slot4, several Bcol4 blocks in slot5, and Acol4/Bcol4 plus Arow0/Brow1 in slot59. The small samples do not identify a persistent isolated defect.")
FILLING = ("A remains fifteen monomials with augmentation rank one. New canonical B cards have one or two trinomial cells, column totals3/4, row totals5/6, total terms15..17 and augmentation rank three. Zeros compensate the larger cells: CSS weight remains<=9 and six-round CNOT count is24480..26112. Avoid two pure trinomial columns in the same row, which would force a weight-six component word. Fresh voltages retain verified-action annealing, exact kernel<=6 exclusion, component boundaries, sampled witnesses and short cycles. Final component profiles additionally run common-noise normalized min-sum stress tests, balancing the worse component before boundary tie-breaking. Complete verified matrices enforce k=72, connectivity, logical<=4 exclusion and CSS. Stress noise is degree-weighted and deliberately uncalibrated: it is neither relay-BP nor a circuit pL estimate. Target the parent's rare collective X failures while retaining k-kb=4; d2s=14 alone did not prevent slot59's two failures and39 logical flips.")
FAMILY_NAME = "r010_metacyclic17x1_35x35_gate_neutral_trinomial_stress_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same split metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    m, r, s = 17, 1, 1
    if q != m * r:
        raise ValueError("this metacyclic bed requires order 17")
    if pow(s, r, m) != 1:
        raise ValueError("invalid metacyclic action")
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m) * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m) * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _design_supports(idx):
    """Twenty different support designs, not permutation-orbit copies."""
    profiles = (
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 2, 2, 2),
        (1, 2, 3, 2, 2, 1),
        (1, 2, 3, 3, 2, 1),
        (1, 3, 2, 2, 1, 3),
        (3, 2, 2, 1, 2, 2),
        (3, 3, 1, 1, 1, 3),
        (3, 2, 2, 1, 3, 1),
        (3, 2, 1, 2, 2, 2),
        (3, 2, 2, 2, 2, 2),
        (3, 3, 1, 2, 1, 3),
        (3, 2, 2, 1, 2, 3),
    )
    pairs = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (0, 10), (0, 11), (1, 9), (2, 9), (3, 3),
        (3, 4), (3, 9), (4, 4), (4, 9), (9, 9),
    )
    a, b = pairs[idx % len(pairs)]
    return ([list(profiles[a][:3]), list(profiles[a][3:])],
            [list(profiles[b][:3]), list(profiles[b][3:])])


def _adjacent_coset_grid(rng, supports):
    """Every intra-cell ratio has order 6 or 7."""
    grid = []
    for row in supports:
        out = []
        for weight in row:
            t = rng.randrange(6)
            bs = [t] if weight == 1 else [t, (t + 1) % 6]
            while len(bs) < weight:
                bs.append((t + rng.randrange(2)) % 6)
            cell = []
            for b in bs:
                g = (rng.randrange(7), b)
                while g in cell:
                    g = (rng.randrange(7), b)
                cell.append(g)
            out.append(cell)
        grid.append(out)
    return grid


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
    """Gate-neutral trinomial cards and matched-noise decoder screening."""
    import math
    from itertools import combinations_with_replacement, permutations

    assert G.q == 17
    q = G.q
    width = 5 * q
    row_permutations = list(permutations(range(3)))

    # Allow a degree-three column inside one trinomial block, or a
    # degree-four column split 3+1. Neither increases its column budget.
    # These contracts are absent from the parent's <=2-term cells.
    column_types = [
        (a, b, c)
        for a in range(4) for b in range(4) for c in range(4)
        if a + b + c in (3, 4)
    ]
    patterns = set()
    for pattern in combinations_with_replacement(column_types, 5):
        totals = [sum(column[i] for column in pattern)
                  for i in range(3)]
        if min(totals) < 5 or max(totals) > 6 or sum(totals) > 17:
            continue
        triples = sum(t == 3 for column in pattern for t in column)
        if triples not in (1, 2):
            continue
        # Two pure trinomials in one component row commute, producing
        # a component kernel word of weight six regardless of voltages.
        if any(sum(column[i] == 3 and sum(column) == 3
                   for column in pattern) > 1 for i in range(3)):
            continue
        augmentation = [
            sum((column[i] & 1) << j
                for j, column in enumerate(pattern))
            for i in range(3)
        ]
        if gf2_rank(augmentation) != 3:
            continue
        canonical = min(
            tuple(sorted(tuple(column[p[i]] for i in range(3))
                         for column in pattern))
            for p in row_permutations
        )
        patterns.add(canonical)

    def card_priority(pattern):
        totals = [sum(column[i] for column in pattern)
                  for i in range(3)]
        return (sum(totals), max(totals) - min(totals),
                sum(0 in column for column in pattern), pattern)

    # These are different count architectures after explicit row/column
    # canonicalization, not saved-code automorphism images. Complete
    # Tanner equivalence remains the evaluator's decision.
    cards = sorted(patterns, key=card_priority)
    assert len(cards) >= max(20, N_SLOTS)
    design = idx
    mode = 0
    pattern = cards[design]
    sa = [[1] * 5 for _ in range(3)]
    sb = [[column[i] for column in pattern] for i in range(3)]
    assert _grid_weights(sa, sb) <= min(9, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert any(count == 0 for row in sb for count in row)
    assert gf2_rank([sum((w & 1) << j for j, w in enumerate(row))
                     for row in sa]) == 1
    assert gf2_rank([sum((w & 1) << j for j, w in enumerate(row))
                     for row in sb]) == 3

    def grid(support):
        return [[rng.sample(G.elements, count) for count in row]
                for row in support]

    # Cache only actions returned by the construction helper. XOR and
    # block placement below use linearity, not independently coded
    # group left/right conventions.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid_cells, side):
        rows = []
        for cell_row in grid_cells:
            out = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, word in enumerate(block):
                        out[r] ^= word << (j * q)
            rows.extend(out)
        return rows

    def components(hx, hz):
        """Project A-left/B-right from the complete verified CSS lift."""
        mask = (1 << q) - 1
        ar = []
        for i in range(3):
            for r in range(q):
                row = hx[(i * 5) * q + r]
                packed = 0
                for j in range(5):
                    packed |= ((row >> (j * 5 * q)) & mask) << (j * q)
                ar.append(packed)
        bmask = (1 << width) - 1
        br = [hz[a * q + r] & bmask
              for a in range(3) for r in range(q)]
        return ar, br

    def columns(rows):
        cols = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return cols

    def short_and_boundary(cols):
        """Exact component kernel<=6 exclusion and triple-boundary cost."""
        seen = {0}
        for value in cols:
            if value in seen:
                return None
            seen.add(value)
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return None
                seen.add(value)
        boundary = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return None
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 6:
                        boundary += (6 - weight) ** 2
        return boundary

    def sampled_words(cols, side):
        """Verified kernel witnesses; minimum remains only an upper bound.

        Give finalists the same permutation stream within this slot.
        Count only distinct sampled short supports, not logical qubits.
        """
        srng = random.Random(81371 + 1009 * idx + int(side == "B"))
        order = list(range(width))
        best = width + 1
        short = set()
        for _ in range(32):
            srng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    candidates = [word]
                    candidates.extend(word ^ other for other in basis)
                    for v in candidates:
                        if not v:
                            continue
                        weight = v.bit_count()
                        best = min(best, weight)
                        if weight <= 12:
                            short.add(v)
                    basis.append(word)
        return best, len(short)

    def decoder_stress(rows, side):
        """Matched-noise classical component decoding, not circuit pL.

        All incidence data come from the verified component rows.
        Common uniforms make competing voltage profiles see the same
        channel samples. Degree weighting is only a gate-burden model.
        A wrong valid codeword counts as failure, as does nonconvergence.
        """
        import numpy as np

        nr = len(rows)
        check_edges = []
        variable_edges = [[] for _ in range(width)]
        edge_variables = []
        flat_locations = []
        maximum = max(row.bit_count() for row in rows)
        matrix = np.zeros((nr, width), dtype=np.uint8)

        for r, row in enumerate(rows):
            group = []
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                edge = len(edge_variables)
                group.append(edge)
                variable_edges[c].append(edge)
                edge_variables.append(c)
                matrix[r, c] = 1
                flat_locations.append(r * maximum + len(group) - 1)
                row ^= bit
            check_edges.append(group)

        ne = len(edge_variables)
        checks = np.full((nr, maximum), ne, dtype=np.int32)
        for r, group in enumerate(check_edges):
            checks[r, :len(group)] = group
        degree = np.array([len(group) for group in variable_edges],
                          dtype=np.float64)
        max_degree = max(map(len, variable_edges))
        variables = np.full((width, max_degree), ne, dtype=np.int32)
        for c, group in enumerate(variable_edges):
            variables[c, :len(group)] = group
        edge_variables = np.array(edge_variables, dtype=np.int32)
        flat_locations = np.array(flat_locations, dtype=np.int32)
        positions = np.arange(maximum)[None, None, :]

        shots = 160
        srng = random.Random(97003 + 1013 * idx + int(side == "B"))
        uniforms = np.array([
            [srng.random() for _ in range(width)]
            for _ in range(shots)
        ])
        outcomes = []

        for base_probability in (0.035, 0.060):
            probabilities = base_probability * degree / 3.0
            errors = uniforms < probabilities[None, :]
            magnitude = np.log((1.0 - probabilities) / probabilities)
            prior = np.where(errors, -magnitude[None, :],
                             magnitude[None, :])
            messages = prior[:, edge_variables].copy()
            done = np.zeros(shots, dtype=bool)
            failed = np.zeros(shots, dtype=bool)
            late = 0

            for iteration in range(1, 33):
                padded = np.concatenate(
                    (messages, np.full((shots, 1), 1.0e6)), axis=1)
                incoming = padded[:, checks]
                absolute = np.abs(incoming)
                minimum_index = absolute.argmin(axis=2)
                minimum = absolute.min(axis=2)
                second = np.partition(absolute, 1, axis=2)[:, :, 1]
                signs = np.where(incoming >= 0.0, 1.0, -1.0)
                product = signs.prod(axis=2)
                excluded_minimum = np.where(
                    positions == minimum_index[:, :, None],
                    second[:, :, None], minimum[:, :, None])
                outgoing = (0.8 * product[:, :, None]
                            * signs * excluded_minimum)
                check_messages = outgoing.reshape(shots, -1)[
                    :, flat_locations]
                padded_checks = np.concatenate(
                    (check_messages, np.zeros((shots, 1))), axis=1)
                posterior = prior + padded_checks[:, variables].sum(axis=2)
                proposal = posterior[:, edge_variables] - check_messages
                messages = 0.2 * messages + 0.8 * proposal

                if iteration % 4 == 0:
                    hard = (posterior < 0.0).astype(np.uint8)
                    syndrome = (hard @ matrix.T) & 1
                    resolved = (~done) & ~syndrome.any(axis=1)
                    failed[resolved] = hard[resolved].any(axis=1)
                    if iteration >= 16:
                        late += int(resolved.sum())
                    done |= resolved
                    if done.all():
                        break

            unresolved = int((~done).sum())
            outcomes.append((int((failed | ~done).sum()),
                             unresolved, late))

        # First protect the lower-stress sample; the stronger channel
        # separates tied profiles. No optimistic distance is rewarded.
        return (outcomes[0][0], outcomes[1][0],
                outcomes[0][1] + outcomes[1][1],
                outcomes[0][2] + outcomes[1][2])

    def six_cycles(rows):
        """Exact Tanner six-cycles using three distinct variables."""
        nr = len(rows)
        overlap = [[0] * nr for _ in range(nr)]
        for i in range(nr):
            for j in range(i):
                overlap[i][j] = overlap[j][i] = (
                    rows[i] & rows[j]).bit_count()
        total = 0
        for i in range(nr):
            for j in range(i):
                x = overlap[i][j]
                if not x:
                    continue
                common = rows[i] & rows[j]
                for h in range(j):
                    y, z = overlap[j][h], overlap[i][h]
                    if y and z:
                        t = (common & rows[h]).bit_count()
                        total += x * y * z - t * (x + y + z) + 2 * t
        return total

    def cycle_cost(rows):
        """Exact C4 count and a cheap C6 proposal heuristic.

        The second statistic is exact when C4=0. Final refinement
        below always uses the exact six-cycle routine.
        """
        nr = len(rows)
        adjacency = [0] * nr
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adjacency[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count() for c in columns(rows)))
        return four, max(0, triangles - stars)

    def clone(cells):
        return [[list(cell) for cell in row] for row in cells]

    def portfolio(side, support):
        """Anneal labels inside this slot's fixed support architecture."""
        positions = [(i, j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]
        wanted_rank = 3 * q - 2 if side == "A" else 3 * q
        pool = []
        seen = set()
        serial = 0

        def energy(cost):
            return (64.0 * cost[0] + cost[1] / 6.0) / q

        def remember(cells, rows, cost):
            nonlocal serial
            if len(pool) >= 10 and cost > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in cells for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            # These component ranks are only a preliminary screen.
            # Complete verified CSS ranks still decide final k.
            if gf2_rank(rows) != wanted_rank:
                return
            serial += 1
            pool.append((cost, serial, clone(cells), list(rows)))
            pool.sort(key=lambda entry: entry[:2])
            del pool[10:]

        for restart in range(3):
            cells = grid(support)
            rows = component(cells, side)
            current = cycle_cost(rows)
            remember(cells, rows, current)
            for step in range(240):
                i, j = rng.choice(positions)
                old = cells[i][j]
                cells[i][j] = rng.sample(G.elements, support[i][j])
                proposed_rows = component(cells, side)
                proposed = cycle_cost(proposed_rows)
                delta = energy(proposed) - energy(current)
                temperature = 8.0 * (0.985 ** step) + 0.05
                accept = (delta <= 0 or rng.random() <
                          math.exp(-min(700.0, delta / temperature)))
                if not accept:
                    cells[i][j] = old
                    continue
                improved = proposed < current
                rows, current = proposed_rows, proposed
                if improved or step % 16 == 0:
                    remember(cells, rows, current)
            remember(cells, rows, current)
        return pool

    for batch in range(3):
        apool = portfolio("A", sa)
        bpool = portfolio("B", sb)
        pool = []
        trial = 0
        for acost, _, A, expected_a in apool:
            for bcost, _, B, expected_b in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                ar, br = components(hx, hz)
                # Confirm both cached action conventions against the
                # complete matrices that will actually be submitted.
                assert ar == expected_a and br == expected_b
                coarse = (max(acost[0], bcost[0]),
                          acost[0] + bcost[0],
                          max(acost[1], bcost[1]),
                          acost[1] + bcost[1])
                pool.append((coarse, trial, A, B, hx, hz, n, ar, br))
                trial += 1
        pool.sort(key=lambda item: (item[0], item[1]))

        finalists = []
        eligible = 0
        refined = {}

        def profile(rows, side):
            signature = side, tuple(rows)
            if signature not in refined:
                cols = columns(rows)
                boundary = short_and_boundary(cols)
                if boundary is None:
                    refined[signature] = None
                else:
                    distance, count = sampled_words(cols, side)
                    stress = decoder_stress(rows, side)
                    refined[signature] = (
                        boundary, distance, count, six_cycles(rows), stress)
            return refined[signature]

        for item in pool:
            coarse, trial, A, B, hx, hz, n, ar, br = item
            # Augmentation and component ranks motivate 72 but never
            # replace ranks of the actual evaluator-convention matrices.
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 578 and k == 72 and 8 * k < n):
                continue
            if not _connected_checks(hx, hz, n):
                continue
            eligible += 1
            ap = profile(ar, "A")
            bp = profile(br, "B") if ap is not None else None
            if ap is not None and bp is not None:
                ba, da, na, ca, stress_a = ap
                bb, db, nb, cb, stress_b = bp
                distance = min(da, db)
                boundary = max(ba, bb), ba + bb
                multiplicity = max(na, nb), na + nb
                stress = tuple(
                    value for t in range(len(stress_a))
                    for value in (max(stress_a[t], stress_b[t]),
                                  stress_a[t] + stress_b[t]))
                # Component stress does not reproduce relay-BP, noisy
                # measurements or ASC hooks. It supplies a new risk
                # hypothesis after the existing protection warning.
                key = (int(distance < 12), stress, boundary, multiplicity,
                       coarse, max(ca, cb), ca + cb,
                       -min(distance, 14), abs(da - db), trial)
                finalists.append((key, item, k))
            if eligible >= 48:
                break

        finalists.sort(key=lambda entry: entry[0])
        for key, item, k in finalists:
            _, _, A, B, hx, hz, n, _, _ = item
            if _has_short_logical(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            expected_cnots = 6 * q * 8 * (
                sum(map(sum, sa)) + sum(map(sum, sb)))
            assert any(len(cell) == 3 for row in B for cell in row)
            assert 24480 <= expected_cnots <= 26112
            assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
            return {"cells": {"A": A, "B": B}, "design": design,
                    "mode": mode, "k": k, "screen": key,
                    "support_card": pattern,
                    "cnots": expected_cnots}

    raise ValueError("No gate-neutral trinomial k=72 filling passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened cell, including prescribed trinomials."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: k=66, check weight ten and d2s=15 certify zero-failure memory on this order-42 bed: slot25 had four Z failures, 47 logical flips and s=6.888, while slots2,5,13,28 with the same reported parameters had zero failures and s=7.332.
# REFUTED: This screened sparse-A-tripleton family improves the lighter order-17 bed: fitness instead fell from 7.2396 to 6.7924, and slot22 had 23 X failures/200000 with 623 logical flips despite d2s=14; supports and labels both changed, so this does not isolate a causal term-count effect.
# REFUTED: k=72, check weight nine and d2s=14 guarantee ceiling-level memory on C17 (3,5)x(3,5): current slot59 had two X failures, 39 collective logical flips and s=7.078, while slot44 with the same reported parameters saturated at s=7.370.
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
