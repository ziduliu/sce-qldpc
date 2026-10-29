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
CONCEPT = ("Cyclic C17 in the frozen metacyclic encoding, order 17, shapes (3,5)x(3,5): one even-augmentation degree-four A column and decoder-stress-selected voltages should improve worse-basis ASC memory above 7.2396 at verified k=72 and check weight at most ten.")
GROUP = ("Keep the exact group definition and order byte-for-byte under L1. This C17 (3,5)x(3,5) bed reached 7.3338 in the shown measurements, while the current parent scored 7.2396. These support a filling refinement, not a causal group advantage.")
SHAPES = ("Frozen equal (3,5) components each have 85 variables and 51 checks. Size factor34 permits q<=17, giving n=578, kb=68 and guaranteed rate2/17. Require complete verified k=72, rate0.124567 in cell0.100. Current failures implicated varying Acol/Bcol and Arow/Brow blocks: slot1 included Acol2/Bcol4, slot4 several Bcol3 blocks, and slot7 Acol2/Bcol1. One or two failed shots do not identify a persistent hotspot.")
FILLING = ("A has four monomial columns and one count column(0,2,2): row totals4,6,6, column totals4,3,3,3,3 and augmentation rank one. B uses canonical zero/monomial/doubleton cards with column totals3/4, row totals4..6, total terms15..17 and augmentation rank three. CSS weight<=10 and six-round CNOT count25296..26928. Each B support card is deliberate; five canonical relative-ratio contracts diversify A's doubletons. Fresh labels are annealed with helper-derived actions, then screened for connected target-rank components, exact kernel exclusion through six and sampled protection warnings. Common-noise normalized-min-sum stress tests with approximate correlated pairs and independent holdouts select profiles before boundary/cycle tie-breaks. They omit ASC scheduling, measurement histories and relay/LSD and are not circuit pL. Complete verified matrices enforce k=72, CSS and quantum logical exclusion through four. Target the parent's rare collective failures and late cascade resolutions while preserving k-kb=4; do not reward distance above a sampled warning floor.")
FAMILY_NAME = "r010_m17r1s1_35x35_evenAcolumn_holdoutstress_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same metacyclic rule; factor 34 requires q<=17.
    m, r, s = 17, 1, 1
    if q != m * r:
        raise ValueError("this metacyclic bed requires order 17")
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1] % r, m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _shape_cards():
    """Canonical B count architectures, not row/column orbit copies.

    A column mask identifies its doubled cells. Mask seven is excluded,
    keeping column totals <=5. Row totals remain <=8. Canonicalization
    here concerns supports only, not complete Tanner-isomorphism testing.
    """
    from itertools import combinations_with_replacement, permutations

    row_permutations = list(permutations(range(3)))
    cards = set()
    for pattern in combinations_with_replacement(range(7), 5):
        extras = [sum((mask >> i) & 1 for mask in pattern)
                  for i in range(3)]
        if max(extras) > 3:
            continue
        augmentation = [
            sum((1 ^ ((mask >> i) & 1)) << j
                for j, mask in enumerate(pattern))
            for i in range(3)
        ]
        if gf2_rank(augmentation) != 3:
            continue
        canonical = min(
            tuple(sorted(
                sum(((mask >> p[i]) & 1) << i for i in range(3))
                for mask in pattern))
            for p in row_permutations
        )
        cards.add(canonical)

    def priority(pattern):
        extras = [sum((mask >> i) & 1 for mask in pattern)
                  for i in range(3)]
        return sum(extras), max(extras), tuple(sorted(extras)), pattern

    book = sorted(cards, key=priority)
    assert len(book) >= 20
    return book


def NEW_CTX(rng, G, idx):
    """Even-A-column contracts and held-out component decoder selection."""
    import math
    from itertools import combinations_with_replacement, permutations

    q = G.q
    assert q == 17
    component_n = 5 * q
    block_mask = (1 << q) - 1
    row_permutations = tuple(permutations(range(3)))

    # Canonical count cards, not row/column images of saved fillings.
    # Unlike the parent, both sides may contain zero cells.
    types = [
        (a, b, c)
        for a in range(3) for b in range(3) for c in range(3)
        if a + b + c in (3, 4)
        and sum(t > 0 for t in (a, b, c)) >= 2
    ]
    cards = set()
    for columns_card in combinations_with_replacement(types, 5):
        totals = [sum(col[i] for col in columns_card) for i in range(3)]
        if not all(4 <= t <= 6 for t in totals):
            continue
        if not 15 <= sum(totals) <= 17 or 5 not in totals:
            continue
        parity = [
            sum((col[i] & 1) << j for j, col in enumerate(columns_card))
            for i in range(3)
        ]
        if gf2_rank(parity) != 3:
            continue
        canonical = min(
            tuple(sorted(tuple(col[i] for i in p)
                         for col in columns_card))
            for p in row_permutations
        )
        cards.add(canonical)

    def priority(card):
        totals = [sum(col[i] for col in card) for i in range(3)]
        return (sum(totals), max(totals) - min(totals),
                sum(t == 0 for col in card for t in col), card)

    book = sorted(cards, key=priority)
    assert len(book) >= max(20, N_SLOTS)
    design = idx
    pattern = book[design]
    ratio = (1, 2, 3, 4, 5)[idx % 5]
    sa = [[0, 1, 1, 1, 1],
          [2, 1, 1, 1, 1],
          [2, 1, 1, 1, 1]]
    sb = [[col[i] for col in pattern] for i in range(3)]
    assert _grid_weights(sa, sb) <= min(10, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                     for row in sa]) == 1
    assert gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                     for row in sb]) == 3

    def draw_cell(side, i, j, count):
        if side == "A" and j == 0 and count:
            # The two undirected differences have ratio in one of
            # five inversion/reversal classes over C17.
            step = 1 if i == 1 else ratio
            origin = G.pick(rng)
            return [origin, G.mul(origin, (step, 0))]
        return rng.sample(G.elements, count)

    def grid(support, side):
        return [[draw_cell(side, i, j, count)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    # Cache only the singleton actions returned by the verified helper.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz)

    def component(cells, side):
        rows = []
        for cell_row in cells:
            out = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    for r, word in enumerate(actions[side][G.idx(g)]):
                        out[r] ^= word << (j * q)
            rows.extend(out)
        return rows

    def clone(cells):
        return [[list(cell) for cell in row] for row in cells]

    def components(hx, hz):
        """Project the new components from the verified full lift."""
        ar = []
        for i in range(3):
            for r in range(q):
                row = hx[(i * 5) * q + r]
                packed = 0
                for j in range(5):
                    packed |= ((row >> (j * 5 * q)) &
                               block_mask) << (j * q)
                ar.append(packed)
        br = [hz[a * q + r] & ((1 << component_n) - 1)
              for a in range(3) for r in range(q)]
        return ar, br

    def columns(rows, ncols):
        result = [0] * ncols
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def connected(cols, nchecks):
        parent = list(range(nchecks))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for syndrome in cols:
            if not syndrome:
                return False
            first = syndrome & -syndrome
            a = first.bit_length() - 1
            syndrome ^= first
            while syndrome:
                bit = syndrome & -syndrome
                b = bit.bit_length() - 1
                parent[root(b)] = root(a)
                syndrome ^= bit
        return len({root(r) for r in range(nchecks)}) == 1

    def four(rows):
        return sum(v * (v - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for v in [(row & other).bit_count()])

    def six(rows):
        """Exact Tanner six-cycles, including repeated intersections."""
        total = 0
        nr = len(rows)
        overlap = [[0] * nr for _ in range(nr)]
        for i in range(nr):
            for j in range(i):
                overlap[i][j] = overlap[j][i] = (
                    rows[i] & rows[j]).bit_count()
        for i in range(nr):
            for j in range(i):
                x = overlap[i][j]
                if not x:
                    continue
                common = rows[i] & rows[j]
                for h in range(j):
                    y, z = overlap[j][h], overlap[i][h]
                    if y and z:
                        triple = (common & rows[h]).bit_count()
                        total += (x * y * z -
                                  triple * (x + y + z) + 2 * triple)
        return total

    def boundary_screen(cols):
        """Exact kernel-word<=6 exclusion; triple-boundary heuristic."""
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

    def sampled_kernel(cols, side):
        """Actual witnesses; the sampled minimum is only an upper bound."""
        srng = random.Random(170160091 + 1009 * idx + int(side == "B"))
        order = list(range(len(cols)))
        best = len(cols) + 1
        short = set()
        for _ in range(48):
            srng.shuffle(order)
            pivots, basis = {}, []
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
                    basis.append(word)
            for i, word in enumerate(basis):
                candidates = [word]
                candidates.extend(word ^ other for other in basis[:i])
                for v in candidates:
                    if v:
                        weight = v.bit_count()
                        best = min(best, weight)
                        if weight <= 12:
                            short.add(v)
            if best <= 6:
                break
        loads = [0] * 5
        for word in short:
            for j in range(5):
                loads[j] += bool((word >> (j * q)) & block_mask)
        return best, len(short), max(loads)

    def decoder_stress(rows, side, shots, stream):
        """Common-noise classical stress, NOT ASC or relay-BP fitness.

        Decode a transmitted zero component word. Nonconvergence and
        wrong valid codewords both fail. Shared-check pairs approximate
        correlated exposure; they are not actual scheduled ASC hooks.
        """
        import numpy as np

        nr = len(rows)
        ne = sum(row.bit_count() for row in rows)
        maximum = max(row.bit_count() for row in rows)
        check_edges = np.full((nr, maximum), ne, dtype=np.int32)
        variable_lists = [[] for _ in range(component_n)]
        edge_variables, locations, neighbors = [], [], []
        matrix = np.zeros((nr, component_n), dtype=np.uint8)
        for r, row in enumerate(rows):
            ns = []
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                edge = len(edge_variables)
                check_edges[r, len(ns)] = edge
                locations.append(r * maximum + len(ns))
                variable_lists[c].append(edge)
                edge_variables.append(c)
                matrix[r, c] = 1
                ns.append(c)
                row ^= bit
            neighbors.append(ns)

        degree = np.asarray(list(map(len, variable_lists)), dtype=float)
        assert np.all(degree >= 3)
        variables = np.full(
            (component_n, int(degree.max())), ne, dtype=np.int32)
        for c, edges in enumerate(variable_lists):
            variables[c, :len(edges)] = edges
        edge_variables = np.asarray(edge_variables, dtype=np.int32)
        locations = np.asarray(locations, dtype=np.int32)
        positions = np.arange(maximum)[None, None, :]
        seed = (170160301 + 1009 * idx + int(side == "B")
                + 1000003 * stream)
        channel = np.random.default_rng(seed)
        uniforms = channel.random((shots, component_n))
        pair_uniforms = channel.random((shots, nr))
        pair_positions = channel.integers(
            0, 1000000, size=(shots, nr, 2))
        outcomes = []

        for probability in (0.025, 0.045):
            # Both variable degree and check length affect this
            # deliberately uncalibrated gate-burden approximation.
            rates = probability * degree / 3.0
            rates *= ne / (nr * 5.0)
            errors = uniforms < rates[None, :]
            for shot, r in zip(*np.nonzero(pair_uniforms < 0.003)):
                ns = neighbors[int(r)]
                a = int(pair_positions[shot, r, 0]) % len(ns)
                b = int(pair_positions[shot, r, 1]) % (len(ns) - 1)
                b += b >= a
                errors[shot, ns[a]] ^= True
                errors[shot, ns[b]] ^= True

            magnitude = np.log((1.0 - rates) / rates)
            prior = np.where(errors, -magnitude[None, :],
                             magnitude[None, :])
            messages = prior[:, edge_variables].copy()
            done = np.zeros(shots, dtype=bool)
            failed = np.zeros(shots, dtype=bool)
            effort = 0
            for iteration in range(1, 37):
                padded = np.concatenate(
                    (messages, np.full((shots, 1), 1.0e6)), axis=1)
                incoming = padded[:, check_edges]
                absolute = np.abs(incoming)
                where = absolute.argmin(axis=2)
                minimum = absolute.min(axis=2)
                second = np.partition(absolute, 1, axis=2)[:, :, 1]
                signs = np.where(incoming < 0.0, -1.0, 1.0)
                excluded = np.where(
                    positions == where[:, :, None],
                    second[:, :, None], minimum[:, :, None])
                outgoing = (0.8 * signs.prod(axis=2)[:, :, None]
                            * signs * excluded)
                checks = outgoing.reshape(shots, -1)[:, locations]
                padded_checks = np.concatenate(
                    (checks, np.zeros((shots, 1))), axis=1)
                posterior = prior + padded_checks[:, variables].sum(axis=2)
                proposal = posterior[:, edge_variables] - checks
                messages = np.clip(
                    0.3 * messages + 0.7 * proposal, -30.0, 30.0)
                if iteration % 4:
                    continue
                hard = (posterior < 0.0).astype(np.uint8)
                syndrome = (hard @ matrix.T) & 1
                newly = (~done) & ~syndrome.any(axis=1)
                failed[newly] = hard[newly].any(axis=1)
                effort += iteration * int(newly.sum())
                done |= newly
                if done.all():
                    break
            unresolved = int((~done).sum())
            outcomes.append((int((failed | ~done).sum()),
                             unresolved, effort + 37 * unresolved))
        return (outcomes[0][0], outcomes[1][0],
                outcomes[0][1] + outcomes[1][1],
                outcomes[0][2] + outcomes[1][2])

    def short_logical(checks, stabilizers, n):
        """Exact non-stabilizer logical existence through weight four."""
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

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    def cycle_cost(rows):
        """C4 and a cheap C6 proposal statistic from verified actions."""
        nr = len(rows)
        adjacency = [0] * nr
        c4 = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                c4 += overlap * (overlap - 1) // 2
                if overlap:
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
                    for d in (c.bit_count()
                              for c in columns(rows, component_n)))
        return c4, max(0, triangles - stars)

    def portfolio(side, support):
        wanted_rank = 49 if side == "A" else 51
        sites = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
        pool, seen = [], set()
        serial = 0

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
            if gf2_rank(rows) != wanted_rank:
                return
            serial += 1
            pool.append((cost, serial, clone(cells), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[10:]

        def energy(cost):
            return (100.0 * cost[0] + cost[1] / 6.0) / q

        for restart in range(4):
            cells = grid(support, side)
            rows = component(cells, side)
            current = cycle_cost(rows)
            remember(cells, rows, current)
            for step in range(260):
                i, j = rng.choice(sites)
                old = cells[i][j]
                cells[i][j] = draw_cell(side, i, j, support[i][j])
                proposed_rows = component(cells, side)
                proposed = cycle_cost(proposed_rows)
                delta = energy(proposed) - energy(current)
                temperature = 12.0 * (0.985 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    rows, current = proposed_rows, proposed
                    if delta < 0 or step % 12 == 0:
                        remember(cells, rows, current)
                else:
                    cells[i][j] = old
            remember(cells, rows, current)

        finalists = []
        for coarse, number, cells, rows in pool:
            cols = columns(rows, component_n)
            if not connected(cols, 3 * q):
                continue
            boundary = boundary_screen(cols)
            if boundary is None:
                continue
            distance, multiplicity, peak = sampled_kernel(cols, side)
            stress = decoder_stress(rows, side, 96, 0)
            exact_six = six(rows)
            key = ((int(distance < 12),) + stress
                   + (boundary, multiplicity, peak,
                      coarse[0], exact_six, -min(distance, 14)))
            finalists.append((key, number, cells, rows))
        finalists.sort(key=lambda item: item[:2])

        # Held-out streams are independent of the shortlist channel,
        # but common across finalists within this side and slot.
        heldout = []
        for key, number, cells, rows in finalists[:3]:
            independent = decoder_stress(rows, side, 192, 1)
            combined = tuple(key[1 + t] + independent[t]
                             for t in range(4))
            heldout.append((key[:1] + combined + key[5:],
                            number, cells, rows))
        heldout.sort(key=lambda item: item[:2])
        return heldout

    for batch in range(3):
        apool = portfolio("A", sa)
        bpool = portfolio("B", sb)
        options = []
        for ak, at, A, expected_a in apool:
            for bk, bt, B, expected_b in bpool:
                # Warning floor first; then worse-component stress.
                paired = tuple(
                    value for a, b in zip(ak, bk)
                    for value in (max(a, b), a + b))
                options.append((paired, at, bt, A, B,
                                expected_a, expected_b))
        options.sort(key=lambda item: item[:3])

        for key, at, bt, A, B, expected_a, expected_b in options:
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = components(hx, hz)
            assert ar == expected_a and br == expected_b
            # Component ranks motivate the desired dimension; complete
            # evaluator-convention ranks remain the acceptance test.
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 578 or k != 72 or 8 * k >= n:
                continue
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(10, W_MAX)
            assert gf2_rank(ar) == 49 and gf2_rank(br) == 51
            cnots = 6 * sum(row.bit_count() for row in hx + hz)
            expected = 6 * q * 8 * (
                sum(map(sum, sa)) + sum(map(sum, sb)))
            assert cnots == expected and 25296 <= cnots <= 26928
            assert all(len(cell) == len({G._key(g) for g in cell})
                       for cells in (A, B) for row in cells for cell in row)
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": design, "ratio": ratio,
                    "support_card": pattern, "k": k,
                    "screen": key, "cnots": cnots}

    raise ValueError("no even-A-column k=72 filling passed held-out screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete held-out-screened cell, including prescribed zeros."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
# REFUTED: k=72, check weight nine and d2s=12 guarantee zero-failure C17 memory: current slot1 had one Z and two X failures in 200000 shots per basis, including 62 collective X logical flips, while slot0 with those same reported parameters had no failures.
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
