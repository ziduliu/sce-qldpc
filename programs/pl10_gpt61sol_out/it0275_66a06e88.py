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
CONCEPT = ("For S3 direct-product C5, order 30, shapes (1,3)x(2,5), noncommuting A trinomials with relative orders ten/fifteen and degree-three B columns should raise worse-basis circuit fitness beyond 5.0888 in cell 0.350 while reducing extraction exposure.")
GROUP = ("Jump from cyclic C13 to S3xC5. The direct C5 factor allows noncommuting triangles whose pairwise relative elements avoid orders two and three. BEDS contains this centre-five order-30 group at other shapes, but not (1,3)x(2,5); those readings establish neither success nor failure here. The fixed order gives n=510.")
SHAPES = ("Target cell 0.350 with (1,3)x(2,5): n=17*30=510, kb=30*2*3=180, guaranteed rate 6/17=0.352941. A has 30 checks on 90 variables; B has 60 checks on 150 variables. Require actual 180<=k<=191. Parent weight-eight witnesses and collective failures spanned many Acol/Bcol blocks; Acol2 and Zchk:Acol2/Brow2 were mildly implicated, but 1.1-1.2x enrichment did not identify a unique defective block.")
FILLING = ("Twenty contracts cross four A central-triangle architectures with five B degree-three support profiles. A contains three noncommuting trinomials; B rows total seven/eight and every B column totals three. Thus HX<=12, HZ<=11 and six-round extraction uses 22140 CNOTs, versus the parent's 27690/28548. A cards vary how many reflection terms are central-coordinate midpoints of the rotation pair. Additional batches restrict B doubleton orders rather than relabel saved codes. Cache only helper-derived singleton actions, verify final component equality, screen short kernels and compare common-pattern component BP burden symmetrically. Full helper-built CSS ranks retain the target rate. The objective is fewer circuit failures; this BP benchmark omits ASC scheduling and relay/LSD decoding.")
FAMILY_NAME = "r035_S3xC5_q30_13x25_longrelative_jointburden_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 30

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 direct-product C5 requires order 30")
    # x^3=y^2=z^5=1; y*x*y^-1=x^-1; z is central.
    # Elements are x^a y^b z^c.
    els = [(a, b, c) for c in range(5)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 5)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        sign = -1 if b else 1
        return ((a + sign*d) % 3, (b+e) % 2, (c+f) % 5)

    def inv(g):
        a, b, c = red(g)
        sign = -1 if b else 1
        return ((-sign*a) % 3, b, (-c) % 5)

    def idx(g):
        a, b, c = red(g)
        return a + 3*b + 6*c

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty noncommuting count/central-geometry contracts on S3xC5."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    assert q == 30
    design, batch = idx % 20, idx // 20
    architecture, b_model = divmod(design, 5)

    top = (
        (1, 1, 1, 2, 2),
        (0, 1, 2, 2, 2),
        (3, 0, 1, 1, 2),
        (3, 1, 1, 1, 1),
        (3, 0, 0, 2, 2),
    )[b_model]
    sb = [list(top), [3-t for t in top]]
    counts = {"A": [[3, 3, 3]], "B": sb}
    assert tuple(map(sum, sb)) == (7, 8)
    assert all(sum(column) == 3 for column in zip(*sb))
    assert _grid_weights(counts["A"], sb) == 12
    assert 9 + max(map(sum, zip(*sb))) <= W_MAX
    assert max(map(sum, sb)) + 3 <= W_MAX
    assert _no_dead_lines(counts["A"], sb)

    # Marker three makes the reflection coordinate the midpoint of
    # the central coordinates zero and d: 2*(3*d)=d modulo five.
    # Marker two does not. Cards vary the number of midpoint triangles,
    # not just their column order or a common automorphism.
    a_markers = (
        (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)
    )[architecture]

    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if power == G.e:
                orders[G.idx(g)] = order
                break
        assert G.idx(g) in orders
    b_orders = ((5, 10, 15), (10,), (15,), (5,))[batch % 4]
    b_relatives = [
        g for g in G.elements if orders[G.idx(g)] in b_orders
    ]
    assert b_relatives

    def triangle(marker):
        d = rng.randrange(1, 5)
        u = (rng.choice((1, 2)), 0, d)
        v = (rng.randrange(3), 1, marker*d % 5)
        relative = [G.e, u, v]
        assert G.mul(u, v) != G.mul(v, u)
        assert all(
            orders[G.idx(G.mul(G.inv(relative[a]), relative[b]))]
            in (10, 15)
            for a in range(3) for b in range(a)
        )
        anchor = G.pick(rng)
        return [G.mul(anchor, h) for h in relative]

    def draw_cell(side, i, j):
        t = counts[side][i][j]
        if t == 0:
            return []
        if t == 1:
            return [G.pick(rng)]
        if t == 2:
            anchor = G.pick(rng)
            return [anchor, G.mul(anchor, rng.choice(b_relatives))]
        assert t == 3
        marker = (a_markers[j] if side == "A"
                  else 2 + ((i+j+batch) % 2))
        return triangle(marker)

    def ratio_contract(grid):
        # draw_cell preserves every prescribed triangle architecture.
        return True

    def draw(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def verified_component(side, grid):
        # Zero probes isolate precisely the scaffold's left/right action.
        rows = (lifted_matrices(G, grid, [[[]]])[0]
                if side == "A" else
                lifted_matrices(G, [[[]]], grid)[1])
        width = len(grid[0]) * q
        keep = (1 << width) - 1
        return [row & keep for row in rows]

    def transpose(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length()-1] ^= 1 << r
                row ^= bit
        return columns

    # Cache only singleton matrices returned by the verified helper.
    # Linear composition is checked against complete components at emit.
    actions = {"A": {}, "B": {}}
    for side in actions:
        for g in G.elements:
            actions[side][G.idx(g)] = tuple(
                transpose(verified_component(side, [[[g]]]), q)
            )

    def component_columns(side, grid):
        columns = [0] * (len(grid[0]) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, value in enumerate(block):
                        columns[j*q+t] ^= value << (i*q)
        expected = [sum(row[j] for row in counts[side])
                    for j in range(len(grid[0]))]
        assert all(value.bit_count() == expected[j // q]
                   for j, value in enumerate(columns))
        return columns

    def cycle_data(cols):
        """Exact four-cycle count for either component and mixed degrees."""
        pairs = {}
        adj = [0] * max(c.bit_length() for c in cols)
        conflicts = 0
        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a, u in enumerate(nodes):
                for v in nodes[a + 1:]:
                    key = (1 << u) | (1 << v)
                    old = pairs.get(key, 0)
                    conflicts += old
                    pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return conflicts, adj

    def kernel_metric(cols, adj):
        """Short syndrome ambiguities, including odd-weight B witnesses.

        Counts are collision proxies if shorter words exist. They are
        not quantum distances and do not distinguish stabilizers.
        """
        singles = {}
        for syn in cols:
            singles[syn] = singles.get(syn, 0) + 1
        n2 = sum(v * (v - 1) // 2 for v in singles.values())
        pair_hist = {}
        n3 = n4 = 0
        nc = len(cols)
        for a in range(nc - 1):
            for b in range(a + 1, nc):
                syn = cols[a] ^ cols[b]
                n3 += singles.get(syn, 0)
                old = pair_hist.get(syn, 0)
                n4 += old
                pair_hist[syn] = old + 1

        triangles = 0
        for u in range(len(adj)):
            bits = adj[u] & ~((1 << (u + 1)) - 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() - 1
                triangles += (
                    adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                ).bit_count()
                bits ^= bit
        stars = sum(c.bit_count() * (c.bit_count() - 1)
                    * (c.bit_count() - 2) // 6 for c in cols)
        # With C4=0 this is exact. Reinforced components may have
        # unavoidable C4, so otherwise retain only a triangle proxy.
        n6cycles = max(0, triangles - stars)
        if n2 or n3 or n4:
            return (n2, n3, n4, 10**9, 10**9, n6cycles)

        triple_hist = {}
        n5 = n6 = 0
        for a in range(nc - 2):
            for b in range(a + 1, nc - 1):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc):
                    syn = ab ^ cols[c]
                    n5 += pair_hist.get(syn, 0)
                    old = triple_hist.get(syn, 0)
                    n6 += old
                    triple_hist[syn] = old + 1
        return (0, 0, 0, n5, n6, n6cycles)

    def rows_from_columns(cols):
        """Transpose the columns extracted from the verified lift."""
        rows = [0] * max(c.bit_length() for c in cols)
        for j, col in enumerate(cols):
            while col:
                bit = col & -col
                rows[bit.bit_length() - 1] |= 1 << j
                col ^= bit
        return rows

    def kernel_probe(cols, trials):
        """Random systematic bases give verified component witnesses only."""
        source = rows_from_columns(cols)
        nc = len(cols)
        minimum = nc + 1
        small = set()
        for _ in range(trials):
            rows = list(source)
            order = list(range(nc))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                hit = next((t for t in range(rank, len(rows))
                            if rows[t] & bit), None)
                if hit is None:
                    continue
                rows[rank], rows[hit] = rows[hit], rows[rank]
                pivot = rows[rank]
                for t in range(len(rows)):
                    if t != rank and rows[t] & bit:
                        rows[t] ^= pivot
                pivots.append(c)
                rank += 1
                if rank == len(rows):
                    break
            occupied = set(pivots)
            for c in order:
                if c in occupied:
                    continue
                word = 1 << c
                for p, row in zip(pivots, rows[:rank]):
                    if (row >> c) & 1:
                        word |= 1 << p
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    small.add(word)
            if minimum <= 6:
                break
        return minimum, len(small)

    def signature(grid):
        """Remove component column gauges using the actual group law."""
        columns = []
        for j in range(len(grid[0])):
            anchors = [g for row in grid for g in row[j]]
            columns.append(min(
                tuple(tuple(sorted(
                    G.idx(G.mul(g, G.inv(anchor))) for g in row[j]
                )) for row in grid)
                for anchor in anchors
            ))
        return tuple(sorted(columns))

    def bp_burden(cols, side):
        """Classical component benchmark, NOT the circuit evaluator.

        Stop each pattern at its first valid syndrome solution. A
        valid-but-wrong solution is a component kernel miscorrection;
        unresolved patterns represent extra decoder burden. Neither
        quantity is asserted to be the quantum circuit failure rate.
        """
        import numpy as np

        nc = len(cols)
        nr = max(c.bit_length() for c in cols)
        ev = []
        checks = [[] for _ in range(nr)]
        variables = [[] for _ in range(nc)]
        for v, column in enumerate(cols):
            while column:
                bit = column & -column
                c = bit.bit_length() - 1
                edge = len(ev)
                ev.append(v)
                checks[c].append(edge)
                variables[v].append(edge)
                column ^= bit
        ne = len(ev)
        edge_vars = np.asarray(ev, dtype=np.int64)
        ce = np.full((nr, max(map(len, checks))), ne, dtype=np.int64)
        ve = np.full((nc, max(map(len, variables))), ne, dtype=np.int64)
        for c, edges in enumerate(checks):
            ce[c, :len(edges)] = edges
        for v, edges in enumerate(variables):
            ve[v, :len(edges)] = edges
        real = ce != ne
        check_vars = np.asarray(ev + [nc], dtype=np.int64)[ce]

        # Same pattern bank for every finalist of this side and slot.
        # Sampling weights include the component's degree exposure.
        srng = np.random.default_rng(
            317031 + 1009 * idx + 7919 * int(side == "B"))
        degrees = np.asarray([c.bit_count() for c in cols], dtype=float)
        probabilities = degrees / degrees.sum()
        weights = (3, 4, 5) if side == "A" else (4, 5, 6)
        shots = 128 * len(weights)
        errors = np.zeros((shots, nc + 1), dtype=bool)
        for s in range(shots):
            weight = weights[s // 128]
            selected = srng.choice(nc, weight, replace=False,
                                   p=probabilities)
            errors[s, selected] = True
        syndrome = np.logical_xor.reduce(errors[:, check_vars], axis=2)
        signs = 1.0 - 2.0 * syndrome.astype(float)
        p = np.clip((sum(weights) / len(weights) / nc)
                    * degrees / degrees.mean(), 0.02, 0.16)
        prior = np.log((1.0 - p) / p)
        v2c = np.empty((shots, ne + 1), dtype=float)
        v2c[:, :ne] = prior[edge_vars]
        v2c[:, ne] = 30.0
        c2v = np.zeros_like(v2c)
        active = np.ones(shots, dtype=bool)
        wrong = 0
        early = None
        ones = np.ones((shots, nr, 1), dtype=float)

        for iteration in range(36):
            t = np.tanh(np.clip(v2c[:, ce], -18.0, 18.0) / 2.0)
            t[:, ~real] = 1.0
            left = np.concatenate(
                (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            product = signs[:, :, None] * left * right
            outgoing = 2.0 * np.arctanh(
                np.clip(product, -0.999999, 0.999999))
            c2v[:, ce[real]] = outgoing[:, real]
            posterior = prior[None, :] + c2v[:, ve].sum(axis=2)
            estimate = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)),
                axis=1)
            matched = np.logical_xor.reduce(
                estimate[:, check_vars], axis=2)
            settled = active & np.all(matched == syndrome, axis=1)
            incorrect = np.any(estimate[:, :nc] != errors[:, :nc], axis=1)
            wrong += int(np.count_nonzero(settled & incorrect))
            active[settled] = False
            if iteration == 11:
                early = int(active.sum())
            if not active.any():
                if early is None:
                    early = 0
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, edge_vars] - c2v[:, :ne]
            v2c[:, :ne] = (damping * v2c[:, :ne]
                           + (1.0 - damping) * proposal)

        remaining = int(active.sum())
        if early is None:
            early = remaining
        risk = (3.0 * wrong + remaining + early / 16.0) / shots
        return risk, (wrong, remaining, early, shots)

    def unavoidable_four_cycles(side):
        """General check-pair packing bound, with no cyclic-orbit assumption."""
        grid = counts[side]
        degrees = [sum(row[j] for row in grid)
                   for j in range(len(grid[0]))]
        total = q * sum(d * (d-1) // 2 for d in degrees)
        checks = len(grid) * q
        bins = checks * (checks-1) // 2
        a, b = divmod(total, bins)
        return bins * a * (a-1) // 2 + b * a

    def optimize(side):
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        number_rows = len(counts[side])
        augmentation = [
            sum((t % 2) << j for j, t in enumerate(row))
            for row in counts[side]
        ]
        allowed_deficit = number_rows - gf2_rank(augmentation)
        cycle_floor = unavoidable_four_cycles(side)

        def rank_penalty(columns):
            deficit = number_rows * q - gf2_rank(
                rows_from_columns(columns))
            return max(0, deficit - allowed_deficit)

        current = draw(side)
        cols = component_columns(side, current)
        cost, adj = cycle_data(cols)
        fallback, fallback_cost = current, cost
        shortlist = []
        tested = set()

        def retain(grid, columns, cycles, adjacency):
            sig = signature(grid)
            if sig in tested:
                return
            tested.add(sig)
            metric = ((rank_penalty(columns),)
                      + kernel_metric(columns, adjacency))
            shortlist.append((metric, sig, grid, columns, cycles))
            shortlist.sort(key=lambda item: (
                item[0][:-1], item[4], item[0][-1], item[1]))
            del shortlist[4:]

        for step in range(1600):
            if cost <= cycle_floor + 2 * q:
                retain(current, cols, cost, adj)
                if len(tested) >= 20:
                    break

            i, j = sites[rng.randrange(len(sites))]
            trial = [list(row) for row in current]
            trial[i][j] = draw_cell(side, i, j)
            if side == "A" and not ratio_contract(trial):
                continue
            trial_cols = component_columns(side, trial)
            trial_cost, trial_adj = cycle_data(trial_cols)
            if trial_cost < fallback_cost:
                fallback, fallback_cost = trial, trial_cost
            if trial_cost <= cost or rng.random() < 0.02:
                current, cols = trial, trial_cols
                cost, adj = trial_cost, trial_adj
            if step % 240 == 239 and cost > cycle_floor:
                current = draw(side)
                cols = component_columns(side, current)
                cost, adj = cycle_data(cols)
                if cost < fallback_cost:
                    fallback, fallback_cost = current, cost

        if not shortlist:
            columns = component_columns(side, fallback)
            cycles, adjacency = cycle_data(columns)
            retain(fallback, columns, cycles, adjacency)

        finalists = []
        for metric, sig, grid, columns, cycles in shortlist:
            risk, bp = bp_burden(columns, side)
            distance_upper, multiplicity = kernel_probe(columns, 24)
            # Exact short-kernel evidence precedes the decoder surrogate.
            # Longer sampled distance is only a capped late tie-breaker.
            short = metric[:4] + (int(metric[4] > 0),
                                  int(metric[5] > 0))
            key = (short, risk, cycles, metric[-1],
                   -min(distance_upper, 10), multiplicity, sig)
            finalists.append({
                "key": key, "grid": grid, "short": short,
                "risk": risk, "bp": bp, "c4": cycles,
                "d": distance_upper,
            })
        finalists.sort(key=lambda item: item["key"])
        return finalists[:4]

    def full_four_cycles(rows):
        return sum(t * (t - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for t in [(row & other).bit_count()])

    # Pair independently screened components. No submitted illegal slot
    # is repaired, and component ranks never substitute for CSS ranks.
    for attempt in range(3):
        apool, bpool = optimize("A"), optimize("B")
        best = None
        for a in apool:
            for b in bpool:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (180 <= k <= 191):
                    continue
                xc4, zc4 = full_four_cycles(hx), full_four_cycles(hz)
                mixed_x = xc4 - cb * a["c4"] - ra * b["c4"]
                mixed_z = zc4 - rb * a["c4"] - ca * b["c4"]
                assert mixed_x >= 0 and mixed_z >= 0
                mx, mz = mixed_x / len(hx), mixed_z / len(hz)
                short = tuple(max(x, y) for x, y in
                              zip(a["short"], b["short"]))
                # The new bed has no measured basis attribution yet.
                # Balance both component benchmarks instead of carrying
                # the cyclic parent's A-first weighting across the jump.
                key = short + (
                    max(a["risk"], b["risk"])
                    + 0.015 * max(mx, mz),
                    a["risk"] + b["risk"]
                    + 0.01 * (mx + mz),
                    max(xc4 / len(hx), zc4 / len(hz)),
                    xc4 + zc4,
                    -min(a["d"], b["d"], 10),
                    a["key"][-1], b["key"][-1],
                )
                if best is None or key < best[0]:
                    best = (key, A, B, hx, hz, n, k, a, b)

        if best is None:
            continue
        key, A, B, hx, hz, n, k, a, b = best
        if not css_ok(hx, hz):
            raise ValueError("verified CSS commutation failed")
        assert n == 510
        assert 325*n <= 1000*k < 375*n
        assert max(row.bit_count() for row in hx) <= 12
        assert max(row.bit_count() for row in hz) <= 11
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        for side, grid in (("A", A), ("B", B)):
            assert transpose(
                verified_component(side, grid), len(grid[0])*q
            ) == component_columns(side, grid)
            assert all(len(cell) == len(set(cell))
                       for row in grid for cell in row)
        a_terms = sum(map(sum, counts["A"]))
        b_terms = sum(map(sum, counts["B"]))
        expected_cnots = 6*q*((cb+rb)*a_terms + (ca+ra)*b_terms)
        assert expected_cnots == 22140
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "bp_burden": (a["bp"], b["bp"]),
                "screen": key, "cnots": expected_cnots,
                "contract": (architecture, b_model, batch)}
    raise ValueError("no verified filling in the target rate interval")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases imply balanced memory reliability: current parent slot 20 had 100/8250 Z failures versus 100/50250 X failures, with only diffuse block enrichment.
# REFUTED: Additional B terms universally improve worse-basis memory at d2s=8: Z13 slots 1/5/9 scored 4.851/4.883/4.962, below unreinforced slots 0/8 at 5.202/5.097; labels also changed, so this is not an isolated gate-count comparison.
# REFUTED: The A17/A18 component-BP recipe exceeds the 0.350 frontier: measured fitness was 4.9197, all five measured codes hit failure caps, and 27690/28548-CNOT extraction retained weight-eight witnesses; this does not isolate reinforcement from labels or the BP surrogate.
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
