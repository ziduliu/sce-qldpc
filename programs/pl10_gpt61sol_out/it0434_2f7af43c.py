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
CONCEPT = ("For C5 x C5, order 25, shapes (2,4)x(2,4), noncollinear relative supports and decoder-stress selection at fixed degree-three column protection should improve the cell-0.200 incumbent fitness 7.2133 through weight-nine extraction, not distance as fitness.")
GROUP = ("Jump from cyclic Z17 to noncyclic C5 x C5. Every nonidentity element has order five, whereas cyclic C25 contains order-25 elements, so this is not a relabeling of either group. This order-25 equal-shape bed is absent from BEDS. Related cyclic C25/C27 beds reached about 6.9/7.2 on these shapes; these readings motivate an odd-order low-load experiment without establishing a group advantage.")
SHAPES = ("Equal (2,4) components have 100 variables and 50 checks. Size factor twenty allows q<=29 and this fixed group uses q=25, n=500, kb=100 and guaranteed rate 0.200. Complete verified ranks require 100<=k<=112, entirely inside cell 0.200. Parent failures implicated different Acol/Bcol and Arow/Brow data blocks and Acol/Brow checks; slot53 affected 76 of 84 logicals. Sparse failure maps do not establish a persistent isolated hotspot.")
FILLING = ("Four balanced support pairs receive indexed relative geometries, yielding twenty deliberate contracts and further fresh contracts for larger harness budgets. Every component row has six terms and every column three: both CSS check weights are nine and six-round extraction uses 21600 CNOTs, versus the parent's measured 28560/29376. Trinomials must span both C5 coordinates; internal differences cannot repeat within a component row. Conservatively quotient relative contracts by GL(2,5), independent cell translations and protograph permutations; downstream Tanner equivalence remains authoritative. Anneal fresh origins without changing relative supports. Exact component kernels through six and sampled protection capped at twelve precede common-stream component BP stress, then short-witness and boundary exposure. Complete ranks, CSS, connectivity and logical exclusion through four are verified. Parent slot53 had d2s14 but eight X failures/200000, versus slot8's d2s13 and one X failure, motivating decoder difficulty and lower extraction exposure rather than larger distance upper bounds. The stress test is not ASC scheduling or measured circuit pL.")
FAMILY_NAME = "r020_C5xC5_24x24_affinecontracts_decoderstress_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 25

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 x C5 requires order 25")
    els = [(a, b) for b in range(5) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 5)

    def mul(g, h):
        return ((g[0] + h[0]) % 5, (g[1] + h[1]) % 5)

    def inv(g):
        return ((-g[0]) % 5, (-g[1]) % 5)

    def idx(g):
        a, b = red(g)
        return 5*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Indexed affine-support contracts, fresh origins and verified screening."""
    import math
    from itertools import product

    q = G.q
    mode = (idx // 20) % 3
    assert q == 25 and A_SHAPE == B_SHAPE == (2, 4)
    assert idx >= 0

    profiles = (
        (1, 1, 2, 2),
        (0, 1, 2, 3),
        (0, 2, 2, 2),
    )
    briefs = ((0, 0), (0, 1), (1, 1), (0, 2))
    geometry_number, architecture = divmod(idx, len(briefs))
    pa, pb = briefs[architecture]

    def support(p):
        first = profiles[p]
        return [list(first), [3-w for w in first]]

    sa, sb = support(pa), support(pb)
    supports_by_side = {"A": sa, "B": sb}
    assert _grid_weights(sa, sb) == 9 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) == 6 for grid in (sa, sb) for row in grid)
    assert all(sum(column) == 3
               for grid in (sa, sb) for column in zip(*grid))
    assert all(0 <= w <= CELL_MAX
               for grid in (sa, sb) for row in grid for w in row)
    deficits = {
        side: 2 - gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in grid])
        for side, grid in supports_by_side.items()
    }

    # All 480 linear automorphisms of the elementary abelian group.
    # Canonicalize relative contracts, not copies of selected matrices.
    mappings = []
    for a, b, c, d in product(range(5), repeat=4):
        if (a*d-b*c) % 5 == 0:
            continue
        mappings.append(tuple(
            G.idx(((a*x+b*y) % 5, (c*x+d*y) % 5))
            for x, y in G.elements))
    assert len(mappings) == 480

    def normalized(cell, mapping):
        if not cell:
            return ()
        values = [G.elements[mapping[G.idx(g)]] for g in cell]
        return min(
            tuple(sorted(G.idx(((x-u) % 5, (y-v) % 5))
                         for x, y in values))
            for u, v in values)

    def relative_key(grids):
        images = []
        for mapping in mappings:
            sides = []
            for grid in grids:
                cols = [
                    tuple(normalized(grid[i][j], mapping)
                          for i in range(2))
                    for j in range(4)
                ]
                direct = tuple(sorted(cols))
                swapped = tuple(sorted((b, a) for a, b in cols))
                sides.append(min(direct, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    book_rng = random.Random(250024101 + 7919*architecture)

    def relative_grid(counts):
        grid = []
        for row_counts in counts:
            row, used = [], set()
            for weight in row_counts:
                if weight == 0:
                    cell = []
                elif weight == 1:
                    cell = [G.e]
                else:
                    cell = [G.e] + book_rng.sample(
                        [g for g in G.elements if g != G.e], weight-1)
                    if weight == 3:
                        u, v = cell[1:]
                        if (u[0]*v[1]-u[1]*v[0]) % 5 == 0:
                            return None
                local = set()
                for i, g in enumerate(cell):
                    for h in cell[:i]:
                        difference = G.mul(g, G.inv(h))
                        direction = min(
                            G.idx(difference), G.idx(G.inv(difference)))
                        if direction in used or direction in local:
                            return None
                        local.add(direction)
                used.update(local)
                row.append(cell)
            grid.append(row)
        return grid

    seen_contracts = set()
    relative_geometry = contract_key = None
    number = 0
    for proposal in range(30000):
        grids = (relative_grid(sa), relative_grid(sb))
        if any(grid is None for grid in grids):
            continue
        key = relative_key(grids)
        if key in seen_contracts:
            continue
        seen_contracts.add(key)
        if number == geometry_number:
            relative_geometry, contract_key = grids, key
            break
        number += 1
    if relative_geometry is None:
        raise ValueError("Unused affine relative-contract book exhausted")
    relative_by_side = {
        "A": relative_geometry[0], "B": relative_geometry[1]}

    def draw_cell(side, i, j):
        relative = relative_by_side[side][i][j]
        origin = G.pick(rng)
        return [G.mul(origin, g) for g in relative]

    def draw_grid(side):
        return [
            [draw_cell(side, i, j) for j in range(4)]
            for i in range(2)
        ]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        # Use the verified construction with an identity companion.
        # The first sector is precisely the required component matrix.
        identity = [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, identity)
        else:
            _, rows, _ = lifted_matrices(G, identity, grid)
        nc = len(grid[0]) * q
        mask = (1 << nc) - 1
        return [row & mask for row in rows], nc

    def columns(rows, nc):
        out = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def connected(rows, nc):
        parent = list(range(nc))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            first = (row & -row).bit_length()-1
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length()-1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(c) for c in range(nc)}) == 1

    def decoder_stress(rows, cols, side):
        """Common-stream component BP difficulty, NOT circuit fitness.

        Perfect syndromes omit extraction faults, measurement histories
        and the evaluator's relay/LSD cascade. A settled wrong kernel
        word and an unresolved pattern are recorded separately.
        """
        import numpy as np

        nc, nr = len(cols), len(rows)
        neighbors = []
        for row in rows:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length()-1)
                row ^= bit
            neighbors.append(sites)
        assert all(len(sites) == 6 for sites in neighbors)
        incidence = np.asarray(neighbors, dtype=np.int32)
        flat = incidence.ravel()
        edges = len(flat)
        degree = np.bincount(flat, minlength=nc)
        assert np.all(degree == 3)
        variable_edges = np.empty((nc, 3), dtype=np.int32)
        used = np.zeros(nc, dtype=np.int32)
        for edge, c in enumerate(flat):
            variable_edges[c, used[c]] = edge
            used[c] += 1

        shots = 96
        channel = np.random.default_rng(
            250024733 + 1009*idx + 7919*int(side == "B"))
        errors = channel.random((shots, nc)) < 0.04
        for shot in range(shots//2, shots):
            errors[shot] = False
            sites = channel.choice(nc, 3+shot % 4, replace=False)
            errors[shot, sites] = True
        target = (
            errors[:, flat].reshape(shots, nr, 6).sum(axis=2) & 1
        ).astype(bool)
        signs = np.where(target, -1.0, 1.0)
        prior = math.log(0.96/0.04)
        messages = np.full((shots, edges), prior)
        active = np.ones(shots, dtype=bool)
        ones = np.ones((shots, nr, 1))
        wrong = work = 0
        early = None

        for iteration in range(36):
            work += int(active.sum())
            t = np.tanh(np.clip(messages, -18, 18).reshape(
                shots, nr, 6)/2)
            left = np.concatenate(
                (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            outgoing = 2*np.arctanh(np.clip(
                signs[:, :, None]*left*right,
                -0.999999, 0.999999)).reshape(shots, edges)
            posterior = prior + outgoing[:, variable_edges].sum(axis=2)
            hard = posterior < 0
            observed = (
                hard[:, flat].reshape(shots, nr, 6).sum(axis=2) & 1
            ).astype(bool)
            settled = active & np.all(observed == target, axis=1)
            wrong += int(np.count_nonzero(
                settled & np.any(hard != errors, axis=1)))
            active[settled] = False
            if iteration == 11:
                early = int(active.sum())
            if not active.any():
                early = 0 if early is None else early
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, flat]-outgoing
            messages = np.clip(
                damping*messages + (1-damping)*proposal, -24, 24)

        remaining = int(active.sum())
        if early is None:
            early = remaining
        return wrong+remaining, wrong, early, work

    def cycle_energy(rows, nc):
        nr = len(rows)
        adj = [0] * nr
        four = 0
        for i in range(nr):
            for j in range(i):
                common = (rows[i] & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adj[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adj[i] & adj[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = 0
        for col in columns(rows, nc):
            d = col.bit_count()
            stars += d * (d - 1) * (d - 2) // 6
        # Exact six-cycle count without four-cycles; otherwise a heuristic.
        six = max(0, triangles - stars)
        return (80.0 * four + six / 4.0) / q, four, six

    def boundary_scan(cols):
        """Exact existence test for kernel words of weights 1..6.

        Collisions between subsets of size at most three cover all six
        weights. Boundary statistics are only trapping-set heuristics.
        """
        nc = len(cols)
        seen = {0: 0}
        bad = False
        risk = 0
        loads = [0] * (nc // q)

        def record(syndrome, word):
            nonlocal bad
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif old != word:
                bad = True

        for i, col in enumerate(cols):
            record(col, 1 << i)
        for i in range(nc):
            for j in range(i):
                record(cols[i] ^ cols[j], (1 << i) | (1 << j))
        for i in range(nc - 2):
            for j in range(i + 1, nc - 1):
                pair = cols[i] ^ cols[j]
                word = (1 << i) | (1 << j)
                for t in range(j + 1, nc):
                    syndrome = pair ^ cols[t]
                    record(syndrome, word | (1 << t))
                    weight = syndrome.bit_count()
                    if weight <= 6:
                        cost = (8 - weight) ** 2
                        risk += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[t // q] += cost
        return int(bad), risk / nc, max(loads) / q

    def kernel_exposure(cols, trials, side):
        """Verified sampled kernel words, not a distance certificate.

        Common permutation streams compare finalists fairly. Counts
        describe sampled words through ten, not the complete spectrum.
        """
        srng = random.Random(
            250024919 + 1009*idx + 7919*int(side == "B"))
        nc = len(cols)
        order = list(range(nc))
        distance = nc + 1
        small = set()
        for _ in range(trials):
            srng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome = cols[c]
                word = 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    w = word.bit_count()
                    distance = min(distance, w)
                    if w <= 10:
                        small.add(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    v = word ^ other
                    if not v:
                        continue
                    w = v.bit_count()
                    distance = min(distance, w)
                    if w <= 10:
                        small.add(v)
            if distance <= 6:
                break
        return distance, len(small)

    def design_component(supports, side):
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        locals_found = []
        for restart in range(6):
            grid = draw_grid(side)
            rows, nc = component(grid, side)
            current = cycle_energy(rows, nc)
            local = copy_grid(grid)
            local_stats = current

            for step in range(260):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                trial_rows, _ = component(grid, side)
                proposed = cycle_energy(trial_rows, nc)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.98 ** step) + 0.12
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    current = proposed
                    if (current[0] < local_stats[0] or
                            (current[0] == local_stats[0]
                             and rng.random() < 0.06)):
                        local = copy_grid(grid)
                        local_stats = current
                else:
                    grid[i][j] = old

            rows, nc = component(local, side)
            # Admit only augmentation-forced deficits. Complete CSS
            # ranks below remain the authority for the final rate.
            if gf2_rank(rows) != len(rows) - deficits[side]:
                continue
            if not connected(rows, nc):
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        # A finite origin search can miss the required rank stratum.
        # Retry fresh origins under the SAME relative-support contract.
        if not locals_found:
            for trial in range(128):
                grid = draw_grid(side)
                rows, nc = component(grid, side)
                if gf2_rank(rows) != len(rows) - deficits[side]:
                    continue
                if not connected(rows, nc):
                    continue
                stats = cycle_energy(rows, nc)
                locals_found.append((stats, trial, grid, rows, nc))
                locals_found.sort(key=lambda item: item[:2])
                del locals_found[6:]

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found[:3]:
            cols = columns(rows, nc)
            bad, risk, peak = boundary_scan(cols)
            distance, low = kernel_exposure(cols, 48, side)
            stress = decoder_stress(rows, cols, side)
            if mode == 1:
                refinement = (risk, peak, low)
            elif mode == 2:
                refinement = (peak, low, risk)
            else:
                refinement = (low, peak, risk)
            key = (bad, max(0, 12-distance), stress,
                   refinement, stats[1], stats[2])
            finalists.append({
                "grid": grid, "key": key, "d": distance,
                "low": low, "bad": bad, "risk": risk,
                "peak": peak, "four": stats[1], "six": stats[2],
                "stress": stress,
            })
        finalists.sort(key=lambda item: item["key"])
        if not finalists:
            raise ValueError("No rank-safe component filling")
        return finalists

    def logical4(checks, stabilizers, nc):
        """Exact nontrivial quantum-logical existence through weight four."""
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

        cols = columns(checks, nc)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, syndrome in enumerate(cols):
            for j in range(i):
                value = syndrome ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(value)
                if previous is None:
                    seen[value] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    apool = design_component(sa, "A")
    bpool = design_component(sb, "B")
    winner = None
    winning_key = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # Only the complete evaluator-convention ranks decide rate.
            if n != 500 or not 100 <= k <= 112:
                continue
            da, db = max(0, 12-a["d"]), max(0, 12-b["d"])
            stress = tuple(
                value for x, y in zip(a["stress"], b["stress"])
                for value in (max(x, y), x+y))
            key = (
                max(a["bad"], b["bad"]),
                a["bad"] + b["bad"],
                max(da, db), da+db,
            ) + stress + (
                max(a["low"], b["low"]),
                a["low"] + b["low"],
                max(a["peak"], b["peak"]),
                max(a["risk"], b["risk"]),
                a["risk"] + b["risk"],
                a["four"] + b["four"],
                a["six"] + b["six"],
                -k,
            )
            if winning_key is None or key < winning_key:
                if logical4(hx, hz, n) or logical4(hz, hx, n):
                    continue
                if not connected(hx + hz, n):
                    continue
                winning_key = key
                winner = (A, B, hx, hz, k, a["stress"], b["stress"])

    if winner is None:
        raise ValueError("No finalist retained verified cell 0.200")
    A, B, hx, hz, k, stress_a, stress_b = winner
    assert css_ok(hx, hz)
    assert all(row.bit_count() == 9 for row in hx + hz)
    assert 9 <= W_MAX and 100 <= k <= 112
    assert 0.175 < k/500 < 0.225
    assert 6*sum(row.bit_count() for row in hx + hz) == 21600
    assert all(
        len(cell) == supports_by_side[side][i][j]
        and len({G._key(g) for g in cell}) == len(cell)
        for side, grid in (("A", A), ("B", B))
        for i, row in enumerate(grid) for j, cell in enumerate(row))
    for side, grid in (("A", A), ("B", B)):
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                relative = relative_by_side[side][i][j]
                if cell:
                    origin = cell[0]
                    assert [
                        G.mul(G.inv(origin), g) for g in cell
                    ] == relative
    return {"idx": idx, "architecture": (pa, pb),
            "contract": (geometry_number, contract_key),
            "k": k, "cnots": 21600,
            "component_stress": (stress_a, stress_b),
            "screen": winning_key, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without subsequent random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: Raising the screened distance guarantees fewer memory failures: current-parent slot53 had d2s14 and eight X failures/200000, versus slot8's d2s13 and one X failure/200000; labels and extraction exposure also differed, so this is not an isolated gate-count comparison.
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
