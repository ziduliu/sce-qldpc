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
CONCEPT = ("Cyclic C19, order 19, shapes (2,5)x(2,5): augmentation-zero even-term components, long internal cycles and decoder-aware voltage selection should improve the 0.350 frontier's worse-basis circuit fitness beyond about 5.1.")
GROUP = ("Jump from C17 to C19. The existing C19 (2,5)x(2,5) bed reached about 5.6, motivating but not establishing performance for this different rank architecture. The order of 2 modulo 19 is eighteen, so x^19-1 has one irreducible nontrivial factor over F2. Order 19 gives n=551; order 20 is the shape's maximum legal lift, but odd order keeps the augmentation decomposition semisimple.")
SHAPES = ("Equal (2,5) components have 95 variables and 38 checks each. n=551 and kb=171 give guaranteed rate 9/29=0.31034. Zero augmentation and verified component ranks 36 predict k=191; complete evaluator-convention ranks must confirm it, giving rate 0.34664 in target cell 0.350. Parent failures implicated Bcol4 in slots8/36 and Acol0/Bcol2, Acol1/Bcol1 and Arow1/Brow2 in slot53; these sparse samples do not establish a persistent local defect.")
FILLING = ("Three support architectures use only zero, doubleton and four-term cells, component row totals six/eight and column totals two/four. Six unordered support pairs combine with indexed canonical internal-geometry contracts; extra slots receive new contracts, not saved-code relabelings. Directed differences are distinct within each row. Optimize fresh cell voltages using verified lifts, exact kernel exclusion through six, sampled protection capped at ten, and common-stream stressed classical BP ambiguity. Final full ranks, CSS and quantum-logical exclusion through four are checked. Both CSS row-weight sums are <=12 and extraction uses 22344-25536 six-round CNOTs, versus the parent's measured 28560-29376. Target fewer collective failures, not a larger distance upper bound: parent slot53 had eight X failures and 230 logical flips despite d2s=14. Probes omit ASC ordering and measurement histories and are not circuit-pL estimates.")
FAMILY_NAME = "r035_C19_25x25_evenaugmentation_longcycle_bp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order 19")
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

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Indexed even-augmentation contracts and decoder-aware fresh voltages."""
    import math

    q = G.q
    assert q == 19
    assert A_SHAPE == B_SHAPE == (2, 5)

    # Each row has a pure even-term column, and the rows share at least
    # two columns. Pure doubletons have order-19 internal cycles.
    # The four-term architecture replaces two pure doubletons with one
    # higher-degree column, rather than merely permuting the first grid.
    profiles = (
        ((0, 2, 2, 2, 2), (2, 0, 2, 2, 2)),
        ((2, 0, 0, 2, 2), (0, 2, 2, 2, 2)),
        ((4, 0, 0, 2, 2), (0, 2, 2, 2, 2)),
    )
    briefs = [(a, b) for a in range(3) for b in range(a, 3)]
    geometry_number, architecture = divmod(idx, len(briefs))
    pa, pb = briefs[architecture]
    sa = [list(row) for row in profiles[pa]]
    sb = [list(row) for row in profiles[pb]]

    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert wx <= 12 and wz <= 12
    assert _grid_weights(sa, sb) == max(wx, wz) <= W_MAX
    assert _no_dead_lines(sa, sb)
    for support in (sa, sb):
        assert all(sum(row) in (6, 8) for row in support)
        assert all(sum(col) in (2, 4) for col in zip(*support))
        assert all(w in (0, 2, 4) and w <= CELL_MAX
                   for row in support for w in row)
        assert gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support]) == 0

    def normalized(values, multiplier=1):
        if not values:
            return ()
        image = [(multiplier * value) % q for value in values]
        return min(
            tuple(sorted((value - origin) % q for value in image))
            for origin in image)

    def geometry_key(grids):
        """Remove obvious internal affine/protograph orbits.

        This is deliberately more conservative than cell translations:
        equal keys are discarded even when independent cell translations
        would not extend to a full-code Tanner relabeling. It is not a
        complete arbitrary-Tanner-isomorphism classifier.
        """
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                columns_ = [
                    (normalized(grid[0][j], u),
                     normalized(grid[1][j], u))
                    for j in range(5)]
                sides.append(min(
                    tuple(sorted(columns_)),
                    tuple(sorted((b, a) for a, b in columns_))))
            images.append(tuple(sorted(sides)))
        return min(images)

    grng = random.Random(19052027 + 1009 * architecture)

    def relative_grid(support):
        result = []
        for row in support:
            used_differences = set()
            cells = []
            for weight in row:
                for attempt in range(192):
                    values = normalized(grng.sample(range(q), weight))
                    differences = [
                        (a - b) % q for a in values for b in values
                        if a != b]
                    if len(set(differences)) != len(differences):
                        continue
                    if used_differences.isdisjoint(differences):
                        break
                else:
                    return None
                used_differences.update(differences)
                cells.append(values)
            result.append(cells)
        return result

    # Initial twenty slots and additional harness slots select different
    # canonical joint internal geometries. No optimized complete filling
    # is transformed into another slot.
    book, seen = [], set()
    for proposal in range(20000):
        grids = [relative_grid(sa), relative_grid(sb)]
        if any(grid is None for grid in grids):
            continue
        identity = geometry_key(grids)
        if identity in seen:
            continue
        seen.add(identity)
        book.append(grids)
        if len(book) > geometry_number:
            break
    if len(book) <= geometry_number:
        raise ValueError("No unused even-augmentation geometry contract")
    geometry = dict(zip(("A", "B"), book[geometry_number]))

    def draw_grid(side):
        result = []
        for row in geometry[side]:
            cells = []
            for relative in row:
                origin = G.pick(rng)
                cells.append([G.mul(origin, (delta,))
                              for delta in relative])
            result.append(cells)
        return result

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
        """Actual kernel witnesses, not a lower-bound distance certificate.

        Finalists share a permutation stream. Fundamental circuits,
        pair XORs and bounded descent expose sampled short-word burden.
        """
        nc = len(cols)
        srng = random.Random(19051031 + 1009 * idx + int(side == "B"))
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
                    if w <= 12:
                        small.add(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    v = word ^ other
                    if not v:
                        continue
                    w = v.bit_count()
                    distance = min(distance, w)
                    if w <= 12:
                        small.add(v)
            for word in sorted(basis, key=int.bit_count)[:6]:
                for sweep in range(2):
                    best = word
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best.bit_count():
                            best = candidate
                    if best == word:
                        break
                    word = best
                    w = word.bit_count()
                    distance = min(distance, w)
                    if w <= 12:
                        small.add(word)
            if distance <= 6:
                break
        return distance, len(small)

    def small_logical(checks, stabilizers, width):
        """Exact quantum-logical existence test through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, width)
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

    def bp_probe(rows, cols, side):
        """Stressed classical BP selection, NOT circuit memory simulation.

        Settled nonzero residuals are classical kernel failures.
        Unresolved shots are ambiguity penalties. Correlated pairs do
        not claim to reproduce ASC hooks or measurement histories.
        """
        import numpy as np

        width, nr, shots = len(cols), len(rows), 96
        neighborhoods = []
        edge_variables = []
        incident = [[] for _ in range(width)]
        for row in rows:
            ns = []
            while row:
                bit = row & -row
                c = bit.bit_length() - 1
                ns.append(c)
                incident[c].append(len(edge_variables))
                edge_variables.append(c)
                row ^= bit
            neighborhoods.append(ns)

        ne = len(edge_variables)
        max_check = max(map(len, neighborhoods))
        max_variable = max(map(len, incident))
        assert min(map(len, incident)) > 0
        check_edges = np.full((nr, max_check), ne, dtype=np.int32)
        check_sites = np.zeros((nr, max_check), dtype=np.int32)
        valid = np.zeros((nr, max_check), dtype=bool)
        offset = 0
        for r, ns in enumerate(neighborhoods):
            degree = len(ns)
            check_edges[r, :degree] = np.arange(offset, offset + degree)
            check_sites[r, :degree] = ns
            valid[r, :degree] = True
            offset += degree
        variable_edges = np.full(
            (width, max_variable), ne, dtype=np.int32)
        for c, edges in enumerate(incident):
            variable_edges[c, :len(edges)] = edges
        edge_variables = np.asarray(edge_variables, dtype=np.int32)

        channel = np.random.default_rng(
            19051033 + 1009 * idx + int(side == "B"))
        degrees = np.asarray([len(v) for v in incident], dtype=float)
        levels = np.asarray((0.025, 0.050, 0.075))
        rates = levels[np.arange(shots) % 3, None] * degrees[None, :] / 3.0
        error = channel.random((shots, width)) < rates
        trigger = channel.random(shots) < 0.12
        hook_rows = channel.integers(nr, size=shots)
        for shot in np.flatnonzero(trigger):
            ns = neighborhoods[int(hook_rows[shot])]
            a, b = channel.choice(len(ns), 2, replace=False)
            error[shot, ns[int(a)]] ^= True
            error[shot, ns[int(b)]] ^= True

        prior_rates = np.clip(
            rates + 0.24 * degrees[None, :] / ne, 1e-6, 0.25)
        prior = np.log((1.0 - prior_rates) / prior_rates)
        target = np.logical_xor.reduce(
            error[:, check_sites] & valid[None, :, :], axis=2)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, edge_variables].copy()
        outgoing = np.zeros((shots, ne + 1))
        done = np.zeros(shots, dtype=bool)
        decoded = np.zeros((shots, width), dtype=bool)
        completion = np.full(shots, 41, dtype=np.int32)
        positions = np.arange(max_check)[None, None, :]

        for iteration in range(40):
            padded = np.concatenate(
                (messages, np.zeros((shots, 1))), axis=1)
            shaped = padded[:, check_edges]
            magnitude = np.where(valid[None, :, :], np.abs(shaped), np.inf)
            signs = np.where(shaped < 0, -1.0, 1.0)
            product = signs.prod(axis=2) * target_sign
            minimum_at = np.argmin(magnitude, axis=2)
            smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
            size = np.where(
                positions == minimum_at[:, :, None],
                smallest[:, :, 1, None], smallest[:, :, 0, None])
            update = 0.80 * product[:, :, None] * signs * size
            update = np.where(valid[None, :, :], update, 0.0)
            outgoing[:, :ne] = update[:, valid]
            posterior = prior + outgoing[:, variable_edges].sum(axis=2)
            hard = posterior < 0
            parity = np.logical_xor.reduce(
                hard[:, check_sites] & valid[None, :, :], axis=2)
            settled = ~np.any(parity != target, axis=1)
            newly = settled & ~done
            decoded[newly] = hard[newly]
            completion[newly] = iteration + 1
            done |= settled
            if np.all(done):
                break
            proposal = posterior[:, edge_variables] - outgoing[:, :ne]
            messages = (0.30 * messages
                        + 0.70 * np.clip(proposal, -24.0, 24.0))

        difference = decoded ^ error
        failed = ~done | np.any(difference, axis=1)
        return (int(failed.sum()), int((~done).sum()),
                int(difference[done].sum()), int(completion.sum()))

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

            for step in range(240):
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                shift = G.elements[rng.randrange(1, q)]
                # Preserve the indexed internal geometry, changing only
                # this cell's inter-cell voltage.
                grid[i][j] = [G.mul(shift, g) for g in old]
                trial_rows, _ = component(grid, side)
                proposed = cycle_energy(trial_rows, nc)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.982 ** step) + 0.10
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
            # Two zero-augmentation rows force two missing ranks.
            # Do not substitute the algebraic expectation for this check.
            if gf2_rank(rows) != 36:
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found[:4]:
            cols = columns(rows, nc)
            bad, risk, peak = boundary_scan(cols)
            distance, low = kernel_exposure(cols, 48, side)
            probe = bp_probe(rows, cols, side)
            # Sampled protection is capped; after the floor, prefer
            # decoder ambiguity rather than larger distance upper bounds.
            key = (bad, max(0, 10 - distance),
                   probe[0], probe[1], probe[2], probe[3],
                   low, peak, risk, stats[1], stats[2])
            finalists.append({
                "grid": grid, "key": key, "d": distance,
                "probe": probe,
                "low": low, "bad": bad, "risk": risk,
                "peak": peak, "four": stats[1], "six": stats[2],
            })
        finalists.sort(key=lambda item: item["key"])
        if not finalists:
            raise ValueError("No rank-safe component filling")
        return finalists

    apool = design_component(sa, "A")
    bpool = design_component(sb, "B")
    winner = None
    winning_key = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # The augmentation prediction is never substituted for the
            # evaluator-convention ranks of the complete CSS matrices.
            if n != 551 or k != 191:
                continue
            da, db = max(0, 10 - a["d"]), max(0, 10 - b["d"])
            key = (
                max(a["bad"], b["bad"]),
                max(da, db), da + db,
                max(a["probe"][0], b["probe"][0]),
                a["probe"][0] + b["probe"][0],
                max(a["probe"][1], b["probe"][1]),
                a["probe"][1] + b["probe"][1],
                a["probe"][2] + b["probe"][2],
                a["probe"][3] + b["probe"][3],
                max(a["low"], b["low"]), a["low"] + b["low"],
                max(a["peak"], b["peak"]),
                max(a["risk"], b["risk"]),
                a["risk"] + b["risk"],
                a["four"] + b["four"],
                a["six"] + b["six"],
            )
            if winning_key is None or key < winning_key:
                if (small_logical(hx, hz, n)
                        or small_logical(hz, hx, n)):
                    continue
                winning_key = key
                winner = (A, B, hx, hz)

    if winner is None:
        raise ValueError("No short-logical-safe finalist in cell 0.350")
    A, B, hx, hz = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert all(len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    expected_cnots = 6 * 7 * q * (sum(map(sum, sa)) + sum(map(sum, sb)))
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
    assert expected_cnots in (22344, 23940, 25536)
    assert 325 * 551 <= 1000 * 191 < 375 * 551
    return {"idx": idx, "architecture": (pa, pb, geometry_number),
            "k": 191, "screen": winning_key, "cnots": expected_cnots,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without subsequent random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: Raising d2s from thirteen to fourteen guarantees better C17 memory: slot53 had eight X failures/200000 and s=6.806, while slot10 had one failure per basis and s=7.258.
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
