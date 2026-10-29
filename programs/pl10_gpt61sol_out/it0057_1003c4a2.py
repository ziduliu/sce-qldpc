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
CONCEPT = ("D7 = metacyclic Z7:Z2(s=6), order 14, shapes (2,4)x(3,5): component-column redundancy and non-involutive multi-term cells should reduce collective circuit failures relative to the single-row parent; this is a falsifiable pL hypothesis.")
GROUP = ("Keep the parent's order-14 group unchanged to isolate the shape/filling move. Its related (2,4)x(2,5) bed reached 4.7, and the same metacyclic family's order-28 (2,4)x(2,4) bed reached 6.6. These readings motivate testing redundancy but do not isolate a group advantage.")
SHAPES = ("Target cell 0.150 with (2,4)x(3,5): n=364, kb=56 and guaranteed rate 2/13=0.153846. A has 28 checks on 56 variables; B has 42 checks on 70 variables. Verified ranks require 0.125<=k/n<0.175. Parent weight-three witnesses occupied many sector-one Acol/Bcol blocks, with diffuse failed-shot enrichment and flips across all logical qubits; strengthen both components instead of treating one block as a proven hotspot.")
FILLING = ("Twenty architectures cross five A count grids with four dispersed B reinforcement patterns. A column totals are 3-5 and B column totals 3-4; A rows total at most eight and B rows at most six, so both check-weight bounds are <=12. Multi-term cells lie in one rotation coset, making every intra-cell quotient order seven, while whole-cell translations can use reflections. Independently optimize verified component lifts for short-word ambiguity, rank deficits and four-cycles, then screen target-rate combinations by exact quantum logical counts through weight four and full-lift cycle exposure. Retire the parent's min-sum surrogate from the slot entry point; these structural screens are hypotheses about measured pL, not circuit scores.")
FAMILY_NAME = "r015_m7r2s6_24x35_redundant_rotationcoset_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # metacyclic Z_7 x| Z_2: y x y^-1 = x^6 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 7, 2, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


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


def _designed_side(rng, G, counts, side, model):
    """Rotation-difference binomials plus policy-controlled heavy cells."""
    rotations = [g for g in G.elements if g[1] == 0]
    reflections = [g for g in G.elements if g[1] == 1]
    differences = [1, 2, 3]
    rng.shuffle(differences)
    cells = []
    binomial = 0

    for weight in counts:
        if weight == 1:
            cell = [G.pick(rng)]
        elif weight == 2:
            # Distinct undirected rotation differences avoid repeated
            # binomial edge types; no involutive relative label.
            d = differences[binomial]
            binomial += 1
            base = G.pick(rng)
            cell = [base, G.mul(base, (d, 0))]
        else:
            if model == 3:
                # Unrestricted higher-weight supports are a separate
                # design policy, not an automorphism of a saved filling.
                cell = rng.sample(G.elements, weight)
            else:
                rotation_only = (
                    (model == 0 and side == "A") or
                    (model == 1 and side == "B")
                )
                if rotation_only:
                    cell = rng.sample(rotations, weight)
                else:
                    nr = 1 if weight == 3 else 2
                    cell = (rng.sample(rotations, weight - nr) +
                            rng.sample(reflections, nr))
        cells.append(cell)
    return [cells]


def _retired_single_row_ctx(rng, G, idx):
    """Legacy single-row experiment; no longer called when filling slots."""
    profiles = (
        (1, 2, 2, 2),  # 0: low-exposure distance-three component
        (1, 2, 2, 3),
        (1, 2, 2, 4),
        (1, 2, 2, 5),
        (1, 2, 3, 3),
        (1, 2, 3, 4),
        (1, 3, 3, 3),
        (2, 2, 2, 3),
        (2, 2, 2, 4),
        (2, 2, 3, 3),
        (1, 1, 2, 2),  # sparse controls, not orbit copies
        (1, 1, 1, 2),
        (1, 1, 2, 3),
        (1, 1, 3, 3),
    )
    designs = (
        (0, 0), (0, 1), (0, 2), (0, 4), (0, 7),
        (0, 9), (0, 8), (0, 6), (0, 5), (0, 3),
        (1, 7), (4, 7),
        (0, 10), (1, 10), (2, 10), (4, 10),
        (0, 11), (0, 12), (1, 12), (0, 13),
    )
    parent_pairs = {
        ((1, 2, 3, 3), (1, 2, 3, 3)),
        ((1, 2, 2, 3), (1, 2, 3, 3)),
        ((1, 2, 2, 3), (1, 2, 2, 3)),
        ((1, 2, 2, 3), (1, 2, 2, 4)),
        ((2, 2, 2, 3), (2, 2, 2, 3)),
    }
    architectures = [(profiles[a], profiles[b]) for a, b in designs]

    # If the harness requests more than twenty slots, extend the count
    # architecture list instead of recycling the original twenty cases.
    all_profiles = [
        (a, b, c, d)
        for a in range(1, 6)
        for b in range(a, 6)
        for c in range(b, 6)
        for d in range(c, 6)
        if a + b + c + d <= 11
    ]
    extras = []
    for ai, ac0 in enumerate(all_profiles):
        for bc0 in all_profiles[ai:]:
            pair = (ac0, bc0)
            if pair in parent_pairs or pair in architectures:
                continue
            if max(sum(ac0) + max(bc0),
                   sum(bc0) + max(ac0)) <= W_MAX:
                extras.append(pair)
    extras.sort(key=lambda p: (
        sum(p[0]) + sum(p[1]),
        abs(sum(p[0]) - sum(p[1])), p))
    architectures.extend(extras)
    if idx >= len(architectures):
        raise ValueError("No unused count architecture for this slot")
    ac, bc = architectures[idx]
    assert _grid_weights([list(ac)], [list(bc)]) <= W_MAX
    assert _no_dead_lines([list(ac)], [list(bc)])

    def draw(counts):
        """Non-involutive binomials; unrestricted distinct heavy supports."""
        differences = [1, 2, 3]
        rng.shuffle(differences)
        cells = []
        for weight in counts:
            if weight == 1:
                cell = [G.pick(rng)]
            elif weight == 2:
                d = differences.pop() if differences else rng.randint(1, 3)
                base = G.pick(rng)
                cell = [base, G.mul(base, (d, 0))]
            else:
                cell = rng.sample(G.elements, weight)
            cells.append(cell)
        return [cells]

    def component_rows(hx, hz):
        """Project only from the scaffold's verified left/right lifts."""
        q = G.q
        mask = (1 << q) - 1
        ar = [
            sum(((row >> (j * 4 * q)) & mask) << (j * q)
                for j in range(4))
            for row in hx[:q]
        ]
        br = [row & ((1 << (4 * q)) - 1) for row in hz[:q]]
        return ar, br

    def component_stats(rows):
        """Short component words and same-basis four-cycle exposure."""
        columns = [0] * (4 * G.q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        singles = {}
        for i, syndrome in enumerate(columns):
            singles.setdefault(syndrome, []).append(i)
        two = three = 0
        for a, syndrome in enumerate(columns):
            for b in range(a + 1, len(columns)):
                pair = syndrome ^ columns[b]
                two += int(pair == 0)
                three += sum(c > b for c in singles.get(pair, ()))
        four_cycles = 0
        for r, row in enumerate(rows):
            for other in rows[:r]:
                common = (row & other).bit_count()
                four_cycles += common * (common - 1) // 2
        return two, three, four_cycles

    pool = []
    for attempt in range(64):
        A, B = draw(ac), draw(bc)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = component_rows(hx, hz)
        ast, bst = component_stats(ar), component_stats(br)
        key = (
            ast[0] + bst[0],
            max(ast[1], bst[1]), ast[1] + bst[1],
            max(ast[2], bst[2]), ast[2] + bst[2],
        )
        pool.append((key, attempt, A, B, hx, hz, n))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[6:]

    try:
        import numpy as np
    except ImportError:
        np = None

    def stress_failures(checks, stabilizers, n, probabilities, seed):
        """Ideal-syndrome stressed min-sum surrogate, NOT circuit pL.

        Probabilities depend on both bases' gate exposure. Every finalist
        receives the same uniforms, reducing screening noise. A settled
        residual counts as success only when it lies in the stabilizer span.
        """
        shots = 256
        width = max(row.bit_count() for row in checks)
        incidence = np.full((len(checks), width), n, dtype=np.int64)
        for r, row in enumerate(checks):
            t = 0
            while row:
                bit = row & -row
                incidence[r, t] = bit.bit_length() - 1
                row ^= bit
                t += 1
        valid = incidence != n
        stress = np.repeat(np.array([1.5, 3.5]), shots // 2)
        p = np.minimum(0.15, probabilities[:, None] * stress[None, :])
        uniforms = np.random.default_rng(seed).random((n, shots))
        errors = uniforms < p
        padded_errors = np.vstack((errors, np.zeros((1, shots), dtype=bool)))
        syndrome = np.logical_xor.reduce(padded_errors[incidence], axis=1)
        channel = np.vstack((
            np.log((1.0 - p) / p),
            np.full((1, shots), 1.0e6),
        ))
        messages = channel[incidence].copy()
        done = np.zeros(shots, dtype=bool)
        decoded = np.zeros((n, shots), dtype=bool)
        positions = np.arange(width)[None, :, None]

        for iteration in range(32):
            signs = np.where(messages < 0.0, -1.0, 1.0)
            signs = np.where(valid[:, :, None], signs, 1.0)
            magnitudes = np.where(
                valid[:, :, None], np.abs(messages), np.inf)
            smallest = np.partition(magnitudes, 1, axis=1)[:, :2, :]
            minimum_position = np.argmin(magnitudes, axis=1)
            excluded_minimum = np.where(
                positions == minimum_position[:, None, :],
                smallest[:, 1:2, :], smallest[:, 0:1, :])
            parity_sign = np.where(syndrome, -1.0, 1.0)
            outgoing = (
                0.8 * excluded_minimum * signs *
                (np.prod(signs, axis=1) * parity_sign)[:, None, :])
            outgoing = np.where(valid[:, :, None], outgoing, 0.0)
            belief = channel.copy()
            np.add.at(
                belief, incidence.ravel(),
                outgoing.reshape((-1, shots)))
            hard = belief < 0.0
            hard[n, :] = False
            reproduced = np.logical_xor.reduce(hard[incidence], axis=1)
            settled = np.all(reproduced == syndrome, axis=0)
            newly_done = settled & ~done
            decoded[:, newly_done] = hard[:n, newly_done]
            done |= settled
            if np.all(done):
                break
            proposal = belief[incidence] - outgoing
            messages = np.clip(
                0.25 * messages + 0.75 * proposal, -40.0, 40.0)
            messages = np.where(
                valid[:, :, None], messages, 1.0e6)

        pivots = {}
        for row in stabilizers:
            while row:
                pivot = row.bit_length() - 1
                if pivot in pivots:
                    row ^= pivots[pivot]
                else:
                    pivots[pivot] = row
                    break
        failures = int(np.count_nonzero(~done))
        residual = errors ^ decoded
        for shot in np.flatnonzero(done):
            word = sum(1 << int(c) for c in np.flatnonzero(residual[:, shot]))
            while word:
                pivot = word.bit_length() - 1
                if pivot not in pivots:
                    failures += 1
                    break
                word ^= pivots[pivot]
        return failures

    best_key = None
    winner = None
    for base, attempt, A, B, hx, hz, n in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        assert css_ok(hx, hz)
        x = _short_logical_counts(hz, hx, n)
        z = _short_logical_counts(hx, hz, n)
        edges = sum(row.bit_count() for row in hx + hz)

        # Degree and check fanout approximate exposure to data-gate and
        # ancilla-propagation faults; they do not reproduce the ASC circuit.
        risk = [1.0] * n
        for row in hx + hz:
            increment = 0.60 + 0.05 * (row.bit_count() - 1)
            while row:
                bit = row & -row
                risk[bit.bit_length() - 1] += increment
                row ^= bit
        if np is not None:
            probabilities = 0.002 * np.asarray(risk)
            seed = 104729 + 1009 * idx
            fx = stress_failures(hz, hx, n, probabilities, seed)
            fz = stress_failures(hx, hz, n, probabilities, seed)
            decoding_key = (max(fx, fz), fx + fz)
        else:
            # Dependency-free fallback retains the exposure experiment.
            decoding_key = (
                max(x[2], z[2]) * (edges / n) ** 2,
                (x[2] + z[2]) * (edges / n) ** 2,
            )
        key = (
            x[0] + z[0], x[1] + z[1],
            decoding_key,
            max(x[2], z[2]), x[2] + z[2],
            max(x[3], z[3]), x[3] + z[3],
            base[3:], edges, attempt,
        )
        if best_key is None or key < best_key:
            best_key = key
            winner = {"A": A, "B": B}

    if winner is None:
        raise ValueError("No positive-k finalist in count architecture")
    return {"idx": idx, "case": idx, "cells": winner,
            "screen_key": best_key}


def NEW_CTX(rng, G, idx):
    """Twenty redundant-component architectures, screened on verified lifts."""
    import math

    a_profiles = (
        ((1, 2, 2, 2), (2, 1, 2, 2)),
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((1, 1, 2, 2), (2, 2, 1, 2)),
        ((1, 2, 2, 2), (2, 2, 2, 2)),
        ((1, 2, 2, 2), (2, 1, 2, 3)),
    )
    b_extras = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    )
    design = idx % 20
    sa = [list(row) for row in a_profiles[design // 4]]
    sb = [[1] * B_SHAPE[1] for _ in range(B_SHAPE[0])]

    # Reinforcements occupy different columns. This changes support
    # structure, not a saved code's labeling or permutation orbit.
    columns = list(range(B_SHAPE[1]))
    rng.shuffle(columns)
    cursor = 0
    for i, extra in enumerate(b_extras[design % 4]):
        for j in columns[cursor:cursor + extra]:
            sb[i][j] = 2
        cursor += extra

    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    cosets = [
        [g for g in G.elements if g[1] == b]
        for b in range(2)
    ]

    def draw_cell(weight):
        if weight == 1:
            return [G.pick(rng)]
        # Distinct elements in one coset have nonidentity rotation
        # quotients of order seven, never involutory quotients.
        return rng.sample(cosets[rng.randrange(2)], weight)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component_rows(side, grid):
        """Use zero probes only to extract the verified component action."""
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        mask = (1 << (len(grid[0]) * G.q)) - 1
        return [row & mask for row in rows]

    def four_cycles(rows):
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def connected(rows, width):
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
            first = bit.bit_length() - 1
            row ^= bit
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(i) for i in range(width)}) == 1

    def component_statistics(rows, width):
        """Kernel ambiguities and cycles, not a quantum-distance claim.

        With distinct nonzero columns, triple and pair-collision counts
        count each weight-three/four kernel support three times.
        Otherwise the duplicate statistic takes precedence.
        """
        syndromes = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                syndromes[bit.bit_length() - 1] |= tag
                row ^= bit

        singles = {}
        duplicates = 0
        for syndrome in syndromes:
            duplicates += int(syndrome == 0) + singles.get(syndrome, 0)
            singles[syndrome] = singles.get(syndrome, 0) + 1

        pairs = {}
        triples = collisions = boundary = 0
        for i, x in enumerate(syndromes):
            for y in syndromes[:i]:
                syndrome = x ^ y
                triples += singles.get(syndrome, 0)
                old = pairs.get(syndrome, 0)
                collisions += old
                pairs[syndrome] = old + 1
                w = syndrome.bit_count()
                if w < 3:
                    boundary += (3 - w) ** 2
        return (duplicates, triples, collisions,
                four_cycles(rows), boundary)

    def component_pool(side, support):
        """Independent multistart search, retaining distinct complete grids."""
        height, width = len(support), len(support[0])
        ncols = width * G.q
        positions = [
            (i, j) for i in range(height) for j in range(width)
            if support[i][j]
        ]
        pool = []
        fingerprints = set()

        def assess(grid):
            rows = component_rows(side, grid)
            stats = component_statistics(rows, ncols)
            energy = (
                100000 * stats[0] + 1000 * stats[1] +
                100 * stats[2] + 12 * stats[3] + stats[4]
            )
            return rows, stats, energy

        def remember(grid, rows, stats):
            # Rank deficits are screened, not rewarded for extra k.
            deficit = len(rows) - gf2_rank(rows)
            key = stats[:3] + (deficit,) + stats[3:]
            if len(pool) >= 4 and key >= pool[-1][0]:
                return
            if not connected(rows, ncols):
                return
            fingerprint = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            if fingerprint in fingerprints:
                return
            fingerprints.add(fingerprint)
            pool.append((key, clone(grid), rows))
            pool.sort(key=lambda item: item[0])
            del pool[4:]

        # Extra harness slots independently optimize the same twenty
        # architectures; no saved filling is transformed into another slot.
        for restart in range(3):
            grid = [
                [draw_cell(weight) for weight in row]
                for row in support
            ]
            rows, stats, energy = assess(grid)
            remember(grid, rows, stats)
            for step in range(80):
                trial = clone(grid)
                i, j = rng.choice(positions)
                trial[i][j] = draw_cell(support[i][j])
                rr, ss, ee = assess(trial)
                remember(trial, rr, ss)
                temperature = 150.0 * (1.0 - step / 80.0) + 4.0
                delta = ee - energy
                if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    grid, rows, stats, energy = trial, rr, ss, ee

        if not pool:
            raise ValueError("no connected redundant component")
        return pool

    apool = component_pool("A", sa)
    bpool = component_pool("B", sb)
    combinations = []
    bound = G.q * (A_SHAPE[1] - A_SHAPE[0]) * (
        B_SHAPE[1] - B_SHAPE[0])

    for ak, A, ar in apool:
        for bk, B, br in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # Exact evaluator ranks determine the rate cell, not kb.
            if not (k > 0 and 8 * k >= n and 40 * k < 7 * n):
                continue
            if not connected(hx + hz, n):
                continue
            key = tuple(
                v for t in range(len(ak))
                for v in (max(ak[t], bk[t]), ak[t] + bk[t])
            )
            combinations.append((key, A, B, hx, hz, n, k))

    if not combinations:
        raise ValueError("no verified-rank filling in the 0.150 band")
    combinations.sort(key=lambda item: item[0])

    best = None
    for base, A, B, hx, hz, n, k in combinations[:4]:
        # Distinguish true quantum logicals from component kernel words.
        x = _short_logical_counts(hz, hx, n)
        z = _short_logical_counts(hx, hz, n)
        fx, fz = four_cycles(hx), four_cycles(hz)
        edges = sum(row.bit_count() for row in hx + hz)
        key = (
            x[0] + z[0], x[1] + z[1],
            max(x[2], z[2]), x[2] + z[2],
            max(x[3], z[3]), x[3] + z[3],
            max(fx, fz), fx + fz,
            edges, k - bound, base,
        )
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, n, k)

    key, A, B, hx, hz, n, k = best
    assert css_ok(hx, hz)
    assert n == 364
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert 8 * k >= n and 40 * k < 7 * n
    return {"idx": idx, "case": design, "k": k,
            "cells": {"A": A, "B": B}, "screen_key": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: The previous short-logical/gate-balance move improved overall circuit fitness: measured fitness fell from 3.4511 to 2.9485 despite reducing six-round CNOTs to 7140-7560; this comparison does not isolate the shape or filling effect.
# REFUTED: Lower-exposure single-row profiles plus the gate-weighted min-sum surrogate improved fitness over 2.9485: measured fitness was 2.9194, all five measured codes had d2s=3, and slot 6 X failed 175/250 shots.
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
