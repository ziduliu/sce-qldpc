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
CONCEPT = ("C7 semidirect C6, action 2, order 42, shapes (1,2)x(2,6): gate-neutral weak/strong B columns and stabilizer-aware decoder screening should improve the measured cell-0.300 incumbent 6.1594 without increasing its 25704 extraction CNOTs.")
GROUP = ("Jump from the order-39 C13:C3 parent to C7:C6(action 2), order 42. The action has order three and its kernel supplies a central C2. This exact (1,2)x(2,6) bed measured 6.1594; slot10 had 59 Z and 52 X failures in 200000 shots. Those readings motivate a new filling experiment, not a causal group advantage or an untried-bed claim.")
SHAPES = ("New unequal (1,2) and (2,6) components have 84 variables/42 checks and 252 variables/84 checks. Size factor14 admits q<=42, giving n=588, kb=168 and guaranteed rate2/7. Complete verified ranks require 168<=k<=184, inside cell0.300. Current parent slots6 and11 each had one failed shot in one basis, implicating different Acol/Bcol and Arow/Brow blocks; three measured siblings had none. This does not establish a persistent hotspot to preserve or reinforce.")
FILLING = ("A has two noncommuting trinomials; a deterministic catalogue reserves different rooted uncolored component invariants for every supplied slot. B has row totals9 and total18, with canonical count cards containing degree-two and degree-four/five columns rather than the archived uniform degree-three architecture. Cells hold0..4 distinct elements. HX<=6+5=11 and HZ=9+3=12; six-round CNOTs remain25704, below the parent's28080 despite larger n and k. Helper-derived actions anneal fresh B voltages. Exact component and quantum exclusion through four, sampled warning thresholds and complete rank/CSS checks precede common-noise stabilizer-aware BP probes and independent holdouts. The probes include approximate data/spread faults but omit measurement histories, ASC ordering and relay/LSD, so they are not circuit pL estimates. Target the order-42 incumbent's collective failures and late settlement rather than maximize d2s.")
FAMILY_NAME = "r030_C7sdC6_action2_12x26_gate_neutral_degreecards_holdout_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C6 requires order 42")
    # x^7=y^6=1, y*x*y^-1=x^2; elements are x^a*y^b.
    # 2^3=1 modulo 7, so this defines an action of C6.
    els = [(a, b) for b in range(6) for a in range(7)]
    action = (1, 2, 4, 1, 2, 4)

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-action[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


# Profiles are distinct under component row/column permutations.
# Every row has weight 6 or 7; every column has weight 4 or 5.
_SUPPORT_PROFILES = (
    ((2, 2, 2), (2, 2, 2)),
    ((1, 2, 3), (3, 2, 1)),
    ((1, 1, 4), (3, 3, 1)),
    ((1, 2, 3), (3, 2, 2)),
    ((1, 3, 2), (3, 1, 3)),
    ((2, 2, 2), (2, 2, 3)),
    ((2, 3, 1), (2, 1, 4)),
    ((1, 2, 4), (3, 3, 1)),
    ((1, 3, 3), (3, 2, 2)),
    ((2, 1, 4), (2, 4, 1)),
    ((2, 2, 3), (2, 3, 2)),
)

# Twenty support designs, not automorphism images of one labeling.
_DESIGN_PAIRS = (
    (0, 0), (0, 1), (0, 3), (0, 4), (0, 5),
    (0, 8), (0, 9), (0, 10), (1, 1), (1, 3),
    (1, 5), (1, 8), (1, 10), (3, 3), (3, 5),
    (3, 10), (5, 5), (5, 8), (5, 9), (5, 10),
)


def _balanced_cell(rng, G, count):
    """Balance terms across C3 cosets; offsets remain independently variable."""
    phase = rng.randrange(3)
    direction = 1 if rng.randrange(2) else -1
    cell = []
    used = set()
    for t in range(count):
        b = (phase + direction * t) % 3
        a = rng.randrange(13)
        while (a, b) in used:
            a = rng.randrange(13)
        used.add((a, b))
        cell.append((a, b))
    return cell


def _label_profile(rng, G, profile):
    return [[_balanced_cell(rng, G, count) for count in row]
            for row in profile]


def _overlap_cost(hx, hz):
    """Same-basis graph heuristic; never presented as a measured pL."""
    worst = 0
    excess = 0
    cycles = 0
    for rows in (hx, hz):
        for a, row in enumerate(rows):
            for other in rows[:a]:
                overlap = (row & other).bit_count()
                worst = max(worst, overlap)
                excess += max(0, overlap - 2) ** 2
                cycles += overlap * (overlap - 1) // 2
    return worst, excess, cycles


def _retired_f39_ctx(rng, G, idx):
    """Inactive order-39 experiment retained for provenance."""
    from itertools import combinations_with_replacement

    supports = ((2, 2, 2), (2, 2, 2))
    assert _grid_weights(supports, supports) <= min(10, W_MAX)
    assert _no_dead_lines(supports, supports)

    # Canonical unordered homogeneous-step triples under all normal-
    # subgroup multipliers and independent reversal of doubletons.
    # These are ten geometry orbits, not ten automorphism images.
    geometries = set()
    for steps in combinations_with_replacement(range(1, 7), 3):
        images = []
        for u in range(1, 13):
            images.append(tuple(sorted(
                min((u * d) % 13, 13 - (u * d) % 13)
                for d in steps)))
        geometries.add(min(images))
    book = sorted(geometries)
    assert len(book) == 10

    def joint_key(first, second):
        """Normalize explicit joint step symmetries, not arbitrary graphs.

        A common normal-subgroup multiplier is a group automorphism.
        Independent inner conjugation of one side contributes powers
        of three. Doubleton reversal and component column permutations
        are included by unsigned sorting; A/B interchange is included.
        """
        images = []
        for u in range(1, 13):
            for relative_inner in (1, 3, 9):
                sides = []
                for steps, scale in ((first, u),
                                     (second, u * relative_inner)):
                    sides.append(tuple(sorted(
                        min((scale * d) % 13,
                            13 - (scale * d) % 13)
                        for d in steps)))
                images.append(tuple(sorted(sides)))
        return min(images)

    # Exclude every step-pair architecture used by the current parent,
    # including its larger-budget phase variants. This exclusion is
    # deliberately broader than exact-label archive comparison.
    old_pairs = {
        joint_key(book[p], book[(p + shift) % len(book)])
        for p in range(len(book)) for shift in (3, 4)
    }
    joint_geometries = set()
    for first in book:
        for second in book:
            if len(set(first)) == 1 or len(set(second)) == 1:
                continue
            for scale in range(1, 7):
                scaled = tuple((scale * d) % 13 for d in second)
                key = joint_key(first, scaled)
                if key not in old_pairs:
                    joint_geometries.add(key)

    def geometry_priority(pair):
        a, b = pair
        # Repeated directions and cross-side direction coincidences
        # are exposure hypotheses, not logical-distance certificates.
        return (6 - len(set(a)) - len(set(b)),
                len(set(a) & set(b)), pair)

    joint_book = sorted(joint_geometries, key=geometry_priority)
    assert len(joint_book) >= 20
    design = idx % len(joint_book)
    mode = idx // len(joint_book)
    ga, gb = joint_book[design]
    phase_patterns = ((0, 0, 0), (0, 0, 1),
                      (0, 0, 2), (0, 1, 2))
    phase_pairs = ((1, 2), (2, 3), (1, 3), (0, 3))
    if mode >= len(phase_pairs):
        raise ValueError("joint geometry/phase contract book exhausted")
    ha, hb = phase_pairs[(design + mode) % len(phase_pairs)]

    def grid(geometry, phase):
        result = [[], []]
        for j in range(3):
            t = rng.randrange(3)
            # Nonzero projection at both nontrivial C3 characters.
            result[0].append([
                (rng.randrange(13), t),
                (rng.randrange(13), (t + 1) % 3)])
            # This row projects to zero on the entire C3 quotient.
            b = (t + phase_patterns[phase][j]) % 3
            a = rng.randrange(13)
            delta = geometry[j]
            result[1].append([(a, b), ((a + delta) % 13, b)])
        return result

    q = G.q
    component_n = 3 * q
    block_mask = (1 << q) - 1

    def components(hx, hz):
        """Project only from the scaffold's verified full lift."""
        ar = []
        for i in range(2):
            for r in range(q):
                row = hx[(i * 3) * q + r]
                packed = 0
                for j in range(3):
                    packed |= ((row >> (j * 3 * q)) &
                               block_mask) << (j * q)
                ar.append(packed)
        br = [hz[a * q + r] & ((1 << component_n) - 1)
              for a in range(2) for r in range(q)]
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

    def sampled_kernel(cols):
        """True kernel witnesses, but only an upper-bound search."""
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(40):
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
                    for other in basis:
                        best = min(best, (word ^ other).bit_count())
                    basis.append(word)
            if best < 16:
                break
        return best

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

    def quantum_probe(checks, stabilizers, n):
        """Sample actual logical words and possible shared-check pairs.

        Every offered word is in the verified check kernel. Reduction
        modulo the verified stabilizer span decides nontriviality.
        Minima and multiplicities are sampled upper-bound observations,
        never distance certificates or measured circuit failure rates.
        """
        cols = columns(checks, n)
        span = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = span.get(p)
                if old is None:
                    span[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = span.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        order = list(range(n))
        minimum = n + 1
        short = set()
        for _ in range(12):
            rng.shuffle(order)
            pivots = {}
            words = []
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
                if not syndrome and word:
                    words.append(word)

            # Stabilizers and previous kernel circuits are legitimate
            # shortening moves. Kernel-circuit moves can change the
            # logical class, so nontriviality is checked afterwards.
            words.sort(key=int.bit_count)
            reducers = list(stabilizers)
            for word in words[:96]:
                weight = word.bit_count()
                for sweep in range(2):
                    rng.shuffle(reducers)
                    changed = False
                    for reducer in reducers:
                        proposal = word ^ reducer
                        proposed_weight = proposal.bit_count()
                        if proposed_weight < weight:
                            word, weight = proposal, proposed_weight
                            changed = True
                    if not changed:
                        break
                if not word:
                    continue
                if nontrivial(word):
                    minimum = min(minimum, weight)
                    if weight <= 24:
                        short.add(word)
                reducers.append(word)

        # Same opposite-basis check membership marks a possible
        # ancilla-spread pair. It does NOT identify an ASC hook ordering,
        # and this pair-cover cost is not an actual fault-distance bound.
        neighbors = [0] * n
        for row in stabilizers:
            bits = row
            while bits:
                bit = bits & -bits
                c = bit.bit_length() - 1
                neighbors[c] |= row ^ bit
                bits ^= bit

        minimum_pair_cost = n + 1
        low_pair_exposure = 0
        witnesses = sorted(short, key=lambda w: (w.bit_count(), w))[:64]
        for word in witnesses:
            vertices = []
            bits = word
            while bits:
                bit = bits & -bits
                vertices.append(bit.bit_length() - 1)
                bits ^= bit
            best_pairs = 0
            for _ in range(4):
                rng.shuffle(vertices)
                remaining = word
                pairs = 0
                for c in vertices:
                    bit = 1 << c
                    if not remaining & bit:
                        continue
                    remaining ^= bit
                    available = neighbors[c] & remaining
                    if available:
                        partner = available & -available
                        remaining ^= partner
                        pairs += 1
                best_pairs = max(best_pairs, pairs)
            pair_cost = word.bit_count() - best_pairs
            minimum_pair_cost = min(minimum_pair_cost, pair_cost)
            low_pair_exposure += int(pair_cost < 8)

        # Above coarse protection thresholds, do not reward an
        # indefinitely increasing optimistic distance estimate.
        risk = (int(minimum < 16), int(minimum_pair_cost < 8),
                low_pair_exposure,
                sum(word.bit_count() <= 20 for word in short))
        return risk, minimum, minimum_pair_cost, len(short)

    for batch in range(3):
        pool = []
        for trial in range(96):
            A, B = grid(ga, ha), grid(gb, hb)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = components(hx, hz)
            fa, fb = four(ar), four(br)
            coarse = (max(fa, fb), fa + fb)
            pool.append((coarse, trial, A, B, hx, hz, n, ar, br))
        pool.sort(key=lambda item: (item[0], item[1]))

        finalists = []
        eligible = 0
        for item in pool:
            coarse, trial, A, B, hx, hz, n, ar, br = item
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (59 <= k <= 63 and 8 * k < n):
                continue
            ac, bc = columns(ar, component_n), columns(br, component_n)
            if not connected(ac, 2 * q) or not connected(bc, 2 * q):
                continue
            eligible += 1
            ba = boundary_screen(ac)
            bb = boundary_screen(bc) if ba is not None else None
            if ba is not None and bb is not None:
                da, db = sampled_kernel(ac), sampled_kernel(bc)
                sa, sb = six(ar), six(br)
                distance = min(da, db)
                # Keep a component protection warning, then exposure.
                # The final decision also searches the complete CSS
                # quotient; component words alone miss mixed-sector risks.
                key = (int(distance < 18), max(ba, bb), ba + bb,
                       max(sa, sb), sa + sb, abs(da - db),
                       -min(distance, 20), -k, coarse, trial)
                finalists.append((key, item, k))
            if eligible >= 8:
                break

        finalists.sort(key=lambda entry: entry[0])
        quantum_finalists = []
        for key, item, k in finalists[:4]:
            _, _, A, B, hx, hz, n, _, _ = item
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            xprofile = quantum_probe(hz, hx, n)
            zprofile = quantum_probe(hx, hz, n)
            xrisk, zrisk = xprofile[0], zprofile[0]
            quantum_key = tuple(
                value for t in range(len(xrisk))
                for value in (max(xrisk[t], zrisk[t]),
                              xrisk[t] + zrisk[t]))
            # Protect the worse basis, then total sampled exposure.
            # Neither profile is substituted for measured circuit pL.
            quantum_finalists.append(
                (quantum_key + key, item, k, xprofile, zprofile))

        if quantum_finalists:
            quantum_finalists.sort(key=lambda entry: entry[0])
            key, item, k, xp, zp = quantum_finalists[0]
            _, _, A, B, hx, hz, n, _, _ = item
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 10
            assert 6 * sum(row.bit_count() for row in hx + hz) == 28080
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": design, "mode": mode,
                    "joint_geometry": (ga, gb),
                    "k": k, "screen": key,
                    "quantum_profiles": (xp, zp)}

    raise ValueError("no fresh joint-step filling passed verified CSS screening")


def _degree_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _degree_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    return pivots


def _degree_nontrivial(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _degree_low4(rows, stabilizers, width):
    """Exact kernel/logical existence through four, using verified rows."""
    cols = _degree_columns(rows, width)
    pivots = _degree_span(stabilizers)
    seen = {0: 0}
    for c, syndrome in enumerate(cols):
        word = 1 << c
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif _degree_nontrivial(word ^ old, pivots):
            return True
    for c, syndrome in enumerate(cols):
        for d in range(c):
            value = syndrome ^ cols[d]
            word = (1 << c) | (1 << d)
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _degree_nontrivial(word ^ old, pivots):
                return True
    return False


def _degree_geometry(rows, width):
    """C4, a C6 proposal statistic, and verified Tanner connectivity."""
    nr = len(rows)
    adjacency = [0] * nr
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap * (overlap - 1) // 2
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
    cols = _degree_columns(rows, width)
    stars = sum(d * (d - 1) * (d - 2) // 6
                for d in (s.bit_count() for s in cols))
    # Exact C6 when C4=0; otherwise deliberately only a proposal heuristic.
    six = max(0, triangles - stars)
    reached = frontier = 1
    while frontier:
        following = 0
        while frontier:
            bit = frontier & -frontier
            following |= adjacency[bit.bit_length() - 1]
            frontier ^= bit
        frontier = following & ~reached
        reached |= frontier
    connected = (reached == (1 << nr) - 1 and all(cols))
    return four, six, connected


def _degree_fingerprint(rows, width):
    """Rooted uncolored component invariant, not full-code canonization.

    Single-row regular components have one transitive check orbit.
    Unequal fingerprints therefore distinguish these component graphs;
    arbitrary equivalence of complete LP codes is still evaluator-tested.
    """
    import hashlib
    from collections import Counter

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, syndrome in enumerate(_degree_columns(rows, width)):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            syndrome ^= bit
    initial = [(int(v >= nr), len(ns), int(v == 0))
               for v, ns in enumerate(adjacency)]
    palette = {key: t for t, key in enumerate(sorted(set(initial)))}
    colors = [palette[key] for key in initial]
    digest = hashlib.sha256()
    for _ in range(12):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in ns)))
            for v, ns in enumerate(adjacency)]
        digest.update(repr(sorted(
            Counter(signatures).items())).encode("ascii"))
        palette = {key: t for t, key in enumerate(sorted(set(signatures)))}
        colors = [palette[key] for key in signatures]
    return digest.hexdigest()


def _degree_sample(rows, width, seed):
    """Genuine sampled component words; minima are only upper bounds."""
    source = random.Random(seed)
    cols = _degree_columns(rows, width)
    order = list(range(width))
    best = width + 1
    small = set()
    for _ in range(24):
        source.shuffle(order)
        pivots, basis = {}, []
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
                basis.append(word)
        basis.sort(key=int.bit_count)
        reducers = basis[:32]
        for word in basis:
            for _ in range(2):
                proposal = min(
                    [word] + [word ^ other for other in reducers
                              if word != other],
                    key=int.bit_count)
                if proposal == word:
                    break
                word = proposal
            weight = word.bit_count()
            if weight:
                best = min(best, weight)
                if weight <= 10:
                    small.add(word)
        if best <= 4:
            break
    return best, len(small)


def _degree_bp_probe(checks, stabilizers, n, seed, shots):
    """Stabilizer-aware noisy-data BP stress, NOT the ASC memory circuit.

    Common random streams compare finalists. Degree-weighted aggregate
    data rates and shared-check spread pairs are intentionally approximate.
    No measurement history, ASC gate order, relay BP or LSD is reproduced.
    """
    import numpy as np

    def sites(row):
        out = []
        while row:
            bit = row & -row
            out.append(bit.bit_length() - 1)
            row ^= bit
        return out

    ns = [sites(row) for row in checks]
    opposite = [sites(row) for row in stabilizers]
    incident = [[] for _ in range(n)]
    edge_variables = []
    for row in ns:
        for c in row:
            incident[c].append(len(edge_variables))
            edge_variables.append(c)
    ne = len(edge_variables)
    maximum = max(map(len, ns))
    ce = np.full((len(ns), maximum), ne, dtype=np.int32)
    cv = np.full((len(ns), maximum), n, dtype=np.int32)
    valid = np.zeros((len(ns), maximum), dtype=bool)
    position = 0
    for r, row in enumerate(ns):
        degree = len(row)
        ce[r, :degree] = np.arange(position, position + degree)
        cv[r, :degree] = row
        valid[r, :degree] = True
        position += degree
    ve = np.full((n, max(map(len, incident))), ne, dtype=np.int32)
    for c, edges in enumerate(incident):
        ve[c, :len(edges)] = edges
    ev = np.asarray(edge_variables, dtype=np.int32)

    sensed = np.asarray(list(map(len, incident)), dtype=float)
    spread = np.asarray([
        c.bit_count() for c in _degree_columns(stabilizers, n)
    ], dtype=float)
    levels = np.asarray((0.35, 0.60, 0.85))[np.arange(shots) % 3]
    rates = np.minimum(
        0.24, levels[:, None] * 0.002
        * (2.0 + 4.0 * (sensed + spread))[None, :])
    channel = np.random.default_rng(seed)
    errors = channel.random((shots, n)) < rates
    triggers = (channel.random((shots, len(opposite)))
                < (0.0015 * levels[:, None]))
    for shot, r in zip(*np.nonzero(triggers)):
        row = opposite[int(r)]
        if len(row) >= 2:
            a, b = channel.choice(len(row), 2, replace=False)
            errors[shot, row[int(a)]] ^= True
            errors[shot, row[int(b)]] ^= True

    prior_rates = np.minimum(
        0.28, rates + 0.003 * levels[:, None] * spread[None, :])
    prior = np.log((1.0 - prior_rates) / prior_rates)
    padded_errors = np.concatenate(
        (errors, np.zeros((shots, 1), dtype=bool)), axis=1)
    target = np.logical_xor.reduce(
        padded_errors[:, cv] & valid[None, :, :], axis=2)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, ev].copy()
    outgoing = np.zeros((shots, ne + 1), dtype=float)
    done = np.zeros(shots, dtype=bool)
    decoded = np.zeros((shots, n), dtype=bool)
    completion = np.full(shots, 37, dtype=np.int32)
    positions = np.arange(maximum)[None, None, :]

    for iteration in range(1, 37):
        padded = np.concatenate(
            (messages, np.full((shots, 1), 1.0e6)), axis=1)
        incoming = padded[:, ce]
        magnitude = np.abs(incoming)
        smallest_at = magnitude.argmin(axis=2)
        smallest = magnitude.min(axis=2)
        second = np.partition(magnitude, 1, axis=2)[:, :, 1]
        signs = np.where(incoming < 0.0, -1.0, 1.0)
        product = signs.prod(axis=2) * target_sign
        excluded = np.where(
            positions == smallest_at[:, :, None],
            second[:, :, None], smallest[:, :, None])
        update = 0.8 * product[:, :, None] * signs * excluded
        outgoing[:, :ne] = update[:, valid]
        posterior = prior + outgoing[:, ve].sum(axis=2)
        hard = posterior < 0.0
        padded_hard = np.concatenate(
            (hard, np.zeros((shots, 1), dtype=bool)), axis=1)
        syndrome = np.logical_xor.reduce(
            padded_hard[:, cv] & valid[None, :, :], axis=2)
        settled = ~np.any(syndrome != target, axis=1)
        newly = settled & ~done
        decoded[newly] = hard[newly]
        completion[newly] = iteration
        done |= settled
        if done.all():
            break
        proposal = posterior[:, ev] - outgoing[:, :ne]
        messages = np.clip(
            0.30 * messages + 0.70 * proposal, -30.0, 30.0)

    pivots = _degree_span(stabilizers)
    wrong = flips = 0
    for shot in np.flatnonzero(done):
        residual = decoded[shot] ^ errors[shot]
        word = sum(1 << int(c) for c in np.flatnonzero(residual))
        if _degree_nontrivial(word, pivots):
            wrong += 1
            flips += int(residual.sum())
    unresolved = int((~done).sum())
    return wrong + unresolved, unresolved, wrong, flips, int(completion.sum())


def NEW_CTX(rng, G, idx):
    """Reserved component classes, gate-neutral cards and decoder holdouts."""
    import math
    from itertools import combinations, combinations_with_replacement

    q = G.q
    wanted = max(20, N_SLOTS)
    assert q == 42 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    assert 0 <= idx < N_SLOTS
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        # Cache exclusively scaffold-derived actions.
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for cell_row in grid:
            out = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    for r, word in enumerate(actions[side][G.idx(g)]):
                        out[r] ^= word << (j * q)
            rows.extend(out)
        return rows

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[G.idx(g)] = order
                break
        assert G.idx(g) in orders
    long_rotations = [g for g in G.elements
                      if orders[G.idx(g)] in (7, 14)]
    ordinary = [g for g in G.elements if orders[G.idx(g)] >= 3]

    def normalized(cell):
        # Right column gauges of the one-row A component.
        return min(
            tuple(sorted(G.idx(G.mul(g, G.inv(origin))) for g in cell))
            for origin in cell)

    triangles = set()
    for u, v in combinations([g for g in G.elements if g != G.e], 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        cell = [G.e, u, v]
        if any(orders[G.idx(G.mul(G.inv(a), b))] < 3
               for a, b in combinations(cell, 2)):
            continue
        triangles.add(normalized(cell))
    triangles = [[G.elements[t] for t in key] for key in sorted(triangles)]
    assert triangles

    # Rebuild the same local catalogue in every pure call. Slots own
    # invariant-distinct A graphs, not independent draws of one model.
    source = random.Random(4201202609)
    proposals, seen = [], set()
    for serial in range(max(5000, 80 * wanted)):
        A = [[list(source.choice(triangles)),
              list(source.choice(triangles))]]
        label = tuple(sorted(tuple(sorted(G.idx(g) for g in cell))
                             for cell in A[0]))
        if label in seen:
            continue
        seen.add(label)
        ar = component(A, "A")
        four, six, connected = _degree_geometry(ar, 2 * q)
        if four or not connected or gf2_rank(ar) != q:
            continue
        proposals.append(((six, label), serial, A, ar))
    proposals.sort(key=lambda item: item[:2])

    bank, identities = [], set()
    for coarse, serial, A, ar in proposals:
        if _degree_low4(ar, [], 2 * q):
            continue
        identity = _degree_fingerprint(ar, 2 * q)
        if identity in identities:
            continue
        identities.add(identity)
        bank.append((identity, serial, A, ar))
        if len(bank) >= wanted:
            break
    if len(bank) < wanted:
        raise ValueError("insufficient distinct connected A component classes")
    a_identity, a_serial, A, ar = bank[idx]
    da, na = _degree_sample(ar, 2 * q, 4200101 + idx)

    # Count cards are canonical under both B-row exchange and B-column
    # permutation. Every card differs from the archived all-degree-three
    # rule, while conserving its eighteen terms and row totals nine.
    column_types = [
        (a, b) for a in range(5) for b in range(5)
        if 2 <= a + b <= 5
    ]
    card_keys = set()
    for columns in combinations_with_replacement(column_types, 6):
        if sum(a for a, b in columns) != 9:
            continue
        if sum(b for a, b in columns) != 9:
            continue
        degrees = [a + b for a, b in columns]
        if min(degrees) != 2 or max(degrees) < 4:
            continue
        parity = [
            sum((column[i] & 1) << j
                for j, column in enumerate(columns))
            for i in range(2)]
        if gf2_rank(parity) != 2:
            continue
        card_keys.add(min(
            tuple(sorted(columns)),
            tuple(sorted((b, a) for a, b in columns))))

    def priority(card):
        degrees = [sum(column) for column in card]
        return (max(degrees), degrees.count(2),
                sum(0 in column for column in card),
                max(max(column) for column in card), card)

    cards = sorted(card_keys, key=priority)
    assert len(cards) >= wanted
    card = cards[idx]
    sa = [[3, 3]]
    sb = [[column[i] for column in card] for i in range(2)]
    assert tuple(map(sum, sb)) == (9, 9)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    def draw_cell(i, j):
        count = sb[i][j]
        if count == 0:
            return []
        anchor = G.pick(rng)
        if count == 1:
            return [anchor]
        if count == 2:
            palette = long_rotations if sum(card[j]) == 2 else ordinary
            delta = rng.choice(palette)
            return [anchor, G.mul(anchor, delta)]
        if count == 3:
            return [G.mul(anchor, g) for g in rng.choice(triangles)]
        return rng.sample(G.elements, count)

    def draw_B():
        return [[draw_cell(i, j) for j in range(6)] for i in range(2)]

    sites = [(i, j) for i in range(2) for j in range(6) if sb[i][j]]

    def state(B):
        br = component(B, "B")
        four, six, connected = _degree_geometry(br, 6 * q)
        deficit = 2 * q - gf2_rank(br)
        energy = (1000 * deficit + 100 * int(not connected)
                  + (80 * four + six / 8.0) / q)
        return energy, deficit, four, six, connected, br

    def quality(x, z):
        return tuple(value for t in range(len(x))
                     for value in (max(x[t], z[t]), x[t] + z[t]))

    seed = 420588031 + 1009 * idx
    for batch in range(3):
        pool, encountered = [], set()
        serial = 0

        def retain(B, result):
            nonlocal serial
            _, deficit, four, six, connected, br = result
            if deficit > 2 or not connected:
                return
            label = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
            if label in encountered:
                return
            encountered.add(label)
            serial += 1
            pool.append(((deficit, four, six), serial, clone(B), list(br)))
            pool.sort(key=lambda item: item[:2])
            del pool[10:]

        for restart in range(4):
            B = draw_B()
            current = state(B)
            retain(B, current)
            for step in range(200):
                i, j = rng.choice(sites)
                old = B[i][j]
                B[i][j] = draw_cell(i, j)
                proposed = state(B)
                delta = proposed[0] - current[0]
                temperature = 14.0 * (0.982 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if delta < 0 or step % 16 == 0:
                        retain(B, current)
                else:
                    B[i][j] = old
            retain(B, current)

        prepared = []
        for coarse, number, B, expected_b in pool:
            if _degree_low4(expected_b, [], 6 * q):
                continue
            db, nb = _degree_sample(
                expected_b, 6 * q, 4200201 + idx)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 588 or not 168 <= k <= 184:
                continue
            # Confirm cached actions against the complete final lift.
            actual_a = [
                sum(((hx[r] >> (j * 6 * q)) & mask) << (j * q)
                    for j in range(2))
                for r in range(q)]
            actual_b = [row & ((1 << (6 * q)) - 1)
                        for row in hz[:2 * q]]
            assert actual_a == ar and actual_b == expected_b
            if (_degree_low4(hx, hz, n)
                    or _degree_low4(hz, hx, n)):
                continue
            warning = max(0, 8 - min(da, db))
            key = (warning, coarse, max(na, nb), na + nb, number)
            prepared.append((key, A, B, hx, hz, k, db))
        prepared.sort(key=lambda item: item[0])

        coarse_probes = []
        for item in prepared[:4]:
            protection, _, B, hx, hz, k, db = item
            x = _degree_bp_probe(hz, hx, 588, seed, 64)
            z = _degree_bp_probe(hx, hz, 588, seed + 1, 64)
            # A capped component warning precedes the decoder benchmark;
            # beyond it, neither k*d^2/n nor optimistic distance is rewarded.
            key = (protection[0],) + quality(x, z) + protection[1:]
            coarse_probes.append((key, item, x, z))
        coarse_probes.sort(key=lambda item: item[0])

        best = None
        for _, item, cx, cz in coarse_probes[:2]:
            protection, _, B, hx, hz, k, db = item
            x = _degree_bp_probe(hz, hx, 588, seed + 1000003, 128)
            z = _degree_bp_probe(hx, hz, 588, seed + 1000004, 128)
            tx = tuple(a + b for a, b in zip(cx, x))
            tz = tuple(a + b for a, b in zip(cz, z))
            key = (protection[0],) + quality(tx, tz) + protection[1:]
            if best is None or key < best[0]:
                best = key, B, hx, hz, k, db, tx, tz
        if best is None:
            continue

        key, B, hx, hz, k, db, tx, tz = best
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx) <= 11
        assert max(row.bit_count() for row in hz) == 12
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert 6 * sum(row.bit_count() for row in hx + hz) == 25704
        assert lifted_k(G, A, B) == k
        assert 0.275 <= k / 588 < 0.325
        assert all(len(cell) == len(set(cell)) <= CELL_MAX
                   for grid in (A, B) for row in grid for cell in row)
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "A_class": (a_identity, a_serial), "B_card": card,
            "component_upper_bounds": (da, db),
            "decoder_surrogate": (tx, tz), "screen": key,
            "cnots": 25704,
        }

    raise ValueError("no gate-neutral rate-0.300 finalist passed verified screens")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without late draws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
# REFUTED: Equal k=59, weight ten and d2s=20 determine the measured score: current slots6 and11 each had one failed shot in one basis and s=7.104, while slots16 and29 had no failures in either basis and saturated at s=7.283.
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
