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
CONCEPT = ("Cyclic C37, order 37, shapes (2,3)x(2,4): sum-free A binomials and odd-degree-three B columns selected by decoder stress should improve the cell-0.150 incumbent while retaining verified k=83 and X/Z check weights nine/ten.")
GROUP = ("Jump from nonabelian C13:C3 to the genuinely different cyclic C37. The size factor sixteen admits q<=37 and gives n=592. ord_37(2)=36 supports simple nontrivial-sector rank control, but complete verified ranks decide acceptance. This C37 (2,3)x(2,4) bed is absent from BEDS; the recorded high-rate C37 (1,3)x(1,5) result of about 1.3 is a warning, not evidence of a circuit advantage.")
SHAPES = ("Target cell 0.150 with unequal (2,3) and (2,4) components: 111 variables/74 checks and 148 variables/74 checks. kb=74 gives guaranteed rate 0.125; require complete verified k=83, rate 83/592=0.140203 and k-kb=9. Parent slot6 had one Z failure on logical57 with Acol2/Bcol2 enriched 1.6x; slot11 had one X failure on logical56 with Acol1/Bcol1 enriched 1.6x. Three measured siblings had no failures. These tiny samples do not establish a persistent isolated block defect.")
FILLING = ("A has six doubletons. B count rows are (0,2,2,2) and (3,1,1,1): every B column has odd degree three, both component row totals are six, HX weight is nine and HZ weight ten. Six-round CNOTs are29304, versus the parent's28080, but per-qubit extraction work falls. Canonical sum-free A gap cards and affine-distinct trinomial geometries prescribe designs before fresh voltage optimization. Helper-derived actions screen component ranks72/73, connectivity, exact kernels<=6, capped sampled protection warnings and balanced boundaries. Common-noise normalized-min-sum stress tests and independent holdouts select profiles; full verified CSS ranks, logicals<=4 and possible shared-check pair exposure select the final pair. These probes omit ASC ordering, noisy measurements and relay/LSD and are uncalibrated pL surrogates. Test retained rare failures at the higher k ceiling, not distance as fitness.")
FAMILY_NAME = "r015_C37_23x24_odd3_rank01_holdoutstress_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C37 requires order 37")
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

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def _retired_frob13_ctx(rng, G, idx):
    """Inactive previous-group experiment retained for provenance."""
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


def NEW_CTX(rng, G, idx):
    """Structural C37 cards, verified actions and independent stress holdouts."""
    import math
    import random
    from itertools import combinations, permutations

    q = G.q
    assert q == 37
    assert A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    assert pow(2, 36, q) == 1
    assert all(pow(2, d, q) != 1 for d in (12, 18))
    sa = [[2, 2, 2], [2, 2, 2]]
    sb = [[0, 2, 2, 2], [3, 1, 1, 1]]
    supports = {"A": sa, "B": sb}
    widths = {"A": 3*q, "B": 4*q}
    targets = {"A": 2*q - 2, "B": 2*q - 1}
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 10 <= W_MAX
    assert all(sum(column) == 3 for column in zip(*sb))

    def direction(value):
        value %= q
        return min(value, q - value)

    def gap_key(upper, lower):
        images = []
        for u in range(1, (q + 1)//2):
            pairs = [(direction(u*a), direction(u*b))
                     for a, b in zip(upper, lower)]
            images.append(tuple(sorted(pairs)))
            images.append(tuple(sorted((b, a) for a, b in pairs)))
        return min(images)

    # Three signed odd gaps have odd sum of magnitude at most33:
    # neither zero nor a multiple of37. Each row also has distinct gaps.
    # Enumerate structural cards rather than assigning independent draws.
    palette = (1, 3, 5, 7, 9, 11)
    cards = {}
    for upper in combinations(palette, 3):
        for subset in combinations(palette, 3):
            for lower in permutations(subset):
                if any(a == b for a, b in zip(upper, lower)):
                    continue
                identity = gap_key(upper, lower)
                representative = (tuple(upper), tuple(lower))
                old = cards.get(identity)
                if old is None or representative < old:
                    cards[identity] = representative
    book = sorted(cards.items())
    assert len(book) >= max(20, N_SLOTS)
    position = idx * len(book) // max(20, N_SLOTS)
    identity, agaps = book[position]
    bgaps = agaps[1]

    def triangle_key(t):
        points = (0, 1, t)
        return min(
            tuple(sorted(((x-a)*pow((b-a) % q, q-2, q)) % q
                         for x in points))
            for a in points for b in points if a != b)

    # Remove translation, scaling and term-order copies of a trinomial.
    # Arithmetic progressions are excluded because their repeated
    # differences force four-cycles inside the pure trinomial column.
    triangles = sorted({triangle_key(t) for t in range(3, 19)})
    triangle = triangles[idx % len(triangles)]
    assert len(set(triangle)) == 3
    differences = [(a-b) % q for a in triangle for b in triangle if a != b]
    assert len(set(differences)) == 6

    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        rows = []
        for cell_row in grid:
            packed = [0]*q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    for t, block in enumerate(actions[side][G.idx(g)]):
                        packed[t] ^= block << (j*q)
            rows.extend(packed)
        return rows

    def columns(rows, width):
        result = [0]*width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length()-1] |= tag
                row ^= bit
        return result

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                return False
            bit = row & -row
            first = bit.bit_length()-1
            row ^= bit
            while row:
                bit = row & -row
                parent[root(bit.bit_length()-1)] = root(first)
                row ^= bit
        return len({root(c) for c in range(width)}) == 1

    def new_cell(side, i, j):
        if side == "A":
            # Remove ordinary component column and common-row shifts.
            origin = 0 if i == 0 or j == 0 else rng.randrange(q)
            gap = agaps[i][j]
            return [(origin,), ((origin+gap) % q,)]
        if i == 0:
            if j == 0:
                return []
            origin = rng.randrange(q)
            gap = bgaps[j-1]
            return [(origin,), ((origin+gap) % q,)]
        if j == 0:
            scale = rng.randrange(1, q)
            return [((scale*t) % q,) for t in triangle]
        return [G.e]

    def draw(side):
        return [[new_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(supports[side])]

    def cycles(rows, width):
        adjacency = [0]*len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap*(overlap-1)//2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles_found = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles_found += (
                    neighbors & adjacency[j] & ((1 << j)-1)
                ).bit_count()
                lower ^= bit
        stars = sum(d*(d-1)*(d-2)//6
                    for d in (c.bit_count() for c in columns(rows, width)))
        # Exact C6 when C4=0; otherwise only a proposal heuristic.
        six = max(0, triangles_found-stars)
        return four, six

    def short_boundary(cols):
        """Exact component kernel<=6 exclusion and a boundary surrogate."""
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
        burden = 0
        loads = [0]*(len(cols)//q)
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return None
                    seen.add(value)
                    weight = value.bit_count()
                    if weight <= 4:
                        cost = (6-weight)**2
                        burden += cost
                        for block in {i//q, j//q, h//q}:
                            loads[block] += cost
        return max(loads)/q, burden/len(cols)

    def sampled_words(cols, side):
        """Actual kernel witnesses; sampled minima remain upper bounds."""
        srng = random.Random(37015083 + 1009*idx + (side == "B"))
        order = list(range(len(cols)))
        minimum = len(cols)+1
        short = set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 12:
                    short.add(word)

        for _ in range(32):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
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
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)
            for word in sorted(basis, key=int.bit_count)[:6]:
                for _ in range(2):
                    best = word
                    for other in basis:
                        proposal = word ^ other
                        if proposal and proposal.bit_count() < best.bit_count():
                            best = proposal
                    if best == word:
                        break
                    word = best
                    offer(word)
        return minimum, len(short)/len(cols)

    def stress(rows, side, shots, stream):
        """Classical component BSC stress, NOT ASC or relay-BP fitness."""
        import numpy as np

        width = widths[side]
        nr = len(rows)
        maximum = max(row.bit_count() for row in rows)
        variable_lists = [[] for _ in range(width)]
        edge_variables, locations = [], []
        groups = []
        matrix = np.zeros((nr, width), dtype=np.uint8)
        for r, row in enumerate(rows):
            edges = []
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                edge = len(edge_variables)
                edges.append(edge)
                variable_lists[c].append(edge)
                edge_variables.append(c)
                locations.append(r*maximum + len(edges)-1)
                matrix[r, c] = 1
                row ^= bit
            groups.append(edges)

        ne = len(edge_variables)
        checks = np.full((nr, maximum), ne, dtype=np.int32)
        for r, edges in enumerate(groups):
            checks[r, :len(edges)] = edges
        degrees = np.array(list(map(len, variable_lists)), dtype=float)
        variables = np.full(
            (width, int(degrees.max())), ne, dtype=np.int32)
        for c, edges in enumerate(variable_lists):
            variables[c, :len(edges)] = edges
        ev = np.asarray(edge_variables, dtype=np.int32)
        locations = np.asarray(locations, dtype=np.int32)
        positions = np.arange(maximum)[None, None, :]
        channel = np.random.default_rng(
            37015091 + 1009*idx + 7919*(side == "B") + 1000003*stream)
        uniforms = channel.random((shots, width))
        outcomes = []

        for probability in (0.025, 0.045):
            rates = probability*degrees/3.0
            magnitude = np.log((1.0-rates)/rates)
            errors = uniforms < rates[None, :]
            prior = np.where(errors, -magnitude[None, :],
                             magnitude[None, :])
            messages = prior[:, ev].copy()
            done = np.zeros(shots, dtype=bool)
            wrong = np.zeros(shots, dtype=bool)
            effort = 0
            for iteration in range(1, 33):
                padded = np.concatenate(
                    (messages, np.full((shots, 1), 1.0e6)), axis=1)
                incoming = padded[:, checks]
                absolute = np.abs(incoming)
                minimum_at = absolute.argmin(axis=2)
                minimum = absolute.min(axis=2)
                second = np.partition(absolute, 1, axis=2)[:, :, 1]
                signs = np.where(incoming >= 0, 1.0, -1.0)
                excluded = np.where(
                    positions == minimum_at[:, :, None],
                    second[:, :, None], minimum[:, :, None])
                outgoing = (0.8*signs.prod(axis=2)[:, :, None]
                            * signs * excluded)
                check_messages = outgoing.reshape(shots, -1)[:, locations]
                padded_checks = np.concatenate(
                    (check_messages, np.zeros((shots, 1))), axis=1)
                posterior = prior + padded_checks[:, variables].sum(axis=2)
                proposal = posterior[:, ev] - check_messages
                messages = np.clip(
                    0.30*messages + 0.70*proposal, -30.0, 30.0)
                if iteration % 4 == 0:
                    hard = (posterior < 0).astype(np.uint8)
                    syndrome = (hard @ matrix.T) & 1
                    resolved = (~done) & ~syndrome.any(axis=1)
                    wrong[resolved] = hard[resolved].any(axis=1)
                    effort += iteration*int(resolved.sum())
                    done |= resolved
                    if done.all():
                        break
            unresolved = int((~done).sum())
            effort += 33*unresolved
            outcomes.append((int((wrong | ~done).sum()),
                             unresolved, effort))
        return (outcomes[0][0], outcomes[1][0],
                outcomes[0][1]+outcomes[1][1],
                outcomes[0][2]+outcomes[1][2])

    mutable = {"A": [(1, 1), (1, 2)],
               "B": [(0, 1), (0, 2), (0, 3), (1, 0)]}

    def portfolio(side):
        width = widths[side]
        pool, seen = [], set()
        serial = 0

        def energy(cost):
            return (90*cost[0] + cost[1]/5.0)/q

        def remember(grid, rows, cost):
            nonlocal serial
            if len(pool) >= 8 and cost > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            if gf2_rank(rows) != targets[side]:
                return
            if not connected(rows, width):
                return
            serial += 1
            pool.append((cost, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(5):
            grid = draw(side)
            rows = component(grid, side)
            current = cycles(rows, width)
            remember(grid, rows, current)
            for step in range(240):
                i, j = rng.choice(mutable[side])
                old = grid[i][j]
                grid[i][j] = new_cell(side, i, j)
                proposed_rows = component(grid, side)
                proposed = cycles(proposed_rows, width)
                delta = energy(proposed)-energy(current)
                temperature = 10.0*(0.985**step) + 0.06
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    rows, current = proposed_rows, proposed
                    if delta < 0 or step % 24 == 0:
                        remember(grid, rows, current)
                else:
                    grid[i][j] = old
            remember(grid, rows, current)

        finalists = []
        for coarse, number, grid, rows in pool:
            cols = columns(rows, width)
            boundary = short_boundary(cols)
            if boundary is None:
                continue
            distance, multiplicity = sampled_words(cols, side)
            probe = stress(rows, side, 96, 0)
            # Twelve is a sampled warning floor, not a certificate.
            # Beyond it, decoding behavior and exposure decide.
            key = ((max(0, 12-distance),) + probe
                   + boundary + (multiplicity, coarse[0]/q, coarse[1]/q))
            finalists.append((key, number, grid, rows, probe))
        finalists.sort(key=lambda item: item[:2])

        heldout = []
        for key, number, grid, rows, probe in finalists[:4]:
            independent = stress(rows, side, 192, 1)
            combined = tuple(a+b for a, b in zip(probe, independent))
            refined = key[:1] + combined + key[5:]
            heldout.append((refined, number, grid, rows))
        heldout.sort(key=lambda item: item[:2])
        return heldout

    def short_logical(checks, stabilizers, width):
        """Exact stabilizer-quotiented logical exclusion through weight four."""
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

        cols = columns(checks, width)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif nontrivial(word ^ old):
                return True
        for i, syndrome in enumerate(cols):
            for j in range(i):
                value = syndrome ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(value)
                if old is None:
                    seen[value] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    def pair_exposure(checks, opposite, width):
        """Possible spread-pair boundaries, not ASC-certified hooks."""
        cols = columns(checks, width)
        loads = [0]*(width//q)
        opportunities = [0]*(width//q)
        burden = count = 0
        for row in opposite:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length()-1)
                row ^= bit
            for t, i in enumerate(sites):
                for j in sites[:t]:
                    count += 1
                    opportunities[i//q] += 1
                    opportunities[j//q] += 1
                    boundary = (cols[i] ^ cols[j]).bit_count()
                    if boundary <= 4:
                        cost = (5-boundary)**2
                        burden += cost
                        loads[i//q] += cost
                        loads[j//q] += cost
        peak = max(a/max(1, b) for a, b in zip(loads, opportunities))
        return peak, burden/max(1, count)

    for batch in range(3):
        apool, bpool = portfolio("A"), portfolio("B")
        options = []
        for ak, at, A, ar in apool:
            for bk, bt, B, br in bpool:
                paired = tuple(v for t in range(len(ak))
                               for v in (max(ak[t], bk[t]), ak[t]+bk[t]))
                options.append((paired, at, bt, A, B, ar, br))
        options.sort(key=lambda item: item[:3])
        winner = None

        for paired, at, bt, A, B, ar, br in options[:6]:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 592 or k != 83:
                continue
            if not connected(hx+hz, n):
                continue
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            actual_a = [
                sum(((hx[(i*4)*q+t] >> (j*4*q)) & mask) << (j*q)
                    for j in range(3))
                for i in range(2) for t in range(q)]
            actual_b = [row & ((1 << (4*q))-1) for row in hz[:2*q]]
            assert actual_a == ar and actual_b == br
            px = pair_exposure(hz, hx, n)
            pz = pair_exposure(hx, hz, n)
            exposure = (max(px[0], pz[0]), max(px[1], pz[1]),
                        px[1]+pz[1])
            key = paired[:10] + exposure + paired[10:] + (at, bt)
            if winner is None or key < winner[0]:
                winner = (key, A, B, hx, hz, px, pz)

        if winner is not None:
            key, A, B, hx, hz, px, pz = winner
            assert css_ok(hx, hz)
            assert lifted_k(G, A, B) == 83
            assert max(row.bit_count() for row in hx) == 9
            assert max(row.bit_count() for row in hz) == 10 <= W_MAX
            assert 6*sum(row.bit_count() for row in hx+hz) == 29304
            assert 0.125 <= 83/592 < 0.175
            assert [[len(cell) for cell in row] for row in A] == sa
            assert [[len(cell) for cell in row] for row in B] == sb
            assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "architecture": (identity, triangle),
                    "k": 83, "screen": key, "cnots": 29304,
                    "pair_exposure": (px, pz)}

    raise ValueError("no verified k=83 C37 stress-screened filling")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected C37 cell without subsequent draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
# REFUTED: First-pass syndrome settlement guarantees logical success: order-39 slots6 and11 settled all200000 shots at stage1 in both bases, but each had one scored logical failure.
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
