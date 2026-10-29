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
CONCEPT = ("C13 semidirect C3, action 3, order 39, shapes (2,3)x(2,3): fresh joint homogeneous-step geometries plus full-CSS logical-witness and shared-check exposure screening should improve worse-basis memory fitness beyond 7.1375 at unchanged check weight ten.")
GROUP = ("Frozen byte-for-byte under L1. This exact bed now measures 7.1375; slots 3, 5 and 14 had zero failures in both bases and saturated at 7.283. That supports refining its filling, not asserting a causal group advantage from related order-42 or order-44 beds.")
SHAPES = ("Frozen equal (2,3) components have 117 variables and 78 checks each: n=507, guaranteed k>=39 and rate 1/13. These shapes allow q<=46, while the unchanged group fixes q=39. Verified ranks retain k=59..63 below rate 0.125. Slot 4 mildly enriched Acol/Bcol1 blocks; slot 8 enriched Acol2/Bcol0 at 1.9x and Arow1/Brow0 at 1.3x. Six and one failed shots respectively do not establish an isolated block defect.")
FILLING = ("Retain twelve distinct-element doubletons, check weight ten and 28080 six-round CNOTs. Jointly canonicalize A/B homogeneous-step triples, including relative scale, and exclude the parent's offset-three/four pair architectures under explicit automorphism, inner-conjugation and side-swap symmetries. Twenty initial slots receive different joint geometries; larger budgets extend coset-phase contracts. Verified component screening precedes randomized full-CSS kernel searches modulo stabilizers, including mixed-sector witnesses and shared-opposite-check pair exposure. These are circuit-risk surrogates, not distance certificates or ASC hook-order predictions. Target slot 4's six Z failures concentrated on logicals 56-58 and slot 8's one X failure flipping six logicals, while retaining k-kb around twenty.")
FAMILY_NAME = "r010_frob13x3_23x23_jointstep_quantumexposure_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    if q != 39:
        raise ValueError("C13 semidirect C3 has order 39")
    els = [(a, b) for b in range(3) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def mul(g, h):
        return ((g[0] + pow(3, g[1] % 3, 13) * h[0]) % 13,
                (g[1] + h[1]) % 3)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-pow(3, bi, 13) * g[0]) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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


def NEW_CTX(rng, G, idx):
    """Optimize fresh joint geometries, then screen the full CSS quotient."""
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


def CELL(rng, G, side, i, j, ctx):
    """Emit the jointly designed, full-CSS-screened doubleton unchanged."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
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
