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
CONCEPT = ("For C13 semidirect C3, action 3, order 39, shapes (2,3)x(2,3), degree-distinguished support pairs and exact order-three invariant-kernel screening should produce five measured distance readings of at least 26 in cell 0.100; this is a falsifiable search hypothesis.")
GROUP = ("Jump to the genuinely different order-39 Frobenius group: 3^3=1 mod 13 and 3!=1. Its measured (2,3)x(2,3) bed reached fitness 45.2, while the related order-111 Frobenius bed reached 99.4. Transfer of the norm-screening advantage is unverified; fresh labels are required on this archived bed.")
SHAPES = ("Target cell 0.100 with equal (2,3) components of 117 variables and 78 checks: n=507, kb=39, guaranteed rate 1/13. The shape permits q<=46; this group fixes q=39. Parent weight-20..23 minima were entirely sector 1, often one A-column or B-column across its opposite-side blocks. Two check rows replace both single-row components; verified full ranks require 0<k/n<0.125.")
FILLING = ("Twenty prescribed support-profile pairs vary augmentation ranks and column degrees, with cells of one to four distinct terms. Most profiles have row sums<=7 and column sums<=5; exceptional profiles are paired only where both LP check-weight sums remain<=12. Compile actions from verified unit lifts, screen ordinary kernels through four and order-three folded kernels through seven, independently deepen component witnesses, then screen assembled CSS logicals and invariant fingerprints. Restricted and randomized readings are selection heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r010_frob13c3_23x23_supportcards_norm3_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # C13:C3: x^13=y^3=1, y*x*y^-1=x^3.
    if not valid_q(q):
        raise ValueError("C13:C3 requires order 39")
    m, r, s = 13, 3, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m) * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m) * a) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)

from functools import lru_cache

# Row-major 2x3 profiles. Counts change augmentation rank and degrees,
# not merely the gauge or group-automorphism orbit of a fixed labeling.
_LOW_PROFILES = (
    (2, 2, 2, 2, 2, 2),
    (3, 2, 2, 1, 2, 2),
    (3, 2, 2, 2, 1, 2),
    (3, 2, 2, 2, 2, 2),
    (3, 3, 1, 1, 1, 3),
    (3, 2, 1, 1, 2, 3),
    (3, 2, 2, 2, 3, 2),
    (3, 2, 2, 3, 2, 2),
    (4, 2, 2, 1, 2, 2),
)
_LOW_CARDS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 1), (1, 2), (1, 3), (1, 6),
    (2, 2), (2, 3), (2, 6),
    (3, 3), (3, 6), (4, 6), (6, 6),
)


def _profile_grid(profile):
    return [list(profile[:3]), list(profile[3:])]


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _small_logical_3(checks, stabilizers, width):
    """Exactly detect any non-stabilizer logical of weight one to three."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]

    def in_span(v):
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                return False
            v ^= basis[p]
        return True

    syndromes = _component_columns(checks, width)
    singles = {}
    for i, syndrome in enumerate(syndromes):
        v = 1 << i
        if syndrome == 0 and not in_span(v):
            return True
        if syndrome in singles:
            if not in_span(v ^ singles[syndrome]):
                return True
        else:
            singles[syndrome] = v

    # After the singleton test, equal-syndrome singleton representatives
    # differ only by stabilizers, so one representative per syndrome suffices.
    for i in range(width):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            v = vi | (1 << j)
            if syndrome == 0 and not in_span(v):
                return True
            if syndrome in singles:
                if not in_span(v ^ singles[syndrome]):
                    return True
    return False


def _row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _outside_span(v, basis):
    while v:
        p = v.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        v ^= old
    return False


def _kernel_le4(cols):
    """Exact classical kernel test using all singleton/pair syndromes."""
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        support = 1 << i
        old = seen.get(syndrome)
        if old is not None:
            return True
        seen[syndrome] = support
    for i, syndrome in enumerate(cols):
        for j in range(i):
            s = syndrome ^ cols[j]
            v = (1 << i) | (1 << j)
            old = seen.get(s)
            if old is not None and old != v:
                return True
            seen[s] = v
    return False


def _kernel_probe(cols, rng, trials, stabilizers=()):
    """Return the weight of a found kernel/logical witness.

    Dependencies come from the supplied verified column syndromes.
    Pair reductions preserve the kernel. Stabilizer reductions preserve
    the logical class, which is explicitly tested before counting it.
    A sentinel means no suitable witness was found, not a distance proof.
    """
    width = len(cols)
    best = width + 1
    basis = _row_basis(stabilizers)
    generators = list(stabilizers)
    order = list(range(width))

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        dependencies = []
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, support)
                    break
                syndrome ^= old[0]
                support ^= old[1]
            if not syndrome:
                dependencies.append(support)

        # A broad dependency gives a fallback genuine logical witness.
        if generators:
            for v in dependencies:
                if _outside_span(v, basis):
                    best = min(best, v.bit_count())
                    break

        words = sorted(dependencies, key=int.bit_count)[:48]
        for _ in range(3):
            changed = False
            for a in range(len(words)):
                v = words[a]
                weight = v.bit_count()
                for b in range(len(words)):
                    if a == b:
                        continue
                    z = v ^ words[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                words[a] = v
            if not changed:
                break

        for v in words:
            if generators and not _outside_span(v, basis):
                continue
            weight = v.bit_count()
            if generators:
                for _ in range(3):
                    changed = False
                    for row in generators:
                        z = v ^ row
                        zw = z.bit_count()
                        if zw < weight:
                            v, weight = z, zw
                            changed = True
                    if not changed:
                        break
            if v:
                best = min(best, weight)
    return best


def _quotient_cell(rng, G, count, style):
    """Count-aware quotient occupancies with fresh independent labels."""
    patterns = {
        1: ((1, 0, 0),),
        2: ((1, 1, 0), (2, 0, 0)),
        3: ((1, 1, 1), (2, 1, 0)),
        4: ((2, 1, 1), (2, 2, 0), (3, 1, 0)),
    }
    choices = patterns[count]
    counts = choices[style % len(choices)]
    offset = rng.randrange(3)
    direction = rng.choice((-1, 1))
    cell = []
    for b, amount in enumerate(counts):
        coset = (offset + direction * b) % 3
        pool = [g for g in G.elements if g[1] == coset]
        cell.extend(rng.sample(pool, amount))
    assert len(cell) == count and len(set(cell)) == count
    return cell


@lru_cache(maxsize=1)
def _verified_component_atlas(G):
    """Compile linear component contributions from verified unit-cell lifts.

    No regular-representation matrix is implemented here. XOR superposition
    of these verified contributions is precisely F2-linear cell filling.
    Internal unit-cell references need not satisfy final-slot support rules.
    """
    q = G.q
    mask = (1 << q) - 1
    # A 1x1 unit lift isolates one block action independently of the
    # final protograph dimensions. Its contributions are F2-linear.
    stub = [[[G.e]]]
    tables = {}
    orbits = {}
    generator = (0, 1)

    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g]]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            rows = [row & mask for row in source[:q]]
            blocks.append(tuple(_component_columns(rows, q)))
        tables[side] = tuple(blocks)

        # Left-lifted A commutes with right translation; right-lifted B
        # commutes with left translation. These define orbit partitions,
        # not a second implementation of either lifted parity matrix.
        covered = set()
        partition = []
        for g in G.elements:
            if G.idx(g) in covered:
                continue
            orbit = []
            h = g
            for _ in range(3):
                orbit.append(G.idx(h))
                h = (G.mul(h, generator) if side == "A"
                     else G.mul(generator, h))
            assert G._key(h) == G._key(g)
            assert len(set(orbit)) == 3
            covered.update(orbit)
            partition.append(tuple(sorted(orbit)))
        assert len(covered) == q and len(partition) * 3 == q
        orbits[side] = tuple(partition)
    return tables, orbits


def _fold_component(cols, orbits, q):
    """Restrict the verified 2q-by-3q component to orbit-constant words.

    Three input protograph blocks and two output blocks are folded.
    Weight t expands to physical weight 3t. Equivariance is checked
    on every output orbit rather than assumed from a separate lift.
    """
    masks = [sum(1 << h for h in orbit) for orbit in orbits]
    orbit_count = len(orbits)
    qmask = (1 << q) - 1
    folded = []
    assert len(cols) == 3 * q
    for block in range(3):
        for orbit in orbits:
            syndrome = 0
            for h in orbit:
                syndrome ^= cols[block * q + h]
            compact = 0
            for output_block in range(2):
                part = (syndrome >> (output_block * q)) & qmask
                for r, mask in enumerate(masks):
                    restricted = part & mask
                    assert restricted == 0 or restricted == mask
                    if restricted:
                        compact |= 1 << (output_block * orbit_count + r)
            folded.append(compact)
    return folded


def _folded_screen7(cols):
    """Find a folded kernel word through seven, or exactly exclude them.

    The returned value below eight is a found witness weight, not
    necessarily the minimum. Eight means exhaustive exclusion through
    seven, not a witness at eight and not a quantum-distance certificate.
    """
    width = len(cols)
    bits = [1 << i for i in range(width)]
    seen = {0: 0}

    def insert(syndrome, support):
        old = seen.get(syndrome)
        if old is not None:
            difference = old ^ support
            if difference:
                return difference.bit_count()
        else:
            seen[syndrome] = support
        return None

    for i, syndrome in enumerate(cols):
        hit = insert(syndrome, bits[i])
        if hit is not None:
            return hit
    for i in range(1, width):
        for j in range(i):
            hit = insert(cols[i] ^ cols[j], bits[i] | bits[j])
            if hit is not None:
                return hit
    for i in range(2, width):
        for j in range(1, i):
            syndrome = cols[i] ^ cols[j]
            support = bits[i] | bits[j]
            for h in range(j):
                hit = insert(syndrome ^ cols[h], support | bits[h])
                if hit is not None:
                    return hit

    # Collisions between subsets of sizes <=3 cover every weight <=6.
    # A 4+3 partition covers every remaining weight-seven kernel word.
    for i in range(3, width):
        for j in range(2, i):
            syndrome = cols[i] ^ cols[j]
            for h in range(1, j):
                triple_syndrome = syndrome ^ cols[h]
                for t in range(h):
                    old = seen.get(triple_syndrome ^ cols[t])
                    if old is not None:
                        support = bits[i] | bits[j] | bits[h] | bits[t]
                        return (support ^ old).bit_count()
    return 8


def _component_pool(rng, G, side, profile, style, quota):
    """Optimize a 2x3 component using verified ordinary and folded kernels."""
    q = G.q
    tables, partitions = _verified_component_atlas(G)
    atlas = tables[side]
    orbits = partitions[side]
    pool = []
    seen = set()

    for attempt in range(quota):
        # Balanced and concentrated coset occupancies are separate
        # candidate architectures, not transported copies of a labeling.
        palette = (style + attempt // 16) % 3
        cells = [
            [_quotient_cell(rng, G, profile[3*i+j],
                            palette + i + 2*j)
             for j in range(3)]
            for i in range(2)
        ]
        key = tuple(tuple(sorted(cell))
                    for row in cells for cell in row)
        if key in seen:
            continue
        seen.add(key)

        cols = [0] * (3 * q)
        for i, row in enumerate(cells):
            for j, cell in enumerate(row):
                for g in cell:
                    contribution = atlas[G.idx(g)]
                    for h, syndrome in enumerate(contribution):
                        cols[j*q+h] ^= syndrome << (i*q)

        rank = gf2_rank(cols)
        if rank < 2*q - 8 or _kernel_le4(cols):
            continue
        folded = _fold_component(cols, orbits, q)
        quick_clean = not _kernel_le4(folded)
        cycles = _component_four_cycles(cols)
        pool.append((not quick_clean, cycles, -rank, attempt,
                     cells, cols, folded))
        pool.sort(key=lambda item: item[:4])
        del pool[20:]

    clean = []
    reserve = []
    for _, cycles, negative_rank, attempt, cells, cols, folded in pool:
        floor = _folded_screen7(folded)
        candidate = (floor, -cycles, -negative_rank, -attempt,
                     cells, cols, folded)
        if floor == 8:
            clean.append(candidate)
        else:
            reserve.append(candidate)
        if len(clean) >= 5:
            break

    reserve.sort(key=lambda item: item[:4], reverse=True)
    # Keep a tested reserve alongside clean candidates: exact exclusion
    # on one restricted subspace does not imply a better full component.
    finalists = clean[:5] + reserve[:2]
    ranked = []
    for floor, negative_cycles, rank, negative_attempt, cells, cols, folded in finalists:
        ordinary_d = _kernel_probe(cols, rng, trials=48)
        folded_d = _kernel_probe(folded, rng, trials=64)
        if floor < 8:
            folded_d = min(folded_d, floor)
        if ordinary_d > len(cols) or folded_d > len(folded):
            continue
        d = min(ordinary_d, 3 * folded_d)
        ranked.append((d, negative_cycles, rank, negative_attempt, cells))

    ranked.sort(key=lambda item: item[:4], reverse=True)
    return ranked[:3]


def _walk_signature(rows, width):
    """Permutation-invariant weighted row-intersection graph statistics.

    These are graph fingerprints, not logical-operator counts. Different
    fingerprints exclude Tanner isomorphism; equal ones may be false
    duplicates and are conservatively discarded.
    """
    cols = _component_columns(rows, width)
    adjacency = [{} for _ in rows]
    for syndrome in cols:
        checks = []
        while syndrome:
            low = syndrome & -syndrome
            checks.append(low.bit_length() - 1)
            syndrome ^= low
        for a, i in enumerate(checks):
            for j in checks[:a]:
                adjacency[i][j] = adjacency[i].get(j, 0) + 1
                adjacency[j][i] = adjacency[j].get(i, 0) + 1

    overlap_hist = {}
    edges = 0
    for i, neighbors in enumerate(adjacency):
        for j, overlap in neighbors.items():
            if j < i:
                overlap_hist[overlap] = overlap_hist.get(overlap, 0) + 1
                edges += 1
    overlap_hist[0] = len(rows) * (len(rows) - 1) // 2 - edges

    triangle_walks = 0
    fourth_walks = 0
    for i, neighbors in enumerate(adjacency):
        walks = {}
        for j, ij in neighbors.items():
            for h, jh in adjacency[j].items():
                walks[h] = walks.get(h, 0) + ij * jh
        triangle_walks += sum(
            ij * walks.get(j, 0) for j, ij in neighbors.items())
        fourth_walks += sum(value * value for value in walks.values())

    return (
        tuple(sorted(row.bit_count() for row in rows)),
        tuple(sorted(c.bit_count() for c in cols)),
        tuple(sorted(overlap_hist.items())),
        triangle_walks, fourth_walks,
    )


def _code_signature(hx, hz, n, k):
    # Also invariant under a global X/Z exchange.
    return (k, tuple(sorted((
        _walk_signature(hx, n), _walk_signature(hz, n)))))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Optimized low-rate support cards with distinct Tanner invariants."""
    assert G.q == 39
    bank = []
    fingerprints = set()
    for idx in range(N_SLOTS):
        rng = random.Random(39023191 + idx * 104729)
        pa, pb = _LOW_CARDS[idx % len(_LOW_CARDS)]
        ta, tb = _LOW_PROFILES[pa], _LOW_PROFILES[pb]
        sa, sb = _profile_grid(ta), _profile_grid(tb)
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        chosen = None

        # Each additional slot independently optimizes labels and coset
        # palettes; no stored code is relabeled to fill another slot.
        for retry in range(4):
            quota = 96 + 32 * retry + 16 * (idx // len(_LOW_CARDS))
            ap = _component_pool(rng, G, "A", ta, idx + retry, quota)
            bp = _component_pool(rng, G, "B", tb, idx + retry + 1, quota)
            candidates = []

            for ai, a in enumerate(ap):
                for bi, b in enumerate(bp):
                    A, B = a[4], b[4]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (0 < k and 8 * k < n):
                        continue
                    d = min(a[0], b[0])
                    score = (k * d * d, d, a[1] + b[1], -ai, -bi)
                    candidates.append((score, A, B, hx, hz, n, k, d))

            candidates.sort(key=lambda item: item[0], reverse=True)
            measured = []
            for candidate in candidates[:4]:
                _, A, B, hx, hz, n, k, component_d = candidate
                if _small_logical_3(hx, hz, n):
                    continue
                if _small_logical_3(hz, hx, n):
                    continue

                # Unlike component-only selection, these probes include
                # both LP sectors and explicitly exclude stabilizers.
                dx = _kernel_probe(
                    _component_columns(hz, n), rng, 12, stabilizers=hx)
                dz = _kernel_probe(
                    _component_columns(hx, n), rng, 12, stabilizers=hz)
                d = min(component_d, dx, dz)
                score = (k * d * d, d, min(dx, dz), -abs(dx - dz))
                measured.append((score, candidate))

            measured.sort(key=lambda item: item[0], reverse=True)
            for score, candidate in measured:
                _, A, B, hx, hz, n, k, _ = candidate
                fingerprint = _code_signature(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert n == 507 and 0 < k and 8*k < n
                assert max(row.bit_count() for row in hx + hz) <= W_MAX
                assert all(0 < len(cell) <= CELL_MAX
                           and len(cell) == len(set(cell))
                           for grid in (A, B) for row in grid for cell in row)
                fingerprints.add(fingerprint)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row) for row in A),
                    tuple(tuple(tuple(cell) for cell in row) for row in B),
                    k, score,
                )
                break
            if chosen is not None:
                break

        # A failed recipe voids only its own slot, not the cached bank.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this support card's independently selected low-rate code."""
    selected = _design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh low-rate filling passed this support card")
    A, B, k, score = selected
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "probe_score": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: Odd triple upgrades and exact exclusion of weight<=3 logicals at Z9, (2,7)x(2,8), remove the distance-four plateau: all five measured parent codes still had d30s=4, with thousands of sector-1 weight-four witnesses.
# REFUTED: Four-term quotient cards with the existing ordinary-component and mixed-sector probes exclude sector-1 weight-21 logicals: current-parent measured slots 7,24,31,36,51 all retained d30s=21 and sector-1 minima.
# REFUTED: Norm-screened selection on C37:C3 ensures every measured reading exceeds 21: current-parent slot 0 fell from d2s=23 to d30s=20 and retained 68 sector-1 Z witnesses at weight 20.
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
