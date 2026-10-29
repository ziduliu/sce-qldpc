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
CONCEPT = ("For C37 semidirect C3, action 10, order 111, shapes (2,1)x(2,1), fresh transpose components selected by exact order-three invariant screening and cross-information-set witness recombination should produce five measured d30s readings >=23; this is a falsifiable search hypothesis, not a distance certificate.")
GROUP = ("Keep the group implementation and order byte-for-byte under L2. The order-111 wide bed reached fitness 99.370, while the tall bed reached 91.378. Its valid order-three action supplies the invariant-subspace screen. These measurements motivate improved tall fillings but do not establish an orientation or distance advantage.")
SHAPES = ("Target cell 0.200 with A=(2,1), B=(2,1): n=555, kb=111*(1-2)*(1-2)=111, guaranteed rate 0.200, and q<=119. Equal component maps have 111 variables and 222 checks; their transpose kernels have 222 coordinates and 111 checks. Parent measured minima were entirely sector 1, especially Acol1 across both B columns for X and Bcol1 across both A columns for Z. Fresh tall components expose corresponding kernels in sector 2; orientation alone is not an improvement claim. Verified assembled ranks enforce the target band.")
FILLING = ("Twenty four-term quotient-occupancy policies fill Arow0,Arow1,Brow0,Brow1 independently, giving row totals four and column totals eight, hence HX=HZ=12. Compile transpose actions from verified tall unit-cell lifts and fold along commuting order-three translations. Exact folded screening excludes invariant component words through physical weight 21, with explicitly tested fallback components. Recombine kernel words across information sets, give finalists equal deeper ordinary/folded budgets, and retain full-CSS logical probes and invariant-distinct screening. Fresh seeds never transpose stored parent labels. Witness readings remain upper bounds.")
FAMILY_NAME = "r020_m37r3s10_21x21_fresh_dual_norm3_recombine_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # Same metacyclic construction rule, resized for factor five.
    # y*x*y^-1=x^10; 10^3=1 mod 37 and 10 != 1 mod 37.
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (2, 1)               # (rb, cb)

from functools import lru_cache

# Four-term quotient occupancies: balanced, two-coset, and skew.
_QUOTIENT_MODELS = ((2, 1, 1), (2, 2, 0), (3, 1, 0))

# Entries prescribe Arow0,Arow1,Brow0,Brow1.
# Every policy optimizes fresh labels, never a transported parent filling.
_QUOTIENT_CARDS = (
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2),
    (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 2, 2),
    (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 2, 2),
    (0, 2, 0, 2), (0, 1, 1, 1), (0, 1, 1, 2),
    (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 2, 2),
    (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2),
    (1, 2, 1, 2), (2, 2, 2, 2),
)


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
    memory = []

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

        # Retain discoveries across independent information sets.
        # Every retained word remains in this same verified kernel.
        words = sorted(set(dependencies + memory), key=int.bit_count)[:48]
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

        short = sorted(set(words), key=int.bit_count)[:24]
        combined = set(words)
        for a, v in enumerate(short):
            for u in short[:a]:
                if v ^ u:
                    combined.add(v ^ u)
        # Component searches also recombine short triples. Full CSS
        # searches retain the cheaper pair pass and stabilizer descent.
        if not generators:
            small = short[:8]
            for a, v in enumerate(small):
                for b in range(a):
                    pair = v ^ small[b]
                    for c in range(b):
                        if pair ^ small[c]:
                            combined.add(pair ^ small[c])
        words = sorted(combined, key=int.bit_count)[:48]
        memory = words[:24]

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


def _quotient_cell(rng, G, model):
    """Distinct sampling in disjoint C3 quotient cosets."""
    counts = _QUOTIENT_MODELS[model]
    offset = rng.randrange(3)
    cell = []
    for b, count in enumerate(counts):
        pool = [g for g in G.elements if g[1] == (b + offset) % 3]
        cell.extend(rng.sample(pool, count))
    return cell


@lru_cache(maxsize=1)
def _verified_component_atlas(G):
    """Compile tall transpose components exclusively from verified lifts.

    Tall sector 1 is the first q columns. Sector-2 blocks begin at
    q,2q,3q,4q in (Arow,Brow) order. HZ exposes A^T and HX exposes
    B^T. XOR superposition preserves the verified cell action exactly.
    """
    q = G.q
    mask = (1 << q) - 1
    assert A_SHAPE == B_SHAPE == (2, 1)
    stub = [[[G.e]], [[G.e]]]
    tables, orbits = {}, {}
    generator = (0, 1)

    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g]], [[]]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 5*q
            source = hz if side == "A" else hx
            rows = [(row >> q) & mask for row in source[:q]]
            blocks.append(tuple(_component_columns(rows, q)))
        tables[side] = tuple(blocks)

        # A^T remains a left action and commutes with right translation;
        # B^T remains a right action and commutes with left translation.
        # These partitions describe invariant supports, not custom lifts.
        covered, partition = set(), []
        for g in G.elements:
            if G.idx(g) in covered:
                continue
            orbit, h = [], g
            for _ in range(3):
                orbit.append(G.idx(h))
                h = (G.mul(h, generator) if side == "A"
                     else G.mul(generator, h))
            assert G._key(h) == G._key(g)
            assert len(set(orbit)) == 3
            covered.update(orbit)
            partition.append(tuple(sorted(orbit)))
        assert len(covered) == q and len(partition)*3 == q
        orbits[side] = tuple(partition)
    return tables, orbits


def _fold_component(cols, orbits, q):
    """Verified syndromes for words constant on the order-three orbits.

    A folded support of weight t expands to a component word of weight 3t.
    Every syndrome is constant on output orbits; selecting one row from
    each orbit therefore preserves the kernel exactly on this subspace.
    """
    masks = [sum(1 << h for h in orbit) for orbit in orbits]
    folded = []
    for block in range(2):
        for orbit in orbits:
            syndrome = 0
            for h in orbit:
                syndrome ^= cols[block * q + h]
            compact = 0
            for r, (output_orbit, mask) in enumerate(zip(orbits, masks)):
                restricted = syndrome & mask
                assert restricted == 0 or restricted == mask
                if restricted:
                    compact |= 1 << r
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


def _component_pool(rng, G, side, models, quota):
    """Optimize fresh tall transpose kernels, not archived transpositions."""
    q = G.q
    tables, partitions = _verified_component_atlas(G)
    atlas, orbits = tables[side], partitions[side]
    assert len(models) == 2
    pool, seen = [], set()

    for attempt in range(quota):
        cells = [[_quotient_cell(rng, G, model)] for model in models]
        key = tuple(tuple(sorted(row[0])) for row in cells)
        if key in seen:
            continue
        seen.add(key)

        cols = [0] * (2*q)
        for i, row in enumerate(cells):
            for g in row[0]:
                contribution = atlas[G.idx(g)]
                for h, syndrome in enumerate(contribution):
                    cols[i*q + h] ^= syndrome

        rank = gf2_rank(cols)
        if rank < q-4 or _kernel_le4(cols):
            continue
        folded = _fold_component(cols, orbits, q)
        quick_clean = not _kernel_le4(folded)
        cycles = _component_four_cycles(cols)
        pool.append((not quick_clean, cycles, -rank, attempt,
                     cells, cols, folded))
        pool.sort(key=lambda item: item[:4])
        del pool[32:]

    clean, reserve = [], []
    for _, cycles, negative_rank, attempt, cells, cols, folded in pool:
        floor = _folded_screen7(folded)
        item = (floor, -cycles, -negative_rank, -attempt,
                cells, cols, folded)
        (clean if floor == 8 else reserve).append(item)
        if len(clean) >= 6:
            break

    reserve.sort(key=lambda item: item[:4], reverse=True)
    finalists = clean if clean else reserve[:6]
    ranked = []
    for item in finalists:
        floor, nc, rank, na, cells, cols, folded = item
        ordinary = _kernel_probe(cols, rng, trials=12)
        norm = _kernel_probe(folded, rng, trials=24)
        if floor < 8:
            norm = min(norm, floor)
        # Both readings are actual component witnesses, not the
        # exclusion sentinel returned by _folded_screen7.
        if ordinary > len(cols) or norm > len(folded):
            continue
        d = min(ordinary, 3*norm)
        ranked.append((d, nc, rank, na, cells, cols, folded,
                       ordinary, norm, floor))
    ranked.sort(key=lambda item: item[:4], reverse=True)

    deep = []
    for item in ranked[:3]:
        d, nc, rank, na, cells, cols, folded, ordinary, norm, floor = item
        ordinary = min(ordinary, _kernel_probe(cols, rng, trials=64))
        norm = min(norm, _kernel_probe(folded, rng, trials=48))
        if floor < 8:
            norm = min(norm, floor)

        # Check atlas assembly against a complete verified tall lift.
        stub = [[[G.e]], [[G.e]]]
        A, B = (cells, stub) if side == "A" else (stub, cells)
        hx, hz, _ = lifted_matrices(G, A, B)
        mask = (1 << q)-1
        if side == "A":
            rows = [((row >> q) & mask)
                    | (((row >> (3*q)) & mask) << q)
                    for row in hz[:q]]
        else:
            rows = [(row >> q) & ((1 << (2*q))-1)
                    for row in hx[:q]]
        assert _component_columns(rows, 2*q) == cols

        deep.append((min(ordinary, 3*norm), nc, rank, na, cells))
    deep.sort(key=lambda item: item[:4], reverse=True)
    return deep


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
    """Deterministic bank of fresh, invariant-distinct tall card designs."""
    sa = [[4], [4]]
    sb = [[4], [4]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert G.q == 111

    bank = []
    fingerprints = set()
    for idx in range(N_SLOTS):
        rng = random.Random(20261027 * 100019 + idx * 104729)
        card = _QUOTIENT_CARDS[idx % len(_QUOTIENT_CARDS)]
        chosen = None

        # New tall labels are optimized directly. Neither a saved filling
        # nor its transpose is used as a proposal for any requested slot.
        for retry in range(3):
            quota = 192 + 48 * retry + 16 * (idx // len(_QUOTIENT_CARDS))
            ap = _component_pool(rng, G, "A", card[:2], quota)
            bp = _component_pool(rng, G, "B", card[2:], quota)
            candidates = []

            for ai, a in enumerate(ap):
                for bi, b in enumerate(bp):
                    A, B = a[4], b[4]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (175 * n <= 1000 * k < 225 * n):
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
                    _component_columns(hz, n), rng, 16, stabilizers=hx)
                dz = _kernel_probe(
                    _component_columns(hx, n), rng, 16, stabilizers=hz)
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
                assert max(row.bit_count() for row in hx + hz) <= W_MAX
                fingerprints.add(fingerprint)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row) for row in A),
                    tuple(tuple(tuple(cell) for cell in row) for row in B),
                    k, score,
                )
                break
            if chosen is not None:
                break

        # A failed policy voids only its own slot, not the entire bank.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this independently optimized tall design without relabeling."""
    selected = _design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh tall target-rate design passed screening")
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
# REFUTED: Preferential norm-three screening with tested fallback guarantees five measured readings >=22 on the wide order-111 bed: measured slot 0 had d30s=20, although the other four measured readings were 23,22,22,22.
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
