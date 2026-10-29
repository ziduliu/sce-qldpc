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
CONCEPT = ("C17 x C17, order 289, shapes (1,1)x(1,1): five-term supports sharing three or four eight-dimensional Frobenius character orbits should yield fresh k=48 or k=64 codes whose jointly screened logical readings improve the cell-0.100 frontier fitness 74.6.")
GROUP = ("L3 jump from the nonabelian order-40 parent to the abelian order-289 group. This existing single-cell bed measured k=32 with distance readings 34..46. Odd order makes F2[G] semisimple; the 288 nonidentity characters form 36 Frobenius orbits of size eight. Target additional shared spectral zeros rather than copy any recorded support. Complete ranks are checked on scaffold-generated matrices.")
SHAPES = ("Equal (1,1) sides give n=578, kb=0, and equal 289-variable/289-check constituents. The window allows q<=299 and the fixed custom group uses q=289. Verified k=48 or 64 gives rates 0.0830 or 0.1107, both in the open-ended 0.100 cell. Redirect from the parent's 0.500 plateau: its weight-five X witnesses occupied sector-one Acol/Bcol blocks, particularly Acol1/Bcol2. Replace that containment architecture with two single-cell spectral supports.")
FILLING = ("Enumerate normalized five-term supports with zeros on both coordinate-character orbits and at least one additional orbit. Pair supports with exactly three or four shared zero orbits, then verify predicted k=48/64 using both full scaffold ranks. Twenty policies cross five support-geometry preferences with four additional-character preferences; broader fresh-library proposals are reserve choices. Both CSS row weights are ten. Exactly exclude logicals through four, reject repeated uncoloured Tanner invariants, and select actual stabilizer-aware logical witnesses with independent rescreening. A capped search reading is a heuristic upper bound, never a distance certificate. Historical containment banks remain inactive.")
FAMILY_NAME = "r010_C17xC17_11x11_threeorbit_five_support_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 289

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C17 x C17 requires order 289")
    els = [(a, b) for b in range(17) for a in range(17)]

    def red(g):
        return (g[0] % 17, g[1] % 17)

    def mul(g, h):
        return ((g[0] + h[0]) % 17, (g[1] + h[1]) % 17)

    def inv(g):
        return ((-g[0]) % 17, (-g[1]) % 17)

    def idx(g):
        a, b = red(g)
        return a + 17*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 1)               # (ra, ca)
B_SHAPE = (1, 1)               # (rb, cb)


def _four_cycle_count(rows):
    """Four-cycles in one Tanner graph, using the verified bit rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _has_small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight 1..4.

    Enumerate supports of size at most two. Equal syndromes yield
    every commuting support of size at most four. Test differences
    against the opposite stabilizer row space, not merely the kernel.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(v):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    syndrome = [0] * n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndrome[low.bit_length() - 1] ^= bit
            row ^= low

    seen = {0: 0}
    for i, s in enumerate(syndrome):
        v = 1 << i
        old = seen.get(s)
        if old is not None:
            if outside(v ^ old):
                return True
        else:
            seen[s] = v

    for i in range(n):
        vi = 1 << i
        si = syndrome[i]
        for j in range(i):
            s = si ^ syndrome[j]
            v = vi | (1 << j)
            old = seen.get(s)
            if old is not None:
                if outside(v ^ old):
                    return True
            else:
                seen[s] = v
    return False


@__import__("functools").lru_cache(maxsize=1)
def _shared_cokernel_bank(G):
    """Pure, cached optimization of fresh rank-aware design cards."""
    from collections import Counter

    q = G.q
    mask = (1 << q) - 1
    dummy_a = [[[G.e] for _ in range(3)]]
    dummy_b = [[[G.e] for _ in range(4)]]

    def component_rows(cells, side):
        if side == "A":
            hx, hz, n = lifted_matrices(G, cells, dummy_b)
            rows = []
            for row in hx[:q]:
                rows.append(sum(
                    ((row >> (4*j*q)) & mask) << (j*q)
                    for j in range(3)))
            return rows, 3*q
        hx, hz, n = lifted_matrices(G, dummy_a, cells)
        bm = (1 << (4*q)) - 1
        return [row & bm for row in hz[:q]], 4*q

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def dependency_space(rows):
        """Canonical exact space of dependencies between verified rows."""
        pivots, dependencies = {}, []
        for r, original in enumerate(rows):
            row, word = original, 1 << r
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (row, word)
                    break
                row ^= old[0]
                word ^= old[1]
            if not row:
                dependencies.append(word)

        basis = {}
        for word in dependencies:
            while word:
                p = word.bit_length() - 1
                if p in basis:
                    word ^= basis[p]
                else:
                    basis[p] = word
                    break
        for p in sorted(basis):
            for other in sorted(basis):
                if other > p and ((basis[other] >> p) & 1):
                    basis[other] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    def fingerprint(hx, hz, n):
        """Permutation-invariant combined, uncoloured incidence signature.

        Different signatures imply different incidence graphs. Equal
        signatures are conservatively rejected, not asserted isomorphic.
        """
        rows = list(hx) + list(hz)
        adj = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adj[i][j] = overlap
                    adj[j][i] = overlap
        profiles = []
        for r, neighbors in enumerate(adj):
            items = list(neighbors.items())
            triangles = 0
            for a, (s, ws) in enumerate(items):
                for t, wt in items[:a]:
                    triangles += 2*ws*wt*adj[s].get(t, 0)
            profiles.append((
                rows[r].bit_count(),
                tuple(sorted(Counter(neighbors.values()).items())),
                sum(w*w for w in neighbors.values()),
                triangles))
        degrees = [0] * n
        for row in rows:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return (tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(degrees).items())))

    # A single-block matrix is extracted from a full, verified LP lift.
    # Right translations merely permute its columns and are removed by
    # the column-incidence signature before blocks enter a class.
    classes = {}
    gauges = ([G.e] + [(a, 0) for a in range(1, 5)]
              + [(0, b) for b in range(1, 8)])
    for gauge in gauges:
        for a in range(1, q):
            for b in range(a + 1, q):
                cell = tuple(G.mul(gauge, g)
                             for g in (G.e, G.elements[a], G.elements[b]))
                A = [[list(cell), [G.e], [G.e]]]
                hx, hz, n = lifted_matrices(G, A, dummy_b)
                rows = [row & mask for row in hx[:q]]
                deficit = q - gf2_rank(rows)
                if deficit not in (4, 8):
                    continue
                if _has_small_logical(rows, [], q):
                    continue
                space = dependency_space(rows)
                incidence = tuple(sorted(columns(rows, q)))
                bucket = classes.setdefault(space, {})
                if incidence not in bucket and len(bucket) < 32:
                    bucket[incidence] = cell
        if sum(len(bucket) >= 3 for bucket in classes.values()) >= 24:
            break

    library = tuple(
        (space, tuple(bucket.values()))
        for space, bucket in classes.items() if len(bucket) >= 3)
    library = tuple(sorted(
        library, key=lambda item: (len(item[0]), item[0])))

    # These quotient models differ intrinsically from the parent's
    # preferred identity-plus-two-distinct-nonzero-quotient templates.
    b_models = ((0, 0, 1), (0, 0, 3), (0, 2, 6), (0, 1, 5))
    target_deficits = (4, 8, 4, 8, 4)
    target_spreads = (6, 6, 9, 9, 3)
    fingerprints, bank = set(), []

    def generators(rng):
        u = rng.randrange(5)
        v = rng.choice([a for a in range(5) if a != 2*u % 5])
        return ((u, 1), (v, 3), (rng.randrange(1, 5), 0))

    def triple(rng, model):
        used, cell = set(), []
        for b in model:
            choices = [(a, b) for a in range(5) if (a, b) not in used]
            g = rng.choice(choices)
            used.add(g)
            cell.append(g)
        base = G.pick(rng)
        return [G.mul(base, g) for g in cell]

    def draw_b(rng, mode, fallback):
        gs = generators(rng)
        if fallback:
            return [[[G.e, gs[0]], [G.e, gs[1]],
                     triple(rng, b_models[mode]),
                     triple(rng, b_models[(mode + 1) % 4])]]
        return [[[G.e, g] for g in gs]
                + [triple(rng, b_models[mode])]]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    for slot in range(N_SLOTS):
        card = slot % 20
        policy, mode = divmod(card, 4)
        rng = random.Random(40013573 + 100003*slot)
        winner = None

        # Rank deficit and quotient diversity are prescribed policies;
        # independent labels are optimized inside each policy.
        eligible = [
            item for item in library
            if len(item[0]) == target_deficits[policy]]
        if not eligible:
            eligible = list(library)

        for fallback in (False, True):
            if not fallback and not eligible:
                continue
            sa = [[2, 2, 2]] if fallback else [[3, 3, 3]]
            sb = [[2, 2, 3, 3]] if fallback else [[2, 2, 2, 3]]
            assert _grid_weights(sa, sb) <= 12
            assert _no_dead_lines(sa, sb)

            apool, aseen = [], set()
            for attempt in range(160):
                if fallback:
                    A = [[[G.e, g] for g in generators(rng)]]
                else:
                    space, cells = rng.choice(eligible)
                    A = [[list(cell) for cell in rng.sample(cells, 3)]]
                sig = signature(A, dummy_b)
                if sig in aseen:
                    continue
                aseen.add(sig)
                ar, width = component_rows(A, "A")
                if _has_small_logical(ar, [], width):
                    continue
                deficit = q - gf2_rank(ar)
                spread = sum(len({g[1] for g in cell}) for cell in A[0])
                key = (deficit,
                       -abs(spread - target_spreads[policy]),
                       -_four_cycle_count(ar))
                apool.append((key, A))
                apool.sort(key=lambda item: item[0], reverse=True)
                del apool[6:]
                if len(apool) >= 6 and attempt >= 48:
                    break
            if not apool:
                continue

            bpool, bseen = [], set()
            for attempt in range(128):
                B = draw_b(rng, mode, fallback)
                sig = signature(dummy_a, B)
                if sig in bseen:
                    continue
                bseen.add(sig)
                br, width = component_rows(B, "B")
                if _has_small_logical(br, [], width):
                    continue
                key = (_four_cycle_count(br), -gf2_rank(br))
                bpool.append((key, B))
                bpool.sort(key=lambda item: item[0])
                del bpool[6:]
                if len(bpool) >= 6 and attempt >= 48:
                    break
            if not bpool:
                continue

            proposals = []
            for ak, A in apool:
                for bk, B in bpool:
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (240 <= k and 1000*k < 525*n):
                        continue
                    cycles = -ak[2] + bk[0]
                    # Five is a conservative optimization target, not
                    # a newly certified or sampled distance reading.
                    key = (k, -cycles, ak[1])
                    proposals.append((key, A, B, hx, hz, n))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, A, B, hx, hz, n in proposals[:12]:
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in fingerprints:
                    continue
                assert css_ok(hx, hz), "verified lift failed CSS commutation"
                k = n - gf2_rank(hx) - gf2_rank(hz)
                assert _grid_weights(
                    [[len(cell) for cell in row] for row in A],
                    [[len(cell) for cell in row] for row in B]) <= 12
                winner = (A, B, k, fp, fallback)
                break
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B, k, fp, fallback = winner
        fingerprints.add(fp)
        bank.append({
            "cells": {"A": A, "B": B}, "k": k,
            "card": card, "fallback": fallback,
            "screen_no_le4": True})

    return tuple(bank)


@__import__("functools").lru_cache(maxsize=1)
def _cokernel_clique_bank(G):
    """Exhaust structural compatibility before selecting fresh B cards."""
    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 40
    mask = (1 << q) - 1
    dummy_a = [[[G.e] for _ in range(3)]]
    dummy_b = [[[G.e] for _ in range(4)]]

    def columns(rows, width):
        out = [0] * width
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return tuple(out)

    def rows_from_columns(cols):
        rows = [0] * q
        for c, original in enumerate(cols):
            syndrome = original
            while syndrome:
                bit = syndrome & -syndrome
                rows[bit.bit_length() - 1] |= 1 << c
                syndrome ^= bit
        return rows

    def clean4(cols):
        """Exactly exclude nonzero component kernel words through four."""
        seen = {0}
        for syndrome in cols:
            if syndrome in seen:
                return False
            seen.add(syndrome)
        for i, syndrome in enumerate(cols):
            for other in cols[:i]:
                pair = syndrome ^ other
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    def canonical_span(cols):
        """Canonical column span; equality means equal row cokernels."""
        basis = {}
        for original in cols:
            word = original
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old
        for p in sorted(basis):
            for other in sorted(basis):
                if other > p and ((basis[other] >> p) & 1):
                    basis[other] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    # All component actions below are XOR sums of VERIFIED unit lifts.
    # No separate implementation of left/right regular actions is used.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                unit = [[[g], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, unit, dummy_b)
                source = hx
            else:
                unit = [[[g], [G.e], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, dummy_a, unit)
                source = hz
            blocks.append(columns([r & mask for r in source[:q]], q))
        atlas[side] = tuple(blocks)

    def block(cell, side):
        out = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h in range(q):
                out[h] ^= contribution[h]
        return tuple(out)

    # Right normalization removes a cell's common column factor.
    # Every three-element support has a representative containing e.
    # Sorted column incidences remove further single-block gauge copies.
    nonidentity = [g for g in G.elements if G._key(g) != G._key(G.e)]
    classes = {}
    for g, h in combinations(nonidentity, 2):
        cell = (G.e, g, h)
        cols = block(cell, "A")
        space = canonical_span(cols)
        deficit = q - len(space)
        if deficit not in (4, 8) or not clean4(cols):
            continue
        incidence = tuple(sorted(cols))
        bucket = classes.setdefault(space, {})
        if incidence not in bucket:
            bucket[incidence] = (cell, cols)

    # First require every two-block union to be clean. Then enumerate
    # triangles of that compatibility graph and test the complete union:
    # a 2+1+1 relation can survive all pairwise compatibility tests.
    apool = []
    counts = Counter()
    ordered_classes = sorted(
        classes.items(), key=lambda item: (len(item[0]), item[0]))
    for space, bucket in ordered_classes:
        deficit = q - len(space)
        if counts[deficit] >= 48:
            continue
        entries = sorted(bucket.values(), key=lambda item: item[0])
        if len(entries) < 3:
            continue
        neighbors = [set() for _ in entries]
        for i in range(len(entries)):
            for j in range(i):
                if clean4(entries[j][1] + entries[i][1]):
                    neighbors[j].add(i)
        exhausted = False
        for i in range(len(entries)):
            for j in sorted(neighbors[i]):
                for t in sorted(neighbors[i] & neighbors[j]):
                    cols = entries[i][1] + entries[j][1] + entries[t][1]
                    if not clean4(cols):
                        continue
                    ar = rows_from_columns(cols)
                    assert q - gf2_rank(ar) == deficit
                    A = [[list(entries[z][0]) for z in (i, j, t)]]
                    apool.append((deficit, _four_cycle_count(ar), A))
                    counts[deficit] += 1
                    if counts[deficit] >= 48:
                        exhausted = True
                        break
                if exhausted:
                    break
            if exhausted:
                break
    apool.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not apool:
        raise ValueError("no clean common-cokernel compatibility clique")

    def fingerprint(hx, hz, n):
        """Invariant of the combined, uncoloured Tanner graph.

        Unequal fingerprints prove nonisomorphism. Equal fingerprints
        are discarded conservatively, not asserted to prove isomorphism.
        """
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        overlap = Counter()
        local_checks = []
        for i, row in enumerate(rows):
            neighborhood = Counter()
            for j, other in enumerate(rows):
                if i == j:
                    continue
                size = (row & other).bit_count()
                if size:
                    neighborhood[(weights[j], size)] += 1
                if j < i:
                    overlap[(min(weights[i], weights[j]),
                             max(weights[i], weights[j]), size)] += 1
            local_checks.append(
                (weights[i], tuple(sorted(neighborhood.items()))))
        qubits = [[] for _ in range(n)]
        for i, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                qubits[bit.bit_length() - 1].append(weights[i])
                row ^= bit
        return (
            tuple(sorted(Counter(local_checks).items())),
            tuple(sorted(overlap.items())),
            tuple(sorted(Counter(
                tuple(sorted(v)) for v in qubits).items())))

    # Ten quotient geometries absent from the parent's explicit B cards.
    # Labels within each geometry are exhaustively enumerated, not
    # transported from an archived filling.
    geometries = (
        (0, 2), (0, 4), (1, 2), (1, 6), (2, 3),
        (2, 4), (3, 4), (3, 6), (4, 6), (2, 6),
    )
    fingerprints = set()
    bank = []

    for slot in range(N_SLOTS):
        rotation_policy, mode = divmod(slot % 20, 10)
        c = (1, 2)[rotation_policy]
        b, d = geometries[mode]
        local = random.Random(40026419 + slot * 100003)
        skeleton = [[G.e, (0, 1)], [G.e, (1, 3)],
                    [G.e, (c, 0)]]
        base_cols = tuple(
            syndrome for cell in skeleton for syndrome in block(cell, "B"))
        bpool = []
        for a in range(5):
            for z in range(5):
                triple = [G.e, (a, b), (z, d)]
                if len(set(triple)) != 3:
                    continue
                cols = base_cols + block(triple, "B")
                if not clean4(cols):
                    continue
                br = rows_from_columns(cols)
                B = [[list(cell) for cell in skeleton] + [triple]]
                bpool.append((_four_cycle_count(br), B))
        local.shuffle(bpool)
        bpool.sort(key=lambda item: item[0])

        # Additional harness slots revisit a structural card with a
        # different A-clique ordering and exclude earlier graph invariants.
        ordered_a = list(apool)
        local.shuffle(ordered_a)
        ordered_a.sort(key=lambda item: (-item[0], item[1]))
        proposals = []
        for deficit, acycles, A in ordered_a:
            for bcycles, B in bpool:
                proposals.append(
                    ((-deficit, acycles + bcycles), A, B))
        local.shuffle(proposals)
        proposals.sort(key=lambda item: item[0])

        winner = None
        for key, A, B in proposals:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (475*n <= 1000*k < 525*n):
                continue
            # Component cleanliness is not substituted for the complete
            # stabilizer-aware quantum test, including both LP sectors.
            if _has_small_logical(hx, hz, n):
                continue
            if _has_small_logical(hz, hx, n):
                continue
            fp = fingerprint(hx, hz, n)
            if fp in fingerprints:
                continue
            assert css_ok(hx, hz)
            assert max(r.bit_count() for r in hx + hz) <= W_MAX
            fingerprints.add(fp)
            winner = {
                "cells": {
                    "A": tuple(tuple(tuple(cell) for cell in row) for row in A),
                    "B": tuple(tuple(tuple(cell) for cell in row) for row in B),
                },
                "k": k, "card": slot % 20,
                "screen_no_le4": True,
                "component_cokernel": -key[0],
            }
            break
        bank.append(winner)
    return tuple(bank)


@__import__("functools").lru_cache(maxsize=1)
def _containment_voltage_bank(G):
    """Fresh containment designs; every matrix action is scaffold-derived."""
    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 40
    mask = (1 << q) - 1
    stub_a = [[[G.e] for _ in range(3)]]
    stub_b = [[[G.e] for _ in range(4)]]

    def columns(rows, width):
        out = [0] * width
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return tuple(out)

    def rows_from_columns(cols):
        rows = [0] * q
        for c, original in enumerate(cols):
            value = original
            while value:
                bit = value & -value
                rows[bit.bit_length() - 1] |= 1 << c
                value ^= bit
        return rows

    def basis(words):
        out = {}
        for original in words:
            word = original
            while word:
                p = word.bit_length() - 1
                old = out.get(p)
                if old is None:
                    out[p] = word
                    break
                word ^= old
        return out

    def outside(word, span):
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def canonical_span(words):
        out = basis(words)
        pivots = sorted(out)
        for p in pivots:
            for h in pivots:
                if h > p and ((out[h] >> p) & 1):
                    out[h] ^= out[p]
        return tuple(out[p] for p in pivots)

    def clean4(cols):
        """Exact component-kernel exclusion through four."""
        seen = {0}
        for value in cols:
            if value in seen:
                return False
            seen.add(value)
        for i, value in enumerate(cols):
            for other in cols[:i]:
                pair = value ^ other
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    def five_count(cols):
        """Exact weight-five census, after clean4.

        Every five-support has ten triple/pair partitions. Clean4
        guarantees that colliding supports are disjoint and that each
        syndrome has at most one pair representative.
        """
        pairs = {
            cols[i] ^ cols[j]
            for i in range(len(cols)) for j in range(i)}
        hits = 0
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                for h in range(j):
                    hits += int((pair ^ cols[h]) in pairs)
        assert hits % 10 == 0
        return hits // 10

    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                unit = [[[g], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, unit, stub_b)
                source = hx
            else:
                unit = [[[g], [G.e], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, stub_a, unit)
                source = hz
            blocks.append(columns([r & mask for r in source[:q]], q))
        atlas[side] = tuple(blocks)

    def block(cell, side):
        out = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h, value in enumerate(contribution):
                out[h] ^= value
        return tuple(out)

    # Normalization removes independent right-column factors. Unlike
    # the parent, retain every clean singularity stratum and permit
    # unequal images whose union still has a sufficiently large cokernel.
    records = {}
    others = [g for g in G.elements if G._key(g) != G._key(G.e)]
    for g, h in combinations(others, 2):
        cell = (G.e, g, h)
        cols = block(cell, "A")
        span = canonical_span(cols)
        if len(span) > q - 4 or not clean4(cols):
            continue
        incidence = tuple(sorted(cols))
        records.setdefault(incidence, (cell, cols, span))

    entries = sorted(records.values(),
                     key=lambda item: (len(item[2]), item[0]))
    neighbors = [set() for _ in entries]
    pair_spaces = {}
    for i, first in enumerate(entries):
        for j in range(i):
            second = entries[j]
            joined = (first[2] if first[2] == second[2]
                      else canonical_span(first[2] + second[2]))
            if len(joined) > q - 4:
                continue
            if not clean4(second[1] + first[1]):
                continue
            neighbors[j].add(i)
            pair_spaces[j, i] = joined

    # Exhaust triangles; storage truncation never terminates enumeration.
    # Large individual deficits may join to deficit eight even when no
    # three blocks have identical rank-32 images.
    pools = {}
    for i in range(len(entries)):
        for j in sorted(neighbors[i]):
            pair_space = pair_spaces[i, j]
            for t in sorted(neighbors[i] & neighbors[j]):
                joined = (pair_space if pair_space == entries[t][2]
                          else canonical_span(pair_space + entries[t][2]))
                deficit = q - len(joined)
                if not 4 <= deficit <= 10:
                    continue
                cols = entries[i][1] + entries[j][1] + entries[t][1]
                if not clean4(cols):
                    continue
                ar = rows_from_columns(cols)
                assert q - gf2_rank(ar) == deficit
                A = [[list(entries[z][0]) for z in (i, j, t)]]
                labels = tuple(tuple(G.idx(g) for g in c) for c in A[0])
                item = (deficit, _four_cycle_count(ar), labels, A)
                pool = pools.setdefault(deficit, [])
                pool.append(item)
                if len(pool) > 192:
                    pool.sort(key=lambda x: (x[1], x[2]))
                    del pool[96:]

    apool = []
    for pool in pools.values():
        pool.sort(key=lambda x: (x[1], x[2]))
        apool.extend(pool[:96])
    apool.sort(key=lambda x: (-x[0], x[1], x[2]))
    if not apool:
        raise ValueError("no clean unequal-image containment assembly")

    def logical_five(checks, stabilizers, n):
        """Exact logical existence through five, after logical-four exclusion.

        Anchor one support coordinate. Any remaining commuting support
        has at most four coordinates. At each unsatisfied check at least
        one remaining coordinate must touch it; two branching levels
        followed by the complete singleton/pair table exhaust the search.

        One tail representative suffices modulo stabilizers: differences
        of equal-syndrome tails have weight <=4 and, by the preceding
        screen, lie in the stabilizer span. XOR handles tail overlaps.
        """
        cols = columns(checks, n)
        span = basis(stabilizers)
        bits = [1 << i for i in range(n)]
        tails = {0: 0}
        incidence = [[] for _ in checks]
        for i, value in enumerate(cols):
            tails.setdefault(value, bits[i])
            remaining = value
            while remaining:
                bit = remaining & -remaining
                incidence[bit.bit_length() - 1].append(i)
                remaining ^= bit
        for i, value in enumerate(cols):
            for j in range(i):
                tails.setdefault(value ^ cols[j], bits[i] | bits[j])
        max_degree = max(value.bit_count() for value in cols)

        def visit(value, remaining, word):
            if not value:
                return outside(word, span)
            if value.bit_count() > max_degree * remaining:
                return False
            if remaining <= 2:
                tail = tails.get(value)
                return (tail is not None
                        and outside(word ^ tail, span))
            check = (value & -value).bit_length() - 1
            for j in incidence[check]:
                if word & bits[j]:
                    continue
                if visit(value ^ cols[j], remaining - 1,
                         word ^ bits[j]):
                    return True
            return False

        return any(visit(value, 4, bits[i])
                   for i, value in enumerate(cols))

    def fingerprint(hx, hz, n):
        """Uncoloured Tanner invariant; omit colour-dependent k."""
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        adjacency = [0] * len(rows)
        overlaps = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                size = (row & rows[j]).bit_count()
                if size:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
                    overlaps[i][j] = overlaps[j][i] = size
        profiles = []
        for i, neighborhood in enumerate(overlaps):
            local = Counter(
                (weights[j], size,
                 (adjacency[i] & adjacency[j]).bit_count())
                for j, size in neighborhood.items())
            profiles.append((weights[i], tuple(sorted(local.items()))))
        qubits = [[] for _ in range(n)]
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                qubits[bit.bit_length() - 1].append(weights[r])
                row ^= bit
        return (n, tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(
                    tuple(sorted(v)) for v in qubits).items())))

    odd_pairs = ((1, 3), (1, 5), (1, 7), (3, 5), (3, 7))
    cubic_models = ((0, 0, 0), (0, 0, 4),
                    (0, 2, 2), (0, 6, 6))
    emitted, bank = set(), []

    for slot in range(N_SLOTS):
        policy, mode = divmod(slot % 20, 4)
        local = random.Random(40051873 + 100019 * slot)
        b0, b1 = odd_pairs[policy]
        even_b = 4 if mode in (1, 2) else 0
        chosen = None

        for batch in range(2):
            raw, seen_b = [], set()
            for attempt in range(160 + 96 * batch):
                u = local.randrange(5)
                parameter = u * pow(
                    (1 - pow(2, b0, 5)) % 5, -1, 5) % 5
                forbidden = (1 - pow(2, b1, 5)) * parameter % 5
                v = local.choice(
                    [a for a in range(5) if a != forbidden])
                generators = ((u, b0), (v, b1),
                              (local.randrange(1, 5), even_b))
                triple = []
                for b in cubic_models[mode]:
                    legal = [(a, b) for a in range(5)
                             if (a, b) not in triple]
                    triple.append(local.choice(legal))
                # Remove a common right factor rather than spend
                # proposal diversity on translated copies of the cell.
                inverse = G.inv(triple[0])
                triple = [G.mul(g, inverse) for g in triple]
                B = [[[G.e, g] for g in generators] + [triple]]
                labels = tuple(sorted(
                    tuple(sorted(G.idx(g) for g in cell))
                    for cell in B[0]))
                if labels in seen_b:
                    continue
                seen_b.add(labels)
                bc = tuple(value for cell in B[0]
                           for value in block(cell, "B"))
                if not clean4(bc):
                    continue
                br = rows_from_columns(bc)
                raw.append((_four_cycle_count(br), labels, B, bc))
                raw.sort(key=lambda x: (x[0], x[1]))
                del raw[12:]

            bp = [(five_count(bc), cycles, labels, B)
                  for cycles, labels, B, bc in raw]
            bp.sort(key=lambda x: x[:3])
            bp = bp[:6]
            if not bp:
                continue

            ordered_a = list(apool)
            local.shuffle(ordered_a)
            ordered_a.sort(key=lambda x: (-x[0], x[1]))
            proposals = [
                ((-a[0], b[0], a[1] + b[1]), a[3], b[3])
                for a in ordered_a[:32] for b in bp]
            local.shuffle(proposals)
            proposals.sort(key=lambda x: x[0])

            winner, tested = None, 0
            for key, A, B in proposals:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (475 * n <= 1000 * k < 525 * n):
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                if _has_small_logical(hx, hz, n):
                    continue
                if _has_small_logical(hz, hx, n):
                    continue

                # Six is a proved exclusion floor only when the complete
                # tests exclude all logicals through five in both types.
                # Otherwise no sampled miss receives distance credit.
                floor = 5
                if (not logical_five(hx, hz, n)
                        and not logical_five(hz, hx, n)):
                    floor = 6
                objective = (k * floor * floor, k, -key[1], -key[2])
                if winner is None or objective > winner[0]:
                    winner = (objective, A, B, k, fp, floor)
                tested += 1
                if floor == 6 or tested >= 8:
                    break

            if winner is None:
                continue
            objective, A, B, k, fp, floor = winner
            hx, hz, n = lifted_matrices(G, A, B)
            assert css_ok(hx, hz)
            assert max(r.bit_count() for r in hx + hz) <= W_MAX
            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            assert _grid_weights(sa, sb) <= W_MAX
            assert _no_dead_lines(sa, sb)
            assert all(len(c) == len(set(c)) <= 5
                       for grid in (A, B) for row in grid for c in row)
            emitted.add(fp)
            chosen = {
                "cells": {
                    side: tuple(tuple(tuple(c) for c in row) for row in grid)
                    for side, grid in (("A", A), ("B", B))},
                "k": k, "card": slot % 20,
                "verified_exclusion_floor": floor,
                "selection": objective}
            break

        # Failed policies never call either archived parent generator.
        bank.append(chosen)
    return tuple(bank)


def _sp_columns(rows, width):
    cols = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    return tuple(cols)


def _sp_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _sp_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _sp_probe(cols, stabilizers, local, trials, seeds=()):
    """Actual logical witnesses, not a lower bound on distance.

    All directions have zero syndrome. Stabilizer descent preserves the
    logical class; information-set recombinations are classified again.
    Keep logical memory separately from short stabilizer directions.
    """
    span = _sp_basis(stabilizers)
    width = len(cols)
    best = width + 1
    memory = []
    order = list(range(width))

    def observe(word):
        nonlocal best
        if (word and word.bit_count() < best
                and _sp_outside(word, span)):
            best = word.bit_count()

    for word in seeds:
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length() - 1]
            remaining ^= bit
        assert syndrome == 0
        if word and _sp_outside(word, span):
            memory.append(word)
            observe(word)
    memory = sorted(set(memory), key=lambda w: (w.bit_count(), w))[:16]

    for _ in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                observe(word)

        dependencies.sort(key=lambda w: (w.bit_count(), w))
        logical = [w for w in dependencies if _sp_outside(w, span)]
        directions = sorted(
            set(dependencies[:48] + memory),
            key=lambda w: (w.bit_count(), w))[:64]
        discoveries = set(memory)
        starts = sorted(set(logical + memory),
                        key=lambda w: (w.bit_count(), w))[:8]

        for seed in starts:
            word = seed
            for _ in range(2):
                changed = False
                for direction in directions:
                    candidate = word ^ direction
                    if (candidate and
                            candidate.bit_count() < word.bit_count()):
                        word = candidate
                        changed = True
                if not changed:
                    break
            observe(word)
            if word and _sp_outside(word, span):
                discoveries.add(word)

            # Single- and pair-stabilizer dressing of the same class.
            for _ in range(2):
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda x: x[0])[:12]
                chosen, weight = word, word.bit_count()
                for _, row in nearby:
                    candidate = word ^ row
                    if candidate.bit_count() < weight:
                        chosen, weight = candidate, candidate.bit_count()
                for i, (_, row) in enumerate(nearby):
                    for _, other in nearby[:i]:
                        candidate = word ^ row ^ other
                        if candidate.bit_count() < weight:
                            chosen, weight = candidate, candidate.bit_count()
                if chosen == word:
                    break
                word = chosen
                observe(word)
            if word and _sp_outside(word, span):
                discoveries.add(word)

        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:16]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                observe(candidate)
                if candidate and _sp_outside(candidate, span):
                    discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:16]
    return best, tuple(memory)


def _sp_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; unequal values prove nonisomorphism.

    Equal values are discarded conservatively. Neither check type nor
    spectral coordinates enter the invariant.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    adjacency = [0] * len(rows)
    overlaps = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            if size:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlaps[i][j] = overlaps[j][i] = size
    profiles = []
    for i, neighbors in enumerate(overlaps):
        profile = Counter(
            (rows[j].bit_count(), size,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, size in neighbors.items())
        profiles.append((rows[i].bit_count(),
                         tuple(sorted(profile.items()))))
    cols = _sp_columns(rows, n)
    return (
        n, tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())))


@__import__("functools").lru_cache(maxsize=1)
def _spectral_threeorbit_bank(G):
    """Fresh spectral architectures; historical order-40 banks are unused."""
    from itertools import combinations, permutations
    from collections import Counter

    assert G.q == 289 and A_SHAPE == B_SHAPE == (1, 1)
    q, n = G.q, 2*G.q
    mask = (1 << q) - 1

    # F256 = F2[t]/(t^8+t^4+t^3+t^2+1); t is primitive.
    def fm(a, b):
        answer = 0
        while b:
            if b & 1:
                answer ^= a
            b >>= 1
            a <<= 1
            if a & 256:
                a ^= 285
        return answer

    alpha = 1
    for _ in range(15):
        alpha = fm(alpha, 2)
    roots = [1]
    for _ in range(16):
        roots.append(fm(roots[-1], alpha))
    assert fm(roots[-1], alpha) == 1 and len(set(roots)) == 17

    # Coordinate projections are binary length-17 weight-five words
    # annihilated by a primitive seventeenth root.
    qr = []
    for tail in combinations(range(1, 17), 4):
        exponents = (0,) + tail
        value = 0
        for a in exponents:
            value ^= roots[a]
        if not value:
            qr.append(exponents)
    assert qr, "no normalized weight-five coordinate support"

    representatives, covered = [], {(0, 0)}
    axis = {}
    for a in range(17):
        for b in range(17):
            if (a, b) in covered:
                continue
            orbit, point = [], (a, b)
            while point not in orbit:
                orbit.append(point)
                point = (2*point[0] % 17, 2*point[1] % 17)
            assert len(orbit) == 8
            index = len(representatives)
            representatives.append((a, b))
            covered.update(orbit)
            for coordinate in ((1, 0), (0, 1)):
                if coordinate in orbit:
                    axis[coordinate] = index
    assert len(representatives) == 36 and len(covered) == q
    axis_mask = sum(1 << index for index in axis.values())

    records, seen = [], set()
    for aa in qr:
        for bb in qr:
            for bp in permutations(bb[1:]):
                cell = (G.e,) + tuple(zip(aa[1:], bp))
                labels = tuple(sorted(G.idx(g) for g in cell))
                if labels in seen:
                    continue
                seen.add(labels)

                # Remove one-dimensional supports before pairing.
                if not any(
                    (cell[i][0]*cell[j][1] -
                     cell[j][0]*cell[i][1]) % 17
                    for i in range(1, 5) for j in range(1, i)):
                    continue

                zeros = 0
                for r, (u, v) in enumerate(representatives):
                    value = 0
                    for a, b in cell:
                        value ^= roots[(u*a + v*b) % 17]
                    if not value:
                        zeros |= 1 << r
                assert zeros & axis_mask == axis_mask
                if zeros.bit_count() < 3:
                    continue

                differences = Counter(
                    ((a-c) % 17, (b-d) % 17)
                    for a, b in cell for c, d in cell
                    if (a, b) != (c, d))
                collisions = sum(v*(v-1)//2
                                 for v in differences.values())
                collinear = 0
                for p0, p1, p2 in combinations(cell, 3):
                    collinear += int(
                        ((p1[0]-p0[0])*(p2[1]-p0[1]) -
                         (p2[0]-p0[0])*(p1[1]-p0[1])) % 17 == 0)
                records.append(dict(
                    cell=tuple(sorted(cell)), zeros=zeros,
                    collisions=collisions, collinear=collinear,
                    labels=labels))
    records.sort(key=lambda r: r["labels"])

    pairs = []
    for i, a in enumerate(records):
        for j, b in enumerate(records[:i]):
            shared = a["zeros"] & b["zeros"]
            count = shared.bit_count()
            if count not in (3, 4):
                continue
            extra = shared & ~axis_mask
            extra_class = (extra & -extra).bit_length() - 1
            pairs.append((
                i, j, 16*count, extra_class,
                a["collisions"] + b["collisions"],
                a["collinear"] + b["collinear"],
                abs(a["zeros"].bit_count() - b["zeros"].bit_count())))
    if not pairs:
        raise ValueError("no three/four-common-orbit support pair")

    # Compile all four matrix contributions from VERIFIED unit lifts.
    # One lift with A=B=[g] supplies both sides; masks merely separate
    # the two documented LP sectors. No independent regular lift is used.
    atoms = []
    for g in G.elements:
        hx, hz, vn = lifted_matrices(G, [[[g]]], [[[g]]])
        assert vn == n
        atoms.append((
            tuple(row & mask for row in hx),
            tuple(row & ~mask for row in hx),
            tuple(row & ~mask for row in hz),
            tuple(row & mask for row in hz)))
    compiled = {}

    def contributions(index):
        old = compiled.get(index)
        if old is not None:
            return old
        answer = [[0]*q for _ in range(4)]
        for g in records[index]["cell"]:
            for target, source in zip(answer, atoms[G.idx(g)]):
                for r, value in enumerate(source):
                    target[r] ^= value
        old = tuple(tuple(row) for row in answer)
        compiled[index] = old
        return old

    def build(i, j):
        aa, bb = contributions(i), contributions(j)
        hx = [a ^ b for a, b in zip(aa[0], bb[1])]
        hz = [a ^ b for a, b in zip(aa[2], bb[3])]
        return hx, hz

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        policy, mode = divmod(slot % 20, 4)
        lane = slot // 20
        local = random.Random(289103117 + 104729*slot)
        ordered = list(pairs)
        local.shuffle(ordered)
        target_collinear = (0, 1, 2, 3, 4)[policy]
        target_k = 64 if policy == 4 else 48
        ordered.sort(key=lambda p: (
            p[2] != target_k,
            p[3] % 4 != (mode + lane) % 4,
            abs(p[5] - target_collinear),
            p[4], p[6]))

        finalists, local_pairs = [], set()
        for i, j, predicted_k, _, _, _, _ in ordered[:160]:
            key = (i, j)
            if key in local_pairs:
                continue
            local_pairs.add(key)
            hx, hz = build(i, j)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # Spectral predictions never replace actual matrix ranks.
            if k != predicted_k or k not in (48, 64):
                continue
            fp = _sp_fingerprint(hx, hz, n)
            if fp in emitted:
                continue
            if _has_small_logical(hx, hz, n):
                continue
            if _has_small_logical(hz, hx, n):
                continue

            xc, zc = _sp_columns(hz, n), _sp_columns(hx, n)
            dx, xwords = _sp_probe(xc, hx, local, 8)
            dz, zwords = _sp_probe(zc, hz, local, 8)
            if dx > n or dz > n:
                continue
            finalists.append(dict(
                i=i, j=j, hx=hx, hz=hz, k=k, fp=fp,
                xc=xc, zc=zc, dx=dx, dz=dz,
                xwords=xwords, zwords=zwords))
            if len(finalists) >= 4:
                break

        def merit(item):
            # Forty is only a selection cap on found upper bounds.
            reading = min(item["dx"], item["dz"], 40)
            return (item["k"]*reading*reading/n,
                    reading, item["k"])

        finalists.sort(key=merit, reverse=True)
        for item in finalists[:2]:
            dx, _ = _sp_probe(
                item["xc"], item["hx"], local, 12, item["xwords"])
            dz, _ = _sp_probe(
                item["zc"], item["hz"], local, 12, item["zwords"])
            item["dx"] = min(item["dx"], dx)
            item["dz"] = min(item["dz"], dz)
        # Only independently rescreened finalists can win.
        finalists = finalists[:2]
        if not finalists:
            bank.append(None)
            continue
        winner = max(finalists, key=merit)
        A = [[list(records[winner["i"]]["cell"])]]
        B = [[list(records[winner["j"]]["cell"])]]
        vx, vz, vn = lifted_matrices(G, A, B)
        assert vx == winner["hx"] and vz == winner["hz"] and vn == n
        assert winner["k"] == n - gf2_rank(vx) - gf2_rank(vz)
        assert css_ok(vx, vz)
        assert max(row.bit_count() for row in vx + vz) == 10
        assert _grid_weights([[5]], [[5]]) == 10
        assert _no_dead_lines([[5]], [[5]])
        assert all(len(c) == len(set(c)) == 5
                   for grid in (A, B) for row in grid for c in row)
        assert 0 < 1000*winner["k"] < 125*n
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={
                "A": ((tuple(A[0][0]),),),
                "B": ((tuple(B[0][0]),),)},
            k=winner["k"], card=(policy, mode, lane),
            shared_character_orbits=winner["k"]//16,
            logical_upper_readings=(winner["dx"], winner["dz"]),
            selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh, jointly screened three/four-orbit design."""
    selected = _spectral_threeorbit_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh spectral rank-48/64 design passed this policy")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the complete five-term cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting odd-C8 binomial differences and four-cycle selection suffice to remove weight-four X logicals; all five measured parent codes have d30s=4, including the k=242 rank-deficit winners.
# REFUTED: Rotation-cage binomials plus mandatory logical-four exclusion raise this bed's measured distance above five; all five measured current-parent codes had d30s=5, and repeated connected-binomial recipes produced isomorphic duplicates.
# REFUTED: Shared-cokernel enrichment and component-four-cycle selection alone eliminate sector-1 weight-five X witnesses; all five measured current codes had k=252 and d30s=5, with thousands of weight-five witnesses.
# REFUTED: Exhaustive equal-image compatibility cliques uncover clean dimension-eight cokernels and k=264 leaders; the measured clique program instead produced five k=252, d30s=5 leaders and unchanged fitness 12.1150.
# REFUTED: Unequal-image containment and long-voltage logical-five cards improve the half-rate bed; the measured current program emitted only seven distinct codes and all five leaders remained k=252, d30s=5, fitness 12.1150.
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
