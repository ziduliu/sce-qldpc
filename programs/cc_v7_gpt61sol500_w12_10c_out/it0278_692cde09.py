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
CONCEPT = ("C5 semidirect C8 with action 2, order 40, shapes (1,3)x(1,3): equal noncommuting cubic components, optimized by witnessed short-word repairs rather than singular-image enrichment, should produce five fresh distance-ten readings and improve the 0.400 frontier above 27.0. This is a measured-distance hypothesis, not a certificate.")
GROUP = ("Keep the exact group and order unchanged under L2. This group's equal (1,3) bed has reached fitness 16.0, while the parent repeatedly measured [520,252,5]. Those readings motivate removing the binomial bottleneck and testing unrestricted cubic images; they do not establish a group-distance advantage. Every screened action comes from verified lifts.")
SHAPES = ("Change B from (1,4) to (1,3), retaining A=(1,3): n=400, kb=160, guaranteed rate 0.400, and equal 120-variable/40-check components. The shape admits q<=59, but the unchanged group fixes q=40. Parent weight-five X minima were entirely sector one, especially Acol1/Bcol2; Z minima weighed six. Exact full ranks enforce 0.375<=k/n<0.425 for the requested 0.400 cell.")
FILLING = ("All six cells are distinct-term trinomials, giving HX=HZ=12 and odd column degrees in both complete matrices. Thus odd-weight commuting words are impossible. Twenty architectures cross five A and four B quotient-occupancy profiles. Compile a normalized noncommuting cell library from verified unit lifts, optimize full-rank constituents with exact short-word decoding and witness-directed replacements, and assemble fresh invariant-distinct codes. Require complete stabilizer-aware exclusion through four; exact six exclusion earns floor eight, and completed eight exclusion earns floor ten. Bounded-test exhaustion earns no additional floor. No archived filling or old containment bank is used.")
FAMILY_NAME = "r040_C5sdC8_action2_13x13_cubic_even_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    # C5 semidirect C8; action 2 has order 4 and 2^8=1 mod 5.
    if not valid_q(q):
        raise ValueError("C5 semidirect C8 requires order 40")
    els = [(a, b) for b in range(8) for a in range(5)]
    action = [pow(2, b, 5) for b in range(8)]

    def red(g):
        return (g[0] % 5, g[1] % 8)

    def mul(g, h):
        return ((g[0] + action[g[1] % 8] * h[0]) % 5,
                (g[1] + h[1]) % 8)

    def inv(g):
        bi = (-g[1]) % 8
        return ((-action[bi] * g[0]) % 5, bi)

    def idx(g):
        a, b = red(g)
        return a + 5 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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


def _cubic_columns(rows, width):
    out = [0] * width
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return tuple(out)


def _cubic_basis(rows):
    span = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = span.get(p)
            if previous is None:
                span[p] = word
                break
            word ^= previous
    return span


def _cubic_outside(word, span):
    while word:
        p = word.bit_length() - 1
        previous = span.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _cubic_tails(cols, strict=False):
    """Singleton/pair representatives; strict mode excludes kernels <=4."""
    tails = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        if syndrome in tails:
            if strict and tails[syndrome] != word:
                return None
        else:
            tails[syndrome] = word
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            if value in tails:
                if strict and tails[value] != word:
                    return None
            else:
                tails[value] = word
    return tails


def _cubic_bounded_word(cols, height, stabilizers, bound,
                        anchors=None, node_cap=0, tails=None):
    """Return an actual word, zero for completed exclusion, None on cutoff.

    Branch on an unsatisfied check and finish with the complete <=2
    syndrome table. The caller has already excluded nontrivial words
    through four, so equal-syndrome tail representatives differ only
    by stabilizers. XOR, rather than union, handles tail overlaps.

    Constituent callers use one regular-action anchor per block.
    Complete quantum callers use EVERY physical qubit.
    """
    if tails is None:
        tails = _cubic_tails(cols)
    span = None if stabilizers is None else _cubic_basis(stabilizers)
    bits = [1 << i for i in range(len(cols))]
    incidence = [[] for _ in range(height)]
    for i, original in enumerate(cols):
        syndrome = original
        while syndrome:
            bit = syndrome & -syndrome
            incidence[bit.bit_length() - 1].append(i)
            syndrome ^= bit
    maximum = max(value.bit_count() for value in cols)
    nodes = 0

    class Exhausted(Exception):
        pass

    def nontrivial(word):
        return bool(word) and (
            span is None or _cubic_outside(word, span))

    def visit(value, remaining, word):
        nonlocal nodes
        nodes += 1
        if node_cap and nodes > node_cap:
            raise Exhausted
        if not value:
            return word if nontrivial(word) else 0
        if value.bit_count() > maximum * remaining:
            return 0
        if remaining <= 2:
            tail = tails.get(value)
            if tail is not None and tail.bit_count() <= remaining:
                candidate = word ^ tail
                if nontrivial(candidate):
                    return candidate
            return 0
        check = (value & -value).bit_length() - 1
        for j in incidence[check]:
            if word & bits[j]:
                continue
            hit = visit(value ^ cols[j], remaining - 1,
                        word ^ bits[j])
            if hit:
                return hit
        return 0

    if anchors is None:
        anchors = range(len(cols))
    try:
        for i in anchors:
            hit = visit(cols[i], bound - 1, bits[i])
            if hit:
                assert hit.bit_count() <= bound
                syndrome = 0
                remaining = hit
                while remaining:
                    bit = remaining & -remaining
                    syndrome ^= cols[bit.bit_length() - 1]
                    remaining ^= bit
                assert syndrome == 0
                return hit
    except Exhausted:
        return None
    return 0


def _cubic_fingerprint(hx, hz, n):
    """Uncoloured incidence invariant, including check-intersection walks.

    Unequal signatures certify nonisomorphism. Equal signatures are
    conservatively rejected, not asserted to prove isomorphism.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * len(rows)
    overlap = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            if size:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlap[i][j] = overlap[j][i] = size
    profiles = []
    for i in range(len(rows)):
        local = Counter(
            (weights[j], overlap[i].get(j, 0),
             (adjacency[i] & adjacency[j]).bit_count())
            for j in range(len(rows)) if j != i)
        profiles.append((weights[i], tuple(sorted(local.items()))))
    degrees = _cubic_columns(rows, n)
    return (
        n, tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(c.bit_count() for c in degrees).items())))


@__import__("functools").lru_cache(maxsize=1)
def _balanced_cubic_bank(G):
    """Fresh equal-shape cubic architectures; old banks are inactive."""
    from itertools import combinations

    assert G.q == 40
    assert A_SHAPE == B_SHAPE == (1, 3)
    q, width = G.q, 3 * G.q
    mask = (1 << q) - 1
    zero = [[[], [], []]]
    atlas = {}

    # Compile the two regular actions solely through verified unit lifts.
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            block = _cubic_columns(
                [row & mask for row in source[:q]], q)
            assert all(col.bit_count() == 1 for col in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_block(side, cell):
        cols = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(contribution):
                cols[h] ^= syndrome
        return tuple(cols)

    def component_rows(cols):
        rows = [0] * q
        for c, original in enumerate(cols):
            syndrome = original
            while syndrome:
                bit = syndrome & -syndrome
                rows[bit.bit_length() - 1] |= 1 << c
                syndrome ^= bit
        return rows

    # Right-normalized support geometries, not translated code copies.
    # kind 0: repeated quotient coordinate;
    # kind 1: three quotient coordinates including an antipodal pair;
    # kind 2: three non-antipodal quotient coordinates.
    libraries = {side: [[], [], []] for side in ("A", "B")}
    supports = set()
    others = [g for g in G.elements if G._key(g) != G._key(G.e)]
    for g, h in combinations(others, 2):
        original = (G.e, g, h)
        key = min(
            tuple(sorted(G.idx(G.mul(t, G.inv(anchor)))
                         for t in original))
            for anchor in original)
        if key in supports:
            continue
        supports.add(key)
        cell = tuple(G.elements[t] for t in key)
        relatives = [G.mul(t, G.inv(cell[0])) for t in cell[1:]]
        if G.mul(relatives[0], relatives[1]) == G.mul(
                relatives[1], relatives[0]):
            continue
        bs = {t[1] for t in cell}
        # Retain whole-C8 quotient generation, rather than a smaller
        # even-voltage subgroup that changes the constituent bed.
        if not any(b & 1 for b in bs):
            continue
        if len(bs) == 2:
            kind = 0
        elif len(bs) == 3:
            kind = 1 if any(
                (b - c) % 8 == 4 for b in bs for c in bs) else 2
        else:
            continue
        for side in ("A", "B"):
            cols = compile_block(side, cell)
            if _cubic_tails(cols, strict=True) is None:
                continue
            libraries[side][kind].append((cell, cols, key))

    for side in libraries:
        for bucket in libraries[side]:
            bucket.sort(key=lambda entry: entry[2])

    a_cards = (
        (2, 2, 2), (0, 2, 2), (1, 2, 2),
        (0, 1, 2), (1, 1, 2))
    b_cards = (
        (2, 2, 2), (0, 2, 2), (1, 2, 2), (0, 1, 2))
    pools = {}

    def optimize_component(side, kinds):
        """Exact floors plus local replacements directed by actual words."""
        pool_key = (side, kinds)
        if pool_key in pools:
            return pools[pool_key]
        if any(not libraries[side][kind] for kind in kinds):
            pools[pool_key] = ()
            return ()
        policy_number = sum(kind * (4 ** j)
                            for j, kind in enumerate(kinds))
        local = random.Random(
            40031027 + 1009 * policy_number
            + (0 if side == "A" else 1000003))
        retained, labels_seen = [], set()

        def merit(item):
            return (item["floor"], -item["cycles"], item["labels"])

        for attempt in range(640):
            if retained and attempt >= 96 and attempt % 3:
                parent = local.choice(retained[:min(8, len(retained))])
                entries = list(parent["entries"])
                word = parent["word"]
                pressure = [
                    1 + ((word >> (j * q)) & mask).bit_count()
                    for j in range(3)]
                j = local.choices(range(3), weights=pressure, k=1)[0]
                entries[j] = local.choice(libraries[side][kinds[j]])
                if attempt % 11 == 0:
                    t = (j + 1 + local.randrange(2)) % 3
                    entries[t] = local.choice(libraries[side][kinds[t]])
            else:
                entries = [
                    local.choice(libraries[side][kind]) for kind in kinds]
            labels = tuple(sorted(entry[2] for entry in entries))
            if labels in labels_seen:
                continue
            labels_seen.add(labels)
            cols = tuple(c for entry in entries for c in entry[1])
            tails = _cubic_tails(cols, strict=True)
            if tails is None:
                continue
            rows = component_rows(cols)
            if gf2_rank(rows) != q:
                continue
            assert all(c.bit_count() == 3 for c in cols)

            # Left-lift components commute with right translations;
            # right-lift components commute with left translations.
            # These actions are transitive inside every q-column block,
            # so one anchor per block exhausts constituent existence.
            anchors = tuple(j * q for j in range(3))
            word = _cubic_bounded_word(
                cols, q, None, 6, anchors=anchors, tails=tails)
            assert word is not None
            floor = 6
            if not word:
                floor = 8
                hit8 = _cubic_bounded_word(
                    cols, q, None, 8, anchors=anchors,
                    node_cap=24000, tails=tails)
                if hit8 == 0:
                    floor = 10
                elif hit8 is not None:
                    word = hit8
                # Unknown leaves floor eight, never ten.
            item = {
                "entries": tuple(entries), "cols": cols,
                "floor": floor, "word": word,
                "cycles": _four_cycle_count(rows), "labels": labels,
                "cells": [[list(entry[0]) for entry in entries]]}
            retained.append(item)
            retained.sort(key=merit, reverse=True)
            del retained[24:]
        pools[pool_key] = tuple(retained)
        return pools[pool_key]

    for kinds in a_cards:
        optimize_component("A", kinds)
    for kinds in b_cards:
        optimize_component("B", kinds)

    emitted, bank = set(), []
    sa = sb = [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    for slot in range(N_SLOTS):
        acard, bcard = divmod(slot % 20, 4)
        ap = optimize_component("A", a_cards[acard])
        bp = optimize_component("B", b_cards[bcard])
        local = random.Random(40041073 + 104729 * slot)
        proposals = [
            (min(a["floor"], b["floor"]),
             a["cycles"] + b["cycles"], a, b)
            for a in ap for b in bp]
        local.shuffle(proposals)
        proposals.sort(key=lambda item: (-item[0], item[1]))
        winner, tested = None, 0

        for component_floor, cycles, a, b in proposals[:192]:
            A, B = a["cells"], b["cells"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (375 * n <= 1000 * k < 425 * n):
                continue

            # Assert the compiled components agree with the full lift.
            ar = [
                sum(((row >> (3 * j * q)) & mask) << (j * q)
                    for j in range(3))
                for row in hx[:q]]
            br = [row & ((1 << width) - 1) for row in hz[:q]]
            assert _cubic_columns(ar, width) == a["cols"]
            assert _cubic_columns(br, width) == b["cols"]

            if _has_small_logical(hx, hz, n):
                continue
            if _has_small_logical(hz, hx, n):
                continue
            fingerprint = _cubic_fingerprint(hx, hz, n)
            if fingerprint in emitted:
                continue
            xc, zc = _cubic_columns(hz, n), _cubic_columns(hx, n)
            assert all(c.bit_count() & 1 for c in xc + zc)

            # Odd check-column degrees exclude every odd-weight kernel
            # word. Complete four exclusion therefore establishes six.
            floor = 6
            tx, tz = _cubic_tails(xc), _cubic_tails(zc)
            x6 = _cubic_bounded_word(
                xc, len(hz), hx, 6, tails=tx)
            z6 = _cubic_bounded_word(
                zc, len(hx), hz, 6, tails=tz)
            assert x6 is not None and z6 is not None
            if x6 == z6 == 0:
                floor = 8
                x8 = _cubic_bounded_word(
                    xc, len(hz), hx, 8, node_cap=40000, tails=tx)
                z8 = _cubic_bounded_word(
                    zc, len(hx), hz, 8, node_cap=40000, tails=tz)
                if x8 == z8 == 0:
                    floor = 10
                # A found eight-word or a cutoff both leave floor eight.

            objective = (
                k * floor * floor, floor, component_floor, -cycles)
            if winner is None or objective > winner[0]:
                winner = (objective, A, B, hx, hz, n, k,
                          fingerprint, floor)
            tested += 1
            if floor == 10 or tested >= 4:
                break

        if winner is None:
            bank.append(None)
            continue
        objective, A, B, hx, hz, n, k, fingerprint, floor = winner
        assert n == 400 and k > 0
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 12
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(fingerprint)
        bank.append({
            "cells": {
                side: tuple(tuple(tuple(cell) for cell in row)
                            for row in grid)
                for side, grid in (("A", A), ("B", B))},
            "k": k, "card": slot % 20,
            "verified_exclusion_floor": floor,
            "selection": objective})
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh equal-cubic architecture, never an old-bank fallback."""
    selected = _balanced_cubic_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh target-band balanced cubic design passed screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly screened full cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting odd-C8 binomial differences and four-cycle selection suffice to remove weight-four X logicals; all five measured parent codes have d30s=4, including the k=242 rank-deficit winners.
# REFUTED: Rotation-cage binomials plus mandatory logical-four exclusion raise this bed's measured distance above five; all five measured current-parent codes had d30s=5, and repeated connected-binomial recipes produced isomorphic duplicates.
# REFUTED: Shared-cokernel enrichment and component-four-cycle selection alone eliminate sector-1 weight-five X witnesses; all five measured current codes had k=252 and d30s=5, with thousands of weight-five witnesses.
# REFUTED: Exhaustive equal-image compatibility cliques uncover clean dimension-eight cokernels and k=264 leaders; the measured clique program instead produced five k=252, d30s=5 leaders and unchanged fitness 12.1150.
# REFUTED: Unequal-image containment and long-voltage B policies improve the half-rate plateau; the current program again measured five k=252, d30s=5 leaders, with only seven distinct outputs and unchanged fitness 12.1150.
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
