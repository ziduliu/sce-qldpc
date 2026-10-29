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
CONCEPT = ("Dic21, order 84, shapes (1,2)x(1,3): quotient-separated cubic fillings optimized against retained component and full-code witnesses should supply five fresh distance-twelve readings and exceed the 0.300 frontier fitness of 38.0; measurement can refute this.")
GROUP = ("Jump from the order-40 C5 semidirect C8 group to Dic21. The recorded order-84, centre-two, 24-class (1,2)x(1,3) bed reached distance-twelve readings, but its five leaders included eleven and nine. Its cyclic rotation subgroups support explicit fiber searches. These measurements motivate fresh optimization, not a transferred distance certificate.")
SHAPES = ("Target 0.300 instead of the parent's half-rate plateau. Shapes (1,2)x(1,3) give n=588, kb=168 and guaranteed rate 2/7. The size window permits q<=85; Dic21 fixes q=84. Component lengths are 168/252 with 84 checks each. Parent minima were entirely sector-1 Acol/Bcol witnesses, especially Acol1 and Bcol2. Remove its binomial-rich B component and gate actual assembled rates to [0.275,0.325).")
FILLING = ("Twenty policies cross five rotation-residue separation moduli with four mixed-coset cubic occupancies. Every A/B cell has three distinct terms: HX=6+3=9 and HZ=9+3=12. Use only verified lifts to optimize rank-stratified component pools, retain true kernel words including rotation-fiber supports, repair witness-heavy cells, independently deepen finalists, and verify embeddings and logical classes against complete stabilizer spaces. Full quantum probes include both sectors and stabilizer dressings. Exact logical-four rejection and uncoloured Tanner invariants screen final outputs. Search readings remain upper bounds, not distance certificates.")
FAMILY_NAME = "r030_Dic21_12x13_cubic_fiber_rankstrata_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Dic21: x^42=1, y^2=x^21, y*x*y^-1=x^-1.
    # Canonical (a,b) represents x^a y^b, b in {0,1}.
    if not valid_q(q):
        raise ValueError("Dic21 requires order 84")
    els = [(a,b) for b in range(2) for a in range(42)]

    def red(g):
        a,b = g
        return ((a + 21*(b//2)) % 42, b % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + (c if b == 0 else -c) + 21*b*d) % 42,
                (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-a) % 42, 0) if b == 0 else ((a+21) % 42, 1)

    def idx(g):
        a,b = red(g)
        return 42*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
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


def _d21_columns(rows, width):
    cols = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return cols


def _d21_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _d21_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _d21_clean4(cols):
    """Exact absence of nonzero component kernel supports through four."""
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


def _d21_words(cols, rng, trials, memory=()):
    """Retain actual kernel words; high search readings are not proofs."""
    retained = sorted(
        {w for w in memory if w},
        key=lambda w: (w.bit_count(), w))[:40]
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome and word:
                dependencies.append(word)

        words = sorted(set(retained+dependencies),
                       key=lambda w: (w.bit_count(), w))
        for sweep in range(2):
            reducers = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        word, weight = trial, tw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))

        discoveries = set(words+retained)
        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    discoveries.add(trial)
        retained = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:40]
    return tuple(retained)


def _d21_draw(rng, G, side, modulus, mode, phase):
    """Mixed-coset cubic cells with distinct projected rotation labels."""
    from math import gcd
    width = A_SHAPE[1] if side == "A" else B_SHAPE[1]
    pools = [
        [g for g in G.elements if g[1] == b] for b in range(2)]
    row = []
    for j in range(width):
        if mode == 0:
            outer = 1
        elif mode == 1:
            outer = 2
        elif mode == 2:
            outer = 1 + ((j + int(side == "B")) % 2)
        else:
            outer = 1 + ((j + phase//8 + int(side == "A")) % 2)
        quota = [0]*(3-outer) + [1]*outer
        selected = None
        for restart in range(96):
            rng.shuffle(quota)
            cell, projected = [], set()
            for b in quota:
                legal = []
                for g in pools[b]:
                    if (g[0] % modulus, b) in projected:
                        continue
                    if any(g == h or
                           G.mul(G.inv(h), g) == (21,0)
                           for h in cell):
                        continue
                    if any(g[1] == h[1] and
                           42 // gcd((g[0]-h[0]) % 42, 42) < 7
                           for h in cell):
                        continue
                    legal.append(g)
                if not legal:
                    break
                g = rng.choice(legal)
                cell.append(g)
                projected.add((g[0] % modulus, b))
            if len(cell) != 3:
                continue
            anchor = G.inv(cell[0])
            relatives = [G.mul(anchor, g) for g in cell[1:]]
            if G.mul(relatives[0], relatives[1]) == G.mul(
                    relatives[1], relatives[0]):
                continue
            selected = cell
            break
        if selected is None:
            raise ValueError("Dic21 mixed-cubic residue recipe exhausted")
        row.append(selected)
    return [row]


def _d21_fiber_words(cols, fibers, q):
    """Actual kernels among bounded rotation-fiber unions.

    Syndromes are sums of VERIFIED columns. This restricted search
    neither exhausts all supports nor certifies quantum distance.
    """
    found = set()
    for h in (2,3,7):
        supports = [
            fiber << (block*q)
            for block in range(len(cols)//q)
            for fiber in fibers[h]]
        projected = []
        for support in supports:
            syndrome, word = 0, support
            while word:
                bit = word & -word
                syndrome ^= cols[bit.bit_length()-1]
                word ^= bit
            projected.append(syndrome)

        seen = {0: 0}

        def visit(syndrome, word):
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif word ^ old:
                found.add(word ^ old)

        for i, syndrome in enumerate(projected):
            visit(syndrome, supports[i])
        if h <= 3:
            for i, syndrome in enumerate(projected):
                for j in range(i):
                    visit(syndrome ^ projected[j],
                          supports[i] ^ supports[j])
    return tuple(sorted(found,
                        key=lambda w: (w.bit_count(), w))[:40])


def _d21_probe(checks, stabilizers, n, rng, trials, seeds=()):
    """Find genuine full-code logicals, including stabilizer dressings."""
    span = _d21_basis(stabilizers)
    cols = _d21_columns(checks, n)
    retained, best = set(), n+1

    def observe(word):
        nonlocal best, retained
        if not word or not _d21_outside(word, span):
            return
        weight = word.bit_count()
        if weight < best:
            best, retained = weight, {word}
        elif weight == best:
            retained.add(word)

    for word in seeds:
        if not any((word & row).bit_count() & 1 for row in checks):
            observe(word)

    order, memory = list(range(n)), []
    split = A_SHAPE[1]*B_SHAPE[1]*84
    for trial in range(trials):
        rng.shuffle(order)
        if trial % 3 != 2:
            order.sort(key=lambda c:
                       (c < split) if trial % 3 == 0 else (c >= split))
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome and word:
                dependencies.append(word)
                observe(word)

        words = sorted(set(dependencies+memory),
                       key=lambda w: (w.bit_count(), w))[:64]
        for sweep in range(2):
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in words[:32]:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                observe(word)
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
        discoveries = set(words)
        for i, word in enumerate(words[:32]):
            for other in words[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        logical = [w for w in dependencies if _d21_outside(w, span)]
        logical.extend(retained)
        for word in sorted(set(logical),
                           key=lambda w: (w.bit_count(), w))[:4]:
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda item: item[0])
                moves = [row for _, row in nearby[:12]]
                for row in moves:
                    candidate = word ^ row
                    if candidate.bit_count() < weight:
                        chosen, weight = candidate, candidate.bit_count()
                for i, row in enumerate(moves):
                    for other in moves[:i]:
                        candidate = word ^ row ^ other
                        if candidate.bit_count() < weight:
                            chosen, weight = candidate, candidate.bit_count()
                if chosen == word:
                    break
                word = chosen
                observe(word)
            discoveries.add(word)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:48]
        if best <= 4:
            break
    return best, tuple(sorted(retained)[:16])


def _d21_fingerprint(hx, hz, n):
    """Unequal invariants certify unequal uncoloured Tanner graphs."""
    from collections import Counter
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    local = [Counter() for _ in rows]
    overlaps = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            overlaps[(min(weights[i],weights[j]),
                      max(weights[i],weights[j]),size)] += 1
            if size:
                local[i][(weights[j],size)] += 1
                local[j][(weights[i],size)] += 1
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length()-1].append(weights[r])
            source ^= bit
    return (
        tuple(sorted(overlaps.items())),
        tuple(sorted(Counter(
            (weights[i],tuple(sorted(profile.items())))
            for i,profile in enumerate(local)).items())),
        tuple(sorted(Counter(
            tuple(sorted(checks)) for checks in incidents).items())))


@__import__("functools").lru_cache(maxsize=1)
def _d21_bank(G):
    """Optimize twenty policies and emit only invariant-distinct fillings."""
    q = G.q
    assert q == 84
    mask = (1 << q)-1
    zero_a = [[[],[]]]
    zero_b = [[[],[],[]]]
    fibers = {}
    for h in (2,3,7):
        unseen, partition = set(range(q)), []
        while unseen:
            g = G.elements[min(unseen)]
            orbit = {
                G.idx(G.mul(g,(t*(42//h),0))) for t in range(h)}
            assert len(orbit) == h
            unseen.difference_update(orbit)
            partition.append(sum(1 << t for t in orbit))
        fibers[h] = tuple(sorted(partition))

    bank, fingerprints = [], set()
    moduli = (7,14,21,6,3)
    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20,4)
        modulus = moduli[geometry]
        local = random.Random(300840121 + 100003*slot)

        def component_pool(side, quota):
            width = (2 if side == "A" else 3)*q
            strata, seen = {}, set()

            def merit(item):
                return (
                    min(item["d"],14),
                    -sum(w.bit_count() == item["d"]
                         for w in item["words"]),
                    -item["cycles"])

            def admit(grid):
                signature = tuple(sorted(
                    tuple(sorted(cell)) for cell in grid[0]))
                if signature in seen:
                    return
                seen.add(signature)
                A,B = ((grid,zero_b) if side == "A"
                       else (zero_a,grid))
                hx,hz,_ = lifted_matrices(G,A,B)
                if side == "A":
                    rows = [
                        sum(((row >> (j*3*q)) & mask) << (j*q)
                            for j in range(2))
                        for row in hx[:q]]
                else:
                    rows = [row & ((1 << width)-1) for row in hz[:q]]
                cols = _d21_columns(rows,width)
                if not _d21_clean4(cols):
                    return
                structured = _d21_fiber_words(cols,fibers,q)
                words = _d21_words(cols,local,1,structured)
                if not words:
                    return
                rank = gf2_rank(rows)
                item = dict(
                    grid=grid,cols=cols,rank=rank,words=words,
                    d=words[0].bit_count(),
                    cycles=_four_cycle_count(rows))
                bucket = strata.setdefault(rank,[])
                bucket.append(item)
                bucket.sort(key=merit,reverse=True)
                del bucket[2:]

            for attempt in range(quota):
                admit(_d21_draw(
                    local,G,side,modulus,mode,slot+attempt))

            for step in range(20):
                pool = sorted(
                    (item for bucket in strata.values() for item in bucket),
                    key=merit,reverse=True)
                if not pool:
                    admit(_d21_draw(
                        local,G,side,modulus,mode,slot+quota+step))
                    continue
                parent = local.choice(pool[:min(4,len(pool))])
                grid = [[list(cell) for cell in parent["grid"][0]]]
                replacement = _d21_draw(
                    local,G,side,modulus,mode,slot+quota+step)
                positions = list(range(len(grid[0])))
                pressure = [
                    1 + sum(((word >> (j*q)) & mask).bit_count()
                            for word in parent["words"][:8])
                    for j in positions]
                j = local.choices(positions,weights=pressure,k=1)[0]
                grid[0][j] = replacement[0][j]
                if step % 7 == 6:
                    other = (j+1) % len(positions)
                    grid[0][other] = replacement[0][other]
                admit(grid)

            pool = sorted(
                (item for bucket in strata.values() for item in bucket),
                key=merit,reverse=True)[:8]
            for item in pool:
                item["words"] = _d21_words(
                    item["cols"],local,16,item["words"])
                item["d"] = min(item["d"],
                                item["words"][0].bit_count())
            pool.sort(key=merit,reverse=True)
            return pool[:6]

        winner = None
        for retry in range(2):
            ap = component_pool("A",48+24*retry+4*(slot//20))
            bp = component_pool("B",64+24*retry+4*(slot//20))
            proposals = []
            for ai in ap:
                for bi in bp:
                    A,B = ai["grid"],bi["grid"]
                    hx,hz,n = lifted_matrices(G,A,B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (275*n <= 1000*k < 325*n):
                        continue
                    d = min(ai["d"],bi["d"],14)
                    proposals.append(dict(
                        A=A,B=B,hx=hx,hz=hz,n=n,k=k,ai=ai,bi=bi,
                        cycles=ai["cycles"]+bi["cycles"],
                        key=(k*d*d,d,-ai["cycles"]-bi["cycles"])))
            proposals.sort(key=lambda item: item["key"],reverse=True)

            finalists = []
            for item in proposals[:10]:
                hx,hz,n = item["hx"],item["hz"],item["n"]
                if _has_small_logical(hx,hz,n):
                    continue
                if _has_small_logical(hz,hx,n):
                    continue
                fp = _d21_fingerprint(hx,hz,n)
                if fp in fingerprints:
                    continue

                a_seeds = [
                    sum(((word >> (j*q)) & mask)
                        << ((j*3+beta)*q) for j in range(2))
                    for word in item["ai"]["words"][:24]
                    for beta in range(3)]
                b_seeds = [
                    word << (j*3*q)
                    for word in item["bi"]["words"][:24]
                    for j in range(2)]
                dx,wx = _d21_probe(hz,hx,n,local,4,b_seeds)
                dz,wz = _d21_probe(hx,hz,n,local,4,a_seeds)
                if dx > n or dz > n:
                    continue
                item.update(dx=dx,dz=dz,wx=wx,wz=wz,fp=fp)
                finalists.append(item)
                if len(finalists) == 4:
                    break

            def merit(item):
                d = min(item["dx"],item["dz"],
                        item["ai"]["d"],item["bi"]["d"],14)
                return (item["k"]*d*d/item["n"],d,-item["cycles"])

            # Every finalist receives an independent deeper full-code search.
            for item in finalists:
                dx,wx = _d21_probe(
                    item["hz"],item["hx"],item["n"],
                    local,12,item["wx"])
                dz,wz = _d21_probe(
                    item["hx"],item["hz"],item["n"],
                    local,12,item["wz"])
                item["dx"],item["dz"] = min(item["dx"],dx),min(item["dz"],dz)
                item["wx"],item["wz"] = wx,wz
            if finalists:
                winner = max(finalists,key=merit)
                break

        if winner is None:
            bank.append(None)
            continue
        A,B = winner["A"],winner["B"]
        hx,hz,n = winner["hx"],winner["hz"],winner["n"]
        assert n == 588 and css_ok(hx,hz)
        assert max(row.bit_count() for row in hx+hz) <= W_MAX
        assert 275*n <= 1000*winner["k"] < 325*n
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa,sb) <= W_MAX
        assert _no_dead_lines(sa,sb)
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A,B) for row in grid for cell in row)
        fingerprints.add(winner["fp"])
        bank.append(dict(
            cells={
                "A": tuple(tuple(tuple(cell) for cell in row) for row in A),
                "B": tuple(tuple(tuple(cell) for cell in row) for row in B)},
            k=winner["k"],card=(geometry,mode),
            selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the card's fresh cubic, witness-repaired Dic21 filling."""
    selected = _d21_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh Dic21 cubic filling passed this policy")
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
# REFUTED: Exhaustive compatibility-clique assembly raises the order-40 bed's measured leaders to k=264; all five measured current leaders instead retained k=252 and d30s=5.
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
