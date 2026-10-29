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
CONCEPT = ("C5 semidirect C8 with action 2, order 40, shapes (1,3)x(1,3): independently packing two singular cubic constituents, excluding their words through five and minimizing verified weight-six multiplicity, should produce five fresh cell-0.450 codes with measured readings at least seven.")
GROUP = ("Keep the parent's metacyclic group and order byte-for-byte under L2. Its five measured [520,252,5] codes established clean singular cubic constituents but retained thousands of sector-one weight-five witnesses. Test replacing the binomial-rich constituent, not an assumed group-driven distance advantage.")
SHAPES = ("Change B to (1,3), retaining A=(1,3) and q=40: n=400, kb=160 and guaranteed rate 0.400. Equal components have 120 variables and 40 checks each; q<=59 is allowed, so retaining order 40 needs no group edit. Parent lightest witnesses occupied sector-one Acol/Bcol blocks, particularly Acol1 and Bcol2. Exact full ranks target 0.425<=k/n<0.475, deliberately addressing the weaker 0.450 frontier instead of widening the parent's half-rate bed.")
FILLING = ("All six cells are distinct-element trinomials: component row totals nine and column totals three give both CSS check weights twelve. Compile separate A/B actions exclusively from verified unit lifts. Enumerate normalized singular blocks, pack compatible column spans, and exactly exclude component words through five while counting weight-six words. Five rank/spread preferences cross four quotient-occupancy preferences. Independently deepen actual kernel witnesses, require complete stabilizer-aware logical-four exclusion, and probe full logical classes before selection. Unequal uncoloured Tanner invariants ensure emitted designs are not orbit copies. Found witness readings remain upper bounds.")
FAMILY_NAME = "r045_C5sdC8_action2_13x13_dualcubic_exact5_v1"

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
def _dualcubic_bank(G):
    """Fresh equal-shape optimization; the historical bank is inactive."""
    from collections import Counter
    from itertools import combinations

    assert G.q == 40 and A_SHAPE == B_SHAPE == (1, 3)
    q, width = G.q, 3 * G.q
    mask = (1 << q) - 1
    stub = [[[G.e] for _ in range(3)]]

    def columns(rows, size):
        out = [0] * size
        for r, original in enumerate(rows):
            word = original
            while word:
                bit = word & -word
                out[bit.bit_length() - 1] |= 1 << r
                word ^= bit
        return tuple(out)

    def rows_from_columns(cols):
        rows = [0] * q
        for c, original in enumerate(cols):
            word = original
            while word:
                bit = word & -word
                rows[bit.bit_length() - 1] |= 1 << c
                word ^= bit
        return rows

    def canonical_span(words):
        basis = {}
        for original in words:
            word = original
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old
        for p in sorted(basis):
            for h in sorted(basis):
                if h > p and ((basis[h] >> p) & 1):
                    basis[h] ^= basis[p]
        return tuple(basis[p] for p in sorted(basis))

    def clean4(cols):
        """Exact conservative component-kernel exclusion through four."""
        seen = {0}
        for col in cols:
            if col in seen:
                return False
            seen.add(col)
        for i, col in enumerate(cols):
            for other in cols[:i]:
                syndrome = col ^ other
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def profile56(cols):
        """After clean4, exactly exclude five and count six.

        A weight-five word has a 3+2 partition. Once words through
        five are absent, equal triple syndromes have disjoint supports;
        each weight-six support contributes ten triple-pair collisions.
        """
        low = {0}
        low.update(cols)
        for i, col in enumerate(cols):
            low.update(col ^ other for other in cols[:i])
        triples, collisions = {}, 0
        for i in range(2, len(cols)):
            x = cols[i]
            for j in range(1, i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in low:
                        return None
                    previous = triples.get(syndrome, 0)
                    collisions += previous
                    triples[syndrome] = previous + 1
        assert collisions % 10 == 0
        return collisions // 10

    def kernel_probe(cols, local, trials, memory=()):
        """Retain actual kernel words; a large reading is not a certificate."""
        order = list(range(len(cols)))
        retained = sorted(set(memory), key=int.bit_count)[:32]
        best = min((w.bit_count() for w in retained),
                   default=len(cols) + 1)
        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
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
                    words.append(word)
                    best = min(best, word.bit_count())
            for sweep in range(2):
                donors = sorted(set(words + retained),
                                key=int.bit_count)[:32]
                for i, word in enumerate(words):
                    weight = word.bit_count()
                    for other in donors:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            word, weight = trial, tw
                    words[i] = word
                    best = min(best, weight)
            short = sorted(set(words + retained), key=int.bit_count)[:32]
            extras = []
            for i, word in enumerate(short):
                for other in short[:i]:
                    trial = word ^ other
                    if trial:
                        best = min(best, trial.bit_count())
                        extras.append(trial)
            retained = sorted(set(short + extras),
                              key=lambda w: (w.bit_count(), w))[:32]
        return best, retained

    # Both atlases are compiled from the evaluator's verified convention.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [G.e], [G.e]]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(columns([r & mask for r in source[:q]], q))
        atlas[side] = tuple(blocks)

    def block(side, cell):
        out = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(contribution):
                out[h] ^= syndrome
        return tuple(out)

    def component_pool(side):
        """Pack different block incidences, not translated cell copies."""
        classes = {}
        nonidentity = [g for g in G.elements if g != G.e]
        for g, h in combinations(nonidentity, 2):
            cell = (G.e, g, h)
            cols = block(side, cell)
            space = canonical_span(cols)
            deficit = q - len(space)
            if deficit < 4 or not clean4(cols):
                continue
            incidence = tuple(sorted(cols))
            classes.setdefault(space, {}).setdefault(
                incidence, (cell, cols))

        local = random.Random(40133017 + (side == "B") * 100003)
        raw, seen = [], set()
        for space, bucket in sorted(
                classes.items(), key=lambda item: (-len(item[0]), item[0])):
            entries = tuple(bucket.values())
            if len(entries) < 3:
                continue
            accepted = []
            for attempt in range(256):
                selected = local.sample(entries, 3)
                selected.sort(key=lambda item: item[0])
                signature = tuple(item[0] for item in selected)
                if signature in seen:
                    continue
                seen.add(signature)
                cols = tuple(s for item in selected for s in item[1])
                if not clean4(cols):
                    continue
                rows = rows_from_columns(cols)
                deficit = q - gf2_rank(rows)
                assert deficit == q - len(space)
                cells = [[list(item[0]) for item in selected]]
                spread = sum(len({g[1] for g in cell})
                             for cell in cells[0])
                occupancy = tuple(sorted(
                    tuple(sorted(Counter(g[1] % 4 for g in cell).values()))
                    for cell in cells[0]))
                accepted.append(dict(
                    cells=cells, cols=cols, deficit=deficit,
                    cycles=_four_cycle_count(rows),
                    spread=spread, occupancy=occupancy))
                if len(accepted) >= 12:
                    break
            accepted.sort(key=lambda item: (
                item["cycles"], -item["spread"], item["cells"]))
            raw.extend(accepted[:4])

        # Keep incidence and singularity diversity before expensive screens.
        buckets = {}
        for item in raw:
            key = item["deficit"], item["spread"], item["occupancy"]
            buckets.setdefault(key, []).append(item)
        ordered = []
        for key in sorted(buckets):
            bucket = sorted(buckets[key], key=lambda x: x["cycles"])
            ordered.extend(bucket[:4])
        ordered.sort(key=lambda item: (
            item["cycles"], -item["deficit"], -item["spread"]))

        pool = []
        for item in ordered[:96]:
            six = profile56(item["cols"])
            if six is None:
                continue
            d, words = kernel_probe(item["cols"], local, 12)
            assert words and d == min(w.bit_count() for w in words)
            item.update(six=six, d=d, words=words)
            pool.append(item)
        pool.sort(key=lambda item: (
            item["d"], -item["six"], -item["cycles"]), reverse=True)
        return pool

    pools = {side: component_pool(side) for side in ("A", "B")}
    if not pools["A"] or not pools["B"]:
        raise ValueError("no compatible singular cubic component clean through five")

    def fingerprint(hx, hz, n):
        """Uncoloured Tanner invariant, including overlap triangles."""
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        adjacency = [0] * len(rows)
        neighborhoods = [Counter() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
                    neighborhoods[i][weights[j], overlap] += 1
                    neighborhoods[j][weights[i], overlap] += 1
        profiles = []
        for i, neighbors in enumerate(adjacency):
            remaining, triangles = neighbors, 0
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() - 1
                triangles += (neighbors & adjacency[j]).bit_count()
                remaining ^= bit
            profiles.append((weights[i],
                             tuple(sorted(neighborhoods[i].items())),
                             triangles // 2))
        qubits = [[] for _ in range(n)]
        for i, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                qubits[bit.bit_length() - 1].append(weights[i])
                row ^= bit
        return (tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(tuple(sorted(v))
                                     for v in qubits).items())))

    def full_probe(checks, stabilizers, n, local, trials):
        """Search genuine logical classes, including stabilizer dressing."""
        cols = columns(checks, n)
        span = {}
        for original in stabilizers:
            word = original
            while word:
                p = word.bit_length() - 1
                old = span.get(p)
                if old is None:
                    span[p] = word
                    break
                word ^= old

        def outside(word):
            while word:
                p = word.bit_length() - 1
                old = span.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        best, order = n + 1, list(range(n))
        for _ in range(trials):
            local.shuffle(order)
            pivots, words = {}, []
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
                    words.append(word)
            logical = sorted((w for w in words if outside(w)),
                             key=int.bit_count)[:12]
            donors = sorted(words, key=int.bit_count)[:32]
            for seed in logical:
                word, weight = seed, seed.bit_count()
                for sweep in range(2):
                    for other in donors:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if trial and tw < weight and outside(trial):
                            word, weight = trial, tw
                    for row in stabilizers:
                        trial = word ^ row
                        tw = trial.bit_count()
                        if tw < weight:
                            word, weight = trial, tw
                best = min(best, weight)
            for i, word in enumerate(logical):
                for other in logical[:i]:
                    trial = word ^ other
                    if trial and outside(trial):
                        best = min(best, trial.bit_count())
        return best

    # Twenty deliberate ranking cards; additional harness slots must
    # still pass the same incidence-distinctness gate.
    desired_deficit = (4, 8, 4, 8, 4)
    desired_spread = (6, 7, 8, 9, 9)
    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        policy, model = divmod(slot % 20, 4)
        local = random.Random(40133491 + 100003 * slot)

        def card_order(side):
            entries = list(pools[side])
            local.shuffle(entries)
            entries.sort(key=lambda item: (
                item["d"],
                -abs(item["deficit"] - desired_deficit[policy]),
                -abs(item["spread"] - desired_spread[policy]),
                -abs(sum(len(t) for t in item["occupancy"]) - (6 + model)),
                -item["six"], -item["cycles"]), reverse=True)
            return entries[:12]

        proposals = []
        for a in card_order("A"):
            for b in card_order("B"):
                A, B = a["cells"], b["cells"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (425 * n <= 1000 * k < 475 * n):
                    continue
                d = min(a["d"], b["d"])
                key = (k * d * d, d, -a["six"] - b["six"],
                       -a["cycles"] - b["cycles"])
                proposals.append((key, a, b, hx, hz, n, k))
        local.shuffle(proposals)
        proposals.sort(key=lambda item: item[0], reverse=True)

        finalists = []
        for key, a, b, hx, hz, n, k in proposals:
            fp = fingerprint(hx, hz, n)
            if fp in emitted:
                continue
            if _has_small_logical(hx, hz, n):
                continue
            if _has_small_logical(hz, hx, n):
                continue
            # Equal independent deepening budgets for both constituents.
            da, aw = kernel_probe(a["cols"], local, 24, a["words"])
            db, bw = kernel_probe(b["cols"], local, 24, b["words"])
            dx = full_probe(hz, hx, n, local, 4)
            dz = full_probe(hx, hz, n, local, 4)
            d = min(da, db, dx, dz)
            score = (k * d * d, d, key[2], key[3])
            finalists.append((score, a, b, hx, hz, n, k, fp))
            if len(finalists) >= 4:
                break

        if not finalists:
            bank.append(None)
            continue
        score, a, b, hx, hz, n, k, fp = max(
            finalists, key=lambda item: item[0])
        assert css_ok(hx, hz)
        assert n == 400 and max(r.bit_count() for r in hx + hz) == 12
        assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
        assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
        emitted.add(fp)
        bank.append({
            "cells": {
                side: tuple(tuple(tuple(cell) for cell in row)
                            for row in item["cells"])
                for side, item in (("A", a), ("B", b))
            },
            "k": k, "card": slot % 20,
            "selection": score,
            "component_six_counts": (a["six"], b["six"]),
        })
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh equal-shape cubic card; never call the legacy bank."""
    selected = _dualcubic_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh dual-cubic design in target cell 0.450")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return every new-shape cell from the jointly screened filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting odd-C8 binomial differences and four-cycle selection suffice to remove weight-four X logicals; all five measured parent codes have d30s=4, including the k=242 rank-deficit winners.
# REFUTED: Rotation-cage binomials plus mandatory logical-four exclusion raise this bed's measured distance above five; all five measured current-parent codes had d30s=5, and repeated connected-binomial recipes produced isomorphic duplicates.
# REFUTED: Shared-cokernel cubic enrichment with rank-first selection removes the distance-five bottleneck on the (1,3)x(1,4) bed; all five measured current codes instead had k=252 and d30s=5, with thousands of sector-one weight-five witnesses.
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
