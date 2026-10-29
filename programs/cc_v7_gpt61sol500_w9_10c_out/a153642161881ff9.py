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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("D23, order 46, A=B=(2,3): all-odd saturated supports without quotient matching, independently optimized against explicit constituent and full-code witnesses, should supply five fresh k>=50 codes with measured distance readings >=24 and fitness above 46.5234 in cell 0.100. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from Dic11 to the dihedral group C23 semidirect C2 with inversion. D23 has 23 involutions, whereas Dic11 has one, so this is not a relabeling. Order 46 is the largest lift admitted by factor thirteen and gives n=598. This bed is absent from BEDS. The parent quotient-matched Dic11 program scored 36.3566 with measured distances 18..20; test a larger normal odd-order subgroup without imposing its failed quotient constraint.")
SHAPES = ("Keep equal (2,3) sides: each constituent has 138 variables and 92 checks, n=598, kb=46 and guaranteed rate 1/13. All-odd support ((1,1,3),(1,3,1)) adds two exact augmentation dependencies per CSS type, hence k>=50. Complete verified ranks require 50<=k<n/8. Parent minima broadly occupied sector-1 Acol/Bcol blocks; measured slot 44 also had mixed-sector weight-18 Z witnesses touching Arow0/Brow1 and Arow1/Brow0.")
FILLING = ("Twenty cards vary trinomial coset patterns and coupled nonzero C23 rotation gaps on support ((1,1,3),(1,3,1)). Each row totals five and columns total (2,4,4), giving BOTH CSS maxima nine, below both stated caps. Independently select equally refined constituents using verified lifts, exact C23-invariant kernel enumeration and randomized witness searches. Actual full ranks, CSS and exact logical exclusion through weight four gate publication; explicit logical witnesses receive stabilizer-pair descent. A pure deterministic portfolio conservatively excludes repeated strengthened complete Tanner invariants instead of coarse ownership buckets. Larger-distance survival and novelty against the unseen archive remain evaluator-tested hypotheses.")
FAMILY_NAME = "r010_dihedral23_q46_23x23_oddtri_gap_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("D23 has order 46")
    # x^23=y^2=1, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(23)]

    def red(g):
        return (int(g[0]) % 23, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c)) % 23,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 23, 0) if b == 0 else (a, 1)

    def idx(g):
        a, b = red(g)
        return 23 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _d23_candidate(rng, G, idx, forbidden):
    """Optimize one structural card against a read-only exclusion set."""
    assert G.q == 46
    sa = sb = ((1, 1, 3), (1, 3, 1))
    patterns = ((0, 0), (0, 1), (1, 1), (1, 0))
    ratios = (1, 2, 3, 5, 7)
    card, epoch = idx % 20, idx // 20
    pattern, ratio_index = divmod(card, 5)
    prescriptions = {
        "A": (patterns[pattern], ratios[ratio_index]),
        "B": (patterns[(pattern + 1 + epoch) % 4],
              ratios[(ratio_index + 2 + epoch) % 5]),
    }
    assert _no_dead_lines(sa, sb)
    wx = max(sum(row) + sum(sb[i][j] for i in range(2))
             for row in sa for j in range(3))
    wz = max(sum(row) + sum(sa[i][j] for i in range(2))
             for row in sb for j in range(3))
    assert wx == wz == 9
    assert max(wx, wz) <= min(W_MAX, 9, 12)

    def draw_grid(profile, side):
        assert profile == sa
        kinds, ratio = prescriptions[side]
        first_gap = rng.randrange(1, 23)
        gaps = (first_gap, ratio * first_gap % 23)
        grid, number = [], 0
        for counts in profile:
            row = []
            for count in counts:
                anchor = G.pick(rng)
                if count == 1:
                    cell = [anchor]
                else:
                    gap = gaps[number]
                    kind = kinds[number]
                    number += 1
                    if kind == 0:
                        # Complete relative-order spectrum (23,23,23).
                        other = rng.choice(
                            [a for a in range(1, 23) if a != gap])
                        relative = ((gap, 0), (other, 0))
                    else:
                        # Complete relative-order spectrum (23,2,2).
                        relative = ((gap, 0), (rng.randrange(23), 1))
                    cell = [anchor] + [
                        G.mul(anchor, g) for g in relative]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                row.append(cell)
            grid.append(row)
        assert number == 2
        return grid

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def component_quality(rows, width):
        """Small-kernel collision count and Tanner 4/6-cycle penalties."""
        cols = columns(rows, width)
        seen = {0}
        bad = 0
        for s in cols:
            bad += int(s in seen)
            seen.add(s)
        c4 = 0
        for i, s in enumerate(cols):
            for t in cols[:i]:
                u = s ^ t
                bad += int(u in seen)
                seen.add(u)
                overlap = _popcount(s & t)
                c4 += overlap * (overlap - 1) // 2

        # Project onto check nodes. With no 4-cycles, subtracting
        # within-column triangles leaves exactly the Tanner 6-cycles.
        adj = [0] * len(rows)
        internal = 0
        for s in cols:
            degree = _popcount(s)
            internal += degree * (degree - 1) * (degree - 2) // 6
            bits = s
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= s ^ bit
                bits ^= bit
        triangles = 0
        for u, neighbors in enumerate(adj):
            bits = neighbors >> (u + 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() + u
                triangles += _popcount((neighbors & adj[v]) >> (v + 1))
                bits ^= bit
        # Enumerate the COMPLETE normal-C23 invariant subspace.
        # D23/C23 has two cosets, so there are only six orbit variables.
        # This is exact within that subspace, not over the full kernel.
        orbit_words = []
        orbit_syndromes = []
        for j in range(3):
            for b in range(2):
                word = syndrome = 0
                for a in range(23):
                    position = j * G.q + G.idx((a, b))
                    word |= 1 << position
                    syndrome ^= cols[position]
                orbit_words.append(word)
                orbit_syndromes.append(syndrome)

        previous = 0
        word = syndrome = 0
        orbit_distance = width + 1
        orbit_witness = 0
        for number in range(1, 1 << len(orbit_words)):
            gray = number ^ (number >> 1)
            changed = (gray ^ previous).bit_length() - 1
            word ^= orbit_words[changed]
            syndrome ^= orbit_syndromes[changed]
            if syndrome == 0 and word.bit_count() < orbit_distance:
                orbit_distance = word.bit_count()
                orbit_witness = word
            previous = gray
        return (bad, c4, max(0, triangles - internal),
                orbit_distance, orbit_witness)

    def kernel_probe(rows, width, passes, seed):
        """Heuristic component-kernel upper bound and explicit witness.

        Random information sets expose kernel bases. Search every
        basis singleton, pair and triple, then descend by pairs.
        All returned witnesses are actual kernel words; the search
        is not exhaustive over the whole kernel.
        """
        best_word = seed
        best_weight = seed.bit_count() if seed else width + 1
        syndromes = columns(rows, width)
        for _ in range(passes):
            order = list(range(width))
            rng.shuffle(order)
            # Column elimination carries supports in the ORIGINAL
            # coordinates. Dependent columns give a systematic kernel
            # basis without rebuilding a permuted row matrix.
            pivots = {}
            basis = []
            for original in order:
                syndrome = syndromes[original]
                word = 1 << original
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if syndrome == 0:
                    basis.append(word)

            # Elementary basis exchanges preserve the entire kernel.
            # They expose different low-weight combinations before the
            # exhaustive singleton/pair/triple scan.
            rng.shuffle(basis)
            weights = [word.bit_count() for word in basis]
            for i in range(len(basis)):
                for j in range(len(basis)):
                    if i == j:
                        continue
                    trial = basis[i] ^ basis[j]
                    weight = trial.bit_count()
                    if 0 < weight < weights[i]:
                        basis[i], weights[i] = trial, weight
                if weights[i] < best_weight:
                    best_weight, best_word = weights[i], basis[i]

            for i, first in enumerate(basis):
                for j in range(i):
                    pair = first ^ basis[j]
                    weight = pair.bit_count()
                    if weight < best_weight:
                        best_weight, best_word = weight, pair
                    for h in range(j):
                        triple = pair ^ basis[h]
                        weight = triple.bit_count()
                        if weight < best_weight:
                            best_weight, best_word = weight, triple

            # Extend beyond three basis vectors without enumerating
            # the exponentially large component kernel.
            for descent in range(3):
                starting = best_word
                improved = False
                for i, first in enumerate(basis):
                    word = starting ^ first
                    weight = word.bit_count()
                    if 0 < weight < best_weight:
                        best_weight, best_word = weight, word
                        improved = True
                    for j in range(i):
                        word = starting ^ first ^ basis[j]
                        weight = word.bit_count()
                        if 0 < weight < best_weight:
                            best_weight, best_word = weight, word
                            improved = True
                if not improved:
                    break
            if best_weight <= 18:
                break

        assert best_word
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        return best_weight, best_word

    def low_logical(checks, stabilizers, width):
        """Exact existence test for a nontrivial logical of weight <=4."""
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

        syndromes = columns(checks, width)
        representatives = {0: 0}
        for i, syndrome in enumerate(syndromes):
            word = 1 << i
            if syndrome in representatives:
                if nontrivial(word ^ representatives[syndrome]):
                    return True
            else:
                representatives[syndrome] = word
        for i, syndrome in enumerate(syndromes):
            bit = 1 << i
            for j in range(i):
                s = syndrome ^ syndromes[j]
                word = bit | (1 << j)
                if s in representatives:
                    if nontrivial(word ^ representatives[s]):
                        return True
                else:
                    representatives[s] = word
        # All words of weight <=4 split into two words of weight <=2.
        # One representative per syndrome suffices because the
        # stabilizer space is linear.
        return False

    q = G.q
    width = 3 * q
    def verified_component(grid, side):
        """Zero auxiliary cells isolate the exact scaffold constituent."""
        zero = [[[]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, zero)
        else:
            _, rows, _ = lifted_matrices(G, zero, grid)
        assert len(rows) == 2 * q
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def constituent_finalists(profile, side):
        # Independent selection avoids discarding an excellent A merely
        # because its randomly paired B was weak, and conversely.
        pool = []
        cycle_bias = 12 + 8 * ((card + int(side == "B")) % 4)
        for trial in range(192):
            grid = draw_grid(profile, side)
            rows = verified_component(grid, side)
            stats = component_quality(rows, width)
            if stats[0]:
                continue
            cheap = (int(stats[3] <= 23), -stats[3],
                     cycle_bias * stats[1] + stats[2], stats[1])
            pool.append(dict(grid=grid, rows=rows, stats=stats,
                             cheap=cheap, trial=trial))
        pool.sort(key=lambda item: (item["cheap"], item["trial"]))
        if not pool:
            raise ValueError("No small-kernel-free constituent candidate")

        shortlist = []
        for record in pool[:12]:
            record["rank"] = gf2_rank(record["rows"])
            # The two all-odd augmentation rows coincide. Do not impose
            # quotient matching or seek the parent's four-row deficit.
            assert record["rank"] <= 2 * q - 1
            distance, witness = kernel_probe(
                record["rows"], width, 4, record["stats"][4])
            record["distance"], record["witness"] = distance, witness
            record["quality"] = (
                distance, -record["cheap"][2], -record["stats"][1])
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # Every publication-eligible constituent receives the same
        # four-plus-twenty-four-pass budget; witnesses only get lighter.
        finalists = shortlist[:6]
        for record in finalists:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 24, record["witness"])
            record["quality"] = (
                record["distance"], -record["cheap"][2],
                -record["stats"][1])
        finalists.sort(key=lambda item: item["quality"], reverse=True)
        return finalists

    def intrinsic_key(hx, hz, n):
        """Conservative complete-Tanner invariant, not an iso solver.

        Pool CSS types. Use exact check-intersection profiles, projected
        common neighbours, qubit degrees and three incidence refinements.
        Equal keys are rejected; different keys distinguish Tanner graphs
        under the stated check/qubit relabeling convention.
        """
        from collections import Counter
        from hashlib import sha256

        rows = list(hx) + list(hz)
        degrees = [row.bit_count() for row in rows]
        m = len(rows)
        masks = [0] * m
        overlaps = [{} for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    overlaps[i][j] = overlap
                    overlaps[j][i] = overlap
                    masks[i] |= 1 << j
                    masks[j] |= 1 << i

        features = []
        for i in range(m):
            profile = Counter(
                (degrees[j], overlaps[i].get(j, 0),
                 masks[j].bit_count(),
                 (masks[i] & masks[j]).bit_count())
                for j in range(m) if i != j)
            features.append((
                degrees[i], masks[i].bit_count(),
                tuple(sorted(profile.items()))))
        palette = {f: c for c, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]

        incidence = []
        for syndrome in columns(rows, n):
            neighbors = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbors.append(bit.bit_length() - 1)
                syndrome ^= bit
            incidence.append(neighbors)

        history = [tuple(sorted(Counter(features).items()))]
        for _ in range(3):
            qfeatures = [tuple(sorted(colors[r] for r in ns))
                         for ns in incidence]
            qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
            check_profiles = [[] for _ in rows]
            for ns, feature in zip(incidence, qfeatures):
                for r in ns:
                    check_profiles[r].append(qp[feature])
            refined = [
                (colors[r], tuple(sorted(check_profiles[r])))
                for r in range(m)]
            history.append((
                tuple(sorted(Counter(qfeatures).items())),
                tuple(sorted(Counter(refined).items()))))
            palette = {f: c for c, f in enumerate(sorted(set(refined)))}
            colors = [palette[f] for f in refined]

        signature = (
            n, m,
            tuple(sorted(Counter(len(ns) for ns in incidence).items())),
            tuple(history))
        return sha256(repr(signature).encode("ascii")).digest()

    aa = constituent_finalists(sa, "A")
    bb = constituent_finalists(sb, "B")
    pending = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["distance"], b["distance"])
            preliminary = (
                weak, a["distance"] + b["distance"],
                -(a["stats"][1] + b["stats"][1]), -ai, -bi)
            pending.append((preliminary, ai, bi))
    pending.sort(reverse=True)

    pairs, pair_keys = [], set()
    for _, ai, bi in pending:
        a, b = aa[ai], bb[bi]
        hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        # All-odd augmentation supplies four complete CSS dependencies.
        # Actual ranks, not a quotient rank prediction, decide admission.
        assert n == 598 and k >= 50
        if not (50 <= k and 8 * k < n):
            continue
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        signature = intrinsic_key(hx, hz, n)
        if signature in forbidden or signature in pair_keys:
            continue
        pair_keys.add(signature)
        weak = min(a["distance"], b["distance"])
        quality = (k * weak * weak / float(n), weak,
                   a["distance"] + b["distance"],
                   -(a["stats"][1] + b["stats"][1]))
        pairs.append(dict(a=a, b=b, hx=hx, hz=hz, n=n,
                          k=k, quality=quality, signature=signature))
        # Four distinct proposals receive identical complete screens.
        if len(pairs) >= 4:
            break
    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No fresh D23 pair in the target rate cell")

    def quantum_witness_reading(record):
        """Actual nontrivial logical witnesses, not a distance certificate.

        Embed the component witnesses using the verified sector-1 column
        order. Check commutation and nontriviality explicitly before
        reducing their weights by one- and two-stabilizer moves.
        """
        best = record["n"] + 1
        found = False
        for kind, checks, stabilizers in (
                ("X", record["hz"], record["hx"]),
                ("Z", record["hx"], record["hz"])):
            pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = row
                        break
                    row ^= old

            def nontrivial(word):
                while word:
                    p = word.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        return True
                    word ^= old
                return False

            for block in range(3):
                if kind == "X":
                    word = record["b"]["witness"] << (block * width)
                else:
                    word = 0
                    component_word = record["a"]["witness"]
                    while component_word:
                        bit = component_word & -component_word
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 3 + block) * q + h)
                        component_word ^= bit
                if any((row & word).bit_count() & 1 for row in checks):
                    raise ValueError("Component witness embedding failed")
                if not nontrivial(word):
                    continue
                found = True
                weight = word.bit_count()
                # Adding stabilizers preserves both commutation and
                # logical class, including when the support enters sector 2.
                for descent in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        trial = starting ^ first
                        trial_weight = trial.bit_count()
                        if trial_weight < weight:
                            word, weight = trial, trial_weight
                        for j in range(i):
                            trial = starting ^ first ^ stabilizers[j]
                            trial_weight = trial.bit_count()
                            if trial_weight < weight:
                                word, weight = trial, trial_weight
                    if word == starting:
                        break
                assert word and nontrivial(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
        return best if found else None

    # Compare equally refined pairs with the same quantum witness budget.
    # Screening additional pairs supplies fallbacks without allowing
    # unrefined constituents to win through a shorter search.
    survivors = []
    for record in pairs:
        hx, hz, n = record["hx"], record["hz"], record["n"]
        if low_logical(hz, hx, n) or low_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        reading = quantum_witness_reading(record)
        record["quantum_reading"] = reading
        weak = (reading if reading is not None else
                min(record["a"]["distance"], record["b"]["distance"]))
        record["final_quality"] = (
            int(reading is not None),
            record["k"] * weak * weak / float(n),
            weak, record["quality"])
        survivors.append(record)
        if len(survivors) >= 4:
            break
    if not survivors:
        raise ValueError("No selected pair passed the weight-four logical screen")
    winner = max(survivors, key=lambda item: item["final_quality"])
    return {
        "idx": idx,
        "cells": {"A": winner["a"]["grid"], "B": winner["b"]["grid"]},
        "component_probe": (
            winner["a"]["distance"], winner["b"]["distance"]),
        "quantum_witness_probe": winner["quantum_reading"],
        "k": winner["k"],
        "card": (card, epoch),
        "trinomial_prescriptions": prescriptions,
        "tanner_signature": winner["signature"],
    }


from functools import lru_cache


@lru_cache(maxsize=1)
def _d23_portfolio(G, count):
    """Pure deterministic catalogue; memoization changes only build cost."""
    entries, excluded = [], set()
    for position in range(count):
        winner = None
        for retry in range(3):
            attempt = position + retry * count
            local = random.Random(
                BASE_SEED ^ (0xD2346023 + 1000003 * attempt))
            try:
                proposal = _d23_candidate(
                    local, G, attempt, frozenset(excluded))
            except ValueError:
                continue
            signature = proposal["tanner_signature"]
            if signature in excluded:
                continue
            # Freeze published cells so cached results cannot be mutated.
            proposal["cells"] = {
                side: tuple(tuple(tuple(cell) for cell in row)
                            for row in proposal["cells"][side])
                for side in ("A", "B")}
            excluded.add(signature)
            winner = proposal
            break
        entries.append(winner)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish one complete design independent of slot invocation order."""
    assert G.q == 46 and 0 <= idx < N_SLOTS
    entry = _d23_portfolio(G, N_SLOTS)[idx]
    if entry is None:
        raise ValueError("fresh screened D23 structural card exhausted")
    assert 50 <= entry["k"] and 8 * entry["k"] < 598
    return dict(entry)


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh list from the already screened complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Weight-four rejection plus component cycle minimization excludes weight-20 mixed-sector logicals on Dic11 (2,3)x(2,3): measured slot 42 had d30s=20 with Z witnesses in both sectors.
# REFUTED: The v2 mixed-coset/component-kernel policy yields five measured readings >=24: its measured slots 8 and 9 instead had d30s=23, with both logical types entirely in sector 1.
# REFUTED: The v3 independently optimized saturated-constituent policy yields five measured readings >=25: all five measured codes instead had d30s=24, with minimum witnesses entirely in sector 1.
# REFUTED: Fully quotient-matched endpoint-ratio constituents raise five-code fitness above 46.5234 through k>=60: the measured v5 program scored 36.3566 with d30s=18..20, including mixed-sector weight-18 witnesses.
# REFUTED: The v5 intrinsic-bucket publisher supplies twenty inequivalent initial designs: its measured sixty-slot table retained eleven distinct codes, with numerous evaluator isomorphism duplicates and one bucket failure.
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
