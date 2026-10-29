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
CONCEPT = ("Dic11, order 44, shapes (3,2)x(3,2): quotient-matched odd trinomials should give five measured distance readings >=22 at k>=60, exceeding fitness 46.5234 in cell 0.100. Five [572,60,22] codes would score 50.7692; this is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Keep Dic11 and q=44 unchanged. Its measured (2,3) bed reached five d30s=24 readings and fitness 46.5234. The normal odd-order C11 averaging projector splits off F2[C4], allowing a controlled rank contribution. The shown quotient-matched binomial program gained k=60 but scored 36.3566 with measured distances 18..20; odd trinomials test a different distance/rank tradeoff on the same group.")
SHAPES = ("Change both sides to (3,2): each tall constituent has 88 variables and 132 checks, n=572, and rank counting still guarantees k>=44, rate 1/13. The size window allows q<=46 and retains q=44. Verified constituent transposes screen 132-coordinate cokernels. Parent minimum witnesses occupied sector-1 Acol/Bcol blocks; new cokernel witnesses are embedded in sector-2 Arow/Brow blocks. Transposition alone is an archived equivalence, so publication requires the new quotient-matched filling and intrinsic slot separation.")
FILLING = ("Use support ((1,1),(1,3),(3,1)): row totals (2,4,4), column totals (5,5), and BOTH CSS maxima nine. Four monomial cells are normalized to identity. Every trinomial has one nonidentity C11 element and a pair in one other C11 coset, hence C4 projection equal to identity. Both quotient constituents have rank one: their LP contributes 20 dimensions, while rank counting on the complementary 40-dimensional summand contributes at least 40, giving k>=60. Twenty endpoint-ratio cards vary genuine relative labels. Equally refined verified cokernel probes, full ranks, CSS, exact logical-four rejection, stabilizer-pair descent and uncoloured Tanner-invariant buckets screen publication. Larger readings and unseen-archive novelty remain uncertified.")
FAMILY_NAME = "r010_dic11_32x32_quotientmatched_oddtri_cokernel_v6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    if q != 44:
        raise ValueError("Dic11 has order 44")
    # x^22=1, y^2=x^11, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((a + 11 * (b // 2)) % 22, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11 * b * d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return 22 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 2)               # (ra, ca)
B_SHAPE = (3, 2)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize complete fillings; CELL only returns the selected cells."""
    sa = sb = ((1, 1), (1, 3), (3, 1))
    ratios = (1, 2, 3, 4, 5)
    plans = tuple((a, b) for a in range(5) for b in range(5) if a != b)
    card = idx % len(plans)
    pa, pb = plans[card]
    prescriptions = {
        "A": (ratios[pa], ratios[pb]),
        "B": (ratios[(pa + 2) % 5], ratios[(pb + 1) % 5]),
    }
    assert G.q == 44
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9 <= min(W_MAX, 9, 12)

    def quotient(g):
        # Dic11/<x^2> is C4: x maps to y^2.
        return (2 * g[0] + g[1]) % 4

    def draw_grid(profile, side):
        """Normalized tall constituent, with two genuine triangle ratios.

        In each triangle the reflected pair cancels in F2[C4].
        Its singleton projects to the identity. The complete quotient
        matrix therefore consists of six units, not six zero cells.
        """
        assert profile == sa
        triangles = []
        for ratio in prescriptions[side]:
            gap = rng.randrange(1, 11)
            singleton = (2 * ((ratio * gap) % 11), 0)
            first = (rng.randrange(22), 1)
            second = G.mul(first, (2 * gap, 0))
            cell = [singleton, first, second]
            assert singleton != G.e
            assert len({G._key(g) for g in cell}) == 3
            assert quotient(first) == quotient(second)
            projection = 0
            for g in cell:
                projection ^= 1 << quotient(g)
            assert projection == 1
            triangles.append(cell)

        # These four unit entries fix the monomial spanning-tree gauges.
        # They do not turn different triangle ratios into label draws.
        grid = [
            [[G.e], [G.e]],
            [[G.e], triangles[0]],
            [triangles[1], [G.e]],
        ]
        assert [[len(c) for c in row] for row in grid] == [list(r) for r in sa]
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
        # Normal C11=<x^2> has four cosets in Dic11. Enumerate the
        # complete invariant component subspace using syndromes from
        # the VERIFIED component matrix. In particular this finds
        # every component kernel word comprising one or two complete
        # C11 cosets, of weights 11 and 22.
        orbit_words = []
        orbit_syndromes = []
        for j in range(3):
            for b in range(2):
                for parity in range(2):
                    word = 0
                    syndrome = 0
                    for a in range(parity, 22, 2):
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
        """Screen the cokernel of the actual tall scaffold constituent.

        Zero auxiliary cells isolate the physical left/right action.
        Transposing these verified bit rows gives precisely the
        sector-2 check action used by the complete code.
        """
        if side == "A":
            tall, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, tall, _ = lifted_matrices(G, [[[]]], grid)
        assert len(tall) == 3 * q
        assert all(row.bit_length() <= 2 * q for row in tall)
        rows = columns(tall, 2 * q)
        assert len(rows) == 2 * q
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def constituent_finalists(profile, side):
        # Independent selection avoids discarding an excellent A merely
        # because its randomly paired B was weak, and conversely.
        pool = []
        cycle_bias = 12 + 8 * ((card + int(side == "B")) % 4)
        for trial in range(256):
            grid = draw_grid(profile, side)
            rows = verified_component(grid, side)
            stats = component_quality(rows, width)
            if stats[0]:
                continue
            cheap = (int(stats[3] <= 22), -stats[3],
                     cycle_bias * stats[1] + stats[2], stats[1])
            pool.append(dict(grid=grid, rows=rows, stats=stats,
                             cheap=cheap, trial=trial))
        pool.sort(key=lambda item: (item["cheap"], item["trial"]))
        if not pool:
            raise ValueError("No small-kernel-free constituent candidate")

        shortlist = []
        for record in pool[:24]:
            record["rank"] = gf2_rank(record["rows"])
            # The controlled C4 summand supplies four exact dependencies.
            assert record["rank"] <= 2 * q - 4
            distance, witness = kernel_probe(
                record["rows"], width, 4, record["stats"][4])
            record["distance"], record["witness"] = distance, witness
            record["quality"] = (
                distance, -record["rank"],
                -record["cheap"][2], -record["stats"][1])
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # Every publishable constituent receives exactly 4+28 passes.
        # A wider equally refined pool also supplies slot-bucket choices.
        finalists = shortlist[:8]
        for record in finalists:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 28, record["witness"])
            record["quality"] = (
                record["distance"], -record["rank"],
                -record["cheap"][2], -record["stats"][1])
        finalists.sort(key=lambda item: item["quality"], reverse=True)
        return finalists

    def intrinsic_bucket(hx, hz, n):
        """Permutation-invariant full Tanner features, without X/Z colours.

        Different buckets certify nonisomorphic Tanner graphs. Equality
        is inconclusive; neither this nor the ratio card certifies novelty
        against the unseen archive.
        """
        from hashlib import sha256

        rows = list(hx) + list(hz)
        cols = columns(rows, n)
        overlaps = [{} for _ in rows]
        for syndrome in cols:
            neighbors = []
            while syndrome:
                bit = syndrome & -syndrome
                neighbors.append(bit.bit_length() - 1)
                syndrome ^= bit
            for position, i in enumerate(neighbors):
                for j in neighbors[:position]:
                    overlaps[i][j] = overlaps[i].get(j, 0) + 1
                    overlaps[j][i] = overlaps[j].get(i, 0) + 1

        masks = [sum(1 << j for j in neighbors)
                 for neighbors in overlaps]
        weights = [row.bit_count() for row in rows]
        features = []
        for i, neighbors in enumerate(overlaps):
            histogram = {}
            for j, overlap in neighbors.items():
                feature = (
                    weights[j], overlap, len(overlaps[j]),
                    (masks[i] & masks[j]).bit_count())
                histogram[feature] = histogram.get(feature, 0) + 1
            features.append((
                weights[i], len(neighbors),
                tuple(sorted(histogram.items()))))

        palette = {feature: i for i, feature
                   in enumerate(sorted(set(features)))}
        colors = [palette[feature] for feature in features]
        qubit_profiles = {}
        for syndrome in cols:
            neighborhood = []
            while syndrome:
                bit = syndrome & -syndrome
                neighborhood.append(colors[bit.bit_length() - 1])
                syndrome ^= bit
            feature = tuple(sorted(neighborhood))
            qubit_profiles[feature] = qubit_profiles.get(feature, 0) + 1

        signature = (
            n, len(rows), tuple(sorted(features)),
            tuple(sorted(qubit_profiles.items())))
        digest = sha256(repr(signature).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % N_SLOTS

    pairs = []
    for restart in range(3):
        aa = constituent_finalists(sa, "A")
        bb = constituent_finalists(sb, "B")
        pending = []
        for ai, a in enumerate(aa):
            for bi, b in enumerate(bb):
                weak = min(a["distance"], b["distance"])
                priority = (
                    weak, a["distance"] + b["distance"],
                    -a["rank"] - b["rank"],
                    -(a["stats"][1] + b["stats"][1]), -ai, -bi)
                pending.append((priority, ai, bi))
        pending.sort(reverse=True)

        for _, ai, bi in pending:
            a, b = aa[ai], bb[bi]
            hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
            assert n == 572
            if intrinsic_bucket(hx, hz, n) != idx:
                continue
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            # C11 averaging is central because its order is odd.
            # The C4 LP contributes 20; the complementary summand has
            # dimension 40 and contributes at least another 40.
            assert k >= 60
            if not (8 * k < n):
                continue
            assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
            weak = min(a["distance"], b["distance"])
            quality = (
                k * weak * weak / float(n), weak,
                a["distance"] + b["distance"],
                -(a["stats"][1] + b["stats"][1]))
            pairs.append(dict(a=a, b=b, hx=hx, hz=hz, n=n,
                              k=k, quality=quality))
            if len(pairs) >= 4:
                break
        if pairs:
            break

    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No k>=60 cell-0.100 pair in the intrinsic slot bucket")

    def quantum_witness_reading(record):
        """Actual nontrivial logical witnesses, not a distance certificate.

        Embed verified constituent-cokernel witnesses in sector 2.
        Check commutation and nontriviality explicitly before reducing
        their weights by one- and two-stabilizer moves. Transposition
        does not supply a distance or an archive-novelty certificate.
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
                sector_two = A_SHAPE[1] * B_SHAPE[1] * q
                if kind == "Z":
                    # B^T cokernel word across Brow alpha, fixed Arow i.
                    word = record["b"]["witness"] << (
                        sector_two + block * width)
                else:
                    # A^T cokernel word across Arow i, fixed Brow alpha.
                    word = 0
                    component_word = record["a"]["witness"]
                    while component_word:
                        bit = component_word & -component_word
                        i, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << (
                            sector_two + (i * B_SHAPE[0] + block) * q + h)
                        component_word ^= bit
                if any((row & word).bit_count() & 1 for row in checks):
                    raise ValueError("Verified cokernel witness embedding failed")
                if not nontrivial(word):
                    continue
                found = True
                weight = word.bit_count()
                # Adding stabilizers preserves commutation and homology,
                # including representatives that also enter sector 1.
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
        "card": card,
        "endpoint_ratios": prescriptions,
        "intrinsic_bucket": idx,
        "controlled_quotient_k": 20,
    }


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
# REFUTED: Quotient-matched mixed-coset binomials at k>=60 exceed fitness 46.5234: the shown v5 program scored 36.3566, with measured distance readings 18..20.
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
