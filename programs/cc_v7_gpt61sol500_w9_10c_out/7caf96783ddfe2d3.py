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
CONCEPT = ("C5 semidirect_-1 C2, order 10, A=B=(2,7): connected seven-matching constituents and singly grounded six-matching alternatives, admitted only with exact quantum distance four, should supply five novel cell-0.500 finalists exceeding fitness 7.8. Archive novelty and sufficient portfolio diversity are falsifiable hypotheses.")
GROUP = ("Retain the parent's inversion-semidir­ect construction rule, changing m=3,r=16 to m=5,r=2 because factor 53 excludes order 48. Order ten gives n=530 and is the largest admissible even order for this shape. The recorded Z5:Z2(s=4) (2,7)x(2,7) bed reached fitness 4.6; test exact short-logical certification on this bed rather than attributing performance to relative orders.")
SHAPES = ("Equal (2,7) constituents have 70 variables and 20 checks, n=530, kb=250 and guaranteed rate 25/53=0.471698, just below the requested band. Actual verified ranks must give 0.475<=k/n<0.525. Connected ungrounded matching pairs are expected to have k=262; singly grounded alternatives k=255. Parent weight-nine X minima broadly occupied sector-1 Acol0/1 across all five Bcols, with almost no sector-2 minimum. This move deliberately trades that distance for higher rate, rather than claiming to strengthen the parent's witnesses.")
FILLING = ("Enumerate finite gauge-normalized seven-matching constituents and six-matching constituents with one pendant singleton. Ungrounded rows total seven; grounded rows total seven/six; column degrees are two, or one/two. BOTH CSS check maxima remain at most nine. Complete verified lifts exclude every kernel word through weight three, exhibit an explicit nontrivial weight-four logical, enforce actual target-band ranks and CSS, and therefore certify distance exactly four. Rooted uncoloured Tanner invariants conservatively reject repeated published graphs. Prefer ungrounded pairs before singly grounded diversity fallbacks; at least three novel k262 finalists would lift the five-code score above 7.8. Unshown archive novelty remains unverified.")
FAMILY_NAME = "r050_c5c2_inv_q10_27x27_exact4_matching_pendant_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect C2 has order 10")
    # Same inversion-semidir­ect rule, with new parameters m=5,r=2.
    # x^5=y^2=1, y*x*y^-1=x^-1; element x^a y^b.
    m, r = 5, 2
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        return ((a if b & 1 else -a) % m, (-b) % r)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _unused_dic11_reference(rng, G, idx):
    """Inactive historical reference; never called by the new NEW_CTX."""
    profiles = (
        ((1, 2, 2), (2, 1, 2)),
        ((1, 1, 3), (2, 2, 1)),
        ((1, 1, 3), (1, 3, 1)),
        ((1, 2, 2), (1, 2, 2)),
    )
    # Ten unordered incidence pairs, each with two coset prescriptions.
    # The last two profiles have degrees (2,4,4), not permutation
    # copies of the parent's (3,3,3)/(3,3,4) profiles.
    plans = tuple((a, b) for a in range(4) for b in range(a, 4))
    card = idx % 20
    pa, pb = plans[card // 2]
    mode = card % 2
    epoch = idx // 20
    sa, sb = profiles[pa], profiles[pb]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(W_MAX, 9)

    def draw_grid(profile, side):
        doubles = [(i, j) for i in range(2) for j in range(3)
                   if profile[i][j] == 2]
        doubles.sort(key=lambda ij: (
            int(profile[0][ij[1]] != profile[1][ij[1]]),
            (ij[0] + int(side == "B")) % 2, ij[1]))
        number = len(doubles)
        quota = number if mode else (number + 1) // 2
        mixed = set(doubles[:quota])
        # A pure-binomial degree-four column must cross the quotient
        # somewhere: two rotation-only cells can retain a short
        # invariant annihilator. This is a construction constraint,
        # not an assertion about the final distance.
        for j in range(3):
            if profile[0][j] == profile[1][j] == 2:
                if not any((i, j) in mixed for i in range(2)):
                    mixed.add(((j + int(side == "B") + epoch) % 2, j))

        parity = (epoch + int(side == "B") + card // 2) % 2
        rotation_steps = [
            a for a in range(1, 22)
            if a != 11 and (mode or a % 2 == parity)
        ]
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, count in enumerate(counts):
                first = G.pick(rng)
                if count == 1:
                    cell = [first]
                elif count == 2:
                    relative = ((rng.randrange(22), 1)
                                if (i, j) in mixed else
                                (rng.choice(rotation_steps), 0))
                    cell = [first, G.mul(first, relative)]
                else:
                    rotation = (rng.choice(rotation_steps), 0)
                    cross = (rng.randrange(22), 1)
                    # Change both the within-coset pair and its placement
                    # relative to the singleton in the other coset.
                    if (i + j + mode + int(side == "B") + epoch) % 2:
                        second = G.mul(first, cross)
                        cell = [first, second, G.mul(second, rotation)]
                    else:
                        cell = [first, G.mul(first, rotation),
                                G.mul(first, cross)]
                assert len({G._key(g) for g in cell}) == count
                row.append(cell)
            grid.append(row)
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
    mask = (1 << q) - 1
    dummy = [[[G.e] for _ in range(3)] for _ in range(2)]

    def verified_component(grid, side):
        """The other side is a dummy; no separate lift convention."""
        A, B = (grid, dummy) if side == "A" else (dummy, grid)
        hx, hz, _ = lifted_matrices(G, A, B)
        if side == "B":
            return [row & ((1 << width) - 1) for row in hz[:2 * q]]
        return [
            sum(((hx[(i * 3) * q + r] >> (j * 3 * q)) & mask)
                << (j * q) for j in range(3))
            for i in range(2) for r in range(q)
        ]

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
        for record in pool[:20]:
            distance, witness = kernel_probe(
                record["rows"], width, 4, record["stats"][4])
            record["distance"], record["witness"] = distance, witness
            record["quality"] = (
                distance, -record["cheap"][2], -record["stats"][1])
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # All published constituent finalists receive this same budget.
        # Carrying witnesses forward makes each reading nonincreasing.
        finalists = shortlist[:4]
        for record in finalists:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 28, record["witness"])
            record["quality"] = (
                record["distance"], -record["cheap"][2],
                -record["stats"][1])
        finalists.sort(key=lambda item: item["quality"], reverse=True)
        return finalists

    aa = constituent_finalists(sa, "A")
    bb = constituent_finalists(sb, "B")
    pairs = []
    for a in aa:
        for b in bb:
            hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (k > 0 and 8 * k < n):
                continue
            weak = min(a["distance"], b["distance"])
            quality = (k * weak * weak / float(n), weak,
                       a["distance"] + b["distance"],
                       -(a["stats"][1] + b["stats"][1]))
            pairs.append(dict(a=a, b=b, hx=hx, hz=hz, n=n,
                              k=k, quality=quality))
    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No constituent pair in the target rate cell")

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
        "card": card,
    }


from functools import lru_cache


def _c16_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _c16_rows(G, grid, side):
    """Isolate the EXACT scaffold constituent with a zero dummy side."""
    width = len(grid[0]) * G.q
    zero = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _c16_clean4(cols):
    """Exact exclusion of every nonzero kernel word of weight 1..4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _c16_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _c16_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c16_signature(rows, width):
    """Conservative invariant of the uncoloured bipartite Tanner graph.

    Check intersections, common-neighbour profiles and qubit neighbourhood
    colour multisets survive arbitrary qubit/check relabeling. Equality
    is inconclusive and is rejected; inequality separates isomorphisms.
    Published checks have degree 8/9, qubits degree 5/6/12, so the two
    bipartite vertex classes are intrinsically distinguishable.
    """
    from collections import Counter
    from hashlib import sha256

    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            count = (row & rows[j]).bit_count()
            if count:
                overlaps[(i, j)] = count
                adjacency = one if count == 1 else many
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlaps.get((i, j), 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(rows[j].bit_count(),) + feature] += 1
            profiles[j][(rows[i].bit_count(),) + feature] += 1
    features = [
        (row.bit_count(), tuple(sorted(profile.items())))
        for row, profile in zip(rows, profiles)]
    palette = {feature: i
               for i, feature in enumerate(sorted(set(features)))}
    colors = [palette[feature] for feature in features]
    qubits = Counter()
    for col in _c16_columns(rows, width):
        neighborhood = []
        while col:
            bit = col & -col
            neighborhood.append(colors[bit.bit_length() - 1])
            col ^= bit
        qubits[tuple(sorted(neighborhood))] += 1
    value = (width, tuple(sorted(Counter(features).items())),
             tuple(sorted(qubits.items())))
    return sha256(repr(value).encode("ascii")).digest()


def _c16_probe(rows, width, rng, passes, seed=0):
    """Heuristic kernel upper bound carried by an explicit verified word."""
    cols = _c16_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        if word and word.bit_count() < weight:
            best, weight = word, word.bit_count()

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        rng.shuffle(kernel)
        # Elementary exchanges retain a complete kernel basis.
        for i in range(len(kernel)):
            word = kernel[i]
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if trial and trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)

        ranked = sorted(kernel, key=int.bit_count)
        short = ranked[:24]
        tail = ranked[24:]
        short += rng.sample(tail, min(8, len(tail)))
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        for _ in range(2):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break
    assert best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _c16_core_data(G, g, h):
    """Exact girth, distances and path supports of the verified B core."""
    grid = [[[G.e], [G.e], [G.e]],
            [[G.e], [g], [h]]]
    rows, width = _c16_rows(G, grid, "B")
    cols = _c16_columns(rows, width)
    adjacency = [[] for _ in rows]
    endpoints = set()
    for edge, col in enumerate(cols):
        if col.bit_count() != 2:
            return None
        bit = col & -col
        a = bit.bit_length() - 1
        b = (col ^ bit).bit_length() - 1
        pair = (min(a, b), max(a, b))
        if pair in endpoints:
            return None
        endpoints.add(pair)
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))

    distances, paths = [], []
    cycle = 0
    for root in range(len(rows)):
        distance = [10000] * len(rows)
        path = [0] * len(rows)
        distance[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if distance[b] == 10000:
                    distance[b] = distance[a] + 1
                    path[b] = path[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = path[a] ^ path[b] ^ (1 << edge)
                    if word and (not cycle
                                 or word.bit_count() < cycle.bit_count()):
                        cycle = word
                        if cycle.bit_count() < 8:
                            return None
        if len(queue) != len(rows):
            return None
        distances.append(tuple(distance))
        paths.append(tuple(path))
    if not cycle:
        return None
    assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
    return dict(g=g, h=h, columns=tuple(cols),
                distances=tuple(distances), paths=tuple(paths),
                seed=cycle, girth=cycle.bit_count())


def _c16_two_cubic_word(rows, width, data, q):
    """Exact minimum T-join for every anchored pair of cubic variables.

    Translation of the verified right constituent lets each two-cubic
    support contain an identity-coordinate cubic variable. This searches
    the two-cubic family only, not all constituent or quantum logicals.
    """
    cols = _c16_columns(rows, width)
    assert tuple(cols[:3*q]) == data["columns"]
    assert all(col.bit_count() == 3 for col in cols[3*q:])
    best = data["seed"]
    distances, paths = data["distances"], data["paths"]

    def join(boundary, budget):
        if not boundary:
            return (0, 0)
        if budget < 0 or boundary.bit_count() > 2 * budget:
            return None
        bit = boundary & -boundary
        a = bit.bit_length() - 1
        rest = boundary ^ bit
        choices, answer = rest, None
        while choices:
            other = choices & -choices
            b = other.bit_length() - 1
            distance = distances[a][b]
            if distance <= budget:
                tail = join(rest ^ other, budget - distance)
                if tail is not None:
                    value = (distance + tail[0],
                             paths[a][b] ^ tail[1])
                    if answer is None or value[0] < answer[0]:
                        answer = value
            choices ^= other
        return answer

    for anchor in (3*q + G_IDENTITY_INDEX,
                   4*q + G_IDENTITY_INDEX):
        for other in range(3*q, width):
            if other == anchor:
                continue
            result = join(cols[anchor] ^ cols[other],
                          best.bit_count() - 3)
            if result is not None:
                word = result[1] ^ (1 << anchor) ^ (1 << other)
                if word.bit_count() < best.bit_count():
                    best = word
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


# The new custom element list starts with its identity.
G_IDENTITY_INDEX = 0


@lru_cache(maxsize=1)
def _c16_portfolio(G, count):
    """Pure deterministic portfolio; cached outputs are immutable."""
    from itertools import combinations

    q, e = G.q, G.e
    if q != 48:
        raise ValueError("reinforced-core portfolio requires order 48")
    assert G.idx(e) == G_IDENTITY_INDEX

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid group element")

    orders = {g: order(g) for g in G.elements}
    nonidentity = [g for g in G.elements if g != e]
    pools = {
        degree: [g for g in nonidentity if orders[g] == degree]
        for degree in (3, 6, 12, 16, 24)}
    high = [g for g in nonidentity if orders[g] >= 6]

    def core_key(g, h):
        # Genuine group automorphisms: x->x^u, y->x^v*y^t,
        # with u nonzero modulo three and t odd modulo sixteen.
        return min(
            tuple(sorted(
                G.idx(((u*a + v*(b & 1)) % 3, t*b % 16))
                for a, b in (g, h)))
            for u in (1, 2) for v in range(3)
            for t in range(1, 16, 2))

    cores, core_keys = [], set()
    for g, h in combinations(nonidentity, 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        key = core_key(g, h)
        if key in core_keys:
            continue
        core_keys.add(key)
        data = _c16_core_data(G, g, h)
        if data is not None:
            cores.append(data)
    if not cores:
        raise ValueError("no connected matching core of girth at least eight")
    cores.sort(key=lambda data: (-data["girth"],
                                G.idx(data["g"]), G.idx(data["h"])))
    longest = max(data["girth"] for data in cores)
    preferred = [data for data in cores if data["girth"] == longest]

    def triangle(local, policy):
        first_pool = pools[(3, 6, 12, 16, 24)[policy % 5]]
        for _ in range(128):
            g = local.choice(first_pool)
            h = local.choice(high)
            if g != h and G.mul(g, h) != G.mul(h, g):
                return [e, g, h]
        raise ValueError("noncommuting trinomial policy exhausted")

    def draw_A(local, card):
        return [[triangle(local, card % 5),
                 triangle(local, (card // 5 + card + 1) % 5)]]

    def draw_B(local, card):
        core_pool = preferred if card % 4 != 3 else cores
        data = local.choice(core_pool)
        C = [[[e], [e], [e], [], []],
             [[e], [data["g"]], [data["h"]], [], []]]
        policy = card % 5
        if card < 10:
            left_pool = pools[(3, 6, 12, 16, 24)[policy]]
            right_pool = pools[(3, 6, 12, 16, 24)[(policy + 2) % 5]]
            a, b = G.pick(local), G.pick(local)
            r, s = local.choice(left_pool), local.choice(right_pool)
            C[0][3], C[1][3] = [a, G.mul(a, r)], [e]
            C[0][4], C[1][4] = [e], [b, G.mul(b, s)]
        else:
            C[0][3] = triangle(local, policy)
            C[1][4] = triangle(local, (policy + 1) % 5)
        return C, data

    def cycle_cost(cols):
        total = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                overlap = (first & second).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    banks = {"A": [], "B": []}
    component_keys = {"A": set(), "B": set()}
    for side in ("A", "B"):
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0x481225 + 104729*card
                             + 1000003*int(side == "B")))
            records, fillings = [], set()
            for trial in range(48 if side == "A" else 72):
                if side == "A":
                    grid, data = draw_A(local, card), None
                else:
                    grid, data = draw_B(local, card)
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, width = _c16_rows(G, grid, side)
                rank = gf2_rank(rows)
                if rank < len(rows) - (1 if side == "A" else 2):
                    continue
                cols = _c16_columns(rows, width)
                if not _c16_clean4(cols):
                    continue
                seed = (0 if side == "A" else
                        _c16_two_cubic_word(rows, width, data, q))
                if seed and seed.bit_count() < 8:
                    continue
                cycles = cycle_cost(cols)
                cheap = (
                    -(seed.bit_count() if seed else width),
                    cycles, -rank, trial)
                records.append(dict(
                    grid=grid, rows=rows, width=width, seed=seed,
                    rank=rank, cycles=cycles, cheap=cheap,
                    card=card, trial=trial))
            records.sort(key=lambda record: record["cheap"])
            shortlist = records[:8]
            for record in shortlist:
                record["word"] = _c16_probe(
                    record["rows"], record["width"], local, 4,
                    record["seed"])
            shortlist.sort(key=lambda record: (
                record["word"].bit_count(), -record["cycles"],
                -record["rank"]), reverse=True)

            # Every constituent eligible for publication receives exactly
            # four preliminary and twelve additional probe passes.
            for record in shortlist[:2]:
                record["word"] = _c16_probe(
                    record["rows"], record["width"], local, 12,
                    record["word"])
                record["reading"] = record["word"].bit_count()
                signature = _c16_signature(record["rows"], record["width"])
                if signature in component_keys[side]:
                    continue
                component_keys[side].add(signature)
                banks[side].append(record)
    if not banks["A"] or not banks["B"]:
        raise ValueError("no screened asymmetric constituent bank")

    # Reject the explicitly shown archived frontier code. This is not
    # a claim to know every unshown code in the evaluator's archive.
    old_A = [[[(1, 2), (1, 9), (2, 11)],
              [(2, 7), (1, 9), (0, 15)]]]
    old_B = [
        [[(0, 0), (2, 1)], [(0, 5)], [e], [e], [e]],
        [[(2, 2)], [(1, 7), (2, 14)], [e], [(0, 5)], [(1, 12)]],
    ]
    old_hx, old_hz, old_n = lifted_matrices(G, old_A, old_B)
    excluded = {_c16_signature(old_hx + old_hz, old_n)}

    plans = []
    for ai, a in enumerate(banks["A"]):
        for bi, b in enumerate(banks["B"]):
            preliminary = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                -a["cycles"] - b["cycles"],
                int(a["card"] != b["card"]))
            plans.append((preliminary, ai, bi))
    plans.sort(key=lambda item: (item[0], -item[1], -item[2]),
               reverse=True)

    def embedded(record, kind, checks, own_rows, own):
        best = 0
        for block in range(2 if kind == "X" else 5):
            if kind == "X":
                word = record["word"] << (block * 5*q)
            else:
                word, source = 0, record["word"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _c16_outside(word, own):
                continue
            for _ in range(2):
                starting, weight = word, word.bit_count()
                for i, first in enumerate(own_rows):
                    trial = starting ^ first
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                    for second in own_rows[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and _c16_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, tested = [], 0
    for preliminary, ai, bi in plans:
        a, b = banks["A"][ai], banks["B"][bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(5))
        wz = max(sum(row) + sa[0][j]
                 for row in sb for j in range(2))
        assert wx == wz == 9
        assert max(wx, wz) <= min(W_MAX, 9, 12)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 576 or not (225*n <= 1000*k < 275*n):
            continue
        signature = _c16_signature(hx + hz, n)
        if signature in excluded:
            continue
        excluded.add(signature)
        tested += 1
        if not _c16_clean4(_c16_columns(hx, n)):
            continue
        if not _c16_clean4(_c16_columns(hz, n)):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
        bx, bz = _c16_basis(hx), _c16_basis(hz)
        xword = embedded(b, "X", hz, hx, bx)
        zword = embedded(a, "Z", hx, hz, bz)
        if not xword or not zword:
            continue
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 8:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k*reading*reading/n, reading, k, preliminary)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]), signature))
        if len(results) >= count:
            break
        if tested >= max(256, 8*count):
            break
    results.sort(key=lambda record: record[0], reverse=True)
    return tuple(results)


def _r050_clear3(cols):
    """Exactly exclude all nonzero kernel words of weights one to three."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    for i, first in enumerate(cols):
        for second in cols[:i]:
            if (first ^ second) in singles:
                return False
    return True


def _r050_logical4(checks, stabilizers, width):
    """Return an explicit weight-four logical, or zero.

    Called after exclusion through three. Equal pair syndromes then
    have disjoint supports. Span membership uses the complete stabilizers.
    """
    cols = _c16_columns(checks, width)
    own = _c16_basis(stabilizers)
    seen = {}
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
                continue
            difference = old ^ word
            if difference and _c16_outside(difference, own):
                assert difference.bit_count() == 4
                assert all((row & difference).bit_count() % 2 == 0
                           for row in checks)
                return difference
    return 0


def _r050_rooted_key(hx, hz, width):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive.

    Highest-degree qubits identify sector two without using block labels.
    Root EVERY such qubit: no unverified regular-action orbit reduction.
    Unequal signatures separate isomorphic qubit/check Tanner graphs;
    equal signatures are conservatively rejected.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _c16_columns(rows, width)
    m = len(rows)

    def positions(word):
        out = []
        while word:
            bit = word & -word
            out.append(bit.bit_length() - 1)
            word ^= bit
        return out

    adjacency = [[m + j for j in positions(row)] for row in rows]
    adjacency += [positions(col) for col in cols]
    labels = [(0, len(ns)) for ns in adjacency[:m]]
    labels += [(1, len(ns)) for ns in adjacency[m:]]
    highest = max(col.bit_count() for col in cols)
    roots = [m + j for j, col in enumerate(cols)
             if col.bit_count() == highest]

    def compress(features):
        palette = {feature: i
                   for i, feature in enumerate(sorted(set(features)))}
        return [palette[feature] for feature in features]

    rooted = []
    for root in roots:
        colors = compress([
            (label, int(v == root)) for v, label in enumerate(labels)])
        history = []
        for _ in range(6):
            features = [
                (colors[v], tuple(sorted(colors[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            colors = compress(features)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    return (
        width, m,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        tuple(sorted(rooted)),
    )


def _r050_component_bank(G, side, grounded):
    """Finite relative-matching catalogue, not random gauge copies."""
    from itertools import combinations

    q, e = G.q, G.e
    number = 6 if grounded else 7

    def normalize(values):
        # Quotient known row gauges and row-exchange images at proposal
        # level. Publication inequivalence is decided on complete graphs.
        images = []
        for anchor in values:
            inverse = G.inv(anchor)
            moved = [G.mul(inverse, g) for g in values]
            images.append(tuple(sorted(G.idx(g) for g in moved)))
            images.append(tuple(sorted(G.idx(G.inv(g)) for g in moved)))
        return min(images)

    cards = set()
    other = [g for g in G.elements if g != e]
    for tail in combinations(other, number - 1):
        cards.add(normalize((e,) + tail))

    bank = []
    for card in sorted(cards):
        ratios = [G.elements[j] for j in card]
        grid = [
            [[e] for _ in ratios],
            [[g] for g in ratios],
        ]
        if grounded:
            grid[0].append([e])
            grid[1].append([])
        assert len(grid[0]) == len(grid[1]) == 7
        rows, width = _c16_rows(G, grid, side)
        rank = gf2_rank(rows)
        expected = 2 * q if grounded else 2 * q - 1
        if rank != expected:
            continue
        cols = _c16_columns(rows, width)
        if not _r050_clear3(cols):
            continue

        # Count every constituent weight-four support exactly. This
        # provides an intrinsic multiplicity tie-break, not a surrogate
        # lower bound on quantum distance.
        multiplicities = {}
        collisions = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                previous = multiplicities.get(syndrome, 0)
                collisions += previous
                multiplicities[syndrome] = previous + 1
        assert collisions % 3 == 0
        bank.append((collisions // 3, card, grid))
    bank.sort(key=lambda record: (record[0], record[1]))
    return tuple(bank)


@lru_cache(maxsize=1)
def _r050_portfolio(G, count):
    """Immutable, exact-distance-four, target-rate portfolio.

    Full matching pairs come first. Singly grounded pairs supply
    structurally different fallbacks without lowering the exact distance.
    All ranks, kernels and logical witnesses use verified scaffold lifts.
    """
    assert G.q == 10
    assert A_SHAPE == B_SHAPE == (2, 7)
    banks = {
        (side, grounded): _r050_component_bank(G, side, grounded)
        for side in ("A", "B") for grounded in (False, True)
    }
    plans = []
    for ga, gb in ((False, False), (True, False), (False, True)):
        for ai, a in enumerate(banks["A", ga]):
            for bi, b in enumerate(banks["B", gb]):
                plans.append((
                    int(ga or gb), max(a[0], b[0]), a[0] + b[0],
                    ga, gb, ai, bi))
    plans.sort()

    results, used, fillings = [], set(), set()
    cap = min(W_MAX, 9)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for _, _, _, ga, gb, ai, bi in plans:
        a, b = banks["A", ga][ai], banks["B", gb][bi]
        A, B = a[2], b[2]
        filling = (freeze(A), freeze(B))
        if filling in fillings:
            continue
        fillings.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb):
            continue
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(7))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(7))
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 530 or not (475 * n <= 1000 * k < 525 * n):
            continue
        # Do not admit lower-scoring rank outcomes as hidden fallbacks.
        if k < 255:
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if not _r050_clear3(_c16_columns(hx, n)):
            continue
        if not _r050_clear3(_c16_columns(hz, n)):
            continue
        assert css_ok(hx, hz)

        # A lower bound of four plus an actual logical of weight four
        # certifies the exact distance, independently of GPU search luck.
        witness = _r050_logical4(hz, hx, n)
        kind = "X"
        if not witness:
            witness = _r050_logical4(hx, hz, n)
            kind = "Z"
        if not witness:
            continue

        signature = _r050_rooted_key(hx, hz, n)
        if signature in used:
            continue
        used.add(signature)
        assert lifted_k(G, A, B) == k
        results.append((
            freeze(A), freeze(B), k, kind, witness,
            (ga, gb, a[1], b[1]),
            (a[0], b[0]),
            signature,
        ))
        if len(results) >= count:
            break

    # Sorting only changes slot ownership, not the selected matrices.
    results.sort(key=lambda record: (
        -record[2], max(record[6]), sum(record[6]), record[5]))
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure lookup; no resampling after exact distance/rate admission."""
    catalogue = _r050_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("invariant-distinct exact-four portfolio exhausted")
    A, B, k, kind, witness, card, counts, signature = catalogue[idx]
    return {
        "cells": {"A": A, "B": B},
        "screen_k": k,
        "certified_distance": 4,
        "logical_witness_type": kind,
        "logical_witness": witness,
        "design_card": card,
        "component_four_counts": counts,
        "exact_score": 16 * k / 530,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh cell from the immutable screened portfolio."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Weight-four rejection plus component cycle minimization excludes weight-20 mixed-sector logicals on Dic11 (2,3)x(2,3): measured slot 42 had d30s=20 with Z witnesses in both sectors.
# REFUTED: The v2 mixed-coset/component-kernel policy yields five measured readings >=24: its measured slots 8 and 9 instead had d30s=23, with both logical types entirely in sector 1.
# REFUTED: The v3 independently optimized saturated-constituent policy yields five measured readings >=25: all five measured finalists instead had d30s=24, with minimum witnesses entirely in sector 1.
# REFUTED: The C3:C16 reinforced-core v1 portfolio exceeds fitness 20.9: its measured fitness was 20.25, with all five finalists at k=144,d30s=9 and only eighteen distinct codes across sixty slots.
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
