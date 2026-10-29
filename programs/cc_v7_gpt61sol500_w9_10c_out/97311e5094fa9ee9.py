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
CONCEPT = ("Dic73, order 292, A=B=(1,1): sparse rotation polynomials with prescribed shared F512 roots, optimized by equally refined verified quantum witnesses, should supply five fresh low-rate codes exceeding fitness 46.5234 in cell 0.100; this is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Retain the dicyclic construction rule x^(2m)=1, y^2=x^m, y*x*y^-1=x^-1, changing m from 11 to 73. The parent Dic11 bed scored 46.5234 but all five measured minima remained at 24. Order 292 gives n=584 and a normal C146 subgroup; order-73 roots over F512 provide explicit sparse shared-root prescriptions. The new order is necessary because square shapes would give n=88 at q=44.")
SHAPES = ("Change both sides to (1,1). Each constituent has 292 variables and 292 checks, the two LP sectors each have 292 qubits, n=584, and the rank-count bound is zero. Require actual 0<k<73, strictly inside open-ended cell 0.100. The size window permits q<=299; the selected dicyclic order is 292. Parent minima occupied sector-1 Acol/Bcol blocks, predominantly columns 1 and 2; the square cell removes that three-column component bottleneck.")
FILLING = ("Optimize weight-four/five rotation polynomials sharing an order-73 F512 root, with reciprocal-root prescriptions when available and distinct central parity patterns. Both CSS row sums are at most nine. Polynomial gcds prioritize proposals only: verified complete lifts determine k and expose both cyclic halves, including the inverse-B half. Every eligible finalist receives equal quantum kernel-basis and stabilizer-descent budgets. Actual nontrivial witnesses, exact logical-four rejection, CSS, target-band ranks and different uncoloured Tanner invariants gate publication. Rotation-supported graphs may be disconnected; neither larger readings nor novelty against the unseen archive is certified.")
FAMILY_NAME = "r010_dic73_11x11_sharedroot_bicycle_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 292

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic73 has order 292")
    # Same dicyclic rule; only its parameter m changes.
    m, rotation = 73, 146
    els = [(a, b) for b in range(2) for a in range(rotation)]

    def red(g):
        a, b = g
        return ((a + m * (b // 2)) % rotation, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + m * b * d) % rotation,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % rotation, 0) if b == 0 else ((a + m) % rotation, 1)

    def idx(g):
        a, b = red(g)
        return rotation * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 1)               # (ra, ca)
B_SHAPE = (1, 1)               # (rb, cb)


def _unused_dic11_saturated_reference(rng, G, idx):
    """Inactive historical rule; never called by the square portfolio."""
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


@lru_cache(maxsize=1)
def _square_root_banks():
    """Finite sparse-polynomial prescriptions, not completed code draws."""
    from itertools import combinations

    def mul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 512:
                a ^= 0x211                 # x^9+x^4+1
        return out

    def power(a, e):
        out = 1
        while e:
            if e & 1:
                out = mul(out, a)
            a = mul(a, a)
            e >>= 1
        return out

    root = power(2, 7)
    values = [power(root, a) for a in range(73)]
    assert len(set(values)) == 73 and power(root, 73) == 1

    def signature(a, reciprocal):
        value = values[a]
        return value | (values[-a % 73] << 9) if reciprocal else value

    def canonical(support):
        # Independent translations are polynomial-unit gauges.
        return min(tuple(sorted((a - anchor) % 73 for a in support))
                   for anchor in support)

    def bank(weight, reciprocal):
        pairs = {}
        for a, b in combinations(range(1, 73), 2):
            value = signature(a, reciprocal) ^ signature(b, reciprocal)
            pairs.setdefault(value, []).append((a, b))
        target = signature(0, reciprocal)
        found = set()
        if weight == 4:
            # Fix exponent zero and join one singleton to one pair.
            for a in range(1, 73):
                wanted = target ^ signature(a, reciprocal)
                for b, c in pairs.get(wanted, ()):
                    if a not in (b, c):
                        found.add(canonical((0, a, b, c)))
        else:
            # Fix zero and join two disjoint pairs.
            for value, firsts in pairs.items():
                seconds = pairs.get(value ^ target, ())
                for a, b in firsts:
                    for c, d in seconds:
                        if len({a, b, c, d}) == 4:
                            found.add(canonical((0, a, b, c, d)))
        return tuple(sorted(found))

    four = bank(4, False)
    five = bank(5, False)
    reciprocal_five = bank(5, True)
    if not four or not five:
        raise ValueError("order-73 sparse shared-root banks absent")
    return four, five, reciprocal_five


def _square_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _square_span(rows):
    span = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = span.get(p)
            if old is None:
                span[p] = row
                break
            row ^= old
    return span


def _square_nontrivial(word, span):
    while word:
        p = word.bit_length() - 1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _square_clean_four(checks, own, width):
    """Exactly exclude nontrivial logical supports of weights <=4."""
    cols = _square_columns(checks, width)
    span = _square_span(own)
    pivots = sorted(span, reverse=True)

    def residue(word):
        for p in pivots:
            if (word >> p) & 1:
                word ^= span[p]
        return word

    residues = [residue(1 << j) for j in range(width)]
    seen = {0: 0}
    for i, s in enumerate(cols):
        r = residues[i]
        old = seen.get(s)
        if old is not None:
            if old != r:
                return False
        else:
            seen[s] = r
    for i, s in enumerate(cols):
        for j in range(i):
            value, r = s ^ cols[j], residues[i] ^ residues[j]
            old = seen.get(value)
            if old is not None:
                if old != r:
                    return False
            else:
                seen[value] = r
    return True


def _square_probe(local, checks, own, width, passes, seed=0):
    """Explicit quantum witnesses; a larger reading is not a lower bound."""
    cols = _square_columns(checks, width)
    span = _square_span(own)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight and _square_nontrivial(word, span):
            best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
        pivots, basis = {}, []
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
                basis.append(word)
                inspect(word)

        # Elementary exchanges preserve the complete kernel span.
        local.shuffle(basis)
        for i, word in enumerate(basis):
            w = word.bit_count()
            for j, other in enumerate(basis):
                if i == j:
                    continue
                trial = word ^ other
                t = trial.bit_count()
                if 0 < t < w:
                    word, w = trial, t
            basis[i] = word
            inspect(word)
        for i, first in enumerate(basis):
            for second in basis[:i]:
                inspect(first ^ second)

        ranked = sorted(basis, key=int.bit_count)
        # Retain logical-bearing words explicitly: the shortest kernel
        # basis vectors can all be stabilizers.
        logical = [w for w in ranked if _square_nontrivial(w, span)]
        short = list(dict.fromkeys(ranked[:18] + logical[:12]))
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)

    if not best:
        return None
    # Equal stabilizer-pair descent budget for every published witness.
    for _ in range(2):
        starting = best
        for i, first in enumerate(own):
            inspect(starting ^ first)
            for second in own[:i]:
                inspect(starting ^ first ^ second)
        if best == starting:
            break
    assert _square_nontrivial(best, span)
    assert all((row & best).bit_count() % 2 == 0 for row in checks)
    return weight, best


def _square_halves(G, hx, hz):
    """Extract both cyclic halves from the VERIFIED complete matrices."""
    q, length = G.q, G.q // 2
    mask = (1 << length) - 1
    result = []
    for b in range(2):
        def compact(row):
            out = ((row >> (b * length)) & mask)
            out |= (((row >> (q + b * length)) & mask) << length)
            assert out.bit_count() == row.bit_count()
            return out
        xx = [compact(row) for row in hx[b * length:(b + 1) * length]]
        zz = [compact(row) for row in hz[b * length:(b + 1) * length]]
        k = 2 * length - gf2_rank(xx) - gf2_rank(zz)
        result.append((xx, zz, k))
    return result


def _square_signature(hx, hz, n):
    """Conservative full Tanner invariant, without X/Z colouring."""
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    m = len(rows)
    degrees = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, first in enumerate(rows):
        for j in range(i):
            shared = (first & rows[j]).bit_count()
            if shared:
                overlaps[i, j] = shared
                relation = one if shared == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlaps.get((i, j), 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count(),
            )
            profiles[i][(degrees[j],) + feature] += 1
            profiles[j][(degrees[i],) + feature] += 1
    labels = [(degrees[i], tuple(sorted(profiles[i].items())))
              for i in range(m)]
    palette = {label: i for i, label in enumerate(sorted(set(labels)))}
    colors = [palette[label] for label in labels]
    neighborhoods = Counter()
    for column in _square_columns(rows, n):
        values = []
        while column:
            bit = column & -column
            values.append(colors[bit.bit_length() - 1])
            column ^= bit
        neighborhoods[tuple(sorted(values))] += 1
    value = (n, tuple(sorted(Counter(labels).items())),
             tuple(sorted(neighborhoods.items())))
    return sha256(repr(value).encode("ascii")).digest()


def _square_design(G, idx, excluded, restart):
    """Optimize one structural card with actual quantum witness scores."""
    local = random.Random(
        BASE_SEED * 10007 + 0x734409
        + 65537 * idx + 1000003 * restart)
    four, five, reciprocal = _square_root_banks()
    card = idx % 20
    kind = card % 4
    abank = five if kind == 2 else four
    bbank = (four if kind in (0, 2) else
             reciprocal if kind == 3 and reciprocal else five)
    length = 146
    modulus = (1 << length) | 1

    def polynomial(exponents):
        return sum(1 << a for a in exponents)

    def remainder(a, b):
        while a and a.bit_length() >= b.bit_length():
            a ^= b << (a.bit_length() - b.bit_length())
        return a

    def gcd(a, b):
        while b:
            a, b = b, remainder(a, b)
        return a

    def draw(support, policy):
        if policy == 0:
            parity = [a & 1 for a in support]       # even exponents
        elif policy == 1:
            parity = [1 - (a & 1) for a in support]
        else:
            parity = [local.randrange(2) for _ in support]
            # A genuine mixed central lift, not an all-even gauge.
            actual = [(a + 73 * p) & 1
                      for a, p in zip(support, parity)]
            if len(set(actual)) == 1:
                parity[-1] ^= 1
        return tuple(sorted(a + 73 * p
                            for a, p in zip(support, parity)))

    proposals = {}
    for trial in range(160):
        # Indexed support/root and central-lift prescriptions define
        # cards. Trials optimize them; completed draws are not emitted.
        ai = (card // 4 + trial // 4) % len(abank)
        bi = (37 * card + (11 + card % 5) * trial) % len(bbank)
        policy = (trial + card // 4) % 5
        ae = draw(abank[ai], 0 if policy == 0 else 2)
        be = draw(bbank[bi], 0 if policy in (0, 1) else 2)
        key = (ae, be)
        if key in proposals:
            continue
        ap, bp = polynomial(ae), polynomial(be)
        reverse_b = polynomial(tuple((-a) % length for a in be))
        # The nonabelian rotation-supported lift has BOTH the B and
        # inverse-B cyclic halves. These gcds are proposal priorities,
        # never substitutes for the complete verified ranks below.
        g0 = gcd(gcd(ap, bp), modulus).bit_length() - 1
        g1 = gcd(gcd(ap, reverse_b), modulus).bit_length() - 1
        prediction = 2 * (g0 + g1)
        priority = (int(0 < prediction < 73), prediction, -trial)
        proposals[key] = (priority, policy, ae, be)
    ordered = sorted(proposals.values(), reverse=True)
    # Reserve candidates outside the largest predicted-rank class:
    # disconnected repeated-root lifts must not monopolize the probes.
    chosen = ordered[:6]
    reserve = ordered[6:]
    local.shuffle(reserve)
    chosen += reserve[:6]

    candidates = []
    for _, policy, ae, be in chosen:
        A = [[[(a, 0) for a in ae]]]
        B = [[[(a, 0) for a in be]]]
        sa, sb = [[len(ae)]], [[len(be)]]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= min(W_MAX, 9)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (0 < k and 8 * k < n):
            continue
        halves = _square_halves(G, hx, hz)
        assert sum(part[2] for part in halves) == k
        probes = []
        for xx, zz, hk in halves:
            if hk == 0:
                probes.append(None)
                continue
            dx = _square_probe(local, zz, xx, 292, 2)
            dz = _square_probe(local, xx, zz, 292, 2)
            if dx is None or dz is None:
                break
            probes.append((dx, dz))
        if len(probes) != 2:
            continue
        reading = min(p[t][0] for p in probes if p is not None
                      for t in range(2))
        candidates.append(dict(
            A=A, B=B, hx=hx, hz=hz, n=n, k=k, halves=halves,
            probes=probes, policy=policy,
            quality=(k * reading * reading / float(n), reading, k)))
    candidates.sort(key=lambda r: r["quality"], reverse=True)

    finalists = candidates[:3]
    for record in finalists:
        probes = []
        for (xx, zz, hk), previous in zip(
                record["halves"], record["probes"]):
            if hk == 0:
                probes.append(None)
                continue
            dx = _square_probe(local, zz, xx, 292, 12, previous[0][1])
            dz = _square_probe(local, xx, zz, 292, 12, previous[1][1])
            probes.append((dx, dz))
        record["probes"] = probes
        reading = min(p[t][0] for p in probes if p is not None
                      for t in range(2))
        record["quality"] = (
            record["k"] * reading * reading / float(record["n"]),
            reading, record["k"])
    finalists.sort(key=lambda r: r["quality"], reverse=True)

    for record in finalists:
        legal = True
        for xx, zz, hk in record["halves"]:
            if hk and (not _square_clean_four(zz, xx, 292)
                       or not _square_clean_four(xx, zz, 292)):
                legal = False
                break
        if not legal:
            continue
        key = _square_signature(record["hx"], record["hz"], record["n"])
        if key in excluded:
            continue
        assert css_ok(record["hx"], record["hz"])
        assert max(row.bit_count()
                   for row in record["hx"] + record["hz"]) <= min(W_MAX, 9)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        return (
            freeze(record["A"]), freeze(record["B"]), key,
            record["k"], record["quality"][1],
            tuple(part[2] for part in record["halves"]),
            tuple(None if p is None else (p[0][0], p[1][0])
                  for p in record["probes"]),
        )
    raise ValueError("square shared-root card exhausted its verified finalists")


@lru_cache(maxsize=None)
def _square_prefix(G, count):
    """Immutable deterministic prefix; memoization changes only cost."""
    if count == 0:
        return ()
    previous = _square_prefix(G, count - 1)
    excluded = {entry[2] for entry in previous}
    for restart in range(3):
        try:
            entry = _square_design(G, count - 1, excluded, restart)
            return previous + (entry,)
        except ValueError:
            continue
    raise ValueError("invariant-distinct square portfolio exhausted")


def NEW_CTX(rng, G, idx):
    """Pure lookup of the selected new-shape design."""
    assert G.q == 292
    A, B, key, k, reading, half_k, probes = _square_prefix(G, idx + 1)[-1]
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "screen_k": k, "half_dimensions": half_k,
        "quantum_witness_upper_bound": reading,
        "half_witness_upper_bounds": probes,
        "tanner_signature": key.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the entire selected square cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Weight-four rejection plus component cycle minimization excludes weight-20 mixed-sector logicals on Dic11 (2,3)x(2,3): measured slot 42 had d30s=20 with Z witnesses in both sectors.
# REFUTED: The v2 mixed-coset/component-kernel policy yields five measured readings >=24: its measured slots 8 and 9 instead had d30s=23, with both logical types entirely in sector 1.
# REFUTED: Independently optimized saturated Dic11 constituents plus stabilizer-pair descent yield five measured readings >=25: all five current measured codes instead had d30s=24, with minimum witnesses entirely in sector 1.
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
