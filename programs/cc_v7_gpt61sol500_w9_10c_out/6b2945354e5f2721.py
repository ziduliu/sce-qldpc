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
CONCEPT = ("C7 semidirect_3 C6, order 42, A=B=(2,3): faithful order-six conjugation and independently optimized saturated constituents should supply five fresh distance-24 readings with fitness above 46.5234 in cell 0.100; this is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from Dic11 to x^7=y^6=1, y*x*y^-1=x^3. Since 3 has order six modulo seven, this is a faithful Frobenius group with centre one and seven conjugacy classes. Its order 42 proves it is not a relabeling of the order-44 parent. RUN STATE records an order-42 centre-one/seven-class bed at fitness 16.2 on (1,2)x(2,5), but no order-42 (2,3)x(2,3) bed. The parent measured five distance-24 finalists; transferring that reading is a hypothesis.")
SHAPES = ("Keep equal (2,3) sides: constituents now have 126 variables and 84 checks, n=546, kb=42 and guaranteed rate 1/13 in cell 0.100. The size window permits q<=46; the fixed new group uses q=42. Two augmentation-rank-one support profiles provide at least four additional full-check dependencies, hence k>=46 for those pairings. Parent weight-24 minima occupied sector-1 Acol/Bcol blocks, especially columns one and two; no reported minimum used sector two. Actual full ranks determine acceptance.")
FILLING = ("Twenty support/quotient prescriptions retain row totals five and column totals at most four, so BOTH CSS maxima are at most nine, also below twelve. Rotation differences have order seven; active complement differences have order six. Verified constituent syndromes exactly enumerate the normal-C7 invariant kernel, followed by equal-budget information-set, basis-triple and descent searches. Full lifts enforce the target rate, CSS and exact logical-four exclusion, then compare explicit stabilizer-reduced logical witnesses. A pure immutable prefix rejects equal uncoloured full-Tanner invariants conservatively. Larger readings and novelty against the unseen archive remain evaluator-tested hypotheses.")
FAMILY_NAME = "r010_c7sd3_c6_q42_23x23_faithful_saturated_prefix_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect_3 C6 has order 42")
    # (a,b)=x^a*y^b; 3 has order six modulo seven.
    powers = [pow(3, b, 7) for b in range(6)]
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _faithful_design(rng, G, idx, excluded):
    """Optimize a complete filling outside the published invariant prefix."""
    assert G.q == 42
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

        # Every nonidentity normal-subgroup difference has order seven.
        # Complement exponents 1 and 5 have order six, never two or three.
        rotation_steps = list(range(1, 7))
        cross_exponents = (1,) if mode == 0 else (1, 5)
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, count in enumerate(counts):
                first = G.pick(rng)
                if count == 1:
                    cell = [first]
                elif count == 2:
                    relative = ((rng.randrange(7),
                                 rng.choice(cross_exponents))
                                if (i, j) in mixed else
                                (rng.choice(rotation_steps), 0))
                    cell = [first, G.mul(first, relative)]
                else:
                    rotation = (rng.choice(rotation_steps), 0)
                    cross = (rng.randrange(7),
                             rng.choice(cross_exponents))
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
        # Normal C7 has six cosets. Eliminate the eighteen coset
        # syndromes, then enumerate their COMPLETE compressed kernel.
        # Both syndromes and expanded witnesses use the verified lift.
        pivots, orbit_basis = {}, []
        for j in range(3):
            for b in range(6):
                word = syndrome = 0
                for a in range(7):
                    position = j*G.q + G.idx((a, b))
                    word |= 1 << position
                    syndrome ^= cols[position]
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    orbit_basis.append(word)

        # Every support has a monomial column, so its compressed rank
        # is at least six and this enumeration has at most 4096 words.
        assert len(orbit_basis) <= 12
        previous = word = 0
        orbit_distance = width + 1
        orbit_witness = 0
        for number in range(1, 1 << len(orbit_basis)):
            gray = number ^ (number >> 1)
            changed = (gray ^ previous).bit_length()-1
            word ^= orbit_basis[changed]
            if word and word.bit_count() < orbit_distance:
                orbit_distance = word.bit_count()
                orbit_witness = word
            previous = gray
        if orbit_witness:
            assert all((row & orbit_witness).bit_count() % 2 == 0
                       for row in rows)
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
            cheap = (int(stats[3] <= 21), -stats[3],
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
        signature = _faithful_graph_key(hx, hz, n)
        if signature in excluded:
            continue
        if low_logical(hz, hx, n) or low_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        assert n == 546
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        record["signature"] = signature
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
        "signature": winner["signature"],
    }


def _faithful_graph_key(hx, hz, n):
    """Conservative invariant of the complete uncoloured Tanner graph.

    Different keys certify different check/qubit incidence graphs.
    Equal keys are rejected; they are not claimed to prove isomorphism.
    X/Z types, sectors, group coordinates and protograph labels are absent.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0]*m, [0]*m
    intersections = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                relation = one if overlap == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i
                intersections[i][j] = overlap
                intersections[j][i] = overlap

    features = []
    for i in range(m):
        profile = Counter()
        for j in range(m):
            if i == j:
                continue
            profile[(
                weights[j], intersections[i].get(j, 0),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count(),
            )] += 1
        features.append((weights[i], tuple(sorted(profile.items()))))

    columns = [0]*n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= 1 << r
            row ^= bit
    incidence = []
    for column in columns:
        neighbors = []
        while column:
            bit = column & -column
            neighbors.append(bit.bit_length()-1)
            column ^= bit
        incidence.append(neighbors)

    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]
    for _ in range(3):
        variable_features = [
            tuple(sorted(colors[r] for r in neighbors))
            for neighbors in incidence]
        vp = {f: i for i, f in
              enumerate(sorted(set(variable_features)))}
        neighborhoods = [[] for _ in rows]
        for neighbors, feature in zip(incidence, variable_features):
            for r in neighbors:
                neighborhoods[r].append(vp[feature])
        refined = [
            (colors[r], tuple(sorted(neighborhoods[r])))
            for r in range(m)]
        history.append((
            tuple(sorted(Counter(variable_features).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: i for i, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]

    signature = (
        n, m,
        tuple(sorted(Counter(len(v) for v in incidence).items())),
        tuple(history))
    return sha256(repr(signature).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=None)
def _faithful_prefix(G, count):
    """Pure deterministic prefix; cached published cells are immutable."""
    if count == 0:
        return ()
    previous = _faithful_prefix(G, count-1)
    excluded = {entry[-1] for entry in previous}
    idx = count-1
    winner = None
    for restart in range(4):
        local = random.Random(
            BASE_SEED*10007 + 0x426323
            + 104729*idx + 1000003*restart)
        try:
            winner = _faithful_design(
                local, G, idx + 20*restart, excluded)
            break
        except ValueError:
            continue
    if winner is None:
        raise ValueError("fresh faithful saturated prescription exhausted")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    entry = (
        freeze(winner["cells"]["A"]),
        freeze(winner["cells"]["B"]),
        winner["k"], winner["component_probe"],
        winner["quantum_witness_probe"], winner["signature"])
    assert entry[-1] not in excluded
    return previous + (entry,)


def NEW_CTX(rng, G, idx):
    """Publish a complete invariant-distinct optimized prescription."""
    A, B, k, components, reading, signature = (
        _faithful_prefix(G, idx+1)[-1])
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    wx = max(sum(row) + sum(sb[i][j] for i in range(2))
             for row in sa for j in range(3))
    wz = max(sum(row) + sum(sa[i][j] for i in range(2))
             for row in sb for j in range(3))
    assert max(wx, wz) <= min(9, 12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert 0 < k and 8*k < 546
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "k": k, "component_probe": components,
        "quantum_witness_probe": reading,
        "tanner_signature": signature.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh list without mutating the immutable prefix."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Weight-four rejection plus component cycle minimization excludes weight-20 mixed-sector logicals on Dic11 (2,3)x(2,3): measured slot 42 had d30s=20 with Z witnesses in both sectors.
# REFUTED: The v2 mixed-coset/component-kernel policy yields five measured readings >=24: its measured slots 8 and 9 instead had d30s=23, with both logical types entirely in sector 1.
# REFUTED: Independently optimized saturated Dic11 constituents plus stabilizer-pair descent yield five measured readings >=25: all five current measured finalists had d30s=24, with minimum witnesses entirely in sector one.
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
