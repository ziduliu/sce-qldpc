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
CONCEPT = ("C7 semidirect C15 with action 2, order 105, A=(1,2), B=(1,2): independently selected noncommuting trinomials, normal-C7 witness screening and complete-CSS witness reduction should supply five measured readings >=14 in cell 0.200; measurements can refute this.")
GROUP = ("Jump from A4 x C4 to C7:C15(s=2). The action is valid because 2^3=1 modulo 7, and the centre is C5 rather than the parent's C4. This order-105 bed is absent from BEDS. The shown order-117 single-row experiment reached fitness 61.1, motivating a related long-complement experiment without transferring its distance readings or copying its cells.")
SHAPES = ("Use equal A=(1,2), B=(1,2): q=105, n=525, kb=105 and guaranteed rate 0.200. Each constituent has 210 variables and 105 checks; the size window allows q<=119. Parent weight-eight minima occupied both Acols across Bcols, particularly Bcols 2-4. Remove that three-matching B core and admit only verified full rates in [0.175,0.225).")
FILLING = ("All four cells contain three distinct elements, giving BOTH CSS check sums 6+3=9, below nine and twelve. Twenty cards pair six unordered constituent prescriptions from noncommuting spectra (7,15,15), (15,15,15), (15,15,35). Verified zero-side lifts expose the exact constituents. Short-dependency exclusion, Tanner-cycle penalties and actual one/two normal-C7-coset witnesses shortlist proposals. Equally refined kernel probes retain explicit witnesses. Full lifts enforce rate, CSS and logical-four exclusion; nontrivial embeddings undergo stabilizer descent and mixed-sector probes. Equal intrinsic Tanner signatures are conservatively rejected. All larger readings remain witness upper bounds, not distance certificates; unseen-archive novelty remains a hypothesis.")
FAMILY_NAME = "r020_c7sd_c15_s2_q105_12x12_orbit_witness_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 105

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7:C15(s=2) has order 105")
    # x^7=y^15=1, y*x*y^-1=x^2; 2^3=1 modulo 7.
    # Elements are x^a y^b. The centre is <y^3>, of order five.
    powers = tuple(pow(2, b, 7) for b in range(15))
    els = [(a, b) for b in range(15) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 15)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 7, (b + d) % 15)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 15
        return ((-powers[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _triangle_design(rng, G, idx, excluded):
    """Select one structural card using verified constituent/CSS witnesses."""
    assert G.q == 105
    assert A_SHAPE == B_SHAPE == (1, 2)
    counts = ((3, 3),)
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 9 <= min(W_MAX, 9, 12)
    q, width = G.q, 2 * G.q

    spectra = ((7, 15, 15), (15, 15, 15), (15, 15, 35))
    policies = tuple((a, b) for a in range(3) for b in range(a, 3))
    plans = tuple((a, b) for a in range(6) for b in range(a, 6))[:20]
    card = idx % 20
    pa, pb = plans[card]

    def order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid group element order")

    orders = {g: order(g) for g in G.elements}
    banks = [[] for _ in spectra]
    for r in G.elements:
        if r == G.e:
            continue
        ri = G.inv(r)
        for s in G.elements:
            if s == G.e or s == r or G.mul(r, s) == G.mul(s, r):
                continue
            spectrum = tuple(sorted(
                (orders[r], orders[s], orders[G.mul(ri, s)])))
            if spectrum in spectra:
                banks[spectra.index(spectrum)].append((r, s))
    if any(not bank for bank in banks):
        raise ValueError("required noncommuting spectrum absent")

    cosets = tuple(
        tuple(G.idx(g) for g in G.elements if g[1] == b)
        for b in range(15))
    assert all(len(coset) == 7 for coset in cosets)

    def columns(rows, n):
        result = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    pivots[p] = word
                    break
                word ^= pivots[p]
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    def short_word(rows, n, stabilizers=None):
        """Exact kernel/logical existence through weight four."""
        own = span(stabilizers) if stabilizers is not None else {}
        cols = columns(rows, n)
        representatives = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            if syndrome in representatives:
                if outside(word ^ representatives[syndrome], own):
                    return True
            else:
                representatives[syndrome] = word
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                if syndrome in representatives:
                    if outside(word ^ representatives[syndrome], own):
                        return True
                else:
                    representatives[syndrome] = word
        return False

    def component(grid, side):
        # A zero auxiliary cell isolates the evaluator's exact action.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        assert len(rows) == q
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def component_features(rows):
        cols = columns(rows, width)
        if short_word(rows, width):
            return None
        c4 = 0
        adjacency = [0] * q
        for i, first in enumerate(cols):
            for second in cols[:i]:
                overlap = (first & second).bit_count()
                c4 += overlap * (overlap - 1) // 2
            bits = first
            while bits:
                bit = bits & -bits
                adjacency[bit.bit_length() - 1] |= first ^ bit
                bits ^= bit
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            bits = neighbors >> (i + 1)
            while bits:
                bit = bits & -bits
                j = bit.bit_length() + i
                triangles += (
                    (neighbors & adjacency[j]) >> (j + 1)).bit_count()
                bits ^= bit
        # Each cubic variable contributes one internal projected triangle.
        c6 = max(0, triangles - width)

        # Actual complete-normal-C7-coset witnesses, not an exhaustive
        # search of the 30-dimensional coset-uniform support space.
        best_word = (1 << width) - 1
        best = width
        representatives = {0: 0}
        for block in range(2):
            for coset in cosets:
                word = syndrome = 0
                for h in coset:
                    position = block * q + h
                    word |= 1 << position
                    syndrome ^= cols[position]
                if syndrome in representatives:
                    candidate = word ^ representatives[syndrome]
                    if candidate and candidate.bit_count() < best:
                        best, best_word = candidate.bit_count(), candidate
                else:
                    representatives[syndrome] = word
        assert all((row & best_word).bit_count() % 2 == 0 for row in rows)
        return c4, c6, best, best_word

    def probe(rows, n, passes, seed=0, stabilizers=None):
        """Return only actual kernel or nontrivial logical witnesses."""
        cols = columns(rows, n)
        own = span(stabilizers) if stabilizers is not None else None
        best_word = seed
        best = seed.bit_count() if seed else n + 1
        if seed:
            assert all((row & seed).bit_count() % 2 == 0 for row in rows)
            assert own is None or outside(seed, own)

        def record(word):
            nonlocal best, best_word
            weight = word.bit_count()
            if not word or weight >= best:
                return
            if own is not None and not outside(word, own):
                return
            best, best_word = weight, word

        permutation = list(range(n))
        for _ in range(passes):
            rng.shuffle(permutation)
            pivots, kernel = {}, []
            for j in permutation:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = (syndrome, word)
                        break
                    previous = pivots[p]
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    kernel.append(word)

            if own is None:
                rng.shuffle(kernel)
                for i, word in enumerate(kernel):
                    weight = word.bit_count()
                    for j, other in enumerate(kernel):
                        if i == j:
                            continue
                        candidate = word ^ other
                        w = candidate.bit_count()
                        if 0 < w < weight:
                            word, weight = candidate, w
                    kernel[i] = word
            for word in kernel:
                record(word)
            for i, word in enumerate(kernel):
                for other in kernel[:i]:
                    record(word ^ other)
            short = sorted(kernel, key=int.bit_count)[:32]
            if own is None:
                for i, first in enumerate(short):
                    for j in range(i):
                        pair = first ^ short[j]
                        for h in range(j):
                            record(pair ^ short[h])
            else:
                logicals = sorted(
                    (word for word in kernel if outside(word, own)),
                    key=int.bit_count)[:16]
                for word in logicals:
                    for other in short:
                        record(word ^ other)

            for _ in range(2):
                if not best_word:
                    break
                starting = best_word
                for i, first in enumerate(short):
                    record(starting ^ first)
                    for second in short[:i]:
                        record(starting ^ first ^ second)
                if starting == best_word:
                    break
        if best_word:
            assert all((row & best_word).bit_count() % 2 == 0 for row in rows)
            assert own is None or outside(best_word, own)
        return best, best_word

    def finalists(side, policy):
        pool = []
        prescription = policies[policy]
        cycle_bias = 12 + 8 * ((card + int(side == "B")) % 4)
        for trial in range(64):
            cells = []
            for kind in prescription:
                r, s = rng.choice(banks[kind])
                anchor = G.pick(rng)
                cell = [anchor, G.mul(anchor, r), G.mul(anchor, s)]
                assert len(set(cell)) == 3
                cells.append(cell)
            grid = [cells]
            rows = component(grid, side)
            features = component_features(rows)
            if features is None:
                continue
            c4, c6, orbit_reading, seed = features
            cheap = (
                int(orbit_reading <= 14),
                cycle_bias * c4 + c6, -orbit_reading, trial)
            pool.append(dict(
                grid=grid, rows=rows, features=features,
                cheap=cheap, witness=seed))
        pool.sort(key=lambda record: record["cheap"])
        shortlist = pool[:10]
        for record in shortlist:
            record["reading"], record["witness"] = probe(
                record["rows"], width, 4, record["witness"])
        shortlist.sort(key=lambda record: (
            record["reading"], -record["cheap"][1]), reverse=True)
        result = shortlist[:4]
        # Carry witnesses forward: every publishable constituent gets
        # the same total probe budget and a nonincreasing reading.
        for record in result:
            record["reading"], record["witness"] = probe(
                record["rows"], width, 12, record["witness"])
        result.sort(key=lambda record: (
            record["reading"], -record["cheap"][1]), reverse=True)
        return result

    def logical_witnesses(a, b, hx, hz, n):
        """Embed and reduce verified nontrivial component witnesses."""
        witnesses = {}
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = span(stabilizers)
            best, best_word = n + 1, 0
            for block in range(2):
                if kind == "X":
                    word = b["witness"] << (block * width)
                else:
                    source, word = a["witness"], 0
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 2 + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not outside(word, own):
                    continue
                weight = word.bit_count()
                for _ in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        candidate = starting ^ first
                        if candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                        for second in stabilizers[:i]:
                            candidate = starting ^ first ^ second
                            if candidate.bit_count() < weight:
                                word, weight = candidate, candidate.bit_count()
                    if word == starting:
                        break
                assert word and outside(word, own)
                if weight < best:
                    best, best_word = weight, word
            # Search the complete matrix too, including mixed-sector
            # logicals. An absent witness is not treated as high distance.
            best, best_word = probe(
                checks, n, 4, best_word, stabilizers)
            witnesses[kind] = (best, best_word)
        return witnesses

    winner, winner_quality = None, None
    for batch in range(2):
        aa, bb = finalists("A", pa), finalists("B", pb)
        pending = sorted(
            ((min(a["reading"], b["reading"]),
              a["reading"] + b["reading"],
              -a["cheap"][1] - b["cheap"][1], ai, bi)
             for ai, a in enumerate(aa) for bi, b in enumerate(bb)),
            reverse=True)
        examined = 0
        for _, _, _, ai, bi in pending:
            a, b = aa[ai], bb[bi]
            hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
            assert n == 525
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 7 * n <= 40 * k < 9 * n):
                continue
            if short_word(hz, n, hx) or short_word(hx, n, hz):
                continue
            signature = _quotient_signature(hx, hz)
            if signature in excluded:
                continue
            witnesses = logical_witnesses(a, b, hx, hz, n)
            if not all(word for reading, word in witnesses.values()):
                continue
            reading = min(value[0] for value in witnesses.values())
            quality = (
                k * reading * reading / n, reading, k,
                -a["cheap"][1] - b["cheap"][1])
            examined += 1
            if winner_quality is None or quality > winner_quality:
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
                winner_quality = quality
                winner = {
                    "idx": idx,
                    "cells": {"A": a["grid"], "B": b["grid"]},
                    "signature": signature,
                    "screen": {
                        "k": k, "card": card,
                        "constituent_policies": (policies[pa], policies[pb]),
                        "component_witness_readings": (
                            a["reading"], b["reading"]),
                        "normal_coset_pair_readings": (
                            a["features"][2], b["features"][2]),
                        "quantum_witness_readings": (
                            witnesses["X"][0], witnesses["Z"][0]),
                        "selection_reading": reading,
                        "below14_fallback": reading < 14,
                    },
                }
            if examined >= 4:
                break
        if winner is not None and winner_quality[1] >= 14:
            break
    if winner is None:
        raise ValueError("No invariant-novel single-row target-rate pair")
    return winner


def _quotient_design(rng, G, idx, excluded):
    """Uncalled historical order-48 optimizer; never publishes a candidate."""
    card = idx % 20
    amodel, bmodel = card % 5, card // 5
    sa = ((3, 3),)
    sb = ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1))
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9
    assert _grid_weights(sa, sb) <= W_MAX

    def element_order(g):
        power = G.e
        for order in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return order
        raise ValueError("unexpected element order")

    orders = {G.idx(g): element_order(g) for g in G.elements}
    by_order = {order: [g for g in G.elements
                        if orders[G.idx(g)] == order]
                for order in (2, 3, 4, 6, 12)}
    triangles = {}
    for r in G.elements:
        for s in G.elements:
            if r == G.e or s == G.e or r == s:
                continue
            if G.mul(r, s) == G.mul(s, r):
                continue
            spectrum = tuple(sorted((
                orders[G.idx(r)], orders[G.idx(s)],
                orders[G.idx(G.mul(G.inv(r), s))])))
            triangles.setdefault(spectrum, []).append((r, s))
    available = sorted(triangles)
    assert available
    reinforcement = ((12, 12), (12, 6), (12, 4), (6, 6))[bmodel]
    assert all(by_order[order] for order in reinforcement)

    # Complete unordered ratio spectra distinguish the A prescriptions.
    # Do not count anchor changes or automorphism images as new cards.
    wanted = ((3, 12, 12), (6, 12, 12), (4, 6, 12),
              (3, 6, 6), (6, 6, 6))
    spectrum_bank = [s for s in wanted if s in triangles]
    if len(spectrum_bank) < 3:
        spectrum_bank = sorted(
            (s for s in available if min(s) >= 3),
            key=lambda s: (max(s), min(s), sum(s)), reverse=True)
    assert len(spectrum_bank) >= 3
    spectrum_pairs = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
    preferred_pair = spectrum_pairs[amodel]

    def draw_A(trial):
        cells = []
        for j in range(2):
            spectrum = spectrum_bank[preferred_pair[j]]
            # A bounded broad tail supplies genuinely different candidates
            # if a prescribed spectrum pair fails the mandatory screens.
            if trial >= 72:
                spectrum = rng.choice(spectrum_bank)
            r, s = rng.choice(triangles[spectrum])
            anchor = G.pick(rng)
            cell = [anchor, G.mul(anchor, r), G.mul(anchor, s)]
            assert len(set(cell)) == 3
            cells.append(cell)
        return [cells]

    def draw_B(core):
        B = [[[G.pick(rng)] for _ in range(5)] for _ in range(2)]
        for i, j, order in ((0, 0, reinforcement[0]),
                            (1, 1, reinforcement[1])):
            anchor = G.pick(rng)
            B[i][j] = [anchor, G.mul(anchor, rng.choice(by_order[order]))]
        for j in range(3):
            B[0][j + 2] = list(core[0][j])
            B[1][j + 2] = list(core[1][j])
        return B

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
                overlap = (s & t).bit_count()
                c4 += overlap * (overlap - 1) // 2

        # Project onto check nodes. With no 4-cycles, subtracting
        # within-column triangles leaves exactly the Tanner 6-cycles.
        adj = [0] * len(rows)
        internal = 0
        for s in cols:
            degree = s.bit_count()
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
                triangles += ((neighbors & adj[v]) >> (v + 1)).bit_count()
                bits ^= bit
        return bad, c4, max(0, triangles - internal)

    def no_kernel6(rows, width):
        """Exact absence test for constituent kernel words of weight <=6.

        All such supports split into two supports of size <=3.
        Different supports with equal syndromes expose a nonzero word.
        A False result is only a failed screen, not a minimum distance.
        """
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return False
        pairs = set()
        for i, s in enumerate(cols):
            for j in range(i):
                syndrome = s ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        triples = set()
        for i, s in enumerate(cols):
            for j in range(i):
                pair = s ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if (syndrome == 0 or syndrome in singles
                            or syndrome in pairs or syndrome in triples):
                        return False
                    triples.add(syndrome)
        return True

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
    unit = [[[G.e]]]

    def component_rows(side, grid):
        # Unit-side extraction uses precisely the evaluator's actions.
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
        return [row & mask for row in rows]

    def graph_data(core):
        """Graph and distances extracted from a verified incidence lift."""
        rows = component_rows("B", core)
        adj = [[] for _ in rows]
        for syn in columns(rows, 3 * q):
            vertices = []
            while syn:
                bit = syn & -syn
                vertices.append(bit.bit_length() - 1)
                syn ^= bit
            assert len(vertices) == 2
            a, b = vertices
            adj[a].append(b)
            adj[b].append(a)

        dist, parent, ways = [-1] * (2 * q), [-1] * (2 * q), [0] * (2 * q)
        dist[0], ways[0] = 0, 1
        queue, girth = [0], 2 * q + 1
        for a in queue:
            for b in adj[a]:
                if dist[b] < 0:
                    dist[b], parent[b] = dist[a] + 1, a
                    ways[b] = ways[a]
                    queue.append(b)
                else:
                    if dist[b] == dist[a] + 1:
                        ways[b] += ways[a]
                    if parent[a] != b and parent[b] != a:
                        girth = min(girth, dist[a] + dist[b] + 1)
        if len(queue) != 2 * q:
            return None
        # Every row-0 check is in the same group-action orbit.
        # Thus a shortest cycle can be translated through check zero.
        load = sum(w * (w - 1) // 2
                   for d, w in zip(dist, ways) if d == girth // 2)
        return (girth, load), adj

    # An order-three relative matching already supplies a six-edge
    # kernel cycle. Admit order twelve and exclude that mechanism.
    reps = []
    for order in (4, 6, 12):
        for g in by_order[order]:
            key = min(G.idx(G.mul(G.mul(h, g), G.inv(h)))
                      for h in G.elements)
            if G.idx(g) == key:
                reps.append(g)
    cores = []
    for u in reps:
        for v in G.elements:
            if orders[G.idx(v)] not in (4, 6, 12):
                continue
            if orders[G.idx(G.mul(G.inv(u), v))] not in (4, 6, 12):
                continue
            core = [[[G.e], [G.e], [G.e]], [[G.e], [u], [v]]]
            data = graph_data(core)
            if data is not None:
                quality, adj = data
                cores.append((quality, core, adj))
    if not cores:
        raise ValueError("no connected verified long-order monomial core")
    cores.sort(key=lambda item: (item[0][0], -item[0][1]), reverse=True)
    best_girth = cores[0][0][0]
    cores = [item for item in cores if item[0][0] == best_girth][:18]

    def all_distances(adj):
        distances = []
        for source in range(len(adj)):
            d = [-1] * len(adj)
            d[source] = 0
            queue = [source]
            for a in queue:
                for b in adj[a]:
                    if d[b] < 0:
                        d[b] = d[a] + 1
                        queue.append(b)
            assert min(d) >= 0
            distances.append(d)
        return distances

    def core_cycle_words(core):
        """Actual core kernel witnesses obtained from verified columns.

        Two simple five-edge paths with equal endpoints have equal
        boundaries. Their XOR is therefore a verified word of weight <=10.
        Retain actual witnesses, including ten-edge core cycles; this
        is not a full B-kernel test or a quantum-distance certificate.
        """
        cols = columns(component_rows("B", core), 3 * q)
        adj = [[] for _ in range(2 * q)]
        for c, syn in enumerate(cols):
            bit = syn & -syn
            a = bit.bit_length() - 1
            b = (syn ^ bit).bit_length() - 1
            adj[a].append((b, c))
            adj[b].append((a, c))
        words = set()
        for root in range(2 * q):
            paths = {}
            stack = [(root, 0, 1 << root, 0)]
            while stack:
                u, depth, visited, word = stack.pop()
                if depth == 5:
                    for old in paths.get(u, ()):
                        witness = old ^ word
                        if witness and witness.bit_count() <= 10:
                            # Core occupies B columns 2,3,4.
                            words.add(witness << (2 * q))
                    paths.setdefault(u, []).append(word)
                    continue
                for v, edge in adj[u]:
                    if not ((visited >> v) & 1):
                        stack.append((v, depth + 1,
                                      visited | (1 << v),
                                      word | (1 << edge)))
        return tuple(sorted(words, key=lambda w: (w.bit_count(), w)))

    core_bank = [(quality, core, all_distances(adj),
                  core_cycle_words(core))
                 for quality, core, adj in cores]

    def two_triple_reading(cols, distances, girth):
        """Exact minimum among B words with zero or two cubic columns.

        The remaining columns are graph edges. Minimum T-join weight
        equals minimum perfect-matching cost in the shortest-path metric.
        Translate one cubic variable to group coordinate zero: the two
        cubic block anchors therefore cover every pair orbit.
        This restricted minimum is an UPPER bound on component distance.
        """
        cache = {}

        def join_weight(syn):
            if syn in cache:
                return cache[syn]
            nodes = []
            while syn:
                bit = syn & -syn
                nodes.append(bit.bit_length() - 1)
                syn ^= bit
            assert len(nodes) % 2 == 0

            def match(vertices):
                if not vertices:
                    return 0
                a = vertices[0]
                return min(
                    distances[a][vertices[t]]
                    + match(vertices[1:t] + vertices[t + 1:])
                    for t in range(1, len(vertices)))
            value = match(tuple(nodes))
            # Recover the mask for this tiny memo table.
            cache[sum(1 << v for v in nodes)] = value
            return value

        reading, hits = girth, 0
        for anchor in (0, q):
            for other in range(2 * q):
                if other == anchor:
                    continue
                weight = 2 + join_weight(cols[anchor] ^ cols[other])
                if weight < reading:
                    reading, hits = weight, 1
                elif weight == reading:
                    hits += 1
        return reading, hits

    def probe(rows, width, rounds, stabilizers=None, retain=False):
        """Actual information-set witnesses; an unresolved result is no bound.

        Component basis exchanges preserve the kernel basis. Retained
        words are subsequently tested in the complete quantum quotient.
        """
        cols = columns(rows, width)
        own = None
        if stabilizers is not None:
            own = {}
            for word in stabilizers:
                while word:
                    p = word.bit_length() - 1
                    if p in own:
                        word ^= own[p]
                    else:
                        own[p] = word
                        break

        def nontrivial(word):
            if own is None:
                return bool(word)
            while word:
                p = word.bit_length() - 1
                if p not in own:
                    return True
                word ^= own[p]
            return False

        best, found = width + 1, set()

        def record(word):
            nonlocal best
            weight = word.bit_count()
            if not word:
                return
            if retain and weight <= 12:
                found.add(word)
            if weight < best and nontrivial(word):
                best = weight

        order = list(range(width))
        for _ in range(rounds):
            rng.shuffle(order)
            pivots, nulls = {}, []
            for j in order:
                syn, word = cols[j], 1 << j
                while syn:
                    p = syn.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = (syn, word)
                        break
                    s, v = pivots[p]
                    syn ^= s
                    word ^= v
                if not syn:
                    nulls.append(word)

            if own is None:
                rng.shuffle(nulls)
                for i, word in enumerate(nulls):
                    weight = word.bit_count()
                    for j, other in enumerate(nulls):
                        if i == j:
                            continue
                        candidate = word ^ other
                        w = candidate.bit_count()
                        if 0 < w < weight:
                            word, weight = candidate, w
                    nulls[i] = word

            for word in nulls:
                record(word)
            short = sorted(nulls, key=int.bit_count)[:24]
            for i, word in enumerate(short):
                for j in range(i):
                    pair = word ^ short[j]
                    record(pair)
                    if own is None:
                        for h in range(j):
                            record(pair ^ short[h])

            # Do not let short stabilizers crowd every logical basis
            # vector out of the complete-code combination pass.
            if own is not None:
                logical = sorted(
                    (w for w in nulls if nontrivial(w)),
                    key=int.bit_count)[:16]
                for word in logical:
                    for other in short:
                        record(word ^ other)

        if retain:
            witnesses = tuple(sorted(
                found, key=lambda w: (w.bit_count(), w))[:96])
            assert all(all((row & w).bit_count() % 2 == 0
                           for row in rows) for w in witnesses)
            return best, witnesses
        return best

    def eight_screen(rows):
        """A weight-eight screen that also retains actual witnesses.

        Called only after excluding kernels through six. The absence
        result is exact; an early capped result supplies actual witnesses
        but not their complete multiplicity.
        """
        from itertools import combinations
        cols = columns(rows, 2 * q)
        bits = [1 << i for i in range(2 * q)]
        seen, words, hits, cap = {}, set(), 0, 96

        def collision(old, word):
            nonlocal hits
            difference = old ^ word
            if difference:
                assert difference.bit_count() == 8
                words.add(difference)
                hits += 1
            return len(words) >= cap

        for anchor in (0, q):
            others = [i for i in range(2 * q) if i != anchor]
            for a, b, c in combinations(others, 3):
                syn = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                word = bits[anchor] | bits[a] | bits[b] | bits[c]
                old = seen.get(syn)
                if old is None:
                    seen[syn] = word
                elif collision(old, word):
                    return False, hits, tuple(sorted(words))
        for a, b, c, d in combinations(range(2 * q), 4):
            syn = cols[a] ^ cols[b] ^ cols[c] ^ cols[d]
            old = seen.get(syn)
            if old is not None:
                word = bits[a] | bits[b] | bits[c] | bits[d]
                if collision(old, word):
                    return False, hits, tuple(sorted(words))
        return hits == 0, hits, tuple(sorted(words))

    def sector_witness_profile(hx, hz, awords, bwords):
        """Classify discovered constituent witnesses in the actual CSS pair.

        A translates on the right, B on the left. These are constituent
        actions only: every embedded nontrivial candidate is additionally
        checked against the complete verified opposite-check matrix.
        Twelve is an unresolved cutoff when this witness list finds none
        below twelve, not a lower bound on quantum distance.
        """
        def basis(rows):
            span = {}
            for word in rows:
                while word:
                    p = word.bit_length() - 1
                    if p not in span:
                        span[p] = word
                        break
                    word ^= span[p]
            return span

        def outside(word, span):
            while word:
                p = word.bit_length() - 1
                if p not in span:
                    return True
                word ^= span[p]
            return False

        ax, az = basis(hx), basis(hz)
        left = [[G.idx(G.mul(g, h)) for h in G.elements]
                for g in G.elements]
        right = [[G.idx(G.mul(h, g)) for h in G.elements]
                 for g in G.elements]

        def translated(word, permutation):
            result = 0
            while word:
                bit = word & -word
                block, t = divmod(bit.bit_length() - 1, q)
                result |= 1 << (block * q + permutation[t])
                word ^= bit
            return result

        floor, hits, tested = 12, 0, set()
        for side, witnesses, actions in (
                ("A", awords, right), ("B", bwords, left)):
            for original in witnesses:
                for action in actions:
                    word = translated(original, action)
                    key = (side, word)
                    if key in tested:
                        continue
                    tested.add(key)
                    for block in range(5 if side == "A" else 2):
                        if side == "B":
                            embedded = word << (block * 5 * q)
                            checks, own = hz, ax
                        else:
                            embedded, remaining = 0, word
                            while remaining:
                                bit = remaining & -remaining
                                j, t = divmod(bit.bit_length() - 1, q)
                                embedded |= 1 << ((j * 5 + block) * q + t)
                                remaining ^= bit
                            checks, own = hx, az
                        if not outside(embedded, own):
                            continue
                        assert all((row & embedded).bit_count() % 2 == 0
                                   for row in checks)
                        floor = min(floor, embedded.bit_count())
                        hits += 1
                        if hits >= 96:
                            return floor, hits
        return floor, hits

    winner, winner_score = None, None
    for batch in range(2):
        apool = []
        for trial in range(96):
            A = draw_A(trial)
            rows = component_rows("A", A)
            quality = component_quality(rows, 2 * q)
            apool.append((quality, trial, A, rows))
        apool.sort(key=lambda item: (item[0], item[1]))
        afinal = []
        for quality, trial, A, rows in apool[:18]:
            clean6 = no_kernel6(rows, 2 * q)
            if not clean6:
                continue
            clean8, hits8, awords = eight_screen(rows)
            reading, probed_words = probe(rows, 2 * q, 16, retain=True)
            awords = tuple(set(awords).union(probed_words))
            if not clean8:
                reading = min(reading, 8)
            # Retain singular-cell pairs as well as the strongest
            # classical pair: their quantum quotients can differ.
            cell_ranks = [
                gf2_rank(component_rows("A", [[cell]]))
                for cell in A[0]
            ]
            score = (clean8, int(max(cell_ranks) < q),
                     -gf2_rank(rows), -hits8,
                     -quality[1], -quality[2], -trial)
            afinal.append(
                (score, A, reading, clean6, clean8, hits8, awords))
            if len(afinal) >= 8:
                break
        afinal.sort(key=lambda item: item[0], reverse=True)
        afinal = afinal[:4]

        bpool = []
        for trial in range(96):
            ci = (trial + 3 * bmodel + idx // 20 + batch) % len(core_bank)
            core_quality, core, distances, bwords = core_bank[ci]
            B = draw_B(core)
            rows = component_rows("B", B)
            cols = columns(rows, 5 * q)
            restricted, hits = two_triple_reading(
                cols, distances, core_quality[0])
            quality = component_quality(rows, 5 * q)
            score = (restricted, -hits, -quality[0],
                     -quality[1], -quality[2])
            bpool.append((score, trial, B, rows, restricted, bwords))
        bpool.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        bfinal = []
        for score, trial, B, rows, restricted, bwords in bpool[:16]:
            clean6 = no_kernel6(rows, 5 * q)
            if not clean6:
                continue
            probed, probed_words = probe(rows, 5 * q, 16, retain=True)
            reading = min(restricted, probed)
            bwords = tuple(set(bwords).union(probed_words))
            quality = (reading, -gf2_rank(rows), score, -trial)
            bfinal.append((quality, B, reading, clean6, bwords))
            if len(bfinal) >= 6:
                break
        bfinal.sort(key=lambda item: item[0], reverse=True)
        bfinal = bfinal[:4]

        for _, A, da, a6, a8, hits8, awords in afinal:
            for _, B, db, b6, bwords in bfinal:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k > 0 and 9 * n <= 40 * k < 11 * n):
                    continue
                if low_logical(hz, hx, n) or low_logical(hx, hz, n):
                    continue
                witness_cutoff, logical_hits = sector_witness_profile(
                    hx, hz, awords, bwords)
                dx = probe(hz, n, 16, stabilizers=hx)
                dz = probe(hx, n, 16, stabilizers=hz)
                # A discovered constituent word caps the quantum reading
                # ONLY if its embedding survives stabilizer reduction.
                # Twelve otherwise remains an unresolved selection cap.
                reading = min(dx, dz, witness_cutoff)
                score = (k * reading * reading / n,
                         -logical_hits, k, min(dx, dz),
                         int(a8), -hits8)
                if winner_score is not None and score <= winner_score:
                    continue
                signature = _quotient_signature(hx, hz)
                if signature in excluded:
                    continue
                winner_score = score
                winner = (A, B, hx, hz, n, k, da, db, a6, a8, b6,
                          signature, witness_cutoff, logical_hits, dx, dz)
        if winner is not None and min(winner[-2:]) >= 10 and winner[-4] >= 10:
            break

    if winner is None:
        raise ValueError("No novel target-band quotient-screened pair")
    (A, B, hx, hz, n, k, da, db, a6, a8, b6,
     signature, witness_cutoff, logical_hits, dx, dz) = winner
    assert n == 576
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "signature": signature,
            "screen": {"k": k, "A_kernel6": a6, "A_kernel8": a8,
                       "B_kernel6": b6, "component_readings": (da, db),
                       "core_girth": best_girth,
                       "sector_witness_cutoff": witness_cutoff,
                       "sector_logical_hits_capped": logical_hits,
                       "logical_probe_readings": (dx, dz)}}


def _quotient_signature(hx, hz):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive.

    Unequal signatures certify inequivalence. Equal signatures are
    conservatively discarded, including potentially different graphs.
    """
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i, j] = overlap
            if overlap:
                adj = one if overlap == 1 else many
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            common = (
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count(),
            )
            for u, v in ((i, j), (j, i)):
                key = (weights[v], overlaps[i, j]) + common
                profiles[u][key] = profiles[u].get(key, 0) + 1
    return tuple(sorted(
        (weights[i], tuple(sorted(profiles[i].items())))
        for i in range(m)))


from functools import lru_cache


@lru_cache(maxsize=1)
def _quotient_portfolio(G):
    """Immutable portfolio; caching does not change the function's result."""
    excluded, entries = set(), []
    for idx in range(N_SLOTS):
        entry = None
        for restart in range(2):
            local = random.Random(
                BASE_SEED * 10007 + 0x715020 + 65537 * idx
                + 1000003 * restart)
            try:
                result = _triangle_design(local, G, idx, excluded)
            except ValueError:
                continue
            excluded.add(result["signature"])
            freeze = lambda grid: tuple(
                tuple(tuple(cell) for cell in row) for row in grid)
            entry = (
                freeze(result["cells"]["A"]),
                freeze(result["cells"]["B"]),
                tuple(sorted(result["screen"].items())),
            )
            break
        entries.append(entry)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a deterministically screened structural card."""
    entry = _quotient_portfolio(G)[idx]
    if entry is None:
        raise ValueError("Novel quotient-screened portfolio card exhausted")
    A, B, screen = entry
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh list without altering the immutable portfolio."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: The Dic11 parent's 2-second distance reading always survives the longer search: slot 42 fell from 22 to 20, with mixed-sector Z witnesses.
# REFUTED: S4 x C2 cycle shortlisting and kernel-six preferences suffice to eliminate weight-seven logicals: measured slots 16, 18 and 30 retained d30s=7, predominantly in sector 1.
# REFUTED: The all-singleton B profile supplies usable logical-four-screened target-band fillings under the parent's order policies: all twelve corresponding slots were voided.
# REFUTED: Exact A weight-eight screening plus connected-core T-join selection raises this bed above its distance-eight plateau: all five current measured finalists had k=156,d30s=8, and sixty slots collapsed to twenty-one distinct codes.
# REFUTED: Joint quotient-aware selection with order-three B reinforcement raises S4 x C2 fitness above 17.333: the current program still measured five [576,156,8] finalists and scored 17.333 despite retaining 43 distinct codes.
# REFUTED: The current order-48 long-core/retained-witness experiment supplies five readings above eight: the supplied feedback instead reports five [576,156,8] measured finalists and fitness 17.333.
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
