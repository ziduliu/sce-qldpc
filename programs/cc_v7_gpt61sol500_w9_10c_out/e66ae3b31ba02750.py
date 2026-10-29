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
CONCEPT = ("C3 semidirect C16 with inversion action, order 48, A=(1,2), B=(1,2): replacing B's degree-two matching core by two singular trinomials and screening the complete quantum quotient should supply five measured readings >=10 and improve cell 0.250; measurements can refute this.")
GROUP = ("Keep CUSTOM_GROUP, GTYPE, A0 and order 48 unchanged. The parent reached fitness 20.9252 with five distance-nine finalists; its X minima remained lighter than its Z minima. The supplied BEDS has no order-48 equal-(1,2) bed. Retain the measured group while testing a new constituent mechanism, not an element-order distance guarantee.")
SHAPES = ("Change B to (1,2), keeping A=(1,2): equal constituents have 96 variables and 48 checks, n=240, kb=48 and guaranteed rate 0.200. The size window permits q<=119, while valid_q still fixes q=48. Actual admission requires k=54..65, rate in [0.225,0.275), through verified rank gains rather than the bound alone. Parent weight-nine X minima occupied sector-1 Acol0/1 across Bcols0..4; no measured minimum required Brow blocks.")
FILLING = ("All four cells contain three distinct elements, giving BOTH CSS check maxima nine. Twenty intrinsic prescriptions combine complete triangle-order spectra, commutativity and individual-cell rank deficits. Optimize verified constituents with row-rank deficit two or four, exactly exclude kernel supports through six and refine explicit kernel witnesses. Pairings enforce actual target-band k, complete logical exclusion through four, embedded-witness classification, stabilizer-pair reduction and full-code probes. An immutable portfolio conservatively rejects equal full uncoloured Tanner signatures. Witness weights and all larger probe readings are upper bounds, not distance certificates; unseen-archive novelty remains unproved.")
FAMILY_NAME = "r025_c3sd_c16_inv_q48_12x12_singular_triangle_quotient_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C3 semidirect C16 has order 48")
    # (a,b)=x^a y^b; x^3=y^16=1, y*x*y^-1=x^-1.
    # Inversion has order two, dividing sixteen; centre=<y^2>.
    els = [(a, b) for b in range(16) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 16)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 3,
                (b + d) % 16)

    def inv(g):
        a, b = red(g)
        return ((a if b & 1 else -a) % 3, (-b) % 16)

    def idx(g):
        a, b = red(g)
        return 3 * b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _lo_design(rng, G, idx, excluded):
    """Optimize an intrinsic triangle/reinforcement card, not an orbit copy."""
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 1, 1, 1, 0), (0, 1, 1, 1, 3)),
    )
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    card = idx % 20
    profile, policy = card // 10, card % 10
    apolicy = pairs[policy]
    sa, sb = ((3, 3),), profiles[profile]
    mono = [j for j in range(5) if sb[0][j] == sb[1][j] == 1]
    assert len(mono) == 3
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(W_MAX, 9)

    def element_order(g):
        power = G.e
        for order in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return order
        raise ValueError("unexpected element order")

    orders = {g: element_order(g) for g in G.elements}
    odd16 = [g for g in G.elements if orders[g] == 16]
    # Classes distinguish commuting structure, not just anchor choices.
    # The two commuting classes deliberately overlap, but have different
    # admissible short-ratio spectra; full Tanner invariants gate novelty.
    banks = [[] for _ in range(4)]
    for r in G.elements:
        if r == G.e:
            continue
        for s in G.elements:
            if s == G.e or s == r:
                continue
            spectrum = tuple(sorted((
                orders[r], orders[s],
                orders[G.mul(G.inv(r), s)])))
            commute = G.mul(r, s) == G.mul(s, r)
            if not commute and spectrum[0] >= 12:
                banks[0].append((r, s))
            if commute and spectrum[0] >= 8:
                banks[1].append((r, s))
            if commute and spectrum[0] >= 4 and spectrum[-1] >= 12:
                banks[2].append((r, s))
            if not commute and spectrum[0] >= 3:
                banks[3].append((r, s))
    if not odd16 or any(not bank for bank in banks):
        raise ValueError("required odd16/commuting triangle class absent")

    def draw_cell(count, side, j=0):
        if count == 0:
            return []
        first = G.pick(rng)
        if count == 1:
            return [first]
        if count == 2:
            # These odd ratios are absent from the parent's even
            # order-12/24 reinforcement model.
            return [first, G.mul(first, rng.choice(odd16))]
        kind = apolicy[j] if side == "A" else (1 if j == 0 else 0)
        r, s = rng.choice(banks[kind])
        cell = [first, G.mul(first, r), G.mul(first, s)]
        assert len(set(cell)) == count == 3
        return cell

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

    def core_girth(cols, checks):
        """Girth of the degree-two core extracted from the verified B.

        Graph edges are constituent columns, so graph-cycle length
        equals the weight of a core kernel word. Search through eight.
        """
        adj = [[] for _ in range(checks)]
        edges = set()
        for syndrome in cols:
            if syndrome.bit_count() != 2:
                return 0
            a = (syndrome & -syndrome).bit_length() - 1
            b = (syndrome ^ (1 << a)).bit_length() - 1
            edge = (min(a, b), max(a, b))
            if edge in edges:
                return 2
            edges.add(edge)
            adj[a].append(b)
            adj[b].append(a)

        best = checks + 1
        for root in range(checks):
            distance = [-1] * checks
            parent = [-1] * checks
            distance[root] = 0
            queue = [root]
            for u in queue:
                if distance[u] >= 4:
                    continue
                for v in adj[u]:
                    if distance[v] < 0:
                        distance[v] = distance[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v:
                        best = min(best, distance[u] + distance[v] + 1)
                        if best < 8:
                            return best
        return best

    def clear_A8(rows):
        """Exact A exclusion through eight, called after no_kernel6.

        Odd column degree excludes odd kernel weights. Every eight-word
        has a regular-action translate containing an identity-coordinate
        variable in one of the two blocks. Compare anchored quartets with
        all quartets using syndromes from the verified component.
        """
        from itertools import combinations
        width = 2 * G.q
        cols = columns(rows, width)
        bits = [1 << i for i in range(width)]
        seen = {}
        for anchor in (G.idx(G.e), G.q + G.idx(G.e)):
            others = [i for i in range(width) if i != anchor]
            for a, b, c in combinations(others, 3):
                syndrome = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                word = bits[anchor] | bits[a] | bits[b] | bits[c]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif old != word:
                    return False
        for a, b, c, d in combinations(range(width), 4):
            syndrome = cols[a] ^ cols[b] ^ cols[c] ^ cols[d]
            old = seen.get(syndrome)
            if old is not None:
                word = bits[a] | bits[b] | bits[c] | bits[d]
                if old != word:
                    return False
        return True

    def join_reading(cols, distances, girth, four=False):
        """Actual B witness upper bound from zero/two/four cubic variables.

        A minimum T-join is a minimum matching in the graph-distance
        metric. Two cubic-block anchors cover pair/four-support orbits.
        Budget truncation only avoids witnesses unable to improve the
        current reading. Six/eight-cubic branches are NOT excluded here.
        """
        from itertools import combinations
        cubic = [i for i, c in enumerate(cols) if c.bit_count() == 3]
        assert len(cubic) == 2 * G.q
        anchors = [
            j * G.q + G.idx(G.e)
            for j in range(5)
            if cols[j * G.q + G.idx(G.e)].bit_count() == 3
        ]
        assert len(anchors) == 2
        memo = {}

        def join(boundary, budget):
            if not boundary:
                return 0
            if budget < 0 or boundary.bit_count() > 2 * budget:
                return budget + 1
            key = (boundary, budget)
            if key in memo:
                return memo[key]
            bit = boundary & -boundary
            u = bit.bit_length() - 1
            rest = boundary ^ bit
            choices = rest
            best = budget + 1
            while choices:
                other = choices & -choices
                v = other.bit_length() - 1
                distance = distances[u][v]
                if distance < best:
                    value = distance + join(rest ^ other, budget - distance)
                    best = min(best, value)
                choices ^= other
            memo[key] = best
            return best

        best = girth
        for anchor in anchors:
            for other in cubic:
                if other == anchor:
                    continue
                boundary = cols[anchor] ^ cols[other]
                best = min(best, 2 + join(boundary, best - 3))
        if not four or best < 9:
            return best

        # Four cubic variables require only a small boundary to improve
        # a nine/ten-weight reading. Most triples are discarded by parity
        # boundary size before any matching recursion.
        for anchor in anchors:
            others = [i for i in cubic if i != anchor]
            for a, b, c in combinations(others, 3):
                boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                budget = best - 5
                if boundary.bit_count() > 2 * budget:
                    continue
                best = min(best, 4 + join(boundary, budget))
                if best < 9:
                    return best
        return best

    def probe(rows, width, tries, stabilizers=None):
        """Refined kernel-basis probes of the VERIFIED matrix.

        Component returns are actual kernel-witness weights. For CSS
        probes twelve is an unresolved cap, never a lower distance bound.
        """
        cols = columns(rows, width)
        bits = [1 << i for i in range(width)]
        best = 12 if stabilizers is not None else width + 1
        best_word = 0
        span = {}
        if stabilizers is not None:
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    if p in span:
                        row ^= span[p]
                    else:
                        span[p] = row
                        break

        def record(word):
            nonlocal best, best_word
            weight = word.bit_count()
            if not word or weight >= best:
                return
            if stabilizers is not None:
                reduced = word
                while reduced:
                    p = reduced.bit_length() - 1
                    if p not in span:
                        break
                    reduced ^= span[p]
                if not reduced:
                    return
            best, best_word = weight, word

        for _ in range(tries):
            order = list(range(width))
            rng.shuffle(order)
            pivots = {}
            kernel = []
            for i in order:
                syndrome, word = cols[i], bits[i]
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = (syndrome, word)
                        break
                    old_s, old_w = pivots[p]
                    syndrome ^= old_s
                    word ^= old_w
                if not syndrome:
                    kernel.append(word)
                    record(word)
            # Elementary exchanges preserve the complete kernel while
            # exposing combinations missed by the original basis scan.
            rng.shuffle(kernel)
            for i, word in enumerate(kernel):
                weight = word.bit_count()
                for j, other in enumerate(kernel):
                    if i == j:
                        continue
                    candidate = word ^ other
                    new_weight = candidate.bit_count()
                    if 0 < new_weight < weight:
                        word, weight = candidate, new_weight
                kernel[i] = word
                record(word)
            for i, a in enumerate(kernel):
                for b in kernel[:i]:
                    record(a ^ b)
            short = sorted(kernel, key=int.bit_count)[:28]
            for i, a in enumerate(short):
                for j in range(i):
                    pair = a ^ short[j]
                    for h in range(j):
                        record(pair ^ short[h])
            for _ in range(2):
                if not best_word:
                    break
                starting = best_word
                for i, a in enumerate(short):
                    record(starting ^ a)
                    for b in short[:i]:
                        record(starting ^ a ^ b)
                if best_word == starting:
                    break
            if best < 8:
                return best
        return best

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
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    unit = [[[G.e]]]

    def component_rows(grid, side):
        """Expose the exact evaluator action using a unit opposite side."""
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, unit)
            width, checks = ca * q, ra * q
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
            width, checks = cb * q, rb * q
        mask = (1 << width) - 1
        return [row & mask for row in rows[:checks]]

    core_bank = _lo_core_bank(G)
    winner = None
    winner_quality = None
    for batch in range(2):
        aproposals = []
        for trial in range(112):
            A = [[draw_cell(c, "A", j) for j, c in enumerate(row)]
                 for row in sa]
            ar = component_rows(A, "A")
            qa = component_quality(ar, ca * q)
            if qa[0]:
                continue
            cheap = (20 * qa[1] + qa[2], qa[1], trial)
            aproposals.append((cheap, A, ar, qa))
        aproposals.sort(key=lambda item: item[0])

        aset = []
        for cheap, A, ar, qa in aproposals[:20]:
            if not no_kernel6(ar, ca * q):
                continue
            clean8 = clear_A8(ar)
            da = probe(ar, ca * q, 20)
            assert da >= 8
            if not clean8:
                da = min(da, 8)
            quality = (int(clean8), min(da, 12), -cheap[0], -cheap[2])
            aset.append((quality, A, da, qa, clean8))
            if len(aset) >= 8:
                break
        aset.sort(key=lambda item: item[0], reverse=True)
        aset = aset[:5]
        if not aset:
            continue

        bproposals = []
        for trial in range(176):
            B = [[draw_cell(c, "B", j) for j, c in enumerate(row)]
                 for row in sb]
            g, h, girth, distances = rng.choice(core_bank)
            for j, relative in zip(mono, (G.e, g, h)):
                B[0][j] = [G.e]
                B[1][j] = [relative]
            br = component_rows(B, "B")
            bcols = columns(br, cb * q)
            qb = component_quality(br, cb * q)
            if qb[0]:
                continue
            restricted = join_reading(bcols, distances, girth)
            cheap = (-restricted, 20 * qb[1] + qb[2], trial)
            bproposals.append(
                (cheap, B, br, bcols, qb, girth, distances))
        bproposals.sort(key=lambda item: item[0])

        bset = []
        for cheap, B, br, bcols, qb, girth, distances in bproposals[:16]:
            if not no_kernel6(br, cb * q):
                continue
            restricted = join_reading(bcols, distances, girth, four=True)
            db = min(restricted, probe(br, cb * q, 36))
            assert db >= 7
            quality = (min(db, 12), -cheap[1], -cheap[2])
            bset.append((quality, B, db, qb, girth))
            if len(bset) >= 8:
                break
        bset.sort(key=lambda item: item[0], reverse=True)
        bset = bset[:5]
        if not bset:
            continue

        pairs_to_test = []
        for aq, A, da, qa, clean8 in aset:
            for bq, B, db, qb, girth in bset:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k > 0 and 9 * n <= 40 * k < 11 * n):
                    continue
                reading = min(da, db, 12)
                quality = (
                    k * reading * reading / float(n),
                    int(clean8), reading,
                    -(qa[1] + qb[1]), aq, bq,
                )
                pairs_to_test.append(
                    (quality, A, B, hx, hz, n, k,
                     da, db, girth, clean8))
        pairs_to_test.sort(key=lambda item: item[0], reverse=True)

        tested = 0
        for candidate in pairs_to_test:
            (_, A, B, hx, hz, n, k, da, db, girth, clean8) = candidate
            if low_logical(hz, hx, n) or low_logical(hx, hz, n):
                continue
            signature = _lo_signature(hx, hz, n)
            if signature in excluded:
                continue
            tested += 1
            dx = probe(hz, n, 6, hx)
            dz = probe(hx, n, 6, hz)
            if min(dx, dz) < 7:
                continue
            reading = min(da, db, dx, dz, 12)
            quality = (
                k * reading * reading / float(n),
                reading, int(clean8), k,
            )
            if winner_quality is None or quality > winner_quality:
                assert n == 576 and css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
                winner_quality = quality
                winner = {
                    "idx": idx,
                    "cells": {"A": A, "B": B},
                    "signature": signature,
                    "screen": {
                        "k": k, "exact_core_girth": girth,
                        "kernel6_A": True, "kernel6_B": True,
                        "kernel8_A": clean8,
                        "component_probe_A": da,
                        "component_probe_B": db,
                        "logical_probe_X_cap12": dx,
                        "logical_probe_Z_cap12": dz,
                        "selection_reading": reading,
                        "below9_fallback": reading < 9,
                    },
                }
            if tested >= 5:
                break
        if winner is not None and winner_quality[1] >= 9:
            break
    if winner is None:
        raise ValueError("No invariant-novel odd16 pair passed target-rate CSS screens")
    return winner


from functools import lru_cache


@lru_cache(maxsize=1)
def _lo_core_bank(G):
    """Immutable exact-girth core bank built only through verified lifts.

    Include relative orders six/eight, which the parent excluded.
    Conjugacy representatives prune proposals, not arbitrary graph isos.
    Disconnected cores are allowed: reinforcement columns may connect
    them. BFS covers every check, so the recorded girth is exact.
    """
    def order(g):
        value = G.e
        for t in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    steps = [g for g in G.elements if orders[g] >= 6]
    representatives = [
        g for g in steps
        if G.idx(g) == min(
            G.idx(G.mul(G.mul(a, g), G.inv(a)))
            for a in G.elements)
    ]
    unit = [[[G.e]]]
    records = []
    for g in representatives:
        for h in steps:
            if g == h or orders[G.mul(G.inv(g), h)] < 6:
                continue
            core = [[[G.e], [G.e], [G.e]], [[G.e], [g], [h]]]
            _, rows, _ = lifted_matrices(G, unit, core)
            mask = (1 << (3 * G.q)) - 1
            rows = [row & mask for row in rows[:2 * G.q]]
            cols = [0] * (3 * G.q)
            for r, row in enumerate(rows):
                while row:
                    bit = row & -row
                    cols[bit.bit_length() - 1] |= 1 << r
                    row ^= bit
            adj = [[] for _ in rows]
            for edge, syndrome in enumerate(cols):
                assert syndrome.bit_count() == 2
                bit = syndrome & -syndrome
                a = bit.bit_length() - 1
                b = (syndrome ^ bit).bit_length() - 1
                adj[a].append((b, edge))
                adj[b].append((a, edge))

            girth = len(rows) + 1
            distances = []
            for root in range(len(rows)):
                distance = [10000] * len(rows)
                parent_edge = [-1] * len(rows)
                distance[root] = 0
                queue = [root]
                for a in queue:
                    for b, edge in adj[a]:
                        if distance[b] == 10000:
                            distance[b] = distance[a] + 1
                            parent_edge[b] = edge
                            queue.append(b)
                        elif (edge != parent_edge[a]
                              and edge != parent_edge[b]):
                            girth = min(
                                girth, distance[a] + distance[b] + 1)
                distances.append(tuple(distance))
            if girth >= 8:
                records.append((g, h, girth, tuple(distances)))
    if not records:
        raise ValueError("no verified matching core of girth at least eight")
    best = max(record[2] for record in records)
    records = [record for record in records if record[2] == best]
    records.sort(key=lambda record: (G.idx(record[0]), G.idx(record[1])))
    return tuple(records)


def _lo_signature(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant, not a complete iso test.

    Unequal signatures certify inequivalence; equality causes conservative
    rejection. No label indices or X/Z colouring enter the signature.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i, j] = overlap
            if overlap:
                adjacency = one if overlap == 1 else many
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            common = (
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count(),
            )
            for u, v in ((i, j), (j, i)):
                profiles[u][(weights[v], overlaps[i, j]) + common] += 1
    degrees = [0] * n
    for row in rows:
        while row:
            bit = row & -row
            degrees[bit.bit_length() - 1] += 1
            row ^= bit
    return (
        tuple(sorted(Counter(degrees).items())),
        tuple(sorted(
            (weights[i], tuple(sorted(profiles[i].items())))
            for i in range(m))),
    )


def _tp_columns(rows, width):
    cols = [0] * width
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << i
            row ^= bit
    return cols


def _tp_basis(rows):
    span = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                span[p] = word
                break
            word ^= old
    return span


def _tp_outside(word, span):
    while word:
        p = word.bit_length() - 1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _tp_component(G, grid, right=False):
    """The zero auxiliary side exposes exactly the verified action."""
    zero = [[[]]]
    if right:
        _, rows, _ = lifted_matrices(G, zero, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, zero)
    width = len(grid[0]) * G.q
    assert all(row.bit_length() <= width for row in rows)
    return rows, width


def _tp_clear6(rows, width):
    """Exact constituent exclusion through six, not a probe reading."""
    cols = _tp_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, a in enumerate(cols):
        for j in range(i):
            value = a ^ cols[j]
            if value in singles or value in pairs:
                return False
            pairs.add(value)
    triples = set()
    for i, a in enumerate(cols):
        for j in range(i):
            pair = a ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                if (value == 0 or value in singles
                        or value in pairs or value in triples):
                    return False
                triples.add(value)
    return True


def _tp_logical4(checks, own, width):
    """Exactly detect a nontrivial logical support of weight <=4."""
    cols = _tp_columns(checks, width)
    seen = {0: 0}

    def inspect(syndrome, word):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
            return False
        return _tp_outside(word ^ old, own)

    for i, syndrome in enumerate(cols):
        if inspect(syndrome, 1 << i):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            if inspect(syndrome ^ cols[j], (1 << i) | (1 << j)):
                return True
    return False


def _tp_probe(rows, width, rng, passes, own=None, seed=0):
    """Return actual kernel/logical witnesses, never an unresolved cap."""
    cols = _tp_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1
    retained = set()

    def inspect(word):
        nonlocal best, weight
        if not word:
            return
        w = word.bit_count()
        if own is None and w <= 16:
            retained.add(word)
        if w < weight and (own is None or _tp_outside(word, own)):
            best, weight = word, w

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

        # Elementary exchanges retain a complete kernel basis.
        rng.shuffle(kernel)
        for i, word in enumerate(kernel):
            w = word.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            kernel[i] = word
            inspect(word)

        for i, a in enumerate(kernel):
            for b in kernel[:i]:
                inspect(a ^ b)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, a in enumerate(short):
            for j in range(i):
                pair = a ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])

        if own is not None:
            # Short stabilizers must not displace every logical vector.
            logical = sorted(
                (w for w in kernel if _tp_outside(w, own)),
                key=int.bit_count)[:16]
            for a in logical:
                for i, b in enumerate(short):
                    inspect(a ^ b)
                    for c in short[:i]:
                        inspect(a ^ b ^ c)
        if best and weight < 8:
            break

    assert best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    assert own is None or _tp_outside(best, own)
    retained.add(best)
    words = tuple(sorted(retained, key=lambda w: (w.bit_count(), w))[:24])
    return weight, best, words


def _tp_seed(words, kind, G, checks, stabilizers, own):
    """Classify constituent embeddings and reduce actual logical cosets."""
    q = G.q
    candidates = set()
    for source in words:
        for block in range(2):
            if kind == "X":
                word = source << (block * 2 * q)
            else:
                word, remaining = 0, source
                while remaining:
                    bit = remaining & -remaining
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((2 * j + block) * q + t)
                    remaining ^= bit
            if not _tp_outside(word, own):
                continue
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            candidates.add(word)
    if not candidates:
        return 0

    moves = list(stabilizers)
    moves.extend(a ^ b for i, a in enumerate(stabilizers)
                 for b in stabilizers[:i])
    best = 0
    for word in sorted(candidates, key=lambda w: (w.bit_count(), w))[:4]:
        for _ in range(3):
            starting, weight = word, word.bit_count()
            for move in moves:
                trial = starting ^ move
                tw = trial.bit_count()
                if 0 < tw < weight:
                    word, weight = trial, tw
            if word == starting:
                break
        assert _tp_outside(word, own)
        if not best or word.bit_count() < best.bit_count():
            best = word
    return best


@lru_cache(maxsize=1)
def _lo_portfolio(G):
    """Pure equal-(1,2) portfolio; old matching-core helpers are uncalled.

    Twenty intrinsic prescriptions optimize candidates before publication.
    Only final, full-Tanner-separated matrices become slots. Cached entries
    are immutable, and no saved or archived filling seeds this search.
    """
    from itertools import combinations, combinations_with_replacement

    assert G.q == 48
    assert A_SHAPE == B_SHAPE == (1, 2)
    q, width = G.q, 2 * G.q
    supports = ((3, 3),)
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 9
    assert 9 <= min(W_MAX, 12)

    def order(g):
        value = G.e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return length
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    nonidentity = [g for g in G.elements if g != G.e]
    classes = {}
    # A unit cell makes the complete row full rank, defeating the intended
    # rank-gain mechanism. Enumerate singular cells using VERIFIED ranks.
    for r, s in combinations(nonidentity, 2):
        cell = [G.e, r, s]
        rows, _ = _tp_component(G, [[cell]])
        deficit = q - gf2_rank(rows)
        if deficit not in (2, 4, 6, 8):
            continue
        spectrum = tuple(sorted((
            orders[r], orders[s], orders[G.mul(G.inv(r), s)])))
        commute = G.mul(r, s) == G.mul(s, r)
        classes.setdefault((deficit, commute, spectrum), []).append(cell)
    if not classes:
        raise ValueError("no usable singular trinomial classes")

    keys = sorted(classes)
    prescriptions = list(combinations_with_replacement(range(len(keys)), 2))
    prescriptions.sort(key=lambda p: (
        -min(keys[p[0]][2][0], keys[p[1]][2][0]),
        -sum(keys[p[0]][2]) - sum(keys[p[1]][2]),
        keys[p[0]][1] == keys[p[1]][1], p))

    bank, fillings = [], set()
    for card in range(20):
        desired = 2 if card % 2 == 0 else 4
        eligible = [
            p for p in prescriptions
            if min(keys[p[0]][0], keys[p[1]][0]) >= desired]
        if not eligible:
            continue
        preferred = eligible[(card // 2) % len(eligible)]
        local = random.Random(BASE_SEED ^ (0x481212 + 104729 * card))
        proposals = []
        for trial in range(96):
            # The broad tail repairs a failed algebraic prescription, not
            # an emitted illegal code. Exact row deficit remains mandatory.
            policy = preferred if trial < 64 else local.choice(eligible)
            cells = []
            for ci in policy:
                anchor = G.pick(local)
                cell = [G.mul(anchor, g)
                        for g in local.choice(classes[keys[ci]])]
                assert len(set(cell)) == 3
                cells.append(cell)
            grid = [cells]
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for cell in cells)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows, _ = _tp_component(G, grid)
            if q - gf2_rank(rows) != desired:
                continue
            penalty = 0
            for i, row in enumerate(rows):
                for other in rows[:i]:
                    shared = (row & other).bit_count()
                    penalty += shared * (shared - 1) // 2
            proposals.append((penalty, trial, policy, grid, rows))
        proposals.sort(key=lambda r: (r[0], r[1]))

        finalists = []
        for penalty, trial, policy, grid, rows in proposals[:24]:
            if not _tp_clear6(rows, width):
                continue
            reading, witness, words = _tp_probe(rows, width, local, 6)
            # Every constituent column has odd degree three: no odd kernel
            # support exists. The exact through-six gate therefore gives 8.
            assert reading >= 8
            finalists.append(dict(
                grid=grid, rows=rows, deficit=desired, card=card,
                policy=policy, penalty=penalty, trial=trial,
                reading=reading, witness=witness, left=words))
            if len(finalists) >= 6:
                break
        finalists.sort(key=lambda r: (
            -r["reading"], r["penalty"], r["trial"]))
        for record in finalists[:4]:
            reading, witness, words = _tp_probe(
                record["rows"], width, local, 18,
                seed=record["witness"])
            assert reading >= 8
            right_rows, _ = _tp_component(G, record["grid"], right=True)
            assert q - gf2_rank(right_rows) == desired

            # Inversion transfers left-component coordinates to the exact
            # right action. Verify the transfer instead of assuming it.
            right_words = []
            for source in words:
                word = 0
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    ti = G.idx(G.inv(G.elements[t]))
                    word |= 1 << (j * q + ti)
                    source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in right_rows)
                right_words.append(word)
            record.update(
                reading=reading, witness=witness, left=words,
                right=tuple(right_words))
            bank.append(record)
    if len(bank) < 5:
        raise ValueError("too few kernel-six-clean singular constituents")

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            quality = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                -a["penalty"] - b["penalty"],
                int(a["card"] != b["card"]))
            plans.append((quality, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    results, excluded, tested = [], set(), 0
    wanted = max(N_SLOTS + 12, (3 * N_SLOTS) // 2)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 240 or not (225 * n <= 1000 * k < 275 * n):
            continue
        assert max(row.bit_count() for row in hx + hz) == 9
        signature = _lo_signature(hx, hz, n)
        if signature in excluded:
            continue
        bx, bz = _tp_basis(hx), _tp_basis(hz)
        if (_tp_logical4(hz, bx, n)
                or _tp_logical4(hx, bz, n)):
            continue
        assert css_ok(hx, hz)
        excluded.add(signature)
        tested += 1

        xseed = _tp_seed(b["right"], "X", G, hz, hx, bx)
        zseed = _tp_seed(a["left"], "Z", G, hx, hz, bz)
        local = random.Random(
            BASE_SEED ^ (0x240025 + 65537 * ai + 1000003 * bi))
        dx, xword, _ = _tp_probe(hz, n, local, 8, bx, xseed)
        dz, zword, _ = _tp_probe(hx, n, local, 8, bz, zseed)
        reading = min(dx, dz)
        if reading >= 8:
            screen = {
                "k": k, "rank_gain_above_bound": k - q,
                "constituent_rank_deficits": (a["deficit"], b["deficit"]),
                "constituent_kernel6_excluded": True,
                "component_witness_weights": (a["reading"], b["reading"]),
                "logical_witness_upper_bounds": (dx, dz),
                "design_cards": (a["card"], b["card"]),
                "below10_fallback": reading < 10,
            }
            quality = (k * reading * reading / n, reading,
                       preliminary, k)
            results.append((
                quality, freeze(A), freeze(B),
                tuple(sorted(screen.items()))))
        if len(results) >= wanted or tested >= max(160, 4 * N_SLOTS):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    entries = [(A, B, screen) for _, A, B, screen in results[:N_SLOTS]]
    entries.extend([None] * (N_SLOTS - len(entries)))
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a deliberately screened invariant-distinct design."""
    entry = _lo_portfolio(G)[idx]
    if entry is None:
        raise ValueError("Equal-shape singular-triangle portfolio exhausted")
    A, B, screen = entry
    return {"idx": idx, "cells": {"A": A, "B": B}, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without resampling or shared mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: The Dic11 parent's 2-second distance reading always survives the longer search: slot 42 fell from 22 to 20, with mixed-sector Z witnesses.
# REFUTED: Cycle-first selection with preferential boolean constituent screens suffices for five distance-eight readings on this bed: measured slots 16, 18 and 30 instead had d30s=7, predominantly with sector-1 X witnesses.
# REFUTED: All twenty S4 x C2 spectrum/core cards are publishable under the mandatory gates: the current parent voided 30 of 60 slots, retaining only 18 distinct codes.
# REFUTED: Long-order voltages with even order-12/24 reinforcements and bounded constituent probes yield five readings >=9 on C3 semidirect C16: four measured parent finalists had d30s=8 and only slot 16 reached nine.
# REFUTED: Twenty odd16 triangle/support prescriptions yield twenty distinct first-pass codes: measured slots 5 and 7 were isomorphic to slot 4, and slot 6 was isomorphic to slot 1.
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
