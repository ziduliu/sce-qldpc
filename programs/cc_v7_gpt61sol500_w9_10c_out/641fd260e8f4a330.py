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
CONCEPT = ("C7 semidirect_-1 C4, order 28, A=(1,2), B=(3,7): long degree-two cores reinforced by four cubic columns, paired with weight-eight-screened A constituents, should supply five measured distance-ten readings and improve cell 0.250. This is a falsifiable measurement hypothesis.")
GROUP = ("Preserve the inversion-sem idirect construction rule, resizing its parameters to m=7,r=4,s=-1. The related Z7:Z4(s=6), order-28 (2,4)x(2,4) bed reached fitness 29.0. The new factor seventeen makes order 48 invalid; order 28 gives n=476. Rotation orders seven/fourteen and outside order four motivate proposals, not a distance guarantee.")
SHAPES = ("Change to A=(1,2), B=(3,7): q=28, n=476, kb=112 and guaranteed rate 4/17=0.235294 in cell 0.250. The size window permits q<=35. A has 56 variables/28 checks and B has 196 variables/84 checks. Parent measured weight-nine X minima occupied sector-1 Acol0/1 across Bcols and were lighter than Z minima. Three B check blocks replace the two-row bottleneck; verified complete ranks enforce the target band.")
FILLING = ("Twenty cards combine triangle versus parallel/pendant B cores, five cubic quotient patterns and two A trinomial structures. A rows total six with column degree three; every B row totals six and B columns have degrees two or three, so HX<=9 and HZ=9. Verified zero-side lifts provide all constituent matrices. Exact kernel-six screening, explicit A weight-eight collision witnesses and equal-budget information-set searches select independent banks. Complete lifts enforce ranks, CSS, kernel-four exclusion and actual nontrivial embedded/mixed-sector witnesses. An immutable portfolio rejects equal full uncoloured Tanner invariants. Larger readings and unseen-archive novelty remain hypotheses.")
FAMILY_NAME = "r025_c7sd_c4_inv_q28_12x37_cubic_core_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C4 has order 28")
    # Same inversion-sem idirect rule: x^7=y^4=1,
    # y*x*y^-1=x^-1; element (a,b)=x^a y^b.
    # Inversion has order two, dividing four; centre=<y^2>.
    els = [(a, b) for b in range(4) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 7,
                (b + d) % 4)

    def inv(g):
        a, b = red(g)
        return ((a if b & 1 else -a) % 7, (-b) % 4)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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


@lru_cache(maxsize=1)
def _lo_portfolio(G):
    """Deterministic immutable portfolio; call order changes no result.

    Local exclusions ensure that no two emitted codes have equal intrinsic
    Tanner signatures. They do not claim novelty against the unavailable
    archive. No archived cells seed this construction.
    """
    entries, excluded = [], set()
    for idx in range(N_SLOTS):
        entry = None
        for restart in range(2):
            local = random.Random(
                BASE_SEED * 10007 + 0x481609
                + 65537 * idx + 1000003 * restart)
            try:
                result = _lo_design(local, G, idx, excluded)
            except ValueError:
                continue
            excluded.add(result["signature"])

            def freeze(grid):
                return tuple(tuple(tuple(cell) for cell in row)
                             for row in grid)

            entry = (
                freeze(result["cells"]["A"]),
                freeze(result["cells"]["B"]),
                tuple(sorted(result["screen"].items())),
            )
            break
        entries.append(entry)
    return tuple(entries)


def _s37_columns(rows, width):
    """Syndromes of matrices supplied by the verified scaffold."""
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _s37_basis(rows):
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


def _s37_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _s37_clean4(cols):
    """Exactly exclude every nonzero kernel support of weight <=4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _s37_clean6(cols, q, blocks):
    """Exact constituent exclusion through six using regular-action anchors.

    Every support can be translated to contain the identity coordinate
    in one of its blocks. An anchored triple and an arbitrary support
    of size <=3 detect every remaining five- or six-word.
    This symmetry is used only for constituents, never full CSS codes.
    """
    from itertools import combinations
    if not _s37_clean4(cols):
        return False
    low = {0} | set(cols)
    for i, first in enumerate(cols):
        for second in cols[:i]:
            low.add(first ^ second)

    anchored = {}
    width = len(cols)
    for anchor in range(0, blocks * q, q):
        others = [j for j in range(width) if j != anchor]
        for a, b in combinations(others, 2):
            syndrome = cols[anchor] ^ cols[a] ^ cols[b]
            if syndrome in low:
                return False
            word = (1 << anchor) | (1 << a) | (1 << b)
            old = anchored.get(syndrome)
            if old is not None and old != word:
                return False
            anchored[syndrome] = word

    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in low:
                    return False
                old = anchored.get(syndrome)
                if old is not None:
                    word = (1 << i) | (1 << j) | (1 << h)
                    if old != word:
                        return False
    return True


def _s37_A8(cols, q):
    """Return an actual weight-eight A word, or zero if eight is excluded.

    Called after exact kernel-six exclusion. Odd column degrees exclude
    odd weights, and anchored quartet collisions cover every eight-word.
    """
    from itertools import combinations
    assert len(cols) == 2 * q
    assert all(col.bit_count() & 1 for col in cols)
    seen = {}
    for anchor in (0, q):
        others = [j for j in range(2 * q) if j != anchor]
        for a, b, c in combinations(others, 3):
            syndrome = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
            word = (1 << anchor) | (1 << a) | (1 << b) | (1 << c)
            old = seen.get(syndrome)
            if old is not None and old != word:
                witness = old ^ word
                assert witness.bit_count() == 8
                return witness
            seen[syndrome] = word
    for a, b, c, d in combinations(range(2 * q), 4):
        syndrome = cols[a] ^ cols[b] ^ cols[c] ^ cols[d]
        old = seen.get(syndrome)
        if old is not None:
            word = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            if old != word:
                witness = old ^ word
                assert witness.bit_count() == 8
                return witness
    return 0


def _s37_probe(rows, width, rng, passes, seed=0, own=None):
    """Actual, nonincreasing kernel/logical witness; no unresolved cap."""
    cols = _s37_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _s37_outside(word, own):
            return
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

        # Elementary basis exchanges preserve the complete kernel.
        rng.shuffle(kernel)
        for i, word in enumerate(kernel):
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        if own is not None:
            logical = sorted(
                (word for word in kernel if _s37_outside(word, own)),
                key=int.bit_count)[:12]
            short = list(dict.fromkeys(short + logical))
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
        if own is not None and weight < 8:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _s37_outside(best, own)
    return best


def _s37_rows(G, grid, side):
    """Zero auxiliary cells isolate the exact evaluator constituent."""
    zero = [[[]]]
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    assert len(rows) == len(grid) * G.q
    assert all(row.bit_length() <= width for row in rows)
    return rows, width


def _s37_draw(rng, G, side, card):
    """Twenty support/quotient/trinomial prescriptions, not saved gauges."""
    profile, policy = card // 10, (card // 2) % 5
    mode = card % 2
    e = G.e

    if side == "A":
        cells = []
        for j in range(2):
            first = G.pick(rng)
            if mode and j == 1:
                a, b = rng.sample(range(7), 2)
                r = (a, 1)
                s = (b, 1 if policy % 2 == 0 else 3)
            else:
                r = (rng.randrange(1, 7), 2 if mode else 0)
                s = (rng.randrange(7), rng.choice((1, 3)))
            assert G.mul(r, s) != G.mul(s, r)
            cell = [first, G.mul(first, r), G.mul(first, s)]
            assert len(set(cell)) == 3
            cells.append(cell)
        return [cells]

    C = [[[] for _ in range(7)] for _ in range(3)]
    voltage = (rng.randrange(1, 7), 2 if policy < 3 else 0)
    if profile == 0:
        # Three matching edges around a triangle. The voltage has
        # order fourteen or seven; no distance is inferred from that.
        C[0][0], C[1][0] = [e], [e]
        C[1][1], C[2][1] = [e], [e]
        C[2][2], C[0][2] = [e], [voltage]
        full = range(3, 7)
    else:
        # Two parallel matchings, a pendant matching, three full
        # cubic columns and one crossed binomial/monomial column.
        C[1][0], C[2][0] = [e], [e]
        C[1][1], C[2][1] = [e], [voltage]
        C[0][2], C[2][2] = [e], [e]
        first = G.pick(rng)
        relative = (rng.randrange(1, 7), 2 if mode else 0)
        C[0][6] = [first, G.mul(first, relative)]
        C[1][6] = [G.pick(rng)]
        full = range(3, 6)

    quotient_cards = (
        ((0, 1), (0, 1), (0, 1), (0, 1)),
        ((1, 1), (1, 1), (1, 1), (1, 1)),
        ((0, 0), (0, 0), (0, 1), (0, 1)),
        ((0, 1), (0, 1), (1, 1), (1, 1)),
        ((1, 0), (0, 1), (0, 1), (0, 1)),
    )
    pools = [
        [g for g in G.elements if g[1] % 2 == parity]
        for parity in (0, 1)]
    for position, j in enumerate(full):
        b, c = quotient_cards[policy][position]
        C[0][j] = [e]
        C[1][j] = [rng.choice(pools[b])]
        C[2][j] = [rng.choice(pools[c])]
    assert all(sum(map(len, row)) == 6 for row in C)
    assert sorted(sum(len(C[i][j]) for i in range(3))
                  for j in range(7)) == [2, 2, 2, 3, 3, 3, 3]
    return C


def _s37_bank(G, side):
    """Independent constituent optimization with equal finalist budgets."""
    blocks = A_SHAPE[1] if side == "A" else B_SHAPE[1]
    bank, fillings = [], set()
    for card in range(20):
        local = random.Random(
            BASE_SEED ^ (0x283710 + 104729 * card
                         + 1000003 * int(side == "B")))
        raw = []
        for trial in range(80 if side == "A" else 120):
            grid = _s37_draw(local, G, side, card)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows, width = _s37_rows(G, grid, side)
            if gf2_rank(rows) != len(rows):
                continue
            cols = _s37_columns(rows, width)
            if not _s37_clean4(cols):
                continue
            penalty = 0
            for i, row in enumerate(rows):
                for other in rows[:i]:
                    overlap = (row & other).bit_count()
                    penalty += overlap * (overlap - 1) // 2
            raw.append((penalty, trial, grid, rows, cols, width))
        raw.sort(key=lambda record: record[:2])

        short = []
        for penalty, trial, grid, rows, cols, width in raw[:8]:
            word = _s37_probe(rows, width, local, 4)
            if word:
                short.append(dict(
                    grid=grid, rows=rows, cols=cols, width=width,
                    word=word, penalty=penalty, card=card, trial=trial))
        short.sort(key=lambda record: (
            -record["word"].bit_count(),
            record["penalty"], record["trial"]))

        for record in short[:3]:
            if not _s37_clean6(record["cols"], G.q, blocks):
                continue
            seed = record["word"]
            clear8 = None
            if side == "A":
                eight = _s37_A8(record["cols"], G.q)
                clear8 = not eight
                if eight and eight.bit_count() < seed.bit_count():
                    seed = eight
            # Every publishable constituent receives four preliminary
            # and the same side-specific deeper search budget.
            word = _s37_probe(
                record["rows"], record["width"], local,
                20 if side == "A" else 28, seed)
            assert word and word.bit_count() >= 7
            if side == "A":
                assert word.bit_count() % 2 == 0
                assert not clear8 or word.bit_count() >= 10
            record["word"] = word
            record["reading"] = word.bit_count()
            record["clear8"] = clear8
            bank.append(record)
    bank.sort(key=lambda record: (
        -record["reading"], record["penalty"],
        record["card"], record["trial"]))
    return bank


def _s37_logical(G, a, b, hx, hz, n, rng):
    """Verify embedded logicals, shorten them, then probe mixed sectors."""
    q = G.q
    overall = n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _s37_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(2 if kind == "X" else 7):
            if kind == "X":
                word = source << (block * 7 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 7 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _s37_outside(word, own):
                continue
            for _ in range(2):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == previous:
                    break
            seeds.append(word)
        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        for _ in range(2):
            starting = seed
            moves = sorted(
                (row for row in stabilizers
                 if (row & starting).bit_count() >= 2),
                key=lambda row: -(row & starting).bit_count())[:64]
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < seed.bit_count():
                        seed = trial
            if seed == starting:
                break
        assert seed and _s37_outside(seed, own)
        word = _s37_probe(checks, n, rng, 3, seed, own)
        if not word:
            return None
        overall = min(overall, word.bit_count())
    return overall


@lru_cache(maxsize=1)
def _s37_portfolio(G, count):
    """Pure deterministic immutable catalogue; no post-screen resampling."""
    assert G.q == 28
    assert A_SHAPE == (1, 2) and B_SHAPE == (3, 7)
    aa, bb = _s37_bank(G, "A"), _s37_bank(G, "B")
    if not aa or not bb:
        raise ValueError("no kernel-six-clean three-row constituent bank")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            priority = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                int(bool(a["clear8"])),
                -a["penalty"] - b["penalty"],
                int(a["card"] != b["card"]))
            plans.append((priority, ai, bi))
    plans.sort(key=lambda plan: (plan[0], -plan[1], -plan[2]),
               reverse=True)

    results, signatures = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        wx = max(sum(row) + sum(sb[i][j] for i in range(3))
                 for row in sa for j in range(7))
        wz = max(sum(row) + sum(sa[i][j] for i in range(1))
                 for row in sb for j in range(2))
        assert wx <= 9 and wz == 9
        assert max(wx, wz) == _grid_weights(sa, sb) <= min(W_MAX, 9)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 476 or not (225 * n <= 1000 * k < 275 * n):
            continue
        signature = _lo_signature(hx, hz, n)
        if signature in signatures:
            continue
        signatures.add(signature)
        tested += 1
        if not _s37_clean4(_s37_columns(hx, n)):
            continue
        if not _s37_clean4(_s37_columns(hz, n)):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
        local = random.Random(
            BASE_SEED ^ (0x2837CC + 65537 * ai + 1009 * bi))
        reading = _s37_logical(G, a, b, hx, hz, n, local)
        if reading is None or reading < 8:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, reading, k, preliminary)
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("A_kernel8_excluded", a["clear8"]),
            ("constituent_kernel6_excluded", True),
            ("complete_kernel4_excluded", True),
            ("design_cards", (a["card"], b["card"])),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + 16:
            break
        if tested >= max(384, 8 * count):
            break
    results.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen) for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one complete, immutable, invariant-separated new-shape design."""
    catalogue = _s37_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("three-row target-band portfolio exhausted")
    A, B, screen = catalogue[idx]
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
# REFUTED: The odd16 full-Tanner-signature portfolio supplies sixty evaluator-distinct designs: the current measured table retained only 33 distinct codes, with numerous iso_dup slots.
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
