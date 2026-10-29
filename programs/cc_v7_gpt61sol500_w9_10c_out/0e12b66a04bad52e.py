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
CONCEPT = ("Faithful C7 semidirect_3 C6, order 42, A=(1,2), B=(2,5): independently optimized cubic A constituents and reinforced maximum-girth B cores should supply five fresh measured finalists exceeding cell 0.250 fitness 20.9252; this is falsifiable, not a distance certificate.")
GROUP = ("Jump to faithful C7:C6: 3 has order six modulo seven, and the group has trivial centre, unlike the order-48 parent's central C8. Related faithful single-row beds measured stronger portfolios; the recorded order-42 asymmetric bed scored 16.2. Exact core and reinforcement optimization tests that weaker bed rather than assuming faithful action guarantees distance.")
SHAPES = ("Keep A=(1,2), B=(2,5): q=42, n=504, kb=126 and guaranteed rate 0.250. Component variables/checks are 84/42 and 210/84; the size factor allows q<=49. Parent weight-eight X minima broadly occupied sector-1 Acol0/1 across Bcols, while its best slot reached nine. Complete verified ranks retain [0.225,0.275).")
FILLING = ("Twenty spectrum/support prescriptions optimize noncommuting A trinomials and split-binomial or isolated-trinomial B reinforcements. A and B rows each contain six terms; columns contain at most three, so BOTH CSS maxima are nine, also below twelve. Verified lifts select exact maximum-girth cores, retain shortest-cycle and two/four-cubic path witnesses, exclude A kernels through six and complete logicals through four, and refine actual nonincreasing kernel/logical witnesses. An immutable portfolio conservatively rejects equal full uncoloured Tanner invariants. Larger readings and unseen-archive novelty remain hypotheses.")
FAMILY_NAME = "r025_faithful_c7sd_c6_q42_12x25_exactcore_witness_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("faithful C7:C6 has order 42")
    # x^7=y^6=1, y*x*y^-1=x^3.
    # 3^3=-1 and 3^6=1 modulo seven: the action is faithful.
    powers = [pow(3, b, 7) for b in range(6)]
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 7, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _unused_inversion_reference(rng, G, idx):
    """Inactive historical reference; never called on the faithful group."""
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 1, 1, 1, 0), (0, 1, 1, 1, 3)),
    )
    spectra = ((3, 16, 16), (6, 16, 16),
               (12, 16, 16), (16, 16, 24))
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    card = idx % 20
    profile, policy = card // 10, card % 10
    apolicy = pairs[policy]
    sa, sb = ((3, 3),), profiles[profile]
    mono = [j for j in range(5) if sb[0][j] == sb[1][j] == 1]
    assert len(mono) == 3
    assert _no_dead_lines(sa, sb)
    # Every B row has six terms, every B column two or three.
    assert _grid_weights(sa, sb) <= min(W_MAX, 9)

    def element_order(g):
        power = G.e
        for order in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return order
        raise ValueError("unexpected element order")

    orders = {g: element_order(g) for g in G.elements}
    by_order = {order: [g for g in G.elements if orders[g] == order]
                for order in (3, 6, 12, 16, 24)}
    triangles = {spectrum: [] for spectrum in spectra}
    nonidentity = [g for g in G.elements if g != G.e]
    for r in nonidentity:
        ri = G.inv(r)
        for s in nonidentity:
            if r == s or G.mul(r, s) == G.mul(s, r):
                continue
            spectrum = tuple(sorted(
                (orders[r], orders[s], orders[G.mul(ri, s)])))
            if spectrum in triangles:
                triangles[spectrum].append((r, s))
    if any(not triangles[spectrum] for spectrum in spectra):
        raise ValueError("required complete trinomial spectrum absent")

    # These are only algebraic proposals. The actual matching-core
    # girth is checked below using columns from the VERIFIED lift.
    high_order = by_order[12] + by_order[16] + by_order[24]
    core_pairs = [
        (g, h) for g in high_order for h in high_order
        if orders[G.mul(G.inv(g), h)] >= 12
        and G.mul(g, h) != G.mul(h, g)
    ]
    if not core_pairs:
        raise ValueError("no proposed high-order matching core")

    def draw_cell(count, side, j=0):
        if count == 0:
            return []
        first = G.pick(rng)
        if count == 1:
            return [first]
        if count == 2:
            order = 12 if (policy + j + idx // 20) % 2 == 0 else 24
            relative = rng.choice(by_order[order])
            return [first, G.mul(first, relative)]

        # Specify all three pairwise relative orders, rather than
        # only two ratios from an arbitrarily chosen anchor.
        spectrum = spectra[apolicy[j]] if side == "A" else (16, 16, 24)
        r, s = rng.choice(triangles[spectrum])
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

    def probe(rows, width, tries, stabilizers=None):
        """Randomized kernel-basis probes of the VERIFIED matrix.

        Component results are weights of actual kernel witnesses.
        For CSS probes, ten is an unresolved cap, not a witness or
        lower bound. Discovered logicals influence pair selection.
        """
        cols = columns(rows, width)
        bits = [1 << i for i in range(width)]
        best = 10 if stabilizers is not None else width + 1
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
            nonlocal best
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
            best = weight

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
            for i, a in enumerate(kernel):
                for b in kernel[:i]:
                    record(a ^ b)
            short = sorted(kernel, key=int.bit_count)[:28]
            for i, a in enumerate(short):
                for j in range(i):
                    pair = a ^ short[j]
                    for h in range(j):
                        record(pair ^ short[h])
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
            da = probe(ar, ca * q, 8)
            # Odd column weight three excludes every odd kernel weight.
            assert da >= 8
            quality = (min(da, 12), -cheap[0], -cheap[2])
            aset.append((quality, A, da, qa))
        aset.sort(key=lambda item: item[0], reverse=True)
        aset = aset[:4]
        if not aset:
            continue

        bproposals = []
        for trial in range(144):
            B = [[draw_cell(c, "B", j) for j, c in enumerate(row)]
                 for row in sb]
            g, h = rng.choice(core_pairs)
            for j, relative in zip(mono, (G.e, g, h)):
                B[0][j] = [G.e]
                B[1][j] = [relative]
            br = component_rows(B, "B")
            bcols = columns(br, cb * q)
            core = [bcols[j * q + t] for j in mono for t in range(q)]
            girth = core_girth(core, rb * q)
            if girth < 8:
                continue
            qb = component_quality(br, cb * q)
            if qb[0]:
                continue
            # The girth helper resolves cycles through eight. A larger
            # return only means none through eight, not exact girth.
            core_cap = 10 if girth > 8 else 8
            cheap = (-core_cap, 20 * qb[1] + qb[2], trial)
            bproposals.append((cheap, B, br, qb, girth, core_cap))
        bproposals.sort(key=lambda item: item[0])

        bset = []
        for cheap, B, br, qb, girth, core_cap in bproposals[:12]:
            if not no_kernel6(br, cb * q):
                continue
            db = probe(br, cb * q, 12)
            assert db >= 7
            reading = min(db, core_cap, 12)
            quality = (int(reading >= 8), reading,
                       -cheap[1], -cheap[2])
            bset.append((quality, B, db, qb, girth, core_cap))
        bset.sort(key=lambda item: item[0], reverse=True)
        bset = bset[:4]
        if not bset:
            continue

        pairs_to_test = []
        for aq, A, da, qa in aset:
            for bq, B, db, qb, girth, core_cap in bset:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k > 0 and 9 * n <= 40 * k < 11 * n):
                    continue
                reading = min(da, db, core_cap, 12)
                quality = (
                    int(reading >= 8),
                    k * reading * reading / float(n),
                    reading,
                    -(qa[1] + qb[1]),
                    aq, bq,
                )
                pairs_to_test.append(
                    (quality, A, B, hx, hz, n, k,
                     da, db, girth, core_cap))
        pairs_to_test.sort(key=lambda item: item[0], reverse=True)

        survivors = []
        for candidate in pairs_to_test:
            (_, A, B, hx, hz, n, k,
             da, db, girth, core_cap) = candidate
            if low_logical(hz, hx, n) or low_logical(hx, hz, n):
                continue
            dx = probe(hz, n, 3, hx)
            dz = probe(hx, n, 3, hz)
            if min(dx, dz) < 7:
                continue
            reading = min(da, db, core_cap, dx, dz, 10)
            quality = (
                int(reading >= 8),
                k * reading * reading / float(n),
                reading, k,
            )
            survivors.append((quality, candidate, dx, dz, reading))
            if len(survivors) >= 3:
                break

        if survivors:
            _, winner, dx, dz, reading = max(
                survivors, key=lambda item: item[0])
            (_, A, B, hx, hz, n, k,
             da, db, girth, core_cap) = winner
            assert n == 576 and css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
            return {
                "idx": idx,
                "cells": {"A": A, "B": B},
                "screen": {
                    "k": k,
                    "core_cycle_search_return": girth,
                    "core_no_cycle_through8": girth > 8,
                    "kernel6_A": True, "kernel6_B": True,
                    "component_probe_A": da,
                    "component_probe_B": db,
                    "logical_probe_X_cap10": dx,
                    "logical_probe_Z_cap10": dz,
                    "selection_reading": reading,
                    "weight7_fallback": reading < 8,
                },
            }
    raise ValueError("No independent long-order pair passed target-rate CSS screens")


from functools import lru_cache


def _f42_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _f42_rows(G, grid, side):
    """Expose only the scaffold's actual constituent action."""
    zero = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    width = len(grid[0]) * G.q
    assert len(rows) == len(grid) * G.q
    assert all(row.bit_length() <= width for row in rows)
    return rows, width


def _f42_span(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in basis:
                basis[p] = word
                break
            word ^= basis[p]
    return basis


def _f42_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        if p not in basis:
            return True
        word ^= basis[p]
    return False


def _f42_clean(rows, width, through6=False, own=None):
    """Exact kernel/logical exclusion through four, optionally kernel six."""
    cols = _f42_columns(rows, width)
    seen = {0: 0}

    def add(syndrome, word):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
            return True
        difference = previous ^ word
        return not difference or (
            own is not None and not _f42_outside(difference, own))

    for i, syndrome in enumerate(cols):
        if not add(syndrome, 1 << i):
            return False
    for i, syndrome in enumerate(cols):
        for j in range(i):
            if not add(syndrome ^ cols[j], (1 << i) | (1 << j)):
                return False
    if through6:
        assert own is None
        for i, syndrome in enumerate(cols):
            for j in range(i):
                pair = syndrome ^ cols[j]
                for h in range(j):
                    word = (1 << i) | (1 << j) | (1 << h)
                    if not add(pair ^ cols[h], word):
                        return False
    return True


def _f42_probe(rows, width, rng, passes, seed=0, own=None):
    """Return a verified witness, never an unresolved distance cap."""
    cols = _f42_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _f42_outside(word, own):
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

        rng.shuffle(kernel)
        for i, word in enumerate(kernel):
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if trial and trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        for i, word in enumerate(kernel):
            for other in kernel[:i]:
                inspect(word ^ other)
        short = sorted(kernel, key=int.bit_count)[:28]
        for i, word in enumerate(short):
            for j in range(i):
                pair = word ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        for _ in range(2):
            if not best:
                break
            starting = best
            for i, word in enumerate(short):
                inspect(starting ^ word)
                for other in short[:i]:
                    inspect(starting ^ word ^ other)
            if starting == best:
                break
        if weight < 8:
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _f42_outside(best, own)
    return best


def _f42_core(G, g, h):
    """Exact shortest cycle, distances and path supports from a verified lift."""
    C = [[[G.e], [G.e], [G.e]], [[G.e], [g], [h]]]
    rows, width = _f42_rows(G, C, "B")
    cols = _f42_columns(rows, width)
    adjacency = [[] for _ in rows]
    endpoints = set()
    for edge, syndrome in enumerate(cols):
        if syndrome.bit_count() != 2:
            return None
        bit = syndrome & -syndrome
        a = bit.bit_length() - 1
        b = (syndrome ^ bit).bit_length() - 1
        key = (min(a, b), max(a, b))
        if key in endpoints:
            return None
        endpoints.add(key)
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))

    distances, paths, cycle = [], [], 0
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
        if len(queue) != len(rows):
            return None
        distances.append(tuple(distance))
        paths.append(tuple(path))
        if cycle and cycle.bit_count() < 8:
            return None
    if not cycle:
        return None
    assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
    return (g, h, tuple(cols), tuple(distances), tuple(paths), cycle)


def _f42_join(rows, width, data, G, four=False):
    """Actual B witnesses from zero/two/four cubic variables.

    Path matching is exhaustive within the requested budget and family.
    Six or more cubic variables and other quantum families remain unscreened.
    """
    from itertools import combinations
    q = G.q
    cols = _f42_columns(rows, width)
    assert tuple(cols[:3*q]) == data[2]
    cubic = list(range(3*q, width))
    assert all(cols[j].bit_count() == 3 for j in cubic)
    anchors = (3*q + G.idx(G.e), 4*q + G.idx(G.e))
    distances, paths = data[3:5]
    best = data[5]
    memo = {}

    def join(boundary, budget):
        if not boundary:
            return (0, 0)
        if budget < 0 or boundary.bit_count() > 2 * budget:
            return None
        key = (boundary, budget)
        if key in memo:
            return memo[key]
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
                    result = (distance + tail[0], paths[a][b] ^ tail[1])
                    if answer is None or result[0] < answer[0]:
                        answer = result
            choices ^= other
        memo[key] = answer
        return answer

    for anchor in anchors:
        for other in cubic:
            if anchor == other:
                continue
            result = join(cols[anchor] ^ cols[other],
                          best.bit_count() - 3)
            if result is not None:
                word = result[1] ^ (1 << anchor) ^ (1 << other)
                if word.bit_count() < best.bit_count():
                    best = word
                    if best.bit_count() < 8:
                        return best

    if four and best.bit_count() >= 9:
        for anchor in anchors:
            others = [j for j in cubic if j != anchor]
            for a, b, c in combinations(others, 3):
                boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                budget = best.bit_count() - 5
                if boundary.bit_count() > 2 * budget:
                    continue
                result = join(boundary, budget)
                if result is not None:
                    word = result[1] ^ (
                        (1 << anchor) | (1 << a) | (1 << b) | (1 << c))
                    if word.bit_count() < best.bit_count():
                        best = word
                        if best.bit_count() < 9:
                            return best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _f42_key(rows, width):
    """Conservative full uncoloured Tanner invariant; equality is rejected.

    Published checks have degree nine, qubits degree five/six/twelve,
    so their bipartite vertex classes are intrinsically distinguishable.
    """
    from collections import Counter
    from hashlib import sha256
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            count = (row & rows[j]).bit_count()
            overlaps[i, j] = count
            if count:
                adjacency = one if count == 1 else many
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i in range(m):
        for j in range(i):
            feature = (
                overlaps[i, j],
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count())
            profiles[i][(weights[j],) + feature] += 1
            profiles[j][(weights[i],) + feature] += 1
    features = [
        (weights[i], tuple(sorted(profiles[i].items())))
        for i in range(m)]
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    incidence = []
    for col in _f42_columns(rows, width):
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(checks)
    history = [tuple(sorted(Counter(features).items()))]
    for _ in range(2):
        qfeatures = [tuple(sorted(colors[r] for r in checks))
                     for checks in incidence]
        qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        check_profiles = [[] for _ in rows]
        for checks, feature in zip(incidence, qfeatures):
            for r in checks:
                check_profiles[r].append(qp[feature])
        refined = [(colors[r], tuple(sorted(check_profiles[r])))
                   for r in range(m)]
        history.append((tuple(sorted(Counter(qfeatures).items())),
                        tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    value = (width, m,
             tuple(sorted(Counter(map(len, incidence)).items())),
             tuple(history))
    return sha256(repr(value).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _f42_portfolio(G, count):
    """Deterministic immutable portfolio; cache changes cost, not results."""
    from itertools import combinations
    q, e = G.q, G.e
    assert q == 42 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    assert min(W_MAX, 9, 12) >= 9

    def order(g):
        power = e
        for length in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return length
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != e]
    high = [g for g in nonzero if orders[g] >= 4]
    representatives = [
        g for g in high
        if G.idx(g) == min(
            G.idx(G.mul(G.mul(a, g), G.inv(a))) for a in G.elements)]
    cores = []
    for g in representatives:
        for h in high:
            if g == h or orders[G.mul(G.inv(g), h)] < 4:
                continue
            data = _f42_core(G, g, h)
            if data is not None:
                cores.append(data)
    if not cores:
        raise ValueError("no connected verified core of girth at least eight")
    maximum = max(data[5].bit_count() for data in cores)
    cores = [data for data in cores if data[5].bit_count() == maximum]

    requested = ((3, 6, 6), (6, 6, 7), (3, 3, 7), (2, 6, 6))
    triangles = {s: [] for s in requested}
    broad = []
    for r, s in combinations(nonzero, 2):
        if G.mul(r, s) == G.mul(s, r):
            continue
        spectrum = tuple(sorted((
            orders[r], orders[s], orders[G.mul(G.inv(r), s)])))
        if spectrum in triangles:
            triangles[spectrum].append((r, s))
        if spectrum[0] >= 3:
            broad.append((r, s))
    assert broad
    banks = [triangles[s] or broad for s in requested]
    apairs = [(a, b) for a in range(4) for b in range(a, 4)]
    reinforcement = ((6, 6), (6, 7), (7, 7), (3, 6), (7, 3))
    ratios = {degree: [g for g in nonzero if orders[g] == degree]
              for degree in (3, 6, 7)}

    def draw_A(local, card):
        a, b = apairs[card % 10]
        if card >= 10:
            b = (b + 1) % 4
        return [[[e, *local.choice(banks[kind])] for kind in (a, b)]]

    def draw_B(local, card):
        data = local.choice(cores)
        g, h = data[:2]
        B = [[[e], [e], [e], [], []],
             [[e], [g], [h], [], []]]
        if card < 10:
            d0, d1 = reinforcement[card % 5]
            a, b = G.pick(local), G.pick(local)
            B[0][3] = [a, G.mul(a, local.choice(ratios[d0]))]
            B[1][3] = [e]
            B[0][4] = [e]
            B[1][4] = [b, G.mul(b, local.choice(ratios[d1]))]
        else:
            r, s = local.choice(banks[card % 4])
            u, v = local.choice(banks[(card + 1) % 4])
            B[0][3], B[1][4] = [e, r, s], [e, u, v]
        return B, data

    def cycle_cost(rows):
        return sum(
            (shared * (shared - 1) // 2)
            for i, row in enumerate(rows)
            for other in rows[:i]
            for shared in ((row & other).bit_count(),))

    components = {"A": [], "B": []}
    for side in ("A", "B"):
        seen_fillings = set()
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0xF42625 + 104729*card
                             + 1000003*int(side == "B")))
            raw = []
            for trial in range(64 if side == "A" else 96):
                if side == "A":
                    grid, data = draw_A(local, card), None
                else:
                    grid, data = draw_B(local, card)
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for row in grid for cell in row)
                if filling in seen_fillings:
                    continue
                seen_fillings.add(filling)
                rows, width = _f42_rows(G, grid, side)
                if not _f42_clean(rows, width):
                    continue
                seed = 0
                if side == "B":
                    seed = _f42_join(rows, width, data, G)
                    if seed.bit_count() < 8:
                        continue
                cost = cycle_cost(rows)
                priority = (-(seed.bit_count() if seed else 14),
                            cost, trial)
                raw.append((priority, grid, rows, width, seed, data))
            raw.sort(key=lambda record: record[0])
            refined = []
            for priority, grid, rows, width, seed, data in raw[:8]:
                if side == "A" and not _f42_clean(rows, width, True):
                    continue
                if side == "B":
                    seed = _f42_join(rows, width, data, G, four=True)
                word = _f42_probe(rows, width, local, 4, seed)
                if word and word.bit_count() >= 8:
                    refined.append((
                        word.bit_count(), priority[1], priority[2],
                        grid, rows, width, word))
            refined.sort(key=lambda r: (-r[0], r[1], r[2]))
            for _, cost, trial, grid, rows, width, seed in refined[:2]:
                word = _f42_probe(rows, width, local, 20, seed)
                if word and word.bit_count() >= 8:
                    components[side].append({
                        "grid": grid, "word": word,
                        "reading": word.bit_count(), "cost": cost,
                        "card": card, "trial": trial})
    aa, bb = components["A"], components["B"]
    if len(aa) < 3 or len(bb) < 3:
        raise ValueError("too few faithful witness-screened constituents")

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"],
         a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(2 if kind == "X" else 5):
            if kind == "X":
                word = record["word"] << (block * 5*q)
            else:
                word, source = 0, record["word"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _f42_outside(word, own):
                continue
            for _ in range(2):
                starting, weight = word, word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:64]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and _f42_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, keys, tested = [], set(), 0
    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(sa[0]) + sum(sb[r][j] for r in range(2))
                 for j in range(5))
        wz = max(sum(row) + sa[0][j] for row in sb for j in range(2))
        assert wx == wz == 9 <= min(W_MAX, 9, 12)
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 504 or not (225*n <= 1000*k < 275*n):
            continue
        assert k >= 126
        bx, bz = _f42_span(hx), _f42_span(hz)
        if not _f42_clean(hz, n, own=bx):
            continue
        if not _f42_clean(hx, n, own=bz):
            continue
        signature = _f42_key(hx + hz, n)
        if signature in keys:
            continue
        keys.add(signature)
        tested += 1
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0xF42CC0 + 65537*ai + bi))
        xword = _f42_probe(hz, n, local, 3, xseed, bx)
        zword = _f42_probe(hx, n, local, 3, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 8:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 9

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, reading, k,
                   -a["cost"] - b["cost"])
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("design_cards", (a["card"], b["card"])),
            ("exact_core_girth", maximum),
            ("complete_logical_exclusion_through", 4),
            ("tanner_signature", signature.hex()))
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + 12:
            break
        if tested >= max(256, 6*count):
            break
    results.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen) for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup; no mutable reservations or post-screen resampling."""
    portfolio = _f42_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("faithful exact-core portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"cells": {"A": A, "B": B}, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell from the immutable selected design."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: The Dic11 parent's 2-second distance reading always survives the longer search: slot 42 fell from 22 to 20, with mixed-sector Z witnesses.
# REFUTED: Cycle-first selection with preferential boolean constituent screens suffices for five distance-eight readings on this bed: measured slots 16, 18 and 30 instead had d30s=7, predominantly with sector-1 X witnesses.
# REFUTED: All twenty S4 x C2 spectrum/core cards are publishable under the mandatory gates: the current parent voided 30 of 60 slots, retaining only 18 distinct codes.
# REFUTED: Independent long-order inversion-group screens supply five measured readings >=9: four current finalists measured eight, and only slot 16 measured nine.
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
