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
CONCEPT = ("D17, order 34, A=(1,2), B=(3,7): an order-17 triangle matching core, reinforced by four cubic columns and paired with weight-eight-clean A constituents, should supply five distance-ten readings in cell 0.250. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from S4 x C2 of order 48 to the dihedral group D17 of order 34. Different orders certify a genuine group change. The run's order-34 centre-one/ten-class bed reached fitness 20.8 on (2,3)x(2,4); the present (1,2)x(3,7) bed is untried. Order-17 rotations support a verified 51-edge triangle core, while reflections couple its two components.")
SHAPES = ("Use A=(1,2), B=(3,7): n=578, kb=136, guaranteed rate 4/17=0.235294 in cell 0.250. Components have 68 variables/34 checks and 238 variables/102 checks. Factor seventeen permits q<=35; q=34 fits. Parent minima lay in sector-1 Acol0/1 across Bcols. Three B check layers replace its short three-parallel-matching mechanism; verified full ranks enforce [0.225,0.275).")
FILLING = ("Optimize mixed rotation/reflection A trinomials and twenty B cards combining five triangle voltages with four cubic parity profiles. A row weight is six; B rows have six terms and columns have degrees two or three, so HX<=9 and HZ=9, below both nine and twelve. All screened matrices come from verified lifts. Exact constituent exclusion through six, exact A exclusion through eight, actual B two-cubic T-join witnesses, complete logical-four gates and equally refined explicit logical witnesses select an immutable full-Tanner-invariant-separated portfolio. Larger distance readings and unseen-archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r025_dihedral17_q34_12x37_triangle51_cubic_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("D17 has order 34")
    # x^17=y^2=1, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(17)]

    def red(g):
        return (int(g[0]) % 17, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c)) % 17, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a if b else -a) % 17, b)

    def idx(g):
        a, b = red(g)
        return 17*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def _unused_s4_reference(rng, G, idx):
    """Inactive historical reference; never publishes new-group cells."""
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 1, 1, 1, 0), (0, 1, 1, 1, 3)),
    )
    spectra = ((3, 3, 3), (3, 4, 4),
               (3, 6, 6), (4, 4, 6))
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
        for order in range(1, 13):
            power = G.mul(power, g)
            if power == G.e:
                return order
        raise ValueError("unexpected element order")

    orders = {g: element_order(g) for g in G.elements}
    by_order = {order: [g for g in G.elements if orders[g] == order]
                for order in (3, 4, 6)}
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
    high_order = by_order[4] + by_order[6]
    core_pairs = [
        (g, h) for g in high_order for h in high_order
        if orders[G.mul(G.inv(g), h)] >= 4
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
            order = 4 if (policy + j + idx // 20) % 2 == 0 else 6
            relative = rng.choice(by_order[order])
            return [first, G.mul(first, relative)]

        # Specify all three pairwise relative orders, rather than
        # only two ratios from an arbitrarily chosen anchor.
        spectrum = spectra[apolicy[j]] if side == "A" else (4, 4, 6)
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
        For CSS probes, eight is an unresolved cap: only discovered
        nontrivial witnesses below eight cause rejection.
        """
        cols = columns(rows, width)
        bits = [1 << i for i in range(width)]
        best = 8 if stabilizers is not None else width + 1
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
    mask = (1 << q) - 1
    for batch in range(3):
        pool = []
        for trial in range(64):
            A = [[draw_cell(c, "A", j) for j, c in enumerate(row)]
                 for row in sa]
            B = [[draw_cell(c, "B", j) for j, c in enumerate(row)]
                 for row in sb]
            g, h = rng.choice(core_pairs)
            for j, relative in zip(mono, (G.e, g, h)):
                B[0][j] = [G.e]
                B[1][j] = [relative]

            hx, hz, n = lifted_matrices(G, A, B)

            # Extract BOTH constituents from the VERIFIED evaluator lift.
            ar = []
            for i in range(ra):
                for r in range(q):
                    row = hx[(i * cb) * q + r]
                    ar.append(sum(
                        ((row >> (j * cb * q)) & mask) << (j * q)
                        for j in range(ca)))
            br = [row & ((1 << (cb * q)) - 1)
                  for row in hz[:rb * q]]
            bcols = columns(br, cb * q)
            core = [bcols[j * q + t] for j in mono for t in range(q)]
            girth = core_girth(core, rb * q)
            if girth < 8:
                continue

            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 9 * n <= 40 * k < 11 * n):
                continue
            qa = component_quality(ar, ca * q)
            qb = component_quality(br, cb * q)
            cycle_bias = 12 + 8 * ((idx // 20 + policy) % 4)
            score = (
                qa[0] + qb[0],
                cycle_bias * (qa[1] + qb[1]) + qa[2] + qb[2],
                max(qa[2], qb[2]),
                -k,
            )
            pool.append((score, trial, A, B, hx, hz, n, k,
                         ar, br, girth))

        # With the short-collision gate tied, prefer the parent's
        # demonstrated rank-gain mechanism before cycle tie-breaks.
        pool.sort(key=lambda item:
                  (item[0][0], item[0][3], item[0][1],
                   item[0][2], item[1]))
        survivors = []
        btests = 0
        for candidate in pool[:16]:
            score, trial, A, B, hx, hz, n, k, ar, br, girth = candidate
            if not no_kernel6(ar, ca * q):
                continue
            btests += 1
            if btests > 8:
                break
            if not no_kernel6(br, cb * q):
                continue

            # A columns have odd weight three, so its exact no-kernel6
            # result also excludes weight seven. B has mixed parity:
            # probing for seven is heuristic, not an exact exclusion.
            da = probe(ar, ca * q, 4)
            db = probe(br, cb * q, 12)
            if min(da, db) < 8:
                continue
            if low_logical(hz, hx, n) or low_logical(hx, hz, n):
                continue
            if probe(hz, n, 3, hx) < 8:
                continue
            if probe(hx, n, 3, hz) < 8:
                continue

            quality = (k, da, -score[1], -score[2])
            survivors.append((quality, -trial, candidate, da, db))
            if len(survivors) >= 3:
                break

        if survivors:
            _, _, winner, da, db = max(
                survivors, key=lambda item: (item[0], item[1]))
            _, _, A, B, hx, hz, n, k, _, _, girth = winner
            assert css_ok(hx, hz)
            return {
                "idx": idx,
                "cells": {"A": A, "B": B},
                "screen": {"k": k, "core_girth": girth,
                           "kernel6_A": True, "kernel6_B": True,
                           "component_probe_A": da,
                           "component_probe_B": db,
                           "mixed_probes_found_below8": False},
            }
    raise ValueError("No target-rate girth-eight filling passed mandatory screens")


from functools import lru_cache


def _dg_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _dg_rows(G, grid, side):
    """Expose only the scaffold's actual constituent action."""
    zero = [[[]]]
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _dg_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in basis:
                basis[p] = word
                break
            word ^= basis[p]
    return basis


def _dg_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        if p not in basis:
            return True
        word ^= basis[p]
    return False


def _dg_kernel6(rows, width, q):
    """Exact constituent witness through six, or zero if none exists.

    Regular actions translate any kernel support to contain coordinate
    zero in some block. Store anchored triples rather than every triple.
    This symmetry is used only for constituents, not complete CSS codes.
    """
    from itertools import combinations
    cols = _dg_columns(rows, width)
    bits = [1 << j for j in range(width)]
    seen = {0: 0}

    def add(s, word):
        old = seen.get(s)
        if old is not None:
            return old ^ word
        seen[s] = word
        return 0

    for j, s in enumerate(cols):
        word = add(s, bits[j])
        if word:
            return word
    for a in range(width):
        for b in range(a):
            word = add(cols[a] ^ cols[b], bits[a] | bits[b])
            if word:
                return word
    for anchor in range(0, width, q):
        others = [j for j in range(width) if j != anchor]
        for a, b in combinations(others, 2):
            word = add(cols[anchor] ^ cols[a] ^ cols[b],
                       bits[anchor] | bits[a] | bits[b])
            if word:
                return word
    for a, b, c in combinations(range(width), 3):
        old = seen.get(cols[a] ^ cols[b] ^ cols[c])
        if old is not None:
            word = old ^ (bits[a] | bits[b] | bits[c])
            if word:
                return word
    return 0


def _dg_A8(rows, q):
    """Exact weight-eight witness after exclusion through six."""
    from itertools import combinations
    width = 2*q
    cols = _dg_columns(rows, width)
    bits = [1 << j for j in range(width)]
    seen = {}
    for anchor in (0, q):
        others = [j for j in range(width) if j != anchor]
        for a, b, c in combinations(others, 3):
            s = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
            word = bits[anchor] | bits[a] | bits[b] | bits[c]
            old = seen.get(s)
            if old is None:
                seen[s] = word
            elif old != word:
                return old ^ word
    for a, b, c, d in combinations(range(width), 4):
        old = seen.get(cols[a] ^ cols[b] ^ cols[c] ^ cols[d])
        if old is not None:
            word = old ^ (bits[a] | bits[b] | bits[c] | bits[d])
            if word:
                return word
    return 0


def _dg_logical4(checks, own_rows, width):
    """Exact nontrivial-logical existence test through weight four."""
    own = _dg_basis(own_rows)
    cols = _dg_columns(checks, width)
    seen = {0: 0}
    for a, s in enumerate(cols):
        word = 1 << a
        old = seen.get(s)
        if old is None:
            seen[s] = word
        elif _dg_outside(old ^ word, own):
            return True
    for a, s in enumerate(cols):
        for b in range(a):
            syndrome = s ^ cols[b]
            word = (1 << a) | (1 << b)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _dg_outside(old ^ word, own):
                return True
    return False


def _dg_probe(rows, width, rng, passes, seed=0, own=None):
    """Actual verified kernel/logical witnesses; no unresolved cap."""
    cols = _dg_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _dg_outside(word, own):
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
        for i in range(len(kernel)):
            word = kernel[i]
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        short = sorted(kernel, key=int.bit_count)[:28]
        for i, a in enumerate(short):
            for j in range(i):
                pair = a ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])
        for _ in range(2):
            starting = best
            if not starting:
                break
            for i, a in enumerate(short):
                inspect(starting ^ a)
                for b in short[:i]:
                    inspect(starting ^ a ^ b)
            if best == starting:
                break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _dg_outside(best, own)
    return best


def _dg_signature(rows, width):
    """Conservative invariant of the uncoloured Tanner graph.

    Equal signatures are rejected, not claimed to prove isomorphism.
    No X/Z labels or group-coordinate indices enter the signature.
    """
    from collections import Counter
    from hashlib import sha256
    m = len(rows)
    one, many = [0]*m, [0]*m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            c = (row & rows[j]).bit_count()
            if c:
                overlaps[i, j] = c
                masks = one if c == 1 else many
                masks[i] |= 1 << j
                masks[j] |= 1 << i
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
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    neighborhoods = Counter()
    for col in _dg_columns(rows, width):
        value = []
        while col:
            bit = col & -col
            value.append(colors[bit.bit_length() - 1])
            col ^= bit
        neighborhoods[tuple(sorted(value))] += 1
    value = (width, tuple(sorted(Counter(features).items())),
             tuple(sorted(neighborhoods.items())))
    return sha256(repr(value).encode("ascii")).digest()


def _dg_core(G, voltage):
    """Exact graph paths and shortest-cycle witness from a verified lift."""
    e = G.e
    grid = [[[e], [], [voltage]],
            [[e], [e], []],
            [[], [e], [e]]]
    rows, width = _dg_rows(G, grid, "B")
    cols = _dg_columns(rows, width)
    adjacency = [[] for _ in rows]
    for edge, col in enumerate(cols):
        assert col.bit_count() == 2
        bit = col & -col
        a = bit.bit_length() - 1
        b = (col ^ bit).bit_length() - 1
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    distances, paths, cycle = [], [], 0
    for root in range(len(rows)):
        dist = [10000]*len(rows)
        path = [0]*len(rows)
        dist[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if dist[b] == 10000:
                    dist[b] = dist[a] + 1
                    path[b] = path[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = path[a] ^ path[b] ^ (1 << edge)
                    if word and (not cycle
                                 or word.bit_count() < cycle.bit_count()):
                        cycle = word
        distances.append(tuple(dist))
        paths.append(tuple(path))
    assert cycle and cycle.bit_count() == 51
    assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
    return tuple(cols), tuple(distances), tuple(paths), cycle


def _dg_two_cubic(G, rows, data):
    """Exact two-cubic-family T-join witness, not a distance certificate."""
    q = G.q
    cols = _dg_columns(rows, 7*q)
    core_cols, distances, paths, best = data
    assert tuple(cols[:3*q]) == core_cols
    assert all(col.bit_count() == 3 for col in cols[3*q:])
    memo = {}

    def join(boundary, budget):
        if not boundary:
            return (0, 0)
        if budget < 0 or boundary.bit_count() > 2*budget:
            return None
        key = (boundary, budget)
        if key in memo:
            return memo[key]
        bit = boundary & -boundary
        a = bit.bit_length() - 1
        rest, answer = boundary ^ bit, None
        choices = rest
        while choices:
            other = choices & -choices
            b = other.bit_length() - 1
            distance = distances[a][b]
            if distance <= budget:
                tail = join(rest ^ other, budget - distance)
                if tail is not None:
                    candidate = (distance + tail[0],
                                 paths[a][b] ^ tail[1])
                    if answer is None or candidate[0] < answer[0]:
                        answer = candidate
            choices ^= other
        memo[key] = answer
        return answer

    for anchor in (j*q + G.idx(G.e) for j in range(3, 7)):
        for other in range(3*q, 7*q):
            if other == anchor:
                continue
            answer = join(cols[anchor] ^ cols[other],
                          best.bit_count() - 3)
            if answer is not None:
                word = answer[1] ^ (1 << anchor) ^ (1 << other)
                if word.bit_count() < best.bit_count():
                    best = word
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


@lru_cache(maxsize=1)
def _dg_catalogue(G, count):
    """Pure deterministic portfolio; cached published grids are immutable."""
    assert G.q == 34
    q, e = G.q, G.e
    local = random.Random(BASE_SEED ^ 0x173437)
    aa, raw = [], []

    # Normalize one trinomial; vary genuine relative gaps and reflections.
    # Equal/opposite rotation gaps have an immediate six-word mechanism.
    for t in range(2, 16):
        for phase in range(17):
            A = [[[e, (1, 0), (0, 1)],
                  [e, (t, 0), (phase, 1)]]]
            rows, width = _dg_rows(G, A, "A")
            if gf2_rank(rows) != q or _dg_kernel6(rows, width, q):
                continue
            word = _dg_probe(rows, width, local, 4)
            penalty = sum(
                ((a & b).bit_count()
                 * ((a & b).bit_count() - 1) // 2)
                for i, a in enumerate(rows) for b in rows[:i])
            raw.append((word.bit_count(), penalty, t, phase,
                        A, rows, word))
    raw.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))
    used = set()
    for _, penalty, t, phase, A, rows, seed in raw[:64]:
        eight = _dg_A8(rows, q)
        if eight and eight.bit_count() < seed.bit_count():
            seed = eight
        word = _dg_probe(rows, 2*q, local, 16, seed)
        key = _dg_signature(rows, 2*q)
        if key in used:
            continue
        used.add(key)
        aa.append(dict(grid=A, rows=rows, word=word,
                       reading=word.bit_count(), penalty=penalty,
                       card=(t, phase), clean8=not bool(eight)))
    aa.sort(key=lambda r: (-r["reading"], -int(r["clean8"]),
                           r["penalty"], r["card"]))
    aa = aa[:32]

    # Twenty prescriptions: five relative triangle voltages and four
    # intrinsic parity allocations among the cubic columns.
    voltages = (1, 2, 3, 5, 7)
    parities = (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (0, 1), (1, 0), (1, 0)),
        ((1, 1), (1, 1), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (0, 1), (1, 1)),
    )
    cores = {v: _dg_core(G, (v, 0)) for v in voltages}
    bb, used = [], set()
    for card in range(20):
        v = voltages[card % 5]
        parity = parities[card // 5]
        rng = random.Random(BASE_SEED ^ (0x173700 + 104729*card))
        records = []
        for trial in range(64):
            B = [[[e], [], [(v, 0)]] + [[e] for _ in range(4)],
                 [[e], [e], []] + [[] for _ in range(4)],
                 [[], [e], [e]] + [[] for _ in range(4)]]
            for j, (b, c) in enumerate(parity, 3):
                B[1][j] = [(rng.randrange(17), b)]
                B[2][j] = [(rng.randrange(17), c)]
            rows, width = _dg_rows(G, B, "B")
            if gf2_rank(rows) < 3*q - 2:
                continue
            seed = _dg_two_cubic(G, rows, cores[v])
            if seed.bit_count() < 8:
                continue
            penalty = sum(
                ((a & b).bit_count()
                 * ((a & b).bit_count() - 1) // 2)
                for i, a in enumerate(rows) for b in rows[:i])
            records.append((seed.bit_count(), penalty, trial,
                            B, rows, seed))
        records.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, penalty, trial, B, rows, seed in records[:6]:
            word = _dg_probe(rows, 7*q, rng, 4, seed)
            refined.append((word.bit_count(), penalty, trial,
                            B, rows, word))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))
        kept = 0
        for _, penalty, trial, B, rows, seed in refined[:4]:
            if _dg_kernel6(rows, 7*q, q):
                continue
            word = _dg_probe(rows, 7*q, rng, 16, seed)
            if word.bit_count() < 8:
                continue
            key = _dg_signature(rows, 7*q)
            if key in used:
                continue
            used.add(key)
            bb.append(dict(grid=B, rows=rows, word=word,
                           reading=word.bit_count(), penalty=penalty,
                           card=(card, trial)))
            kept += 1
            if kept == 2:
                break
    bb.sort(key=lambda r: (-r["reading"], r["penalty"], r["card"]))
    if not aa or not bb:
        raise ValueError("no screened D17 constituent banks")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            priority = (min(a["reading"], b["reading"]),
                        a["reading"] + b["reading"],
                        -a["penalty"] - b["penalty"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        blocks = range(2) if kind == "X" else range(7)
        for block in blocks:
            if kind == "X":
                word = record["word"] << (block * 7*q)
            else:
                source, word = record["word"], 0
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*7 + block)*q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _dg_outside(word, own):
                continue
            for _ in range(2):
                starting, weight = word, word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                moves = sorted(
                    (row for row in stabilizers
                     if (row & word).bit_count() >= 2),
                    key=lambda row: -(row & word).bit_count())[:48]
                starting = word
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            assert word and _dg_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    results, signatures, tested = [], set(), 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        wx = max(sum(sa[0]) + sum(sb[i][j] for i in range(3))
                 for j in range(7))
        wz = max(sum(row) + sa[0][j]
                 for row in sb for j in range(2))
        assert wx == wz == 9
        assert max(wx, wz) <= min(W_MAX, 9, 12)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 578 or not (225*n <= 1000*k < 275*n):
            continue
        signature = _dg_signature(hx + hz, n)
        if signature in signatures:
            continue
        tested += 1
        if (_dg_logical4(hz, hx, n)
                or _dg_logical4(hx, hz, n)):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(W_MAX, 9)
        bx, bz = _dg_basis(hx), _dg_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        rng = random.Random(BASE_SEED ^ (0x1737CC + 65537*ai + bi))
        xword = _dg_probe(hz, n, rng, 2, xseed, bx)
        zword = _dg_probe(hx, n, rng, 2, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 8:
            continue
        signatures.add(signature)
        quality = (k*reading*reading/n, reading, k, preliminary)
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("A_kernel_exclusion_through", 8 if a["clean8"] else 6),
            ("B_kernel_exclusion_through", 6),
            ("core_cycle_length", 51),
            ("design_cards", (a["card"], b["card"])),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + 24:
            break
        if tested >= max(480, 12*count):
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen) for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of one independently optimized, screened design."""
    catalogue = _dg_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("D17 target-rate triangle portfolio exhausted")
    A, B, screen = catalogue[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell without resampling or mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: The Dic11 parent's 2-second distance reading always survives the longer search: slot 42 fell from 22 to 20, with mixed-sector Z witnesses.
# REFUTED: Cycle-first selection with preferential boolean constituent screens suffices for five distance-eight readings on this bed: measured slots 16, 18 and 30 instead had d30s=7, predominantly with sector-1 X witnesses.
# REFUTED: The mandatory S4 x C2 girth-eight/spectrum-card gates publish every slot: the measured parent table voided 30 of 60 slots and retained only 18 distinct codes.
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
