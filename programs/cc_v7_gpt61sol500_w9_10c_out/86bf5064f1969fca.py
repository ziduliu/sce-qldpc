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
CONCEPT = ("Affine C5:C4 with action 2, order 20, A=(1,4), B=(1,4): complete-incidence single-root and two-root refinement should uncover at least five archive-new certified-distance-three codes hidden by the parent's coarser walk catalogue.")
GROUP = ("Keep the frozen affine presentation and order byte-for-byte. This bed measured five k=180,d30s=3 finalists, but retained only seven evaluator-distinct codes. Order twenty gives legal n=340; the new hypothesis concerns structural catalogue yield, not a group-distance advantage.")
SHAPES = ("Keep equal 80-variable, 20-check constituents: n=340, kb=180 and guaranteed rate 9/17=0.529412 in cell 0.550. The shape permits q<=35, while q=20 is frozen. Parent weight-three minima occupied sector-1 Acol/Bcol blocks broadly, particularly pivot columns. Under cap nine, avoiding two monomial columns on either side forces [1,2,2,2] on both sides; unit pivots make k=180 exact. Certified distance-three codes therefore score 81/17, with no strict score increase claimed.")
FILLING = ("Exhaustively enumerate inverse-distinct order-four/five direction triples, quotient known common affine automorphisms and side swaps, and exclude all seven reported parent representatives. Do not reject a candidate merely because its old walk signature repeats. Refine the complete uncoloured check-qubit incidence graph after individualizing every intrinsic degree-fourteen qubit and every unordered pair of those qubits. Unequal signatures certify non-isomorphism; equal signatures remain inconclusive and are discarded conservatively. Immutable portfolio entries pass verified ranks, CSS, complete kernel exclusion through weight two and an actual weight-three logical certificate. Archive-new yield remains evaluator-tested.")
FAMILY_NAME = "r055_affine5by4_14x14_pivot_binomial_fullincidence_pairroot_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5:C4 with action 2 has order 20")
    # x^5=y^4=1, y*x*y^-1=x^2; (a,b) represents x^a y^b.
    powers = (1, 2, 4, 3)
    els = [(a, b) for b in range(4) for a in range(5)]

    def red(g):
        return (g[0] % 5, g[1] % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 5, (b + d) % 4)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        return ((-powers[bi] * a) % 5, bi)

    def idx(g):
        a, b = red(g)
        return a + 5 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _sd_columns(rows, n):
    """Column syndromes of a supplied, verified binary matrix."""
    columns = [0] * n
    for i, row in enumerate(rows):
        bit = 1 << i
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low
    return columns


def _sd_pairs(columns):
    """None means a classical dependency of weight at most four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = set()
    for i, x in enumerate(columns):
        for y in columns[i + 1:]:
            syndrome = x ^ y
            if syndrome in singles or syndrome in pairs:
                return None
            pairs.add(syndrome)
    return pairs


def _sd_fives(columns, pairs, cap=80):
    """Count triple/pair matches, capped; each weight-five word counts ten."""
    count = 0
    for i in range(len(columns) - 2):
        for j in range(i + 1, len(columns) - 1):
            syndrome = columns[i] ^ columns[j]
            for z in columns[j + 1:]:
                if syndrome ^ z in pairs:
                    count += 1
                    if count >= cap:
                        return cap
    return count


def _sd_basis(rows):
    basis = {}
    for value in rows:
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return basis


def _sd_in_span(value, basis):
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def _sd_no_short_logical(checks, stabilizers, n):
    """Exact rejection of nontrivial logicals of weights one through four.

    Compare syndromes of all supports of weight <=2. A collision gives
    a commuting support of weight <=4; span membership distinguishes
    stabilizers from logicals.
    """
    columns = _sd_columns(checks, n)
    basis = _sd_basis(stabilizers)
    seen = {0: 0}

    def accept(syndrome, support):
        if syndrome in seen:
            return _sd_in_span(support ^ seen[syndrome], basis)
        seen[syndrome] = support
        return True

    for i, syndrome in enumerate(columns):
        if not accept(syndrome, 1 << i):
            return False
    for i, x in enumerate(columns):
        bit = 1 << i
        for j in range(i + 1, n):
            if not accept(x ^ columns[j], bit | (1 << j)):
                return False
    return True


def _sd_components(hx, hz, q):
    """Extract components from verified lifts, not a separate lift model."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    arows = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            arows.append(row)
    bmask = (1 << (cb * q)) - 1
    brows = [hz[a * q + t] & bmask
             for a in range(rb) for t in range(q)]
    return (_sd_columns(arows, ca * q),
            _sd_columns(brows, cb * q))


def _af_clean6(columns, pairs):
    """Exact constituent exclusion through six, given a clean <=4 test."""
    singles = set(columns)
    triples = set()
    for i, first in enumerate(columns):
        for j in range(i):
            pair = first ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if (syndrome == 0 or syndrome in singles
                        or syndrome in pairs or syndrome in triples):
                    return False
                triples.add(syndrome)
    return True


def _af_probe(columns, local, rounds, own=None):
    """Search actual kernel witnesses; twelve is an unresolved cutoff."""
    width = len(columns)
    best = 12
    order = list(range(width))

    def inspect(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if own is not None and _sd_in_span(word, own):
            return
        best = word.bit_count()

    for _ in range(rounds):
        local.shuffle(order)
        pivots, nulls = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, word)
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome:
                nulls.append(word)
                inspect(word)
        for i, word in enumerate(nulls):
            for other in nulls[:i]:
                inspect(word ^ other)
        short = sorted(nulls, key=int.bit_count)[:20]
        for i, word in enumerate(short):
            for j in range(i):
                pair = word ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if best <= 6:
            break
    return best


def _af_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _af_component_rows(G, side, grid):
    """Obtain the constituent from the verified left/right construction."""
    dummy = [[[G.e]]]
    mask = (1 << (5 * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz]


def _af_bank(G, side):
    """Optimize six structural prescriptions, with no cross-slot state."""
    from itertools import combinations_with_replacement

    local = random.Random(BASE_SEED ^ (0xAF520 if side == "A" else 0xAF521))
    spectra = ((2, 2, 5), (2, 4, 4), (4, 4, 5))
    policies = list(combinations_with_replacement(range(3), 2))

    def order(g):
        power = G.e
        for length in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return length
        raise ValueError("broken group order")

    orders = {g: order(g) for g in G.elements}
    nonidentity = [g for g in G.elements if g != G.e]
    triangles = {s: [] for s in spectra}
    cores = []
    for g in nonidentity:
        for h in nonidentity:
            if g == h or G.mul(g, h) == G.mul(h, g):
                continue
            relative = G.mul(G.inv(g), h)
            spectrum = tuple(sorted((orders[g], orders[h], orders[relative])))
            if spectrum in triangles:
                triangles[spectrum].append((g, h))
            if orders[g] == orders[h] == 4 and orders[relative] == 5:
                cores.append((g, h))
    if not cores or any(not triangles[s] for s in spectra):
        raise ValueError("required affine core or triangle spectrum absent")

    buckets = [[] for _ in policies]
    fingerprints = [set() for _ in policies]
    for attempt in range(192):
        policy = attempt % len(policies)
        s0, s1 = policies[policy]
        g, h = local.choice(cores)
        grid = [
            [[G.e], [G.e], [G.e], [], []],
            [[G.e], [g], [h], [], []],
        ]
        for row, col, spectrum_id in ((0, 3, s0), (1, 4, s1)):
            r, s = local.choice(triangles[spectra[spectrum_id]])
            anchor = G.pick(local)
            cell = [anchor, G.mul(anchor, r), G.mul(anchor, s)]
            if len(set(cell)) != 3:
                raise ValueError("triangle contains duplicate elements")
            grid[row][col] = cell

        fingerprint = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
        if fingerprint in fingerprints[policy]:
            continue
        fingerprints[policy].add(fingerprint)
        rows = _af_component_rows(G, side, grid)
        columns = _sd_columns(rows, 5 * G.q)
        pairs = _sd_pairs(columns)
        if pairs is None:
            continue
        clean6 = _af_clean6(columns, pairs)
        reading = _af_probe(columns, local, 3)
        # A failed exact-six test exposes a word <=6, even if probing
        # misses it. Never let that candidate advertise a higher cutoff.
        if not clean6:
            reading = min(reading, 6)
        reading = min(reading, 8)
        cycles = _af_cycles(rows)
        key = (reading, clean6, -cycles)
        buckets[policy].append((key, attempt, grid, policy))
        buckets[policy].sort(key=lambda item: (item[0], -item[1]),
                             reverse=True)
        del buckets[policy][3:]

    bank = [entry for bucket in buckets for entry in bucket]
    if not bank:
        raise ValueError("no affine constituent passed exact weight-four screen")
    return bank


def _af_signature(hx, hz, n):
    """Invariant of the full uncoloured Tanner graph.

    Unequal signatures certify non-isomorphism. Equal signatures are
    rejected conservatively; they are not an isomorphism certificate.
    """
    rows = list(hx) + list(hz)
    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlaps.append((i, j, value))
            if value:
                relation = one if value == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [{} for _ in rows]
    for i, j, value in overlaps:
        key = (value,
               rows[j].bit_count(),
               (one[i] & one[j]).bit_count(),
               (one[i] & many[j]).bit_count()
               + (many[i] & one[j]).bit_count(),
               (many[i] & many[j]).bit_count())
        profiles[i][key] = profiles[i].get(key, 0) + 1
        reverse_key = (value, rows[i].bit_count()) + key[2:]
        profiles[j][reverse_key] = profiles[j].get(reverse_key, 0) + 1

    columns = _sd_columns(rows, n)
    degrees = tuple(sorted(column.bit_count() for column in columns))
    roots = tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profile.items())))
        for i, profile in enumerate(profiles)))
    return degrees, roots


def _hp_walk_signature(hx, hz, n, q):
    """Strengthen the existing full, uncoloured Tanner invariant.

    Degree-fourteen qubits form an intrinsic class. Project checks onto
    that class, retaining edge multiplicities. Rooted walk profiles are
    invariant under arbitrary Tanner relabelings, including X/Z exchange.
    Equality is inconclusive and is conservatively discarded.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    columns = _sd_columns(rows, n)
    high = [j for j, col in enumerate(columns)
            if col.bit_count() == 14]
    if len(high) != q:
        raise ValueError("identity-pivot lift lost its intrinsic degree class")
    position = {j: i for i, j in enumerate(high)}
    high_mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]
    anchors = [Counter() for _ in high]

    for row in rows:
        bits = row & high_mask
        endpoints = []
        while bits:
            bit = bits & -bits
            endpoints.append(position[bit.bit_length() - 1])
            bits ^= bit
        if len(endpoints) == 1:
            anchors[endpoints[0]][row.bit_count()] += 1
        elif len(endpoints) == 2:
            u, v = endpoints
            adjacency[u][v] += 1
            adjacency[v][u] += 1
        else:
            raise ValueError("check does not meet one/two pivot-sector qubits")

    colors = [
        (sum(adjacency[v].values()), tuple(sorted(anchors[v].items())))
        for v in range(q)
    ]
    profiles = []
    for root in range(q):
        vector = [0] * q
        vector[root] = 1
        diagonal = []
        features = [[] for _ in range(q)]
        for length in range(1, 11):
            following = [0] * q
            for u, count in enumerate(vector):
                if count:
                    for v, multiplicity in adjacency[u].items():
                        following[v] += count * multiplicity
            vector = following
            diagonal.append(vector[root])
            for v, count in enumerate(vector):
                features[v].append(count)
        profiles.append((
            colors[root], tuple(diagonal),
            tuple(sorted((colors[v], tuple(features[v]))
                         for v in range(q))),
        ))
    return _af_signature(hx, hz, n), tuple(sorted(profiles))


def _hp_three_logical(hx, hz, n, q):
    """Construct and verify a weight-three X logical in Acol1.

    B's identity cell supplies one column for every singleton syndrome.
    A B-binomial column has two such syndromes. Its cancellation word
    commutes with HZ; span membership explicitly verifies nontriviality.
    """
    columns = _sd_columns(hz, n)
    base = 4 * q                  # Acol1/Bcol0
    pivots = {columns[base + h]: base + h for h in range(q)}
    syndrome = columns[base + q]  # Acol1/Bcol1, group index zero
    word = 1 << (base + q)
    while syndrome:
        bit = syndrome & -syndrome
        if bit not in pivots:
            raise ValueError("binomial syndrome is not canceled by B pivot")
        word ^= 1 << pivots[bit]
        syndrome ^= bit
    if word.bit_count() != 3:
        raise ValueError("non-involutory binomial did not give three terms")
    if any((row & word).bit_count() & 1 for row in hz):
        raise ValueError("constructed three-word does not commute")
    if _sd_in_span(word, _sd_basis(hx)):
        raise ValueError("constructed three-word is a stabilizer")
    return word


from functools import lru_cache


@lru_cache(maxsize=1)
def _hp_pairroot_portfolio(G, count):
    """Deterministic immutable portfolio, independent of slot call order.

    Memoization changes computation cost, not the selected fillings.
    All matrix screening uses the scaffold's verified construction.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations, combinations_with_replacement

    if G.q != 20:
        raise ValueError("the frozen affine bed requires order twenty")
    support = [[1, 2, 2, 2]]
    if not _no_dead_lines(support, support):
        raise ValueError("pivot/binomial support has a dead line")
    if _grid_weights(support, support) > min(W_MAX, 9):
        raise ValueError("both CSS weight sums require cap nine")

    def element_order(g):
        value = G.e
        for length in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return length
        raise ValueError("invalid finite-group element order")

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    labels = sorted({
        inverse_class(g) for g in G.elements
        if element_order(g) in (4, 5)
    })
    cards = list(combinations(labels, 3))
    orders = {t: element_order(G.elements[t]) for t in labels}

    def grid(card):
        return [[[G.e]] +
                [[G.e, G.elements[t]] for t in card]]

    def freeze(cells):
        return tuple(tuple(tuple(cell) for cell in row) for row in cells)

    def full_key(hx, hz, n):
        """Intrinsic signature of the COMPLETE Tanner incidence graph.

        Check/qubit type and degree are intrinsic allowed labels; no CSS
        type, protograph coordinate or group label enters refinement.
        Root sets range over an intrinsically identified qubit class.
        Unequal signatures prove non-isomorphism. Equality proves nothing.
        """
        rows = list(hx) + list(hz)
        columns = _sd_columns(rows, n)
        m = len(rows)
        adjacency = [[] for _ in range(m + n)]
        for j, column in enumerate(columns):
            vertex = m + j
            while column:
                bit = column & -column
                check = bit.bit_length() - 1
                adjacency[check].append(vertex)
                adjacency[vertex].append(check)
                column ^= bit

        base = ([(0, row.bit_count()) for row in rows]
                + [(1, column.bit_count()) for column in columns])
        high = [m + j for j, column in enumerate(columns)
                if column.bit_count() == 14]
        if len(high) != G.q:
            raise ValueError("intrinsic pivot-sector degree class changed")

        def refine(roots):
            roots = set(roots)
            initial = [(kind, degree, int(v in roots))
                       for v, (kind, degree) in enumerate(base)]
            palette = {feature: c for c, feature in
                       enumerate(sorted(set(initial)))}
            colours = [palette[feature] for feature in initial]
            history = []
            for _ in range(12):
                features = [
                    (colours[v],
                     tuple(sorted(colours[t] for t in neighbours)))
                    for v, neighbours in enumerate(adjacency)
                ]
                description = tuple(sorted(Counter(features).items()))
                history.append(sha256(
                    repr(description).encode("ascii")).digest())
                palette = {feature: c for c, feature in
                           enumerate(sorted(set(features)))}
                following = [palette[feature] for feature in features]
                stable = len(set(following)) == len(set(colours))
                colours = following
                if stable:
                    break
            return tuple(history)

        singles = tuple(sorted(refine((root,)) for root in high))
        pairs = tuple(sorted(refine(pair)
                             for pair in combinations(high, 2)))
        return (_hp_walk_signature(hx, hz, n, G.q), singles, pairs)

    # All seven evaluator-distinct representatives in the current table.
    # Each packet gives three A directions and three B directions in
    # the FROZEN element ordering. These are exclusion references only.
    archived = (
        (5, 1, 6, 5, 1, 2),       # slot 0
        (5, 6, 16, 5, 1, 6),      # slot 2
        (5, 6, 16, 5, 1, 2),      # slot 4
        (5, 1, 6, 5, 1, 16),      # slot 5
        (5, 1, 6, 5, 1, 18),      # slot 6
        (5, 1, 2, 1, 6, 2),       # slot 8
        (5, 6, 16, 5, 6, 7),      # slot 9
    )
    blocked = set()
    for packet in archived:
        hx, hz, n = lifted_matrices(
            G, grid(packet[:3]), grid(packet[3:]))
        blocked.add(full_key(hx, hz, n))

    # All affine-group automorphisms:
    # x -> x^u, y -> x^v*y; u in F5*, v in F5.
    sums = (0, 1, 3, 2)
    actions = [(u, v) for u in range(1, 5) for v in range(5)]
    images = []
    for card in cards:
        transformed = []
        for u, v in actions:
            transformed.append(tuple(sorted(
                inverse_class(((u * G.elements[t][0]
                                + v * sums[G.elements[t][1]]) % 5,
                               G.elements[t][1]))
                for t in card)))
        images.append(tuple(transformed))

    def orbit_key(a, b):
        return min(tuple(sorted((left, right)))
                   for left, right in zip(images[a], images[b]))

    plans = list(combinations_with_replacement(range(len(cards)), 2))
    plans.sort(key=lambda pair: (
        cards[pair[0]] == cards[pair[1]],
        tuple(sorted(orders[t] for t in cards[pair[0]]))
        == tuple(sorted(orders[t] for t in cards[pair[1]])),
        pair,
    ))

    orbits, entries = set(), []
    for a, b in plans:
        orbit = orbit_key(a, b)
        if orbit in orbits:
            continue
        orbits.add(orbit)

        A, B = grid(cards[a]), grid(cards[b])
        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if n != 340 or rx != 80 or rz != 80 or k != 180:
            raise ValueError("verified identity-pivot rank calculation failed")
        if max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            raise ValueError("built matrices exceed the strict cap")

        clean = True
        for checks in (hx, hz):
            columns = _sd_columns(checks, n)
            if 0 in columns or len(set(columns)) != n:
                clean = False
                break
        if not clean:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified affine lift failed CSS")
        witness = _hp_three_logical(hx, hz, n, G.q)

        # Crucially, a repeated OLD walk signature is not a rejection.
        # Only the stronger, uniformly computed full-incidence key gates
        # archive references and within-program invariant separation.
        signature = full_key(hx, hz, n)
        if signature in blocked:
            continue
        blocked.add(signature)
        entries.append((
            freeze(A), freeze(B), witness, (cards[a], cards[b])))
        if len(entries) >= count:
            break

    # Never pad an exhausted catalogue with archived or orbit copies.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed lookup in the jointly screened immutable portfolio."""
    entries = _hp_pairroot_portfolio(G, N_SLOTS)
    if not 0 <= idx < len(entries):
        raise ValueError("archive-new full-incidence affine catalogue exhausted")
    A, B, witness, card = entries[idx]
    return {
        "cells": {"A": A, "B": B},
        "idx": idx, "k": 180, "card": card,
        "exact_distance": 3,
        "verified_three_logical": witness,
        "catalogue_position": idx,
        "incidence_refinement_rounds": 12,
        "root_set_sizes": (1, 2),
        "reported_archive_references": 7,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without resampling or portfolio mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Different cyclic gap/triangle prescriptions guarantee inequivalent codes: the parent produced 60 slots but only 22 distinct Tanner graphs.
# REFUTED: C5xC7 quotient-order/orbit cards supply five inequivalent high-rate finalists: sixty slots collapsed to four distinct codes, all measured at distance three, leaving one zero contribution.
# REFUTED: Affine order-spectrum matching/trinomial selection raises the (2,5)x(2,5) measured minimum above six: all five measured leaders had d30s=6, with sector-1 X logical families.
# REFUTED: The affine pivot/binomial weighted-walk catalogue supplies twenty evaluator-distinct slots: the measured program retained seven distinct codes, collapsed three emitted slots as isomorphic, and exhausted its catalogue after ten emitted slots.
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
