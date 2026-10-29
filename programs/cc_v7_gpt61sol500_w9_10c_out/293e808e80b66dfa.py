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
CONCEPT = ("Metacyclic Z5:Z2, action 4, order 10, A=(2,7), B=(2,7): balanced rotational-cycle constituents and opposite-layer grounded constituents should supply at least five novel certified-distance-three-or-four codes in cell 0.500, improving the parent's two-code fitness.")
GROUP = ("Keep the frozen group and order byte-for-byte. This bed measured distance four at k=262 and 255, but sixty matching-catalogue slots retained only two classes. Order-five rotations permit new internal-cycle columns without the parallel-edge mechanism of involutory binomials; this is a structural opportunity, not a transferred distance guarantee.")
SHAPES = ("Keep equal 70-variable, 20-check components, n=530, kb=250 and guaranteed rate 25/53=0.471698. The shape permits q<=11; the frozen group fixes q=10. Verified full ranks retain [0.475,0.525), including possible regular rank-surplus gains beyond twelve. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly, with no sector-2 minima reported, motivating independent constituent geometry.")
FILLING = ("Enumerate regular cards with zero, one or two rotational-cycle columns per layer, and grounded cards with an optional opposite-layer cycle. Every published pair contains a cycle column absent from the parent's two classes. Regular rows total seven; grounded rows total seven/six or six/seven; columns total at most two, so both LP maxima are at most nine. Exact scaffold matrices exclude commuting weights one and two, exhaustively test logical weight three, and verify an actual weight-three/four non-stabilizer witness. An immutable catalogue ranks certified kd^2/n, rejects the shown parent references and conservatively separates complete uncoloured Tanner graphs using intrinsic high-degree contraction joint-walk keys. Five distance-three codes with k>=255 score at least 4.330; five-code yield and novelty against the unseen archive remain falsifiable hypotheses.")
FAMILY_NAME = "r050_m5r2s4_27x27_rotcycles_ranked_cert34_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this metacyclic group requires order 10")
    # Same split metacyclic rule: x^5=y^2=1, y*x*y^-1=x^4.
    # Factor 53 requires q<=11, excluding the parent's q=14.
    m, r, s = 5, 2, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _matching_columns(rows, n):
    columns = [0] * n
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << i
            row ^= bit
    return columns


def _matching_clean3(columns):
    """Exactly exclude all nonzero kernel words of weights 1, 2, 3."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    for i, first in enumerate(columns):
        for j in range(i):
            if (first ^ columns[j]) in singles:
                return False
    return True


def _matching_span(rows):
    basis = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                break
            row ^= basis[pivot]
    return basis


def _matching_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        if pivot not in basis:
            return True
        word ^= basis[pivot]
    return False


def _matching_four_witness(columns, stabilizers):
    """Find and verify an actual weight-four non-stabilizer kernel word."""
    basis = _matching_span(stabilizers)
    representatives = {}
    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            word = (1 << i) | (1 << j)
            if syndrome in representatives:
                witness = word ^ representatives[syndrome]
                if (witness.bit_count() == 4
                        and _matching_outside(witness, basis)):
                    check = 0
                    bits = witness
                    while bits:
                        bit = bits & -bits
                        check ^= columns[bit.bit_length() - 1]
                        bits ^= bit
                    assert check == 0
                    return witness
            else:
                representatives[syndrome] = word
    return None


def _matching_fingerprint(hx, hz, n):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Includes exact check intersections, common-neighbour profiles,
    rooted distance spheres, and qubit-degree neighbourhood profiles.
    Different signatures certify non-isomorphism. Equality only causes
    conservative rejection; it is not an isomorphism certificate.
    """
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    m = len(rows)
    columns = _matching_columns(rows, n)
    degrees = [row.bit_count() for row in rows]
    overlap = [[0] * m for _ in rows]
    adjacency = [0] * m
    multiple = [0] * m
    for i, row in enumerate(rows):
        for j in range(i):
            weight = (row & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = weight
            if weight:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
            if weight >= 2:
                multiple[i] |= 1 << j
                multiple[j] |= 1 << i

    def spheres(root, relation):
        visited = frontier = 1 << root
        result = []
        while frontier:
            neighbors = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                neighbors |= relation[bit.bit_length() - 1]
                bits ^= bit
            frontier = neighbors & ~visited
            if frontier:
                result.append(frontier.bit_count())
                visited |= frontier
        return tuple(result)

    roots = []
    for i in range(m):
        profile = Counter(
            (degrees[j], overlap[i][j],
             (adjacency[i] & adjacency[j]).bit_count(),
             (multiple[i] & multiple[j]).bit_count())
            for j in range(m) if j != i
        )
        roots.append((
            degrees[i], tuple(sorted(profile.items())),
            spheres(i, adjacency), spheres(i, multiple),
        ))

    qubit_profiles = Counter()
    for column in columns:
        checks = []
        bits = column
        while bits:
            bit = bits & -bits
            checks.append(bit.bit_length() - 1)
            bits ^= bit
        internal = Counter(
            overlap[a][b]
            for turn, a in enumerate(checks) for b in checks[:turn]
        )
        qubit_profiles[(
            len(checks), tuple(sorted(degrees[a] for a in checks)),
            tuple(sorted(internal.items())),
        )] += 1

    signature = (
        n, m, tuple(sorted(roots)),
        tuple(sorted(qubit_profiles.items())),
    )
    return sha256(repr(signature).encode("ascii")).digest()


def _matching_cells(G, complement, singleton=False):
    """A complement prescribes a voltage subset, not an orbit draw."""
    excluded = set(complement)
    labels = [g for g in G.elements if G.idx(g) not in excluded]
    wanted = 6 if singleton else 7
    assert len(labels) == wanted
    top = [[G.e] for _ in labels]
    bottom = [[g] for g in labels]
    if singleton:
        top.append([G.e])
        bottom.append([])
    return [top, bottom]


from functools import lru_cache


@lru_cache(maxsize=1)
def _rotcycle_portfolio(G, count):
    """Immutable finite portfolio; caching changes cost, not outputs.

    Constituents are enumerated structurally. Full-code invariants,
    actual ranks and exact logical witnesses decide publication.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    if G.q != 10:
        raise ValueError("rotational-cycle portfolio requires order ten")
    q, e, els = G.q, G.e, G.elements
    cap = min(9, W_MAX)
    inverse = [G.idx(G.inv(g)) for g in els]

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if G._key(value) == G._key(e):
                return length
        raise ValueError("invalid finite-group order")

    # Involutions would duplicate the two columns of an internal edge.
    directions = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in els if order(g) > 2
    })
    if len(directions) != 2:
        raise ValueError("expected two order-five inverse classes")
    direction_pos = {d: i for i, d in enumerate(directions)}

    def automorphism(t, u, v):
        a, b = els[t]
        return G.idx(((u*a + v*b) % 5, b))

    # Known constituent gauges plus automorphisms prune proposals only.
    # They do not replace the full-code intrinsic separation below.
    gauge_actions = set()
    for u in range(1, 5):
        for v in range(5):
            phi = tuple(automorphism(t, u, v) for t in range(q))
            dm = tuple(min(phi[d], inverse[phi[d]])
                       for d in directions)
            for left in els:
                for right in els:
                    permutation = tuple(
                        G.idx(G.mul(left, G.mul(els[phi[t]], right)))
                        for t in range(q))
                    gauge_actions.add((dm, permutation))
    gauge_actions = sorted(gauge_actions)

    def canonical(card):
        kind, upper, lower, values = card
        best = None
        for dm, permutation in gauge_actions:
            a = tuple(sorted(dm[direction_pos[d]] for d in upper))
            b = tuple(sorted(dm[direction_pos[d]] for d in lower))
            labels = tuple(sorted(permutation[t] for t in values))
            images = [(kind, a, b, labels)]
            if kind == 0:
                images.append((
                    kind, b, a,
                    tuple(sorted(inverse[t] for t in labels))))
            for image in images:
                if best is None or image < best:
                    best = image
        return best

    def image(card, u, v):
        kind, upper, lower, values = card
        phi = tuple(automorphism(t, u, v) for t in range(q))
        return (
            kind,
            tuple(sorted(min(phi[d], inverse[phi[d]]) for d in upper)),
            tuple(sorted(min(phi[d], inverse[phi[d]]) for d in lower)),
            tuple(sorted(phi[t] for t in values)),
        )

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def cells(card):
        kind, upper, lower, values = card
        if kind == 0:
            assert len(upper) == len(lower)
            assert len(values) + len(upper) + len(lower) == 7
            grid = [
                [[e, els[d]] for d in upper]
                + [[e] for _ in values] + [[] for _ in lower],
                [[] for _ in upper]
                + [[els[t]] for t in values]
                + [[e, els[d]] for d in lower],
            ]
            assert [sum(map(len, row)) for row in grid] == [7, 7]
        elif not lower:
            assert not upper and len(values) == 6
            grid = [
                [[e] for _ in values] + [[e]],
                [[els[t]] for t in values] + [[]],
            ]
        else:
            # Ground the loop-free upper layer; put the ring below.
            assert not upper and len(lower) == 1 and len(values) == 5
            grid = [
                [[e] for _ in values] + [[], [e]],
                [[els[t]] for t in values]
                + [[e, els[lower[0]]], []],
            ]
        assert all(len(row) == 7 for row in grid)
        assert all(len({G._key(g) for g in cell}) == len(cell)
                   for row in grid for cell in row)
        return freeze(grid)

    descriptors = set()
    for p in range(3):
        steps = list(combinations(directions, p))
        for upper in steps:
            for lower in steps:
                for values in combinations(range(q), 7 - 2*p):
                    descriptors.add(canonical(
                        (0, upper, lower, values)))
    for values in combinations(range(q), 6):
        descriptors.add(canonical((1, (), (), values)))
    for d in directions:
        for values in combinations(range(q), 5):
            descriptors.add(canonical((1, (), (d,), values)))

    def no_two(cols):
        return 0 not in cols and len(set(cols)) == len(cols)

    regular, grounded = [], []
    width = 7*q
    mask = (1 << width) - 1
    dummy = [[[]]]
    for card in sorted(descriptors):
        grid = cells(card)
        # Both actions are exposed by the EXACT scaffold helper.
        left, _, _ = lifted_matrices(G, grid, dummy)
        _, right, _ = lifted_matrices(G, dummy, grid)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if not (no_two(_matching_columns(left, width))
                and no_two(_matching_columns(right, width))):
            continue
        ranks = (gf2_rank(left), gf2_rank(right))
        record = (card, ranks, grid)
        (regular if card[0] == 0 else grounded).append(record)
    if not regular:
        raise ValueError("no verified regular rotational constituents")

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the COMPLETE uncoloured Tanner graph.

        Qubit degrees identify the high-degree class without coordinates.
        A check becomes an edge or grounding loop on that class.
        Vertex labels retain qubit degree and incident check weights.
        Joint walk profiles through four and diagonal walks through eight
        are invariant under every check/qubit relabeling.

        Unequal keys certify inequivalence; equal keys are discarded
        conservatively. No CSS colours or group labels enter this key.
        """
        rows = hx + hz
        cols = _matching_columns(rows, n)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() > 9]
        if len(high) != 4*q:
            raise ValueError("unexpected intrinsic high-degree class")
        positions = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        adjacency = [Counter() for _ in high]
        incident = [Counter() for _ in high]
        for row in rows:
            bits = row & high_mask
            endpoints = []
            while bits:
                bit = bits & -bits
                endpoints.append(positions[bit.bit_length() - 1])
                bits ^= bit
            if len(endpoints) not in (1, 2):
                raise ValueError("contracted check is not edge or grounding")
            u = endpoints[0]
            v = u if len(endpoints) == 1 else endpoints[1]
            label = (row.bit_count(), len(endpoints))
            adjacency[u][v] += 1
            incident[u][label] += 1
            if u != v:
                adjacency[v][u] += 1
                incident[v][label] += 1

        size = len(high)
        neighbours = [tuple(sorted(a.items())) for a in adjacency]
        colors = [
            (cols[j].bit_count(), tuple(sorted(incident[i].items())))
            for i, j in enumerate(high)]
        diagonals, walks = [], []
        for root in range(size):
            vector = [0]*size
            vector[root] = 1
            diagonal = []
            pairs = [[] for _ in high]
            for length in range(1, 9):
                following = [0]*size
                for u, value in enumerate(vector):
                    if value:
                        for v, multiplicity in neighbours[u]:
                            following[v] += value*multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 4:
                    for v, value in enumerate(vector):
                        pairs[v].append(value)
            diagonals.append(tuple(diagonal))
            walks.append(tuple(tuple(p) for p in pairs))

        profiles = []
        for u in range(size):
            histogram = Counter(
                (colors[v], diagonals[v], walks[u][v])
                for v in range(size))
            profiles.append((
                colors[u], diagonals[u],
                tuple(sorted(histogram.items()))))
        intrinsic = (
            n, len(rows),
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(profiles)),
        )
        return sha256(repr(intrinsic).encode("ascii")).digest()

    # Fingerprint-only references: neither parent class is published.
    parent_A = freeze([
        [[e] for _ in range(7)],
        [[els[t]] for t in range(3, 10)],
    ])
    parent_leaf = freeze([
        [[e] for _ in range(7)],
        [[els[t]] for t in range(4, 10)] + [[]],
    ])
    blocked = set()
    for A, B in ((parent_A, parent_A), (parent_A, parent_leaf)):
        hx, hz, n = lifted_matrices(G, A, B)
        blocked.add(graph_key(hx, hz, n))

    def cyclic(card):
        return bool(card[1] or card[2])

    variants = {}
    for record in regular + grounded:
        card = record[0]
        variants[card] = sorted({
            image(card, u, v)
            for u in range(1, 5) for v in range(5)
        })

    plans = []
    for ai, a in enumerate(regular):
        for bi in range(ai, len(regular)):
            b = regular[bi]
            if not (cyclic(a[0]) or cyclic(b[0])):
                continue
            for bcard in variants[b[0]]:
                plans.append((
                    a[1][0] + b[1][0], ai, 0, bi, bcard))
        for bi, b in enumerate(grounded):
            if not (cyclic(a[0]) or cyclic(b[0])):
                continue
            for bcard in variants[b[0]]:
                plans.append((
                    a[1][0] + b[1][0], ai, 1, bi, bcard))
    # Lower constituent ranks are a proposal priority only.
    # The complete measured-rank formula decides final quality.
    plans.sort()

    def three_witness(cols, stabilizers):
        """Exhaustive logical-three test, not just triangle counting."""
        singles = {syndrome: i for i, syndrome in enumerate(cols)}
        basis = _matching_span(stabilizers)
        for i, first in enumerate(cols):
            for j in range(i):
                h = singles.get(first ^ cols[j])
                if h is None:
                    continue
                word = (1 << i) | (1 << j) | (1 << h)
                if (word.bit_count() == 3
                        and _matching_outside(word, basis)):
                    assert cols[i] ^ cols[j] ^ cols[h] == 0
                    return word
        return None

    graph_keys, filling_keys = set(), set()
    entries = []
    grid_cache = {}
    for _, ai, boundary, bi, bcard in plans[:4096]:
        a = regular[ai]
        b = (grounded if boundary else regular)[bi]
        A = a[2]
        if bcard not in grid_cache:
            grid_cache[bcard] = cells(bcard)
        B = grid_cache[bcard]
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > cap
                or not _no_dead_lines(sa, sb)):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 530 or max(map(int.bit_count, hx + hz)) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            continue

        cx = _matching_columns(hz, n)
        cz = _matching_columns(hx, n)
        if not (no_two(cx) and no_two(cz)):
            continue
        signature = graph_key(hx, hz, n)
        if signature in blocked or signature in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified rotational-cycle lift failed CSS")

        witness = three_witness(cx, hx)
        witness_type = "X"
        if witness is None:
            witness = three_witness(cz, hz)
            witness_type = "Z"
        distance = 3
        if witness is None:
            # Both complete logical-three tests were exhaustive.
            distance = 4
            witness = _matching_four_witness(cx, hx)
            witness_type = "X"
            if witness is None:
                witness = _matching_four_witness(cz, hz)
                witness_type = "Z"
        if witness is None:
            continue

        checks, stabilizers = ((hz, hx) if witness_type == "X"
                               else (hx, hz))
        assert witness.bit_count() == distance
        assert all((row & witness).bit_count() % 2 == 0
                   for row in checks)
        assert _matching_outside(witness, _matching_span(stabilizers))
        graph_keys.add(signature)
        quality = (k*distance*distance/n, distance, k, signature)
        entries.append((
            quality, A, B, k, distance, witness, witness_type,
            (a[0], bcard), signature,
        ))

    # Never pad missing entries with orbit copies or reuse parent cells.
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed selection of a ranked immutable structural design."""
    catalogue = _rotcycle_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(catalogue)):
        raise ValueError("novel rotational-cycle certified catalogue exhausted")
    quality, A, B, k, distance, witness, kind, cards, signature = catalogue[idx]
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "screen_k": k, "distance_certificate": distance,
        "witness": witness, "witness_type": kind,
        "prescription": cards, "certified_score": quality[0],
        "intrinsic_signature": signature.hex(),
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell without changing the portfolio."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Weight-legal uniform full-support (6,2)x(6,2) fillings reliably avoid weight-2 logicals: refuted by parent slot 5, d30s=2, with 252 reported weight-2 operators in sector 2.
# REFUTED: The leaf-edge cycle-fingerprint bank supplies twenty inequivalent publishable designs: this parent retained only three distinct codes, all measured at distance three, and exhausted its bank from slot 31 onward.
# REFUTED: Matching-complement and one-leaf cards supply five novel distance-four classes on D10 (2,7)x(2,7): sixty evaluated slots retained only two measured classes, with sixteen isomorphic repeats and forty-two catalogue-exhaustion errors.
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
