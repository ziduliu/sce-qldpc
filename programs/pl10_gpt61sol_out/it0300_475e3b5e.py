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
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("For (C9 semidirect C3, action 4) direct product C2, order 54, shapes (1,3)x(1,3), an order-eighteen binomial beside two screened noncommuting trinomials should improve worse-basis ASC memory above the order-57 parent's fitness 4.8170 by lowering extraction load without admitting short component dependencies.")
GROUP = ("Jump to x^9=y^6=1, y*x*y^-1=x^4, isomorphic to (C9:C3) x C2. Its centre has order six and suitable elements have order eighteen, unlike the order-57 parent. No order-54 bed appears in BEDS. The shown order-55 mixed-degree program scored 5.0213, motivating a longer-binomial, lower-load experiment without establishing an advantage for this group.")
SHAPES = ("Keep equal (1,3) sides for cell 0.400: each component has 54 checks and 162 variables, n=540, kb=216 and guaranteed rate 0.400. Require complete helper-built k=216. Parent failures affected all 228 logicals, with weight-eight sector-1 witnesses and mostly 1.0-1.1x Acol/Bcol and Acol/Brow enrichment; redesign both factors rather than reinforce an unsupported isolated hotspot.")
FILLING = ("Both components use (2,3,3): row total eight, maximum CSS weight eleven and 20736 six-round CNOTs, versus the parent's twelve and 24624. Canonical unordered contracts prescribe triangle relative-order signatures and commutation with a fixed order-eighteen binomial direction. At least one triangle per component must survive the binomial's index-three quotient as a monomial. Fresh searches use verified singleton actions, exact rank/connectivity/C4/kernel-through-four tests, rooted component fingerprints, and exact weight-five/six counts with triple-boundary exposure. Complete verified finalists require CSS, k=216 and no logical through four, then receive common-stream stabilizer-aware BP probes and independent holdout probes. These include degree-weighted data noise and possible spread pairs from opposite checks, but omit ASC ordering, measurement histories and relay/LSD; they are not circuit fitness. Target broad collective flips and slot14's 102/5000 X failures despite d2s=8.")
FAMILY_NAME = "r040_C9C3xC2_q54_13x13_long18_mixeddegree_holdoutbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("(C9:C3) direct product C2 requires q=54")
    # x^9=y^6=1, y*x*y^-1=x^4; 4^3=1 modulo nine.
    # <y^2> is the C3 complement and y^3 is a central C2.
    # (a,b) encodes x^a*y^b. The centre is <x^3,y^3>.
    action = (1, 4, 7, 1, 4, 7)
    els = [(a, b) for b in range(6) for a in range(9)]

    def red(g):
        return (int(g[0]) % 9, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 9, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-action[bi]*a) % 9, bi)

    def idx(g):
        a, b = red(g)
        return a + 9*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _f57_columns(rows, n):
    """Column syndromes of a verified lifted matrix."""
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _f57_briefs(G):
    """Relative-order/centralizer contracts, not saved filling images."""
    from itertools import combinations, combinations_with_replacement

    direction = (1, 3)
    orders = {}
    for g in G.elements:
        value = G.e
        for t in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[G.idx(g)] = t
                break
        else:
            raise ValueError("invalid group element order")
    assert orders[G.idx(direction)] == 18

    triangles = {}
    nonidentity = [g for g in G.elements if g != G.e]
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        differences = (u, v, G.mul(G.inv(u), v))
        signature = tuple(sorted(
            (orders[G.idx(g)],
             int(G.mul(g, direction) == G.mul(direction, g)))
            for g in differences
        ))
        if any(order <= 2 for order, _ in signature):
            continue
        triangles.setdefault(signature, []).append((u, v))

    # <direction> is normal of index three. If a triangle occupies
    # all three quotient cosets, its projection is 1+y+y^2.
    # A repeated quotient coset instead leaves a monomial in F2[C3].
    # At least one such triangle avoids an unavoidable rank deficit.
    types = sorted(triangles)
    book = [
        pair for pair in combinations_with_replacement(types, 2)
        if any(any(commutes for _, commutes in tag) for tag in pair)
    ]
    assert len(book) >= 20
    return book, triangles


def _f57_draw(rng, G, brief):
    """One long binomial and fresh triangles in the prescribed classes."""
    tags, triangles = brief
    cells = [[G.e, (1, 3)]]
    for tag in tags:
        u, v = rng.choice(triangles[tag])
        cell = [G.e, u, v]
        assert len({G.idx(g) for g in cell}) == 3
        assert G.mul(u, v) != G.mul(v, u)
        cells.append(cell)
    return [cells]


def _f57_distinct_differences(G, grid, side):
    """Action-aware prefilter; verified component matrices decide."""
    seen = set()
    for cell in grid[0]:
        for a in cell:
            for b in cell:
                if a == b:
                    continue
                relative = (G.mul(a, G.inv(b)) if side == "A"
                            else G.mul(G.inv(a), b))
                difference = G.idx(relative)
                if difference in seen:
                    return False
                seen.add(difference)
    return True


def _f57_component(G, grid, side):
    """Extract the actual A-left or B-right component using a unit partner."""
    unit = [[[G.e]]]
    if side == "A":
        rows = lifted_matrices(G, grid, unit)[0]
    else:
        rows = lifted_matrices(G, unit, grid)[1]
    width = 3 * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _f57_clean4(columns):
    """Exact exclusion of component kernels through weight four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = set()
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome == 0 or syndrome in singles or syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _f57_cycles(rows, columns):
    """Verified four-cycles and exact six-cycles when four-cycles are absent."""
    m = len(rows)
    adjacency = [0] * m
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            common = (row & rows[j]).bit_count()
            four += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    if four:
        return four, None

    triangles = 0
    for i in range(m):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6
                for d in (c.bit_count() for c in columns))
    six = triangles - stars
    assert six >= 0
    return 0, six


def _f57_six_boundary(columns, limit=None):
    """Exact weight-five/six counts after exclusion through four.

    Mixed even/odd column degrees require both counts. Equal-syndrome
    pair/triple or triple/triple supports must be disjoint after the
    <=4 exclusion. Each five/six word contributes ten partitions.
    """
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    triples = {}
    five = six = one = exposure = 0
    size = len(columns)
    for a in range(size - 2):
        for b in range(a + 1, size - 1):
            pair = columns[a] ^ columns[b]
            for c in range(b + 1, size):
                syndrome = pair ^ columns[c]
                five += pairs.get(syndrome, 0)
                six += triples.get(syndrome, 0)
                triples[syndrome] = triples.get(syndrome, 0) + 1
                weight = syndrome.bit_count()
                if weight <= 4:
                    exposure += (5-weight)**2
                    one += int(weight == 1)
    assert five % 10 == 0 and six % 10 == 0
    return five//10, six//10, one, exposure


def _m54_fingerprint(rows, width):
    """Rooted component Tanner invariant; not full-code canonization.

    Single-row regular components are transitive on their checks.
    Equal fingerprints are conservatively discarded within the pool.
    """
    from collections import Counter
    import hashlib

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, value in enumerate(_f57_columns(rows, width)):
        while value:
            bit = value & -value
            r = bit.bit_length()-1
            adjacency[r].append(nr+c)
            adjacency[nr+c].append(r)
            value ^= bit
    colors = [0]*nr + [len(adjacency[nr+c]) for c in range(width)]
    colors[0] = 20
    digest = hashlib.sha256()
    for _ in range(14):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in adjacency[v])))
            for v in range(len(adjacency))
        ]
        digest.update(repr(sorted(Counter(signatures).items())).encode())
        palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
        colors = [palette[s] for s in signatures]
    return digest.hexdigest()


def _f57_select(rng, G, side, brief, batch, actions):
    """Fresh mixed-degree searches using only helper-derived actions."""
    q = G.q
    pool, seen = [], set()
    accepted = 0

    def component(grid):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    def connected(rows):
        adjacency = [0]*q
        for i, row in enumerate(rows):
            for j in range(i):
                if row & rows[j]:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length()-1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        return reached == (1 << q)-1

    for trial in range(2400):
        grid = _f57_draw(rng, G, brief)
        if not _f57_distinct_differences(G, grid, side):
            continue
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in grid[0])
        if signature in seen:
            continue
        seen.add(signature)
        rows = component(grid)
        if gf2_rank(rows) != q:
            continue
        columns = _f57_columns(rows, 3*q)
        assert all(c.bit_count() == (2 if j < q else 3)
                   for j, c in enumerate(columns))
        four, cycles = _f57_cycles(rows, columns)
        if four or not connected(rows) or not _f57_clean4(columns):
            continue
        accepted += 1
        pool.append((cycles, trial, grid, rows, columns))
        pool.sort(key=lambda item: item[:2])
        del pool[6:]
        if accepted >= 48:
            break
    if not pool:
        raise ValueError("no clean full-rank order-eighteen component")

    refined, identities = [], set()
    for cycles, trial, grid, rows, columns in pool:
        identity = _m54_fingerprint(rows, 3*q)
        if identity in identities:
            continue
        identities.add(identity)
        five, six, one, exposure = _f57_six_boundary(columns)
        tail = ((cycles, exposure) if batch % 2
                else (exposure, cycles))
        key = (five, six, one) + tail
        refined.append((key, trial, grid, rows, identity))
    refined.sort(key=lambda item: item[:2])
    return refined[:2]


def _m54_probe(checks, stabilizers, n, seed, shots):
    """Stabilizer-aware BP data-noise surrogate, NOT ASC simulation.

    Degree-weighted data exposure and sampled opposite-check pairs
    approximate some extraction costs. Measurement histories, actual
    gate order, relay BP and LSD are deliberately not modeled.
    """
    import numpy as np

    def sites(row):
        result = []
        while row:
            bit = row & -row
            result.append(bit.bit_length()-1)
            row ^= bit
        return result

    neighbors = [sites(row) for row in checks]
    hooks = [sites(row) for row in stabilizers]
    nr = len(neighbors)
    row_width = max(map(len, neighbors))
    incidence = np.full((nr, row_width), n, dtype=np.int32)
    for r, ns in enumerate(neighbors):
        incidence[r, :len(ns)] = ns
    flat = incidence.ravel()
    valid = flat != n
    edges = len(flat)
    degrees = np.bincount(flat[valid], minlength=n)
    opposite = np.asarray(
        [c.bit_count() for c in _f57_columns(stabilizers, n)])
    assert np.all(degrees > 0)
    variable_edges = np.full(
        (n+1, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge in np.flatnonzero(valid):
        c = int(flat[edge])
        variable_edges[c, used[c]] = edge
        used[c] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.008, 0.016, 0.026))[np.arange(shots) % 3]
    rates = np.minimum(
        levels[:, None]*(degrees+opposite)[None, :]/6.0, 0.18)
    error = np.zeros((shots, n+1), dtype=bool)
    error[:, :n] = channel.random((shots, n)) < rates
    chosen = channel.integers(len(hooks), size=shots)
    active = channel.random(shots) < 0.25
    for shot in np.flatnonzero(active):
        ns = hooks[int(chosen[shot])]
        a = int(channel.integers(len(ns)))
        b = int(channel.integers(len(ns)-1))
        b += b >= a
        error[shot, ns[a]] ^= True
        error[shot, ns[b]] ^= True

    prior_rates = np.minimum(
        rates + 0.5*opposite[None, :]/sum(opposite), 0.25)
    prior = np.full((shots, n+1), 30.0)
    prior[:, :n] = np.log((1-prior_rates)/prior_rates)
    target = (
        error[:, flat].reshape(shots, nr, row_width).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(word):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    failures = iteration_sum = 0
    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, row_width)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logs = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
        products = np.minimum(
            np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
        outgoing = (
            (np.log1p(products)-np.log1p(-products))
            * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)
        outgoing[:, ~valid] = 0.0
        padded = np.zeros((count, edges+1))
        padded[:, :edges] = outgoing
        posterior = prior + padded[:, variable_edges].sum(axis=2)
        posterior[:, n] = 30.0
        hard = posterior < 0
        syndrome = (
            hard[:, flat].reshape(count, nr, row_width).sum(axis=2) & 1
        ).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local, :n] ^ hard[local, :n]):
                word |= 1 << int(c)
            failures += int(nontrivial(word))
            iteration_sum += iteration

        keep = ~settled
        if not np.any(keep):
            return failures, 0, iteration_sum
        messages = np.clip(
            0.30*messages + 0.70*(posterior[:, flat]-outgoing),
            -28, 28)[keep]
        messages[:, ~valid] = 30.0
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failures+unresolved, unresolved, iteration_sum+33*unresolved


def _f57_small_logical(checks, stabilizers, n):
    """Exact existence test for a quantum logical of weight <=4."""
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

    columns = _f57_columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is not None:
            if nontrivial(word ^ previous):
                return True
        else:
            seen[syndrome] = word
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(syndrome)
            if previous is not None:
                if nontrivial(word ^ previous):
                    return True
            else:
                seen[syndrome] = word
    return False


def NEW_CTX(rng, G, idx):
    """Deliberate long-binomial contracts and verified joint selection."""
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 54 and A_SHAPE == B_SHAPE == (1, 3)
    supports = [[2, 3, 3]]
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 11
    assert 8+3 <= min(12, W_MAX)  # HX and HZ separately.

    book, triangles = _f57_briefs(G)
    contracts = list(combinations_with_replacement(range(len(book)), 2))
    budget = max(20, N_SLOTS)
    assert len(contracts) >= budget and 0 <= idx < budget
    # Unordered pairs avoid deliberately spending slots on A/B exchange.
    # Larger slot budgets spread over more architecture contracts.
    position = idx*len(contracts)//budget
    ia, ib = contracts[position]
    ba, bb = book[ia], book[ib]
    batch = idx//20

    # Two bulk probes provide exact singleton actions. Probe sizes need
    # not equal the submitted size; no handwritten lift convention.
    mask = (1 << q)-1
    probe = [[[g] for g in G.elements]]
    zero = [[[]]]
    ax, _, _ = lifted_matrices(G, probe, zero)
    _, bz, _ = lifted_matrices(G, zero, probe)
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(G.elements):
        actions["A"][G.idx(g)] = tuple(
            (row >> (t*q)) & mask for row in ax)
        actions["B"][G.idx(g)] = tuple(
            (row >> (t*q)) & mask for row in bz)

    seed = 540018003 + 1009*idx

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2])

    for attempt in range(3):
        aa = _f57_select(
            rng, G, "A", (ba, triangles), batch, actions["A"])
        bb_entries = _f57_select(
            rng, G, "B", (bb, triangles), batch, actions["B"])
        options = []
        identities = set()
        for ka, ta, A, ar, fa in aa:
            for kb, tb, B, br, fb in bb_entries:
                identity = tuple(sorted((fa, fb)))
                if identity in identities:
                    continue
                identities.add(identity)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 540 or k != 216:
                    continue
                actual_a = [
                    sum(((row >> (3*j*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]
                ]
                actual_b = [
                    row & ((1 << (3*q))-1) for row in hz[:q]
                ]
                assert actual_a == ar and actual_b == br
                if _f57_small_logical(hx, hz, n):
                    continue
                if _f57_small_logical(hz, hx, n):
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) == 11
                assert 6*sum(row.bit_count() for row in hx+hz) == 20736
                geometry = tuple(v for a, b in zip(ka, kb)
                                 for v in (max(a, b), a+b))
                options.append((geometry, ta, tb, A, B, hx, hz, identity))
        options.sort(key=lambda item: item[:3])
        if not options:
            continue

        coarse = []
        for number, option in enumerate(options[:3]):
            geometry, ta, tb, A, B, hx, hz, identity = option
            x = _m54_probe(hz, hx, 540, seed, 48)
            z = _m54_probe(hx, hz, 540, seed+1, 48)
            coarse.append((quality(x, z)+geometry+(number,),
                           number, option, x, z))
        coarse.sort(key=lambda item: item[:2])

        finalists = []
        for _, number, option, cx, cz in coarse[:2]:
            geometry, ta, tb, A, B, hx, hz, identity = option
            # Independent holdout streams, shared across finalists.
            x = _m54_probe(hz, hx, 540, seed+1000003, 96)
            z = _m54_probe(hx, hz, 540, seed+1000004, 96)
            tx = tuple(a+b for a, b in zip(cx, x))
            tz = tuple(a+b for a, b in zip(cz, z))
            score = quality(tx, tz)+geometry+(number,)
            finalists.append((score, A, B, tx, tz, identity))
        finalists.sort(key=lambda item: item[0])
        score, A, B, x, z, identity = finalists[0]
        return {"idx": idx, "briefs": (ba, bb), "k": 216,
                "component_identity": identity,
                "bp_surrogate": (x, z), "screen": score,
                "cells": {"A": A, "B": B}}
    raise ValueError("no mixed-degree order-54 target-rate finalist")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone give reliable r040f memory: parent fitness 3.2187, all five measured codes hit a failed-shot cap, with weight-4 sector-1 witnesses and broadly distributed logical flips.
# REFUTED: Balanced short-collision optimization on D58 (1,3)x(1,3) ensures rare circuit failures: fitness was 4.3650, every measured basis hit the failed-shot cap, and slot 0 had 105/3500 X failures despite d2s=8.
# REFUTED: Removing involutions and independently selecting four-cycle-free trinomials by short kernels and boundaries suffices for rare ASC failures: the order-57 parent scored 4.8170, all ten measured bases reached failed-shot caps, and slot14 X failed 102/5000 shots with collective flips across all 228 logicals.
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
