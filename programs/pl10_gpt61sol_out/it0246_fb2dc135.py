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
CONCEPT = ("Cyclic Z17, order 17, shapes (3,5)x(3,5): full-rank augmentation and predominantly six-term component rows should reduce collective circuit failures while retaining k=68 in rate cell 0.100.")
GROUP = ("Preserve the metacyclic construction rule m=1,s=1. The new size factor 34 makes order 23 illegal at n=782; order 17 gives n=578 and is the largest legal lift. The existing Z17 bed reached about 7.2, motivating this order without establishing a circuit advantage for full-rank augmentation.")
SHAPES = ("Equal (3,5) components each have 85 variables and 51 checks, hence dimension at least 34. The guaranteed bound is kb=68, rate 2/17=0.11765. Require actual component ranks 51 and verified complete k=68, below 0.125; unlike prior rank-one-augmentation Z17 programs, no augmentation deficit is deliberately retained. Parent X failures implicated several Acol/Bcol blocks and different Brow blocks at modest enrichment, motivating distributed protection rather than a unique-hotspot repair.")
FILLING = ("Enumerate canonical support pairs with column degrees three/four, row totals five through seven and augmentation rank three; prioritize six-term rows. Regular-six pairs have CSS weight ten and 29376 six-round CNOTs; every pair has weight <=11. Optimize fresh labels using helper-derived singleton actions, then compare exact kernel exclusion through six, a capped sampled protection floor of sixteen, short-witness multiplicity and triple-boundary concentration. Complete verified matrices enforce k=68 and exclude logicals through four; opposite-check pair-boundary exposure breaks ties. Target the parent's 5..11 X failures/200000 and collective flips across most logicals, versus predominantly zero Z failures. These structural screens are circuit-pL hypotheses, not ASC hook certificates or measured failure estimates.")
FAMILY_NAME = "r010_m1r17s1_35x35_fullrank_gatelean_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same metacyclic construction rule, with only the order changed.
    # The new size factor 34 would give n=782 at order 23.
    assert q == 17
    m, r, s = 1, 17, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Canonical support contracts with full-rank, helper-verified lifts."""
    import math
    from itertools import combinations_with_replacement, permutations

    q = G.q
    assert q == 17 and A_SHAPE == B_SHAPE == (3, 5)
    width = 5 * q
    mask = (1 << q) - 1
    row_permutations = tuple(permutations(range(3)))

    # Canonicalize counts before optimizing any labels. Different slots
    # change incidence, not just the random seed or a saved-code labeling.
    column_types = [
        (a, b, c) for a in range(3) for b in range(3)
        for c in range(3) if 3 <= a + b + c <= 4
    ]
    profile_keys = set()
    for entries in combinations_with_replacement(column_types, 5):
        totals = [sum(col[i] for col in entries) for i in range(3)]
        if not all(5 <= total <= 7 for total in totals):
            continue
        parity_rows = [
            sum((col[i] & 1) << j for j, col in enumerate(entries))
            for i in range(3)
        ]
        if gf2_rank(parity_rows) != 3:
            continue
        key = min(
            tuple(sorted(tuple(col[i] for i in p) for col in entries))
            for p in row_permutations
        )
        profile_keys.add(key)

    profiles = [
        [[col[i] for col in key] for i in range(3)]
        for key in sorted(profile_keys)
    ]

    def profile_cost(grid):
        return (
            sum((sum(row) - 6) ** 2 for row in grid),
            sum(w == 0 for row in grid for w in row),
            tuple(tuple(row) for row in grid),
        )

    profiles.sort(key=profile_cost)
    briefs = [(a, b) for a in range(len(profiles))
              for b in range(a, len(profiles))]
    briefs.sort(key=lambda pair: (
        sum(profile_cost(profiles[p])[0] for p in pair),
        max(profile_cost(profiles[p])[0] for p in pair),
        sum(profile_cost(profiles[p])[1] for p in pair),
        pair,
    ))
    assert len(briefs) >= 20
    if idx >= len(briefs):
        raise ValueError("No unused canonical support pair")

    pa, pb = briefs[idx]
    sa, sb = profiles[pa], profiles[pb]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    for support in (sa, sb):
        assert gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support]) == 3
        assert all(3 <= sum(row[j] for row in support) <= 4
                   for j in range(5))
        assert all(0 <= w <= min(2, CELL_MAX)
                   for row in support for w in row)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def columns(rows, nc):
        out = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    # Compose only actions extracted from the verified construction.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows

    def geometry(rows):
        adj = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(len(rows)):
            lower = adj[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adj[i] & adj[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        cols = columns(rows, width)
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count() for c in cols))
        # Exact six cycles when four=0; proposal heuristic otherwise.
        six = max(0, triangles - stars)
        deficit = len(rows) - gf2_rank(rows)
        energy = (200000 * deficit + 120 * four + six / 5) / q
        return energy, deficit, four, six

    def short_boundary(cols):
        """Exact <=6 existence and a separate triple-boundary surrogate."""
        seen = {0}
        bad = False

        def record(value):
            nonlocal bad
            if value in seen:
                bad = True
            seen.add(value)

        for value in cols:
            record(value)
        for i, x in enumerate(cols):
            for y in cols[:i]:
                record(x ^ y)
        burden = 0
        loads = [0] * 5
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    record(syndrome)
                    weight = syndrome.bit_count()
                    if weight <= 4:
                        cost = (6 - weight) ** 2
                        burden += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[h // q] += cost
        return int(bad), burden / width, max(loads) / q

    def sampled_words(cols, side):
        """Actual sampled kernel words: upper bounds, not certificates."""
        srng = random.Random(170068 + 1009 * idx + int(side == "B"))
        order = list(range(width))
        minimum = width + 1
        short = set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 14:
                    short.add(word)

        for trial in range(64):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)
            for word in sorted(basis, key=int.bit_count)[:6]:
                for step in range(3):
                    best = word
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best.bit_count():
                            best = candidate
                    if best == word:
                        break
                    word = best
                    offer(word)
            if minimum <= 6:
                break

        loads = [0] * 5
        for word in short:
            for j in range(5):
                loads[j] += ((word >> (j * q)) & mask).bit_count()
        return minimum, len(short), max(loads)

    def portfolio(support, side):
        sites = [(i, j) for i, row in enumerate(support)
                 for j, w in enumerate(row) if w]
        pool, seen = [], set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            if stats[1]:
                return
            key = (stats[2], stats[3])
            if len(pool) >= 8 and key > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((key, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(6):
            grid = [[rng.sample(G.elements, w) for w in row]
                    for row in support]
            rows = component(grid, side)
            current = geometry(rows)
            remember(grid, rows, current)
            for step in range(240):
                i, j = rng.choice(sites)
                old = grid[i][j]
                proposal = list(old)
                if rng.random() < 0.25:
                    shift = G.pick(rng)
                    proposal = [G.mul(shift, g) for g in old]
                else:
                    t = rng.randrange(len(proposal))
                    legal = [g for g in G.elements
                             if G._key(g) not in {G._key(x) for x in old}]
                    proposal[t] = rng.choice(legal)
                grid[i][j] = proposal
                trial_rows = component(grid, side)
                proposed = geometry(trial_rows)
                delta = proposed[0] - current[0]
                temperature = 14.0 * (0.98 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    rows, current = trial_rows, proposed
                    if delta < 0 or step % 12 == 0:
                        remember(grid, rows, current)
                else:
                    grid[i][j] = old
            remember(grid, rows, current)

        # Rank-safe fallback remains inside the same support contract.
        if not pool:
            for trial in range(128):
                grid = [[rng.sample(G.elements, w) for w in row]
                        for row in support]
                rows = component(grid, side)
                remember(grid, rows, geometry(rows))
        if not pool:
            raise ValueError("No full-rank component in support contract")

        finalists = []
        for coarse, number, grid, rows in pool:
            cols = columns(rows, width)
            bad, boundary, peak = short_boundary(cols)
            d, multiplicity, load = sampled_words(cols, side)
            key = (bad, max(0, 16 - d), multiplicity, load,
                   peak, boundary, coarse[0], coarse[1])
            finalists.append((key, number, grid, rows))
        finalists.sort(key=lambda item: item[:2])
        return finalists[:4]

    def small_logical(checks, stabilizers, n):
        """Exact nontrivial logical existence through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = row
                    break
                row ^= previous

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    return True
                word ^= previous
            return False

        cols = columns(checks, n)
        seen = {0: 0}
        for i, value in enumerate(cols):
            word = 1 << i
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(value)
                if previous is None:
                    seen[value] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    def pair_exposure(hx, hz, n):
        """Possible correlated-fault opportunities, not ASC-certified hooks."""
        rates, peaks = [], []
        for checks, opposite in ((hx, hz), (hz, hx)):
            cols = columns(checks, n)
            burden = [0] * (n // q)
            opportunities = [0] * (n // q)
            total = count = 0
            for row in opposite:
                neighbors = []
                while row:
                    bit = row & -row
                    neighbors.append(bit.bit_length() - 1)
                    row ^= bit
                for t, i in enumerate(neighbors):
                    for j in neighbors[:t]:
                        count += 1
                        bi, bj = i // q, j // q
                        opportunities[bi] += 1
                        opportunities[bj] += 1
                        boundary = (cols[i] ^ cols[j]).bit_count()
                        if boundary <= 4:
                            cost = (5 - boundary) ** 2
                            total += cost
                            burden[bi] += cost
                            burden[bj] += cost
            rates.append(total / max(1, count))
            peaks.append(max(b / max(1, o)
                             for b, o in zip(burden, opportunities)))
        return max(peaks), max(rates), sum(rates)

    apool = portfolio(sa, "A")
    bpool = portfolio(sb, "B")
    options = []
    for ak, at, A, ar in apool:
        for bk, bt, B, br in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 578 or k != 68:
                continue
            paired = tuple(v for t in range(len(ak))
                           for v in (max(ak[t], bk[t]), ak[t] + bk[t]))
            options.append((paired, at, bt, A, B, hx, hz, ar, br))
    options.sort(key=lambda item: item[:3])

    winner = None
    for paired, at, bt, A, B, hx, hz, ar, br in options[:6]:
        if small_logical(hx, hz, 578) or small_logical(hz, hx, 578):
            continue
        exposure = pair_exposure(hx, hz, 578)
        # Short-word protection is capped; then compare fault-opportunity
        # concentration before further sampled-witness multiplicity.
        key = paired[:4] + exposure + paired[4:] + (at, bt)
        if winner is None or key < winner[0]:
            winner = (key, A, B, hx, hz, ar, br)
    if winner is None:
        raise ValueError("No verified rate-0.100 short-logical-safe finalist")

    key, A, B, hx, hz, ar, br = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
    assert lifted_k(G, A, B) == 68 and 8 * 68 < 578

    # Confirm cached actions against the exact final component blocks.
    actual_a = [
        sum(((hx[(i * 5) * q + t] >> (j * 5 * q)) & mask)
            << (j * q) for j in range(5))
        for i in range(3) for t in range(q)
    ]
    actual_b = [row & ((1 << width) - 1) for row in hz[:3 * q]]
    assert actual_a == ar and actual_b == br
    assert gf2_rank(ar) == gf2_rank(br) == 51
    cnots = 6 * sum(row.bit_count() for row in hx + hz)
    expected = 6 * q * 8 * (sum(map(sum, sa)) + sum(map(sum, sb)))
    assert cnots == expected
    if all(sum(row) == 6 for support in (sa, sb) for row in support):
        assert cnots == 29376
    return {"idx": idx, "architecture": (pa, pb), "k": 68,
            "screen": key, "cnots": cnots, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Stage-1 settlement is not a reliability certificate: parent slot 4 settled 1498/1500 X shots at stage 1 yet failed 102/1500.
# REFUTED: Equal d2s predicts equal circuit reliability: current parent slots 17/24 both had d2s=12, but X failed 11/5 times in 200000 shots and scored 6.758/7.021.
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
