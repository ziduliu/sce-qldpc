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
CONCEPT = ("For cyclic C25, order 25, shapes (2,4)x(2,4), check weights at most ten and suppression of small component syndrome ambiguities should improve worse-basis circuit fitness above the Dic7 parent's 6.6789.")
GROUP = ("Jump from nonabelian Dic7 of order 28 to cyclic C25 of order 25, an untried bed in BEDS. The C29 bed measured 7.1770 with weight-nine/ten checks, motivating a lower-load cyclic experiment without proving a group advantage. ord_5(2)=4 and ord_25(2)=20 give two nontrivial binary cyclotomic factors; actual lifted ranks, not this algebra alone, determine acceptance.")
SHAPES = ("Keep equal (2,4) components: each has 50 checks on 100 variables; n=500 and kb=100 give guaranteed rate 0.200. The size window permits q<=29, but this family fixes q=25. Require verified 100<=k<=110. Parent slot15's Z map implicated Acol0 across B columns, while slot36 implicated Acol3 and Zchk:Acol3/Brow1; these changing, mild concentrations motivate protecting both components globally.")
FILLING = ("Twenty designs cross five low-load support architectures with four internal geometries canonicalized against cyclic automorphisms, cell translations, row/column permutations and side exchange. Extra harness batches extend the geometry book rather than relabel saved codes. Binomial differences have order 25; trinomials have distinct internal directions. Optimize verified component lifts, then prefer no kernel word through weight six, balanced small triple boundaries and a coarse sampled protection floor. Final verified matrices screen rate, CSS and logical weight<=4, retaining a rate-legal fallback if the optional logical screen finds no passing pair. Both check-weight sums are <=10; six-round CNOT counts are 21600..22500. Parent slot15 had d2s=14 but 87 Z failures and s=6.098, versus slot19's d2s=12, seven worst-basis failures and s=6.992: reducing circuit exposure and ambiguity, not increasing d2s alone, is the hypothesis.")
FAMILY_NAME = "r020_C25_24x24_lowload_geometry_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 25

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This cyclic parameter set requires order 25")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _c25_brief(G, idx):
    """An indexed support/geometry contract, independent of voltage search."""
    q = G.q
    profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),  # column 3, rows 6/6
        ((1, 1, 1, 2), (2, 2, 2, 1)),  # column 3, rows 5/7
        ((1, 1, 2, 2), (2, 2, 1, 2)),  # one column 4, rows 6/7
        ((3, 0, 2, 1), (0, 3, 1, 2)),  # column 3, rows 6/6
    )
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2), (0, 3))
    design = idx % 20
    architecture = design // 4
    a, b = pairs[architecture]
    supports = [[list(row) for row in profiles[p]] for p in (a, b)]
    assert _grid_weights(*supports) <= min(10, W_MAX)
    assert _no_dead_lines(*supports)

    units = [u for u in range(1, q) if u % 5]
    directions = [d for d in range(1, (q + 1) // 2) if d % 5]
    wanted = 4 * (idx // 20) + design % 4
    grng = random.Random(250019 + 997 * architecture)

    def direction(d):
        d %= q
        return min(d, q - d)

    def canonical(grids):
        # Internal cell geometry is invariant under its translation.
        # Exclude the obvious common automorphism/permutation orbits.
        best = None
        for u in units:
            sides = []
            for grid in grids:
                normalized = []
                for row in grid:
                    cells = []
                    for cell in row:
                        cells.append(min(
                            tuple(sorted(u * (v - anchor) % q
                                         for v in cell))
                            for anchor in cell) if cell else ())
                    normalized.append(cells)
                columns = list(zip(*normalized))
                normal = tuple(sorted(columns))
                swapped = tuple(sorted((y, x) for x, y in columns))
                sides.append(min(normal, swapped))
            key = tuple(sorted(sides))
            if best is None or key < best:
                best = key
        return best

    seen = set()
    serial = -1
    for attempt in range(10000):
        geometries = []
        for counts in supports:
            doubles = [(i, j) for i in range(2) for j in range(4)
                       if counts[i][j] == 2]
            steps = dict(zip(doubles, grng.sample(directions, len(doubles))))
            grid = []
            for i, row in enumerate(counts):
                cells = []
                occupied = {steps[i, j] for j in range(4)
                            if (i, j) in steps}
                for j, count in enumerate(row):
                    if count < 2:
                        cell = [0] if count else []
                    elif count == 2:
                        cell = [0, steps[i, j]]
                    else:
                        for _ in range(2000):
                            d, e = grng.sample(units, 2)
                            internal = {direction(d), direction(e),
                                        direction(e - d)}
                            if (len(internal) == 3
                                    and (e - d) % 5
                                    and not internal.intersection(occupied)):
                                cell = [0, d, e]
                                break
                        else:
                            raise ValueError("cannot realize trinomial geometry")
                        occupied.update(internal)
                    cells.append(cell)
                grid.append(cells)
            geometries.append(grid)
        key = canonical(geometries)
        if key in seen:
            continue
        seen.add(key)
        serial += 1
        if serial == wanted:
            return supports, geometries, (architecture, wanted)
    raise ValueError("insufficient unused internal geometry contracts")


def _c25_cell(rng, template):
    """Change inter-cell voltage without changing the indexed geometry."""
    shift = rng.randrange(25)
    return [((shift + d) % 25,) for d in template]


def _c25_rows(G, grid, side):
    """Exact component lift with a zero partner; no independent group lift."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (4 * G.q)) - 1
    return [row & mask for row in rows]


def _c25_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _c25_boundary(rows, q):
    """Exact kernel-weight<=6 exclusion and per-block triple boundaries.

    Boundary counts are structural surrogates, not circuit failure rates.
    """
    columns = _c25_columns(rows, 4 * q)
    seen = {0}
    for value in columns:
        if value in seen:
            return 1, 10**9, 10**9
        seen.add(value)
    for i, x in enumerate(columns):
        for j in range(i):
            value = x ^ columns[j]
            if value in seen:
                return 1, 10**9, 10**9
            seen.add(value)

    boundary = [0] * 4
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in seen:
                    return 1, 10**9, 10**9
                seen.add(syndrome)
                weight = syndrome.bit_count()
                if weight < 5:
                    penalty = (5 - weight) ** 2
                    for block in {i // q, j // q, h // q}:
                        boundary[block] += penalty
    return 0, max(boundary), sum(boundary)


def _c25_short_logical(checks, stabilizers, n):
    """Exact existence test for non-stabilizer logicals of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    # Linear canonical remainders modulo the stabilizer row space.
    residues = [1 << c for c in range(n)]
    for p in sorted(pivots):
        tail = pivots[p] ^ (1 << p)
        value = 0
        while tail:
            bit = tail & -tail
            value ^= residues[bit.bit_length() - 1]
            tail ^= bit
        residues[p] = value

    columns = _c25_columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(columns):
        value = residues[i]
        old = seen.get(syndrome)
        if old is not None and old != value:
            return True
        seen[syndrome] = value
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            value = residues[i] ^ residues[j]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
    return False


def _dic_component_rows(G, A, B):
    """Extract component matrices from the evaluator's exact lift."""
    hx, hz, n = lifted_matrices(G, A, B)
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(row)
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ar, br, hx, hz, n


def _dic_cycles(rows, width):
    """Exact four-cycles; six-cycle proxy is exact when four=0."""
    adjacent = [0] * len(rows)
    degrees = [0] * width
    four = 0
    for i, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            degrees[bit.bit_length() - 1] += 1
            bits ^= bit
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap * (overlap - 1) // 2
            if overlap:
                adjacent[i] |= 1 << j
                adjacent[j] |= 1 << i
    triangles = 0
    for i, neighbors in enumerate(adjacent):
        later = neighbors >> (i + 1)
        while later:
            bit = later & -later
            j = i + 1 + bit.bit_length() - 1
            triangles += ((neighbors & adjacent[j]) >> (j + 1)).bit_count()
            later ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6 for d in degrees)
    return four, max(0, triangles - stars)


def _dic_small_kernel(rows, width):
    """Exactly determine whether the component has kernel weight <=4."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    if 0 in columns:
        return 1
    if len(set(columns)) != width:
        return 2
    singles = set(columns)
    pairs = set()
    found_four = False
    for a in range(width):
        for b in range(a):
            s = columns[a] ^ columns[b]
            if s in singles:
                return 3
            if s in pairs:
                found_four = True
            pairs.add(s)
    return 4 if found_four else 5


def _dic_kernel_scan(rows, width, seed, trials=48):
    """Sample actual kernel words; returned distances are upper bounds."""
    srng = random.Random(seed)
    best = width + 1
    small = set()
    for _ in range(trials):
        rr = list(rows)
        order = list(range(width))
        srng.shuffle(order)
        pivots = []
        rank = 0
        for c in order:
            bit = 1 << c
            hit = next((r for r in range(rank, len(rr))
                        if rr[r] & bit), None)
            if hit is None:
                continue
            rr[rank], rr[hit] = rr[hit], rr[rank]
            pivot = rr[rank]
            for r in range(len(rr)):
                if r != rank and rr[r] & bit:
                    rr[r] ^= pivot
            pivots.append(c)
            rank += 1
            if rank == len(rr):
                break
        occupied = set(pivots)
        for c in order:
            if c in occupied:
                continue
            v = 1 << c
            for p, row in zip(pivots, rr[:rank]):
                if (row >> c) & 1:
                    v |= 1 << p
            weight = v.bit_count()
            best = min(best, weight)
            if weight <= 10:
                small.add(v)
        if best <= 4:
            break
    return best, len(small)


def NEW_CTX(rng, G, idx):
    """Optimize fresh inter-cell voltages within an indexed C25 contract."""
    import math
    supports, geometries, brief = _c25_brief(G, idx)
    q, width = G.q, 4 * G.q
    sa, sb = supports
    assert _grid_weights(sa, sb) <= min(10, W_MAX)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def optimize(side, templates):
        sites = [(i, j) for i in range(2) for j in range(4)
                 if templates[i][j]]
        pool, seen = [], set()

        def draw():
            return [[_c25_cell(rng, cell) for cell in row]
                    for row in templates]

        def evaluate(grid):
            rows = _c25_rows(G, grid, side)
            four, six = _dic_cycles(rows, width)
            deficit = 2 * q - gf2_rank(rows)
            energy = (100 * four + six / 6.0) / q + 20 * deficit
            return energy, four, six, deficit, rows

        def offer(grid, result, serial):
            energy, four, six, deficit, rows = result
            if deficit:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            pool.append(((four, six), serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: (item[0], item[1]))
            del pool[6:]

        for restart in range(4):
            grid = draw()
            current = evaluate(grid)
            offer(grid, current, 1000 * restart)
            best_grid, best_result = clone(grid), current
            for step in range(144):
                i, j = rng.choice(sites)
                old = grid[i][j]
                # Shift the whole cell: preserve its deliberate geometry,
                # but change the component's inter-cell voltages.
                shift = rng.randrange(1, q)
                grid[i][j] = [((g[0] + shift) % q,) for g in old]
                proposed = evaluate(grid)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.975 ** step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if current[:4] < best_result[:4]:
                        best_grid, best_result = clone(grid), current
                    if step % 12 == 0:
                        offer(grid, current, 1000 * restart + step + 1)
                else:
                    grid[i][j] = old
            offer(best_grid, best_result, 1000 * restart + 999)

        # Preserve the contract even when optional annealing misses a
        # full-rank endpoint. No support repair or orbit copying occurs.
        if not pool:
            for attempt in range(192):
                grid = draw()
                offer(grid, evaluate(grid), 10000 + attempt)
                if len(pool) >= 4:
                    break
        if not pool:
            raise ValueError("no full-rank C25 component in contract")

        finalists = []
        for cycles, serial, grid, rows in pool:
            bad, worst_boundary, total_boundary = _c25_boundary(rows, q)
            distance_upper, multiplicity = _dic_kernel_scan(
                rows, width,
                250003 + 1009 * idx + 53 * serial + (side == "B"),
                trials=64)
            finalists.append(dict(
                grid=grid, rows=rows, cycles=cycles, serial=serial,
                bad=bad, boundary=(worst_boundary, total_boundary),
                d=distance_upper, multiplicity=multiplicity))
        return finalists

    ap = optimize("A", geometries[0])
    bp = optimize("B", geometries[1])
    options = []
    for a in ap:
        for b in bp:
            da, db = min(a["d"], 12), min(b["d"], 12)
            a4, a6 = a["cycles"]
            b4, b6 = b["cycles"]
            key = (
                a["bad"] + b["bad"],
                int(min(da, db) < 10),
                a4 + b4,
                max(a["boundary"][0], b["boundary"][0]),
                a["boundary"][1] + b["boundary"][1],
                -min(da, db), abs(da - db),
                a["multiplicity"] + b["multiplicity"],
                a6 + b6, a["serial"], b["serial"])
            options.append((key, a, b))
    options.sort(key=lambda item: item[0])

    fallback = None
    examined = 0
    for key, a, b in options:
        A, B = a["grid"], b["grid"]
        ar, br, hx, hz, n = _dic_component_rows(G, A, B)
        # Independently optimized zero-partner components must equal
        # the components actually present in the final quantum lift.
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 100 <= k <= 110:
            continue
        assert n == 500 and css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(10, W_MAX)
        candidate = {
            "idx": idx, "brief": brief, "k": k, "screen": key,
            "cells": {"A": clone(A), "B": clone(B)},
            "small_logical_safe": False}
        if fallback is None:
            fallback = candidate
        if (not _c25_short_logical(hx, hz, n)
                and not _c25_short_logical(hz, hx, n)):
            candidate["small_logical_safe"] = True
            return candidate
        examined += 1
        if examined >= 8:
            break

    if fallback is not None:
        return fallback
    raise ValueError("no verified C25 pair remains in cell 0.200")


def CELL(rng, G, side, i, j, ctx):
    """Return the independently selected full cell, without new draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and near-bound k alone do not ensure balanced memory reliability: parent slot 21 failed 100/46750 X shots versus 101/150500 Z shots, despite k-kb=4 and diffuse block enrichment.
# REFUTED: Larger d2s guarantees a better circuit score: Dic7 slot15 had d2s=14 and s=6.098 with 87 Z failures, while slot19 had d2s=12 and s=6.992 with seven worst-basis failures, all at 200000 shots.
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
