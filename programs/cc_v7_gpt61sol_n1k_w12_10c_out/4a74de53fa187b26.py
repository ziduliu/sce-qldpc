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
CONCEPT = ("On cyclic Z16, order 16, A=B=(3,9), balanced cubic components with full-rank binary support parity and independently retested kernel-witness selection should produce five distance readings >=12 in cell 0.400. This is a falsifiable measurement target, not a distance certificate.")
GROUP = ("Keep the split-metacyclic Z_m:Z1 construction with trivial action, resizing only m from 37 to 16. Factor 90 gives n=1440. F2[C16]=F2[x]/((x+1)^16), so component row rank three after evaluation at x=1 guarantees full lifted component row rank. The parent C37 bed measured five minima of ten; no distance improvement on this new bed is established.")
SHAPES = ("A=B=(3,9): n=1440, k>=16*6*6=576, guaranteed rate 0.400; q=16 is the largest lift order fitting factor 90. Equal components have 144 variables and 48 checks. Parent weight-ten minima occupied sector-1 Acol/Bcol blocks across both components, with no listed Arow/Brow minima. A third check row changes the algebraic cofactor families instead of reinforcing one block.")
FILLING = ("Six cubic profiles have nine terms per row, three per column, and binary support-parity rank three. Twenty unordered profile pairs are deliberate support designs. Independently optimize verified component cycle geometry, screen actual four-column cofactors and their translated cancellations, search genuine elimination witnesses, and repair labels using surviving witnesses. Independently deepen finalists before pairing. Both quantum row weights are twelve; final verified ranks, CSS and target rate are checked. All sampled readings remain upper readings, not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r040_m16r1s1_39x39_cubic_cofactor_witness"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Same split-metacyclic construction, with only its order resized.
    # Z_16 x| Z_1 has trivial action and order sixteen.
    if not valid_q(q):
        raise ValueError("this cyclic metacyclic bed requires order 16")
    m, r, s = 16, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 9)               # (ra, ca)
B_SHAPE = (3, 9)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Three-row cubic portfolio, with verified witness-directed selection."""
    import itertools
    import math

    q = G.q
    width, height = 9 * q, 3 * q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    mode = (card + cohort) % 4
    diagnostic_seed = rng.randrange(1 << 62)
    forward = [(2, 1, 0), (0, 2, 1), (1, 0, 2)]
    reverse = [(1, 2, 0), (0, 1, 2), (2, 0, 1)]
    profiles = (
        [(1, 1, 1)] * 6 + [(3, 0, 0), (0, 3, 0), (0, 0, 3)],
        [(1, 1, 1)] * 3 + forward + reverse,
        [(1, 1, 1)] * 6 + forward,
        [(1, 1, 1)] * 4 + [(3, 0, 0), (0, 2, 1),
                                   (0, 1, 2), (1, 2, 0), (1, 0, 2)],
        forward * 2 + reverse,
        [(1, 1, 1)] * 3 + [(3, 0, 0), (0, 3, 0),
                                   (2, 0, 1), (0, 2, 1),
                                   (1, 0, 2), (0, 1, 2)],
    )
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (4, 4)]
    pa, pb = plans[card]
    for pattern in profiles:
        assert len(pattern) == 9
        assert all(sum(column) == 3 for column in pattern)
        assert all(sum(column[i] for column in pattern) == 9
                   for i in range(3))
        parity = [sum((column[i] & 1) << j
                      for j, column in enumerate(pattern)) for i in range(3)]
        assert gf2_rank(parity) == 3

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def bits(value):
        while value:
            bit = value & -value
            yield bit.bit_length() - 1
            value ^= bit

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def rotate(poly, shift):
        shift %= q
        if not shift:
            return poly
        return ((poly << shift) & mask) | (poly >> (q - shift))

    def multiply(a, b):
        answer = 0
        for t in bits(b):
            answer ^= rotate(a, t)
        return answer

    def translate(word, shift):
        return sum(rotate((word >> (j*q)) & mask, shift) << (j*q)
                   for j in range(9))

    def inspect(grid, side):
        # The dead dummy side is screening-only, never submitted.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        rows = [row & ((1 << width) - 1) for row in rows]
        assert len(rows) == height
        columns = [0] * width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        assert all(column.bit_count() == 3 for column in columns)

        uses, adjacency = {}, [0] * height
        four = 0
        for column in columns:
            checks = list(bits(column))
            for t, u in enumerate(checks):
                adjacency[u] |= column ^ (1 << u)
                for v in checks[:t]:
                    key = (v, u)
                    old = uses.get(key, 0)
                    four += old
                    uses[key] = old + 1
        triangles = projected = peak = 0
        for u, neighbors in enumerate(adjacency):
            local = 0
            for v in bits(neighbors):
                local += (neighbors & adjacency[v]).bit_count()
                if v > u:
                    triangles += (
                        neighbors & adjacency[v] & (-1 << (v + 1))).bit_count()
            peak = max(peak, local // 2)
            if mode in (1, 3):
                for v in range(u + 1, height):
                    common = (neighbors & adjacency[v]).bit_count()
                    projected += common * (common - 1) // 2
        excess = max(0, triangles - width)
        cost = 10000 * four + excess
        if mode == 1:
            cost += projected / 32.0
        elif mode == 2:
            cost += 3.0 * peak
        elif mode == 3:
            cost += projected / 8.0 + peak
        return {"grid": copy_grid(grid), "rows": rows, "cols": columns,
                "cost": cost, "cycles": (four, excess, peak),
                "key": signature(grid)}

    def algebra(record, exhaustive=False):
        """Actual cyclic cofactors derived from VERIFIED component columns."""
        columns = record["cols"]
        if "words" not in record:
            polys = [[(columns[j*q] >> (i*q)) & mask
                      for j in range(9)] for i in range(3)]
            for j in range(9):
                for t in range(q):
                    expected = sum(rotate(polys[i][j], t) << (i*q)
                                   for i in range(3))
                    assert columns[j*q+t] == expected

            minors = {}
            for a, b in itertools.combinations(range(9), 2):
                minors[a, b] = (
                    multiply(polys[1][a], polys[2][b]) ^
                    multiply(polys[1][b], polys[2][a]))
            determinants = {}
            for a, b, c in itertools.combinations(range(9), 3):
                determinants[a, b, c] = (
                    multiply(polys[0][a], minors[b, c]) ^
                    multiply(polys[0][b], minors[a, c]) ^
                    multiply(polys[0][c], minors[a, b]))
            words = set()
            for support in itertools.combinations(range(9), 4):
                word = 0
                for j in support:
                    other = tuple(c for c in support if c != j)
                    word |= determinants[other] << (j*q)
                if word:
                    assert syndrome(columns, word) == 0
                    words.add(word)
            record["words"] = sorted(words)
            record["d"], record["word"] = width + 1, None
            for word in record["words"]:
                for shift in range(q // 2 + 1):
                    candidate = word if shift == 0 else word ^ translate(word, shift)
                    if candidate and candidate.bit_count() < record["d"]:
                        record["d"], record["word"] = candidate.bit_count(), candidate

        if exhaustive:
            words = record["words"]
            orbits = [[translate(word, t) for t in range(q)] for word in words]
            for a, word in enumerate(words):
                for orbit in orbits[a:]:
                    for other in orbit:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < record["d"]:
                            record["d"], record["word"] = candidate.bit_count(), candidate

    def kernel_probe(columns, seed, trials):
        """Search genuine kernel vectors; never a distance lower bound."""
        local = random.Random(seed)
        order = list(range(width))
        best, witness = width + 1, None
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    if word.bit_count() < best:
                        best, witness = word.bit_count(), word

            light = sorted(basis, key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() < best:
                        best, witness = candidate.bit_count(), candidate
            for a, b, c in itertools.combinations(light[:8], 3):
                candidate = a ^ b ^ c
                if candidate and candidate.bit_count() < best:
                    best, witness = candidate.bit_count(), candidate
            local.shuffle(basis)
            for word in light[:4]:
                weight = word.bit_count()
                for sweep in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    if weight < best:
                        best, witness = weight, word
                    if not changed:
                        break
            if best <= 6:
                break
        assert witness is not None and syndrome(columns, witness) == 0
        return best, witness

    def optimize(profile, side):
        pattern = list(profiles[profile])
        rng.shuffle(pattern)
        support = [[column[i] for column in pattern] for i in range(3)]
        positions = [(i, j) for i in range(3) for j in range(9)
                     if support[i][j]]
        seed = diagnostic_seed + (0 if side == "A" else 1000003)
        bank, geometries = set(), {}

        def fresh():
            return [[rng.sample(G.elements, count) for count in row]
                    for row in support]

        def retain_geometry(record):
            geometries[record["key"]] = record
            if len(geometries) > 10:
                worst = max(geometries, key=lambda k: geometries[k]["cost"])
                del geometries[worst]

        def remember(word):
            if word is not None:
                bank.add(word)
                if len(bank) > 64:
                    keep = sorted(bank, key=int.bit_count)[:48]
                    bank.clear()
                    bank.update(keep)

        def rating(record):
            reading = record["d"]
            for word in bank:
                if word.bit_count() < reading and syndrome(record["cols"], word) == 0:
                    reading = word.bit_count()
            return reading, -record["cost"]

        def deepen(record, trials, offset, exhaustive=False):
            algebra(record, exhaustive)
            d, word = kernel_probe(record["cols"], seed + offset, trials)
            if d < record["d"]:
                record["d"], record["word"] = d, word
            remember(record["word"])

        for _ in range(24):
            retain_geometry(inspect(fresh(), side))
        current = min(geometries.values(), key=lambda r: r["cost"])
        grid = copy_grid(current["grid"])
        steps = 180 + 20 * (cohort % 3)
        for step in range(steps):
            i, j = rng.choice(positions)
            old = list(grid[i][j])
            if step % 9 == 0:
                grid[i][j] = rng.sample(G.elements, len(old))
            else:
                t = rng.randrange(len(old))
                choices = [g for g in G.elements if g not in old]
                grid[i][j][t] = rng.choice(choices)
            proposed = inspect(grid, side)
            delta = proposed["cost"] - current["cost"]
            temperature = q * (5.0 * (1.0 - step / steps) ** 2 + 0.05)
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current = proposed
                retain_geometry(current)
            else:
                grid[i][j] = old

        pool = {}
        for record in geometries.values():
            deepen(record, 12, 17)
            pool[record["key"]] = record

        initial = sorted(pool.values(), key=rating, reverse=True)[:3]
        for record in initial:
            deepen(record, 64, 1009, True)
        current = max(initial, key=rating)

        # Actual old witnesses are checked on each new matrix. This avoids
        # accepting a worse code merely because a short probe missed them.
        for step in range(48):
            surviving = [w for w in bank if syndrome(current["cols"], w) == 0]
            witness = min(surviving, key=int.bit_count) if surviving else current["word"]
            active = set(c // q for c in bits(witness)) if witness else set()
            candidates = [(i, j) for i, j in positions if j in active]
            i, j = rng.choice(candidates if candidates and rng.random() < 0.85
                              else positions)
            grid = copy_grid(current["grid"])
            grid[i][j] = rng.sample(G.elements, support[i][j])
            proposed = inspect(grid, side)
            deepen(proposed, 8, 31)
            pk, ck = rating(proposed), rating(current)
            if pk > ck or (pk[0] == ck[0] and
                           proposed["cost"] <= current["cost"] + q and
                           rng.random() < 0.08):
                current = proposed
            pool[proposed["key"]] = proposed
            if len(pool) > 8:
                worst = min(pool, key=lambda k: rating(pool[k]))
                del pool[worst]

        finalists = sorted(pool.values(), key=rating, reverse=True)[:3]
        for record in finalists:
            deepen(record, 128 + 16 * mode, 700001, True)
            deepen(record, 64, 1700003)
            assert gf2_rank(record["rows"]) == height
            record["effective"] = rating(record)[0]
        finalists.sort(key=lambda r: (r["effective"], -r["cost"]), reverse=True)
        return finalists[:2]

    aa, bb = optimize(pa, "A"), optimize(pb, "B")
    pairs = sorted(((a, b) for a in aa for b in bb),
                   key=lambda ab: (
                       min(ab[0]["effective"], ab[1]["effective"]),
                       ab[0]["effective"] + ab[1]["effective"],
                       -ab[0]["cost"] - ab[1]["cost"]), reverse=True)
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1440 and k == 576):
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx + hz) == 12
        assert css_ok(hx, hz)
        return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
                "screen_k": k, "profiles": (pa, pb),
                "component_readings": (a["effective"], b["effective"]),
                "component_cycles": (a["cycles"], b["cycles"])}
    raise ValueError("no verified three-row cubic pair in cell 0.400")


def CELL(rng, G, side, i, j, ctx):
    """Return the optimized full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal positive-cell rolling with uniform Z37 labels reliably avoids short sector-1 logicals; the parent instead has many d2s=2 slots and all five measured distances equal six, with minimum witnesses confined to sector 1.
# REFUTED: Global-gap cofactor-twelve selection on C37 (2,6)x(2,6) produces five readings >=12; the shown experiment instead measured 8,8,10,8,10 and fitness 31.36.
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
