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
CONCEPT = ("On D148 = Z74 semidirect-inversion Z2, order 148, shapes (1,3)x(1,3), mixed trinomials selected against verified cofactor and circuit witnesses should raise five-code fitness above the 0.400 frontier's 40.0. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Frozen inversion-action metacyclic Z74:Z2(s=73), q=148. This parent scored 6.4086 and all five measured minima were four. A proposed cell 1+x^p*y+x^(p+d)*y has cyclic determinant 1+x^d+x^(-d), invertible modulo x^74-1 because 3 does not divide 74. Keep the group while testing this full-rank element model.")
SHAPES = ("Frozen A=B=(1,3): n=1480, k>=592, guaranteed rate 0.400; both components have 444 variables and 148 checks. The size window allows q<=149 and this frozen group has q=148. Parent minima were entirely sector-1 Acol/Bcol witnesses, frequently involving Acol1; none used Arow/Brow. Invertible cells should give full check-row ranks and k=592 instead of the parent's occasional small rank excess.")
FILLING = ("Three distinct terms in every cell give both row weights exactly twelve. Twenty cards pair six automorphism-separated relative-gap profiles. Independently shortlist 32 fresh normalized mixed-trinomial fillings per side using cofactors and binomial multiples extracted from verified components. Search shifted cofactor cancellations, independently retest randomized circuit cancellations, and prefer final pairs with no full-matrix commuting operator through weight four. Exact ranks, CSS and weight checks gate the result. Valid fallbacks preserve slots; component readings and these structural screens do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040f_m74r2s73_13x13_mixed_trinomials_cofactor12"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 148

def custom_ops(q):
    # metacyclic Z_74 x| Z_2: y x y^-1 = x^73 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 74, 2, 73
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Independently select mixed-trinomial components on verified lifts."""
    import itertools
    import math

    q = G.q
    m = q // 2
    width = 3 * q
    mask = (1 << m) - 1
    card = idx % 20
    phase = (idx // 20) % 3
    diagnostic_seed = rng.randrange(1 << 62)
    units = [u for u in range(1, m) if math.gcd(u, m) == 1]

    def gap_key(gaps):
        # Scaling, inversion and cell permutation cannot produce another
        # chosen gap profile. This is not a full Tanner-isomorphism test.
        return min(tuple(sorted(min((u * d) % m, (-u * d) % m)
                                for d in gaps)) for u in units)

    preferred = ((1, 5, 17), (1, 7, 23), (1, 9, 21),
                 (1, 4, 19), (1, 8, 25), (1, 6, 14))
    palette = list(preferred)
    palette.extend((1, d, e) for d in range(2, m // 2)
                   for e in range(d + 1, m // 2))
    profiles, used = [], set()
    for gaps in palette:
        if any(sum(s * d for s, d in zip(signs, gaps)) % m == 0
               for signs in itertools.product((-1, 1), repeat=3)):
            continue
        if any((2 * a) % m in (b % m, (-b) % m)
               for a in gaps for b in gaps if a != b):
            continue
        key = gap_key(gaps)
        if key not in used:
            used.add(key)
            profiles.append(gaps)
        if len(profiles) == 6:
            break
    assert len(profiles) == 6
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    pa, pb = plans[card]

    def fresh(profile):
        gaps = profiles[profile]
        origins = (0, rng.randrange(m), rng.randrange(m))
        cells = [[G.e, (p, 1), ((p + d) % m, 1)]
                 for p, d in zip(origins, gaps)]
        assert all(len({G._key(g) for g in cell}) == 3
                   for cell in cells)
        return [cells]

    def columns(rows, nc):
        answer = [0] * nc
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                answer[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return answer

    def rotate(poly, shift):
        shift %= m
        if not shift:
            return poly
        return ((poly << shift) & mask) | (poly >> (m - shift))

    def multiply(a, b):
        result = 0
        while b:
            bit = b & -b
            result ^= rotate(a, bit.bit_length() - 1)
            b ^= bit
        return result

    def translate(word, shift):
        result = 0
        for j in range(6):
            result |= rotate((word >> (j * m)) & mask,
                             shift) << (j * m)
        return result

    def component(grid, side):
        # Dummy zero sides are used only to extract verified components.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        rows = [row & ((1 << width) - 1) for row in rows]
        original = columns(rows, width)

        if side == "A":
            # Reindex both reflection fibers by a -> -a. This exposes
            # the commuting right-rotation action of the actual left lift.
            rmap = list(range(m)) + [m + (-a) % m for a in range(m)]
            cols = []
            for j in range(3):
                for b in range(2):
                    for a in range(m):
                        source = j * q + b * m + (a if b == 0 else (-a) % m)
                        syndrome = original[source]
                        value = 0
                        while syndrome:
                            bit = syndrome & -syndrome
                            value |= 1 << rmap[bit.bit_length() - 1]
                            syndrome ^= bit
                        cols.append(value)
        else:
            # The verified right lift already has a common left-rotation
            # coordinate on both fibers.
            cols = original

        polys = [(cols[j * m] & mask, cols[j * m] >> m)
                 for j in range(6)]
        for j, (a, b) in enumerate(polys):
            assert all(cols[j * m + t] ==
                       (rotate(a, t) | (rotate(b, t) << m))
                       for t in range(m)), "cyclic coordinates disagree with lift"

        minors = {}
        for a, b in itertools.combinations(range(6), 2):
            minors[a, b] = (multiply(polys[a][0], polys[b][1]) ^
                            multiply(polys[b][0], polys[a][1]))
        words = []
        for a, b, c in itertools.combinations(range(6), 3):
            word = (minors[b, c] << (a * m) |
                    minors[a, c] << (b * m) |
                    minors[a, b] << (c * m))
            if not word:
                continue
            syndrome = 0
            remaining = word
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length() - 1]
                remaining ^= bit
            assert syndrome == 0, "cofactor is not a verified kernel word"
            words.append(word)

        reading, short = width + 1, 0
        for word in words:
            w = word.bit_count()
            reading = min(reading, w)
            short += int(w < 12)
            for shift in range(1, m // 2 + 1):
                candidate = word ^ translate(word, shift)
                if candidate:
                    w = candidate.bit_count()
                    reading = min(reading, w)
                    short += int(w < 12)

        # Exact Tanner C4 count; projected triangle excess is a heuristic.
        pair_uses = {}
        adjacency = [0] * q
        four = 0
        for syndrome in cols:
            checks = []
            value = syndrome
            while value:
                bit = value & -value
                checks.append(bit.bit_length() - 1)
                value ^= bit
            assert len(checks) == 3
            for t, a in enumerate(checks):
                adjacency[a] |= syndrome ^ (1 << a)
                for b in checks[:t]:
                    key = (b, a)
                    previous = pair_uses.get(key, 0)
                    four += previous
                    pair_uses[key] = previous + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            remaining = neighbors & (-1 << (a + 1))
            while remaining:
                bit = remaining & -remaining
                b = bit.bit_length() - 1
                triangles += (neighbors & adjacency[b] &
                              (-1 << (b + 1))).bit_count()
                remaining ^= bit
        return {"grid": grid, "cols": cols, "words": words,
                "d": reading, "short": short,
                "cycles": (four, max(0, triangles - width))}

    def kernel4_reading(cols):
        """Return a witnessed weight <=4, or a sentinel; exact exclusion."""
        singles = set(cols)
        if 0 in singles:
            return 1
        if len(singles) != len(cols):
            return 2
        seen = singles.copy()
        for i, a in enumerate(cols):
            for b in cols[:i]:
                value = a ^ b
                if value in singles:
                    return 3
                if value in seen:
                    return 4
                seen.add(value)
        return len(cols) + 1

    def geometry(record):
        four, projected = record["cycles"]
        if phase == 1:
            return (-projected, -four)
        if phase == 2:
            return (-(four + projected / 8.0), -four)
        return (-four, -projected)

    def merit(record):
        return (record["d"], -record["short"]) + geometry(record)

    def cofactor_retest(record):
        record["d"] = min(record["d"], kernel4_reading(record["cols"]))
        orbits = [[translate(word, t) for t in range(m)]
                  for word in record["words"]]
        short = record["short"]
        for a, word in enumerate(record["words"]):
            for orbit in orbits[a:]:
                for other in orbit:
                    candidate = word ^ other
                    if candidate:
                        w = candidate.bit_count()
                        record["d"] = min(record["d"], w)
                        short += int(w < 12)
        record["short"] = short
        record["orbits"] = [word for orbit in orbits for word in orbit]

    def circuit_probe(record, seed, trials):
        """Search actual verified kernel words; readings are upper bounds."""
        diagnostic = random.Random(seed)
        cols = record["cols"]
        reading = record["d"]
        order = list(range(width))
        for _ in range(trials):
            if reading <= 8:
                break
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                syndrome, word = cols[v], 1 << v
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
                    reading = min(reading, word.bit_count())

            light = sorted(basis, key=int.bit_count)[:12]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            for word in light[:3]:
                for other in record["orbits"]:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            diagnostic.shuffle(basis)
            for word in light[:4]:
                weight = word.bit_count()
                for sweep in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    reading = min(reading, weight)
                    if not changed:
                        break
        return reading

    def select_side(profile, side):
        pool, signatures = [], set()
        for _ in range(32):
            grid = fresh(profile)
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in grid[0])
            if signature in signatures:
                continue
            signatures.add(signature)
            record = component(grid, side)
            pool.append(record)
            pool.sort(key=merit, reverse=True)
            del pool[6:]
        if not pool:
            raise ValueError("empty verified trinomial component pool")

        for record in pool:
            cofactor_retest(record)
        pool.sort(key=merit, reverse=True)
        finalists = pool[:2]
        offset = 0 if side == "A" else 100003
        for record in finalists:
            record["d"] = circuit_probe(
                record, diagnostic_seed + offset, 16)
        finalists.sort(key=merit, reverse=True)
        for record in finalists:
            # A separate longer probe can only lower the first reading.
            record["d"] = circuit_probe(
                record, diagnostic_seed + 700001 + offset,
                48 if phase == 2 else 32)
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa = select_side(pa, "A")
    bb = select_side(pb, "B")

    def pair_merit(pair):
        a, b = pair
        ac, bc = a["cycles"], b["cycles"]
        return (min(a["d"], b["d"]), a["d"] + b["d"],
                -a["short"] - b["short"],
                -max(ac[0], bc[0]), -ac[0] - bc[0],
                -ac[1] - bc[1])

    pairs = sorted(((a, b) for a in aa for b in bb),
                   key=pair_merit, reverse=True)
    fallback, selected = None, None
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1480 and k == 592):
            continue
        candidate = (A, B, hx, hz, k, a, b)
        if fallback is None:
            fallback = candidate
        if kernel4_reading(columns(hx, n)) <= 4:
            continue
        if kernel4_reading(columns(hz, n)) <= 4:
            continue
        selected = candidate
        break

    cleared_full4 = selected is not None
    if selected is None:
        selected = fallback
    if selected is None:
        raise ValueError("no full-rank trinomial pair in cell 0.400")
    A, B, hx, hz, k, a, b = selected
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
            "screen_k": k, "gap_profiles": (profiles[pa], profiles[pb]),
            "component_readings": (a["d"], b["d"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "cleared_full_kernel4": cleared_full4}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without post-screen randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled counts and uniform D148 labels reliably suppress weight-four sector-1 logicals; all five measured parent minima were four, while most screened slots returned two.
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
