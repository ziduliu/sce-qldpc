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
CONCEPT = ("For the affine Frobenius group C19 semidirect C3, order 57, shapes (1,3)x(1,3), balanced noncommuting supports and exact exclusion of logicals of weight <=4 should improve the 0.400 frontier incumbent's d30s=4 readings; larger distance remains a search hypothesis.")
GROUP = ("The order-39 nonabelian (2,3)x(2,3) bed reached fitness 44.8, motivating another odd-order Frobenius group without assuming transfer across rates. Affine multipliers {1,7,11} modulo 19 give a nonabelian order-57 group with no involutions and n=570.")
SHAPES = ("Target 0.400: A=B=(1,3), n=10*57=570, kb=4*57=228, guaranteed rate=0.400. Both components have 171 coordinates and 57 checks. Parent weight-six logicals occupied sector-1 Acol/Bcol blocks broadly; screen both quantum logical types and retain only measured ranks in the 0.400 band.")
FILLING = ("Four support pairs times five coset-occupancy recipes obey HX/HZ caps of 12 and use at most one binomial per component. Binomial ratios have order 19; exclude three-term subgroup orbits. Disjoint hashes of an isomorphism-invariant full-Tanner degree/four-cycle fingerprint separate every slot. Optimize four admissible fillings per bucket using exact quantum weight<=4 exclusion, component kernel tests through weight 5, dependency witnesses and cycle penalties; the latter are heuristics.")
FAMILY_NAME = "r040_aff19c3_13x13_tanner_bucket_quotient_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    # Affine maps t -> a+b*t over F_19; 7 has multiplicative order 3.
    if q != 57:
        raise ValueError("this affine group has order 57")
    multipliers = (1, 7, 11)
    positions = {b: i for i, b in enumerate(multipliers)}
    els = [(a, b) for b in multipliers for a in range(19)]

    def red(g):
        return (g[0] % 19, g[1] % 19)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + b*c) % 19, (b*d) % 19)

    def inv(g):
        a, b = red(g)
        bi = pow(b, 17, 19)
        return ((-a*bi) % 19, bi)

    def idx(g):
        a, b = red(g)
        return 19*positions[b] + a

    return dict(elements=els, ident=(0, 1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _support_design(idx):
    """Four legal support pairs; no component has two binomial cells."""
    profiles = (
        ((3, 3, 3), (3, 3, 3)),
        ((3, 3, 3), (2, 3, 3)),
        ((2, 3, 3), (2, 3, 3)),
        ((2, 3, 4), (2, 3, 3)),
    )
    a, b = profiles[idx % 4]
    sa, sb = [list(a)], [list(b)]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb, (idx // 4) % 5


def _designed_cell(rng, G, weight, rainbow, phase):
    """Distinct affine labels with controlled multiplier occupancy."""
    from itertools import combinations
    multipliers = (1, 7, 11)
    shift = (rng.randrange(3) + phase) % 3
    pools = [[g for g in G.elements
              if g[1] == multipliers[(shift+b) % 3]]
             for b in range(3)]
    if weight == 2:
        # Every nonidentity ratio inside this coset has order 19.
        return rng.sample(pools[0], 2)
    if weight == 3:
        counts = (1, 1, 1) if rainbow else (2, 1, 0)
    elif weight == 4:
        counts = (2, 1, 1)
    else:
        raise ValueError("unsupported deliberate cell weight")
    for _ in range(96):
        cell = [g for pool, count in zip(pools, counts)
                for g in rng.sample(pool, count)]
        bad = False
        for x, y, z in combinations(cell, 3):
            r = G.mul(G.inv(x), y)
            r2 = G.mul(r, r)
            if G.mul(r2, r) == G.e and G.mul(G.inv(x), z) == r2:
                bad = True
                break
        if not bad:
            return cell
    raise ValueError("could not avoid a three-term subgroup orbit")


def _tanner_bucket(hx, hz, n, buckets):
    """An uncolored full-Tanner invariant: distinct buckets imply nonisomorphism."""
    from collections import Counter
    from hashlib import blake2b
    counts = {}
    row_pairs = []
    degrees = [0] * n
    weights = []
    for row in hx + hz:
        bits = []
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            bits.append(c)
            degrees[c] += 1
            row ^= bit
        weights.append(len(bits))
        keys = []
        for i, x in enumerate(bits):
            for y in bits[i+1:]:
                key = x*n + y
                keys.append(key)
                counts[key] = counts.get(key, 0) + 1
        row_pairs.append(keys)
    check_cycles = [sum(counts[key] - 1 for key in keys)
                    for keys in row_pairs]
    qubit_cycles = [0] * n
    for key, count in counts.items():
        cycles = count*(count-1)//2
        if cycles:
            x, y = divmod(key, n)
            qubit_cycles[x] += cycles
            qubit_cycles[y] += cycles
    fingerprint = (
        tuple(sorted(Counter(zip(weights, check_cycles)).items())),
        tuple(sorted(Counter(zip(degrees, qubit_cycles)).items())),
    )
    digest = blake2b(repr(fingerprint).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _small_logical(rows, stabilizers, n):
    """Exact existence test for a non-stabilizer logical of weight <=4.
    Every such vector is the difference of two subsets of size <=2.
    One representative per syndrome suffices modulo the stabilizer span.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def outside_span(v):
        while v:
            p = v.bit_length() - 1
            if p not in pivots:
                return True
            v ^= pivots[p]
        return False

    cols = _syndrome_columns(rows, n)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        mask = 1 << i
        if syndrome in seen:
            if outside_span(mask ^ seen[syndrome]):
                return True
        else:
            seen[syndrome] = mask
    for i in range(n):
        for j in range(i):
            syndrome = cols[i] ^ cols[j]
            mask = (1 << i) | (1 << j)
            if syndrome in seen:
                if outside_span(mask ^ seen[syndrome]):
                    return True
            else:
                seen[syndrome] = mask
    return False


def _syndrome_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= check
            row ^= bit
    return cols


def _kernel_floor(cols):
    """Exact smallest kernel weight through 5; return 6 if none exists."""
    singles = {}
    for i, s in enumerate(cols):
        if s == 0:
            return 1
        if s in singles:
            return 2
        singles[s] = i
    pairs = {}
    has_four = False
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            if s in singles:
                return 3
            if s in pairs:
                has_four = True
            else:
                pairs[s] = (i, j)
    if has_four:
        return 4
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            for h in range(j):
                if (s ^ cols[h]) in pairs:
                    # With weights <=4 excluded, the matching pair is disjoint.
                    return 5
    return 6


def _dependency_reading(rng, cols, trials=12):
    """Upper-bound witnesses for a component kernel, not quantum distance."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, support)
                    break
                s, v = pivots[p]
                syndrome ^= s
                support ^= v
            if not syndrome:
                best = min(best, support.bit_count())
    return best


def _four_cycle_cost(cols):
    cost = 0
    for i in range(len(cols)):
        for j in range(i):
            overlap = (cols[i] & cols[j]).bit_count()
            cost += overlap * (overlap - 1) // 2
    return cost


def NEW_CTX(rng, G, idx):
    """Optimize a support/coset recipe inside a slot-specific invariant bucket."""
    sa, sb, mode = _support_design(idx)
    rainbow_counts = ((0, 0), (1, 0), (1, 1), (2, 1), (3, 3))[mode]
    best, best_score, best_matrices = None, None, None
    q = G.q
    admitted = 0
    for attempt in range(max(256, 16*N_SLOTS)):
        cells = {}
        for side_number, (side, sup) in enumerate((("A", sa), ("B", sb))):
            triple_columns = [j for j, count in enumerate(sup[0])
                              if count == 3]
            rainbow = set(triple_columns[:rainbow_counts[side_number]])
            cells[side] = [[
                _designed_cell(rng, G, count, j in rainbow,
                               mode + j + side_number + idx//20)
                for j, count in enumerate(sup[0])]]
        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])

        # Isomorphic union Tanner graphs necessarily have the same bucket.
        # Therefore no two emitted slots can be orbit/relabeling copies.
        if _tanner_bucket(hx, hz, n, N_SLOTS) != idx:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (375*n <= 1000*k < 425*n):
            continue
        if _small_logical(hz, hx, n) or _small_logical(hx, hz, n):
            continue

        # Both restrictions come from the scaffold's verified lift.
        ac_full = _syndrome_columns(hx[:q], n)
        ac = [ac_full[(j * B_SHAPE[1]) * q + t]
              for j in range(A_SHAPE[1]) for t in range(q)]
        bc = _syndrome_columns(hz[:q], n)[:B_SHAPE[1]*q]
        fa, fb = _kernel_floor(ac), _kernel_floor(bc)
        da = fa if fa < 6 else _dependency_reading(rng, ac)
        db = fb if fb < 6 else _dependency_reading(rng, bc)
        cycles = _four_cycle_cost(ac) + _four_cycle_cost(bc)
        score = (min(fa, fb), min(da, db), da+db,
                 -cycles, -abs(k-4*q))
        if best_score is None or score > best_score:
            best, best_score = cells, score
            best_matrices = (hx, hz)
        admitted += 1
        if admitted >= 4:
            break
    if best is None:
        raise ValueError("no rate-admissible small-logical-free filling in bucket")
    if not css_ok(*best_matrices):
        raise ValueError("verified construction failed CSS check")
    return {"idx": idx, "cells": best}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, without changing its screened labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Transferring the earlier bed's d30s=8..10 readings to the parent's (1,3)x(2,5) bed fails: all five measured codes had d30s=6, despite component screening and order-14 binomials.
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
