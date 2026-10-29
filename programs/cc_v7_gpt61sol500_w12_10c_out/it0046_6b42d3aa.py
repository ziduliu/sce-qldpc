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
CONCEPT = ("On C5 semidirect C8 with action 2, order 40, shapes (1,3)x(1,4), mixing a normal-C5 rotation with two noncommuting odd-C8 binomial differences should remove the parent's bipartite weight-four component-cycle obstruction and raise its distance readings.")
GROUP = ("Frozen under L1. This bed's five measured codes all have d30s=4. Three odd-quotient binomials give a bipartite degree-six graph on 40 vertices, whereas girth at least six requires at least 62. The same group supplies order-five rotations that break this obstruction; no group superiority is assumed.")
SHAPES = ("Frozen A=(1,3), B=(1,4): n=520, guaranteed k>=240 and rate 6/13, targeting cell 0.450. Component lengths/check counts are 120/40 and 160/40. Parent minimum X witnesses use sector-1 Acol/Bcol blocks across all columns; address their component relations rather than prioritize the parent's k-kb=2 bonus. Verified ranks retain 240<=k<=246.")
FILLING = ("Twenty cards cross four support architectures with five ternary quotient policies. Three-binomial components mix a C5 rotation with odd-C8 differences; all-binomial A gives B two trinomials. Normalize cells to avoid gratuitous column-gauge copies. Up to 192 proposals jointly optimize B short-kernel collisions and A labels. Verified lifts supply exhaustive component-kernel tests through weight four, exact rank, and stabilizer-tested full CSS screening. Four-cycle counts break ties; a bounded fallback keeps the strongest rank-legal finalist, not an assumed distance certificate.")
FAMILY_NAME = "r045_C5sdC8_action2_13x14_normal5_joint4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    # C5 semidirect C8; action 2 has order 4 and 2^8=1 mod 5.
    if not valid_q(q):
        raise ValueError("C5 semidirect C8 requires order 40")
    els = [(a, b) for b in range(8) for a in range(5)]
    action = [pow(2, b, 5) for b in range(8)]

    def red(g):
        return (g[0] % 5, g[1] % 8)

    def mul(g, h):
        return ((g[0] + action[g[1] % 8] * h[0]) % 5,
                (g[1] + h[1]) % 8)

    def inv(g):
        bi = (-g[1]) % 8
        return ((-action[bi] * g[0]) % 5, bi)

    def idx(g):
        a, b = red(g)
        return a + 5 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _four_cycle_count(rows):
    """Four-cycles in one Tanner graph, using the verified bit rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _has_small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight 1..4.

    Enumerate supports of size at most two. Equal syndromes yield
    every commuting support of size at most four. Test differences
    against the opposite stabilizer row space, not merely the kernel.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(v):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    syndrome = [0] * n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndrome[low.bit_length() - 1] ^= bit
            row ^= low

    seen = {0: 0}
    for i, s in enumerate(syndrome):
        v = 1 << i
        old = seen.get(s)
        if old is not None:
            if outside(v ^ old):
                return True
        else:
            seen[s] = v

    for i in range(n):
        vi = 1 << i
        si = syndrome[i]
        for j in range(i):
            s = si ^ syndrome[j]
            v = vi | (1 << j)
            old = seen.get(s)
            if old is not None:
                if outside(v ^ old):
                    return True
            else:
                seen[s] = v
    return False


def NEW_CTX(rng, G, idx):
    """Twenty architectures; screen the matrices actually evaluated."""
    card = idx % 20
    family, mode = divmod(card, 5)
    architectures = (
        ((3, 3, 3), (2, 2, 2, 3)),
        ((2, 3, 3), (2, 2, 2, 3)),
        ((2, 2, 3), (2, 2, 2, 3)),
        ((2, 2, 2), (2, 2, 3, 3)),
    )
    profiles = (
        (0, 1, 2), (0, 1, 3), (0, 2, 3),
        (0, 1, 4), (0, 0, 1),
    )
    aw, bw = architectures[family]
    assert _grid_weights([list(aw)], [list(bw)]) <= 12
    assert _no_dead_lines([list(aw)], [list(bw)])
    q = G.q
    mask = (1 << q) - 1
    e = G.e

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def ternary(profile):
        # Fix one term at identity; do not spend slots on base translates.
        cell = [e]
        for b in profile[1:]:
            pool = [g for g in G.elements
                    if g[1] == b and g not in cell]
            cell.append(rng.choice(pool))
        return cell

    def cage_differences():
        # The odd generators must have different C5 axes.
        # Quotients 1 and 3 avoid the order-four single-cell kernels.
        v = rng.randrange(5)
        w = rng.choice([a for a in range(5) if a != 2*v % 5])
        return [(v, 1), (w, 3), (rng.randrange(1, 5), 0)]

    def fill_A():
        differences = cage_differences()
        row = []
        binomial = 0
        number_bins = aw.count(2)
        for j, count in enumerate(aw):
            if count == 2:
                if number_bins == 1 and mode == 4:
                    d = differences[2]
                else:
                    d = differences[binomial]
                row.append([e, d])
                binomial += 1
            else:
                row.append(ternary(profiles[(mode + j) % 5]))
        return [row]

    # Three-binomial B includes an order-five rotation. Its graph is
    # no longer forced to be bipartite. Two-binomial B has spare budget
    # for a second ternary cell.
    rotation = (1 + mode % 2, 0)
    odd1, odd3 = (0, 1), (1, 3)
    if bw.count(2) == 3:
        bd = [rotation, odd1, odd3]
    elif mode < 3:
        bd = [odd1, odd3]
    else:
        bd = [rotation, odd1]
    tri_positions = [j for j, count in enumerate(bw) if count == 3]

    def fresh_B(attempt):
        row = []
        bindex = 0
        for j, count in enumerate(bw):
            if count == 2:
                row.append([e, bd[bindex]])
                bindex += 1
            elif attempt < 48:
                row.append(ternary(profiles[(mode + 2*j) % 5]))
            else:
                row.append([e] + rng.sample(
                    [g for g in G.elements if g != e], 2))
        return [row]

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            check = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= check
                row ^= bit
        return result

    def component_columns(hx, hz):
        # Extract beta=0 from HX and j=0 from HZ. No independent lift.
        ar = []
        for row in hx[:q]:
            ar.append(sum(
                ((row >> (j * 4*q)) & mask) << (j*q)
                for j in range(3)))
        bm = (1 << (4*q)) - 1
        br = [row & bm for row in hz[:q]]
        return columns(ar, 3*q), columns(br, 4*q)

    def short_profile(cols):
        """Exact kernel existence through weight four.
        5 is a sentinel for exclusion, not a distance estimate.
        Hits count representative collisions, not distinct witnesses.
        """
        seen = {0: 0}
        best, hits = 5, 0

        def insert(s, v):
            nonlocal best, hits
            old = seen.get(s)
            if old is None:
                seen[s] = v
                return
            word = old ^ v
            if not word:
                return
            weight = word.bit_count()
            if weight < best:
                best, hits = weight, 1
            elif weight == best:
                hits += 1

        bits = [1 << i for i in range(len(cols))]
        for i, s in enumerate(cols):
            insert(s, bits[i])
        for i, s in enumerate(cols):
            for j in range(i):
                insert(s ^ cols[j], bits[i] | bits[j])
        return best, hits

    best_b = None
    ranked = []
    clean = []

    for attempt in range(192):
        A = fill_A()
        if best_b is not None and (
                best_b[0][0] == 5 or attempt % 4 != 0):
            B = copy_grid(best_b[1])
            if best_b[0][0] != 5:
                j = rng.choice(tri_positions)
                t = rng.randrange(1, 3)
                cell = B[0][j]
                # Start inside the card's quotient policy, then allow
                # broader mutations to escape a short-relation minimum.
                pool = [g for g in G.elements if g not in cell
                        and (attempt >= 96 or g[1] == cell[t][1])]
                cell[t] = rng.choice(pool)
        else:
            B = fresh_B(attempt)

        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = component_columns(hx, hz)
        pb = short_profile(bc)
        bkey = (pb[0], -pb[1])
        if best_b is None or bkey > best_b[0]:
            best_b = (bkey, copy_grid(B))

        pa = short_profile(ac)
        cycles = _four_cycle_count(hx) + _four_cycle_count(hz)
        preliminary = (
            min(pa[0], pb[0]), pa[0] + pb[0],
            -pa[1] - pb[1], -cycles,
        )
        if len(ranked) == 6 and preliminary < ranked[-1][0][:4]:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 240 <= k <= 246:
            continue
        key = preliminary + (k,)
        candidate = (key, A, B, hx, hz, n, k)
        ranked.append(candidate)
        ranked.sort(key=lambda item: item[0], reverse=True)
        ranked = ranked[:6]

        if pa[0] == pb[0] == 5:
            # Component exclusions do not exclude mixed-sector logicals.
            if not _has_small_logical(hx, hz, n) and not (
                    _has_small_logical(hz, hx, n)):
                clean.append(candidate)
                if len(clean) >= 4:
                    break

    if not ranked:
        raise ValueError("no rank-legal mixed-C5/C8 candidate")

    if clean:
        # Once <=4 is excluded, prefer rate without sacrificing that test.
        chosen = max(clean, key=lambda item: (item[6], item[0][3]))
    else:
        chosen = ranked[0]
        for candidate in ranked:
            _, A, B, hx, hz, n, k = candidate
            if not _has_small_logical(hx, hz, n) and not (
                    _has_small_logical(hz, hx, n)):
                chosen = candidate
                break

    key, A, B, hx, hz, n, k = chosen
    assert css_ok(hx, hz), "verified lift failed CSS commutation"
    return {"idx": idx, "card": card, "k": k, "screen": key,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Odd-C8-only noncommuting B binomials plus four-cycle ranking suppress weight-four X logicals; all five parent measurements instead found d30s=4, with the minimum X witnesses entirely in sector-1 Acol/Bcol blocks.
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
