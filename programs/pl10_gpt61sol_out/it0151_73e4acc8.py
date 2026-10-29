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
CONCEPT = ("S3 x C7, order 42, shapes (1,2)x(2,6): noncommuting A trinomials and long-order B differences should improve worse-basis circuit memory in cell 0.300 through balanced short-boundary screening.")
GROUP = ("Jump to a permutation direct-product construction, genuinely different from the order-39 Frobenius parent. S3 x C7 supplies order-14 and order-21 relative elements, avoiding involutory binomial differences. The related centre-seven order-42 bed scored about 4.6 on (1,3)x(2,4); this untried component allocation tests a new bed, not a demonstrated group advantage.")
SHAPES = ("Target the weaker 0.300 frontier cell. A=(1,2) has 84 variables and 42 checks; B=(2,6) has 252 variables and 84 checks. At q=42, n=588 and kb=168 give guaranteed rate 2/7. Verified ranks retain 168<=k<=190. Parent slot4 mildly enriched Bcol1, while slot8 implicated Acol2/Bcol0; these rare failures do not establish one isolated block to repair.")
FILLING = ("Twenty contracts cross five balanced B support profiles with four relative-order policies. A has two noncommuting trinomials; B has column totals 2..4 and row totals nine. HX weights are <=10 and HZ weights twelve, with 25704 six-round CNOTs. Fresh labels compete using verified full ranks, component connectivity, kernel-word<=4 exclusion, anchored triple boundaries, short cycles, capped sampled kernel witnesses and quantum logical<=4 exclusion. The objective is lowering worse-basis failures, not increasing distance: parent slot4 and slot14 shared k=59,w=10,d2s=18 but scored 6.734 and 7.283.")
FAMILY_NAME = "r030_S3xC7_12x26_noncommuting_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    from itertools import permutations
    if q != 42:
        raise ValueError("S3 x C7 requires order 42")
    # (p0,p1,p2,c): a permutation of three points and an additive C7 label.
    perms = list(permutations(range(3)))
    els = [p + (c,) for c in range(7) for p in perms]

    def red(g):
        return tuple(int(x) for x in g[:3]) + (int(g[3]) % 7,)

    def mul(g, h):
        return (g[h[0]], g[h[1]], g[h[2]], (g[3] + h[3]) % 7)

    def inv(g):
        p = tuple(g[:3])
        return tuple(p.index(i) for i in range(3)) + ((-g[3]) % 7,)

    def idx(g):
        g = red(g)
        return 6 * g[3] + perms.index(g[:3])

    return dict(elements=els, ident=(0, 1, 2, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _relative_order(G, g):
    """Exact order under the validated group law."""
    value = G.e
    for order in range(1, G.q + 1):
        value = G.mul(value, g)
        if G._key(value) == G._key(G.e):
            return order
    raise ValueError("element order exceeds the group order")


def NEW_CTX(rng, G, idx):
    """Twenty support/order contracts, optimized using verified lifts."""
    # Column count pairs. All five profiles have B-row totals (9,9).
    # They differ in triple/zero cells and degree-two/four columns.
    profiles = (
        ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (2,1), (1,2), (1,2)),
        ((3,0), (3,0), (0,3), (0,3), (2,1), (1,2)),
        ((1,1), (2,2), (2,1), (2,1), (1,2), (1,2)),
        ((0,2), (2,0), (2,2), (2,2), (2,1), (1,2)),
    )
    design, mode = idx % 20, idx // 20
    profile, policy = divmod(design, 4)
    sa = [[3, 3]]
    sb = [[column[i] for column in profiles[profile]]
          for i in range(2)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert sum(sb[0]) == sum(sb[1]) == 9

    orders = {7: [], 14: [], 21: []}
    for g in G.elements:
        order = _relative_order(G, g)
        if order in orders:
            orders[order].append(g)
    assert all(orders.values())

    bin_sites = [(i, j) for i in range(2) for j in range(6)
                 if sb[i][j] == 2]
    central_site = bin_sites[(profile + mode) % len(bin_sites)]
    a_kinds = ((0, 0), (0, 1), (1, 1), (0, 1))[policy]
    if mode % 3 == 1:
        a_kinds = (a_kinds[0], 1 - a_kinds[1])
    elif mode % 3 == 2:
        a_kinds = (1 - a_kinds[0], a_kinds[1])

    def triple(kind):
        # kind 0: reflection/rotation; kind 1: two reflections.
        # Long C7 coordinates avoid involutory relative elements.
        for _ in range(96):
            u = rng.choice(orders[14])
            v = rng.choice(orders[14 if kind else 21])
            if G._key(G.mul(u, v)) != G._key(G.mul(v, u)):
                return [G.e, u, v]
        raise ValueError("failed to draw a noncommuting trinomial")

    def draw():
        # Fix whole-cell gauges on the single-row A component.
        A = [[triple(a_kinds[j]) for j in range(2)]]
        B = [[[] for _ in range(6)] for _ in range(2)]
        for i in range(2):
            for j in range(6):
                count = sb[i][j]
                if not count:
                    continue
                base = G.pick(rng)
                if count == 1:
                    relative = [G.e]
                elif count == 2:
                    if policy == 0:
                        order = 14
                    elif policy == 1:
                        order = 21
                    elif policy == 2:
                        order = 14 if (i + j + mode) % 2 == 0 else 21
                    else:
                        order = (7 if (i, j) == central_site else
                                 14 if (i + j + mode) % 2 == 0 else 21)
                    relative = [G.e, rng.choice(orders[order])]
                else:
                    relative = triple((policy + i + j + mode) % 2)
                cell = [G.mul(base, g) for g in relative]
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return A, B

    q = G.q
    block_mask = (1 << q) - 1

    def components(hx, hz):
        """Project the unequal components from the verified full lift."""
        ar = []
        for r in range(q):
            row = hx[r]  # A-row zero, B-column zero.
            ar.append((row & block_mask) |
                      (((row >> (6 * q)) & block_mask) << q))
        bmask = (1 << (6 * q)) - 1
        br = [hz[a * q + r] & bmask
              for a in range(2) for r in range(q)]
        return ar, br

    def columns(rows, ncols):
        result = [0] * ncols
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def connected(cols, nchecks):
        parent = list(range(nchecks))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for syndrome in cols:
            if not syndrome:
                return False
            first = syndrome & -syndrome
            a = first.bit_length() - 1
            syndrome ^= first
            while syndrome:
                bit = syndrome & -syndrome
                b = bit.bit_length() - 1
                parent[root(b)] = root(a)
                syndrome ^= bit
        return len({root(r) for r in range(nchecks)}) == 1

    def four(rows):
        return sum(v * (v - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for v in [(row & other).bit_count()])

    def six(rows):
        """Exact Tanner six-cycles, including repeated intersections."""
        total = 0
        nr = len(rows)
        overlap = [[0] * nr for _ in range(nr)]
        for i in range(nr):
            for j in range(i):
                overlap[i][j] = overlap[j][i] = (
                    rows[i] & rows[j]).bit_count()
        for i in range(nr):
            for j in range(i):
                x = overlap[i][j]
                if not x:
                    continue
                common = rows[i] & rows[j]
                for h in range(j):
                    y, z = overlap[j][h], overlap[i][h]
                    if y and z:
                        triple = (common & rows[h]).bit_count()
                        total += (x * y * z -
                                  triple * (x + y + z) + 2 * triple)
        return total

    def boundary_screen(cols):
        """Exact <=4 exclusion and orbit-anchored triple exposure.

        Anchoring avoids storing millions of triple syndromes for B.
        Pair/triple matches warn of actual short words; the boundary
        statistic is not a decoder or circuit-failure certificate.
        """
        seen = {0}
        for value in cols:
            if value in seen:
                return None
            seen.add(value)
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return None
                seen.add(value)
        boundary = 0
        for anchor in range(0, len(cols), q):
            for i, x in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ x
                for j in range(i):
                    if j == anchor:
                        continue
                    value = pair ^ cols[j]
                    if value in seen:
                        boundary += 200
                    weight = value.bit_count()
                    if weight <= 4:
                        boundary += (5 - weight) ** 2
        return boundary

    def sampled_kernel(cols):
        """True kernel witnesses, but only an upper-bound search."""
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(40):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = syndrome, word
                        break
                    old_s, old_w = pivots[p]
                    syndrome ^= old_s
                    word ^= old_w
                if not syndrome:
                    best = min(best, word.bit_count())
                    for other in basis:
                        best = min(best, (word ^ other).bit_count())
                    basis.append(word)
            if best <= 6:
                break
        return best

    def short_logical(checks, stabilizers, n):
        """Exact non-stabilizer logical existence through weight four."""
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

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    for batch in range(3):
        pool = []
        for trial in range(160):
            A, B = draw()
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = components(hx, hz)
            fa, fb = four(ar), four(br)
            # B has twice as many component checks as A.
            coarse = (max(2 * fa, fb), 2 * fa + fb)
            pool.append((coarse, trial, A, B, hx, hz, n, ar, br))
        pool.sort(key=lambda item: (item[0], item[1]))

        finalists = []
        eligible = 0
        for item in pool:
            coarse, trial, A, B, hx, hz, n, ar, br = item
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 588 and 168 <= k <= 190):
                continue
            ac, bc = columns(ar, 2 * q), columns(br, 6 * q)
            if not connected(ac, q) or not connected(bc, 2 * q):
                continue
            eligible += 1
            ba = boundary_screen(ac)
            bb = boundary_screen(bc) if ba is not None else None
            if ba is not None and bb is not None:
                da, db = sampled_kernel(ac), sampled_kernel(bc)
                cycle_a, cycle_b = six(ar), six(br)
                distance = min(da, db)
                # Coarse protection warning, then normalized boundary
                # exposure. A has two anchored blocks versus B's six.
                # Sampled distances remain upper bounds, not certificates.
                key = (int(distance < 10), max(3 * ba, bb), 3 * ba + bb,
                       max(2 * cycle_a, cycle_b),
                       2 * cycle_a + cycle_b, abs(da - db),
                       -min(distance, 14), coarse, trial)
                finalists.append((key, item, k))
            if eligible >= 12:
                break

        finalists.sort(key=lambda entry: entry[0])
        for key, item, k in finalists:
            _, _, A, B, hx, hz, n, _, _ = item
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            assert 6 * sum(row.bit_count() for row in hx + hz) == 25704
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": design, "mode": mode,
                    "k": k, "screen": key}

    raise ValueError("no S3 x C7 contract passed target-rate screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the screened full cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
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
