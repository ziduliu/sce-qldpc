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
CONCEPT = ("On A4 x C3, order 36, shapes (1,3)x(1,4), transferring unavoidable short component cycles from B to a connected binomial A component, while lowering gate count, should improve worse-basis circuit reliability over the parent's 3.2178.")
GROUP = ("Frozen by L1. Order-six elements lie in V4 x C3, so an all-order-six binomial component can split into three components. Two distinct order-six generators plus an order-three generator outside V4 x C3 connect all 36 lift positions; verified component rank checks test this construction.")
SHAPES = ("Frozen (1,3)x(1,4): n=468, k>=216 and guaranteed rate 6/13. Component lengths are 108 and 144. Exact full-code ranks retain k=216..222 in cell 0.450. Parent Z failures were 112..137/250 shots, versus much lower X failed-shot rates, with diffuse Acol/Bcol enrichment and widespread sector-1 weight-three witnesses.")
FILLING = ("Twenty briefs combine A counts (2,2,2), B counts (3,3,2,2) or (3,3,3,1), five triple-label policies, and two order-three placements. A uses two distinct order-six differences and one order-three difference outside V4 x C3. B has only two binomials, both order six, or a systematic monomial and three triples. Verified lifts require component ranks 35 and 36, reject duplicate component syndrome columns, and shortlist exact weight-three/four kernel multiplicities. Final selection counts actual weight-three/four logicals modulo the opposite stabilizer span. Both profiles use 15120 CNOTs, below the parent's measured 15336..17496; improved circuit pL remains the falsifiable objective.")
FAMILY_NAME = "r045_A4xC3_13x14_triangle_transfer_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C3 requires order 36")
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(3)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 3,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + ((g[4] + h[4]) % 3,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % 3,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _designed_cell(rng, G, count, target_order, mode, orders):
    """Avoid order-two differences; prescribe binomial cycle lengths."""
    for _ in range(512):
        cell = rng.sample(G.elements, count)
        # Distinct projected permutations prevent pure-central pairs.
        if len({tuple(g[:4]) for g in cell}) != count:
            continue
        relative = [
            orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))]
            for a in range(count) for b in range(a + 1, count)
        ]
        if min(relative) < 3:
            continue
        if count == 2:
            if relative[0] != target_order:
                continue
        else:
            central_span = len({g[4] for g in cell})
            wanted = 3 if mode == 0 else 2
            if central_span != wanted:
                continue
        return cell
    raise ValueError("could not realize cell design")


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Twenty structural briefs; all screening uses the verified lift."""
    q = G.q
    brief = idx % 20
    profile = brief % 2
    mode = (brief // 2) % 5
    placement = brief // 10
    pa = (2, 2, 2)
    pb = (3, 3, 2, 2) if profile == 0 else (3, 3, 3, 1)
    assert _grid_weights([list(pa)], [list(pb)]) <= W_MAX

    orders = _orders(G)
    v4 = [g for g in G.elements
          if g[4] == 0 and orders[G.idx(g)] <= 2]

    def quotient(g):
        # Canonical projected coset in A4/V4; central C3 is ignored.
        return min(tuple(G.mul(h, g)[:4]) for h in v4)

    quotient_id = quotient(G.e)
    quotients = {G.idx(g): quotient(g) for g in G.elements}
    six = {}
    for g in G.elements:
        if orders[G.idx(g)] == 6:
            six.setdefault(tuple(g[:4]), []).append(g)
    six_classes = sorted(six)
    outside_three = [
        g for g in G.elements
        if orders[G.idx(g)] == 3
        and quotients[G.idx(g)] != quotient_id
        and g[4] == placement
    ]

    def generators():
        # Distinct inverse families prevent parallel binomial edges.
        families = rng.sample(six_classes, 2)
        return [rng.choice(six[f]) for f in families]

    def binomial(d, side):
        g = G.pick(rng)
        if side == "A":
            # Actual left-component row displacement is d.
            return [g, G.mul(d, g)]
        # Actual right-component row displacement is d.
        return [g, G.mul(G.inv(d), g)]

    def triple(number):
        for _ in range(1024):
            cell = rng.sample(G.elements, 3)
            central_span = len({g[4] for g in cell})
            quotient_span = len({quotients[G.idx(g)] for g in cell})

            if mode == 0:
                if central_span != 3 or quotient_span != 2:
                    continue
            elif mode == 1:
                if central_span != 2 or quotient_span != 2:
                    continue
            elif mode == 2:
                wanted = 2 if number == 0 else 3
                if central_span != 2 or quotient_span != wanted:
                    continue
            elif mode == 3:
                # Deliberately test pure-central-coordinate triples:
                # unlike the parent, involutory differences are allowed.
                if central_span != 1 or quotient_span != 2:
                    continue
            else:
                if central_span != 3 or quotient_span < 2:
                    continue

            a = G.mul(G.inv(cell[0]), cell[1])
            b = G.mul(G.inv(cell[0]), cell[2])
            if G._key(G.mul(a, b)) == G._key(G.mul(b, a)):
                continue
            if mode < 3:
                relatives = [
                    G.mul(G.inv(cell[i]), cell[j])
                    for i in range(3) for j in range(i + 1, 3)
                ]
                if any(orders[G.idx(g)] < 3 for g in relatives):
                    continue
            return cell
        raise ValueError("triple policy could not be realized")

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def components(hx, hz):
        # A is extracted from the Bcol0 X-check block.
        mask = (1 << q) - 1
        ar = []
        for row in hx[:q]:
            compact = 0
            for j in range(3):
                compact |= ((row >> (j * 4 * q)) & mask) << (j * q)
            ar.append(compact)
        # Brow0/Acol0 Z checks contain the actual right-lifted B.
        bmask = (1 << (4 * q)) - 1
        br = [row & bmask for row in hz[:q]]
        return ar, br

    def kernel_stats(cols):
        """Exact weight-three/four kernel counts for unique columns."""
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        singles = set(cols)
        pairs = {}
        triples = collisions = cycles = 0
        weights = [c.bit_count() for c in cols]
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                triples += int(s in singles)
                old = pairs.get(s, 0)
                collisions += old
                pairs[s] = old + 1
                overlap = (weights[i] + weights[j] - s.bit_count()) // 2
                cycles += overlap * (overlap - 1) // 2
        assert triples % 3 == 0 and collisions % 3 == 0
        return triples // 3, collisions // 3, cycles

    def logical_stats(checks, stabilizers, n):
        """Exact logical multiplicities through weight four.

        Unique nonzero columns exclude weights one and two. Reducing
        every unit vector modulo the opposite check-row span lets
        syndrome collisions be classified as stabilizer or logical.
        Each weight-three/four support has three pair partitions.
        """
        cols = columns(checks, n)
        if 0 in cols or len(set(cols)) != n:
            return None
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items(), reverse=True)
        residuals = []
        for i in range(n):
            v = 1 << i
            for p, row in ordered:
                if (v >> p) & 1:
                    v ^= row
            residuals.append(v)

        singles = {s: residuals[i] for i, s in enumerate(cols)}
        totals = {}
        classes = {}
        triples = fours = 0
        for i, x in enumerate(cols):
            ri = residuals[i]
            for j in range(i):
                s = x ^ cols[j]
                residue = ri ^ residuals[j]
                single = singles.get(s)
                if single is not None and residue != single:
                    triples += 1
                key = (s, residue)
                old_total = totals.get(s, 0)
                old_class = classes.get(key, 0)
                fours += old_total - old_class
                totals[s] = old_total + 1
                classes[key] = old_class + 1
        assert triples % 3 == 0 and fours % 3 == 0
        return triples // 3, fours // 3

    shortlist = []
    exact_seen = set()
    for trial in range(144):
        ad = generators()
        ad.append(rng.choice(outside_three))
        A = [[binomial(d, "A") for d in ad]]
        bd = iter(generators())
        brow = []
        triple_number = 0
        try:
            for count in pb:
                if count == 1:
                    brow.append([G.e])
                elif count == 2:
                    brow.append(binomial(next(bd), "B"))
                else:
                    brow.append(triple(triple_number))
                    triple_number += 1
        except ValueError:
            continue
        B = [brow]

        identity = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row
        )
        if identity in exact_seen:
            continue
        exact_seen.add(identity)

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = components(hx, hz)
        # Connected binomial A has one row dependency. Full-rank B
        # avoids introducing a second augmentation-sector deficit.
        if gf2_rank(ar) != q - 1 or gf2_rank(br) != q:
            continue
        sa = kernel_stats(columns(ar, 3 * q))
        sb = kernel_stats(columns(br, 4 * q))
        if sa is None or sb is None:
            continue

        key = (max(sa[0], sb[0]), sa[0] + sb[0], sb[0],
               max(sa[1], sb[1]), sa[1] + sb[1],
               max(sa[2], sb[2]), sa[2] + sb[2])
        shortlist.append((key, A, B, hx, hz, n))
        shortlist.sort(key=lambda item: item[0])
        del shortlist[5:]

    best = None
    best_key = None
    for component_key, A, B, hx, hz, n in shortlist:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (216 <= k <= 222):
            continue
        lx = logical_stats(hx, hz, n)
        lz = logical_stats(hz, hx, n)
        if lx is None or lz is None:
            continue
        fx = _same_basis_cycles(hx)
        fz = _same_basis_cycles(hz)
        key = (max(lx[0], lz[0]), lx[0] + lz[0],
               max(lx[1], lz[1]), lx[1] + lz[1],
               max(fx / len(hx), fz / len(hz)),
               fx + fz, component_key, k)
        if best_key is None or key < best_key:
            best_key = key
            best = (A, B, hx, hz)

    if best is None:
        raise ValueError("no connected triangle-transfer filling passed screening")
    A, B, hx, hz = best
    assert css_ok(hx, hz)
    assert 6 * sum(row.bit_count() for row in hx + hz) == 15120
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing four-cycles alone did not protect this A4 x C3 bed: fitness 3.2178, Z failures 112..137/250, and widespread weight-three sector-1 witnesses despite mostly stage-1 settlement.
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
