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
CONCEPT = ("GL(2,4), order 180, shapes (1,1)x(1,1): determinant-controlled four-/five-term square maps selected by stabilizer-tested logical witnesses should exceed the 0.100 frontier fitness 51.9 while retaining measured rates below 0.125.")
GROUP = ("Keep the invertible-2x2-matrix-over-a-finite-field construction family. Square shapes would give n=96 at the parent's order 48, below the size window, so change the field parameter from F3 to F4. GL(2,4) has (16-1)*(16-4)=180 elements and gives n=360. The parent's GL(2,3) bed scored 37.05; the new order/shape bed is untried. F4 is a field, not integers modulo four.")
SHAPES = ("Target cell 0.100 with A=(1,1), B=(1,1): n=360, kb=0, and equal 180-variable/180-check components. Each LP sector contains 180 qubits. Positive low-rate homology must come from actual rank deficits; verified ranks enforce 0<8*k<n. Parent measured weight-12 X and Z minima occupied sector-one Acol/Bcol blocks broadly, with none in sector two. Replace its rate-one-half A component and unequal sectors rather than relabeling those witnesses.")
FILLING = ("Twenty determinant-composition cards prescribe four or five terms in each square cell. Sample fresh internal labels, exclude every relative involution, and vary preference for longer relative orders. Both CSS check weights are the sum of the two cell sizes, at most ten. Exact lifted ranks gate the target rate. Full-CSS information-set witnesses, pair sums and kernel descent reject stabilizers exactly; deeper finalist probes and exact quantum weight<=4 tests select the winner. Witness readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r010_GL2F4_11x11_square_det_logical20"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 180

def custom_ops(q):
    # Same general-linear matrix-group rule, now over F4.
    # 0,1,2,3 encode 0,1,u,u+1, with u^2=u+1.
    # Addition is XOR; this is not arithmetic modulo four.
    if not valid_q(q):
        raise ValueError("GL(2,4) requires order 180")
    times = ((0,0,0,0), (0,1,2,3),
             (0,2,3,1), (0,3,1,2))
    reciprocals = (0,1,3,2)
    els = [(a,b,c,d) for a in range(4) for b in range(4)
           for c in range(4) for d in range(4)
           if times[a][d] ^ times[b][c]]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x & 3 for x in g)

    def mul(g, h):
        a,b,c,d = g
        e,f,u,v = h
        return (times[a][e] ^ times[b][u],
                times[a][f] ^ times[b][v],
                times[c][e] ^ times[d][u],
                times[c][f] ^ times[d][v])

    def inv(g):
        a,b,c,d = g
        t = reciprocals[times[a][d] ^ times[b][c]]
        # Minus equals plus in characteristic two.
        return (times[t][d], times[t][b],
                times[t][c], times[t][a])

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 1)               # (ra, ca)
B_SHAPE = (1, 1)               # (rb, cb)

_F4_TIMES = ((0,0,0,0), (0,1,2,3),
             (0,2,3,1), (0,3,1,2))

# Counts in determinant classes 1,u,u+1. These change quotient
# polynomials, not just the translation of a fixed labeled cell.
_FOUR_QUOTAS = ((2,2,0), (2,1,1), (3,1,0))
_FIVE_QUOTAS = ((3,1,1), (3,2,0), (2,2,1))
_SQUARE_CARDS = (
    # Six unordered four/four composition pairs.
    (4,0,4,0), (4,0,4,1), (4,0,4,2),
    (4,1,4,1), (4,1,4,2), (4,2,4,2),
    # Nine four/five pairs.
    (4,0,5,0), (4,0,5,1), (4,0,5,2),
    (4,1,5,0), (4,1,5,1), (4,1,5,2),
    (4,2,5,0), (4,2,5,1), (4,2,5,2),
    # Five five/five pairs.
    (5,0,5,0), (5,0,5,1), (5,0,5,2),
    (5,1,5,1), (5,1,5,2),
)


def _determinant(g):
    return _F4_TIMES[g[0]][g[3]] ^ _F4_TIMES[g[1]][g[2]]


def _element_orders(G):
    orders = {}
    for g in G.elements:
        h = G.e
        for order in range(1, G.q + 1):
            h = G.mul(h, g)
            if h == G.e:
                orders[g] = order
                break
        if g not in orders:
            raise ValueError("element order computation failed")
    return orders


def _square_cell(rng, G, pools, quota, orders, style):
    """Fresh internal geometry; equal or involution-separated terms forbidden."""
    cosets = [d for d, amount in enumerate(quota, 1)
              for _ in range(amount)]
    for _ in range(48):
        rng.shuffle(cosets)
        cell, inverses = [], []
        for d in cosets:
            legal, preferences = [], []
            for g in pools[d]:
                relative = [orders[G.mul(hi, g)] for hi in inverses]
                if any(order <= 2 for order in relative):
                    continue
                legal.append(g)
                # Bias, not a hard order restriction: avoid forcing
                # every cell into a small cyclic subgroup.
                if style == 0:
                    preferences.append(1)
                elif style == 1:
                    preferences.append(1 + sum(order >= 5
                                               for order in relative))
                else:
                    preferences.append(1 + sum(order == 15
                                               for order in relative))
            if not legal:
                break
            g = rng.choices(legal, weights=preferences, k=1)[0]
            cell.append(g)
            inverses.append(G.inv(g))
        if len(cell) == len(cosets):
            assert len(cell) == len(set(cell))
            return cell
    raise ValueError("could not complete determinant-constrained cell")


def _square_probe(cols, stabilizers, rng, trials, ceiling=None):
    """Full-CSS logical witnesses, including both LP sectors.

    Every accepted word has zero check syndrome and lies outside the
    stabilizer span. High readings still do not certify distance.
    """
    width = len(cols)
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    pivot_order = sorted(basis, reverse=True)

    def logical_class(word):
        # Canonical remainder is needed when comparing quotient classes.
        for p in pivot_order:
            if (word >> p) & 1:
                word ^= basis[p]
        return word

    best = width + 1 if ceiling is None else ceiling
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                cls = logical_class(word)
                words.append((word, cls))
                if cls:
                    best = min(best, word.bit_count())

        short = sorted(words, key=lambda item: item[0].bit_count())[:32]
        for i, (word, cls) in enumerate(short):
            for other, oc in short[:i]:
                if cls ^ oc:
                    best = min(best, (word ^ other).bit_count())

        seeds = sorted((item for item in words if item[1]),
                       key=lambda item: item[0].bit_count())[:6]
        rng.shuffle(words)
        for word, cls in seeds:
            weight = word.bit_count()
            for _ in range(4):
                changed = False
                for other, oc in words:
                    trial, tc = word ^ other, cls ^ oc
                    tw = trial.bit_count()
                    if tc and tw < weight:
                        word, cls, weight = trial, tc, tw
                        changed = True
                if not changed:
                    break
            best = min(best, weight)
        if best <= 4:
            break
    return best


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _small_logical_4(checks, stabilizers, width):
    """Exact quantum-logical existence test through weight four.

    Equal syndromes of supports of size <=2 cover every commuting
    support through four. One representative suffices: if a collision
    is a stabilizer, it preserves the quotient class of that syndrome.
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

    cols = _component_columns(checks, width)
    seen = {0: 0}
    bits = [1 << i for i in range(width)]
    for i, s in enumerate(cols):
        old = seen.get(s)
        if old is None:
            seen[s] = bits[i]
        elif outside(bits[i] ^ old):
            return True

    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            v = bits[i] | bits[j]
            old = seen.get(s)
            if old is None:
                seen[s] = v
            elif outside(v ^ old):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Jointly optimize one square-map determinant-composition card."""
    assert G.q == 180
    card = idx % len(_SQUARE_CARDS)
    wa, pa, wb, pb = _SQUARE_CARDS[card]
    qa = (_FOUR_QUOTAS if wa == 4 else _FIVE_QUOTAS)[pa]
    qb = (_FOUR_QUOTAS if wb == 4 else _FIVE_QUOTAS)[pb]
    sa, sb = [[wa]], [[wb]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    pools = {d: [g for g in G.elements if _determinant(g) == d]
             for d in (1, 2, 3)}
    orders = _element_orders(G)
    finalists, accepted, seen = [], 0, set()

    def merit(item):
        d = min(item["dx"], item["dz"])
        return (item["k"] * min(d, 60)**2,
                d, item["dx"] + item["dz"], -item["cycles"])

    for attempt in range(80):
        # Each card has fixed counts/compositions; later batches explore
        # genuinely fresh internal labelings, not transformed templates.
        A = [[_square_cell(rng, G, pools, qa, orders, card % 3)]]
        B = [[_square_cell(rng, G, pools, qb, orders, (card + 1) % 3)]]
        key = (tuple(sorted(A[0][0])), tuple(sorted(B[0][0])))
        if key in seen:
            continue
        seen.add(key)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 8 * k < n):
            continue

        xc = _component_columns(hz, n)
        zc = _component_columns(hx, n)
        dx = _square_probe(xc, hx, rng, 4)
        dz = _square_probe(zc, hz, rng, 4)
        cycles = (_component_four_cycles(xc)
                  + _component_four_cycles(zc))
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    xc=xc, zc=zc, dx=dx, dz=dz, cycles=cycles)
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[4:]
        accepted += 1
        if accepted >= 18:
            break

    if not finalists:
        raise ValueError("no positive-k square filling below rate 0.125")

    for item in finalists:
        item["dx"] = _square_probe(
            item["xc"], item["hx"], rng, 32, item["dx"])
        item["dz"] = _square_probe(
            item["zc"], item["hz"], rng, 32, item["dz"])
    finalists.sort(key=merit, reverse=True)

    # Exact small-logical exclusion, followed by another independent
    # witness rescreen. Never replace a measured witness by a higher one.
    for item in finalists[:3]:
        item["clean"] = not (
            _small_logical_4(item["hz"], item["hx"], item["n"]) or
            _small_logical_4(item["hx"], item["hz"], item["n"]))
        if not item["clean"]:
            item["dx"] = min(item["dx"], 4)
            item["dz"] = min(item["dz"], 4)
        else:
            item["dx"] = _square_probe(
                item["xc"], item["hx"], rng, 24, item["dx"])
            item["dz"] = _square_probe(
                item["zc"], item["hz"], rng, 24, item["dz"])

    chosen = max(finalists[:3], key=lambda item: (
        int(item["clean"]),) + merit(item))
    A, B = chosen["A"], chosen["B"]
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert css_ok(hx, hz)
    assert n == 360 and 0 < k and 8 * k < n
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert all(len(cell) == len(set(cell))
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection": merit(chosen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: C9 monomial voltage rows with isolated odd trinomial upgrades and exact weight<=3 logical rejection did not exceed distance four: all five measured parent codes had k=283 and d30s=4, with minima entirely in sector 1.
# REFUTED: Saturated weight-12 checks and balanced four-term A cells on GL(2,3) (1,2)x(2,5) eliminate weight-12 sector-one logicals; all five measured parent codes instead retained d30s=12 with both types confined to sector one.
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
