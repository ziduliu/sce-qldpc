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
CONCEPT = ("Cyclic Z23, order 23, encoded as Z1:Z23, shapes (2,3)x(2,4): degree-four columns on both components and joint logical-witness optimization should yield five measured d30s readings >=14 in cell 0.150; this is a falsifiable search-reading hypothesis.")
GROUP = ("Keep the exact metacyclic construction, parameters and encoding. The parent's fitness is 25.1888, with measured Z witnesses at 12-13 versus X witnesses at 17. Retaining q=23 isolates shape and filling effects; the new shape fits without changing the group.")
SHAPES = ("Target 0.150: n=16*23=368, kb=23*(3-2)*(4-2)=46 and guaranteed rate 1/8. A has 69 component variables/46 checks and B has 92/46. Parent minima were sector-1 Z witnesses spread across A columns, especially Bcol1-3. Reducing A's component rate from one-half to one-third strengthens the limiting side; verified ranks require 47<=k<=64.")
FILLING = ("Enumerate canonical degree-four-column support pairs, not row/column permutation copies. A rows weigh at most eight and B rows exactly eight, so HX<=12 and HZ=12; cells contain 0..4 distinct elements and no component line is dead. Optimize both actions equally using verified lifts, screen both component kernels through weight six, and deepen stabilizer-tested sector-1 logical reduction. A deterministic bank conservatively discards equal combined Tanner fingerprints. A legal fallback is explicitly not a distance certificate.")
FAMILY_NAME = "r015_m1r23s1_23x24_joint_deg4_descent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # metacyclic Z_1 x| Z_23: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 1, 23, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)

from functools import lru_cache
from itertools import combinations_with_replacement
from collections import Counter


@lru_cache(maxsize=1)
def _joint_support_cards():
    """Canonical component row/column orbits; retain distinct architectures."""
    def profiles(width):
        out = []
        for top in combinations_with_replacement(range(5), width):
            flipped = tuple(sorted(4 - x for x in top))
            if top > flipped:
                continue
            if width == 3:
                if not 4 <= sum(top) <= 8:
                    continue
            elif sum(top) != 8:
                continue
            out.append(top)
        return out

    cards = []
    for a in profiles(3):
        for b in profiles(4):
            entries = a + b
            zeros = sum(x in (0, 4) for x in entries)
            priority = (
                -zeros,
                min(min(x, 4 - x) for x in entries),
                -sum(abs(x - 2) for x in entries),
                -abs(sum(a) - 6),
            )
            cards.append((priority, a, b))
    cards.sort(reverse=True)
    if len(cards) < 20:
        raise ValueError("fewer than twenty canonical support cards")
    return tuple((a, b) for _, a, b in cards)


def _joint_graph_signature(hx, hz, n):
    """Invariants of the combined untyped Tanner graph.

    Unequal signatures prove nonisomorphism. Equal signatures are
    conservatively discarded without asserting that they are isomorphic.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    overlaps = Counter()
    adjacency = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                overlaps[(min(weights[i], weights[j]),
                          max(weights[i], weights[j]), overlap)] += 1
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap

    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    neighborhoods = Counter()
    for col in cols:
        profile = []
        while col:
            bit = col & -col
            profile.append(weights[bit.bit_length() - 1])
            col ^= bit
        neighborhoods[tuple(sorted(profile))] += 1

    # Weighted triangles distinguish designs sharing degree/overlap counts.
    triangles = 0
    for i, neighbors in enumerate(adjacency):
        for j, ij in neighbors.items():
            if j >= i:
                continue
            for h, ih in neighbors.items():
                if h < j:
                    triangles += ij * ih * adjacency[j].get(h, 0)

    return (tuple(sorted(Counter(weights).items())),
            tuple(sorted(overlaps.items())),
            tuple(sorted(neighborhoods.items())),
            triangles)


def _select_joint_ctx(rng, G, idx):
    """Prescribed count architecture, followed by bounded label optimization.
    Every matrix used for screening comes from lifted_matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1

    cards = _joint_support_cards()
    at, bt = cards[idx % len(cards)]
    sa = [list(at), [4 - x for x in at]]
    sb = [list(bt), [4 - x for x in bt]]
    assert q == 23 and (ra, ca, rb, cb) == (2, 3, 2, 4)
    assert all(sum(sa[i][j] for i in range(ra)) == 4
               for j in range(ca))
    assert all(sum(sb[i][j] for i in range(rb)) == 4
               for j in range(cb))
    assert max(map(sum, sa)) <= 8 and tuple(map(sum, sb)) == (8, 8)
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def columns(rows, length):
        out = [0] * length
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def components(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        # Extract beta=0 from HX and j=0 from HZ, preserving the
        # verified lift's actions. These are the two component matrices.
        ar = []
        for i in range(ra):
            for r in range(q):
                row = hx[(i * cb) * q + r]
                compact = 0
                for j in range(ca):
                    compact |= ((row >> (j * cb * q)) & mask) << (j * q)
                ar.append(compact)
        bm = (1 << (cb * q)) - 1
        br = [row & bm for row in hz[:rb * q]]
        return hx, hz, n, columns(ar, ca * q), columns(br, cb * q)

    def four_cycles(cols):
        total = 0
        for i, x in enumerate(cols):
            for j in range(i):
                t = (x & cols[j]).bit_count()
                total += t * (t - 1) // 2
        return total

    def short_kernel(cols):
        """Return a kernel witness of weight <=6, or None.
        Collisions among all subsets of size <=3 cover every such word."""
        seen = {0: 0}

        def insert(s, v):
            old = seen.get(s)
            if old is not None:
                return old ^ v
            seen[s] = v
            return None

        length = len(cols)
        for i in range(length):
            hit = insert(cols[i], 1 << i)
            if hit is not None:
                return hit
        for i in range(length):
            for j in range(i):
                hit = insert(cols[i] ^ cols[j], (1 << i) | (1 << j))
                if hit is not None:
                    return hit
        for i in range(length):
            for j in range(i):
                s = cols[i] ^ cols[j]
                v = (1 << i) | (1 << j)
                for h in range(j):
                    hit = insert(s ^ cols[h], v | (1 << h))
                    if hit is not None:
                        return hit
        return None

    def row_basis(rows):
        basis = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    basis[p] = row
                    break
                row ^= old
        return basis

    def outside_span(v, basis):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    def logical_estimate(cols, opposite_basis, side):
        """Sample component kernel words, rejecting LP stabilizers.
        The returned positive weight is an upper bound from actual witnesses."""
        best = [ca * cb * q + ra * rb * q + 1]

        def consider(v):
            w = v.bit_count()
            if not v or w >= best[0]:
                return
            if side == "A":
                for beta in range(cb):
                    full = 0
                    for j in range(ca):
                        full |= ((v >> (j * q)) & mask) << (
                            (j * cb + beta) * q)
                    if outside_span(full, opposite_basis):
                        best[0] = w
                        return
            else:
                for j in range(ca):
                    full = v << (j * cb * q)
                    if outside_span(full, opposite_basis):
                        best[0] = w
                        return

        order = list(range(len(cols)))
        rounds = 24
        for _ in range(rounds):
            rng.shuffle(order)
            pivots = {}
            words = []
            for i in order:
                s, v = cols[i], 1 << i
                while s:
                    p = s.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (s, v)
                        break
                    s ^= old[0]
                    v ^= old[1]
                if not s:
                    words.append(v)
                    consider(v)
            # XOR descent preserves the exact component kernel. Every
            # candidate is still tested against the actual LP stabilizers.
            words = list(set(words))
            for sweep in range(3):
                words.sort(key=int.bit_count)
                changed = False
                for a in range(len(words)):
                    v = words[a]
                    weight = v.bit_count()
                    for b in range(a):
                        trial = v ^ words[b]
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            v, weight = trial, tw
                            changed = True
                    words[a] = v
                    consider(v)
                if not changed:
                    break
            words.sort(key=int.bit_count)
            for i, x in enumerate(words):
                for j in range(i):
                    consider(x ^ words[j])
            small = words[:12]
            for i, x in enumerate(small):
                for j in range(i):
                    xy = x ^ small[j]
                    for h in range(j):
                        consider(xy ^ small[h])
            # Additional starts can expose reductions missed by a
            # shortest-first basis sweep; these remain actual witnesses.
            for start in words[:10]:
                v = start
                for sweep in range(4):
                    improved = v
                    weight = v.bit_count()
                    for other in words:
                        trial = v ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            improved, weight = trial, tw
                    if improved == v:
                        break
                    v = improved
                    consider(v)
            if best[0] <= 6:
                break
        return best[0] if best[0] <= ca * cb * q + ra * rb * q else 0

    targets = {
        "A": [(i, j) for i in range(ra) for j in range(ca) if sa[i][j]],
        "B": [(i, j) for i in range(rb) for j in range(cb) if sb[i][j]],
    }
    winner = None
    reserve = None
    accepted = 0

    for attempt in range(32):
        A = [[rng.sample(G.elements, count) for count in row] for row in sa]
        B = [[rng.sample(G.elements, count) for count in row] for row in sb]
        cells = {"A": A, "B": B}
        data = components(A, B)
        cycle = [four_cycles(data[3]), four_cycles(data[4])]

        # The shape strengthens A; neither remaining component may be
        # ignored. Mutations change labels only, never the term budgets.
        for step in range(32):
            side = "A" if step % 2 == 0 else "B"
            which = 0 if side == "A" else 1
            i, j = rng.choice(targets[side])
            cell = cells[side][i][j]
            t = rng.randrange(len(cell))
            old = cell[t]
            cell[t] = rng.choice([g for g in G.elements if g not in cell])
            trial = components(A, B)
            cost = four_cycles(trial[3 + which])
            if cost < cycle[which] or (
                    cost == cycle[which] and rng.random() < 0.6):
                data = trial
                cycle[which] = cost
            else:
                cell[t] = old

        hx, hz, n, ac, bc = data
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Strictly inside [0.125,0.175); avoid the lower boundary tie.
        # Generic augmentation-rank cases give k=51,54,55 or 60, but
        # only the ranks of these actual verified matrices are trusted.
        if n != 368 or not 47 <= k <= 64:
            continue

        short_a = short_kernel(ac)
        short_b = short_kernel(bc)
        wa = short_a.bit_count() if short_a is not None else 7
        wb = short_b.bit_count() if short_b is not None else 7
        reserve_key = (min(wa, wb), wa + wb, -sum(cycle), k)
        if reserve is None or reserve_key > reserve[0]:
            reserve = (reserve_key, A, B)
        if short_a is not None or short_b is not None:
            continue

        dz = logical_estimate(ac, row_basis(hz), "A")
        dx = logical_estimate(bc, row_basis(hx), "B")
        if not dz or not dx:
            continue
        d = min(dx, dz)
        key = (k * d * d / n, d, dx + dz, -sum(cycle))
        if winner is None or key > winner[0]:
            winner = (key, A, B)
        accepted += 1
        if accepted >= 8:
            break

    chosen = winner if winner is not None else reserve
    if chosen is None:
        raise ValueError("no positive-k candidate in the requested rate band")
    A, B = chosen[1], chosen[2]
    hx, hz, _ = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B}}


@lru_cache(maxsize=1)
def _joint_design_bank(G):
    """Deterministic search with invariant-distinct final submissions."""
    bank = []
    used = set()
    for idx in range(N_SLOTS):
        local_rng = random.Random(BASE_SEED * 10007 + idx + 368150)
        chosen = None
        for retry in range(3):
            try:
                ctx = _select_joint_ctx(local_rng, G, idx)
            except ValueError:
                continue
            A, B = ctx["cells"]["A"], ctx["cells"]["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            signature = _joint_graph_signature(hx, hz, n)
            if signature in used:
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert 47 <= n - gf2_rank(hx) - gf2_rank(hz) <= 64
            used.add(signature)
            chosen = ctx
            break
        # One unsuccessful architecture must not void successful slots.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the optimized architecture; never emit an orbit copy."""
    ctx = _joint_design_bank(G)[idx]
    if ctx is None:
        raise ValueError("no invariant-distinct target-rate design found")
    return ctx


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled, predominantly singleton A fillings reliably suppress weight-six Z witnesses; rejected by the parent: four of five measured codes have Z weight 6, while X witnesses weigh 16-20.
# REFUTED: Degree-four A reinforcement, four-cycle optimization and A-kernel exclusion through weight six suffice for d30s>=14 on Z23 (2,4)x(3,5); the current parent measured four distances of 12 and one of 13, with X witnesses at 17.
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
