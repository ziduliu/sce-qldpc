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
CONCEPT = ("Frozen S3 x C4, order 24, shapes (1,4)x(1,5): verified rank-surplus optimization with exact stabilizer-aware logical-four classification should beat archived generator deletions; progress requires five fresh codes, with k>297 at distance four or exclusion through weight four.")
GROUP = ("Keep the exact L1 group and order. This bed measured five distance-four codes with k=297 and fitness 9.429. Their archived matrices cannot earn credit again. Central involution pairs and order-three polynomial supports supply different rank mechanisms, whose short logicals must be tested rather than assumed harmless.")
SHAPES = ("Keep (1,4)x(1,5) for the open-ended 0.550 cell: n=504, kb=288, guaranteed rate 4/7; component variable/check counts are 96/24 and 120/24. The shape allows q<=28, but the frozen group fixes q=24. Parent minima were entirely sector-1 Acol/Bcol operators spread broadly, motivating complete two-type, two-sector screening.")
FILLING = ("Enumerate legal positive count profiles and the inverse-distinct triangle-free binomial library. Search unrestricted, central-parity-mixed and deliberately rank-deficient cell supports. Exact scaffold-derived matrices classify logicals through four modulo their stabilizer spans. Optimize k times squared exclusion floor, then rank. Uncoloured check-intersection closed-walk signatures conservatively exclude known principal parent classes and repeated outputs. Sentinel five is an exclusion result, not a distance estimate.")
FAMILY_NAME = "r055_S3xC4_14x15_rankescape_quotient4_walknovel_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # (a,b,c) = r^a t^b z^c:
    # r^3=t^2=z^4=1, t*r*t=r^-1, z central.
    if not valid_q(q):
        raise ValueError("S3 x C4 requires order 24")
    els = [(a, b, c) for c in range(4)
           for b in range(2) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 2, g[2] % 4)

    def mul(g, h):
        return ((g[0] + (1 if g[1] == 0 else -1) * h[0]) % 3,
                (g[1] + h[1]) % 2, (g[2] + h[2]) % 4)

    def inv(g):
        return ((-g[0] if g[1] == 0 else g[0]) % 3,
                g[1], (-g[2]) % 4)

    def idx(g):
        a, b, c = red(g)
        return (2*c + b)*3 + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


def _no_kernel_le3(rows, n):
    """Exhaustive conservative test: no nonzero kernel of weight <=3.
    Rejecting even a small stabilizer is intentional here."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, x in enumerate(columns):
        for j in range(i):
            if (x ^ columns[j]) in singles:
                return False
    return True


from functools import lru_cache
from itertools import combinations, combinations_with_replacement


def _re_columns(rows, n):
    cols = [0] * n
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    return cols


def _re_floor(checks, stabilizers, n):
    """Exact minimum through four, or exclusion sentinel five."""
    basis = {}
    for word in stabilizers:
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous

    def outside(word):
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                return True
            word ^= previous
        return False

    cols = _re_columns(checks, n)
    bits = [1 << i for i in range(n)]
    singles = {0: 0}
    for i, syndrome in enumerate(cols):
        if syndrome == 0 and outside(bits[i]):
            return 1
        old = singles.get(syndrome)
        if old is None:
            singles[syndrome] = bits[i]
        elif outside(bits[i] ^ old):
            return 2

    # All single/pair partitions: exact exclusion through three.
    for i, syndrome in enumerate(cols):
        for j in range(i):
            old = singles.get(syndrome ^ cols[j])
            if old is not None and outside(bits[i] ^ bits[j] ^ old):
                return 3

    # One representative per syndrome is sufficient: previous
    # differences have already been classified as stabilizers.
    seen = dict(singles)
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif outside(word ^ old):
                return 4
    return 5


def _re_signature(hx, hz, n):
    """Uncoloured Tanner invariant, strengthened by Gram closed walks.

    Unequal signatures certify nonisomorphism. Equal signatures are
    discarded conservatively; equality does not prove isomorphism.
    """
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap

    profiles = []
    for i, links in enumerate(adjacency):
        # Weighted length-two walks in the check-intersection graph.
        two = {}
        for j, first in links.items():
            for t, second in adjacency[j].items():
                two[t] = two.get(t, 0) + first * second
        triangle = sum(value * two.get(j, 0)
                       for j, value in links.items())
        fourth = sum(value * value for value in two.values())
        profile = tuple(sorted(
            (weights[j], links.get(j, 0), two.get(j, 0))
            for j in range(len(rows)) if j != i))
        profiles.append((weights[i], two.get(i, 0),
                         triangle, fourth, profile))

    qubits = [[] for _ in range(n)]
    for weight, source in zip(weights, rows):
        while source:
            bit = source & -source
            qubits[bit.bit_length() - 1].append(weight)
            source ^= bit
    return (tuple(sorted(profiles)),
            tuple(sorted(tuple(sorted(p)) for p in qubits)))


@lru_cache(maxsize=1)
def _re_bank(G, slots):
    """Deterministic joint architecture search; cache is transparent."""
    assert G.q == 24
    local = random.Random(550240519)
    n = 504
    elements = tuple(G.elements)
    zero_a = [[[] for _ in range(4)]]
    zero_b = [[[] for _ in range(5)]]

    # Linear superposition of complete VERIFIED single-term LP lifts.
    # No independent implementation of either regular action is used.
    atlas = []
    for position in range(9):
        terms = []
        for g in elements:
            A = [[list(cell) for cell in zero_a[0]]]
            B = [[list(cell) for cell in zero_b[0]]]
            if position < 4:
                A[0][position] = [g]
            else:
                B[0][position - 4] = [g]
            hx, hz, width = lifted_matrices(G, A, B)
            assert width == n
            terms.append((tuple(hx), tuple(hz)))
        atlas.append(tuple(terms))

    def compile_pair(A, B):
        hx, hz = [0] * 120, [0] * 96
        for position, cell in enumerate(A[0] + B[0]):
            for g in cell:
                xrows, zrows = atlas[position][G.idx(g)]
                for r, word in enumerate(xrows):
                    hx[r] ^= word
                for r, word in enumerate(zrows):
                    hz[r] ^= word
        return hx, hz

    archived = set()
    generators = [(a, b, 1) for b in range(2) for a in range(3)]
    for missing_a in ((0, 3), (0, 1), (3, 4), (3, 1), (1, 2)):
        for missing_b in (0, 3, 1):
            A = [[[G.e, g] for t, g in enumerate(generators)
                  if t not in missing_a]]
            B = [[[G.e, g] for t, g in enumerate(generators)
                  if t != missing_b]]
            hx, hz = compile_pair(A, B)
            archived.add(_re_signature(hx, hz, n))

    keep, seen_labels = {}, set()
    capacity = max(32, 2 * slots)

    def admit(A, B):
        label = tuple(tuple(sorted(G.idx(g) for g in cell))
                      for cell in A[0] + B[0])
        if label in seen_labels:
            return
        seen_labels.add(label)
        hx, hz = compile_pair(A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        assert k >= 288
        fx = _re_floor(hz, hx, n)
        fz = _re_floor(hx, hz, n)
        floor = min(fx, fz)
        # Five is an exact exclusion sentinel, not a sampled distance.
        merit = (k * floor * floor, floor, k, fx + fz)
        if len(keep) >= capacity:
            worst = min(item[0] for item in keep.values())
            if merit < worst:
                return
        signature = _re_signature(hx, hz, n)
        if signature in archived:
            return
        previous = keep.get(signature)
        if previous is None or merit > previous[0]:
            keep[signature] = (merit, A, B, k, (fx, fz))
        if len(keep) > capacity:
            weakest = min(keep, key=lambda key: keep[key][0])
            del keep[weakest]

    representatives = [
        g for g in elements
        if g != G.e and G.mul(g, g) != G.e
        and G.idx(g) < G.idx(G.inv(g))]

    def binomial_library(count):
        out = []
        for steps in combinations(representatives, count):
            signed = set(steps)
            signed.update(G.inv(g) for g in steps)
            if any(G.mul(g, h) in signed
                   for g in signed for h in signed):
                continue
            out.append(steps)
        return out

    # Exhaust the admissible architecture library rather than randomly
    # regenerate one of the prescribed deletion cards.
    for a in binomial_library(4):
        for b in binomial_library(5):
            admit([[[G.e, g] for g in a]],
                  [[[G.e, g] for g in b]])

    profiles = []
    for a in combinations_with_replacement(range(1, 6), 4):
        for b in combinations_with_replacement(range(1, 6), 5):
            if sum(a) + max(b) > W_MAX:
                continue
            if sum(b) + max(a) > W_MAX:
                continue
            profiles.append((a, b))
    profiles.sort(key=lambda p: (
        -(min(p[0]) + min(p[1])),
        p[0].count(1) + p[1].count(1),
        -(sum(p[0]) + sum(p[1])), p))

    odd = [g for g in elements if g[2] & 1]
    even = [g for g in elements if not (g[2] & 1)]
    r, z2 = (1, 0, 0), (0, 0, 2)

    def draw(counts, mode, phase):
        row = []
        for j, count in enumerate(counts):
            if count == 1:
                cell = [local.choice(elements)]
            elif mode == 0:
                cell = local.sample(elements, count)
            elif mode == 1:
                amount = (phase + j) % (count + 1)
                cell = (local.sample(odd, amount)
                        + local.sample(even, count - amount))
            elif mode == 2 and count == 3:
                # A translated C3 norm is deliberately singular.
                base = local.choice(elements)
                cell = [base, G.mul(base, r),
                        G.mul(G.mul(base, r), r)]
            elif mode == 2 and count >= 4:
                # Complete central pairs test degeneracy explicitly;
                # they are never presumed to improve distance.
                base = local.choice(elements)
                cell = [base, G.mul(base, z2)]
                available = [g for g in elements if g not in cell]
                second = local.choice(available)
                cell.extend((second, G.mul(second, z2)))
                if count == 5:
                    cell.append(local.choice(
                        [g for g in elements if g not in cell]))
            else:
                step = local.choice(representatives)
                cell = [G.e, step]
                if count > 2:
                    cell.extend(local.sample(
                        [g for g in elements if g not in cell],
                        count - 2))
            assert len(cell) == len(set(cell)) == count
            row.append(cell)
        return [row]

    quota = max(64, 2 * slots)
    for ordinal, (a, b) in enumerate(profiles):
        for attempt in range(quota):
            mode = attempt % 4
            admit(draw(a, mode, ordinal),
                  draw(b, (mode + attempt // 4) % 4, ordinal + 1))

    # Local replacements optimize genuinely different supports, not
    # transported copies. The complete matrices are rescreened each time.
    for step in range(max(256, 12 * slots)):
        leaders = sorted(keep.values(), key=lambda item: item[0],
                         reverse=True)[:12]
        if not leaders:
            break
        _, parent_a, parent_b, _, _ = local.choice(leaders)
        A = [[list(cell) for cell in parent_a[0]]]
        B = [[list(cell) for cell in parent_b[0]]]
        grid = A if step & 1 else B
        counts = tuple(len(cell) for cell in grid[0])
        proposal = draw(counts, step % 4, step)
        j = local.randrange(len(counts))
        grid[0][j] = proposal[0][j]
        admit(A, B)

    selected = sorted(keep.values(), key=lambda item: item[0],
                      reverse=True)[:slots]
    bank = []
    for merit, A, B, k, floors in selected:
        hx, hz, width = lifted_matrices(G, A, B)
        compiled_x, compiled_z = compile_pair(A, B)
        assert hx == compiled_x and hz == compiled_z
        assert width == n and css_ok(hx, hz)
        assert n - gf2_rank(hx) - gf2_rank(hz) == k
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert _no_dead_lines(
            [[len(cell) for cell in A[0]]],
            [[len(cell) for cell in B[0]]])
        bank.append((
            tuple(tuple(tuple(cell) for cell in row) for row in A),
            tuple(tuple(tuple(cell) for cell in row) for row in B),
            k, floors, merit))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a jointly optimized invariant-distinct filling."""
    bank = _re_bank(G, N_SLOTS)
    if idx >= len(bank):
        raise ValueError("no further fresh invariant-distinct screened design")
    A, B, k, floors, merit = bank[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "logical_floors": floors, "selection": merit}


def CELL(rng, G, side, i, j, ctx):
    """Return the exact jointly screened full cell, without new draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen prescribed odd-central deletion cards provide fifteen inequivalent codes: parent slots 0-14 collapsed to five classes, all measured with k=297 and d30s=4.
# REFUTED: Unscreened mixed-count experiments improve the distance-four leaders: parent slots 15-59 had d2s<=3 and none advanced over the principal codes.
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
