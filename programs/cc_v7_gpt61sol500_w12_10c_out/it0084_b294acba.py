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
CONCEPT = ("On cyclic Z23, order 23, shapes (1,3)x(2,7), zero/trinomial redistribution of B's degree-three columns plus exact weight-six component-word counting should improve the five-code fitness over 15.6520. Commuting A trinomials still supply weight-six relations, so this is a measurement hypothesis, not a claim of distance above six.")
GROUP = ("Frozen under L1. This Z23 bed already gives five measured distance-six readings; related Z23 beds reached fitness 25.2 and 22.9. Test the filling bottleneck without changing the group or its order.")
SHAPES = ("Frozen A=(1,3), B=(2,7): n=529, kb=230, guaranteed rate 10/23=0.434783 in cell 0.450. Components have 69 variables/23 checks and 161 variables/46 checks. Parent minima were entirely in sector-1 Acol/Bcol blocks, broadly across B columns and especially Acol1. Verified ranks retain 0.425<=k/n<0.475.")
FILLING = ("Twenty initial cards cross three zero-containing B support geometries with seven A difference-overlap/AP policies. Four B columns have degree three and three have degree two; both B row sums are nine. Require exact component exclusion through weight five and stabilizer-aware full-CSS exclusion through weight four. Finalists are ranked by verified k, then exactly counted weight-six component words and Tanner four-cycles. Later slots change the prescribed A/B difference-overlap coupling rather than apply orbit transformations. All CSS row weights are <=12.")
FAMILY_NAME = "r045_Z23_13x27_zero_triples_exact6_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    if q != 23:
        raise ValueError("this cyclic bed requires order 23")
    els = [(a,) for a in range(q)]

    def red(g):
        return (g[0] % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _q85_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _q85_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return True
        v ^= basis[p]
    return False


def _q85_short_screen(hx, hz, n):
    """Exact existence test for nontrivial CSS logicals of weight <=4.

    Every such support splits into two supports of size <=2.
    Equal-syndrome collisions are tested modulo the actual stabilizer
    row space. Keeping one representative per syndrome is sufficient:
    if all collisions with it are stabilizers, their pairwise differences
    are stabilizers too.
    """
    cycles = 0
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = _q85_basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Count actual Tanner 4-cycles, not protograph coincidences.
        overlaps = {}
        for col in cols:
            ids = []
            v = col
            while v:
                bit = v & -v
                ids.append(bit.bit_length() - 1)
                v ^= bit
            for a in range(len(ids)):
                for b in range(a):
                    key = (ids[b], ids[a])
                    old = overlaps.get(key, 0)
                    cycles += old
                    overlaps[key] = old + 1

        representatives = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            mask = bits[i]
            if syndrome in representatives:
                v = mask ^ representatives[syndrome]
                if _q85_outside(v, basis):
                    return v.bit_count(), cycles
            else:
                representatives[syndrome] = mask

        for i in range(n):
            si, bi = cols[i], bits[i]
            for j in range(i):
                syndrome = si ^ cols[j]
                mask = bi | bits[j]
                if syndrome in representatives:
                    v = mask ^ representatives[syndrome]
                    if _q85_outside(v, basis):
                        return v.bit_count(), cycles
                else:
                    representatives[syndrome] = mask

    # Sentinel: no logical of weight <=4, not a distance estimate.
    return 5, cycles


def _z23_columns(rows, length):
    cols = [0] * length
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _z23_components(hx, hz, q):
    """Extract both components from the scaffold's exact lift."""
    ca, cb, rb = A_SHAPE[1], B_SHAPE[1], B_SHAPE[0]
    mask = (1 << q) - 1
    ar = []
    for row in hx[:q]:
        compact = 0
        for j in range(ca):
            compact |= ((row >> (j * cb * q)) & mask) << (j * q)
        ar.append(compact)
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return (_z23_columns(ar, ca * q),
            _z23_columns(br, cb * q))


def _z23_short(cols, through_five=False):
    """Exact existence screening, not a heuristic distance reading.

    Singles/pairs cover weight <=4. Triples against their representatives
    cover weight <=5 without storing the much larger triple dictionary.
    The sentinel is 5 or 6 when the requested weights were excluded.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        old = seen.get(syndrome)
        if old is not None:
            return (old ^ word).bit_count()
        seen[syndrome] = word
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is not None:
                return (old ^ word).bit_count()
            seen[syndrome] = word
    if not through_five:
        return 5
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            for t in range(j):
                old = seen.get(syndrome ^ cols[t])
                if old is not None:
                    return (old ^ word ^ bits[t]).bit_count()
    return 6


def _z23_reading(cols, rng, trials):
    """Upper-bound readings from actual nonzero component-kernel words."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for i in order:
            syndrome, word = cols[i], 1 << i
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for other in words[:i]:
                best = min(best, (word ^ other).bit_count())
        small = words[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for t in range(j):
                    best = min(best, (pair ^ small[t]).bit_count())
        if best <= 4:
            break
    return best


def _z23_cycles(cols):
    total = 0
    for i, col in enumerate(cols):
        for other in cols[:i]:
            overlap = (col & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _z23_candidate(rng, G, sb, ap_count):
    """Fresh labels for the card, never an automorphism-copy generator."""
    q, cb = G.q, B_SHAPE[1]
    A = [[]]
    for j in range(A_SHAPE[1]):
        if j < ap_count:
            step = rng.randrange(1, q)
            offsets = (0, step, (2 * step) % q)
        else:
            while True:
                u, v = rng.sample(range(1, q), 2)
                # None of the three terms is an AP midpoint.
                if (v != (2 * u) % q and u != (2 * v) % q
                        and (u + v) % q != 0):
                    offsets = (0, u, v)
                    break
        base = G.pick(rng)
        A[0].append([G.mul(base, (a,)) for a in offsets])

    # Distinct relative labels prevent repeated all-monomial columns.
    relatives = rng.sample(range(q), cb)
    B = [[None] * cb for _ in range(2)]
    for j, relative in enumerate(relatives):
        top = G.pick(rng)
        bottom = G.mul(top, (relative,))
        B[0][j], B[1][j] = [top], [bottom]
    for i in range(2):
        for j in range(cb):
            if sb[i][j] == 2:
                delta = (rng.randrange(1, q),)
                B[i][j].append(G.mul(B[i][j][0], delta))
    return A, B


def NEW_CTX(rng, G, idx):
    """Joint zero-block design; all screening uses verified lift actions."""
    q = G.q
    assert q == 23
    style = idx % 7
    geometry = (idx // 7) % 3
    batch = idx // 21

    # Each entry is the top term count of a degree-three B column.
    # The complementary bottom counts also sum to six. Three
    # singleton/singleton columns complete both row sums to nine.
    top_cards = ((0, 0, 3, 3), (0, 1, 2, 3), (0, 2, 2, 2))
    top_counts = top_cards[geometry]
    sa = [[3, 3, 3]]
    sb = [list(top_counts) + [1, 1, 1],
          [3 - t for t in top_counts] + [1, 1, 1]]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def unsigned(t):
        t %= q
        return min(t, q - t)

    ordinary, progressions = [], []
    for u in range(1, q):
        for v in range(u + 1, q):
            offsets = (0, u, v)
            differences = frozenset(
                (unsigned(u), unsigned(v), unsigned(v - u)))
            ap = (v == 2*u % q or u == 2*v % q
                  or (u + v) % q == 0)
            record = (offsets, differences)
            (progressions if ap else ordinary).append(record)

    def draw_A():
        # Styles 0..3 prescribe exactly 0..3 repeated unsigned
        # differences across three individually non-AP trinomials.
        # Styles 4..6 prescribe one, two, or three AP cells.
        ap_count = 0 if style < 4 else style - 3
        for _ in range(4096):
            records = [
                rng.choice(progressions if j < ap_count else ordinary)
                for j in range(3)]
            union = frozenset().union(*(r[1] for r in records))
            if style < 4 and len(union) != 9 - style:
                continue
            A = [[[(a,) for a in record[0]] for record in records]]
            return A, union
        raise ValueError("could not realize prescribed A difference policy")

    def six_word_count(cols):
        """Exact number of weight-six component kernel words.

        Called only after weights 1..5 have been exhaustively excluded.
        Equal-syndrome distinct triples must then be disjoint. Every
        sextet has ten unordered 3+3 partitions, so collision count/10
        counts words, rather than duplicate partitions.
        """
        counts = {}
        collisions = 0
        for i, ci in enumerate(cols):
            for j in range(i):
                pair = ci ^ cols[j]
                for t in range(j):
                    syndrome = pair ^ cols[t]
                    old = counts.get(syndrome, 0)
                    collisions += old
                    counts[syndrome] = old + 1
        assert collisions % 10 == 0
        return collisions // 10

    # Select A policies using the scaffold's action, independently of
    # B's eventual labels. A dummy partner is used only for extraction.
    dummy_B = [[[G.e] for _ in range(7)] for _ in range(2)]
    a_pool, seen_a = [], set()
    wanted_overlap = (2, 3, 1)[batch % 3]
    for attempt in range(512):
        A, differences = draw_A()
        signature = tuple(tuple(g[0] for g in cell) for cell in A[0])
        if signature in seen_a:
            continue
        seen_a.add(signature)
        hx, hz, n = lifted_matrices(G, A, dummy_B)
        ac, _ = _z23_components(hx, hz, q)
        if _z23_short(ac) != 5:
            continue
        if _z23_short(ac, through_five=True) != 6:
            continue

        # This is a relative-geometry coupling, not a common
        # automorphism applied to an already existing filling.
        mono_pool = [
            record for record in ordinary
            if len(record[1] & differences) == wanted_overlap]
        if not mono_pool:
            continue
        a_pool.append({
            "A": A, "ac": ac, "mono": mono_pool,
            "cycles": _z23_cycles(ac), "six": None,
        })
        if len(a_pool) >= 12:
            break

    if not a_pool:
        raise ValueError("A policy found no component-clean filling")

    def draw_B(a_item):
        B = [[[] for _ in range(7)] for _ in range(2)]
        for j, top_count in enumerate(top_counts):
            if top_count == 0:
                offsets = rng.choice(ordinary)[0]
                B[1][j] = [(a,) for a in offsets]
            elif top_count == 3:
                offsets = rng.choice(ordinary)[0]
                B[0][j] = [(a,) for a in offsets]
            elif top_count == 2:
                delta = rng.randrange(1, q)
                B[0][j] = [G.e, (delta,)]
                B[1][j] = [G.pick(rng)]
            else:
                B[0][j] = [G.e]
                B[1][j] = rng.sample(G.elements, 2)

        # Distinct ordered differences of these three matching
        # relatives exclude four-cycles in their degree-two subgraph.
        # The AP-free template also enforces the slot's A/B overlap.
        relatives = rng.choice(a_item["mono"])[0]
        for j, relative in enumerate(relatives, 4):
            B[0][j] = [G.e]
            B[1][j] = [(relative,)]
        return B

    finalists, seen = [], set()
    accepted = 0
    for attempt in range(192):
        a_item = rng.choice(a_pool)
        A, B = a_item["A"], draw_B(a_item)
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)

        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _z23_components(hx, hz, q)
        if _z23_short(bc) != 5:
            continue
        if _z23_short(bc, through_five=True) != 6:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (425*n <= 1000*k < 475*n):
            continue

        cycles = a_item["cycles"] + _z23_cycles(bc)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
            "k": k, "bc": bc, "a_item": a_item, "cycles": cycles,
        }
        finalists.append(item)
        finalists.sort(
            key=lambda x: (x["k"], -x["cycles"]), reverse=True)
        del finalists[4:]
        accepted += 1
        if accepted >= 12:
            break

    if not finalists:
        raise ValueError("B policy found no component-clean filling")

    winner = None
    for item in finalists:
        low, cycles = _q85_short_screen(
            item["hx"], item["hz"], item["n"])
        if low != 5:
            continue
        a_item = item["a_item"]
        if a_item["six"] is None:
            a_item["six"] = six_word_count(a_item["ac"])
        na = a_item["six"]
        nb = six_word_count(item["bc"])
        # Exact component counts are not counts of quantum logicals.
        # Prefer rank first; do not reward a sampled reading above
        # the unavoidable weight-six A component relation.
        key = (item["k"], -(na + nb), -max(na, nb), -cycles)
        if winner is None or key > winner[0]:
            winner = (key, item, na, nb)

    if winner is None:
        raise ValueError("no finalist passed full CSS logical-four exclusion")
    key, item, na, nb = winner
    assert css_ok(item["hx"], item["hz"])
    return {
        "idx": idx, "cells": {"A": item["A"], "B": item["B"]},
        "k": item["k"], "screen": key,
        "component_six_words": (na, nb),
        "policy": (geometry, style, wanted_overlap),
    }


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Q8 x Z5 order-20 quaternion-axis binomials plus exact-four candidate selection did not break the weight-four plateau: all five measured codes had d30s=4, with minima entirely in sector 1.
# REFUTED: Repeating the Z23 support/AP recipes with fresh sampling guarantees distinct codes: slots 20,22,23,26,27 and many later slots were marked isomorphic duplicates.
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
