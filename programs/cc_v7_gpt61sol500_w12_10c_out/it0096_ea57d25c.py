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
CONCEPT = ("Metacyclic Z37:Z3(s=10), order 111, shapes (1,2)x(1,2): mixed-coset tetranomials with prescribed rotation-gap ratios and optimized component-kernel witnesses should produce five d30s readings of at least 22; this is a falsifiable search hypothesis, not a distance certificate.")
GROUP = ("Retain the metacyclic construction rule under L2. The new shapes would give n=40 at the parent's q=8, requiring a resized lift. Choose m=37,r=3,s=10, with 10^3=1 mod 37: this measured order-111 bed leads cell 0.200 at fitness 91.7 with five distance readings of 21; the related order-118 inversion-action bed reached 45.0.")
SHAPES = ("Target cell 0.200 with A=(1,2), B=(1,2): n=5*111=555, kb=111, guaranteed rate 0.200, and equal components of length 222 with 111 checks. The size window allows q<=119. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly. Exact assembled ranks enforce 0.175<=k/n<0.225 rather than assuming the rank bound is attained.")
FILLING = ("Twenty cards cross five rotation-gap geometries with four quotient-coset architectures. Every cell has four distinct terms, giving both HX and HZ row weight 12. Include tetranomials spanning all three quotient cosets; two-coset cells have different rotation-gap orbits to avoid elementary binomial factorizations. Normalize translation gauges, shortlist by exact component screening through four and four-cycles, optimize actual kernel witnesses, then select assembled finalists by exact rank and full-matrix short-syndrome statistics. Witness readings are upper bounds; component screening does not certify quantum distance.")
FAMILY_NAME = "r020_m37r3s10_12x12_coset_gap_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # Same metacyclic construction rule, resized for the new shapes.
    # Z_37 x| Z_3: y x y^-1 = x^10; 10^3 = 1 mod 37.
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


# Entries correspond to Acol0, Acol1, Bcol0, Bcol1.
# S spans all three quotient cosets with counts (2,1,1).
# T spans two quotient cosets with counts (2,2,0).
_COSET_CARDS = (
    ("S", "S", "S", "S"),
    ("S", "S", "S", "T"),
    ("S", "S", "T", "T"),
    ("T", "T", "T", "T"),
)
# 1,2,3,5,6,9 represent six different orbits under multiplication
# by +/-10 modulo 37. Ratios, not common scalar images, are prescribed.
_GAP_CARDS = (
    (1, 2, 3, 5),
    (1, 3, 5, 6),
    (1, 5, 6, 9),
    (1, 6, 9, 2),
    (1, 9, 2, 3),
)


def _support_for_slot(idx):
    return [[4, 4]], [[4, 4]]


def _label_component(rng, G, side, idx):
    """Construct the slot's relative-voltage design, not a gauge orbit."""
    card = idx % 20
    gaps = _GAP_CARDS[card // 4]
    kinds = _COSET_CARDS[card % 4]
    offset = 0 if side == "A" else 2
    row = []
    for j in range(2):
        loc = offset + j
        gap = gaps[loc]
        # Normalize one term to identity. Do not spend proposals on
        # multiplying an existing cell by a common group element.
        cell = [G.e, (gap, 0)]
        u = rng.randrange(37)
        if kinds[loc] == "S":
            cell.extend([(u, 1), (rng.randrange(37), 2)])
        else:
            second_gap = gaps[(loc + 1) % 4]
            # The two gaps occupy different +/-10 orbits. In particular
            # they cannot give the elementary left/right binomial factors.
            forbidden = {(sign * pow(10, t, 37) * gap) % 37
                         for sign in (-1, 1) for t in range(3)}
            assert second_gap not in forbidden
            cell.extend([(u, 1), ((u + second_gap) % 37, 1)])
        assert len({G._key(g) for g in cell}) == 4
        row.append(cell)
    return [row]


def _component_columns(hx, hz, q):
    """Extract both components solely from the verified LP construction."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            packed = 0
            for j in range(ca):
                packed |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(packed)
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]

    def transpose(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return columns

    return transpose(ar, ca * q), transpose(br, cb * q)


def _kernel_profile(columns):
    """Exact component-kernel screening through four.

    Five is a sentinel meaning no word through four, not a distance
    reading. With distinct nonzero columns, equal distinct pairs
    are disjoint; each weight-four word has three pair partitions.
    """
    singles = set(columns)
    if 0 in singles:
        return (1, 0)
    if len(singles) != len(columns):
        return (2, 0)
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return (3, 0)
            previous = pairs.get(syndrome, 0)
            collisions += previous
            pairs[syndrome] = previous + 1
    return (4, collisions // 3) if collisions else (5, 0)


def _four_cycles(columns):
    """Four-cycles in a component extracted from the verified lift."""
    total = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            overlap = (x & y).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _kernel_witness(columns, rng, trials):
    """Find actual component-kernel words using randomized information sets.

    Greedy XOR reduction and short-word combinations remain in the
    kernel. The minimum found is an upper bound, never a certificate.
    """
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome and word:
                words.append(word)
                best = min(best, word.bit_count())

        for sweep in range(2):
            donors = sorted(words, key=int.bit_count)[:24]
            for i, word in enumerate(words):
                weight = word.bit_count()
                for other in donors:
                    proposal = word ^ other
                    proposed_weight = proposal.bit_count()
                    if 0 < proposed_weight < weight:
                        word, weight = proposal, proposed_weight
                words[i] = word
                best = min(best, weight)

        short = sorted(words, key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                weight = (word ^ other).bit_count()
                if weight:
                    best = min(best, weight)
        small = short[:8]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    weight = (pair ^ small[h]).bit_count()
                    if weight:
                        best = min(best, weight)
        if best <= 4:
            break
    return best


def _short_profile(rows, n):
    """Use columns of the verified matrix as opposite-check syndromes.
    Severity 0 excludes every nonzero commuting operator of weight <=3.
    Pair-syndrome collisions then count weight-4 kernel supports three
    times each; they need not all be logical operators."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return (2, 0)
    for i, x in enumerate(columns):
        for j in range(i):
            if x ^ columns[j] in singles:
                return (1, 0)
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            previous = pairs.get(syndrome, 0)
            collisions += previous
            pairs[syndrome] = previous + 1
    return (0, collisions)


def NEW_CTX(rng, G, idx):
    """Optimize the card's relative voltages, then its assembled LP code."""
    assert G.q == 111
    sa, sb = _support_for_slot(idx)
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    zero = [[[], []]]
    elites = {}

    def component_merit(item):
        return (int(item["flag"] == 5), item["d"], item["flag"],
                -item["hits"], -item["cycles"], item["rank"])

    for side in ("A", "B"):
        proposals = []
        seen = set()
        for attempt in range(40 + 4 * (idx // 20)):
            grid = _label_component(rng, G, side, idx)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
            if key in seen:
                continue
            seen.add(key)

            # Zero companions are used only to extract a component.
            # Every emitted LP code has four nonzero tetranomial cells.
            A, B = (grid, zero) if side == "A" else (zero, grid)
            hx, hz, _ = lifted_matrices(G, A, B)
            ac, bc = _component_columns(hx, hz, G.q)
            columns = ac if side == "A" else bc
            rank = gf2_rank(columns)
            if rank < G.q - 3:
                continue
            flag, hits = _kernel_profile(columns)
            cycles = _four_cycles(columns)
            priority = (-flag, cycles, -rank, attempt)
            proposals.append((priority, grid, columns, rank,
                              flag, hits, cycles))

        if not proposals:
            raise ValueError("no sufficiently ranked component proposal")
        proposals.sort(key=lambda item: item[0])
        finalists = []
        for _, grid, columns, rank, flag, hits, cycles in proposals[:6]:
            reading = _kernel_witness(columns, rng, 10)
            if flag < 5:
                reading = min(reading, flag)
            if reading > len(columns):
                continue
            finalists.append({
                "grid": grid, "columns": columns, "rank": rank,
                "flag": flag, "hits": hits, "cycles": cycles,
                "d": reading,
            })

        if not finalists:
            raise ValueError("component probing found no actual witness")
        finalists.sort(key=component_merit, reverse=True)
        finalists = finalists[:3]
        for item in finalists:
            item["d"] = min(
                item["d"], _kernel_witness(item["columns"], rng, 32))
        finalists.sort(key=component_merit, reverse=True)
        elites[side] = finalists

    assembled = []
    for ai, a in enumerate(elites["A"]):
        for bi, b in enumerate(elites["B"]):
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (175 * n <= 1000 * k < 225 * n):
                continue
            d = min(a["d"], b["d"])
            # Classical witnesses guide selection but are not asserted
            # to be non-stabilizer logicals of the assembled quantum code.
            proxy = k * d * d / float(n)
            key = (proxy, d, a["d"] + b["d"],
                   -(a["cycles"] + b["cycles"]))
            assembled.append((key, ai, bi, A, B, hx, hz, n, k))

    if not assembled:
        raise ValueError("no assembled filling in the target rate cell")
    assembled.sort(key=lambda item: item[0], reverse=True)

    winner = None
    for key, ai, bi, A, B, hx, hz, n, k in assembled[:3]:
        px = _short_profile(hx, n)
        pz = _short_profile(hz, n)
        severity = max(px[0], pz[0])
        four = px[1] + pz[1]
        # Severity zero excludes all commuting supports through three.
        # With severity zero, four=0 additionally excludes weight four.
        # Penalizing small stabilizers too is deliberately conservative.
        score = (-severity, int(severity == 0 and four == 0),
                 key[0], key[1], key[2], -four, key[3], k)
        if winner is None or score > winner[0]:
            winner = (score, A, B, hx, hz, n, k)

    _, A, B, hx, hz, n, k = winner
    assert css_ok(hx, hz)
    assert n == 555 and 175 * n <= 1000 * k < 225 * n
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert all(len(cell) == len({G._key(g) for g in cell}) == 4
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection": winner[0]}


def CELL(rng, G, side, i, j, ctx):
    """All labels and counts were selected jointly in NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct relative monomial labels and short-syndrome screening on D8 at (2,7)x(2,8) did not escape weight-four sector-1 logicals; all five measured d30s readings were 4.
# REFUTED: Degree-three component columns and order-four binomial ratios on D8 at (3,8)x(2,6) did not raise the measured readings to eight: all five d30s readings were four, with minima exclusively in sector 1.
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
