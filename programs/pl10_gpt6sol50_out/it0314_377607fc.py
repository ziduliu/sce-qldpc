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
CONCEPT = ("In cyclic Z29 of order 29 with A=(2,4) and B=(2,4), suppressing short three-column component words and balancing verified full-lift six-cycles should reduce circuit failures in rate cell 0.250.")
GROUP = ("Keep Z29 of order 29 for a filling-only comparison. This measured (2,4)x(2,4) bed produces [[580,132]] codes; the previous local-syndrome screen scored 6.9876 against its 7.0367 parent.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: n=580, two 116-variable components, guaranteed k>=116 and rate>=0.200. The measured all-binomial k=132 gives rate 0.228 in cell 0.250. Parent failures were distributed among sector-1 Acol/Bcol blocks, with no persistent check-block hotspot.")
FILLING = ("Use twenty new pairs of inequivalent eight-gap spectra and distinct binomials throughout, keeping HX/HZ row weight 12. Screen explicit short component-kernel words, then verified HX/HZ six-cycles and k=132. Test for fewer stage-three fall-throughs and fewer failures than the parent's 18 Z failures in slot 1 and 10 X failures in slot 16; these screens are not the fitness.")
FAMILY_NAME = "z29_24x24_component_words_full_sixes_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Additive cyclic group Z29.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return int(g[0]) % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 29 = 580


def _gap_canon(gaps):
    """Identify spectra related by a Z29 automorphism or gap reversal."""
    return min(
        tuple(sorted(min((u * d) % 29, 29 - (u * d) % 29)
                     for d in gaps))
        for u in range(1, 15)
    )


def _make_gap_designs():
    from itertools import combinations
    choices = list(combinations(range(1, 15), 8))
    designs, seen = [], set()
    for t in range(len(choices)):
        gaps = choices[(211 * t + 47) % len(choices)]
        signature = _gap_canon(gaps)
        if signature not in seen:
            seen.add(signature)
            designs.append(gaps)
            if len(designs) == 120:
                break
    assert len(designs) == 120
    return tuple(designs)


_GAP_DESIGNS = _make_gap_designs()


def _binomial_grid(rng, gaps):
    """Use each distinct, full-order cell gap once on this side."""
    order = rng.sample(range(8), 8)
    grid = []
    for i in range(2):
        row = []
        for j in range(4):
            start = rng.randrange(29)
            gap = gaps[order[4 * i + j]]
            if rng.randrange(2):
                gap = -gap
            row.append([(start, 0), ((start + gap) % 29, 0)])
        grid.append(row)
    return grid


def _component_defects(G, grid, side):
    """Short kernel relations and four-cycles of an actual lift action.

    A variable h meets check g*h; B uses h*g^-1. These are the
    component actions used by lifted_matrices, not cyclic-shift proxies.
    """
    columns = []
    for row_column in range(len(grid[0])):
        for h in G.elements:
            syndrome = 0
            for i, row in enumerate(grid):
                for g in row[row_column]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)

    pairs = {}
    twos = fours = cycles = 0
    for u, first in enumerate(columns):
        for second in columns[u + 1:]:
            syndrome = first ^ second
            twos += syndrome == 0
            fours += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
            overlap = (first & second).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return twos, fours, cycles


def _rooted_component_cycles(G, grid, side):
    """Count genuine four- and six-cycles at check-orbit representatives."""
    q = G.q
    checks = [0] * (len(grid) * q)
    for j in range(len(grid[0])):
        for h in G.elements:
            variable = 1 << (j * q + G.idx(h))
            for i, row in enumerate(grid):
                for g in row[j]:
                    r = G.mul(g, h) if side == "A" else G.mul(h, G.inv(g))
                    checks[i * q + G.idx(r)] |= variable

    fours = sixes = 0
    for i in range(len(grid)):
        root_index = i * q + G.idx(G.e)
        root = checks[root_index]
        nearby = [(v, root & mask) for v, mask in enumerate(checks)
                  if v != root_index and root & mask]
        for a, (u, ru) in enumerate(nearby):
            size = ru.bit_count()
            fours += size * (size - 1) // 2
            for v, rv in nearby[:a]:
                # The middle variable is outside root; the first and
                # last variables must also differ from one another.
                middle = (checks[u] & checks[v] & ~root).bit_count()
                if middle:
                    distinct_ends = (ru.bit_count() * rv.bit_count()
                                     - (ru & rv).bit_count())
                    sixes += middle * distinct_ends
    return fours, sixes


def _component_word_profile(grid, side):
    """Find explicit kernel words from every three-column minor relation.

    For a 2x3 matrix over F2[x]/(x^29-1), its three 2x2 minors form
    a kernel word. Also test sums of two and three cyclic translates.
    B uses inverse exponents, matching its right-lift component action.
    """
    from itertools import combinations

    q = 29
    mask = (1 << q) - 1

    def rotate(value, shift):
        return ((value << shift) | (value >> (q - shift))) & mask

    def multiply(left, right):
        result = 0
        while left:
            bit = left & -left
            result ^= rotate(right, bit.bit_length() - 1)
            left ^= bit
        return result

    polys = [
        [sum(1 << ((g[0] if side == "A" else -g[0]) % q)
             for g in grid[i][j]) for j in range(4)]
        for i in range(2)
    ]
    minimum = 87
    counts = [0, 0, 0]
    for triple in combinations(range(4), 3):
        a, b, c = triple

        def minor(j, k):
            return (multiply(polys[0][j], polys[1][k])
                    ^ multiply(polys[0][k], polys[1][j]))

        word = (minor(b, c), minor(a, c), minor(a, b))
        shifts = [tuple(rotate(piece, t) for piece in word)
                  for t in range(q)]
        for degree in (1, 2, 3):
            for selected in combinations(range(1, q), degree - 1):
                pieces = [word[i] for i in range(3)]
                for t in selected:
                    pieces = [pieces[i] ^ shifts[t][i] for i in range(3)]
                weight = sum(piece.bit_count() for piece in pieces)
                if not weight:
                    continue
                minimum = min(minimum, weight)
                counts[0] += weight <= 12
                counts[1] += weight <= 13
                counts[2] += weight <= 15
    return minimum, tuple(counts)


def _rooted_full_sixes(rows, q):
    """Exact four- and simple six-cycle counts at each check-block root."""
    six_total = six_peak = four_total = 0
    for block in range(len(rows) // q):
        root_index = block * q
        root = rows[root_index]
        nearby = []
        for index, row in enumerate(rows):
            if index == root_index:
                continue
            shared = root & row
            size = shared.bit_count()
            if size:
                nearby.append((index, size, shared))
                four_total += size * (size - 1) // 2

        six = 0
        for position, (u, ru, ru_bits) in enumerate(nearby):
            for v, rv, rv_bits in nearby[:position]:
                uv = (rows[u] & rows[v]).bit_count()
                if uv:
                    triple = (ru_bits & rv_bits).bit_count()
                    six += ru * rv * uv - triple * (ru + rv + uv - 2)
        six_total += six
        six_peak = max(six_peak, six)
    return four_total, six_peak, six_total


def NEW_CTX(rng, G, idx):
    """Select an indexed gap design by component words and verified cycles."""
    profile = idx % 20
    tier = idx // 20
    # Earlier versions used spectra 0..79; these pairs use 80..119.
    gaps_a = _GAP_DESIGNS[80 + profile]
    gaps_b = _GAP_DESIGNS[100 + profile]
    counts = [[2] * 4 for _ in range(2)]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    shortlist = []
    for _ in range(42 + 2 * tier):
        a = _binomial_grid(rng, gaps_a)
        b = _binomial_grid(rng, gaps_b)
        da = _component_defects(G, a, "A")
        db = _component_defects(G, b, "B")
        ac = _rooted_component_cycles(G, a, "A")
        bc = _rooted_component_cycles(G, b, "B")
        preliminary = (da[0] + db[0],
                       max(ac[0], bc[0]), ac[0] + bc[0],
                       max(ac[1], bc[1]), ac[1] + bc[1],
                       max(da[1], db[1]), da[1] + db[1])
        shortlist.append((preliminary, rng.random(), a, b))
    shortlist.sort(key=lambda item: item[:2])

    finalists = []
    for preliminary, tie, a, b in shortlist[:12]:
        aw = _component_word_profile(a, "A")
        bw = _component_word_profile(b, "B")
        words = (min(aw[0], bw[0]) <= 12,
                 aw[1][0] + bw[1][0],
                 max(aw[1][1], bw[1][1]),
                 aw[1][1] + bw[1][1],
                 aw[1][2] + bw[1][2],
                 -min(aw[0], bw[0]))
        finalists.append((words, preliminary, tie, a, b))
    finalists.sort(key=lambda item: item[:3])

    best_key, best = None, None
    for words, preliminary, tie, a, b in finalists[:8]:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        xc = _rooted_full_sixes(hx, G.q)
        zc = _rooted_full_sixes(hz, G.q)
        key = (k != 132, words,
               max(xc[0], zc[0]), xc[0] + zc[0],
               max(xc[1], zc[1]), xc[1] + zc[1],
               max(xc[2], zc[2]), xc[2] + zc[2],
               preliminary, tie)
        if best_key is None or key < best_key:
            best_key, best = key, {"A": a, "B": b}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return one complete, independently screened cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Uniform rolled D29 (1,3)x(1,3) did not protect the parent: its five
# measured slots had abundant weight-four logicals and at least one
# basis reached the 100-failure cap within 750 shots.
# Higher row weight does not always suppress X failures: the Z43
# parent's w12 slot 3 had two in 200000 shots; w11 slots 9 and 39 had zero.
# Component four-/six-cycle screening on Z31 (2,3)x(2,5) did not
# balance the bases: all five measured slots had zero X failures,
# while Z reached 100 failures in 66750-170250 shots.
# Short-relation screening on C9 semidirect C4 did not guarantee a
# zero-failure Z basis: its five measured slots had 2-8 Z failures
# per 200000 shots, and slots 23 and 40 also had an X failure.
# REFUTED: Distinct gap spectra screened only for component relations and cycles eliminate Z failures; current slots 9 and 14 had 14 and 10 Z failures per 200000 shots.
# REFUTED: Adding clustered-error full-lift local-syndrome screening
# reliably improves the five-code score: fitness fell from 7.0367 to
# 6.9876; measured slot 1 had 18 Z failures and slot 16 had 10 X
# failures, each in 200000 shots.
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
