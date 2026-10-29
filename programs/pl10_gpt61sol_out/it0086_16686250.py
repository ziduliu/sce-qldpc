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
CONCEPT = ("Dic5, order 20, shapes (2,5)x(2,5): asymmetric rotation/mixed-coset doubletons and verified two-sided neighborhood screening should improve worse-basis circuit fitness beyond the cyclic parent's 5.608.")
GROUP = ("Jump from C20 to nonabelian Dic5: x^10=1, y^2=x^5, y*x*y^-1=x^-1. Order 20 preserves n=580. The Dic11 (2,3)x(2,3) bed reached 7.338 with zero measured failures, motivating this algebraic transfer but not proving a group advantage at higher rate. Dic5 has shorter rotation cycles, which remains a risk.")
SHAPES = ("Keep equal (2,5) components with 100 variables and 40 checks each, n=580 and kb=180, guaranteed rate 9/29=0.310345. Verified full ranks keep every submitted code in cell 0.300. Parent failures affected all 180 logicals and many Acol/Bcol blocks with mostly 1.1-1.2x enrichment; no unique Acol/Brow repair is justified.")
FILLING = ("Twenty prescribed support-pair policies retain row sums 7..8 and column sums 2..4, so both CSS row-weight sums remain <=12. A doubletons primarily use order-ten rotation ratios and B doubletons order-four mixed-coset ratios; additional slots change the ratio policy, not a saved labeling. Optimize verified left/right component actions separately, require connected full-row-rank components, and select against kernels through weight six and small-boundary triples. Final verified matrices must pass actual-rate, CSS and weight-four logical checks. This targets collective circuit failures and decoder difficulty, not distance as fitness.")
FAMILY_NAME = "r030_dic5_25x25_asym_rightlift_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic5 requires order 20")
    # (a,b)=x^a*y^b; x^10=1, y^2=x^5, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        return (int(g[0]) % 10, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 5*b*d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 10, 0) if b == 0 else ((a + 5) % 10, 1)

    def idx(g):
        a, b = red(g)
        return 10*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# Distinct structural policies, not automorphism/permutation orbit copies.
# Every profile has row sums <=8 and column sums <=4.
_SUPPORT_PROFILES = (
    ((1, 1, 2, 2, 1), (2, 2, 1, 1, 2)),  # columns all degree 3
    ((2, 1, 2, 2, 1), (2, 2, 1, 1, 2)),  # one degree-4 column
    ((1, 1, 2, 2, 1), (1, 2, 1, 1, 2)),  # one degree-2 column
    ((0, 1, 3, 2, 1), (3, 2, 0, 1, 2)),  # balanced zero/trinomial cells
    ((0, 1, 3, 2, 1), (2, 2, 0, 1, 2)),  # sparse degree-2 variant
)


def _component_rows(G, side, cells):
    """Probe the verified A-left or B-right action, never a private lift."""
    if side == "A":
        rows, _, _ = lifted_matrices(G, cells, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], cells)
    mask = (1 << (len(cells[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _four_cycles(rows):
    count = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            count += overlap * (overlap - 1) // 2
    return count


def _column_syndromes(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _connected_component(rows, width):
    """Connectivity of the component Tanner graph from verified rows."""
    parent = list(range(len(rows)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for col in _column_syndromes(rows, width):
        if not col:
            return False
        first_bit = col & -col
        first = first_bit.bit_length() - 1
        col ^= first_bit
        while col:
            bit = col & -col
            a, b = root(first), root(bit.bit_length() - 1)
            if a != b:
                parent[b] = a
            col ^= bit
    return len({root(a) for a in range(len(rows))}) == 1


def _small_component_kernels(rows, width):
    """Subset collisions detect existence of component kernels <=6.

    A representative per syndrome suffices for the existence test.
    Detected minimum and collision multiplicities are ranking heuristics,
    not a complete spectrum. Triple boundaries are a trapping heuristic.
    """
    cols = _column_syndromes(rows, width)
    masks = [1 << j for j in range(width)]
    seen = {0: 0}
    bad = [0] * 7
    boundary = 0

    def register(syndrome, mask):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = mask
        else:
            w = (previous ^ mask).bit_count()
            if 0 < w <= 6:
                bad[w] += 1

    for i in range(width):
        register(cols[i], masks[i])
    for i in range(width):
        for j in range(i):
            register(cols[i] ^ cols[j], masks[i] | masks[j])
    for i in range(width):
        for j in range(i):
            syndrome = cols[i] ^ cols[j]
            mask = masks[i] | masks[j]
            for h in range(j):
                triple = syndrome ^ cols[h]
                weight = triple.bit_count()
                if weight < 4:
                    boundary += (4 - weight) ** 2
                register(triple, mask | masks[h])
    minimum = next((w for w in range(1, 7) if bad[w]), 7)
    penalty = sum(bad[w] * (7 - w) ** 2 for w in range(1, 7))
    return minimum, penalty, boundary


def _dic_cell(rng, G, terms, side, i, j, mode):
    """Distinct labels with prescribed non-involutory relative ratios."""
    if terms == 0:
        return []
    if terms == 1:
        return [G.pick(rng)]

    rotations10 = [(a, 0) for a in (1, 3, 7, 9)]
    rotations5 = [(a, 0) for a in (2, 4, 6, 8)]
    mixed = [(a, 1) for a in range(10)]

    if terms == 2:
        if mode == 0:
            # Asymmetric transfer from the successful dicyclic bed:
            # mainly rotation ratios in A, mainly mixed ratios in B.
            rotate = ((i + j) % 4 != 0) if side == "A" else (
                (i + j) % 4 == 0)
            ratios = rotations10 if rotate else mixed
        elif mode == 1:
            # A different order model, not an image of mode zero.
            ratios = rotations5 if (i + j) % 2 else mixed
        else:
            ratios = mixed
        g = G.pick(rng)
        return [g, G.mul(g, rng.choice(ratios))]

    # Trinomials must have noncommuting normalized supports.
    # No pair may differ by the central involution.
    central = (5, 0)
    for _ in range(256):
        cell = rng.sample(G.elements, terms)
        relative = [G.mul(G.inv(cell[0]), h) for h in cell[1:]]
        if not any(G.mul(u, v) != G.mul(v, u)
                   for t, u in enumerate(relative)
                   for v in relative[:t]):
            continue
        if any(G.mul(G.inv(cell[a]), cell[b]) == central
               for a in range(terms) for b in range(a)):
            continue
        return cell
    raise ValueError("noncommuting dicyclic cell policy unavailable")


def _copy_component(cells):
    return [[list(cell) for cell in row] for row in cells]


def _optimized_component(rng, G, profile, side, mode):
    """Independent multistart refinement of one prescribed architecture."""
    import math

    positions = [(i, j) for i, row in enumerate(profile)
                 for j, terms in enumerate(row) if terms]
    proposals = []
    for restart in range(3):
        cells = [[_dic_cell(rng, G, terms, side, i, j, mode)
                  for j, terms in enumerate(row)]
                 for i, row in enumerate(profile)]
        score = _four_cycles(_component_rows(G, side, cells))
        local_cells = _copy_component(cells)
        local_score = score

        for step in range(96):
            i, j = rng.choice(positions)
            old = cells[i][j]
            cells[i][j] = _dic_cell(
                rng, G, profile[i][j], side, i, j, mode)
            trial_score = _four_cycles(_component_rows(G, side, cells))
            delta = trial_score - score
            temperature = 16.0 * (1.0 - step / 96.0) + 0.25
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                score = trial_score
                if score < local_score:
                    local_score = score
                    local_cells = _copy_component(cells)
            else:
                cells[i][j] = old
            if step in (31, 63, 95):
                proposals.append(_copy_component(cells))
        proposals.append(local_cells)

    width = len(profile[0]) * G.q
    eligible = []
    seen = set()
    for serial, cells in enumerate(proposals):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in cells for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        rows = _component_rows(G, side, cells)
        if gf2_rank(rows) != len(rows):
            continue
        if not _connected_component(rows, width):
            continue
        eligible.append((_four_cycles(rows), serial, cells, rows))
    eligible.sort(key=lambda item: item[:2])

    best_key, best_cells = None, None
    for four, serial, cells, rows in eligible[:6]:
        minimum, penalty, boundary = _small_component_kernels(rows, width)
        key = (-minimum, penalty, boundary, four, serial)
        if best_key is None or key < best_key:
            best_key, best_cells = key, cells
    if best_cells is None:
        raise ValueError("no connected full-row-rank dicyclic component")
    return best_cells


def _short_quantum_logical(checks, stabilizers, width):
    """Exact existence test for a nontrivial logical of weight <=4."""
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

    cols = _column_syndromes(checks, width)
    masks = [1 << j for j in range(width)]
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = masks[i]
        elif nontrivial(previous ^ masks[i]):
            return True
    for i in range(width):
        for j in range(i):
            syndrome = cols[i] ^ cols[j]
            word = masks[i] | masks[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(previous ^ word):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Twenty support briefs; extra slots change relative-order policies."""
    policy = idx % 20
    mode = (idx // 20) % 3
    sa = _SUPPORT_PROFILES[policy % 5]
    sb = _SUPPORT_PROFILES[policy // 5]

    # Check the two independent extraction-weight budgets explicitly.
    wx = max(sum(row) for row in sa) + max(
        sum(row[j] for row in sb) for j in range(B_SHAPE[1]))
    wz = max(sum(row) for row in sb) + max(
        sum(row[j] for row in sa) for j in range(A_SHAPE[1]))
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    for attempt in range(4):
        try:
            A = _optimized_component(rng, G, sa, "A", mode)
            B = _optimized_component(rng, G, sb, "B", mode)
        except ValueError:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 580 and k > 0
                and 275 * n <= 1000 * k < 325 * n):
            continue
        if _short_quantum_logical(hx, hz, n):
            continue
        if _short_quantum_logical(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        return {"idx": idx, "policy": policy, "mode": mode, "k": k,
                "cells": {"A": A, "B": B}}
    raise ValueError("no dicyclic filling passed rank/connectivity/logical screens")


def CELL(rng, G, side, i, j, ctx):
    """All labels and term counts were jointly chosen in NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A screened d2s=6 implies basis-balanced circuit protection; parent slot 56 had Z 107/1250 failures versus X 53/200000.
# REFUTED: Balanced C20 degrees plus component-kernel-six selection ensure ceiling-level circuit fitness: all five measured parent codes reached the failed-shot cap in both bases, scoring only 5.52-5.67 despite ceilings above 7.
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
