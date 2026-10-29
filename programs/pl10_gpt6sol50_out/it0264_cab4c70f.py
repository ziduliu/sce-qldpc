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
CONCEPT = ("In cyclic Z29 of order 29 with A=(2,4), B=(2,4), screening binomial component-kernel words should reduce worse-basis circuit failures among [[580,132]] codes.")
GROUP = ("Keep the measured Z29 order-29 bed for a filling-only comparison; its equal-shape codes have reached k=132, but the current five-code fitness is 7.1079.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: q=29 gives n=580, two equal-length 116-variable components, and guaranteed k>=116, rate>=0.200. The parent's k=132 gives rate 0.228 in cell 0.250. Its failures enriched varying sector-1 Acol/Bcol data blocks, not one persistent check block.")
FILLING = ("Give every cell two distinct shifts, so HX and HZ rows have weight 12. Use separate indexed gap-spectrum motifs and screen both components for short words formed from three-column minors and their combinations; then check cycles and verified k=132. Test for fewer failed shots and decoder fall-throughs than the parent's 15 Z failures in slot 12 and 12 X failures in slot 6. Kernel screening is a selection hypothesis, not the fitness.")
FAMILY_NAME = "r025_z29_24x24_minor_kernel_screen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Additive cyclic group Z29, encoded as pairs.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def idx(g):
        return int(g[0]) % q

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 29 = 580


def _component_short_cycles(G, cells, side):
    """Count rooted simple Tanner cycles of lengths four, six and eight.

    Group translation maps every lift coordinate of a given check row
    onto every other one, so one root per protograph check row suffices.
    Use the component's actual left/right lift conventions.
    """
    q = G.q
    check_vars = [[] for _ in range(2 * q)]
    var_checks = [[] for _ in range(len(cells[0]) * q)]
    for i, row in enumerate(cells):
        for r in G.elements:
            check = i * q + G.idx(r)
            for j, cell in enumerate(row):
                for g in cell:
                    c = G.mul(G.inv(g), r) if side == "A" else G.mul(r, g)
                    variable = j * q + G.idx(c)
                    check_vars[check].append(variable)
                    var_checks[variable].append(check)

    adjacent = [[] for _ in check_vars]
    for check, variables in enumerate(check_vars):
        for variable in variables:
            adjacent[check].extend(
                (other, variable) for other in var_checks[variable]
                if other != check
            )

    fours = sixes = eights = 0
    identity = G.idx(G.e)
    for i in range(2):
        root = i * q + identity
        root_edges = {}
        for other, variable in adjacent[root]:
            old = root_edges.setdefault(other, [])
            fours += len(old)
            old.append(variable)

        paths = {}
        for middle, first_var in adjacent[root]:
            for end, second_var in adjacent[middle]:
                if end == root or second_var == first_var:
                    continue
                for last_var in root_edges.get(end, ()):
                    if last_var != first_var and last_var != second_var:
                        sixes += 1

                old_paths = paths.setdefault(end, [])
                for old_middle, old_first, old_second in old_paths:
                    if old_middle != middle and len({
                        first_var, second_var, old_first, old_second
                    }) == 4:
                        eights += 1
                old_paths.append((middle, first_var, second_var))
    return fours, sixes, eights


from functools import lru_cache
from itertools import combinations


@lru_cache(None)
def _gap_spectra():
    """Eight-gap sets, one representative per Z29 automorphism orbit."""
    seen = set()
    representatives = []
    for gaps in combinations(range(1, 15), 8):
        if gaps in seen:
            continue
        orbit = {
            tuple(sorted(min((u * d) % 29, 29 - (u * d) % 29)
                         for d in gaps))
            for u in range(1, 15)
        }
        seen.update(orbit)
        representatives.append(min(orbit))
    return tuple(sorted(representatives))


def _trial_cells(rng, G, idx):
    """Label a separate gap-orbit motif for each of the first 20 slots."""
    spectra = _gap_spectra()
    design, phase = idx % 20, idx // 20
    position = design * len(spectra) // 20
    gaps_a = spectra[(position + 3 + 5 * phase) % len(spectra)]
    gaps_b = spectra[(position + 51 + 7 * idx) % len(spectra)]

    def make_side(gaps):
        assigned = list(gaps)
        rng.shuffle(assigned)
        cells = []
        for i in range(2):
            row = []
            for j in range(4):
                start = rng.randrange(G.q)
                gap = assigned[4 * i + j]
                if rng.randrange(2):
                    gap = -gap
                row.append([(start, 0), ((start + gap) % G.q, 0)])
            cells.append(row)
        return cells

    return make_side(gaps_a), make_side(gaps_b)


def _component_pair_relations(G, cells, side):
    """Count short column dependencies under the component's lift action."""
    columns = []
    for j in range(4):
        for h in G.elements:
            syndrome = 0
            for i in range(2):
                for g in cells[i][j]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * G.q + G.idx(check))
            columns.append(syndrome)

    pairs = {}
    twos = repeated_pairs = 0
    for i, first in enumerate(columns):
        for second in columns[:i]:
            value = first ^ second
            twos += value == 0
            repeated_pairs += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
    return twos, repeated_pairs


def _rooted_full_cycles(rows, q):
    """Four- and simple six-cycles at one check per lifted check block.

    Translation of Z29 carries each check coordinate in a block to
    every other coordinate. A six-cycle uses three distinct checks and
    three distinct data qubits; exclude shared-qubit degeneracies.
    """
    sixes, fours = [], []
    for block in range(len(rows) // q):
        root_index = block * q
        root = rows[root_index]
        adjacent = []
        four = 0
        for index, row in enumerate(rows):
            if index == root_index:
                continue
            common = root & row
            if common:
                count = common.bit_count()
                four += count * (count - 1) // 2
                adjacent.append((index, common, count))

        six = 0
        for a, (u, ru, nu) in enumerate(adjacent):
            for v, rv, nv in adjacent[:a]:
                uv = rows[u] & rows[v]
                if not uv:
                    continue
                n_uv = uv.bit_count()
                triple = (ru & rv).bit_count()
                six += nu * nv * n_uv - triple * (
                    nu + nv + n_uv - 2
                )
        sixes.append(six)
        fours.append(four)
    return max(sixes), sum(sixes), max(fours), sum(fours)


def _minor_kernel_profile(cells, side):
    """Weights of explicit cyclic-component kernel words.

    The 2x3 minors give syzygies because Z29 is commutative. Each
    binomial is divisible by 1+x; divide the minors by that factor
    and test all nonzero combinations of the four resulting words.
    """
    q = 29
    whole = (1 << q) - 1

    def cyclic_product(left, right):
        out = 0
        while left:
            bit = left & -left
            shift = bit.bit_length() - 1
            out ^= ((right << shift) | (right >> (q - shift))) & whole
            left ^= bit
        return out

    def divide_even(poly):
        value = result = 0
        for t in range(q):
            value ^= (poly >> t) & 1
            if value:
                result |= 1 << t
        assert value == 0
        return result

    sign = 1 if side == "A" else -1
    f = [[sum(1 << ((sign * g[0]) % q) for g in cells[i][j])
          for j in range(4)] for i in range(2)]

    def minor(j, k):
        return cyclic_product(f[0][j], f[1][k]) ^ \
               cyclic_product(f[0][k], f[1][j])

    words = []
    for triple in combinations(range(4), 3):
        word = [0] * 4
        for j in triple:
            other = [k for k in triple if k != j]
            word[j] = divide_even(minor(*other))
        words.append(word)

    weights = []
    for choice in range(1, 16):
        combined = [0] * 4
        for t in range(4):
            if (choice >> t) & 1:
                for j in range(4):
                    combined[j] ^= words[t][j]
        weight = sum(min(v.bit_count(), q - v.bit_count())
                     for v in combined)
        if weight:
            weights.append(weight)
    weights.sort()
    return (weights[0], sum(w <= 13 for w in weights),
            sum(w <= 15 for w in weights), weights[min(4, len(weights) - 1)])


def NEW_CTX(rng, G, idx):
    """Screen kernel words first, then cycles and the verified LP rank."""
    supports = [[2] * 4 for _ in range(2)]
    assert _grid_weights(supports, supports) == 12
    assert _no_dead_lines(supports, supports)

    preliminary = []
    for _ in range(48 + idx % 9):
        A, B = _trial_cells(rng, G, idx)
        ap = _minor_kernel_profile(A, "A")
        bp = _minor_kernel_profile(B, "B")
        kernel_key = (-min(ap[0], bp[0]),
                      ap[1] + bp[1], ap[2] + bp[2],
                      -min(ap[3], bp[3]))
        preliminary.append((kernel_key, rng.random(), A, B))
    preliminary.sort(key=lambda item: item[:2])

    finalists = []
    for kernel_key, tie, A, B in preliminary[:12]:
        ac = _component_short_cycles(G, A, "A")
        bc = _component_short_cycles(G, B, "B")
        cycle_key = (max(ac[0], bc[0]), ac[0] + bc[0],
                     max(ac[1], bc[1]), ac[1] + bc[1],
                     max(ac[2], bc[2]), ac[2] + bc[2])
        finalists.append((kernel_key, cycle_key, tie, A, B))
    finalists.sort(key=lambda item: item[:3])

    best_key = None
    best = None
    for kernel_key, cycle_key, tie, A, B in finalists[:6]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        xc = _rooted_full_cycles(hx, G.q)
        zc = _rooted_full_cycles(hz, G.q)
        key = (k != 132, kernel_key, cycle_key[:4],
               max(xc[0], zc[0]), xc[0] + zc[0],
               cycle_key[4:], tie)
        if best_key is None or key < best_key:
            best_key, best = key, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: cyclic-order-21 rolled sparse fillings do not consistently
# suppress sector-1 weight-six logicals or worst-basis failures: the five
# measured parent slots scored only 3.91-4.63.
# Refuted: minimizing component four-cycles alone consistently prevents
# circuit failures on this D23 bed. Parent slot 42 had six X-basis failures
# in 200000 shots despite w10 and a d2s estimate of 22.
# Refuted: balanced-cycle D23 binomials guarantee zero failures; current
# slots 2, 6 and 27 had respectively 1, 2 and 1 Z-basis failed shots.
# Refuted: admitting singleton-balanced profiles alongside binomials
# consistently improves this Z37 bed's measured five-code score. Slot 1
# had k=83 and two Z failures; measured all-binomial slots 30, 36 and
# 54 each had k=88 and zero failures in both bases.
# Refuted: indexing all-binomial gap spectra and selecting by component
# cycles alone consistently improves this bed: fitness fell from 7.3872
# to 7.2909, with five Z failures across its five measured k=88 slots.
# Refuted: selecting rooted HX-cycle minima on Z37 guarantees zero Z failures; the current five measured slots had 2, 1, 1, 4 and 1 Z-basis failures.
# Refuted: distinct Z29 gap spectra screened by component relations and
# rooted cycles reliably improve this bed: fitness was 7.1079; measured
# slots 12 and 6 still had 15 Z and 12 X failures, respectively.
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
