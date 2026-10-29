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
CONCEPT = ("AGL(1,11), order 110, A=(1,3), B=(1,4): connected "
           "binomial A and two-trinomial B, screened through component "
           "weight five and small mixed-sector cancellations, should give "
           "five new distance readings >=6 with k>=663, improving 16.615.")
GROUP = ("Frozen AGL(1,11), q=110. This bed repeatedly measures distance "
         "six; retain it and target reliable sixes plus a verified rank "
         "deficit, rather than infer larger distance from unrelated beds. "
         "The related frontier filling has k=663 and distance six.")
SHAPES = ("Frozen A=(1,3), B=(1,4): n=1430, k>=660, guaranteed rate "
          "6/13=0.461538 in cell 0.450. Components have 330 and 440 "
          "variables with 110 checks. Current minimum X witnesses are "
          "sector-1 operators, especially using B columns 0..2 across "
          "A columns 1/2; those are the three binomial B columns.")
FILLING = ("Use A=(2,2,2), B=(2,2,3,3): row sums six and ten, so "
           "the CSS maxima are nine and twelve. Twenty cards cross four "
           "A multiplier patterns with five B relative geometries, including "
           "order-eleven translations and mixed odd/even trinomials. "
           "Verified components exclude kernels through weight five when "
           "available. Test small mixed-sector cancellations and gate exact "
           "full ranks for k>=663 in cell 0.450. Valid lower-quality "
           "fallbacks remain; no distance above six is claimed.")
HOW_AB = FILLING
FAMILY_NAME = "r045_agl1F11_13x14_binA_twoTriB_kernel5_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 110

def custom_ops(q):
    # AGL(1,11): (a,b) acts as t -> 2^b*t+a modulo 11.
    # 2 has multiplicative order 10 modulo 11.
    if not valid_q(q):
        raise ValueError("AGL(1,11) requires order 110")
    powers = tuple(pow(2, b, 11) for b in range(10))
    els = [(a, b) for b in range(10) for a in range(11)]

    def red(g):
        return (g[0] % 11, g[1] % 10)

    def mul(g, h):
        return ((g[0] + powers[g[1]] * h[0]) % 11,
                (g[1] + h[1]) % 10)

    def inv(g):
        b = (-g[1]) % 10
        return ((-powers[b] * g[0]) % 11, b)

    def idx(g):
        a, b = red(g)
        return 11*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


# Five ternary relative-order spectra, not automorphism images:
# mode 0: (5,10,10); mode 1: (5,5,5);
# mode 2: (10,10,11); mode 3: (5,5,11);
# mode 4: (2,5,10).
# Cross these with the count of order-five B binomial ratios.
_ODD_V = (1, 3, 7, 9)
_EVEN_V = (2, 4, 6, 8)


def _affine_trinomial(rng, G, mode):
    """Anchor one term; preserve the prescribed relative-order spectrum."""
    if mode == 0:
        v, w = rng.sample(_ODD_V, 2)
        g, h = (rng.randrange(11), v), (rng.randrange(11), w)
    elif mode == 1:
        v, w = rng.sample(_EVEN_V, 2)
        g, h = (rng.randrange(11), v), (rng.randrange(11), w)
    elif mode in (2, 3):
        pool = _ODD_V if mode == 2 else _EVEN_V
        g = (rng.randrange(1, 11), 0)
        h = (rng.randrange(11), rng.choice(pool))
    else:
        v = rng.choice(_ODD_V)
        g = (rng.randrange(11), v)
        h = (rng.randrange(11), (v + 5) % 10)
    cell = [G.e, g, h]
    assert len({G._key(x) for x in cell}) == 3
    return cell


def _affine_side(rng, G, side, card):
    """Twenty deliberate order-spectrum cards with a full-action bridge."""
    mode, five = card % 5, card // 5
    if side == "A":
        # The first column's odd multipliers bridge the index-two subgroup.
        return [_affine_trinomial(rng, G, 0),
                _affine_trinomial(rng, G, mode),
                _affine_trinomial(rng, G, mode)]
    cells = []
    for j in range(3):
        pool = _EVEN_V if j < five else _ODD_V
        cells.append([G.e, (rng.randrange(11), rng.choice(pool))])
    cells.append(_affine_trinomial(rng, G, 0))
    return cells


def _verified_single_component(G, cells, side):
    """Use the scaffold lift, with an empty dummy side for extraction only."""
    if side == "A":
        hx, _, _ = lifted_matrices(G, [cells], [[[]]])
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, [[[]]], [cells])
        rows = hz
    width = len(cells) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _component_kernel4(rows, width):
    """Exact exclusion of component kernel words of weights one through four."""
    columns = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            columns[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, a in enumerate(columns):
        for b in columns[:i]:
            syndrome = a ^ b
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _component_geometry(rows):
    """Connectivity and exact four/six cycles of a verified bipartite lift."""
    count = len(rows)
    common = [[0] * count for _ in range(count)]
    upper = [[] for _ in rows]
    adjacent = [0] * count
    four = 0
    for a in range(count):
        for b in range(a + 1, count):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                adjacent[a] |= 1 << b
                adjacent[b] |= 1 << a
                four += overlap * (overlap - 1) // 2

    reached = frontier = 1
    while frontier:
        neighbors = 0
        while frontier:
            bit = frontier & -frontier
            neighbors |= adjacent[bit.bit_length() - 1]
            frontier ^= bit
        frontier = neighbors & ~reached
        reached |= frontier
    if reached != (1 << count) - 1:
        return None

    six = 0
    for a, neighbors in enumerate(upper):
        for t, b in enumerate(neighbors):
            ab = common[a][b]
            for c in neighbors[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    six += ab * ac * bc - triple * (ab + ac + bc - 2)
    return four, six


def _kernel_witness_reading(rows, width, rng, trials):
    """Randomized information-set search for genuine classical kernel words.

    A high reading means this search missed lighter words. These are
    component witnesses, not certified quantum logical distances.
    """
    best = width + 1
    for _ in range(trials):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapping = [1 << p for p in permutation]
        pivots = {}
        for source in rows:
            row = 0
            while source:
                bit = source & -source
                row ^= mapping[bit.bit_length() - 1]
                source ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ordered = sorted(pivots)
        for at, p in enumerate(ordered):
            bit = 1 << p
            reduced = pivots[p]
            for higher in ordered[at + 1:]:
                if pivots[higher] & bit:
                    pivots[higher] ^= reduced

        words = []
        for f in range(width):
            if f in pivots:
                continue
            bit = 1 << f
            word = bit
            for p in ordered:
                if pivots[p] & bit:
                    word |= 1 << p
            best = min(best, word.bit_count())
            words.append(word)
        words.sort(key=int.bit_count)
        shortlist = words[:24]
        for i, word in enumerate(shortlist):
            for other in shortlist[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 5:
            break
    return best


def _screened_component_pool(rng, G, side, card):
    """Compare independent fillings of one prescribed order-spectrum card."""
    pool = []
    width = (3 if side == "A" else 4) * G.q
    for trial in range(96):
        cells = _affine_side(rng, G, side, card)
        rows = _verified_single_component(G, cells, side)
        geometry = _component_geometry(rows)
        if geometry is None:
            continue
        if not _component_kernel4(rows, width):
            continue
        # Full component rank avoids selecting rank-deficient fallbacks.
        if gf2_rank(rows) != G.q:
            continue
        pool.append((geometry, trial, cells, rows))
        if len(pool) >= 24:
            break
    if not pool:
        raise ValueError("no connected full-rank component passed kernel-4")

    pool.sort(key=lambda record: (record[0], record[1]))
    ranked = []
    for geometry, trial, cells, rows in pool[:6]:
        reading = _kernel_witness_reading(rows, width, rng, 8)
        key = (reading, -geometry[0], -geometry[1])
        ranked.append((key, trial, cells, rows, geometry))
    ranked.sort(key=lambda record: record[0], reverse=True)

    finalists = []
    for key, trial, cells, rows, geometry in ranked[:3]:
        reading = min(key[0], _kernel_witness_reading(
            rows, width, rng, 32))
        final_key = (reading, -geometry[0], -geometry[1])
        finalists.append((final_key, trial, cells))
    finalists.sort(key=lambda record: record[0], reverse=True)
    return finalists


def NEW_CTX(rng, G, idx):
    """Twenty ratio-geometry cards; all matrix screens use verified lifts."""
    q = G.q
    card, cohort = idx % 20, idx // 20
    mode, deck = card % 4, card // 4
    mask = (1 << q) - 1
    sa, sb = [[2, 2, 2]], [[2, 2, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    a_patterns = ((1, 3, 7), (1, 1, 3),
                  (3, 3, 7), (0, 3, 7))
    tri_patterns = (
        ((1, 2), (3, 4)),
        ((3, 2), (7, 6)),
        ((1, 3), (7, 9)),
        ((2, 4), (3, 4)),
        ((0, 3), (0, 7)),
    )

    def element(b):
        # Nonidentity translations have order 11; odd non-five
        # multipliers have order 10. No binomial ratio has order <=5.
        return (rng.randrange(1, 11) if b == 0
                else rng.randrange(11), b)

    def draw(side):
        if side == "A":
            return [[G.e, element(b)] for b in a_patterns[mode]]
        if deck in (1, 3):
            bins = (0, rng.choice((3, 7)))
        else:
            bins = tuple(rng.sample(_ODD_V, 2))
        cells = [[G.e, element(b)] for b in bins]
        for v, w in tri_patterns[deck]:
            cell = [G.e, element(v), element(w)]
            assert len({G._key(g) for g in cell}) == 3
            cells.append(cell)
        return cells

    def columns(rows, width):
        answer = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                answer[bit.bit_length() - 1] |= 1 << r
                source ^= bit
        return answer

    def pair_table(cols):
        """Exact exclusion through four, retaining all pair syndromes."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        pairs = set()
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs.add(syndrome)
        return singles, pairs

    def graph_record(cols):
        """Verified degree-two graph, bipartiteness and closed-walk counts."""
        adjacency = [[] for _ in range(q)]
        for syndrome in cols:
            assert syndrome.bit_count() == 2
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            adjacency[a].append(b)
            adjacency[b].append(a)
        color = {0: 0}
        queue = [0]
        bipartite = True
        for a in queue:
            for b in adjacency[a]:
                if b not in color:
                    color[b] = 1 - color[a]
                    queue.append(b)
                elif color[b] == color[a]:
                    bipartite = False
        if len(color) != q:
            return None
        counts = [0] * q
        counts[0] = 1
        spectrum = []
        for length in range(1, 9):
            nxt = [0] * q
            for a, value in enumerate(counts):
                if value:
                    for b in adjacency[a]:
                        nxt[b] += value
            counts = nxt
            if length in (6, 8):
                spectrum.append(counts[0])
        return bipartite, tuple(spectrum)

    def free_five(cols, pairs, nc):
        """Exact classical weight-five test using regular-action anchors.

        A component's regular group action is transitive within each
        variable block. Translate one member of any five-word to an
        anchor, then match its triple syndrome against all pairs.
        This symmetry is NOT assumed for the full nonabelian LP code.
        """
        width = len(cols)
        for root in range(0, nc * q, q):
            cr = cols[root]
            for i in range(width):
                if i == root:
                    continue
                value = cr ^ cols[i]
                for j in range(i):
                    if j != root and (value ^ cols[j]) in pairs:
                        return False
        return True

    def component_pool(side):
        nc = 3 if side == "A" else 4
        pool, good = [], 0
        for trial in range(192):
            cells = draw(side)
            rows = _verified_single_component(G, cells, side)
            cols = columns(rows, nc * q)
            table = pair_table(cols)
            if table is None:
                continue
            singles, pairs = table
            if side == "A":
                graph = graph_record(cols)
                if graph is None:
                    continue
                bipartite, spectrum = graph
                clean = bipartite or free_five(cols, pairs, nc)
                # A connected incidence matrix has rank q-1.
                assert gf2_rank(rows) == q - 1
                cycle_key = spectrum if (mode + cohort) % 2 == 0 else (
                    spectrum[0], spectrum[1] + spectrum[0])
            else:
                geometry = _component_geometry(rows)
                if geometry is None:
                    continue
                clean = free_five(cols, pairs, nc)
                cycle_key = geometry
            quality = (int(clean), -cycle_key[0], -cycle_key[1])
            pool.append((quality, trial, cells, cols, singles, pairs))
            pool.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            del pool[8:]
            good += int(clean)
            if trial >= 23 and good >= 8:
                break
        if not pool:
            raise ValueError("no component passed the through-four screen")
        return pool

    def mixed_five_free(hx, hz, n, bsingles, bpairs):
        """Check the remaining small mixed-sector possibilities.

        Each A binomial is an incidence graph of cycles of length 10 or
        11. Each B binomial likewise has no kernel word through five.
        B trinomials have distinct columns and odd column weight.

        Consequently HX has no mixed word through five: odd sector-2
        counts fail parity in a trinomial block; two sector-2 variables
        leave all four check blocks nonzero; four leave both binomial
        blocks nonzero. For HZ, only one or two sector-2 variables can
        fit the budget. Their residuals below come from the FULL lift.
        """
        n1 = 12 * q
        zcols = columns(hz, n)
        sector2 = zcols[n1:]

        # Confirm the distinct-column premise directly on both matrices.
        xcols = columns(hx, n)
        for cols, blocks in ((xcols[n1:], 4), (sector2, 3)):
            for block in range(blocks):
                local = [(word >> (block * q)) & mask for word in cols]
                if 0 in local or len(set(local)) != q:
                    return False

        # One sector-2 variable needs a nonzero cancellation in each
        # of three A-column check blocks: at most one may use a pair.
        for word in sector2:
            cost = 0
            for block in range(3):
                residual = (word >> (block * q)) & mask
                if residual in bsingles:
                    cost += 1
                elif residual in bpairs:
                    cost += 2
                else:
                    cost = 100
                    break
            if cost <= 4:
                return False

        # Two sector-2 variables leave all three blocks nonzero, so
        # a total weight <=5 requires exactly one singleton per block.
        for i, word in enumerate(sector2):
            for other in sector2[:i]:
                residual = word ^ other
                if all(((residual >> (block * q)) & mask) in bsingles
                       for block in range(3)):
                    return False
        return True

    apool, bpool = component_pool("A"), component_pool("B")
    combinations = []
    for a in apool:
        for b in bpool:
            key = (a[0][0] + b[0][0], min(a[0][0], b[0][0]),
                   a[0][1] + b[0][1], a[0][2] + b[0][2])
            combinations.append((key, a, b))
    combinations.sort(key=lambda item: item[0], reverse=True)

    fallback = None
    for quality, a, b in combinations:
        A, B = [a[2]], [b[2]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= 663 and 425*n <= 1000*k < 475*n):
            continue
        candidate = (quality, A, B, hx, hz, n, k)
        if fallback is None:
            fallback = candidate
        if quality[0] != 2:
            continue
        if mixed_five_free(hx, hz, n, b[4], b[5]):
            fallback = candidate
            break
    if fallback is None:
        raise ValueError("no new filling met the exact dimension/rate gate")

    quality, A, B, hx, hz, n, k = fallback
    assert n == 1430
    assert max(row.bit_count() for row in hx + hz) <= 12
    if not css_ok(hx, hz):
        raise ValueError("verified construction failed CSS")
    return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
            "k": k, "quality": quality}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-saturating rolled supports on Z26:Z2, (2,6)x(2,4), did not suppress small logicals: all five measured codes had d=4, with abundant sector-1 witnesses.
# REFUTED: Degree>=3 columns and local syndrome-collision selection on Z17:Z2, (2,5)x(3,7), did not eliminate weight-10 sector-1 logical families: all five measured codes returned d=10 across many Acol/Bcol blocks.
# REFUTED: All-trinomial A with three-binomial B and order-spectrum/component-four screening on AGL(1,11), (1,3)x(1,4), raised this bed beyond six: all five measured minima remained six, concentrated on the binomial B columns, with k-kb=0.
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
