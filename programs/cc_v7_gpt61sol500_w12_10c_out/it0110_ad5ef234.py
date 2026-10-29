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
CONCEPT = ("SL(2,3), order 24, shapes (1,3)x(2,7): noncommuting trinomials and only three singleton/singleton B columns should permit exclusion of component dependencies through weight six and improve the parent's five distance-six readings in cell 0.450.")
GROUP = ("Replace Dic6 by the genuinely different determinant-one matrix group SL(2,F3). It is an index-two subgroup of GL(2,F3), whose related order-48 bed reached fitness 38.25; transfer is a hypothesis, not evidence of superiority. Order-four/six elements provide new noncommuting relative-label patterns. Order 24 retains n=552.")
SHAPES = ("Retain A=(1,3), B=(2,7): n=552, kb=240 and guaranteed rate 10/23 in cell 0.450. Component variable/check counts remain 72/24 and 168/48. All five measured parent minima had weight six and occupied sector-1 Acol/Bcol blocks, with broad participation across both components. Verified ranks require 0.425<=k/n<0.475.")
FILLING = ("Twenty cards cross four balanced dense B geometries with five noncommuting triple palettes. A row sum is nine; B column sums are two or three and both B row sums are nine, hence both CSS row-weight maxima are twelve. Screen verified component kernels through four, refine actual kernel witnesses, and count exact weight-six dependencies after excluding weights through five. Require full commuting-vector exclusion through four. Weighted check-overlap triangle profiles strengthen permutation-invariant deduplication. Component exclusions and witness readings are not quantum-distance certificates.")
FAMILY_NAME = "r045_SL2F3_13x27_noncommuting_dense_exact6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # SL(2,F3), row-major matrix encoding; determinant exactly one.
    if not valid_q(q):
        raise ValueError("SL(2,F3) requires order 24")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3 == 1]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = red(g)
        e,f,u,v = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a,b,c,d = red(g)
        return (d, (-b) % 3, (-c) % 3, a)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)

from functools import lru_cache


def _element_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid element order")


# Each pair specifies orders of the two identity-relative terms.
# The terms must not commute; their mutual difference must not be an involution.
_TRIPLE_MODELS = (
    ((6, 6), (6, 6), (6, 6)),
    ((4, 6), (4, 6), (4, 6)),
    ((3, 6), (4, 6), (6, 6)),
    ((4, 4), (4, 6), (6, 6)),
    ((3, 3), (3, 6), (4, 6)),
)

_B_COLUMN_PLANS = (
    ((2,1), (2,1), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (0,3), (2,1), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (3,0), (0,3), (0,3), (1,1), (1,1), (1,1)),
)


def _support_recipe(profile):
    columns = _B_COLUMN_PLANS[profile]
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in columns] for i in range(2)]
    return sa, sb


def _sidon_relatives(rng, G, count, orders, prefer_six):
    """Pack mono-column relatives; exact component tests remain authoritative."""
    # All cards exclude order-two/three pair differences when a packing exists.
    # Some cards first attempt the stricter all-order-six geometry.
    for strict in ((True, False) if prefer_six else (False,)):
        for _ in range(64):
            selected = [G.e]
            differences = set()
            pool = list(G.elements)
            rng.shuffle(pool)
            for g in pool:
                if g in selected:
                    continue
                added = []
                for h in selected:
                    added.extend((G.mul(g, G.inv(h)),
                                  G.mul(h, G.inv(g))))
                if any(orders[d] < 4 or (strict and orders[d] != 6)
                       for d in added):
                    continue
                if len(set(added)) != len(added):
                    continue
                if any(d in differences for d in added):
                    continue
                selected.append(g)
                differences.update(added)
                if len(selected) == count:
                    rng.shuffle(selected)
                    return selected
    # This is only a proposal; it receives the same mandatory exact screens.
    return rng.sample(G.elements, count)


def _draw_triple(rng, G, palette, triple_pools):
    x, y = rng.choice(triple_pools[palette])
    return [G.e, x, y]


def _draw_grids(rng, G, sb, triple_model, double_order,
                order_pools, triple_pools, orders):
    A = [[_draw_triple(rng, G, palette, triple_pools)
          for palette in triple_model]]
    B = [[None] * 7 for _ in range(2)]
    mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
    relatives = _sidon_relatives(
        rng, G, len(mono), orders, double_order == 6)
    for j, relative in zip(mono, relatives):
        # This gives top * bottom^-1 = relative in the verified B action.
        B[0][j], B[1][j] = [G.e], [G.inv(relative)]
    for i in range(2):
        for j in range(7):
            if B[i][j] is not None:
                continue
            count = sb[i][j]
            if count == 0:
                B[i][j] = []
            elif count == 1:
                B[i][j] = [G.pick(rng)]
            elif count == 2:
                anchor = G.pick(rng)
                difference = rng.choice(order_pools[double_order])
                B[i][j] = [anchor, G.mul(anchor, difference)]
            else:
                palette = triple_model[(i + j + 1) % 3]
                anchor = G.pick(rng)
                B[i][j] = [
                    G.mul(anchor, g)
                    for g in _draw_triple(rng, G, palette, triple_pools)
                ]
    return A, B


def _syndrome_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _component_columns(hx, hz, q):
    """Extract both components from the scaffold's verified actions."""
    mask = (1 << q) - 1
    ar = []
    for r in range(q):
        row = hx[r]             # A row zero, B column zero
        ar.append(sum(((row >> (j*7*q)) & mask) << (j*q)
                      for j in range(3)))
    br = [row & ((1 << (7*q)) - 1) for row in hz[:2*q]]
    return (_syndrome_columns(ar, 3*q),
            _syndrome_columns(br, 7*q))


def _component_cycles(columns):
    overlaps = {}
    cycles = 0
    for syndrome in columns:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(checks):
            for b in checks[:i]:
                key = (b, a)
                old = overlaps.get(key, 0)
                cycles += old
                overlaps[key] = old + 1
    return cycles


def _kernel_reading(rng, columns, trials):
    """Actual component-kernel witnesses; returned minima are upper bounds."""
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
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        short = sorted(words, key=int.bit_count)[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 3:
            break
    return best


def _component_short_floor(columns):
    """Exact component-kernel existence through four; five is a sentinel."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_six_profile(columns):
    """Exact exclusion through five and exact weight-six kernel count.

    After exclusion through four, a triple colliding with a singleton/pair
    detects weight five. Otherwise colliding triples are disjoint, and
    each weight-six support contributes ten unordered triple partitions.
    Seven denotes exclusion through six, not a distance estimate.
    """
    floor = _component_short_floor(columns)
    if floor < 5:
        return floor, 0
    base = {0}
    base.update(columns)
    for i, x in enumerate(columns):
        for y in columns[:i]:
            base.add(x ^ y)
    counts = {}
    hits = 0
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for t in range(j):
                syndrome = pair ^ columns[t]
                if syndrome in base:
                    return 5, 0
                old = counts.get(syndrome, 0)
                hits += old
                counts[syndrome] = old + 1
    assert hits % 10 == 0
    return (6, hits // 10) if hits else (7, 0)


def _small_kernel_stats(rows, n):
    """Exact commuting-vector screening through weight three.

    With distinct nonzero column syndromes, equal pair syndromes
    involve disjoint pairs. Every weight-four kernel vector has
    exactly three pair partitions, permitting exact counting.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= bit
            row ^= lo

    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None

    pairs = {}
    for i, a in enumerate(cols):
        for j in range(i + 1, n):
            syndrome = a ^ cols[j]
            if syndrome in singles:
                return None
            packed = i * n + j
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = packed
            elif isinstance(previous, int):
                pairs[syndrome] = [previous, packed]
            else:
                previous.append(packed)

    histogram = {}
    participation = [0] * n
    collisions = 0
    for syndrome, value in pairs.items():
        count = 1 if isinstance(value, int) else len(value)
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
        if count > 1:
            collisions += count * (count - 1) // 2
            for packed in value:
                i, j = divmod(packed, n)
                participation[i] += count - 1
                participation[j] += count - 1

    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in cols),
            tuple(p // 3 for p in participation))


def _overlap_triangle_profile(rows):
    """Permutation-invariant weighted check-overlap projection profiles.

    These are projection-graph triangles, not a claimed Tanner-cycle census.
    They distinguish some designs whose low-weight syndrome histograms agree.
    """
    size = len(rows)
    adjacency = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap
    triangles = [0] * size
    for i in range(size):
        lower = [j for j in adjacency[i] if j < i]
        for j in lower:
            for t in adjacency[j]:
                if t < j and t in adjacency[i]:
                    weight = (adjacency[i][j] * adjacency[j][t]
                              * adjacency[i][t])
                    triangles[i] += weight
                    triangles[j] += weight
                    triangles[t] += weight
    return tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(adjacency[i].values())),
         triangles[i])
        for i in range(size)))


def _incidence_fingerprint(hx, hz, sx, sz):
    # Invariant under qubit/check permutations and a global X/Z swap.
    joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
    swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
    row_profiles = tuple(sorted((
        tuple(sorted(r.bit_count() for r in hx)),
        tuple(sorted(r.bit_count() for r in hz)))))
    projections = tuple(sorted((
        _overlap_triangle_profile(hx),
        _overlap_triangle_profile(hz))))
    return (tuple(sorted((sx[1], sz[1]))),
            min(joint, swapped), row_profiles, projections)


@lru_cache(maxsize=1)
def _design_bank(G):
    """Four support geometries x five triple palettes; no orbit fallback."""
    orders = {g: _element_order(G, g) for g in G.elements}
    order_pools = {
        order: [g for g in G.elements if orders[g] == order]
        for order in (3, 4, 6)
    }
    palettes = {pair for model in _TRIPLE_MODELS for pair in model}
    triple_pools = {}
    for palette in palettes:
        ox, oy = palette
        pairs = [
            (x, y) for x in order_pools[ox] for y in order_pools[oy]
            if x != y and G.mul(x, y) != G.mul(y, x)
            and orders[G.mul(G.inv(x), y)] > 2
        ]
        if not pairs:
            raise ValueError("empty noncommuting triple palette")
        triple_pools[palette] = pairs

    fingerprints = set()
    bank = []
    for slot in range(N_SLOTS):
        recipe = slot % 20
        profile, mode = divmod(recipe, 5)
        triple_model = _TRIPLE_MODELS[mode]
        double_order = (6, 6, 6, 4, 4)[mode]
        sa, sb = _support_recipe(profile)
        assert sum(sa[0]) == 9
        assert all(sum(row) == 9 for row in sb)
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        rng = random.Random(450240731 + 100003*slot)

        def merit(item):
            return (min(item["da"], item["db"]),
                    item["da"] + item["db"], -item["cycles"],
                    -abs(item["k"] - 240))

        winner = None
        for batch in range(2):
            pool = []
            for attempt in range(48):
                A, B = _draw_grids(
                    rng, G, sb, triple_model, double_order,
                    order_pools, triple_pools, orders)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (425*n <= 1000*k < 475*n):
                    continue
                ac, bc = _component_columns(hx, hz, G.q)
                if _component_short_floor(ac) != 5:
                    continue
                if _component_short_floor(bc) != 5:
                    continue
                pool.append({
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": _kernel_reading(rng, ac, 8),
                    "db": _kernel_reading(rng, bc, 8),
                    "cycles": _component_cycles(ac) + _component_cycles(bc),
                })

            finalists = sorted(pool, key=merit, reverse=True)[:6]
            for item in finalists:
                item["da"] = min(
                    item["da"], _kernel_reading(rng, item["ac"], 32))
                item["db"] = min(
                    item["db"], _kernel_reading(rng, item["bc"], 32))

            for item in sorted(finalists, key=merit, reverse=True)[:4]:
                fa, ha = _component_six_profile(item["ac"])
                fb, hb = _component_six_profile(item["bc"])
                # An exact dependency can cap a missed sampled reading.
                if fa < 7:
                    item["da"] = min(item["da"], fa)
                if fb < 7:
                    item["db"] = min(item["db"], fb)

                hx, hz, n = item["hx"], item["hz"], item["n"]
                sx = _small_kernel_stats(hx, n)
                if sx is None or sx[0]:
                    continue
                sz = _small_kernel_stats(hz, n)
                if sz is None or sz[0]:
                    continue
                # Mandatory full commuting-vector exclusion through four;
                # component screening alone does not justify this test.
                fingerprint = _incidence_fingerprint(hx, hz, sx, sz)
                if fingerprint in fingerprints:
                    continue
                objective = (
                    min(fa, fb), min(item["da"], item["db"]),
                    -ha-hb, item["da"] + item["db"],
                    -item["cycles"], item["k"],
                )
                if winner is None or objective > winner[0]:
                    winner = (objective, item, fingerprint)
            # Only exact exclusion through six on BOTH components ends
            # the search early; a high witness reading alone never does.
            if winner is not None and winner[0][0] == 7:
                break

        if winner is None:
            bank.append(None)
            continue
        _, item, fingerprint = winner
        if not css_ok(item["hx"], item["hz"]):
            raise ValueError("verified lift failed CSS check")
        assert max(row.bit_count() for row in item["hx"] + item["hz"]) <= 12
        assert all(len(cell) == len(set(cell))
                   for grid in (item["A"], item["B"])
                   for row in grid for cell in row)
        fingerprints.add(fingerprint)
        bank.append((item["A"], item["B"]))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five fresh full-kernel-four-clean designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the independently optimized filling of this indexed recipe."""
    bank = _design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("recipe found no fresh rate-legal screened filling")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free rotation binomials plus one trinomial on each D34 (1,4) component raise measured distance above four; all five measured parent codes had d30s=4 and sector-1 minimum logicals.
# REFUTED: Dic6's component-witness and four-kernel objective removes sector-1 weight-six logicals; all five measured codes instead had d30s=6, with both minimum families entirely in sector 1.
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
