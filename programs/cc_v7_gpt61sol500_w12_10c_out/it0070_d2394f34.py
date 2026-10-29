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
CONCEPT = ("Dic6, order 24, A=(1,3), B=(2,5): removing degree-two B columns and selecting both verified components against weight-six kernels should improve the five-code score in cell 0.350.")
GROUP = ("Keep the exact Dic6 group: its parent scored 15.7824 with five measured distance-six readings. The new size factor 17 gives n=408 at the existing order 24, so no order change is required. This isolates a shape-and-filling experiment without assuming that related dicyclic beds transfer their distances.")
SHAPES = ("Target cell 0.350: (1,3)x(2,5) gives n=408, kb=144 and guaranteed rate 6/17=0.352941. Component variable/check counts are 72/24 and 120/48. All five parental minimum-logical families occupied sector-1 Acol/Bcol blocks. Shortening B permits degree three in every column; exact scaffold ranks enforce 0.325<=k/n<0.375.")
FILLING = ("Twenty recipes cross four B support geometries with five mixed-trinomial rotation-pair schedules. A row sum is nine; all B column sums are three and B row sums are eight/seven, giving HX weight twelve and HZ weight at most eleven. Independently optimize components extracted from verified lifts using exhaustive kernel screening through weight six, refine sampled witnesses, then combine rank-eligible finalists and screen complete commuting kernels through weight four. Distinct permutation-invariant incidence fingerprints exclude emitted orbit copies. Component screening is not a quantum-distance certificate.")
FAMILY_NAME = "r035_dic6_13x25_degree3_exact6_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Dic6: x^12=1, y^2=x^6, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic6 requires order 24")
    els = [(a, b) for b in range(2) for a in range(12)]

    def red(g):
        a, b = g
        return ((a + 6 * (b // 2)) % 12, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 6*b*d) % 12,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 12, 0) if b == 0 else ((a + 6) % 12, 1)

    def idx(g):
        a, b = red(g)
        return 12*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

from functools import lru_cache


def _element_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid element order")


# Five distinct rotation-pair order multisets; all triples mix cosets.
_TRIPLE_MODELS = (
    (12, 12, 12),
    (12, 12, 6),
    (12, 6, 3),
    (12, 3, 3),
    (6, 12, 6),
)


def _support_recipe(geometry):
    # All columns have degree three. The number of pure columns is
    # respectively 0, 1, 2, 3: these are not permutation copies.
    profiles = (
        ((2, 1), (2, 1), (2, 1), (1, 2), (1, 2)),
        ((3, 0), (2, 1), (1, 2), (1, 2), (1, 2)),
        ((3, 0), (0, 3), (2, 1), (2, 1), (1, 2)),
        ((3, 0), (3, 0), (0, 3), (1, 2), (1, 2)),
    )
    columns = profiles[geometry]
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in columns] for i in range(2)]
    return sa, sb


def _sidon_relatives(rng, G, count):
    """Distinct ordered differences exclude four-cycles among mono columns."""
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
            if len(set(added)) != len(added):
                continue
            if any(d in differences for d in added):
                continue
            selected.append(g)
            differences.update(added)
            if len(selected) == count:
                rng.shuffle(selected)
                return selected
    # A fallback is screened on the complete matrices, not silently accepted.
    return rng.sample(G.elements, count)


def _draw_triple(rng, G, pair_order, order_pools):
    if pair_order is None:
        rotations = [g for g in G.elements if g[1] == 0]
        for _ in range(128):
            cell = rng.sample(rotations, 3)
            orders = [
                _element_order(G, G.mul(G.inv(cell[i]), cell[j]))
                for i in range(3) for j in range(i)
            ]
            if 2 not in orders and 12 in orders:
                break
        else:
            cell = [(0, 0), (1, 0), (3, 0)]
    else:
        a = rng.randrange(12)
        delta = rng.choice(order_pools[pair_order])
        cell = [(a, 0), G.mul((a, 0), delta),
                (rng.randrange(12), 1)]
    # Independent cell gauges diversify sampling but are not counted as designs.
    gauge = G.pick(rng)
    return [G.mul(gauge, g) for g in cell]


def _draw_grids(rng, G, sb, triple_model, double_order, order_pools):
    A = [[_draw_triple(rng, G, kind, order_pools)
          for kind in triple_model]]
    rb, cb = B_SHAPE
    B = [[[] for _ in range(cb)] for _ in range(rb)]
    for i in range(rb):
        for j in range(cb):
            count = sb[i][j]
            if count == 0:
                continue
            if count == 3:
                kind = triple_model[(i + j) % len(triple_model)]
                B[i][j] = _draw_triple(rng, G, kind, order_pools)
            else:
                g = G.pick(rng)
                B[i][j] = ([g] if count == 1 else
                           [g, G.mul(g, rng.choice(
                               order_pools[double_order]))])
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
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return (_syndrome_columns(ar, ca * q),
            _syndrome_columns(br, cb * q))


def _component_six_screen(columns, cap=3600):
    """Exact exclusion through six; capped multiplicity is a heuristic.

    Every column here has weight three, so every kernel word is even.
    After excluding weights two/four, distinct colliding triples are
    disjoint. Each weight-six word has ten unordered triple partitions.
    Flag eight means no component kernel word of weight below eight;
    it does not establish the quantum distance.
    """
    assert all(c.bit_count() == 3 for c in columns)
    if len(set(columns)) != len(columns):
        return 2, 1

    pairs = set()
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in pairs:
                return 4, 1
            pairs.add(syndrome)

    triples = {}
    hits = 0
    for i in range(2, len(columns)):
        x = columns[i]
        for j in range(1, i):
            xy = x ^ columns[j]
            for h in range(j):
                syndrome = xy ^ columns[h]
                old = triples.get(syndrome, 0)
                hits += old
                if hits >= cap:
                    return 6, cap // 10
                triples[syndrome] = old + 1
    return (6, hits // 10) if hits else (8, 0)


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


def _incidence_fingerprint(hx, hz, sx, sz):
    # Invariant under qubit/check permutations and a global X/Z swap.
    joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
    swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
    row_profiles = tuple(sorted((
        tuple(sorted(r.bit_count() for r in hx)),
        tuple(sorted(r.bit_count() for r in hz)))))
    return (tuple(sorted((sx[1], sz[1]))),
            min(joint, swapped), row_profiles)


@lru_cache(maxsize=1)
def _design_bank(G):
    """Optimize components separately, then test their actual CSS combinations."""
    order_pools = {
        order: [g for g in G.elements if _element_order(G, g) == order]
        for order in (3, 4, 6, 12)
    }
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    base_A = [[[(0, 0), (1, 0), (0, 1)]
               for _ in range(ca)] for _ in range(ra)]
    base_B = [[[G.e] for _ in range(cb)] for _ in range(rb)]
    fingerprints = set()
    bank = []

    for slot in range(N_SLOTS):
        recipe = slot % 20
        geometry, mode = divmod(recipe, 5)
        triple_model = _TRIPLE_MODELS[mode]
        double_order = (12, 6)[mode % 2]
        sa, sb = _support_recipe(geometry)
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        rng = random.Random(35024025 + 100003 * slot)
        winner = None

        def component_key(item):
            return (item["floor"], -item["hits"], -item["cycles"])

        for batch in range(2):
            apool, bpool = [], []
            # Dummy partners only expose the verified component action.
            # No separately implemented group lift is used.
            for side, draws, pool in (("A", 64, apool), ("B", 40, bpool)):
                for attempt in range(draws):
                    if side == "A":
                        grid = [[_draw_triple(rng, G, kind, order_pools)
                                 for kind in triple_model]]
                        hx, hz, _ = lifted_matrices(G, grid, base_B)
                        columns = _component_columns(hx, hz, G.q)[0]
                    else:
                        _, grid = _draw_grids(
                            rng, G, sb, triple_model,
                            double_order, order_pools)
                        hx, hz, _ = lifted_matrices(G, base_A, grid)
                        columns = _component_columns(hx, hz, G.q)[1]

                    floor, hits = _component_six_screen(columns)
                    item = {
                        "grid": grid, "columns": columns,
                        "floor": floor, "hits": hits,
                        "cycles": _component_cycles(columns),
                    }
                    pool.append(item)
                    pool.sort(key=component_key, reverse=True)
                    del pool[6:]

            # A found short witness takes precedence over a high sampled
            # reading. Exact short screens and sampled readings are distinct.
            for item in apool + bpool:
                reading = _kernel_reading(rng, item["columns"], 24)
                item["d"] = (min(reading, item["floor"])
                             if item["floor"] < 8 else reading)

            combinations = []
            for a in apool:
                for b in bpool:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (325 * n <= 1000 * k < 375 * n):
                        continue
                    d = min(a["d"], b["d"])
                    priority = (
                        min(a["floor"], b["floor"]),
                        k * d * d / n, d,
                        -(a["hits"] + b["hits"]),
                        -(a["cycles"] + b["cycles"]),
                    )
                    combinations.append(
                        (priority, A, B, hx, hz, n, k, a, b))
            combinations.sort(key=lambda item: item[0], reverse=True)

            # Inspect several pairings, not just the independently best pair:
            # rank deficits and mixed-sector short vectors can depend on both.
            for _, A, B, hx, hz, n, k, a, b in combinations[:16]:
                sx = _small_kernel_stats(hx, n)
                if sx is None:
                    continue
                sz = _small_kernel_stats(hz, n)
                if sz is None:
                    continue
                fingerprint = _incidence_fingerprint(hx, hz, sx, sz)
                if fingerprint in fingerprints:
                    continue
                light = sx[0] + sz[0]
                d = min(a["d"], b["d"])
                objective = (
                    light == 0,
                    min(a["floor"], b["floor"]),
                    k * d * d / n,
                    -light,
                    -(a["hits"] + b["hits"]),
                    -(a["cycles"] + b["cycles"]),
                )
                if winner is None or objective > winner[0]:
                    winner = (objective, A, B, hx, hz, fingerprint)
            if winner is not None and winner[0][0] and winner[0][1] >= 8:
                break

        if winner is None:
            bank.append(None)
            continue
        _, A, B, hx, hz, fingerprint = winner
        if not css_ok(hx, hz):
            raise ValueError("verified lift failed CSS check")
        if max(row.bit_count() for row in hx + hz) > 12:
            raise ValueError("selected matrix exceeded the row cap")
        fingerprints.add(fingerprint)
        bank.append((A, B))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct legal designs")
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
# REFUTED: Balanced degree-two/three B columns with trinomial A eliminate weight-six sector-1 logicals on Dic6 (1,3)x(2,7); all five measured parent codes had d30s=6 with both types confined to sector 1.
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
