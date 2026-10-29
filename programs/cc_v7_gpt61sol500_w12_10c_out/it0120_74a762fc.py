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
CONCEPT = ("SL(2,F3), order 24, shapes (1,3)x(2,7): full-group noncommuting A triangles and balanced zero/triple B designs should raise the five measured distance readings beyond the parent's weight-six plateau in cell 0.450.")
GROUP = ("Jump from Dic6 to the determinant-one matrix group SL(2,F3). SL(2,3) has maximum element order six, unlike Dic6's order-twelve elements, so this is not a relabeling. Order 24 preserves n=552. Related GL(2,3) fillings reached distance-twelve readings at (1,2)x(2,5), motivating this experiment without establishing transfer.")
SHAPES = ("Keep A=(1,3), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.434783. Component variable/check counts are 72/24 and 168/48. All five parental measurements found weight-six minima entirely in sector-1 Acol/Bcol blocks, broadly across the components. Verified ranks retain 0.425<=k/n<0.475.")
FILLING = ("Twenty cards cross four balanced B geometries with five generating-triangle order policies. A row sum is nine; B has four degree-three columns and three singleton/singleton columns, with both row sums nine. Thus both CSS weight maxima are twelve. A relative generators must generate all 24 elements; involution-separated terms are excluded. Verified component screens exclude weights through four; finalists receive exact exclusion through five and weight-six word counts, deeper witness searches, and full commuting-vector tests through four. Different uncoloured combined-Tanner invariants prevent emitted orbit copies. Finite component screens and sampled witnesses do not certify quantum distance.")
FAMILY_NAME = "r045_SL2F3_13x27_generating_triangles_exact6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # SL(2,F3), row-major matrix entries, determinant exactly one.
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


# Desired sorted orders of the three internal triangle differences.
# If a signature is absent, use the closest AVAILABLE generating
# signature, explicitly rather than inventing an unavailable element.
_TRIPLE_MODELS = (
    (3, 3, 4),
    (3, 4, 6),
    (3, 6, 6),
    (4, 6, 6),
    (6, 6, 6),
)


def _support_recipe(geometry):
    # Four degree-three columns, followed by three degree-two columns.
    # Top counts sum to six; complementary bottom counts also sum to six.
    tops = ((0, 0, 3, 3), (0, 1, 2, 3),
            (0, 2, 2, 2), (1, 1, 2, 2))[geometry]
    sa = [[3, 3, 3]]
    sb = [list(tops) + [1, 1, 1],
          [3 - t for t in tops] + [1, 1, 1]]
    return sa, sb


def _sidon_relatives(rng, G, count):
    """Pack actual ordered differences, not just distinct group labels."""
    for _ in range(64):
        selected, differences = [G.e], set()
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
            if differences.intersection(added):
                continue
            selected.append(g)
            differences.update(added)
            if len(selected) == count:
                rng.shuffle(selected)
                return selected
    return rng.sample(G.elements, count)


def _generating_triangles(G, orders):
    """Enumerate intrinsic triangles; neither translations nor orbits
    are themselves counted as separate emitted designs."""
    available = {}
    for u in G.elements:
        if orders[u] <= 2:
            continue
        for v in G.elements:
            if v == u or orders[v] <= 2:
                continue
            uv = G.mul(G.inv(u), v)
            if orders[uv] <= 2 or G.mul(u, v) == G.mul(v, u):
                continue
            subgroup, todo = {G.e}, [G.e]
            while todo:
                g = todo.pop()
                for generator in (u, v):
                    h = G.mul(g, generator)
                    if h not in subgroup:
                        subgroup.add(h)
                        todo.append(h)
            if len(subgroup) != G.q:
                continue
            signature = tuple(sorted((orders[u], orders[v], orders[uv])))
            available.setdefault(signature, []).append((G.e, u, v))
    if not available:
        raise ValueError("no full-group generating triangles")

    templates = []
    for wanted in _TRIPLE_MODELS:
        cost = lambda sig: sum(abs(a-b) for a, b in zip(sig, wanted))
        best = min(map(cost, available))
        pool = [cell for sig in sorted(available) if cost(sig) == best
                for cell in available[sig]]
        templates.append(tuple(pool))
    return tuple(templates)


def _draw_triple(rng, G, mode, templates):
    cell = rng.choice(templates[mode % len(templates)])
    gauge = G.pick(rng)
    return [G.mul(gauge, g) for g in cell]


def _draw_grids(rng, G, sb, triple_model, double_order,
                order_pools, templates):
    A = [[_draw_triple(rng, G, mode, templates)
          for mode in triple_model]]
    B = [[None] * 7 for _ in range(2)]
    mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
    for j, relative in zip(mono, _sidon_relatives(rng, G, len(mono))):
        top = G.pick(rng)
        bottom = G.mul(G.inv(relative), top)
        B[0][j], B[1][j] = [top], [bottom]

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
                g = G.pick(rng)
                delta = rng.choice(order_pools[double_order])
                B[i][j] = [g, G.mul(g, delta)]
            else:
                mode = triple_model[(i + j) % 3] + i + j
                B[i][j] = _draw_triple(rng, G, mode, templates)
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


def _component_floor4(columns):
    """Exact classical-kernel minimum through four; five is a sentinel."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs, has_four = set(), False
    for i, ci in enumerate(columns):
        for j in range(i):
            syndrome = ci ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                has_four = True
            pairs.add(syndrome)
    return 4 if has_four else 5


def _component_six_stats(columns):
    """Exact exclusion through five and exact weight-six word count.

    After weights <=4 are excluded, equal-syndrome distinct triples
    are disjoint. Every weight-six word contributes ten unordered
    triple partitions. A returned seven means only exclusion through six.
    """
    floor = _component_floor4(columns)
    if floor != 5:
        return floor, 0
    low = {0}
    low.update(columns)
    for i, ci in enumerate(columns):
        for j in range(i):
            low.add(ci ^ columns[j])

    counts, collisions = {}, 0
    for i, ci in enumerate(columns):
        for j in range(i):
            pair = ci ^ columns[j]
            for t in range(j):
                syndrome = pair ^ columns[t]
                if syndrome in low:
                    # With <=4 already excluded, this reveals weight five.
                    return 5, 0
                previous = counts.get(syndrome, 0)
                collisions += previous
                counts[syndrome] = previous + 1
    assert collisions % 10 == 0
    return (6, collisions // 10) if collisions else (7, 0)


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
        for start in short[:4]:
            word = start
            for descent in range(3):
                improved = word
                weight = word.bit_count()
                for other in words:
                    trial = word ^ other
                    if trial and trial.bit_count() < weight:
                        improved, weight = trial, trial.bit_count()
                if improved == word:
                    break
                word = improved
                best = min(best, weight)
        if best <= 4:
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
    """Uncoloured combined-Tanner invariant.

    Ignore X/Z colours entirely. Different fingerprints therefore rule
    out even an uncoloured check/qubit relabeling; equal fingerprints
    conservatively discard some genuinely different codes.
    """
    from collections import Counter

    rows = hx + hz
    n = len(sx[2])
    columns = _syndrome_columns(rows, n)
    profiles = []
    for i, row in enumerate(rows):
        overlaps = Counter(
            (row & other).bit_count()
            for j, other in enumerate(rows) if i != j)
        profiles.append((row.bit_count(), tuple(sorted(overlaps.items()))))
    qubit_profiles = []
    weights = [row.bit_count() for row in rows]
    for column in columns:
        adjacent_weights = []
        v = column
        while v:
            bit = v & -v
            adjacent_weights.append(weights[bit.bit_length() - 1])
            v ^= bit
        qubit_profiles.append(tuple(sorted(adjacent_weights)))
    return (tuple(sorted(Counter(profiles).items())),
            tuple(sorted(Counter(qubit_profiles).items())))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Optimize twenty structural cards, not twenty relabeling draws."""
    orders = {g: _element_order(G, g) for g in G.elements}
    order_pools = {
        order: [g for g in G.elements if orders[g] == order]
        for order in (3, 4, 6)
    }
    assert all(order_pools.values())
    templates = _generating_triangles(G, orders)
    fingerprints, bank = set(), []

    for slot in range(N_SLOTS):
        recipe, extension = slot % 20, slot // 20
        geometry, mode = divmod(recipe, 5)
        # Additional harness slots alter intrinsic A signature occupancy:
        # three palettes, two palettes, then one repeated palette.
        occupancy = extension % 3
        if occupancy == 0:
            triple_model = (mode, (mode + 1) % 5, (mode + 2) % 5)
        elif occupancy == 1:
            triple_model = (mode, mode, (mode + 2) % 5)
        else:
            triple_model = (mode, mode, mode)
        double_order = (3, 4, 6)[(mode + extension) % 3]
        sa, sb = _support_recipe(geometry)
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        rng = random.Random(450243031 + 100003*slot)

        def merit(item):
            d = min(item["da"], item["db"], 12)
            return (d, min(item["da"], 16) + min(item["db"], 16),
                    -item["cycles"], item["k"])

        winner = None
        for batch in range(2):
            pool, signatures = [], set()
            for attempt in range(48):
                A, B = _draw_grids(
                    rng, G, sb, triple_model, double_order,
                    order_pools, templates)
                signature = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for grid in (A, B) for row in grid for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)

                hx, hz, n = lifted_matrices(G, A, B)
                ac, bc = _component_columns(hx, hz, G.q)
                if _component_floor4(ac) != 5:
                    continue
                if _component_floor4(bc) != 5:
                    continue
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (425*n <= 1000*k < 475*n):
                    continue
                item = {
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": _kernel_reading(rng, ac, 6),
                    "db": _kernel_reading(rng, bc, 6),
                    "cycles": _component_cycles(ac) + _component_cycles(bc),
                }
                pool.append(item)

            finalists = sorted(pool, key=merit, reverse=True)[:4]
            deep = []
            for item in finalists:
                item["da"] = min(
                    item["da"], _kernel_reading(rng, item["ac"], 24))
                item["db"] = min(
                    item["db"], _kernel_reading(rng, item["bc"], 24))
                fa, na = _component_six_stats(item["ac"])
                fb, nb = _component_six_stats(item["bc"])
                item["finite_floor"] = min(fa, fb)
                item["six_words"] = na + nb
                # A found short relation corrects a sampling miss.
                if fa < 7:
                    item["da"] = min(item["da"], fa)
                if fb < 7:
                    item["db"] = min(item["db"], fb)
                deep.append(item)

            deep.sort(
                key=lambda item: (
                    item["finite_floor"], merit(item),
                    -item["six_words"]), reverse=True)
            for item in deep:
                hx, hz, n = item["hx"], item["hz"], item["n"]
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
                d = min(item["da"], item["db"], 12)
                objective = (
                    light == 0,
                    item["finite_floor"],
                    item["k"] * d * d,
                    -light,
                    -item["six_words"],
                    -item["cycles"],
                )
                if winner is None or objective > winner[0]:
                    winner = (objective, item, fingerprint)
            # Do not stop at the parent's merely-four-clean criterion.
            if (winner is not None and winner[0][0]
                    and winner[0][1] == 7):
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
# REFUTED: Treating Dic6 involution avoidance and logical-four screening as removal of the sector-1 weight-six obstruction is unsupported: all five measured parent codes still had d30s=6, with minima entirely in sector 1.
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
