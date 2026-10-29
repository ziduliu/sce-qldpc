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
CONCEPT = ("For metacyclic C13:C1(s=1), order 13, shapes (2,6)x(2,7), two-row components with internal degree-two connections and balanced short-boundary exposure should improve worse-basis circuit fitness above the parent's 4.0144.")
GROUP = ("L2 necessary order adjustment within the same metacyclic construction rule: use (m,r,s)=(13,1,1). Shape factor 46 excludes the parent's q=21 and permits at most q=13, giving n=598. The existing cyclic q13 (2,6)x(2,7) bed scored about 3.8, while q13 (2,5)x(3,8) reached 5.1; these motivate a fresh filling experiment, not a presumed group advantage.")
SHAPES = ("Target cell 0.450 with (2,6)x(2,7): n=598, kb=260 and guaranteed rate 20/46=0.43478. Components have 26 checks on 78 and 91 variables, respectively; both sides now have two check-row orbits. Require verified k=260. Parent weight-six witnesses spanned sector-one Acol/Bcol blocks, and diffuse 1.0-1.2x failure enrichment supplied no consistent Brow hotspot, motivating global structural repair.")
FILLING = ("Twenty contracts cross four A count architectures with five B architectures; several replace degree-two cross-row matchings by internal order-13 connections. A has 16-18 terms and B 17-18; HX and HZ weights remain <=12. Fresh helper-derived component searches prioritize short-kernel exclusion, capped sampled protection and small-boundary exposure. Complete lifts verify exact k, CSS and both caps, then balance degree-weighted boundary exposure. Six-round CNOT budgets are 21840-23868 versus the parent's 19278: larger total circuits are an explicit tradeoff, not a gate-count improvement. Target the parent's failed-shot caps, collective flips across all 210 logicals and late X decoder settlement; structural screens are not calibrated pL.")
FAMILY_NAME = "r045_m13r1s1_26x27_internalcycle_threshold_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 13")
    # Same metacyclic rule, with parameters required by shape factor 46.
    # x^13=y^1=1, y*x*y^-1=x; elements remain encoded as x^a*y^b.
    m, r, s = 13, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(s, b, m) * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-pow(s, bi, m) * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return a + m * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Four A architectures: row totals (9,9), (9,9), (9,8), (8,8).
# Columns have degree two or three; every row and column remains live.
_A_PROFILES = (
    ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
    ((3,0), (0,3), (2,1), (2,1), (1,2), (1,2)),
    ((3,0), (1,2), (1,2), (1,2), (2,1), (1,1)),
    ((2,1), (2,1), (1,2), (1,2), (2,0), (0,2)),
)

# B architectures distinguish cross-row and internal degree-two edges.
# The first four have row totals (9,9); the last has (9,8).
_B_PROFILES = (
    ((2,1), (2,1), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((2,1), (2,1), (1,2), (1,2), (2,0), (0,2), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (2,0), (0,2), (1,1)),
    ((3,0), (0,3), (2,1), (1,2), (2,0), (0,2), (1,1)),
    ((2,1), (1,2), (2,1), (1,1), (2,0), (0,2), (1,1)),
)


def _component_rows(G, cells, side):
    """Use verified lifts, with a zero opposite side only as a probe."""
    if side == "A":
        return lifted_matrices(G, cells, [[[]]])[0]
    return lifted_matrices(G, [[[]]], cells)[1]


def _columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _kernel4(rows, width):
    """Exact weight-three/four counts after excluding weights one/two."""
    cols = _columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = {}
    three = four = boundary = 0
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            three += int(value in singles)
            old = pairs.get(value, 0)
            four += old
            pairs[value] = old + 1
            w = value.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    assert three % 3 == 0 and four % 3 == 0
    return (three // 3, four // 3, boundary), cols


def _kernel56(cols):
    """Exact counts, called only when component weights <=4 are absent.

    Pair/triple matches count each weight-five support ten times;
    collisions of distinct triples count each weight-six support ten
    times. The <=4 exclusion makes these matching supports disjoint.
    """
    pairs = {}
    for i, x in enumerate(cols):
        for y in cols[:i]:
            value = x ^ y
            pairs[value] = pairs.get(value, 0) + 1
    triples = {}
    five = six = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                value = pair ^ cols[h]
                five += pairs.get(value, 0)
                old = triples.get(value, 0)
                six += old
                triples[value] = old + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _four_cycles(rows):
    return sum(v * (v - 1) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Exact cycles using three checks and three distinct variables."""
    size = len(rows)
    overlaps = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i):
            overlaps[i][j] = overlaps[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(size):
        for j in range(i):
            x = overlaps[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlaps[j][h], overlaps[i][h]
                if y and z:
                    t = (common & rows[h]).bit_count()
                    total += x * y * z - t * (x + y + z) + 2 * t
    return total


def _boundary3(cols):
    """Count-weighted small syndrome boundaries of variable triples.

    This is an exact neighborhood statistic of a verified component,
    not a trapping-set certificate or a circuit failure estimate.
    """
    exposure = 0
    histogram = [0] * 4
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                weight = (pair ^ cols[h]).bit_count()
                if weight <= 3:
                    histogram[weight] += 1
                    exposure += (4 - weight) ** 2
    return exposure, tuple(histogram)


def _component_score(short, cols, rows, four):
    # Short words take priority. A failed <=4 exclusion is not presented
    # as an exact weight-five/six spectrum.
    if short[0] or short[1]:
        five = six = 1 << 60
    else:
        five, six = _kernel56(cols)
    return (short[0], short[1], five, six,
            four, _six_cycles(rows), short[2])


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def NEW_CTX(rng, G, idx):
    """Twenty two-row support contracts; all optimization state is local."""
    import math
    from itertools import combinations

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    assert q == 13 and (ra, ca, rb, cb) == (2, 6, 2, 7)
    design, batch = idx % 20, idx // 20
    ai, bi = divmod(design, 5)
    counts = {
        "A": [[column[i] for column in _A_PROFILES[ai]]
              for i in range(ra)],
        "B": [[column[i] for column in _B_PROFILES[bi]]
              for i in range(rb)],
    }
    sa, sb = counts["A"], counts["B"]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    # Both components admit full augmentation rank; actual ranks below
    # are still mandatory, rather than inferred from this parity test.
    for grid in (sa, sb):
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in grid
        ]
        assert gf2_rank(augmentation) == 2

    # Compose only singleton actions supplied by the verified helper.
    # This is linearity of those blocks, not a separate lift convention.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        ar = lifted_matrices(G, [[[g]]], [[[]]])[0]
        br = lifted_matrices(G, [[[]]], [[[g]]])[1]
        actions["A"][G._key(g)] = tuple(row & mask for row in ar)
        actions["B"][G._key(g)] = tuple(row & mask for row in br)

    def component(side, grid):
        result = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, word in enumerate(block):
                        out[t] ^= word << (j * q)
            result.extend(out)
        return result

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def fingerprint(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    # Trinomials have six distinct directed differences. Further
    # harness batches change local affine triangle briefs, not saved
    # matrices or their group-automorphism images.
    sidon = []
    nonzero = list(range(1, q))
    for u, v in combinations(nonzero, 2):
        differences = {
            (a - b) % q for a in (0, u, v)
            for b in (0, u, v) if a != b
        }
        if len(differences) == 6:
            sidon.append((0, u, v))

    def draw_cell(side, i, j, count):
        if count == 0:
            return []
        origin = rng.randrange(q)
        if count == 1:
            values = (0,)
        elif count == 2:
            values = (0, rng.randrange(1, q))
        else:
            model = (batch + i + j + (side == "B")) % 3
            if batch == 0 or model == 0:
                values = rng.choice(sidon)
            else:
                step = rng.randrange(1, q)
                ratio = 3 if model == 1 else 4
                values = (0, step, (ratio * step) % q)
        cell = [((origin + value) % q, 0) for value in values]
        assert len(cell) == count
        assert len({G._key(g) for g in cell}) == count
        return cell

    def sampled_words(rows, width, side):
        """Verified kernel witnesses; sampled minimum is an upper bound.

        Stop rewarding the minimum at eight. Counts and block loads
        describe this sample only, not a complete logical spectrum.
        """
        srng = random.Random(91337 + 1009 * idx + (side == "B"))
        cols = _columns(rows, width)
        order = list(range(width))
        minimum = width + 1
        small = set()
        for _ in range(32):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
            for i, word in enumerate(basis):
                candidates = [word]
                candidates.extend(word ^ other for other in basis[:i])
                for value in candidates:
                    if not value:
                        continue
                    weight = value.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 8 and len(small) < 8192:
                        small.add(value)
            if minimum <= 4:
                break
        loads = [0] * (width // q)
        for word in small:
            for j in range(len(loads)):
                loads[j] += int(bool((word >> (j * q)) & mask))
        return minimum, len(small), max(loads, default=0)

    def portfolio(side):
        support = counts[side]
        width = len(support[0]) * q
        sites = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
        pool, seen = [], set()
        serial = 0

        def state(grid):
            rows = component(side, grid)
            four, six = _four_cycles(rows), _six_cycles(rows)
            return rows, four, six, (60 * four + six / 5) / q

        def remember(grid, rows, four, six):
            nonlocal serial
            signature = fingerprint(grid)
            if signature in seen:
                return
            seen.add(signature)
            if gf2_rank(rows) != 2 * q:
                return
            result = _kernel4(rows, width)
            if result is None:
                return
            short, cols = result
            coarse = (short[0], short[1], four, six, short[2])
            if len(pool) == 12 and coarse > pool[-1][0]:
                return
            assert rows == _component_rows(G, grid, side)
            serial += 1
            pool.append((coarse, serial, clone(grid), rows,
                         cols, short, four))
            pool.sort(key=lambda item: item[:2])
            del pool[12:]

        for restart in range(4):
            grid = [[draw_cell(side, i, j, count)
                     for j, count in enumerate(row)]
                    for i, row in enumerate(support)]
            rows, four, six, energy = state(grid)
            remember(grid, rows, four, six)
            for step in range(300):
                i, j = rng.choice(sites)
                previous = grid[i][j]
                grid[i][j] = draw_cell(side, i, j, support[i][j])
                trial_rows, c4, c6, proposed = state(grid)
                delta = proposed - energy
                temperature = 9 * (0.984 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    rows, four, six, energy = trial_rows, c4, c6, proposed
                else:
                    grid[i][j] = previous
                if step >= 60 and step % 15 == 0:
                    remember(grid, rows, four, six)
            remember(grid, rows, four, six)

        if not pool:
            # Fresh recovery stays inside the exact same support brief.
            for _ in range(512):
                grid = [[draw_cell(side, i, j, count)
                         for j, count in enumerate(row)]
                        for i, row in enumerate(support)]
                rows, four, six, _ = state(grid)
                remember(grid, rows, four, six)
                if len(pool) >= 6:
                    break
        if not pool:
            raise ValueError("no full-rank distinct-column component")

        refined = []
        for coarse, serial, grid, rows, cols, short, four in pool[:8]:
            raw = _component_score(short, cols, rows, four)
            boundary, histogram = _boundary3(cols)
            minimum, low8, peak = sampled_words(rows, width, side)
            key = (raw[0], raw[1], int(bool(raw[2])),
                   int(bool(raw[3])), max(0, 8 - minimum),
                   low8, peak, boundary / width,
                   raw[2], raw[3], raw[4], raw[5])
            refined.append((key, serial, grid, raw, histogram,
                            (minimum, low8, peak), boundary))
        refined.sort(key=lambda item: item[:2])
        return refined[:3]

    def exposure(checks, other, n):
        """Degree-weighted small boundaries, not ASC-certified hooks."""
        cols, mate = _columns(checks, n), _columns(other, n)
        degree = [a.bit_count() + b.bit_count()
                  for a, b in zip(cols, mate)]
        total = 0
        for i, x in enumerate(cols):
            for j in range(i):
                boundary = (x ^ cols[j]).bit_count()
                if boundary <= 3:
                    shared = (mate[i] & mate[j]).bit_count()
                    total += (degree[i] * degree[j]
                              * (4 - boundary) ** 2 * (1 + shared))
        return total / len(checks)

    aa, bb = portfolio("A"), portfolio("B")
    best = None
    for akey, an, A, ac, ah, ast, ab in aa:
        for bkey, bn, B, bc, bh, bst, bboundary in bb:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 598 or k != 260:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            ex, ez = exposure(hx, hz, n), exposure(hz, hx, n)
            fx, fz = _four_cycles(hx), _four_cycles(hz)
            # Component words can replicate across opposite columns.
            # These scaled counts are risk surrogates, not logical counts.
            a5, b5 = cb * ac[2], ca * bc[2]
            a6, b6 = cb * ac[3], ca * bc[3]
            key = (
                max(ac[0], bc[0]), ac[0] + bc[0],
                max(ac[1], bc[1]), ac[1] + bc[1],
                int(bool(ac[2] or bc[2])),
                int(bool(ac[3] or bc[3])),
                max(max(0, 8 - ast[0]), max(0, 8 - bst[0])),
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
                max(ex, ez), ex + ez,
                max(a5, b5), a5 + b5,
                max(a6, b6), a6 + b6,
                max(ab / (ca*q), bboundary / (cb*q)),
                ab / (ca*q) + bboundary / (cb*q),
                max(cb * ast[1], ca * bst[1]),
                cb * ast[1] + ca * bst[1],
                max(fx / len(hx), fz / len(hz)), fx + fz,
                an, bn,
            )
            if best is None or key < best[0]:
                best = (key, A, B, hx, hz, n, k,
                        ac, bc, ah, bh, ast, bst)

    if best is None:
        raise ValueError("no exact-rate two-row CSS finalist")
    key, A, B, hx, hz, n, k, ac, bc, ah, bh, ast, bst = best
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == k == 260
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    cnots = 6 * sum(row.bit_count() for row in hx + hz)
    expected = 6*q*((cb+rb)*sum(map(sum, sa))
                    + (ca+ra)*sum(map(sum, sb)))
    assert cnots == expected
    assert 100 <= n < 600 and 0.425 <= k / n < 0.475
    return {"A": A, "B": B, "design": design, "batch": batch,
            "count_contract": (ai, bi), "k": k, "cnots": cnots,
            "component_profiles": (ac, bc),
            "boundary_histograms": (ah, bh),
            "sampled_witness_profiles": (ast, bst),
            "ambiguity_score": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 still failed 110/1250 Z and 121/1250 X shots, with ubiquitous sector-1 weight-six witnesses and diffuse failure enrichment.
# REFUTED: Quotient-controlled trinomials and component spectrum/boundary minimization suffice for rare collective failures on Frob21 (1,3)x(2,7): fitness was 4.0144; all five measured codes reached failed-shot caps, retained abundant sector-one weight-six witnesses, and failures collectively touched all 210 logicals.
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
