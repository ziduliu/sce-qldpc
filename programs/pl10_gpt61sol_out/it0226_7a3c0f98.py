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
CONCEPT = ("On dicyclic Dic3, order 12, shapes (2,6)x(2,7), independently connected two-row components should reduce collective circuit failures relative to the single-row-A Frobenius21 parent at guaranteed rate 10/23.")
GROUP = ("Jump to Dic3: x^6=1, y^2=x^3, yxy^-1=x^-1. Its unique central involution and noncentral orders three, four and six distinguish it from the parent's odd-order Frobenius group. BEDS contains no order-12 centre-two Dic3 bed. Successful lower-rate dicyclic beds motivate this experiment without establishing a high-rate group advantage.")
SHAPES = ("Target 0.450 with (2,6)x(2,7): q=12 gives n=552, kb=240 and guaranteed rate 20/46=10/23. The window permits q<=13. A has 24 checks on 72 variables; B has 24 checks on 84 variables. Require verified k=240. Two A rows replace the parent's single-row trinomial mechanism. Parent failures affected all 210 logicals, with diffuse 1.0-1.2x Acol/Bcol enrichment and sector-one weight-six witnesses, not a consistent Brow hotspot.")
FILLING = ("Twenty support contracts cross four degree-three A architectures with five B architectures. A and B each have nine terms per row; B has four degree-three and three degree-two columns. Independent connection searches use verified component lifts, full ranks, exact short-kernel counts, cycles and triple boundaries. Complete verified matrices screen exact k, both syndrome ambiguity profiles and opposite-check-weighted small pair boundaries. Pure B doubletons use order-six ratios, never the central involution. HX weights are eleven or twelve and HZ weights twelve; six-round CNOTs are 22032 versus the parent's 19278. Target the parent's 119/1500 X failures in slot 1 and 105/750 Z failures in slot 6, collective logical flips and late decoder settlement. Structural scores are not calibrated pL estimates.")
FAMILY_NAME = "r045_Dic3_q12_26x27_twocheck_connections_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic3 requires order 12")
    # x^6=1, y^2=x^3, y*x*y^-1=x^-1.
    # Canonical elements are x^a*y^b with b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(6)]

    def red(g):
        return (int(g[0]) % 6, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 3*b*d) % 6,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 6, 0) if b == 0 else ((a + 3) % 6, 1)

    def idx(g):
        a, b = red(g)
        return a + 6*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Column term-count pairs. Every architecture has row totals (9,9),
# four degree-three columns and three degree-two columns.
_B_PROFILES = (
    ((2,1), (2,1), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (0,3), (2,1), (1,2), (1,1), (1,1), (1,1)),
    ((2,1), (2,1), (1,2), (1,2), (2,0), (0,2), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (2,0), (0,2), (1,1)),
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
    """Twenty two-row support contracts; optimize genuine connections."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 12
    design, batch = idx % 20, idx // 20
    policy = design // 5

    # Distinct zero-cell architectures, not column-permutation copies.
    # Every A column has degree three and both A rows total nine.
    a_profiles = (
        ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (2,1), (1,2), (1,2)),
        ((3,0), (1,2), (1,2), (1,2), (1,2), (2,1)),
        ((3,0), (3,0), (0,3), (0,3), (2,1), (1,2)),
    )
    profiles = {"A": a_profiles[policy],
                "B": _B_PROFILES[design % 5]}
    counts = {
        side: [[column[i] for column in profile] for i in range(2)]
        for side, profile in profiles.items()
    }
    sa, sb = counts["A"], counts["B"]
    assert tuple(map(sum, sa)) == (9, 9)
    assert tuple(map(sum, sb)) == (9, 9)
    assert all(sum(column) == 3 for column in profiles["A"])
    assert sorted(map(sum, profiles["B"])) == [2, 2, 2, 3, 3, 3, 3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    orders = {}
    for g in G.elements:
        value = G.e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if G._key(value) == G._key(G.e):
                orders[G.idx(g)] = length
                break
        else:
            raise ValueError("validated element has no finite order")
    palettes = {
        length: [g for g in G.elements if orders[G.idx(g)] == length]
        for length in (3, 4, 6)
    }
    nonidentity = [g for g in G.elements if g != G.e]
    triangles = []
    for u, v in combinations(nonidentity, 2):
        if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
            continue
        cell = [G.e, u, v]
        # No central-involution difference inside a trinomial.
        if min(orders[G.idx(G.mul(G.inv(a), b))]
               for a, b in combinations(cell, 2)) < 3:
            continue
        triangles.append(cell)
    assert triangles and all(palettes.values())

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def fingerprint(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def draw(side):
        profile = profiles[side]
        grid = [[[] for _ in profile] for _ in range(2)]
        for j, column in enumerate(profile):
            for i, count in enumerate(column):
                if count == 0:
                    continue
                base = G.pick(rng)
                if count == 1:
                    relative = [G.e]
                elif count == 2:
                    if side == "B" and sum(column) == 2:
                        length = 6
                    else:
                        # Larger harness batches alter relative-order
                        # prescriptions, not just saved-code gauges.
                        length = (6, 4, 3)[
                            (policy + batch + i + j) % 3]
                    relative = [G.e, rng.choice(palettes[length])]
                else:
                    relative = rng.choice(triangles)
                cell = [G.mul(base, g) for g in relative]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                grid[i][j] = cell
        return grid

    def portfolio(side):
        width = len(profiles[side]) * q
        pool, seen = [], set()

        def assess(grid):
            # The zero-partner probe uses the scaffold's exact lift.
            rows = _component_rows(G, grid, side)
            if gf2_rank(rows) != 2 * q:
                return None
            result = _kernel4(rows, width)
            if result is None:
                return None
            short, cols = result
            four = _four_cycles(rows)
            energy = (
                400 * short[0] + 60 * short[1]
                + 2 * four + short[2] / 8
            ) / q
            return rows, cols, short, four, energy

        def offer(grid, state):
            if state is None:
                return
            identity = fingerprint(grid)
            if identity in seen:
                return
            seen.add(identity)
            rows, cols, short, four, energy = state
            coarse = (short[0], short[1], four, short[2])
            pool.append((coarse, identity, clone(grid), state))
            pool.sort(key=lambda entry: entry[:2])
            del pool[10:]

        for _ in range(64):
            grid = draw(side)
            offer(grid, assess(grid))

        # Mutate whole component columns. Row budgets and column
        # degrees therefore survive every proposal without repair.
        for restart in range(3):
            for _ in range(96):
                grid = draw(side)
                state = assess(grid)
                if state is not None:
                    break
            else:
                continue
            offer(grid, state)
            for step in range(320):
                proposal = clone(grid)
                fresh = draw(side)
                j = rng.randrange(len(profiles[side]))
                for i in range(2):
                    proposal[i][j] = fresh[i][j]
                candidate = assess(proposal)
                if candidate is None:
                    continue
                change = candidate[-1] - state[-1]
                temperature = 5.0 * (0.983 ** step) + 0.04
                if change <= 0 or rng.random() < math.exp(
                        -min(700.0, change / temperature)):
                    grid, state = proposal, candidate
                    if step % 8 == 0:
                        offer(grid, state)
            offer(grid, state)

        if not pool:
            raise ValueError("no full-rank Dic3 two-row component")

        refined = {}
        for coarse, identity, grid, state in pool:
            rows, cols, short, four, energy = state
            raw = _component_score(short, cols, rows, four)
            boundary, histogram = _boundary3(cols)
            key = raw[:4] + (
                boundary, four, raw[5], short[2])
            # Distinct fingerprints preserve several neighborhoods.
            # Equal statistics are not a proof of Tanner equivalence;
            # evaluator deduplication remains authoritative.
            old = refined.get(key)
            entry = (key, identity, grid, raw, boundary, histogram)
            if old is None or identity < old[1]:
                refined[key] = entry
        return sorted(refined.values(), key=lambda entry: entry[:2])[:4]

    def pair_exposure(checks, other, n):
        """Small-boundary fault-opportunity surrogate, not ASC hooks."""
        cols = _columns(checks, n)
        mate = _columns(other, n)
        degree = [a.bit_count() + b.bit_count()
                  for a, b in zip(cols, mate)]
        burden = 0
        for i, x in enumerate(cols):
            for j in range(i):
                boundary = (x ^ cols[j]).bit_count()
                if boundary <= 5:
                    shared = (mate[i] & mate[j]).bit_count()
                    burden += (degree[i] * degree[j]
                               * (6 - boundary) ** 2 * (1 + shared))
        return burden / len(checks)

    aa, bb = portfolio("A"), portfolio("B")
    options = []
    for ak, aid, A, ar, ab, ah in aa:
        for bk, bid, B, br, bbnd, bh in bb:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 552 or k != 240:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue

            # Replication-weighted component counts are a shortlist
            # surrogate, not an exact census of quantum logicals.
            protection = (
                max(ar[0], br[0]), ar[0] + br[0],
                max(ar[1], br[1]), ar[1] + br[1],
                max(7*ar[2], 6*br[2]), 7*ar[2] + 6*br[2],
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
            )
            spectrum = (
                max(7*ar[3], 6*br[3]), 7*ar[3] + 6*br[3],
                max(ab, bbnd), ab + bbnd,
                max(ar[4], br[4]), ar[4] + br[4],
                ar[5] + br[5],
            )
            options.append((protection, spectrum, (aid, bid),
                            A, B, hx, hz, ar, br, ah, bh))
    if not options:
        raise ValueError("no exact-rate Dic3 complete candidate")
    options.sort(key=lambda entry: entry[:3])

    best = None
    for item in options[:6]:
        protection, spectrum, identity, A, B, hx, hz, ar, br, ah, bh = item
        ex = pair_exposure(hx, hz, 552)
        ez = pair_exposure(hz, hx, 552)
        fx, fz = _four_cycles(hx), _four_cycles(hz)
        # Compare circuit-opportunity exposure before further rewarding
        # multiplicity reductions once short-word protection is matched.
        key = protection + (
            max(ex, ez), ex + ez,
        ) + spectrum + (
            max(fx / len(hx), fz / len(hz)), fx + fz,
        )
        if best is None or (key, identity) < best[:2]:
            best = (key, identity, A, B, hx, hz,
                    ar, br, ah, bh, ex, ez)

    key, _, A, B, hx, hz, ar, br, ah, bh, ex, ez = best
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 240
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 22032
    assert 0.425 <= 240 / 552 < 0.475
    return {"A": A, "B": B, "design": design, "batch": batch,
            "k": 240, "component_profiles": (ar, br),
            "boundary_histograms": (ah, bh),
            "pair_exposure": (ex, ez), "ambiguity_score": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 still failed 110/1250 Z and 121/1250 X shots, with ubiquitous sector-1 weight-six witnesses and diffuse failure enrichment.
# REFUTED: Quotient-controlled trinomials plus short-kernel and boundary minimization suffice for rare failures on Frobenius21 (1,3)x(2,7): fitness was 4.0144, slot 1 failed 119/1500 X shots, slot 6 failed 105/750 Z shots, and measured codes retained extensive sector-one weight-six witnesses and collective flips across all 210 logicals.
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
