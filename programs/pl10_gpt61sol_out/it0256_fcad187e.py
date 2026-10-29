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
CONCEPT = ("Cyclic C19, order 19, shapes (1,4)x(2,7): an arithmetic-progression A trinomial and lower worst-edge quartet exposure in graphic B should improve worse-basis circuit fitness above 3.3458 at unchanged extraction cost.")
GROUP = ("Frozen under L1: retain C19, order 19, multiplication and encoding byte-for-byte. This exact bed scored 3.3458; its measured collective failures motivate a filling comparison, not a group-only reliability claim.")
SHAPES = ("Frozen (1,4)x(2,7): the size factor thirty permits q<=19, and q=19 gives n=570, kb=285 and guaranteed rate 0.500. Unequal components have 76 variables/19 checks and 133 variables/38 checks. Require verified k=288, rate 0.5053 and k-kb=3. Parent failures affected all 288 logicals across many Acol/Bcol blocks, with mostly 1.0-1.2x data/check enrichment and no persistent Acol/Brow repair site.")
FILLING = ("Twenty contracts cross five canonical AP-containing A geometries with four B architectures and five quartet-concentration penalties; further batches consume further canonical A geometries. A retains weights [2,2,3,3], but at least one trinomial has repeated internal differences, unlike the parent's disjoint Sidon catalogue. Verified short-kernel screening compares this local-cycle versus logical-ambiguity tradeoff. B retains two terms per column and row totals 7/7 or 8/6, adding a three-internal-edge-per-row architecture. Cached singleton actions come only from verified lifts. Deeper annealing minimizes exact graph triangles, quartets and worst-edge quartet load; full verified CSS selection prioritizes gate-weighted nontrivial quartic ambiguities before shared-check exposure. Both weight caps remain <=12 and extraction remains 18240 six-round CNOTs. Target the parent's 106-148 failed Z shots/250 and widespread logical flips; these structural costs are not calibrated pL estimates.")
FAMILY_NAME = "r050_C19_14x27_AP_quartet_concentration_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order 19")
    els = [(a,) for a in range(19)]

    def red(g):
        return (int(g[0]) % 19,)

    def mul(g, h):
        return ((g[0] + h[0]) % 19,)

    def inv(g):
        return ((-g[0]) % 19,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Five B architectures: column totals 3,3,3,3,2,2,2;
# both rows total nine. Zero cells are deliberate, never repairs.
_B_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,1),(1,1),(1,1)),
    ((3,0),(1,2),(1,2),(1,2),(1,1),(1,1),(1,1)),
    ((3,0),(0,3),(2,1),(1,2),(1,1),(1,1),(1,1)),
    ((2,1),(2,1),(1,2),(1,2),(2,0),(0,2),(1,1)),
    ((3,0),(1,2),(1,2),(1,2),(2,0),(0,2),(1,1)),
)


def _kernel56(rows, ncols):
    """Exact weight-five/six counts when kernel weights <=4 are absent.

    Pair/triple syndrome matches count each weight-five word ten times.
    Distinct triple collisions count each weight-six word ten times.
    Smaller-word exclusion makes these colliding supports disjoint.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            s = x ^ y
            pairs[s] = pairs.get(s, 0) + 1
    triples = {}
    five = six = 0
    for i in range(ncols):
        for j in range(i):
            s = columns[i] ^ columns[j]
            for h in range(j):
                value = s ^ columns[h]
                five += pairs.get(value, 0)
                six += triples.get(value, 0)
                triples[value] = triples.get(value, 0) + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def NEW_CTX(rng, G, idx):
    """AP-containing A contracts and concentration-aware graphic B search."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 19
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    load_weight = (0.0, 0.2, 0.5, 1.0, 2.0)[marker]

    def direction(d):
        d %= q
        return min(d, q - d)

    def directions(cell):
        return {direction(a - b) for a, b in combinations(cell, 2)}

    def normalized(cell, multiplier=1):
        values = [(multiplier * a) % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def geometry_key(cells):
        # Quotient cyclic multipliers, independent cell origins and
        # column permutations. Arbitrary Tanner equivalence remains
        # the evaluator's responsibility.
        return min(tuple(sorted(normalized(cell, u) for cell in cells))
                   for u in range(1, q))

    # Cache the two actual singleton actions, not a hand-written lift.
    # Cell addition and block placement are linear over F2. The selected
    # components are checked against the complete verified lift below.
    block_mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz1)

    def component(grid, side):
        width = len(grid[0]) * q
        rows = [0] * (len(grid) * q)
        for i, grid_row in enumerate(grid):
            for j, cell in enumerate(grid_row):
                shift = j * q
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        rows[i * q + t] ^= bits << shift
        return rows, width

    # Every retained geometry contains an AP trinomial. In C19 these
    # have two undirected internal differences, versus three for Sidon
    # trinomials. This is a deliberate departure from the archived model.
    # Normalize one nonzero binomial direction to one using a common
    # group automorphism, then quotient the remaining obvious gauges.
    trinomials = sorted({
        normalized((0, a, b))
        for a, b in combinations(range(1, q), 2)
    })
    ap_cells = {cell for cell in trinomials
                if len(directions(cell)) == 2}
    geometries = set()
    for first, second in combinations(trinomials, 2):
        if first not in ap_cells and second not in ap_cells:
            continue
        for b in range(2, (q + 1) // 2):
            geometries.add(geometry_key(
                ((0, 1), (0, b), first, second)))

    book = []
    for signature in sorted(geometries):
        grid = [[[(a,) for a in cell] for cell in signature]]
        rows, width = component(grid, "A")
        stats = _kernel_statistics(rows, width)
        if stats[0] or stats[1] or gf2_rank(rows) != q:
            continue
        # Actual kernel multiplicity comes before the local Tanner-cycle
        # penalty. AP cycles are admitted, not asserted to help BP.
        key = (stats[2], _four_cycles(rows), _six_cycles(rows),
               stats[3], signature)
        book.append((key, grid))
    book.sort(key=lambda entry: entry[0])
    assert len(book) >= 5
    wanted = 5 * batch + marker
    if wanted >= len(book):
        raise ValueError("No unused canonical A geometry for this batch")
    akey, A = book[wanted]
    bin_directions = {
        next(iter(directions(tuple(g[0] for g in cell))))
        for cell in A[0] if len(cell) == 2
    }

    # Each column is a graph edge. Four structural contracts prescribe
    # internal-edge counts 2/2, 2/1, 1/1 and 3/3 on the two row orbits.
    # All have fourteen terms; the last contract is new to this model.
    profiles = (
        ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
        ((2,0),(2,0),(0,2),(1,1),(1,1),(1,1),(1,1)),
        ((2,0),(0,2),(1,1),(1,1),(1,1),(1,1),(1,1)),
        ((2,0),(2,0),(2,0),(0,2),(0,2),(0,2),(1,1)),
    )
    profile = profiles[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2, 2, 3, 3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [(1, j) for j in cross]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def gauge(grid):
        # Remove the common cross-row voltage origin, not a new draw.
        origin = grid[1][cross[0]][0][0]
        for j in cross:
            grid[1][j] = [((grid[1][j][0][0] - origin) % q,)]
        return grid

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for i, j in internal:
            grid[i][j] = [(0,), (rng.randrange(1, (q + 1) // 2),)]
        ratios = rng.sample(range(q), len(cross))
        for j, ratio in zip(cross, ratios):
            grid[0][j], grid[1][j] = [(0,)], [(ratio,)]
        return gauge(grid)

    def assess_B(grid):
        """Exact simple-graph statistics from verified incidence actions.

        When parallel edges are absent, triangles and graph quartets
        are precisely component kernel supports of weights three/four.
        Worst-edge quartet load measures concentration, not ASC hooks.
        """
        rows, width = component(grid, "B")
        endpoints = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                endpoints[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        edges = set()
        adjacency = [0] * len(rows)
        for value in endpoints:
            assert value.bit_count() == 2
            low = value & -value
            a = low.bit_length() - 1
            b = (value ^ low).bit_length() - 1
            edges.add((a, b))
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a
        repeated = width - len(edges)

        triangle_sum = sum(
            (adjacency[a] & adjacency[b]).bit_count()
            for a, b in edges)
        assert triangle_sum % 3 == 0
        triangles = triangle_sum // 3

        partitions = 0
        for a, neighbors in enumerate(adjacency):
            for other in adjacency[:a]:
                overlap = (neighbors & other).bit_count()
                partitions += overlap * (overlap - 1) // 2
        assert partitions % 2 == 0
        quartets = partitions // 2

        peak = total_load = 0
        if not repeated and not triangles:
            for a, b in edges:
                neighbors = adjacency[a] & ~(1 << b)
                load = 0
                while neighbors:
                    bit = neighbors & -neighbors
                    c = bit.bit_length() - 1
                    load += (
                        adjacency[c] & adjacency[b]).bit_count() - 1
                    neighbors ^= bit
                peak = max(peak, load)
                total_load += load
            assert total_load == 4 * quartets

        stats = (repeated, triangles, quartets, peak)
        energy = ((300 * repeated + 50 * triangles + quartets) / q
                  + load_weight * peak)
        return energy, stats, rows

    pool, seen = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != 2 * q - 1:
            return
        signature = tuple(tuple(g[0] for g in cell)
                          for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        coupling = sum(
            direction(grid[i][j][1][0] - grid[i][j][0][0])
            in bin_directions for i, j in internal)
        key = (stats[2] / q + load_weight * stats[3],
               stats[3], stats[2], coupling)
        serial += 1
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    # Cached verified actions and graph intersection counts permit a
    # substantially deeper search than the parent's pair-syndrome census.
    for restart in range(6):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(960):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [
                    (0,), (rng.randrange(1, (q + 1) // 2),)]
            else:
                proposal[i][j] = [(rng.randrange(q),)]
            gauge(proposal)
            proposed, ps, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 6.0 * (0.994 ** step) + 0.03
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No connected triangle-free B graph in contract")

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def quotient_columns(stabilizers, n):
        # A linear canonical remainder modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residual = [0] * n
        for c in range(n):
            row = pivots.get(c)
            if row is None:
                residual[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def logical_risk(checks, stabilizers, n):
        """Exact short-logical classification, with circuit-risk surrogates.

        Shared opposite-basis checks mark possible ancilla-spread pairs;
        they do not certify a particular ASC hook ordering.
        """
        cs = columns(checks, n)
        spread = columns(stabilizers, n)
        if 0 in cs or len(set(cs)) != n:
            return None
        residual = quotient_columns(stabilizers, n)
        singles = dict(zip(cs, residual))
        exposure = [s.bit_count() + h.bit_count()
                    for s, h in zip(cs, spread)]
        buckets = {}
        hooks = weighted = double_hooks = partitions = 0

        for i, si in enumerate(cs):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                syndrome = si ^ cs[j]
                label = ri ^ residual[j]
                third = singles.get(syndrome)
                if third is not None and third != label:
                    return None
                h = (hi & spread[j]).bit_count()
                w = exposure[i] * exposure[j]
                bucket = buckets.get(syndrome)
                if bucket is None:
                    buckets[syndrome] = [1, h, w, {label: [1, h, w]}]
                    continue
                same = bucket[3].get(label)
                sc, sh, sw = same if same is not None else (0, 0, 0)
                dc, dh, dw = bucket[0] - sc, bucket[1] - sh, bucket[2] - sw
                # Equal syndrome and unequal quotient labels give a
                # nontrivial quartet. Distinct syndrome columns make
                # colliding different pairs disjoint.
                partitions += dc
                hooks += h * dc + dh
                weighted += w * dc + dw
                double_hooks += h * dh
                bucket[0] += 1
                bucket[1] += h
                bucket[2] += w
                if same is None:
                    bucket[3][label] = [1, h, w]
                else:
                    same[0] += 1
                    same[1] += h
                    same[2] += w
        assert partitions % 3 == 0
        return hooks, weighted, double_hooks, partitions // 3

    winner = None
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 288:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        # Two-data-fault gate exposure now precedes shared-check hook
        # opportunities. Neither quantity certifies the measured ASC
        # ordering. Protect both bases rather than one unproven block.
        score = tuple(value for t in (1, 0, 2, 3)
                      for value in (max(px[t], pz[t]), px[t] + pz[t]))
        score += coarse
        if winner is None or score < winner[0]:
            winner = (score, B, hx, hz, px, pz)

    if winner is None:
        raise ValueError("No k=288 filling passed full-CSS short-logical screen")
    score, B, hx, hz, px, pz = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == component(A, "A")[0]
    assert actual_b == component(B, "B")[0]
    assert any(len(cell) == 3 and
               len(directions(tuple(g[0] for g in cell))) == 2
               for cell in A[0])
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 288
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 18240
    _, bstats, _ = assess_B(B)
    return {"A": A, "B": B, "design": (architecture, wanted),
            "k": 288, "logical_risk": (px, pz),
            "component_contract": akey, "cycle_score": score,
            "quartet_contract": load_weight,
            "B_quartets_and_peak": bstats[2:]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete AP/graph-contract cell without late redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: Asymmetric A<=3/B<=4 component exclusion balances C19 memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored only 3.1933 despite much lower Z failed-shot rates.
# REFUTED: Difference-disjoint Sidon A cells, triangle-free graphic B and shared-check quartic minimization suffice for rare C19 circuit failures: fitness was 3.3458, measured Z failures were 106..148/250, and every measured code's failures collectively affected all 288 logicals.
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
