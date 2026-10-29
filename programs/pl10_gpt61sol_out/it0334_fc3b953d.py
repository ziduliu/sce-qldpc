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
CONCEPT = ("For C3 semidirect C6 with action 2, order 18, shapes (1,4)x(2,7), mixed-degree A protection and lower-load two-row B architectures should improve worse-basis circuit fitness above 3.1980 while retaining measured rate in cell 0.500.")
GROUP = ("L2 resize within the same split-metacyclic construction rule: m=3, r=6, s=2, with 2^6=1 mod 3. The new size factor thirty cannot retain q=36 and permits q<=19; choose nonabelian q=18 and n=540. Order-six ratios allow non-involutory bipartite binomial backbones. Related (1,4)x(2,7) beds reached 3.3458 for C19 and 3.2584 for Q16, but do not establish an advantage for these untried parameters.")
SHAPES = ("Change to (1,4)x(2,7): n=540, kb=270 and guaranteed rate 0.500. Unequal components have 72 variables/18 checks and 126 variables/36 checks. Require complete verified 0.475<=k/n<0.525 rather than assuming component deficits predict k. Parent weight-four witnesses widely touched Acol1/2 and many B columns; diffuse 1.0-1.1x data and Acol/Brow check enrichment does not identify a localized repair site.")
FILLING = ("Twenty contracts cross five canonical A geometries selected from [2,2,2,3] and [2,2,3,3] with four B architectures: seven cross-row edges, one internal edge, one internal edge per row, or one singleton anchor. Further batches consume successive canonical A geometries. Helper-derived singleton actions screen exact component kernels through three, rank and connectivity. Rooted uncolored refinement conservatively collapses component proposals. Complete verified CSS selection minimizes worse-basis stabilizer-quotiented, gate-weighted two-fault ambiguity with a diluted possible-spread term. HX weight is <=12 and HZ <=11; six-round CNOT budgets are 15768,16308,16740 or17280 versus the parent's 16416. Target the parent's 146-181 Z failures/250 and collective flips across all 296 logicals. Structural costs omit ASC ordering and are not calibrated pL measurements.")
FAMILY_NAME = "r050_m3r6s2_14x27_mixeddegree_anchor_faultmass_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 18

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires q=18")
    # Same split metacyclic rule: x^3=y^6=1, y*x*y^-1=x^2.
    m, r, s = 3, 6, 2
    assert pow(s, r, m) == 1 and m*r == q
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# A profile is (number of order-four directions,
#               number of order-six even-coset directions).
# Remaining directions have order eighteen.
_A_DIRECTION_PROFILES = (
    (1, 0), (1, 1), (2, 0), (2, 1), (3, 0),
)
_B_DIRECTION_BATCHES = (
    ((2, 0), (3, 0), (4, 0), (5, 0)),
    ((1, 1), (2, 1), (3, 1), (4, 1)),
    ((2, 0), (2, 1), (3, 0), (3, 1)),
)


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


def _retired_rotation9_CTX(rng, G, idx):
    """Inactive previous-shape filling, retained only for provenance."""
    design, batch = idx % 20, idx // 20
    # Counts are (order four, order nine, order six).
    # Remaining directions have order eighteen. Avoid two order-nine
    # rotation directions: their available undirected ratios force
    # triangles in C9. Component screening checks the actual matrices.
    a_profiles = (
        (1, 1, 0), (1, 1, 1), (1, 0, 1),
        (2, 1, 0), (2, 0, 0),
    )
    b_profiles = (
        (2, 1, 1), (3, 1, 0),
        (3, 1, 1), (4, 1, 0),
    )
    ap = a_profiles[design // 4]
    bp = b_profiles[design % 4]
    sa, sb = [[2] * A_SHAPE[1]], [[2] * B_SHAPE[1]]
    assert G.q == 36
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    # Pure local state: every batch searches the same contract catalogue
    # and consumes a different ranked finalist, rather than a fresh
    # draw of a potentially identical canonical geometry.
    rng = random.Random(369041 + 1009 * design)

    def draw(width, profile):
        odd, rotation, short = profile
        long = width - odd - rotation - short
        assert 0 <= long <= 3 and rotation <= 1 and short <= 1
        directions = [(a, 1) for a in rng.sample(range(9), odd)]
        directions += [(a, 0) for a in rng.sample((1, 2, 4), rotation)]
        directions += [(3, 2)] * short
        directions += [(a, 2) for a in rng.sample((1, 2, 4), long)]
        assert len(directions) == width
        assert len({G._key(g) for g in directions}) == width
        return [[[G.e, g] for g in directions]]

    def signature(A, B):
        """Joint group automorphisms, binomial inversion and permutations.

        x -> x^u, y -> x^v*y^w, with u a unit modulo nine and w odd.
        Every orbit has an A odd direction with rotation coordinate
        zero, so anchor there instead of enumerating every v.
        This is not a complete arbitrary Tanner-isomorphism test.
        """
        images = []
        anchors = [cell[1][0] for cell in A[0] if cell[1][1] % 2]
        for u in (1, 2, 4, 5, 7, 8):
            for anchor in anchors:
                v = (-u * anchor) % 9
                for w in (1, 3):
                    sides = []
                    for grid in (A, B):
                        values = []
                        for cell in grid[0]:
                            a, b = cell[1]
                            h = ((u*a + v*(b % 2)) % 9, (w*b) % 4)
                            values.append(min(G.idx(h), G.idx(G.inv(h))))
                        sides.append(tuple(sorted(values)))
                    images.append(tuple(sides))
        return min(images)

    def connected(rows, width):
        """Connectivity of the extracted component Tanner graph."""
        neighbors = [[] for _ in range(width)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                neighbors[bit.bit_length() - 1].append(r)
                row ^= bit
        adjacency = [set() for _ in rows]
        for checks in neighbors:
            for r in checks:
                adjacency[r].update(checks)
        seen, todo = {0}, [0]
        while todo:
            r = todo.pop()
            for t in adjacency[r]:
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return len(seen) == len(rows)

    def assess(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 576
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 19 * n <= 40 * k < 21 * n):
            return None
        ar, br = _component_rows(hx, hz, G.q)
        ast = _kernel_statistics(ar, A_SHAPE[1] * G.q)
        bst = _kernel_statistics(br, B_SHAPE[1] * G.q)
        # Distinct nonzero columns and no triples establish component
        # kernel exclusion through weight three, not quantum distance.
        if any(ast[:2]) or any(bst[:2]):
            return None
        if not connected(ar, A_SHAPE[1] * G.q):
            return None
        if not connected(br, B_SHAPE[1] * G.q):
            return None
        # Replication-weighted component counts are only a shortlist
        # surrogate. B supplies the HZ-syndrome channel; Z-memory was
        # markedly worse in the parent. Test a twofold emphasis, without
        # interpreting this coefficient as a calibrated failure model.
        az, bx = B_SHAPE[1] * ast[2], A_SHAPE[1] * bst[2]
        fa, fb = _four_cycles(ar), _four_cycles(br)
        key = (max(az, 2*bx), az + 2*bx, fa + fb,
               max(ast[3], 2*bst[3]), ast[3] + 2*bst[3])
        return key, hx, hz, k

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def quotient_columns(rows, width):
        """Canonical linear quotient by the VERIFIED stabilizer span."""
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        # Each pivot row's tail uses smaller columns. Process ascending.
        residual = [0] * width
        for c in range(width):
            pivot = pivots.get(c)
            if pivot is None:
                residual[c] = 1 << c
                continue
            tail, value = pivot ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def fault_pair_profile(checks, stabilizers, width):
        """Stabilizer-aware <=4 logical ambiguity and hook surrogates.

        A shared opposite check indicates a possible spread pair, not
        a guarantee that the final ASC ordering realizes that hook.
        """
        syndrome = columns(checks, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        spread = columns(stabilizers, width)
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        degrees = [x.bit_count() + z.bit_count()
                   for x, z in zip(syndrome, spread)]
        buckets = {}
        double_hook = hook_exposure = weighted_pairs = logical_pairs = 0

        for i in range(width):
            si, ri, hi = syndrome[i], residual[i], spread[i]
            for j in range(i):
                s = si ^ syndrome[j]
                label = ri ^ residual[j]
                third = singles.get(s)
                if third is not None and third != label:
                    return None
                hook = (hi & spread[j]).bit_count()
                exposure = degrees[i] * degrees[j]
                old = buckets.get(s)
                if old is None:
                    buckets[s] = [1, hook, exposure,
                                  {label: [1, hook, exposure]}]
                    continue
                same = old[3].get(label)
                nc, nh, ne = same if same is not None else (0, 0, 0)
                different = old[0] - nc
                different_hooks = old[1] - nh
                different_exposure = old[2] - ne
                logical_pairs += different
                double_hook += hook * different_hooks
                hook_exposure += hook * different + different_hooks
                weighted_pairs += exposure * different + different_exposure
                old[0] += 1
                old[1] += hook
                old[2] += exposure
                if same is None:
                    old[3][label] = [1, hook, exposure]
                else:
                    same[0] += 1
                    same[1] += hook
                    same[2] += exposure
        # With distinct nonzero syndrome columns, colliding distinct
        # pairs are disjoint. Each quartet has three pair partitions.
        assert logical_pairs % 3 == 0
        return double_hook, hook_exposure, weighted_pairs, logical_pairs // 3

    pool, seen = [], set()
    # Only complete legal doubletons are proposed. Canonicalization
    # selects representatives; it does not count orbit copies as designs.
    for trial in range(960):
        A = draw(A_SHAPE[1], ap)
        B = draw(B_SHAPE[1], bp)
        sig = signature(A, B)
        if sig in seen:
            continue
        seen.add(sig)
        A, B = [
            [[[G.e, G.elements[t]] for t in side]]
            for side in sig
        ]
        result = assess(A, B)
        if result is None:
            continue
        key, hx, hz, k = result
        pool.append((key, trial, A, B, hx, hz, k))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[8:]

    finalists = []
    for coarse, trial, A, B, hx, hz, k in pool:
        px = fault_pair_profile(hz, hx, 576)
        pz = fault_pair_profile(hx, hz, 576)
        if px is None or pz is None:
            continue
        # X faults flip a Z-memory state and have HZ syndromes.
        # Retain minimax protection, but emphasize this worse measured
        # channel before shared-check and gate-exposure tie breakers.
        key = tuple(v for t in range(4)
                    for v in (max(2*px[t], pz[t]), 2*px[t] + pz[t]))
        key += coarse
        finalists.append((key, trial, A, B, hx, hz, k, px, pz))

    finalists.sort(key=lambda item: (item[0], item[1]))
    if batch >= len(finalists):
        raise ValueError("Canonical rotation-nine contract finalists exhausted")
    key, trial, A, B, hx, hz, k, px, pz = finalists[batch]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx) == 8
    assert max(row.bit_count() for row in hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 16416
    assert any(g[1] == 0 and g != G.e for cell in B[0] for g in cell)
    return {"A": A, "B": B, "design": design, "batch": batch,
            "direction_contract": (ap, bp), "k": k,
            "canonical_geometry": signature(A, B),
            "fault_pair_profile": (px, pz), "screen": key}


def _shape_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _shape_quotient(rows, width):
    """Linear canonical remainders modulo verified stabilizer rows."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    residual = [0] * width
    for c in range(width):
        pivot = pivots.get(c)
        if pivot is None:
            residual[c] = 1 << c
            continue
        tail, value = pivot ^ (1 << c), 0
        while tail:
            bit = tail & -tail
            value ^= residual[bit.bit_length() - 1]
            tail ^= bit
        residual[c] = value
    return residual


def _shape_fingerprint(rows, width, roots):
    """Rooted uncolored Tanner invariant, not complete canonization.

    Regular translations cover each check-row orbit. Taking the minimum
    over one root per orbit makes this invariant under check relabeling.
    Equal fingerprints are conservatively discarded.
    """
    import hashlib
    from collections import Counter

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, value in enumerate(_shape_columns(rows, width)):
        while value:
            bit = value & -value
            r = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            value ^= bit

    images = []
    for root in roots:
        colors = [0] * nr + [1] * width
        colors[root] = 2
        digest = hashlib.sha256()
        for iteration in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(len(adjacency))
            ]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        images.append(digest.hexdigest())
    return min(images)


def NEW_CTX(rng, G, idx):
    """Mixed-degree contracts and verified two-row logical-fault selection."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 18
    assert A_SHAPE == (1, 4) and B_SHAPE == (2, 7)
    architecture, marker = divmod(idx % 20, 5)
    wanted = 5 * (idx // 20) + marker
    mask = (1 << q) - 1

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only singleton blocks produced by the verified helper.
    # Final component projections are independently checked below.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz1)

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        rows[i*q+t] ^= bits << (j*q)
        return rows, len(grid[0]) * q

    def order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("Element order exceeded group order")

    long = [g for g in G.elements if order(g) == 6]
    representatives = [
        g for g in long if G.idx(g) < G.idx(G.inv(g))]
    assert len(representatives) >= 3
    triangles = [
        [G.e, a, b]
        for a, b in combinations(
            [g for g in G.elements if g != G.e], 2)
    ]

    def connected(rows, width):
        adjacency = [0] * len(rows)
        for value in _shape_columns(rows, width):
            sites = []
            while value:
                bit = value & -value
                sites.append(bit.bit_length() - 1)
                value ^= bit
            for r in sites:
                for s in sites:
                    adjacency[r] |= 1 << s
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length() - 1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        return reached == (1 << len(rows)) - 1

    # Rebuild a deterministic relative-geometry bank in every pure call.
    # The slot contract chooses an invariant-distinct geometry, not a
    # fresh labeling draw or an automorphism of a saved code.
    catalogue_rng = random.Random(18031427)
    bank, seen_raw, seen_classes = [], set(), set()
    for trial in range(6144):
        number = 3 if trial % 3 == 0 else 2
        bins = catalogue_rng.sample(representatives, number)
        cells = [[G.e, g] for g in bins]
        cells += [
            list(cell) for cell in
            catalogue_rng.sample(triangles, 4-number)
        ]
        raw = tuple(sorted(
            tuple(sorted(G.idx(g) for g in cell)) for cell in cells))
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        A = [[[G.elements[t] for t in cell] for cell in raw]]
        rows, width = component(A, "A")
        stats = _kernel_statistics(rows, width)
        if any(stats[:2]) or gf2_rank(rows) != q:
            continue
        if not connected(rows, width):
            continue
        identity = _shape_fingerprint(rows, width, (0,))
        if identity in seen_classes:
            continue
        seen_classes.add(identity)
        key = (stats[2], sum(map(len, A[0])),
               _four_cycles(rows), stats[3], identity)
        bank.append((key, identity, A, rows))
        if len(bank) >= max(24, wanted + 8):
            break

    bank.sort(key=lambda entry: entry[0])
    if wanted >= len(bank):
        raise ValueError("Invariant-distinct mixed-degree A bank exhausted")
    akey, aid, A, ar = bank[wanted]
    sa = [[len(cell) for cell in A[0]]]
    assert sorted(sa[0]) in ([2,2,2,3], [2,2,3,3])

    profiles = (
        ((1,1),) * 7,
        ((2,0),) + ((1,1),) * 6,
        ((2,0),(0,2)) + ((1,1),) * 5,
        ((1,0),) + ((1,1),) * 6,
    )
    profile = profiles[architecture]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(map(sum, sb)) + max(sa[0]) <= 11

    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    cross = [j for j, p in enumerate(profile) if p == (1,1)]
    mutable = internal + [(1, j) for j in cross]
    expected_rank = 2*q if architecture == 3 else 2*q-1

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for j, p in enumerate(profile):
            if p == (1,0):
                grid[0][j] = [G.e]
        for i, j in internal:
            grid[i][j] = [G.e, rng.choice(long)]
        voltages = rng.sample(G.elements, len(cross))
        for j, g in zip(cross, voltages):
            grid[0][j], grid[1][j] = [G.e], [g]
        return grid

    def graph_statistics(grid):
        """Simple-graph statistics from verified component actions.

        The anchor architecture has degree-one incidence columns only
        on the upper orbit. Its other columns are cross-row edges.
        """
        rows, width = component(grid, "B")
        cols = _shape_columns(rows, width)
        repeated = width - len(set(cols)) + int(0 in cols)
        adjacency = [0] * len(rows)
        edges = set()
        for value in cols:
            if value.bit_count() == 1:
                continue
            assert value.bit_count() == 2
            low = value & -value
            a = low.bit_length() - 1
            b = (value ^ low).bit_length() - 1
            edges.add((a, b))
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a

        triangle_sum = sum(
            (adjacency[a] & adjacency[b]).bit_count()
            for a, b in edges)
        assert triangle_sum % 3 == 0
        triangles_count = triangle_sum // 3
        partitions = sum(
            v*(v-1)//2
            for a, neighbors in enumerate(adjacency)
            for other in adjacency[:a]
            for v in [(neighbors & other).bit_count()])
        assert partitions % 2 == 0
        quartets = partitions // 2

        colors = [-1] * len(rows)
        colors[0] = 0
        todo, reached, bipartite = [0], 1, True
        while todo:
            a = todo.pop()
            neighbors = adjacency[a]
            while neighbors:
                bit = neighbors & -neighbors
                b = bit.bit_length() - 1
                neighbors ^= bit
                if colors[b] < 0:
                    colors[b] = 1-colors[a]
                    reached += 1
                    todo.append(b)
                elif colors[b] == colors[a]:
                    bipartite = False
        disconnected = int(reached != len(rows))
        wrong_parity = int(
            architecture in (1,2) and bipartite)
        energy = (
            400*repeated + 100*triangles_count + quartets
            + 100*disconnected + 4*wrong_parity) / q
        return (energy,
                (repeated, triangles_count, quartets,
                 disconnected, wrong_parity), rows)

    pool, seen_B = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or stats[3] or stats[4]:
            return
        key = (stats[2], _four_cycles(rows))
        if len(pool) >= 12 and key > pool[-1][0]:
            return
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in grid for cell in row)
        if signature in seen_B:
            return
        if gf2_rank(rows) != expected_rank:
            return
        seen_B.add(signature)
        serial += 1
        pool.append((key, serial, clone(grid), list(rows)))
        pool.sort(key=lambda item: item[:2])
        del pool[12:]

    for restart in range(5):
        grid = draw_B()
        energy, stats, rows = graph_statistics(grid)
        remember(grid, stats, rows)
        for step in range(440):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [G.e, rng.choice(long)]
            else:
                occupied = {
                    proposal[1][c][0] for c in cross if c != j}
                choices = [g for g in G.elements if g not in occupied]
                proposal[i][j] = [rng.choice(choices)]
            proposed, ps, pr = graph_statistics(proposal)
            delta = proposed-energy
            temperature = 5.0*(0.989**step) + 0.04
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    # Do not spend the full-CSS comparison on several graph relabelings.
    bclasses, shortlist = set(), []
    for coarse, serial, B, br in pool:
        identity = _shape_fingerprint(br, 7*q, (0, q))
        if identity in bclasses:
            continue
        bclasses.add(identity)
        bst = _kernel_statistics(br, 7*q)
        if any(bst[:2]):
            continue
        shortlist.append((coarse, serial, identity, B, br))
        if len(shortlist) == 4:
            break
    if not shortlist:
        raise ValueError("No distinct connected B graph passed its contract")

    def fault_mass(checks, stabilizers, width):
        """Truncated fault ambiguity, NOT ASC circuit simulation.

        Singles and pairs are classified by actual syndrome and
        stabilizer quotient. Gate degree supplies exposure weights.
        Shared checks add a diluted possible-spread contribution;
        they do not certify the measured ASC ordering.
        """
        syndrome = _shape_columns(checks, width)
        spread = _shape_columns(stabilizers, width)
        residue = _shape_quotient(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        exposure = [
            1+a.bit_count()+b.bit_count()
            for a, b in zip(syndrome, spread)]
        coefficients = []
        for row in stabilizers:
            w = row.bit_count()
            coefficients.append(
                840*w // max(1, w*(w-1)//2))

        buckets = {0: {0: 100000000}}
        singles = dict(zip(syndrome, residue))

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0)+mass

        for i, s in enumerate(syndrome):
            add(s, residue[i], 65536*exposure[i])
        for i, si in enumerate(syndrome):
            ri, hi = residue[i], spread[i]
            for j in range(i):
                s, label = si ^ syndrome[j], ri ^ residue[j]
                third = singles.get(s)
                if third is not None and third != label:
                    return None
                common, possible_spread = hi & spread[j], 0
                while common:
                    bit = common & -common
                    possible_spread += coefficients[
                        bit.bit_length()-1]
                    common ^= bit
                add(s, label,
                    64*exposure[i]*exposure[j] + 4*possible_spread)

        loss = peak = ambiguous = 0
        for classes in buckets.values():
            if len(classes) > 1:
                error = sum(classes.values())-max(classes.values())
                loss += error
                peak = max(peak, error)
                ambiguous += 1
        return loss, peak, ambiguous

    winner = None
    for coarse, serial, bid, B, br in shortlist:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 540 or not (19*n <= 40*k < 21*n):
            continue
        px = fault_mass(hz, hx, n)
        pz = fault_mass(hx, hz, n)
        if px is None or pz is None:
            continue
        score = (
            max(px[0], pz[0])/k, (px[0]+pz[0])/k,
            max(px[1], pz[1]), px[1]+pz[1],
            px[2]+pz[2], coarse, bid)
        if winner is None or score < winner[0]:
            winner = (score, B, br, hx, hz, k, px, pz, bid)
    if winner is None:
        raise ValueError("No target-rate CSS finalist passed logical classification")

    score, B, br, hx, hz, k, px, pz, bid = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == ar and actual_b == br
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == k
    assert 0.475 <= k/540 < 0.525
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert max(row.bit_count() for row in hz) <= 11
    assert _no_dead_lines(sa, sb)
    assert all(len(cell) <= CELL_MAX and
               len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    expected_cnots = 6*q*(
        9*sum(map(sum, sa)) + 5*sum(map(sum, sb)))
    assert expected_cnots in (15768, 16308, 16740, 17280)
    assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
    return {"A": A, "B": B, "k": k,
            "design": (architecture, wanted),
            "component_classes": (aid, bid),
            "A_contract": akey, "fault_mass": (px, pz),
            "selection_score": score, "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified mixed-degree/graph-contract cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Dispersed doubletons and component short-word/cycle optimization sufficiently suppress collective D8 high-rate failures: fitness was 3.2454, measured worse-basis failures were 117-147/250, and every measured code retained abundant weight-four logicals.
# REFUTED: Connected triangle-free order-four/eighteen doubletons plus quotient hook minimization sufficiently suppress q36 high-rate failures: fitness was 3.1814, Z failures were 158-182/250, all 296 logicals were affected, and every measured code retained abundant weight-four witnesses.
# REFUTED: Adding order-nine directions and twofold HZ-channel ambiguity weighting sufficiently suppresses q36 (1,3)x(1,5) collective failures: fitness was 3.1980, measured Z failures were 146-181/250, all 296 logicals were affected, and abundant weight-four witnesses remained despite first-pass settlement.
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
