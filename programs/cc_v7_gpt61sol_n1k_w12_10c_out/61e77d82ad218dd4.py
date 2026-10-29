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
CONCEPT = ("Cyclic Z11, order 11, A=B=(3,11): exact weight-five component-kernel multiplicity minimization followed by witness-directed label/support repair should yield five new measured distance readings >=6. This hypothesis is falsifiable; component screens do not certify quantum distance.")
GROUP = ("Frozen cyclic specialization Z11:Z1, s=1, q=11. This bed repeatedly measured fitness 12.308 with five distance-five readings. Keep the group and test whether early-return kernel screening concealed useful distinctions between fillings; all measured distances remain provisional upper bounds.")
SHAPES = ("Frozen A=B=(3,11): equal components have 121 variables and 33 checks. n=1430, k>=704, and guaranteed rate 64/130=0.4923077 target cell 0.500. Parent minimum witnesses were entirely sector one, often involving Acol2-6 across many B columns; repair both component families and retain the actual lifted-rate gate.")
FILLING = ("Six degree-two and five degree-three columns retain nine terms per row and CSS cap 9+3=12. Twenty cards cross 0..3 degree-two loops with 0..4 mixed degree-three columns, newly including all-monomial triple columns. Verified component lifts supply exact kernel exclusion through four and complete weight-five counts using the cyclic orbit action. Repair detected witnesses by fresh labels and margin-preserving triple-column support switches; rank by kernel floor, residual five-word multiplicity and four-cycles. Full verified lifts check CSS, weights and rate.")
HOW_AB = FILLING
FAMILY_NAME = "r050_m11r1s1_311x311_exact5_witness_switch_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same metacyclic construction, resized cyclic specialization.
    # q=11 and size factor 130 give n=1430.
    m, r, s = 11, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 11)              # (ra, ca)
B_SHAPE = (3, 11)              # (rb, cb)


def _balanced_cells(rng, G, idx, attempt, side):
    """Six double columns, five triple columns; every row has nine terms."""
    q = G.q
    cells = [[[] for _ in range(11)] for _ in range(3)]
    if side == "A":
        d2 = 2 + idx % 4
        choice = idx // 4 + attempt // 11
    else:
        d2 = 2 + (idx // 4) % 4
        choice = idx + attempt // 11
    options = [d for d in range(2, 6) if d != d2]
    d3 = options[choice % len(options)]

    # Relative edge differences have distinct magnitudes modulo 11.
    # Select triangle-free holonomy for the degree-two subgraph.
    forbidden = {(v - u - w) % q
                 for u in (0, 1) for v in (0, d2) for w in (0, d3)}
    allowed = [h for h in range(q) if h not in forbidden]
    h = allowed[rng.randrange(len(allowed))]
    for start, r0, r1, shifts in (
            (0, 0, 1, (0, 1)),
            (2, 0, 2, (0, d2)),
            (4, 1, 2, (h, (h + d3) % q))):
        for off, shift in enumerate(shifts):
            cells[r0][start + off] = [G.e]
            cells[r1][start + off] = [(shift, 0)]

    # Fix one column gauge, but vary both independent triple-column labels.
    for j in range(6, 11):
        cells[0][j] = [G.e]
        cells[1][j] = [G.pick(rng)]
        cells[2][j] = [G.pick(rng)]
    return cells


def _component_rows(hx, hz, q):
    """Extract both components from the verified evaluator-convention lift."""
    mask = (1 << q) - 1
    ar = []
    for i in range(3):
        for r in range(q):
            raw = hx[(i * 11) * q + r]
            row = 0
            for j in range(11):
                row |= ((raw >> (j * 11 * q)) & mask) << (j * q)
            ar.append(row)
    bmask = (1 << (11 * q)) - 1
    br = [hz[a * q + r] & bmask
          for a in range(3) for r in range(q)]
    return ar, br


def _kernel_quality(rows, width):
    """Screen all component kernels through weight five; count triple collisions.

    A return floor of six means no kernel of weight <=5 was found by
    exhaustive syndrome enumeration. It is NOT a quantum-distance bound.
    """
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    if 0 in cols:
        return (1, 0)
    singles = set(cols)
    if len(singles) != width:
        return (2, 0)

    pairs = {}
    for i in range(width):
        for j in range(i):
            syndrome = cols[i] ^ cols[j]
            if syndrome in singles:
                return (3, 0)
            if syndrome in pairs:
                return (4, 0)
            pairs[syndrome] = (1 << i) | (1 << j)

    triples = {}
    collisions = 0
    for i in range(width):
        ci = cols[i]
        for j in range(i):
            cij = ci ^ cols[j]
            for k in range(j):
                syndrome = cij ^ cols[k]
                if syndrome in pairs:
                    used = (1 << i) | (1 << j) | (1 << k)
                    if not (pairs[syndrome] & used):
                        return (5, 0)
                old = triples.get(syndrome, 0)
                collisions += old
                triples[syndrome] = old + 1
    # With weights <=4 excluded, distinct colliding triples are disjoint.
    return (6, -collisions)


def NEW_CTX(rng, G, idx):
    """Optimize support topology and labels, not copies of one voltage model."""
    from itertools import permutations
    from math import exp

    q = G.q
    card = idx % 20
    loop_budget = card % 4
    mixed_budget = card // 4
    mixed_types = tuple(sorted(set(permutations((2, 1, 0)))))

    def component_rows(cells, side):
        # Empty partners remove the other sector only for screening.
        # Both actions still come from the verified construction helper.
        if side == "A":
            rows, _, _ = lifted_matrices(G, cells, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], cells)
        return rows

    def columns(rows, width):
        out = [0] * width
        for check, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << check
                row ^= bit
        return out

    def exact_five(cells, side):
        """Exact component kernel floor through five and five-word count.

        All syndromes come from lifted_matrices. Global cyclic translation
        is transitive inside each column block on this frozen abelian bed.
        A weight-five word has a free orbit of size eleven.
        """
        width = 11 * q
        cols = columns(component_rows(cells, side), width)
        singles = {}
        for i, syndrome in enumerate(cols):
            if not syndrome:
                return (1, 0, 0), 1 << i
            if syndrome in singles:
                return (2, 0, 0), singles[syndrome] | (1 << i)
            singles[syndrome] = 1 << i

        pairs = {}
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                if syndrome in singles:
                    return (3, 0, 0), word | singles[syndrome]
                previous = pairs.get(syndrome)
                if previous is not None:
                    return (4, 0, 0), word ^ previous
                pairs[syndrome] = word

        hits = 0
        witness = None
        for root in range(0, width, q):
            cr = cols[root]
            rootbit = 1 << root
            for i, x in enumerate(cols):
                if i == root:
                    continue
                partial = cr ^ x
                for j in range(i):
                    if j == root:
                        continue
                    pair = pairs.get(partial ^ cols[j])
                    if pair is None:
                        continue
                    triple = rootbit | (1 << i) | (1 << j)
                    if pair & triple:
                        continue
                    hits += 1
                    if witness is None or rng.randrange(64) == 0:
                        witness = pair | triple

        # For each word and distinguished variable, six pair/triple
        # partitions contain that variable in the triple. Summing over
        # one representative per column orbit gives 30*n5/q hits.
        assert (hits * q) % 30 == 0
        n5 = hits * q // 30

        check_pairs = {}
        four = 0
        for syndrome in cols:
            checks = []
            while syndrome:
                bit = syndrome & -syndrome
                checks.append(bit.bit_length() - 1)
                syndrome ^= bit
            for a, u in enumerate(checks):
                for v in checks[:a]:
                    key = (v, u)
                    old = check_pairs.get(key, 0)
                    four += old
                    check_pairs[key] = old + 1
        return (6 if n5 == 0 else 5, -n5, -four), witness

    def repair_component(cells, side, quality, witness):
        """Repair actual short words; never manufacture orbit-copy slots."""
        current = [[list(cell) for cell in row] for row in cells]
        best = [[list(cell) for cell in row] for row in cells]
        current_quality = best_quality = quality
        current_witness = witness

        for step in range(48):
            proposal = [[list(cell) for cell in row] for row in current]
            switched = False

            # A 2x2 incidence switch preserves every row total and both
            # affected column degrees. Only degree-three columns switch,
            # so the deliberate degree-two loop budget stays unchanged.
            if rng.random() < 0.22 and mixed_budget:
                i, k = rng.sample(range(3), 2)
                j, l = rng.sample(range(6, 11), 2)
                counts = [[len(cell) for cell in row] for row in proposal]
                if counts[i][j] and counts[k][l]:
                    counts[i][j] -= 1
                    counts[k][l] -= 1
                    counts[k][j] += 1
                    counts[i][l] += 1
                    valid = (
                        max(counts[r][c]
                            for r in range(3) for c in (j, l)) <= 2
                        and sum(max(counts[r][c] for r in range(3)) == 2
                                for c in range(6, 11)) == mixed_budget
                    )
                    if valid:
                        for r, c in ((i, j), (k, l), (k, j), (i, l)):
                            proposal[r][c] = rng.sample(
                                G.elements, counts[r][c])
                        switched = True

            if not switched:
                touched = set()
                word = current_witness or 0
                while word:
                    bit = word & -word
                    touched.add((bit.bit_length() - 1) // q)
                    word ^= bit
                places = [
                    (i, j) for i in range(3) for j in range(11)
                    if proposal[i][j] and
                    (not touched or j in touched)
                ]
                if rng.random() < 0.15:
                    places = [(i, j) for i in range(3) for j in range(11)
                              if proposal[i][j]]
                i, j = rng.choice(places)
                t = rng.randrange(len(proposal[i][j]))
                old = proposal[i][j][t]
                available = [
                    g for g in G.elements
                    if g != old and g not in proposal[i][j]
                ]
                proposal[i][j][t] = rng.choice(available)

            trial_quality, trial_witness = exact_five(proposal, side)
            if (trial_quality > current_quality or
                    (trial_quality == current_quality and
                     rng.random() < 0.30)):
                current = proposal
                current_quality = trial_quality
                current_witness = trial_witness
                if trial_quality > best_quality:
                    best = [[list(cell) for cell in row] for row in current]
                    best_quality = trial_quality
            if best_quality[0] == 6 and step >= 16:
                break

        assert all(sum(map(len, row)) == 9 for row in best)
        return best, best_quality

    def support_profiles():
        profiles = {}
        for _ in range(96):
            hyper = tuple(sorted(
                [mixed_types[rng.randrange(len(mixed_types))]
                 for _ in range(mixed_budget)]
                + [(1, 1, 1)] * (5 - mixed_budget)))
            degree = tuple(9 - sum(h[i] for h in hyper)
                           for i in range(3))
            if min(degree) < 0:
                continue
            for l0 in range(loop_budget + 1):
                for l1 in range(loop_budget - l0 + 1):
                    loops = (l0, l1, loop_budget - l0 - l1)
                    if any(2 * loops[i] > degree[i] for i in range(3)):
                        continue
                    cross = (
                        (degree[0] + degree[1] - degree[2]) // 2
                        - loops[0] - loops[1] + loops[2],
                        (degree[0] + degree[2] - degree[1]) // 2
                        - loops[0] - loops[2] + loops[1],
                        (degree[1] + degree[2] - degree[0]) // 2
                        - loops[1] - loops[2] + loops[0])
                    if min(cross) < 0 or sum(v > 0 for v in cross) < 2:
                        continue
                    pairs = []
                    for i, count in enumerate(loops):
                        pairs.extend([(i, i)] * count)
                    for pair, count in zip(((0, 1), (0, 2), (1, 2)), cross):
                        pairs.extend([pair] * count)
                    if len(pairs) == 6:
                        key = (loops, cross, hyper)
                        profiles[key] = (tuple(pairs), hyper)
        result = list(profiles.values())
        rng.shuffle(result)
        if not result:
            raise ValueError("no support profile satisfies the deliberate card")
        return result[:8]

    def graph_cells(pairs, voltages):
        cells = [[[] for _ in range(6)] for _ in range(3)]
        for j, ((a, b), voltage) in enumerate(zip(pairs, voltages)):
            g = (voltage, 0)
            if a == b:
                cells[a][j] = [G.e, g]
            else:
                cells[a][j] = [G.e]
                cells[b][j] = [g]
        return cells

    def graph_cost(cells, side):
        # Degree-two variables are edges between actual lifted checks.
        cols = columns(component_rows(cells, side), 6 * q)
        duplicate = len(cols) - len(set(cols))
        adj = [0] * (3 * q)
        edges = set()
        for syndrome in cols:
            assert syndrome.bit_count() == 2
            low = syndrome & -syndrome
            a = low.bit_length() - 1
            b = (syndrome ^ low).bit_length() - 1
            adj[a] |= 1 << b
            adj[b] |= 1 << a
            edges.add((min(a, b), max(a, b)))

        triangles = sum((adj[a] & adj[b]).bit_count()
                        for a, b in edges) // 3
        four = 0
        for a in range(len(adj)):
            for b in range(a):
                common = (adj[a] & adj[b]).bit_count()
                four += common * (common - 1) // 2
        four //= 2

        pentagons = 0
        if duplicate == triangles == four == 0:
            twohop = []
            for a, neighbors in enumerate(adj):
                reach = 0
                while neighbors:
                    low = neighbors & -neighbors
                    reach |= adj[low.bit_length() - 1]
                    neighbors ^= low
                twohop.append(reach & ~(1 << a))
            # With no triangles or squares, each intersection represents
            # a simple pentagon through the selected closing edge.
            pentagons = sum((twohop[a] & twohop[b]).bit_count()
                            for a, b in edges) // 5
        return 100000 * duplicate + 10000 * triangles + 100 * four + pentagons

    def component_pool(side):
        graphs = []
        for pairs, hyper in support_profiles():
            voltages = [rng.randrange(1 if a == b else 0, q)
                        for a, b in pairs]
            cells = graph_cells(pairs, voltages)
            cost = graph_cost(cells, side)
            best_cost = cost
            best_voltages = list(voltages)
            for step in range(128):
                if best_cost == 0:
                    break
                j = rng.randrange(6)
                old = voltages[j]
                a, b = pairs[j]
                proposal = rng.randrange(1 if a == b else 0, q)
                if proposal == old:
                    continue
                voltages[j] = proposal
                trial = graph_cost(graph_cells(pairs, voltages), side)
                temperature = max(0.6, 1800 * (1 - step / 128) ** 3)
                if trial <= cost or rng.random() < exp((cost - trial) / temperature):
                    cost = trial
                    if trial < best_cost:
                        best_cost = trial
                        best_voltages = list(voltages)
                else:
                    voltages[j] = old
            graphs.append((best_cost, pairs, hyper, best_voltages))
        graphs.sort(key=lambda record: record[0])

        pool = []
        for cost, pairs, hyper, voltages in graphs[:4]:
            graph = graph_cells(pairs, voltages)
            for _ in range(8):
                cells = [[list(cell) for cell in row] for row in graph]
                for h in hyper:
                    for i in range(3):
                        cells[i].append(rng.sample(G.elements, h[i]))
                assert all(sum(map(len, row)) == 9 for row in cells)
                quality, witness = exact_five(cells, side)
                pool.append((quality + (-cost,), cells, witness))
        pool.sort(key=lambda record: record[0], reverse=True)

        refined = []
        for score, cells, witness in pool[:3]:
            repaired, quality = repair_component(
                cells, side, score[:3], witness)
            graph = [[list(cell) for cell in row[:6]]
                     for row in repaired]
            cost = graph_cost(graph, side)
            refined.append((quality + (-cost,), repaired))

        # Retain an independently generated alternative as well as the
        # repaired candidates; do not fill slots with their relabelings.
        refined.append((pool[0][0], pool[0][1]))
        refined.sort(key=lambda record: record[0], reverse=True)
        return refined[:4]

    apool = component_pool("A")
    bpool = component_pool("B")
    best = None
    best_score = None
    for qa, A in apool:
        for qb, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (0.475 <= k / n < 0.525):
                continue
            score = (min(qa[0], qb[0]), qa[0] + qb[0],
                     qa[1] + qb[1], qa[2] + qb[2],
                     qa[3] + qb[3], k)
            if best_score is None or score > best_score:
                best_score = score
                best = {"A": A, "B": B, "k": k, "quality": score}
    if best is None:
        raise ValueError("no new topology remained in target rate band")

    A, B = best["A"], best["B"]
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, including binomials and zero blocks."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The fixed doubled-triangle monomial topology with 72 voltage trials suppressed sector-1 weight-five logicals; all five measured codes instead had distance five and many sector-1 minimum witnesses.
# REFUTED: Optimizing degree-two graph cycles and using an early-return component kernel-five screen raised the five-code distance reading above five; all five measured current codes still returned five, entirely in sector one.
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
