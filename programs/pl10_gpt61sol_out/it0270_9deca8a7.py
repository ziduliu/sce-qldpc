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
CONCEPT = ("For C16 semidirect C4 with action x->x^5, order 64, shapes A=(1,2), B=(1,4), long-order noncommuting relative supports and stabilizer-aware joint BP screening should improve the weak 0.350 frontier's circuit reliability; ASC measurements must test the hypothesis.")
GROUP = ("Jump from the order-24 C3:C8 parent to C16:C4(s=5), order 64. The action is valid because 5^4=1 modulo 16. Elements of orders eight and sixteen supply noncommuting long-cycle directions. No order-64 bed appears in BEDS. This is an untried group/shape experiment, not a claim that nilpotence or lift order guarantees better pL.")
SHAPES = ("Target cell 0.350 instead of repeating the parent's 0.450 bed. Shapes (1,2)x(1,4) give n=576, kb=192 and guaranteed rate 1/3; q_max=66 and this family uses q=64. A has 128 variables/64 checks and B has 256 variables/64 checks. Require verified 192<=k<=210. The parent's abundant sector-one weight-five/six witnesses and diffuse 1.0-1.2x Acol/Bcol enrichment, without a persistent Brow hotspot, motivate changing both component architectures.")
FILLING = ("Twenty deliberate contracts cross five support allocations with four relative-order policies. A counts are (3,3), (2,3) or (2,2); B uses long-ratio doubletons plus trinomials or a quartet. Both CSS weight sums are <=12; six-round CNOT budgets are 18048-21888. Singleton actions are extracted from lifted_matrices, composed linearly and checked against fresh helper-built component matrices. Exclude component kernels through four, then shortlist capped sampled witnesses and cycles. Full matrices require target k, connectivity, CSS and no quantum logical through four. Joint finalists receive common-random-number BP probes with degree-weighted data errors and correlated opposite-check pairs; settled residuals are classified modulo actual stabilizers. These probes omit ASC scheduling and measurement histories and are not measured pL. Target the parent's 109-138 Z failures in 750-1000 shots, collective flips across all 240 logicals and the failure of extraction-load reduction alone.")
FAMILY_NAME = "r035_C16byC4_s5_q64_12x14_longratio_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 64

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C16 semidirect C4 with action five requires q=64")
    # x^16=y^4=1, y*x*y^-1=x^5; 5^4=1 modulo 16.
    # Canonical element (a,b) is x^a*y^b.
    els = [(a, b) for b in range(4) for a in range(16)]

    def red(g):
        return (int(g[0]) % 16, int(g[1]) % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(5, b, 16)*c) % 16, (b + d) % 4)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        return ((-pow(5, bi, 16)*a) % 16, bi)

    def idx(g):
        a, b = red(g)
        return a + 16*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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


def _mc_orders(G):
    """Element orders from the actual validated multiplication."""
    orders = {}
    for g in G.elements:
        h = G.e
        for order in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                orders[G.idx(g)] = order
                break
        else:
            raise ValueError("element order missing")
    return orders


def _mc_pool(rng, G, side, draw, trials):
    """Fresh component search within one explicit structural contract."""
    width = (3 if side == "A" else 7) * G.q
    expected_rank = (1 if side == "A" else 2) * G.q
    pool, seen = [], set()
    for trial in range(trials):
        grid = draw()
        identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                         for row in grid for cell in row)
        if identity in seen:
            continue
        seen.add(identity)
        rows = _component_rows(G, grid, side)
        if gf2_rank(rows) != expected_rank:
            continue
        result = _kernel4(rows, width)
        if result is None:
            continue
        short, cols = result
        four = _four_cycles(rows)
        coarse = (short[0], short[1], four, short[2])
        pool.append((coarse, trial, grid, rows, cols, short, four))
        pool.sort(key=lambda entry: entry[:2])
        del pool[6:]

    if not pool:
        raise ValueError("no full-rank long-ratio component")

    finalists = []
    for _, trial, grid, rows, cols, short, four in pool:
        spectrum = _component_score(short, cols, rows, four)
        # Restrict to the actual degree-two component variables.
        # With <=4 excluded, equal triple syndromes count weight-six
        # supports exactly. This is a graph-cycle mechanism, not pL.
        two_six = 1 << 60
        if not (short[0] or short[1]):
            degree_two = [c for c in cols if c.bit_count() == 2]
            _, two_six = _kernel56(degree_two)
        key = (spectrum[0], spectrum[1], spectrum[2],
               two_six, spectrum[3], spectrum[4],
               spectrum[5], spectrum[6])
        finalists.append((key, trial, grid, spectrum, two_six))
    finalists.sort(key=lambda entry: entry[:2])
    return [(key, grid, spectrum, two_six)
            for key, _, grid, spectrum, two_six in finalists[:3]]


def _mc_exposure(checks, other, n):
    """Degree-weighted small pair boundaries on the verified full code.

    Shared opposite-basis checks indicate possible correlated-fault
    opportunities, not ASC-certified hooks or a measured failure rate.
    """
    cols = _columns(checks, n)
    mate = _columns(other, n)
    degrees = [x.bit_count() + y.bit_count()
               for x, y in zip(cols, mate)]
    burden = 0
    for i, x in enumerate(cols):
        for j in range(i):
            boundary = (x ^ cols[j]).bit_count()
            if boundary <= 3:
                shared = (mate[i] & mate[j]).bit_count()
                burden += (degrees[i] * degrees[j]
                           * (4 - boundary) ** 2 * (1 + shared))
    return burden / len(checks)


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


def _retired_load_ctx(rng, G, idx):
    """Retired order-24 experiment; never called by the new NEW_CTX."""
    import math
    from itertools import combinations

    q = G.q
    design, batch = idx % 20, idx // 20
    policy = design // 5

    # Two degree-three columns and five degree-two columns.
    # Each architecture has exactly eight terms in each B row.
    low_profiles = (
        ((2,1), (1,2), (1,1), (1,1), (1,1), (1,1), (1,1)),
        ((3,0), (0,3), (1,1), (1,1), (1,1), (1,1), (1,1)),
        ((2,1), (1,2), (2,0), (0,2), (1,1), (1,1), (1,1)),
        ((3,0), (1,2), (0,2), (1,1), (1,1), (1,1), (1,1)),
        ((3,0), (0,3), (2,0), (0,2), (1,1), (1,1), (1,1)),
    )
    # Concentrate the same eighteen B terms into two degree-four
    # columns. A's eight-term row makes these columns weight-legal.
    quartet_profiles = (
        ((3,1), (1,3), (1,1), (1,1), (1,1), (1,1), (1,1)),
        ((4,0), (0,4), (1,1), (1,1), (1,1), (1,1), (1,1)),
        ((2,2), (2,2), (2,0), (0,2), (1,1), (1,1), (1,1)),
        ((3,1), (1,3), (2,0), (0,2), (1,1), (1,1), (1,1)),
        ((4,0), (1,3), (0,2), (1,1), (1,1), (1,1), (1,1)),
    )
    book = (low_profiles if policy < 2 else
            _B_PROFILES if policy == 2 else quartet_profiles)
    profile = book[design % 5]
    triangle_orders = (
        (12, 12, 12), (12, 6, 12),
        (12, 6, 12), (12, 12, 6),
    )[policy]
    sa = [[3, 3, 3]] if policy == 1 else [[2, 3, 3]]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert q == 24
    assert tuple(map(sum, sb)) == ((8, 8) if policy < 2 else (9, 9))
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert not (sum(sa[0]) == 9 and sum(map(sum, sb)) == 18)

    orders = _mc_orders(G)
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (3, 6, 8, 12)
    }

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def cell_key(cell):
        # Right translations are gauges of a one-row A component.
        # Do not apply independent group automorphisms to its cells.
        return min(
            tuple(sorted(G.idx(G.mul(g, G.inv(origin))) for g in cell))
            for origin in cell
        )

    triangles = {length: set() for length in (3, 6, 12)}
    nonidentity = [g for g in G.elements if g != G.e]
    for u, v in combinations(nonidentity, 2):
        if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
            continue
        signature = tuple(sorted((
            orders[G.idx(u)], orders[G.idx(v)],
            orders[G.idx(G.mul(G.inv(u), v))],
        )))
        for length in triangles:
            if signature == tuple(sorted((length, 8, 8))):
                triangles[length].add(cell_key([G.e, u, v]))
    triangles = {
        length: [[G.elements[t] for t in key] for key in sorted(book)]
        for length, book in triangles.items()
    }
    assert all(triangles.values())

    def a_identity(grid):
        # Actual automorphisms: x->x^u, y->x^c*y^v,
        # u in {1,2}, c in C3 and v an odd unit modulo eight.
        # This removes obvious component orbit copies, not arbitrary
        # full-code Tanner equivalences; the evaluator checks those.
        images = []
        for u in (1, 2):
            for c in range(3):
                for v in (1, 3, 5, 7):
                    cells = [
                        cell_key([((u*g[0] + c*(g[1] % 2)) % 3,
                                   (v*g[1]) % 8) for g in cell])
                        for cell in grid[0]
                    ]
                    images.append(tuple(sorted(cells)))
        return min(images)

    strong_orders = (
        (8, 12), (6, 8, 12), (8, 12), (8, 12),
    )[policy]
    matching_orders = ((6, 8, 12), (8, 12), (8,))[batch % 3]
    matching_pool = [
        g for order in matching_orders for g in palettes[order]
    ]

    def draw_A():
        row = []
        for count, length in zip(sa[0], triangle_orders):
            if count == 2:
                # This is a real support change, not a translated
                # trinomial. Long ratios avoid a short isolated cycle.
                order = 12 if batch % 2 == 0 else 8
                row.append([G.e, rng.choice(palettes[order])])
            else:
                row.append(list(rng.choice(triangles[length])))
        return [row]

    def draw_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        # The first matching removes a common voltage gauge. Further
        # matching voltages are design variables, not saved orbit images.
        used_holonomies = [G.e]
        pair_number = 0
        pure_number = 0
        for j, column in enumerate(profile):
            if column == (1, 1):
                if pair_number == 0:
                    t = G.e
                else:
                    legal = [
                        g for g in matching_pool
                        if all(orders[G.idx(G.mul(G.inv(h), g))] > 3
                               for h in used_holonomies)
                    ]
                    if not legal:
                        legal = [
                            g for g in matching_pool
                            if G._key(g) not in {
                                G._key(h) for h in used_holonomies}
                        ]
                    if not legal:
                        raise ValueError("matching-voltage palette exhausted")
                    t = rng.choice(legal)
                    used_holonomies.append(t)
                pair_number += 1
                B[0][j], B[1][j] = [G.e], [G.inv(t)]
                continue

            for i, count in enumerate(column):
                if count == 0:
                    continue
                base = G.pick(rng)
                if count == 1:
                    cell = [base]
                elif count == 2:
                    if sum(column) == 2:
                        order = (8, 12)[(pure_number + batch + policy) % 2]
                        pure_number += 1
                    else:
                        order = rng.choice(strong_orders)
                    delta = rng.choice(palettes[order])
                    cell = [base, G.mul(base, delta)]
                elif count == 3:
                    length = (6, 12)[(i + j + policy + batch) % 2]
                    relative = rng.choice(triangles[length])
                    cell = [G.mul(base, g) for g in relative]
                else:
                    # Quartets are not doubled trinomials. Search
                    # distinct noncommuting supports, excluding
                    # involutory internal differences before lifting.
                    for _ in range(256):
                        relative = [G.e] + rng.sample(nonidentity, 3)
                        pairs = list(combinations(relative, 2))
                        if any(orders[G.idx(G.mul(G.inv(u), v))] < 3
                               for u, v in pairs):
                            continue
                        if not any(
                            G._key(G.mul(u, v)) != G._key(G.mul(v, u))
                            for u, v in pairs
                        ):
                            continue
                        break
                    else:
                        raise ValueError("noncommuting quartet unavailable")
                    cell = [G.mul(base, g) for g in relative]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return B

    def triple_boundary(cols):
        """Exact small triple boundaries, not a trapping-set certificate."""
        burden = 0
        loads = [0] * (len(cols) // q)
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    weight = (pair ^ cols[h]).bit_count()
                    if weight <= 3:
                        cost = (4 - weight) ** 2
                        burden += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[h // q] += cost
        return burden / len(cols), max(loads) / q

    def portfolio(side, draw):
        width = (3 if side == "A" else 7) * q
        target_rank = (1 if side == "A" else 2) * q
        pool, seen = [], set()
        serial = 0

        def assess(grid):
            rows = _component_rows(G, grid, side)
            if gf2_rank(rows) != target_rank:
                return None
            result = _kernel4(rows, width)
            if result is None:
                return None
            short, cols = result
            four = _four_cycles(rows)
            # This proposal energy is cheap. Exact spectra, rather than
            # this energy, determine the retained A geometries.
            energy = (500*short[0] + 50*short[1]
                      + 8*four + short[2]/16) / q
            return rows, cols, short, four, energy

        def offer(grid, state):
            nonlocal serial
            if state is None:
                return
            identity = (a_identity(grid) if side == "A" else
                        tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row))
            if identity in seen:
                return
            seen.add(identity)
            rows, cols, short, four, _ = state
            raw = None
            sparse6 = 1 << 60
            if not (short[0] or short[1]):
                if side == "A":
                    raw = _component_score(short, cols, rows, four)
                else:
                    # Exact census on the degree-two subgraph; it is
                    # valid because the whole component excludes <=4.
                    degree_two = [c for c in cols if c.bit_count() == 2]
                    _, sparse6 = _kernel56(degree_two)
            if raw is not None:
                key = raw[:4] + (four, raw[5], short[2])
            else:
                key = (short[0], short[1], sparse6, four, short[2])
            serial += 1
            pool.append((key, serial, clone(grid), state, raw, sparse6))
            pool.sort(key=lambda entry: entry[:2])
            del pool[8:]

        # Retain distinct relative structures before any local search.
        for _ in range(80):
            grid = draw()
            offer(grid, assess(grid))

        for restart in range(2):
            for _ in range(96):
                grid = draw()
                state = assess(grid)
                if state is not None:
                    break
            else:
                continue
            offer(grid, state)
            for step in range(128):
                proposal = clone(grid)
                fresh = draw()
                if side == "A":
                    j = rng.randrange(3)
                    proposal[0][j] = fresh[0][j]
                else:
                    # Mutate a complete B column so matching contracts
                    # and term-count budgets survive every proposal.
                    j = rng.randrange(7)
                    proposal[0][j] = fresh[0][j]
                    proposal[1][j] = fresh[1][j]
                proposed = assess(proposal)
                if proposed is None:
                    continue
                difference = proposed[-1] - state[-1]
                temperature = 8.0 * (0.975 ** step) + 0.08
                if difference <= 0 or rng.random() < math.exp(
                        -min(700.0, difference / temperature)):
                    grid, state = proposal, proposed
                    if step % 12 == 0:
                        offer(grid, state)
            offer(grid, state)

        if not pool:
            raise ValueError("no full-rank component in relative-support card")

        finalists = []
        for _, number, grid, state, raw, sparse6 in pool:
            rows, cols, short, four, _ = state
            if raw is None:
                raw = _component_score(short, cols, rows, four)
            boundary, peak = triple_boundary(cols)
            key = (raw[0], raw[1], raw[2], raw[3],
                   peak, boundary, four, raw[5], number)
            finalists.append((key, number, {
                "grid": grid, "spectrum": raw,
                "sparse6": sparse6, "boundary": boundary, "peak": peak,
            }))
        finalists.sort(key=lambda entry: entry[:2])
        return [entry[2] for entry in finalists[:4]]

    def pair_exposure(checks, other, n):
        """Opposite-check-weighted pair boundaries through weight five.

        Shared opposite checks mark possible correlated-fault
        opportunities, not hooks certified for an ASC gate ordering.
        """
        cols = _columns(checks, n)
        mate = _columns(other, n)
        degrees = [a.bit_count() + b.bit_count()
                   for a, b in zip(cols, mate)]
        burden = 0
        for i, x in enumerate(cols):
            for j in range(i):
                boundary = (x ^ cols[j]).bit_count()
                if boundary <= 5:
                    shared = (mate[i] & mate[j]).bit_count()
                    burden += (degrees[i] * degrees[j]
                               * (6 - boundary) ** 2 * (1 + shared))
        return burden / len(checks)

    apool = portfolio("A", draw_A)
    bpool = portfolio("B", draw_B)
    options = []
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 552 or k != 240:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            ac, bc = a["spectrum"], b["spectrum"]
            protection = (
                max(ac[0], bc[0]), ac[0] + bc[0],
                max(ac[1], bc[1]), ac[1] + bc[1],
                max(ac[2], bc[2]), ac[2] + bc[2],
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
            )
            # Replication weights are only a shortlisting surrogate,
            # not an exact census of nontrivial quantum logicals.
            witnesses = (
                max(7*ac[3], 3*bc[3]), 7*ac[3] + 3*bc[3],
                b["sparse6"],
                max(a["peak"], b["peak"]), a["peak"] + b["peak"],
                max(a["boundary"], b["boundary"]),
                a["boundary"] + b["boundary"],
            )
            options.append((protection, witnesses, len(options),
                            A, B, hx, hz, a, b))
    if not options:
        raise ValueError("no exact-rate short-ratio CSS finalist")
    options.sort(key=lambda entry: entry[:3])

    best = None
    for protection, witnesses, number, A, B, hx, hz, a, b in options[:6]:
        ex = pair_exposure(hx, hz, 552)
        ez = pair_exposure(hz, hx, 552)
        fx, fz = _four_cycles(hx), _four_cycles(hz)
        # Once short protection is matched, compare a broader fault-
        # opportunity statistic before further distance multiplicity.
        key = protection + (
            max(ex, ez), ex + ez,
        ) + witnesses + (
            max(fx / len(hx), fz / len(hz)), fx + fz, number,
        )
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, a, b, ex, ez)

    key, A, B, hx, hz, a, b, ex, ez = best
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 240
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert max(row.bit_count() for row in hx + hz) == _grid_weights(sa, sb)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    expected_cnots = 6 * q * (
        9 * sum(map(sum, sa)) + 4 * sum(map(sum, sb)))
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
    assert expected_cnots in (19584, 20880, 20736)
    assert expected_cnots < 22032
    assert 0.425 <= 240 / 552 < 0.475
    return {"A": A, "B": B, "design": design, "batch": batch,
            "load_contract": policy, "supports": (sa, sb),
            "triangle_orders": triangle_orders, "k": 240,
            "cnots": expected_cnots,
            "component_profiles": (a["spectrum"], b["spectrum"]),
            "degree_two_weight_six": b["sparse6"],
            "triple_boundary": (a["boundary"], b["boundary"]),
            "pair_exposure": (ex, ez), "ambiguity_score": key}


def _wide_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _wide_nontrivial(word, pivots):
    while word:
        p = word.bit_length() - 1
        previous = pivots.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _wide_low4(checks, stabilizers, n):
    """Exact existence of a non-stabilizer logical through weight four."""
    columns = _columns(checks, n)
    pivots = _wide_span(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
        elif _wide_nontrivial(word ^ previous, pivots):
            return True
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif _wide_nontrivial(word ^ previous, pivots):
                return True
    return False


def _wide_connected(hx, hz, n):
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        bit = row & -row
        first = bit.bit_length() - 1
        row ^= bit
        while row:
            bit = row & -row
            a, b = root(first), root(bit.bit_length() - 1)
            if a != b:
                parent[b] = a
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def _wide_probe_kernel(columns, seed, q):
    """Sample genuine component words; minima are upper bounds only.

    Rewards saturate at ten. Counts and block loads are sampled
    vulnerability statistics, not exact spectra or circuit predictions.
    """
    srng = random.Random(seed)
    permutation = list(range(len(columns)))
    minimum = len(columns) + 1
    short = set()

    def offer(word):
        nonlocal minimum
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 8:
                short.add(word)

    for trial in range(24):
        srng.shuffle(permutation)
        pivots, basis = {}, []
        for c in permutation:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome and word:
                basis.append(word)
                offer(word)
        srng.shuffle(basis)
        for i, word in enumerate(basis):
            weight = word.bit_count()
            for other in basis[:i]:
                proposal = word ^ other
                offer(proposal)
                pw = proposal.bit_count()
                if proposal and pw < weight:
                    word, weight = proposal, pw
            basis[i] = word
        if minimum <= 4:
            break

    mask = (1 << q) - 1
    loads = [
        sum(bool((word >> (j*q)) & mask) for word in short)
        for j in range(len(columns) // q)
    ]
    return min(minimum, 10), len(short), max(loads, default=0)


def _wide_bp_probe(checks, stabilizers, n, seed, shots=96):
    """Stabilizer-aware BP surrogate, NOT an ASC memory experiment.

    Handles irregular check degrees. Includes degree-weighted data
    noise and correlated pairs from opposite-basis check supports.
    Omits measurement histories, relay decoding and ASC gate ordering.
    """
    import numpy as np

    neighbors = []
    for row in checks:
        current = []
        while row:
            bit = row & -row
            current.append(bit.bit_length() - 1)
            row ^= bit
        neighbors.append(current)
    width = max(map(len, neighbors))
    nr = len(neighbors)
    incidence = np.full((nr, width), n, dtype=np.int32)
    valid = np.zeros((nr, width), dtype=bool)
    for r, row in enumerate(neighbors):
        incidence[r, :len(row)] = row
        valid[r, :len(row)] = True
    flat = incidence.ravel()
    real = valid.ravel()
    edges = len(flat)
    degree = np.bincount(flat[real], minlength=n)

    mate_columns = _columns(stabilizers, n)
    total_degree = degree + np.asarray(
        [c.bit_count() for c in mate_columns], dtype=np.int32)
    assert np.all(degree > 0)
    variable_edges = np.full(
        (n, int(degree.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge in np.flatnonzero(real):
        variable = int(flat[edge])
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.012, 0.022, 0.035))
    rates = np.minimum(
        levels[np.arange(shots) % 3, None]
        * total_degree[None, :] / 6, 0.25)
    error = channel.random((shots, n)) < rates

    # Pair faults are sampled opportunities, not ASC-certified hooks.
    hooks = []
    for row in stabilizers:
        support = []
        while row:
            bit = row & -row
            support.append(bit.bit_length() - 1)
            row ^= bit
        if len(support) >= 2:
            hooks.append(support)
    if hooks:
        for shot in np.flatnonzero(channel.random(shots) < 0.25):
            support = hooks[int(channel.integers(len(hooks)))]
            a = int(channel.integers(len(support)))
            b = int(channel.integers(len(support) - 1))
            b += b >= a
            error[shot, support[a]] ^= True
            error[shot, support[b]] ^= True

    prior_rates = np.minimum(rates + 0.5/n, 0.30)
    prior = np.log((1-prior_rates)/prior_rates)
    extended_error = np.concatenate(
        (error, np.zeros((shots, 1), dtype=bool)), axis=1)
    target = (
        extended_error[:, incidence].sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    extended_prior = np.concatenate(
        (prior, np.full((shots, 1), 40.0)), axis=1)
    messages = extended_prior[:, flat].copy()
    pivots = _wide_span(stabilizers)
    failed = iteration_sum = 0

    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, width)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logarithms = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
        logarithms[:, ~valid] = 0.0
        signs[:, ~valid] = 1.0
        products = np.minimum(np.exp(
            logarithms.sum(axis=2, keepdims=True)-logarithms),
            1-1e-7)
        magnitude = np.log1p(products)-np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None])
        outgoing[:, ~valid] = 0.0
        outgoing = outgoing.reshape(count, edges)

        padded = np.zeros((count, edges+1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior + padded[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        extended_hard = np.concatenate(
            (hard, np.zeros((count, 1), dtype=bool)), axis=1)
        syndrome = (
            extended_hard[:, incidence].sum(axis=2) & 1
        ).astype(bool)
        settled = np.all(syndrome == target, axis=1)

        for shot in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[shot] ^ hard[shot]):
                word |= 1 << int(c)
            failed += int(_wide_nontrivial(word, pivots))
            iteration_sum += iteration

        keep = ~settled
        if not np.any(keep):
            return failed, 0, iteration_sum

        extended_posterior = np.concatenate(
            (posterior, np.full((count, 1), 40.0)), axis=1)
        messages = np.clip(
            0.35*messages
            + 0.65*(extended_posterior[:, flat]-outgoing),
            -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failed+unresolved, unresolved, iteration_sum+33*unresolved


def NEW_CTX(rng, G, idx):
    """Twenty order/support contracts on the new order-64 bed."""
    from itertools import combinations

    assert G.q == 64
    assert A_SHAPE == (1, 2) and B_SHAPE == (1, 4)
    q = G.q
    design, batch = idx % 20, idx // 20
    architecture, mode = divmod(design, 4)
    contracts = (
        ((3, 3), (2, 2, 2, 3)),
        ((2, 3), (2, 2, 2, 3)),
        ((2, 2), (2, 2, 3, 3)),
        ((2, 2), (2, 2, 2, 3)),
        ((2, 2), (2, 2, 2, 4)),
    )
    acounts, bcounts = contracts[architecture]
    sa, sb = [list(acounts)], [list(bcounts)]
    assert _no_dead_lines(sa, sb)
    assert sum(acounts) + max(bcounts) <= min(12, W_MAX)
    assert sum(bcounts) + max(acounts) <= min(12, W_MAX)
    assert _grid_weights(sa, sb) <= W_MAX

    orders = _mc_orders(G)
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (8, 16)
    }
    assert all(palettes.values())
    nonidentity = [g for g in G.elements if g != G.e]

    def ratio_order(u, v):
        return orders[G.idx(G.mul(G.inv(u), v))]

    triangles = []
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        signature = (
            orders[G.idx(u)], orders[G.idx(v)], ratio_order(u, v))
        if min(signature) < (8 if mode == 0 else 4):
            continue
        span = len({0, u[1], v[1]})
        if mode == 2 and span != 3:
            continue
        if mode == 3 and span != 2:
            continue
        triangles.append([G.e, u, v])
    assert triangles

    # Only helper-derived singleton actions are composed; no independent
    # implementation of left/right regular representations is used.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for side in ("A", "B"):
        for g in G.elements:
            rows = _component_rows(G, [[[g]]], side)
            actions[side][G.idx(g)] = tuple(row & mask for row in rows)

    def component(side, grid):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    def draw(side, counts):
        cells = []
        for j, count in enumerate(counts):
            if count == 2:
                if mode == 0:
                    order = 16
                elif mode == 1:
                    order = 8 if side == "B" and (j+batch) % 2 else 16
                elif mode == 2:
                    order = 16 if (j+batch) % 2 else 8
                else:
                    order = 8
                cell = [G.e, rng.choice(palettes[order])]
            elif count == 3:
                cell = list(rng.choice(triangles))
            else:
                for attempt in range(128):
                    cell = [G.e] + rng.sample(nonidentity, 3)
                    if any(ratio_order(u, v) < 4
                           for u, v in combinations(cell, 2)):
                        continue
                    if not any(G.mul(u, v) != G.mul(v, u)
                               for u, v in combinations(cell, 2)):
                        continue
                    break
                else:
                    raise ValueError("quartet contract exhausted")
            assert len(cell) == count
            assert len(set(cell)) == count
            cells.append(cell)
        return [cells]

    def select(side, counts):
        width = len(counts)*q
        pool, seen = [], set()
        for trial in range(256):
            grid = draw(side, counts)
            identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for cell in grid[0])
            if identity in seen:
                continue
            seen.add(identity)
            rows = component(side, grid)
            deficit = q-gf2_rank(rows)
            if deficit > 4:
                continue
            result = _kernel4(rows, width)
            if result is None:
                continue
            short, columns = result
            if short[0] or short[1]:
                continue
            four = _four_cycles(rows)
            key = (four, short[2], deficit)
            pool.append((key, trial, grid, rows, columns))
            pool.sort(key=lambda entry: entry[:2])
            del pool[6:]

        if not pool:
            raise ValueError("no long-ratio component excludes kernels <=4")

        entries = []
        # Identical probe streams across component alternatives.
        seed = 640051 + 1009*idx + int(side == "B")
        for coarse, trial, grid, rows, columns in pool:
            minimum, count, peak = _wide_probe_kernel(columns, seed, q)
            if minimum <= 4:
                continue
            key = (
                int(minimum < 6), 10-minimum,
                peak, count, coarse[0], coarse[1], coarse[2], trial)
            # Check the cached linear composition against a fresh lift.
            actual = _component_rows(G, grid, side)
            cmask = (1 << width)-1
            assert rows == [row & cmask for row in actual]
            entries.append((key, grid))
        entries.sort(key=lambda entry: entry[0])
        return entries[:3]

    apool = select("A", acounts)
    bpool = select("B", bcounts)
    proposals = []
    for ia, (akey, A) in enumerate(apool):
        for ib, (bkey, B) in enumerate(bpool):
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 576 or not 192 <= k <= 210:
                continue
            if not _wide_connected(hx, hz, n):
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified group lift failed CSS")
            geometry = (
                max(akey[0], bkey[0]), akey[0]+bkey[0],
                max(akey[1], bkey[1]), akey[1]+bkey[1],
                max(akey[2], bkey[2]), akey[2]+bkey[2],
                akey[3]+bkey[3], akey[4]+bkey[4])
            proposals.append((geometry, ia, ib, A, B, hx, hz, k))
    proposals.sort(key=lambda entry: entry[:3])

    finalists = []
    seed = 640012004 + 7919*idx
    for geometry, ia, ib, A, B, hx, hz, k in proposals[:4]:
        if _wide_low4(hx, hz, 576) or _wide_low4(hz, hx, 576):
            continue
        x = _wide_bp_probe(hz, hx, 576, seed)
        z = _wide_bp_probe(hx, hz, 576, seed+1)
        # Failed/unsatisfied probes precede sampled-distance multiplicity.
        # This ordering is a hypothesis about pL, not a calibrated model.
        key = (
            max(x[0], z[0]), x[0]+z[0],
            max(x[1], z[1]), x[1]+z[1],
            x[2]+z[2]) + geometry + (ia, ib)
        finalists.append((key, A, B, hx, hz, k, x, z))
    if not finalists:
        raise ValueError("no connected rate-0.350 joint finalist")
    finalists.sort(key=lambda entry: entry[0])
    key, A, B, hx, hz, k, x, z = finalists[0]

    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert lifted_k(G, A, B) == k
    assert max(row.bit_count() for row in hx+hz) == _grid_weights(sa, sb)
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    cnots = 6*q*(5*sum(acounts)+3*sum(bcounts))
    assert 6*sum(row.bit_count() for row in hx+hz) == cnots
    assert cnots in (21888, 19968, 19200, 18048)
    assert 0.325 <= k/576 < 0.375
    return {"A": A, "B": B, "idx": idx, "k": k,
            "contract": (architecture, mode, batch),
            "supports": (sa, sb), "cnots": cnots,
            "bp_surrogate": (x, z), "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 failed 110/1250 Z and 121/1250 X shots, and all five measured codes retained abundant sector-one weight-six witnesses.
# REFUTED: Long-order trinomial ratios and degree-two cycle screening suffice for rare collective failures on C3:C8 (1,3)x(2,7): fitness was 4.0835, all five measured codes reached failed-shot caps, and abundant weight-six witnesses remained across sector one while failures collectively affected all 240 logicals.
# REFUTED: Adding order-three trinomial ratios and broader pair-boundary screening suffices for rare collective failures at unchanged extraction load: fitness was 4.1155, all five measured codes reached failed-shot caps, Z failed 105-137/1250-1500 shots, and weight-six sector-one witnesses remained abundant.
# REFUTED: Reducing extraction load with rank-preserving support reallocations improved this C3:C8 parent: measured fitness instead declined from 4.1155 to 3.8841; every measured code reached failed-shot caps, Z failed 109-138/750-1000 shots, and abundant sector-one weight-five/six witnesses remained.
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
