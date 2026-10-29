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
CONCEPT = ("On C3 semidirect C8 with inversion action, order 24, shapes A=(1,3), B=(2,7), reducing extraction to 19584-20880 CNOTs while screening short component kernels should improve worse-basis circuit fitness above 4.1155; the experiment tests extraction load rather than distance alone.")
GROUP = ("Frozen for L1: retain exactly C3:C8(s=2), order 24 and its encoding. The current bed scored 4.1155, but every measured code reached failed-shot caps and failures collectively affected all 240 logicals. Keep the group to isolate a support-and-labeling experiment.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=552, kb=240 and guaranteed rate 10/23=0.43478; the size window permits q<=26 and this group uses q=24. Unequal components have 72 variables/24 checks and 168 variables/48 checks. Require verified k=240. Parent weight-six witnesses were spread across sector-one Acol/Bcol blocks; failure enrichment was diffuse at 1.0-1.2x without a persistent Brow hotspot.")
FILLING = ("Twenty contracts cross four load allocations with five B support architectures: A=(2,3,3), B rows eight; A=(3,3,3), B rows eight; A=(2,3,3), B rows nine with degree-three columns; and A=(2,3,3), B rows nine with two degree-four columns. Maximum CSS weights are eleven or twelve, with 19584, 20880 or 20736 six-round CNOTs instead of 22032. Long-order A binomials replace one trinomial where prescribed; strengthened cells use noncommuting trinomials or distinct quartets without involutory internal differences. Verified component spectra and full-code pair boundaries select labels, while exact ranks retain k=240 and CSS. Target the parent's 105-137 Z failures in 1250-1500 shots, collective flips across all logicals, and late decoder settlement; none of these structural screens is a calibrated circuit-pL estimate.")
FAMILY_NAME = "r045_C3byC8_q24_13x27_load_reallocation_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C3 semidirect C8 requires order 24")
    # x^3=y^8=1, y*x*y^-1=x^-1; canonical elements x^a*y^b.
    # The inversion action has order two, which divides eight.
    els = [(a, b) for b in range(8) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 8)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b % 2 == 0 else -c)) % 3,
                (b + d) % 8)

    def inv(g):
        a, b = red(g)
        return ((-a if b % 2 == 0 else a) % 3, (-b) % 8)

    def idx(g):
        a, b = red(g)
        return a + 3*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
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


def NEW_CTX(rng, G, idx):
    """Twenty load/support contracts; screen only scaffold-built matrices."""
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


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 failed 110/1250 Z and 121/1250 X shots, and all five measured codes retained abundant sector-one weight-six witnesses.
# REFUTED: Long-order trinomial ratios and degree-two cycle screening suffice for rare collective failures on C3:C8 (1,3)x(2,7): fitness was 4.0835, all five measured codes reached failed-shot caps, and abundant weight-six witnesses remained across sector one while failures collectively affected all 240 logicals.
# REFUTED: Adding order-three trinomial ratios and broader pair-boundary screening suffices for rare collective failures at unchanged extraction load: fitness was 4.1155, all five measured codes reached failed-shot caps, Z failed 105-137/1250-1500 shots, and weight-six sector-one witnesses remained abundant.
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
