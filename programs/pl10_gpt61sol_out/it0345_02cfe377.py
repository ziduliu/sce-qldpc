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
CONCEPT = ("On C3 semidirect C8 with inversion action, order 24, A=(1,3), B=(2,7), A counts (2,3,3) and sixteen-term B components should improve worse-basis circuit fitness beyond 4.1250 by reducing extraction work while screening verified short-word protection.")
GROUP = ("Frozen L1: retain exactly C3:C8(s=2), order 24, its encoding and multiplication. This bed scored 4.0835-4.1250, but measured codes still reached failed-shot caps. These readings motivate a filling-only gate-exposure experiment and establish no group advantage.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=552, kb=240 and guaranteed rate 10/23=0.43478. Size factor 23 permits q<=26; the fixed group uses q=24. Unequal components have 72/168 variables and 24/48 checks. Require verified k=240. Parent sector-one witnesses covered many Acol/Bcol blocks; enrichment was diffuse at 1.0-1.2x, without a persistent Brow hotspot or one consistently worst logical qubit.")
FILLING = ("Twenty contracts cross four relative-order briefs with five B architectures. Primary A counts are (2,3,3); B row totals are (8,8), with two degree-three and five degree-two columns. Both check-weight sums are <=11 and six-round CNOTs fall from 22032 to 19584. Extra batches deliberately test A=(2,2,3), giving 18288 CNOTs, or transfer one B term between the strengthened columns, introducing a degree-four column without increasing total gates. Fresh verified component portfolios prefer exclusion through weight four, then compare exact spectra when applicable. Complete lifts verify k and CSS. Common-stream six-round noisy-syndrome min-sum probes and independent holdouts select finalists before multiplicity tie-breaks. Target the parent's 102-120 Z failures per 1000-1500 shots, widespread collective flips across all 240 logicals, and late settlement. Probes omit ASC ordering and relay/LSD and are not measured circuit fitness.")
FAMILY_NAME = "r045_C3byC8_q24_13x27_twoheavy_gatelean_historybp_v4"

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


def _inactive_longratio_ctx(rng, G, idx):
    """Historical eighteen-term B search; not called by the new builder."""
    from itertools import combinations

    q = G.q
    design, batch = idx % 20, idx // 20
    policy = design // 5
    profile = _B_PROFILES[design % 5]
    architectures = (
        (12, 12, 12),
        (12, 12, 6),
        (12, 6, 6),
        (6, 6, 6),
    )
    triangle_orders = architectures[policy]
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert q == 24
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    assert sum(sa[0]) + max(map(sum, profile)) == 12
    assert max(map(sum, sb)) + max(sa[0]) == 12

    orders = _mc_orders(G)
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (6, 8, 12)
    }
    nonidentity = [g for g in G.elements if g != G.e]
    triangles = {(length, model): []
                 for length in (6, 12) for model in (0, 1)}
    for u, v in combinations(nonidentity, 2):
        relative = G.mul(G.inv(u), v)
        signature = sorted((orders[G.idx(u)], orders[G.idx(v)],
                            orders[G.idx(relative)]))
        if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
            continue
        # Model 0: one odd-y and one even-y nonidentity term.
        # Model 1: two odd-y terms. The triangle-order spectrum
        # remains prescribed; these are different search contracts.
        model = int(u[1] % 2 == 1 and v[1] % 2 == 1)
        for length in (6, 12):
            if signature == sorted((8, 8, length)):
                triangles[length, model].append([G.e, u, v])
    assert all(triangles.values())
    assert all(palettes.values())

    def draw_A():
        return [[list(rng.choice(
                    triangles[length, (j + policy + batch) % 2]))
                 for j, length in enumerate(triangle_orders)]]

    def draw_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        # For a singleton-pair column, t=b0*b1^-1 is the actual
        # right-component connection. Normalize b0=e, then prescribe
        # noncommuting long-order connections rather than reflections.
        long_order = 6 if batch % 3 == 1 else 12
        u = rng.choice(palettes[8])
        v = rng.choice(palettes[long_order])
        assert G._key(G.mul(u, v)) != G._key(G.mul(v, u))
        holonomies = (G.e, u, v)
        single_pair = 0
        for j, column in enumerate(profile):
            if column == (1, 1):
                t = holonomies[single_pair]
                single_pair += 1
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
                        # A pure doubleton supplies a component cycle.
                        # Never give it a ratio of order two, four or six.
                        order = 12
                    elif policy == 0:
                        order = 12
                    elif policy == 1:
                        order = 8 if (i + j + batch) % 2 else 12
                    elif policy == 2:
                        order = 8
                    else:
                        order = 6 if (i + j + batch) % 2 else 8
                    delta = rng.choice(palettes[order])
                    cell = [base, G.mul(base, delta)]
                else:
                    length = triangle_orders[(i + j + batch) % 3]
                    model = (i + j + policy + batch) % 2
                    relative = rng.choice(triangles[length, model])
                    cell = [G.mul(base, g) for g in relative]
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return B

    apool = _mc_pool(rng, G, "A", draw_A, 224)
    bpool = _mc_pool(rng, G, "B", draw_B, 384)

    options = []
    for akey, A, ascore, a_two6 in apool:
        for bkey, B, bscore, b_two6 in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 552 or k != 240:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            preliminary = (
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
                max(ascore[0], bscore[0]), ascore[0] + bscore[0],
                max(ascore[1], bscore[1]), ascore[1] + bscore[1],
                int(bool(ascore[2] or bscore[2])),
                b_two6,
                int(bool(ascore[3] or bscore[3])),
            )
            options.append((preliminary, A, B, hx, hz,
                            ascore, bscore, b_two6))
    if not options:
        raise ValueError("no exact-rate long-ratio CSS candidate")

    best = None
    for preliminary, A, B, hx, hz, ascore, bscore, b_two6 in options:
        ex = _mc_exposure(hx, hz, 552)
        ez = _mc_exposure(hz, hx, 552)
        fx, fz = _four_cycles(hx), _four_cycles(hz)
        # Stop rewarding distance thresholds at exclusion through six;
        # then compare a circuit-exposure surrogate before multiplicity.
        key = preliminary + (
            max(ex, ez), ex + ez,
            max(fx / len(hx), fz / len(hz)), fx + fz,
            max(ascore[2], bscore[2]), ascore[2] + bscore[2],
            max(ascore[3], bscore[3]), ascore[3] + bscore[3],
            ascore[5] + bscore[5],
        )
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, ascore, bscore, b_two6)

    key, A, B, hx, hz, ascore, bscore, b_two6 = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 22032
    assert 0.425 <= 240 / 552 < 0.475
    return {"A": A, "B": B, "design": design, "batch": batch,
            "triangle_orders": triangle_orders, "k": 240,
            "component_profiles": (ascore, bscore),
            "degree_two_weight_six": b_two6,
            "ambiguity_score": key}


def NEW_CTX(rng, G, idx):
    """Gate-lean support contracts, screened with evaluator-convention lifts."""
    import math
    import hashlib
    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 24
    design, batch = idx % 20, idx // 20
    architecture, policy = divmod(design, 4)

    # Each architecture has sixteen terms and row totals (8,8).
    # Columns have degrees (3,3,2,2,2,2,2), but their internal versus
    # cross-row connections are deliberate structural alternatives.
    profiles = (
        ((2,1), (1,2), (1,1), (1,1), (1,1), (1,1), (1,1)),
        ((3,0), (0,3), (1,1), (1,1), (1,1), (1,1), (1,1)),
        ((3,0), (1,2), (0,2), (1,1), (1,1), (1,1), (1,1)),
        ((2,1), (1,2), (2,0), (0,2), (1,1), (1,1), (1,1)),
        ((3,0), (0,3), (2,0), (2,0), (0,2), (0,2), (1,1)),
    )
    profile = [list(column) for column in profiles[architecture]]
    extension = batch % 3
    if extension == 2:
        # Preserve row totals, but change the column-degree distribution
        # to (4,2,2,2,2,2,2). This is not a labeling orbit.
        source_rows = [i for i in range(2) if profile[1][i]]
        i = source_rows[(policy + batch) % len(source_rows)]
        profile[0][i] += 1
        profile[1][i] -= 1

    sa = [[2, 2, 3]] if extension == 1 else [[2, 3, 3]]
    sb = [[column[i] for column in profile] for i in range(2)]
    briefs = (
        (8, 3, 6), (12, 3, 12),
        (8, 6, 12), (12, 12, 12),
    )
    bin_order, first_triangle, last_triangle = briefs[policy]
    assert tuple(map(sum, sb)) == (8, 8)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert sum(sa[0]) + max(map(sum, profile)) <= 12
    assert max(map(sum, sb)) + max(sa[0]) <= 12

    orders = _mc_orders(G)
    palettes = {
        order: [g for g in G.elements if orders[G.idx(g)] == order]
        for order in (3, 6, 8, 12)
    }
    nonidentity = [g for g in G.elements if g != G.e]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def cell_key(cell):
        # Quotient right cell translations, not independent automorphisms.
        return min(
            tuple(sorted(G.idx(G.mul(g, G.inv(origin))) for g in cell))
            for origin in cell
        )

    triangles = {length: set() for length in (3, 6, 12)}
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
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
        images = []
        # Actual automorphisms x->x^u, y->x^c*y^v.
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

    def draw_A():
        if extension == 1:
            # Noncommuting order-eight/twelve directions avoid importing
            # the obvious parallel-edge and commuting-square mechanisms.
            for _ in range(64):
                u = rng.choice(palettes[8])
                v = rng.choice(palettes[12])
                if G.mul(u, v) != G.mul(v, u):
                    return [[[G.e, u], [G.e, v],
                             list(rng.choice(triangles[last_triangle]))]]
            raise ValueError("noncommuting binomial brief not realized")
        return [[[G.e, rng.choice(palettes[bin_order])],
                 list(rng.choice(triangles[first_triangle])),
                 list(rng.choice(triangles[last_triangle]))]]

    def draw_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        holonomies = []
        ordered_differences = set()
        for j, column in enumerate(profile):
            if column == [1, 1]:
                if not holonomies:
                    t = G.e
                else:
                    choices = []
                    for g in G.elements:
                        if g in holonomies:
                            continue
                        differences = [
                            G.idx(G.mul(G.inv(h), g))
                            for h in holonomies
                        ] + [
                            G.idx(G.mul(G.inv(g), h))
                            for h in holonomies
                        ]
                        # A Sidon preference on the matching connections,
                        # not a substitute for the verified kernel test.
                        collisions = (
                            len(differences) - len(set(differences))
                            + sum(d in ordered_differences
                                  for d in differences)
                        )
                        short_ratios = sum(
                            orders[d] <= 3 for d in differences)
                        choices.append(((collisions, short_ratios), g))
                    floor = min(key for key, _ in choices)
                    t = rng.choice([g for key, g in choices if key == floor])
                for h in holonomies:
                    ordered_differences.add(
                        G.idx(G.mul(G.inv(h), t)))
                    ordered_differences.add(
                        G.idx(G.mul(G.inv(t), h)))
                holonomies.append(t)
                B[0][j], B[1][j] = [G.e], [G.inv(t)]
                continue

            for i, count in enumerate(column):
                if not count:
                    continue
                base = G.pick(rng)
                if count == 1:
                    relative = [G.e]
                elif count == 2:
                    if sum(column) == 2:
                        order = 12
                    else:
                        order = (6, 8, 12)[
                            (i + j + policy + batch) % 3]
                    relative = [G.e, rng.choice(palettes[order])]
                elif count == 3:
                    length = (3, 6, 12)[
                        (i + j + policy + batch) % 3]
                    relative = rng.choice(triangles[length])
                else:
                    # The term transfer can only increase a cell to four.
                    relative = rng.sample(G.elements, count)
                cell = [G.mul(base, g) for g in relative]
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return B

    def fingerprint(rows, width):
        """Conservative rooted component invariant, not canonization."""
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, syndrome in enumerate(_columns(rows, width)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                syndrome ^= bit
        keys = []
        for root in range(0, nr, q):
            colors = [0]*nr + [1]*width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(8):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr + width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            keys.append(digest.hexdigest())
        return tuple(sorted(keys))

    def portfolio(side, draw):
        width = (3 if side == "A" else 7)*q
        target_rank = (1 if side == "A" else 2)*q
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
            energy = (500*short[0] + 60*short[1]
                      + 4*four + short[2]/16)/q
            return rows, cols, short, four, energy

        def remember(grid, state):
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
            serial += 1
            key = (short[0], short[1], four, short[2])
            pool.append((key, serial, clone(grid), state))
            pool.sort(key=lambda entry: entry[:2])
            del pool[10:]

        for _ in range(128 if side == "A" else 256):
            grid = draw()
            remember(grid, assess(grid))

        for restart in range(2):
            for _ in range(128):
                grid = draw()
                state = assess(grid)
                if state is not None:
                    break
            else:
                continue
            remember(grid, state)
            for step in range(128):
                fresh, proposal = draw(), clone(grid)
                j = rng.randrange(3 if side == "A" else 7)
                for i in range(len(proposal)):
                    proposal[i][j] = fresh[i][j]
                trial = assess(proposal)
                if trial is None:
                    continue
                delta = trial[-1] - state[-1]
                temperature = 10*(0.975**step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    grid, state = proposal, trial
                    if step % 16 == 0:
                        remember(grid, state)
            remember(grid, state)

        finalists, fingerprints = [], set()
        for coarse, number, grid, state in pool:
            rows, cols, short, four, _ = state
            identity = fingerprint(rows, width)
            if identity in fingerprints:
                continue
            fingerprints.add(identity)
            spectrum = _component_score(short, cols, rows, four)
            key = spectrum[:4] + (
                four, spectrum[5], short[2], number)
            finalists.append((key, grid, spectrum))
        finalists.sort(key=lambda entry: entry[0])
        if not finalists:
            raise ValueError("no rank-safe component in gate-lean contract")
        return finalists[:4]

    apool, bpool = portfolio("A", draw_A), portfolio("B", draw_B)
    options = []
    for _, A, ac in apool:
        for _, B, bc in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 552 or k != 240:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            protection = (
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
                max(ac[0], bc[0]), ac[0] + bc[0],
                max(ac[1], bc[1]), ac[1] + bc[1],
            )
            # Component counts are not a quantum-logical census.
            structural = (
                max(7*ac[2], 3*bc[2]), 7*ac[2] + 3*bc[2],
                max(7*ac[3], 3*bc[3]), 7*ac[3] + 3*bc[3],
                ac[4] + bc[4], ac[5] + bc[5],
            )
            options.append((protection, structural, len(options),
                            A, B, hx, hz, ac, bc))
    if not options:
        raise ValueError("no verified k=240 gate-lean finalist")
    options.sort(key=lambda entry: entry[:3])

    def history_probe(checks, stabilizers, seed, shots):
        """Six-round noisy-syndrome min-sum surrogate, NOT ASC fitness."""
        import numpy as np

        n, m, rounds = 552, len(checks), 6

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length() - 1)
                row ^= bit
            return out

        neighbors = [sites(row) for row in checks]
        opposite = [sites(row) for row in stabilizers]
        variables = rounds*(n + m)
        dummy = variables
        detectors = (rounds + 1)*m
        row_width = max(map(len, neighbors)) + 2
        incidence = np.full((detectors, row_width), dummy,
                            dtype=np.int32)
        for t in range(rounds + 1):
            for r, row in enumerate(neighbors):
                position = 0
                if t < rounds:
                    incidence[t*m+r, :len(row)] = [
                        t*n+c for c in row]
                    position = len(row)
                    incidence[t*m+r, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = (
                        rounds*n+(t-1)*m+r)

        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (variables + 1, int(degree.max())), edges,
            dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        gate_degree = np.asarray([
            a.bit_count() + b.bit_count()
            for a, b in zip(_columns(checks, n),
                            _columns(stabilizers, n))
        ], dtype=float)
        # Approximate aggregate gate exposure, not ASC fault probabilities.
        data_rate = np.minimum(0.002*(2 + 2*gate_degree), 0.12)
        measure_rate = np.asarray([
            0.002*(2 + len(row))*2/3 for row in neighbors])
        base = np.concatenate((np.tile(data_rate, rounds),
                               np.tile(measure_rate, rounds)))
        levels = np.asarray((0.65, 0.90, 1.15))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None]*base[None, :], 0.25)
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables + 1), dtype=bool)
        error[:, :variables] = (
            channel.random((shots, variables)) < rates)

        # Approximate shared-ancilla pairs; these are not certified hooks.
        for t in range(rounds):
            chosen = channel.integers(len(opposite), size=shots)
            active = channel.random(shots) < 0.35
            for shot in np.flatnonzero(active):
                row = opposite[int(chosen[shot])]
                u = int(channel.integers(len(row)))
                v = int(channel.integers(len(row) - 1))
                v += v >= u
                error[shot, t*n+row[u]] ^= True
                error[shot, t*n+row[v]] ^= True

        target = (error[:, flat].reshape(
            shots, detectors, row_width).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.empty((shots, variables + 1), dtype=float)
        prior[:, :variables] = np.log((1 - rates)/rates)
        prior[:, dummy] = 30.0
        messages = prior[:, flat].copy()
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        mask = valid.reshape(detectors, row_width)
        positions = np.arange(row_width)[None, None, :]
        failures = effort = 0
        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            magnitude = np.where(mask[None, :, :],
                                 np.abs(shaped), np.inf)
            minimum_at = np.argmin(magnitude, axis=2)
            smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
            size = np.where(
                positions == minimum_at[:, :, None],
                smallest[:, :, 1, None], smallest[:, :, 0, None])
            outgoing = np.clip(
                0.80*size*signs*signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None], -24, 24
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges + 1), dtype=float)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, detectors, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(
                        decoded_final[shot] ^ true_final[shot]):
                    word |= 1 << int(c)
                failures += int(nontrivial(word))
                effort += iteration

            keep = ~settled
            if not np.any(keep):
                return failures, 0, effort
            messages = np.clip(
                0.30*messages
                + 0.70*(posterior[:, flat] - outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        # Relay/LSD could rescue unresolved cases in the real experiment.
        return failures + unresolved, unresolved, effort + 33*unresolved

    def quality(x, z):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                max(x[2], z[2]), x[2] + z[2])

    # Common streams compare candidates within a slot; holdouts are fresh.
    seed = 241958403 + 1009*idx
    coarse = []
    floor = options[0][0]
    eligible = [entry for entry in options if entry[0] == floor][:3]
    for option in eligible:
        protection, structural, number, A, B, hx, hz, ac, bc = option
        x = history_probe(hz, hx, seed, 12)
        z = history_probe(hx, hz, seed + 1, 12)
        key = protection + quality(x, z) + structural + (number,)
        coarse.append((key, number, option, x, z))
    coarse.sort(key=lambda entry: entry[:2])

    best = None
    for _, number, option, cx, cz in coarse[:2]:
        protection, structural, _, A, B, hx, hz, ac, bc = option
        x = history_probe(hz, hx, seed + 1000003, 36)
        z = history_probe(hx, hz, seed + 1000004, 36)
        total_x = tuple(a + b for a, b in zip(cx, x))
        total_z = tuple(a + b for a, b in zip(cz, z))
        key = protection + quality(total_x, total_z) + structural + (
            _four_cycles(hx)/len(hx) + _four_cycles(hz)/len(hz),
            number,
        )
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, ac, bc, total_x, total_z)

    key, A, B, hx, hz, ac, bc, total_x, total_z = best
    actual_sa = [[len(cell) for cell in row] for row in A]
    actual_sb = [[len(cell) for cell in row] for row in B]
    cnots = 6*q*(9*sum(map(sum, actual_sa))
                 + 4*sum(map(sum, actual_sb)))
    assert actual_sa == sa and actual_sb == sb
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 240
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert 6*sum(row.bit_count() for row in hx + hz) == cnots
    assert cnots == (18288 if extension == 1 else 19584)
    assert 0.425 <= 240/552 < 0.475
    return {"A": A, "B": B, "design": (architecture, policy, batch),
            "k": 240, "cnots": cnots,
            "component_profiles": (ac, bc),
            "history_min_sum_surrogate": (total_x, total_z),
            "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected gate-lean full cell without subsequent draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 failed 110/1250 Z and 121/1250 X shots, and all five measured codes retained abundant sector-one weight-six witnesses.
# REFUTED: Long-order trinomial ratios and degree-two cycle screening suffice for rare collective failures on C3:C8 (1,3)x(2,7): fitness was 4.0835, all five measured codes reached failed-shot caps, and abundant sector-one weight-six witnesses remained while failures collectively affected all 240 logicals.
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
