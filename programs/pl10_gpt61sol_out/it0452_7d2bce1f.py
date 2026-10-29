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
CONCEPT = ("On Dic6, order 24, A=(1,3), B=(2,7), an order-twelve binomial plus two mixed-coset trinomials and lower-load B count contracts should improve worse-basis ASC circuit fitness above 4.0112; measured circuit pL decides.")
GROUP = ("Frozen L1: retain exactly Dic6, order 24, its presentation and encoding. The current program scored 4.0112 but all five measured codes hit failed-shot caps in both bases. This motivates changing filling rather than claiming a group advantage. Order-twelve rotations supply binomials without an unavoidable weight-four component cycle.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=552, kb=240 and guaranteed rate 10/23=0.43478; require complete helper-built k=240. The size factor 23 permits q<=26, and this group uses q=24. Unequal components have 72 variables/24 checks and 168 variables/48 checks. Parent sector-one weight-six witnesses span many Acol/Bcol blocks; enrichment is diffuse at 1.0-1.2x and no persistent Acol/Brow hotspot supports isolated reinforcement.")
FILLING = ("Twenty deliberate cards consume distinct canonical B count architectures, interleaving degree profiles (3,3,2,2,2,2,2), (3,3,3,2,2,2,2), (4,3,2,2,2,2,2) and (4,3,3,2,2,2,2). A counts are (2,3,3); B row totals are (8,8), (8,9) or (9,9). Further harness slots consume further count cards, not labeling orbits. Fresh relative supports use order-twelve pure binomials, mixed-coset trinomials and new mixed-coset quartets. Verified component spectra shortlist alternatives; complete CSS lifts require exact k and compare noisy-intermediate-syndrome BP training probes followed by independent holdouts. Six-round CNOTs fall from 22032 to 19584/20160/20736 and maximum row weight is eleven or twelve. Target the parent's 102-125 Z and 103-118 X failures in 750-1500 shots, collective flips on all 240 logicals and late settlement. Two-interval probes include approximate spread faults but omit ASC ordering, the six-round history and relay/LSD; they are not measured pL.")
FAMILY_NAME = "r045_Dic6_q24_13x27_lowload_countcatalogue_twotime_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic6 requires order 24")
    # x^12=1, y^2=x^6, y*x*y^-1=x^-1.
    # Canonical elements are x^a*y^b, b in {0,1}.
    els = [(a, b) for b in range(2) for a in range(12)]

    def red(g):
        return (int(g[0]) % 12, int(g[1]) % 2)

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
        return a + 12*b

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


def _dic_delta(rng, order):
    """Rotation displacements with exactly the prescribed order."""
    palettes = {12: (1, 5, 7, 11), 6: (2, 10), 4: (3, 9)}
    return rng.choice(palettes[order])


def _dic_triple(rng, G, order, model):
    """Normalized mixed-coset trinomial; no cell-translation search."""
    d = _dic_delta(rng, order)
    a = rng.randrange(12)
    if model % 2 == 0:
        cell = [G.e, (d, 0), (a, 1)]
    else:
        cell = [G.e, (a, 1), ((a + d) % 12, 1)]
    assert len({G._key(g) for g in cell}) == 3
    return cell


def _dic_component_pool(rng, G, side, draw, trials):
    """Optimize a structural brief on the scaffold's exact component.

    Component spectra remove redundant finalists, not arbitrary
    full-code Tanner isomorphisms. The evaluator remains authoritative.
    """
    width = (3 if side == "A" else 7) * G.q
    expected_rank = (1 if side == "A" else 2) * G.q
    pool = []
    exact_seen = set()
    for trial in range(trials):
        grid = draw()
        identity = tuple(tuple(tuple(sorted(G.idx(g) for g in cell))
                               for cell in row) for row in grid)
        if identity in exact_seen:
            continue
        exact_seen.add(identity)
        rows = _component_rows(G, grid, side)
        if gf2_rank(rows) != expected_rank:
            continue
        result = _kernel4(rows, width)
        if result is None:
            continue
        short, cols = result
        four = _four_cycles(rows)
        coarse = (short[0], short[1], short[2], four)
        pool.append((coarse, trial, grid, rows, cols, short, four))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[10:]

    if not pool:
        raise ValueError("no full-rank distinct-syndrome Dic6 component")

    catalogue = {}
    for _, trial, grid, rows, cols, short, four in pool:
        score = _component_score(short, cols, rows, four)
        old = catalogue.get(score)
        if old is None or trial < old[0]:
            catalogue[score] = (trial, grid)
    ordered = sorted((score, value[0], value[1])
                     for score, value in catalogue.items())
    return [(score, grid) for score, _, grid in ordered[:4]]


def _dic_boundary(checks, other, n):
    """Degree-weighted small pair boundaries and co-check opportunities.

    This is a trapping/hook-opportunity heuristic, not an ASC fault
    certificate, logical-distance certificate or measured failure rate.
    """
    cols = _columns(checks, n)
    mate = _columns(other, n)
    degree = [x.bit_count() + y.bit_count()
              for x, y in zip(cols, mate)]
    exposure = 0
    for i, x in enumerate(cols):
        for j in range(i):
            boundary = (x ^ cols[j]).bit_count()
            if boundary <= 3:
                shared = (mate[i] & mate[j]).bit_count()
                exposure += (degree[i] * degree[j] *
                             (4 - boundary) ** 2 * (1 + shared))
    return exposure


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
    """Distinct low-load count cards and stabilizer-aware holdout selection."""
    from itertools import combinations_with_replacement, combinations

    q = G.q
    assert q == 24
    assert A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    sa = [[2, 3, 3]]

    def canonical(columns):
        direct = tuple(sorted(columns))
        swapped = tuple(sorted((b, a) for a, b in columns))
        return min(direct, swapped)

    # Canonical count architectures, not independent label draws.
    # Every additional slot consumes a previously unused count card.
    families = ((0, 2), (0, 3), (1, 1), (1, 2))
    books = []
    for family, (n4, n3) in enumerate(families):
        n2 = 7 - n3 - n4
        desired = {
            16: (8, 8), 17: (8, 9), 18: (9, 9),
        }[14 + n3 + 2*n4]
        seen = set()
        for fours in combinations_with_replacement(
                tuple((a, 4-a) for a in range(5)), n4):
            for threes in combinations_with_replacement(
                    tuple((a, 3-a) for a in range(4)), n3):
                for twos in combinations_with_replacement(
                        ((0, 2), (1, 1), (2, 0)), n2):
                    columns = fours + threes + twos
                    totals = tuple(sorted(
                        sum(column[i] for column in columns)
                        for i in range(2)))
                    if totals != desired:
                        continue
                    if not any(a and b for a, b in columns):
                        continue
                    profile = canonical(columns)
                    augmentation = [
                        sum((column[i] & 1) << j
                            for j, column in enumerate(profile))
                        for i in range(2)
                    ]
                    # Necessary full-rank augmentation condition.
                    # Actual component and complete ranks decide later.
                    if gf2_rank(augmentation) != 2:
                        continue
                    seen.add(profile)
        book = sorted(seen)
        random.Random(624161718 + 1009*family).shuffle(book)
        assert len(book) >= 5
        books.append(book)

    cards = []
    for position in range(max(map(len, books))):
        for family, book in enumerate(books):
            if position < len(book):
                cards.append((family, position, book[position]))
    if not 0 <= idx < len(cards):
        raise ValueError("unused low-load count catalogue exhausted")
    family, position, profile = cards[idx]
    sb = [[column[i] for column in profile] for i in range(2)]
    terms = sum(map(sum, sb))
    expected_cnots = 6*q*(9*8 + 4*terms)
    assert terms in (16, 17, 18)
    assert expected_cnots in (19584, 20160, 20736)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert sum(sa[0]) + max(map(sum, profile)) <= 12
    assert max(map(sum, sb)) + max(sa[0]) <= 12

    element_orders = {}
    for g in G.elements:
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if G._key(value) == G._key(G.e):
                element_orders[G.idx(g)] = t
                break
        else:
            raise ValueError("Dic6 element order missing")

    def relative_order(a, b):
        return element_orders[G.idx(G.mul(G.inv(a), b))]

    mode = (family + position) % 4
    triangle_orders = ((12, 12), (12, 6),
                       (6, 6), (12, 4))[mode]

    def draw_A():
        return [[[G.e, (_dic_delta(rng, 12), 0)],
                 _dic_triple(rng, G, triangle_orders[0], mode),
                 _dic_triple(rng, G, triangle_orders[1], mode + 1)]]

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        used = []
        for j, column in enumerate(profile):
            if column == (1, 1):
                if not used:
                    t = G.e
                else:
                    legal = [
                        g for g in G.elements
                        if all(relative_order(h, g) > 2 for h in used)
                    ]
                    if not legal:
                        raise ValueError("matching holonomy palette exhausted")
                    t = rng.choice(legal)
                used.append(t)
                grid[0][j], grid[1][j] = [G.e], [G.inv(t)]
                continue
            for i, count in enumerate(column):
                if not count:
                    continue
                base = G.pick(rng)
                if count == 1:
                    cell = [base]
                elif count == 2:
                    # Pure degree-two cells must not import order-four
                    # mixed-coset cycles into the weak subgraph.
                    if sum(column) == 2:
                        delta = (_dic_delta(rng, 12), 0)
                    elif (i + j + mode) % 3 == 0:
                        delta = (rng.randrange(12), 1)
                    else:
                        order = 6 if mode == 2 else 12
                        delta = (_dic_delta(rng, order), 0)
                    cell = [base, G.mul(base, delta)]
                elif count == 3:
                    order = (12, 6, 4)[(i + j + mode) % 3]
                    cell = [G.mul(base, g) for g in
                            _dic_triple(rng, G, order, i + j + mode)]
                elif count == 4:
                    # A fresh support object, not a saved trinomial
                    # with an appended orbit label.
                    for attempt in range(128):
                        cell = rng.sample(G.elements, 4)
                        if len({g[1] for g in cell}) != 2:
                            continue
                        if all(relative_order(a, b) > 2
                               for a, b in combinations(cell, 2)):
                            break
                    else:
                        raise ValueError("mixed quartet brief exhausted")
                else:
                    raise ValueError("unsupported low-load cell count")
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                grid[i][j] = cell
        return grid

    # These searches use the existing helper-built component convention.
    # Longer component spectra shortlist alternatives, not circuit scores.
    apool = _dic_component_pool(rng, G, "A", draw_A, 320)
    bpool = _dic_component_pool(rng, G, "B", draw_B, 640)
    options = []
    for ac, A in apool:
        for bc, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            if n != 552 or n - gf2_rank(hx) - gf2_rank(hz) != 240:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            guard = (
                max(ac[0], bc[0]), ac[0] + bc[0],
                max(ac[1], bc[1]), ac[1] + bc[1],
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
            )
            bx = _dic_boundary(hx, hz, n)
            bz = _dic_boundary(hz, hx, n)
            coarse = (
                max(7*ac[2], 3*bc[2]), 7*ac[2] + 3*bc[2],
                max(bx, bz), bx + bz,
                max(7*ac[3], 3*bc[3]), 7*ac[3] + 3*bc[3],
                ac[4] + bc[4], ac[5] + bc[5],
            )
            options.append((guard, coarse, len(options),
                            A, B, hx, hz, ac, bc))
    if not options:
        raise ValueError("no exact-rate low-load Dic6 pairing")
    options.sort(key=lambda item: item[:3])

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two integrated fault intervals, NOT an ASC memory experiment.

        Equations are H*e1+m=s1 and H*(e1+e2)=s2.
        Intermediate observations are noisy; final observations perfect.
        Random proper suffixes model possible spread, not ASC gate order.
        """
        import numpy as np

        def sites(word):
            result = []
            while word:
                bit = word & -word
                result.append(bit.bit_length() - 1)
                word ^= bit
            return result

        m = len(checks)
        width = 2*n + m
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        neighborhoods = [sites(row) for row in augmented]
        dc = max(map(len, neighborhoods))
        incidence = np.full((2*m, dc), width, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            incidence[r, :len(ns)] = ns
        flat = incidence.ravel()
        valid = flat != width
        ne = len(flat)
        degree = np.bincount(flat[valid], minlength=width)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (width+1, int(degree.max())), ne, dtype=np.int32)
        used_edges = np.zeros(width, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used_edges[c]] = edge
            used_edges[c] += 1

        sensed = np.asarray(
            [c.bit_count() for c in _columns(checks, n)], dtype=float)
        opposite = np.asarray(
            [c.bit_count() for c in _columns(stabilizers, n)], dtype=float)
        levels = np.asarray((0.25, 0.40, 0.55))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2 + sensed + opposite)*2/3
        measurement_mu = np.asarray([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measurement_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, width+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < data_p
        error[:, n:2*n] = channel.random((shots, n)) < data_p
        error[:, 2*n:width] = channel.random((shots, m)) < measure_p

        for row in stabilizers:
            order = channel.permutation(sites(row))
            if len(order) < 2:
                continue
            event_p = levels*3*0.002*len(order)/3
            for interval in range(2):
                events = np.flatnonzero(channel.random(shots) < event_p)
                for shot in events:
                    cut = int(channel.integers(1, len(order)))
                    error[shot, interval*n+order[cut:]] ^= True

        # This marginal allowance is heuristic, not a calibrated fault
        # channel. Relay BP, LSD and six-round histories are omitted.
        marginal_mu = data_mu + 3*0.002*opposite*0.25
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate(
            (marginal_p, marginal_p, np.clip(measure_p, 0.002, 0.25)),
            axis=1)
        prior = np.full((shots, width+1), 30.0)
        prior[:, :width] = np.log((1-probabilities)/probabilities)
        target = (error[:, flat].reshape(
            shots, 2*m, dc).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = error[:, :n] ^ error[:, n:2*n]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = row
                    break
                row ^= previous

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    return True
                word ^= previous
            return False

        wrong = late = work = 0
        for iteration in range(1, 37):
            count = len(prior)
            shaped = messages.reshape(count, 2*m, dc)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, ne)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, ne+1))
            padded[:, :ne] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, width] = 30.0
            hard = posterior < 0
            matched = (hard[:, flat].reshape(
                count, 2*m, dc).sum(axis=2) & 1).astype(bool)
            settled = np.all(matched == target, axis=1)
            final = hard[:, :n] ^ hard[:, n:2*n]
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(final[local] ^ true_final[local]):
                    word |= 1 << int(c)
                wrong += int(nontrivial(word))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, late, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (wrong+unresolved, unresolved,
                late+unresolved, work+37*unresolved)

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2], x[3]+z[3])

    # Compare equally short-protected finalists. Independent holdout
    # results lead selection; training only breaks subsequent ties.
    floor = options[0][0]
    eligible = [entry for entry in options if entry[0] == floor][:3]
    seed = 624552161 + 1009*idx
    training = []
    for option in eligible:
        guard, coarse, number, A, B, hx, hz, ac, bc = option
        x = noisy_probe(hz, hx, 552, seed, 32)
        z = noisy_probe(hx, hz, 552, seed+1, 32)
        training.append((quality(x, z)+coarse+(number,),
                         number, option, x, z))
    training.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, option, tx, tz in training[:2]:
        guard, coarse, _, A, B, hx, hz, ac, bc = option
        x = noisy_probe(hz, hx, 552, seed+1000003, 64)
        z = noisy_probe(hx, hz, 552, seed+1000004, 64)
        key = guard + quality(x, z) + quality(tx, tz) + coarse + (number,)
        finalists.append((key, A, B, hx, hz, ac, bc, x, z, tx, tz))
    finalists.sort(key=lambda item: item[0])
    key, A, B, hx, hz, ac, bc, x, z, tx, tz = finalists[0]

    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 240
    assert tuple(map(len, A[0])) == (2, 3, 3)
    assert [[len(cell) for cell in row] for row in B] == sb
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert max(row.bit_count() for row in hx + hz) == _grid_weights(sa, sb)
    assert 6*sum(row.bit_count() for row in hx + hz) == expected_cnots
    assert expected_cnots < 22032
    assert 0.425 <= 240/552 < 0.475
    assert all(len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "idx": idx, "k": 240,
            "count_contract": (family, position, profile),
            "triangle_orders": triangle_orders,
            "component_profiles": (ac, bc), "cnots": expected_cnots,
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z), "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete low-load, holdout-selected cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: Full-rank odd-degree A trinomials and strengthened B columns suffice for rare collective failures on D26 (1,3)x(2,7): fitness was 3.9894, all five measured codes retained abundant weight-six sector-one witnesses, and both bases reached failed-shot caps.
# REFUTED: Dic6 noncommuting connections, component weight-six screening and pair-boundary selection suffice for rare collective circuit failures: fitness was 4.0112; all five measured codes had d2s=6, depth18 and 22032 CNOTs, both bases hit failed-shot caps, and failures collectively affected all 240 logicals.
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
