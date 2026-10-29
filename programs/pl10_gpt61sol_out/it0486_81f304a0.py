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
CONCEPT = ("On C3:C8 with inversion action, order 24, A=(1,3) and B=(2,7), lean A components combined with four degree-three B columns and gate-propagating noisy-history selection should improve worse-basis circuit fitness above 4.0835.")
GROUP = ("Frozen byte-for-byte under L1: retain C3 semidirect C8, order 24 and the existing encoding. This bed reached 4.1250 with a different filling, versus the parent's 4.0835; neither reading establishes a group advantage. Its order-eight/twelve elements support lean binomials without involutory parallel columns.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=552, kb=240 and guaranteed rate 10/23=0.43478. Size factor23 permits q<=26; this group uses q=24. Unequal component matrices are24x72 and48x168. Require verified k=240, preserving zero k-kb. Parent sector-one witnesses span many Acol/Bcol blocks, failures affect all240 logicals, and diffuse1.0-1.2x enrichment establishes no persistent Brow hotspot.")
FILLING = ("Twenty contracts cross four binomial/trinomial ratio briefs with five B architectures. A counts are(2,3,3); additional batches test internal degree-two B edges or A counts(2,2,3). B always has four degree-three and three degree-two columns, with row totals9/9 and no degree-four column. HX maximum weight is11 or10; HZ maximum is12. Six-round CNOTs are20736 or19440, versus22032 in the parent. Verified lifts screen ranks and short kernels; rooted component invariants conservatively collapse finalists. Six-round probes propagate categorical two-qubit depolarising faults through conflict-free CNOT schedules, include measured-ancilla preparation/readout faults, and decode noisy histories with a closing data interval. Settled residuals are classified modulo actual stabilizers. Independent noise and schedule holdouts lead final selection. Target parent Z failed-shot rates7.4-10.2%, X rates4.5-8.8%, collective logical flips and late settlement. Probe schedules, preparation conventions and BP differ from ASC/relay/LSD; these are uncalibrated selection experiments, not measured pL.")
FAMILY_NAME = "r045_C3byC8_q24_13x27_lean_regularB_gatehistory_v5"

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
    """Lean support contracts selected by independent gate-history holdouts."""
    from itertools import combinations
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 24
    design, batch = idx % 20, idx // 20
    policy, architecture = divmod(design, 5)
    extension = batch % 3

    # Every head contributes (6,6), every tail (3,3).
    # Unlike the shown reinforced lean filling, there is NO degree-four
    # B column. The two new heads alter whole zero-cell architectures.
    heads = (
        ((2,1), (2,1), (1,2), (1,2)),
        ((3,0), (1,2), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (1,2)),
        ((3,0), (3,0), (0,3), (0,3)),
        ((2,1), (2,1), (2,1), (0,3)),
    )
    tail = (((2,0), (0,2), (1,1)) if extension == 1 else
            ((1,1), (1,1), (1,1)))
    profile = heads[architecture] + tail
    sa = [[2, 2, 3]] if extension == 2 else [[2, 3, 3]]
    sb = [[column[i] for column in profile] for i in range(2)]
    brief = ((8,3,12), (12,6,12),
             (8,12,6), (12,4,12))[policy]
    assert tuple(map(sum, sb)) == (9, 9)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert sum(sa[0]) + max(map(sum, profile)) <= 11
    assert max(map(sum, sb)) + max(sa[0]) == 12
    expected_cnots = 6*q*(9*sum(sa[0]) + 4*18)
    assert expected_cnots in (19440, 20736)

    orders = _mc_orders(G)
    palettes = {
        t: [g for g in G.elements if orders[G.idx(g)] == t]
        for t in (3, 4, 6, 8, 12)
    }

    def cell_key(cell):
        return min(tuple(sorted(
            G.idx(G.mul(g, G.inv(origin))) for g in cell))
            for origin in cell)

    triangles = {t: set() for t in (3, 4, 6, 12)}
    nonidentity = [g for g in G.elements if g != G.e]
    for u, v in combinations(nonidentity, 2):
        signature = tuple(sorted((
            orders[G.idx(u)], orders[G.idx(v)],
            orders[G.idx(G.mul(G.inv(u), v))],
        )))
        commute = G._key(G.mul(u, v)) == G._key(G.mul(v, u))
        for t in triangles:
            if commute != (t == 4):
                continue
            if signature == tuple(sorted((t, 8, 8))):
                triangles[t].add(cell_key([G.e, u, v]))
    triangles = {
        t: [[G.elements[c] for c in key] for key in sorted(book)]
        for t, book in triangles.items()
    }
    assert all(triangles.values()) and all(palettes.values())

    def a_identity(grid):
        # Collapse obvious one-row column gauges and common group
        # automorphisms, not arbitrary complete-code Tanner equivalence.
        images = []
        for u in (1, 2):
            for c in range(3):
                for v in (1, 3, 5, 7):
                    images.append(tuple(sorted(
                        cell_key([((u*g[0] + c*(g[1] % 2)) % 3,
                                   v*g[1] % 8) for g in cell])
                        for cell in grid[0])))
        return min(images)

    def draw_A():
        order, first, second = brief
        if extension == 2:
            return [[[G.e, rng.choice(palettes[8])],
                     [G.e, rng.choice(palettes[12])],
                     list(rng.choice(triangles[first]))]]
        return [[[G.e, rng.choice(palettes[order])],
                 list(rng.choice(triangles[first])),
                 list(rng.choice(triangles[second]))]]

    strong_orders = ((8,12), (6,8,12),
                     (3,8,12), (4,8,12))[policy]
    matching_pool = [
        g for t in ((8,12) if extension == 0 else (6,8,12))
        for g in palettes[t]
    ]

    def draw_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        used, matching_number, pure_number = [], 0, 0
        for j, column in enumerate(profile):
            if column == (1, 1):
                if matching_number == 0:
                    t = G.e
                else:
                    legal = [
                        g for g in matching_pool
                        if all(orders[G.idx(G.mul(G.inv(h), g))] > 3
                               for h in used)
                    ]
                    if not legal:
                        legal = [g for g in matching_pool if g not in used]
                    t = rng.choice(legal)
                used.append(t)
                matching_number += 1
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
                        order = (8, 12)[(pure_number + policy) % 2]
                        pure_number += 1
                    else:
                        order = rng.choice(strong_orders)
                    delta = rng.choice(palettes[order])
                    cell = [base, G.mul(base, delta)]
                else:
                    length = (3, 6, 12)[(i+j+policy+batch) % 3]
                    cell = [G.mul(base, g)
                            for g in rng.choice(triangles[length])]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                B[i][j] = cell
        return B

    def fingerprint(rows, width):
        """Rooted component invariant, NOT full Tanner canonization."""
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, syndrome in enumerate(_columns(rows, width)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(nr+c)
                adjacency[nr+c].append(r)
                syndrome ^= bit
        result = []
        for root in range(0, nr, q):
            colors = [0]*nr + [1]*width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(10):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                              for w in adjacency[v])))
                    for v in range(nr+width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            result.append(digest.hexdigest())
        return tuple(sorted(result))

    def portfolio(side, draw, trials):
        width = (3 if side == "A" else 7)*q
        rank = (1 if side == "A" else 2)*q
        pool, seen = [], set()
        for trial in range(trials):
            grid = draw()
            identity = (a_identity(grid) if side == "A" else
                        tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row))
            if identity in seen:
                continue
            seen.add(identity)
            rows = _component_rows(G, grid, side)
            if gf2_rank(rows) != rank:
                continue
            result = _kernel4(rows, width)
            if result is None:
                continue
            short, cols = result
            four = _four_cycles(rows)
            coarse = (short[0], short[1], four, short[2])
            pool.append((coarse, trial, grid, rows, cols, short, four))
            pool.sort(key=lambda entry: entry[:2])
            del pool[16:]

        finalists, classes = [], set()
        for _, trial, grid, rows, cols, short, four in pool:
            identity = fingerprint(rows, width)
            if identity in classes:
                continue
            classes.add(identity)
            spectrum = _component_score(short, cols, rows, four)
            key = spectrum[:4] + (four, spectrum[5], short[2], trial)
            finalists.append((key, grid, spectrum, identity))
            if len(finalists) == 6:
                break
        finalists.sort(key=lambda entry: entry[0])
        if not finalists:
            raise ValueError("no rank-safe lean component in contract")
        return finalists[:3]

    apool = portfolio("A", draw_A, 320)
    bpool = portfolio("B", draw_B, 640)
    options = []
    for _, A, ac, aid in apool:
        for _, B, bc, bid in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 552 or k != 240:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            protection = (
                max(ac[0], bc[0]), ac[0]+bc[0],
                max(ac[1], bc[1]), ac[1]+bc[1],
                max(rx[0], rz[0]), rx[0]+rz[0],
                max(rx[1], rz[1]), rx[1]+rz[1],
            )
            coarse = (
                max(7*ac[2], 3*bc[2]), 7*ac[2]+3*bc[2],
                max(7*ac[3], 3*bc[3]), 7*ac[3]+3*bc[3],
                ac[4]+bc[4], ac[5]+bc[5], aid, bid,
            )
            options.append((protection, coarse, A, B, hx, hz, ac, bc))
    options.sort(key=lambda entry: entry[:2])
    if not options:
        raise ValueError("no verified k=240 lean CSS finalist")

    def gate_history_probe(hx, hz, error_x, noise_seed, schedule_seed, shots):
        """Gate-propagating selection probe, NOT the ASC experiment.

        One CSS Pauli component is propagated exactly through the chosen
        CNOT circuit. Two-qubit depolarising faults retain their categorical
        endpoint correlation. There is no idling noise. Separate X/Z
        phases guarantee ideal CSS extraction, but are not ASC schedules.

        The decoder is approximate spacetime sum-product BP, not relay/LSD.
        Its independent priors do not reproduce the correlated circuit
        channel. A seventh data interval permits faults between the last
        noisy readout and the perfect closing syndrome.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length()-1)
                row ^= bit
            return out

        neighborhoods = {
            "X": [sites(row) for row in hx],
            "Z": [sites(row) for row in hz],
        }
        own = "Z" if error_x else "X"
        opposite = "X" if error_x else "Z"
        checks = hz if error_x else hx
        stabilizers = hx if error_x else hz
        ns = neighborhoods[own]
        n, m, rounds = 552, len(checks), 6

        def schedule(rows, seed):
            # Greedy edge colorings are conflict-free at both ancillas
            # and data. Different holdout seeds change suffix geometry.
            edges = [(r, c) for r, row in enumerate(rows) for c in row]
            srng = random.Random(seed)
            best = None
            for _ in range(3):
                order = list(range(len(edges)))
                srng.shuffle(order)
                row_used, data_used = [0]*len(rows), [0]*n
                layers = []
                for edge in order:
                    r, c = edges[edge]
                    free = ~(row_used[r] | data_used[c])
                    bit = free & -free
                    color = bit.bit_length()-1
                    while len(layers) <= color:
                        layers.append([])
                    layers[color].append(edge)
                    row_used[r] |= bit
                    data_used[c] |= bit
                if best is None or len(layers) < len(best):
                    best = layers
            return [
                (np.asarray([edges[e][0] for e in layer], dtype=np.int32),
                 np.asarray([edges[e][1] for e in layer], dtype=np.int32),
                 np.asarray(layer, dtype=np.int32))
                for layer in best
            ], len(edges)

        schedules = {}
        for basis in ("X", "Z"):
            schedules[basis] = schedule(
                neighborhoods[basis],
                schedule_seed + int(basis == "Z"))

        channel = np.random.default_rng(noise_seed)
        p = np.asarray((0.0016, 0.0020, 0.0024))[
            np.arange(shots) % 3]
        data = np.zeros((shots, n+1), dtype=bool)
        observed = []
        component_shift = 0 if error_x else 1

        # Ancilla errors in the opposite preparation eigenbasis are
        # gauge errors: their complete spread is a stabilizer. Since
        # that phase precedes the sensed phase, they can be omitted.
        for t in range(rounds):
            for basis in (opposite, own):
                rows = neighborhoods[basis]
                layers, ne = schedules[basis]
                ancilla = np.zeros((shots, len(rows)), dtype=bool)
                if basis == own:
                    ancilla ^= channel.random(ancilla.shape) < p[:, None]
                    if basis == "X":
                        ancilla ^= (
                            channel.random(ancilla.shape)
                            < (2*p/3)[:, None])

                active = channel.random((shots, ne)) < p[:, None]
                pauli = channel.integers(
                    1, 16, size=(shots, ne), dtype=np.uint8)
                control_fault = active & (
                    ((pauli >> component_shift) & 1) != 0)
                target_fault = active & (
                    ((pauli >> (component_shift+2)) & 1) != 0)

                for rs, cs, es in layers:
                    if basis == own:
                        ancilla[:, rs] ^= data[:, cs]
                    else:
                        data[:, cs] ^= ancilla[:, rs]
                    if basis == "X":
                        ancilla[:, rs] ^= control_fault[:, es]
                        data[:, cs] ^= target_fault[:, es]
                    else:
                        data[:, cs] ^= control_fault[:, es]
                        ancilla[:, rs] ^= target_fault[:, es]

                if basis == own:
                    readout = ancilla.copy()
                    if basis == "X":
                        readout ^= (
                            channel.random(readout.shape)
                            < (2*p/3)[:, None])
                    readout ^= channel.random(readout.shape) < p[:, None]
                    observed.append(readout)

        width = max(map(len, ns))
        sensed_sites = np.full((m, width), n, dtype=np.int32)
        for r, row in enumerate(ns):
            sensed_sites[r, :len(row)] = row
        perfect = (data[:, sensed_sites].sum(axis=2) & 1).astype(bool)
        observations = np.stack(observed, axis=1)
        target = np.empty((shots, rounds+1, m), dtype=bool)
        target[:, 0] = observations[:, 0]
        target[:, 1:rounds] = (
            observations[:, 1:] ^ observations[:, :-1])
        target[:, rounds] = perfect ^ observations[:, -1]
        target = target.reshape(shots, (rounds+1)*m)
        true_final = data[:, :n].copy()

        intervals = rounds+1
        variables = intervals*n + rounds*m
        dummy = variables
        detectors = intervals*m
        row_width = width+2
        incidence = np.full(
            (detectors, row_width), dummy, dtype=np.int32)
        for t in range(intervals):
            for r, row in enumerate(ns):
                incidence[t*m+r, :len(row)] = [t*n+c for c in row]
                position = len(row)
                if t < rounds:
                    incidence[t*m+r, position] = intervals*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = (
                        intervals*n+(t-1)*m+r)
        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degrees = np.bincount(flat[valid], minlength=variables)
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (variables+1, int(degrees.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        sensed_degree = np.asarray([
            c.bit_count() for c in _columns(checks, n)], dtype=float)
        spread_degree = np.asarray([
            c.bit_count() for c in _columns(stabilizers, n)], dtype=float)
        hook_load = np.zeros(n, dtype=float)
        before = np.zeros(len(neighborhoods[opposite]), dtype=np.int32)
        for rs, cs, _ in schedules[opposite][0]:
            hook_load[cs] += before[rs]
            before[rs] += 1

        # Marginal-exposure priors are approximations. The sampled
        # circuit above, not these priors, supplies the target syndromes.
        ordinary_mu = (8/15)*(
            sensed_degree + spread_degree + hook_load)
        closing_mu = (4/15)*sensed_degree
        data_mu = np.concatenate(
            [ordinary_mu for _ in range(rounds)] + [closing_mu])
        measure_mu = np.asarray([
            (8/15)*len(row) + 2 + (4/3 if own == "X" else 0)
            for row in ns])
        mu = np.concatenate((data_mu, np.tile(measure_mu, rounds)))
        rates = np.clip(
            0.5*(1-np.exp(-2*p[:, None]*mu[None, :])), 1e-5, 0.25)
        prior = np.empty((shots, variables+1), dtype=float)
        prior[:, :variables] = np.log((1-rates)/rates)
        prior[:, dummy] = 30.0
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        target_sign = np.where(target, -1.0, 1.0)

        pivots = {}
        for row in stabilizers:
            while row:
                pivot = row.bit_length()-1
                old = pivots.get(pivot)
                if old is None:
                    pivots[pivot] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                pivot = word.bit_length()-1
                old = pivots.get(pivot)
                if old is None:
                    return True
                word ^= old
            return False

        logical = effort = residual_weight = 0
        for iteration in range(1, 37):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(np.exp(
                logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1), dtype=float)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, detectors, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :intervals*n].reshape(count, intervals, n),
                axis=1)
            for shot in np.flatnonzero(settled):
                residual = decoded_final[shot] ^ true_final[shot]
                word = 0
                for c in np.flatnonzero(residual):
                    word |= 1 << int(c)
                if nontrivial(word):
                    logical += 1
                    residual_weight += int(residual.sum())
                effort += iteration

            keep = ~settled
            if not np.any(keep):
                return logical, 0, effort, residual_weight
            messages = np.clip(
                0.30*messages + 0.70*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        return (logical+unresolved, unresolved,
                effort+37*unresolved, residual_weight)

    def quality(x, z):
        # Any-logical shot failures lead; residual support is only a tie
        # break, not an estimate of per-logical failure probability.
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2],
                x[3]+z[3])

    floor = options[0][0]
    eligible = [option for option in options if option[0] == floor][:4]
    seed = 240552719 + 1009*idx
    training = []
    for option in eligible:
        protection, coarse, A, B, hx, hz, ac, bc = option
        x = gate_history_probe(hx, hz, True, seed, seed+100003, 48)
        z = gate_history_probe(hx, hz, False, seed+1, seed+100004, 48)
        key = protection + quality(x, z) + coarse
        training.append((key, option, x, z))
    training.sort(key=lambda entry: entry[0])

    winner = None
    for _, option, tx, tz in training[:2]:
        protection, coarse, A, B, hx, hz, ac, bc = option
        # Both noise and gate ordering are independent holdouts.
        x = gate_history_probe(
            hx, hz, True, seed+1000003, seed+2000003, 128)
        z = gate_history_probe(
            hx, hz, False, seed+1000004, seed+2000004, 128)
        ex = _mc_exposure(hx, hz, 552)
        ez = _mc_exposure(hz, hx, 552)
        key = (protection + quality(x, z) + quality(tx, tz)
               + (max(ex, ez), ex+ez) + coarse)
        if winner is None or key < winner[0]:
            winner = (key, option, x, z, tx, tz)

    key, option, x, z, tx, tz = winner
    protection, coarse, A, B, hx, hz, ac, bc = option
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 240
    assert tuple(map(len, A[0])) == tuple(sa[0])
    actual_sb = [[len(cell) for cell in row] for row in B]
    assert actual_sb == sb
    assert sorted(map(sum, zip(*actual_sb))) == [2,2,2,3,3,3,3]
    assert max(row.bit_count() for row in hx) == sum(sa[0])+3
    assert max(row.bit_count() for row in hz) == 12
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
    assert 0.425 <= 240/552 < 0.475
    return {"A": A, "B": B, "design": design, "batch": batch,
            "ratio_brief": brief, "k": 240, "cnots": expected_cnots,
            "component_profiles": (ac, bc),
            "gate_history_holdout": (x, z),
            "gate_history_training": (tx, tz), "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete lean, gate-history-selected cell without redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 failed 110/1250 Z and 121/1250 X shots, and all five measured codes retained abundant sector-one weight-six witnesses.
# REFUTED: Long-order noncommuting ratios and degree-two multiplicity/boundary screening alone yield rare collective failures on C3:C8 (1,3)x(2,7): fitness was4.0835; all ten measured basis experiments reached failed-shot caps, all240 logicals were affected, and abundant sector-one weight-six witnesses remained.
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
