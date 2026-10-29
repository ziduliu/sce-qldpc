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
CONCEPT = ("On Frobenius C7 semidirect C3(s=2), order 21, shapes (1,3)x(2,7), gate-lean count architectures selected by a six-step noisy-syndrome BP stress test should improve worse-basis circuit fitness beyond the parent's 4.0144.")
GROUP = ("Keep the exact order-21 group for L1. This bed produced five measured codes with k=210 and fitness 4.0144; related lower-rate beds reached about 6.9, but those readings do not isolate a group advantage. This experiment changes filling only.")
SHAPES = ("Frozen (1,3)x(2,7) targets cell 0.450: n=483, kb=210 and guaranteed rate 10/23=0.43478. The size window permits q<=26; the frozen group fixes q=21. Unequal components have 63 variables/21 checks and 147 variables/42 checks. Require exact k=210. Parent sector-one witnesses spanned Acol/Bcol blocks, with diffuse 1.0-1.2x failure enrichment and no consistent Brow hotspot.")
FILLING = ("Twenty cards cross ten count architectures with two relative-element policies. A counts are (3,3,3) or (2,3,3); B has 16-18 terms, including new degree-four columns. Weight caps remain <=12 and six-round CNOT budgets are 17136-19278. Anneal fresh voltages using helper-derived singleton actions, screen verified component kernels through weight six, then test full-matrix finalists under common six-step data/measurement noise with final perfect syndrome and stabilizer-aware residual classification. This sum-product BP stress test is neither ASC nor relay-BP and is not calibrated pL. It targets failed-shot caps, collective flips across all 210 logicals, and late decoder settlement, rather than treating d2s=6 as circuit protection.")
FAMILY_NAME = "r045_Frob21_13x27_gatelean_decodingstress_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 21

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C3 requires order 21")
    # x^7=y^3=1, y*x*y^-1=x^2; elements are x^a*y^b.
    # 2^3=1 mod 7 makes the action well defined.
    els = [(a, b) for b in range(3) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(2, b, 7) * c) % 7, (b + d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-pow(2, bi, 7) * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return a + 7 * b

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
    """Twenty count/element cards; fresh search and decoding stress."""
    import math
    from itertools import combinations

    try:
        import numpy as np
    except ImportError:
        np = None

    q = G.q
    assert q == 21
    design, batch = idx % 20, idx // 20
    architecture, policy = divmod(design, 2)
    # Entries are (A counts, B column-count pairs). These are count
    # contracts, not twenty random draws from one support architecture.
    cards = (
        ((3,3,3), ((2,1),(2,1),(1,2),(1,2),(1,1),(1,1),(1,1))),
        ((3,3,3), ((2,1),(1,2),(1,1),(1,1),(1,1),(1,1),(1,1))),
        ((3,3,3), ((2,1),(2,1),(1,2),(1,1),(1,1),(1,1),(1,1))),
        ((3,3,3), ((3,0),(1,2),(1,2),(1,1),(1,1),(1,1),(1,1))),
        ((3,3,3), ((3,0),(0,3),(1,2),(2,1),(1,1),(1,1),(1,1))),
        ((2,3,3), ((2,1),(1,2),(1,1),(1,1),(1,1),(1,1),(1,1))),
        ((2,3,3), ((2,2),(2,1),(1,2),(1,1),(1,1),(1,1),(1,1))),
        ((2,3,3), ((3,1),(1,2),(1,2),(1,1),(1,1),(1,1),(1,1))),
        ((2,3,3), ((3,0),(1,3),(1,2),(1,1),(1,1),(1,1),(1,1))),
        ((2,3,3), ((3,1),(1,2),(2,1),(2,0),(0,2),(1,1),(0,2))),
    )
    aw, profile = cards[architecture]
    sa = [list(aw)]
    sb = [[c[i] for c in profile] for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)

    rotations = [g for g in G.elements if g[1] == 0 and g != G.e]
    complements = [g for g in G.elements if g[1] != 0]
    nonidentity = [g for g in G.elements if g != G.e]
    triples = {2: [], 3: []}
    for u, v in combinations(nonidentity, 2):
        span = len({0, u[1], v[1]})
        if span not in triples:
            continue
        if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
            continue
        triples[span].append([G.e, u, v])

    # Cache ONLY singleton actions returned by the verified helper.
    # No separately implemented left/right permutation convention.
    actions = {"A": {}, "B": {}}
    mask = (1 << q) - 1
    for g in G.elements:
        x = lifted_matrices(G, [[[g]]], [[[]]])[0]
        z = lifted_matrices(G, [[[]]], [[[g]]])[1]
        actions["A"][G._key(g)] = tuple(r & mask for r in x)
        actions["B"][G._key(g)] = tuple(r & mask for r in z)

    def component(side, grid):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, value in enumerate(block):
                        out[t] ^= value << (j * q)
            rows.extend(out)
        return rows

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def identity(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def draw_cell(side, i, j, count):
        if not count:
            return []
        if side == "A":
            if count == 2:
                # The sole A doubleton has order-seven ratio, not a
                # length-three complementary cycle.
                return [G.e, rng.choice(rotations)]
            span = 2 if policy == 0 else (3 if (j + batch) % 2 == 0 else 2)
            return list(rng.choice(triples[span]))

        base = G.pick(rng)
        if count == 1:
            return [base]
        if count == 2:
            # Mixed ratios are allowed only in strengthened columns.
            mixed = policy == 1 and sum(profile[j]) >= 3
            delta = rng.choice(complements if mixed else rotations)
            return [base, G.mul(base, delta)]
        if count == 3:
            span = 2 if policy == 0 else 3
            relative = rng.choice(triples[span])
            return [G.mul(base, g) for g in relative]
        # Degree-four cards test supports the parent never submitted.
        for _ in range(128):
            cell = rng.sample(G.elements, count)
            if len({g[1] for g in cell}) >= 2:
                return cell
        raise ValueError("mixed-coset heavy cell unavailable")

    def portfolio(side, counts):
        width = len(counts[0]) * q
        rank_target = len(counts) * q
        sites = [(i,j) for i, row in enumerate(counts)
                 for j, w in enumerate(row) if w]
        pool, seen = [], set()
        serial = 0

        def state(grid):
            rows = component(side, grid)
            four, six = _four_cycles(rows), _six_cycles(rows)
            return rows, four, six, (90 * four + six / 6) / q

        def remember(grid, rows, four, six):
            nonlocal serial
            ident = identity(grid)
            if ident in seen:
                return
            seen.add(ident)
            if gf2_rank(rows) != rank_target:
                return
            result = _kernel4(rows, width)
            if result is None:
                return
            short, cols = result
            key = (short[0], short[1], four, six, short[2])
            if len(pool) == 8 and key > pool[-1][0]:
                return
            # Check cached action composition against the actual helper.
            assert rows == _component_rows(G, grid, side)
            serial += 1
            pool.append((key, serial, clone(grid), rows, cols, short, four))
            pool.sort(key=lambda e: e[:2])
            del pool[8:]

        for restart in range(4):
            grid = [[draw_cell(side, i, j, w)
                     for j, w in enumerate(row)]
                    for i, row in enumerate(counts)]
            rows, four, six, energy = state(grid)
            remember(grid, rows, four, six)
            steps = 240 + 20 * (batch % 3)
            for step in range(steps):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j, counts[i][j])
                trial_rows, c4, c6, proposed = state(grid)
                delta = proposed - energy
                temperature = 12 * (0.984 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    rows, four, six, energy = trial_rows, c4, c6, proposed
                else:
                    grid[i][j] = old
                if step >= 80 and step % 20 == 0:
                    remember(grid, rows, four, six)
            remember(grid, rows, four, six)

        # A fresh-draw recovery stays inside the same count/element card.
        if not pool:
            for _ in range(384):
                grid = [[draw_cell(side, i, j, w)
                         for j, w in enumerate(row)]
                        for i, row in enumerate(counts)]
                rows, four, six, _ = state(grid)
                remember(grid, rows, four, six)
                if len(pool) >= 4:
                    break
        if not pool:
            raise ValueError("no full-rank component in count/element card")

        refined = []
        for _, number, grid, rows, cols, short, four in pool[:6]:
            spectrum = _component_score(short, cols, rows, four)
            boundary, histogram = _boundary3(cols)
            key = spectrum[:4] + (boundary,) + spectrum[4:]
            refined.append((key, number, grid, spectrum, histogram))
        refined.sort(key=lambda e: e[:2])
        return refined[:3]

    def decoding_stress(checks, stabilizers, other, n, seed):
        """Six-step phenomenological sum-product BP, NOT the ASC circuit.

        Gate-degree-dependent independent noise is an uncalibrated
        exposure model. There are no correlated hooks, ASC ordering,
        relay-BP stages, or LSD here. Final measurement is perfect.
        Only the scaffold-built matrices define all check incidences.
        """
        rounds, shots, iterations = 6, 32, 50
        m = len(checks)
        incidence = []
        for row in checks:
            columns = []
            while row:
                bit = row & -row
                columns.append(bit.bit_length() - 1)
                row ^= bit
            incidence.append(columns)

        edge_variables, offsets = [], []
        for t in range(rounds):
            for r, columns in enumerate(incidence):
                offsets.append(len(edge_variables))
                edge_variables.extend(t * n + c for c in columns)
                if t:
                    edge_variables.append(rounds*n + (t-1)*m + r)
                if t < rounds - 1:
                    edge_variables.append(rounds*n + t*m + r)

        ev = np.asarray(edge_variables, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        lengths = np.diff(np.append(offsets, len(ev)))
        edge_checks = np.repeat(np.arange(rounds*m), lengths)
        nv = rounds*n + (rounds-1)*m

        degrees = [x.bit_count() + y.bit_count()
                   for x, y in zip(_columns(checks, n), _columns(other, n))]
        pd = np.minimum(0.08, 0.0035 * (np.asarray(degrees) + 2))
        pm = np.minimum(0.05, 0.0015 * (
            np.asarray([r.bit_count() for r in checks]) + 2))
        probabilities = np.concatenate((
            np.tile(pd, rounds), np.tile(pm, rounds-1)))
        prior = np.log((1-probabilities) / probabilities)
        nrng = np.random.RandomState(seed)
        errors = nrng.random_sample((shots, nv)) < probabilities
        observed = np.add.reduceat(
            errors[:, ev].astype(np.int8), offsets, axis=1) % 2
        signs = 1.0 - 2.0 * observed
        incoming = np.broadcast_to(prior[ev], (shots, len(ev))).copy()
        decoded = np.zeros((shots, nv), dtype=bool)
        settled = np.zeros(shots, dtype=bool)
        late = 0

        for step in range(iterations):
            tangent = np.tanh(np.clip(incoming, -30, 30) / 2)
            tangent = np.where(np.abs(tangent) < 1e-10, 1e-10, tangent)
            products = np.multiply.reduceat(tangent, offsets, axis=1)
            extrinsic = (products * signs)[:, edge_checks] / tangent
            messages = 2 * np.arctanh(np.clip(
                extrinsic, -1 + 1e-8, 1 - 1e-8))
            totals = np.asarray([
                np.bincount(ev, weights=line, minlength=nv)
                for line in messages
            ])
            posterior = prior + totals
            hard = posterior < 0
            predicted = np.add.reduceat(
                hard[:, ev].astype(np.int8), offsets, axis=1) % 2
            good = np.all(predicted == observed, axis=1)
            fresh = good & ~settled
            decoded[fresh] = hard[fresh]
            settled |= good
            late += int(np.count_nonzero(~settled))
            if np.all(settled):
                break
            proposed = prior[ev] + totals[:, ev] - messages
            incoming = 0.4 * incoming + 0.6 * proposed
        decoded[~settled] = hard[~settled]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        # Space-time parity equations imply zero final syndrome for
        # settled shots. Classify the final residual modulo stabilizers.
        residual = np.logical_xor.reduce(
            (errors[:, :rounds*n] ^ decoded[:, :rounds*n]).reshape(
                shots, rounds, n), axis=1)
        failed = 0
        for s in range(shots):
            if not settled[s]:
                failed += 1
                continue
            word = sum(1 << int(c) for c in np.flatnonzero(residual[s]))
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    break
                word ^= old
            failed += int(bool(word))
        return failed, int(np.count_nonzero(~settled)), late

    aa, bb = portfolio("A", sa), portfolio("B", sb)
    options = []
    for _, an, A, ac, ah in aa:
        for _, bn, B, bc, bh in bb:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 483 or k != 210:
                continue
            rx, rz = _syndrome_risk(hx, n), _syndrome_risk(hz, n)
            if rx is None or rz is None:
                continue
            # Exclude genuinely short mechanisms before using an
            # uncalibrated decoder. Do not chase sampled distance.
            protection = (
                max(ac[0], bc[0]), ac[0] + bc[0],
                max(ac[1], bc[1]), ac[1] + bc[1],
                int(bool(ac[2] or bc[2])),
            )
            fx, fz = _four_cycles(hx), _four_cycles(hz)
            structural = (
                max(rx[0], rz[0]), rx[0] + rz[0],
                max(rx[1], rz[1]), rx[1] + rz[1],
                max(7*ac[3], 3*bc[3]), 7*ac[3] + 3*bc[3],
                max(fx / len(hx), fz / len(hz)), fx + fz,
                ac[5] + bc[5],
            )
            options.append((protection, structural, an, bn,
                            A, B, hx, hz, ac, bc, ah, bh))
    if not options:
        raise ValueError("no exact-rate full-matrix finalist")
    options.sort(key=lambda e: e[:4])

    best = None
    for entry in options[:6]:
        protection, structural, an, bn, A, B, hx, hz, ac, bc, ah, bh = entry
        if np is not None:
            # Identical noise streams for finalists in a slot reduce
            # selection noise; fresh slots never reuse saved matrices.
            seed = 17293 + 101*idx
            sx = decoding_stress(hx, hz, hz, 483, seed)
            sz = decoding_stress(hz, hx, hx, 483, seed + 1)
            stress_key = tuple(
                v for t in range(3)
                for v in (max(sx[t], sz[t]), sx[t] + sz[t]))
        else:
            # Portable structural fallback, explicitly not a decoding
            # measurement. NumPy is optional, not a validity dependency.
            sx = sz = None
            stress_key = ()
        key = protection + stress_key + structural + (an, bn)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, ac, bc, ah, bh, sx, sz)

    key, A, B, hx, hz, ac, bc, ah, bh, sx, sz = best
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 210
    assert max(r.bit_count() for r in hx + hz) <= min(12, W_MAX)
    cnots = 6 * sum(r.bit_count() for r in hx + hz)
    expected = 6*q*(9*sum(aw) + 4*sum(map(sum, sb)))
    assert cnots == expected
    assert 0.425 <= 210 / 483 < 0.475
    return {"A": A, "B": B, "design": design, "batch": batch,
            "architecture": architecture, "policy": policy,
            "k": 210, "cnots": cnots, "screen": key,
            "component_profiles": (ac, bc),
            "boundary_histograms": (ah, bh),
            "phenomenological_stress": (sx, sz)}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Parallel-syndrome removal and ambiguity minimization suffice for rare collective failures on D14 (1,4)x(2,7): fitness was 3.2219, measured X failures were 108..139/250, and abundant weight-three witnesses remained across sector 1 despite mostly first-pass settlement.
# REFUTED: D26 odd-degree trinomials and strengthened B columns suffice for rare collective circuit failures: fitness was 3.9894; slot 6 still failed 110/1250 Z and 121/1250 X shots, with ubiquitous sector-1 weight-six witnesses and diffuse failure enrichment.
# REFUTED: Quotient-controlled trinomials plus component spectrum/boundary selection suffice for rare collective failures on Frob21 (1,3)x(2,7): fitness was 4.0144; all five measured codes reached failed-shot caps, retained abundant sector-one weight-six witnesses, and failures collectively touched all 210 logicals.
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
