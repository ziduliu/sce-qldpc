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
CONCEPT = ("Cyclic C19, order 19, shapes (1,4)x(2,7): new component degree profiles selected by independent six-round temporal-BP probes should improve worse-basis ASC circuit fitness above 3.3458 within cell 0.500.")
GROUP = ("Frozen under L1: retain C19, order 19, its multiplication and encoding byte-for-byte. This exact bed scored 3.3458; frequent collective failures motivate a controlled filling experiment, not a group-only reliability claim.")
SHAPES = ("Frozen (1,4)x(2,7): the size factor thirty permits q<=19, giving n=570, kb=285 and guaranteed rate 0.500. Unequal components have 76 variables/19 checks and 133 variables/38 checks. Require complete verified k=285,288 or295, all within cell 0.500. Parent failures affected all 288 logicals with diffuse 1.0-1.2x Acol/Bcol and Acol/Brow enrichment, so no persistent localized repair site is established.")
FILLING = ("Twenty contracts cross five canonical A geometries with four degree briefs: A=[2,2,2,3] with graphic B or two strengthened degree-three B columns; even A=[2,2,2,4] with graphic B; and unit-pivot A=[1,3,3,3] with graphic B. Additional batches consume additional canonical geometries. Graphic B has fourteen terms; strengthened B has sixteen. Six-round CNOT budgets are 17214,18354 or18240, versus the parent's 18240; all weights remain <=12. Verified component and full-CSS screens exclude logicals through three and retain the specified ranks. Degree-weighted noisy six-round spacetime BP probes compare structural finalists, followed by independent holdout selection. They omit the actual ASC ordering and relay/LSD cascade and are not calibrated pL measurements. Target the parent's 106-148 failed Z shots/250 and widespread logical flips rather than distance alone.")
FAMILY_NAME = "r050_C19_14x27_degreebrief_temporalbp_v4"

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
    """Degree contracts, verified screens and independent temporal probes."""
    import math
    import random
    from itertools import combinations

    q = G.q
    assert q == 19
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    kind = 0 if architecture < 2 else architecture - 1
    expected_k = (288, 285, 295, 288)[architecture]
    expected_b_rank = 2*q if architecture == 1 else 2*q - 1

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
        # Remove common group automorphisms, column origins and
        # permutations; complete Tanner equivalence remains downstream.
        return min(tuple(sorted(normalized(cell, u) for cell in cells))
                   for u in range(1, q))

    def component(grid, side):
        # Zero partners are probes, never submitted fillings.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    trinomials = sorted({
        normalized((0, a, b))
        for a, b in combinations(range(1, q), 2)
    })
    quartics = sorted({
        normalized((0,) + tail)
        for tail in combinations(range(1, q), 3)
    })
    step_sets = []
    for steps in combinations(range(1, (q + 1)//2), 3):
        signed = steps + tuple(-d for d in steps)
        if all((a+b+c) % q != 0
               for a in signed for b in signed for c in signed):
            step_sets.append(steps)
    assert step_sets

    # A deterministic relative-geometry catalogue supplies contracts.
    # Slot randomness is used for optimization, not for twenty orbit
    # copies of one filling. Every brief changes the parent's degree
    # multiset [2,2,3,3].
    catalogue_rng = random.Random(190427 + kind)
    book, seen_geometry = [], set()
    required_rank = q - 1 if kind == 1 else q
    for trial in range(4096):
        if kind < 2:
            steps = catalogue_rng.choice(step_sets)
            terminal = catalogue_rng.choice(
                quartics if kind == 1 else trinomials)
            cells = tuple((0, d) for d in steps) + (terminal,)
        else:
            cells = ((0,),) + tuple(
                catalogue_rng.sample(trinomials, 3))
        signature = geometry_key(cells)
        if signature in seen_geometry:
            continue
        seen_geometry.add(signature)
        grid = [[[(a,) for a in cell] for cell in signature]]
        rows, width = component(grid, "A")
        stats = _kernel_statistics(rows, width)
        if stats[0] or stats[1] or gf2_rank(rows) != required_rank:
            continue
        key = (stats[2], _four_cycles(rows), stats[3],
               _six_cycles(rows), signature)
        book.append((key, grid))
        if len(book) >= 48:
            break
    book.sort(key=lambda entry: entry[0])
    wanted = 5*batch + marker
    if wanted >= len(book):
        raise ValueError("Canonical degree-contract catalogue exhausted")
    akey, A = book[wanted]
    bin_directions = {
        next(iter(directions(tuple(g[0] for g in cell))))
        for cell in A[0] if len(cell) == 2
    }

    # Three graphic briefs are balanced degree-two incidence matrices.
    # The strengthened brief has two degree-three columns, row totals
    # eight and full augmentation rank, without restoring the archived
    # eighteen-term B model.
    graphic_profiles = (
        ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
        ((2,0),(0,2),(1,1),(1,1),(1,1),(1,1),(1,1)),
        ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
        ((2,0),(2,0),(2,0),(0,2),(0,2),(0,2),(1,1)),
    )
    if architecture == 1:
        profile = ((2,1),(1,2),(1,1),(1,1),(1,1),(1,1),(1,1))
    else:
        profile = graphic_profiles[(marker + batch) % 4]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == (
        [2,2,2,3] if kind == 0 else
        [2,2,2,4] if kind == 1 else [1,3,3,3])
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    expected_cnots = 6*q*(9*sum(sa[0]) + 5*sum(map(sum, sb)))

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [
        (i, j) for j, p in enumerate(profile) for i in range(2)
        if p[i] == 1 and p != (1, 1)
    ] + [(1, j) for j in cross[1:]]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def gauge(grid):
        # A common translation of the complete lower row is a gauge.
        # Apply it to strengthened cells as well as matching cells.
        origin = grid[1][cross[0]][0][0]
        grid[1] = [
            [((g[0] - origin) % q,) for g in cell]
            for cell in grid[1]
        ]
        return grid

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        if architecture == 1:
            for j, p in enumerate(profile):
                if p == (1, 1):
                    continue
                for i, count in enumerate(p):
                    grid[i][j] = rng.sample(G.elements, count)
        else:
            for i, j in internal:
                grid[i][j] = [
                    (0,), (rng.randrange(1, (q + 1)//2),)]
        ratios = rng.sample(range(q), len(cross))
        for j, ratio in zip(cross, ratios):
            grid[0][j], grid[1][j] = [(0,)], [(ratio,)]
        return gauge(grid)

    def assess_B(grid):
        rows, width = component(grid, "B")
        stats = _kernel_statistics(rows, width)
        # Exact short-word exclusions come from verified component
        # syndromes, including mixed degree-two/three columns.
        energy = (300*stats[0] + 50*stats[1] + stats[2]) / q
        return energy, stats, rows

    pool, seen = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != expected_b_rank:
            return
        signature = tuple(tuple(g[0] for g in cell)
                          for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        coupling = sum(
            direction(grid[i][j][1][0] - grid[i][j][0][0])
            in bin_directions for i, j in internal)
        key = (stats[2], coupling)
        serial += 1
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    for restart in range(4):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(280):
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
            temperature = 4.0 * (0.985 ** step) + 0.06
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No target-rank B component passed the short-word contract")

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

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Noisy six-round BP surrogate, NOT the ASC memory experiment.

        Jointly decode data increments and measurement flips. Degree
        exposure and occasional opposite-check pairs approximate gate
        noise; the actual schedule and relay/LSD cascade are omitted.
        """
        import numpy as np

        def neighbors(rows):
            result = []
            for row in rows:
                ns = []
                while row:
                    bit = row & -row
                    ns.append(bit.bit_length() - 1)
                    row ^= bit
                result.append(ns)
            return result

        ns = neighbors(checks)
        opposite = neighbors(stabilizers)
        m, rounds = len(checks), 6
        variables = rounds*(n + m)
        dummy = variables
        row_width = max(map(len, ns)) + 2
        detector_count = (rounds + 1)*m
        incidence = np.full(
            (detector_count, row_width), dummy, dtype=np.int32)
        for t in range(rounds + 1):
            for r, sites in enumerate(ns):
                position = 0
                if t < rounds:
                    for c in sites:
                        incidence[t*m+r, position] = t*n+c
                        position += 1
                    incidence[t*m+r, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = rounds*n+(t-1)*m+r

        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (variables+1, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        dc = np.asarray([c.bit_count() for c in columns(checks, n)])
        ds = np.asarray([c.bit_count() for c in columns(stabilizers, n)])
        data_rate = 0.002*(2 + dc + ds)
        measurement_rate = np.asarray([
            0.002*(2 + len(sites))*2/3 for sites in ns])
        base = np.concatenate((
            np.tile(data_rate, rounds),
            np.tile(measurement_rate, rounds)))
        channel = np.random.default_rng(seed)
        levels = np.asarray((1.0, 1.25, 1.5))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None]*base[None, :], 0.25)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :variables] = (
            channel.random((shots, variables)) < rates)

        # Possible ancilla-spread pairs, not ASC-certified hooks.
        for t in range(rounds):
            selected = channel.integers(len(opposite), size=shots)
            active = channel.random(shots) < 0.20
            for shot in np.flatnonzero(active):
                sites = opposite[int(selected[shot])]
                if len(sites) < 2:
                    continue
                a = int(channel.integers(len(sites)))
                b = int(channel.integers(len(sites)-1))
                b += b >= a
                error[shot, t*n+sites[a]] ^= True
                error[shot, t*n+sites[b]] ^= True

        target = (
            error[:, flat].reshape(
                shots, detector_count, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.empty((shots, variables+1), dtype=np.float64)
        prior[:, :variables] = np.log((1-rates)/rates)
        prior[:, dummy] = 30.0
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)
        quotient = quotient_columns(stabilizers, n)

        failures = iteration_sum = 0
        for iteration in range(1, 29):
            count = len(prior)
            shaped = messages.reshape(
                count, detector_count, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs),
                1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(
                    count, detector_count, row_width).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for local in np.flatnonzero(settled):
                remainder = 0
                for c in np.flatnonzero(
                        decoded_final[local] ^ true_final[local]):
                    remainder ^= quotient[int(c)]
                failures += int(bool(remainder))
                iteration_sum += iteration

            keep = ~settled
            if not np.any(keep):
                return failures, 0, iteration_sum
            messages = np.clip(
                0.30*messages +
                0.70*(posterior[:, flat]-outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        return (failures+unresolved, unresolved,
                iteration_sum+28*unresolved)

    options = []
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 570 or k != expected_k:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        structural = tuple(
            value for t in (1, 0, 2, 3)
            for value in (max(px[t], pz[t]), px[t] + pz[t]))
        structural += coarse
        options.append((structural, serial, B, hx, hz, px, pz))
    if not options:
        raise ValueError("No target-rate CSS filling passed the logical screen")
    options.sort(key=lambda item: item[:2])

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                x[2]+z[2])

    # Common streams compare label alternatives within this contract.
    # Independent holdouts, not the training shots, decide the winner.
    seed = 190427570 + 1009*idx
    coarse_probes = []
    for number, option in enumerate(options[:4]):
        structural, serial, B, hx, hz, px, pz = option
        x = temporal_probe(hz, hx, 570, seed, 24)
        z = temporal_probe(hx, hz, 570, seed+1, 24)
        coarse_probes.append((
            quality(x, z)+structural+(number,), number, option, x, z))
    coarse_probes.sort(key=lambda item: item[:2])

    finalists = []
    for training, number, option, tx, tz in coarse_probes[:2]:
        structural, serial, B, hx, hz, px, pz = option
        x = temporal_probe(hz, hx, 570, seed+1000003, 64)
        z = temporal_probe(hx, hz, 570, seed+1000004, 64)
        score = quality(x, z) + quality(tx, tz) + structural + (number,)
        finalists.append((score, option, x, z))
    finalists.sort(key=lambda item: item[0])
    score, option, x, z = finalists[0]
    structural, serial, B, hx, hz, px, pz = option
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == expected_k
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz)*6 == expected_cnots
    assert 0.475 <= expected_k/570 < 0.525
    return {"A": A, "B": B, "design": (architecture, wanted),
            "k": expected_k, "logical_risk": (px, pz),
            "component_contract": akey, "cycle_score": structural,
            "temporal_bp_holdout": (x, z), "selection_score": score,
            "expected_cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Return the verified degree-contract cell without late resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: Asymmetric A<=3/B<=4 component exclusion balances C19 memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored only 3.1933 despite much lower Z failed-shot rates.
# REFUTED: Difference-separated [2,2,3,3] A, triangle-free graphic B and shared-check quartic selection suffice for rare C19 memory failures: fitness was 3.3458, measured Z failures were 106..148/250, and failures affected all 288 logicals.
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
