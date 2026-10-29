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
CONCEPT = ("AGL(1,8) x C2, order 112, shapes (2,3)x(2,3): independent component selection with localized, full-component and partial-orbit kernel searches should raise five-code fitness above 310.3540 in cell 0.100; sampled readings remain provisional upper bounds.")
GROUP = ("Keep the frozen affine-field group and order: this bed leads cell 0.100 at fitness 310.3540. Its central C2 and seven order-four translation subgroups provide searchable component subspaces beyond the parent's order-eight translation-fixed screen. All restrictions are extracted from verified lifts.")
SHAPES = ("Keep equal components with 336 variables and 224 checks: q=112 gives n=1456, k>=112 and guaranteed rate 1/13. Exact ranks retain k/n<0.125. Parent minima occupied sector 1: slot 2 used Acol2 across every Bcol; slots 16 and 34 used Bcol1/2 across every Acol. Independently protect both component kernels.")
FILLING = ("Keep twenty legal support/parity cards with column totals 4..5 and rows <=7. Independently shortlist A and B, compare translation refinements by actual kernel witnesses, enumerate small localized kernels, and retest finalists with cancellation descent and central/translation-plane orbit probes. Canonical linear stabilizer remainders strengthen the actual-span-verified normal-logical screen. No sampled reading certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r010_affineF8xC2_23x23_independent_partial_orbit_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    # (u,v,z): affine map t -> alpha^v*t + u over F8,
    # times central C2. F8 uses alpha^3 + alpha + 1 = 0.
    if not valid_q(q):
        raise ValueError("this affine-field bed requires order 112")
    powers = (1, 2, 4, 3, 6, 7, 5)
    els = [(u, v, z) for z in range(2)
           for v in range(7) for u in range(8)]

    def field_mul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return out

    def red(g):
        return (g[0] & 7, g[1] % 7, g[2] & 1)

    def mul(g, h):
        return (g[0] ^ field_mul(powers[g[1]], h[0]),
                (g[1] + h[1]) % 7, g[2] ^ h[2])

    def inv(g):
        vi = (-g[1]) % 7
        return (field_mul(powers[vi], g[0]), vi, g[2])

    def idx(g):
        u, v, z = red(g)
        return (7 * z + v) * 8 + u

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


# Different density allocations, not permutations of one labeling.
# Every column has >=4 terms. Templates 3..8 are paired only with
# template 0: its row/column totals of 6/4 accommodate their density.
_SUPPORT_TEMPLATES = (
    ((2, 2, 2), (2, 2, 2)),  # 0: uniform baseline
    ((3, 2, 2), (2, 2, 2)),  # 1: one reinforced cell
    ((3, 2, 2), (2, 3, 2)),  # 2: balanced reinforcement
    ((3, 2, 2), (3, 2, 2)),  # 3: one degree-six column
    ((3, 3, 2), (2, 2, 3)),  # 4: asymmetric row density
    ((3, 3, 2), (3, 2, 3)),  # 5: distributed dense side
    ((3, 3, 2), (3, 3, 2)),  # 6: two degree-six columns
    ((4, 2, 2), (2, 4, 2)),  # 7: concentrated dense side
    ((4, 2, 2), (2, 3, 3)),  # 8: mixed concentration
    ((3, 3, 1), (2, 2, 3)),  # 9: uneven cells, balanced degrees
    ((4, 1, 2), (1, 4, 2)),  # 10: crossed concentrated cells
    ((3, 3, 1), (2, 1, 3)),  # 11: asymmetric balanced degrees
)
_SUPPORT_PAIRS = (
    (2, 2), (1, 2), (0, 2), (1, 1), (0, 1),
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (2, 9), (1, 9), (0, 9), (9, 9),
    (2, 10), (1, 10), (9, 10), (10, 10), (2, 11),
)


def _set_bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _affine_grid(rng, supports, phase):
    """Distinct v coordinates make every within-cell ratio non-involutory."""
    grid = []
    for i, counts in enumerate(supports):
        row = []
        for j, count in enumerate(counts):
            vs = list(range(7))
            rng.shuffle(vs)
            flip = rng.randrange(2)
            cell = []
            for t in range(count):
                if phase == 0:
                    z = flip ^ (t & 1)
                elif phase == 1:
                    z = flip ^ ((t & 1) if (i + j) & 1 else 0)
                else:
                    z = rng.randrange(2)
                cell.append((rng.randrange(8), vs[t], z))
            row.append(cell)
        grid.append(row)
    return grid


def _component_stats(rows, ncols):
    """Four cycles and check-projection triangle excess, not distance."""
    columns = [0] * ncols
    pairs = {}
    four = 0
    for r, row in enumerate(rows):
        ns = list(_set_bits(row))
        for u in ns:
            columns[u] |= 1 << r
        for a, u in enumerate(ns):
            for v in ns[a + 1:]:
                key = u * ncols + v
                previous = pairs.get(key, 0)
                four += previous
                pairs[key] = previous + 1

    # Reject isolated variables and identical component columns:
    # the latter supply classical weight-two kernel vectors.
    if not all(columns) or len(set(columns)) != ncols:
        return None

    adjacency = [0] * len(rows)
    forced_triangles = 0
    for signature in columns:
        degree = signature.bit_count()
        forced_triangles += degree * (degree - 1) * (degree - 2) // 6
        for r in _set_bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    # Require a connected component Tanner graph.
    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _set_bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _set_bits(neighbors >> (u + 1)):
            v += u + 1
            triangles += (neighbors & adjacency[v] &
                          (-1 << (v + 1))).bit_count()

    # With no four cycles, this is the number of genuine six cycles.
    # Otherwise it remains only a secondary projection statistic.
    return four, max(0, triangles - forced_triangles)


def _verified_components(G, hx, hz):
    """Extract both classical lifts from the scaffold's verified matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            lifted_row = hx[i * cb * q + r]  # beta = 0
            ar.append(sum(((lifted_row >> (j * cb * q)) & mask)
                          << (j * q) for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + r] & bmask             # A-column j = 0
          for a in range(rb) for r in range(q)]
    return ar, br


def NEW_CTX(rng, G, idx):
    """Joint quotient, translation, component-kernel and logical screening."""
    # All profiles have columns 4..5 and rows <=7. Profiles 1 and 3
    # add concentration patterns absent from the parent's templates.
    profiles = (
        ((3, 3, 1), (2, 2, 3)),
        ((4, 2, 1), (1, 3, 3)),
        ((3, 3, 1), (2, 1, 3)),
        ((3, 3, 1), (1, 1, 4)),
        ((3, 2, 2), (2, 3, 2)),
    )
    plans = (
        (0,0), (0,1), (1,1), (0,2), (1,2),
        (2,2), (0,3), (1,3), (2,3), (3,3),
        (0,4), (1,4), (2,4), (3,4), (4,4),
        (0,0), (0,1), (1,1), (0,2), (1,2),
    )
    ia, ib = plans[idx % len(plans)]
    sa, sb = profiles[ia], profiles[ib]
    phase = (idx // 5 + idx // 15 + idx // 20) % 3
    phase_b = (phase + (idx // 2) % 2) % 3
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    # This encoding orders the eight translations contiguously.
    assert all(G.idx((u, v, z)) == (7*z + v)*8 + u
               for z in range(2) for v in range(7) for u in range(8))
    n = 13 * G.q
    diagnostic_seed = rng.randrange(1 << 62)

    def pivots_of(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return pivots

    def remainder(word, pivots):
        """Canonical linear remainder, including below nonpivot positions."""
        residual = 0
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                bit = 1 << p
                residual |= bit
                word ^= bit
            else:
                word ^= old
        return residual

    def project(rows, width):
        """Restrict VERIFIED check rows to translation-constant vectors.
        Keep all check rows: no independently implemented group lift."""
        fw = width // 8
        return list(dict.fromkeys(
            sum((((row >> (8*j)) & 255).bit_count() & 1) << j
                for j in range(fw))
            for row in rows))

    def expand(word):
        """Embed a projected vector in the actual qubit coordinates."""
        out = 0
        for j in _set_bits(word):
            out |= 255 << (8*j)
        return out

    def fixed_minimum(rows, width):
        """EXACT minimum in the normal-fixed classical subspace only."""
        projected = project(rows, width)
        fw = width // 8
        pivots = pivots_of(projected)
        ordered = sorted(pivots.items())
        basis = []
        for free in range(fw):
            if free in pivots:
                continue
            word = 1 << free
            for p, row in ordered:
                if (word & row).bit_count() & 1:
                    word |= 1 << p
            basis.append(word)
        # Avoid spending exponential work on highly deficient components.
        if not basis or len(basis) > 18:
            return 0
        word = 0
        reading = fw + 1
        for t in range(1, 1 << len(basis)):
            word ^= basis[(t & -t).bit_length() - 1]
            reading = min(reading, word.bit_count())
        return 8 * reading

    def columns_of(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            for c in _set_bits(row):
                columns[c] |= 1 << r
        return columns

    def dependency_basis(columns, order):
        """Kernel words in the supplied column coordinates."""
        pivots, basis = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
        return basis

    def kernel_probe(columns, diagnostic, trials):
        """Nonzero fundamental words, pair/triple sums and descent."""
        reading = len(columns) + 1
        order = list(range(len(columns)))
        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = dependency_basis(columns, order)
            if not basis:
                continue
            basis.sort(key=int.bit_count)
            reading = min(reading, basis[0].bit_count())
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            short = basis[:8]
            descent = list(basis)
            diagnostic.shuffle(descent)
            for word in short:
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in descent:
                        candidate = word ^ other
                        if not candidate:
                            continue
                        w = candidate.bit_count()
                        if w < weight:
                            word, weight = candidate, w
                            changed = True
                    reading = min(reading, weight)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(96):
                    a, b, c = diagnostic.sample(basis, 3)
                    candidate = a ^ b ^ c
                    if candidate:
                        reading = min(reading, candidate.bit_count())
        return reading

    def localized_minimum(columns):
        """Exact minima when a one/two-block kernel has <=18 dimensions.

        A larger restricted kernel is sampled instead; neither case
        supplies a lower bound on the unrestricted quantum distance.
        """
        reading = len(columns) + 1
        blocks = [list(range(j*G.q, (j+1)*G.q)) for j in range(3)]
        restrictions = blocks + [
            blocks[a] + blocks[b]
            for a in range(3) for b in range(a+1, 3)
        ]
        for positions in restrictions:
            restricted = [columns[c] for c in positions]
            basis = dependency_basis(restricted, range(len(restricted)))
            if not basis:
                continue
            if len(basis) <= 18:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    reading = min(reading, word.bit_count())
            else:
                reading = min(reading, min(map(int.bit_count, basis)))
                for i, word in enumerate(basis):
                    for other in basis[:i]:
                        candidate = word ^ other
                        if candidate:
                            reading = min(reading, candidate.bit_count())
        return reading

    # These are coordinate partitions, not independently implemented lifts.
    # A's left multiplication commutes with right subgroup actions;
    # B's right multiplication commutes with left subgroup actions.
    planes = sorted({
        tuple(sorted((0, a, b, a ^ b)))
        for a in range(1, 8) for b in range(a+1, 8)
    })
    subgroups = [
        [G.e, (0, 0, 1)]
    ] + [[(u, 0, 0) for u in plane] for plane in planes]
    orbit_partitions = [[], []]
    for side in range(2):
        for subgroup in subgroups:
            remaining = set(range(G.q))
            orbits = []
            while remaining:
                root = min(remaining)
                g = G.elements[root]
                orbit = sorted({
                    G.idx(G.mul(g, h) if side == 0 else G.mul(h, g))
                    for h in subgroup
                })
                assert len(orbit) == len(subgroup)
                remaining.difference_update(orbit)
                orbits.append(orbit)
            orbit_partitions[side].append((len(subgroup), orbits))

    def component_reading(rows, width, diagnostic, trials=24, side=0):
        """Search genuine kernels of VERIFIED rows and their restrictions."""
        columns = columns_of(rows, width)
        reading = min(localized_minimum(columns),
                      kernel_probe(columns, diagnostic, trials))
        if trials >= 24:
            for orbit_size, orbits in orbit_partitions[side]:
                restricted = []
                for block in range(3):
                    for orbit in orbits:
                        syndrome = 0
                        for t in orbit:
                            syndrome ^= columns[block*G.q + t]
                        restricted.append(syndrome)
                d = kernel_probe(restricted, diagnostic,
                                 max(2, trials // 12))
                # Every selected orbit expands to exactly orbit_size bits.
                if d <= len(restricted):
                    reading = min(reading, orbit_size*d)
        return reading

    def normal_logical_reading(hx, hz, diagnostic, trials=48):
        """Search fixed-subspace logicals, verifying actual nontriviality.
        Quotient homology alone is insufficient in characteristic two:
        every retained witness is tested modulo the ORIGINAL span."""
        reading = n + 1
        fw = n // 8
        for opposite, stabilizers in ((hz, hx), (hx, hz)):
            qchecks = project(opposite, n)
            columns = columns_of(qchecks, fw)
            stabilizer_pivots = pivots_of(stabilizers)
            signatures = [
                remainder(255 << (8*c), stabilizer_pivots)
                for c in range(fw)
            ]
            # Only actual stabilizers are allowed in the descent.
            qstabs = [
                row for row in project(stabilizers, n)
                if row and not remainder(expand(row), stabilizer_pivots)
            ]
            for _ in range(trials):
                order = list(range(fw))
                diagnostic.shuffle(order)
                pivots = {}
                witnesses = []
                kernel_stabs = []
                for c in order:
                    syndrome = columns[c]
                    word = 1 << c
                    signature = signatures[c]
                    while syndrome:
                        p = syndrome.bit_length() - 1
                        old = pivots.get(p)
                        if old is None:
                            pivots[p] = (syndrome, word, signature)
                            break
                        syndrome ^= old[0]
                        word ^= old[1]
                        signature ^= old[2]
                    if not syndrome:
                        if signature:
                            witnesses.append(word)
                        else:
                            # Canonical zero signature proves that its
                            # expanded vector belongs to the actual span.
                            kernel_stabs.append(word)
                witnesses.sort(key=int.bit_count)
                descent = list(qstabs) + kernel_stabs
                diagnostic.shuffle(descent)
                for word in witnesses[:3]:
                    for sweep in range(2):
                        changed = False
                        for row in descent:
                            candidate = word ^ row
                            if candidate.bit_count() < word.bit_count():
                                word = candidate
                                changed = True
                        if not changed:
                            break
                    weight = 8 * word.bit_count()
                    if weight >= reading:
                        continue
                    actual = expand(word)
                    if any((actual & row).bit_count() & 1
                           for row in opposite):
                        continue
                    if not remainder(actual, stabilizer_pivots):
                        continue
                    reading = weight
        return reading

    def cycle_key(ast, bst):
        return (max(ast[0], bst[0]), ast[0] + bst[0],
                max(ast[1], bst[1]), ast[1] + bst[1])

    # A weak partner must not discard a promising component. Each side
    # retains its own quotient/cycle shortlist before any final pairing.
    pools = [[], []]
    width = 3 * G.q
    for trial in range(64):
        A = _affine_grid(rng, sa, phase)
        B = _affine_grid(rng, sb, phase_b)
        hx, hz, built_n = lifted_matrices(G, A, B)
        assert built_n == n
        ar, br = _verified_components(G, hx, hz)
        for side, grid, rows in ((0, A, ar), (1, B, br)):
            fixed = fixed_minimum(rows, width)
            pool = pools[side]
            if not fixed or (len(pool) == 5 and fixed < pool[-1]["fixed"]):
                continue
            stats = _component_stats(rows, width)
            if stats is None:
                continue
            pool.append(dict(grid=grid, rows=rows, fixed=fixed,
                             cycles=stats, trial=trial))
            pool.sort(key=lambda item: (
                -item["fixed"], item["cycles"], item["trial"]))
            del pool[5:]

    if not all(pools):
        raise ValueError("empty independently screened component pool")
    anchors = [pools[0][0]["grid"], pools[1][0]["grid"]]

    def component_merit(item):
        return (item["d"], item["fixed"],
                -item["cycles"][0], -item["cycles"][1])

    def select_component(pool, side):
        records = []
        for item in pool:
            alternatives = [item]
            for refinement in range(4):
                grid = [
                    [[(rng.randrange(8), v, z) for u, v, z in cell]
                     for cell in row]
                    for row in item["grid"]
                ]
                A, B = ((grid, anchors[1]) if side == 0
                        else (anchors[0], grid))
                xx, zz, built_n = lifted_matrices(G, A, B)
                assert built_n == n
                aa, bb = _verified_components(G, xx, zz)
                rows = aa if side == 0 else bb
                stats = _component_stats(rows, width)
                if stats is None:
                    continue
                alternatives.append(dict(
                    grid=grid, rows=rows, fixed=item["fixed"],
                    cycles=stats, trial=item["trial"]))

            # Cycles shortlist translations, but cannot choose the winner
            # without actual localized and full-component kernel probes.
            alternatives.sort(key=lambda record: record["cycles"])
            for record in alternatives[:2]:
                diagnostic = random.Random(
                    diagnostic_seed + 100003*side + 17)
                record["d"] = min(
                    record["fixed"],
                    component_reading(record["rows"], width,
                                      diagnostic, trials=8, side=side))
                records.append(record)

        records.sort(key=component_merit, reverse=True)
        finalists = records[:2]
        for record in finalists:
            # Independent longer searches include smaller subgroup orbits.
            diagnostic = random.Random(
                diagnostic_seed + 700001 + 100003*side)
            record["d"] = min(
                record["d"],
                component_reading(record["rows"], width,
                                  diagnostic, trials=48, side=side))
        finalists.sort(key=component_merit, reverse=True)
        return finalists

    aa = select_component(pools[0], 0)
    bb = select_component(pools[1], 1)
    pairs = sorted(
        ((a, b) for a in aa for b in bb),
        key=lambda pair: (
            min(pair[0]["d"], pair[1]["d"]),
            pair[0]["d"] + pair[1]["d"],
            min(pair[0]["fixed"], pair[1]["fixed"]),
            -pair[0]["cycles"][0] - pair[1]["cycles"][0]),
        reverse=True)

    best = None
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, built_n = lifted_matrices(G, A, B)
        assert built_n == n
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= G.q and 8*k < n):
            continue
        # Common diagnostic orders across pairings reduce selection noise.
        diagnostic = random.Random(diagnostic_seed + 900001)
        dn = normal_logical_reading(hx, hz, diagnostic, trials=48)
        da, db = a["d"], b["d"]
        detected = min(da, db, dn)
        quotient_key = (min(a["fixed"], b["fixed"]),
                        a["fixed"] + b["fixed"])
        ck = cycle_key(a["cycles"], b["cycles"])
        merit = (k*detected*detected, detected, quotient_key,
                 da + db, tuple(-x for x in ck))
        if best is None or merit > best[0]:
            best = (merit, A, B, k, (da, db, dn), hx, hz)

    if best is None:
        raise ValueError("no independent component pair passed rate gate")
    assert css_ok(best[5], best[6])
    assert max(row.bit_count() for row in best[5] + best[6]) <= min(12, W_MAX)
    return {"cells": {"A": best[1], "B": best[2]},
            "screen": best[0], "k": best[3],
            "kernel_readings": best[4], "design": idx}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently selected, kernel-screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A row-weight cap of 12 alone excludes abundant weight-6 logicals: the rolled C37 (2,6)x(2,6) parent had d=6 in all five measured slots, entirely in sector 1, despite k-kb ranging from 0 to 10.
# REFUTED: Cycle-only selection plus a high short-budget reading reliably excludes sub-60 logicals on this affine-field bed: parent slot 40 fell from d2s=70 to d30s=40, with sector-1 X witnesses spanning multiple Acol/Bcol blocks.
# REFUTED: Translation-fixed screening plus cycle-selected translation refinements excludes sub-60 measured minima: current slot 2 fell from d2s=68 to d30s=56, with an X witness on Acol2 across all three Bcols.
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
