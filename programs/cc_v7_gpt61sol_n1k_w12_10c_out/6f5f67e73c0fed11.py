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
CONCEPT = ("For the affine group (C2^4):C5 over F16, order 80, A=(1,3), B=(2,5), invertible cubic cells and independently screened degree-three component columns should raise five-code fitness above 35.5 in cell 0.350. Failure to exceed that target refutes this experiment; distance readings remain provisional upper bounds.")
GROUP = ("Jump to the additive F16 group acted on by its order-five multiplicative subgroup: q=80, n=1360. This elementary-abelian-normal construction is genuinely different from the parent's cyclic-normal metacyclic group. In F2[G], the normal-subgroup augmentation ideal is nilpotent. A trinomial projects to weight one or three in F2[C5]; Phi5 is irreducible over F2, so that projection, and hence the cubic cell, is a unit. Verify the resulting block ranks rather than use this fact as a distance claim. BEDS contains no order-80 bed.")
SHAPES = ("Change to A=(1,3), B=(2,5): n=17*80=1360 and k>=6*80=480 give guaranteed rate 6/17=0.352941. The size window permits q<=88. Components have variables/checks 240/80 and 400/160, nominal rates 2/3 and 3/5. The 0.350 incumbent scores 35.5 with readings ten. Parent minima were sector-1 witnesses using one Acol across both Bcols or one Bcol across both Acols; change both component lengths and add a second B check row. Exact full ranks retain [0.325,0.375).")
FILLING = ("A has three terms in each cell; every B column has total degree three. Five B weighted topologies crossed with four exact C4/C6 objectives define twenty design cards, not orbit copies. B row totals are at most nine, so HX=9+3=12 and HZ<=9+3=12. Mixed complement coordinates exclude involutory within-cell ratios; cubic relative generators must not commute. Authoritative monomial lifts assemble all screening matrices. Independently shortlist connected, rank-screened components, exclude kernel words through four, search genuine kernel witnesses with cancellation descent, and independently deepen finalists. Final authoritative lifts verify component assembly, exact ranks, both weight sums and CSS. None of the sampled component readings certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_F16add_C5_13x25_cubicunits_degree3_c6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 80

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("the F16-additive C5 affine bed requires order 80")

    # F16 = F2[z]/(z^4+z+1); z is primitive.
    # Elements (a,b) act as t -> a + z^(3b)*t.
    def fmul(a, b):
        a, b = int(a) & 15, int(b) & 15
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 16:
                a ^= 19
        return out

    action = [1, 8, 12, 10, 15]
    assert fmul(action[-1], action[1]) == 1
    els = [(a, b) for b in range(5) for a in range(16)]

    def red(g):
        return (int(g[0]) & 15, int(g[1]) % 5)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return (a ^ fmul(action[b], c), (b + d) % 5)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 5
        return (fmul(action[bi], a), bi)

    def idx(g):
        a, b = red(g)
        return 16*b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _syndrome_columns(rows, n):
    """Columns of a verified bit-row matrix, as syndrome integers."""
    cols = [0] * n
    mask = (1 << n) - 1
    for r, row in enumerate(rows):
        x = row & mask
        while x:
            bit = x & -x
            cols[bit.bit_length() - 1] |= 1 << r
            x ^= bit
    return cols


def _no_kernel4(cols):
    """Exact exclusion of every nonzero kernel vector of weight <=4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    # A pair matching a singleton gives weight three; matching an
    # earlier pair gives weight four. Shared indices would imply
    # duplicate columns, already excluded above.
    seen = singles.copy()
    for i, a in enumerate(cols):
        for j in range(i):
            v = a ^ cols[j]
            if v in seen:
                return False
            seen.add(v)
    return True


# Four inequivalent density allocations crossed with five occupancy models.
# Entries are the two cell counts on A and B, respectively.
_COUNT_PAIRS = (
    ((4, 4), (4, 4)),
    ((3, 4), (4, 4)),
    ((3, 5), (3, 4)),
    ((3, 4), (3, 4)),
)
_COSET_COUNTS = (
    {3: (1, 1, 1), 4: (2, 1, 1), 5: (2, 2, 1)},
    {3: (2, 1, 0), 4: (2, 2, 0), 5: (3, 2, 0)},
    {3: (2, 1, 0), 4: (3, 1, 0), 5: (3, 1, 1)},
)


def _bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _mixed_cells(rng, counts, side, model, separation):
    """Every cell mixes cosets; no single-coset polynomial pair recipe."""
    cells = []
    for j, count in enumerate(counts):
        if model < 3:
            kind = model
        elif model == 3:
            kind = 0 if side == "A" else 1
        else:
            kind = (0, 1)[j] if side == "A" else (2, 0)[j]
        fibers = list(range(3))
        rng.shuffle(fibers)
        cell = []
        for b, size in zip(fibers, _COSET_COUNTS[kind][count]):
            if size == 0:
                continue
            for _ in range(2000):
                rotations = rng.sample(range(97), size)
                ds = [(a - c) % 97 for a in rotations
                      for c in rotations if a != c]
                if (len(set(ds)) == len(ds)
                        and all(min(d, 97 - d) >= separation for d in ds)):
                    break
            else:
                raise ValueError("rotation support construction failed")
            cell.extend((a, b) for a in rotations)
        assert len(cell) == count
        assert len(set(cell)) == count
        assert len({b for a, b in cell}) >= 2
        cells.append(cell)
    return [cells]


def _verified_components(hx, hz, q):
    """Extract the two components from the scaffold's actual full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            ar.append(sum(((source >> (j * cb * q)) & mask)
                          << (j * q) for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ar, br


def _component_quality(rows, width):
    """Exact four cycles; projection-triangle excess is only a heuristic."""
    columns = _syndrome_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    pair_uses = {}
    four = 0
    for row in rows:
        ns = list(_bits(row))
        for t, u in enumerate(ns):
            for v in ns[t + 1:]:
                key = u * width + v
                previous = pair_uses.get(key, 0)
                four += previous
                pair_uses[key] = previous + 1

    adjacency = [0] * len(rows)
    forced = 0
    for signature in columns:
        degree = signature.bit_count()
        forced += degree * (degree - 1) * (degree - 2) // 6
        for r in _bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v]
                          & (-1 << (v + 1))).bit_count()
    return (four, max(0, triangles - forced)), columns


def _kernel_reading(rng, rows, width):
    """Sample actual component kernel words; an upper reading, not a bound below."""
    reading = width + 1
    for _ in range(6):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapped = [1 << p for p in permutation]
        pivots = {}
        for original in rows:
            row = 0
            for c in _bits(original):
                row |= mapped[c]
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items())
        basis = []
        for free in range(width):
            if free in pivots:
                continue
            word = 1 << free
            for p, row in ordered:
                if (word & row).bit_count() & 1:
                    word |= 1 << p
            basis.append(word)
            reading = min(reading, word.bit_count())
        for i, word in enumerate(basis):
            for other in basis[:i]:
                reading = min(reading, (word ^ other).bit_count())
        if len(basis) >= 3:
            for _ in range(160):
                a, b, c = rng.sample(basis, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
    return reading


def _legacy_fourterm_ctx(rng, G, idx):
    """Historical parent reference only; never called on the new group."""
    q = G.q
    width = 2 * q
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    # Each profile specifies an unordered pair of within-cell occupancies.
    # Twenty unordered profile pairs, excluding the entirely even-fiber
    # pair, are explicit designs rather than permutation images.
    occupancies = ((2, 1, 1), (3, 1, 0), (2, 2, 0))
    profiles = ((0, 0), (0, 1), (1, 1),
                (0, 2), (1, 2), (2, 2))
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    pa, pb = plans[idx % 20]
    phase = (idx // 20) % 3
    diagnostic_seed = rng.randrange(1 << 62)

    def fresh(profile):
        cells = []
        for kind in profiles[profile]:
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, size in zip(fibers, occupancies[kind]):
                if not size:
                    continue
                if size == 3 and phase:
                    # Three-point affine classes are unchanged by common
                    # rotation scaling/translation. Classes 3 and 5 are
                    # distinct even after permuting the three points.
                    t = (3, 5)[phase - 1]
                    origin = rng.randrange(97)
                    scale = rng.randrange(1, 97)
                    rotations = [origin, (origin + scale) % 97,
                                 (origin + t * scale) % 97]
                else:
                    for _ in range(256):
                        rotations = rng.sample(range(97), size)
                        differences = [(a - c) % 97 for a in rotations
                                       for c in rotations if a != c]
                        if len(set(differences)) == len(differences):
                            break
                    else:
                        raise ValueError("Sidon fiber generation failed")
                cell.extend((a, b) for a in rotations)
            assert len(cell) == len(set(cell)) == 4
            cells.append(cell)
        return [cells]

    def kernel_probe(columns, diagnostic, trials):
        """Search genuine classical kernel words, never certify distance.

        Column elimination tracks words in original coordinates. Pair
        sums, triple sums and nonzero cancellation descent all remain
        in the kernel of the VERIFIED component matrix.
        """
        reading = width + 1
        order = list(range(width))
        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome = columns[c]
                word = 1 << c
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
                    reading = min(reading, word.bit_count())

            light = sorted(basis, key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            # Descent explores combinations beyond the parent's single
            # systematic basis and its pair sums. Zero is never a witness.
            diagnostic.shuffle(basis)
            for word in light[:6]:
                weight = word.bit_count()
                for sweep in range(2):
                    changed = False
                    for other in basis:
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
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    candidate = a ^ b ^ c
                    if candidate:
                        reading = min(reading, candidate.bit_count())
            if reading <= 24:
                break
        return reading

    pools = [[], []]
    signatures = [set(), set()]
    # Each call supplies both components under the exact evaluator
    # conventions. Their original pairing does not constrain selection.
    for trial in range(40):
        A, B = fresh(pa), fresh(pb)
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 5 * q == 1455
        ar, br = _verified_components(hx, hz, q)
        for side, grid, rows in ((0, A, ar), (1, B, br)):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in grid[0])
            if signature in signatures[side]:
                continue
            quality = _component_quality(rows, width)
            if quality is None:
                continue
            signatures[side].add(signature)
            cycles, columns = quality
            pools[side].append({
                "grid": grid, "rows": rows, "columns": columns,
                "cycles": cycles, "trial": trial,
            })

    def select_component(pool, side):
        if not pool:
            raise ValueError("empty verified component pool")
        floor = min(record["cycles"][0] for record in pool)
        # Extra slots explore fewer projection triangles while allowing
        # one additional orbit of four-cycles; this changes graph selection.
        allowance = q if phase == 1 else 0
        near = [record for record in pool
                if record["cycles"][0] <= floor + allowance]
        if phase == 1:
            near.sort(key=lambda record: (
                record["cycles"][1], record["cycles"][0],
                record["trial"]))
        else:
            near.sort(key=lambda record: (
                record["cycles"], record["trial"]))

        screened = []
        mask = (1 << q) - 1
        # Continue past a rejected record rather than wasting its slot.
        for record in near:
            rows = record["rows"]
            rank = gf2_rank(rows)
            if rank < q - 3:
                continue
            # Inspect individual blocks extracted from verified rows.
            # A loss beyond the three quotient dimensions is rejected;
            # no separate left/right action is implemented here.
            cell_ranks = (
                gf2_rank([row & mask for row in rows]),
                gf2_rank([row >> q for row in rows]),
            )
            if min(cell_ranks) < q - 3:
                continue
            if not _no_kernel4(record["columns"]):
                continue
            diagnostic = random.Random(
                diagnostic_seed + 100003 * side)
            d = kernel_probe(record["columns"], diagnostic,
                             16 if phase == 2 else 12)
            record["rank"] = rank
            record["cell_ranks"] = cell_ranks
            record["d"] = d
            screened.append(record)
            if len(screened) == 6:
                break
        if not screened:
            raise ValueError("no component passed rank/kernel screening")

        def merit(record):
            four, six = record["cycles"]
            return (record["d"], -four, -six, record["rank"])

        screened.sort(key=merit, reverse=True)
        finalists = screened[:2]
        for record in finalists:
            # A fresh, longer search can only lower the first reading.
            # Common diagnostic seeds reduce candidate-specific luck.
            diagnostic = random.Random(
                diagnostic_seed + 700001 + 100003 * side)
            record["d"] = min(
                record["d"],
                kernel_probe(record["columns"], diagnostic,
                             64 if phase == 2 else 48))
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa = select_component(pools[0], 0)
    bb = select_component(pools[1], 1)

    def pair_merit(pair):
        a, b = pair
        ac, bc = a["cycles"], b["cycles"]
        return (min(a["d"], b["d"]), a["d"] + b["d"],
                -max(ac[0], bc[0]), -ac[0] - bc[0],
                -max(ac[1], bc[1]), -ac[1] - bc[1])

    pairs = sorted(((a, b) for a in aa for b in bb),
                   key=pair_merit, reverse=True)
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= q and 7 * n <= 40 * k < 9 * n):
            continue
        if not _no_kernel4(_syndrome_columns(hx, n)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        assert max(row.bit_count() for row in hx + hz) == 12
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "profiles": (pa, pb), "phase": phase,
            "component_readings": (a["d"], b["d"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
        }
    raise ValueError("no independently selected pair passed full screening")


def NEW_CTX(rng, G, idx):
    """Twenty degree-three cards on the new affine-group bed."""
    q = G.q
    assert q == 80
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    topology, objective = card // 4, card % 4
    seed = rng.randrange(1 << 62)

    # All column sums are three. Row sums distinguish the first two
    # cards; subsequent cards introduce inequivalent zero arrangements.
    b_topologies = (
        ((1, 1, 1, 2, 2), (2, 2, 2, 1, 1)),
        ((1, 2, 2, 2, 2), (2, 1, 1, 1, 1)),
        ((3, 1, 1, 2, 2), (0, 2, 2, 1, 1)),
        ((3, 0, 2, 2, 2), (0, 3, 1, 1, 1)),
        ((3, 3, 1, 1, 1), (0, 0, 2, 2, 2)),
    )
    supports = {
        "A": [[3, 3, 3]],
        "B": [list(row) for row in b_topologies[topology]],
    }
    sa, sb = supports["A"], supports["B"]
    assert all(sum(row[j] for row in sb) == 3 for j in range(5))
    assert max(map(sum, sb)) <= 9
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Native group operations are cached for label proposals only.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    ident = G.idx(G.e)
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    quotient = [g[1] for g in G.elements]

    # Every binary permutation block comes from lifted_matrices.
    # The zero partners are extraction devices, never submitted codes.
    raw_a, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, raw_b, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    blocks = {
        "A": [[(row >> (j*q)) & mask for row in raw_a]
              for j in range(q)],
        "B": [[(row >> (j*q)) & mask for row in raw_b]
              for j in range(q)],
    }
    del raw_a, raw_b

    mixed = [g for g in range(q) if quotient[g] != 0]

    def fresh_cell(count):
        if count == 0:
            return []
        anchor = rng.randrange(q)
        if count == 1:
            return [G.elements[anchor]]
        if count == 2:
            # A cross-complement ratio has order five, not order two.
            u = rng.choice(mixed)
            ids = (ident, u)
        else:
            assert count == 3
            for _ in range(512):
                u, v = rng.sample(mixed, 2)
                if quotient[u] == quotient[v]:
                    continue
                if table[u][v] == table[v][u]:
                    continue
                ids = (ident, u, v)
                break
            else:
                raise ValueError("no noncommuting cubic relatives")
            # Noncommuting relatives generate the whole affine group:
            # their nonzero normal commutator has an irreducible C5 orbit.
            reached, todo = {ident}, [ident]
            for x in todo:
                for y in (u, v):
                    z = table[x][y]
                    if z not in reached:
                        reached.add(z)
                        todo.append(z)
            assert len(reached) == q
        labels = [table[anchor][g] for g in ids]
        assert len(labels) == len(set(labels)) == count
        return [G.elements[g] for g in labels]

    def fresh(side):
        return [[fresh_cell(count) for count in row]
                for row in supports[side]]

    def assemble(grid, side):
        """GF2-linear assembly of authoritative monomial blocks."""
        monomials = blocks[side]
        rows = []
        for line in grid:
            indexed = [[G.idx(g) for g in cell] for cell in line]
            for t in range(q):
                row = 0
                for j, cell in enumerate(indexed):
                    block = 0
                    for g in cell:
                        block ^= monomials[g][t]
                    row |= block << (j*q)
                rows.append(row)
        return rows

    def six_cycles(rows, columns):
        """Exact simple Tanner C6 count using component regular symmetry."""
        neighbors = [list(_bits(c)) for c in columns]
        checks = [list(_bits(row)) for row in rows]
        rooted = 0
        for root in range(0, len(columns), q):
            root_checks = set(neighbors[root])
            for c1 in neighbors[root]:
                for v in checks[c1]:
                    if v == root:
                        continue
                    for c2 in neighbors[v]:
                        if c2 == c1:
                            continue
                        for end in checks[c2]:
                            if end == root or end == v:
                                continue
                            rooted += sum(
                                c3 != c1 and c3 != c2
                                for c3 in root_checks.intersection(
                                    neighbors[end]))
        # Three variable roots and two orientations per simple cycle.
        assert (q*rooted) % 6 == 0
        return q*rooted // 6

    def cycle_key(record):
        c4, c6 = record["cycles"]
        if objective == 0:
            return (c4, c6)
        if objective == 1:
            return (c6, c4)
        if objective == 2:
            return (4*c4 + c6, c4, c6)
        return (16*c4 + c6, c6, c4)

    def kernel_probe(record, diagnostic, trials):
        """Actual classical null words; sampled upper readings only."""
        columns = record["columns"]
        width = len(columns)
        order = list(range(width))
        best, witness = width + 1, None
        retained = []

        def note(word):
            nonlocal best, witness
            if word and word.bit_count() < best:
                best, witness = word.bit_count(), word

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word)

            light = sorted(set(basis + retained),
                           key=int.bit_count)[:20]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    note(candidate)
                    if candidate and candidate.bit_count() <= best + 4:
                        additions.append(candidate)

            moves = basis + retained
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    note(word)
                    if not changed:
                        break
                additions.append(word)

            if len(basis) >= 3:
                for _ in range(96):
                    a, b, c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:24]
            if best <= 6:
                break

        if witness is None:
            raise ValueError("component unexpectedly has no kernel")
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])
        return best, witness

    def component_pool(side):
        width = len(supports[side][0])*q
        side_id = int(side == "B")
        records, signatures = [], set()

        for attempt in range(3):
            for trial in range(40 + 4*(cohort % 3)):
                grid = fresh(side)
                signature = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                rows = assemble(grid, side)
                quality = _component_quality(rows, width)
                if quality is None:
                    continue
                stats, columns = quality
                assert all(c.bit_count() == 3 for c in columns)
                records.append({
                    "grid": grid, "rows": rows, "columns": columns,
                    "cycles": (stats[0], six_cycles(rows, columns)),
                    "trial": len(signatures),
                })
            records.sort(key=lambda r: (cycle_key(r), r["trial"]))

            selected = []
            for record in records:
                rows = record["rows"]
                rank = gf2_rank(rows)
                if rank < len(rows) - 5:
                    continue
                if side == "A":
                    cell_ranks = tuple(
                        gf2_rank([(row >> (j*q)) & mask for row in rows])
                        for j in range(3))
                    # Every trinomial is a unit on this particular group.
                    assert cell_ranks == (q, q, q)
                    assert rank == q
                    record["cell_ranks"] = cell_ranks
                if not _no_kernel4(record["columns"]):
                    continue
                record["rank"] = rank
                d, word = kernel_probe(
                    record, random.Random(seed + 100003*side_id), 8)
                record["d"], record["word"] = d, word
                selected.append(record)
                if len(selected) == 6:
                    break
            if not selected:
                continue

            def merit(record):
                return (record["d"],
                        tuple(-x for x in cycle_key(record)),
                        record["rank"])

            selected.sort(key=merit, reverse=True)
            finalists = selected[:3]
            for record in finalists:
                for offset, trials in (
                        (700001, 32 + 8*(cohort % 3)),
                        (1900003, 24)):
                    d, word = kernel_probe(
                        record,
                        random.Random(seed + offset + 100003*side_id),
                        trials)
                    record["d"] = min(record["d"], d)
                    if word.bit_count() < record["word"].bit_count():
                        record["word"] = word
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("no legal connected degree-three component")

    aa, bb = component_pool("A"), component_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"] + ab[1]["d"],
        tuple(-(x+y) for x, y in
              zip(cycle_key(ab[0]), cycle_key(ab[1])))),
        reverse=True)

    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 17*q == 1360
        actual_a, actual_b = _verified_components(hx, hz, q)
        assert actual_a == a["rows"] and actual_b == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= 6*q and 13*n <= 40*k < 15*n):
            continue
        if not _no_kernel4(_syndrome_columns(hx, n)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, n)):
            continue
        assert _grid_weights(sa, sb) <= 12
        assert max(row.bit_count() for row in hx + hz) <= 12
        if not css_ok(hx, hz):
            raise ValueError("authoritative affine lift failed CSS")
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (topology, objective, cohort),
            "component_readings": (a["d"], b["d"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "cubic_cell_ranks": a["cell_ranks"],
        }
    raise ValueError("new affine pairings missed full validity/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected new-group cell without randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
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
