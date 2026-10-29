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
CONCEPT = ("Hypothesis: C61 semidirect_13 C3, order 183, A=(1,2), B=(2,3), with coupled repeated-fiber gaps in four/five-term A cells and quotient-rank-controlled B components improves five-code fitness beyond 270.7344 in cell 0.150. Five measured readings >=48 are the falsifiable target; all finite-search readings remain provisional upper bounds.")
GROUP = ("Jump from order-129 C43:C3 to order-183 C61:C3(s=13). Different orders establish a genuine nonisomorphic group change. This order-183 bed has reached fitness 270.7, whereas the parent retained abundant weight-twelve sector-one logicals at fitness 58.2256. Keep a faithful order-three action and a large prime normal subgroup, without claiming distance transfer or a certified distance.")
SHAPES = ("Change A to (1,2) and B to (2,3): n=1464, k>=183 and guaranteed rate 1/8, the lower boundary of cell 0.150. The shape permits q<=187. Unequal components have 366 variables/183 checks and 549 variables/366 checks, nominal rates 1/2 and 1/3. Parent minima were entirely sector one, concentrated on Acol1/2 and Bcol1/2 strips. Replace three cubic A columns by two denser columns; authoritative full ranks enforce 0.125<=k/n<0.175.")
FILLING = ("A counts (4,5), B counts ((1,1,1),(2,2,2)) give HX=9+3=12 and HZ<=6+5=11. Twenty cards couple the two repeated-fiber gaps of the five-term A cell to the four-term cell's gap using distinct unordered classes in F61*/<-1,13>. Both cells are bidirectionally Sidon and have disjoint right differences; actual lifted rows independently verify zero A four-cycles. B has singleton identity cells above mixed (normal, complement) binomials, affine normal-root triples and verified rank 363. Independently select components using actual unrestricted and C3-fixed witnesses, restricted four-circuit cancellation, witness-directed normal-label repair and fresh retests. Original full matrices challenge localized nontrivial logicals with original stabilizers. Full ranks, kernel-four, weights, native-component agreement and CSS gate submission. Finite witness searches and cycle objectives do not certify distance.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m61r3s13_12x23_coupled_gapclasses_Bquotientrank_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 183

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C61 semidirect_13 C3 requires order 183")
    m, r = 61, 3
    action = (1, 13, 47)
    assert action[2] == 13*13 % m
    assert action[2]*13 % m == 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        return ((g[0] + action[g[1]]*h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        b = (-g[1]) % r
        return ((-action[b]*g[0]) % m, b)

    def idx(g):
        a, b = red(g)
        return a + m*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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
    """Historical order-291 filling; never called by the new NEW_CTX."""
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


def _retired_trinomial_ctx(rng, G, idx):
    """Inactive historical order-129 portfolio; NEW_CTX below replaces it."""
    q = G.q
    width = 3 * q
    mask = (1 << q) - 1
    assert q == 129 and A_SHAPE == B_SHAPE == (1, 3)
    support = [[3, 3, 3]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    card, cohort = idx % 20, idx // 20
    diagnostic_seed = rng.randrange(1 << 62)

    # Relative-order spectra of these occupancies differ:
    # mixed-111: (3,3,3), mixed-210: (3,3,43), pure: (43,43,43).
    # There is never a pair of pure cells in one component.
    occupancies = ((1, 1, 1), (2, 1, 0), (3, 0, 0))
    profiles = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (0, 0, 2),
        (0, 1, 2),
    )
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    assert len(plans) == 20
    pa, pb = plans[card]
    trade = cohort % 3

    # Cache native group operations, not a separately implemented lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverse[g]

    P = Probe()

    # Batches keep unused dummy matrices small. Every permutation block
    # comes from the scaffold; subsequent assembly is GF2 linearity only.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j * q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j * q)) & mask for row in zz])

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for cell in grid[0])

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def expand(word, groups):
        actual = 0
        for c in _bits(word):
            actual ^= groups[c]
        return actual

    # Only coordinate restrictions are formed here. Their syndromes are
    # XORs of authoritative columns, and expanded words are verified.
    h = G.idx((0, 1))
    assert table[table[h][h]][h] == identity
    partitions = {}
    for side in ("A", "B"):
        remaining = set(range(q))
        orbits = []
        while remaining:
            root = min(remaining)
            orbit = [root]
            for _ in range(2):
                previous = orbit[-1]
                orbit.append(table[previous][h] if side == "A"
                             else table[h][previous])
            assert len(set(orbit)) == 3 and set(orbit) <= remaining
            remaining.difference_update(orbit)
            orbits.append(orbit)
        assert len(orbits) == 43
        partitions[side] = [
            sum(1 << (j * q + c) for c in orbit)
            for j in range(3) for orbit in orbits
        ]

    def sidon(cell):
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverse[a]][b])
                    right.add(table[a][inverse[b]])
        return (len(cell) == len(set(cell)) == 3
                and len(left) == len(right) == 6)

    def fresh_cell(kind):
        for _ in range(1024):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, count in zip(fibers, occupancies[kind]):
                cell.extend(G.idx((a, b))
                            for a in rng.sample(range(43), count))
            if sidon(cell):
                return cell
        raise ValueError("trinomial occupancy could not satisfy Sidon gate")

    def fresh(profile):
        return [[fresh_cell(kind) for kind in profiles[profile]]]

    def small_word(columns):
        """Exact detection of kernel words through weight four.

        A returned word is genuine; None excludes all weights 1..4.
        The first returned word need not be the minimum at weights 3..4.
        """
        singles = {}
        for c, value in enumerate(columns):
            bit = 1 << c
            if not value:
                return bit
            if value in singles:
                return bit ^ singles[value]
            singles[value] = bit
        pairs = {}
        for c, value in enumerate(columns):
            for d in range(c):
                v = value ^ columns[d]
                word = (1 << c) | (1 << d)
                old = singles.get(v)
                if old is not None:
                    return word ^ old
                old = pairs.get(v)
                if old is not None:
                    return word ^ old
                pairs[v] = word
        return None

    def cycle_key(cycles):
        four, excess = cycles
        if trade == 0:
            return four, excess
        if trade == 1:
            return 8 * four + excess, four
        return 24 * four + excess, excess

    def inspect(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j * q)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        assert all(column.bit_count() == 3 for column in columns)
        rank = gf2_rank(rows)
        if rank < q - 2:
            return None
        if small_word(columns) is not None:
            return None

        groups = partitions[side]
        fixed_columns = [syndrome(columns, group) for group in groups]
        # Each orbit sums three odd-degree columns, so its parity is odd.
        assert all(value.bit_count() & 1 for value in fixed_columns)
        fw = small_word(fixed_columns)
        words = []
        fixed_beam = []
        fixed_d = width + 1
        if fw is not None:
            actual = expand(fw, groups)
            assert actual.bit_count() == 3 * fw.bit_count()
            assert syndrome(columns, actual) == 0
            words.append(actual)
            fixed_beam.append(fw)
            fixed_d = actual.bit_count()

        return {
            "grid": copy_grid(grid), "signature": signature(grid),
            "rows": rows, "columns": columns, "rank": rank,
            "cycles": cycles, "cycle_key": cycle_key(cycles),
            "groups": groups, "fixed_columns": fixed_columns,
            "fixed_beam": fixed_beam, "beam": words,
            "fixed_small_free": fw is None,
            "fixed_d": fixed_d,
            "d": fixed_d,
            "word": words[0] if words else None,
        }

    def kernel_search(columns, seed, trials, seeds=(), stop=8):
        """Accumulated genuine witnesses from independently changed bases."""
        local = random.Random(seed)
        count = len(columns)
        beam = sorted({word for word in seeds if word},
                      key=int.bit_count)[:20]
        best_word = None

        def observe(word):
            nonlocal best_word
            if word and (best_word is None or
                         word.bit_count() < best_word.bit_count()):
                best_word = word

        for word in beam:
            assert syndrome(columns, word) == 0
            observe(word)

        order = list(range(count))
        for turn in range(trials):
            local.shuffle(order)
            if turn % 5 == 4 and beam:
                anchor = local.choice(beam[:6])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
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
                    observe(word)
            if not basis:
                continue

            light = sorted(set(basis + beam), key=int.bit_count)[:12]
            grown = list(light) + beam
            for a in light:
                for b in basis:
                    candidate = a ^ b
                    observe(candidate)
                    if (candidate and candidate.bit_count()
                            <= best_word.bit_count() + 6):
                        grown.append(candidate)
                if len(grown) > 160:
                    grown = sorted(set(grown), key=int.bit_count)[:32]

            moves = basis + beam
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if (candidate and candidate.bit_count()
                                < word.bit_count()):
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
                grown.append(word)

            shortened = sorted(set(grown), key=int.bit_count)[:20]
            for i, a in enumerate(shortened):
                for b in shortened[:i]:
                    observe(a ^ b)
            grown.append(best_word)
            beam = sorted({word for word in grown if word},
                          key=int.bit_count)[:20]
            if best_word.bit_count() <= stop:
                break

        if best_word is None:
            raise ValueError("no component kernel witness found")
        assert syndrome(columns, best_word) == 0
        return best_word, beam

    def parity_completion(record, seed, budget=20000):
        """Budgeted search through ten; exhaustion is not exclusion."""
        local = random.Random(seed)
        columns = record["columns"]
        best = record["word"]
        cap = min(10, best.bit_count() - 1)
        if cap < 5:
            return best
        singles, pairs = {}, {}
        incidence = [0] * q
        for c, value in enumerate(columns):
            bit = 1 << c
            singles.setdefault(value, bit)
            for r in _bits(value):
                incidence[r] |= bit
            for d in range(c):
                pairs.setdefault(value ^ columns[d],
                                 bit | (1 << d))

        priorities = [local.random() for _ in columns]
        roots = [j * q for j in range(3)]
        local.shuffle(roots)
        all_variables = (1 << width) - 1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                assert syndrome(columns, word) == 0
                best = word

        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    observe(word)
                    return
                left = min(cap, best.bit_count() - 1) - word.bit_count()
                if left <= 0 or value.bit_count() > 3 * left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    observe(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    observe(word | pair)
                left = min(cap, best.bit_count() - 1) - word.bit_count()
                if left <= 2:
                    return

                chosen, smallest = 0, width + 1
                for r in _bits(value):
                    candidates = incidence[r] & allowed
                    size = candidates.bit_count()
                    if not size:
                        return
                    if size < smallest:
                        chosen, smallest = candidates, size
                candidates = list(_bits(chosen))
                candidates.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            visit(columns[root], 1 << root,
                  all_variables ^ (1 << root))
            if best.bit_count() <= 6:
                break
        return best

    def component_pool(profile, side):
        offset = 100003 * int(side == "B")
        bank, seen = set(), set()

        def probe(record, turn, full_trials, fixed_trials):
            transferred = [
                word for word in bank
                if syndrome(record["columns"], word) == 0
            ]
            word, beam = kernel_search(
                record["columns"],
                diagnostic_seed + offset + 700001 * turn,
                full_trials, record["beam"] + transferred)
            record["beam"] = beam
            fw, fixed_beam = kernel_search(
                record["fixed_columns"],
                diagnostic_seed + offset + 1700003 + 900001 * turn,
                fixed_trials, record["fixed_beam"], stop=4)
            record["fixed_beam"] = fixed_beam
            actual = expand(fw, record["groups"])
            assert actual.bit_count() == 3 * fw.bit_count()
            assert syndrome(record["columns"], actual) == 0
            assert all((row & actual).bit_count() % 2 == 0
                       for row in record["rows"])
            record["fixed_d"] = min(
                record["fixed_d"], actual.bit_count())
            for witness in (word, actual):
                bank.add(witness)
                if witness.bit_count() < record["d"]:
                    record["d"] = witness.bit_count()
                    record["word"] = witness
            record["beam"] = sorted(
                set(record["beam"] + [actual, record["word"]]),
                key=int.bit_count)[:20]
            if len(bank) > 80:
                keep = sorted(bank, key=int.bit_count)[:56]
                bank.clear()
                bank.update(keep)

        def merit(record):
            for word in bank:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"], record["word"] = word.bit_count(), word
            return (record["d"], record["fixed_d"],
                    tuple(-x for x in record["cycle_key"]),
                    record["rank"])

        for restart in range(3):
            records = []
            for _ in range(64 + 8 * min(cohort, 2)):
                grid = fresh(profile)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                record = inspect(grid, side)
                if record is not None:
                    records.append(record)
            records.sort(key=lambda record: (
                not record["fixed_small_free"], record["cycle_key"]))

            selected = records[:8]
            if not selected:
                continue
            for record in selected:
                probe(record, 1 + 300 * restart, 8, 16)
            current = max(selected, key=merit)
            floor4 = min(record["cycles"][0] for record in selected)

            # Change actual terms, never apply a common gauge or automorphism.
            # The card's three-cell relative-order composition is preserved.
            for step in range(18):
                used = list(_bits(current["word"]))
                j = (rng.choice(used) // q
                     if used and rng.random() < 0.85 else rng.randrange(3))
                grid = copy_grid(current["grid"])
                if step % 6 == 5:
                    grid = fresh(profile)
                else:
                    grid[0][j] = fresh_cell(profiles[profile][j])
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                if proposed["cycles"][0] > floor4 + 2 * q:
                    continue
                probe(proposed, 11 + step + 300 * restart, 4, 8)
                selected.append(proposed)
                pk, ck = merit(proposed), merit(current)
                if pk > ck or (pk[0] == ck[0] and rng.random() < 0.10):
                    current = proposed

            finalists = sorted(selected, key=merit, reverse=True)[:4]
            for record in finalists:
                probe(record, 101 + 300 * restart, 24, 48)
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:3]
            for record in finalists:
                word = parity_completion(
                    record, diagnostic_seed + offset + 2700011,
                    20000 + 4000 * min(cohort, 2))
                bank.add(word)
                if word.bit_count() < record["d"]:
                    record["d"], record["word"] = word.bit_count(), word
                record["beam"].append(word)
                # Independent retesting retains every earlier witness.
                probe(record, 211 + 300 * restart,
                      64 + 8 * min(cohort, 2), 96)
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("no trinomial component passed authoritative gates")

    aa = component_pool(pa, "A")
    bb = component_pool(pb, "B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"] + ab[1]["d"],
        min(ab[0]["fixed_d"], ab[1]["fixed_d"]),
        tuple(-(x + y) for x, y in
              zip(ab[0]["cycle_key"], ab[1]["cycle_key"]))),
        reverse=True)

    for a, b in pairs:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 10 * q == 1290
        ar, br = _verified_components(hx, hz, q)
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= 4 * q and 15 * n <= 40 * k < 17 * n):
            continue
        if not _no_kernel4(_syndrome_columns(hx, n)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, n)):
            continue
        assert max(row.bit_count() for row in hx + hz) == 12
        assert _grid_weights(support, support) == 12
        assert css_ok(hx, hz)
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (pa, pb, trade),
            "component_readings": (a["d"], b["d"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "component_ranks": (a["rank"], b["rank"]),
        }
    raise ValueError("trinomial pairings missed full validity or rate gates")


def NEW_CTX(rng, G, idx):
    """Twenty coupled-gap cards; every binary action comes from the scaffold."""
    q, mask = G.q, (1 << G.q)-1
    assert q == 183 and A_SHAPE == (1, 2) and B_SHAPE == (2, 3)
    sa, sb = [[4, 5]], [[1, 1, 1], [2, 2, 2]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    card, cohort = idx % 20, idx // 20
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0x61C03A25 + 1000003*idx))

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    P = Probe()
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start+16)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    def copy(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    # Native group arithmetic proposes labels, never a substitute lift.
    signed_action = {(sign*pow(13, b, 61)) % 61
                     for sign in (1, -1) for b in range(3)}
    primitive = next(a for a in range(2, 61)
                     if len({pow(a, t, 61) for t in range(60)}) == 60)
    gap_classes = tuple(
        tuple(sorted({pow(primitive, t, 61)*z % 61
                      for z in signed_action}))
        for t in range(10))
    assert len(signed_action) == 6
    assert len(set().union(*(set(c) for c in gap_classes))) == 60
    plans = [(a, b) for a in range(1, 10) for b in range(a+1, 10)][:20]
    ga, gb = plans[card]
    triangle = (3, 5, 7, 11)[(card+cohort) % 4]

    def differences(cell, right=False):
        return {table[a][inverses[b]] if right
                else table[inverses[a]][b]
                for a in cell for b in cell if a != b}

    def sidon(cell):
        wanted = len(cell)*(len(cell)-1)
        return (len(set(cell)) == len(cell)
                and len(differences(cell)) == wanted
                and len(differences(cell, True)) == wanted)

    def legal_a(grid):
        first, second = grid[0]
        return (sidon(first) and sidon(second)
                and differences(first, True).isdisjoint(
                    differences(second, True)))

    def fresh_a():
        for _ in range(2048):
            gap = rng.randrange(1, 61)
            t, s = rng.choice(gap_classes[ga]), rng.choice(gap_classes[gb])
            b, c, u, v, w = [rng.randrange(61) for _ in range(5)]
            grid = [[
                [ident, G.idx((gap, 0)), G.idx((b, 1)), G.idx((c, 2))],
                [G.idx((u, 0)), G.idx(((u+t*gap) % 61, 0)),
                 G.idx((v, 1)), G.idx(((v+s*gap) % 61, 1)),
                 G.idx((w, 2))],
            ]]
            if legal_a(grid):
                return grid
        raise ValueError("coupled Sidon gap classes exhausted")

    def fresh_b():
        origin, scale = rng.randrange(61), rng.randrange(1, 61)
        roots = [(origin+scale*t) % 61 for t in (0, 1, triangle)]
        other = rng.sample(range(61), 3)
        # All quotient columns are (1, 1+y)^T. Actual binary ranks
        # below, rather than a quotient prediction, decide acceptance.
        return [[[ident] for _ in range(3)],
                [[G.idx((a, 0)), G.idx((b, 1))]
                 for a, b in zip(roots, other)]]

    # Restrictions are sums of authoritative columns along actual
    # regular-action orbits, not separately constructed matrices.
    h = G.idx((0, 1))
    orbit_masks = {}
    for side, nc in (("A", 2), ("B", 3)):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {root}
            value = root
            for _ in range(2):
                value = table[value][h] if side == "A" else table[h][value]
                orbit.add(value)
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        assert len(orbits) == 61
        orbit_masks[side] = [
            sum(1 << (j*q+c) for c in orbit)
            for j in range(nc) for orbit in orbits]

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i*q+t] ^= value << (j*q)
        quality = _component_quality(rows, nc*q)
        if quality is None:
            return None
        cycles, columns = quality
        rank = gf2_rank(rows)
        if side == "A":
            if not legal_a(grid) or cycles[0] != 0 or rank != q:
                return None
            low = _component_quality([row & mask for row in rows], q)
            low_six = low[0][1] if low is not None else cycles[1]
            cost = ((2, 4, 8, 16)[card % 4]*low_six + cycles[1],
                    cycles[1], low_six)
        else:
            if rank != 2*q-3:
                return None
            cost = (cycles[0], cycles[1], 0)
        groups = orbit_masks[side]
        fixed = [syndrome(columns, word) for word in groups]
        return dict(grid=copy(grid), rows=rows, columns=columns,
                    rank=rank, cycles=cycles, cost=cost,
                    signature=signature(grid), groups=groups,
                    fixed_columns=fixed, words=[], fixed_words=[],
                    d=nc*q+1, fixed_d=nc*q+1, word=None)

    def search(columns, diagnostic_seed, trials, seeds=(), stern=False):
        """Find genuine kernel words; exhaustion never means exclusion."""
        local = random.Random(diagnostic_seed)
        count = len(columns)
        retained = sorted({w for w in seeds if w},
                          key=lambda w: (w.bit_count(), w))[:24]
        best = None

        def note(word):
            nonlocal best
            if word and (best is None or word.bit_count() < best.bit_count()):
                best = word

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)
        order = list(range(count))
        for turn in range(trials):
            local.shuffle(order)
            if retained and turn % 4 == 3:
                anchor = local.choice(retained[:8])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
            pivots, basis, independent = {}, [], []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        independent.append(c)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word)
            if not basis:
                continue
            light = sorted(set(basis+retained),
                           key=lambda w: (w.bit_count(), w))[:16]
            grown = list(light)+retained
            for first in light:
                for second in basis:
                    candidate = first ^ second
                    note(candidate)
                    if candidate and candidate.bit_count() <= best.bit_count()+4:
                        grown.append(candidate)
                if len(grown) > 160:
                    grown = sorted(set(grown), key=int.bit_count)[:32]

            if stern and len(basis) <= 120 and independent:
                sample = local.sample(independent, min(8, len(independent)))
                hash_mask = sum(1 << c for c in sample)
                buckets = {}
                for i, first in enumerate(basis):
                    for second in basis[:i]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for previous in bucket:
                            candidate = pair ^ previous
                            note(candidate)
                            if (candidate and candidate.bit_count()
                                    <= best.bit_count()+2):
                                grown.append(candidate)
                        bucket.append(pair)
                    if len(grown) > 256:
                        grown = sorted(set(grown), key=int.bit_count)[:32]

            moves = basis+retained
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    note(word)
                    if not changed:
                        break
                grown.append(word)
            retained = sorted(set(grown+[best]),
                              key=lambda w: (w.bit_count(), w))[:24]
            if best.bit_count() <= (4 if stern else 12):
                break
        if best is None:
            raise ValueError("verified component had no sampled kernel word")
        assert syndrome(columns, best) == 0
        return best, retained

    def query(rec, side, turn, trials, fixed_trials):
        offset = 100003*int(side == "B")
        word, words = search(
            rec["columns"], seed+offset+700001*turn,
            trials, rec["words"])
        fixed_word, fixed_words = search(
            rec["fixed_columns"], seed+offset+1700003+900001*turn,
            fixed_trials, rec["fixed_words"], stern=fixed_trials >= 24)
        actual = 0
        for c in _bits(fixed_word):
            actual ^= rec["groups"][c]
        assert actual.bit_count() == 3*fixed_word.bit_count()
        assert syndrome(rec["columns"], actual) == 0
        for candidate in (word, actual):
            if rec["word"] is None or candidate.bit_count() < rec["d"]:
                rec["word"], rec["d"] = candidate, candidate.bit_count()
        rec["fixed_d"] = min(rec["fixed_d"], actual.bit_count())
        rec["fixed_words"] = fixed_words
        rec["words"] = sorted(set(words+[actual, rec["word"]]),
                              key=lambda w: (w.bit_count(), w))[:24]

    def merit(rec):
        return (rec["d"], rec["fixed_d"],
                tuple(-x for x in rec["cost"]), rec["rank"])

    def repair(rec, side):
        grid = copy(rec["grid"])
        occupied = list(_bits(rec["word"]))
        j = rng.choice(occupied)//q if occupied else rng.randrange(len(grid[0]))
        if side == "A":
            if j == 0:
                t = rng.choice((2, 3))
                old = G.elements[grid[0][0][t]]
                grid[0][0][t] = G.idx((rng.randrange(61), old[1]))
            else:
                fiber = rng.randrange(3)
                if fiber < 2:
                    positions = (0, 1) if fiber == 0 else (2, 3)
                    shift = rng.randrange(1, 61)
                    for t in positions:
                        old = G.elements[grid[0][1][t]]
                        grid[0][1][t] = G.idx(((old[0]+shift) % 61, old[1]))
                else:
                    grid[0][1][4] = G.idx((rng.randrange(61), 2))
        else:
            # Preserve the affine first-root triple and quotient support.
            grid[1][j][1] = G.idx((rng.randrange(61), 1))
        return grid

    def pool(side):
        seen, records = set(), []
        fresh = fresh_a if side == "A" else fresh_b
        for _ in range(96 if side == "A" else 80):
            grid = fresh()
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            rec = inspect(grid, side)
            if rec is not None:
                records.append(rec)
        records.sort(key=lambda r: r["cost"])
        selected = []
        for rec in records:
            if not _no_kernel4(rec["columns"]):
                continue
            query(rec, side, 1, 8, 12)
            selected.append(rec)
            if len(selected) == 8:
                break
        if not selected:
            return []
        current = max(selected, key=merit)
        for step in range(16):
            grid = fresh() if step % 5 == 4 else repair(current, side)
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            rec = inspect(grid, side)
            if rec is None or not _no_kernel4(rec["columns"]):
                continue
            query(rec, side, 11+step, 6, 8)
            selected.append(rec)
            if merit(rec) > merit(current):
                current = rec
        finalists = sorted(selected, key=merit, reverse=True)[:4]
        for rec in finalists:
            query(rec, side, 101, 32, 32)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:3]
        for rec in finalists:
            # Independent tests retain every earlier binding witness.
            query(rec, side, 211, 80+8*min(cohort, 2), 64)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(a, b, hx, hz, n):
        """Challenge exact logicals in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            located = set()
            for word in rec["words"][:4]:
                assert syndrome(rec["columns"], word) == 0
                for location in range(3 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j+location)*q) for j in range(2))
                    else:
                        actual = word << (location*3*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        located.add(actual)
            reading = n+1
            local = random.Random(seed+(3100009 if kind == "Z" else 4100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(located, key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    neighborhood = sorted(
                        stabilizers, key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    winner = actual
                    for i, first in enumerate(neighborhood):
                        for second in neighborhood[:i]:
                            candidate = actual ^ first ^ second
                            if candidate.bit_count() < winner.bit_count():
                                winner = candidate
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    for restart in range(2):
        aa, bb = pool("A"), pool("B")
        survivors = []
        for a in aa:
            for b in bb:
                A = [[[G.elements[g] for g in cell] for cell in row]
                     for row in a["grid"]]
                B = [[[G.elements[g] for g in cell] for cell in row]
                     for row in b["grid"]]
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 1464
                ar, br = _verified_components(hx, hz, q)
                assert ar == a["rows"] and br == b["rows"]
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (k >= q and 125*n <= 1000*k < 175*n):
                    continue
                logicals = logical_readings(a, b, hx, hz, n)
                # An optimism cap is a selection policy, not a lower bound.
                detected = min(52, a["d"], b["d"], *logicals)
                key = (k*detected*detected, detected,
                       min(a["d"], b["d"]),
                       tuple(-(x+y) for x, y in zip(a["cost"], b["cost"])))
                survivors.append((key, A, B, hx, hz, k, a, b, logicals))
        survivors.sort(key=lambda entry: entry[0], reverse=True)
        for key, A, B, hx, hz, k, a, b, logicals in survivors:
            if not _no_kernel4(_syndrome_columns(hx, 1464)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, 1464)):
                continue
            assert [[len(c) for c in row] for row in A] == sa
            assert [[len(c) for c in row] for row in B] == sb
            assert max(row.bit_count() for row in hx+hz) <= 12
            assert css_ok(hx, hz)
            return dict(
                A=A, B=B, idx=idx, k=k, card=(ga, gb, triangle, cohort),
                target_cell=0.150, quality=key,
                component_readings=(a["d"], b["d"]),
                C3_fixed_readings=(a["fixed_d"], b["fixed_d"]),
                component_ranks=(a["rank"], b["rank"]),
                component_cycles=(a["cycles"], b["cycles"]),
                verified_localized_logical_readings=logicals,
                gap_class_policy=(ga, gb),
                B_quotient_policy="three columns (1,1+y)^T")
    raise ValueError("coupled-gap pairs missed authoritative validity/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected dense-A or quotient-controlled-B cell."""
    cell = list(ctx[side][i][j])
    assert 1 <= len(cell) <= 5
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term within-coset Sidon supports and independent component descent uniformly suppress sub-30 logical readings: the measured parent still screened slots 18, 26 and 48 at 27, 24 and 21.
# REFUTED: The order-129 cubic occupancy portfolio improves cell-0.400 fitness beyond 58.9: it measured 58.2256, with all five selected readings twelve and abundant sector-one minimum witnesses.
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
