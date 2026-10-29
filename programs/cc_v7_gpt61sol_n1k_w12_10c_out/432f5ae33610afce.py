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
CONCEPT = ("On (C7 semidirect_2 C3) x C4, order 84, A=(1,3), B=(2,5), verified common C3-quotient annihilators on cubic A components plus independently challenged cubic-column B components should produce five new k>=510 codes with measured readings >=12, improving cell 0.350 beyond fitness 50.8235. Failure of this five-code target refutes the experiment; readings remain provisional upper bounds.")
GROUP = ("Jump from the parent's order-129 Z43:C3 to the order-84 direct product (C7 semidirect_2 C3) x C4. Different orders establish a genuine group change. This product is isomorphic to the run's C7:C12(s=2), whose (1,3)x(2,5) bed reached 50.8. That measured bed motivates a new rank-gain filling, not certified distance transfer. Its earlier codes are archived; no shown filling is reused.")
SHAPES = ("Keep A=(1,3), change B to (2,5). Factor seventeen gives n=1428, k>=504 and guaranteed rate 6/17=0.352941; the size-law lift maximum is 88 and this group uses q=84. Unequal components have 252 variables/84 checks and 420 variables/168 checks. Parent minima were entirely sector one, especially Acol1/2 across B columns and Bcol1/2 across A columns. Preserve cubic column degrees but test unequal components and actual k>=510 in [0.325,0.375).")
FILLING = ("A counts (3,3,3); B column totals three and row totals seven/eight give HX=9+3=12 and HZ<=8+3=11. Twenty cards cross five relative C4-coordinate prescriptions with four B support profiles distinguished by their numbers of concentrated cubic columns. Every A cell meets the C3 quotient equally; two explicit common row functionals are verified on authoritative monomial blocks. Independently select and repair both components using exact kernel-four exclusion, retained genuine circuit witnesses, bounded parity completion and independent retests. Original stabilizer-span tests challenge localized logicals before pairing by actual k times a squared reading capped at twelve. Full ranks, native component agreement, CSS, weights and the target rate decide acceptance. Neither finite searches nor component exclusions certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Frob21xC4_13x25_balancedA_rankgain_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius21 x C4 requires order 84")
    # (a,b,z) = x^a y^b times a central C4 coordinate.
    # y*x*y^-1=x^2, with 2^3=1 modulo seven.
    action = (1, 2, 4)
    els = [(a,b,z) for z in range(4)
           for b in range(3) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 3, int(g[2]) % 4)

    def mul(g, h):
        a,b,z = red(g)
        c,d,w = red(h)
        return ((a + action[b]*c) % 7,
                (b+d) % 3, (z+w) % 4)

    def inv(g):
        a,b,z = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 7, bi, (-z) % 4)

    def idx(g):
        a,b,z = red(g)
        return 7*(3*z+b)+a

    return dict(elements=els, ident=(0,0,0),
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


def _legacy_balanced_ctx(rng, G, idx):
    """Unused historical order-129 optimizer; NEW_CTX below replaces it."""
    q = G.q
    width = 3 * q
    mask = (1 << q) - 1
    assert q == 129 and A_SHAPE == B_SHAPE == (1, 3)
    support = [[3, 3, 3]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    card, cohort = idx % 20, idx // 20
    diagnostic_seed = rng.randrange(1 << 62)

    # All cells meet the three quotient cosets equally. Their relative
    # commutators instead distinguish the construction cards.
    #
    # 3 generates F43*: its seventh power is -6. The subgroup <-1,6>
    # has six elements, so there are seven multiplicative classes.
    # Changing the root/order of a normalized balanced triple multiplies
    # its commutator by this subgroup. A global group automorphism shifts
    # all classes together; it does not change their relative composition.
    signed_action = {
        (sign * pow(6, b, 43)) % 43
        for sign in (1, -1) for b in range(3)
    }
    commutator_classes = tuple(
        tuple(sorted({pow(3, t, 43) * z % 43 for z in signed_action}))
        for t in range(7)
    )
    assert len(signed_action) == 6
    assert len(set().union(*(set(c) for c in commutator_classes))) == 42
    profiles = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 1, 2),
        (0, 1, 3),
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

    def commutator(cell):
        # Native multiplication supplies this invariant, not a new lift.
        root = inverse[cell[0]]
        u = table[root][cell[1]]
        v = table[root][cell[2]]
        return table[table[u][v]][inverse[table[v][u]]]

    def fresh_cell(kind):
        for _ in range(128):
            a = rng.randrange(43)
            value = rng.choice(commutator_classes[kind])
            delta = (26 * value) % 43  # 26 = 5^-1 mod 43.
            cell = [
                identity,
                G.idx((a, 1)),
                G.idx(((7 * a + delta) % 43, 2)),
            ]
            # For u=(a,1), v=(7a+delta,2), [u,v]=(5delta,0).
            # Nonzero commutator generates the prime normal subgroup;
            # u also generates the quotient, hence the pair generates G.
            assert commutator(cell) == G.idx((value, 0))
            if sidon(cell):
                return cell
        raise ValueError("balanced commutator class failed Sidon gate")

    def fresh(profile):
        for _ in range(128):
            row = [fresh_cell(kind) for kind in profiles[profile]]
            if (len({cell[1] for cell in row}) == 3
                    and len({commutator(cell) for cell in row}) == 3):
                return [row]
        raise ValueError("distinct balanced component geometries exhausted")

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
        expected = profiles[pa if side == "A" else pb]
        commutators = [commutator(cell) for cell in grid[0]]
        if (len({cell[1] for cell in grid[0]}) != 3
                or len(set(commutators)) != 3):
            return None
        for cell, value, kind in zip(grid[0], commutators, expected):
            assert cell[0] == identity
            assert tuple(G.elements[g][1] for g in cell) == (0, 1, 2)
            assert G.elements[value][0] in commutator_classes[kind]
            assert G.elements[value][1] == 0
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

    def localized_logical_challenge(a, b, hx, hz, n):
        """Found nontrivial logicals give upper readings, never exclusions.

        Component words are placed in the ORIGINAL qubit coordinates.
        Exact commutation and original stabilizer-span membership decide
        whether each placement is a logical before any weight reduction.
        """
        def span(rows):
            pivots = {}
            for word in rows:
                while word:
                    lead = word.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = word
                        break
                    word ^= old
            return pivots

        def outside(word, pivots):
            while word:
                old = pivots.get(word.bit_length() - 1)
                if old is None:
                    return True
                word ^= old
            return False

        readings = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(
                set(record["beam"] + [record["word"]]),
                key=lambda word: (word.bit_count(), word))[:6]
            candidates = set()
            for word in words:
                assert syndrome(record["columns"], word) == 0
                for location in range(3):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j + location)*q)
                            for j in range(3))
                    else:
                        actual = word << (location * width)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            reading = n + 1
            local = random.Random(
                diagnostic_seed + (4100011 if side == "A" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed

                    # Exhaust all pairs in an overlap-selected neighborhood.
                    # This is only a bounded reduction challenge, not an
                    # exhaustive search of the stabilizer equivalence class.
                    neighborhood = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    best_word = actual
                    best_weight = actual.bit_count()
                    for i, first in enumerate(neighborhood):
                        partial = actual ^ first
                        for second in neighborhood[:i]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word, best_weight = proposed, weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa = component_pool(pa, "A")
    bb = component_pool(pb, "B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(12, ab[0]["d"], ab[1]["d"]),
        min(12, ab[0]["d"]) + min(12, ab[1]["d"]),
        min(ab[0]["fixed_d"], ab[1]["fixed_d"]),
        tuple(-(x + y) for x, y in
              zip(ab[0]["cycle_key"], ab[1]["cycle_key"]))),
        reverse=True)

    survivors = []
    for a, b in pairs[:6]:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 10 * q == 1290
        ar, br = _verified_components(hx, hz, q)
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Quotient balance is a proposal mechanism; measured full ranks,
        # not an assumed defect formula, enforce the rank-gain target.
        if not (k >= 528 and 15 * n <= 40 * k < 17 * n):
            continue
        logicals = localized_logical_challenge(a, b, hx, hz, n)
        detected = min(12, a["d"], b["d"], *logicals)
        quality = (
            k * detected * detected,
            detected,
            min(12, a["d"]) + min(12, b["d"]),
            tuple(-(x + y) for x, y in
                  zip(a["cycle_key"], b["cycle_key"])),
        )
        survivors.append(
            (quality, A, B, a, b, hx, hz, n, k, logicals))

    survivors.sort(key=lambda entry: entry[0], reverse=True)
    for quality, A, B, a, b, hx, hz, n, k, logicals in survivors:
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
            "commutator_class_profiles": (profiles[pa], profiles[pb]),
            "component_readings": (a["d"], b["d"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "verified_localized_logical_readings": logicals,
            "component_cycles": (a["cycles"], b["cycles"]),
            "component_ranks": (a["rank"], b["rank"]),
            "conservative_selection": quality,
        }
    raise ValueError("balanced commutator pairs missed full rank/rate gates")


def _rg_span(rows):
    pivots = {}
    for word in rows:
        while word:
            lead = word.bit_length() - 1
            old = pivots.get(lead)
            if old is None:
                pivots[lead] = word
                break
            word ^= old
    return pivots


def _rg_outside(word, pivots):
    while word:
        old = pivots.get(word.bit_length() - 1)
        if old is None:
            return True
        word ^= old
    return False


def _rg_syndrome(columns, word):
    answer = 0
    for c in _bits(word):
        answer ^= columns[c]
    return answer


def _rg_probe(record, seed, trials):
    """Retained genuine component witnesses, never distance certificates."""
    local = random.Random(seed)
    columns = record["columns"]
    order = list(range(len(columns)))
    retained = sorted(set(record.get("words", ())),
                      key=lambda w: (w.bit_count(), w))[:24]
    best = min(retained, key=int.bit_count) if retained else None

    def observe(word, grown):
        nonlocal best
        if not word:
            return
        if best is None or word.bit_count() < best.bit_count():
            best = word
        if word.bit_count() <= best.bit_count() + 4:
            grown.append(word)

    for word in retained:
        assert _rg_syndrome(columns, word) == 0

    for turn in range(trials):
        local.shuffle(order)
        if retained and turn % 4 == 3:
            active = list(_bits(local.choice(retained[:6])))
            preferred = set(local.sample(active, max(1, len(active)//2)))
            order.sort(key=lambda c: c not in preferred)
        pivots, basis = {}, []
        grown = list(retained)
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
                observe(word, grown)
        light = sorted(set(basis + retained),
                       key=lambda w: (w.bit_count(), w))[:12]
        for word in light:
            for other in basis + retained:
                observe(word ^ other, grown)
            if len(grown) > 160:
                grown = sorted(set(grown),
                               key=lambda w: (w.bit_count(), w))[:32]
        moves = basis + retained
        local.shuffle(moves)
        for start in light[:5]:
            word = start
            for sweep in range(3):
                before = word.bit_count()
                for other in moves:
                    proposed = word ^ other
                    if proposed and proposed.bit_count() < word.bit_count():
                        word = proposed
                observe(word, grown)
                if word.bit_count() == before:
                    break
        if len(light) >= 3:
            for _ in range(64):
                a, b, c = local.sample(light, 3)
                observe(a ^ b ^ c, grown)
        grown.append(best)
        retained = sorted({w for w in grown if w},
                          key=lambda w: (w.bit_count(), w))[:24]
        if best.bit_count() <= 6:
            break
    assert best and _rg_syndrome(columns, best) == 0
    record["words"] = sorted(set(retained + [best]),
                             key=lambda w: (w.bit_count(), w))[:24]
    record["word"], record["d"] = best, best.bit_count()


def _rg_completion(record, q, seed, budget):
    """Budgeted anchored completion through ten; failure proves nothing."""
    local = random.Random(seed)
    columns = record["columns"]
    width = len(columns)
    best = record["word"]
    cap = min(10, best.bit_count() - 1)
    if cap < 5:
        return
    singles, pairs = {}, {}
    incidence = [0] * len(record["rows"])
    for c, value in enumerate(columns):
        singles.setdefault(value, 1 << c)
        for r in _bits(value):
            incidence[r] |= 1 << c
        for d in range(c):
            pairs.setdefault(value ^ columns[d], (1 << c) | (1 << d))
    priorities = [local.random() for _ in columns]
    roots = list(range(0, width, q))
    local.shuffle(roots)
    all_variables = (1 << width) - 1

    def note(word):
        nonlocal best
        if word and word.bit_count() < best.bit_count():
            assert _rg_syndrome(columns, word) == 0
            best = word

    for root in roots:
        remaining = [max(1, budget // len(roots))]

        def visit(value, word, allowed):
            if remaining[0] <= 0:
                return
            remaining[0] -= 1
            if not value:
                note(word)
                return
            left = min(cap, best.bit_count()-1) - word.bit_count()
            if left <= 0 or value.bit_count() > 3*left:
                return
            one = singles.get(value, 0)
            if one and one & allowed == one:
                note(word | one)
            pair = pairs.get(value, 0)
            if left >= 2 and pair and pair & allowed == pair:
                note(word | pair)
            left = min(cap, best.bit_count()-1) - word.bit_count()
            if left <= 2:
                return
            chosen, smallest = 0, width + 1
            for r in _bits(value):
                candidates = incidence[r] & allowed
                count = candidates.bit_count()
                if not count:
                    return
                if count < smallest:
                    chosen, smallest = candidates, count
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
    record["words"] = sorted(set(record["words"] + [best]),
                             key=lambda w: (w.bit_count(), w))[:24]
    record["word"], record["d"] = best, best.bit_count()


def NEW_CTX(rng, G, idx):
    """Twenty unequal-component rank-gain cards on Frobenius21 x C4."""
    q = G.q
    assert q == 84 and A_SHAPE == (1,3) and B_SHAPE == (2,5)
    card, cohort = idx % 20, idx // 20
    composition, topology = card % 5, card // 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g,h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

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
    gallery = [[[g] for g in range(q)]]
    xx, _, _ = lifted_matrices(P, gallery, [[[]]])
    _, zz, _ = lifted_matrices(P, [[[]]], gallery)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx] for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz] for g in range(q)],
    }
    # Row functionals constant on the normal C7 and central C4 fibers.
    functionals = (
        sum(1 << t for t,g in enumerate(G.elements) if g[1] in (0,1)),
        sum(1 << t for t,g in enumerate(G.elements) if g[1] in (1,2)),
    )
    assert gf2_rank(functionals) == 2

    central_profiles = (
        ((1,1), (1,1), (1,1)),
        ((1,1), (1,1), (1,3)),
        ((1,1), (1,3), (1,2)),
        ((1,2), (1,2), (1,2)),
        ((1,1), (1,2), (2,3)),
    )
    bplans = (
        ((1,1,1,2,2), (2,2,2,1,1)),
        ((0,1,2,2,2), (3,2,1,1,1)),
        ((0,0,2,2,3), (3,3,1,1,0)),
        ((0,0,1,3,3), (3,3,2,0,0)),
    )
    sa = [[3,3,3]]
    sb = [list(row) for row in bplans[topology]]
    assert [sum(row) for row in sb] == [7,8]
    assert all(sum(sb[i][j] for i in range(2)) == 3 for j in range(5))
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group")
    order_policies = ((12,), (12,28), (7,12,28), (7,28))
    ratios = [g for g in range(q)
              if orders[g] in order_policies[topology]]
    assert ratios

    def sidon(cell):
        if len(cell) != len(set(cell)):
            return False
        wanted = len(cell)*(len(cell)-1)
        left = {table[inverses[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverses[b]]
                 for a in cell for b in cell if a != b}
        return len(left) == len(right) == wanted

    def draw(side, i, j):
        count = sa[i][j] if side == "A" else sb[i][j]
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            return [root, table[root][rng.choice(ratios)]]
        for _ in range(512):
            if side == "A":
                z1,z2 = central_profiles[composition][j]
                # Later cohorts change relative central phases, not a
                # common transformation of an already constructed code.
                if cohort % 2 and j == 1:
                    z2 = (z2 + 2) % 4
                a = G.idx((rng.randrange(7),1,z1))
                b = G.idx((rng.randrange(7),2,z2))
                cell = [ident,a,b]
            else:
                cell = rng.sample(range(q), 3)
                root_inv = inverses[cell[0]]
                a = table[root_inv][cell[1]]
                b = table[root_inv][cell[2]]
            if table[a][b] != table[b][a] and sidon(cell):
                return cell
        raise ValueError("new cubic geometry sampler exhausted")

    def fresh(side):
        support = sa if side == "A" else sb
        return [[draw(side,i,j) for j in range(len(row))]
                for i,row in enumerate(support)]

    def inspect(grid, side):
        rows = []
        nc = len(grid[0])
        for cells in grid:
            for r in range(q):
                row = 0
                for j,cell in enumerate(cells):
                    value = 0
                    for g in cell:
                        value ^= blocks[side][g][r]
                    row |= value << (j*q)
                rows.append(row)
        quality = _component_quality(rows, nc*q)
        if quality is None:
            return None
        cycles, columns = quality
        assert all(c.bit_count() == 3 for c in columns)
        rank = gf2_rank(rows)
        if side == "A":
            for functional in functionals:
                image = 0
                for r in _bits(functional):
                    image ^= rows[r]
                assert image == 0
            if not q-8 <= rank <= q-2:
                return None
        elif rank < 2*q-4:
            return None
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return {
            "grid": [[list(cell) for cell in row] for row in grid],
            "rows": rows, "columns": columns, "rank": rank,
            "cycles": cycles, "signature": signature, "words": [],
        }

    def pool(side):
        offset = 100003 * int(side == "B")
        seen, records = set(), []
        for _ in range(160 if side == "A" else 112):
            rec = inspect(fresh(side), side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        multiplier = (24,48,96,192,384)[composition]
        records.sort(key=lambda r: (
            multiplier*r["cycles"][0]+r["cycles"][1], r["signature"]))
        selected = []
        for rec in records:
            if not _no_kernel4(rec["columns"]):
                continue
            _rg_probe(rec, seed+offset, 8)
            selected.append(rec)
            if len(selected) == (12 if side == "A" else 8):
                break
        if not selected:
            raise ValueError("empty new verified component shortlist")

        def merit(rec):
            multiplicity = sum(
                word.bit_count() == rec["d"] for word in rec["words"])
            rank_preference = -rec["rank"] if side == "A" else rec["rank"]
            return (
                min(12,rec["d"]), -multiplicity,
                rank_preference,
                -multiplier*rec["cycles"][0]-rec["cycles"][1],
            )

        current = max(selected, key=merit)
        mutable = [(i,j) for i,row in enumerate(sa if side == "A" else sb)
                   for j,count in enumerate(row) if count]
        for step in range(24):
            occupied = {c//q for c in _bits(current["word"])}
            targeted = [p for p in mutable if p[1] in occupied]
            i,j = rng.choice(targeted if targeted else mutable)
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            grid[i][j] = draw(side,i,j)
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            if not _no_kernel4(rec["columns"]):
                continue
            rec["words"] = [
                word for word in current["words"]
                if _rg_syndrome(rec["columns"],word) == 0]
            _rg_probe(rec, seed+offset+700001+1009*step, 6)
            selected.append(rec)
            if merit(rec) >= merit(current):
                current = rec
        finalists = sorted(selected, key=merit, reverse=True)[:4]
        for rec in finalists:
            _rg_probe(rec, seed+offset+1700003, 40)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:3]
        for rec in finalists:
            _rg_completion(rec,q,seed+offset+2700011,18000)
            _rg_probe(rec,seed+offset+3700013,80)
        finalists.sort(key=merit, reverse=True)
        return finalists

    def localized(a,b,hx,hz,n):
        """Actual logical upper readings in the ORIGINAL stabilizer quotient."""
        readings = []
        for side,rec,opposite,stabilizers in (
                ("A",a,hx,hz), ("B",b,hz,hx)):
            span = _rg_span(stabilizers)
            candidates = set()
            for word in rec["words"][:4]:
                assert _rg_syndrome(rec["columns"],word) == 0
                for fixed in range(5 if side == "A" else 3):
                    actual = (
                        sum(((word >> (j*q)) & mask)
                            << ((5*j+fixed)*q) for j in range(3))
                        if side == "A" else word << (5*fixed*q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if _rg_outside(actual,span):
                        candidates.add(actual)
            reading = n+1
            local = random.Random(seed + (
                4100011 if side == "A" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates,key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    winner = actual
                    for at,first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:at]:
                            proposed = partial ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and _rg_outside(actual,span)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading,actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa,bb = pool("A"),pool("B")
    survivors = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx,hz,n = lifted_matrices(P,a["grid"],b["grid"])
            assert n == 17*q == 1428
            ar,br = _verified_components(hx,hz,q)
            assert ar == a["rows"] and br == b["rows"]
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 510 and 325*n <= 1000*k < 375*n):
                continue
            logicals = localized(a,b,hx,hz,n)
            detected = min(12,a["d"],b["d"],*logicals)
            key = (
                k*detected*detected, detected,
                min(12,a["d"])+min(12,b["d"]),
                -a["cycles"][0]-b["cycles"][0],
                -a["cycles"][1]-b["cycles"][1],
            )
            survivors.append((key,A,B,a,b,hx,hz,k,logicals))
    survivors.sort(key=lambda entry: entry[0],reverse=True)
    for key,A,B,a,b,hx,hz,k,logicals in survivors:
        if not _no_kernel4(_syndrome_columns(hx,1428)):
            continue
        if not _no_kernel4(_syndrome_columns(hz,1428)):
            continue
        actual_x,actual_z,n = lifted_matrices(G,A,B)
        assert actual_x == hx and actual_z == hz and n == 1428
        assert css_ok(hx,hz)
        assert max(row.bit_count() for row in hx) == 12
        assert max(row.bit_count() for row in hz) <= 11
        assert [[len(cell) for cell in row] for row in A] == sa
        assert [[len(cell) for cell in row] for row in B] == sb
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (composition,topology,cohort),
            "component_readings": (a["d"],b["d"]),
            "component_ranks": (a["rank"],b["rank"]),
            "component_cycles": (a["cycles"],b["cycles"]),
            "verified_A_annihilators": functionals,
            "verified_localized_logical_readings": logicals,
            "conservative_selection": key,
        }
    raise ValueError("new rank-gain pairings missed cell 0.350 gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cell, including deliberate zero blocks."""
    cell = list(ctx[side][i][j])
    assert len(cell) <= 5
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term within-coset Sidon supports and independent component descent uniformly suppress sub-30 logical readings: the measured parent still screened slots 18, 26 and 48 at 27, 24 and 21.
# REFUTED: The Z43:Z3 occupancy portfolio with complement-fixed/unrestricted component repair beats fitness 58.9 on (1,3)x(1,3); it measured 58.2256, with four selected k=520 codes and one k=528 code, all reading twelve and retaining abundant sector-1 minimum witnesses.
# REFUTED: Affine-coupled Z43:C3 balanced cubic cards and forced-anchor challenges produce five readings >=14; the shown affine-triangle program measured five readings of twelve, retaining abundant sector-one minimum witnesses.
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
