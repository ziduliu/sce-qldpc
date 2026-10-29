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
CONCEPT = ("Hypothesis: Frobenius (C5 x C5) semidirect C8, order 200, A=(1,2), B=(1,3), with relative-order-controlled cubic supports and retained component/logical challenges improves five-code fitness beyond the cell-0.300 incumbent 76.1. Measurement can refute this target; finite readings remain upper bounds.")
GROUP = ("Jump from the order-291 metacyclic parent to a two-dimensional affine construction over F5. M=[[0,2],[1,0]] satisfies M^2=2I and has order eight; every nonidentity power is fixed-point-free. The resulting order-200 group has centre one and eleven conjugacy classes, matching the signature of the strongest recorded order-200 cell-0.300 bed, fitness 76.1 with five distance-16 readings. This motivates the group, not certified distance transfer or an assumed identification from its signature alone.")
SHAPES = ("A=(1,2), B=(1,3): factor seven gives n=1400, k>=400 and guaranteed rate 2/7=0.285714; the size law permits q<=214. Unequal horizontal components have 400 and 600 variables with 200 checks each. Parent minima occupied sector-one strips, especially Acol1 across both Bcols and Bcol1 across both Acols. Add Bcol2 and explicitly challenge the higher-rate three-cell component. Authoritative full ranks enforce [0.275,0.325).")
FILLING = ("All five cells contain three distinct terms, giving HX=6+3=9 and HZ=9+3=12. Twenty cards cross five A and four B compositions of relative-order spectra (5,8,8), (4,8,8), and (4,4,5); every side includes full-action cells. Catalog one representative per right-cell gauge, require bidirectional Sidon supports, and select independently new relations among cells. Scaffold-derived monomial blocks supply component geometry and genuine unrestricted/C8-orbit-constant kernel witnesses. Retain all binding readings, repair implicated cells without changing the card, and independently deepen finalists. Compare full pairings using exact ranks and commuting, nontrivial sector-one logicals reduced modulo ORIGINAL stabilizers. Component agreement, full kernel-four exclusion, distinctness, weights and CSS remain gates. Neither spectra nor finite searches certify distance.")
HOW_AB = FILLING
FAMILY_NAME = "r030_F25_C8_12x13_cubic_spectra_retained_logical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 200

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F5^2 semidirect C8 requires order 200")
    # (u,v,b) denotes translation (u,v) followed by M^b.
    # M(u,v)=(2v,u), M^2=2I, M^8=I.
    powers = []
    matrix = (1,0,0,1)
    for b in range(8):
        powers.append(matrix)
        a,c,d,e = matrix
        matrix = (c % 5, 2*a % 5, e % 5, 2*d % 5)
    assert matrix == (1,0,0,1)
    assert len(set(powers)) == 8
    els = [(u,v,b) for b in range(8)
           for v in range(5) for u in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 5, int(g[2]) % 8)

    def mul(g, h):
        u,v,b = red(g)
        w,t,c = red(h)
        a,d,e,f = powers[b]
        return ((u+a*w+d*t) % 5,
                (v+e*w+f*t) % 5, (b+c) % 8)

    def inv(g):
        u,v,b = red(g)
        bi = (-b) % 8
        a,d,e,f = powers[bi]
        return ((-a*u-d*v) % 5, (-e*u-f*v) % 5, bi)

    def idx(g):
        u,v,b = red(g)
        return 25*b+5*v+u

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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


def _retired_metacyclic_ctx(rng, G, idx):
    """Inactive order-291 experiment; the affine NEW_CTX below replaces it."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (1, 2)
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    trade = (card + cohort) % 4
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xC09712A4 + 15485863 * (idx + 1)))

    # Cache native operations only. Every binary action block below
    # comes from the authoritative construction, not a handwritten lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
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
            return inverses[g]

    probe_group = Probe()

    # Narrow tall galleries provide transpose monomials. Transpose
    # those EXACT bit blocks to obtain the horizontal monomials.
    monomials = {"A": [], "B": []}
    zero = [[[]]]
    for start in range(0, q, 12):
        ids = list(range(start, min(start + 12, q)))
        gallery = [[[g]] for g in ids]
        _, source_a, _ = lifted_matrices(probe_group, gallery, zero)
        source_b, _, _ = lifted_matrices(probe_group, zero, gallery)
        assert len(source_a) == len(source_b) == q
        for j in range(len(ids)):
            shift = (j + 1) * q
            for side, source in (("A", source_a), ("B", source_b)):
                rows = [0] * q
                for r, source_row in enumerate(source):
                    value = (source_row >> shift) & mask
                    assert value.bit_count() == 1
                    rows[value.bit_length() - 1] |= 1 << r
                monomials[side].append(rows)

    def whole_sidon(cell):
        """Exclude repeated directed differences in both orientations."""
        if len(cell) != 4 or len(set(cell)) != 4:
            return False
        ids = [G.idx(g) for g in cell]
        left, right = set(), set()
        for a in ids:
            for b in ids:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        return len(left) == len(right) == 12

    def component(grid, side):
        """GF2 superposition of scaffold-supplied horizontal blocks."""
        assert len(grid) == 1
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, value in enumerate(monomials[side][G.idx(g)]):
                    rows[r] ^= value << (j * q)
        assert all(row >> (len(grid[0]) * q) == 0 for row in rows)
        return rows

    # A full-coset quartic has a useful normalized coupling parameter:
    # [e,x,y,x^t*y^2]. Swapping the two normal-coset terms replaces
    # t by -t-s after renormalization. Identify that elementary symmetry
    # before assigning cards; do not count its images as new designs.
    alphabet = []
    for t in range(97):
        if t > (-t - 35) % 97:
            continue
        cell = [(0, 0), (1, 0), (0, 1), (t, 2)]
        if not whole_sidon(cell):
            continue
        ranks = tuple(gf2_rank(component([[cell]], side))
                      for side in ("A", "B"))
        if min(ranks) >= q - 3:
            alphabet.append(t)
    if len(alphabet) < 6:
        raise ValueError("insufficient screened quartic coupling classes")

    # Spread the atom classes across the legal catalog. Later cohorts
    # choose genuinely different class subsets, not affine label images.
    start = (5 * cohort) % len(alphabet)
    alphabet = alphabet[start:] + alphabet[:start]
    atom_types = tuple(alphabet[i * len(alphabet) // 6]
                       for i in range(6))
    assert len(set(atom_types)) == 6
    profiles = (
        (0, 1), (0, 2), (1, 3),
        (2, 4), (3, 5), (4, 5),
    )
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    assert len(plans) == 20
    pa, pb = plans[card]

    def fresh_cell(atom):
        """Preserve the assigned coupling while changing relative supports.

        o,d,a and the quotient phase are fresh. This is a proposal
        family, not a claim that different draws are inequivalent.
        """
        for _ in range(512):
            o = rng.randrange(97)
            d = rng.randrange(1, 97)
            a = rng.randrange(97)
            phase = rng.randrange(3)
            c = (36 * a - 35 * o + atom * d) % 97
            cell = [
                (o, phase), ((o + d) % 97, phase),
                (a, (phase + 1) % 3), (c, (phase + 2) % 3),
            ]
            if whole_sidon(cell):
                return cell
        raise ValueError("coupled whole-cell Sidon recipe exhausted")

    def fresh(profile):
        return [[fresh_cell(atom_types[k])
                 for k in profiles[profile]]]

    def eight_cycles(rows, columns):
        """Exact simple Tanner eight-cycles, including cycles with chords."""
        checks = [list(_bits(row)) for row in rows]
        neighbors = [list(_bits(column)) for column in columns]
        rooted = 0
        # The regular component action is transitive within each block.
        for root in (0, q):
            paths = {}
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
                            previous = paths.setdefault(end, [])
                            for d1, u, d2 in previous:
                                if (u != v and c1 != d1 and c1 != d2
                                        and c2 != d1 and c2 != d2):
                                    rooted += 1
                            previous.append((c1, v, c2))
        assert q * rooted % 4 == 0
        return q * rooted // 4

    def cycle_key(stats):
        four, six, eight = stats
        if trade == 0:
            return (four, six, eight)
        if trade == 1:
            return (four, eight, six)
        if trade == 2:
            return (four, 10 * six + eight, six)
        return (four, 30 * six + eight, eight)

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for cell in grid[0])

    def inspect(grid, side):
        rows = component(grid, side)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        rank = gf2_rank(rows)
        ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if min((rank,) + ranks) < q - 3:
            return None
        stats, columns = quality
        stats = stats + (eight_cycles(rows, columns),)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats), "signature": signature(grid),
            "d": width + 1, "words": [],
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

    def kernel_probe(record, diagnostic_seed, trials):
        """Persistent genuine kernel witnesses; never certify distance."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        retained = sorted(
            {w for w in record["words"] if w},
            key=lambda w: (w.bit_count(), w))[:24]
        reading, witness = width + 1, None
        order = list(range(width))
        additions = []

        def note(word):
            nonlocal reading, witness
            if not word:
                return
            weight = word.bit_count()
            if weight < reading:
                reading, witness = weight, word
            if weight <= reading + 6:
                additions.append(word)
                if len(additions) > 192:
                    additions[:] = sorted(
                        set(additions),
                        key=lambda w: (w.bit_count(), w))[:32]

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)

        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 4 == 3:
                active = list(_bits(diagnostic.choice(retained[:6])))
                diagnostic.shuffle(active)
                preferred = set(active[:max(1, len(active) // 2)])
                order.sort(key=lambda c: c not in preferred)

            pivots, basis, independent = {}, [], []
            additions = list(retained)
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        independent.append(c)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word)

            light = sorted(
                set(basis + retained),
                key=lambda w: (w.bit_count(), w))[:20]
            moves = basis + retained
            for word in light:
                for other in moves:
                    note(word ^ other)

            # Bounded hashed four-circuit cancellation. Every result is
            # a kernel word, but hash/list limits can miss lighter ones.
            if trials >= 32 and turn % 4 == 0 and independent:
                chosen = sorted(basis, key=int.bit_count)[:72]
                coords = diagnostic.sample(
                    independent, min(9, len(independent)))
                hash_mask = sum(1 << c for c in coords)
                buckets = {}
                for at, first in enumerate(chosen):
                    for second in chosen[:at]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for previous in bucket:
                            note(pair ^ previous)
                        if len(bucket) < 12:
                            bucket.append(pair)

            diagnostic.shuffle(moves)
            for initial in light[:8]:
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
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c)
            if witness is not None:
                additions.append(witness)
            retained = sorted(
                {w for w in retained + additions if w},
                key=lambda w: (w.bit_count(), w))[:24]
            if reading <= 24:
                break

        if witness is None:
            raise ValueError("horizontal component has no kernel witness")
        assert syndrome(columns, witness) == 0
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])
        record["words"] = sorted(
            set(retained + [witness]),
            key=lambda w: (w.bit_count(), w))[:24]
        return reading, witness

    def optimize(profile, side):
        bank = set()
        seen = set()
        records = []
        offset = 0 if side == "A" else 100003

        def probe(record, trials, turn):
            d, word = kernel_probe(
                record, seed + offset + 700001 * turn, trials)
            record["d"] = min(record["d"], d)
            bank.add(word)
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)

        def merit(record):
            # A witness from another labeling is relevant only after its
            # syndrome is verified on this particular component.
            for word in bank:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"] = word.bit_count()
            # Reduce selection pressure on optimistic finite searches.
            # This cap is not a lower bound on component distance.
            return (min(76, record["d"]),
                    tuple(-x for x in record["cycle_key"]),
                    min(96, record["d"]), record["rank"])

        for trial in range(80 + 4 * min(cohort, 2)):
            record = inspect(fresh(profile), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            records.append(record)
        records.sort(key=lambda record: record["cycle_key"])

        selected = []
        for record in records:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 12, 0)
            selected.append(record)
            if len(selected) >= 8:
                break
        if not selected:
            return []

        floor4 = min(record["cycles"][0] for record in selected)
        allowance = 0 if trade == 0 else q
        current = max(selected, key=merit)

        # Repair actual labels at the protograph rows supporting the
        # current light witnesses. No group automorphism creates proposals.
        for step in range(24):
            active = [word for word in bank
                      if syndrome(current["columns"], word) == 0]
            blocks = []
            if active:
                word = min(active, key=int.bit_count)
                blocks = [c // q for c in _bits(word)]
            j = (rng.choice(blocks) if blocks and rng.random() < 0.85
                 else rng.randrange(2))
            atom = atom_types[profiles[profile][j]]
            witnesses = sorted(
                set(current["words"]),
                key=lambda w: (w.bit_count(), w))[:6]
            proposals = []
            # Whole-cell replacements preserve the assigned normalized
            # atom, but change its relation to the OTHER component cell.
            # Compare actual disruption of genuine retained witnesses.
            for attempt in range(5):
                grid = [[list(cell) for cell in current["grid"][0]]]
                grid[0][j] = fresh_cell(atom)
                columns = _syndrome_columns(component(grid, side), width)
                boundaries = [
                    syndrome(columns, word).bit_count()
                    for word in witnesses
                ]
                key = (
                    sum(value > 0 for value in boundaries),
                    min(boundaries, default=0), sum(boundaries),
                )
                proposals.append((key, grid))
            proposals.sort(key=lambda item: item[0], reverse=True)
            grid = proposals[0][1]
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if proposed["cycles"][0] > floor4 + allowance:
                continue
            # Transfer only words that remain genuine for the NEW matrix.
            proposed["words"] = [
                word for word in current["words"]
                if syndrome(proposed["columns"], word) == 0
            ]
            probe(proposed, 8, 1 + step)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.15):
                current = proposed
                selected.append(proposed)

        selected.sort(key=merit, reverse=True)
        finalists = []
        for record in selected:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 48, 101)
            finalists.append(record)
            if len(finalists) >= 3:
                break
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent confirmation only lowers a retained reading.
            probe(record, 128 + 16 * min(cohort, 2), 211)
        finalists.sort(key=merit, reverse=True)
        return finalists

    for restart in range(2):
        aa = optimize(pa, "A")
        bb = optimize(pb, "B")
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"] + ab[1]["d"],
            tuple(-(x + y) for x, y in
                  zip(ab[0]["cycle_key"], ab[1]["cycle_key"]))),
            reverse=True)

        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 5 * q == 1455
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= q and 7 * n <= 40 * k < 9 * n):
                continue

            # Verify cached horizontal components against the FINAL
            # native full lift, including its authoritative column order.
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert actual_a == a["rows"] and actual_b == b["rows"]
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            assert max(row.bit_count() for row in hx + hz) == 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (pa, pb, trade, cohort),
                "coupling_types": atom_types,
                "coupling_profiles": (profiles[pa], profiles[pb]),
                "component_readings": (a["d"], b["d"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
            }
    raise ValueError("no coupled horizontal pair passed full gates")


def NEW_CTX(rng, G, idx):
    """Twenty affine cubic cards with binding kernel and logical challenges."""
    from itertools import combinations

    q = G.q
    assert q == 200 and A_SHAPE == (1,2) and B_SHAPE == (1,3)
    mask = (1 << q)-1
    sa, sb = [[3,3]], [[3,3,3]]
    assert _grid_weights(sa,sb) == 12 <= W_MAX
    assert _no_dead_lines(sa,sb)
    card, cohort = idx % 20, idx // 20
    ap, bp = divmod(card,4)
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xF250C8 + 32452843*(idx+1)))
    trade = (card+cohort) % 3

    # Native operations only. All action blocks come from the scaffold.
    table = [[G.idx(G.mul(g,h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self,g):
            return g

        def mul(self,g,h):
            return table[g][h]

        def inv(self,g):
            return inverse[g]

    P = Probe()
    blocks = {"A": [], "B": []}
    zero = [[[]]]
    for start in range(0,q,16):
        ids = list(range(start,min(start+16,q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P,gallery,zero)
        _, zz, _ = lifted_matrices(P,zero,gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j*q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j*q)) & mask for row in zz])

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns,word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def expand(word,masks):
        value = 0
        for c in _bits(word):
            value ^= masks[c]
        return value

    def signature(grid):
        return tuple(tuple(sorted(cell)) for cell in grid[0])

    orders = []
    for g in range(q):
        value = ident
        for order in range(1,q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("native group has invalid element order")
    assert set(orders) == {1,2,4,5,8}

    spectra = ((5,8,8),(4,8,8),(4,4,5))
    libraries = [[] for _ in spectra]

    def sidon(cell):
        if len(cell) != 3 or len(set(cell)) != 3:
            return False
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        return len(left) == len(right) == 6

    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            spectrum = tuple(sorted((
                orders[a],orders[b],orders[table[inverse[a]][b]])))
            if spectrum not in spectra:
                continue
            support = tuple(sorted((ident,a,b)))
            # p -> p*h is a cell-column gauge. Do not populate the
            # proposal catalog with its alternative anchor choices.
            representative = min(
                tuple(sorted(table[g][inverse[root]] for g in support))
                for root in support)
            if support != representative or not sidon(support):
                continue
            # Require a nonzero normal commutator. Odd quotient powers
            # act irreducibly, hence types 0/1 generate the whole group.
            commutator = table[
                table[table[a][b]][inverse[a]]][inverse[b]]
            if commutator == ident:
                continue
            libraries[spectra.index(spectrum)].append(support)
    if not all(libraries):
        raise ValueError("one prescribed cubic spectrum has no support")

    aplans = ((0,0),(0,1),(1,1),(0,2),(1,2))
    bplans = ((0,0,0),(0,0,1),(0,1,1),(0,0,2))
    plans = {"A": aplans[ap], "B": bplans[bp]}
    assert len(aplans)*len(bplans) == 20
    assert all(any(kind < 2 for kind in plan)
               for plan in plans.values())

    def draw(kind):
        return list(rng.choice(libraries[kind]))

    def fresh(side):
        for _ in range(256):
            cells = [draw(kind) for kind in plans[side]]
            if len({tuple(cell) for cell in cells}) == len(cells):
                return [cells]
        raise ValueError("distinct cubic component supports exhausted")

    # Coordinate restrictions only: XORs of verified component columns.
    # A-left action commutes with right translations; B-right with left.
    subgroup = [G.idx((0,0,b)) for b in range(8)]
    fixed_masks = {}
    for side,nc in (("A",2),("B",3)):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == "A" else table[h][root]
                for h in subgroup}
            assert len(orbit) == 8 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        assert len(orbits) == 25
        fixed_masks[side] = [
            sum(1 << (j*q+c) for c in orbit)
            for j in range(nc) for orbit in orbits]

    def inspect(grid,side):
        nc = len(grid[0])
        width = nc*q
        rows = [0]*q
        for j,cell in enumerate(grid[0]):
            if not sidon(cell):
                return None
            for g in cell:
                for r,value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j*q)
        quality = _component_quality(rows,width)
        if quality is None:
            return None
        cycles,columns = quality
        assert all(c.bit_count() == 3 for c in columns)
        rank = gf2_rank(rows)
        # This is only a proposal gate. Full quantum ranks decide rate.
        if rank < q-20:
            return None
        cell_ranks = tuple(
            gf2_rank([(row >> (j*q)) & mask for row in rows])
            for j in range(nc))
        if min(cell_ranks) < q-40:
            return None
        four,six = cycles
        if trade == 0:
            cost = (four,six)
        elif trade == 1:
            cost = (8*four+six,four)
        else:
            cost = (24*four+six,six)
        masks = fixed_masks[side]
        fixed_columns = [syndrome(columns,part) for part in masks]
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            rank=rank, cell_ranks=cell_ranks,
            cycles=cycles, cost=cost, signature=signature(grid),
            width=width, words=[], fixed_words=[],
            fixed_columns=fixed_columns, masks=masks,
            d=width+1, fixed_d=width+1)

    def basis_words(columns,order):
        pivots,words = {}, []
        for c in order:
            value,word = columns[c],1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value,word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                words.append(word)
        return words

    def search(columns,diagnostic_seed,trials,seeds=(),complete_pairs=False):
        """Persistent genuine null words; finite failure excludes nothing."""
        local = random.Random(diagnostic_seed)
        order = list(range(len(columns)))
        retained = sorted({w for w in seeds if w},
                          key=lambda w:(w.bit_count(),w))[:20]
        best = None

        def observe(word,found):
            nonlocal best
            if not word:
                return
            if best is None or word.bit_count() < best.bit_count():
                best = word
            if word.bit_count() <= best.bit_count()+4:
                found.append(word)

        for word in retained:
            assert syndrome(columns,word) == 0
            observe(word,[])
        for turn in range(trials):
            local.shuffle(order)
            if retained and turn % 4 == 3:
                support = list(_bits(local.choice(retained[:6])))
                local.shuffle(support)
                preferred = set(support[:max(1,len(support)//2)])
                order.sort(key=lambda c:c not in preferred)
            basis = basis_words(columns,order)
            found = []
            for word in basis:
                observe(word,found)
            light = sorted(set(basis+retained),
                           key=lambda w:(w.bit_count(),w))[:16]
            if complete_pairs:
                for at,word in enumerate(basis):
                    for other in basis[:at]:
                        observe(word ^ other,found)
            else:
                for word in light:
                    for other in basis:
                        observe(word ^ other,found)
                    if len(found) > 128:
                        found = sorted(set(found),key=int.bit_count)[:24]
            moves = retained+basis
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if (candidate and candidate.bit_count()
                                < word.bit_count()):
                            word,changed = candidate,True
                    observe(word,found)
                    if not changed:
                        break
            if complete_pairs and len(basis) >= 3:
                for _ in range(160):
                    a,b,c = local.sample(basis,3)
                    observe(a ^ b ^ c,found)
            retained = sorted(
                {w for w in retained+light+found+([best] if best else [])
                 if w},key=lambda w:(w.bit_count(),w))[:20]
            if best is not None and best.bit_count() <= (
                    2 if complete_pairs else 8):
                break
        if best is None:
            raise ValueError("component challenge found no null word")
        assert syndrome(columns,best) == 0
        return best,retained

    def challenge(rec,side,full_trials,fixed_trials,offset):
        word,beam = search(
            rec["columns"],seed+offset,full_trials,rec["words"])
        fixed,fixed_beam = search(
            rec["fixed_columns"],seed+offset+5000011,
            fixed_trials,rec["fixed_words"],complete_pairs=True)
        rec["fixed_words"] = fixed_beam
        expanded = [expand(w,rec["masks"]) for w in fixed_beam]
        actual = expand(fixed,rec["masks"])
        assert actual.bit_count() == 8*fixed.bit_count()
        assert syndrome(rec["columns"],actual) == 0
        rec["fixed_d"] = min(rec["fixed_d"],actual.bit_count())
        rec["words"] = sorted(
            set(rec["words"]+beam+expanded+[word,actual]),
            key=lambda w:(w.bit_count(),w))[:24]
        rec["d"] = min(rec["d"],word.bit_count(),actual.bit_count())
        assert all(syndrome(rec["columns"],w) == 0
                   for w in rec["words"])
        assert all((row & rec["words"][0]).bit_count() % 2 == 0
                   for row in rec["rows"])

    def merit(rec):
        # Capping reduces pressure on optimistic finite-budget readings.
        return (min(18,rec["d"]),
                tuple(-x for x in rec["cost"]),
                min(28,rec["d"]),-rec["rank"])

    def pool(side,restart):
        offset = 100003*int(side == "B")+300007*restart
        records,seen = [],set()
        count = (72 if side == "A" else 112)+8*min(cohort,2)
        for _ in range(count):
            rec = inspect(fresh(side),side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda rec:rec["cost"])
        selected = []
        for rec in records:
            if not _no_kernel4(rec["columns"]):
                continue
            challenge(rec,side,8,8,offset)
            selected.append(rec)
            if len(selected) == (8 if side == "A" else 12):
                break
        if not selected:
            return []
        current = max(selected,key=merit)

        for step in range(16 if side == "A" else 28):
            witnesses = current["words"][:6]
            active = rng.choice(witnesses)
            occupied = sorted({c//q for c in _bits(active)})
            j = rng.choice(occupied)
            proposals = []
            for _ in range(4):
                grid = copy_grid(current["grid"])
                grid[0][j] = draw(plans[side][j])
                sig = signature(grid)
                if sig in seen:
                    continue
                proposed = inspect(grid,side)
                if proposed is None:
                    continue
                disruption = [
                    syndrome(proposed["columns"],w).bit_count()
                    for w in witnesses]
                key = (sum(x > 0 for x in disruption),
                       min(disruption),sum(disruption))
                proposals.append((key,proposed))
            if not proposals:
                continue
            proposals.sort(key=lambda item:item[0],reverse=True)
            proposed = proposals[0][1]
            seen.add(proposed["signature"])
            if not _no_kernel4(proposed["columns"]):
                continue
            # Transfer only witnesses still genuine for this NEW filling.
            proposed["words"] = [
                w for w in current["words"]
                if syndrome(proposed["columns"],w) == 0]
            challenge(proposed,side,6,8,offset+700001+7919*step)
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        unique = {rec["signature"]:rec for rec in selected}
        finalists = sorted(unique.values(),key=merit,reverse=True)[:4]
        for rec in finalists:
            challenge(rec,side,24,32,offset+1700021)
        finalists.sort(key=merit,reverse=True)
        finalists = finalists[:2]
        for rec in finalists:
            # Independent challenges retain every earlier upper reading.
            challenge(rec,side,80,64,offset+2700011)
        finalists.sort(key=merit,reverse=True)
        return finalists

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

    def outside(word,pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_challenge(a,b,hx,hz,n):
        """Actual sector-one logicals in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind,rec,opposite,stabilizers in (
                ("Z",a,hx,hz),("X",b,hz,hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in rec["words"][:5]:
                for location in range(3 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j+location)*q) for j in range(2))
                    else:
                        actual = word << (3*location*q)
                    assert actual >> n == 0
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual,pivots):
                        candidates.add(actual)
            reading = n+1
            local = random.Random(
                seed+(4100011 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates,key=int.bit_count)[:6]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row:(row & actual).bit_count(),
                        reverse=True)[:28]
                    winner = actual
                    for first,second in combinations(nearby,2):
                        proposed = actual ^ first ^ second
                        if proposed.bit_count() < winner.bit_count():
                            winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual,pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading,actual.bit_count())
            # n+1 means no nontrivial placement was found, not exclusion.
            readings.append(reading)
        return tuple(readings)

    for restart in range(3):
        aa,bb = pool("A",restart),pool("B",restart)
        survivors = []
        for a in aa:
            for b in bb:
                hx,hz,n = lifted_matrices(P,a["grid"],b["grid"])
                assert n == 7*q == 1400
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (k >= 400 and 275*n <= 1000*k < 325*n):
                    continue
                actual_a,actual_b = _verified_components(hx,hz,q)
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_challenge(a,b,hx,hz,n)
                detected = min(18,a["d"],b["d"],*logicals)
                quality = (
                    k*detected*detected,detected,
                    tuple(-(x+y) for x,y in zip(a["cost"],b["cost"])),
                    min(28,a["d"])+min(28,b["d"]))
                survivors.append(
                    (quality,a,b,hx,hz,k,logicals))
        survivors.sort(key=lambda item:item[0],reverse=True)
        for quality,a,b,cached_hx,cached_hz,k,logicals in survivors:
            if not _no_kernel4(_syndrome_columns(cached_hx,1400)):
                continue
            if not _no_kernel4(_syndrome_columns(cached_hz,1400)):
                continue
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            assert [[len(c) for c in row] for row in A] == sa
            assert [[len(c) for c in row] for row in B] == sb
            assert all(len(set(c)) == 3
                       for grid in (A,B) for row in grid for c in row)
            hx,hz,n = lifted_matrices(G,A,B)
            assert hx == cached_hx and hz == cached_hz and n == 1400
            assert k == n-gf2_rank(hx)-gf2_rank(hz)
            assert max(row.bit_count() for row in hx) == 9
            assert max(row.bit_count() for row in hz) == 12
            assert css_ok(hx,hz)
            return dict(
                idx=idx,cells={"A":A,"B":B},k=k,quality=quality,
                card=(ap,bp,trade,cohort),
                relative_order_plans=(plans["A"],plans["B"]),
                spectrum_catalog=spectra,
                component_readings=(a["d"],b["d"]),
                C8_fixed_readings=(a["fixed_d"],b["fixed_d"]),
                component_ranks=(a["rank"],b["rank"]),
                cell_ranks=(a["cell_ranks"],b["cell_ranks"]),
                component_cycles=(a["cycles"],b["cycles"]),
                verified_sector_one_logical_readings=logicals)
    raise ValueError("affine cubic pair missed verified cell-0.300 gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cubic cell; no post-screen randomization."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Coupled full-coset quartics with persistent component challenges uniformly exclude sub-30 logicals; current horizontal slots 30 and 33 instead screened at 27 with k=295 and w=12.
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
