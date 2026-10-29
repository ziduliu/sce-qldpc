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
CONCEPT = ("Hypothesis: on GL2(F3) x C2, order 96, A=(1,2), B=(2,5), long-order binomial histograms and binding central-fixed kernel challenges produce five NEW measured readings >=23 in cell 0.250, exceeding its 124.5794 record. At the guaranteed rate, five readings 23 score at least 132.25. Measurement can refute this target; finite searches certify no distance.")
GROUP = ("Jump from C97:C3 to the matrix/direct-product rule GL2(F3) x C2, order 48*2=96, with centre C2 x C2 and sixteen conjugacy classes. This bed is absent from BEDS. The 0.250 incumbent SL2(F5) bed scored 124.5794 with readings 20..23; compare order-eight matrix-group ratios against its order-five/ten recipes. Central restrictions below are sums of authoritative columns, not separately implemented lifts or assumed distance transfer.")
SHAPES = ("Set A=(1,2), B=(2,5): n=12*96=1152, k>=3*96=288 and guaranteed rate 0.250; the shape permits q_max=124 but this group fixes q=96. Unequal components have 192 variables/96 checks and 480 variables/192 checks, nominal rates 1/2 and 3/5. Current parent minima were sector-one Acol/Bcol strips, especially Acol1 across both Bcols and Bcol1 across both Acols; slot 54 also reached all four sector-one blocks. Change the B check structure and component lengths, and enforce the actual full-rank cell interval [0.225,0.275).")
FILLING = ("Twenty binding cards cross two positive B supports, five exact binomial-order histograms and two A determinant/central-bit occupancy profiles. A counts (4,4), B row totals eight and degrees (4,3,3,3,3) give BOTH maxima twelve. Native bidirectional Sidon cells forbid involutory differences. Independently anneal scaffold-derived component geometry, enumerate small single-block kernels, retain real information-set/cancellation witnesses, and challenge all three central involutions plus the full centre-four restriction. Repairs preserve supports and histograms while changing actual labels. Equal independent finalist tests only lower retained readings. Complete pairings are challenged by commuting, nontrivial sector-one representatives reduced modulo ORIGINAL stabilizers. Exact full k, component agreement, kernel-four exclusion, distinctness, caps and CSS remain gates. Neither cards, geometry nor finite readings prove distance or arbitrary Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r025_GL2F3xC2_12x25_orderhist_centre4_persistent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 96

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("GL2(F3) x C2 requires order 96")
    # (a,b,c,d,z) represents an invertible 2x2 matrix over F3
    # and an independent central bit. GL2(F3) has (9-1)*(9-3)=48 elements.
    els = [(a,b,c,d,z)
           for z in range(2)
           for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3 != 0]
    positions = {g: i for i,g in enumerate(els)}

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3,
                int(g[2]) % 3, int(g[3]) % 3, int(g[4]) % 2)

    def mul(g, h):
        a,b,c,d,z = red(g)
        e,f,u,v,w = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3, z ^ w)

    def inv(g):
        a,b,c,d,z = red(g)
        determinant = (a*d-b*c) % 3
        if determinant == 0:
            raise ValueError("singular matrix is not a group element")
        scale = 1 if determinant == 1 else 2
        return ((scale*d) % 3, (-scale*b) % 3,
                (-scale*c) % 3, (scale*a) % 3, z)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
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


def _retired_metacyclic_coupling_ctx(rng, G, idx):
    """Inactive order-291 experiment; never called on the matrix group."""
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
    """Twenty matrix-group cards with binding centre-four challenges."""
    from math import exp

    q = G.q
    assert q == 96 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    topology = card // 10
    policy = (card // 2) % 5
    occupancy = card % 2
    seed = rng.randrange(1 << 62) ^ (0x963025C4 + 1000003*idx)
    rng = random.Random(seed ^ 0x48C296)
    mask = (1 << q)-1
    ident = G.idx(G.e)

    supports = {
        "A": [[4,4]],
        "B": [list(row) for row in (
            ((2,2,2,1,1), (2,1,1,2,2)),
            ((1,2,2,2,1), (3,1,1,1,2)),
        )[topology]],
    }
    sa, sb = supports["A"], supports["B"]
    assert [sum(row) for row in sb] == [8,8]
    assert [sum(row[j] for row in sb) for j in range(5)] == [4,3,3,3,3]
    assert _grid_weights(sa,sb) == 12 <= W_MAX
    assert _no_dead_lines(sa,sb)

    # Native operation cache only. Every binary block below comes from
    # the scaffold, with exactly the evaluator's indexing.
    table = [[G.idx(G.mul(g,h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]

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
            return inverse[g]

    P = Probe()
    orders = []
    for g in range(q):
        value = ident
        for order in range(1,q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite matrix group")

    centre = [g for g in range(q)
              if all(table[g][h] == table[h][g] for h in range(q))]
    involutions = [g for g in centre if orders[g] == 2]
    assert len(centre) == 4 and len(involutions) == 3
    subgroups = [[ident,g] for g in involutions] + [centre]
    partitions = []
    for subgroup in subgroups:
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {table[root][h] for h in subgroup}
            assert len(orbit) == len(subgroup) and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        partitions.append(orbits)

    # Determinant and the independent central bit give C2 x C2.
    quotient = []
    for a,b,c,d,z in G.elements:
        quotient.append(2*int((a*d-b*c) % 3 == 2)+z)
    fibres = [[g for g in range(q) if quotient[g] == v]
              for v in range(4)]
    assert all(len(f) == 24 for f in fibres)
    recipes = ((8,), (6,), (4,8), (6,8), (4,6,8))
    ratio_pools = {o: [g for g in range(q) if orders[g] == o]
                   for o in (4,6,8)}
    assert all(ratio_pools.values())

    def sidon(cell):
        if len(set(cell)) != len(cell):
            return False
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        wanted = len(cell)*(len(cell)-1)
        return len(left) == len(right) == wanted

    def quotient_ok(cell):
        counts = sorted(sum(quotient[g] == v for g in cell)
                        for v in range(4))
        return counts == ([0,1,1,2] if occupancy == 0 else [1,1,1,1])

    def draw(count, side, wanted=None):
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            assert wanted in recipes[policy]
            root = rng.randrange(q)
            cell = [root,table[root][rng.choice(ratio_pools[wanted])]]
            assert sidon(cell)
            return cell
        for _ in range(2048):
            if side == "A":
                counts = [2,1,1,0] if occupancy == 0 else [1,1,1,1]
                rng.shuffle(counts)
                cell = [g for v,count_v in enumerate(counts)
                        for g in rng.sample(fibres[v],count_v)]
            else:
                cell = rng.sample(range(q),count)
            if sidon(cell):
                assert len(cell) == count
                return cell
        raise ValueError("matrix-group Sidon occupancy exhausted")

    def quotas():
        count = sum(c == 2 for row in sb for c in row)
        wanted = recipes[policy]
        result = {o: count//len(wanted) for o in wanted}
        for t in range(count % len(wanted)):
            result[wanted[(t+cohort) % len(wanted)]] += 1
        assert all(result.values()) and sum(result.values()) == count
        return result

    prescribed_histogram = quotas()

    def fresh(side):
        if side == "A":
            return [[draw(4,side),draw(4,side)]]
        assigned = [o for o,count in prescribed_histogram.items()
                    for _ in range(count)]
        rng.shuffle(assigned)
        assigned = iter(assigned)
        return [[draw(count,side,next(assigned) if count == 2 else None)
                 for count in row] for row in sb]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def legal(grid, side):
        support = supports[side]
        if [[len(c) for c in row] for row in grid] != support:
            return False
        if not all(sidon(c) for row in grid for c in row):
            return False
        if side == "A":
            return all(quotient_ok(c) for c in grid[0])
        histogram = {}
        for row in grid:
            for cell in row:
                if len(cell) == 2:
                    o = orders[table[inverse[cell[0]]][cell[1]]]
                    histogram[o] = histogram.get(o,0)+1
        return histogram == prescribed_histogram

    blocks = {"A": [], "B": []}
    for start in range(0,q,12):
        ids = list(range(start,min(start+12,q)))
        gallery = [[[g] for g in ids]]
        xx,_,_ = lifted_matrices(P,gallery,[[[]]])
        _,zz,_ = lifted_matrices(P,[[[]]],gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def dependencies(columns, order):
        pivots, basis = {}, []
        for c in order:
            value, word = columns[c],1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value,word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    def sample(columns, trials, diagnostic, seeds=()):
        """Persistent genuine null words, not an exhaustive distance search."""
        width = len(columns)
        order = list(range(width))
        retained = sorted({w for w in seeds if w},
                          key=lambda w:(w.bit_count(),w))[:24]
        best = retained[0] if retained else None

        def note(word, additions):
            nonlocal best
            if not word:
                return
            if best is None or word.bit_count() < best.bit_count():
                best = word
            if word.bit_count() <= best.bit_count()+4:
                additions.append(word)
                if len(additions) > 192:
                    additions[:] = sorted(set(additions),
                                          key=int.bit_count)[:32]

        for word in retained:
            assert syndrome(columns,word) == 0
        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 3 == 2:
                positions = list(_bits(diagnostic.choice(retained[:6])))
                diagnostic.shuffle(positions)
                preferred = set(positions[:max(1,len(positions)//2)])
                order.sort(key=lambda c:c not in preferred)
            basis = dependencies(columns,order)
            additions = []
            for word in basis:
                note(word,additions)
            light = sorted(set(retained+basis),
                           key=lambda w:(w.bit_count(),w))[:16]
            moves = basis+retained
            for word in light:
                for other in moves:
                    note(word ^ other,additions)
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    before = word.bit_count()
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    note(word,additions)
                    if word.bit_count() == before:
                        break
            if len(basis) >= 3:
                for _ in range(96):
                    a,b,c = diagnostic.sample(basis,3)
                    note(a ^ b ^ c,additions)
            retained = sorted(
                set(retained+light+additions+([best] if best else [])),
                key=lambda w:(w.bit_count(),w))[:24]
            if best is not None and best.bit_count() <= 4:
                break
        if best is None:
            raise ValueError("verified component restriction has no found word")
        assert best and syndrome(columns,best) == 0
        return best,retained

    def component(grid, side):
        nr,nc = len(grid),len(grid[0])
        rows = [0]*(nr*q)
        for i,row in enumerate(grid):
            for j,cell in enumerate(row):
                for g in cell:
                    for r,value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        return rows

    def inspect(grid, side):
        if not legal(grid,side):
            return None
        rows = component(grid,side)
        width = len(grid[0])*q
        quality = _component_quality(rows,width)
        if quality is None:
            return None
        cycles,columns = quality
        rank = gf2_rank(rows)
        if rank < len(rows)-(8 if side == "A" else 12):
            return None

        words = []
        # Complete single-block enumeration when its actual nullity is small.
        for j in range(len(grid[0])):
            basis = dependencies(columns,range(j*q,(j+1)*q))
            if len(basis) <= 12:
                word = 0
                shortest = None
                for t in range(1,1 << len(basis)):
                    word ^= basis[(t & -t).bit_length()-1]
                    if shortest is None or word.bit_count() < shortest.bit_count():
                        shortest = word
                if shortest is not None:
                    words.append(shortest)
            else:
                words.extend(sorted(basis,key=int.bit_count)[:4])
        if any(word.bit_count() <= 4 for word in words):
            return None

        restrictions = []
        for orbits in partitions:
            masks = [sum(1 << (j*q+c) for c in orbit)
                     for j in range(len(grid[0])) for orbit in orbits]
            compressed = [syndrome(columns,part) for part in masks]
            restrictions.append(dict(
                masks=masks,columns=compressed,words=[],
                d=width+1))
        four,six = cycles
        cost = (800+200*policy)*four+(1+policy)*six
        return dict(
            grid=copy_grid(grid),rows=rows,columns=columns,rank=rank,
            cycles=cycles,cost=cost,restrictions=restrictions,
            words=sorted(set(words),key=int.bit_count),
            d=min(map(int.bit_count,words),default=width+1),
            signature=tuple(tuple(sorted(c)) for row in grid for c in row))

    def challenge(rec, side, full_trials, fixed_trials, turn):
        offset = 100003*int(side == "B")
        word,beam = sample(
            rec["columns"],full_trials,
            random.Random(seed+offset+700001*turn),rec["words"])
        found = list(rec["words"])+beam+[word]
        rec["d"] = min(rec["d"],word.bit_count())
        for at,restricted in enumerate(rec["restrictions"]):
            compressed,compressed_beam = sample(
                restricted["columns"],fixed_trials,
                random.Random(seed+offset+1700021*turn+1009*at),
                restricted["words"])
            restricted["words"] = compressed_beam
            expanded = []
            for candidate in compressed_beam:
                actual = 0
                for c in _bits(candidate):
                    actual |= restricted["masks"][c]
                assert actual and syndrome(rec["columns"],actual) == 0
                expanded.append(actual)
            actual = 0
            for c in _bits(compressed):
                actual |= restricted["masks"][c]
            size = len(partitions[at][0])
            assert actual.bit_count() == size*compressed.bit_count()
            assert syndrome(rec["columns"],actual) == 0
            restricted["d"] = min(restricted["d"],actual.bit_count())
            rec["d"] = min(rec["d"],actual.bit_count())
            found.extend(expanded+[actual])
        rec["words"] = sorted(
            set(found),key=lambda w:(w.bit_count(),w))[:28]
        assert all(syndrome(rec["columns"],w) == 0 for w in rec["words"])

    def pool(side):
        seen,records = set(),[]
        positions = [(i,j) for i,row in enumerate(supports[side])
                     for j in range(len(row))]
        known = set()

        def remember(grid):
            signature = tuple(tuple(sorted(c)) for row in grid for c in row)
            if signature in seen:
                return None
            seen.add(signature)
            rec = inspect(grid,side)
            return rec

        def redraw(grid, i, j):
            old = grid[i][j]
            wanted = (orders[table[inverse[old[0]]][old[1]]]
                      if side == "B" and len(old) == 2 else None)
            grid[i][j] = draw(len(old),side,wanted)
            assert legal(grid,side)

        for _ in range(48 if side == "A" else 64):
            rec = remember(fresh(side))
            if rec is not None:
                records.append(rec)
        if not records:
            return []
        records.sort(key=lambda r:(r["cost"],-r["rank"]))
        starts = records[:3]
        steps = 48 if side == "A" else 80
        for initial in starts:
            current = initial
            for step in range(steps):
                grid = copy_grid(current["grid"])
                if side == "B" and step % 9 == 4:
                    bins = [(i,j) for i,j in positions
                            if len(grid[i][j]) == 2]
                    (i,j),(u,v) = rng.sample(bins,2)
                    grid[i][j],grid[u][v] = grid[u][v],grid[i][j]
                else:
                    i,j = rng.choice(positions)
                    redraw(grid,i,j)
                rec = remember(grid)
                if rec is None:
                    continue
                delta = rec["cost"]-current["cost"]
                temperature = q*(8*(1-step/steps)**2+0.15)
                if delta <= 0 or rng.random() < exp(
                        min(0.0,-delta/temperature)):
                    current = rec
                    records.append(rec)
                    records.sort(key=lambda r:(r["cost"],-r["rank"]))
                    del records[48:]

        def refresh(rec):
            for word in known:
                if (word.bit_count() < rec["d"]
                        and syndrome(rec["columns"],word) == 0):
                    rec["d"] = word.bit_count()
                    rec["words"].append(word)
            rec["words"] = sorted(set(rec["words"]),
                                  key=lambda w:(w.bit_count(),w))[:28]

        def merit(rec):
            refresh(rec)
            cap = 32 if side == "A" else 24
            return (min(cap,rec["d"]),-rec["cost"],
                    min(cap+4,rec["d"]),rec["rank"])

        selected = []
        for rec in sorted(records,key=lambda r:(r["cost"],-r["rank"])):
            if not _no_kernel4(rec["columns"]):
                continue
            challenge(rec,side,4,3,1)
            known.update(rec["words"][:4])
            selected.append(rec)
            if len(selected) == (8 if side == "A" else 12):
                break
        if not selected:
            return []
        current = max(selected,key=merit)

        for step in range(12 if side == "A" else 20):
            refresh(current)
            word = rng.choice(current["words"][:6])
            active = {c//q for c in _bits(word)}
            targeted = [p for p in positions if p[1] in active]
            grid = copy_grid(current["grid"])
            i,j = rng.choice(targeted if targeted else positions)
            redraw(grid,i,j)
            if step % 5 == 4:
                remaining = [p for p in positions if p != (i,j)]
                u,v = rng.choice(remaining)
                redraw(grid,u,v)
            rec = remember(grid)
            if rec is None or not _no_kernel4(rec["columns"]):
                continue
            rec["words"].extend(
                w for w in current["words"]
                if syndrome(rec["columns"],w) == 0)
            challenge(rec,side,4,3,10+step)
            known.update(rec["words"][:4])
            selected.append(rec)
            if merit(rec) > merit(current):
                current = rec
            if len(known) > 96:
                known = set(sorted(known,key=int.bit_count)[:64])

        finalists = sorted(selected,key=merit,reverse=True)[:4]
        for rec in finalists:
            challenge(rec,side,24,12,101)
        finalists.sort(key=merit,reverse=True)
        finalists = finalists[:2]
        for rec in finalists:
            # EVERY contender receives equal independent deep testing.
            # Earlier witnesses remain binding on this actual matrix.
            challenge(rec,side,48+8*(cohort % 3),24,211)
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

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_challenge(a,b,hx,hz,n):
        """Genuine logical upper readings modulo ORIGINAL stabilizer spans."""
        readings = []
        for kind,rec,opposite,stabilizers in (
                ("Z",a,hx,hz),("X",b,hz,hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in rec["words"][:5]:
                assert syndrome(rec["columns"],word) == 0
                for location in range(5 if kind == "Z" else 2):
                    actual = (
                        sum(((word >> (j*q)) & mask)
                            << ((5*j+location)*q) for j in range(2))
                        if kind == "Z" else word << (5*location*q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual,pivots):
                        candidates.add(actual)
            reading = n+1
            diagnostic = random.Random(
                seed+(4100011 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            diagnostic.shuffle(reducers)
            for actual in sorted(candidates,key=int.bit_count)[:6]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    nearby = sorted(
                        stabilizers,key=lambda row:(row & actual).bit_count(),
                        reverse=True)[:32]
                    winner = actual
                    for at,first in enumerate(nearby):
                        for second in nearby[:at]:
                            candidate = actual ^ first ^ second
                            if candidate.bit_count() < winner.bit_count():
                                winner = candidate
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual,pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading,actual.bit_count())
            # n+1 is a missed embedding, never a distance lower bound.
            readings.append(reading)
        return tuple(readings)

    for restart in range(2):
        aa,bb = pool("A"),pool("B")
        evaluated = []
        for a in aa:
            for b in bb:
                hx,hz,n = lifted_matrices(P,a["grid"],b["grid"])
                k = n-gf2_rank(hx)-gf2_rank(hz)
                assert n == 12*q == 1152
                if not (k >= 3*q and 225*n <= 1000*k < 275*n):
                    continue
                actual_a,actual_b = _verified_components(hx,hz,q)
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_challenge(a,b,hx,hz,n)
                detected = min(24,a["d"],b["d"],*logicals)
                quality = (k*detected*detected,detected,
                           -a["cost"]-b["cost"],
                           min(32,a["d"])+min(28,b["d"]))
                evaluated.append((quality,a,b,hx,hz,k,logicals))
        evaluated.sort(key=lambda item:item[0],reverse=True)

        for quality,a,b,cached_hx,cached_hz,k,logicals in evaluated:
            if not _no_kernel4(_syndrome_columns(cached_hx,1152)):
                continue
            if not _no_kernel4(_syndrome_columns(cached_hz,1152)):
                continue
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            assert legal(a["grid"],"A") and legal(b["grid"],"B")
            assert all(len(set(c)) == len(c)
                       for grid in (A,B) for row in grid for c in row)
            hx,hz,n = lifted_matrices(G,A,B)
            assert hx == cached_hx and hz == cached_hz
            assert n == 1152 and k == n-gf2_rank(hx)-gf2_rank(hz)
            assert max(row.bit_count() for row in hx+hz) == 12 <= W_MAX
            assert css_ok(hx,hz)
            return dict(
                A=A,B=B,idx=idx,k=k,quality=quality,
                card=(topology,policy,occupancy,cohort),
                B_binomial_order_histogram=dict(prescribed_histogram),
                component_readings=(a["d"],b["d"]),
                component_ranks=(a["rank"],b["rank"]),
                component_cycles=(a["cycles"],b["cycles"]),
                central_fixed_readings=(
                    tuple(r["d"] for r in a["restrictions"]),
                    tuple(r["d"] for r in b["restrictions"])),
                central_restriction_sizes=(2,2,2,4),
                verified_sector_one_logical_readings=logicals,
                target_cell=0.250)
    raise ValueError("matrix-group centre-challenged pairs missed full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored matrix-group card, without extra draws."""
    cell = list(ctx[side][i][j])
    assert 0 < len(cell) <= CELL_MAX
    assert len({G.idx(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
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
