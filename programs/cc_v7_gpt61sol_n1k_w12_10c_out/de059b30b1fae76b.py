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
CONCEPT = ("Hypothesis: on split-metacyclic Z97:Z3(s=35), order 291, A=B=(2,1), zero-four-cycle paired-coset quartics with only the verified three quotient deficits produce five new k=303 codes whose measured fitness exceeds 982.5226 in cell 0.200. Five readings at least 69 would suffice. Finite witness readings remain upper bounds, not distance certificates.")
GROUP = ("Keep the exact Z97:Z3(s=35) definition and q=291 unchanged. Its horizontal bed leads cell 0.200 at 982.5226 and its tall bed reached 965.8564. The normal C97 subgroup supplies three coset-constant coordinates. Even term occupancy in every coset annihilates this restriction, but authoritative full ranks, not quotient reasoning alone, determine the submitted dimension.")
SHAPES = ("Change BOTH sides from (1,2) to (2,1). Factor five retains n=1455, k>=291 and guaranteed rate 0.200; q_max=299 and q remains 291. Equal distance-bearing transpose components each have 582 variables and 291 checks. Parent minima occupied sector-one single-Acol or single-Bcol strips, commonly Acol1 across Bcol0/1 and Bcol1 across Acol0/1. The analogous tall component families occupy sector two. Require actual k=303, rate 0.208247, inside [0.175,0.225); new quotient-sector logicals are explicitly challenged.")
FILLING = ("Every cell has four distinct terms with occupancy (2,2,0), so BOTH CSS maxima remain 4+8=12. Normalize atoms to [e,x,y,x^t*y], conservatively identify signed-action/inversion images of doublet ratios, and screen actual cell ranks before choosing six classes. Twenty unordered coupling-profile pairs prescribe deliberate designs. Fresh labels and whole-cell repair preserve the assigned ratio. Require connected authoritative transpose components, zero Tanner four-cycles, and component/cell ranks q-3. Verify their coset-constant weight-97 kernel words, accumulate unrestricted elimination/cancellation witnesses, and independently deepen finalists. Complete pairings use exact k=303 and commuting non-stabilizer sector-one and sector-two words reduced by ORIGINAL stabilizers, including bounded uphill routes. Native component agreement, full kernel-four exclusion, weights and CSS gate submission. Neither ratio classes nor finite searches certify distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_21x21_pairedcoset_rank303_jointbeam_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 291

def custom_ops(q):
    # Same split-metacyclic construction rule, resized for factor 5.
    # 35^2=61 and 35^3=1 mod 97; the action has order three.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 291")
    m, r, s = 97, 3, 35
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (2, 1)               # (rb, cb)


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


def _retired_horizontal_coupling_ctx(rng, G, idx):
    """Inactive historical horizontal experiment; NEW_CTX below replaces it."""
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
    """Twenty paired-coset tall cards with binding rank/logical challenges."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (2, 1)
    mask = (1 << q) - 1
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    card, cohort = idx % 20, idx // 20
    trade = (card + cohort) % 4
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xE09721D4 + 49979687 * (idx + 1)))

    # Cache native operations only. All binary action blocks come from
    # lifted_matrices, never from independently implemented lift formulas.
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
    blocks = {"A": [], "B": []}
    zero = [[[]]]
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g]] for g in ids]
        _, az, _ = lifted_matrices(P, gallery, zero)
        bx, _, _ = lifted_matrices(P, zero, gallery)
        assert len(az) == len(bx) == q
        for j in range(len(ids)):
            for side, source in (("A", az), ("B", bx)):
                block = [(row >> ((1 + j) * q)) & mask
                         for row in source]
                assert all(row.bit_count() == 1 for row in block)
                blocks[side].append(block)

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def component(grid, side):
        rows = [0] * q
        for j, base_row in enumerate(grid):
            assert len(base_row) == 1
            for g in base_row[0]:
                for r, value in enumerate(blocks[side][G.idx(g)]):
                    rows[r] ^= value << (j * q)
        return rows

    def sidon(cell):
        if len(cell) != 4 or len(set(cell)) != 4:
            return False
        ids = [G.idx(g) for g in cell]
        left = {table[inverse[a]][b]
                for a in ids for b in ids if a != b}
        right = {table[a][inverse[b]]
                 for a in ids for b in ids if a != b}
        return len(left) == len(right) == 12

    signed = tuple(
        sign * pow(35, b, 97) % 97
        for sign in (1, -1) for b in range(3))

    def ratio_key(t):
        # Conservatively collapse proposal classes, not submitted codes.
        ti = pow(t, -1, 97)
        return min(action * value % 97
                   for action in signed for value in (t, ti))

    catalog = []
    for t in range(1, 97):
        if t != ratio_key(t) or ratio_key(t) == 1:
            continue
        cell = [(0, 0), (1, 0), (0, 1), (t, 1)]
        if not sidon(cell):
            continue
        ranks = [gf2_rank(component([[cell]], side))
                 for side in ("A", "B")]
        if min(ranks) == q - 3:
            catalog.append(t)
    if len(catalog) < 6:
        raise ValueError("insufficient full-rank paired-coset ratio classes")

    atom_types = tuple(
        catalog[(i * len(catalog) // 6 + cohort) % len(catalog)]
        for i in range(6))
    assert len(set(atom_types)) == 6
    profiles = ((0, 1), (0, 2), (1, 3),
                (2, 4), (3, 5), (4, 5))
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    assert len(plans) == 20
    pa, pb = plans[card]

    def fresh_cell(atom):
        for _ in range(256):
            o, a = rng.randrange(97), rng.randrange(97)
            d, phase = rng.randrange(1, 97), rng.randrange(3)
            cell = [
                (o, phase), ((o + d) % 97, phase),
                (a, (phase + 1) % 3),
                ((a + atom * d) % 97, (phase + 1) % 3),
            ]
            if sidon(cell):
                return cell
        raise ValueError("paired-coset Sidon recipe exhausted")

    def fresh(profile):
        return [[fresh_cell(atom_types[k])] for k in profiles[profile]]

    quotient_masks = [
        sum(1 << G.idx((a, b)) for a in range(97))
        for b in range(3)
    ]
    assert gf2_rank(quotient_masks) == 3

    def eight_cycles(rows, columns):
        checks = [list(_bits(row)) for row in rows]
        neighbors = [list(_bits(column)) for column in columns]
        rooted = 0
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
                            if end in (root, v):
                                continue
                            previous = paths.setdefault(end, [])
                            for d1, u, d2 in previous:
                                if (u != v and c1 not in (d1, d2)
                                        and c2 not in (d1, d2)):
                                    rooted += 1
                            previous.append((c1, v, c2))
        assert q * rooted % 4 == 0
        return q * rooted // 4

    def geometry(stats):
        four, six, eight = stats
        if trade == 0:
            return (four, six, eight)
        if trade == 1:
            return (four, eight, six)
        if trade == 2:
            return (four, 10 * six + eight, six)
        return (four, 30 * six + eight, eight)

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in row[0]))
                     for row in grid)

    def inspect(grid, side):
        rows = component(grid, side)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        stats, columns = quality
        if stats[0] != 0:
            return None
        assert all(c.bit_count() == 4 for c in columns)
        rank = gf2_rank(rows)
        ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if rank != q - 3 or ranks != (q - 3, q - 3):
            return None

        # Three independent verified localized null words per cell,
        # together with rank q-3, describe its ENTIRE localized kernel.
        # Its seven nonzero words have weights 97, 194 or 291.
        words = []
        for j in range(2):
            for word in quotient_masks:
                actual = word << (j * q)
                if syndrome(columns, actual):
                    raise ValueError("paired-coset quotient word failed verification")
                words.append(actual)
        assert all(w.bit_count() == 97 for w in words)
        stats = stats + (eight_cycles(rows, columns),)
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            rank=rank, cell_ranks=ranks, cycles=stats,
            cost=geometry(stats), signature=signature(grid),
            words=words, d=97)

    def challenge(rec, trials, offset):
        """Persistent genuine kernel words; failures never certify absence."""
        local = random.Random(seed + offset)
        columns = rec["columns"]
        retained = sorted(set(rec["words"]),
                          key=lambda w: (w.bit_count(), w))[:24]
        best = retained[0]
        order = list(range(width))

        def observe(word, found):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            if weight < best.bit_count():
                best = word
            if weight <= best.bit_count() + 6:
                found.append(word)
                if len(found) > 192:
                    found[:] = sorted(set(found),
                                      key=int.bit_count)[:32]

        for turn in range(trials):
            local.shuffle(order)
            if retained and turn % 4 == 3:
                hint = list(_bits(local.choice(retained[:6])))
                local.shuffle(hint)
                preferred = set(hint[:max(1, len(hint) // 2)])
                order.sort(key=lambda c: c not in preferred)
            pivots, basis, independent = {}, [], []
            found = []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        independent.append(c)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word, found)

            light = sorted(set(retained + basis),
                           key=lambda w: (w.bit_count(), w))[:16]
            moves = retained + basis
            for word in light:
                for other in moves:
                    observe(word ^ other, found)

            if trials >= 32 and turn % 4 == 0 and independent:
                chosen = sorted(basis, key=int.bit_count)[:72]
                coords = local.sample(independent, min(9, len(independent)))
                hash_mask = sum(1 << c for c in coords)
                buckets = {}
                for at, first in enumerate(chosen):
                    for second in chosen[:at]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for previous in bucket:
                            observe(pair ^ previous, found)
                        if len(bucket) < 12:
                            bucket.append(pair)

            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    before = word.bit_count()
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    if word.bit_count() == before:
                        break
                observe(word, found)
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = local.sample(basis, 3)
                    observe(a ^ b ^ c, found)
            retained = sorted(set(retained + light + found + [best]),
                              key=lambda w: (w.bit_count(), w))[:24]
            if best.bit_count() <= 24:
                break

        assert best and syndrome(columns, best) == 0
        rec["words"] = sorted(set(rec["words"] + retained + [best]),
                              key=lambda w: (w.bit_count(), w))[:24]
        assert all(syndrome(columns, w) == 0 for w in rec["words"])
        rec["d"] = min(rec["d"], best.bit_count())

    def pool(profile, side, restart):
        offset = 100003 * int(side == "B") + 300007 * restart
        seen, records, bank = set(), [], set()

        def probe(rec, trials, turn):
            challenge(rec, trials, offset + 700001 * turn)
            bank.update(rec["words"][:4])
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)

        def merit(rec):
            for word in bank:
                if (word.bit_count() < rec["d"]
                        and syndrome(rec["columns"], word) == 0):
                    rec["d"] = word.bit_count()
            return (min(76, rec["d"]),
                    tuple(-x for x in rec["cost"]), min(96, rec["d"]))

        for _ in range(128 + 8 * min(cohort, 2)):
            rec = inspect(fresh(profile), side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
            if len(records) >= 64:
                break
        records.sort(key=lambda r: r["cost"])
        selected = records[:8]
        if not selected:
            return []
        for rec in selected:
            probe(rec, 12, 0)
        current = max(selected, key=merit)

        for step in range(20):
            word = rng.choice(current["words"][:6])
            occupied = [c // q for c in _bits(word)]
            j = rng.choice(occupied) if occupied else rng.randrange(2)
            proposals = []
            for _ in range(6):
                grid = copy_grid(current["grid"])
                grid[j][0] = fresh_cell(atom_types[profiles[profile][j]])
                cols = _syndrome_columns(component(grid, side), width)
                boundaries = [syndrome(cols, w).bit_count()
                              for w in current["words"][:6]]
                key = (sum(v > 0 for v in boundaries),
                       min(boundaries), sum(boundaries))
                proposals.append((key, grid))
            proposals.sort(key=lambda item: item[0], reverse=True)
            proposed = None
            for _, grid in proposals:
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side)
                if proposed is not None:
                    break
            if proposed is None:
                continue
            proposed["words"] += [
                w for w in current["words"]
                if syndrome(proposed["columns"], w) == 0]
            probe(proposed, 8, step + 1)
            selected.append(proposed)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.12):
                current = proposed

        finalists = sorted(selected, key=merit, reverse=True)[:4]
        for rec in finalists:
            probe(rec, 48, 101)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for rec in finalists:
            # Independent confirmation retains every previous witness.
            probe(rec, 128 + 16 * min(cohort, 2), 211)
            assert _no_kernel4(rec["columns"])
        finalists.sort(key=merit, reverse=True)
        return finalists

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

    def logical_challenge(a, b, hx, hz, n):
        """Exactly verified logicals, including the new quotient sector.

        Bounded uphill moves only find representatives; they do not
        prove that lighter representatives or other families are absent.
        """
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            pivots = span(stabilizers)
            located = set()
            for word in rec["words"][:4]:
                for location in range(2):
                    actual = (
                        sum(((word >> (i * q)) & mask)
                            << ((1 + 2 * i + location) * q)
                            for i in range(2))
                        if kind == "X"
                        else word << ((1 + 2 * location) * q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        located.add(actual)

            quotient = set()
            for subset in range(1, 8):
                actual = 0
                for t in range(3):
                    if (subset >> t) & 1:
                        actual ^= quotient_masks[t]
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                if outside(actual, pivots):
                    quotient.add(actual)

            starts = sorted(located, key=int.bit_count)[:4]
            starts += sorted(quotient, key=int.bit_count)[:2]
            reading = n + 1
            for initial in starts:
                best = initial
                ceiling = initial.bit_count() + 4
                frontier, visited = [initial], {initial}
                for _ in range(3):
                    candidates = []
                    for actual in frontier:
                        near = sorted(
                            stabilizers,
                            key=lambda row: (row & actual).bit_count(),
                            reverse=True)[:32]
                        for row in near:
                            proposed = actual ^ row
                            if proposed in visited:
                                continue
                            visited.add(proposed)
                            if proposed.bit_count() <= ceiling:
                                candidates.append(proposed)
                                if proposed.bit_count() < best.bit_count():
                                    best = proposed
                    frontier = sorted(set(candidates),
                                      key=lambda w: (w.bit_count(), w))[:6]
                    if not frontier:
                        break

                actual = best
                for _ in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    near = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:28]
                    winner = actual
                    for at, first in enumerate(near):
                        for second in near[:at]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
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
        aa, bb = pool(pa, "A", restart), pool(pb, "B", restart)
        evaluated = []
        for a in aa:
            for b in bb:
                hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                assert n == 5 * q == 1455
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if k != 303 or not (7 * n <= 40 * k < 9 * n):
                    continue
                actual_a = [
                    sum(((hz[t] >> ((1 + 2 * i) * q)) & mask)
                        << (i * q) for i in range(2))
                    for t in range(q)]
                actual_b = [
                    (hx[t] >> q) & ((1 << width) - 1)
                    for t in range(q)]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_challenge(a, b, hx, hz, n)
                detected = min(a["d"], b["d"], *logicals)
                capped = min(76, detected)
                key = (
                    k * capped * capped, min(96, detected),
                    tuple(-(x + y) for x, y in zip(a["cost"], b["cost"])),
                    min(96, a["d"]) + min(96, b["d"]))
                evaluated.append((key, a, b, hx, hz, k, logicals))
        evaluated.sort(key=lambda item: item[0], reverse=True)

        for key, a, b, hx, hz, k, logicals in evaluated:
            if not _no_kernel4(_syndrome_columns(hx, 1455)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, 1455)):
                continue
            A, B = a["grid"], b["grid"]
            assert [[len(c) for c in row] for row in A] == support
            assert [[len(c) for c in row] for row in B] == support
            assert max(row.bit_count() for row in hx + hz) == 12
            assert css_ok(hx, hz)
            return dict(
                A=A, B=B, idx=idx, k=k,
                card=(pa, pb, trade, cohort),
                doublet_ratio_types=atom_types,
                ratio_profiles=(profiles[pa], profiles[pb]),
                component_readings=(a["d"], b["d"]),
                component_ranks=(a["rank"], b["rank"]),
                cell_ranks=(a["cell_ranks"], b["cell_ranks"]),
                component_cycles=(a["cycles"], b["cycles"]),
                verified_logical_upper_readings=logicals,
                pairing_merit=key,
                localized_kernel_dimension=3,
                target_cell=0.200)
    raise ValueError("paired-coset tall pair missed verified k=303/full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete rank-gaining paired-coset quartic cell."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 4
    assert sorted(sum(g[1] == b for g in cell)
                  for b in range(3)) == [0, 2, 2]
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Coupled full-coset quartics with persistent component challenges uniformly exclude sub-30 logical readings; current horizontal slots 30 and 33 instead screened at 27 with k=295 and w=12.
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
