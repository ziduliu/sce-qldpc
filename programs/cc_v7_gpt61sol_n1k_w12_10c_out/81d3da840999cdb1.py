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
CONCEPT = ("Hypothesis: on dicyclic Dic22, order 88, A=(1,3), B=(2,5), long normal-subgroup ratios and independently challenged components improve cell 0.350 beyond fitness 50.824. Five new readings fourteen score at least 69.176; rank gain at reading twelve can also improve the incumbent. Measurement can refute these targets; finite searches remain upper readings.")
GROUP = ("Jump to the dicyclic presentation x^44=1, y^2=x^22, y*x*y^-1=x^-1, order 88. This nonsplit dicyclic construction differs from the parent's order-291 split-metacyclic group. Its cyclic index-two subgroup supplies relative orders 11,22,44, while every outside element has order four. This order-88 bed is absent from BEDS. Related order-88 and order-84 cell-0.350 readings motivate screening, not a claim of distance transfer.")
SHAPES = ("Change A to (1,3) and B to (2,5). Factor seventeen gives n=1496, k>=528 and guaranteed rate 6/17=0.352941; q=88 attains this shape's size-law maximum. Unequal components have 264 variables/88 checks and 440 variables/176 checks, nominal rates 2/3 and 3/5. Parent minima occupied sector-one strips, often Acol1 across both Bcols or Bcol1 across both Acols. Challenge both expanded sector-one families and enforce [0.325,0.375) using authoritative full ranks.")
FILLING = ("A counts are (3,3,3); B column degrees are (2,3,3,3,3), with both B row totals seven. Consequently HX<=9+3=12 and HZ<=7+3=10. Twenty cards cross four A normal-ratio order prescriptions with five B zero-cell budgets. A cubics contain a normal ratio of prescribed order and an outside term; B binomial ratios have order 11,22,44, never four. Bidirectional Sidon proposals compete using scaffold-derived connectivity, cycle geometry and exact ranks. Retain genuine kernel witnesses, repair implicated occupied cells, independently deepen every finalist, and challenge finite regular-translation cancellations and budgeted parity completions. Compare complete pairings through commuting non-stabilizer strip logicals reduced by ORIGINAL checks. Native component agreement, full kernel-four rejection, weights, CSS and measured-rate gates remain binding. No support card or finite reading certifies distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Dic22_13x25_longratio_originalquotient_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 88

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic22 requires order 88")
    # (a,b)=x^a*y^b; x^44=1, y^2=x^22, y*x*y^-1=x^-1.
    m, rotation_order = 22, 44
    els = [(a,b) for b in range(2) for a in range(rotation_order)]

    def red(g):
        return (int(g[0]) % rotation_order, int(g[1]) % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        carry = m if b == d == 1 else 0
        return ((a + (c if b == 0 else -c) + carry)
                % rotation_order, (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-a if b == 0 else a+m) % rotation_order, b)

    def idx(g):
        a,b = red(g)
        return rotation_order*b+a

    return dict(elements=els, ident=(0,0),
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


def _retired_coupled_quartic_ctx(rng, G, idx):
    """Inactive order-291 predecessor; never called on the dicyclic group."""
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
    """Twenty dicyclic order/support cards with original-code challenges."""
    from functools import lru_cache
    from math import gcd

    q = G.q
    assert q == 88 and A_SHAPE == (1,3) and B_SHAPE == (2,5)
    card, cohort = idx % 20, idx // 20
    concentration, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62) ^ (0xD1C2235 + 15485863*(idx+1))
    local = random.Random(seed)
    mask = (1 << q)-1
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g,h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        def __init__(self):
            self.q, self.e = q, ident
            self.elements = list(range(q))
        def idx(self, g):
            return g
        def mul(self, g, h):
            return table[g][h]
        def inv(self, g):
            return inverse[g]

    P = Probe()
    # Only the scaffold constructs regular-action matrices.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 12):
        ids = list(range(start, min(q, start+12)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            for side, rows in (("A",xx), ("B",zz)):
                blocks[side].append([
                    (row >> (j*q)) & mask for row in rows])

    normal = {
        order: [G.idx((a,0)) for a in range(1,44)
                if 44//gcd(44,a) == order]
        for order in (11,22,44)
    }
    outside = [G.idx((a,1)) for a in range(44)]
    long_normal = normal[11]+normal[22]+normal[44]
    order_plans = (
        (44,44,44), (44,44,22),
        (44,22,22), (44,44,11),
    )
    requested = order_plans[policy]

    def copy_grid(grid):
        return [[list(c) for c in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(c)) for row in grid for c in row)

    def syndrome(columns, word):
        value = 0
        for v in _bits(word):
            value ^= columns[v]
        return value

    def sidon(cell):
        if len(cell) != len(set(cell)):
            return False
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        return len(left) == len(right) == len(cell)*(len(cell)-1)

    def draw(count, side, column):
        if count == 0:
            return []
        if count == 1:
            return [local.randrange(q)]
        if count == 2:
            # An outside ratio has order four and immediately supplies
            # a component four-cycle word. Do not propose it here.
            root = local.randrange(q)
            ratio = local.choice(long_normal)
            return [root, table[root][ratio]]
        assert count == 3
        choices = normal[requested[column]] if side == "A" else long_normal
        for _ in range(256):
            g, h = local.choice(choices), local.choice(outside)
            if table[g][h] == table[h][g]:
                continue
            cell = [ident,g,h]
            if not sidon(cell):
                continue
            if side == "B":
                root = local.randrange(q)
                cell = [table[root][x] for x in cell]
            return cell
        raise ValueError("dicyclic Sidon cubic prescription exhausted")

    def b_support():
        degrees = (2,3,3,3,3)

        @lru_cache(None)
        def feasible(j, upper, zeros):
            if upper < 0 or zeros < 0:
                return False
            if j == 5:
                return upper == zeros == 0
            d = degrees[j]
            return any(feasible(j+1, upper-a,
                                zeros-int(a == 0 or a == d))
                       for a in range(d+1))

        assert feasible(0,7,concentration)
        upper, zeros = 7, concentration
        grid = [[],[]]
        for j,d in enumerate(degrees):
            choices = [
                a for a in range(d+1)
                if feasible(j+1, upper-a,
                            zeros-int(a == 0 or a == d))]
            a = local.choice(choices)
            grid[0].append(a)
            grid[1].append(d-a)
            upper -= a
            zeros -= int(a == 0 or a == d)
        assert [sum(row) for row in grid] == [7,7]
        assert _grid_weights([[3,3,3]],grid) == 12
        assert _no_dead_lines([[3,3,3]],grid)
        return grid

    def fresh(side):
        support = [[3,3,3]] if side == "A" else b_support()
        return [[draw(count,side,j) for j,count in enumerate(row)]
                for row in support]

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i,row in enumerate(grid):
            for j,cell in enumerate(row):
                for g in cell:
                    for t,value in enumerate(blocks[side][g]):
                        rows[i*q+t] ^= value << (j*q)
        quality = _component_quality(rows,nc*q)
        if quality is None:
            return None
        cycles, columns = quality
        rank = gf2_rank(rows)
        # Preferences omit proposals, never substitute a rank formula
        # for the final authoritative quantum ranks.
        if side == "A" and rank < q-10:
            return None
        if side == "B" and rank < 2*q-6:
            return None
        cost = ((cycles[0],cycles[1]) if policy % 2 == 0 else
                (500*cycles[0]+cycles[1],cycles[0]))
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            rank=rank, cycles=cycles, cost=cost,
            signature=signature(grid), side=side,
            words=[], d=nc*q+1)

    def challenge(rec, trials, offset):
        """Retained genuine kernel words; high readings certify nothing."""
        diagnostic = random.Random(seed+offset)
        columns = rec["columns"]
        width = len(columns)
        retained = sorted(set(rec["words"]),key=int.bit_count)[:24]
        best = retained[0] if retained else None
        order = list(range(width))

        def observe(word, grown):
            nonlocal best
            if not word:
                return
            if best is None or word.bit_count() < best.bit_count():
                best = word
            if word.bit_count() <= best.bit_count()+4:
                grown.append(word)

        for word in retained:
            assert syndrome(columns,word) == 0
        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 4 == 3:
                active = list(_bits(diagnostic.choice(retained[:6])))
                preferred = set(diagnostic.sample(
                    active,max(1,len(active)//2)))
                order.sort(key=lambda v:v not in preferred)
            pivots, basis = {}, []
            grown = list(retained)
            for v in order:
                value, word = columns[v], 1 << v
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
                    observe(word,grown)
            light = sorted(set(retained+basis),key=int.bit_count)[:16]
            moves = retained+basis
            for first in light:
                for second in moves:
                    observe(first ^ second,grown)
                if len(grown) > 192:
                    grown = sorted(set(grown),key=int.bit_count)[:32]
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for _ in range(3):
                    before = word.bit_count()
                    for move in moves:
                        proposed = word ^ move
                        if proposed and proposed.bit_count() < word.bit_count():
                            word = proposed
                    observe(word,grown)
                    if word.bit_count() == before:
                        break
            retained = sorted(
                set(grown+([best] if best else [])),
                key=lambda w:(w.bit_count(),w))[:24]
            if best is not None and best.bit_count() <= 8:
                break
        if best is None:
            raise ValueError("verified component search found no kernel word")

        if trials >= 32:
            # Orbit words diagnose this matrix; they are not code slots.
            comparisons = sorted(
                set(retained+[best]),key=int.bit_count)[:16]
            grown = list(comparisons)
            for source in comparisons[:2]:
                positions = [divmod(v,q) for v in _bits(source)]
                for multiplier in range(q):
                    image = 0
                    for block,c in positions:
                        moved = (table[c][multiplier]
                                 if rec["side"] == "A"
                                 else table[multiplier][c])
                        image |= 1 << (block*q+moved)
                    assert syndrome(columns,image) == 0
                    for word in comparisons:
                        observe(image ^ word,grown)
            retained = sorted(set(grown+[best]),key=int.bit_count)[:24]

            # Budgeted parity completion. A budget stop is not exclusion.
            cap = min(14,best.bit_count()-1)
            if cap >= 5:
                singles, pairs = {}, {}
                incidence = [0]*len(rec["rows"])
                for v,column in enumerate(columns):
                    singles.setdefault(column,1 << v)
                    for r in _bits(column):
                        incidence[r] |= 1 << v
                    for u in range(v):
                        pairs.setdefault(column ^ columns[u],
                                         (1 << v) | (1 << u))
                priorities = [diagnostic.random() for _ in columns]
                roots = list(range(0,width,q))
                diagnostic.shuffle(roots)
                all_variables = (1 << width)-1
                maximum_degree = max(c.bit_count() for c in columns)

                for root in roots:
                    remaining = [max(1,6000//len(roots))]

                    def visit(value, word, allowed):
                        if remaining[0] <= 0:
                            return
                        remaining[0] -= 1
                        if not value:
                            observe(word,retained)
                            return
                        left = min(cap,best.bit_count()-1)-word.bit_count()
                        if left <= 0 or value.bit_count() > maximum_degree*left:
                            return
                        one = singles.get(value,0)
                        if one and one & allowed == one:
                            observe(word | one,retained)
                        pair = pairs.get(value,0)
                        if left >= 2 and pair and pair & allowed == pair:
                            observe(word | pair,retained)
                        left = min(cap,best.bit_count()-1)-word.bit_count()
                        if left <= 2:
                            return
                        chosen, smallest = 0,width+1
                        for r in _bits(value):
                            available = incidence[r] & allowed
                            count = available.bit_count()
                            if not count:
                                return
                            if count < smallest:
                                chosen,smallest = available,count
                        candidates = list(_bits(chosen))
                        candidates.sort(key=lambda v:(
                            -(columns[v] & value).bit_count(),priorities[v]))
                        live = allowed
                        for v in candidates:
                            bit = 1 << v
                            live &= ~bit
                            visit(value ^ columns[v],word | bit,live)
                            if remaining[0] <= 0:
                                break

                    visit(columns[root],1 << root,
                          all_variables & ~((1 << (root+1))-1))
                    if best.bit_count() <= 4:
                        break
        assert best and syndrome(columns,best) == 0
        rec["words"] = sorted(
            set(retained+[best]),key=lambda w:(w.bit_count(),w))[:24]
        assert all(syndrome(columns,w) == 0 for w in rec["words"])
        rec["d"] = min(rec["d"],best.bit_count())

    def merit(rec):
        return (min(14,rec["d"]),rec["d"],
                tuple(-x for x in rec["cost"]),-rec["rank"])

    def pool(side, restart):
        offset = 100003*int(side == "B")+300007*restart
        records, seen = [],set()
        for _ in range(64+4*min(cohort,2)):
            rec = inspect(fresh(side),side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda r:r["cost"])
        tested = []
        for rec in records:
            if not _no_kernel4(rec["columns"]):
                continue
            challenge(rec,8,offset)
            tested.append(rec)
            if len(tested) == 8:
                break
        if not tested:
            return []
        current = max(tested,key=merit)

        for step in range(24):
            source = local.choice(current["words"][:6])
            occupied = {v//q for v in _bits(source)}
            mutable = [(i,j) for i,row in enumerate(current["grid"])
                       for j,c in enumerate(row) if c and j in occupied]
            if not mutable:
                continue
            i,j = local.choice(mutable)
            proposals = []
            for _ in range(3):
                grid = copy_grid(current["grid"])
                grid[i][j] = draw(len(grid[i][j]),side,j)
                rec = inspect(grid,side)
                if rec is None or rec["signature"] in seen:
                    continue
                disruption = [
                    syndrome(rec["columns"],w).bit_count()
                    for w in current["words"][:6]]
                proposals.append((
                    (sum(v > 0 for v in disruption),sum(disruption)),rec))
            if not proposals:
                continue
            _,rec = max(proposals,key=lambda item:item[0])
            seen.add(rec["signature"])
            rec["words"] = [w for w in current["words"]
                            if syndrome(rec["columns"],w) == 0]
            if not _no_kernel4(rec["columns"]):
                continue
            challenge(rec,6,offset+700001+7919*step)
            tested.append(rec)
            if merit(rec) >= merit(current):
                current = rec

        finalists = sorted(tested,key=merit,reverse=True)[:3]
        # Equal independent budgets for every retained contender.
        for rec in finalists:
            challenge(rec,32,offset+1700021)
            challenge(rec,48+8*min(cohort,2),offset+2700011)
        finalists.sort(key=merit,reverse=True)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                previous = pivots.get(lead)
                if previous is None:
                    pivots[lead] = word
                    break
                word ^= previous
        return pivots

    def outside_span(word,pivots):
        while word:
            previous = pivots.get(word.bit_length()-1)
            if previous is None:
                return True
            word ^= previous
        return False

    def logical_challenge(a,b,hx,hz,n):
        """Finite upper readings in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind,rec,opposite,stabilizers in (
                ("Z",a,hx,hz),("X",b,hz,hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in rec["words"][:4]:
                for location in range(5 if kind == "Z" else 3):
                    actual = (sum(
                        ((word >> (j*q)) & mask) << ((5*j+location)*q)
                        for j in range(3)) if kind == "Z"
                        else word << (5*location*q))
                    assert actual >> n == 0
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside_span(actual,pivots):
                        candidates.add(actual)
            reading = n+1
            for actual in sorted(candidates,key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,key=lambda r:(r & actual).bit_count(),
                        reverse=True)[:24]
                    winner = actual
                    for at,first in enumerate(nearby):
                        for second in nearby[:at]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside_span(actual,pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading,actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    for restart in range(3):
        aa,bb = pool("A",restart),pool("B",restart)
        evaluated = []
        for a in aa:
            for b in bb:
                hx,hz,n = lifted_matrices(P,a["grid"],b["grid"])
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 1496 and k >= 528 and
                        325*n <= 1000*k < 375*n):
                    continue
                actual_a = [
                    sum(((hx[t] >> (5*j*q)) & mask) << (j*q)
                        for j in range(3)) for t in range(q)]
                actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_challenge(a,b,hx,hz,n)
                detected = min(14,a["d"],b["d"],*logicals)
                key = (k*detected*detected,detected,
                       -a["cycles"][0]-b["cycles"][0],
                       -a["cycles"][1]-b["cycles"][1])
                evaluated.append((key,a,b,hx,hz,k,logicals))
        evaluated.sort(key=lambda item:item[0],reverse=True)
        for key,a,b,cached_hx,cached_hz,k,logicals in evaluated:
            if not _no_kernel4(_syndrome_columns(cached_hx,1496)):
                continue
            if not _no_kernel4(_syndrome_columns(cached_hz,1496)):
                continue
            A = [[[G.elements[g] for g in c] for c in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in c] for c in row]
                 for row in b["grid"]]
            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            assert sa == [[3,3,3]]
            assert [sum(row) for row in sb] == [7,7]
            assert [sum(row[j] for row in sb) for j in range(5)] == [2,3,3,3,3]
            assert sum(c == 0 for row in sb for c in row) == concentration
            assert _grid_weights(sa,sb) == 12 <= W_MAX
            assert _no_dead_lines(sa,sb)
            assert all(len(set(c)) == len(c) <= 3
                       for grid in (A,B) for row in grid for c in row)
            hx,hz,n = lifted_matrices(G,A,B)
            assert hx == cached_hx and hz == cached_hz and n == 1496
            assert k == n-gf2_rank(hx)-gf2_rank(hz)
            assert max(row.bit_count() for row in hx) <= 12
            assert max(row.bit_count() for row in hz) <= 10
            assert css_ok(hx,hz)
            return dict(
                idx=idx,cells={"A":A,"B":B},k=k,
                card=(concentration,policy,cohort),
                A_normal_ratio_orders=requested,
                component_readings=(a["d"],b["d"]),
                component_ranks=(a["rank"],b["rank"]),
                component_cycles=(a["cycles"],b["cycles"]),
                verified_sector_one_logical_readings=logicals,
                pairing_quality=key,target_cell=0.350)
    raise ValueError("dicyclic order/support card missed full rate/kernel gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the entire selected dicyclic cell without further sampling."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) <= 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Coupled full-coset quartics with persistent component challenges uniformly exclude sub-30 logical readings; current horizontal slots 30 and 33 both screened at 27 with k=295 and w=12.
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
