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
CONCEPT = ("Hypothesis: on metacyclic Z97:Z3(s=35), order 291, A=B=(2,1), coupled-gap quartics with verified zero component four-cycles and independently retested anchored min-sum/OSD challenges produce five NEW measured readings >=70 in cell 0.200. At k=295 that gives fitness >=993.4708. Failure refutes this experiment, not an algebraic distance bound.")
GROUP = ("Keep the exact Z97:Z3(s=35) definition and order 291 byte-for-byte unchanged. The tall bed reached 965.8564; its horizontal counterpart now leads cell 0.200 at 982.5. The current spectral-first gap program scored 937.9176 and admitted short readings 21 and 27. Retain the successful order-three action and large normal subgroup, but improve verified witness selection rather than infer distance from mixing estimates.")
SHAPES = ("Keep A=B=(2,1) byte-for-byte unchanged. Factor five gives n=1455, k>=291 and guaranteed rate 0.200; the size window permits q<=299 and the group fixes q=291. Raw components are tall, while their equal distance-bearing transpose components each have 582 variables and 291 checks. Current measured minima all occupied sector two: Z strips used one Arow across both Brows, and X strips used both Arows at one Brow. Preserve the authoritative rate gate and challenge these families in the ORIGINAL stabilizer quotient; parent k-kb=4 is not assumed for new labels.")
FILLING = ("Four distinct terms per cell with occupancy (2,1,1) retain both CSS caps twelve. Twenty unordered normal-gap-class pairs remain binding construction cards, with fresh support streams. Require bidirectional Sidon cells and disjoint directed differences, independently verifying zero Tanner four-cycles on scaffold-derived transpose components. Direct C6/C8 objectives replace finite-power spectral priorities. Forced-anchor damped min-sum searches, reliability-ordered exact syndrome completion and genuine circuit cancellation augment retained information-set witnesses. Independently deepen several finalists; never discard an earlier witness. Pair using exact full k, accumulated readings and commuting nontrivial strip logicals reduced by ORIGINAL stabilizers. Full ranks, kernel-four exclusion, weights and CSS remain gates. Finite searches and selection caps are not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_21x21_gap_zero4_anchored_minsum_osd_v5"

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


def NEW_CTX(rng, G, idx):
    """Twenty coupled-gap cards with zero-C4 and anchored kernel challenges."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (2, 1)
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1

    # Full-coset cells avoid using the occupancy profile itself as a
    # source of diversity. Diversity instead comes from relative gaps.
    profiles = (((2, 1, 1), (2, 1, 1)),)
    pa = pb = 0
    card, cohort = idx % 20, idx // 20
    gap_pairs = [
        (a, b) for a in range(6) for b in range(a, 6)
        if (a, b) != (5, 5)
    ]
    assert len(gap_pairs) == 20
    ga, gb = gap_pairs[card]
    # F97*/<-1,35> has sixteen classes. Classes 1..8 are representatives
    # modulo inversion; no global scaling produces a different card.
    gap_classes = (
        1 + (ga + 2 * cohort) % 8,
        1 + (gb + 2 * cohort) % 8,
    )
    primitive = next(
        a for a in range(2, 97)
        if pow(a, 48, 97) != 1 and pow(a, 32, 97) != 1)
    ratios = {
        "A": pow(primitive, gap_classes[0], 97),
        "B": pow(primitive, gap_classes[1], 97),
    }
    trade = card % 4
    seed = rng.randrange(1 << 62)
    # Fresh proposals, not archived fillings followed by relabeling.
    rng = random.Random(seed + 49979687 * (idx + 1) + 97000231)

    # Cache native group operations only. Every matrix below is built
    # by lifted_matrices, never by a separately implemented action.
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

    def directed_differences(cell):
        ids = [G.idx(g) for g in cell]
        return {table[inverses[a]][b]
                for a in ids for b in ids if a != b}

    signed_action = (1, 35, 61, 96, 62, 36)

    def fresh_cell(gap):
        """A doublet and two singleton cosets, with the prescribed gap class."""
        assert gap % 97
        for _ in range(512):
            double = rng.randrange(3)
            origin = rng.randrange(97)
            directed = gap * rng.choice(signed_action) % 97
            cell = [
                (origin, double),
                ((origin + directed) % 97, double),
            ]
            cell.extend(
                (rng.randrange(97), b)
                for b in range(3) if b != double)
            if whole_sidon(cell):
                return cell
        raise ValueError("full-coset Sidon doublet could not be filled")

    def fresh(profile, side):
        """Preserve the gap card while avoiding cross-cell C4 proposals."""
        assert profile == 0
        for _ in range(256):
            delta = rng.randrange(1, 97)
            first = fresh_cell(delta)
            second = fresh_cell(delta * ratios[side] % 97)
            if directed_differences(first).isdisjoint(
                    directed_differences(second)):
                return [[first], [second]]
        raise ValueError("disjoint-difference gap construction exhausted")

    def preserves_card(grid, side):
        gaps = []
        for row in grid:
            cell = row[0]
            if len(cell) != 4 or len(set(cell)) != 4:
                return False
            fibers = {}
            for a, b in cell:
                fibers.setdefault(b, []).append(a)
            if sorted(map(len, fibers.values())) != [1, 1, 2]:
                return False
            pair = next(v for v in fibers.values() if len(v) == 2)
            gaps.append((pair[1] - pair[0]) % 97)
        relative = gaps[1] * pow(gaps[0], -1, 97) % 97
        return relative in {
            ratios[side] * action % 97 for action in signed_action
        }

    def component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        unit = [[[identity]]]
        if side == "A":
            _, rows, _ = lifted_matrices(probe_group, indexed, unit)
        else:
            rows, _, _ = lifted_matrices(probe_group, unit, indexed)
        # Sector one of these unit probes occupies exactly q columns.
        # Sector two contains the distance-bearing transpose component.
        rows = [row >> q for row in rows]
        assert len(rows) == q
        assert all(row >> width == 0 for row in rows)
        return rows

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
        """Direct verified geometry, not a finite-power spectral estimate."""
        four, six, eight = stats
        if trade == 0:
            return (four, six, eight)
        if trade == 1:
            return (four, eight, six)
        if trade == 2:
            return (four, 12 * six + eight, six)
        return (four, 36 * six + eight, eight)

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in row[0]))
                     for row in grid)

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
        if not preserves_card(grid, side):
            raise ValueError("normal-gap card was erased")
        # Difference disjointness is a proposal rule; the ACTUAL
        # Tanner graph supplies the binding zero-four-cycle decision.
        if stats[0] != 0:
            return None
        if not directed_differences(grid[0][0]).isdisjoint(
                directed_differences(grid[1][0])):
            return None
        stats = stats + (eight_cycles(rows, columns),)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats),
            "signature": signature(grid), "side": side,
            "d": width + 1, "words": [],
            "sparse_d": width + 1, "anchored_attempts": 0,
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

    def sparse_challenge(record, diagnostic_seed, attempts):
        """Forced-anchor min-sum/OSD search on VERIFIED component columns.

        Fix one variable to one and complete its syndrome with the
        remaining variables. Every returned word is an actual nonzero
        kernel word. Nonconvergence and unsuccessful attempts never
        imply exclusion of any weight.
        """
        local = random.Random(diagnostic_seed)
        columns = record["columns"]
        if "bp_graph" not in record:
            edge_variables = []
            check_edges = [[] for _ in record["rows"]]
            for c, value in enumerate(columns):
                for r in _bits(value):
                    edge = len(edge_variables)
                    edge_variables.append(c)
                    check_edges[r].append(edge)
            record["bp_graph"] = (edge_variables, check_edges)
        edge_variables, check_edges = record["bp_graph"]
        retained = list(record["words"])
        found = []
        best = min(retained, key=int.bit_count) if retained else None

        def observe(word):
            nonlocal best
            if not word:
                return
            assert syndrome(columns, word) == 0
            if best is None or word.bit_count() < best.bit_count():
                best = word
            found.append(word)

        for attempt in range(attempts):
            block = attempt % 2
            root = block * q + identity
            root_bit = 1 << root
            channel = [2.0 + local.uniform(-0.65, 0.65)
                       for _ in columns]

            # Regular translations are diagnostic word permutations
            # only. They never generate a submitted code or its labels.
            warm = [word for word in retained
                    if (word >> (block * q)) & mask]
            if attempt % 3 == 1 and warm:
                source = local.choice(warm[:8])
                anchor = local.choice(list(_bits(
                    (source >> (block * q)) & mask)))
                multiplier = inverses[anchor]
                image = 0
                for c in _bits(source):
                    j, coordinate = divmod(c, q)
                    translated = (
                        table[coordinate][multiplier]
                        if record["side"] == "A"
                        else table[multiplier][coordinate])
                    image |= 1 << (j * q + translated)
                assert image & root_bit
                assert syndrome(columns, image) == 0
                for c in _bits(image ^ root_bit):
                    channel[c] = local.uniform(0.08, 0.75)
            elif attempt % 3 == 2:
                nearby = sorted({
                    edge_variables[e]
                    for r in _bits(columns[root])
                    for e in check_edges[r]
                    if edge_variables[e] != root
                })
                for c in local.sample(nearby, min(3, len(nearby))):
                    channel[c] = -local.uniform(0.15, 1.1)

            # The negative fixed message enforces the nonzero anchor.
            channel[root] = -48.0
            v_messages = [channel[c] for c in edge_variables]
            c_messages = [0.0] * len(edge_variables)
            posterior = list(channel)
            hard = root_bit

            for iteration in range(12 + 4 * (trade % 2)):
                for incident in check_edges:
                    parity = 1
                    first = second = 48.0
                    smallest = -1
                    for e in incident:
                        value = v_messages[e]
                        if value < 0:
                            parity = -parity
                        magnitude = abs(value)
                        if magnitude < first:
                            second, first = first, magnitude
                            smallest = e
                        elif magnitude < second:
                            second = magnitude
                    for e in incident:
                        sign = -1 if v_messages[e] < 0 else 1
                        magnitude = second if e == smallest else first
                        proposed = 0.78 * parity * sign * magnitude
                        c_messages[e] = (
                            0.55 * c_messages[e] + 0.45 * proposed)

                posterior = list(channel)
                for e, c in enumerate(edge_variables):
                    if c != root:
                        posterior[c] += c_messages[e]
                posterior[root] = -48.0
                hard = root_bit | sum(
                    1 << c for c in range(width)
                    if c != root and posterior[c] < 0)
                if syndrome(columns, hard) == 0:
                    observe(hard)
                    break
                for e, c in enumerate(edge_variables):
                    if c == root:
                        v_messages[e] = -48.0
                    else:
                        proposed = posterior[c] - c_messages[e]
                        v_messages[e] = max(-32.0, min(
                            32.0, 0.4 * v_messages[e] + 0.6 * proposed))

            # Reliability ordering proposes an information set. Exact
            # elimination, not BP convergence, decides syndrome completion.
            reliability = [
                abs(value) + local.uniform(0.0, 0.3)
                for value in posterior]
            order = [c for c in range(width) if c != root]
            order.sort(key=lambda c: reliability[c])
            pivots, circuits = {}, []
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
                    circuits.append(word)

            residual = syndrome(columns, hard)
            repair = 0
            while residual:
                old = pivots.get(residual.bit_length() - 1)
                if old is None:
                    break
                residual ^= old[0]
                repair ^= old[1]
            if residual:
                continue
            candidate = hard ^ repair
            assert candidate & root_bit
            assert syndrome(columns, candidate) == 0

            # Every move excludes the anchor, so zero cannot become a
            # spurious winner. Pair moves permit non-greedy cancellation.
            moves = circuits + [
                word for word in retained if not (word & root_bit)]
            for sweep in range(3):
                before = candidate.bit_count()
                local.shuffle(moves)
                for move in moves:
                    proposed = candidate ^ move
                    if proposed.bit_count() < candidate.bit_count():
                        candidate = proposed
                near = sorted(
                    moves, key=lambda w: (candidate ^ w).bit_count())[:24]
                winner = candidate
                for i, first in enumerate(near):
                    partial = candidate ^ first
                    for second in near[:i]:
                        proposed = partial ^ second
                        if proposed.bit_count() < winner.bit_count():
                            winner = proposed
                candidate = winner
                if candidate.bit_count() == before:
                    break
            assert candidate & root_bit
            observe(candidate)
            retained = sorted(
                set(retained + found), key=lambda w: (w.bit_count(), w))[:24]

        record["anchored_attempts"] += attempts
        if best is None:
            return width + 1, None
        record["words"] = sorted(
            set(record["words"] + found + [best]),
            key=lambda w: (w.bit_count(), w))[:24]
        assert all(syndrome(columns, word) == 0
                   for word in record["words"])
        record["sparse_d"] = min(
            record["sparse_d"], best.bit_count())
        return best.bit_count(), best

    def kernel_probe(record, diagnostic_seed, trials):
        """Retained, verified kernel witnesses challenge later information sets."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        retained = sorted(
            set(record["words"]), key=lambda w: (w.bit_count(), w))[:24]
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
            if weight <= min(100, reading + 6):
                additions.append(word)
                if len(additions) > 256:
                    additions[:] = sorted(
                        set(additions),
                        key=lambda w: (w.bit_count(), w))[:48]

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)

        for turn in range(trials):
            diagnostic.shuffle(order)
            if turn % 4 == 3 and retained:
                # Reinsert half of a known support early, rather than
                # testing only unrelated random systematic bases.
                anchor = diagnostic.choice(retained[:6])
                preferred = list(_bits(anchor))
                diagnostic.shuffle(preferred)
                preferred = set(
                    preferred[:max(1, len(preferred) // 2)])
                order.sort(key=lambda c: c not in preferred)

            pivots, basis = {}, []
            additions = list(retained)
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
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
                set(retained + additions),
                key=lambda w: (w.bit_count(), w))[:24]
            if reading <= 24:
                break

        if witness is None:
            raise ValueError("transpose component has no kernel witness")
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
            diagnostic_seed = seed + offset + 700001 * turn
            d, word = kernel_probe(record, diagnostic_seed, trials)
            record["d"] = min(record["d"], d)
            bank.add(word)
            # Independent soft-decoding landscapes complement random
            # information sets. Every reading remains witness-based.
            attempts = (8 if trials >= 96 else
                        4 if trials >= 48 else
                        1 if trials >= 12 else 0)
            if attempts:
                sd, sparse_word = sparse_challenge(
                    record, diagnostic_seed + 19000043, attempts)
                record["d"] = min(record["d"], sd)
                if sparse_word is not None:
                    bank.add(sparse_word)
            bank.update(record["words"][:4])
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
            # Accumulated witness readings precede geometry again.
            # The caps limit optimistic selection; they are not lower bounds.
            return (min(80, record["d"]), min(96, record["d"]),
                    tuple(-x for x in record["cycle_key"]),
                    record["rank"])

        for trial in range(96 + 8 * min(cohort, 2)):
            record = inspect(fresh(profile, side), side)
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
            if len(selected) >= 10:
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
            grid = [[list(row[0])] for row in current["grid"]]
            if step % 6 == 0:
                grid = fresh(profile, side)
            elif step % 3 == 0:
                # Simultaneously change both normal gaps while retaining
                # their relative class. This is not a common voltage gauge.
                delta = rng.randrange(1, 97)
                for block in range(2):
                    cell = grid[block][0]
                    fibers = {}
                    for t, g in enumerate(cell):
                        fibers.setdefault(g[1], []).append(t)
                    double = next(
                        positions for positions in fibers.values()
                        if len(positions) == 2)
                    b = cell[double[0]][1]
                    origin = rng.randrange(97)
                    gap = delta * (
                        1 if block == 0 else ratios[side]) % 97
                    gap = gap * rng.choice(signed_action) % 97
                    cell[double[0]] = (origin, b)
                    cell[double[1]] = ((origin + gap) % 97, b)
            else:
                cell = grid[j][0]
                fibers = {}
                for t, g in enumerate(cell):
                    fibers.setdefault(g[1], []).append(t)
                double = next(
                    positions for positions in fibers.values()
                    if len(positions) == 2)
                if rng.random() < 0.35:
                    # Move the doublet origin, not its gap class.
                    shift = rng.randrange(1, 97)
                    for t in double:
                        a, b = cell[t]
                        cell[t] = ((a + shift) % 97, b)
                else:
                    t = rng.choice([
                        t for t in range(4) if t not in double])
                    old = cell[t]
                    cell[t] = (
                        rng.choice([a for a in range(97)
                                    if a != old[0]]), old[1])
            if not all(whole_sidon(row[0]) for row in grid):
                continue
            if not preserves_card(grid, side):
                raise ValueError("repair failed to preserve its gap card")
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if proposed["cycles"][0] > floor4 + allowance:
                continue
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
            if len(finalists) >= 4:
                break
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:3]
        for record in finalists:
            # Independent confirmation preserves all earlier witnesses,
            # including anchored decoding and OSD completions.
            probe(record, 96 + 16 * min(cohort, 2), 211)
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

    def logical_retest(a, b, hx, hz, n):
        """Genuine tall-strip logicals tested against ORIGINAL stabilizers.

        Commutation and nontriviality are exact. The bounded reductions
        only produce upper readings; they do not exclude other families.
        """
        reading = n + 1
        weights = []
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            pivots = span(stabilizers)
            local = random.Random(
                seed + (6100009 if kind == "X" else 7100011))
            words = sorted(
                set(record["words"]),
                key=lambda w: (w.bit_count(), w))[:4]
            candidates = set()
            for word in words:
                assert syndrome(record["columns"], word) == 0
                for strip in range(2):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i * q)) & mask)
                            << ((1 + 2 * i + strip) * q)
                            for i in range(2))
                    else:
                        actual = word << ((1 + 2 * strip) * q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            for initial in sorted(
                    candidates, key=lambda w: (w.bit_count(), w))[:6]:
                actual = initial
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    near = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best = actual
                    for i, first in enumerate(near):
                        for second in near[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < best.bit_count():
                                best = proposed
                    if best != actual:
                        actual, changed = best, True
                    if not changed:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
                weights.append((kind, actual.bit_count()))
        return reading, weights

    for restart in range(2):
        aa = optimize(pa, "A")
        bb = optimize(pb, "B")
        survivors = []
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 5 * q == 1455
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k >= q and 7 * n <= 40 * k < 9 * n):
                    continue

                # Verify unit-probe extraction against this actual pairing.
                # Sector two is ordered (Arow, Brow), Brow varying first.
                actual_a = [
                    sum(((hz[t] >> ((1 + 2 * i) * q)) & mask)
                        << (i * q) for i in range(2))
                    for t in range(q)
                ]
                actual_b = [
                    (hx[t] >> q) & ((1 << width) - 1)
                    for t in range(q)
                ]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logical_d, logical_weights = logical_retest(
                    a, b, hx, hz, n)
                detected = min(a["d"], b["d"], logical_d)
                capped = min(76, detected)
                key = (
                    k * capped * capped,
                    min(96, detected),
                    min(96, a["d"]) + min(96, b["d"]),
                    tuple(-(x + y) for x, y in
                          zip(a["cycle_key"], b["cycle_key"])),
                    -(k - q),
                )
                survivors.append((
                    key, a, b, hx, hz, n, k,
                    logical_d, logical_weights))
        survivors.sort(key=lambda item: item[0], reverse=True)

        # Expensive exhaustive kernel-four gates apply to the selected
        # pair first, rather than being repeated on every comparison.
        for key, a, b, hx, hz, n, k, logical_d, weights in survivors:
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            assert max(row.bit_count() for row in hx + hz) == 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            assert preserves_card(a["grid"], "A")
            assert preserves_card(b["grid"], "B")
            return {
                "A": a["grid"], "B": b["grid"], "idx": idx, "k": k,
                "card": (gap_classes, trade, cohort),
                "component_readings": (a["d"], b["d"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "anchored_kernel_readings": (a["sparse_d"], b["sparse_d"]),
                "anchored_attempts": (
                    a["anchored_attempts"], b["anchored_attempts"]),
                "verified_component_zero_four": (
                    a["cycles"][0] == 0, b["cycles"][0] == 0),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
                "verified_localized_logical_reading": logical_d,
                "localized_logical_weights": weights,
                "pairing_merit": key,
                "coset_occupancy": (2, 1, 1),
            }
    raise ValueError("no anchored-challenged zero-C4 gap pair passed full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete zero-C4, anchored-challenged coupled-gap cell."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 4
    assert sorted(sum(g[1] == b for g in cell) for b in range(3)) == [1, 1, 2]
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports plus transpose-kernel repair uniformly exclude sub-50 logical readings: current slots 21,43,54,58 screened at 30,39,45,36 despite four terms per cell and w=12.
# REFUTED: Adding verified C3-fixed screening improves the 965.8564 tall-bed record: shown Program 1 scored 932.7676 with selected readings 65..71.
# REFUTED: Bidirectional difference-disjoint supports with fixed/even C3-module challenges improve the tall-bed record: shown Program D2 scored 869.0234 with selected readings 61,63,69,66,68.
# REFUTED: Full-coset coupled-gap cards with spectral-first expansion selection improve the 965.8564 tall-bed record: the current program scored 937.9176 with measured readings 66..70 and also admitted screened readings 21 and 27.
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
