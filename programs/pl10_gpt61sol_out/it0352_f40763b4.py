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
CONCEPT = ("For cyclic C27, order 27, shapes (2,4)x(2,4), reinforcing exactly one B column, requiring exact k=108 and selecting by six-round temporal BP ambiguity should improve mean worse-basis circuit fitness above 7.2133. Actual ASC memory measurements must test this principle.")
GROUP = ("Frozen byte-for-byte under L1: retain the C27 bed whose current fitness is 7.2133. Related C29 fillings reached 7.1850, while selective reinforcement on C29 reached 7.1490; neither establishes a group advantage or guarantees this filling succeeds. Odd-order C27 has no involutions; exclude order-three internal differences and use verified ranks rather than character predictions.")
SHAPES = ("Frozen equal (2,4) components have 108 variables and 54 checks each. The size factor twenty admits q<=29, but frozen q=27 gives n=540, kb=108 and guaranteed rate 0.200. Require complete verified k=108. Parent failure maps implicated several Acol/Bcol and Acol/Brow blocks, including Bcol2 and Zchk:Acol0/Brow0, with only zero to five failures per basis. Collective flips, including slot18's worst logical #7 in four failed Z shots, motivate balanced decoder screening rather than a claimed isolated hotspot.")
FILLING = ("Twenty contracts cross five twelve/thirteen-term support pairs with four prescribed order-nine/order-27 and cross-row-overlap briefs; extra harness batches extend the relative book. A has row totals six/six and column degrees three. B has row totals six/seven and exactly one degree-four column, giving maximum CSS weight ten and 24300 six-round CNOTs versus 23328. Full component ranks and exact complete k=108 prevent automatic rank-deficit promotion: parent k=110 slots16/18 had four/five Z failures, whereas regular slot3 had one failure per basis, all at d2s=12 and 200000 shots. Verified component screens exclude short kernels, check minor witnesses, and compare boundaries and short-word orbits without rewarding sampled bounds above twelve. Up to six complete structurally eligible codes receive common-stream six-round temporal BP probes; fresh holdout streams compare the two coarse winners. Degree-weighted data noise, measurement noise and approximate opposite-check pairs test collective ambiguity, but omit actual ASC ordering and relay/LSD and are not measured pL. Relative canonicalization excludes obvious orbit copies, not arbitrary Tanner isomorphisms.")
FAMILY_NAME = "r020_C27_24x24_selective13_temporalbp_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 27

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This cyclic family requires q=27")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _design_supports(idx):
    """Four degree-three support architectures, each with five order briefs."""
    profiles = ((1, 1, 2, 2), (0, 1, 2, 3), (0, 2, 2, 2))
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    a, b = pairs[(idx // 5) % len(pairs)]

    def support(p):
        first = profiles[p]
        return [list(first), [3 - w for w in first]]

    return support(a), support(b)


def _lowload_geometry(G, idx, sa, sb):
    """Indexed relative-support book, independent of offset-search RNG.

    Canonicalization excludes obvious affine and protograph orbit copies,
    not all possible isomorphisms of the final quantum Tanner graphs.
    """
    from math import gcd

    q = G.q
    architecture = (idx // 5) % 4
    wanted = 5 * (idx // 20) + idx % 5
    cards = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
    units = [d for d in range(1, (q + 1) // 2) if gcd(d, q) == 1]
    order_nine = [d for d in range(1, (q + 1) // 2)
                  if gcd(d, q) == 3]
    multipliers = [u for u in range(1, q) if gcd(u, q) == 1]
    grng = random.Random(270041 + 1009 * architecture)

    def normalized(cell, u):
        if not cell:
            return ()
        values = [u * a % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def signature(grids):
        images = []
        for u in multipliers:
            sides = []
            for grid in grids:
                columns = [(normalized(grid[0][j], u),
                            normalized(grid[1][j], u)) for j in range(4)]
                normal = tuple(sorted(columns))
                swapped = tuple(sorted((b, a) for a, b in columns))
                sides.append(min(normal, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    book, seen = [], set()
    for attempt in range(30000):
        card = len(book) % 5
        grids = []
        legal = True
        for side, counts in enumerate((sa, sb)):
            sites = [(i, j) for i in range(2) for j in range(4)
                     if counts[i][j] == 2]
            charged = set(grng.sample(sites, cards[card][side]))
            grid = []
            for i, row_counts in enumerate(counts):
                row = []
                used_differences = set()
                for j, weight in enumerate(row_counts):
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [0]
                    elif weight == 2:
                        palette = order_nine if (i, j) in charged else units
                        cell = [0, grng.choice(palette)]
                    else:
                        cell = [0] + grng.sample(range(1, q), weight - 1)

                    for a in cell:
                        for b in cell:
                            if a == b:
                                continue
                            difference = (a - b) % q
                            # Exclude order-three internal differences
                            # and repeated differences within a row.
                            if gcd(difference, q) == 9:
                                legal = False
                            if difference in used_differences:
                                legal = False
                            used_differences.add(difference)
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        if not legal:
            continue
        key = signature(grids)
        if key in seen:
            continue
        seen.add(key)
        book.append(grids)
        if len(book) > wanted:
            return book[wanted]
    raise ValueError("relative-support book exhausted before indexed design")


def _shape_component_rows(G, side, grid):
    """Use zero probes only to extract the scaffold's exact component."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _shape_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _shape_short_counts(cols):
    """Kernel collision statistics through weight four.

    Counts may overcount partitions. A zero tuple certifies absence
    of component kernel words of weights one through four.
    """
    singles = {}
    one = two = three = four = 0
    for value in cols:
        one += int(value == 0)
        two += singles.get(value, 0)
        singles[value] = singles.get(value, 0) + 1
    pairs = {}
    for i, value in enumerate(cols):
        for other in cols[:i]:
            syndrome = value ^ other
            three += singles.get(syndrome, 0)
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return one, two, three, four


def _shape_higher(cols):
    """Triple boundaries and weight-five/six collision exposure.

    These are circuit-performance surrogates, not measured pL.
    When shorter kernel words are absent, collisions correspond to
    partitions of weight-five or weight-six component words.
    """
    pairs = set()
    for i, value in enumerate(cols):
        for other in cols[:i]:
            pairs.add(value ^ other)
    triples = {}
    boundary = five = six = 0
    for i in range(len(cols)):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                weight = syndrome.bit_count()
                if weight < 5:
                    boundary += (5 - weight) ** 2
                five += int(syndrome in pairs)
                six += triples.get(syndrome, 0)
                triples[syndrome] = triples.get(syndrome, 0) + 1
    return boundary, five, six


def _shape_kernel_estimate(rng, cols, trials, q):
    """Sample actual kernel witnesses and their block concentration.

    The minimum is an upper bound. Witness counts are sampled exposure
    statistics, not an exact spectrum or a circuit-failure estimate.
    """
    order = list(range(len(cols)))
    best = len(cols) + 1
    short = set()

    def offer(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        best = min(best, weight)
        if weight <= 12:
            short.add(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, word)
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome and word:
                basis.append(word)
                offer(word)

        # Elementary basis operations preserve verified kernel membership.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for j in range(i):
                    proposal = word ^ basis[j]
                    offer(proposal)
                    pw = proposal.bit_count()
                    if proposal and pw < weight:
                        word, weight = proposal, pw
                basis[i] = word
        if best <= 6:
            break

    vulnerable = [word for word in short if word.bit_count() <= 10]
    mask = (1 << q) - 1
    loads = [sum(bool((word >> (j * q)) & mask) for word in vulnerable)
             for j in range(len(cols) // q)]
    return best, len(vulnerable), max(loads, default=0)


def _lowload_cycles(rows, cols):
    """Exact Tanner four/six-cycle counts from verified component rows."""
    from itertools import combinations

    adj = [dict() for _ in rows]
    neighborhoods = []
    for syndrome in cols:
        ns = []
        while syndrome:
            bit = syndrome & -syndrome
            ns.append(bit.bit_length() - 1)
            syndrome ^= bit
        neighborhoods.append(ns)
        for a, b in combinations(ns, 2):
            adj[a][b] = adj[a].get(b, 0) + 1
            adj[b][a] = adj[b].get(a, 0) + 1

    four = sum(v * (v - 1) // 2
               for a, neighbors in enumerate(adj)
               for b, v in neighbors.items() if a < b)
    six = 0
    for a, neighbors in enumerate(adj):
        for b, ab in neighbors.items():
            if b <= a:
                continue
            for c, bc in adj[b].items():
                if c > b and c in neighbors:
                    six += ab * bc * neighbors[c]
    # Correct triangles that reuse a variable for two or three edges.
    for ns in neighborhoods:
        for a, b, c in combinations(ns, 3):
            six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
    return four, six


def _shape_component(rng, G, side, supports, geometry, mode):
    """Optimize offsets without changing the slot's relative-support brief."""
    n = len(supports[0]) * G.q

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw():
        result = []
        for row in geometry:
            out = []
            for relative in row:
                origin = rng.randrange(G.q)
                out.append([((origin + a) % G.q,) for a in relative])
            result.append(out)
        return result

    def assess(grid):
        rows = _shape_component_rows(G, side, grid)
        deficit = len(rows) - gf2_rank(rows)
        if deficit > 2 or not _connected_checks(rows, [], n):
            return None
        cols = _shape_columns(rows, n)
        assert all(col.bit_count() == 3 for col in cols)
        short = _shape_short_counts(cols)
        cycles = _lowload_cycles(rows, cols)
        return (short, deficit, cycles), cols

    pool = []
    for trial in range(128):
        grid = draw()
        result = assess(grid)
        if result is None:
            continue
        base, cols = result
        pool.append((base, trial, grid, cols))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[6:]

    sites = [(i, j) for i in range(2) for j in range(4)
             if supports[i][j]]
    finalists = []
    for base, trial, grid, cols in pool:
        # Cell translations alter inter-cell holonomy while preserving
        # every indexed internal difference and every term count.
        for step in range(72):
            proposal = clone(grid)
            i, j = rng.choice(sites)
            shift = rng.randrange(1, G.q)
            proposal[i][j] = [((g[0] + shift) % G.q,)
                              for g in proposal[i][j]]
            result = assess(proposal)
            if result is not None and result[0] <= base:
                base, cols = result
                grid = proposal

        boundary, five, six = _shape_higher(cols)
        estimate, short_load, peak = _shape_kernel_estimate(
            rng, cols, 48, G.q)
        # Above a coarse sampled protection floor, prioritize collective
        # ambiguity and boundaries; do not reward larger upper bounds.
        exposure = (peak, short_load) if mode % 2 else (short_load, peak)
        key = (base[0], int(estimate < 12), five, six,
               exposure, boundary, base[2], base[1], trial)
        finalists.append((key, grid))

    if not finalists:
        raise ValueError("No connected degree-three component passed screening")
    finalists.sort(key=lambda item: item[0])
    return finalists[0][1], finalists[0][0]


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def NEW_CTX(rng, G, idx):
    """Fresh selective-column contracts and verified temporal-BP selection."""
    from itertools import combinations
    from math import gcd

    q = G.q
    assert q == 27 and A_SHAPE == B_SHAPE == (2, 4)
    width, mask = 4 * q, (1 << q) - 1

    # Profiles 0/1 have twelve terms. Profiles 2/3/4 reinforce one
    # column, respectively with splits 2+2, 3+1 and 1+3.
    profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (3, 2, 1, 0)),
        ((1, 1, 2, 2), (2, 2, 1, 2)),
        ((1, 1, 1, 3), (2, 2, 2, 1)),
        ((1, 1, 2, 2), (3, 2, 1, 1)),
    )
    pairs = ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3))
    architecture = idx % 5
    card, batch = (idx // 5) % 4, idx // 20
    pa, pb = pairs[architecture]
    sa, sb = ([list(row) for row in profiles[p]]
              for p in (pa, pb))
    counts = (sa, sb)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 10 and 10 <= W_MAX
    assert list(map(sum, sa)) == [6, 6]
    assert sorted(map(sum, sb)) == [6, 7]
    assert sorted(map(sum, zip(*sa))) == [3, 3, 3, 3]
    assert sorted(map(sum, zip(*sb))) == [3, 3, 3, 4]
    for support in counts:
        assert gf2_rank([
            sum((t & 1) << j for j, t in enumerate(row))
            for row in support
        ]) == 2
        assert all(0 <= t <= CELL_MAX
                   for row in support for t in row)

    units = [u for u in range(1, q) if gcd(u, q) == 1]
    long_dirs = [u for u in range(1, (q + 1) // 2)
                 if gcd(u, q) == 1]
    nine_dirs = [u for u in range(1, (q + 1) // 2)
                 if gcd(u, q) == 3]
    overlap_cards = ((0, 0), (0, 1), (1, 1), (1, 2))
    charged_cards = ((0, 1), (1, 1), (1, 2), (2, 2))
    markers = (3, 4, 5, 8)

    def normalized(cell, u):
        if not cell:
            return ()
        values = [u * a % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def relative_key(grids):
        # Exclude common group automorphisms, cell-origin gauges,
        # component row/column permutations and component exchange.
        # The evaluator still performs arbitrary Tanner deduplication.
        images = []
        for u in units:
            sides = []
            for grid in grids:
                columns = [(normalized(grid[0][j], u),
                            normalized(grid[1][j], u))
                           for j in range(4)]
                sides.append(min(
                    tuple(sorted(columns)),
                    tuple(sorted((b, a) for a, b in columns))))
            images.append(tuple(sorted(sides)))
        return min(images)

    grng = random.Random(271307 + 1613 * architecture)
    wanted = 4 * batch + card
    book, seen = [], set()
    for proposal in range(120000):
        brief = len(book) % 4
        book_batch = len(book) // 4
        grids, legal = [], True
        for side, support in enumerate(counts):
            sites = [(i, j) for i in range(2) for j in range(4)
                     if support[i][j] == 2]
            charged = set(grng.sample(
                sites, charged_cards[brief][side]))
            grid, row_sets = [], []
            for i, row_counts in enumerate(support):
                row, used, directions = [], set(), set()
                for j, weight in enumerate(row_counts):
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [0]
                    elif weight == 2:
                        palette = (nine_dirs if (i, j) in charged
                                   else long_dirs)
                        cell = [0, grng.choice(palette)]
                    else:
                        step = grng.choice(units)
                        marker = markers[
                            (brief + side + i + book_batch) % 4]
                        cell = [0, step, step * marker % q]
                    assert len(set(cell)) == weight
                    for a in cell:
                        for b in cell:
                            if a == b:
                                continue
                            difference = (a - b) % q
                            if gcd(difference, q) == 9 or difference in used:
                                legal = False
                            used.add(difference)
                            directions.add(min(difference, q - difference))
                    row.append(cell)
                grid.append(row)
                row_sets.append(directions)
            if len(row_sets[0] & row_sets[1]) != overlap_cards[brief][side]:
                legal = False
            grids.append(grid)
        if not legal:
            continue
        signature = relative_key(grids)
        if signature in seen:
            continue
        seen.add(signature)
        book.append(grids)
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("selective C27 relative-support book exhausted")
    geometry = book[wanted]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def filling_key(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def kernel_exposure(cols, side):
        """Verified minor and sampled witnesses; upper bounds only.

        Count translation orbits through fourteen, but give no
        additional protection preference above twelve.
        """
        best, short = width + 1, set()

        def offer(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight <= 14:
                short.add(word)

        coefficients = [
            [(cols[j * q] >> (i * q)) & mask for j in range(4)]
            for i in range(2)
        ]

        def convolution(a, b):
            result = 0
            while a:
                bit = a & -a
                shift = bit.bit_length() - 1
                result ^= ((b << shift) | (b >> (q - shift))) & mask
                a ^= bit
            return result

        generators = set()
        for a, b, c in combinations(range(4), 3):
            parts = [0] * 4
            for destination, (u, v) in zip(
                    (a, b, c), ((b, c), (a, c), (a, b))):
                parts[destination] = (
                    convolution(coefficients[0][u], coefficients[1][v])
                    ^ convolution(coefficients[0][v], coefficients[1][u]))
            for shift in range(q):
                word = sum(
                    (((part << shift) | (part >> (q - shift))) & mask)
                    << (j * q) for j, part in enumerate(parts))
                if not word:
                    continue
                syndrome, occupied = 0, word
                while occupied:
                    bit = occupied & -occupied
                    syndrome ^= cols[bit.bit_length() - 1]
                    occupied ^= bit
                assert syndrome == 0, "minor failed verified kernel check"
                generators.add(word)
        ordered = sorted(generators)
        for i, word in enumerate(ordered):
            offer(word)
            for other in ordered[:i]:
                offer(word ^ other)

        probe = random.Random(271909 + 1009 * idx + side)
        order = list(range(width))
        for trial in range(32):
            probe.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome and word:
                    basis.append(word)
                    offer(word)
            for sweep in range(2):
                probe.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        proposal = word ^ other
                        offer(proposal)
                        if proposal and proposal.bit_count() < weight:
                            word, weight = proposal, proposal.bit_count()
                    basis[i] = word
            if best <= 6:
                break

        orbits = set()
        for word in short:
            parts = [(word >> (j * q)) & mask for j in range(4)]
            representative = min(
                sum(
                    (((part << shift) | (part >> (q - shift))) & mask)
                    << (j * q) for j, part in enumerate(parts))
                for shift in range(q))
            orbits.add(representative)
        loads = [
            sum(((word >> (j * q)) & mask).bit_count()
                for word in orbits)
            for j in range(4)]
        return best, len(orbits), max(loads, default=0)

    def component_portfolio(side):
        """Optimize origins within the indexed, immutable relative brief."""
        support = counts[side]
        label = "A" if side == 0 else "B"
        expected_degrees = [sum(row[j] for row in support)
                            for j in range(4)]

        def draw():
            result = []
            for row in geometry[side]:
                output = []
                for relative in row:
                    origin = rng.randrange(q)
                    output.append([((origin + a) % q,) for a in relative])
                result.append(output)
            return result

        def assess(grid):
            rows = _shape_component_rows(G, label, grid)
            if gf2_rank(rows) != 2 * q:
                return None
            if not _connected_checks(rows, [], width):
                return None
            cols = _shape_columns(rows, width)
            assert all(c.bit_count() == expected_degrees[j // q]
                       for j, c in enumerate(cols))
            short = _shape_short_counts(cols)
            cycles = _lowload_cycles(rows, cols)
            return (short, cycles), cols

        pool, tested = [], set()
        for trial in range(160):
            grid = draw()
            signature = filling_key(grid)
            if signature in tested:
                continue
            tested.add(signature)
            result = assess(grid)
            if result is None:
                continue
            cheap, cols = result
            pool.append((cheap, trial, grid, cols))
            pool.sort(key=lambda entry: entry[:2])
            del pool[6:]

        sites = [(i, j) for i in range(2) for j in range(4)
                 if support[i][j]]
        refined, retained = [], set()
        for cheap, trial, grid, cols in pool:
            for step in range(72):
                proposal = clone(grid)
                i, j = rng.choice(sites)
                shift = rng.randrange(1, q)
                proposal[i][j] = [((g[0] + shift) % q,)
                                  for g in proposal[i][j]]
                result = assess(proposal)
                if result is not None and result[0] <= cheap:
                    cheap, cols = result
                    grid = proposal
            signature = filling_key(grid)
            if signature in retained or any(cheap[0]):
                continue
            retained.add(signature)
            boundary, five, six = _shape_higher(cols)
            distance, orbits, peak = kernel_exposure(cols, side)
            four, cycles = cheap[1]
            key = (int(distance < 12), int(bool(five or six)), four,
                   peak, orbits, boundary, five, six, cycles, trial)
            refined.append({
                "key": key, "grid": grid, "distance": distance,
                "five": five, "six": six, "four": four,
                "orbits": orbits, "peak": peak,
                "boundary": boundary, "cycles": cycles,
            })
        refined.sort(key=lambda entry: entry["key"])
        return refined

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round phenomenological surrogate, NOT ASC circuit fitness.

        Decode data increments and measurement errors jointly, ending
        with a perfect syndrome. Opposite-check pairs approximate
        correlations without claiming any particular ASC hook.
        """
        import numpy as np

        def neighborhoods(rows):
            result = []
            for row in rows:
                ns = []
                while row:
                    bit = row & -row
                    ns.append(bit.bit_length() - 1)
                    row ^= bit
                result.append(ns)
            return result

        ns = neighborhoods(checks)
        opposite = neighborhoods(stabilizers)
        m, rounds = len(ns), 6
        spatial_width = max(map(len, ns))
        row_width = spatial_width + 2
        variables = rounds * (n + m)
        dummy = variables
        detectors = (rounds + 1) * m
        incidence = np.full(
            (detectors, row_width), dummy, dtype=np.int32)
        for t in range(rounds + 1):
            for r, row in enumerate(ns):
                position = 0
                if t < rounds:
                    for c in row:
                        incidence[t * m + r, position] = t * n + c
                        position += 1
                    incidence[t * m + r, position] = rounds * n + t * m + r
                    position += 1
                if t:
                    incidence[t * m + r, position] = (
                        rounds * n + (t - 1) * m + r)

        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        degrees = np.bincount(flat[valid], minlength=variables)
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (variables + 1, int(degrees.max())),
            edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        spatial_degree = np.asarray([
            c.bit_count() for c in _shape_columns(checks, n)])
        other_degree = np.asarray([
            c.bit_count() for c in _shape_columns(stabilizers, n)])
        data_rate = 0.002 * (2 + spatial_degree + other_degree)
        measure_rate = np.asarray([
            0.002 * (2 + len(row)) * 2 / 3 for row in ns])
        base = np.concatenate((
            np.tile(data_rate, rounds),
            np.tile(measure_rate, rounds)))
        channel = np.random.default_rng(seed)
        levels = np.asarray((1.0, 1.4, 1.8))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None] * base[None, :], 0.25)
        errors = np.zeros((shots, variables + 1), dtype=bool)
        errors[:, :variables] = channel.random(
            (shots, variables)) < rates
        for t in range(rounds):
            active = channel.random(shots) < 0.30
            choices = channel.integers(len(opposite), size=shots)
            for shot in np.flatnonzero(active):
                row = opposite[int(choices[shot])]
                endpoints = channel.choice(len(row), 2, replace=False)
                for endpoint in endpoints:
                    errors[shot, t * n + row[int(endpoint)]] ^= True

        target = (
            errors[:, flat].reshape(shots, detectors, row_width)
            .sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        true_final = np.logical_xor.reduce(
            errors[:, :rounds * n].reshape(shots, rounds, n), axis=1)
        prior = np.empty((shots, variables + 1), dtype=np.float64)
        prior[:, :variables] = np.log((1 - rates) / rates)
        prior[:, dummy] = 30.0
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        span = _row_span(stabilizers)
        failures = iteration_sum = 0

        for iteration in range(1, 25):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped) / 2), 1e-12, 1 - 1e-8))
            products = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True) - logs),
                1 - 1e-8)
            outgoing = (
                (np.log1p(products) - np.log1p(-products))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges + 1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, detectors, row_width)
                .sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds * n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(
                        true_final[shot] ^ decoded_final[shot]):
                    word |= 1 << int(c)
                failures += int(not _in_row_span(word, span))
                iteration_sum += iteration

            keep = ~settled
            if not np.any(keep):
                return failures, 0, iteration_sum
            messages = np.clip(
                0.30 * messages
                + 0.70 * (posterior[:, flat] - outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        return (failures + unresolved, unresolved,
                iteration_sum + 25 * unresolved)

    def probe_quality(x, z):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1], x[2] + z[2])

    for attempt in range(3):
        aa, bb = component_portfolio(0), component_portfolio(1)
        if not aa or not bb:
            continue
        options = []
        for a in aa:
            for b in bb:
                key = (
                    int(min(a["distance"], b["distance"]) < 12),
                    int(bool(a["five"] or a["six"]
                             or b["five"] or b["six"])),
                    a["four"] + b["four"],
                    max(a["peak"], b["peak"]),
                    a["orbits"] + b["orbits"],
                    max(a["boundary"], b["boundary"]),
                    a["boundary"] + b["boundary"],
                    a["cycles"] + b["cycles"],
                    a["key"][-1], b["key"][-1])
                options.append((key, a, b))
        options.sort(key=lambda entry: entry[0])

        verified, seen_fillings = [], set()
        for key, a, b in options:
            if verified and key[:3] != verified[0][0][:3]:
                break
            A, B = a["grid"], b["grid"]
            signature = (filling_key(A), filling_key(B))
            if signature in seen_fillings:
                continue
            seen_fillings.add(signature)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 540 or k != 108:
                continue
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 10
            assert 6 * sum(row.bit_count() for row in hx + hz) == 24300
            verified.append((key, A, B, hx, hz, a, b))
            if len(verified) == 6:
                break
        if not verified:
            continue

        # Structural guards precede the noisy decoder surrogate.
        # Common streams compare candidates, not independent noise luck.
        seed = 27217003 + 1009 * idx
        coarse = []
        for number, option in enumerate(verified):
            key, A, B, hx, hz, a, b = option
            x = temporal_probe(hz, hx, 540, seed, 12)
            z = temporal_probe(hx, hz, 540, seed + 1, 12)
            quality = probe_quality(x, z) + key + (number,)
            coarse.append((quality, number, option, x, z))
        coarse.sort(key=lambda entry: entry[:2])

        finalists = []
        for _, number, option, cx, cz in coarse[:2]:
            key, A, B, hx, hz, a, b = option
            # Fresh common holdout streams reduce coarse selection noise.
            # They still do not simulate the scored ASC/relay/LSD circuit.
            x = temporal_probe(hz, hx, 540, seed + 1000003, 24)
            z = temporal_probe(hx, hz, 540, seed + 1000004, 24)
            total_x = tuple(u + v for u, v in zip(cx, x))
            total_z = tuple(u + v for u, v in zip(cz, z))
            quality = probe_quality(total_x, total_z) + key + (number,)
            finalists.append((quality, A, B, a, b, total_x, total_z))
        finalists.sort(key=lambda entry: entry[0])
        quality, A, B, a, b, x, z = finalists[0]
        return {
            "cells": {"A": A, "B": B}, "k": 108,
            "contract": (architecture, card, batch),
            "relative_geometry": geometry,
            "component_screen": (a["key"], b["key"]),
            "temporal_bp_surrogate": (x, z),
            "screen": quality, "cnots": 24300,
        }

    raise ValueError("No selective C27 filling passed exact-k=108 screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened 2x4 cell without late randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: Even-quotient doubletons at weight ten uniformly retain zero-failure performance: current slot 25 had four failed Z shots out of 200000, flipping 47 logical qubits, despite d2s=15 and k=66.
# REFUTED: Raising the screened distance within the degree-three/four parent necessarily improves circuit reliability: slot17 had d2s=14,s=6.509 and 29 X failures, while slot28 had d2s=13,s=6.969 and seven failures in each basis, all at 200000 shots.
# REFUTED: Larger k at equal d2s guarantees better C27 memory: slot18 had k=110,d2s=12,s=7.054 and five Z failures/200000, while slot3 had k=108,d2s=12,s=7.367 and one failure per basis; this does not establish rank deficit as the cause.
# REFUTED: Canonical internal relative-support signatures guarantee Tanner-distinct submissions: the measured C27 parent retained only 24 distinct codes from 60 slots, including slot2 marked iso_dup_of_1.
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
