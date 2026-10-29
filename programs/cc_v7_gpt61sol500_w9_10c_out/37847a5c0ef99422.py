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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Q8, also Dic2, order eight, A=(2,6), B=(2,7): central-involution and row-correlated fillings selected in the full CSS logical quotient should supply five novel finalists with k>=172 or distance>=5, improving fitness above 7.4350 in cell 0.450.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This Q8 bed measured five k=171,d30s=4 leaders and fitness 7.4350. The unique central involution supplies a previously excluded mechanism for rank deficits and short stabilizers. Neither its order nor a constituent rank deficit establishes quantum distance; verified full matrices decide.")
SHAPES = ("Keep A=(2,6), B=(2,7): n=368, kb=160 and guaranteed rate 10/23=0.43478. The shape permits q<=13, while the frozen group fixes q=8. Unequal constituents have 48 variables/16 checks and 56 variables/16 checks. Current minimum logicals broadly occupy sector-1 Acol0..5/Bcol0..6, with prominent X participation at Acol0..1 and Z participation at Bcol2..4. Actual ranks retain [0.425,0.475).")
FILLING = ("Twenty proposal schedules explore additional internal-edge allocations, central-involution binomials, centrally correlated rows and degree-three A columns paired with six-term B rows. Both CSS weight sums are checked before lifting. Constituent short words are diagnostics, not automatic rejection: exhaustive full-matrix syndrome collisions through weight four are reduced modulo stabilizers. Every output excludes logicals below four; exclusion through four receives priority. Globally rank the immutable portfolio by k times the squared certified logical floor, then weight-four collision burden. Rebuilt parent fillings are exclusion references only, and strengthened intrinsic Tanner invariants reject repeats conservatively. Five high-rank or distance-five finalists and novelty against the unseen archive remain falsifiable hypotheses.")
FAMILY_NAME = "r045_dic2_q8_26x27_asymmetric_incidence_portfolio_v3_logical_socle"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic2 requires order eight")
    # Same dicyclic rule: x^(2t)=1, y^2=x^t, y*x*y^-1=x^-1.
    # Factor 46 fits the retained q=8; t=2 gives Dic2, also called Q8.
    t, m = 2, 4
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + t * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else -1) * c + t * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + t) % m, 1)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


_A_INTERNAL = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (1, 2),
)


def _incidence_filling(rng, G, card, epoch):
    """Twenty optimized support/relative-voltage policies, not orbit draws."""
    amode, bmode = divmod(card % 20, 4)
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    })
    assert G.q == 8 and len(directions) == 3

    def grade(g, mode):
        a, b = g
        return (a if mode == 0 else b if mode == 1 else a + b) % 2

    def incidence(width, internal, mode, graded):
        C = [[], []]
        gaps = ([g for g in directions if grade(g, mode)]
                if graded else directions)
        if any(p > len(gaps) for p in internal):
            return None
        for i, count in enumerate(internal):
            for gap in rng.sample(gaps, count):
                C[i].append([G.e, gap])
                C[1 - i].append([])

        cross = width - sum(internal)
        parity = rng.randrange(2)
        labels = ([g for g in G.elements if grade(g, mode) == parity]
                  if graded and sum(internal) else list(G.elements))
        if cross < 0 or cross > len(labels):
            return None
        for voltage in rng.sample(labels, cross):
            C[0].append([G.e])
            C[1].append([voltage])
        return C

    # The last two policies share five/seven row totals, but one
    # constructs a grading and the other tests unrestricted couplings.
    # Their final Tanner invariants, not their policy names, separate codes.
    agrad = (epoch + amode) % 3
    graded_a = amode != 4 and epoch % 5 != 4
    A = incidence(6, _A_INTERNAL[amode], agrad, graded_a)
    if A is None:
        return None

    # Seven distinct cross matchings and two internal pairs are the
    # two regular B supports. A grading for B is chosen independently
    # of A; relative grading alignment is a genuine coupling parameter.
    if bmode == 0:
        B = incidence(7, (0, 0), (epoch // 3) % 3, False)
    elif bmode == 1:
        bgrad = (epoch // 3 + amode + 1) % 3
        B = incidence(7, (2, 2), bgrad, epoch % 5 != 4)
    else:
        # Degree-three B columns cannot coexist with an A row of
        # weight seven under cap nine. Skip that proposal, never repair
        # a completed slot or weaken either weight gate.
        if max(sum(map(len, row)) for row in A) > 6:
            return None
        profiles = (
            ((1, 0), (1, 2)) + ((1, 1),) * 5,
            ((1, 0), (3, 0)) + ((1, 1),) * 3
            + ((0, 2),) * 2,
        )
        B = [[], []]
        for counts in profiles[bmode - 2]:
            for i, count in enumerate(counts):
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    gap = rng.choice(directions)
                    cell = [first, G.mul(first, gap)]
                else:
                    cell = rng.sample(list(G.elements), count)
                B[i].append(cell)

    if B is None:
        return None
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert len(A) == 2 and all(len(row) == 6 for row in A)
    assert len(B) == 2 and all(len(row) == 7 for row in B)
    assert sorted(map(sum, sa)) in ([6, 6], [5, 7])
    assert all(sa[0][j] + sa[1][j] == 2 for j in range(6))
    assert all(sum(row) == 7 for row in sb)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(9, W_MAX)
    assert all(len(set(cell)) == len(cell) <= 5
               for grid in (A, B) for row in grid for cell in row)
    return A, B


def _extract_components(G, hx, hz):
    """Extract components from the verified lift, not a separate lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    # j=0 HZ rows already have consecutive B-column blocks.
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _column_syndromes(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _component_floor(rows, n):
    """Exact minimum through weight four; return 5 if none is found."""
    columns = _column_syndromes(rows, n)
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2
    pairs = set()
    found4 = False
    for i, first in enumerate(columns):
        for second in columns[:i]:
            syndrome = first ^ second
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    return 4 if found4 else 5


def _no_logicals_below_four(hx, hz, n):
    """Complete CSS logical test for weights one, two, and three.

    A weight-three zero syndrome is a pair/single collision.
    Row-space reduction distinguishes logicals from stabilizers.
    """
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = {}
        for row in stabilizers:
            while row:
                pivot = row.bit_length() - 1
                if pivot in basis:
                    row ^= basis[pivot]
                else:
                    basis[pivot] = row
                    break

        def nontrivial(word):
            while word:
                pivot = word.bit_length() - 1
                if pivot not in basis:
                    return True
                word ^= basis[pivot]
            return False

        columns = _column_syndromes(checks, n)
        singles = {0: 0}
        for i, syndrome in enumerate(columns):
            support = 1 << i
            previous = singles.get(syndrome)
            if previous is None:
                singles[syndrome] = support
            elif nontrivial(previous ^ support):
                return False
        for i, first in enumerate(columns):
            for j in range(i):
                previous = singles.get(first ^ columns[j])
                if previous is not None:
                    if nontrivial(previous ^ (1 << i) ^ (1 << j)):
                        return False
    return True


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _portfolio_signature(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant, not a complete iso test.

    High-degree qubits are identified by degree, never LP block labels.
    Rooted refinement retains check hyperedges and low-qubit corners.
    Unequal signatures certify inequivalence; equality is discarded.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _column_syndromes(rows, n)
    high = [j for j, c in enumerate(columns) if c.bit_count() >= 10]
    positions = {j: i for i, j in enumerate(high)}
    endpoints, tags, check_ends = [], [], []

    for row in rows:
        ends = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                ends.append(positions[j])
            bits ^= bit
        ends = tuple(sorted(ends))
        check_ends.append(ends)
        endpoints.append(ends)
        tags.append((0, row.bit_count()))

    for j, syndrome in enumerate(columns):
        if j in positions:
            continue
        ends, weights = [], []
        bits = syndrome
        while bits:
            bit = bits & -bits
            r = bit.bit_length() - 1
            ends.extend(check_ends[r])
            weights.append(rows[r].bit_count())
            bits ^= bit
        endpoints.append(tuple(sorted(ends)))
        tags.append((1, tuple(sorted(weights))))

    incident = [[] for _ in high]
    for edge, ends in enumerate(endpoints):
        for v in ends:
            incident[v].append(edge)

    def compress(signatures):
        mapping = {s: i for i, s in enumerate(sorted(set(signatures)))}
        return [mapping[s] for s in signatures]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(4):
            es = [(tag, tuple(sorted(colours[v] for v in ends)))
                  for tag, ends in zip(tags, endpoints)]
            ec = compress(es)
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            colours = compress(vs)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    # Check-intersection profiles supplement rooted contraction.
    overlaps = [Counter() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i][(rows[j].bit_count(), overlap)] += 1
            overlaps[j][(row.bit_count(), overlap)] += 1

    return (
        n,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(rooted)),
        tuple(sorted((rows[i].bit_count(),
                      tuple(sorted(overlaps[i].items())))
                     for i in range(len(rows)))),
    )


from functools import lru_cache


@lru_cache(maxsize=1)
def _incidence_portfolio(G):
    """Build one deterministic immutable portfolio, independent of call order.

    Proposals are optimized in batches. Only invariant-distinct screened
    winners enter the portfolio; no mutable cross-slot selection state
    is exposed to NEW_CTX or CELL.
    """
    local = random.Random(BASE_SEED ^ 0x8262757)
    cap = min(9, W_MAX)
    entries, graph_keys, filling_keys = [], set(), set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for batch in range(96):
        candidates = []
        # Three trials for each of the twenty structural prescriptions.
        for trial in range(60):
            attempt = 60 * batch + trial
            card, epoch = attempt % 20, attempt // 20
            proposal = _incidence_filling(local, G, card, epoch)
            if proposal is None:
                continue
            A, B = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                raise ValueError("new-shape support budget failed")
            if any(len(set(cell)) != len(cell)
                   for grid in (A, B) for row in grid for cell in row):
                raise ValueError("new-shape cell contains duplicate elements")

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 368 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            ar, br = _extract_components(G, hx, hz)
            da = _component_floor(ar, 6 * G.q)
            db = _component_floor(br, 7 * G.q)
            if min(da, db) < 4:
                continue
            cycles = _four_cycles(ar) + _four_cycles(br)
            # Rank is measured on exactly the submitted matrices.
            # At the protected distance-four floor, larger k wins.
            priority = (-k, -min(da, db), cycles, attempt)
            candidates.append(
                (priority, A, B, hx, hz, n, k, da, db, rx, rz, card))

        candidates.sort(key=lambda record: record[0])
        for _, A, B, hx, hz, n, k, da, db, rx, rz, card in candidates:
            if not _no_logicals_below_four(hx, hz, n):
                continue
            signature = _portfolio_signature(hx, hz, n)
            if signature in graph_keys:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified new-shape matrices failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            graph_keys.add(signature)
            screen = (
                ("k", k),
                ("component_floors", (da, db)),
                ("quantum_below_four_excluded", True),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("design_card", card),
                ("A_row_totals", tuple(sum(map(len, row)) for row in A)),
                ("rank_gain_above_bound", k - 160),
            )
            entries.append((freeze(A), freeze(B), screen))
            if len(entries) == N_SLOTS:
                return tuple(entries)

    # Never recycle an earlier accepted code to fill missing positions.
    return tuple(entries)


def _logical_quotient_four(checks, stabilizers, n):
    """Exact logical floor through four; five means exclusion through four.

    Compare syndromes of all supports of weights zero, one and two.
    Their differences cover every support of weight at most four.
    Linear stabilizer remainders distinguish logicals from stabilizers.
    Burden is a collision diagnostic, not a count of distinct logicals.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def remainder(word):
        free = 0
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                bit = 1 << p
                free |= bit
                word ^= bit
            else:
                word ^= old
        return free

    cols = _column_syndromes(checks, n)
    units = [remainder(1 << i) for i in range(n)]
    seen = {0: (0, 0)}
    floor, burden = 5, 0

    for i, syndrome in enumerate(cols):
        word, quotient = 1 << i, units[i]
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = (word, quotient)
        elif quotient != old[1]:
            weight = (word ^ old[0]).bit_count()
            floor = min(floor, weight)
            burden += 1
    if floor < 4:
        return floor, burden

    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            word = (1 << i) | (1 << j)
            quotient = units[i] ^ units[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = (word, quotient)
            elif quotient != old[1]:
                weight = (word ^ old[0]).bit_count()
                floor = min(floor, weight)
                burden += 1
                if floor < 4:
                    return floor, burden
    return floor, burden


def _socle_signature(hx, hz, n):
    """Strengthen the existing intrinsic, uncoloured Tanner invariant.

    Project low- and high-degree qubits separately onto checks.
    Full qubit degrees, check weights and common-neighbour counts are
    intrinsic; no group labels or assumed X/Z colouring enter the key.
    Equality is inconclusive and is discarded conservatively.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    cols = _column_syndromes(rows, n)
    low, high = [0] * len(rows), [0] * len(rows)
    for column in cols:
        relation = high if column.bit_count() >= 10 else low
        bits = column
        while bits:
            bit = bits & -bits
            relation[bit.bit_length() - 1] |= column ^ bit
            bits ^= bit

    profiles = []
    for i, row in enumerate(rows):
        histogram = Counter()
        for j, other in enumerate(rows):
            if i == j:
                continue
            histogram[(
                other.bit_count(),
                (row & other).bit_count(),
                (low[i] >> j) & 1,
                (high[i] >> j) & 1,
                (low[i] & low[j]).bit_count(),
                (high[i] & high[j]).bit_count(),
                (low[i] & high[j]).bit_count()
                + (high[i] & low[j]).bit_count(),
            )] += 1
        profiles.append((
            row.bit_count(), low[i].bit_count(), high[i].bit_count(),
            tuple(sorted(histogram.items())),
        ))
    return _portfolio_signature(hx, hz, n), tuple(sorted(profiles))


def _socle_filling(rng, G, card, epoch):
    """Five A schedules crossed with four B schedules, before screening."""
    amode, bmode = divmod(card % 20, 4)
    central = next(g for g in G.elements
                   if g != G.e and G.mul(g, g) == G.e)
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    })
    labels = list(G.elements)
    model = epoch % 4
    agrad, bgrad = epoch % 3, (epoch // 3 + 1) % 3

    def grade(g, mode):
        a, b = g
        return (a if mode == 0 else b if mode == 1 else a + b) % 2

    def incidence(width, internal, grading):
        C = [[], []]
        graded = model == 0 and sum(internal) > 0
        gaps = ([g for g in directions if grade(g, grading)]
                if graded else directions)
        if any(count > len(gaps) for count in internal):
            return None

        for i, count in enumerate(internal):
            chosen = rng.sample(gaps, count)
            if model == 1 and count and (epoch + i) % 3 == 0:
                # This order-two direction was forbidden by the parent.
                chosen[0] = central
            for gap in chosen:
                C[i].append([G.e, gap])
                C[1 - i].append([])

        cross = width - sum(internal)
        parity = (epoch // 4) % 2
        pool = ([g for g in labels if grade(g, grading) == parity]
                if graded else labels)
        if cross < 0 or cross > len(pool):
            return None
        voltages = rng.sample(pool, cross)
        for voltage in voltages:
            C[0].append([G.e])
            C[1].append([voltage])
        return C

    def filled_profile(profile):
        C = [[], []]
        phase = rng.choice(labels)
        for j, counts in enumerate(profile):
            cells = []
            for i, count in enumerate(counts):
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [rng.choice(labels)]
                elif count == 2 and model == 1:
                    first = rng.choice(labels)
                    cell = [first, G.mul(first, central)]
                elif count == 2 and model == 0:
                    first = rng.choice(labels)
                    gap = rng.choice(directions)
                    cell = [first, G.mul(first, gap)]
                elif count == 3 and model == 1:
                    first = rng.choice(labels)
                    gap = rng.choice(directions)
                    second = G.mul(first, gap)
                    cell = [first, second, G.mul(second, central)]
                else:
                    cell = rng.sample(labels, count)
                cells.append(cell)

            if model == 2 and counts[0] == counts[1] and counts[0]:
                # Equal-size rows are coupled by a common phase and a
                # column-dependent central factor. Exact quotient tests,
                # not this relation, decide whether short words are safe.
                shift = G.mul(phase, central) if (j + epoch) % 2 else phase
                cells[1] = [G.mul(shift, g) for g in cells[0]]
            for i in range(2):
                C[i].append(cells[i])
        return C

    if amode < 3:
        A = incidence(6, (amode, amode), agrad)
    elif amode == 3:
        A = incidence(6, (1, 2), agrad)
    elif epoch % 2:
        # No singleton A column: degrees 2,2,2,2,3,3; rows 7,7.
        A = filled_profile(((1, 1),) * 4 + ((2, 1), (1, 2)))
    else:
        # Degrees 1,3,2,2,2,2; rows 6,6.
        A = filled_profile(((1, 0), (1, 2)) + ((1, 1),) * 4)
    if A is None:
        return None

    if bmode == 0:
        # One internal edge per row was absent from the parent schedules.
        internal = (1, 1) if epoch % 3 else (0, 0)
        B = incidence(7, internal, bgrad)
    elif bmode == 1:
        # Include the three-internal-edge prescription, rather than
        # restricting the regular internal branch to two edges per row.
        internal = (3, 3) if epoch % 2 else (2, 2)
        B = incidence(7, internal, bgrad)
    elif bmode == 2:
        profiles = (
            ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
            ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
            ((1, 0), (1, 2)) + ((1, 1),) * 5,
        )
        B = filled_profile(profiles[(epoch // 4) % len(profiles)])
    else:
        # Six terms per B row allow degree-three A columns. The two
        # grounding singletons are deliberately screened modulo the
        # complete stabilizer span, not assumed to protect distance.
        B = filled_profile(((1, 0), (0, 1)) + ((1, 1),) * 5)
    if B is None:
        return None

    sa = [[len(c) for c in row] for row in A]
    sb = [[len(c) for c in row] for row in B]
    if not _no_dead_lines(sa, sb):
        return None
    if _grid_weights(sa, sb) > min(9, W_MAX):
        return None
    assert len(A) == 2 and all(len(row) == 6 for row in A)
    assert len(B) == 2 and all(len(row) == 7 for row in B)
    assert all(len(set(c)) == len(c) <= 5
               for grid in (A, B) for row in grid for c in row)
    return A, B, (card, model, agrad, bgrad)


@lru_cache(maxsize=1)
def _socle_portfolio(G):
    """Optimize one immutable portfolio; never publish an archive reference."""
    local = random.Random(BASE_SEED ^ 0x8C264507)
    cap = min(9, W_MAX)
    graph_keys, filling_keys, entries = set(), set(), []

    # The unchanged historical helper reconstructs the parent rule.
    # Its outputs are ONLY exclusion references, never fallback outputs.
    # The evaluator remains authoritative for the unshown archive.
    for old_A, old_B, _ in _incidence_portfolio(G):
        hx, hz, n = lifted_matrices(G, old_A, old_B)
        graph_keys.add(_socle_signature(hx, hz, n))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for batch in range(64):
        candidates = []
        for trial in range(60):
            attempt = 60 * batch + trial
            card, epoch = attempt % 20, attempt // 20
            proposal = _socle_filling(local, G, card, epoch)
            if proposal is None:
                continue
            A, B, prescription = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 368 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            ar, br = _extract_components(G, hx, hz)
            da = _component_floor(ar, 6 * G.q)
            db = _component_floor(br, 7 * G.q)
            # Do not infer quantum failure from a short constituent word.
            priority = (-k, -min(da, db), attempt)
            candidates.append((
                priority, A, B, hx, hz, n, k, rx, rz,
                prescription, (da, db),
            ))

        candidates.sort(key=lambda record: record[0])
        # Reserve a best proposal for each support/element schedule.
        # A rank-only beam could discard all the degenerate experiments.
        selected, represented, identities = [], set(), set()
        for record in candidates:
            schedule = (record[9][0], record[9][1])
            if schedule not in represented:
                represented.add(schedule)
                selected.append(record)
                identities.add(record[0][2])
        for record in candidates[:8]:
            if record[0][2] not in identities:
                selected.append(record)
                identities.add(record[0][2])

        for record in selected:
            (_, A, B, hx, hz, n, k, rx, rz,
             prescription, component_floors) = record
            xf = _logical_quotient_four(hz, hx, n)
            if xf[0] < 4:
                continue
            zf = _logical_quotient_four(hx, hz, n)
            if zf[0] < 4:
                continue

            signature = _socle_signature(hx, hz, n)
            if signature in graph_keys:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified logical-quotient lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            graph_keys.add(signature)

            floor = min(xf[0], zf[0])
            quality = (
                k * floor * floor,
                floor, k, -(xf[1] + zf[1]),
                -record[0][2],
            )
            screen = (
                ("k", k),
                ("logical_profiles_through_four", (xf, zf)),
                ("certified_logical_floor", floor),
                ("component_floors_diagnostic_only", component_floors),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("rank_gain_above_bound", k - 160),
                ("prescription", prescription),
                ("A_row_totals", tuple(sum(map(len, row)) for row in A)),
                ("B_row_totals", tuple(sum(map(len, row)) for row in B)),
            )
            entries.append((quality, freeze(A), freeze(B), screen))

        # Search beyond the first N_SLOTS successes before choosing the
        # portfolio, so later stronger logical floors can displace them.
        if batch >= 31 and len(entries) >= max(80, 3 * N_SLOTS):
            break

    entries.sort(key=lambda entry: entry[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a fresh, globally ranked logical-quotient design."""
    portfolio = _socle_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("fresh logical-quotient portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without sampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Twenty relative-order/quotient-support cards provide twenty inequivalent Dic6 designs: current slots 0..19 collapsed to four classes, and all five measured leaders still had sector-1 weight-four logicals.
# REFUTED: The Q8 (3,9)x(2,7) twenty-policy prefix supplies twenty evaluator-distinct designs or escapes weight four: slots 0..19 retained seven classes, and all five measured leaders had d30s=4 with sector-1 witnesses.
# REFUTED: Independent graded incidence and the rooted portfolio filter eliminate relabeling copies or escape the weight-four plateau on Q8 (2,6)x(2,7): current slots include repeated iso_dup classes, and all five measured leaders have k=171,d30s=4 with abundant sector-1 logicals.
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
