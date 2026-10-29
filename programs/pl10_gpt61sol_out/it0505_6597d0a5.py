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
CONCEPT = ("Cyclic C32, order 32, shapes (1,4)x(1,4): complementary [2,2,3,3] and [2,2,2,2] components selected by independent noisy-syndrome holdouts should improve worse-basis ASC fitness above 3.4852 by removing degree-six sector-one qubits at unchanged CNOT count.")
GROUP = ("Frozen byte-for-byte under L1: retain C32, order 32 and its one-coordinate tuple encoding. This exact bed measured 3.4852; related order-32 beds near 3.5 motivate a filling experiment, not a group advantage. Odd-augmentation cells in F2[C32] are units, motivating full A rank. Connected binary incidence B has rank31. Complete scaffold ranks, not these component arguments, decide the submitted k.")
SHAPES = ("Frozen equal (1,4) component lengths are128 variables and32 checks. n=544 and kb=288 give guaranteed rate9/17 in open-ended cell0.550; require exact k=288 and zero excess dimension. The size factor seventeen permits q<=35, while the frozen group fixes32. Parent failures reached all288 logicals across diffuse Acol/Bcol data and Acol/Brow check blocks, generally enriched only1.0-1.1x. Slot0 X #97 flipped65/500 times and slot3 X #118 flipped51/250; these maps do not establish an isolated repair block.")
FILLING = ("A changes to [2,2,3,3] and B to [2,2,2,2]: component totals10/8, HX weight12, HZ weights10/11, and unchanged17280 six-round CNOTs. Sector-one degrees become four/five, eliminating the parent's degree-six vertices. This deliberately trades component symmetry for complementary protection; it is not asserted to improve both bases automatically. Twenty contracts cross five unordered pairs of trinomial relative-order geometries with four binary B order profiles. Helper-derived singleton actions screen connectivity, ranks, Tanner C4 and kernels through three. Pure full-Tanner reservations consume fresh finalists; independent two-interval noisy-syndrome BP holdouts lead selection, followed by training and exact stabilizer-quotient hook ambiguity. Target the parent's30-46 percent observed failed-shot fractions, diffuse collective flips and abundant weight-four witnesses, rather than distance as fitness. Probes use degree exposure and possible opposite-check suffix faults but omit actual ASC ordering, six-round histories and relay/LSD; they are uncalibrated surrogates, not measured pL. The evaluator remains authoritative for archive equivalence.")
FAMILY_NAME = "r055_C32_14x14_complementary10_8_noisysyndrome_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this cyclic C32 bed requires q=32")
    # Additive C32: abelian, with elements of order 32.
    els = [(a,) for a in range(32)]

    def red(g):
        return (int(g[0]) % 32,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 32,)

    def inv(g):
        return ((-int(g[0])) % 32,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _distinct_terms(rng, G, count):
    terms = []
    seen = set()
    while len(terms) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in seen:
            seen.add(key)
            terms.append(g)
    return terms


def _b_profile(rng, extra, zeros):
    """Seven live columns; fixed total terms and fixed zero-cell count."""
    for _ in range(2000):
        triple = set(rng.sample(range(7), extra))
        sparse = set(rng.sample(range(7), zeros))
        sb = [[0] * 7 for _ in range(2)]
        for j in range(7):
            degree = 3 if j in triple else 2
            if j in sparse:
                a, b = 0, degree
            else:
                a, b = 1, degree - 1
            if rng.randrange(2):
                a, b = b, a
            sb[0][j], sb[1][j] = a, b
        if 0 < sum(sb[0]) <= 9 and 0 < sum(sb[1]) <= 9:
            return sb
    raise ValueError("could not realize deliberate B profile")


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def _retired_lowload_ctx(rng, G, idx):
    """Unused previous-shape experiment retained for provenance."""
    profiles = (
        (1, 2, 2, 3),
        (1, 2, 3, 3),
        (1, 2, 2, 4),
        (1, 2, 2, 2),
    )
    acard, bcard = (idx % 20) // 5, idx % 5
    batch = idx // 20
    extra = batch % 3
    model = 0 if bcard < 3 else bcard
    wanted = 3 * (batch // 3) + bcard if model == 0 else batch // 3

    # The first three B briefs select different invariant classes from
    # the SAME deterministic bank, rather than independent orbit draws.
    srng = random.Random(1407301 + 1009 * acard
                        + 65537 * model + 104729 * extra)
    q = G.q
    mask = (1 << q) - 1
    bmask = (1 << (7 * q)) - 1

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def components(hx, hz):
        # Project only the scaffold's verified matrices.
        ar = [
            sum(((hx[t] >> (j * 7 * q)) & mask) << (j * q)
                for j in range(4))
            for t in range(q)
        ]
        br = [row & bmask for row in hz[:2 * q]]
        return ar, br

    def draw_A():
        steps = srng.sample((1, 2, 3), profiles[acard].count(2))
        row = []
        for weight in profiles[acard]:
            if weight == 1:
                row.append([G.e])
            elif weight == 2:
                row.append([G.e, (steps.pop(), 0)])
            else:
                row.append(srng.sample(G.elements, weight))
        return [row]

    def independent_coset(step, parity):
        shift = srng.randrange(7)
        return [((shift + t * step) % 7, parity) for t in (0, 2, 4)]

    def draw_B():
        B = [[[] for _ in range(7)] for _ in range(2)]
        if model == 0:
            ratios = srng.sample(G.elements, 7)
            start = 0
        else:
            step = srng.choice((1, 2, 3))
            B[0][0] = [G.e, (step, 0)]
            rotations = independent_coset(step, 0)
            reflections = independent_coset(step, 1)
            if model == 3:
                ratios = rotations + reflections
                start = 1
            else:
                B[1][1] = [G.e, (step, 0)]
                ratios = rotations + srng.sample(reflections, 2)
                start = 2
            srng.shuffle(ratios)
        for j, ratio in enumerate(ratios, start):
            B[0][j], B[1][j] = [G.e], [ratio]

        # Extra batches change edge counts, not just labels or priorities.
        # Every B column has degree two or three, and rows stay <=8.
        for _ in range(extra):
            totals = [sum(map(len, row)) for row in B]
            sites = [(i, j) for i in range(2) for j in range(7)
                     if totals[i] < 8
                     and len(B[0][j]) + len(B[1][j]) == 2]
            smallest = min(totals[i] for i, j in sites)
            sites = [(i, j) for i, j in sites if totals[i] == smallest]
            i, j = srng.choice(sites)
            choices = [g for g in G.elements if g not in B[i][j]]
            B[i][j].append(srng.choice(choices))
        return B

    def quotient_columns(rows, width):
        """Linear remainders modulo the actual stabilizer row space."""
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        result = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                result[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= result[bit.bit_length() - 1]
                tail ^= bit
            result[c] = value
        return result

    def fault_profile(checks, stabilizers, width):
        """Truncated-channel ML ambiguity, not ASC circuit simulation.

        Integer masses make the returned profile invariant under Tanner
        relabelings. Quotient labels distinguish logical differences.
        """
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        residue = quotient_columns(stabilizers, width)
        exposure = [a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]

        # Approximate six-round single-data rates in units of 1/10000.
        rates = [64 * (degree + 1) for degree in exposure]
        hooks = {}
        for row in stabilizers:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length() - 1)
                row ^= bit
            weight = len(sites)
            if weight < 2:
                continue
            # Spread one pessimistic weight-two hook mass over all pairs.
            # Actual ASC suffixes and measurement faults are not modeled.
            mass = 80000 // (weight * (weight - 1) // 2)
            for t, b in enumerate(sites):
                for a in sites[:t]:
                    key = (b, a)
                    hooks[key] = hooks.get(key, 0) + mass

        buckets = {0: {0: 100000000}}

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0) + mass

        singles = {}
        for i, s in enumerate(syndrome):
            label = residue[i]
            if s == 0 and label:
                return None
            if s in singles and singles[s] != label:
                return None
            singles[s] = label
            add(s, label, rates[i] * 10000)

        for i, si in enumerate(syndrome):
            ri = residue[i]
            for j in range(i):
                s, label = si ^ syndrome[j], ri ^ residue[j]
                if s == 0 and label:
                    return None
                mass = rates[i] * rates[j] + hooks.get((i, j), 0)
                add(s, label, mass)

        loss = ambiguous = worst = 0
        for classes in buckets.values():
            if len(classes) <= 1:
                continue
            error = sum(classes.values()) - max(classes.values())
            loss += error
            ambiguous += 1
            worst = max(worst, error)
        return loss, ambiguous, worst

    # Component screening is cheap; full fault classification is reserved
    # for a diverse shortlist rather than many equivalent best-cost draws.
    groups = {}
    target_k = 213 if extra == 0 else 210
    for trial in range(192):
        A, B = draw_A(), draw_B()
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != target_k:
            continue
        ar, br = components(hx, hz)
        ast = _syndrome_risk(ar, 4 * q)
        bst = _syndrome_risk(br, 7 * q)
        if ast is None or bst is None:
            continue
        coarse = (max(ast[0], bst[0]), ast[0] + bst[0],
                  max(ast[1], bst[1]), ast[1] + bst[1])
        bucket = groups.setdefault(coarse, [])
        if len(bucket) < 3:
            bucket.append((trial, A, B, hx, hz, n))

    shortlist = []
    # Round-robin retains different ambiguity levels, not just one orbit.
    keys = sorted(groups)
    for position in range(3):
        for key in keys:
            if position < len(groups[key]):
                shortlist.append((key,) + groups[key][position])
                if len(shortlist) == 24:
                    break
        if len(shortlist) == 24:
            break

    book = {}
    for coarse, trial, A, B, hx, hz, n in shortlist:
        pz = fault_profile(hx, hz, n)
        if pz is None:
            continue
        px = fault_profile(hz, hx, n)
        if px is None:
            continue
        # These statistics are invariant: distinct values rule out orbit
        # copies within this bank. Equality is conservatively discarded.
        invariant = tuple(sorted((px, pz)))
        score = (max(px[0], pz[0]), px[0] + pz[0],
                 max(px[2], pz[2]), px[2] + pz[2],
                 max(px[1], pz[1]), px[1] + pz[1])
        item = (score, trial, A, B, hx, hz, invariant)
        previous = book.get(invariant)
        if previous is None or (score, trial) < previous[:2]:
            book[invariant] = item

    entries = sorted(book.values(), key=lambda item: item[:2])
    if wanted >= len(entries):
        raise ValueError("insufficient invariant-distinct fault designs")
    score, trial, A, B, hx, hz, invariant = entries[wanted]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return {"A": A, "B": B, "k": target_k,
            "brief": (acard, bcard, extra, wanted),
            "fault_cost": score, "invariant": invariant}


def _balanced_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= marker
            row ^= bit
    return out


def _balanced_components(hx, hz, q):
    """Project component rows exclusively from the verified CSS lift."""
    mask = (1 << q) - 1
    ar = [
        sum(((hx[(i * 8) * q + t] >> (j * 8 * q)) & mask)
            << (j * q) for j in range(8))
        for i in range(2) for t in range(q)
    ]
    br = [row & ((1 << (8 * q)) - 1) for row in hz[:2 * q]]
    return ar, br


def _balanced_quotient(rows, width):
    """Canonical linear remainders modulo the actual stabilizer span."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    result = [0] * width
    for c in range(width):
        row = pivots.get(c)
        if row is None:
            result[c] = 1 << c
            continue
        tail = row ^ (1 << c)
        value = 0
        while tail:
            bit = tail & -tail
            value ^= result[bit.bit_length() - 1]
            tail ^= bit
        result[c] = value
    return result


def _balanced_fault_cost(checks, stabilizers, n):
    """Truncated fault-channel ambiguity, not an ASC simulation.

    Quotient classes separate logical differences from stabilizers.
    Shared opposite checks mark possible spread pairs, not certified
    ASC hooks. Integer masses allow invariant comparisons.
    """
    syndrome = _balanced_columns(checks, n)
    spread = _balanced_columns(stabilizers, n)
    residue = _balanced_quotient(stabilizers, n)
    if 0 in syndrome or len(set(syndrome)) != n:
        return None
    rates = [1 + a.bit_count() + b.bit_count()
             for a, b in zip(syndrome, spread)]
    buckets = {0: {0: 100000000}}

    def add(s, label, mass):
        classes = buckets.setdefault(s, {})
        classes[label] = classes.get(label, 0) + mass

    for i, s in enumerate(syndrome):
        add(s, residue[i], 500 * rates[i])
    for i, si in enumerate(syndrome):
        ri, hi = residue[i], spread[i]
        for j in range(i):
            s = si ^ syndrome[j]
            label = ri ^ residue[j]
            if s == 0 and label:
                return None
            # A heuristic first-order spread contribution supplements
            # the second-order independent-data contribution.
            mass = rates[i] * rates[j]
            mass += 128 * (hi & spread[j]).bit_count()
            add(s, label, mass)

    loss = worst = ambiguous = 0
    spectrum = {}
    for classes in buckets.values():
        if len(classes) > 1:
            error = sum(classes.values()) - max(classes.values())
            loss += error
            worst = max(worst, error)
            ambiguous += 1
            spectrum[error] = spectrum.get(error, 0) + 1
    return loss, worst, ambiguous, tuple(sorted(spectrum.items()))


def _retired_balanced_ctx(rng, G, idx):
    """Unused previous-shape support-card search retained for provenance."""
    from itertools import combinations_with_replacement

    # Six architectures, all with 17 or 18 terms.
    # An internal column is (0,2) or (2,0), never a dead column.
    def support(card):
        grid = [[1] * 8 for _ in range(2)]
        if card in (0, 1):
            grid[0][0] = 2
            if card == 1:
                grid[1][1] = 2
        elif card in (2, 5):
            grid[0][0], grid[1][0] = 0, 2
            grid[0][1] = 2
            if card == 5:
                grid[0][2] = 2
        else:
            grid[0][0], grid[1][0] = 0, 2
            grid[0][1], grid[1][1] = 2, 0
            grid[0][2] = 2
            if card == 4:
                grid[1][3] = 2
        return grid

    # Equal-length sides make A/B swapping redundant, so use unordered
    # pairs rather than spending slots on the two side assignments.
    cards = tuple(pair for pair in combinations_with_replacement(
        range(6), 2) if pair != (0, 0))
    assert len(cards) == 20
    design = idx % 20
    batch = idx // 20
    policy = batch % 3
    pa, pb = cards[design]
    sa, sb = support(pa), support(pb)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(map(sum, sa + sb)) <= 9
    assert all(sum(grid[i][j] for i in range(2)) <= 3
               for grid in (sa, sb) for j in range(8))

    rotations = ((1, 0), (3, 0))
    involutions = ((2, 0), (0, 1), (1, 1), (2, 1), (3, 1))
    directions = (rotations,
                  tuple(g for g in G.elements if g != G.e),
                  involutions)[policy]

    def draw(counts):
        # Normalize the upper monomial cells rather than wasting trials
        # on arbitrary column gauges. Lower monomial voltages are distinct.
        lower_sites = [j for j in range(8) if counts[1][j] == 1]
        voltages = dict(zip(
            lower_sites, rng.sample(G.elements, len(lower_sites))))
        grid = [[[] for _ in range(8)] for _ in range(2)]
        for i in range(2):
            for j in range(8):
                number = counts[i][j]
                if not number:
                    continue
                if number == 1:
                    grid[i][j] = [G.e if i == 0 else voltages[j]]
                else:
                    internal = counts[1 - i][j] == 0
                    # Order-two ratios inside a degree-two column create
                    # parallel incidence columns; internal ratios must
                    # therefore have order four.
                    d = rng.choice(rotations if internal else directions)
                    g = G.e if i == 0 else G.pick(rng)
                    grid[i][j] = [g, G.mul(g, d)]
        return grid

    pool = []
    seen = set()
    for trial in range(192):
        A, B = draw(sa), draw(sb)
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k != 288:
            continue
        ar, br = _balanced_components(hx, hz, G.q)
        aa = _syndrome_risk(ar, 8 * G.q)
        bb = _syndrome_risk(br, 8 * G.q)
        if aa is None or bb is None:
            continue
        key = (max(aa[0], bb[0]), aa[0] + bb[0],
               max(aa[1], bb[1]), aa[1] + bb[1])
        pool.append((key, trial, A, B, hx, hz))
        pool.sort(key=lambda item: item[:2])
        del pool[10:]

    # Equal invariant profiles are conservatively collapsed within the
    # shortlist. No submitted slot is produced by transforming a saved
    # code; arbitrary Tanner isomorphism remains the evaluator's test.
    book = {}
    for coarse, trial, A, B, hx, hz in pool:
        px = _balanced_fault_cost(hz, hx, 544)
        pz = _balanced_fault_cost(hx, hz, 544)
        if px is None or pz is None:
            continue
        invariant = tuple(sorted((px, pz)))
        score = (max(px[0], pz[0]), px[0] + pz[0],
                 max(px[1], pz[1]), px[1] + pz[1],
                 max(px[2], pz[2]), px[2] + pz[2], coarse)
        item = (score, trial, A, B, hx, hz, invariant)
        old = book.get(invariant)
        if old is None or item[:2] < old[:2]:
            book[invariant] = item

    entries = sorted(book.values(), key=lambda item: item[:2])
    wanted = batch // 3
    if wanted >= len(entries):
        raise ValueError("no unused balanced fault-profile finalist")
    score, trial, A, B, hx, hz, invariant = entries[wanted]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    return {"A": A, "B": B, "k": 288, "design": design,
            "ratio_policy": policy, "fault_cost": score,
            "invariant": invariant}


def _single_component_class(rows, width, q):
    """Verified component statistics; none is a circuit score."""
    columns = _balanced_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    if gf2_rank(rows) != q:
        return None

    adjacency = [0] * q
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap * (overlap - 1) // 2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    if four:
        return None

    reached = frontier = 1
    while frontier:
        following = 0
        while frontier:
            bit = frontier & -frontier
            following |= adjacency[bit.bit_length() - 1]
            frontier ^= bit
        frontier = following & ~reached
        reached |= frontier
    if reached != (1 << q) - 1:
        return None

    short = _syndrome_risk(rows, width)
    if short is None or short[0]:
        return None

    # With C4 excluded, check-graph triangles minus variable stars
    # count Tanner six-cycles exactly.
    triangles = 0
    for i, neighbors in enumerate(adjacency):
        higher = neighbors >> (i + 1)
        while higher:
            bit = higher & -higher
            j = i + 1 + bit.bit_length() - 1
            triangles += (
                (neighbors & adjacency[j]) >> (j + 1)
            ).bit_count()
            higher ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6
                for d in (column.bit_count() for column in columns))
    six = triangles - stars
    assert six >= 0
    return short[1], six


def _single_component_fingerprint(rows, width):
    """Rooted uncolored Tanner invariant, not full-code canonization.

    Regular lift translations are transitive on a single-row component's
    checks. Thus rooting at check zero does not spend proposals on a
    check relabeling. Equal fingerprints are conservatively discarded.
    """
    from collections import Counter
    import hashlib

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, syndrome in enumerate(_balanced_columns(rows, width)):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            syndrome ^= bit

    colors = [0] * nr + [
        1 if len(adjacency[nr + c]) == 2 else 2
        for c in range(width)
    ]
    colors[0] = 3
    digest = hashlib.sha256()
    for _ in range(14):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in adjacency[v])))
            for v in range(len(adjacency))
        ]
        digest.update(repr(sorted(
            Counter(signatures).items())).encode("ascii"))
        palette = {signature: t for t, signature in enumerate(
            sorted(set(signatures)))}
        colors = [palette[signature] for signature in signatures]
    return digest.hexdigest()


def _retired_metacyclic_ctx(rng, G, idx):
    """Inactive parent catalogue retained for provenance; never called."""
    from itertools import combinations, combinations_with_replacement

    q = G.q
    assert q == 32
    width = 4 * q
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 12

    def element_order(g):
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return order
        raise ValueError("invalid metacyclic element order")

    orders = [element_order(g) for g in G.elements]
    representatives = [
        t for t, g in enumerate(G.elements)
        if t < G.idx(G.inv(g)) and orders[t] in (4, 8)
    ]

    # These relative triples generate respectively:
    # the full nonabelian group of order 32;
    # a nonabelian subgroup of order 16;
    # an abelian C8 x C2 subgroup of order 16;
    # a cyclic subgroup of order eight.
    triples = (
        (G.e, (1, 0), (0, 1)),
        (G.e, (2, 0), (0, 1)),
        (G.e, (1, 0), (2, 2)),
        (G.e, (1, 0), (3, 0)),
    )
    policies = ((4, 8, 8), (4, 4, 8))
    type_pairs = tuple(combinations_with_replacement(range(4), 2))
    cards = [(policy, a, b)
             for policy in range(len(policies))
             for a, b in type_pairs]
    assert len(cards) == 20

    # Component probes use the scaffold construction itself. The zero
    # opposite component is never submitted as a candidate code.
    def component(grid, side):
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    books = {}
    for geometry, triple in enumerate(triples):
        for policy, wanted in enumerate(policies):
            raw = []
            for generators in combinations(representatives, 3):
                if tuple(sorted(orders[t] for t in generators)) != wanted:
                    continue
                grid = [[[G.e, G.elements[t]] for t in generators] +
                        [list(triple)]]
                rows = component(grid, "A")
                quality = _single_component_class(rows, width, q)
                if quality is None:
                    continue
                labeling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for cell in grid[0])
                raw.append((quality, labeling, grid, rows))

            # Inspect quality-ranked proposals, retaining different
            # uncolored classes rather than many gauges of one minimum.
            unique = {}
            for quality, labeling, grid, rows in sorted(
                    raw, key=lambda item: item[:2]):
                identity = _single_component_fingerprint(rows, width)
                if identity not in unique:
                    unique[identity] = (quality, identity, labeling, grid)
                if len(unique) >= 12:
                    break
            books[geometry, policy] = sorted(
                unique.values(), key=lambda item: item[:3])

    catalogues = []
    for policy, ta, tb in cards:
        candidates = {}
        for a in books[ta, policy]:
            for b in books[tb, policy]:
                # A/B exchange must not consume two slots.
                identity = tuple(sorted((a[1], b[1])))
                coarse = (
                    max(a[0][0], b[0][0]), a[0][0] + b[0][0],
                    max(a[0][1], b[0][1]), a[0][1] + b[0][1],
                )
                item = (coarse, identity, a[3], b[3])
                previous = candidates.get(identity)
                if previous is None or coarse < previous[0]:
                    candidates[identity] = item
        catalogues.append(sorted(
            candidates.values(), key=lambda item: item[:2]))

    # Rebuild the same reservation prefix in each pure call. Extra
    # slots select unused class pairs, not new random orbit images.
    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx + 1):
        card = slot % len(cards)
        available = [item for item in catalogues[card]
                     if item[1] not in reserved]
        if not available:
            alternatives = [
                (item[0], item[1], c)
                for c, catalogue in enumerate(catalogues)
                for item in catalogue if item[1] not in reserved
            ]
            if not alternatives:
                raise ValueError("unused structural component pairs exhausted")
            card = min(alternatives)[2]
            available = [item for item in catalogues[card]
                         if item[1] not in reserved]
        chosen = available[:3]
        actual_card = card
        # Reserve all finalists, including those not eventually selected.
        reserved.update(item[1] for item in chosen)

    best = None
    seen_profiles = set()
    mask = (1 << q) - 1
    for coarse, identity, A, B in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k != 288:
            continue

        # Verify the screened components in the final evaluator matrices.
        ar = [
            sum(((hx[t] >> (j * 4 * q)) & mask) << (j * q)
                for j in range(4))
            for t in range(q)
        ]
        br = [row & ((1 << width) - 1) for row in hz[:q]]
        assert ar == component(A, "A")
        assert br == component(B, "B")
        if _single_component_class(br, width, q) is None:
            continue

        px = _balanced_fault_cost(hz, hx, n)
        pz = _balanced_fault_cost(hx, hz, n)
        if px is None or pz is None:
            continue
        invariant = tuple(sorted((px, pz)))
        if invariant in seen_profiles:
            continue
        seen_profiles.add(invariant)

        # Component short-word counts only make the shortlist. Final
        # selection protects the worse basis's weighted logical-fault
        # ambiguity, rather than maximizing a sampled distance.
        key = (
            max(px[0], pz[0]), px[0] + pz[0],
            max(px[1], pz[1]), px[1] + pz[1],
            max(px[2], pz[2]), px[2] + pz[2],
            coarse, identity,
        )
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, invariant)

    if best is None:
        raise ValueError("no connected exact-rate fault-screened finalist")
    key, A, B, hx, hz, invariant = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert sum(row.bit_count() for row in hx + hz) * 6 == 17280
    assert css_ok(hx, hz)
    return {"A": A, "B": B, "k": 288,
            "design": cards[actual_card], "fault_cost": key,
            "invariant": invariant}


def _c32_full_fingerprint(hx, hz, n, q):
    """Rooted full Tanner invariant, with X/Z check colors merged.

    The degree-18 data vertices are precisely sector two. Cyclic
    translations act transitively on them, so rooting at its first
    vertex is invariant under Tanner isomorphism for this support.
    Equal hashes are conservatively discarded, not accepted as proof
    that two codes are isomorphic.
    """
    from collections import Counter
    import hashlib

    checks = hx + hz
    nr = len(checks)
    adjacency = [[] for _ in range(nr + n)]
    for r, word in enumerate(checks):
        while word:
            bit = word & -word
            c = bit.bit_length() - 1
            adjacency[r].append(nr + c)
            adjacency[nr + c].append(r)
            word ^= bit

    root = 16*q
    assert len(adjacency[nr + root]) == 18
    assert all(len(adjacency[nr + c]) <= 6 for c in range(root))
    assert all(len(adjacency[nr + c]) == 18
               for c in range(root, n))

    colors = [0]*nr + [1]*n
    colors[nr + root] = 2
    digest = hashlib.sha256()
    for _ in range(12):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in adjacency[v])))
            for v in range(nr + n)]
        digest.update(repr(sorted(
            Counter(signatures).items())).encode("ascii"))
        palette = {s: t for t, s in enumerate(sorted(set(signatures)))}
        colors = [palette[s] for s in signatures]
    return digest.hexdigest()


def _c32_hook_cost(checks, stabilizers, n):
    """Logical pair ambiguities on the exact verified CSS matrices.

    Shared opposite checks count possible pair-spread opportunities.
    They are not hooks certified by the ASC CNOT ordering.
    Quotient labels distinguish stabilizers from logical differences.
    """
    syndrome = _balanced_columns(checks, n)
    if 0 in syndrome or len(set(syndrome)) != n:
        return None
    spread = _balanced_columns(stabilizers, n)
    residue = _balanced_quotient(stabilizers, n)
    singles = dict(zip(syndrome, residue))
    pairs = {}
    three = four = double_hook = single_hook = 0

    for i, si in enumerate(syndrome):
        ri, hi = residue[i], spread[i]
        for j in range(i):
            s = si ^ syndrome[j]
            label = ri ^ residue[j]
            third = singles.get(s)
            if third is not None and third != label:
                three += 1
            hook = (hi & spread[j]).bit_count()

            old = pairs.get(s)
            if old is None:
                pairs[s] = [1, hook, {label: [1, hook]}]
                continue
            same = old[2].get(label)
            same_count, same_hooks = same if same else (0, 0)
            different = old[0] - same_count
            different_hooks = old[1] - same_hooks

            four += different
            double_hook += hook*different_hooks
            single_hook += hook*different + different_hooks
            old[0] += 1
            old[1] += hook
            if same is None:
                old[2][label] = [1, hook]
            else:
                same[0] += 1
                same[1] += hook

    assert three % 3 == 0 and four % 3 == 0
    return three//3, double_hook, single_hook, four//3


def _retired_c32_balanced_ctx(rng, G, idx):
    """Inactive balanced C32 catalogue retained for provenance."""
    from itertools import combinations

    q = G.q
    assert q == 32 and A_SHAPE == B_SHAPE == (1, 4)
    counts = [[2, 2, 2, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 12
    assert 9 + 3 <= W_MAX
    width = 4*q
    mask = (1 << q)-1
    els = G.elements

    # These blocks come exclusively from the verified construction.
    # Their XOR assembly retains its exact left/right conventions.
    probe = [[[g] for g in els]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {
        "A": [tuple((word >> (t*q)) & mask for word in ax)
              for t in range(q)],
        "B": [tuple((word >> (t*q)) & mask for word in bz)
              for t in range(q)],
    }

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    def direction(t):
        t %= q
        return min(t, q-t)

    def order(t):
        # Orders are checked against the actual validated group law.
        value = G.e
        for power in range(1, q+1):
            value = G.mul(value, els[t])
            if value == G.e:
                return power
        raise ValueError("invalid C32 element")

    orders = [order(t) for t in range(q)]
    pools = {
        length: [t for t in range(1, q//2)
                 if orders[t] == length]
        for length in (8, 16, 32)
    }
    profiles = (
        (32, 32, 32),
        (16, 32, 32),
        (16, 16, 32),
        (8, 32, 32),
    )
    geometries = (
        (16, 32, 32),
        (8, 32, 32),
        (4, 32, 32),
        (8, 16, 16),
        (4, 16, 16),
    )

    triangles = [[] for _ in geometries]
    for a, b in combinations(range(1, q), 2):
        internal = {direction(a), direction(b), direction(b-a)}
        if len(internal) != 3 or q//2 in internal:
            continue
        signature = tuple(sorted(orders[t] for t in internal))
        for geometry, wanted in enumerate(geometries):
            if signature == wanted:
                triangles[geometry].append((a, b, internal))
    assert all(triangles)

    # Twenty explicit order contracts, not twenty transformed copies.
    cards = [(geometry, policy)
             for geometry in range(5) for policy in range(4)]
    books = {}
    for geometry, policy in cards:
        crng = random.Random(32055317 + 104729*(4*geometry+policy))
        raw = {}
        profile = profiles[policy]
        for trial in range(96):
            a, b, internal = crng.choice(triangles[geometry])
            generators = []
            possible = True
            for length in (8, 16, 32):
                available = [t for t in pools[length]
                             if t not in internal]
                number = profile.count(length)
                if len(available) < number:
                    possible = False
                    break
                generators.extend(crng.sample(available, number))
            if not possible:
                continue
            generators.sort()

            # No binomial triangles, repeated directions, involutions,
            # or repeated within-trinomial differences.
            signed = {s*t % q for t in generators for s in (-1, 1)}
            if any((x+y) % q in signed for x in signed for y in signed):
                continue
            labeling = tuple(generators) + (a, b)
            if labeling in raw:
                continue
            grid = [[[G.e, els[t]] for t in generators]
                    + [[G.e, els[a], els[b]]]]
            rows = component(grid, "A")
            quality = _single_component_class(rows, width, q)
            if quality is not None:
                raw[labeling] = (quality, labeling, grid, rows)

        unique = {}
        for quality, labeling, grid, rows in sorted(
                raw.values(), key=lambda entry: entry[:2]):
            identity = _single_component_fingerprint(rows, width)
            if identity not in unique:
                unique[identity] = (quality, identity, labeling, grid)
            if len(unique) >= 10:
                break
        books[geometry, policy] = sorted(
            unique.values(), key=lambda entry: entry[:3])

    catalogues = []
    for geometry, policy in cards:
        opposite = ((geometry+2+policy % 2) % 5, (policy+1) % 4)
        candidates = {}
        for a in books[geometry, policy]:
            for b in books[opposite]:
                identity = tuple(sorted((a[1], b[1])))
                coarse = (
                    max(a[0][0], b[0][0]), a[0][0]+b[0][0],
                    max(a[0][1], b[0][1]), a[0][1]+b[0][1])
                entry = (coarse, identity, a[3], b[3])
                previous = candidates.get(identity)
                if previous is None or coarse < previous[0]:
                    candidates[identity] = entry
        catalogues.append(sorted(
            candidates.values(), key=lambda entry: entry[:2]))

    # Rebuild the same prefix in each pure call. Reserve all finalists,
    # including unsubmitted ones, without mutable cross-slot state.
    # Full-graph rooting removes equivalences missed by component pairs.
    reserved_pairs = set()
    reserved_graphs = set()
    cursors = [0]*len(cards)
    chosen = None
    actual_card = None
    for slot in range(idx+1):
        selected = []
        preferred = slot % len(cards)
        visit = [preferred] + [
            c for c in range(len(cards)) if c != preferred]
        for card in visit:
            catalogue = catalogues[card]
            while cursors[card] < len(catalogue) and len(selected) < 2:
                coarse, identity, A, B = catalogue[cursors[card]]
                cursors[card] += 1
                if identity in reserved_pairs:
                    continue
                reserved_pairs.add(identity)

                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 544 or k != 288:
                    continue
                graph = _c32_full_fingerprint(hx, hz, n, q)
                if graph in reserved_graphs:
                    continue
                reserved_graphs.add(graph)
                selected.append((coarse, graph, A, B, hx, hz))
                actual_card = card
            if len(selected) == 2:
                break
        if not selected:
            raise ValueError("fresh C32 full-Tanner catalogue exhausted")
        chosen = selected

    best = None
    for coarse, graph, A, B, hx, hz in chosen:
        # Cross-check cached components against the final evaluator lift.
        ar = [
            sum(((hx[t] >> (j*4*q)) & mask) << (j*q)
                for j in range(4))
            for t in range(q)]
        br = [word & ((1 << width)-1) for word in hz[:q]]
        assert ar == component(A, "A")
        assert br == component(B, "B")
        if _single_component_class(br, width, q) is None:
            continue

        px = _c32_hook_cost(hz, hx, 544)
        pz = _c32_hook_cost(hx, hz, 544)
        if px is None or pz is None or px[0] or pz[0]:
            continue
        # Protect the worse basis. At low physical error rate, compare
        # possible two-hook ambiguities before one-hook opportunities
        # and ordinary pair collisions; this is a hypothesis about pL.
        key = tuple(v for t in range(1, 4)
                    for v in (max(px[t], pz[t]), px[t]+pz[t]))
        key += (coarse, graph)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, graph, px, pz)

    if best is None:
        raise ValueError("no exact-rate C32 hook-screened finalist")
    key, A, B, hx, hz, graph, px, pz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert all(len(set(cell)) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    assert max(word.bit_count() for word in hx+hz) == 12
    assert 6*sum(word.bit_count() for word in hx+hz) == 17280
    assert 544-gf2_rank(hx)-gf2_rank(hz) == 288
    assert css_ok(hx, hz)
    return {"A": A, "B": B, "k": 288,
            "design": cards[actual_card], "screen": key,
            "full_graph": graph, "hook_profile": (px, pz)}


def NEW_CTX(rng, G, idx):
    """Complementary degree contracts and independent syndrome holdouts."""
    from itertools import combinations
    import numpy as np

    q = G.q
    assert q == 32 and A_SHAPE == B_SHAPE == (1, 4)
    sa, sb = [[2, 2, 3, 3]], [[2, 2, 2, 2]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    width, mask = 4*q, (1 << q)-1
    els = G.elements

    # Assemble only singleton blocks extracted from the verified helper.
    probe = [[[g] for g in els]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {
        "A": [tuple((word >> (t*q)) & mask for word in ax)
              for t in range(q)],
        "B": [tuple((word >> (t*q)) & mask for word in bz)
              for t in range(q)],
    }

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, word in enumerate(actions[side][G.idx(g)]):
                    rows[r] ^= word << (j*q)
        return rows

    def direction(t):
        t %= q
        return min(t, q-t)

    def order(t):
        word = G.e
        for power in range(1, q+1):
            word = G.mul(word, els[t])
            if word == G.e:
                return power
        raise ValueError("invalid frozen C32 element")

    orders = [order(t) for t in range(q)]
    units = tuple(t for t in range(1, q, 2))

    def normal(cell, multiplier=1):
        values = [(multiplier*t) % q for t in cell]
        return min(tuple(sorted((t-anchor) % q for t in values))
                   for anchor in values)

    def canonical(cells):
        # Component gauges only; no saved complete code is transformed.
        # Full-code equivalence is screened separately below.
        return min(tuple(
            tuple(sorted(normal(cell, u) for cell in cells
                         if len(cell) == degree))
            for degree in (2, 3))
            for u in units)

    def encode(geometry):
        return [[[els[t] for t in cell]
                 for part in geometry for cell in part]]

    def brief(rows, rank):
        """Verified component tests; quartet/six counts are surrogates."""
        adjacency = [0]*q
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap > 1:
                    return None
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length()-1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        if reached != (1 << q)-1 or gf2_rank(rows) != rank:
            return None
        short = _syndrome_risk(rows, width)
        if short is None or short[0]:
            return None
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles += (neighbors & adjacency[j]
                              & ((1 << j)-1)).bit_count()
                lower ^= bit
        stars = sum(d*(d-1)*(d-2)//6 for d in (
            c.bit_count() for c in _balanced_columns(rows, width)))
        six = triangles-stars
        assert six >= 0 and short[1] % 3 == 0
        return short[1]//3, six

    # The smallest trinomial subgroup geometry is new to the parent's
    # five-geometry catalogue. Its presence is a contract, not a claim
    # that a small subgroup improves the memory experiment.
    geometries = (
        (4, 8, 8), (4, 16, 16), (8, 16, 16),
        (4, 32, 32), (8, 32, 32), (16, 32, 32),
    )
    triangle_sets = [set() for _ in geometries]
    for a, b in combinations(range(1, q), 2):
        internal = {direction(a), direction(b), direction(b-a)}
        if len(internal) != 3 or q//2 in internal:
            continue
        descriptor = tuple(sorted(orders[t] for t in internal))
        if descriptor in geometries:
            triangle_sets[geometries.index(descriptor)].add(
                normal((0, a, b)))
    triangles = [sorted(book) for book in triangle_sets]
    assert all(triangles)

    def internal(cell):
        return {direction(a-b) for a, b in combinations(cell, 2)}

    a_contracts = ((0, 5), (1, 5), (2, 4), (3, 4), (4, 5))
    b_profiles = (
        (32, 32, 32, 32),
        (16, 32, 32, 32),
        (16, 16, 32, 32),
        (8, 32, 32, 32),
    )
    long_directions = [t for t in range(1, q//2)
                       if orders[t] == 32]

    def retain(raw, side, rank):
        unique = {}
        for quality, labeling, grid, rows in sorted(
                raw.values(), key=lambda entry: entry[:2]):
            fp = _single_component_fingerprint(rows, width)
            if fp not in unique:
                unique[fp] = (quality, fp, labeling, grid)
            if len(unique) >= 24:
                break
        return sorted(unique.values(), key=lambda entry: entry[:3])

    abooks = []
    for card, (first_type, second_type) in enumerate(a_contracts):
        crng = random.Random(321080791 + 104729*card)
        seen, raw = set(), {}
        for trial in range(3072):
            first = crng.choice(triangles[first_type])
            second = crng.choice(triangles[second_type])
            fi, si = internal(first), internal(second)
            if fi & si:
                continue
            available = [t for t in long_directions
                         if t not in fi | si]
            if len(available) < 2:
                continue
            gaps = crng.sample(available, 2)
            geometry = canonical(
                ((0, gaps[0]), (0, gaps[1]), first, second))
            if geometry in seen:
                continue
            seen.add(geometry)
            grid = encode(geometry)
            rows = component(grid, "A")
            quality = brief(rows, q)
            if quality is not None:
                raw[geometry] = (quality, geometry, grid, rows)
        abooks.append(retain(raw, "A", q))

    bbooks = []
    for profile in b_profiles:
        seen, raw = set(), {}
        for gaps in combinations(range(1, q//2), 4):
            if tuple(sorted(orders[t] for t in gaps)) != profile:
                continue
            signed = {s*t % q for t in gaps for s in (-1, 1)}
            if any((a+b) % q in signed
                   for a in signed for b in signed):
                continue
            geometry = canonical(tuple((0, t) for t in gaps))
            if geometry in seen:
                continue
            seen.add(geometry)
            grid = encode(geometry)
            rows = component(grid, "B")
            quality = brief(rows, q-1)
            if quality is not None:
                raw[geometry] = (quality, geometry, grid, rows)
        bbooks.append(retain(raw, "B", q-1))

    if not any(abooks) or not any(bbooks):
        raise ValueError("complementary component books are empty")

    # Five A hyperedge contracts crossed with four B graph contracts.
    cards = [(a, b) for a in range(5) for b in range(4)]
    assert len(cards) == 20
    catalogues = []
    for a_card, b_card in cards:
        proposals = {}
        for a in abooks[a_card]:
            for b in bbooks[b_card]:
                identity = (a[1], b[1])
                coarse = (
                    max(a[0][0], b[0][0]), a[0][0]+b[0][0],
                    max(a[0][1], b[0][1]), a[0][1]+b[0][1])
                proposals[identity] = (coarse, identity, a[3], b[3])
        catalogues.append(sorted(
            proposals.values(), key=lambda entry: entry[:2]))

    # Reconstruct deterministic reservations in every pure call.
    # Degree-18 sector-two roots remain one cyclic translation orbit,
    # so the existing full-Tanner invariant applies to this architecture.
    reserved_pairs, reserved_graphs = set(), set()
    cursors = [0]*len(cards)
    chosen = None
    for slot in range(idx+1):
        selected = []
        preferred = slot % len(cards)
        visit = [preferred]+[c for c in range(len(cards))
                             if c != preferred]
        for card in visit:
            catalogue = catalogues[card]
            while cursors[card] < len(catalogue) and len(selected) < 2:
                coarse, identity, A, B = catalogue[cursors[card]]
                cursors[card] += 1
                if identity in reserved_pairs:
                    continue
                reserved_pairs.add(identity)
                hx, hz, n = lifted_matrices(G, A, B)
                if n != 544 or n-gf2_rank(hx)-gf2_rank(hz) != 288:
                    continue
                fp = _c32_full_fingerprint(hx, hz, n, q)
                if fp in reserved_graphs:
                    continue
                reserved_graphs.add(fp)
                selected.append((coarse, fp, card, A, B, hx, hz))
            if len(selected) == 2:
                break
        if not selected:
            raise ValueError("fresh complementary full-Tanner book exhausted")
        chosen = selected

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two noisy fault intervals, NOT the six-round ASC experiment.

        Equations H*e1+m=s1 and H*(e1+e2)=s2 use a perfect closing
        syndrome. Proper opposite-check suffixes have unknown random
        orders; neither ASC scheduling nor relay/LSD is reproduced.
        Settled final residuals are classified modulo actual stabilizers.
        """
        def sites(row):
            result = []
            while row:
                bit = row & -row
                result.append(bit.bit_length()-1)
                row ^= bit
            return result

        m = len(checks)
        variables = 2*n+m
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        neighborhoods = [sites(row) for row in augmented]
        opposite = [sites(row) for row in stabilizers]
        row_width = max(map(len, neighborhoods))
        incidence = np.full((2*m, row_width), variables, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            incidence[r, :len(ns)] = ns
        flat = incidence.ravel()
        valid = flat != variables
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        ve = np.full((variables+1, int(degree.max())),
                     edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            ve[c, used[c]] = edge
            used[c] += 1

        sensed = np.asarray([
            c.bit_count() for c in _balanced_columns(checks, n)])
        spread = np.asarray([
            c.bit_count() for c in _balanced_columns(stabilizers, n)])
        levels = np.asarray((0.75, 1.00, 1.25))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+sensed+spread)*2/3
        measurement_mu = np.asarray([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(
            -2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measurement_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < data_p
        error[:, n:2*n] = channel.random((shots, n)) < data_p
        error[:, 2*n:variables] = (
            channel.random((shots, m)) < measure_p)

        for ns in opposite:
            ordering = channel.permutation(ns)
            event_p = levels*0.25*3*0.002*len(ns)*2/3
            for interval in range(2):
                for shot in np.flatnonzero(
                        channel.random(shots) < event_p):
                    cut = int(channel.integers(1, len(ns)))
                    error[shot, interval*n+ordering[cut:]] ^= True

        # This marginal allowance is deliberately not a calibrated
        # representation of the correlated circuit-level channel.
        marginal_mu = data_mu+0.25*3*0.002*spread*0.35
        marginal = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate(
            (marginal, marginal, np.clip(measure_p, 0.002, 0.25)),
            axis=1)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log((1-probabilities)/probabilities)
        target = (
            error[:, flat].reshape(shots, 2*m, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = error[:, :n] ^ error[:, n:2*n]
        residue = _balanced_quotient(stabilizers, n)
        wrong = late = work = 0

        for iteration in range(1, 37):
            count = len(prior)
            shaped = messages.reshape(count, 2*m, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            products = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(products)-np.log1p(-products))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, ve].sum(axis=2)
            posterior[:, variables] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, 2*m, row_width).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            final = hard[:, :n] ^ hard[:, n:2*n]
            for local in np.flatnonzero(settled):
                label = 0
                for c in np.flatnonzero(final[local] ^ true_final[local]):
                    label ^= residue[int(c)]
                wrong += int(bool(label))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, late, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                +(1-damping)*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (wrong+unresolved, unresolved,
                late+unresolved, work+37*unresolved)

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2], x[3]+z[3])

    seed = 321081791+1009*idx
    finalists = []
    for coarse, fp, card, A, B, hx, hz in chosen:
        # Verify every cached action against the complete evaluator lift.
        ar = [
            sum(((hx[t] >> (j*4*q)) & mask) << (j*q)
                for j in range(4)) for t in range(q)]
        br = [row & ((1 << width)-1) for row in hz[:q]]
        assert ar == component(A, "A")
        assert br == component(B, "B")
        assert brief(ar, q) is not None
        assert brief(br, q-1) is not None

        px = _c32_hook_cost(hz, hx, 544)
        pz = _c32_hook_cost(hx, hz, 544)
        if px is None or pz is None or px[0] or pz[0]:
            continue
        tx = noisy_probe(hz, hx, 544, seed, 24)
        tz = noisy_probe(hx, hz, 544, seed+1, 24)
        x = noisy_probe(hz, hx, 544, seed+1000003, 64)
        z = noisy_probe(hx, hz, 544, seed+1000004, 64)
        ambiguity = tuple(
            value for t in range(1, 4)
            for value in (max(px[t], pz[t]), px[t]+pz[t]))
        # Holdout failures lead, not BP settlement, sampled distance
        # or a possible-hook census presented as measured circuit pL.
        key = quality(x, z)+quality(tx, tz)+ambiguity+coarse+(fp,)
        finalists.append((key, A, B, hx, hz, card, fp,
                          x, z, tx, tz, px, pz))

    if not finalists:
        raise ValueError("no complementary syndrome-holdout finalist")
    finalists.sort(key=lambda entry: entry[0])
    key, A, B, hx, hz, card, fp, x, z, tx, tz, px, pz = finalists[0]
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert all(len({G._key(g) for g in cell}) == len(cell) <= CELL_MAX
               for grid in (A, B) for row in grid for cell in row)
    assert all(row.bit_count() == 12 for row in hx)
    assert set(row.bit_count() for row in hz) == {10, 11}
    degrees = _balanced_columns(hx+hz, 544)
    assert set(c.bit_count() for c in degrees[:16*q]) == {4, 5}
    assert all(c.bit_count() == 18 for c in degrees[16*q:])
    assert 6*sum(row.bit_count() for row in hx+hz) == 17280
    assert 544-gf2_rank(hx)-gf2_rank(hz) == 288
    assert lifted_k(G, A, B) == 288
    assert css_ok(hx, hz)
    return {"A": A, "B": B, "k": 288,
            "design": cards[card], "screen": key, "full_graph": fp,
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z), "hook_profile": (px, pz)}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete complementary holdout-selected cell without draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Distinct rotational binomial steps and raw syndrome-ambiguity screening eliminate the immediate parent's short-logical problem: measured codes retained abundant weight-three witnesses, 108-139 X failures per 250 shots, and fitness 3.2219.
# REFUTED: Replacing weight-three witnesses with weight-four witnesses alone improves this bed's circuit fitness: the shown odd-A triangle-free-B program scored 3.2061 versus 3.2219, with 115-131 X failures per 250 shots.
# REFUTED: A unit A column and lower-load truncated-fault optimization sufficiently suppress collective failures on the unequal D14 bed: fitness was 3.1978, measured X failures were 110-147/250, and first-pass settlement still accompanied errors on all 213 logicals.
# REFUTED: Balanced D8 two-row support cards plus truncated-fault optimization suffice for rare collective memory failures: fitness was 3.2319, measured X failures were 132-165/250, and thousands of weight-four witnesses remained with mostly diffuse block enrichment.
# REFUTED: Connected, four-cycle-free, full-rank C8:C4(action 3) components plus weighted pair-ambiguity screening suffice for rare collective memory failures: fitness was 3.4145; measured 250-shot samples had 101-127 failures, essentially every logical was affected, and both bases retained abundant weight-four witnesses.
# REFUTED: Long-order balanced C32 components and full-code two-hook ambiguity selection suffice for rare collective ASC failures: fitness was3.4852, slot0 X failed160/500 shots and slot3 X failed115/250, all288 logicals were affected, and abundant weight-four witnesses remained with diffuse block enrichment.
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
