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
CONCEPT = ("For generalized dihedral (C3xC3):C2 with simultaneous inversion, order 18, shapes (1,4)x(2,7), unit-pivot A=[1,3,3,3] and connected bipartite B selected with ordered-spread decoder probes should raise worse-basis ASC fitness above 3.3458 in cell 0.500.")
GROUP = ("Jump from C19 to (C3xC3):C2, order 18. Inversion acts on both C3 coordinates; the group has trivial centre and differs from the tested order-18 C3:C6, whose centre has order three. This exact bed is absent from BEDS. The group law is a semidirect product, but no nonabelian reliability advantage is assumed.")
SHAPES = ("Retain (1,4)x(2,7). The size factor thirty permits q<=19; q=18 gives n=540, kb=270 and guaranteed rate 0.500. Components have 72 variables/18 checks and 126 variables/36 checks. A unit column and connected incidence B are checked on complete verified matrices to give k=273, rate 0.505556. Parent failures affected all 288 logicals with diffuse 1.0-1.2x Acol/Bcol and Acol/Brow enrichment, so no persistent localized repair site is established.")
FILLING = ("Twenty initial contracts cross five invariant-distinct unit-pivot A geometries with four B voltage briefs; extra slots consume further A geometries. A has one singleton and three trinomials, replacing the parent's two binomials. B has seven distinct cross-row matchings, row totals 7/7 and column totals two. HX weight is twelve, HZ at most ten, and six-round extraction uses 17280 CNOTs versus 18240. Verified lifts supply component actions, ranks, exact short-logical classification and stabilizer quotients. Rooted component refinement conservatively removes equivalent proposals; B finalist classes are reserved across the four briefs. Final selection uses independent BP holdouts with degree-weighted data noise and ordered opposite-check suffix faults, followed by truncated logical-class ambiguity. The suffix ordering is a legal greedy one-basis schedule surrogate, not the measured ASC order; measurement histories and relay/LSD are omitted. Target the parent's 106-148 failed Z shots/250 and collective logical flips, not distance alone.")
FAMILY_NAME = "r050_DihC3sq18_14x27_unitA_orderedspread_bp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 18

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dih(C3xC3) requires order 18")
    # (v,t)(w,u) = (v + (-1)^t*w, t+u).
    # The involution inverts BOTH independent C3 generators.
    els = [(a, b, t) for t in range(2)
           for b in range(3) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3, int(g[2]) % 2)

    def mul(g, h):
        a, b, t = red(g)
        c, d, u = red(h)
        sign = 1 if t == 0 else -1
        return ((a + sign*c) % 3, (b + sign*d) % 3, (t+u) % 2)

    def inv(g):
        a, b, t = red(g)
        sign = 1 if t == 0 else -1
        return ((-sign*a) % 3, (-sign*b) % 3, t)

    def idx(g):
        a, b, t = red(g)
        return a + 3*b + 9*t

    return dict(elements=els, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


# Five B architectures: column totals 3,3,3,3,2,2,2;
# both rows total nine. Zero cells are deliberate, never repairs.
_B_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,1),(1,1),(1,1)),
    ((3,0),(1,2),(1,2),(1,2),(1,1),(1,1),(1,1)),
    ((3,0),(0,3),(2,1),(1,2),(1,1),(1,1),(1,1)),
    ((2,1),(2,1),(1,2),(1,2),(2,0),(0,2),(1,1)),
    ((3,0),(1,2),(1,2),(1,2),(2,0),(0,2),(1,1)),
)


def _kernel56(rows, ncols):
    """Exact weight-five/six counts when kernel weights <=4 are absent.

    Pair/triple syndrome matches count each weight-five word ten times.
    Distinct triple collisions count each weight-six word ten times.
    Smaller-word exclusion makes these colliding supports disjoint.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            s = x ^ y
            pairs[s] = pairs.get(s, 0) + 1
    triples = {}
    five = six = 0
    for i in range(ncols):
        for j in range(i):
            s = columns[i] ^ columns[j]
            for h in range(j):
                value = s ^ columns[h]
                five += pairs.get(value, 0)
                six += triples.get(value, 0)
                triples[value] = triples.get(value, 0) + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def _retired_C19_CTX(rng, G, idx):
    """Inactive parent-group filling, retained only for provenance."""
    import math
    from itertools import combinations

    q = G.q
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)

    def direction(d):
        d %= q
        return min(d, q - d)

    def directions(cell):
        return {direction(a - b) for a, b in combinations(cell, 2)}

    def normalized(cell, multiplier=1):
        values = [(multiplier * a) % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def geometry_key(cells):
        # Quotient cyclic multipliers, independent cell origins and
        # column permutations. Arbitrary Tanner equivalence remains
        # the evaluator's responsibility.
        return min(tuple(sorted(normalized(cell, u) for cell in cells))
                   for u in range(1, q))

    def component(grid, side):
        # Extract only from a verified zero-partner construction.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    # Build the same deterministic geometry catalogue in each pure call.
    # A contract fixes relative supports; it is not a random cell-origin
    # draw or an automorphism image of a previously submitted filling.
    trinomials = sorted({
        normalized((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if len(directions((0, a, b))) == 3
    })
    all_directions = set(range(1, (q + 1) // 2))
    geometries = set()
    for first, second in combinations(trinomials, 2):
        df, ds = directions(first), directions(second)
        if df & ds:
            continue
        available = sorted(all_directions - df - ds)
        for a, b in combinations(available, 2):
            geometries.add(geometry_key(
                ((0, a), (0, b), first, second)))

    book = []
    for signature in sorted(geometries):
        grid = [[[(a,) for a in cell] for cell in signature]]
        rows, width = component(grid, "A")
        stats = _kernel_statistics(rows, width)
        if stats[0] or stats[1] or gf2_rank(rows) != q:
            continue
        key = (stats[2], _six_cycles(rows), stats[3], signature)
        book.append((key, grid))
    book.sort(key=lambda entry: entry[0])
    assert len(book) >= 5
    wanted = 5 * batch + marker
    if wanted >= len(book):
        raise ValueError("No unused canonical A geometry for this batch")
    akey, A = book[wanted]
    bin_directions = {
        next(iter(directions(tuple(g[0] for g in cell))))
        for cell in A[0] if len(cell) == 2
    }

    # Each column is a graph edge: internal cyclic edges or a matching
    # between the two check-row orbits. Stronger non-bipartite briefs
    # come first; all four architectures have fourteen terms.
    profiles = (
        ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
        ((2,0),(2,0),(0,2),(1,1),(1,1),(1,1),(1,1)),
        ((2,0),(0,2),(1,1),(1,1),(1,1),(1,1),(1,1)),
        ((2,0),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    )
    profile = profiles[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2, 2, 3, 3]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    cross = [j for j, p in enumerate(profile) if p == (1, 1)]
    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    mutable = internal + [(1, j) for j in cross]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def gauge(grid):
        # Remove the common cross-row voltage origin, not a new draw.
        origin = grid[1][cross[0]][0][0]
        for j in cross:
            grid[1][j] = [((grid[1][j][0][0] - origin) % q,)]
        return grid

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for i, j in internal:
            grid[i][j] = [(0,), (rng.randrange(1, (q + 1) // 2),)]
        ratios = rng.sample(range(q), len(cross))
        for j, ratio in zip(cross, ratios):
            grid[0][j], grid[1][j] = [(0,)], [(ratio,)]
        return gauge(grid)

    def assess_B(grid):
        rows, width = component(grid, "B")
        stats = _kernel_statistics(rows, width)
        # Duplicate edges, triangles and four-cycles are measured from
        # the actual component incidence columns.
        energy = (300 * stats[0] + 50 * stats[1] + stats[2]) / q
        return energy, stats, rows

    pool, seen = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or gf2_rank(rows) != 2 * q - 1:
            return
        signature = tuple(tuple(g[0] for g in cell)
                          for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        coupling = sum(
            direction(grid[i][j][1][0] - grid[i][j][0][0])
            in bin_directions for i, j in internal)
        key = (stats[2], coupling)
        serial += 1
        pool.append((key, serial, clone(grid)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    for restart in range(4):
        grid = draw_B()
        energy, stats, rows = assess_B(grid)
        remember(grid, stats, rows)
        for step in range(280):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [
                    (0,), (rng.randrange(1, (q + 1) // 2),)]
            else:
                proposal[i][j] = [(rng.randrange(q),)]
            gauge(proposal)
            proposed, ps, pr = assess_B(proposal)
            delta = proposed - energy
            temperature = 4.0 * (0.985 ** step) + 0.06
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("No connected triangle-free B graph in contract")

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def quotient_columns(stabilizers, n):
        # A linear canonical remainder modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residual = [0] * n
        for c in range(n):
            row = pivots.get(c)
            if row is None:
                residual[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def logical_risk(checks, stabilizers, n):
        """Exact short-logical classification, with circuit-risk surrogates.

        Shared opposite-basis checks mark possible ancilla-spread pairs;
        they do not certify a particular ASC hook ordering.
        """
        cs = columns(checks, n)
        spread = columns(stabilizers, n)
        if 0 in cs or len(set(cs)) != n:
            return None
        residual = quotient_columns(stabilizers, n)
        singles = dict(zip(cs, residual))
        exposure = [s.bit_count() + h.bit_count()
                    for s, h in zip(cs, spread)]
        buckets = {}
        hooks = weighted = double_hooks = partitions = 0

        for i, si in enumerate(cs):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                syndrome = si ^ cs[j]
                label = ri ^ residual[j]
                third = singles.get(syndrome)
                if third is not None and third != label:
                    return None
                h = (hi & spread[j]).bit_count()
                w = exposure[i] * exposure[j]
                bucket = buckets.get(syndrome)
                if bucket is None:
                    buckets[syndrome] = [1, h, w, {label: [1, h, w]}]
                    continue
                same = bucket[3].get(label)
                sc, sh, sw = same if same is not None else (0, 0, 0)
                dc, dh, dw = bucket[0] - sc, bucket[1] - sh, bucket[2] - sw
                # Equal syndrome and unequal quotient labels give a
                # nontrivial quartet. Distinct syndrome columns make
                # colliding different pairs disjoint.
                partitions += dc
                hooks += h * dc + dh
                weighted += w * dc + dw
                double_hooks += h * dh
                bucket[0] += 1
                bucket[1] += h
                bucket[2] += w
                if same is None:
                    bucket[3][label] = [1, h, w]
                else:
                    same[0] += 1
                    same[1] += h
                    same[2] += w
        assert partitions % 3 == 0
        return hooks, weighted, double_hooks, partitions // 3

    winner = None
    for coarse, serial, B in pool:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 288:
            continue
        pz = logical_risk(hx, hz, n)
        px = logical_risk(hz, hx, n)
        if pz is None or px is None:
            continue
        score = tuple(value for t in range(4)
                      for value in (max(px[t], pz[t]), px[t] + pz[t]))
        score += coarse
        if winner is None or score < winner[0]:
            winner = (score, B, hx, hz, px, pz)

    if winner is None:
        raise ValueError("No k=288 filling passed full-CSS short-logical screen")
    score, B, hx, hz, px, pz = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 18240
    return {"A": A, "B": B, "design": (architecture, wanted),
            "k": 288, "logical_risk": (px, pz),
            "component_contract": akey, "cycle_score": score}


def _d18_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length()-1] |= tag
            row ^= bit
    return out


def _d18_quotient(rows, width):
    """Linear canonical remainders modulo verified stabilizer rows."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            previous = pivots.get(p)
            if previous is None:
                pivots[p] = row
                break
            row ^= previous
    out = [0] * width
    for c in range(width):
        row = pivots.get(c)
        if row is None:
            out[c] = 1 << c
        else:
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= out[bit.bit_length()-1]
                tail ^= bit
            out[c] = value
    return out


def _d18_fingerprint(rows, width, roots):
    """Rooted component invariant, not full-code canonization.

    A component's verified regular action is transitive within each
    check-row orbit. Minimize over the specified orbit representatives.
    Equal fingerprints are conservatively discarded.
    """
    import hashlib
    from collections import Counter

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, value in enumerate(_d18_columns(rows, width)):
        while value:
            bit = value & -value
            r = bit.bit_length()-1
            adjacency[r].append(nr+c)
            adjacency[nr+c].append(r)
            value ^= bit
    images = []
    for root in roots:
        colors = [0]*nr + [1]*width
        colors[root] = 2
        digest = hashlib.sha256()
        for _ in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(nr + width)
            ]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        images.append(digest.hexdigest())
    return min(images)


def _d18_connected(rows, width):
    adjacency = [0] * len(rows)
    for value in _d18_columns(rows, width):
        sites, bits = [], value
        while bits:
            bit = bits & -bits
            sites.append(bit.bit_length()-1)
            bits ^= bit
        for r in sites:
            adjacency[r] |= value
    reached = frontier = 1
    while frontier:
        following = 0
        while frontier:
            bit = frontier & -frontier
            following |= adjacency[bit.bit_length()-1]
            frontier ^= bit
        frontier = following & ~reached
        reached |= frontier
    return reached == (1 << len(rows))-1


def _d18_spread_order(stabilizers, width):
    """Greedy legal one-basis interaction order, NOT the ASC schedule.

    Edge colors prevent simultaneous use of a data qubit or ancilla.
    Opposite-basis scheduling constraints are intentionally not modeled.
    """
    source = []
    for row in stabilizers:
        sites = []
        while row:
            bit = row & -row
            sites.append(bit.bit_length()-1)
            row ^= bit
        source.append(sites)
    schedule_rng = random.Random(18180427 + len(stabilizers))
    row_order = list(range(len(source)))
    schedule_rng.shuffle(row_order)
    data_used = [0] * width
    ordered = [None] * len(source)
    for r in row_order:
        sites = list(source[r])
        schedule_rng.shuffle(sites)
        ancilla_used, edges = 0, []
        for c in sites:
            occupied = ancilla_used | data_used[c]
            color = 0
            while (occupied >> color) & 1:
                color += 1
            tag = 1 << color
            ancilla_used |= tag
            data_used[c] |= tag
            edges.append((color, c))
        ordered[r] = [c for color, c in sorted(edges)]
    return ordered


def _d18_fault_profile(checks, stabilizers, width, ordered):
    """Exact <=3 logical rejection plus a truncated ambiguity surrogate.

    Singles, pairs and ordered suffixes are grouped by actual syndrome
    and stabilizer quotient. Suffix masses are comparison weights, not
    calibrated ASC fault probabilities.
    """
    syndrome = _d18_columns(checks, width)
    spread = _d18_columns(stabilizers, width)
    residue = _d18_quotient(stabilizers, width)
    if 0 in syndrome or len(set(syndrome)) != width:
        return None
    singles = dict(zip(syndrome, residue))
    exposure = [1 + a.bit_count() + b.bit_count()
                for a, b in zip(syndrome, spread)]
    buckets = {0: {0: 100000000}}

    def add(s, label, mass):
        classes = buckets.setdefault(s, {})
        classes[label] = classes.get(label, 0)+mass

    for c, s in enumerate(syndrome):
        add(s, residue[c], 1024*exposure[c])
    for i, si in enumerate(syndrome):
        for j in range(i):
            s = si ^ syndrome[j]
            label = residue[i] ^ residue[j]
            other = singles.get(s)
            if other is not None and other != label:
                return None
            add(s, label, 16*exposure[i]*exposure[j])

    # Each suffix is a possible propagated ancilla fault for this
    # explicitly stated surrogate ordering. Full checks are stabilizers.
    for sites in ordered:
        s = label = 0
        for position, c in enumerate(reversed(sites)):
            s ^= syndrome[c]
            label ^= residue[c]
            if position+1 < len(sites):
                add(s, label, 256)

    loss = peak = ambiguous = 0
    spectrum = {}
    for classes in buckets.values():
        if len(classes) > 1:
            error = sum(classes.values())-max(classes.values())
            loss += error
            peak = max(peak, error)
            ambiguous += 1
            spectrum[error] = spectrum.get(error, 0)+1
    return loss, peak, ambiguous, tuple(sorted(spectrum.items()))


def _d18_bp_probe(checks, stabilizers, width, ordered, seed, shots=72):
    """Noisy-data and ordered-spread BP surrogate, not circuit pL.

    Eighteen shots isolate one suffix fault. Remaining shots mix
    degree-weighted data noise and occasional suffix faults.
    Measurement histories, actual ASC ordering, relay BP and LSD
    are absent. Settled residuals are classified modulo stabilizers.
    """
    import numpy as np

    neighbors = []
    incident = [[] for _ in range(width)]
    edge_variables = []
    for row in checks:
        sites = []
        while row:
            bit = row & -row
            sites.append(bit.bit_length()-1)
            row ^= bit
        neighbors.append(sites)
        for c in sites:
            edge = len(edge_variables)
            edge_variables.append(c)
            incident[c].append(edge)

    ne = len(edge_variables)
    nr = len(neighbors)
    max_check = max(map(len, neighbors))
    max_variable = max(map(len, incident))
    check_edges = np.full((nr, max_check), ne, dtype=np.int32)
    check_sites = np.zeros((nr, max_check), dtype=np.int32)
    valid = np.zeros((nr, max_check), dtype=bool)
    cursor = 0
    for r, sites in enumerate(neighbors):
        count = len(sites)
        check_edges[r, :count] = np.arange(cursor, cursor+count)
        check_sites[r, :count] = sites
        valid[r, :count] = True
        cursor += count
    variable_edges = np.full(
        (width, max_variable), ne, dtype=np.int32)
    for c, edges in enumerate(incident):
        variable_edges[c, :len(edges)] = edges
    edge_variables = np.asarray(edge_variables, dtype=np.int32)

    sensed = np.asarray([len(edges) for edges in incident], dtype=float)
    spread = np.asarray([
        v.bit_count() for v in _d18_columns(stabilizers, width)
    ], dtype=float)
    levels = np.asarray((0.012, 0.025, 0.040))
    rates = np.minimum(
        0.22, levels[np.arange(shots) % 3, None]
        * (sensed+spread)[None, :]/5.0)
    channel = np.random.default_rng(seed)
    errors = channel.random((shots, width)) < rates
    isolated = min(18, shots)
    errors[:isolated] = False

    for shot in range(isolated):
        sites = ordered[int(channel.integers(len(ordered)))]
        cut = int(channel.integers(1, len(sites)))
        errors[shot, sites[cut:]] ^= True
    triggers = channel.random(
        (shots-isolated, len(ordered))) < 0.004
    for local, r in zip(*np.nonzero(triggers)):
        sites = ordered[int(r)]
        cut = int(channel.integers(1, len(sites)))
        errors[int(local)+isolated, sites[cut:]] ^= True

    llr = np.log((1.0-rates)/rates)
    target = np.logical_xor.reduce(
        errors[:, check_sites] & valid[None, :, :], axis=2)
    target_sign = np.where(target, -1.0, 1.0)
    messages = llr[:, edge_variables].copy()
    outgoing = np.zeros((shots, ne+1), dtype=float)
    done = np.zeros(shots, dtype=bool)
    decoded = np.zeros((shots, width), dtype=bool)
    completion = np.full(shots, 49, dtype=np.int32)
    positions = np.arange(max_check)[None, None, :]

    for iteration in range(48):
        padded = np.concatenate(
            (messages, np.zeros((shots, 1))), axis=1)
        shaped = padded[:, check_edges]
        magnitude = np.where(
            valid[None, :, :], np.abs(shaped), np.inf)
        signs = np.where(shaped < 0, -1.0, 1.0)
        product = signs.prod(axis=2)*target_sign
        minimum_at = magnitude.argmin(axis=2)
        smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
        size = np.where(
            positions == minimum_at[:, :, None],
            smallest[:, :, 1, None], smallest[:, :, 0, None])
        update = 0.80*product[:, :, None]*signs*size
        update = np.where(valid[None, :, :], update, 0.0)
        outgoing[:, :ne] = update[:, valid]
        posterior = llr + outgoing[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        parity = np.logical_xor.reduce(
            hard[:, check_sites] & valid[None, :, :], axis=2)
        settled = ~np.any(parity != target, axis=1)
        newly = settled & ~done
        decoded[newly] = hard[newly]
        completion[newly] = iteration+1
        done |= settled
        if np.all(done):
            break
        proposal = posterior[:, edge_variables]-outgoing[:, :ne]
        messages = (
            0.30*messages + 0.70*np.clip(proposal, -24.0, 24.0))

    residue = _d18_quotient(stabilizers, width)
    failed = ~done
    for shot in np.flatnonzero(done):
        label = 0
        for c in np.flatnonzero(decoded[shot] ^ errors[shot]):
            label ^= residue[int(c)]
        if label:
            failed[shot] = True
    return (int(failed.sum()), int(failed[:isolated].sum()),
            int((~done).sum()), int(completion.sum()))


def NEW_CTX(rng, G, idx):
    """Unit-pivot class contracts and independent ordered-spread holdouts."""
    from itertools import combinations

    q = G.q
    assert q == 18
    assert A_SHAPE == (1, 4) and B_SHAPE == (2, 7)
    wanted, brief = divmod(idx, 4)
    block_mask = (1 << q)-1

    # Cache ONLY singleton actions returned by the verified helper.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz)

    def component(grid, side):
        rows = [0] * (len(grid)*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        rows[i*q+t] ^= bits << (j*q)
        return rows, len(grid[0])*q

    rotations = [g for g in G.elements if g[2] == 0 and g != G.e]
    reflections = [g for g in G.elements if g[2] == 1]
    # Independent rotation pairs avoid subgroup-sum triples whose
    # regular columns repeat. Mixed triples have relative orders 3,2,2.
    rotation_triples = [
        [G.e, a, b] for a, b in combinations(rotations, 2)
        if b != G.inv(a)
    ]
    mixed_triples = [[G.e, a, b]
                     for a in rotations for b in reflections]

    catalogue_rng = random.Random(1818042701)
    bank, raw_seen, class_seen = [], set(), set()
    target_size = max(28, wanted+8)
    for trial in range(4096):
        number_mixed = 1 + trial % 3
        cells = catalogue_rng.sample(mixed_triples, number_mixed)
        cells += catalogue_rng.sample(
            rotation_triples, 3-number_mixed)
        raw = tuple(sorted(
            tuple(sorted(G.idx(g) for g in cell)) for cell in cells))
        if raw in raw_seen:
            continue
        raw_seen.add(raw)
        A = [[[G.e]] + [[G.elements[t] for t in cell] for cell in raw]]
        ar, awidth = component(A, "A")
        stats = _kernel_statistics(ar, awidth)
        if any(stats[:2]) or gf2_rank(ar) != q:
            continue
        if not _d18_connected(ar, awidth):
            continue
        identity = _d18_fingerprint(ar, awidth, (0,))
        if identity in class_seen:
            continue
        class_seen.add(identity)
        key = (stats[2], _four_cycles(ar), stats[3],
               _six_cycles(ar), identity)
        bank.append((key, identity, A, ar))
        if len(bank) >= target_size:
            break
    bank.sort(key=lambda item: item[0])
    if wanted >= len(bank):
        raise ValueError("Invariant-distinct unit-pivot A book exhausted")
    akey, aid, A, ar = bank[wanted]

    sa = [[1, 3, 3, 3]]
    sb = [[1]*7, [1]*7]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert sum(sa[0])+max(map(sum, zip(*sb))) == 12
    assert max(map(sum, sb))+max(sa[0]) == 10

    def graph_key(grid):
        br, width = component(grid, "B")
        if gf2_rank(br) != 2*q-1:
            return None
        # B is a simple bipartite incidence matrix: common-neighbor
        # partitions count component weight-four supports exactly.
        adjacency = [0] * (2*q)
        for value in _d18_columns(br, width):
            assert value.bit_count() == 2
            bit = value & -value
            a = bit.bit_length()-1
            b = (value ^ bit).bit_length()-1
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a
        partitions = sum(
            overlap*(overlap-1)//2
            for i, neighbors in enumerate(adjacency)
            for other in adjacency[:i]
            for overlap in [(neighbors & other).bit_count()])
        assert partitions % 2 == 0
        return partitions//2, br

    # Each pure call reconstructs the same preceding B reservations
    # for this A geometry. Counts 2/5 and 3/4 can be gauge-related, so
    # they are NOT accepted as distinct without component refinement.
    reserved = set()
    chosen = None
    for model in range(brief+1):
        stream = random.Random(
            1818042711 + 1009*wanted + 104729*model)
        rotation_count = (2, 3, 4, 5)[model]
        proposals, seen_raw = [], set()
        for trial in range(512):
            voltages = (
                stream.sample(rotations+[G.e], rotation_count)
                + stream.sample(reflections, 7-rotation_count))
            signature = tuple(sorted(G.idx(g) for g in voltages))
            if signature in seen_raw:
                continue
            seen_raw.add(signature)
            B = [[[G.e] for _ in range(7)],
                 [[G.elements[t]] for t in signature]]
            result = graph_key(B)
            if result is None:
                continue
            quartets, br = result
            proposals.append((quartets, signature, B, br))
        proposals.sort(key=lambda item: item[:2])
        classes, options = set(), []
        for quartets, signature, B, br in proposals:
            identity = _d18_fingerprint(br, 7*q, (0, q))
            if identity in classes or identity in reserved:
                continue
            classes.add(identity)
            options.append((quartets, identity, B, br))
            if len(options) == 3:
                break
        if not options:
            raise ValueError("Unused connected B classes exhausted")
        reserved.update(option[1] for option in options)
        chosen = options

    options = []
    for quartets, bid, B, br in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 540 or k != 273:
            continue
        sx = _d18_spread_order(hx, n)
        sz = _d18_spread_order(hz, n)
        px = _d18_fault_profile(hz, hx, n, sx)
        pz = _d18_fault_profile(hx, hz, n, sz)
        if px is None or pz is None:
            continue
        structural = (
            max(px[0], pz[0]), px[0]+pz[0],
            max(px[1], pz[1]), px[1]+pz[1],
            px[2]+pz[2], quartets, bid)
        options.append((structural, B, br, hx, hz, sx, sz, px, pz))
    if not options:
        raise ValueError("No k=273 candidate passed short-logical classification")

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2],
                x[3]+z[3])

    seed = 1818042751 + 1009*idx
    training = []
    for number, option in enumerate(options):
        structural, B, br, hx, hz, sx, sz, px, pz = option
        x = _d18_bp_probe(hz, hx, 540, sx, seed)
        z = _d18_bp_probe(hx, hz, 540, sz, seed+1)
        training.append((
            quality(x, z)+structural, number, option, x, z))
    training.sort(key=lambda item: item[:2])

    finalists = []
    for train_key, number, option, tx, tz in training[:2]:
        structural, B, br, hx, hz, sx, sz, px, pz = option
        x = _d18_bp_probe(hz, hx, 540, sx, seed+1000003)
        z = _d18_bp_probe(hx, hz, 540, sz, seed+1000004)
        score = quality(x, z)+quality(tx, tz)+structural
        finalists.append((score, number, option, x, z))
    finalists.sort(key=lambda item: item[:2])
    score, number, option, x, z = finalists[0]
    structural, B, br, hx, hz, sx, sz, px, pz = option

    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == ar and actual_b == br
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 273
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) == 10
    assert 6*sum(row.bit_count() for row in hx+hz) == 17280
    assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": 273,
            "design": (wanted, brief),
            "component_classes": (aid, structural[-1]),
            "A_contract": akey, "fault_profile": (px, pz),
            "ordered_spread_holdout": (x, z),
            "selection_score": score, "cnots": 17280}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete verified unit-pivot/matching cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Minimizing short-kernel counts on the q8 (2,7)x(2,8) bed suppresses collective circuit failures: measured Z failures were 117..147/250 and affected all 240 logicals, despite mostly first-pass settlement.
# REFUTED: Asymmetric A<=3/B<=4 component exclusion balances C19 memory: all five measured codes retained abundant weight-four Z-type logicals, failed 153..171/250 X shots, and scored only 3.1933 despite much lower Z failed-shot rates.
# REFUTED: Difference-separated Sidon A, triangle-free graphic B and shared-check quartet selection suffice for rare C19 circuit failures: fitness was 3.3458, measured Z failures were 106..148/250, and failures collectively affected all 288 logicals despite mostly first-pass settlement.
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
