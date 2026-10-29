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
CONCEPT = ("C7 semidirect_2 C6, order 42, A=(1,2), B=(2,5): cubic A constituents paired with long-girth reinforced B constituents should supply five measured distance-ten readings in cell 0.250; constituent screens do not certify quantum distance ten.")
GROUP = ("Jump from Q32 to x^7=y^6=1, y*x*y^-1=x^2. Since 2 has order three modulo seven, the presentation is valid. Its centre has order two and useful relative orders are 6,7,14, unlike the centre-one order-42 bed recorded at fitness 16.7. This exact group/shape bed is absent from BEDS. The related order-48 reinforced bed reached fitness 20.9252; transfer of its distance performance is a hypothesis.")
SHAPES = ("A=(1,2), B=(2,5) gives n=504, kb=126 and guaranteed rate 0.250. Components have respectively 84 variables/42 checks and 210 variables/84 checks; the size window permits q<=49 and this group fixes q=42. Parent weight-three minima occupied sector-1 Acol/Bcol blocks, often the unit columns. Remove that unit-pivot mechanism and require the actual full-code rate to remain in [0.225,0.275).")
FILLING = ("A has two trinomials. B uses either crossed binomial/monomial cubic columns or isolated trinomials around three matching columns. Every component row totals six, every column totals at most three, and BOTH CSS check maxima are nine, also below twelve. Enumerate verified matching cores and optimize twenty triangle/reinforcement prescriptions. Exact constituent screens exclude weights through six when possible; certified floor-six fallbacks remain explicit. Actual kernel and non-stabilizer logical witnesses guide selection, never certify larger distance. A deterministic immutable portfolio rejects equal intrinsic full-Tanner signatures and verifies full ranks, CSS and absence of full kernel supports through four. Unseen archive novelty and distance-ten survival remain evaluator-tested hypotheses.")
FAMILY_NAME = "r025_c7sd_c6_action2_q42_12x25_reinforced_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect_2 C6 has order 42")
    # (a,b)=x^a y^b; x^7=y^6=1, y*x*y^-1=x^2.
    # 2^3=1 mod 7, so the action factors through C3.
    powers = (1, 2, 4, 1, 2, 4)
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 7, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _generator_cards(G):
    """Enumerate inverse-normalized generating triples, not label draws."""
    from itertools import combinations

    # One representative of each inverse pair, excluding e and x^8.
    representatives = [(a, 0) for a in range(1, 8)]
    representatives += [(a, 1) for a in range(8)]

    def order(g):
        h, count = G.e, 0
        while True:
            h = G.mul(h, g)
            count += 1
            if h == G.e:
                return count

    cards = []
    for triple in combinations(representatives, 3):
        reached, frontier = {G.e}, [G.e]
        while frontier:
            h = frontier.pop()
            for g in triple:
                t = G.mul(h, g)
                if t not in reached:
                    reached.add(t)
                    frontier.append(t)
        if len(reached) == G.q:
            cards.append(triple)
    cards.sort(key=lambda t: (tuple(sorted(order(g) for g in t)),
                              sum(g[1] for g in t), t))
    return cards


def _unit_binomial_cells(G, triple):
    return [[[G.e]] + [[G.e, g] for g in triple]]


def _tanner_signature(hx, hz):
    """An uncoloured Tanner-isomorphism invariant, not an isomorphism test.

    Check intersections define two adjacency relations. Each check's
    profile counts common neighbours in both relations for every other
    check. Different signatures certify non-isomorphic Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                adj = one if c == 1 else two
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            kind = ((one[i] >> j) & 1) + 2 * ((two[i] >> j) & 1)
            key = (kind,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & two[j]).bit_count()
                   + (two[i] & one[j]).bit_count(),
                   (two[i] & two[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(tuple(sorted(p.items())) for p in profiles))


def _check_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _no_weight_two_kernel(rows, n):
    """Exact exclusion of every nonzero kernel vector of weight <=2."""
    columns = _check_columns(rows, n)
    seen = set(columns)
    return 0 not in seen and len(seen) == n


def _degree_signature(hx, hz, n):
    """Additional intrinsic Tanner invariants; no X/Z colouring assumed.

    Project only degree-three qubits onto checks. Quibit degree is
    invariant under any Tanner relabeling, so different signatures
    certify non-isomorphic full graphs.
    """
    rows = hx + hz
    columns = _check_columns(rows, n)
    degree_counts = {}
    adj = [0] * len(rows)
    for c in columns:
        degree = c.bit_count()
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        if degree == 3:
            bits = c
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= c ^ bit
                bits ^= bit

    profiles = []
    for i, neighbors in enumerate(adj):
        histogram = {}
        for j, other in enumerate(adj):
            if i == j:
                continue
            key = (((neighbors >> j) & 1),
                   other.bit_count(),
                   (neighbors & other).bit_count(),
                   rows[j].bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        profiles.append((rows[i].bit_count(), neighbors.bit_count(),
                         tuple(sorted(histogram.items()))))
    return (tuple(sorted(degree_counts.items())), tuple(sorted(profiles)))


def _verify_three_witness(G, B, hx, hz):
    """Verify a sector-1 X logical of weight three in the actual lift."""
    q = G.q
    g = B[0][1][1]
    # B=[1, 1+g, ...]. Its right-lifted binomial column at h=e
    # is corrected by the two unit-column qubits at e and g^-1.
    word = ((1 << (q + G.idx(G.e)))
            ^ (1 << G.idx(G.e))
            ^ (1 << G.idx(G.inv(g))))
    if word.bit_count() != 3:
        return False
    if any((word & row).bit_count() % 2 for row in hz):
        return False

    pivots = {}
    for row in hx:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _retired_q32_context(rng, G, idx):
    """Uncalled historical Q32 catalogue; never publishes new cells."""
    from itertools import combinations

    assert G.q == 32
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    # Enumerate all inverse-class triples, excluding the identity and
    # unique involution. The first component is proper, the second full:
    # neither the generating/generating nor common-subgroup archive rule.
    directions = (tuple((a, 0) for a in range(1, 8))
                  + tuple((a, 1) for a in range(8)))
    cards = list(combinations(directions, 3))
    generating = set(_generator_cards(G))
    proper_ids = [i for i, triple in enumerate(cards)
                  if triple not in generating]
    full_ids = [i for i, triple in enumerate(cards)
                if triple in generating]
    plans = [(a, b) for a in proper_ids for b in full_ids]
    if not proper_ids or not full_ids:
        raise ValueError("mixed component prescriptions unexpectedly absent")

    # All Q32 automorphisms: x -> x^u, y -> x^v*y, u odd.
    # Inverting a binomial direction and permuting columns are gauges.
    actions = [(u, v) for u in range(1, 16, 2) for v in range(16)]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    images = []
    for triple in cards:
        images.append([
            tuple(sorted(inverse_class(
                ((u * a + v * b) % 16, b))
                for a, b in triple))
            for u, v in actions
        ])

    def orbit_key(a, b):
        return min(tuple(sorted((x, y)))
                   for x, y in zip(images[a], images[b]))

    def coset_signature(hx, hz, n):
        """Intrinsic invariants of the actual uncoloured Tanner graph.

        Check projection preserves full-graph connectivity because every
        qubit has positive degree. Different signatures certify
        non-isomorphism; equal signatures are discarded conservatively.
        """
        rows = list(hx) + list(hz)
        m = len(rows)
        columns = _check_columns(rows, n)
        relations = {}
        adjacency = [0] * m

        # These projections use intrinsic full-graph qubit degrees,
        # never protograph labels or an assumed X/Z check colouring.
        # A Tanner isomorphism preserves every projection separately.
        degree_relations = {}
        for column in columns:
            degree = column.bit_count()
            if not degree:
                raise ValueError("isolated qubit in certified catalogue")
            relation = degree_relations.setdefault(degree, [0] * m)
            bits = column
            while bits:
                bit = bits & -bits
                relation[bit.bit_length() - 1] |= column ^ bit
                bits ^= bit

        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if not overlap:
                    continue
                if overlap not in relations:
                    relations[overlap] = [0] * m
                relation = relations[overlap]
                relation[i] |= 1 << j
                relation[j] |= 1 << i
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

        def neighbors(bits, relation):
            out = 0
            while bits:
                bit = bits & -bits
                out |= relation[bit.bit_length() - 1]
                bits ^= bit
            return out

        components = []
        remaining = (1 << m) - 1
        while remaining:
            visited = frontier = remaining & -remaining
            while frontier:
                frontier = neighbors(frontier, adjacency) & ~visited
                visited |= frontier
            components.append(visited)
            remaining &= ~visited

        # Do not filter connectivity here: this same invariant must
        # also describe disconnected archive references faithfully.
        # Candidate connectivity is checked after signature extraction.
        component_profiles = []
        for mask in components:
            degrees = {}
            for column in columns:
                if column & mask:
                    degree = column.bit_count()
                    degrees[degree] = degrees.get(degree, 0) + 1
            component_profiles.append(
                (mask.bit_count(), tuple(sorted(degrees.items()))))

        def spheres(root, relation):
            visited = frontier = 1 << root
            counts = []
            while frontier:
                frontier = neighbors(frontier, relation) & ~visited
                if frontier:
                    counts.append(frontier.bit_count())
                    visited |= frontier
            return tuple(counts)

        kinds = sorted(relations)
        root_profiles = []
        for root in range(m):
            root_profiles.append((
                rows[root].bit_count(),
                spheres(root, adjacency),
                tuple((kind, spheres(root, relations[kind]))
                      for kind in kinds),
                tuple((degree, spheres(root, degree_relations[degree]))
                      for degree in sorted(degree_relations)),
            ))

        return (
            tuple(sorted(component_profiles)),
            tuple(sorted(root_profiles)),
            _tanner_signature(hx, hz),
            _degree_signature(hx, hz, n),
        )

    # Exclusion references only: twelve generating/generating codes
    # from the direct-parent table, then twelve common-subgroup codes
    # from the current table. Each packet contains three A directions
    # followed by three B directions in the frozen element ordering.
    archived = (
        (4,18,19, 16,17,21),
        (20,21,22, 4,5,20),
        (4,5,17, 7,16,18),
        (3,17,21, 1,3,22),
        (7,17,18, 19,20,21),
        (2,7,16, 1,20,23),
        (1,7,21, 1,6,17),
        (1,16,21, 1,16,21),
        (7,20,22, 7,20,22),
        (18,19,23, 1,19,23),
        (4,22,23, 1,3,20),
        (1,17,23, 3,22,23),
        (2,4,6, 3,4,7),
        (3,5,6, 5,6,7),
        (2,6,22, 2,16,20),
        (4,18,20, 6,16,18),
        (1,2,5, 3,4,7),
        (4,20,22, 4,20,22),
        (2,6,16, 2,6,22),
        (2,4,22, 6,16,22),
        (1,2,5, 1,2,5),
        (2,16,20, 2,18,22),
        (2,6,18, 6,16,18),
        (4,18,22, 6,16,20),
    )
    graph_keys = set()
    for packet in archived:
        A = _unit_binomial_cells(
            G, tuple(G.elements[t] for t in packet[:3]))
        B = _unit_binomial_cells(
            G, tuple(G.elements[t] for t in packet[3:]))
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(coset_signature(hx, hz, n))

    # Shuffle an exhaustive list of mixed structural prescriptions,
    # not independently sampled completed codes. Every slot rebuilds
    # the same prefix with the same archive and within-program exclusions.
    local = random.Random(BASE_SEED ^ 0xC032A3)
    plans = sorted(plans)
    local.shuffle(plans)
    orbit_keys = set()
    accepted = 0

    for ia, ib in plans:
        orbit = orbit_key(ia, ib)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A = _unit_binomial_cells(G, cards[ia])
        B = _unit_binomial_cells(G, cards[ib])
        hx, hz, n = lifted_matrices(G, A, B)
        if max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            continue
        if not (_no_weight_two_kernel(hx, n)
                and _no_weight_two_kernel(hz, n)):
            continue

        signature = coset_signature(hx, hz, n)
        if len(signature[0]) != 1 or signature in graph_keys:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k != 288 or 1000 * k <= 525 * n:
            continue
        if not _verify_three_witness(G, B, hx, hz):
            continue

        graph_keys.add(signature)
        if accepted == idx:
            if not css_ok(hx, hz):
                raise ValueError("verified lift failed CSS screening")
            return {
                "A": A, "B": B, "screen_k": k,
                "distance_certificate": 3,
                "component_profile": signature[0],
                "component_prescription": ("proper-subgroup", "generating"),
                "reported_archive_references": len(archived),
                "directions": (cards[ia], cards[ib]),
            }
        accepted += 1

    raise ValueError("archive-screened mixed distance-three catalogue exhausted")


from functools import lru_cache


def _r42_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in basis:
                basis[p] = word
                break
            word ^= basis[p]
    return basis


def _r42_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        if p not in basis:
            return True
        word ^= basis[p]
    return False


def _r42_rows(G, grid, side):
    """Extract only scaffold-built constituent actions."""
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    dummy = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows], width


def _r42_pairs(cols):
    """None iff a nonzero kernel support of weight <=4 exists."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = {}
    bits = [1 << i for i in range(len(cols))]
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs[syndrome] = bits[i] | bits[j]
    return pairs


def _r42_floor6(rows, width):
    """Exact constituent screen, returning any discovered 5/6 witness.

    A floor of seven means only that all kernel weights <=6 were
    excluded. It is not a quantum-distance certificate.
    """
    cols = _check_columns(rows, width)
    pairs = _r42_pairs(cols)
    if pairs is None:
        return None
    bits = [1 << i for i in range(width)]
    triples = {}
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            support = bits[i] | bits[j]
            for h in range(j):
                value = syndrome ^ cols[h]
                word = support | bits[h]
                old = pairs.get(value)
                if old is not None:
                    witness = word ^ old
                    return (5, witness)
                old = triples.get(value)
                if old is not None:
                    witness = word ^ old
                    return (6, witness)
                triples[value] = word
    return (7, 0)


def _r42_probe(rows, width, rng, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, never an unresolved cap."""
    cols = _check_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1
    order = list(range(width))

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _r42_outside(word, own):
            return
        best, weight = word, w

    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        short = sorted(kernel, key=int.bit_count)[:20]
        for i, word in enumerate(kernel):
            for other in short:
                trial = word ^ other
                if trial and trial.bit_count() < word.bit_count():
                    word = trial
            kernel[i] = word
            inspect(word)
        for i, word in enumerate(kernel):
            for other in kernel[:i]:
                inspect(word ^ other)
        short = sorted(kernel, key=int.bit_count)[:18]
        for i, word in enumerate(short):
            for j in range(i):
                pair = word ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if weight <= 6:
            break

    if best:
        if any((row & best).bit_count() & 1 for row in rows):
            raise ValueError("kernel probe produced a noncommuting word")
        if own is not None and not _r42_outside(best, own):
            raise ValueError("logical probe produced a stabilizer")
    return best


def _r42_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _r42_key(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Unequal keys certify non-isomorphism under check/qubit relabelings.
    Equal keys are conservatively rejected, not declared isomorphic.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    cols = _check_columns(rows, width)
    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlaps.append((i, j, value))
            if value:
                relation = one if value == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i
    profiles = [Counter() for _ in rows]
    for i, j, value in overlaps:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count())
        profiles[i][(rows[j].bit_count(), value) + common] += 1
        profiles[j][(rows[i].bit_count(), value) + common] += 1
    features = [
        (row.bit_count(), tuple(sorted(profile.items())))
        for row, profile in zip(rows, profiles)]
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    neighborhoods = Counter()
    for col in cols:
        neighborhood = []
        while col:
            bit = col & -col
            neighborhood.append(colors[bit.bit_length() - 1])
            col ^= bit
        neighborhoods[tuple(sorted(neighborhood))] += 1
    return (
        tuple(sorted(Counter(features).items())),
        tuple(sorted(neighborhoods.items())))


def _r42_girth(rows, width):
    """Exact girth of a verified simple degree-two constituent."""
    cols = _check_columns(rows, width)
    adjacency = [[] for _ in rows]
    edges = set()
    for col in cols:
        if col.bit_count() != 2:
            return 0
        bit = col & -col
        a = bit.bit_length() - 1
        b = (col ^ bit).bit_length() - 1
        edge = (min(a, b), max(a, b))
        if edge in edges:
            return 2
        edges.add(edge)
        adjacency[a].append(b)
        adjacency[b].append(a)
    best = len(rows) + 1
    for root in range(len(rows)):
        distance = [-1] * len(rows)
        parent = [-1] * len(rows)
        distance[root] = 0
        queue = [root]
        for a in queue:
            for b in adjacency[a]:
                if distance[b] < 0:
                    distance[b] = distance[a] + 1
                    parent[b] = a
                    queue.append(b)
                elif parent[a] != b:
                    best = min(best, distance[a] + distance[b] + 1)
        if best < 8:
            return best
    return best


@lru_cache(maxsize=1)
def _r42_portfolio(G, count):
    """Pure deterministic portfolio with immutable cached outputs.

    Memoization changes build cost only. No archived cells are seeds,
    and publication cannot mutate another slot's design.
    """
    from itertools import combinations_with_replacement

    if G.q != 42:
        raise ValueError("reinforced portfolio requires order 42")
    q, e = G.q, G.e
    cap = min(9, W_MAX)

    def order(g):
        power = e
        for length in range(1, q + 1):
            power = G.mul(power, g)
            if power == e:
                return length
        raise ValueError("invalid finite-group order")

    orders = {g: order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != e]
    long = [g for g in nonzero if orders[g] >= 6]
    representatives = [
        g for g in long
        if G.idx(g) == min(
            G.idx(G.mul(G.mul(a, g), G.inv(a)))
            for a in G.elements)]

    cores = []
    for g in representatives:
        for h in long:
            if g == h or orders[G.mul(G.inv(g), h)] < 4:
                continue
            grid = [[[e], [e], [e]], [[e], [g], [h]]]
            rows, width = _r42_rows(G, grid, "B")
            girth = _r42_girth(rows, width)
            if girth >= 8:
                cores.append((girth, g, h))
    if not cores:
        raise ValueError("no verified matching core of girth at least eight")
    best_girth = max(c[0] for c in cores)
    cores = [c for c in cores if c[0] == best_girth]

    triangles = [[] for _ in range(4)]
    for r in nonzero:
        for s in nonzero:
            if r == s:
                continue
            minimum = min(orders[r], orders[s],
                          orders[G.mul(G.inv(r), s)])
            commute = G.mul(r, s) == G.mul(s, r)
            if not commute and minimum >= 6:
                triangles[0].append((r, s))
            if not commute and minimum >= 3:
                triangles[1].append((r, s))
            if commute and minimum >= 6:
                triangles[2].append((r, s))
            if commute and minimum >= 3:
                triangles[3].append((r, s))
    broad = triangles[0] + triangles[1] + triangles[2] + triangles[3]
    if not broad:
        raise ValueError("no admissible triangle prescriptions")
    triangles = [pool or broad for pool in triangles]
    triangle_pairs = list(combinations_with_replacement(range(4), 2))
    ratio_pools = [
        [g for g in long if orders[g] == degree] or long
        for degree in (6, 7, 14)]

    def triangle(local, kind):
        r, s = local.choice(triangles[kind])
        return [e, r, s]

    def draw(local, side, card):
        a, b = triangle_pairs[card % 10]
        if side == "A":
            return [[triangle(local, a), triangle(local, b)]]
        profile = card // 10
        _, g, h = local.choice(cores)
        if profile == 0:
            C = [[[], [], [e], [e], [e]],
                 [[], [], [e], [g], [h]]]
            for row, col in ((0, 0), (1, 1)):
                anchor = G.pick(local)
                ratio = local.choice(ratio_pools[card % 3])
                C[row][col] = [anchor, G.mul(anchor, ratio)]
                C[1 - row][col] = [e]
        else:
            C = [[triangle(local, a), [e], [e], [e], []],
                 [[], [e], [g], [h], triangle(local, b)]]
        return C

    banks = {}
    for side in ("A", "B"):
        bank, component_keys = [], set()
        expected_rank = (1 if side == "A" else 2) * q
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0x427A10 + 104729 * card
                             + (0 if side == "A" else 1000003)))
            proposals, fillings = [], set()
            for trial in range(64):
                C = draw(local, side, card)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in C for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, width = _r42_rows(G, C, side)
                if gf2_rank(rows) != expected_rank:
                    continue
                if _r42_pairs(_check_columns(rows, width)) is None:
                    continue
                penalty = _r42_cycles(rows)
                proposals.append((penalty, trial, C, rows, width))
            proposals.sort(key=lambda p: p[:2])

            records = []
            for penalty, trial, C, rows, width in proposals[:14]:
                screened = _r42_floor6(rows, width)
                if screened is None or screened[0] < 6:
                    continue
                floor, seed = screened
                witness = _r42_probe(rows, width, local, 8, seed)
                if not witness:
                    continue
                if witness.bit_count() < floor:
                    raise ValueError("probe contradicted exact constituent floor")
                signature = _r42_key(rows, [], width)
                records.append((
                    (floor, witness.bit_count(), -penalty),
                    trial, C, signature))
            records.sort(key=lambda r: (r[0], -r[1]), reverse=True)
            kept = 0
            for quality, trial, C, signature in records:
                if signature in component_keys:
                    continue
                component_keys.add(signature)
                bank.append((quality, card, trial, C))
                kept += 1
                if kept == 2:
                    break
        if len(bank) < 5:
            raise ValueError("too few distinct reinforced constituents")
        bank.sort(key=lambda r: (r[0], -r[1], -r[2]), reverse=True)
        banks[side] = bank

    plans = []
    for ai, a in enumerate(banks["A"]):
        for bi, b in enumerate(banks["B"]):
            quality = (
                min(a[0][0], b[0][0]),
                min(a[0][1], b[0][1]),
                a[0][2] + b[0][2],
                int(a[1] != b[1]))
            plans.append((quality, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    results, graph_keys = [], set()
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = banks["A"][ai], banks["B"][bi]
        A, B = a[3], b[3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 504 or not (225 * n <= 1000 * k < 275 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = _r42_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1

        # These gates include both LP sectors and all mixed supports.
        if _r42_pairs(_check_columns(hx, n)) is None:
            continue
        if _r42_pairs(_check_columns(hz, n)) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified reinforced lift failed CSS")

        local = random.Random(BASE_SEED ^ (0x42C025 + 65537 * ai + bi))
        xword = _r42_probe(hz, n, local, 4, own=_r42_basis(hx))
        zword = _r42_probe(hx, n, local, 4, own=_r42_basis(hz))
        if not xword or not zword:
            continue
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 6:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / n, preliminary, k)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a[1], b[1]), (a[0][0], b[0][0]),
            best_girth))
        # Optimize over a surplus of genuinely different paired designs.
        if len(results) >= max(count + 8, 2 * count):
            break
        if tested >= max(160, 6 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an invariant-separated joint portfolio."""
    catalogue = _r42_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("reinforced order-42 portfolio exhausted")
    quality, A, B, k, reading, cards, floors, girth = catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "actual_logical_witness_upper_bound": reading,
        "constituent_kernel_floors": floors,
        "design_cards": cards,
        "verified_matching_core_girth": girth,
        "selection_quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh full cell without modifying the cached portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: The Q32 common-proper-subgroup coset catalogue supplies twenty inequivalent slots: its measured sixty slots retained twelve distinct codes and forty-eight iso_dup slots, with all five measured distances remaining three.
# REFUTED: Mixed proper-subgroup/generating Q32 triples supply twenty evaluator-distinct codes: the current sixty-slot evaluation retained only nine classes, and all five measured distances remained three.
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
