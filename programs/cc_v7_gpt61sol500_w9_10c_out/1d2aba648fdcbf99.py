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
CONCEPT = ("Dicyclic Dic7, order 28, A=B=(2,4): balanced cubic constituents selected by exact short-kernel tests and verified logical-witness searches should supply five archive-new measured distance readings at least ten in cell 0.200.")
GROUP = ("Retain the dicyclic construction rule, repairing only its order parameter: x^14=1, y^2=x^7, y*x*y^-1=x^-1. Factor twenty excludes the parent's order 32, which would give n=640; order 28 is the largest admissible dicyclic lift. The related Z7:Z4 inversion bed on (2,4)x(2,4) reached fitness 29.0. That measured result motivates this geometry but does not certify the new fillings' distances or novelty.")
SHAPES = ("Equal (2,4) constituents have 112 variables and 56 checks. With q=28, n=560 and kb=112, giving guaranteed rate 4/20=0.200. The size window permits q<=29. Verified complete ranks enforce [0.175,0.225) for every output. Parent weight-three minima occupied sector-1 Acol/Bcol blocks broadly, with no reported sector-2 minima; cubic columns remove the parent's unit-column/binomial cancellation mechanism.")
FILLING = ("Five balanced support profiles and four relative-order policies define twenty optimized prescriptions. Each constituent row has six terms and each column three, so BOTH CSS check maxima are nine, also below twelve. Verified constituent matrices receive exact kernel screens through four and six, followed by equally budgeted actual-witness searches. Heterogeneous pairings pass full ranks, CSS, complete kernel rejection through four, explicit nontrivial logical searches and conservative intrinsic uncoloured Tanner deduplication. The immutable portfolio prioritizes the largest equally screened witness readings. Constituent floors do not certify mixed-sector quantum distance; distance-ten readings and unseen-archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r020_dic7_q28_24x24_balancedcubic_mixedportfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic7 has order 28")
    # Same dicyclic rule, with its parameter reduced from 8 to 7:
    # x^14=1, y^2=x^7, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        a, b = g
        return ((a + 7 * (b // 2)) % 14, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 7 * b * d) % 14,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 14, 0) if b == 0 else ((a + 7) % 14, 1)

    def idx(g):
        a, b = red(g)
        return 14 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def _retired_q32_mixed_context(rng, G, idx):
    """Inactive historical Q32 catalogue; never publishes new-shape cells."""
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


def _cp_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                pivots[p] = word
                break
            word ^= previous
    return pivots


def _cp_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        previous = pivots.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _cp_pairs(columns):
    """Exact exclusion of all nonzero kernel supports through weight four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = {}
    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs[syndrome] = (1 << i) | (1 << j)
    return pairs


def _cp_six_screen(columns):
    """Exact constituent floor and six-word count.

    All columns have odd weight three, hence all kernel words have even
    weight. After excluding weights <=4, colliding triples detect every
    weight-six word; absence therefore certifies constituent floor eight.
    """
    if any(c.bit_count() != 3 for c in columns):
        raise ValueError("cubic constituent column profile changed")
    if _cp_pairs(columns) is None:
        return None
    triples = {}
    collisions = 0
    witness = 0
    for i, first in enumerate(columns):
        for j in range(i):
            pair = first ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                word = (1 << i) | (1 << j) | (1 << h)
                previous = triples.get(syndrome)
                if previous is None:
                    triples[syndrome] = (word, 1)
                else:
                    old_word, multiplicity = previous
                    collisions += multiplicity
                    if not witness:
                        witness = word ^ old_word
                        if witness.bit_count() != 6:
                            raise ValueError("six-screen contradicted pair screen")
                    triples[syndrome] = (old_word, multiplicity + 1)
    if collisions % 10:
        raise ValueError("six-word partition count inconsistent")
    return (6 if collisions else 8, collisions // 10, witness)


def _cp_component(G, grid, right=False):
    """Expose a constituent using ONLY the scaffold's verified lift."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    return [row & mask for row in rows], width


def _cp_cycle_cost(rows):
    """Tanner four-cycle count, used only as an optimization tie-break."""
    cost = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            cost += overlap * (overlap - 1) // 2
    return cost


def _cp_probe(columns, local, passes, seed=0, own=None, stabilizers=()):
    """Search actual kernel/logical witnesses, never an unresolved cutoff."""
    width = len(columns)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _cp_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        local.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            w = word.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            kernel[i] = word
            inspect(word)

        # Adding stabilizers preserves commutation and logical class.
        if stabilizers:
            reduced = []
            for word in kernel:
                w = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    if tw < w:
                        word, w = trial, tw
                inspect(word)
                if word:
                    reduced.append(word)
            kernel = reduced

        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(set(kernel), key=int.bit_count)[:28]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])

        if own is None and weight <= 6:
            break

    if best:
        syndrome = 0
        bits = best
        while bits:
            bit = bits & -bits
            syndrome ^= columns[bit.bit_length() - 1]
            bits ^= bit
        if syndrome:
            raise ValueError("kernel search returned a noncommuting witness")
        if own is not None and not _cp_outside(best, own):
            raise ValueError("logical search returned a stabilizer")
    return best


def _cp_reduce(word, stabilizers):
    """Two rounds of actual stabilizer-single/pair descent."""
    for _ in range(2):
        starting = word
        weight = word.bit_count()
        for i, first in enumerate(stabilizers):
            trial = starting ^ first
            if trial.bit_count() < weight:
                word, weight = trial, trial.bit_count()
            for second in stabilizers[:i]:
                trial = starting ^ first ^ second
                if trial.bit_count() < weight:
                    word, weight = trial, trial.bit_count()
        if word == starting:
            break
    return word


def _cp_graph_key(hx, hz, n):
    """Intrinsic invariant of the complete UNCOLOURED Tanner graph.

    Check degrees nine and qubit degrees six/twelve intrinsically
    distinguish the vertex classes. Intersection profiles initialize
    complete-incidence refinement. Different keys certify inequivalence;
    equal keys are conservatively discarded, not called isomorphic.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _check_columns(rows, n)
    profiles = _tanner_signature(hx, hz)
    # Recompute the per-check profiles before sorting, so their colors
    # can be attached to the actual incidence graph invariantly.
    m = len(rows)
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            c = (row & rows[j]).bit_count()
            overlaps.append((i, j, c))
            if c:
                relation = one if c == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    roots = [Counter() for _ in rows]
    for i, j, c in overlaps:
        common = (
            c,
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        roots[i][(rows[j].bit_count(),) + common] += 1
        roots[j][(rows[i].bit_count(),) + common] += 1

    initial = [
        (rows[i].bit_count(), tuple(sorted(roots[i].items())))
        for i in range(m)
    ]
    palette = {f: c for c, f in enumerate(sorted(set(initial)))}
    colors = [palette[f] for f in initial]
    incidence = []
    for column in columns:
        checks = []
        while column:
            bit = column & -column
            checks.append(bit.bit_length() - 1)
            column ^= bit
        incidence.append(checks)

    history = [tuple(sorted(Counter(initial).items()))]
    for _ in range(3):
        qfeatures = [
            tuple(sorted(colors[r] for r in checks))
            for checks in incidence
        ]
        qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        neighborhoods = [[] for _ in rows]
        for checks, feature in zip(incidence, qfeatures):
            for r in checks:
                neighborhoods[r].append(qp[feature])
        refined = [
            (colors[r], tuple(sorted(neighborhoods[r])))
            for r in range(m)
        ]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items())),
        ))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]

    intrinsic = (
        n, m,
        tuple(sorted(Counter(len(c) for c in incidence).items())),
        profiles, tuple(history),
    )
    return sha256(repr(intrinsic).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _cp_catalogue(G, count):
    """Deterministic optimized portfolio; cached published grids are immutable."""
    from itertools import combinations

    q, e = G.q, G.e
    if q != 28 or A_SHAPE != (2, 4) or B_SHAPE != (2, 4):
        raise ValueError("balanced cubic portfolio requires the repaired bed")
    width = 4 * q
    cap = min(W_MAX, 9, 12)
    profiles = (
        (3, 3, 0, 0),
        (3, 2, 1, 0),
        (2, 2, 2, 0),
        (3, 1, 1, 1),
        (2, 2, 1, 1),
    )

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid dicyclic element order")

    orders = {g: order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != e]
    pools = {
        degree: [g for g in nonzero if orders[g] == degree]
        for degree in (4, 7, 14)
    }

    def draw(local, profile, policy):
        top = profiles[profile]
        support = [top, tuple(3 - t for t in top)]
        grid = [[[] for _ in range(4)] for _ in range(2)]
        first_pool = pools[(4, 7, 14)[policy]] if policy < 3 else nonzero
        for i, row in enumerate(support):
            for j, terms in enumerate(row):
                if not terms:
                    continue
                anchor = G.pick(local)
                if terms == 1:
                    grid[i][j] = [anchor]
                elif terms == 2:
                    r = local.choice(first_pool)
                    grid[i][j] = [anchor, G.mul(anchor, r)]
                else:
                    r = local.choice(first_pool)
                    second_pool = pools[4] if policy < 3 else nonzero
                    choices = [
                        h for h in second_pool
                        if h != r and G.mul(r, h) != G.mul(h, r)
                    ]
                    if not choices:
                        choices = [h for h in nonzero if h != r]
                    h = local.choice(choices)
                    grid[i][j] = [
                        anchor, G.mul(anchor, r), G.mul(anchor, h)
                    ]
        return grid

    bank, fillings = [], set()
    for card in range(20):
        profile, policy = divmod(card, 4)
        local = random.Random(BASE_SEED ^ (0xC72820 + 104729 * card))
        raw = []
        for trial in range(192):
            grid = draw(local, profile, policy)
            support = [[len(cell) for cell in row] for row in grid]
            if not _no_dead_lines(support, support):
                raise ValueError("cubic prescription has a dead line")
            if _grid_weights(support, support) != 9 or cap < 9:
                raise ValueError("both cubic CSS maxima require cap nine")
            fingerprint = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            if fingerprint in fillings:
                continue
            fillings.add(fingerprint)
            rows, actual_width = _cp_component(G, grid)
            if actual_width != width:
                raise ValueError("constituent extraction width changed")
            rank = gf2_rank(rows)
            if rank < 2*q - 2:
                continue
            columns = _check_columns(rows, width)
            if _cp_pairs(columns) is None:
                continue
            raw.append((
                _cp_cycle_cost(rows), -rank, trial,
                grid, rows, columns
            ))
        raw.sort(key=lambda r: r[:3])

        screened = []
        for cost, negative_rank, trial, grid, rows, columns in raw[:30]:
            exact = _cp_six_screen(columns)
            if exact is None:
                continue
            floor, sixes, seed = exact
            witness = _cp_probe(columns, local, 4, seed)
            if not witness or witness.bit_count() < floor:
                raise ValueError("constituent probe contradicted exact screen")
            screened.append((
                floor, witness.bit_count(), sixes, cost, trial,
                grid, rows, columns, witness
            ))
        screened.sort(key=lambda r: (-r[0], -r[1], r[2], r[3], r[4]))

        # Every retained constituent receives the same additional budget.
        for record in screened[:4]:
            floor, _, sixes, cost, trial, grid, rows, columns, seed = record
            witness = _cp_probe(columns, local, 24, seed)
            if not witness or witness.bit_count() < floor:
                raise ValueError("final constituent probe violated its floor")
            right_rows, right_width = _cp_component(G, grid, True)
            right_word, source = 0, witness
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length() - 1, q)
                right_word |= 1 << (
                    j*q + G.idx(G.inv(G.elements[t]))
                )
                source ^= bit
            if right_width != width or any(
                    (row & right_word).bit_count() & 1
                    for row in right_rows):
                raise ValueError("verified inversion transfer failed")
            if gf2_rank(right_rows) != gf2_rank(rows):
                raise ValueError("left/right constituent ranks disagree")
            bank.append(dict(
                grid=grid, floor=floor, sixes=sixes, cost=cost,
                card=card, trial=trial, left=witness, right=right_word,
                reading=witness.bit_count(),
            ))

    if len(bank) < 5:
        raise ValueError("too few cubic constituents passed verified screens")
    bank.sort(key=lambda r: (
        -r["floor"], -r["reading"], r["sixes"],
        r["cost"], r["card"], r["trial"]
    ))

    plans = sorted(
        (-min(a["floor"], b["floor"]),
         -min(a["reading"], b["reading"]),
         a["sixes"] + b["sixes"], a["cost"] + b["cost"],
         int(a["card"] == b["card"]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai)
    )

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(4):
            if kind == "X":
                word = record["right"] << (block * width)
            else:
                word, source = 0, record["left"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*4 + block)*q + t)
                    source ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("embedded constituent witness does not commute")
            if not _cp_outside(word, own):
                continue
            word = _cp_reduce(word, stabilizers)
            if not word or not _cp_outside(word, own):
                raise ValueError("stabilizer descent changed the logical class")
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, graph_keys = [], set()
    tested = 0
    for _, _, _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 560 or not (175*n <= 1000*k < 225*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = _cp_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        cx, cz = _check_columns(hx, n), _check_columns(hz, n)
        if _cp_pairs(cx) is None or _cp_pairs(cz) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified cubic pair failed CSS")

        bx, bz = _cp_basis(hx), _cp_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0xC72890 + 1009*ai + bi))
        xword = _cp_probe(cz, local, 2, xseed, bx, hx)
        zword = _cp_probe(cx, local, 2, zseed, bz, hz)
        if not xword or not zword:
            continue
        reading = min(xword.bit_count(), zword.bit_count())

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (
            k * reading * reading / n,
            min(a["floor"], b["floor"]),
            -a["sixes"] - b["sixes"],
            -a["cost"] - b["cost"],
        )
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["floor"], b["floor"]),
            xword, zword, signature.hex(),
        ))
        if len(results) >= count + 8:
            break
        if tested >= max(192, 4*count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of a jointly screened immutable portfolio."""
    catalogue = _cp_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct target-rate cubic portfolio exhausted")
    quality, A, B, k, reading, cards, floors, xword, zword, signature = \
        catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k, "quality": quality,
        "design_cards": cards, "constituent_kernel_floors": floors,
        "logical_witness_upper_bound": reading,
        "verified_X_witness": xword, "verified_Z_witness": zword,
        "full_kernel_floor": 5, "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without resampling or mutating the portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: The Q32 common-proper-subgroup coset catalogue supplies twenty inequivalent slots: its measured sixty slots retained twelve distinct codes and forty-eight iso_dup slots, with all five measured distances remaining three.
# REFUTED: The reported Q32 mixed/archive-screened catalogue supplies twenty evaluator-distinct improved codes: sixty evaluated slots retained nine distinct classes, and all five measured leaders still had k=288 and d30s=3.
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
