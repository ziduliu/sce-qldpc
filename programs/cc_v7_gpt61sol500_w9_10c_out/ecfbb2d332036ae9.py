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
CONCEPT = ("Dic11, order 44, A=(1,2), B=(3,5): independently optimized cubic constituents followed by coupled logical-witness screening should supply five archive-new measured readings >=13 in cell 0.150; this is a falsifiable measurement hypothesis.")
GROUP = ("Jump from cyclic C8 to nonabelian Dic11: x^22=1, y^2=x^11, y*x*y^-1=x^-1. The order-44 (1,2)x(3,5) bed reached fitness 22.8, while Dic11 also leads the low-rate frontier at 46.5. These measurements motivate optimization, not a transferred distance certificate. Order 44 gives legal n=572; Dic12 would exceed this shape's size window.")
SHAPES = ("Use unequal (1,2) and (3,5) constituents, with 88 variables/44 checks and 220 variables/132 checks. Factor 13 permits q<=46; q=44 gives n=572, kb=88 and guaranteed rate 2/13=0.153846. Verified complete ranks retain [0.125,0.175). Parent weight-three minima were entirely sector 1, particularly Bcol6/7 for X and Acol6/7 for Z. Removing its singleton-pivot columns replaces those local three-word mechanisms.")
FILLING = ("A has two noncommuting three-term cells. B has five degree-three columns under five support profiles and four relative-coset policies. Its row totals never exceed six: BOTH LP maxima are <=9, also below twelve. Independently optimize exact scaffold-built left-A and right-B constituents, excluding kernels through four and refining actual witnesses with equal budgets. An immutable coupled portfolio verifies ranks, CSS, full short-kernel exclusion and nontrivial logical witnesses, rejecting repeated intrinsic full-Tanner invariants. Witness readings remain upper bounds; archive novelty and five measured readings >=13 are evaluator-tested hypotheses.")
FAMILY_NAME = "r015_dic11_q44_12x35_cubic_independent_coupledportfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic11 requires order 44")
    # (a,b)=x^a*y^b; x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((int(a) + 11 * (int(b) // 2)) % 22, int(b) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11*b*d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return 22*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _component_spec(rng, mode, q):
    """Matching columns, or two cycle columns plus five matchings."""
    if mode == 0:
        return (), tuple(sorted(rng.sample(range(q), 7)))
    if mode == 1:
        d = rng.randrange(1, q)
        phase = rng.randrange(q)
        # Five independent vertices of the eleven-cycle with step d.
        labels = tuple(sorted((phase + d * t) % q
                              for t in (0, 2, 4, 6, 8)))
        return (d, d), labels
    return (rng.randrange(1, q), rng.randrange(1, q)), \
        tuple(sorted(rng.sample(range(q), 5)))


def _component_cells(G, spec):
    ds, labels = spec
    e, els = G.e, G.elements
    if not ds:
        return [[[e] for _ in labels],
                [[els[s]] for s in labels]]
    # Column weights remain two; both row weights remain seven.
    return [[[e, els[ds[0]]]] + [[e] for _ in labels] + [[]],
            [[]] + [[els[s]] for s in labels] + [[e, els[ds[1]]]]]


def _translation_key(labels, q):
    return min(tuple(sorted((s + t) % q for s in labels))
               for t in range(q))


def _scaled_spec_key(spec, u, q):
    ds, labels = spec
    s = tuple((u * x) % q for x in labels)
    forward = _translation_key(s, q)
    backward = _translation_key(tuple((-x) % q for x in s), q)
    if not ds:
        return (0, min(forward, backward))
    d0, d1 = ((u * d) % q for d in ds)
    d0, d1 = min(d0, q - d0), min(d1, q - d1)
    return min((1, d0, d1, forward),
               (1, d1, d0, backward))


def _pair_key(a, b, q):
    """Remove translations, row swaps, common automorphisms and A/B swap."""
    keys = []
    for u in range(1, q):
        ka = _scaled_spec_key(a, u, q)
        kb = _scaled_spec_key(b, u, q)
        keys.append(tuple(sorted((ka, kb))))
    return min(keys)


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


def _no_short_kernel(rows, n):
    """Exactly exclude kernel vectors of weights one, two and three."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    seen = set(columns)
    if 0 in seen or len(seen) != n:
        return False
    for i, c in enumerate(columns):
        for j in range(i):
            if (c ^ columns[j]) in seen:
                return False
    return True


def _retired_z11_singleton_context(rng, G, idx):
    """Inactive parent catalogue; never used to fill the new shapes.

    A is a triangle-free incidence component. B has one singleton
    column and six distinct cross-layer matchings. Thus B has no
    kernel of weight <=3 despite its degree-one column: a short path
    between singleton endpoints needs at least two matching edges.
    No completed code is varied solely by a common group automorphism.
    """
    from itertools import combinations

    q = G.q
    e, els = G.e, G.elements

    def affine_key(labels):
        # Remove component translations and scalings before making
        # cards. Relative A/B voltage orientations are explored later.
        return min(_translation_key(tuple((u * s) % q for s in labels), q)
                   for u in range(1, q))

    def matching_cards(count):
        return sorted({affine_key(labels)
                       for labels in combinations(range(q), count)})

    acards = matching_cards(7)
    bcards = matching_cards(6)

    def matching_A(labels):
        return [[[e] for _ in labels],
                [[els[s]] for s in labels]]

    def cycle_A(steps, labels):
        # Each internal cycle column contributes two terms to one
        # row and none to the other. Matching columns contribute one
        # to each row. All seven columns retain total weight two.
        p = len(steps)
        upper = [[e, els[d]] for d in steps]
        lower = [[] for _ in steps]
        upper += [[e] for _ in labels]
        lower += [[els[s]] for s in labels]
        upper += [[] for _ in steps]
        lower += [[e, els[d]] for d in steps]
        assert len(upper) == len(lower) == 7
        assert sum(map(len, upper)) == sum(map(len, lower)) == 7
        return [upper, lower]

    afillings = [matching_A(labels) for labels in acards]

    # One eleven-cycle on each layer; five cross-matchings form an
    # independent set in the step-one cycle.
    afillings.append(cycle_A((1,), (0, 2, 4, 6, 8)))

    # Two cycles per layer. Steps 1 and 3 have no internal triangles;
    # these matching triples have no differences +/-1 or +/-3.
    # The arithmetic-progression and non-progression triples are
    # genuinely different voltage cards, not random relabelings.
    afillings.append(cycle_A((1, 3), (0, 2, 4)))
    afillings.append(cycle_A((1, 3), (0, 2, 6)))

    def singleton_B(labels, scale):
        shifts = sorted((scale * s) % q for s in labels)
        # Column six is the singleton. Row totals are seven and six,
        # column totals are two for columns 0..5 and one for column 6.
        return [[[e] for _ in range(7)],
                [[els[s]] for s in shifts] + [[]]]

    # Diagonal ordering spreads the first twenty entries over both
    # component catalogues and all three A support profiles. Extra
    # slots explore relative voltage orientations, subject to the
    # same full-graph invariant rejection.
    plans = [(a, (turn + 3 * a) % len(bcards))
             for turn in range(len(bcards))
             for a in range(len(afillings))]

    graph_keys = set()
    accepted = 0
    for scale in range(1, q):
        for ai, bi in plans:
            A = afillings[ai]
            B = singleton_B(bcards[bi], scale)
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb):
                continue
            if _grid_weights(sa, sb) > min(9, W_MAX):
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != 280 or not (475 * n <= 1000 * k < 525 * n):
                continue

            signature = _tanner_signature(hx, hz)
            if signature in graph_keys:
                continue
            graph_keys.add(signature)

            if accepted == idx:
                # Screen the actual evaluator matrices, not a
                # separately implemented incidence or voltage lift.
                if not (_no_short_kernel(hx, n)
                        and _no_short_kernel(hz, n)):
                    raise ValueError("asymmetric card has a kernel of weight <=3")
                if not css_ok(hx, hz):
                    raise ValueError("verified asymmetric lift failed CSS")
                return {"A": A, "B": B}
            accepted += 1

    raise ValueError("non-isomorphic asymmetric catalogue exhausted")


def _retired_z8_two_pivot_context(rng, G, idx):
    """Inactive Z8-only historical catalogue, never used by CELL.

    No random orbit copies or mutable cross-slot cache. Equal invariant
    keys are conservatively discarded; different keys certify that the
    corresponding graphs cannot be isomorphic.
    """
    from itertools import combinations

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    if q != 8:
        raise ValueError("two-pivot catalogue requires order eight")
    elements = {G._key(g): g for g in G.elements}

    def x(t):
        return elements[(t % q, 0)]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def clear_two(rows, width):
        cols = columns(rows, width)
        return 0 not in cols and len(set(cols)) == width

    def walk_key(adj):
        """Exact permutation invariant of an undirected multigraph.

        Repeated neighbours retain edge multiplicities. Joint walk
        profiles through length eight are structural invariants, not
        distance estimates or complete isomorphism tests.
        """
        size = len(adj)
        power = [[int(u == v) for v in range(size)]
                 for u in range(size)]
        features = [[[] for _ in range(size)] for _ in range(size)]
        diagonal = [[] for _ in range(size)]
        for length in range(1, 9):
            power = [
                [sum(power[w][v] for w in adj[u])
                 for v in range(size)]
                for u in range(size)
            ]
            for u in range(size):
                diagonal[u].append(power[u][u])
                for v in range(size):
                    features[u][v].append(power[u][v])
        return tuple(sorted(
            (len(adj[u]), tuple(diagonal[u]),
             tuple(sorted(
                 (len(adj[v]), tuple(diagonal[v]),
                  tuple(features[u][v]))
                 for v in range(size))))
            for u in range(size)))

    def build(desc):
        upper, lower, labels = desc
        grid = [
            [[e, x(d)] for d in upper]
            + [[e] for _ in labels]
            + [[] for _ in lower] + [[e], []],
            [[] for _ in upper]
            + [[x(t)] for t in labels]
            + [[e, x(d)] for d in lower] + [[], [e]],
        ]
        assert all(len(row) == 8 for row in grid)
        assert all(sum(map(len, row)) == 7 for row in grid)
        assert [len(grid[0][j]) + len(grid[1][j])
                for j in range(8)] == [2] * 6 + [1, 1]
        return grid

    def component_key(grid):
        # Extract the exact A action using the scaffold's unit-side lift.
        hx, _, _ = lifted_matrices(G, grid, [[[e]]])
        width = 8 * q
        mask = (1 << width) - 1
        rows = [row & mask for row in hx]
        cols = columns(rows, width)
        if 0 in cols or len(set(cols)) != width:
            return None
        if gf2_rank(rows) != 2 * q:
            return None
        adj = [[] for _ in rows]
        for col in cols:
            if col.bit_count() == 1:
                continue
            if col.bit_count() != 2:
                return None
            first = col & -col
            u = first.bit_length() - 1
            v = (col ^ first).bit_length() - 1
            adj[u].append(v)
            adj[v].append(u)
        assert all(len(neighbours) == 6 for neighbours in adj)
        return walk_key(adj)

    # Noninvolutory inverse classes 1,2,3 give simple internal cycles.
    # Step four is excluded: its binomial produces duplicate columns.
    representatives = {}
    for p in range(4):
        steps = list(combinations(range(1, 4), p))
        cross = 6 - 2 * p
        label_sets = sorted({
            _translation_key(values, q)
            for values in combinations(range(q), cross)
        })
        for ai, upper in enumerate(steps):
            for lower in steps[ai:]:
                for labels in label_sets:
                    desc = (upper, lower, labels)
                    grid = build(desc)
                    key = component_key(grid)
                    if key is not None and key not in representatives:
                        representatives[key] = (desc, grid)

    # Interleave internal-cycle support profiles rather than exhausting
    # the matching-only family in the initial measured portfolio.
    by_profile = [
        sorted((item for item in representatives.values()
                if len(item[0][0]) == p), key=lambda item: item[0])
        for p in range(4)
    ]
    bank = []
    for turn in range(max(map(len, by_profile))):
        for p in (2, 1, 0, 3):
            if turn < len(by_profile[p]):
                bank.append(by_profile[p][turn])
    if not bank:
        raise ValueError("no simple two-pivot constituents")

    def full_key(hx, hz, width):
        """Intrinsic invariants of the complete, uncoloured Tanner graph.

        Degree fourteen identifies sector-two qubits without knowing
        their protograph labels. Checks meeting two such qubits become
        multigraph edges; singleton checks are omitted from this
        projection. Both this projection and the complete check
        intersection invariant survive arbitrary Tanner relabelings.
        """
        rows = hx + hz
        cols = columns(rows, width)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() == 14]
        if len(high) != 4 * q:
            raise ValueError("intrinsic high-degree sector changed")
        position = {j: v for v, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        adj = [[] for _ in high]
        markers = [0] * len(high)
        for row in rows:
            support = row & mask
            first = support & -support
            if support.bit_count() == 1:
                markers[position[first.bit_length() - 1]] += 1
            elif support.bit_count() == 2:
                u = position[first.bit_length() - 1]
                v = position[(support ^ first).bit_length() - 1]
                adj[u].append(v)
                adj[v].append(u)
            else:
                raise ValueError("unexpected projected check incidence")
        assert all(count == 2 for count in markers)
        return (_tanner_signature(hx, hz), walk_key(adj))

    def three_witness(B):
        # Derive endpoint syndromes from the VERIFIED B action.
        _, hz, _ = lifted_matrices(G, [[[e]]], B)
        width = 8 * q
        mask = (1 << width) - 1
        cols = columns([row & mask for row in hz], width)
        single = {col: 1 << j for j, col in enumerate(cols)
                  if col.bit_count() == 1}
        for j, col in enumerate(cols):
            if col.bit_count() == 2:
                first = col & -col
                if first in single and (col ^ first) in single:
                    return (1 << j) | single[first] | single[col ^ first]
        raise ValueError("two-pivot constituent lost its weight-three word")

    graph_keys = set()
    accepted = 0
    # Cover different intrinsic constituent pairings before exploring
    # relative orientations. These are structural cards, not draws.
    for separation in range(len(bank)):
        for ai in range(len(bank) - separation):
            bi = ai + separation
            A, B = bank[ai][1], bank[bi][1]
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            hx, hz, n = lifted_matrices(G, A, B)
            if n != 544 or max(row.bit_count() for row in hx + hz) > cap:
                continue
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if k != 288 or 40 * k <= 21 * n:
                continue
            signature = full_key(hx, hz, n)
            if signature in graph_keys:
                continue
            if not (clear_two(hx, n) and clear_two(hz, n)):
                continue
            graph_keys.add(signature)

            if accepted == idx:
                witness = three_witness(B)  # Acol0, entirely sector one.
                if witness.bit_count() != 3:
                    raise ValueError("invalid weight-three witness")
                if any((row & witness).bit_count() & 1 for row in hz):
                    raise ValueError("verified witness does not commute")
                if gf2_rank(hx + [witness]) != rx + 1:
                    raise ValueError("weight-three word is a stabilizer")
                if not css_ok(hx, hz):
                    raise ValueError("verified two-pivot lift failed CSS")
                return {
                    "A": A, "B": B, "k": k,
                    "card": (bank[ai][0], bank[bi][0]),
                    "distance_certificate": 3,
                    "logical_witness": witness,
                }
            accepted += 1

    raise ValueError("invariant-distinct two-pivot catalogue exhausted")


from functools import lru_cache


def _dc_columns(rows, width):
    """Syndromes in the coordinates of supplied verified matrices."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _dc_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _dc_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _dc_kernel4(rows, width):
    """Exact exclusion of every nonzero kernel support through four."""
    cols = _dc_columns(rows, width)
    seen = {0}
    for col in cols:
        if col in seen:
            return None
        seen.add(col)
    squares = 0
    for i, col in enumerate(cols):
        for other in cols[:i]:
            syndrome = col ^ other
            if syndrome in seen:
                return None
            seen.add(syndrome)
            overlap = (col & other).bit_count()
            squares += overlap * (overlap - 1) // 2
    return cols, squares


def _dc_cycle_cost(rows, cols, squares):
    """A short-cycle selection penalty, not a distance certificate."""
    adjacency = [0] * len(rows)
    internal = 0
    for col in cols:
        degree = col.bit_count()
        internal += degree * (degree - 1) * (degree - 2) // 6
        bits = col
        while bits:
            bit = bits & -bits
            adjacency[bit.bit_length() - 1] |= col ^ bit
            bits ^= bit
    triangles = 0
    for i, neighbors in enumerate(adjacency):
        bits = neighbors >> (i + 1)
        while bits:
            bit = bits & -bits
            j = bit.bit_length() + i
            triangles += ((neighbors & adjacency[j]) >> (j + 1)).bit_count()
            bits ^= bit
    return 16*squares + max(0, triangles - internal)


def _dc_probe(rows, width, local, passes, seed=0, own=None,
              stabilizers=()):
    """Carry actual kernel/logical witnesses across randomized searches.

    Large readings are upper bounds. Span membership excludes stabilizers
    when this function probes a complete quantum matrix.
    """
    cols = _dc_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        value = word.bit_count()
        if not word or value >= weight:
            return
        if own is not None and not _dc_outside(word, own):
            return
        best, weight = word, value

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
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

        if own is None:
            # Elementary basis exchanges preserve the entire kernel.
            local.shuffle(kernel)
            for i in range(len(kernel)):
                current = kernel[i]
                for j, other in enumerate(kernel):
                    if i != j:
                        trial = current ^ other
                        if 0 < trial.bit_count() < current.bit_count():
                            current = trial
                kernel[i] = current
                inspect(current)
            for i, first in enumerate(kernel):
                for second in kernel[:i]:
                    inspect(first ^ second)
        else:
            reduced = []
            for word in kernel:
                for row in stabilizers:
                    trial = word ^ row
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
                inspect(word)
                if _dc_outside(word, own):
                    reduced.append(word)
            kernel = reduced

        short = sorted(set(kernel), key=int.bit_count)[:28]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])

        if best:
            moves = short if own is None else stabilizers
            starting = best
            for i, first in enumerate(moves):
                inspect(starting ^ first)
                for second in moves[:i]:
                    inspect(starting ^ first ^ second)
        if weight <= (10 if own is None else 8):
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _dc_outside(best, own)
    return best


def _dc_graph_key(rows, width):
    """Intrinsic full-Tanner invariant; equality is inconclusive.

    Published checks have degrees <=9, qubits have degrees 6 or 12.
    Thus their bipartite classes are intrinsically distinguished.
    No group labels, protograph coordinates or X/Z colours enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    m = len(rows)
    adjacency = [0] * m
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            if value:
                overlaps[(i, j)] = value
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    walks = [0] * m
    for i in range(m):
        for j in range(i):
            common = (adjacency[i] & adjacency[j]).bit_count()
            value = overlaps.get((i, j), 0)
            profiles[i][(rows[j].bit_count(), value, common)] += 1
            profiles[j][(rows[i].bit_count(), value, common)] += 1
            walks[i] += common*common
            walks[j] += common*common
    features = [
        (row.bit_count(), adjacency[i].bit_count(),
         walks[i] + adjacency[i].bit_count()**2,
         tuple(sorted(profiles[i].items())))
        for i, row in enumerate(rows)
    ]
    history = [tuple(sorted(Counter(features).items()))]
    palette = {f: i for i, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    cols = _dc_columns(rows, width)
    incidence = []
    for col in cols:
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(checks)

    for _ in range(3):
        qfeatures = [tuple(sorted(colors[r] for r in checks))
                     for checks in incidence]
        qp = {f: i for i, f in enumerate(sorted(set(qfeatures)))}
        neighbors = [[] for _ in rows]
        for checks, feature in zip(incidence, qfeatures):
            for r in checks:
                neighbors[r].append(qp[feature])
        refined = [(colors[r], tuple(sorted(neighbors[r])))
                   for r in range(m)]
        history.append((tuple(sorted(Counter(qfeatures).items())),
                        tuple(sorted(Counter(refined).items()))))
        palette = {f: i for i, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    signature = (width, m,
                 tuple(sorted(Counter(len(c) for c in incidence).items())),
                 tuple(history))
    return sha256(repr(signature).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _dc_portfolio(G, count):
    """Deterministic joint portfolio; cached publications are immutable."""
    q, e = G.q, G.e
    if q != 44:
        raise ValueError("Dic11 cubic portfolio requires order 44")
    cap = min(W_MAX, 9, 12)

    # B column descriptions. Every column has three terms.
    # These profiles vary cell multiplicities, not merely label gauges.
    profiles = (
        ((1,1,1),)*5,
        ((2,1,0), (0,2,1), (1,0,2), (1,1,1), (1,1,1)),
        ((3,0,0), (0,3,0), (0,0,3), (1,1,1), (1,1,1)),
        ((2,1,0), (0,2,1), (1,0,2), (2,0,1), (0,1,2)),
        ((2,1,0), (2,0,1), (0,2,1), (0,1,2), (1,1,1)),
    )
    sa = [[3, 3]]
    for profile in profiles:
        sb = [[profile[j][i] for j in range(5)] for i in range(3)]
        wx = max(sum(sa[0]) + sum(sb[i][j] for i in range(3))
                 for j in range(5))
        wz = max(sum(row) + sa[0][j] for row in sb for j in range(2))
        assert wx <= cap and wz <= cap
        assert _no_dead_lines(sa, sb)

    rotations = [
        [(a, 0) for a in range(1, 22)
         if a != 11 and (policy >= 2 or a % 2 == policy)]
        for policy in range(4)
    ]

    def triangle(local, policy):
        rotation = local.choice(rotations[policy])
        cross = (local.randrange(22), 1)
        if policy == 3:
            return [e, cross, G.mul(cross, rotation)]
        return [e, rotation, cross]

    def draw(local, side, card):
        policy = card % 4
        if side == "A":
            return [[triangle(local, policy),
                     triangle(local, (policy + 1) % 4)]]
        profile = profiles[card // 4]
        grid = [[[] for _ in range(5)] for _ in range(3)]
        for j, counts in enumerate(profile):
            if counts == (1, 1, 1):
                # Normalize one column endpoint. The other two are
                # genuine relative voltages, not a saved constituent.
                g = local.choice(rotations[policy])
                h = (local.randrange(22), 1)
                if policy >= 2:
                    g = (local.randrange(22), 1)
                    while h == g:
                        h = (local.randrange(22), 1)
                for i, voltage in enumerate((e, g, h)):
                    grid[i][j] = [voltage]
                continue
            for i, terms in enumerate(counts):
                anchor = G.pick(local)
                if terms == 1:
                    grid[i][j] = [anchor]
                elif terms == 2:
                    relative = ((local.randrange(22), 1)
                                if (i+j+policy) % 2
                                else local.choice(rotations[policy]))
                    grid[i][j] = [anchor, G.mul(anchor, relative)]
                elif terms == 3:
                    grid[i][j] = [
                        G.mul(anchor, g) for g in triangle(local, policy)]
        assert all(len(set(cell)) == len(cell)
                   for row in grid for cell in row)
        return grid

    def component(grid, side):
        # Zero auxiliary cells expose EXACT scaffold actions and are
        # never emitted as final cells.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        width = len(grid[0])*q
        mask = (1 << width) - 1
        assert all(row.bit_length() <= width for row in rows)
        return [row & mask for row in rows], width

    def bank(side):
        records, filling_keys, graph_keys = [], set(), set()
        cards = range(8) if side == "A" else range(20)
        for card in cards:
            local = random.Random(
                BASE_SEED ^ (0xD11440 + 104729*card
                             + (0 if side == "A" else 7919)))
            raw = []
            for trial in range(128):
                grid = draw(local, side, card)
                filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for row in grid for cell in row)
                if filling in filling_keys:
                    continue
                filling_keys.add(filling)
                rows, width = component(grid, side)
                if gf2_rank(rows) != len(grid)*q:
                    continue
                stats = _dc_kernel4(rows, width)
                if stats is None:
                    continue
                cols, squares = stats
                cost = _dc_cycle_cost(rows, cols, squares)
                raw.append((cost, trial, grid, rows, width))
            raw.sort(key=lambda item: item[:2])

            shortlist = []
            for cost, trial, grid, rows, width in raw[:10]:
                witness = _dc_probe(rows, width, local, 4)
                if witness:
                    shortlist.append(dict(
                        grid=grid, rows=rows, width=width, cost=cost,
                        trial=trial, card=card, witness=witness))
            shortlist.sort(key=lambda r: (
                -r["witness"].bit_count(), r["cost"], r["trial"]))

            # Every retained constituent receives the same 4+20 passes.
            finalists = shortlist[:3]
            for record in finalists:
                record["witness"] = _dc_probe(
                    record["rows"], record["width"], local, 20,
                    record["witness"])
            finalists.sort(key=lambda r: (
                -r["witness"].bit_count(), r["cost"], r["trial"]))
            for record in finalists:
                key = _dc_graph_key(record["rows"], record["width"])
                if key in graph_keys:
                    continue
                graph_keys.add(key)
                records.append(record)
        records.sort(key=lambda r: (
            -r["witness"].bit_count(), r["cost"], r["card"], r["trial"]))
        return records

    aa, bb = bank("A"), bank("B")
    if len(aa) < 2 or len(bb) < 3:
        raise ValueError("too few independently screened cubic constituents")

    plans = sorted(
        (-min(a["witness"].bit_count(), b["witness"].bit_count()),
         a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))
    results, graph_keys = [], set()
    screened = 0

    def embedded(a, b, kind, checks, stabilizers, own):
        best = 0
        for block in range(2 if kind == "X" else 5):
            if kind == "X":
                word = b["witness"] << (block*5*q)
            else:
                word, source = 0, a["witness"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _dc_outside(word, own):
                continue
            for _ in range(2):
                starting = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == starting:
                    break
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    for _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        support_A = [[len(c) for c in row] for row in A]
        support_B = [[len(c) for c in row] for row in B]
        assert _no_dead_lines(support_A, support_B)
        assert _grid_weights(support_A, support_B) <= cap
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 572 or not (125*n <= 1000*k < 175*n):
            continue
        signature = _dc_graph_key(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        screened += 1
        if _dc_kernel4(hx, n) is None or _dc_kernel4(hz, n) is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified Dic11 coupled lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= cap

        bx, bz = _dc_basis(hx), _dc_basis(hz)
        xseed = embedded(a, b, "X", hz, hx, bx)
        zseed = embedded(a, b, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0xD11C00 + 1009*ai + bi))
        xword = _dc_probe(hz, n, local, 2, xseed, bx, hx)
        zword = _dc_probe(hx, n, local, 2, zseed, bz, hz)
        reading = min(xword.bit_count(), zword.bit_count())

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k*reading*reading/n, reading, -a["cost"]-b["cost"])
        results.append((quality, freeze(A), freeze(B), k, reading,
                        (a["card"], b["card"]), signature.hex()))
        if len(results) >= max(count, 5) + 12:
            break
        if screened >= max(256, 8*count):
            break
    results.sort(key=lambda item: item[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of jointly screened deliberate designs."""
    catalogue = _dc_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("invariant-separated Dic11 portfolio exhausted")
    quality, A, B, k, reading, cards, signature = catalogue[idx]
    return {"A": A, "B": B, "k": k, "quality": quality,
            "design_cards": cards,
            "logical_witness_upper_bound": reading,
            "kernel_floor": 5, "tanner_signature": signature}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct good designs: 60 evaluated slots retained only four distinct codes, with d30s readings three or four.
# REFUTED: One bare singleton plus affine matching-set variation supplies a broad high-fitness portfolio: sixty parent slots collapsed to one measured k=280, d30s=4 code, scoring 1.5368.
# REFUTED: Constituent-walk representatives supply twenty inequivalent two-pivot Z8 codes: sixty measured slots retained only six distinct classes, with all five finalists at k=288 and d30s=3.
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
