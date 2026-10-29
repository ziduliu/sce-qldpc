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
CONCEPT = ("C3 semidirect_-1 C16, order 48, A=(1,2), B=(2,5): independently optimized noncommuting trinomials and balanced cubic B columns should supply five novel measured distance readings >=10 in cell 0.250; this is a measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from the dicyclic Dic7 rule to the split semidirect product x^3=y^16=1, y*x*y^-1=x^-1. This group has centre eight and 24 conjugacy classes, matching the structural description of the order-48 bed leading cell 0.250 at fitness 20.9. Odd complement elements have order sixteen; their orders are construction prescriptions, not distance guarantees.")
SHAPES = ("Use unequal (1,2)x(2,5) constituents with 96/48 and 240/96 variables/checks. The size window permits q<=49; q=48 gives n=576 and kb=144, guaranteed rate 0.250. Actual verified ranks retain [0.225,0.275). Parent weight-three minima occupied sector-1 Acol/Bcol blocks; remove its monomial A pivot rather than preserve that certified distance-three mechanism.")
FILLING = ("A has two noncommuting trinomials. Four B profiles have row totals six and maximum column total three, giving BOTH LP check maxima nine, also below twelve. Twenty support/order prescriptions undergo independent constituent optimization followed by complete kernel-four screens, actual non-stabilizer quantum probes and stabilizer descent. An immutable portfolio rejects repeated intrinsic full-Tanner signatures and the shown destination incumbent. Probe readings remain upper bounds; unseen archive novelty and distance-ten yield are evaluator-tested hypotheses.")
FAMILY_NAME = "r025_split3by16_q48_12x25_cubic_quantum_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C3 semidirect C16 requires order 48")
    # (a,b)=x^a*y^b; x^3=y^16=1 and y*x*y^-1=x^-1.
    # Inversion squared is identity, so the action factors through C16.
    els = [(a, b) for b in range(16) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 16)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b % 2 else c)) % 3,
                (b + d) % 16)

    def inv(g):
        a, b = red(g)
        return ((a if b % 2 else -a) % 3, (-b) % 16)

    def idx(g):
        a, b = red(g)
        return 3*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


_DIC_PROFILES = (
    (4, 4, 4), (4, 4, 7), (4, 4, 14), (4, 7, 7),
    (4, 7, 14), (4, 14, 14), (7, 7, 7), (7, 7, 14),
    (7, 14, 14), (14, 14, 14),
)
_DIC_PRIMARY = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 4), (0, 9), (1, 5), (1, 8), (2, 3),
    (2, 7), (3, 5), (4, 6), (5, 8), (6, 9),
)
_DIC_CARDS = _DIC_PRIMARY + tuple(
    (a, b) for a in range(10) for b in range(a, 10)
    if (a, b) not in _DIC_PRIMARY
)


def _dic_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid finite-group order")


def _dic_component(rng, G, orders, pools):
    """A fixed monomial pivot plus three prescribed-order binomials."""
    row = [[G.e]]
    used = set()
    for order in orders:
        available = [h for h in pools[order]
                     if h not in used and G.inv(h) not in used]
        if not available:
            raise ValueError("relative-order card exhausted")
        h = rng.choice(available)
        used.add(h)
        g = G.pick(rng)
        row.append([g, G.mul(g, h)])
    return [row]


def _dic_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _dic_excludes_two(rows, n):
    """Complete exclusion of commuting supports of weights one and two."""
    columns = _dic_columns(rows, n)
    return 0 not in columns and len(set(columns)) == n


def _dic_three_witness(G, B):
    """X witness in Acol1: one B-binomial bit and two B-pivot bits.

    B[0][0]=identity. For binomial labels u,v, selecting its column
    at identity has check syndromes at u^-1 and v^-1, canceled by
    those two columns of the monomial pivot.
    """
    q = G.q
    u, v = B[0][1]
    base = 4*q                  # Acol1/Bcol0
    return ((1 << (base + q + G.idx(G.e)))
            | (1 << (base + G.idx(G.inv(u))))
            | (1 << (base + G.idx(G.inv(v)))))


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


def _component_probe(rows, n, rng):
    """Exact search through weight 4, then heuristic kernel probes."""
    from itertools import combinations

    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2

    pairs = set()
    found4 = False
    for i in range(n):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    if found4:
        return 4

    best = n + 1
    for _ in range(4):
        order = list(range(n))
        rng.shuffle(order)
        image = [1 << order[c] for c in range(n)]
        pivots = {}
        for original in rows:
            row = 0
            bits = original
            while bits:
                bit = bits & -bits
                row ^= image[bit.bit_length() - 1]
                bits ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ascending = sorted(pivots)
        basis = []
        for f in range(n):
            if f in pivots:
                continue
            word = 1 << f
            for p in ascending:
                if (pivots[p] & word).bit_count() & 1:
                    word ^= 1 << p
            basis.append(word)

        for word in basis:
            best = min(best, word.bit_count())
        for a, b in combinations(basis, 2):
            best = min(best, (a ^ b).bit_count())
        for a, b, c in combinations(basis, 3):
            best = min(best, (a ^ b ^ c).bit_count())
        if best <= 5:
            break
    return best


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _retired_dic7_context(rng, G, idx):
    """Uncalled historical Dic7 catalogue; never publishes new cells.

    Selection is by inequivalent graph structure, not twenty independent
    draws or twenty order cards. No mutable cross-slot cache is used.
    """
    from itertools import combinations, combinations_with_replacement
    from math import gcd

    assert G.q == 28
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def graph_signature(hx, hz):
        """Invariant of the complete uncoloured bipartite Tanner graph.

        Check intersections and their common-neighbour profiles are
        unchanged by arbitrary permutations of checks and qubits.
        Unequal signatures prove non-isomorphism. Equality is NOT an
        isomorphism certificate and is used only to reject conservatively.
        """
        rows = list(hx) + list(hz)
        m = len(rows)
        one, many = [0] * m, [0] * m
        overlaps = []
        for i in range(m):
            for j in range(i):
                c = (rows[i] & rows[j]).bit_count()
                overlaps.append((i, j, c))
                if c:
                    relation = one if c == 1 else many
                    relation[i] |= 1 << j
                    relation[j] |= 1 << i

        profiles = [{} for _ in rows]
        for i, j, c in overlaps:
            key = (c,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & many[j]).bit_count()
                   + (many[i] & one[j]).bit_count(),
                   (many[i] & many[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
        return tuple(sorted(
            (rows[i].bit_count(), tuple(sorted(p.items())))
            for i, p in enumerate(profiles)
        ))

    # All thirteen distinct representatives in the reported parent table.
    # Each packet encodes its six binomial cells by exact element indices:
    # three A pairs followed by three B pairs; both pivots are identity.
    # These are exclusion references only, never candidate outputs.
    archived = (
        (5,25, 1,17, 9,26, 3,22, 7,20, 5,14),
        (1,21, 0,14, 21,27, 6,22, 6,21, 14,26),
        (1,26, 7,14, 18,19, 10,21, 12,15, 16,19),
        (13,25, 15,27, 17,27, 11,23, 19,25, 16,26),
        (6,21, 15,21, 8,11, 9,21, 11,13, 2,11),
        (24,26, 16,22, 0,4, 15,17, 6,10, 2,10),
        (0,8, 0,2, 0,9, 15,25, 3,11, 14,27),
        (8,20, 1,17, 13,23, 4,21, 16,20, 16,21),
        (13,17, 9,22, 2,18, 26,27, 1,12, 4,9),
        (9,18, 5,24, 20,24, 10,19, 0,5, 1,4),
        (7,15, 7,19, 13,15, 9,22, 3,22, 15,23),
        (12,26, 5,7, 17,25, 2,8, 0,2, 19,23),
        (8,18, 0,15, 12,16, 12,23, 11,26, 12,21),
    )
    graph_keys = set()
    for packet in archived:
        sides = []
        for offset in (0, 6):
            row = [[G.e]]
            for p in range(offset, offset + 6, 2):
                row.append([G.elements[packet[p]],
                            G.elements[packet[p + 1]]])
            sides.append([row])
        hx, hz, _ = lifted_matrices(G, sides[0], sides[1])
        graph_keys.add(graph_signature(hx, hz))

    # Thirteen inverse-pair directions: six noncentral rotations and
    # seven order-four directions. The involution would create duplicate
    # component columns, and is therefore excluded.
    directions = ([G.elements[a] for a in range(1, 7)]
                  + [G.elements[14 + a] for a in range(7)])
    triples = list(combinations(range(len(directions)), 3))

    def inverse_key(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    # All Dic7 automorphisms x -> x^u, y -> x^v y.
    maps = []
    for u in range(14):
        if gcd(u, 14) != 1:
            continue
        for v in range(14):
            maps.append(tuple(
                inverse_key(((u * a + v * b) % 14, b))
                for a, b in directions
            ))

    def orbit_key(a, b):
        return min(tuple(sorted((
            tuple(sorted(mapping[t] for t in a)),
            tuple(sorted(mapping[t] for t in b))
        ))) for mapping in maps)

    def component(triple):
        return [[[G.e]] +
                [[G.e, directions[t]] for t in triple]]

    local = random.Random(BASE_SEED ^ 0xD1C703)
    plans = list(combinations_with_replacement(range(len(triples)), 2))
    local.shuffle(plans)
    orbit_keys = set()
    accepted = 0

    for ia, ib in plans:
        ta, tb = triples[ia], triples[ib]
        orbit = orbit_key(ta, tb)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A, B = component(ta), component(tb)
        hx, hz, n = lifted_matrices(G, A, B)
        signature = graph_signature(hx, hz)
        if signature in graph_keys:
            continue
        if not (_dic_excludes_two(hx, n)
                and _dic_excludes_two(hz, n)):
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 9 * G.q or 1000 * k <= 525 * n:
            continue

        # Validate the explicit witness against the evaluator's exact
        # matrices; do not infer quantum distance from a component graph.
        witness = _dic_three_witness(G, B)
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue

        graph_keys.add(signature)
        if accepted == idx:
            if not css_ok(hx, hz):
                raise ValueError("verified lift failed CSS screening")
            return {"A": A, "B": B, "k": k,
                    "distance_certificate": 3,
                    "directions": (ta, tb), "witness": witness}
        accepted += 1

    raise ValueError("archive-disjoint certified catalogue exhausted")


from functools import lru_cache


_Q48_SUPPORTS = (
    ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
    ((3, 0, 1, 1, 1), (0, 3, 1, 1, 1)),
    ((2, 0, 2, 1, 1), (1, 3, 0, 1, 1)),
    ((2, 2, 0, 1, 1), (1, 0, 3, 1, 1)),
)


def _q48_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in basis:
                basis[p] = word
                break
            word ^= basis[p]
    return basis


def _q48_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        if p not in basis:
            return True
        word ^= basis[p]
    return False


def _q48_clean4(rows, width):
    """Exact exclusion of all nonzero kernel supports through weight four."""
    cols = _dic_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _q48_probe(rows, width, local, passes, own=None, seed=0):
    """Actual kernel/logical witnesses; never a lower distance bound."""
    cols = _dic_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            if own is None or _q48_outside(word, own):
                best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
        local.shuffle(order)
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

        # Elementary exchanges preserve a basis of the same kernel.
        short = sorted(kernel, key=int.bit_count)[:28]
        for i, word in enumerate(kernel):
            for other in short:
                trial = word ^ other
                if 0 < trial.bit_count() < word.bit_count():
                    word = trial
            kernel[i] = word
            inspect(word)
        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _q48_outside(best, own)
    return best


def _q48_component(G, grid, side):
    """Constituent extracted using the evaluator's exact action."""
    unit = [[[G.e]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    return [row & mask for row in rows], width


def _q48_key(hx, hz, width):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive."""
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _dic_columns(rows, width)
    m = len(rows)
    incidence = [[] for _ in range(m + width)]
    adjacency = [0] * m
    for j, col in enumerate(cols):
        while col:
            bit = col & -col
            i = bit.bit_length() - 1
            incidence[i].append(m + j)
            incidence[m + j].append(i)
            col ^= bit
    for i, row in enumerate(rows):
        for j in range(i):
            if row & rows[j]:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    colors = [len(neighbors) for neighbors in incidence]
    history = []
    for _ in range(5):
        features = [
            (colors[v], tuple(sorted(colors[t] for t in neighbors)))
            for v, neighbors in enumerate(incidence)]
        history.append(tuple(sorted(Counter(features).items())))
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]

    roots = []
    for root in range(m):
        visited = frontier = 1 << root
        shells = []
        while frontier:
            neighbors = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                neighbors |= adjacency[bit.bit_length() - 1]
                bits ^= bit
            frontier = neighbors & ~visited
            if frontier:
                shells.append(frontier.bit_count())
                visited |= frontier
        roots.append((rows[root].bit_count(), tuple(shells)))
    signature = (tuple(history), tuple(sorted(roots)))
    return sha256(repr(signature).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q48_portfolio(G, count):
    """Immutable portfolio; memoization changes cost, not slot semantics."""
    from itertools import combinations

    assert G.q == 48
    q, e = G.q, G.e
    cap = min(9, W_MAX)
    orders = {g: _dic_order(G, g) for g in G.elements}
    ratios = {
        0: [g for g in G.elements if orders[g] == 16],
        1: [g for g in G.elements if orders[g] == 8],
        2: [g for g in G.elements if orders[g] in (3, 6, 12)],
        3: [g for g in G.elements if orders[g] >= 8],
        4: [g for g in G.elements if orders[g] > 2],
    }
    triangles = [[] for _ in range(5)]
    for g, h in combinations(G.elements[1:], 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        spectrum = tuple(sorted(
            (orders[g], orders[h], orders[G.mul(G.inv(g), h)])))
        triangle = (e, g, h)
        triangles[4].append(triangle)
        if 16 in spectrum:
            triangles[0].append(triangle)
        if 8 in spectrum:
            triangles[1].append(triangle)
        if any(x in (3, 6, 12) for x in spectrum):
            triangles[2].append(triangle)
        if min(spectrum) >= 4:
            triangles[3].append(triangle)
    for policy in range(4):
        if not triangles[policy]:
            triangles[policy] = list(triangles[4])

    def draw(local, support, policy):
        grid = []
        for counts in support:
            row = []
            for terms in counts:
                anchor = G.pick(local)
                if terms == 0:
                    cell = []
                elif terms == 1:
                    cell = [anchor]
                elif terms == 2:
                    relative = local.choice(ratios[policy])
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    cell = [G.mul(anchor, g)
                            for g in local.choice(triangles[policy])]
                assert len(set(cell)) == terms
                row.append(cell)
            grid.append(row)
        return grid

    def bank(local, support, side, policy):
        records, fillings = [], set()
        for trial in range(32):
            grid = draw(local, support, policy)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows, width = _q48_component(G, grid, side)
            if not _q48_clean4(rows, width):
                continue
            word = _q48_probe(rows, width, local, 2)
            if not word:
                continue
            quality = (word.bit_count(), -_four_cycles(rows))
            records.append((quality, trial, grid, word))
        records.sort(key=lambda r: (r[0], -r[1]), reverse=True)
        return records[:3]

    # Exclusion reference only: the shown order-48 destination incumbent.
    reference_A = [[[(1, 2), (1, 9), (2, 11)],
                    [(2, 7), (1, 9), (0, 15)]]]
    reference_B = [
        [[e, (2, 1)], [(0, 5)], [e], [e], [e]],
        [[(2, 2)], [(1, 7), (2, 14)], [e], [(0, 5)], [(1, 12)]],
    ]
    hx, hz, n = lifted_matrices(G, reference_A, reference_B)
    excluded = {_q48_key(hx, hz, n)}
    results = []

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def quantum_word(record, kind, checks, stabilizers, own, local):
        """Embed verified constituent words and search both LP sectors."""
        source = record["b"] if kind == "X" else record["a"]
        best = 0
        for block in range(2 if kind == "X" else 5):
            if kind == "X":
                word = source[3] << (block * 5*q)
            else:
                word, bits = 0, source[3]
                while bits:
                    bit = bits & -bits
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j*5 + block)*q + t)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _q48_outside(word, own):
                continue
            if not best or word.bit_count() < best.bit_count():
                best = word

        best = _q48_probe(checks, record["n"], local, 2, own, best)
        if not best:
            return 0
        for _ in range(3):
            previous = best
            for row in stabilizers:
                trial = best ^ row
                if trial.bit_count() < best.bit_count():
                    best = trial
            starting = best
            moves = [row for row in stabilizers
                     if (row & starting).bit_count() >= 2]
            for i, first in enumerate(moves):
                for second in moves[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < best.bit_count():
                        best = trial
            if best == previous:
                break
        assert best and _q48_outside(best, own)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
        return best

    for idx in range(count):
        support = _Q48_SUPPORTS[(idx % 20) // 5]
        policy = idx % 5
        sa = ((3, 3),)
        assert _no_dead_lines(sa, support)
        assert _grid_weights(sa, support) == 9 <= cap
        assert max(map(sum, sa)) + max(
            sum(row[j] for row in support) for j in range(5)) == 9
        assert max(map(sum, support)) + 3 == 9

        winner = None
        for restart in range(3):
            local = random.Random(
                BASE_SEED * 10007 + 0x481225
                + 65537*idx + 1000003*restart)
            aa = bank(local, sa, "A", policy)
            bb = bank(local, support, "B", (policy + 1) % 5)
            proposals = sorted(
                ((min(a[0][0], b[0][0]), a, b)
                 for a in aa for b in bb),
                key=lambda r: r[0], reverse=True)
            survivors = []
            for _, a, b in proposals:
                hx, hz, n = lifted_matrices(G, a[2], b[2])
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 576 or not (225*n <= 1000*k < 275*n):
                    continue
                if not (_q48_clean4(hx, n) and _q48_clean4(hz, n)):
                    continue
                signature = _q48_key(hx, hz, n)
                if signature in excluded:
                    continue
                record = dict(a=a, b=b, n=n)
                xword = quantum_word(
                    record, "X", hz, hx, _q48_basis(hx), local)
                zword = quantum_word(
                    record, "Z", hx, hz, _q48_basis(hz), local)
                if not xword or not zword:
                    continue
                reading = min(xword.bit_count(), zword.bit_count())
                quality = (k*reading*reading/n, reading, a[0], b[0])
                survivors.append((quality, a[2], b[2], k,
                                  reading, signature, hx, hz))
                if len(survivors) >= 4:
                    break
            if survivors:
                winner = max(survivors, key=lambda r: r[0])
                break

        if winner is None:
            results.append(None)
            continue
        quality, A, B, k, reading, signature, hx, hz = winner
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= cap
        assert lifted_k(G, A, B) == k
        excluded.add(signature)
        results.append((freeze(A), freeze(B), k, reading,
                        (idx % 20, idx // 20), quality))
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an independently optimized design card."""
    entry = _q48_portfolio(G, N_SLOTS)[idx]
    if entry is None:
        raise ValueError("target-band cubic quantum design card exhausted")
    A, B, k, reading, card, quality = entry
    return {"A": A, "B": B, "screen_k": k, "card": card,
            "logical_witness_upper_bound": reading, "quality": quality}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct relative-order cards provide distinct Dic7 codes: the measured sixty-slot parent collapsed to thirteen distinct codes, with multiple primary cards marked iso_dup.
# REFUTED: The normalized Dic7 direction catalogue supplies twenty evaluator-distinct classes: its measured sixty slots retained fourteen classes, and all five measured distances remained three.
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
