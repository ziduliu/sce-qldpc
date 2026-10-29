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
CONCEPT = ("Metacyclic C3 semidirect C16 with inversion action, order 48, A=(1,2), B=(2,5): internal-edge and hybrid degree-two B cores, paired with independently optimized cubic A constituents, should supply novel cell-0.250 finalists with measured distance readings at least nine.")
GROUP = ("Keep the metacyclic construction rule, changing parameters to m=3,r=16,s=2 because factor twelve excludes the parent's order 55. Inversion has order two dividing sixteen, so this is a valid semidirect product of order 48 and n=576. The measured C3:C16 target-cell bed reached fitness 20.9252; its long-order elements motivate new core topologies, not a distance guarantee.")
SHAPES = ("Change to A=(1,2), B=(2,5): n=576, kb=144 and guaranteed rate 3/12=0.250. The size window permits q<=49; q=48 fits. Constituent variables/checks are 96/48 and 240/96. Parent weight-five minima occupied sector-1 Acol/Bcol blocks broadly, sometimes all A columns at Bcol2; cubic A columns and reinforced two-row B systems replace that binomial bottleneck. Verified full ranks enforce [0.225,0.275).")
FILLING = ("Twenty optimized cards combine four cubic relative-element policies with matching, balanced-internal and hybrid-internal B cores. A counts are (3,3); B rows each contain six terms and columns at most three, hence BOTH CSS maxima are nine. Core girth and distances, constituent ranks, exact small-kernel gates, and explicit refined logical witnesses all use verified lifts. A screening through eight exploits constituent translations only after checking them on the actual matrix. B two/four-cubic join tests are restricted witness searches, not exhaustive distance certificates. Full stabilizer-aware logical-four screening and distinct intrinsic uncoloured Tanner invariants gate publication. No archived cells seed this portfolio; unseen archive novelty and distance-nine survival remain evaluator-tested hypotheses.")
FAMILY_NAME = "r025_metacyclic3by16_inv_12x25_hybridcore_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # Same metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    m, r, s = 3, 16, 2
    if not valid_q(q) or m*r != q or pow(s, r, m) != 1:
        raise ValueError("invalid metacyclic order/action")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _kernel_profile(rows, n):
    """Exact low-weight kernel screen, not a distance-search estimate.

    A reading of five means only that weights one through four were
    excluded. Equal syndromes of words of weight <=2 detect all such
    kernel words. Collision counts are only a secondary selection feature.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= check
            row ^= low
    seen = {0: 0}
    lightest, collisions = 5, 0

    def inspect(syndrome, word):
        nonlocal lightest, collisions
        if syndrome in seen:
            difference = word ^ seen[syndrome]
            if difference:
                lightest = min(lightest, difference.bit_count())
                collisions += 1
        else:
            seen[syndrome] = word

    for j, syndrome in enumerate(columns):
        inspect(syndrome, 1 << j)
    for j, syndrome in enumerate(columns):
        for i in range(j):
            inspect(syndrome ^ columns[i], (1 << j) | (1 << i))
    return lightest, collisions


def _component_rows(G, side, grid):
    """Expose a shape-independent component using the verified lift."""
    dummy = [[[G.e]]]
    width = len(grid[0]) * G.q
    checks = len(grid) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:checks]]


def _generator_key(G, generators):
    """Quotient known component orbit copies, not arbitrary graph isos.

    Affine-group automorphisms have x->x^u, y->x^v*y. Binomial
    generators can also be inverted and their columns permuted.
    """
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    keys = []
    for u in range(1, 11):
        for v in range(11):
            images = []
            for a, b in generators:
                g = ((u * a + v * sums[b]) % 11, b)
                images.append(min(G.idx(g), G.idx(G.inv(g))))
            keys.append(tuple(sorted(images)))
    return min(keys)


def _component_bank(G):
    """Deterministically optimize different quotient prescriptions."""
    local = random.Random(BASE_SEED ^ 0x550403)
    # Quotient-coordinate profiles distinguish construction prescriptions.
    # Avoid two rotations, which commute and immediately create a 4-word.
    profiles = ((0, 1, 2), (0, 1, 1), (1, 1, 2),
                (1, 2, 2), (0, 2, 2), (1, 2, 3))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    entries, used = [], set()
    for attempt in range(256):
        profile = profiles[attempt % len(profiles)]
        generators = [local.choice(pools[b]) for b in profile]
        undirected = [min(G.idx(g), G.idx(G.inv(g)))
                      for g in generators]
        if len(set(undirected)) != 3:
            continue
        key = _generator_key(G, generators)
        if key in used:
            continue
        used.add(key)
        A = [[[G.e, g] for g in generators]]
        reading, collisions = _kernel_profile(
            _component_rows(G, "A", A), 3 * G.q)
        entries.append(((reading, -collisions), key, A))
    if not entries:
        raise ValueError("component bank unexpectedly empty")
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Prefer every available clean component. If the hypothesis fails,
    # retain the best legal components instead of invalidating all slots.
    clean = [item for item in entries if item[0][0] == 5]
    return (clean if clean else entries[:12])[:16]


def _draw_B(rng, G, attempt, bank):
    """Indexed support/order prescriptions, subsequently screened."""
    if attempt % 4 == 0:
        source = bank[(attempt // 4) % len(bank)][2]
        # Independent samples of quotients, not copies of source labels.
        profile = [cell[1][1] for cell in source[0]]
        cells = []
        for b in profile:
            pool = [g for g in G.elements if g != G.e and g[1] == b]
            cells.append([G.e, rng.choice(pool)])
        return [cells]

    # One degree-three column and two degree-two columns: total seven.
    # Vary the trinomial's quotient-coordinate pattern deliberately.
    patterns = ((0, 1), (0, 2), (1, 1), (1, 2),
                (1, 3), (2, 2), (2, 3), (2, 4))
    b0, b1 = patterns[(attempt // 4) % len(patterns)]
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    g = rng.choice(pools[b0])
    h = rng.choice(pools[b1])
    while h == g:
        h = rng.choice(pools[b1])
    cells = [[G.e, g, h]]
    for j in range(2):
        b = (attempt // (8 + 3 * j) + j) % 5
        cells.append([G.e, rng.choice(pools[b])])
    return [cells]


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


from functools import lru_cache


def _s25_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _s25_basis(rows):
    out = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            if p not in out:
                out[p] = word
                break
            word ^= out[p]
    return out


def _s25_outside(word, own):
    while word:
        p = word.bit_length()-1
        if p not in own:
            return True
        word ^= own[p]
    return False


def _s25_clean4(rows, width):
    """Exactly exclude nonzero kernel supports of weights one to four."""
    cols = _s25_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for j, first in enumerate(cols):
        for i in range(j):
            syndrome = first ^ cols[i]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _s25_clean6(rows, width):
    """Exact component screen, using all supports of size at most three."""
    cols = _s25_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for j, first in enumerate(cols):
        for i in range(j):
            s = first ^ cols[i]
            if s in singles or s in pairs:
                return False
            pairs.add(s)
    triples = set()
    for j, first in enumerate(cols):
        for i in range(j):
            pair = first ^ cols[i]
            for h in range(i):
                s = pair ^ cols[h]
                if not s or s in singles or s in pairs or s in triples:
                    return False
                triples.add(s)
    return True


def _s25_A8(G, rows):
    """Exclude A kernel supports through eight; return a found witness.

    Right regular translations preserve the left-lifted A constituent.
    Verify the two generating translations before using anchored quartets.
    Every eight-support has a translate containing a block identity.
    """
    from itertools import combinations
    from collections import Counter

    width, q = 2*G.q, G.q
    for step in ((1, 0), (0, 1)):
        permutation = [
            block*q + G.idx(G.mul(g, step))
            for block in range(2) for g in G.elements
        ]
        moved = []
        for row in rows:
            word = 0
            while row:
                bit = row & -row
                word |= 1 << permutation[bit.bit_length()-1]
                row ^= bit
            moved.append(word)
        if Counter(moved) != Counter(rows):
            raise ValueError("A regular translation failed verification")

    cols = _s25_columns(rows, width)
    bits = [1 << j for j in range(width)]
    seen = {}
    for anchor in (G.idx(G.e), q+G.idx(G.e)):
        others = [j for j in range(width) if j != anchor]
        for a, b, c in combinations(others, 3):
            s = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
            word = bits[anchor] | bits[a] | bits[b] | bits[c]
            old = seen.get(s)
            if old is not None and old != word:
                return False, old ^ word
            seen[s] = word
    for a, b, c, d in combinations(range(width), 4):
        s = cols[a] ^ cols[b] ^ cols[c] ^ cols[d]
        old = seen.get(s)
        if old is not None:
            word = bits[a] | bits[b] | bits[c] | bits[d]
            if old != word:
                return False, old ^ word
    return True, 0


def _s25_probe(rng, rows, width, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, never an unresolved cap."""
    cols = _s25_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width+1

    def record(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _s25_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            s, word = cols[j], 1 << j
            while s:
                p = s.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s, word)
                    break
                s ^= old[0]
                word ^= old[1]
            if not s:
                kernel.append(word)
                record(word)

        # Exchanges preserve the complete kernel. Limit exchange partners,
        # not the space from which the information set is constructed.
        partners = sorted(kernel, key=int.bit_count)[:24]
        rng.shuffle(kernel)
        for i, word in enumerate(kernel):
            w = word.bit_count()
            for other in partners:
                trial = word ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    word, w = trial, tw
            kernel[i] = word
            record(word)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                record(pair)
                if i < 16:
                    for h in range(j):
                        record(pair ^ short[h])
        for _ in range(2):
            starting = best
            if not starting:
                break
            for i, first in enumerate(short):
                record(starting ^ first)
                for second in short[:i]:
                    record(starting ^ first ^ second)
            if starting == best:
                break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _s25_outside(best, own)
    return best


def _s25_penalty(rows, width):
    """Tanner four-cycle count and check-projection triangle pressure."""
    cols = _s25_columns(rows, width)
    four = sum(
        overlap*(overlap-1)//2
        for i, row in enumerate(rows) for other in rows[:i]
        for overlap in [(row & other).bit_count()])
    adj = [0] * len(rows)
    internal = 0
    for col in cols:
        d = col.bit_count()
        internal += d*(d-1)*(d-2)//6
        bits = col
        while bits:
            bit = bits & -bits
            adj[bit.bit_length()-1] |= col ^ bit
            bits ^= bit
    triangles = 0
    for i, neighbours in enumerate(adj):
        bits = neighbours & ~((1 << (i+1))-1)
        while bits:
            bit = bits & -bits
            j = bit.bit_length()-1
            triangles += ((neighbours & adj[j]) >> (j+1)).bit_count()
            bits ^= bit
    return four, max(0, triangles-internal)


def _s25_logical4(checks, stabilizers, width):
    """Exact logical-four exclusion; short stabilizers are permitted."""
    own = _s25_basis(stabilizers)
    pivots = sorted(own, reverse=True)
    residues = []
    for j in range(width):
        word = 1 << j
        for p in pivots:
            if (word >> p) & 1:
                word ^= own[p]
        residues.append(word)
    cols = _s25_columns(checks, width)
    seen = {0: 0}
    for j, s in enumerate(cols):
        r = residues[j]
        if s in seen:
            if seen[s] != r:
                return False
        else:
            seen[s] = r
    for j, first in enumerate(cols):
        for i in range(j):
            s, r = first ^ cols[i], residues[j] ^ residues[i]
            if s in seen:
                if seen[s] != r:
                    return False
            else:
                seen[s] = r
    return True


def _s25_key(hx, hz, width):
    """Intrinsic uncoloured Tanner invariant, conservatively deduplicated."""
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _s25_columns(rows, width)
    m = len(rows)

    def positions(word):
        out = []
        while word:
            bit = word & -word
            out.append(bit.bit_length()-1)
            word ^= bit
        return out

    adjacency = [[m+j for j in positions(row)] for row in rows]
    adjacency += [positions(col) for col in cols]

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    colors = compress(
        [(0, len(ns)) for ns in adjacency[:m]]
        + [(1, len(ns)) for ns in adjacency[m:]])
    history = []
    for _ in range(5):
        features = [
            (colors[i], tuple(sorted(colors[j] for j in ns)))
            for i, ns in enumerate(adjacency)]
        history.append(tuple(sorted(Counter(features).items())))
        colors = compress(features)
    value = (
        width, m, tuple(history),
        _tanner_signature(rows, []))
    return sha256(repr(value).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _s25_models(G):
    """Relative triangle classes and long-order reinforcement directions."""
    def order(g):
        power = G.e
        for t in range(1, G.q+1):
            power = G.mul(power, g)
            if power == G.e:
                return t
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != G.e]
    banks = [[] for _ in range(4)]
    for g in nonzero:
        for h in nonzero:
            if g == h:
                continue
            spectrum = sorted((
                orders[g], orders[h], orders[G.mul(G.inv(g), h)]))
            commuting = G.mul(g, h) == G.mul(h, g)
            if not commuting and spectrum[0] >= 8:
                banks[0].append((g, h))
            if commuting and spectrum[0] >= 8:
                banks[1].append((g, h))
            if commuting and spectrum[0] >= 4 and spectrum[-1] >= 12:
                banks[2].append((g, h))
            if not commuting and spectrum[0] >= 3:
                banks[3].append((g, h))
    if any(not bank for bank in banks):
        raise ValueError("required cubic relative class absent")
    long = tuple(g for g in nonzero if orders[g] >= 12)
    odd16 = tuple(g for g in nonzero if orders[g] == 16)
    return tuple(tuple(bank) for bank in banks), long, odd16, orders


def _s25_draw_cell(rng, G, count, policy, odd=False):
    if not count:
        return []
    first = G.pick(rng)
    if count == 1:
        return [first]
    banks, long, odd16, _ = _s25_models(G)
    if count == 2:
        direction = rng.choice(odd16 if odd and odd16 else long)
        return [first, G.mul(first, direction)]
    g, h = rng.choice(banks[policy % 4])
    cell = [first, G.mul(first, g), G.mul(first, h)]
    assert len({G._key(x) for x in cell}) == count == 3
    return cell


def _s25_core_data(G, grid):
    """Exact graph girth and all check distances from a VERIFIED component."""
    rows = _component_rows(G, "B", grid)
    cols = _s25_columns(rows, 3*G.q)
    adj = [[] for _ in rows]
    edges = set()
    for edge, col in enumerate(cols):
        if col.bit_count() != 2:
            return 0, ()
        bit = col & -col
        a, b = bit.bit_length()-1, (col ^ bit).bit_length()-1
        key = (min(a, b), max(a, b))
        if key in edges:
            return 2, ()
        edges.add(key)
        adj[a].append((b, edge))
        adj[b].append((a, edge))

    girth, distances = len(rows)+1, []
    for root in range(len(rows)):
        distance = [10000] * len(rows)
        parent = [-1] * len(rows)
        distance[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adj[a]:
                if distance[b] == 10000:
                    distance[b] = distance[a]+1
                    parent[b] = edge
                    queue.append(b)
                elif edge != parent[a] and edge != parent[b]:
                    girth = min(girth, distance[a]+distance[b]+1)
        if girth < 8:
            return girth, ()
        distances.append(tuple(distance))
    return girth, tuple(distances)


@lru_cache(maxsize=1)
def _s25_core_bank(G):
    """Enumerate three genuinely different degree-two core topologies."""
    _, _, _, orders = _s25_models(G)
    steps = [g for g in G.elements if orders[g] >= 6]
    representatives = [
        g for g in steps
        if G.idx(g) == min(
            G.idx(G.mul(G.mul(t, g), G.inv(t)))
            for t in G.elements)]
    result = []
    for mode in range(3):
        records = []
        for g in representatives:
            for h in steps:
                if g == h:
                    continue
                e = G.e
                if mode == 0:
                    core = [[[e], [e], [e]], [[e], [g], [h]]]
                elif mode == 1:
                    core = [[[e, g], [], [e]], [[], [e, h], [e]]]
                else:
                    core = [[[e, g], [e], [e]], [[], [e], [h]]]
                girth, distances = _s25_core_data(G, core)
                if girth < 8:
                    continue
                frozen = tuple(tuple(tuple(c) for c in row) for row in core)
                records.append((girth, G.idx(g), G.idx(h), frozen))
        records.sort(key=lambda r: (-r[0], r[1], r[2]))
        selected = []
        for girth, _, _, core in records[:24]:
            verified_girth, distances = _s25_core_data(G, core)
            assert verified_girth == girth
            selected.append((core, girth, distances))
        result.append(tuple(selected))
    if not result[0]:
        raise ValueError("no verified long-girth matching core")
    return tuple(result)


def _s25_join_reading(G, rows, distances, girth, four=False):
    """Restricted B witness search with zero, two or four cubic variables.

    Minimum graph T-joins are metric perfect matchings. Six or more
    cubic variables are NOT screened, and no complete-LP symmetry is used.
    """
    from itertools import combinations
    cols = _s25_columns(rows, 5*G.q)
    cubic = [j for j, col in enumerate(cols) if col.bit_count() == 3]
    anchors = [
        block*G.q + G.idx(G.e) for block in range(5)
        if cols[block*G.q + G.idx(G.e)].bit_count() == 3]
    assert len(cubic) == 2*G.q and len(anchors) == 2
    memo = {}

    def join(boundary, budget):
        if not boundary:
            return 0
        if budget < 0 or boundary.bit_count() > 2*budget:
            return budget+1
        key = boundary, budget
        if key in memo:
            return memo[key]
        bit = boundary & -boundary
        a = bit.bit_length()-1
        rest = boundary ^ bit
        choices, best = rest, budget+1
        while choices:
            other = choices & -choices
            b = other.bit_length()-1
            d = distances[a][b]
            if d <= budget and d < best:
                best = min(best, d+join(rest ^ other, budget-d))
            choices ^= other
        memo[key] = best
        return best

    best = girth
    for anchor in anchors:
        for other in cubic:
            if other != anchor:
                best = min(best, 2+join(cols[anchor] ^ cols[other], best-3))
    if four and best >= 9:
        for anchor in anchors:
            others = [j for j in cubic if j != anchor]
            for a, b, c in combinations(others, 3):
                boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                budget = best-5
                if boundary.bit_count() <= 2*budget:
                    best = min(best, 4+join(boundary, budget))
                    if best < 9:
                        return best
    return best


@lru_cache(maxsize=1)
def _s25_A_bank(G):
    """Independently refine cubic A constituents with equal final budgets."""
    local = random.Random(BASE_SEED ^ 0x4825A1)
    raw = []
    for trial in range(384):
        policy = trial % 4
        A = [[_s25_draw_cell(local, G, 3, policy),
              _s25_draw_cell(local, G, 3, (policy+trial//16) % 4)]]
        rows = _component_rows(G, "A", A)
        if not _s25_clean4(rows, 2*G.q):
            continue
        rank = gf2_rank(rows)
        if rank < G.q-6:
            continue
        penalty = _s25_penalty(rows, 2*G.q)
        raw.append((penalty, -min(G.q-rank, 6), trial, A, rows, rank))
    raw.sort(key=lambda r: (20*r[0][0]+r[0][1], r[1], r[2]))

    shortlist = []
    for penalty, _, trial, A, rows, rank in raw[:72]:
        if not _s25_clean6(rows, 2*G.q):
            continue
        word = _s25_probe(local, rows, 2*G.q, 4)
        if word:
            shortlist.append((word.bit_count(), penalty, trial,
                              A, rows, rank, word))
    shortlist.sort(key=lambda r: (
        -(144+2*(G.q-r[5]))*min(r[0], 12)**2, r[1], r[2]))

    bank = []
    for _, penalty, trial, A, rows, rank, word in shortlist[:16]:
        clean8, witness = _s25_A8(G, rows)
        if witness and witness.bit_count() < word.bit_count():
            word = witness
        word = _s25_probe(local, rows, 2*G.q, 24, word)
        assert word and word.bit_count() >= 8
        bank.append({
            "A": A, "rows": rows, "rank": rank, "word": word,
            "reading": word.bit_count(), "clean8": clean8,
            "penalty": penalty, "policy": trial % 4})
    bank.sort(key=lambda a: (
        (144+2*(G.q-a["rank"]))*min(a["reading"], 12)**2,
        a["clean8"], -a["penalty"][0], -a["penalty"][1]),
        reverse=True)
    if not bank:
        raise ValueError("no screened cubic A constituent")
    return tuple(bank)


def _s25_logical_witness(G, a, b, hx, hz, n, rng):
    """Embed, verify and refine actual nontrivial X and Z witnesses."""
    q, best = G.q, n+1
    for kind, checks, stabilizers in (("X", hz, hx), ("Z", hx, hz)):
        own = _s25_basis(stabilizers)
        seeds = []
        source = b["word"] if kind == "X" else a["word"]
        for block in range(2 if kind == "X" else 5):
            if kind == "X":
                word = source << (block*5*q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*5+block)*q+h)
                    bits ^= bit
            if any((row & word).bit_count() % 2 for row in checks):
                raise ValueError("verified component embedding failed")
            if _s25_outside(word, own):
                seeds.append(word)

        reduced = []
        for word in seeds:
            for _ in range(2):
                previous = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting = word
                moves = sorted(
                    (row for row in stabilizers if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:40]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == previous:
                    break
            reduced.append(word)
        seed = min(reduced, key=int.bit_count) if reduced else 0
        word = _s25_probe(rng, checks, n, 3, seed, own)
        if not word:
            return None
        best = min(best, word.bit_count())
    return best


def _s25_design(rng, G, position, restart, excluded):
    """Optimize one topology/policy card, not one completed random draw."""
    card = position % 20
    topology = card % 5
    mode, variant = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))[topology]
    cores = _s25_core_bank(G)
    if not cores[mode]:
        mode, variant = 0, card % 2
    if mode == 2:
        profile = ((2, 1, 1, 0, 2), (0, 1, 1, 3, 1))
    elif mode == 1:
        profile = (((2, 0, 1, 2, 1), (0, 2, 1, 1, 2))
                   if variant == 0 else
                   ((2, 0, 1, 3, 0), (0, 2, 1, 0, 3)))
    else:
        profile = (((1, 1, 1, 2, 1), (1, 1, 1, 1, 2))
                   if variant == 0 else
                   ((1, 1, 1, 3, 0), (1, 1, 1, 0, 3)))
    assert _no_dead_lines([[3, 3]], profile)
    assert _grid_weights([[3, 3]], profile) == 9

    raw = []
    for trial in range(112):
        core, girth, distances = cores[mode][
            (card//5+restart+trial//8) % len(cores[mode])]
        B = [[list(cell) for cell in row] for row in core]
        for i in range(2):
            for j in (3, 4):
                policy = (card//5+i+j+trial//24) % 4
                B[i].append(_s25_draw_cell(
                    rng, G, profile[i][j], policy, odd=card % 2 == 0))
        rows = _component_rows(G, "B", B)
        if not _s25_clean4(rows, 5*G.q):
            continue
        reading = _s25_join_reading(G, rows, distances, girth)
        penalty = _s25_penalty(rows, 5*G.q)
        raw.append((reading, penalty, trial, B, rows, girth, distances))
    raw.sort(key=lambda r: (-r[0], 20*r[1][0]+r[1][1], r[2]))

    bb = []
    for reading, penalty, trial, B, rows, girth, distances in raw[:10]:
        restricted = _s25_join_reading(G, rows, distances, girth, four=True)
        word = _s25_probe(rng, rows, 5*G.q, 12)
        if not word:
            continue
        estimate = min(restricted, word.bit_count())
        bb.append({
            "B": B, "word": word, "reading": estimate,
            "penalty": penalty, "girth": girth,
            "restricted_join": restricted, "trial": trial})
    bb.sort(key=lambda b: (
        b["reading"], -b["penalty"][0], -b["penalty"][1]), reverse=True)
    if not bb:
        return None

    aa = list(_s25_A_bank(G))
    preferred = card//5
    aa.sort(key=lambda a: (
        a["policy"] == preferred, a["clean8"],
        (144+2*(G.q-a["rank"]))*min(a["reading"], 12)**2),
        reverse=True)
    plans = []
    for a in aa[:6]:
        for b in bb[:4]:
            A, B = a["A"], b["B"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 576 or not (225*n <= 1000*k < 275*n):
                continue
            reading = min(a["reading"], b["reading"], 12)
            plans.append((k*reading*reading/n, a, b, hx, hz, n, k))
    plans.sort(key=lambda r: r[0], reverse=True)

    survivors, local_keys = [], set()
    for preliminary, a, b, hx, hz, n, k in plans[:10]:
        if not _s25_logical4(hz, hx, n) or not _s25_logical4(hx, hz, n):
            continue
        signature = _s25_key(hx, hz, n)
        if signature in excluded or signature in local_keys:
            continue
        local_keys.add(signature)
        witness = _s25_logical_witness(G, a, b, hx, hz, n, rng)
        if witness is None:
            continue
        reading = min(witness, a["reading"], b["reading"], 12)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= min(9, W_MAX)
        assert lifted_k(G, a["A"], b["B"]) == k
        quality = (k*reading*reading/n, reading, k, preliminary)
        screen = (
            ("k", k), ("logical_witness_upper_bound", witness),
            ("component_witness_upper_bounds",
             (a["word"].bit_count(), b["word"].bit_count())),
            ("restricted_B_join_reading", b["restricted_join"]),
            ("A_kernel8_excluded", a["clean8"]),
            ("B_core_girth", b["girth"]),
            ("design_card", (card, mode, variant, restart)),
            ("complete_logical_exclusion_through", 4))
        survivors.append((quality, signature, a["A"], b["B"], screen))
        if len(survivors) >= 4:
            break
    return max(survivors, key=lambda r: r[0]) if survivors else None


@lru_cache(maxsize=1)
def _s25_portfolio(G):
    """Immutable, call-order-independent portfolio with intrinsic exclusions."""
    assert G.q == 48 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    _s25_core_bank(G)
    _s25_A_bank(G)
    entries, excluded = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for position in range(N_SLOTS):
        winner = None
        for restart in range(3):
            local = random.Random(
                BASE_SEED*10007 + 0x4825C1
                + 65537*position + 1000003*restart)
            candidate = _s25_design(local, G, position, restart, excluded)
            if candidate is not None:
                winner = candidate
                break
        if winner is None:
            entries.append(None)
            continue
        quality, signature, A, B, screen = winner
        excluded.add(signature)
        entries.append((freeze(A), freeze(B), screen, quality))
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Publish a complete screened filling; no cell-level resampling."""
    entry = _s25_portfolio(G)[idx]
    if entry is None:
        raise ValueError("target-cell hybrid-core card exhausted")
    A, B, screen, quality = entry
    return {"A": A, "B": B, "screen": dict(screen),
            "selection_quality": quality}


from functools import lru_cache


@lru_cache(maxsize=1)
def _homology5_portfolio(G):
    """Optimize once; return only immutable, invariant-distinct fillings.

    Constituent short words are proposal features, not quantum vetoes.
    All acceptance and distance-floor calculations use verified lifts.
    """
    from itertools import combinations
    from hashlib import sha256

    q, width = G.q, 3 * G.q
    assert q == 55
    local = random.Random(BASE_SEED ^ 0x55A850)

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    pivots[p] = word
                    break
                word ^= pivots[p]
        return pivots

    def logical_four(checks, own, n):
        """Return syndrome representatives iff no logical of weight <=4.

        Canonical residues modulo the stabilizer span distinguish
        harmless short stabilizers from nontrivial logical collisions.
        One representative per syndrome suffices by linearity.
        """
        pivots = basis(own)
        ordered = sorted(pivots, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in ordered:
                if (word >> p) & 1:
                    word ^= pivots[p]
            residues.append(word)

        cols = columns(checks, n)
        seen = {0: (0, 0)}
        for j, syndrome in enumerate(cols):
            residue, word = residues[j], 1 << j
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = (residue, word)
            elif old[0] != residue:
                return None

        for j, first in enumerate(cols):
            bit, residue = 1 << j, residues[j]
            for i in range(j):
                syndrome = first ^ cols[i]
                r = residue ^ residues[i]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = (r, bit | (1 << i))
                elif old[0] != r:
                    return None
        return cols, residues, seen

    def logical_five(screen):
        """Exact weight-five logical test after logical_four passed.

        Every five-support splits into a triple and a pair. This scans
        ALL triples: no transitivity of the nonabelian full LP graph is
        assumed. Zero means exhaustive exclusion, not a missing probe.
        """
        cols, residues, seen = screen
        bits = [1 << j for j in range(len(cols))]
        for j, first in enumerate(cols):
            rj, bj = residues[j], bits[j]
            for i in range(j):
                syndrome = first ^ cols[i]
                residue = rj ^ residues[i]
                word = bj | bits[i]
                for h in range(i):
                    old = seen.get(syndrome ^ cols[h])
                    if old is not None and old[0] != residue ^ residues[h]:
                        witness = word ^ bits[h] ^ old[1]
                        assert witness.bit_count() == 5
                        return witness
        return 0

    def five_count(rows):
        """Exact constituent five-count, or None when a <=4 word exists.

        Opposite regular translations act within each constituent
        column block. They need not act on the complete LP graph.
        """
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return None
        pairs = {}
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs[syndrome] = (i, j)

        incidences = 0
        for block in range(3):
            anchor = block * q + G.idx(G.e)
            hits = 0
            for syndrome, pair in pairs.items():
                if anchor in pair:
                    continue
                other = pairs.get(syndrome ^ cols[anchor])
                if other is not None:
                    assert anchor not in other
                    assert not set(pair).intersection(other)
                    hits += 1
            assert hits % 6 == 0
            incidences += hits // 6
        assert q * incidences % 5 == 0
        return q * incidences // 5

    def signature(hx, hz, n):
        rows = hx + hz
        value = (
            tuple(sorted(row.bit_count() for row in rows)),
            tuple(sorted(c.bit_count() for c in columns(rows, n))),
            _tanner_signature(hx, hz),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def binomials(key, scale=1):
        return [[[G.e, ((scale * G.elements[t][0]) % 11,
                        G.elements[t][1])] for t in key]]

    # Enumerate constituent orbits, but deliberately remove the parent's
    # connectedness and kernel-four-clean restrictions.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != G.e
    })
    orbit_keys = set()
    for anchor in (G.idx((1, 0)), G.idx((0, 1)), G.idx((0, 2))):
        others = [t for t in reps if t != anchor]
        for pair in combinations(others, 2):
            orbit_keys.add(_generator_key(
                G, [G.elements[t] for t in (anchor,) + pair]))

    bank = []
    for key in sorted(orbit_keys):
        grid = binomials(key)
        rows = _component_rows(G, "A", grid)
        reading, collisions = _kernel_profile(rows, width)
        if reading < 3:
            continue
        rank = gf2_rank(rows)
        fives = five_count(rows) if reading == 5 else None
        bank.append((key, grid, reading, rank, fives, collisions))
    if not bank:
        raise ValueError("rank-relaxed binomial bank empty")

    # Lower-rank and short-cycle constituents get actual homology tests
    # before the conventional connected girth-five constituents.
    bank.sort(key=lambda item: (item[3], item[2], item[0]))
    clean = [item for item in bank if item[2] == 5]
    clean.sort(key=lambda item: (
        item[4] if item[4] is not None else width ** 3, item[0]))
    if not clean:
        raise ValueError("no clean constituent for supplementary cards")

    excluded = set()
    known_binomial_pairs = (
        (((9, 0), (2, 2), (3, 2)),
         ((10, 0), (8, 2), (3, 2))),
        (((10, 0), (1, 1), (9, 2)),
         ((10, 0), (10, 1), (1, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((1, 0), (0, 2), (5, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((0, 1), (1, 2), (10, 2))),
        (((1, 0), (0, 2), (5, 2)),
         ((0, 1), (1, 2), (10, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((2, 0), (0, 1), (10, 2))),
    )
    for aa, bb in known_binomial_pairs:
        A = [[[G.e, g] for g in aa]]
        B = [[[G.e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(signature(hx, hz, n))

    filling_keys = set()

    def consider(A, B, pressure):
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            return None
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > min(9, W_MAX)
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or not (375 * n <= 1000 * k < 425 * n):
            return None
        sig = signature(hx, hz, n)
        if sig in excluded:
            return None

        sx = logical_four(hz, hx, n)
        if sx is None:
            return None
        sz = logical_four(hx, hz, n)
        if sz is None:
            return None

        wx = logical_five(sx)
        if wx:
            assert all((row & wx).bit_count() % 2 == 0 for row in hz)
            floor = 5
        else:
            wz = logical_five(sz)
            if wz:
                assert all((row & wz).bit_count() % 2 == 0 for row in hx)
                floor = 5
            else:
                floor = 6
        del sx, sz

        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        quality = (k * floor * floor / float(n), floor, k, -pressure)
        entry = (freeze(A), freeze(B), k, floor, pressure)
        return quality, sig, entry

    # Relative outer orientations are trials, never assumed novelty.
    # Known archived pure-binomial classes have already been excluded.
    primary = []
    primary_signatures = set()
    for a in range(len(bank)):
        for b in range(a, len(bank)):
            for scale in (1, 2):
                afive, bfive = bank[a][4], bank[b][4]
                pressure = sum(
                    value if value is not None else width ** 3
                    for value in (afive, bfive))
                result = consider(bank[a][1],
                                  binomials(bank[b][0], scale), pressure)
                if result is None or result[1] in primary_signatures:
                    continue
                primary_signatures.add(result[1])
                primary.append(result)
    primary.sort(key=lambda item: item[0], reverse=True)

    entries = []
    for quality, sig, entry in primary[:N_SLOTS]:
        excluded.add(sig)
        entries.append(entry)

    # Twenty optimization cards: ten intrinsic trinomial quotient
    # patterns, each with two reinforcement prescriptions. Additional
    # epochs change the reinforcement and the multiplicity pressure.
    triangles = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
                 (1, 2), (1, 3), (1, 4), (2, 2), (2, 3))
    pools = [[g for g in G.elements if g != G.e and g[1] == b]
             for b in range(5)]
    component_cache = {}

    while len(entries) < N_SLOTS:
        position = len(entries)
        card, epoch = position % 20, position // 20
        b0, b1 = triangles[card % 10]
        reinforcement = 1 + (card // 10 + epoch) % 4
        winner = None

        for batch in range(4):
            proposals = []
            for trial in range(48):
                g, h = local.choice(pools[b0]), local.choice(pools[b1])
                if g == h:
                    continue
                # Alternate rotation/order-five and two distinct
                # order-five prescriptions; neither is a saved filling.
                c0 = 0 if card % 2 == 0 else reinforcement
                c1 = reinforcement if card % 2 == 0 else 1 + reinforcement % 4
                s, t = local.choice(pools[c0]), local.choice(pools[c1])
                if s == t or s == G.inv(t):
                    continue
                B = [[[G.e, g, h], [G.e, s], [G.e, t]]]
                key = tuple(tuple(sorted(G.idx(x) for x in cell))
                            for cell in B[0])
                profile = component_cache.get(key)
                if profile is None:
                    rows = _component_rows(G, "B", B)
                    reading, collisions = _kernel_profile(rows, width)
                    fives = five_count(rows) if reading == 5 else None
                    profile = reading, collisions, fives
                    component_cache[key] = profile
                reading, collisions, bfive = profile
                if reading < 4:
                    continue

                # Periodically test short-cycle constituents as well;
                # the complete homology gate, not their kernel, decides.
                source = (bank[(position + trial + batch) % len(bank)]
                          if trial % 7 == 0
                          else clean[(position + trial + batch) % len(clean)])
                afive = source[4]
                a_pressure = afive if afive is not None else width ** 3
                b_pressure = bfive if bfive is not None else width ** 3
                weight = 1 + (card + epoch) % 3
                pressure = weight * a_pressure + (4 - weight) * b_pressure
                proposals.append(
                    ((pressure, collisions, trial), source[1], B))
                if len(proposals) >= 10:
                    break
            proposals.sort(key=lambda item: item[0])

            survivors, batch_signatures = [], set()
            for preliminary, A, B in proposals:
                result = consider(A, B, preliminary[0])
                if result is None or result[1] in batch_signatures:
                    continue
                batch_signatures.add(result[1])
                survivors.append(result)
                if len(survivors) >= 3:
                    break
            if survivors:
                candidate = max(survivors, key=lambda item: item[0])
                if winner is None or candidate[0] > winner[0]:
                    winner = candidate
                # A fully excluded weight-five logical space is a real
                # lower bound of six, not a heuristic unresolved cap.
                if winner[2][3] >= 6:
                    break
                # Continue one additional batch to optimize rank and
                # multiplicity without rebuilding the entire catalogue.
                if batch >= 1:
                    break

        if winner is None:
            entries.append(None)
        else:
            quality, sig, entry = winner
            excluded.add(sig)
            entries.append(entry)

    return tuple(entries)


def _legacy_exhaustive_quantum4_reference(rng, G, idx):
    """Inactive reference; the emitted portfolio uses homology-five selection."""
    from itertools import combinations

    local = random.Random(BASE_SEED ^ 0x5504E1)
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            mark = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= mark
                row ^= bit
        return out

    def quantum_clean_four(checks, stabilizers, width):
        """Exact exclusion of nontrivial logical supports of weight <=4.

        Every such support splits into two words of weight <=2.
        For each syndrome, all enumerated words must have the same
        residue modulo the stabilizer space. Short stabilizers are legal.
        """
        basis = {}
        for original in stabilizers:
            word = original
            while word:
                p = word.bit_length() - 1
                if p in basis:
                    word ^= basis[p]
                else:
                    basis[p] = word
                    break
        pivots = sorted(basis, reverse=True)

        def reduce(word):
            for p in pivots:
                if (word >> p) & 1:
                    word ^= basis[p]
            return word

        syndromes = columns(checks, width)
        residues = [reduce(1 << j) for j in range(width)]
        seen = {0: 0}
        for j, syndrome in enumerate(syndromes):
            residue = residues[j]
            if syndrome in seen:
                if residue != seen[syndrome]:
                    return False
            else:
                seen[syndrome] = residue
        for j, syndrome in enumerate(syndromes):
            residue = residues[j]
            for i in range(j):
                s = syndrome ^ syndromes[i]
                r = residue ^ residues[i]
                if s in seen:
                    if r != seen[s]:
                        return False
                else:
                    seen[s] = r
        return True

    def binomials(key):
        return [[[G.e, G.elements[t]] for t in key]]

    def image(key, u, v=0):
        result = []
        for t in key:
            a, b = G.elements[t]
            g = ((u * a + v * sums[b]) % 11, b)
            result.append(min(G.idx(g), G.idx(G.inv(g))))
        return tuple(sorted(result))

    # Every triple can be mapped to one containing one of these anchors:
    # a rotation maps to x; otherwise a generator, inverted if necessary,
    # has quotient coordinate 1 or 2 and its translation can be removed.
    undirected = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if g != G.e
    })
    anchors = [G.idx(g) for g in ((1, 0), (0, 1), (0, 2))]
    orbit_keys = set()
    for anchor in anchors:
        others = [t for t in undirected if t != anchor]
        for pair in combinations(others, 2):
            generators = [G.elements[t] for t in (anchor,) + pair]
            orbit_keys.add(_generator_key(G, generators))

    legacy = {entry[1] for entry in _component_bank(G)}
    bank = []
    for key in sorted(orbit_keys):
        A = binomials(key)
        ar = _component_rows(G, "A", A)
        if _kernel_profile(ar, 3 * G.q)[0] != 5:
            continue
        # Connected component incidence has rank q-1; disconnected
        # graphs are not allowed to win merely through rank deficits.
        if gf2_rank(ar) != G.q - 1:
            continue
        br = _component_rows(G, "B", A)
        if _kernel_profile(br, 3 * G.q)[0] != 5:
            continue
        bank.append(key)
    bank.sort(key=lambda key: (key in legacy, key))
    if not bank:
        raise ValueError("exhaustive binomial bank has no clean component")

    def signature(hx, hz, n):
        cols = columns(hx + hz, n)
        return (
            tuple(sorted(row.bit_count() for row in hx + hz)),
            tuple(sorted(c.bit_count() for c in cols)),
            _tanner_signature(hx, hz),
        )

    graph_keys = set()
    # Explicit archived pure-binomial examples from the current table
    # and the live frontier. This is conservative invariant rejection,
    # not a claim to have access to the entire evaluator archive.
    archived = (
        (((9, 0), (2, 2), (3, 2)),
         ((10, 0), (8, 2), (3, 2))),
        (((10, 0), (1, 1), (9, 2)),
         ((10, 0), (10, 1), (1, 2))),
        (((1, 0), (0, 1), (5, 2)),
         ((1, 0), (0, 2), (5, 2))),
    )
    for aa, bb in archived:
        A = [[[G.e, g] for g in aa]]
        B = [[[G.e, g] for g in bb]]
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(signature(hx, hz, n))

    filling_keys = set()
    accepted = 0

    def inspect(A, B):
        nonlocal accepted
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            return None
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > min(9, W_MAX)
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (375 * n <= 1000 * k < 425 * n):
            return None
        sig = signature(hx, hz, n)
        if sig in graph_keys:
            return None
        # Reject this entire invariant class after its first inspection.
        # Different accepted signatures certify non-isomorphic graphs.
        graph_keys.add(sig)
        if not quantum_clean_four(hz, hx, n):
            return None
        if not quantum_clean_four(hx, hz, n):
            return None
        if accepted == idx:
            assert n == 550 and k > 0
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "quantum_logical_floor": 5,
                    "support": (tuple(sa[0]), tuple(sb[0]))}
        accepted += 1
        return None

    # Cover new component-family pairings first. Relative automorphism
    # images are only trials: Tanner invariants discard orbit copies.
    plans = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    plans.sort(key=lambda ab: (
        int(bank[ab[0]] in legacy) + int(bank[ab[1]] in legacy),
        ab[0] == ab[1], ab))
    for a, b in plans:
        for u in range(1, 11):
            result = inspect(binomials(bank[a]),
                             binomials(image(bank[b], u)))
            if result is not None:
                return result

    # Asymmetric supports remain a fallback, not a competitor against
    # available k=226, weight-eight candidates. Their component and
    # full quantum screens are equally strict.
    descriptors = [((5, 0), key, binomials(key)) for key in bank]
    for attempt in range(4096):
        if attempt % 4 == 0:
            continue
        A = binomials(bank[attempt % len(bank)])
        B = _draw_B(local, G, attempt, descriptors)
        if _kernel_profile(
                _component_rows(G, "B", B), 3 * G.q)[0] != 5:
            continue
        result = inspect(A, B)
        if result is not None:
            return result
    raise ValueError("novel quantum-four-clean catalogue exhausted")


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: The Frobenius component-screened catalogue consistently gives d30s>4: parent slot 30 fell from d2s=5 to d30s=4, with both X and Z logicals touching both LP sectors.
# REFUTED: Exhaustive connected binomial enumeration supplies five novel k=226 finalists: the current table measured only one fresh k=226 code and four k=222 codes; slots 0 and 1 reproduced archived binomial codes.
# REFUTED: Homology-five rank reservation raises the emitted rank surplus or logical floor on the Frobenius (1,3)x(1,3) bed: all 32 distinct current codes had k=222,d2s=5, and all five measured finalists retained d30s=5.
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
