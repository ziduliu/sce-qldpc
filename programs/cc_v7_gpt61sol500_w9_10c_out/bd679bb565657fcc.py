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
CONCEPT = ("C7 semidirect C15 with action 2, order 105, A=B=(1,2): long-order noncommuting trinomials screened against normal-C7 and mixed-sector witnesses should supply five novel distance readings >=14 and improve the 29.9 frontier in cell 0.200; this is falsifiable, not a distance certificate.")
GROUP = ("Jump from Dic9 to C7:C15(s=2), order 105, a bed absent from BEDS. The action is valid because 2^3=1 modulo 7. Normal C7, central C5 and order-15 elements provide different triangle and invariant-word scales. The related order-117 C13:C9 example measured four distance-18 finalists and one mixed-sector distance-15 failure, motivating both long-order triangles and complete-code witness searches without transferring its distance.")
SHAPES = ("Equal A=B=(1,2) give 210-variable, 105-check constituents, n=525, kb=105 and guaranteed rate 0.200. The size window permits q<=119; the new fixed group uses q=105. Parent weight-12 X witnesses occupied sector-1 Acol/Bcol blocks broadly; slot 55's weight-13 mixed witness also touched Arow0/Brow0-1. Verified full ranks retain [0.175,0.225), targeting the order-117-inspired gap in the 0.200 frontier.")
FILLING = ("Twenty prescriptions pair noncommuting triangle spectra (15,15,15), (7,15,15) and (15,15,35). Every published cell contains three distinct terms, so BOTH CSS maxima are nine, below both nine and twelve. Verified constituents exclude kernel weights through four and enumerate the normal-C7 invariant kernel when its dimension is <=18, otherwise testing explicit basis combinations. Equal information-set refinement carries actual witnesses. Full lifts enforce actual rate and kernel-four exclusion; independent logical-class-aware probes include both sectors and stabilizer-pair descent. An immutable portfolio rejects equal full uncoloured Tanner invariants, without arbitrary ownership buckets. Larger readings remain upper bounds; archive novelty remains evaluator-tested.")
FAMILY_NAME = "r020_c7sd_c15_s2_q105_12x12_orbit_mixed_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 105

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7:C15(s=2) has order 105")
    # x^7=y^15=1, y*x*y^-1=x^2; 2 has order three modulo 7.
    powers = [pow(2, b, 7) for b in range(15)]
    els = [(a, b) for b in range(15) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 15)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 7, (b + d) % 15)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 15
        return ((-powers[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


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
    """Extract arbitrary-shaped constituents using the verified lift."""
    dummy = [[[G.e]]]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz]


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


def _legacy_frobenius_evenfirst_reference(rng, G, idx):
    """Inactive order-55 reference; the Dic9 NEW_CTX never calls it.

    First enumerate even/even designs. Later entries use a new
    rotation-only trinomial profile. All screens use verified lifts.
    """
    from itertools import combinations
    from collections import Counter

    assert G.q == 55
    local = random.Random(BASE_SEED ^ 0x5504E1)
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    representatives = sorted({
        min(j, inverse[j]) for j in range(G.q)
        if G._key(G.elements[j]) != G._key(G.e)
    })

    # Precompute the action on undirected generators. Enumerating all
    # 27-choose-3 triples avoids the parent's small sampled bank.
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    actions = []
    for u in range(1, 11):
        for v in range(11):
            image = []
            for a, b in G.elements:
                j = G.idx(((u * a + v * sums[b]) % 11, b))
                image.append(min(j, inverse[j]))
            actions.append(image)

    orbit_keys = set()
    for triple in combinations(representatives, 3):
        key = min(tuple(sorted(action[j] for j in triple))
                  for action in actions)
        orbit_keys.add(key)

    bank = []
    for key in sorted(orbit_keys):
        grid = [[[G.e, G.elements[j]] for j in key]]
        rows = _component_rows(G, "A", grid)
        if gf2_rank(rows) != G.q - 1:
            continue
        if _kernel_profile(rows, 3 * G.q)[0] != 5:
            continue
        bank.append(grid)
    if not bank:
        raise ValueError("no connected binomial component passed kernel4")

    def walk_signature(rows):
        """An additional uncoloured Tanner invariant.

        The relation joins checks sharing exactly one qubit. Histograms
        of joint two- and three-walk counts retain more information than
        the parent's independent check-intersection profiles.
        """
        m = len(rows)
        adj = [0] * m
        for i, row in enumerate(rows):
            for j in range(i):
                if (row & rows[j]).bit_count() == 1:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i

        neighbours = []
        for bits in adj:
            vertices = []
            while bits:
                bit = bits & -bits
                vertices.append(bit.bit_length() - 1)
                bits ^= bit
            neighbours.append(vertices)

        square = [[(adj[i] & adj[j]).bit_count()
                   for j in range(m)] for i in range(m)]
        profiles = []
        for i in range(m):
            third = [0] * m
            for neighbour in neighbours[i]:
                for j, value in enumerate(square[neighbour]):
                    third[j] += value
            histogram = Counter(
                (((adj[i] >> j) & 1), square[i][j], third[j])
                for j in range(m))
            profiles.append((rows[i].bit_count(),
                             len(neighbours[i]),
                             tuple(sorted(histogram.items()))))
        return tuple(sorted(profiles))

    graph_buckets = {}
    filling_keys = set()
    accepted = 0

    def novel(hx, hz):
        """Conservatively merge equal invariants, not label fingerprints."""
        basic = _tanner_signature(hx, hz)
        rows = hx + hz
        bucket = graph_buckets.get(basic)
        if bucket is None:
            graph_buckets[basic] = [(rows, None)]
            return True

        extended = walk_signature(rows)
        for position, (old_rows, old_extended) in enumerate(bucket):
            if old_extended is None:
                old_extended = walk_signature(old_rows)
                bucket[position] = (old_rows, old_extended)
            if extended == old_extended:
                return False
        bucket.append((rows, extended))
        return True

    def consider(A, B, phase, required_k=None):
        nonlocal accepted
        filling = tuple(
            tuple(tuple(sorted(G.idx(g) for g in cell)) for cell in row)
            for grid in (A, B) for row in grid)
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
        if required_k is not None and k != required_k:
            return None

        # These are hard acceptance conditions, not diagnostics.
        # They include every mixed-sector support of weight <=4.
        if _kernel_profile(hx, n)[0] != 5:
            return None
        if _kernel_profile(hz, n)[0] != 5:
            return None
        if not novel(hx, hz):
            return None

        if accepted == idx:
            assert css_ok(hx, hz), "verified lift failed CSS"
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "phase": phase, "centralizer_cutoff": 5,
                    "bank_size": len(bank)}
        accepted += 1
        return None

    # Inner transformations of A and B independently are gauge copies.
    # Only two outer classes remain: scaling x by 1 or by nonsquare 2.
    # Same-orbit untwisted pairs are omitted: they supply the obvious
    # three-sector-1/one-sector-2 weight-four matching mechanism.
    plans = [(a, b, twist)
             for a in range(len(bank))
             for b in range(a, len(bank))
             for twist in (0, 1)
             if a != b or twist == 1]
    plans.sort(key=lambda plan: (plan[0] == plan[1],
                                plan[2], plan[0], plan[1]))

    for a, b, twist in plans:
        A = bank[a]
        scale = 2 if twist else 1
        B = [[[(scale * g[0] % 11, g[1]) for g in cell]
              for cell in bank[b][0]]]
        answer = consider(A, B, "even-even", required_k=226)
        if answer is not None:
            return answer

    # A rotation-only trinomial has three order-eleven relative
    # quotients. The parent's trinomial patterns always included
    # an order-five relative quotient.
    pure_count = accepted
    patterns = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 4))
    pools = {
        b: [g for g in G.elements if g != G.e and g[1] == b]
        for b in range(5)
    }
    tested_components = {}

    for attempt in range(max(4096, 128 * (idx + 1))):
        card = (accepted - pure_count) % 20
        b0, b1 = patterns[(card + attempt // 32) % len(patterns)]
        A = bank[(card // 5 + attempt // 7) % len(bank)]

        # Cards prescribe a relative rotation ratio; the selected
        # scale and the two order-five generators are jointly optimized.
        ratio = 2 + card % 9
        delta = local.randrange(1, 11)
        triple = [G.e, (delta, 0), (ratio * delta % 11, 0)]
        B = [[triple,
              [G.e, local.choice(pools[b0])],
              [G.e, local.choice(pools[b1])]]]

        component_key = tuple(
            tuple(sorted(G.idx(g) for g in cell)) for cell in B[0])
        clean = tested_components.get(component_key)
        if clean is None:
            rows = _component_rows(G, "B", B)
            clean = (_kernel_profile(rows, 3 * G.q)[0] == 5
                     and gf2_rank(rows) == G.q)
            tested_components[component_key] = clean
        if not clean:
            continue

        answer = consider(A, B, "rotation-trinomial", required_k=222)
        if answer is not None:
            return answer

    raise ValueError("fresh full-kernel4 catalogue exhausted")


def _d9_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _d9_span(rows):
    span = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                span[p] = word
                break
            word ^= old
    return span


def _d9_outside(word, span):
    while word:
        p = word.bit_length() - 1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _d9_coset_seed(G, rows, count):
    """Exact minimum in the complete <x^2>-invariant kernel subspace.

    First find its nullspace, then enumerate that nullspace rather than
    all orbit selections. Every returned support is checked against the
    verified constituent matrix.
    """
    cols = _d9_columns(rows, count * G.q)
    pivots, nulls = {}, []
    for j in range(count):
        for b in range(2):
            for parity in range(2):
                word = syn = 0
                for a in range(parity, 18, 2):
                    t = j * G.q + G.idx((a, b))
                    word |= 1 << t
                    syn ^= cols[t]
                while syn:
                    p = syn.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syn, word)
                        break
                    syn ^= old[0]
                    word ^= old[1]
                if not syn:
                    nulls.append(word)

    previous = current = best = 0
    for number in range(1, 1 << len(nulls)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous).bit_length() - 1
        current ^= nulls[changed]
        if current and (not best or current.bit_count() < best.bit_count()):
            best = current
        previous = gray
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _d9_probe(rng, rows, width, passes, seed=0):
    """Verified information-set kernel witnesses, not distance proofs."""
    cols = _d9_columns(rows, width)
    order = list(range(width))
    best = seed

    def record(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

    for _ in range(passes):
        rng.shuffle(order)
        pivots, nulls = {}, []
        for j in order:
            syn, word = cols[j], 1 << j
            while syn:
                p = syn.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syn, word)
                    break
                syn ^= old[0]
                word ^= old[1]
            if not syn:
                nulls.append(word)
                record(word)

        short = sorted(nulls, key=int.bit_count)[:28]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                record(pair)
                if i < 18:
                    for h in range(j):
                        record(pair ^ short[h])
        starting = best
        for first in short:
            record(starting ^ first)
        if best and best.bit_count() <= 8:
            break

    if not best:
        raise ValueError("positive-rate constituent has no kernel witness")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best.bit_count(), best


def _d9_draw(rng, G, support, side, mode, epoch):
    """Relative-label policies change subgroup usage, not just gauges."""
    units = (1, 5, 7, 11, 13, 17)
    even = (2, 4, 8, 10, 14, 16)
    grid = []
    for i, counts in enumerate(support):
        row = []
        for j, count in enumerate(counts):
            if not count:
                row.append([])
                continue
            first = G.pick(rng)
            if count == 1:
                row.append([first])
                continue
            rotation = (rng.choice(units if mode != 2 else even), 0)
            cross = (rng.randrange(18), 1)
            if mode == 3:
                cell = [first]
                while len(cell) < count:
                    g = G.pick(rng)
                    if g not in cell:
                        cell.append(g)
            elif count == 2:
                mixed = (mode == 0 or
                         (mode == 2 and
                          (i + j + epoch + int(side == "B")) % 2))
                relative = cross if mixed else rotation
                cell = [first, G.mul(first, relative)]
            elif mode == 0:
                cell = [first, G.mul(first, rotation),
                        G.mul(first, cross)]
            elif mode == 1:
                second = G.mul(first, cross)
                cell = [first, second, G.mul(second, rotation)]
            else:
                second = G.mul(first, rotation)
                cell = [first, second, G.mul(second, rotation)]
            assert len(cell) == count and len(set(cell)) == count
            row.append(cell)
        grid.append(row)
    return grid


def _d9_pool(rng, G, support, side, mode, epoch):
    width = len(support[0]) * G.q
    pool, used = [], set()
    for trial in range(64):
        grid = _d9_draw(rng, G, support, side, mode, epoch)
        key = tuple(tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
        if key in used:
            continue
        used.add(key)
        rows = _component_rows(G, side, grid)
        if _kernel_profile(rows, width)[0] != 5:
            continue
        seed = _d9_coset_seed(G, rows, len(support[0]))
        reading, word = _d9_probe(rng, rows, width, 2, seed)
        cost = sum(
            ((row & other).bit_count()
             * ((row & other).bit_count() - 1)) // 2
            for i, row in enumerate(rows) for other in rows[:i])
        pool.append({"grid": grid, "rows": rows, "word": word,
                     "reading": reading, "cost": cost,
                     "rank": gf2_rank(rows), "trial": trial})
    pool.sort(key=lambda r: (
        r["reading"], -r["cost"], r["rank"], -r["trial"]), reverse=True)
    finalists = pool[:8]
    for item in finalists:
        reading, word = _d9_probe(
            rng, item["rows"], width, 12, item["word"])
        item.update(reading=reading, word=word)
    finalists.sort(key=lambda r: (
        r["reading"], -r["cost"], r["rank"], -r["trial"]), reverse=True)
    return finalists[:6]


def _d9_logical_reading(G, a, b, hx, hz, n):
    """Reduce actual nontrivial embedded witnesses in both logical types.

    Membership and commutation are verified. Descent is heuristic and
    can expose mixed-sector supports; it is not exhaustive.
    """
    q = G.q
    found = []
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        span = _d9_span(stabilizers)
        words = []
        for block in range(3 if kind == "X" else 4):
            if kind == "X":
                word = b["word"] << (block * 4 * q)
            else:
                word, source = 0, a["word"]
                while source:
                    bit = source & -source
                    acol, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((acol * 4 + block) * q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if _d9_outside(word, span):
                words.append(word)
        if not words:
            continue
        word = min(words, key=int.bit_count)
        for _ in range(3):
            starting = word
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            touching = sorted(
                (row for row in stabilizers if row & word),
                key=lambda row: (
                    2 * (row & word).bit_count() - row.bit_count()),
                reverse=True)[:96]
            starting_pairs = word
            for i, first in enumerate(touching):
                for second in touching[:i]:
                    trial = starting_pairs ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == starting:
                break
        assert word and _d9_outside(word, span)
        assert all((row & word).bit_count() % 2 == 0 for row in checks)
        found.append(word.bit_count())
    # Failure to embed a witness is not rewarded with a large sentinel.
    return min(found) if found else 5


def _legacy_dic9_ctx_reference(rng, G, idx):
    """Inactive Dic9 reference; the order-105 construction never calls it."""
    from hashlib import blake2b

    assert G.q == 36
    plans = (
        (((1, 1, 3), (1, 3, 1)),
         ((0, 1, 2, 2), (2, 1, 1, 1))),
        (((0, 2, 3), (3, 2, 0)),
         ((1, 1, 1, 2), (2, 1, 1, 1))),
        (((1, 2, 2), (2, 2, 1)),
         ((1, 1, 1, 2), (2, 1, 1, 1))),
        (((2, 2, 2), (2, 2, 2)),
         ((0, 2, 1, 2), (3, 0, 1, 1))),
        (((0, 2, 2), (3, 1, 1)),
         ((1, 1, 2, 2), (2, 2, 1, 1))),
    )
    card, epoch = idx % 20, idx // 20
    family, mode = card // 4, card % 4
    pressure = 1 + (card + epoch) % 3
    seen = set()

    for batch in range(32):
        # Initial batches respect the card; later batches deliberately
        # broaden its support/model prescription if its owned bin is empty.
        f = (family + batch // 4) % len(plans)
        m = (mode + batch // 8) % 4
        sa, sb = plans[f]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= min(9, W_MAX)
        aa = _d9_pool(rng, G, sa, "A", m, epoch + batch)
        bb = _d9_pool(rng, G, sb, "B", m, epoch + batch)
        if not aa or not bb:
            continue

        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            min(ab[0]["reading"], ab[1]["reading"]),
            pressure * ab[0]["reading"]
            + (4 - pressure) * ab[1]["reading"],
            -(ab[0]["cost"] + ab[1]["cost"])), reverse=True)
        survivors = []
        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 576 and k > 0
                    and 125 * n <= 1000 * k < 175 * n):
                continue

            # This is a genuine uncoloured graph invariant. Isomorphic
            # Tanner graphs necessarily have the same owner, regardless
            # of the group labeling or either slot's sampling stream.
            signature = (
                tuple(sorted(row.bit_count() for row in hx + hz)),
                _tanner_signature(hx, hz))
            digest = blake2b(
                repr(signature).encode("ascii"), digest_size=16).digest()
            if int.from_bytes(digest, "little") % N_SLOTS != idx:
                continue
            if digest in seen:
                continue
            seen.add(digest)

            # Includes all mixed-sector supports through weight four.
            # Short stabilizers are conservatively excluded as well.
            if _kernel_profile(hx, n)[0] != 5:
                continue
            if _kernel_profile(hz, n)[0] != 5:
                continue
            assert css_ok(hx, hz)
            reading = _d9_logical_reading(G, a, b, hx, hz, n)
            quality = (
                k * reading * reading / float(n),
                reading, min(a["reading"], b["reading"]),
                -(a["cost"] + b["cost"]), k)
            survivors.append({
                "A": A, "B": B, "screen_k": k,
                "quality": quality,
                "component_witness_readings": (
                    a["reading"], b["reading"]),
                "quantum_witness_reading": reading,
                "card": (family, mode, epoch, f, m, batch),
                "tanner_owner": idx, "tanner_digest": digest.hex(),
            })

        if survivors:
            winner = max(survivors, key=lambda item: item["quality"])
            assert lifted_k(G, winner["A"], winner["B"]) == winner["screen_k"]
            hx, hz, _ = lifted_matrices(G, winner["A"], winner["B"])
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return winner
    raise ValueError("no target-rate kernel4-clean code in owned Tanner bin")


def _f105_triangles(G):
    """Intrinsic noncommuting triangle classes, not anchor-label classes."""
    spectra = ((15, 15, 15), (7, 15, 15), (15, 15, 35))
    orders = {}
    for g in G.elements:
        power = G.e
        for order in range(1, G.q + 1):
            power = G.mul(power, g)
            if power == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid group element order")

    banks = [[] for _ in spectra]
    nonidentity = [g for g in G.elements if g != G.e]
    for i, g in enumerate(nonidentity):
        for h in nonidentity[:i]:
            if G.mul(g, h) == G.mul(h, g):
                continue
            spectrum = tuple(sorted((
                orders[g], orders[h],
                orders[G.mul(G.inv(g), h)])))
            if spectrum in spectra:
                banks[spectra.index(spectrum)].append((g, h))
    if any(not bank for bank in banks):
        raise ValueError("required noncommuting triangle class absent")
    return tuple(tuple(bank) for bank in banks)


def _f105_orbit_seed(G, rows, count):
    """Actual words in the normal-C7 invariant constituent kernel.

    Enumeration is exhaustive in this subspace only when its kernel
    dimension is <=18. Larger dimensions use explicit basis combinations.
    Neither branch certifies unrestricted constituent or quantum distance.
    """
    from itertools import combinations

    width = count * G.q
    cols = _d9_columns(rows, width)
    pivots, nulls = {}, []
    for j in range(count):
        for b in range(15):
            word = syndrome = 0
            for a in range(7):
                position = j * G.q + G.idx((a, b))
                word |= 1 << position
                syndrome ^= cols[position]
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                nulls.append(word)

    best = (1 << width) - 1
    assert all((row & best).bit_count() % 2 == 0 for row in rows)

    def record(word):
        nonlocal best
        if word and word.bit_count() < best.bit_count():
            best = word

    if len(nulls) <= 18:
        previous = current = 0
        for number in range(1, 1 << len(nulls)):
            gray = number ^ (number >> 1)
            changed = (gray ^ previous).bit_length() - 1
            current ^= nulls[changed]
            record(current)
            previous = gray
    else:
        for size in (1, 2, 3):
            for words in combinations(nulls, size):
                word = 0
                for value in words:
                    word ^= value
                record(word)

    assert best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _f105_probe(rng, rows, width, passes, seed=0, stabilizers=None):
    """Information-set witnesses with exact logical-class bookkeeping.

    With stabilizers=None, return an actual constituent kernel word.
    Otherwise return an actual nontrivial CSS logical word. The search
    budget is finite; its returned weight is an upper bound only.
    """
    cols = _d9_columns(rows, width)
    span = None if stabilizers is None else _d9_span(stabilizers)
    best = 0

    def logical_class(word):
        if span is None:
            return 1
        free = 0
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                bit = 1 << p
                free |= bit
                word ^= bit
            else:
                word ^= old
        return free

    def record(word, label=None):
        nonlocal best
        if not word or (best and word.bit_count() >= best.bit_count()):
            return
        if span is not None:
            if label is None:
                label = logical_class(word)
            if not label:
                return
        best = word

    if seed:
        assert all((row & seed).bit_count() % 2 == 0 for row in rows)
        record(seed)

    for pass_number in range(passes):
        order = list(range(width))
        rng.shuffle(order)
        if span is not None and pass_number % 2:
            # A separate information set explicitly prioritizes sector 2.
            order.sort(key=lambda j: j < 4 * 105)
        pivots, nulls = {}, []
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
                label = logical_class(word)
                nulls.append((word, label))
                record(word, label)

        # Elementary exchanges retain a basis of the complete kernel.
        # Stabilizer-valued vectors remain useful moves in the CSS case.
        rng.shuffle(nulls)
        for i in range(len(nulls)):
            current, label = nulls[i]
            weight = current.bit_count()
            for j, (other, other_label) in enumerate(nulls):
                if i == j:
                    continue
                trial = current ^ other
                w = trial.bit_count()
                if 0 < w < weight:
                    current, label, weight = trial, label ^ other_label, w
            nulls[i] = (current, label)
            record(current, label)

        for i, (first, label) in enumerate(nulls):
            for second, other_label in nulls[:i]:
                record(first ^ second, label ^ other_label)

        ranked = sorted(nulls, key=lambda item: item[0].bit_count())
        short = ranked[:32]
        tail = ranked[32:]
        short += rng.sample(tail, min(8, len(tail)))
        for i, (first, label) in enumerate(short):
            for j in range(i):
                pair = first ^ short[j][0]
                pair_label = label ^ short[j][1]
                for h in range(j):
                    record(pair ^ short[h][0], pair_label ^ short[h][1])

        for _ in range(2):
            if not best:
                break
            starting = best
            starting_label = logical_class(starting)
            for i, (first, label) in enumerate(short):
                record(starting ^ first, starting_label ^ label)
                for second, other_label in short[:i]:
                    record(starting ^ first ^ second,
                           starting_label ^ label ^ other_label)
            if best == starting:
                break

    if not best:
        raise ValueError("positive-rate screened kernel yielded no witness")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    assert span is None or _d9_outside(best, span)
    return best.bit_count(), best


def _f105_bank(rng, G, side, prescription, triangles):
    """Equally refined constituents for one two-triangle prescription."""
    width = 2 * G.q
    raw, fillings = [], set()
    for trial in range(128):
        cells = []
        for kind in prescription:
            anchor = G.pick(rng)
            g, h = rng.choice(triangles[kind])
            if rng.randrange(2):
                g, h = h, g
            cell = [anchor, G.mul(anchor, g), G.mul(anchor, h)]
            assert len(set(cell)) == 3
            cells.append(cell)
        grid = [cells]
        key = tuple(tuple(sorted(G.idx(g) for g in cell))
                    for cell in cells)
        if key in fillings:
            continue
        fillings.add(key)

        rows = _component_rows(G, side, grid)
        if _kernel_profile(rows, width)[0] != 5:
            continue
        cost = sum(
            overlap * (overlap - 1) // 2
            for i, row in enumerate(rows)
            for other in rows[:i]
            for overlap in ((row & other).bit_count(),))
        raw.append(dict(grid=grid, rows=rows, cost=cost,
                        rank=gf2_rank(rows), trial=trial))

    raw.sort(key=lambda item: (item["cost"], item["trial"]))
    shortlist = []
    for record in raw[:16]:
        seed = _f105_orbit_seed(G, record["rows"], 2)
        reading, word = _f105_probe(
            rng, record["rows"], width, 3, seed)
        record.update(reading=reading, word=word,
                      orbit_reading=seed.bit_count())
        shortlist.append(record)
    shortlist.sort(key=lambda item: (
        item["reading"], -item["cost"], item["rank"],
        -item["trial"]), reverse=True)

    finalists = shortlist[:8]
    for record in finalists:
        reading, word = _f105_probe(
            rng, record["rows"], width, 12, record["word"])
        record.update(reading=reading, word=word)
    finalists.sort(key=lambda item: (
        item["reading"], -item["cost"], item["rank"],
        -item["trial"]), reverse=True)
    return finalists


def _f105_signature(hx, hz, n):
    """Intrinsic uncoloured full-Tanner invariant, not an iso test.

    Equal signatures are conservatively rejected. Unequal signatures
    certify non-isomorphism under arbitrary check/qubit permutations.
    No group labels, protograph labels or CSS colouring enter this key.
    """
    rows = list(hx) + list(hz)
    cols = _d9_columns(rows, n)
    return (
        tuple(sorted(row.bit_count() for row in rows)),
        tuple(sorted(column.bit_count() for column in cols)),
        _tanner_signature(list(hx), list(hz)),
    )


def _f105_quantum_reading(rng, G, a, b, hx, hz, n):
    """Embedded and independently generated logical witnesses in both types."""
    best = n + 1
    q = G.q
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        span = _d9_span(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(2):
            if kind == "X":
                word = source << (block * 2 * q)
            else:
                word = 0
                bits = source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 2 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if _d9_outside(word, span):
                seeds.append(word)

        seed = min(seeds, key=int.bit_count) if seeds else 0
        _, word = _f105_probe(
            rng, checks, n, 3, seed, stabilizers=stabilizers)
        for _ in range(3):
            previous = word
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            touching = sorted(
                (row for row in stabilizers if row & word),
                key=lambda row: (
                    2 * (row & word).bit_count() - row.bit_count()),
                reverse=True)[:48]
            starting = word
            for i, first in enumerate(touching):
                for second in touching[:i]:
                    trial = starting ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
            if word == previous:
                break
        assert word and _d9_outside(word, span)
        assert all((row & word).bit_count() % 2 == 0 for row in checks)
        best = min(best, word.bit_count())
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _f105_portfolio(G):
    """Deterministic immutable portfolio; call order changes no result.

    Memoization only avoids repeating a pure calculation. Every emitted
    entry owns a different intrinsic full-Tanner signature. No archived
    filling is used as a seed, and archive novelty is not presumed.
    """
    assert G.q == 105
    counts = ((3, 3),)
    assert _no_dead_lines(counts, counts)
    # HX: A row 6 + B column 3 = 9.
    # HZ: B row 6 + A column 3 = 9.
    assert _grid_weights(counts, counts) == 9 <= min(9, W_MAX)

    triangles = _f105_triangles(G)
    specifications = tuple(
        (a, b) for a in range(3) for b in range(a, 3))
    # Omit the pair whose four triangles all have a normal-C7 edge.
    # Its invariant two-coset mechanism is least attractive here.
    plans = tuple(
        (a, b) for a in range(6) for b in range(a, 6)
        if (a, b) != (3, 3))
    assert len(plans) == 20

    banks = {}
    for side_number, side in enumerate(("A", "B")):
        for policy, prescription in enumerate(specifications):
            local = random.Random(
                BASE_SEED * 10007 + 0x1052015
                + 1000003 * side_number + 65537 * policy)
            banks[side, policy] = _f105_bank(
                local, G, side, prescription, triangles)

    entries, excluded = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for idx in range(N_SLOTS):
        pa, pb = plans[idx % 20]
        aa, bb = banks["A", pa], banks["B", pb]
        proposals = [
            ((min(a["reading"], b["reading"]),
              a["reading"] + b["reading"],
              -a["cost"] - b["cost"], -ai, -bi), a, b)
            for ai, a in enumerate(aa) for bi, b in enumerate(bb)]
        proposals.sort(key=lambda item: item[0], reverse=True)

        survivors, local_signatures = [], set()
        for preliminary, a, b in proposals:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 525 and k >= 105
                    and 175 * n <= 1000 * k < 225 * n):
                continue
            signature = _f105_signature(hx, hz, n)
            if signature in excluded or signature in local_signatures:
                continue
            local_signatures.add(signature)

            # This includes every mixed-sector support through four.
            # Small stabilizers are conservatively rejected as well.
            if _kernel_profile(hx, n)[0] != 5:
                continue
            if _kernel_profile(hz, n)[0] != 5:
                continue
            local = random.Random(
                BASE_SEED * 10007 + 0x105CC
                + 65537 * idx + 1000003 * len(survivors))
            reading = _f105_quantum_reading(
                local, G, a, b, hx, hz, n)
            weak = min(reading, a["reading"], b["reading"])
            survivors.append(dict(
                a=a, b=b, hx=hx, hz=hz, n=n, k=k,
                signature=signature, reading=reading,
                quality=(k * weak * weak, weak, k, preliminary)))
            if len(survivors) >= 3:
                break

        if not survivors:
            entries.append(None)
            continue
        winner = max(survivors, key=lambda item: item["quality"])
        assert css_ok(winner["hx"], winner["hz"])
        assert lifted_k(
            G, winner["a"]["grid"], winner["b"]["grid"]) == winner["k"]
        assert max(row.bit_count() for row in
                   winner["hx"] + winner["hz"]) <= min(9, W_MAX)
        excluded.add(winner["signature"])
        entries.append((
            freeze(winner["a"]["grid"]),
            freeze(winner["b"]["grid"]), winner["k"],
            winner["a"]["reading"], winner["b"]["reading"],
            winner["reading"], pa, pb))
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of an optimized structural prescription."""
    entry = _f105_portfolio(G)[idx]
    if entry is None:
        raise ValueError("order-105 invariant-novel target-rate card exhausted")
    A, B, k, da, db, reading, pa, pb = entry
    return {
        "A": A, "B": B, "screen_k": k,
        "card": (idx % 20, pa, pb),
        "component_witness_readings": (da, db),
        "quantum_witness_reading": reading,
        "complete_kernel4_excluded": True,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without post-selection draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Component-first selection with complete short-word screens used only as diagnostics prevents weight-four failures: current slot 30 had d2s=5 but d30s=4, with minimum witnesses touching sector 2 and three sector-1 blocks.
# REFUTED: The even-first rotation-trinomial programme supplies five fresh k=226 finalists: its three k=226 entries were archive duplicates, and all five measured finalists instead had k=222 and d30s=5.
# REFUTED: Dic9 degree-three/four constituent selection with C9 seeds and embedded-witness descent supplies five readings >=14 above fitness 24.3168: the measured current finalists returned 12,12,12,13,14 and fitness 20.9146.
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
