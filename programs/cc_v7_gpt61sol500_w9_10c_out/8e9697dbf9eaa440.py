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
CONCEPT = ("C35, order 35, A=(2,3), B=(2,4): separate rank-one evaluations at the two degree-three order-seven factors should supply five novel cell-0.150 codes retaining distance readings >=14 while gaining k above the rank-count bound.")
GROUP = ("Jump from nonabelian C3:C4 to custom cyclic C35, a genuinely different group. Since 2 has order three modulo seven, F2[C35] contains two distinct degree-three order-seven factors. Related C31 and C37 cubic beds measured distance fourteen. Those readings motivate sparse spectral optimization here, but neither distance survival nor archive novelty follows from the algebra.")
SHAPES = ("A=(2,3), B=(2,4), q=35 give n=560, kb=70 and guaranteed rate 0.125; the size window permits q<=37. Unequal constituents have 105/140 variables and 70 checks each. Separate degree-three deficits predict k=79 or 81, inside cell 0.150; actual full ranks enforce admission. Parent minima broadly occupied sector-1 Acol0..5/Bcol0..6 at weight four. Narrow cubic-column constituents change that light-word mechanism and all four sector-2 block pairs.")
FILLING = ("Twenty constituent prescriptions combine two support profiles, five gap policies and the two independently optimized sides. GF8 evaluations impose rank one at alpha for A and alpha^3 for B, distinct Frobenius orbits. Verified constituent ranks require exactly the proposed losses. A rows total four/five and B rows six; every column totals three, so HX<=8 and HZ<=9, also below cap twelve. Explicit cofactor and randomized kernel witnesses receive equal refinement budgets. Full verified kernels exclude weights 1..4; exact target-band ranks, CSS checks and conservative uncoloured Tanner invariants gate an immutable portfolio. Larger readings remain upper bounds, not certificates.")
FAMILY_NAME = "r015_c35_23x24_splitGF8_cubic_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 35

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C35 requires order 35")
    els = [(a,) for a in range(35)]

    def red(g):
        return (int(g[0]) % 35,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 35,)

    def inv(g):
        return ((-int(g[0])) % 35,)

    def idx(g):
        return int(g[0]) % 35

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def _extract_components(G, hx, hz):
    """Both constituent matrices, extracted from the verified full lift."""
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
    br = [row & ((1 << (cb * q)) - 1)
          for row in hz[:rb * q]]
    return ar, br


def _odd_inverse_classes(G):
    """Three order-four edge classes and the normal order-six subgroup."""
    steps = sorted({min(g, G.inv(g))
                    for g in G.elements if g[1] % 2})
    subgroup = [g for g in G.elements if g[1] % 2 == 0]
    assert len(steps) == 3 and len(subgroup) == 6
    assert all(G.mul(g, g) != G.e for g in steps)
    return steps, subgroup


def _draw_incidence(rng, G, attempt, pools):
    """Four A prescriptions times five B support prescriptions."""
    steps, subgroup = pools
    card, epoch = attempt % 20, attempt // 20
    amode, bmode = card % 4, card // 4

    def regular(width, internal, labels=None):
        C = [[], []]
        for i in range(2):
            for gap in rng.sample(steps, internal):
                C[i].append([G.e, gap])
                C[1 - i].append([])
        cross = width - 2 * internal
        if labels is None:
            # Odd internal gaps reverse b-parity. Cross differences
            # preserve it, giving a sufficient triangle-free grading.
            pool = subgroup if internal else G.elements
            labels = rng.sample(pool, cross)
        assert len(labels) == cross
        for voltage in labels:
            C[0].append([G.e])
            C[1].append([voltage])
        assert all(len(row) == width for row in C)
        assert all(sum(map(len, row)) == width for row in C)
        return C

    if amode == 0:
        # Six distinct subgroup matchings give two K6,6 components.
        A = regular(6, 0, list(subgroup))
    elif amode == 1:
        outside = [g for g in G.elements if g[1] % 2]
        count = 1 + epoch % 5
        labels = (rng.sample(subgroup, 6 - count)
                  + rng.sample(outside, count))
        A = regular(6, 0, labels)
    elif amode == 2:
        A = regular(6, 1)
    else:
        A = regular(6, 2 + epoch % 2)

    if bmode < 4:
        B = regular(7, bmode)
    else:
        # Row sums seven, column degrees 1,3,2,2,2,2,2.
        counts = ((1, 2, 1, 1, 1, 1, 0),
                  (0, 1, 1, 1, 1, 1, 2))
        B = [[[] for _ in range(7)] for _ in range(2)]
        for i in range(2):
            for j, count in enumerate(counts[i]):
                if count == 1:
                    B[i][j] = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    gap = rng.choice(steps)
                    B[i][j] = [first, G.mul(first, gap)]

    assert all(sum(map(len, row)) == 6 for row in A)
    assert all(sum(map(len, row)) == 7 for row in B)
    assert all(len(A[0][j]) + len(A[1][j]) == 2
               for j in range(6))
    return A, B


def _no_kernel_through_three(rows, n):
    """Exactly exclude all nonzero kernel words of weights 1, 2 and 3."""
    columns = [0] * n
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, first in enumerate(columns):
        for j in range(i):
            if (first ^ columns[j]) in singles:
                return False
    return True


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
    return tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)))


def _catalogue_invariant(hx, hz, n):
    """Intrinsic rooted invariant, including nonregular hyperedges.

    Degree-13 qubits identify sector 2 without using block labels.
    Check weights colour its projected hyperedges. Root every such
    qubit, rather than assuming a transitive full-code group action.
    Equality is inconclusive and is conservatively rejected.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    high = [j for j, col in enumerate(columns)
            if col.bit_count() == 13]
    positions = {j: t for t, j in enumerate(high)}
    edges, weights = [], []
    incident = [[] for _ in high]
    for row in rows:
        endpoints = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                endpoints.append(positions[j])
            bits ^= bit
        edge = len(edges)
        edges.append(tuple(endpoints))
        weights.append(row.bit_count())
        for v in endpoints:
            incident[v].append(edge)

    def compress(signatures):
        table = {s: i for i, s in enumerate(sorted(set(signatures)))}
        return [table[s] for s in signatures]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(5):
            es = [(w, tuple(sorted(colours[v] for v in endpoints)))
                  for w, endpoints in zip(weights, edges)]
            ec = compress(es)
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            colours = compress(vs)
        rooted.append(sha256(repr(tuple(history)).encode()).digest())
    return (n, tuple(sorted(Counter(weights).items())),
            tuple(sorted(Counter(c.bit_count()
                                 for c in columns).items())),
            tuple(sorted(rooted)))


def _shown_archive_invariants():
    """Only exclusion references; never submit these archived fillings."""
    els = [(a, b) for b in range(2) for a in range(12)]

    def mul(g, h):
        return ((g[0] + pow(5, g[1], 12) * h[0]) % 12,
                (g[1] + h[1]) % 2)

    def inv(g):
        b = (-g[1]) % 2
        return ((-pow(5, b, 12) * g[0]) % 12, b)

    old = _G(dict(elements=els, ident=(0, 0), mul=mul, inv=inv), 24)
    e = old.e

    def incidence(top, bottom, ratios):
        B = [[], []]
        for gap in top:
            B[0].append([e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([e, gap])
        for voltage in ratios:
            B[0].append([e])
            B[1].append([voltage])
        return B

    references = [
        (((3,1),(5,1),(1,1)), (), (),
         ((7,1),(9,1),(10,1),(5,1),(0,0),(3,1),(5,0))),
        (((3,0),(1,0),(5,0)), ((5,1),), ((1,0),),
         ((0,0),(8,1),(6,0),(10,1),(2,0))),
        (((5,1),(1,1),(3,1)), ((1,1),(1,0)), ((3,0),(5,1)),
         ((6,1),(8,1),(10,1))),
        (((3,0),(5,0),(1,0)), ((1,1),(1,0)), ((3,0),(5,0)),
         ((6,1),(10,1),(4,0))),
        (((5,1),(1,1),(3,1)), ((3,1),(1,1),(3,0)),
         ((1,0),(5,1),(3,1)), ((2,1),)),
    ]
    common = (
        ((), (), (0,1,2,3,6,7,9)),
        ((1,), (5,), (0,2,4,6,8)),
        ((1,3), (3,5), (0,2,6)),
        ((1,3,5), (1,3,5), (0,)),
        ((5,1), (5,3), (0,4,8)),
    )
    for top, bottom, ratios in common:
        references.append(
            (((1,0),(3,0),(5,0)),
             tuple((d,0) for d in top),
             tuple((d,0) for d in bottom),
             tuple((d,0) for d in ratios)))

    result = set()
    for generators, top, bottom, ratios in references:
        A = [[[e, g] for g in generators]]
        B = incidence(top, bottom, ratios)
        hx, hz, n = lifted_matrices(old, A, B)
        result.add(_catalogue_invariant(hx, hz, n))
    return result


def _c35_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _c35_basis(rows):
    basis = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            previous = basis.get(pivot)
            if previous is None:
                basis[pivot] = word
                break
            word ^= previous
    return basis


def _c35_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        previous = basis.get(pivot)
        if previous is None:
            return True
        word ^= previous
    return False


def _c35_component_rows(G, grid, side):
    """Extract the actual evaluator action using a one-cell dummy."""
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    dummy = [[[G.e]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:2 * G.q]]


def _c35_draw(rng, G, profile, policy, root):
    """Evaluation constraints only; matrix construction stays verified."""
    # GF8 = F2[t]/(t^3+t+1). Roots 1 and 3 are different orbits.
    def multiply(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 0xB
        return result

    beta = 1
    for _ in range(root):
        beta = multiply(beta, 2)
    powers = [1]
    for _ in range(6):
        powers.append(multiply(powers[-1], beta))
    assert len(set(powers)) == 7
    logarithm = {value: exponent
                 for exponent, value in enumerate(powers)}

    grid = [[[] for _ in profile[0]] for _ in range(2)]
    extent = (3, 7, 11, 15, 17)[policy]
    used = set()
    for j in range(len(profile[0])):
        counts = profile[0][j], profile[1][j]
        assert sum(counts) == 3
        offsets, values = [], []
        for count in counts:
            if count == 0:
                terms = []
            elif count == 1:
                terms = [0]
            elif count == 2:
                choices = [d for d in range(1, extent + 1)
                           if d % 7 and d not in used]
                if not choices:
                    choices = [d for d in range(1, extent + 1)
                               if d % 7]
                gap = rng.choice(choices)
                used.add(gap)
                terms = [0, gap if rng.randrange(2) else -gap]
            else:
                # A trinomial in an exclusive column vanishes at beta.
                a = rng.randrange(1, 7)
                b = logarithm[1 ^ powers[a]]
                terms = [0, a + 7 * rng.randrange(5),
                         b + 7 * rng.randrange(5)]
            value = 0
            for exponent in terms:
                value ^= powers[exponent % 7]
            offsets.append(terms)
            values.append(value)

        anchors = [rng.randrange(35), rng.randrange(35)]
        if counts[0] and counts[1]:
            residue = (anchors[0] + logarithm[values[0]]
                       - logarithm[values[1]]) % 7
            anchors[1] = residue + 7 * rng.randrange(5)
        else:
            assert values == [0, 0]

        evaluations = []
        for i in range(2):
            labels = sorted((anchors[i] + d) % 35
                            for d in offsets[i])
            assert len(set(labels)) == counts[i]
            grid[i][j] = [(a,) for a in labels]
            value = 0
            for a in labels:
                value ^= powers[a % 7]
            evaluations.append(value)
        assert evaluations[0] == evaluations[1]
    return grid


def _c35_cofactor(rows, count, q):
    """Cyclic cofactor witnesses from VERIFIED column syndromes."""
    from itertools import combinations

    mask = (1 << q) - 1
    columns = _c35_columns(rows, count * q)
    polys = [[(columns[j * q] >> (i * q)) & mask
              for j in range(count)] for i in range(2)]

    def rotate(poly, shift):
        return poly if not shift else (
            ((poly << shift) | (poly >> (q - shift))) & mask)

    def multiply(a, b):
        result = 0
        while a:
            bit = a & -a
            result ^= rotate(b, bit.bit_length() - 1)
            a ^= bit
        return result

    best = 0
    for js in combinations(range(count), 3):
        pieces = [0] * count
        for omitted, j in enumerate(js):
            a, b = js[:omitted] + js[omitted + 1:]
            pieces[j] = (multiply(polys[0][a], polys[1][b])
                         ^ multiply(polys[0][b], polys[1][a]))
        word = sum(poly << (j * q)
                   for j, poly in enumerate(pieces))
        if not word:
            continue
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        if not best or word.bit_count() < best.bit_count():
            best = word
        for shift in range(1, q):
            moved = sum(rotate(poly, shift) << (j * q)
                        for j, poly in enumerate(pieces))
            trial = word ^ moved
            if trial and trial.bit_count() < best.bit_count():
                best = trial
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _c35_probe(rng, rows, width, passes, seed):
    """Heuristic upper bound accompanied by an actual kernel witness."""
    columns = _c35_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        rng.shuffle(kernel)
        for i in range(len(kernel)):
            current = kernel[i]
            w = current.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                trial = current ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    current, w = trial, tw
            kernel[i] = current
            inspect(current)

        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 18:
                    for h in range(j):
                        inspect(pair ^ short[h])
        if weight <= 10:
            break

    assert best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _c35_bank(G, side):
    """Ten deliberate support/gap prescriptions per constituent side."""
    profiles = (
        (((1, 1, 2), (2, 2, 1)),
         ((0, 2, 2), (3, 1, 1)))
        if side == "A" else
        (((1, 1, 2, 2), (2, 2, 1, 1)),
         ((0, 1, 2, 3), (3, 2, 1, 0)))
    )
    bank, keys = [], set()
    for card in range(10):
        pi, policy = divmod(card, 5)
        profile = profiles[pi]
        augmentation = gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in profile
        ])
        expected_rank = 2 * (G.q - 1) + augmentation - 3
        local = random.Random(
            BASE_SEED ^ (0xC350801 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        width = len(profile[0]) * G.q
        raw, fillings = [], set()

        for trial in range(128):
            grid = _c35_draw(local, G, profile, policy,
                             1 if side == "A" else 3)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _c35_component_rows(G, grid, side)
            rank = gf2_rank(rows)
            if rank != expected_rank:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            seed = _c35_cofactor(rows, len(profile[0]), G.q)
            if not seed or seed.bit_count() < 12:
                continue
            cycles = sum(
                ((rows[i] & rows[j]).bit_count()
                 * ((rows[i] & rows[j]).bit_count() - 1) // 2)
                for i in range(len(rows)) for j in range(i))
            raw.append((seed.bit_count(), cycles, trial,
                        grid, rows, seed, rank))

        raw.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, cycles, trial, grid, rows, seed, rank in raw[:10]:
            word = _c35_probe(local, rows, width, 6, seed)
            refined.append((word.bit_count(), cycles, trial,
                            grid, rows, word, rank))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))

        kept = 0
        for _, cycles, trial, grid, rows, seed, rank in refined[:6]:
            word = _c35_probe(local, rows, width, 20, seed)
            if word.bit_count() < 12:
                continue
            key = _tanner_signature(rows, [])
            if key in keys:
                continue
            keys.add(key)
            bank.append({
                "grid": grid, "word": word, "rank": rank,
                "reading": word.bit_count(), "cycles": cycles,
                "card": card, "trial": trial,
                "augmentation": augmentation,
            })
            kept += 1
            if kept == 3:
                break
    bank.sort(key=lambda a: (
        -a["reading"], a["cycles"], a["card"], a["trial"]))
    return bank


def _c35_logical_witness(a, b, hx, hz, q):
    """Embed and reduce actual nontrivial witnesses in both CSS types."""
    best = None
    for kind, checks, stabilizers in (("X", hz, hx), ("Z", hx, hz)):
        own = _c35_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        count = 3 if kind == "X" else 4
        candidates = []
        for block in range(count):
            if kind == "X":
                word = source << (block * 4 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 4 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _c35_outside(word, own):
                continue

            for _ in range(2):
                previous = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:40]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == previous:
                    break
            assert word and _c35_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            candidates.append(word.bit_count())
        if not candidates:
            return None
        reading = min(candidates)
        best = reading if best is None else min(best, reading)
    return best


from functools import lru_cache


@lru_cache(maxsize=1)
def _c35_portfolio(G, count):
    """Immutable deterministic optimization result; no slot-selection state."""
    aa, bb = _c35_bank(G, "A"), _c35_bank(G, "B")
    if not aa or not bb:
        raise ValueError("split-GF8 constituent bank exhausted")
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["reading"], b["reading"])
            predicted = 79 + 2 * (2 - a["augmentation"])
            priority = (predicted * weak * weak, weak,
                        -a["cycles"] - b["cycles"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    graph_keys, results = set(), []
    cap = min(9, 12, W_MAX)
    tested = 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert wx <= 8 and wz <= 9
        assert _no_dead_lines(sa, sb)
        assert max(wx, wz) == _grid_weights(sa, sb)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 560 and k > 0
                and 125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        columns = _c35_columns(hx + hz, n)
        key = (_tanner_signature(hx, hz),
               tuple(sorted(c.bit_count() for c in columns)))
        if key in graph_keys:
            continue
        graph_keys.add(key)
        tested += 1

        # Exact complete-centralizer gates include both LP sectors.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        reading = _c35_logical_witness(a, b, hx, hz, G.q)
        if reading is None or reading < 12:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified C35 lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified C35 rank helpers disagree")

        quality = (k * reading * reading / n, reading, k, preliminary)
        screen = (
            ("k", k),
            ("logical_witness_upper_bound", reading),
            ("component_witness_upper_bounds",
             (a["reading"], b["reading"])),
            ("component_ranks", (a["rank"], b["rank"])),
            ("spectral_roots", (1, 3)),
            ("design_cards", (a["card"], b["card"])),
            ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("complete_kernel_exclusion_through", 4),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + 24:
            break
        if tested >= max(240, 6 * count):
            break

    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete independently optimized spectral design."""
    if G.q != 35:
        raise ValueError("split-GF8 portfolio requires C35")
    portfolio = _c35_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel target-band C35 portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supply twenty inequivalent initial designs on C12:C2(s=5): slots 0..19 collapsed to four classes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: Independent order-12 incidence cards escape the distance-four plateau or supply twenty initial evaluator classes: slots 0..19 retained seven classes, and all five measured leaders had d30s=4 with sector-1 minima.
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
