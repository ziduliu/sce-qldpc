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
CONCEPT = ("On cyclic C37, order 37, A=(2,3), B=(2,4), augmentation-deficient degree-three constituents selected by equally refined kernel witnesses should supply five novel measured distance readings >=15 in cell 0.150; this is a falsifiable measurement hypothesis.")
GROUP = ("Jump from nonabelian order 55 to cyclic order 37, a genuinely different group. The cyclic order-37 (2,3)x(2,4) bed reached 25.2, while order 31 reached 28.5 with distance readings fourteen. Factor sixteen permits q<=37, so q=37 gives the largest lift. These readings motivate constituent optimization, not a transferred distance certificate.")
SHAPES = ("Use unequal (2,3) and (2,4) constituents with lengths/check counts 111/74 and 148/74. Then n=592, kb=74 and guaranteed rate 0.125. A's prescribed augmentation rank is one; B's is one or two, suggesting k=79 or 76 when the other representation blocks have full rank. Actual verified ranks require k>=76 and rate in [0.125,0.175). Parent weight-five minima occupied sector-1 Acol/Bcol blocks broadly; earlier mixed-sector failures motivate complete-matrix screening.")
FILLING = ("Twenty support/polynomial cards independently optimize A and B. A counts are (1,1,1)/(2,2,2); three B profiles have row totals six and column totals three. A rows total three/six and columns three, so BOTH CSS maxima are nine, also below twelve. Exact constituent kernel-four gates precede equal-budget information-set, basis-exchange and triple-witness searches. Heterogeneous pairs receive verified ranks, full kernel-four rejection, explicit nontrivial logical embeddings, stabilizer descent and full logical-witness probes. Different intrinsic Tanner invariants separate emitted designs. All larger witness readings remain upper bounds, and novelty against the unseen archive remains evaluator-tested.")
FAMILY_NAME = "r015_c37_23x24_augmentation_cubic_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this cyclic group has order 37")
    els = [(a,) for a in range(37)]

    def red(g):
        return (int(g[0]) % 37,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 37,)

    def inv(g):
        return ((-int(g[0])) % 37,)

    def idx(g):
        return int(g[0]) % 37

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


def _component_rows(G, side, grid):
    """Extract the component from the verified construction.

    A one-cell dummy side reduces screening cost without changing the
    left/right action used for the component being extracted.
    """
    dummy = [[[G.e]]]
    mask = (1 << (3 * G.q)) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx[:G.q]]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz[:G.q]]


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


def _c37_columns(rows, width):
    """Syndromes from the scaffold's verified matrices."""
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _c37_clean_four(rows, width):
    """Exactly exclude every nonzero kernel support of weights 1..4."""
    cols = _c37_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for j, first in enumerate(cols):
        for i in range(j):
            value = first ^ cols[i]
            if value in singles or value in pairs:
                return False
            pairs.add(value)
    return True


def _c37_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _c37_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c37_component(G, grid, right=False):
    """Expose the actual left/right action with an unpublished zero dummy."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    return [row & mask for row in rows], width


def _c37_cycle_cost(rows, width):
    """Short-incidence-cycle penalty, not a distance estimate."""
    cols = _c37_columns(rows, width)
    squares = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            shared = (row & other).bit_count()
            squares += shared * (shared - 1) // 2

    adj = [0] * len(rows)
    internal = 0
    for col in cols:
        degree = col.bit_count()
        internal += degree * (degree - 1) * (degree - 2) // 6
        bits = col
        while bits:
            bit = bits & -bits
            adj[bit.bit_length() - 1] |= col ^ bit
            bits ^= bit
    triangles = 0
    for i, neighbors in enumerate(adj):
        later = neighbors >> (i + 1)
        while later:
            bit = later & -later
            j = i + bit.bit_length()
            triangles += (
                (neighbors & adj[j]) >> (j + 1)).bit_count()
            later ^= bit
    return 16 * squares + max(0, triangles - internal)


def _c37_component_key(rows, width):
    """Conservative intrinsic constituent invariant; no voltage labels."""
    from collections import Counter
    profiles = []
    degrees = [row.bit_count() for row in rows]
    for i, row in enumerate(rows):
        histogram = Counter(
            (degrees[j], (row & other).bit_count())
            for j, other in enumerate(rows) if j != i)
        profiles.append((degrees[i], tuple(sorted(histogram.items()))))
    cols = _c37_columns(rows, width)
    return (
        tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
    )


def _c37_probe(rows, width, local, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, never a distance sentinel.

    Randomized column elimination carries supports in the original
    coordinates. Elementary exchanges preserve the complete kernel.
    """
    cols = _c37_columns(rows, width)
    order = list(range(width))
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _c37_outside(word, own):
            return
        best, weight = word, w

    for _ in range(passes):
        local.shuffle(order)
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
            if syndrome == 0:
                nulls.append(word)
                inspect(word)

        if own is None:
            local.shuffle(nulls)
            for i in range(len(nulls)):
                current = nulls[i].bit_count()
                for j, other in enumerate(nulls):
                    if i == j:
                        continue
                    trial = nulls[i] ^ other
                    w = trial.bit_count()
                    if 0 < w < current:
                        nulls[i], current = trial, w
                inspect(nulls[i])
            for i, first in enumerate(nulls):
                for second in nulls[:i]:
                    inspect(first ^ second)

        short = sorted(nulls, key=int.bit_count)[
            :24 if own is None else 32]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < (24 if own is None else 18):
                    for h in range(j):
                        inspect(pair ^ short[h])

        if best and own is None:
            for _ in range(2):
                starting = best
                for i, first in enumerate(short):
                    inspect(starting ^ first)
                    for second in short[:i]:
                        inspect(starting ^ first ^ second)
                if best == starting:
                    break
        elif best:
            starting = best
            for stabilizer in own.values():
                inspect(starting ^ stabilizer)

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _c37_outside(best, own)
    return best


@lru_cache(maxsize=1)
def _c37_portfolio(G, count):
    """Pure deterministic portfolio with immutable published fillings.

    Memoization only avoids repeating the same optimization for every
    slot. No slot can mutate a cached filling or affect another slot.
    """
    from hashlib import sha256

    q, e = G.q, G.e
    assert q == 37 and A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    cap = min(9, W_MAX, 12)
    ap = ((1, 1, 1), (2, 2, 2))
    bprofiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (3, 2, 1, 0)),
        ((1, 1, 1, 3), (2, 2, 2, 0)),
    )
    for bp in bprofiles:
        assert _no_dead_lines(ap, bp)
        wx = max(sum(row) + sum(bp[i][j] for i in range(2))
                 for row in ap for j in range(4))
        wz = max(sum(row) + sum(ap[i][j] for i in range(2))
                 for row in bp for j in range(3))
        assert wx == wz == 9 <= cap

    ratios = (2, 3, 4, 5, 6, 8, 10)

    def x(a):
        return G.elements[a % q]

    def draw(local, side, card):
        ratio = ratios[card // 3]
        if side == "A":
            # Normalize the monomial row and one lower-row anchor.
            # The other anchors and gaps are genuine relative voltages.
            base = local.randrange(1, q)
            gaps = (
                [base, base * ratio % q, base * ratio * ratio % q]
                if card % 2 else local.sample(range(1, q), 3))
            C = [[[e] for _ in range(3)], []]
            for j, gap in enumerate(gaps):
                anchor = 0 if j == 0 else local.randrange(q)
                C[1].append([x(anchor), x(anchor + gap)])
            return C

        profile = bprofiles[card % 3]
        C = []
        for i, counts in enumerate(profile):
            row = []
            for j, terms in enumerate(counts):
                if terms == 0:
                    cell = []
                else:
                    anchor = local.randrange(q)
                    cell = [x(anchor)]
                    if terms >= 2:
                        gap = local.randrange(1, q)
                        cell.append(x(anchor + gap))
                    if terms == 3:
                        r = ratios[(card // 3 + i + j) % len(ratios)]
                        cell.append(x(anchor + r * gap))
                assert len(set(cell)) == terms
                row.append(cell)
            C.append(row)
        return C

    def filling_key(C):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in C for cell in row)

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    def constituent_bank(side):
        bank, seen_fillings, seen_graphs = [], set(), set()
        width = (3 if side == "A" else 4) * q
        for card in range(20):
            raw = []
            for trial in range(96):
                local = random.Random(
                    BASE_SEED ^ (0x37C150 + 104729 * card
                                 + 8191 * trial
                                 + 65537 * int(side == "B")))
                C = draw(local, side, card)
                fingerprint = filling_key(C)
                if fingerprint in seen_fillings:
                    continue
                seen_fillings.add(fingerprint)
                rows, actual_width = _c37_component(G, C)
                assert actual_width == width
                rank = gf2_rank(rows)
                if rank < 2 * q - 2:
                    continue
                if not _c37_clean_four(rows, width):
                    continue
                cost = _c37_cycle_cost(rows, width)
                raw.append((cost, trial, C, rows, rank))
            raw.sort(key=lambda r: r[:2])

            shortlist = []
            for cost, trial, C, rows, rank in raw[:6]:
                local = random.Random(
                    BASE_SEED ^ (0x37C251 + 104729 * card
                                 + 8191 * trial
                                 + 65537 * int(side == "B")))
                word = _c37_probe(rows, width, local, 5)
                if word:
                    shortlist.append(
                        (word.bit_count(), cost, trial, C, rows,
                         rank, word))
            shortlist.sort(key=lambda r: (-r[0], r[1], r[2]))

            # Both retained finalists receive exactly the same refinement.
            refined = []
            for _, cost, trial, C, rows, rank, seed in shortlist[:2]:
                local = random.Random(
                    BASE_SEED ^ (0x37C352 + 104729 * card
                                 + 8191 * trial
                                 + 65537 * int(side == "B")))
                word = _c37_probe(rows, width, local, 28, seed)
                if not word or word.bit_count() < 10:
                    continue
                signature = _c37_component_key(rows, width)
                refined.append(
                    (word.bit_count(), cost, trial, C, rows,
                     rank, word, signature))
            refined.sort(key=lambda r: (-r[0], r[1], r[2]))

            for reading, cost, trial, C, rows, rank, word, sig in refined:
                if sig in seen_graphs:
                    continue
                right, _ = _c37_component(G, C, True)
                transferred, source = 0, word
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    transferred |= 1 << (
                        j * q + G.idx(G.inv(G.elements[t])))
                    source ^= bit
                assert all((r & transferred).bit_count() % 2 == 0
                           for r in right)
                assert gf2_rank(right) == rank
                seen_graphs.add(sig)
                bank.append(dict(
                    grid=C, left=word, right=transferred,
                    reading=reading, cost=cost, rank=rank,
                    card=card, trial=trial))
        bank.sort(key=lambda r:
                  (-r["reading"], r["cost"], r["card"], r["trial"]))
        return bank

    aa, bb = constituent_bank("A"), constituent_bank("B")
    if len(aa) < 3 or len(bb) < 3:
        raise ValueError("too few independently refined C37 constituents")

    def embedded(record, kind, checks, stabilizers, own):
        """Verify the LP embedding before reducing its logical class."""
        best = 0
        blocks = 3 if kind == "X" else 4
        for block in range(blocks):
            if kind == "X":
                word = record["right"] << (block * 4 * q)
            else:
                word, source = 0, record["left"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 4 + block) * q + t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not _c37_outside(word, own):
                continue

            for _ in range(3):
                starting = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                moves = sorted(
                    (row for row in stabilizers
                     if (row & word).bit_count() >= 2),
                    key=lambda row: -(row & word).bit_count())[:32]
                starting_pair = word
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting_pair ^ first ^ second
                        if trial.bit_count() < word.bit_count():
                            word = trial
                if word == starting:
                    break
            assert word and _c37_outside(word, own)
            assert all((row & word).bit_count() % 2 == 0
                       for row in checks)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"],
         a["cost"] + b["cost"],
         int(a["card"] == b["card"]), ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))

    graph_keys, results = set(), []
    tested = 0
    for _, _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 592 or k < 76:
            continue
        if not (125 * n <= 1000 * k < 175 * n):
            continue
        if not _c37_clean_four(hx, n):
            continue
        if not _c37_clean_four(hz, n):
            continue

        cols = _c37_columns(hx + hz, n)
        intrinsic = (
            tuple(sorted(row.bit_count() for row in hx + hz)),
            tuple(sorted(c.bit_count() for c in cols)),
            _tanner_signature(hx, hz),
        )
        signature = sha256(repr(intrinsic).encode("ascii")).digest()
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1

        assert css_ok(hx, hz), "verified C37 lift failed CSS"
        assert max(row.bit_count() for row in hx + hz) <= cap
        bx, bz = _c37_basis(hx), _c37_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(
            BASE_SEED ^ (0x37D150 + 65537 * ai + 257 * bi))
        xword = _c37_probe(hz, n, local, 3, xseed, bx)
        zword = _c37_probe(hx, n, local, 3, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 10:
            continue

        quality = (
            k * reading * reading / float(n), reading,
            -a["cost"] - b["cost"], k)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["reading"], b["reading"]), signature.hex()))
        if len(results) >= count + 12:
            break
        if tested >= max(192, 8 * count):
            break

    results.sort(key=lambda r: (r[0], r[7]), reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one complete, independently optimized immutable design."""
    catalogue = _c37_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("C37 augmentation-cubic portfolio exhausted")
    quality, A, B, k, reading, cards, components, signature = catalogue[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "logical_witness_upper_bound": reading,
        "constituent_witness_upper_bounds": components,
        "design_cards": cards, "selection_quality": quality,
        "tanner_signature": signature,
        "complete_kernel_exclusion_through": 4,
    }


def _legacy_reinforcement_reference(rng, G, idx):
    """Rebuild a deterministic prefix of independently optimized designs.

    Every published design has a different full-Tanner invariant.
    Short stabilizers are permitted; short nontrivial logicals are not.
    """
    local = random.Random(BASE_SEED ^ 0x55A71E)
    width = 3 * G.q
    bank = [item for item in _component_bank(G) if item[0][0] == 5]
    if not bank:
        raise ValueError("no kernel-four-clean binomial constituent")

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
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def logical_four(cols, own):
        """Exactly detect nontrivial logical supports of weights 1..4.

        For each syndrome retain one support of size <=2. A collision
        is harmless only when its difference belongs to the stabilizer
        span. Linearity makes one representative per syndrome sufficient.
        """
        seen = {0: 0}
        for j, syn in enumerate(cols):
            word = 1 << j
            old = seen.get(syn)
            if old is None:
                seen[syn] = word
            elif outside(word ^ old, own):
                return True
        for j, syn in enumerate(cols):
            bit = 1 << j
            for i in range(j):
                syndrome = syn ^ cols[i]
                word = bit | (1 << i)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif outside(word ^ old, own):
                    return True
        return False

    def probe(cols, own, rounds):
        """Return a verified logical witness, or zero if none was found."""
        order = list(range(len(cols)))
        best = 0

        def inspect(word):
            nonlocal best
            if (word and (not best or word.bit_count() < best.bit_count())
                    and outside(word, own)):
                syndrome = 0
                remaining = word
                while remaining:
                    bit = remaining & -remaining
                    syndrome ^= cols[bit.bit_length() - 1]
                    remaining ^= bit
                assert syndrome == 0
                best = word

        for _ in range(rounds):
            local.shuffle(order)
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
                    inspect(word)

            short = sorted(nulls, key=int.bit_count)[:32]
            for j, first in enumerate(short):
                for second in short[:j]:
                    inspect(first ^ second)

            # Adding stabilizers preserves the logical class. Reduction
            # is equally budgeted for every finalist, including mixed
            # sector supports.
            if best:
                starting = best
                for row in own.values():
                    inspect(starting ^ row)
            if best and best.bit_count() <= 5:
                break
        return best

    pools = [[g for g in G.elements if g != G.e and g[1] == b]
             for b in range(5)]

    def commuting_triple(mode, position, attempt):
        """Three distinct elements; modes change intrinsic subgroup type."""
        if mode == 0:
            # Three-term polynomials in F2[C11]. Their augmentation is
            # one, and Phi_11 is irreducible over F2, so these are units
            # in that subgroup algebra. Full ranks are still measured.
            step = local.randrange(1, 11)
            ratio = (2, 3, 4, 5)[(position + attempt) % 4]
            return [G.e, (step, 0), ((ratio * step) % 11, 0)]

        if mode == 1:
            # A conjugate order-five subgroup, not necessarily <y>.
            g = local.choice(pools[1 + (position % 4)])
            power = G.e
            powers = [G.e]
            for _ in range(4):
                power = G.mul(power, g)
                powers.append(power)
            exponent = (2, 3, 4)[(position + attempt) % 3]
            return [G.e, g, powers[exponent]]

        # Independent mixed reinforcement keeps a broader fallback
        # without returning to the parent's deterministic card stream.
        b0 = (position + attempt) % 5
        b1 = (2 * position + attempt // 3 + 1) % 5
        g, h = local.choice(pools[b0]), local.choice(pools[b1])
        while h == g:
            h = local.choice(pools[b1])
        return [G.e, g, h]

    def reinforced(position, attempt):
        mode = (position // 2 + attempt // 12) % 3
        triple = commuting_triple(mode, position, attempt)
        prescriptions = ((0, 1), (0, 2), (1, 1),
                         (1, 2), (2, 2), (1, 3), (2, 4))
        b0, b1 = prescriptions[(position + attempt) % len(prescriptions)]
        for _ in range(32):
            g, h = local.choice(pools[b0]), local.choice(pools[b1])
            if G.mul(g, h) != G.mul(h, g):
                return [[triple, [G.e, g], [G.e, h]]], mode
        return None, mode

    graph_keys, filling_keys = set(), set()
    for position in range(idx + 1):
        survivors = []
        for attempt in range(512):
            strengthened, mode = reinforced(position, attempt)
            if strengthened is None:
                continue

            source = bank[(position + attempt // 7) % len(bank)][2]
            ordinary = [[list(cell) for cell in source[0]]]

            # Relative outer orientation is a search parameter, not
            # assumed novelty. The final graph invariant decides novelty.
            scale = (1, 2, 4, 7)[(position + attempt // 5) % 4]
            ordinary = [[[
                ((scale * g[0]) % 11, g[1]) for g in cell
            ] for cell in ordinary[0]]]
            A, B = ((ordinary, strengthened) if position % 2 == 0
                    else (strengthened, ordinary))

            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) <= min(9, W_MAX)

            # Cheap constituent screening precedes complete-code work.
            # No hand-written left/right lift enters either screen.
            strengthened_side = "B" if position % 2 == 0 else "A"
            component = _component_rows(
                G, strengthened_side, strengthened)
            component_floor, collisions = _kernel_profile(component, width)
            if component_floor < 4:
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 375 * n <= 1000 * k < 425 * n):
                continue

            # Different invariants certify nonisomorphism. Equality is
            # conservatively merged, not mistaken for an exact iso test.
            signature = _tanner_signature(hx, hz)
            if signature in graph_keys:
                continue

            bx, bz = basis(hx), basis(hz)
            cx, cz = columns(hz, n), columns(hx, n)
            if logical_four(cx, bx) or logical_four(cz, bz):
                continue
            graph_keys.add(signature)

            wx = probe(cx, bx, 6)
            wz = probe(cz, bz, 6)
            readings = [word.bit_count() for word in (wx, wz) if word]
            reading = min(readings) if readings else 5
            # Missing witnesses are not rewarded with a large sentinel.
            quality = (
                k * reading * reading / float(n),
                k, -collisions, int(mode < 2))
            survivors.append({
                "A": A, "B": B, "screen_k": k,
                "quality": quality, "probe": tuple(readings),
                "logical_floor": 5, "mode": mode,
                "checks": (hx, hz),
            })
            if len(survivors) >= 3:
                break

        if not survivors:
            raise ValueError("novel reinforced logical-four catalogue exhausted")

        # All survivors have already been reserved in graph_keys.
        # Discarded finalists therefore cannot reappear in another slot.
        winner = max(survivors, key=lambda item: item["quality"])
        if position == idx:
            hx, hz = winner.pop("checks")
            assert css_ok(hx, hz)
            assert lifted_k(G, winner["A"], winner["B"]) == winner["screen_k"]
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return winner

    raise ValueError("unreachable catalogue position")


def _legacy_cross_cayley_reference(rng, G, idx):
    """Inactive reference retained; NEW_CTX no longer calls this catalogue."""
    from itertools import combinations, product

    def clean4(rows, width):
        """Exactly exclude all nonzero kernel supports of weights 1..4."""
        cols = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= tag
                row ^= bit
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return False
        pairs = set()
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        # Distinct nonzero columns make a pair-pair collision a
        # four-distinct-qubit support. Pair-single collisions detect three.
        return True

    # A binomial is an undirected Cayley edge. Enumerate ALL triples,
    # rather than sampling a small bank and correlating its two sides.
    reps = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements if G._key(g) != G._key(G.e)
    })
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    automorphisms = []
    for u in range(1, 11):
        for v in range(11):
            image = {}
            for label in reps:
                a, b = G.elements[label]
                g = ((u * a + v * sums[b]) % 11, b)
                image[label] = min(G.idx(g), G.idx(G.inv(g)))
            automorphisms.append(image)

    keys = set()
    for triple in combinations(reps, 3):
        generators = [G.elements[t] for t in triple]
        # Commuting distinct edge generators immediately make a square.
        if any(G.mul(generators[i], generators[j])
               == G.mul(generators[j], generators[i])
               for i in range(3) for j in range(i)):
            continue
        key = min(tuple(sorted(image[t] for t in triple))
                  for image in automorphisms)
        keys.add(key)

    bank = []
    for key in sorted(keys):
        grid = [[[G.e, G.elements[t]] for t in key]]
        rows = _component_rows(G, "A", grid)
        if gf2_rank(rows) != G.q - 1:
            continue
        if not clean4(rows, 3 * G.q):
            continue
        quotient = tuple(sorted(
            min(G.elements[t][1], (-G.elements[t][1]) % 5)
            for t in key))
        bank.append((quotient, key, grid))
    if not bank:
        raise ValueError("no connected kernel-four-clean constituent")

    graph_keys, filling_keys = set(), set()
    accepted = 0

    def consider(A, B, prescription):
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
        if not (k > 0 and 375 * n <= 1000 * k < 425 * n):
            return None
        # Constituent cleanliness does not exclude mixed-sector words.
        # Both complete centralizers must pass, including sector 2.
        if not clean4(hx, n) or not clean4(hz, n):
            return None
        signature = _tanner_signature(hx, hz)
        if signature in graph_keys:
            return None
        graph_keys.add(signature)

        if accepted == idx:
            assert n == 550 and css_ok(hx, hz)
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "centralizer_screens": (5, 5),
                    "prescription": prescription}
        accepted += 1
        return None

    # Inner scalings are powers of 3. The multiplier 2 represents the
    # other outer-automorphism coset. Its relative orientation is tested,
    # not assumed to produce a different code.
    plans = []
    for a in range(len(bank)):
        for b in range(a, len(bank)):
            for twist in (1, 2):
                plans.append((bank[a][0] == bank[b][0], a, b, twist))
    plans.sort()
    for _, a, b, twist in plans:
        A = bank[a][2]
        B = [[[
            G.elements[G.idx(((twist * g[0]) % 11, g[1]))]
            for g in cell
        ] for cell in bank[b][2][0]]]
        result = consider(A, B, ("cross-binomial", a, b, twist))
        if result is not None:
            return result

    # Additional structural cards use one noncommuting trinomial and
    # two binomials. Total counts are (6,7), so both weight maxima are 9.
    # Cards prescribe quotient coordinates; intrinsic invariants and
    # exact screens, not random label fingerprints, select the catalogue.
    cards = sorted(
        (b0, b1, c0, c1)
        for b0, b1, c0, c1 in product(range(5), repeat=4)
        if b0 <= b1 and c0 <= c1
        and (b0 or b1) and (c0 or c1))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    local = random.Random(BASE_SEED ^ 0x55C405)
    for attempt in range(4096):
        card = attempt % len(cards)
        b0, b1, c0, c1 = cards[card]
        g, h = local.choice(pools[b0]), local.choice(pools[b1])
        if g == h or G.mul(g, h) == G.mul(h, g):
            continue
        s, t = local.choice(pools[c0]), local.choice(pools[c1])
        if G.mul(s, t) == G.mul(t, s):
            continue
        B = [[[G.e, g, h], [G.e, s], [G.e, t]]]
        if not clean4(_component_rows(G, "B", B), 3 * G.q):
            continue
        a = (attempt // len(cards) + card) % len(bank)
        result = consider(bank[a][2], B, ("trinomial-card", a, card))
        if result is not None:
            return result
    raise ValueError("strict full-kernel-four-clean catalogue exhausted")


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent screens and the two-second proxy suffice to protect the Frobenius-bed finalists from weight-four logicals; slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: Cross-Cayley pairing supplies five inequivalent k=226 finalists: the measured parent supplied only three, while slot 3 collapsed to slot 0 and the remaining two finalists had k=222.
# REFUTED: Commuting reinforcement improves the order-55 cell-0.400 record: the current program scored 10.0910 versus 10.2002, and all five measured finalists retained k=222,d30s=5.
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
