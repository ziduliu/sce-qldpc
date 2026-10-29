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
CONCEPT = ("C7 x C5, order 35, A=(2,3), B=(2,4): sparse rank-one evaluations on the C7 quotient, with independently optimized C5 labels, should supply five novel rate-0.150 finalists exceeding fitness 25.3; survival of distance-fourteen readings is a falsifiable hypothesis.")
GROUP = ("Jump from nonabelian C12:C2(s=5), order 24, to the direct product C7 x C5, which is cyclic of order 35. Its (2,3)x(2,4) bed is absent from BEDS; only the different (1,4)x(1,4) C35 bed has been measured. Related C31/C37 unequal-component beds measured distance fourteen. Since 2 has order three modulo seven, the C7 quotient supplies two degree-three factors for sparse evaluation constraints. These facts motivate the experiment, not a distance transfer.")
SHAPES = ("A=(2,3), B=(2,4), q=35 give n=560, kb=70 and guaranteed rate 0.125; the size window permits q<=37. Constituents have 105 and 140 variables, each with 70 checks. Actual full ranks select [0.125,0.175), rather than assuming a homology gain. The parent's measured weight-four minima occupied sector-1 Acol0..2/Bcol0..6 broadly. Cubic columns and two A rows replace that incidence mechanism.")
FILLING = ("Twenty constituent prescriptions combine two A supports, two C7 factor orientations and five C5 labeling policies. A rows total four/five or five/five, B rows six, and all columns total three: HX<=8 and HZ<=9, also below the assignment cap twelve. At y=1 in the C5 coordinate, both polynomial rows agree at a prescribed GF8 root; remaining C5 coordinates are optimized independently. Verified constituent ranks require exactly the intended degree-three loss. Cofactor witnesses extracted from verified matrices, information-set searches and complete-CSS stabilizer descent prioritize the immutable portfolio. Full kernels through four and intrinsic Tanner invariants gate publication. All larger witness readings remain upper bounds, and the unseen archive remains authoritative.")
FAMILY_NAME = "r015_c7xc5_23x24_GF8quotient_C5fiber_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 35

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 x C5 has order 35")
    els = [(a, b) for b in range(5) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 5)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 7,
                (int(g[1]) + int(h[1])) % 5)

    def inv(g):
        return ((-int(g[0])) % 7, (-int(g[1])) % 5)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


def _quotient_draw(rng, G, profile, policy, root):
    """GF8 constraints on C7 only; C5 coordinates remain independent.

    This constructs group-ring cells, not a substitute matrix lift.
    At the C5 trivial character, both rows evaluate identically at beta.
    Exact constituent ranks subsequently test the intended loss.
    """
    assert G.q == 35 and root in (1, 3)

    def multiply(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 0xB              # t^3+t+1
        return result

    beta = 1
    for _ in range(root):
        beta = multiply(beta, 2)
    powers = [1]
    for _ in range(6):
        powers.append(multiply(powers[-1], beta))
    assert len(set(powers)) == 7
    logarithm = {v: a for a, v in enumerate(powers)}
    grid = [[[] for _ in profile[0]] for _ in range(2)]

    for j in range(len(profile[0])):
        counts = (profile[0][j], profile[1][j])
        assert sum(counts) == 3
        offsets, values = [], []
        for count in counts:
            if count == 0:
                terms = []
            elif count == 1:
                terms = [0]
            elif count == 2:
                terms = [0, rng.randrange(1, 7)]
            elif count == 3:
                a = rng.randrange(1, 7)
                terms = [0, a, logarithm[1 ^ powers[a]]]
            else:
                raise ValueError("quotient card is not cubic")
            value = 0
            for a in terms:
                value ^= powers[a]
            offsets.append(terms)
            values.append(value)

        anchors = [rng.randrange(7), rng.randrange(7)]
        if counts[0] and counts[1]:
            assert values[0] and values[1]
            anchors[1] = (
                anchors[0] + logarithm[values[0]]
                - logarithm[values[1]]) % 7
        else:
            assert values == [0, 0]

        evaluations = []
        for i, count in enumerate(counts):
            aa = [(anchors[i] + a) % 7 for a in offsets[i]]
            base = rng.randrange(5)
            if policy == 0:
                bb = rng.sample(range(5), count)
            elif policy == 1:
                bb = [base] * count
            elif policy == 2:
                step = rng.randrange(1, 5)
                bb = [(base + step * t) % 5 for t in range(count)]
            elif policy == 3:
                step = rng.randrange(1, 5)
                bb = [(base + step * a) % 5 for a in aa]
            else:
                bb = [rng.randrange(5) for _ in aa]
            cell = [(a, b) for a, b in zip(aa, bb)]
            assert len(set(cell)) == count <= 5
            grid[i][j] = cell
            value = 0
            for a, _ in cell:
                value ^= powers[a]
            evaluations.append(value)
        assert evaluations[0] == evaluations[1]
    return grid


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


def _fiber_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _fiber_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _fiber_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _fiber_probe(rng, rows, width, passes, seed, own=None):
    """Heuristic upper bound carried by an actual kernel/logical witness."""
    cols = _fiber_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _fiber_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for pass_number in range(passes):
        rng.shuffle(order)
        if own is not None and pass_number % 2:
            n1 = A_SHAPE[1] * B_SHAPE[1] * 35
            order.sort(key=lambda j: j < n1)
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

        rng.shuffle(kernel)
        for _ in range(2):
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

        short = sorted(kernel, key=int.bit_count)[:20]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 12:
                    for h in range(j):
                        inspect(pair ^ short[h])
        if weight <= (9 if own is None else 8):
            break

    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _fiber_outside(best, own)
    return best


def _fiber_component(G, grid, side):
    """Use the verified lift even for constituent-only screening."""
    dummy = [[[G.e]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:len(grid) * G.q]]


def _fiber_cofactor(G, rows, count):
    """Abelian cofactor witnesses extracted from verified column syndromes.

    The group-ring convolution is algebra on those extracted coefficients,
    not an independently implemented left/right matrix lift. Every
    resulting witness is checked against the supplied verified rows.
    """
    from itertools import combinations

    q, mask = G.q, (1 << G.q) - 1
    cols = _fiber_columns(rows, count * q)
    coefficients = [
        [(cols[j * q + G.idx(G.e)] >> (i * q)) & mask
         for j in range(count)] for i in range(2)]
    translations = [
        [G.idx(G.mul(g, h)) for h in G.elements]
        for g in G.elements]

    def move(poly, permutation):
        result = 0
        while poly:
            bit = poly & -poly
            result ^= 1 << permutation[bit.bit_length() - 1]
            poly ^= bit
        return result

    def multiply(a, b):
        result = 0
        while a:
            bit = a & -a
            result ^= move(b, translations[bit.bit_length() - 1])
            a ^= bit
        return result

    best = 0

    def inspect(word):
        nonlocal best
        if word and (not best or word.bit_count() < best.bit_count()):
            best = word

    for js in combinations(range(count), 3):
        pieces = [0] * count
        for omitted, j in enumerate(js):
            a, b = js[:omitted] + js[omitted + 1:]
            pieces[j] = (
                multiply(coefficients[0][a], coefficients[1][b])
                ^ multiply(coefficients[0][b], coefficients[1][a]))
        word = sum(poly << (j * q) for j, poly in enumerate(pieces))
        if not word:
            continue
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        inspect(word)
        for permutation in translations[1:]:
            moved = sum(move(poly, permutation) << (j * q)
                        for j, poly in enumerate(pieces))
            inspect(word ^ moved)
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _fiber_bank(G, side):
    """Optimize ten quotient/support prescriptions per constituent."""
    if side == "A":
        profiles = (
            ((1, 1, 2), (2, 2, 1)),
            ((0, 2, 2), (3, 1, 1)),
        )
        plans = [(pi, policy, 1)
                 for pi in range(2) for policy in range(5)]
    else:
        profiles = (((1, 1, 2, 2), (2, 2, 1, 1)),)
        plans = [(0, policy, root)
                 for root in (1, 3) for policy in range(5)]

    bank, component_keys = [], set()
    for card, (pi, policy, root) in enumerate(plans):
        profile = profiles[pi]
        augmentation = gf2_rank([
            sum((count & 1) << j for j, count in enumerate(row))
            for row in profile])
        expected_rank = 2 * (G.q - 1) + augmentation - 3
        width = len(profile[0]) * G.q
        local = random.Random(
            BASE_SEED ^ (0x735F1500 + 104729 * card
                         + (1000003 if side == "B" else 0)))
        raw, fillings = [], set()
        for trial in range(128):
            grid = _quotient_draw(local, G, profile, policy, root)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _fiber_component(G, grid, side)
            rank = gf2_rank(rows)
            # Require the intended factor loss, not additional accidental
            # rank collapses that could conceal lighter constituent words.
            if rank != expected_rank:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            seed = _fiber_cofactor(G, rows, len(profile[0]))
            if not seed or seed.bit_count() < 10:
                continue
            cycles = sum(
                (rows[i] & rows[j]).bit_count()
                * ((rows[i] & rows[j]).bit_count() - 1) // 2
                for i in range(len(rows)) for j in range(i))
            raw.append((seed.bit_count(), cycles, trial,
                        grid, rows, seed, rank))

        raw.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, cycles, trial, grid, rows, seed, rank in raw[:16]:
            word = _fiber_probe(local, rows, width, 4, seed)
            refined.append((word.bit_count(), cycles, trial,
                            grid, rows, word, rank))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))

        kept = 0
        for _, cycles, trial, grid, rows, seed, rank in refined[:8]:
            word = _fiber_probe(local, rows, width, 20, seed)
            if not word or word.bit_count() < 10:
                continue
            key = _tanner_signature(rows, [])
            if key in component_keys:
                continue
            component_keys.add(key)
            bank.append({
                "grid": grid, "word": word, "rank": rank,
                "reading": word.bit_count(), "cycles": cycles,
                "card": card, "root": root,
                "augmentation": augmentation,
            })
            kept += 1
            if kept == 3:
                break
    bank.sort(key=lambda r: (-r["reading"], r["cycles"], r["card"]))
    return bank


def _fiber_signature(hx, hz, n):
    """Intrinsic uncoloured invariant; equal keys are rejected conservatively.

    Degree>=10 qubits intrinsically identify the high-degree sector.
    Weighted clique expansion of projected checks supplies rooted walk
    moments through six. Complete check-intersection profiles and qubit
    degrees supplement this contraction. No group labels enter the key.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    columns = _fiber_columns(rows, n)
    high = [j for j, c in enumerate(columns) if c.bit_count() >= 10]
    positions = {j: i for i, j in enumerate(high)}
    high_mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]
    marks = [Counter() for _ in high]

    for row in rows:
        ends, bits = [], row & high_mask
        while bits:
            bit = bits & -bits
            ends.append(positions[bit.bit_length() - 1])
            bits ^= bit
        for u in ends:
            marks[u][row.bit_count()] += 1
            for v in ends:
                if u != v:
                    adjacency[u][v] += 1

    rooted = []
    for u, neighbours in enumerate(adjacency):
        second, third = Counter(), Counter()
        for v, count in neighbours.items():
            for w, multiplicity in adjacency[v].items():
                second[w] += count * multiplicity
        for v, count in second.items():
            for w, multiplicity in adjacency[v].items():
                third[w] += count * multiplicity
        moments = (
            second[u], third[u],
            sum(c * c for c in second.values()),
            sum(c * third[v] for v, c in second.items()),
            sum(c * c for c in third.values()),
        )
        pair_features = Counter(
            (columns[high[v]].bit_count(),
             second.get(v, 0), third.get(v, 0))
            for v in range(len(high)))
        rooted.append((
            columns[high[u]].bit_count(),
            tuple(sorted(marks[u].items())),
            moments, tuple(sorted(pair_features.items()))))
    return (
        n,
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(rooted)),
        _tanner_signature(hx, hz),
    )


def _fiber_quantum_reading(G, a, b, hx, hz, n, rng):
    """Actual logical witnesses, including stabilizer and mixed-sector search."""
    q, ca, cb = G.q, A_SHAPE[1], B_SHAPE[1]
    readings = []
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _fiber_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(ca if kind == "X" else cb):
            if kind == "X":
                word = source << (block * cb * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _fiber_outside(word, own):
                continue

            for _ in range(3):
                starting = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                origin = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & origin).bit_count() >= 2),
                    key=lambda row: -(row & origin).bit_count())[:40]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = origin ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == starting:
                    break
            seeds.append(word)

        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        word = _fiber_probe(rng, checks, n, 2, seed, own)
        readings.append(word.bit_count())
    return min(readings)


from functools import lru_cache


@lru_cache(maxsize=1)
def _fiber_portfolio(G, count):
    """Deterministic immutable portfolio; independent of slot call order."""
    aa, bb = _fiber_bank(G, "A"), _fiber_bank(G, "B")
    if not aa or not bb:
        raise ValueError("quotient-field constituent bank exhausted")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["reading"], b["reading"])
            # Matching roots may correlate homology losses; this is only
            # preliminary ordering. The complete verified ranks decide.
            priority = (
                weak, int(a["root"] == b["root"]),
                70 - a["rank"] + 70 - b["rank"],
                -a["cycles"] - b["cycles"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    entries, graph_keys, filling_keys = [], set(), set()
    cap = min(9, 12, W_MAX)
    examined = 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert wx <= 8 and wz <= 9
        assert max(wx, wz) == _grid_weights(sa, sb)
        assert _no_dead_lines(sa, sb)
        assert all(len(set(c)) == len(c) <= 5
                   for grid in (A, B) for row in grid for c in row)
        if max(wx, wz) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 560 and k >= 76
                and 125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        examined += 1

        # Complete kernels, not just constituent kernels, are screened.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quotient-field lift failed CSS")

        local = random.Random(
            BASE_SEED ^ (0x735CC150 + 65537 * ai + 1009 * bi))
        reading = _fiber_quantum_reading(G, a, b, hx, hz, n, local)
        if reading is None or reading < 10:
            continue
        signature = _fiber_signature(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        assert lifted_k(G, A, B) == k
        quality = (k * reading * reading / n, reading, k, preliminary)
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("constituent_witnesses", (a["reading"], b["reading"])),
            ("component_ranks", (a["rank"], b["rank"])),
            ("design_cards", (a["card"], b["card"])),
            ("quotient_roots", (a["root"], b["root"])),
            ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("complete_kernel_exclusion_through", 4),
        )
        entries.append((quality, freeze(A), freeze(B), screen))
        if len(entries) >= count + 16:
            break
        if examined >= max(320, 10 * count):
            break

    entries.sort(key=lambda entry: entry[0], reverse=True)
    # Never fill exhausted positions with orbit copies or archived anchors.
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete quotient-constrained, verified design."""
    assert G.q == 35 and 0 <= idx < N_SLOTS
    portfolio = _fiber_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct target-rate quotient portfolio exhausted")
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
# REFUTED: Rank-prioritized odd-incidence cards supply twenty inequivalent initial designs on C12:C2(s=5): slots 0..19 collapsed to four classes, and all five measured leaders retained abundant sector-1 weight-four logicals at k=254.
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
