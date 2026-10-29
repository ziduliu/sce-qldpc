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
CONCEPT = ("(C3 x C3) semidirect C4 with quarter-turn action, order 36, A=(2,3), B=(2,4): noncommuting cubic constituents selected against short subgroup-invariant witnesses should supply five novel distance readings >=16 in cell 0.150 and exceed fitness 25.162; this is falsifiable, not a distance certificate.")
GROUP = ("Jump from C37 to the Frobenius group (C3 x C3):C4, with action (a,b)->(-b,a). Every nonidentity complement power fixes no nonzero kernel vector, giving trivial centre and six conjugacy classes. No recorded order-36 bed has these invariants. The related dicyclic bed scored 24.3, while the cyclic parent's five leaders all measured fourteen. Removing commutative cofactor constraints motivates this experiment; group structure alone predicts no distance.")
SHAPES = ("Keep A=(2,3), B=(2,4). The size window permits q<=37; the prescribed q=36 gives n=576, kb=72 and guaranteed rate 0.125. Constituents have 108 and 144 variables, each with 72 checks. Parent weight-14 X minima concentrated in sector-1 Acol1/2 across B columns; slots 3 and 6 also exposed mixed-sector witnesses. Verified full ranks retain [0.125,0.175), without imposing the parent's k=76.")
FILLING = ("Twenty cards combine five saturated cubic support pairings with four order-three/order-four binomial quotas and noncommuting trinomial policies. A rows total 4/5, B rows 6/6, and every component column has three terms: HX<=8 and HZ<=9, satisfying both nine and twelve caps. Verified constituent syndromes screen explicit C2/C3/C4-orbit words before equal-budget kernel refinement. Complete lifts enforce actual target-band ranks, CSS and kernel-four exclusion; embedded and independently generated logical witnesses undergo stabilizer descent. A deterministic portfolio conservatively rejects equal full check-intersection and bipartite-refinement signatures. Larger witness and GPU readings remain upper bounds.")
FAMILY_NAME = "r015_e9sd_c4_q36_23x24_noncommuting_orbit_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("(C3 x C3):C4 has order 36")
    # (a,b,t) represents the kernel vector followed by J^t.
    # J(a,b)=(-b,a), J^4=I over F3.
    els = [(a, b, t) for t in range(4)
           for b in range(3) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 3, int(g[2]) % 4)

    def act(a, b, t):
        t %= 4
        if t == 0:
            return a, b
        if t == 1:
            return -b, a
        if t == 2:
            return -a, -b
        return b, -a

    def mul(g, h):
        a, b, t = red(g)
        c, d, u = red(h)
        c, d = act(c, d, t)
        return ((a + c) % 3, (b + d) % 3, (t + u) % 4)

    def inv(g):
        a, b, t = red(g)
        a, b = act(-a, -b, -t)
        return (a % 3, b % 3, (-t) % 4)

    def idx(g):
        a, b, t = red(g)
        return 9 * t + 3 * b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
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


def _w_columns(rows, width):
    """Syndromes of a supplied verified matrix."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _w_probe(rng, rows, width, passes, seed=0):
    """Heuristic kernel witness search, never a distance certificate."""
    cols = _w_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def record(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
        rng.shuffle(order)
        pivots, basis = {}, []
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
                basis.append(word)

        # Elementary exchanges preserve the whole kernel basis.
        rng.shuffle(basis)
        for i in range(len(basis)):
            current = basis[i]
            current_weight = current.bit_count()
            for j in range(len(basis)):
                if i == j:
                    continue
                trial = current ^ basis[j]
                w = trial.bit_count()
                if 0 < w < current_weight:
                    current, current_weight = trial, w
            basis[i] = current
            record(current)

        for i, first in enumerate(basis):
            for j in range(i):
                pair = first ^ basis[j]
                record(pair)
                for h in range(j):
                    record(pair ^ basis[h])

        for descent in range(2):
            starting = best
            for i, first in enumerate(basis):
                record(starting ^ first)
                for j in range(i):
                    record(starting ^ first ^ basis[j])
            if best == starting:
                break
        if weight <= 10:
            break

    if not best:
        raise ValueError("component kernel witness unexpectedly absent")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return weight, best


def _w_coset_seed(G, rows, columns):
    """Enumerate the complete C9=<x^2>-invariant constituent subspace."""
    width = columns * G.q
    cols = _w_columns(rows, width)
    words, syndromes = [], []
    for j in range(columns):
        for b in range(2):
            for parity in range(2):
                word = syndrome = 0
                for a in range(parity, 18, 2):
                    position = j * G.q + G.idx((a, b))
                    word |= 1 << position
                    syndrome ^= cols[position]
                words.append(word)
                syndromes.append(syndrome)

    previous = word = syndrome = 0
    best, weight = 0, width + 1
    for number in range(1, 1 << len(words)):
        gray = number ^ (number >> 1)
        changed = (gray ^ previous).bit_length() - 1
        word ^= words[changed]
        syndrome ^= syndromes[changed]
        if syndrome == 0 and word.bit_count() < weight:
            best, weight = word, word.bit_count()
            if weight == 9:
                break
        previous = gray
    return weight, best


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
    intersection_signature = tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)))

    # Supplement the projected-check invariant with refinement of the
    # actual bipartite Tanner graph. X/Z check colours are not used.
    from collections import Counter
    width = max((row.bit_length() for row in rows), default=0)
    adjacency = [[] for _ in range(m + width)]
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            j = m + bit.bit_length() - 1
            adjacency[i].append(j)
            adjacency[j].append(i)
            row ^= bit
    initial = [(int(i >= m), len(neighbors))
               for i, neighbors in enumerate(adjacency)]
    palette = {v: i for i, v in enumerate(sorted(set(initial)))}
    colours = [palette[v] for v in initial]
    histories = [tuple(sorted(Counter(initial).items()))]
    for _ in range(5):
        refined = [
            (colours[i], tuple(sorted(Counter(
                colours[j] for j in neighbors).items())))
            for i, neighbors in enumerate(adjacency)]
        histories.append(tuple(sorted(Counter(refined).items())))
        palette = {v: i for i, v in enumerate(sorted(set(refined)))}
        colours = [palette[v] for v in refined]
    return intersection_signature, tuple(histories)


def _dic9_ctx_reference(rng, G, idx):
    """Uncalled reference to the former Dic9-specific construction."""
    # Nonuniform B columns spend the parent's unused HX weight budget.
    bprofiles = (
        ((1, 1, 1, 3), (1, 1, 3, 1)),
        ((0, 2, 1, 3), (2, 0, 3, 1)),
        ((1, 1, 2, 2), (1, 2, 1, 2)),
        ((0, 1, 2, 3), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (2, 3, 1, 0)),
    )
    atops = ((1, 1, 2), (0, 1, 3), (0, 2, 2))
    card = idx % 20
    mode, epoch = card % 4, idx // 20
    top = atops[(card // 4 + mode) % 3]
    sa = (top, tuple(3 - c for c in top))
    sb = bprofiles[card // 4]
    assert _no_dead_lines(sa, sb)
    assert sorted(map(sum, sa)) == [4, 5]
    assert list(map(sum, sb)) == [6, 6]
    assert _grid_weights(sa, sb) == 9
    assert _grid_weights(sa, sb) <= min(9, W_MAX)
    q = G.q
    assert q == 36 and 0 <= idx < N_SLOTS
    # These rotations have order nine or eighteen, not three or six.
    rotations = [a for a in range(1, 18) if a % 3]
    even_rotations = [a for a in rotations if a % 2 == 0]
    odd_rotations = [a for a in rotations if a % 2 == 1]

    def draw_grid(profile, side):
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, count in enumerate(counts):
                if not count:
                    row.append([])
                    continue
                first = G.pick(rng)
                if count == 1:
                    cell = [first]
                else:
                    steps = (even_rotations if mode == 2 else
                             odd_rotations if mode == 3 else rotations)
                    rotation = (rng.choice(steps), 0)
                    cross = (rng.randrange(18), 1)
                    if count == 2:
                        mixed = (mode == 0 or
                                 (mode == 1 and
                                  (i + j + epoch + int(side == "B")) % 2)
                                 or (mode == 3 and rng.randrange(3) == 0))
                        # A pure binomial column with an outside ratio
                        # has an unavoidable order-four kernel word.
                        degree = sum(r[j] for r in profile)
                        if degree == 2:
                            mixed = False
                        # Avoid rotation-only pairs in a 2+2 column.
                        if profile[0][j] == profile[1][j] == 2:
                            if i == (j + epoch + int(side == "B")) % 2:
                                mixed = True
                        relative = cross if mixed else rotation
                        cell = [first, G.mul(first, relative)]
                    elif (i + j + mode + epoch + int(side == "B")) % 2:
                        second = G.mul(first, cross)
                        cell = [first, second, G.mul(second, rotation)]
                    else:
                        cell = [first, G.mul(first, rotation),
                                G.mul(first, cross)]
                assert len(set(cell)) == count
                row.append(cell)
            grid.append(row)
        return grid

    def verified_component(grid, side):
        # A one-cell dummy gives the exact evaluator action cheaply.
        dummy = [[[G.e]]]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, dummy)
            return [row & mask for row in hx[:len(grid) * q]]
        _, hz, _ = lifted_matrices(G, dummy, grid)
        return [row & mask for row in hz[:len(grid) * q]]

    def cycle_cost(rows, width):
        cols = _w_columns(rows, width)
        c4 = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                overlap = (first & second).bit_count()
                c4 += overlap * (overlap - 1) // 2
        # This is only a construction tie-break, not a distance estimate.
        return c4

    def orbit_witness(rows, columns, deeper=False):
        """Actual invariant kernel witnesses from verified syndromes.

        C9 enumeration is exhaustive in its kernel, rather than over
        all coset assignments. C3 collisions inspect up to four cosets
        cheaply, or up to six cosets during finalist refinement.
        """
        cols = _w_columns(rows, columns * q)
        best, weight = 0, columns * q + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if 0 < w < weight:
                best, weight = word, w

        for stride in (2, 6):
            words, syndromes = [], []
            for block in range(columns):
                for b in range(2):
                    for residue in range(stride):
                        word = syndrome = 0
                        for a in range(residue, 18, stride):
                            position = block * q + G.idx((a, b))
                            word |= 1 << position
                            syndrome ^= cols[position]
                        words.append(word)
                        syndromes.append(syndrome)

            if stride == 2:
                pivots, basis = {}, []
                for syndrome, word in zip(syndromes, words):
                    while syndrome:
                        p = syndrome.bit_length() - 1
                        old = pivots.get(p)
                        if old is None:
                            pivots[p] = (syndrome, word)
                            break
                        syndrome ^= old[0]
                        word ^= old[1]
                    if not syndrome:
                        basis.append(word)
                previous = word = 0
                for number in range(1, 1 << len(basis)):
                    gray = number ^ (number >> 1)
                    changed = (gray ^ previous).bit_length() - 1
                    word ^= basis[changed]
                    inspect(word)
                    previous = gray
            else:
                seen = {0: 0}

                def collision(syndrome, word):
                    if syndrome in seen:
                        inspect(word ^ seen[syndrome])
                    else:
                        seen[syndrome] = word

                for syndrome, word in zip(syndromes, words):
                    collision(syndrome, word)
                for i, first in enumerate(syndromes):
                    for j in range(i):
                        collision(first ^ syndromes[j],
                                  words[i] ^ words[j])
                if deeper:
                    for i, first in enumerate(syndromes):
                        for j in range(i):
                            pair = first ^ syndromes[j]
                            word = words[i] ^ words[j]
                            for h in range(j):
                                collision(pair ^ syndromes[h],
                                          word ^ words[h])
        if best:
            assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return weight, best

    def constituent_finalists(profile, side):
        columns = len(profile[0])
        width = columns * q
        pool, used = [], set()
        for trial in range(192 if side == "A" else 256):
            grid = draw_grid(profile, side)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
            if key in used:
                continue
            used.add(key)
            rows = verified_component(grid, side)
            if _kernel_profile(rows, width)[0] != 5:
                continue
            orbit_weight, seed = orbit_witness(rows, columns)
            # These are explicit constituent words, not estimated floors.
            if orbit_weight <= 12:
                continue
            cost = cycle_cost(rows, width)
            pool.append({"grid": grid, "rows": rows,
                         "cost": cost, "trial": trial,
                         "orbit_weight": orbit_weight, "seed": seed})

        if not pool:
            raise ValueError("no constituent passed kernel4/C3 screens")
        pool.sort(key=lambda r: (
            -min(r["orbit_weight"], 30), r["cost"], r["trial"]))

        shortlist = []
        for record in pool[:32]:
            distance, witness = _w_probe(
                rng, record["rows"], width, 4, record["seed"])
            record.update(distance=distance, witness=witness)
            record["quality"] = (
                distance, min(record["orbit_weight"], 30), -record["cost"])
            shortlist.append(record)
        shortlist.sort(key=lambda r: r["quality"], reverse=True)

        finalists = shortlist[:6]
        for record in finalists:
            orbit_weight, seed = orbit_witness(
                record["rows"], columns, deeper=True)
            if not seed or record["witness"].bit_count() <= seed.bit_count():
                seed = record["witness"]
            distance, witness = _w_probe(
                rng, record["rows"], width, 24, seed)
            record.update(distance=distance, witness=witness,
                          orbit_weight=orbit_weight)
            record["quality"] = (
                distance, min(orbit_weight, 30), -record["cost"])
        finalists.sort(key=lambda r: r["quality"], reverse=True)
        return finalists

    def tanner_bucket(hx, hz, width):
        """Hash an intrinsic, uncoloured full-Tanner invariant.

        Different buckets certify non-isomorphism. Equal buckets are
        inconclusive and are used only to partition the search.
        """
        from collections import Counter
        import hashlib

        rows = hx + hz
        weights = [row.bit_count() for row in rows]
        incidence = [[] for _ in range(width)]
        for i, row in enumerate(rows):
            while row:
                bit = row & -row
                incidence[bit.bit_length() - 1].append(i)
                row ^= bit
        shared = Counter()
        for neighbors in incidence:
            for a, i in enumerate(neighbors):
                for j in neighbors[:a]:
                    shared[(min(i, j), max(i, j))] += 1
        adj = [{} for _ in rows]
        for (i, j), overlap in shared.items():
            adj[i][j] = overlap
            adj[j][i] = overlap
        masks = [sum(1 << j for j in neighbors) for neighbors in adj]
        initial = [
            (weights[i], tuple(sorted(Counter(
                (weights[j], overlap,
                 (masks[i] & masks[j]).bit_count())
                for j, overlap in neighbors.items()).items())))
            for i, neighbors in enumerate(adj)]
        palette = {v: i for i, v in enumerate(sorted(set(initial)))}
        colors = [palette[v] for v in initial]
        histories = [tuple(sorted(Counter(initial).items()))]
        for _ in range(2):
            refined = [
                (colors[i], tuple(sorted(Counter(
                    (overlap, colors[j])
                    for j, overlap in neighbors.items()).items())))
                for i, neighbors in enumerate(adj)]
            histories.append(tuple(sorted(Counter(refined).items())))
            palette = {v: i for i, v in enumerate(sorted(set(refined)))}
            colors = [palette[v] for v in refined]
        signature = (tuple(histories),
                     tuple(sorted(Counter(map(len, incidence)).items())))
        digest = hashlib.sha256(repr(signature).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % N_SLOTS

    aa = constituent_finalists(sa, "A")
    pairs, fillings = [], set()
    # Search independent B banks, not automorphism images of one winner.
    # A bounded search can publish fewer than four local finalists, but
    # never relaxes the slot's intrinsic invariant partition.
    for retry in range(6):
        if retry and retry % 3 == 0:
            aa = constituent_finalists(sa, "A")
        try:
            bb = constituent_finalists(sb, "B")
        except ValueError:
            continue
        for a in aa:
            for b in bb:
                key = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for grid in (a["grid"], b["grid"])
                            for row in grid for cell in row)
                if key in fillings:
                    continue
                fillings.add(key)
                hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                if tanner_bucket(hx, hz, n) != idx:
                    continue
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 576 and k > 0
                        and 5 * n <= 40 * k < 7 * n):
                    continue
                weak = min(a["distance"], b["distance"])
                quality = (k * weak * weak, weak,
                           a["distance"] + b["distance"],
                           -(a["cost"] + b["cost"]))
                pairs.append({"a": a, "b": b, "hx": hx, "hz": hz,
                              "n": n, "k": k, "quality": quality})
        if len(pairs) >= 4:
            break
    pairs.sort(key=lambda r: r["quality"], reverse=True)
    if not pairs:
        raise ValueError("no refined target-band pair reached Tanner bucket")

    def quantum_reading(record):
        """Reduce actual nontrivial witnesses; not an exhaustive search."""
        best, found = record["n"] + 1, False
        ca, cb = A_SHAPE[1], B_SHAPE[1]
        for kind, checks, stabilizers in (
                ("X", record["hz"], record["hx"]),
                ("Z", record["hx"], record["hz"])):
            span = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        span[p] = row
                        break
                    row ^= old

            def nontrivial(word):
                while word:
                    p = word.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        return True
                    word ^= old
                return False

            blocks = ca if kind == "X" else cb
            for block in range(blocks):
                if kind == "X":
                    word = record["b"]["witness"] << (block * cb * q)
                else:
                    word = 0
                    source = record["a"]["witness"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * cb + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not nontrivial(word):
                    continue
                found = True
                weight = word.bit_count()
                for descent in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        trial = starting ^ first
                        w = trial.bit_count()
                        if w < weight:
                            word, weight = trial, w
                        for j in range(i):
                            trial = starting ^ first ^ stabilizers[j]
                            w = trial.bit_count()
                            if w < weight:
                                word, weight = trial, w
                    if word == starting:
                        break
                assert word and nontrivial(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
        return best if found else None

    def mixed_reading(record, ceiling):
        """Independent full-code kernel probes, with exact logical classes.

        Random column elimination includes both LP sectors. Stabilizer
        normal forms identify nontrivial words before basis-pair/triple
        searches and local stabilizer descent. This remains heuristic.
        """
        best = (ceiling if ceiling is not None else
                min(record["a"]["distance"], record["b"]["distance"]))
        width = record["n"]
        for checks, stabilizers in (
                (record["hz"], record["hx"]),
                (record["hx"], record["hz"])):
            cols = _w_columns(checks, width)
            span = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        span[p] = row
                        break
                    row ^= old

            def logical_class(word):
                while word:
                    p = word.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        # Preserve this free coordinate while reducing
                        # the remaining, lower coordinates.
                        upper = 1 << p
                        return upper | logical_class(word ^ upper)
                    word ^= old
                return 0

            seeds = []

            def consider(word, label):
                if not label or not word:
                    return
                w = word.bit_count()
                if len(seeds) == 4 and w >= seeds[-1][0]:
                    return
                if any(word == old[1] for old in seeds):
                    return
                seeds.append((w, word))
                seeds.sort(key=lambda item: item[0])
                del seeds[4:]

            for pass_number in range(2):
                order = list(range(width))
                rng.shuffle(order)
                # Bias the second information set toward sector two;
                # this is not a separate or hand-written lift.
                if pass_number:
                    n1 = A_SHAPE[1] * B_SHAPE[1] * q
                    order.sort(key=lambda j: j < n1)
                pivots, basis = {}, []
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
                        basis.append((word, label))
                        consider(word, label)

                for i, (first, label) in enumerate(basis):
                    for second, other_label in basis[:i]:
                        if label != other_label:
                            word = first ^ second
                            if word.bit_count() <= best + 6:
                                consider(word, label ^ other_label)
                short = sorted(basis, key=lambda item: item[0].bit_count())[:20]
                for i, (first, label) in enumerate(short):
                    for j in range(i):
                        pair = first ^ short[j][0]
                        pair_label = label ^ short[j][1]
                        for h in range(j):
                            word = pair ^ short[h][0]
                            if word.bit_count() <= best + 6:
                                consider(word, pair_label ^ short[h][1])

            for _, word in seeds:
                assert logical_class(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                weight = word.bit_count()
                for _ in range(3):
                    starting = word
                    for row in stabilizers:
                        trial = starting ^ row
                        w = trial.bit_count()
                        if w < weight:
                            word, weight = trial, w
                    # Restricted pair moves are a heuristic, not an
                    # exhaustive mixed-sector distance search.
                    moves = [row for row in stabilizers
                             if (row & starting).bit_count() >= 2]
                    for i, first in enumerate(moves):
                        for second in moves[:i]:
                            trial = starting ^ first ^ second
                            w = trial.bit_count()
                            if w < weight:
                                word, weight = trial, w
                    if word == starting:
                        break
                assert word and logical_class(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
        return best

    survivors, signatures = [], set()
    for record in pairs:
        hx, hz, n = record["hx"], record["hz"], record["n"]
        # Exact full-matrix gate includes both sectors and all supports <=4.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        signature = _tanner_signature(hx, hz)
        if signature in signatures:
            continue
        signatures.add(signature)
        assert css_ok(hx, hz)
        reading = quantum_reading(record)
        reading = mixed_reading(record, reading)
        weak = reading
        record["reading"] = reading
        record["final_quality"] = (
            record["k"] * weak * weak, weak, record["quality"])
        survivors.append(record)
        if len(survivors) >= 4:
            break

    if not survivors:
        raise ValueError("no target-band pair passed complete kernel4 screen")
    winner = max(survivors, key=lambda r: r["final_quality"])
    A, B = winner["a"]["grid"], winner["b"]["grid"]
    assert lifted_k(G, A, B) == winner["k"]
    assert max(row.bit_count() for row in
               winner["hx"] + winner["hz"]) <= min(9, W_MAX)
    return {
        "A": A, "B": B, "screen_k": winner["k"],
        "design_card": (card // 4, mode, epoch),
        "invariant_bucket": idx,
        "B_column_degrees": tuple(sum(row[j] for row in sb)
                                  for j in range(B_SHAPE[1])),
        "component_witness_readings": (
            winner["a"]["distance"], winner["b"]["distance"]),
        "quantum_witness_reading": winner["reading"],
        "complete_kernel4_excluded": True,
    }


def _e36_design(rng, G, idx, excluded):
    """Optimize a support/order card using only verified matrix actions."""
    q = G.q
    assert q == 36
    card, mode = idx % 20, idx % 4
    plans = (
        (((1, 1, 2), (2, 2, 1)),
         ((1, 1, 2, 2), (2, 2, 1, 1))),
        (((0, 1, 3), (3, 2, 0)),
         ((1, 1, 2, 2), (2, 2, 1, 1))),
        (((0, 2, 2), (3, 1, 1)),
         ((1, 1, 2, 2), (2, 2, 1, 1))),
        (((1, 1, 2), (2, 2, 1)),
         ((0, 1, 2, 3), (3, 2, 1, 0))),
        (((0, 2, 2), (3, 1, 1)),
         ((0, 2, 2, 2), (3, 1, 1, 1))),
    )
    sa, sb = plans[card // 4]
    cap = min(9, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert sorted(map(sum, sa)) == [4, 5]
    assert list(map(sum, sb)) == [6, 6]
    assert all(sum(row[j] for row in sa) == 3 for j in range(3))
    assert all(sum(row[j] for row in sb) == 3 for j in range(4))
    assert _grid_weights(sa, sb) == 9 <= cap
    assert max(map(sum, sa)) + 3 == 8
    assert max(map(sum, sb)) + 3 == 9

    threes = [g for g in G.elements if g[2] == 0 and g != G.e]
    fours = [g for g in G.elements if g[2] % 2]
    odd_cosets = {
        t: [g for g in fours if g[2] == t] for t in (1, 3)}
    assert len(threes) == 8 and len(fours) == 18

    def draw(profile, side):
        """Prescribe relative orders, not automorphism images of a winner."""
        doubles = [(i, j) for i, counts in enumerate(profile)
                   for j, count in enumerate(counts) if count == 2]
        shift = (idx // 20 + int(side == "B")) % max(1, len(doubles))
        doubles = doubles[shift:] + doubles[:shift]
        quota = (0, 1, (len(doubles) + 1) // 2, len(doubles))[mode]
        normal_pairs = set(doubles[:quota])
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, count in enumerate(counts):
                if not count:
                    row.append([])
                    continue
                first = G.pick(rng)
                cell = [first]
                if count == 2:
                    pool = threes if (i, j) in normal_pairs else fours
                    cell.append(G.mul(first, rng.choice(pool)))
                elif count == 3:
                    while True:
                        if mode == 1:
                            t = 1 if (i + j + int(side == "B")) % 2 else 3
                            r, s = rng.sample(odd_cosets[t], 2)
                        elif mode == 2:
                            r = rng.choice(odd_cosets[1])
                            s = rng.choice(odd_cosets[3])
                        else:
                            r, s = rng.choice(threes), rng.choice(fours)
                        if r != s and G.mul(r, s) != G.mul(s, r):
                            break
                    cell.extend((G.mul(first, r), G.mul(first, s)))
                assert len(set(cell)) == count
                row.append(cell)
            grid.append(row)
        return grid

    def component(grid, side):
        unit = [[[G.e]]]
        width = len(grid[0]) * q
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
        block_mask = (1 << width) - 1
        return [row & block_mask for row in rows[:2 * q]]

    # Two C3-line representatives, one C4 complement and its C2.
    # The orbit action commutes with the corresponding constituent lift.
    orbit_partitions = {}
    for side in ("A", "B"):
        partitions = []
        for generator in ((1, 0, 0), (1, 1, 0),
                          (0, 0, 1), (0, 0, 2)):
            subgroup, power = [], G.e
            while power not in subgroup:
                subgroup.append(power)
                power = G.mul(power, generator)
            assert power == G.e
            remaining, partition = set(range(q)), []
            while remaining:
                h = G.elements[min(remaining)]
                orbit = tuple(sorted(
                    G.idx(G.mul(h, s) if side == "A" else G.mul(s, h))
                    for s in subgroup))
                assert len(set(orbit)) == len(subgroup)
                remaining.difference_update(orbit)
                partition.append(orbit)
            partitions.append(tuple(partition))
        orbit_partitions[side] = tuple(partitions)

    def orbit_seed(rows, count, side):
        """Actual invariant kernel words from verified column syndromes.

        Equal syndromes of zero/one/two orbit unions expose words using
        at most four complete subgroup orbits. Other supports are not
        excluded. Every retained seed is checked against the full matrix.
        """
        cols = _w_columns(rows, count * q)
        best, weight = 0, count * q + 1
        for partition in orbit_partitions[side]:
            words, syndromes = [], []
            for block in range(count):
                for orbit in partition:
                    word = syndrome = 0
                    for h in orbit:
                        position = block * q + h
                        word |= 1 << position
                        syndrome ^= cols[position]
                    words.append(word)
                    syndromes.append(syndrome)
            seen = {0: 0}

            def inspect(syndrome, word):
                nonlocal best, weight
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                    return
                difference = word ^ previous
                w = difference.bit_count()
                if 0 < w < weight:
                    best, weight = difference, w

            for syndrome, word in zip(syndromes, words):
                inspect(syndrome, word)
            for i, first in enumerate(syndromes):
                for j in range(i):
                    inspect(first ^ syndromes[j], words[i] ^ words[j])
        if best:
            assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return weight, best

    def cycle_cost(rows, width):
        cols = _w_columns(rows, width)
        cost = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                overlap = (first & second).bit_count()
                cost += overlap * (overlap - 1) // 2
        return cost

    def bank(profile, side):
        count, width = len(profile[0]), len(profile[0]) * q
        raw, fillings = [], set()
        for trial in range(128):
            grid = draw(profile, side)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
            if key in fillings:
                continue
            fillings.add(key)
            rows = component(grid, side)
            rank = gf2_rank(rows)
            # This is only a cheap rank-deficit prefilter. The final
            # CSS ranks, not a constituent formula, decide the rate.
            if rank < 2 * q - 6:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            upper, seed = orbit_seed(rows, count, side)
            cost = cycle_cost(rows, width)
            raw.append(dict(grid=grid, rows=rows, seed=seed,
                            upper=upper, cost=cost, rank=rank, trial=trial))
        raw.sort(key=lambda r: (
            -min(r["upper"], 24), r["cost"], r["trial"]))

        shortlist = []
        for record in raw[:16]:
            distance, word = _w_probe(
                rng, record["rows"], width, 4, record["seed"])
            record.update(distance=distance, witness=word)
            record["quality"] = (distance, -record["cost"])
            shortlist.append(record)
        shortlist.sort(key=lambda r: r["quality"], reverse=True)

        finalists = shortlist[:5]
        for record in finalists:
            # Carry the existing witness: deeper readings cannot rise.
            distance, word = _w_probe(
                rng, record["rows"], width, 24, record["witness"])
            record.update(distance=distance, witness=word)
            record["quality"] = (distance, -record["cost"])
        finalists.sort(key=lambda r: r["quality"], reverse=True)
        return finalists[:4]

    def logical_reading(record):
        """Actual nontrivial witnesses, including mixed-sector proposals."""
        best = record["n"] + 1
        for kind, checks, stabilizers in (
                ("X", record["hz"], record["hx"]),
                ("Z", record["hx"], record["hz"])):
            cols = _w_columns(checks, record["n"])
            span = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        span[p] = row
                        break
                    row ^= old

            def logical_class(word):
                result = 0
                while word:
                    p = word.bit_length() - 1
                    old = span.get(p)
                    if old is None:
                        bit = 1 << p
                        result |= bit
                        word ^= bit
                    else:
                        word ^= old
                return result

            seeds = {}

            def inspect(word, label=None):
                if not word:
                    return
                weight = word.bit_count()
                if len(seeds) >= 4 and weight >= max(seeds.values()):
                    return
                if label is None:
                    label = logical_class(word)
                if not label:
                    return
                seeds[word] = weight
                if len(seeds) > 4:
                    del seeds[max(seeds, key=seeds.get)]

            source = record["b"] if kind == "X" else record["a"]
            for block in range(3 if kind == "X" else 4):
                if kind == "X":
                    word = source["witness"] << (block * 4 * q)
                else:
                    word = 0
                    bits = source["witness"]
                    while bits:
                        bit = bits & -bits
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 4 + block) * q + h)
                        bits ^= bit
                assert all((row & word).bit_count() % 2 == 0 for row in checks)
                inspect(word)

            for pass_number in range(2):
                order = list(range(record["n"]))
                rng.shuffle(order)
                if pass_number:
                    order.sort(key=lambda j: j < 12 * q)
                pivots, basis = {}, []
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
                        basis.append((word, label))
                        inspect(word, label)

                short = sorted(basis, key=lambda x: x[0].bit_count())[:24]
                nontrivial = sorted(
                    (entry for entry in basis if entry[1]),
                    key=lambda x: x[0].bit_count())[:12]
                for first, label in nontrivial:
                    current, current_label = first, label
                    weight = current.bit_count()
                    for other, other_label in short:
                        trial = current ^ other
                        w = trial.bit_count()
                        if 0 < w < weight:
                            current, current_label, weight = (
                                trial, current_label ^ other_label, w)
                    inspect(current, current_label)
                moves = short + nontrivial
                for i, (first, label) in enumerate(moves):
                    for second, other_label in moves[:i]:
                        inspect(first ^ second, label ^ other_label)

            for word in list(seeds):
                assert logical_class(word)
                assert all((row & word).bit_count() % 2 == 0 for row in checks)
                weight = word.bit_count()
                for _ in range(3):
                    previous = word
                    for row in stabilizers:
                        trial = word ^ row
                        w = trial.bit_count()
                        if w < weight:
                            word, weight = trial, w
                    starting = word
                    moves = [row for row in stabilizers
                             if (row & starting).bit_count() >= 2]
                    for i, first in enumerate(moves):
                        for second in moves[:i]:
                            trial = starting ^ first ^ second
                            w = trial.bit_count()
                            if w < weight:
                                word, weight = trial, w
                    if word == previous:
                        break
                assert word and logical_class(word)
                assert all((row & word).bit_count() % 2 == 0 for row in checks)
                best = min(best, weight)
        return best

    aa, bb = bank(sa, "A"), bank(sb, "B")
    proposals = []
    for a in aa:
        for b in bb:
            weak = min(a["distance"], b["distance"])
            proposals.append(((weak, a["distance"] + b["distance"],
                               -a["cost"] - b["cost"]), a, b))
    proposals.sort(key=lambda item: item[0], reverse=True)

    survivors = []
    for preliminary, a, b in proposals:
        hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 576 and k >= 72 and 5 * n <= 40 * k < 7 * n):
            continue
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        signature = _tanner_signature(hx, hz)
        if signature in excluded:
            continue
        record = dict(a=a, b=b, hx=hx, hz=hz, n=n, k=k,
                      signature=signature)
        assert css_ok(hx, hz)
        reading = logical_reading(record)
        weak = min(reading, a["distance"], b["distance"])
        record.update(reading=reading,
                      quality=(k * weak * weak, weak, preliminary))
        survivors.append(record)
        if len(survivors) >= 4:
            break
    if not survivors:
        return None
    winner = max(survivors, key=lambda r: r["quality"])
    assert lifted_k(G, winner["a"]["grid"],
                    winner["b"]["grid"]) == winner["k"]
    assert max(row.bit_count() for row in
               winner["hx"] + winner["hz"]) <= cap
    return winner


from functools import lru_cache


@lru_cache(maxsize=1)
def _e36_portfolio(G):
    """Deterministic immutable support/order portfolio.

    Equal intrinsic signatures are rejected conservatively. Neither
    their inequality nor this cache establishes novelty against archive
    entries that the program cannot inspect.
    """
    entries, excluded = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for idx in range(N_SLOTS):
        entry = None
        for restart in range(3):
            local = random.Random(
                BASE_SEED * 10007 + 0xE936C4
                + 65537 * idx + 1000003 * restart)
            winner = _e36_design(local, G, idx, excluded)
            if winner is None:
                continue
            excluded.add(winner["signature"])
            entry = (freeze(winner["a"]["grid"]),
                     freeze(winner["b"]["grid"]), winner["k"],
                     winner["a"]["distance"], winner["b"]["distance"],
                     winner["reading"])
            break
        entries.append(entry)
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of an independently optimized noncommuting card."""
    assert 0 <= idx < N_SLOTS and G.q == 36
    entry = _e36_portfolio(G)[idx]
    if entry is None:
        raise ValueError("E9:C4 subgroup-witness portfolio card exhausted")
    A, B, k, da, db, reading = entry
    return {"A": A, "B": B, "screen_k": k, "design_card": idx % 20,
            "component_witness_readings": (da, db),
            "quantum_witness_reading": reading}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Distinct local Tanner-signature selections prevent downstream duplicate collapse on the parity-incidence catalogue; the measured sixty-slot parent retained only 21 codes, with 39 iso_dup statuses.
# REFUTED: Independent saturated constituent refinement plus embedded-witness stabilizer-pair descent supplies five Dic9 readings >=16; the measured parent instead returned 14,14,14,14,12.
# REFUTED: The parent's embedded constituent witnesses protect its top proxy candidate from mixed-sector weight twelve; slot 31 fell from d2s=14 to d30s=12 and touched Arow0/Brow0.
# REFUTED: Saturated nonuniform B degrees plus C3/C9 screening supply five Dic9 readings >=16; all five measured current finalists instead had d30s=13, with predominantly sector-1 X witnesses.
# REFUTED: AP/Sidon and gap-scale C37 cards survive as twenty inequivalent initial designs; measured slots 13,15,17,19 were iso_dup, and the sixty-slot portfolio retained only 31 distinct codes.
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
