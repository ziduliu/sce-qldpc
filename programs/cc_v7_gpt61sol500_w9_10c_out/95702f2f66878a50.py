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
CONCEPT = ("Metacyclic C5 semidirect_4 C2, order 10, A=B=(3,7): verified length-fifteen triangle cores and witness-repaired cubic attachments should supply five fresh measured distance readings >=8 in cell 0.300.")
GROUP = ("Keep the metacyclic semidirect-product construction rule. Factor 58 excludes order twenty and permits q<=10; parameters m=5,r=2,s=4 give the largest admissible lift, n=580. The recorded order-ten (3,7)x(3,7) bed reached fitness 13.5, while the parent order-twenty bed measured five distance-seven finalists. Stronger attachment screening, not an assumed group advantage, is the proposed improvement.")
SHAPES = ("Change both sides to (3,7), giving equal 70-variable, 30-check constituents. n=580, kb=160 and guaranteed rate 16/58=0.275862 target cell 0.300. Parent weight-seven minima occupied sector-1 Acol/Bcol blocks, often Acol1/2, without reported sector-2 minima. Three-row triangle cores and four cubic columns replace the parent's two-row, two-cubic geometry. Verified full ranks retain [0.275,0.325), rather than assuming k equals its bound.")
FILLING = ("Twenty normalized cubic-support/element prescriptions have three degree-two core columns and four degree-three attachment columns, six terms per constituent row, and BOTH CSS maxima nine. Verified core cycles have weight fifteen. Exact pair/triple and anchored quartet/triple screens exclude constituent kernels through seven when possible; floor-seven fallbacks are recorded explicitly. Actual short witnesses guide single-label repair. Complete verified lifts enforce target rate, CSS, full kernel-four exclusion and intrinsic uncoloured Tanner separation. Embedded and randomized nontrivial logical witnesses rank an immutable portfolio. Constituent floors do not certify mixed-sector quantum distance, and novelty against the unseen archive remains measured.")
FAMILY_NAME = "r030_m5r2s4_37x37_triangle15_cubic_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    # Same metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 58 requires q <= 10; inversion has order two modulo five.
    m, r, s = 5, 2, 4
    if not valid_q(q):
        raise ValueError("this metacyclic group has order 10")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + powers[g[1] % r] * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-powers[bi] * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 7)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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
    """Extract an arbitrary-shaped constituent using the verified lift."""
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


def NEW_CTX(rng, G, idx):
    """Pure lookup; the old shape-specific portfolio is never called."""
    catalogue = _t7_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("triangle-core target-rate portfolio exhausted")
    quality, A, B, k, reading, cards, floors = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "design_cards": cards, "constituent_kernel_floors": floors,
            "certified_logical_floor": 5, "quality": quality}


from functools import lru_cache


def _t7_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _t7_basis(rows):
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


def _t7_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _t7_probe(rows, width, local, passes, seed=0, own=None):
    """Return an attained kernel/logical witness, never a distance cap."""
    cols = _t7_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _t7_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, nulls = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                nulls.append(word)
                inspect(word)
        local.shuffle(nulls)
        for i in range(len(nulls)):
            word = nulls[i]
            for j, other in enumerate(nulls):
                if i == j:
                    continue
                trial = word ^ other
                if 0 < trial.bit_count() < word.bit_count():
                    word = trial
            nulls[i] = word
            inspect(word)
        short = sorted(nulls, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                for h in range(j):
                    inspect(pair ^ short[h])
        if own is not None and weight <= 7:
            break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _t7_outside(best, own)
    return best


def _t7_screen(rows, width, q, core_word):
    """Exact constituent exclusion through seven, or a real short word.

    Returns below seven are witnesses, not asserted minimum distances.
    Seven/eight returns certify absence below that truncated floor.
    Regular-action translation anchors constituent supports only; it
    is never assumed for the complete quantum code.
    """
    from itertools import combinations
    cols = _t7_columns(rows, width)
    bits = [1 << j for j in range(width)]
    seen = {0: 0}
    for j, syndrome in enumerate(cols):
        if syndrome in seen:
            return (bits[j] ^ seen[syndrome]).bit_count(), \
                bits[j] ^ seen[syndrome]
        seen[syndrome] = bits[j]
    for j, first in enumerate(cols):
        for i in range(j):
            syndrome = first ^ cols[i]
            word = bits[j] | bits[i]
            old = seen.get(syndrome)
            if old is not None:
                witness = word ^ old
                return witness.bit_count(), witness
            seen[syndrome] = word

    triples = {}
    for j, first in enumerate(cols):
        for i in range(j):
            pair = first ^ cols[i]
            word = bits[j] | bits[i]
            for h in range(i):
                syndrome = pair ^ cols[h]
                support = word | bits[h]
                old = seen.get(syndrome)
                if old is not None:
                    witness = support ^ old
                    return witness.bit_count(), witness
                seen[syndrome] = support
                triples[syndrome] = support
    del seen

    # Every seven-word has a regular-action translate containing an
    # identity-coordinate variable in one of the seven blocks.
    for anchor in range(0, width, q):
        others = [j for j in range(width) if j != anchor]
        for a, b, c in combinations(others, 3):
            old = triples.get(
                cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c])
            if old is not None:
                witness = old ^ bits[anchor] ^ bits[a] ^ bits[b] ^ bits[c]
                assert witness.bit_count() == 7
                assert all((row & witness).bit_count() % 2 == 0
                           for row in rows)
                return 7, witness
    assert all((row & core_word).bit_count() % 2 == 0 for row in rows)
    return 8, core_word


def _t7_core(G, voltage):
    """Verified three-edge triangle cover and an actual shortest cycle."""
    e, q = G.e, G.q
    C = [[[e], [], [e]], [[e], [e], []],
         [[], [e], [voltage]]]
    rows = _component_rows(G, "A", C)
    cols = _t7_columns(rows, 3*q)
    adjacency = [[] for _ in rows]
    for j, syndrome in enumerate(cols):
        assert syndrome.bit_count() == 2
        bit = syndrome & -syndrome
        a = bit.bit_length() - 1
        b = (syndrome ^ bit).bit_length() - 1
        adjacency[a].append((b, j))
        adjacency[b].append((a, j))
    shortest = 0
    for root in range(len(rows)):
        paths = [None] * len(rows)
        paths[root] = 0
        queue = [root]
        for a in queue:
            for b, edge in adjacency[a]:
                if paths[b] is None:
                    paths[b] = paths[a] ^ (1 << edge)
                    queue.append(b)
                else:
                    word = paths[a] ^ paths[b] ^ (1 << edge)
                    if word and (not shortest or
                                 word.bit_count() < shortest.bit_count()):
                        shortest = word
    assert shortest and shortest.bit_count() == 15
    assert all((row & shortest).bit_count() % 2 == 0 for row in rows)
    return C, shortest


def _t7_graph_key(hx, hz, n):
    """Canonical Tanner certificate, or conservative intrinsic invariants."""
    from collections import Counter
    from hashlib import sha256
    rows = list(hx) + list(hz)
    m = len(rows)
    adjacency = [[] for _ in range(m+n)]
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            j = m + bit.bit_length() - 1
            adjacency[i].append(j)
            adjacency[j].append(i)
            row ^= bit
    try:
        import pynauty
    except ImportError:
        pynauty = None
    if pynauty is not None:
        degrees = {}
        for i, neighbours in enumerate(adjacency):
            degrees.setdefault(len(neighbours), set()).add(i)
        graph = pynauty.Graph(
            number_of_vertices=m+n, directed=False,
            adjacency_dict={i: a for i, a in enumerate(adjacency)},
            vertex_coloring=[degrees[d] for d in sorted(degrees)])
        return pynauty.certificate(graph)

    # No block, group-coordinate or X/Z labels enter this refinement.
    colors = [len(a) for a in adjacency]
    histograms = []
    for _ in range(6):
        features = [(colors[i], tuple(sorted(Counter(
            colors[j] for j in neighbours).items())))
            for i, neighbours in enumerate(adjacency)]
        histograms.append(tuple(sorted(Counter(features).items())))
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]
    value = (tuple(histograms),
             _tanner_signature(list(hx), list(hz)))
    # Equality discards candidates; hash collisions cannot create novelty.
    return sha256(repr(value).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _t7_portfolio(G, count):
    """Immutable optimized portfolio, independent of slot call order."""
    from itertools import combinations_with_replacement, permutations

    q, e, width = G.q, G.e, 7*G.q
    assert q == 10 and A_SHAPE == B_SHAPE == (3, 7)
    cap = min(9, W_MAX)
    if cap < 9:
        raise ValueError("triangle/cubic profiles require row cap nine")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    # Enumerate cubic allocations with four attachment terms per row.
    allocations = [
        (a, b, 3-a-b)
        for a in range(4) for b in range(4-a)]
    profiles = set()
    for columns in combinations_with_replacement(allocations, 4):
        if any(sum(c[i] for c in columns) != 4 for i in range(3)):
            continue
        canonical = min(
            tuple(sorted(tuple(c[i] for i in p) for c in columns))
            for p in permutations(range(3)))
        profiles.add(canonical)
    profiles = sorted(profiles, key=lambda p: (
        -p.count((1, 1, 1)), max(max(c) for c in p), p))
    assert profiles
    cores = [_t7_core(G, (a, 0)) for a in (1, 2)]

    bank, component_keys, used_fillings = [], set(), set()
    for card in range(20):
        profile = profiles[card % len(profiles)]
        base, core_word = cores[(card // len(profiles)) % len(cores)]
        local = random.Random(BASE_SEED ^ (0x103715 + 104729*card))
        records, cursor = [], None
        for trial in range(256):
            if cursor is not None and trial % 6:
                old, witness = cursor
                C = [[list(cell) for cell in row] for row in old]
                touched = [j for j in range(3, 7)
                           if (witness >> (j*q)) & ((1 << q)-1)]
                if not touched:
                    raise ValueError("short dependency unexpectedly lies in core")
                j = local.choice(touched)
                active = [i for i in range(3) if C[i][j]]
                i = local.choice(active)
                position = local.randrange(len(C[i][j]))
                replacement = G.pick(local)
                while replacement in C[i][j]:
                    replacement = G.pick(local)
                C[i][j][position] = replacement
            else:
                C = [[list(cell) for cell in row] + [[] for _ in range(4)]
                     for row in base]
                for j, counts in enumerate(profile, 3):
                    for i, terms in enumerate(counts):
                        C[i][j] = local.sample(G.elements, terms)
                    # Remove a whole-column gauge, not a design variable.
                    first = next(cell[0] for cell in
                                 (C[i][j] for i in range(3)) if cell)
                    inverse = G.inv(first)
                    for i in range(3):
                        C[i][j] = [G.mul(g, inverse) for g in C[i][j]]

            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in used_fillings:
                continue
            used_fillings.add(filling)
            counts = [[len(cell) for cell in row] for row in C]
            assert all(sum(row) == 6 for row in counts)
            assert [sum(row[j] for row in counts)
                    for j in range(7)] == [2, 2, 2, 3, 3, 3, 3]
            assert _no_dead_lines(counts, counts)
            assert _grid_weights(counts, counts) == 9
            rows = _component_rows(G, "A", C)
            if gf2_rank(rows) != 3*q:
                continue
            floor, witness = _t7_screen(rows, width, q, core_word)
            if floor < 8:
                cursor = C, witness
            else:
                cursor = None
            if floor < 7:
                continue
            penalty = sum(
                (rows[i] & rows[j]).bit_count()
                * ((rows[i] & rows[j]).bit_count()-1) // 2
                for i in range(len(rows)) for j in range(i))
            records.append((floor, penalty, trial, C, rows, witness))
            if sum(r[0] == 8 for r in records) >= 5:
                break

        records.sort(key=lambda r: (-r[0], r[1], r[2]))
        kept = 0
        for floor, penalty, trial, C, rows, seed in records:
            key = _t7_graph_key(rows, [], width)
            if key in component_keys:
                continue
            word = _t7_probe(rows, width, local, 10, seed)
            if not word or word.bit_count() < floor:
                raise ValueError("probe contradicted exact constituent floor")
            right = _component_rows(G, "B", C)
            if gf2_rank(right) != 3*q:
                continue
            right_word, source = 0, word
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length()-1, q)
                right_word |= 1 << (j*q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            assert all((row & right_word).bit_count() % 2 == 0
                       for row in right)
            component_keys.add(key)
            bank.append(dict(
                grid=freeze(C), floor=floor, penalty=penalty,
                left=word, right=right_word, reading=word.bit_count(),
                card=card, trial=trial))
            kept += 1
            if kept >= 4:
                break
    if len(bank) < 4:
        raise ValueError("too few repaired triangle/cubic constituents")
    bank.sort(key=lambda r: (
        -r["floor"], -r["reading"], r["penalty"], r["card"], r["trial"]))

    def embedded(record, kind, checks, stabilizers, own):
        best = 0
        for block in range(7):
            if kind == "X":
                word = record["right"] << (block*width)
            else:
                word, source = 0, record["left"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*7+block)*q+t)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _t7_outside(word, own):
                continue
            for _ in range(2):
                starting = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                moves = sorted(
                    (row for row in stabilizers
                     if (row & word).bit_count() >= 2),
                    key=lambda row: -(row & word).bit_count())[:40]
                starting_pair = word
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting_pair ^ first ^ second
                        if trial.bit_count() < word.bit_count():
                            word = trial
                if word == starting:
                    break
            assert _t7_outside(word, own)
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            priority = (
                min(a["floor"], b["floor"]),
                min(a["reading"], b["reading"]),
                -a["penalty"]-b["penalty"],
                int(a["card"] != b["card"]))
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    results, graph_keys, tested = [], set(), 0
    for priority, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 580 or not (275*n <= 1000*k < 325*n):
            continue
        assert max(row.bit_count() for row in hx+hz) <= cap
        # Complete matrices, including sector 2, must exclude kernels <=4.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        key = _t7_graph_key(hx, hz, n)
        if key in graph_keys:
            continue
        graph_keys.add(key)
        tested += 1
        if tested > max(240, 6*count):
            break
        if not css_ok(hx, hz):
            raise ValueError("verified triangle/cubic lift failed CSS")
        bx, bz = _t7_basis(hx), _t7_basis(hz)
        xseed = embedded(b, "X", hz, hx, bx)
        zseed = embedded(a, "Z", hx, hz, bz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0x1037CC + 65537*ai + bi))
        xword = _t7_probe(hz, n, local, 3, xseed, bx)
        zword = _t7_probe(hx, n, local, 3, zseed, bz)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 7:
            continue
        assert lifted_k(G, A, B) == k
        quality = (k*reading*reading/n, priority, k)
        results.append((quality, A, B, k, reading,
                        (a["card"], b["card"]),
                        (a["floor"], b["floor"])))
        if len(results) >= count + 16:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


@lru_cache(maxsize=1)
def _cubic_core_portfolio(G):
    """Memoization changes cost, not the deterministic published cells.

    All matrix screening uses lifted_matrices. Unequal graph invariants
    certify within-program inequivalence; unseen archive novelty remains
    the evaluator's responsibility.
    """
    from collections import Counter
    from hashlib import sha256

    assert G.q == 20 and A_SHAPE == B_SHAPE == (2, 5)
    local = random.Random(BASE_SEED ^ 0x205C081)
    q, width, cap = G.q, 5 * G.q, min(9, W_MAX)
    if cap < 9:
        raise ValueError("cubic-core profiles require row cap nine")

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

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
        out = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = out.get(p)
                if old is None:
                    out[p] = word
                    break
                word ^= old
        return out

    def outside(word, span):
        while word:
            p = word.bit_length() - 1
            old = span.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def clean_six(rows, n):
        """Exactly exclude all nonzero constituent words of weight <=6.

        Every such support splits into two supports of size at most
        three. Equal syndromes of distinct supports detect their XOR.
        """
        cols = columns(rows, n)
        seen = {0}
        for syndrome in cols:
            if syndrome in seen:
                return False
            seen.add(syndrome)
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        for j, first in enumerate(cols):
            for i in range(j):
                pair = first ^ cols[i]
                for h in range(i):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return False
                    seen.add(syndrome)
        return True

    def quantum_four(checks, own, n):
        """Exact logical exclusion through four; short stabilizers are legal."""
        span = basis(own)
        pivots = sorted(span, reverse=True)
        residues = []
        for j in range(n):
            word = 1 << j
            for p in pivots:
                if (word >> p) & 1:
                    word ^= span[p]
            residues.append(word)

        cols = columns(checks, n)
        seen = {0: 0}
        for j, syndrome in enumerate(cols):
            residue = residues[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = residue
            elif old != residue:
                return False
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                residue = residues[j] ^ residues[i]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = residue
                elif old != residue:
                    return False
        return True

    def probe(rows, n, rounds, own=None):
        """Return an actual kernel/logical witness weight, or zero.

        An unresolved probe receives no optimistic sentinel. Every
        inspected support is constructed in the verified matrix kernel.
        """
        cols = columns(rows, n)
        span = basis(own) if own is not None else None
        order = list(range(n))
        best = 0

        def inspect(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            if best and weight >= best:
                return
            if span is not None and not outside(word, span):
                return
            best = weight

        for _ in range(rounds):
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
                if not syndrome:
                    nulls.append(word)
                    inspect(word)

            short = sorted(nulls, key=int.bit_count)[:36]
            for word in nulls:
                for other in short:
                    candidate = word ^ other
                    if 0 < candidate.bit_count() < word.bit_count():
                        word = candidate
                inspect(word)
            for j, first in enumerate(short):
                for i in range(j):
                    pair = first ^ short[i]
                    inspect(pair)
                    for h in range(i):
                        inspect(pair ^ short[h])
            if best and best <= 6:
                break
        return best

    def graph_key(hx, hz, n):
        """Intrinsic uncoloured full-Tanner invariant, not label data."""
        rows = list(hx) + list(hz)
        value = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count()
                                 for c in columns(rows, n)).items())),
            _tanner_signature(list(hx), list(hz)),
        )
        return sha256(repr(value).encode("ascii")).digest()

    def core_info(g, h):
        core = [[[G.e], [G.e], [G.e]],
                [[G.e], [g], [h]]]
        rows = _component_rows(G, "B", core)
        cols = columns(rows, 3 * q)
        adj = [[] for _ in rows]
        for edge, syndrome in enumerate(cols):
            if syndrome.bit_count() != 2:
                return None
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            adj[a].append((b, edge))
            adj[b].append((a, edge))
        girth = len(rows) + 1
        for root in range(len(rows)):
            distance = [-1] * len(rows)
            parent_edge = [-1] * len(rows)
            distance[root] = 0
            queue = [root]
            for a in queue:
                for b, edge in adj[a]:
                    if distance[b] < 0:
                        distance[b] = distance[a] + 1
                        parent_edge[b] = edge
                        queue.append(b)
                    elif edge != parent_edge[a] and edge != parent_edge[b]:
                        girth = min(girth, distance[a] + distance[b] + 1)
        return girth

    cores = []
    for g in G.elements:
        if g == G.e:
            continue
        for h in G.elements:
            if h == G.e or h == g:
                continue
            girth = core_info(g, h)
            if girth is not None and girth >= 8:
                cores.append((girth, g, h))
    if not cores:
        raise ValueError("no verified matching core of girth eight")
    best_girth = max(item[0] for item in cores)
    cores = [item for item in cores if item[0] == best_girth]

    def element_order(g):
        power = G.e
        for t in range(1, q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return t
        raise ValueError("invalid group order")

    rotations = [g for g in G.elements if element_order(g) == 5]
    quartics = [g for g in G.elements if element_order(g) == 4]
    long = rotations + quartics

    def draw(profile, policy):
        _, g, h = local.choice(cores)
        C = [[[] for _ in range(5)] for _ in range(2)]
        mono = (2, 3, 4) if profile == 0 else (1, 2, 3)
        for j, relative in zip(mono, (G.e, g, h)):
            C[0][j], C[1][j] = [G.e], [relative]

        pool = rotations if policy == 0 else quartics if policy == 1 else long
        if profile == 0:
            u, v = local.choice(pool), local.choice(pool)
            p, t = G.pick(local), G.pick(local)
            C[0][0], C[1][0] = [G.e, u], [p]
            C[0][1], C[1][1] = [G.e], [t, G.mul(t, v)]
        else:
            for row, j in ((0, 0), (1, 4)):
                for _ in range(64):
                    u, v = local.choice(pool), local.choice(pool)
                    if u == v:
                        continue
                    commute = G.mul(u, v) == G.mul(v, u)
                    if policy == 3 and commute:
                        continue
                    if policy == 4 and not commute:
                        continue
                    C[row][j] = [G.e, u, v]
                    break
                if not C[row][j]:
                    return None
        return C

    # Build new constituent matrices, not perturbations of archived cells.
    bank, fill_keys, component_keys = [], set(), set()
    for attempt in range(8192):
        profile, policy = attempt % 2, (attempt // 2) % 5
        C = draw(profile, policy)
        if C is None:
            continue
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in C for cell in row)
        if filling in fill_keys:
            continue
        fill_keys.add(filling)
        counts = [[len(cell) for cell in row] for row in C]
        assert all(sum(row) == 6 for row in counts)
        assert max(sum(row[j] for row in counts) for j in range(5)) == 3
        assert _no_dead_lines(counts, counts)
        assert _grid_weights(counts, counts) == 9

        left = _component_rows(G, "A", C)
        right = _component_rows(G, "B", C)
        if gf2_rank(left) != 2 * q or gf2_rank(right) != 2 * q:
            continue
        if _kernel_profile(left, width)[0] != 5:
            continue
        if not clean_six(left, width) or not clean_six(right, width):
            continue
        signature = _tanner_signature(left, [])
        if signature in component_keys:
            continue
        component_keys.add(signature)
        dl, dr = probe(left, width, 10), probe(right, width, 10)
        if not dl or not dr:
            continue
        assert min(dl, dr) >= 7
        bank.append((min(dl, dr), profile, policy, freeze(C), dl, dr))
        if len(bank) >= 64:
            break
    if len(bank) < 8:
        raise ValueError("too few distinct kernel-six-clean constituents")
    bank.sort(key=lambda item: (-item[0], item[1], item[2],
                                repr(item[3])))

    # Reprobe stronger candidates with the same extra budget.
    refined = []
    for entry in bank[:40]:
        _, profile, policy, C, dl, dr = entry
        dl = min(dl, probe(_component_rows(G, "A", C), width, 24))
        dr = min(dr, probe(_component_rows(G, "B", C), width, 24))
        refined.append((min(dl, dr), profile, policy, C, dl, dr))
    bank = sorted(refined, key=lambda item:
                  (-item[0], item[1], item[2], repr(item[3])))

    excluded = set()
    # The reported F20 frontier filling may use action 2 or its inverse.
    reference = [
        [[(0, 1), (2, 1)], [(3, 2)], [G.e], [G.e], [G.e]],
        [[(0, 2)], [G.e, (3, 1)], [G.e], [(2, 0)], [(1, 3)]],
    ]
    for sign in (1, -1):
        C = [[[(a, sign * b % 4) for a, b in cell]
              for cell in row] for row in reference]
        hx, hz, n = lifted_matrices(G, C, C)
        excluded.add(graph_key(hx, hz, n))

    plans = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    tested = set()
    entries = []
    for position in range(N_SLOTS):
        card, epoch = position % 20, position // 20
        support_pair = ((1, 1), (0, 1), (0, 0))[card % 3]
        pressure = 1 + (card // 3 + epoch) % 3
        ranked = list(plans)
        local.shuffle(ranked)
        ranked.sort(key=lambda ab: (
            tuple(sorted((bank[ab[0]][1], bank[ab[1]][1]))) != support_pair,
            -(pressure * bank[ab[0]][0]
              + (4 - pressure) * bank[ab[1]][0]),
            bank[ab[0]][2] == bank[ab[1]][2],
        ))
        survivors = []
        for a, b in ranked:
            if (a, b) in tested:
                continue
            tested.add((a, b))
            A, B = bank[a][3], bank[b][3]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 580 or k != 180:
                continue
            if not (275 * n <= 1000 * k < 325 * n):
                continue
            signature = graph_key(hx, hz, n)
            if signature in excluded:
                continue
            if not quantum_four(hz, hx, n):
                continue
            if not quantum_four(hx, hz, n):
                continue

            # Reserve even discarded survivors, preventing slot recycling.
            excluded.add(signature)
            dx = probe(hz, n, 3, hx)
            dz = probe(hx, n, 3, hz)
            reading = min(dx, dz) if dx and dz else 5
            quality = (
                k * reading * reading / float(n),
                min(bank[a][0], bank[b][0]),
                -(pressure * bank[a][4] + (4 - pressure) * bank[b][5]),
            )
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 9
            survivors.append((quality, A, B, k, reading,
                              (card, epoch, bank[a][1:3], bank[b][1:3])))
            if len(survivors) >= 3:
                break
        if not survivors:
            raise ValueError("invariant-distinct cubic-core portfolio exhausted")
        winner = max(survivors, key=lambda item: item[0])
        _, A, B, k, reading, card_info = winner
        entries.append((A, B, k, reading, card_info))
    return tuple(entries)


def _inactive_order55_catalogue(rng, G, idx):
    """Historical reference only; NEW_CTX never calls this old catalogue."""
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
# REFUTED: Exhaustive canonical binomial selection supplies five fresh k=226 finalists: the current evaluation measured one fresh k=226 code and four k=222 codes, while slots 0 and 1 reproduced archived codes.
# REFUTED: The order-twenty cubic-core portfolio supplies five fresh distance-eight readings: all five current measured codes were [580,180,7], and sixty slots retained only six distinct classes.
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
