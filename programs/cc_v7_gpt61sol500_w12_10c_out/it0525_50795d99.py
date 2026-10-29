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
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("S4 x C2, order 48, A=(1,2), B=(2,5): noncommuting four-term A constituents and degree-three/four B columns, independently optimized against actual witnesses, should produce five fresh measured readings >=14 and improve cell-0.250 fitness above 44.8. This is a falsifiable measurement target, not a distance certificate.")
GROUP = ("L3 jump from cyclic C28 to the direct product S4 x C2. The new group has centre order two and ten conjugacy classes; this order-48 rectangular bed is absent from BEDS. Other order-48 groups on (1,2)x(2,5) reached approximately 43..45, motivating a controlled comparison without assuming a group-distance advantage. Noncommuting permutation voltages replace cyclic polynomial syzygies.")
SHAPES = ("Target cell 0.250 with (1,2)x(2,5): n=12*48=576, kb=3*48=144 and guaranteed rate 1/4. Unequal constituents have variables/checks 96/48 and 240/96. The size window permits q<=49, nearly saturated by order 48. Parent weight-fourteen minima were entirely sector one, especially Acol1/Acol2 and Bcol1/Bcol2. Replace both constituent geometries and require exact assembled rates in [0.225,0.275), while full logical searches inspect both sectors.")
FILLING = ("Twenty primary cards cross five B incidence architectures with four A occupancy profiles in the intrinsic sign/C2 quotient. A cells have four distinct terms, row total eight and column totals four; B rows total eight and columns have at most four terms, giving both CSS row caps twelve. Compile every component action exclusively from scaffold unit lifts. Independently select true component witnesses, exactly exclude component kernels through six, repair witness-heavy cells, and equally deepen survivors. Exact assembled ranks, complete kernel-through-four exclusion, syndrome-verified embeddings, stabilizer-aware full logical searches and unequal uncoloured Tanner fingerprints gate emission. No archived filling or orbit-copy fallback is used; bounded-search readings remain upper witnesses.")
FAMILY_NAME = "r025_S4xC2_12x25_quotientoccupancy_component6_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # (p0,p1,p2,p3,c): permutation p in S4, central c in C2.
    # Product permutation is composition p after h.
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("S4 x C2 requires order 48")
    ps = list(permutations(range(4)))
    els = [p + (c,) for c in range(2) for p in ps]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        p = tuple(int(x) for x in g[:4])
        if len(p) != 4 or sorted(p) != [0, 1, 2, 3]:
            raise ValueError("invalid S4 permutation")
        return p + (int(g[4]) % 2,)

    def mul(g, h):
        g, h = red(g), red(h)
        return tuple(g[h[i]] for i in range(4)) + ((g[4]+h[4]) % 2,)

    def inv(g):
        g = red(g)
        p = [0]*4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + (g[4],)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0,1,2,3,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

from itertools import combinations, combinations_with_replacement
from functools import lru_cache


# Five component-pair architectures, crossed with four coordinate models.
# Additional harness lanes change B's relative quotient patterns.
_AFF_PATTERNS = (
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (2, 6), (3, 5),
)
_AFF_CARDS = (
    (0, 0, 0, 0, 4, 6),
    (0, 1, 2, 3, 4, 5),
    (0, 2, 4, 0, 3, 7),
    (0, 4, 6, 1, 3, 5),
    (0, 1, 4, 0, 2, 7),
)


def _aff_columns(rows, width):
    out = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            out[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    return tuple(out)


def _aff_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous
    return basis


def _aff_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _aff_field_mul(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 8:
            a ^= 11
    return result


def _aff_trace(a):
    square = _aff_field_mul(a, a)
    return a ^ square ^ _aff_field_mul(square, square)


@lru_cache(maxsize=1)
def _aff_atlas(G):
    """Both regular actions are extracted from verified unit lifts."""
    q = G.q
    mask = (1 << q) - 1
    stub = [[[G.e], [G.e], [G.e]]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(_aff_columns(
                [row & mask for row in source[:q]], q))
        atlas[side] = tuple(blocks)
    return atlas


def _aff_component_columns(G, atlas, side, grid):
    """Linear superposition of scaffold-generated single-term actions."""
    columns = [0] * (3 * G.q)
    for j, cell in enumerate(grid[0]):
        for g in cell:
            for h, value in enumerate(atlas[side][G.idx(g)]):
                columns[j * G.q + h] ^= value
    return tuple(columns)


def _aff_clean4(cols):
    """Exact component-kernel exclusion through four.

    Cubic columns have odd weight, so odd kernel weights are impossible.
    Distinct singleton/pair syndromes exclude weights two and four.
    """
    seen = {0}
    for value in cols:
        if value in seen:
            return False
        seen.add(value)
    for i, value in enumerate(cols):
        for other in cols[:i]:
            pair = value ^ other
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _aff_clean6(cols, q):
    """Exact constituent exclusion through six, after clean4.

    Each component regular action is transitive inside each q-column
    block. Anchors 0,q,2q therefore cover constituent weight-six words.
    This symmetry is not assumed for the complete quantum matrices.
    """
    bits = [1 << i for i in range(len(cols))]
    tails = {0: 0}
    incidence = [[] for _ in range(q)]
    for i, value in enumerate(cols):
        tails[value] = bits[i]
        remaining = value
        while remaining:
            bit = remaining & -remaining
            incidence[bit.bit_length() - 1].append(i)
            remaining ^= bit
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            assert syndrome not in tails
            tails[syndrome] = bits[i] | bits[j]

    def exists(value, remaining, word, forbidden):
        if not value:
            return True
        if value.bit_count() > 3 * remaining:
            return False
        if remaining <= 2:
            tail = tails.get(value)
            return (tail is not None
                    and tail.bit_count() <= remaining
                    and not (tail & (word | forbidden)))
        check = (value & -value).bit_length() - 1
        for j in incidence[check]:
            if bits[j] & (word | forbidden):
                continue
            if exists(value ^ cols[j], remaining - 1,
                      word | bits[j], forbidden):
                return True
        return False

    return not any(exists(cols[a], 5, 0, bits[a])
                   for a in range(0, len(cols), q))


def _aff_probe(cols, local, trials, seeds=(), stabilizers=None):
    """Find actual kernel/logical witnesses; never certify a distance.

    All supports use verified column syndromes. Quantum readings test
    the exact stabilizer span and permit sparse stabilizer dressings.
    """
    width = len(cols)
    basis = None if stabilizers is None else _aff_basis(stabilizers)
    memory = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:24]
    best = width + 1

    def consider(word):
        nonlocal best
        if (word and word.bit_count() < best
                and (basis is None or _aff_outside(word, basis))):
            best = word.bit_count()

    for word in memory:
        consider(word)
    order = list(range(width))
    for _ in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                dependencies.append(word)
                consider(word)

        directions = sorted(set(dependencies + memory),
                            key=lambda w: (w.bit_count(), w))[:32]
        discoveries = set(memory)
        for seed in sorted(dependencies + memory,
                           key=int.bit_count)[:20]:
            word = seed
            for sweep in range(2):
                changed = False
                for direction in directions:
                    candidate = word ^ direction
                    if candidate and candidate.bit_count() < word.bit_count():
                        word = candidate
                        changed = True
                if not changed:
                    break
            consider(word)
            discoveries.add(word)

        short = sorted(discoveries, key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    consider(candidate)
                    discoveries.add(candidate)

        if stabilizers is not None:
            for seed in short[:8]:
                word = seed
                for sweep in range(2):
                    previous_weight = word.bit_count()
                    for row in stabilizers:
                        candidate = word ^ row
                        if candidate.bit_count() < word.bit_count():
                            word = candidate
                    if word.bit_count() == previous_weight:
                        break
                consider(word)
                discoveries.add(word)
        memory = sorted(
            (w for w in discoveries if w),
            key=lambda w: (w.bit_count(), w))[:24]
    return best, tuple(memory)


def _aff_draw(local, G, kinds, model, side, lane):
    """Pack directed differences, then leave acceptance to matrix screens."""
    for restart in range(24):
        used, row = set(), []
        for j, kind in enumerate(kinds):
            b, d = _AFF_PATTERNS[kind]
            accepted = None
            for attempt in range(96):
                if model == 0:
                    a, c = local.randrange(8), local.randrange(8)
                elif model == 1:
                    a, c = local.sample(range(1, 8), 2)
                elif model == 2:
                    a = local.randrange(8)
                    c = a ^ (1, 2, 4)[(j + lane) % 3]
                else:
                    a = local.choice([x for x in range(8)
                                      if _aff_trace(x) == 0])
                    c = local.choice([x for x in range(8)
                                      if _aff_trace(x) == 1])
                g, h = (a, b), (c, d)
                if G.mul(g, h) == G.mul(h, g):
                    continue
                cell = (G.e, g, h)
                differences = {
                    (G.mul(x, G.inv(y)) if side == "A"
                     else G.mul(G.inv(x), y))
                    for x in cell for y in cell if x != y
                }
                if len(differences) != 6 or differences & used:
                    continue
                accepted = cell
                used.update(differences)
                break
            if accepted is None:
                break
            row.append(accepted)
        if len(row) == 3:
            return (tuple(row),)
    return None


def _aff_component_pool(local, G, atlas, side, kinds, model, lane, batch):
    """Independent constituent selection and equal-budget rescreening."""
    pool, seen = [], set()
    for attempt in range(64 + 32 * batch):
        grid = _aff_draw(local, G, kinds, model, side, lane)
        if grid is None or grid in seen:
            continue
        seen.add(grid)
        cols = _aff_component_columns(G, atlas, side, grid)
        assert all(value.bit_count() == 3 for value in cols)
        if not _aff_clean4(cols):
            continue
        reading, words = _aff_probe(cols, local, 2)
        if reading <= 6:
            continue
        rank = gf2_rank(cols)
        item = dict(grid=grid, cols=cols, reading=reading,
                    words=words, rank=rank, attempt=attempt)
        pool.append(item)
        pool.sort(key=lambda x: (
            -x["reading"], x["rank"], x["attempt"]))
        del pool[6:]

    survivors = []
    for item in pool:
        if not _aff_clean6(item["cols"], G.q):
            continue
        reading, words = _aff_probe(
            item["cols"], local, 16, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        survivors.append(item)
    survivors.sort(key=lambda x: (
        -x["reading"], x["rank"], x["attempt"]))
    return survivors[:3]


def _aff_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; inequality certifies nonisomorphism.

    Equality is rejected conservatively, not claimed to prove isomorphism.
    Check types and protograph labels are deliberately absent.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * len(rows)
    overlaps = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            if size:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlaps[i][j] = overlaps[j][i] = size
    keys = []
    for i, neighbors in enumerate(overlaps):
        profile = Counter(
            (weights[j], size,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, size in neighbors.items())
        keys.append((weights[i], tuple(sorted(profile.items()))))
    history = []
    for _ in range(2):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: t for t, key in enumerate(sorted(histogram))}
        colours = [palette[key] for key in keys]
        keys = [
            (colours[i], tuple(sorted(Counter(
                (colours[j], size)
                for j, size in neighbors.items()).items())))
            for i, neighbors in enumerate(overlaps)
        ]
    incidents = [[] for _ in range(n)]
    for weight, source in zip(weights, rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length() - 1].append(weight)
            source ^= bit
    return (n, tuple(history), tuple(sorted(Counter(
        tuple(sorted(v)) for v in incidents).items())))


def _small_kernel_stats(rows, n):
    """Exact commuting-vector screening through weight three.

    With distinct nonzero column syndromes, equal pair syndromes
    involve disjoint pairs. Every weight-four kernel vector has
    exactly three pair partitions, permitting exact counting.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= bit
            row ^= lo

    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None

    pairs = {}
    for i, a in enumerate(cols):
        for j in range(i + 1, n):
            syndrome = a ^ cols[j]
            if syndrome in singles:
                return None
            packed = i * n + j
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = packed
            elif isinstance(previous, int):
                pairs[syndrome] = [previous, packed]
            else:
                previous.append(packed)

    histogram = {}
    participation = [0] * n
    collisions = 0
    for syndrome, value in pairs.items():
        count = 1 if isinstance(value, int) else len(value)
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
        if count > 1:
            collisions += count * (count - 1) // 2
            for packed in value:
                i, j = divmod(packed, n)
                participation[i] += count - 1
                participation[j] += count - 1

    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in cols),
            tuple(p // 3 for p in participation))


def _incidence_fingerprint(hx, hz, sx, sz):
    # Invariant under qubit/check permutations and a global X/Z swap.
    joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
    swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
    row_profiles = tuple(sorted((
        tuple(sorted(r.bit_count() for r in hx)),
        tuple(sorted(r.bit_count() for r in hz)))))
    return (tuple(sorted((sx[1], sz[1]))),
            min(joint, swapped), row_profiles)


@lru_cache(maxsize=1)
def _design_bank(G):
    """Fresh deterministic policy bank; no archived or orbit-copy fallback."""
    assert G.q == 56 and A_SHAPE == B_SHAPE == (1, 3)
    q = G.q
    atlas = _aff_atlas(G)
    mask = (1 << q) - 1
    counts = [[3, 3, 3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)
    emitted, bank = set(), []

    for slot in range(N_SLOTS):
        architecture, model = divmod(slot % 20, 4)
        lane = slot // 20
        card = _AFF_CARDS[architecture]
        ak = card[:3]
        bk = tuple((kind + lane) % len(_AFF_PATTERNS)
                   for kind in card[3:])
        winner = None

        for batch in range(3):
            local = random.Random(
                BASE_SEED * 100003 + 56040073
                + 100019 * slot + 7919 * batch)
            ap = _aff_component_pool(
                local, G, atlas, "A", ak, model, lane, batch)
            bp = _aff_component_pool(
                local, G, atlas, "B", bk, model, lane + 1, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    reading = min(a["reading"], b["reading"])
                    proposals.append((
                        (reading, a["reading"] + b["reading"]),
                        a, b))
            proposals.sort(key=lambda item: item[0], reverse=True)

            finalists = []
            for _, a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (375*n <= 1000*k < 425*n):
                    continue

                # Compare the compiled columns with the complete verified
                # construction before using constituent selection readings.
                ar = [
                    sum(((row >> (3*j*q)) & mask) << (j*q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
                assert _aff_columns(ar, 3*q) == a["cols"]
                assert _aff_columns(br, 3*q) == b["cols"]

                sx = _small_kernel_stats(hx, n)
                if sx is None or sx[0]:
                    continue
                sz = _small_kernel_stats(hz, n)
                if sz is None or sz[0]:
                    continue
                fingerprint = _aff_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue

                # These are complete-code logical witnesses, including
                # both sectors, classified modulo exact stabilizer spans.
                dx, xwords = _aff_probe(
                    _aff_columns(hz, n), local, 4, stabilizers=hx)
                dz, zwords = _aff_probe(
                    _aff_columns(hx, n), local, 4, stabilizers=hz)
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    a=a, b=b, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    fingerprint=fingerprint))
                if len(finalists) == 3:
                    break

            for item in finalists:
                dx, _ = _aff_probe(
                    _aff_columns(item["hz"], item["n"]), local, 8,
                    item["xwords"], item["hx"])
                dz, _ = _aff_probe(
                    _aff_columns(item["hx"], item["n"]), local, 8,
                    item["zwords"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                # Classical witnesses are conservative selection heuristics;
                # they are not automatically claimed as quantum logicals.
                reading = min(
                    item["a"]["reading"], item["b"]["reading"],
                    item["dx"], item["dz"])
                objective = (
                    item["k"] * reading * reading / item["n"],
                    reading, item["k"],
                    item["a"]["reading"] + item["b"]["reading"])
                if winner is None or objective > winner[0]:
                    winner = (objective, item)
            if winner is not None and winner[0][1] >= 9:
                break

        if winner is None:
            # An exhausted policy is voided rather than refilled with
            # another slot's code or an automorphism image.
            bank.append(None)
            continue
        objective, item = winner
        A, B = item["A"], item["B"]
        assert item["n"] == 560 and item["k"] >= 224
        assert 375*item["n"] <= 1000*item["k"] < 425*item["n"]
        assert css_ok(item["hx"], item["hz"])
        assert max(row.bit_count()
                   for row in item["hx"] + item["hz"]) == 12
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(item["fingerprint"])
        bank.append((A, B))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five fresh affine designs passed screening")
    return tuple(bank)


# Active C28 optimizer. Historical affine policies above are not called.
# Generic column, basis, witness and incidence helpers remain applicable.
_C28_GAPS = (
    None,
    ((1, 3, 5, 11), (2, 3, 6, 5)),
    ((1, 3, 5, 9), (2, 7, 4, 11)),
    ((1, 3, 9, 13), (6, 5, 10, 7)),
    ((1, 5, 11, 13), (4, 3, 12, 9)),
)


def _c28_syndrome(cols, word):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length() - 1]
        word ^= bit
    return answer


def _c28_components(hx, hz, q):
    """Extract both rectangular components from the verified full lift."""
    mask = (1 << q) - 1
    ar = [
        sum(((hx[(i*4)*q+t] >> (j*4*q)) & mask) << (j*q)
            for j in range(4))
        for i in range(2) for t in range(q)]
    br = [row & ((1 << (4*q)) - 1) for row in hz[:2*q]]
    return ar, br


def _c28_cycles(cols):
    """Four-cycle count on supplied verified component columns."""
    pairs, total = {}, 0
    for value in cols:
        positions = []
        while value:
            bit = value & -value
            positions.append(bit.bit_length() - 1)
            value ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = pairs.get(key, 0)
                total += old
                pairs[key] = old + 1
    return total


def _c28_draw(local, architecture, mode):
    """Intrinsic gap/offset cards; no saved filling is transported."""
    from math import gcd
    q = 28
    units = [a for a in range(1, q) if gcd(a, q) == 1]
    if architecture == 0:
        top = local.sample(units, 4)
        # Both parities occur, making the leading two rows independent.
        bottom = [
            local.choice([a for a in range(1, q) if a % 2 == j % 2])
            for j in range(4)]
    else:
        top, bottom = map(list, _C28_GAPS[architecture])
    unit = local.choice(units)
    top = [unit*a % q for a in top]
    bottom = [unit*a % q for a in bottom]

    # Joint packing is a soft heuristic; actual matrix tests decide admission.
    selected = None
    for attempt in range(12):
        phase = local.randrange(q)
        alpha, beta = local.randrange(q), local.randrange(q)
        if mode == 0:
            offsets = [local.randrange(q) for _ in range(4)]
        elif mode == 1:
            offsets = [(phase + alpha*d + beta*e) % q
                       for d, e in zip(top, bottom)]
        elif mode == 2:
            offsets = [(phase + alpha*d*d + beta*e) % q
                       for d, e in zip(top, bottom)]
        else:
            offsets = local.sample(range(q), 4)
        crosses = [
            value % q
            for d, e, a in zip(top, bottom, offsets)
            for value in (-a, -a-e, d-a, d-a-e)]
        cost = len(crosses) - len(set(crosses))
        key = (cost, local.random())
        if selected is None or key < selected[0]:
            selected = key, offsets
    offsets = selected[1]
    return (
        tuple(((0,), (d,)) for d in top),
        tuple(((a,), ((a+e) % q,))
              for a, e in zip(offsets, bottom)))


def _c28_cofactors(G, grid, cols, inverse):
    """Construct ordinary/divided syzygies, then verify their syndromes.

    Polynomial arithmetic constructs candidate supports only. It never
    substitutes a handwritten regular action for the scaffold lift.
    """
    q = G.q
    entries = [
        [set(G.idx(G.inv(g) if inverse else g) for g in cell)
         for cell in row] for row in grid]

    def divide(cell):
        polynomial = sum(1 << a for a in cell)
        quotient = 0
        while polynomial:
            p = polynomial.bit_length() - 1
            assert p > 0, "even cell polynomial had a remainder"
            quotient ^= 1 << (p-1)
            polynomial ^= 3 << (p-1)
        return {a for a in range(q) if quotient & (1 << a)}

    def product(left, right):
        answer = set()
        for a in left:
            for b in right:
                value = (a+b) % q
                if value in answer:
                    answer.remove(value)
                else:
                    answer.add(value)
        return answer

    divided = [[divide(cell) for cell in row] for row in entries]
    found = set()
    for matrix in (entries, divided):
        for triple in combinations(range(4), 3):
            word = 0
            for omitted in triple:
                s, t = [j for j in triple if j != omitted]
                minor = (product(matrix[0][s], matrix[1][t])
                         ^ product(matrix[0][t], matrix[1][s]))
                for a in minor:
                    word ^= 1 << (omitted*q+a)
            if word and _c28_syndrome(cols, word) == 0:
                found.add(word)
    mask = (1 << q) - 1
    for j in range(4):
        word = mask << (j*q)
        if _c28_syndrome(cols, word) == 0:
            found.add(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w)))


def _c28_orbits(cols, seeds):
    """Bounded translated-witness recombination; not an exclusion test."""
    q, mask = 28, (1 << 28) - 1

    def move(word, shift):
        return sum(
            (((((word >> (j*q)) & mask) << shift)
              | (((word >> (j*q)) & mask) >> (q-shift))) & mask)
            << (j*q) for j in range(4))

    def canonical(word):
        return min(move(word, t) for t in range(q))

    reps = sorted(
        {canonical(w) for w in seeds if w},
        key=lambda w: (w.bit_count(), w))[:12]
    for sweep in range(2):
        discoveries = set(reps)
        short = reps[:8]
        for i, left in enumerate(short):
            for right in short[:i+1]:
                candidates = [
                    left ^ move(right, t) for t in range(q)]
                for word in sorted(
                        {w for w in candidates if w},
                        key=lambda w: (w.bit_count(), w))[:3]:
                    discoveries.add(canonical(word))
        reps = sorted(discoveries,
                      key=lambda w: (w.bit_count(), w))[:12]
    assert all(_c28_syndrome(cols, w) == 0 for w in reps)
    return tuple(reps)


def _c28_words(cols, local, trials, seeds=(), orbit=False):
    """Actual classical witnesses with earlier minima retained."""
    checked = tuple(w for w in seeds
                    if w and _c28_syndrome(cols, w) == 0)
    reading, words = _aff_probe(cols, local, trials, checked)
    if orbit:
        words = tuple(sorted(
            set(words) | set(_c28_orbits(cols, words + checked)),
            key=lambda w: (w.bit_count(), w))[:24])
        reading = min(reading, min(w.bit_count() for w in words))
    assert words and reading <= len(cols)
    assert all(_c28_syndrome(cols, w) == 0 for w in words)
    return reading, words


@lru_cache(maxsize=1)
def _c28_bank(G):
    """Independently optimize and jointly screen twenty structural cards."""
    assert G.q == 28 and A_SHAPE == B_SHAPE == (2, 4)
    q, width = G.q, 4*G.q
    mask = (1 << q) - 1
    counts = [[2]*4 for _ in range(2)]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)
    zero = [[[] for _ in range(4)] for _ in range(2)]

    # Compile only F2 superpositions of verified single-term actions.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(4)] for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = _c28_components(hx, hz, q)
            rows = ar if side == "A" else br
            block = _aff_columns(rows, width)[:q]
            assert all(value.bit_count() == 1 for value in block)
            assert all(value.bit_length() <= q for value in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+t] ^= value << (i*q)
        assert all(value.bit_count() == 4 for value in cols)
        return tuple(cols)

    def component_key(item):
        d = min(item["d"], 24)
        load = sum(w.bit_count() <= d for w in item["words"])
        return (d, -load, -item["cycles"], -item["serial"])

    def joint_key(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("dx", 561), item.get("dz", 561), 24)
        return (128*d*d/560, d,
                -item["a"]["cycles"]-item["b"]["cycles"])

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        lane = slot // 20
        chosen = None

        for batch in range(2):
            local = random.Random(
                BASE_SEED*100003 + 28025129
                + 104729*slot + 7919*batch)

            def pool(side):
                kind = (architecture if side == "A"
                        else (architecture + 2 + lane) % 5)
                raw, labels, serial = [], set(), 0

                def admit(grid):
                    nonlocal serial
                    serial += 1
                    signature = tuple(tuple(sorted(cell))
                                      for row in grid for cell in row)
                    if signature in labels:
                        return
                    labels.add(signature)
                    cols = compile_component(side, grid)
                    # Mixed parity alone is not used as a rank prediction.
                    if gf2_rank(cols) != 2*q-2:
                        return
                    if not _aff_clean4(cols):
                        return
                    seeds = _c28_cofactors(
                        G, grid, cols, side == "B")
                    d, words = _c28_words(cols, local, 3, seeds)
                    item = dict(
                        grid=grid, cols=cols, d=d, words=words,
                        cycles=_c28_cycles(cols), serial=serial)
                    raw.append(item)
                    raw.sort(key=component_key, reverse=True)
                    del raw[6:]

                for attempt in range(96 + 32*batch):
                    admit(_c28_draw(local, kind, mode))

                # Repairs alter actual cells, not a common translation.
                # Changed matrices receive freshly verified witness seeds.
                for step in range(24):
                    if not raw:
                        admit(_c28_draw(local, kind, mode))
                        continue
                    parent = local.choice(raw[:min(3, len(raw))])
                    pressure = [
                        1 + sum(((w >> (j*q)) & mask).bit_count()
                                for w in parent["words"][:8])
                        for j in range(4)]
                    j = local.choices(
                        range(4), weights=pressure, k=1)[0]
                    grid = [[tuple(cell) for cell in row]
                            for row in parent["grid"]]
                    fresh = _c28_draw(local, kind, mode)
                    # Offset-only repairs preserve this slot's gap card.
                    a, b = grid[1][j]
                    delta = local.randrange(1, q)
                    grid[1][j] = (
                        ((a[0]+delta) % q,),
                        ((b[0]+delta) % q,))
                    if step % 7 == 6:
                        h = (j + 1 + local.randrange(3)) % 4
                        grid[0][h] = fresh[0][h]
                        grid[1][h] = fresh[1][h]
                    admit(tuple(tuple(row) for row in grid))

                # Every retained component receives the same fresh budgets.
                for item in raw:
                    d, words = _c28_words(
                        item["cols"], local, 16,
                        item["words"], orbit=True)
                    item["d"] = min(item["d"], d)
                    item["words"] = words
                raw.sort(key=component_key, reverse=True)
                leaders = raw[:3]
                for item in leaders:
                    d, words = _c28_words(
                        item["cols"], local, 48,
                        item["words"], orbit=True)
                    item["d"] = min(item["d"], d)
                    item["words"] = words
                return sorted(
                    leaders, key=component_key, reverse=True)

            ap, bp = pool("A"), pool("B")
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 560 or k != 128:
                        continue
                    ar, br = _c28_components(hx, hz, q)
                    assert _aff_columns(ar, width) == a["cols"]
                    assert _aff_columns(br, width) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz,
                        n=n, k=k, a=a, b=b))
            proposals.sort(key=joint_key, reverse=True)

            finalists = []
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                xc, zc = _aff_columns(hz, n), _aff_columns(hx, n)
                # Conservative exact exclusion of all kernels through four.
                if not (_aff_clean4(xc) and _aff_clean4(zc)):
                    continue
                fingerprint = _aff_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue

                xseeds = tuple(
                    word << (fixed*width)
                    for word in item["b"]["words"][:12]
                    for fixed in range(4))
                zseeds = tuple(
                    sum(((word >> (j*q)) & mask)
                        << ((j*4+fixed)*q) for j in range(4))
                    for word in item["a"]["words"][:12]
                    for fixed in range(4))
                assert all(_c28_syndrome(xc, w) == 0 for w in xseeds)
                assert all(_c28_syndrome(zc, w) == 0 for w in zseeds)

                dx, wx = _aff_probe(xc, local, 6, xseeds, hx)
                dz, wz = _aff_probe(zc, local, 6, zseeds, hz)
                if dx > n or dz > n:
                    continue
                item.update(
                    xc=xc, zc=zc, dx=dx, dz=dz,
                    wx=wx, wz=wz, xseeds=xseeds, zseeds=zseeds,
                    fingerprint=fingerprint)
                finalists.append(item)
                if len(finalists) == 3:
                    break

            # Equal independent full-code rescreening for every finalist.
            for item in finalists:
                dx, _ = _aff_probe(
                    item["xc"], local, 16,
                    item["xseeds"] + item["wx"], item["hx"])
                dz, _ = _aff_probe(
                    item["zc"], local, 16,
                    item["zseeds"] + item["wz"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)

            if not finalists:
                continue
            winner = max(finalists, key=joint_key)
            hx, hz = winner["hx"], winner["hz"]
            assert css_ok(hx, hz)
            assert winner["n"] == 560 and winner["k"] == 128
            assert 225*560 <= 1000*128 < 275*560
            assert max(row.bit_count() for row in hx+hz) == 12
            A, B = winner["A"], winner["B"]
            assert all(len(cell) == len(set(cell)) == 2
                       for grid in (A, B) for row in grid for cell in row)
            assert lifted_k(G, A, B) == 128
            emitted.add(winner["fingerprint"])
            chosen = (A, B, joint_key(winner))
            break

        # Exhausted cards never borrow another card or an archived filling.
        bank.append(chosen)
    return tuple(bank)


# Active S4 x C2 optimizer. Historical affine/cyclic banks are inactive.
_S4_B_COUNTS = (
    ((2,2,1,1,2), (1,1,2,2,2)),
    ((3,0,2,1,2), (0,3,1,2,2)),
    ((3,2,1,1,1), (1,1,2,2,2)),
    ((3,3,0,0,2), (0,0,3,3,2)),
    ((3,3,0,1,1), (1,0,3,2,2)),
)
_S4_A_QUOTAS = (
    (0,1,2,3),
    (0,0,1,3),
    (0,1,1,2),
    (0,0,2,3),
)


def _s4_components(hx, hz, q):
    """Extract the actual unequal-shape components from verified LP rows."""
    mask = (1 << q)-1
    ar = [
        (row & mask) | (((row >> (5*q)) & mask) << q)
        for row in hx[:q]]
    br = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
    return ar, br


def _s4_quotient(g):
    sign = sum(g[i] > g[j]
               for i in range(4) for j in range(i+1, 4)) & 1
    return sign + 2*g[4]


@lru_cache(maxsize=1)
def _s4_atlas(G):
    """Both actions are F2 superpositions of scaffold unit lifts only."""
    za = [[[], []]]
    zb = [[[] for _ in range(5)] for _ in range(2)]
    atlas = {}
    for side in ("A", "B"):
        atoms = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in za]
            B = [[list(c) for c in row] for row in zb]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 12*G.q
            ar, br = _s4_components(hx, hz, G.q)
            rows = ar if side == "A" else br
            width = (2 if side == "A" else 5)*G.q
            atom = _aff_columns(rows, width)[:G.q]
            assert all(v.bit_count() == 1 and v.bit_length() <= G.q
                       for v in atom)
            atoms.append(atom)
        atlas[side] = tuple(atoms)
    return atlas


def _s4_compile(G, atlas, side, grid):
    q = G.q
    cols = [0]*(len(grid[0])*q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                for t, atom in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+t] ^= atom << (i*q)
    return tuple(cols)


def _s4_short6(cols, height, q):
    """Exact component-kernel existence through six after clean4.

    Regular component actions are transitive inside each q-column block.
    Block-zero anchors therefore suffice here, not for full quantum codes.
    Branches partition completions by their first available variable at
    an unsatisfied check. A complete unique singleton/pair table ends tails.
    """
    bits = [1 << j for j in range(len(cols))]
    incidence = [0]*height
    tails = {0: 0}
    degree = max(v.bit_count() for v in cols)
    for j, value in enumerate(cols):
        assert value not in tails
        tails[value] = bits[j]
        pending = value
        while pending:
            bit = pending & -pending
            incidence[bit.bit_length()-1] |= bits[j]
            pending ^= bit
    for j, value in enumerate(cols):
        for h in range(j):
            target = value ^ cols[h]
            assert target not in tails
            tails[target] = bits[j] | bits[h]

    def solve(value, remaining, banned):
        if not value:
            return 0
        if remaining <= 0 or value.bit_count() > degree*remaining:
            return None
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not tail & banned):
                return tail
            return None
        pending, choices, least = value, 0, len(cols)+1
        while pending:
            bit = pending & -pending
            available = incidence[bit.bit_length()-1] & ~banned
            count = available.bit_count()
            if not count:
                return None
            if count < least:
                choices, least = available, count
            pending ^= bit
        order = []
        while choices:
            bit = choices & -choices
            order.append(bit.bit_length()-1)
            choices ^= bit
        order.sort(key=lambda j: (-(cols[j] & value).bit_count(), j))
        excluded = banned
        for j in order:
            excluded |= bits[j]
            tail = solve(value ^ cols[j], remaining-1, excluded)
            if tail is not None:
                return bits[j] | tail
        return None

    for anchor in range(0, len(cols), q):
        tail = solve(cols[anchor], 5, bits[anchor])
        if tail is not None:
            word = bits[anchor] | tail
            assert word and _c28_syndrome(cols, word) == 0
            return word
    return 0


def _s4_probe(cols, local, trials, seeds=(), stabilizers=None):
    """Actual kernel/logical witnesses with separate logical memory.

    All seeds are syndrome checked. Logical candidates are classified
    modulo the exact stabilizer span; adding stabilizers preserves class.
    Earlier found minima remain binding. A sentinel receives no credit.
    """
    n = len(cols)
    span = None if stabilizers is None else _aff_basis(stabilizers)
    generators = () if stabilizers is None else tuple(stabilizers)
    memory = sorted(
        {w for w in seeds if w and _c28_syndrome(cols, w) == 0},
        key=lambda w: (w.bit_count(), w))[:32]
    found, best = set(), n+1

    def logical(word):
        return word and (span is None or _aff_outside(word, span))

    def observe(word):
        nonlocal best
        if word and word.bit_count() <= best+4 and logical(word):
            found.add(word)
            best = min(best, word.bit_count())

    for word in memory:
        observe(word)
    order = list(range(n))
    for trial in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            value, word = cols[c], 1 << c
            while value:
                p = value.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (value, word)
                    break
                value ^= previous[0]
                word ^= previous[1]
            if not value and word:
                dependencies.append(word)
                observe(word)

        all_words = set(memory+dependencies) | found
        donors = sorted(all_words,
                        key=lambda w: (w.bit_count(), w))[:32]
        starts = donors[:20]
        if span is not None:
            starts += sorted(found,
                             key=lambda w: (w.bit_count(), w))[:8]
        discoveries = set(memory)
        for start in starts:
            word = start
            for sweep in range(2):
                before = word
                for direction in donors:
                    candidate = word ^ direction
                    if candidate and candidate.bit_count() < word.bit_count():
                        word = candidate
                if word == before:
                    break
            observe(word)
            discoveries.add(word)

        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)

        if generators:
            for start in sorted(
                    found, key=lambda w: (w.bit_count(), w))[:6]:
                word = start
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in generators),
                        key=lambda entry: entry[0])
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    moves = [row for _, row in nearby[:12]]
                    for i, row in enumerate(moves):
                        for other in moves[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)

        discoveries.update(dependencies)
        found = set(sorted(
            found, key=lambda w: (w.bit_count(), w))[:24])
        ordinary = sorted(
            discoveries, key=lambda w: (w.bit_count(), w))[:24]
        memory = sorted(
            set(ordinary) | found,
            key=lambda w: (w.bit_count(), w))
        if best <= 4:
            break
    assert all(w and _c28_syndrome(cols, w) == 0 for w in memory)
    return best, tuple(memory)


def _s4_draw(local, G, pools, counts, side, mode, lane):
    """Intrinsic quotient occupancies plus noncommuting relative labels."""
    grid = []
    used_rows = [set() for _ in counts]
    for i, row in enumerate(counts):
        output = []
        for j, count in enumerate(row):
            if not count:
                output.append(())
                continue
            winner = None
            for attempt in range(32):
                if side == "A":
                    quota = _S4_A_QUOTAS[mode]
                    cell = []
                    for quotient in quota:
                        available = [g for g in pools[quotient]
                                     if g not in cell]
                        cell.append(local.choice(available))
                else:
                    if mode == 0:
                        cell = local.sample(G.elements, count)
                    else:
                        phase = (i+2*j+lane) % 4
                        steps = ((0,1,2), (0,1,3), (0,2,3))[mode-1]
                        cell = []
                        for t in range(count):
                            quotient = (phase ^ steps[t % 3])
                            available = [g for g in pools[quotient]
                                         if g not in cell]
                            cell.append(local.choice(available))
                if count == 2:
                    relative = G.mul(G.inv(cell[0]), cell[1])
                    if G.mul(relative, relative) == G.e:
                        continue
                differences = [
                    (G.mul(x, G.inv(y)) if side == "A"
                     else G.mul(G.inv(x), y))
                    for x in cell for y in cell if x != y]
                distinct = set(differences)
                cost = len(differences)-len(distinct)
                cost += 2*len(distinct & used_rows[i])
                commuting = sum(
                    G.mul(x, y) == G.mul(y, x)
                    for x, y in combinations(cell, 2))
                key = (cost, commuting, local.random())
                if winner is None or key < winner[0]:
                    winner = (key, tuple(cell), distinct)
                if cost == 0 and commuting == 0:
                    break
            if winner is None:
                return None
            output.append(winner[1])
            used_rows[i].update(winner[2])
        grid.append(tuple(output))
    return tuple(grid)


def _s4_pool(local, G, atlas, pools, side, counts, mode, lane, batch):
    """Independent constituent selection and witness-heavy cell repair."""
    height = len(counts)*G.q
    raw, labels = [], set()
    positions = [(i,j) for i,row in enumerate(counts)
                 for j,count in enumerate(row) if count]

    def key(item):
        d = min(item["d"], 22)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"], -item["serial"])

    def admit(grid):
        if grid is None:
            return
        label = tuple(tuple(sorted(c)) for row in grid for c in row)
        if label in labels:
            return
        labels.add(label)
        cols = _s4_compile(G, atlas, side, grid)
        if not _aff_clean4(cols):
            return
        d, words = _s4_probe(cols, local, 2)
        if not words or d <= 6 or d > len(cols):
            return
        raw.append(dict(
            grid=grid, cols=cols, d=d, words=words,
            rank=gf2_rank(cols), cycles=_c28_cycles(cols),
            serial=len(labels)))
        raw.sort(key=key, reverse=True)
        del raw[8:]

    quota = (96 if side == "A" else 160) + 48*batch + 8*lane
    for attempt in range(quota):
        admit(_s4_draw(local, G, pools, counts, side, mode, lane))

    mask = (1 << G.q)-1
    for step in range(24):
        fresh = _s4_draw(local, G, pools, counts, side, mode, lane)
        if fresh is None:
            continue
        if not raw:
            admit(fresh)
            continue
        parent = local.choice(raw[:4])
        pressure = [
            1+sum(((w >> (j*G.q)) & mask).bit_count()
                  for w in parent["words"][:8])
            for i,j in positions]
        i,j = local.choices(positions, weights=pressure, k=1)[0]
        grid = [[tuple(c) for c in row] for row in parent["grid"]]
        grid[i][j] = fresh[i][j]
        if step % 6 == 5:
            h,k = local.choice(positions)
            grid[h][k] = fresh[h][k]
        admit(tuple(tuple(row) for row in grid))

    survivors = []
    for item in raw:
        if _s4_short6(item["cols"], height, G.q):
            continue
        d, words = _s4_probe(
            item["cols"], local, 12, item["words"])
        item["d"] = min(item["d"], d)
        item["words"] = words
        survivors.append(item)
    survivors.sort(key=key, reverse=True)
    leaders = survivors[:3]
    for item in leaders:
        d, words = _s4_probe(
            item["cols"], local, 32, item["words"])
        item["d"] = min(item["d"], d)
        item["words"] = words
    return sorted(leaders, key=key, reverse=True)


@lru_cache(maxsize=1)
def _s4_bank(G):
    """Fresh structural cards; no archived or emitted-code fallback."""
    assert G.q == 48 and A_SHAPE == (1,2) and B_SHAPE == (2,5)
    q, mask = G.q, (1 << G.q)-1
    atlas = _s4_atlas(G)
    pools = tuple(
        tuple(g for g in G.elements if _s4_quotient(g) == t)
        for t in range(4))
    assert all(len(pool) == 12 for pool in pools)
    sa = ((4,4),)
    bank, emitted = [], set()

    def merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("dx", 577), item.get("dz", 577), 22)
        load = sum(
            w.bit_count() <= d+1
            for component in (item["a"], item["b"])
            for w in component["words"])
        return (item["k"]*d*d/item["n"], d, -load)

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        lane = slot//20
        sb = _S4_B_COUNTS[architecture]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        local = random.Random(
            BASE_SEED*100003 + 480251733 + 104729*slot)
        winner = None

        for batch in range(3):
            ap = _s4_pool(
                local, G, atlas, pools, "A", sa, mode, lane, batch)
            bp = _s4_pool(
                local, G, atlas, pools, "B", sb, mode, lane, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (225*n <= 1000*k < 275*n):
                        continue
                    ar, br = _s4_components(hx, hz, q)
                    assert _aff_columns(ar, 2*q) == a["cols"]
                    assert _aff_columns(br, 5*q) == b["cols"]
                    proposals.append(dict(
                        A=a["grid"], B=b["grid"], a=a, b=b,
                        hx=hx, hz=hz, n=n, k=k))
            proposals.sort(key=merit, reverse=True)

            finalists, local_signatures = [], set()
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                xc, zc = _aff_columns(hz, n), _aff_columns(hx, n)
                # Conservative full-kernel exclusion, including both sectors.
                if not (_aff_clean4(xc) and _aff_clean4(zc)):
                    continue
                signature = _aff_fingerprint(hx, hz, n)
                if signature in emitted or signature in local_signatures:
                    continue
                local_signatures.add(signature)
                xseeds = tuple(
                    word << (j*5*q)
                    for word in item["b"]["words"][:16]
                    for j in range(2))
                zseeds = tuple(
                    ((word & mask) << (beta*q))
                    | ((word >> q) << ((5+beta)*q))
                    for word in item["a"]["words"][:16]
                    for beta in range(5))
                assert all(_c28_syndrome(xc, w) == 0 for w in xseeds)
                assert all(_c28_syndrome(zc, w) == 0 for w in zseeds)
                dx, wx = _s4_probe(xc, local, 4, xseeds, hx)
                dz, wz = _s4_probe(zc, local, 4, zseeds, hz)
                if dx > n or dz > n:
                    continue
                item.update(
                    xc=xc, zc=zc, dx=dx, dz=dz, wx=wx, wz=wz,
                    xseeds=xseeds, zseeds=zseeds, signature=signature)
                finalists.append(item)
                if len(finalists) == 3:
                    break

            # Equal independent budgets; previous lighter witnesses bind.
            for item in finalists:
                dx, _ = _s4_probe(
                    item["xc"], local, 12,
                    item["xseeds"]+item["wx"], item["hx"])
                dz, _ = _s4_probe(
                    item["zc"], local, 12,
                    item["zseeds"]+item["wz"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 14:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 576 and winner["k"] >= 144
        assert css_ok(hx, hz)
        assert lifted_k(G, A, B) == winner["k"]
        assert 225*n <= 1000*winner["k"] < 275*n
        actual_a = [[len(c) for c in row] for row in A]
        actual_b = [[len(c) for c in row] for row in B]
        assert _grid_weights(actual_a, actual_b) == 12
        assert _no_dead_lines(actual_a, actual_b)
        assert max(row.bit_count() for row in hx+hz) <= W_MAX
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (A,B) for row in grid for c in row)
        emitted.add(winner["signature"])
        bank.append(dict(A=A, B=B, k=winner["k"],
                         selection=merit(winner),
                         policy=(architecture, mode, lane)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the independently screened S4 x C2 structural policy."""
    bank = _s4_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("no fresh S4 x C2 target-rate design passed this card")
    return bank[idx]


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell; all new-shape indices are defined."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Affine cubic packing, exact component exclusion through six and the existing witness budgets supply five measured readings >=9: the current parent instead measured 10,10,10,8,8 at k=224, with sector-one minima.
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
