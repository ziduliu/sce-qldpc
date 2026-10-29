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
CONCEPT = ("Heisenberg H3=UT(3,3), order 27, shapes (2,4)x(2,4): noncommuting doubleton relatives in every constituent column should remove the cyclic divided-cofactor mechanism, supply five fresh measured readings >=16, and exceed cell-0.250 fitness 44.8. This is a measurement hypothesis, not a distance certificate.")
GROUP = ("L3 jump from cyclic C28 to the nonabelian exponent-three Heisenberg group. Noncommuting pairs generate all 27 elements, unlike the commuting relatives behind the parent's polynomial cofactors. No order-27 (2,4)x(2,4) bed appears in BEDS. Central and order-nine subgroup fibers supply candidate witnesses, not an assumed group advantage.")
SHAPES = ("Keep equal (2,4) components for cell 0.250: n=540, kb=108, guaranteed rate 0.200, and 108 variables/54 checks on each side. The size window allows q<=29; this group fixes q=27. Require verified constituent rank 52 and assembled 0.225<=k/n<0.275; expected k=124 would give rate 0.22963 but is not assumed. Parent weight-fourteen minima were sector-one-only, especially Acol1/Acol2 and Bcol1/Bcol2. Complete probes inspect both sectors.")
FILLING = ("All sixteen cells retain two distinct terms, so both CSS row weights are 8+4=12. Twenty cards cross five projective relative-direction arrangements with four offset geometries. Every column has noncommuting top/bottom relatives. Compile only scaffold unit actions; search actual central/order-nine fiber kernels, verified translations and ordinary dependencies. Repair individual bottom offsets and independently deepen every survivor. Exact ranks, kernel-four exclusion, syndrome-checked embeddings, stabilizer-tested full witnesses and unequal uncoloured Tanner fingerprints gate emission. Cyclic polynomial helpers are inactive; bounded readings remain upper witnesses.")
FAMILY_NAME = "r025_HeisenbergH3_24x24_noncommuting_normfiber_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 27

def custom_ops(q):
    # UT(3,3): (a,b,c) represents [[1,a,c],[0,1,b],[0,0,1]].
    if not valid_q(q):
        raise ValueError("Heisenberg H3 requires order 27")
    els = [(a,b,c) for c in range(3)
           for b in range(3) for a in range(3)]

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c = red(g)
        d,e,f = red(h)
        return ((a+d) % 3, (b+e) % 3, (c+f+a*e) % 3)

    def inv(g):
        a,b,c = red(g)
        return ((-a) % 3, (-b) % 3, (-c+a*b) % 3)

    def idx(g):
        a,b,c = red(g)
        return 9*c+3*b+a

    return dict(elements=els, ident=(0,0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)

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


# Active Heisenberg policies. Historical affine/C28 banks are inactive.
# Every paired direction is different, hence its relatives do not commute.
_H27_DIRECTIONS = ((1,0), (0,1), (1,1), (1,2))
_H27_CARDS = (
    ((0,0,0,1), (1,1,2,2)),
    ((0,0,1,1), (1,2,2,3)),
    ((0,0,1,2), (1,2,3,0)),
    ((0,1,2,3), (1,2,3,0)),
    ((0,0,1,2), (1,1,0,0)),
)


def _h27_draw(local, G, architecture, mode, side):
    """Fresh projective cards; packing cost is only a selection heuristic."""
    top, bottom = _H27_CARDS[architecture]
    best = None
    for attempt in range(12):
        upper, lower = [], []
        used = [set(), set()]
        crossing = set()
        cost = 0
        phases = list((0,0,1,2))
        local.shuffle(phases)
        for j, (u, v) in enumerate(zip(top, bottom)):
            relatives = []
            for direction in (u, v):
                a, b = _H27_DIRECTIONS[direction]
                scale = local.choice((1,2))
                relatives.append((scale*a % 3, scale*b % 3,
                                  local.randrange(3)))
            g, h = relatives
            assert G.mul(g,h) != G.mul(h,g)
            if mode == 0:
                offset = G.pick(local)
            elif mode == 1:
                offset = (0,0,local.randrange(3))
            elif mode == 2:
                offset = G.mul(g,h)
            else:
                offset = (local.randrange(3), local.randrange(3), phases[j])
            cells = ((G.e,g), (offset,G.mul(offset,h)))
            for i, cell in enumerate(cells):
                ds = {
                    (G.mul(x,G.inv(y)) if side == "A"
                     else G.mul(G.inv(x),y))
                    for x in cell for y in cell if x != y}
                cost += len(ds & used[i])
                used[i].update(ds)
            cs = {
                (G.mul(x,G.inv(y)) if side == "A"
                 else G.mul(G.inv(x),y))
                for x in cells[0] for y in cells[1]}
            cost += 4-len(cs)+len(cs & crossing)
            crossing.update(cs)
            upper.append(cells[0])
            lower.append(cells[1])
        grid = (tuple(upper),tuple(lower))
        key = (cost,local.random())
        if best is None or key < best[0]:
            best = (key,grid)
    return best[1]


def _h27_partitions(G):
    """Group operations construct restricted supports, never parity lifts."""
    central = [(0,0,c) for c in range(3)]
    subgroups = [central]
    for a,b in _H27_DIRECTIONS:
        g = (a,b,0)
        powers = (G.e,g,G.mul(g,g))
        subgroups.append(sorted({
            G.mul(x,z) for x in powers for z in central}))
    partitions = []
    for subgroup in subgroups:
        unseen, fibers = set(range(G.q)), []
        while unseen:
            g = G.elements[min(unseen)]
            coset = {G.idx(G.mul(g,h)) for h in subgroup}
            unseen.difference_update(coset)
            fibers.append(sum(1 << t for t in coset))
        partitions.append(tuple(sorted(fibers)))
    return tuple(partitions)


def _h27_norm_words(cols, partitions, local, q):
    """Actual witnesses in subgroup-fiber subspaces; not an exclusion test."""
    found = set()
    for j in range(4):
        word = ((1 << q)-1) << (j*q)
        assert _c28_syndrome(cols,word) == 0
        found.add(word)
    for fibers in partitions:
        atoms = [fiber << (j*q) for j in range(4) for fiber in fibers]
        restricted = [_c28_syndrome(cols,w) for w in atoms]
        _, words = _aff_probe(restricted,local,2)
        for compact in words:
            word = 0
            while compact:
                bit = compact & -compact
                word ^= atoms[bit.bit_length()-1]
                compact ^= bit
            if word:
                assert _c28_syndrome(cols,word) == 0
                found.add(word)
    return tuple(sorted(found,key=lambda w:(w.bit_count(),w))[:24])


def _h27_translate(cols, words, maps, q):
    """Translate witnessed supports only; every image is syndrome-checked."""
    short = sorted(set(words),key=lambda w:(w.bit_count(),w))[:8]
    images = set()
    for word in short:
        positions, remaining = [], word
        while remaining:
            bit = remaining & -remaining
            positions.append(divmod(bit.bit_length()-1,q))
            remaining ^= bit
        for permutation in maps:
            moved = sum(1 << (j*q+permutation[h]) for j,h in positions)
            if moved and _c28_syndrome(cols,moved) == 0:
                images.add(moved)
    discoveries = set(words)
    for word in short:
        for other in images:
            candidate = word ^ other
            if candidate:
                discoveries.add(candidate)
    return tuple(sorted(discoveries,
                        key=lambda w:(w.bit_count(),w))[:24])


def _h27_probe(cols, local, trials, seeds=(), stabilizers=None):
    """True kernel/logical witnesses; keep logical memory separately."""
    width = len(cols)
    span = None if stabilizers is None else _aff_basis(stabilizers)
    memory = sorted({
        w for w in seeds if w and _c28_syndrome(cols,w) == 0},
        key=lambda w:(w.bit_count(),w))[:48]
    logicals = set()
    best = width+1

    def observe(word):
        nonlocal best
        if word and (span is None or _aff_outside(word,span)):
            best = min(best,word.bit_count())
            logicals.add(word)

    for word in memory:
        observe(word)
    order = list(range(width))
    for trial in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome,word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                observe(word)
        donors = sorted(set(memory+dependencies),
                        key=lambda w:(w.bit_count(),w))[:40]
        starts = donors[:20]+sorted(
            logicals,key=lambda w:(w.bit_count(),w))[:8]
        discoveries = set(memory+dependencies)
        for start in starts:
            word = start
            for sweep in range(2):
                before = word
                for donor in donors:
                    candidate = word ^ donor
                    if candidate and candidate.bit_count() < word.bit_count():
                        word = candidate
                if word == before:
                    break
            observe(word)
            discoveries.add(word)
        short = sorted(discoveries,key=lambda w:(w.bit_count(),w))[:24]
        for i,word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)
        if stabilizers is not None:
            for start in sorted(logicals,
                                key=lambda w:(w.bit_count(),w))[:6]:
                word = start
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(),row)
                         for row in stabilizers),
                        key=lambda x:x[0])[:12]
                    for cw,row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    for i,(_,row) in enumerate(nearby):
                        for _,other in nearby[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)
        logicals = set(sorted(logicals,
                              key=lambda w:(w.bit_count(),w))[:16])
        memory = sorted(discoveries,
                        key=lambda w:(w.bit_count(),w))[:24]
        memory = sorted(set(memory) | logicals,
                        key=lambda w:(w.bit_count(),w))
    assert all(_c28_syndrome(cols,w) == 0 for w in memory)
    return best,tuple(memory)


@lru_cache(maxsize=1)
def _h27_bank(G):
    """Independent structural tasks with exact assembled-rate screening."""
    assert G.q == 27 and A_SHAPE == B_SHAPE == (2,4)
    q, width = G.q, 4*G.q
    mask = (1 << q)-1
    counts = [[2]*4 for _ in range(2)]
    assert _grid_weights(counts,counts) == 12
    assert _no_dead_lines(counts,counts)
    zero = [[[] for _ in range(4)] for _ in range(2)]
    atlas = {}
    for side in ("A","B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(4)] for _ in range(2)]
            unit[0][0] = [g]
            A,B = (unit,zero) if side == "A" else (zero,unit)
            hx,hz,_ = lifted_matrices(G,A,B)
            ar,br = _c28_components(hx,hz,q)
            block = _aff_columns(ar if side == "A" else br,width)[:q]
            assert all(v.bit_count() == 1 and v.bit_length() <= q
                       for v in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)
    partitions = _h27_partitions(G)
    translations = {
        side: tuple(
            tuple(G.idx(G.mul(g,shift) if side == "A"
                        else G.mul(shift,g)) for g in G.elements)
            for shift in G.elements)
        for side in ("A","B")}

    def compile_component(side,grid):
        cols = [0]*width
        for i,row in enumerate(grid):
            for j,cell in enumerate(row):
                for g in cell:
                    for t,value in enumerate(atlas[side][G.idx(g)]):
                        cols[j*q+t] ^= value << (i*q)
        assert all(v.bit_count() == 4 for v in cols)
        return tuple(cols)

    def component_key(item):
        return (item["d"],
                -sum(w.bit_count() <= item["d"] for w in item["words"]),
                -item["cycles"],-item["serial"])

    def joint_key(item):
        d = min(item["a"]["d"],item["b"]["d"],
                item["dx"],item["dz"],27)
        return (item["k"]*d*d/item["n"],d,
                -item["a"]["cycles"]-item["b"]["cycles"])

    bank, emitted = [],set()
    for slot in range(N_SLOTS):
        architecture,mode = divmod(slot % 20,4)
        lane = slot//20
        winner = None
        for batch in range(2):
            local = random.Random(
                BASE_SEED*100003+27025171+104729*slot+7919*batch)

            def pool(side):
                kind = (architecture if side == "A"
                        else (architecture+2+lane) % 5)
                raw,seen,serial = [],set(),0

                def admit(grid):
                    nonlocal serial
                    serial += 1
                    if grid in seen:
                        return
                    seen.add(grid)
                    cols = compile_component(side,grid)
                    if gf2_rank(cols) != 2*q-2 or not _aff_clean4(cols):
                        return
                    seeds = _h27_norm_words(cols,partitions,local,q)
                    d,words = _h27_probe(cols,local,3,seeds)
                    raw.append(dict(
                        grid=grid,cols=cols,d=d,words=words,
                        cycles=_c28_cycles(cols),serial=serial))
                    raw.sort(key=component_key,reverse=True)
                    del raw[6:]

                for attempt in range(112+48*batch):
                    admit(_h27_draw(local,G,kind,mode,side))
                for step in range(24):
                    if not raw:
                        admit(_h27_draw(local,G,kind,mode,side))
                        continue
                    parent = local.choice(raw[:3])
                    pressure = [
                        1+sum(((w >> (j*q)) & mask).bit_count()
                              for w in parent["words"][:8])
                        for j in range(4)]
                    j = local.choices(range(4),weights=pressure,k=1)[0]
                    grid = [list(row) for row in parent["grid"]]
                    delta = G.pick(local)
                    grid[1][j] = tuple(G.mul(delta,g)
                                       for g in grid[1][j])
                    if step % 7 == 6:
                        fresh = _h27_draw(local,G,kind,mode,side)
                        h = (j+1+local.randrange(3)) % 4
                        grid[0][h],grid[1][h] = fresh[0][h],fresh[1][h]
                    admit(tuple(tuple(row) for row in grid))
                for item in raw:
                    seeds = _h27_translate(
                        item["cols"],item["words"],translations[side],q)
                    d,words = _h27_probe(item["cols"],local,24,seeds)
                    item["d"] = min(item["d"],d)
                    item["words"] = words
                raw.sort(key=component_key,reverse=True)
                leaders = raw[:3]
                for item in leaders:
                    seeds = _h27_translate(
                        item["cols"],item["words"],translations[side],q)
                    d,words = _h27_probe(item["cols"],local,48,seeds)
                    item["d"] = min(item["d"],d)
                    item["words"] = words
                return sorted(leaders,key=component_key,reverse=True)

            ap,bp = pool("A"),pool("B")
            proposals = []
            for a in ap:
                for b in bp:
                    hx,hz,n = lifted_matrices(G,a["grid"],b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (n == 540 and 225*n <= 1000*k < 275*n):
                        continue
                    ar,br = _c28_components(hx,hz,q)
                    assert _aff_columns(ar,width) == a["cols"]
                    assert _aff_columns(br,width) == b["cols"]
                    proposals.append(dict(
                        A=a["grid"],B=b["grid"],a=a,b=b,
                        hx=hx,hz=hz,n=n,k=k))
            proposals.sort(
                key=lambda x:(min(x["a"]["d"],x["b"]["d"]),
                              x["a"]["d"]+x["b"]["d"]),
                reverse=True)
            finalists = []
            for item in proposals:
                hx,hz,n = item["hx"],item["hz"],item["n"]
                xc,zc = _aff_columns(hz,n),_aff_columns(hx,n)
                if not (_aff_clean4(xc) and _aff_clean4(zc)):
                    continue
                fingerprint = _aff_fingerprint(hx,hz,n)
                if fingerprint in emitted:
                    continue
                xs = tuple(
                    w << (fixed*width)
                    for w in item["b"]["words"][:12]
                    for fixed in range(4))
                zs = tuple(
                    sum(((w >> (j*q)) & mask)
                        << ((j*4+fixed)*q) for j in range(4))
                    for w in item["a"]["words"][:12]
                    for fixed in range(4))
                assert all(_c28_syndrome(xc,w) == 0 for w in xs)
                assert all(_c28_syndrome(zc,w) == 0 for w in zs)
                dx,wx = _h27_probe(xc,local,6,xs,hx)
                dz,wz = _h27_probe(zc,local,6,zs,hz)
                if dx > n or dz > n:
                    continue
                item.update(
                    xc=xc,zc=zc,dx=dx,dz=dz,wx=wx,wz=wz,
                    xs=xs,zs=zs,fingerprint=fingerprint)
                finalists.append(item)
                if len(finalists) == 2:
                    break
            for item in finalists:
                dx,_ = _h27_probe(
                    item["xc"],local,18,item["xs"]+item["wx"],item["hx"])
                dz,_ = _h27_probe(
                    item["zc"],local,18,item["zs"]+item["wz"],item["hz"])
                item["dx"],item["dz"] = min(item["dx"],dx),min(item["dz"],dz)
                if winner is None or joint_key(item) > joint_key(winner):
                    winner = item
            if winner is not None and joint_key(winner)[1] >= 16:
                break
        if winner is None:
            bank.append(None)
            continue
        assert css_ok(winner["hx"],winner["hz"])
        assert max(r.bit_count()
                   for r in winner["hx"]+winner["hz"]) == 12
        assert 225*540 <= 1000*winner["k"] < 275*540
        assert all(len(c) == len(set(c)) == 2
                   for grid in (winner["A"],winner["B"])
                   for row in grid for c in row)
        emitted.add(winner["fingerprint"])
        bank.append(dict(
            A=winner["A"],B=winner["B"],k=winner["k"],
            selection=joint_key(winner),policy=(architecture,mode,lane)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh noncommuting Heisenberg rectangular task."""
    bank = _h27_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("no fresh Heisenberg filling in cell 0.250")
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
