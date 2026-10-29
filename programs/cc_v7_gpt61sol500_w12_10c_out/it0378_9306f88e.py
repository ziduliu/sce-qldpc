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
CONCEPT = ("AGL(1,8), order 56, shapes (1,3)x(1,3): exact constituent dependency exclusion through eight, followed by witness-directed voltage repairs and verified logical dressing, should supply five measured readings >=10 and improve cell-0.400 fitness above 34.24.")
GROUP = ("Frozen byte-for-byte under L1. This exact affine order-56 bed measured three distance-ten codes and fitness 34.24, exceeding the displayed D29 score 25.6 and D9 x C3 score 27.022 at equal shapes. These readings support further filling optimization, not a certified group advantage.")
SHAPES = ("Frozen equal (1,3) sides: n=560, kb=224, guaranteed rate 0.400, maximum admissible lift order 59, and equal 168-variable/56-check constituents at the fixed q=56. Parent minima were entirely sector-one Acol/Bcol witnesses, prominently Acol1 and Bcol1; no measured minimum used Arow/Brow blocks. Verify assembled ranks inside 0.375<=k/n<0.425.")
FILLING = ("All six cells remain cubic, retaining both check weights twelve. Three quotient cards follow the geometries of the parent's distance-ten leaders; two new common-quotient-factor cards test rank surplus without assuming it. Cross these with four coordinate policies, exhaust constituent dependencies through eight before admission, repair witness-heavy cells, and independently rescreen survivors. Syndrome-check constituent embeddings before stabilizer-aware full-code searches and single/pair dressing. Reject all 59 archived parent Tanner signatures and repeated new signatures. Exact constituent exclusions do not certify quantum distance; found logical readings remain upper bounds.")
FAMILY_NAME = "r040_AGL1F8_13x13_cubic_exact8_witnessrepair_archive_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 56

def custom_ops(q):
    # (a,b) acts on F8 as z -> alpha^b*z + a.
    # F8 = F2[t]/(t^3+t+1), alpha=t has order seven.
    if not valid_q(q):
        raise ValueError("AGL(1,8) requires order 56")

    def field_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return result

    action = [1]
    for _ in range(6):
        action.append(field_mul(action[-1], 2))
    els = [(a, b) for b in range(7) for a in range(8)]

    def red(g):
        return (g[0] & 7, g[1] % 7)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return (a ^ field_mul(action[b], c), (b + d) % 7)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 7
        return (field_mul(action[bi], a), bi)

    def idx(g):
        a, b = red(g)
        return 8*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)

from itertools import combinations, combinations_with_replacement
from functools import lru_cache


# Five component-pair architectures, crossed with four coordinate models.
# Additional harness lanes change B's relative quotient patterns.
_AFF_PATTERNS = (
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (2, 6), (3, 5),
)
_AFF_CARDS = (
    # Quotient geometries of measured leaders 30, 38 and 51.
    # Their labels are NOT reused; archived signatures are excluded.
    (0, 2, 4, 1, 4, 0),
    (0, 1, 4, 1, 3, 0),
    (0, 2, 4, 2, 5, 1),
    # Common non-AP quotient factors on one side only.
    # Actual assembled ranks decide whether any surplus is useful.
    (1, 1, 1, 0, 2, 4),
    (0, 1, 4, 1, 1, 1),
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
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda item: item[0])[:12]
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    # Two stabilizers can shorten a representative even
                    # when neither individual addition decreases weight.
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    consider(word)
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


def _aff_clean8(cols, q):
    """Exact constituent-kernel exclusion through eight after clean4.

    At an odd check, classify completions by their first selected
    available variable and ban earlier alternatives. Unique supports
    of size <=2 finish branches. There is no node/time cutoff.

    The constituent regular action is transitive within each q-column
    block, so three anchors cover all constituent support orbits.
    No such anchor reduction is assumed for the complete quantum code.
    """
    bits = [1 << j for j in range(len(cols))]
    rows = [0] * q
    tails = {0: 0}
    for j, syndrome in enumerate(cols):
        tails[syndrome] = bits[j]
        value = syndrome
        while value:
            bit = value & -value
            rows[bit.bit_length() - 1] |= bits[j]
            value ^= bit
    for j, syndrome in enumerate(cols):
        for h in range(j):
            value = syndrome ^ cols[h]
            assert value not in tails
            tails[value] = bits[j] | bits[h]

    def exists(value, remaining, banned):
        if not value:
            return True
        # Cubic columns impose parity on every possible completion.
        if (value.bit_count() ^ remaining) & 1:
            remaining -= 1
        if remaining <= 0 or value.bit_count() > 3 * remaining:
            return False
        if remaining <= 2:
            tail = tails.get(value)
            return (tail is not None
                    and tail.bit_count() <= remaining
                    and not (tail & banned))

        pending = value
        choices = 0
        least = len(cols) + 1
        while pending:
            check = pending & -pending
            options = rows[check.bit_length() - 1] & ~banned
            count = options.bit_count()
            if not count:
                return False
            if count < least:
                choices, least = options, count
                if count == 1:
                    break
            pending ^= check

        alternatives = []
        while choices:
            bit = choices & -choices
            choices ^= bit
            alternatives.append(bit.bit_length() - 1)
        alternatives.sort(
            key=lambda j: (-(cols[j] & value).bit_count(), j))
        excluded = banned
        for j in alternatives:
            excluded |= bits[j]
            if exists(value ^ cols[j], remaining - 1, excluded):
                return True
        return False

    return not any(
        exists(cols[anchor], 7, bits[anchor])
        for anchor in range(0, len(cols), q))


def _aff_component_pool(local, G, atlas, side, kinds, model, lane, batch):
    """Admit only eight-free constituents; then repair actual witnesses."""
    pool, seen = [], set()
    q, mask = G.q, (1 << G.q) - 1
    serial = 0

    def merit(item):
        d = min(item["reading"], 12)
        load = sum(w.bit_count() <= d for w in item["words"])
        return (d, -load, item["reading"], -item["rank"],
                -item["attempt"])

    def packed(grid):
        used = set()
        for cell in grid[0]:
            if len(cell) != 3 or len(set(cell)) != 3:
                return False
            differences = {
                (G.mul(x, G.inv(y)) if side == "A"
                 else G.mul(G.inv(x), y))
                for x in cell for y in cell if x != y
            }
            if len(differences) != 6 or differences & used:
                return False
            used.update(differences)
        return True

    def admit(grid):
        nonlocal serial
        serial += 1
        if grid is None or grid in seen:
            return
        seen.add(grid)
        if not packed(grid):
            return
        cols = _aff_component_columns(G, atlas, side, grid)
        assert all(value.bit_count() == 3 for value in cols)
        if not _aff_clean4(cols):
            return

        # A found short word cheaply rejects a candidate. A high
        # randomized reading never replaces the exhaustive eight test.
        reading, words = _aff_probe(cols, local, 2)
        if reading <= 8 or not _aff_clean8(cols, q):
            return
        item = dict(
            grid=grid, cols=cols, reading=reading, words=words,
            rank=gf2_rank(cols), attempt=serial)
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[8:]

    for attempt in range(192 + 64 * batch):
        admit(_aff_draw(local, G, kinds, model, side, lane))

    # Replace actual voltages in a witness-heavy column. Old words
    # are NOT reused as witnesses for a changed matrix.
    for step in range(32 + 16 * batch):
        replacement = _aff_draw(local, G, kinds, model, side, lane)
        if replacement is None:
            continue
        if not pool:
            admit(replacement)
            continue
        parent = local.choice(pool[:min(4, len(pool))])
        pressure = [
            1 + sum(((word >> (j * q)) & mask).bit_count()
                    for word in parent["words"][:8])
            for j in range(3)]
        j = local.choices(range(3), weights=pressure, k=1)[0]
        row = list(parent["grid"][0])
        row[j] = replacement[0][j]
        if step % 7 == 0:
            h = (j + 1 + local.randrange(2)) % 3
            row[h] = replacement[0][h]
        admit((tuple(row),))

    # Every survivor receives the same independent witness budget.
    for item in pool:
        reading, words = _aff_probe(
            item["cols"], local, 24, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        assert item["reading"] >= 10
    pool.sort(key=merit, reverse=True)
    leaders = pool[:4]
    for item in leaders:
        reading, words = _aff_probe(
            item["cols"], local, 32, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        assert item["reading"] >= 10
    leaders.sort(key=merit, reverse=True)
    return leaders[:3]


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

    # Negative archive only: reconstruct the 59 parent outputs from
    # their recorded normal coordinates and original quotient cards.
    # These cells are never used as initial candidates or emitted.
    old_cards = (
        (0, 0, 0, 0, 4, 6),
        (0, 1, 2, 3, 4, 5),
        (0, 2, 4, 0, 3, 7),
        (0, 4, 6, 1, 3, 5),
        (0, 1, 4, 0, 2, 7),
    )
    old_normals = (
        (5,5,3,6,2,3,2,7,0,6,5,3),
        (6,5,5,2,1,6,2,4,7,4,6,2),
        (6,7,2,0,1,5,5,7,4,0,0,1),
        (6,3,2,5,0,7,2,5,7,6,7,7),
        (4,5,5,1,0,2,7,3,2,4,1,2),
        (4,6,6,1,3,1,6,1,6,1,2,3),
        (4,3,1,6,2,1,1,3,3,5,5,7),
        (0,1,4,3,6,7,0,3,0,5,3,7),
        (4,3,2,1,5,6,7,5,3,2,6,4),
        (2,1,3,5,6,7,3,4,4,3,5,5),
        (4,5,5,7,4,0,5,7,2,6,4,5),
        (0,1,4,5,3,4,4,1,1,5,2,3),
        (4,1,5,0,7,0,4,0,1,0,3,6),
        (5,6,5,6,2,5,6,2,7,1,7,5),
        (0,1,7,5,3,7,2,3,4,1,2,3),
        (6,5,0,7,3,7,0,1,2,3,5,5),
        (3,3,7,6,6,5,2,7,1,6,2,1),
        (5,2,4,3,4,1,6,1,4,7,5,5),
        (6,7,3,1,5,2,6,4,5,1,6,7),
        (2,5,0,1,7,7,0,1,4,1,3,5),
        (3,7,4,6,7,3,0,3,1,3,4,4),
        (3,6,5,3,7,2,2,3,4,5,6,4),
        (4,6,7,1,0,2,1,5,2,7,3,0),
        (4,5,0,1,2,7,4,3,0,7,3,5),
        (7,2,1,3,1,7,2,4,3,7,0,2),
        (6,5,1,3,7,3,7,4,6,1,3,4),
        (0,2,1,5,5,4,5,1,2,3,0,2),
        (4,1,6,7,0,1,4,7,0,7,6,5),
        (1,4,5,0,0,6,7,5,7,0,5,5),
        (2,1,3,5,6,7,6,2,6,0,4,1),
        (6,4,1,5,6,7,3,7,4,5,4,6),
        (6,5,4,3,6,7,6,1,0,1,2,3),
        (4,1,5,1,7,0,1,7,4,0,2,1),
        (2,1,6,5,7,3,3,4,4,5,6,2),
        (6,4,5,1,7,6,7,3,1,0,5,7),
        (4,5,2,5,4,5,0,4,4,7,3,6),
        (1,0,4,5,7,0,2,1,3,5,4,6),
        (2,1,4,3,7,5,3,7,6,2,7,4),
        (7,5,4,0,4,5,6,2,2,3,3,1),
        (6,5,4,7,4,1,0,5,6,5,2,3),
        (4,0,0,1,7,3,3,0,2,1,7,5),
        (3,6,2,5,6,3,7,6,1,7,1,6),
        (3,7,4,5,0,2,7,4,0,2,0,4),
        (4,5,2,7,0,1,0,3,2,6,4,3),
        (3,2,1,4,7,0,0,6,6,7,7,5),
        (6,2,3,4,1,6,4,5,1,5,6,2),
        (6,2,0,1,7,5,1,0,2,0,0,2),
        (0,7,2,7,4,1,0,3,4,5,4,1),
        (1,6,7,4,3,5,5,7,6,7,3,5),
        (5,2,1,7,2,4,5,7,2,3,3,3),
        None,
        (0,7,2,7,4,5,4,3,6,1,6,7),
        (0,7,3,1,4,2,4,2,0,7,1,6),
        (5,6,1,6,5,1,3,7,3,5,3,5),
        (6,2,5,4,0,2,0,1,0,2,1,7),
        (2,5,0,7,4,6,2,5,6,5,6,5),
        (6,7,1,1,3,3,5,1,3,0,4,4),
        (6,5,1,5,3,7,7,1,5,3,3,3),
        (3,7,0,1,0,2,3,2,3,1,4,6),
        (2,5,6,1,6,5,0,1,6,7,1,3),
    )
    for old_slot, aa in enumerate(old_normals):
        if aa is None:
            continue
        old_card = old_cards[(old_slot % 20) // 4]
        old_lane = old_slot // 20
        grids = []
        for side_index in range(2):
            row = []
            for j in range(3):
                kind = old_card[3 * side_index + j]
                if side_index:
                    kind = (kind + old_lane) % len(_AFF_PATTERNS)
                b, d = _AFF_PATTERNS[kind]
                offset = 6 * side_index + 2 * j
                row.append((G.e, (aa[offset], b), (aa[offset + 1], d)))
            grids.append((tuple(row),))
        old_hx, old_hz, old_n = lifted_matrices(G, *grids)
        emitted.add(_aff_fingerprint(old_hx, old_hz, old_n))

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

                # Seed complete-code searches with constituent supports,
                # but first verify every embedding's actual syndrome.
                # A words commute with HX; B words commute with HZ.
                xseeds = [
                    word << (j * 3 * q)
                    for word in b["words"][:8] for j in range(3)]
                zseeds = [
                    sum(((word >> (j * q)) & mask)
                        << ((j * 3 + beta) * q) for j in range(3))
                    for word in a["words"][:8] for beta in range(3)]
                xseeds = [
                    word for word in xseeds
                    if not any((word & row).bit_count() & 1
                               for row in hz)]
                zseeds = [
                    word for word in zseeds
                    if not any((word & row).bit_count() & 1
                               for row in hx)]
                # Complete-code readings classify supports modulo the
                # matching exact stabilizer span and permit both sectors.
                dx, xwords = _aff_probe(
                    _aff_columns(hz, n), local, 6, xseeds, hx)
                dz, zwords = _aff_probe(
                    _aff_columns(hx, n), local, 6, zseeds, hz)
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    a=a, b=b, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    fingerprint=fingerprint))
                if len(finalists) == 3:
                    break

            for item in finalists:
                dx, _ = _aff_probe(
                    _aff_columns(item["hz"], item["n"]), local, 12,
                    item["xwords"], item["hx"])
                dz, _ = _aff_probe(
                    _aff_columns(item["hx"], item["n"]), local, 12,
                    item["zwords"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                # Classical witnesses are conservative selection heuristics;
                # they are not automatically claimed as quantum logicals.
                reading = min(
                    item["a"]["reading"], item["b"]["reading"],
                    item["dx"], item["dz"], 12)
                if reading < 10:
                    # A found mixed-sector obstruction triggers another
                    # batch rather than admitting an already weak code.
                    continue
                objective = (
                    item["k"] * reading * reading / item["n"],
                    reading, item["k"],
                    item["a"]["reading"] + item["b"]["reading"])
                if winner is None or objective > winner[0]:
                    winner = (objective, item)
            if winner is not None and winner[0][1] >= 10:
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


def NEW_CTX(rng, G, idx):
    """Assign the independently optimized, invariant-distinct policy result."""
    bank = _design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("no fresh affine design passed this policy")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell; all new-shape indices are defined."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Affine cubic difference packing, exact constituent exclusion through six and the parent's bounded witness probes supply five measured readings >=9: the measured leaders instead read 10,10,10,8,8 at k=224, with sector-one minima.
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
