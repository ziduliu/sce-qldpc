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
CONCEPT = ("AGL(1,8), order 56, shapes (3,1)x(3,1): admitting one repeated-quotient trinomial in selected transpose components, relaxing difference packing and repairing actual witnesses should yield fresh codes with either readings above ten or verified rank surplus at ten, improving cell-0.400 fitness above 40.")
GROUP = ("Frozen byte-for-byte under L1. The exact order-56 affine group reached fitness 40 on both tall and wide cubic beds. The current five measured codes all had k=224 and d30s=10; these readings motivate broader filling selection, not a certified group or distance advantage.")
SHAPES = ("Frozen equal (3,1) sides: n=560, kb=224 and guaranteed rate 0.400. The size window permits q<=59, while the frozen group fixes q=56. Tall components have 56 variables and 168 checks; transpose kernels have 168 coordinates and 56 checks. Parent measured minima occupied only sector-two Arow/Brow blocks, prominently Arow1 and Brow1. Exact assembled ranks retain 0.375<=k/n<0.425.")
FILLING = ("All six cells remain cubic, retaining check weight twelve. Five architectures crossed with four coordinate policies introduce nonidentity translations and exactly one repeated quotient on each prescribed side. Relax six-distinct-difference packing: verified matrix kernels, not that sufficient construction filter, decide admission. Repair witness-heavy rows, exhaust constituent exclusion through eight on finalists, and search through ten with explicit completion flags and actual words. Common quotient factors on the opposite side test rank surplus without assuming it. Full-code searches verify embeddings and classify modulo stabilizers; unequal uncoloured Tanner invariants separate emitted designs. Bounded witness loads are not complete censuses or distance certificates.")
FAMILY_NAME = "r040_AGL1F8_31x31_translation_cubic_repair_short10_v3"

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

A_SHAPE = (3, 1)               # (ra, ca)
B_SHAPE = (3, 1)               # (rb, cb)

from itertools import combinations, combinations_with_replacement
from functools import lru_cache


# Five architectures crossed with four coordinate policies.
# Indices 15..20 introduce one nonidentity translation and one
# nonzero quotient term. More than one such cell on a component
# is deliberately avoided; actual matrix screens remain decisive.
_AFF_PATTERNS = (
    tuple(combinations(range(1, 7), 2))
    + tuple((0, b) for b in range(1, 7))
)
_AFF_CARDS = (
    (15, 1, 6, 0, 8, 14),       # translation on A only
    (0, 8, 12, 17, 1, 11),      # translation on B only
    (16, 1, 11, 19, 5, 6),      # one translation on each side
    (18, 0, 6, 1, 8, 12),       # B common quotient factor
    (3, 5, 13, 20, 0, 11),      # A opposite common factor
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
    """Compile tall transpose actions exclusively from verified unit lifts."""
    assert A_SHAPE == B_SHAPE == (3, 1)
    q = G.q
    mask = (1 << q) - 1
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[]] for _ in range(3)]
            B = [[[]] for _ in range(3)]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 10*q
            # Sector one has q columns. Sector-two block (0,0)
            # exposes A^T in HZ and B^T in HX.
            source = hz if side == "A" else hx
            rows = [(row >> q) & mask for row in source[:q]]
            block = _aff_columns(rows, q)
            assert all(value.bit_count() == 1 for value in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)
    return atlas


def _aff_component_columns(G, atlas, side, grid):
    """Linear superposition of scaffold-generated transpose unit blocks."""
    columns = [0] * (3 * G.q)
    for i, row in enumerate(grid):
        assert len(row) == 1
        for g in row[0]:
            for h, value in enumerate(atlas[side][G.idx(g)]):
                columns[i * G.q + h] ^= value
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


def _aff_eight_words(cols, q, node_cap, hit_cap=32, maximum=8):
    """Witness search through an even bound, with a completion flag.

    The caller has excluded kernels through four. Singleton/pair
    syndromes therefore have unique supports. Component block anchors
    cover regular-action orbits; no such anchor claim is made for the
    complete quantum code. Exhaustion without hits excludes component
    weights through maximum. A cutoff is only a bounded witness search.
    """
    width = len(cols)
    bits = [1 << i for i in range(width)]
    rows = [0] * q
    tails = {0: 0}
    for i, syndrome in enumerate(cols):
        assert syndrome.bit_count() == 3
        assert syndrome not in tails
        tails[syndrome] = bits[i]
        remaining = syndrome
        while remaining:
            bit = remaining & -remaining
            rows[bit.bit_length() - 1] |= bits[i]
            remaining ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            assert value not in tails
            tails[value] = bits[i] | bits[j]

    found = set()
    nodes = 0
    stopped = False

    def visit(syndrome, word, banned, weight):
        nonlocal nodes, stopped
        if stopped:
            return
        if node_cap is not None and nodes >= node_cap:
            stopped = True
            return
        nodes += 1
        if not syndrome:
            found.add(word)
            if len(found) >= hit_cap:
                stopped = True
            return

        remaining = maximum - weight
        if remaining <= 0 or syndrome.bit_count() > 3*remaining:
            return
        if remaining <= 2:
            tail = tails.get(syndrome)
            if (tail is not None and tail.bit_count() <= remaining
                    and not (tail & banned)):
                found.add(word | tail)
                if len(found) >= hit_cap:
                    stopped = True
            return

        pending = syndrome
        choices = 0
        least = width + 1
        while pending:
            bit = pending & -pending
            options = rows[bit.bit_length() - 1] & ~banned
            count = options.bit_count()
            if not count:
                return
            if count < least:
                choices, least = options, count
                if count == 1:
                    break
            pending ^= bit

        alternatives = []
        while choices:
            bit = choices & -choices
            choices ^= bit
            alternatives.append(bit.bit_length() - 1)
        alternatives.sort(
            key=lambda i: (-(cols[i] & syndrome).bit_count(), i))
        excluded = banned
        for i in alternatives:
            if stopped:
                break
            # Partition completions by their first selected alternative
            # at this odd check, excluding earlier alternatives.
            excluded |= bits[i]
            visit(syndrome ^ cols[i], word | bits[i],
                  excluded, weight + 1)

    for anchor in range(0, width, q):
        if stopped:
            break
        visit(cols[anchor], bits[anchor], bits[anchor], 1)

    words = tuple(sorted(found, key=lambda w: (w.bit_count(), w)))
    for word in words:
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length() - 1]
            remaining ^= bit
        assert word and syndrome == 0 and word.bit_count() <= maximum
    return words, not stopped, nodes


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
    """Fresh cubic voltages; matrix screens replace difference packing."""
    for restart in range(16):
        cells, seen = [], set()
        for j, kind in enumerate(kinds):
            b, d = _AFF_PATTERNS[kind]
            accepted = None
            for attempt in range(128):
                if model == 0:
                    a, c = local.randrange(8), local.randrange(8)
                elif model == 1:
                    a = local.randrange(8)
                    wanted = (j + lane + int(side == "B")) & 1
                    c = local.choice([
                        x for x in range(8)
                        if _aff_trace(a ^ x) == wanted])
                elif model == 2:
                    a = local.randrange(8)
                    c = _aff_field_mul(
                        a ^ 1, local.choice((2, 3, 4, 5, 6, 7)))
                else:
                    a = local.randrange(8)
                    c = a ^ local.choice((1, 2, 4, 7))

                # b=0 must mean an actual nonidentity translation.
                if b == 0 and a == 0:
                    continue
                g, h = (a, b), (c, d)
                cell = (G.e, g, h)
                if len(set(cell)) != 3:
                    continue
                if G.mul(g, h) == G.mul(h, g):
                    continue

                # Remove cell-gauge copies within this draw, but permit
                # repeated directed differences and Tanner four-cycles.
                # Neither necessarily implies a weight-four kernel.
                key = min(
                    tuple(sorted(G.idx(G.mul(x, G.inv(anchor)))
                                 for x in cell))
                    for anchor in cell)
                if key in seen:
                    continue
                seen.add(key)
                accepted = cell
                break
            if accepted is None:
                break
            cells.append(accepted)
        if len(cells) == len(kinds):
            return tuple((cell,) for cell in cells)
    return None


def _aff_component_pool(local, G, atlas, side, kinds, model, lane, batch):
    """Repair true witnesses, then exhaust eight and inspect ten."""
    pool, seen = [], set()
    q, mask = G.q, (1 << G.q) - 1
    serial = 0

    def geometry(cols):
        pairs, adjacency = {}, [0] * q
        cycles = 0
        for syndrome in cols:
            positions = []
            while syndrome:
                bit = syndrome & -syndrome
                positions.append(bit.bit_length() - 1)
                syndrome ^= bit
            for i, a in enumerate(positions):
                for b in positions[:i]:
                    key = (b, a)
                    old = pairs.get(key, 0)
                    cycles += old
                    pairs[key] = old + 1
                    adjacency[a] |= 1 << b
                    adjacency[b] |= 1 << a
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            pending = neighbors >> (a + 1)
            while pending:
                bit = pending & -pending
                b = a + 1 + bit.bit_length() - 1
                triangles += (
                    (neighbors & adjacency[b]) >> (b + 1)).bit_count()
                pending ^= bit
        return cycles, triangles

    def merit(item):
        return (
            min(item["reading"], 12),
            int(item.get("clean10", False)),
            int(item.get("clean8", False)),
            -len(item.get("eight", ())),
            -len(item.get("ten", ())),
            -item["rank"],
            -item["geometry"][0], -item["geometry"][1],
            -item["attempt"])

    def admit(grid):
        nonlocal serial
        serial += 1
        if grid is None or grid in seen:
            return
        seen.add(grid)
        cols = _aff_component_columns(G, atlas, side, grid)
        assert all(value.bit_count() == 3 for value in cols)
        if not _aff_clean4(cols):
            return

        reading, words = _aff_probe(cols, local, 2)
        if reading <= 6:
            return
        eight, complete, nodes = _aff_eight_words(
            cols, q, node_cap=4000, hit_cap=1)
        if eight:
            reading = min(reading, eight[0].bit_count())
            if reading <= 6:
                return
        seeds = tuple(sorted(
            set(words) | set(eight),
            key=lambda w: (w.bit_count(), w)))
        item = dict(
            grid=grid, cols=cols, reading=reading, words=seeds,
            eight=eight, clean8=complete and not eight,
            nodes8=nodes, rank=gf2_rank(cols),
            geometry=geometry(cols), attempt=serial)
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[20:]

    for attempt in range(160 + 64*batch):
        admit(_aff_draw(local, G, kinds, model, side, lane))

    # The old supports choose repair locations only. Changed matrices
    # get fresh syndrome searches; old words are never trusted as seeds.
    for step in range(64 + 24*batch):
        replacement = _aff_draw(
            local, G, kinds, model, side, lane + step)
        if replacement is None:
            continue
        if not pool:
            admit(replacement)
            continue
        parent = local.choice(pool[:min(6, len(pool))])
        witnesses = parent["eight"] or parent["words"][:8]
        pressure = [
            1 + sum(((word >> (i*q)) & mask).bit_count()
                    for word in witnesses)
            for i in range(3)]
        i = local.choices(range(3), weights=pressure, k=1)[0]
        rows = list(parent["grid"])
        rows[i] = replacement[i]
        if step % 9 == 8:
            h = (i + 1 + local.randrange(2)) % 3
            rows[h] = replacement[h]
        admit(tuple(rows))

    survivors = []
    for item in pool:
        if item["eight"]:
            continue
        if not item["clean8"]:
            # No node cutoff: a hit is a true obstruction; no hit
            # exhaustively excludes constituent kernels through eight.
            eight, complete, nodes = _aff_eight_words(
                item["cols"], q, node_cap=None, hit_cap=1)
            item.update(
                eight=eight, clean8=complete and not eight,
                nodes8=nodes)
            if eight:
                continue
            assert complete

        reading, words = _aff_probe(
            item["cols"], local, 16, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        assert words and item["reading"] >= 10

        # Identical bounded searches for every survivor. The returned
        # load is a procedural witness statistic, not a complete census.
        ten, complete, nodes = _aff_eight_words(
            item["cols"], q, node_cap=40000,
            hit_cap=12, maximum=10)
        item.update(
            ten=ten, complete10=complete, nodes10=nodes,
            clean10=complete and not ten)
        seeds = tuple(sorted(
            set(words) | set(ten),
            key=lambda w: (w.bit_count(), w)))
        reading, words = _aff_probe(
            item["cols"], local, 16, seeds)
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        if item["clean10"]:
            assert item["reading"] >= 12
        survivors.append(item)

    survivors.sort(key=merit, reverse=True)
    finalists = survivors[:4]
    for item in finalists:
        # Independent equal-budget probing prevents selecting solely
        # on an unusually favourable first information-set reading.
        reading, words = _aff_probe(
            item["cols"], local, 32, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        assert item["reading"] >= 10
    finalists.sort(key=merit, reverse=True)
    return finalists[:3]


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
    """Fresh tall designs, not transposed copies of historical fillings."""
    assert G.q == 56 and A_SHAPE == B_SHAPE == (3, 1)
    q = G.q
    mask = (1 << q) - 1
    atlas = _aff_atlas(G)
    counts = [[3], [3], [3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)
    emitted, bank = set(), []

    def embeddings(words, side):
        result = []
        for word in sorted(set(words), key=int.bit_count)[:12]:
            if side == "A":
                # A^T kernels commute with HZ at each fixed B row.
                for alpha in range(3):
                    result.append(sum(
                        ((word >> (i*q)) & mask)
                        << (q + (3*i + alpha)*q)
                        for i in range(3)))
            else:
                # B^T kernels commute with HX at each fixed A row.
                result.extend(word << (q + 3*i*q) for i in range(3))
        return tuple(result)

    def verified_seeds(words, cols):
        checked = []
        for word in words:
            syndrome, remaining = 0, word
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length() - 1]
                remaining ^= bit
            assert word and syndrome == 0
            checked.append(word)
        return tuple(checked)

    def merit(item):
        reading = min(
            item["a"]["reading"], item["b"]["reading"],
            item["dx"], item["dz"], 12)
        return (
            item["k"]*reading*reading/item["n"], reading,
            int(item["a"]["clean8"]) + int(item["b"]["clean8"]),
            -len(item["a"]["eight"]) - len(item["b"]["eight"]),
            -sum(item["a"]["geometry"]) - sum(item["b"]["geometry"]))

    for slot in range(N_SLOTS):
        architecture, model = divmod(slot % 20, 4)
        lane = slot // 20
        card = _AFF_CARDS[architecture]
        ak, bk = card[:3], card[3:]

        # Extra lanes alter relative geometry without removing the
        # prescribed translation or destroying a common-factor card.
        if architecture == 3:
            if lane & 1:
                bk = (3, 5, 13)
        elif architecture == 4:
            if lane & 1:
                ak = (1, 8, 12)
            if lane:
                bk = (15 + (bk[0] - 15 + lane) % 6,
                      bk[1], bk[2])
        elif lane:
            classes = (
                (0, 2, 4, 6, 7, 9, 10, 11, 14),
                (1, 3, 5, 8, 12, 13))
            changed = []
            for kind in bk:
                if kind >= 15:
                    changed.append(15 + (kind - 15 + lane) % 6)
                else:
                    family = next(c for c in classes if kind in c)
                    changed.append(
                        family[(family.index(kind) + lane) % len(family)])
            bk = tuple(changed)
        winner = None

        for batch in range(3):
            local = random.Random(
                BASE_SEED*100003 + 56314197
                + 104729*slot + 7919*batch)
            ap = _aff_component_pool(
                local, G, atlas, "A", ak, model, lane, batch)
            bp = _aff_component_pool(
                local, G, atlas, "B", bk, model, lane + 1, batch)
            proposals = sorted(
                ((min(a["reading"], b["reading"]),
                  int(a["clean8"]) + int(b["clean8"]),
                  -len(a["eight"]) - len(b["eight"]), ai, bi, a, b)
                 for ai, a in enumerate(ap)
                 for bi, b in enumerate(bp)),
                key=lambda entry: entry[:5], reverse=True)
            finalists = []

            for _, _, _, _, _, a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 560 or not (375*n <= 1000*k < 425*n):
                    continue

                # Compare BOTH compiled transpose components with the
                # complete verified construction, including column order.
                ar = [
                    sum(((row >> (q + 3*i*q)) & mask) << (i*q)
                        for i in range(3))
                    for row in hz[:q]]
                br = [
                    (row >> q) & ((1 << (3*q)) - 1)
                    for row in hx[:q]]
                assert _aff_columns(ar, 3*q) == a["cols"]
                assert _aff_columns(br, 3*q) == b["cols"]

                xc = _aff_columns(hz, n)
                zc = _aff_columns(hx, n)
                # All these full columns are odd-weight. This conservative
                # screen excludes all kernels through four, not merely
                # the constituent kernels.
                assert all(c.bit_count() & 1 for c in xc + zc)
                if not (_aff_clean4(xc) and _aff_clean4(zc)):
                    continue
                fingerprint = _aff_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue

                xseeds = verified_seeds(embeddings(a["words"], "A"), xc)
                zseeds = verified_seeds(embeddings(b["words"], "B"), zc)
                dx, xwords = _aff_probe(
                    xc, local, 6, xseeds, stabilizers=hx)
                dz, zwords = _aff_probe(
                    zc, local, 6, zseeds, stabilizers=hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    a=a, b=b, xc=xc, zc=zc, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    fingerprint=fingerprint))
                if len(finalists) == 4:
                    break

            # Identical fresh budgets for every assembled finalist.
            for item in finalists:
                dx, _ = _aff_probe(
                    item["xc"], local, 16,
                    item["xwords"], item["hx"])
                dz, _ = _aff_probe(
                    item["zc"], local, 16,
                    item["zwords"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 10:
                break

        if winner is None:
            # A failed policy never borrows another slot's filling.
            bank.append(None)
            continue
        assert winner["n"] == 560 and winner["k"] >= 224
        assert 375*winner["n"] <= 1000*winner["k"] < 425*winner["n"]
        assert css_ok(winner["hx"], winner["hz"])
        assert max(row.bit_count()
                   for row in winner["hx"] + winner["hz"]) == 12
        A, B = winner["A"], winner["B"]
        assert [[len(cell) for cell in row] for row in A] == counts
        assert [[len(cell) for cell in row] for row in B] == counts
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(winner["fingerprint"])
        bank.append((A, B))

    # Preserve successful slots even when another policy is exhausted.
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
# REFUTED: Exact-six constituent exclusion plus the existing witness probes supply five AGL(1,8) readings >=9: the measured parent readings were 10,10,10,8,8, all with k=224.
# REFUTED: Common-factor cards with strict six-difference packing reliably supply exact-eight-free affine constituents: both common-factor architectures in the shown exact8 wide program were void.
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
