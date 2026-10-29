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
CONCEPT = ("AGL(1,8), order 56, shapes (3,1)x(3,1): rank-stratified cubic fillings with regular-orbit recombination and stabilizer-aware classification of found constituent witnesses should exceed five-code fitness 40 in cell 0.400. Evaluation can refute this; constituent search misses are not quantum-distance certificates.")
GROUP = ("Frozen byte-for-byte under L1. This exact tall affine bed measured five [560,224,10] leaders and fitness 40, while its remaining slots read eight or ten. Keep that measured bed and change only filling and selection; no group-distance advantage is inferred.")
SHAPES = ("Frozen equal tall (3,1) sides: n=560, kb=224, guaranteed rate 0.400, maximum admissible lift order 59 and actual q=56. Each component has 56 variables and 168 checks; each transpose kernel has 168 coordinates and 56 checks. Parent measured minima occupied only sector-two Arow/Brow blocks, especially Brow1 for X and Arow1 for Z. Require exact assembled 0.375<=k/n<0.425.")
FILLING = ("All six cells remain cubic, retaining both check weights twelve. Twenty cards cross five AP/chiral/common-factor/translation quotient architectures with four fixed-point policies. Soft relative-difference packing permits new incidence geometries. Preserve component-rank strata, repair actual witness-heavy cells, and recombine syndrome-verified regular-action witness orbits. Search constituent weights eight and ten with explicit completion flags, but classify found short-word orbit embeddings modulo the actual full-code stabilizer spaces rather than automatically treating their classical weights as logical distances. Equal deeper full-code budgets determine selection. Reject all 60 recorded parent Tanner signatures and repeated emitted signatures; the evaluator handles the remaining archive.")
FAMILY_NAME = "r040_AGL1F8_31x31_cubic_rankhomology_orbit_v3"

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


# Five new relative quotient schedules crossed with four coordinate
# policies. These are construction constraints, not diversity proofs.
# Extra harness lanes alter only B's relative schedule.
_AFF_PATTERNS = tuple(combinations(range(1, 7), 2))
_AFF_CARDS = (
    (0, 5, 9, 2, 8, 14),
    (1, 4, 11, 3, 6, 12),
    (0, 6, 13, 1, 7, 10),
    (2, 9, 14, 0, 4, 12),
    (4, 8, 11, 5, 10, 13),
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


def _aff_eight_words(cols, q, node_cap, hit_cap=32):
    """Witness search through eight, with an explicit completion flag.

    The caller has excluded kernels through four. Singleton/pair
    syndromes therefore have unique supports. Component block anchors
    cover regular-action orbits; no such anchor claim is made for the
    complete quantum code. Exhaustion without hits excludes component
    weights through eight. A cutoff is only a bounded witness search.
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

        remaining = 8 - weight
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
        assert word and syndrome == 0 and word.bit_count() <= 8
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
    """Fresh tall cells; relative schedules are not transported winners."""
    for restart in range(32):
        used, cells = set(), []
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

                g, h = (a, b), (c, d)
                if G.mul(g, h) == G.mul(h, g):
                    continue
                cell = (G.e, g, h)
                differences = {
                    G.mul(G.inv(x), y)
                    for x in cell for y in cell if x != y
                }
                if len(differences) != 6 or differences & used:
                    continue
                assert len(set(cell)) == 3
                accepted = cell
                used.update(differences)
                break
            if accepted is None:
                break
            cells.append(accepted)
        if len(cells) == len(kinds):
            return tuple((cell,) for cell in cells)
    return None


def _aff_component_pool(local, G, atlas, side, kinds, model, lane, batch):
    """Optimize transpose kernels against actual eight-support witnesses."""
    pool, seen = [], set()

    def geometry(cols):
        pairs, adjacency = {}, [0] * G.q
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
        return (item["reading"], int(item.get("clean8", False)),
                -len(item.get("eight", ())),
                -item["geometry"][0], -item["geometry"][1],
                -item["rank"], -item["attempt"])

    for attempt in range(96 + 48*batch):
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
        item = dict(
            grid=grid, cols=cols, reading=reading, words=words,
            rank=gf2_rank(cols), geometry=geometry(cols),
            attempt=attempt)
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[16:]

    survivors = []
    for item in pool:
        if not _aff_clean6(item["cols"], G.q):
            continue
        eight, complete, nodes = _aff_eight_words(
            item["cols"], G.q, node_cap=12000, hit_cap=16)
        seeds = tuple(set(item["words"]) | set(eight))
        reading, words = _aff_probe(item["cols"], local, 8, seeds)
        item.update(
            reading=min(item["reading"], reading),
            words=words, eight=eight,
            clean8=complete and not eight, nodes8=nodes)
        survivors.append(item)

    survivors.sort(key=merit, reverse=True)
    finalists = survivors[:4]
    for item in finalists:
        eight, complete, nodes = _aff_eight_words(
            item["cols"], G.q, node_cap=120000, hit_cap=96)
        witnessed = tuple(sorted(
            set(item["eight"]) | set(eight),
            key=lambda w: (w.bit_count(), w)))
        seeds = tuple(set(item["words"]) | set(witnessed))
        reading, words = _aff_probe(item["cols"], local, 32, seeds)
        item.update(
            reading=min(item["reading"], reading),
            words=words, eight=witnessed,
            clean8=complete and not eight, nodes8=nodes)
        assert words and reading <= len(item["cols"])
        if item["clean8"]:
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
        ak = card[:3]
        bk = tuple((kind + 3*lane) % len(_AFF_PATTERNS)
                   for kind in card[3:])
        winner = None

        for batch in range(2):
            local = random.Random(
                BASE_SEED*100003 + 56314089
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


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure replacement bank; the historical _design_bank is inactive."""
    if idx >= 0:
        bank = NEW_CTX(None, G, -1)
        if idx >= len(bank) or bank[idx] is None:
            raise ValueError("no fresh rank/orbit-screened tall cubic filling")
        A, B, k, selection = bank[idx]
        return {"A": A, "B": B, "k": k, "selection": selection}

    from collections import Counter

    assert G.q == 56 and A_SHAPE == B_SHAPE == (3, 1)
    q, width = G.q, 3*G.q
    mask = (1 << q)-1
    atlas = _aff_atlas(G)
    counts = [[3], [3], [3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    def syndrome(word, cols):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    # These permutations construct candidate supports, never parity
    # matrices. Every retained transformed word is syndrome-verified.
    maps = {}
    for side in ("A", "B"):
        permutations = []
        for t in G.elements:
            permutation = [
                G.idx(G.mul(g, t) if side == "A" else G.mul(t, g))
                for g in G.elements]
            assert len(set(permutation)) == q
            permutations.append(tuple(
                1 << (j*q+permutation[h])
                for j in range(3) for h in range(q)))
        maps[side] = tuple(permutations)

    def moved(word, permutation):
        out = 0
        while word:
            bit = word & -word
            out ^= permutation[bit.bit_length()-1]
            word ^= bit
        return out

    def orbit_reduce(cols, words, side):
        """Recombine true kernel orbits; absence of a light word proves nothing."""
        cache = {}

        def canonical(word):
            old = cache.get(word)
            if old is not None:
                return old
            orbit = [moved(word, p) for p in maps[side]]
            representative = min(orbit)
            assert syndrome(representative, cols) == 0
            for value in orbit:
                cache[value] = representative
            return representative

        representatives = sorted(
            {canonical(w) for w in words if w and syndrome(w, cols) == 0},
            key=lambda w: (w.bit_count(), w))[:12]
        discoveries = set(representatives)
        for i, left in enumerate(representatives[:8]):
            for right in representatives[:i+1]:
                candidates = {
                    left ^ moved(right, p) for p in maps[side]}
                candidates.discard(0)
                for word in sorted(
                        candidates, key=lambda w: (w.bit_count(), w))[:3]:
                    assert syndrome(word, cols) == 0
                    discoveries.add(canonical(word))
        return tuple(sorted(
            discoveries, key=lambda w: (w.bit_count(), w))[:20])

    def through_ten(cols, node_cap=36000, hit_cap=24):
        """Anchored constituent search through ten, with a completion flag.

        clean4 has passed. No-hit exhaustion excludes constituent
        weights through ten; a node/hit cutoff is only a bounded search.
        These anchors are not asserted to cover full quantum-code orbits.
        """
        bits = [1 << i for i in range(width)]
        rows = [0]*q
        tails = {0: 0}
        for i, original in enumerate(cols):
            assert original.bit_count() == 3 and original not in tails
            tails[original] = bits[i]
            value = original
            while value:
                bit = value & -value
                rows[bit.bit_length()-1] |= bits[i]
                value ^= bit
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                assert pair not in tails
                tails[pair] = bits[i] | bits[j]

        found, nodes, stopped = set(), 0, False

        def record(word):
            nonlocal stopped
            assert word and syndrome(word, cols) == 0
            found.add(word)
            if len(found) >= hit_cap:
                stopped = True

        def visit(value, remaining, banned, word):
            nonlocal nodes, stopped
            if stopped:
                return
            if nodes >= node_cap:
                stopped = True
                return
            nodes += 1
            if not value:
                record(word)
                return
            if (value.bit_count() ^ remaining) & 1:
                remaining -= 1
            if remaining <= 0 or value.bit_count() > 3*remaining:
                return
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & banned):
                    record(word | tail)
                return

            pending, choices, least = value, 0, width+1
            while pending:
                bit = pending & -pending
                options = rows[bit.bit_length()-1] & ~banned
                size = options.bit_count()
                if not size:
                    return
                if size < least:
                    choices, least = options, size
                    if size == 1:
                        break
                pending ^= bit
            alternatives = []
            while choices:
                bit = choices & -choices
                choices ^= bit
                alternatives.append(bit.bit_length()-1)
            alternatives.sort(
                key=lambda j: (-(cols[j] & value).bit_count(), j))
            excluded = banned
            for j in alternatives:
                if stopped:
                    break
                excluded |= bits[j]
                visit(value ^ cols[j], remaining-1,
                      excluded, word | bits[j])

        for anchor in range(0, width, q):
            if stopped:
                break
            visit(cols[anchor], 9, bits[anchor], bits[anchor])
        return tuple(sorted(
            found, key=lambda w: (w.bit_count(), w))), not stopped

    def cycles(cols):
        pairs, answer = {}, 0
        for value in cols:
            positions = []
            while value:
                bit = value & -value
                positions.append(bit.bit_length()-1)
                value ^= bit
            for i, a in enumerate(positions):
                for b in positions[:i]:
                    key = (b, a)
                    old = pairs.get(key, 0)
                    answer += old
                    pairs[key] = old+1
        return answer

    action = [1]
    for _ in range(6):
        action.append(_aff_field_mul(action[-1], 2))
    ap_pairs = tuple(
        (b, d) for b, d in combinations(range(1, 7), 2)
        if (b+d) % 7 == 0 or (2*b-d) % 7 == 0 or (2*d-b) % 7 == 0)
    chiral_pairs = tuple(
        pair for pair in combinations(range(1, 7), 2)
        if pair not in ap_pairs)
    factors = (
        ((1, 3), (2, 6), (4, 5)),
        ((1, 5), (2, 3), (4, 6)),
    )
    # AP/chiral controls, mixed controls, translation experiment,
    # complementary common factors, and a one-sided common factor.
    architectures = (
        ((0, 0, 1), (0, 1, 0)),
        ((0, 1, 1), (1, 0, 1)),
        ((2, 0, 1), (1, 2, 0)),
        ((3, 3, 3), (4, 4, 4)),
        ((3, 3, 3), (0, 1, 1)),
    )

    def draw(local, side, architecture, mode, stage=0):
        """Prescribed intrinsic families with soft difference packing."""
        kinds = list(architectures[architecture][int(side == "B")])
        if stage and architecture in (3, 4):
            # Explicit full-rank control inside the same experimental task.
            kinds[(architecture+int(side == "B")) % 3] = 0
        shared = local.randrange(8)
        used, grid = Counter(), []
        for j, kind in enumerate(kinds):
            best = None
            for attempt in range(10):
                if kind == 0:
                    b, d = local.choice(ap_pairs)
                elif kind == 1:
                    b, d = local.choice(chiral_pairs)
                elif kind == 2:
                    b, d = 0, local.randrange(1, 7)
                else:
                    b, d = factors[kind-3][
                        (j+local.randrange(3)) % 3]

                if mode == 0:
                    a, c = local.randrange(8), local.randrange(8)
                else:
                    p = shared if mode == 3 else local.randrange(8)
                    if mode == 1:
                        r = p ^ (1, 2, 4)[j]
                    elif mode == 2:
                        p = local.choice(
                            [x for x in range(8) if _aff_trace(x) == 0])
                        r = local.choice(
                            [x for x in range(8) if _aff_trace(x) == 1])
                    else:
                        r = p ^ local.choice((1, 2, 3, 4, 5, 6, 7))
                    a = (_aff_field_mul(1 ^ action[b], p) if b
                         else local.randrange(1, 8))
                    c = _aff_field_mul(1 ^ action[d], r)

                g, h = (a, b), (c, d)
                cell = (G.e, g, h)
                if len(set(cell)) != 3 or G.mul(g, h) == G.mul(h, g):
                    continue
                differences = [
                    G.mul(G.inv(x), y)
                    for x in cell for y in cell if x != y]
                penalty = (
                    3*(6-len(set(differences)))
                    + sum(used[value] for value in differences))
                key = (penalty, local.random())
                if best is None or key < best[0]:
                    best = (key, cell, differences)
            if best is None:
                return None
            _, cell, differences = best
            used.update(differences)
            grid.append((cell,))
        return tuple(grid)

    pools = {}

    def component_pool(side, architecture, mode):
        """Share immutable screened pools, not transported final codes."""
        key = side, architecture, mode
        if key in pools:
            return pools[key]
        local = random.Random(
            BASE_SEED*100003 + 56314197 + 104729*architecture
            + 7919*mode + 1000003*int(side == "B"))
        strata, seen, serial = {}, set(), 0

        def early(item):
            return (min(item["reading"], 12),
                    -item["cycles"], -item["serial"])

        def admit(grid):
            nonlocal serial
            serial += 1
            if grid is None or grid in seen:
                return
            seen.add(grid)
            cols = _aff_component_columns(G, atlas, side, grid)
            assert all(c.bit_count() == 3 for c in cols)
            if not _aff_clean4(cols):
                return
            rank = gf2_rank(cols)
            if rank < q-6:
                return
            reading, words = _aff_probe(cols, local, 2)
            if reading <= 6 or not words:
                return
            item = dict(
                grid=grid, cols=cols, rank=rank, reading=reading,
                words=words, cycles=cycles(cols), serial=serial)
            bucket = strata.setdefault(rank, [])
            bucket.append(item)
            bucket.sort(key=early, reverse=True)
            del bucket[6:]

        for stage, quota in ((0, 128), (1, 96)):
            for attempt in range(quota):
                admit(draw(local, side, architecture, mode, stage))

        # Pressure comes from actual words of the parent matrix.
        # A changed matrix starts a new, independently checked memory.
        for step in range(32):
            fresh = draw(local, side, architecture, mode, step & 1)
            if fresh is None:
                continue
            if not strata:
                admit(fresh)
                continue
            rank = local.choice(sorted(strata))
            parent = local.choice(strata[rank][:3])
            pressure = [
                1+sum(((w >> (j*q)) & mask).bit_count()
                      for w in parent["words"][:10])
                for j in range(3)]
            j = local.choices(range(3), weights=pressure, k=1)[0]
            grid = list(parent["grid"])
            grid[j] = fresh[j]
            if step % 5 == 4:
                h = (j+1+local.randrange(2)) % 3
                grid[h] = fresh[h]
            admit(tuple(grid))

        # Preserve deficient-image experiments alongside full-rank controls.
        ordered = [
            item for rank in sorted(strata)
            for item in strata[rank][:2]]
        extras = sorted(
            (item for bucket in strata.values() for item in bucket[2:]),
            key=early, reverse=True)
        ordered.extend(extras[:max(0, 12-len(ordered))])
        survivors = []
        for item in ordered[:12]:
            cols = item["cols"]
            if not _aff_clean6(cols, q):
                continue
            eight, complete8, _ = _aff_eight_words(
                cols, q, node_cap=18000, hit_cap=12)
            if not eight and not complete8:
                # Finish the no-hit constituent-eight test without a cutoff.
                eight, complete8, _ = _aff_eight_words(
                    cols, q, node_cap=None, hit_cap=1)
            ten, complete10 = through_ten(cols)
            seeds = tuple(set(item["words"]) | set(eight) | set(ten))
            seeds = orbit_reduce(cols, seeds, side)
            reading, words = _aff_probe(cols, local, 12, seeds)
            words = orbit_reduce(cols, tuple(words)+seeds, side)
            reading = min(reading, min(w.bit_count() for w in words))
            item.update(
                reading=min(item["reading"], reading), words=words,
                short=tuple(set(eight) | set(ten)),
                clean8=complete8 and not eight,
                clean10=complete10 and not ten,
                census_complete=complete10,
                ten_load=len(ten) if complete10 else 24)
            survivors.append(item)

        def refined(item):
            d = min(item["reading"], 12)
            return (d, int(item["clean10"]),
                    -item["ten_load"], -item["cycles"], -item["serial"])

        # Keep two candidates per rank before using the remaining positions.
        by_rank = {}
        for item in survivors:
            by_rank.setdefault(item["rank"], []).append(item)
        retained = []
        for rank in sorted(by_rank):
            by_rank[rank].sort(key=refined, reverse=True)
            retained.extend(by_rank[rank][:2])
        extras = sorted(
            (item for bucket in by_rank.values() for item in bucket[2:]),
            key=refined, reverse=True)
        retained.extend(extras[:max(0, 6-len(retained))])
        retained = retained[:6]
        for item in retained:
            reading, words = _aff_probe(
                item["cols"], local, 24, item["words"])
            words = orbit_reduce(item["cols"], words+item["words"], side)
            item["reading"] = min(
                item["reading"], reading,
                min(w.bit_count() for w in words))
            item["words"] = words
            # Preserve found short-word orbit representatives separately:
            # a randomized memory limit must not erase a known obstruction.
            item["short"] = orbit_reduce(
                item["cols"], item["short"], side)
        pools[key] = tuple(sorted(retained, key=refined, reverse=True))
        return pools[key]

    def low_logical(cols, stabilizers):
        """Exact stabilizer-aware logical existence through weight four."""
        span = _aff_basis(stabilizers)
        seen = {0: 0}
        for i, value in enumerate(cols):
            word = 1 << i
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _aff_outside(word ^ old, span):
                return True
        for i, value in enumerate(cols):
            for j in range(i):
                syn = value ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syn)
                if old is None:
                    seen[syn] = word
                elif _aff_outside(word ^ old, span):
                    return True
        return False

    def embedded_seeds(component, side, full_cols):
        """Classify found short-word orbits, not only one lucky representative."""
        originals = sorted(
            set(component["short"]) | set(component["words"]),
            key=lambda w: (w.bit_count(), w))[:20]
        expanded, seen = [], set()
        for original in originals:
            orbit = (
                {moved(original, p) for p in maps[side]}
                if original.bit_count() <= 10 else {original})
            for word in sorted(orbit):
                assert syndrome(word, component["cols"]) == 0
                for fixed in range(3):
                    if side == "A":
                        full = sum(
                            ((word >> (i*q)) & mask)
                            << (q+(3*i+fixed)*q) for i in range(3))
                    else:
                        full = word << (q+3*fixed*q)
                    if full in seen:
                        continue
                    seen.add(full)
                    assert syndrome(full, full_cols) == 0
                    expanded.append(full)
        return tuple(expanded)

    def logical_probe(cols, stabilizers, seeds, local, trials):
        """True logical witnesses with separate logical/kernel memories."""
        span = _aff_basis(stabilizers)
        n = len(cols)
        best, found = n+1, set()

        def observe(word):
            nonlocal best
            if not word or word.bit_count() > best+4:
                return
            if _aff_outside(word, span):
                best = min(best, word.bit_count())
                found.add(word)

        for word in seeds:
            assert syndrome(word, cols) == 0
            observe(word)
            if best < 10:
                return best, tuple(sorted(
                    found, key=lambda w: (w.bit_count(), w))[:12])

        memory = sorted(found, key=lambda w: (w.bit_count(), w))[:16]
        order = list(range(n))
        for trial in range(trials):
            local.shuffle(order)
            if trial % 3 != 2:
                # Expose each LP sector in alternating information sets.
                order.sort(key=lambda c:
                           (c < q) if trial % 3 == 0 else (c >= q))
            pivots, dependencies = {}, []
            for c in order:
                value, word = cols[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = value, word
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    dependencies.append(word)
                    observe(word)

            reducers = sorted(
                set(dependencies), key=lambda w: (w.bit_count(), w))[:32]
            logicals = sorted(
                {w for w in dependencies+memory if _aff_outside(w, span)},
                key=lambda w: (w.bit_count(), w))[:8]
            for word in logicals:
                for sweep in range(2):
                    chosen, weight = word, word.bit_count()
                    for reducer in reducers+memory:
                        candidate = word ^ reducer
                        cw = candidate.bit_count()
                        if 0 < cw < weight and _aff_outside(candidate, span):
                            chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                for sweep in range(3):
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda item: item[0])[:12]
                    chosen, weight = word, word.bit_count()
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)

            short = sorted(found, key=lambda w: (w.bit_count(), w))[:16]
            for i, word in enumerate(short):
                for other in short[:i]:
                    observe(word ^ other)
            memory = sorted(found, key=lambda w: (w.bit_count(), w))[:16]
            found = set(memory)
            if best < 10:
                break
        assert all(syndrome(w, cols) == 0 for w in memory)
        return best, tuple(memory)

    # Negative archive only: current parent's 60 recorded normal-coordinate
    # outputs. None is a proposal, repair parent or positive fallback.
    old_normals = (
        (3,7,6,4,5,4,3,4,5,7,5,2),
        (4,0,5,4,2,2,6,2,1,4,0,6),
        (0,5,0,5,3,4,1,0,7,4,1,0),
        (4,0,2,5,7,5,0,4,1,4,6,2),
        (5,7,0,6,6,3,4,5,5,2,6,4),
        (2,6,3,6,4,0,3,1,2,4,6,3),
        (0,3,6,6,7,1,4,4,2,6,7,1),
        (6,1,5,1,7,0,1,6,3,5,4,2),
        (6,6,5,6,3,6,3,7,7,1,1,0),
        (7,3,7,2,7,7,2,0,2,1,3,3),
        (4,3,2,7,2,7,6,7,3,7,6,2),
        (5,1,7,5,2,5,7,5,7,0,6,2),
        (1,7,6,0,4,4,0,5,2,2,2,4),
        (4,4,7,0,7,5,6,2,1,4,7,2),
        (6,4,4,2,3,6,6,2,2,4,4,3),
        (6,2,7,0,7,5,1,0,3,2,0,7),
        (1,4,6,6,7,0,4,4,2,1,1,2),
        (2,4,0,7,5,7,2,5,7,5,6,4),
        (4,4,1,0,3,7,6,6,7,2,0,4),
        (1,6,3,7,5,3,6,4,7,5,1,3),
        (6,3,5,5,2,6,0,1,7,2,3,3),
        (4,5,1,5,4,1,3,0,6,4,4,7),
        (4,6,5,7,4,1,2,6,3,7,0,1),
        (3,1,3,2,3,2,1,0,2,5,4,4),
        (5,0,0,1,3,2,0,1,5,2,1,4),
        (1,0,4,0,6,1,3,0,5,5,4,2),
        (1,0,4,6,1,0,6,2,7,5,6,4),
        (4,6,2,5,1,3,0,4,6,7,0,1),
        (6,0,7,2,4,7,5,0,4,5,0,4),
        (4,1,3,3,0,7,1,0,3,1,4,6),
        (7,2,2,6,7,5,3,7,3,3,5,6),
        (2,0,4,0,5,7,7,5,2,0,3,4),
        (6,7,6,1,1,1,4,4,0,1,3,5),
        (1,2,3,5,5,2,7,2,6,3,0,1),
        (1,3,1,1,4,1,6,5,0,1,5,2),
        (3,1,2,5,0,5,4,1,2,0,3,6),
        (2,4,2,0,3,6,4,0,7,6,5,7),
        (6,5,2,0,1,4,0,6,2,0,2,6),
        (0,4,7,3,4,2,0,6,7,5,0,1),
        (4,5,5,4,6,1,0,0,2,5,3,5),
        (3,4,1,0,3,7,1,2,6,3,3,0),
        (0,6,5,0,4,7,6,6,6,3,4,7),
        (7,6,2,7,1,1,7,4,1,5,6,3),
        (0,1,3,2,7,4,3,7,4,4,7,0),
        (7,2,1,3,3,2,4,2,4,5,4,1),
        (1,5,5,2,7,2,1,0,1,6,1,1),
        (2,1,1,0,7,2,0,3,1,0,2,1),
        (5,2,2,5,0,4,1,0,4,5,5,7),
        (2,7,0,2,7,5,5,7,7,4,5,0),
        (2,2,3,2,7,6,7,5,4,1,1,6),
        (5,6,3,1,7,6,4,3,2,0,4,1),
        (5,7,5,1,2,0,4,6,2,0,3,1),
        (1,1,5,4,2,2,3,7,7,2,4,0),
        (3,3,6,5,0,6,6,6,2,5,5,3),
        (1,0,2,1,0,0,5,2,4,7,3,5),
        (3,1,5,2,6,6,3,2,2,4,4,1),
        (2,2,5,0,4,6,0,3,0,2,1,2),
        (2,6,6,7,0,3,1,1,0,3,7,6),
        (1,0,4,2,3,4,7,1,1,4,5,3),
        (4,3,1,5,2,0,5,4,0,4,7,3),
    )
    emitted, bank = set(), []
    for old_slot, normals in enumerate(old_normals):
        card = _AFF_CARDS[(old_slot % 20)//4]
        grids = []
        for side_index in range(2):
            grid = []
            for j in range(3):
                kind = card[3*side_index+j]
                if side_index:
                    kind = (kind+3*(old_slot//20)) % len(_AFF_PATTERNS)
                b, d = _AFF_PATTERNS[kind]
                t = 6*side_index+2*j
                grid.append(((G.e, (normals[t], b),
                              (normals[t+1], d)),))
            grids.append(tuple(grid))
        hx, hz, n = lifted_matrices(G, *grids)
        emitted.add(_aff_fingerprint(hx, hz, n))

    def merit(item):
        # Only actual full-code logical readings limit this score.
        # A classical word can become a stabilizer after embedding.
        d = min(item["dx"], item["dz"], 12)
        return (item["k"]*d*d/item["n"], d, item["k"],
                int(item["a"]["clean10"])+int(item["b"]["clean10"]),
                -item["a"]["ten_load"]-item["b"]["ten_load"])

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        ap = component_pool("A", architecture, mode)
        bp = component_pool("B", architecture, mode)
        local = random.Random(
            BASE_SEED*100003 + 56314271 + 104729*slot)
        proposals = [
            (min(a["reading"], b["reading"]),
             2*q-a["rank"]-b["rank"], ai, bi, a, b)
            for ai, a in enumerate(ap) for bi, b in enumerate(bp)]
        normal = sorted(proposals, key=lambda p: p[:4], reverse=True)
        surplus = sorted(
            (p for p in proposals if p[1]),
            key=lambda p: (p[1], p[0], p[2], p[3]), reverse=True)
        ordered, visited = [], set()
        for proposal in surplus[:6]+normal:
            pair = proposal[2:4]
            if pair not in visited:
                visited.add(pair)
                ordered.append(proposal)

        finalists = []
        for _, _, _, _, a, b in ordered:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 560 or not (375*n <= 1000*k < 425*n):
                continue
            ar = [
                sum(((row >> (q+3*i*q)) & mask) << (i*q)
                    for i in range(3))
                for row in hz[:q]]
            br = [(row >> q) & ((1 << width)-1) for row in hx[:q]]
            assert _aff_columns(ar, width) == a["cols"]
            assert _aff_columns(br, width) == b["cols"]

            xc, zc = _aff_columns(hz, n), _aff_columns(hx, n)
            if low_logical(xc, hx) or low_logical(zc, hz):
                continue
            fingerprint = _aff_fingerprint(hx, hz, n)
            if fingerprint in emitted:
                continue
            xseeds = embedded_seeds(a, "A", xc)
            zseeds = embedded_seeds(b, "B", zc)
            dx, wx = logical_probe(xc, hx, xseeds, local, 4)
            dz, wz = logical_probe(zc, hz, zseeds, local, 4)
            if min(dx, dz) < 10 or dx > n or dz > n:
                continue
            finalists.append(dict(
                A=A, B=B, a=a, b=b, hx=hx, hz=hz, n=n, k=k,
                xc=xc, zc=zc, dx=dx, dz=dz, wx=wx, wz=wz,
                fingerprint=fingerprint))
            if len(finalists) == 4:
                break

        # Equal fresh full-code budgets for every admitted joint finalist.
        for item in finalists:
            dx, _ = logical_probe(
                item["xc"], item["hx"], item["wx"], local, 16)
            dz, _ = logical_probe(
                item["zc"], item["hz"], item["wz"], local, 16)
            item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)

        passing = [item for item in finalists if merit(item)[1] >= 10]
        if not passing:
            bank.append(None)
            continue
        winner = max(passing, key=merit)
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["n"] == 560 and winner["k"] >= 224
        assert 375*560 <= 1000*winner["k"] < 425*560
        assert max(row.bit_count()
                   for row in winner["hx"]+winner["hz"]) == 12
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(winner["fingerprint"])
        bank.append((A, B, winner["k"], merit(winner)))

    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete rank/orbit-selected cubic cell without new draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free three-binomial plus trinomial labeling and weight-four kernel multiplicity selection remove the D34 weight-four bottleneck: all five measured parent codes retained d30s=4 and k=306.
# REFUTED: Exact-six constituent exclusion plus the existing witness probes supply five AGL(1,8) readings >=9: the measured parent readings were 10,10,10,8,8, all with k=224.
# REFUTED: The displayed explicit-ten/common-quotient-factor cubic selection exceeds fitness 40 by retaining rank surplus or distance twelve: D2's five measured leaders instead all had k=224 and d30s=10.
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
