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
CONCEPT = ("Dic14, order 56, shapes (1,3)x(1,3): one non-AP pure-rotation trinomial and two mixed trinomials per constituent, selected against explicit weight-ten dependencies, should raise five-code cell-0.400 fitness above forty; measured full-code searches can refute this.")
GROUP = ("Frozen byte-for-byte under L1, including the dicyclic presentation and order 56. This exact bed measured five [560,224,10] leaders and fitness forty, while other distinct slots read eight. Keep the measured bed and change its constituent algebra rather than infer a group-distance advantage.")
SHAPES = ("Frozen equal A=(1,3), B=(1,3): n=560, kb=224, guaranteed rate 0.400, and equal 168-variable/56-check constituents. The size window permits q<=59. Current-parent minimum logicals were entirely sector-one Acol/Bcol supports, prominently Acol1 and Bcol1; none listed used Arow/Brow. Actual scaffold ranks retain 0.375<=k/n<0.425.")
FILLING = ("Twenty deliberate assemblies cross five A pure-triangle/mixed-gap recipes with four B recipes. Exactly one cell on each side is a non-AP rotation trinomial; the other two are mixed, avoiding a pair of commuting pure-rotation cells. Every cell remains cubic, giving HX=HZ=12. Compile verified unit actions, exactly exclude constituent dependencies through eight, search through ten with explicit completion flags, and repair witness-heavy cells without inheriting invalid witnesses. Reward a constituent floor of twelve only after completed through-ten exclusion. Verify complete ranks, embeddings, stabilizer-aware logical witnesses and CSS commutation. Reject repeated uncoloured Tanner fingerprints and the five measured parental signatures; the evaluator applies the complete archive. Constituent exclusions and bounded full-code misses are not quantum-distance certificates.")
FAMILY_NAME = "r040_dic14_13x13_onepure_cubic_explicit10_repair_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 56

def custom_ops(q):
    # Same dicyclic construction rule, now Dic14.
    # x^28=1, y^2=x^14, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic14 requires order 56")
    m = q // 4
    rotation = 2 * m
    els = [(a, b) for b in range(2) for a in range(rotation)]

    def red(g):
        a, b = g
        return ((a + m * (b // 2)) % rotation, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + m*b*d) % rotation,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % rotation, 0) if b == 0 else ((a + m) % rotation, 1)

    def idx(g):
        a, b = red(g)
        return rotation*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)

from functools import lru_cache


def _element_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid element order")


# Mixed triples are distinguished by their rotation-pair order.
# None denotes one broad pure-rotation triple, not a third mixed model.
_TRIPLE_MODELS = (
    (12, 12, 12),
    (6, 6, 6),
    (12, 6, 3),
    (None, 12, 12),
    (None, 6, 6),
)


def _support_recipe(dense):
    # Four or three singleton/singleton columns remain, respectively.
    columns = ([(2, 1), (1, 2), (1, 2)] + [(1, 1)] * 4
               if not dense else
               [(2, 1)] * 2 + [(1, 2)] * 2 + [(1, 1)] * 3)
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in columns] for i in range(2)]
    return sa, sb


def _sidon_relatives(rng, G, count):
    """Distinct ordered differences exclude four-cycles among mono columns."""
    for _ in range(64):
        selected = [G.e]
        differences = set()
        pool = list(G.elements)
        rng.shuffle(pool)
        for g in pool:
            if g in selected:
                continue
            added = []
            for h in selected:
                added.extend((G.mul(g, G.inv(h)),
                              G.mul(h, G.inv(g))))
            if len(set(added)) != len(added):
                continue
            if any(d in differences for d in added):
                continue
            selected.append(g)
            differences.update(added)
            if len(selected) == count:
                rng.shuffle(selected)
                return selected
    # A fallback is screened on the complete matrices, not silently accepted.
    return rng.sample(G.elements, count)


def _draw_triple(rng, G, pair_order, order_pools):
    if pair_order is None:
        rotations = [g for g in G.elements if g[1] == 0]
        for _ in range(128):
            cell = rng.sample(rotations, 3)
            orders = [
                _element_order(G, G.mul(G.inv(cell[i]), cell[j]))
                for i in range(3) for j in range(i)
            ]
            if 2 not in orders and 12 in orders:
                break
        else:
            cell = [(0, 0), (1, 0), (3, 0)]
    else:
        a = rng.randrange(12)
        delta = rng.choice(order_pools[pair_order])
        cell = [(a, 0), G.mul((a, 0), delta),
                (rng.randrange(12), 1)]
    # Independent cell gauges diversify sampling but are not counted as designs.
    gauge = G.pick(rng)
    return [G.mul(gauge, g) for g in cell]


def _draw_grids(rng, G, sb, triple_model, double_order, order_pools):
    A = [[_draw_triple(rng, G, kind, order_pools)
          for kind in triple_model]]
    B = [[None] * 7 for _ in range(2)]
    mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
    relatives = _sidon_relatives(rng, G, len(mono))
    for j, relative in zip(mono, relatives):
        top = G.pick(rng)
        # In the verified B action, top * bottom^-1 is the edge relative.
        bottom = G.mul(G.inv(relative), top)
        B[0][j], B[1][j] = [top], [bottom]
    for i in range(2):
        for j in range(7):
            if B[i][j] is not None:
                continue
            g = G.pick(rng)
            B[i][j] = ([g] if sb[i][j] == 1 else
                       [g, G.mul(g, rng.choice(order_pools[double_order]))])
    return A, B


def _syndrome_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _component_columns(hx, hz, q):
    """Extract complete component actions from the verified full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i*cb)*q+r]
            ar.append(sum(((row >> (j*cb*q)) & mask) << (j*q)
                          for j in range(ca)))
    br = [row & ((1 << (cb*q)) - 1) for row in hz[:rb*q]]
    return (_syndrome_columns(ar, ca*q),
            _syndrome_columns(br, cb*q))


def _component_cycles(columns):
    overlaps = {}
    cycles = 0
    for syndrome in columns:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(checks):
            for b in checks[:i]:
                key = (b, a)
                old = overlaps.get(key, 0)
                cycles += old
                overlaps[key] = old + 1
    return cycles


def _kernel_reading(rng, columns, trials):
    """Actual component-kernel witnesses; returned minima are upper bounds."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        short = sorted(words, key=int.bit_count)[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 3:
            break
    return best


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


# Historical Dic6 label generators above are inactive on the new bed.
_D56_A_CARDS = (
    (28, 28, 28), (28, 28, 14), (28, 14, 7),
    (14, 14, 7), (28, 7, 7),
)
_D56_B_CARDS = (
    (28, 28, 28), (28, 28, 14),
    (28, 14, 7), (14, 7, 7),
)


def _d56_basis(rows):
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


def _d56_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _d56_clean4(cols):
    """Exact kernel exclusion through four; cubic parity excludes odd words."""
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


def _d56_short_profile(cols, q):
    """Exact constituent exclusions through six/eight after clean4.

    Anchors cover every constituent block by its regular-action symmetry.
    No such transitivity is assumed for the full quantum code.
    Returns an exclusion floor and an actual witness, when found.
    """
    bits = [1 << i for i in range(len(cols))]
    tails = {0: 0}
    incidence = [[] for _ in range(q)]
    for i, value in enumerate(cols):
        tails[value] = bits[i]
        remaining = value
        while remaining:
            bit = remaining & -remaining
            incidence[bit.bit_length()-1].append(i)
            remaining ^= bit
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            assert syndrome not in tails
            tails[syndrome] = bits[i] | bits[j]

    def decode(value, limit, forbidden):
        def visit(syndrome, remaining, word):
            if not syndrome:
                return word
            if syndrome.bit_count() > 3*remaining:
                return None
            if remaining <= 2:
                tail = tails.get(syndrome)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
                    return word | tail
                return None
            check = (syndrome & -syndrome).bit_length()-1
            choices = sorted(
                incidence[check],
                key=lambda j: -(cols[j] & syndrome).bit_count())
            for j in choices:
                if bits[j] & (word | forbidden):
                    continue
                hit = visit(syndrome ^ cols[j], remaining-1,
                            word | bits[j])
                if hit is not None:
                    return hit
            return None
        return visit(value, limit, 0)

    for bound in (6, 8):
        for anchor in range(0, len(cols), q):
            hit = decode(cols[anchor], bound-1, bits[anchor])
            if hit is not None:
                witness = hit | bits[anchor]
                return witness.bit_count(), witness
    return 10, 0


def _d56_probe(cols, local, trials, seeds=(), stabilizers=None):
    """Actual kernel/logical witnesses, with retained information-set memory."""
    span = None if stabilizers is None else _d56_basis(stabilizers)
    memory = sorted({w for w in seeds if w},
                    key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(len(cols)))
    best = len(cols) + 1

    def observe(word):
        nonlocal best
        if (word and word.bit_count() < best
                and (span is None or _d56_outside(word, span))):
            best = word.bit_count()

    for word in memory:
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        assert syndrome == 0
        observe(word)

    for _ in range(trials):
        local.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                observe(word)

        directions = sorted(set(dependencies + memory),
                            key=lambda w: (w.bit_count(), w))[:32]
        discoveries = set(memory)
        for seed in sorted(dependencies, key=int.bit_count)[:24]:
            word = seed
            for sweep in range(2):
                previous = word.bit_count()
                for direction in directions:
                    trial = word ^ direction
                    if trial and trial.bit_count() < word.bit_count():
                        word = trial
                if word.bit_count() == previous:
                    break
            observe(word)
            discoveries.add(word)

        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    observe(trial)
                    discoveries.add(trial)

        if stabilizers is not None:
            logicals = [w for w in discoveries
                        if _d56_outside(w, span)]
            for seed in sorted(logicals, key=int.bit_count)[:4]:
                word = seed
                for sweep in range(2):
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers))[:12]
                    chosen, weight = word, word.bit_count()
                    for i, (_, row) in enumerate(nearby):
                        trial = word ^ row
                        if trial.bit_count() < weight:
                            chosen, weight = trial, trial.bit_count()
                        for _, other in nearby[:i]:
                            trial = word ^ row ^ other
                            if trial.bit_count() < weight:
                                chosen, weight = trial, trial.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                observe(word)
                discoveries.add(word)

        memory = sorted((w for w in discoveries if w),
                        key=lambda w: (w.bit_count(), w))[:32]
        if span is not None:
            # Preserve logical directions separately from short stabilizers.
            logicals = sorted(
                (w for w in discoveries if w and _d56_outside(w, span)),
                key=lambda w: (w.bit_count(), w))[:8]
            memory = sorted(set(memory + logicals),
                            key=lambda w: (w.bit_count(), w))
    return best, tuple(memory)


def _d56_logical6(checks, stabilizers, n):
    """Complete logical existence through six after full kernel clean4.

    Every qubit is anchored. Singleton/pair tails are unique because the
    preceding full commuting-four exclusion passed. There is no cutoff.
    """
    cols = _syndrome_columns(checks, n)
    span = _d56_basis(stabilizers)
    bits = [1 << i for i in range(n)]
    tails = {0: 0}
    incidence = [[] for _ in checks]
    degree = max(c.bit_count() for c in cols)
    for i, value in enumerate(cols):
        assert value not in tails
        tails[value] = bits[i]
        remaining = value
        while remaining:
            bit = remaining & -remaining
            incidence[bit.bit_length()-1].append(i)
            remaining ^= bit
    for i, value in enumerate(cols):
        for j in range(i):
            syndrome = value ^ cols[j]
            assert syndrome not in tails
            tails[syndrome] = bits[i] | bits[j]

    def visit(value, remaining, word):
        if not value:
            return _d56_outside(word, span)
        if value.bit_count() > degree*remaining:
            return False
        if remaining <= 2:
            tail = tails.get(value)
            return (tail is not None and tail.bit_count() <= remaining
                    and _d56_outside(word ^ tail, span))
        check = (value & -value).bit_length()-1
        for j in incidence[check]:
            if word & bits[j]:
                continue
            if visit(value ^ cols[j], remaining-1, word ^ bits[j]):
                return True
        return False

    return any(visit(value, 5, bits[i])
               for i, value in enumerate(cols))


def _d56_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; inequality certifies nonisomorphism."""
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
    keys = [
        (weights[i], tuple(sorted(Counter(
            (weights[j], size,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, size in neighbors.items()).items())))
        for i, neighbors in enumerate(overlaps)]
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
            for i, neighbors in enumerate(overlaps)]
    degrees = _syndrome_columns(rows, n)
    return (n, tuple(history),
            tuple(sorted(Counter(c.bit_count() for c in degrees).items())))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Twenty independently optimized new-shape cards; no orbit fallback."""
    assert G.q == 56 and A_SHAPE == B_SHAPE == (1, 3)
    q, mask = G.q, (1 << G.q)-1
    counts = [[3, 3, 3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    # Compile both actions exclusively from verified single-term lifts.
    atlas = {}
    zero = [[[], [], []]]
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            ac, bc = _component_columns(hx, hz, q)
            block = tuple((ac if side == "A" else bc)[:q])
            assert all(value.bit_count() == 1 for value in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_grid(side, grid):
        cols = [0] * (3*q)
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for h, value in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+h] ^= value
        return tuple(cols)

    rotation_pools = {
        order: [g for g in G.elements
                if g[1] == 0 and _element_order(G, g) == order]
        for order in (7, 14, 28)}

    def draw(local, side, recipe):
        for restart in range(24):
            used, row = set(), []
            for order in recipe:
                accepted = None
                for attempt in range(96):
                    delta = local.choice(rotation_pools[order])
                    reflection = (local.randrange(28), 1)
                    cell = (G.e, delta, reflection)
                    differences = {
                        (G.mul(g, G.inv(h)) if side == "A"
                         else G.mul(G.inv(g), h))
                        for g in cell for h in cell if g != h}
                    if len(differences) != 6 or differences & used:
                        continue
                    used.update(differences)
                    accepted = cell
                    break
                if accepted is None:
                    break
                row.append(accepted)
            if len(row) == 3:
                return (tuple(row),)
        return None

    def component_pool(local, side, recipe, batch):
        raw, seen = [], set()
        for attempt in range(160 + 80*batch):
            grid = draw(local, side, recipe)
            if grid is None or grid in seen:
                continue
            seen.add(grid)
            cols = compile_grid(side, grid)
            assert all(value.bit_count() == 3 for value in cols)
            if not _d56_clean4(cols):
                continue
            cycles = _component_cycles(cols)
            if cycles:
                continue
            rank = gf2_rank(cols)
            reading, words = _d56_probe(cols, local, 2)
            if reading <= 6:
                continue
            raw.append(dict(
                grid=grid, cols=cols, rank=rank,
                d=reading, words=words, attempt=attempt))
            raw.sort(key=lambda a: (-min(a["d"], 14), a["attempt"]))
            del raw[10:]

        pool = []
        for item in raw:
            floor, witness = _d56_short_profile(item["cols"], q)
            if witness and witness.bit_count() <= 6:
                continue
            seeds = item["words"] + ((witness,) if witness else ())
            reading, words = _d56_probe(
                item["cols"], local, 32, seeds)
            item.update(
                d=min(item["d"], reading), words=words,
                clean8=not witness, floor=floor)
            pool.append(item)
        pool.sort(key=lambda a: (
            a["clean8"], min(a["d"], 12), -a["rank"],
            -sum(w.bit_count() == a["d"] for w in a["words"])),
            reverse=True)
        return pool[:4]

    def embeddings(words, side):
        result = []
        for seed in sorted(set(words), key=int.bit_count)[:12]:
            for fixed in range(3):
                if side == "A":
                    result.append(sum(
                        ((seed >> (j*q)) & mask) << ((j*3+fixed)*q)
                        for j in range(3)))
                else:
                    result.append(seed << (fixed*3*q))
        return result

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, model = divmod(slot % 20, 4)
        ak = _D56_A_CARDS[architecture]
        bk = _D56_B_CARDS[model]
        local = random.Random(
            BASE_SEED*100003 + 56040149 + 104729*slot)
        winner = None

        for batch in range(2):
            ap = component_pool(local, "A", ak, batch)
            bp = component_pool(local, "B", bk, batch)
            pairs = sorted(
                ((int(a["clean8"]) + int(b["clean8"]),
                  min(a["d"], b["d"], 12),
                  a["d"] + b["d"], ai, bi, a, b)
                 for ai, a in enumerate(ap)
                 for bi, b in enumerate(bp)),
                key=lambda entry: entry[:5], reverse=True)
            finalists = []

            for _, _, _, _, _, a, b in pairs:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (375*n <= 1000*k < 425*n):
                    continue
                ac, bc = _component_columns(hx, hz, q)
                assert tuple(ac) == a["cols"]
                assert tuple(bc) == b["cols"]

                sx = _small_kernel_stats(hx, n)
                if sx is None or sx[0]:
                    continue
                sz = _small_kernel_stats(hz, n)
                if sz is None or sz[0]:
                    continue
                fingerprint = _d56_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                if _d56_logical6(hx, hz, n):
                    continue
                if _d56_logical6(hz, hx, n):
                    continue

                # Embeddings are verified, not assumed to be logicals.
                xc, zc = _syndrome_columns(hz, n), _syndrome_columns(hx, n)
                xseeds = [
                    w for w in embeddings(b["words"], "B")
                    if not any((w & row).bit_count() & 1 for row in hz)]
                zseeds = [
                    w for w in embeddings(a["words"], "A")
                    if not any((w & row).bit_count() & 1 for row in hx)]
                dx, xwords = _d56_probe(xc, local, 6, xseeds, hx)
                dz, zwords = _d56_probe(zc, local, 6, zseeds, hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    xc=xc, zc=zc, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    a=a, b=b, fingerprint=fingerprint))
                if len(finalists) >= 3:
                    break

            # Equal fresh budgets; old true witnesses can only lower readings.
            for item in finalists:
                dx, _ = _d56_probe(
                    item["xc"], local, 20, item["xwords"], item["hx"])
                dz, _ = _d56_probe(
                    item["zc"], local, 20, item["zwords"], item["hz"])
                reading = min(
                    item["dx"], item["dz"], dx, dz,
                    item["a"]["d"], item["b"]["d"], 12)
                objective = (
                    item["k"]*reading*reading/item["n"], reading,
                    int(item["a"]["clean8"]) + int(item["b"]["clean8"]),
                    item["k"])
                if winner is None or objective > winner[0]:
                    winner = (objective, item)
            if winner is not None and winner[0][1] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        objective, item = winner
        assert item["n"] == 560 and item["k"] >= 224
        assert css_ok(item["hx"], item["hz"])
        assert max(row.bit_count()
                   for row in item["hx"] + item["hz"]) == 12
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (item["A"], item["B"])
                   for row in grid for cell in row)
        emitted.add(item["fingerprint"])
        bank.append((item["A"], item["B"]))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five fresh Dic14 cubic designs passed")
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure replacement bank; the historical all-mixed bank is inactive."""
    if idx >= 0:
        bank = NEW_CTX(None, G, -1)
        if idx >= len(bank) or bank[idx] is None:
            raise ValueError("no fresh one-pure Dic14 assembly for this card")
        A, B, selection = bank[idx]
        return {"A": A, "B": B, "selection": selection}

    from math import gcd

    assert G.q == 56 and A_SHAPE == B_SHAPE == (1, 3)
    q, width = G.q, 3*G.q
    mask = (1 << q)-1
    counts = [[3, 3, 3]]
    assert _grid_weights(counts, counts) == 12
    assert _no_dead_lines(counts, counts)

    # Every parity atom comes from the scaffold's verified construction.
    atlas = {}
    zero = [[[], [], []]]
    for side in ("A", "B"):
        atoms = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            ac, bc = _component_columns(hx, hz, q)
            atom = tuple((ac if side == "A" else bc)[:q])
            assert all(v.bit_count() == 1 for v in atom)
            atoms.append(atom)
        atlas[side] = tuple(atoms)

    def compile_component(side, grid):
        cols = [0]*width
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for h, value in enumerate(atlas[side][G.idx(g)]):
                    cols[j*q+h] ^= value
        assert all(v.bit_count() == 3 for v in cols)
        return tuple(cols)

    def syndrome(cols, word):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length()-1]
            word ^= bit
        return value

    units = tuple(a for a in range(1, 28) if gcd(a, 28) == 1)
    orders = {
        order: tuple(g for g in G.elements
                     if g[1] == 0 and _element_order(G, g) == order)
        for order in (4, 7, 14, 28)}

    # Different relative triangles, not translations of saved fillings.
    # All five ratios are non-AP in C28.
    a_cards = (
        (3, (28, 28)),
        (5, (28, 14)),
        (9, (28, 7)),
        (11, (14, 7)),
        (13, (28, 4)),
    )
    b_cards = (
        (5, (28, 14)),
        (9, (28, 7)),
        (11, (14, 7)),
        (13, (28, 4)),
    )

    def draw(local, side, card):
        ratio, mixed_orders = card
        for restart in range(20):
            step = local.choice(units)
            pure = (G.e, (step, 0), (ratio*step % 28, 0))
            aa = [g[0] for g in pure]
            assert len(set(pure)) == 3
            assert not any(
                (2*aa[t]-aa[(t+1)%3]-aa[(t+2)%3]) % 28 == 0
                for t in range(3))

            def differences(cell):
                return {
                    (G.mul(x, G.inv(y)) if side == "A"
                     else G.mul(G.inv(x), y))
                    for x in cell for y in cell if x != y}

            used = differences(pure)
            if len(used) != 6:
                continue
            row = [pure]
            for order in mixed_orders:
                selected = None
                for attempt in range(96):
                    delta = local.choice(orders[order])
                    reflected = (local.randrange(28), 1)
                    cell = (G.e, delta, reflected)
                    ds = differences(cell)
                    if len(ds) != 6 or ds & used:
                        continue
                    selected = cell
                    used.update(ds)
                    break
                if selected is None:
                    break
                row.append(selected)
            if len(row) == 3:
                return (tuple(row),)
        return None

    def make_dependency_search(cols):
        """Anchored constituent search with an explicit completion flag.

        clean4 has passed, so singleton/pair tails are unique.
        Regular-action anchors cover constituent support orbits only.
        An exhausted no-hit search excludes the requested constituent
        weights. A node cutoff is never interpreted as an exclusion.
        """
        bits = [1 << j for j in range(width)]
        rows = [0]*q
        tails = {0: 0}
        for j, value in enumerate(cols):
            assert value not in tails
            tails[value] = bits[j]
            pending = value
            while pending:
                bit = pending & -pending
                rows[bit.bit_length()-1] |= bits[j]
                pending ^= bit
        for j, value in enumerate(cols):
            for h in range(j):
                pair = value ^ cols[h]
                assert pair not in tails
                tails[pair] = bits[j] | bits[h]

        def search(bound, node_cap=None):
            nodes, stopped, found = 0, False, 0

            def visit(value, remaining, banned, word):
                nonlocal nodes, stopped, found
                if found or stopped:
                    return
                if node_cap is not None and nodes >= node_cap:
                    stopped = True
                    return
                nodes += 1
                if not value:
                    found = word
                    return
                # Every column has odd weight three.
                if (value.bit_count() ^ remaining) & 1:
                    remaining -= 1
                if remaining <= 0 or value.bit_count() > 3*remaining:
                    return
                if remaining <= 2:
                    tail = tails.get(value)
                    if (tail is not None and tail.bit_count() <= remaining
                            and not (tail & banned)):
                        found = word | tail
                    return

                pending, choices, least = value, 0, width+1
                while pending:
                    bit = pending & -pending
                    options = rows[bit.bit_length()-1] & ~banned
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
                    alternatives.append(bit.bit_length()-1)
                alternatives.sort(
                    key=lambda j: (-(cols[j] & value).bit_count(), j))
                excluded = banned
                for j in alternatives:
                    excluded |= bits[j]
                    visit(value ^ cols[j], remaining-1,
                          excluded, word | bits[j])
                    if found or stopped:
                        break

            for anchor in range(0, width, q):
                visit(cols[anchor], bound-1,
                      bits[anchor], bits[anchor])
                if found or stopped:
                    break
            if found:
                assert found.bit_count() <= bound
                assert syndrome(cols, found) == 0
            return found, not stopped and not found, nodes

        return search

    def component_merit(item):
        # Twelve is rewarded only after completed through-ten exclusion.
        floor = 12 if item["clean10"] else 10
        d = min(item["d"], floor)
        load = sum(w.bit_count() <= item["d"] for w in item["words"])
        return (d, int(item["clean10"]), -load,
                -item["cycles"], -item["serial"])

    @lru_cache(maxsize=9)
    def component_pool(side, architecture):
        card = (a_cards if side == "A" else b_cards)[architecture]
        local = random.Random(
            BASE_SEED*100003 + 561040217 + 104729*architecture
            + 1000003*int(side == "B"))
        pool, seen, serial = [], set(), 0

        def admit(grid):
            nonlocal serial
            serial += 1
            if grid is None or grid in seen:
                return
            seen.add(grid)
            assert all(len(c) == len(set(c)) == 3 for c in grid[0])
            assert all(g[1] == 0 for g in grid[0][0])
            assert all(any(g[1] for g in c) for c in grid[0][1:])
            cols = compile_component(side, grid)
            if not _d56_clean4(cols):
                return

            reading, words = _d56_probe(cols, local, 2)
            if reading <= 8:
                return
            search = make_dependency_search(cols)
            eight, complete8, _ = search(8)
            if eight:
                return
            assert complete8

            ten, complete10, nodes = search(10, 50000)
            seeds = words + ((ten,) if ten else ())
            reading, words = _d56_probe(cols, local, 4, seeds)
            assert reading >= 10
            if complete10:
                assert reading >= 12
            item = dict(
                grid=grid, cols=cols, rank=gf2_rank(cols),
                d=reading, words=words, clean10=complete10,
                cycles=_component_cycles(cols), serial=serial,
                nodes10=nodes)
            pool.append(item)
            pool.sort(key=component_merit, reverse=True)
            del pool[10:]

        for batch in range(2):
            for attempt in range(256 + 128*batch):
                admit(draw(local, side, card))
            if len(pool) >= 8 and sum(p["clean10"] for p in pool) >= 3:
                break

        # Replace real cells. A changed matrix receives fresh syndromes,
        # dependency tests and witness memory, never unchecked old words.
        for step in range(64):
            fresh = draw(local, side, card)
            if fresh is None:
                continue
            if not pool:
                admit(fresh)
                continue
            parent = local.choice(pool[:min(4, len(pool))])
            pressure = [
                1 + sum(((w >> (j*q)) & mask).bit_count()
                        for w in parent["words"][:12])
                for j in range(3)]
            j = local.choices(range(3), weights=pressure, k=1)[0]
            row = list(parent["grid"][0])
            row[j] = fresh[0][j]
            if step % 7 == 6:
                other = (j+1+local.randrange(2)) % 3
                row[other] = fresh[0][other]
            admit((tuple(row),))

        # Equal independent budgets; previously found minima are retained.
        for item in pool:
            reading, words = _d56_probe(
                item["cols"], local, 24, item["words"])
            item["d"] = min(item["d"], reading)
            item["words"] = words
            if item["clean10"]:
                assert item["d"] >= 12
        pool.sort(key=component_merit, reverse=True)
        leaders = pool[:5]
        for item in leaders:
            # Deepen incomplete ten searches rather than reward their misses.
            if not item["clean10"] and item["d"] > 10:
                ten, complete, nodes = make_dependency_search(
                    item["cols"])(10, 300000)
                seeds = item["words"] + ((ten,) if ten else ())
                reading, words = _d56_probe(
                    item["cols"], local, 32, seeds)
                item["d"] = min(item["d"], reading)
                item["words"] = words
                item["clean10"] = complete
                item["nodes10"] = nodes
            else:
                reading, words = _d56_probe(
                    item["cols"], local, 32, item["words"])
                item["d"] = min(item["d"], reading)
                item["words"] = words
            if item["clean10"]:
                assert item["d"] >= 12
        leaders.sort(key=component_merit, reverse=True)
        return tuple(leaders[:4])

    def embeddings(words, side):
        result = []
        for seed in sorted(set(words), key=int.bit_count)[:12]:
            for fixed in range(3):
                if side == "A":
                    result.append(sum(
                        ((seed >> (j*q)) & mask) << ((3*j+fixed)*q)
                        for j in range(3)))
                else:
                    result.append(seed << (fixed*width))
        return tuple(result)

    # Negative archive only: the five measured current-parent codes.
    # Their cells are never candidates, repair templates or emitted outputs.
    parental = (
        (27,27,5,25,19,9,5,18,11,16,27,20),
        (13,17,25,13,5,0,25,25,2,4,20,27),
        (1,22,11,24,6,25,9,12,5,4,6,14),
        (25,26,1,17,18,2,17,8,6,15,16,14),
        (3,7,26,14,4,27,13,1,1,8,17,3),
    )
    emitted, bank = set(), []
    for labels in parental:
        grids = []
        for side_index in range(2):
            row = []
            for j in range(3):
                t = 6*side_index + 2*j
                row.append((G.e, (labels[t], 0), (labels[t+1], 1)))
            grids.append((tuple(row),))
        hx, hz, n = lifted_matrices(G, *grids)
        emitted.add(_d56_fingerprint(hx, hz, n))

    def joint_merit(item):
        af = 12 if item["a"]["clean10"] else 10
        bf = 12 if item["b"]["clean10"] else 10
        d = min(item["a"]["d"], item["b"]["d"], af, bf,
                item.get("dx", 561), item.get("dz", 561))
        return (
            item["k"]*d*d/item["n"], d,
            int(item["a"]["clean10"]) + int(item["b"]["clean10"]),
            -item["a"]["cycles"]-item["b"]["cycles"])

    for slot in range(N_SLOTS):
        architecture, profile = divmod(slot % 20, 4)
        ap = component_pool("A", architecture)
        bp = component_pool("B", profile)
        local = random.Random(
            BASE_SEED*100003 + 561040331 + 104729*slot)
        proposals = []
        for a in ap:
            for b in bp:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 560 or not (375*n <= 1000*k < 425*n):
                    continue
                ac, bc = _component_columns(hx, hz, q)
                assert tuple(ac) == a["cols"]
                assert tuple(bc) == b["cols"]
                proposals.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
        proposals.sort(key=joint_merit, reverse=True)

        finalists = []
        for item in proposals:
            hx, hz, n = item["hx"], item["hz"], item["n"]
            fingerprint = _d56_fingerprint(hx, hz, n)
            if fingerprint in emitted:
                continue
            xc, zc = _syndrome_columns(hz, n), _syndrome_columns(hx, n)
            if not (_d56_clean4(xc) and _d56_clean4(zc)):
                continue

            xseeds = embeddings(item["b"]["words"], "B")
            zseeds = embeddings(item["a"]["words"], "A")
            assert all(syndrome(xc, w) == 0 for w in xseeds)
            assert all(syndrome(zc, w) == 0 for w in zseeds)
            dx, wx = _d56_probe(xc, local, 8, xseeds, hx)
            dz, wz = _d56_probe(zc, local, 8, zseeds, hz)
            if dx > n or dz > n:
                continue
            item.update(
                xc=xc, zc=zc, dx=dx, dz=dz, wx=wx, wz=wz,
                fingerprint=fingerprint)
            finalists.append(item)
            if len(finalists) == 4:
                break

        for item in finalists:
            dx, _ = _d56_probe(
                item["xc"], local, 20, item["wx"], item["hx"])
            dz, _ = _d56_probe(
                item["zc"], local, 20, item["wz"], item["hz"])
            item["dx"] = min(item["dx"], dx)
            item["dz"] = min(item["dz"], dz)

        finalists = [p for p in finalists if joint_merit(p)[1] >= 10]
        if not finalists:
            bank.append(None)
            continue
        winner = max(finalists, key=joint_merit)
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert winner["n"] == 560 and winner["k"] >= 224
        assert max(r.bit_count()
                   for r in winner["hx"] + winner["hz"]) == 12
        assert all(len(c) == len(set(c)) == 3
                   for grid in (A, B) for row in grid for c in row)
        assert all(
            sum(all(g[1] == 0 for g in c) for c in grid[0]) == 1
            for grid in (A, B))
        emitted.add(winner["fingerprint"])
        bank.append((A, B, joint_merit(winner)))

    # Exhausted cards are voided, never refilled with another card's code.
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the verified one-pure cubic cell without subsequent relabeling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free rotation binomials plus one trinomial on each D34 (1,4) component raise measured distance above four; all five measured parent codes had d30s=4 and sector-1 minimum logicals.
# REFUTED: Packed all-mixed Dic14 cubic constituents with through-eight preference and full-logical-six screening raise five-code fitness above forty; all five measured current-parent leaders instead retained k=224, d30s=10 and sector-one minima.
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
