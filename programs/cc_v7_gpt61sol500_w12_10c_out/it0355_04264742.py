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
CONCEPT = ("Metacyclic C28:C1, action 1, order 28, shapes (2,3)x(3,5): more strongly checked B components with all-even degree-four columns, selected against verified cofactors and subgroup-invariant witnesses, should exceed parent five-code fitness 43.8378 in cell 0.150; measured evaluation can refute this.")
GROUP = ("Keep GTYPE and the exact metacyclic construction rule, changing only m=28,r=1,s=1 because factor 21 makes order 37 illegal. Order 28 gives n=588 and is the largest cyclic lift the new size factor permits. The C37 parent scored 43.8378; the related order-28 C7:C4 bed scored 36.2, but neither transfers a distance claim to cyclic C28. Repeated characteristic-two factors make verified complete ranks and explicit subgroup witnesses essential.")
SHAPES = ("Target cell 0.150 with A=(2,3), B=(3,5): n=588, kb=28*(3-2)*(5-3)=56 and guaranteed rate 2/21. Components have 84 variables/56 checks and 140 variables/84 checks: B loses eight variables but gains ten checks relative to the parent. Parent limiting X minima had weight 17 or 18 and occupied only sector-one Acol/Bcol blocks across all B columns, often emphasizing Acol2; no listed minimum used Arow/Brow. All-even supports seek useful rank surplus, but exact complete ranks enforce 0.125<=k/n<0.175 rather than assuming augmentation determines k.")
FILLING = ("Twenty tasks cross five sparse all-even incidence architectures with four primitive/mixed-order binomial policies. Every component column contains four terms; A rows total six and B rows at most eight, so HX<=10 and HZ<=12. Pack within-row and cross-row differences softly, require a primitive binomial in each binomial-only column, and exclude involution differences inside quartics. Compile actions only from verified unit lifts. Generalized two-/three-row cofactors, translated witness pairs and subgroup-invariant kernels guide fresh voltage repairs. Independently deepen finalists, gate complete ranks, test actual logical classes and mixed-sector dressings, and reject equal uncoloured Tanner walk signatures. All sampled readings remain upper bounds, never distance certificates.")
FAMILY_NAME = "r015_m28r1s1_23x35_even4_subgroup_cofactor_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Same metacyclic construction rule; factor 21 requires q<=28.
    # C28:C1 with action 1 is cyclic of order 28.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 28")
    m, r, s = 28, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)

from functools import lru_cache

# Every column totals four; every A row totals six.
# B row totals are permutations of (6,6,8) or (4,8,8).
# Consequently every card has HX<=10 and HZ<=12.
# Quartic-only columns and different omitted-row multiplicities change
# actual incidence architecture, not merely group-element labels.
_COUNT_CARDS = (
    (((2, 2, 2), (2, 2, 2)),
     ((0, 0, 2, 2, 2),
      (2, 2, 0, 0, 2),
      (2, 2, 2, 2, 0))),
    (((2, 2, 2), (2, 2, 2)),
     ((0, 0, 0, 2, 2),
      (2, 2, 2, 0, 2),
      (2, 2, 2, 2, 0))),
    (((4, 2, 0), (0, 2, 4)),
     ((0, 0, 2, 2, 2),
      (2, 2, 0, 0, 2),
      (2, 2, 2, 2, 0))),
    (((2, 2, 2), (2, 2, 2)),
     ((4, 0, 0, 2, 2),
      (0, 2, 2, 0, 2),
      (0, 2, 2, 2, 0))),
    (((4, 2, 0), (0, 2, 4)),
     ((4, 0, 2, 2, 0),
      (0, 4, 2, 0, 2),
      (0, 0, 0, 2, 2))),
)


def _extract_components(hx, hz, q):
    """Extract both component actions from the VERIFIED complete lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            ar.append(sum(
                ((source >> (j * cb * q)) & mask) << (j * q)
                for j in range(ca)))
    bm = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bm
          for a in range(rb) for t in range(q)]
    return ar, br


@lru_cache(maxsize=1)
def _component_atlas(G):
    """Single-term actions compiled solely from scaffold-generated lifts."""
    q = G.q
    assert q == 28
    mask = (1 << q) - 1
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    stub_a = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    stub_b = [[[G.e] for _ in range(cb)] for _ in range(rb)]
    atlas = {}
    for side, shape in (("A", A_SHAPE), ("B", B_SHAPE)):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(shape[1])]
                    for _ in range(shape[0])]
            unit[0][0] = [g]
            A, B = ((unit, stub_b) if side == "A"
                    else (stub_a, unit))
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == (ca * cb + ra * rb) * q
            ar, br = _extract_components(hx, hz, q)
            source = ar if side == "A" else br
            rows = [row & mask for row in source[:q]]
            cols = tuple(_component_columns(rows, q))
            assert all(c.bit_count() == 1 for c in cols)
            blocks.append(cols)
        atlas[side] = tuple(blocks)
    return atlas


def _compile_component(G, side, grid):
    """F2-linear superposition of verified single-term contributions."""
    q = G.q
    atlas = _component_atlas(G)[side]
    cols = [0] * (len(grid[0]) * q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                block = atlas[G.idx(g)]
                for h, syndrome in enumerate(block):
                    cols[j * q + h] ^= syndrome << (i * q)
    return cols


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _small_logical_3(checks, stabilizers, width):
    """Exactly detect any non-stabilizer logical of weight one to three."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]

    def in_span(v):
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                return False
            v ^= basis[p]
        return True

    syndromes = _component_columns(checks, width)
    singles = {}
    for i, syndrome in enumerate(syndromes):
        v = 1 << i
        if syndrome == 0 and not in_span(v):
            return True
        if syndrome in singles:
            if not in_span(v ^ singles[syndrome]):
                return True
        else:
            singles[syndrome] = v

    # After the singleton test, equal-syndrome singleton representatives
    # differ only by stabilizers, so one representative per syndrome suffices.
    for i in range(width):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            v = vi | (1 << j)
            if syndrome == 0 and not in_span(v):
                return True
            if syndrome in singles:
                if not in_span(v ^ singles[syndrome]):
                    return True
    return False


def _row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _outside_span(v, basis):
    while v:
        p = v.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        v ^= old
    return False


def _kernel_le4(cols):
    """Exact classical kernel test using all singleton/pair syndromes."""
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        support = 1 << i
        old = seen.get(syndrome)
        if old is not None:
            return True
        seen[syndrome] = support
    for i, syndrome in enumerate(cols):
        for j in range(i):
            s = syndrome ^ cols[j]
            v = (1 << i) | (1 << j)
            old = seen.get(s)
            if old is not None and old != v:
                return True
            seen[s] = v
    return False


def _kernel_probe(cols, rng, trials, stabilizers=()):
    """Return the weight of a found kernel/logical witness.

    Dependencies come from the supplied verified column syndromes.
    Pair reductions preserve the kernel. Stabilizer reductions preserve
    the logical class, which is explicitly tested before counting it.
    A sentinel means no suitable witness was found, not a distance proof.
    """
    width = len(cols)
    best = width + 1
    basis = _row_basis(stabilizers)
    generators = list(stabilizers)
    order = list(range(width))

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        dependencies = []
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, support)
                    break
                syndrome ^= old[0]
                support ^= old[1]
            if not syndrome:
                dependencies.append(support)

        # A broad dependency gives a fallback genuine logical witness.
        if generators:
            for v in dependencies:
                if _outside_span(v, basis):
                    best = min(best, v.bit_count())
                    break

        words = sorted(dependencies, key=int.bit_count)[:48]
        for _ in range(3):
            changed = False
            for a in range(len(words)):
                v = words[a]
                weight = v.bit_count()
                for b in range(len(words)):
                    if a == b:
                        continue
                    z = v ^ words[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                words[a] = v
            if not changed:
                break

        for v in words:
            if generators and not _outside_span(v, basis):
                continue
            weight = v.bit_count()
            if generators:
                for _ in range(3):
                    changed = False
                    for row in generators:
                        z = v ^ row
                        zw = z.bit_count()
                        if zw < weight:
                            v, weight = z, zw
                            changed = True
                    if not changed:
                        break
            if v:
                best = min(best, weight)
    return best


def _cell_differences(cell, q):
    return {(g[0] - h[0]) % q
            for g in cell for h in cell if g != h}


def _recipe_ok(grid, mode, q):
    """Order constraints on supports, not an alternative matrix lift."""
    from math import gcd
    for row in grid:
        for cell in row:
            if len(cell) != len(set(cell)):
                return False
            differences = _cell_differences(cell, q)
            if q // 2 in differences:
                return False
            if len(cell) == 4 and not any(
                    gcd(d, q) == 1 for d in differences):
                return False
    for j in range(len(grid[0])):
        occupied = [grid[i][j] for i in range(len(grid))
                    if grid[i][j]]
        if occupied and all(len(cell) == 2 for cell in occupied):
            # Do not admit a column whose individual binomials all
            # annihilate a proper-subgroup indicator of weight <=14.
            if not any(
                    gcd((cell[0][0]-cell[1][0]) % q, q) == 1
                    for cell in occupied):
                return False
    return True


def _draw_component(rng, G, counts, mode):
    """Fresh all-even columns with soft check-pair difference packing."""
    from math import gcd
    q = G.q
    height, width = len(counts), len(counts[0])
    steps = {
        d: [v for v in range(1, q) if gcd(v, q) == d]
        for d in (1, 2, 4)}
    best = None
    for restart in range(8):
        used = [set() for _ in counts]
        cross_used = {
            (i, h): set() for i in range(height) for h in range(i)}
        columns, total = [None]*width, 0
        order = list(range(width))
        rng.shuffle(order)
        order.sort(key=lambda j: -max(row[j] for row in counts))
        for j in order:
            occupied = [i for i in range(height) if counts[i][j]]
            primitive = rng.choice(occupied)
            selected = None
            for attempt in range(24):
                values = []
                for i in range(height):
                    size = counts[i][j]
                    if not size:
                        vals = []
                    elif size == 2:
                        divisor = (1 if mode < 2 or i == primitive
                                   else 2 if mode == 2 else 4)
                        base = rng.randrange(q)
                        delta = rng.choice(steps[divisor])
                        vals = [base, (base+delta) % q]
                    else:
                        for _ in range(128):
                            vals = rng.sample(range(q), size)
                            ds = {(a-b) % q for a in vals
                                  for b in vals if a != b}
                            if (q//2 not in ds and any(
                                    gcd(d, q) == 1 for d in ds)):
                                break
                        else:
                            raise ValueError("primitive quartic recipe exhausted")
                    values.append(vals)
                anchor = values[primitive][0]
                values = [sorted((v-anchor) % q for v in vals)
                          for vals in values]
                local, crossing, cost = [], {}, 0
                for i, vals in enumerate(values):
                    differences = [(a-b) % q for a in vals
                                   for b in vals if a != b]
                    ds = set(differences)
                    cost += len(differences)-len(ds)
                    cost += (3 if mode == 1 else 2)*len(ds & used[i])
                    local.append(ds)
                    for h in range(i):
                        ds_cross = [(a-b) % q for a in vals
                                    for b in values[h]]
                        cs = set(ds_cross)
                        cost += len(ds_cross)-len(cs)
                        cost += 2*len(cs & cross_used[i, h])
                        crossing[i, h] = cs
                if selected is None or cost < selected[0]:
                    selected = (cost, values, local, crossing)
                if cost == 0:
                    break
            cost, values, local, crossing = selected
            columns[j] = values
            total += cost
            for i in range(height):
                used[i].update(local[i])
            for pair, ds in crossing.items():
                cross_used[pair].update(ds)
        grid = [[[(v, 0) for v in columns[j][i]]
                 for j in range(width)] for i in range(height)]
        assert _recipe_ok(grid, mode, q)
        if best is None or total < best[0]:
            best = (total, grid)
        if total == 0:
            break
    # Packing cost is a selection heuristic, never a validity repair.
    assert all(len(cell) == counts[i][j]
               for i, row in enumerate(best[1])
               for j, cell in enumerate(row))
    return best[1]


def _syndrome_of(word, cols):
    syndrome = 0
    while word:
        bit = word & -word
        syndrome ^= cols[bit.bit_length() - 1]
        word ^= bit
    return syndrome


def _block_apply(cols, q, row, column, word):
    """Apply one block extracted from VERIFIED component columns."""
    mask = (1 << q) - 1
    result = 0
    while word:
        bit = word & -word
        h = bit.bit_length() - 1
        result ^= (cols[column * q + h] >> (row * q)) & mask
        word ^= bit
    return result


def _component_shift(word, q, blocks, shift):
    """Translate test supports in cyclic coordinates, never parity matrices."""
    mask = (1 << q) - 1
    result = 0
    for j in range(blocks):
        part = (word >> (j * q)) & mask
        if shift:
            part = ((part << shift) | (part >> (q - shift))) & mask
        result |= part << (j * q)
    return result


def _cofactor_words(cols, q, height):
    """Kernel words from determinants of VERIFIED commuting operators.

    This handles both the two-row A and three-row B component.
    Determinant signs disappear in characteristic two. Operators are
    applied through scaffold-derived columns, never hand-written lifts.
    """
    from itertools import combinations, permutations
    blocks = len(cols)//q
    originals = []
    for chosen in combinations(range(blocks), height+1):
        word = 0
        for column in chosen:
            minor_columns = [j for j in chosen if j != column]
            minor = 0
            for ordering in permutations(minor_columns):
                value = 1
                for row, j in enumerate(ordering):
                    value = _block_apply(cols, q, row, j, value)
                minor ^= value
            word |= minor << (column*q)
        if word:
            assert _syndrome_of(word, cols) == 0
            originals.append(word)

    discoveries = set(originals)
    best = min((w.bit_count() for w in originals),
               default=len(cols)+1)
    for i, first in enumerate(originals):
        for second in originals[:i+1]:
            shifts = (range(1, q//2+1)
                      if first == second else range(q))
            for shift in shifts:
                combined = first ^ _component_shift(
                    second, q, blocks, shift)
                if combined:
                    weight = combined.bit_count()
                    best = min(best, weight)
                    if weight <= best+4:
                        discoveries.add(combined)
    words = sorted(discoveries, key=lambda w: (w.bit_count(), w))[:32]
    assert all(_syndrome_of(w, cols) == 0 for w in words)
    return words


def _witness_probe(cols, seeds, rng, trials):
    """Retain actual kernel words across independent information sets."""
    width = len(cols)
    memory = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:32]
    best = min((w.bit_count() for w in memory), default=width + 1)
    order = list(range(width))
    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome and word:
                dependencies.append(word)
                best = min(best, word.bit_count())

        words = sorted(set(memory + dependencies),
                       key=lambda w: (w.bit_count(), w))
        for sweep in range(2):
            donors = sorted(set(words + memory), key=int.bit_count)[:32]
            rng.shuffle(donors)
            reduced = []
            for word in words:
                weight = word.bit_count()
                for donor in donors:
                    candidate = word ^ donor
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
                best = min(best, weight)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))

        discoveries = set(words + memory)
        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    best = min(best, candidate.bit_count())
                    discoveries.add(candidate)
        if trials >= 16 and trial % 4 == 0:
            small = short[:8]
            for i, word in enumerate(small):
                for j in range(i):
                    for h in range(j):
                        candidate = word ^ small[j] ^ small[h]
                        if candidate:
                            best = min(best, candidate.bit_count())
                            discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        if best <= 8:
            break
    assert memory
    seeds[:] = memory
    return best


def _subgroup_words(cols, q, rng):
    """Actual kernels in cyclic-subgroup indicator subspaces.

    Physical supports are expanded before weighing. These restricted
    searches can reveal light witnesses, but cannot certify their absence.
    """
    blocks = len(cols)//q
    found = set()
    for size in (4, 7, 14):
        step = q//size
        indicators = [
            sum(1 << (j*q+a+t*step) for t in range(size))
            for j in range(blocks) for a in range(step)]
        projected = [_syndrome_of(w, cols) for w in indicators]
        compact = []
        _witness_probe(projected, compact, rng, 6)
        for coefficient in compact:
            word = 0
            while coefficient:
                bit = coefficient & -coefficient
                word ^= indicators[bit.bit_length()-1]
                coefficient ^= bit
            if word:
                assert _syndrome_of(word, cols) == 0
                found.add(word)
    return sorted(found, key=lambda w: (w.bit_count(), w))[:32]


def _component_pool(rng, G, side, counts, mode, quota):
    """Generalized cofactors, subgroup witnesses, and fresh voltage repairs."""
    q = G.q
    raw, seen = [], set()

    def labels(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

    def merit(item):
        return (item["d"], -item["hits"], -item["cycles"], item["rank"])

    def admit(grid, attempt):
        key = labels(grid)
        if key in seen or not _recipe_ok(grid, mode, q):
            return
        seen.add(key)
        cols = _compile_component(G, side, grid)
        rank = gf2_rank(cols)
        height = len(counts)
        # Preserve several primary-factor rank strata; only the complete
        # scaffold-built code decides whether a surplus fits cell 0.150.
        if rank < height*(q-1)-height-4 or _kernel_le4(cols):
            return
        words = sorted(set(
            _cofactor_words(cols, q, height)
            + _subgroup_words(cols, q, rng)),
            key=lambda w: (w.bit_count(), w))[:32]
        d = min((w.bit_count() for w in words), default=len(cols) + 1)
        if d <= 8:
            return
        item = dict(
            grid=grid, cols=cols, rank=rank, words=words, d=d,
            hits=sum(w.bit_count() == d for w in words),
            cycles=_component_four_cycles(cols), attempt=attempt)
        raw.append(item)
        raw.sort(key=merit, reverse=True)
        del raw[12:]

    for attempt in range(quota):
        admit(_draw_component(rng, G, counts, mode), attempt)

    for item in raw:
        item["d"] = min(item["d"], _witness_probe(
            item["cols"], item["words"], rng, 8))
        item["hits"] = sum(
            w.bit_count() == item["d"] for w in item["words"])
    raw.sort(key=merit, reverse=True)
    elites = raw[:6]

    for step in range(16):
        if not elites:
            break
        parent = rng.choice(elites[:min(3, len(elites))])
        grid = [[list(cell) for cell in row] for row in parent["grid"]]
        mask = (1 << q) - 1
        pressure = [
            1 + sum(((w >> (j * q)) & mask).bit_count()
                    for w in parent["words"][:8])
            for j in range(len(counts[0]))]
        j = rng.choices(range(len(pressure)), weights=pressure, k=1)[0]
        i = rng.randrange(len(counts))
        replacement = _draw_component(rng, G, counts, mode)
        grid[i][j] = replacement[i][j]
        if step % 5 == 4:
            other = (j + 1 + rng.randrange(len(pressure) - 1)) % len(pressure)
            other_row = (i+1+rng.randrange(len(counts)-1)) % len(counts)
            grid[other_row][other] = replacement[other_row][other]
        before = len(seen)
        raw = list(elites)
        admit(grid, quota + step)
        if len(seen) == before:
            continue
        for item in raw:
            if item not in elites:
                item["d"] = min(item["d"], _witness_probe(
                    item["cols"], item["words"], rng, 8))
                item["hits"] = sum(
                    w.bit_count() == item["d"] for w in item["words"])
        raw.sort(key=merit, reverse=True)
        elites = raw[:6]

    finalists = elites[:4]
    for item in finalists:
        # Equal deeper budgets; previously found witnesses remain valid.
        item["d"] = min(item["d"], _witness_probe(
            item["cols"], item["words"], rng, 40))
        item["hits"] = sum(
            w.bit_count() == item["d"] for w in item["words"])
        assert all(_syndrome_of(w, item["cols"]) == 0
                   for w in item["words"])
    finalists.sort(key=merit, reverse=True)
    return finalists[:3]


def _dressed_component_reading(item, q):
    """Verified logical embeddings followed by genuine stabilizer dressings."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    hx, hz = item["hx"], item["hz"]
    mask = (1 << q) - 1
    best = item["n"] + 1
    for side, words, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        span = _row_basis(stabilizers)
        seeds = []
        for seed in sorted(set(words), key=int.bit_count)[:8]:
            for fixed in range(cb if side == "A" else ca):
                if side == "A":
                    word = sum(
                        ((seed >> (j * q)) & mask)
                        << ((j * cb + fixed) * q)
                        for j in range(ca))
                else:
                    word = seed << (fixed * cb * q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _outside_span(word, span):
                    continue
                best = min(best, word.bit_count())
                seeds.append(word)

        for word in sorted(set(seeds), key=int.bit_count)[:6]:
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                nearby = []
                for row in stabilizers:
                    candidate = word ^ row
                    cw = candidate.bit_count()
                    nearby.append((cw, row))
                    if cw < weight:
                        chosen, weight = candidate, cw
                nearby.sort(key=lambda x: x[0])
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
                best = min(best, weight)
    return best


def _walk_signature(rows, width):
    """Permutation-invariant weighted row-intersection graph statistics.

    These are graph fingerprints, not logical-operator counts. Different
    fingerprints exclude Tanner isomorphism; equal ones may be false
    duplicates and are conservatively discarded.
    """
    cols = _component_columns(rows, width)
    adjacency = [{} for _ in rows]
    for syndrome in cols:
        checks = []
        while syndrome:
            low = syndrome & -syndrome
            checks.append(low.bit_length() - 1)
            syndrome ^= low
        for a, i in enumerate(checks):
            for j in checks[:a]:
                adjacency[i][j] = adjacency[i].get(j, 0) + 1
                adjacency[j][i] = adjacency[j].get(i, 0) + 1

    overlap_hist = {}
    edges = 0
    for i, neighbors in enumerate(adjacency):
        for j, overlap in neighbors.items():
            if j < i:
                overlap_hist[overlap] = overlap_hist.get(overlap, 0) + 1
                edges += 1
    overlap_hist[0] = len(rows) * (len(rows) - 1) // 2 - edges

    triangle_walks = 0
    fourth_walks = 0
    for i, neighbors in enumerate(adjacency):
        walks = {}
        for j, ij in neighbors.items():
            for h, jh in adjacency[j].items():
                walks[h] = walks.get(h, 0) + ij * jh
        triangle_walks += sum(
            ij * walks.get(j, 0) for j, ij in neighbors.items())
        fourth_walks += sum(value * value for value in walks.values())

    return (
        tuple(sorted(row.bit_count() for row in rows)),
        tuple(sorted(c.bit_count() for c in cols)),
        tuple(sorted(overlap_hist.items())),
        triangle_walks, fourth_walks,
    )


def _code_signature(hx, hz, n, k):
    # Ignore check colours and omit colour-dependent k. Unequal
    # signatures exclude even Tanner relabelings that intermix X and Z.
    return (n, _walk_signature(list(hx) + list(hz), n))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Five count architectures times four independently optimized policies."""
    assert G.q == 28
    assert A_SHAPE == (2, 3) and B_SHAPE == (3, 5)
    q = G.q
    _component_atlas(G)
    bank, fingerprints = [], set()

    def merit(item):
        d = min(item["da"], item["db"],
                item.get("logical", item["n"] + 1),
                item.get("full", item["n"] + 1))
        return (item["k"] * d * d, d,
                item["da"] + item["db"], -item["cycles"])

    for idx in range(N_SLOTS):
        architecture, mode = divmod(idx % 20, 4)
        sa, sb = _COUNT_CARDS[architecture]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(
            BASE_SEED * 100003 + 28035197 + 104729 * idx)
        chosen = None

        for retry in range(3):
            quota = 112 + 48 * retry + 16 * (idx // 20)
            ap = _component_pool(rng, G, "A", sa, mode, quota)
            bp = _component_pool(rng, G, "B", sb, mode, quota)
            candidates = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (125 * n <= 1000 * k < 175 * n):
                        continue
                    assert n == 588 and k >= 56
                    ar, br = _extract_components(hx, hz, q)
                    assert _component_columns(ar, A_SHAPE[1] * q) == a["cols"]
                    assert _component_columns(br, B_SHAPE[1] * q) == b["cols"]
                    candidates.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        aw=list(a["words"]), bw=list(b["words"]),
                        ac=a["cols"], bc=b["cols"],
                        da=a["d"], db=b["d"],
                        cycles=a["cycles"] + b["cycles"]))

            candidates.sort(key=merit, reverse=True)
            measured = []
            for item in candidates[:4]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if (_small_logical_3(hx, hz, n)
                        or _small_logical_3(hz, hx, n)):
                    continue
                item["da"] = min(item["da"], _witness_probe(
                    item["ac"], item["aw"], rng, 24))
                item["db"] = min(item["db"], _witness_probe(
                    item["bc"], item["bw"], rng, 24))
                item["logical"] = _dressed_component_reading(item, q)
                # These searches cover both sectors and exclude the exact
                # stabilizer span. Their sampling misses certify nothing.
                dx = _kernel_probe(
                    _component_columns(hz, n), rng, 8, stabilizers=hx)
                dz = _kernel_probe(
                    _component_columns(hx, n), rng, 8, stabilizers=hz)
                item["full"] = min(dx, dz)
                measured.append(item)

            measured.sort(key=merit, reverse=True)
            for item in measured:
                A, B = item["A"], item["B"]
                hx, hz, n, k = (
                    item["hx"], item["hz"], item["n"], item["k"])
                fingerprint = _code_signature(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= W_MAX
                actual_a = [[len(c) for c in row] for row in A]
                actual_b = [[len(c) for c in row] for row in B]
                assert _grid_weights(actual_a, actual_b) <= W_MAX
                assert _no_dead_lines(actual_a, actual_b)
                assert all(len(cell) == len(set(cell)) <= 4
                           for grid in (A, B)
                           for row in grid for cell in row)
                assert actual_a == [list(row) for row in sa]
                assert actual_b == [list(row) for row in sb]
                assert 125 * n <= 1000 * k < 175 * n
                fingerprints.add(fingerprint)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row) for row in A),
                    tuple(tuple(tuple(cell) for cell in row) for row in B),
                    k, merit(item))
                break
            if chosen is not None:
                break

        # A failed policy is not replaced by a saved or relabeled code,
        # and does not prevent other policies from being built.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return fresh structural designs for every cell of the new shapes."""
    selected = _design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh cofactor-screened filling in cell 0.150")
    A, B, k, score = selected
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "probe_score": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: Odd triple upgrades and exact exclusion of weight<=3 logicals at Z9, (2,7)x(2,8), remove the distance-four plateau: all five measured parent codes still had d30s=4, with thousands of sector-1 weight-four witnesses.
# REFUTED: Quartic C37:C3 component-through-four exclusion and mixed-sector probing suppress the sector-one bottleneck beyond weight 21; all five measured current codes retained d30s=21 with minimum witnesses entirely in sector 1.
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
