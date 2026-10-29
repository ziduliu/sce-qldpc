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
CONCEPT = ("Cyclic C31, order 31, A=(2,3), B=(2,4): doubleton A columns matched at one degree-five GF32 character, combined with independently optimized B supports, should supply five fresh cell-0.150 codes with measured fitness above 45.4260. Exact ranks and measured distances can refute this.")
GROUP = ("L3 jump from C37 to genuinely different cyclic C31, absent from BEDS at these shapes. ord_31(2)=5 permits a controlled nontrivial five-dimensional component deficit, unlike C37's degree-36 nontrivial factor. The C37 parent measured five [592,83,18] leaders; its score motivates testing rank surplus rather than claiming a group-distance advantage.")
SHAPES = ("Keep (2,3)x(2,4) for cell 0.150: n=496, kb=62 and guaranteed rate 1/8. The size window permits q<=37; the new group fixes q=31. Unequal constituents have 93/62 and 124/62 variables/checks. Parent minimum X witnesses occupied sector-one Acol/Bcol blocks, especially Acol2, while Z readings were 22 and no listed minimum used Arow/Brow. Require exact assembled 0.125<=k/n<0.175.")
FILLING = ("Twenty tasks cross five B count architectures with four distinct C31 trinomial classes and corresponding A gap policies. Every A cell remains doubleton; its two row evaluations agree at a primitive GF32 character. Require verified A rank 55, rejecting additional uncontrolled deficits. B rows sum to eight and columns have at most six terms, so HX<=12 and HZ=12. Soft difference packing avoids exhausted hard-packing cards. Coupled A-column repairs preserve the character constraint; B retains local voltage repairs and larger witness budgets. Verified cofactors, independent kernel probes, exact logical-through-four tests, stabilizer-aware embeddings and uncoloured Tanner signatures govern complete selection. Found readings remain upper bounds, not distance certificates.")
FAMILY_NAME = "r015_C31_23x24_GF32_character5_Bpressure_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 31

def custom_ops(q):
    # Cyclic C31; retain two-coordinate tuples for the voltage helpers.
    if not valid_q(q):
        raise ValueError("C31 requires order 31")
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return g[0] % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)

from functools import lru_cache

# A remains doubleton-only: row totals six, column totals four.
# B row totals are eight in every card. Column maxima are 4,5,4,4,6.
# Thus the corresponding (HX,HZ) caps are (10,12),(11,12),
# (10,12),(10,12),(12,12). Exact lifted ranks decide actual k.
_COUNT_CARDS = (
    (((2, 2, 2), (2, 2, 2)),
     ((3, 3, 1, 1), (1, 1, 3, 3))),
    (((2, 2, 2), (2, 2, 2)),
     ((2, 2, 2, 2), (1, 1, 3, 3))),
    (((2, 2, 2), (2, 2, 2)),
     ((3, 2, 2, 1), (1, 2, 2, 3))),
    (((2, 2, 2), (2, 2, 2)),
     ((4, 2, 1, 1), (0, 2, 3, 3))),
    (((2, 2, 2), (2, 2, 2)),
     ((3, 3, 1, 1), (3, 1, 3, 1))),
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
    assert q == 31
    mask = (1 << q) - 1
    stub_a = [[[G.e] for _ in range(3)] for _ in range(2)]
    stub_b = [[[G.e] for _ in range(4)] for _ in range(2)]
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
            assert n == 16 * q
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
    """Historical name; now exactly test logical existence through four.

    Every support through four partitions into singleton/pair supports.
    One syndrome representative suffices modulo the stabilizer span:
    if every collision difference is a stabilizer, all representatives
    of that syndrome have the same quotient class.
    """
    basis = _row_basis(stabilizers)
    cols = _component_columns(checks, width)
    seen = {0: 0}
    bits = [1 << i for i in range(width)]
    for i, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[i]
        elif _outside_span(bits[i] ^ old, basis):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _outside_span(word ^ old, basis):
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
    """Intrinsic support legality; inter-cell collisions are soft costs."""
    ratio = (3, 4, 5, 6)[mode]
    for row in grid:
        for cell in row:
            if len(cell) != len(set(cell)):
                return False
            differences = _cell_differences(cell, q)
            if len(cell) >= 3:
                if len(differences) != len(cell)*(len(cell)-1):
                    return False
            if len(cell) == 3:
                aa = [g[0] for g in cell]
                if not any(
                    (aa[c]-aa[a]) % q
                    == ratio*((aa[b]-aa[a]) % q) % q
                    for a in range(3) for b in range(3) if a != b
                    for c in range(3) if c != a and c != b
                ):
                    return False
    return True


@lru_cache(maxsize=1)
def _gf32_character():
    """Character arithmetic only, never a replacement parity-matrix lift.

    GF32 = F2[z]/(z^5+z^2+1). Its nonidentity element z has order 31.
    """
    powers = [1]
    for _ in range(30):
        value = powers[-1] << 1
        if value & 32:
            value ^= 0x25
        powers.append(value)
    value = powers[-1] << 1
    if value & 32:
        value ^= 0x25
    assert value == 1 and len(set(powers)) == 31
    logs = {value: exponent for exponent, value in enumerate(powers)}
    return tuple(powers), logs


def _draw_character_a(rng, G, mode):
    """Match both A-row evaluations at one fixed nontrivial character.

    Each top cell is 1+x^d. A bottom cell x^b(1+x^e) is chosen so
    alpha^b(1+alpha^e)=1+alpha^d. The resulting rank is NOT assumed:
    component admission demands rank 55 in the verified construction.
    """
    q = G.q
    powers, logs = _gf32_character()
    best = None
    for attempt in range(32):
        gaps = rng.sample(range(1, 16), 6)
        if mode == 1:
            # One cross-column repeated gap, not equal gaps in one column.
            gaps[3] = gaps[1]
        elif mode == 2:
            # An additive bottom-gap geometry is not an affine orbit draw.
            top = gaps[:3]
            gaps[3:] = [
                min((top[j]+top[(j+1) % 3]) % q,
                    (-top[j]-top[(j+1) % 3]) % q)
                for j in range(3)]
        elif mode == 3:
            gaps = rng.sample((1,2,3,5,7,11,13,15), 6)
        if any(not gaps[j+3] or gaps[j] == gaps[j+3]
               for j in range(3)):
            continue

        grid = [[], []]
        used, crossing_used, cost = [set(), set()], set(), 0
        for j in range(3):
            d, e = gaps[j], gaps[j+3]
            b = (logs[1 ^ powers[d]]-logs[1 ^ powers[e]]) % q
            x, y = (0, d), tuple(sorted((b, (b+e) % q)))
            assert powers[y[0]] ^ powers[y[1]] == 1 ^ powers[d]
            for i, vals in enumerate((x, y)):
                ds = {(a-h) % q for a in vals for h in vals if a != h}
                cost += 2*len(ds & used[i])
                used[i].update(ds)
                grid[i].append([(a, 0) for a in vals])
            crossing = [(a-h) % q for a in x for h in y]
            cs = set(crossing)
            cost += len(crossing)-len(cs)+2*len(cs & crossing_used)
            crossing_used.update(cs)
        if best is None or cost < best[0]:
            best = (cost, grid)
        if cost == 0:
            break
    if best is None:
        raise ValueError("GF32 character gap policy exhausted")
    assert _recipe_ok(best[1], mode, q)
    return best[1]


def _draw_component(rng, G, counts, mode):
    """Character-matched A; independently packed intrinsic B supports."""
    q = G.q
    assert q == 31
    width = len(counts[0])
    if width == 3:
        assert tuple(map(tuple, counts)) == ((2,2,2), (2,2,2))
        return _draw_character_a(rng, G, mode)

    # Unlike C37, ratios 4 and 8 are equivalent over C31. Use four
    # different affine classes; ratio six has a two-element S3 orbit.
    ratio = (3, 4, 5, 6)[mode]

    def draw(size):
        if size == 0:
            return []
        if size == 3:
            base, step = rng.randrange(q), rng.randrange(1, q)
            return [(base+t*step) % q for t in (0,1,ratio)]
        for _ in range(96):
            vals = rng.sample(range(q), size)
            ds = {(a-b) % q for a in vals for b in vals if a != b}
            if size < 3 or len(ds) == size*(size-1):
                return vals
        return None

    best = None
    for restart in range(8):
        used, global_used, crossing_used = [set(), set()], set(), set()
        columns = [None]*width
        total = 0
        order = list(range(width))
        rng.shuffle(order)
        order.sort(key=lambda j: -max(counts[0][j], counts[1][j]))
        for j in order:
            chosen = None
            for attempt in range(40):
                x, y = draw(counts[0][j]), draw(counts[1][j])
                if x is None or y is None:
                    continue
                anchor = x[0] if x else y[0]
                x, y = [sorted((a-anchor) % q for a in vals)
                        for vals in (x,y)]
                local = [
                    {(a-b) % q for a in vals for b in vals if a != b}
                    for vals in (x,y)]
                crossings = [(a-b) % q for a in x for b in y]
                cs = set(crossings)
                cost = len(crossings)-len(cs)+2*len(cs & crossing_used)
                cost += sum(2*len(local[i] & used[i]) for i in range(2))
                if mode == 3:
                    cost += len(local[0] & local[1])
                    cost += len((local[0] | local[1]) & global_used)
                if chosen is None or cost < chosen[0]:
                    chosen = (cost, x, y, local, cs)
                if cost == 0:
                    break
            if chosen is None:
                break
            cost, x, y, local, cs = chosen
            columns[j] = (x,y)
            total += cost
            for i in range(2):
                used[i].update(local[i])
            global_used.update(local[0] | local[1])
            crossing_used.update(cs)
        if any(column is None for column in columns):
            continue
        grid = [
            [[(a,0) for a in columns[j][i]] for j in range(width)]
            for i in range(2)]
        assert _recipe_ok(grid, mode, q)
        if best is None or total < best[0]:
            best = (total, grid)
        if total == 0:
            break
    if best is None:
        raise ValueError("C31 soft voltage packing exhausted")
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


def _cofactor_words(cols, q):
    """Actual kernel supports from commuting VERIFIED block operators.

    For each three-column minor, the three two-by-two cofactors form
    a component-kernel word. All assertions use actual column syndromes.
    Translated pair combinations are also genuine kernel words.
    """
    from itertools import combinations
    blocks = len(cols) // q
    mask = (1 << q) - 1
    originals = []
    for triple in combinations(range(blocks), 3):
        word = 0
        for position, column in enumerate(triple):
            other = [j for j in triple if j != column]
            b, c = other
            bc = _block_apply(
                cols, q, 0, b, (cols[c * q] >> q) & mask)
            cb = _block_apply(
                cols, q, 0, c, (cols[b * q] >> q) & mask)
            word |= (bc ^ cb) << (column * q)
        if word:
            assert _syndrome_of(word, cols) == 0
            originals.append(word)

    discoveries = set(originals)
    best = min((w.bit_count() for w in originals),
               default=len(cols) + 1)
    for i, first in enumerate(originals):
        for second in originals[:i + 1]:
            shifts = range(1, (q + 1) // 2) if first == second else range(q)
            for shift in shifts:
                translated = _component_shift(second, q, blocks, shift)
                combined = first ^ translated
                if not combined:
                    continue
                weight = combined.bit_count()
                best = min(best, weight)
                if weight <= best + 4:
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


def _translation_close(cols, seeds, q, rounds=2):
    """Recombine actual kernel orbits without filling memory with copies.

    Translation acts on supports only. Every returned representative is
    checked against scaffold-derived syndromes. This is a bounded witness
    search, not an exhaustive distance or multiplicity calculation.
    """
    blocks = len(cols)//q

    def move(word, shift):
        return _component_shift(word, q, blocks, shift)

    def canonical(word):
        return min(move(word, shift) for shift in range(q))

    representatives = sorted(
        {canonical(w) for w in seeds if w},
        key=lambda w: (w.bit_count(), w))[:24]
    for _ in range(rounds):
        discoveries = set(representatives)
        short = representatives[:12]
        for i, left in enumerate(short):
            for right in short[:i+1]:
                alternatives = []
                for shift in range(q):
                    word = left ^ move(right, shift)
                    if word:
                        alternatives.append((word.bit_count(), word))
                for _, word in sorted(alternatives)[:3]:
                    discoveries.add(canonical(word))
        representatives = sorted(
            discoveries, key=lambda w: (w.bit_count(), w))[:24]
    assert all(_syndrome_of(w, cols) == 0 for w in representatives)
    return representatives


def _component_pool(rng, G, side, counts, mode, quota):
    """Rank-stratified B-pressure search with actual local voltage edits."""
    q, mask = G.q, (1 << G.q)-1
    strata, seen = {}, set()
    positions = [(i, j) for i, row in enumerate(counts)
                 for j, count in enumerate(row) if count]

    def merit(item):
        return (min(item["d"], 28), -item["hits"], -item["cycles"])

    def refresh(item, trials, close=False):
        if close:
            item["words"] = _translation_close(
                item["cols"], item["words"], q)
        item["d"] = min(item["d"], _witness_probe(
            item["cols"], item["words"], rng, trials))
        item["hits"] = sum(
            w.bit_count() <= item["d"]+1 for w in item["words"])
        assert all(_syndrome_of(w, item["cols"]) == 0
                   for w in item["words"])

    def admit(grid, attempt, trials):
        if grid is None or not _recipe_ok(grid, mode, q):
            return
        key = tuple(tuple(sorted(cell)) for row in grid for cell in row)
        if key in seen:
            return
        seen.add(key)
        cols = _compile_component(G, side, grid)
        rank = gf2_rank(cols)
        # Two trivial A dependencies plus exactly one degree-five factor.
        # Reject accidental additional factors instead of letting them
        # move the complete program into a different rate band.
        if side == "A":
            if rank != 2*q-7:
                return
        elif rank < 2*q-3:
            return
        if _kernel_le4(cols):
            return
        words = _cofactor_words(cols, q)
        if not words or words[0].bit_count() <= 8:
            return
        item = dict(
            grid=grid, cols=cols, rank=rank, words=words,
            d=words[0].bit_count(), hits=0,
            cycles=_component_four_cycles(cols), attempt=attempt)
        refresh(item, trials)
        bucket = strata.setdefault(rank, [])
        bucket.append(item)
        bucket.sort(key=merit, reverse=True)
        del bucket[6:]

    def draw():
        try:
            return _draw_component(rng, G, counts, mode)
        except ValueError:
            return None

    draws = quota if side == "A" else 4*quota
    for attempt in range(draws):
        admit(draw(), attempt, 1 if side == "A" else 2)

    for bucket in strata.values():
        for item in bucket:
            refresh(item, 8 if side == "A" else 12)
        bucket.sort(key=merit, reverse=True)

    # A changed matrix always receives freshly computed cofactor seeds.
    # No witness from the old matrix is inherited by a voltage mutation.
    repairs = 32 if side == "A" else 128
    for step in range(repairs):
        if not strata:
            admit(draw(), draws+step, 4)
            continue
        rank = rng.choice(sorted(strata))
        bucket = strata[rank]
        parent = rng.choice(bucket[:min(3, len(bucket))])
        grid = [[list(cell) for cell in row] for row in parent["grid"]]
        pressure = [
            1+sum(((word >> (j*q)) & mask).bit_count()
                  for word in parent["words"][:12])
            for i, j in positions]
        i, j = rng.choices(positions, weights=pressure, k=1)[0]
        count = counts[i][j]

        if side == "A" or step % 8 == 7:
            # Every fresh A column has the same character row ratio.
            # Replacing both cells preserves it; a one-term edit does not.
            replacement = draw()
            if replacement is None:
                continue
            for row in range(2):
                grid[row][j] = replacement[row][j]
        elif count >= 3 and step % 4 != 0:
            # Preserve the card's intrinsic triangle class while
            # changing its internal scale and relative row coupling.
            replacement = draw()
            if replacement is None:
                continue
            grid[i][j] = replacement[i][j]
        else:
            cell = list(grid[i][j])
            t = rng.randrange(count)
            old = cell[t][0]
            step_size = (rng.choice((-3, -2, -1, 1, 2, 3))
                         if step % 3 else rng.randrange(1, q))
            g = ((old+step_size) % q, 0)
            if g in cell:
                continue
            cell[t] = g
            grid[i][j] = cell
        admit(grid, draws+step, 4 if side == "A" else 6)

    # First preserve one candidate from each actual rank stratum.
    heads = sorted((bucket[0] for bucket in strata.values()),
                   key=merit, reverse=True)
    pool = heads[:8]
    extras = sorted(
        (item for bucket in strata.values() for item in bucket[1:]),
        key=merit, reverse=True)
    pool.extend(extras[:max(0, 8-len(pool))])
    for item in pool:
        refresh(item, 32 if side == "A" else 64, close=True)
    pool.sort(key=merit, reverse=True)
    finalists = pool[:4]

    # Equal additional budgets within each side; prior light witnesses
    # remain valid and can only lower each candidate's reading.
    for item in finalists:
        refresh(item, 64 if side == "A" else 128, close=True)
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
    """Five B architectures times four character/gap/support policies."""
    assert G.q == 31
    assert A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
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
            BASE_SEED * 100003 + 31015543 + 104729 * idx)
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
                    assert n == 496 and k >= 62
                    ar, br = _extract_components(hx, hz, q)
                    assert _component_columns(ar, 3 * q) == a["cols"]
                    assert _component_columns(br, 4 * q) == b["cols"]
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
                    _component_columns(hz, n), rng, 12, stabilizers=hx)
                dz = _kernel_probe(
                    _component_columns(hx, n), rng, 12, stabilizers=hz)
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
                assert all(len(cell) == len(set(cell)) <= 5
                           for grid in (A, B)
                           for row in grid for cell in row)
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
    """Return this card's fresh character-singular C31 design."""
    selected = _design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh C31 character-screened filling in cell 0.150")
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
# REFUTED: Trading B density for one or two A-trinomial upgrades supplies distance-eighteen leaders on C37 (2,3)x(2,4); all 24 parent slots in those two architectures had d2s<=16 and none advanced to measurement.
# REFUTED: Independently changed voltage labels alone ensure design diversity; the measured C37 B-pressure parent collapsed 18 of 60 slots as isomorphic copies despite different fillings.
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
