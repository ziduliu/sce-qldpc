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
CONCEPT = ("A4 x C12, order 144, A=B=(1,3): independently challenged tetrahedral-sharing cubic components should produce five new verified k>=596 codes with measured readings >=12 and beat cell-0.400 fitness 59.451. Failure of this five-code target refutes the experiment; readings remain provisional upper bounds.")
GROUP = ("Jump from the centre-trivial order-110 affine group to A4 x C12, whose centre has order twelve. This order-144, centre-twelve bed is absent from BEDS. Related A4 x C11 programs measured readings twelve at k=540..548, but their rank-gain proposals also produced lighter readings. The central extension and shared tetrahedral operators therefore motivate an experiment, not distance transfer or a rank-deficit formula. Native operations propose supports; authoritative matrices decide acceptance.")
SHAPES = ("Keep A_SHAPE=B_SHAPE=(1,3) unchanged. Order 144 gives n=1440, k>=576 and guaranteed rate 0.400; q_max=149. Equal components have 432 variables and 144 checks, nominal rate 2/3. The parent's five measured codes had k=440 and reading twelve, with thousands of sector-one minima concentrated on Acol1/2 and Bcol1/2 strips. Actual full ranks enforce [0.375,0.425); five k>=596/readings-twelve codes would score at least 59.6.")
FILLING = ("All six cells remain distinct-term cubics, so HX=HZ=9+3=12. Twenty cards cross five tetrahedral-sharing A/B pairings with four central-ratio portfolios. Repeated tetrahedral supports have independently varied C12 coordinates; every cubic must generate the whole group and pass bidirectional difference distinctness. Assemble only scaffold monomial blocks, exclude component kernels through four, and retain genuine localized, central-fixed, Klein-fixed and unrestricted witnesses. Repairs redraw occupied cells while preserving their support portfolio. Every finalist receives independent deeper challenges. Original-stabilizer logical membership, full native-component agreement, ranks, weights and CSS remain binding. Select by actual k times a conservative squared reading capped at twelve; no finite search or component exclusion certifies quantum distance or arbitrary Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r040_A4xC12_13x13_tetrasharing_subgroup_challenge_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 144

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C12 requires order 144")
    # (p0,p1,p2,p3,z): even permutation i -> pi, times central C12.
    tetra = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i+1, 4)) % 2 == 0]
    els = [p + (z,) for z in range(12) for p in tetra]
    positions = {g: i for i, g in enumerate(els)}
    assert len(tetra) == 12 and len(els) == q

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 12,)

    def mul(g, h):
        g, h = red(g), red(h)
        return tuple(g[h[i]] for i in range(4)) + (
            (g[4] + h[4]) % 12,)

    def inv(g):
        g = red(g)
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % 12,)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0,1,2,3,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Four density allocations crossed with five quotient-coordinate profiles.
# HX/HZ maxima, respectively: (12,12), (9,12), (11,12), (10,12).
_COUNT_RECIPES = (
    ((3, 3, 3), (2, 2, 2, 3)),
    ((2, 2, 2), (2, 2, 3, 3)),
    ((2, 3, 3), (2, 2, 2, 3)),
    ((2, 2, 3), (2, 2, 2, 3)),
)
_TRI_QUOTIENT = ((1, 2), (1, 4), (3, 2), (3, 4), (1, 6))


def _syndrome_columns(rows, n):
    """Syndromes from the scaffold's actual lift, never a separate lift."""
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _pairs_no_kernel4(cols):
    """Exclude weights 1..4 and retain all pair syndromes."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = set()
    for i, a in enumerate(cols):
        for j in range(i):
            value = a ^ cols[j]
            if value in singles or value in pairs:
                return None
            pairs.add(value)
    return pairs


def _anchored_kernel5_free(cols, pairs, anchors):
    """Check triples containing an orbit representative against all pairs.

    Caller supplies representatives of a VALID symmetry action.
    After weights <=4 are excluded, a matching triple and pair must
    be disjoint, and therefore give a weight-five kernel vector.
    """
    width = len(cols)
    for root in anchors:
        cr = cols[root]
        for i in range(width):
            if i == root:
                continue
            value = cr ^ cols[i]
            for j in range(i):
                if j != root and (value ^ cols[j]) in pairs:
                    return False
    return True


def _component_kernel5_free(rows, width, q):
    cols = _syndrome_columns(rows, width)
    pairs = _pairs_no_kernel4(cols)
    if pairs is None:
        return False
    # Each classical component has a regular group action, transitive
    # within every column block. One representative per block suffices.
    return _anchored_kernel5_free(cols, pairs, range(0, width, q))


def _short_cycle_free_bins(rng, G, count, model):
    """Prefer few C4 collisions; never require an unverified feasible set.

    Primitive complement coordinates give order-ten ratios. Their odd
    quotient parity makes the binomial-only Cayley graph bipartite.
    """
    if count == 0:
        return []
    eligible = [g for g in G.elements if g[1] in (1, 3, 7, 9)]
    target = (1, 3, 7, 9)[model % 4]
    best = None
    for attempt in range(64):
        gs = rng.sample(eligible, count)
        ns = gs + [G.inv(g) for g in gs]
        if len({G._key(g) for g in ns}) != 2 * count:
            continue
        products = {}
        collisions = 0
        for g in ns:
            gi = G._key(G.inv(g))
            for h in ns:
                if G._key(h) == gi:
                    continue
                value = G._key(G.mul(g, h))
                collisions += products.get(value, 0)
                products[value] = products.get(value, 0) + 1
        commuting = sum(
            G._key(G.mul(g, h)) == G._key(G.mul(h, g))
            for i, g in enumerate(gs) for h in gs[:i])
        geometry = sum(min((g[1] - target) % 10,
                           (target - g[1]) % 10) for g in gs)
        key = (collisions, commuting, geometry,
               abs(len({g[1] for g in gs}) - (1 + model % count)))
        if best is None or key < best[0]:
            best = (key, gs)
    # This fallback is legal even if every sampled set contained an
    # inverse pair. Subsequent verified-component screening ranks it.
    return best[1] if best is not None else rng.sample(eligible, count)


def _planned_row(rng, G, counts, model, offset):
    bins = _short_cycle_free_bins(
        rng, G, sum(c == 2 for c in counts), model + offset)
    row = []
    bi = ti = 0
    for count in counts:
        if count == 2:
            row.append([G.e, bins[bi]])
            bi += 1
            continue
        b1, b2 = _TRI_QUOTIENT[(model + offset + ti) % 5]
        ti += 1
        # Exhaustively available choices prevent another labeling gate
        # from destroying the slot. The two nonidentity terms do not commute.
        choices = [
            ((a, b1), (b, b2))
            for a in range(11) for b in range(11)
            if G._key(G.mul((a, b1), (b, b2))) !=
               G._key(G.mul((b, b2), (a, b1)))
        ]
        a, b = rng.choice(choices)
        row.append([G.e, a, b])
    return [row]


def _verified_components(hx, hz, q):
    """Extract the actual components for the program's current shapes."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            ar.append(sum(
                ((source >> (j * cb * q)) & mask) << (j * q)
                for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + r] & bmask
          for a in range(rb) for r in range(q)]
    return ar, br


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _component_floor(rows, width, q):
    """Exact classical-kernel screening through five on a verified component.

    Return six means no component kernel word of weight <=5 exists.
    It is not a quantum-distance lower bound.
    """
    cols = _syndrome_columns(rows, width)
    singles = set(cols)
    if 0 in singles:
        return 1, cols
    if len(singles) != width:
        return 2, cols
    pairs = set()
    for i, a in enumerate(cols):
        for j in range(i):
            value = a ^ cols[j]
            if value in singles:
                return 3, cols
            if value in pairs:
                return 4, cols
            pairs.add(value)
    free = _anchored_kernel5_free(
        cols, pairs, range(0, width, q))
    return (6 if free else 5), cols


def _component_reading(cols, rng, trials):
    """Search actual classical kernel circuits; readings are upper bounds."""
    width = len(cols)
    best = width + 1
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
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
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        light = sorted(words, key=int.bit_count)[:20]
        for i, word in enumerate(light):
            for other in light[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 4:
            break
    return best


def _retired_plane_ctx(rng, G, idx):
    """Unused old-shape implementation; NEW_CTX below replaces it."""
    q = G.q
    card = idx % 20

    # Cache the native group operations, not an independently written lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = G.idx(G.e)

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()

    def component(cells, side):
        indexed = [[G.idx(g) for g in cell] for cell in cells]
        if side == "A":
            rows, _, _ = lifted_matrices(probe, [indexed], [[[]]])
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], [indexed])
        width = len(cells) * q
        assert len(rows) == q
        assert all(row >> width == 0 for row in rows)
        return rows

    # F1024 = F2[z]/Phi_11(z). Phi_11 is irreducible over F2.
    # Here z has order eleven, matching the normal-subgroup generator.
    def fmul(a, b):
        answer = 0
        while b:
            if b & 1:
                answer ^= a
            b >>= 1
            a <<= 1
            if a & 1024:
                a ^= 2047
        return answer

    def frobenius(a, b):
        for _ in range(b % 10):
            a = fmul(a, a)
        return a

    roots = [1]
    for _ in range(10):
        roots.append(fmul(roots[-1], 2))
    assert fmul(roots[-1], 2) == 1

    # Adjoint images on a binary basis of F1024. This representation
    # proposes supports only; all lift screens below use the scaffold.
    adjoints = [
        [frobenius(fmul(1 << t, roots[g[0]]), -g[1])
         for t in range(10)]
        for g in G.elements
    ]
    lines, planes = {}, {}
    ident = G.idx(G.e)

    def adjoint_kernel(support):
        """Exact binary kernel of the proposed three-term adjoint map."""
        pivots = {}
        basis = []
        for t in range(10):
            syndrome = 0
            for g in support:
                syndrome ^= adjoints[g][t]
            word = 1 << t
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
        return basis

    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            support = tuple(sorted((ident, a, b)))
            # p -> p*h is a column gauge and preserves left annihilators.
            # Keep one such operator class, not three anchor choices.
            gauges = [
                tuple(sorted(table[g][inverses[root]] for g in support))
                for root in support
            ]
            if support != min(gauges):
                continue
            basis = adjoint_kernel(support)
            if not basis:
                continue
            # A large kernel is unnecessary for this proposal bank.
            # Truncation only omits proposals, never changes a screen.
            values = [0]
            for v in basis[:5]:
                values += [u ^ v for u in values]
            values = sorted(values[1:])
            for v in values:
                lines.setdefault(v, []).append(support)
            for i, u in enumerate(values):
                for v in values[i + 1:]:
                    w = u ^ v
                    if v < w:
                        planes.setdefault((u, v, w), []).append(support)

    catalogs = {}
    for kind, bank in (("plane", planes), ("line", lines)):
        catalog = [
            (functional, supports)
            for functional, supports in bank.items()
            if len(supports) >= 3
        ]
        catalog.sort(key=lambda item: (-len(item[1]), item[0]))
        catalogs[kind] = catalog
    selections = {"plane": 0, "line": 0}

    def constrained_cells(kind):
        """Spread fallback searches through the whole annihilator catalog."""
        catalog = catalogs[kind]
        if not catalog:
            return None, None
        turn = selections[kind]
        selections[kind] += 1
        # Interleave large classes with the rest instead of permanently
        # restricting every slot to the same sixteen constraints.
        if turn % 3 == 0:
            position = (card + 7 * turn + idx // 20) % min(32, len(catalog))
        else:
            position = (37 * card + 101 * turn + 53 * (idx // 20)) % len(catalog)
        functional, supports = catalog[position]
        options = [
            [G.elements[t] for t in support]
            for support in supports
        ]
        return options, functional

    # Two terms occupy the same quotient fiber. Their relative ratio is
    # a nonzero order-eleven translation, not a change of cell gauge.
    # These even/nonprimitive repeated-fiber profiles differ from both
    # the parent's distinct-fiber recipes and its earlier odd recipes.
    spectra = (2, 4, 5, 6, 8)
    bin_profiles = ((1, 3, 7), (1, 3, 9), (1, 1, 3), (1, 1, 7))

    def fresh_tri(attempt):
        b = spectra[card % 5]
        if attempt >= 80:
            # A difficult preferred profile need not void the whole slot.
            # These alternatives retain the new repeated-fiber mechanism.
            b = (2, 4, 6, 8)[(card + attempt + idx // 20) % 4]
        g = (rng.randrange(1, 11), 0)
        h = (rng.randrange(11), b)
        assert G._key(G.mul(g, h)) != G._key(G.mul(h, g))
        return [G.e, g, h]

    def fresh_bins(attempt):
        # Late attempts provide a feasible primitive-coordinate fallback
        # while retaining this card's new trinomial recipe.
        if attempt >= 80:
            return _short_cycle_free_bins(
                rng, G, 3, card + 2 + idx // 20)
        coordinates = bin_profiles[card // 5]
        best = None
        for _ in range(32):
            gs = [
                (rng.randrange(1, 11) if b == 0 else rng.randrange(11), b)
                for b in coordinates
            ]
            ns = gs + [G.inv(g) for g in gs]
            if len({G._key(g) for g in ns}) != 6:
                continue
            products = {}
            collisions = 0
            for g in ns:
                inverse = G._key(G.inv(g))
                for h in ns:
                    if G._key(h) == inverse:
                        continue
                    key = G._key(G.mul(g, h))
                    collisions += products.get(key, 0)
                    products[key] = products.get(key, 0) + 1
            commuting = sum(
                G._key(G.mul(g, h)) == G._key(G.mul(h, g))
                for i, g in enumerate(gs) for h in gs[:i])
            key = (collisions, commuting)
            if best is None or key < best[0]:
                best = (key, gs)
        if best is None:
            return _short_cycle_free_bins(rng, G, 3, card)
        return best[1]

    def fresh(side, options, attempt):
        if side == "A":
            return [list(cell) for cell in rng.sample(options, 3)]
        return [[G.e, g] for g in fresh_bins(attempt)] + [fresh_tri(attempt)]

    def pool(side, options, kind="line"):
        """Actual component ranks, exact kernel-five gates and resampling."""
        width = (3 if side == "A" else 4) * q
        allowed = ({80, 90} if kind == "plane" else
                   {80, 90, 100}) if side == "A" else {110}
        records = []
        signatures = set()
        trials = 192 if side == "A" and kind == "plane" else 128
        for trial in range(trials):
            cells = fresh(side, options, trial)
            signature = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell))
                for cell in cells))
            if signature in signatures:
                continue
            signatures.add(signature)
            rows = component(cells, side)
            rank = gf2_rank(rows)
            if rank not in allowed:
                continue
            cycles = _four_cycles(rows)
            records.append((rank, cycles, trial, cells, rows))

        # Larger verified rank deficits are useful only after exact
        # exclusion of the short component words.
        records.sort(key=lambda record: record[:3])
        survivors = []
        for rank, cycles, trial, cells, rows in records:
            floor, columns = _component_floor(rows, width, q)
            if floor != 6:
                continue
            reading = _component_reading(columns, rng, 12)
            survivors.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": cycles, "reading": reading, "trial": trial,
            })
            if len(survivors) >= 8:
                break
        survivors.sort(key=lambda record: (
            -record["rank"], record["reading"], -record["cycles"]),
            reverse=True)
        survivors = survivors[:3]
        for record in survivors:
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], rng, 48))
        survivors.sort(key=lambda record: (
            -record["rank"], record["reading"], -record["cycles"]),
            reverse=True)
        return survivors

    # All cached matrices below are outputs of lifted_matrices. Joining
    # their disjoint column blocks is not an independently implemented lift.
    gain_cache = {"rows": {}, "columns": {}, "pairs": {}, "classes": None}

    def plane_pool():
        """Search plane orbits by exact cell/pair/triple compatibility.

        The old search prioritized catalog cardinality and considered
        only sixteen functionals. A large catalog can contain many
        mutually incompatible operators. Here compatibility, actual
        rank and complete triple screening decide acceptance.
        """
        if gain_cache["classes"] is None:
            action_cache = {}

            def image(g, value):
                key = (g, value)
                if key not in action_cache:
                    answer = 0
                    word = value
                    while word:
                        bit = word & -word
                        answer ^= adjoints[g][bit.bit_length() - 1]
                        word ^= bit
                    action_cache[key] = answer
                return action_cache[key]

            orbits = {}
            for functional, supports in catalogs["plane"]:
                orbit = min(
                    tuple(sorted(image(g, value) for value in functional))
                    for g in range(q))
                orbits.setdefault(orbit, []).append((functional, supports))

            # Each entry is an orbit of search constraints. We still draw
            # genuinely different operator triples and B matrices; no
            # symmetry images are returned as additional code designs.
            classes = []
            for orbit in sorted(orbits):
                entries = orbits[orbit]
                largest = max(len(entry[1]) for entry in entries)
                entries = [entry for entry in entries
                           if len(entry[1]) == largest]
                classes.append(rng.choice(entries))
            gain_cache["classes"] = classes

        classes = list(gain_cache["classes"])
        if not classes:
            return []
        turn = selections["plane"]
        selections["plane"] += 1
        rng.shuffle(classes)
        offset = (card + 17 * turn + 11 * (idx // 20)) % len(classes)
        classes = classes[offset:] + classes[:offset]

        def cell_columns(support):
            cache = gain_cache["columns"]
            if support not in cache:
                # A single left-action cell, isolated only for screening.
                rows, _, width = lifted_matrices(
                    probe, [[list(support)]], [[[]]])
                assert width == 2 * q
                assert all(row >> q == 0 for row in rows)
                cols = _syndrome_columns(rows, q)
                assert all(column.bit_count() == 3 for column in cols)
                gain_cache["rows"][support] = rows
                cache[support] = (
                    cols if _pairs_no_kernel4(cols) is not None else None)
            return cache[support]

        def compatible(left, right):
            pair = tuple(sorted((left, right)))
            cache = gain_cache["pairs"]
            if pair not in cache:
                lc, rc = cell_columns(left), cell_columns(right)
                cache[pair] = (
                    lc is not None and rc is not None and
                    _pairs_no_kernel4(lc + rc) is not None)
            return cache[pair]

        records = []
        signatures = set()
        # Round-robin visits prevent one large, incompatible catalog
        # from consuming the complete search budget.
        budget = 384 + 64 * (card % 3)
        for attempt in range(budget):
            functional, supports = classes[attempt % len(classes)]
            signature = tuple(sorted(rng.sample(supports, 3)))
            if signature in signatures:
                continue
            signatures.add(signature)
            if not all(
                    compatible(signature[i], signature[j])
                    for i in range(3) for j in range(i)):
                continue

            columns = []
            for support in signature:
                columns.extend(cell_columns(support))
            # Pair compatibility alone misses four-words touching all
            # three blocks. Test the complete component as well.
            if _pairs_no_kernel4(columns) is None:
                continue

            # Every column has odd weight three, so a zero syndrome
            # cannot have odd support weight. The exhaustive <=4 test
            # therefore also excludes all component words of weight five.
            rows = [
                sum(gain_cache["rows"][support][r] << (j * q)
                    for j, support in enumerate(signature))
                for r in range(q)
            ]
            rank = gf2_rank(rows)
            if rank not in (80, 90):
                continue
            cells = [[G.elements[t] for t in support]
                     for support in signature]
            cycles = _four_cycles(rows)
            reading = _component_reading(columns, rng, 12)
            records.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": cycles, "reading": reading,
                "trial": attempt, "functional": functional,
            })
            records.sort(key=lambda record: (
                -record["rank"], record["reading"], -record["cycles"]),
                reverse=True)
            del records[8:]
            if (len(records) >= 6 and
                    all(record["rank"] == 80 for record in records[:3])):
                break

        finalists = records[:3]
        for record in finalists:
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], rng, 48))
        finalists.sort(key=lambda record: (
            -record["rank"], record["reading"], -record["cycles"]),
            reverse=True)
        return finalists

    # B is independent of the annihilator constraint and can be reused
    # while several different A annihilator classes are examined.
    bb = pool("B", None)
    if not bb:
        bb = pool("B", None)
    if not bb:
        raise ValueError("new B card failed component kernel-five screening")

    def build(kind):
        if kind == "plane":
            aa = plane_pool()
            functional = None
        else:
            options, functional = constrained_cells(kind)
            if options is None:
                return None
            aa = pool("A", options, kind)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            -ab[0]["rank"],
            min(ab[0]["reading"], ab[1]["reading"]),
            ab[0]["reading"] + ab[1]["reading"],
            -ab[0]["cycles"] - ab[1]["cycles"]), reverse=True)

        for a, b in proposals:
            A, B = [a["cells"]], [b["cells"]]
            sa = [[len(cell) for cell in A[0]]]
            sb = [[len(cell) for cell in B[0]]]
            assert _grid_weights(sa, sb) == 12
            assert _grid_weights(sa, sb) <= W_MAX
            assert _no_dead_lines(sa, sb)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1430 and 690 <= k <= 750):
                continue
            # No nonabelian full-code symmetry is assumed.
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue
            assert 19 * n <= 40 * k < 21 * n
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert css_ok(hx, hz)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": (kind, card // 5, card % 5),
                "functional": a.get("functional", functional),
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
            }
        return None

    # Two broad compatibility searches replace four narrow catalog draws.
    # A failed gain search still has independently generated, new B labels
    # and broadly selected line fallbacks; no archived fillings are copied.
    for _ in range(2):
        answer = build("plane")
        if answer is not None:
            return answer
    for _ in range(8):
        answer = build("line")
        if answer is not None:
            return answer
    raise ValueError("no compatibility-screened annihilator design passed ranks")


def NEW_CTX(rng, G, idx):
    """Twenty tetrahedral-sharing cards with binding subgroup challenges."""
    q = G.q
    assert q == 144 and A_SHAPE == B_SHAPE == (1, 3)
    card, cohort = idx % 20, idx // 20
    composition, model = divmod(card, 4)
    seed = rng.randrange(1 << 62) ^ (
        0xA4C12D144 + 15485863 * (idx + 1))
    rng = random.Random(seed ^ 0x71313C12)
    width = 3 * q
    mask = (1 << q) - 1

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverse[g]

    P = Probe()

    # Obtain both lift conventions from the authoritative helper.
    monomials = [[[g] for g in range(q)]]
    source_a, _, _ = lifted_matrices(P, monomials, [[[]]])
    _, source_b, _ = lifted_matrices(P, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g * q)) & mask for row in source_a]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in source_b]
              for g in range(q)],
    }

    # Native A4/V4 charges distinguish the two oriented three-cycle roles.
    tetra = [g for g in G.elements if g[4] == 0]
    klein = [g for g in tetra
             if all(g[g[t]] == t for t in range(4))]
    assert len(klein) == 4
    generator = next(g for g in tetra if g not in klein)
    roots = (G.e, generator, G.mul(generator, generator))
    charge = {}
    for label, root in enumerate(roots):
        for h in klein:
            permutation = G.mul(root, h)[:4]
            assert permutation not in charge
            charge[permutation] = label
    assert len(charge) == 12
    oriented = {}
    for g in tetra:
        label = charge[g[:4]]
        if label:
            fixed = next(t for t in range(4) if g[t] == t)
            oriented[label, fixed] = g[:4]
    assert len(oriented) == 8

    # Sharing concerns tetrahedral operators, not entire group labels.
    # Independent central coordinates make the actual cells different.
    arcs = (
        ((0,1), (0,1), (0,1)),
        ((0,1), (0,1), (0,2)),
        ((0,1), (0,2), (0,3)),
        ((0,1), (1,0), (0,1)),
        ((0,1), (1,2), (2,0)),
    )
    sharing_pairs = ((0,0), (0,1), (1,1), (0,2), (3,4))
    ratio_portfolios = (
        ((0,4), (0,4), (0,4)),
        ((1,3,9), (1,3,9), (6,7,10)),
        ((6,7,10), (6,7,10), (6,7,10)),
        ((0,1,3,4), (6,7,10), (0,1,3,4,6,7,9,10)),
    )
    ak, bk = sharing_pairs[composition]
    a_arcs = arcs[ak]
    # Relative disjoint fixed-point pairs, not a global code relabeling.
    b_arcs = tuple(((u+2) % 4, (v+2) % 4) for u, v in arcs[bk])
    plans = {
        "A": tuple((u, v, choices) for (u, v), choices in
                   zip(a_arcs, ratio_portfolios[model])),
        "B": tuple((u, v, choices) for (u, v), choices in
                   zip(b_arcs, ratio_portfolios[
                       (model + composition + 1 + cohort) % 4])),
    }
    desired_defects = {
        "A": (6,4,2,4,2)[ak],
        "B": (6,4,2,4,2)[bk],
    }

    # Restriction syndromes are XORs of authoritative columns.
    # Expanded words will be checked against the original component.
    ep = G.e[:4]
    subgroups = (
        tuple(G.idx(ep + (z,)) for z in (0,4,8)),
        tuple(G.idx(ep + (z,)) for z in (0,3,6,9)),
        tuple(G.idx(ep + (z,)) for z in range(12)),
        tuple(G.idx(g) for g in klein),
    )
    partitions = {"A": [], "B": []}
    for side in ("A", "B"):
        for subgroup in subgroups:
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {
                    table[root][h] if side == "A" else table[h][root]
                    for h in subgroup}
                assert len(orbit) == len(subgroup) and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(tuple(sorted(orbit)))
            groups = [
                sum(1 << (j*q+c) for c in orbit)
                for j in range(3) for orbit in orbits]
            partitions[side].append((len(subgroup), groups))

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def syndrome(columns, word):
        answer = 0
        for c in bits(word):
            answer ^= columns[c]
        return answer

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    def fresh_cell(profile):
        u, v, ratios = profile
        assert u != v
        for _ in range(128):
            central = rng.choice((1,5,7,11))
            ratio = rng.choice(ratios)
            # ratio != 2 mod 3 avoids a diagonal index-three subgroup.
            assert ratio % 3 != 2
            a = G.idx(oriented[1, u] + (central,))
            b = G.idx(oriented[2, v] + ((ratio*central) % 12,))
            cell = [identity, a, b]
            if table[a][b] == table[b][a]:
                continue
            left, right = set(), set()
            for first in cell:
                for second in cell:
                    if first != second:
                        left.add(table[inverse[first]][second])
                        right.add(table[first][inverse[second]])
            if len(left) != 6 or len(right) != 6:
                continue
            reached, pending = {identity}, [identity]
            for root in pending:
                for g in (a, b):
                    image = table[root][g]
                    if image not in reached:
                        reached.add(image)
                        pending.append(image)
            if len(reached) == q:
                return cell
        raise ValueError("full-generating tetrahedral central portfolio failed")

    def inspect(grid, side):
        # GF2 linearity of exact scaffold monomial blocks, not a new lift.
        rows = []
        mono = blocks[side]
        for r in range(q):
            row = 0
            for j, cell in enumerate(grid[0]):
                block = 0
                for g in cell:
                    block ^= mono[g][r]
                row |= block << (j * q)
            rows.append(row)
        columns = _syndrome_columns(rows, width)
        assert all(column.bit_count() == 3 for column in columns)
        rank = gf2_rank(rows)
        if not q - 12 <= rank <= q - 2:
            return None
        if _pairs_no_kernel4(columns) is None:
            return None
        fixed = [
            (order, [syndrome(columns, group) for group in groups], groups)
            for order, groups in partitions[side]]
        # Odd column degree excludes odd component kernels, not logicals
        # involving both LP sectors.
        return {
            "grid": copy_grid(grid), "rows": rows, "columns": columns,
            "rank": rank, "cycles": _four_cycles(rows), "fixed": fixed,
            "signature": signature(grid), "words": set(), "d": width + 1,
        }

    def dependencies(columns, positions):
        pivots, basis = {}, []
        for c in positions:
            value, word = columns[c], 1 << c
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    local_cache = {}

    def localized(record, side):
        for j, cell in enumerate(record["grid"][0]):
            key = (side, tuple(sorted(cell)))
            if key not in local_cache:
                columns = record["columns"][j*q:(j+1)*q]
                basis = dependencies(columns, range(q))
                best = None

                def note(word):
                    nonlocal best
                    if word and (best is None or
                                 word.bit_count() < best.bit_count()):
                        best = word

                if len(basis) <= 14:
                    word = 0
                    for t in range(1, 1 << len(basis)):
                        word ^= basis[(t & -t).bit_length() - 1]
                        note(word)
                else:
                    for i, word in enumerate(basis):
                        note(word)
                        for other in basis[:i]:
                            note(word ^ other)
                if best is not None:
                    assert syndrome(columns, best) == 0
                local_cache[key] = best
            word = local_cache[key]
            if word is not None:
                actual = word << (j * q)
                record["words"].add(actual)
                record["d"] = min(record["d"], actual.bit_count())

    def query(record, trials, diagnostic_seed):
        """Actual classical null words; all readings are upper bounds."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        order = list(range(width))
        retained = sorted(record["words"], key=int.bit_count)[:24]

        def observe(word, found):
            if not word:
                return
            weight = word.bit_count()
            record["d"] = min(record["d"], weight)
            if weight <= record["d"] + 4:
                found.append(word)

        # Challenge central/Klein-fixed words before unrestricted search.
        # These are coordinate restrictions, not independently lifted codes.
        fixed_found = []
        for order_size, compressed, groups in record["fixed"]:
            ordering = list(range(len(compressed)))
            for _ in range(max(1, trials // 8)):
                diagnostic.shuffle(ordering)
                basis = dependencies(compressed, ordering)
                light = sorted(basis, key=lambda w: (w.bit_count(), w))[:8]
                candidates = list(light)
                for i, first in enumerate(light):
                    candidates.extend(first ^ second for second in light[:i])
                for source in candidates:
                    if not source:
                        continue
                    if order_size * source.bit_count() > record["d"] + 4:
                        continue
                    actual = 0
                    for c in bits(source):
                        actual ^= groups[c]
                    assert actual.bit_count() == order_size * source.bit_count()
                    assert syndrome(columns, actual) == 0
                    observe(actual, fixed_found)
        retained = sorted(set(retained + fixed_found),
                          key=lambda w: (w.bit_count(), w))[:24]

        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = dependencies(columns, order)
            found = []
            for word in basis:
                observe(word, found)
            light = sorted(set(retained + basis), key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    observe(word ^ other, found)
            moves = retained + basis
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word, found)
                    if not changed:
                        break
            retained = sorted(set(retained + light + found),
                              key=int.bit_count)[:24]
            if record["d"] <= 6:
                break
        record["words"].update(retained)
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:48])
        for word in record["words"]:
            assert syndrome(columns, word) == 0
        if record["words"]:
            record["d"] = min(record["d"],
                              min(map(int.bit_count, record["words"])))

    def pool(side, restart):
        offset = 100003 * int(side == "B") + 300007 * restart
        profiles = plans[side]
        records, seen, historical = [], set(), set()

        def fresh():
            return [[fresh_cell(profile) for profile in profiles]]

        def prepare(grid):
            sig = signature(grid)
            if sig in seen:
                return None
            seen.add(sig)
            rec = inspect(grid, side)
            if rec is not None:
                localized(rec, side)
            return rec

        def refresh(rec):
            for word in historical:
                if word.bit_count() <= rec["d"] and syndrome(
                        rec["columns"], word) == 0:
                    rec["words"].add(word)
                    rec["d"] = min(rec["d"], word.bit_count())

        def merit(rec):
            refresh(rec)
            multiplicity = sum(w.bit_count() == rec["d"]
                               for w in rec["words"])
            return (min(12, rec["d"]), -multiplicity,
                    -abs(q-rec["rank"]-desired_defects[side]),
                    -rec["cycles"], rec["d"])

        for _ in range(64):
            rec = prepare(fresh())
            if rec is not None:
                records.append(rec)
        records.sort(key=lambda r: (r["cycles"], -r["rank"]))
        selected = records[:10]
        if not selected:
            return []
        for rec in selected:
            query(rec, 6, seed + offset)
            historical.update(rec["words"])
        selected.sort(key=merit, reverse=True)

        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(16):
                refresh(current)
                active = sorted(current["words"], key=int.bit_count)[:8]
                proposals = []
                for _ in range(2):
                    grid = copy_grid(current["grid"])
                    occupied = (sorted({c // q for c in bits(rng.choice(active))})
                                if active else list(range(3)))
                    j = rng.choice(occupied)
                    grid[0][j] = fresh_cell(profiles[j])
                    rec = prepare(grid)
                    if rec is None or rec["cycles"] > current["cycles"] + q:
                        continue
                    query(rec, 6, seed + offset + 700001 +
                          7919 * (16 * chain + step))
                    historical.update(rec["words"])
                    proposals.append(rec)
                    selected.append(rec)
                if proposals:
                    proposed = max(proposals, key=merit)
                    if merit(proposed) >= merit(current):
                        current = proposed
                if len(historical) > 128:
                    historical = set(sorted(
                        historical, key=lambda w: (w.bit_count(), w))[:96])

        selected.sort(key=merit, reverse=True)
        finalists = selected[:4]
        for rec in finalists:
            query(rec, 48 + 8 * (cohort % 3), seed + offset + 1700003)
            query(rec, 16, seed + offset + 2700011)
            historical.update(rec["words"])
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

    def span(rows):
        pivots = {}
        for value in rows:
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = value
                    break
                value ^= old
        return pivots

    def nontrivial(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def logical_challenge(hx, hz, a, b, n):
        """Verified logical upper readings using original stabilizer spans."""
        readings = []
        for side, opposite, checks, rec in (
                ("A", hx, hz, a), ("B", hz, hx, b)):
            pivots = span(checks)
            candidates = []
            for word in sorted(rec["words"], key=int.bit_count)[:6]:
                for position in range(3):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask) <<
                            ((3*j + position)*q) for j in range(3))
                    else:
                        actual = word << (3 * position * q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if nontrivial(actual, pivots):
                        candidates.append(actual)
            detected = n + 1
            diagnostic = random.Random(
                seed + (4100011 if side == "A" else 5100011))
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                reducers = list(checks)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                neighborhood = sorted(
                    checks, key=lambda row: (actual & row).bit_count(),
                    reverse=True)[:40]
                for _ in range(128):
                    first, second = diagnostic.sample(neighborhood, 2)
                    candidate = actual ^ first ^ second
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                assert actual and nontrivial(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                detected = min(detected, actual.bit_count())
            readings.append(detected)
        return tuple(readings)

    best = None
    for restart in range(2):
        aa, bb = pool("A", restart), pool("B", restart)
        proposals = sorted(
            ((a, b) for a in aa for b in bb),
            key=lambda ab: (
                min(ab[0]["d"], ab[1]["d"]),
                ab[0]["d"] + ab[1]["d"],
                -ab[0]["cycles"] - ab[1]["cycles"]),
            reverse=True)
        for a, b in proposals:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1440 and k >= 576 and 15*n <= 40*k < 17*n):
                continue
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert actual_a == a["rows"] and actual_b == b["rows"]
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue
            logicals = logical_challenge(hx, hz, a, b, n)
            # Limit selection pressure on optimistic finite-budget readings.
            # Twelve is a selection cap, never a distance lower bound.
            detected = min(12, a["d"], b["d"], *logicals)
            key = (k * detected * detected, detected,
                   min(18, a["d"]) + min(18, b["d"]),
                   -a["cycles"] - b["cycles"])
            if best is None or key > best[0]:
                best = (key, a, b, hx, hz, k, logicals)
        if best is not None and best[0][1] == 12 and best[5] >= 596:
            break

    if best is None:
        raise ValueError("symmetric trinomial pairs missed exact rate gates")
    key, a, b, cached_hx, cached_hz, k, logicals = best
    A = [[[G.elements[g] for g in cell] for cell in row]
         for row in a["grid"]]
    B = [[[G.elements[g] for g in cell] for cell in row]
         for row in b["grid"]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1440
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12 <= W_MAX
    assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
    assert all(len(set(cell)) == len(cell) == 3
               for grid in (A, B) for row in grid for cell in row)
    return {
        "A": A, "B": B, "idx": idx, "k": k,
        "card": (composition, model, cohort),
        "tetrahedral_central_profiles": plans,
        "requested_sharing_pair": sharing_pairs[composition],
        "requested_component_defects": desired_defects,
        "subgroup_restriction_orders": (3,4,12,4),
        "selection_reading_cap": 12,
        "target_cell": 0.400,
        "component_readings": (a["d"], b["d"]),
        "verified_localized_logical_readings": logicals,
        "component_ranks": (a["rank"], b["rank"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "selection": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Preferring the largest sixteen annihilator-plane catalogs produces k=720..750 survivors under the component gates; every distinct code in the current parent's table instead had k=690, and all five measured distances remained six.
# REFUTED: Compatibility-guided plane-orbit searches escape the old affine shape's rank/distance plateau; all 54 distinct current codes reported k=690, and all five measured codes again returned distance six.
# REFUTED: Symmetric affine trinomial retained-witness repair beats cell-0.400 fitness 58.9; the current program instead measured five k=440/readings-twelve codes, scoring 57.6 with thousands of sector-one minima.
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
