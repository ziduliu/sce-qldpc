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
CONCEPT = ("On (C7 semidirect_2 C3) x C4, order 84, A=(1,3), B=(2,5), generating C3-balanced A trinomials should retain distance-twelve readings while producing five new codes with k>504, raising cell 0.350 fitness above 50.8235. Actual full ranks and measured distances can refute this hypothesis.")
GROUP = ("Jump from the order-110 affine group to (C7:C3) x C4, order 84. Different orders already prove the groups are nonisomorphic. The destination group is isomorphic to the run's Z7:Z12(s=2), whose exact (1,3)x(2,5) bed reached 50.8 with k=504 and readings twelve. Its centre has order four. This measured bed motivates fresh supports and quotient-balanced rank gain, not certified distance transfer.")
SHAPES = ("Change to A=(1,3), B=(2,5): n=17*84=1428, k>=6*84=504, guaranteed rate 6/17=0.352941, and q_max=88. Components have 252 and 420 variables with 84 and 168 checks. The parent's weight-six minima were entirely sector one, concentrated on Acol1/2 across Bcol0/1/2. A two-row B component replaces its three-binomial bottleneck. Native full ranks enforce 0.325<=k/n<0.375; k=510 with reading twelve would score 51.4286.")
FILLING = ("A has three weight-three cells, each meeting all C3 cosets and generating the complete order-84 group. This supplies a common quotient-annihilator proposal without copying the affine representation catalog. Twenty cards cross four relative C4 profiles with five inequivalent B count grids. B rows have weight seven or eight, and columns have weight two or three, so HX<=9+3=12 and HZ<=8+3=11. All labels are independently optimized, not automorphism images. Scaffold-supplied monomial blocks assemble authoritative components; exact kernel-four exclusion, retained witnesses, label repair and independent deeper challenges select finalists. Original-coordinate logical strips are checked against original stabilizer spans and reduced by original checks. Actual k, assembly agreement, row weights and CSS gate submission. Finite witness searches certify neither distance nor Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Frob21xC4_13x25_balanced_rankgain_strip_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius21 x C4 requires order 84")
    action = (1, 2, 4)
    els = [(a, b, z) for z in range(4)
           for b in range(3) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 3, int(g[2]) % 4)

    def mul(g, h):
        a, b, z = red(g)
        c, d, w = red(h)
        return ((a + action[b] * c) % 7,
                (b + d) % 3, (z + w) % 4)

    def inv(g):
        a, b, z = red(g)
        bi = (-b) % 3
        return ((-action[bi] * a) % 7, bi, (-z) % 4)

    def idx(g):
        a, b, z = red(g)
        return 21 * z + 7 * b + a

    return dict(elements=els, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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
    """Compact A-left/B-right components extracted from verified rows."""
    mask = (1 << q) - 1
    ar = []
    for r in range(q):
        source = hx[r]  # Arow=0, Bcol=0.
        row = 0
        for j in range(3):
            row |= ((source >> (j * 4 * q)) & mask) << (j * q)
        ar.append(row)
    bmask = (1 << (4 * q)) - 1
    br = [hz[r] & bmask for r in range(q)]  # Acol=0, Brow=0.
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


def _retired_affine_ctx(rng, G, idx):
    """Historical implementation; never called on the new group or shapes."""
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

    # Odd repeated fibers change actual relative supports. The last two
    # cards test distinct odd fibers instead of the parent's translation
    # plus nonprimitive-complement prescription.
    spectra = ((1, 1), (3, 3), (9, 9), (1, 7), (3, 7))
    bin_profiles = ((1, 1, 1), (1, 1, 3),
                    (1, 3, 3), (3, 3, 3))

    def fresh_tri(attempt):
        b1, b2 = spectra[card % 5]
        for _ in range(128):
            a, c = rng.randrange(11), rng.randrange(11)
            g, h = (a, b1), (c, b2)
            if g == h:
                continue
            if table[G.idx(g)][G.idx(h)] == table[G.idx(h)][G.idx(g)]:
                continue
            assert len({G.idx(G.e), G.idx(g), G.idx(h)}) == 3
            return [G.e, g, h]
        raise ValueError("odd-fiber noncommuting trinomial exhausted")

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
        trials = (256 if side == "B" else
                  192 if kind == "plane" else 128)
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
        survivors = survivors[:4]
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

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length() - 1)
            if old is None:
                return True
            word ^= old
        return False

    def bin_six_words(record):
        """Bounded genuine six-cycles of the verified binomial graph.

        Check zero is an orbit root. Native left translates provide
        additional witnesses, but the finite enumeration is not exclusion.
        """
        columns = record["columns"]
        adjacency = [[] for _ in range(q)]
        for c, value in enumerate(columns[:3 * q]):
            ns = list(bits(value))
            assert len(ns) == 2
            u, v = ns
            adjacency[u].append((v, c))
            adjacency[v].append((u, c))
        found = set()
        cap = 128

        def visit(vertex, depth, visited, word):
            if len(found) >= cap:
                return
            for neighbor, edge in adjacency[vertex]:
                bit = 1 << edge
                if word & bit:
                    continue
                if depth == 1:
                    if neighbor == 0:
                        candidate = word | bit
                        if candidate.bit_count() == 6:
                            assert syndrome(columns, candidate) == 0
                            found.add(candidate)
                    continue
                if visited & (1 << neighbor):
                    continue
                visit(neighbor, depth - 1,
                      visited | (1 << neighbor), word | bit)
                if len(found) >= cap:
                    return

        visit(0, 6, 1, 0)
        return sorted(found)

    def component_words(record, side):
        """Accumulate actual null words; do not infer logical distance."""
        if "logical_probe_words" in record:
            return record["logical_probe_words"]
        columns = record["columns"]
        width = len(columns)
        diagnostic = random.Random(
            BASE_SEED + 1000003 * idx + 100003 * int(side == "B"))
        order = list(range(width))
        retained = set()
        for turn in range(24):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            light = sorted(set(basis) | retained,
                           key=lambda w: (w.bit_count(), w))[:16]
            grown = set(light)
            for first in light:
                for second in basis:
                    candidate = first ^ second
                    if candidate:
                        grown.add(candidate)
            retained = set(sorted(
                retained | grown,
                key=lambda w: (w.bit_count(), w))[:32])
            if turn >= 7 and min(
                    (w.bit_count() for w in retained), default=width) <= 6:
                break
        words = sorted(retained, key=lambda w: (w.bit_count(), w))[:24]
        if side == "B":
            words += bin_six_words(record)
        words = sorted(set(words), key=lambda w: (w.bit_count(), w))
        assert all(syndrome(columns, word) == 0 for word in words)
        record["logical_probe_words"] = words
        return words

    def logical_challenge(a, b, hx, hz, n):
        """Original-coordinate nontrivial logical witnesses only.

        A returned n+1 means a miss. Capping the selection objective at
        eight is conservative bookkeeping, not a distance certificate.
        """
        mask = (1 << q) - 1
        reading = n + 1
        tested = trivial = 0
        witness = None

        # Challenge the parent's persistent B-binomial bottleneck first.
        cases = (
            ("B", b, hz, hx, 3),
            ("A", a, hx, hz, 4),
        )
        diagnostic = random.Random(BASE_SEED + 2700011 + 7919 * idx)

        for side, record, opposite, stabilizers, locations in cases:
            pivots = span(stabilizers)
            words = component_words(record, side)
            translations = list(range(q))
            diagnostic.shuffle(translations)
            identity = G.idx(G.e)
            translations.remove(identity)
            translations.insert(0, identity)

            # First challenge several translates of every retained word.
            # Only promising pairs receive the remaining translations.
            stages = (translations[:4], translations[4:])
            for stage, shifts in enumerate(stages):
                active = words if stage == 0 else [
                    word for word in words if word.bit_count() <= 8]
                for shift in shifts:
                    for word in active:
                        moved = 0
                        for c in bits(word):
                            block, coordinate = divmod(c, q)
                            image = (table[shift][coordinate] if side == "B"
                                     else table[coordinate][shift])
                            moved |= 1 << (block * q + image)
                        assert syndrome(record["columns"], moved) == 0

                        for location in range(locations):
                            if side == "B":
                                actual = moved << (location * 4 * q)
                            else:
                                actual = sum(
                                    ((moved >> (j * q)) & mask)
                                    << ((4 * j + location) * q)
                                    for j in range(3))
                            tested += 1
                            if not outside(actual, pivots):
                                trivial += 1
                                continue
                            assert all((actual & row).bit_count() % 2 == 0
                                       for row in opposite)

                            # Original stabilizers preserve the logical
                            # coset. These finite reductions only find words.
                            if actual.bit_count() <= 14:
                                nearby = sorted(
                                    stabilizers,
                                    key=lambda row: (row & actual).bit_count(),
                                    reverse=True)[:24]
                                for _ in range(2):
                                    before = actual.bit_count()
                                    for row in nearby:
                                        proposed = actual ^ row
                                        if proposed.bit_count() < actual.bit_count():
                                            actual = proposed
                                    best_word = actual
                                    best_weight = actual.bit_count()
                                    for i, first in enumerate(nearby):
                                        partial = actual ^ first
                                        for second in nearby[:i]:
                                            proposed = partial ^ second
                                            weight = proposed.bit_count()
                                            if weight < best_weight:
                                                best_word, best_weight = proposed, weight
                                    actual = best_word
                                    if actual.bit_count() == before:
                                        break
                            assert actual and outside(actual, pivots)
                            assert all((actual & row).bit_count() % 2 == 0
                                       for row in opposite)
                            if actual.bit_count() < reading:
                                reading = actual.bit_count()
                                witness = actual
                            if reading <= 6:
                                return reading, tested, trivial, witness
        return reading, tested, trivial, witness

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
        best = None
        best_key = None

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
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert _syndrome_columns(actual_a, 3 * q) == a["columns"]
            assert _syndrome_columns(actual_b, 4 * q) == b["columns"]
            logical, tested, trivial, witness = logical_challenge(
                a, b, hx, hz, n)
            detected = min(8, logical)
            key = (
                k * detected * detected, detected,
                min(12, a["reading"]) + min(12, b["reading"]),
                -a["cycles"] - b["cycles"])
            if best_key is not None and key <= best_key:
                continue
            best_key = key
            best = {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": (kind, card // 5, card % 5),
                "functional": a.get("functional", functional),
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
                "verified_logical_upper_reading": logical,
                "logical_placements_tested": tested,
                "stabilizer_placements_found": trivial,
                "logical_witness": witness,
                "pairing_merit": key,
            }
        return best

    # Compare several actual annihilator constraints against the same
    # independently new B bank. Do not stop at the first legal pairing.
    best = build("plane")
    for _ in range(3):
        answer = build("line")
        if answer is not None and (
                best is None or answer["pairing_merit"] > best["pairing_merit"]):
            best = answer
    if best is not None:
        return best
    raise ValueError("no homology-challenged annihilator pairing passed ranks")


def NEW_CTX(rng, G, idx):
    """Twenty balanced-rank/count cards with binding logical challenges."""
    q = G.q
    assert q == 84 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    frame, topology = card // 5, card % 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1

    # Different numbers of pure cubic columns distinguish the first
    # four supports. The fifth also changes a column's total degree.
    b_counts = (
        ((1, 1, 1, 2, 2), (2, 2, 2, 1, 1)),
        ((1, 1, 1, 1, 3), (2, 2, 2, 2, 0)),
        ((3, 0, 1, 1, 2), (0, 3, 2, 2, 1)),
        ((3, 3, 0, 1, 1), (0, 0, 3, 2, 2)),
        ((1, 1, 1, 2, 2), (1, 2, 2, 1, 1)),
    )
    a_frames = (
        ((1, 1), (1, 2), (2, 3)),
        ((1, 1), (1, 3), (1, 3)),
        ((1, 2), (1, 2), (2, 3)),
        ((1, 3), (2, 3), (1, 2)),
    )
    sa = [[3, 3, 3]]
    sb = [list(row) for row in b_counts[topology]]
    assert _grid_weights(sa, sb) <= 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache group operations only, never an independently written lift.
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

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    # Both action conventions come directly from the scaffold.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 14):
        ids = list(range(start, min(start + 14, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j * q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j * q)) & mask for row in zz])

    orders = []
    for g in range(q):
        value, order = identity, 0
        while True:
            value = table[value][g]
            order += 1
            if value == identity:
                break
            assert order <= q
        orders.append(order)
    long_ratios = [g for g in range(q) if orders[g] >= 6]

    def generating(cell):
        root = inverse[cell[0]]
        generators = [table[root][g] for g in cell[1:]]
        reached, pending = {identity}, [identity]
        for value in pending:
            for g in generators:
                image = table[value][g]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    def cubic(zs):
        # One term per C3 coset is an actual quotient constraint.
        # Relative central coordinates, not overall cell gauges, vary.
        choices = []
        for a in range(7):
            for b in range(7):
                cell = [identity, G.idx((a, 1, zs[0])),
                        G.idx((b, 2, zs[1]))]
                u, v = cell[1:]
                if table[u][v] == table[v][u]:
                    continue
                left = {table[inverse[g]][h]
                        for g in cell for h in cell if g != h}
                right = {table[g][inverse[h]]
                         for g in cell for h in cell if g != h}
                if len(left) == len(right) == 6 and generating(cell):
                    choices.append(cell)
        if not choices:
            raise ValueError("balanced generating cubic unavailable")
        return list(rng.choice(choices))

    def fresh_cell(side, i, j):
        if side == "A":
            return cubic(a_frames[frame][j])
        count = sb[i][j]
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            ratio = rng.choice(long_ratios)
            return [root, table[root][ratio]]
        # Independently chosen relative central geometry, rather than
        # a common automorphism applied to an existing B component.
        zs = rng.choice(((1, 1), (1, 2), (1, 3), (2, 3)))
        cell = cubic(zs)
        root = rng.randrange(q)
        return [table[root][g] for g in cell]

    def fresh(side):
        shape = A_SHAPE if side == "A" else B_SHAPE
        return [[fresh_cell(side, i, j) for j in range(shape[1])]
                for i in range(shape[0])]

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        width = len(grid[0]) * q
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                assert len(cell) == len(set(cell))
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i * q + r] ^= value << (j * q)
        return rows, _syndrome_columns(rows, width)

    def kernel_probe(columns, trials, diagnostic, retained=()):
        """Accumulate genuine words; misses provide no lower bound."""
        width = len(columns)
        beam = sorted({w for w in retained if w},
                      key=lambda w: (w.bit_count(), w))[:20]
        assert all(syndrome(columns, w) == 0 for w in beam)
        order = list(range(width))
        for turn in range(trials):
            diagnostic.shuffle(order)
            if beam and turn % 5 == 4:
                anchor = diagnostic.choice(beam[:5])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            light = sorted(set(basis + beam),
                           key=lambda w: (w.bit_count(), w))[:12]
            grown = list(light)
            for first in light:
                for second in basis:
                    word = first ^ second
                    if word:
                        grown.append(word)
                if len(grown) > 256:
                    grown = sorted(set(grown), key=int.bit_count)[:40]
            moves = list(basis) + beam
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                grown.append(word)
            beam = sorted({w for w in grown + beam if w},
                          key=lambda w: (w.bit_count(), w))[:20]
            if beam and beam[0].bit_count() <= 4:
                break
        if not beam:
            raise ValueError("component kernel search found no word")
        assert all(syndrome(columns, word) == 0 for word in beam)
        return beam[0].bit_count(), beam

    def inspect(grid, side):
        rows, columns = component(grid, side)
        rank = gf2_rank(rows)
        if side == "A":
            # Quotient balance proposes the deficit; this measures it.
            if not q - 10 <= rank <= q - 2:
                return None
            assert all(c.bit_count() == 3 for c in columns)
        elif rank < 2 * q - 12:
            return None
        if _pairs_no_kernel4(columns) is None:
            return None
        return {
            "grid": [[list(cell) for cell in row] for row in grid],
            "rows": rows, "columns": columns, "rank": rank,
            "cycles": _four_cycles(rows), "words": [],
            "d": len(columns) + 1, "signature": signature(grid),
        }

    def pool(side):
        side_id = int(side == "B")
        seen, records, bank = set(), [], set()

        def challenge(record, trials, turn):
            transferred = [w for w in bank
                           if syndrome(record["columns"], w) == 0]
            diagnostic = random.Random(
                seed + 100003 * side_id + 700001 * turn)
            reading, words = kernel_probe(
                record["columns"], trials, diagnostic,
                record["words"] + transferred)
            record["d"] = min(record["d"], reading)
            record["words"] = words
            bank.update(words[:4])
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)

        def merit(record):
            for word in bank:
                if syndrome(record["columns"], word) == 0:
                    record["d"] = min(record["d"], word.bit_count())
            rank_key = -record["rank"] if side == "A" else record["rank"]
            return (min(12, record["d"]), min(18, record["d"]),
                    rank_key, -record["cycles"])

        for restart in range(2):
            for _ in range(56 if side == "A" else 44):
                grid = fresh(side)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                record = inspect(grid, side)
                if record is not None:
                    records.append(record)
            records.sort(key=lambda r: (r["cycles"], r["signature"]))
            selected = records[:10]
            if not selected:
                continue
            for record in selected:
                challenge(record, 8, 1 + 100 * restart)
            current = max(selected, key=merit)

            for step in range(16):
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                word = current["words"][0]
                active = sorted({c // q for c in bits(word)})
                j = rng.choice(active) if active else rng.randrange(len(grid[0]))
                occupied = [i for i in range(len(grid)) if grid[i][j]]
                i = rng.choice(occupied)
                grid[i][j] = fresh_cell(side, i, j)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                if proposed["cycles"] > current["cycles"] + 2 * q:
                    continue
                challenge(proposed, 6, 11 + step + 100 * restart)
                selected.append(proposed)
                pk, ck = merit(proposed), merit(current)
                if pk > ck or (pk[:2] == ck[:2] and rng.random() < 0.12):
                    current = proposed

            finalists = sorted(selected, key=merit, reverse=True)[:4]
            for record in finalists:
                challenge(record, 32, 211 + 100 * restart)
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:3]
            for record in finalists:
                challenge(record, 48 + 8 * min(cohort, 2),
                          401 + 100 * restart)
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("new component card exhausted authoritative screens")

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length() - 1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_strips(a, b, hx, hz, n):
        """Binding upper readings in the original stabilizer quotient."""
        readings = []
        for side, record, opposite, stabilizers, locations in (
                ("A", a, hx, hz, 5),
                ("B", b, hz, hx, 3)):
            pivots = span(stabilizers)
            candidates = set()
            for word in record["words"][:6]:
                assert syndrome(record["columns"], word) == 0
                for location in range(locations):
                    if side == "A":
                        actual = sum(
                            ((word >> (j * q)) & mask)
                            << ((5 * j + location) * q)
                            for j in range(3))
                    else:
                        actual = word << (5 * location * q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n + 1
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:24]
                for _ in range(3):
                    before = actual.bit_count()
                    for row in nearby:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    best_word = actual
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            proposed = partial ^ second
                            if proposed.bit_count() < best_word.bit_count():
                                best_word = proposed
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool("A"), pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(12, ab[0]["d"], ab[1]["d"]),
        min(18, ab[0]["d"]) + min(18, ab[1]["d"]),
        -ab[0]["rank"], ab[1]["rank"],
        -ab[0]["cycles"] - ab[1]["cycles"]), reverse=True)

    best = None
    for a, b in pairs[:5]:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 1428
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 504 and 13 * n <= 40 * k < 15 * n):
            continue

        # Check every cached component row against the final native lift.
        actual_a = [
            sum(((row >> (j * 5 * q)) & mask) << (j * q)
                for j in range(3))
            for row in hx[:q]
        ]
        actual_b = [
            row & ((1 << (5 * q)) - 1)
            for row in hz[:2 * q]
        ]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        logicals = logical_strips(a, b, hx, hz, n)
        detected = min(12, a["d"], b["d"], *logicals)
        quality = (
            k * detected * detected, detected,
            min(18, a["d"]) + min(18, b["d"]),
            -a["cycles"] - b["cycles"])
        answer = {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (frame, topology, cohort),
            "component_ranks": (a["rank"], b["rank"]),
            "component_readings": (a["d"], b["d"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "verified_logical_strip_readings": logicals,
            "pairing_merit": quality,
        }
        if best is None or quality > best[0]:
            best = (quality, answer, hx, hz)
    if best is None:
        raise ValueError("balanced card missed verified rank-gain/rate gate")
    quality, answer, hx, hz = best
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert css_ok(hx, hz)
    return answer


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cell; no additional random draws."""
    cell = list(ctx[side][i][j])
    assert len(cell) <= 5 and len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Preferring the largest sixteen annihilator-plane catalogs produces k=720..750 survivors under the component gates; every distinct code in the current parent's table instead had k=690, and all five measured distances remained six.
# REFUTED: Compatibility-guided plane-orbit searches with translation-containing B trinomials produce additional accepted rank gain or remove the weight-six family; all 54 distinct current codes had k=690 and all five measured minima remained six, entirely in sector 1.
# REFUTED: Odd repeated-fiber B supports and original-stabilizer six-cycle pairing remove the affine bed's weight-six bottleneck; all five measured current codes still had k=690 and minimum reading six, entirely in sector one.
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
