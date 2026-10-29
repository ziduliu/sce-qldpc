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
CONCEPT = ("On C97 semidirect_35 C3, order 291, A=B=(2,1), independently challenged four-term transpose components and original-span logical reduction should exceed the 0.200 incumbent fitness 965.8564. This is a falsifiable measurement hypothesis; no sampled reading certifies distance.")
GROUP = ("Retain the parent's split-semidi­rect construction rule, resizing its parameters to m=97, r=3, s=35. The new factor-five shape requires a larger order; 35 has order three modulo 97 and q=291 gives n=1455. This run's tall bed reached fitness 965.9 with readings 66..71. These measurements motivate new independent fillings, not a claim of certified distance or a new group family.")
SHAPES = ("Change A and B to (2,1): n=1455, k>=291 and guaranteed rate 0.200, with q=291 below q_max=299. Equal raw components have 291 variables and 582 checks; their distance-bearing transpose components have 582 variables and 291 checks. Parent weight-six minima occupied sector-1 Acol/Bcol strips; analogous tall-component families occupy sector-2 Arow/Brow strips. Exact full ranks enforce [0.175,0.225). Fresh labels, not archived transpose images, are submitted.")
FILLING = ("Every cell contains four distinct terms, giving HX=4+8=12 and HZ=4+8=12. Twenty cards cross five occupancy-pair prescriptions with four verified geometry objectives. Require bidirectional native Sidon supports, screen actual component ranks and kernels through four, and challenge localized, complement-fixed and unrestricted kernels. Retain witnesses across information sets and deepen finalists with syndrome-verified regular-action cancellations. Pairings use genuine sector-2 logical embeddings modulo ORIGINAL stabilizer spans and stabilizer descent before actual-k selection. Full ranks, CSS, weights and kernel-four screens are binding; finite witness searches remain upper-reading heuristics.")
HOW_AB = FILLING
FAMILY_NAME = "r020_split_m97r3s35_21x21_retained_orbit_span_v6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 291

def custom_ops(q):
    # Same split-semidi­rect rule, with resized parameters.
    if not valid_q(q):
        raise ValueError("C97:C3 requires order 291")
    m, r, s = 97, 3, 35
    assert pow(s, r, m) == 1 and s != 1
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi] * a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (2, 1)               # (rb, cb)


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


def _retired_order110_ctx(rng, G, idx):
    """Historical order-110 implementation; never called on the resized bed."""
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
    """Fresh tall cards with retained witnesses and original-span challenges."""
    import math

    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (2, 1)
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    card, cohort = idx % 20, idx // 20
    mode = card % 4
    seed = rng.randrange(1 << 62)
    # Different construction stream from every shown predecessor.
    rng = random.Random(seed ^ (0x972135D6 + 100003 * idx))
    mask = (1 << q) - 1

    profiles = (
        ((2,1,1), (2,1,1)),
        ((2,1,1), (2,2,0)),
        ((2,1,1), (3,1,0)),
        ((2,2,0), (3,1,0)),
        ((3,1,0), (3,1,0)),
    )
    plans = ((0,0), (0,2), (1,1), (1,4), (2,3))
    pa, pb = plans[card // 4]

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

    # Only authoritative unit-probe blocks are assembled below.
    unit = [[[identity]]]
    monomials = [[[g]] for g in range(q)]
    mono = {}
    for side in ("A", "B"):
        if side == "A":
            _, source, _ = lifted_matrices(P, monomials, unit)
        else:
            source, _, _ = lifted_matrices(P, unit, monomials)
        assert len(source) == q
        mono[side] = [
            [(row >> ((1 + g) * q)) & mask for row in source]
            for g in range(q)
        ]

    def sidon(cell):
        if len(set(cell)) != 4:
            return False
        left, right = set(), set()
        for g in cell:
            for h in cell:
                if g != h:
                    left.add(table[inverse[g]][h])
                    right.add(table[g][inverse[h]])
        return len(left) == len(right) == 12

    def fresh_cell(occupancy):
        for _ in range(1024):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, count in zip(fibers, occupancy):
                cell.extend(G.idx((a, b))
                            for a in rng.sample(range(97), count))
            if sidon(cell):
                return cell
        raise ValueError("four-term native Sidon filling exhausted")

    def fresh(profile):
        return [[fresh_cell(occupancy)]
                for occupancy in profiles[profile]]

    def signature(grid):
        return tuple(tuple(sorted(row[0])) for row in grid)

    def component(grid, side):
        rows = []
        for r in range(q):
            row = 0
            for j in range(2):
                block = 0
                for g in grid[j][0]:
                    block ^= mono[side][g][r]
                row |= block << (j * q)
            rows.append(row)
        return rows

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def dependencies(columns, positions):
        pivots, basis = {}, []
        for c in positions:
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
        return basis

    def geometry(columns):
        uses, four = {}, 0
        adjacency = [0] * q
        forced = 0
        for column in columns:
            ns = list(bits(column))
            forced += len(ns) * (len(ns)-1) * (len(ns)-2) // 6
            for i, u in enumerate(ns):
                adjacency[u] |= column ^ (1 << u)
                for v in ns[:i]:
                    key = (v, u)
                    previous = uses.get(key, 0)
                    four += previous
                    uses[key] = previous + 1
        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u+1))):
                triangles += (
                    neighbors & adjacency[v] & (-1 << (v+1))
                ).bit_count()
        six = max(0, triangles - forced)
        # This projection statistic is not an exact Tanner C8 count.
        squares = 0
        if mode in (1, 3):
            for u in range(q):
                for v in range(u):
                    count = (adjacency[u] & adjacency[v]).bit_count()
                    squares += count * (count-1) // 2
            squares //= 2
        objectives = (
            (four, six, squares),
            (four, squares, six),
            (four, 8*six+squares, six),
            (four, 24*six+squares, squares),
        )
        return objectives[mode], (four, six, squares)

    def inspect(grid, side):
        rows = component(grid, side)
        rank = gf2_rank(rows)
        if rank < q - 3:
            return None
        columns = _syndrome_columns(rows, width)
        if 0 in columns or len(set(columns)) != width:
            return None
        cell_ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]))
        if min(cell_ranks) < q - 8:
            return None
        key, stats = geometry(columns)
        words = set()
        # Manageable single-block restrictions are enumerated exactly.
        for j in range(2):
            basis = dependencies(columns, range(j*q, (j+1)*q))
            if len(basis) <= 12:
                word, shortest = 0, None
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length()-1]
                    if shortest is None or word.bit_count() < shortest.bit_count():
                        shortest = word
                if shortest is not None:
                    words.add(shortest)
            else:
                words.update(sorted(basis, key=int.bit_count)[:4])
        return dict(grid=[[list(row[0])] for row in grid],
                    rows=rows, columns=columns, rank=rank,
                    cell_ranks=cell_ranks, signature=signature(grid),
                    geometry=key, cycles=stats, words=words,
                    d=min((w.bit_count() for w in words),
                          default=width+1), side=side)

    # Coordinate restrictions only: compressed syndromes are sums of
    # authoritative columns. No independent action lift is implemented.
    partitions = {}
    subgroup = [G.idx((0, b)) for b in range(3)]
    for side in ("A", "B"):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == "A" else table[h][root]
                for h in subgroup
            }
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        partitions[side] = [
            tuple(j*q+c for c in orbit)
            for j in range(2) for orbit in orbits
        ]

    def challenge(record, trials, turn, fixed=False):
        diagnostic = random.Random(
            seed + 700001*turn + 100003*int(record["side"] == "B")
            + sum((i+1)*g for i, cell in enumerate(record["signature"])
                  for g in cell))
        original = record["columns"]
        if fixed:
            parts = partitions[record["side"]]
            columns = [syndrome(original, sum(1 << c for c in part))
                       for part in parts]
        else:
            columns = original
        order = list(range(len(columns)))
        retained = [] if fixed else sorted(
            record["words"], key=int.bit_count)[:24]

        def expand(word):
            if not fixed:
                return word
            actual = 0
            for c in bits(word):
                for position in parts[c]:
                    actual |= 1 << position
            return actual

        def observe(word):
            if not word:
                return
            actual = expand(word)
            weight = actual.bit_count()
            if weight <= record["d"] + 4:
                assert syndrome(original, actual) == 0
                record["words"].add(actual)
            record["d"] = min(record["d"], weight)

        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = dependencies(columns, order)
            light = sorted(set(retained + basis),
                           key=int.bit_count)[:16]
            additions = []
            for word in light:
                observe(word)
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= word.bit_count():
                        additions.append(candidate)
            moves = retained + basis
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                observe(word)
                additions.append(word)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:24]
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:32])

    def orbit_challenge(record):
        columns, side = record["columns"], record["side"]
        sources = sorted(record["words"], key=int.bit_count)[:2]
        comparisons = sorted(record["words"], key=int.bit_count)[:12]
        additions = []
        for source in sources:
            positions = [divmod(c, q) for c in bits(source)]
            for multiplier in rng.sample(range(q), 48):
                image = 0
                for block, h in positions:
                    c = (table[h][multiplier] if side == "A"
                         else table[multiplier][h])
                    image |= 1 << (block*q+c)
                assert syndrome(columns, image) == 0
                for other in comparisons:
                    word = image ^ other
                    if word:
                        record["d"] = min(record["d"], word.bit_count())
                        additions.append(word)
        record["words"].update(additions)
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:32])

    def pool(profile, side, restart):
        records, seen = [], set()
        for _ in range(56):
            rec = inspect(fresh(profile), side)
            if rec is not None and rec["signature"] not in seen:
                seen.add(rec["signature"])
                records.append(rec)
        records.sort(key=lambda r: r["geometry"])
        if not records:
            return []
        # Geometry repair changes actual terms, never a group gauge.
        current = records[0]
        for step in range(48):
            grid = [[list(row[0])] for row in current["grid"]]
            j = rng.randrange(2)
            if step % 5 == 0:
                grid[j][0] = fresh_cell(profiles[profile][j])
            else:
                at = rng.randrange(4)
                a, b = G.elements[grid[j][0][at]]
                grid[j][0][at] = G.idx((rng.randrange(97), b))
                if not sidon(grid[j][0]):
                    continue
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            old_cost = 1000*current["geometry"][0] + current["geometry"][1]
            new_cost = 1000*proposed["geometry"][0] + proposed["geometry"][1]
            temperature = q * (8*(1-step/48.0)**2 + 0.2)
            if new_cost <= old_cost or rng.random() < math.exp(
                    min(0.0, (old_cost-new_cost)/temperature)):
                current = proposed
                records.append(proposed)

        records.sort(key=lambda r: r["geometry"])
        selected = []
        for rec in records:
            if _pairs_no_kernel4(rec["columns"]) is None:
                continue
            challenge(rec, 4, 10+restart, fixed=True)
            challenge(rec, 8, 20+restart)
            selected.append(rec)
            if len(selected) >= 6:
                break
        if not selected:
            return []

        def merit(rec):
            return (rec["d"], tuple(-x for x in rec["geometry"]), rec["rank"])

        current = max(selected, key=merit)
        for step in range(16):
            word = min(current["words"], key=int.bit_count)
            active = [c//q for c in bits(word)]
            j = rng.choice(active) if active else rng.randrange(2)
            grid = [[list(row[0])] for row in current["grid"]]
            grid[j][0] = fresh_cell(profiles[profile][j])
            proposed = inspect(grid, side)
            if proposed is None or _pairs_no_kernel4(proposed["columns"]) is None:
                continue
            challenge(proposed, 3, 100+step+31*restart, fixed=True)
            challenge(proposed, 6, 200+step+31*restart)
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        finalists = sorted(selected, key=merit, reverse=True)[:3]
        for rec in finalists:
            challenge(rec, 12, 501+restart, fixed=True)
            challenge(rec, 32, 601+restart)
            orbit_challenge(rec)
            challenge(rec, 24, 701+restart)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    def span_pivots(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(a, b, hx, hz, n):
        """Actual tall-sector witnesses modulo original stabilizer spans."""
        readings = []
        for rec, opposite, stabilizers, kind in (
                (a, hz, hx, "X"), (b, hx, hz, "Z")):
            pivots = span_pivots(stabilizers)
            candidates = set()
            for word in sorted(rec["words"], key=int.bit_count)[:4]:
                for fixed in range(2):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i*q)) & mask)
                            << ((1+2*i+fixed)*q) for i in range(2))
                    else:
                        actual = word << ((1+2*fixed)*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            best = n+1
            for word in sorted(candidates, key=int.bit_count)[:4]:
                reducers = list(stabilizers)
                rng.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        proposed = word ^ row
                        if proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    nearby = sorted(
                        stabilizers, key=lambda row: (row & word).bit_count(),
                        reverse=True)[:24]
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = word ^ first ^ second
                            if proposed.bit_count() < word.bit_count():
                                word, changed = proposed, True
                    if not changed:
                        break
                assert word and outside(word, pivots)
                assert all((word & row).bit_count() % 2 == 0
                           for row in opposite)
                best = min(best, word.bit_count())
            readings.append(best)
        return tuple(readings)

    best, best_key = None, None
    for restart in range(2):
        aa, bb = pool(pa, "A", restart), pool(pb, "B", restart)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"]+ab[1]["d"]), reverse=True)
        for a, b in proposals:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1455 and k >= q and 7*n <= 40*k < 9*n):
                continue
            actual_a = [
                sum(((hz[r] >> ((1+2*i)*q)) & mask) << (i*q)
                    for i in range(2)) for r in range(q)]
            actual_b = [(hx[r] >> q) & ((1 << width)-1)
                        for r in range(q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue
            assert max(row.bit_count() for row in hx+hz) == 12
            assert css_ok(hx, hz)
            logicals = logical_readings(a, b, hx, hz, n)
            detected = min(a["d"], b["d"], *logicals)
            key = (k*detected*detected, detected,
                   a["d"]+b["d"],
                   tuple(-(x+y) for x, y in
                         zip(a["geometry"], b["geometry"])))
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "A": [[[G.elements[g] for g in row[0]]]
                          for row in a["grid"]],
                    "B": [[[G.elements[g] for g in row[0]]]
                          for row in b["grid"]],
                    "idx": idx, "k": k, "card": (pa, pb, mode, cohort),
                    "component_readings": (a["d"], b["d"]),
                    "component_ranks": (a["rank"], b["rank"]),
                    "component_cycles": (a["cycles"], b["cycles"]),
                    "verified_logical_upper_readings": logicals,
                    "selection_quality": key,
                }
        if best is not None:
            return best
    raise ValueError("fresh tall pairings failed verified rate and kernel gates")


def CELL(rng, G, side, i, j, ctx):
    """Return every selected four-term cell of the new tall shapes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Preferring the largest sixteen annihilator-plane catalogs produces k=720..750 survivors under the component gates; every distinct code in the current parent's table instead had k=690, and all five measured distances remained six.
# REFUTED: Compatibility-guided annihilator-plane orbit searches produce accepted k=720 or 750 improvements on the order-110 bed; all 54 distinct current codes instead had k=690 and the five measured readings remained six.
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
