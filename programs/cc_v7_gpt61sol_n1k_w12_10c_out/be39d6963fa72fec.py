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
CONCEPT = ("Hypothesis: on (C11 semidirect_4 C5) x C5, order 275, A=B=(2,1), independent whole-cell Sidon quartics with transpose-kernel witness repair remove the parent's binomial weight-six bottleneck and produce five distance readings >=70 in cell 0.200. At the guaranteed rate these readings score >=980, above the current 965.9 frontier reading. This is a falsifiable search-budget hypothesis, not a distance certificate.")
GROUP = ("Preserve the direct-product construction rule and the exact C11:C5 factor; resize only the central cyclic factor from two to five. The factor-five shape permits q<=299, so q=55*5=275 is the largest order with this retained factor. The group now has odd order, making F2[G] semisimple and removing the parent's central-involution nilpotent mechanism. Strong odd-order factor-five beds motivate the experiment, but their provisional readings do not establish distance transfer.")
SHAPES = ("Change both sides to (2,1): n=5*275=1375, k>=275 and guaranteed rate 0.200. Both raw components are tall; their distance-bearing transpose components each have 550 variables and 275 checks. The related order-291 tall bed currently leads this rate cell. Parent minima were exclusively sector-one Acol/Bcol strips, especially Acol1/2 across Bcol0/1/2; analogous transpose-component families now occupy sector two. Verify every full k lies in [0.175,0.225), rather than inferring rate from component ranks.")
FILLING = ("Every submitted cell contains four distinct terms, giving HX=4+8=12 and HZ=4+8=12. Twenty cards cross five quotient/central occupancy plans with four cycle objectives. Require twelve distinct directed differences in both orientations of each quartic and full generation by its relative labels. Cache only scaffold-built transpose monomials; screen actual joined ranks and kernels through four. Select cycle geometry, retain genuine null words, repair implicated row blocks, and independently deepen finalists. Preferred coordinate plans have fresh unrestricted alternatives, not archived fallbacks. Final native matrices verify extraction, full rank, weight, CSS and kernel-four gates. Component readings and cycle objectives certify no quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r020_Frob55xC5_21x21_quartic_transpose_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 275

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius55 x C5 requires order 275")
    # Same direct-product rule and same order-55 Frobenius factor.
    # Only the independent central cyclic order changes from two to five.
    central_order = 5
    powers = [pow(4, b, 11) for b in range(5)]
    els = [(a, b, z) for z in range(central_order)
           for b in range(5) for a in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 5,
                int(g[2]) % central_order)

    def mul(g, h):
        a, b, z = red(g)
        c, d, w = red(h)
        return ((a + powers[b] * c) % 11,
                (b + d) % 5, (z + w) % central_order)

    def inv(g):
        a, b, z = red(g)
        bi = (-b) % 5
        return ((-powers[bi] * a) % 11, bi,
                (-z) % central_order)

    def idx(g):
        a, b, z = red(g)
        return 55 * z + 11 * b + a

    return dict(elements=els, ident=(0, 0, 0),
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


def _retired_antipodal_ctx(rng, G, idx):
    """Inactive order-110 predecessor; never called on the resized bed."""
    q = G.q
    assert q == 110
    card, cohort = idx % 20, idx // 20
    backbone, model = card // 5, card % 5
    # Unscreened pure-central proposals caused the parent's weight-four
    # losses. They now enter only through systematic_gain below.
    mixed_budget = 2 + ((backbone + model + cohort) & 1)
    seed = rng.randrange(1 << 62)

    # Cache native operations only; all matrices use lifted_matrices.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
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

    # F1024 with zeta of order eleven. This representation factors
    # through the quotient by the new central C2.
    def fmul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 1024:
                a ^= 2047
        return result

    def frobenius(a, steps):
        for _ in range(steps % 10):
            a = fmul(a, a)
        return a

    roots = [1]
    for _ in range(10):
        roots.append(fmul(roots[-1], 2))
    assert fmul(roots[-1], 2) == 1

    # T_(a,b)(w)=zeta^a*w^(4^b). Relative to the binary trace pairing,
    # the adjoint maps v to (v*zeta^a)^(2^(-2b)).
    adjoints = [
        [frobenius(fmul(1 << t, roots[g[0]]), -2 * g[1])
         for t in range(10)]
        for g in G.elements[:55]
    ]

    def adjoint_kernel(support):
        pivots, basis = {}, []
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

    lines = {}
    for a in range(1, 55):
        for b in range(1, a):
            support = (identity, b, a)
            gauges = [
                tuple(sorted(table[g][inverses[root]] for g in support))
                for root in support
            ]
            if support != min(gauges):
                continue
            basis = adjoint_kernel(support)
            if not basis:
                continue
            values = [0]
            for v in basis:
                values += [u ^ v for u in values]
            for v in values[1:]:
                lines.setdefault(v, []).append(support)

    # Different scalar functionals may specify the same support catalog.
    # Collapse those proposal duplicates; they are not separate designs.
    catalogs = {}
    for functional, supports in lines.items():
        supports = tuple(sorted(set(supports)))
        if len(supports) >= 3:
            catalogs.setdefault(supports, functional)
    catalog = [(v, supports) for supports, v in catalogs.items()]
    catalog.sort(key=lambda item: (-len(item[1]), item[0]))

    gain_bank = None

    def systematic_gain():
        """Search genuinely distinct pure-central triples, not gauge copies.

        The 55-coordinate restriction below is extracted from authoritative
        order-110 monomial lifts. It is used only as an inexpensive screen.
        Every survivor is rebuilt and screened on the full native component.
        Work limits are not claims of algebraic impossibility.
        """
        nonlocal gain_bank
        if gain_bank is not None:
            return gain_bank
        gain_bank = []
        if not catalog:
            return gain_bank

        from itertools import combinations

        h = 55
        qm = (1 << h) - 1
        full_mask = (1 << q) - 1
        source, _, _ = lifted_matrices(
            probe, [[[g] for g in range(h)]], [[[]]])
        monomials = []
        for g in range(h):
            block = [(row >> (g * q)) & full_mask
                     for row in source[:h]]
            assert all(value >> h == 0 for value in block)
            monomials.append(block)

        cell_cache, pair_cache = {}, {}
        pair_work = complete_work = 0
        tested = set()
        records = []

        def cell_record(support):
            if support not in cell_cache:
                rows = [
                    monomials[support[0]][r] ^
                    monomials[support[1]][r] ^
                    monomials[support[2]][r]
                    for r in range(h)
                ]
                columns = _syndrome_columns(rows, h)
                assert all(c.bit_count() == 3 for c in columns)
                cell_cache[support] = dict(
                    rows=rows, columns=columns,
                    rank=gf2_rank(rows),
                    legal=_pairs_no_kernel4(columns) is not None)
            return cell_cache[support]

        def compatible(left, right):
            nonlocal pair_work
            key = tuple(sorted((left, right)))
            if key not in pair_cache:
                if pair_work >= 4096:
                    return False
                pair_work += 1
                a, b = cell_record(left), cell_record(right)
                pair_cache[key] = (
                    a["legal"] and b["legal"] and
                    _pairs_no_kernel4(
                        a["columns"] + b["columns"]) is not None)
            return pair_cache[key]

        ordered = list(catalog)
        rng.shuffle(ordered)
        offset = (17 * card + 29 * cohort) % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]

        for functional, proposed in ordered:
            options = [
                support for support in sorted(set(proposed))
                if cell_record(support)["legal"]
                and cell_record(support)["rank"] <= 45
            ]
            rng.shuffle(options)
            for triple in combinations(options, 3):
                signature = tuple(sorted(triple))
                if signature in tested:
                    continue
                tested.add(signature)
                if not all(
                        compatible(signature[i], signature[j])
                        for i in range(3) for j in range(i)):
                    continue
                if complete_work >= 1024:
                    break
                complete_work += 1
                columns = [
                    c for support in signature
                    for c in cell_record(support)["columns"]
                ]
                # Pair compatibility can miss a four-word touching all
                # three blocks, so test the complete joined component.
                if _pairs_no_kernel4(columns) is None:
                    continue
                rows = [
                    sum(cell_record(support)["rows"][r] << (j * h)
                        for j, support in enumerate(signature))
                    for r in range(h)
                ]
                if gf2_rank(rows) != 45:
                    continue

                cells = [[G.elements[g] for g in support]
                         for support in signature]
                actual = component(cells, "A")
                compact = [
                    sum(((actual[r] >> (j * q)) & qm) << (j * h)
                        for j in range(3))
                    for r in range(h)
                ]
                assert compact == rows
                if gf2_rank(actual) != 90:
                    continue
                full_columns = _syndrome_columns(actual, 3 * q)
                assert all(c.bit_count() == 3 for c in full_columns)
                if _pairs_no_kernel4(full_columns) is None:
                    continue
                # Odd column degree excludes odd-weight null words.
                # The exact <=4 screen therefore also excludes five.
                records.append(dict(
                    cells=cells, rows=actual, rank=90,
                    columns=full_columns, floor=6,
                    cycles=_four_cycles(actual),
                    functional=functional, signature=signature,
                    reading=_component_reading(
                        full_columns, random.Random(seed + 910003), 12),
                    exact_gain=True))
                if len(records) >= 12:
                    break
            if len(records) >= 12 or complete_work >= 1024:
                break

        records.sort(key=lambda r: (
            r["reading"], -r["cycles"]), reverse=True)
        finalists = records[:4]
        for at, record in enumerate(finalists):
            record["reading"] = min(
                record["reading"],
                _component_reading(
                    record["columns"],
                    random.Random(seed + 1910003 + 1009 * at), 64))
        finalists.sort(key=lambda r: (
            r["reading"], -r["cycles"]), reverse=True)
        gain_bank = finalists[:3]
        return gain_bank

    def draw_a(options, budget):
        selected = rng.sample(options, 3)
        mixed = set(rng.sample(range(3), budget))
        cells = []
        for j, support in enumerate(selected):
            # An overall central translation of a cell is a gauge.
            # Only mixed placement changes its relative central labels.
            marked = rng.randrange(3) if j in mixed else -1
            cell = []
            for t, g in enumerate(support):
                a, b, _ = G.elements[g]
                cell.append((a, b, int(t == marked)))
            cells.append(cell)
        return cells

    # Each new trinomial contains an antipodal pair g,g*z. This cannot
    # arise from the parent's noncommuting trinomial prescription.
    bin_profiles = ((1, 1, 1), (1, 1, 2),
                    (1, 2, 3), (1, 2, 4))
    tri_coordinates = (1, 2, 3, 4, 0)
    central = (0, 0, 1)

    def draw_b():
        coordinates = bin_profiles[backbone]
        best = None
        for attempt in range(48):
            ratios = [(rng.randrange(11), b, 1)
                      for b in coordinates]
            directed = ratios + [G.inv(g) for g in ratios]
            if len({G.idx(g) for g in directed}) != 6:
                continue
            products = {}
            collisions = 0
            for g in directed:
                inverse = G.idx(G.inv(g))
                for h in directed:
                    if G.idx(h) == inverse:
                        continue
                    value = G.idx(G.mul(g, h))
                    collisions += products.get(value, 0)
                    products[value] = products.get(value, 0) + 1
            commuting = sum(
                G.idx(G.mul(g, h)) == G.idx(G.mul(h, g))
                for i, g in enumerate(ratios) for h in ratios[:i])
            key = (collisions, commuting)
            if best is None or key < best[0]:
                best = (key, ratios)
            if collisions == 0:
                break
        if best is None:
            raise ValueError("new central-odd backbone draw exhausted")

        cells = [[G.e, g] for g in best[1]]
        b = tri_coordinates[model]
        a = rng.randrange(1, 11) if b == 0 else rng.randrange(11)
        g = (a, b, 0)
        gz = G.mul(g, central)
        tri = [G.e, g, gz]
        assert len({G.idx(t) for t in tri}) == 3
        # p=1+(1+z)g satisfies p^2=1. Verify this directly in the
        # native group algebra rather than relying only on the formula.
        square = set()
        for u in tri:
            for v in tri:
                value = G.idx(G.mul(u, v))
                if value in square:
                    square.remove(value)
                else:
                    square.add(value)
        assert square == {identity}
        cells.append(tri)
        return cells

    def select_pool(side, options=None, budget=0):
        records, signatures = [], set()
        # The new unit trinomial guarantees full single-block rank;
        # short multi-block kernels still require actual screening.
        for trial in range(224 if side == "B" else 112):
            cells = draw_a(options, budget) if side == "A" else draw_b()
            signature = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell)) for cell in cells))
            if signature in signatures:
                continue
            signatures.add(signature)
            rows = component(cells, side)
            rank = gf2_rank(rows)
            if side == "A":
                if not (80 <= rank <= 100):
                    continue
            elif rank != q:
                continue
            records.append({
                "cells": cells, "rows": rows, "rank": rank,
                "cycles": _four_cycles(rows), "trial": trial,
            })
        records.sort(key=lambda r: (r["cycles"], r["trial"]))
        width = (3 if side == "A" else 4) * q
        screened = []
        for record in records[:40]:
            floor, columns = _component_floor(record["rows"], width, q)
            record["floor"], record["columns"] = floor, columns
            diagnostic = random.Random(seed + 1009 * int(side == "B"))
            record["reading"] = _component_reading(columns, diagnostic, 12)
            screened.append(record)
            if len(screened) >= 8 and sum(r["floor"] == 6
                                          for r in screened) >= 4:
                break

        def merit(record):
            return (record["floor"], record["reading"],
                    -record["cycles"], -abs(record["rank"] - 90))

        screened.sort(key=merit, reverse=True)
        finalists = screened[:3]
        for record in finalists:
            # Independent longer challenges only lower the reading.
            diagnostic = random.Random(
                seed + 700001 + 1009 * int(side == "B"))
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], diagnostic, 48))
        finalists.sort(key=merit, reverse=True)
        return finalists

    bb = select_pool("B")
    if not bb:
        bb = select_pool("B")
    if not bb:
        raise ValueError("no full-rank B component")

    candidates = []
    # Rank gains enter only after exact native short-kernel exclusion.
    # Their B partners use the new antipodal-unit mechanism regardless
    # of whether the finite gain search succeeds.
    for a in systematic_gain():
        for b in bb:
            key = (
                min(a["floor"], b["floor"]),
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                -a["cycles"] - b["cycles"])
            candidates.append(
                (key, a, b, a["functional"], 0))
    if catalog:
        count = len(catalog)
        # Try independent operator catalogs, not group images of a code.
        for turn in range(min(8, count)):
            functional, options = catalog[
                (card + cohort * 7 + turn) % count]
            aa = select_pool("A", options, mixed_budget)
            for a in aa:
                for b in bb:
                    key = (
                        min(a["floor"], b["floor"]),
                        min(a["reading"], b["reading"]),
                        a["reading"] + b["reading"],
                        -a["cycles"] - b["cycles"])
                    candidates.append((key, a, b, functional, mixed_budget))
            if len(candidates) >= 12:
                break

        # Do not revive the parent's unverified pure-central fallback:
        # those proposals gained k but repeatedly exposed weight four.
        if not candidates:
            other_budget = 5 - mixed_budget
            for turn in range(min(4, count)):
                functional, options = catalog[(card + turn) % count]
                for a in select_pool("A", options, other_budget):
                    for b in bb:
                        key = (
                            min(a["floor"], b["floor"]),
                            min(a["reading"], b["reading"]),
                            a["reading"] + b["reading"],
                            -a["cycles"] - b["cycles"])
                        candidates.append(
                            (key, a, b, functional, other_budget))

    # A broader bank also finds accidental common annihilators. This
    # avoids making the representation-level catalog a feasibility gate.
    if not candidates:
        broad = tuple(
            (identity, b, a)
            for a in range(1, 55) for b in range(1, a))
        for budget in (mixed_budget, 5 - mixed_budget):
            for a in select_pool("A", broad, budget):
                for b in bb:
                    key = (
                        min(a["floor"], b["floor"]),
                        min(a["reading"], b["reading"]),
                        a["reading"] + b["reading"],
                        -a["cycles"] - b["cycles"])
                    candidates.append((key, a, b, None, budget))

    # A larger deficit is useful only after the same exact short-kernel
    # floor survives. Full k is still measured below, never inferred.
    candidates.sort(key=lambda item: (
        item[0][0], -item[1]["rank"],
        item[0][1], item[0][2], item[0][3]), reverse=True)
    fallback = None
    for key, a, b, functional, used_budget in candidates:
        A, B = [a["cells"]], [b["cells"]]
        sa = [[len(cell) for cell in A[0]]]
        sb = [[len(cell) for cell in B[0]]]
        assert sa == [[3, 3, 3]] and sb == [[2, 2, 2, 3]]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(probe,
            [[[G.idx(g) for g in cell] for cell in row] for row in A],
            [[[G.idx(g) for g in cell] for cell in row] for row in B])
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1430 and 19 * n <= 40 * k < 21 * n):
            continue
        result = {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (mixed_budget, model),
            "used_mixed_budget": used_budget,
            "functional": functional,
            "component_ranks": (a["rank"], b["rank"]),
            "component_floors": (a["floor"], b["floor"]),
            "exact_pure_central_gain": a.get("exact_gain", False),
            "B_trinomial_model": "antipodal involutory unit",
            "screen": key,
        }
        if fallback is None:
            fallback = (result, hx, hz)
        if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
            continue
        if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
            continue
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return result

    if fallback is not None:
        result, hx, hz = fallback
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        result["full_kernel4_fallback"] = True
        return result
    raise ValueError("new direct-product candidates missed the measured-rate gate")


def _retired_affine_plane_ctx(rng, G, idx):
    """Unused predecessor implementation; not called on the new group."""
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
        """Select a common annihilator, not relabelings of one operator."""
        catalog = catalogs[kind]
        if not catalog:
            return None, None
        # Examine several large classes rather than betting the slot
        # on one representation-level constraint.
        count = min(16, len(catalog))
        turn = selections[kind]
        selections[kind] += 1
        functional, supports = catalog[(card + 3 * turn) % count]
        options = [
            [G.elements[t] for t in support]
            for support in supports
        ]
        return options, functional

    # New quotient profiles; the first two use only the order-five
    # complement subgroup, unlike the parent's B-trinomial recipes.
    spectra = ((2, 4), (2, 6), (3, 6), (4, 7), (3, 7))
    bin_profiles = ((1, 3, 7), (1, 3, 9), (1, 1, 3), (0, 3, 7))

    def fresh_tri():
        b1, b2 = spectra[card % 5]
        choices = [
            ((a, b1), (b, b2))
            for a in range(11) for b in range(11)
            if G._key(G.mul((a, b1), (b, b2))) !=
               G._key(G.mul((b, b2), (a, b1)))
        ]
        g, h = rng.choice(choices)
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
        return [[G.e, g] for g in fresh_bins(attempt)] + [fresh_tri()]

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

    # B is independent of the annihilator constraint and can be reused
    # while several different A annihilator classes are examined.
    bb = pool("B", None)
    if not bb:
        bb = pool("B", None)
    if not bb:
        raise ValueError("new B card failed component kernel-five screening")

    def build(kind):
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
                "functional": functional,
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
            }
        return None

    # Plane exclusion is a preference, not another infeasible hard gate.
    # Every fallback still uses three A trinomials, a new B recipe and
    # verified k>=690; no k=663 companions dilute the five-code score.
    for _ in range(4):
        answer = build("plane")
        if answer is not None:
            return answer
    for _ in range(8):
        answer = build("line")
        if answer is not None:
            return answer
    raise ValueError("no new annihilator design passed the verified rank gates")


def NEW_CTX(rng, G, idx):
    """Twenty tall quartic cards, screened using native transpose components."""
    q, width = G.q, 2 * G.q
    assert q == 275 and A_SHAPE == B_SHAPE == (2, 1)
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    plan, trade = divmod(card, 4)
    seed = rng.randrange(1 << 62)

    # These alter relative coordinate occupancies, not whole-code gauges.
    b_patterns = (
        (0, 0, 1, 2), (0, 1, 2, 3), (0, 0, 1, 1),
        (0, 0, 0, 1), (0, 0, 1, 3),
    )
    z_patterns = (
        (0, 1, 2, 3), (0, 0, 1, 2), (0, 0, 1, 1),
        (0, 1, 2, 3), (0, 0, 0, 1),
    )
    plans = ((0, 1), (1, 2), (2, 4), (3, 0), (4, 3))
    pa = plans[plan]
    pb = plans[(plan + 2) % 5][::-1]

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        # Cached native operations, not a separately implemented lift.
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    # Small batches avoid constructing q^2 long, sparse probe rows.
    # Each extracted monomial is an actual output of lifted_matrices.
    blocks = {"A": [None] * q, "B": [None] * q}
    for start in range(0, q, 10):
        batch = list(range(start, min(q, start + 10)))
        horizontal = [[[g] for g in batch]]
        _, at, _ = lifted_matrices(probe, horizontal, [[[]]])
        bt, _, _ = lifted_matrices(probe, [[[]]], horizontal)
        shift = len(batch) * q
        for j, g in enumerate(batch):
            blocks["A"][g] = [
                row >> shift for row in at[j*q:(j+1)*q]]
            blocks["B"][g] = [
                row >> shift for row in bt[j*q:(j+1)*q]]
            assert all(row >> q == 0
                       for side in ("A", "B") for row in blocks[side][g])
        del at, bt

    def whole_sidon(cell):
        if len(cell) != 4 or len(set(cell)) != 4:
            return False
        left = {table[inverses[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverses[b]]
                 for a in cell for b in cell if a != b}
        return len(left) == len(right) == 12

    def generates(cell):
        root = inverses[cell[0]]
        generators = [table[root][g] for g in cell[1:]]
        reached, todo = {identity}, [identity]
        for g in todo:
            for h in generators:
                value = table[g][h]
                if value not in reached:
                    reached.add(value)
                    todo.append(value)
        return len(reached) == q

    def fresh_cell(kind, side, broad=False):
        for attempt in range(768):
            if broad or attempt >= 256:
                cell = rng.sample(range(q), 4)
            else:
                bs = list(b_patterns[kind])
                zs = list(z_patterns[kind])
                rng.shuffle(zs)
                # Associations between quotient and central coordinates
                # change; no group-automorphism orbit creates a slot.
                cell = [G.idx((rng.randrange(11), b, z))
                        for b, z in zip(bs, zs)]
            if not whole_sidon(cell) or not generates(cell):
                continue
            root = inverses[cell[0]]
            # Remove a genuine independent row-block variable gauge.
            if side == "A":
                cell = [table[root][g] for g in cell]
            else:
                cell = [table[g][root] for g in cell]
            return sorted(cell)
        raise ValueError("fresh generating Sidon quartic unavailable")

    def signature(grid):
        # Removes cell ordering and the tall-side row permutation only.
        # This is not a full Tanner-isomorphism test.
        return tuple(sorted(tuple(sorted(row[0])) for row in grid))

    def syndrome(columns, word):
        value = 0
        for v in bits(word):
            value ^= columns[v]
        return value

    def eight_cycles(rows, columns):
        """Exact simple Tanner eight-cycles, including cycles with chords."""
        checks = [list(bits(row)) for row in rows]
        neighbors = [list(bits(column)) for column in columns]
        rooted = 0
        # The actual transpose component has a regular action within
        # each variable block. No full quantum-code symmetry is assumed.
        for root in (0, q):
            paths = {}
            for c1 in neighbors[root]:
                for v in checks[c1]:
                    if v == root:
                        continue
                    for c2 in neighbors[v]:
                        if c2 == c1:
                            continue
                        for end in checks[c2]:
                            if end == root or end == v:
                                continue
                            previous = paths.setdefault(end, [])
                            for d1, u, d2 in previous:
                                if (u != v and c1 != d1 and c1 != d2
                                        and c2 != d1 and c2 != d2):
                                    rooted += 1
                            previous.append((c1, v, c2))
        assert q * rooted % 4 == 0
        return q * rooted // 4

    def cycle_key(stats):
        four, excess, eight = stats
        if trade == 0:
            return four, excess, eight
        if trade == 1:
            return four, eight, excess
        if trade == 2:
            return four, 10 * excess + eight, excess
        return four, 30 * excess + eight, eight

    def inspect(grid, side):
        rows = [0] * q
        for j, row in enumerate(grid):
            for g in row[0]:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j*q)
        # Even column degrees force a check dependency. Require the
        # measured joined rank, not a representation-level prediction.
        rank = gf2_rank(rows)
        if rank != q - 1:
            return None
        columns = _syndrome_columns(rows, width)
        if any(c.bit_count() != 4 for c in columns):
            raise ValueError("native transpose column degree mismatch")
        if len(set(columns)) != width:
            return None

        adjacency = [0] * q
        for column in columns:
            for r in bits(column):
                adjacency[r] |= column ^ (1 << r)
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << q) - 1:
            return None

        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))).bit_count()
        # Projection excess is a heuristic, not an exact six-cycle count.
        excess = max(0, triangles - 4 * width)
        stats = (_four_cycles(rows), excess, eight_cycles(rows, columns))
        return dict(
            grid=[[list(row[0])] for row in grid],
            rows=rows, columns=columns, rank=rank,
            cycles=stats, cycle_key=cycle_key(stats),
            signature=signature(grid), d=width+1, words=[],
        )

    def kernel_probe(record, diagnostic_seed, trials):
        """Retain genuine null words; high readings certify no exclusion."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        order = list(range(width))
        retained = list(record["words"])
        reading, witness = width+1, None

        def note(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)
        for _ in range(trials):
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
                    note(word)

            light = sorted(set(basis + retained), key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    note(word ^ other)
            moves = basis + retained
            diagnostic.shuffle(moves)
            additions = []
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    note(word)
                    if not changed:
                        break
                additions.append(word)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:12]
            if reading <= 20:
                break

        if witness is None:
            raise ValueError("transpose kernel witness search failed")
        retained = sorted(set(retained + [witness]),
                          key=int.bit_count)[:12]
        assert all(syndrome(columns, word) == 0 for word in retained)
        record["words"] = retained
        record["d"] = min(record["d"], reading)
        return witness

    def optimize(kinds, side, broad):
        seen, bank, records = set(), set(), []
        offset = 100003 * int(side == "B")

        def probe_record(record, trials, turn):
            word = kernel_probe(
                record, seed + offset + 700001*turn, trials)
            bank.update(record["words"])
            bank.add(word)
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)

        def merit(record):
            # Transferred witnesses bind only after checking the NEW
            # component syndrome, not merely their coordinate weight.
            for word in bank:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"] = word.bit_count()
            return (record["d"],
                    tuple(-x for x in record["cycle_key"]))

        for trial in range(40 + 4*min(cohort, 2)):
            grid = [[fresh_cell(kind, side, broad)] for kind in kinds]
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            record = inspect(grid, side)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: r["cycle_key"])
        selected = []
        for record in records:
            if _pairs_no_kernel4(record["columns"]) is None:
                continue
            probe_record(record, 12, 0)
            selected.append(record)
            if len(selected) >= 6:
                break
        if not selected:
            return []

        current = max(selected, key=merit)
        for step in range(18):
            active = [word for word in bank
                      if syndrome(current["columns"], word) == 0]
            implicated = []
            if active:
                word = min(active, key=int.bit_count)
                implicated = [v//q for v in bits(word)]
            j = (rng.choice(implicated)
                 if implicated and rng.random() < 0.85 else rng.randrange(2))
            grid = [[list(row[0])] for row in current["grid"]]
            if step % 4 == 0:
                grid[j][0] = fresh_cell(kinds[j], side, broad)
            else:
                t = rng.randrange(4)
                old = G.elements[grid[j][0][t]]
                choices = [G.idx((a, old[1], old[2])) for a in range(11)
                           if G.idx((a, old[1], old[2])) not in grid[j][0]]
                if not choices:
                    continue
                grid[j][0][t] = rng.choice(choices)
                if not whole_sidon(grid[j][0]) or not generates(grid[j][0]):
                    continue
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if _pairs_no_kernel4(proposed["columns"]) is None:
                continue
            probe_record(proposed, 6, step+1)
            selected.append(proposed)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.12):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists = selected[:3]
        for record in finalists:
            probe_record(record, 32, 101)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent confirmation can only lower the retained reading.
            probe_record(record, 64 + 8*min(cohort, 2), 211)
        finalists.sort(key=merit, reverse=True)
        return finalists

    for restart in range(2):
        aa = optimize(pa, "A", bool(restart))
        bb = optimize(pb, "B", bool(restart))
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"] + ab[1]["d"],
            tuple(-(x+y) for x, y in
                  zip(ab[0]["cycle_key"], ab[1]["cycle_key"]))),
            reverse=True)
        for a, b in pairs:
            A = [[[G.elements[g] for g in row[0]]] for row in a["grid"]]
            B = [[[G.elements[g] for g in row[0]]] for row in b["grid"]]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 5*q == 1375
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= q and 7*n <= 40*k < 9*n):
                continue

            # Confirm every cached component against the FINAL native lift.
            # Sector two is ordered (Arow, Brow), with Brow varying first.
            actual_a = [
                sum(((hz[t] >> ((1+2*i)*q)) & mask) << (i*q)
                    for i in range(2))
                for t in range(q)
            ]
            actual_b = [(hx[t] >> q) & ((1 << width)-1)
                        for t in range(q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue
            assert max(row.bit_count() for row in hx+hz) == 12
            assert css_ok(hx, hz)
            assert all(len(cell) == len({G.idx(g) for g in cell}) == 4
                       for grid in (A, B) for row in grid for cell in row)
            return dict(
                A=A, B=B, idx=idx, k=k, target_cell=0.200,
                card=(plan, trade, cohort), broad_alternative=bool(restart),
                component_ranks=(a["rank"], b["rank"]),
                component_readings=(a["d"], b["d"]),
                component_cycles=(a["cycles"], b["cycles"]),
                filling_model="generating Sidon quartic transpose repair",
            )
    raise ValueError("fresh tall quartic pairings missed native full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected four-term cell without new draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane bank produces accepted plane rank gains beyond k=690; all 51 distinct current codes instead had k=690, and all five measured distance readings remained six.
# REFUTED: Blind pure-central quotient-annihilator rank gain improves the distance-six score on Frob55 x C2; current pure-central slots had k=720 and distance four, while all five measured winners instead had k=690 and distance six.
# REFUTED: Exact pure-central gain screening plus antipodal-unit B partners supplies accepted k=720 distance-six gains; all 45 distinct current codes had k=690, and all five measured winners remained at six.
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
