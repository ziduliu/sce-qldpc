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
CONCEPT = ("C7 semidirect_3 C12, order 84, A=(1,3), B=(2,5): generating cubic A operators and independently repaired degree-three B columns should produce five new distance readings >=12 and improve cell 0.350 fitness above 47.8. Failure of that five-code target refutes the experiment; readings remain provisional upper bounds.")
GROUP = ("Jump from the order-110 affine group to C7:C12 with action three and order 84. The destination (1,3)x(2,5) bed reached fitness 47.8 with measured readings 12,12,12,12,10. This is evidence for another independently constructed portfolio, not distance transfer or a certificate. The action has order six and the complement has order twelve; generating trinomials and long relative binomials can be proposed without an affine-annihilator constraint.")
SHAPES = ("Keep A=(1,3), change B to (2,5): q=84 gives n=1428, k>=504 and guaranteed rate 6/17=0.352941, targeting cell 0.350; the shape permits q<=88. Components have 252 variables/84 checks and 420 variables/168 checks. Parent minima were entirely sector one, especially Acol1/2 across Bcol0/1/2. Two B check-block rows and exclusively degree-three B columns replace that single-row binomial bottleneck. Exact full ranks enforce 0.325<=k/n<0.375.")
FILLING = ("Twenty cards cross five generating-trinomial prescriptions with four B multiplicity profiles. A counts (3,3,3); B column totals are three and row totals seven/eight, giving HX=9+3=12 and HZ<=8+3=11. Cache only scaffold-built monomial blocks. Independently optimize exact four-cycle and projection-excess geometry, detect component kernels through four, retain genuine null witnesses, repair implicated whole columns, and deepen every finalist with information-set and budgeted parity-completion searches. Odd degree excludes odd component kernels, but no component screen certifies quantum distance. Pair by actual k and original-span-verified localized logical upper readings, then verify native-lift agreement, target rate, weights and CSS. No archived filling or orbit image is deliberately reused.")
HOW_AB = FILLING
FAMILY_NAME = "r035_C7_C12_action3_13x25_cubic_binding_columnrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect_3 C12 requires order 84")
    # (a,b)=x^a y^b; y*x*y^-1=x^3.
    # ord_7(3)=6 divides 12, so this is a valid semidirect product.
    powers = tuple(pow(3, b, 7) for b in range(12))
    els = [(a, b) for b in range(12) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 12)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 7, (b + d) % 12)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 12
        return ((-powers[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0),
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


def _retired_affine_regular_ctx(rng, G, idx):
    """Inactive historical order-110 implementation; never called."""
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
    lines, planes, annihilators = {}, {}, {}
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
            # Retain the COMPLETE representation-level kernel. It only
            # proposes labels; authoritative screens still use the lift.
            values = [0]
            for v in basis:
                values += [u ^ v for u in values]
            values = sorted(values[1:])
            annihilators[support] = sum(1 << v for v in values)
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

    # Repeated quotient fibers introduce a relative order-eleven pair.
    # These are different support mechanisms, not translated copies of
    # the parent's three-distinct-fiber trinomial prescriptions.
    spectra = (0, 2, 4, 6, 8)
    bin_profiles = ((1, 3, 7), (1, 1, 7),
                    (1, 3, 9), (3, 7, 9))

    def fresh_tri():
        b = spectra[card % 5]
        translations = range(1, 11) if b == 0 else range(11)
        a, c = rng.sample(list(translations), 2)
        g, h = (a, b), (c, b)
        assert len({G.idx(G.e), G.idx(g), G.idx(h)}) == 3
        if b:
            assert G._key(G.mul(g, h)) != G._key(G.mul(h, g))
        # The homogeneous t=0 card deliberately tests commuting
        # translation trinomials instead of imposing noncommutation.
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

    a_banks = {}
    regular_cache = {}
    intersection_cache = {}
    compatibility_cache = {}
    mask = (1 << q) - 1

    # Obtain authoritative left-action monomials once. All later
    # component assembly is GF2 linearity of these exact probe outputs.
    source, _, _ = lifted_matrices(
        probe, [[[g] for g in range(q)]], [[[]]])
    assert len(source) == q
    monomials = [
        [(row >> (g * q)) & mask for row in source]
        for g in range(q)
    ]

    def bit_positions(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def xor_selected(words, selector):
        answer = 0
        for position in bit_positions(selector):
            answer ^= words[position]
        return answer

    def dependency_basis(vectors):
        """Dependencies in original input coordinates, over GF2."""
        pivots, basis = {}, []
        for i, value in enumerate(vectors):
            word = 1 << i
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

    def regular_cell(support):
        if support not in regular_cache:
            rows = [
                monomials[support[0]][r] ^
                monomials[support[1]][r] ^
                monomials[support[2]][r]
                for r in range(q)
            ]
            columns = _syndrome_columns(rows, q)
            left = dependency_basis(rows)
            assert len(left) == q - gf2_rank(rows)
            assert all(xor_selected(rows, word) == 0 for word in left)
            regular_cache[support] = {
                "rows": rows, "columns": columns, "left": left,
                "short_ok": None,
            }
        return regular_cache[support]

    def intersect_with_cell(basis, rec):
        """Intersect a regular left subspace with ker(rec['rows']^T).

        No representation dimension or independently implemented lift
        enters this computation.
        """
        if not basis:
            return []
        images = [xor_selected(rec["rows"], word) for word in basis]
        return [
            xor_selected(basis, dependency)
            for dependency in dependency_basis(images)
        ]

    def pair_basis(left, right):
        key = tuple(sorted((left, right)))
        if key not in intersection_cache:
            first, second = regular_cell(left), regular_cell(right)
            if len(first["left"]) > len(second["left"]):
                first, second = second, first
            intersection_cache[key] = intersect_with_cell(
                first["left"], second)
        return intersection_cache[key]

    def individually_legal(support):
        rec = regular_cell(support)
        if rec["short_ok"] is None:
            rec["short_ok"] = (
                _pairs_no_kernel4(rec["columns"]) is not None)
        return rec["short_ok"]

    def compatible(left, right):
        key = tuple(sorted((left, right)))
        if key not in compatibility_cache:
            compatibility_cache[key] = (
                individually_legal(left) and individually_legal(right)
                and _pairs_no_kernel4(
                    regular_cell(left)["columns"] +
                    regular_cell(right)["columns"]) is not None)
        return compatibility_cache[key]

    def systematic_A(kind):
        """Search actual regular intersections, including line extensions.

        All catalogs are proposal catalogs. Work caps limit construction
        effort; exhausting them is not an algebraic impossibility result.
        """
        if kind in a_banks:
            return a_banks[kind]

        targets = (30, 20) if kind == "plane" else (10,)
        allowed = {80, 90} if kind == "plane" else {80, 90, 100}
        records, tested = [], set()
        ordered = list(catalogs["line"])
        rng.shuffle(ordered)
        if ordered:
            offset = (17 * card + 29 * (idx // 20)) % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]

        pair_cap = 60000 if kind == "plane" else 30000
        triple_cap = 3000 if kind == "plane" else 1500
        pair_work = triple_work = 0

        for target in targets:
            for functional, proposed in ordered:
                options = [
                    support for support in set(proposed)
                    if len(regular_cell(support)["left"]) >= target
                ]
                if len(options) < 3:
                    continue
                # Independent triples, not right-gauge copies; shuffle
                # removes the parent's persistent lexicographic shortlist.
                options.sort()
                rng.shuffle(options)
                adjacent = [set() for _ in options]

                for i, left in enumerate(options):
                    for j, right in enumerate(options[:i]):
                        key = tuple(sorted((left, right)))
                        if key not in intersection_cache:
                            if pair_work >= pair_cap:
                                continue
                            pair_work += 1
                        common = pair_basis(left, right)
                        if len(common) < target:
                            continue
                        if not compatible(left, right):
                            continue
                        adjacent[i].add(j)
                        adjacent[j].add(i)

                for i in range(len(options)):
                    for j in sorted(v for v in adjacent[i] if v > i):
                        third = adjacent[i] & adjacent[j]
                        for k_ in sorted(v for v in third if v > j):
                            signature = tuple(sorted(
                                (options[i], options[j], options[k_])))
                            if signature in tested:
                                continue
                            if triple_work >= triple_cap:
                                break
                            tested.add(signature)
                            triple_work += 1
                            common = intersect_with_cell(
                                pair_basis(signature[0], signature[1]),
                                regular_cell(signature[2]))
                            rank = q - len(common)
                            if rank not in allowed or len(common) < target:
                                continue

                            columns = [
                                column for support in signature
                                for column in regular_cell(support)["columns"]
                            ]
                            # Odd column degree excludes odd-weight null
                            # words. Complete <=4 exclusion therefore
                            # excludes component kernels through five.
                            if _pairs_no_kernel4(columns) is None:
                                continue
                            rows = [
                                sum(regular_cell(support)["rows"][r] << (j_ * q)
                                    for j_, support in enumerate(signature))
                                for r in range(q)
                            ]
                            assert gf2_rank(rows) == rank
                            assert all(column.bit_count() == 3
                                       for column in columns)
                            records.append({
                                "cells": [
                                    [G.elements[g] for g in support]
                                    for support in signature
                                ],
                                "columns": columns, "rank": rank,
                                "cycles": _four_cycles(rows),
                                "functional": functional,
                                "dimension": len(common),
                                "signature": signature,
                            })
                        if triple_work >= triple_cap:
                            break
                    if triple_work >= triple_cap:
                        break
                if triple_work >= triple_cap or len(records) >= 64:
                    break
            if records or triple_work >= triple_cap:
                break

        records.sort(key=lambda rec: (
            rec["rank"], rec["cycles"], rec["signature"]))
        if records:
            best_rank = records[0]["rank"]
            near = [rec for rec in records if rec["rank"] == best_rank]
            # Preserve a cycle-quality preference without repeatedly
            # returning the same few canonical A triples in every slot.
            near = near[:32]
            near = rng.sample(near, min(8, len(near)))
            for rec in near:
                rec["reading"] = _component_reading(
                    rec["columns"], rng, 12)
            near.sort(key=lambda rec: (
                rec["reading"], -rec["cycles"]), reverse=True)
            near = near[:3]
            for rec in near:
                rec["reading"] = min(
                    rec["reading"],
                    _component_reading(rec["columns"], rng, 48))
            near.sort(key=lambda rec: (
                rec["reading"], -rec["cycles"]), reverse=True)
            records = near

        a_banks[kind] = records
        return records

    def build(kind):
        aa = systematic_A(kind)
        if not aa:
            return None
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
                "functional": a["functional"],
                "regular_annihilator_dimension": a["dimension"],
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
            }
        return None

    # Each systematic bank is constructed once. Repeating a failed
    # random draw is replaced by testing distinct canonical joins.
    answer = build("plane")
    if answer is not None:
        return answer
    answer = build("line")
    if answer is not None:
        return answer
    raise ValueError("no systematic annihilator join passed verified gates")


def NEW_CTX(rng, G, idx):
    """Twenty cubic cards with binding witnesses and native verification."""
    import math

    q = G.q
    assert q == 84 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    model, topology = card % 5, card // 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    sa = [[3, 3, 3]]
    profiles = (
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((3, 1, 1, 1, 1), (0, 2, 2, 2, 2)),
        ((3, 2, 1, 1, 0), (0, 1, 2, 2, 3)),
        ((3, 3, 1, 0, 0), (0, 0, 2, 3, 3)),
    )
    sb = [list(row) for row in profiles[topology]]
    assert [sum(row) for row in sb] == [7, 8]
    assert all(sb[0][j] + sb[1][j] == 3 for j in range(5))
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverse[g]

    P = Probe()
    # Dummy zero partners are extraction probes, never submitted codes.
    monomials = [[[g] for g in range(q)]]
    blocks = {}
    source, _, _ = lifted_matrices(P, monomials, [[[]]])
    blocks["A"] = [[(row >> (g*q)) & mask for row in source]
                   for g in range(q)]
    _, source, _ = lifted_matrices(P, [[[]]], monomials)
    blocks["B"] = [[(row >> (g*q)) & mask for row in source]
                   for g in range(q)]

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        answer = 0
        for v in bits(word):
            answer ^= columns[v]
        return answer

    def generates(a, b):
        reached, todo = {ident}, [ident]
        for g in todo:
            for h in (a, b):
                value = table[g][h]
                if value not in reached:
                    reached.add(value)
                    todo.append(value)
        return len(reached) == q

    def sidon(cell):
        if len(set(cell)) != len(cell):
            return False
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        wanted = len(cell) * (len(cell) - 1)
        return len(left) == len(right) == wanted

    quotient_plans = ((1, 2), (1, 4), (1, 5), (2, 3), (3, 4))
    libraries = []
    for b, d in quotient_plans:
        library = set()
        for a in range(7):
            for c in range(7):
                g, h = G.idx((a, b)), G.idx((c, d))
                cell = (ident, g, h)
                if not sidon(cell) or table[g][h] == table[h][g]:
                    continue
                if not generates(g, h):
                    continue
                # Remove anchored right-column-gauge presentations.
                canonical = min(
                    tuple(sorted(table[t][inverse[root]] for t in cell))
                    for root in cell)
                library.add(canonical)
        if not library:
            raise ValueError("empty generating cubic prescription")
        libraries.append(sorted(library))

    long_ratios = []
    normal_ratios = []
    for g in range(q):
        value = ident
        for order in range(1, q + 1):
            value = table[value][g]
            if value == ident:
                break
        if order == 12:
            long_ratios.append(g)
        if order == 7:
            normal_ratios.append(g)
    assert long_ratios and normal_ratios

    def draw_cell(count, side, i, j):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            # Relative orders vary across cards, not through a global
            # automorphism or a relabeling of an existing design.
            normal = (topology & 1) and (i + j + model + cohort) % 4 == 0
            ratio = rng.choice(normal_ratios if normal else long_ratios)
            root = rng.randrange(q)
            return [root, table[root][ratio]]
        policy = (model + 2*j + 3*i + cohort) % 5
        cell = list(rng.choice(libraries[policy]))
        if side == "B":
            root = rng.randrange(q)
            cell = [table[root][g] for g in cell]
        return cell

    def fresh(side):
        if side == "B":
            return [[draw_cell(count, side, i, j)
                     for j, count in enumerate(row)]
                    for i, row in enumerate(sb)]
        cells, differences = [], set()
        for j in range(3):
            best = None
            for _ in range(24):
                cell = draw_cell(3, "A", 0, j)
                diff = {table[a][inverse[b]]
                        for a in cell for b in cell if a != b}
                key = len(diff & differences)
                if best is None or key < best[0]:
                    best = (key, cell, diff)
                if key == 0:
                    break
            cells.append(best[1])
            differences.update(best[2])
        return [cells]

    def inspect(grid, side):
        """Linear assembly of authoritative monomial blocks only."""
        rows = []
        width = len(grid[0]) * q
        for cells in grid:
            for r in range(q):
                row = 0
                for j, cell in enumerate(cells):
                    block = 0
                    for g in cell:
                        block ^= blocks[side][g][r]
                    row |= block << (j*q)
                rows.append(row)
        columns = _syndrome_columns(rows, width)
        assert all(column.bit_count() == 3 for column in columns)

        uses, four = {}, 0
        adjacency = [0] * len(rows)
        for column in columns:
            checks = list(bits(column))
            for t, a in enumerate(checks):
                adjacency[a] |= column ^ (1 << a)
                for b in checks[:t]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0) + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a + 1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b + 1))).bit_count()
        # Projection excess is a heuristic when four-cycles are present.
        excess = max(0, triangles - width)
        cost = (500 + 100*model)*four + (1 + cohort % 3)*excess
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cost=cost, cycles=(four, excess), signature=signature,
                    d=width + 1, words=[])

    def challenge(rec, trials, offset, budget=0):
        """Binding genuine kernel witnesses; budget stops prove nothing."""
        local = random.Random(seed + offset)
        columns = rec["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(rec["words"])
        best = min(retained, key=int.bit_count) if retained else None

        def observe(word):
            nonlocal best
            if word and (best is None or
                         word.bit_count() < best.bit_count()):
                best = word

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
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
                    observe(word)
            light = sorted(set(retained + basis), key=int.bit_count)[:16]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= 16:
                        additions.append(candidate)
            moves = basis + retained
            local.shuffle(moves)
            for word in light[:5]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
                additions.append(word)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:24]
            if best is not None and best.bit_count() <= 4:
                break
        assert best is not None

        if budget and best.bit_count() > 6:
            singles, pairs = {}, {}
            incidence = [0] * len(rec["rows"])
            for v, value in enumerate(columns):
                singles.setdefault(value, 1 << v)
                for r in bits(value):
                    incidence[r] |= 1 << v
                for u in range(v):
                    pairs.setdefault(value ^ columns[u],
                                     (1 << v) | (1 << u))
            priorities = [local.random() for _ in columns]
            roots = list(range(0, width, q))
            local.shuffle(roots)
            all_variables = (1 << width) - 1
            cap = min(11, best.bit_count() - 1)
            for root in roots:
                remaining = [max(1, budget // len(roots))]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    if value == 0:
                        observe(word)
                        return
                    left = min(cap, best.bit_count()-1) - word.bit_count()
                    if left <= 0 or value.bit_count() > 3*left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        observe(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        observe(word | pair)
                    left = min(cap, best.bit_count()-1) - word.bit_count()
                    if left <= 2:
                        return
                    chosen, smallest = 0, width + 1
                    for r in bits(value):
                        available = incidence[r] & allowed
                        count = available.bit_count()
                        if not count:
                            return
                        if count < smallest:
                            chosen, smallest = available, count
                    choices = list(bits(chosen))
                    choices.sort(key=lambda v: (
                        -(columns[v] & value).bit_count(), priorities[v]))
                    live = allowed
                    for v in choices:
                        bit = 1 << v
                        live &= ~bit
                        visit(value ^ columns[v], word | bit, live)
                        if remaining[0] <= 0:
                            break

                # Regular component symmetry supplies orbit anchors.
                # No symmetry of the full quantum code is assumed.
                allowed = all_variables & ~((1 << (root+1))-1)
                visit(columns[root], 1 << root, allowed)

        assert best and syndrome(columns, best) == 0
        rec["d"] = min(rec["d"], best.bit_count())
        rec["words"] = sorted(set(retained + [best]),
                              key=lambda w: (w.bit_count(), w))[:24]
        assert all(syndrome(columns, w) == 0 for w in rec["words"])

    def prepare(rec, side):
        rec["rank"] = gf2_rank(rec["rows"])
        target = q if side == "A" else 2*q
        if rec["rank"] < target - (6 if side == "A" else 8):
            return False
        rec["free4"] = _pairs_no_kernel4(rec["columns"]) is not None
        # All columns have odd degree three, so odd kernel weights,
        # including five, are impossible. This is a COMPONENT statement.
        return True

    def merit(rec):
        return (rec["free4"], min(14, rec["d"]),
                -rec["cost"], -rec["rank"])

    def pool(side):
        support = sa if side == "A" else sb
        offset = 100003 * int(side == "B")
        bank, seen = {}, set()

        def remember(rec):
            if rec["signature"] in seen:
                return
            seen.add(rec["signature"])
            bank[rec["signature"]] = rec
            if len(bank) > 10:
                worst = max(bank, key=lambda s: bank[s]["cost"])
                del bank[worst]

        starts = [inspect(fresh(side), side) for _ in range(40)]
        starts.sort(key=lambda r: r["cost"])
        mutable = [(i, j) for i, row in enumerate(support)
                   for j, count in enumerate(row) if count]
        for initial in starts[:2]:
            current = initial
            remember(current)
            for step in range(96):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(mutable)
                grid[i][j] = draw_cell(support[i][j], side, i, j)
                proposed = inspect(grid, side)
                temperature = q*(6*(1-step/96.0)**2 + 0.1)
                delta = proposed["cost"] - current["cost"]
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(current)

        candidates = []
        for at, rec in enumerate(sorted(bank.values(),
                                        key=lambda r: r["cost"])):
            if not prepare(rec, side):
                continue
            challenge(rec, 10, offset + 1009*at, 4000)
            candidates.append(rec)
        if not candidates:
            raise ValueError("no rank-eligible cubic component")
        current = max(candidates, key=merit)

        # Replace implicated column cells independently. This changes
        # relative labels, rather than applying a common column gauge.
        for step in range(20):
            grid = copy_grid(current["grid"])
            word = rng.choice(current["words"][:6])
            occupied = sorted({v//q for v in bits(word)})
            j = rng.choice(occupied)
            for i in range(len(grid)):
                grid[i][j] = draw_cell(support[i][j], side, i, j)
            rec = inspect(grid, side)
            if rec["signature"] in seen or not prepare(rec, side):
                continue
            seen.add(rec["signature"])
            rec["words"] = [w for w in current["words"]
                            if syndrome(rec["columns"], w) == 0]
            challenge(rec, 8, offset + 300007 + 1013*step, 4000)
            candidates.append(rec)
            if merit(rec) >= merit(current):
                current = rec

        candidates.sort(key=merit, reverse=True)
        finalists = candidates[:4]
        for at, rec in enumerate(finalists):
            challenge(rec, 40, offset + 700001 + 1009*at, 12000)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:3]
        for at, rec in enumerate(finalists):
            challenge(rec, 24, offset + 1900003 + 1013*at, 12000)
        finalists.sort(key=merit, reverse=True)
        return finalists

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
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def localized(a, b, hx, hz, n):
        """Upper readings of actual logicals in the ORIGINAL quotient."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            found = set()
            for word in rec["words"][:4]:
                for location in range(5 if kind == "Z" else 3):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j+location)*q) for j in range(3))
                    else:
                        actual = word << (5*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        found.add(actual)
            local = random.Random(seed + (3100009 if kind == "Z"
                                         else 4100011))
            best = n + 1
            for actual in sorted(found, key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:32]
                for _ in range(96):
                    first, second = local.sample(nearby, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                best = min(best, actual.bit_count())
            # No found embedding is not an exclusion certificate.
            readings.append(best)
        return tuple(readings)

    aa, bb = pool("A"), pool("B")
    proposals = []
    for a in aa:
        for b in bb:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1428 and k >= 504
                    and 325*n <= 1000*k < 375*n):
                continue
            actual_a = [
                sum(((hx[r] >> (5*j*q)) & mask) << (j*q)
                    for j in range(3)) for r in range(q)]
            actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logicals = localized(a, b, hx, hz, n)
            detected = min(14, a["d"], b["d"], *logicals)
            quality = (
                a["free4"] and b["free4"],
                k*detected*detected, detected,
                -a["cost"]-b["cost"])
            proposals.append((quality, a, b, hx, hz, k, logicals))
    proposals.sort(key=lambda item: item[0], reverse=True)
    if not proposals:
        raise ValueError("new cubic pairs missed cell 0.350")

    chosen = proposals[0]
    full4 = False
    for proposal in proposals:
        _, a, b, hx, hz, k, logicals = proposal
        if (_pairs_no_kernel4(_syndrome_columns(hx, 1428)) is not None
                and _pairs_no_kernel4(
                    _syndrome_columns(hz, 1428)) is not None):
            chosen, full4 = proposal, True
            break

    quality, a, b, cached_hx, cached_hz, k, logicals = chosen
    A = [[[G.elements[g] for g in cell] for cell in row]
         for row in a["grid"]]
    B = [[[G.elements[g] for g in cell] for cell in row]
         for row in b["grid"]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1428
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return dict(
        A=A, B=B, idx=idx, k=k, target_cell=0.350,
        card=(model, topology, cohort), quality=quality,
        component_readings=(a["d"], b["d"]),
        component_ranks=(a["rank"], b["rank"]),
        component_cycles=(a["cycles"], b["cycles"]),
        component_kernel4_free=(a["free4"], b["free4"]),
        verified_localized_logical_readings=logicals,
        full_kernel4_free=full4,
    )


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cubic or deliberate zero cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Random plane-bank preference with rank-100 fallbacks reliably finds additional rank gain on this bed; every distinct current slot reported k=690 and k-kb=30, and all five measured distances remained six.
# REFUTED: Bounded systematic joins prioritized by F1024 common-space dimension produce accepted rank gains beyond k=690; all 22 distinct current codes instead had k=690, and all five measured distances remained six.
# REFUTED: The current canonical-join and B-card selection reliably supplies one inequivalent filling per slot; many slots were isomorphic duplicates and four slots failed labeling gates.
# REFUTED: Complete regular-annihilator joins with repeated-fiber B cards yield accepted k=720 or 750 gains while retaining six; all 51 distinct current codes instead had k=690, and all five measured readings remained six.
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
