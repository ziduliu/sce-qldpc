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
CONCEPT = ("C7 semidirect_3 C12, order 84, A=(1,3), B=(2,5): mostly cubic component columns with detected-witness repair should supply five new cell-0.350 codes with measured readings >=12 and improve fitness 47.802. Failure of that five-code target refutes this experiment; readings remain provisional upper bounds.")
GROUP = ("Jump from order-110 C11:C10 to order-84 C7:C12 with action three, whose order modulo seven is six. Its central complement involution and its order distinguish it from the parent. The related action-two order-84 bed reached fitness 47.802 on (1,3)x(2,5), motivating a different action rather than claiming distance transfer. All binary screening blocks and final matrices come from lifted_matrices.")
SHAPES = ("A=(1,3), B=(2,5), q=84 give n=1428, k>=504 and guaranteed rate 6/17=0.352941, targeting the weaker cell 0.350. The shapes permit q<=88. Components have 252 variables/84 checks and 420 variables/168 checks. Parent weight-six minima occupied sector 1, especially Acol1/2 across Bcol0/1/2. Both backbones are replaced; exact final ranks enforce 0.325<=k/n<0.375.")
FILLING = ("A counts are (3,3,3). B column degrees are (2,3,3,3,3), with both row totals seven, giving HX<=9+3=12 and HZ<=7+3=10. Twenty cards cross five B zero budgets with four quotient/relative-order prescriptions. Cache only scaffold-derived monomial blocks; optimize actual component cycle statistics and penalize surviving detected kernel words. Exact detection through four, bounded parity completion and independent accumulated-witness challenges select finalists. Final pairings use actual k and challenge nontrivial localized logicals with original stabilizers. No sampled component or quantum reading is a distance certificate.")
HOW_AB = FILLING
FAMILY_NAME = "r035_m7r12s3_13x25_cubic_surviving_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7:C12 with action three requires order 84")
    # x^7=y^12=1, y*x*y^-1=x^3.
    # 3 has order six modulo seven, so this is a valid C12 action.
    powers = [pow(3, b, 7) for b in range(12)]
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
    """Inactive order-110 implementation; never called on the new group."""
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
    """Twenty new support/action cards with binding witness retests."""
    from functools import lru_cache
    from math import exp, gcd

    q = G.q
    assert q == 84 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    zero_budget, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    identity = G.idx(G.e)

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length() - 1
            word ^= low

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]

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

    # Both lift conventions are supplied by the authoritative helper.
    monomials = [[[g] for g in range(q)]]
    left, _, _ = lifted_matrices(P, monomials, [[[]]])
    _, right, _ = lifted_matrices(P, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in left]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in right]
              for g in range(q)],
    }
    del left, right

    orders = []
    for g in range(q):
        value = identity
        for order in range(1, q + 1):
            value = table[value][g]
            if value == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid native group order")

    bin_orders = (
        (7, 12, 14), (6, 12), (4, 7, 14), (4, 6, 12, 14)
    )
    bin_pools = [[g for g in range(q) if orders[g] in target]
                 for target in bin_orders]
    tri_profiles = ((0, 1, 2), (0, 0, 1),
                    (0, 1, 3), (0, 2, 5))

    def sidon(cell):
        if len(set(cell)) != len(cell):
            return False
        left_d, right_d = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left_d.add(table[inverse[a]][b])
                    right_d.add(table[a][inverse[b]])
        required = len(cell) * (len(cell) - 1)
        return len(left_d) == len(right_d) == required

    def fresh_cell(count, mode):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            ratio = rng.choice(bin_pools[mode])
            return [root, table[root][ratio]]

        assert count == 3
        profile = tri_profiles[mode]
        for _ in range(192):
            phase = rng.randrange(12)
            native = [(rng.randrange(7), (phase + b) % 12)
                      for b in profile]
            cell = [G.idx(g) for g in native]
            if not sidon(cell):
                continue
            relative = [table[inverse[cell[0]]][g] for g in cell[1:]]
            if table[relative[0]][relative[1]] == \
                    table[relative[1]][relative[0]]:
                continue
            quotient_gcd = 12
            for g in relative:
                quotient_gcd = gcd(quotient_gcd, G.elements[g][1])
            if quotient_gcd != 1:
                continue
            if any(orders[table[inverse[a]][b]] <= 3
                   for i, a in enumerate(cell) for b in cell[:i]):
                continue
            return cell
        # Proposal preferences never override distinctness or row budgets.
        return rng.sample(range(q), 3)

    def support_b():
        degrees = (2, 3, 3, 3, 3)

        @lru_cache(None)
        def feasible(j, upper, zeros):
            if upper < 0 or zeros < 0:
                return False
            if j == 5:
                return upper == zeros == 0
            degree = degrees[j]
            return any(feasible(
                j + 1, upper - a,
                zeros - int(a == 0 or a == degree))
                for a in range(degree + 1))

        assert feasible(0, 7, zero_budget)
        upper, zeros = 7, zero_budget
        grid = [[], []]
        for j, degree in enumerate(degrees):
            choices = [a for a in range(degree + 1)
                       if feasible(j + 1, upper - a,
                                   zeros - int(a == 0 or a == degree))]
            a = rng.choice(choices)
            grid[0].append(a)
            grid[1].append(degree - a)
            upper -= a
            zeros -= int(a == 0 or a == degree)
        assert [sum(row) for row in grid] == [7, 7]
        assert _grid_weights([[3, 3, 3]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[3, 3, 3]], grid)
        return grid

    def fresh(side):
        counts = [[3, 3, 3]] if side == "A" else support_b()
        mode = (policy + int(side == "B") + cohort) % 4
        return [[fresh_cell(count, mode) for count in row]
                for row in counts]

    def inspect(grid, side):
        """GF2-linear assembly of scaffold-derived blocks only."""
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        columns = _syndrome_columns(rows, nc*q)

        uses, four = {}, 0
        adjacency = [0] * len(rows)
        forced = 0
        for column in columns:
            neighbors = list(bits(column))
            degree = len(neighbors)
            forced += degree * (degree - 1) * (degree - 2) // 6
            for at, a in enumerate(neighbors):
                adjacency[a] |= column ^ (1 << a)
                for b in neighbors[:at]:
                    key = (b, a)
                    previous = uses.get(key, 0)
                    four += previous
                    uses[key] = previous + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a + 1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b + 1))
                ).bit_count()
        # This is a projection heuristic, not an exact C6 count with C4.
        excess = max(0, triangles - forced)
        peak = max((a.bit_count() for a in adjacency), default=0)
        cost = (600 * (1 + policy) * four + excess
                + (0, 2*peak, excess, 6*peak)[policy])
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cycles=(four, excess, peak), cost=cost)

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def short_word(columns):
        """Return a real kernel witness or exclude all weights through four."""
        singles, pairs = {}, {}
        for c, value in enumerate(columns):
            word = 1 << c
            if value == 0:
                return word
            old = singles.get(value)
            if old is not None:
                return old ^ word
            singles[value] = word
        for a, value in enumerate(columns):
            for b in range(a):
                combined = value ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(combined)
                if old is None:
                    old = pairs.get(combined)
                if old is not None:
                    return old ^ word
                pairs[combined] = word
        return None

    def probe(record, trials, diagnostic_seed, completion_budget):
        """Accumulate genuine kernel words; budget stops prove nothing."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(record.get("words", ()))
        witness = short_word(columns)

        def observe(word):
            nonlocal witness
            if word and (witness is None or
                         word.bit_count() < witness.bit_count()):
                witness = word

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
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
                    observe(word)
            light = sorted(set(retained + basis), key=int.bit_count)[:16]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= 16:
                        additions.append(candidate)
            moves = retained + basis
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
                observe(word)
                additions.append(word)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:28]
            if witness is not None and witness.bit_count() <= 4:
                break

        assert witness is not None
        cap = min(12, witness.bit_count() - 1)
        if cap >= 5 and completion_budget:
            singles, pairs = {}, {}
            incidence = [0] * len(record["rows"])
            for c, value in enumerate(columns):
                singles.setdefault(value, 1 << c)
                for r in bits(value):
                    incidence[r] |= 1 << c
                for b in range(c):
                    pairs.setdefault(value ^ columns[b],
                                     (1 << c) | (1 << b))
            roots = list(range(0, width, q))
            diagnostic.shuffle(roots)
            priorities = [diagnostic.random() for _ in columns]
            all_variables = (1 << width) - 1

            for root in roots:
                remaining = [max(1, completion_budget // len(roots))]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    if not value:
                        observe(word)
                        return
                    left = min(cap, witness.bit_count()-1) - word.bit_count()
                    if left <= 0 or value.bit_count() > 3*left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        observe(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        observe(word | pair)
                    left = min(cap, witness.bit_count()-1) - word.bit_count()
                    if left <= 2:
                        return
                    chosen, smallest = 0, width + 1
                    for r in bits(value):
                        available = incidence[r] & allowed
                        count = available.bit_count()
                        if count == 0:
                            return
                        if count < smallest:
                            chosen, smallest = available, count
                    candidates = list(bits(chosen))
                    candidates.sort(key=lambda c: (
                        -(columns[c] & value).bit_count(), priorities[c]))
                    live = allowed
                    for c in candidates:
                        low = 1 << c
                        live &= ~low
                        visit(value ^ columns[c], word | low, live)
                        if remaining[0] <= 0:
                            break

                visit(columns[root], 1 << root,
                      all_variables & ~((1 << (root+1)) - 1))

        assert witness and syndrome(columns, witness) == 0
        record["words"] = sorted(set(retained + [witness]),
                                 key=int.bit_count)[:32]
        record["word"] = witness
        record["d"] = min(record.get("d", width + 1), witness.bit_count())

    def component_pool(side, restart):
        offset = 100003*int(side == "B") + 300007*restart
        known, tested, screened = [], set(), []
        mode = (policy + int(side == "B") + cohort) % 4

        def loss(record):
            penalty = sum(
                4000 * max(1, 13-word.bit_count())**2
                for word in known
                if syndrome(record["columns"], word) == 0)
            return record["cost"] + penalty

        def optimize(grid, steps):
            current = inspect(grid, side)
            best = current
            current_cost = best_cost = loss(current)
            mutable = [(i, j) for i, row in enumerate(grid)
                       for j, cell in enumerate(row) if cell]
            for step in range(steps):
                positions = mutable
                active = [word for word in known
                          if syndrome(current["columns"], word) == 0]
                if active and step % 3 != 2:
                    occupied = {c//q for c in bits(rng.choice(active))}
                    targeted = [p for p in mutable if p[1] in occupied]
                    if targeted:
                        positions = targeted
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                grid[i][j] = fresh_cell(len(old), mode)
                proposed = inspect(grid, side)
                proposed_cost = loss(proposed)
                temperature = 90*(1-step/steps)**2 + 0.2
                if proposed_cost <= current_cost or rng.random() < exp(
                        min(0.0, (current_cost-proposed_cost)/temperature)):
                    current, current_cost = proposed, proposed_cost
                    if proposed_cost < best_cost:
                        best, best_cost = proposed, proposed_cost
                else:
                    grid[i][j] = old
            return best

        def refresh(record):
            for word in known:
                if word.bit_count() < record["d"] and \
                        syndrome(record["columns"], word) == 0:
                    record["d"], record["word"] = word.bit_count(), word
                    record["words"].append(word)

        def merit(record):
            refresh(record)
            return (record["free4"], record["d"],
                    -loss(record), record["rank"])

        pool = [optimize(fresh(side), 28) for _ in range(6)]
        for stage in range(2):
            selected = []
            for record in sorted(pool, key=loss):
                signature = tuple(tuple(sorted(cell))
                                  for row in record["grid"] for cell in row)
                if signature in tested:
                    continue
                tested.add(signature)
                record["free4"] = short_word(record["columns"]) is None
                record["rank"] = gf2_rank(record["rows"])
                probe(record, 8, seed + offset + 1009*stage, 6000)
                selected.append(record)
                screened.append(record)
                word = record["word"]
                if word.bit_count() <= 12 and word not in known:
                    known.append(word)
                    known.sort(key=int.bit_count)
                    del known[32:]
            selected.sort(key=merit, reverse=True)
            if stage == 0:
                starts = selected[:2]
                pool = [optimize(copy_grid(r["grid"]), 56) for r in starts]
                pool += [optimize(fresh(side), 32) for _ in range(4)]

        if not screened:
            return []
        screened.sort(key=merit, reverse=True)
        finalists = screened[:3]
        for record in finalists:
            probe(record, 32, seed + offset + 700001, 16000)
            word = record["word"]
            if word.bit_count() <= 12 and word not in known:
                known.append(word)
        finalists.sort(key=merit, reverse=True)
        for record in finalists[:2]:
            probe(record, 48, seed + offset + 1700021, 24000)
            word = record["word"]
            if word.bit_count() <= 12 and word not in known:
                known.append(word)
        finalists.sort(key=merit, reverse=True)
        return finalists

    def localized_reading(a, b, hx, hz, n):
        """Verify nontrivial localized logicals and reduce with true checks."""
        def span(rows):
            pivots = {}
            for value in rows:
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = value
                        break
                    value ^= old
            return pivots

        def outside(value, pivots):
            while value:
                old = pivots.get(value.bit_length() - 1)
                if old is None:
                    return True
                value ^= old
            return False

        reading = n + 1
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(set(record["words"] + [record["word"]]),
                           key=int.bit_count)[:3]
            retained = []
            for word in words:
                assert syndrome(record["columns"], word) == 0
                count = 5 if kind == "Z" else 3
                for block in range(count):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j + block)*q) for j in range(3))
                    else:
                        actual = word << (block*5*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        retained.append(actual)
            diagnostic = random.Random(
                seed + (4100009 if kind == "Z" else 5100011))
            for actual in sorted(set(retained), key=int.bit_count)[:4]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                neighborhood = sorted(
                    stabilizers, key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:24]
                for _ in range(64):
                    first, second = diagnostic.sample(neighborhood, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
        return reading

    for restart in range(3):
        aa = component_pool("A", restart)
        bb = component_pool("B", restart)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            ab[0]["free4"] and ab[1]["free4"],
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"] + ab[1]["d"],
            -ab[0]["cost"] - ab[1]["cost"]), reverse=True)
        best = None
        for a, b in proposals:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1428 and k >= 504
                    and 325*n <= 1000*k < 375*n):
                continue
            actual_a = [
                sum(((hx[r] >> (j*5*q)) & mask) << (j*q)
                    for j in range(3)) for r in range(q)]
            actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logical = localized_reading(a, b, hx, hz, n)
            detected = min(14, a["d"], b["d"], logical)
            key = (a["free4"] and b["free4"],
                   k*detected*detected, detected,
                   a["d"] + b["d"], -a["cost"] - b["cost"])
            if best is None or key > best[0]:
                best = key, a, b, hx, hz, k, logical
        if best is None:
            continue

        key, a, b, hx, hz, k, logical = best
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3, 3, 3]]
        assert [sum(row) for row in sb] == [7, 7]
        assert [sum(row[j] for row in sb) for j in range(5)] == [2, 3, 3, 3, 3]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx) <= 12
        assert max(row.bit_count() for row in hz) <= 10
        assert css_ok(hx, hz)
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (zero_budget, policy, cohort),
            "component_readings": (a["d"], b["d"]),
            "component_kernel4_free": (a["free4"], b["free4"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "localized_logical_reading": logical,
            "pairing_merit": key,
        }
    raise ValueError("new action-six cubic pairings missed the verified rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored new-group cell without additional draws."""
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
# REFUTED: Bounded regular-annihilator intersection searches yield accepted k=720/750 improvements on the affine bed; all 51 distinct current codes instead had k=690, and all five measured readings remained six.
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
