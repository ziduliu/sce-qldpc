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
CONCEPT = ("Split-metacyclic C16:C4(s=9), order 64, A=(1,3), B=(2,7): noncommuting order-sixteen ring backbones and witness-directed cubic repair should produce five new measured readings >=9 and improve cell 0.450 beyond 28.268. This is a falsifiable measurement target, not a distance certificate.")
GROUP = ("Keep the parent's split-metacyclic construction rule; change only its parameters for the new size factor. Action 9 modulo 16 is nontrivial and satisfies 9^2=1. Order 64 gives n=1472. In this 2-group, odd-augmentation group-ring elements are units, motivating cubic A cells without imposed annihilators. The parent repeatedly measured distance six despite rank-gain searches; authoritative component and full binary ranks still decide acceptance.")
SHAPES = ("Change to A=(1,3), B=(2,7): factor 23 permits q<=65, and q=64 gives n=1472, k>=640 and guaranteed rate 10/23=0.434783 in cell 0.450. Components have 192 variables/64 checks and 448 variables/128 checks. Parent weight-six minima were entirely sector 1, especially Acol1/2 across Bcol0/1/2. The second B row permits two noncommuting rings and a cross matching instead of three single-row binomial cells. Exact full ranks gate [0.425,0.475).")
FILLING = ("A has three cubic cells; B has three degree-two and four cubic columns, with both B row totals nine. HX<=9+3=12 and HZ<=9+3=12. Twenty cards cross four B support profiles with five A relative-order compositions. Select ring backbones by actual shortest cycle words, assemble components only from scaffold monomial blocks, optimize cycle geometry, and repair columns supporting genuine kernel witnesses. Independently deepen elimination and budgeted parity-completion searches. Test localized logical nontriviality and reductions using original stabilizers. Final original-group ranks, cached-lift agreement, CSS, distinctness and weights gate every submission; no component reading or exhausted budget certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r045_m16r4s9_13x27_noncommuting_rings_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 64

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("split-metacyclic C16:C4 requires order 64")
    # Same split-metacyclic law as the parent, with new parameters.
    # (a,b)=x^a y^b; x^16=y^4=1 and y*x*y^-1=x^9.
    m, r, s = 16, 4, 9
    assert pow(s, r, m) == 1
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

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _retired_annihilator_ctx(rng, G, idx):
    """Unused order-110 implementation; the new-shape NEW_CTX is below."""
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

    # Cross four binomial-coordinate multiplicities with five trinomial
    # prescriptions. Coordinates five are allowed in TRINOMIALS, not
    # binomials: an involutory binomial would duplicate lifted columns.
    spectra = ((1, 5), (2, 5), (3, 5), (4, 5), (4, 8))
    bin_profiles = ((1, 1, 1), (1, 1, 3), (1, 3, 3), (3, 3, 3))

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

    a_banks = {}

    def systematic_A(kind):
        """Join canonical operators, then screen actual binary matrices.

        Enumeration limits bound construction work, not distance claims.
        A's weight-three columns make every odd-weight kernel impossible,
        so the exhaustive collision screen through four also excludes five.
        """
        from itertools import combinations

        if kind in a_banks:
            return a_banks[kind]
        catalog = catalogs[kind]
        if not catalog:
            a_banks[kind] = []
            return []

        # Visit the whole catalog, rather than restricting every card to
        # its sixteen largest classes. Rotation changes which proposals
        # receive a finite budget, not the labels of an existing code.
        offset = (17 * card + 29 * (idx // 20)) % len(catalog)
        ordered = catalog[offset:] + catalog[:offset]
        plans = {}
        join_cap = 4096 if kind == "plane" else 1024

        for functional, supports in ordered:
            # Higher individual nullities receive priority within a
            # class. All supports remain distinct right-gauge classes.
            supports = sorted(set(supports), key=lambda support: (
                -annihilators[support].bit_count(), support))
            for triple in combinations(supports, 3):
                signature = tuple(sorted(triple))
                if signature in plans:
                    continue
                common = (annihilators[triple[0]] &
                          annihilators[triple[1]] &
                          annihilators[triple[2]])
                count = common.bit_count()
                if count < (3 if kind == "plane" else 1):
                    continue
                # A d-dimensional space has 2^d-1 nonzero elements.
                dimension = (count + 1).bit_length() - 1
                assert count == (1 << dimension) - 1
                plans[signature] = (dimension, functional, signature)
                if len(plans) >= join_cap:
                    break
            if len(plans) >= join_cap:
                break

        # Representation dimensions prioritize proposals only. Actual
        # ranks below are never inferred from this representation.
        proposals = sorted(plans.values(), key=lambda item: (
            -item[0],
            sum((support[1] // 11 - support[2] // 11) ** 2
                for support in item[2]) if card & 1 else 0,
            item[2]))
        records = []
        check_cap = 512 if kind == "plane" else 192

        for dimension, functional, signature in proposals[:check_cap]:
            cells = [[G.elements[t] for t in support]
                     for support in signature]
            rows = component(cells, "A")
            rank = gf2_rank(rows)
            if rank not in ({80, 90} if kind == "plane"
                            else {80, 90, 100}):
                continue
            columns = _syndrome_columns(rows, 3 * q)
            assert all(column.bit_count() == 3 for column in columns)
            if _pairs_no_kernel4(columns) is None:
                continue
            records.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": _four_cycles(rows),
                "functional": functional,
                "dimension": dimension,
                "signature": signature,
            })

        # Challenge several genuinely different triples, not gauge
        # images. A longer independent search can only lower a reading.
        records.sort(key=lambda record: (
            record["rank"], record["cycles"], record["signature"]))
        if records:
            best_rank = records[0]["rank"]
            near = [record for record in records
                    if record["rank"] == best_rank]
            near = near[:8]
            for record in near:
                record["reading"] = _component_reading(
                    record["columns"], rng, 12)
            near.sort(key=lambda record: (
                record["reading"], -record["cycles"]), reverse=True)
            near = near[:3]
            for record in near:
                record["reading"] = min(
                    record["reading"],
                    _component_reading(record["columns"], rng, 48))
            near.sort(key=lambda record: (
                record["reading"], -record["cycles"]), reverse=True)
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
                "annihilator_dimension": a["dimension"],
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
    """Twenty new-shape cards with exact ring words and witness repair."""
    from itertools import combinations
    from math import exp

    q = G.q
    assert q == 64 and A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    card, cohort = idx % 20, idx // 20
    topology, composition = card // 5, card % 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    identity = G.idx(G.e)
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
    monomials = [[[g] for g in range(q)]]
    raw_a, _, _ = lifted_matrices(P, monomials, [[[]]])
    _, raw_b, _ = lifted_matrices(P, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for v in bits(word):
            value ^= columns[v]
        return value

    orders = []
    for g in range(q):
        value = identity
        for order in range(1, q + 1):
            value = table[value][g]
            if value == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid metacyclic element order")

    def generates(a, b):
        reached, todo = {identity}, [identity]
        for x in todo:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    todo.append(z)
        return len(reached) == q

    # Different relative-order spectra, not group-automorphism copies.
    libraries = {4: [], 8: [], "broad": []}
    for a in range(q):
        if a == identity:
            continue
        for b in range(a):
            if b == identity or table[a][b] == table[b][a]:
                continue
            triple = (identity, a, b)
            libraries["broad"].append(triple)
            if not generates(a, b):
                continue
            spectrum = sorted((
                orders[a], orders[b], orders[table[inverse[a]][b]]))
            if spectrum == [4, 16, 16]:
                libraries[4].append(triple)
            elif spectrum == [8, 16, 16]:
                libraries[8].append(triple)
    assert libraries[4] and libraries[8] and libraries["broad"]

    profiles = (
        ((2, 1), (2, 1), (1, 2), (1, 2)),
        ((3, 0), (0, 3), (2, 1), (1, 2)),
        ((3, 0), (3, 0), (0, 3), (0, 3)),
        ((3, 0), (1, 2), (1, 2), (1, 2)),
    )
    sa = [[3, 3, 3]]
    sb = [[2, 0, 1] + [p[0] for p in profiles[topology]],
          [0, 2, 1] + [p[1] for p in profiles[topology]]]
    assert all(sum(row) == 9 for row in sa + sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    def inspect(grid, side):
        """GF2 superposition of authoritative scaffold monomial blocks."""
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nr*q)
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i*q+t] ^= value << (j*q)
        columns = _syndrome_columns(rows, nc*q)
        adjacency = [0] * len(rows)
        uses, four, forced = {}, 0, 0
        for column in columns:
            checks = list(bits(column))
            degree = len(checks)
            forced += degree*(degree-1)*(degree-2)//6
            for r in checks:
                adjacency[r] |= column ^ (1 << r)
            for a, b in combinations(checks, 2):
                key = (a, b)
                old = uses.get(key, 0)
                four += old
                uses[key] = old + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        excess = max(0, triangles-forced)
        bad = sum(c == 0 for c in columns)
        bad += len(columns)-len(set(columns))
        cost = 1000000*bad + (600+150*composition)*four
        cost += (1+cohort % 3)*excess
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cost=cost, cycles=(four, excess), words=set(),
                    reading=len(columns)+1)

    def shortest_cycle(record):
        """Exact shortest graph cycle, returned in component coordinates."""
        adjacency = [[] for _ in record["rows"]]
        for v, column in enumerate(record["columns"]):
            checks = list(bits(column))
            assert len(checks) == 2
            a, b = checks
            adjacency[a].append((b, v))
            adjacency[b].append((a, v))
        best = None
        for root in range(len(adjacency)):
            distance = [-1] * len(adjacency)
            paths = [0] * len(adjacency)
            parents = [-1] * len(adjacency)
            distance[root] = 0
            todo = [root]
            for a in todo:
                for b, edge in adjacency[a]:
                    if distance[b] < 0:
                        distance[b] = distance[a]+1
                        paths[b] = paths[a] ^ (1 << edge)
                        parents[b] = edge
                        todo.append(b)
                    elif edge != parents[a] and edge != parents[b]:
                        word = paths[a] ^ paths[b] ^ (1 << edge)
                        if word and (best is None or
                                     word.bit_count() < best.bit_count()):
                            best = word
        assert best and syndrome(record["columns"], best) == 0
        return best

    ring_pairs = [
        (a, b) for a in range(q) if orders[a] == 16
        for b in range(a) if orders[b] == 16
        and table[a][b] != table[b][a] and generates(a, b)
    ]
    assert ring_pairs
    backbones = []
    for a, b in rng.sample(ring_pairs, min(96, len(ring_pairs))):
        grid = [[[identity, a], [], [identity]],
                [[], [identity, b], [identity]]]
        record = inspect(grid, "B")
        word = shortest_cycle(record)
        # Relative quotient coordinates supply additional non-gauge choices.
        geometry = (
            (G.elements[a][1]-G.elements[b][1]) % 4,
            (G.elements[a][0]-G.elements[b][0]) % 16)
        backbones.append((word.bit_count(), geometry, grid, word))
    best_cycle = max(item[0] for item in backbones)
    choices = [item for item in backbones if item[0] == best_cycle]
    choices.sort(key=lambda item: item[1])
    window = choices[(composition+7*cohort) % len(choices):]
    if not window:
        window = choices
    _, backbone_geometry, skeleton, backbone_word = rng.choice(
        window[:max(1, len(choices)//3)])

    aplans = (
        (4, 4, 4), (4, 4, 8), (4, 8, 8),
        (8, 8, 8), ("broad", 4, 8),
    )
    aplan = aplans[composition]
    long_ratios = [g for g in range(q) if orders[g] == 16]

    def draw(count, side, j):
        if count == 0:
            return []
        root = rng.randrange(q)
        if count == 1:
            return [root]
        if count == 2:
            ratio = rng.choice(long_ratios)
            return [root, table[root][ratio]]
        assert count == 3
        kind = aplan[j] if side == "A" else (
            4 if (j+topology+cohort) % 2 == 0 else 8)
        triple = rng.choice(libraries[kind])
        # An overall gauge is not used to create another slot. These
        # independently drawn ratios are subsequently optimized jointly.
        return [table[root][g] for g in triple]

    def fresh(side):
        if side == "A":
            return [[draw(3, side, j) for j in range(3)]]
        grid = copy_grid(skeleton)
        for j, counts in enumerate(profiles[topology], 3):
            for i in range(2):
                grid[i].append(draw(counts[i], side, j))
        return grid

    def note(record, word):
        if not word:
            return
        assert syndrome(record["columns"], word) == 0
        record["reading"] = min(record["reading"], word.bit_count())
        if word.bit_count() <= record["reading"]+4:
            record["words"].add(word)
        if len(record["words"]) > 32:
            record["words"] = set(sorted(
                record["words"], key=lambda w: (w.bit_count(), w))[:16])

    def prepare(record):
        """Exhaustive component collision test through weight four."""
        if "completion" in record:
            return
        columns = record["columns"]
        singles, pairs = {}, {}
        incidence = [0] * len(record["rows"])
        free4 = True
        for v, column in enumerate(columns):
            word = 1 << v
            if not column:
                note(record, word)
                free4 = False
            old = singles.get(column)
            if old is not None:
                note(record, old ^ word)
                free4 = False
            else:
                singles[column] = word
            for r in bits(column):
                incidence[r] |= word
        for a, column in enumerate(columns):
            for b in range(a):
                value = column ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(value)
                if old is not None:
                    note(record, old ^ word)
                    free4 = False
                old = pairs.get(value)
                if old is not None:
                    note(record, old ^ word)
                    free4 = False
                else:
                    pairs[value] = word
        record["free4"] = free4
        record["completion"] = singles, pairs, incidence

    def challenge(record, side, trials, budget, offset):
        """Actual null words; completion budget stops never certify absence."""
        local = random.Random(seed+offset+100003*int(side == "B"))
        prepare(record)
        if side == "B":
            note(record, backbone_word)
        columns = record["columns"]
        width = len(columns)
        if record["reading"] <= 4:
            return
        order = list(range(width))
        retained = sorted(record["words"], key=int.bit_count)[:16]
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(record, word)
            light = sorted(set(retained+basis), key=int.bit_count)[:16]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        note(record, candidate)
                        if candidate.bit_count() <= record["reading"]+2:
                            additions.append(candidate)
            moves = retained+basis
            local.shuffle(moves)
            for initial in light[:4]:
                word = initial
                for sweep in range(2):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    note(record, word)
                    if not changed:
                        break
                additions.append(word)
            retained = sorted(set(retained+light+additions),
                              key=int.bit_count)[:20]

        singles, pairs, incidence = record["completion"]
        priorities = [local.random() for _ in columns]
        roots = list(range(0, width, q))
        local.shuffle(roots)
        all_variables = (1 << width)-1
        for root in roots:
            remaining = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    note(record, word)
                    return
                left = min(10, record["reading"])-word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    note(record, word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(record, word | pair)
                left = min(10, record["reading"])-word.bit_count()
                if left <= 2:
                    return
                chosen, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                candidates = list(bits(chosen))
                candidates.sort(key=lambda v: (
                    -(columns[v] & value).bit_count(), priorities[v]))
                live = allowed
                for v in candidates:
                    bit = 1 << v
                    live &= ~bit
                    visit(value ^ columns[v], word | bit, live)
                    if remaining[0] <= 0:
                        break

            # Only classical component regular symmetry is used.
            allowed = all_variables & ~((1 << (root+1))-1)
            visit(columns[root], 1 << root, allowed)

    def signature(record):
        return tuple(tuple(sorted(cell))
                     for row in record["grid"] for cell in row)

    def merit(record):
        d = min(12, record["reading"])
        multiplicity = sum(w.bit_count() == record["reading"]
                           for w in record["words"])
        return (record["free4"], d, -multiplicity,
                -record["cost"], record["rank"])

    def component_pool(side):
        support = sa if side == "A" else sb
        mutable = list(range(3)) if side == "A" else list(range(3, 7))
        bank = {}

        def remember(record):
            sig = signature(record)
            bank.setdefault(sig, record)
            if len(bank) > 8:
                worst = max(bank, key=lambda s: bank[s]["cost"])
                del bank[worst]

        starts = [inspect(fresh(side), side) for _ in range(20)]
        starts.sort(key=lambda r: r["cost"])
        for current in starts[:3]:
            remember(current)
            for step in range(72):
                grid = copy_grid(current["grid"])
                j = rng.choice(mutable)
                occupied = [i for i in range(len(grid)) if support[i][j]]
                i = rng.choice(occupied)
                grid[i][j] = draw(support[i][j], side, j)
                proposed = inspect(grid, side)
                delta = proposed["cost"]-current["cost"]
                temperature = q*(8*(1-step/72.0)**2+0.10)
                if delta <= 0 or rng.random() < exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(current)

        selected = []
        for record in sorted(bank.values(), key=lambda r: r["cost"]):
            record["rank"] = gf2_rank(record["rows"])
            challenge(record, side, 8, 5000, 700001)
            selected.append(record)
        assert selected
        selected.sort(key=merit, reverse=True)
        current = selected[0]
        keepers = selected[:4]
        seen = {signature(r) for r in selected}
        historical = set()
        for record in selected:
            historical.update(record["words"])

        # Change the actual implicated column, not a global relabeling.
        for step in range(16):
            active = sorted(current["words"], key=int.bit_count)[:12]
            proposed_records = []
            for move in range(3):
                grid = copy_grid(current["grid"])
                eligible = mutable
                if active:
                    occupied = {v//q for v in bits(rng.choice(active))}
                    targeted = [j for j in mutable if j in occupied]
                    if targeted:
                        eligible = targeted
                j = rng.choice(eligible)
                for i in range(len(grid)):
                    grid[i][j] = draw(support[i][j], side, j)
                record = inspect(grid, side)
                sig = signature(record)
                if sig in seen:
                    continue
                seen.add(sig)
                survived = sum(syndrome(record["columns"], w) == 0
                               for w in historical)
                proposed_records.append((survived, record["cost"], record))
            proposed_records.sort(key=lambda item: item[:2])
            for _, _, record in proposed_records[:2]:
                record["rank"] = gf2_rank(record["rows"])
                for word in historical:
                    if syndrome(record["columns"], word) == 0:
                        note(record, word)
                challenge(record, side, 6, 4000,
                          1000003+7919*step)
                historical.update(record["words"])
                if len(historical) > 64:
                    historical = set(sorted(
                        historical, key=lambda w: (w.bit_count(), w))[:64])
                if merit(record) > merit(current):
                    current = record
                keepers.append(record)
                keepers.sort(key=merit, reverse=True)
                del keepers[6:]

        unique = {signature(r): r for r in keepers+[current]}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:3]
        for record in finalists:
            challenge(record, side, 48, 24000, 2700011)
            record["words"] = set(sorted(
                record["words"], key=lambda w: (w.bit_count(), w))[:12])
            record.pop("completion", None)
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa, bb = component_pool("A"), component_pool("B")

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
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
        """Exact logical membership; bounded stabilizer reduction only."""
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            local = random.Random(seed+(4100011 if kind == "Z"
                                        else 5100011))
            candidates = []
            for word in sorted(record["words"], key=int.bit_count)[:4]:
                for block in range(7 if kind == "Z" else 3):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((7*j+block)*q)
                            for j in range(3))
                    else:
                        actual = word << (block*7*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.append(actual)
            reading = n+1
            for actual in sorted(set(candidates), key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposal = actual ^ row
                        if proposal.bit_count() < actual.bit_count():
                            actual, changed = proposal, True
                    if not changed:
                        break
                neighbors = sorted(
                    stabilizers, key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:32]
                for _ in range(128):
                    first, second = local.sample(neighbors, 2)
                    proposal = actual ^ first ^ second
                    if proposal.bit_count() < actual.bit_count():
                        actual = proposal
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        min(ab[0]["reading"], ab[1]["reading"]),
        ab[0]["reading"]+ab[1]["reading"],
        -ab[0]["cost"]-ab[1]["cost"]), reverse=True)
    best = None
    for a, b in proposals:
        hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1472 and k >= 640
                and 425*n <= 1000*k < 475*n):
            continue
        authoritative_a = [
            sum(((hx[t] >> (j*7*q)) & mask) << (j*q)
                for j in range(3)) for t in range(q)]
        authoritative_b = [
            row & ((1 << (7*q))-1) for row in hz[:2*q]]
        assert authoritative_a == a["rows"]
        assert authoritative_b == b["rows"]
        readings = localized(a, b, hx, hz, n)
        conservative = min(12, a["reading"], b["reading"], *readings)
        key = (k*conservative*conservative, conservative,
               a["free4"] and b["free4"],
               -a["cost"]-b["cost"])
        if best is None or key > best[0]:
            best = (key, a, b, hx, hz, k, readings)
    if best is None:
        raise ValueError("ring-backed pairings missed exact cell 0.450 gate")

    key, a, b, cached_hx, cached_hz, k, readings = best
    A = [[[G.elements[g] for g in cell] for cell in row]
         for row in a["grid"]]
    B = [[[G.elements[g] for g in cell] for cell in row]
         for row in b["grid"]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1472
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert all(len({G.idx(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {
        "A": A, "B": B, "idx": idx, "k": k,
        "card": (topology, composition, cohort),
        "component_readings": (a["reading"], b["reading"]),
        "component_ranks": (a["rank"], b["rank"]),
        "component_kernel4_free": (a["free4"], b["free4"]),
        "backbone_cycle_weight": best_cycle,
        "backbone_geometry": backbone_geometry,
        "verified_localized_logical_readings": readings,
        "quality": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return each selected new-shape cell, including deliberate zero cells."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Random plane-bank preference with rank-100 fallbacks reliably finds additional rank gain on this bed; every distinct current slot reported k=690 and k-kb=30, and all five measured distances remained six.
# REFUTED: Bounded systematic common-annihilator joins raise the measured affine distance-six plateau; all 22 distinct current codes still had k=690, and all five measured codes returned six.
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
