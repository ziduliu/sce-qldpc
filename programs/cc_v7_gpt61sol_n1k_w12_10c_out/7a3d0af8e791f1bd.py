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
CONCEPT = ("Hypothesis: on split-metacyclic C11:C5(s=3), order 55, A=(1,3), B=(2,8), shared-annihilator A cubics and kernel-five-screened six-matching B backbones produce five new cell-0.500 codes with fitness above 17.654. A verified k=720 and distance reading six scores 18.126; measurement can refute the improvement.")
GROUP = ("Keep the parent's split-metacyclic construction rule, resizing m,r,s from 11,10,2 to 11,5,3 because the new factor-26 shape cannot use order 110. Since 3^5=1 modulo 11, the order-55 group is valid. The parent repeatedly measured k=690 and distance six, motivating a structural rather than another rank-bank mutation. F1024 with action w->w^(2^8) proposes annihilators; authoritative lifts determine accepted ranks and kernels. No distance transfer is assumed.")
SHAPES = ("Change B from (1,4) to (2,8), retaining A=(1,3). Order 55 gives n=1430, k>=660 and guaranteed rate 6/13; q_max=57. Components have 165 variables/55 checks and 440 variables/110 checks. Parent minima occupied only sector-one Acol/Bcol blocks, especially Acol1/2 across Bcol0/1/2. A rank-45 component and full-rank B are proposed to yield k=720, but only final full ranks decide acceptance in [0.475,0.525).")
FILLING = ("A has three distinct cubic operator classes. B has six degree-two cross-matchings and two degree-three tails, with both B row totals nine: HX maxima 9+3=12 and HZ maxima 9+3=12. Twenty cards cross two tail concentration profiles, five A quotient-spread preferences and two geometry priorities. Native Sidon matching voltages nominate backbones; scaffold-derived components undergo exact kernel-four tests. Odd A degree excludes five. Bipartite B binomials plus anchored two-cubic/one-binomial collision tests exactly challenge B weight five. Hash a recoverable B Tanner invariant into disjoint slot buckets to prevent intra-program orbit copies without claiming complete archive recognition. Independently deepen component readings, cap selection at six, and bind final ranks, native-component agreement, full kernel-four exclusion, weights and CSS. No component screen certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r050_m11r5s3_13x28_sixmatching_annihilator_graphbucket_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C11:C5(s=3) requires order 55")
    # Same split-metacyclic construction rule as the parent.
    # (a,b)=x^a*y^b; y*x*y^-1=x^3, and 3 has order five mod 11.
    m, r, s = 11, 5, 3
    powers = [pow(s, b, m) for b in range(r)]
    assert pow(s, r, m) == 1 and len(set(powers)) == r
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
B_SHAPE = (2, 8)               # (rb, cb)


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


def _retired_order110_homology_ctx(rng, G, idx):
    """Inactive predecessor; never called on the resized shape or group."""
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

    # Both action conventions are obtained from the authoritative helper.
    # Subsequent component assembly is only GF2 linear superposition.
    mask = (1 << q) - 1
    mono = [[[g] for g in range(q)]]
    raw_a, _, _ = lifted_matrices(probe, mono, [[[]]])
    _, raw_b, _ = lifted_matrices(probe, [[[]]], mono)
    blocks = {
        "A": [[(row >> (g * q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def component(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            for element in cell:
                action = blocks[side][G.idx(element)]
                for r, word in enumerate(action):
                    rows[r] ^= word << (j * q)
        assert all(row >> (len(cells) * q) == 0 for row in rows)
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

    # Twenty structural cards: four backbone prescriptions crossed with
    # five preferences for the ACTUAL regular rank of a trinomial block.
    # Rank preferences are objectives, not infeasible hard labeling gates.
    bin_profiles = ((1, 1, 1), (1, 1, 3), (1, 3, 7), (3, 3, 3))
    preferred_tri_rank = (110, 100, 90, 80, 70)[card % 5]
    tri_bank, tri_seen = [], set()
    eligible = [g for g in range(q) if g != ident]

    for trial in range(1024):
        a, b = rng.sample(eligible, 2)
        support = tuple(sorted((ident, a, b)))
        canonical = min(
            tuple(sorted(table[g][inverses[root]] for g in support))
            for root in support)
        if canonical in tri_seen:
            continue
        tri_seen.add(canonical)
        cell = [G.elements[g] for g in canonical]
        rows = component([cell], "B")
        tri_bank.append({
            "cell": cell,
            "rank": gf2_rank(rows),
            "cycles": _four_cycles(rows),
            "signature": canonical,
        })
    if not tri_bank:
        raise ValueError("empty independently generated trinomial bank")

    # A common left cell gauge on B changes only component column labels.
    # Bank diversity therefore comes from canonical relative supports,
    # not from translated copies of one operator.
    tri_bank.sort(key=lambda rec: (
        abs(rec["rank"] - preferred_tri_rank),
        rec["cycles"], rec["signature"]))

    def fresh_tri(attempt):
        # Initially favor the requested rank; progressively broaden the
        # bank so a difficult objective cannot destroy an otherwise legal
        # slot. Every chosen operator still has three distinct terms.
        width = min(len(tri_bank), 24 + 2 * attempt)
        rec = rng.choice(tri_bank[:width])
        return list(rec["cell"])

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
        rng.shuffle(ordered)
        plans = {}
        join_cap = 4096 if kind == "plane" else 3072

        for functional, supports in ordered:
            # Shuffle actual operator classes before applying a bounded
            # budget. This is not a relabeling of a submitted filling.
            supports = sorted(set(supports))
            rng.shuffle(supports)
            if kind == "plane":
                supports.sort(key=lambda support:
                              -annihilators[support].bit_count())
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
        proposals = list(plans.values())
        rng.shuffle(proposals)
        if kind == "plane":
            proposals.sort(key=lambda item: -item[0])
        # The line bank must also visit ordinary rank-100 alternatives,
        # rather than spending its whole budget on failed gain proposals.
        records = []
        check_cap = 768

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
            # A short classical word is only a proposal diagnostic.
            # The complete CSS homology gate below decides whether it
            # survives as a nontrivial quantum logical.
            short_free = _pairs_no_kernel4(columns) is not None
            records.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": _four_cycles(rows), "short_free": short_free,
                "functional": functional,
                "dimension": dimension,
                "signature": signature,
            })

        # Challenge several genuinely different triples, not gauge
        # images. A longer independent search can only lower a reading.
        records.sort(key=lambda record: (
            record["rank"], not record["short_free"],
            record["cycles"], record["signature"]))
        finalists = []
        for target_rank in (80, 90, 100):
            near = [record for record in records
                    if record["rank"] == target_rank]
            if not near:
                continue
            # Retain several independent joins at EVERY rank. Failed
            # gain proposals cannot suppress all feasible alternatives.
            near = near[:48]
            near = rng.sample(near, min(10, len(near)))
            for record in near:
                record["reading"] = _component_reading(
                    record["columns"], rng, 8)
            near.sort(key=lambda record: (
                record["short_free"], record["reading"],
                -record["cycles"]), reverse=True)
            near = near[:4]
            for record in near:
                record["reading"] = min(
                    record["reading"],
                    _component_reading(record["columns"], rng, 24))
            finalists.extend(near)
        records = finalists

        a_banks[kind] = records
        return records

    def stabilizer_pivots(rows):
        pivots = {}
        for source in rows:
            word = source
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def nontrivial(word, pivots):
        """Exact failure of membership in the stabilizer row space."""
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def logical_four_free(columns, pivots):
        """Exhaustively classify complete-code kernels through four.

        Equal-syndrome subsets differ by a kernel word. Keeping one
        representative is sufficient: if every difference from it is a
        stabilizer, every difference between representatives is too.
        No full-code group symmetry is assumed.
        """
        singles = {}
        for v, value in enumerate(columns):
            word = 1 << v
            if value == 0 and nontrivial(word, pivots):
                return False
            old = singles.get(value)
            if old is not None and nontrivial(word ^ old, pivots):
                return False
            singles.setdefault(value, word)

        pairs = {}
        for v, value in enumerate(columns):
            for u in range(v):
                syndrome = value ^ columns[u]
                word = (1 << v) | (1 << u)
                if syndrome == 0 and nontrivial(word, pivots):
                    return False
                old = singles.get(syndrome)
                if old is not None and nontrivial(word ^ old, pivots):
                    return False
                old = pairs.get(syndrome)
                if old is not None and nontrivial(word ^ old, pivots):
                    return False
                pairs.setdefault(syndrome, word)
        return True

    def logical_reading(columns, pivots, trials, diagnostic):
        """Budgeted genuine logical witnesses, classified modulo stabilizers.

        A large result means this CPU search missed lighter logicals;
        unlike logical_four_free, this is not an exhaustive exclusion.
        """
        order = list(range(len(columns)))
        best = len(columns) + 1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best and nontrivial(word, pivots):
                best = word.bit_count()

        for _ in range(trials):
            diagnostic.shuffle(order)
            elimination, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
                while value:
                    p = value.bit_length() - 1
                    old = elimination.get(p)
                    if old is None:
                        elimination[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            light = sorted(basis, key=int.bit_count)[:20]
            for word in light:
                for other in basis:
                    observe(word ^ other)
            diagnostic.shuffle(basis)
            for initial in light[:4]:
                word = initial
                for _ in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
            if best <= 5:
                break
        return best

    diagnostic_seed = rng.randrange(1 << 62)

    def homology_screen(hx, hz, n):
        xp = stabilizer_pivots(hx)
        zp = stabilizer_pivots(hz)
        xc = _syndrome_columns(hz, n)
        zc = _syndrome_columns(hx, n)
        if not logical_four_free(xc, xp):
            return None
        if not logical_four_free(zc, zp):
            return None
        readings = []
        for side_id, columns, pivots in ((0, xc, xp), (1, zc, zp)):
            diagnostic = random.Random(diagnostic_seed + 100003 * side_id)
            reading = logical_reading(columns, pivots, 8, diagnostic)
            if reading <= 5:
                return None
            diagnostic = random.Random(
                diagnostic_seed + 700001 + 100003 * side_id)
            reading = min(reading, logical_reading(
                columns, pivots, 24, diagnostic))
            if reading <= 5:
                return None
            readings.append(reading)
        return tuple(readings)

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
            assert 19 * n <= 40 * k < 21 * n
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert css_ok(hx, hz)

            # Independently validate the cached component superpositions.
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert actual_a == component(A[0], "A")
            assert actual_b == component(B[0], "B")

            logical_screen = homology_screen(hx, hz, n)
            if logical_screen is None:
                continue
            tri_rows = component([B[0][-1]], "B")
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": (kind, card // 5, card % 5),
                "functional": a["functional"],
                "annihilator_dimension": a["dimension"],
                "component_ranks": (a["rank"], b["rank"]),
                "a_classical_kernel4_free": a["short_free"],
                "b_trinomial_rank": gf2_rank(tri_rows),
                "preferred_b_trinomial_rank": preferred_tri_rank,
                "logical_screen": logical_screen,
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
    """Two-row B backbones paired with independently screened cubic joins."""
    from hashlib import sha256
    from math import exp

    q = G.q
    assert q == 55 and A_SHAPE == (1, 3) and B_SHAPE == (2, 8)
    card, cohort = idx % 20, idx // 20
    concentration = card % 2
    model = (card // 2) % 5
    geometry = card // 10
    seed = rng.randrange(1 << 62) ^ (0x551328 + 1000003 * idx)
    local = random.Random(seed)
    ident = G.idx(G.e)
    mask = (1 << q) - 1

    # Cache native operations only. Both binary action conventions below
    # are obtained from the exact scaffold constructor.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]

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
    gallery = [[[g] for g in range(q)]]
    raw_a, _, _ = lifted_matrices(P, gallery, [[[]]])
    _, raw_b, _ = lifted_matrices(P, [[[]]], gallery)
    blocks = {
        "A": [[(row >> (g * q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in raw_b]
              for g in range(q)],
    }
    del raw_a, raw_b

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length() - 1
            word ^= low

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i * q + t] ^= value << (j * q)
        return rows

    def native(grid):
        return [[[G.elements[g] for g in cell] for cell in row]
                for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    # The same Phi_11 field as the parent, with the RESIZED action:
    # 3 = 2^8 modulo 11. Representation constraints propose labels only.
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

    def frob(a, steps):
        for _ in range(steps % 10):
            a = fmul(a, a)
        return a

    roots = [1]
    for _ in range(10):
        roots.append(fmul(roots[-1], 2))
    assert fmul(roots[-1], 2) == 1
    assert pow(2, 8, 11) == 3
    adjoints = [
        [frob(fmul(1 << t, roots[a]), -8 * b) for t in range(10)]
        for a, b in G.elements
    ]

    def proposed_kernel(support):
        pivots, basis = {}, []
        for t in range(10):
            value = 0
            for g in support:
                value ^= adjoints[g][t]
            word = 1 << t
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
        values = [0]
        for word in basis:
            values += [v ^ word for v in values]
        return values[1:]

    line_bank, operators = {}, {}
    eligible = [g for g in range(q) if g != ident]
    for at, first in enumerate(eligible):
        for second in eligible[:at]:
            support = tuple(sorted((ident, first, second)))
            canonical = min(
                tuple(sorted(table[g][inverse[root]] for g in support))
                for root in support)
            if support != canonical:
                continue
            values = proposed_kernel(support)
            if not values:
                continue
            rows = component([[list(support)]], "A")
            columns = _syndrome_columns(rows, q)
            if _pairs_no_kernel4(columns) is None:
                continue
            operators[support] = (rows, columns)
            for value in values:
                line_bank.setdefault(value, []).append(support)

    # Collapse identical proposal catalogs, not submitted code designs.
    catalogs = sorted({
        tuple(sorted(set(options)))
        for options in line_bank.values()
        if len(set(options)) >= 3
    })
    if not catalogs:
        raise ValueError("order-55 cubic annihilator catalog is empty")

    spread_target = (6, 7, 8, 9, 5)[model]
    aa, seen_a = [], set()
    for trial in range(4096):
        options = catalogs[
            (13 * card + 17 * cohort + trial // 16) % len(catalogs)]
        selected = tuple(sorted(local.sample(options, 3)))
        if selected in seen_a:
            continue
        seen_a.add(selected)
        rows = [
            sum(operators[s][0][t] << (j * q)
                for j, s in enumerate(selected))
            for t in range(q)
        ]
        # This is an ACTUAL component rank, not a representation inference.
        if gf2_rank(rows) != 45:
            continue
        columns = [c for s in selected for c in operators[s][1]]
        assert all(c.bit_count() == 3 for c in columns)
        if _pairs_no_kernel4(columns) is None:
            continue
        # Odd column weight excludes every odd-weight kernel word.
        spread = sum(len({G.elements[g][1] for g in s})
                     for s in selected)
        cycles = _four_cycles(rows)
        aa.append(dict(
            grid=[[list(s) for s in selected]],
            rows=rows, columns=columns, rank=45,
            cycles=cycles, spread=spread, signature=selected))
        if len(aa) >= 12:
            break
    if not aa:
        raise ValueError("no rank-45 cubic join passed kernel-four screening")

    aa.sort(key=lambda rec: (
        abs(rec["spread"] - spread_target),
        rec["cycles"], rec["signature"]))
    aa = aa[:6]
    for at, rec in enumerate(aa):
        rec["reading"] = _component_reading(
            rec["columns"], random.Random(seed + 1009 * at), 8)
    aa.sort(key=lambda rec: (
        min(6, rec["reading"]), rec["reading"],
        -abs(rec["spread"] - spread_target), -rec["cycles"]),
        reverse=True)
    aa = aa[:3]
    for at, rec in enumerate(aa):
        rec["reading"] = min(
            rec["reading"], _component_reading(
                rec["columns"],
                random.Random(seed + 700001 + 1013 * at), 32))

    # Six cross-matchings define a bipartite degree-six check graph.
    # Unique native differences nominate C4-free matching sets. No
    # handwritten permutation matrix or approximate lift is used.
    def matching_cost(gs):
        differences = [
            table[inverse[g]][h]
            for g in gs for h in gs if g != h
        ]
        return len(differences) - len(set(differences))

    backbones, seen_backbones = [], set()
    for restart in range(48):
        gs = [ident] + local.sample(eligible, 5)
        cost = matching_cost(gs)
        for step in range(384):
            if cost == 0:
                sig = tuple(sorted(gs))
                if sig not in seen_backbones:
                    seen_backbones.add(sig)
                    backbones.append(list(gs))
                break
            position = local.randrange(1, 6)
            old = gs[position]
            available = [g for g in range(q) if g not in gs]
            gs[position] = local.choice(available)
            proposed = matching_cost(gs)
            temperature = 1.4 * (1 - step / 384) + 0.10
            if (proposed <= cost or
                    local.random() < exp((cost - proposed) / temperature)):
                cost = proposed
            else:
                gs[position] = old
        if len(backbones) >= 12:
            break
    if not backbones:
        raise ValueError("no native Sidon six-matching backbone found")

    tri_profiles = (
        (0, 1, 2), (0, 1, 3), (0, 2, 4),
        (0, 0, 1), (0, 1, 1),
    )

    def fresh_cell(count, trial):
        if count == 0:
            return []
        if count == 1:
            return [local.randrange(q)]
        if count == 2:
            return local.sample(range(q), 2)
        for _ in range(48):
            if trial < 384:
                coordinates = tri_profiles[model]
                cell = [G.idx((local.randrange(11), b))
                        for b in coordinates]
            else:
                cell = local.sample(range(q), 3)
            if len(set(cell)) != 3:
                continue
            root = inverse[cell[0]]
            u, v = table[root][cell[1]], table[root][cell[2]]
            if table[u][v] != table[v][u]:
                return cell
        return local.sample(range(q), 3)

    def fresh_b(trial):
        gs = local.choice(backbones)
        grid = [
            [[ident] for _ in range(6)],
            [[g] for g in gs],
        ]
        tails = ((3, 0), (0, 3)) if concentration == 0 else (
            (2, 1), (1, 2))
        for counts in tails:
            for i in range(2):
                grid[i].append(fresh_cell(counts[i], trial))
        return grid

    def b_kernel5_free(columns, pairs):
        """Exact B weight-five test after complete kernel-four exclusion.

        Binomial columns form a bipartite graph, so five binomials cannot
        be a kernel. Ordinary syndrome parity requires an even number of
        degree-three variables. Every remaining five-word therefore has
        a partition containing two cubics and one binomial on its triple
        side. Regular COMPONENT symmetry allows one cubic anchor/block.
        No symmetry of the full nonabelian CSS code is assumed.
        """
        bins = [v for v, c in enumerate(columns) if c.bit_count() == 2]
        tris = [v for v, c in enumerate(columns) if c.bit_count() == 3]
        assert len(bins) == 6 * q and len(tris) == 2 * q
        for root in (6 * q, 7 * q):
            for other in tris:
                if other == root:
                    continue
                value = columns[root] ^ columns[other]
                for third in bins:
                    if value ^ columns[third] in pairs:
                        return False
        return True

    def graph_fingerprint(rows, columns):
        """Invariant of the B graph recoverable from the full Tanner graph.

        Each component check-row and variable-column block is a regular
        orbit. Sorted rooted profiles consequently give invariant orbit
        multisets, including multiplicities. This is not a complete
        isomorphism algorithm; different fingerprints imply inequivalence.
        """
        width = len(columns)
        adjacency = [
            [width + r for r in bits(c)] for c in columns
        ] + [list(bits(row)) for row in rows]
        colors = [
            0 if c.bit_count() == 2 else 1 for c in columns
        ] + [2] * len(rows)

        def rooted(root):
            distances = [-1] * len(adjacency)
            distances[root] = 0
            queue, histogram = [root], []
            for vertex in queue:
                depth = distances[vertex]
                while len(histogram) <= depth:
                    histogram.append([0, 0, 0])
                histogram[depth][colors[vertex]] += 1
                for neighbor in adjacency[vertex]:
                    if distances[neighbor] < 0:
                        distances[neighbor] = depth + 1
                        queue.append(neighbor)
            # Rooted closed-walk counts are invariant diagnostics, not
            # counts of simple cycles or distance certificates.
            current, returns = {root: 1}, []
            for _ in range(10):
                following = {}
                for vertex, count in current.items():
                    for neighbor in adjacency[vertex]:
                        following[neighbor] = (
                            following.get(neighbor, 0) + count)
                current = following
                returns.append(current.get(root, 0))
            return (tuple(tuple(h) for h in histogram), tuple(returns))

        return (
            tuple(sorted(rooted(width + i * q) for i in range(2))),
            tuple(sorted(rooted(j * q) for j in range(8))),
        )

    bb, seen_b, fingerprints = [], set(), set()
    for trial in range(1536):
        grid = fresh_b(trial)
        sig = signature(grid)
        if sig in seen_b:
            continue
        seen_b.add(sig)
        rows = component(grid, "B")
        columns = _syndrome_columns(rows, 8 * q)
        pairs = _pairs_no_kernel4(columns)
        if pairs is None:
            continue
        if gf2_rank(rows) != 2 * q:
            continue
        if not b_kernel5_free(columns, pairs):
            continue
        fingerprint = graph_fingerprint(rows, columns)
        bucket = int.from_bytes(
            sha256(repr(fingerprint).encode("ascii")).digest(), "big")
        if bucket % N_SLOTS != idx or fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        cycles = _four_cycles(rows)
        bb.append(dict(
            grid=grid, rows=rows, columns=columns, rank=2 * q,
            cycles=cycles, fingerprint=fingerprint,
            bucket=bucket % N_SLOTS, trial=trial))
        if len(bb) >= 2:
            break
    if not bb:
        raise ValueError("no kernel-five-free B graph reached this slot bucket")

    for at, rec in enumerate(bb):
        rec["reading"] = _component_reading(
            rec["columns"], random.Random(seed + 100003 + 1009 * at), 8)
        rec["reading"] = min(
            rec["reading"], _component_reading(
                rec["columns"],
                random.Random(seed + 800009 + 1013 * at), 32))

    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        min(6, ab[0]["reading"], ab[1]["reading"]),
        min(ab[0]["reading"], ab[1]["reading"]),
        -abs(ab[0]["spread"] - spread_target),
        -(ab[0]["cycles"] + (1 + geometry) * ab[1]["cycles"])),
        reverse=True)

    for a, b in proposals:
        A, B = native(a["grid"]), native(b["grid"])
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3, 3, 3]]
        assert [sum(row) for row in sb] == [9, 9]
        assert [sum(row[j] for row in sb) for j in range(8)] \
            == [2, 2, 2, 2, 2, 2, 3, 3]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1430 and 19 * n <= 40 * k < 21 * n):
            continue
        actual_a = [
            sum(((hx[t] >> (j * 8 * q)) & mask) << (j * q)
                for j in range(3))
            for t in range(q)
        ]
        actual_b = [
            hz[alpha * q + t] & ((1 << (8 * q)) - 1)
            for alpha in range(2) for t in range(q)
        ]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert max(row.bit_count() for row in hx + hz) == 12

        # Recoverability is checked on the actual FULL graph. Degree-18
        # variables are sector two. HZ checks uniquely see six degree-5,
        # three degree-6 and three degree-18 neighbors. Deleting degree-18
        # variables leaves three copies of the fingerprinted B graph.
        full_columns = _syndrome_columns(hx + hz, n)
        degrees = [c.bit_count() for c in full_columns]
        assert set(degrees) == {5, 6, 18}

        def profile(row):
            values = [degrees[c] for c in bits(row)]
            return tuple(values.count(d) for d in (5, 6, 18))

        assert all(profile(row) == (6, 3, 3) for row in hz)
        assert all(profile(row) != (6, 3, 3) for row in hx)
        assert b["bucket"] == idx
        if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
            continue
        if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
            continue
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, idx=idx, k=k,
            card=(concentration, model, geometry, cohort),
            component_ranks=(a["rank"], b["rank"]),
            actual_A_defect=q - a["rank"],
            component_readings=(a["reading"], b["reading"]),
            component_cycles=(a["cycles"], b["cycles"]),
            A_quotient_spread=a["spread"],
            recoverable_B_graph_bucket=b["bucket"],
            graph_bucket_modulus=N_SLOTS,
            selection_reading_cap=6,
            target_cell=0.500)
    raise ValueError("new two-row pairings missed full rank/kernel gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell at every new-shape position."""
    cell = list(ctx[side][i][j])
    assert len(cell) <= 5
    assert len({G.idx(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Random plane-bank preference with rank-100 fallbacks reliably finds additional rank gain on this bed; every distinct current slot reported k=690 and k-kb=30, and all five measured distances remained six.
# REFUTED: Bounded systematic joins prioritized by representation annihilator dimension yield accepted k>690 under classical-kernel rejection; all 22 distinct measured-program candidates instead had k=690 and all five promoted distances remained six.
# REFUTED: Canonical-join selection with fixed quotient B cards reliably supplies inequivalent valid slots; the current table contains many isomorphic duplicates and four labeling failures.
# REFUTED: Stabilizer-aware rank-stratified affine joins improve the order-110 (1,3)x(1,4) plateau; all 52 distinct current codes had k=690 and all five measured readings remained six, with abundant sector-one minima.
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
