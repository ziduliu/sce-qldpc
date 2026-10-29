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
CONCEPT = ("C29 semidirect_16 C7, order 203, A=(1,2), B=(1,3): quotient-composition cubic cards with retained kernel witnesses and independent logical challenges should produce five new measured readings >=18 and beat fitness 73.4 in cell 0.300. Failure of that five-code target refutes the experiment; readings remain provisional upper bounds.")
GROUP = ("Keep the split-metacyclic construction rule, resizing only m,r,s to 29,7,16. The action has order seven because 16^7=1 mod 29 and 16!=1. This odd-order nonabelian bed is absent from BEDS. Removing the parent's involutions and binomial columns motivates the test, but establishes no distance transfer. The parent measured five distance-six codes with k=690 despite its homology-aware rank-gain selection.")
SHAPES = ("Change to A=(1,2), B=(1,3): q=203 gives n=1421, k>=406 and guaranteed rate 2/7=0.285714, targeting cell 0.300. The shape permits q<=214. Unequal components have 406 variables/203 checks and 609 variables/203 checks, nominal rates one-half and two-thirds. Parent minima occupied sector-one Acol1/2 across Bcol0/1/2; remove its three binomial B columns rather than enlarge their annihilators. Authoritative full ranks must retain [0.275,0.325).")
FILLING = ("Every submitted cell has three distinct terms, so HX weight is nine and HZ weight twelve. Twenty cards pair six component quotient compositions, changing relative support geometry rather than applying automorphisms. Native noncommuting Sidon triples generate the full group. Monomial blocks come only from lifted_matrices; final component extraction confirms their assembly. Exact component kernel-four gates, retained genuine null words, C7-orbit restrictions, witness-directed cell replacement and independent finalist challenges select labels. Actual commuting logical placements are classified and reduced modulo original stabilizers before pairing. Full ranks, weights, CSS and rate are verified; no finite search certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r030_m29r7s16_12x13_cubic_retained_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 203

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C29:C7 requires order 203")
    # Same split-metacyclic rule: (a,b) represents x^a*y^b.
    m, r, s = 29, 7, 16
    assert m*r == q and pow(s, r, m) == 1 and s != 1
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

A_SHAPE = (1, 2)               # (ra, ca)
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
    """Extract components from the final authoritative matrix pair."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = [
        sum(((hx[(i * cb) * q + t] >> (j * cb * q)) & mask)
            << (j * q) for j in range(ca))
        for i in range(ra) for t in range(q)
    ]
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
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


def _retired_affine_annihilator_ctx(rng, G, idx):
    """Inactive order-110 predecessor; never called by the new policy."""
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
    """Twenty cubic quotient-composition cards with binding witnesses."""
    q = G.q
    assert q == 203 and A_SHAPE == (1, 2) and B_SHAPE == (1, 3)
    card, cohort = idx % 20, idx // 20
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    ident = G.idx(G.e)
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

    # Batch authoritative monomial lifts to avoid large dummy matrices.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j * q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j * q)) & mask for row in zz])

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length() - 1
            word ^= low

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    # Relative quotient triples: consecutive, uneven, and repeated-fiber.
    # Their compositions constrain actual cells throughout replacement.
    patterns = ((1, 2), (1, 3), (0, 1))
    a_profiles = ((0, 0), (0, 1), (1, 1),
                  (0, 2), (1, 2), (2, 2))
    b_profiles = ((0, 0, 0), (0, 0, 1), (0, 1, 1),
                  (0, 0, 2), (0, 1, 2), (0, 2, 2))
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    assert len(plans) == 20
    pa, pb = plans[card]
    recipes = {"A": a_profiles[pa], "B": b_profiles[pb]}

    def generates(a, b):
        reached, pending = {ident}, [ident]
        for x in pending:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    pending.append(z)
        return len(reached) == q

    def fresh_cell(kind):
        b1, b2 = patterns[kind]
        for _ in range(512):
            a = rng.randrange(1, 29) if b1 == 0 else rng.randrange(29)
            u = G.idx((a, b1))
            v = G.idx((rng.randrange(29), b2))
            cell = (ident, u, v)
            if len(set(cell)) != 3 or table[u][v] == table[v][u]:
                continue
            left = {table[inverse[g]][h]
                    for g in cell for h in cell if g != h}
            right = {table[g][inverse[h]]
                     for g in cell for h in cell if g != h}
            if len(left) == len(right) == 6 and generates(u, v):
                return list(cell)
        raise ValueError("generating quotient-prescribed cubic cell exhausted")

    # Coordinate restrictions only: their compressed syndromes are XORs
    # of authoritative columns, not another implementation of the lift.
    h = G.idx((0, 1))
    partitions = {}
    for side in ("A", "B"):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit, value = [], root
            for _ in range(7):
                orbit.append(value)
                value = (table[value][h] if side == "A"
                         else table[h][value])
            assert value == root and len(set(orbit)) == 7
            assert set(orbit) <= remaining
            remaining.difference_update(orbit)
            orbits.append(orbit)
        count = len(recipes[side])
        partitions[side] = [
            sum(1 << (j * q + c) for c in orbit)
            for j in range(count) for orbit in orbits
        ]

    def inspect(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            assert len(set(cell)) == len(cell) == 3
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j * q)
        width = len(cells) * q
        columns = _syndrome_columns(rows, width)
        assert all(c.bit_count() == 3 for c in columns)
        if len(set(columns)) != width:
            return None
        rank = gf2_rank(rows)
        if rank < q - 4:
            return None
        four = _four_cycles(rows)
        return {
            "cells": [list(c) for c in cells],
            "rows": rows, "columns": columns, "rank": rank,
            "cycles": four, "words": set(), "d": width + 1,
            "signature": tuple(tuple(sorted(c)) for c in cells),
        }

    def kernel_search(columns, diagnostic, trials, initial=()):
        """Retain genuine null words; high readings certify no distance."""
        width = len(columns)
        retained = sorted({w for w in initial if w},
                          key=int.bit_count)[:20]
        assert all(syndrome(columns, w) == 0 for w in retained)
        order = list(range(width))
        best = min(retained, key=int.bit_count) if retained else None

        def observe(word):
            nonlocal best
            if word and (best is None or
                         word.bit_count() < best.bit_count()):
                best = word

        for turn in range(trials):
            diagnostic.shuffle(order)
            if turn % 4 == 3 and retained:
                active = set(bits(diagnostic.choice(retained[:6])))
                order.sort(key=lambda c: c not in active)
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
            light = sorted(set(basis + retained),
                           key=int.bit_count)[:12]
            grown = list(light) + retained
            moves = basis + retained
            for a in light:
                for b in moves:
                    word = a ^ b
                    observe(word)
                    if word and word.bit_count() <= best.bit_count() + 4:
                        grown.append(word)
                if len(grown) > 160:
                    grown = sorted(set(grown), key=int.bit_count)[:32]
            diagnostic.shuffle(moves)
            for word in light[:5]:
                for _ in range(3):
                    before = word.bit_count()
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    observe(word)
                    if word.bit_count() == before:
                        break
                grown.append(word)
            grown.append(best)
            retained = sorted({w for w in grown if w},
                              key=int.bit_count)[:20]
            if best.bit_count() <= 4:
                break
        assert best and syndrome(columns, best) == 0
        return best, retained

    def completion(record, diagnostic, budget):
        """Bounded sparse parity completion; exhaustion is not exclusion."""
        columns = record["columns"]
        width = len(columns)
        best = min(record["words"], key=int.bit_count)
        cap = min(16, best.bit_count() - 1)
        if cap <= 4:
            return best
        singles, pairs = {}, {}
        incidence = [0] * q
        for c, value in enumerate(columns):
            singles.setdefault(value, 1 << c)
            for r in bits(value):
                incidence[r] |= 1 << c
            for d in range(c):
                pairs.setdefault(value ^ columns[d],
                                 (1 << c) | (1 << d))
        priorities = [diagnostic.random() for _ in columns]
        roots = list(range(0, width, q))
        diagnostic.shuffle(roots)
        all_variables = (1 << width) - 1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                assert syndrome(columns, word) == 0
                best = word

        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    observe(word)
                    return
                left = min(cap, best.bit_count() - 1) - word.bit_count()
                if left <= 0 or value.bit_count() > 3 * left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    observe(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    observe(word | pair)
                left = min(cap, best.bit_count() - 1) - word.bit_count()
                if left <= 2:
                    return
                chosen, smallest = 0, width + 1
                for r in bits(value):
                    available = incidence[r] & allowed
                    size = available.bit_count()
                    if not size:
                        return
                    if size < smallest:
                        chosen, smallest = available, size
                choices = list(bits(chosen))
                choices.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in choices:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            visit(columns[root], 1 << root,
                  all_variables ^ (1 << root))
        return best

    def component_pool(side):
        recipe = recipes[side]
        offset = 100003 * int(side == "B")
        records, seen, witness_bank = [], set(), set()

        def admit(cells):
            record = inspect(cells, side)
            if record is None or record["signature"] in seen:
                return None
            seen.add(record["signature"])
            return record

        def challenge(record, turn, trials, fixed_trials, budget=0):
            columns = record["columns"]
            record["words"].update(
                w for w in witness_bank if syndrome(columns, w) == 0)
            local = random.Random(seed + offset + 700001 * turn)
            word, retained = kernel_search(
                columns, local, trials, record["words"])
            record["words"].update(retained)
            record["words"].add(word)

            groups = partitions[side]
            compressed = [syndrome(columns, group) for group in groups]
            fw, _ = kernel_search(
                compressed,
                random.Random(seed + offset + 1700003 + 900001 * turn),
                fixed_trials)
            actual = 0
            for c in bits(fw):
                actual ^= groups[c]
            assert actual.bit_count() == 7 * fw.bit_count()
            assert syndrome(columns, actual) == 0
            record["words"].add(actual)

            if budget:
                record["words"].add(completion(
                    record,
                    random.Random(seed + offset + 2700011 + turn),
                    budget))
            record["words"] = set(sorted(
                record["words"], key=int.bit_count)[:20])
            record["d"] = min(
                record["d"],
                min(w.bit_count() for w in record["words"]))
            witness_bank.update(record["words"])
            if len(witness_bank) > 80:
                keep = sorted(witness_bank, key=int.bit_count)[:56]
                witness_bank.clear()
                witness_bank.update(keep)
            assert all(syndrome(columns, w) == 0
                       for w in record["words"])

        def merit(record):
            for w in witness_bank:
                if (w.bit_count() < record["d"]
                        and syndrome(record["columns"], w) == 0):
                    record["words"].add(w)
                    record["d"] = w.bit_count()
            return record["d"], -record["cycles"], record["rank"]

        for restart in range(3):
            pool = []
            for _ in range(64):
                record = admit([fresh_cell(kind) for kind in recipe])
                if record is not None:
                    pool.append(record)
            pool.sort(key=lambda r: (r["cycles"], -r["rank"]))
            selected = []
            for record in pool:
                if _pairs_no_kernel4(record["columns"]) is None:
                    continue
                challenge(record, 1 + 100 * restart, 8, 12)
                selected.append(record)
                if len(selected) == 8:
                    break
            if not selected:
                continue
            current = max(selected, key=merit)

            # Replace real terms in implicated blocks, retaining the card.
            for step in range(20):
                cells = [list(c) for c in current["cells"]]
                word = min(current["words"], key=int.bit_count)
                implicated = sorted({c // q for c in bits(word)})
                j = rng.choice(implicated)
                cells[j] = fresh_cell(recipe[j])
                proposed = admit(cells)
                if proposed is None:
                    continue
                if proposed["cycles"] > current["cycles"] + q:
                    continue
                if _pairs_no_kernel4(proposed["columns"]) is None:
                    continue
                proposed["words"].update(
                    w for w in current["words"]
                    if syndrome(proposed["columns"], w) == 0)
                challenge(proposed, 11 + step + 100 * restart, 5, 8)
                selected.append(proposed)
                if merit(proposed) >= merit(current):
                    current = proposed
                elif (proposed["d"] == current["d"]
                      and rng.random() < 0.10):
                    current = proposed

            finalists = sorted(selected, key=merit, reverse=True)[:3]
            for at, record in enumerate(finalists):
                challenge(record, 51 + at + 100 * restart, 40, 48, 16000)
                challenge(record, 71 + at + 100 * restart, 32, 48, 16000)
            finalists.sort(key=merit, reverse=True)
            records.extend(finalists)
            break
        if not records:
            raise ValueError("no cubic component passed kernel-four gates")
        return sorted(records, key=merit, reverse=True)[:2]

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

    def localized_readings(a, b, hx, hz, n):
        """Verify and reduce nontrivial logicals in original coordinates."""
        answer = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            found = set()
            for word in sorted(record["words"], key=int.bit_count)[:6]:
                for location in range(3 if side == "A" else 2):
                    if side == "A":
                        actual = sum(
                            ((word >> (j * q)) & mask)
                            << ((3 * j + location) * q)
                            for j in range(2))
                    else:
                        actual = word << (location * 3 * q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        found.add(actual)
            reading = n + 1
            local = random.Random(
                seed + (4100009 if side == "A" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(found, key=int.bit_count)[:6]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    winner = actual
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            candidate = partial ^ second
                            if candidate.bit_count() < winner.bit_count():
                                winner = candidate
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            answer.append(reading)
        return tuple(answer)

    aa, bb = component_pool("A"), component_pool("B")
    proposals = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
            hx, hz, n = lifted_matrices(P, [a["cells"]], [b["cells"]])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1421 and k >= 406
                    and 275 * n <= 1000 * k < 325 * n):
                continue
            ar, br = _verified_components(hx, hz, q)
            assert ar == a["rows"] and br == b["rows"]
            logicals = localized_readings(a, b, hx, hz, n)
            detected = min(18, a["d"], b["d"], *logicals)
            key = (k * detected * detected, detected,
                   -a["cycles"] - b["cycles"])
            proposals.append((key, A, B, hx, hz, k, a, b, logicals))
    proposals.sort(key=lambda entry: entry[0], reverse=True)

    for key, A, B, hx, hz, k, a, b, logicals in proposals:
        if _pairs_no_kernel4(_syndrome_columns(hx, 1421)) is None:
            continue
        if _pairs_no_kernel4(_syndrome_columns(hz, 1421)) is None:
            continue
        sa, sb = [[3, 3]], [[3, 3, 3]]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx) == 9
        assert max(row.bit_count() for row in hz) == 12
        assert css_ok(hx, hz)
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (pa, pb, cohort),
            "quotient_compositions": (recipes["A"], recipes["B"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_readings": (a["d"], b["d"]),
            "component_four_cycles": (a["cycles"], b["cycles"]),
            "verified_localized_logical_readings": logicals,
            "selection_quality": key, "target_cell": 0.300,
        }
    raise ValueError("cubic pairings missed full kernel-four/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cubic cell without further draws."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 3
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
# REFUTED: Homology-aware larger-annihilator selection improves this affine bed beyond fitness 17.371; the current program still measured five k=690, distance-six codes and no accepted larger rank gain.
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
