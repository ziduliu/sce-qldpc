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
CONCEPT = ("Twisted C20-by-C4(s=3,t=10), order 80, A=(1,3), B=(2,5): cubic generating operators with binding central-C2-fixed and unrestricted witness challenges should improve cell 0.350 beyond 50.824 through readings fourteen, or verified dimension gain while retaining twelve. Five fourteen readings at the rank bound score 69.1765. Measurement can refute this hypothesis; no search reading certifies distance.")
GROUP = ("Jump to the nonsplit presentation x^20=1, y*x*y^-1=x^3, y^4=x^10. Since 3^4=1 modulo 20 and 3 fixes x^10, the carry-twisted multiplication is associative. Every x^a*y has fourth power x^10, so no order-four complement splits this extension. Its centre is {1,x^10}, unlike the parent's order-four centre, and its order is 80 rather than 84. This bed is absent from BEDS. The parent's five [1428,504,12] measurements motivate a different group law, not assumed distance transfer.")
SHAPES = ("Retain A=(1,3), B=(2,5). Factor seventeen with q=80 gives n=1360 and k>=480, guaranteed rate 6/17=0.352941 in cell 0.350; the shapes allow q<=88. Unequal components have 240 variables/80 checks and 400 variables/160 checks, nominal rates 2/3 and 3/5. Parent measured minima were entirely sector one, particularly across Acol0..2 at Bcol3/4. Preserve these component rates and enforce exact 13*n<=40*k<15*n on the complete matrices.")
FILLING = ("Twenty cards cross five soft A-defect/annihilator objectives with four non-permutation-equivalent B support profiles. A cells and B columns remain cubic, with B row totals seven/eight: HX<=12 and HZ<=11. Scaffold-derived monomials supply all operators, joins, ranks and syndromes. Add an exact central-C2-fixed challenge through five compressed variables; every hit expands to a verified component word of weight <=10 and becomes a binding retained witness. Preserve unrestricted witness repair and independent deepening. Cap pairing readings at fourteen, and explore a second portfolio unless fourteen or verified dimension gain with twelve is reached. Original stabilizer spans verify localized logicals; final weights, rate and CSS remain gates. Restricted exclusions and finite readings certify no quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_twistedC20_C4_s3t10_13x25_C2challenge_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 80

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("twisted C20-by-C4 requires order 80")
    # (a,b)=x^a*y^b, with 0<=a<20 and 0<=b<4.
    # y*x*y^-1=x^3 and y^4=x^10.
    # 3^4=1 mod 20 and 3*10=10 mod 20 make the carry consistent.
    powers = (1, 3, 9, 7)
    els = [(a, b) for b in range(4) for a in range(20)]

    def red(g):
        return (int(g[0]) % 20, int(g[1]) % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        carry, exponent = divmod(b + d, 4)
        return ((a + powers[b] * c + 10 * carry) % 20,
                exponent)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        carry = (b + bi) // 4
        return ((-powers[bi] * (a + 10 * carry)) % 20, bi)

    def idx(g):
        a, b = red(g)
        return 20 * b + a

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


def _retired_110_annihilator_ctx(rng, G, idx):
    """Unused historical order-110 implementation; never called."""
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
    """Twenty twisted-group cubic cards with binding central challenges."""
    q = G.q
    assert q == 80 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    model, topology = card // 4, card % 4
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xC20A4D10 + 1000003 * idx))
    mask = (1 << q) - 1
    ident = G.idx(G.e)

    # The profiles differ in the number and placement of zero cells,
    # not just row/column permutations of one positive support profile.
    profiles = (
        ((2,1), (2,1), (1,2), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (1,2), (1,2)),
        ((3,0), (3,0), (0,3), (0,3), (1,2)),
        ((3,0), (3,0), (1,2), (1,2), (0,3)),
    )
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in profiles[topology]]
          for i in range(2)]
    assert sorted(map(sum, sb)) == [7, 8]
    assert all(sum(sb[i][j] for i in range(2)) == 3
               for j in range(5))
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

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

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def dependencies(vectors):
        """Exact dependencies in the original input coordinates."""
        pivots, basis = {}, []
        for i, source in enumerate(vectors):
            value, word = source, 1 << i
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

    # Every action block is extracted from lifted_matrices. Assembly
    # below uses only GF2 linearity and disjoint column concatenation.
    monomials = [[[g] for g in range(q)]]
    raw_a, _, _ = lifted_matrices(P, monomials, [[[]]])
    _, raw_b, _ = lifted_matrices(P, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g * q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def component(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nr * q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i*q+t] ^= value << (j*q)
        return rows

    orders = []
    for g in range(q):
        h = ident
        for order in range(1, q + 1):
            h = table[h][g]
            if h == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group")
    long_ratios = [g for g in range(q) if orders[g] >= 4]
    assert long_ratios

    def generates(a, b):
        reached, pending = {ident}, [ident]
        for value in pending:
            for g in (a, b):
                image = table[value][g]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    def sidon(support):
        left, right = set(), set()
        for a in support:
            for b in support:
                if a != b:
                    left.add(table[inverse[a]][b])
                    right.add(table[a][inverse[b]])
        return len(left) == len(right) == 6

    library, operator_rows, memberships = [], {}, {}
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            support = tuple(sorted((ident, a, b)))
            # One representative per right-column-gauge operator class.
            representative = min(
                tuple(sorted(table[g][inverse[root]] for g in support))
                for root in support)
            if support != representative:
                continue
            if not sidon(support) or not generates(a, b):
                continue
            rows = component([[list(support)]], "A")
            columns = _syndrome_columns(rows, q)
            if _pairs_no_kernel4(columns) is None:
                continue
            library.append(support)
            operator_rows[support] = rows
            left = dependencies(rows)
            assert len(left) == q - gf2_rank(rows)
            if not left or len(left) > 15:
                continue
            values = [0]
            # A bounded proposal catalog, not an exclusion argument.
            for word in left[:6]:
                values += [old ^ word for old in values]
            for word in set(values[1:]) | set(left):
                # Avoid constraints living only in the C12 quotient.
                normal_mask = (1 << 20) - 1
                quotient_only = all(
                    ((word >> (20*t)) & normal_mask) in (0, normal_mask)
                    for t in range(4))
                if not quotient_only:
                    memberships.setdefault(word, set()).add(support)
    library.sort()
    if len(library) < 3:
        raise ValueError("too few generating cubic operator classes")

    unique_catalogs = {}
    for functional, members in memberships.items():
        members = tuple(sorted(members))
        if len(members) >= 3:
            unique_catalogs.setdefault(members, functional)
    catalog = [(functional, members)
               for members, functional in unique_catalogs.items()]

    def catalog_key(item):
        functional, members = item
        weight = functional.bit_count()
        policy = (
            (-len(members), weight),
            (weight, -len(members)),
            (-len(members), abs(weight - q//2)),
            (-weight, -len(members)),
            (abs(weight - q//2), -len(members)),
        )[model]
        return policy + (functional,)

    catalog.sort(key=catalog_key)
    # Soft objectives only: inaccessible defects do not void a slot.
    # Complete ranks, rather than a module-size formula, decide k.
    desired_defect = (0, 2, 4, 6, 8)[model]

    def fresh_a(attempt):
        # Independent full-rank candidates compete with shared-module
        # joins; an empty or unfavorable catalog cannot void every slot.
        options, functional = library, None
        if catalog and model and attempt % 4 != 3:
            limit = min(256, len(catalog))
            position = (31*topology + 43*cohort + 17*attempt + model) % limit
            functional, options = catalog[position]
        cells = rng.sample(options, 3)
        return [[list(cell) for cell in cells]], (functional, options)

    def draw_cell(count):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        root = rng.randrange(q)
        if count == 2:
            return [root, table[root][rng.choice(long_ratios)]]
        assert count == 3
        # Independent cell holonomies, not a common gauge applied to
        # an already-built code.
        return [table[root][g] for g in rng.choice(library)]

    def fresh_b():
        return [[draw_cell(count) for count in row] for row in sb]

    def inspect(grid, side, frame=None):
        rows = component(grid, side)
        nr, nc = len(grid), len(grid[0])
        width = nc*q
        rank = gf2_rank(rows)
        if side == "A":
            if not q-9 <= rank <= q:
                return None
        elif not 2*q-8 <= rank <= 2*q:
            return None
        columns = _syndrome_columns(rows, width)
        if 0 in columns or len(set(columns)) != width:
            return None
        assert all(column.bit_count() == 3 for column in columns)

        uses, four = {}, 0
        adjacency = [0] * len(rows)
        for column in columns:
            checks = list(bits(column))
            for at, u in enumerate(checks):
                adjacency[u] |= column ^ (1 << u)
                for v in checks[:at]:
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
        # Projection excess is only a selection statistic in the
        # presence of four-cycles, not an exact six-cycle count.
        excess = max(0, triangles - width)
        cost = (400 + 100*topology)*four + excess
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            rank=rank, cycles=(four, excess), cost=cost,
            signature=signature(grid), frame=frame, side=side,
            d=width+1, words=[], free4=None)

    central = G.idx((10, 0))
    assert central != ident and table[central][central] == ident
    assert all(table[central][g] == table[g][central]
               for g in range(q))
    pending, central_orbits = set(range(q)), []
    while pending:
        u = min(pending)
        v = table[u][central]
        assert u != v and v in pending
        central_orbits.append((u, v))
        pending.difference_update((u, v))
    assert len(central_orbits) == q // 2

    def fixed_short(record):
        """Exact <=5 detection ONLY in the central-C2-fixed restriction.

        Restricted syndromes are XORs of authoritative original columns.
        A hit expands to a genuine word of weight <=10. A miss is not
        an unrestricted component or quantum-distance certificate.
        """
        if "C2_fixed_word" in record:
            return record["C2_fixed_word"]
        original = record["columns"]
        columns, expansions = [], []
        for block in range(len(original) // q):
            for u, v in central_orbits:
                a, b = block*q + u, block*q + v
                columns.append(original[a] ^ original[b])
                expansions.append((1 << a) | (1 << b))

        def finish(word):
            actual = 0
            for c in bits(word):
                actual ^= expansions[c]
            assert actual
            assert actual.bit_count() == 2 * word.bit_count()
            assert syndrome(original, actual) == 0
            record["C2_fixed_word"] = actual
            return actual

        singles, pairs = {}, {}
        for c, value in enumerate(columns):
            word = 1 << c
            if not value:
                return finish(word)
            previous = singles.get(value)
            if previous is not None:
                return finish(word ^ previous)
            singles[value] = word

        for a, value in enumerate(columns):
            for b in range(a):
                target = value ^ columns[b]
                word = (1 << a) | (1 << b)
                previous = singles.get(target)
                if previous is None:
                    previous = pairs.get(target)
                if previous is not None:
                    return finish(word ^ previous)
                pairs[target] = word

        # Centrality makes the quotient regular action valid within
        # each orbit block. Move one occupied orbit to its representative
        # and partition a five-word into a rooted triple and a pair.
        step = q // 2
        for root in range(0, len(columns), step):
            for a, value in enumerate(columns):
                if a == root:
                    continue
                target = columns[root] ^ value
                occupied = (1 << root) | (1 << a)
                for b in range(a):
                    if b == root:
                        continue
                    previous = pairs.get(target ^ columns[b])
                    used = occupied | (1 << b)
                    if previous is not None and not previous & used:
                        word = previous | used
                        assert word.bit_count() == 5
                        return finish(word)

        record["C2_fixed_word"] = None
        return None

    def challenge(record, trials, diagnostic_seed):
        """Binding fixed-subspace and unrestricted genuine kernel witnesses."""
        local = random.Random(diagnostic_seed)
        columns = record["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(record["words"])
        fixed = fixed_short(record)
        if fixed is not None:
            retained.append(fixed)
        reading = record["d"]

        def observe(word):
            nonlocal reading
            if word:
                reading = min(reading, word.bit_count())

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
        for _ in range(trials):
            local.shuffle(order)
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
                    if candidate and candidate.bit_count() <= reading+2:
                        additions.append(candidate)
            moves = retained + basis
            local.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
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
                              key=lambda w: (w.bit_count(), w))[:24]
            if reading <= 4:
                break
        assert retained
        assert all(syndrome(columns, word) == 0 for word in retained)
        record["words"] = retained
        record["d"] = min(reading, min(map(int.bit_count, retained)))

    def merit(record):
        defect_preference = (
            -abs(q-record["rank"]-desired_defect)
            if record["side"] == "A" else record["rank"])
        return (bool(record["free4"]), min(14, record["d"]),
                defect_preference, min(20, record["d"]), -record["cost"])

    def pool(side, restart):
        offset = 100003*int(side == "B") + 300007*restart
        records, seen = [], set()
        for attempt in range(80):
            if side == "A":
                grid, frame = fresh_a(attempt + 80*restart)
            else:
                grid, frame = fresh_b(), None
            record = inspect(grid, side, frame)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            records.append(record)
        records.sort(key=lambda r: (r["cost"], r["signature"]))
        # Include both good geometry and requested-defect proposals.
        ordered = records[:16]
        if side == "A":
            ordered += sorted(records, key=lambda r: (
                abs(q-r["rank"]-desired_defect), r["cost"]))[:12]
        tested, selected = set(), []
        for record in ordered:
            if record["signature"] in tested:
                continue
            tested.add(record["signature"])
            record["free4"] = (
                _pairs_no_kernel4(record["columns"]) is not None)
            # Odd column degree makes odd-weight component null words
            # impossible; <=4 exclusion therefore also excludes five.
            challenge(record, 12, seed + offset + 1009*len(selected))
            selected.append(record)
            if len(selected) >= 10 and sum(
                    bool(r["free4"]) for r in selected) >= 5:
                break
        if not selected:
            return []

        known = set(word for record in selected for word in record["words"])

        def refresh(record):
            for word in known:
                if word.bit_count() < record["d"] and (
                        syndrome(record["columns"], word) == 0):
                    record["d"] = word.bit_count()
                    record["words"] = sorted(
                        set(record["words"]) | {word},
                        key=lambda w: (w.bit_count(), w))[:24]

        current = max(selected, key=merit)
        for step in range(20):
            refresh(current)
            word = min(current["words"], key=int.bit_count)
            implicated = sorted({c//q for c in bits(word)})
            j = rng.choice(implicated)
            grid = copy_grid(current["grid"])
            frame = current["frame"]
            if side == "A":
                options = frame[1] if frame is not None else library
                occupied = {tuple(sorted(cell)) for cell in grid[0]}
                available = [cell for cell in options if cell not in occupied]
                if not available:
                    continue
                grid[0][j] = list(rng.choice(available))
            else:
                # Replace one or both cells in an implicated variable
                # column, preserving its degree and row budgets.
                changed = range(2) if step % 4 == 3 else (rng.randrange(2),)
                for i in changed:
                    if sb[i][j]:
                        grid[i][j] = draw_cell(sb[i][j])
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side, frame)
            if proposed is None:
                continue
            proposed["free4"] = (
                _pairs_no_kernel4(proposed["columns"]) is not None)
            if current["free4"] and not proposed["free4"]:
                continue
            proposed["words"] = [
                old for old in current["words"]
                if syndrome(proposed["columns"], old) == 0]
            challenge(proposed, 8, seed + offset + 700001*(step+1))
            known.update(proposed["words"])
            refresh(proposed)
            selected.append(proposed)
            if merit(proposed) >= merit(current):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists, distinct = [], set()
        for record in selected:
            if record["signature"] in distinct:
                continue
            distinct.add(record["signature"])
            refresh(record)
            challenge(record, 48, seed + offset + 1700009)
            challenge(record, 32, seed + offset + 2700011)
            known.update(record["words"])
            finalists.append(record)
            if len(finalists) == 4:
                break
        for record in finalists:
            refresh(record)
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

    def span(rows):
        pivots = {}
        for source in rows:
            word = source
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

    def logical_readings(a, b, hx, hz, xp, zp, n):
        """Actual sector-one logical witnesses in the ORIGINAL quotient."""
        answer = []
        for kind, record, opposite, stabilizers, pivots in (
                ("Z", a, hx, hz, zp), ("X", b, hz, hx, xp)):
            candidates = set()
            for word in record["words"][:4]:
                assert syndrome(record["columns"], word) == 0
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
                        candidates.add(actual)
            reading = n+1
            local = random.Random(seed + 4100009 + 100003*int(kind == "X"))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates,
                                 key=lambda w: (w.bit_count(), w))[:4]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best_word, best_weight = actual, actual.bit_count()
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            candidate = partial ^ second
                            if candidate.bit_count() < best_weight:
                                best_word, best_weight = candidate, candidate.bit_count()
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            # No found embedding means a search miss, not exclusion.
            answer.append(reading)
        return tuple(answer)

    best, best_key = None, None
    for restart in range(2):
        aa, bb = pool("A", restart), pool("B", restart)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            bool(ab[0]["free4"] and ab[1]["free4"]),
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"]+ab[1]["d"],
            -ab[0]["cost"]-ab[1]["cost"]), reverse=True)
        for a, b in proposals:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            xp, zp = span(hx), span(hz)
            k = n - len(xp) - len(zp)
            if not (n == 1360 and k >= 6*q and
                    13*n <= 40*k < 15*n):
                continue
            actual_a = [
                sum(((hx[t] >> (5*j*q)) & mask) << (j*q)
                    for j in range(3))
                for t in range(q)]
            actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logicals = logical_readings(a, b, hx, hz, xp, zp, n)
            # Limit selection pressure on optimistic finite searches.
            # Fourteen is an experimental target, not a lower bound.
            detected = min(14, a["d"], b["d"], *logicals)
            key = (
                bool(a["free4"] and b["free4"]),
                k*detected*detected, detected,
                a["d"]+b["d"], -a["cost"]-b["cost"])
            if best_key is not None and key <= best_key:
                continue
            best_key = key
            best = (
                dict(
                    A=[[[G.elements[g] for g in cell] for cell in row]
                       for row in a["grid"]],
                    B=[[[G.elements[g] for g in cell] for cell in row]
                       for row in b["grid"]],
                    idx=idx, k=k, card=(model, topology, cohort),
                    component_ranks=(a["rank"], b["rank"]),
                    component_readings=(a["d"], b["d"]),
                    component_kernel4_free=(a["free4"], b["free4"]),
                    component_cycles=(a["cycles"], b["cycles"]),
                    common_left_functional=(
                        a["frame"][0] if a["frame"] is not None else None),
                    verified_logical_upper_readings=logicals,
                    requested_A_defect=desired_defect,
                    central_fixed_kernel5_free=(
                        a["C2_fixed_word"] is None,
                        b["C2_fixed_word"] is None),
                    central_fixed_witness_weights=(
                        a["C2_fixed_word"].bit_count()
                        if a["C2_fixed_word"] is not None else None,
                        b["C2_fixed_word"].bit_count()
                        if b["C2_fixed_word"] is not None else None),
                    target_cell=0.350,
                    pairing_merit=key),
                hx, hz)
        if best is not None and best_key[0] and (
                best_key[2] >= 14 or
                (best[0]["k"] > 6*q and best_key[2] >= 12)):
            break
    if best is None:
        raise ValueError("odd-column pairings missed the exact 0.350 rate gate")
    answer, hx, hz = best
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert css_ok(hx, hz)
    assert all(len(cell) == len({G.idx(g) for g in cell})
               for side in ("A", "B")
               for row in answer[side] for cell in row)
    return answer


def CELL(rng, G, side, i, j, ctx):
    """Return every stored cell of the new wide protographs."""
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
# REFUTED: Shared-left odd-column joins improve cell 0.350 beyond 50.824 through selected rank gain or distance thirteen; all five current measured codes instead had k=504 and distance reading twelve.
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
