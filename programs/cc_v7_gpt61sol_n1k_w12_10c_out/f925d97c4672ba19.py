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
CONCEPT = ("On (C11 semidirect_4 C5) x C2, order 110, A=(1,3), B=(1,4), centrally mixed trinomials sharing a free dual-number annihilator should produce verified k=720 codes without component kernels through five. Five new distance-six readings would score 18.1259 and improve cell 0.500; failure to obtain that five-code result refutes the measurement hypothesis.")
GROUP = ("Frozen (C11:C5) x C2, order 110. Its central involution gives epsilon=z+1 with epsilon^2=0, and the faithful F1024 quotient representation is F4-linear. Current quotient-annihilator proposals produced k=720 candidates screening at four, whereas all five measured k=690 codes remained at six. Enforcing the nilpotent compatibility equation is a new proposal mechanism, not an inferred rank or distance certificate.")
SHAPES = ("Frozen A=(1,3), B=(1,4): n=1430, k>=660, guaranteed rate 6/13, and maximum lift order 115. Unequal components have 330 and 440 variables with 110 checks each. Parent minima were entirely sector 1, especially Acol1/2 across Bcol0/1/2. Actual full ranks must place every accepted filling in [0.475,0.525); component rank deficits are never substituted for measured k.")
FILLING = ("Keep A=(3,3,3), B=(2,2,2,3), hence both CSS row caps twelve. Enumerate mixed central supports satisfying P0*(v)=0 and P0*(w)=P1*(v), proposing a shared free annihilator v+epsilon*w. Collapse scalar and nilpotent rescalings of proposal constraints, not submitted code diversity. Screen verified cell/pair/triple kernels and A ranks 80/90; fresh quotient-line central deformations provide rank-100 alternatives. Every B backbone uses one central-even order-eleven translation and two central-odd ratios. Twenty cards cross four cycle objectives with five trinomial relative spectra. Independently challenge readings and require full rank-band, kernel-four, weight and CSS gates. Readings remain provisional upper bounds.")
HOW_AB = FILLING
FAMILY_NAME = "r050_Frob55xC2_13x14_dual_annihilator_translation_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 110

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius55 x C2 requires order 110")
    # Direct product: (a,b,z), with y*x*y^-1 = x^4
    # and z a separate central involution. 4 has order five mod 11.
    powers = [pow(4, b, 11) for b in range(5)]
    els = [(a, b, z) for z in range(2)
           for b in range(5) for a in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 5, int(g[2]) % 2)

    def mul(g, h):
        a, b, z = red(g)
        c, d, w = red(h)
        return ((a + powers[b] * c) % 11,
                (b + d) % 5, (z + w) % 2)

    def inv(g):
        a, b, z = red(g)
        bi = (-b) % 5
        return ((-powers[bi] * a) % 11, bi, z)

    def idx(g):
        a, b, z = red(g)
        return 55 * z + 11 * b + a

    return dict(elements=els, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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


def NEW_CTX(rng, G, idx):
    """Twenty quotient-line/central-mixing cards on a different group."""
    q = G.q
    assert q == 110
    card, cohort = idx % 20, idx // 20
    mixed_budget, model = card // 5, card % 5
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

    tri_profiles = ((0, 1), (1, 2), (1, 3), (2, 4), (0, 2))

    def draw_b():
        cells = []
        # Central-odd binomial ratios give a bipartite binomial backbone.
        # Coordinate prescriptions vary relative orders, not gauges.
        patterns = ((1, 1, 2), (1, 2, 3), (1, 3, 4),
                    (2, 2, 4), (1, 2, 4))
        coordinates = patterns[(model + cohort) % len(patterns)]
        for b in coordinates:
            ratio = (rng.randrange(11), b, 1)
            root = G.pick(rng)
            cells.append([root, G.mul(root, ratio)])
        b1, b2 = tri_profiles[model]
        for _ in range(128):
            a = rng.randrange(1, 11) if b1 == 0 else rng.randrange(11)
            g = (a, b1, rng.randrange(2))
            h = (rng.randrange(11), b2, rng.randrange(2))
            if table[G.idx(g)][G.idx(h)] == table[G.idx(h)][G.idx(g)]:
                continue
            root = G.pick(rng)
            cells.append([root, G.mul(root, g), G.mul(root, h)])
            return cells
        raise ValueError("noncommuting direct-product trinomial unavailable")

    def select_pool(side, options=None, budget=0):
        records, signatures = [], set()
        for trial in range(112):
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
        for record in records[:24]:
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

        # If a particular central-mixing budget is unfavorable, retain a
        # fresh pure-central proposal. Its B card remains independently new.
        if not candidates and mixed_budget:
            for turn in range(min(4, count)):
                functional, options = catalog[(card + turn) % count]
                for a in select_pool("A", options, 0):
                    for b in bb:
                        key = (
                            min(a["floor"], b["floor"]),
                            min(a["reading"], b["reading"]),
                            a["reading"] + b["reading"],
                            -a["cycles"] - b["cycles"])
                        candidates.append((key, a, b, functional, 0))

    # A broader bank also finds accidental common annihilators. This
    # avoids making the representation-level catalog a feasibility gate.
    if not candidates:
        broad = tuple(
            (identity, b, a)
            for a in range(1, 55) for b in range(1, a))
        for budget in (mixed_budget, 0):
            for a in select_pool("A", broad, budget):
                for b in bb:
                    key = (
                        min(a["floor"], b["floor"]),
                        min(a["reading"], b["reading"]),
                        a["reading"] + b["reading"],
                        -a["cycles"] - b["cycles"])
                    candidates.append((key, a, b, None, budget))

    candidates.sort(key=lambda item: item[0], reverse=True)
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
    """Active model: compatible central lifts, not quotient-only mixing."""
    from itertools import combinations

    q = G.q
    assert q == 110
    assert A_SHAPE == (1, 3) and B_SHAPE == (1, 4)
    assert _grid_weights([[3, 3, 3]], [[2, 2, 2, 3]]) == 12 <= W_MAX
    card, cohort = idx % 20, idx // 20
    trade, model = card // 5, card % 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    assert ident == 0

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
            return inverses[g]

    probe = Probe()

    # Cache monomials from the verified constructor, not a separate lift.
    singleton_row = [[[g] for g in range(q)]]
    left, _, _ = lifted_matrices(probe, singleton_row, [[[]]])
    _, right, _ = lifted_matrices(probe, [[[]]], singleton_row)
    blocks = {
        "A": [[(row >> (j * q)) & mask for row in left]
              for j in range(q)],
        "B": [[(row >> (j * q)) & mask for row in right]
              for j in range(q)],
    }
    del left, right

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length() - 1
            word ^= low

    def canonical_cell(ids):
        # Independent right-column gauges preserve the left annihilator.
        return min(
            tuple(sorted(table[g][inverses[root]] for g in ids))
            for root in ids)

    atoms = {}

    def atom(side, ids):
        key = (side, tuple(ids))
        if key not in atoms:
            rows = [0] * q
            for g in ids:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value
            columns = _syndrome_columns(rows, q)
            atoms[key] = (rows, columns)
        return atoms[key]

    def assemble(side, cells):
        rows = [0] * q
        columns = []
        for j, ids in enumerate(cells):
            rr, cc = atom(side, ids)
            columns.extend(cc)
            for r, value in enumerate(rr):
                rows[r] |= value << (j * q)
        return rows, columns

    def geometry(rows, columns):
        four = _four_cycles(rows)
        adjacency = [0] * q
        forced = 0
        for column in columns:
            degree = column.bit_count()
            forced += degree * (degree - 1) * (degree - 2) // 6
            for r in bits(column):
                adjacency[r] |= column ^ (1 << r)
        triangles = 0
        for r, neighbors in enumerate(adjacency):
            for s in bits(neighbors & (-1 << (r + 1))):
                triangles += (neighbors & adjacency[s]
                              & (-1 << (s + 1))).bit_count()
        # Projection excess is a heuristic when four-cycles are present.
        return four, max(0, triangles - forced)

    def cycle_key(stats):
        four, excess = stats
        if trade == 0:
            return four, excess
        if trade == 1:
            return excess, four
        if trade == 2:
            return 4 * four + excess, four
        return 8 * four + excess, excess

    def inspect(side, cells):
        cells = tuple(tuple(sorted(cell)) for cell in cells)
        if len(set(cells)) != len(cells):
            return None
        rows, columns = assemble(side, cells)
        rank = gf2_rank(rows)
        if side == "A":
            if rank not in (80, 90, 100):
                return None
            assert all(column.bit_count() == 3 for column in columns)
            if _pairs_no_kernel4(columns) is None:
                return None
            # Odd column weights exclude every odd-weight kernel word.
            floor = 6
        else:
            if rank != q:
                return None
            floor, columns = _component_floor(rows, 4 * q, q)
            if floor < 5:
                return None
        return {
            "ids": cells, "rows": rows, "columns": columns,
            "rank": rank, "floor": floor,
            "cycles": geometry(rows, columns),
            "reading": len(columns) + 1,
        }

    def challenge(records, side):
        records.sort(key=lambda r: (
            -r["floor"], r["rank"] if side == "A" else 0,
            cycle_key(r["cycles"])))
        records = records[:8]
        for record in records:
            diagnostic = random.Random(seed + 1009 * int(side == "B"))
            record["reading"] = _component_reading(
                record["columns"], diagnostic, 12)

        def merit(record):
            return (record["floor"],
                    -record["rank"] if side == "A" else 0,
                    record["reading"],
                    tuple(-x for x in cycle_key(record["cycles"])))

        records.sort(key=merit, reverse=True)
        records = records[:3]
        for record in records:
            diagnostic = random.Random(
                seed + 700001 + 1009 * int(side == "B"))
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], diagnostic, 48))
        records.sort(key=merit, reverse=True)
        return records

    # F1024 = F2[t]/Phi_11(t); the quotient action is F4-linear.
    def fmul(a, b):
        value = 0
        while b:
            if b & 1:
                value ^= a
            b >>= 1
            a <<= 1
            if a & 1024:
                a ^= 2047
        return value

    def frobenius(a, steps):
        for _ in range(steps % 10):
            a = fmul(a, a)
        return a

    roots = [1]
    for _ in range(10):
        roots.append(fmul(roots[-1], 2))
    omega = next(a for a in range(2, 1024)
                 if fmul(a, a) ^ a ^ 1 == 0)
    scalars = (1, omega, fmul(omega, omega))
    f4 = (0,) + scalars
    adjoints = [
        [frobenius(fmul(1 << t, roots[g[0]]), -2 * g[1])
         for t in range(10)]
        for g in G.elements[:55]
    ]

    def apply(images, value):
        answer = 0
        for t in bits(value):
            answer ^= images[t]
        return answer

    def free_line_key(v, w):
        # Multiplication by an F4 scalar and by 1+epsilon*c changes
        # the representative, not its annihilator constraint.
        return min(
            (vv, min(ww ^ fmul(c, vv) for c in f4))
            for vv, ww in (
                (fmul(c, v), fmul(c, w)) for c in scalars))

    lifted_catalog, quotient_catalog = {}, {}
    for a in range(1, 55):
        for b in range(1, a):
            support = (ident, b, a)
            if support != canonical_cell(support):
                continue
            images = [adjoints[ident][t] ^ adjoints[b][t]
                      ^ adjoints[a][t] for t in range(10)]
            pivots, kernel = {}, []
            for t, value in enumerate(images):
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
                    kernel.append(word)
            if not kernel:
                continue
            values = [0]
            for word in kernel:
                values += [old ^ word for old in values]

            def solve(value):
                word = 0
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        return None
                    value ^= old[0]
                    word ^= old[1]
                return word

            for v in values[1:]:
                qkey = min(fmul(c, v) for c in scalars)
                quotient_catalog.setdefault(qkey, set()).add(support)
                # All three relative central patterns are genuinely mixed.
                for zb, za in ((1, 0), (0, 1), (1, 1)):
                    rhs = ((apply(adjoints[b], v) if zb else 0)
                           ^ (apply(adjoints[a], v) if za else 0))
                    particular = solve(rhs)
                    if particular is None:
                        continue
                    ids = canonical_cell(
                        (ident, b + 55 * zb, a + 55 * za))
                    for nullword in values:
                        w = particular ^ nullword
                        key = free_line_key(v, w)
                        lifted_catalog.setdefault(key, set()).add(ids)

    # P=P0+epsilon*P1 annihilates v+epsilon*w precisely when
    # P0*(v)=0 and P0*(w)=P1*(v). This proposes cells only;
    # actual binary ranks and kernels below remain authoritative.
    pair_cache, atom_free = {}, {}

    def compatible(first, second):
        key = tuple(sorted((first, second)))
        if key not in pair_cache:
            columns = atom("A", first)[1] + atom("A", second)[1]
            pair_cache[key] = _pairs_no_kernel4(columns) is not None
        return pair_cache[key]

    def free_pool():
        entries = [(key, sorted(options))
                   for key, options in lifted_catalog.items()
                   if len(options) >= 3]
        entries.sort(key=lambda item: (-len(item[1]), item[0]))
        if not entries:
            return []
        offset = (17 * card + 29 * cohort) % len(entries)
        entries = entries[offset:] + entries[:offset]
        records, seen = [], set()
        tested = 0
        for functional, options in entries:
            eligible = []
            for ids in options:
                if ids not in atom_free:
                    atom_free[ids] = (
                        _pairs_no_kernel4(atom("A", ids)[1]) is not None)
                if atom_free[ids]:
                    eligible.append(ids)
            if len(eligible) < 3:
                continue
            rng.shuffle(eligible)
            eligible = eligible[:24]
            triples = list(combinations(eligible, 3))
            rng.shuffle(triples)
            for triple in triples[:24]:
                signature = tuple(sorted(triple))
                if signature in seen:
                    continue
                seen.add(signature)
                tested += 1
                if not all(compatible(triple[i], triple[j])
                           for i in range(3) for j in range(i)):
                    continue
                record = inspect("A", triple)
                if record is None or record["rank"] not in (80, 90):
                    continue
                record["functional"] = functional
                record["kind"] = "free_dual_line"
                records.append(record)
            if tested >= 640 or len(records) >= 16:
                break
        return challenge(records, "A")

    def quotient_pool():
        # Fresh central deformations are fallback proposals, not archived
        # matrices. Their B backbone also differs structurally from parent.
        entries = [(v, sorted(options))
                   for v, options in quotient_catalog.items()
                   if len(options) >= 3]
        entries.sort(key=lambda item: (-len(item[1]), item[0]))
        records, seen = [], set()
        if not entries:
            return []
        for trial in range(192):
            v, options = entries[
                (17 * card + 7 * cohort + trial // 8) % len(entries)]
            selected = rng.sample(options, 3)
            marked = set(rng.sample(range(3), 1 + trade % 3))
            cells = []
            for j, support in enumerate(selected):
                ids = list(support)
                if j in marked:
                    # Relative placement, not a common central gauge.
                    ids[rng.randrange(3)] += 55
                cells.append(canonical_cell(ids))
            signature = tuple(sorted(cells))
            if signature in seen:
                continue
            seen.add(signature)
            record = inspect("A", cells)
            if record is None:
                continue
            record["functional"] = v
            record["kind"] = "fresh_quotient_deformation"
            records.append(record)
            if len(records) >= 12:
                break
        return challenge(records, "A")

    # Five relative-spectrum cards; the repeated quotient-fiber cards
    # distinguish order-eleven from order-twenty-two relative ratios.
    tri_cards = (
        ((0, 1), (0, 1)),
        ((0, 2), (0, 1)),
        ((1, 2), (1, 0)),
        ((1, 3), (0, 1)),
        ((0, 1), (1, 0)),
    )
    bin_cards = ((1, 1), (1, 2), (1, 2), (1, 1), (1, 2))

    def draw_b(trial):
        coordinates = (bin_cards[model] if trial < 128 else (1, 2))
        # Never replace this translation with an all-central-odd fallback.
        ratios = [(rng.randrange(1, 11), 0, 0)]
        ratios += [(rng.randrange(11), b, 1) for b in coordinates]
        cells = [canonical_cell((ident, G.idx(g))) for g in ratios]
        (b1, b2), (z1, z2) = tri_cards[model]
        for _ in range(128):
            g = (rng.randrange(1, 11) if b1 == 0
                 else rng.randrange(11), b1, z1)
            h = (rng.randrange(11), b2, z2)
            gi, hi = G.idx(g), G.idx(h)
            if len({ident, gi, hi}) != 3:
                continue
            if table[gi][hi] == table[hi][gi]:
                continue
            cells.append(canonical_cell((ident, gi, hi)))
            return cells
        raise ValueError("translation-backbone trinomial unavailable")

    def b_pool():
        records, seen = [], set()
        for trial in range(224):
            cells = draw_b(trial)
            signature = tuple(sorted(cells))
            if signature in seen:
                continue
            seen.add(signature)
            record = inspect("B", cells)
            if record is not None:
                records.append(record)
        return challenge(records, "B")

    bb = b_pool()
    if not bb:
        bb = b_pool()
    if not bb:
        raise ValueError("new translation backbone failed component gates")

    def pair_pool(aa):
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            min(ab[0]["floor"], ab[1]["floor"]),
            min(ab[0]["reading"], ab[1]["reading"]),
            -ab[0]["rank"],
            ab[0]["reading"] + ab[1]["reading"],
            tuple(-x-y for x, y in zip(
                cycle_key(ab[0]["cycles"]),
                cycle_key(ab[1]["cycles"])))),
            reverse=True)
        for a, b in proposals:
            A = [[[G.elements[g] for g in cell] for cell in a["ids"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["ids"]]]
            hx, hz, n = lifted_matrices(probe, [list(a["ids"])],
                                       [list(b["ids"])])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1430 and 19 * n <= 40 * k < 21 * n):
                continue
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert actual_a == a["rows"] and actual_b == b["rows"]
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert css_ok(hx, hz)
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (trade, model, cohort),
                "annihilator_kind": a["kind"],
                "functional": a["functional"],
                "component_ranks": (a["rank"], b["rank"]),
                "component_floors": (a["floor"], b["floor"]),
                "component_readings": (a["reading"], b["reading"]),
                "component_geometry": (a["cycles"], b["cycles"]),
                "backbone": "one C11 translation, two central-odd ratios",
            }
        return None

    answer = pair_pool(free_pool())
    if answer is not None:
        return answer
    for _ in range(3):
        answer = pair_pool(quotient_pool())
        if answer is not None:
            return answer
    raise ValueError("compatible central lifts missed verified full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected dual-annihilator/translation-backbone cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane bank produces accepted plane rank gains beyond k=690; all 51 distinct current codes instead had k=690, and all five measured distance readings remained six.
# REFUTED: Quotient-annihilator constraints with unconstrained central mixing improve the five-code score while retaining six at larger rank deficit; reported k=720 candidates screened at four, and all five measured k=690 candidates remained at six with unchanged fitness 17.371.
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
