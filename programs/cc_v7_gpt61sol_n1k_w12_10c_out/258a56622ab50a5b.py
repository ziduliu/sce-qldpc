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
CONCEPT = ("Hypothesis: on faithful C19 semidirect_8 C6, order 114, A=(3,1), B=(4,1), canonical cubic joins sharing F8-linear annihilators produce five new cell-0.500 codes and fitness above 17.371. An actual common regular deficit of eighteen is preferred over the parent's ten; measured distances remain provisional upper bounds.")
GROUP = ("Jump from (C11:C5)xC2 to faithful C19:C6. Since 8^3=-1 and 8^6=1 mod 19, the new order-114 group has trivial centre, unlike the parent's central involution. Related order-114, trivial-centre, nine-class beds recorded 303.9 on (3,2)x(3,2) and 111.3 on (1,2)x(2,5). Those readings motivate this untried tall high-rate bed, not distance transfer. Since ord_19(2)=18 and 8=2^3, F262144 provides an F8-linear proposal representation; only native lifted matrices determine accepted ranks.")
SHAPES = ("Keep A=(3,1), B=(4,1). At q=114, n=1482 and k>=684 give guaranteed rate 6/13; the shape permits q<=115. Raw components have 114 variables and 342/456 checks, while their distance-bearing transposes have 342/456 variables and 114 checks. Parent minimum witnesses were entirely sector two, touching all Arows and especially Brow0..2. Authoritative full ranks enforce [0.475,0.525), and rank gain competes only after short-component screening.")
FILLING = ("Keep A counts (3,3,3), B counts (2,2,2,3): HX=3+9=12 and HZ<=3+9=12. Enumerate right-gauge-canonical cubic operators and complete representation kernels through dimension six. Join distinct operators sharing an F8-linear annihilator, then screen actual transpose ranks, kernels through five and four-cycle geometry. Twenty cards cross four regular-deficit preferences with five B quotient prescriptions; backbones use order-six ratios and optionally one order-nineteen translation. Retain genuine kernel witnesses across independently seeded challenges, deepen finalists, and cap selection readings at six before rewarding actual k. Final native component agreement, full kernel-four exclusion, weights, rate and CSS remain gates. No component or finite-search screen certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r050_C19C6s8_31x41_F8_join_retained_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 114

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("faithful C19:C6 requires order 114")
    # (a,b)=x^a*y^b, y*x*y^-1=x^8.
    # 8^3=-1 mod 19, so this action has order exactly six.
    powers = [pow(8, b, 19) for b in range(6)]
    els = [(a, b) for b in range(6) for a in range(19)]

    def red(g):
        return (int(g[0]) % 19, int(g[1]) % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 19, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi] * a) % 19, bi)

    def idx(g):
        a, b = red(g)
        return 19 * b + a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 1)               # (ra, ca)
B_SHAPE = (4, 1)               # (rb, cb)


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


def _retired_quotient_line_ctx(rng, G, idx):
    """Unused historical wide-shape implementation."""
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


def _retired_tall_centralmix_ctx(rng, G, idx):
    """Inactive order-110 central-mixing implementation."""
    q = G.q
    assert q == 110
    assert A_SHAPE == (3, 1) and B_SHAPE == (4, 1)
    card, cohort = idx % 20, idx // 20
    policy, model = card // 5, card % 5
    mask = (1 << q) - 1
    sa, sb = [[3], [3], [3]], [[2], [2], [2], [3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

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
    unit = [[[identity]]]
    all_cells = [[[g]] for g in range(q)]
    mono = {}
    # Bulk probes supply the actual transposed left/right actions.
    # Linear assembly below only joins/XORs these verified blocks.
    for side in ("A", "B"):
        if side == "A":
            _, source, _ = lifted_matrices(P, all_cells, unit)
        else:
            source, _, _ = lifted_matrices(P, unit, all_cells)
        assert len(source) == q
        mono[side] = [
            [(row >> ((1 + g) * q)) & mask for row in source]
            for g in range(q)
        ]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

    def component(grid, side):
        rows = []
        for r in range(q):
            row = 0
            for j, raw in enumerate(grid):
                block = 0
                for g in raw[0]:
                    block ^= mono[side][g][r]
                row |= block << (j * q)
            rows.append(row)
        return rows

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
    adjoints = [
        [frobenius(fmul(1 << t, roots[g[0]]), -2 * g[1])
         for t in range(10)]
        for g in G.elements[:55]
    ]

    def adjoint_kernel(support):
        pivots, basis = {}, []
        for t in range(10):
            value = 0
            for g in support:
                value ^= adjoints[g][t]
            word = 1 << t
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

    lines = {}
    nullities = {}
    for a in range(1, 55):
        for b in range(1, a):
            support = (identity, b, a)
            gauges = [
                tuple(sorted(table[g][inverse[root]] for g in support))
                for root in support
            ]
            if support != min(gauges):
                continue
            basis = adjoint_kernel(support)
            if not basis:
                continue
            nullities[support] = len(basis)
            values = [0]
            for v in basis:
                values += [u ^ v for u in values]
            for v in values[1:]:
                lines.setdefault(v, set()).add(support)

    unique_catalogs = {}
    for functional, supports in lines.items():
        supports = tuple(sorted(supports))
        if len(supports) >= 3:
            unique_catalogs.setdefault(supports, functional)
    catalogs = [
        (functional, supports)
        for supports, functional in unique_catalogs.items()
    ]
    catalogs.sort(key=lambda item: (
        -max(nullities[s] for s in item[1]), -len(item[1]), item[0]))
    if not catalogs:
        raise ValueError("no quotient annihilator proposal catalog")

    def tall_from_horizontal(cells):
        # Inversion makes the tall unit-probe component use the original
        # horizontal action. New relative supports, not gauges, supply
        # diversity; inversion itself is not counted as a new design.
        return [[[inverse[g] for g in cell]] for cell in cells]

    def draw_a(attempt):
        if attempt % 3 == 0:
            position = (card + 7 * (attempt // 3) + 11 * cohort) % min(
                32, len(catalogs))
        else:
            position = (37 * card + 101 * attempt + 53 * cohort) % len(
                catalogs)
        functional, options = catalogs[position]
        selected = rng.sample(options, 3)
        budget = 2 + ((policy + attempt + cohort) & 1)
        # Occasionally challenge a higher-deficit pure-central proposal;
        # it still has to survive the same actual short-kernel screens.
        if attempt % 29 == 0:
            budget = 0
        mixed = set(rng.sample(range(3), budget))
        cells = []
        for j, support in enumerate(selected):
            marked = rng.randrange(3) if j in mixed else -1
            cell = []
            for t, g in enumerate(support):
                a, b, _ = G.elements[g]
                cell.append(G.idx((a, b, int(t == marked))))
            cells.append(cell)
        return tall_from_horizontal(cells), functional

    # These are actual relative-coordinate prescriptions. In particular,
    # the repeated fibers are absent from the parent's B recipe bank.
    bin_profiles = (
        ((1, 1), (2, 1), (4, 1)),
        ((0, 1), (1, 1), (3, 1)),
        ((1, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 1), (3, 1)),
    )
    tri_profiles = ((0, 0), (1, 1), (1, 4), (0, 3), (2, 3))
    central_profiles = ((0, 0, 1), (0, 1, 1),
                        (0, 1, 0), (0, 0, 0), (0, 1, 1))

    def draw_b(attempt):
        cells = []
        profile = bin_profiles[policy]
        for b, z in profile:
            a = rng.randrange(1, 11) if b == 0 else rng.randrange(11)
            ratio = G.idx((a, b, z))
            cells.append([identity, ratio])
        b1, b2 = tri_profiles[model]
        zs = central_profiles[(model + cohort) % 5]
        for _ in range(128):
            a = rng.randrange(1, 11) if b1 == 0 else rng.randrange(11)
            c = rng.randrange(1, 11) if b2 == 0 else rng.randrange(11)
            tri = [G.idx((0, 0, zs[0])),
                   G.idx((a, b1, zs[1])),
                   G.idx((c, b2, zs[2]))]
            if len(set(tri)) != 3:
                continue
            # Repeated normal-subgroup fibers deliberately allow
            # commuting trinomials; other fibers prefer noncommutativity.
            if model != 0:
                u, v = tri[1], tri[2]
                if table[u][v] == table[v][u]:
                    continue
            cells.append(tri)
            return tall_from_horizontal(cells), None
        raise ValueError("new repeated-fiber trinomial unavailable")

    def challenge(record, trials):
        """Retained genuine classical witnesses; upper readings only."""
        columns = record["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(record.get("words", ()))
        reading = min((w.bit_count() for w in retained), default=width + 1)

        def observe(word):
            nonlocal reading
            if word:
                reading = min(reading, word.bit_count())

        for _ in range(trials):
            rng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
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
                    observe(word)
            light = sorted(set(retained + basis), key=int.bit_count)[:16]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= reading + 2:
                        additions.append(candidate)
            moves = retained + basis
            rng.shuffle(moves)
            for word in light[:6]:
                for _ in range(2):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    if not changed:
                        break
                observe(word)
                additions.append(word)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:24]
            if reading <= 4:
                break
        for word in retained:
            value, bits = 0, word
            while bits:
                bit = bits & -bits
                value ^= columns[bit.bit_length() - 1]
                bits ^= bit
            assert value == 0
        record["words"] = retained
        record["reading"] = min(record.get("reading", width + 1), reading)

    def pool(side, turn):
        width = (3 if side == "A" else 4) * q
        bank, seen = [], set()
        trials = 288 if side == "A" else 224
        for attempt in range(trials):
            grid, functional = (
                draw_a(attempt + turn * trials) if side == "A"
                else draw_b(attempt + turn * trials))
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            rows = component(grid, side)
            rank = gf2_rank(rows)
            if not (80 <= rank <= 105 if side == "A"
                    else q - 20 <= rank <= q):
                continue
            bank.append(dict(
                grid=grid, rows=rows, rank=rank,
                cycles=_four_cycles(rows), signature=sig,
                functional=functional, attempt=attempt))
        bank.sort(key=lambda r: (
            r["cycles"], r["rank"], r["attempt"]))

        selected = []
        # Test low-cycle proposals and also lower-rank proposals:
        # a rank deficit is useful only if its short kernels survive.
        ordered = bank[:32] + sorted(
            bank, key=lambda r: (r["rank"], r["cycles"]))[:16]
        tested = set()
        for record in ordered:
            if record["signature"] in tested:
                continue
            tested.add(record["signature"])
            columns = _syndrome_columns(record["rows"], width)
            if side == "A":
                assert all(c.bit_count() == 3 for c in columns)
                pairs = _pairs_no_kernel4(columns)
                # Odd column degree excludes every odd-weight kernel.
                floor = 6 if pairs is not None else 4
            else:
                floor, columns = _component_floor(
                    record["rows"], width, q)
            record["floor"], record["columns"] = floor, columns
            record["reading"] = width + 1
            challenge(record, 10)
            selected.append(record)
            if len(selected) >= 10 and sum(
                    r["floor"] == 6 for r in selected) >= 5:
                break

        def merit(record):
            return (record["floor"], record["reading"],
                    -record["rank"], -record["cycles"])

        selected.sort(key=merit, reverse=True)
        finalists = selected[:4]
        for record in finalists:
            challenge(record, 48)
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

    best = None
    best_key = None
    for turn in range(3):
        aa, bb = pool("A", turn), pool("B", turn)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            min(ab[0]["floor"], ab[1]["floor"]),
            min(ab[0]["reading"], ab[1]["reading"]),
            -ab[0]["rank"] - ab[1]["rank"],
            -ab[0]["cycles"] - ab[1]["cycles"]), reverse=True)
        for a, b in proposals:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(P, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1430 and 19 * n <= 40 * k < 21 * n):
                continue

            # Confirm the cached distance-bearing components against
            # these exact final matrices. Sector two is (Arow, Brow).
            actual_a = [
                sum(((hz[r] >> ((1 + 4 * i) * q)) & mask) << (i * q)
                    for i in range(3))
                for r in range(q)
            ]
            actual_b = [
                (hx[r] >> q) & ((1 << (4 * q)) - 1)
                for r in range(q)
            ]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            assert max(row.bit_count() for row in hx + hz) == 12
            floor = min(a["floor"], b["floor"])
            detected = min(a["reading"], b["reading"])
            full4 = (
                _pairs_no_kernel4(_syndrome_columns(hx, n)) is not None
                and _pairs_no_kernel4(_syndrome_columns(hz, n)) is not None)
            key = (int(full4), floor, k * detected * detected,
                   detected, -a["cycles"] - b["cycles"])
            if best_key is not None and key <= best_key:
                continue
            assert css_ok(hx, hz)
            best_key = key
            best = {
                "A": [[[G.elements[g] for g in cell] for cell in row]
                      for row in A],
                "B": [[[G.elements[g] for g in cell] for cell in row]
                      for row in B],
                "idx": idx, "k": k, "card": (policy, model, cohort),
                "component_ranks": (a["rank"], b["rank"]),
                "component_floors": (a["floor"], b["floor"]),
                "component_readings": (a["reading"], b["reading"]),
                "functional": a["functional"],
                "full_kernel4_free": full4, "pairing_merit": key,
            }
        # Do not spend additional restarts merely to improve a lucky
        # reading. A valid screened fallback prevents needless void slots.
        if best is not None and best_key[0] and best_key[1] == 6:
            return best
    if best is not None:
        return best
    raise ValueError("new tall-shape pairings missed the verified rate gate")


def NEW_CTX(rng, G, idx):
    """Twenty faithful-action cards; all binary screens use native lifts."""
    q = G.q
    assert q == 114 and A_SHAPE == (3, 1) and B_SHAPE == (4, 1)
    card, cohort = idx % 20, idx // 20
    policy, model = divmod(card, 5)
    seed = rng.randrange(1 << 62) ^ (0x191146 + 1000003 * idx)
    rng = random.Random(seed)
    mask = (1 << q) - 1
    sa, sb = [[3], [3], [3]], [[2], [2], [2], [3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q, self.e = q, identity
            self.elements = list(range(q))

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverse[g]

    P = Probe()
    mono = {"A": [], "B": []}
    unit = [[[identity]]]
    # Narrow galleries avoid constructing a large unused dummy matrix.
    # Their extracted blocks are exactly the evaluator's transpose actions.
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g]] for g in ids]
        _, ax, _ = lifted_matrices(P, gallery, unit)
        bx, _, _ = lifted_matrices(P, unit, gallery)
        assert len(ax) == len(bx) == q
        for j in range(len(ids)):
            for side, source in (("A", ax), ("B", bx)):
                mono[side].append([
                    (row >> ((1 + j) * q)) & mask for row in source])

    def component(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, value in enumerate(mono[side][g]):
                    rows[r] ^= value << (j * q)
        return rows

    def tall(cells):
        return [[list(cell)] for cell in cells]

    def signature(cells):
        return tuple(sorted(tuple(sorted(cell)) for cell in cells))

    # Phi_19 is irreducible over F2 because ord_19(2)=18.
    # This field proposes supports only: no field rank is accepted as
    # the rank of a lifted component.
    def fmul(a, b):
        answer = 0
        while b:
            if b & 1:
                answer ^= a
            b >>= 1
            a <<= 1
            if a & (1 << 18):
                a ^= (1 << 19) - 1
        return answer

    def frobenius(a, steps):
        for _ in range(steps % 18):
            a = fmul(a, a)
        return a

    roots = [1]
    for _ in range(18):
        roots.append(fmul(roots[-1], 2))
    assert fmul(roots[-1], 2) == 1
    adjoints = [
        [frobenius(fmul(1 << t, roots[g[0]]), -3 * g[1])
         for t in range(18)] for g in G.elements]

    def proposed_kernel(support):
        pivots, basis = {}, []
        for t in range(18):
            value = 0
            for g in support:
                value ^= adjoints[g][t]
            word = 1 << t
            while value:
                lead = value.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = value, word
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    memberships, nullities, operators = {}, {}, []
    for a in range(q):
        if a == identity:
            continue
        for b in range(a):
            if b == identity:
                continue
            support = tuple(sorted((identity, b, a)))
            gauges = [
                tuple(sorted(table[g][inverse[root]] for g in support))
                for root in support]
            if support != min(gauges):
                continue
            basis = proposed_kernel(support)
            # Complete enumeration for the selected proposal dimensions.
            # Omitting larger kernels is not an exclusion claim.
            if len(basis) not in (3, 6):
                continue
            nullities[support] = len(basis)
            operators.append(support)
            values = [0]
            for word in basis:
                values += [v ^ word for v in values]
            for word in values[1:]:
                memberships.setdefault(word, set()).add(support)

    unique = {}
    for functional, supports in memberships.items():
        supports = tuple(sorted(supports))
        if len(supports) >= 3:
            unique.setdefault(supports, functional)
    catalogs = [(functional, supports)
                for supports, functional in unique.items()]
    catalogs.sort(key=lambda item: (-len(item[1]), item[0]))

    def syndrome(columns, word):
        value = 0
        while word:
            bit = word & -word
            value ^= columns[bit.bit_length() - 1]
            word ^= bit
        return value

    def challenge(record, trials, diagnostic_seed):
        """Retained genuine component words; readings can only decrease."""
        local = random.Random(diagnostic_seed)
        columns = record["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(record.get("words", ()))
        reading = record.get("reading", width + 1)

        def observe(word):
            nonlocal reading
            if word:
                reading = min(reading, word.bit_count())

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
        for turn in range(trials):
            local.shuffle(order)
            if retained and turn % 5 == 4:
                anchor = local.choice(retained[:6])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = value, word
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            light = sorted(set(retained + basis),
                           key=lambda w: (w.bit_count(), w))[:12]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= reading + 4:
                        additions.append(candidate)
                if len(additions) > 160:
                    additions = sorted(set(additions),
                                       key=int.bit_count)[:32]
            moves = retained + basis
            local.shuffle(moves)
            for initial in light[:4]:
                word = initial
                for _ in range(2):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                observe(word)
                additions.append(word)
            retained = sorted(
                {w for w in retained + light + additions if w},
                key=lambda w: (w.bit_count(), w))[:24]
            if reading <= 4:
                break
        assert retained
        assert all(syndrome(columns, word) == 0 for word in retained)
        record["words"], record["reading"] = retained, reading

    atom_cache = {}

    def atom_legal(support):
        if support not in atom_cache:
            cells = [[inverse[g] for g in support]]
            rows = component(cells, "A")
            columns = _syndrome_columns(rows, q)
            atom_cache[support] = (
                _pairs_no_kernel4(columns) is not None)
        return atom_cache[support]

    # These change within-cell relative quotient geometry. They are not
    # common quotient shifts or inversion images of one submitted code.
    tri_profiles = ((1, 2), (1, 3), (1, 4), (2, 4), (1, 1))
    preferred_defects = (18, 21, 24, 27)
    atom_profiles = ((3, 3, 3), (3, 3, 6),
                     (3, 6, 6), (6, 6, 6))

    def draw_a(attempt, turn):
        functional = None
        if catalogs:
            position = (37 * card + 101 * attempt + 53 * cohort
                        + 71 * turn) % len(catalogs)
            functional, options = catalogs[position]
        else:
            # A representation catalog is a preference, not an assumed
            # feasibility theorem. Accidental actual intersections compete.
            options = operators
        if len(options) < 3:
            return None
        selected = rng.sample(options, 3)
        if not all(atom_legal(support) for support in selected):
            return None
        cells = [[inverse[g] for g in support] for support in selected]
        profile = tuple(sorted(nullities[s] for s in selected))
        mismatch = sum(abs(a - b) for a, b in
                       zip(profile, atom_profiles[policy]))
        return cells, functional, mismatch

    def draw_b(attempt):
        # Primitive complement ratios have order six. At most one
        # translation ratio is used: two commuting binomials force C4.
        translated = (policy + cohort) & 1
        cells = []
        for j in range(3):
            if translated and j == 0:
                ratio = G.idx((rng.randrange(1, 19), 0))
            else:
                ratio = G.idx((rng.randrange(19), rng.choice((1, 5))))
            cells.append([identity, ratio])
        b1, b2 = tri_profiles[model]
        for _ in range(128):
            g = G.idx((rng.randrange(19), b1))
            h = G.idx((rng.randrange(19), b2))
            support = [identity, g, h]
            if len(set(support)) != 3:
                continue
            left = {table[inverse[u]][v] for u in support
                    for v in support if u != v}
            right = {table[u][inverse[v]] for u in support
                     for v in support if u != v}
            if len(left) == len(right) == 6:
                cells.append(support)
                return [[inverse[g] for g in cell] for cell in cells]
        return None

    def pool(side, turn):
        bank, seen = [], set()
        for attempt in range(384 if side == "A" else 224):
            if side == "A":
                drawn = draw_a(attempt, turn)
                if drawn is None:
                    continue
                cells, functional, mismatch = drawn
            else:
                cells = draw_b(attempt)
                if cells is None:
                    continue
                functional, mismatch = None, 0
            sig = signature(cells)
            if sig in seen:
                continue
            seen.add(sig)
            rows = component(cells, side)
            rank = gf2_rank(rows)
            if side == "A":
                if not q - 36 <= rank <= q - 6:
                    continue
            elif rank != q:
                continue
            bank.append(dict(
                cells=cells, rows=rows, rank=rank, functional=functional,
                mismatch=mismatch, signature=sig,
                cycles=_four_cycles(rows), attempt=attempt))

        bank.sort(key=lambda r: (
            r["cycles"],
            abs(q - r["rank"] - preferred_defects[policy])
            if side == "A" else 0,
            r["mismatch"], r["attempt"]))
        ordered = bank[:40]
        if side == "A":
            ordered += sorted(bank, key=lambda r: (
                abs(q - r["rank"] - preferred_defects[policy]),
                r["mismatch"], r["cycles"]))[:24]
        selected, tested = [], set()
        width = (3 if side == "A" else 4) * q
        for rec in ordered:
            if rec["signature"] in tested:
                continue
            tested.add(rec["signature"])
            columns = _syndrome_columns(rec["rows"], width)
            if side == "A":
                assert all(c.bit_count() == 3 for c in columns)
                # Odd column degree excludes all odd-weight null words.
                if _pairs_no_kernel4(columns) is None:
                    continue
                floor = 6
            else:
                floor, columns = _component_floor(rec["rows"], width, q)
                if floor != 6:
                    continue
            rec.update(columns=columns, floor=floor, reading=width + 1)
            offset = 100003 * int(side == "B")
            challenge(rec, 12, seed + offset + 1009 * len(selected))
            selected.append(rec)
            if len(selected) >= 8:
                break

        def merit(rec):
            return (min(6, rec["reading"]),
                    -abs(q - rec["rank"] - preferred_defects[policy])
                    if side == "A" else 0,
                    -rec["cycles"], -rec["mismatch"],
                    min(12, rec["reading"]))

        selected.sort(key=merit, reverse=True)
        finalists = selected[:4]
        for at, rec in enumerate(finalists):
            offset = 100003 * int(side == "B")
            challenge(rec, 48, seed + offset + 700001 + 1013 * at)
        finalists.sort(key=merit, reverse=True)
        for at, rec in enumerate(finalists[:3]):
            offset = 100003 * int(side == "B")
            challenge(rec, 64, seed + offset + 1700021 + 1019 * at)
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

    for turn in range(3):
        aa, bb = pool("A", turn), pool("B", turn)
        candidates = []
        for a in aa:
            for b in bb:
                A, B = tall(a["cells"]), tall(b["cells"])
                hx, hz, n = lifted_matrices(P, A, B)
                assert n == 1482
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (475 * n <= 1000 * k < 525 * n):
                    continue
                actual_a = [
                    sum(((hz[r] >> ((1 + 4 * i) * q)) & mask)
                        << (i * q) for i in range(3))
                    for r in range(q)]
                actual_b = [
                    (hx[r] >> q) & ((1 << (4 * q)) - 1)
                    for r in range(q)]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                detected = min(6, a["reading"], b["reading"])
                key = (k * detected * detected, detected,
                       -a["cycles"] - b["cycles"],
                       -a["mismatch"])
                candidates.append((key, A, B, hx, hz, k, a, b))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for key, A, B, hx, hz, k, a, b in candidates:
            if _pairs_no_kernel4(_syndrome_columns(hx, 1482)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, 1482)) is None:
                continue
            assert max(row.bit_count() for row in hx + hz) == 12
            assert css_ok(hx, hz)
            native_a = [[[G.elements[g] for g in cell] for cell in row]
                        for row in A]
            native_b = [[[G.elements[g] for g in cell] for cell in row]
                        for row in B]
            return {
                "A": native_a, "B": native_b, "idx": idx, "k": k,
                "card": (policy, model, cohort),
                "requested_regular_A_defect": preferred_defects[policy],
                "actual_regular_A_defect": q - a["rank"],
                "component_ranks": (a["rank"], b["rank"]),
                "component_readings": (a["reading"], b["reading"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "functional": a["functional"],
                "pairing_merit": key, "target_cell": 0.500,
            }
    raise ValueError("faithful-action joins missed full rank/kernel/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected native-group cell."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane bank produces accepted plane rank gains beyond k=690; all 51 distinct current codes instead had k=690, and all five measured distance readings remained six.
# REFUTED: Quotient-annihilator lines with deliberate central mixing improve the order-110 wide-shape five-code score; the current measured program again returned five k=690, distance-six codes and fitness 17.371.
# REFUTED: Transpose-component challenges and repeated-fiber B prescriptions improve the order-110 direct-product score; all five measured tall-shape codes still had k=690, distance six and fitness 17.371, with minimum witnesses entirely in sector two.
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
