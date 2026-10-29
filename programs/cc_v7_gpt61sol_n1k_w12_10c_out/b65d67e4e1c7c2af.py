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
CONCEPT = ("On (C11 semidirect_-1 C4) x C2, order 88, A=(1,3), B=(2,5), replacing three degree-two B columns with one and repairing genuine component witnesses should produce five new distance readings >=12, exceeding the 0.350 frontier fitness 47.802. Failure of this five-code target refutes the measurement hypothesis; readings remain upper bounds.")
GROUP = ("Keep the metacyclic-times-central-C2 construction rule, resizing to m=11,r=4,s=10 and order 88 because the new size factor seventeen excludes q=110. The action is valid since (-1)^4=1 modulo eleven. Relative orders four, eleven and twenty-two replace the parent's order-five complement prescriptions. This bed is untried; related order-84 (1,3)x(2,5) measurements reached twelve, but do not establish distance transfer. Every authoritative block and rank comes from the scaffold.")
SHAPES = ("Change to A=(1,3), B=(2,5): q=88 gives n=1496, k>=528 and guaranteed rate 6/17=0.352941. The shape allows q<=88, so this uses its largest lift order. Components have 264 variables/88 checks and 440 variables/176 checks. Redirect to cell 0.350: parent minima were exclusively sector 1, particularly Acol1/2 across Bcol0..2. A remains cubic, while two B check rows permit four cubic columns and just one degree-two column. Exact final ranks enforce [0.325,0.375).")
FILLING = ("A has counts (3,3,3). B column degrees are (2,3,3,3,3), with row totals seven, giving HX<=9+3=12 and HZ<=7+3=10. Twenty cards cross five B zero budgets with four relative-translation patterns across A cells. Full-generating cubic proposals avoid involutory relative ratios; individual-term and whole-cell moves repair detected short words. Use only scaffold-derived monomial blocks, exact kernel-through-four detection, cycle heuristics, bounded parity completion and independently retained kernel witnesses. Final original-group pairings verify ranks, assembly, weights and CSS. No component search or exhausted budget certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Dic11xC2_13x25_cubic_relativegap_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 88

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C11:C4 times C2 requires order 88")
    # Same construction rule: split metacyclic group times central C2.
    # x^11=y^4=z^2=1, y*x*y^-1=x^-1, and z is central.
    m, r, s = 11, 4, 10
    powers = [pow(s, b, m) for b in range(r)]
    assert pow(s, r, m) == 1
    els = [(a, b, z) for z in range(2)
           for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r, int(g[2]) % 2)

    def mul(g, h):
        a, b, z = red(g)
        c, d, w = red(h)
        return ((a + powers[b] * c) % m,
                (b + d) % r, (z + w) % 2)

    def inv(g):
        a, b, z = red(g)
        bi = (-b) % r
        return ((-powers[bi] * a) % m, bi, z)

    def idx(g):
        a, b, z = red(g)
        return m * r * z + m * b + a

    return dict(elements=els, ident=(0, 0, 0),
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


def _retired_quotient_line_ctx(rng, G, idx):
    """Unused historical order-110 implementation."""
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
    """Twenty new support/relative-gap cards with binding witness repair."""
    from functools import lru_cache
    from math import exp

    q = G.q
    assert q == 88 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    zeros, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    identity = G.idx(G.e)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

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

    probe = Probe()
    mono = [[[g] for g in range(q)]]
    left, _, _ = lifted_matrices(probe, mono, [[[]]])
    _, right, _ = lifted_matrices(probe, [[[]]], mono)
    blocks = {
        "A": [[(row >> (g * q)) & mask for row in left]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in right]
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
            raise ValueError("invalid native finite group")

    def generates(a, b):
        reached, pending = {identity}, [identity]
        for x in pending:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    pending.append(z)
        return len(reached) == q

    long_ids = [
        [g for g in range(q) if orders[g] in allowed]
        for allowed in ((4, 22), (11, 22), (4, 11, 22), (22,))
    ]
    gap_patterns = ((1, 1, 1), (1, 1, 2),
                    (1, 2, 3), (1, 2, 5))

    def cubic(gap=None):
        """Relative gap changes one cell, not a global automorphism."""
        for _ in range(96):
            delta = rng.randrange(1, 11) if gap is None else gap
            a, b, z = rng.randrange(11), rng.choice((1, 3)), rng.randrange(2)
            g = G.idx((a, b, z))
            h = G.idx(((a + delta) % 11, b, z ^ 1))
            if not generates(g, h):
                continue
            ids = [identity, g, h]
            assert len(set(ids)) == 3
            assert sorted((orders[g], orders[h],
                           orders[table[inverse[g]][h]])) == [4, 4, 22]
            return ids
        raise ValueError("full-generating cubic proposal unavailable")

    def fresh_cell(count, side, j=0):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            ratio = rng.choice(long_ids[(policy + int(side == "B")) % 4])
            return [root, table[root][ratio]]
        assert count == 3
        gap = gap_patterns[policy][j] if side == "A" else None
        ids = cubic(gap)
        if side == "B":
            root = rng.randrange(q)
            ids = [table[root][g] for g in ids]
        return ids

    def support_b():
        degrees = (2, 3, 3, 3, 3)

        @lru_cache(None)
        def feasible(j, upper, remaining_zeros):
            if upper < 0 or remaining_zeros < 0:
                return False
            if j == len(degrees):
                return upper == remaining_zeros == 0
            degree = degrees[j]
            return any(
                feasible(j + 1, upper - a,
                         remaining_zeros - int(a == 0 or a == degree))
                for a in range(degree + 1)
            )

        assert feasible(0, 7, zeros)
        upper, remaining_zeros = 7, zeros
        grid = [[], []]
        for j, degree in enumerate(degrees):
            choices = [
                a for a in range(degree + 1)
                if feasible(j + 1, upper - a,
                            remaining_zeros - int(a == 0 or a == degree))
            ]
            a = rng.choice(choices)
            grid[0].append(a)
            grid[1].append(degree - a)
            upper -= a
            remaining_zeros -= int(a == 0 or a == degree)
        assert [sum(row) for row in grid] == [7, 7]
        assert _grid_weights([[3, 3, 3]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[3, 3, 3]], grid)
        return grid

    def fresh(side):
        support = [[3, 3, 3]] if side == "A" else support_b()
        return [[fresh_cell(count, side, j)
                 for j, count in enumerate(row)] for row in support]

    def record(grid, side):
        """GF2 assembly exclusively from scaffold-produced monomials."""
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nr * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i * q + t] ^= value << (j * q)
        columns = _syndrome_columns(rows, nc * q)

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
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0) + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a + 1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b + 1))).bit_count()
        excess = max(0, triangles - forced)
        peak = max((x.bit_count() for x in adjacency), default=0)
        cost = (600 * (1 + policy) * four + excess
                + (0, 2 * peak, excess, 6 * peak)[policy])
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cycles=(four, excess, peak), cost=cost,
                    signature=signature(grid))

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def short_word(columns):
        """A real witness, or exact exclusion of kernel weights <=4."""
        singles, pairs = {}, {}
        for c, value in enumerate(columns):
            word = 1 << c
            if not value:
                return word
            old = singles.get(value)
            if old is not None:
                return word ^ old
            singles[value] = word
        for a, value in enumerate(columns):
            for b in range(a):
                combined = value ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(combined)
                if old is None:
                    old = pairs.get(combined)
                if old is not None:
                    return word ^ old
                pairs[combined] = word
        return None

    def challenge(rec, trials, diagnostic_seed):
        """Binding actual kernel witnesses, never inferred lower bounds."""
        diagnostic = random.Random(diagnostic_seed)
        columns = rec["columns"]
        width = len(columns)
        order = list(range(width))
        retained = list(rec.get("words", ()))
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
            for initial in light[:5]:
                word = initial
                for _ in range(2):
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
                              key=int.bit_count)[:28]
            if witness is not None and witness.bit_count() <= 4:
                break

        assert witness is not None
        cap = min(12, witness.bit_count() - 1)
        if cap >= 5:
            singles, pairs = {}, {}
            incidence = [0] * len(rec["rows"])
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
            budget = min(32000, max(4000, 700 * trials))

            for root in roots:
                remaining = [max(1, budget // len(roots))]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    if not value:
                        observe(word)
                        return
                    left = min(cap, witness.bit_count() - 1) - word.bit_count()
                    if left <= 0 or value.bit_count() > 3 * left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        observe(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        observe(word | pair)
                    left = min(cap, witness.bit_count() - 1) - word.bit_count()
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
                    choices.sort(key=lambda c: (
                        -(columns[c] & value).bit_count(), priorities[c]))
                    live = allowed
                    for c in choices:
                        bit = 1 << c
                        live &= ~bit
                        visit(value ^ columns[c], word | bit, live)
                        if remaining[0] <= 0:
                            break

                # Component symmetry is transitive inside each column block.
                # This is budgeted searching, not an exclusion certificate.
                visit(columns[root], 1 << root,
                      all_variables & ~((1 << (root + 1)) - 1))
                if witness.bit_count() <= 4:
                    break

        assert witness and syndrome(columns, witness) == 0
        rec["words"] = sorted(set(retained + [witness]),
                              key=int.bit_count)[:28]
        rec["word"] = witness
        rec["d"] = min(rec.get("d", width + 1), witness.bit_count())
        return witness

    def term_move(old, side):
        if len(old) < 2:
            return None
        proposed = list(old)
        position = rng.choice((1, 2)) if side == "A" else rng.randrange(len(old))
        available = [g for g in range(q) if g not in old]
        proposed[position] = rng.choice(available)
        if len(proposed) == 3:
            root_inverse = inverse[proposed[0]]
            a = table[root_inverse][proposed[1]]
            b = table[root_inverse][proposed[2]]
            if not generates(a, b):
                return None
            if min(orders[a], orders[b],
                   orders[table[inverse[a]][b]]) <= 2:
                return None
        return proposed

    def component_pool(side):
        known, screened, seen = [], [], set()
        offset = 100003 * int(side == "B")

        def loss(rec):
            penalty = sum(
                4000 * max(1, 13 - word.bit_count()) ** 2
                for word in known
                if syndrome(rec["columns"], word) == 0)
            return rec["cost"] + penalty

        def optimize(grid, steps):
            current = record(grid, side)
            best = current
            current_cost = best_cost = loss(current)
            mutable = [(i, j) for i, row in enumerate(grid)
                       for j, cell in enumerate(row) if cell]
            for step in range(steps):
                positions = mutable
                active = [word for word in known
                          if syndrome(current["columns"], word) == 0]
                if active and step % 3 != 2:
                    occupied = {c // q for c in bits(rng.choice(active))}
                    targeted = [p for p in mutable if p[1] in occupied]
                    if targeted:
                        positions = targeted
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                proposed = term_move(old, side) if step % 3 else None
                grid[i][j] = (proposed if proposed is not None
                              else fresh_cell(len(old), side, j))
                trial = record(grid, side)
                trial_cost = loss(trial)
                temperature = 70 * (1 - step / steps) ** 2 + 0.2
                if trial_cost <= current_cost or rng.random() < exp(
                        min(0.0, (current_cost - trial_cost) / temperature)):
                    current, current_cost = trial, trial_cost
                    if trial_cost < best_cost:
                        best, best_cost = trial, trial_cost
                else:
                    grid[i][j] = old
            return best

        pool = [optimize(fresh(side), 48) for _ in range(4)]
        for stage in range(3):
            tested = []
            for rec in sorted(pool, key=loss):
                if rec["signature"] in seen:
                    continue
                seen.add(rec["signature"])
                word = short_word(rec["columns"])
                rec["free4"] = word is None
                if word is not None:
                    rec["words"] = [word]
                rec["rank"] = gf2_rank(rec["rows"])
                word = challenge(rec, 8, seed + offset + 1009 * stage)
                tested.append(rec)
                screened.append(rec)
                if word.bit_count() <= 12 and word not in known:
                    known.append(word)
                    del known[32:]
            tested.sort(key=lambda r: (
                r["free4"], r["d"], -loss(r), r["rank"]), reverse=True)
            if stage < 2:
                starts = tested[:2] if tested else pool[:2]
                pool = [
                    optimize(copy_grid(rec["grid"]), 80) for rec in starts
                ] + [optimize(fresh(side), 48) for _ in range(2)]

        if not screened:
            raise ValueError("empty new cubic component portfolio")
        screened.sort(key=lambda r: (
            r["free4"], r["d"], -loss(r), r["rank"]), reverse=True)
        finalists = screened[:3]
        for rec in finalists:
            challenge(rec, 32, seed + offset + 700001)
        finalists.sort(key=lambda r: (
            r["free4"], r["d"], -loss(r), r["rank"]), reverse=True)
        for rec in finalists[:2]:
            challenge(rec, 64, seed + offset + 1700021)
        finalists.sort(key=lambda r: (
            r["free4"], r["d"], -loss(r), r["rank"]), reverse=True)
        return finalists

    aa, bb = component_pool("A"), component_pool("B")
    best = None
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(probe, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1496 and k >= 528
                    and 325 * n <= 1000 * k < 375 * n):
                continue
            detected = min(12, a["d"], b["d"])
            key = (a["free4"] and b["free4"],
                   k * detected * detected, detected,
                   min(12, a["d"]) + min(12, b["d"]),
                   -a["cost"] - b["cost"])
            if best is None or key > best[0]:
                best = key, A, B, hx, hz, k, a, b
    if best is None:
        raise ValueError("new pairings missed the verified 0.350 rate gate")

    key, A, B, hx, hz, k, a, b = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert sa == [[3, 3, 3]]
    assert [sum(row) for row in sb] == [7, 7]
    assert [sum(row[j] for row in sb) for j in range(5)] == [2, 3, 3, 3, 3]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    actual_a = [
        sum(((hx[t] >> (j * 5 * q)) & mask) << (j * q)
            for j in range(3)) for t in range(q)]
    actual_b = [row & ((1 << (5 * q)) - 1) for row in hz[:2 * q]]
    assert actual_a == a["rows"] and actual_b == b["rows"]
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert css_ok(hx, hz)

    # Native encodings are returned; probes only accelerate screening.
    return {
        "A": [[[G.elements[g] for g in cell] for cell in row] for row in A],
        "B": [[[G.elements[g] for g in cell] for cell in row] for row in B],
        "idx": idx, "k": k, "card": (zeros, policy, cohort),
        "component_readings": (a["d"], b["d"]),
        "component_ranks": (a["rank"], b["rank"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "component_kernel4_free": (a["free4"], b["free4"]),
        "initial_relative_gap_policy": gap_patterns[policy],
        "pairing_merit": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return every selected cell of the new rectangular protographs."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane bank produces accepted plane rank gains beyond k=690; all 51 distinct current codes instead had k=690, and all five measured distance readings remained six.
# REFUTED: Quotient-annihilator trinomials with central mixing improve the order-110 direct-product high-rate score; all five measured current codes retained k=690, distance six and fitness 17.371.
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
