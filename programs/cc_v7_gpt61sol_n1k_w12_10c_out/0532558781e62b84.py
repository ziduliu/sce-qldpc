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
CONCEPT = ("For (C7 semidirect_3 C6) x C2, order 84, A=(1,3), B=(2,5), generating cubic cells with binding component-witness repair should produce five new measured readings >=12 and improve cell 0.350 above fitness 47.8422. Five rank-bound readings twelve score 50.8235. This is a falsifiable measurement target, not a distance certificate.")
GROUP = ("Jump from the order-110 affine group to the split direct product (C7:C6) x C2. Three has exact order six modulo seven; the new group has exponent 42 and a central involution. The related nonsplit C7:C12 action-three bed measured fitness 47.8, motivating a split-extension comparison without assuming distance transfer. Native group operations propose labels; every screening matrix comes from lifted_matrices.")
SHAPES = ("Change to A=(1,3), B=(2,5): q=84 gives n=1428, k>=504 and guaranteed rate 6/17=0.352941, targeting the weaker 0.350 frontier rather than the parent's 0.500 cell. The shape permits q<=88. Components have 252/420 variables and 84/168 checks, nominal rates 2/3 and 3/5. Parent minimum witnesses occupied only sector-one Acol/Bcol blocks. Make every component column cubic and challenge localized logicals in those original strips; exact full ranks retain [0.325,0.375).")
FILLING = ("A counts are (3,3,3). B has five degree-three columns with row totals seven/eight, so HX=9+3=12 and HZ<=8+3=11. Twenty cards cross zero through four concentrated B columns with four genuinely different generating relative-element prescriptions. Use native bidirectional Sidon labels, authoritative component-cycle prescreening, exact kernel-through-four detection, central-C2-fixed challenges, retained kernel witnesses and witness-directed whole-cell replacement. Independently deepen finalists and verify localized logicals against ORIGINAL stabilizer spans. Select with a conservative reading cap fourteen. Final ranks, component agreement, CSS, cell distinctness, weights and rate gate submission; finite searches certify no quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_Frob42xC2_13x25_cubic_binding_splitextension_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius42 x C2 requires order 84")
    # (a,b,z)=x^a*y^b*c^z, y*x*y^-1=x^3.
    # The C6 action is faithful; c is a separate central involution.
    powers = (1, 3, 2, 6, 4, 5)
    els = [(a,b,z) for z in range(2)
           for b in range(6) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 6, int(g[2]) % 2)

    def mul(g, h):
        a,b,z = red(g)
        c,d,w = red(h)
        return ((a + powers[b]*c) % 7,
                (b+d) % 6, (z+w) % 2)

    def inv(g):
        a,b,z = red(g)
        bi = (-b) % 6
        return ((-powers[bi]*a) % 7, bi, z)

    def idx(g):
        a,b,z = red(g)
        return 42*z + 7*b + a

    return dict(elements=els, ident=(0,0,0),
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


def _retired_affine_trace_ctx(rng, G, idx):
    """Inactive order-110 predecessor; never called on the new group."""
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

    def constrained_cells():
        """Enumerate three-term operators with a common left functional.

        T_(a,b)(w)=z^a*w^(2^b) is a representation of the frozen group.
        Relative to Tr(v*w), its adjoint sends v to
        (v*z^a)^(2^(-b)). A zero sum of these adjoint images means
        that the corresponding operator has the chosen left annihilator.

        This representation only proposes labels. All component ranks,
        kernels and final quantum matrices are measured by the scaffold.
        """
        for _ in range(40):
            v = rng.randrange(1, 1024)
            images = [
                frobenius(fmul(v, roots[g[0]]), -g[1])
                for g in G.elements
            ]
            buckets = {}
            for t, image in enumerate(images):
                buckets.setdefault(image, []).append(t)
            supports = set()
            for a in range(q):
                for b in range(a):
                    for c in buckets.get(images[a] ^ images[b], ()):
                        if c >= b:
                            continue
                        original = [G.elements[c], G.elements[b],
                                    G.elements[a]]
                        # Right column gauges preserve the left annihilator.
                        anchor = G.inv(original[0])
                        normalized = [
                            G.mul(g, anchor) for g in original
                        ]
                        supports.add(tuple(sorted(
                            G.idx(g) for g in normalized)))
            if len(supports) >= 24:
                return [
                    [G.elements[t] for t in support]
                    for support in sorted(supports)
                ], v
        return None, None

    # Each new trinomial has a translation or an involution relative label.
    # These are not the parent's (0, odd, nonzero-even) prescriptions.
    spectra = ((0, 1), (0, 3), (0, 7), (1, 5), (2, 5))

    def fresh_tri(mode):
        b1, b2 = spectra[mode % 5]
        for _ in range(200):
            a = rng.randrange(1, 11) if b1 == 0 else rng.randrange(11)
            g = (a, b1)
            h = (rng.randrange(11), b2)
            if G._key(G.mul(g, h)) == G._key(G.mul(h, g)):
                continue
            return [G.e, g, h]
        raise ValueError("new noncommuting trinomial spectrum failed")

    def fresh(side, special, options):
        if side == "A":
            if special:
                # Different triples from one common-annihilator class,
                # not three column gauges of the same operator.
                return [list(cell) for cell in rng.sample(options, 3)]
            gs = _short_cycle_free_bins(rng, G, 3, card + idx // 20)
            return [[G.e, g] for g in gs]

        count = 3 if special else 2
        gs = _short_cycle_free_bins(
            rng, G, count, card + 2 + idx // 20)
        cells = [[G.e, g] for g in gs]
        cells.append(fresh_tri(card % 5))
        if not special:
            cells.append(fresh_tri(
                (card % 5 + 1 + card // 5) % 5))
        return cells

    def pool(side, special, options):
        """Independent generation, exact component gates and resampling."""
        width = (3 if side == "A" else 4) * q
        required_rank = (100 if special else 109) if side == "A" else 110
        records = []
        signatures = set()
        trials = 160 if special and side == "A" else 80
        for trial in range(trials):
            cells = fresh(side, special, options)
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in cells)
            if signature in signatures:
                continue
            signatures.add(signature)
            rows = component(cells, side)
            if gf2_rank(rows) != required_rank:
                continue
            cycles = _four_cycles(rows)
            records.append((cycles, trial, cells, rows))

        records.sort(key=lambda record: record[:2])
        survivors = []
        for cycles, trial, cells, rows in records:
            floor, columns = _component_floor(rows, width, q)
            if floor != 6:
                continue
            reading = _component_reading(columns, rng, 12)
            survivors.append({
                "cells": cells, "columns": columns,
                "cycles": cycles, "reading": reading,
                "trial": trial,
            })
            if len(survivors) >= 6:
                break
        survivors.sort(key=lambda record: (
            record["reading"], -record["cycles"]), reverse=True)
        for record in survivors[:3]:
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], rng, 48))
        survivors = survivors[:3]
        survivors.sort(key=lambda record: (
            record["reading"], -record["cycles"]), reverse=True)
        return survivors

    def build(special):
        options, functional = (None, None)
        if special:
            options, functional = constrained_cells()
            if options is None:
                return None
        aa = pool("A", special, options)
        bb = pool("B", special, options)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            min(ab[0]["reading"], ab[1]["reading"]),
            ab[0]["reading"] + ab[1]["reading"],
            -ab[0]["cycles"] - ab[1]["cycles"]), reverse=True)

        target_k = 690 if special else 663
        for a, b in proposals:
            A, B = [a["cells"]], [b["cells"]]
            sa = [[len(cell) for cell in A[0]]]
            sb = [[len(cell) for cell in B[0]]]
            assert _grid_weights(sa, sb) <= min(12, W_MAX)
            assert _no_dead_lines(sa, sb)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != target_k:
                continue
            # Full-code symmetry is not assumed. This test is exhaustive
            # over all full columns, but only through weight four.
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue
            assert n == 1430 and k >= 6 * q
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert css_ok(hx, hz)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": ("trace" if special else "companion", card),
                "functional": functional,
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
            }
        return None

    # Exactly three rank-gain slots, even when the harness builds 60.
    # Three k=690 and two k=663 promoted codes remain just below 0.475.
    if idx < 3:
        for _ in range(3):
            answer = build(True)
            if answer is not None:
                return answer

    # Do not void a slot merely because the new algebraic constraint failed.
    # These use genuinely new B spectra rather than archived parent recipes.
    for _ in range(4):
        answer = build(False)
        if answer is not None:
            return answer
    raise ValueError("no new kernel-five-screened companion with k=663")


def NEW_CTX(rng, G, idx):
    """Twenty split-extension cubic designs using authoritative matrices."""
    from functools import lru_cache
    from math import gcd

    q = G.q
    assert q == 84 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    concentration, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    ident = G.idx(G.e)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    # Native operation caching, not an independently implemented lift.
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

    probe = Probe()
    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q + 1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("native group order failed")
    assert max(orders) == 14
    assert all(42 % order == 0 for order in orders)

    # Both conventions come solely from the verified constructor.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 14):
        ids = list(range(start, min(start + 14, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe, gallery, [[[]]])
        _, zz, _ = lifted_matrices(probe, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j*q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j*q)) & mask for row in zz])

    central = G.idx((0, 0, 1))
    assert table[central][central] == ident
    assert all(table[central][g] == table[g][central] for g in range(q))
    remaining, central_pairs = set(range(q)), []
    while remaining:
        u = min(remaining)
        v = table[u][central]
        assert u != v and v in remaining
        remaining.remove(u)
        remaining.remove(v)
        central_pairs.append((u, v))

    def support_b():
        @lru_cache(None)
        def feasible(j, upper, pure):
            if upper < 0 or pure < 0:
                return False
            if j == 5:
                return upper == pure == 0
            return any(
                feasible(j+1, upper-a, pure-int(a in (0, 3)))
                for a in range(4))

        assert feasible(0, 7, concentration)
        upper, pure = 7, concentration
        grid = [[], []]
        for j in range(5):
            choices = [
                a for a in range(4)
                if feasible(j+1, upper-a, pure-int(a in (0, 3)))]
            a = rng.choice(choices)
            grid[0].append(a)
            grid[1].append(3-a)
            upper -= a
            pure -= int(a in (0, 3))
        assert [sum(row) for row in grid] == [7, 8]
        assert [grid[0][j] + grid[1][j] for j in range(5)] == [3]*5
        assert _grid_weights([[3, 3, 3]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[3, 3, 3]], grid)
        return grid

    def sidon(cell):
        wanted = len(cell)*(len(cell)-1)
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        return (len(set(cell)) == len(cell)
                and len(left) == len(right) == wanted)

    def generating(u, v):
        # Nonzero commutator supplies the prime normal C7.
        # Independent mod-two coordinates and a nonzero C3 coordinate
        # supply the entire abelianization C6 x C2.
        if table[u][v] == table[v][u]:
            return False
        _, b, z = G.elements[u]
        _, d, w = G.elements[v]
        return gcd(6, gcd(b, d)) == 1 and (b*w - d*z) % 2 == 1

    # Relative central/quotient associations differ between cards.
    # They are not common translations or automorphism-generated copies.
    tri_profiles = (
        ((1, 0), (0, 1)),
        ((1, 0), (2, 1)),
        ((1, 1), (2, 1)),
        ((1, 1), (0, 1)),
    )
    bin_orders = ((6,), (14,), (6, 14), (6, 7, 14))
    order_pools = {
        order: [g for g in range(q) if orders[g] == order]
        for order in (6, 7, 14)
    }

    def fresh_cell(count, mode):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            order = rng.choice(bin_orders[mode])
            root = rng.randrange(q)
            ratio = rng.choice(order_pools[order])
            cell = [root, table[root][ratio]]
            assert sidon(cell)
            return cell

        assert count == 3
        (b, z), (d, w) = tri_profiles[mode]
        for _ in range(256):
            u = G.idx((rng.randrange(7), b, z))
            v = G.idx((rng.randrange(7), d, w))
            cell = [ident, u, v]
            if not sidon(cell) or not generating(u, v):
                continue
            rels = [orders[table[inverse[a]][c]]
                    for at, a in enumerate(cell) for c in cell[:at]]
            if min(rels) >= 4:
                return cell
        # Fresh legal alternative, never a stored predecessor filling.
        for _ in range(4096):
            u, v = rng.sample(range(q), 2)
            cell = [ident, u, v]
            if sidon(cell) and generating(u, v):
                rels = [orders[table[inverse[a]][c]]
                        for at, a in enumerate(cell) for c in cell[:at]]
                if min(rels) >= 4:
                    return cell
        raise ValueError("generating split-extension cubic cell unavailable")

    def fresh(side):
        support = [[3, 3, 3]] if side == "A" else support_b()
        mode = (policy + int(side == "B") + cohort) % 4
        return [
            [fresh_cell(count, mode) for count in row]
            for row in support]

    def inspect(grid, side):
        height = len(grid)*q
        width = len(grid[0])*q
        rows = [0]*height
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i*q+t] ^= value << (j*q)
        columns = _syndrome_columns(rows, width)
        assert all(column.bit_count() == 3 for column in columns)
        if len(set(columns)) != width:
            return None

        uses, four = {}, 0
        adjacency = [0]*height
        for column in columns:
            ns = list(bits(column))
            for at, a in enumerate(ns):
                adjacency[a] |= column ^ (1 << a)
                for b in ns[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0) + 1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))).bit_count()
        excess = max(0, triangles - width)
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            cycles=(four, excess),
            cost=(600 + 150*policy)*four + excess,
            rank=gf2_rank(rows), words=set(), d=width+1)

    def small_word(columns):
        """Return a genuine witness, or exactly exclude weights <=4."""
        singles, pairs = {}, {}
        for c, value in enumerate(columns):
            word = 1 << c
            if not value:
                return word
            old = singles.get(value)
            if old is not None:
                return word ^ old
            singles[value] = word
        for c, value in enumerate(columns):
            for d in range(c):
                key = value ^ columns[d]
                word = (1 << c) | (1 << d)
                old = singles.get(key)
                if old is None:
                    old = pairs.get(key)
                if old is not None:
                    return word ^ old
                pairs[key] = word
        return None

    def kernel_search(columns, diagnostic, trials, seeds=()):
        """Genuine null vectors; search exhaustion is never exclusion."""
        width = len(columns)
        retained = sorted({w for w in seeds if w}, key=int.bit_count)[:20]
        best = None

        def observe(word):
            nonlocal best
            if word and (best is None or word.bit_count() < best.bit_count()):
                best = word

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
        order = list(range(width))
        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 4 == 3:
                anchor = diagnostic.choice(retained[:5])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
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

            light = sorted(set(basis + retained), key=int.bit_count)[:14]
            grown = list(light)
            for word in light:
                for other in basis:
                    proposed = word ^ other
                    observe(proposed)
                    if proposed and proposed.bit_count() <= best.bit_count()+4:
                        grown.append(proposed)
                if len(grown) > 160:
                    grown = sorted(set(grown), key=int.bit_count)[:28]

            moves = basis + retained
            diagnostic.shuffle(moves)
            for initial in light[:5]:
                word = initial
                for _ in range(2):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    observe(word)
                    if not changed:
                        break
                grown.append(word)
            if best is not None:
                grown.append(best)
            retained = sorted(
                {word for word in grown if word}, key=int.bit_count)[:24]
            if best is not None and best.bit_count() <= 4:
                break
        assert best is not None and syndrome(columns, best) == 0
        return best, retained

    def fixed_challenge(record, diagnostic, trials):
        columns, expansions = [], []
        original = record["columns"]
        for j in range(len(original)//q):
            for u, v in central_pairs:
                a, b = j*q+u, j*q+v
                columns.append(original[a] ^ original[b])
                expansions.append((1 << a) | (1 << b))
        initial = small_word(columns)
        word, _ = kernel_search(
            columns, diagnostic, trials,
            () if initial is None else (initial,))
        actual = 0
        for c in bits(word):
            actual ^= expansions[c]
        assert actual.bit_count() == 2*word.bit_count()
        assert syndrome(original, actual) == 0
        return actual

    def completion(record, diagnostic, budget):
        """Budgeted parity completion returns witnesses, not exclusions."""
        columns = record["columns"]
        width = len(columns)
        best = min(record["words"], key=int.bit_count)
        cap = min(12, best.bit_count()-1)
        if cap < 5 or budget == 0:
            return best
        singles, pairs = {}, {}
        incidence = [0]*len(record["rows"])
        for c, value in enumerate(columns):
            singles.setdefault(value, 1 << c)
            for r in bits(value):
                incidence[r] |= 1 << c
            for d in range(c):
                pairs.setdefault(value ^ columns[d], (1 << c) | (1 << d))
        priorities = [diagnostic.random() for _ in columns]
        roots = list(range(0, width, q))
        diagnostic.shuffle(roots)
        all_variables = (1 << width)-1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                assert syndrome(columns, word) == 0
                best = word

        for root in roots:
            remaining = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
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
                chosen, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                candidates = list(bits(chosen))
                candidates.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            visit(columns[root], 1 << root,
                  all_variables ^ (1 << root))
            if best.bit_count() <= 4:
                break
        return best

    def challenge(record, side, turn, trials, fixed_trials, budget):
        offset = 100003*int(side == "B")
        diagnostic = random.Random(seed + offset + 700001*turn)
        initial = small_word(record["columns"])
        record["free4"] = initial is None
        if initial is not None:
            record["words"].add(initial)
        fixed = fixed_challenge(record, diagnostic, fixed_trials)
        record["words"].add(fixed)
        word, retained = kernel_search(
            record["columns"], diagnostic, trials, record["words"])
        record["words"].update(retained)
        record["words"].add(word)
        record["words"].add(completion(record, diagnostic, budget))
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:24])
        assert all(syndrome(record["columns"], w) == 0
                   for w in record["words"])
        record["d"] = min(
            record["d"], min(w.bit_count() for w in record["words"]))

    def component_pool(side, restart):
        records, seen = [], set()

        def remember(grid):
            signature = tuple(
                tuple(sorted(cell)) for row in grid for cell in row)
            if signature in seen:
                return None
            seen.add(signature)
            record = inspect(grid, side)
            if record is not None:
                records.append(record)
            return record

        for _ in range(40):
            remember(fresh(side))
        if not records:
            raise ValueError("empty split-extension component pool")
        records.sort(key=lambda r: (r["cost"], -r["rank"]))
        selected = records[:6]
        for at, record in enumerate(selected):
            challenge(record, side, 1+at+100*restart, 8, 8, 4000)

        def merit(record):
            return (record["free4"], record["d"],
                    -record["cost"], record["rank"])

        current = max(selected, key=merit)
        mode = (policy + int(side == "B") + cohort) % 4
        for step in range(18):
            grid = copy_grid(current["grid"])
            witness = rng.choice(sorted(
                current["words"], key=int.bit_count)[:5])
            occupied = {c//q for c in bits(witness)}
            places = [
                (i, j) for i, row in enumerate(grid)
                for j, cell in enumerate(row)
                if cell and j in occupied]
            if not places:
                places = [
                    (i, j) for i, row in enumerate(grid)
                    for j, cell in enumerate(row) if cell]
            i, j = rng.choice(places)
            grid[i][j] = fresh_cell(len(grid[i][j]), mode)
            proposed = remember(grid)
            if proposed is None:
                continue
            proposed["words"].update(
                w for w in current["words"]
                if syndrome(proposed["columns"], w) == 0)
            challenge(proposed, side, 21+step+100*restart, 6, 6, 3000)
            selected.append(proposed)
            if merit(proposed) >= merit(current):
                current = proposed
            elif (proposed["free4"] == current["free4"]
                  and proposed["d"] == current["d"]
                  and rng.random() < 0.10):
                current = proposed

        finalists = sorted(selected, key=merit, reverse=True)[:3]
        for at, record in enumerate(finalists):
            # Every earlier witness remains binding under retesting.
            challenge(record, side, 61+at+100*restart, 32, 24, 18000)
            challenge(record, side, 71+at+100*restart, 32, 24, 18000)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

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

    def localized_logicals(a, b, hx, hz, n):
        """Check genuine logicals modulo ORIGINAL stabilizer row spans."""
        readings = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(
                    record["words"], key=int.bit_count)[:5]:
                for fixed in range(5 if side == "A" else 3):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j+fixed)*q) for j in range(3))
                    else:
                        actual = word << (fixed*5*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            diagnostic = random.Random(
                seed + (3100009 if side == "A" else 4100011))
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:28]
                    best_word = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < best_word.bit_count():
                                best_word = proposed
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            # n+1 means no nontrivial placement found, not exclusion.
            readings.append(reading)
        return tuple(readings)

    for restart in range(2):
        aa = component_pool("A", restart)
        bb = component_pool("B", restart)
        proposals = []
        for a in aa:
            for b in bb:
                A = [[[G.elements[g] for g in cell] for cell in row]
                     for row in a["grid"]]
                B = [[[G.elements[g] for g in cell] for cell in row]
                     for row in b["grid"]]
                hx, hz, n = lifted_matrices(probe, a["grid"], b["grid"])
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 1428 and k >= 504
                        and 325*n <= 1000*k < 375*n):
                    continue
                actual_a = [
                    sum(((hx[t] >> (j*5*q)) & mask) << (j*q)
                        for j in range(3)) for t in range(q)]
                actual_b = [
                    row & ((1 << (5*q))-1) for row in hz[:2*q]]
                assert actual_a == a["rows"]
                assert actual_b == b["rows"]
                logicals = localized_logicals(a, b, hx, hz, n)
                detected = min(14, a["d"], b["d"], *logicals)
                key = (
                    a["free4"] and b["free4"],
                    k*detected*detected, detected,
                    a["d"]+b["d"], -a["cost"]-b["cost"])
                proposals.append(
                    (key, A, B, hx, hz, k, a, b, logicals))
        if not proposals:
            continue

        key, A, B, hx, hz, k, a, b, logicals = max(
            proposals, key=lambda item: item[0])
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3, 3, 3]]
        assert [sum(row) for row in sb] == [7, 8]
        assert [sum(row[j] for row in sb) for j in range(5)] == [3]*5
        assert all(len(cell) == len(set(cell))
                   for grid in (A, B) for row in grid for cell in row)
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, idx=idx, k=k,
            card=(concentration, policy, cohort),
            quality=key,
            component_readings=(a["d"], b["d"]),
            component_ranks=(a["rank"], b["rank"]),
            component_cycles=(a["cycles"], b["cycles"]),
            component_kernel4_free=(a["free4"], b["free4"]),
            verified_localized_logical_readings=logicals,
            target_cell=0.350)
    raise ValueError("fresh cubic pairings missed the actual 0.350 rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored split-extension cell without new draws."""
    cell = list(ctx[side][i][j])
    assert len(cell) <= 3 and len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
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
