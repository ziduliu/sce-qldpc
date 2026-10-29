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
CONCEPT = ("On split-metacyclic C7 semidirect_2 C12, order 84, A=(1,3), B=(2,5), all-cubic components with independently repaired kernel witnesses should improve the 0.350 incumbent's fitness 47.8. The measured destination bed had four selected readings of twelve and one of ten; eliminating its degree-two B-column is the falsifiable intervention, not a distance certificate.")
GROUP = ("Keep the parent's split-metacyclic construction rule, changing only parameters to m=7,r=12,s=2 because factor seventeen excludes q=110. Since 2^12=1 mod 7, the resulting group has order 84 and n=1428. This exact destination bed measured twelve on four selected codes. Its best shown B support had column degrees (2,3,3,3,3), leaving a legal cubic promotion to test. All search readings remain provisional upper bounds.")
SHAPES = ("Change A=(1,3), B=(2,5): factor seventeen gives n=1428, k>=504 and guaranteed rate 6/17=0.352941; the size window permits q<=88. Components have 252 variables/84 checks and 420 variables/168 checks. The current parent's lightest logicals were entirely sector 1, especially Acol1/2 across Bcol0/1/2. A second B check row changes that component kernel rather than merely relabeling it. Exact full ranks retain every submitted code in cell 0.350.")
FILLING = ("A consists of three distinct-term trinomials. Every B column has total degree three, with row totals seven and eight; HX<=9+3=12 and HZ<=8+3=11. Twenty cards cross B zero budgets 0..3 with five relative-coordinate prescriptions. Cache monomial blocks obtained only from lifted_matrices, exclude component kernels through four, and exploit odd column degree to exclude weight five. Budgeted parity completion, genuine kernel witnesses, witness-directed cell repair and independent finalist challenges guide selection. Final original-group ranks, localized logical retests, full kernel-four exclusion, weights and CSS gate pairings. Neither component screens nor budget exhaustion certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_m7r12s2_13x25_allcubic_witness_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7:C12(s=2) requires order 84")
    # Same split-metacyclic rule: (a,b)=x^a*y^b.
    # The action need not be faithful; 2 has order three modulo seven.
    m, r, s = 7, 12, 2
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


def _retired_affine_plane_ctx(rng, G, idx):
    """Unused predecessor implementation; never called on the resized bed."""
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
    """Twenty all-cubic cards, independently repaired on the resized bed."""
    from itertools import product

    q = G.q
    assert q == 84 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    zero_budget, model = card // 5, card % 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
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
            return inverses[g]

    probe = Probe()

    # These are authoritative isolated monomial lifts. Subsequent assembly
    # uses their GF2 linearity, not a separately implemented group action.
    left, _, _ = lifted_matrices(
        probe, [[[g] for g in range(q)]], [[[]]])
    _, right, _ = lifted_matrices(
        probe, [[[]]], [[[g] for g in range(q)]])
    monomials = {
        "A": [[(row >> (g*q)) & mask for row in left]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in right]
              for g in range(q)],
    }

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q + 1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid resized group order")

    # Relative-order preferences concern actual ratios, not cell gauges.
    penalties = (
        {3: 4, 4: 3, 6: 2, 7: 1, 12: 0, 14: 1, 28: 0},
        {3: 6, 4: 5, 6: 3, 7: 0, 12: 1, 14: 0, 28: 0},
        {3: 3, 4: 6, 6: 0, 7: 2, 12: 0, 14: 2, 28: 1},
        {3: 2, 4: 2, 6: 1, 7: 1, 12: 0, 14: 0, 28: 0},
        {3: 5, 4: 4, 6: 2, 7: 2, 12: 0, 14: 1, 28: 0},
    )
    quotient_profiles = (
        (0, 1, 4), (0, 1, 7), (0, 3, 4),
        (0, 3, 7), (0, 1, 5),
    )

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    def cell_cost(ids, policy):
        relative = [
            orders[table[inverses[g]][h]]
            for i, g in enumerate(ids) for h in ids[:i]
        ]
        if 2 in relative:
            return None
        return sum(penalties[policy].get(order, 8) for order in relative)

    def fresh_cell(count, policy):
        if not count:
            return []
        best, key = None, None
        for _ in range(16):
            ids = rng.sample(range(q), count)
            cost = cell_cost(ids, policy)
            if cost is not None and (key is None or cost < key):
                best, key = ids, cost
        if best is None:
            for _ in range(128):
                ids = rng.sample(range(q), count)
                if cell_cost(ids, policy) is not None:
                    return ids
            raise ValueError("relative-order cell proposal unavailable")
        return best

    def fresh_a_cell(j):
        profile = quotient_profiles[(model + j + cohort) % 5]
        best, key = None, None
        for _ in range(24):
            ids = [G.idx((rng.randrange(7), b)) for b in profile]
            if len(set(ids)) != 3:
                continue
            cost = cell_cost(ids, (model + j) % 5)
            if cost is not None and (key is None or cost < key):
                best, key = ids, cost
        return best if best is not None else fresh_cell(3, model)

    split_plans = [
        top for top in product(range(4), repeat=5)
        if sum(top) == 7
        and sum(value in (0, 3) for value in top) == zero_budget
    ]
    assert split_plans

    def fresh(side):
        if side == "A":
            return [[fresh_a_cell(j) for j in range(3)]]
        top = rng.choice(split_plans)
        support = [list(top), [3 - value for value in top]]
        assert sorted(map(sum, support)) == [7, 8]
        assert all(sum(row[j] for row in support) == 3
                   for j in range(5))
        return [[fresh_cell(count, (model + i + j + cohort) % 5)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def component(grid, side):
        rows = []
        blocks = monomials[side]
        for row in grid:
            for r in range(q):
                combined = 0
                for j, cell in enumerate(row):
                    block = 0
                    for g in cell:
                        block ^= blocks[g][r]
                    combined |= block << (j*q)
                rows.append(combined)
        width = len(grid[0]) * q
        columns = _syndrome_columns(rows, width)
        # Every submitted component column is cubic, including split cells.
        assert all(column.bit_count() == 3 for column in columns)
        return rows, columns

    def cycle_cost(columns):
        uses, four = {}, 0
        for column in columns:
            neighbors = list(bits(column))
            for i, u in enumerate(neighbors):
                for v in neighbors[:i]:
                    key = (v, u)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0) + 1
        return four

    def syndrome(columns, word):
        result = 0
        for c in bits(word):
            result ^= columns[c]
        return result

    def kernel_probe(record, diagnostic_seed, trials):
        """Return genuine classical witnesses; readings are upper bounds."""
        local = random.Random(diagnostic_seed)
        columns = record["columns"]
        width = len(columns)
        order = list(range(width))
        best, witness = width + 1, None
        retained = []

        def observe(word):
            nonlocal best, witness
            if word and word.bit_count() < best:
                best, witness = word.bit_count(), word

        for _ in range(trials):
            local.shuffle(order)
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

            light = sorted(set(basis + retained),
                           key=int.bit_count)[:20]
            new = list(light)
            for word in light:
                for other in basis + retained:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= min(18, best + 4):
                        new.append(candidate)

            local.shuffle(basis)
            for start in light[:6]:
                word, weight = start, start.bit_count()
                for _ in range(3):
                    changed = False
                    for other in basis + retained:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < weight:
                            word, weight = proposed, proposed.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
                new.append(word)
            retained = sorted(set(retained + new),
                              key=int.bit_count)[:24]
            if best <= 6:
                break

        assert witness is not None and syndrome(columns, witness) == 0
        return best, witness

    def challenge(record, side, budget, cap=14):
        """Budgeted parity completion, not an exhaustive distance claim."""
        columns = record["columns"]
        width = len(columns)
        local = random.Random(seed + 3100009 + 100003*int(side == "B"))

        def retain(word):
            if not word or word.bit_count() > cap:
                return
            assert syndrome(columns, word) == 0
            record["words"].add(word)
            if word.bit_count() < record["d"]:
                record["d"], record["word"] = word.bit_count(), word

        if "completion" not in record:
            singles, pairs = {}, {}
            incidence = [0] * len(record["rows"])
            for c, value in enumerate(columns):
                word = 1 << c
                singles[value] = word
                for r in bits(value):
                    incidence[r] |= word
            for i, value in enumerate(columns):
                for j in range(i):
                    pairs[value ^ columns[j]] = (1 << i) | (1 << j)
            record["completion"] = singles, pairs, incidence
        singles, pairs, incidence = record["completion"]

        priorities = [local.random() for _ in columns]
        all_variables = (1 << width) - 1
        roots = list(range(0, width, q))
        local.shuffle(roots)

        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                used = word.bit_count()
                left = min(cap, record["d"]) - used
                if value == 0:
                    retain(word)
                    return
                if left <= 0 or value.bit_count() > 3*left:
                    return

                one = singles.get(value, 0)
                if one and one & allowed == one:
                    retain(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    retain(word | pair)
                left = min(cap, record["d"]) - used
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

            # Only the classical component's regular action is used.
            # It is transitive in each variable block on either lift side.
            allowed = all_variables & ~((1 << (root + 1)) - 1)
            visit(columns[root], 1 << root, allowed)

        record["words"] = set(sorted(record["words"],
                                     key=int.bit_count)[:24])

    def inspect(grid, side):
        rows, columns = component(grid, side)
        if _pairs_no_kernel4(columns) is None:
            return None
        rank = gf2_rank(rows)
        lower = q - 3 if side == "A" else 2*q - 4
        if rank < lower:
            return None
        # Cubic columns make every odd-weight kernel word impossible.
        # The exact <=4 gate therefore also excludes weight five.
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cycles": cycle_cost(columns),
            "signature": signature(grid), "d": len(columns) + 1,
            "words": set(),
        }

    def optimize(side):
        side_id = int(side == "B")
        diagnostic_seed = seed + 100003*side_id
        seen, records = set(), []

        def probe(record, trials, turn):
            d, word = kernel_probe(
                record, diagnostic_seed + 700001*turn, trials)
            record["words"].add(word)
            if d < record["d"]:
                record["d"], record["word"] = d, word

        def merit(record):
            return (record["d"], -record["cycles"], record["rank"],
                    -len(record["words"]))

        for trial in range(72):
            grid = fresh(side)
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            record = inspect(grid, side)
            if record is not None:
                records.append(record)
        records.sort(key=lambda record: (record["cycles"], -record["rank"]))

        selected = []
        for record in records[:10]:
            probe(record, 10, 0)
            challenge(record, side, 6000)
            selected.append(record)
        if not selected:
            return []

        selected.sort(key=merit, reverse=True)
        current = selected[0]
        keepers = selected[:4]

        for step in range(20):
            active = sorted(current["words"], key=int.bit_count)[:6]
            witness = rng.choice(active) if active else current["word"]
            implicated = sorted({c // q for c in bits(witness)})
            j = rng.choice(implicated) if implicated else rng.randrange(
                len(current["grid"][0]))
            occupied = [i for i, row in enumerate(current["grid"]) if row[j]]
            if not occupied:
                continue
            i = rng.choice(occupied)
            grid = [[list(cell) for cell in row] for row in current["grid"]]
            old_cell = grid[i][j]

            if side == "A" and step % 4 == 0:
                grid[i][j] = fresh_a_cell(j)
            elif step % 4 == 0:
                grid[i][j] = fresh_cell(
                    len(old_cell), (model + i + j + cohort) % 5)
            else:
                t = rng.randrange(len(old_cell))
                old = G.elements[old_cell[t]]
                choices = [
                    G.idx((a, old[1])) for a in range(7)
                    if G.idx((a, old[1])) not in old_cell
                ]
                if not choices:
                    continue
                grid[i][j][t] = rng.choice(choices)

            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if proposed["cycles"] > current["cycles"] + q:
                continue
            probe(proposed, 8, step + 1)
            for word in active:
                if syndrome(proposed["columns"], word) == 0:
                    proposed["words"].add(word)
                    if word.bit_count() < proposed["d"]:
                        proposed["d"], proposed["word"] = word.bit_count(), word
            challenge(proposed, side, 4000)

            if merit(proposed) >= merit(current):
                current = proposed
            elif (proposed["d"] == current["d"] and rng.random() < 0.10):
                current = proposed
            keepers.append(proposed)
            keepers.sort(key=merit, reverse=True)
            del keepers[6:]

        unique, signatures = [], set()
        for record in sorted(keepers + [current], key=merit, reverse=True):
            if record["signature"] in signatures:
                continue
            signatures.add(record["signature"])
            # Independent challenges only lower retained readings.
            probe(record, 32, 101)
            probe(record, 48, 211)
            challenge(record, side, 18000)
            record.pop("completion", None)
            unique.append(record)
            if len(unique) >= 3:
                break
        unique.sort(key=merit, reverse=True)
        return unique

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
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

    def localized_reading(a, b, hx, hz, n):
        """Retest genuine localized logicals using ORIGINAL stabilizers."""
        reading = n + 1
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(record["words"], key=int.bit_count)[:3]
            local = random.Random(seed + (4100009 if kind == "Z" else 5100011))
            candidates = []

            for word in words:
                assert syndrome(record["columns"], word) == 0
                positions = range(5) if kind == "Z" else range(3)
                for block in positions:
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((5*j + block)*q)
                            for j in range(3))
                    else:
                        actual = word << (block * 5*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.append(actual)

            for actual in sorted(set(candidates), key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                neighborhood = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:24]
                for _ in range(96):
                    first, second = local.sample(neighborhood, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
        return reading

    for restart in range(2):
        aa, bb = optimize("A"), optimize("B")
        choices = [(a, b) for a in aa for b in bb]
        choices.sort(key=lambda ab: (
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"] + ab[1]["d"],
            -ab[0]["cycles"] - ab[1]["cycles"]), reverse=True)
        best, best_key = None, None

        for a, b in choices[:6]:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert sa == [[3, 3, 3]]
            assert sorted(map(sum, sb)) == [7, 8]
            assert all(sum(row[j] for row in sb) == 3 for j in range(5))
            assert _grid_weights(sa, sb) == 12 <= W_MAX
            assert _no_dead_lines(sa, sb)

            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1428 and k >= 504 and
                    13*n <= 40*k < 15*n):
                continue
            if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
                continue
            if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
                continue

            # Confirm cached assembly against the FINAL original-group lift.
            actual_a = [
                sum(((hx[r] >> (j*5*q)) & mask) << (j*q)
                    for j in range(3))
                for r in range(q)]
            actual_b = [hz[r] & ((1 << (5*q)) - 1) for r in range(2*q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]

            logical = localized_reading(a, b, hx, hz, n)
            conservative = min(14, a["d"], b["d"], logical)
            key = (k * conservative * conservative, conservative,
                   -a["cycles"] - b["cycles"], k)
            if best_key is None or key > best_key:
                best_key = key
                best = (A, B, hx, hz, k, a, b, logical)

        if best is not None:
            A, B, hx, hz, k, a, b, logical = best
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert css_ok(hx, hz)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (zero_budget, model, cohort),
                "component_ranks": (a["rank"], b["rank"]),
                "component_readings": (a["d"], b["d"]),
                "component_four_cycles": (a["cycles"], b["cycles"]),
                "localized_logical_reading": logical,
                "B_column_degrees": (3, 3, 3, 3, 3),
                "target_cell": 0.350,
            }
    raise ValueError("no new all-cubic pair passed the verified rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell of the new one-row/two-row shapes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane preference produces accepted k=720..750 gains; all 51 distinct current codes instead had k=690, and all five measured readings remained six.
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
