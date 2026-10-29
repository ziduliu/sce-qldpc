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
CONCEPT = ("Hypothesis: on (C5 x C5) semidirect C3, order 75, A=(1,3), B=(2,5), cubic columns and independently witness-repaired components produce five new measured readings >=12 in cell 0.350, exceeding its incumbent fitness 47.8. Five such readings at the rank bound would score 50.8235. Failure of this five-code target refutes the hypothesis; readings remain provisional upper bounds.")
GROUP = ("Jump from SL(2,F5), order 120, to the untried order-75 Frobenius group (F5^2,+):C3. The action M(a,b)=(-b,a-b) has order three and irreducible polynomial x^2+x+1 over F5. A nonzero translation and its action orbit span the normal C5^2. Different order establishes a genuine group change. Successful SL2, A4 x C11 and F9:C16 cubic beds motivate this experiment without establishing distance transfer.")
SHAPES = ("Change to A=(1,3), B=(2,5), q=75: n=1275, k>=450 and guaranteed rate 6/17=0.352941, targeting the weaker 0.350 frontier. The shape permits q<=88. Unequal components have 225 variables/75 checks and 375 variables/150 checks, nominal rates 2/3 and 3/5. Parent weight-ten Z minima were sector 1, concentrated on Acol1/2 at Bcol1/2; both corresponding component families are challenged. Exact full ranks require every submitted rate below 0.375.")
FILLING = ("All component columns have three terms. A has three cubic cells; B has five cubic columns with row totals seven/eight, giving HX=9+3=12 and HZ<=8+3=11. Twenty cards cross zero through three quotient-balanced A cells with five B concentration/relative-order policies. Whole-cell directed-difference uniqueness and exact generation screen cubic proposals. Assemble only authoritative scaffold monomial blocks, retain genuine kernel witnesses across independent circuit/descent and budgeted parity searches, and replace actual implicated cells. Deep finalists receive exact full ranks and original-stabilizer localized-logical challenges. Full kernel-four exclusion, native-lift agreement, CSS and both weight sums gate submission. Search exhaustion and component readings do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_F5sq_C3_13x25_cubic_witness_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 75

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F5^2 semidirect C3 requires order 75")
    # (a,b,t) = translation (a,b), followed by M^t.
    # M(a,b)=(-b,a-b); M^2(a,b)=(b-a,-a); M^3=I.
    # x^2+x+1 is irreducible over F5, so the action is fixed-point-free.
    els = [(a,b,t) for t in range(3)
           for b in range(5) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 5, int(g[2]) % 3)

    def act(t, a, b):
        t %= 3
        if t == 0:
            return (a % 5, b % 5)
        if t == 1:
            return ((-b) % 5, (a-b) % 5)
        return ((b-a) % 5, (-a) % 5)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        c,d = act(t, c, d)
        return ((a+c) % 5, (b+d) % 5, (t+u) % 3)

    def inv(g):
        a,b,t = red(g)
        a,b = act(-t, a, b)
        return ((-a) % 5, (-b) % 5, (-t) % 3)

    def idx(g):
        a,b,t = red(g)
        return 25*t+5*b+a

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


def _legacy_affine_ctx(rng, G, idx):
    """Unused historical affine proposal; the new NEW_CTX is below."""
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


def _legacy_sl2_ctx(rng, G, idx):
    """Retired SL2 spectrum optimizer; NEW_CTX below replaces it."""
    q = G.q
    width = 3 * q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    model, composition = card // 5, card % 5
    diagnostic_seed = rng.randrange(1 << 62)
    assert q == 120
    assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
    assert W_MAX >= 12

    # These screening-only dummy partners are never submitted.
    # Every cached monomial block comes from the verified constructor.
    raw_left, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, raw_right, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    blocks = {
        "A": [[(row >> (j*q)) & mask for row in raw_left]
              for j in range(q)],
        "B": [[(row >> (j*q)) & mask for row in raw_right]
              for j in range(q)],
    }
    del raw_left, raw_right

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    orders = []
    for j in range(q):
        value = ident
        for order in range(1, q + 1):
            value = table[value][j]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("matrix group element has no finite order")

    def generates(a, b):
        reached = {ident}
        todo = [ident]
        for x in todo:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    todo.append(z)
        return len(reached) == q

    def signature(ids):
        # Remove root-normalization duplicates inside the cell library.
        # This is not a claim of complete Tanner-isomorphism detection.
        return min(tuple(sorted(table[inverses[root]][g] for g in ids))
                   for root in ids)

    targets = ((4, 5, 5), (4, 5, 10), (3, 4, 5), (4, 6, 10))
    library, signatures = [], set()
    nonidentity = [j for j in range(q) if j != ident]
    for attempt in range(6000):
        if len(library) >= 72:
            break
        a, b = rng.sample(nonidentity, 2)
        spectrum = tuple(sorted(
            (orders[a], orders[b], orders[table[inverses[a]][b]])))
        if 2 in spectrum:
            continue
        errors = tuple(sum(abs(x-y) for x, y in zip(spectrum, target))
                       for target in targets)
        # Enrichment is soft: later proposals supply a general fallback.
        if attempt < 600 and min(errors):
            continue
        ids = (ident, a, b)
        sig = signature(ids)
        if sig in signatures or not generates(a, b):
            continue
        signatures.add(sig)
        record = {"ids": ids, "errors": errors, "spectrum": spectrum}
        for side in ("A", "B"):
            monomials = blocks[side]
            rows = [monomials[ident][r] ^ monomials[a][r]
                    ^ monomials[b][r] for r in range(q)]
            record[side] = rows
            record["rank"+side] = gf2_rank(rows)
        library.append(record)
    if len(library) < 8:
        raise ValueError("insufficient full-generating cubic geometries")

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

    def kernel_probe(columns, seed, trials):
        """Actual classical kernel words; sampled upper readings only."""
        local = random.Random(seed)
        order = list(range(width))
        reading, witness = width + 1, None

        def consider(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

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
                    consider(word)

            light = sorted(basis, key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    consider(word ^ other)

            # Every descent step stays in the kernel; zero is excluded.
            local.shuffle(basis)
            for start in light[:4]:
                word, weight = start, start.bit_count()
                for sweep in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    consider(word)
                    if not changed:
                        break
            if len(light) >= 3:
                for _ in range(64):
                    a, b, c = local.sample(light, 3)
                    consider(a ^ b ^ c)
            if reading <= 6:
                break

        assert witness is not None and syndrome(columns, witness) == 0
        return reading, witness

    def geometry(rows, columns):
        four = _four_cycles(rows)
        adjacency = [0] * q
        forced = 0
        for column in columns:
            ns = list(bits(column))
            degree = len(ns)
            forced += degree * (degree-1) * (degree-2) // 6
            for r in ns:
                adjacency[r] |= column ^ (1 << r)
        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u + 1))):
                triangles += (neighbors & adjacency[v]
                              & (-1 << (v + 1))).bit_count()
        # Projection excess is a heuristic, not an exact C6 count.
        return four, max(0, triangles - forced)

    offsets = ((0, 0, 0), (0, 0, 1), (0, 1, 1),
               (0, 1, 2), (0, 2, 3))

    def optimize(side):
        shift = 0 if side == "A" else 1 + cohort % 3
        plan = tuple((model + shift + t) % 4
                     for t in offsets[composition])
        eligible = [
            sorted(range(len(library)),
                   key=lambda t: (-library[t]["rank"+side],
                                  library[t]["errors"][kind]))[:24]
            for kind in plan
        ]
        records, seen = [], set()
        penalty = (128, 256, 512, 1024, 2048)[composition]

        for trial in range(48):
            ids = tuple(rng.choice(pool) for pool in eligible)
            if len(set(ids)) != 3 or tuple(sorted(ids)) in seen:
                continue
            seen.add(tuple(sorted(ids)))
            rows = [
                sum(library[t][side][r] << (j*q)
                    for j, t in enumerate(ids))
                for r in range(q)
            ]
            columns = _syndrome_columns(rows, width)
            if 0 in columns or len(set(columns)) != width:
                continue
            stats = geometry(rows, columns)
            error = sum(library[t]["errors"][kind]
                        for t, kind in zip(ids, plan))
            rank = gf2_rank(rows)
            records.append({
                "ids": ids, "rows": rows, "columns": columns,
                "stats": stats, "error": error, "rank": rank,
                "cost": penalty*stats[0] + stats[1],
            })
        if not records:
            raise ValueError("no distinct cubic component proposals")

        records.sort(key=lambda r: (r["cost"], r["error"], -r["rank"]))
        screened = []
        for record in records:
            record["free4"] = (
                _pairs_no_kernel4(record["columns"]) is not None)
            if not record["free4"]:
                continue
            d, word = kernel_probe(
                record["columns"],
                diagnostic_seed + (0 if side == "A" else 100003), 6)
            record["reading"], record["word"] = d, word
            screened.append(record)
            if len(screened) >= 6:
                break

        # Preserve legal construction if this preference proves infeasible.
        if not screened:
            for record in records[:3]:
                d, word = kernel_probe(
                    record["columns"], diagnostic_seed + 1709, 6)
                record["reading"], record["word"] = d, word
                record["free4"] = False
                screened.append(record)

        def merit(record):
            return (record["free4"], record["reading"],
                    -record["cost"], -record["error"], record["rank"])

        screened.sort(key=merit, reverse=True)
        finalists = screened[:3]
        for record in finalists:
            # Independent confirmation can only decrease a reading.
            for offset, trials in ((700001, 32), (1900003, 32)):
                d, word = kernel_probe(
                    record["columns"],
                    diagnostic_seed + offset
                    + (0 if side == "A" else 100003), trials)
                record["reading"] = min(record["reading"], d)
                if word.bit_count() < record["word"].bit_count():
                    record["word"] = word
            record["cells"] = [[
                [G.elements[g] for g in library[t]["ids"]]
                for t in record["ids"]
            ]]
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa, bb = optimize("A"), optimize("B")
    best, best_key = None, None
    for a in aa:
        for b in bb:
            A, B = a["cells"], b["cells"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            assert n == 1200 and k >= 480
            in_band = 375*n <= 1000*k < 425*n
            key = (in_band, a["free4"] and b["free4"],
                   min(a["reading"], b["reading"]),
                   a["reading"] + b["reading"],
                   -a["cost"] - b["cost"],
                   -a["error"] - b["error"],
                   -abs(5*k - 2*n))
            if best_key is None or key > best_key:
                best_key = key
                best = (a, b, hx, hz, n, k)

    a, b, hx, hz, n, k = best
    A, B = a["cells"], b["cells"]
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert sa == sb == [[3, 3, 3]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    assert max(row.bit_count() for row in hx + hz) == 12
    assert css_ok(hx, hz)

    # Verify the cached linear component assembly against the final lift.
    authoritative_a = [
        sum(((hx[r] >> (j*3*q)) & mask) << (j*q)
            for j in range(3))
        for r in range(q)
    ]
    authoritative_b = [hz[r] & ((1 << width)-1) for r in range(q)]
    assert authoritative_a == a["rows"]
    assert authoritative_b == b["rows"]
    return {
        "A": A, "B": B, "idx": idx, "k": k,
        "card": (model, composition, cohort),
        "target_band": bool(best_key[0]),
        "component_readings": (a["reading"], b["reading"]),
        "component_ranks": (a["rank"], b["rank"]),
        "component_cycles": (a["stats"], b["stats"]),
        "kernel4_free": (a["free4"], b["free4"]),
    }


def NEW_CTX(rng, G, idx):
    """Twenty cubic-column cards on F5^2:C3, using authoritative lifts."""
    q = G.q
    assert q == 75 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    balanced, topology = divmod(card, 5)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    sa = [[3, 3, 3]]
    bplans = (
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((3, 1, 1, 1, 1), (0, 2, 2, 2, 2)),
        ((3, 0, 2, 1, 1), (0, 3, 1, 2, 2)),
        ((3, 3, 0, 1, 1), (0, 0, 3, 2, 2)),
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
    )
    sb = [list(row) for row in bplans[topology]]
    assert all(sum(sb[i][j] for i in range(2)) == 3
               for j in range(5))
    assert sorted(map(sum, sb)) == [7, 8]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

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

    P = Probe()

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length() - 1
            word ^= low

    def syndrome(columns, word):
        answer = 0
        for c in bits(word):
            answer ^= columns[c]
        return answer

    # Both lift conventions are obtained from the verified constructor.
    # Small dummy batches are screening objects, never submitted codes.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 15):
        ids = list(range(start, min(q, start + 15)))
        dummy = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, dummy, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], dummy)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    def copy(grid):
        return [[list(cell) for cell in row] for row in grid]

    def whole_sidon(ids):
        if len(ids) != len(set(ids)):
            return False
        left, right = set(), set()
        for a in ids:
            for b in ids:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(ids) * (len(ids) - 1)
        return len(left) == len(right) == wanted

    def generates(a, b):
        reached, pending = {identity}, [identity]
        for g in pending:
            for h in (a, b):
                value = table[g][h]
                if value not in reached:
                    reached.add(value)
                    pending.append(value)
        return len(reached) == q

    # Balanced and repeated quotient occupancies are distinct cell policies.
    # Cards vary their allocation, not a global automorphism of one filling.
    atom_pools = [[], []]
    for kind in range(2):
        seen = set()
        for attempt in range(2400):
            if kind == 0:
                gs = (G.e,
                      (rng.randrange(5), rng.randrange(5), 1),
                      (rng.randrange(5), rng.randrange(5), 2))
            else:
                u = rng.choice([(a, b) for a in range(5)
                                for b in range(5) if a or b])
                gs = (G.e, u + (0,),
                      (rng.randrange(5), rng.randrange(5), 1))
            ids = tuple(G.idx(g) for g in gs)
            key = tuple(sorted(ids))
            if key in seen or not whole_sidon(ids):
                continue
            if not generates(ids[1], ids[2]):
                continue
            seen.add(key)
            atom_pools[kind].append(ids)
            if len(atom_pools[kind]) >= 64:
                break
        if len(atom_pools[kind]) < 12:
            raise ValueError("cubic generation/Sidon library exhausted")

    nonzero_translations = [G.idx((a, b, 0))
                            for a in range(5) for b in range(5) if a or b]
    mixed_ratios = [G.idx((a, b, t))
                    for t in (1, 2) for a in range(5) for b in range(5)]

    def draw_cell(side, i, j):
        count = (sa if side == "A" else sb)[i][j]
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 3:
            kind = (0 if j < balanced else 1) if side == "A" else (
                (balanced + j + i + cohort) % 2)
            return list(rng.choice(atom_pools[kind]))
        # B card four uses exclusively order-five binomial ratios.
        # Other cards use mixed order-three/order-five prescriptions.
        normal = topology == 4 or (
            topology == 2 and (i + j + cohort) % 2 == 0)
        ratios = nonzero_translations if normal else mixed_ratios
        anchor, ratio = rng.randrange(q), rng.choice(ratios)
        return [anchor, table[anchor][ratio]]

    def fresh(side):
        support = sa if side == "A" else sb
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(support)]

    def inspect(grid, side):
        width = len(grid[0]) * q
        rows = []
        for cells in grid:
            for r in range(q):
                row = 0
                for j, cell in enumerate(cells):
                    block = 0
                    for g in cell:
                        block ^= blocks[side][g][r]
                    row |= block << (j*q)
                rows.append(row)
        columns = _syndrome_columns(rows, width)
        assert all(column.bit_count() == 3 for column in columns)
        if 0 in columns or len(set(columns)) != width:
            return None
        rank = gf2_rank(rows)
        if rank < len(rows) - 6:
            return None

        four = _four_cycles(rows)
        adjacency = [0] * len(rows)
        for column in columns:
            for r in bits(column):
                adjacency[r] |= column ^ (1 << r)
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << len(rows)) - 1:
            return None
        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u + 1))):
                triangles += (
                    neighbors & adjacency[v] & (-1 << (v + 1))
                ).bit_count()
        # Column degree three contributes one forced projection triangle.
        excess = max(0, triangles - width)
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(grid=copy(grid), rows=rows, columns=columns,
                    signature=signature, rank=rank,
                    cycles=(four, excess),
                    cost=(300 + 160*topology)*four + excess,
                    d=width + 1, word=None, words=set(), free4=False)

    def note(rec, word):
        if not word:
            return
        weight = word.bit_count()
        if weight < rec["d"]:
            rec["d"], rec["word"] = weight, word
        if weight <= max(18, rec["d"] + 4):
            rec["words"].add(word)
        if len(rec["words"]) > 160:
            rec["words"] = set(sorted(
                rec["words"], key=int.bit_count)[:32])

    def circuit_query(rec, trials, diagnostic_seed):
        """Actual component kernel witnesses; no exclusion certificate."""
        local = random.Random(diagnostic_seed)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = sorted(rec["words"], key=int.bit_count)[:32]
        for _ in range(trials):
            local.shuffle(order)
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
                    note(rec, word)
            light = sorted(set(basis + retained),
                           key=int.bit_count)[:20]
            moves = basis + retained
            for word in light:
                for other in moves:
                    note(rec, word ^ other)
            local.shuffle(moves)
            for start in light[:6]:
                word = start
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    note(rec, word)
                    if not changed:
                        break
            if len(light) >= 3:
                for _ in range(64):
                    a, b, c = local.sample(light, 3)
                    note(rec, a ^ b ^ c)
            retained = sorted(rec["words"], key=int.bit_count)[:32]
            rec["words"] = set(retained)
            if rec["d"] <= 4:
                break
        assert rec["word"] is not None
        assert syndrome(columns, rec["word"]) == 0

    def parity_query(rec, budget, diagnostic_seed):
        """Exact <=4 collisions, then budgeted parity completion through twelve."""
        columns = rec["columns"]
        width = len(columns)
        singles, pairs = {}, {}
        incidence = [0] * len(rec["rows"])
        for c, value in enumerate(columns):
            word = 1 << c
            previous = singles.get(value)
            if previous is not None:
                note(rec, previous ^ word)
            else:
                singles[value] = word
            for r in bits(value):
                incidence[r] |= word
        for i, value in enumerate(columns):
            for j in range(i):
                value2 = value ^ columns[j]
                word = (1 << i) | (1 << j)
                previous = singles.get(value2)
                if previous is not None:
                    note(rec, previous ^ word)
                previous = pairs.get(value2)
                if previous is not None:
                    note(rec, previous ^ word)
                else:
                    pairs[value2] = word
        rec["free4"] = rec["d"] > 4
        if not rec["free4"] or budget <= 0:
            return
        local = random.Random(diagnostic_seed)
        priorities = [local.random() for _ in columns]
        roots = list(range(0, width, q))
        local.shuffle(roots)
        all_variables = (1 << width) - 1

        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    note(rec, word)
                    return
                left = min(12, rec["d"]) - word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    note(rec, word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(rec, word | pair)
                left = min(12, rec["d"]) - word.bit_count()
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

            # Regular COMPONENT symmetry is transitive within each block.
            # No full nonabelian quantum-code symmetry is assumed.
            allowed = all_variables & ~((1 << (root + 1)) - 1)
            visit(columns[root], 1 << root, allowed)
        rec["words"] = set(sorted(rec["words"], key=int.bit_count)[:32])

    def optimize(side):
        offset = 0 if side == "A" else 100003
        records, seen, witness_bank = [], set(), set()

        def add(grid):
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in seen:
                return None
            seen.add(rec["signature"])
            return rec

        def query(rec, trials, turn, budget):
            for word in witness_bank:
                if syndrome(rec["columns"], word) == 0:
                    note(rec, word)
            circuit_query(rec, trials, seed + offset + 700001*turn)
            parity_query(rec, budget, seed + offset + 310001*turn)
            witness_bank.update(rec["words"])
            if len(witness_bank) > 80:
                keep = sorted(witness_bank, key=int.bit_count)[:48]
                witness_bank.clear()
                witness_bank.update(keep)

        def merit(rec):
            # Multiplicities are sampled, not exhaustive.
            return (rec["free4"], min(16, rec["d"]),
                    -sum(w.bit_count() == rec["d"] for w in rec["words"]),
                    -rec["cost"], rec["rank"])

        for _ in range(96):
            rec = add(fresh(side))
            if rec is not None:
                records.append(rec)
        if not records:
            raise ValueError("no connected cubic-column proposals")
        records.sort(key=lambda rec: (rec["cost"], -rec["rank"]))

        keepers = []
        for number, rec in enumerate(records):
            if _pairs_no_kernel4(rec["columns"]) is None:
                continue
            query(rec, 8, 1 + number, 8000)
            keepers.append(rec)
            if len(keepers) >= 8:
                break
        if not keepers:
            # Preserve a legal independently filled slot if the preference fails.
            keepers = records[:3]
            for number, rec in enumerate(keepers):
                query(rec, 8, 31 + number, 8000)
        current = max(keepers, key=merit)

        for step in range(24):
            occupied = sorted({c // q for c in bits(current["word"])})
            j = rng.choice(occupied) if occupied else rng.randrange(
                len(current["grid"][0]))
            proposals = []
            for move in range(2):
                grid = copy(current["grid"])
                active = [i for i in range(len(grid)) if grid[i][j]]
                i = rng.choice(active)
                grid[i][j] = draw_cell(side, i, j)
                rec = add(grid)
                if rec is None:
                    continue
                if rec["cycles"][0] > current["cycles"][0] + q:
                    continue
                for word in current["words"]:
                    if syndrome(rec["columns"], word) == 0:
                        note(rec, word)
                query(rec, 5, 101 + 2*step + move, 6000)
                proposals.append(rec)
            if not proposals:
                continue
            proposed = max(proposals, key=merit)
            if merit(proposed) >= merit(current):
                current = proposed
            elif (merit(proposed)[:2] == merit(current)[:2]
                  and rng.random() < 0.10):
                current = proposed
            keepers.extend(proposals)
            keepers.sort(key=merit, reverse=True)
            del keepers[8:]

        unique = {rec["signature"]: rec for rec in keepers + [current]}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:3]
        for number, rec in enumerate(finalists):
            # Independent challenges retain all previous witnesses.
            query(rec, 40, 211 + number, 30000)
            query(rec, 64, 311 + number, 45000)
            rec["words"] = set(sorted(rec["words"], key=int.bit_count)[:16])
            assert syndrome(rec["columns"], rec["word"]) == 0
        finalists.sort(key=merit, reverse=True)
        return finalists

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

    def localized(a, b, hx, hz, n):
        """Actual nontrivial logical witnesses with ORIGINAL stabilizers."""
        answer = n + 1
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            local = random.Random(seed + (4100009 if kind == "Z" else 5100011))
            words = sorted(rec["words"] | {rec["word"]},
                           key=int.bit_count)[:4]
            actual_words = []
            for word in words:
                assert syndrome(rec["columns"], word) == 0
                for block in range(5 if kind == "Z" else 3):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((5*j + block)*q)
                            for j in range(3))
                    else:
                        actual = word << (block * 5*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        actual_words.append(actual)
            for actual in sorted(set(actual_words), key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                neighborhood = sorted(
                    stabilizers, key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:32]
                for _ in range(96):
                    first, second = local.sample(neighborhood, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                answer = min(answer, actual.bit_count())
        return answer

    aa, bb = optimize("A"), optimize("B")
    proposals = []
    for a in aa:
        for b in bb:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1275 and k >= 450
                    and 325*n <= 1000*k < 375*n):
                continue
            actual_a = [
                sum(((hx[r] >> (j*5*q)) & mask) << (j*q)
                    for j in range(3)) for r in range(q)
            ]
            actual_b = [hz[r] & ((1 << (5*q)) - 1) for r in range(2*q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logical = localized(a, b, hx, hz, n)
            conservative = min(12, a["d"], b["d"], logical)
            key = (a["free4"] and b["free4"],
                   k*conservative*conservative, conservative,
                   min(16, a["d"]) + min(16, b["d"]),
                   -a["cost"] - b["cost"])
            proposals.append((key, a, b, hx, hz, k, logical))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for key, a, b, cached_hx, cached_hz, k, logical in proposals:
        # This is full-check-matrix kernel exclusion, not only a component test.
        if _pairs_no_kernel4(_syndrome_columns(cached_hx, 1275)) is None:
            continue
        if _pairs_no_kernel4(_syndrome_columns(cached_hz, 1275)) is None:
            continue
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert hx == cached_hx and hz == cached_hz and n == 1275
        assert n - gf2_rank(hx) - gf2_rank(hz) == k > 0
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert [[len(cell) for cell in row] for row in A] == sa
        assert [[len(cell) for cell in row] for row in B] == sb
        assert all(len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (balanced, topology, cohort),
            "component_readings": (a["d"], b["d"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "verified_localized_logical_reading": logical,
            "quality": key,
        }
    raise ValueError("challenged cubic-column pairs missed full/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected order-75 cell, including deliberate zeros."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane bank produces verified k=720..750 improvements; all 51 distinct current codes instead had k=690, and all five measured distances remained six.
# REFUTED: Full-generating SL(2,F5) cubic spectrum/descent selection makes all five selected readings exceed ten; current slot zero returned ten, with 52 sector-1 Z witnesses concentrated on Acol1/2 at Bcol1/2.
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
