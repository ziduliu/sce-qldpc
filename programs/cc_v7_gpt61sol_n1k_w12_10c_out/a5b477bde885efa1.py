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
CONCEPT = ("SL(2,5), order 120, A=B=(1,3): witness-directed central-sign changes and whole-cell replacements should produce five new distance readings >=12 and improve cell 0.400 beyond fitness 54.0800. Failure of that five-code target refutes the selection hypothesis; readings remain provisional upper bounds.")
GROUP = ("Frozen SL(2,F5), order 120. This bed measured four selected distances of twelve and one of ten; related beds reached fitness 109.4 on (1,2)x(2,5) and 69.4 on (1,2)x(3,4). Keep the group and use its central involution for local labeling repairs. These measurements motivate selection changes, not a distance-transfer theorem.")
SHAPES = ("Frozen A=B=(1,3): equal components have 360 variables and 120 checks. n=1200 and k>=480 guarantee rate 0.400; the shape permits lift order at most 149. Cubic cells give both quantum row weights twelve. Parent slot zero's weight-ten Z witnesses occupied sector 1, especially Acol1/2 at Bcol1/2; both components are repaired. All measured k-kb values were zero. Exact full ranks retain [0.375,0.425), rather than deliberately adding a common-annihilator deficit.")
FILLING = ("Three distinct terms per cell, hence w=12. Twenty cards cross four new relative-order targets with five spectrum compositions. A larger full-generating trinomial library is assembled from verified monomial lifts. Independently screen both components using genuine circuit witnesses and budgeted anchored parity completion through weight twelve. Deduplicate detected witness orbits, then repair implicated cells by central-sign changes or whole-cell replacement. Stronger independent searches challenge finalists. Authoritative lifts check ranks, component assembly, CSS and row weights. Search exhaustion is not exhaustive exclusion, and component readings do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040_SL2F5_13x13_cubic_witness12_centralrepair_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL(2,F5) requires order 120")

    # Matrices (a,b;c,d) with determinant one, without quotienting by -I.
    els = [(a,b,c,d) for a in range(5) for b in range(5)
           for c in range(5) for d in range(5)
           if (a*d-b*c) % 5 == 1]
    assert len(els) == q
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 5 for x in g)

    def mul(g, h):
        a,b,c,d = g
        e,f,u,v = h
        return ((a*e+b*u) % 5, (a*f+b*v) % 5,
                (c*e+d*u) % 5, (c*f+d*v) % 5)

    def inv(g):
        a,b,c,d = g
        return (d % 5, (-b) % 5, (-c) % 5, a % 5)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1,0,0,1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
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


def NEW_CTX(rng, G, idx):
    """Twenty cubic spectrum cards on the new equal-component bed."""
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

    targets = ((3, 5, 5), (3, 5, 10), (5, 5, 6), (5, 6, 10))
    central = next(
        c for c in range(q) if c != ident
        and orders[c] == 2
        and all(table[c][g] == table[g][c] for g in range(q)))

    def make_atom(ids):
        assert len(set(ids)) == 3 and ids[0] == ident
        a, b = ids[1:]
        spectrum = tuple(sorted(
            (orders[a], orders[b], orders[table[inverses[a]][b]])))
        errors = tuple(sum(abs(x-y) for x, y in zip(spectrum, target))
                       for target in targets)
        record = {"ids": ids, "errors": errors, "spectrum": spectrum}
        for side in ("A", "B"):
            monomials = blocks[side]
            rows = [monomials[ids[0]][r] ^ monomials[ids[1]][r]
                    ^ monomials[ids[2]][r] for r in range(q)]
            record[side] = rows
            record["rank"+side] = gf2_rank(rows)
        return record

    library, signatures = [], set()
    nonidentity = [j for j in range(q) if j != ident]
    for attempt in range(12000):
        if len(library) >= 192:
            break
        a, b = rng.sample(nonidentity, 2)
        spectrum = tuple(sorted(
            (orders[a], orders[b], orders[table[inverses[a]][b]])))
        if 2 in spectrum:
            continue
        errors = tuple(sum(abs(x-y) for x, y in zip(spectrum, target))
                       for target in targets)
        # Enrichment is a preference, never an infeasible labeling gate.
        if attempt < 1200 and min(errors):
            continue
        ids = (ident, a, b)
        sig = signature(ids)
        if sig in signatures or not generates(a, b):
            continue
        signatures.add(sig)
        library.append(make_atom(ids))
    if len(library) < 12:
        raise ValueError("insufficient full-generating cubic geometries")

    atom_cache = {
        tuple(sorted(record["ids"])): t
        for t, record in enumerate(library)
    }

    def central_variant(t, position):
        ids = list(library[t]["ids"])
        ids[position] = table[central][ids[position]]
        ids = tuple(ids)
        if len(set(ids)) != 3 or not generates(ids[1], ids[2]):
            return t
        key = tuple(sorted(ids))
        old = atom_cache.get(key)
        if old is not None:
            return old
        atom_cache[key] = len(library)
        library.append(make_atom(ids))
        return len(library) - 1

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

        retained = []
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

            light = sorted(set(basis + retained), key=int.bit_count)[:20]
            candidates = list(light)
            for word in light:
                for other in basis + retained:
                    candidate = word ^ other
                    consider(candidate)
                    if candidate and candidate.bit_count() <= min(18, reading + 4):
                        candidates.append(candidate)

            # Accumulated circuits permit cancellations across trials.
            local.shuffle(basis)
            for start in light[:6]:
                word, weight = start, start.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in basis + retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    consider(word)
                    if not changed:
                        break
                candidates.append(word)
            if len(light) >= 3:
                for _ in range(96):
                    a, b, c = local.sample(light, 3)
                    consider(a ^ b ^ c)
            retained = sorted(set(retained + candidates),
                              key=int.bit_count)[:28]
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

    def orbit_key(word, side):
        """Deduplicate witnesses under the actual component symmetry.

        Left-lifted A admits right multiplication; right-lifted B admits
        left multiplication. Symmetry images are never submitted as codes.
        """
        first = (word & -word).bit_length() - 1
        block = first // q
        anchors = (word >> (block*q)) & mask
        positions = list(bits(word))
        answer = None
        for anchor in bits(anchors):
            multiplier = inverses[anchor]
            moved = 0
            for position in positions:
                j, h = divmod(position, q)
                image = (table[h][multiplier] if side == "A"
                         else table[multiplier][h])
                moved |= 1 << (j*q + image)
            answer = moved if answer is None else min(answer, moved)
        return answer

    def challenge(record, side, budget, cap=12):
        """Budgeted parity completion; only found words are conclusions."""
        columns = record["columns"]
        witnesses = record.setdefault("witnesses", {})

        def retain(word):
            if not word or word.bit_count() > cap:
                return
            assert syndrome(columns, word) == 0
            key = orbit_key(word, side)
            witnesses.setdefault(key, word)
            if word.bit_count() < record["reading"]:
                record["reading"], record["word"] = word.bit_count(), word

        retain(record["word"])
        if "completion" not in record:
            singles, pairs = {}, {}
            incidence = [0] * q
            for c, value in enumerate(columns):
                word = 1 << c
                previous = singles.get(value)
                if value == 0:
                    retain(word)
                if previous is not None:
                    retain(previous ^ word)
                else:
                    singles[value] = word
                for r in bits(value):
                    incidence[r] |= word
            for i, value in enumerate(columns):
                for j in range(i):
                    combined = value ^ columns[j]
                    word = (1 << i) | (1 << j)
                    previous = singles.get(combined)
                    if previous is not None:
                        retain(previous ^ word)
                    previous = pairs.get(combined)
                    if previous is not None:
                        retain(previous ^ word)
                    else:
                        pairs[combined] = word
            record["completion"] = singles, pairs, incidence

        singles, pairs, incidence = record["completion"]
        if record["reading"] <= 4:
            return
        local = random.Random(diagnostic_seed + 3100009
                              + (0 if side == "A" else 100003))
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
                used = word.bit_count()
                left = min(cap, record["reading"]) - used
                if value == 0:
                    retain(word)
                    return
                if left <= 0 or value.bit_count() > 3 * left:
                    return

                one = singles.get(value, 0)
                if one and one & allowed == one:
                    retain(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    retain(word | pair)
                left = min(cap, record["reading"]) - used
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

            # Regular symmetry moves a word's first occupied block term
            # to coordinate zero. No full quantum symmetry is assumed.
            allowed = all_variables & ~((1 << (root + 1)) - 1)
            visit(columns[root], 1 << root, allowed)

        if witnesses:
            minimum = min(word.bit_count() for word in witnesses.values())
            record["witnesses"] = {
                key: word for key, word in witnesses.items()
                if word.bit_count() == minimum
            }

    offsets = ((0, 0, 0), (0, 0, 1), (0, 1, 1),
               (0, 1, 2), (0, 2, 3))

    def optimize(side):
        shift = 0 if side == "A" else 1 + cohort % 3
        plan = tuple((model + shift + t) % 4
                     for t in offsets[composition])
        eligible = [
            sorted(range(len(library)),
                   key=lambda t: (library[t]["errors"][kind],
                                  -library[t]["rank"+side]))[:48]
            for kind in plan
        ]
        records, seen = [], set()
        penalty = (128, 256, 512, 1024, 2048)[composition]
        side_seed = diagnostic_seed + (0 if side == "A" else 100003)

        def assemble(ids):
            signature_ids = tuple(sorted(
                tuple(sorted(library[t]["ids"])) for t in ids))
            if len(set(signature_ids)) != 3 or signature_ids in seen:
                return None
            seen.add(signature_ids)
            rows = [
                sum(library[t][side][r] << (j*q)
                    for j, t in enumerate(ids))
                for r in range(q)
            ]
            columns = _syndrome_columns(rows, width)
            if 0 in columns or len(set(columns)) != width:
                return None
            stats = geometry(rows, columns)
            return {
                "ids": tuple(ids), "rows": rows, "columns": columns,
                "stats": stats,
                "error": sum(library[t]["errors"][kind]
                             for t, kind in zip(ids, plan)),
                "rank": gf2_rank(rows),
                "cost": penalty*stats[0] + stats[1],
                "witnesses": {},
            }

        for trial in range(96):
            record = assemble(tuple(rng.choice(pool) for pool in eligible))
            if record is not None:
                records.append(record)
        if not records:
            raise ValueError("no distinct cubic component proposals")
        records.sort(key=lambda r: (r["cost"], r["error"], -r["rank"]))

        def initial_probe(record, budget):
            record["free4"] = (
                _pairs_no_kernel4(record["columns"]) is not None)
            d, word = kernel_probe(record["columns"], side_seed, 8)
            record["reading"], record["word"] = d, word
            challenge(record, side, budget)

        screened = []
        for record in records:
            if _pairs_no_kernel4(record["columns"]) is None:
                continue
            initial_probe(record, 12000)
            screened.append(record)
            if len(screened) >= 8:
                break
        if not screened:
            for record in records[:3]:
                initial_probe(record, 6000)
                screened.append(record)

        def merit(record):
            # Witness counts are sampled orbit counts, not exhaustive
            # multiplicities or distance certificates.
            return (record["free4"], record["reading"],
                    -len(record["witnesses"]),
                    -record["cost"], -record["error"], record["rank"])

        screened.sort(key=merit, reverse=True)
        current = screened[0]
        keepers = screened[:4]

        for step in range(24):
            words = list(current["witnesses"].values())
            if words:
                witness = rng.choice(words)
                implicated = sorted({c // q for c in bits(witness)})
                j = rng.choice(implicated)
            else:
                j = rng.randrange(3)
            proposals = []
            for move in range(2):
                ids = list(current["ids"])
                if move == 0:
                    ids[j] = central_variant(
                        ids[j], 1 + (step + model + cohort) % 2)
                else:
                    ids[j] = rng.choice(eligible[j])
                proposed = assemble(tuple(ids))
                if proposed is None:
                    continue
                if proposed["stats"][0] > current["stats"][0] + q:
                    continue

                d, word = kernel_probe(proposed["columns"], side_seed, 6)
                proposed["reading"], proposed["word"] = d, word
                proposed["free4"] = (
                    _pairs_no_kernel4(proposed["columns"]) is not None)
                # Old witnesses remain relevant only if their NEW
                # authoritative component syndrome is still zero.
                for old_word in words:
                    if syndrome(proposed["columns"], old_word) == 0:
                        key = orbit_key(old_word, side)
                        proposed["witnesses"][key] = old_word
                        if old_word.bit_count() < proposed["reading"]:
                            proposed["reading"] = old_word.bit_count()
                            proposed["word"] = old_word
                challenge(proposed, side, 6000)
                proposals.append(proposed)
            if not proposals:
                continue
            proposed = max(proposals, key=merit)
            if merit(proposed) >= merit(current):
                current = proposed
            elif (proposed["free4"] == current["free4"]
                  and proposed["reading"] == current["reading"]
                  and rng.random() < 0.10 * (1 - step / 24.0)):
                current = proposed
            keepers.extend(proposals)
            keepers.sort(key=merit, reverse=True)
            del keepers[5:]

        finalists = sorted(keepers + [current], key=merit, reverse=True)
        unique, final_seen = [], set()
        for record in finalists:
            sig = tuple(sorted(
                tuple(sorted(library[t]["ids"])) for t in record["ids"]))
            if sig in final_seen:
                continue
            final_seen.add(sig)
            unique.append(record)
            if len(unique) == 3:
                break

        for record in unique:
            # Fresh searches can only lower retained readings.
            for offset, trials in ((700001, 40), (1900003, 48)):
                d, word = kernel_probe(
                    record["columns"], side_seed + offset, trials)
                if d < record["reading"]:
                    record["reading"], record["word"] = d, word
                if word.bit_count() <= 12:
                    key = orbit_key(word, side)
                    record["witnesses"].setdefault(key, word)
            challenge(record, side, 36000)
            record["cells"] = [[
                [G.elements[g] for g in library[t]["ids"]]
                for t in record["ids"]
            ]]
            record.pop("completion", None)
        unique.sort(key=merit, reverse=True)
        return unique

    aa, bb = optimize("A"), optimize("B")
    best, best_key = None, None
    for a in aa:
        for b in bb:
            A, B = a["cells"], b["cells"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            assert n == 1200 and k >= 480
            in_band = 375*n <= 1000*k < 425*n
            shared_cells = len(
                {tuple(sorted(cell)) for cell in a["cells"][0]} &
                {tuple(sorted(cell)) for cell in b["cells"][0]})
            key = (in_band, a["free4"] and b["free4"],
                   min(a["reading"], b["reading"]),
                   a["reading"] + b["reading"],
                   -len(a["witnesses"]) - len(b["witnesses"]),
                   -shared_cells,
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
        "detected_minimum_orbits": (
            len(a["witnesses"]), len(b["witnesses"])),
        "repair_model": "central-sign and witness-directed cell replacement",
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete witness-repaired cubic cell without randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: The affine annihilator-plane bank produces verified k=720..750 improvements; all 51 distinct current codes instead had k=690, and all five measured distances remained six.
# REFUTED: Full-generating SL(2,5) cubic spectrum/descent selection yields five measured readings above ten; selected slot zero instead fell from twelve to ten, with 52 found minimum Z witnesses entirely in sector 1.
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
