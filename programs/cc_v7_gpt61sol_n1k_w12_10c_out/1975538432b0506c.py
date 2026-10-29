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
CONCEPT = ("C11 semidirect_2 C10, order 110, A=(1,3), B=(1,4): complete cubic extensions of compatible pairs sharing verified regular annihilator modules should find rank-80/90 A components missed by random plane triples, yielding new k>=720 candidates and improving cell 0.500 above 17.371 if distance-six readings survive. This is a falsifiable construction hypothesis, not a distance certificate.")
GROUP = ("Frozen affine group and order 110. The current table contains 54 distinct k=690 codes and five distance-six measurements; its finite plane searches supplied no additional accepted rank gain. F1024 only nominates pairs. Complete regular left-annihilator equations formed from scaffold-derived monomial matrices determine every extension, and native full ranks determine acceptance.")
SHAPES = ("Frozen A=(1,3), B=(1,4): n=1430, k>=660 and guaranteed rate 6/13, with q=110 below q_max=115. Unequal components have 330/440 variables and 110 checks, nominal rates 2/3 and 3/4. Parent minima were entirely sector 1, concentrated on Acol1/2 across Bcol0/1/2. Retain verified 690<=k<=750 for cell 0.500; seek rank gain without depending on a higher provisional distance reading.")
FILLING = ("Keep counts A=(3,3,3), B=(2,2,2,3), hence both CSS caps twelve. Nominate cubic pairs from complete representation-plane catalogs, reject actual pair kernels through four, and compute complete regular left-annihilators of rank-80/90 pairs. Enumerate every normalized cubic operator preserving each exact module, including extensions outside the nominating catalog. Screen complete triples through four; odd degree also excludes five. Work caps limit pair starts, not the extension census of an accepted pair. Twenty B cards cross four primitive-binomial initialization profiles with five binding noncommuting odd/even trinomial profiles generating the full affine group. Independently challenge finalists; verified line fallbacks receive the same new B model. Full ranks, kernel-four screens, weights and CSS remain gates. No finite search reading certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r050_affine11_13x14_regular_pair_extension_fullbridge_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 110

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Affine C11:C10 requires order 110")
    # (a,b) represents the affine map t -> 2^b*t + a over F_11.
    # Composition is associative; 2 has multiplicative order ten.
    powers = [pow(2, b, 11) for b in range(10)]
    els = [(a, b) for b in range(10) for a in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 10)

    def mul(g, h):
        return ((g[0] + powers[g[1] % 10] * h[0]) % 11,
                (g[1] + h[1]) % 10)

    def inv(g):
        b = (-g[1]) % 10
        return ((-powers[b] * g[0]) % 11, b)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0),
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
    """Twenty rank-gain cards, preferring common annihilator planes."""
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
            # Complete representation kernels nominate pair constraints.
            # Actual regular-module equations decide the extensions.
            values = [0]
            for v in basis:
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
            if len(supports) >= (2 if kind == "plane" else 3)
        ]
        catalog.sort(key=lambda item: (-len(item[1]), item[0]))
        catalogs[kind] = catalog
    selections = {"plane": 0, "line": 0}

    def constrained_cells(kind):
        """Spread fallback searches through the whole annihilator catalog."""
        catalog = catalogs[kind]
        if not catalog:
            return None, None
        turn = selections[kind]
        selections[kind] += 1
        # Interleave large classes with the rest instead of permanently
        # restricting every slot to the same sixteen constraints.
        if turn % 3 == 0:
            position = (card + 7 * turn + idx // 20) % min(32, len(catalog))
        else:
            position = (37 * card + 101 * turn + 53 * (idx // 20)) % len(catalog)
        functional, supports = catalog[position]
        options = [
            [G.elements[t] for t in support]
            for support in supports
        ]
        return options, functional

    # Binding full-affine trinomial bridges: distinct odd/even fibers.
    # A primitive odd multiplier and a nonzero commutator generate
    # the normal C11 and all of the quotient C10.
    spectra = ((1, 2), (1, 4), (3, 4), (3, 6), (7, 8))
    bin_profiles = ((1, 1, 9), (3, 3, 7),
                    (1, 7, 9), (3, 7, 9))

    def fresh_tri(attempt):
        b1, b2 = spectra[card % 5]
        choices = [
            ((a, b1), (c, b2))
            for a in range(11) for c in range(11)
            if G._key(G.mul((a, b1), (c, b2))) !=
               G._key(G.mul((c, b2), (a, b1)))
        ]
        g, h = rng.choice(choices)
        # Independently verify full generation using native operations.
        generators = (G.idx(g), G.idx(h))
        reached, todo = {G.idx(G.e)}, [G.idx(G.e)]
        for value in todo:
            for generator in generators:
                image = table[value][generator]
                if image not in reached:
                    reached.add(image)
                    todo.append(image)
        assert len(reached) == q
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
        return [[G.e, g] for g in fresh_bins(attempt)] + [fresh_tri(attempt)]

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

    # All cached matrices below are outputs of lifted_matrices. Joining
    # their disjoint column blocks is not an independently implemented lift.
    gain_cache = {"rows": {}, "columns": {}, "pairs": {}, "classes": None}

    def plane_pool():
        """Extend complete VERIFIED regular modules, not random triples.

        Pair-start budgets are finite. For each accepted pair, however,
        every normalized cubic preserving its entire regular annihilator
        is enumerated before component screening.
        """
        from itertools import combinations

        mask = (1 << q) - 1
        if "monomials" not in gain_cache:
            source, _, _ = lifted_matrices(
                probe, [[[g] for g in range(q)]], [[[]]])
            assert len(source) == q
            gain_cache["monomials"] = [
                [(row >> (g * q)) & mask for row in source]
                for g in range(q)
            ]
            gain_cache["visited_pairs"] = set()
            gain_cache["extension_banks"] = {}
            gain_cache["tested_triples"] = set()

        monomials = gain_cache["monomials"]

        def positions(word):
            while word:
                bit = word & -word
                yield bit.bit_length() - 1
                word ^= bit

        def selected_xor(rows, selector):
            answer = 0
            for r in positions(selector):
                answer ^= rows[r]
            return answer

        def dependencies(rows):
            """Exact left kernel in original regular check coordinates."""
            pivots, basis = {}, []
            for r, source in enumerate(rows):
                word = 1 << r
                while source:
                    lead = source.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (source, word)
                        break
                    source ^= old[0]
                    word ^= old[1]
                if not source:
                    basis.append(word)
            return basis

        def cell_columns(support):
            if support not in gain_cache["columns"]:
                rows = [
                    monomials[support[0]][r] ^
                    monomials[support[1]][r] ^
                    monomials[support[2]][r]
                    for r in range(q)
                ]
                columns = _syndrome_columns(rows, q)
                assert all(c.bit_count() == 3 for c in columns)
                gain_cache["rows"][support] = rows
                gain_cache["columns"][support] = (
                    columns if _pairs_no_kernel4(columns) is not None
                    else None)
            return gain_cache["columns"][support]

        def canonical(support):
            # Right-column gauges preserve the complete left kernel.
            return min(
                tuple(sorted(table[g][inverses[root]]
                             for g in support))
                for root in support)

        def extensions(module):
            """All normalized cubics annihilated by EVERY module vector."""
            key = tuple(module)
            banks = gain_cache["extension_banks"]
            if key in banks:
                return banks[key]

            # Each image is a concatenation of actual regular syndromes.
            # No independently implemented lift or field representation
            # enters these authoritative preservation equations.
            images = []
            for rows in monomials:
                image = 0
                for t, functional in enumerate(module):
                    image |= selected_xor(rows, functional) << (t * q)
                images.append(image)

            by_image = {}
            for g, image in enumerate(images):
                by_image.setdefault(image, []).append(g)

            candidates = set()
            for a in range(q):
                if a == ident:
                    continue
                wanted = images[ident] ^ images[a]
                for b in by_image.get(wanted, ()):
                    if b == ident or b == a:
                        continue
                    support = tuple(sorted((ident, a, b)))
                    candidates.add(canonical(support))

            answer = sorted(candidates)
            for support in answer:
                rows = [
                    monomials[support[0]][r] ^
                    monomials[support[1]][r] ^
                    monomials[support[2]][r]
                    for r in range(q)
                ]
                assert all(selected_xor(rows, functional) == 0
                           for functional in module)
            banks[key] = answer
            return answer

        # Collapse identical proposal catalogs, not submitted matrices.
        unique = {}
        for functional, supports in catalogs["plane"]:
            signature = tuple(sorted(set(supports)))
            if len(signature) >= 2:
                unique.setdefault(signature, functional)
        classes = [(functional, list(signature))
                   for signature, functional in unique.items()]
        classes.sort(key=lambda item: (-len(item[1]), item[0]))
        if not classes:
            return []

        turn = selections["plane"]
        selections["plane"] += 1
        offset = (37 * card + 101 * turn + 53 * (idx // 20)) % len(classes)
        classes = classes[offset:] + classes[:offset]

        records = []
        work = 0
        pair_cap = 1536 + 256 * (card % 3)
        exhausted = False
        for functional, supports in classes:
            # This only changes finite search priorities; no symmetry
            # image is submitted as an additional design.
            rng.shuffle(supports)
            for left, right in combinations(supports, 2):
                pair = tuple(sorted((left, right)))
                if pair in gain_cache["visited_pairs"]:
                    continue
                gain_cache["visited_pairs"].add(pair)
                work += 1
                if work > pair_cap:
                    exhausted = True
                    break

                lc, rc = cell_columns(left), cell_columns(right)
                if lc is None or rc is None:
                    continue
                if _pairs_no_kernel4(lc + rc) is None:
                    continue

                pair_rows = [
                    gain_cache["rows"][left][r] |
                    (gain_cache["rows"][right][r] << q)
                    for r in range(q)
                ]
                module = dependencies(pair_rows)
                rank = q - len(module)
                if rank not in (80, 90):
                    continue
                assert rank == gf2_rank(pair_rows)
                assert all(selected_xor(pair_rows, word) == 0
                           for word in module)

                for third in extensions(module):
                    if third in pair:
                        continue
                    signature = tuple(sorted((left, right, third)))
                    if signature in gain_cache["tested_triples"]:
                        continue
                    gain_cache["tested_triples"].add(signature)
                    tc = cell_columns(third)
                    if tc is None:
                        continue
                    columns = [
                        column for support in signature
                        for column in cell_columns(support)
                    ]
                    if _pairs_no_kernel4(columns) is None:
                        continue

                    rows = [
                        sum(gain_cache["rows"][support][r] << (j * q)
                            for j, support in enumerate(signature))
                        for r in range(q)
                    ]
                    # The pair bounds the intersection from above;
                    # preservation of its entire module bounds it below.
                    assert gf2_rank(rows) == rank
                    assert all(c.bit_count() == 3 for c in columns)
                    cycles = _four_cycles(rows)
                    reading = _component_reading(columns, rng, 12)
                    records.append({
                        "cells": [[G.elements[g] for g in support]
                                  for support in signature],
                        "columns": columns, "rank": rank,
                        "cycles": cycles, "reading": reading,
                        "trial": work, "functional": functional,
                        "regular_module_dimension": len(module),
                        "extension_catalog_size": len(extensions(module)),
                    })
                    records.sort(key=lambda rec: (
                        -rec["rank"], rec["reading"], -rec["cycles"]),
                        reverse=True)
                    del records[8:]

                if len(records) >= 6 and records[2]["rank"] == 80:
                    exhausted = True
                    break
            if exhausted:
                break

        finalists = records[:3]
        for rec in finalists:
            rec["reading"] = min(
                rec["reading"],
                _component_reading(rec["columns"], rng, 48))
        finalists.sort(key=lambda rec: (
            -rec["rank"], rec["reading"], -rec["cycles"]), reverse=True)
        return finalists

    def _retired_plane_pool():
        """Search plane orbits by exact cell/pair/triple compatibility.

        The old search prioritized catalog cardinality and considered
        only sixteen functionals. A large catalog can contain many
        mutually incompatible operators. Here compatibility, actual
        rank and complete triple screening decide acceptance.
        """
        if gain_cache["classes"] is None:
            action_cache = {}

            def image(g, value):
                key = (g, value)
                if key not in action_cache:
                    answer = 0
                    word = value
                    while word:
                        bit = word & -word
                        answer ^= adjoints[g][bit.bit_length() - 1]
                        word ^= bit
                    action_cache[key] = answer
                return action_cache[key]

            orbits = {}
            for functional, supports in catalogs["plane"]:
                orbit = min(
                    tuple(sorted(image(g, value) for value in functional))
                    for g in range(q))
                orbits.setdefault(orbit, []).append((functional, supports))

            # Each entry is an orbit of search constraints. We still draw
            # genuinely different operator triples and B matrices; no
            # symmetry images are returned as additional code designs.
            classes = []
            for orbit in sorted(orbits):
                entries = orbits[orbit]
                largest = max(len(entry[1]) for entry in entries)
                entries = [entry for entry in entries
                           if len(entry[1]) == largest]
                classes.append(rng.choice(entries))
            gain_cache["classes"] = classes

        classes = list(gain_cache["classes"])
        if not classes:
            return []
        turn = selections["plane"]
        selections["plane"] += 1
        rng.shuffle(classes)
        offset = (card + 17 * turn + 11 * (idx // 20)) % len(classes)
        classes = classes[offset:] + classes[:offset]

        def cell_columns(support):
            cache = gain_cache["columns"]
            if support not in cache:
                # A single left-action cell, isolated only for screening.
                rows, _, width = lifted_matrices(
                    probe, [[list(support)]], [[[]]])
                assert width == 2 * q
                assert all(row >> q == 0 for row in rows)
                cols = _syndrome_columns(rows, q)
                assert all(column.bit_count() == 3 for column in cols)
                gain_cache["rows"][support] = rows
                cache[support] = (
                    cols if _pairs_no_kernel4(cols) is not None else None)
            return cache[support]

        def compatible(left, right):
            pair = tuple(sorted((left, right)))
            cache = gain_cache["pairs"]
            if pair not in cache:
                lc, rc = cell_columns(left), cell_columns(right)
                cache[pair] = (
                    lc is not None and rc is not None and
                    _pairs_no_kernel4(lc + rc) is not None)
            return cache[pair]

        records = []
        signatures = set()
        # Round-robin visits prevent one large, incompatible catalog
        # from consuming the complete search budget.
        budget = 384 + 64 * (card % 3)
        for attempt in range(budget):
            functional, supports = classes[attempt % len(classes)]
            signature = tuple(sorted(rng.sample(supports, 3)))
            if signature in signatures:
                continue
            signatures.add(signature)
            if not all(
                    compatible(signature[i], signature[j])
                    for i in range(3) for j in range(i)):
                continue

            columns = []
            for support in signature:
                columns.extend(cell_columns(support))
            # Pair compatibility alone misses four-words touching all
            # three blocks. Test the complete component as well.
            if _pairs_no_kernel4(columns) is None:
                continue

            # Every column has odd weight three, so a zero syndrome
            # cannot have odd support weight. The exhaustive <=4 test
            # therefore also excludes all component words of weight five.
            rows = [
                sum(gain_cache["rows"][support][r] << (j * q)
                    for j, support in enumerate(signature))
                for r in range(q)
            ]
            rank = gf2_rank(rows)
            if rank not in (80, 90):
                continue
            cells = [[G.elements[t] for t in support]
                     for support in signature]
            cycles = _four_cycles(rows)
            reading = _component_reading(columns, rng, 12)
            records.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": cycles, "reading": reading,
                "trial": attempt, "functional": functional,
            })
            records.sort(key=lambda record: (
                -record["rank"], record["reading"], -record["cycles"]),
                reverse=True)
            del records[8:]
            if (len(records) >= 6 and
                    all(record["rank"] == 80 for record in records[:3])):
                break

        finalists = records[:3]
        for record in finalists:
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], rng, 48))
        finalists.sort(key=lambda record: (
            -record["rank"], record["reading"], -record["cycles"]),
            reverse=True)
        return finalists

    # B is independent of the annihilator constraint and can be reused
    # while several different A annihilator classes are examined.
    bb = pool("B", None)
    if not bb:
        bb = pool("B", None)
    if not bb:
        raise ValueError("new B card failed component kernel-five screening")

    def build(kind):
        if kind == "plane":
            aa = plane_pool()
            functional = None
        else:
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
                "functional": a.get("functional", functional),
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
            }
        return None

    # Two broad compatibility searches replace four narrow catalog draws.
    # A failed gain search still has independently generated, new B labels
    # and broadly selected line fallbacks; no archived fillings are copied.
    for _ in range(2):
        answer = build("plane")
        if answer is not None:
            return answer
    for _ in range(8):
        answer = build("line")
        if answer is not None:
            return answer
    raise ValueError("no compatibility-screened annihilator design passed ranks")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting Z27:Z3 binomials, a trinomial and kernel-four rejection suppress weight-five logicals: all five measured parent codes still returned five, entirely in sector 1.
# REFUTED: The A4 x C8 odd-central three-cycle binomial sampler supplies feasible four-cycle-free sets reliably; all sixty evaluated slots failed that labeling gate before producing a code.
# REFUTED: Primitive order-ten binomials, noncommuting trinomials and component kernel-five selection suppress the affine bed's weight-six logical family; all five measured parent codes still returned six, with thousands of sector-1 witnesses of both types.
# REFUTED: The shared trace-annihilator rank gain also raises measured distance above six; the current parent's five measured codes all remained at six, with minimum witnesses entirely in sector 1.
# REFUTED: Preferring the largest sixteen annihilator-plane catalogs produces k=720..750 survivors under the component gates; every distinct code in the current parent's table instead had k=690, and all five measured distances remained six.
# REFUTED: Finite plane-orbit compatibility searches reliably produce accepted rank gains beyond k=690; all 54 distinct current codes instead had k=690, and all five measured distances remained six. This does not establish that a complete regular-module extension census is empty.
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
