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
CONCEPT = ("C11 semidirect_2 C10, order 110, A=(1,3), B=(1,4): a quadratic operator joined to two cubics sharing its actual regular left-annihilator intersection should produce k>690 without component words through weight five and improve five-code fitness above 17.371 in cell 0.500. This mixed-degree mechanism replaces cubic-only plane completion; bounded search failure proves no impossibility.")
GROUP = ("Keep the exact affine group and order 110 unchanged. The current table contains 54 distinct k=690 codes and five distance-six readings, with no additional accepted dimension gain. Quadratic ratios of complement order five or ten have different regular kernels to the cubic-only bank. Their intersections are measured using authoritative scaffold monomials, never inferred from F1024 dimensions.")
SHAPES = ("Keep exact A=(1,3), B=(1,4): n=1430, k>=660, guaranteed rate 6/13 and maximum admissible lift order 115. Unequal components have 330 and 440 variables with 110 checks each. Parent minima occupied only sector one, particularly Acol1/2 across Bcol0/1/2. Require actual 690<=k<=750, inside cell 0.500. At reading six, k=720 would score 18.126; no distance increase is assumed.")
FILLING = ("Prefer A counts (2,3,3), retaining B=(2,2,2,3): HX<=11 and HZ<=12. Twenty cards cross four quadratic quotient coordinates with five B-trinomial prescriptions and geometry objectives. Enumerate canonical cubics, bucket their exact intersections with the quadratic left kernel, and join distinct operators sharing an identical actual subspace. Complete mixed components receive exact kernel-five screening, since odd-degree parity no longer excludes five. B preferentially uses one order-eleven translation binomial and two primitive-complement binomials; verified shortest backbone cycles and complete component gates reject words below six. Independently generated primitive backbones and fresh cubic-only A alternatives preserve feasibility. Original-span six-word embeddings, full ranks, assembly agreement, kernel-four exclusion, weights and CSS remain binding; no finite reading certifies quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r050_affine11_13x14_quadratic_cubic_intersections_translation_spine_v8"

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
    """Systematic annihilator joins paired with twenty new B prescriptions."""
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

    # Relative geometry is coupled to THIS backbone, rather than drawing
    # the trinomial independently. No group image is submitted as a card.
    spectra = ((0, 2), (0, 4), (1, 2), (1, 4), (2, 3))
    bin_profiles = ((1, 1, 1), (1, 3, 7),
                    (3, 3, 7), (1, 7, 9))

    def commutator_id(a, b):
        return table[table[a][b]][inverses[table[b][a]]]

    def fresh_tri(gs):
        b1, b2 = spectra[card % 5]
        reference = commutator_id(G.idx(gs[0]), G.idx(gs[1]))
        reference_a, reference_b = G.elements[reference]
        assert reference_b == 0 and reference_a != 0
        ratio = ((1, 2, 4, 7)[card // 5] *
                 (1, 3, 5)[(idx // 20) % 3]) % 11
        target = (ratio * reference_a) % 11

        # Exhaust the small affine-coordinate fiber instead of making a
        # low-probability rejection draw a slot-feasibility gate.
        choices = []
        for a in range(11):
            for c in range(11):
                g, h = (a, b1), (c, b2)
                ids = (G.idx(G.e), G.idx(g), G.idx(h))
                if len(set(ids)) != 3:
                    continue
                comm = commutator_id(ids[1], ids[2])
                if G.elements[comm] != (target, 0):
                    continue
                left = {table[inverses[u]][v]
                        for u in ids for v in ids if u != v}
                right = {table[u][inverses[v]]
                         for u in ids for v in ids if u != v}
                if len(left) == len(right) == 6:
                    choices.append([G.e, g, h])
        if not choices:
            raise ValueError("backbone-coupled Sidon trinomial unavailable")
        return list(rng.choice(choices))

    def positions(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def backbone_word(rows, columns):
        """Find a genuine shortest graph-cycle word on verified columns.

        All backbone columns have degree two. Every check root is
        examined; no full-code symmetry or independently written lift
        is used. Paths are variable-support bitsets.
        """
        adjacency = [[] for _ in rows]
        for v, column in enumerate(columns):
            ends = list(positions(column))
            if len(ends) != 2:
                return None
            a, b = ends
            adjacency[a].append((b, v))
            adjacency[b].append((a, v))

        shortest = None
        for root in range(len(rows)):
            distance = [-1] * len(rows)
            paths = [0] * len(rows)
            parents = [-1] * len(rows)
            distance[root] = 0
            pending = [root]
            for a in pending:
                for b, edge in adjacency[a]:
                    if distance[b] < 0:
                        distance[b] = distance[a] + 1
                        paths[b] = paths[a] ^ (1 << edge)
                        parents[b] = edge
                        pending.append(b)
                    elif edge != parents[a] and edge != parents[b]:
                        word = paths[a] ^ paths[b] ^ (1 << edge)
                        if word and (shortest is None or
                                     word.bit_count() < shortest.bit_count()):
                            shortest = word
                            if word.bit_count() <= 4:
                                return word
        if shortest is not None:
            value = 0
            for v in positions(shortest):
                value ^= columns[v]
            assert value == 0
        return shortest

    # Independent whole backbones, not a list of automorphism images.
    # Their individual regular symmetries are not counted as code diversity:
    # complete A triples and B trinomials are selected independently.
    spine_bank, spine_words, spine_seen = [], {}, set()
    for attempt in range(192):
        profile = bin_profiles[card // 5]
        if profile is None or attempt >= 96:
            gs = _short_cycle_free_bins(
                rng, G, 3, card + attempt + idx // 20)
        else:
            gs = [
                (rng.randrange(11),
                 b if rng.randrange(2) else (-b) % 10)
                for b in profile
            ]
        key = tuple(G.idx(g) for g in gs)
        if key in spine_seen:
            continue
        spine_seen.add(key)
        rows = component([[G.e, g] for g in gs], "B")
        columns = _syndrome_columns(rows, 3 * q)
        if _pairs_no_kernel4(columns) is None:
            continue
        word = backbone_word(rows, columns)
        if word is None or word.bit_count() != 6:
            continue
        assert all((word & row).bit_count() % 2 == 0 for row in rows)
        spine_bank.append(gs)
        spine_words[key] = word
        if len(spine_bank) >= 12:
            break
    if not spine_bank:
        raise ValueError("no independently verified six-cycle B backbone")

    def fresh_bins(attempt):
        return list(rng.choice(spine_bank))

    def fresh(side, options, attempt):
        if side == "A":
            return [list(cell) for cell in rng.sample(options, 3)]
        gs = fresh_bins(attempt)
        return [[G.e, g] for g in gs] + [fresh_tri(gs)]

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
            preference = cycles
            if side == "B":
                peak = max((row & other).bit_count()
                           for i, row in enumerate(rows)
                           for other in rows[:i])
                tri_rank = gf2_rank([row >> (3 * q) for row in rows])
                policy = card // 5
                if policy == 1:
                    preference = 64 * peak + cycles // q
                elif policy == 2:
                    preference = 64 * (q - tri_rank) + cycles // q
                elif policy == 3:
                    preference = cycles + 16 * (q - tri_rank) + q * peak
            records.append((rank, preference, cycles, trial, cells, rows))

        # Rank gain matters only after actual short-word exclusion.
        # Cards deliberately optimize different native B geometry.
        records.sort(key=lambda record: record[:4])
        survivors = []
        for rank, preference, cycles, trial, cells, rows in records:
            floor, columns = _component_floor(rows, width, q)
            if floor != 6:
                continue
            reading = _component_reading(columns, rng, 12)
            word = None
            if side == "B":
                key = tuple(G.idx(cell[1]) for cell in cells[:3])
                word = spine_words[key]
                value = 0
                for v in positions(word):
                    value ^= columns[v]
                assert value == 0 and word.bit_count() == 6
                reading = min(reading, word.bit_count())
            survivors.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": cycles, "reading": reading, "trial": trial,
                "preference": preference, "backbone_word": word,
            })
            if len(survivors) >= 8:
                break
        survivors.sort(key=lambda record: (
            -record["rank"], record["reading"],
            -record["preference"], -record["cycles"]), reverse=True)
        survivors = survivors[:3]
        for record in survivors:
            record["reading"] = min(
                record["reading"],
                _component_reading(record["columns"], rng, 48))
        survivors.sort(key=lambda record: (
            -record["rank"], record["reading"],
            -record["preference"], -record["cycles"]), reverse=True)
        return survivors

    # B is independent of the annihilator constraint and can be reused
    # while several different A annihilator classes are examined.
    bb = pool("B", None)
    if not bb:
        bb = pool("B", None)
    if not bb:
        raise ValueError("new B card failed component kernel-five screening")

    a_banks = {}
    regular_cache = {}
    intersection_cache = {}
    compatibility_cache = {}
    mask = (1 << q) - 1

    # Obtain authoritative left-action monomials once. All later
    # component assembly is GF2 linearity of these exact probe outputs.
    source, _, _ = lifted_matrices(
        probe, [[[g] for g in range(q)]], [[[]]])
    assert len(source) == q
    monomials = [
        [(row >> (g * q)) & mask for row in source]
        for g in range(q)
    ]

    def bit_positions(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def xor_selected(words, selector):
        answer = 0
        for position in bit_positions(selector):
            answer ^= words[position]
        return answer

    def dependency_basis(vectors):
        """Dependencies in original input coordinates, over GF2."""
        pivots, basis = {}, []
        for i, value in enumerate(vectors):
            word = 1 << i
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

    def regular_cell(support):
        if support not in regular_cache:
            rows = [
                monomials[support[0]][r] ^
                monomials[support[1]][r] ^
                monomials[support[2]][r]
                for r in range(q)
            ]
            columns = _syndrome_columns(rows, q)
            left = dependency_basis(rows)
            assert len(left) == q - gf2_rank(rows)
            assert all(xor_selected(rows, word) == 0 for word in left)
            regular_cache[support] = {
                "rows": rows, "columns": columns, "left": left,
                "short_ok": None,
            }
        return regular_cache[support]

    def intersect_with_cell(basis, rec):
        """Intersect a regular left subspace with ker(rec['rows']^T).

        No representation dimension or independently implemented lift
        enters this computation.
        """
        if not basis:
            return []
        images = [xor_selected(rec["rows"], word) for word in basis]
        return [
            xor_selected(basis, dependency)
            for dependency in dependency_basis(images)
        ]

    def pair_basis(left, right):
        key = tuple(sorted((left, right)))
        if key not in intersection_cache:
            first, second = regular_cell(left), regular_cell(right)
            if len(first["left"]) > len(second["left"]):
                first, second = second, first
            intersection_cache[key] = intersect_with_cell(
                first["left"], second)
        return intersection_cache[key]

    def individually_legal(support):
        rec = regular_cell(support)
        if rec["short_ok"] is None:
            rec["short_ok"] = (
                _pairs_no_kernel4(rec["columns"]) is not None)
        return rec["short_ok"]

    def compatible(left, right):
        key = tuple(sorted((left, right)))
        if key not in compatibility_cache:
            compatibility_cache[key] = (
                individually_legal(left) and individually_legal(right)
                and _pairs_no_kernel4(
                    regular_cell(left)["columns"] +
                    regular_cell(right)["columns"]) is not None)
        return compatibility_cache[key]

    def completed_regular_planes():
        """Complete actual pair annihilators by an exact cancellation hash.

        Operator enumeration is native, not restricted to F1024 catalogs.
        Work caps limit construction effort, never imply distance or
        algebraic impossibility. Every retained syndrome uses the lift.
        """
        def canonical_operator(support):
            return min(
                tuple(sorted(table[g][inverses[root]] for g in support))
                for root in support)

        def canonical_space(basis):
            # Reduced GF2 row basis: equal subspaces receive equal keys.
            pivots = {}
            for value in basis:
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = value
                        break
                    value ^= old
            leads = sorted(pivots)
            for at, lead in enumerate(leads):
                bit = 1 << lead
                for higher in leads[at + 1:]:
                    if pivots[higher] & bit:
                        pivots[higher] ^= pivots[lead]
            return tuple(pivots[lead] for lead in leads)

        eligible = []
        # Anchoring a term is a right-column gauge. Enumerate its
        # canonical representatives directly using native operations.
        for a in range(q):
            if a == ident:
                continue
            for b in range(a):
                if b == ident:
                    continue
                support = tuple(sorted((ident, a, b)))
                if support != canonical_operator(support):
                    continue
                rec = regular_cell(support)
                if len(rec["left"]) < 20:
                    continue
                if individually_legal(support):
                    eligible.append(support)

        rng.shuffle(eligible)
        policy = card // 5
        if policy == 0:
            eligible.sort(key=lambda s: len(regular_cell(s)["left"]))
        elif policy == 1:
            eligible.sort(key=lambda s: -len(regular_cell(s)["left"]))
        elif policy == 2:
            eligible.sort(key=lambda s: _four_cycles(
                regular_cell(s)["rows"]))

        records, signatures, spaces = [], set(), set()
        pair_work = 0
        pair_cap, space_cap = 180000, 192

        for i, first in enumerate(eligible):
            for second in eligible[:i]:
                if pair_work >= pair_cap or len(spaces) >= space_cap:
                    break
                pair_work += 1
                common = pair_basis(first, second)
                if len(common) < 20 or not compatible(first, second):
                    continue
                space = canonical_space(common)
                if space in spaces:
                    continue
                spaces.add(space)

                # For each monomial M_g, concatenate v^T M_g over a
                # basis v of the ACTUAL regular left intersection.
                # A third cubic annihilates that entire space exactly
                # when its three concatenated images XOR to zero.
                images = []
                by_image = {}
                for g in range(q):
                    image = 0
                    for t, word in enumerate(space):
                        image |= xor_selected(
                            monomials[g], word) << (t * q)
                    images.append(image)
                    by_image.setdefault(image, []).append(g)

                candidates = set()
                anchor_image = images[ident]
                for a in range(q):
                    if a == ident:
                        continue
                    wanted = anchor_image ^ images[a]
                    for b in by_image.get(wanted, ()):
                        if b == ident or b == a:
                            continue
                        candidates.add(canonical_operator(
                            (ident, a, b)))

                for third in sorted(candidates):
                    if third in (first, second):
                        continue
                    signature = tuple(sorted((first, second, third)))
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    if not individually_legal(third):
                        continue
                    if not (compatible(first, third) and
                            compatible(second, third)):
                        continue

                    complete = intersect_with_cell(
                        common, regular_cell(third))
                    rank = q - len(complete)
                    if rank not in (80, 90):
                        continue
                    columns = [
                        column for support in signature
                        for column in regular_cell(support)["columns"]
                    ]
                    if _pairs_no_kernel4(columns) is None:
                        continue
                    rows = [
                        sum(regular_cell(support)["rows"][r] << (j * q)
                            for j, support in enumerate(signature))
                        for r in range(q)
                    ]
                    assert gf2_rank(rows) == rank
                    assert all(column.bit_count() == 3
                               for column in columns)
                    # Odd column degree and exact <=4 exclusion also
                    # exclude every odd-weight component word through five.
                    records.append({
                        "cells": [
                            [G.elements[g] for g in support]
                            for support in signature
                        ],
                        "columns": columns, "rank": rank,
                        "cycles": _four_cycles(rows),
                        "functional": ("native-completion", space),
                        "dimension": len(complete),
                        "signature": signature,
                    })
                    if len(records) >= 48:
                        return records
            if pair_work >= pair_cap or len(spaces) >= space_cap:
                break
        return records

    def systematic_A(kind):
        """Native plane completion with independent catalog-line alternatives."""
        if kind in a_banks:
            return a_banks[kind]

        if kind == "plane":
            records = completed_regular_planes()
            records.sort(key=lambda rec: (
                rec["rank"], rec["cycles"], rec["signature"]))
            if records:
                best_rank = records[0]["rank"]
                near = [rec for rec in records
                        if rec["rank"] == best_rank][:24]
                near = rng.sample(near, min(8, len(near)))
                for rec in near:
                    rec["reading"] = _component_reading(
                        rec["columns"], rng, 16)
                near.sort(key=lambda rec: (
                    rec["reading"], -rec["cycles"]), reverse=True)
                near = near[:3]
                for rec in near:
                    rec["reading"] = min(
                        rec["reading"],
                        _component_reading(rec["columns"], rng, 64))
                near.sort(key=lambda rec: (
                    rec["reading"], -rec["cycles"]), reverse=True)
                records = near
            a_banks[kind] = records
            return records

        # A dimension-twenty intersection also includes any dimension-
        # thirty survivor. Do not consume a shared budget on thirty first.
        targets = (20,) if kind == "plane" else (10,)
        allowed = {80, 90} if kind == "plane" else {80, 90, 100}
        records, tested = [], set()

        if kind == "plane":
            # A short word inside one cell cannot disappear when other
            # column blocks are appended. This unary rejection is exact.
            # Different functionals can induce the same viable operator
            # catalog; collapse those search duplicates before pair work.
            legal = {}
            unique = {}
            for functional, proposed in catalogs["plane"]:
                options = []
                for support in set(proposed):
                    if support not in legal:
                        legal[support] = (
                            len(regular_cell(support)["left"]) >= 20
                            and individually_legal(support))
                    if legal[support]:
                        options.append(support)
                signature = tuple(sorted(options))
                if len(signature) >= 3:
                    unique.setdefault(signature, functional)
            ordered = [(functional, list(options))
                       for options, functional in unique.items()]
            rng.shuffle(ordered)
            policy = card // 5
            if policy == 0:
                ordered.sort(key=lambda item: len(item[1]))
            elif policy == 1:
                ordered.sort(key=lambda item: -len(item[1]))
            elif policy == 3:
                ordered.sort(key=lambda item: (
                    -sum(len(regular_cell(s)["left"]) for s in item[1]),
                    len(item[1])))
        else:
            ordered = list(catalogs["line"])
            rng.shuffle(ordered)

        if ordered:
            offset = (17 * card + 29 * (idx // 20)) % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]

        pair_cap = 120000 if kind == "plane" else 30000
        triple_cap = 12000 if kind == "plane" else 1500
        pair_work = triple_work = 0

        for target in targets:
            for functional, proposed in ordered:
                options = [
                    support for support in set(proposed)
                    if len(regular_cell(support)["left"]) >= target
                    and individually_legal(support)
                ]
                if len(options) < 3:
                    continue
                # Independent triples, not right-gauge copies; shuffle
                # removes the parent's persistent lexicographic shortlist.
                options.sort()
                rng.shuffle(options)
                adjacent = [set() for _ in options]

                for i, left in enumerate(options):
                    for j, right in enumerate(options[:i]):
                        key = tuple(sorted((left, right)))
                        if key not in intersection_cache:
                            if pair_work >= pair_cap:
                                continue
                            pair_work += 1
                        common = pair_basis(left, right)
                        if len(common) < target:
                            continue
                        if not compatible(left, right):
                            continue
                        adjacent[i].add(j)
                        adjacent[j].add(i)

                for i in range(len(options)):
                    for j in sorted(v for v in adjacent[i] if v > i):
                        third = adjacent[i] & adjacent[j]
                        for k_ in sorted(v for v in third if v > j):
                            signature = tuple(sorted(
                                (options[i], options[j], options[k_])))
                            if signature in tested:
                                continue
                            if triple_work >= triple_cap:
                                break
                            tested.add(signature)
                            triple_work += 1
                            common = intersect_with_cell(
                                pair_basis(signature[0], signature[1]),
                                regular_cell(signature[2]))
                            rank = q - len(common)
                            if rank not in allowed or len(common) < target:
                                continue

                            columns = [
                                column for support in signature
                                for column in regular_cell(support)["columns"]
                            ]
                            # Odd column degree excludes odd-weight null
                            # words. Complete <=4 exclusion therefore
                            # excludes component kernels through five.
                            if _pairs_no_kernel4(columns) is None:
                                continue
                            rows = [
                                sum(regular_cell(support)["rows"][r] << (j_ * q)
                                    for j_, support in enumerate(signature))
                                for r in range(q)
                            ]
                            assert gf2_rank(rows) == rank
                            assert all(column.bit_count() == 3
                                       for column in columns)
                            records.append({
                                "cells": [
                                    [G.elements[g] for g in support]
                                    for support in signature
                                ],
                                "columns": columns, "rank": rank,
                                "cycles": _four_cycles(rows),
                                "functional": functional,
                                "dimension": len(common),
                                "signature": signature,
                            })
                        if triple_work >= triple_cap:
                            break
                    if triple_work >= triple_cap:
                        break
                if triple_work >= triple_cap or len(records) >= 64:
                    break
            if records or triple_work >= triple_cap:
                break

        records.sort(key=lambda rec: (
            rec["rank"], rec["cycles"], rec["signature"]))
        if records:
            best_rank = records[0]["rank"]
            near = [rec for rec in records if rec["rank"] == best_rank]
            # Preserve a cycle-quality preference without repeatedly
            # returning the same few canonical A triples in every slot.
            near = near[:128]
            near = rng.sample(near, min(16, len(near)))
            for rec in near:
                rec["reading"] = _component_reading(
                    rec["columns"], rng, 12)
            near.sort(key=lambda rec: (
                rec["reading"], -rec["cycles"]), reverse=True)
            near = near[:3]
            for rec in near:
                rec["reading"] = min(
                    rec["reading"],
                    _component_reading(rec["columns"], rng, 48))
            near.sort(key=lambda rec: (
                rec["reading"], -rec["cycles"]), reverse=True)
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

            # Challenge the ORIGINAL stabilizer quotient with the retained
            # component six-word. A commuting word is logical only if it
            # lies outside span(HX). Failure to find an embedding is not
            # a distance exclusion and does not reject this pairing.
            six_logical = None
            word = b.get("backbone_word")
            if word is not None:
                pivots = {}
                for source in hx:
                    while source:
                        lead = source.bit_length() - 1
                        old = pivots.get(lead)
                        if old is None:
                            pivots[lead] = source
                            break
                        source ^= old
                for j in range(3):
                    actual = word << (j * 4 * q)
                    assert actual.bit_count() == 6
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in hz)
                    reduced = actual
                    while reduced:
                        old = pivots.get(reduced.bit_length() - 1)
                        if old is None:
                            break
                        reduced ^= old
                    if reduced:
                        six_logical = actual
                        break

            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": (kind, card // 5, card % 5),
                "functional": a["functional"],
                "regular_annihilator_dimension": a["dimension"],
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
                "binding_backbone_weight": (
                    None if word is None else word.bit_count()),
                "verified_six_logical_weight": (
                    None if six_logical is None else six_logical.bit_count()),
                "B_geometry_preference": b["preference"],
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


# Retain the existing constructor only as a source of independently
# generated cubic A alternatives. Its B filling is never submitted here.
_cubic_reference_ctx = NEW_CTX


def NEW_CTX(rng, G, idx):
    """Mixed-degree intersection cards; all binary actions use the scaffold."""
    from itertools import combinations

    q = G.q
    assert q == 110 and A_SHAPE == (1, 3) and B_SHAPE == (1, 4)
    card, cohort = idx % 20, idx // 20
    quadratic_policy, model = divmod(card, 5)
    seed = rng.randrange(1 << 62) ^ (0x23A110B + 1000003 * idx)
    local = random.Random(seed)
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
    blocks = {"A": [], "B": []}
    # Small authoritative galleries, not an independently written lift.
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start + 16)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            for side, source in (("A", xx), ("B", zz)):
                block = [(row >> (j * q)) & mask for row in source]
                assert all(value.bit_count() == 1 for value in block)
                blocks[side].append(block)

    def positions(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def xor_selected(vectors, word):
        answer = 0
        for i in positions(word):
            answer ^= vectors[i]
        return answer

    def dependencies(vectors):
        pivots, basis = {}, []
        for i, source in enumerate(vectors):
            word = 1 << i
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

    def reduced_space(basis):
        pivots = {}
        for word in basis:
            while word:
                lead = word.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        leads = sorted(pivots)
        for at, lead in enumerate(leads):
            for higher in leads[at + 1:]:
                if (pivots[higher] >> lead) & 1:
                    pivots[higher] ^= pivots[lead]
        return tuple(pivots[lead] for lead in leads)

    def intersection(basis, rows):
        images = [xor_selected(rows, word) for word in basis]
        return reduced_space([
            xor_selected(basis, selector)
            for selector in dependencies(images)
        ])

    def component(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j * q)
        return rows

    def canonical_operator(support):
        # A right-column gauge preserves the actual left kernel.
        return min(tuple(sorted(table[g][inverse[root]]
                                for g in support))
                   for root in support)

    def commutator(a, b):
        return table[table[a][b]][inverse[table[b][a]]]

    def shortest_backbone(columns):
        """Exact shortest cycle of this verified degree-two check graph."""
        adjacency = [[] for _ in range(q)]
        for variable, column in enumerate(columns):
            ends = list(positions(column))
            if len(ends) != 2:
                return None
            a, b = ends
            adjacency[a].append((b, variable))
            adjacency[b].append((a, variable))
        best = None
        for root in range(q):
            depth = [-1] * q
            path = [0] * q
            parent = [-1] * q
            pending = [root]
            depth[root] = 0
            for a in pending:
                for b, edge in adjacency[a]:
                    if depth[b] < 0:
                        depth[b] = depth[a] + 1
                        path[b] = path[a] ^ (1 << edge)
                        parent[b] = edge
                        pending.append(b)
                    elif edge != parent[a] and edge != parent[b]:
                        word = path[a] ^ path[b] ^ (1 << edge)
                        if word and (best is None or
                                     word.bit_count() < best.bit_count()):
                            best = word
                            if best.bit_count() < 6:
                                return best
        if best is not None:
            assert xor_selected(columns, best) == 0
        return best

    # Native quadratic kernels nominate a genuinely different A density.
    # Each card fixes a quotient coordinate, not an automorphism image.
    qb = (1, 2, 3, 4)[quadratic_policy]
    ratio = G.idx((local.randrange(11), qb))
    quadratic = tuple(sorted((ident, ratio)))
    qr = component([quadratic], "A")
    qleft = dependencies(qr)
    assert len(qleft) == q - gf2_rank(qr)
    assert all(xor_selected(qr, word) == 0 for word in qleft)

    operators, spaces = {}, {}
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            support = tuple(sorted((ident, a, b)))
            if canonical_operator(support) != support:
                continue
            rows = component([support], "A")
            common = intersection(qleft, rows)
            if not 7 <= len(common) <= 30:
                continue
            columns = _syndrome_columns(rows, q)
            if _pairs_no_kernel4(columns) is None:
                continue
            operators[support] = (rows, columns)
            spaces.setdefault(common, []).append(support)

    # Equal reduced-space keys are exact regular intersections. Their
    # equality is a search constraint, not a Tanner-equivalence claim.
    groups = list(spaces.items())
    local.shuffle(groups)
    groups.sort(key=lambda item: -len(item[0]))
    plans, seen = [], set()
    for common, proposed in groups:
        options = list(proposed)
        local.shuffle(options)
        options = options[:64]
        for first, second in combinations(options, 2):
            signature = (quadratic,) + tuple(sorted((first, second)))
            if signature in seen:
                continue
            seen.add(signature)
            cells = [list(s) for s in signature]
            rows = component(cells, "A")
            rank = gf2_rank(rows)
            assert rank == q - len(common)
            columns = _syndrome_columns(rows, 3 * q)
            if _pairs_no_kernel4(columns) is None:
                continue
            cycles = _four_cycles(rows)
            # Five policies actually order different component geometry.
            peak = max((row & other).bit_count()
                       for i, row in enumerate(rows)
                       for other in rows[:i])
            quotient_mix = sum(
                len({G.elements[g][1] for g in support})
                for support in (first, second))
            geometry = (
                cycles, cycles + q * peak,
                cycles + q * (6 - quotient_mix),
                4 * cycles + q * peak,
                cycles + 2 * q * (6 - quotient_mix),
            )[model]
            plans.append(dict(
                cells=cells, rows=rows, columns=columns, rank=rank,
                defect=len(common), cycles=cycles, geometry=geometry,
                signature=signature, density="quadratic-cubic-cubic"))
            if len(plans) >= 256:
                break
        if len(plans) >= 256:
            break
    plans.sort(key=lambda rec: (-rec["defect"], rec["geometry"],
                                rec["signature"]))
    aa = []
    for rec in plans[:64]:
        # Mixed column parity invalidates the predecessor's odd-weight
        # shortcut. Use the complete regular-component weight-five gate.
        floor, columns = _component_floor(rec["rows"], 3 * q, q)
        if floor != 6:
            continue
        assert columns == rec["columns"]
        rec["reading"] = _component_reading(columns, local, 24)
        aa.append(rec)
        if len(aa) >= 6:
            break

    # Prefer a translation/primitive/primitive B backbone. Coordinates
    # three/seven avoid the immediate five-edge conjugation relation
    # supplied by action two; authoritative graph checks still decide.
    tri_profiles = ((0, 1), (0, 3), (0, 7), (0, 9), (2, 7))
    b1, b2 = tri_profiles[model]

    def trinomial(backbone):
        reference = commutator(backbone[0], backbone[1])
        x, y = G.elements[reference]
        if y or not x:
            return None
        target = ((1 + model + 2 * cohort) % 10 + 1) * x % 11
        choices = []
        for a in range(11):
            for b in range(11):
                u, v = G.idx((a, b1)), G.idx((b, b2))
                support = (ident, u, v)
                if len(set(support)) != 3:
                    continue
                if G.elements[commutator(u, v)] != (target, 0):
                    continue
                left = {table[inverse[g]][h]
                        for g in support for h in support if g != h}
                right = {table[g][inverse[h]]
                         for g in support for h in support if g != h}
                if len(left) == len(right) == 6:
                    choices.append(list(support))
        return local.choice(choices) if choices else None

    def b_pool(translation):
        bank, signatures = [], set()
        for trial in range(256):
            if translation:
                gs = [
                    G.idx((local.randrange(1, 11), 0)),
                    G.idx((local.randrange(11), local.choice((3, 7)))),
                    G.idx((local.randrange(11), local.choice((3, 7)))),
                ]
            else:
                # Fresh alternative backbones, never archived labels.
                gs = [G.idx(g) for g in _short_cycle_free_bins(
                    local, G, 3, card + trial + cohort)]
            cells = [[ident, g] for g in gs]
            spine_rows = component(cells, "B")
            spine_columns = _syndrome_columns(spine_rows, 3 * q)
            if _pairs_no_kernel4(spine_columns) is None:
                continue
            word = shortest_backbone(spine_columns)
            if word is None or word.bit_count() != 6:
                continue
            tri = trinomial(gs)
            if tri is None:
                continue
            cells.append(tri)
            signature = tuple(sorted(tuple(sorted(c)) for c in cells))
            if signature in signatures:
                continue
            signatures.add(signature)
            rows = component(cells, "B")
            if gf2_rank(rows) != q:
                continue
            bank.append(dict(
                cells=cells, rows=rows, rank=q, word=word,
                cycles=_four_cycles(rows), signature=signature,
                translation=translation))
        bank.sort(key=lambda rec: (rec["cycles"], rec["signature"]))
        accepted = []
        for rec in bank[:48]:
            floor, columns = _component_floor(rec["rows"], 4 * q, q)
            if floor != 6:
                continue
            assert xor_selected(columns, rec["word"]) == 0
            rec["columns"] = columns
            rec["reading"] = min(
                6, _component_reading(columns, local, 48))
            accepted.append(rec)
            if len(accepted) == 3:
                break
        return accepted

    bb = b_pool(True)
    if not bb:
        bb = b_pool(False)
    if not bb:
        raise ValueError("new translation/primitive B cards missed kernel-five gates")

    if not aa:
        # This fallback is a fresh A draw only. Its old B is discarded,
        # and the new B prescription above remains binding.
        old = _cubic_reference_ctx(local, G, idx)
        cells = [[G.idx(g) for g in cell] for cell in old["A"][0]]
        rows = component(cells, "A")
        floor, columns = _component_floor(rows, 3 * q, q)
        assert floor == 6
        rank = gf2_rank(rows)
        aa = [dict(
            cells=cells, rows=rows, columns=columns, rank=rank,
            defect=q - rank, cycles=_four_cycles(rows),
            geometry=_four_cycles(rows),
            reading=_component_reading(columns, local, 48),
            density="fresh-cubic-alternative")]

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

    survivors = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1430 and 690 <= k <= 750):
                continue
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert actual_a == a["rows"] and actual_b == b["rows"]
            sa = [[len(cell) for cell in A[0]]]
            sb = [[len(cell) for cell in B[0]]]
            assert sa in ([[2, 3, 3]], [[3, 3, 3]])
            assert sb == [[2, 2, 2, 3]]
            assert _grid_weights(sa, sb) <= W_MAX
            assert _no_dead_lines(sa, sb)
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert 19 * n <= 40 * k < 21 * n
            # Binding witness in the ORIGINAL stabilizer quotient.
            pivots = span(hx)
            logical = None
            for j in range(3):
                actual = b["word"] << (j * 4 * q)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in hz)
                if outside(actual, pivots):
                    logical = actual
                    break
            detected = min(6, a["reading"], b["reading"])
            merit = (k * detected * detected,
                     a["density"] == "quadratic-cubic-cubic",
                     b["translation"], -a["geometry"] - b["cycles"])
            survivors.append((merit, A, B, hx, hz, k, a, b, logical))

    survivors.sort(key=lambda item: item[0], reverse=True)
    for merit, A, B, hx, hz, k, a, b, logical in survivors:
        # No full-code nonabelian symmetry assumption enters these gates.
        if _pairs_no_kernel4(_syndrome_columns(hx, 1430)) is None:
            continue
        if _pairs_no_kernel4(_syndrome_columns(hz, 1430)) is None:
            continue
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, idx=idx, k=k,
            card=(qb, model, cohort), density=a["density"],
            actual_A_left_defect=a["defect"],
            component_ranks=(a["rank"], b["rank"]),
            component_readings=(a["reading"], b["reading"]),
            component_cycles=(a["cycles"], b["cycles"]),
            translation_backbone=b["translation"],
            verified_backbone_weight=b["word"].bit_count(),
            verified_original_span_logical_weight=(
                None if logical is None else logical.bit_count()),
            pairing_merit=merit, target_cell=0.500)
    raise ValueError("mixed-degree/new-backbone pairings missed full validity gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected mixed-degree/new-backbone cell."""
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
# REFUTED: Random plane-bank preference with rank-100 fallbacks reliably finds additional rank gain on this bed; every distinct current slot reported k=690 and k-kb=30, and all five measured distances remained six.
# REFUTED: Bounded systematic joins prioritized by F1024 common-space dimension produce accepted rank gains beyond k=690; all 22 distinct current codes instead had k=690, and all five measured distances remained six.
# REFUTED: The current canonical-join and B-card selection reliably supplies one inequivalent filling per slot; many slots were isomorphic duplicates and four slots failed labeling gates.
# REFUTED: The bounded regular-annihilator LINE search with repeated-even-fiber B cards produces additional accepted rank gain; all 51 distinct current codes instead had k=690, and all five measured distance readings remained six.
# REFUTED: Individually legal representation-plane catalogs, larger compatible-join budgets and primitive-odd B trinomials produce additional accepted rank gain; all 51 distinct current codes still had k=690, and all five measured distance readings remained six.
# REFUTED: Native monomial-signature completion of cubic-only regular intersections with coupled-commutator B cards produces additional accepted dimension gain; all 54 distinct current codes instead had k=690, and all five measured distance readings remained six.
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
