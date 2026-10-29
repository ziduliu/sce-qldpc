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
CONCEPT = ("Split-metacyclic C3 semidirect_-1 C16, order 48, A=(1,4), B=(2,7): connected bipartite B incidence components without four-cycles, paired with kernel-five-screened mixed-degree A components, should produce five new distance-six readings and fitness above 17.654 in cell 0.500. Five [1440,723,6] readings score 18.075; measurement can refute this target.")
GROUP = ("Retain the parent's split-metacyclic construction rule, resizing only m,r,s and q to 3,16,2 and 48. The parent repeatedly measured k=690 and distance six without the proposed additional rank gain. Here odd complement coordinates have order sixteen, avoiding involutory or order-four binomial ratios. This is an algebraically motivated new shape bed, not an assertion of distance transfer. Every binary screen uses scaffold-derived operators.")
SHAPES = ("Change A=(1,3), B=(1,4) to A=(1,4), B=(2,7). Factor thirty permits q=34..49; order 48 gives n=1440 and k>=720, guaranteed rate exactly 0.500. Unequal components have 192 variables/48 checks and 336 variables/96 checks. Parent minima occupied sector one, especially Acol1/2 across Bcol0/1/2. The additional B check row changes that backbone. Actual 720<=k and 40*k<21*n keep every submitted code in cell 0.500.")
FILLING = ("Use A counts (3,3,2,2) and B column degrees all two, with B row totals (7,7) or (8,6): HX=12 and HZ<=11. Twenty cards cross four internal-edge/bridge policies with five generating A trinomial prescriptions. Both action conventions come from authoritative monomial galleries. Optimize actual bipartite B graphs against duplicate edges and four-cycles; connectedness gives component rank 95. Screen A component kernels through five and independently lower witness readings. Recoverable B rooted-distance and closed-walk invariants assign disjoint slot buckets, preventing intra-program Tanner-isomorphic copies without claiming archive completeness. Final native ranks, component agreement, weights and CSS are binding. Component exclusion and finite readings do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r050_m3r16s2_14x27_bipartite_incidence_bucket_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # The same split-metacyclic construction rule as the parent.
    # 2^16=1 mod 3; the action has order two.
    if not valid_q(q):
        raise ValueError("C3 semidirect_-1 C16 requires order 48")
    m, r, s = 3, 16, 2
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

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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
    """Extract either shape's components from the authoritative full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            ar.append(sum(
                ((source >> (j * cb * q)) & mask) << (j * q)
                for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
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


def _retired_affine_homology_ctx(rng, G, idx):
    """Inactive order-110 predecessor; never called on the resized bed."""
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

    # Both action conventions are obtained from the authoritative helper.
    # Subsequent component assembly is only GF2 linear superposition.
    mask = (1 << q) - 1
    mono = [[[g] for g in range(q)]]
    raw_a, _, _ = lifted_matrices(probe, mono, [[[]]])
    _, raw_b, _ = lifted_matrices(probe, [[[]]], mono)
    blocks = {
        "A": [[(row >> (g * q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def component(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            for element in cell:
                action = blocks[side][G.idx(element)]
                for r, word in enumerate(action):
                    rows[r] ^= word << (j * q)
        assert all(row >> (len(cells) * q) == 0 for row in rows)
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

    # Twenty structural cards: four backbone prescriptions crossed with
    # five preferences for the ACTUAL regular rank of a trinomial block.
    # Rank preferences are objectives, not infeasible hard labeling gates.
    bin_profiles = ((1, 1, 1), (1, 1, 3), (1, 3, 7), (3, 3, 3))
    preferred_tri_rank = (110, 100, 90, 80, 70)[card % 5]
    tri_bank, tri_seen = [], set()
    eligible = [g for g in range(q) if g != ident]

    for trial in range(1024):
        a, b = rng.sample(eligible, 2)
        support = tuple(sorted((ident, a, b)))
        canonical = min(
            tuple(sorted(table[g][inverses[root]] for g in support))
            for root in support)
        if canonical in tri_seen:
            continue
        tri_seen.add(canonical)
        cell = [G.elements[g] for g in canonical]
        rows = component([cell], "B")
        tri_bank.append({
            "cell": cell,
            "rank": gf2_rank(rows),
            "cycles": _four_cycles(rows),
            "signature": canonical,
        })
    if not tri_bank:
        raise ValueError("empty independently generated trinomial bank")

    # A common left cell gauge on B changes only component column labels.
    # Bank diversity therefore comes from canonical relative supports,
    # not from translated copies of one operator.
    tri_bank.sort(key=lambda rec: (
        abs(rec["rank"] - preferred_tri_rank),
        rec["cycles"], rec["signature"]))

    def fresh_tri(attempt):
        # Initially favor the requested rank; progressively broaden the
        # bank so a difficult objective cannot destroy an otherwise legal
        # slot. Every chosen operator still has three distinct terms.
        width = min(len(tri_bank), 24 + 2 * attempt)
        rec = rng.choice(tri_bank[:width])
        return list(rec["cell"])

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
        rng.shuffle(ordered)
        plans = {}
        join_cap = 4096 if kind == "plane" else 3072

        for functional, supports in ordered:
            # Shuffle actual operator classes before applying a bounded
            # budget. This is not a relabeling of a submitted filling.
            supports = sorted(set(supports))
            rng.shuffle(supports)
            if kind == "plane":
                supports.sort(key=lambda support:
                              -annihilators[support].bit_count())
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
        proposals = list(plans.values())
        rng.shuffle(proposals)
        if kind == "plane":
            proposals.sort(key=lambda item: -item[0])
        # The line bank must also visit ordinary rank-100 alternatives,
        # rather than spending its whole budget on failed gain proposals.
        records = []
        check_cap = 768

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
            # A short classical word is only a proposal diagnostic.
            # The complete CSS homology gate below decides whether it
            # survives as a nontrivial quantum logical.
            short_free = _pairs_no_kernel4(columns) is not None
            records.append({
                "cells": cells, "columns": columns, "rank": rank,
                "cycles": _four_cycles(rows), "short_free": short_free,
                "functional": functional,
                "dimension": dimension,
                "signature": signature,
            })

        # Challenge several genuinely different triples, not gauge
        # images. A longer independent search can only lower a reading.
        records.sort(key=lambda record: (
            record["rank"], not record["short_free"],
            record["cycles"], record["signature"]))
        finalists = []
        for target_rank in (80, 90, 100):
            near = [record for record in records
                    if record["rank"] == target_rank]
            if not near:
                continue
            # Retain several independent joins at EVERY rank. Failed
            # gain proposals cannot suppress all feasible alternatives.
            near = near[:48]
            near = rng.sample(near, min(10, len(near)))
            for record in near:
                record["reading"] = _component_reading(
                    record["columns"], rng, 8)
            near.sort(key=lambda record: (
                record["short_free"], record["reading"],
                -record["cycles"]), reverse=True)
            near = near[:4]
            for record in near:
                record["reading"] = min(
                    record["reading"],
                    _component_reading(record["columns"], rng, 24))
            finalists.extend(near)
        records = finalists

        a_banks[kind] = records
        return records

    def stabilizer_pivots(rows):
        pivots = {}
        for source in rows:
            word = source
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def nontrivial(word, pivots):
        """Exact failure of membership in the stabilizer row space."""
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def logical_four_free(columns, pivots):
        """Exhaustively classify complete-code kernels through four.

        Equal-syndrome subsets differ by a kernel word. Keeping one
        representative is sufficient: if every difference from it is a
        stabilizer, every difference between representatives is too.
        No full-code group symmetry is assumed.
        """
        singles = {}
        for v, value in enumerate(columns):
            word = 1 << v
            if value == 0 and nontrivial(word, pivots):
                return False
            old = singles.get(value)
            if old is not None and nontrivial(word ^ old, pivots):
                return False
            singles.setdefault(value, word)

        pairs = {}
        for v, value in enumerate(columns):
            for u in range(v):
                syndrome = value ^ columns[u]
                word = (1 << v) | (1 << u)
                if syndrome == 0 and nontrivial(word, pivots):
                    return False
                old = singles.get(syndrome)
                if old is not None and nontrivial(word ^ old, pivots):
                    return False
                old = pairs.get(syndrome)
                if old is not None and nontrivial(word ^ old, pivots):
                    return False
                pairs.setdefault(syndrome, word)
        return True

    def logical_reading(columns, pivots, trials, diagnostic):
        """Budgeted genuine logical witnesses, classified modulo stabilizers.

        A large result means this CPU search missed lighter logicals;
        unlike logical_four_free, this is not an exhaustive exclusion.
        """
        order = list(range(len(columns)))
        best = len(columns) + 1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best and nontrivial(word, pivots):
                best = word.bit_count()

        for _ in range(trials):
            diagnostic.shuffle(order)
            elimination, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
                while value:
                    p = value.bit_length() - 1
                    old = elimination.get(p)
                    if old is None:
                        elimination[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            light = sorted(basis, key=int.bit_count)[:20]
            for word in light:
                for other in basis:
                    observe(word ^ other)
            diagnostic.shuffle(basis)
            for initial in light[:4]:
                word = initial
                for _ in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
            if best <= 5:
                break
        return best

    diagnostic_seed = rng.randrange(1 << 62)

    def homology_screen(hx, hz, n):
        xp = stabilizer_pivots(hx)
        zp = stabilizer_pivots(hz)
        xc = _syndrome_columns(hz, n)
        zc = _syndrome_columns(hx, n)
        if not logical_four_free(xc, xp):
            return None
        if not logical_four_free(zc, zp):
            return None
        readings = []
        for side_id, columns, pivots in ((0, xc, xp), (1, zc, zp)):
            diagnostic = random.Random(diagnostic_seed + 100003 * side_id)
            reading = logical_reading(columns, pivots, 8, diagnostic)
            if reading <= 5:
                return None
            diagnostic = random.Random(
                diagnostic_seed + 700001 + 100003 * side_id)
            reading = min(reading, logical_reading(
                columns, pivots, 24, diagnostic))
            if reading <= 5:
                return None
            readings.append(reading)
        return tuple(readings)

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
            assert 19 * n <= 40 * k < 21 * n
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert css_ok(hx, hz)

            # Independently validate the cached component superpositions.
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert actual_a == component(A[0], "A")
            assert actual_b == component(B[0], "B")

            logical_screen = homology_screen(hx, hz, n)
            if logical_screen is None:
                continue
            tri_rows = component([B[0][-1]], "B")
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": (kind, card // 5, card % 5),
                "functional": a["functional"],
                "annihilator_dimension": a["dimension"],
                "component_ranks": (a["rank"], b["rank"]),
                "a_classical_kernel4_free": a["short_free"],
                "b_trinomial_rank": gf2_rank(tri_rows),
                "preferred_b_trinomial_rank": preferred_tri_rank,
                "logical_screen": logical_screen,
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
    """Twenty new mixed-degree/incidence cards on authoritative blocks."""
    from hashlib import sha256
    from math import exp

    q = G.q
    assert q == 48 and A_SHAPE == (1, 4) and B_SHAPE == (2, 7)
    policy, model = divmod(idx % 20, 5)
    cohort = idx // 20
    seed = rng.randrange(1 << 62) ^ (0x3142716 + 1000003 * idx)
    local = random.Random(seed)
    ident = G.idx(G.e)
    mask = (1 << q) - 1

    # Four genuine backbone prescriptions. The last two-internal-edge
    # policy changes relative complement separations, not a cell gauge.
    upper_counts = (
        (2, 0, 1, 1, 1, 1, 1),
        (2, 2, 0, 0, 1, 1, 1),
        (2, 2, 0, 1, 1, 1, 1),
        (2, 2, 0, 0, 1, 1, 1),
    )[policy]
    sb = [list(upper_counts), [2 - c for c in upper_counts]]
    sa = [[3, 3, 2, 2]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert max(map(sum, sb)) <= 8

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
    gallery = [[[g] for g in range(q)]]
    xx, _, _ = lifted_matrices(P, gallery, [[[]]])
    _, zz, _ = lifted_matrices(P, [[[]]], gallery)
    block_rows = {
        "A": [[(row >> (g * q)) & mask for row in xx]
              for g in range(q)],
        "B": [[(row >> (g * q)) & mask for row in zz]
              for g in range(q)],
    }
    block_columns = {
        side: [_syndrome_columns(rows, q) for rows in blocks]
        for side, blocks in block_rows.items()
    }
    del xx, zz

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length() - 1
            word ^= low

    def columns(grid, side):
        """Only GF2 superposition of scaffold-supplied monomial columns."""
        width = len(grid[0]) * q
        answer = [0] * width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    source = block_columns[side][g]
                    for t, value in enumerate(source):
                        answer[j * q + t] ^= value << (i * q)
        return answer

    def rows_of(cols, checks):
        rows = [0] * checks
        for v, column in enumerate(cols):
            for r in bits(column):
                rows[r] |= 1 << v
        return rows

    def native(grid):
        return [[[G.elements[g] for g in cell] for cell in row]
                for row in grid]

    # Odd internal ratios and even bridge voltages make this ACTUAL
    # check graph bipartite under color (check-row XOR quotient parity).
    internal = {}
    for i in range(2):
        places = [j for j in range(7) if sb[i][j] == 2]
        patterns = (
            ((1,), (3,)),
            ((1, 3), (1, 3)),
            ((1, 3), (5,)),
            ((1, 5), (1, 5)),
        )[policy][i]
        assert len(places) == len(patterns)
        for j, b in zip(places, patterns):
            internal[i, j] = b

    def fresh_b():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for j in range(7):
            if sb[0][j] == 1:
                grid[0][j] = [ident]
                grid[1][j] = [G.idx((
                    local.randrange(3), 2 * local.randrange(8)))]
            else:
                i = 0 if sb[0][j] == 2 else 1
                grid[i][j] = [
                    ident, G.idx((local.randrange(3), internal[i, j]))]
        return grid

    def mutate_b(grid):
        candidate = [[list(cell) for cell in row] for row in grid]
        j = local.randrange(7)
        if sb[0][j] == 1:
            candidate[1][j] = [G.idx((
                local.randrange(3), 2 * local.randrange(8)))]
        else:
            i = 0 if sb[0][j] == 2 else 1
            candidate[i][j] = [
                ident, G.idx((local.randrange(3), internal[i, j]))]
        return candidate

    colors = [
        (i ^ (G.elements[t][1] & 1))
        for i in range(2) for t in range(q)
    ]

    def graph_record(grid):
        cols = columns(grid, "B")
        assert all(column.bit_count() == 2 for column in cols)
        duplicates = len(cols) - len(set(cols))
        adjacency = [0] * (2 * q)
        for column in cols:
            a, b = tuple(bits(column))
            assert colors[a] != colors[b]
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a

        # Count four-cycles through one representative of each regular
        # check orbit. Each global cycle is counted at four vertices.
        rooted4 = 0
        for root in (0, q):
            neighbors = adjacency[root]
            for end in range(2 * q):
                if end == root:
                    continue
                shared = (neighbors & adjacency[end]).bit_count()
                rooted4 += shared * (shared - 1) // 2
        four = q * rooted4 // 4
        return dict(
            grid=grid, columns=cols, adjacency=adjacency,
            duplicates=duplicates, four=four,
            cost=10000 * duplicates + rooted4)

    def connected(adjacency):
        reached = frontier = 1
        while frontier:
            nxt = 0
            for v in bits(frontier):
                nxt |= adjacency[v]
            frontier = nxt & ~reached
            reached |= frontier
        return reached == (1 << (2 * q)) - 1

    def graph_fingerprint(adjacency):
        """Invariant of the B graph recoverable from the full Tanner graph.

        The regular action is transitive within each check-row block.
        Thus these two rooted profiles, each repeated q times, give
        the full rooted-profile multiset. This is not a complete
        graph-isomorphism test.
        """
        neighbors = [list(bits(word)) for word in adjacency]
        degrees = [len(ns) for ns in neighbors]

        def rooted(root):
            distance = [-1] * len(adjacency)
            distance[root] = 0
            queue = [root]
            hist = []
            for v in queue:
                d = distance[v]
                while len(hist) <= d:
                    hist.append([0] * 9)
                hist[d][degrees[v]] += 1
                for u in neighbors[v]:
                    if distance[u] < 0:
                        distance[u] = d + 1
                        queue.append(u)

            # Diagonal adjacency powers are rooted closed-walk counts,
            # not estimates of the number of simple cycles.
            walk = [0] * len(adjacency)
            walk[root] = 1
            closed = []
            for length in range(1, 13):
                walk = [sum(walk[u] for u in ns) for ns in neighbors]
                if length >= 3:
                    closed.append(walk[root])
            return (degrees[root],
                    tuple(tuple(layer) for layer in hist),
                    tuple(closed))

        return tuple(sorted((rooted(0), rooted(q))))

    bb, seen_fingerprints = [], set()

    def retain_b(rec):
        if rec["duplicates"] or not connected(rec["adjacency"]):
            return
        fingerprint = graph_fingerprint(rec["adjacency"])
        if fingerprint in seen_fingerprints:
            return
        seen_fingerprints.add(fingerprint)
        bucket = int.from_bytes(
            sha256(repr(fingerprint).encode("ascii")).digest(), "big")
        if bucket % N_SLOTS != idx:
            return
        rows = rows_of(rec["columns"], 2 * q)
        # A connected binary incidence matrix has rank |V|-1.
        assert gf2_rank(rows) == 2 * q - 1
        rec.update(rows=rows, fingerprint=fingerprint, bucket=idx)
        bb.append(rec)
        bb.sort(key=lambda r: (r["four"], r["cost"]))
        del bb[4:]

    # Geometry is measured on authoritative B columns. Annealing changes
    # relative voltages; it never outputs gauge images as extra slots.
    # Nonzero-cycle alternatives remain legal, rather than voiding a
    # slot merely because the ambitious girth target was not reached.
    for restart in range(12):
        current = graph_record(fresh_b())
        best_cost = current["cost"]
        retain_b(current)
        for step in range(384):
            proposed = graph_record(mutate_b(current["grid"]))
            temperature = 6.0 * (1 - step / 384) ** 2 + 0.15
            delta = proposed["cost"] - current["cost"]
            if delta <= 0 or local.random() < exp(
                    min(0.0, -delta / temperature)):
                current = proposed
            improved = current["cost"] < best_cost
            best_cost = min(best_cost, current["cost"])
            if improved or current["cost"] == 0 or step % 32 == 0:
                retain_b(current)
            if sum(rec["four"] == 0 for rec in bb) >= 2:
                break
        if sum(rec["four"] == 0 for rec in bb) >= 2:
            break
    if not bb:
        raise ValueError("no connected B graph reached this invariant bucket")

    tri_profiles = ((1, 2), (1, 4), (1, 7), (3, 8), (1, 0))

    def fresh_tri(profile):
        t, u = profile
        options = []
        for a in range(3):
            for b in range(3):
                g, h = G.idx((a, t)), G.idx((b, u))
                if len({ident, g, h}) != 3:
                    continue
                if table[g][h] == table[h][g]:
                    continue
                options.append([ident, g, h])
        if not options:
            raise ValueError("noncommuting trinomial profile is empty")
        return list(local.choice(options))

    def fresh_a(attempt):
        first = tri_profiles[model]
        second = tri_profiles[(model + 1 + cohort) % 5]
        # Broaden only the second operator after the preferred search.
        if attempt >= 160:
            second = tri_profiles[local.randrange(5)]
        # Different square coordinates avoid the forced four-word
        # obtained from two odd ratios with identical squares.
        g = G.idx((local.randrange(3), 1))
        h = G.idx((local.randrange(3), 3))
        return [[fresh_tri(first), fresh_tri(second),
                 [ident, g], [ident, h]]]

    a_records, a_seen = [], set()
    for attempt in range(256):
        grid = fresh_a(attempt)
        signature = tuple(tuple(sorted(cell)) for cell in grid[0])
        if signature in a_seen:
            continue
        a_seen.add(signature)
        cols = columns(grid, "A")
        if _pairs_no_kernel4(cols) is None:
            continue
        rows = rows_of(cols, q)
        if gf2_rank(rows) != q:
            continue
        a_records.append(dict(
            grid=grid, columns=cols, rows=rows,
            cycles=_four_cycles(rows), signature=signature))
    if not a_records:
        raise ValueError("new A portfolio has no full-rank kernel-four-free entry")

    a_records.sort(key=lambda rec: (rec["cycles"], rec["signature"]))
    aa = []
    for at, rec in enumerate(a_records[:32]):
        floor, cols = _component_floor(rec["rows"], 4 * q, q)
        rec["floor"] = floor
        rec["reading"] = min(
            _component_reading(cols, random.Random(seed + 1009 * at), 8),
            _component_reading(
                cols, random.Random(seed + 700001 + 1013 * at), 32))
        aa.append(rec)
        if sum(r["floor"] == 6 for r in aa) >= 8:
            break
    aa.sort(key=lambda rec: (
        rec["floor"], min(6, rec["reading"]),
        rec["reading"], -rec["cycles"]), reverse=True)
    aa = aa[:4]

    for at, rec in enumerate(bb):
        floor, cols = _component_floor(rec["rows"], 7 * q, q)
        # For these simple bipartite incidence graphs, a zero C4 count
        # exactly excludes component kernels through weight five.
        assert (floor == 6) == (rec["four"] == 0)
        rec["floor"] = floor
        rec["reading"] = min(
            _component_reading(
                cols, random.Random(seed + 100003 + 1009 * at), 8),
            _component_reading(
                cols, random.Random(seed + 800009 + 1013 * at), 32))

    candidates = []
    for a in aa:
        for b in bb:
            A, B = native(a["grid"]), native(b["grid"])
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1440 and k >= 720 and 40 * k < 21 * n):
                continue
            detected = min(6, a["reading"], b["reading"])
            key = (
                min(a["floor"], b["floor"]),
                k * detected * detected,
                -b["four"], -a["cycles"])
            candidates.append((key, A, B, hx, hz, k, a, b))
    if not candidates:
        raise ValueError("incidence pairings missed the actual rate gate")
    candidates.sort(key=lambda item: item[0], reverse=True)

    # Complete-matrix short-kernel exclusion is a preference rather than
    # a repair or a claim that a missed classical word is a logical.
    chosen = candidates[0]
    for candidate in candidates[:4]:
        _, A, B, hx, hz, k, a, b = candidate
        if (_pairs_no_kernel4(_syndrome_columns(hx, 1440)) is not None
                and _pairs_no_kernel4(
                    _syndrome_columns(hz, 1440)) is not None):
            chosen = candidate
            break

    key, A, B, hx, hz, k, a, b = chosen
    actual_a, actual_b = _verified_components(hx, hz, q)
    assert actual_a == a["rows"] and actual_b == b["rows"]
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert css_ok(hx, hz)

    # Recoverability justifies the disjoint invariant buckets:
    # HX checks have degree 12, while every HZ check has degree <=11.
    # Sector-two variables have degree >=16; sector-one variables have
    # degree 4 or 5. Deleting sector two from the recognizable HZ checks
    # leaves four copies of precisely the fingerprinted B graph.
    full_columns = _syndrome_columns(hx + hz, 1440)
    degrees = [column.bit_count() for column in full_columns]
    assert set(degrees[:28 * q]) <= {4, 5}
    assert min(degrees[28 * q:]) >= 16
    assert b["bucket"] == idx

    return dict(
        A=A, B=B, idx=idx, k=k,
        card=(policy, model, cohort),
        component_ranks=(q, 2 * q - 1),
        component_floors=(a["floor"], b["floor"]),
        component_readings=(a["reading"], b["reading"]),
        A_four_cycles=a["cycles"],
        B_check_graph_four_cycles=b["four"],
        recoverable_B_bucket=b["bucket"],
        graph_bucket_modulus=N_SLOTS,
        conservative_pairing_merit=key,
        target_cell=0.500)


def CELL(rng, G, side, i, j, ctx):
    """Return every complete new-shape cell without further randomization."""
    cell = list(ctx[side][i][j])
    assert len(cell) <= CELL_MAX
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
# REFUTED: Bounded systematic joins prioritized by representation annihilator dimension yield accepted k>690 under classical-kernel rejection; all 22 distinct measured-program candidates instead had k=690 and all five promoted distances remained six.
# REFUTED: Canonical-join selection with fixed quotient B cards reliably supplies inequivalent valid slots; the current table contains many isomorphic duplicates and four labeling failures.
# REFUTED: Stabilizer-aware rank-stratified affine joins uncover accepted k=720 or 750 improvements; all 52 distinct current candidates instead retained k=690, and all five measured distances remained six.
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
