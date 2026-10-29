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
CONCEPT = ("C7 semidirect_-1 C16, order 112, A=(1,3), B=(1,4): distinct cubic operators sharing actual regular-module annihilators should raise verified k within cell 0.500 while retaining provisional distance-six readings, improving five-code fitness above 17.371. Selection readings are capped at six; measurement can refute the proposed rank-distance tradeoff.")
GROUP = ("Jump from centre-free C11:C10 to C7:C16 with inversion action and central C8. The order changes from 110 to 112, so this is not a relabeling. The listed successful order-112 beds have centre two, not eight; this bed has no demonstrated distance advantage. Two odd-coordinate binomials and one noncentral even-coordinate binomial avoid the central-square obstruction of three odd binomials. Only scaffold-derived matrices determine ranks, kernels and logical witnesses.")
SHAPES = ("Keep A=(1,3), B=(1,4): q=112 gives n=1456, k>=672 and guaranteed rate 6/13; q_max=115. Components have 336 and 448 variables with 112 checks each, nominal rates 2/3 and 3/4. Parent minima were exclusively sector one, especially Acol1/2 across Bcol0/1/2. Exact full ranks retain 0.475<=k/n<0.525. At a reading of six, larger verified k can improve the weak 0.500 frontier without assuming a distance increase.")
FILLING = ("Counts A=(3,3,3), B=(2,2,2,3) give HX=9+3=12 and HZ=9+3=12. Enumerate distinct right-gauge-canonical cubic A operators and compute their complete regular left kernels from authoritative monomial blocks. Sample common-vector catalogs only to propose distinct triples; actual component rank and exhaustive kernel-four exclusion decide acceptance, and odd A column degree also excludes weight five. Four module/geometry objectives cross five binding B quotient recipes, each with two odd ratios and one noncentral even ratio. Broaden legal component portfolios, independently challenge finalists, then verify full rates, short kernels, original-span localized logicals, weights and CSS. Cap pairing readings at six. No sampled kernel catalog, finite witness search or geometry statistic certifies distance or inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r050_C7invC16_13x14_regular_modules_mixedbins_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7 semidirect C16 requires order 112")
    # x^7=y^16=1, y*x*y^-1=x^-1.
    # The action C16 -> Aut(C7) is parity followed by inversion.
    # Consequently y^2 generates a central subgroup of order eight.
    els = [(a, b) for b in range(16) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 16)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 7,
                (b + d) % 16)

    def inv(g):
        a, b = red(g)
        return ((a if b & 1 else -a) % 7, (-b) % 16)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

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
    """Twenty regular-module/mixed-binomial cards on C7:C16."""
    import itertools
    import math

    q = G.q
    assert q == 112
    assert A_SHAPE == (1, 3) and B_SHAPE == (1, 4)
    card, cohort = idx % 20, idx // 20
    objective, model = divmod(card, 5)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    identity = G.idx(G.e)

    # Native operations nominate supports and their column gauges.
    # Every binary block below comes from the authoritative constructor.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    monomials = [[[g] for g in G.elements]]
    source_a, _, _ = lifted_matrices(G, monomials, [[[]]])
    _, source_b, _ = lifted_matrices(G, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in source_a]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in source_b]
              for g in range(q)],
    }
    del source_a, source_b

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def selected_xor(words, selector):
        value = 0
        for position in bits(selector):
            value ^= words[position]
        return value

    def dependencies(vectors):
        """Complete GF2 dependencies in the ORIGINAL input coordinates."""
        pivots, answer = {}, []
        for position, value in enumerate(vectors):
            word = 1 << position
            while value:
                lead = value.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                answer.append(word)
        return answer

    def component(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j*q)
        return rows

    def signature(cells):
        return tuple(sorted(tuple(sorted(cell)) for cell in cells))

    def native_cells(cells):
        return [[G.elements[g] for g in cell] for cell in cells]

    def syndrome(columns, word):
        return selected_xor(columns, word)

    # Canonicalize only the proposal operators, not submitted codes.
    # p -> p*h preserves the regular left annihilator and is an A-column
    # gauge; keeping different choices of the anchor would waste search.
    operators, memberships = {}, {}
    for a in range(q):
        if a == identity:
            continue
        for b in range(a):
            if b == identity:
                continue
            support = tuple(sorted((identity, a, b)))
            representative = min(
                tuple(sorted(table[g][inverse[root]] for g in support))
                for root in support)
            if support != representative:
                continue
            rows = [
                blocks["A"][support[0]][r] ^
                blocks["A"][support[1]][r] ^
                blocks["A"][support[2]][r]
                for r in range(q)
            ]
            left = dependencies(rows)
            if len(left) < 7:
                continue
            assert len(left) == q - gf2_rank(rows)
            assert all(selected_xor(rows, word) == 0 for word in left)
            operators[support] = {
                "rows": rows, "left": left, "columns": None,
                "individual_ok": None,
            }

            # This is a bounded proposal catalog, NOT an enumeration of
            # the whole annihilator or a kernel-exclusion certificate.
            values = set(left)
            if len(left) <= 7:
                complete = [0]
                for word in left:
                    complete += [value ^ word for value in complete]
                values.update(complete[1:])
            else:
                pairs = [u ^ v for t, u in enumerate(left)
                         for v in left[:t]]
                rng.shuffle(pairs)
                values.update(pairs[:96])
                for _ in range(32):
                    chosen = rng.sample(left, min(3, len(left)))
                    value = 0
                    for word in chosen:
                        value ^= word
                    if value:
                        values.add(value)
            for value in sorted(values):
                memberships.setdefault(value, []).append(support)

    catalogs = {}
    for functional, members in memberships.items():
        members = tuple(sorted(set(members)))
        if len(members) >= 3:
            catalogs.setdefault(members, functional)
    catalog = [(functional, members)
               for members, functional in catalogs.items()]
    catalog.sort(key=lambda item: (-len(item[1]), item[0]))

    def individual_columns(support):
        rec = operators[support]
        if rec["individual_ok"] is None:
            columns = _syndrome_columns(rec["rows"], q)
            rec["columns"] = columns
            rec["individual_ok"] = _pairs_no_kernel4(columns) is not None
        return rec["columns"] if rec["individual_ok"] else None

    pair_cache = {}

    def compatible(left, right):
        key = tuple(sorted((left, right)))
        if key not in pair_cache:
            lc, rc = individual_columns(left), individual_columns(right)
            pair_cache[key] = (
                lc is not None and rc is not None and
                _pairs_no_kernel4(lc + rc) is not None)
        return pair_cache[key]

    # Four genuinely different selection objectives. None transforms an
    # existing code into additional slots. The full rate remains binding.
    desired = (12, 18, 24, 30)[objective]

    def a_geometry(rec):
        defect = q - rec["rank"]
        if objective == 0:
            return (rec["cycles"], abs(defect-desired), -defect)
        if objective == 1:
            return (abs(defect-desired), rec["cycles"], -defect)
        if objective == 2:
            return (-defect, rec["cycles"], abs(defect-desired))
        return (rec["cycles"] + q*abs(defect-desired),
                -defect, rec["cycles"])

    a_records, a_seen = [], set()

    def inspect_a(supports, functional):
        supports = tuple(sorted(supports))
        if len(set(supports)) != 3 or supports in a_seen:
            return
        a_seen.add(supports)
        if not all(compatible(supports[i], supports[j])
                   for i in range(3) for j in range(i)):
            return
        rows = [
            sum(operators[support]["rows"][r] << (j*q)
                for j, support in enumerate(supports))
            for r in range(q)
        ]
        rank = gf2_rank(rows)
        if not 7 <= q-rank <= 30:
            return
        columns = [column for support in supports
                   for column in operators[support]["columns"]]
        assert all(column.bit_count() == 3 for column in columns)
        if _pairs_no_kernel4(columns) is None:
            return
        # All columns have odd degree: a zero syndrome cannot have odd
        # support weight. Thus this exhaustive <=4 test also excludes five.
        a_records.append({
            "cells": [list(support) for support in supports],
            "rows": rows, "columns": columns, "rank": rank,
            "cycles": _four_cycles(rows), "functional": functional,
        })

    if catalog:
        # Interleave large catalogs with independent catalogs throughout
        # the bank; avoid the parent's persistent largest-sixteen cutoff.
        ordered = list(catalog)
        offset = (37*card + 101*cohort) % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]
        broad = list(ordered)
        rng.shuffle(broad)
        ordered = ordered[:24] + broad[:72]
        for turn, (functional, members) in enumerate(ordered):
            legal = [support for support in members
                     if individual_columns(support) is not None]
            if len(legal) < 3:
                continue
            possibilities = math.comb(len(legal), 3)
            if possibilities <= 48:
                triples = list(itertools.combinations(legal, 3))
                rng.shuffle(triples)
            else:
                triples = [tuple(rng.sample(legal, 3))
                           for _ in range(48)]
            for triple in triples:
                inspect_a(triple, functional)
            if len(a_records) >= 40:
                break

    # Independent triples compete if the sampled common-vector catalog
    # missed a useful intersection. Their accepted ranks are still exact.
    if len(a_records) < 8:
        legal = [support for support in operators
                 if individual_columns(support) is not None]
        if len(legal) >= 3:
            for _ in range(768):
                inspect_a(rng.sample(legal, 3), None)
                if len(a_records) >= 24:
                    break
    if not a_records:
        raise ValueError("regular-module triples failed exact component gates")

    # Two odd ratios plus a noncentral even ratio. Three odd ratios would
    # have a central-square pigeonhole obstruction to kernel-four freedom.
    bin_profiles = (
        (1, 3, 2), (1, 3, 4), (1, 3, 6),
        (1, 5, 2), (1, 5, 4),
    )
    tri_profiles = ((1, 2), (1, 4), (2, 3), (2, 5), (1, 8))

    def fresh_b():
        cells = []
        for b in bin_profiles[model]:
            # A nonzero normal coordinate makes the even ratio noncentral.
            a = rng.randrange(1, 7) if not b & 1 else rng.randrange(7)
            ratio = G.idx((a, b))
            cells.append([identity, ratio])
        b1, b2 = tri_profiles[model]
        g = G.idx((rng.randrange(7), b1))
        h = G.idx((rng.randrange(7), b2))
        cells.append([identity, g, h])
        assert all(len(cell) == len(set(cell)) for cell in cells)
        return cells

    b_records, b_seen = [], set()

    def b_batch(trials):
        proposals = []
        for _ in range(trials):
            cells = fresh_b()
            key = signature(cells)
            if key in b_seen:
                continue
            b_seen.add(key)
            rows = component(cells, "B")
            if gf2_rank(rows) != q:
                continue
            proposals.append({
                "cells": cells, "rows": rows, "rank": q,
                "cycles": _four_cycles(rows),
            })
        proposals.sort(key=lambda rec: rec["cycles"])
        for rec in proposals:
            floor, columns = _component_floor(rec["rows"], 4*q, q)
            if floor != 6:
                continue
            rec["columns"] = columns
            b_records.append(rec)
            if len(b_records) >= 10:
                break

    b_batch(192)
    if len(b_records) < 4:
        b_batch(384)
    if not b_records:
        raise ValueError("mixed-binomial B card failed exact kernel-five gates")

    def challenge(rec, side, trials, offset):
        """Binding actual kernel witnesses, with independent local streams."""
        diagnostic = random.Random(seed + offset + 100003*int(side == "B"))
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = set(rec.get("words", ()))
        best = rec.get("reading", len(columns)+1)

        def observe(word, additions):
            nonlocal best
            if word:
                best = min(best, word.bit_count())
                if word.bit_count() <= best+4:
                    additions.add(word)

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            additions = set()
            light = sorted(retained | set(basis),
                           key=lambda w: (w.bit_count(), w))[:16]
            for word in basis:
                observe(word, additions)
            for word in light:
                for other in basis:
                    observe(word ^ other, additions)
            moves = list(retained) + basis
            diagnostic.shuffle(moves)
            for initial in light[:4]:
                word = initial
                for _ in range(2):
                    before = word.bit_count()
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word = proposed
                    observe(word, additions)
                    if word.bit_count() == before:
                        break
            retained = set(sorted(
                retained | additions | set(light),
                key=lambda w: (w.bit_count(), w))[:24])
            if best <= 6 and len(retained) >= 8:
                break
        assert retained
        assert all(syndrome(columns, word) == 0 for word in retained)
        rec["reading"] = min(best, min(w.bit_count() for w in retained))
        rec["words"] = sorted(retained,
                              key=lambda w: (w.bit_count(), w))[:16]

    a_records.sort(key=a_geometry)
    aa = a_records[:12]
    for rec in aa:
        challenge(rec, "A", 8, 700001)

    def a_merit(rec):
        defect = q-rec["rank"]
        if objective == 0:
            secondary = (-rec["cycles"], defect)
        elif objective == 1:
            secondary = (rec["reading"], -rec["cycles"], defect)
        elif objective == 2:
            secondary = (defect, -rec["cycles"])
        else:
            multiplicity = sum(word.bit_count() == rec["reading"]
                               for word in rec["words"])
            secondary = (-multiplicity, defect, -rec["cycles"])
        return (min(6, rec["reading"]), secondary)

    aa.sort(key=a_merit, reverse=True)
    aa = aa[:4]
    for rec in aa:
        challenge(rec, "A", 32, 1700021)
    aa.sort(key=a_merit, reverse=True)

    bb = b_records[:8]
    for rec in bb:
        challenge(rec, "B", 8, 700001)
    bb.sort(key=lambda rec: (
        min(6, rec["reading"]), rec["reading"], -rec["cycles"]),
        reverse=True)
    bb = bb[:4]
    for rec in bb:
        challenge(rec, "B", 32, 1700021)
    bb.sort(key=lambda rec: (
        min(6, rec["reading"]), rec["reading"], -rec["cycles"]),
        reverse=True)

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
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

    def localized(a, b, hx, hz, n):
        """Found logical upper readings in the ORIGINAL stabilizer quotient."""
        reading, count = n+1, 0
        for side, rec, opposite, stabilizers, locations in (
                ("B", b, hz, hx, 3), ("A", a, hx, hz, 4)):
            pivots = span(stabilizers)
            for word in rec["words"][:6]:
                for location in range(locations):
                    if side == "B":
                        actual = word << (4*location*q)
                    else:
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((4*j+location)*q) for j in range(3))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if not outside(actual, pivots):
                        continue
                    count += 1
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    for _ in range(2):
                        before = actual.bit_count()
                        for row in nearby:
                            proposed = actual ^ row
                            if proposed.bit_count() < actual.bit_count():
                                actual = proposed
                        winner = actual
                        for i, first in enumerate(nearby):
                            for second in nearby[:i]:
                                proposed = actual ^ first ^ second
                                if proposed.bit_count() < winner.bit_count():
                                    winner = proposed
                        actual = winner
                        if actual.bit_count() == before:
                            break
                    assert actual and outside(actual, pivots)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    reading = min(reading, actual.bit_count())
                    if reading < 6:
                        return reading, count
        return reading, count

    pairings = [(a, b) for a in aa for b in bb]
    pairings.sort(key=lambda ab: (
        q-ab[0]["rank"],
        min(ab[0]["reading"], ab[1]["reading"]),
        -ab[0]["cycles"]-ab[1]["cycles"]), reverse=True)
    best = None
    for a, b in pairings:
        A, B = [native_cells(a["cells"])], [native_cells(b["cells"])]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1456 and 19*n <= 40*k < 21*n):
            continue
        # These tests concern the FULL verified matrices. No full-code
        # nonabelian symmetry or component-to-quantum inference is used.
        if _pairs_no_kernel4(_syndrome_columns(hx, n)) is None:
            continue
        if _pairs_no_kernel4(_syndrome_columns(hz, n)) is None:
            continue
        actual_a, actual_b = _verified_components(hx, hz, q)
        assert actual_a == a["rows"] and actual_b == b["rows"]
        logical, placements = localized(a, b, hx, hz, n)
        detected = min(6, logical)
        key = (k*detected*detected, detected,
               a["reading"]+b["reading"],
               -a["cycles"]-b["cycles"])
        if best is None or key > best[0]:
            best = (key, A, B, hx, hz, k, a, b, logical, placements)
    if best is None:
        raise ValueError("regular-module pairings missed full short-kernel/rate gates")

    key, A, B, hx, hz, k, a, b, logical, placements = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert sa == [[3, 3, 3]] and sb == [[2, 2, 2, 3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    assert css_ok(hx, hz)
    return {
        "A": A, "B": B, "idx": idx, "k": k,
        "card": (objective, model, cohort),
        "group_model": "C7:C16 inversion with central C8",
        "regular_A_defect": q-a["rank"],
        "common_vector_proposal": a["functional"],
        "component_ranks": (a["rank"], b["rank"]),
        "component_readings": (a["reading"], b["reading"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "verified_logical_upper_reading": logical,
        "nontrivial_localized_placements": placements,
        "pairing_merit": key, "selection_ceiling": 6,
        "target_cell": 0.500,
    }


def _retired_affine_ctx(rng, G, idx):
    """Unused predecessor; NEW_CTX uses the new group's regular modules."""
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

    # Odd repeated fibers change actual relative supports. The last two
    # cards test distinct odd fibers instead of the parent's translation
    # plus nonprimitive-complement prescription.
    spectra = ((1, 1), (3, 3), (9, 9), (1, 7), (3, 7))
    bin_profiles = ((1, 1, 1), (1, 1, 3),
                    (1, 3, 3), (3, 3, 3))

    def fresh_tri(attempt):
        b1, b2 = spectra[card % 5]
        for _ in range(128):
            a, c = rng.randrange(11), rng.randrange(11)
            g, h = (a, b1), (c, b2)
            if g == h:
                continue
            if table[G.idx(g)][G.idx(h)] == table[G.idx(h)][G.idx(g)]:
                continue
            assert len({G.idx(G.e), G.idx(g), G.idx(h)}) == 3
            return [G.e, g, h]
        raise ValueError("odd-fiber noncommuting trinomial exhausted")

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
        survivors = survivors[:4]
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

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def syndrome(columns, word):
        answer = 0
        for c in bits(word):
            answer ^= columns[c]
        return answer

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

    def bin_six_words(record):
        """Bounded genuine six-cycles of the verified binomial graph.

        Check zero is an orbit root. Native left translates provide
        additional witnesses, but the finite enumeration is not exclusion.
        """
        columns = record["columns"]
        adjacency = [[] for _ in range(q)]
        for c, value in enumerate(columns[:3 * q]):
            ns = list(bits(value))
            assert len(ns) == 2
            u, v = ns
            adjacency[u].append((v, c))
            adjacency[v].append((u, c))
        found = set()
        cap = 128

        def visit(vertex, depth, visited, word):
            if len(found) >= cap:
                return
            for neighbor, edge in adjacency[vertex]:
                bit = 1 << edge
                if word & bit:
                    continue
                if depth == 1:
                    if neighbor == 0:
                        candidate = word | bit
                        if candidate.bit_count() == 6:
                            assert syndrome(columns, candidate) == 0
                            found.add(candidate)
                    continue
                if visited & (1 << neighbor):
                    continue
                visit(neighbor, depth - 1,
                      visited | (1 << neighbor), word | bit)
                if len(found) >= cap:
                    return

        visit(0, 6, 1, 0)
        return sorted(found)

    def component_words(record, side):
        """Accumulate actual null words; do not infer logical distance."""
        if "logical_probe_words" in record:
            return record["logical_probe_words"]
        columns = record["columns"]
        width = len(columns)
        diagnostic = random.Random(
            BASE_SEED + 1000003 * idx + 100003 * int(side == "B"))
        order = list(range(width))
        retained = set()
        for turn in range(24):
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
            light = sorted(set(basis) | retained,
                           key=lambda w: (w.bit_count(), w))[:16]
            grown = set(light)
            for first in light:
                for second in basis:
                    candidate = first ^ second
                    if candidate:
                        grown.add(candidate)
            retained = set(sorted(
                retained | grown,
                key=lambda w: (w.bit_count(), w))[:32])
            if turn >= 7 and min(
                    (w.bit_count() for w in retained), default=width) <= 6:
                break
        words = sorted(retained, key=lambda w: (w.bit_count(), w))[:24]
        if side == "B":
            words += bin_six_words(record)
        words = sorted(set(words), key=lambda w: (w.bit_count(), w))
        assert all(syndrome(columns, word) == 0 for word in words)
        record["logical_probe_words"] = words
        return words

    def logical_challenge(a, b, hx, hz, n):
        """Original-coordinate nontrivial logical witnesses only.

        A returned n+1 means a miss. Capping the selection objective at
        eight is conservative bookkeeping, not a distance certificate.
        """
        mask = (1 << q) - 1
        reading = n + 1
        tested = trivial = 0
        witness = None

        # Challenge the parent's persistent B-binomial bottleneck first.
        cases = (
            ("B", b, hz, hx, 3),
            ("A", a, hx, hz, 4),
        )
        diagnostic = random.Random(BASE_SEED + 2700011 + 7919 * idx)

        for side, record, opposite, stabilizers, locations in cases:
            pivots = span(stabilizers)
            words = component_words(record, side)
            translations = list(range(q))
            diagnostic.shuffle(translations)
            identity = G.idx(G.e)
            translations.remove(identity)
            translations.insert(0, identity)

            # First challenge several translates of every retained word.
            # Only promising pairs receive the remaining translations.
            stages = (translations[:4], translations[4:])
            for stage, shifts in enumerate(stages):
                active = words if stage == 0 else [
                    word for word in words if word.bit_count() <= 8]
                for shift in shifts:
                    for word in active:
                        moved = 0
                        for c in bits(word):
                            block, coordinate = divmod(c, q)
                            image = (table[shift][coordinate] if side == "B"
                                     else table[coordinate][shift])
                            moved |= 1 << (block * q + image)
                        assert syndrome(record["columns"], moved) == 0

                        for location in range(locations):
                            if side == "B":
                                actual = moved << (location * 4 * q)
                            else:
                                actual = sum(
                                    ((moved >> (j * q)) & mask)
                                    << ((4 * j + location) * q)
                                    for j in range(3))
                            tested += 1
                            if not outside(actual, pivots):
                                trivial += 1
                                continue
                            assert all((actual & row).bit_count() % 2 == 0
                                       for row in opposite)

                            # Original stabilizers preserve the logical
                            # coset. These finite reductions only find words.
                            if actual.bit_count() <= 14:
                                nearby = sorted(
                                    stabilizers,
                                    key=lambda row: (row & actual).bit_count(),
                                    reverse=True)[:24]
                                for _ in range(2):
                                    before = actual.bit_count()
                                    for row in nearby:
                                        proposed = actual ^ row
                                        if proposed.bit_count() < actual.bit_count():
                                            actual = proposed
                                    best_word = actual
                                    best_weight = actual.bit_count()
                                    for i, first in enumerate(nearby):
                                        partial = actual ^ first
                                        for second in nearby[:i]:
                                            proposed = partial ^ second
                                            weight = proposed.bit_count()
                                            if weight < best_weight:
                                                best_word, best_weight = proposed, weight
                                    actual = best_word
                                    if actual.bit_count() == before:
                                        break
                            assert actual and outside(actual, pivots)
                            assert all((actual & row).bit_count() % 2 == 0
                                       for row in opposite)
                            if actual.bit_count() < reading:
                                reading = actual.bit_count()
                                witness = actual
                            if reading <= 6:
                                return reading, tested, trivial, witness
        return reading, tested, trivial, witness

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
        best = None
        best_key = None

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
            actual_a, actual_b = _verified_components(hx, hz, q)
            assert _syndrome_columns(actual_a, 3 * q) == a["columns"]
            assert _syndrome_columns(actual_b, 4 * q) == b["columns"]
            logical, tested, trivial, witness = logical_challenge(
                a, b, hx, hz, n)
            detected = min(8, logical)
            key = (
                k * detected * detected, detected,
                min(12, a["reading"]) + min(12, b["reading"]),
                -a["cycles"] - b["cycles"])
            if best_key is not None and key <= best_key:
                continue
            best_key = key
            best = {
                "A": A, "B": B, "idx": idx, "k": k,
                "recipe": (kind, card // 5, card % 5),
                "functional": a.get("functional", functional),
                "component_ranks": (a["rank"], b["rank"]),
                "screen": (a["reading"], b["reading"],
                           a["cycles"], b["cycles"]),
                "verified_logical_upper_reading": logical,
                "logical_placements_tested": tested,
                "stabilizer_placements_found": trivial,
                "logical_witness": witness,
                "pairing_merit": key,
            }
        return best

    # Compare several actual annihilator constraints against the same
    # independently new B bank. Do not stop at the first legal pairing.
    best = build("plane")
    for _ in range(3):
        answer = build("line")
        if answer is not None and (
                best is None or answer["pairing_merit"] > best["pairing_merit"]):
            best = answer
    if best is not None:
        return best
    raise ValueError("no homology-challenged annihilator pairing passed ranks")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete homology-challenged cell without new draws."""
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
# REFUTED: Preferring the largest sixteen annihilator-plane catalogs produces k=720..750 survivors under the component gates; every distinct code in the current parent's table instead had k=690, and all five measured distances remained six.
# REFUTED: Compatibility-guided plane-orbit searches with translation-containing B trinomials produce additional accepted rank gain or remove the weight-six family; all 54 distinct current codes had k=690 and all five measured minima remained six, entirely in sector 1.
# REFUTED: Odd repeated-fiber B operators and original-span six-cycle pairing challenges raise the affine bed's measured minima above six; all five current measured codes remained at six and k=690, with abundant sector-one witnesses.
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
