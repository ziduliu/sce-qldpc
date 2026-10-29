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
CONCEPT = ("For frozen SL2(F5), order 120, shapes (1,2)x(2,5), subgroup-fixed bottleneck protection followed by individual central-sign repair should raise five-code fitness above 109.4042 in cell 0.250. Non-improvement refutes this experiment; all budgeted distance readings remain provisional upper bounds.")
GROUP = ("Keep SL2(F5), order 5*(25-1)=120, byte-for-byte unchanged. This exact bed leads cell 0.250 at fitness 109.4042; shown full-generation/witness-repair alternatives measured 100.9998 and 99.2004. The central involution and cyclic subgroups of orders three and five supply further restricted kernel challenges. Their syndromes are sums of authoritative lifted columns, not a separately implemented lift.")
SHAPES = ("Keep A=(1,2), B=(2,5), q=120: n=1440, k>=360 and guaranteed rate 0.250; q_max=124. Unequal components have 240 variables/120 checks and 600 variables/240 checks, nominal rates 1/2 and 3/5. Parent measured minima 20..22 occupied sector 1: X witnesses followed Acol1 across B columns; Z witnesses commonly used both Acols within Bcol2/3/4. Exact full ranks retain [0.225,0.275).")
FILLING = ("Twenty cards cross four B incidence profiles with five initial relative-order recipes. A=(4,4), B row totals eight and degrees either (4,3,3,3,3) or (4,4,3,3,2) retain both quantum caps twelve. Cells have distinct directed differences after projection through the centre. Challenge central-, C3- and C5-fixed component words and exactly enumerate single-A-block kernels when nullity<=14. Repair quotient labels when restricted words dominate, or individual central signs when unrestricted words are lighter. Independent finalist searches only lower accumulated readings. Final pairings challenge actual localized quantum logicals modulo original stabilizer spans and verify ranks, native components, weights and CSS. Restricted or unrestricted component searches do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_SL2F5_12x25_subgroup_bottleneck_signrepair_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL2(F5) requires order 120")
    p = 5
    # Encoding (a,b,c,d) represents the matrix [[a,b],[c,d]].
    els = [(a,b,c,d)
           for a in range(p) for b in range(p)
           for c in range(p) for d in range(p)
           if (a*d-b*c) % p == 1]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % p for x in g)

    def mul(g, h):
        a,b,c,d = g
        e,f,u,v = h
        return ((a*e+b*u) % p, (a*f+b*v) % p,
                (c*e+d*u) % p, (c*f+d*v) % p)

    def inv(g):
        a,b,c,d = g
        return (d % p, (-b) % p, (-c) % p, a % p)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty subgroup-bottleneck cards with local double-cover sign repair."""
    q = G.q
    assert q == 120
    topology = (idx % 20) // 5
    model = idx % 5
    cohort = idx // 20
    diagnostic_seed = rng.randrange(1 << 62)

    # Cache group operations only. All binary matrices still come from
    # lifted_matrices, with exactly the original element indexing.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    orders = []
    for g in range(q):
        h, order = ident, 0
        while True:
            h = table[h][g]
            order += 1
            if h == ident:
                break
            if order > q:
                raise ValueError("invalid group operation")
        orders.append(order)
    relative_order = [
        [orders[table[inverses[g]][h]] for h in range(q)]
        for g in range(q)
    ]

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

    central = next(g for g in range(q) if orders[g] == 2)
    assert sum(order == 2 for order in orders) == 1
    assert all(table[central][g] == table[g][central] for g in range(q))
    quotient_id = [min(g, table[central][g]) for g in range(q)]
    assert len(set(quotient_id)) == 60

    # Coordinate restrictions only. A-left components commute with right
    # multiplication; B-right components commute with left multiplication.
    # Every compressed syndrome below is a sum of verified columns.
    partitions = {"A": {}, "B": {}}
    for prime in (2, 3, 5):
        generator = next(g for g in range(q) if orders[g] == prime)
        subgroup, value = [], ident
        for _ in range(prime):
            subgroup.append(value)
            value = table[value][generator]
        assert value == ident and len(set(subgroup)) == prime
        for side in ("A", "B"):
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {
                    table[root][h] if side == "A" else table[h][root]
                    for h in subgroup
                }
                assert len(orbit) == prime and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(tuple(sorted(orbit)))
            blocks = 2 if side == "A" else 5
            partitions[side][prime] = [
                tuple(j*q+c for c in orbit)
                for j in range(blocks) for orbit in orbits
            ]

    # Preferences concern g^-1*h, invariant under common left
    # translation and conjugation. They are not coset relabelings.
    penalties = (
        {2: 30, 3: 0, 4: 0, 5: 0, 6: 0, 10: 0},
        {2: 30, 3: 4, 4: 1, 5: 0, 6: 4, 10: 0},
        {2: 30, 3: 0, 4: 4, 5: 1, 6: 0, 10: 1},
        {2: 30, 3: 2, 4: 0, 5: 2, 6: 2, 10: 2},
        {2: 30, 3: 1, 4: 1, 5: 0, 6: 1, 10: 2},
    )

    def quotient_sidon(cell):
        if len({quotient_id[g] for g in cell}) != len(cell):
            return False
        left, right = set(), set()
        for g in cell:
            for h in cell:
                if g != h:
                    left.add(quotient_id[table[inverses[g]][h]])
                    right.add(quotient_id[table[g][inverses[h]]])
        wanted = len(cell)*(len(cell)-1)
        return len(left) == len(right) == wanted

    def fresh_cell(count, policy):
        if not count:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            recipes = ((3,), (5,), (6,), (10,), (3,5,6,10))
            wanted = rng.choice(recipes[policy])
            ratios = [g for g in range(q) if orders[g] == wanted]
            root = rng.randrange(q)
            cell = [root, table[root][rng.choice(ratios)]]
            assert quotient_sidon(cell)
            return [G.elements[g] for g in cell]

        best, best_cost = None, None
        for attempt in range(1024):
            cell = rng.sample(range(q), count)
            if not quotient_sidon(cell):
                continue
            rels = [relative_order[g][h]
                    for i, g in enumerate(cell) for h in cell[:i]]
            cost = sum(penalties[policy].get(order, 30)
                       for order in rels)
            if best_cost is None or cost < best_cost:
                best, best_cost = cell, cost
            if best_cost == 0 or (best is not None and attempt >= 31):
                break
        if best is None:
            raise ValueError("central-quotient Sidon recipe exhausted")
        return [G.elements[g] for g in best]

    def support_b():
        # Cards 2/3 exchange one degree-three column for a second
        # degree-four column and a split degree-two column.
        profiles = (
            ((2,2,2,1,1), (2,1,1,2,2)),
            ((1,3,2,1,1), (3,0,1,2,2)),
            ((2,2,1,2,1), (2,2,2,1,1)),
            ((1,3,2,1,1), (3,1,1,2,1)),
        )
        grid = [list(row) for row in profiles[topology]]
        assert [sum(row) for row in grid] == [8, 8]
        degrees = tuple(sum(row[j] for row in grid) for j in range(5))
        assert degrees in ((4,3,3,3,3), (4,4,3,3,2))
        assert _grid_weights([[4, 4]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[4, 4]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def verified_component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        # Empty dummy sides isolate components in the verified helper.
        # These dummy constructions are never submitted as codes.
        if side == "A":
            rows, _, _ = lifted_matrices(probe, indexed, [[[]]])
            width = 2*q
            assert len(rows) == q
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], indexed)
            width = 5*q
            assert len(rows) == 2*q
        assert all(row >> width == 0 for row in rows)
        columns = [0]*width
        for check, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << check
        return rows, columns

    def quality(rows, columns, exact=True):
        """Actual cycle geometry; exact kernel-3/4 counts when requested."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return None
        three = four_words = 0
        if exact:
            pairs = {}
            for i, a in enumerate(columns):
                for b in columns[:i]:
                    syndrome = a ^ b
                    three += int(syndrome in singles)
                    previous = pairs.get(syndrome, 0)
                    four_words += previous
                    pairs[syndrome] = previous+1

        uses, four_cycles = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for column in columns:
            neighbors = list(bits(column))
            degree = len(neighbors)
            forced += degree*(degree-1)*(degree-2)//6
            for i, u in enumerate(neighbors):
                adjacency[u] |= column ^ (1 << u)
                for v in neighbors[:i]:
                    key = (v, u)
                    previous = uses.get(key, 0)
                    four_cycles += previous
                    uses[key] = previous+1

        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u+1))):
                triangles += (
                    neighbors & adjacency[v] & (-1 << (v+1))
                ).bit_count()
        return ((three//3, four_words//3),
                (four_cycles, max(0, triangles-forced)))

    def kernel_probe(rows, columns, trials, diagnostic):
        """Genuine classical kernel witnesses, never a distance certificate."""
        width = len(columns)
        order = list(range(width))
        best, witness = width+1, None

        def observe(word):
            nonlocal best, witness
            if word:
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    observe(word)

            light = sorted(basis, key=int.bit_count)[:24]
            for word in light:
                for other in basis:
                    observe(word ^ other)

            # Cancellation descent searches beyond individual basis words.
            diagnostic.shuffle(basis)
            for initial in light[:8]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word = candidate
                            weight = word.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c)
            if best <= 4:
                break

        if witness is None:
            raise ValueError("no component kernel witness")
        assert all((witness & row).bit_count() % 2 == 0
                   for row in rows)
        return best, witness

    def merit(record):
        return (tuple(-x for x in record["small"]),
                record["d"],
                tuple(-x for x in record["cycles"]),
                record["rank"])

    def component_pool(side):
        side_id = int(side == "B")
        policy = (model + side_id + cohort) % 5
        support = [[4, 4]] if side == "A" else support_b()
        seed = diagnostic_seed + 100003*side_id
        records, signatures, bank = [], set(), set()
        local_cache = {}

        def syndrome(columns, word):
            value = 0
            for c in bits(word):
                value ^= columns[c]
            return value

        def rows_from_columns(columns, height):
            rows = [0]*height
            for c, value in enumerate(columns):
                for r in bits(value):
                    rows[r] |= 1 << c
            return rows

        def localized(columns):
            """Exact single-block minimum at nullity<=14; witnesses otherwise."""
            key = tuple(columns)
            if key in local_cache:
                return local_cache[key]
            pivots, basis = {}, []
            for c, source in enumerate(columns):
                value, word = source, 1 << c
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
            shortest = None

            def note(word):
                nonlocal shortest
                if word and (shortest is None or
                             word.bit_count() < shortest.bit_count()):
                    shortest = word

            if len(basis) <= 14:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length()-1]
                    note(word)
            else:
                for i, word in enumerate(basis):
                    note(word)
                    for other in basis[:i]:
                        note(word ^ other)
            if shortest is not None:
                assert syndrome(columns, shortest) == 0
            local_cache[key] = shortest
            return shortest

        def inspect(grid, trial):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return None
            signatures.add(signature)
            rows, columns = verified_component(grid, side)
            stats = quality(rows, columns, exact=False)
            if stats is None:
                return None
            words = set()
            if side == "A":
                for j in range(2):
                    word = localized(columns[j*q:(j+1)*q])
                    if word is not None:
                        words.add(word << (j*q))
            restricted = {}
            for prime, parts in partitions[side].items():
                compressed = []
                for positions in parts:
                    value = 0
                    for c in positions:
                        value ^= columns[c]
                    compressed.append(value)
                restricted[prime] = (
                    compressed, rows_from_columns(compressed, len(rows)))
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "small": stats[0], "small_checked": False,
                "cycles": stats[1], "trial": trial,
                "rank": gf2_rank(rows), "words": words,
                "restricted": restricted,
                "fixed": {p: len(columns)+1 for p in (2,3,5)},
                "full": len(columns)+1,
                "d": min((w.bit_count() for w in words),
                         default=len(columns)+1),
            }

        def retain(record, word):
            assert word and syndrome(record["columns"], word) == 0
            record["words"].add(word)
            record["d"] = min(record["d"], word.bit_count())
            bank.add(word)
            if len(bank) > 128:
                kept = sorted(bank, key=lambda w: (w.bit_count(), w))[:96]
                bank.clear()
                bank.update(kept)

        def refresh(record):
            for word in bank:
                if (word.bit_count() <= record["d"]+2
                        and syndrome(record["columns"], word) == 0):
                    record["words"].add(word)
                    record["d"] = min(record["d"], word.bit_count())
            if record["words"]:
                record["words"] = set(sorted(
                    record["words"], key=lambda w: (w.bit_count(), w))[:12])

        def fixed_probe(record, prime, trials, turn):
            columns, rows = record["restricted"][prime]
            diagnostic = random.Random(seed+1000003*prime+700001*turn)
            reading, compressed_word = kernel_probe(
                rows, columns, trials, diagnostic)
            actual = 0
            parts = partitions[side][prime]
            for c in bits(compressed_word):
                for position in parts[c]:
                    actual |= 1 << position
            assert actual.bit_count() == prime*reading
            record["fixed"][prime] = min(
                record["fixed"][prime], actual.bit_count())
            retain(record, actual)

        def full_probe(record, trials, turn):
            diagnostic = random.Random(seed+900001+700001*turn)
            reading, word = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            record["full"] = min(record["full"], reading)
            retain(record, word)

        def score(record):
            refresh(record)
            return (record["d"],
                    tuple(-x for x in record["small"]),
                    tuple(-x for x in record["cycles"]),
                    record["rank"])

        # Geometry is only a cheap prescreen, not a kernel exclusion.
        # The central-fixed challenge precedes unrestricted optimization.
        for trial in range(40):
            grid = [[fresh_cell(count, policy) for count in row]
                    for row in support]
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (r["cycles"], r["trial"]))
        screened = records[:18]
        if not screened:
            raise ValueError("empty quotient-Sidon component pool")
        for record in screened:
            fixed_probe(record, 2, 6, 0)
        screened.sort(key=score, reverse=True)
        selected = screened[:6]
        for record in selected:
            for prime in (3, 5):
                fixed_probe(record, prime, 6, 0)
            full_probe(record, 8, 0)
        current = max(selected, key=score)

        positions = [(i,j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]
        for step in range(12):
            refresh(current)
            word = min(current["words"], key=int.bit_count)
            active = {c//q for c in bits(word)}
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            changed = set()
            sign_mode = (
                current["full"] < current["fixed"][2] and step % 3 != 0)
            for move in range(2 if step % 4 == 3 else 1):
                available = [p for p in positions if p not in changed]
                targeted = [p for p in available if p[1] in active]
                choices = targeted if targeted else available
                dense = [p for p in choices if support[p[0]][p[1]] >= 2]
                if dense:
                    choices = dense
                if not choices:
                    break
                i, j = rng.choice(choices)
                changed.add((i,j))
                if sign_mode:
                    # Flip ONE term of ONE cell, not an entire gauge block
                    # or a global group automorphism. Central projection and
                    # projected Sidon differences are preserved exactly.
                    at = rng.randrange(len(grid[i][j]))
                    g = G.idx(grid[i][j][at])
                    grid[i][j][at] = G.elements[table[central][g]]
                    assert quotient_sidon([G.idx(g) for g in grid[i][j]])
                else:
                    grid[i][j] = fresh_cell(support[i][j], policy)
            proposed = inspect(grid, 100+step)
            if proposed is None:
                continue
            if proposed["cycles"][0] > current["cycles"][0]+2*q:
                continue
            fixed_probe(proposed, 2, 6, step+1)
            for prime in (3,5):
                fixed_probe(proposed, prime, 3, step+1)
            full_probe(proposed, 4, step+1)
            selected.append(proposed)
            pk, ck = score(proposed), score(current)
            if pk >= ck or (
                    pk[0] == ck[0]
                    and proposed["cycles"][0] <= current["cycles"][0]+q
                    and rng.random() < 0.10):
                current = proposed

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for record in finalists:
            # Independent challenges accumulate on THIS actual matrix.
            fixed_probe(record, 2, 32, 101)
            for prime in (3,5):
                fixed_probe(record, prime, 12, 101)
            full_probe(record, 40+4*(cohort % 3), 101)
            full_probe(record, 16, 211)
            stats = quality(record["rows"], record["columns"], exact=True)
            assert stats is not None
            record["small"], record["cycles"] = stats
            record["small_checked"] = True
            refresh(record)
            assert all(syndrome(record["columns"], word) == 0
                       for word in record["words"])
        finalists.sort(key=score, reverse=True)
        return finalists[:2]

    def span_pivots(rows):
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

    def localized_logicals(a, b, hx, hz, n):
        """Actual wide-sector logicals, checked modulo ORIGINAL row spans."""
        mask = (1 << q)-1
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span_pivots(stabilizers)
            diagnostic = random.Random(
                diagnostic_seed+(4100009 if kind == "Z" else 5100011))
            words = sorted(record["words"],
                           key=lambda w: (w.bit_count(), w))[:6]
            candidates = []
            for word in words:
                for fixed in range(5 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j+fixed)*q)
                            for j in range(2))
                    else:
                        actual = word << (fixed*5*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.append(actual)

            reading = n+1
            for actual in sorted(set(candidates), key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    # Exhaustive pairs within a sampled local neighborhood.
                    # This is not an exhaustive logical-class search.
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < actual.bit_count():
                                actual, changed = proposed, True
                    if not changed:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            # n+1 means no nontrivial embedding was found, not an exclusion.
            readings.append(reading)
        return tuple(readings)

    aa, bb = component_pool("A"), component_pool("B")
    best = None
    mask = (1 << q)-1
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 12*q == 1440
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 3*q and 9*n <= 40*k < 11*n):
                continue
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert _grid_weights(sa, sb) == 12 <= W_MAX
            assert _no_dead_lines(sa, sb)
            assert max(row.bit_count() for row in hx+hz) <= 12

            # Confirm the screening components against the FINAL native lift.
            actual_a = [
                sum(((hx[r] >> (j*5*q)) & mask) << (j*q)
                    for j in range(2))
                for r in range(q)
            ]
            actual_b = [
                hz[alpha*q+r] & ((1 << (5*q))-1)
                for alpha in range(2) for r in range(q)
            ]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logicals = localized_logicals(a, b, hx, hz, n)
            detected = min(a["d"], b["d"], *logicals)
            key = (
                tuple(-(x+y) for x,y in zip(a["small"], b["small"])),
                k*detected*detected, detected,
                a["d"]+b["d"], -a["cycles"][0]-b["cycles"][0])
            if best is None or key > best[0]:
                best = (key, a, b, hx, hz, k, logicals)

    if best is None:
        raise ValueError("subgroup-challenged pairings missed the target-rate gate")
    key, a, b, hx, hz, k, logicals = best
    assert css_ok(hx, hz)
    return {
        "idx": idx, "cells": {"A": a["grid"], "B": b["grid"]}, "k": k,
        "card": (topology, model, cohort),
        "component_readings": (a["d"], b["d"]),
        "component_fixed_readings": (dict(a["fixed"]), dict(b["fixed"])),
        "component_unrestricted_readings": (a["full"], b["full"]),
        "component_small_kernels": (a["small"], b["small"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "verified_localized_logical_readings": tuple(
            None if d > 1440 else d for d in logicals),
        "repair_model": "subgroup bottleneck and individual central signs",
        "selection_quality": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the jointly designed and screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on this SL2(F5) bed; incumbent slots 6, 28 and 34 fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Full-relative-generation and witness-directed component repair alone improve this incumbent; shown same-bed alternatives measured 100.9998 and 99.2004, below 109.4042.
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
