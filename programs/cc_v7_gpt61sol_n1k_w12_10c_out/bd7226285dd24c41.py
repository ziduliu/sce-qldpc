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
CONCEPT = ("For C5 semidirect_2 C24, order 120, A=(1,2), B=(2,5), fully generating dense cells and binding long-order binomial recipes should raise five-code fitness above 124.2768 in cell 0.250. Non-improvement refutes this experiment; budgeted distance readings remain provisional upper bounds.")
GROUP = ("Jump to the solvable semidirect product C5:C24 with y*x*y^-1=x^2. Since 2 has order four modulo five and four divides twenty-four, this defines an order-120 group with centre C6, genuinely different from perfect SL2(F5) with centre C2. Related SL2(F3)xC5 and C19:C6 jumps measured 115.8068 and 111.3178, below the incumbent: group change alone is insufficient. The order-24 quotient permits relative orders twelve and twenty-four at unchanged n=1440. All binary challenges use authoritative lifted columns.")
SHAPES = ("Keep A=(1,2), B=(2,5), q=120: n=1440, k>=360, guaranteed rate 0.250 and q_max=124. Unequal components retain 240 variables/120 checks and 600 variables/240 checks, nominal rates 1/2 and 3/5. Parent measured X minima 22..23 occupied sector-one Acol strips, predominantly Acol1 across all B columns; accompanying Z readings were 24. No reported minimum used sector-two Arow/Brow blocks. Preserve component lengths and attack subgroup-confined B words; exact full ranks retain [0.225,0.275).")
FILLING = ("Twenty cards cross two positive B incidence profiles, two geometry objectives and five binding long-order recipes. A=(4,4), B row totals eight and degrees (4,3,3,3,3) give HX<=8+4=12 and HZ<=8+4=12. Dense cells must have native bidirectional Sidon differences, antipodal separation and full normalized generation of C5:C24. Binomial recipes use orders twelve/twenty-four alongside fifteen/five/ten/thirty; every prescribed order must remain represented in B after mutation. Challenge C2/C3/normal-C5, C4, central-C6 and C24-fixed words, plus unrestricted and anchored min-sum/OSD witnesses. Independent finalist retests accumulate readings. Original-span logical challenges, native-component agreement, actual ranks, weights and CSS remain binding. Finite searches and found witness-orbit counts certify neither distance nor Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r025_C5C24s2_12x25_generating_longquotient_fixed6_fixed24_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect_2 C24 requires order 120")
    # (a,b)=x^a y^b, y*x*y^-1=x^2.
    # An order-four automorphism of C5 defines the C24 action.
    action = tuple(pow(2, b, 5) for b in range(24))
    assert pow(2, 4, 5) == pow(2, 24, 5) == 1
    assert pow(2, 2, 5) != 1
    els = [(a,b) for b in range(24) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 24)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a+action[b]*c) % 5, (b+d) % 24)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 24
        return ((-action[bi]*a) % 5, bi)

    def idx(g):
        a,b = red(g)
        return 5*b+a

    return dict(elements=els, ident=(0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty generating C5:C24 cards with anchored and subgroup challenges."""
    import math

    q = G.q
    assert q == 120
    topology = (idx % 20) // 5
    support_profile = topology % 2
    geometry_mode = topology // 2
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

    # y^12 is the unique involution; <y^4> is the central C6.
    central = next(g for g in range(q) if orders[g] == 2)
    assert sum(order == 2 for order in orders) == 1
    centre = {
        g for g in range(q)
        if all(table[g][h] == table[h][g] for h in range(q))
    }
    assert len(centre) == 6 and central in centre
    quotient_id = [min(g, table[central][g]) for g in range(q)]
    assert len(set(quotient_id)) == 60

    # Coordinate restrictions only. A-left components commute with right
    # multiplication; B-right components commute with left multiplication.
    # Every compressed syndrome below is a sum of verified columns.
    partitions = {"A": {}, "B": {}}
    for prime in (2, 3, 4, 5, 6, 24):
        generator = next(g for g in range(q) if orders[g] == prime)
        if prime in (2, 3, 6):
            assert generator in centre
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
        {2: 30, 3: 6, 4: 3, 5: 3, 6: 5, 10: 2,
         12: 1, 15: 2, 24: 0, 30: 1},
        {2: 30, 3: 6, 4: 2, 5: 3, 6: 4, 10: 2,
         12: 0, 15: 2, 24: 0, 30: 2},
        {2: 30, 3: 5, 4: 3, 5: 2, 6: 4, 10: 2,
         12: 2, 15: 0, 24: 0, 30: 1},
        {2: 30, 3: 5, 4: 1, 5: 0, 6: 4, 10: 2,
         12: 0, 15: 2, 24: 1, 30: 2},
        {2: 30, 3: 4, 4: 3, 5: 1, 6: 2, 10: 0,
         12: 2, 15: 0, 24: 2, 30: 0},
    )

    def native_sidon(cell):
        """Native Sidon, no antipodes, full relative generation when dense."""
        if len(set(cell)) != len(cell):
            return False
        if len({quotient_id[g] for g in cell}) != len(cell):
            return False
        left, right = set(), set()
        for g in cell:
            for h in cell:
                if g != h:
                    left.add(table[inverses[g]][h])
                    right.add(table[g][inverses[h]])
        wanted = len(cell)*(len(cell)-1)
        if len(left) != wanted or len(right) != wanted:
            return False
        if len(cell) >= 3:
            roots = [table[inverses[cell[0]]][g] for g in cell[1:]]
            divisor = 24
            for root in roots:
                divisor = math.gcd(divisor, G.elements[root][1])
            if divisor != 1:
                return False
            # A nonzero commutator generates the prime normal C5.
            # Together with surjection onto C24 this generates all G.
            if not any(table[a][b] != table[b][a]
                       for at, a in enumerate(roots) for b in roots[:at]):
                return False
        return True

    ratio_pools = {
        order: [g for g in range(q) if orders[g] == order]
        for order in (5, 10, 12, 15, 24, 30)
    }
    assert all(ratio_pools.values())
    # Binding spectra, rechecked after every accepted label mutation.
    bin_recipes = (
        (24,), (12,24), (15,24), (5,12,24), (10,15,30)
    )

    def admissible(cell, policy):
        if not native_sidon(cell):
            return False
        return (len(cell) != 2 or
                relative_order[cell[0]][cell[1]] in bin_recipes[policy])

    def fresh_cell(count, policy):
        if not count:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            wanted = rng.choice(bin_recipes[policy])
            root = rng.randrange(q)
            cell = [root, table[root][rng.choice(ratio_pools[wanted])]]
            assert admissible(cell, policy)
            return [G.elements[g] for g in cell]

        best, best_cost = None, None
        for attempt in range(1024):
            cell = rng.sample(range(q), count)
            if not native_sidon(cell):
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
            raise ValueError("native bidirectional Sidon recipe exhausted")
        return [G.elements[g] for g in best]

    def support_b():
        # Positive supports only: no degree-two block or isolated raw cell.
        # Profile 1 concentrates the degree-four column without a zero.
        profiles = (
            ((2,2,2,1,1), (2,1,1,2,2)),
            ((1,2,2,2,1), (3,1,1,1,2)),
        )
        grid = [list(row) for row in profiles[support_profile]]
        assert [sum(row) for row in grid] == [8, 8]
        assert all(count > 0 for row in grid for count in row)
        assert tuple(sum(row[j] for row in grid) for j in range(5)) \
            == (4,3,3,3,3)
        assert _grid_weights([[4, 4]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[4, 4]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    # Cache only blocks supplied by the authoritative constructor.
    # GF2-linear assembly makes the larger geometry pool affordable.
    block_mask = (1 << q)-1
    monomial_blocks = {"A": [], "B": []}
    for start in range(0, q, 20):
        ids = list(range(start, min(start+20, q)))
        monomials = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe, monomials, [[[]]])
        _, zz, _ = lifted_matrices(probe, [[[]]], monomials)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            monomial_blocks["A"].append([
                (row >> (j*q)) & block_mask for row in xx])
            monomial_blocks["B"].append([
                (row >> (j*q)) & block_mask for row in zz])

    def verified_component(grid, side):
        width = len(grid[0])*q
        rows = [0]*(len(grid)*q)
        mono = monomial_blocks[side]
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    for r, value in enumerate(mono[G.idx(g)]):
                        rows[i*q+r] ^= value << (j*q)
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
        retained = []

        def observe(word):
            nonlocal best, witness
            if word:
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word

        for trial in range(trials):
            diagnostic.shuffle(order)
            if witness is not None and trial % 3 == 1:
                active = list(bits(witness))
                front = diagnostic.sample(active, max(1, len(active)//2))
                chosen = set(front)
                order = front + [c for c in order if c not in chosen]
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

            light = sorted(set(basis+retained),
                           key=lambda w: (w.bit_count(), w))[:24]
            additions = list(light)
            moves = basis+retained
            for word in light:
                for other in moves:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= best+4:
                        additions.append(candidate)

            # Retained genuine circuits permit cancellation across trials.
            diagnostic.shuffle(moves)
            for initial in light[:8]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word = candidate
                            weight = word.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
                additions.append(word)
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c)
            if witness is not None:
                additions.append(witness)
            retained = sorted(set(retained+additions),
                              key=lambda w: (w.bit_count(), w))[:28]
            if best <= 4:
                break

        if witness is None:
            raise ValueError("no component kernel witness")
        assert all((witness & row).bit_count() % 2 == 0
                   for row in rows)
        return best, witness

    def eight_cycles(rows, columns):
        """Exact simple component Tanner C8 count, including chorded cycles.

        A-left and B-right components each have a regular symmetry within
        every variable block. This symmetry is NOT assumed for full codes.
        Pair internally disjoint length-four paths from each orbit root.
        """
        checks = [list(bits(row)) for row in rows]
        neighbors = [list(bits(column)) for column in columns]
        rooted = 0
        for root in range(0, len(columns), q):
            paths = {}
            for first in neighbors[root]:
                for middle in checks[first]:
                    if middle == root:
                        continue
                    for second in neighbors[middle]:
                        if second == first:
                            continue
                        for end in checks[second]:
                            if end == root or end == middle:
                                continue
                            previous = paths.setdefault(end, [])
                            for old_first, old_middle, old_second in previous:
                                if (old_middle != middle
                                        and first != old_first
                                        and first != old_second
                                        and second != old_first
                                        and second != old_second):
                                    rooted += 1
                            previous.append((first, middle, second))
        assert q*rooted % 4 == 0
        return q*rooted//4

    def component_pool(side):
        side_id = int(side == "B")
        policy = (model + side_id + cohort) % 5
        support = [[4, 4]] if side == "A" else support_b()
        seed = diagnostic_seed + 100003*side_id
        records, signatures, bank = [], set(), set()
        local_cache = {}
        orbit_cache = {}

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
            indexed = [[ [G.idx(g) for g in cell] for cell in row]
                       for row in grid]
            if not all(admissible(cell, policy)
                       for row in indexed for cell in row):
                return None
            if side == "B":
                represented = {
                    relative_order[cell[0]][cell[1]]
                    for row in indexed for cell in row if len(cell) == 2
                }
                if represented != set(bin_recipes[policy]):
                    return None
            signature = tuple(tuple(sorted(cell))
                              for row in indexed for cell in row)
            if signature in signatures:
                return None
            signatures.add(signature)
            rows, columns = verified_component(grid, side)
            stats = quality(rows, columns, exact=False)
            if stats is None:
                return None
            four, six_proxy = stats[1]
            if side == "A" and four:
                return None

            # C6 projection excess is exact when C4 is zero. C8 below is
            # counted directly on Tanner paths, not on a projection graph.
            eight = low_six = low_eight = 0
            if four == 0:
                eight = eight_cycles(rows, columns)
                if geometry_mode:
                    if side == "B":
                        # Whole degree-three blocks retain regular symmetry.
                        restrictions = [[c for c in columns
                                         if c.bit_count() == 3]]
                    else:
                        # Challenge geometry within each Acol separately.
                        restrictions = [
                            columns[j*q:(j+1)*q] for j in range(2)]
                    for low_columns in restrictions:
                        if not low_columns:
                            continue
                        assert len(low_columns) % q == 0
                        low_rows = rows_from_columns(low_columns, len(rows))
                        low_stats = quality(
                            low_rows, low_columns, exact=False)
                        assert low_stats is not None
                        assert low_stats[1][0] == 0
                        low_six += low_stats[1][1]
                        low_eight += eight_cycles(low_rows, low_columns)

            cost = (5000*four + (8+4*model)*six_proxy + eight
                    + (4+2*model)*(24*low_six+low_eight))
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
                "cycles": (four, six_proxy, eight, low_six, low_eight),
                "cost": cost, "trial": trial,
                "rank": gf2_rank(rows), "words": words,
                "restricted": restricted,
                "fixed": {p: len(columns)+1 for p in (2,3,4,5,6,24)},
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

        def sparse_challenge(record, attempts, turn):
            """Forced-anchor min-sum/OSD witnesses; never an exclusion test.

            Remove one variable and solve H_rest*y = H_anchor. Every
            successful completion, with the anchor restored, is an actual
            nonzero kernel word in the original component coordinates.
            """
            assert side == "B"
            columns, rows = record["columns"], record["rows"]
            width = len(columns)
            local = random.Random(seed+2300009+1000033*turn)
            best = min(record["words"],
                       key=lambda w: (w.bit_count(), w))

            def note(word):
                nonlocal best
                assert word and syndrome(columns, word) == 0
                if word.bit_count() < best.bit_count():
                    best = word
                if word.bit_count() <= record["d"]+2:
                    retain(record, word)

            for attempt in range(attempts):
                block = (attempt+model+support_profile) % 5
                root = block*q+ident
                root_bit = 1 << root
                target = columns[root]

                edge_variables = []
                variable_edges = [[] for _ in range(width)]
                check_edges = [[] for _ in rows]
                for c, value in enumerate(columns):
                    if c == root:
                        continue
                    for r in bits(value):
                        edge = len(edge_variables)
                        edge_variables.append(c)
                        variable_edges[c].append(edge)
                        check_edges[r].append(edge)

                channel = [2.0+local.uniform(-0.7, 0.7)
                           for _ in columns]
                nearby = sorted({
                    edge_variables[e]
                    for r in bits(target) for e in check_edges[r]
                })

                # Different landscapes, not different submitted labels.
                # Warm supports are translated only as search witnesses.
                warm = [
                    w for w in record["words"]
                    if (w >> (block*q)) & ((1 << q)-1)
                ]
                if attempt % 3 == 1 and warm:
                    source = local.choice(warm)
                    anchors = list(bits(
                        (source >> (block*q)) & ((1 << q)-1)))
                    multiplier = inverses[local.choice(anchors)]
                    translated = sum(
                        1 << ((c//q)*q+table[multiplier][c % q])
                        for c in bits(source))
                    assert translated & root_bit
                    assert syndrome(columns, translated) == 0
                    for c in bits(translated ^ root_bit):
                        channel[c] = local.uniform(0.05, 0.65)
                elif attempt % 3 == 2 and nearby:
                    for c in local.sample(nearby, min(3, len(nearby))):
                        channel[c] = -local.uniform(0.15, 1.2)
                elif geometry_mode:
                    for c in nearby:
                        channel[c] *= local.uniform(0.25, 1.0)

                v_messages = [channel[c] for c in edge_variables]
                c_messages = [0.0]*len(edge_variables)
                posterior = list(channel)
                hard = 0

                for iteration in range(18+6*geometry_mode):
                    for r, incident in enumerate(check_edges):
                        parity = -1 if (target >> r) & 1 else 1
                        first = second = 24.0
                        smallest = -1
                        for e in incident:
                            value = v_messages[e]
                            if value < 0:
                                parity = -parity
                            magnitude = abs(value)
                            if magnitude < first:
                                second, first = first, magnitude
                                smallest = e
                            elif magnitude < second:
                                second = magnitude
                        for e in incident:
                            sign = -1 if v_messages[e] < 0 else 1
                            magnitude = second if e == smallest else first
                            proposed = 0.8*parity*sign*magnitude
                            c_messages[e] = (
                                0.55*c_messages[e]+0.45*proposed)

                    posterior = list(channel)
                    for e, c in enumerate(edge_variables):
                        posterior[c] += c_messages[e]
                    hard = sum(
                        1 << c for c in range(width)
                        if c != root and posterior[c] < 0)
                    residual = target ^ syndrome(columns, hard)
                    if not residual:
                        note(hard ^ root_bit)
                        break
                    for e, c in enumerate(edge_variables):
                        proposed = posterior[c]-c_messages[e]
                        v_messages[e] = max(-24.0, min(
                            24.0, 0.4*v_messages[e]+0.6*proposed))

                # Reliability-ordered elimination supplies exact syndrome
                # completions even when the finite BP iteration did not
                # converge. This is OSD-style witness search, not optimal
                # decoding or a distance lower bound.
                order = [c for c in range(width) if c != root]
                reliability = [
                    abs(value)+local.uniform(0.0, 0.25)
                    for value in posterior]
                order.sort(key=lambda c: reliability[c])
                pivots, circuits = {}, []
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
                        circuits.append(word)

                residual = target ^ syndrome(columns, hard)
                repair = 0
                while residual:
                    old = pivots.get(residual.bit_length()-1)
                    if old is None:
                        break
                    residual ^= old[0]
                    repair ^= old[1]
                if residual:
                    continue

                candidate = hard ^ repair ^ root_bit
                assert candidate & root_bit
                assert syndrome(columns, candidate) == 0
                # Circuits exclude the root, so cancellation cannot erase
                # the forced nonzero anchor or leave the genuine kernel.
                for sweep in range(2):
                    before = candidate.bit_count()
                    local.shuffle(circuits)
                    for move in circuits:
                        proposed = candidate ^ move
                        if proposed.bit_count() < candidate.bit_count():
                            candidate = proposed
                    light = sorted(
                        circuits,
                        key=lambda w: (candidate ^ w).bit_count())[:24]
                    winner = candidate
                    for i, first in enumerate(light):
                        partial = candidate ^ first
                        for second in light[:i]:
                            proposed = partial ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    candidate = winner
                    if candidate.bit_count() == before:
                        break
                note(candidate)

            assert all((best & row).bit_count() % 2 == 0 for row in rows)
            return best.bit_count(), best

        def full_probe(record, trials, turn):
            diagnostic = random.Random(seed+900001+700001*turn)
            reading, word = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            record["full"] = min(record["full"], reading)
            retain(record, word)
            if side == "B" and trials >= 8:
                attempts = 3 if trials < 16 else (6 if trials < 32 else 12)
                reading, word = sparse_challenge(record, attempts, turn)
                record["full"] = min(record["full"], reading)
                retain(record, word)

        def orbit_key(word):
            """Normalize a FOUND word under component regular translations.

            This does not quotient the submitted code, identify arbitrary
            Tanner isomorphisms, or enumerate all minimum-weight words.
            """
            if word not in orbit_cache:
                occupied = list(bits(word))
                first_block = occupied[0]//q
                anchors = [c % q for c in occupied
                           if c//q == first_block]
                images = []
                for anchor in anchors:
                    multiplier = inverses[anchor]
                    images.append(sum(
                        1 << ((c//q)*q + (
                            table[c % q][multiplier] if side == "A"
                            else table[multiplier][c % q]))
                        for c in occupied))
                orbit_cache[word] = min(images)
            return orbit_cache[word]

        def score(record):
            refresh(record)
            minimum_orbits = {
                orbit_key(word) for word in record["words"]
                if word.bit_count() == record["d"]
            }
            record["short_orbits"] = len(minimum_orbits)
            return (record["d"],
                    tuple(-x for x in record["small"]),
                    -record["short_orbits"],
                    -record["cost"], record["rank"])

        # Optimize the WHOLE component, including cross-cell cycles.
        # Geometry remains a prescreen, never a kernel exclusion.
        for trial in range(80 if side == "A" else 48):
            grid = [[fresh_cell(count, policy) for count in row]
                    for row in support]
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (r["cost"], r["trial"]))
        if not records:
            raise ValueError("empty native-Sidon component geometry pool")

        positions = [(i,j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]
        starts = records[:2]
        for chain, initial in enumerate(starts):
            current = initial
            for step in range(72 if side == "A" else 112):
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                i, j = rng.choice(positions)
                old = grid[i][j]
                if step % 5 == 0 or len(old) == 1:
                    grid[i][j] = fresh_cell(len(old), policy)
                else:
                    # Change one actual term. This changes relative labels,
                    # not a common cell gauge or a global automorphism.
                    cell = [G.idx(g) for g in old]
                    new = None
                    for _ in range(64):
                        proposed_cell = list(cell)
                        proposed_cell[rng.randrange(len(cell))] = rng.randrange(q)
                        if admissible(proposed_cell, policy):
                            new = [G.elements[g] for g in proposed_cell]
                            break
                    if new is None:
                        continue
                    grid[i][j] = new
                proposed = inspect(grid, 1000+200*chain+step)
                if proposed is None:
                    continue
                delta = proposed["cost"]-current["cost"]
                steps = 72 if side == "A" else 112
                temperature = q*(20*(1-step/steps)**2+0.3)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    records.append(current)
                    records.sort(key=lambda r: (r["cost"], r["trial"]))
                    del records[32:]

        records.sort(key=lambda r: (r["cost"], r["trial"]))
        zero_four = [r for r in records if r["cycles"][0] == 0]
        # B has a legal fallback if the finite anneal misses zero C4.
        # A zero-C4 is required by inspect, not inferred from a reading.
        screened = (zero_four if zero_four else records)[:18]
        for record in screened:
            fixed_probe(record, 2, 6, 0)
        screened.sort(key=score, reverse=True)
        selected = screened[:6]
        for record in selected:
            for prime in (3, 4, 5, 6, 24):
                fixed_probe(record, prime, 6, 0)
            full_probe(record, 8, 0)
        current = max(selected, key=score)
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
                    # Flip ONE term, preserving antipodal separation.
                    # Native Sidon legality must be rechecked: unlike the
                    # parent's quotient rule it is not automatically preserved.
                    at = rng.randrange(len(grid[i][j]))
                    g = G.idx(grid[i][j][at])
                    grid[i][j][at] = G.elements[table[central][g]]
                    if not admissible(
                            [G.idx(g) for g in grid[i][j]], policy):
                        grid[i][j] = fresh_cell(support[i][j], policy)
                else:
                    grid[i][j] = fresh_cell(support[i][j], policy)
            proposed = inspect(grid, 100+step)
            if proposed is None:
                continue
            if proposed["cycles"][0] > current["cycles"][0]:
                continue
            fixed_probe(proposed, 2, 6, step+1)
            for prime in (3,4,5,6,24):
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
            for prime in (3,4,5,6,24):
                fixed_probe(record, prime, 12, 101)
            full_probe(record, 40+4*(cohort % 3), 101)
            full_probe(record, 16, 211)
            stats = quality(record["rows"], record["columns"], exact=True)
            assert stats is not None
            assert tuple(record["cycles"][:2]) == stats[1]
            record["small"] = stats[0]
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
        "card": (support_profile, geometry_mode, model, cohort),
        "component_readings": (a["d"], b["d"]),
        "component_fixed_readings": (dict(a["fixed"]), dict(b["fixed"])),
        "component_unrestricted_readings": (a["full"], b["full"]),
        "component_small_kernels": (a["small"], b["small"]),
        "component_cycles": (a["cycles"], b["cycles"]),
        "verified_localized_logical_readings": tuple(
            None if d > 1440 else d for d in logicals),
        "repair_model": "C5:C24 generating cells, long-order bins, C6/C24 and anchored challenges",
        "component_found_minimum_regular_orbits": (
            a["short_orbits"], b["short_orbits"]),
        "geometry_costs": (a["cost"], b["cost"]),
        "native_zero_four": (a["cycles"][0] == 0, b["cycles"][0] == 0),
        "selection_quality": key,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete generating long-quotient, independently challenged cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on this SL2(F5) bed; incumbent slots 6, 28 and 34 fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Full-relative-generation and witness-directed component repair alone improve this incumbent; shown same-bed alternatives measured 100.9998 and 99.2004, below 109.4042.
# REFUTED: Redistributing B degrees to (4,4,3,3,2) supplies leading candidates on this bed; none of those current slots advanced to measurement, and their maximum short reading was 21 versus 23 for the standard degree profile.
# REFUTED: Native zero-four-cycle/C8 optimization and subgroup screens make short readings >=24 exclude sub-22 logicals on this bed; current slots 10 and 12 fell from 24 to 20 and 25 to 21, with sector-one X witnesses.
# REFUTED: Switching to SL2(F3)xC5 with generating dense cells and long-order binomials necessarily beats 124.2768 on (1,2)x(2,5); the shown experiment measured 115.8068.
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
