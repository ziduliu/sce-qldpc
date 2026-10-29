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
CONCEPT = ("For F25-additive semidirect C8, order 200, A=(1,2), B=(1,3), verified shared component annihilators should increase measured k while retaining five selected distance readings of sixteen, raising cell 0.300 fitness above 73.1430. Failure of this measured target refutes the experiment; readings remain provisional upper bounds.")
GROUP = ("Keep the frozen validated centre-free F25-additive semidirect C8 group, order 200, with irreducible order-eight action x^2=2 over F5. This bed leads cell 0.300, but independent occupancy, primitive-stride and normal-unit repair models retained the weight-sixteen bottleneck. Test actual rank gain on this established bed rather than assume another labeling objective raises distance.")
SHAPES = ("Keep frozen A=(1,2), B=(1,3), q=200: n=1400, k>=400 and guaranteed rate 2/7 target cell 0.300; q_max=214. Unequal components have 400 and 600 variables with 200 checks each and nominal rates 1/2 and 2/3. Parent minimum X witnesses occupied sector-one single-Acol strips across all three Bcols, and every measured k-kb was zero. Preserve these lengths, challenge both strip types and retain exact full rates in [0.275,0.325).")
FILLING = ("Five distinct-label trinomials retain HX=9 and HZ=12. Twenty cards cross four component-rank allocations with five geometry objectives. Exact dependencies of scaffold-built single-cell rows provide annihilator functionals; lookup of their actual monomial images constructs multiple different supports sharing that functional. Constrained components must have exact rank 192, controls rank 200; support gauges are removed only from internal libraries. Localized words, bounded parity completion and independent kernel probes remain binding. Complete pairings compare measured k times squared conservative readings after exact non-stabilizer strip tests and original-stabilizer reduction. Component deficits do not determine quantum k, and finite searches do not certify distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r030_F25add_C8_12x13_sharedannihilator_rankgain_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 200

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F25-additive semidirect C8 requires order 200")
    # F25 = F5[x]/(x^2-2). The element x has exact order eight.
    # (a,b,t) acts as z -> x^t*z + (a+b*x).
    powers = ((1,0), (0,1), (2,0), (0,2),
              (4,0), (0,4), (3,0), (0,3))
    els = [(a,b,t) for t in range(8)
           for b in range(5) for a in range(5)]

    def red(g):
        return (g[0] % 5, g[1] % 5, g[2] % 8)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        v,w = powers[t]
        return ((a+v*c+2*w*d) % 5,
                (b+v*d+w*c) % 5, (t+u) % 8)

    def inv(g):
        a,b,t = red(g)
        ti = (-t) % 8
        v,w = powers[ti]
        return ((-v*a-2*w*b) % 5,
                (-v*b-w*a) % 5, ti)

    def idx(g):
        a,b,t = red(g)
        return 25*t+5*b+a

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty shared-annihilator/rank-allocation cards with binding diagnostics."""
    q = G.q
    assert q == 200
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

    # Twenty occupancy prescriptions, not translations of submitted codes.
    # A normal control deliberately drops per-cell full generation:
    # its actual block must instead be invertible. Never put two such
    # commuting controls on the same side.
    normal = (0, 0, 0)
    repeated = (0, 0, 1)
    repeated3 = (0, 0, 3)
    consecutive = (0, 1, 2)
    consecutive3 = (0, 3, 6)
    asymmetric = (0, 1, 3)
    a_plans = (
        (repeated, repeated3),
        (repeated, consecutive3),
        (repeated3, consecutive),
        (consecutive, asymmetric),
        (normal, asymmetric),
    )
    b_plans = (
        (repeated, repeated, consecutive),
        (repeated, repeated3, consecutive),
        (repeated3, repeated3, consecutive),
        (repeated, repeated, consecutive3),
        (repeated, repeated3, consecutive3),
        (repeated3, repeated3, consecutive3),
        (repeated, consecutive, asymmetric),
        (repeated3, consecutive, asymmetric),
        (repeated, consecutive3, asymmetric),
        (repeated3, consecutive3, asymmetric),
        (normal, repeated, consecutive),
        (normal, repeated3, consecutive),
        (normal, repeated, consecutive3),
        (normal, repeated3, consecutive3),
        (normal, consecutive, asymmetric),
        (normal, consecutive3, asymmetric),
        (consecutive, consecutive, asymmetric),
        (consecutive3, consecutive3, asymmetric),
        (consecutive, consecutive3, asymmetric),
        (asymmetric, asymmetric, asymmetric),
    )
    plans = {"A": a_plans[model], "B": b_plans[idx % 20]}
    span_cache = {}

    def spans_group(cell):
        roots = tuple(sorted(
            table[inverses[cell[0]]][g] for g in cell[1:]))
        if roots not in span_cache:
            reached, pending = {ident}, [ident]
            for value in pending:
                for root in roots:
                    image = table[value][root]
                    if image not in reached:
                        reached.add(image)
                        pending.append(image)
            span_cache[roots] = len(reached) == q
        return span_cache[roots]

    def legal_cell(cell, profile):
        if len(cell) != 3 or len(set(cell)) != 3:
            return False
        if profile == normal:
            a, b, t = cell[0]
            u, v = (cell[1][0]-a) % 5, (cell[1][1]-b) % 5
            x, y = (cell[2][0]-a) % 5, (cell[2][1]-b) % 5
            return (all(g[2] == t for g in cell)
                    and (u*y-v*x) % 5 != 0)
        ids = [G.idx(g) for g in cell]
        if any(relative_order[g][h] == 2
               for i, g in enumerate(ids) for h in ids[:i]):
            return False
        return spans_group(ids)

    def fresh_cell(count, profile):
        assert count == 3
        for _ in range(512):
            phase = rng.randrange(8)
            cell = []
            for t in profile:
                g = (rng.randrange(5), rng.randrange(5),
                     (t+phase) % 8)
                while g in cell:
                    g = (rng.randrange(5), rng.randrange(5),
                         (t+phase) % 8)
                cell.append(g)
            if legal_cell(cell, profile):
                return cell
        raise ValueError("affine trinomial occupancy recipe exhausted")

    def support_b():
        grid = [[3, 3, 3]]
        assert _grid_weights([[3, 3]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[3, 3]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    # Obtain both monomial conventions from the authoritative helper.
    # Small batches avoid constructing a q-by-q dummy protograph.
    mask = (1 << q)-1
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start+16, q)))
        monomials = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe, monomials, [[[]]])
        _, zz, _ = lifted_matrices(probe, [[[]]], monomials)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j*q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j*q)) & mask for row in zz])

    def verified_component(grid, side):
        # GF2 linear assembly of verified monomial blocks; the final
        # complete construction independently checks this assembly.
        width = len(grid[0])*q
        rows = [0]*(len(grid)*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    block = blocks[side][G.idx(g)]
                    for r, word in enumerate(block):
                        rows[i*q+r] ^= word << (j*q)
        columns = [0]*width
        for check, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << check
        return rows, columns

    local_cache = {}

    def localized_word(columns, key):
        """Exact localized minimum for nullity<=14; witnesses otherwise."""
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
        best = None

        def note(word):
            nonlocal best
            if word and (best is None or
                         word.bit_count() < best.bit_count()):
                best = word

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
            local = random.Random(
                diagnostic_seed+sum((i+1)*g
                                    for i, g in enumerate(key[1])))
            for _ in range(128):
                word = 0
                for other in local.sample(basis, min(4, len(basis))):
                    word ^= other
                note(word)
        if best is not None:
            value = 0
            for c in bits(best):
                value ^= columns[c]
            assert value == 0
        local_cache[key] = best
        return best

    def quality(rows, columns):
        """Exact kernel-3/4 and four-cycle counts; six-cycle proxy."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return None
        pairs = {}
        three = four_words = 0
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
            return width+1, None
        assert all((witness & row).bit_count() % 2 == 0
                   for row in rows)
        return best, witness

    def bounded_short(record, side, budget, diagnostic):
        """Budgeted anchored parity completion through weight sixteen.

        Every output is syndrome-verified. A stopped search never claims
        exhaustive exclusion. Regular actions canonicalize witnesses
        within THIS component; they never generate submitted fillings.
        """
        columns = record["columns"]
        width = len(columns)
        all_variables = (1 << width)-1
        short = record["short_words"]

        def syndrome(word):
            value = 0
            for c in bits(word):
                value ^= columns[c]
            return value

        def orbit_key(word):
            first = (word & -word).bit_length()-1
            block = first // q
            anchors = (word >> (block*q)) & mask
            terms = list(bits(word))
            answer = None
            for anchor in bits(anchors):
                t = inverses[anchor]
                moved = 0
                for c in terms:
                    j, h = divmod(c, q)
                    image = (table[h][t] if side == "A"
                             else table[t][h])
                    moved |= 1 << (j*q+image)
                answer = moved if answer is None else min(answer, moved)
            assert answer and syndrome(answer) == 0
            return answer

        def note(word):
            if not word or word.bit_count() > 16:
                return
            assert syndrome(word) == 0
            weight = word.bit_count()
            if weight < record["d"]:
                record["d"], record["word"] = weight, word
            if len(short) < 64:
                short.add(orbit_key(word))

        if record["word"] is not None:
            note(record["word"])
        if record["small"] != (0, 0) or record["d"] <= 8:
            return

        singles = {value: 1 << c for c, value in enumerate(columns)}
        pairs = {}
        incidence = [0]*len(record["rows"])
        variable_bits = [1 << c for c in range(width)]
        for c, value in enumerate(columns):
            for r in bits(value):
                incidence[r] |= variable_bits[c]
            for j in range(c):
                pairs[value ^ columns[j]] = (
                    variable_bits[c] | variable_bits[j])

        priorities = [diagnostic.random() for _ in columns]
        roots = list(range(0, width, q))
        diagnostic.shuffle(roots)
        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed, used):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                limit = min(16, record["d"])
                left = limit-used
                if value == 0:
                    note(word)
                    return
                if left <= 0 or value.bit_count() > 3*left:
                    return

                one = singles.get(value, 0)
                if one and one & allowed == one:
                    note(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(word | pair)
                left = min(16, record["d"])-used
                if left <= 2:
                    return

                # Every completion must hit each odd check. Branch on
                # the most constrained one and exclude prior alternatives
                # only within this branch, not by global variable order.
                chosen, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if count == 0:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                candidates = list(bits(chosen))
                candidates.sort(key=lambda c: (
                    (value ^ columns[c]).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = variable_bits[c]
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live, used+1)
                    if remaining[0] <= 0 or record["d"] <= 8:
                        break

            # Translate a term in the first occupied column block to
            # coordinate zero. Earlier blocks then contain no terms.
            allowed = all_variables & ~((1 << (root+1))-1)
            visit(columns[root], variable_bits[root], allowed, 1)
            if record["d"] <= 8:
                break

    def short_count(record):
        return sum(word.bit_count() == record["d"]
                   for word in record["short_words"])

    def merit(record):
        return (tuple(-x for x in record["small"]),
                record["d"], -short_count(record),
                tuple(-x for x in record["cycles"]),
                record["rank"])

    def component_pool(side):
        side_id = int(side == "B")
        support = [[3, 3]] if side == "A" else support_b()
        profiles = plans[side]
        seed = diagnostic_seed+100003*side_id
        records, signatures, bank = [], set(), set()

        def survives(record, word):
            value = 0
            for c in bits(word):
                value ^= record["columns"][c]
            return value == 0

        def refresh(record):
            for word in bank:
                if word.bit_count() < record["d"] and survives(record, word):
                    record["d"], record["word"] = word.bit_count(), word

        def score(record):
            refresh(record)
            return merit(record)

        def inspect(grid, trial):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return None
            signatures.add(signature)
            rows, columns = verified_component(grid, side)
            stats = quality(rows, columns)
            if stats is None:
                return None
            words = []
            for j, cell in enumerate(grid[0]):
                key = (side, tuple(sorted(G.idx(g) for g in cell)))
                word = localized_word(columns[j*q:(j+1)*q], key)
                # This routine returns None exactly when the square
                # single-cell block has no kernel. Do not assume a
                # proposed normal polynomial is a unit without this test.
                if profiles[j] == normal and word is not None:
                    return None
                if word is not None:
                    words.append(word << (j*q))
            word = min(words, key=int.bit_count) if words else None
            reading = (word.bit_count() if word is not None
                       else len(columns)+1)
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "small": stats[0], "cycles": stats[1], "trial": trial,
                "rank": gf2_rank(rows), "localized": reading,
                "local_words": words, "d": reading, "word": word,
                "short_words": set(),
            }

        def diagnose(record, trials, offset):
            diagnostic = random.Random(seed+offset)
            reading, word = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            if word is not None:
                bank.add(word)
                if reading < record["d"]:
                    record["d"], record["word"] = reading, word
            budget = ((1500 if trials <= 8 else 6000)
                      if side == "A" else
                      (5000 if trials <= 8 else 24000))
            bounded_short(record, side, budget,
                          random.Random(seed+offset+1709))
            bank.update(record["short_words"])
            bank.update(record["local_words"])
            if len(bank) > 128:
                retained = sorted(
                    bank, key=lambda w: (w.bit_count(), w))[:96]
                bank.clear()
                bank.update(retained)
            refresh(record)
            if record["word"] is not None:
                assert survives(record, record["word"])

        for trial in range(36 if side == "B" else 24):
            grid = [[fresh_cell(count, profiles[j])
                     for j, count in enumerate(row)]
                    for row in support]
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["small"], -r["localized"], r["cycles"], -r["rank"]))
        selected = records[:8 if side == "B" else 6]
        if not selected:
            raise ValueError("empty affine component shortlist")
        for record in selected:
            diagnose(record, 8, 0)
        selected.sort(key=score, reverse=True)

        # Two independent chains change actual labels in columns used
        # by verified words. No symmetry images are submitted as codes.
        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(12 if side == "B" else 6):
                refresh(current)
                active = (sorted({c//q for c in bits(current["word"])})
                          if current["word"] is not None else [])
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                choices = list(range(len(profiles)))
                changed = set()
                for move in range(2 if step % 4 == 3 else 1):
                    targeted = [j for j in active if j not in changed]
                    available = [j for j in choices if j not in changed]
                    if not available:
                        break
                    j = rng.choice(targeted if targeted and
                                   rng.random() < 0.85 else available)
                    changed.add(j)
                    if rng.random() < 0.75:
                        # Retain useful quotient and two-term geometry;
                        # alter one genuinely relative translation label.
                        old_cell = grid[0][j]
                        repaired = None
                        for _ in range(8):
                            cell = list(old_cell)
                            t = rng.randrange(3)
                            old = cell[t]
                            cell[t] = (rng.randrange(5), rng.randrange(5),
                                       old[2])
                            if cell != old_cell and legal_cell(
                                    cell, profiles[j]):
                                repaired = cell
                                break
                        grid[0][j] = (
                            repaired if repaired is not None
                            else fresh_cell(3, profiles[j]))
                    else:
                        grid[0][j] = fresh_cell(3, profiles[j])
                proposed = inspect(grid, 100+16*chain+step)
                if proposed is None or proposed["small"] > current["small"]:
                    continue
                if proposed["cycles"][0] > current["cycles"][0]+2*q:
                    continue
                diagnose(proposed, 6, 300007)
                selected.append(proposed)
                pk, ck = score(proposed), score(current)
                if pk >= ck or (
                        pk[:2] == ck[:2] and rng.random() < 0.12):
                    current = proposed

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for record in finalists:
            # Readings accumulate independently for each actual matrix.
            diagnose(record, 48+8*(cohort % 3), 700001)
            diagnose(record, 16, 900001)
        finalists.sort(key=score, reverse=True)
        for record in finalists:
            refresh(record)
            assert record["word"] is not None
            assert survives(record, record["word"])
        return finalists

    def row_dependencies(rows):
        """Exact left dependencies in the supplied original row coordinates."""
        pivots, dependencies = {}, []
        for r, source in enumerate(rows):
            value, word = source, 1 << r
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                dependencies.append(word)
        return dependencies

    def functional_image(functional, rows):
        answer = 0
        for r in bits(functional):
            answer ^= rows[r]
        return answer

    def new_atom():
        for _ in range(1024):
            ids = [ident] + rng.sample(
                [g for g in range(q) if g != ident], 2)
            if not spans_group(ids):
                continue
            left = {table[inverses[a]][b]
                    for a in ids for b in ids if a != b}
            right = {table[a][inverses[b]]
                     for a in ids for b in ids if a != b}
            if len(left) == len(right) == 6:
                return tuple(ids)
        raise ValueError("new full-generating Sidon atom exhausted")

    def annihilator_library(side):
        """Construct actual supports annihilating one verified row functional."""
        monomials = blocks[side]
        for _ in range(128):
            atom = new_atom()
            rows = [
                monomials[atom[0]][r] ^ monomials[atom[1]][r]
                ^ monomials[atom[2]][r] for r in range(q)
            ]
            dependencies = row_dependencies(rows)
            if len(dependencies) != 8:
                continue
            functional = rng.choice(dependencies)
            assert functional_image(functional, rows) == 0
            images = [functional_image(functional, block)
                      for block in monomials]
            lookup = {}
            for g, image in enumerate(images):
                lookup.setdefault(image, []).append(g)

            library, gauges = [], set()
            for g in range(q):
                if g == ident:
                    continue
                for h in lookup.get(images[ident] ^ images[g], ()):
                    ids = (ident, g, h)
                    if len(set(ids)) != 3 or not spans_group(ids):
                        continue
                    left = {table[inverses[a]][b]
                            for a in ids for b in ids if a != b}
                    right = {table[a][inverses[b]]
                             for a in ids for b in ids if a != b}
                    if len(left) != 6 or len(right) != 6:
                        continue
                    # Remove known single-cell column-gauge copies.
                    # Do not transform a submitted code to create a slot.
                    gauge = min(
                        tuple(sorted(
                            table[a][inverses[root]] if side == "A"
                            else table[inverses[root]][a]
                            for a in ids))
                        for root in ids)
                    if gauge in gauges:
                        continue
                    gauges.add(gauge)
                    assert images[ident] ^ images[g] ^ images[h] == 0
                    library.append(ids)
            if len(library) >= 8:
                rng.shuffle(library)
                return functional, library
        raise ValueError("no sufficiently varied verified annihilator library")

    def rankgain_pool(side):
        side_id = int(side == "B")
        count = 2 if side == "A" else 3
        constrained = (
            topology in (1, 2, 3) if side == "A"
            else topology in (0, 2, 3))
        required_rank = q-8 if constrained else q
        functional, library = (
            annihilator_library(side) if constrained else
            (None, [new_atom() for _ in range(64)]))
        seed = diagnostic_seed + 100003*side_id
        seen, records, bank = set(), [], set()

        def geometry(grid, cycles):
            label_cost = 0
            for cell in grid[0]:
                ids = [G.idx(g) for g in cell]
                rels = [relative_order[a][b]
                        for t, a in enumerate(ids) for b in ids[:t]]
                occupied = len({g[2] for g in cell})
                target = (2, 3, 3, 2, 3)[model]
                label_cost += 4*abs(occupied-target)
                label_cost += (model+1)*rels.count(4)
                label_cost += (4-model)*rels.count(5)
            return ((30, 90, 270, 810, 2430)[model]*cycles[0]
                    + cycles[1] + q*label_cost)

        def inspect(atoms):
            grid = [[[G.elements[g] for g in atom] for atom in atoms]]
            signature = tuple(tuple(sorted(atom)) for atom in atoms)
            if signature in seen:
                return None
            seen.add(signature)
            rows, columns = verified_component(grid, side)
            rank = gf2_rank(rows)
            if rank != required_rank:
                return None
            if functional is not None:
                assert functional_image(functional, rows) == 0
            stats = quality(rows, columns)
            if stats is None:
                return None
            local_words = []
            for j, cell in enumerate(grid[0]):
                key = (side, tuple(sorted(G.idx(g) for g in cell)))
                word = localized_word(columns[j*q:(j+1)*q], key)
                if word is not None:
                    local_words.append(word << (j*q))
            word = min(local_words, key=int.bit_count) if local_words else None
            reading = word.bit_count() if word is not None else count*q+1
            return {
                "grid": grid, "atoms": tuple(atoms), "rows": rows,
                "columns": columns, "rank": rank, "small": stats[0],
                "cycles": stats[1], "localized": reading,
                "local_words": local_words, "word": word, "d": reading,
                "short_words": set(), "annihilator": functional,
                "geometry": geometry(grid, stats[1]),
            }

        def diagnose(record, trials, offset):
            _, word = kernel_probe(
                record["rows"], record["columns"], trials,
                random.Random(seed+offset))
            found = list(record["local_words"])
            if word is not None:
                found.append(word)
            for known in bank:
                value = 0
                for c in bits(known):
                    value ^= record["columns"][c]
                if value == 0:
                    found.append(known)
            for word in found:
                bank.add(word)
                if word.bit_count() < record["d"]:
                    record["d"], record["word"] = word.bit_count(), word
            bounded_short(
                record, side, 5000 if trials <= 8 else 24000,
                random.Random(seed+offset+1709))
            bank.update(record["short_words"])
            if len(bank) > 128:
                bank.intersection_update(sorted(
                    bank, key=lambda w: (w.bit_count(), w))[:96])
            if record["word"] is not None:
                assert all((record["word"] & row).bit_count() % 2 == 0
                           for row in record["rows"])

        def score(record):
            return (tuple(-x for x in record["small"]),
                    min(24, record["d"]), -short_count(record),
                    -record["geometry"])

        for _ in range(64):
            atoms = rng.sample(library, count)
            record = inspect(atoms)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["small"], -min(32, r["localized"]), r["geometry"]))
        selected = records[:6]
        if not selected:
            raise ValueError("no component passed the exact rank-allocation gate")
        for record in selected:
            diagnose(record, 8, 101)
        current = max(selected, key=score)

        for step in range(12):
            atoms = list(current["atoms"])
            active = (sorted({c//q for c in bits(current["word"])})
                      if current["word"] is not None else list(range(count)))
            j = rng.choice(active)
            atoms[j] = rng.choice(library) if constrained else new_atom()
            record = inspect(atoms)
            if record is None:
                continue
            diagnose(record, 8, 300007+7919*step)
            selected.append(record)
            if score(record) >= score(current):
                current = record

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for record in finalists:
            diagnose(record, 48+8*(cohort % 3), 700001)
            diagnose(record, 24, 1900003)
        finalists.sort(key=score, reverse=True)
        return finalists

    def logical_strip_reading(a, b, hx, hz, n):
        """Verified nontrivial strips reduced using original stabilizers."""
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

        reading = n+1
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            sources = set(record["local_words"]) | record["short_words"]
            if record["word"] is not None:
                sources.add(record["word"])
            candidates = []
            for source in sorted(sources, key=int.bit_count)[:4]:
                for strip in range(3 if side == "A" else 2):
                    actual = 0
                    for c in bits(source):
                        block, coordinate = divmod(c, q)
                        target = ((3*block+strip)*q+coordinate
                                  if side == "A"
                                  else (3*strip+block)*q+coordinate)
                        actual |= 1 << target
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.append(actual)
            diagnostic = random.Random(
                diagnostic_seed + (4100009 if side == "A" else 5100011))
            for actual in sorted(set(candidates), key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                nearby = sorted(
                    stabilizers, key=lambda r: (r & actual).bit_count(),
                    reverse=True)[:32]
                for _ in range(96):
                    first, second = diagnostic.sample(nearby, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
        return reading

    aa, bb = rankgain_pool("A"), rankgain_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        tuple(-(x+y) for x, y in
              zip(ab[0]["small"], ab[1]["small"])),
        min(ab[0]["d"], ab[1]["d"]),
        -short_count(ab[0])-short_count(ab[1]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]/(2*q)-ab[1]["cycles"][0]/(3*q)),
        reverse=True)

    best = None
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 7*q == 1400
        k = n-gf2_rank(hx)-gf2_rank(hz)
        # Component deficits do not decide the actual quantum dimension.
        if not (k > 2*q and 11*n <= 40*k < 13*n):
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3, 3]] and sb == [[3, 3, 3]]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx) == 9
        assert max(row.bit_count() for row in hz) == 12
        # Confirm cached GF2 assemblies against the authoritative full lift.
        actual_a = [
            sum(((row >> (j*3*q)) & mask) << (j*q)
                for j in range(2))
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << (3*q))-1) for row in hz[:q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert css_ok(hx, hz)
        logical = logical_strip_reading(a, b, hx, hz, n)
        detected = min(18, a["d"], b["d"], logical)
        key = (
            tuple(-(x+y) for x, y in zip(a["small"], b["small"])),
            k*detected*detected, detected,
            -short_count(a)-short_count(b),
            -a["geometry"]-b["geometry"])
        answer = {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (topology, model, cohort),
            "component_ranks": (a["rank"], b["rank"]),
            "verified_common_annihilators": (
                a["annihilator"], b["annihilator"]),
            "component_readings": (a["d"], b["d"]),
            "verified_logical_strip_reading": (
                logical if logical <= n else None),
            "detected_minimum_kernel_orbits": (
                short_count(a), short_count(b)),
            "component_small_kernels": (a["small"], b["small"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "pairing_merit": key,
        }
        if best is None or key > best[0]:
            best = (key, answer)
    if best is None:
        raise ValueError("shared-annihilator pairings missed k>400/cell 0.300")
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the complete rank-gain-selected and logically challenged trinomial."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == 3 and len({G._key(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on the SL2(F5) bed; current slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Per-cell full generation suffices to exclude sector-1 weight-sixteen logicals on the order-200 bed; all five measured parent codes retained X minima of weight sixteen, with at least 85 found operators each.
# REFUTED: Normal-unit controls and orbit-sixteen repair raise the order-200 bed above fitness 73.1430; all five current measured codes retained k=400 and distance reading sixteen, leaving fitness unchanged.
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
