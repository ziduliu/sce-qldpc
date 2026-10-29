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
CONCEPT = ("For F25-additive semidirect C8, order 200, shapes (1,2)x(1,3), verified common eight-dimensional check dependencies on A, optionally also B, should raise cell 0.300 five-code fitness above 73.143 while retaining measured readings of at least sixteen. Failure of either measured target refutes this experiment; distance readings remain provisional upper bounds.")
GROUP = ("Keep the frozen centre-free F25-additive semidirect C8, order 200, with irreducible order-eight action x^2=2 over F5. This exact bed leads cell 0.300, but every current measured winner has k=400 and zero rank gain. Scaffold-derived monomial operators allow exact common-annihilator proposals. Neither the group structure nor a component rank deficit establishes quantum distance.")
SHAPES = ("Keep frozen A=(1,2), B=(1,3), q=200: n=1400, k>=400 and guaranteed rate 2/7 target cell 0.300; q_max=214. Components have 400 and 600 variables with 200 checks each, nominal rates 1/2 and 2/3. Parent weight-sixteen X minima occupied sector-1 single-Acol strips across all Bcols. Preserve cubic density, require component ranks 192 on constrained sides and 200 otherwise, and accept only authoritative full k>400 in [0.275,0.325).")
FILLING = ("Five distinct-label trinomials retain HX weight nine and HZ weight twelve. Twenty cards cross five quotient-profile/functional preferences with A-only or A-and-B common dependencies and two catalog preferences. Derive check functionals from actual singular cubic blocks, require their verified monomial-image span to have dimension eight, and enumerate full-generating cubic supports annihilating each functional. Remove single-cell right-gauge copies from proposal catalogs. Exact combined ranks enforce the intended dependency budget. Existing accumulated kernel and parity-completion challenges remain; constrained repairs change catalog members rather than apply gauges. Final pairings additionally verify non-stabilizer logical strips using original stabilizer spans and cancellation descent. Select by exact k times a conservative reading capped at sixteen. Search misses, rank gain and sampled witnesses certify no quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r030_F25add_C8_12x13_common_annihilator_rankgain_v3"

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
    """Twenty common-dependency cards with verified logical challenges."""
    q = G.q
    assert q == 200
    topology = (idx % 20) // 5
    model = idx % 5
    cohort = idx // 20
    diagnostic_seed = rng.randrange(1 << 62)
    # Two dependency allocations crossed with two catalog preferences.
    # No full-code rank-deficit formula is used for acceptance.
    target_deficits = {"A": 8, "B": 8 if topology >= 2 else 0}

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

    # Actual quotient prescriptions, not translations of submitted codes.
    # Stride three changes which primitive action appears in a relative
    # generator. We do not assume these recipes prove Tanner inequivalence.
    # Every profile bridges the quotient and avoids antipodal differences.
    repeated = (0, 0, 1)
    repeated3 = (0, 0, 3)
    consecutive = (0, 1, 2)
    consecutive3 = (0, 3, 6)
    asymmetric = (0, 1, 3)
    reflected = (0, 2, 3)
    a_plans = (
        (repeated, repeated3),
        (repeated3, consecutive3),
        (consecutive, consecutive3),
        (repeated3, asymmetric),
        (consecutive3, reflected),
    )
    b_plans = (
        (repeated3, repeated3, consecutive3),
        (repeated, consecutive3, reflected),
        (consecutive, consecutive3, asymmetric),
        (repeated3, asymmetric, reflected),
    )
    plans = {"A": a_plans[model], "B": b_plans[topology]}
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
            ids = [G.idx(g) for g in cell]
            if any(relative_order[g][h] == 2
                   for i, g in enumerate(ids) for h in ids[:i]):
                continue
            if spans_group(ids):
                return cell
        raise ValueError("full-generating affine trinomial recipe exhausted")

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

    def left_dependencies(rows):
        """Exact dependencies among authoritative check rows."""
        pivots, basis = {}, []
        for r, source in enumerate(rows):
            word = 1 << r
            while source:
                lead = source.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (source, word)
                    break
                source ^= old[0]
                word ^= old[1]
            if not source:
                basis.append(word)
        return basis

    def common_catalog(side):
        """Fresh annihilator proposals, not a separate binary lift.

        Every retained cubic annihilates the same actual check functional.
        The combined component rank, checked later, decides whether the
        dependency dimension is exactly eight.
        """
        catalogs = []
        side_blocks = blocks[side]
        seed_profiles = plans[side]
        for attempt in range(128):
            seed_cell = fresh_cell(
                3, seed_profiles[(model+attempt) % len(seed_profiles)])
            rows, _ = verified_component([[seed_cell]], side)
            basis = left_dependencies(rows)
            if len(basis) != 8:
                continue

            mixed = 0
            for word in basis:
                if rng.randrange(2):
                    mixed ^= word
            functionals = [
                min(basis, key=int.bit_count),
                max(basis, key=int.bit_count),
                mixed or basis[0],
            ]
            # Different cards prefer different actual functional geometry.
            if model in (1, 2):
                functionals = functionals[2:] + functionals[:2]
            elif model == 4:
                functionals = functionals[1:] + functionals[:1]

            for functional in list(dict.fromkeys(functionals))[:2]:
                images = []
                for g in range(q):
                    image = 0
                    for r in bits(functional):
                        image ^= side_blocks[g][r]
                    images.append(image)
                if gf2_rank(images) != 8:
                    continue

                by_image = {}
                for g, image in enumerate(images):
                    by_image.setdefault(image, []).append(g)
                members = set()
                for u in range(q):
                    if u == ident:
                        continue
                    for v in by_image.get(functional ^ images[u], ()):
                        if v == ident or v == u:
                            continue
                        proposed = (ident, u, v)
                        # p -> p*h is a single-cell column gauge.
                        # Keep one representative, not its anchor copies.
                        cell = min(
                            tuple(sorted(table[g][inverses[root]]
                                         for g in proposed))
                            for root in proposed)
                        if cell in members or not spans_group(cell):
                            continue
                        if any(relative_order[g][h] == 2
                               for at, g in enumerate(cell)
                               for h in cell[:at]):
                            continue
                        image = 0
                        for g in cell:
                            image ^= images[g]
                        if image:
                            continue
                        members.add(cell)

                if len(members) < 6:
                    continue
                members = sorted(members)
                weight = functional.bit_count()
                preferred_weight = (80, 96, 112, 64, 128)[model]
                if topology % 2:
                    key = (-abs(weight-preferred_weight), len(members))
                else:
                    key = (len(members), -abs(weight-preferred_weight))
                catalogs.append({
                    "functional": functional, "members": members,
                    "key": key, "image_rank": gf2_rank(images),
                })
                break
            if len(catalogs) >= 3:
                break
        if not catalogs:
            raise ValueError("no fresh eight-dimensional annihilator catalog")
        return max(catalogs, key=lambda rec: rec["key"])

    def profile_cost(cell, profile):
        """Quotient occupancy preference, not an equivalence certificate."""
        actual = [G.elements[g][2] for g in cell]
        return min(
            3-sum(min(actual.count(t),
                      sum((u+phase) % 8 == t for u in profile))
                  for t in range(8))
            for phase in range(8))

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

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def translate_word(word, h, side):
        """Witness symmetry only; never creates a submitted code."""
        moved = 0
        for c in bits(word):
            block, g = divmod(c, q)
            image = table[g][h] if side == "A" else table[h][g]
            moved ^= 1 << (block*q+image)
        return moved

    def kernel_probe(rows, columns, trials, diagnostic, side):
        """Accumulate genuine circuits and verified within-code translations."""
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

        for turn in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)

            light = sorted(set(basis+retained), key=int.bit_count)[:24]
            for word in light:
                for other in basis:
                    observe(word ^ other)

            diagnostic.shuffle(basis)
            moves = retained+basis
            improved = []
            for initial in light[:6]:
                word, weight = initial, initial.bit_count()
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
                improved.append(word)

            orbit = []
            if witness is not None and turn % 4 == 0:
                for h in diagnostic.sample(range(q), 32):
                    word = translate_word(witness, h, side)
                    # Challenge the proposed symmetry on actual syndromes.
                    assert word and syndrome(columns, word) == 0
                    orbit.append(word)
                    for other in retained+light[:12]:
                        observe(word ^ other)
            if len(light) >= 3:
                for _ in range(96):
                    a, b, c = diagnostic.sample(light, 3)
                    observe(a ^ b ^ c)
            retained = sorted(
                set(retained+light+improved+orbit),
                key=lambda w: (w.bit_count(), w))[:40]
            if best <= 4:
                break

        if witness is None:
            return width+1, None
        assert syndrome(columns, witness) == 0
        assert all((witness & row).bit_count() % 2 == 0
                   for row in rows)
        return best, witness

    def register_word(record, word, side):
        """Retain actual words and capped counts of their regular orbits."""
        if not word:
            return
        weight = word.bit_count()
        assert syndrome(record["columns"], word) == 0
        if weight < record["d"]:
            record["d"], record["word"] = weight, word
        if weight > 16:
            return
        first = next(bits(word))
        block = first//q
        anchors = (word >> (block*q)) & mask
        canonical = min(
            translate_word(word, inverses[g], side)
            for g in bits(anchors))
        assert syndrome(record["columns"], canonical) == 0
        known = record["short_orbits"]
        if canonical in known or len(known) < 32:
            known[canonical] = canonical

    def parity_completion(record, side, budget, seed, cap=16):
        """Budgeted anchored search for actual component kernel words.

        The first occupied block can be normalized to its identity term.
        Branch exclusions avoid duplicate completions. Exhausting budget
        makes NO claim of exhaustive exclusion through cap.
        """
        columns = record["columns"]
        width = len(columns)
        assert ident == 0
        assert all(c.bit_count() == 3 for c in columns)
        local = random.Random(seed)
        singles, pairs = {}, {}
        incidence = [0]*len(record["rows"])

        for c, value in enumerate(columns):
            word = 1 << c
            if value == 0:
                register_word(record, word, side)
            old = singles.get(value)
            if old is not None:
                register_word(record, old ^ word, side)
            else:
                singles[value] = word
            for r in bits(value):
                incidence[r] |= word
        for c, value in enumerate(columns):
            for d in range(c):
                key = value ^ columns[d]
                word = (1 << c) | (1 << d)
                old = singles.get(key)
                if old is not None:
                    register_word(record, old ^ word, side)
                old = pairs.get(key)
                if old is not None:
                    register_word(record, old ^ word, side)
                else:
                    pairs[key] = word

        if record["d"] <= 4:
            return
        priorities = [local.random() for _ in columns]
        roots = list(range(0, width, q))
        local.shuffle(roots)
        all_variables = (1 << width)-1

        for root in roots:
            remaining = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                used = word.bit_count()
                left = min(cap, record["d"])-used
                if value == 0:
                    register_word(record, word, side)
                    return
                if left <= 0 or value.bit_count() > 3*left:
                    return

                one = singles.get(value, 0)
                if one and (one & allowed) == one:
                    register_word(record, word | one, side)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and (pair & allowed) == pair:
                    register_word(record, word | pair, side)
                left = min(cap, record["d"])-used
                if left <= 2:
                    return

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
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            allowed = all_variables & ~((1 << (root+1))-1)
            visit(columns[root], 1 << root, allowed)

    def merit(record):
        words = record["short_orbits"]
        multiplicity = tuple(
            -sum(w.bit_count() <= cutoff for w in words)
            for cutoff in (12, 14, 16))
        return (tuple(-x for x in record["small"]),
                record["d"], multiplicity,
                tuple(-x for x in record["cycles"]),
                record["rank"])

    def component_pool(side):
        side_id = int(side == "B")
        support = [[3, 3]] if side == "A" else support_b()
        profiles = plans[side]
        seed = diagnostic_seed+100003*side_id
        records, signatures, bank = [], set(), set()
        deficit = target_deficits[side]
        catalog = common_catalog(side) if deficit else None
        functional = catalog["functional"] if catalog else None
        choices = []
        if catalog:
            for profile in profiles:
                ranked = list(catalog["members"])
                rng.shuffle(ranked)
                ranked.sort(key=lambda cell: profile_cost(cell, profile))
                choices.append(ranked[:min(16, len(ranked))])

        def draw(j, previous=()):
            if not catalog:
                return fresh_cell(3, profiles[j])
            used = {tuple(sorted(G.idx(g) for g in cell))
                    for cell in previous}
            available = [cell for cell in choices[j] if cell not in used]
            if not available:
                available = [cell for cell in catalog["members"]
                             if cell not in used]
            if not available:
                raise ValueError("common catalog has no distinct next cell")
            return [G.elements[g] for g in rng.choice(available)]

        def fresh_grid():
            row = []
            for j in range(len(profiles)):
                row.append(draw(j, row))
            return [row]

        def constrained_repair(cell, j):
            old = set(G.idx(g) for g in cell)
            available = [candidate for candidate in choices[j]
                         if set(candidate) != old]
            nearby = [candidate for candidate in available
                      if len(old.intersection(candidate)) >= 2]
            pool = nearby if nearby and rng.random() < 0.75 else available
            if not pool:
                pool = [candidate for candidate in catalog["members"]
                        if set(candidate) != old]
            return [G.elements[g] for g in rng.choice(pool)]

        def survives(record, word):
            value = 0
            for c in bits(word):
                value ^= record["columns"][c]
            return value == 0

        def refresh(record):
            for word in bank:
                relevant = (word.bit_count() < record["d"]
                            or (side == "B" and word.bit_count() <= 16))
                if relevant and survives(record, word):
                    register_word(record, word, side)

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
            rank = gf2_rank(rows)
            if rank != q-deficit:
                return None
            if functional is not None:
                image = 0
                for r in bits(functional):
                    image ^= rows[r]
                assert image == 0
                native_classes = [
                    tuple(sorted(G.idx(g) for g in cell))
                    for cell in grid[0]]
                if len(set(native_classes)) != len(native_classes):
                    return None
            stats = quality(rows, columns)
            if stats is None:
                return None
            words = []
            for j, cell in enumerate(grid[0]):
                key = (side, tuple(sorted(G.idx(g) for g in cell)))
                word = localized_word(columns[j*q:(j+1)*q], key)
                if word is not None:
                    words.append(word << (j*q))
            word = min(words, key=int.bit_count) if words else None
            reading = (word.bit_count() if word is not None
                       else len(columns)+1)
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "small": stats[0], "cycles": stats[1], "trial": trial,
                "rank": rank, "localized": reading,
                "local_words": words, "d": reading, "word": word,
                "short_orbits": {}, "functional": functional,
                "dependency_target": deficit,
            }

        def diagnose(record, trials, offset, short_budget=0):
            diagnostic = random.Random(seed+offset)
            reading, word = kernel_probe(
                record["rows"], record["columns"],
                trials, diagnostic, side)
            if word is not None:
                bank.add(word)
                register_word(record, word, side)
            for word in record["local_words"]:
                bank.add(word)
                register_word(record, word, side)
            if short_budget and side == "B" and record["d"] > 4:
                parity_completion(
                    record, side, short_budget, seed+offset+170003)
            bank.update(record["short_orbits"].values())
            if len(bank) > 128:
                retained = sorted(
                    bank, key=lambda w: (w.bit_count(), w))[:96]
                bank.clear()
                bank.update(retained)
            refresh(record)
            if record["word"] is not None:
                assert survives(record, record["word"])

        for trial in range(64 if catalog else
                           (48 if side == "B" else 24)):
            grid = fresh_grid()
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["small"], -r["localized"], r["cycles"], -r["rank"]))
        selected = records[:8 if side == "B" else 6]
        if not selected:
            raise ValueError("empty affine component shortlist")
        for record in selected:
            diagnose(record, 12 if side == "B" else 8, 0, 7000)
        selected.sort(key=score, reverse=True)

        # Retain two labels in most moves; alter an actual relative
        # translation coordinate in a block used by a verified word.
        # Independent chains are searches, not submitted orbit images.
        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(16 if side == "B" else 8):
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
                    cell = list(grid[0][j])
                    if catalog:
                        grid[0][j] = constrained_repair(cell, j)
                    elif rng.random() < 0.75:
                        t = rng.randrange(3)
                        old = cell[t]
                        cell[t] = (rng.randrange(5), rng.randrange(5), old[2])
                        ids = [G.idx(g) for g in cell]
                        legal = (
                            len(set(ids)) == 3 and cell[t] != old
                            and spans_group(ids)
                            and all(relative_order[g][h] != 2
                                    for at, g in enumerate(ids)
                                    for h in ids[:at]))
                        grid[0][j] = (cell if legal
                                      else fresh_cell(3, profiles[j]))
                    else:
                        grid[0][j] = fresh_cell(3, profiles[j])
                proposed = inspect(grid, 100+16*chain+step)
                if proposed is None or proposed["small"] > current["small"]:
                    continue
                if proposed["cycles"][0] > current["cycles"][0]+2*q:
                    continue
                diagnose(proposed, 10 if side == "B" else 6,
                         300007, 6000)
                selected.append(proposed)
                pk, ck = score(proposed), score(current)
                if pk >= ck or (
                        pk[:2] == ck[:2] and rng.random() < 0.12):
                    current = proposed

        finalists = sorted(selected, key=score, reverse=True)[:3]
        for record in finalists:
            # Only independently challenged records can leave this pool.
            # Both phases retain old words, so readings cannot increase.
            trials = (96 if side == "B" else 48)+8*(cohort % 3)
            diagnose(record, trials, 700001,
                     80000+12000*(cohort % 3))
            diagnose(record, 32 if side == "B" else 16,
                     900001, 40000)
        finalists.sort(key=score, reverse=True)
        for record in finalists:
            refresh(record)
            assert record["word"] is not None
            assert survives(record, record["word"])
        return finalists

    def stabilizer_span(rows):
        pivots = {}
        for source in rows:
            while source:
                lead = source.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = source
                    break
                source ^= old
        return pivots

    def outside_span(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_challenge(hx, hz, a, b, n):
        """Actual logical strips in the ORIGINAL stabilizer quotient.

        A returned reading is a genuine upper bound. n+1 means no
        logical was found by this restriction, not a distance claim.
        """
        readings = []
        for side, record, opposite, stabilizers, strips in (
                ("A", a, hx, hz, 3),
                ("B", b, hz, hx, 2)):
            pivots = stabilizer_span(stabilizers)
            sources = set(record["local_words"])
            sources.update(record["short_orbits"])
            sources.add(record["word"])
            sources.discard(None)
            candidates = set()
            for source in sorted(sources, key=int.bit_count)[:6]:
                assert syndrome(record["columns"], source) == 0
                for strip in range(strips):
                    actual = 0
                    for c in bits(source):
                        block, coordinate = divmod(c, q)
                        target = ((3*block+strip)*q+coordinate
                                  if side == "A" else
                                  (3*strip+block)*q+coordinate)
                        actual |= 1 << target
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside_span(actual, pivots):
                        candidates.add(actual)
            diagnostic = random.Random(
                diagnostic_seed+(3100009 if side == "A" else 4100011))
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
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
                    reverse=True)[:32]
                for move in range(96):
                    first, second = diagnostic.sample(neighborhood, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside_span(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                readings.append(actual.bit_count())
        return min(readings, default=n+1), tuple(readings)

    aa, bb = component_pool("A"), component_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        tuple(-(x+y) for x, y in
              zip(ab[0]["small"], ab[1]["small"])),
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]/(2*q)-ab[1]["cycles"][0]/(3*q)),
        reverse=True)

    best = None
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 7*q == 1400
        k = n-gf2_rank(hx)-gf2_rank(hz)
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
        logical_d, logical_weights = logical_challenge(hx, hz, a, b, n)
        # This cap limits selection on optimistic finite-budget readings.
        # It is not a lower bound on the distance.
        detected = min(16, a["d"], b["d"], logical_d)
        key = (
            k*detected*detected, detected,
            a["d"]+b["d"],
            -len(a["short_orbits"])-len(b["short_orbits"]),
            -a["cycles"][0]-b["cycles"][0],
        )
        answer = {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (topology, model, cohort),
            "coset_profiles": plans,
            "component_readings": (a["d"], b["d"]),
            "component_small_kernels": (a["small"], b["small"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "component_ranks": (a["rank"], b["rank"]),
            "common_check_functionals": (
                a["functional"], b["functional"]),
            "dependency_targets": target_deficits,
            "verified_logical_reading": logical_d,
            "verified_logical_weights": logical_weights,
            "pairing_merit": key,
        }
        if best is None or key > best[0]:
            best = (key, answer)
    if best is None:
        raise ValueError("common-dependency pairings missed k-gain/rate gates")
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the complete rank-gain-selected trinomial without draws."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == 3 and len({G._key(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on the SL2(F5) bed; current slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Primitive-stride-three cubic profiles and detected weight-sixteen family repair improve the order-200 bed above 73.143; all five current measured winners instead retained k=400, d=16 and fitness 73.143.
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
