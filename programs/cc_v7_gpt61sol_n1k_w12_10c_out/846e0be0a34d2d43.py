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
CONCEPT = ("On F9-additive semidirect C4, order 36, A=(2,5), B=(3,7), balanced column degrees and independently challenged multirow component kernels should produce five measured readings >=17 and fitness above 76.069 in cell 0.300. Failure refutes this measurement hypothesis; readings remain provisional upper bounds.")
GROUP = ("Preserve the additive quadratic-field semidirect construction rule, resizing F5[x]/(x^2-2) and C8 to F3[x]/(x^2-2) and C4. Over F3 the polynomial is irreducible and multiplication by x has exact order four, giving q=36. Related order-36 centre-one six-class beds reached 42.4 on these shapes; matching invariants are not a fingerprint identification, and those modest readings motivate stronger filling selection rather than distance transfer.")
SHAPES = ("Change to A=(2,5), B=(3,7): n=41*36=1476, k>=12*36=432 and guaranteed rate 12/41=0.292683 target cell 0.300; q_max=36. Components have 180 variables/72 checks and 252 variables/108 checks, nominal rates 3/5 and 4/7, balancing the parent's 1/2 and 2/3 rates. Parent minima occupied sector-one Acol/Bcol strips, especially single-Acol X strips across B columns. Replace both kernels with multirow components and require exact full rates [0.275,0.325).")
FILLING = ("A has five degree-three columns and row totals seven/eight; B column totals are (3,4,4,4,4,4,4) and all three row totals are nine. Thus HX<=8+4=12 and HZ=9+3=12. Twenty cards cross B zero budgets 0..3 with five relative-order/generation preferences; every column touches at least two base checks. Scaffold-built monomial blocks supply all binary screening matrices. Exact localized dependencies, kernel-four detection, connectivity and cycle statistics shortlist proposals; verified kernel witnesses direct whole-column replacement. Independent finalist searches only lower accumulated readings, with exact Tanner eight-cycles as a tie-breaker. Original full matrices verify component agreement, ranks, weights, non-stabilizer logical embeddings and CSS. Search exhaustion and component readings certify neither quantum distance nor Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r030_F9add_C4_25x37_balanced_column_witness_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F9-additive semidirect C4 requires order 36")
    # Same quadratic-field additive semidirect construction rule.
    # F9 = F3[x]/(x^2-2); x has exact order four.
    # (a,b,t) acts as z -> x^t*z + (a+b*x).
    p, r = 3, 4
    powers = ((1,0), (0,1), (2,0), (0,2))
    els = [(a,b,t) for t in range(r)
           for b in range(p) for a in range(p)]

    def red(g):
        return (g[0] % p, g[1] % p, g[2] % r)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        v,w = powers[t]
        return ((a+v*c+2*w*d) % p,
                (b+v*d+w*c) % p, (t+u) % r)

    def inv(g):
        a,b,t = red(g)
        ti = (-t) % r
        v,w = powers[ti]
        return ((-v*a-2*w*b) % p,
                (-v*b-w*a) % p, ti)

    def idx(g):
        a,b,t = red(g)
        return p*p*t+p*b+a

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def _retired_commonleft_ctx(rng, G, idx):
    """Inactive historical order-200 common-annihilator experiment."""
    q = G.q
    assert q == 200
    topology = (idx % 20) // 5
    model = idx % 5
    cohort = idx // 20
    diagnostic_seed = rng.randrange(1 << 62)
    rng = random.Random(
        diagnostic_seed ^ (0xA11F25C8 + 1000003*idx))

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

    # A's two operators will share an actual left annihilator, rather
    # than merely follow quotient occupancy recipes. B remains cubic.
    repeated = (0, 0, 1)
    repeated3 = (0, 0, 3)
    consecutive3 = (0, 3, 6)
    asymmetric = (0, 1, 3)
    reflected = (0, 2, 3)
    separated = (0, 2, 5)
    b_plans = (
        (repeated3, repeated3, consecutive3),
        (repeated3, asymmetric, reflected),
        (repeated, separated, reflected),
        (consecutive3, separated, asymmetric),
    )
    plans = {"A": (None, None), "B": b_plans[topology]}
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

    def grid_signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def coefficient_kernel(images):
        """Exact dependencies of the supplied authoritative images."""
        pivots, basis = {}, []
        for c, image in enumerate(images):
            value, word = image, 1 << c
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
        return basis

    # A-left monomial operators preserve right-C8-fixed check vectors.
    # Restrict their ACTUAL rows, then verify expansion back to all 200
    # coordinates. No affine permutation matrix is implemented here.
    normal_width = 25
    normal_mask = (1 << normal_width)-1

    def expand_functional(word):
        return sum(word << (normal_width*t) for t in range(8))

    check_orbits = [
        [G.idx((a, b, t)) for t in range(8)]
        for b in range(5) for a in range(5)
    ]
    restricted_mono = []
    for g in range(q):
        images = []
        for orbit in check_orbits:
            full = 0
            for r in orbit:
                full ^= blocks["A"][g][r]
            image = full & normal_mask
            assert full == expand_functional(image)
            images.append(image)
        restricted_mono.append(images)

    annihilator_bank = {}
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            support = tuple(sorted((ident, a, b)))

            # Right-column gauges preserve a left annihilator. Retain
            # one operator representative, never submit its gauge orbit.
            gauges = [
                tuple(sorted(table[g][inverses[root]]
                             for g in support))
                for root in support
            ]
            if support != min(gauges):
                continue
            if any(relative_order[g][h] == 2
                   for at, g in enumerate(support)
                   for h in support[:at]):
                continue
            left = {table[inverses[g]][h]
                    for g in support for h in support if g != h}
            right = {table[g][inverses[h]]
                     for g in support for h in support if g != h}
            if len(left) != 6 or len(right) != 6:
                continue
            u = table[inverses[support[0]]][support[1]]
            v = table[inverses[support[0]]][support[2]]
            if table[u][v] == table[v][u] or not spans_group(support):
                continue

            images = [
                restricted_mono[support[0]][c]
                ^ restricted_mono[support[1]][c]
                ^ restricted_mono[support[2]][c]
                for c in range(normal_width)
            ]
            basis = coefficient_kernel(images)
            if not basis:
                continue
            values = [0]
            # Truncation only omits proposals; it never changes a gate.
            for word in basis[:5]:
                values += [old ^ word for old in values]
            for functional in values[1:]:
                image = 0
                for c in bits(functional):
                    image ^= images[c]
                assert image == 0
                annihilator_bank.setdefault(functional, []).append(support)

    catalog = [
        (functional, tuple(supports))
        for functional, supports in annihilator_bank.items()
        if len(supports) >= 2
    ]
    if not catalog:
        raise ValueError("no distinct cubic operators share a verified annihilator")

    def catalog_key(item):
        functional, supports = item
        weight = functional.bit_count()
        quotient_shapes = set()
        for support in supports:
            coordinates = [G.elements[g][2] for g in support]
            quotient_shapes.add(min(
                tuple(sorted((t-phase) % 8 for t in coordinates))
                for phase in range(8)))
        preference = (
            (-len(supports), weight),
            (weight, -len(supports)),
            (abs(weight-12), -len(supports)),
            (-weight, -len(supports)),
            (-len(quotient_shapes), -len(supports)),
        )[model]
        return preference + (functional,)

    catalog.sort(key=catalog_key)
    a_frames = {}

    def fresh_a(trial):
        # Five different bank objectives cross four B prescriptions.
        # Every actual operator pair is independently chosen and screened.
        limit = min(256, len(catalog))
        turn = (31*topology + 43*cohort + 5*trial + model) % limit
        functional, options = catalog[turn]
        selected = rng.sample(options, 2)
        grid = [[[G.elements[g] for g in support]
                 for support in selected]]
        a_frames[grid_signature(grid)] = (functional, options)
        return grid

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

    def merit(record):
        # Sixteen is the measured bottleneck, not a certified distance.
        # Once that diagnostic threshold ties, protect dimension gain
        # before rewarding an optimistic larger component reading.
        rank_preference = (
            -record["rank"] if record["side"] == "A"
            else record["rank"])
        return (tuple(-x for x in record["small"]),
                min(16, record["d"]), rank_preference,
                min(24, record["d"]),
                tuple(-x for x in record["cycles"]))

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
            if stats is None or stats[0] != (0, 0):
                return None
            rank = gf2_rank(rows)
            constraint = None
            if side == "A":
                constraint = a_frames.get(signature)
                if constraint is None:
                    raise ValueError("A proposal lost its annihilator frame")
                functional, options = constraint
                expanded = expand_functional(functional)
                assert expanded
                assert all((column & expanded).bit_count() % 2 == 0
                           for column in columns)
                if not q-24 <= rank <= q-8:
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
                "side": side, "constraint": constraint,
            }

        def diagnose(record, trials, offset):
            diagnostic = random.Random(seed+offset)
            reading, word = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            if word is not None:
                bank.add(word)
                if reading < record["d"]:
                    record["d"], record["word"] = reading, word
            bank.update(record["local_words"])
            if len(bank) > 128:
                retained = sorted(
                    bank, key=lambda w: (w.bit_count(), w))[:96]
                bank.clear()
                bank.update(retained)
            refresh(record)
            if record["word"] is not None:
                assert survives(record, record["word"])

        for trial in range(64 if side == "A" else 32):
            grid = (fresh_a(trial) if side == "A" else
                    [[fresh_cell(count, profiles[j])
                      for j, count in enumerate(row)]
                     for row in support])
            record = inspect(grid, trial)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: (
            r["small"], -min(20, r["localized"]),
            r["rank"] if side == "A" else -r["rank"],
            r["cycles"]))
        selected = records[:8]
        if not selected:
            raise ValueError("empty affine component shortlist")
        for record in selected:
            diagnose(record, 8, 0)
        selected.sort(key=score, reverse=True)

        # Two independent chains change actual labels in columns used
        # by verified words. No symmetry images are submitted as codes.
        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(8):
                refresh(current)
                active = (sorted({c//q for c in bits(current["word"])})
                          if current["word"] is not None else [])
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                if side == "A":
                    if step % 4 == 3:
                        # Also explore a genuinely different constraint,
                        # not just labels inside one annihilator class.
                        grid = fresh_a(100+20*chain+step)
                    else:
                        functional, options = current["constraint"]
                        j = rng.choice(active if active else (0, 1))
                        other = tuple(sorted(
                            G.idx(g) for g in grid[0][1-j]))
                        available = [
                            option for option in options if option != other
                        ]
                        chosen = rng.choice(available)
                        grid[0][j] = [G.elements[g] for g in chosen]
                        a_frames[grid_signature(grid)] = (
                            functional, options)
                else:
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
                        grid[0][j] = fresh_cell(3, profiles[j])
                proposed = inspect(grid, 100+8*chain+step)
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

    def span_pivots(rows):
        pivots = {}
        for source in rows:
            word = source
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside_span(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def localized_logicals(a, b, hx, hz, n):
        """Actual logical witnesses in the original stabilizer quotient.

        A miss returns n+1, never a lower bound. Adding original
        stabilizers preserves commutation and nontriviality.
        """
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span_pivots(stabilizers)
            sources = [record["word"]] + record["local_words"]
            candidates = set()
            for word in sources:
                if word is None:
                    continue
                for location in range(3 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j+location)*q) for j in range(2))
                    else:
                        actual = word << (3*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside_span(actual, pivots):
                        candidates.add(actual)

            reading = n+1
            diagnostic = random.Random(
                diagnostic_seed + (4100009 if kind == "Z" else 5100011))
            for actual in sorted(
                    candidates, key=lambda w: (w.bit_count(), w))[:4]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                nearby = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:32]
                for attempt in range(96):
                    first, second = diagnostic.sample(nearby, 2)
                    candidate = actual ^ first ^ second
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                for row in reducers:
                    candidate = actual ^ row
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                assert actual and outside_span(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

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
        if not (416 <= k <= 448 and 11*n <= 40*k < 13*n):
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
        logicals = localized_logicals(a, b, hx, hz, n)
        detected = min(16, a["d"], b["d"], *logicals)
        key = (
            tuple(-(x+y) for x, y in zip(a["small"], b["small"])),
            k*detected*detected, detected,
            min(24, a["d"])+min(24, b["d"]),
            -a["cycles"][0]-b["cycles"][0],
        )
        answer = {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (topology, model, cohort),
            "coset_profiles": plans,
            "common_left_functional": a["constraint"][0],
            "component_ranks": (a["rank"], b["rank"]),
            "component_readings": (a["d"], b["d"]),
            "component_small_kernels": (a["small"], b["small"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "verified_logical_upper_readings": logicals,
            "pairing_merit": key,
        }
        if best is None or key > best[0]:
            best = (key, answer, hx, hz)

    if best is None:
        raise ValueError("common-left pairings missed verified 416..448 dimension gate")
    key, answer, hx, hz = best
    assert css_ok(hx, hz)
    return answer


def NEW_CTX(rng, G, idx):
    """Twenty balanced multirow cards; all syndromes use verified blocks."""
    from functools import lru_cache

    q = G.q
    assert q == 36 and A_SHAPE == (2, 5) and B_SHAPE == (3, 7)
    card, cohort = idx % 20, idx // 20
    zeros, model = divmod(card, 5)
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xF9C42537 + 1000003*idx))
    mask = (1 << q)-1
    ident = G.idx(G.e)
    shapes = {"A": A_SHAPE, "B": B_SHAPE}

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length()-1
            word ^= low

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    orders = []
    for g in range(q):
        value, count = ident, 0
        while True:
            value = table[value][g]
            count += 1
            if value == ident:
                break
            assert count <= q
        orders.append(count)
    relative = [[orders[table[inverse[g]][h]] for h in range(q)]
                for g in range(q)]

    # Obtain BOTH conventions from the scaffold, not a new group lift.
    gallery = [[[g] for g in G.elements]]
    left, _, _ = lifted_matrices(G, gallery, [[[]]])
    _, right, _ = lifted_matrices(G, [[[]]], gallery)
    blocks = {
        "A": [[(row >> (j*q)) & mask for row in left]
              for j in range(q)],
        "B": [[(row >> (j*q)) & mask for row in right]
              for j in range(q)],
    }
    del left, right

    def support(side):
        if side == "A":
            heavier = set(rng.sample(range(5), 2))
            upper = [2 if j in heavier else 1 for j in range(5)]
            grid = [upper, [3-a for a in upper]]
            assert sorted(map(sum, grid)) == [7, 8]
            return grid

        degrees = (3, 4, 4, 4, 4, 4, 4)
        options = []
        for degree in degrees:
            choices = [
                (a, b, degree-a-b)
                for a in range(4) for b in range(4)
                if 0 <= degree-a-b <= 3
                and sum(v > 0 for v in (a, b, degree-a-b)) >= 2
            ]
            rng.shuffle(choices)
            options.append(choices)

        @lru_cache(None)
        def feasible(j, a, b, c, z):
            if min(a, b, c, z) < 0:
                return False
            if j == 7:
                return (a, b, c, z) == (0, 0, 0, 0)
            if a+b+c != sum(degrees[j:]) or z > 7-j:
                return False
            return any(
                feasible(j+1, a-v[0], b-v[1], c-v[2],
                         z-sum(x == 0 for x in v))
                for v in options[j])

        state = (9, 9, 9, zeros)
        assert feasible(0, *state)
        grid = [[], [], []]
        for j in range(7):
            choices = []
            for v in options[j]:
                nxt = tuple(state[t]-v[t] for t in range(3))
                nxt += (state[3]-sum(x == 0 for x in v),)
                if feasible(j+1, *nxt):
                    choices.append((v, nxt))
            v, state = rng.choice(choices)
            for i in range(3):
                grid[i].append(v[i])
        assert list(map(sum, grid)) == [9, 9, 9]
        assert [sum(row[j] for row in grid) for j in range(7)] == list(degrees)
        return grid

    def generated(ids):
        roots = [table[inverse[ids[0]]][g] for g in ids[1:]]
        reached, pending = {ident}, [ident]
        for value in pending:
            for root in roots:
                image = table[value][root]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached)

    def fresh_cell(count, side, i, j):
        if count <= 1:
            return rng.sample(range(q), count)
        policy = (model + int(side == "B") + cohort) % 5
        best, best_cost = None, None
        for _ in range(96):
            ids = rng.sample(range(q), count)
            rels = [relative[g][h]
                    for at, g in enumerate(ids) for h in ids[:at]]
            # Avoid involution differences, which create doubled geometry.
            if 2 in rels:
                continue
            dl = {table[inverse[g]][h]
                  for g in ids for h in ids if g != h}
            dr = {table[g][inverse[h]]
                  for g in ids for h in ids if g != h}
            if len(dl) != count*(count-1) or len(dr) != count*(count-1):
                continue
            span = generated(ids) if count == 3 else 0
            if policy == 0:
                cost = 0
            elif policy == 1:
                cost = rels.count(3) + (6 if count == 3 and span < q else 0)
            elif policy == 2:
                cost = 3*rels.count(4)
            elif policy == 3:
                cost = (rels.count(4) if count == 2 else
                        8*int(span < q) + rels.count(3))
            else:
                preferred = 3 if (i+j+int(side == "B")) % 2 else 4
                cost = sum(order != preferred for order in rels)
            if best_cost is None or cost < best_cost:
                best, best_cost = ids, cost
        if best is None:
            raise ValueError("distinct relative-order cell recipe exhausted")
        return best

    def assemble(grid, side):
        rows_count, width_blocks = shapes[side]
        rows = [0]*(rows_count*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        columns = [0]*(width_blocks*q)
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return rows, columns

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def dependencies(columns):
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
        return basis

    def small_word(columns):
        singles, pairs = {}, {}
        for c, value in enumerate(columns):
            bit = 1 << c
            if not value:
                return bit
            old = singles.get(value)
            if old is not None:
                return bit ^ old
            singles[value] = bit
        for c, value in enumerate(columns):
            for d in range(c):
                key = value ^ columns[d]
                word = (1 << c) | (1 << d)
                old = singles.get(key)
                if old is None:
                    old = pairs.get(key)
                if old is not None:
                    return word ^ old
                pairs[key] = word
        return None

    def cycle_geometry(rows, columns):
        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for column in columns:
            checks = list(bits(column))
            degree = len(checks)
            forced += degree*(degree-1)*(degree-2)//6
            for at, a in enumerate(checks):
                adjacency[a] |= column ^ (1 << a)
                for b in checks[:at]:
                    key = (b, a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        connected = reached == (1 << len(rows))-1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        return connected, (four, max(0, triangles-forced))

    def exact_eight(record):
        """Exact simple Tanner eight-cycles; regular roots per variable block."""
        rows, columns = record["rows"], record["columns"]
        checks = [list(bits(row)) for row in rows]
        neighbors = [list(bits(column)) for column in columns]
        rooted = 0
        for root in range(0, len(columns), q):
            paths = {}
            for c1 in neighbors[root]:
                for v in checks[c1]:
                    if v == root:
                        continue
                    for c2 in neighbors[v]:
                        if c2 == c1:
                            continue
                        for end in checks[c2]:
                            if end in (root, v):
                                continue
                            previous = paths.setdefault(end, [])
                            for d1, u, d2 in previous:
                                if (u != v and c1 not in (d1, d2)
                                        and c2 not in (d1, d2)):
                                    rooted += 1
                            previous.append((c1, v, c2))
        assert q*rooted % 4 == 0
        return q*rooted//4

    def note(record, word):
        if not word:
            return
        assert syndrome(record["columns"], word) == 0
        record["d"] = min(record["d"], word.bit_count())
        record["words"].add(word)
        if len(record["words"]) > 64:
            record["words"] = set(sorted(
                record["words"], key=lambda w: (w.bit_count(), w))[:24])

    def inspect(grid, side):
        rows, columns = assemble(grid, side)
        rank = gf2_rank(rows)
        if rank < len(rows)-6:
            return None
        connected, cycles = cycle_geometry(rows, columns)
        if not connected:
            return None
        record = dict(
            grid=[[list(cell) for cell in row] for row in grid],
            rows=rows, columns=columns, rank=rank, cycles=cycles,
            d=len(columns)+1, words=set(), eight=0,
            signature=tuple(tuple(sorted(cell))
                            for row in grid for cell in row))
        word = small_word(columns)
        record["free4"] = word is None
        note(record, word)
        # Every single-column dependency uses the complete multirow syndrome.
        for j in range(shapes[side][1]):
            basis = dependencies(columns[j*q:(j+1)*q])
            if len(basis) <= 10:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length()-1]
                    note(record, word << (j*q))
            else:
                for at, word in enumerate(basis):
                    note(record, word << (j*q))
                    for other in basis[:at]:
                        note(record, (word ^ other) << (j*q))
        return record

    def challenge(record, diagnostic_seed, trials):
        """Genuine accumulated kernel witnesses, never an exclusion claim."""
        local = random.Random(diagnostic_seed)
        columns = record["columns"]
        order = list(range(len(columns)))
        retained = sorted(record["words"], key=int.bit_count)[:20]
        for turn in range(trials):
            local.shuffle(order)
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
            light = sorted(set(basis+retained), key=int.bit_count)[:20]
            additions = list(light)
            threshold = min(record["d"], light[0].bit_count())+4
            for word in light:
                note(record, word)
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= threshold:
                        note(record, candidate)
                        additions.append(candidate)
            moves = basis+retained
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(record, word)
                additions.append(word)
            retained = sorted(
                set(retained+additions+list(record["words"])),
                key=lambda w: (w.bit_count(), w))[:24]
            if record["d"] <= 4:
                break
        assert record["words"]

    def merit(record):
        multiplicity = sum(w.bit_count() == record["d"]
                           for w in record["words"])
        return (record["free4"], min(22, record["d"]), -multiplicity,
                -record["cycles"][0], -record["cycles"][1],
                -record["eight"], record["rank"])

    def component_pool(side, restart):
        offset = 100003*int(side == "B")+10000019*restart
        seen, records, historical = set(), [], set()

        def remember(grid):
            signature = tuple(tuple(sorted(cell))
                              for row in grid for cell in row)
            if signature in seen:
                return None
            seen.add(signature)
            record = inspect(grid, side)
            if record is not None:
                records.append(record)
            return record

        def draw():
            counts = support(side)
            return [[fresh_cell(count, side, i, j)
                     for j, count in enumerate(row)]
                    for i, row in enumerate(counts)]

        for _ in range(32):
            remember(draw())
        records.sort(key=lambda r: (
            not r["free4"], r["cycles"], -r["rank"]))
        selected = records[:6]
        if not selected:
            return []
        for record in selected:
            challenge(record, seed+offset, 8)
            historical.update(record["words"])
        current = max(selected, key=merit)

        for step in range(18):
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            if step % 6 == 5:
                grid = draw()
            else:
                word = rng.choice(sorted(
                    current["words"], key=int.bit_count)[:6])
                active = sorted({c//q for c in bits(word)})
                changed = set()
                for _ in range(2 if step % 4 == 3 else 1):
                    available = [j for j in active if j not in changed]
                    if not available:
                        available = [j for j in range(shapes[side][1])
                                     if j not in changed]
                    if not available:
                        break
                    j = rng.choice(available)
                    changed.add(j)
                    for i in range(shapes[side][0]):
                        grid[i][j] = fresh_cell(
                            len(grid[i][j]), side, i, j)
            proposed = remember(grid)
            if proposed is None:
                continue
            if current["free4"] and not proposed["free4"]:
                continue
            if proposed["cycles"][0] > current["cycles"][0]+2*q:
                continue
            for known in historical:
                if syndrome(proposed["columns"], known) == 0:
                    note(proposed, known)
            challenge(proposed, seed+offset+700001+7919*step, 6)
            selected.append(proposed)
            historical.update(proposed["words"])
            historical = set(sorted(historical, key=int.bit_count)[:80])
            if merit(proposed) >= merit(current):
                current = proposed
            elif (proposed["free4"] == current["free4"]
                  and proposed["d"] == current["d"]
                  and rng.random() < 0.10):
                current = proposed

        unique = {record["signature"]: record for record in selected}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:3]
        for record in finalists:
            for known in historical:
                if syndrome(record["columns"], known) == 0:
                    note(record, known)
            challenge(record, seed+offset+1700021, 48+8*(cohort % 3))
            challenge(record, seed+offset+2900003, 32)
            record["eight"] = exact_eight(record)
        finalists.sort(key=merit, reverse=True)
        return finalists

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

    def logical_challenge(a, b, hx, hz, n):
        """Actual non-stabilizer embeddings in original qubit coordinates."""
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(record["words"], key=int.bit_count)[:6]:
                for location in range(7 if kind == "Z" else 5):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((7*j+location)*q) for j in range(5))
                    else:
                        actual = word << (location*7*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            local = random.Random(
                seed+(4100011 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:4]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best_word = actual
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            candidate = partial ^ second
                            if candidate.bit_count() < best_word.bit_count():
                                best_word = candidate
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    for restart in range(3):
        aa = component_pool("A", restart)
        bb = component_pool("B", restart)
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            ab[0]["free4"] and ab[1]["free4"],
            min(22, ab[0]["d"], ab[1]["d"]),
            -ab[0]["cycles"][0]-ab[1]["cycles"][0]),
            reverse=True)
        survivors = []
        for a, b in pairs:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) == 12 <= W_MAX
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 41*q == 1476
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 432 and 275*n <= 1000*k < 325*n):
                continue

            actual_a = [
                sum(((hx[(i*7)*q+t] >> (j*7*q)) & mask) << (j*q)
                    for j in range(5))
                for i in range(2) for t in range(q)]
            actual_b = [
                hz[alpha*q+t] & ((1 << (7*q))-1)
                for alpha in range(3) for t in range(q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            assert max(row.bit_count() for row in hx+hz) == 12
            logicals = logical_challenge(a, b, hx, hz, n)
            detected = min(22, a["d"], b["d"], *logicals)
            key = (
                a["free4"] and b["free4"],
                k*detected*detected, detected,
                -a["cycles"][0]-b["cycles"][0],
                -a["cycles"][1]-b["cycles"][1],
                -a["eight"]-b["eight"])
            survivors.append((key, A, B, hx, hz, k, a, b, logicals))
        if not survivors:
            continue
        key, A, B, hx, hz, k, a, b, logicals = max(
            survivors, key=lambda entry: entry[0])
        assert css_ok(hx, hz)
        return dict(
            idx=idx, cells={"A": A, "B": B}, k=k,
            card=(zeros, model, cohort),
            component_ranks=(a["rank"], b["rank"]),
            component_upper_readings=(a["d"], b["d"]),
            component_kernel4_free=(a["free4"], b["free4"]),
            component_cycles=(a["cycles"], b["cycles"]),
            component_eight_cycles=(a["eight"], b["eight"]),
            verified_logical_upper_readings=logicals,
            pairing_merit=key, target_cell=0.300)
    raise ValueError("balanced multirow pairings missed the exact 0.300 rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected multirow cell without additional draws."""
    cell = list(ctx["cells"][side][i][j])
    assert 0 <= len(cell) <= 3
    assert len({G._key(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on the SL2(F5) bed; current slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Full-generating cubic coset mixing and localized/surviving-witness repair remove the order-200 bed's weight-sixteen bottleneck or increase measured dimension; all five current measured codes retained sector-one X minima of sixteen and k=400.
# REFUTED: Common-left dimension gain alone preserves the order-200 bed's weight-sixteen reading; current slots 1, 6, 11, 16 and 21 had k=432 but screened at weight eight, while the five measured k=416 selections retained weight sixteen.
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
