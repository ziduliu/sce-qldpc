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
CONCEPT = ("Hypothesis: on ((F8,+) semidirect C7) x C2, order 112, A=B=(3,2), balanced column-seven supports and independently challenged transpose-component kernels raise five-code fitness above 351.6 in cell 0.100. Failure refutes the measurement hypothesis; readings remain provisional upper bounds.")
GROUP = ("Jump from the parent's order-200 F25-additive semidirect C8 to the order-112 affine-binary direct product. Multiplication by a primitive element of F8 gives a faithful order-seven action; the independent C2 is central. Centre two and sixteen conjugacy classes match the invariants of the strong order-112 bed, whose (3,2)x(3,2) reading reached 351.6. Matching invariants are motivation, not a distance certificate.")
SHAPES = ("A=B=(3,2), q=112: n=13*112=1456, k>=112 and guaranteed rate 1/13 target the open-ended 0.100 cell; q_max=115. Both distance-bearing transpose components have 336 variables and 224 checks, nominal rate one-third. Parent weight-sixteen minima occupied sector-one Acol/Bcol strips. The new orientation moves analogous strip families to sector two and replaces both component lengths and rates. Actual full ranks must retain k/n<0.125.")
FILLING = ("Twenty cards cross four positive support profiles with five binding complement/central prescriptions. Both sides have row totals (4,5,5) and column totals (7,7), so HX and HZ maxima are twelve. Distinct complement coordinates, native bidirectional Sidon differences and relative full generation for dense cells enrich initialization and remain binding after repair. Every transpose block comes from lifted_matrices. Cycle selection precedes exact component kernel-through-four gates, accumulated circuit cancellation and independent finalist challenges. Witness-directed whole-cell repair changes actual relations, not submitted symmetry images. Final native agreement, ranks, weights, CSS and original-span-verified sector-two logical reductions decide pairing. Finite searches do not certify distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r010_AffineF8xC2_32x32_balanced_transpose_challenge_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Affine F8 x C2 requires order 112")
    # F8 = F2[x]/(x^3+x+1), encoded by three polynomial bits.
    # (u,t,z): affine map v -> x^t*v+u and an independent C2.
    powers = (1,2,4,3,6,7,5)

    def field_mul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return out & 7

    actions = tuple(tuple(field_mul(p, u) for u in range(8))
                    for p in powers)
    els = [(u,t,z) for z in range(2)
           for t in range(7) for u in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 7, int(g[2]) % 2)

    def mul(g, h):
        u,t,z = red(g)
        v,s,w = red(h)
        return (u ^ actions[t][v], (t+s) % 7, z ^ w)

    def inv(g):
        u,t,z = red(g)
        ti = (-t) % 7
        return (actions[ti][u], ti, z)

    def idx(g):
        u,t,z = red(g)
        return 56*z+8*t+u

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 2)               # (ra, ca)
B_SHAPE = (3, 2)               # (rb, cb)


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
    """Twenty balanced affine-binary cards; only native lifts supply blocks."""
    import math

    q = G.q
    assert q == 112 and A_SHAPE == B_SHAPE == (3, 2)
    card, cohort = idx % 20, idx // 20
    topology, model = divmod(card, 5)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    width = 3*q
    ident = G.idx(G.e)

    # Four support types modulo the obvious row/column symmetries.
    profiles = (
        ((2,2), (2,3), (3,2)),
        ((1,3), (2,3), (4,1)),
        ((2,2), (1,4), (4,1)),
        ((1,3), (3,2), (3,2)),
    )
    sa = [list(row) for row in profiles[topology]]
    sb = [list(row) for row in profiles[(topology+model+1) % 4]]
    assert [sum(row) for row in sa] == [4,5,5]
    assert [sum(row) for row in sb] == [4,5,5]
    assert [sum(row[j] for row in sa) for j in range(2)] == [7,7]
    assert [sum(row[j] for row in sb) for j in range(2)] == [7,7]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

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

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length()-1
            word ^= low

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    # Unit probes isolate the actual transposed left/right monomials.
    # No independently written regular-representation matrix is used.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start+16)))
        monomials = [[[g] for g in ids]]
        unit = [[[ident]]]
        _, ar, _ = lifted_matrices(P, monomials, unit)
        br, _, _ = lifted_matrices(P, unit, monomials)
        cut = len(ids)*q
        for j in range(len(ids)):
            blocks["A"].append([
                (ar[j*q+r] >> cut) & mask for r in range(q)])
            blocks["B"].append([
                (br[j*q+r] >> cut) & mask for r in range(q)])

    generation_cache = {}

    def generates(roots):
        key = tuple(sorted(roots))
        if key not in generation_cache:
            reached, pending = {ident}, [ident]
            for value in pending:
                for root in key:
                    image = table[value][root]
                    if image not in reached:
                        reached.add(image)
                        pending.append(image)
            generation_cache[key] = len(reached) == q
        return generation_cache[key]

    def legal(cell):
        if len(cell) != len(set(cell)):
            return False
        if len(cell) < 2:
            return True
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        wanted = len(cell)*(len(cell)-1)
        if len(left) != wanted or len(right) != wanted:
            return False
        if len(cell) >= 3:
            roots = [table[inverse[cell[0]]][g] for g in cell[1:]]
            if not generates(roots):
                return False
        return True

    def fresh_cell(count, policy):
        if count == 1:
            return [rng.randrange(q)]
        # AP versus non-AP complement triples; quartics use complements.
        # These prescriptions remain binding after whole-cell repair.
        for _ in range(2048):
            phase = rng.randrange(7)
            scale = rng.choice((1,2,4))
            if count == 2:
                ts = (0, (1,3,1,3,1)[policy])
            elif count == 3:
                ts = ((0,1,3) if policy in (0,3)
                      else (0,1,2))
                if policy == 4:
                    ts = tuple(sorted(rng.sample(range(7), 3)))
            else:
                omitted = ((0,1,2) if policy in (0,2)
                           else (0,1,3))
                ts = tuple(t for t in range(7) if t not in omitted)
                if policy == 4:
                    ts = tuple(sorted(rng.sample(range(7), 4)))
            ts = tuple((phase+scale*t) % 7 for t in ts)
            ones = count//2 if policy % 2 == 0 else 1
            if policy == 3 and count == 2:
                ones = 0
            if policy == 4:
                ones = rng.randrange(1, count)
            colored = set(rng.sample(range(1, count), ones))
            cell = [G.idx((rng.randrange(8), t, int(at in colored)))
                    for at, t in enumerate(ts)]
            if legal(cell):
                return cell
        raise ValueError("binding affine-binary cell recipe exhausted")

    def fresh(support, policy):
        return [[fresh_cell(count, policy) for count in row]
                for row in support]

    def inspect(grid, side):
        # Transpose components: two lifted check blocks, three variable blocks.
        rows = [0]*(2*q)
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[j*q+r] ^= value << (i*q)
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if not all(columns) or len(set(columns)) != width:
            return None
        assert [columns[i*q].bit_count() for i in range(3)] == [4,5,5]

        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for column in columns:
            ns = list(bits(column))
            degree = len(ns)
            forced += degree*(degree-1)*(degree-2)//6
            for at, a in enumerate(ns):
                adjacency[a] |= column ^ (1 << a)
                for b in ns[:at]:
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
        if reached != (1 << len(rows))-1:
            return None
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        excess = max(0, triangles-forced)
        cost = (500,900,1500,2400,3600)[model]*four
        cost += (1,2,4,8,12)[model]*excess
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cycles=(four, excess), cost=cost, signature=signature,
                    words=set(), d=width+1)

    def prepare(rec):
        # Exact component rank and exclusion of every kernel word <=4.
        rec["rank"] = gf2_rank(rec["rows"])
        if rec["rank"] < 2*q-12:
            return False
        singles = set(rec["columns"])
        seen = set()
        for i, value in enumerate(rec["columns"]):
            for other in rec["columns"][:i]:
                pair = value ^ other
                if pair in singles or pair in seen:
                    return False
                seen.add(pair)
        return True

    def challenge(rec, trials, offset):
        """Accumulate genuine circuits; independent searches only lower d."""
        diagnostic = random.Random(seed+offset)
        columns = rec["columns"]
        order = list(range(width))
        retained = sorted(rec["words"],
                          key=lambda w: (w.bit_count(), w))[:24]

        def note(word, found):
            if not word:
                return
            weight = word.bit_count()
            rec["d"] = min(rec["d"], weight)
            if weight <= rec["d"]+8:
                found.append(word)

        for trial in range(trials):
            diagnostic.shuffle(order)
            if retained and trial % 3 == 1:
                active = list(bits(diagnostic.choice(retained[:6])))
                front = diagnostic.sample(active, max(1, len(active)//2))
                selected = set(front)
                order = front+[v for v in order if v not in selected]
            pivots, basis, independent = {}, [], []
            found = []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        independent.append(c)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word, found)
            if not basis:
                raise ValueError("transpose component unexpectedly has no kernel")
            light = sorted(set(retained+basis),
                           key=lambda w: (w.bit_count(), w))[:20]
            for at, word in enumerate(basis):
                for other in basis[:at]:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= rec["d"]+4:
                        note(candidate, found)

            moves = basis+retained
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(word, found)

            # Finite hashed four-circuit cancellation, not exhaustive decoding.
            if trial % 4 == 3 and len(basis) >= 4:
                sampled = diagnostic.sample(
                    independent, min(9, len(independent)))
                hash_mask = sum(1 << c for c in sampled)
                buckets = {}
                for at, first in enumerate(basis):
                    for second in basis[:at]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for other in bucket[:12]:
                            note(pair ^ other, found)
                        if len(bucket) < 12:
                            bucket.append(pair)
            if len(basis) >= 3:
                for _ in range(96):
                    a,b,c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c, found)
            retained = sorted(set(retained+light+found),
                              key=lambda w: (w.bit_count(), w))[:24]
            if rec["d"] <= 8:
                break
        assert retained
        rec["words"].update(retained)
        rec["words"] = set(sorted(
            rec["words"], key=lambda w: (w.bit_count(), w))[:24])
        for word in rec["words"]:
            assert word and syndrome(columns, word) == 0
        rec["d"] = min(rec["d"], min(map(int.bit_count, rec["words"])))

    def merit(rec):
        multiplicity = sum(w.bit_count() == rec["d"] for w in rec["words"])
        return (min(80, rec["d"]), -multiplicity,
                -rec["cost"], rec["rank"])

    def pool(side, support):
        side_id = int(side == "B")
        policy = (model+2*side_id+cohort) % 5
        offset = 100003*side_id
        bank, seen = {}, set()

        def remember(rec):
            if rec is None or rec["signature"] in seen:
                return False
            seen.add(rec["signature"])
            bank[rec["signature"]] = rec
            if len(bank) > 16:
                worst = max(bank, key=lambda key: bank[key]["cost"])
                del bank[worst]
            return True

        starts = []
        for _ in range(40):
            rec = inspect(fresh(support, policy), side)
            if remember(rec):
                starts.append(rec)
        starts.sort(key=lambda r: r["cost"])
        if not starts:
            raise ValueError("empty connected affine-binary geometry pool")
        positions = [(i,j) for i in range(3) for j in range(2)]
        for initial in starts[:2]:
            current = initial
            for step in range(48):
                grid = copy_grid(current["grid"])
                i,j = rng.choice(positions)
                grid[i][j] = fresh_cell(support[i][j], policy)
                rec = inspect(grid, side)
                if rec is None or rec["signature"] in seen:
                    continue
                delta = rec["cost"]-current["cost"]
                temperature = q*(12*(1-step/48.0)**2+0.2)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta/temperature)):
                    current = rec
                    remember(rec)

        selected = []
        for rec in sorted(bank.values(), key=lambda r: r["cost"]):
            if prepare(rec):
                challenge(rec, 8, offset)
                selected.append(rec)
            if len(selected) == 6:
                break
        # A finite geometry shortlist is not allowed to erase a legal card.
        for _ in range(96 if not selected else 0):
            rec = inspect(fresh(support, policy), side)
            if rec is not None and remember(rec) and prepare(rec):
                challenge(rec, 8, offset+300007)
                selected.append(rec)
                if len(selected) == 4:
                    break
        if not selected:
            raise ValueError("no component passed exact rank/kernel-four gates")

        selected.sort(key=merit, reverse=True)
        for rec in selected[:3]:
            challenge(rec, 24, offset+700001)
        current = max(selected, key=merit)
        historical = set().union(*(r["words"] for r in selected))
        for step in range(16):
            grid = copy_grid(current["grid"])
            word = rng.choice(sorted(current["words"],
                                     key=int.bit_count)[:6])
            active = sorted({c//q for c in bits(word)})
            i = rng.choice(active)
            j = rng.randrange(2)
            grid[i][j] = fresh_cell(support[i][j], policy)
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            if rec["cycles"][0] > current["cycles"][0]+q:
                continue
            if not prepare(rec):
                continue
            for known in historical:
                if syndrome(rec["columns"], known) == 0:
                    rec["words"].add(known)
                    rec["d"] = min(rec["d"], known.bit_count())
            challenge(rec, 8, offset+1000003+7919*step)
            historical.update(rec["words"])
            historical = set(sorted(historical, key=int.bit_count)[:96])
            selected.append(rec)
            if merit(rec) >= merit(current):
                current = rec

        unique = {r["signature"]: r for r in selected}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:3]
        for rec in finalists:
            challenge(rec, 64+8*(cohort % 3), offset+1700009)
            challenge(rec, 32, offset+2700011)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

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
        """Actual sector-two logical witnesses modulo ORIGINAL stabilizers."""
        readings = []
        for side, rec, opposite, stabilizers in (
                ("A", a, hz, hx), ("B", b, hx, hz)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(rec["words"], key=int.bit_count)[:5]:
                for fixed in range(3):
                    actual = 0
                    for block in range(3):
                        location = (3*block+fixed if side == "A"
                                    else 3*fixed+block)
                        actual |= ((word >> (block*q)) & mask) \
                                  << ((4+location)*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            diagnostic = random.Random(
                seed+(3100009 if side == "A" else 4100011))
            reducers = list(stabilizers)
            diagnostic.shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:28]
                    winner = actual
                    for at, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:at]:
                            proposed = partial ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool("A", sa), pool("B", sb)
    best = None
    for a in aa:
        for b in bb:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            assert n == 13*q == 1456
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= q and 8*k < n):
                continue
            actual_a = [
                sum(((hz[(3*j)*q+r] >> ((4+3*i)*q)) & mask)
                    << (i*q) for i in range(3))
                for j in range(2) for r in range(q)]
            actual_b = [
                (row >> (4*q)) & ((1 << width)-1)
                for row in hx[:2*q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logicals = logical_challenge(a, b, hx, hz, n)
            detected = min(72, a["d"], b["d"], *logicals)
            key = (k*detected*detected, detected,
                   min(80,a["d"])+min(80,b["d"]),
                   -a["cost"]-b["cost"])
            if best is None or key > best[0]:
                best = (key, a, b, hx, hz, k, logicals)
    if best is None:
        raise ValueError("affine-binary pairings missed open 0.100 rate gate")
    key, a, b, cached_hx, cached_hz, k, logicals = best
    A = [[[G.elements[g] for g in cell] for cell in row]
         for row in a["grid"]]
    B = [[[G.elements[g] for g in cell] for cell in row]
         for row in b["grid"]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1456
    assert [[len(c) for c in row] for row in A] == sa
    assert [[len(c) for c in row] for row in B] == sb
    assert all(len(set(cell)) == len(cell)
               for grid in (A,B) for row in grid for cell in row)
    assert max(row.bit_count() for row in hx+hz) == 12
    assert css_ok(hx, hz)
    return dict(
        idx=idx, cells={"A": A, "B": B}, k=k,
        card=(topology,model,cohort), target_cell=0.100,
        component_ranks=(a["rank"],b["rank"]),
        component_upper_readings=(a["d"],b["d"]),
        component_cycles=(a["cycles"],b["cycles"]),
        verified_logical_upper_readings=logicals,
        pairing_merit=key,
    )


def CELL(rng, G, side, i, j, ctx):
    """Return the complete balanced, independently challenged native cell."""
    cell = list(ctx["cells"][side][i][j])
    assert 1 <= len(cell) <= 4
    assert len({G._key(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on the SL2(F5) bed; current slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Full-generating cubic coset mixing and localized/surviving-witness repair remove the order-200 bed's weight-sixteen bottleneck or increase measured dimension; all five current measured codes retained sector-one X minima of sixteen and k=400.
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
