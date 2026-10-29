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
CONCEPT = ("On C13 semidirect_3 C3, order 39, A=(2,5), B=(2,6), cubic column degrees and independently challenged concentrated-support kernels should yield five measured readings >=12 and improve cell 0.350 above fitness 47.8. This is falsifiable measurement guidance, not a distance certificate.")
GROUP = ("Jump from the order-200 F25-additive semidirect C8 group to C13:C3, order 39. The action is associative because 3 has exact order three modulo 13; different orders prove nonisomorphism. Related order-57 and order-111 metacyclic beds have strong readings, but do not establish distance transfer. This order-39 shape bed is absent from BEDS.")
SHAPES = ("A=(2,5), B=(2,6), q=39 give n=34*39=1326 and k>=12*39=468, guaranteed rate 6/17=0.352941 in cell 0.350; q_max=44. Components have 195 and 234 variables with 78 checks each, nominal rates 3/5 and 2/3. Parent weight-16 minimum X families used sector-1 single-Acol strips across all B columns. Replace both single-row kernels with two-row cubic-column components and retain exact full rates in [0.325,0.375).")
FILLING = ("Twenty cards cross A zero budgets 0..4 with B zero budgets 0..3. Every column has three terms; A row totals (7,8), B row totals (9,9) give HX<=11 and HZ<=12. Concentrated trinomials must generate the whole group relatively; binomial policies distinguish order-13 and order-three ratios. All binary components come from verified dummy-side lifts. Exact kernel-through-four statistics, cycle geometry, accumulated genuine kernel words and witness-directed label repair select independent fillings. Every returned finalist receives fresh deeper diagnostics that can only lower readings. Full native ranks, component equality, weights and CSS gate submissions. Sampled component readings do not certify quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r035_C13C3_action3_25x26_cubic_concentration_challenge_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13 semidirect C3 requires order 39")
    # (a,b)=x^a*y^b; y*x*y^-1=x^3.
    action = (1,3,9)
    els = [(a,b) for b in range(3) for a in range(13)]

    def red(g):
        return (int(g[0]) % 13, int(g[1]) % 3)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a+action[b]*c) % 13, (b+d) % 3)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 13, bi)

    def idx(g):
        a,b = red(g)
        return 13*b+a

    return dict(elements=els, ident=(0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _legacy_stride3_ctx(rng, G, idx):
    """Unused order-200 experiment; NEW_CTX below replaces it."""
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
                "rank": gf2_rank(rows), "localized": reading,
                "local_words": words, "d": reading, "word": word,
                "short_orbits": {},
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

        for trial in range(48 if side == "B" else 24):
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
                    if rng.random() < 0.75:
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

    aa, bb = component_pool("A"), component_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        tuple(-(x+y) for x, y in
              zip(ab[0]["small"], ab[1]["small"])),
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]/(2*q)-ab[1]["cycles"][0]/(3*q)),
        reverse=True)

    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 7*q == 1400
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= 2*q and 11*n <= 40*k < 13*n):
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
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (topology, model, cohort),
            "coset_profiles": plans,
            "component_readings": (a["d"], b["d"]),
            "component_small_kernels": (a["small"], b["small"]),
            "component_cycles": (a["cycles"], b["cycles"]),
        }
    raise ValueError("affine pairings missed the exact 0.300 rate gate")


def NEW_CTX(rng, G, idx):
    """Twenty cubic-column support cards on the new order-39 bed."""
    from functools import lru_cache

    q = G.q
    assert q == 39
    card, cohort = idx % 20, idx // 20
    seed = rng.randrange(1 << 62)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        # Native group operations cached only to accelerate verified lifts.
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    P = Probe()

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def syndrome(columns, word):
        answer = 0
        for c in bits(word):
            answer ^= columns[c]
        return answer

    def support(columns, target, zeros):
        """Exact cubic column margins and deliberate concentration budget."""
        @lru_cache(None)
        def feasible(j, total, z):
            if total < 0 or z < 0:
                return False
            if j == columns:
                return total == z == 0
            return any(feasible(j+1, total-a, z-int(a in (0,3)))
                       for a in range(4))

        assert feasible(0, target, zeros)
        upper = []
        total, z = target, zeros
        for j in range(columns):
            options = [
                a for a in range(4)
                if feasible(j+1, total-a, z-int(a in (0,3)))
            ]
            a = rng.choice(options)
            upper.append(a)
            total -= a
            z -= int(a in (0,3))
        return [upper, [3-a for a in upper]]

    sa = support(5, 7, card % 5)
    sb = support(6, 9, card // 5)
    assert [sum(row) for row in sa] == [7,8]
    assert [sum(row) for row in sb] == [9,9]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Three-term cells are full-generating, preventing intentionally
    # commuting concentrated normal-subgroup blocks. Two-term cells
    # distinguish normal order-13 ratios from complement order-three ones.
    policies = (
        (0,0), (0,1), (1,0), (1,1), (2,2)
    )
    a_policy, b_policy = policies[(card+cohort) % 5]
    ratio_pools = {
        0: [g for g in range(q) if G.elements[g][1] == 0
            and g != identity],
        1: [g for g in range(q) if G.elements[g][1] != 0],
        2: [g for g in range(q) if g != identity],
    }

    def generates(cell):
        roots = [table[inverses[cell[0]]][g] for g in cell[1:]]
        reached, pending = {identity}, [identity]
        for g in pending:
            for h in roots:
                image = table[g][h]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    def draw(count, policy):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            return [root, table[root][rng.choice(ratio_pools[policy])]]
        assert count == 3
        for _ in range(256):
            cell = rng.sample(range(q), 3)
            if generates(cell):
                return cell
        # A known generating relative pair makes the sampler total.
        root = rng.randrange(q)
        return [root, table[root][G.idx((1,0))],
                table[root][G.idx((0,1))]]

    def fresh(counts, policy):
        return [[draw(count, policy) for count in row] for row in counts]

    def inspect(grid, side):
        # Empty dummy sides isolate the exact component conventions.
        if side == "A":
            rows, _, _ = lifted_matrices(P, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(P, [[[]]], grid)
        width = len(grid[0])*q
        assert len(rows) == 2*q
        assert all(row >> width == 0 for row in rows)
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        assert all(col.bit_count() == 3 for col in columns)

        singles = set(columns)
        small = width-len(singles)+int(0 in singles)
        pairs = {}
        for i, col in enumerate(columns):
            for other in columns[:i]:
                value = col ^ other
                small += int(value in singles)
                small += pairs.get(value, 0)
                pairs[value] = pairs.get(value, 0)+1

        uses, four = {}, 0
        adjacency = [0]*len(rows)
        for col in columns:
            checks = list(bits(col))
            for i, a in enumerate(checks):
                adjacency[a] |= col ^ (1 << a)
                for b in checks[:i]:
                    key = (b,a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(grid=grid, rows=rows, columns=columns,
                    small=small, cycles=(four, max(0,triangles-width)),
                    rank=gf2_rank(rows), signature=signature,
                    d=width+1, words=[])

    def query(rec, trials, offset):
        """Genuine kernel words; fresh trials only lower accumulated readings."""
        local = random.Random(seed+offset)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = list(rec["words"])

        def note(word, found):
            if not word:
                return
            weight = word.bit_count()
            rec["d"] = min(rec["d"], weight)
            if weight <= rec["d"]+4:
                found.append(word)

        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            found = []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value,word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word, found)
            light = sorted(set(basis+retained), key=int.bit_count)[:20]
            for word in light:
                for other in basis+retained:
                    note(word ^ other, found)
                if len(found) > 128:
                    found = sorted(set(found), key=int.bit_count)[:24]
            moves = basis+retained
            local.shuffle(moves)
            for word in light[:5]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(word, found)
            retained = sorted(set(retained+light+found),
                              key=lambda w: (w.bit_count(),w))[:24]
            if rec["d"] <= 6:
                break
        rec["words"] = retained
        assert retained
        assert rec["d"] == min(w.bit_count() for w in retained)
        assert all(syndrome(columns, w) == 0 for w in retained)

    def merit(rec):
        multiplicity = sum(w.bit_count() == rec["d"] for w in rec["words"])
        return (-rec["small"], rec["d"], -multiplicity,
                -rec["cycles"][0], -rec["cycles"][1], rec["rank"])

    def pool(side, counts, policy):
        offset = 100003*int(side == "B")
        records, seen = [], set()
        for _ in range(48):
            rec = inspect(fresh(counts, policy), side)
            if rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda r: (
            r["small"], r["cycles"], -r["rank"]))
        selected = records[:6]
        for rec in selected:
            query(rec, 12, offset)
        selected.sort(key=merit, reverse=True)
        positions = [(i,j) for i,row in enumerate(counts)
                     for j,count in enumerate(row) if count]

        for chain, initial in enumerate(selected[:2]):
            current = initial
            for step in range(16):
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                occupied = {c//q for c in bits(current["words"][0])}
                targeted = [p for p in positions if p[1] in occupied]
                i,j = rng.choice(targeted if targeted and
                                 rng.random() < 0.85 else positions)
                count = counts[i][j]
                old = grid[i][j]
                if count == 3 and step % 4:
                    proposal = list(old)
                    t = rng.randrange(3)
                    options = [g for g in range(q) if g not in old]
                    proposal[t] = rng.choice(options)
                    grid[i][j] = (proposal if generates(proposal)
                                  else draw(count, policy))
                else:
                    grid[i][j] = draw(count, policy)
                rec = inspect(grid, side)
                if rec["signature"] in seen:
                    continue
                seen.add(rec["signature"])
                if rec["small"] > current["small"]:
                    continue
                if rec["cycles"][0] > current["cycles"][0]+q:
                    continue
                # Historical words count only if they remain exact kernels.
                rec["words"] = [
                    w for w in current["words"]
                    if syndrome(rec["columns"], w) == 0
                ]
                if rec["words"]:
                    rec["d"] = min(w.bit_count() for w in rec["words"])
                query(rec, 8, offset+300007+7919*step)
                selected.append(rec)
                if merit(rec) >= merit(current):
                    current = rec

        unique = {r["signature"]: r for r in selected}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:3]
        for rec in finalists:
            query(rec, 64+8*(cohort % 3), offset+700001)
            query(rec, 24, offset+900001)
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa = pool("A", sa, a_policy)
    bb = pool("B", sb, b_policy)
    proposals = [(a,b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        -ab[0]["small"]-ab[1]["small"],
        min(ab[0]["d"],ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]-ab[1]["cycles"][0]), reverse=True)

    mask = (1 << q)-1
    for a,b in proposals:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx,hz,n = lifted_matrices(G,A,B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        assert n == 1326 and k >= 468
        if not (13*n <= 40*k < 15*n):
            continue
        actual_a = [
            sum(((hx[(i*6)*q+t] >> (j*6*q)) & mask) << (j*q)
                for j in range(5))
            for i in range(2) for t in range(q)
        ]
        actual_b = [row & ((1 << (6*q))-1) for row in hz[:2*q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert [[len(c) for c in row] for row in A] == sa
        assert [[len(c) for c in row] for row in B] == sb
        assert all(len(set(c)) == len(c)
                   for grid in (A,B) for row in grid for c in row)
        assert max(row.bit_count() for row in hx) <= 11
        assert max(row.bit_count() for row in hz) <= 12
        assert css_ok(hx,hz)
        return {
            "idx": idx, "cells": {"A":A,"B":B}, "k":k,
            "card": (card % 5,card // 5,cohort),
            "component_readings": (a["d"],b["d"]),
            "component_short_statistics": (a["small"],b["small"]),
            "component_cycles": (a["cycles"],b["cycles"]),
        }
    raise ValueError("order-39 cubic pairings missed cell 0.350")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: Favorable brief readings reliably survive longer measurement on the SL2(F5) bed; current slots 6, 28 and 34 instead fell from 24 to 22, 22 to 20 and 23 to 20.
# REFUTED: Primitive-stride-three and detected-weight-16-family repair raises the order-200 bed above fitness 73.143; the current program instead retained five measured readings of sixteen and fitness 73.143.
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
