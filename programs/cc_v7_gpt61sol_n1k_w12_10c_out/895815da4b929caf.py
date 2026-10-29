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
CONCEPT = ("On AGL(1,7)=C7 semidirect C6, order 42, A=(2,5), B=(2,6), degree-four A reinforcement with full-generating paired binomial ratios, balanced against a verified fourteen-cycle degree-two B backbone, should produce five measured distance readings above ten in cell 0.350.")
GROUP = ("Keep frozen AGL(1,7), q=42, byte-for-byte unchanged. This bed leads cell 0.350 at fitness 35.518, but all five measured distances remain ten. The new experiment reallocates term counts and imposes joint column constraints instead of assuming another affine-label refinement transfers a distance gain.")
SHAPES = ("Keep frozen A=(2,5), B=(2,6): n=1428, k>=504, guaranteed rate 12/34=0.352941. Components have 210 and 252 variables with 84 checks each, nominal rates 3/5 and 2/3. Parent minima were entirely sector-1 Acol/Bcol witnesses, often emphasizing Acol2/3; no minimum witnesses occupied Arow/Brow blocks. Measured k was 504 or 512. Exact full ranks retain [0.325,0.375); the shape permits q<=44, with this frozen group requiring q=42.")
FILLING = ("Twenty cards cross four A reinforcement profiles with five B topology/relative-order policies. A has one to three degree-four columns and row totals at most nine; B has two degree-two and four degree-three columns with row totals eight. Column maxima four/three give HX<=9+3=12 and HZ<=8+4=12. Every paired-binomial A column has noncommuting relative ratios generating all 42 elements, preventing their common short subgroup annihilator. Coupled B voltages make its degree-two subgraph six fourteen-cycles, checked on verified syndromes. Exact short collisions, budgeted parity completion through twelve, and deeper independent kernel searches challenge finalists before exact ranks and CSS gate pairing. No budget stop or sampled component reading certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_agl1F7_25x26_reinforcedA_backbone14_witness12_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # (a,b) acts on F7 as t -> 3^b*t+a.
    # 3 has exact order six modulo seven.
    if not valid_q(q):
        raise ValueError("AGL(1,7) requires order 42")
    powers = (1, 3, 2, 6, 4, 5)
    els = [(a, b) for b in range(6) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _aff7_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _aff7_support(rng, width, concentrated, first_total):
    """Three terms per column; concentrated columns split as 0+3."""
    import itertools
    options = [(0, 3) if j < concentrated else (1, 2)
               for j in range(width)]
    choices = [p for p in itertools.product(*options)
               if sum(p) == first_total]
    first = list(rng.choice(choices))
    return [first, [3-x for x in first]]


def _aff7_cell(rng, G, count, rotation_fraction):
    if count == 0:
        return []
    root = G.pick(rng)
    if count == 1:
        return [root]
    if count == 2:
        if rng.random() < rotation_fraction:
            ratio = (rng.randrange(1, 7), 0)  # exact order seven
        else:
            ratio = (rng.randrange(7), rng.choice((1, 5)))
            # Primitive multiplier: exact order six.
        return [root, G.mul(root, ratio)]
    if count != 3:
        raise ValueError("this profile uses only 0..3 terms")
    translation = (rng.randrange(1, 7), 0)
    affine = (rng.randrange(7), rng.choice((1, 5)))
    # A nonzero translation generates C7; the primitive multiplier
    # generates the quotient C6. Together they generate all of AGL(1,7).
    return [root, G.mul(root, translation), G.mul(root, affine)]


def _aff7_columns(rows, width):
    columns = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            columns[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return columns


def _aff7_record(G, grid, side, mode):
    """Identity probes use the scaffold's exact component conventions."""
    width = len(grid[0]) * G.q
    unit = [[[G.e]]]
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, unit)
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, unit, grid)
        rows = hz
    mask = (1 << width)-1
    rows = [row & mask for row in rows]
    columns = _aff7_columns(rows, width)

    bad = sum(c == 0 for c in columns) + width-len(set(columns))
    adjacency = [0] * len(rows)
    forced = 0
    for syndrome in columns:
        degree = syndrome.bit_count()
        forced += degree*(degree-1)*(degree-2)//6
        word = syndrome
        while word:
            bit = word & -word
            adjacency[bit.bit_length()-1] |= syndrome ^ bit
            word ^= bit

    four = 0
    for r, row in enumerate(rows):
        for other in rows[:r]:
            overlap = (row & other).bit_count()
            four += overlap*(overlap-1)//2
    triangles = 0
    for u, neighbors in enumerate(adjacency):
        upper = neighbors & (-1 << (u+1))
        while upper:
            bit = upper & -upper
            v = bit.bit_length()-1
            triangles += (
                neighbors & adjacency[v] & (-1 << (v+1))
            ).bit_count()
            upper ^= bit
    # Exact C6 only when C4=0; otherwise a projection statistic.
    excess = max(0, triangles-forced)
    peak = max(a.bit_count() for a in adjacency)
    cost = 1000000*bad + 600*four + excess + mode*peak
    return {"grid": grid, "rows": rows, "columns": columns,
            "cost": cost, "cycles": (four, excess, peak)}


def _aff7_short4(columns):
    """Collision penalty detects every component kernel of weight <=4."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return 1000000
    pairs = set()
    collisions = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            collisions += int(syndrome in singles or syndrome in pairs)
            pairs.add(syndrome)
    return collisions


def _aff7_fixed_floor(rows, width):
    """Exact capped minimum in the translation-constant subspace.

    Return its minimum when <=6 quotient coordinates, or seven when
    all such words are excluded. Seven is a capped lower bound in THIS
    SUBSPACE, not an upper distance reading or a general certificate.
    """
    import itertools
    # In the group encoding, each seven-element translation coset is
    # contiguous. Restrict actual check rows to constant-on-coset words.
    fw = width // 7
    projected = list(dict.fromkeys(
        sum((((row >> (7*j)) & 127).bit_count() & 1) << j
            for j in range(fw))
        for row in rows))
    columns = _aff7_columns(projected, fw)
    singles = set(columns)
    if 0 in singles:
        return 7
    if len(singles) != fw:
        return 14

    pairs = set()
    three = four = False
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            three |= syndrome in singles
            four |= syndrome in pairs
            pairs.add(syndrome)
    if three:
        return 21
    if four:
        return 28

    triples = set()
    five = six = False
    for a, b, c in itertools.combinations(columns, 3):
        syndrome = a ^ b ^ c
        five |= syndrome in pairs
        six |= syndrome in triples
        triples.add(syndrome)
    if five:
        return 35
    if six:
        return 42
    return 49


def _aff7_kernel_reading(rng, columns, trials):
    """Find genuine classical null words; readings are upper bounds."""
    width = len(columns)
    order = list(range(width))
    reading = width+1
    retained = []
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if syndrome:
                continue
            for old in retained:
                candidate = word ^ old
                if candidate and candidate.bit_count() < word.bit_count():
                    word = candidate
            reading = min(reading, word.bit_count())
            words.append(word)

        light = sorted(words, key=int.bit_count)[:24]
        for word in light:
            for other in light + retained:
                candidate = word ^ other
                if candidate:
                    reading = min(reading, candidate.bit_count())
        retained = sorted(set(retained + light), key=int.bit_count)[:32]
        if reading <= 6:
            break
    return reading


def _aff7_pool(rng, G, support, side, card, cohort):
    import math
    mode = (card + cohort) % 4
    rotation_fraction = (card % 4) / 12.0

    def fresh():
        return [[_aff7_cell(rng, G, count, rotation_fraction)
                 for count in row] for row in support]

    keepers = {}

    def remember(record):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in record["grid"] for cell in row)
        if signature in keepers:
            return
        saved = dict(record)
        saved["grid"] = _aff7_copy(record["grid"])
        keepers[signature] = saved
        if len(keepers) > 5:
            worst = max(keepers, key=lambda key: keepers[key]["cost"])
            del keepers[worst]

    starts = []
    for _ in range(32):
        grid = fresh()
        record = _aff7_record(G, grid, side, mode)
        starts.append(record)
        remember(record)
    starts.sort(key=lambda r: r["cost"])

    positions = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
    for initial in starts[:2]:
        grid = _aff7_copy(initial["grid"])
        current = initial
        for step in range(128):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = _aff7_cell(
                rng, G, support[i][j], rotation_fraction)
            proposed = _aff7_record(G, grid, side, mode)
            temperature = G.q*(10*(1-step/128.0)**2 + 0.15)
            delta = proposed["cost"]-current["cost"]
            if delta <= 0 or rng.random() < math.exp(-delta/temperature):
                current = proposed
                remember(proposed)
            else:
                grid[i][j] = old

    records = list(keepers.values())
    for record in records:
        record["short"] = _aff7_short4(record["columns"])
        record["fixed"] = _aff7_fixed_floor(
            record["rows"], len(record["columns"]))
        record["d"] = _aff7_kernel_reading(
            rng, record["columns"], 16)
        record["rank"] = gf2_rank(record["rows"])

    def quality(record):
        return (-record["short"], record["d"], record["fixed"],
                -record["cost"], record["rank"])

    records.sort(key=quality, reverse=True)
    finalists = records[:3]
    # Extra independent information sets counter selection on a single
    # optimistic short reading. The longer reading can only decrease d.
    for record in finalists:
        record["d"] = min(record["d"], _aff7_kernel_reading(
            rng, record["columns"], 64))
    finalists.sort(key=quality, reverse=True)
    return finalists


def NEW_CTX(rng, G, idx):
    """Reallocate terms, enforce joint voltages, then repair actual witnesses."""
    import math
    from itertools import product

    q = G.q
    card = idx % 20
    cohort = idx // 20
    reinforcement, geometry = divmod(card, 5)
    extra = (1, 2, 3, 3)[reinforcement]

    # Reinforced A columns total four; all other A columns total three.
    # The fourth profile requires genuinely unbalanced reinforced columns.
    degrees_a = [4] * extra + [3] * (5-extra)
    options_a = [
        ((1, 2, 3) if reinforcement == 3 else (2,))
        if degree == 4 else (1, 2)
        for degree in degrees_a
    ]
    target_a = sum(degrees_a) // 2
    choices_a = [
        p for p in product(*options_a)
        if sum(p) == target_a
        and (reinforcement != 3
             or sum(p[j] != 2 for j in range(extra)) >= 2)
    ]
    upper_a = list(rng.choice(choices_a))
    sa = [upper_a, [d-a for d, a in zip(degrees_a, upper_a)]]

    # Two split degree-two columns, followed by four cubic columns.
    # Duplicate zero budgets have different relative-order policies below.
    concentrated_b = (0, 0, 1, 1, 2)[geometry]
    degrees_b = [2, 2, 3, 3, 3, 3]
    options_b = [
        (1,) if j < 2 else
        ((0, 3) if j < 2+concentrated_b else (1, 2))
        for j in range(6)
    ]
    choices_b = [p for p in product(*options_b) if sum(p) == 8]
    upper_b = list(rng.choice(choices_b))
    sb = [upper_b, [d-a for d, a in zip(degrees_b, upper_b)]]

    backbone_target = G.idx((rng.randrange(1, 7), 0))
    assert max(map(sum, sa)) <= 9
    assert list(map(sum, sb)) == [8, 8]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    # Cache the NATIVE operations. The verified helper still constructs
    # every matrix; this is only an integer encoding of the same group.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
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

    probe = Probe()

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def draw_cell(count, side, i, j):
        if count == 0:
            return []
        root = rng.randrange(q)
        if count == 1:
            return [root]
        if count == 2:
            fraction = (1.0, 0.0, 0.5, 0.25, 0.75)[geometry]
            # An upper binomial in a paired reinforced A column always
            # has a primitive multiplier. Its partner can then be chosen
            # to generate the full group rather than a common subgroup.
            if side == "A" and j < extra and i == 0:
                fraction = 0.0
            if rng.random() < fraction:
                ratio = G.idx((rng.randrange(1, 7), 0))
            else:
                ratio = G.idx((rng.randrange(7), rng.choice((1, 5))))
            return [root, table[root][ratio]]

        assert count == 3
        patterns = ((1, 2), (1, 5), (4, 5))
        while True:
            if geometry in (0, 2, 4):
                x = G.idx((rng.randrange(1, 7), 0))
                y = G.idx((rng.randrange(7), rng.choice((1, 5))))
            else:
                b, d = patterns[(card + cohort + i + j) % 3]
                x = G.idx((rng.randrange(7), b))
                y = G.idx((rng.randrange(7), d))
            if table[x][y] == table[y][x]:
                continue
            cell = [root, table[root][x], table[root][y]]
            # Both directed-difference orientations must be Sidon.
            # In particular, no involution pair is admitted.
            left = {
                table[inverses[a]][b]
                for a in cell for b in cell if a != b
            }
            right = {
                table[a][inverses[b]]
                for a in cell for b in cell if a != b
            }
            if len(left) == len(right) == 6:
                return cell

    def fresh(support, side):
        return [[draw_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def record(grid, side):
        grid = _aff7_copy(grid)
        if side == "A":
            for j in range(extra):
                if sa[0][j] != 2 or sa[1][j] != 2:
                    continue
                upper = grid[0][j]
                u = table[inverses[upper[0]]][upper[1]]
                for _ in range(128):
                    lower = grid[1][j]
                    v = table[inverses[lower[0]]][lower[1]]
                    if table[u][v] != table[v][u]:
                        break
                    grid[1][j] = draw_cell(2, side, 1, j)
                else:
                    raise ValueError("paired A ratios failed to generate AGL")
                # The upper ratio has order six. Noncommutation supplies
                # a nonzero translation, hence together they generate G.
        else:
            # For B's verified right component, a split column joins
            # r in the upper block to r*(top*bottom^-1) in the lower.
            # Set u*v^-1 to a nonzero translation of exact order seven.
            top0, bottom0 = grid[0][0][0], grid[1][0][0]
            top1 = grid[0][1][0]
            u = table[top0][inverses[bottom0]]
            v = table[inverses[backbone_target]][u]
            grid[1][1] = [table[inverses[v]][top1]]

        result = _aff7_record(probe, grid, side, (card + cohort) % 4)
        if side == "B":
            # Validate the intended girth on actual component syndromes,
            # not solely on the preceding native-group calculation.
            adjacency = [set() for _ in result["rows"]]
            for syndrome in result["columns"][:2*q]:
                checks = list(bits(syndrome))
                assert len(checks) == 2
                a, b = checks
                adjacency[a].add(b)
                adjacency[b].add(a)
            assert all(len(neighbors) == 2 for neighbors in adjacency)
            unseen = set(range(len(adjacency)))
            lengths = []
            while unseen:
                reached = {next(iter(unseen))}
                todo = list(reached)
                while todo:
                    a = todo.pop()
                    for b in adjacency[a]:
                        if b not in reached:
                            reached.add(b)
                            todo.append(b)
                unseen.difference_update(reached)
                lengths.append(len(reached))
            assert sorted(lengths) == [14] * 6
        result["side"] = side
        result["witnesses"] = {}
        return result

    def orbit_key(word, side):
        """Regular component symmetry, used only to deduplicate witnesses.

        A's verified left component admits right multiplication;
        B's verified right component admits left multiplication.
        No symmetry images are submitted as new code designs.
        """
        first = (word & -word).bit_length() - 1
        block = first // q
        anchors = (word >> (block * q)) & ((1 << q) - 1)
        terms = list(bits(word))
        answer = None
        for a in bits(anchors):
            t = inverses[a]
            moved = 0
            for pos in terms:
                j, h = divmod(pos, q)
                c = table[h][t] if side == "A" else table[t][h]
                moved |= 1 << (j * q + c)
            answer = moved if answer is None else min(answer, moved)
        return answer

    def prepare(rec):
        """Exact low-weight collisions and actual syndrome lookup tables."""
        if "pairs" in rec:
            return
        columns = rec["columns"]
        singles = {}
        pairs = {}
        incidence = [0] * len(rec["rows"])

        def retain(word):
            if not word:
                return
            key = orbit_key(word, rec["side"])
            rec["witnesses"].setdefault(key, word)

        for c, syndrome in enumerate(columns):
            word = 1 << c
            if syndrome == 0:
                retain(word)
            previous = singles.get(syndrome)
            if previous is not None:
                retain(previous ^ word)
            else:
                singles[syndrome] = word
            for r in bits(syndrome):
                incidence[r] |= word
        for i, x in enumerate(columns):
            for j in range(i):
                syndrome = x ^ columns[j]
                word = (1 << i) | (1 << j)
                previous = singles.get(syndrome)
                if previous is not None:
                    retain(previous ^ word)
                previous = pairs.get(syndrome)
                if previous is not None:
                    retain(previous ^ word)
                else:
                    pairs[syndrome] = word
        rec["singles"] = singles
        rec["pairs"] = pairs
        rec["incidence"] = incidence

    def search_short(rec, budget, cap=12):
        """Budgeted parity completion, returning VERIFIED classical words.

        Anchoring covers regular column-block orbits. Branch exclusions
        avoid repeatedly choosing the same completion. A budget stop
        prevents any claim of exhaustive exclusion through cap.
        """
        prepare(rec)
        columns = rec["columns"]
        width = len(columns)
        witnesses = rec["witnesses"]
        if witnesses and min(w.bit_count()
                             for w in witnesses.values()) <= 4:
            return

        singles = rec["singles"]
        pairs = rec["pairs"]
        incidence = rec["incidence"]
        max_degree = max(column.bit_count() for column in columns)
        all_variables = (1 << width) - 1
        priorities = [rng.random() for _ in columns]
        roots = list(range(0, width, q))
        rng.shuffle(roots)

        def retain(word):
            if not word or word.bit_count() > cap:
                return
            key = orbit_key(word, rec["side"])
            witnesses.setdefault(key, word)

        for root in roots:
            remaining_nodes = [max(1, budget // len(roots))]

            def visit(syndrome, word, allowed):
                if remaining_nodes[0] <= 0:
                    return
                remaining_nodes[0] -= 1
                known = min((w.bit_count() for w in witnesses.values()),
                            default=cap)
                used = word.bit_count()
                left = min(cap, known) - used
                if syndrome == 0:
                    retain(word)
                    return
                if left <= 0 or syndrome.bit_count() > max_degree * left:
                    return

                one = singles.get(syndrome, 0)
                if one and one & allowed == one:
                    retain(word | one)
                pair = pairs.get(syndrome, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    retain(word | pair)
                known = min((w.bit_count() for w in witnesses.values()),
                            default=cap)
                left = min(cap, known) - used
                if left <= 2:
                    return

                # Any completion must include another variable incident
                # on each odd check. Branch on the most constrained one.
                chosen = None
                smallest = width + 1
                for r in bits(syndrome):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if count == 0:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                candidates = list(bits(chosen))
                candidates.sort(key=lambda c: (
                    -(columns[c] & syndrome).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(syndrome ^ columns[c], word | bit, live)
                    if remaining_nodes[0] <= 0:
                        break

            # Translate a word's first occupied column-block term to
            # coordinate zero; earlier blocks can then be forbidden.
            allowed = all_variables & ~((1 << root) - 1)
            allowed &= ~(1 << root)
            visit(columns[root], 1 << root, allowed)

        if witnesses:
            minimum = min(w.bit_count() for w in witnesses.values())
            rec["witnesses"] = {
                key: word for key, word in witnesses.items()
                if word.bit_count() == minimum
            }

    def merit(rec):
        words = rec["witnesses"]
        detected = min((w.bit_count() for w in words.values()), default=14)
        # Fourteen means unresolved through twelve, NOT a certified floor.
        return (detected, -len(words), -rec["cost"])

    def circuits(rec, trials):
        """Independent kernel searches; every retained reading has a witness."""
        columns = rec["columns"]
        width = len(columns)
        order = list(range(width))
        retained = []
        best = min((w.bit_count() for w in rec["witnesses"].values()),
                   default=width+1)

        def note(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight > 12:
                return
            known = min((w.bit_count()
                         for w in rec["witnesses"].values()),
                        default=width+1)
            if weight > known:
                return
            if weight < known:
                rec["witnesses"] = {}
            key = orbit_key(word, rec["side"])
            rec["witnesses"].setdefault(key, word)

        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            words = []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if syndrome:
                    continue
                for sweep in range(2):
                    changed = False
                    for other in retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    if not changed:
                        break
                words.append(word)
                note(word)

            light = sorted(words, key=int.bit_count)[:20]
            # A short basis word can cancel profitably with a longer one.
            # Do not restrict these pairs to the short shortlist.
            for word in light:
                for other in words + retained:
                    note(word ^ other)

            reducers = words + retained
            rng.shuffle(reducers)
            descended = []
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    weight = word.bit_count()
                    for other in reducers:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    note(word)
                    if not changed:
                        break
                descended.append(word)

            # All combinations remain in the verified component kernel.
            for a in range(min(10, len(light))):
                for b in range(a):
                    for c in range(b):
                        note(light[a] ^ light[b] ^ light[c])
            retained = sorted(set(retained + light + descended),
                              key=int.bit_count)[:32]
            if best <= 6:
                break
        return best

    def pool(support, side):
        """Cycle-safe starts followed by short-word-directed column repair."""
        bank = {}

        def remember(rec):
            signature = tuple(tuple(sorted(cell))
                              for row in rec["grid"] for cell in row)
            bank[signature] = rec
            if len(bank) > 6:
                worst = max(bank, key=lambda key: bank[key]["cost"])
                del bank[worst]

        starts = []
        for _ in range(28):
            rec = record(fresh(support, side), side)
            starts.append(rec)
            remember(rec)
        starts.sort(key=lambda rec: rec["cost"])
        positions = [(i, j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]
        for initial in starts[:2]:
            current = initial
            for step in range(96):
                grid = _aff7_copy(current["grid"])
                i, j = rng.choice(positions)
                grid[i][j] = draw_cell(support[i][j], side, i, j)
                proposed = record(grid, side)
                delta = proposed["cost"] - current["cost"]
                temperature = q * (8 * (1 - step / 96.0) ** 2 + 0.1)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta / temperature)):
                    current = proposed
                    remember(current)

        survivors = list(bank.values())
        for rec in survivors:
            search_short(rec, 10000)
        survivors.sort(key=merit, reverse=True)
        current = survivors[0]
        keepers = survivors[:3]

        for step in range(40):
            proposals = []
            words = list(current["witnesses"].values())
            for _ in range(2):
                grid = _aff7_copy(current["grid"])
                if words:
                    witness = rng.choice(words)
                    columns_used = sorted({c // q for c in bits(witness)})
                    j = rng.choice(columns_used)
                else:
                    j = rng.randrange(len(support[0]))
                # Change the entire implicated column's relative voltages,
                # not a gauge image of its previous filling.
                for i in range(2):
                    grid[i][j] = draw_cell(support[i][j], side, i, j)
                proposed = record(grid, side)
                if proposed["cycles"][0] > current["cycles"][0] + q:
                    continue
                search_short(proposed, 7000)
                proposals.append(proposed)
            if not proposals:
                continue
            proposed = max(proposals, key=merit)
            old_key, new_key = merit(current), merit(proposed)
            accept = new_key >= old_key
            if not accept and new_key[0] == old_key[0]:
                temperature = 1.5 * (1 - step / 40.0) + 0.1
                loss = (len(proposed["witnesses"]) -
                        len(current["witnesses"]))
                loss += max(0, proposed["cost"] - current["cost"]) / (20*q)
                accept = rng.random() < math.exp(
                    min(0.0, -loss / temperature))
            if accept:
                current = proposed
            keepers.append(proposed)
            keepers.sort(key=merit, reverse=True)
            del keepers[5:]

        # Stronger independent challenges counter selection on a missed
        # short witness. Known words are retained across all searches.
        final = []
        signatures = set()
        for rec in sorted(keepers + [current], key=merit, reverse=True):
            signature = tuple(tuple(sorted(cell))
                              for row in rec["grid"] for cell in row)
            if signature in signatures:
                continue
            signatures.add(signature)
            search_short(rec, 50000)
            rec["reading"] = circuits(rec, 112)
            observed = min((w.bit_count()
                            for w in rec["witnesses"].values()),
                           default=len(rec["columns"]) + 1)
            rec["reading"] = min(rec["reading"], observed)
            rec["rank"] = gf2_rank(rec["rows"])
            # Explicitly verify the actual retained witnesses.
            for word in list(rec["witnesses"].values())[:8]:
                syndrome = 0
                for c in bits(word):
                    syndrome ^= rec["columns"][c]
                assert syndrome == 0
            final.append(rec)
            if len(final) == 4:
                break
        final.sort(key=lambda rec: (
            min(14, rec["reading"]), merit(rec), rec["rank"]),
            reverse=True)
        return final[:3]

    aa = pool(sa, "A")
    bb = pool(sb, "B")
    best = None
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1428 and k >= 504
                    and 325*n <= 1000*k < 375*n):
                continue
            reading = min(14, a["reading"], b["reading"])
            key = (k * reading * reading, reading,
                   -len(a["witnesses"]) - len(b["witnesses"]),
                   -a["cost"] - b["cost"])
            if best is None or key > best[0]:
                best = (key, A, B, hx, hz, k,
                        (a["reading"], b["reading"]))
    if best is None:
        raise ValueError("witness-repaired fillings missed the rate gate")
    key, A, B, hx, hz, k, readings = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    final_sa = [[len(cell) for cell in row] for row in A]
    final_sb = [[len(cell) for cell in row] for row in B]
    assert final_sa == sa and final_sb == sb
    assert _grid_weights(final_sa, final_sb) <= W_MAX
    assert _no_dead_lines(final_sa, final_sb)
    return {"idx": idx, "card": (reinforcement, geometry, cohort),
            "cells": {"A": A, "B": B}, "k": k,
            "component_readings": readings, "quality": key,
            "B_degree_two_cycle_lengths": (14,) * 6}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly constrained, witness-screened complete cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled positive-cell supports with uniform labels suppress sector-1 weight-6 logicals on this bed; all five measured parent codes instead have d=6 and sector-1 minimum witnesses.
# REFUTED: Balanced abelian cubic columns with component weight-six screening break the distance-ten plateau; all five current measured codes returned ten, and 31 of 60 submitted slots collapsed as isomorphic duplicates.
# REFUTED: The AGL(1,7) group jump with cycle-first spanning-cubic selection raises the five measured distances above ten; all five measured selections instead returned ten, with abundant sector-1 minimum witnesses.
# REFUTED: Distinct-multiplier trinomials and witness-directed repair on the cubic AGL(1,7) (2,5)x(2,6) bed raise all five measured distances above ten; the current measured selections all returned ten, with abundant sector-1 minimum witnesses.
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
