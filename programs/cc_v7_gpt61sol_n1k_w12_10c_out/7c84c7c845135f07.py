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
CONCEPT = ("Cyclic Z32, order 32, A=(2,7), B=(2,6): independently repaired support profiles ranked by verified localized logical witnesses and actual k should improve five-code fitness beyond 21.3040 in cell 0.450. Retained ring/matching profiles can have weight-seven cofactor ceilings; distance eight is not assumed.")
GROUP = ("Keep the exact frozen Z32 construction and order. This bed measured k=640 and five selected distance readings of seven. Retain that measured group while testing support margins and joint pairing; neither related-bed readings nor a missed short witness establishes larger distance.")
SHAPES = ("Keep exact A=(2,7), B=(2,6): n=1472, k>=640 and guaranteed rate 20/46=0.434783 target cell 0.450. The components have 224 and 192 variables, each with 64 checks; q=32 is the largest admissible lift. Parent minima were entirely sector one, especially Acol0..2 across Bcol2..4. Exact full ranks retain [0.425,0.475).")
FILLING = ("Twenty cards cross five ring geometries with four B support profiles: mixed cubic, two-ring lower-margin, unequal-margin ring/matching, and a concentrated rank-deficit experiment. All supports obey both CSS weight budgets. Optimize authoritative component matrices, retain actual cofactor and bounded-completion witnesses, repair implicated columns, and independently challenge finalists. Compare complete pairings using actual k and nontrivial localized logicals reduced with ORIGINAL stabilizers. Known cofactor words cap readings. Completion budgets and sampled readings never certify exclusion or quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r045_m32r1s1_27x26_marginportfolio_jointlogical_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # Same metacyclic construction and cyclic specialization.
    # L2 resize: q=32 and size factor 46 give n=1472.
    m, r, s = 32, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _copy_grid(grid):
    return [[list(cell) for cell in row] for row in grid]


def _backbone_cells(rng, G, profile, mode):
    """Three degree-two columns and four degree-three columns."""
    triples = (
        ((2, 1), (2, 1), (1, 2), (1, 2)),
        ((3, 0), (0, 3), (2, 1), (1, 2)),
        ((3, 0), (3, 0), (0, 3), (0, 3)),
        ((3, 0), (1, 2), (1, 2), (1, 2)),
        ((2, 1), (2, 1), (1, 2), (1, 2)),
    )
    doubles = ((1, 1),) * 3 if profile == 4 else (
        (2, 0), (0, 2), (1, 1))
    counts = doubles + triples[profile]
    grid = [[rng.sample(G.elements, counts[j][i])
             for j in range(7)] for i in range(2)]
    gap = (5, 6, 8, 10)[mode]
    if profile != 4:
        # The upper ring has order 28; one cross matching connects the
        # lower ring's possible components. These are not gauge copies.
        for i, j, delta in ((0, 0, 1), (1, 1, gap)):
            root = G.pick(rng)
            grid[i][j] = [root, G.mul(root, (delta, 0))]
    else:
        # Three cross matchings, with relative shift 1 ensuring that
        # the degree-two Tanner subgraph is connected.
        for j, delta in enumerate((0, 1, gap)):
            root = G.pick(rng)
            grid[0][j] = [root]
            grid[1][j] = [G.mul(root, (delta, 0))]
    return grid


def _component_record(G, grid, side):
    """Extract either new component exclusively through the verified helper."""
    zero = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    rows = [row & mask for row in rows]
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit

    # Exact Tanner four/six-cycle counts, including multi-term cells.
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    c4 = 0
    for a in range(h):
        for b in range(a + 1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                c4 += overlap * (overlap - 1) // 2
    c6 = 0
    for a in range(h):
        for t, b in enumerate(upper[a]):
            for c in upper[a][t + 1:]:
                bc = common[b][c]
                if bc:
                    ab, ac = common[a][b], common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    c6 += ab * ac * bc - triple * (ab + ac + bc - 2)
    return {"grid": _copy_grid(grid), "rows": rows, "cols": cols,
            "cycles": (c4, c6), "rank": gf2_rank(rows)}


def _kernel_floor5(cols):
    """Exact classical exclusion through weight five; not a quantum bound."""
    width = len(cols)
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != width:
        return 2
    pairs = {}
    bits = [1 << i for i in range(width)]
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs[syndrome] = bits[i] | bits[j]
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            used = bits[i] | bits[j]
            for k in range(j):
                previous = pairs.get(pair ^ cols[k])
                if previous is not None and not (
                        previous & (used | bits[k])):
                    return 5
    return 6


def _detected_six(cols):
    """Count verified six-word collisions among connected triples.

    This is deliberately not claimed to exhaust all weight-six words.
    """
    from itertools import combinations
    incidence = {}
    for v, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            incidence.setdefault(bit.bit_length() - 1, []).append(v)
            syndrome ^= bit
    adjacency = [set() for _ in cols]
    for variables in incidence.values():
        for v in variables:
            adjacency[v].update(u for u in variables if u != v)
    triples = set()
    for b, neighbors in enumerate(adjacency):
        for a, c in combinations(sorted(neighbors), 2):
            triples.add(tuple(sorted((a, b, c))))
    seen = {}
    found = set()
    for a, b, c in triples:
        syndrome = cols[a] ^ cols[b] ^ cols[c]
        word = (1 << a) | (1 << b) | (1 << c)
        old = seen.get(syndrome)
        if old is not None:
            witness = old ^ word
            if witness.bit_count() == 6:
                found.add(witness)
        else:
            seen[syndrome] = word
    return len(found)


def _circuit_reading(rng, cols, trials):
    """Repeated genuine classical kernel witnesses: upper readings only."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for v in order:
            syndrome, word = cols[v], 1 << v
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
        for a in light:
            for b in words:
                if a != b:
                    best = min(best, (a ^ b).bit_count())
    return best


def _component_pool(rng, G, profile, mode, side, design):
    """Independent multistart voltage optimization for one construction card."""
    import math
    penalty = 1200
    keepers = []

    def remember(record):
        if record["rank"] != 2 * G.q:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in record["grid"] for cell in row)
        if any(signature == item[1] for item in keepers):
            return
        c4, c6 = record["cycles"]
        cost = penalty * c4 + c6
        keepers.append((cost, signature, record))
        keepers.sort(key=lambda item: item[0])
        del keepers[4:]

    for restart in range(4):
        grid = _backbone_cells(rng, G, profile, mode)
        current = _component_record(G, grid, side)
        cost = penalty * current["cycles"][0] + current["cycles"][1]
        remember(current)
        positions = [(i, j) for i in range(2) for j in range(3, 7)
                     if grid[i][j]]
        for step in range(96):
            i, j = rng.choice(positions)
            old = list(grid[i][j])
            proposal = list(old)
            t = rng.randrange(len(proposal))
            g = G.pick(rng)
            if any(G._key(g) == G._key(h) for h in proposal):
                continue
            proposal[t] = g
            grid[i][j] = proposal
            record = _component_record(G, grid, side)
            c4, c6 = record["cycles"]
            new_cost = penalty * c4 + c6
            temperature = G.q * (
                (10 + design % 4) * (1 - step / 96.0) ** 2 + 0.1)
            accept = new_cost <= cost or rng.random() < math.exp(
                min(0.0, (cost - new_cost) / temperature))
            if accept:
                cost, current = new_cost, record
                remember(record)
            else:
                grid[i][j] = old

    if not keepers:
        raise ValueError("no full-row-rank component")
    answer = []
    for cost, signature, record in keepers:
        floor = _kernel_floor5(record["cols"])
        six = _detected_six(record["cols"]) if floor == 6 else 1000000
        reading = _circuit_reading(rng, record["cols"], 24)
        record["quality"] = (floor, -six, reading, -cost)
        answer.append(record)
    answer.sort(key=lambda record: record["quality"], reverse=True)
    # Extra independent elimination orders reduce lucky pool selection.
    for record in answer[:2]:
        floor, negative_six, reading, negative_cost = record["quality"]
        reading = min(reading, _circuit_reading(
            rng, record["cols"], 64))
        record["quality"] = (floor, negative_six, reading, negative_cost)
    return answer


def _legacy_ctx27(rng, G, idx):
    """Unused historical (2,7)x(2,7) optimizer; NEW_CTX below replaces it."""
    import math

    q = G.q
    width = 7 * q
    mask = (1 << q) - 1
    card = idx % 20
    mode = card % 4
    # The first three columns are two rings and one cross matching.
    # gcd(a,b,28)=1 makes the backbone connected.
    backbones = ((1, 5), (1, 6), (1, 8), (1, 11), (1, 10))
    splits = ((2, 1), (2, 1), (1, 2), (1, 2))

    def orbit_key(word):
        """Canonical cyclic translation of an actual component word.

        Z28 acts regularly within every component column block.
        Normalize a term in the first occupied block to coordinate zero.
        This symmetry is used only on the verified classical components.
        """
        first = (word & -word).bit_length() - 1
        block = first // q
        choices = (word >> (block * q)) & mask
        best = None
        while choices:
            bit = choices & -choices
            shift = bit.bit_length() - 1
            translated = 0
            for j in range(7):
                value = (word >> (j * q)) & mask
                value = ((value >> shift) |
                         (value << (q - shift))) & mask
                translated |= value << (j * q)
            if best is None or translated < best:
                best = translated
            choices ^= bit
        return best

    def short_words(cols):
        """Exhaustive exclusion through weight five, returning orbit witnesses."""
        witnesses = {}
        singles = {}
        pairs = {}
        floor = 6

        def add(word):
            nonlocal floor
            weight = word.bit_count()
            if word and weight <= 5:
                floor = min(floor, weight)
                witnesses.setdefault(orbit_key(word), word)

        for i, syndrome in enumerate(cols):
            word = 1 << i
            if not syndrome:
                add(word)
            old = singles.get(syndrome)
            if old is not None:
                add(old ^ word)
            else:
                singles[syndrome] = word

        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                old = singles.get(syndrome)
                if old is not None:
                    add(old ^ word)
                old = pairs.get(syndrome)
                if old is not None:
                    add(old ^ word)
                else:
                    pairs[syndrome] = word

        # Once weights <=4 are absent, a triple/pair collision is
        # either a genuine weight-five word or an already excluded
        # shorter word. Every weight-five orbit has an anchored term.
        if floor >= 5:
            for root in range(0, width, q):
                cr = cols[root]
                for i in range(width):
                    if i == root:
                        continue
                    value = cr ^ cols[i]
                    used = (1 << root) | (1 << i)
                    for j in range(i):
                        if j == root:
                            continue
                        old = pairs.get(value ^ cols[j])
                        if old is not None:
                            add(old ^ used ^ (1 << j))
        return floor, list(witnesses.values())

    def six_words(record):
        """Exact detection of weight six after excluding weights <=5.

        All columns have degree >=2 and the Tanner graph has no C4.
        In a kernel word each variable therefore has at least two
        distinct neighboring variables. A six-variable word has two
        disjoint adjacent pairs, and can be partitioned into two triples
        each containing such a pair. Enumerating these triples suffices.

        Stored witnesses need not exhaust every weight-six orbit:
        absence of a collision is exhaustive; the count is diagnostic.
        """
        cols = record["cols"]
        adjacency = [set() for _ in cols]
        for row in record["rows"]:
            variables = []
            while row:
                bit = row & -row
                variables.append(bit.bit_length() - 1)
                row ^= bit
            for v in variables:
                adjacency[v].update(u for u in variables if u != v)

        seen = {}
        found = {}
        for a in range(width):
            for b in adjacency[a]:
                if b <= a:
                    continue
                value = cols[a] ^ cols[b]
                used = (1 << a) | (1 << b)
                for c in range(width):
                    if c == a or c == b:
                        continue
                    syndrome = value ^ cols[c]
                    word = used | (1 << c)
                    old = seen.get(syndrome)
                    if old is None:
                        seen[syndrome] = word
                    elif old != word:
                        witness = old ^ word
                        if witness.bit_count() == 6:
                            found.setdefault(orbit_key(witness), witness)
        return list(found.values())

    def redraw_column(grid, j):
        """Replace a whole mixed column, preserving the C4 exclusions."""
        within = [set(), set()]
        cross = set()
        for c in range(7):
            if c == j:
                continue
            for i in range(2):
                values = [g[0] for g in grid[i][c]]
                for t, a in enumerate(values):
                    for b in values[:t]:
                        d = (a - b) % q
                        within[i].add(min(d, (-d) % q))
            for a in grid[0][c]:
                for b in grid[1][c]:
                    cross.add((a[0] - b[0]) % q)

        counts = splits[j - 3]
        for _ in range(512):
            column = [rng.sample(G.elements, count) for count in counts]
            legal = True
            for i in range(2):
                if len(column[i]) == 2:
                    d = (column[i][0][0] - column[i][1][0]) % q
                    key = min(d, (-d) % q)
                    if d == q // 2 or key in within[i]:
                        legal = False
            between = [(a[0] - b[0]) % q
                       for a in column[0] for b in column[1]]
            if len(set(between)) != len(between):
                legal = False
            if any(d in cross for d in between):
                legal = False
            if legal:
                for i in range(2):
                    grid[i][j] = column[i]
                return True
        return False

    def geometry(record):
        """Four distinct objectives on actual component cycle geometry."""
        c6 = record["cycles"][1]
        if mode == 0:
            return c6
        adjacency = [0] * len(record["rows"])
        incidence3 = [0] * len(record["rows"])
        for syndrome in record["cols"]:
            checks = []
            value = syndrome
            while value:
                bit = value & -value
                checks.append(bit.bit_length() - 1)
                value ^= bit
            for r in checks:
                adjacency[r] |= syndrome ^ (1 << r)
                incidence3[r] += len(checks) == 3

        peak = 0
        projected4 = 0
        for r, neighbors in enumerate(adjacency):
            triangles = 0
            value = neighbors
            while value:
                bit = value & -value
                t = bit.bit_length() - 1
                triangles += (neighbors & adjacency[t]).bit_count()
                value ^= bit
            peak = max(peak, triangles // 2 - incidence3[r])
            if mode >= 2:
                for t in range(r):
                    common = (neighbors & adjacency[t]).bit_count()
                    projected4 += common * (common - 1) // 2
        if mode == 1:
            return c6 + 4 * peak
        if mode == 2:
            return c6 + projected4 / 16.0
        return c6 + 2 * peak + projected4 / 8.0

    def inspect(grid, side):
        record = _component_record(G, grid, side)
        if record["rank"] != 2 * q or record["cycles"][0]:
            return None
        floor, words = short_words(record["cols"])
        record["floor"] = floor
        record["short"] = words
        record["geometry"] = geometry(record)
        record["merit5"] = (floor, -len(words), -record["geometry"])
        return record

    def mutable_column(words):
        if words:
            word = rng.choice(words)
            choices = [j for j in range(3, 7)
                       if (word >> (j * q)) & mask]
            if choices:
                return rng.choice(choices)
        return rng.randrange(3, 7)

    def optimize(side, backbone):
        a, b = backbones[backbone]
        keepers = []

        def remember(record):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in record["grid"] for cell in row)
            if any(signature == old[1] for old in keepers):
                return
            keepers.append((record["merit5"], signature, record))
            keepers.sort(key=lambda item: item[0], reverse=True)
            del keepers[4:]

        for start in range(16):
            grid = [[[] for _ in range(7)] for _ in range(2)]
            grid[0][0] = [G.e, (a, 0)]
            grid[1][1] = [G.e, (b, 0)]
            grid[0][2] = [G.e]
            grid[1][2] = [G.e]
            if not all(redraw_column(grid, j) for j in range(3, 7)):
                continue
            record = inspect(grid, side)
            if record is not None:
                remember(record)
        if not keepers:
            raise ValueError("no full-rank mixed component start")

        current = keepers[0][2]
        # Repair columns used by actual short words, not arbitrary labels.
        for step in range(48):
            proposals = []
            for _ in range(2):
                grid = _copy_grid(current["grid"])
                j = mutable_column(current["short"])
                if redraw_column(grid, j):
                    record = inspect(grid, side)
                    if record is not None:
                        remember(record)
                        proposals.append(record)
            if not proposals:
                continue
            candidate = max(proposals, key=lambda r: r["merit5"])
            if candidate["merit5"] >= current["merit5"]:
                current = candidate
            elif candidate["floor"] == current["floor"]:
                delta = (len(candidate["short"]) - len(current["short"])
                         + (candidate["geometry"] -
                            current["geometry"]) / (8.0 * q))
                temperature = 2.5 * (1 - step / 48.0) + 0.15
                if rng.random() < math.exp(min(0.0, -delta / temperature)):
                    current = candidate

        finalists = []
        for _, _, record in keepers:
            words = six_words(record) if record["floor"] == 6 else []
            record["six"] = words
            record["merit6"] = (
                record["floor"], -len(record["short"]),
                -len(words) if record["floor"] == 6 else -1000000,
                -record["geometry"])
            finalists.append(record)
        current = max(finalists, key=lambda r: r["merit6"])

        # A second, smaller repair stage attacks detected six-word orbits.
        if current["floor"] == 6:
            for step in range(8):
                if not current["six"]:
                    break
                for _ in range(2):
                    grid = _copy_grid(current["grid"])
                    j = mutable_column(current["six"])
                    if not redraw_column(grid, j):
                        continue
                    record = inspect(grid, side)
                    if record is None or record["floor"] != 6:
                        continue
                    record["six"] = six_words(record)
                    record["merit6"] = (
                        6, 0, -len(record["six"]), -record["geometry"])
                    if record["merit6"] > current["merit6"]:
                        current = record
                        finalists.append(record)

        finalists.sort(key=lambda r: r["merit6"], reverse=True)
        answer = []
        for record in finalists[:3]:
            reading = _circuit_reading(rng, record["cols"], 32)
            reading = min(reading, _circuit_reading(
                rng, record["cols"], 64))
            floor = (7 if record["floor"] == 6 and not record["six"]
                     else record["floor"])
            record["quality"] = (
                floor, reading, -len(record["short"]),
                -len(record["six"]), -record["geometry"])
            answer.append(record)
        answer.sort(key=lambda r: r["quality"], reverse=True)
        return answer

    aa = optimize("A", card // 4)
    bb = optimize("B", (2 * (card // 4) + 1) % 5)
    pairs = []
    for a in aa:
        for b in bb:
            qa, qb = a["quality"], b["quality"]
            key = (min(qa[0], qb[0]), min(qa[1], qb[1]),
                   qa[0] + qb[0], qa[1] + qb[1],
                   qa[2] + qb[2], qa[3] + qb[3],
                   qa[4] + qb[4])
            pairs.append((key, a, b))
    pairs.sort(key=lambda item: item[0], reverse=True)

    for key, a, b in pairs:
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (425 * n <= 1000 * k < 475 * n):
            continue
        assert n == 1484
        assert max(row.bit_count() for row in hx + hz) <= 12
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        return {"A": A, "B": B, "k": k, "card": card,
                "quality": key,
                "component_floors": (a["quality"][0], b["quality"][0])}
    raise ValueError("repaired pairs missed the 0.450 rate window")


def NEW_CTX(rng, G, idx):
    """Support-margin portfolio with authoritative joint logical challenges."""
    from itertools import combinations
    from math import exp

    q = G.q
    assert q == 32
    card, cohort = idx % 20, idx // 20
    geometry, model = card // 4, card % 4
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    gap = (6, 10, 12, 14, 7)[geometry]

    acounts = ((2,0), (0,2), (1,1),
               (2,1), (2,1), (1,2), (1,2))
    bprofiles = (
        ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
        ((2,0), (0,2), (2,1), (2,1), (1,2), (1,2)),
        ((2,0), (0,2), (1,1), (2,1), (2,1), (1,2)),
        ((2,0), (3,0), (1,2), (1,2), (1,2), (1,2)),
    )
    bcounts = bprofiles[model]
    sa = [[column[i] for column in acounts] for i in range(2)]
    sb = [[column[i] for column in bcounts] for i in range(2)]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def syndrome(cols, word):
        value = 0
        for v in bits(word):
            value ^= cols[v]
        return value

    def multiply(a, b):
        value = 0
        for shift in bits(a):
            value ^= ((b << shift) | (b >> (q-shift))) & mask
        return value

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def draw_column(grid, j, counts):
        """Redraw actual relative voltages, not a common column gauge."""
        for _ in range(96):
            proposal = [rng.sample(G.elements, count) for count in counts]
            # Avoid order-two binomials; quartic annihilator families are
            # still challenged on the actual matrices below.
            if any(len(cell) == 2 and
                   (cell[0][0]-cell[1][0]) % q == q//2
                   for cell in proposal):
                continue
            for i in range(2):
                grid[i][j] = proposal[i]
            return

    def fresh(side, counts):
        grid = [[[] for _ in counts] for _ in range(2)]
        for j, column in enumerate(counts):
            draw_column(grid, j, column)
        if side == "A":
            grid[0][0], grid[1][0] = [G.e, (1,0)], []
            grid[0][1], grid[1][1] = [], [G.e, (gap,0)]
            grid[0][2], grid[1][2] = [G.e], [G.e]
        return grid

    def inspect(grid, side):
        rec = _component_record(G, grid, side)
        nc = len(grid[0])
        cols = rec["cols"]
        if not all(cols) or len(set(cols)) != len(cols):
            return None
        polys = [[(cols[j*q] >> (i*q)) & mask
                  for j in range(nc)] for i in range(2)]
        minors = {}
        for a, b in combinations(range(nc), 2):
            minors[a,b] = (multiply(polys[0][a], polys[1][b]) ^
                           multiply(polys[0][b], polys[1][a]))
        words = []
        for a, b, c in combinations(range(nc), 3):
            word = (minors[b,c] << (a*q) |
                    minors[a,c] << (b*q) |
                    minors[a,b] << (c*q))
            if word:
                assert syndrome(cols, word) == 0
                words.append(word)
        if not words:
            return None
        rec["words"] = sorted(set(words), key=int.bit_count)[:16]
        rec["d"] = rec["words"][0].bit_count()
        c4, c6 = rec["cycles"]
        rec["cost"] = 1200*c4 + (1+geometry)*c6
        rec["sig"] = signature(grid)
        return rec

    def query(rec, trials, offset, budget):
        """Actual classical words; exact detection only through four."""
        local = random.Random(seed+offset)
        cols = rec["cols"]
        width = len(cols)
        found = list(rec["words"])

        def note(word):
            if word:
                found.append(word)

        singles, pairs = {}, {}
        incidence = [0]*len(rec["rows"])
        for v, col in enumerate(cols):
            word = 1 << v
            if not col:
                note(word)
            old = singles.get(col)
            if old is not None:
                note(old ^ word)
            singles.setdefault(col, word)
            for r in bits(col):
                incidence[r] |= word
        for a, col in enumerate(cols):
            for b in range(a):
                value = col ^ cols[b]
                word = (1 << a) | (1 << b)
                old = singles.get(value)
                if old is not None:
                    note(old ^ word)
                old = pairs.get(value)
                if old is not None:
                    note(old ^ word)
                pairs.setdefault(value, word)

        order = list(range(width))
        retained = sorted(set(found), key=int.bit_count)[:20]
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = cols[v], 1 << v
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = value, word
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word)
            light = sorted(set(retained+basis), key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    note(word ^ other)
            local.shuffle(basis)
            for word in light[:4]:
                for _ in range(2):
                    changed = False
                    for other in basis+retained:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(word)
            retained = sorted(set(found), key=int.bit_count)[:20]
            found = retained[:]
            if retained[0].bit_count() <= 4:
                break

        # Budgeted completion challenges cofactors and sampled circuits.
        # Anchors use only the regular symmetry of classical components.
        priorities = [local.random() for _ in cols]
        roots = list(range(0, width, q))
        local.shuffle(roots)
        best = [min(map(int.bit_count, found))]
        allowed_all = (1 << width)-1
        for root in roots:
            nodes = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if nodes[0] <= 0:
                    return
                nodes[0] -= 1
                if not value:
                    note(word)
                    best[0] = min(best[0], word.bit_count())
                    return
                left = min(8, best[0])-word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    note(word | one)
                    best[0] = min(best[0], (word | one).bit_count())
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(word | pair)
                    best[0] = min(best[0], (word | pair).bit_count())
                if left <= 2:
                    return
                chosen, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                choices = sorted(bits(chosen), key=lambda v: (
                    -(cols[v] & value).bit_count(), priorities[v]))
                live = allowed
                for v in choices:
                    live &= ~(1 << v)
                    visit(value ^ cols[v], word | (1 << v), live)
                    if nodes[0] <= 0:
                        break

            visit(cols[root], 1 << root,
                  allowed_all & ~((1 << (root+1))-1))
        rec["words"] = sorted(set(found), key=lambda w: (
            w.bit_count(), w))[:24]
        assert all(syndrome(cols, w) == 0 for w in rec["words"])
        rec["d"] = rec["words"][0].bit_count()
        rec["free4"] = rec["d"] > 4

    def pool(side, counts):
        nc = len(counts)
        mutable = list(range(3, nc)) if side == "A" else list(range(nc))
        records, known, seen = [], [], set()
        offset = 100003*int(side == "B")

        def loss(rec):
            return rec["cost"] + sum(
                6000*max(1, 9-w.bit_count())**2
                for w in known if syndrome(rec["cols"], w) == 0)

        def optimize(grid, steps):
            current = inspect(grid, side)
            if current is None:
                return None
            best = current
            for step in range(steps):
                proposal = _copy_grid(current["grid"])
                active = [w for w in known
                          if syndrome(current["cols"], w) == 0]
                choices = mutable
                if active and step % 3:
                    occupied = {v//q for v in bits(rng.choice(active))}
                    targeted = [j for j in mutable if j in occupied]
                    if targeted:
                        choices = targeted
                j = rng.choice(choices)
                draw_column(proposal, j, counts[j])
                trial = inspect(proposal, side)
                if trial is None:
                    continue
                delta = loss(trial)-loss(current)
                temperature = 100*(1-step/steps)**2+0.2
                if delta <= 0 or rng.random() < exp(
                        min(0.0, -delta/temperature)):
                    current = trial
                    if loss(current) < loss(best):
                        best = current
            return best

        for stage in range(3):
            starts = [fresh(side, counts) for _ in range(5)]
            if records:
                starts += [_copy_grid(r["grid"]) for r in sorted(
                    records, key=lambda r: (r["d"], -loss(r)),
                    reverse=True)[:2]]
            tested = []
            for grid in starts:
                rec = optimize(grid, 72 if stage == 0 else 96)
                if rec is None or rec["sig"] in seen:
                    continue
                seen.add(rec["sig"])
                query(rec, 12, offset+1009*stage, 4000)
                tested.append(rec)
                records.append(rec)
                for word in rec["words"][:4]:
                    if word.bit_count() <= 8 and word not in known:
                        known.append(word)
            known = sorted(set(known), key=int.bit_count)[:48]
        if not records:
            raise ValueError("empty support-margin component portfolio")
        records.sort(key=lambda r: (
            r["free4"], r["d"], -loss(r), r["rank"]), reverse=True)
        finalists = records[:3]
        for at, rec in enumerate(finalists):
            query(rec, 64, offset+700001+7919*at, 24000)
        finalists.sort(key=lambda r: (
            r["free4"], r["d"], -loss(r), r["rank"]), reverse=True)
        return finalists[:2]

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
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
        """Exact logical verification; reductions use original stabilizers."""
        upper = n+1
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = []
            for word in rec["words"][:6]:
                for block in range(6 if kind == "Z" else 7):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((6*j+block)*q)
                            for j in range(7))
                    else:
                        actual = word << (6*block*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.append(actual)
            local = random.Random(seed+3100009+100003*(kind == "X"))
            for word in sorted(set(candidates), key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        trial = word ^ row
                        if trial.bit_count() < word.bit_count():
                            word, changed = trial, True
                    if not changed:
                        break
                nearby = sorted(stabilizers, key=lambda row: (
                    row & word).bit_count(), reverse=True)[:32]
                for _ in range(96):
                    first, second = local.sample(nearby, 2)
                    trial = word ^ first ^ second
                    if trial.bit_count() < word.bit_count():
                        word = trial
                assert word and outside(word, pivots)
                assert all((word & row).bit_count() % 2 == 0
                           for row in opposite)
                upper = min(upper, word.bit_count())
        return upper

    aa, bb = pool("A", acounts), pool("B", bcounts)
    best = None
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1472 and k >= 640
                    and 425*n <= 1000*k < 475*n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) <= 12
            detected = min(a["d"], b["d"], localized(a, b, hx, hz, n))
            key = (a["free4"] and b["free4"],
                   k*detected*detected, detected,
                   -a["cost"]-b["cost"])
            if best is None or key > best[0]:
                best = key, A, B, k, a, b
    if best is None:
        raise ValueError("joint support-margin pairings missed cell 0.450")
    key, A, B, k, a, b = best
    return {"A": A, "B": B, "k": k, "card": (geometry, model, cohort),
            "quality": key, "component_readings": (a["d"], b["d"]),
            "component_ranks": (a["rank"], b["rank"]),
            "B_row_margins": tuple(map(sum, sb)),
            "screen": "original-span localized logical challenge"}

    # Historical implementation below is inactive.
    q = G.q
    mask = (1 << q) - 1
    card = idx % 20
    gap = (5, 6, 7, 9, 13)[card // 4]
    model = card % 4
    cohort = idx // 20

    # All these margins are exactly nine per component row.
    acounts = ((2, 0), (0, 2), (1, 1),
               (2, 1), (2, 1), (1, 2), (1, 2))
    bpatterns = (
        ((2, 1),) * 3 + ((1, 2),) * 3,
        ((2, 1),) * 3 + ((1, 2),) * 3,
        ((3, 0), (2, 1)) + ((1, 2),) * 4,
        ((3, 0), (0, 3), (2, 1), (2, 1), (1, 2), (1, 2)),
    )

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def syndrome(word, cols):
        value = 0
        for v in bits(word):
            value ^= cols[v]
        return value

    def translate_key(word, nc):
        """Canonical regular cyclic translate, only for classical components."""
        first = (word & -word).bit_length() - 1
        block = first // q
        choices = (word >> (block * q)) & mask
        answer = None
        for shift in bits(choices):
            translated = 0
            for j in range(nc):
                value = (word >> (j * q)) & mask
                value = ((value >> shift) |
                         (value << (q - shift))) & mask
                translated |= value << (j * q)
            if answer is None or translated < answer:
                answer = translated
        return answer

    def redraw(grid, j, counts, odd_binomials):
        """Whole-column mutation preserving explicit C4 exclusions."""
        nc = len(grid[0])
        within = [set(), set()]
        cross = set()
        for c in range(nc):
            if c == j:
                continue
            for i in range(2):
                for a, b in combinations(grid[i][c], 2):
                    d = (a[0] - b[0]) % q
                    within[i].add(min(d, (-d) % q))
            for a in grid[0][c]:
                for b in grid[1][c]:
                    cross.add((a[0] - b[0]) % q)

        for _ in range(512):
            column = [rng.sample(G.elements, count) for count in counts]
            legal = True
            for i in range(2):
                local = set()
                for a, b in combinations(column[i], 2):
                    d = (a[0] - b[0]) % q
                    key = min(d, (-d) % q)
                    if d == q // 2 or key in within[i] or key in local:
                        legal = False
                    if odd_binomials and len(column[i]) == 2 and not (d & 1):
                        legal = False
                    local.add(key)
            between = [(a[0] - b[0]) % q
                       for a in column[0] for b in column[1]]
            if len(set(between)) != len(between):
                legal = False
            if any(d in cross for d in between):
                legal = False
            if legal:
                for i in range(2):
                    grid[i][j] = column[i]
                return True
        return False

    def fresh(side, counts):
        nc = len(counts)
        for _ in range(48):
            grid = [[[] for _ in range(nc)] for _ in range(2)]
            if side == "A":
                # These gaps avoid the obvious short ring circuits at q=32.
                # Any stronger claim is tested on the verified component.
                grid[0][0] = [G.e, (1, 0)]
                grid[1][1] = [G.e, (gap, 0)]
                grid[0][2] = [G.e]
                grid[1][2] = [G.e]
                mutable = list(range(3, nc))
            else:
                mutable = list(range(nc))
                rng.shuffle(mutable)
            if all(redraw(grid, j, counts[j],
                          side == "B" and model == 1)
                   for j in mutable):
                return grid
        return None

    def multiply(a, b):
        out = 0
        for shift in bits(a):
            out ^= ((b << shift) | (b >> (q - shift))) & mask
        return out

    def inspect(grid, side):
        record = _component_record(G, grid, side)
        if record["cycles"][0] or record["rank"] != 2 * q:
            return None
        nc = len(grid[0])
        cols = record["cols"]
        # Polynomial coefficients come from VERIFIED column-zero syndromes.
        # The regular cyclic action then supplies their translated columns.
        polys = [[(cols[j * q] >> (i * q)) & mask
                  for j in range(nc)] for i in range(2)]
        minors = {}
        for a, b in combinations(range(nc), 2):
            minors[a, b] = (
                multiply(polys[0][a], polys[1][b]) ^
                multiply(polys[0][b], polys[1][a]))
        weights = []
        lightest = None
        for a, b, c in combinations(range(nc), 3):
            word = (minors[b, c] << (a * q) |
                    minors[a, c] << (b * q) |
                    minors[a, b] << (c * q))
            if word:
                assert syndrome(word, cols) == 0
                weights.append(word.bit_count())
                if lightest is None or word.bit_count() < lightest.bit_count():
                    lightest = word
        if not weights:
            return None
        record["cofactor"] = min(weights)
        record["cofactor_word"] = lightest
        c6 = record["cycles"][1]
        # Distinct cards and cohorts trade cofactor multiplicity against cycles.
        target = 9 + ((model + cohort) & 1)
        record["base"] = (
            4000 * max(0, 8 - min(weights)) +
            120 * sum(max(0, target - w) ** 2 for w in weights) +
            (1 + model) * c6)
        return record

    def short_word(record):
        """Exact detection through six for C4-free degree-at-least-two lifts.

        We first exclude weights <=4 by all pair syndromes and weight five
        by valid cyclic anchors. A six-word has two disjoint adjacent pairs:
        partition into two triples containing those pairs and collide their
        syndromes. Thus absence is exhaustive; a returned witness is verified.
        """
        cols = record["cols"]
        width = len(cols)
        single = {}
        pairs = {}
        for v, value in enumerate(cols):
            word = 1 << v
            if not value:
                return 1, word
            old = single.get(value)
            if old is not None:
                return 2, word ^ old
            single[value] = word
        for a, value in enumerate(cols):
            for b in range(a):
                key = value ^ cols[b]
                word = (1 << a) | (1 << b)
                old = single.get(key)
                if old is not None:
                    return 3, old ^ word
                old = pairs.get(key)
                if old is not None:
                    return 4, old ^ word
                pairs[key] = word
        for root in range(0, width, q):
            for a in range(width):
                if a == root:
                    continue
                value = cols[root] ^ cols[a]
                used = (1 << root) | (1 << a)
                for b in range(a):
                    if b == root:
                        continue
                    old = pairs.get(value ^ cols[b])
                    word = used | (1 << b)
                    if old is not None and not (old & word):
                        return 5, old | word
        del pairs
        adjacent = set()
        for row in record["rows"]:
            adjacent.update(combinations(list(bits(row)), 2))
        seen = {}
        for a, b in sorted(adjacent):
            value = cols[a] ^ cols[b]
            used = (1 << a) | (1 << b)
            for c in range(width):
                if c == a or c == b:
                    continue
                key = value ^ cols[c]
                word = used | (1 << c)
                old = seen.get(key)
                if old is not None and old != word:
                    witness = old ^ word
                    assert witness.bit_count() == 6
                    assert syndrome(witness, cols) == 0
                    return 6, witness
                seen[key] = word
        return 7, None

    def component_pool(side, counts):
        nc = len(counts)
        mutable = list(range(3, nc)) if side == "A" else list(range(nc))
        known = []
        screened = []
        signatures = set()

        def cost(record):
            return record["base"] + 50000 * sum(
                syndrome(word, record["cols"]) == 0 for word in known)

        def optimize(grid, steps):
            current = inspect(grid, side)
            if current is None:
                return None
            best = current
            current_cost = best_cost = cost(current)
            for step in range(steps):
                choices = mutable
                active = [word for word in known
                          if syndrome(word, current["cols"]) == 0]
                if active and step % 3 != 2:
                    occupied = {v // q for v in bits(rng.choice(active))}
                    targeted = [j for j in mutable if j in occupied]
                    if targeted:
                        choices = targeted
                j = rng.choice(choices)
                old = [list(grid[i][j]) for i in range(2)]
                if not redraw(grid, j, counts[j],
                              side == "B" and model == 1):
                    continue
                trial = inspect(grid, side)
                if trial is None:
                    for i in range(2):
                        grid[i][j] = old[i]
                    continue
                trial_cost = cost(trial)
                temperature = 80 * (1 - step / steps) ** 2 + 0.2
                accept = trial_cost <= current_cost or rng.random() < exp(
                    min(0.0, (current_cost - trial_cost) / temperature))
                if accept:
                    current, current_cost = trial, trial_cost
                    if trial_cost < best_cost:
                        best, best_cost = trial, trial_cost
                else:
                    for i in range(2):
                        grid[i][j] = old[i]
            return best

        def new_record(steps):
            for _ in range(8):
                grid = fresh(side, counts)
                if grid is not None:
                    record = optimize(grid, steps)
                    if record is not None:
                        return record
            return None

        pool = [r for r in (new_record(96) for _ in range(8))
                if r is not None]
        for stage in range(3):
            pool.sort(key=cost)
            tested = []
            for record in pool:
                signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                                  for row in record["grid"] for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                floor, word = short_word(record)
                record["floor"] = floor
                record["witness"] = word
                tested.append(record)
                screened.append(record)
                if word is not None:
                    assert word.bit_count() == floor
                    assert syndrome(word, record["cols"]) == 0
                    word = translate_key(word, nc)
                    assert syndrome(word, record["cols"]) == 0
                    if word not in known:
                        known.append(word)
                if len(tested) >= 6:
                    break
            if sum(r["floor"] == 7 for r in tested) >= 2:
                break
            tested.sort(key=lambda r: (
                r["floor"], r["cofactor"], -cost(r)), reverse=True)
            seeds = tested[:3] if tested else pool[:3]
            pool = []
            for record in seeds:
                repaired = optimize(_copy_grid(record["grid"]), 144)
                if repaired is not None:
                    pool.append(repaired)
            pool.extend(r for r in (new_record(64) for _ in range(3))
                        if r is not None)
        if not screened:
            raise ValueError("no full-rank asymmetric component survived")
        screened.sort(key=lambda r: (
            r["floor"], r["cofactor"], -cost(r)), reverse=True)
        answer = screened[:4]
        for record in answer:
            reading = _circuit_reading(rng, record["cols"], 32)
            if record["witness"] is not None:
                reading = min(reading, record["floor"])
            record["reading"] = reading
        answer.sort(key=lambda r: (
            r["floor"], r["reading"], -cost(r)), reverse=True)
        for record in answer[:2]:
            record["reading"] = min(
                record["reading"],
                _circuit_reading(rng, record["cols"], 96))
        for record in answer:
            record["quality"] = (
                record["floor"], record["reading"],
                record["cofactor"], -cost(record))
        answer.sort(key=lambda r: r["quality"], reverse=True)
        return answer

    aa = component_pool("A", acounts)
    bb = component_pool("B", bpatterns[model])
    proposals = []
    for a in aa:
        for b in bb:
            qa, qb = a["quality"], b["quality"]
            key = (min(qa[0], qb[0]), min(qa[1], qb[1]),
                   qa[0] + qb[0], qa[1] + qb[1],
                   qa[2] + qb[2], qa[3] + qb[3])
            proposals.append((key, a, b))
    proposals.sort(key=lambda item: item[0], reverse=True)
    for key, a, b in proposals:
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert all(sum(row) == 9 for row in sa + sb)
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 425 * n <= 1000 * k < 475 * n):
            continue
        assert n == 1472 and k >= 640
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {"A": A, "B": B, "k": k, "card": card,
                "quality": key, "gap": gap, "B_model": model,
                "component_floors": (a["floor"], b["floor"])}
    raise ValueError("asymmetric screened pairs missed cell 0.450")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly screened support-margin cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The Z11 (3,11)x(3,11) degree-2/3 monomial kernel-screening model improved on distance six; all five measured codes instead returned distance five, with thousands of sector-1 witnesses.
# REFUTED: Cycle-first optimization of five Z28 backbone/support profiles suppressed weight-five and weight-six sector-1 logicals; the measured codes instead returned 5,5,6,6,6 at k=700, and several backbone slots were isomorphic duplicates.
# REFUTED: Mixed-column Z28 ring-backbone repair produced five readings >=7; the measured top five instead returned 7,6,6,6,6, with minimum witnesses entirely in sector 1.
# REFUTED: Independent optimized Z32 voltage fillings alone prevent slot equivalences; the measured parent produced 28 distinct codes from 60 slots, with 32 isomorphic duplicates.
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
