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
CONCEPT = ("On (C7 semidirect_2 C3) x C5, order 105, A=B=(1,3), matched GF16 quotient-root trinomials should produce five new codes with k>=444 and provisional distance readings twelve, exceeding the 0.400 frontier's 59.5 incumbent. Failure to retain those readings refutes this experiment; finite witness searches certify no distance.")
GROUP = ("Retain the direct-product construction rule, resizing only its central cyclic factor from C4 to C5 because the new factor-ten shape needs a larger lift. q=105 gives n=1050. The C3 x C5 quotient is cyclic of order fifteen and has degree-four binary characters. Their explicitly checked trinomial zeros propose common annihilators. The related order-147 three-trinomial bed measured distance twelve, but that reading does not establish distance transfer.")
SHAPES = ("Change both sides to (1,3): equal components have 315 variables and 105 checks, nominal rate two thirds. n=1050 and k>=420 guarantee rate 0.400; the shape permits q<=149. Target cell 0.400 instead of the named high-rate plateau: three trinomials fit both row caps, whereas the parent repeatedly measured k=756 and distance six. Its minima were entirely sector one, especially Acol1/2 across Bcol0..2. Actual full ranks must place every selected pairing in [0.375,0.425).")
FILLING = ("All cells contain three distinct terms, giving component row totals nine and both CSS maxima twelve. Twenty cards cross five coupled quotient-support associations with four actual cycle objectives. Each cubic vanishes at a fixed GF16 character; A and B use matched positive quotient zeros, with their different authoritative left/right component annihilators checked explicitly. Prefer component rank q-4, exact kernel-four exclusion and exact component kernel-six exclusion. Retain genuine unrestricted and C5-fixed witnesses across independent information sets. Final localized logicals are checked modulo original stabilizer spans. Actual k>=444, complete kernel-four, native component agreement, weights and CSS remain binding. The target reading twelve is a falsifiable measurement hypothesis, not a certified distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040_Frob21xC5_13x13_GF16root_rank4_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 105

def custom_ops(q):
    # Same direct-product rule; resize the central cyclic factor only.
    # (a,b,z) denotes x^a y^b times the central C5 element z.
    if not valid_q(q):
        raise ValueError("Frobenius21 x C5 requires order 105")
    action = (1, 2, 4)
    central_order = 5
    els = [(a, b, z) for z in range(central_order)
           for b in range(3) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 3, g[2] % central_order)

    def mul(g, h):
        return ((g[0] + action[g[1]] * h[0]) % 7,
                (g[1] + h[1]) % 3,
                (g[2] + h[2]) % central_order)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-action[bi] * g[0]) % 7, bi,
                (-g[2]) % central_order)

    def idx(g):
        a, b, z = red(g)
        return 7 * (3*z + b) + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Six triple columns, written column-major. Every profile sums to
# (7,5,6), completing the anchored skeleton's row totals to (9,9,9).
_ANCHOR_TRIPLES = (
    ((2,0,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
    ((2,1,0), (1,0,2), (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
    ((2,0,1), (1,2,0), (1,0,2), (1,1,1), (1,1,1), (1,1,1)),
    ((2,0,1), (2,1,0), (0,2,1), (1,0,2), (1,1,1), (1,1,1)),
    ((2,0,1), (2,0,1), (1,2,0), (1,1,1), (1,1,1), (0,1,2)),
)
_ANCHOR_FRAMES = ((0,1,3), (0,1,4), (0,1,5), (0,2,5))


def _anchor_columns(G, cells, side):
    """Obtain component columns only through the verified lift helper.

    The empty dummy side contributes no sector-two entries. Its dead
    lines are used only in this extraction call, never in a submitted code.
    """
    width = len(cells[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, cells, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], cells)
    mask = (1 << width) - 1
    cols = [0] * width
    for r, raw in enumerate(rows):
        raw &= mask
        while raw:
            bit = raw & -raw
            cols[bit.bit_length() - 1] |= 1 << r
            raw ^= bit
    return cols


def _anchor_pairs(cols):
    """Exact exclusion of all component kernels of weights one through four."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = set()
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs.add(syndrome)
    return pairs


def _anchor_spectrum(cols, pairs, limit5=None):
    """Exact weight-five/six counts, conditional on the kernel-four screen."""
    triples = {}
    hits5 = hits6 = 0
    limit = None if limit5 is None else 10 * limit5
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for k in range(j):
                syndrome = pair ^ cols[k]
                if syndrome in pairs:
                    hits5 += 1
                    if limit is not None and hits5 > limit:
                        return None
                old = triples.get(syndrome, 0)
                hits6 += old
                triples[syndrome] = old + 1

    # Overlapping matching subsets would give a kernel of weight <=4,
    # already excluded. A five-word has ten pair/triple partitions;
    # a six-word has ten unordered triple/triple partitions.
    assert hits5 % 10 == 0 and hits6 % 10 == 0
    check_pairs = {}
    four = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, u in enumerate(checks):
            for v in checks[:a]:
                key = (v, u)
                old = check_pairs.get(key, 0)
                four += old
                check_pairs[key] = old + 1
    return hits5 // 10, hits6 // 10, four


def _anchor_seed(G, frame):
    """A singleton, a branch, and three Sidon-voltage junction columns."""
    cells = [[[] for _ in range(5)] for _ in range(3)]
    cells[0][0] = [G.e]
    cells[0][1] = [G.e]
    cells[1][1] = [G.e]
    for j, voltage in enumerate(frame, 2):
        cells[1][j] = [G.e]
        cells[2][j] = [(voltage, 0)]
    return cells


def _anchor_component(rng, G, side, profile, frame, emphasis):
    """Greedy kernel-four exclusion, followed by exact spectrum selection."""
    counts = _ANCHOR_TRIPLES[profile]
    seed = _anchor_seed(G, _ANCHOR_FRAMES[frame])
    assert _anchor_pairs(_anchor_columns(G, seed, side)) is not None
    best = None
    best_key = None
    completed = 0

    def key_of(spectrum):
        n5, n6, four = spectrum
        # The first objective always minimizes actual weight-five words.
        # Cards differ in the next objective, not just in a group gauge.
        return (n5, n6, four) if emphasis == 0 else (n5, four, n6)

    for attempt in range(40):
        cells = [[list(cell) for cell in row] for row in seed]
        good = True
        for column_counts in counts:
            accepted = False
            for trial in range(96):
                column = [rng.sample(G.elements, count)
                          for count in column_counts]
                candidate = [row + [column[i]]
                             for i, row in enumerate(cells)]
                cols = _anchor_columns(G, candidate, side)
                pairs = _anchor_pairs(cols)
                if pairs is None:
                    continue
                cells = candidate
                accepted = True
                break
            if not accepted:
                good = False
                break
        if not good:
            continue
        completed += 1
        spectrum = _anchor_spectrum(
            cols, pairs, None if best_key is None else best_key[0])
        if spectrum is not None:
            key = key_of(spectrum)
            if best_key is None or key < best_key:
                best, best_key = cells, key
        if completed >= 8:
            break

    if best is None:
        raise ValueError("anchored component could not exclude kernels <=4")

    # Change actual labels, including binomial differences, while retaining
    # this card's support. Every accepted move is a strict improvement.
    occupied = [(i, j) for i in range(3) for j in range(5, 11)
                if best[i][j]]
    for move in range(16):
        i, j = occupied[rng.randrange(len(occupied))]
        candidate = [[list(cell) for cell in row] for row in best]
        candidate[i][j] = rng.sample(G.elements, len(best[i][j]))
        cols = _anchor_columns(G, candidate, side)
        pairs = _anchor_pairs(cols)
        if pairs is None:
            continue
        spectrum = _anchor_spectrum(cols, pairs, best_key[0])
        if spectrum is None:
            continue
        key = key_of(spectrum)
        if key < best_key:
            best, best_key = candidate, key
    return best, best_key


# Twenty trinomial construction recipes, not automorphism-generated slots.
# First coordinates here are C3 cosets; second profiles are C4 cosets.
_BP_FRAMES = ((0,0,0), (0,0,1), (0,1,1), (0,1,2), (0,2,2))
_BP_TRIANGLES = ((0,0,0), (0,0,2), (0,1,2), (0,1,3))


def _bp_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _bp_backbone(rng, G, frame):
    """Require connectedness and no nonbacktracking length-two collision.

    Odd C4 coordinates make the check graph bipartite. Distinct signed
    generators exclude doubled edges; unique reduced two-step products
    exclude its four-cycles. Final component screens use verified lifts.
    """
    def legal(gs):
        ns = gs + [G.inv(g) for g in gs]
        keys = {G._key(g) for g in ns}
        if len(keys) != 6 or G._key(G.e) in keys:
            return False
        products = set()
        for g in ns:
            inverse = G._key(G.inv(g))
            for h in ns:
                if G._key(h) == inverse:
                    continue
                p = G._key(G.mul(g, h))
                if p in products or p == G._key(G.e):
                    return False
                products.add(p)
        reached = {G._key(G.e)}
        todo = [G.e]
        while todo:
            g = todo.pop()
            for h in ns:
                p = G.mul(g, h)
                key = G._key(p)
                if key not in reached:
                    reached.add(key)
                    todo.append(p)
        return len(reached) == G.q

    for trial in range(128):
        gs = [(rng.randrange(1, 7), 0, rng.choice((1, 3))),
              (rng.randrange(7), 1, rng.choice((1, 3))),
              (rng.randrange(7), 2, rng.choice((1, 3)))]
        if legal(gs):
            return 0, gs

    # Explicit legal seed, independently checked rather than trusted.
    gs = [(1, 0, 1), (0, 1, 1), (4, 2, 1)]
    if not legal(gs):
        raise ValueError("explicit Frobenius21 x C4 backbone failed")
    return 0, gs


def _bp_record(G, cells, side):
    """All matrix syndromes use the scaffold's verified conventions."""
    if side == "A":
        rows, _, _ = lifted_matrices(G, [cells], [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], [cells])
    width = 4 * G.q
    mask = (1 << width) - 1
    rows = [row & mask for row in rows]
    columns = [0] * width
    incidence = [[] for _ in rows]
    for r, row in enumerate(rows):
        for v in _bp_bits(row):
            columns[v] |= 1 << r
            incidence[r].append(v)

    adjacency = [set() for _ in columns]
    for variables in incidence:
        for v in variables:
            adjacency[v].update(u for u in variables if u != v)
    return rows, columns, adjacency


def _bp_small_floor(columns, adjacency):
    """Exact exclusion through five; capped counts of actual five-words.

    Degrees are two or three. A five-word must contain an even number
    of degree-three variables. Five degree-two variables cannot form a
    kernel in the screened bipartite backbone. Every remaining five-word
    therefore has a partition containing two degree-three variables and
    one degree-two variable on its triple side.
    """
    width = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1, 0
    if len(singles) != width:
        return 2, 0
    pairs = {}
    bits = [1 << v for v in range(width)]
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles:
                return 3, 0
            if syndrome in pairs:
                return 4, 0
            pairs[syndrome] = bits[a] | bits[b]

    doubles = [v for v, x in enumerate(columns) if x.bit_count() == 2]
    triples = [v for v, x in enumerate(columns) if x.bit_count() == 3]
    assert len(doubles) + len(triples) == width
    incidence = {}
    for v in doubles:
        for r in _bp_bits(columns[v]):
            incidence.setdefault(r, []).append(v)

    found = set()
    for t, a in enumerate(triples):
        for b in triples[:t]:
            syndrome = columns[a] ^ columns[b]
            used = bits[a] | bits[b]
            if syndrome.bit_count() == 6:
                # A disjoint double would leave weight eight, larger
                # than every possible pair syndrome (at most six).
                candidates = set()
                for r in _bp_bits(syndrome):
                    candidates.update(incidence.get(r, ()))
            else:
                candidates = doubles
            for c in candidates:
                previous = pairs.get(syndrome ^ columns[c])
                if previous is None or previous & (used | bits[c]):
                    continue
                word = previous | used | bits[c]
                assert word.bit_count() == 5
                found.add(word)
                if len(found) >= 256:
                    return 5, 256
    return (5, len(found)) if found else (6, 0)


def _bp_reading(rng, columns, trials):
    """Genuine classical kernel witnesses; sampled upper readings only."""
    width = len(columns)
    order = list(range(width))
    best = width + 1
    for trial in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for v in order:
            syndrome, word = columns[v], 1 << v
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

        words.sort(key=int.bit_count)
        for a, word in enumerate(words[:20]):
            for other in words:
                candidate = word ^ other
                if candidate:
                    best = min(best, candidate.bit_count())
        if best <= 5:
            break
    return best


def _bp_cycles(columns):
    """Exact Tanner four-cycle count, used only as a secondary objective."""
    pairs = {}
    count = 0
    for syndrome in columns:
        checks = list(_bp_bits(syndrome))
        for a, u in enumerate(checks):
            for v in checks[:a]:
                key = (v, u)
                count += pairs.get(key, 0)
                pairs[key] = pairs.get(key, 0) + 1
    return count


def _bp_component_pool(rng, G, side, frame, mode):
    """Optimize actual trinomial geometry within one of twenty recipes."""
    bs = _BP_FRAMES[frame]
    zs = _BP_TRIANGLES[mode]
    pool = []

    def draw_tri():
        tri = []
        for b, z in zip(bs, zs):
            available = [(a, b, z) for a in range(7)
                         if (a, b, z) not in tri]
            tri.append(rng.choice(available))
        return tri

    def inspect(cells):
        rows, columns, adjacency = _bp_record(G, cells, side)
        if gf2_rank(rows) != G.q:
            return None
        floor, n5 = _bp_small_floor(columns, adjacency)
        return {"cells": cells, "columns": columns, "floor": floor,
                "n5": n5, "cycles": _bp_cycles(columns),
                "reading": len(columns) + 1}

    def geometry(item):
        return (item["floor"], -item["n5"], -item["cycles"])

    def remember(item):
        if item is None:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in item["cells"])
        if any(signature == old["signature"] for old in pool):
            return
        item["signature"] = signature
        pool.append(item)
        pool.sort(key=geometry, reverse=True)
        del pool[6:]

    for attempt in range(48):
        _, gs = _bp_backbone(rng, G, frame)
        cells = [[G.e, g] for g in gs] + [draw_tri()]
        remember(inspect(cells))
        if len(pool) >= 6 and sum(x["floor"] == 6 for x in pool) >= 3:
            break
    if not pool:
        raise ValueError("no full-rank direct-product component")

    # Repair actual exponent geometry, preserving the card's quotient
    # profile and the already legal binomial backbone.
    current = pool[0]
    for move in range(16):
        cells = [list(cell) for cell in current["cells"]]
        if move % 4 == 0:
            cells[3] = draw_tri()
        else:
            t = rng.randrange(3)
            old = cells[3][t]
            available = [(a, old[1], old[2]) for a in range(7)
                         if (a, old[1], old[2]) not in cells[3]]
            if not available:
                continue
            cells[3][t] = rng.choice(available)
        item = inspect(cells)
        remember(item)
        if item is not None and geometry(item) >= geometry(current):
            current = item

    finalists = sorted(pool, key=geometry, reverse=True)[:4]
    for item in finalists:
        item["reading"] = _bp_reading(rng, item["columns"], 16)
        item["key"] = (item["floor"], -item["n5"],
                       item["reading"], -item["cycles"])
    finalists.sort(key=lambda item: item["key"], reverse=True)

    # Confirm several independent candidates, not one lucky reading.
    for item in finalists[:3]:
        item["reading"] = min(
            item["reading"], _bp_reading(rng, item["columns"], 48))
        item["key"] = (item["floor"], -item["n5"],
                       item["reading"], -item["cycles"])
    finalists.sort(key=lambda item: item["key"], reverse=True)
    return finalists[:3]


def _bp_full_kernel4_free(rows, width):
    """Exclude every complete-check-matrix kernel word of weight <=4.

    This is stronger than checking only components. It does not exclude
    weight-five logicals or certify the proposed distance-six target.
    """
    columns = [0] * width
    for r, row in enumerate(rows):
        for v in _bp_bits(row):
            columns[v] |= 1 << r
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def NEW_CTX(rng, G, idx):
    """Matched quotient roots, authoritative ranks and accumulated witnesses."""
    q, width = G.q, 3*G.q
    assert q == 105 and A_SHAPE == B_SHAPE == (1, 3)
    card, cohort = idx % 20, idx // 20
    model, objective = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    support = [[3, 3, 3]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    def fm(a, b):
        # GF16 = F2[t]/(t^4+t+1).
        value = 0
        while b:
            if b & 1:
                value ^= a
            b >>= 1
            a <<= 1
            if a & 16:
                a ^= 19
        return value

    def fp(a, exponent):
        value = 1
        while exponent:
            if exponent & 1:
                value = fm(value, a)
            a = fm(a, a)
            exponent >>= 1
        return value

    assert fp(2, 15) == 1
    assert fp(2, 3) != 1 and fp(2, 5) != 1
    alpha, beta = fp(2, 5), fp(2, 3)
    assert fp(alpha, 3) == fp(beta, 5) == 1

    def character(g):
        return fm(fp(alpha, g[1]), fp(beta, g[2]))

    quotient = [(b, z) for b in range(3) for z in range(5)]
    patterns = []
    for at, first in enumerate(quotient):
        if first == (0, 0):
            continue
        x = fm(fp(alpha, first[0]), fp(beta, first[1]))
        for second in quotient[:at]:
            if second == (0, 0):
                continue
            y = fm(fp(alpha, second[0]), fp(beta, second[1]))
            if 1 ^ x ^ y:
                continue
            if first[1] == second[1] == 0:
                continue
            if first[0] == second[0] == 0:
                continue
            patterns.append((first, second))
    assert len(patterns) >= 5

    # Cache native operations, then obtain BOTH binary conventions
    # exclusively from the authoritative constructor.
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

    P = Probe()
    blocks = {"A": [], "B": []}
    for start in range(0, q, 15):
        ids = list(range(start, min(q, start+15)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j*q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j*q)) & mask for row in zz])

    functionals = {}
    for side in ("A", "B"):
        values = [
            character(g) if side == "A" else fp(character(g), 14)
            for g in G.elements]
        functionals[side] = [
            sum(1 << r for r, value in enumerate(values)
                if (value >> bit) & 1)
            for bit in range(4)]
        assert gf2_rank(functionals[side]) == 4

    def columns_of(rows, length):
        columns = [0]*length
        for r, row in enumerate(rows):
            for c in _bp_bits(row):
                columns[c] |= 1 << r
        return columns

    def syndrome(columns, word):
        value = 0
        for c in _bp_bits(word):
            value ^= columns[c]
        return value

    def component(cells, side):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, value in enumerate(blocks[side][G.idx(g)]):
                    rows[r] ^= value << (j*q)
        return rows, columns_of(rows, width)

    def fresh_cell(pattern_id):
        first, second = patterns[pattern_id % len(patterns)]
        for _ in range(64):
            g = (rng.randrange(7), first[0], first[1])
            h = (rng.randrange(7), second[0], second[1])
            if G.mul(g, h) != G.mul(h, g):
                cell = [G.e, g, h]
                assert len(set(cell)) == 3
                assert character(G.e) ^ character(g) ^ character(h) == 0
                return cell
        raise ValueError("noncommuting quotient-root cubic unavailable")

    def exact_six(columns):
        """Exact component six-word challenge after kernel-four exclusion.

        All columns have odd degree three, so odd kernel weights are
        impossible. Regular component symmetry supplies one root per block.
        This is a classical exclusion, not a quantum-distance certificate.
        """
        singles = {value: 1 << c for c, value in enumerate(columns)}
        pairs, incidence = {}, [0]*q
        for c, value in enumerate(columns):
            for r in _bp_bits(value):
                incidence[r] |= 1 << c
            for other in range(c):
                key = value ^ columns[other]
                assert key not in pairs and key not in singles
                pairs[key] = (1 << c) | (1 << other)

        def visit(value, word, allowed):
            if not value:
                return word
            left = 6-word.bit_count()
            if left <= 0 or value.bit_count() > 3*left:
                return None
            one = singles.get(value, 0)
            if one and one & allowed == one:
                return word | one
            pair = pairs.get(value, 0)
            if left >= 2 and pair and pair & allowed == pair:
                return word | pair
            if left <= 2:
                return None
            choices, smallest = 0, width+1
            for r in _bp_bits(value):
                available = incidence[r] & allowed
                size = available.bit_count()
                if not size:
                    return None
                if size < smallest:
                    choices, smallest = available, size
            live = allowed
            for c in sorted(_bp_bits(choices), key=lambda c: (
                    -(columns[c] & value).bit_count(), c)):
                bit = 1 << c
                live &= ~bit
                found = visit(value ^ columns[c], word | bit, live)
                if found is not None:
                    return found
            return None

        all_variables = (1 << width)-1
        for root in range(0, width, q):
            found = visit(columns[root], 1 << root,
                          all_variables ^ (1 << root))
            if found is not None:
                assert found.bit_count() == 6
                assert syndrome(columns, found) == 0
                return found
        return None

    def search(columns, diagnostic, trials, initial=()):
        """Genuine kernel words; independent challenges only lower readings."""
        length = len(columns)
        retained = set(initial)
        order = list(range(length))
        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 4 == 3:
                anchor = diagnostic.choice(sorted(
                    retained, key=lambda w: (w.bit_count(), w))[:6])
                active = list(_bp_bits(anchor))
                front = diagnostic.sample(active, max(1, len(active)//2))
                selected = set(front)
                order = front + [c for c in order if c not in selected]
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length()-1
                    previous = pivots.get(lead)
                    if previous is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= previous[0]
                    word ^= previous[1]
                if not value:
                    basis.append(word)
            light = sorted(set(basis) | retained,
                           key=lambda w: (w.bit_count(), w))[:16]
            grown = set(light) | retained
            moves = basis + list(retained)
            for word in light:
                for other in moves:
                    proposed = word ^ other
                    if proposed and proposed.bit_count() <= 18:
                        grown.add(proposed)
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for _ in range(2):
                    before = word.bit_count()
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word = proposed
                    if word.bit_count() == before:
                        break
                grown.add(word)
            retained = set(sorted(
                grown, key=lambda w: (w.bit_count(), w))[:24])
        assert retained
        assert all(syndrome(columns, word) == 0 for word in retained)
        return sorted(retained, key=lambda w: (w.bit_count(), w))

    central_parts = [
        tuple(j*q + G.idx((a, b, z)) for z in range(5))
        for j in range(3) for b in range(3) for a in range(7)]

    def probe(record, side, trials, turn):
        offset = 0 if side == "A" else 100003
        diagnostic = random.Random(seed+offset+700001*turn)
        words = search(record["columns"], diagnostic, trials,
                       record.get("words", ()))
        compressed = [
            syndrome(record["columns"],
                     sum(1 << c for c in positions))
            for positions in central_parts]
        fixed_words = search(
            compressed, random.Random(seed+offset+1700003+900001*turn),
            max(4, trials//3))
        for word in fixed_words[:4]:
            actual = sum(
                1 << c for at in _bp_bits(word)
                for c in central_parts[at])
            assert syndrome(record["columns"], actual) == 0
            words.append(actual)
        record["words"] = sorted(
            set(words), key=lambda w: (w.bit_count(), w))[:24]
        record["reading"] = record["words"][0].bit_count()

    def inspect(cells, side):
        rows, columns = component(cells, side)
        assert all(column.bit_count() == 3 for column in columns)
        assert all((functional & column).bit_count() % 2 == 0
                   for functional in functionals[side] for column in columns)
        rank = gf2_rank(rows)
        if rank != q-4:
            return None
        cycles = _bp_cycles(columns)
        # Projection triangles are only a secondary geometry heuristic.
        adjacency = [0]*q
        for column in columns:
            for r in _bp_bits(column):
                adjacency[r] |= column ^ (1 << r)
        triangles = 0
        for r, neighbors in enumerate(adjacency):
            for s in _bp_bits(neighbors & (-1 << (r+1))):
                triangles += (
                    neighbors & adjacency[s] & (-1 << (s+1))).bit_count()
        return dict(cells=cells, rows=rows, columns=columns, rank=rank,
                    cycles=cycles, excess=max(0, triangles-width))

    def geometry(record):
        four, six = record["cycles"], record["excess"]
        weights = (8, 16, 32, 64)
        if objective == 3:
            return (four, six, record["cells"][0][1][0])
        return (four, weights[objective]*six+four, six)

    def pool(side):
        shift = (model+cohort) % len(patterns)
        plan = tuple((shift+j+(objective if side == "B" else 0))
                     % len(patterns) for j in range(3))
        records, signatures = [], set()

        def remember(cells):
            signature = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell)) for cell in cells))
            if signature in signatures:
                return None
            signatures.add(signature)
            record = inspect(cells, side)
            if record is not None:
                records.append(record)
                records.sort(key=geometry)
                del records[32:]
            return record

        for _ in range(96):
            remember([fresh_cell(p) for p in plan])
        if not records:
            raise ValueError("no rank-four quotient-root component")

        current = records[0]
        for move in range(32):
            cells = [list(cell) for cell in current["cells"]]
            j = rng.randrange(3)
            cells[j] = fresh_cell(plan[j])
            proposed = remember(cells)
            if proposed is not None and geometry(proposed) <= geometry(current):
                current = proposed

        screened = []
        for record in sorted(records, key=geometry):
            if _anchor_pairs(record["columns"]) is None:
                continue
            word = exact_six(record["columns"])
            record["free6"] = word is None
            record["words"] = [] if word is None else [word]
            probe(record, side, 8, 1+len(screened))
            screened.append(record)
            if len(screened) >= 10:
                break
        if not screened:
            raise ValueError("quotient-root components failed kernel-four")

        def merit(record):
            return (record["free6"], min(12, record["reading"]),
                    record["reading"], -record["cycles"],
                    -record["excess"])

        screened.sort(key=merit, reverse=True)
        finalists = screened[:3]
        for at, record in enumerate(finalists):
            probe(record, side, 32, 101+at)
            probe(record, side, 48+8*(cohort % 3), 211+at)
        finalists.sort(key=merit, reverse=True)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                previous = pivots.get(lead)
                if previous is None:
                    pivots[lead] = word
                    break
                word ^= previous
        return pivots

    def outside(word, pivots):
        while word:
            previous = pivots.get(word.bit_length()-1)
            if previous is None:
                return True
            word ^= previous
        return False

    def logical_reading(a, b, hx, hz, n):
        reading, found = n+1, 0
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            columns = columns_of(opposite, n)
            for word in record["words"][:6]:
                for fixed in range(3):
                    actual = (sum(
                        ((word >> (j*q)) & mask) << ((3*j+fixed)*q)
                        for j in range(3)) if kind == "Z"
                        else word << (fixed*3*q))
                    assert syndrome(columns, actual) == 0
                    if not outside(actual, pivots):
                        continue
                    for _ in range(2):
                        before = actual.bit_count()
                        for row in stabilizers:
                            proposed = actual ^ row
                            if proposed.bit_count() < actual.bit_count():
                                actual = proposed
                        if actual.bit_count() == before:
                            break
                    assert actual and outside(actual, pivots)
                    assert syndrome(columns, actual) == 0
                    reading = min(reading, actual.bit_count())
                    found += 1
        return reading, found

    aa, bb = pool("A"), pool("B")
    proposals = []
    for a in aa:
        for b in bb:
            A, B = [a["cells"]], [b["cells"]]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 1050
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (k >= 444 and 375*n <= 1000*k < 425*n):
                continue
            actual_a = [
                sum(((hx[r] >> (3*j*q)) & mask) << (j*q)
                    for j in range(3)) for r in range(q)]
            actual_b = [row & ((1 << width)-1) for row in hz[:q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            dq, hits = logical_reading(a, b, hx, hz, n)
            detected = min(12, a["reading"], b["reading"], dq)
            key = (k*detected*detected, detected,
                   a["free6"] and b["free6"], -hits,
                   -a["cycles"]-b["cycles"],
                   -a["excess"]-b["excess"])
            proposals.append((key, A, B, hx, hz, k, a, b, dq))

    proposals.sort(key=lambda item: item[0], reverse=True)
    for key, A, B, hx, hz, k, a, b, dq in proposals:
        if not _bp_full_kernel4_free(hx, 1050):
            continue
        if not _bp_full_kernel4_free(hz, 1050):
            continue
        assert max(row.bit_count() for row in hx+hz) == 12
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, k=k, idx=idx, card=(model, objective, cohort),
            quality=key, target_cell=0.400,
            component_ranks=(a["rank"], b["rank"]),
            component_readings=(a["reading"], b["reading"]),
            component_kernel6_free=(a["free6"], b["free6"]),
            verified_quotient_annihilator_dimensions=(4, 4),
            verified_localized_logical_reading=(
                None if dq > 1050 else dq),
            component_cycles=(a["cycles"], b["cycles"]),
            quotient_patterns=patterns)
    raise ValueError("matched-root pairings missed actual rate/validity gates")


def CELL(rng, G, side, i, j, ctx):
    """Every new-shape cell is a stored, complete three-term support."""
    cell = list(ctx[side][i][j])
    assert len(cell) == 3
    assert len({G.idx(g) for g in cell}) == 3
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
# REFUTED: C7:C12(s=3) odd-quotient backbones and kernel-five selection produced five readings >=6; all five measured codes instead returned weight four with abundant sector-one X and Z minima.
# REFUTED: Connected-backbone trinomial repair alone raises this bed beyond its distance-six plateau; all five current measured winners remained k=756, distance six, with abundant sector-one minima.
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
