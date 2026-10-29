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
CONCEPT = ("On (C7 semidirect_2 C3) x C3, order 63, A=(1,3), B=(2,7), cubic A columns and balanced degree-two/three B columns with binding short-witness repair should raise five-code fitness above 28.6 in cell 0.450. Non-improvement refutes this experiment; finite witness searches certify no distance.")
GROUP = ("Preserve the Frobenius-times-cyclic construction rule and the C7 semidirect_2 C3 factor; resize only the central cyclic order from four to three. q=63 gives n=1449. This exact destination bed is untried. Related (1,3)x(2,7) beds measured distance eight, while the parent repeatedly measured six; neither reading establishes distance transfer.")
SHAPES = ("Change to A=(1,3), B=(2,7): factor 23 gives n=1449, k>=630 and guaranteed rate 10/23=0.434783 in cell 0.450, with q_max=65. Components have 189 variables/63 checks and 441 variables/126 checks, nominal rates 2/3 and 5/7. Redirect from cell 0.550 because the parent's abundant weight-six minima remained entirely sector one, especially Acol1/2 across Bcol0..2. Three degree-two B blocks replace its three-binomial degree-six check graph.")
FILLING = ("A counts (3,3,3); five B support topologies have row totals nine and column totals (2,2,2,3,3,3,3), so both CSS row caps are twelve. Twenty cards cross these actual support topologies with four binding cubic quotient prescriptions. Authoritative monomial blocks assemble components; exact kernel-through-four checks, degree-two cycle witnesses, bounded anchored parity completion and independently resampled kernel circuits challenge labels. Witness-directed whole-cell replacement preserves the card. Finalists are challenged independently and paired using actual ranks and nontrivial logical embeddings modulo original stabilizer spans. Complete kernel-four, native-component agreement, weights and CSS remain binding. Missing witnesses are not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r045_Frob21xC3_13x27_balanced_cubic_witness_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 63

def custom_ops(q):
    # Same direct-product construction; only the central order changes.
    # (C7 semidirect_2 C3) x C3 has order 7*3*3=63.
    if not valid_q(q):
        raise ValueError("Frobenius21 x C3 requires order 63")
    action = (1, 2, 4)
    central_order = 3
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
B_SHAPE = (2, 7)               # (rb, cb)


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
    """Twenty balanced-shape cards with binding authoritative witnesses."""
    import math

    q = G.q
    assert q == 63 and A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    card, cohort = idx % 20, idx // 20
    topology, model = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    ident = G.idx(G.e)

    bplans = (
        ((1,1,1,3,3,0,0), (1,1,1,0,0,3,3)),
        ((1,1,1,3,2,1,0), (1,1,1,0,1,2,3)),
        ((1,1,1,2,2,1,1), (1,1,1,1,1,2,2)),
        ((2,0,1,2,2,1,1), (0,2,1,1,1,2,2)),
        ((2,0,1,3,3,0,0), (0,2,1,0,0,3,3)),
    )
    sa = [[3, 3, 3]]
    sb = [list(row) for row in bplans[topology]]
    assert all(sum(row) == 9 for row in sb)
    assert [sum(row[j] for row in sb) for j in range(7)] \
        == [2,2,2,3,3,3,3]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid group order")

    # Cache only monomial blocks supplied by the verified constructor.
    mono = [[[g] for g in G.elements]]
    left, _, _ = lifted_matrices(G, mono, [[[]]])
    _, right, _ = lifted_matrices(G, [[[]]], mono)
    blocks = {
        "A": [[(row >> (j*q)) & mask for row in left]
              for j in range(q)],
        "B": [[(row >> (j*q)) & mask for row in right]
              for j in range(q)],
    }
    del left, right

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def sidon(ids):
        wanted = len(ids)*(len(ids)-1)
        return (len(set(ids)) == len(ids)
                and len({table[inverse[g]][h]
                         for g in ids for h in ids if g != h}) == wanted
                and len({table[g][inverse[h]]
                         for g in ids for h in ids if g != h}) == wanted)

    # Binding quotient pairs, not a common automorphism applied to slots.
    quotient_pairs = (
        ((1,0), (2,1)),
        ((0,1), (1,0)),
        ((1,1), (1,2)),
        ((1,1), (2,0)),
    )
    long_bins = [g for g in range(q) if orders[g] in (7, 21)]

    def fresh_cell(count, side, i, j):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            # Pure degree-two cells must not introduce order-three cycles.
            root = rng.randrange(q)
            ratio = rng.choice(long_bins)
            return [root, table[root][ratio]]

        assert count == 3
        # Relative quotient prescription is retained after every repair.
        policy = (model+j+i+int(side == "B")+cohort) % 4
        (b1, z1), (b2, z2) = quotient_pairs[policy]
        for _ in range(1024):
            ids = [ident,
                   G.idx((rng.randrange(7), b1, z1)),
                   G.idx((rng.randrange(7), b2, z2))]
            if not sidon(ids):
                continue
            if table[ids[1]][ids[2]] == table[ids[2]][ids[1]]:
                continue
            return ids
        raise ValueError("binding cubic quotient prescription exhausted")

    def fresh(support, side):
        return [[fresh_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def inspect(grid, side):
        width = len(grid[0])*q
        rows = [0]*(len(grid)*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if 0 in columns or len(set(columns)) != width:
            return None

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
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        excess = max(0, triangles-forced)
        cost = (600+200*model)*four + (2+model)*excess
        return dict(grid=[[list(c) for c in row] for row in grid],
                    rows=rows, columns=columns, cost=cost,
                    cycles=(four, excess), words=set(), d=width+1)

    def note(record, word):
        if not word:
            return
        assert syndrome(record["columns"], word) == 0
        record["d"] = min(record["d"], word.bit_count())
        record["words"].add(word)
        if len(record["words"]) > 64:
            record["words"] = set(sorted(
                record["words"], key=lambda w: (w.bit_count(), w))[:24])

    def parity_challenge(record, budget, turn):
        """Exact <=4 screen; bounded anchored completion above four."""
        columns = record["columns"]
        width = len(columns)
        singles, pairs = {}, {}
        incidence = [0]*len(record["rows"])
        free4 = True
        for c, value in enumerate(columns):
            word = 1 << c
            if not value:
                note(record, word)
                free4 = False
            old = singles.get(value)
            if old is not None:
                note(record, old ^ word)
                free4 = False
            else:
                singles[value] = word
            for r in bits(value):
                incidence[r] |= word

        for a, value in enumerate(columns):
            for b in range(a):
                combined = value ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(combined)
                if old is not None:
                    note(record, old ^ word)
                    free4 = False
                old = pairs.get(combined)
                if old is not None:
                    note(record, old ^ word)
                    free4 = False
                else:
                    pairs[combined] = word
        record["free4"] = free4
        if not free4 or not budget:
            return

        local = random.Random(seed+700001*turn)
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
                if not value:
                    note(record, word)
                    return
                left = min(10, record["d"]-1)-word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value, 0)
                if one and (one & allowed) == one:
                    note(record, word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and (pair & allowed) == pair:
                    note(record, word | pair)
                left = min(10, record["d"]-1)-word.bit_count()
                if left <= 2:
                    return
                choices, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        choices, smallest = available, count
                candidates = sorted(bits(choices), key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            # Each block's regular action permits a root representative.
            # Finite node budgets do not establish any exclusion above four.
            visit(columns[root], 1 << root,
                  all_variables & ~((1 << (root+1))-1))

    def cycle_challenge(record):
        """Genuine circuits of the degree-two subgraph."""
        adjacency = [[] for _ in record["rows"]]
        for c, value in enumerate(record["columns"]):
            if value.bit_count() == 2:
                a, b = list(bits(value))
                adjacency[a].append((b, c))
                adjacency[b].append((a, c))
        for root in range(0, len(adjacency), q):
            distances = [-1]*len(adjacency)
            paths = [0]*len(adjacency)
            parent = [-1]*len(adjacency)
            distances[root] = 0
            pending = [root]
            for a in pending:
                for b, edge in adjacency[a]:
                    if distances[b] < 0:
                        distances[b] = distances[a]+1
                        paths[b] = paths[a] ^ (1 << edge)
                        parent[b] = edge
                        pending.append(b)
                    elif edge != parent[a] and edge != parent[b]:
                        note(record, paths[a] ^ paths[b] ^ (1 << edge))

    def circuit_challenge(record, trials, turn, side):
        """Independently sampled genuine kernel words; readings only decrease."""
        local = random.Random(seed+1000003*turn
                              +100003*int(side == "B"))
        columns = record["columns"]
        order = list(range(len(columns)))
        retained = sorted(record["words"], key=int.bit_count)[:20]
        for trial in range(trials):
            local.shuffle(order)
            if retained and trial % 3 == 1:
                active = list(bits(local.choice(retained[:8])))
                front = local.sample(active, max(1, len(active)//2))
                selected = set(front)
                order = front+[c for c in order if c not in selected]
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
                    note(record, word)
            light = sorted(set(basis+retained), key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= record["d"]+3:
                        note(record, candidate)
            moves = basis+retained
            local.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(record, word)
            retained = sorted(record["words"], key=int.bit_count)[:24]
            if record["d"] <= 4:
                break
        record["words"] = set(retained)

    def merit(record):
        multiplicity = sum(w.bit_count() == record["d"]
                           for w in record["words"])
        return (record["free4"], min(14, record["d"]),
                -multiplicity, -record["cost"])

    def pool(support, side):
        signatures, geometry = set(), []

        def remember(record):
            if record is None:
                return False
            signature = tuple(tuple(sorted(cell))
                              for row in record["grid"] for cell in row)
            if signature in signatures:
                return False
            signatures.add(signature)
            geometry.append(record)
            geometry.sort(key=lambda r: r["cost"])
            del geometry[16:]
            return True

        for _ in range(64):
            remember(inspect(fresh(support, side), side))
        if not geometry:
            raise ValueError("empty balanced component geometry pool")

        positions = [(i,j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]
        for initial in geometry[:2]:
            current = initial
            for step in range(48):
                grid = [[list(c) for c in row] for row in current["grid"]]
                i, j = rng.choice(positions)
                grid[i][j] = fresh_cell(support[i][j], side, i, j)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                delta = proposed["cost"]-current["cost"]
                temperature = q*(8*(1-step/48.0)**2+0.2)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(current)

        selected = []
        for record in sorted(geometry, key=lambda r: r["cost"])[:8]:
            parity_challenge(record, 6000, 1)
            cycle_challenge(record)
            circuit_challenge(record, 8, 1, side)
            selected.append(record)
        selected.sort(key=merit, reverse=True)
        current = selected[0]

        # Replace actual occupied cells supporting current short witnesses.
        # Old words bind a proposal only after its new syndrome is checked.
        for step in range(16):
            grid = [[list(c) for c in row] for row in current["grid"]]
            word = min(current["words"], key=int.bit_count)
            occupied = {c//q for c in bits(word)}
            candidates = [p for p in positions if p[1] in occupied]
            i, j = rng.choice(candidates or positions)
            grid[i][j] = fresh_cell(support[i][j], side, i, j)
            proposed = inspect(grid, side)
            if not remember(proposed):
                continue
            for known in current["words"]:
                if syndrome(proposed["columns"], known) == 0:
                    note(proposed, known)
            parity_challenge(proposed, 4000, 10+step)
            cycle_challenge(proposed)
            circuit_challenge(proposed, 6, 10+step, side)
            selected.append(proposed)
            if merit(proposed) >= merit(current):
                current = proposed

        finalists = sorted(selected, key=merit, reverse=True)[:4]
        for record in finalists:
            parity_challenge(record, 30000, 101)
            circuit_challenge(record, 32, 101, side)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            parity_challenge(record, 60000, 211+cohort)
            circuit_challenge(record, 64, 211+cohort, side)
            record["rank"] = gf2_rank(record["rows"])
            assert record["words"]
            assert all(syndrome(record["columns"], w) == 0
                       for w in record["words"])
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
        """Actual sector-one logicals modulo ORIGINAL stabilizer spans."""
        reading = n+1
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            found = []
            for word in sorted(record["words"], key=int.bit_count)[:6]:
                for fixed in range(7 if kind == "Z" else 3):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((7*j+fixed)*q) for j in range(3))
                    else:
                        actual = word << (fixed*7*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        found.append(actual)
            for actual in sorted(set(found), key=int.bit_count)[:8]:
                for sweep in range(3):
                    changed = False
                    for row in stabilizers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                nearby = sorted(stabilizers,
                                key=lambda r: (r & actual).bit_count(),
                                reverse=True)[:24]
                winner = actual
                for i, first in enumerate(nearby):
                    for second in nearby[:i]:
                        candidate = actual ^ first ^ second
                        if candidate.bit_count() < winner.bit_count():
                            winner = candidate
                actual = winner
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
        return reading

    aa, bb = pool(sa, "A"), pool(sb, "B")
    proposals = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1449 and k >= 630
                    and 425*n <= 1000*k < 475*n):
                continue
            actual_a = [
                sum(((hx[r] >> (7*j*q)) & mask) << (j*q)
                    for j in range(3)) for r in range(q)]
            actual_b = [
                hz[alpha*q+r] & ((1 << (7*q))-1)
                for alpha in range(2) for r in range(q)]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logical = logical_challenge(a, b, hx, hz, n)
            # An absent nontrivial embedding earns no optimistic reading.
            detected = min(a["d"], b["d"],
                           logical if logical <= n else 6, 14)
            key = (a["free4"] and b["free4"],
                   k*detected*detected, detected,
                   -a["cost"]-b["cost"])
            proposals.append((key, A, B, hx, hz, k, a, b, logical))

    proposals.sort(key=lambda item: item[0], reverse=True)
    for key, A, B, hx, hz, k, a, b, logical in proposals:
        if not _bp_full_kernel4_free(hx, 1449):
            continue
        if not _bp_full_kernel4_free(hz, 1449):
            continue
        assert [[len(c) for c in row] for row in A] == sa
        assert [[len(c) for c in row] for row in B] == sb
        assert all(len(set(cell)) == len(cell)
                   for grid in (A,B) for row in grid for cell in row)
        assert max(row.bit_count() for row in hx+hz) == 12
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, k=k, card=(topology, model, cohort),
            quality=key, target_cell=0.450,
            component_readings=(a["d"], b["d"]),
            component_ranks=(a["rank"], b["rank"]),
            component_cycles=(a["cycles"], b["cycles"]),
            verified_localized_logical_reading=(
                logical if logical <= 1449 else None))
    raise ValueError("balanced cubic pairings missed rate/kernel-four gates")


def CELL(rng, G, side, i, j, ctx):
    """Return every selected cell of the new asymmetric shapes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
# REFUTED: C7:C12(s=3) odd-quotient backbones and kernel-five selection produced five readings >=6; all five measured codes instead returned weight four with abundant sector-one X and Z minima.
# REFUTED: Component kernel-through-five screening removes the abundant quantum weight-six family on Frobenius21 x C4 (1,4)x(1,4); all five measured current codes instead retained k=756 and many sector-one minimum logicals at weight six.
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
