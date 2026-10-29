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
CONCEPT = ("On (C7 semidirect_2 C3) x C4, order 84, A=B=(1,4), coupled support/label repair with separately retained rank-deficient proposals should raise five-code fitness above 19.059 in cell 0.550. Actual original-span logical challenges test short-word absorption; finite searches certify no distance.")
GROUP = ("Frozen Frobenius21 x C4, q=84, byte-for-byte unchanged. This bed measured five distance-six winners, but all 55 distinct parent codes had k=756. A connected binomial backbone has component rank q-1 and an odd trinomial supplies the missing augmentation direction. Retain the feasible group while replacing independent component selection and its forced-full-rank initialization.")
SHAPES = ("Frozen A=B=(1,4): equal components have 336 variables and 84 checks. n=1428, k>=756 and guaranteed rate 9/17=0.529412 target cell 0.550; q_max=88. Parent listed minima were entirely sector one, especially Acol1/2 across Bcol0..2 and Acol0..2 across Bcol1/2. No listed minimum occupied Arow/Brow sector-two blocks. Complete authoritative ranks decide dimension gains.")
FILLING = ("Twenty cards cross five support preferences with four cross-side coupling rules. Compare legal pairs 2223/2223, 2233/2222, 2224/2222, 2222/2222 and the new 1233/1233 pair against independently fresh 2223 competitors. Cache only scaffold-built monomial blocks; verify complete assemblies against lifted_matrices. Preserve separate distance and rank-deficit reservoirs, repair implicated cells on both sides jointly, and challenge genuine component words and complete-check circuits modulo ORIGINAL stabilizer spans. Unknown readings default to six; witnessed readings are capped at eight for selection. Final exhaustive logical-through-four checks permit short stabilizers but reject short logicals. Independent finalist searches, actual k, distinct cells, weights, CSS and the high-rate gate remain binding; search exhaustion certifies no distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r055_Frob21xC4_14x14_coupled_supportrepair_originalquotient_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Direct product (C7 semidirect_2 C3) x C4.
    # (a,b,z) denotes x^a y^b times the central C4 element z.
    if not valid_q(q):
        raise ValueError("Frobenius21 x C4 requires order 84")
    action = (1, 2, 4)
    els = [(a, b, z) for z in range(4)
           for b in range(3) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 3, g[2] % 4)

    def mul(g, h):
        return ((g[0] + action[g[1]] * h[0]) % 7,
                (g[1] + h[1]) % 3, (g[2] + h[2]) % 4)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-action[bi] * g[0]) % 7, bi, (-g[2]) % 4)

    def idx(g):
        a, b, z = red(g)
        return 7 * (3*z + b) + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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
    """Coupled support repair; classical short words are quotient challenges."""
    import random as _random

    q = G.q
    assert q == 84 and A_SHAPE == B_SHAPE == (1, 4)
    card, cohort = idx % 20, idx // 20
    preference, coupling = card % 5, card // 5
    seed = rng.randrange(1 << 62) ^ (0x84C055E5 + 15485863*(idx+1))
    rng = _random.Random(seed ^ 0xC0A1B5)
    mask = (1 << q)-1
    width, n1, n = 4*q, 16*q, 17*q
    profiles = (
        ((2,2,2,3), (2,2,2,3)),
        ((2,2,3,3), (2,2,2,2)),
        ((2,2,2,4), (2,2,2,2)),
        ((2,2,2,2), (2,2,2,2)),
        ((1,2,3,3), (1,2,3,3)),
    )
    for ca, cb in profiles:
        assert _grid_weights([list(ca)], [list(cb)]) <= min(12, W_MAX)
        assert _no_dead_lines([list(ca)], [list(cb)])

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid native group")

    # These are authoritative monomial blocks, not another binary lift.
    xx, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, zz, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx] for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz] for g in range(q)],
    }
    del xx, zz
    component_cache = {}

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def columns_of(rows, length):
        columns = [0]*length
        for r, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << r
        return columns

    def syndrome(columns, word):
        result = 0
        for v in bits(word):
            result ^= columns[v]
        return result

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = word
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

    def small_challenge(columns):
        """Exact through four; a rooted five-word challenge after exclusion."""
        singles = {}
        for v, value in enumerate(columns):
            word = 1 << v
            if not value:
                return word
            if value in singles:
                return word ^ singles[value]
            singles[value] = word
        pairs = {}
        for a, value in enumerate(columns):
            for b in range(a):
                key = value ^ columns[b]
                word = (1 << a) | (1 << b)
                previous = singles.get(key)
                if previous is None:
                    previous = pairs.get(key)
                if previous is not None:
                    return word ^ previous
                pairs[key] = word

        # Component regular symmetry supplies one variable root per block.
        # Finding a word is binding; no quantum symmetry is assumed.
        for root in range(0, width, q):
            used_root = 1 << root
            for a, value in enumerate(columns):
                if a == root:
                    continue
                prefix = columns[root] ^ value
                used = used_root | (1 << a)
                for b in range(a):
                    if b == root:
                        continue
                    previous = pairs.get(prefix ^ columns[b])
                    chosen = used | (1 << b)
                    if previous is not None and not (previous & chosen):
                        return previous | chosen
        return None

    def graph_words(columns):
        """Real cycles of the verified degree-two subgraph."""
        adjacency = [[] for _ in range(q)]
        for v, value in enumerate(columns):
            if value.bit_count() == 2:
                a, b = list(bits(value))
                adjacency[a].append((b, v))
                adjacency[b].append((a, v))
        found = set()
        for root in (0, q//4, q//2, 3*q//4):
            paths, parents = {root: 0}, {root: -1}
            pending = [root]
            for a in pending:
                for b, edge in adjacency[a]:
                    if b not in paths:
                        paths[b] = paths[a] ^ (1 << edge)
                        parents[b] = edge
                        pending.append(b)
                    elif edge != parents[a] and edge != parents[b]:
                        word = paths[a] ^ paths[b] ^ (1 << edge)
                        if word and word.bit_count() <= 8:
                            assert syndrome(columns, word) == 0
                            found.add(word)
        return found

    def kernel_words(columns, trials, diagnostic, retained=()):
        """Genuine circuits; cancellation can combine independent trials."""
        order = list(range(len(columns)))
        kept = set(retained)
        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
                while value:
                    p = value.bit_length()-1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (value, word)
                        break
                    value ^= previous[0]
                    word ^= previous[1]
                if not value:
                    basis.append(word)
            light = sorted(set(basis) | kept,
                           key=lambda w: (w.bit_count(), w))[:16]
            additions = set(light)
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        additions.add(candidate)
            kept = set(sorted(
                kept | additions, key=lambda w: (w.bit_count(), w))[:24])
        assert all(syndrome(columns, word) == 0 for word in kept)
        return kept

    def component(cells, side):
        signature = tuple(tuple(sorted(cell)) for cell in cells)
        key = side, signature
        if key in component_cache:
            return component_cache[key]
        rows = [0]*q
        for j, cell in enumerate(signature):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j*q)
        columns = columns_of(rows, width)
        words = graph_words(columns)
        short = small_challenge(columns)
        if short is not None:
            words.add(short)
        local_seed = seed + 100003*int(side == "B")
        local_seed += sum((j+1)*(g+1)
                          for j, cell in enumerate(signature) for g in cell)
        words = kernel_words(columns, 3, _random.Random(local_seed), words)
        record = dict(cells=[list(c) for c in signature], rows=rows,
                      columns=columns, words=words, rank=gf2_rank(rows),
                      cycles=_bp_cycles(columns), signature=signature)
        component_cache[key] = record
        return record

    def complete(a, b):
        # Sector-two contributions are transposes of verified components.
        # Final selected matrices are compared with lifted_matrices in full.
        hx, hz = [], []
        for beta in range(4):
            for r in range(q):
                row = sum(
                    ((a["rows"][r] >> (j*q)) & mask)
                    << ((4*j+beta)*q) for j in range(4))
                hx.append(row | (b["columns"][beta*q+r] << n1))
        for j in range(4):
            for r in range(q):
                hz.append((b["rows"][r] << (j*4*q))
                          | (a["columns"][j*q+r] << n1))
        return hx, hz

    wanted_orders = (
        (7,12,28), (6,12,14), (7,14,28),
        (4,7,12,28), (6,7,12,14,28),
    )[preference]
    eligible = [g for g in range(q) if orders[g] in wanted_orders]

    def fresh_cell(count):
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            root = rng.randrange(q)
            return [root, table[root][rng.choice(eligible)]]
        best = None
        for attempt in range(48):
            cell = rng.sample(range(q), count)
            left = {table[inverse[g]][h]
                    for g in cell for h in cell if g != h}
            right = {table[g][inverse[h]]
                     for g in cell for h in cell if g != h}
            rels = [orders[table[inverse[g]][h]]
                    for at, g in enumerate(cell) for h in cell[:at]]
            key = (2*count*(count-1)-len(left)-len(right),
                   sum(20 if order <= 2 else
                       int(order not in wanted_orders) for order in rels))
            if best is None or key < best[0]:
                best = key, cell
        return best[1]

    def draw(counts, turn):
        if counts == (2,2,2,3) and turn % 3:
            _, gs = _bp_backbone(rng, G, preference)
            return [[ident, G.idx(g)] for g in gs] + [fresh_cell(3)]
        return [fresh_cell(count) for count in counts]

    def bind(A, B):
        """Cross-side constraints, restored after every coupled repair."""
        A, B = [list(c) for c in A], [list(c) for c in B]
        if coupling == 1:
            aa = [j for j, cell in enumerate(A) if len(cell) == 2]
            bb = [j for j, cell in enumerate(B) if len(cell) == 2]
            if aa and bb:
                B[bb[0]] = list(A[aa[-1]])
        elif coupling == 2:
            aa = [j for j, cell in enumerate(A) if len(cell) >= 3]
            bb = [j for j, cell in enumerate(B) if len(cell) >= 3]
            if aa and bb and len(A[aa[-1]]) == len(B[bb[-1]]):
                B[bb[-1]] = list(A[aa[-1]])
        elif coupling == 3:
            for j in range(4):
                if len(A[j]) != len(B[j]) or len(B[j]) < 3:
                    continue
                replacement = []
                for g in A[j]:
                    _, charge, phase = G.elements[g]
                    available = [G.idx((x, charge, phase))
                                 for x in range(7)
                                 if G.idx((x, charge, phase))
                                 not in replacement]
                    replacement.append(rng.choice(available))
                B[j] = replacement
        return A, B

    def assess(A, B, profile_id, deep=False):
        a, b = component(A, "A"), component(B, "B")
        hx, hz = complete(a, b)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= 756 and 1000*k >= 525*n):
            return None
        found, active = set(), {"A": set(), "B": set()}
        reading = n+1
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            columns = columns_of(opposite, n)
            pivots = span(stabilizers)
            words = record["words"]
            if deep:
                local = _random.Random(seed+1700021+100003*int(side == "B"))
                words = kernel_words(record["columns"], 32, local, words)
                record["words"].update(words)
            candidates = []
            for word in sorted(words, key=lambda w: (w.bit_count(), w))[:12]:
                # If root embeddings all vanish, challenge all regular
                # translations of a short component word before trusting it.
                multipliers = [ident]
                if word.bit_count() <= 5:
                    multipliers = list(range(q))
                detected = False
                for t in multipliers:
                    moved = sum(
                        1 << ((v//q)*q + (
                            table[v % q][t] if side == "A"
                            else table[t][v % q]))
                        for v in bits(word))
                    assert syndrome(record["columns"], moved) == 0
                    for fixed in range(4):
                        actual = (sum(
                            ((moved >> (j*q)) & mask)
                            << ((4*j+fixed)*q) for j in range(4))
                            if side == "A" else moved << (fixed*4*q))
                        assert syndrome(columns, actual) == 0
                        if outside(actual, pivots):
                            detected = True
                            reading = min(reading, actual.bit_count())
                            found.add((side, actual))
                            candidates.append(actual)
                            active[side].update(v//q for v in bits(word))
                    if detected and word.bit_count() <= 5:
                        break

            for actual in sorted(set(candidates), key=int.bit_count)[:3]:
                for sweep in range(2):
                    changed = False
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                assert actual and outside(actual, pivots)
                assert syndrome(columns, actual) == 0
                reading = min(reading, actual.bit_count())
                found.add((side, actual))

            if deep and reading >= 6:
                diagnostic = _random.Random(
                    seed+2300009+100003*int(side == "B"))
                full_words = kernel_words(columns, 8, diagnostic)
                for word in full_words:
                    if outside(word, pivots):
                        reading = min(reading, word.bit_count())
                        found.add((side, word))

        d = min(8, reading) if reading <= n else 6
        cycles = a["cycles"]+b["cycles"]
        preferred = int(profile_id == preference)
        secondary = (
            (-len(found), -cycles),
            (-cycles, -len(found)),
            (preferred, -len(found), -cycles),
            (-max(a["cycles"], b["cycles"]), -len(found)),
        )[coupling]
        merit = (k*d*d, d, k, preferred) + secondary
        return dict(A=a["cells"], B=b["cells"], a=a, b=b, hx=hx, hz=hz,
                    k=k, reading=reading, d=d, hits=len(found), merit=merit,
                    active=active, profile=profile_id)

    records, signatures = [], set()

    def remember(A, B, profile_id, deep=False):
        A, B = bind(A, B)
        signature = (tuple(tuple(sorted(c)) for c in A),
                     tuple(tuple(sorted(c)) for c in B))
        if signature in signatures and not deep:
            return None
        signatures.add(signature)
        record = assess(A, B, profile_id, deep)
        if record is not None:
            records.append(record)
        return record

    # Every mechanism receives quantum testing, including classical-short
    # rank-deficient proposals. They are not filtered by component distance.
    for profile_id, (ca, cb) in enumerate(profiles):
        if profile_id in (1,2) and (idx+cohort) % 2:
            ca, cb = cb, ca
        attempts = 32 if profile_id == 0 else 8
        for turn in range(attempts):
            remember(draw(ca, turn), draw(cb, turn+1), profile_id)

    if not records:
        raise ValueError("empty coupled quantum support portfolio")

    distance_leaders = sorted(records, key=lambda r: r["merit"],
                              reverse=True)[:3]
    deficit_leaders = sorted(
        (r for r in records if r["a"]["rank"] < q or r["b"]["rank"] < q),
        key=lambda r: (2*q-r["a"]["rank"]-r["b"]["rank"], r["merit"]),
        reverse=True)[:2]
    leaders, used = [], set()
    for record in distance_leaders+deficit_leaders:
        signature = (record["a"]["signature"], record["b"]["signature"])
        if signature not in used:
            used.add(signature)
            leaders.append(record)

    # Joint repair differs from drawing independent component finalists:
    # both sides and their coupling participate in the acceptance decision.
    for chain, initial in enumerate(leaders):
        current = initial
        for step in range(12):
            A, B = [list(c) for c in current["A"]], [
                list(c) for c in current["B"]]
            sides = ("A", "B") if step % 3 == 2 else (
                "A" if (step+chain) % 2 == 0 else "B",)
            for side in sides:
                cells = A if side == "A" else B
                implicated = sorted(current["active"][side])
                j = rng.choice(implicated if implicated else range(4))
                cells[j] = fresh_cell(len(cells[j]))
            proposed = remember(A, B, current["profile"])
            if proposed is None:
                continue
            if (proposed["merit"] >= current["merit"]
                    or proposed["d"] == current["d"]
                    and proposed["k"] >= current["k"]
                    and rng.random() < 0.12):
                current = proposed

    finalists = sorted(records, key=lambda r: r["merit"], reverse=True)[:4]
    confirmed = []
    for record in finalists:
        checked = assess(record["A"], record["B"], record["profile"], True)
        if checked is not None:
            confirmed.append(checked)
    confirmed.sort(key=lambda r: r["merit"], reverse=True)

    def logical4_free(opposite, stabilizers):
        """Exhaustive logical exclusion through four in original row spans."""
        columns = columns_of(opposite, n)
        pivots = span(stabilizers)
        singles = {0: 0}
        for v, value in enumerate(columns):
            word = 1 << v
            previous = singles.get(value)
            if previous is not None:
                if outside(word ^ previous, pivots):
                    return False
            else:
                singles[value] = word
        pairs = {}
        for a, value in enumerate(columns):
            for b in range(a):
                key = value ^ columns[b]
                word = (1 << a) | (1 << b)
                previous = singles.get(key)
                if previous is not None and outside(word ^ previous, pivots):
                    return False
                previous = pairs.get(key)
                if previous is not None:
                    if outside(word ^ previous, pivots):
                        return False
                else:
                    pairs[key] = word
        return True

    for record in confirmed:
        hx, hz = record["hx"], record["hz"]
        if not logical4_free(hx, hz) or not logical4_free(hz, hx):
            continue
        A = [[[G.elements[g] for g in cell] for cell in record["A"]]]
        B = [[[G.elements[g] for g in cell] for cell in record["B"]]]
        sa, sb = [[len(c) for c in A[0]]], [[len(c) for c in B[0]]]
        assert _grid_weights(sa, sb) <= min(12, W_MAX)
        assert _no_dead_lines(sa, sb)
        assert all(1 <= len(cell) <= 5 and
                   len({G.idx(g) for g in cell}) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)
        native_hx, native_hz, native_n = lifted_matrices(G, A, B)
        assert native_n == n == 1428
        assert native_hx == hx and native_hz == hz
        k = n-gf2_rank(native_hx)-gf2_rank(native_hz)
        assert k == record["k"] and k >= 756
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(native_hx, native_hz)
        return dict(
            A=A, B=B, k=k, idx=idx, card=(preference, coupling, cohort),
            support=(sa, sb), selected_profile=record["profile"],
            quality=record["merit"],
            component_ranks=(record["a"]["rank"], record["b"]["rank"]),
            verified_logical_upper_reading=(
                record["reading"] if record["reading"] <= n else None),
            verified_logical_witnesses=record["hits"],
            screen="coupled repair; logical <=4 excluded in original spans")
    raise ValueError("no coupled finalist passed original-quotient gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete coupled, quotient-challenged cell."""
    cell = list(ctx[side][i][j])
    assert len(cell) <= 5 and len({G.idx(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
# REFUTED: C7:C12(s=3) odd-quotient backbones and kernel-five selection produced five readings >=6; all five measured codes instead returned weight four with abundant sector-one X and Z minima.
# REFUTED: Independent connected-backbone/trinomial selection supplies dimension gain or selected readings above six on Frobenius21 x C4 (1,4)x(1,4); all 55 distinct parent codes had k=756 and all five measured winners retained abundant weight-six sector-one logicals.
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
