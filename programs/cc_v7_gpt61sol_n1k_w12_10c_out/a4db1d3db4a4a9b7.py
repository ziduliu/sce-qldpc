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
CONCEPT = ("On frozen (C7 semidirect_2 C3) x C4, order 84, A=B=(1,4), binding trinomial relative-order cards, circuit-directed label repair and separately retained rank-deficient proposals should improve five-code fitness above 19.059 in cell 0.550. Non-improvement refutes this experiment; finite witness searches certify no distance.")
GROUP = ("Keep the exact Frobenius21 x C4 construction and order 84 byte-for-byte unchanged. This bed repeatedly produced measured distance six, while the current adjacent-central experiment produced no measured rank gain and many failed component gates. The degree-six Moore bound prevents demanding classical girth seven on 84 checks. Test different actual labels and support allocation using the original quantum quotient, not another cycle-count-only prescription.")
SHAPES = ("Keep frozen (1,4)x(1,4): n=1428, k>=756 and guaranteed rate 9/17=0.529412 target cell 0.550; q_max=88. Equal components each have 336 variables and 84 checks. Current minima were entirely sector one, especially Acol1/2 across Bcol0..2 and Acol0..2 across Bcol1/2. No listed minimum occupied Arow/Brow sector-two blocks. Full native matrix ranks decide every proposed pairing.")
FILLING = ("Twenty cards bind different pairs of intrinsic trinomial relative-order signatures rather than adjacent-central coordinate profiles. Compare 2223/2223 with legal asymmetric 2233/2222, independently oriented across cohorts. Authoritative monomial blocks supply every component syndrome. Retained short circuits penalize only proposals on which their syndromes still vanish; repair actual dense cells and binomial ratios. Keep separate distance and rank-deficit reservoirs, independently deepen their witnesses, then test sector-one embeddings and stabilizer reductions in the ORIGINAL quantum quotient. Final exhaustive logical-through-four screening permits short stabilizers but rejects short logicals. Actual k, verified logical upper readings, weights, component agreement and CSS remain binding; sampled multiplicities and readings certify no distance.")
HOW_AB = FILLING
FAMILY_NAME = "r055_Frob21xC4_14x14_binding_orders_circuit_repair_v5"

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


def _retired_central_geometry_ctx(rng, G, idx):
    """Inactive adjacent-central experiment; not called by NEW_CTX."""
    q = G.q
    card, cohort = idx % 20, idx // 20
    mask = (1 << q) - 1

    # Cache native operations only; every matrix still uses the verified lift.
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

    probe = Probe()

    central_bins = (
        (1, 1, 1), (2, 1, 1), (2, 1, 3),
        (0, 1, 1), (2, 2, 2),
    )
    # All use two adjacent central coordinates, one repeated.
    # This invariant differs from every parent trinomial recipe.
    central_tri = (
        (0, 0, 1), (0, 1, 1), (0, 0, 3), (0, 3, 3),
    )

    def columns_of(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            for v in _bp_bits(row):
                columns[v] |= 1 << r
        return columns

    def component(cells, side):
        indexed = [[G.idx(g) for g in cell] for cell in cells]
        if side == "A":
            rows, _, _ = lifted_matrices(probe, [indexed], [[[]]])
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], [indexed])
        assert len(rows) == q
        assert all(row >> (4*q) == 0 for row in rows)
        return rows, columns_of(rows, 4*q)

    def draw(model, mode, attempt):
        # Late attempts retain a feasible independent backbone, but never
        # restore the parent's trinomial central-coordinate model.
        if attempt >= 48:
            _, gs = _bp_backbone(rng, G, 0)
        else:
            zs = central_bins[model]
            gs = [(rng.randrange(1, 7), 0, zs[0]),
                  (rng.randrange(7), 1, zs[1]),
                  (rng.randrange(7), 2, zs[2])]
            signed = gs + [G.inv(g) for g in gs]
            if len({G.idx(g) for g in signed}) != 6:
                return None
        tri = []
        for b, z in zip(_BP_FRAMES[model], central_tri[mode]):
            available = [(a, b, z) for a in range(7)
                         if (a, b, z) not in tri]
            tri.append(rng.choice(available))
        return [[G.e, g] for g in gs] + [tri]

    def backbone_words(columns):
        """Actual five/six-word witnesses from verified binomial columns.

        Regular component symmetry moves any check vertex to check zero.
        Two simple paths of lengths 2+3 or 3+3 give the rooted words.
        Kernel-four exclusion precedes this call.
        """
        adjacency = [[] for _ in range(q)]
        for v, syndrome in enumerate(columns[:3*q]):
            checks = list(_bp_bits(syndrome))
            assert len(checks) == 2
            a, b = checks
            adjacency[a].append((b, v))
            adjacency[b].append((a, v))

        def paths(length):
            answer = {}
            todo = [(0, (0,), 0)]
            while todo:
                vertex, visited, word = todo.pop()
                if len(visited) == length + 1:
                    answer.setdefault(vertex, []).append(word)
                    continue
                for other, edge in adjacency[vertex]:
                    if other not in visited:
                        todo.append((other, visited + (other,),
                                     word ^ (1 << edge)))
            return answer

        p2, p3 = paths(2), paths(3)
        five, six = set(), set()
        for end, words in p3.items():
            for a in p2.get(end, ()):
                for b in words:
                    word = a ^ b
                    if word.bit_count() == 5:
                        five.add(word)
            for i, a in enumerate(words):
                for b in words[:i]:
                    word = a ^ b
                    if word.bit_count() == 6:
                        six.add(word)
        return five, six

    def rooted_five_words(columns, limit=64):
        """Capped genuine five-words; not an exhaustive quantum screen."""
        pairs = {}
        for i, a in enumerate(columns):
            for j in range(i):
                pairs[a ^ columns[j]] = (1 << i) | (1 << j)
        found = set()
        for root in range(0, 4*q, q):
            bit = 1 << root
            for i, a in enumerate(columns):
                if i == root:
                    continue
                prefix = columns[root] ^ a
                used = bit | (1 << i)
                for j in range(i):
                    if j == root:
                        continue
                    other = pairs.get(prefix ^ columns[j])
                    selected = used | (1 << j)
                    if other is not None and not (other & selected):
                        found.add(other | selected)
                        if len(found) >= limit:
                            return found
        return found

    def query(record, trials):
        """Search genuine kernel words with fresh elimination orders."""
        columns = record["columns"]
        order = list(range(len(columns)))
        retained = set(record["words"])
        for _ in range(trials):
            rng.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                syndrome, word = columns[v], 1 << v
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
            light = sorted(basis, key=int.bit_count)[:16]
            retained.update(light)
            for word in light[:8]:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        retained.add(candidate)
            retained = set(sorted(retained, key=int.bit_count)[:48])
        record["words"] = sorted(retained, key=int.bit_count)[:48]
        record["reading"] = min(map(int.bit_count, record["words"]))

    def select(side, model, mode):
        records, seen = [], set()
        for attempt in range(120):
            cells = draw(model, mode, attempt)
            if cells is None:
                continue
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in cells)
            if signature in seen:
                continue
            seen.add(signature)
            rows, columns = component(cells, side)
            floor, mixed5 = _bp_small_floor(columns, None)
            if floor < 5:
                continue
            five, six = backbone_words(columns)
            floor = 5 if five or mixed5 else 6
            rank = gf2_rank(rows)

            # Full-rank five-word components cannot improve the parent's
            # score at this rate. Rank-deficient exceptions are tested in
            # the actual quantum quotient, never presumed to be logical.
            if floor == 5 and rank == q:
                continue
            if floor == 5:
                five.update(rooted_five_words(columns))
            record = dict(
                cells=cells, columns=columns, rank=rank, floor=floor,
                five=len(five), six=len(six), four=_bp_cycles(columns),
                words=sorted(five | six, key=int.bit_count)[:48],
                signature=signature,
            )
            query(record, 6)
            records.append(record)
            if len(records) >= 8 and attempt >= 48:
                break
        if not records:
            raise ValueError("new central card produced no screened component")

        records.sort(key=lambda r: (
            r["reading"], q-r["rank"], -r["six"], -r["four"]),
            reverse=True)
        finalists = records[:3]
        # Preserve one independently generated rank-gain experiment.
        deficient = [r for r in records if r["rank"] < q]
        if deficient:
            extra = max(deficient, key=lambda r: (
                r["reading"], q-r["rank"], -r["six"]))
            if all(extra["signature"] != r["signature"] for r in finalists):
                finalists.append(extra)
        for record in finalists:
            query(record, 24)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = word
                    break
                word ^= previous
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                return True
            word ^= previous
        return False

    def syndrome(columns, word):
        answer = 0
        for v in _bp_bits(word):
            answer ^= columns[v]
        return answer

    def quantum_query(hx, hz, a, b, n):
        """Only verified nontrivial operators contribute upper readings."""
        xcolumns, zcolumns = columns_of(hx, n), columns_of(hz, n)
        xspan, zspan = span(hx), span(hz)
        found = set()
        reading = n + 1
        translations = [G.idx(G.e), rng.randrange(q), rng.randrange(q)]

        for side, record in (("A", a), ("B", b)):
            for word in record["words"][:24]:
                for t in translations:
                    moved = 0
                    for v in _bp_bits(word):
                        block, h = divmod(v, q)
                        # Native regular component actions, subsequently
                        # verified against the complete quantum syndrome.
                        h = table[h][t] if side == "A" else table[t][h]
                        moved |= 1 << (block*q + h)
                    for fixed in range(4):
                        if side == "A":
                            actual = sum(
                                ((moved >> (j*q)) & mask)
                                << ((4*j + fixed)*q) for j in range(4))
                            columns, pivots, kind = xcolumns, zspan, "Z"
                        else:
                            actual = moved << (fixed*4*q)
                            columns, pivots, kind = zcolumns, xspan, "X"
                        if (syndrome(columns, actual) == 0
                                and outside(actual, pivots)):
                            reading = min(reading, actual.bit_count())
                            found.add((kind, actual))
        return reading, len(found)

    amodel, mode = card // 4, card % 4
    bmodel = (2*amodel + mode + 1) % 5
    bmode = (mode + amodel + 1 + cohort) % 4
    aa = select("A", amodel, mode)
    bb = select("B", bmodel, bmode)

    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        2*q-ab[0]["rank"]-ab[1]["rank"],
        min(ab[0]["reading"], ab[1]["reading"]),
        -ab[0]["six"]-ab[1]["six"]), reverse=True)

    best = None
    for a, b in proposals:
        A, B = [a["cells"]], [b["cells"]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1428 and k >= 756 and 1000*k >= 525*n):
            continue
        dq, hits = quantum_query(hx, hz, a, b, n)
        # No witnessed logical is not evidence of arbitrarily high distance.
        # Six is a conservative selection default, not a certified bound.
        reading = min(dq, 6) if dq == n+1 else dq
        merit = (k*reading*reading/n, -hits,
                 -a["six"]-b["six"], -a["four"]-b["four"])
        if best is None or merit > best[0]:
            best = (merit, A, B, k, hx, hz, a, b, dq, hits)

    if best is None:
        raise ValueError("new witness-selected pair left cell 0.550")

    # Check finalists in merit order if a complete kernel-four gate fails.
    ordered = []
    for a, b in proposals:
        A, B = [a["cells"]], [b["cells"]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if k < 756:
            continue
        dq, hits = quantum_query(hx, hz, a, b, n)
        reading = 6 if dq == n+1 else dq
        merit = (k*reading*reading/n, -hits,
                 -a["six"]-b["six"], -a["four"]-b["four"])
        ordered.append((merit, A, B, k, hx, hz, a, b, dq, hits))
    ordered.sort(key=lambda item: item[0], reverse=True)

    for merit, A, B, k, hx, hz, a, b, dq, hits in ordered:
        if not _bp_full_kernel4_free(hx, 1428):
            continue
        if not _bp_full_kernel4_free(hz, 1428):
            continue
        sa = [[len(cell) for cell in A[0]]]
        sb = [[len(cell) for cell in B[0]]]
        assert sa == sb == [[2, 2, 2, 3]]
        assert _grid_weights(sa, sb) <= min(12, W_MAX)
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, k=k, card=(amodel, mode, bmodel, bmode),
            quality=merit, quantum_witness_reading=dq,
            quantum_witnesses=hits,
            component_ranks=(a["rank"], b["rank"]),
            component_floors=(a["floor"], b["floor"]),
            rooted_six_words=(a["six"], b["six"]),
        )
    raise ValueError("no new pair passed complete kernel-four screening")


def NEW_CTX(rng, G, idx):
    """Binding order cards, circuit repair, and native quantum-quotient tests."""
    q = G.q
    assert q == 84
    card, cohort = idx % 20, idx // 20
    width, mask = 4*q, (1 << q)-1
    seed = rng.randrange(1 << 62)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    orders = []
    for g in range(q):
        value = identity
        for order in range(1, q+1):
            value = table[value][g]
            if value == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid native element order")

    # Binary actions come exclusively from the authoritative constructor.
    xx, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, zz, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz]
              for g in range(q)],
    }
    del xx, zz

    def bits(word):
        return _bp_bits(word)

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def columns_of(rows, n):
        columns = [0]*n
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return columns

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

    def signature(cell):
        # Relative-order multiset is invariant under anchor choice,
        # common translation, conjugation and within-cell permutation.
        return tuple(sorted(
            orders[table[inverse[g]][h]]
            for at, g in enumerate(cell) for h in cell[:at]))

    def sidon(cell):
        if len(set(cell)) != len(cell):
            return False
        left = {table[inverse[g]][h]
                for g in cell for h in cell if g != h}
        right = {table[g][inverse[h]]
                 for g in cell for h in cell if g != h}
        return len(left) == len(right) == len(cell)*(len(cell)-1)

    # Enumerate construction recipes, not twenty random orbit images.
    # Individual cells are subsequently changed independently.
    triangles = {}
    for g in range(q):
        if g == identity:
            continue
        for h in range(g):
            if h == identity:
                continue
            cell = [identity, g, h]
            if not sidon(cell):
                continue
            sig = signature(cell)
            if min(sig) <= 2:
                continue
            triangles.setdefault(sig, []).append(cell)

    recipes = sorted(triangles, key=lambda s: (
        -min(s), -sum(s), -max(s), s))
    recipes = recipes[:8]
    plans = [(a, b) for i, a in enumerate(recipes)
             for b in recipes[i:]]
    # Different paired order prescriptions survive ordinary cell gauges.
    plans.sort(key=lambda p: (
        -(sum(p[0])+sum(p[1])),
        p[0] == p[1], p))
    assert len(plans) >= 20
    sig_a, sig_b = plans[card]

    def draw_triangle(wanted):
        cell = list(rng.choice(triangles[wanted]))
        # This translation is only initialization within a cell.
        # Diversity comes from independently selected relative templates
        # and subsequent independent changes, not from this gauge.
        root = rng.randrange(q)
        return [table[root][g] for g in cell]

    eligible = [g for g in range(q) if orders[g] > 3]

    def fresh(counts, wanted, connected):
        if connected and counts == (2,2,2,3):
            _, gs = _bp_backbone(rng, G, 0)
            cells = [[identity, G.idx(g)] for g in gs]
            return cells + [draw_triangle(wanted)]
        cells, used = [], set()
        for count in counts:
            if count == 2:
                available = [g for g in eligible
                             if g not in used and inverse[g] not in used]
                g = rng.choice(available or eligible)
                used.add(g)
                cells.append([identity, g])
            else:
                cells.append(draw_triangle(wanted))
        return cells

    def component(cells, side):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j*q)
        return rows, columns_of(rows, width)

    def short_circuits(columns, cap=24):
        """Exact <=4 challenges and capped rooted weight-five challenges.

        A miss through five is exact only after the <=4 collision screen
        has passed. Positive collections are capped, not exact spectra.
        """
        singles, pairs, found = {}, {}, set()
        for c, value in enumerate(columns):
            word = 1 << c
            if not value:
                found.add(word)
            old = singles.get(value)
            if old is not None:
                found.add(word ^ old)
            else:
                singles[value] = word
        for a, value in enumerate(columns):
            for b in range(a):
                combined = value ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(combined)
                if old is not None:
                    found.add(word ^ old)
                old = pairs.get(combined)
                if old is not None:
                    found.add(word ^ old)
                else:
                    pairs[combined] = word
                if len(found) >= cap:
                    return found
        if found:
            return found

        # Regular symmetry is used only for the component, never assumed
        # for the full nonabelian quantum Tanner graph.
        for root in range(0, width, q):
            for a, value in enumerate(columns):
                if a == root:
                    continue
                used = (1 << root) | (1 << a)
                partial = columns[root] ^ value
                for b in range(a):
                    if b == root:
                        continue
                    old = pairs.get(partial ^ columns[b])
                    selected = used | (1 << b)
                    if old is not None and not old & selected:
                        found.add(old | selected)
                        if len(found) >= cap:
                            return found
        return found

    def graph_circuits(columns):
        """Verified short cycles of the actual degree-two subgraph."""
        adjacency = [[] for _ in range(q)]
        for c, value in enumerate(columns):
            if value.bit_count() == 2:
                a, b = list(bits(value))
                adjacency[a].append((b, c))
                adjacency[b].append((a, c))
        found = set()
        for root in (0, q//3, 2*q//3):
            path, parent, reached = [0]*q, [-1]*q, {root}
            pending = [root]
            for a in pending:
                for b, edge in adjacency[a]:
                    if b not in reached:
                        reached.add(b)
                        pending.append(b)
                        parent[b] = edge
                        path[b] = path[a] ^ (1 << edge)
                    elif edge != parent[a] and edge != parent[b]:
                        word = path[a] ^ path[b] ^ (1 << edge)
                        if word and word.bit_count() <= 8:
                            found.add(word)
        assert all(syndrome(columns, w) == 0 for w in found)
        return found

    def inspect(cells, side):
        rows, columns = component(cells, side)
        words = short_circuits(columns) | graph_circuits(columns)
        assert all(w and syndrome(columns, w) == 0 for w in words)
        return dict(
            cells=[list(c) for c in cells], rows=rows, columns=columns,
            rank=gf2_rank(rows), cycles=_bp_cycles(columns),
            words=set(sorted(words, key=lambda w: (w.bit_count(), w))[:32]),
            reading=min(map(int.bit_count, words), default=width+1),
            sig=tuple(tuple(sorted(c)) for c in cells))

    def probe(record, trials, offset):
        """Accumulate genuine circuits using independent information sets."""
        local = random.Random(seed+offset)
        columns = record["columns"]
        retained = set(record["words"])
        order = list(range(width))
        for trial in range(trials):
            local.shuffle(order)
            if retained and trial % 3 == 1:
                active = list(bits(min(retained, key=int.bit_count)))
                front = local.sample(active, max(1, len(active)//2))
                chosen = set(front)
                order = front + [c for c in order if c not in chosen]
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
            light = sorted(set(basis) | retained, key=int.bit_count)[:16]
            additions = set(light)
            for word in light:
                for other in basis:
                    proposed = word ^ other
                    if proposed and proposed.bit_count() <= 10:
                        additions.add(proposed)
            moves = list(retained) + basis
            local.shuffle(moves)
            for initial in light[:4]:
                word = initial
                for _ in range(2):
                    changed = False
                    for move in moves:
                        proposed = word ^ move
                        if proposed and proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    if not changed:
                        break
                additions.add(word)
            retained = set(sorted(
                retained | additions, key=lambda w: (w.bit_count(), w))[:32])
        assert retained
        assert all(syndrome(columns, w) == 0 for w in retained)
        record["words"] = retained
        record["reading"] = min(
            record["reading"], min(map(int.bit_count, retained)))

    def pool(side, counts, wanted, channel):
        known, records, seen = set(), [], set()
        side_offset = 100003*int(side == "B") + 300007*channel

        def merit(record):
            # A witness from another filling penalizes this filling ONLY
            # when its authoritative syndrome still vanishes.
            hits = [w for w in known
                    if syndrome(record["columns"], w) == 0]
            reading = min(
                [record["reading"]] + [w.bit_count() for w in hits])
            penalty = sum(max(0, 7-w.bit_count())**2 for w in hits)
            return (min(8, reading), -penalty,
                    -record["cycles"], q-record["rank"])

        def retain(items):
            distance = sorted(items, key=merit, reverse=True)[:5]
            deficit = sorted(
                (r for r in items if r["rank"] < q),
                key=lambda r: (q-r["rank"], merit(r)),
                reverse=True)[:3]
            unique = {}
            for r in distance + deficit:
                unique[r["sig"]] = r
            return list(unique.values())

        for attempt in range(48):
            cells = fresh(counts, wanted, attempt % 3 != 0)
            key = tuple(tuple(sorted(c)) for c in cells)
            if key in seen:
                continue
            seen.add(key)
            record = inspect(cells, side)
            records.append(record)
            records = retain(records)
        if not records:
            raise ValueError("empty binding-order component bank")

        current = max(records, key=merit)
        probe(current, 8, side_offset+17)
        known.update(current["words"])

        for step in range(24):
            cells = [list(c) for c in current["cells"]]
            active = [w for w in known
                      if syndrome(current["columns"], w) == 0]
            occupied = sorted({
                c//q for w in active[:8] for c in bits(w)})
            j = rng.choice(occupied or list(range(4)))
            if counts[j] == 3:
                cells[j] = draw_triangle(wanted)
            else:
                available = [g for g in eligible
                             if g not in cells[j]]
                cells[j] = [identity, rng.choice(available)]
            key = tuple(tuple(sorted(c)) for c in cells)
            if key in seen:
                continue
            seen.add(key)
            proposed = inspect(cells, side)
            probe(proposed, 4, side_offset+700001+7919*step)
            known.update(proposed["words"])
            known = set(sorted(
                known, key=lambda w: (w.bit_count(), w))[:96])
            records = retain(records + [proposed])
            if merit(proposed) >= merit(current):
                current = proposed

        finalists = retain(records)
        distance = sorted(finalists, key=merit, reverse=True)[:2]
        deficient = [r for r in finalists if r["rank"] < q]
        if deficient:
            extra = max(deficient, key=lambda r: (
                q-r["rank"], merit(r)))
            if all(extra["sig"] != r["sig"] for r in distance):
                distance.append(extra)
        for record in distance:
            probe(record, 32, side_offset+1700021)
            known.update(record["words"])
        return sorted(distance, key=merit, reverse=True)

    def quantum_reading(hx, hz, a, b, n):
        """Only commuting words outside ORIGINAL stabilizer spans count."""
        reading, found = n+1, set()
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            columns = columns_of(opposite, n)
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(record["words"], key=int.bit_count)[:16]:
                for fixed in range(4):
                    actual = (
                        sum(((word >> (j*q)) & mask)
                            << ((4*j+fixed)*q) for j in range(4))
                        if kind == "Z" else word << (fixed*4*q))
                    assert syndrome(columns, actual) == 0
                    if outside(actual, pivots):
                        candidates.add(actual)
            for actual in sorted(candidates, key=int.bit_count)[:12]:
                for _ in range(2):
                    changed = False
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                nearby = sorted(
                    stabilizers, key=lambda r: (r & actual).bit_count(),
                    reverse=True)[:24]
                winner = actual
                for i, first in enumerate(nearby):
                    for second in nearby[:i]:
                        proposed = actual ^ first ^ second
                        if proposed.bit_count() < winner.bit_count():
                            winner = proposed
                actual = winner
                assert actual and syndrome(columns, actual) == 0
                assert outside(actual, pivots)
                reading = min(reading, actual.bit_count())
                found.add((kind, actual))
        return reading, len(found)

    def logical4_free(opposite, stabilizers, n):
        """Exhaustively reject logical weights <=4, allowing stabilizers."""
        columns = columns_of(opposite, n)
        pivots = span(stabilizers)
        singles, pairs = {0: 0}, {}
        for c, value in enumerate(columns):
            word = 1 << c
            old = singles.get(value)
            if old is not None:
                if outside(word ^ old, pivots):
                    return False
            else:
                singles[value] = word
        for a, value in enumerate(columns):
            for b in range(a):
                combined = value ^ columns[b]
                word = (1 << a) | (1 << b)
                old = singles.get(combined)
                if old is not None and outside(word ^ old, pivots):
                    return False
                old = pairs.get(combined)
                if old is not None:
                    if outside(word ^ old, pivots):
                        return False
                else:
                    pairs[combined] = word
        return True

    profiles = [
        ((2,2,2,3), (2,2,2,3)),
        ((2,2,3,3), (2,2,2,2)),
    ]
    if cohort % 2:
        profiles[1] = tuple(reversed(profiles[1]))

    proposals, cache = [], {}
    for channel, (ca, cb) in enumerate(profiles):
        assert _grid_weights([list(ca)], [list(cb)]) <= 12
        pools = []
        for side, counts, wanted in (
                ("A", ca, sig_a), ("B", cb, sig_b)):
            key = side, counts, wanted
            if key not in cache:
                cache[key] = pool(side, counts, wanted, channel)
            pools.append(cache[key])
        pairs = [(a,b) for a in pools[0] for b in pools[1]]
        pairs.sort(key=lambda ab: (
            min(ab[0]["reading"], ab[1]["reading"]),
            2*q-ab[0]["rank"]-ab[1]["rank"],
            -ab[0]["cycles"]-ab[1]["cycles"]), reverse=True)
        tested = pairs[:3]
        deficient = [ab for ab in pairs
                     if ab[0]["rank"] < q or ab[1]["rank"] < q]
        if deficient:
            extra = max(deficient, key=lambda ab: (
                2*q-ab[0]["rank"]-ab[1]["rank"],
                min(ab[0]["reading"], ab[1]["reading"])))
            if all((extra[0]["sig"], extra[1]["sig"]) !=
                   (a["sig"], b["sig"]) for a,b in tested):
                tested.append(extra)

        for a, b in tested:
            A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1428 and k >= 756 and 40*k > 21*n):
                continue
            dq, hits = quantum_reading(hx, hz, a, b, n)
            # Unknown distance earns no optimistic selection advantage.
            detected = min(8, dq) if dq <= n else 6
            merit = (k*detected*detected, detected,
                     -hits, -a["cycles"]-b["cycles"])
            proposals.append(
                (merit, A, B, hx, hz, k, a, b, channel, dq))

    proposals.sort(key=lambda item: item[0], reverse=True)
    for merit, A, B, hx, hz, k, a, b, channel, dq in proposals:
        if not logical4_free(hx, hz, 1428):
            continue
        if not logical4_free(hz, hx, 1428):
            continue
        sa = [[len(c) for c in A[0]]]
        sb = [[len(c) for c in B[0]]]
        assert _grid_weights(sa, sb) <= min(12, W_MAX)
        assert _no_dead_lines(sa, sb)
        assert all(len(set(c)) == len(c)
                   for grid in (A,B) for row in grid for c in row)
        actual_a = [
            sum(((hx[r] >> (4*j*q)) & mask) << (j*q)
                for j in range(4)) for r in range(q)]
        actual_b = [r & ((1 << width)-1) for r in hz[:q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert max(r.bit_count() for r in hx+hz) <= 12
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, k=k, quality=merit, card=(sig_a, sig_b, cohort),
            support_channel=channel,
            component_ranks=(a["rank"], b["rank"]),
            component_readings=(a["reading"], b["reading"]),
            quantum_witness_reading=None if dq > 1428 else dq,
            screen="logical <=4 excluded modulo original stabilizer spans")
    raise ValueError("binding-order pairings failed native logical gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete binding-order, circuit-repaired selected cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
# REFUTED: C7:C12(s=3) odd-quotient backbones and kernel-five selection produced five readings >=6; all five measured codes instead returned weight four with abundant sector-one X and Z minima.
# REFUTED: Kernel-through-five exclusion alone demonstrates distance above six on Frobenius21 x C4 (1,4)x(1,4); all five measured parent codes instead had abundant weight-six sector-one logicals and k=756.
# REFUTED: Adjacent-central trinomials and mixed-central backbones improve fitness above 19.059 or produce measured rank gains; all five measured current codes remained k=756 and distance six, with many other slots failing the component labeling gate.
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
