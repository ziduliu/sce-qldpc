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
CONCEPT = ("On C16 semidirect_3 C4, order 64, A=(1,3), B=(2,7), independently repaired cubic component kernels should yield five new distance readings >=10 and improve the 0.450 frontier above 28.6. Failure of this five-code target refutes the experiment; all search readings remain provisional upper bounds.")
GROUP = ("Jump from order-84 Frobenius21 x C4 to the order-64 split metacyclic C16:C4(s=3). The faithful action has order four since 3^2=9 and 3^4=1 mod 16. Different orders establish nonisomorphism. The supplied order-64 beds do not identify this faithful-action group; it is an exploration, not evidence of distance transfer. q=64 gives n=1472.")
SHAPES = ("Change to A=(1,3), B=(2,7): factor 23, n=1472 and k>=640 give guaranteed rate 10/23=0.434783 in cell 0.450; q_max=65. A has 192 variables and 64 checks, B has 448 variables and 128 checks. Parent weight-six minima occupied sector-one Acol1/2 and Bcol0/1/2 strips. Replace its equal binomial-heavy components with cubic A and degree-two/three B; verify actual full ranks and the 0.425..0.475 rate gate.")
FILLING = ("A counts (3,3,3); five inequivalent B weighted supports have row totals nine and column totals (2,2,2,3,3,3,3), giving HX/HZ maxima twelve. Four quotient occupancy prescriptions distinguish twenty cards. Cache only scaffold-built monomial blocks. Exact kernel-through-four screens precede accumulated genuine kernel witnesses, budgeted parity completion, witness-directed cell replacement and independent finalist retesting. Actual localized logicals are tested and reduced modulo original stabilizers before pairing. Final native components, full ranks, CSS, row weights and full kernel-four exclusion are verified. Search exhaustion is never a distance certificate.")
HOW_AB = FILLING
FAMILY_NAME = "r045_C16sC4_s3_13x27_cubic_witness_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 64

def custom_ops(q):
    # x^16=y^4=1, y*x*y^-1=x^3.
    # The automorphism 3 has order four modulo sixteen.
    if not valid_q(q):
        raise ValueError("C16:C4(s=3) requires order 64")
    action = (1, 3, 9, 11)
    els = [(a, b) for b in range(4) for a in range(16)]

    def red(g):
        return (int(g[0]) % 16, int(g[1]) % 4)

    def mul(g, h):
        return ((g[0] + action[g[1] % 4] * h[0]) % 16,
                (g[1] + h[1]) % 4)

    def inv(g):
        b = (-g[1]) % 4
        return ((-action[b] * g[0]) % 16, b)

    def idx(g):
        a, b = red(g)
        return 16*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


def _retired_exact6_ctx(rng, G, idx):
    """Inactive order-84 spectrum experiment; not called by NEW_CTX."""
    from itertools import combinations

    q = G.q
    width = 4 * q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    profile, mode = card // 4, card % 4
    assert q == 84
    assert _grid_weights([[2, 2, 2, 3]], [[2, 2, 2, 3]]) == 12

    # Cache ONLY scaffold-built monomial blocks. Subsequent assembly is
    # XOR linearity and disjoint column concatenation, not another lift.
    hx, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, hz, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    blocks = {
        "A": [[(row >> (j*q)) & mask for row in hx]
              for j in range(q)],
        "B": [[(row >> (j*q)) & mask for row in hz]
              for j in range(q)],
    }
    del hx, hz

    def assembled(cells, side):
        rows = [0] * q
        for j, cell in enumerate(cells):
            part = [0] * q
            for g in cell:
                monomial = blocks[side][G.idx(g)]
                for r in range(q):
                    part[r] ^= monomial[r]
            for r in range(q):
                rows[r] |= part[r] << (j*q)
        return rows

    def columns_of(rows):
        columns = [0] * width
        for r, row in enumerate(rows):
            for v in _bp_bits(row):
                columns[v] |= 1 << r
        return columns

    def kernel_floor(columns):
        """Exact through five, exploiting only regular component symmetry."""
        singles = set(columns)
        if 0 in singles:
            return 1, None
        if len(singles) != width:
            return 2, None
        pairs = set()
        for i, x in enumerate(columns):
            for j in range(i):
                value = x ^ columns[j]
                if value in singles:
                    return 3, None
                if value in pairs:
                    return 4, None
                pairs.add(value)

        # The regular action is transitive inside each component block.
        # Any five-word can move one selected variable to a block root,
        # then partition as a rooted triple and an unrestricted pair.
        for root in range(0, width, q):
            cr = columns[root]
            for i, x in enumerate(columns):
                if i == root:
                    continue
                value = cr ^ x
                for j in range(i):
                    if j != root and (value ^ columns[j]) in pairs:
                        return 5, pairs
        return 6, pairs

    def binomial_six(columns):
        """Exact binomial six-words via simple six-cycles through check 0."""
        adjacency = [[] for _ in range(q)]
        for v, signature in enumerate(columns[:3*q]):
            ends = list(_bp_bits(signature))
            assert len(ends) == 2
            a, b = ends
            adjacency[a].append((b, v))
            adjacency[b].append((a, v))
        assert all(len(ns) == 6 for ns in adjacency)

        paths = {}
        root = 0
        for a, ea in adjacency[root]:
            for b, eb in adjacency[a]:
                if b == root:
                    continue
                for c, ec in adjacency[b]:
                    if c in (root, a):
                        continue
                    internal = (1 << a) | (1 << b)
                    paths.setdefault(c, []).append(internal)

        anchored = 0
        for internal_masks in paths.values():
            for i, used in enumerate(internal_masks):
                anchored += sum(
                    not (used & other) for other in internal_masks[:i])
        # No short backbone cycles exist. Every binomial six-word is a
        # simple six-cycle; each has six check incidences.
        return anchored * q // 6

    def six_spectrum(columns):
        """Exact six-word counts after weights <=5 have been excluded.

        Odd column degrees imply an even number of trinomial variables.
        Regular symmetry permits one fixed trinomial root for the
        two-trinomial and four-trinomial cases.
        """
        doubles = columns[:3*q]
        triples = columns[3*q:]
        assert all(x.bit_count() == 2 for x in doubles)
        assert all(x.bit_count() == 3 for x in triples)

        bin_pairs = {}
        for i, x in enumerate(doubles):
            for j in range(i):
                syndrome = x ^ doubles[j]
                assert syndrome not in bin_pairs
                bin_pairs[syndrome] = (1 << i) | (1 << j)
        pair_values = tuple(bin_pairs)

        # Two trinomials plus four binomials. Each four-binomial set
        # has six ordered pair/pair partitions. Any overlap would give
        # a kernel word of weight <=4, already excluded.
        rooted_two = 0
        t0 = triples[0]
        for t in triples[1:]:
            target = t0 ^ t
            for value in pair_values:
                if (target ^ value) in bin_pairs:
                    rooted_two += 1
        assert rooted_two % 6 == 0
        n2 = (rooted_two // 6) * q // 2

        # Four trinomials plus two binomials: fix one trinomial root,
        # enumerate the other three, and look up the unique binomial pair.
        rooted_four = 0
        for a, b, c in combinations(triples[1:], 3):
            rooted_four += int((t0 ^ a ^ b ^ c) in bin_pairs)
        n4 = rooted_four * q // 4

        # Six variables inside the trinomial block. With kernels <=4
        # absent, distinct colliding triples must be disjoint.
        seen = {}
        collisions = 0
        for a, b, c in combinations(triples, 3):
            syndrome = a ^ b ^ c
            previous = seen.get(syndrome, 0)
            collisions += previous
            seen[syndrome] = previous + 1
        assert collisions % 10 == 0
        n6 = collisions // 10
        return binomial_six(columns), n2, n4, n6

    # All trinomials have central profile (0,0,1), absent from the
    # parent's explicit profiles. The C3 associations distinguish cards.
    quotient_profiles = ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))

    def fresh_tri(p):
        bg, bh = quotient_profiles[p]
        a = rng.randrange(1, 7) if bg == 0 else rng.randrange(7)
        b = rng.randrange(7)
        return [G.e, (a, bg, 0), (b, bh, 1)]

    def signature_of(cells):
        # Remove within-cell ordering, independent right-column gauges
        # and protograph-column permutations. This is not a claim to
        # recognize every full Tanner-graph isomorphism.
        normalized = []
        for cell in cells:
            normalized.append(min(
                tuple(sorted(G.idx(G.mul(g, G.inv(root)))
                             for g in cell))
                for root in cell))
        return tuple(sorted(normalized))

    def spectrum_key(record, objective):
        spectrum = record["six"]
        total = sum(spectrum)
        if objective == 0:
            main = (total, max(spectrum), spectrum[0])
        elif objective == 1:
            main = (max(spectrum), total, spectrum[1])
        elif objective == 2:
            main = (spectrum[0], spectrum[1] + spectrum[2],
                    total)
        else:
            main = (spectrum[3], spectrum[2], total)
        return main + (record["cycles"], -record["tri_rank"])

    def component_pool(side, p, objective):
        records, signatures = [], set()
        fallback = None
        for attempt in range(96):
            _, gs = _bp_backbone(rng, G, p)
            cells = [[G.e, g] for g in gs] + [fresh_tri(p)]
            signature = signature_of(cells)
            if signature in signatures:
                continue
            signatures.add(signature)
            rows = assembled(cells, side)
            columns = columns_of(rows)
            floor, pairs = kernel_floor(columns)
            tri_rows = [(row >> (3*q)) & mask for row in rows]
            record = {
                "cells": cells, "rows": rows, "columns": columns,
                "floor": floor, "cycles": _bp_cycles(columns),
                "tri_rank": gf2_rank(tri_rows),
                "signature": signature,
            }
            # A connected binomial backbone has rank q-1; the odd
            # trinomial supplies the missing augmentation direction.
            assert gf2_rank(rows) == q
            fallback_key = (floor, -record["cycles"],
                            record["tri_rank"])
            if fallback is None or fallback_key > fallback[0]:
                fallback = fallback_key, record
            if floor != 6:
                continue
            records.append(record)
            if len(records) >= 16:
                break

        if not records:
            # Preserve a genuinely new legal filling rather than copying
            # an archived recipe or silently claiming kernel-five success.
            if fallback is None:
                raise ValueError("no new mixed-parity component")
            record = fallback[1]
            record["six"] = None
            record["key"] = (record["floor"], 0,
                             -record["cycles"])
            return [record]

        # Cheap geometry ranks proposals before the exhaustive six count.
        # Different objectives retain different actual cell geometry.
        if objective == 3:
            records.sort(key=lambda r: (
                -r["tri_rank"], r["cycles"], r["signature"]))
        else:
            records.sort(key=lambda r: (
                r["cycles"], -r["tri_rank"], r["signature"]))
        finalists = records[:8]
        for record in finalists:
            record["six"] = six_spectrum(record["columns"])
            record["key"] = spectrum_key(record, objective)
        finalists.sort(key=lambda r: r["key"])
        return finalists[:3]

    bp = (profile + mode + 2 + cohort) % 5
    bm = (mode + 1 + cohort) % 4
    aa = component_pool("A", profile, mode)
    bb = component_pool("B", bp, bm)

    def pair_key(pair):
        a, b = pair
        floor = min(a["floor"], b["floor"])
        if a["six"] is None or b["six"] is None:
            return (-floor, 1, a["cycles"] + b["cycles"])
        totals = (sum(a["six"]), sum(b["six"]))
        # Avoid sacrificing one component to improve the other.
        return (-floor, 0, max(totals), sum(totals),
                max(max(a["six"]), max(b["six"])),
                a["key"], b["key"])

    proposals = sorted(((a, b) for a in aa for b in bb),
                       key=pair_key)
    for a, b in proposals:
        A, B = [a["cells"]], [b["cells"]]
        sa = [[len(cell) for cell in A[0]]]
        sb = [[len(cell) for cell in B[0]]]
        assert sa == sb == [[2, 2, 2, 3]]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1428 and k >= 756 and 1000*k >= 525*n):
            continue
        if not _bp_full_kernel4_free(hx, n):
            continue
        if not _bp_full_kernel4_free(hz, n):
            continue

        # Confirm cached screening against the final authoritative lift.
        ar = []
        for r in range(q):
            ar.append(sum(
                ((hx[r] >> (j*4*q)) & mask) << (j*q)
                for j in range(4)))
        br = [hz[r] & ((1 << width) - 1) for r in range(q)]
        assert ar == a["rows"] and br == b["rows"]
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {
            "A": A, "B": B, "k": k, "card": (profile, mode, bp, bm),
            "component_floors": (a["floor"], b["floor"]),
            "component_six_spectra": (a["six"], b["six"]),
            "trinomial_ranks": (a["tri_rank"], b["tri_rank"]),
            "quality": pair_key((a, b)),
        }
    raise ValueError("no new six-spectrum pairing remained in cell 0.550")


def NEW_CTX(rng, G, idx):
    """Twenty cubic/support cards on a genuinely different group."""
    q = G.q
    assert q == 64 and A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    card, cohort = idx % 20, idx // 20
    topology, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1

    # Columns are written as (upper count, lower count).
    # These differ in zero incidence and within-column multiplicities.
    profiles = (
        ((1,1),(1,1),(1,1),(2,1),(2,1),(1,2),(1,2)),
        ((2,0),(0,2),(1,1),(2,1),(2,1),(1,2),(1,2)),
        ((1,1),(1,1),(1,1),(3,0),(0,3),(2,1),(1,2)),
        ((2,0),(0,2),(1,1),(3,0),(0,3),(2,1),(1,2)),
        ((2,0),(0,2),(1,1),(3,0),(3,0),(0,3),(0,3)),
    )
    sa = [[3,3,3]]
    sb = [[column[i] for column in profiles[topology]]
          for i in range(2)]
    assert [sum(row) for row in sb] == [9,9]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Native operations are used for support geometry, not binary lifting.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    orders = []
    identity = G.idx(G.e)
    for g in range(q):
        value = identity
        for order in range(1, q+1):
            value = table[value][g]
            if value == identity:
                orders.append(order)
                break

    # Authoritative permutation blocks, extracted in modest batches.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        gallery = [[[g] for g in G.elements[start:start+16]]]
        xx, _, _ = lifted_matrices(G, gallery, [[[]]])
        _, zz, _ = lifted_matrices(G, [[[]]], gallery)
        for j in range(len(gallery[0])):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def columns_of(rows, width):
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        return columns

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def fresh_cell(count, i, j):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        best = None
        for attempt in range(24):
            phase = rng.randrange(4)
            if count == 2:
                step = (1, 0, rng.choice((1,2)), 1)[policy]
                quotients = (phase, (phase+step) % 4)
            elif policy == 0:
                quotients = (phase, (phase+1) % 4, (phase+2) % 4)
            elif policy == 1:
                quotients = (phase, phase, (phase+1) % 4)
            elif policy == 2:
                quotients = (phase, phase, (phase+2) % 4)
            else:
                step = 1 if (i+j+cohort) % 2 else 2
                quotients = (phase, (phase+1) % 4, (phase+step) % 4)
            cell = [(rng.randrange(16), b) for b in quotients]
            if len(set(cell)) != count:
                continue
            ids = [G.idx(g) for g in cell]
            left = {table[inverses[g]][h]
                    for g in ids for h in ids if g != h}
            right = {table[g][inverses[h]]
                     for g in ids for h in ids if g != h}
            low_order = sum(
                orders[table[inverses[g]][h]] <= 2
                for t, g in enumerate(ids) for h in ids[:t])
            key = (2*count*(count-1)-len(left)-len(right), low_order)
            if best is None or key < best[0]:
                best = key, cell
            if key == (0,0):
                break
        if best is None:
            return rng.sample(G.elements, count)
        return best[1]

    def fresh(support):
        return [[fresh_cell(count, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def component(grid, side):
        rows = []
        width = len(grid[0])*q
        for row in grid:
            lifted = [0]*q
            for j, cell in enumerate(row):
                for g in cell:
                    mono = blocks[side][G.idx(g)]
                    for r, value in enumerate(mono):
                        lifted[r] ^= value << (j*q)
            rows.extend(lifted)
        return rows, columns_of(rows, width)

    def geometry(columns, nchecks):
        uses, four = {}, 0
        adjacency = [0]*nchecks
        forced = 0
        for value in columns:
            neighbors = list(bits(value))
            degree = len(neighbors)
            forced += degree*(degree-1)*(degree-2)//6
            for t, a in enumerate(neighbors):
                adjacency[a] |= value ^ (1 << a)
                for b in neighbors[:t]:
                    key = (b,a)
                    four += uses.get(key, 0)
                    uses[key] = uses.get(key, 0)+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        return four, max(0, triangles-forced)

    def small_word(columns):
        """Exact detection through four; returned witnesses are genuine."""
        singles = {}
        for c, value in enumerate(columns):
            word = 1 << c
            if value == 0:
                return word, None
            previous = singles.get(value)
            if previous is not None:
                return word ^ previous, None
            singles[value] = word
        pairs = {}
        for a, value in enumerate(columns):
            for b in range(a):
                combined = value ^ columns[b]
                word = (1 << a) | (1 << b)
                previous = singles.get(combined)
                if previous is not None:
                    return word ^ previous, None
                previous = pairs.get(combined)
                if previous is not None:
                    return word ^ previous, None
                pairs[combined] = word
        return None, pairs

    def probe(record, trials, turn):
        """Accumulate real null words; fresh searches cannot erase readings."""
        local = random.Random(seed + 700001*turn
                              + 100003*int(record["side"] == "B"))
        columns = record["columns"]
        order = list(range(len(columns)))
        retained = set(record["words"])
        for trial in range(trials):
            local.shuffle(order)
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
                if value == 0:
                    basis.append(word)
            light = sorted(set(basis) | retained, key=int.bit_count)[:16]
            additions = list(light)
            for word in light[:10]:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        additions.append(candidate)
            moves = basis + list(retained)
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for sweep in range(2):
                    before = word.bit_count()
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    if word.bit_count() == before:
                        break
                additions.append(word)
            retained = set(sorted(
                retained | set(additions), key=lambda w: (w.bit_count(), w))[:24])
        assert retained
        assert all(syndrome(columns, word) == 0 for word in retained)
        record["words"] = retained
        record["d"] = min(record["d"], min(map(int.bit_count, retained)))

    def completion(record, budget):
        """Budgeted parity completion through nine; not an exclusion proof."""
        columns = record["columns"]
        witness, pairs = small_word(columns)
        if witness is not None:
            record["words"].add(witness)
            record["d"] = min(record["d"], witness.bit_count())
            return
        width = len(columns)
        singles = {value: 1 << c for c, value in enumerate(columns)}
        incidence = [0]*len(record["rows"])
        for c, value in enumerate(columns):
            for r in bits(value):
                incidence[r] |= 1 << c
        roots = list(range(0, width, q))
        rng.shuffle(roots)
        best = min(record["words"], key=int.bit_count)
        limit = min(9, best.bit_count()-1)
        if limit < 5:
            return
        all_variables = (1 << width)-1
        degree = max(value.bit_count() for value in columns)

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                assert syndrome(columns, word) == 0
                best = word

        for root in roots:
            remaining = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if value == 0:
                    observe(word)
                    return
                left = min(limit, best.bit_count()-1)-word.bit_count()
                if left <= 0 or value.bit_count() > degree*left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    observe(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    observe(word | pair)
                if left <= 2:
                    return
                chosen, smallest = 0, width+1
                for r in bits(value):
                    candidates = incidence[r] & allowed
                    size = candidates.bit_count()
                    if size == 0:
                        return
                    if size < smallest:
                        chosen, smallest = candidates, size
                choices = list(bits(chosen))
                choices.sort(key=lambda c: -(columns[c] & value).bit_count())
                live = allowed
                for c in choices:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            visit(columns[root], 1 << root, all_variables ^ (1 << root))
        record["words"].add(best)
        record["d"] = min(record["d"], best.bit_count())

    def inspect(grid, side):
        rows, columns = component(grid, side)
        if 0 in columns or len(set(columns)) != len(columns):
            return None
        rank = gf2_rank(rows)
        cycles = geometry(columns, len(rows))
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    rank=rank, cycles=cycles, side=side, words=set(),
                    d=len(columns)+1,
                    signature=tuple(tuple(sorted(G.idx(g) for g in cell))
                                    for row in grid for cell in row))

    def cost(record):
        four, excess = record["cycles"]
        return (four, excess) if policy % 2 == 0 else (
            32*four+excess, four)

    def merit(record):
        return (record["d"], tuple(-x for x in cost(record)),
                record["rank"])

    def pool(support, side):
        bank, seen = [], set()
        for attempt in range(64):
            record = inspect(fresh(support), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            bank.append(record)
            bank.sort(key=cost)
            del bank[16:]
        selected = []
        for record in bank:
            word, _ = small_word(record["columns"])
            if word is not None:
                continue
            probe(record, 8, 1)
            completion(record, 10000)
            selected.append(record)
            if len(selected) >= 6:
                break
        if not selected:
            raise ValueError("new cubic component failed kernel-four screen")

        current = max(selected, key=merit)
        for step in range(24):
            word = min(current["words"], key=int.bit_count)
            implicated = sorted({c//q for c in bits(word)})
            j = rng.choice(implicated)
            occupied = [i for i in range(len(support)) if support[i][j]]
            i = rng.choice(occupied)
            grid = copy_grid(current["grid"])
            grid[i][j] = fresh_cell(support[i][j], i, j)
            record = inspect(grid, side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            short, _ = small_word(record["columns"])
            if short is not None:
                continue
            probe(record, 6, 11+step)
            completion(record, 6000)
            selected.append(record)
            if merit(record) > merit(current):
                current = record

        finalists = sorted(selected, key=merit, reverse=True)[:4]
        for record in finalists:
            probe(record, 24, 101)
            completion(record, 24000)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            probe(record, 48+8*min(cohort, 2), 211)
            completion(record, 48000)
        return sorted(finalists, key=merit, reverse=True)

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

    def logical_challenge(a, b, hx, hz, n):
        readings = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(record["words"], key=int.bit_count)[:4]:
                for fixed in range(7 if side == "A" else 3):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((7*j+fixed)*q) for j in range(3))
                    else:
                        actual = word << (fixed*7*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                for sweep in range(2):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:20]
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < actual.bit_count():
                                actual = proposed
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool(sa, "A"), pool(sb, "B")
    proposals = []
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1472 and k >= 640 and 17*n <= 40*k < 19*n):
                continue
            logicals = logical_challenge(a, b, hx, hz, n)
            detected = min(12, a["d"], b["d"], *logicals)
            key = (k*detected*detected, detected,
                   -sum(a["cycles"])-sum(b["cycles"]))
            proposals.append((key, a, b, A, B, hx, hz, k, logicals))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for key, a, b, A, B, hx, hz, k, logicals in proposals:
        if not _bp_full_kernel4_free(hx, 1472):
            continue
        if not _bp_full_kernel4_free(hz, 1472):
            continue
        ar = [
            sum(((hx[r] >> (7*j*q)) & mask) << (j*q)
                for j in range(3)) for r in range(q)]
        br = [hz[alpha*q+r] & ((1 << (7*q))-1)
              for alpha in range(2) for r in range(q)]
        assert ar == a["rows"] and br == b["rows"]
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)
        assert all(len(cell) == len(set(cell)) <= 5
                   for grid in (A,B) for row in grid for cell in row)
        return dict(A=A, B=B, k=k, idx=idx,
                    card=(topology, policy, cohort),
                    support=(sa, sb), quality=key,
                    component_readings=(a["d"], b["d"]),
                    component_ranks=(a["rank"], b["rank"]),
                    component_cycles=(a["cycles"], b["cycles"]),
                    verified_localized_logical_readings=logicals)
    raise ValueError("new cubic pair missed full kernel/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stored cell; never manufacture a labeling orbit."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
# REFUTED: C7:C12(s=3) odd-quotient backbones and kernel-five selection produced five readings >=6; all five measured codes instead returned weight four with abundant sector-one X and Z minima.
# REFUTED: Exact six-spectrum mixed-parity selection on Frobenius21 x C4 (1,4)x(1,4) improves fitness above 19.059; the measured program remained at 19.059 with five k=756, distance-six codes and abundant sector-one minima.
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
