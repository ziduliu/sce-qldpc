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
CONCEPT = ("On C7 semidirect C12, action s=3, order 84, A=B=(1,4), three odd-quotient binomials and one trinomial should yield five new distance readings >=6 in cell 0.550. Connected bipartite backbones and verified component kernel-five screening test this hypothesis; no quantum-distance certificate is claimed.")
GROUP = ("Jump from cyclic Z11 to C7:C12(s=3). Since 3^12=1 mod 7, the semidirect product is valid; it contains order-12 elements. Nonabelian order-112 and order-291 beds motivate trying noncommuting lifts, but their stronger provisional readings do not establish this bed's distance. q=84 gives n=1428.")
SHAPES = ("A=B=(1,4): equal components have 336 variables and 84 checks. n=17*84=1428 and k>=9*84=756 give guaranteed rate 9/17=0.529412 in cell 0.550, whose current record is 13.235. Parent weight-five minima were entirely sector one, frequently involving Acol2..5 across B columns; replace the anchored component topology entirely.")
FILLING = ("Counts (2,2,2,3) on both sides give row sums nine and both CSS caps 9+3=12. Twenty cards combine five binomial quotient frames with four trinomial quotient patterns. Odd binomial voltages make the backbone bipartite. Independently select connected backbones against short relations, exclude verified component kernels through weight four, exhaustively test weight five using adjacent triples, and refine genuine randomized kernel witnesses. Verify final CSS, ranks and weights; component readings remain upper bounds, not quantum-distance guarantees.")
HOW_AB = FILLING
FAMILY_NAME = "r055_C7sC12_s3_14x14_bipartite_kernel5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # C7 semidirect C12: y*x*y^-1 = x^3.
    # The action has order six and therefore divides twelve.
    if not valid_q(q):
        raise ValueError("C7:C12 requires order 84")
    powers = tuple(pow(3, b, 7) for b in range(12))
    els = [(a, b) for b in range(12) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 12)

    def mul(g, h):
        return ((g[0] + powers[g[1]] * h[0]) % 7,
                (g[1] + h[1]) % 12)

    def inv(g):
        bi = (-g[1]) % 12
        return ((-powers[bi] * g[0]) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


# Quotient profiles are construction cards, not generated orbit copies.
_BP_FRAMES = ((1,1,5), (1,1,7), (1,5,5), (1,5,7), (1,7,7))
_BP_TRIANGLES = ((0,0,1), (0,1,2), (0,1,4), (0,1,6))


def _bp_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _bp_backbone(rng, G, frame):
    """Choose a connected Cayley backbone; score actual short relations."""
    best = None
    for trial in range(96):
        gs = [(rng.randrange(7), b) for b in frame]
        ns = gs + [G.inv(g) for g in gs]
        keys = {G._key(g) for g in ns}
        if len(keys) != 6:
            continue

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
        if len(reached) != G.q:
            continue

        products = {}
        bad = 0
        for g in ns:
            inverse = G._key(G.inv(g))
            for h in ns:
                if G._key(h) == inverse:
                    continue
                p = G._key(G.mul(g, h))
                bad += products.get(p, 0)
                products[p] = products.get(p, 0) + 1
        if best is None or bad < best[0]:
            best = (bad, gs)
        if bad == 0:
            break
    return best


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
    """Exact component kernel exclusion through weight five.

    With weights <=4 excluded, pair syndromes are unique. Every
    weight-five kernel has an adjacent pair, since each nonzero
    column's checks must be cancelled by other selected columns.
    Enumerating triples containing such a pair therefore suffices.
    """
    width = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != width:
        return 2
    pairs = {}
    bits = [1 << v for v in range(width)]
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs[syndrome] = bits[a] | bits[b]

    for a, neighbors in enumerate(adjacency):
        for b in neighbors:
            if b <= a:
                continue
            syndrome = columns[a] ^ columns[b]
            used = bits[a] | bits[b]
            for c, x in enumerate(columns):
                if c == a or c == b:
                    continue
                previous = pairs.get(syndrome ^ x)
                if previous is not None and not (
                        previous & (used | bits[c])):
                    return 5
    return 6


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
    """Optimize one deliberate quotient card using independent labels."""
    pool = []
    fallback = None
    pattern = _BP_TRIANGLES[mode]
    for attempt in range(72):
        backbone = _bp_backbone(rng, G, _BP_FRAMES[frame])
        if backbone is None:
            continue
        relation_cost, gs = backbone
        cells = [[G.e, g] for g in gs]
        tri = []
        for b in pattern:
            available = [(a, b) for a in range(7) if (a, b) not in tri]
            tri.append(rng.choice(available))
        cells.append(tri)
        rows, columns, adjacency = _bp_record(G, cells, side)
        if gf2_rank(rows) != G.q:
            continue
        floor = _bp_small_floor(columns, adjacency)
        cycles = _bp_cycles(columns)
        item = {"cells": cells, "columns": columns, "floor": floor,
                "cycles": cycles, "relations": relation_cost}
        fallback_key = (floor, -cycles, -relation_cost)
        if fallback is None or fallback_key > fallback[0]:
            fallback = (fallback_key, item)
        if floor < 5:
            continue
        reading = _bp_reading(rng, columns, 12)
        item["reading"] = reading
        item["key"] = (floor, reading, -cycles, -relation_cost)
        pool.append(item)
        if len(pool) >= 8:
            break

    if not pool:
        if fallback is None:
            raise ValueError("no connected full-rank component")
        item = fallback[1]
        item["reading"] = _bp_reading(rng, item["columns"], 24)
        item["key"] = (item["floor"], item["reading"],
                       -item["cycles"], -item["relations"])
        pool = [item]

    pool.sort(key=lambda item: item["key"], reverse=True)
    for item in pool[:3]:
        item["reading"] = min(
            item["reading"], _bp_reading(rng, item["columns"], 48))
        item["key"] = (item["floor"], item["reading"],
                       -item["cycles"], -item["relations"])
    pool.sort(key=lambda item: item["key"], reverse=True)
    return pool[:3]


def NEW_CTX(rng, G, idx):
    """Twenty quotient/backbone designs, never relabeling-produced slots."""
    card = idx % 20
    cohort = idx // 20
    af, am = card // 4, card % 4
    bf = (2*af + am + 1) % 5
    bm = (am + af + 1 + cohort) % 4
    apool = _bp_component_pool(rng, G, "A", af, am)
    bpool = _bp_component_pool(rng, G, "B", bf, bm)

    choices = []
    for a in apool:
        for b in bpool:
            key = (min(a["floor"], b["floor"]),
                   min(a["reading"], b["reading"]),
                   a["reading"] + b["reading"],
                   -a["cycles"] - b["cycles"])
            choices.append((key, a, b))
    choices.sort(key=lambda item: item[0], reverse=True)

    for quality, a, b in choices:
        A, B = [a["cells"]], [b["cells"]]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == sb == [[2, 2, 2, 3]]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1428 and k > 0 and 1000*k >= 525*n):
            continue
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {"A": A, "B": B, "k": k, "card": card,
                "quality": quality}
    raise ValueError("no screened pair remained in cell 0.550")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full binomial or trinomial cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The six-double/five-triple monomial profile with early-return kernel screening suppressed weight-five logicals on Z11 (3,11)x(3,11); all five measured codes instead returned weight five, entirely in sector one.
# REFUTED: The Z11 singleton-anchor and exact weight-five/six component selection produced five readings >=6; the measured current program instead returned five distance-five readings, all with abundant sector-one minima.
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
