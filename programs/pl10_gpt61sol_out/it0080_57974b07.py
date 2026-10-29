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
CONCEPT = ("For cyclic C23, order 23, A=(2,4), B=(2,5), independently optimized degree-three/four components with fewer short syndrome ambiguities should improve worse-basis circuit fitness above the parent's 6.2461.")
GROUP = ("Jump from the nonabelian order-48 parent to prime cyclic C23. The recorded C23 (2,4)x(3,5) bed reached 6.9 and C29 (2,4)x(2,4) reached 7.2; these motivate this untried bed without proving a group advantage. Fix q=23, giving n=552.")
SHAPES = ("Target cell 0.250: (2,4)x(2,5) gives n=552, kb=138 and guaranteed rate 0.250. Components have 92/115 variables and 46 checks each. Parent witnesses occupied sector-one Acol/Bcol blocks with mostly diffuse enrichment; slot42's worse X basis included Bcol4 enrichment. Reinforce both components rather than assume one hotspot.")
FILLING = ("Twenty explicit support-pair architectures keep component columns at degree 3-4 and both CSS row-weight sums <=12. Independently optimize verified component projections for connectivity and short cycles, then screen exact kernels through weight six and small-boundary triples. Sampled kernel words are only a capped tie-breaker. Verify final rate, CSS and prefer no logical of weight <=4, retaining a rank-legal fallback. Prescribed six-round CNOT counts are 24012-26772 versus 27936-29952 in parent measured slots5/10/25/26. Reduced extraction exposure and ambiguity are pL hypotheses, not circuit-score estimates.")
FAMILY_NAME = "r025_C23_24x25_mintrap_twocomponent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C23 requires order 23")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def idx(g):
        return red(g)[0]

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _design_four_cycles(rows):
    """Count four-cycles within one basis, not across X and Z."""
    seen = {}
    cycles = 0
    for row in rows:
        cols = []
        while row:
            bit = row & -row
            cols.append(bit.bit_length() - 1)
            row ^= bit
        for a in range(len(cols)):
            for b in range(a):
                pair = (cols[b], cols[a])
                previous = seen.get(pair, 0)
                cycles += previous
                seen[pair] = previous + 1
    return cycles


def _design_small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1..4.

    Equal syndromes of supports of size <=2 exhaust this range.
    Reduce their difference against the opposite stabilizer row space.
    Inputs are exclusively the scaffold's verified lifted matrices.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def trivial(vector):
        while vector:
            p = vector.bit_length() - 1
            if p not in pivots:
                return False
            vector ^= pivots[p]
        return True

    syndromes = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] ^= syndrome_bit
            row ^= bit

    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(syndromes):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif not trivial(previous ^ bits[i]):
            return True

    for i in range(n):
        si, vi = syndromes[i], bits[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            vector = vi ^ bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = vector
            elif not trivial(previous ^ vector):
                return True
    return False


def _lp23_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _lp23_normalize(G, grid):
    """Remove cyclic column-translation gauge, not physical design changes."""
    out = _lp23_copy(grid)
    for j in range(len(out[0])):
        anchors = [i for i in range(2) if len(out[i][j]) == 1]
        i = anchors[0] if anchors else next(
            i for i in range(2) if out[i][j])
        shift = G.inv(out[i][j][0])
        for r in range(2):
            out[r][j] = sorted(G.mul(g, shift) for g in out[r][j])
    return out


def _lp23_component(G, grid, side):
    """Project the scaffold lift; the zero opposite side is only a probe."""
    zero = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _lp23_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _lp23_graph(rows, width):
    """Exact four-cycles; six-cycle heuristic, exact when four=0."""
    adjacency = [0] * len(rows)
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap * (overlap - 1) // 2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    seen = todo = 1
    while todo:
        bit = todo & -todo
        todo ^= bit
        more = adjacency[bit.bit_length() - 1] & ~seen
        seen |= more
        todo |= more
    disconnected = int(seen.bit_count() != len(rows))

    triangles = 0
    for i, neighbors in enumerate(adjacency):
        lower = neighbors & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (
                neighbors & adjacency[j] & ((1 << j) - 1)
            ).bit_count()
            lower ^= bit
    stars = 0
    for col in _lp23_columns(rows, width):
        degree = col.bit_count()
        stars += degree * (degree - 1) * (degree - 2) // 6
    return disconnected, four, max(0, triangles - stars)


def _lp23_short_words(cols):
    """Exact kernel exclusion through four, then counts at five/six.

    Once weights <=4 are absent, matching pair/triple supports are
    disjoint. Weight-five and weight-six words each have ten relevant
    unordered partitions. Triple boundaries are an uncalibrated
    decoder-ambiguity surrogate, not a circuit failure estimate.
    """
    if 0 in cols or len(set(cols)) != len(cols):
        return 1, 10**9, 10**9, 10**9
    singles = set(cols)
    pairs = set()
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return 1, 10**9, 10**9, 10**9
            pairs.add(syndrome)

    triples = {}
    five = six = boundary = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for k in range(j):
                syndrome = pair ^ cols[k]
                five += int(syndrome in pairs)
                old = triples.get(syndrome, 0)
                six += old
                triples[syndrome] = old + 1
                weight = syndrome.bit_count()
                if weight < 4:
                    boundary += (4 - weight) ** 2
    return 0, five // 10, six // 10, boundary


def _lp23_kernel_bound(rng, cols):
    """Sample actual kernel words; this supplies only an upper bound."""
    order = list(range(len(cols)))
    best = len(cols) + 1
    for _ in range(32):
        rng.shuffle(order)
        pivots = {}
        dependencies = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome and word:
                dependencies.append(word)
                best = min(best, word.bit_count())
        for i, word in enumerate(dependencies):
            for other in dependencies[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 6:
            break
    return best


def _lp23_pool(rng, G, supports, side):
    """Independent local search within one prescribed support architecture."""
    import math
    width = len(supports[0]) * G.q
    sites = [(i, j) for i in range(2)
             for j, size in enumerate(supports[i]) if size]
    retained = {}
    serial = 0

    def draw():
        return _lp23_normalize(G, [
            [rng.sample(G.elements, size) for size in row]
            for row in supports])

    def evaluate(grid):
        rows = _lp23_component(G, grid, side)
        metric = _lp23_graph(rows, width)
        energy = 10000 * metric[0] + 100 * metric[1] + metric[2]
        return rows, metric, energy

    def remember(grid, rows, metric):
        nonlocal serial
        signature = tuple(tuple(cell) for row in grid for cell in row)
        if signature in retained:
            return
        serial += 1
        retained[signature] = (metric, serial, _lp23_copy(grid), list(rows))
        if len(retained) > 8:
            worst = max(retained, key=lambda s: retained[s][:2])
            del retained[worst]

    for restart in range(5):
        current = draw()
        rows, metric, energy = evaluate(current)
        remember(current, rows, metric)
        for step in range(220):
            trial = _lp23_copy(current)
            i, j = rng.choice(sites)
            cell = trial[i][j]
            position = rng.randrange(len(cell))
            choices = [g for g in G.elements if g not in cell]
            cell[position] = rng.choice(choices)
            trial = _lp23_normalize(G, trial)
            new_rows, new_metric, new_energy = evaluate(trial)
            temperature = G.q * (6.0 * (0.985 ** step) + 0.10)
            delta = new_energy - energy
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                current, rows = trial, new_rows
                metric, energy = new_metric, new_energy
                remember(current, rows, metric)

    finalists = []
    for metric, serial, grid, rows in retained.values():
        cols = _lp23_columns(rows, width)
        bad, five, six, boundary = _lp23_short_words(cols)
        distance = _lp23_kernel_bound(rng, cols)
        deficit = len(rows) - gf2_rank(rows)
        entry = dict(grid=grid, graph=metric, bad=bad, five=five,
                     six=six, boundary=boundary / width,
                     distance=distance, deficit=deficit, serial=serial)
        key = (bad, metric[0], metric[1], entry["boundary"],
               five, six, -min(distance, 12), metric[2], deficit, serial)
        finalists.append((key, entry))
    finalists.sort(key=lambda item: item[0])
    return [entry for _, entry in finalists[:4]]


def NEW_CTX(rng, G, idx):
    """Twenty architecture briefs, independently optimized on both sides."""
    # A profiles differ in row loads and column-degree multisets.
    a_profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((1, 1, 1, 2), (2, 2, 2, 1)),
        ((1, 1, 2, 2), (2, 2, 1, 2)),
        ((1, 1, 2, 2), (2, 2, 2, 2)),
        ((1, 2, 2, 2), (2, 2, 2, 1)),
    )
    # B: baseline, 2/2 reinforcement, 1/3 reinforcement,
    # and one concentrated degree-three column.
    b_profiles = (
        ((1, 1, 1, 2, 2), (2, 2, 2, 1, 1)),
        ((1, 1, 2, 2, 2), (2, 2, 1, 1, 2)),
        ((1, 2, 2, 2, 1), (3, 1, 1, 1, 2)),
        ((0, 2, 2, 2, 2), (3, 1, 1, 1, 1)),
    )
    design = idx % 20
    sa, sb = a_profiles[design // 4], b_profiles[design % 4]
    assert _no_dead_lines(sa, sb)
    # BOTH sums: max A-row+B-column and max B-row+A-column.
    wx = max(map(sum, sa)) + max(sum(col) for col in zip(*sb))
    wz = max(map(sum, sb)) + max(sum(col) for col in zip(*sa))
    assert max(wx, wz) <= min(12, W_MAX)
    assert all(3 <= sum(col) <= 4 for col in zip(*sa))
    assert all(3 <= sum(col) <= 4 for col in zip(*sb))

    ap = _lp23_pool(rng, G, sa, "A")
    bp = _lp23_pool(rng, G, sb, "B")
    options = []
    for a in ap:
        for b in bp:
            ga, gb = a["graph"], b["graph"]
            da, db = min(a["distance"], 12), min(b["distance"], 12)
            key = (
                a["bad"] + b["bad"], ga[0] + gb[0],
                max(ga[1], gb[1]), ga[1] + gb[1],
                max(a["boundary"], b["boundary"]),
                a["boundary"] + b["boundary"],
                max(a["five"], b["five"]), a["five"] + b["five"],
                max(a["six"], b["six"]), a["six"] + b["six"],
                -min(da, db), abs(da - db),
                ga[2] + gb[2], a["deficit"] + b["deficit"],
                a["serial"], b["serial"],
            )
            options.append((key, a["grid"], b["grid"]))
    options.sort(key=lambda item: item[0])

    fallback = None
    for key, A, B in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 552 or not (225 * n <= 1000 * k < 275 * n):
            continue
        if not css_ok(hx, hz):
            continue
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        candidate = {"idx": idx, "design": design, "k": k,
                     "screen": key, "cells": {"A": A, "B": B}}
        if fallback is None:
            fallback = candidate
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        return candidate

    if fallback is None:
        raise ValueError("no verified C23 filling in the 0.250 rate cell")
    # A missed optional preference does not invalidate a hard-rule-legal code.
    return fallback


def CELL(rng, G, side, i, j, ctx):
    """Return distinct elements of the selected complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: A fixed d2s determines circuit fitness; current slots10/25/26/42 all had d2s=10 but worse-basis scores ranged from 5.748 to 6.356.
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
