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
CONCEPT = ("Cyclic Z28, order 28, A=B=(2,7): a connected sparse degree-two backbone plus four degree-three columns should yield five new distance readings above the parent's five, in rate cell 0.450.")
GROUP = ("Retain the cyclic specialization of the metacyclic construction, changing only m to 28 with r=s=1. Factor 53 gives n=1484 at the largest permitted order. This bed is absent from BEDS; component geometry motivates it, not an established distance claim.")
SHAPES = ("Equal (2,7)x(2,7) components have 196 variables and 56 checks. Guaranteed k>=28*25=700 gives rate 25/53=0.471698, targeting the effectively empty 0.450 cell rather than the occupied 0.500 cell. Parent weight-five minima were all sector 1, frequently touching A columns 2..5 across B columns.")
FILLING = ("Twenty cards combine five support profiles with four degree-two backbone geometries. Every component row has nine terms and every column two or three, so both quantum row caps are 9+3=12. Optimize cycles extracted from verified component lifts; exclude component kernels through weight five where possible, penalize detected weight-six words, and compare repeated elimination readings. Final verified CSS ranks gate [0.425,0.475). Neither component screens nor readings certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r045_m28r1s1_27x27_backbone_kernel5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Same metacyclic construction and cyclic specialization.
    # q=28 and size factor 53 give n=1484.
    m, r, s = 28, 1, 1
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
B_SHAPE = (2, 7)               # (rb, cb)


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
    """Extract the component exclusively through the verified helper."""
    zero = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, zero)
    else:
        _, rows, _ = lifted_matrices(G, zero, grid)
    width = 7 * G.q
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


def NEW_CTX(rng, G, idx):
    """Twenty support/backbone designs, not relabeled copies of one fill."""
    card = idx % 20
    pa, ma = card // 4, card % 4
    pb = (2 * pa + 1) % 5
    mb = (ma + pa + 1) % 4
    aa = _component_pool(rng, G, pa, ma, "A", idx)
    bb = _component_pool(rng, G, pb, mb, "B", idx + 7)
    pairs = []
    for a in aa:
        for b in bb:
            qa, qb = a["quality"], b["quality"]
            key = (min(qa[0], qb[0]), qa[0] + qb[0],
                   qa[1] + qb[1], min(qa[2], qb[2]),
                   qa[2] + qb[2], qa[3] + qb[3])
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
        if not (0.425 <= k / n < 0.475):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= 12
        return {"A": A, "B": B, "k": k,
                "card": card, "quality": key}
    raise ValueError("screened pairs missed the 0.450 rate window")


def CELL(rng, G, side, i, j, ctx):
    """Return the optimized full cell, including deliberate zero blocks."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled supports and uniform cyclic labels did not suppress weight-six sector-1 logicals: all five measured codes returned distance six.
# REFUTED: The Z11 (3,11)x(3,11) degree-2/3 monomial kernel-screening model improved on distance six; all five measured codes instead returned distance five, with thousands of sector-1 witnesses.
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
