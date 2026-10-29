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
CONCEPT = ("For metacyclic Z14:Z2(s=13), order 28, shapes (2,4)x(2,4), degree-three components with balanced extraction loads and suppressed short component kernels should improve worse-basis circuit reliability over the parent's syndrome-ambiguity screen.")
GROUP = ("Keep the metacyclic construction rule, resizing its parameters to m=14,r=2,s=13. The action satisfies 13^2=1 modulo 14; the new factor 20 uses q=28 for n=560. This exact group/shape bed has recorded fitness 7.1, supporting a competitive destination without claiming that noncommutativity alone improves pL.")
SHAPES = ("Target cell 0.200: equal (2,4) components have 56 checks and 112 variables each; n=560 and kb=112 give guaranteed rate 0.200. Require verified k=112. Parent weight-three witnesses occupied sector-1 Acol/Bcol blocks, especially Acol0-2, while Acol/Brow failure enrichment was only about 1.0-1.2x and did not identify a stable hotspot. Strengthen both component architectures rather than one block.")
FILLING = ("Twenty support cards cross five A profiles with four B profiles. Every component column has three terms, rows total 6/6 or 5/7, and both CSS check weights are <=10; six-round extraction has 24192 CNOTs. Same-coset cell supports exclude involutory internal differences. A verified-lift-derived component walk minimizes four-cycles and short kernels, then actual CSS ranks and stabilizer-aware screening reject nontrivial logicals through weight four. This targets the parent's 108-139 X failures per 250 shots and collective logical flips; these screening costs are hypotheses, not the circuit score.")
FAMILY_NAME = "r020_m14r2s13_24x24_col3_verified_kernelwalk"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires q=28")
    # Same metacyclic rule: x^14=y^2=1, y*x*y^-1=x^13.
    m, r, s = 14, 2, 13
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _columns(rows, width):
    result = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            result[bit.bit_length() - 1] |= tag
            row ^= bit
    return result


def _rows_from_columns(columns, height):
    rows = [0] * height
    for c, syndrome in enumerate(columns):
        tag = 1 << c
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length() - 1] |= tag
            syndrome ^= bit
    return rows


def _component_cycles(columns):
    """Four-cycles from shared check pairs; also return check adjacency."""
    pairs = {}
    adjacency = {}
    cycles = 0
    for syndrome in columns:
        nodes = []
        while syndrome:
            bit = syndrome & -syndrome
            nodes.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, u in enumerate(nodes):
            for v in nodes[:a]:
                pair = (v, u)
                old = pairs.get(pair, 0)
                cycles += old
                pairs[pair] = old + 1
                adjacency[u] = adjacency.get(u, 0) | (1 << v)
                adjacency[v] = adjacency.get(v, 0) | (1 << u)
    return cycles, adjacency


def _component_summary(columns, adjacency):
    """Exact weight-four kernels when columns are distinct and nonzero."""
    duplicates = len(columns) - len(set(columns)) + int(0 in columns)
    pairs = {}
    collisions = 0
    for a, x in enumerate(columns):
        for y in columns[:a]:
            syndrome = x ^ y
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1

    triangles = 0
    for u, neighbors in adjacency.items():
        higher = neighbors & ~((1 << (u + 1)) - 1)
        while higher:
            bit = higher & -higher
            v = bit.bit_length() - 1
            triangles += (
                neighbors & adjacency.get(v, 0)
                & ~((1 << (v + 1)) - 1)
            ).bit_count()
            higher ^= bit
    # Exact Tanner six-cycles when four-cycles are absent; otherwise
    # this is only a secondary adjacency statistic.
    six_cycles = max(0, triangles - len(columns))
    return duplicates, collisions // 3, six_cycles


def _six_kernel_count(columns):
    """Exact weight-six count, called only without weight-two/four words."""
    histogram = {}
    collisions = 0
    width = len(columns)
    for a in range(width - 2):
        x = columns[a]
        for b in range(a + 1, width - 1):
            ab = x ^ columns[b]
            for c in range(b + 1, width):
                syndrome = ab ^ columns[c]
                old = histogram.get(syndrome, 0)
                collisions += old
                histogram[syndrome] = old + 1
    # Colliding triples are disjoint under the precondition.
    return collisions // 10


def _small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    # Linear quotient representatives of unit vectors. Each pivot row
    # only uses lower-numbered coordinates, so ascending substitution
    # computes their full stabilizer-row-space remainders.
    residues = [1 << c for c in range(n)]
    for p in sorted(pivots):
        tail = pivots[p] ^ (1 << p)
        value = 0
        while tail:
            bit = tail & -tail
            value ^= residues[bit.bit_length() - 1]
            tail ^= bit
        residues[p] = value

    columns = _columns(checks, n)
    seen = {0: 0}
    for c, syndrome in enumerate(columns):
        value = residues[c]
        old = seen.get(syndrome)
        if old is not None and old != value:
            return True
        seen[syndrome] = value

    for a, x in enumerate(columns):
        ra = residues[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            value = ra ^ residues[b]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
    return False


def NEW_CTX(rng, G, idx):
    """Twenty support architectures, optimized using verified projections."""
    q = G.q
    profiles = (
        (2, 2, 1, 1),    # balanced, no zero cells
        (0, 2, 2, 2),    # balanced, one internal trinomial
        (3, 0, 2, 1),    # balanced, two internal trinomials
        (2, 1, 1, 1),    # row totals 5/7
        (0, 2, 2, 1),    # row totals 5/7 with one zero cell
    )
    design = idx % 20
    phase = (idx // 20) % 3
    pa, pb = profiles[design // 4], profiles[design % 4]
    counts = {
        "A": [list(pa), [3 - t for t in pa]],
        "B": [list(pb), [3 - t for t in pb]],
    }
    assert _grid_weights(counts["A"], counts["B"]) <= min(10, W_MAX)
    assert _no_dead_lines(counts["A"], counts["B"])

    # Obtain one-cell component contributions from the actual verified
    # matrices. The walk only XORs these extracted contributions and
    # shifts check-row indices; it never implements a group action.
    mask = (1 << q) - 1
    templates = {"A": [], "B": []}
    for g in G.elements:
        probe = [[[g] for _ in range(4)] for _ in range(2)]
        hx, hz, _ = lifted_matrices(G, probe, probe)
        templates["A"].append(
            _columns([row & mask for row in hx[:q]], q))
        templates["B"].append(
            _columns([row & mask for row in hz[:q]], q))

    cosets = [
        [g for g in G.elements if g[1] == parity]
        for parity in range(2)
    ]

    def draw_cell(weight):
        if not weight:
            return []
        for _ in range(128):
            cell = rng.sample(cosets[rng.randrange(2)], weight)
            differences = [
                (cell[a][0] - cell[b][0]) % 14
                for a in range(weight) for b in range(a)
            ]
            if 7 in differences:
                continue
            # Extra harness batches prescribe order-seven or order-
            # fourteen binomial ratios, not orbit copies of saved codes.
            if weight == 2 and phase == 1 and differences[0] % 2:
                continue
            if weight == 2 and phase == 2 and not differences[0] % 2:
                continue
            return cell
        raise ValueError("could not realize cell ratio policy")

    def component_columns(side, grid):
        result = [0] * (4 * q)
        for i, row in enumerate(grid):
            shift = i * q
            for j, cell in enumerate(row):
                base = j * q
                for g in cell:
                    contribution = templates[side][G.idx(g)]
                    for t, syndrome in enumerate(contribution):
                        result[base + t] ^= syndrome << shift
        assert all(s.bit_count() == 3 for s in result)
        return result

    def optimize(side):
        weights = counts[side]
        sites = [(i, j) for i in range(2) for j in range(4)
                 if weights[i][j]]

        def draw():
            return [[draw_cell(w) for w in row] for row in weights]

        current = draw()
        columns = component_columns(side, current)
        cost, adjacency = _component_cycles(columns)
        pool = []
        seen = set()

        for step in range(1200):
            if step % 8 == 0 and (cost == 0 or step % 80 == 0):
                signature = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in current for cell in row)
                if signature not in seen:
                    seen.add(signature)
                    summary = _component_summary(columns, adjacency)
                    rows = _rows_from_columns(columns, 2 * q)
                    if gf2_rank(rows) == 2 * q:
                        key = (cost,) + summary
                        pool.append((key, step, current, columns))
                        pool.sort(key=lambda item: (item[0], item[1]))
                        del pool[8:]

            i, j = rng.choice(sites)
            trial = [list(row) for row in current]
            trial[i][j] = draw_cell(weights[i][j])
            tc = component_columns(side, trial)
            trial_cost, trial_adjacency = _component_cycles(tc)
            if trial_cost <= cost or rng.random() < 0.008:
                current, columns = trial, tc
                cost, adjacency = trial_cost, trial_adjacency

            if step % 240 == 239 and cost:
                current = draw()
                columns = component_columns(side, current)
                cost, adjacency = _component_cycles(columns)

        if not pool:
            raise ValueError("component search found no full-row-rank filling")

        best = None
        for key, step, grid, columns in pool[:4]:
            cycles, duplicates, n4, six_cycles = key
            n6 = (_six_kernel_count(columns)
                  if not duplicates and not n4 else 10**9)
            score = (cycles, duplicates, n4, n6, six_cycles)
            if best is None or score < best[0]:
                best = (score, grid)
        return best

    # Independent component searches are followed by a joint test on
    # the actual quantum matrices, including mixed-sector logicals.
    for attempt in range(3):
        a_score, A = optimize("A")
        b_score, B = optimize("B")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 4 * q:
            continue
        if _small_logical(hx, hz, n) or _small_logical(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        return {"A": A, "B": B, "design": design, "phase": phase,
                "k": k, "component_screen": (a_score, b_score)}
    raise ValueError("joint target-rate short-logical screen failed")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Distinct rotational A-binomial steps and raw syndrome-ambiguity screening sufficiently protect the D14 (1,4)x(2,7) bed: fitness was 3.2219, measured slots retained abundant weight-three sector-1 logicals, and X failures were 108-139 per 250 shots.
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
