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
CONCEPT = ("Hypothesis: PSL(2,7), order 168, shapes (1,2)x(1,3), with balanced component-kernel selection and no involutory within-cell ratios, yields five measured distance readings >=12 in cell 0.300. Search readings remain provisional upper bounds.")
GROUP = ("Jump from dihedral D298 to the non-solvable projective matrix group PSL(2,7). This order-168 group is absent from BEDS and is not a metacyclic parameter change. Its simple-group structure tests a different source of component connectivity without cyclic commuting-polynomial assumptions.")
SHAPES = ("A=(1,2), B=(1,3): n=7*168=1176 and k>=2*168=336 give guaranteed rate 2/7. Components have 336 and 504 variables, each with 168 checks. Parent measured minima were entirely sector 1 and used both A columns and multiple B columns; reinforce both component kernels. Exact full ranks enforce 0.275<=k/n<0.325.")
FILLING = ("Twenty cards cross five count allocations with four relative-order spectra. Both CSS row sums are <=12. Binomial ratios have order seven; larger cells exclude involutory ratios and vary their order-seven pair counts. Verified lifts supply component connectivity, four-cycle and projection-triangle statistics, followed by independent randomized kernel searches. Exact full ranks select the target band. Prefer full kernel-four exclusion, retaining a valid screened fallback if this stronger test fails; these heuristics do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r030_PSL2F7_12x13_order_spectrum_circuits_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 168

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("PSL(2,7) requires order 168")

    # SL(2,F7) modulo its central subgroup {I,-I}.
    # Elements encode a matrix (a,b;c,d), choosing min(M,-M).
    def red(g):
        v = tuple(int(x) % 7 for x in g)
        return min(v, tuple((-x) % 7 for x in v))

    els = []
    for a in range(7):
        for b in range(7):
            for c in range(7):
                for d in range(7):
                    g = (a, b, c, d)
                    if (a*d - b*c) % 7 == 1 and red(g) == g:
                        els.append(g)
    assert len(els) == q
    pos = {g: i for i, g in enumerate(els)}

    # Cache the exact quotient law once. Subsequent verified lifts
    # still use the scaffold, not an independently implemented lift.
    table = []
    for a, b, c, d in els:
        table.append([
            red((a*e + b*h, a*f + b*k,
                 c*e + d*h, c*f + d*k))
            for e, f, h, k in els
        ])
    inverses = {
        g: red((g[3], -g[1], -g[2], g[0])) for g in els
    }

    def mul(g, h):
        return table[pos[g]][pos[h]]

    def inv(g):
        return inverses[g]

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Five density allocations, crossed with four within-cell order spectra.
# Maximum HX/HZ weights: (9,12), (11,12), (10,12), (10,12), (9,11).
_PSL_COUNTS = (
    ((3, 3), (3, 3, 3)),
    ((4, 4), (2, 3, 3)),
    ((3, 4), (2, 3, 3)),
    ((3, 3), (2, 3, 4)),
    ((3, 3), (2, 3, 3)),
)


def _psl_bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def _psl_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _psl_bits(row):
            columns[c] |= 1 << r
    return columns


def _psl_no_kernel4(columns):
    """Exact exclusion of nonzero kernel vectors of weights one to four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = singles.copy()
    for i, a in enumerate(columns):
        for j in range(i):
            value = a ^ columns[j]
            if value in seen:
                return False
            seen.add(value)
    return True


def _psl_components(hx, hz, q):
    """Project components from the scaffold's actual LP column ordering."""
    mask = (1 << q) - 1
    ar = []
    for t in range(q):
        source = hx[t]  # Arow 0, Bcol 0.
        ar.append((source & mask)
                  | (((source >> (3*q)) & mask) << q))
    bmask = (1 << (3*q)) - 1
    br = [hz[t] & bmask for t in range(q)]  # Acol 0, Brow 0.
    return ar, br


def _psl_quality(rows, width):
    """Actual component C4 count and a projection-triangle heuristic."""
    columns = _psl_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    adjacency = [0] * len(rows)
    uses = {}
    four = forced = 0
    for signature in columns:
        ns = list(_psl_bits(signature))
        degree = len(ns)
        forced += degree * (degree - 1) * (degree - 2) // 6
        for i, u in enumerate(ns):
            adjacency[u] |= signature ^ (1 << u)
            for v in ns[:i]:
                key = v * len(rows) + u
                previous = uses.get(key, 0)
                four += previous
                uses[key] = previous + 1

    reached = frontier = 1
    while frontier:
        nxt = 0
        for u in _psl_bits(frontier):
            nxt |= adjacency[u]
        frontier = nxt & ~reached
        reached |= frontier
    if reached != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _psl_bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v]
                          & (-1 << (v + 1))).bit_count()
    return (four, max(0, triangles - forced)), columns


def _psl_circuit_reading(rng, columns, trials):
    """Find actual classical kernel words; readings are upper bounds only."""
    reading = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                lead = syndrome.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                reading = min(reading, word.bit_count())
        light = sorted(basis, key=int.bit_count)[:20]
        for a in light:
            for b in basis:
                if a != b:
                    reading = min(reading, (a ^ b).bit_count())
        if len(light) >= 3:
            for _ in range(80):
                a, b, c = rng.sample(light, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
        if reading <= 6:
            break
    return reading


def NEW_CTX(rng, G, idx):
    """Optimize one density/order card without submitting orbit copies."""
    ac, bc = _PSL_COUNTS[idx % 5]
    model = (idx // 5) % 4
    phase = idx // 20
    sa, sb = [list(ac)], [list(bc)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid projective group law")

    def planned_cell(count):
        # These spectra are conjugacy/automorphism invariant, unlike
        # arbitrary names assigned to particular group elements.
        best = None
        for _ in range(512):
            cell = rng.sample(G.elements, count)
            relative = [
                orders[G.mul(G.inv(cell[i]), cell[j])]
                for i in range(count) for j in range(i)
            ]
            if 2 in relative:
                continue
            sevens = relative.count(7)
            if count == 2:
                if sevens == 1:
                    return cell
                continue
            target = model if count == 3 else model + 1
            error = abs(sevens - target)
            if best is None or error < best[0]:
                best = (error, cell)
            if error == 0:
                return cell
        # A near-target spectrum remains legal; never require an
        # unverified group-geometric hypothesis merely to fill a slot.
        if best is not None:
            return best[1]
        raise ValueError("no non-involutory support sampled")

    pool = []
    for trial in range(64):
        A = [[planned_cell(c) for c in ac]]
        B = [[planned_cell(c) for c in bc]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 7 * G.q == 1176
        ar, br = _psl_components(hx, hz, G.q)
        qa = _psl_quality(ar, 2 * G.q)
        qb = _psl_quality(br, 3 * G.q)
        if qa is None or qb is None:
            continue
        ast, acol = qa
        bst, bcol = qb
        if phase % 3 == 0:
            cycle_key = (max(ast[0], bst[0]), ast[0] + bst[0],
                         max(ast[1], bst[1]), ast[1] + bst[1])
        elif phase % 3 == 1:
            cycle_key = (ast[0] + bst[0], max(ast[0], bst[0]),
                         ast[1] + bst[1], max(ast[1], bst[1]))
        else:
            # Additional harness slots deliberately emphasize the
            # higher-rate, longer B component rather than relabel A/B.
            cycle_key = (2*bst[0] + ast[0], bst[0],
                         2*bst[1] + ast[1], ast[1])
        pool.append((cycle_key, trial, A, B, hx, hz, n, acol, bcol))

    pool.sort(key=lambda item: item[:2])
    finalists = []
    for key, trial, A, B, hx, hz, n, acol, bcol in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (11*n <= 40*k < 13*n and k >= 2*G.q):
            continue
        small_free = (_psl_no_kernel4(acol)
                      and _psl_no_kernel4(bcol))
        da = _psl_circuit_reading(rng, acol, 8)
        db = _psl_circuit_reading(rng, bcol, 8)
        merit = (int(small_free), min(da, db), da + db,
                 tuple(-v for v in key), -(k - 2*G.q))
        finalists.append([merit, A, B, hx, hz, n, k, acol, bcol, da, db])
        if len(finalists) >= 8:
            break
    if not finalists:
        raise ValueError("no connected component pair in target rate band")

    finalists.sort(key=lambda item: item[0], reverse=True)
    # Independent extra orders lower sensitivity to a lucky first search.
    for item in finalists[:4]:
        item[9] = min(item[9], _psl_circuit_reading(rng, item[7], 32))
        item[10] = min(item[10], _psl_circuit_reading(rng, item[8], 32))
        old = item[0]
        item[0] = (old[0], min(item[9], item[10]),
                   item[9] + item[10], old[3], old[4])
    # Only independently refined candidates compete in the final choice.
    refined = sorted(finalists[:4], key=lambda item: item[0], reverse=True)
    fallback = refined[0]
    selected = None
    for item in refined:
        hx, hz, n = item[3:6]
        if (_psl_no_kernel4(_psl_columns(hx, n))
                and _psl_no_kernel4(_psl_columns(hz, n))):
            selected = item
            break
    if selected is None:
        selected = fallback

    merit, A, B, hx, hz, n, k = selected[:7]
    assert max(row.bit_count() for row in hx + hz) <= 12
    if not css_ok(hx, hz):
        raise ValueError("verified construction failed CSS")
    return {"idx": idx, "A": A, "B": B, "k": k,
            "card": (idx % 5, model, phase),
            "component_readings": (selected[9], selected[10]),
            "full_kernel4_free": selected is not None and
                                 selected is not fallback}


def CELL(rng, G, side, i, j, ctx):
    """Return the entire selected cell without subsequent randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the row-weight cap suffices to eliminate sub-12 sector-1 logicals; parent D298 slots 4 and 5 had w=12 but measured distance eight, with X minima entirely in sector 1.
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
