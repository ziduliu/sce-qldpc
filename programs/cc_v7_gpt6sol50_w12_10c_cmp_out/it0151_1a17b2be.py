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
CONCEPT = ("On cyclic C23 x C2 of order 46, shapes (2,3)x(2,3), distinct binomial rotation gaps and screened triple coset patterns should produce five d30s readings above 27.")
GROUP = ("Keep the order-46 direct product C23 x C2: the current (2,3)x(2,3) program measured five distance-27 codes at n=598, giving a direct filling-only test.")
SHAPES = ("Keep equal (2,3)x(2,3) sides, with equal-length two-check, three-column components. The bound k>=46 gives rate 46/598 in cell 0.100. The parent's lightest X and Z logicals occupied sector-1 Acol/Bcol blocks across the grid.")
FILLING = ("Use balanced, weight-12 two-/three-term supports. Assign candidate-wide distinct nonzero binomial rotation gaps and vary homogeneous versus split triple cosets. Screen verified component kernels under elimination orders and valid group translations, then limit k above 46; target the parent's weight-27 witnesses.")
FAMILY_NAME = "r010_c23xc2_23x23_distinct_gap_orbit_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # Direct product C23 x C2, isomorphic to cyclic C46; NOT dihedral.
    m, r = 23, 2
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return m * b + a

    def mul(g, h):
        return ((g[0] + h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        return ((-g[0]) % m, (-g[1]) % r)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _component_columns(G, A, B):
    """Extract A and B component columns from the verified LP matrices."""
    hx, hz, _ = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q) - 1
    a_columns = [0] * (3 * q)
    b_columns = [0] * (3 * q)
    for i in range(2):
        for r in range(q):
            # HX row (i, Bcol0, r): Acol j occurs at block 3*j.
            row = hx[(3 * i) * q + r]
            for j in range(3):
                bits = (row >> (3 * j * q)) & mask
                while bits:
                    bit = bits & -bits
                    a_columns[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    bits ^= bit
    for alpha in range(2):
        for r in range(q):
            # HZ row (Acol0, alpha, r): the three B-column blocks.
            row = hz[alpha * q + r]
            for beta in range(3):
                bits = (row >> (beta * q)) & mask
                while bits:
                    bit = bits & -bits
                    b_columns[beta * q + bit.bit_length() - 1] |= 1 << (alpha * q + r)
                    bits ^= bit
    return a_columns, b_columns


def _component_obstructions(columns):
    """Count short zero-syndrome relations and shared check-neighbour pairs."""
    singleton = set(columns)
    pairs = {}
    triples = fours = overlaps = 0
    for i, x in enumerate(columns):
        for j in range(i):
            y = columns[j]
            syndrome = x ^ y
            if syndrome in singleton:
                triples += 1
            previous = pairs.get(syndrome)
            if previous is not None:
                # Shared-column collisions correspond to shorter relations.
                if previous[0] != i and previous[0] != j \
                        and previous[1] != i and previous[1] != j:
                    fours += 1
            else:
                pairs[syndrome] = (i, j)
            shared = (x & y).bit_count()
            if shared >= 2:
                overlaps += shared * (shared - 1) // 2
    return triples, fours, overlaps


def _cyclic_cell(rng, count, gap, triple_style):
    """Distinct labels with a prescribed, nonzero binomial rotation gap."""
    coset = rng.randrange(2)
    start = rng.randrange(23)
    if count == 2:
        # The chosen gap distinguishes this binomial from the other
        # binomials in the candidate, up to sign.
        step = gap if rng.randrange(2) else -gap
        other_coset = coset if triple_style == 3 else 1 - coset
        return [(start, coset), ((start + step) % 23, other_coset)]
    if triple_style == 0:
        a, b, c = rng.sample(range(23), 3)
        return [(a, coset), (b, coset), (c, coset)]
    a, b = rng.sample(range(23), 2)
    return [(a, coset), (b, coset),
            (rng.randrange(23), 1 - coset)]


def _kernel_probe(columns, rng):
    """Search verified component kernels and their C23 x C2 translates."""
    words = []
    for _ in range(7):
        order = list(range(len(columns)))
        rng.shuffle(order)
        pivots = {}
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                entry = pivots.get(p)
                if entry is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= entry[0]
                word ^= entry[1]
            else:
                words.append(word)

    words = sorted(set(words), key=int.bit_count)[:16]
    if not words:
        return (len(columns) + 1, 0)

    mask = (1 << 23) - 1

    def translated(word, shift, flip):
        out = 0
        for block in range(3):
            for coset in range(2):
                part = (word >> (46 * block + 23 * coset)) & mask
                part = ((part << shift) | (part >> (23 - shift))) & mask
                out |= part << (46 * block + 23 * (coset ^ flip))
        return out

    minimum = min(w.bit_count() for w in words)
    short_hits = sum(w.bit_count() <= 9 for w in words)
    for i, left in enumerate(words):
        for right in words[:i]:
            for shift in range(23):
                for flip in range(2):
                    combined = left ^ translated(right, shift, flip)
                    if combined:
                        weight = combined.bit_count()
                        if weight < minimum:
                            minimum = weight
                        if weight <= 9:
                            short_hits += 1
    return minimum, short_hits


def NEW_CTX(rng, G, idx):
    """Screen independent fillings of an indexed gap/coset design."""
    motif = idx % 20
    # Every row totals seven; every column totals four or five.
    supports = {}
    for side in ("A", "B"):
        first, second = rng.sample(range(3), 2)
        grid = [[2, 2, 2], [2, 2, 2]]
        grid[0][first] = grid[1][second] = 3
        supports[side] = grid
    assert _grid_weights(supports["A"], supports["B"]) <= W_MAX

    # Five triple-coset designs crossed with four binomial-coset
    # designs. Independent gaps and labels, not relabelings of a code.
    triple_modes = ((1, 1), (0, 1), (1, 0), (0, 0), (1, 1))
    triple_a, triple_b = triple_modes[motif % 5]
    binomial_mode = motif // 5
    shortlist = []
    for _ in range(28):
        # Eight binomials receive different unsigned C23 gaps.
        gaps = rng.sample(range(1, 12), 8)
        rng.shuffle(gaps)
        cells = {}
        next_gap = 0
        for side, style in (("A", triple_a), ("B", triple_b)):
            grid = []
            for row in supports[side]:
                output = []
                for count in row:
                    if count == 2:
                        gap = gaps[next_gap]
                        next_gap += 1
                        # Some indexed designs admit a same-coset pair;
                        # the remaining pairs have order 46.
                        same = (binomial_mode == 3 and
                                next_gap % 4 == 0)
                        output.append(_cyclic_cell(
                            rng, count, gap, 3 if same else style))
                    else:
                        output.append(_cyclic_cell(rng, count, 0, style))
                grid.append(output)
            cells[side] = grid

        ac, bc = _component_columns(G, cells["A"], cells["B"])
        (da, hits_a), (db, hits_b) = (
            _kernel_probe(ac, rng), _kernel_probe(bc, rng))
        a3, a4, ao = _component_obstructions(ac)
        b3, b4, bo = _component_obstructions(bc)
        grade = (min(da, db), -(hits_a + hits_b), da + db,
                 -(a3 + b3), -(a4 + b4), -(ao + bo))
        shortlist.append((grade, cells))

    best = None
    for grade, cells in sorted(shortlist, key=lambda x: x[0],
                               reverse=True)[:6]:
        k = lifted_k(G, cells["A"], cells["B"])
        if k > 74:
            continue
        quality = (grade[0], grade[1], grade[2],
                   -max(0, k - 48), grade[3], grade[4], grade[5],
                   min(k, 48))
        if best is None or quality > best[0]:
            best = (quality, cells)
    if best is None:
        raise ValueError("no screened filling remains in rate cell 0.100")
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the selected complete cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform rolled dihedral-58 (1,3)x(1,3) fillings do not reliably exceed distance 4: all five measured parent slots had d30s=4.
# REFUTED: Maximizing noncommutator weight among eight action-spanning labelings does not reliably suppress short witnesses: the order-117 parent had numerous d2s=9 slots and a measured d30s=18 slot.
# REFUTED: Penalizing component relations alone eliminates all short logicals on the order-117 bed: the current program still has many distinct d2s=9 slots.
# REFUTED: The dihedral (2,3)x(2,3) component-obstruction screen raises measured distance beyond 26: all five measured slots remained at d30s=26.
# REFUTED: The initial cyclic cross-coset kernel-probe screen raises measured distance beyond 27: all five measured slots remained at d30s=27.
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
