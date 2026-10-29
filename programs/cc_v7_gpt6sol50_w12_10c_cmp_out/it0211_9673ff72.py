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
CONCEPT = ("For metacyclic C13 semidirect C4 (action 5), order 52, with A=B=(3,1), independently selecting three-term components without words through weight six and with fewer rooted weight-eight words will raise at least one measured distance above eight.")
GROUP = ("Keep the measured order-52 action-5 group: this bed's five measured codes had k=208 and d30s=8, allowing a filling-only comparison.")
SHAPES = ("Keep equal (3,1)x(3,1) sides: n=520, guaranteed k=208 and rate 0.400. Both are three-check, one-variable-block components. The parent's lightest X and Z logicals occupied sector-2 Arow/Brow blocks, so both components are screened.")
FILLING = ("Twenty action-coset occupancy patterns receive independent labels, with three distinct terms per cell and row weight 12. Select A and B independently using their verified-lift columns: reject short component relations through weight six, then probe rooted weight-eight words. This targets the parent's weight-eight witnesses while keeping k at its bound.")
FAMILY_NAME = "r040_c13rt4_31x31_independent_root8_components"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 52

def custom_ops(q):
    # C_13 semidirect C_4: y*x*y^-1 = x^5; 5^2 = -1 mod 13.
    m, r, s = 13, 4, 5
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 1)               # (ra, ca)
B_SHAPE = (3, 1)               # (rb, cb)


def _has_six_word(columns):
    """Detect a triple-syndrome collision; used after shorter relations."""
    seen = set()
    for i in range(len(columns) - 2):
        first = columns[i]
        for j in range(i + 1, len(columns) - 1):
            pair = first ^ columns[j]
            for third in columns[j + 1:]:
                syndrome = pair ^ third
                if syndrome in seen:
                    return True
                seen.add(syndrome)
    return False


def _missing_cosets(motif, side):
    """Twenty paired, row-specific C4 occupancy designs."""
    digit, family = motif % 4, motif // 4
    if side == "A":
        return (0, digit, family % 4)
    return (0, (family + 1) % 4,
            (digit + int(family == 4)) % 4)


def _motif_cell(rng, missing):
    """One independently sampled element in each of three C4 cosets."""
    cell = [(rng.randrange(13), b)
            for b in range(4) if b != missing]
    rng.shuffle(cell)
    return cell


def _verified_component_columns(rows, q, starts):
    """Transpose sector-2 blocks of an evaluator-equivalent lift."""
    columns = [0] * (3 * q)
    mask = (1 << q) - 1
    for r, row in enumerate(rows[:q]):
        for j, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << r
                bits ^= low
    return columns


def _pair_obstructions(columns):
    """Count repeated column and pair syndromes, plus check-pair overlaps."""
    duplicates = len(columns) - len(set(columns))
    pairs = {}
    fours = overlaps = 0
    for i, x in enumerate(columns):
        for y in columns[i + 1:]:
            syndrome = x ^ y
            fours += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
            shared = (x & y).bit_count()
            overlaps += shared * (shared - 1) // 2
    return duplicates, fours, overlaps


def _rooted_eight_probe(columns, q, node_cap=1300):
    """Count sampled words through eight, rooted in each variable block.

    Regular component lifts permit a translation of any word to a
    variable at group identity. Exhaustive search is not claimed:
    each root has its own bounded search.
    """
    neighbours = [[] for _ in range(q)]
    for v, column in enumerate(columns):
        bits = column
        while bits:
            low = bits & -bits
            neighbours[low.bit_length() - 1].append(v)
            bits ^= low

    found = set()
    for block in range(3):
        root = block * q
        remaining = [node_cap]

        def visit(syndrome, used, depth):
            if not syndrome:
                found.add(used)
                return
            if depth >= 8 or remaining[0] <= 0:
                return
            if syndrome.bit_count() > 3 * (8 - depth):
                return
            remaining[0] -= 1
            low = syndrome & -syndrome
            check = low.bit_length() - 1
            options = [v for v in neighbours[check]
                       if not (used >> v) & 1]
            options.sort(
                key=lambda v: (columns[v] & syndrome).bit_count(),
                reverse=True)
            for v in options:
                visit(syndrome ^ columns[v],
                      used | (1 << v), depth + 1)
                if len(found) >= 24 or remaining[0] <= 0:
                    return

        visit(columns[root], 1 << root, 1)
        if len(found) >= 24:
            break
    return len(found)


def _select_component(rng, candidates):
    """Select one independently labelled side, not a relabelled copy."""
    candidates.sort(key=lambda item: item[0])
    best = None
    for coarse, grid, columns in candidates[:4]:
        if coarse[0]:
            grade = (2, 0, -coarse[0], -coarse[1], -coarse[2])
        elif coarse[1]:
            grade = (4, 0, 0, -coarse[1], -coarse[2])
        elif _has_six_word(columns):
            grade = (6, 0, 0, 0, -coarse[2])
        else:
            hits = _rooted_eight_probe(columns, 52)
            grade = (9 if hits == 0 else 8, -hits,
                     0, 0, -coarse[2])
        grade += (rng.random(),)
        if best is None or grade > best[0]:
            best = (grade, grid)
    return best[1]


def NEW_CTX(rng, G, idx):
    """Screen each verified sector-2 component independently."""
    motif = idx % 20
    missing_a = _missing_cosets(motif, "A")
    missing_b = _missing_cosets(motif, "B")
    ranked_a, ranked_b = [], []
    q = G.q

    for _ in range(48):
        A = [[_motif_cell(rng, missing_a[i])]
             for i in range(3)]
        B = [[_motif_cell(rng, missing_b[i])]
             for i in range(3)]
        hx, hz, _ = lifted_matrices(G, A, B)

        # HX check block zero restricts to B's three sector-2 blocks.
        bcols = _verified_component_columns(
            hx, q, [q + alpha * q for alpha in range(3)])
        # HZ check block zero restricts to A's three sector-2 blocks.
        acols = _verified_component_columns(
            hz, q, [q + i * 3 * q for i in range(3)])
        ranked_a.append((_pair_obstructions(acols), A, acols))
        ranked_b.append((_pair_obstructions(bcols), B, bcols))

    return {"A": _select_component(rng, ranked_a),
            "B": _select_component(rng, ranked_b)}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently selected complete cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform rolled supports did not reliably suppress light logicals:
# the parent's 60 distinct slots included many d2s=2 readings.
# REFUTED: Adjacent-coset 3/4 labeling removes weight-20 sector-1
# witnesses: all five measured order-116 slots had d30s=20.
# REFUTED: Three-term cells in the parent's (1,3)x(1,3) shape suffice
# to exceed distance eight: all five measured order-52 slots had d30s=8.
# REFUTED: Joint screening of 18 transposed-component fillings on this
# (3,1)x(3,1) bed exceeds distance eight: its five measured slots all
# had k=208 and d30s=8, with sector-2 weight-eight witnesses.
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
