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
CONCEPT = ("On dihedral Z23:Z2 of order 46 with A=(1,2), B=(2,5), balanced B-column degrees and exact component screening through weight five should increase distance-12 readings.")
GROUP = ("Keep the order-46 dihedral group: this bed measured five codes at d30s=8-10, while its related (2,3)x(2,3) bed reached fitness 52.0.")
SHAPES = ("Keep (1,2)x(2,5): n=552, guaranteed k=138 and rate 0.250, with unequal component lengths 92 and 230. All measured parent minimum-weight logicals occupied sector-1 Acol/Bcol blocks.")
FILLING = ("Put four terms in each A cell and give every B column three or four terms. Compare indexed count and coset motifs using exact symmetry-rooted component screening through weight five on verified lifts; monitor k above 138. This targets the parent's weight-8-to-10 sector-1 witnesses.")
FAMILY_NAME = "r025_m23r2s22_12x25_balanced_rooted5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # Metacyclic Z_23 x| Z_2, with inversion action.
    m, r, s = 23, 2, 22
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _component_columns(rows, q, blocks, stride):
    """Extract component columns from the verified LP matrices."""
    cols = [0] * (blocks * q)
    mask = (1 << q) - 1
    for r, row in enumerate(rows):
        for j in range(blocks):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                bit = bits & -bits
                cols[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
    return cols


def _component_obstructions(cols, G):
    """Find short component words, rooting weight five by group symmetry."""
    q = G.q
    singles = set(cols)
    bad2 = len(cols) - len(singles) + int(0 in singles)
    pairs = {}
    bad3 = bad4 = 0
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            bad3 += syndrome in singles
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = [(j, i)]
            else:
                bad4 += len(previous)
                if len(previous) < 4:
                    previous.append((j, i))

    # Left-lift component words admit right translations; right-lift
    # component words admit left translations. Either action moves
    # one variable, in its own protograph block, to the identity.
    # Odd-degree B columns cannot belong to an odd-weight zero word
    # without an even-degree column also belonging to it.
    roots = [block * q + G.idx(G.e)
             for block in range(len(cols) // q)
             if cols[block * q].bit_count() % 2 == 0]
    bad5 = 0
    for root in roots:
        pivot = cols[root]
        for i, x in enumerate(cols):
            if i == root:
                continue
            for j in range(i):
                if j == root:
                    continue
                target = pivot ^ x ^ cols[j]
                if any(u not in (root, i, j) and v not in (root, i, j)
                       for u, v in pairs.get(target, ())):
                    bad5 += 1
                    if bad5 >= 16:
                        break
            if bad5 >= 16:
                break
        if bad5 >= 16:
            break

    seen_check_pairs = set()
    cycles = 0
    for column in cols:
        checks = []
        while column:
            bit = column & -column
            checks.append(bit.bit_length() - 1)
            column ^= bit
        for a, left in enumerate(checks):
            for right in checks[:a]:
                edge = (min(left, right), max(left, right))
                cycles += edge in seen_check_pairs
                seen_check_pairs.add(edge)
    return bad2, bad3, bad4, bad5, cycles


def _b_counts(rng, motif):
    """Five profiles: four 8/8 rows and one 8/7 all-odd-column test."""
    profiles = (
        ((2, 2, 2, 1, 1), (1, 1, 2, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((1, 2, 2, 2, 1), (3, 1, 1, 1, 2)),
        ((2, 2, 2, 1, 1), (1, 1, 1, 2, 2)),
        ((2, 2, 1, 1, 2), (1, 1, 2, 2, 2)),
    )
    order = rng.sample(range(5), 5)
    counts = [[row[j] for j in order] for row in profiles[motif]]
    if rng.randrange(2):
        counts.reverse()
    return counts


def _a_terms(rng, rotations):
    return ([(a, 0) for a in rng.sample(range(23), rotations)] +
            [(a, 1) for a in rng.sample(range(23), 4 - rotations)])


def _b_terms(rng, count, cross):
    if count == 1:
        return [(rng.randrange(23), rng.randrange(2))]
    coset = rng.randrange(2)
    if count == 2 and cross:
        return [(rng.randrange(23), coset),
                (rng.randrange(23), 1 - coset)]
    first = [(a, coset) for a in rng.sample(range(23), 2)]
    if count == 2:
        return first
    return first + [(rng.randrange(23), 1 - coset)]


def NEW_CTX(rng, G, idx):
    """Independently screen labels in twenty count/coset design motifs."""
    motif = idx % 20
    profile, a_mode = motif % 5, motif // 5
    a_occupancies = ((2, 2), (3, 2), (1, 2), (3, 1))
    trials = []
    for trial in range(22):
        A = [[_a_terms(rng, occupancy)
              for occupancy in a_occupancies[a_mode]]]
        counts = _b_counts(rng, profile)
        B = [[_b_terms(rng, counts[i][j],
                       profile == 4 and (i + j + trial) % 3 == 0)
              for j in range(5)] for i in range(2)]
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx[:G.q], G.q, 2, 5)
        bc = _component_columns(hz[:2 * G.q], G.q, 5, 1)
        sa = _component_obstructions(ac, G)
        sb = _component_obstructions(bc, G)
        grade = (sa[0] + sb[0], sa[1] + sb[1],
                 sa[2] + sb[2], max(sa[3], sb[3]),
                 sa[3] + sb[3], sa[4] + sb[4], rng.random())
        trials.append((grade, A, B, hx, hz, n))

    best = None
    for grade, A, B, hx, hz, n in sorted(
            trials, key=lambda item: item[0])[:7]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        rating = (int(k > 151),) + grade[:5] + (-k,) + grade[5:]
        if best is None or rating < best[0]:
            best = (rating, A, B)
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# The measured rolled (5,2)x(4,2) seed's five selected slots had
# d30s only 6-9; uniform unscreened labels did not remove its short
# sector-2 witnesses.
# REFUTED: Screening component words through weight four and sampling
# weight-five triples guarantees d30s>=12 in this bed: the current
# program's five measured readings were 8, 9, 10, 10 and 10.
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
