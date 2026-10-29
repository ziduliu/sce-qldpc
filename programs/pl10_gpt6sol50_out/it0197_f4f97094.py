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
CONCEPT = ("In dihedral Z29:Z2 of order 58, (1,3)x(1,3) sides with screened 3/3/2 supports should lower worse-basis circuit failures relative to all-triple sides.")
GROUP = ("Keep the measured order-58 dihedral group unchanged: the parent's 4.4435 fitness supplies a filling-only comparison.")
SHAPES = ("Keep equal (1,3)x(1,3) sides: each component has length 174, n=580, and guaranteed k>=232, rate 0.400. Parent weight-eight witnesses occupy sector-1 Acol/Bcol blocks; failed-shot enrichment is mild and inconsistent, with no persistent check-block hotspot.")
FILLING = ("Use two mixed-coset triples and one same-coset binomial per side, so both maximum check-row weights fall from 12 to 11. Screen twenty indexed gap-and-coset motifs on verified component columns for weight-2/4/5/6 relations. Test whether fewer circuit edges reduce the parent's roughly 100 worse-basis failures within 1750-3500 shots without creating shorter logicals; verified k keeps the target rate cell.")
FAMILY_NAME = "r040f_m29r2s28_13x13_single_binomial_shortword_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 58

def custom_ops(q):
    # metacyclic Z_29 x| Z_2: y x y^-1 = x^28 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 29, 2, 28
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _component_columns(rows, q, stride):
    """Extract a component's sector-1 columns from verified lifted rows.

    HX's first q rows select A at offsets j*3*q; HZ's first q rows
    select B at offsets j*q. No group action is reconstructed here.
    """
    columns = [0] * (3 * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:q]):
        for j in range(3):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << check
                bits -= low
    return columns


def _short_signature(columns):
    """Count weight-2 and weight-4 component-kernel witnesses."""
    two = len(columns) - len(set(columns)) + sum(c == 0 for c in columns)
    pairs = {}
    four = 0
    short_cycles = 0
    for j, right in enumerate(columns):
        for i in range(j):
            left = columns[i]
            shared = left & right
            if shared & (shared - 1):
                short_cycles += 1
            syndrome = left ^ right
            first = pairs.get(syndrome)
            if first is None:
                pairs[syndrome] = (i, j)
            elif i not in first and j not in first:
                four += 1
    return two, four, short_cycles


def _six_witness(columns, q):
    """Find up to two weight-six component relations.

    A regular group translation moves any occupied column to the
    representative of its protograph column. It therefore suffices to
    search relations containing one of the three representatives.
    """
    pairs = {}
    length = len(columns)
    for j in range(length):
        for i in range(j):
            pairs.setdefault(columns[i] ^ columns[j], (i, j))

    found = 0
    for fixed in (0, q, 2 * q):
        target = columns[fixed]
        for i in range(length - 2):
            if i == fixed:
                continue
            for j in range(i + 1, length - 1):
                if j == fixed:
                    continue
                partial = target ^ columns[i] ^ columns[j]
                for h in range(j + 1, length):
                    if h == fixed:
                        continue
                    pair = pairs.get(partial ^ columns[h])
                    if pair is not None and all(
                            v not in (fixed, i, j, h) for v in pair):
                        found += 1
                        if found == 2:
                            return found
    return found


def _five_witness(columns, q):
    """Detect short odd-weight relations containing a translated anchor."""
    pairs = {}
    length = len(columns)
    for j in range(length):
        for i in range(j):
            syndrome = columns[i] ^ columns[j]
            bucket = pairs.setdefault(syndrome, [])
            if len(bucket) < 4:
                bucket.append((i, j))

    found = 0
    for fixed in (0, q, 2 * q):
        target = columns[fixed]
        for j in range(length):
            if j == fixed:
                continue
            for i in range(j):
                if i == fixed:
                    continue
                for u, v in pairs.get(target ^ columns[i] ^ columns[j], ()):
                    if u not in (fixed, i, j) and v not in (fixed, i, j):
                        found += 1
                        break
                if found >= 2:
                    return found
    return found


def _draw_332_side(rng, idx, side):
    """Two coset-spanning triples and one rotation-gap binomial."""
    pools = {
        "small": tuple(range(1, 5)),
        "middle": tuple(range(5, 10)),
        "wide": tuple(range(10, 15)),
        "any": tuple(range(1, 15)),
    }
    plans = (
        ("small", "wide", "middle"),
        ("wide", "middle", "small"),
        ("middle", "wide", "wide"),
        ("small", "middle", "wide"),
        ("wide", "small", "middle"),
    )
    mode = (idx % 5 + 2 * (side == "B")) % 5
    pattern = (idx // 5) % 4
    binomial = (idx if side == "A" else idx // 3) % 3
    row = []
    triple_number = 0
    for j in range(3):
        kind = 2 if j == binomial else triple_number
        gap = rng.choice(pools[plans[mode][kind]])
        if rng.randrange(2):
            gap = -gap
        majority = ((pattern >> triple_number) & 1) if j != binomial \
                   else ((pattern + (side == "B")) & 1)
        first = rng.randrange(29)
        cell = [(first, majority),
                ((first + gap) % 29, majority)]
        if j != binomial:
            cell.append((rng.randrange(29), 1 - majority))
            triple_number += 1
        rng.shuffle(cell)
        row.append(cell)
    return [row]


def NEW_CTX(rng, G, idx):
    """Screen independent labels of twenty indexed support-and-gap motifs."""
    shortlist = []
    for _ in range(24):
        grids = {"A": _draw_332_side(rng, idx, "A"),
                 "B": _draw_332_side(rng, idx, "B")}
        hx, hz, n = lifted_matrices(G, grids["A"], grids["B"])
        ca = _component_columns(hx, G.q, 3)
        cb = _component_columns(hz, G.q, 1)
        a2, a4, ac = _short_signature(ca)
        b2, b4, bc = _short_signature(cb)
        a5 = _five_witness(ca, G.q) if a2 == a4 == 0 else 2
        b5 = _five_witness(cb, G.q) if b2 == b4 == 0 else 2
        cheap = (a2 + b2, max(a4, b4), a4 + b4,
                 max(a5, b5), a5 + b5,
                 max(ac, bc), ac + bc)
        shortlist.append((cheap, grids, ca, cb, hx, hz, n))

    shortlist.sort(key=lambda item: item[0])
    best_key, best = None, None
    for cheap, grids, ca, cb, hx, hz, n in shortlist[:4]:
        a6 = _six_witness(ca, G.q)
        b6 = _six_witness(cb, G.q)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        key = (not 0 < k <= 246, cheap[:5],
               max(a6, b6), a6 + b6, cheap[5:], abs(k - 234))
        if best_key is None or key < best_key:
            best_key, best = key, grids
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete triple or binomial cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled singleton-heavy supports suffice to suppress short sector-1 logicals at this bed: all five measured parent slots had weight-4 witnesses and worse-basis scores of only 3.17-3.34.
# REFUTED: The current parent's sparse mixed-coset profiles screen comparably to its full triples: many sparse slots had d2s=2, while its five measured slots all used full triples and had d2s=6 or 8.
# REFUTED: All-triple supports with weight-eight witnesses eliminate frequent circuit failures: the current parent's five measured worse bases still reached the 100-failure cap in 1750-3500 shots.
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
