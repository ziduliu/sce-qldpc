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
CONCEPT = ("On cyclic C37 of order 37 with A=(2,3), B=(2,4), "
           "screening five legal B support profiles for translated "
           "component-minor words should raise the five-code fitness "
           "above 39.1566 in the rate-0.150 cell.")
GROUP = ("Keep cyclic C37 and order 37. This n=592 bed reached "
         "fitness 39.1566, making a filling-only comparison direct.")
SHAPES = ("Keep (2,3)x(2,4): n=592 and guaranteed k>=74, rate "
          "at least 0.125. Its unequal two-row A and B components "
          "have 111 and 148 variables. The parent's lightest X "
          "witnesses had weights 16-18 in sector-1 Acol/Bcol blocks; "
          "its measured Z witnesses had weights 18-21.")
FILLING = ("Keep binomial A cells with distinct within-row cyclic "
           "gaps. Test five B support profiles with eight terms per "
           "row, including columns totaling three or five; both "
           "row-weight caps remain 12. Screen independent labels on "
           "verified component minors, penalize words of weights "
           "at most 18 and 20, and check k above the 74 bound.")
FAMILY_NAME = "r015_c37_23x24_multiprofile_minor_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic C37, represented by singleton exponent tuples.
    els = [(a,) for a in range(37)]

    def red(g):
        return (g[0] % 37,)

    def idx(g):
        return g[0] % 37

    def mul(g, h):
        return ((g[0] + h[0]) % 37,)

    def inv(g):
        return ((-g[0]) % 37,)

    return dict(elements=els, ident=(0,), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _draw_design(rng, idx):
    """One independently labelled member of an indexed support motif."""
    profiles = (
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((1, 3, 2, 2), (3, 1, 2, 2)),
        ((1, 1, 3, 3), (3, 3, 1, 1)),
        ((4, 2, 1, 1), (1, 1, 2, 4)),
        ((1, 3, 3, 1), (2, 2, 1, 3)),
    )
    gap_pools = (
        tuple(range(1, 19)),
        tuple(range(5, 19)),
        (1, 3, 5, 7, 9, 11, 13, 15, 17),
        tuple(range(1, 13)),
    )
    pool = gap_pools[(idx // 5) % 4]

    A = [[None] * 3 for _ in range(2)]
    gaps = rng.sample(pool, 6)
    for i in range(2):
        for j in range(3):
            start = rng.randrange(37)
            gap = gaps[3 * i + j]
            A[i][j] = [(start,), ((start + gap) % 37,)]

    counts = profiles[idx % 5]
    columns = rng.sample(range(4), 4)
    if rng.randrange(2):
        counts = counts[::-1]
    B = [[None] * 4 for _ in range(2)]
    for i in range(2):
        used_gaps = set()
        for t, count in enumerate(counts[i]):
            j = columns[t]
            if count == 1:
                values = [rng.randrange(37)]
            elif count == 2:
                choices = [g for g in pool if g not in used_gaps]
                gap = rng.choice(choices)
                used_gaps.add(gap)
                start = rng.randrange(37)
                values = [start, (start + gap) % 37]
            else:
                # Avoid a repeated internal difference in a dense cell.
                for _ in range(100):
                    values = rng.sample(range(37), count)
                    differences = {
                        min((values[u] - values[v]) % 37,
                            (values[v] - values[u]) % 37)
                        for u in range(count) for v in range(u)
                    }
                    if len(differences) >= (
                            3 if count == 3 else 5):
                        break
            B[i][j] = [(a,) for a in values]

    assert all(sum(len(cell) for cell in row) == 8 for row in B)
    assert _grid_weights(
        [[len(cell) for cell in row] for row in A],
        [[len(cell) for cell in row] for row in B]) <= 12
    return A, B


def _minor_grade(hx, hz, q):
    """Find short translated two-row component-kernel minor words.

    Row supports come from the verified lifts, including their distinct
    A-left and B-right conventions. The products below are cyclic
    convolutions; each three-column minor is a kernel word.
    """
    from itertools import combinations

    mask = (1 << q) - 1

    def rotate(x, shift):
        if not shift:
            return x
        return ((x << shift) | (x >> (q - shift))) & mask

    def product(x, y):
        result = 0
        while y:
            bit = y & -y
            result ^= rotate(x, bit.bit_length() - 1)
            y ^= bit
        return result

    def one_side(rows, blocks, stride):
        polynomials = [
            [(rows[r] >> (j * stride * q)) & mask
             for j in range(blocks)]
            for r in range(2)
        ]
        words = []
        for triple in combinations(range(blocks), 3):
            parts = [0] * blocks
            for j in triple:
                u, v = [t for t in triple if t != j]
                parts[j] = (product(polynomials[0][u], polynomials[1][v])
                            ^ product(polynomials[1][u], polynomials[0][v]))
            word = sum(part << (q * j) for j, part in enumerate(parts))
            if word:
                words.append(word)
        if not words:
            return (0, 0, 0)

        minimum = min(w.bit_count() for w in words)
        light18 = light20 = 0
        total = sum(w.bit_count() for w in words)
        for i, left in enumerate(words):
            for j in range(i, len(words)):
                right = words[j]
                for shift in range(q):
                    translated = 0
                    for block in range(blocks):
                        part = (right >> (block * q)) & mask
                        translated |= rotate(part, shift) << (block * q)
                    combined = left ^ translated
                    if combined:
                        weight = combined.bit_count()
                        minimum = min(minimum, weight)
                        light18 += weight <= 18
                        light20 += weight <= 20
        return (minimum, -light18, -light20, total)

    a = one_side((hx[0], hx[4 * q]), 3, 4)
    b = one_side((hz[0], hz[q]), 4, 1)
    return (min(a[0], b[0]), a[0] + b[0],
            a[1] + b[1], a[2] + b[2],
            a[3] + b[3])


def NEW_CTX(rng, G, idx):
    """Compare fresh labels and verify the rank of the best prospects."""
    shortlisted = []
    for trial in range(30):
        A, B = _draw_design(rng, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        minor = _minor_grade(hx, hz, G.q)
        shortlisted.append((minor, trial, A, B, hx, hz, n))
    shortlisted.sort(key=lambda item: item[0], reverse=True)

    best = None
    for minor, trial, A, B, hx, hz, n in shortlisted[:10]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Compare predicted witness weight and exact dimension, while
        # retaining the rate-0.150 cell.
        eligible = 75 <= k <= 103
        predicted = min(minor[0], 20)
        grade = (eligible, k * predicted * predicted,
                 minor[0], minor[1], minor[2], minor[3], -trial)
        if best is None or grade > best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without redrawing."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support variation on the parent's (1,3)x(1,3) bed
# suffices to remove weight-4 logicals: all five measured slots had d30s=4.
# REFUTED: The previous (1,4)x(1,4) penalty screen suffices to remove
# short logicals: its five measured slots all retained d30s=3 witnesses.
# REFUTED: Triangle-free binomial gaps on the parent's order-34
# (1,4)x(1,4) bed eliminate weight-four logicals: all five measured
# slots retained d30s=4, with thousands of verified weight-four words.
# REFUTED: Noncommuting order-six A binomials and screened B columns on
# S3 x C3 (1,4)x(2,7) suffice for d30s>=5: all five measured slots
# retained d30s=4, with k at its rank-count bound.
# REFUTED: The prior translated-minor screen alone selected five
# distances above 18 on C37 (2,3)x(2,4): its measured d30s values
# were 16, 16, 16, 17, and 18.
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
