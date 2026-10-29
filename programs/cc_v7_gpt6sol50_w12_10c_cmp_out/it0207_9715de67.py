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
CONCEPT = ("On C11 semidirect C4 of order 44 and shapes (2,3)x(2,3), balanced supports and verified-component kernel screening can produce five distinct rate-below-0.125 codes with distances above the parent's 12.")
GROUP = ("Keep the measured parent's C11 semidirect C4 law and order 44. The new shape still gives n=13*44=572; the order-46 (2,3)x(2,3) frontier's five distance-27/28 readings motivate testing this previously untried order-44 bed.")
SHAPES = ("A=(2,3), B=(2,3) guarantees k>=44, rate>=44/572=0.077; selecting k<=71 keeps the measured rate below 0.125. Both sides now have equal-length two-check, three-column components. The parent's lightest weight-12 Z witnesses used sector-1 Acol0/1 and Bcol0-4 blocks, not Arow/Brow blocks.")
FILLING = ("Cross five A with four B support profiles, each having row totals at most seven and column totals at most five: both HX and HZ weights are at most 12. Three independent coset-label models and short-kernel screening on verified component lifts target the parent's weight-12 witnesses; exact rank screening limits k above the new bound of 44.")
FAMILY_NAME = "r010_c11sdC4_23x23_balanced_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # x^11=y^4=e, y*x*y^-1=x^-1.
    # This group has elements of order four and is not cyclic.
    els = [(a, b) for b in range(4) for a in range(11)]

    def red(g):
        return (g[0] % 11, g[1] % 4)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    def mul(g, h):
        return ((g[0] + (-1 if g[1] % 2 else 1) * h[0]) % 11,
                (g[1] + h[1]) % 4)

    def inv(g):
        b = (-g[1]) % 4
        return (((-1 if b % 2 else 1) * -g[0]) % 11, b)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _component_columns(rows, starts, q):
    """Transpose component blocks of the scaffold's exact lifted rows."""
    columns = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        mark = 1 << check
        for block, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                columns[block * q + bit.bit_length() - 1] |= mark
                bits ^= bit
    return columns


def _relation_cost(columns):
    """Count short component relations; this is not a distance proof."""
    singles = set(columns)
    two = int(0 in singles) + len(columns) - len(singles)
    pairs = set()
    three = four = cycles = weak = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            three += syndrome in singles
            four += syndrome in pairs
            pairs.add(syndrome)
            common = (left & right).bit_count()
            cycles += common * (common - 1) // 2
            weak += syndrome.bit_count() <= 4
    return two, three, four, cycles, weak


def _support_profile(rng, idx):
    """Twenty A/B count designs; all row totals <=7, column totals <=5."""
    profiles = (
        ((4, 2, 1), (1, 3, 3)),
        ((3, 2, 2), (2, 2, 2)),
        ((3, 3, 1), (2, 1, 3)),
        ((4, 1, 2), (1, 2, 3)),
        ((3, 2, 2), (2, 1, 3)),
    )

    def permute(profile):
        order = rng.sample(range(3), 3)
        rows = [[row[j] for j in order] for row in profile]
        if rng.randrange(2):
            rows.reverse()
        return rows

    a = permute(profiles[idx % 5])
    b = permute(profiles[(idx // 5) % 4])
    assert _grid_weights(a, b) <= 12
    return a, b, (idx // 20) % 3


def _label_cell(rng, count, variation, side):
    """Fresh distinct labels, with three different coset-incidence rules."""
    patterns = (
        {1: (0,), 2: (0, 1), 3: (0, 1, 1),
         4: (0, 0, 1, 1)},
        {1: (0,), 2: (0, 2), 3: (0, 1, 2),
         4: (0, 1, 2, 3)},
        {1: (0,), 2: (0, 0), 3: (0, 1, 3),
         4: (0, 1, 1, 2)},
    )
    offsets = patterns[variation][count]
    base = rng.randrange(4)
    chosen = [(base + offset) % 4 for offset in offsets]
    rng.shuffle(chosen)
    cell = []
    for b in chosen:
        a = rng.randrange(11)
        while (a, b) in cell:
            a = rng.randrange(11)
        cell.append((a, b))
    return cell


def _kernel_probe(columns, rng):
    """Find genuine short component-kernel words using shuffled pivots."""
    best = len(columns) + 1
    for _ in range(5):
        order = list(range(len(columns)))
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                pivot = syndrome.bit_length() - 1
                previous = pivots.get(pivot)
                if previous is None:
                    pivots[pivot] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            else:
                words.append(word)
                best = min(best, word.bit_count())

        short = sorted(words, key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        for i, word in enumerate(short[:12]):
            for j, other in enumerate(short[:i]):
                pair = word ^ other
                for third in short[:j]:
                    best = min(best, (pair ^ third).bit_count())
    return best


def NEW_CTX(rng, G, idx):
    """Screen both exact two-check components, then enforce cell-0 rate."""
    ac, bc, variation = _support_profile(rng, idx)
    q = G.q
    options = []
    for _ in range(14):
        A = [[_label_cell(rng, count, variation, "A")
              for count in row] for row in ac]
        B = [[_label_cell(rng, count, variation, "B")
              for count in row] for row in bc]
        hx, hz, n = lifted_matrices(G, A, B)

        # HX at B-column 0 contains both A check rows. HZ at A-column
        # 0 contains both B check rows; retain all q rows of each.
        a_rows = [hx[(i * 3) * q + r]
                  for i in range(2) for r in range(q)]
        b_rows = hz[:2 * q]
        acol = _component_columns(
            a_rows, [j * 3 * q for j in range(3)], q)
        bcol = _component_columns(
            b_rows, [j * q for j in range(3)], q)
        da = _kernel_probe(acol, rng)
        db = _kernel_probe(bcol, rng)
        sa, sb = _relation_cost(acol), _relation_cost(bcol)
        penalty = (10000 * (sa[0] + sb[0])
                   + 1000 * (sa[1] + sb[1])
                   + 100 * (sa[2] + sb[2])
                   + sa[3] + sb[3] + sa[4] + sb[4])
        grade = (min(da, db), da + db, -penalty, rng.random())
        options.append((grade, A, B, hx, hz, n))

    fallback = None
    best = None
    for grade, A, B, hx, hz, n in sorted(
            options, key=lambda item: item[0], reverse=True)[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if fallback is None or k < fallback[0]:
            fallback = (k, A, B)
        if 44 <= k <= 71:
            key = (grade[0] * grade[0] * k, grade[1],
                   grade[2], grade[3])
            if best is None or key > best[0]:
                best = (key, A, B)
    if best is not None:
        return {"A": best[1], "B": best[2]}
    return {"A": fallback[1], "B": fallback[2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Mostly-singleton rolled supports suffice for d30s >= 10 in this bed:
# refuted by the parent's five measured slots (four d30s=6, one d30s=8).
# REFUTED: All-two-term A cells and one doubled B cell per column
# reliably attain d30s>=12: four of the current program's five
# measured slots had d30s=11.
# REFUTED: Screening several cyclic order-23 support profiles for
# component expansion suffices to exceed distance 12: all five
# measured slots of the current parent had d30s=12.
# REFUTED: Balanced C11 semidirect C4 (1,2)x(3,5) fillings break the
# distance-12 ceiling: all five selected slots had d30s=12.
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
