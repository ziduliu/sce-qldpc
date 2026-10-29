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
CONCEPT = ("On Dic7 of order 28 with A=(1,2), B=(3,8), balanced zero/doublet B columns screened for short component relations should produce five rate-0.300-cell codes scoring above 20.3634.")
GROUP = ("Keep Dic7 and q=28 for a filling-only comparison. This n=532 bed measured d30s=9 in two slots but scored 20.3634 across its selected five.")
SHAPES = ("Keep (1,2)x(3,8): n=532, k>=140 and guaranteed rate 140/532=0.263; measured k>=147 reaches cell 0.300. The unequal A and B components have 56 and 224 variables. The parent's lightest logicals occupied sector-1 Acol/Bcol blocks, often including Bcol1-3 and Bcol5.")
FILLING = ("Keep four distinct terms per A cell. Indexed B profiles use three, five, six or eight columns of type (2,1,0), with the remaining columns (1,1,1); each B row totals eight and each B column three. Thus HX weighs 11 and HZ at most 12. Screen verified component relations, including sampled B weight-six relations, then compare exact k within the target rate cell.")
FAMILY_NAME = "r030_dic7_12x38_balanced_zero_doublet_sixprobe_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Dic7: x^14=1, y^2=x^7, y*x*y^-1=x^-1.
    m, half = 14, 7
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return m * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + half * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a + half if b else -a) % m, b)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _component_columns(hx, hz, q):
    """Extract both complete component syndromes from verified LP lifts."""
    cb, rb = B_SHAPE[1], B_SHAPE[0]
    mask = (1 << q) - 1
    acol = [0] * (2 * q)
    bcol = [0] * (cb * q)
    for r in range(q):
        for j in range(2):
            bits = (hx[r] >> (j * cb * q)) & mask
            while bits:
                bit = bits & -bits
                acol[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
    for r in range(rb * q):
        bits = hz[r] & ((1 << (cb * q)) - 1)
        while bits:
            bit = bits & -bits
            bcol[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    return acol, bcol


def _short_obstructions(cols, rng):
    """Count component dependencies; sampled weight-five hits break ties."""
    singles = set(cols)
    short = 20 * (len(cols) - len(singles) + int(0 in singles))
    pairs = {}
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                short += 3
            if syndrome in pairs:
                short += 1
            else:
                pairs[syndrome] = (i, j)
    fives = 0
    for _ in range(350):
        i, j, k = rng.sample(range(len(cols)), 3)
        pair = pairs.get(cols[i] ^ cols[j] ^ cols[k])
        if pair is not None and pair[0] not in (i, j, k) \
                and pair[1] not in (i, j, k):
            fives += 1
    return short, fives


def _b_support(rng, mode):
    """Directed balanced exchanges: 3, 5, 6 or 8 zero/doublet columns."""
    cycle = [(0, 1), (1, 2), (2, 0)]
    exchanges = (
        cycle,
        cycle + [(0, 1), (1, 0)],
        cycle * 2,
        cycle * 2 + [(0, 1), (1, 0)],
    )[mode]
    rows = [[1] * 8 for _ in range(3)]
    labels = rng.sample(range(3), 3)
    positions = rng.sample(range(8), 8)
    for j, (u, v) in zip(positions, exchanges):
        rows[labels[u]][j] = 2
        rows[labels[v]][j] = 0
    assert all(sum(row) == 8 for row in rows)
    assert all(sum(rows[i][j] for i in range(3)) == 3
               for j in range(8))
    return rows


def _a_cell(rng, order, occupancy):
    """Four distinct labels and a prescribed rotation-pair order."""
    import math
    gaps = [d for d in range(1, 14)
            if 14 // math.gcd(14, d) == order]
    first = rng.randrange(14)
    second = (first + rng.choice(gaps)) % 14
    rotations = [(first, 0), (second, 0)]
    if occupancy == 3:
        third = rng.choice([a for a in range(14)
                            if a not in (first, second)])
        rotations.append((third, 0))
    return rotations + [(a, 1) for a in
                        rng.sample(range(14), 4 - occupancy)]


def _b_cell(rng, G, count, cross_probability):
    if count == 0:
        return []
    if count == 1:
        return [G.pick(rng)]
    a, coset = rng.randrange(14), rng.randrange(2)
    if rng.random() < cross_probability:
        return [(a, coset), (rng.randrange(14), 1 - coset)]
    gap = rng.choice((1, 3, 5, 9, 11, 13))
    return [(a, coset), ((a + gap) % 14, coset)]


def _rooted_five(cols, q):
    """Count five-column A relations rooted by lift translation."""
    pairs = {}
    for i, x in enumerate(cols):
        for j in range(i):
            bucket = pairs.setdefault(x ^ cols[j], [])
            if len(bucket) < 4:
                bucket.append((j, i))
    found = 0
    for block in range(len(cols) // q):
        root = block * q
        if cols[root].bit_count() & 1:
            continue
        for i in range(len(cols)):
            if i == root:
                continue
            for j in range(i):
                if j == root:
                    continue
                target = cols[root] ^ cols[i] ^ cols[j]
                if any(u not in (root, i, j) and v not in (root, i, j)
                       for u, v in pairs.get(target, ())):
                    found += 1
                    if found >= 30:
                        return found
    return found


def _kernel_probe(cols, rng):
    """Probe component-kernel weights under two independent pivot orders."""
    words = []
    for _ in range(2):
        order = list(range(len(cols)))
        rng.shuffle(order)
        pivots = {}
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            else:
                words.append(word)
    short = sorted(words, key=int.bit_count)[:12]
    if not short:
        return len(cols) + 1
    minimum = min(word.bit_count() for word in short)
    for i, left in enumerate(short):
        for right in short[:i]:
            combined = left ^ right
            if combined:
                minimum = min(minimum, combined.bit_count())
    return minimum


def _sample_six(cols, rng):
    """Count disjoint equal-syndrome triples: verified weight-six kernels."""
    seen = {}
    hits = 0
    for _ in range(4200):
        triple = tuple(sorted(rng.sample(range(len(cols)), 3)))
        syndrome = cols[triple[0]] ^ cols[triple[1]] ^ cols[triple[2]]
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = triple
        elif all(j not in previous for j in triple):
            hits += 1
            if hits == 12:
                break
    return hits


def _component_cycles(cols):
    """Count repeated check-neighbour pairs in a verified component."""
    seen = set()
    repeats = 0
    for column in cols:
        checks = []
        while column:
            bit = column & -column
            checks.append(bit.bit_length() - 1)
            column ^= bit
        for pos, check in enumerate(checks):
            for earlier in checks[:pos]:
                pair = (earlier, check)
                repeats += pair in seen
                seen.add(pair)
    return repeats


def NEW_CTX(rng, G, idx):
    """Select independently labelled trials from twenty support/order motifs."""
    mode = (idx // 5) % 4
    a_order0, a_order1, occupancy0, occupancy1 = (
        (14, 14, 2, 2),
        (14, 7, 2, 2),
        (14, 14, 3, 2),
        (7, 14, 3, 2),
        (7, 7, 2, 3),
    )[idx % 5]
    trials = []
    for trial in range(32):
        counts = _b_support(rng, mode)
        A = [[_a_cell(rng, a_order0, occupancy0),
              _a_cell(rng, a_order1, occupancy1)]]
        cross_probability = (0.18, 0.24, 0.30, 0.36)[mode]
        B = [[_b_cell(rng, G, count, cross_probability)
              for count in row] for row in counts]
        hx, hz, n = lifted_matrices(G, A, B)
        acol, bcol = _component_columns(hx, hz, G.q)
        a_short, _ = _short_obstructions(acol, rng)
        b_short, _ = _short_obstructions(bcol, rng)
        a_five = _rooted_five(acol, G.q)
        b_six = _sample_six(bcol, rng)
        cycles = _component_cycles(bcol)
        probe_a = _kernel_probe(acol, rng)
        probe_b = _kernel_probe(bcol, rng)
        screen = (-max(a_short, b_short), -(a_short + b_short),
                  -b_six, -a_five, min(probe_a, probe_b),
                  probe_a + probe_b, -cycles, rng.random())
        trials.append((screen, A, B, hx, hz, n))

    best = best_grade = None
    for screen, A, B, hx, hz, n in sorted(
            trials, key=lambda item: item[0], reverse=True)[:9]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (int(147 <= k <= 169), screen, k)
        if best_grade is None or grade > best_grade:
            best_grade, best = grade, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# REFUTED: Uniform rolled order-24 labels did not reliably avoid short
# sector-two logicals: the parent's selected readings were 6, 7, 8, 8, 9.
# REFUTED: Extra k alone did not improve that bed: parent slot 25 had
# k-kb=12 but d2s=2, versus slot 58's k-kb=0 and d30s=9.
# REFUTED: Indexed order motifs and a 350-sample weight-five screen yield
# five distance-12 codes: only slots 40 and 57 reached d30s=12;
# measured slots 5, 6 and 41 retained d30s=10.
# REFUTED: The initial Dic7 three-row filling reliably exceeds distance
# nine: its five measured readings were 8, 8, 9, 8 and 9.
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
