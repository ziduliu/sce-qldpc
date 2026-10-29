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
CONCEPT = ("C19 semidirect C3 with action 7, order 57, shapes (1,3)x(1,3): mixed-coset triples and exact rejection of weight<=4 logicals should yield five measured d30s>=8 codes in cell 0.400.")
GROUP = ("7^3=1 mod 19 defines an odd-order nonabelian group with no involutions, genuinely different from Dic11. The order-39 C13 semidirect C3 bed reached fitness 44.8 at lower rate; that motivates this related construction but does not establish its distance at rate 0.400.")
SHAPES = ("Target the weak 0.400 frontier, currently fitness 6.5: equal (1,3) sides give n=570, k>=228, and guaranteed rate 0.400. Both components have 171 columns and 57 checks. All five measured Dic11 parent codes had minimum witnesses only in sector-1 Acol/Bcol blocks. Exact lifted ranks keep every submitted rate below 0.425.")
FILLING = ("Four support pairs times five quotient-occupancy models define twenty deliberate designs, with both row-weight sums <=12. Binomials use one coset so their relative labels have order 19, not 3; larger cells mix quotient cosets. Shortlist independent labelings by component four-cycles, refine using actual component-kernel witnesses, and reject full-code logicals of weight<=4 exactly. Larger distance readings remain search upper bounds.")
FAMILY_NAME = "r040_C19sdC3_13x13_quotient_kernel_exact4_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    # x^19=y^3=1, y*x*y^-1=x^7; 7^3=1 modulo 19.
    if q != 57:
        raise ValueError("C19 semidirect C3 has order 57")
    action = (1, 7, 11)
    els = [(a, b) for b in range(3) for a in range(19)]

    def red(g):
        return (g[0] % 19, g[1] % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 19, (b+d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 19, bi)

    def idx(g):
        a, b = red(g)
        return 19*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _support_designs():
    """Four support geometries crossed with five quotient constructions.

    Each side has at most one binomial. Two binomial columns would
    introduce an avoidable commuting-normal-subgroup bottleneck.
    For single-row sides the two caps are sum(A)+max(B) and
    sum(B)+max(A), checked explicitly without scaffold initialization.
    """
    profiles = (
        ((3, 3, 3), (3, 3, 3)),
        ((2, 3, 3), (3, 3, 3)),
        ((2, 3, 3), (2, 3, 3)),
        ((2, 3, 4), (2, 3, 3)),
    )
    designs = []
    for a, b in profiles:
        assert sum(a) + max(b) <= 12
        assert sum(b) + max(a) <= 12
        for mode in range(5):
            designs.append((a, b, mode))
    return tuple(designs)


_SUPPORT_DESIGNS = _support_designs()


def _four_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        bits = []
        while row:
            bit = row & -row
            bits.append(bit.bit_length()-1)
            row ^= bit
        for a, x in enumerate(bits):
            for y in bits[a+1:]:
                key = (x, y)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _column_syndromes(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= check
            row ^= bit
    return cols


def _kernel_probe(rows, width, rng, trials=16):
    """Actual nonzero classical kernel witnesses; no quantum certificate."""
    cols = _column_syndromes(rows, width)
    singles = {}
    for i, syndrome in enumerate(cols):
        if syndrome == 0:
            return 1
        if syndrome in singles:
            return 2
        singles[syndrome] = i

    # Exact detection of component dependencies of weight <=4.
    pairs = {}
    for i in range(width):
        for j in range(i):
            syndrome = cols[i] ^ cols[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs[syndrome] = (i, j)

    best = width + 1
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                hit = pivots.get(p)
                if hit is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= hit[0]
                word ^= hit[1]
            if syndrome == 0:
                words.append(word)
                best = min(best, word.bit_count())

        for a, word in enumerate(words):
            for other in words[:a]:
                best = min(best, (word ^ other).bit_count())

        # Elementary kernel-word reductions, followed by another pair pass.
        light = sorted(words, key=int.bit_count)[:32]
        for _ in range(2):
            changed = False
            for a in range(len(light)):
                word = light[a]
                weight = word.bit_count()
                for b, other in enumerate(light):
                    if a == b:
                        continue
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                        changed = True
                light[a] = word
                best = min(best, weight)
            if not changed:
                break
        for a, word in enumerate(light):
            for other in light[:a]:
                candidate = word ^ other
                if candidate:
                    best = min(best, candidate.bit_count())
        if best <= 5:
            break
    return best


def _small_logical(checks, stabilizers, n):
    """Exact existence test for any nontrivial logical of weight <=4.

    Reduce qubit vectors modulo the stabilizer row space once. Equal
    check syndromes with different reduced vectors certify a logical.
    Single/pair collisions cover every support of weight one to four.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]

    residues = []
    for c in range(n):
        word, remainder = 1 << c, 0
        while word:
            p = word.bit_length()-1
            hit = basis.get(p)
            if hit is None:
                bit = 1 << p
                remainder |= bit
                word ^= bit
            else:
                word ^= hit
        residues.append(remainder)

    cols = _column_syndromes(checks, n)
    singles = {}
    for syndrome, remainder in zip(cols, residues):
        if syndrome == 0 and remainder:
            return True
        if syndrome in singles:
            if remainder != singles[syndrome]:
                return True
        else:
            singles[syndrome] = remainder

    pairs = {}
    for i in range(n):
        si, ri = cols[i], residues[i]
        for j in range(i):
            syndrome = si ^ cols[j]
            remainder = ri ^ residues[j]
            if syndrome == 0 and remainder:
                return True
            if syndrome in singles and remainder != singles[syndrome]:
                return True
            if syndrome in pairs:
                if remainder != pairs[syndrome]:
                    return True
            else:
                pairs[syndrome] = remainder
    return False


def _component_checks(hx, hz, q):
    """Extract both components only from the scaffold's VERIFIED lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    arows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i*cb)*q+r]
            packed = 0
            for j in range(ca):
                packed |= ((source >> (j*cb*q)) & mask) << (j*q)
            arows.append(packed)
    bmask = (1 << (cb*q))-1
    brows = [hz[a*q+r] & bmask for a in range(rb) for r in range(q)]
    return arows, brows


def _label_grid(rng, G, counts, mode, side):
    """Independent rotation labels within a prescribed quotient pattern."""
    row = []
    for j, count in enumerate(counts):
        offset = (j + mode + (side == "B")) % 3
        if count == 2:
            # Different cosets would give relative order 3 and a
            # three-point kernel inside this isolated binomial block.
            bs = [offset, offset]
        elif mode == 0:
            bs = [(offset+t) % 3 for t in range(count)]
        elif mode == 1:
            bs = [offset]*(count-1) + [(offset+1) % 3]
        elif mode == 2:
            bs = [offset]*(count-1) + [(offset+2) % 3]
        elif mode == 3:
            if (j + (side == "B")) % 2:
                bs = [offset]*(count-1) + [(offset+1) % 3]
            else:
                bs = [(offset+t) % 3 for t in range(count)]
        else:
            # One normal-subgroup-concentrated column, with the other
            # columns mixed; not an all-commuting component.
            if j == 0:
                bs = [offset]*count
            else:
                bs = [offset]*(count-1) + [(offset+1) % 3]
        cell = []
        for b in range(3):
            pool = [g for g in G.elements if g[1] == b]
            cell.extend(rng.sample(pool, bs.count(b)))
        row.append(cell)
    return [row]


def NEW_CTX(rng, G, idx):
    """Optimize a support/quotient design, then screen actual small logicals."""
    ta, tb, mode = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa, sb = [list(ta)], [list(tb)]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    pool = []

    for attempt in range(48):
        A = _label_grid(rng, G, ta, mode, "A")
        B = _label_grid(rng, G, tb, mode, "B")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # The rank bound is 228/570=0.400; avoid rank inflation
        # moving this program into the neighbouring 0.450 cell.
        if not (375*n <= 1000*k < 425*n):
            continue
        ar, br = _component_checks(hx, hz, G.q)
        ca, cb = _four_cycles(ar), _four_cycles(br)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ar": ar, "br": br, "cycles": (max(ca, cb), ca+cb),
            "attempt": attempt,
        }
        pool.append(item)
        pool.sort(key=lambda x: (x["cycles"], x["attempt"]))
        del pool[8:]

    if not pool:
        raise ValueError("no rank-admissible labeling in cell 0.400")

    def merit(item):
        d = min(item["da"], item["db"])
        return (d, item["da"]+item["db"],
                -item["cycles"][0], -item["cycles"][1],
                -(item["k"]-4*G.q))

    finalists = []
    for item in pool:
        item["da"] = _kernel_probe(item["ar"], 3*G.q, rng, 16)
        item["db"] = _kernel_probe(item["br"], 3*G.q, rng, 16)
        if min(item["da"], item["db"]) <= 4:
            continue
        finalists.append(item)
    finalists.sort(key=merit, reverse=True)
    finalists = finalists[:4]

    # Longer rescreening lowers optimistic component readings before
    # choosing a winner. These are witnesses, not distance lower bounds.
    for item in finalists:
        item["da"] = min(item["da"],
                         _kernel_probe(item["ar"], 3*G.q, rng, 48))
        item["db"] = min(item["db"],
                         _kernel_probe(item["br"], 3*G.q, rng, 48))
    finalists.sort(key=merit, reverse=True)

    for item in finalists:
        hx, hz, n = item["hx"], item["hz"], item["n"]
        if min(item["da"], item["db"]) <= 4:
            continue
        if _small_logical(hz, hx, n):
            continue
        if _small_logical(hx, hz, n):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified lift failed CSS check")
        return {"A": item["A"], "B": item["B"]}

    raise ValueError("no target-rate finalist without weight<=4 logicals")


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell without further label changes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Excluding involution-separated cell terms prevents sector-1 minimum witnesses; all five measured Dic11 parent codes still had such witnesses at weights 25-26.
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
