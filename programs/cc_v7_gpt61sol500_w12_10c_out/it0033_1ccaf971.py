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
CONCEPT = ("C13 semidirect C3 with action 3, order 39, shapes (2,3)x(2,3): matched order-13 binomials, complementary mixed-coset cells, and robust component-kernel screening should raise five-code fitness above the parent's 41.9294 in cell 0.100.")
GROUP = ("Frozen for L1. This exact order-39 bed has measured fitness 44.7624, versus this parent's 41.9294. Same-coset ratios have order 13 and different-coset ratios have order 3; controlling these local cycle lengths is a filling change, not a group change.")
SHAPES = ("Frozen equal (2,3) sides give length-117 components with 78 checks, n=507, and guaranteed k>=39, rate>=1/13. The size window permits lifts up to order 46, but order 39 is retained. Parent measured minimum logicals occupied only sector-1 Acol/Bcol blocks; screen both component kernels and keep exact k<=46.")
FILLING = ("Enumerate legal support pairs distinguished by combined Tanner degree fingerprints. Put same-coset doubletons on row/column matchings and mix cosets in complementary cells. Reject explicit weight-13 component orbit words, optimize actual kernel witnesses with randomized elimination and basis reduction, independently refine finalists, and retain exact weight<=4 logical screening. Kernel readings are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r010_C13sdC3_23x23_order13_matching_kernel_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # x^13=y^3=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    els = [(a, b) for b in range(3) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    def mul(g, h):
        return ((g[0] + pow(3, g[1] % 3, 13)*h[0]) % 13,
                (g[1] + h[1]) % 3)

    def inv(g):
        b = (-g[1]) % 3
        return ((-pow(3, b, 13)*g[0]) % 13, b)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _cycle_counts(rows):
    """Four/six cycles of a binary component Tanner graph."""
    size = len(rows)
    overlap = [[0]*size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i):
            t = (rows[i] & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = t
            four += t*(t-1)//2
    six = 0
    for i in range(size):
        for j in range(i):
            u = overlap[i][j]
            if not u:
                continue
            common = rows[i] & rows[j]
            for k in range(j):
                v, w = overlap[j][k], overlap[i][k]
                if v and w:
                    t = (common & rows[k]).bit_count()
                    six += u*v*w - t*(u+v+w-2)
    return four, six


def _component_objective(G, A, B, design):
    # Extract components only from the scaffold's verified matrices.
    hx, hz, n = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q)-1
    ar = []
    for i in range(2):
        for r in range(q):
            row = hx[i*3*q+r]
            ar.append(sum(((row >> (j*3*q)) & mask) << (j*q)
                          for j in range(3)))
    br = [row & ((1 << (3*q))-1) for row in hz[:2*q]]
    a4, a6 = _cycle_counts(ar)
    b4, b6 = _cycle_counts(br)
    # Later slots also explore different A/B priorities and cycle tradeoffs.
    priority = design % 3
    wa, wb = ((1, 1), (2, 1), (1, 2))[priority]
    six_weight = (1, 2, 4)[(design//3) % 3]
    score = (1000*max(a4, b4) + 200*(wa*a4+wb*b4)
             + six_weight*(wa*a6+wb*b6))
    return score, hx, hz, n


def _row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]
    return basis


def _in_span(v, basis):
    while v:
        p = v.bit_length()-1
        if p not in basis:
            return False
        v ^= basis[p]
    return True


def _small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1, 2, 3 or 4."""
    basis = _row_basis(stabilizers)
    syndromes = [0]*n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndromes[low.bit_length()-1] |= bit
            row ^= low
    singles = {}
    for i, s in enumerate(syndromes):
        v = 1 << i
        if s == 0 and not _in_span(v, basis):
            return True
        if s in singles:
            if not _in_span(v ^ singles[s], basis):
                return True
        else:
            singles[s] = v
    pairs = {}
    for i in range(n):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            s = si ^ syndromes[j]
            v = vi | (1 << j)
            if s == 0 and not _in_span(v, basis):
                return True
            if s in singles and not _in_span(v ^ singles[s], basis):
                return True
            if s in pairs:
                if not _in_span(v ^ pairs[s], basis):
                    return True
            else:
                pairs[s] = v
    return False


def NEW_CTX(rng, G, idx):
    """Degree-distinguished supports and order-controlled kernel optimization."""
    from itertools import product, permutations, combinations

    # Pure, deterministic enumeration: no cross-slot cache or mutable state.
    perms = tuple(permutations(range(3)))
    profiles = []
    for t in product(range(1, 5), repeat=6):
        rows = (sum(t[:3]), sum(t[3:]))
        cols = tuple(t[j] + t[j+3] for j in range(3))
        if min(rows) < 5 or max(rows) > 8:
            continue
        if min(cols) < 3 or max(cols) > 6:
            continue
        images = (
            tuple(t[3*r+c] for r in order for c in perm)
            for order in ((0, 1), (1, 0)) for perm in perms
        )
        if t != min(images):
            continue
        profiles.append((t, rows, cols))

    pairs = []
    for a, (ta, ar, ac) in enumerate(profiles):
        for tb, br, bc in profiles[a:]:
            if max(ar) + max(bc) > 12:
                continue
            if max(br) + max(ac) > 12:
                continue
            # These are degree multisets of the complete binary Tanner
            # graph, repeated q times. Unequal fingerprints exclude isomorphism.
            checks = tuple(sorted([x+y for x in ar for y in bc] +
                                  [x+y for x in br for y in ac]))
            qubits = tuple(sorted([x+y for x in ac for y in bc] +
                                  [x+y for x in ar for y in br]))
            priority = (
                min(ar+br), min(ac+bc), sum(ta)+sum(tb),
                -max(ar+br), -max(ac+bc),
                -sum(x == 1 for x in ta+tb),
            )
            pairs.append((priority, (checks, qubits), ta, tb))
    pairs.sort(reverse=True)
    designs, fingerprints = [], set()
    for _, fingerprint, ta, tb in pairs:
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            designs.append((ta, tb))
    if len(designs) < 20:
        raise ValueError("insufficient degree-distinguished support designs")

    ta, tb = designs[idx % len(designs)]
    supports = {
        "A": (ta[:3], ta[3:]),
        "B": (tb[:3], tb[3:]),
    }
    assert _grid_weights(supports["A"], supports["B"]) <= 12
    assert _no_dead_lines(supports["A"], supports["B"])

    def matching(side):
        positions = [(i, j) for i in range(2) for j in range(3)
                     if supports[side][i][j] == 2]
        choices = []
        for count in range(3):
            for subset in combinations(positions, count):
                if len({i for i, j in subset}) != count:
                    continue
                if len({j for i, j in subset}) != count:
                    continue
                choices.append(subset)
        largest = max(map(len, choices))
        choices = [s for s in choices if len(s) == largest]
        return set(choices[(idx + (side == "B")) % len(choices)])

    long_pairs = {side: matching(side) for side in ("A", "B")}

    def draw_cell(side, i, j):
        weight = supports[side][i][j]
        b = rng.randrange(3)
        delta = 1 + rng.randrange(2)
        if weight == 1:
            bs = [b]
        elif weight == 2:
            # Relative voltage order 13 versus order 3.
            bs = [b, b] if (i, j) in long_pairs[side] else [
                b, (b+delta) % 3]
        elif weight == 3:
            if (idx + i + 2*j + (side == "B")) % 3 == 0:
                bs = [b, (b+1) % 3, (b+2) % 3]
            else:
                bs = [b, b, (b+delta) % 3]
        else:
            if (idx+i+j) % 2:
                bs = [b, b, (b+delta) % 3, (b+delta) % 3]
            else:
                bs = [b, b, (b+1) % 3, (b+2) % 3]
        return [(a, c) for c in range(3)
                for a in rng.sample(range(13), bs.count(c))]

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(3)]
                for i in range(2)]

    def columns(rows):
        out = [0] * (3*G.q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def four_cycles(cols):
        seen, total = {}, 0
        for syndrome in cols:
            active = []
            while syndrome:
                bit = syndrome & -syndrome
                active.append(bit.bit_length()-1)
                syndrome ^= bit
            for a, x in enumerate(active):
                for y in active[:a]:
                    key = (y, x)
                    old = seen.get(key, 0)
                    total += old
                    seen[key] = old + 1
        return total

    def has_orbit_word(cols):
        # Exact classical-kernel test for every one-block, one-coset
        # norm over the normal C13 subgroup. Such a word has weight 13.
        for j in range(3):
            for b in range(3):
                syndrome = 0
                for g in G.elements:
                    if g[1] == b:
                        syndrome ^= cols[j*G.q + G.idx(g)]
                if syndrome == 0:
                    return True
        return False

    def probe(cols, trials):
        """Actual nonzero classical kernel words, never distance proofs."""
        width = len(cols)
        best = width + 1
        order = list(range(width))
        for _ in range(trials):
            rng.shuffle(order)
            pivots, words = {}, []
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
                if not syndrome:
                    words.append(word)
                    best = min(best, word.bit_count())

            # Invertible elementary basis operations, followed by pair
            # witnesses. Every retained vector remains in the true kernel.
            for sweep in range(2):
                rng.shuffle(words)
                changed = False
                for a in range(len(words)):
                    v, weight = words[a], words[a].bit_count()
                    for b, u in enumerate(words):
                        if a == b:
                            continue
                        z = v ^ u
                        zw = z.bit_count()
                        best = min(best, zw)
                        if 0 < zw < weight:
                            v, weight = z, zw
                            changed = True
                    words[a] = v
                    best = min(best, weight)
                if not changed:
                    break
            if best <= 18:
                break
        return best

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -max(item["cycles"]), -sum(item["cycles"]))

    pool, seen = [], set()

    def admit(A, B):
        key = tuple(tuple(sorted(cell)) for grid in (A, B)
                    for row in grid for cell in row)
        if key in seen:
            return
        seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 39 <= k <= 46:
            return
        q, mask = G.q, (1 << G.q)-1
        ar = []
        for i in range(2):
            for r in range(q):
                row = hx[i*3*q+r]
                ar.append(sum(((row >> (j*3*q)) & mask) << (j*q)
                              for j in range(3)))
        br = [row & ((1 << (3*q))-1) for row in hz[:2*q]]
        ac, bc = columns(ar), columns(br)
        if has_orbit_word(ac) or has_orbit_word(bc):
            return
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ac": ac, "bc": bc,
            "da": probe(ac, 4), "db": probe(bc, 4),
            "cycles": (four_cycles(ac), four_cycles(bc)),
        }
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[8:]

    for _ in range(40):
        admit(draw_grid("A"), draw_grid("B"))

    # Optimize kernel witnesses rather than merely moving downhill in
    # cycle count. Redraw cells on the weaker component preferentially.
    for step in range(24):
        if not pool:
            admit(draw_grid("A"), draw_grid("B"))
            continue
        parent = pool[rng.randrange(min(4, len(pool)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        if step % 4 == 0:
            side = "A" if rng.randrange(2) == 0 else "B"
        else:
            side = "A" if parent["da"] <= parent["db"] else "B"
        grid = A if side == "A" else B
        i, j = rng.randrange(2), rng.randrange(3)
        grid[i][j] = draw_cell(side, i, j)
        admit(A, B)

    if not pool:
        raise ValueError("no rank-legal candidate without a weight-13 orbit word")

    # Independent repeated searches reduce selection of lucky estimates.
    for item in pool:
        item["da"] = min(item["da"], probe(item["ac"], 24))
        item["db"] = min(item["db"], probe(item["bc"], 24))
    pool.sort(key=merit, reverse=True)
    for item in pool[:4]:
        item["da"] = min(item["da"], probe(item["ac"], 80))
        item["db"] = min(item["db"], probe(item["bc"], 80))
    pool.sort(key=merit, reverse=True)

    for item in pool:
        hx, hz, n = item["hx"], item["hz"], item["n"]
        if _small_logical(hz, hx, n) or _small_logical(hx, hz, n):
            continue
        assert 8*item["k"] < n
        assert css_ok(hx, hz)
        return {"A": item["A"], "B": item["B"]}
    raise ValueError("all finalists contain a weight<=4 logical")


def CELL(rng, G, side, i, j, ctx):
    """Return the already optimized full cell; no call-order dependence."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A large dihedral lift with legal rolled supports suffices for d30s>4: all five measured parent codes instead have d30s=4, with sector-1 witnesses.
# REFUTED: Rank surplus reliably compensates for weaker component supports: current-parent slots 12/13/33 have k=48 but d2s=12/14/6, while rank-bound slots 3/42 have k=39 and d30s=24.
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
