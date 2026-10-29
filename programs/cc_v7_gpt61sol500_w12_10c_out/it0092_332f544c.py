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
CONCEPT = ("Dic11, order 44, shapes (1,2)x(1,3): degree-distinguished supports and deeply rescreened component witnesses should produce top-five mean distance reading >=11 in cell 0.300, exceeding its current 30.8 fitness.")
GROUP = ("Keep the exact Dic11 group and order. The parent measured d30s=25..26, and this order-44 dicyclic bed leads the low-rate frontier. Transfer to higher-rate components is a hypothesis; the new size n=308 does not require an order change.")
SHAPES = ("Target cell 0.300 with A=(1,2), B=(1,3): n=7*44=308, kb=2*44=88, guaranteed rate 2/7. Component variable/check counts are 88/44 and 132/44. Parent minimum witnesses occupied sector-1 Acol/Bcol blocks, prominently Acol1/Acol2 and Bcol1/Bcol2. Verified ranks gate every emitted code to 0.275<=k/n<0.325.")
FILLING = ("Enumerate positive canonical support pairs separated by combined Tanner degree fingerprints, checking both weight sums <=12. Binomials have rotation differences of order 22; larger cells search uniform, C4-coset-spread, and mixed-coset models with involution exclusion. Optimize actual component-kernel witnesses, independently deepen finalist searches, and prefer exhaustive full commuting-vector exclusion through weight four. Witness readings remain upper-bound heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r030_dic11_12x13_order22_deep_kernel_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic11: x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    if q != 44:
        raise ValueError("Dic11 has order 44")
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((a + 11 * (b // 2)) % 22, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11*b*d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return 22*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _support_designs():
    """Canonical single-row supports, not permutation images of one grid."""
    from itertools import combinations_with_replacement
    pairs = []
    for ta in combinations_with_replacement(range(2, 6), 2):
        for tb in combinations_with_replacement(range(2, 6), 3):
            ar, br = sum(ta), sum(tb)
            if ar + max(tb) > 12 or br + max(ta) > 12:
                continue
            checks = tuple(sorted(
                [ar + b for b in tb] + [br + a for a in ta]))
            qubits = tuple(sorted(
                [a + b for a in ta for b in tb] + [ar + br]))
            # B has the higher component rate. Prefer its stronger
            # columns, then strengthen A without violating either cap.
            priority = (
                min(tb), -tb.count(2), min(ta),
                sum(tb), sum(ta), -max(tb), -max(ta))
            pairs.append((priority, (checks, qubits), ta, tb))
    pairs.sort(reverse=True)
    seen, designs = set(), []
    for _, fingerprint, ta, tb in pairs:
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        designs.append((ta, tb))
    if len(designs) < 20:
        raise ValueError("fewer than twenty degree-distinguished designs")
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


def _syndrome_columns(rows, width):
    """Transpose only rows supplied by the verified construction."""
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _kernel_probe(cols, rng, trials):
    """Actual nonzero component-kernel witnesses, not distance proofs."""
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
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        # Elementary basis operations retain genuine kernel vectors.
        # Sorting puts useful short reducers first.
        for sweep in range(2 + int(trials >= 16)):
            words.sort(key=int.bit_count)
            changed = False
            for a in range(len(words)):
                word, weight = words[a], words[a].bit_count()
                for other in words[:a]:
                    z = word ^ other
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        word, weight = z, zw
                        changed = True
                words[a] = word
                best = min(best, weight)
            if not changed:
                break
        short = sorted(words, key=int.bit_count)[:48]
        for a, word in enumerate(short):
            for other in short[:a]:
                weight = (word ^ other).bit_count()
                if weight:
                    best = min(best, weight)
        if best <= 4:
            break
    return best


def _kernel_floor(cols, through_six=False):
    """Exact smallest kernel weight through four or six.

    Returning five or seven means only that no kernel word was found
    in the exhaustively enumerated range. Stabilizers are not removed.
    """
    if 0 in cols:
        return 1
    singles = set(cols)
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(cols):
        for y in cols[:i]:
            s = x ^ y
            if s in singles:
                return 3
            if s in pairs:
                four = True
            pairs.add(s)
    if four:
        return 4
    if not through_six:
        return 5
    triples = set()
    six = False
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for h in range(j):
                s = xy ^ cols[h]
                if s in pairs:
                    return 5
                if s in triples:
                    six = True
                triples.add(s)
    return 6 if six else 7


def _component_checks(hx, hz, q):
    """Extract the actual (1,2) and (1,3) component actions."""
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
    brows = [hz[a*q+r] & bmask
             for a in range(rb) for r in range(q)]
    return arows, brows


def _label_grid(rng, G, counts, model):
    """Order-controlled binomials and three larger-cell occupancy models."""
    involution = (11, 0)
    pools = [[] for _ in range(4)]
    for g in G.elements:
        pools[(2*g[0] + g[1]) % 4].append(g)
    row = []
    for count in counts:
        if count == 2:
            # An odd rotation difference other than 11 has order 22.
            # A mixed-coset binomial would instead have a weight-four
            # single-block orbit word from its order-four difference.
            g = G.pick(rng)
            delta = rng.choice([a for a in range(1, 22, 2) if a != 11])
            row.append([g, G.mul(g, (delta, 0))])
            continue

        if model == 0:
            cell = []
            while len(cell) < count:
                g = G.pick(rng)
                if g in cell:
                    continue
                if any(G.mul(G.inv(h), g) == involution for h in cell):
                    continue
                cell.append(g)
        else:
            phase = rng.randrange(4)
            if model == 1:
                direction = rng.choice((-1, 1))
                quota = [(phase + direction*t) % 4 for t in range(count)]
            else:
                # Repeated quotient occupancy but noncommuting terms.
                opposite = (phase + rng.choice((1, 3))) % 4
                quota = [phase] * (count-1) + [opposite]
            cell = []
            for coset in quota:
                available = [
                    g for g in pools[coset] if g not in cell
                    and all(G.mul(G.inv(h), g) != involution for h in cell)
                ]
                cell.append(rng.choice(available))
        row.append(cell)
    return [row]


def NEW_CTX(rng, G, idx):
    """One support geometry, with occupancy search and independent rescreening."""
    ta, tb = _SUPPORT_DESIGNS[idx % len(_SUPPORT_DESIGNS)]
    sa, sb = [list(ta)], [list(tb)]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = G.q * (ca-ra) * (cb-rb)

    def merit(item):
        d = min(item["da"], item["db"])
        return (d, item["da"]+item["db"], -item["cycles"],
                item["k"]*d*d)

    pool, seen = [], set()

    def admit(A, B):
        signature = tuple(tuple(sorted(cell))
                          for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= kb and 275*n <= 1000*k < 325*n):
            return
        ar, br = _component_checks(hx, hz, G.q)
        ac = _syndrome_columns(ar, ca*G.q)
        bc = _syndrome_columns(br, cb*G.q)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n, "k": k,
            "ac": ac, "bc": bc,
            "da": _kernel_probe(ac, rng, 3),
            "db": _kernel_probe(bc, rng, 3),
            "cycles": _four_cycles(ar) + _four_cycles(br),
        }
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[8:]

    # Every support geometry searches all occupancy models. The B model
    # is offset so the search also tests asymmetric component structures.
    for attempt in range(72):
        ma = (idx + attempt//24) % 3
        mb = (ma + attempt % 3) % 3
        admit(_label_grid(rng, G, ta, ma),
              _label_grid(rng, G, tb, mb))

    # Local redraws target the weaker measured component, rather than
    # merely descending in Tanner four-cycle count.
    for step in range(24):
        if not pool:
            admit(_label_grid(rng, G, ta, step % 3),
                  _label_grid(rng, G, tb, (step+1) % 3))
            continue
        parent = pool[rng.randrange(min(4, len(pool)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        side = "A" if parent["da"] <= parent["db"] else "B"
        if step % 4 == 0:
            side = rng.choice(("A", "B"))
        counts = ta if side == "A" else tb
        grid = A if side == "A" else B
        j = rng.randrange(len(counts))
        grid[0][j] = _label_grid(
            rng, G, (counts[j],), (idx+step) % 3)[0][0]
        admit(A, B)

    if not pool:
        raise ValueError("no rank-legal filling in cell 0.300")

    for item in pool:
        # Exact small-word enumeration limits lucky kernel readings.
        fa = _kernel_floor(item["ac"], through_six=True)
        fb = _kernel_floor(item["bc"], through_six=True)
        if fa <= 6:
            item["da"] = min(item["da"], fa)
        if fb <= 6:
            item["db"] = min(item["db"], fb)
        item["da"] = min(item["da"], _kernel_probe(item["ac"], rng, 16))
        item["db"] = min(item["db"], _kernel_probe(item["bc"], rng, 16))

    pool.sort(key=merit, reverse=True)
    finalists = pool[:4]
    for item in finalists:
        item["da"] = min(item["da"], _kernel_probe(item["ac"], rng, 64))
        item["db"] = min(item["db"], _kernel_probe(item["bc"], rng, 64))
        xc = _syndrome_columns(item["hx"], item["n"])
        zc = _syndrome_columns(item["hz"], item["n"])
        item["floor"] = min(_kernel_floor(xc), _kernel_floor(zc))

    # This excludes every commuting vector through weight four when
    # floor==5, including stabilizers. It is a conservative screen.
    # A legal fallback remains available if no finalist passes.
    def final_merit(item):
        d = min(item["da"], item["db"])
        return (item["floor"] == 5, item["floor"], d,
                item["k"]*d*d, item["da"]+item["db"], -item["cycles"])

    winner = max(finalists, key=final_merit)
    assert 275*winner["n"] <= 1000*winner["k"] < 325*winner["n"]
    assert css_ok(winner["hx"], winner["hz"])
    return {"A": winner["A"], "B": winner["B"]}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened full cell without further label changes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
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
