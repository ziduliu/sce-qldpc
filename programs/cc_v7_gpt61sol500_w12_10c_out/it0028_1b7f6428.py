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
CONCEPT = ("On dihedral Z29:Z2(s=28), order 58, shapes (1,3)x(1,3), balanced mixed trinomials with six distinct reflection gaps and verified low-weight screening should raise measured distance above the parent's 4 while retaining k=232.")
GROUP = ("Frozen L1 bed: the Z29:Z2 order-58 parent scores 6.4 with weight-4 measured minima; this experiment tests whether filling, rather than the group, causes that ceiling.")
SHAPES = ("Frozen (1,3)x(1,3): n=580, guaranteed k>=232 and rate>=0.400, equal length-174 lifted component codes. Parent minima occupy sector-1 Acol/Bcol blocks, and every measured filling has singleton cells.")
FILLING = ("Use identity plus two reflections in every cell: component column weight 3 and check weight 12. Assign distinct canonical gap profiles per slot; search offsets using verified lifted matrices, prefer k=232, reject weight<=4 logicals, and minimize component-six and mixed-three-plus-three witness counts. Screening is not a distance certificate above the weights explicitly checked.")
FAMILY_NAME = "r040f_m29r2s28_13x13_gap3_screen46"

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


def NEW_CTX(rng, G, idx):
    """Select offsets within a slot-specific, non-orbit gap design."""
    from itertools import combinations

    q = G.q
    m = 29
    n1 = 9 * q

    def gap(x):
        x %= m
        return min(x, m - x)

    def triangular(t):
        a, b, c = sorted(t)
        return a + b == c or a + b + c == m

    def canonical(a, b):
        images = []
        for scale in range(1, 15):
            aa = tuple(sorted(gap(scale * x) for x in a))
            bb = tuple(sorted(gap(scale * x) for x in b))
            images.append(tuple(sorted((aa, bb))))
        return min(images)

    # Distinct gap profiles modulo common group automorphisms,
    # column permutations, and exchange of the two sides.
    designs = []
    for a in ((1, 4, 10), (1, 5, 12), (1, 6, 11)):
        remaining = [x for x in range(1, 15) if x not in a]
        for b in combinations(remaining, 3):
            if not triangular(b):
                designs.append((a, b))
    designs.sort(key=lambda z: (
        sum((i + 3) * x * x for i, x in enumerate(z[0] + z[1])) % 997,
        z))
    seen = set()
    profiles = []
    for a, b in designs:
        key = canonical(a, b)
        if key not in seen:
            seen.add(key)
            profiles.append((a, b))
    ga, gb = profiles[idx % len(profiles)]

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                low = row & -row
                out[low.bit_length() - 1] |= bit
                row ^= low
        return out

    def pair_quality(cols):
        # All these columns have odd weight. Distinct nonzero singles
        # and distinct nonzero pair syndromes exclude weights 1..4.
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        pairs = {0}
        low2 = low4 = 0
        for i, x in enumerate(cols):
            for y in cols[:i]:
                s = x ^ y
                if s in pairs:
                    return None
                pairs.add(s)
                w = s.bit_count()
                low2 += (w == 2)
                low4 += (w == 4)
        return (10 + idx % 5) * low2 + low4

    def six_collisions(cols, cap=2500):
        # Following the pair test, two equal triple syndromes have
        # disjoint supports and therefore yield a weight-six kernel word.
        syndromes = set()
        count = 0
        for i in range(len(cols)):
            x = cols[i]
            for j in range(i):
                xy = x ^ cols[j]
                for k in range(j):
                    s = xy ^ cols[k]
                    if s in syndromes:
                        count += 1
                        if count >= cap:
                            return cap
                    else:
                        syndromes.add(s)
        return count

    def basis(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        return pivots

    def remainder(v, pivots):
        while v:
            p = v.bit_length() - 1
            if p not in pivots:
                return v
            v ^= pivots[p]
        return 0

    def no_small_logical(cols, stabilizers):
        # Exact syndrome-collision test through weight four.
        # Keep one representative per syndrome; stabilizer-equivalent
        # collisions are harmless.
        representatives = {0: 0}
        for i, s in enumerate(cols):
            v = 1 << i
            if s in representatives:
                if remainder(v ^ representatives[s], stabilizers):
                    return False
            else:
                representatives[s] = v
        for i, x in enumerate(cols):
            bit = 1 << i
            for j in range(i):
                s = x ^ cols[j]
                v = bit | (1 << j)
                if s in representatives:
                    if remainder(v ^ representatives[s], stabilizers):
                        return False
                else:
                    representatives[s] = v
        return True

    def mixed_sixes(cols, stabilizers, x_checks):
        # Enumerate actual 3-sector-2 + 3-sector-1 witnesses.
        # This is a selection statistic, not an exhaustive distance search.
        block_maps = [{} for _ in range(3)]
        mask = (1 << q) - 1
        for c in range(n1):
            block = (c // q) % 3 if x_checks else (c // q) // 3
            s = (cols[c] >> (block * q)) & mask
            block_maps[block][s] = 1 << c
        tail = cols[n1:]
        found = 0
        for i in range(q):
            for j in range(i):
                xy = tail[i] ^ tail[j]
                for k in range(j):
                    s = xy ^ tail[k]
                    v = (1 << (n1 + i)) | (1 << (n1 + j)) | (1 << (n1 + k))
                    for block in range(3):
                        match = block_maps[block].get(
                            (s >> (block * q)) & mask)
                        if match is None:
                            break
                        v |= match
                    else:
                        if remainder(v, stabilizers):
                            found += 1
        return found

    amask = sum(((1 << q) - 1) << (3 * j * q) for j in range(3))
    bmask = (1 << (3 * q)) - 1
    pool = []
    fallback = None

    # Offset search, not independent uniform final fillings.
    for trial in range(96):
        centers = set()
        cells = {}
        for side, gaps in (("A", ga), ("B", gb)):
            row = []
            for d in gaps:
                u = rng.randrange(m)
                while (2 * u + d) % m in centers:
                    u = rng.randrange(m)
                centers.add((2 * u + d) % m)
                row.append([G.e, (u, 1), ((u + d) % m, 1)])
            cells[side] = [row]

        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        if fallback is None:
            fallback = cells

        # Extract component columns from the verified construction.
        aa = columns([r & amask for r in hx[:q]], n)
        ac = [aa[3 * j * q + t] for j in range(3) for t in range(q)]
        bc = columns([r & bmask for r in hz[:q]], 3 * q)
        qa = pair_quality(ac)
        qb = pair_quality(bc)
        if qa is None or qb is None:
            continue

        cheap = max(qa, qb) + (qa + qb) // 4
        pool.append((cheap, trial, cells, hx, hz, n, ac, bc))
        pool.sort(key=lambda z: z[:2])
        del pool[4:]

    best = None
    best_score = None
    for cheap, trial, cells, hx, hz, n, ac, bc in pool:
        if n - gf2_rank(hx) - gf2_rank(hz) != 232:
            continue
        bx, bz = basis(hx), basis(hz)
        sx, sz = columns(hx, n), columns(hz, n)
        if not no_small_logical(sx, bz):
            continue
        if not no_small_logical(sz, bx):
            continue

        a6, b6 = six_collisions(ac), six_collisions(bc)
        z6 = mixed_sixes(sx, bz, True)
        x6 = mixed_sixes(sz, bx, False)
        score = (max(a6, b6) // 10 + x6 + z6,
                 a6 + b6, x6 + z6, cheap, trial)
        if best_score is None or score < best_score:
            best_score, best = score, cells

    # Preserve legal candidates if an unusually difficult profile fails
    # the preferred screening criteria; the evaluator measures it as-is.
    if best is None:
        best = pool[0][2] if pool else fallback
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz)
    assert n - gf2_rank(hx) - gf2_rank(hz) > 0
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell without further randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled counts with uniform labels suffice to escape short sector-1 logicals at rate 0.400; many seed slots have d2s=2, and all five measured seed codes have d30s=4 with sector-1 minima.
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
