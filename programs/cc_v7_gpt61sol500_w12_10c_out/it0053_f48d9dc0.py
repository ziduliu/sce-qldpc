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
CONCEPT = ("For metacyclic Z7:Z4(s=6), order 28, shapes (1,3)x(2,6), balanced cubic components using order-four odd-coset labels should yield five measured distance readings above eight at rate 0.400; component screening does not certify quantum distance.")
GROUP = ("Retain the cyclic-by-cyclic semidirect construction rule, resizing m,r,s to 7,4,6 because the new size factor excludes q=58. Order 28 gives n=560. Odd-coset elements have order four rather than the parent's order-two reflections; testing this central double cover is motivated by the parent's five d30s=8 readings, not by a measured advantage on this untried bed.")
SHAPES = ("Target 0.400 with (1,3)x(2,6): n=20*28=560, kb=8*28=224. Components have 84 variables/28 checks and 168 variables/56 checks. Parent weight-eight witnesses occupied sector-1 Acol/Bcol blocks. A second B check row changes those column relations rather than transposing the parent; verified full component ranks enforce k=224.")
FILLING = ("Twenty cards combine four mixed-coset trinomial models with five cubic B support profiles. A row total is nine; B column totals are three and B row totals are nine, so both quantum check weights are twelve. Avoid central-involution differences within cells. Extract components from verified lifts, require full rank, prefer exact absence of weight-two/four/six kernels, and rank genuine sampled kernel witnesses with deeper finalist rescreening. Check final full matrices for weight-two/four kernels and retain a legal full-rank fallback.")
FAMILY_NAME = "r040_m7r4s6_13x26_order4_cubic_spectrum"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Same metacyclic construction: y x y^-1 = x^6.
    # 6^4 = 1 mod 7; odd y-cosets have order four.
    m, r, s = 7, 4, 6
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
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize twenty structural cards, using only verified lifts."""
    q = G.q
    card = idx % 20
    mode, profile = divmod(card, 5)
    top_profiles = (
        (2, 2, 2, 1, 1, 1),
        (3, 2, 1, 1, 1, 1),
        (3, 2, 2, 1, 1, 0),
        (3, 3, 1, 1, 1, 0),
        (3, 3, 2, 1, 0, 0),
    )
    patterns = ((0, 0, 1), (0, 1, 2),
                (0, 1, 3), (0, 1, 1))
    a_models = ((0, 0, 0), (0, 1, 2),
                (1, 1, 1), (2, 2, 3))
    top = top_profiles[profile]
    supports = {
        "A": [[3, 3, 3]],
        "B": [list(top), [3 - t for t in top]],
    }
    assert _grid_weights(supports["A"], supports["B"]) == 12
    assert _no_dead_lines(supports["A"], supports["B"])
    pools = [[g for g in G.elements if g[1] == b]
             for b in range(4)]

    def filled_cell(size, pattern):
        if not size:
            return []
        for _ in range(64):
            if size == 1:
                return [G.pick(rng)]
            shift = rng.randrange(4)
            bs = [(b + shift) % 4 for b in pattern]
            cell = []
            for b in range(4):
                cell.extend(rng.sample(pools[b], bs.count(b)))
            # Avoid the unique central involution as a relative label.
            if all(G.mul(G.inv(cell[u]), cell[v]) != (0, 2)
                   for u in range(size) for v in range(u)):
                return cell
        raise ValueError("could not avoid central-involution difference")

    def make_side(side):
        result = []
        for i, row in enumerate(supports[side]):
            out = []
            for j, size in enumerate(row):
                if size == 3:
                    p = (a_models[mode][j] if side == "A"
                         else (mode + j + 2 * i) % 4)
                    pattern = patterns[p]
                elif size == 2:
                    # Relative orders seven, four, or fourteen.
                    pattern = ((0, 0), (0, 1), (0, 2))[
                        (mode + j + i + idx // 20) % 3]
                else:
                    pattern = (0,) if size else ()
                out.append(filled_cell(size, pattern))
            result.append(out)
        return result

    def columns(rows, length):
        out = [0] * length
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def component_rows(M, side):
        # Extract actual left/right component actions from the scaffold.
        da = [[[G.e] for _ in range(3)]]
        db = [[[G.e] for _ in range(6)] for _ in range(2)]
        A, B = (M, db) if side == "A" else (da, M)
        hx, hz, _ = lifted_matrices(G, A, B)
        if side == "A":
            mask = (1 << q) - 1
            return [sum(((row >> (j * 6 * q)) & mask) << (j * q)
                        for j in range(3)) for row in hx[:q]]
        mask = (1 << (6 * q)) - 1
        return [row & mask for row in hz[:2 * q]]

    def free_through_four(cols):
        # Cubic columns make every odd-weight kernel impossible.
        if any(c.bit_count() % 2 != 1 for c in cols):
            return False
        if len(set(cols)) != len(cols):
            return False
        seen = set()
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def six_collisions(cols, limit=None):
        # Used only after excluding weights two/four. Each six-word
        # contributes ten collisions; early termination is only pruning.
        seen = {}
        hits = 0
        for i, x in enumerate(cols):
            for j in range(i):
                xy = x ^ cols[j]
                for h in range(j):
                    s = xy ^ cols[h]
                    old = seen.get(s, 0)
                    hits += old
                    if limit is not None and hits > limit:
                        return hits
                    seen[s] = old + 1
        return hits

    def four_cycles(cols):
        total = 0
        for i, x in enumerate(cols):
            for other in cols[:i]:
                overlap = (x & other).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def kernel_probe(cols, trials):
        # Every recorded word is an actual component-kernel witness.
        # Failure to find a lighter word is not a lower-bound certificate.
        best, hits = len(cols) + 1, 0

        def record(word):
            nonlocal best, hits
            if word:
                weight = word.bit_count()
                if weight < best:
                    best, hits = weight, 1
                elif weight == best:
                    hits += 1

        order = list(range(len(cols)))
        for _ in range(trials):
            rng.shuffle(order)
            pivots, words = {}, []
            for j in order:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = (syndrome, word)
                        break
                    ps, pw = pivots[p]
                    syndrome ^= ps
                    word ^= pw
                if not syndrome:
                    words.append(word)
                    record(word)
            for i, word in enumerate(words):
                for other in words[:i]:
                    record(word ^ other)
            if len(words) >= 3:
                for _ in range(96):
                    a, b, c = rng.sample(range(len(words)), 3)
                    record(words[a] ^ words[b] ^ words[c])
        return best, hits

    def select_side(side):
        width = A_SHAPE[1] if side == "A" else B_SHAPE[1]
        checks = q if side == "A" else 2 * q
        pool = []
        for attempt in range(96):
            M = make_side(side)
            rows = component_rows(M, side)
            if gf2_rank(rows) != checks:
                continue
            cols = columns(rows, width * q)
            clean = free_through_four(cols)
            if not clean and pool and pool[0][0][0]:
                continue
            limit = None
            if clean and len(pool) >= 4 and all(p[0][0] for p in pool):
                limit = max(p[4] for p in pool)
            h6 = six_collisions(cols, limit) if clean else 10**9
            if limit is not None and h6 > limit:
                continue
            reading, hits = kernel_probe(cols, 8)
            if not clean:
                reading = min(reading, 4)
            elif h6:
                reading = min(reading, 6)
            cycles = four_cycles(cols)
            score = (int(clean), int(h6 == 0), reading,
                     -h6, -hits, -cycles)
            pool.append((score, M, cols, reading, h6, cycles))
            pool.sort(key=lambda item: item[0], reverse=True)
            del pool[4:]
            if (attempt >= 48 and len(pool) == 4
                    and all(p[4] == 0 and p[3] >= 10 for p in pool)):
                break
        if not pool:
            raise ValueError("no full-rank component for structural card")

        finalists = []
        for score, M, cols, old_reading, h6, cycles in pool:
            reading, hits = kernel_probe(cols, 40)
            reading = min(reading, old_reading)
            key = (score[0], int(h6 == 0), reading,
                   -h6, -hits, -cycles)
            finalists.append((key, M, reading, h6, cycles))
        finalists.sort(key=lambda item: item[0], reverse=True)
        return finalists[:3]

    pa, pb = select_side("A"), select_side("B")
    pairs = []
    for ka, A, da, ha, ca in pa:
        for kb, B, db, hb, cb in pb:
            key = (min(ka[0], kb[0]), min(da, db), da + db,
                   -(ha + hb), -(ca + cb))
            pairs.append((key, A, B))
    pairs.sort(key=lambda item: item[0], reverse=True)

    fallback = None
    for key, A, B in pairs:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 224:
            continue
        if fallback is None:
            fallback = (A, B, hx, hz, k, False, key)
        # Check mixed-sector kernels as well as component relations.
        if (free_through_four(columns(hx, n))
                and free_through_four(columns(hz, n))):
            fallback = (A, B, hx, hz, k, True, key)
            break
    if fallback is None:
        raise ValueError("no pair retained exact target rate")
    A, B, hx, hz, k, screened, selection = fallback
    assert css_ok(hx, hz)
    assert _grid_weights(
        [[len(c) for c in row] for row in A],
        [[len(c) for c in row] for row in B]) == 12
    return {"idx": idx, "A": A, "B": B, "k": k,
            "screened": screened, "selection": selection}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly selected cell without relabeling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Parent evidence: rolled supports plus uniform labels did not produce a measured top-five distance above 4 on D29, q=58, (1,3)x(1,3); all five readings were 4 with k=232.
# REFUTED: Reflection-offset variation makes every gap-partition slot distinct; the measured parent retained only 33 of 60 slots after isomorphism deduplication.
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
