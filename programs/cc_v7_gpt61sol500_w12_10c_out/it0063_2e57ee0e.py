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
CONCEPT = ("On dihedral Z29:Z2(s=28), order 58, shapes (1,3)x(1,3), one rotation-only trinomial per side plus direct suppression of component weight-eight obstructions should raise the parent's five measured distance readings above eight.")
GROUP = ("Frozen under L1. This bed scores 25.6 with all five measured d30s=8. Its verified noncommutative lifts permit replacing an all-mixed filling while retaining the same order 58 and n=580; the proposed distance improvement remains a hypothesis.")
SHAPES = ("Frozen (1,3)x(1,3): n=580, bound k>=232 and guaranteed rate 0.400, with equal 174-variable/58-check components. Parent minimum witnesses were entirely in sector-1 Acol/Bcol blocks, prominently Acol1 and Bcol1; screen both component kernels rather than only one side.")
FILLING = ("Every cell still has three distinct terms, so both check weights are 12. Each side now has one rotation-only trinomial and two mixed trinomials. Twenty prescribed pure-triple/gap cards, extended by relative-gap changes for additional slots, optimize reflection offsets. Verified component lifts exclude weight-two/four kernels, exhaustively search through weight six, and rank bounded symmetry-anchored weight-eight enumerations with their completion flags. Complete component screening is not a quantum-distance certificate; retain a legal fallback and check full CSS ranks and weight-four kernels.")
FAMILY_NAME = "r040f_m29r2s28_13x13_pureblock_anchored8"

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
    """Optimize new quotient profiles, including weight-eight obstructions."""
    q = G.q
    m = q // 2
    mask_q = (1 << q) - 1
    width = 3 * q

    # Five pure-triple geometries times four relative-gap profiles.
    # These constraints change support geometry, not just its translate.
    geometries = (2, 3, 4, 5, 8)
    profiles = (
        ((3, 7), (11, 12)),
        ((5, 11), (12, 2)),
        ((8, 12), (3, 13)),
        ((11, 6), (6, 1)),
    )
    card = idx % 20
    geometry, profile = divmod(card, 4)
    lane = idx // 20
    ga, gb = profiles[profile]
    ga, gb = list(ga), list(gb)

    def shifted_gap(gap, shift, other):
        value = (gap + shift) % m
        while value == 0 or value in (other, (-other) % m):
            value = (value + 1) % m
        return value

    # Additional slots alter relative geometry on only one side.
    # Applying one common automorphism to both sides is not the design.
    if lane:
        if lane % 2:
            ga[1] = shifted_gap(ga[1], 3 * lane, ga[0])
        else:
            gb[1] = shifted_gap(gb[1], 5 * lane, gb[0])
    ta = geometries[geometry]
    tb = geometries[(geometry + profile + 1 + lane) % 5]

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            check_bit = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= check_bit
                row ^= bit
        return out

    def free_through_four(cols):
        if (0 in cols or len(set(cols)) != len(cols)
                or any(c.bit_count() % 2 != 1 for c in cols)):
            return False
        seen = set()
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def cycles(rows):
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def component(grid, side):
        # Both actions come exclusively from the verified construction.
        hx, hz, n = lifted_matrices(G, grid, grid)
        if side == "A":
            rows = [
                sum(((row >> (3 * j * q)) & mask_q) << (j * q)
                    for j in range(3))
                for row in hx[:q]
            ]
        else:
            rows = [row & ((1 << width) - 1) for row in hz[:q]]
        return rows, columns(rows, width)

    def anchored_words(rows, cols, maximum, hit_cap, node_cap):
        """Enumerate short component words, with explicit completion flag.

        Translate a support element to 0,q,2q using the component's
        regular group action. These three anchors cover every orbit.
        At each odd check, branch on the smallest remaining support
        variable there, permanently excluding earlier alternatives.
        This avoids duplicate completions for an individual anchor.

        Exhaustion without hits is exact. A node/hit cutoff is only
        a bounded witness search, never an exclusion certificate.
        """
        hits = 0
        nodes = 0
        complete = True
        stop = False

        def visit(syndrome, word, banned, weight):
            nonlocal hits, nodes, complete, stop
            if stop:
                return
            if node_cap is not None and nodes >= node_cap:
                complete = False
                stop = True
                return
            nodes += 1
            if syndrome == 0:
                hits += 1
                if hits >= hit_cap:
                    complete = False
                    stop = True
                return

            remaining = maximum - weight
            if remaining <= 0 or syndrome.bit_count() > 3 * remaining:
                return

            # Choose the odd check with the fewest available variables.
            check_set = syndrome
            choices = None
            least = width + 1
            while check_set:
                check = check_set & -check_set
                options = rows[check.bit_length() - 1] & ~banned
                count = options.bit_count()
                if count == 0:
                    return
                if count < least:
                    choices, least = options, count
                    if count == 1:
                        break
                check_set ^= check

            # A completion must use at least one variable at this check.
            # In each branch it cannot use any earlier alternative.
            excluded = banned
            while choices and not stop:
                bit = choices & -choices
                choices ^= bit
                excluded |= bit
                v = bit.bit_length() - 1
                visit(syndrome ^ cols[v], word | bit,
                      excluded, weight + 1)

        for anchor in (0, q, 2 * q):
            if stop:
                break
            bit = 1 << anchor
            visit(cols[anchor], bit, bit, 1)
        return hits, complete, nodes

    def optimize_side(side, pure_t, gaps):
        pool = []
        fallback = None
        fallback_key = None
        seen = set()

        for attempt in range(72):
            offsets = (rng.randrange(m), rng.randrange(m))
            if offsets in seen:
                continue
            seen.add(offsets)
            grid = [[
                [(0, 0), (1, 0), (pure_t, 0)],
                [(0, 0), (gaps[0], 0), (offsets[0], 1)],
                [(0, 0), (gaps[1], 0), (offsets[1], 1)],
            ]]
            rows, cols = component(grid, side)
            c4 = cycles(rows)
            if fallback_key is None or c4 < fallback_key:
                fallback, fallback_key = grid, c4
            if not free_through_four(cols):
                continue
            pool.append((c4, attempt, grid, rows, cols))
            pool.sort(key=lambda item: item[:2])
            if len(pool) > 10:
                pool.pop()

        if not pool:
            return fallback, (2, 1, 1, 0, fallback_key)

        chosen = pool[0][2]
        chosen_key = (1, 1, 1, 0, pool[0][0])
        eight_tests = 0
        for c4, attempt, grid, rows, cols in pool:
            # No budget cutoff here: no hits exhaustively excludes <=6.
            # Stop immediately if a shorter obstruction is found.
            six_hits, six_complete, _ = anchored_words(
                rows, cols, 6, hit_cap=1, node_cap=None)
            if six_hits:
                continue
            assert six_complete

            eight_hits, eight_complete, nodes = anchored_words(
                rows, cols, 8, hit_cap=512, node_cap=90000)
            key = (0, int(eight_hits != 0), int(not eight_complete),
                   eight_hits, c4)
            if key < chosen_key:
                chosen, chosen_key = grid, key
            eight_tests += 1
            if eight_complete and eight_hits == 0:
                break
            if eight_tests >= 6:
                break

        return chosen, chosen_key

    A, score_a = optimize_side("A", ta, ga)
    B, score_b = optimize_side("B", tb, gb)
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert 232 <= k < 247
    assert css_ok(hx, hz)
    xc, zc = columns(hx, n), columns(hz, n)
    full_four_screen = (free_through_four(xc)
                        and free_through_four(zc))

    # Every profile is legal even if a bounded search finds no preferred
    # filling. Final distance is determined by the evaluator.
    return {"idx": idx, "A": A, "B": B, "k": k,
            "screened": full_four_screen,
            "selection": (score_a, score_b)}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional random filling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Parent evidence: rolled supports plus uniform labels did not produce a measured top-five distance above 4 on D29, q=58, (1,3)x(1,3); all five readings were 4 with k=232.
# REFUTED: Weight-two/four exclusion plus minimizing component weight-six collisions was sufficient to exceed eight; all five measured current-parent codes returned d30s=8, with thousands of weight-eight witnesses.
# REFUTED: Different reflection offsets and prescribed gap partitions alone ensured distinct designs; the current parent produced only 33 distinct codes from 60 slots, with the remainder collapsed as isomorphic copies.
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
