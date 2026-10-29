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
CONCEPT = ("For metacyclic Z7:Z2(s=6), order 14, shapes (6,2)x(6,2), balanced degree-three sector-2 supports with screened component kernel multiplicities will improve circuit-level fitness above the rolled parent's 3.4511.")
GROUP = ("Frozen by L1: retain the order-14 metacyclic group. This bed scores 3.4511 in the 0.400 band; the shown order-58 alternative scores 3.2187, but its different shapes prevent attributing that difference to the group.")
SHAPES = ("Frozen (6,2)x(6,2): n=560, guaranteed k=224 and rate=0.400, with q=14 the largest admissible lift. Both transposed components have 84 variables and 28 checks. Parent minimum logicals are entirely in Arow/Brow sector 2, while Acol/Brow check enrichment is weak and diffuse.")
FILLING = ("Every row contains a monomial and a binomial, with three binomials per column: row sums 3, column sums 9, HX/HZ check weights 12 and sector-2 degrees 3 in both bases. Build a deterministic library of voltage packings and controlled relaxations; use verified lifted matrices to require full component rank and rank candidates by exact weight-four/six kernel multiplicities. Slots select different unordered component pairs, not relabelings. Target the parent's widespread weight-four sector-2 logicals and basis imbalance, including slot 26's 180/500 Z versus 116/1750 X failures. The resulting 24192 CNOTs are a noise-cost tradeoff to be tested, not a fitness prediction.")
FAMILY_NAME = "r040_m7r2s6_62x62_balanced3_kernelpairs"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # metacyclic Z_7 x| Z_2: y x y^-1 = x^6 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 7, 2, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (6, 2)               # (ra, ca)
B_SHAPE = (6, 2)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Pure deterministic library; idx chooses a different component pair.

    All supports have row sum 3 and column sum 9. Screening extracts
    components from the verified lift, never from a substitute lift.
    """
    rr = random.Random(73190427)
    q = G.q
    e = G.e
    other = [[[e], [e]] for _ in range(6)]
    n1 = 4 * q
    maskq = (1 << q) - 1

    def component_columns(T):
        # HZ restricted to Brow0 in sector 2 is the A-transpose
        # component, with its actual left-action convention.
        _, hz, _ = lifted_matrices(G, T, other)
        rows = []
        for j in range(2):
            for r in range(q):
                source = hz[(j * 6) * q + r]
                row = 0
                for i in range(6):
                    part = (source >> (n1 + i * 6 * q)) & maskq
                    row |= part << (i * q)
                rows.append(row)
        if gf2_rank(rows) != 2 * q:
            return None
        cols = [0] * (6 * q)
        for check, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << check
                row ^= bit
        if len(set(cols)) != len(cols) or any(x == 0 for x in cols):
            return None
        return cols

    def fourth_count(cols):
        # Distinct columns exclude weight two. Every weight-four
        # kernel vector has exactly three unordered pair partitions.
        seen = {}
        collisions = 0
        for a in range(len(cols)):
            for b in range(a):
                s = cols[a] ^ cols[b]
                old = seen.get(s, 0)
                collisions += old
                seen[s] = old + 1
        assert collisions % 3 == 0
        return collisions // 3

    def sixth_signature(cols, n4):
        # Equal triple syndromes have symmetric difference four or six.
        # Each weight-four support contributes 3*(84-4) collisions;
        # each weight-six support contributes ten.
        seen = {}
        collisions = 0
        hist = [0] * (2 * q + 1)
        peak = 0
        for a in range(len(cols) - 2):
            x = cols[a]
            for b in range(a + 1, len(cols) - 1):
                xy = x ^ cols[b]
                for c in range(b + 1, len(cols)):
                    s = xy ^ cols[c]
                    old = seen.get(s, 0)
                    collisions += old
                    if old:
                        hist[s.bit_count()] += old
                    count = old + 1
                    seen[s] = count
                    if count > peak:
                        peak = count
        numerator = collisions - 3 * (len(cols) - 4) * n4
        assert numerator >= 0 and numerator % 10 == 0
        n6 = numerator // 10
        return (n4, n6, peak, tuple(hist))

    def make_design(reflections_left, attempt):
        # Cross-column voltages are normalized by monomial = identity.
        # The first three rows have their binomial in column 0.
        flags_l = ([1] * reflections_left
                   + [0] * (3 - reflections_left))
        flags_r = ([1] * (3 - reflections_left)
                   + [0] * reflections_left)
        rr.shuffle(flags_l)
        rr.shuffle(flags_r)
        flags = flags_l + flags_r
        options = []
        for i in range(6):
            parity = flags[i]
            delta = i % 3 + 1
            if attempt < 144:
                pairs = [
                    [(a, parity), ((a + delta) % 7, parity)]
                    for a in range(7)
                ]
            else:
                # A small mixed-parity relaxation supplies additional
                # graph structures; exact kernel screening judges them.
                pairs = [rr.sample(G.elements, 2) for _ in range(12)]
            rr.shuffle(pairs)
            options.append(pairs)

        chosen = []
        used = set()

        def pack(i):
            if i == 6:
                return True
            for pair in options[i]:
                keys = {G._key(g) for g in pair}
                if attempt < 112 and keys & used:
                    continue
                chosen.append(pair)
                if attempt < 112:
                    used.update(keys)
                if pack(i + 1):
                    return True
                if attempt < 112:
                    used.difference_update(keys)
                chosen.pop()
            return False

        if not pack(0):
            return None
        T = []
        for i, pair in enumerate(chosen):
            if i < 3:
                T.append([list(pair), [e]])
            else:
                T.append([[e], [G.inv(g) for g in pair]])
        return T

    # Identical library in every call: these are deliberate pair
    # assignments, not twenty independent random draws.
    raw = []
    exact_seen = set()
    for reflections_left in (0, 1):
        for attempt in range(160):
            T = make_design(reflections_left, attempt)
            if T is None:
                continue
            key = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in T for cell in row
            )
            if key in exact_seen:
                continue
            exact_seen.add(key)
            cols = component_columns(T)
            if cols is None:
                continue
            n4 = fourth_count(cols)
            raw.append((n4, key, T, cols))

    if not raw:
        raise ValueError("no full-rank balanced component found")
    raw.sort(key=lambda item: (item[0], item[1]))

    # Keep different short-kernel/check-syndrome signatures rather than
    # filling the library with voltage relabelings of one component.
    by_signature = {}
    examined = 0
    for n4, key, T, cols in raw:
        sig = sixth_signature(cols, n4)
        if sig not in by_signature:
            by_signature[sig] = (sig, key, T)
        examined += 1
        if examined >= 48 and len(by_signature) >= 12:
            break
        if examined >= 96:
            break
    library = sorted(
        by_signature.values(),
        key=lambda item: (item[0][0], item[0][1],
                          item[0][2], item[0][3], item[1])
    )[:12]

    pairings = []
    for a in range(len(library)):
        for b in range(a, len(library)):
            sa, sb = library[a][0], library[b][0]
            # Protect both bases, rather than allowing one strong
            # component to conceal a weak partner.
            cost = (
                sa[0] + sb[0], max(sa[0], sb[0]),
                max(sa[1], sb[1]), sa[1] + sb[1],
                max(sa[2], sb[2]), sa, sb
            )
            pairings.append((cost, a, b))
    pairings.sort()

    if idx >= len(pairings):
        raise ValueError("insufficient different screened component pairs")
    _, a, b = pairings[idx]
    A = library[a][2]
    # Inversion converts the screened left-action component into the
    # corresponding right-action B component under group inversion.
    B = [
        [[G.inv(g) for g in cell] for cell in row]
        for row in library[b][2]
    ]

    # Verify the complete code, including the actual B convention.
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k != 224 or not css_ok(hx, hz):
        raise ValueError("complete-code rank or CSS verification failed")
    return {"cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh copy of the already-screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (6,2) shapes alone ensure balanced basis protection; parent slot 26 had 180/500 Z failures versus 116/1750 X failures.
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
