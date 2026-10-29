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
CONCEPT = ("On D14 = Z7:Z2(s=6), q=14, shapes (6,2)x(6,2), balanced degree-three component columns selected against short dependencies will reduce worse-basis circuit failures despite a higher CNOT count.")
GROUP = ("Frozen L1 group. The D14 bed scores 3.4511 and all five measured codes have sector-2 weight-four logical witnesses; this motivates testing a filling correction without changing the group.")
SHAPES = ("Frozen (6,2)x(6,2): n=560, guaranteed k>=224 and rate 0.400; q=14 is the largest admissible lift. Both transposed components have 84 variables and 28 checks. Parent failures span Arow/Brow blocks, with only about 1.1x check enrichments including Acol1/Brow0, rather than one dominant block.")
FILLING = ("Each factor has three (1,2) and three (2,1) rows, giving row sums 3, column sums 9 and CSS weight 12. A deterministic matching catalogue uses disjoint relative labels, verified lifted components, full-rank filtering, and weight-four/cycle-four/weight-six screening. Distinct invariant factor representatives are paired deliberately across slots. This tests suppression of widespread parent logical failures against the increased 24192-CNOT circuit cost; the screen is not the fitness.")
FAMILY_NAME = "r040_m7r2s6_62x62_match3_k4walk"

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
    """Pure deterministic catalogue search, followed by slot-specific pairing.

    Singleton cells are gauge-normalized to identity. Relative cross-block
    labels are disjoint; double cells contain two elements of one coset.
    The catalogue changes matchings, not just labels or row permutations.
    """
    import random as _random

    q = G.q
    search = _random.Random(804731921)
    mask = (1 << q) - 1
    n1 = A_SHAPE[1] * B_SHAPE[1] * q
    seen_cells = set()
    catalogue = {}

    def component_columns(C):
        # Extract A^T from the VERIFIED HZ, at Brow=0. No custom lift.
        _, hz, _ = lifted_matrices(G, C, C)
        columns = [0] * (6 * q)
        for j in range(2):
            for r in range(q):
                row = hz[(j * 6) * q + r]
                syndrome_bit = 1 << (j * q + r)
                for i in range(6):
                    support = (row >> (n1 + i * 6 * q)) & mask
                    while support:
                        bit = support & -support
                        columns[i * q + bit.bit_length() - 1] |= syndrome_bit
                        support ^= bit
        return columns

    def inspect(C, key):
        columns = component_columns(C)
        m = len(columns)
        if len(set(columns)) != m or gf2_rank(columns) != 2 * q:
            return None

        # With distinct columns, equal pair syndromes count each
        # weight-four dependency three times.
        pairs = {}
        collisions = 0
        for a in range(m):
            for b in range(a + 1, m):
                s = columns[a] ^ columns[b]
                old = pairs.get(s, 0)
                collisions += old
                pairs[s] = old + 1
        four = collisions // 3

        # Weighted check-projection graph. Its rooted walk statistics
        # and entry distributions are invariant under graph relabeling.
        size = 2 * q
        adj = [[0] * size for _ in range(size)]
        for column in columns:
            vertices = []
            s = column
            while s:
                bit = s & -s
                vertices.append(bit.bit_length() - 1)
                s ^= bit
            assert len(vertices) == 3
            for a in range(3):
                for b in range(a + 1, 3):
                    u, v = vertices[a], vertices[b]
                    adj[u][v] += 1
                    adj[v][u] += 1
        c4 = sum(adj[u][v] * (adj[u][v] - 1) // 2
                 for u in range(size) for v in range(u + 1, size))
        neighbours = [
            [(j, value) for j, value in enumerate(row) if value]
            for row in adj
        ]
        square = [[0] * size for _ in range(size)]
        for u in range(size):
            for v, uv in neighbours[u]:
                for w, vw in neighbours[v]:
                    square[u][w] += uv * vw
        roots = []
        for u in range(size):
            triangle_walks = sum(square[u][v] * value
                                 for v, value in neighbours[u])
            roots.append((square[u][u], triangle_walks,
                          sum(value * value for value in square[u]),
                          tuple(sorted(square[u]))))
        fingerprint = (four, tuple(sorted(roots)))
        return {
            "cells": C, "columns": columns, "four": four, "c4": c4,
            "walk": sum(root[1] for root in roots),
            "fingerprint": fingerprint, "key": key,
        }

    for _ in range(512):
        relative_pairs = []
        # Leave one unused relative label in each coset. Six disjoint
        # pairs consume the other twelve labels.
        for parity in (0, 1):
            residues = list(range(1, 7))
            search.shuffle(residues)
            for p in range(0, 6, 2):
                relative_pairs.append([
                    (residues[p], parity), (residues[p + 1], parity)
                ])
        search.shuffle(relative_pairs)

        left, right = [], []
        for p, pair in enumerate(relative_pairs):
            if p < 3:
                # For singleton column 0, invert relative labels to
                # obtain the double cell in column 1.
                double = sorted([G.inv(g) for g in pair], key=G.idx)
                left.append([[G.e], double])
            else:
                double = sorted(pair, key=G.idx)
                right.append([double, [G.e]])
        row_key = lambda row: tuple(tuple(G.idx(g) for g in cell)
                                    for cell in row)
        C = sorted(left, key=row_key) + sorted(right, key=row_key)
        key = tuple(row_key(row) for row in C)
        if key in seen_cells:
            continue
        seen_cells.add(key)
        record = inspect(C, key)
        if record is None:
            continue
        fp = record["fingerprint"]
        # Conservatively discard equal invariant fingerprints: retaining
        # orbit copies cannot create additional measured candidates.
        old = catalogue.get(fp)
        if old is None or key < old["key"]:
            catalogue[fp] = record

    preliminary = sorted(
        catalogue.values(),
        key=lambda r: (r["four"], r["c4"], r["walk"], r["key"])
    )[:24]

    for record in preliminary:
        columns = record["columns"]
        m = len(columns)
        triples = {}
        collisions = 0
        for a in range(m - 2):
            for b in range(a + 1, m - 1):
                ab = columns[a] ^ columns[b]
                for c in range(b + 1, m):
                    s = ab ^ columns[c]
                    old = triples.get(s, 0)
                    collisions += old
                    triples[s] = old + 1
        # Shared-index triple collisions arise from weight-four
        # dependencies; disjoint triples count each weight-six set ten
        # times. Weight-two dependencies were excluded above.
        record["six"] = (
            collisions - 3 * (m - 4) * record["four"]
        ) // 10

    library = sorted(
        preliminary,
        key=lambda r: (r["four"], r["c4"], r["six"],
                       r["walk"], r["key"])
    )[:16]
    if len(library) < 7:
        raise ValueError("too few invariant-distinct component designs")

    pairings = []
    for a in range(len(library)):
        for b in range(a + 1, len(library)):
            x, y = library[a], library[b]
            objective = (
                x["four"] + y["four"],
                max(x["c4"], y["c4"]), x["c4"] + y["c4"],
                max(x["six"], y["six"]), x["six"] + y["six"],
                x["walk"] + y["walk"], a, b,
            )
            pairings.append((objective, a, b))
    pairings.sort()
    if idx >= len(pairings):
        raise ValueError("component pair catalogue exhausted")
    _, a, b = pairings[idx]
    A, B = library[a]["cells"], library[b]["cells"]

    # Confirm the actual joint code, including the right-action factor.
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k != 224 or not css_ok(hx, hz):
        raise ValueError("joint verified lift missed the intended rate/CSS")
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
        raise ValueError("matching profile violated support constraints")
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without further random labeling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform-label supports adequately protect this D14 (6,2)x(6,2) bed: all five measured parent codes have sector-2 weight-four witnesses and worse-basis failed-shot fractions of 0.268-0.452, despite mostly stage-1 settlement.
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
