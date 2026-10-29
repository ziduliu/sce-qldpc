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
CONCEPT = ("For metacyclic Z7:Z2(s=6), order 14, shapes (6,2)x(6,2), invariant-distinct pairs of screened 16-18-term components selected by gate-burden-weighted short-kernel multiplicity should improve worse-basis circuit fitness above 4.2382.")
GROUP = ("Frozen by L1: retain the order-14 metacyclic group. Filling changes already improved this bed from 3.4511 to 4.2382; this supports another controlled filling experiment, not a claim that the group is optimal.")
SHAPES = ("Frozen equal (6,2) sides: n=560, guaranteed k=224 and rate=0.400; q=14 is the largest admissible lift. Both transposed components have 84 variables and 28 checks. Parent minimum logicals occupy Arow/Brow sector 2, while Acol/Brow check enrichment and data-block enrichment are diffuse; strengthen both components globally.")
FILLING = ("Search six explicit support profiles: three 18-term degree-three topologies, two balanced 16-term degree-two/three topologies, and one balanced 16-term topology with a degree-four row. Cells include zeros and trinomials; all pairings obey both weight budgets. Verified component lifts screen full rank and weight-three/four/five kernel words, then count weight-six words exactly. Pair selection uses a cubic gate-burden discount on weight-six multiplicity, a falsifiable circuit-noise heuristic rather than fitness. Broaden coset voltages and reject complete-code graph-invariant duplicates before assigning slots. Target the parent's roughly 4-8% failed-shot fractions, numerous stage-3 shots, depth 18 and 24192 CNOTs; balanced sparse pairs use 21504 CNOTs. Verify final k=224 and CSS.")
FAMILY_NAME = "r040_m7r2s6_62x62_gateweighted_invariantpairs_v2"

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
    """Pure deterministic search; slots receive invariant-distinct pairs."""
    from hashlib import sha256

    rr = random.Random(90417263)
    q = G.q
    e = G.e
    other = [[[e]]]

    def component_columns(T):
        # A 1x1 identity partner exposes the actual A-transpose lift
        # contiguously in sector 2: no independent lift convention.
        _, hz, _ = lifted_matrices(G, T, other)
        rows = [row >> (2 * q) for row in hz]
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
        # Distinct nonzero columns exclude weights one and two.
        # A pair syndrome equal to a column detects weight three.
        singles = set(cols)
        seen = {}
        collisions = 0
        for a in range(len(cols)):
            for b in range(a):
                s = cols[a] ^ cols[b]
                if s in singles:
                    return None
                old = seen.get(s, 0)
                collisions += old
                seen[s] = old + 1
        assert collisions % 3 == 0
        return collisions // 3

    def sixth_signature(cols, n4):
        # With no weights one/two/three, a pair/triple collision is
        # disjoint and each weight-five word contributes ten partitions.
        pairs = {}
        for a in range(len(cols)):
            for b in range(a):
                s = cols[a] ^ cols[b]
                pairs[s] = pairs.get(s, 0) + 1
        seen = {}
        collisions = 0
        fifth = 0
        hist = [0] * (2 * q + 1)
        peak = 0
        for a in range(len(cols) - 2):
            x = cols[a]
            for b in range(a + 1, len(cols) - 1):
                xy = x ^ cols[b]
                for c in range(b + 1, len(cols)):
                    s = xy ^ cols[c]
                    fifth += pairs.get(s, 0)
                    old = seen.get(s, 0)
                    collisions += old
                    if old:
                        hist[s.bit_count()] += old
                    count = old + 1
                    seen[s] = count
                    if count > peak:
                        peak = count
        numerator = collisions - 3 * (len(cols) - 4) * n4
        assert fifth % 10 == 0
        assert numerator >= 0 and numerator % 10 == 0
        n5, n6 = fifth // 10, numerator // 10
        four_cycles = 0
        for a, x in enumerate(cols):
            for y in cols[:a]:
                overlap = (x & y).bit_count()
                four_cycles += overlap * (overlap - 1) // 2
        return (n4, n5, n6, four_cycles, peak, tuple(hist))

    profiles = (
        ((2, 1), (2, 1), (2, 1), (1, 2), (1, 2), (1, 2)),
        ((0, 3), (1, 2), (2, 1), (2, 1), (2, 1), (2, 1)),
        ((0, 3), (1, 2), (1, 2), (2, 1), (2, 1), (3, 0)),
        ((1, 1), (1, 1), (2, 1), (2, 1), (1, 2), (1, 2)),
        ((2, 0), (0, 2), (2, 1), (2, 1), (1, 2), (1, 2)),
        ((1, 1), (1, 1), (1, 1), (2, 1), (1, 2), (2, 2)),
    )

    def make_design(layout, attempt):
        counts = list(profiles[layout])
        rr.shuffle(counts)  # Search freedom, not a separate design.
        if rr.randrange(2):
            counts = [(b, a) for a, b in counts]
        T = []
        for sizes in counts:
            row = []
            for weight in sizes:
                if weight == 0:
                    cell = []
                elif weight == 1:
                    cell = [G.pick(rr)]
                elif weight == 2 and attempt % 3:
                    # Retain long-order binomials in two-thirds of
                    # trials; the other third explores both cosets.
                    g = G.pick(rr)
                    cell = [g, G.mul(g, (rr.randrange(1, 7), 0))]
                elif weight == 3 and attempt % 3 == 0:
                    # Mixed-parity trinomials, not parent binomials.
                    g = G.pick(rr)
                    offsets = (e, (rr.randrange(1, 7), 0),
                               (rr.randrange(7), 1))
                    cell = [G.mul(g, h) for h in offsets]
                else:
                    cell = rr.sample(G.elements, weight)
                row.append(cell)
            T.append(row)
        return T

    # Same search in every call; idx selects an invariant-distinct
    # optimized code, not an independent random draw.
    raw = [[] for _ in profiles]
    exact_seen = set()
    for layout in range(len(profiles)):
        for attempt in range(128):
            T = make_design(layout, attempt)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in T for cell in row)
            if key in exact_seen:
                continue
            exact_seen.add(key)
            cols = component_columns(T)
            if cols is None:
                continue
            n4 = fourth_count(cols)
            if n4 is not None:
                raw[layout].append((n4, key, T, cols))

    library = []
    signatures = set()
    for layout, candidates in enumerate(raw):
        candidates.sort(key=lambda item: (item[0], item[1]))
        screened = []
        for n4, key, T, cols in candidates[:32]:
            sig = sixth_signature(cols, n4)
            terms = sum(len(cell) for row in T for cell in row)
            cost = (sig[0], sig[1],
                    sig[2] * (terms / 18.0) ** 3,
                    sig[3], sig[4], sig[5], key)
            screened.append((cost, sig, key, T, terms))
        screened.sort(key=lambda item: item[0])
        kept = 0
        for _, sig, key, T, terms in screened:
            if sig in signatures:
                continue
            signatures.add(sig)
            library.append((sig, key, T, terms))
            kept += 1
            if kept == 4:
                break
    if not library:
        raise ValueError("no full-rank component passed short-word screens")

    pairings = []
    for a in range(len(library)):
        for b in range(a, len(library)):
            sa, sb = library[a][0], library[b][0]
            A, B = library[a][2], library[b][2]
            ga = [[len(cell) for cell in row] for row in A]
            gb = [[len(cell) for cell in row] for row in B]
            w = _grid_weights(ga, gb)
            if w > W_MAX or not _no_dead_lines(ga, gb):
                continue
            terms = library[a][3] + library[b][3]
            # At these fixed shapes CNOT count is 48*q*terms.
            # A cubic discount models leading three-fault events:
            # a testable heuristic, not measured circuit fitness.
            noise = (terms / 36.0) ** 3
            cost = (
                sa[0] + sb[0], max(sa[0], sb[0]),
                sa[1] + sb[1], max(sa[1], sb[1]),
                max(sa[2], sb[2]) * noise,
                (sa[2] + sb[2]) * noise,
                max(sa[3], sb[3]), sa[3] + sb[3],
                w, terms, sa, sb, a, b
            )
            pairings.append((cost, a, b))
    pairings.sort()

    def graph_fingerprint(hx, hz, n):
        """Invariant under arbitrary check/qubit permutations.

        Use the uncoloured combined Tanner graph, so exchanging bases
        cannot manufacture another accepted slot. Different fingerprints
        certify nonisomorphism; equal fingerprints are conservatively
        collapsed even when they could represent different graphs.
        """
        rows = hx + hz
        nr = len(rows)
        degrees = [0] * n
        supports = []
        for row in rows:
            support = []
            while row:
                bit = row & -row
                v = bit.bit_length() - 1
                support.append(v)
                degrees[v] += 1
                row ^= bit
            supports.append(support)
        adj = [{} for _ in rows]
        bits = [0] * nr
        for i, x in enumerate(rows):
            for j in range(i):
                overlap = (x & rows[j]).bit_count()
                if overlap:
                    adj[i][j] = overlap
                    adj[j][i] = overlap
                    bits[i] |= 1 << j
                    bits[j] |= 1 << i
        labels = []
        for i, row in enumerate(rows):
            hist = {}
            for overlap in adj[i].values():
                hist[overlap] = hist.get(overlap, 0) + 1
            triangles = sum((bits[i] & bits[j]).bit_count()
                            for j in adj[i]) // 2
            seed = (row.bit_count(),
                    tuple(sorted(degrees[v] for v in supports[i])),
                    tuple(sorted(hist.items())), triangles)
            labels.append(sha256(repr(seed).encode()).digest())
        for _ in range(3):
            labels = [
                sha256(repr((
                    labels[i],
                    tuple(sorted((weight, labels[j])
                                 for j, weight in adj[i].items()))
                )).encode()).digest()
                for i in range(nr)
            ]
        return sha256(repr((
            tuple(sorted(degrees)), tuple(sorted(labels))
        )).encode()).digest()

    fingerprints = set()
    ordinal = 0
    for _, a, b in pairings:
        A = library[a][2]
        # This inversion gives the corresponding right-action component;
        # final screening still uses the evaluator's actual full lifts.
        B = [[[G.inv(g) for g in cell] for cell in row]
             for row in library[b][2]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 224:
            continue
        fingerprint = graph_fingerprint(hx, hz, n)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        if ordinal == idx:
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"cells": {"A": A, "B": B}}
        ordinal += 1
    raise ValueError("insufficient invariant-distinct optimized pairings")


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh copy of the already-screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (6,2) shapes alone ensure balanced basis protection; parent slot 26 had 180/500 Z failures versus 116/1750 X failures.
# REFUTED: Distinct component short-syndrome signatures guarantee distinct LP pairings; the measured parent produced only 21 distinct codes from 60 slots, with numerous Tanner-isomorphic pairings.
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
