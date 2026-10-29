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
CONCEPT = ("For cyclic metacyclic Z1:Z19, q=19, shapes (2,5)x(2,5), eliminate component four-cycles and preferentially exclude kernel words of weight <=6 to improve on the parent's repeated measured distance 6; this is a testable quantum-distance hypothesis, not a distance certificate.")
GROUP = ("Keep the parent's cyclic metacyclic construction rule. The new size factor 29 makes q=21 invalid (n=609); q=19 is the largest fitting prime, giving n=551. The order of 2 modulo 19 is 18, simplifying the binary cyclic spectrum, but ranks are checked with verified helpers.")
SHAPES = ("Equal (2,5) sides give length-95 components with 38 checks each. The guaranteed quantum rate is 9/29=0.310345, k>=171 at n=551, inside cell 0.300. The parent's measured weight-6 logicals occupied sector-1 Acol/Bcol blocks; replace its numerous degree-2 columns by degree-3/4 columns on both sides.")
FILLING = ("Each slot specifies a support-profile/difference-palette card and searches independent fillings. Pack check-neighbor differences, verify component girth >=6, and prefer exact absence of component kernel words through weight 6; randomized kernel probes rank survivors. Row sums <=8 and column sums <=4 guarantee both quantum row caps <=12. Verified final k must lie in the target band; component screening does not certify quantum distance.")
FAMILY_NAME = "r030_m1r19s1_25x25_pairpack_short6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    # Same cyclic metacyclic rule; order reduced to fit size factor 29.
    m, r, s = 1, 19, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _component_rows(G, M, side):
    """Extract the component from the VERIFIED quantum lift."""
    dummy = [[[G.e] for _ in range(5)] for _ in range(2)]
    A, B = (M, dummy) if side == "A" else (dummy, M)
    hx, hz, _ = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q) - 1
    if side == "A":
        return [
            sum(((hx[(i * 5) * q + r] >> (j * 5 * q)) & mask)
                << (j * q) for j in range(5))
            for i in range(2) for r in range(q)
        ]
    mask = (1 << (5 * q)) - 1
    return [hz[a * q + r] & mask
            for a in range(2) for r in range(q)]


def _column_syndromes(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _four_cycles(cols):
    seen = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length() - 1)
            col ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                key = (neighbors[b], neighbors[a])
                old = seen.get(key, 0)
                total += old
                seen[key] = old + 1
    return total


def _has_kernel_le6(cols):
    """Exact collision test on all subsets of size <=3.

    Any nonempty zero-syndrome support of size <=6 partitions into
    two different subsets of size <=3 with equal syndromes.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, s in enumerate(cols):
        if s in seen:
            return True
        seen[s] = bits[i]
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            if s in seen:
                return True
            seen[s] = bits[i] | bits[j]
    for i in range(len(cols)):
        for j in range(i):
            s2 = cols[i] ^ cols[j]
            mask2 = bits[i] | bits[j]
            for k in range(j):
                s = s2 ^ cols[k]
                if s in seen:
                    return True
                seen[s] = mask2 | bits[k]
    return False


def _kernel_probe(rng, cols, trials):
    """Actual kernel witnesses; readings are only upper bounds."""
    best = len(cols) + 1
    hits = 0

    def record(word):
        nonlocal best, hits
        if not word:
            return
        w = word.bit_count()
        if w < best:
            best, hits = w, 1
        elif w == best:
            hits += 1

    for _ in range(trials):
        order = list(range(len(cols)))
        rng.shuffle(order)
        pivots = {}
        basis = []
        for j in order:
            s, word = cols[j], 1 << j
            while s:
                p = s.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (s, word)
                    break
                ps, pw = pivots[p]
                s ^= ps
                word ^= pw
            if not s:
                basis.append(word)
                record(word)
        for i, word in enumerate(basis):
            for other in basis[:i]:
                record(word ^ other)
        for _ in range(300):
            a, b, c = rng.sample(range(len(basis)), 3)
            record(basis[a] ^ basis[b] ^ basis[c])
    return best, hits


def _packed_component(rng, G, profile, palette):
    """Distinct horizontal gaps and disjoint cross-row differences."""
    profiles = (
        ((2, 1), (2, 1), (1, 2), (1, 2), (2, 2)),
        ((2, 1), (2, 1), (2, 1), (1, 2), (1, 3)),
        ((2, 1), (2, 1), (2, 1), (1, 2), (1, 2)),
    )
    counts = profiles[profile]
    q = G.q
    horizontal = [set(), set()]
    offsets = [[None] * 5 for _ in range(2)]

    def gap(x):
        x %= q
        return min(x, q - x)

    for i in range(2):
        # Place trinomials before binomials to preserve feasible gaps.
        order = sorted(range(5), key=lambda j: -counts[j][i])
        for j in order:
            size = counts[j][i]
            available = set(palette) - horizontal[i]
            if size == 1:
                offsets[i][j] = [0]
            elif size == 2:
                if not available:
                    return None
                d = rng.choice(sorted(available))
                horizontal[i].add(d)
                offsets[i][j] = [0, d if rng.randrange(2) else q - d]
            else:
                choices = []
                for u in range(1, q):
                    for v in range(u + 1, q):
                        ds = {gap(u), gap(v), gap(v - u)}
                        if len(ds) == 3 and ds <= available:
                            choices.append((u, v, ds))
                if not choices:
                    return None
                u, v, ds = rng.choice(choices)
                offsets[i][j] = [0, u, v]
                horizontal[i].update(ds)

    cross = set()
    M = [[None] * 5 for _ in range(2)]
    order = sorted(range(5), key=lambda j: -sum(counts[j]))
    for j in order:
        top, bottom = offsets[0][j], offsets[1][j]
        choices = []
        for t in range(q):
            ds = {(u - v - t) % q for u in top for v in bottom}
            if len(ds) == len(top) * len(bottom) and not (ds & cross):
                choices.append((t, ds))
        if not choices:
            return None
        t, ds = rng.choice(choices)
        cross.update(ds)
        base = rng.randrange(q)
        M[0][j] = [G.elements[(base + u) % q] for u in top]
        M[1][j] = [G.elements[(base + t + v) % q] for v in bottom]
    return M


def _select_component(rng, G, side, profile, palette):
    best = None
    best_score = None
    clean_count = 0
    for _ in range(48):
        M = _packed_component(rng, G, profile, palette)
        if M is None:
            continue
        rows = _component_rows(G, M, side)
        if gf2_rank(rows) != 2 * G.q:
            continue
        cols = _column_syndromes(rows, 5 * G.q)
        if _four_cycles(cols):
            continue
        clean = not _has_kernel_le6(cols)
        reading, hits = _kernel_probe(rng, cols, 12 if clean else 4)
        score = (int(clean), reading, -hits)
        if best_score is None or score > best_score:
            best, best_score = M, score
        clean_count += int(clean)
        if clean_count >= 4:
            break
    if best is None:
        raise ValueError("no full-rank packed component for design card")
    return best


def NEW_CTX(rng, G, idx):
    """Slot-specific design cards, not automorphism/permutation copies."""
    import itertools
    palettes = list(itertools.combinations(range(1, 10), 6))
    pa = palettes[(17 * idx + 3) % len(palettes)]
    pb = palettes[(23 * idx + 11) % len(palettes)]
    A = _select_component(rng, G, "A", idx % 3, pa)
    B = _select_component(rng, G, "B", (idx // 3) % 3, pb)
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert css_ok(hx, hz)
    assert 0.275 <= k / n < 0.325, "verified rate outside target cell"
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating w=12 with rolled supports suffices to avoid weight-6 sector-1 logicals; parent measured slots 23,25,37,44,56 all retained distance 6.
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
