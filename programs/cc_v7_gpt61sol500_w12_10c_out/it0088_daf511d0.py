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
CONCEPT = ("For C37:C3, order 111, shapes (2,1)x(2,1), fresh four-term fillings selected by independently optimized transpose kernels and full-matrix logical descent should raise five-code fitness above 91.7.")
GROUP = ("Retain the metacyclic order-three action s=10, since 10^3=1 mod 37. The current order-111 bed measured d30s=18..21 and its frontier successor reached fitness 91.7. The tall orientation retains size factor five, so no lift-order reduction is needed.")
SHAPES = ("Target 0.200 with A=(2,1), B=(2,1): n=555 and kb=111 from 111*(1-2)*(1-2). Equal component maps have 111 variables/222 checks; their transposes have 222 variables/111 checks. Parent minima occupied sector-one Acol/Bcol blocks; corresponding transpose kernels are exposed in sector two here. Orientation alone is not an improvement claim. Exact ranks retain 0.175<=k/n<0.225.")
FILLING = ("Every cell contains four distinct terms, giving both CSS row-weight maxima twelve. Twenty quotient-composition cards independently optimize new labels, never transpose a stored parent filling. Verified lifts supply transpose components; equal-budget kernel searches refine component finalists. Complete combinations are ranked with exact k and stabilizer-tested sector-two witnesses, then rescreened in both full matrices with stabilizer descent. All witness readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r020_frob37c3_21x21_fresh_dual_component_isd"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C37:C3 requires order 111")
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    def mul(g, h):
        return ((g[0] + pow(s, g[1] % r, m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (2, 1)               # (rb, cb)


def _fp_basis(rows):
    pivots = {}
    for v in rows:
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    return pivots


def _fp_reduce(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return v
        v ^= pivots[p]
    return 0


def _fp_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _fp_cycles(columns):
    total = 0
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            overlap = (a & b).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _fp_witness(rng, columns, embed, stabilizers, trials=8,
                best=None, short=True, descent_rows=()):
    """Actual commuting witnesses, with optional stabilizer descent.
    A supplied best must already come from a nontrivial witness.
    The width+1 sentinel means no witness, never a distance reading."""
    width = len(columns)
    if best is None:
        best = width + 1

    def observe(word):
        nonlocal best
        if not word:
            return
        if not descent_rows and word.bit_count() >= best:
            return
        full = embed(word)
        if descent_rows:
            # Adding checks preserves commutation and logical class.
            weight = full.bit_count()
            for _ in range(2):
                changed = False
                for row in descent_rows:
                    trial = full ^ row
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        full, weight = trial, tw
                        changed = True
                if not changed:
                    break
        weight = full.bit_count()
        if weight < best and _fp_reduce(full, stabilizers):
            best = weight

    if short:
        # Representatives of supports of size <=2 cover weights <=4.
        # Stabilizer-equivalent collisions preserve the quotient class.
        seen = {0: 0}
        for i, syndrome in enumerate(columns):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            else:
                observe(word ^ old)
        for i in range(width):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                else:
                    observe(word ^ old)
        if best <= 4:
            return best

    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)

        words.sort(key=int.bit_count)
        small = words[:24]
        for word in small:
            observe(word)
        for i, word in enumerate(small):
            for other in small[:i]:
                observe(word ^ other)
        # Short triples supplement information-set dependencies.
        for i, word in enumerate(small[:8]):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    observe(pair ^ small[h])

        for start in small[:8]:
            word = start
            for _ in range(4):
                improved, weight = word, word.bit_count()
                for other in words:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        improved, weight = trial, tw
                if improved == word:
                    break
                word = improved
                observe(word)
        if best <= 4:
            break
    return best


# Each card constrains four independently optimized cells:
# Arow0, Arow1, Brow0, Brow1. Types are balanced, skewed, two-coset.
# Some cards share a support orbit; labels are never transported between
# them. Fresh label optimization, not a quotient shift, creates each code.
_FP_TALL_CARDS = (
    (0,0,0,0), (1,1,1,1), (0,0,1,1), (0,1,0,1),
    (0,1,1,0), (0,0,0,1), (0,0,1,0), (0,1,0,0),
    (1,0,0,0), (0,1,1,1), (1,0,1,1), (1,1,0,1),
    (1,1,1,0), (2,0,0,0), (0,2,0,0), (0,0,2,0),
    (0,0,0,2), (2,1,0,1), (0,1,2,1), (2,0,2,0),
)


def NEW_CTX(rng, G, idx):
    """Optimize fresh tall components, then their complete CSS pair."""
    q = G.q
    mask = (1 << q) - 1
    card = _FP_TALL_CARDS[idx % len(_FP_TALL_CARDS)]
    quotient_patterns = ((0,0,1,2), (0,0,0,1), (0,0,1,1))
    pools = [[g for g in G.elements if g[1] == b] for b in range(3)]
    supports = [[4], [4]]
    assert _grid_weights(supports, supports) <= W_MAX
    assert _no_dead_lines(supports, supports)

    def draw_side(side):
        offset = 0 if side == "A" else 2
        grid = []
        for i in range(2):
            pattern = quotient_patterns[card[offset + i]]
            shift = (offset + i + idx // 20) % 3
            counts = [sum((b + shift) % 3 == t for b in pattern)
                      for t in range(3)]
            cell = []
            for b, count in enumerate(counts):
                cell.extend(rng.sample(pools[b], count))
            grid.append([cell])
        return grid

    def transpose_components(hx, hz):
        # Tall sector-one size is q. Sector-two blocks are
        # (Arow0,Brow0), (Arow0,Brow1), (Arow1,Brow0),
        # (Arow1,Brow1), at offsets q,2q,3q,4q.
        # HZ j=0,alpha=0 exposes A^T; HX i=0,beta=0 exposes B^T.
        ar = [((row >> q) & mask)
              | (((row >> (3*q)) & mask) << q)
              for row in hz[:q]]
        br = [(row >> q) & ((1 << (2*q)) - 1)
              for row in hx[:q]]
        return ar, br

    identity = lambda word: word
    skeleton = [[[G.e]], [[G.e]]]

    def component_key(item):
        return (item["d"], -item["cycles"], item["rank"])

    def optimize_side(side):
        retained = []
        for _ in range(24):
            cells = draw_side(side)
            A = cells if side == "A" else skeleton
            B = cells if side == "B" else skeleton
            hx, hz, _ = lifted_matrices(G, A, B)
            ar, br = transpose_components(hx, hz)
            rows = ar if side == "A" else br
            columns = _fp_columns(rows, 2*q)
            rank = gf2_rank(rows)
            reading = _fp_witness(
                rng, columns, identity, {}, trials=6)
            if reading > 2*q:
                continue
            item = {
                "cells": cells, "columns": columns,
                "d": reading, "rank": rank,
                "cycles": _fp_cycles(columns),
            }
            retained.append(item)
            retained.sort(key=component_key, reverse=True)
            del retained[4:]

        if not retained:
            raise ValueError("no transpose-component kernel witness")
        # Equal deeper budgets for every retained component.
        for item in retained:
            item["d"] = _fp_witness(
                rng, item["columns"], identity, {}, trials=48,
                best=item["d"], short=False)
        retained.sort(key=component_key, reverse=True)
        return retained[:3]

    pa, pb = optimize_side("A"), optimize_side("B")
    embed_a = lambda word: (((word & mask) << q)
                            | ((word >> q) << (3*q)))
    embed_b = lambda word: word << q

    def improve(item, trials):
        # A^T kernels commute with HZ and represent X candidates.
        # B^T kernels commute with HX and represent Z candidates.
        item["da"] = _fp_witness(
            rng, item["ac"], embed_a, item["xs"], trials=trials,
            best=item["da"], short=False)
        item["db"] = _fp_witness(
            rng, item["bc"], embed_b, item["zs"], trials=trials,
            best=item["db"], short=False)

    def merit(item):
        d = min(item["da"], item["db"], item.get("full", item["n"]+1))
        return (item["k"] * d*d / item["n"],
                d, item["da"] + item["db"], -item["cycles"])

    combinations = []
    for a in pa:
        for b in pb:
            A, B = a["cells"], b["cells"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (7*n <= 40*k < 9*n):
                continue
            item = {
                "A": A, "B": B, "hx": hx, "hz": hz,
                "n": n, "k": k,
                "ac": a["columns"], "bc": b["columns"],
                "xs": _fp_basis(hx), "zs": _fp_basis(hz),
                "da": 2*q + 1, "db": 2*q + 1,
                "cycles": a["cycles"] + b["cycles"],
            }
            improve(item, 8)
            # Sentinels never earn a score.
            if item["da"] <= 2*q and item["db"] <= 2*q:
                combinations.append(item)

    if not combinations:
        raise ValueError("no nontrivial pair in the 0.200 rate band")
    finalists = sorted(combinations, key=merit, reverse=True)[:3]
    for item in finalists:
        improve(item, 48)
        bound = min(item["da"], item["db"])
        # Both sectors are present in these full-matrix searches.
        # Stabilizer descent can expose lighter representatives without
        # changing the logical class of the information-set word.
        dx = _fp_witness(
            rng, _fp_columns(item["hz"], item["n"]),
            identity, item["xs"], trials=8, best=bound,
            descent_rows=item["hx"])
        dz = _fp_witness(
            rng, _fp_columns(item["hx"], item["n"]),
            identity, item["zs"], trials=8, best=bound,
            descent_rows=item["hz"])
        item["full"] = min(dx, dz)

    winner = max(finalists, key=merit)
    hx, hz = winner["hx"], winner["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 7*winner["n"] <= 40*winner["k"] < 9*winner["n"]
    return {"idx": idx,
            "cells": {"A": winner["A"], "B": winner["B"]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional random filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: uniform labels with legal rolled counts alone exclude light logicals; the parent found many weight-2 witnesses and a measured top-five witness of weight 7.
# REFUTED: Four terms in every C37:C3 cell alone ensure a measured distance floor of 21; current-parent slot 13 has four terms throughout but d30s=18, with sector-one X minima.
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
