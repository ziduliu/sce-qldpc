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
CONCEPT = ("For SL(2,3) x C2, order 48, A=(1,2) and B=(2,5), balancing A density against degree-three/four B columns and excluding involution-relative cell labels should raise five-code measured fitness above the 0.250 frontier record 16.5.")
GROUP = ("New direct-product matrix-group construction, genuinely different from C37:C3. Order 48 gives n=576 on the new shapes. The successful nonabelian order-44 low-rate bed motivates this untried group, without establishing a distance advantage; central involutions are addressed by label constraints and verified screening.")
SHAPES = ("Target cell 0.250 with (1,2)x(2,5): n=576, k>=144, guaranteed rate 0.250. A has 96 variables/48 checks and B has 240 variables/96 checks. Parent measured minima were exclusively sector 1, with X witnesses concentrated around Acol1 and Z witnesses around Bcol1; screen both components and every sector-1 embedding.")
FILLING = ("Five B-degree architectures cross four A-count/central-parity policies. A counts [3,3] permit B row totals nine; [3,4] or [4,4] require totals eight. Both LP weight sums stay <=12. Each cell uses distinct projective matrix classes, excluding involution-relative pairs. Independently optimize sixteen verified, target-rate fillings per slot, refine four finalists with equal kernel-search budgets, and prefer finalists without any full weight<=4 logical. Retain a legal fallback if all finalists have short witnesses. Search readings remain upper bounds, not certificates.")
FAMILY_NAME = "r025_SL2F3xC2_12x25_projective_balanced_isd"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL(2,3) x C2 requires order 48")

    # (a,b,c,d,z): determinant-one matrix over F3, times C2.
    matrices = [(a,b,c,d) for a in range(3) for b in range(3)
                for c in range(3) for d in range(3)
                if (a*d-b*c) % 3 == 1]
    els = [m + (z,) for z in range(2) for m in matrices]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return (g[0] % 3, g[1] % 3, g[2] % 3,
                g[3] % 3, g[4] % 2)

    def idx(g):
        return positions[red(g)]

    def mul(g, h):
        a,b,c,d,z = red(g)
        e,f,u,v,w = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3, z ^ w)

    def inv(g):
        a,b,c,d,z = red(g)
        return (d, (-b) % 3, (-c) % 3, a, z)

    return dict(elements=els, ident=(1,0,0,1,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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


def _fp_witness(rng, columns, embed, stabilizers, rounds=4, initial=None):
    """Search verified component kernels and test every block embedding.
    A returned value <= width comes from an actual logical witness;
    width+1 means that this restricted search found none."""
    width = len(columns)
    best = width + 1 if initial is None else initial

    def observe(word):
        nonlocal best
        weight = word.bit_count()
        if not word or weight >= best:
            return
        for full in embed(word):
            if _fp_reduce(full, stabilizers):
                best = weight
                return

    singles = {}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        if syndrome == 0:
            observe(word)
        if syndrome in singles:
            observe(word ^ singles[syndrome])
        else:
            singles[syndrome] = word
    pairs = {}
    for i in range(width):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            word = (1 << i) | (1 << j)
            if syndrome == 0:
                observe(word)
            if syndrome in singles:
                observe(word ^ singles[syndrome])
            if syndrome in pairs:
                observe(word ^ pairs[syndrome])
            else:
                pairs[syndrome] = word
    if best <= 4:
        return best

    order = list(range(width))
    for _ in range(rounds):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, word)
                    break
                old_syndrome, old_word = pivots[p]
                syndrome ^= old_syndrome
                word ^= old_word
            if not syndrome:
                words.append(word)

        # Reduce the whole dependency population, rather than just
        # descending from a few starts. Every operation stays in kernel.
        for sweep in range(2):
            words.sort(key=int.bit_count)
            short = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in short:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        word, weight = trial, tw
                reduced.append(word)
                observe(word)
            words = reduced

        short = sorted(set(words), key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                observe(word ^ other)
        if best <= 4:
            break
    return best


def _fp_low4(checks, stabilizers, width):
    """Exact full-matrix logical detection through weight four.
    Return a found witness weight, or width+1 if none exists."""
    columns = _fp_columns(checks, width)
    basis = _fp_basis(stabilizers)

    def logical(word):
        return bool(word and _fp_reduce(word, basis))

    singles = {}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        if syndrome == 0 and logical(word):
            return 1
        old = singles.get(syndrome)
        if old is not None:
            witness = word ^ old
            if logical(witness):
                return witness.bit_count()
        else:
            singles[syndrome] = word

    # If the singleton test passes, equal-syndrome representatives
    # differ only by stabilizers. The same invariant holds for pairs.
    pairs = {}
    for i, si in enumerate(columns):
        for j in range(i):
            syndrome = si ^ columns[j]
            word = (1 << i) | (1 << j)
            if syndrome == 0 and logical(word):
                return 2
            old = singles.get(syndrome)
            if old is not None:
                witness = word ^ old
                if logical(witness):
                    return witness.bit_count()
            old = pairs.get(syndrome)
            if old is not None:
                witness = word ^ old
                if logical(witness):
                    return witness.bit_count()
            else:
                pairs[syndrome] = word
    return width + 1


def _fp_draw_cells(rng, G, idx):
    """Twenty count/parity architectures, independently optimized."""
    profiles = (
        ((2,2,2,2,1), (2,2,2,1,2)),
        ((3,2,2,1,1), (1,2,2,2,2)),
        ((3,3,1,1,1), (1,1,3,2,2)),
        ((4,2,1,1,1), (0,2,3,2,2)),
        ((3,2,2,1,1), (2,2,1,2,2)),
    )
    profile, mode = (idx % 20) // 4, idx % 4
    sa = [[3, 4] if mode == 1 else
          ([4, 4] if mode == 2 else [3, 3])]
    sb = [list(row) for row in profiles[profile]]
    if mode in (1, 2):
        # A column of weight four limits BOTH B rows to eight.
        sb[0][0] -= 1
        sb[1][1] -= 1
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    cosets = [[g for g in G.elements if g[4] == z] for z in range(2)]

    def projective(g):
        m = tuple(g[:4])
        return min(m, tuple((-x) % 3 for x in m))

    def draw(count, c):
        if mode == 0:
            ones = count // 2 + (
                ((c + profile) % 2) if count % 2 else 0)
        elif mode == 1:
            ones = count // 2
            if c >= 2 and count % 2:
                ones += (c + profile) % 2
        elif mode == 2:
            ones = 1 if count >= 2 else (c + profile) % 2
        else:
            if c == 0:
                ones = 0
            elif c == 1:
                ones = 1
            else:
                ones = (count - 1 if (c + profile) % 2 else 1)
                ones = min(count, max(0, ones))

        bits = [0] * (count - ones) + [1] * ones
        rng.shuffle(bits)
        cell, used = [], set()
        for z in bits:
            choices = [g for g in cosets[z] if projective(g) not in used]
            g = rng.choice(choices)
            cell.append(g)
            used.add(projective(g))
        # Distinct projective classes exclude relative elements in
        # {+I,-I} x C2, which contains every involution of this group.
        return cell

    flat_counts = sa[0] + sb[0] + sb[1]
    flat = [draw(count, c) for c, count in enumerate(flat_counts)]
    return [flat[:2]], [flat[2:7], flat[7:12]]


def NEW_CTX(rng, G, idx):
    """Select complete fillings using only the verified construction."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    kb = q * (ca - ra) * (cb - rb)

    def embed_a(word):
        return [((word & mask) << (beta * q)) |
                ((word >> q) << ((cb + beta) * q))
                for beta in range(cb)]

    def embed_b(word):
        return [word << (j * cb * q) for j in range(ca)]

    def improve(item, rounds):
        item["da"] = _fp_witness(
            rng, item["ac"], embed_a, item["zs"],
            rounds=rounds, initial=item["da"])
        item["db"] = _fp_witness(
            rng, item["bc"], embed_b, item["xs"],
            rounds=rounds, initial=item["db"])

    def score(item):
        d = min(item["da"], item["db"], item.get("low", 25), 24)
        return (item["k"] * d * d / item["n"],
                d, item["da"] + item["db"],
                -item["cycles"], -(item["k"] - kb))

    pool = []
    for attempt in range(96):
        A, B = _fp_draw_cells(rng, G, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Exact band gate: 0.225 <= k/n < 0.275.
        if k <= 0 or not (9 * n <= 40 * k < 11 * n):
            continue

        # HX at Bcol0 extracts the left-lifted A component.
        ar = [(row & mask) |
              (((row >> (cb * q)) & mask) << q)
              for row in hx[:q]]
        # HZ at Acol0 extracts BOTH right-lifted B check rows.
        br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
        ac = _fp_columns(ar, ca * q)
        bc = _fp_columns(br, cb * q)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz,
            "n": n, "k": k, "ac": ac, "bc": bc,
            "zs": _fp_basis(hz), "xs": _fp_basis(hx),
            "da": ca * q + 1, "db": cb * q + 1,
            "cycles": _fp_cycles(ac) + _fp_cycles(bc),
        }
        improve(item, 4)
        # Do not score an absence of found witnesses as a distance.
        if item["da"] <= ca * q and item["db"] <= cb * q:
            pool.append(item)
        if len(pool) == 16:
            break

    if not pool:
        raise ValueError("no witnessed candidate passed the 0.250 rate gate")

    finalists = sorted(pool, key=score, reverse=True)[:4]
    for rounds in (24, 56):
        for item in finalists:
            improve(item, rounds)

    # Check the complete matrices, including mixed-sector operators.
    # Prefer exact exclusion through weight four, but never void an
    # otherwise legal slot solely because this extra screen failed.
    for item in finalists:
        item["low"] = min(
            _fp_low4(item["hx"], item["hz"], item["n"]),
            _fp_low4(item["hz"], item["hx"], item["n"]))
    clean = [item for item in finalists if item["low"] > 4]
    winner = max(clean if clean else finalists, key=score)

    hx, hz = winner["hx"], winner["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert winner["k"] >= kb
    return {"idx": idx, "cells": {"A": winner["A"], "B": winner["B"]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without additional random filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: uniform labels with legal rolled counts alone exclude light logicals; the parent found many weight-2 witnesses and a measured top-five witness of weight 7.
# REFUTED: Greater rank surplus implies better distance readings on C37:C3; slots 14 and 54 have surplus 12 but d2s=12, while slot 12 has surplus 4 and d30s=21.
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
