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
CONCEPT = ("S3 x C4, order 24, shapes (2,3)x(2,5): strengthened component columns and short-kernel optimization should produce five new cell-0.150 codes with d30s>=13; this is a falsifiable search-reading hypothesis, not a distance certificate.")
GROUP = ("Keep the parent's S3 x C4 direct-product construction and order 24 unchanged. Its five measured generator-deletion codes all had d30s=4. The new shape fits this same group at n=456, isolating the effect of component structure and filling.")
SHAPES = ("Target cell 0.150: (2,3)x(2,5) gives n=19*24=456, kb=24*1*3=72 and guaranteed rate 3/19=0.157895. Component lengths are 72 and 120 with 48 checks each. Exact verified ranks enforce 0.125<=k/n<0.175. Parent weight-four logicals occupied sector-1 Acol/Bcol blocks broadly; both sides now have two check rows.")
FILLING = ("Enumerate canonical support cards with both LP row sums <=12, prioritizing stronger columns on the higher-rate B component. Optimize fresh labels without involutory within-cell differences. Extract components only from verified lifts, reject kernels through weight four, and rank candidates by randomized component-kernel witnesses and four-cycles. Final full lifts also exclude kernels through weight four. Different built Tanner-graph invariants separate submitted slots; no probe is treated as certified quantum distance.")
FAMILY_NAME = "r015_S3xC4_23x25_component_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # (a,b,c) = r^a t^b z^c:
    # r^3=t^2=z^4=1, t*r*t=r^-1, z central.
    if not valid_q(q):
        raise ValueError("S3 x C4 requires order 24")
    els = [(a, b, c) for c in range(4)
           for b in range(2) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 2, g[2] % 4)

    def mul(g, h):
        return ((g[0] + (1 if g[1] == 0 else -1) * h[0]) % 3,
                (g[1] + h[1]) % 2, (g[2] + h[2]) % 4)

    def inv(g):
        return ((-g[0] if g[1] == 0 else g[0]) % 3,
                g[1], (-g[2]) % 4)

    def idx(g):
        a, b, c = red(g)
        return (2*c + b)*3 + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _no_kernel_le3(rows, n):
    """Exhaustive conservative test: no nonzero kernel of weight <=3.
    Rejecting even a small stabilizer is intentional here."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, x in enumerate(columns):
        for j in range(i):
            if (x ^ columns[j]) in singles:
                return False
    return True


from functools import lru_cache
from itertools import combinations_with_replacement
from collections import Counter


@lru_cache(maxsize=1)
def _r015_support_cards():
    """Remove component row/column permutations before assigning cards."""
    def profiles(width, colmax, rowlo, rowhi):
        types = [(a, b) for a in range(5) for b in range(5)
                 if 2 <= a + b <= colmax]
        out = []
        for columns in combinations_with_replacement(types, width):
            swapped = tuple(sorted((b, a) for a, b in columns))
            if columns > swapped:
                continue
            totals = tuple(sum(c[i] for c in columns) for i in range(2))
            if min(totals) < rowlo or max(totals) > rowhi:
                continue
            # Full row rank of the binary augmentation matrix.
            parity = tuple(sum((c[i] & 1) << j
                               for j, c in enumerate(columns))
                           for i in range(2))
            if not parity[0] or not parity[1] or parity[0] == parity[1]:
                continue
            degrees = tuple(a + b for a, b in columns)
            out.append((columns, totals, degrees))
        return out

    aps = profiles(A_SHAPE[1], 5, 4, 8)
    bps = profiles(B_SHAPE[1], 4, 6, 9)
    cards = []
    for ac, ar, ad in aps:
        for bc, br, bd in bps:
            if max(ar) + max(bd) > W_MAX:
                continue
            if max(br) + max(ad) > W_MAX:
                continue
            zeros = sum(v == 0 for c in ac + bc for v in c)
            largest = max(v for c in ac + bc for v in c)
            # B has the higher component rate: spend available support
            # there first, while retaining adequate A column degrees.
            priority = (min(bd), sum(br), min(ad), sum(ar),
                        -zeros, -largest,
                        -abs(ar[0] - ar[1]) - abs(br[0] - br[1]))
            cards.append((priority, ac, bc))
    cards.sort(reverse=True)
    if len(cards) < N_SLOTS:
        raise ValueError("insufficient canonical support cards")
    return tuple((ac, bc) for _, ac, bc in cards)


def _r015_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= check
            row ^= bit
    return cols


def _r015_clean4(cols):
    """Exact conservative exclusion of every kernel word of weight <=4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    pairs = set()
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in singles or s in pairs:
                return False
            pairs.add(s)
    return True


def _r015_cycles(cols):
    seen = {}
    total = 0
    for col in cols:
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length() - 1)
            col ^= bit
        for i, a in enumerate(checks):
            for b in checks[:i]:
                key = (b, a)
                previous = seen.get(key, 0)
                total += previous
                seen[key] = previous + 1
    return total


def _r015_kernel_probe(cols, rng, trials=6):
    """Actual classical kernel witnesses; returned weights are upper bounds."""
    best = len(cols) + 1
    hits = 0

    def record(word):
        nonlocal best, hits
        if word:
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
                basis.append(word)
                record(word)
        for i, word in enumerate(basis):
            for other in basis[:i]:
                record(word ^ other)
        if len(basis) >= 3:
            for _ in range(256):
                a, b, c = rng.sample(range(len(basis)), 3)
                record(basis[a] ^ basis[b] ^ basis[c])
    return best, hits


def _r015_components(hx, hz, q):
    """Extract both actions from the VERIFIED full LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            ar.append(sum(
                ((source >> (j * cb * q)) & mask) << (j * q)
                for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ar, br


def _r015_labels(rng, G, columns):
    grid = [[], []]
    for counts in columns:
        for i, size in enumerate(counts):
            cell = []
            for _ in range(size):
                allowed = []
                for g in G.elements:
                    if g in cell:
                        continue
                    legal = True
                    for h in cell:
                        relative = G.mul(G.inv(h), g)
                        if G.mul(relative, relative) == G.e:
                            legal = False
                            break
                    if legal:
                        allowed.append(g)
                if not allowed:
                    return None
                cell.append(rng.choice(allowed))
            grid[i].append(cell)
    return grid


def _r015_graph_fingerprint(hx, hz, n):
    """Invariant of the combined, untyped Tanner graph.

    Different fingerprints prove nonisomorphism; equal fingerprints
    are conservatively discarded, without claiming an isomorphism.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    overlaps = Counter()
    for i, row in enumerate(rows):
        wi = weights[i]
        for j in range(i):
            wj = weights[j]
            overlaps[(min(wi, wj), max(wi, wj),
                      (row & rows[j]).bit_count())] += 1
    cols = _r015_columns(rows, n)
    neighborhoods = Counter()
    for col in cols:
        neighbor_weights = []
        while col:
            bit = col & -col
            neighbor_weights.append(weights[bit.bit_length() - 1])
            col ^= bit
        neighborhoods[tuple(sorted(neighbor_weights))] += 1
    return (tuple(sorted(weights)),
            tuple(sorted(overlaps.items())),
            tuple(sorted(neighborhoods.items())))


@lru_cache(maxsize=1)
def _r015_design_bank(G):
    """Deterministic, transparently cached search over structural cards."""
    cards = _r015_support_cards()
    used = set()
    bank = []
    kb = G.q * (A_SHAPE[1] - A_SHAPE[0]) * (
        B_SHAPE[1] - B_SHAPE[0])

    for slot in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + slot + 150456)
        chosen = None
        # Alternate cards are genuine support changes, not relabelings.
        for fallback in range(3):
            ac, bc = cards[(slot + fallback * N_SLOTS) % len(cards)]
            sa = [[c[i] for c in ac] for i in range(2)]
            sb = [[c[i] for c in bc] for i in range(2)]
            assert _grid_weights(sa, sb) <= W_MAX
            assert _no_dead_lines(sa, sb)
            pool = []

            for attempt in range(64):
                A = _r015_labels(rng, G, ac)
                B = _r015_labels(rng, G, bc)
                if A is None or B is None:
                    continue
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (125 * n <= 1000 * k < 175 * n):
                    continue

                ar, br = _r015_components(hx, hz, G.q)
                ax = _r015_columns(ar, A_SHAPE[1] * G.q)
                bx = _r015_columns(br, B_SHAPE[1] * G.q)
                if not _r015_clean4(ax) or not _r015_clean4(bx):
                    continue
                da, ha = _r015_kernel_probe(ax, rng)
                db, hb = _r015_kernel_probe(bx, rng)
                cycles = _r015_cycles(ax) + _r015_cycles(bx)
                score = (min(da, db), da + db, -max(ha, hb),
                         -cycles, -abs(k - kb))
                pool.append((score, attempt, A, B, hx, hz, n))
                if len(pool) >= 8:
                    break

            pool.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            for _, _, A, B, hx, hz, n in pool:
                # Component tests alone do not exclude mixed-sector words.
                if not _r015_clean4(_r015_columns(hx, n)):
                    continue
                if not _r015_clean4(_r015_columns(hz, n)):
                    continue
                fingerprint = _r015_graph_fingerprint(hx, hz, n)
                if fingerprint in used:
                    continue
                if not css_ok(hx, hz):
                    raise ValueError("verified lift failed CSS check")
                used.add(fingerprint)
                chosen = (A, B)
                break
            if chosen is not None:
                break

        # A failed slot must not invalidate other successful bank entries.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign one optimized, graph-invariant-distinct design to each slot."""
    chosen = _r015_design_bank(G)[idx]
    if chosen is None:
        raise ValueError("no fresh target-rate design passed kernel screening")
    A, B = chosen
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Every new-shape cell is selected jointly before CELL is called."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen S3 x C4 generator-deletion slots are fifteen distinct designs: parent slots 0-14 collapsed to five isomorphism classes, all measured at d30s=4.
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
