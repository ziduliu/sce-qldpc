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
CONCEPT = ("GL(2,3), order 48, shapes (1,2)x(2,5): four-term A cells and degree-three/four B columns selected against short component kernels should exceed the 0.250 frontier fitness of 17.3; evaluator distances remain upper readings.")
GROUP = ("Jump from S3 x C4 to the invertible 2x2 matrix group over F3, of order (9-1)*(9-3)=48. Related nonabelian beds score well, but this group/shape bed is untried. Noncommuting labels and the determinant C2 quotient provide distinct filling models, not a presumed distance guarantee.")
SHAPES = ("Target cell 0.250 rather than the parent's 0.550 plateau: (1,2)x(2,5) gives n=576, kb=144 and guaranteed rate 1/4. Components have 96 variables/48 checks and 240 variables/96 checks. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly; reinforce both components and require exact k/n<0.275.")
FILLING = ("A counts are [4,4]; canonical B support cards have column sums three/four and row sums at most eight. Thus HX<=8+4=12 and HZ<=8+4=12. Determinant occupancy cards exclude involution differences inside cells. Verified lifts supply exact ranks, component short-kernel tests and progressively deeper kernel-witness probes. Select whole-matrix invariant-distinct designs; component readings are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r025_GL2F3_12x25_degree34_kernelcards"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # GL(2,F3), encoded row-major as (a,b,c,d).
    if not valid_q(q):
        raise ValueError("GL(2,F3) requires order 48")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3]
    positions = {g:i for i,g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = red(g)
        e,f,u,v = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a,b,c,d = red(g)
        t = (a*d-b*c) % 3
        # Both nonzero elements of F3 are their own inverses.
        return ((t*d) % 3, (-t*b) % 3,
                (-t*c) % 3, (t*a) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


from functools import lru_cache
from itertools import combinations_with_replacement


def _det(g):
    return (g[0]*g[3] - g[1]*g[2]) % 3


def _b_support_cards():
    """Canonical count architectures, not permutations of one profile."""
    pairs = tuple((a, total-a)
                  for total in (3, 4) for a in range(total+1))
    cards = set()
    for columns in combinations_with_replacement(pairs, 5):
        top = sum(a for a, b in columns)
        bottom = sum(b for a, b in columns)
        if min(top, bottom) < 7 or max(top, bottom) > 8:
            continue
        swapped = tuple(sorted((b, a) for a, b in columns))
        cards.add(min(columns, swapped))
    return tuple(sorted(cards, key=lambda c: (
        sum(a == 0 or b == 0 for a, b in c),
        -sum(a+b for a, b in c), c)))


_B_SUPPORT_CARDS = _b_support_cards()


def _columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length()-1] |= check
            row ^= bit
    return out


def _cycles4(cols):
    counts = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length()-1)
            col ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                total += old
                counts[key] = old + 1
    return total


def _short_kernel(cols, triples=False):
    """Exact existence test through weight four, or six with triples.

    A collision between distinct subsets of size <=2/3 supplies an
    actual nonzero kernel word. The sentinel 5/7 means none exists.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]

    def insert(syndrome, word):
        old = seen.get(syndrome)
        if old is not None:
            return (old ^ word).bit_count()
        seen[syndrome] = word
        return 0

    for i, syndrome in enumerate(cols):
        hit = insert(syndrome, bits[i])
        if hit:
            return hit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            hit = insert(syndrome ^ cols[j], bits[i] | bits[j])
            if hit:
                return hit
    if triples:
        for i, syndrome in enumerate(cols):
            for j in range(i):
                s = syndrome ^ cols[j]
                word = bits[i] | bits[j]
                for h in range(j):
                    hit = insert(s ^ cols[h], word | bits[h])
                    if hit:
                        return hit
    return 7 if triples else 5


def _kernel_probe(cols, rng, trials):
    """Find actual classical kernel witnesses; never certify distance."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        for sweep in range(2):
            words.sort(key=int.bit_count)
            for i in range(len(words)):
                word = words[i]
                weight = word.bit_count()
                for other in words[:i]:
                    trial = word ^ other
                    w = trial.bit_count()
                    if 0 < w < weight:
                        word, weight = trial, w
                words[i] = word
                best = min(best, weight)
        small = words[:18]
        for i, word in enumerate(small):
            for other in small[:i]:
                trial = word ^ other
                if trial:
                    best = min(best, trial.bit_count())
        if best <= 4:
            break
    return best


def _component_rows(hx, hz, q):
    """Extract actions only from the scaffold's verified LP matrices."""
    mask = (1 << q)-1
    ar = [(row & mask) | (((row >> (5*q)) & mask) << q)
          for row in hx[:q]]
    bm = (1 << (5*q))-1
    br = [row & bm for row in hz[:2*q]]
    return ar, br


def _draw_cell(rng, G, size, model, phase):
    """Determinant occupancy plus exclusion of involution differences."""
    if not size:
        return []
    if model == 0:
        quota = [None] * size
    elif model == 1:
        quota = [1 + ((phase+t) % 2) for t in range(size)]
    else:
        quota = [1 + phase % 2] * size
    cell = []
    for det in quota:
        available = []
        for g in G.elements:
            if g in cell or (det is not None and _det(g) != det):
                continue
            good = True
            for h in cell:
                relative = G.mul(G.inv(h), g)
                if G.mul(relative, relative) == G.e:
                    good = False
                    break
            if good:
                available.append(g)
        if not available:
            return None
        cell.append(rng.choice(available))
    return cell


def _draw_grid(rng, G, counts, model, phase):
    grid = []
    for i, row in enumerate(counts):
        output = []
        for j, size in enumerate(row):
            cell = _draw_cell(rng, G, size, model, phase+i+j)
            if cell is None:
                return None
            output.append(cell)
        grid.append(output)
    return grid


def _syndrome_invariant(rows, width):
    """Invariant under arbitrary qubit and check permutations."""
    cols = _columns(rows, width)
    pairs = {}
    for i, a in enumerate(cols):
        for b in cols[:i]:
            syndrome = a ^ b
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    histogram = {}
    for syndrome, count in pairs.items():
        key = (syndrome.bit_count(), count)
        histogram[key] = histogram.get(key, 0) + 1
    return (tuple(sorted(row.bit_count() for row in rows)),
            tuple(sorted(col.bit_count() for col in cols)),
            tuple(sorted(histogram.items())))


def _matrix_invariant(hx, hz, n, k):
    # Also invariant under exchanging X and Z.
    return (k, tuple(sorted((_syndrome_invariant(hx, n),
                            _syndrome_invariant(hz, n)))))


def _select_a(rng, G, card):
    dummy = [[[G.e] for _ in range(5)] for _ in range(2)]
    finalists = []
    for attempt in range(28):
        model = (card + attempt//10) % 3
        A = _draw_grid(rng, G, [[4, 4]], model, card+attempt)
        if A is None:
            continue
        hx, hz, _ = lifted_matrices(G, A, dummy)
        ar, _ = _component_rows(hx, hz, G.q)
        rank = gf2_rank(ar)
        if rank < G.q-3:
            continue
        ac = _columns(ar, 2*G.q)
        short = _short_kernel(ac)
        if short < 5:
            continue
        reading = _kernel_probe(ac, rng, 3)
        cycles = _cycles4(ac)
        finalists.append((reading, rank, -cycles, attempt, A, ac))
    if not finalists:
        return None
    finalists.sort(key=lambda item: item[:4], reverse=True)
    deep = []
    for reading, rank, negcycles, attempt, A, ac in finalists[:5]:
        clean6 = _short_kernel(ac, triples=True) == 7
        reading = min(reading, _kernel_probe(ac, rng, 16))
        deep.append(((reading, int(clean6), rank, negcycles), A, ac))
    return max(deep, key=lambda item: item[0])


@lru_cache(maxsize=1)
def _design_bank(G):
    """Pure deterministic design search; caching changes no final cells."""
    bank = []
    seen = set()
    for idx in range(N_SLOTS):
        chosen = None
        for retry in range(4):
            card = idx + retry*N_SLOTS
            rng = random.Random(25048001 + 1009*card)
            columns = _B_SUPPORT_CARDS[card % len(_B_SUPPORT_CARDS)]
            sb = [[pair[i] for pair in columns] for i in range(2)]
            assert _grid_weights([[4, 4]], sb) <= W_MAX
            assert _no_dead_lines([[4, 4]], sb)
            selected_a = _select_a(rng, G, card)
            if selected_a is None:
                continue
            akey, A, ac = selected_a
            da = akey[0]
            pool = []
            for attempt in range(56):
                # The architecture and determinant model are design cards;
                # candidate draws are internal optimization, not final slots.
                model = (card//len(_B_SUPPORT_CARDS) + attempt//19) % 3
                B = _draw_grid(rng, G, sb, model, card+attempt)
                if B is None:
                    continue
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (144 <= k and 1000*k < 275*n):
                    continue
                _, br = _component_rows(hx, hz, G.q)
                bc = _columns(br, 5*G.q)
                if _short_kernel(bc) < 5:
                    continue
                db = _kernel_probe(bc, rng, 3)
                cycles = _cycles4(bc)
                d = min(da, db)
                key = (k*d*d, d, da+db, -cycles)
                pool.append((key, B, bc, hx, hz, n, k, db))
            pool.sort(key=lambda item: item[0], reverse=True)

            deep = []
            for item in pool[:8]:
                _, B, bc, hx, hz, n, k, db = item
                db = min(db, _kernel_probe(bc, rng, 20))
                d = min(da, db)
                key = (k*d*d, d, da+db, -_cycles4(bc))
                deep.append((key, B, hx, hz, n, k))
            deep.sort(key=lambda item: item[0], reverse=True)
            for key, B, hx, hz, n, k in deep:
                fingerprint = _matrix_invariant(hx, hz, n, k)
                if fingerprint in seen:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                seen.add(fingerprint)
                chosen = (A, B)
                break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError("no fresh screened design for card %d" % idx)
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Each slot receives a selected invariant-distinct structural design."""
    A, B = _design_bank(G)[idx]
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the stored full cell without additional random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Odd-central S3 x C4 binomial deletion designs lifted the measured distance above four: all five measured parent codes had d30s=4, and deletion variants also collapsed under Tanner-graph isomorphism.
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
