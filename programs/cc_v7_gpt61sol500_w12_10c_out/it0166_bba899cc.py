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
CONCEPT = ("C5 semidirect C16, action 2, order 80, shapes (1,2)x(1,3): long-relative-order trinomials, independently screened against ordinary and normal-subgroup component witnesses, should exceed the 0.300 frontier's 32.5 five-code fitness.")
GROUP = ("New semidirect family and group, not S3 x C4: y*x*y^-1=x^2, with 2^16=1 mod 5. BEDS contains no order-80 bed. The central C4=<y^4> and normal C5=<x> provide explicit restricted witness spaces; their absence of short witnesses does not certify unrestricted distance.")
SHAPES = ("Redirect to cell 0.300: (1,2)x(1,3) gives n=7*80=560, kb=2*80=160 and guaranteed rate 2/7. Components have 160/80 and 240/80 variable/check counts. These shapes admit q<=85. Parent weight-four minima occupied sector-one Acol/Bcol blocks broadly; optimize both components and subsequently probe both complete CSS sectors.")
FILLING = ("Every cell contains three distinct elements, giving HX weight 9 and HZ weight 12. Twenty cards cross five C16 spacing motifs with four cross-component arrangements. Verified lifts supply all syndromes. Reject component kernels through four, retain normal-subgroup and ordinary witnesses across deeper searches, then lower selection readings with stabilizer-aware full-code probes. Enforce the target rate and distinct uncoloured Tanner invariants. All randomized readings remain witness upper bounds or explicitly identified no-witness sentinels.")
FAMILY_NAME = "r030_C5byC16_action2_12x13_spacing_norm_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 80

def custom_ops(q):
    # (a,b)=x^a y^b; x^5=y^16=1, y*x*y^-1=x^2.
    if not valid_q(q):
        raise ValueError("C5 semidirect C16 requires order 80")
    els = [(a, b) for b in range(16) for a in range(5)]

    def red(g):
        return (g[0] % 5, g[1] % 16)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(2, b, 5)*c) % 5, (b+d) % 16)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 16
        return ((-pow(2, bi, 5)*a) % 5, bi)

    def idx(g):
        a, b = red(g)
        return 5*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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
from collections import Counter

_CC_MOTIFS = ((0, 1, 3), (0, 1, 6), (0, 2, 7),
              (0, 3, 9), (0, 5, 11))


def _cc_columns(rows, width):
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return cols


def _cc_basis(rows):
    basis = {}
    for source in rows:
        while source:
            p = source.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = source
                break
            source ^= old
    return basis


def _cc_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _cc_no_four(cols):
    """Exact classical exclusion through four, including odd weights."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    pairs = set()
    for i, x in enumerate(cols):
        for y in cols[:i]:
            syndrome = x ^ y
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _cc_cycles(cols):
    seen = {}
    cycles = 0
    for source in cols:
        positions = []
        while source:
            bit = source & -source
            positions.append(bit.bit_length()-1)
            source ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                pair = (b, a)
                old = seen.get(pair, 0)
                cycles += old
                seen[pair] = old+1
    return cycles


def _cc_probe(cols, rng, trials, seeds=(), stabilizers=()):
    """Verified kernel dependencies, optionally classified as logicals.

    Returns a found witness reading and retained kernel words. width+1
    is a no-witness sentinel, never a claimed distance.
    """
    width = len(cols)
    basis = _cc_basis(stabilizers)
    memory = sorted(set(seeds), key=lambda v: (v.bit_count(), v))[:32]
    best = width+1

    def eligible(word):
        return bool(word) and (not stabilizers or
                               _cc_outside(word, basis))

    for word in memory:
        if eligible(word):
            best = min(best, word.bit_count())

    order = list(range(width))
    for trial in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = list(memory)
        for c in order:
            syndrome, word = cols[c], 1 << c
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
                if eligible(word):
                    best = min(best, word.bit_count())

        words = sorted(set(words), key=lambda v: (v.bit_count(), v))
        short = words[:48]
        discoveries = []
        for i, word in enumerate(short):
            weight = word.bit_count()
            for other in short[:i]:
                combined = word ^ other
                cw = combined.bit_count()
                if combined and cw < weight:
                    discoveries.append(combined)
                    word, weight = combined, cw
                if eligible(combined):
                    best = min(best, cw)
            if eligible(word):
                best = min(best, weight)

        memory = sorted(set(memory+short+discoveries),
                        key=lambda v: (v.bit_count(), v))[:32]

        if stabilizers:
            logicals = [v for v in memory if eligible(v)]
            for word in logicals[:4]:
                weight = word.bit_count()
                for sweep in range(3):
                    chosen = word
                    for row in stabilizers:
                        combined = word ^ row
                        cw = combined.bit_count()
                        if cw < weight:
                            chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
                memory.append(word)
            memory = sorted(set(memory),
                            key=lambda v: (v.bit_count(), v))[:32]
    return best, tuple(memory)


@lru_cache(maxsize=1)
def _cc_fibers(G):
    """Partitions define test supports, not a second lift implementation."""
    subgroups = (
        tuple((a, 0) for a in range(5)),
        tuple((0, b) for b in range(0, 16, 4)),
        ((0, 0), (0, 8)),
    )
    result = []
    for subgroup in subgroups:
        remaining = set(range(G.q))
        partition = []
        while remaining:
            g = G.elements[min(remaining)]
            orbit = tuple(sorted(G.idx(G.mul(g, h))
                                 for h in subgroup))
            assert len(set(orbit)) == len(subgroup)
            remaining.difference_update(orbit)
            partition.append(orbit)
        result.append(tuple(partition))
    return tuple(result)


def _cc_norm_seeds(G, cols, rng, trials):
    """Expand actual restricted-kernel words into component witnesses."""
    words = []
    q = G.q
    for partition in _cc_fibers(G):
        supports = []
        folded = []
        for block in range(len(cols)//q):
            for orbit in partition:
                support, syndrome = 0, 0
                for h in orbit:
                    c = block*q+h
                    support |= 1 << c
                    syndrome ^= cols[c]
                supports.append(support)
                folded.append(syndrome)
        reading, compressed = _cc_probe(folded, rng, trials)
        for source in compressed:
            word = 0
            while source:
                bit = source & -source
                word ^= supports[bit.bit_length()-1]
                source ^= bit
            if word:
                words.append(word)
    return tuple(sorted(set(words), key=int.bit_count)[:24])


def _cc_draw(rng, G, card, side):
    """Different spacing arrangements; no common transported labeling."""
    p, mode = card % 5, (card//5) % 4
    models = ((p, (p+1+mode) % 5) if side == "A" else
              tuple((p+mode+2*j) % 5 for j in range(3)))
    row = []
    for j, model in enumerate(models):
        phase = rng.randrange(16)
        direction = -1 if (mode+j+(side == "B")) % 3 == 0 else 1
        quotient = [(phase+direction*b) % 16
                    for b in _CC_MOTIFS[model]]
        cell = [(rng.randrange(5), b) for b in quotient]
        assert len(cell) == len(set(cell)) == 3
        row.append(cell)
    return [row]


def _cc_component(G, grid, side):
    """Extract components only from the scaffold's exact LP matrices."""
    q = G.q
    stub_a = [[[G.e], [G.e]]]
    stub_b = [[[G.e], [G.e], [G.e]]]
    A, B = (grid, stub_b) if side == "A" else (stub_a, grid)
    hx, hz, n = lifted_matrices(G, A, B)
    if side == "A":
        mask = (1 << q)-1
        rows = [sum(((row >> (j*3*q)) & mask) << (j*q)
                    for j in range(2)) for row in hx[:q]]
        width = 2*q
    else:
        mask = (1 << (3*q))-1
        rows = [row & mask for row in hz[:q]]
        width = 3*q
    return rows, _cc_columns(rows, width)


def _cc_pool(G, card, side, rng):
    """Independent component screening with retained witness memory."""
    pool = []
    seen = set()
    quota = 48 if side == "A" else 64
    for attempt in range(quota):
        grid = _cc_draw(rng, G, card, side)
        key = tuple(tuple(sorted(cell)) for cell in grid[0])
        if key in seen:
            continue
        seen.add(key)
        rows, cols = _cc_component(G, grid, side)
        if not _cc_no_four(cols):
            continue
        structured = _cc_norm_seeds(G, cols, rng, 2)
        d, words = _cc_probe(cols, rng, 3, structured)
        item = dict(grid=grid, cols=cols, words=words, d=d,
                    cycles=_cc_cycles(cols), rank=gf2_rank(rows),
                    attempt=attempt)
        pool.append(item)
        pool.sort(key=lambda x: (x["d"], -x["cycles"], x["rank"],
                                 -x["attempt"]), reverse=True)
        del pool[5:]

    for item in pool:
        structured = _cc_norm_seeds(G, item["cols"], rng, 8)
        d, words = _cc_probe(item["cols"], rng, 32,
                            item["words"]+structured)
        item["d"] = min(item["d"], d)
        item["words"] = words
    pool.sort(key=lambda x: (x["d"], -x["cycles"], x["rank"],
                             -x["attempt"]), reverse=True)
    return pool[:3]


def _cc_signature(hx, hz, n):
    """Uncoloured Tanner invariants; different signatures exclude isomorphism."""
    rows = hx+hz
    degrees = [row.bit_count() for row in rows]
    profiles = [Counter() for _ in rows]
    overlap_counts = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlap_counts[overlap] += 1
            profiles[i][(degrees[j], overlap)] += 1
            profiles[j][(degrees[i], overlap)] += 1
    checks = tuple(sorted(
        (degrees[i], tuple(sorted(profile.items())))
        for i, profile in enumerate(profiles)))
    neighborhoods = [[] for _ in range(n)]
    for degree, source in zip(degrees, rows):
        while source:
            bit = source & -source
            neighborhoods[bit.bit_length()-1].append(degree)
            source ^= bit
    qubits = tuple(sorted(tuple(sorted(p)) for p in neighborhoods))
    return (n, checks, qubits, tuple(sorted(overlap_counts.items())))


def _cc_embeddings(words, side, q):
    """Sector-one seeds; full-matrix commutation is checked before use."""
    mask = (1 << q)-1
    result = []
    for word in words[:12]:
        for fixed in range(3 if side == "A" else 2):
            if side == "A":
                expanded = sum(((word >> (j*q)) & mask)
                               << ((j*3+fixed)*q) for j in range(2))
            else:
                expanded = word << (fixed*3*q)
            result.append(expanded)
    return result


@lru_cache(maxsize=1)
def _cc_bank(G):
    """One optimized spacing card per slot; never fill with an orbit copy."""
    assert G.q == 80
    assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
    bank = []
    signatures = set()

    for card in range(N_SLOTS):
        chosen = None
        for retry in range(3):
            rng = random.Random(8030016*100003 + card*7919 + retry*104729)
            ap = _cc_pool(G, card, "A", rng)
            bp = _cc_pool(G, card, "B", rng)
            candidates = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (275*n <= 1000*k < 325*n):
                        continue
                    d = min(a["d"], b["d"])
                    score = (k*d*d, d, -a["cycles"]-b["cycles"])
                    candidates.append((score, a, b, hx, hz, n, k))
            candidates.sort(key=lambda x: x[0], reverse=True)

            finalists = []
            for preliminary, a, b, hx, hz, n, k in candidates[:4]:
                signature = _cc_signature(hx, hz, n)
                if signature in signatures:
                    continue
                ac = _cc_embeddings(a["words"], "A", G.q)
                bc = _cc_embeddings(b["words"], "B", G.q)
                ac = [v for v in ac if not any(
                    (v & row).bit_count() & 1 for row in hx)]
                bc = [v for v in bc if not any(
                    (v & row).bit_count() & 1 for row in hz)]

                # These tests include both sectors and reject stabilizers.
                dx, xwords = _cc_probe(
                    _cc_columns(hz, n), rng, 8, bc, hx)
                dz, zwords = _cc_probe(
                    _cc_columns(hx, n), rng, 8, ac, hz)
                d = min(a["d"], b["d"], dx, dz)
                score = (k*d*d, d, min(dx, dz),
                         -a["cycles"]-b["cycles"])
                finalists.append((score, a, b, hx, hz, n, k,
                                   signature, xwords, zwords))
            finalists.sort(key=lambda x: x[0], reverse=True)

            if finalists:
                item = finalists[0]
                score, a, b, hx, hz, n, k, signature, xw, zw = item
                dx, _ = _cc_probe(_cc_columns(hz, n), rng, 24, xw, hx)
                dz, _ = _cc_probe(_cc_columns(hx, n), rng, 24, zw, hz)
                d = min(score[1], dx, dz)
                assert css_ok(hx, hz)
                assert n == 560 and k >= 160
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                A = tuple(tuple(tuple(cell) for cell in row)
                          for row in a["grid"])
                B = tuple(tuple(tuple(cell) for cell in row)
                          for row in b["grid"])
                signatures.add(signature)
                chosen = (A, B, k, (k*d*d, d))
                break
        if chosen is None:
            raise ValueError("no new target-rate spacing design")
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the card's jointly screened matrices without relabeling."""
    A, B, k, score = _cc_bank(G)[idx]
    return {"cells": {"A": A, "B": B}, "k": k,
            "witness_selection": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without additional draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen odd-central deletion slots provide fifteen inequivalent designs: parent slots 0-14 collapsed to five classes, whose measured minima all remained weight four.
# REFUTED: Unscreened mixed-count experiments improve the parent's principal leaders: slots 15-59 had d2s<=3 and none advanced over its distance-four codes.
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
