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
CONCEPT = ("Frobenius C7 semidirect C3, action 2, order 21, shapes (3,4)x(3,4): rank-two nontrivial C3-quotient components with four distinct projective column directions should reduce worst-basis circuit failures relative to the parent's rank-one singleton/triple construction.")
GROUP = ("Frozen under L1. This bed previously scored 6.9109 with five zero-failure measurements, whereas the current rank-one quotient filling scored 4.9356 with only four successful measurements. Retain the group for a controlled filling comparison; its normal C7 subgroup supplies quotient structure, not a demonstrated group-only circuit advantage.")
SHAPES = ("Frozen equal (3,4) components have 63 checks and 84 variables each: n=525, guaranteed k>=21 and rate 0.040. The size window permits q<=23, but this group fixes q=21. Exact ranks retain rate below 0.125 in cell 0.100. Parent failures were mostly diffuse across Acol/Bcol and Arow/Brow blocks; slot53's Acol0/Bcol1 enrichment of 1.4x came from only eight Z failures, so both components are redesigned rather than one alleged hotspot.")
FILLING = ("Twenty primary quotient/support contracts distinguish all four columns over F4 while keeping nontrivial quotient rank two. Quotient-zero cells use same-coset doubletons or one term in each of the three cosets; optional nonzero doubletons/triples alter augmentation rank and exposure. Three-coset triples exclude subgroup-coset projectors. Row totals <=6 and column totals <=5 keep CSS weights <=11. Generic primary ranks predict k=33,36,41, but verified full ranks decide acceptance. Screen connectivity, short ambiguities, capped sampled protection, small boundaries, balanced same-basis cycles and quantum logicals <=4. These pL hypotheses target slot12's 100/129750 Z failures, collective flips and the missing ASC measurement, not distance as fitness or a certified scheduler.")
FAMILY_NAME = "r010_frobenius21_34x34_c3ranktwo_projective_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 21

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Frobenius group requires q=21")
    # x^7=y^3=1, y*x*y^-1=x^2; 2^3=1 modulo 7.
    els = [(a, b) for b in range(3) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 3)

    def mul(g, h):
        return ((g[0] + pow(2, g[1], 7) * h[0]) % 7,
                (g[1] + h[1]) % 3)

    def inv(g):
        b = (-g[1]) % 3
        return ((-pow(2, b, 7) * g[0]) % 7, b)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 4)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


def _f21_columns(hx, hz, q):
    """Project component columns from the VERIFIED lifted matrices.

    A: sector 1 of X checks at B-column zero.
    B: sector 1 of Z checks at A-column zero.
    No independent left/right-lift implementation is used.
    """
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    ac = [0] * (ca * q)
    bc = [0] * (cb * q)
    amask = sum(((1 << q) - 1) << (j * cb * q)
                for j in range(ca))
    bmask = (1 << (cb * q)) - 1
    for i in range(ra):
        for r in range(q):
            bits = hx[(i * cb) * q + r] & amask
            check = 1 << (i * q + r)
            while bits:
                low = bits & -bits
                c = low.bit_length() - 1
                ac[(c // (cb * q)) * q + c % q] |= check
                bits ^= low
    for a in range(rb):
        for r in range(q):
            bits = hz[a * q + r] & bmask
            check = 1 << (a * q + r)
            while bits:
                low = bits & -bits
                bc[low.bit_length() - 1] |= check
                bits ^= low
    return ac, bc


def _f21_stats(cols, nchecks):
    """Exact short-word counts, Tanner four-cycles and connectivity."""
    parent = list(range(nchecks))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checkpairs = {}
    c4 = 0
    for col in cols:
        rows = []
        bits = col
        while bits:
            low = bits & -bits
            rows.append(low.bit_length() - 1)
            bits ^= low
        if rows:
            for r in rows[1:]:
                parent[root(r)] = root(rows[0])
        for t, r in enumerate(rows):
            for s in rows[t + 1:]:
                pair = (r, s)
                old = checkpairs.get(pair, 0)
                c4 += old
                checkpairs[pair] = old + 1

    colset = set(cols)
    repeated = len(cols) - len(colset)
    pairseen = {}
    words3 = words4 = 0
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            syndrome = x ^ y
            words3 += int(syndrome in colset)
            old = pairseen.get(syndrome, 0)
            words4 += old
            pairseen[syndrome] = old + 1
    components = len({root(r) for r in range(nchecks)})
    return (repeated, words3, words4), c4, components


def _f21_traps(cols):
    """Small boundaries of three-variable sets; repeated triple syndromes.

    The first statistic is a trapping-set heuristic. The second also
    detects weight-six component words when shorter words are absent.
    Neither statistic is asserted to predict circuit pL exactly.
    """
    boundary = repeated = 0
    seen = {}
    count = len(cols)
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, count):
                syndrome = pair ^ cols[k]
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
                old = seen.get(syndrome, 0)
                repeated += old
                seen[syndrome] = old + 1
    return boundary, repeated


def NEW_CTX(rng, G, idx):
    """Optimize a prescribed rank-two quotient/support contract."""
    # Signature: (number of F4-zero cells, number of odd zero cells,
    #             term count at the optional nonzero reinforcement).
    # The first six policies have trivial-character ranks one or two.
    # Their twenty unordered primary pairings are deliberate contracts.
    policies = (
        (2, 2, 0),
        (2, 2, 3),
        (3, 3, 0),
        (2, 1, 0),
        (2, 1, 3),
        (3, 2, 0),
        # Additional harness slots test different augmentation/support
        # contracts, not transformed copies of the primary fillings.
        (2, 0, 0),
        (2, 0, 3),
        (3, 0, 0),
        (3, 1, 0),
        (2, 1, 2),
        (2, 2, 2),
    )
    cards = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    assert len(cards) == 20
    cards.extend((a, b) for a in range(len(policies))
                 for b in range(a, len(policies))
                 if (a, b) not in cards)
    if idx >= len(cards):
        raise ValueError("slot budget exceeds rank-two contract bank")
    pa, pb = (policies[p] for p in cards[idx])

    def supports(signature):
        zeros, odd, reinforcement = signature
        grid = [[1] * 4 for _ in range(3)]
        for i in range(zeros):
            grid[i][i] = 3 if i < odd else 2
        if reinforcement:
            assert zeros == 2
            grid[2][2] = reinforcement
        return grid

    sa, sb = supports(pa), supports(pb)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(11, W_MAX)

    def quotient(signature):
        """F4 entries, with 2=w and 3=w^2=w+1.

        Every column lies in u+v+t=0. The first two are independent;
        all four columns have different projective directions.
        Zeros occur in distinct rows and columns, fitting the budget.
        """
        zeros = signature[0]
        if zeros == 3:
            columns = ((0, 1, 1), (1, 0, 1),
                       (1, 1, 0), (1, 2, 3))
        else:
            columns = ((0, 1, 1), (1, 0, 1),
                       (1, 2, 3), (1, 3, 2))
        assert all((u ^ v ^ t) == 0 for u, v, t in columns)
        return [[column[i] for column in columns] for i in range(3)]

    def draw(signature):
        """Realize the prescribed quotient, optimizing fresh C7 voltages.

        F4-zero odd cells project to 1+y+y^2, not to identity.
        F4-zero even cells project to zero in the entire C3 algebra.
        Nonzero cells represent their specified F4 entry, with the
        augmentation independently controlled by their term counts.
        """
        counts = supports(signature)
        field = quotient(signature)
        phase = {1: 0, 2: 1, 3: 2}
        grid = []
        for i in range(3):
            row = []
            for j in range(4):
                value, count = field[i][j], counts[i][j]
                if value == 0 and count == 3:
                    a, c = rng.randrange(7), rng.randrange(7)
                    # Exclude g*(1+h+h^2) for an order-three h.
                    # This defect test is only a local algebraic screen;
                    # all matrix decisions still use verified lifts.
                    choices = [d for d in range(7)
                               if (d - a - 3 * (c - a)) % 7]
                    d = rng.choice(choices)
                    cell = [(a, 0), (c, 1), (d, 2)]
                elif value == 0:
                    b = rng.randrange(3)
                    a, c = rng.sample(range(7), 2)
                    cell = [(a, b), (c, b)]
                else:
                    b = phase[value]
                    if count == 1:
                        cell = [(rng.randrange(7), b)]
                    elif count == 2:
                        # w^(b+1)+w^(b+2)=w^b in F4.
                        cell = [(rng.randrange(7), (b + 1) % 3),
                                (rng.randrange(7), (b + 2) % 3)]
                    else:
                        pair_phase = (b + rng.choice((1, 2))) % 3
                        a, c = rng.sample(range(7), 2)
                        cell = [(rng.randrange(7), b),
                                (a, pair_phase), (c, pair_phase)]
                assert len(cell) == count and len(set(cell)) == count
                row.append(cell)
            grid.append(row)
        return grid

    def kernel_upper(cols, trials=64):
        """Verified kernel witnesses from randomized fundamental circuits.

        Combining two sampled basis words also gives genuine kernel
        words. A large result is not a lower-bound distance certificate.
        """
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if syndrome == 0 and word:
                    best = min(best, word.bit_count())
                    for previous in basis:
                        best = min(best, (word ^ previous).bit_count())
                    basis.append(word)
            # Greedy reductions remain genuine kernel witnesses. They
            # make a missed short word less likely without claiming a
            # lower-bound distance certificate.
            for sweep in range(2):
                rng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for previous in basis[:i]:
                        candidate = word ^ previous
                        w = candidate.bit_count()
                        if candidate:
                            best = min(best, w)
                        if candidate and w < weight:
                            word, weight = candidate, w
                    basis[i] = word
            if best <= 14:
                break
        return best

    def low4(checks, stabilizers, n):
        """Exact quantum-logical existence test through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        bits = [1 << c for c in range(n)]
        seen = {0: 0}
        for c, syndrome in enumerate(columns):
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = bits[c]
            elif nontrivial(bits[c] ^ old):
                return True
        for c in range(n):
            for d in range(c):
                syndrome = columns[c] ^ columns[d]
                word = bits[c] ^ bits[d]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    def same_basis_cycles(rows):
        """Exact four-cycles in a verified full same-basis Tanner graph.

        This is a decoder/scheduling heuristic, not an ASC feasibility
        test and not a circuit-level failure-rate measurement.
        """
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    # Quotient algebra predicts generic primary k=33,36,41. It does
    # not replace full CSS ranks, nor exclude nonabelian rank deficits.
    # Every accepted code must remain in the open-ended 0.100 cell.
    for batch in range(2):
        shortlist = []
        for trial in range(160):
            A, B = draw(pa), draw(pb)
            hx, hz, n = lifted_matrices(G, A, B)
            ac, bc = _f21_columns(hx, hz, G.q)
            ast = _f21_stats(ac, 3 * G.q)
            bst = _f21_stats(bc, 3 * G.q)
            if ast[2] != 1 or bst[2] != 1:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (21 <= k <= 65 and 8 * k < n):
                continue
            bad = max(ast[0], bst[0])
            totals = tuple(x + y for x, y in zip(ast[0], bst[0]))
            cycles = (max(ast[1], bst[1]), ast[1] + bst[1])
            base = (bad, totals, cycles)
            item = (base, trial, k, A, B, ac, bc, hx, hz, n)
            shortlist.append(item)
            shortlist.sort(key=lambda entry: (entry[0], entry[1]))
            del shortlist[10:]

        finalists = []
        for item in shortlist:
            base, trial, k, A, B, ac, bc, hx, hz, n = item
            at, bt = _f21_traps(ac), _f21_traps(bc)
            da, db = kernel_upper(ac), kernel_upper(bc)
            distance = min(da, db)
            boundary = (max(at[0], bt[0]), at[0] + bt[0])
            sixwords = (max(at[1], bt[1]), at[1] + bt[1])
            cx, cz = same_basis_cycles(hx), same_basis_cycles(hz)
            full_cycles = (max(cx, cz), cx + cz, abs(cx - cz))
            # The earlier zero-failure codes screened at 18..21; the
            # rank-one parent screened at 14 yet failed in every measured
            # Z basis. Use a capped warning threshold, then balanced
            # ambiguity/exposure screens, rather than chasing distance.
            # Higher k is a late ceiling benefit, not a reliability test.
            key = (base[0], base[1], int(distance < 18),
                   -min(distance, 18), int(sixwords[0] > 0),
                   boundary, full_cycles, sixwords, base[2],
                   -k, abs(da - db), trial)
            finalists.append((key, item))
        finalists.sort(key=lambda entry: entry[0])

        for key, item in finalists:
            _, _, k, A, B, _, _, hx, hz, n = item
            if low4(hx, hz, n) or low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            return {"idx": idx, "architecture": (pa, pb), "k": k,
                    "screen": key, "cells": {"A": A, "B": B}}

    raise ValueError("No connected rank-two quotient design passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete rank-two-contract cell without late changes."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's large q=58 single-row uniform-label seed reliably suppresses collective circuit failures; measured worst-basis s=3.17-3.28, widespread logical flips and many sector-1 weight-four witnesses contradict this.
# REFUTED: Increasing d2s necessarily improves this bed's circuit score: zero-failure slot 16 had d2s=21,k=23,s=6.874, below zero-failure slot 28 with d2s=18,k=30,s=6.989; their sample ceilings limited both.
# REFUTED: Rank-one C3 quotient singleton/triple fillings preserve this bed's zero-failure reliability at higher k: all four measured codes had Z failures, worst-basis s=5.584..6.681, and selected slot18 had no ASC schedule, reducing fitness to 4.9356.
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
