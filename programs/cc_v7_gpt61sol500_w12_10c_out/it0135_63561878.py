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
CONCEPT = ("Cyclic Z23 in the frozen metacyclic Z1:Z23 encoding, order 23, shapes (2,3)x(2,3): independently optimized four-cycle-free binomial components with divided-minor and dressed-logical screening should exceed five-code fitness 22.9390.")
GROUP = ("Frozen L1 group and order. Parent slots 0 and 20 retained k=35, twelve above the rank bound, but measured distance readings of 14. Keep this exact bed and test a stronger filling and witness-selection rule rather than transferring claims from the higher-scoring nonabelian beds.")
SHAPES = ("Frozen equal (2,3) sides for cell 0.100: n=299, guaranteed k>=23 and rate>=1/13, with two 46-check/69-variable components. The size window permits q<=46, but q=23 stays fixed. Parent weight-14..16 minima predominantly occupied sector-1 Acol/Bcol blocks; slot 20 also touched Arow1/Brow0 and Arow1/Brow1. Verified ranks require k=35, rate 35/299<0.125.")
FILLING = ("Twenty canonical relative-gap cards use two distinct terms per cell, hence check weight ten. Pack components with zero verified four-cycles, independently optimize their kernel witnesses, and seed searches with cyclic minors, verified divided-minor words, rotations and constant-column words. Recombine independently optimized A/B finalists and lower selection readings whenever an actual stabilizer-dressed logical is lighter. Exact ranks preserve the twelve-qubit surplus; all witness readings are upper estimates, not distance certificates.")
FAMILY_NAME = "r010_m1r23s1_23x23_gapcards_dividedminor_dressed_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # metacyclic Z_1 x| Z_23: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 1, 23, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)

_SUPPORT_PROFILES = (
    (2, 2, 2, 2, 2, 2),
    (3, 2, 2, 1, 2, 2),
    (3, 2, 2, 2, 1, 2),
    (3, 2, 1, 1, 2, 3),
    (3, 2, 1, 2, 1, 3),
    (3, 2, 1, 3, 1, 2),
    (3, 3, 1, 1, 1, 3),
    (3, 3, 1, 1, 2, 2),
    (3, 1, 1, 1, 3, 3),
    (3, 3, 2, 1, 1, 2),
)
_SUPPORT_PAIRS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 6),
    (1, 7), (1, 8), (1, 9), (2, 2), (3, 4),
)


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _component_probe(cols, rng, trials=56):
    """Find classical kernel witnesses; absence of small ones is not proof."""
    width = len(cols)
    best = width + 1
    for _ in range(trials):
        order = list(range(width))
        rng.shuffle(order)
        pivots = {}
        deps = []
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, support)
                    break
                old_syndrome, old_support = pivots[p]
                syndrome ^= old_syndrome
                support ^= old_support
            if not syndrome:
                deps.append(support)
                best = min(best, support.bit_count())

        # Elementary basis operations preserve nonzero kernel witnesses.
        for _ in range(3):
            changed = False
            for a in range(len(deps)):
                v = deps[a]
                weight = v.bit_count()
                for b in range(len(deps)):
                    if a == b:
                        continue
                    z = v ^ deps[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                deps[a] = v
                best = min(best, weight)
            if not changed:
                break
    return best


def NEW_CTX(rng, G, idx):
    """Optimize two deliberate gap geometries, then screen their pairing."""
    from itertools import combinations
    q = G.q
    width = 3*q
    mask = (1 << q)-1
    assert q == 23
    sa = sb = [[2, 2, 2], [2, 2, 2]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    element = [None]*q
    for g in G.elements:
        element[g[1]] = g

    def gap_signature(top, bottom):
        # Remove common group-automorphism and row-permutation images
        # from the architecture cards, not merely from their labels.
        images = []
        for multiplier in range(1, q):
            rows = []
            for values in (top, bottom):
                scaled = [(multiplier*v) % q for v in values]
                rows.append(tuple(sorted(min(v, q-v) for v in scaled)))
            images.append(tuple(sorted(rows)))
        return min(images)

    templates = ((1, 2, 3), (1, 2, 4), (1, 2, 5),
                 (1, 3, 5), (1, 3, 7))
    triples = list(combinations(range(1, (q+1)//2), 3))
    cards, signatures = [], set()
    for top in templates:
        for overlap in range(4):
            for bottom in triples:
                if len(set(top) & set(bottom)) != overlap:
                    continue
                signature = gap_signature(top, bottom)
                if signature in signatures:
                    continue
                signatures.add(signature)
                cards.append((top, bottom))
                break
            else:
                raise ValueError("gap-card orbit enumeration failed")
    assert len(cards) == 20
    card = idx % 20
    plans = {"A": cards[card], "B": cards[(card+1) % 20]}

    def packed(plan):
        """Disjoint check-pair difference orbits within each component.

        Column gauges fix the upper anchors; a common lower-row gauge
        fixes the first lower anchor. Different offsets are real voltage
        experiments, not deliberate gauge or automorphism copies.
        """
        top, original_bottom = plan
        for restart in range(128):
            bottom = list(original_bottom)
            rng.shuffle(bottom)
            if any(u == v for u, v in zip(top, bottom)):
                continue
            used, offsets = set(), []
            for j, (u, v) in enumerate(zip(top, bottom)):
                choices = [0] if j == 0 else list(range(q))
                rng.shuffle(choices)
                selected = None
                for offset in choices:
                    differences = {
                        (x-y) % q
                        for x in (0, u)
                        for y in (offset, (offset+v) % q)
                    }
                    if len(differences) == 4 and not differences & used:
                        selected = offset
                        used.update(differences)
                        break
                if selected is None:
                    break
                offsets.append(selected)
            if len(offsets) == 3:
                return [
                    [[element[0], element[u]] for u in top],
                    [[element[t], element[(t+v) % q]]
                     for t, v in zip(offsets, bottom)]
                ]
        raise ValueError("four-cycle-free gap packing failed")

    reference = [
        [[element[0], element[1]] for _ in range(3)]
        for _ in range(2)
    ]

    def component(grid, side):
        # All actions and syndromes are restrictions of VERIFIED lifts.
        A, B = (grid, reference) if side == "A" else (reference, grid)
        hx, hz, _ = lifted_matrices(G, A, B)
        if side == "A":
            rows = []
            for i in range(2):
                for t in range(q):
                    source = hx[(i*3)*q+t]
                    rows.append(sum(
                        ((source >> (j*3*q)) & mask) << (j*q)
                        for j in range(3)))
        else:
            rows = [row & ((1 << width)-1) for row in hz[:2*q]]
        return rows, _component_columns(rows, width)

    def syndrome(word, cols):
        result = 0
        while word:
            bit = word & -word
            result ^= cols[bit.bit_length()-1]
            word ^= bit
        return result

    def norm_reduce(word):
        # Every cell is even: each constant column is a true kernel word.
        # Choose the lighter representative modulo these three words.
        result = 0
        for j in range(3):
            block = (word >> (j*q)) & mask
            if 2*block.bit_count() > q:
                block ^= mask
            result |= block << (j*q)
        return result

    def polynomial_product(a, b):
        result = 0
        while a:
            bit = a & -a
            shift = bit.bit_length()-1
            result ^= ((b << shift) | (b >> (q-shift))) & mask
            a ^= bit
        return result

    def divide_one_plus_x(poly):
        if poly.bit_count() & 1:
            return None
        result, previous = 0, 0
        for t in range(q):
            previous ^= (poly >> t) & 1
            result |= previous << t
        return result

    def structured_words(cols):
        """Cyclic minors and divided minors, each syndrome-verified.

        Polynomial entries are read from verified syndrome columns.
        No independent regular-action implementation is used.
        """
        p = [[(cols[j*q] >> (i*q)) & mask
              for j in range(3)] for i in range(2)]
        blocks = [
            polynomial_product(p[0][1], p[1][2])
            ^ polynomial_product(p[0][2], p[1][1]),
            polynomial_product(p[0][0], p[1][2])
            ^ polynomial_product(p[0][2], p[1][0]),
            polynomial_product(p[0][0], p[1][1])
            ^ polynomial_product(p[0][1], p[1][0]),
        ]
        words = set()
        for division in range(3):
            for shift in range(q):
                word = sum(
                    (((b << shift) | (b >> (q-shift))) & mask)
                    << (j*q) for j, b in enumerate(blocks))
                word = norm_reduce(word)
                if word and not syndrome(word, cols):
                    words.add(word)
            divided = [divide_one_plus_x(b) for b in blocks]
            if any(b is None for b in divided):
                break
            # The odd-length norm changes parity and lies in the kernel.
            blocks = [b ^ mask if b.bit_count() & 1 else b
                      for b in divided]
        for j in range(3):
            word = mask << (j*q)
            assert not syndrome(word, cols)
            words.add(word)
        return sorted(words, key=lambda v: (v.bit_count(), v))[:40]

    def probe(item, trials):
        """Retain actual witnesses across independent information sets."""
        cols = item["cols"]
        memory = list(item["words"])
        best = item["d"]
        order = list(range(width))
        for trial in range(trials):
            rng.shuffle(order)
            pivots, dependencies = {}, []
            for c in order:
                syn, word = cols[c], 1 << c
                while syn:
                    p = syn.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syn, word)
                        break
                    syn ^= old[0]
                    word ^= old[1]
                if not syn:
                    word = norm_reduce(word)
                    if word:
                        dependencies.append(word)
                        best = min(best, word.bit_count())
            words = sorted(set(memory+dependencies),
                           key=lambda v: (v.bit_count(), v))[:56]
            discoveries = []
            for a in range(len(words)):
                word, weight = words[a], words[a].bit_count()
                for reducer in words[:a]:
                    combined = norm_reduce(word ^ reducer)
                    cw = combined.bit_count()
                    if combined and cw < weight:
                        word, weight = combined, cw
                words[a] = word
                best = min(best, weight)
            short = sorted(set(words+memory),
                           key=lambda v: (v.bit_count(), v))[:24]
            for a, word in enumerate(short):
                for other in short[:a]:
                    combined = norm_reduce(word ^ other)
                    if combined:
                        cw = combined.bit_count()
                        best = min(best, cw)
                        if cw <= best+3:
                            discoveries.append(combined)
            memory = sorted(set(words+memory+discoveries),
                            key=lambda v: (v.bit_count(), v))[:40]
        item["words"] = memory
        item["d"] = min(item["d"], best)

    def component_key(item):
        weights = sorted(word.bit_count() for word in item["words"])
        return (item["d"],
                -sum(w <= item["d"]+1 for w in weights),
                sum(min(w, item["d"]+5) for w in weights[:12]))

    pools = {"A": [], "B": []}
    seen = {"A": set(), "B": set()}

    def admit(grid, side):
        fingerprint = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
        if fingerprint in seen[side]:
            return
        seen[side].add(fingerprint)
        rows, cols = component(grid, side)
        if gf2_rank(rows) != 2*q-2:
            return
        if _component_four_cycles(cols):
            return
        words = structured_words(cols)
        item = {"grid": grid, "cols": cols, "words": words,
                "d": min(word.bit_count() for word in words)}
        probe(item, 4)
        pools[side].append(item)
        pools[side].sort(key=component_key, reverse=True)
        del pools[side][6:]

    for side in ("A", "B"):
        for attempt in range(40):
            admit(packed(plans[side]), side)
        for step in range(20):
            if not pools[side]:
                admit(packed(plans[side]), side)
                continue
            parent = pools[side][rng.randrange(min(3, len(pools[side])))]
            grid = [[list(cell) for cell in row] for row in parent["grid"]]
            pressure = [
                1 + sum(((word >> (j*q)) & mask).bit_count()
                        for word in parent["words"][:8])
                for j in range(3)]
            j = rng.choices(range(3), weights=pressure, k=1)[0]
            delta = rng.randrange(1, q)
            grid[1][j] = [
                element[(g[1]+delta) % q] for g in grid[1][j]]
            # Restore the common lower-row gauge after the real repair.
            anchor = grid[1][0][0][1]
            grid[1] = [
                [element[(g[1]-anchor) % q] for g in cell]
                for cell in grid[1]]
            admit(grid, side)
        if not pools[side]:
            raise ValueError("no full-rank four-cycle-free component")
        for item in pools[side]:
            probe(item, 16)
        pools[side].sort(key=component_key, reverse=True)
        pools[side] = pools[side][:3]
        for item in pools[side]:
            probe(item, 64)
        pools[side].sort(key=component_key, reverse=True)

    def row_basis(rows):
        basis = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                old = basis.get(p)
                if old is None:
                    basis[p] = word
                    break
                word ^= old
        return basis

    def outside(word, basis):
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def dressed_reading(hx, hz, aw, bw, n):
        """Only syndrome-verified, nonstabilizer representatives count."""
        best = n+1
        for side, seeds, checks, stabilizers in (
                ("A", aw, hx, hz), ("B", bw, hz, hx)):
            basis = row_basis(stabilizers)
            for seed in sorted(set(seeds), key=int.bit_count)[:8]:
                for fixed in range(3):
                    if side == "A":
                        word = sum(
                            ((seed >> (j*q)) & mask)
                            << ((j*3+fixed)*q) for j in range(3))
                    else:
                        word = seed << (fixed*3*q)
                    if any((word & row).bit_count() & 1 for row in checks):
                        continue
                    if not outside(word, basis):
                        continue
                    best = min(best, word.bit_count())
                    for sweep in range(3):
                        chosen, weight = word, word.bit_count()
                        nearby = []
                        for row in stabilizers:
                            combined = word ^ row
                            cw = combined.bit_count()
                            nearby.append((cw, row))
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                        nearby.sort(key=lambda entry: entry[0])
                        reducers = [row for _, row in nearby[:12]]
                        for a, row in enumerate(reducers):
                            for other in reducers[:a]:
                                combined = word ^ row ^ other
                                cw = combined.bit_count()
                                if 0 < cw < weight:
                                    chosen, weight = combined, cw
                        if chosen == word:
                            break
                        word = chosen
                        best = min(best, weight)
        return best

    winner = None
    for a in pools["A"]:
        for b in pools["B"]:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != q+12 or 8*k >= n:
                continue
            logical = dressed_reading(
                hx, hz, a["words"], b["words"], n)
            d = min(a["d"], b["d"], logical)
            score = (k*d*d, d, a["d"]+b["d"],
                     component_key(a)[1]+component_key(b)[1])
            if winner is None or score > winner[0]:
                winner = (score, A, B, hx, hz, k)
    if winner is None:
        raise ValueError("no pairing retained k=35 in cell 0.100")
    score, A, B, hx, hz, k = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert k == 35 and 8*k < 13*q
    assert all(len(cell) == 2 and len(set(cell)) == 2
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "card": card, "screen": score,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently optimized and jointly screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: Coupled four-cycle shortlisting and six component-probed finalists remove the weight-14 bottleneck on Z23 (2,3)x(2,3); measured current-parent slots 0 and 20 still had d30s=14 despite k=35, and slot 20 included a mixed-sector logical.
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
