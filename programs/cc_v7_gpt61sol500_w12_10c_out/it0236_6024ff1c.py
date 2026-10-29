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
CONCEPT = ("On C9 semidirect C6 with inversion, order 54, shapes (1,3)x(1,3), C3-quotient-balanced noncommuting trinomials should retain five measured distance-eight readings while raising verified k to 228, yielding fitness 27.0222 in cell 0.400.")
GROUP = ("Jump from the order-58 dihedral parent to C9 semidirect C6 with inversion, isomorphic to D9 x C3 and genuinely different from the parent. The order-54 bed is absent from BEDS. Its central C3 factor permits controlled quotient-dependent rank surplus; distance preservation remains a hypothesis.")
SHAPES = ("Keep equal (1,3) sides: n=540, kb=216, guaranteed rate 0.400, and equal 162-variable/54-check components. Require assembled k=228, rate 0.422222, within cell 0.400. Parent measured minima were entirely sector-1 Acol/Bcol witnesses, frequently concentrated around Acol1 or Bcol1, motivating simultaneous component screening.")
FILLING = ("Every cell has one term in each C3-quotient coset, three distinct terms total, so both check weights remain 12. Five parity architectures crossed with four normal-coordinate models define twenty policies. Require verified component rank 52, screen component kernels through four, count capped anchored six-witnesses, and independently deepen actual kernel searches. Assemble only k=228 codes, reject complete kernels through four, verify CSS, and separate emitted codes by uncoloured Tanner invariants. Probe readings are upper estimates, not distance certificates.")
FAMILY_NAME = "r040_C9semiC6_inv_13x13_quotientnorm_rank228_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    # x^9=y^6=1; y*x*y^-1=x^-1.
    # y^2 generates a central C3; this group is D9 x C3.
    if not valid_q(q):
        raise ValueError("C9 semidirect C6 requires order 54")
    els = [(a, b) for b in range(6) for a in range(9)]

    def red(g):
        return (g[0] % 9, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b % 2 == 0 else -1) * c) % 9,
                (b + d) % 6)

    def inv(g):
        a, b = red(g)
        return ((-a if b % 2 == 0 else a) % 9, (-b) % 6)

    def idx(g):
        a, b = red(g)
        return 9 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Actual reflection parities, indexed by the three quotient positions.
# Every pattern mixes rotations and reflections.
_Q54_PARITIES = (
    (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (0, 1, 1), (1, 0, 1), (1, 1, 0),
)

# Five arrangements of A0,A1,A2,B0,B1,B2, crossed with four
# normal-coordinate models. Final invariants, not these labels alone,
# enforce nonisomorphism between emitted slots.
_Q54_CARDS = (
    (0, 1, 2, 3, 4, 5),
    (0, 0, 3, 1, 4, 4),
    (0, 3, 4, 2, 2, 5),
    (1, 1, 5, 0, 3, 5),
    (0, 2, 4, 1, 3, 3),
)


def _screen_columns(rows, nc):
    """Transpose rows obtained from the verified construction helper."""
    cols = [0] * nc
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _screen_four(cols, cap=64):
    """Count capped pair-syndrome collisions; zero excludes kernels <=4.

    All columns screened here have odd weight, excluding odd kernels.
    With distinct nonzero columns, colliding different pairs are disjoint.
    """
    if 0 in cols or len(set(cols)) != len(cols):
        return cap
    seen = set()
    hits = 0
    for a in range(len(cols) - 1):
        x = cols[a]
        for b in range(a + 1, len(cols)):
            syndrome = x ^ cols[b]
            if syndrome in seen:
                hits += 1
                if hits >= cap:
                    return cap
            else:
                seen.add(syndrome)
    return hits


def _screen_component(rows, q):
    cols = _screen_columns(rows, 3 * q)
    four = _screen_four(cols)
    # A representative in each fibre suffices for this overlap objective.
    # These overlaps are a selection heuristic, not a distance bound.
    cycles = 0
    for a in (0, q, 2 * q):
        x = cols[a]
        for b, y in enumerate(cols):
            if a != b:
                overlap = (x & y).bit_count()
                cycles += overlap * (overlap - 1) // 2
    return cols, four, cycles


def _screen_six(cols, q, cap=128):
    """Capped weight-6 component witnesses using anchored triple matching.

    The component lifts have a transitive group action on each fibre.
    Translate any six-support to include one of 0,q,2q, then split it
    into two disjoint triples. A zero result exhausts this test.
    """
    length = len(cols)
    bits = [1 << a for a in range(length)]
    anchored = {}
    for a in (0, q, 2 * q):
        for b in range(length - 1):
            if b == a:
                continue
            ab = cols[a] ^ cols[b]
            mask_ab = bits[a] | bits[b]
            for c in range(b + 1, length):
                if c == a:
                    continue
                key = ab ^ cols[c]
                anchored.setdefault(key, []).append(mask_ab | bits[c])

    hits = 0
    lookup = anchored.get
    for a in range(length - 2):
        for b in range(a + 1, length - 1):
            ab = cols[a] ^ cols[b]
            mask_ab = bits[a] | bits[b]
            for c in range(b + 1, length):
                mates = lookup(ab ^ cols[c])
                if mates:
                    mask = mask_ab | bits[c]
                    for other in mates:
                        if not (mask & other):
                            hits += 1
                            if hits >= cap:
                                return cap
    return hits


from functools import lru_cache
from collections import Counter


@lru_cache(maxsize=1)
def _q54_atlas(G):
    """Compile both cell actions exclusively from verified unit lifts."""
    q = G.q
    mask = (1 << q) - 1
    stub = [[[G.e] for _ in range(3)]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            rows = [row & mask for row in source[:q]]
            blocks.append(tuple(_screen_columns(rows, q)))
        atlas[side] = tuple(blocks)
    return atlas


def _q54_columns(G, atlas, side, grid):
    """F2-linear superposition of scaffold-generated single-term blocks."""
    q = G.q
    cols = [0] * (3 * q)
    for j, cell in enumerate(grid[0]):
        for g in cell:
            block = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(block):
                cols[j * q + h] ^= syndrome
    return cols


def _q54_cycles(cols):
    counts = {}
    total = 0
    for syndrome in cols:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                pair = (b, a)
                old = counts.get(pair, 0)
                total += old
                counts[pair] = old + 1
    return total


def _q54_draw(local, G, kinds, mode, slot, side):
    """One term per C3 coset, with prescribed parity/coordinate policies."""
    row = []
    units = (1, 2, 4, 5, 7, 8)
    for j, kind in enumerate(kinds):
        for retry in range(128):
            parity = _Q54_PARITIES[kind]
            phase = local.randrange(3)
            quotient = [(t + phase) % 3 for t in range(3)]
            bs = [b + 3 * ((parity[t] - b) % 2)
                  for t, b in enumerate(quotient)]
            if mode == 0:
                aa = local.sample(range(9), 3)
            elif mode == 1:
                base, step = local.randrange(9), local.choice(units)
                aa = [(base + step * t) % 9 for t in range(3)]
            elif mode == 2:
                base, step = local.randrange(9), local.choice(units)
                aa = [(base + step * t * t) % 9 for t in range(3)]
            else:
                aa = [local.randrange(9) for _ in range(3)]
            cell = list(zip(aa, bs))
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if G.mul(relatives[0], relatives[1]) == G.mul(
                    relatives[1], relatives[0]):
                continue
            assert len(set(cell)) == 3
            assert {g[1] % 3 for g in cell} == {0, 1, 2}
            row.append(cell)
            break
        else:
            raise ValueError("noncommuting quotient recipe exhausted")
    return [row]


def _q54_probe(cols, local, trials, seeds=()):
    """Search actual classical kernel words, never a distance certificate."""
    width = len(cols)
    order = list(range(width))
    memory = sorted(set(seeds), key=lambda v: (v.bit_count(), v))[:24]
    best = min((v.bit_count() for v in memory), default=width + 1)

    for trial in range(trials):
        local.shuffle(order)
        pivots, words = {}, []
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
            if not syndrome:
                assert word
                words.append(word)
                best = min(best, word.bit_count())

        reducers = sorted(set(words + memory),
                          key=lambda v: (v.bit_count(), v))[:32]
        for sweep in range(2):
            local.shuffle(reducers)
            for i, word in enumerate(words):
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                words[i] = word
                best = min(best, weight)
            reducers = sorted(set(words + memory),
                              key=lambda v: (v.bit_count(), v))[:32]

        short = sorted(set(words + memory),
                       key=lambda v: (v.bit_count(), v))[:40]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    best = min(best, combined.bit_count())
                    discoveries.append(combined)
        if trials >= 12 and trial % 4 == 0:
            small = short[:10]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for t in range(j):
                        combined = pair ^ small[t]
                        if combined:
                            best = min(best, combined.bit_count())
                            discoveries.append(combined)
        memory = sorted(set(discoveries + memory),
                        key=lambda v: (v.bit_count(), v))[:24]
        if best <= 6:
            break

    # With 162 columns and rank 52, every information set supplies
    # nonzero kernel words. A no-witness sentinel is not scored.
    assert memory and best <= width
    return best, tuple(memory)


def _q54_select(local, G, atlas, side, kinds, mode, slot, batch):
    """Optimize a prescribed component, then independently rescreen."""
    pool, seen = [], set()
    q = G.q
    for attempt in range(112 + 32 * batch):
        grid = _q54_draw(local, G, kinds, mode, slot, side)
        labels = tuple(tuple(sorted(cell)) for cell in grid[0])
        if labels in seen:
            continue
        seen.add(labels)
        cols = _q54_columns(G, atlas, side, grid)
        assert all(c.bit_count() == 3 for c in cols)
        if gf2_rank(cols) != q - 2:
            continue
        if _screen_four(cols, cap=1):
            continue
        cycles = _q54_cycles(cols)
        reading, words = _q54_probe(cols, local, 2)
        item = dict(grid=grid, cols=cols, cycles=cycles,
                    reading=reading, words=words, attempt=attempt)
        pool.append(item)
        pool.sort(key=lambda x: (
            -x["reading"], x["cycles"], x["attempt"]))
        del pool[8:]

    for item in pool:
        # Zero exhausts the anchored test; reaching the cap is merely
        # a capped multiplicity, not a complete six-word count.
        six = _screen_six(item["cols"], q, cap=128)
        reading, words = _q54_probe(
            item["cols"], local, 12, item["words"])
        item["six"] = six
        item["reading"] = min(item["reading"], reading,
                              6 if six else len(item["cols"]) + 1)
        item["words"] = words

    def merit(item):
        return (item["reading"], int(item["six"] == 0),
                -item["six"], -item["cycles"])

    pool.sort(key=merit, reverse=True)
    finalists = pool[:4]
    for item in finalists:
        reading, words = _q54_probe(
            item["cols"], local, 32, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
    finalists.sort(key=merit, reverse=True)
    return finalists


def _q54_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariants: inequality certifies nonisomorphism.

    Equal fingerprints are rejected conservatively. No claim that
    fingerprint equality proves graph isomorphism is used.
    """
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length() - 1].append(r)
            source ^= bit
    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = Counter()
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] += multiplicity * other
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            sum(neighbors.get(t, 0) * value
                for t, value in square.items()),
            tuple(sorted(Counter(square.values()).items()))))
    return (tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _q54_bank(G):
    """Deterministic policy bank; never fall back to an emitted orbit copy."""
    assert G.q == 54 and A_SHAPE == B_SHAPE == (1, 3)
    q = G.q
    atlas = _q54_atlas(G)
    mask = (1 << q) - 1
    bank, emitted = [], set()

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        card = _Q54_CARDS[architecture]
        lane = slot // 20
        # Extra slots change only B's relative parity architecture,
        # rather than applying a common transformation to a winner.
        ak = card[:3]
        bk = tuple((kind + lane) % len(_Q54_PARITIES)
                   for kind in card[3:])
        chosen = None

        for batch in range(3):
            local = random.Random(
                BASE_SEED * 100003 + 54022801
                + 100019 * slot + 7919 * batch)
            ap = _q54_select(
                local, G, atlas, "A", ak, mode, slot, batch)
            bp = _q54_select(
                local, G, atlas, "B", bk, mode, slot, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    d = min(a["reading"], b["reading"])
                    key = (d, a["reading"] + b["reading"],
                           -a["six"] - b["six"],
                           -a["cycles"] - b["cycles"])
                    proposals.append((key, a, b))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 540 or k != 228:
                    continue

                # Check the compiled component columns against complete
                # verified lifts before trusting their selection statistics.
                ar = [
                    sum(((row >> (3 * j * q)) & mask) << (j * q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (3 * q)) - 1)
                      for row in hz[:q]]
                assert _screen_columns(ar, 3 * q) == a["cols"]
                assert _screen_columns(br, 3 * q) == b["cols"]
                if _screen_four(_screen_columns(hx, n), cap=1):
                    continue
                if _screen_four(_screen_columns(hz, n), cap=1):
                    continue

                fingerprint = _q54_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                assert css_ok(hx, hz)
                assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
                assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
                assert max(row.bit_count() for row in hx + hz) == 12
                assert 375 * n <= 1000 * k < 425 * n
                assert all(len(cell) == len(set(cell)) == 3
                           for grid in (A, B)
                           for row in grid for cell in row)
                emitted.add(fingerprint)
                frozen_a = tuple(tuple(tuple(cell) for cell in row)
                                 for row in A)
                frozen_b = tuple(tuple(tuple(cell) for cell in row)
                                 for row in B)
                chosen = (frozen_a, frozen_b, key)
                break
            if chosen is not None:
                break

        # An unsuccessful policy is voided, not replaced by a duplicate.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return fresh copies of this policy's rank-verified filling."""
    selected = _q54_bank(G)[idx]
    if selected is None:
        raise ValueError("no new rank-228 filling passed this policy")
    A, B, selection = selected
    return {"idx": idx, "k": 228, "selection": selection,
            "cells": {
                "A": [[list(cell) for cell in row] for row in A],
                "B": [[list(cell) for cell in row] for row in B]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-legal rolled fillings do not reliably avoid distance-2 logicals: parent slots 6, 9, 11 and many others returned d2s=2.
# REFUTED: Reflection-gap scheduling with short-component screening suffices to exceed measured distance eight on the current D29 bed; all five measured current-parent codes returned d30s=8 with thousands of sector-1 weight-eight witnesses.
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
