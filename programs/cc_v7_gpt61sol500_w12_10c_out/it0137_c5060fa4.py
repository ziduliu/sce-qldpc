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
CONCEPT = ("On cyclic Z13, order 13, A=(2,6) and B=(2,7), two-check degree-three A components and a girth-seven degree-two B subgraph, selected against exact component words through six, should yield five measured readings >=7 and improve cell-0.450 fitness over 16.0.")
GROUP = ("Keep the parent's cyclic construction rule, resizing order 23 to 13 because the new size factor 46 would otherwise exceed 600. The existing Z13 (2,6)x(2,7) bed reached fitness 16.0 with five distance-six readings. This motivates testing a different support geometry, not assuming a group advantage.")
SHAPES = ("Target cell 0.450 with A=(2,6), B=(2,7), q=13: n=598, kb=260 and guaranteed rate 10/23=0.434783. Components have 78 variables/26 checks and 91 variables/26 checks. Parent minima were entirely sector-1 Acol/Bcol families, broadly across B columns and especially Acol1. A second A check row removes the single-row commuting-trinomial construction; exact full ranks retain 0.425<=k/n<0.475.")
FILLING = ("Twenty cards combine three canonical degree-three A support profiles with prescribed mixed-column B binomial gaps. Both component row totals are nine and column totals <=3, so HX and HZ weights are <=12. B degree-two columns are one isolated binomial per row and one matching column, with relative gaps 1 and 5; this subgraph has girth seven at q=13. Verified unit-cell lifts compile component syndromes. Exact weight-five/six word counts and actual kernel witnesses optimize independent labels, followed by verified joint ranks, CSS and logical-four screening. Unequal combined-Tanner fingerprints exclude emitted orbit copies. Finite component screens do not certify quantum distance.")
FAMILY_NAME = "r045_Z13_26x27_girth7_core_exact6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    if q != 13:
        raise ValueError("this cyclic bed requires order 13")
    els = [(a,) for a in range(q)]

    def red(g):
        return (g[0] % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _q85_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _q85_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return True
        v ^= basis[p]
    return False


def _q85_short_screen(hx, hz, n):
    """Exact existence test for nontrivial CSS logicals of weight <=4.

    Every such support splits into two supports of size <=2.
    Equal-syndrome collisions are tested modulo the actual stabilizer
    row space. Keeping one representative per syndrome is sufficient:
    if all collisions with it are stabilizers, their pairwise differences
    are stabilizers too.
    """
    cycles = 0
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = _q85_basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Count actual Tanner 4-cycles, not protograph coincidences.
        overlaps = {}
        for col in cols:
            ids = []
            v = col
            while v:
                bit = v & -v
                ids.append(bit.bit_length() - 1)
                v ^= bit
            for a in range(len(ids)):
                for b in range(a):
                    key = (ids[b], ids[a])
                    old = overlaps.get(key, 0)
                    cycles += old
                    overlaps[key] = old + 1

        representatives = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            mask = bits[i]
            if syndrome in representatives:
                v = mask ^ representatives[syndrome]
                if _q85_outside(v, basis):
                    return v.bit_count(), cycles
            else:
                representatives[syndrome] = mask

        for i in range(n):
            si, bi = cols[i], bits[i]
            for j in range(i):
                syndrome = si ^ cols[j]
                mask = bi | bits[j]
                if syndrome in representatives:
                    v = mask ^ representatives[syndrome]
                    if _q85_outside(v, basis):
                        return v.bit_count(), cycles
                else:
                    representatives[syndrome] = mask

    # Sentinel: no logical of weight <=4, not a distance estimate.
    return 5, cycles


def _z23_columns(rows, length):
    cols = [0] * length
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _z23_components(hx, hz, q):
    """Extract every component check row from the verified LP lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for row in hx[(i * cb) * q:(i * cb + 1) * q]:
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return (_z23_columns(ar, ca * q),
            _z23_columns(br, cb * q))


def _z23_short(cols, through_five=False):
    """Exact existence screening, not a heuristic distance reading.

    Singles/pairs cover weight <=4. Triples against their representatives
    cover weight <=5 without storing the much larger triple dictionary.
    The sentinel is 5 or 6 when the requested weights were excluded.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        old = seen.get(syndrome)
        if old is not None:
            return (old ^ word).bit_count()
        seen[syndrome] = word
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is not None:
                return (old ^ word).bit_count()
            seen[syndrome] = word
    if not through_five:
        return 5
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            for t in range(j):
                old = seen.get(syndrome ^ cols[t])
                if old is not None:
                    return (old ^ word ^ bits[t]).bit_count()
    return 6


def _z23_reading(cols, rng, trials):
    """Upper-bound readings from actual nonzero component-kernel words."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for i in order:
            syndrome, word = cols[i], 1 << i
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
                best = min(best, word.bit_count())
        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for other in words[:i]:
                best = min(best, (word ^ other).bit_count())
        small = words[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for t in range(j):
                    best = min(best, (pair ^ small[t]).bit_count())
        if best <= 4:
            break
    return best


def _z23_cycles(cols):
    total = 0
    for i, col in enumerate(cols):
        for other in cols[:i]:
            overlap = (col & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _z23_candidate(rng, G, sb, ap_count):
    """Fresh labels for the card, never an automorphism-copy generator."""
    q, cb = G.q, B_SHAPE[1]
    A = [[]]
    for j in range(A_SHAPE[1]):
        if j < ap_count:
            step = rng.randrange(1, q)
            offsets = (0, step, (2 * step) % q)
        else:
            while True:
                u, v = rng.sample(range(1, q), 2)
                # None of the three terms is an AP midpoint.
                if (v != (2 * u) % q and u != (2 * v) % q
                        and (u + v) % q != 0):
                    offsets = (0, u, v)
                    break
        base = G.pick(rng)
        A[0].append([G.mul(base, (a,)) for a in offsets])

    # Distinct relative labels prevent repeated all-monomial columns.
    relatives = rng.sample(range(q), cb)
    B = [[None] * cb for _ in range(2)]
    for j, relative in enumerate(relatives):
        top = G.pick(rng)
        bottom = G.mul(top, (relative,))
        B[0][j], B[1][j] = [top], [bottom]
    for i in range(2):
        for j in range(cb):
            if sb[i][j] == 2:
                delta = (rng.randrange(1, q),)
                B[i][j].append(G.mul(B[i][j][0], delta))
    return A, B


from functools import lru_cache
from collections import Counter
from itertools import combinations_with_replacement


@lru_cache(maxsize=1)
def _r045_atlas(G):
    """F2-linear contributions compiled exclusively from verified lifts."""
    tables = {}
    q = G.q
    mask = (1 << q) - 1
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit, stub = [[[g]]], [[[G.e]]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_z23_columns(
                [row & mask for row in source[:q]], q)))
        tables[side] = tuple(blocks)
    return tables


def _r045_compiled(G, side, grid):
    """Superpose verified unit contributions; no regular action is invented."""
    q = G.q
    atlas = _r045_atlas(G)[side]
    cols = [0] * (len(grid[0]) * q)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                contribution = atlas[G.idx(g)]
                for t, syndrome in enumerate(contribution):
                    cols[j*q + t] ^= syndrome << (i*q)
    return cols


def _r045_stats(cols):
    """Exact exclusion/counting through six after exclusion through four.

    Seven is an exclusion sentinel, not a distance reading.
    A five-word has ten 2+3 partitions; a six-word has ten
    unordered 3+3 partitions. No short word is counted as a logical.
    """
    low = _z23_short(cols)
    if low != 5:
        return (low, len(cols)**3, len(cols)**3)

    pairs = set()
    for i, ci in enumerate(cols):
        for cj in cols[:i]:
            pairs.add(ci ^ cj)

    odd = all(c.bit_count() & 1 for c in cols)
    triples = {}
    five_hits = six_hits = 0
    for i, ci in enumerate(cols):
        for j in range(i):
            pair = ci ^ cols[j]
            for t in range(j):
                syndrome = pair ^ cols[t]
                if not odd and syndrome in pairs:
                    five_hits += 1
                previous = triples.get(syndrome, 0)
                six_hits += previous
                triples[syndrome] = previous + 1
    assert five_hits % 10 == six_hits % 10 == 0
    n5, n6 = five_hits // 10, six_hits // 10
    return (5 if n5 else 6 if n6 else 7, n5, n6)


def _r045_support(model):
    a_columns = (
        ((2, 1),) * 3 + ((1, 2),) * 3,
        ((3, 0), (2, 1)) + ((1, 2),) * 4,
        ((3, 0), (0, 3)) + ((2, 1),) * 2 + ((1, 2),) * 2,
    )[model]
    b_columns = (
        (2, 0), (0, 2), (1, 1),
        (2, 1), (2, 1), (1, 2), (1, 2),
    )
    sa = [[c[i] for c in a_columns] for i in range(2)]
    sb = [[c[i] for c in b_columns] for i in range(2)]
    assert all(sum(row) == 9 for grid in (sa, sb) for row in grid)
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _r045_gap_cards():
    pairs = tuple(combinations_with_replacement((2, 3, 4, 6), 2))
    cards = [(a, b) for a in pairs for b in pairs]
    cards.sort(key=lambda c: (
        -len(set(c[0] + c[1])),
        sum(x == y for x, y in zip(c[0], c[1])),
        c))
    return tuple(cards[:20])


def _r045_draw(rng, G, side, counts, gaps):
    grid = [[[] for _ in row] for row in counts]
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            if count == 1:
                grid[i][j] = [G.pick(rng)]
            elif count == 2:
                grid[i][j] = [G.e, (rng.randrange(1, G.q),)]
            elif count == 3:
                u, v = rng.sample(range(1, G.q), 2)
                grid[i][j] = [G.e, (u,), (v,)]
    if side == "B":
        # The degree-two subgraph is the generalized Petersen graph
        # on 26 checks: top cycle gap 1, bottom cycle gap 5, matching.
        # It has girth seven. These reference columns are not mutated.
        grid[0][0], grid[1][0] = [G.e, (1,)], []
        grid[0][1], grid[1][1] = [], [G.e, (5,)]
        grid[0][2] = grid[1][2] = [G.e]
        for i, positions in enumerate(((3, 4), (5, 6))):
            for j, gap in zip(positions, gaps[i]):
                grid[i][j] = [G.e, (gap,)]
    return grid


def _r045_mutate(rng, G, side, grid, counts):
    out = [[list(cell) for cell in row] for row in grid]
    if side == "B":
        # Preserve both the girth-seven core and the card's four gaps.
        i, j = rng.choice(((1, 3), (1, 4), (0, 5), (0, 6)))
        out[i][j] = [G.pick(rng)]
    else:
        positions = [(i, j) for i, row in enumerate(counts)
                     for j, count in enumerate(row) if count]
        i, j = rng.choice(positions)
        cell = out[i][j]
        t = 0 if len(cell) == 1 else rng.randrange(1, len(cell))
        cell[t] = rng.choice([g for g in G.elements if g not in cell])
    return out


def _r045_pool(rng, G, side, counts, gaps, quota):
    """Independent restarts plus mutations minimizing exact short-word counts."""
    pool, seen = [], set()
    elite, elite_key = None, None
    for attempt in range(quota):
        if elite is not None and attempt % 6:
            grid = _r045_mutate(rng, G, side, elite, counts)
        else:
            grid = _r045_draw(rng, G, side, counts, gaps)
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        cols = _r045_compiled(G, side, grid)
        stats = _r045_stats(cols)
        cycles = _z23_cycles(cols)
        key = (stats[0], -stats[1], -stats[2], -cycles)
        if elite_key is None or key > elite_key or (
                key == elite_key and rng.random() < 0.5):
            elite, elite_key = grid, key
        if stats[0] < 5:
            continue
        pool.append({
            "grid": grid, "cols": cols, "stats": stats,
            "cycles": cycles, "key": key,
        })
        pool.sort(key=lambda item: item["key"], reverse=True)
        del pool[5:]

    for item in pool:
        reading = _z23_reading(item["cols"], rng, 24)
        if item["stats"][0] <= 6:
            reading = min(reading, item["stats"][0])
        item["reading"] = reading
    pool.sort(key=lambda item: (
        item["reading"], item["key"]), reverse=True)
    return pool[:4]


def _r045_fingerprint(hx, hz, n):
    """Invariant of the combined, uncoloured Tanner incidence graph.

    Unequal fingerprints prove nonisomorphism; equal fingerprints
    are conservatively rejected, without claiming they prove equality.
    """
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    overlaps = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlaps[(min(weights[i], weights[j]),
                      max(weights[i], weights[j]),
                      (row & rows[j]).bit_count())] += 1
    neighborhoods = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            neighborhoods[bit.bit_length() - 1].append(weights[i])
            row ^= bit
    local = Counter(tuple(sorted(v)) for v in neighborhoods)
    return (tuple(sorted(weights)), tuple(sorted(overlaps.items())),
            tuple(sorted(local.items())))


@lru_cache(maxsize=1)
def _r045_bank(G):
    """Fresh independently optimized cards, never transported orbit copies."""
    assert G.q == 13
    gap_cards = _r045_gap_cards()
    bank, fingerprints = [], set()
    for idx in range(N_SLOTS):
        sa, sb = _r045_support(idx % 3)
        gaps = gap_cards[idx % len(gap_cards)]
        chosen = None
        for retry in range(2):
            rng = random.Random(
                450132627 + idx * 100003 + retry * 7919)
            ap = _r045_pool(rng, G, "A", sa, gaps, 112)
            bp = _r045_pool(rng, G, "B", sb, gaps, 192)
            candidates = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if n != 598 or not (425*n <= 1000*k < 475*n):
                        continue
                    ac, bc = _z23_components(hx, hz, G.q)
                    # Confirm compiled syndromes against complete verified
                    # matrices before allowing them to influence selection.
                    assert ac == a["cols"] and bc == b["cols"]
                    d = min(a["reading"], b["reading"])
                    merit_d = min(d, 10)  # selection cap, not a witness
                    key = (
                        k * merit_d * merit_d, merit_d,
                        min(a["stats"][0], b["stats"][0]),
                        -(a["stats"][1] + b["stats"][1]),
                        -(a["stats"][2] + b["stats"][2]),
                        -(a["cycles"] + b["cycles"]),
                    )
                    candidates.append((key, A, B, hx, hz, n, k))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for key, A, B, hx, hz, n, k in candidates[:8]:
                low, _ = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                fingerprint = _r045_fingerprint(hx, hz, n)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= 12
                assert all(len(cell) <= 5 and len(cell) == len(set(cell))
                           for grid in (A, B)
                           for row in grid for cell in row)
                fingerprints.add(fingerprint)
                chosen = (A, B, k, key)
                break
            if chosen is not None:
                break
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Every cell of the resized shape comes from this screened card."""
    chosen = _r045_bank(G)[idx]
    if chosen is None:
        raise ValueError("no invariant-distinct target-rate card passed screening")
    A, B, k, key = chosen
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "screen": key}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Q8 x Z5 order-20 quaternion-axis binomials plus exact-four candidate selection did not break the weight-four plateau: all five measured codes had d30s=4, with minima entirely in sector 1.
# REFUTED: Repeating the Z23 support/AP recipes with fresh sampling guarantees distinct codes: slots 20,22,23,26,27 and many later slots were marked isomorphic duplicates.
# REFUTED: Zero/trinomial redistribution and exact weight-six component-word counting on Z23 (1,3)x(2,7) improve fitness above 15.6520; all five measured leaders instead retained k=230 and d30s=6.
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
