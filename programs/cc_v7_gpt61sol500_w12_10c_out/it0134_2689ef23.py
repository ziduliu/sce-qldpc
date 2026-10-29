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
CONCEPT = ("Heisenberg H3, order 27, A=(2,3), B=(3,5): even degree-four columns, noncentral cell differences and joint logical screening should exceed the cell-0.150 fitness record 29.4994.")
GROUP = ("Jump from Dic11 to the upper-unitriangular Heisenberg group over F3. H3 has exponent three, centre of order three and no involutions. The cyclic Z23 (2,3)x(3,5) bed measured four distance-15 codes; q=27 increases component lengths and introduces noncommutative labels. Possible short order-three relations are screened rather than assumed harmless.")
SHAPES = ("Target 0.150: n=21*27=567, rank-count kb=54, and component variable/check counts 81/54 and 135/81. All-even cells annihilate the augmentation check matrices and add at least 19 rank deficits, giving k>=73 and rate>=0.12875. Verified ranks require 73<=k<=99. Parent limiting Z witnesses touched both A columns, particularly Bcol1-3; the second A check row and degree-four B columns replace both weak geometries.")
FILLING = ("Twenty canonical even support pairs use zero-, two- and four-term cells, four terms per component column and row totals at most eight, so both CSS check weights are at most twelve. Avoid central-separated terms and prefer distinct directed differences. Optimize actual component four-cycles using verified lifts; require component kernels free through weight six, exact target rates and full-CSS logical exclusion through four. Stabilizer-tested component witnesses and cross-information-set full-code searches cover both sectors. Combined-Tanner invariants reject emitted orbit copies. Sampled witness readings remain upper bounds, not certificates.")
FAMILY_NAME = "r015_heis3_23x35_even_noncentral_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 27

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Heisenberg H3 requires order 27")
    # (a,b,c) represents [[1,a,c],[0,1,b],[0,0,1]] over F3.
    els = [(a, b, c) for c in range(3)
           for b in range(3) for a in range(3)]

    def red(g):
        return tuple(v % 3 for v in g)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + d) % 3, (b + e) % 3,
                (c + f + a*e) % 3)

    def inv(g):
        a, b, c = red(g)
        return ((-a) % 3, (-b) % 3, (a*b - c) % 3)

    def idx(g):
        a, b, c = red(g)
        return a + 3*b + 9*c

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


from collections import Counter
from functools import lru_cache


@lru_cache(maxsize=1)
def _h27_support_plans():
    """Canonical weighted supports, before assigning any group labels."""
    from itertools import product, combinations_with_replacement, permutations

    def profiles(rows, width):
        types = tuple(t for t in product((0, 2, 4), repeat=rows)
                      if sum(t) == 4)
        orders = tuple(permutations(range(rows)))
        found = set()
        for cols in combinations_with_replacement(types, width):
            totals = [sum(c[i] for c in cols) for i in range(rows)]
            if min(totals) < 4 or max(totals) > 8:
                continue
            canonical = min(
                tuple(sorted(tuple(c[i] for i in order) for c in cols))
                for order in orders)
            found.add(canonical)
        return [
            tuple(tuple(c[i] for c in cols) for i in range(rows))
            for cols in sorted(found)
        ]

    def fours(grid):
        return sum(v == 4 for row in grid for v in row)

    plans = [(a, b) for a in profiles(*A_SHAPE)
             for b in profiles(*B_SHAPE)]
    plans.sort(key=lambda pair: (
        fours(pair[0]) + fours(pair[1]), fours(pair[0]),
        -min(map(sum, pair[1])), -min(map(sum, pair[0])), pair))
    if len(plans) < 20:
        raise ValueError("fewer than twenty canonical even support pairs")
    return tuple(plans[:20])


def _d11_support(card):
    """Legacy helper name; all counts now belong to the H3 bed."""
    a, b = _h27_support_plans()[card % 20]
    sa, sb = [list(r) for r in a], [list(r) for r in b]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(row[j] for row in sa) == 4
               for j in range(A_SHAPE[1]))
    assert all(sum(row[j] for row in sb) == 4
               for j in range(B_SHAPE[1]))
    return sa, sb


def _d11_a_cell(rng, G, weight, mode):
    """Distinct quotient projections exclude central relative elements.

    Directed differences are only a label-construction criterion.
    Actual cycle counts and kernels come from the verified lift.
    """
    if weight == 0:
        return []
    quotient = [(a, b) for a in range(3) for b in range(3)]
    best = None
    for attempt in range(64):
        projections = rng.sample(quotient, weight)
        cell = [(a, b, rng.randrange(3)) for a, b in projections]
        differences = [
            G.mul(g, G.inv(h)) for g in cell for h in cell if g != h]
        cost = len(differences) - len(set(differences))
        if best is None or cost < best[0]:
            best = (cost, cell)
        if cost == 0:
            break
    cell = best[1]
    assert len(cell) == weight and len(set(cell)) == weight
    assert len({g[:2] for g in cell}) == weight
    return cell


def _d11_draw(rng, G, sa, sb, mode):
    A = [[_d11_a_cell(rng, G, count, mode) for count in row]
         for row in sa]
    B = [[_d11_a_cell(rng, G, count, mode) for count in row]
         for row in sb]
    return A, B


def _d11_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _d11_components(hx, hz, q):
    """Extract both new-shape components from the verified LP matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bm = (1 << (cb * q)) - 1
    br = [row & bm for row in hz[:rb * q]]
    return (_d11_columns(ar, ca * q), _d11_columns(br, cb * q))


def _h27_free6(cols):
    """Exactly exclude nonzero component kernel words through weight six.

    A word of weight at most six splits into two subsets of size <=3.
    A collision between distinct subsets supplies that kernel word.
    """
    seen = {0}
    for s in cols:
        if s in seen:
            return False
        seen.add(s)
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in seen:
                return False
            seen.add(s)
    for i, x in enumerate(cols):
        for j in range(i):
            xy = x ^ cols[j]
            for h in range(j):
                s = xy ^ cols[h]
                if s in seen:
                    return False
                seen.add(s)
    return True


def _d11_floor4(cols):
    """Exact kernel screening through four; five is only a sentinel."""
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, ci in enumerate(cols):
        for cj in cols[:i]:
            syndrome = ci ^ cj
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _d11_cycles(cols):
    overlaps = {}
    total = 0
    for col in cols:
        checks = []
        while col:
            low = col & -col
            checks.append(low.bit_length() - 1)
            col ^= low
        for i, a in enumerate(checks):
            for b in checks[:i]:
                pair = (b, a)
                old = overlaps.get(pair, 0)
                total += old
                overlaps[pair] = old + 1
    return total


def _d11_kernel_reading(cols, rng, trials, accept=None, initial=None):
    """Exact kernel witnesses with cross-information-set recombination.

    Full-code calls reject stabilizers through accept. Their initial
    reading must already come from a validated quantum logical witness.
    """
    best = len(cols) + 1 if initial is None else initial
    witnesses, memory = set(), []
    order = list(range(len(cols)))

    def observe(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        if weight > best + 2:
            return
        if accept is not None and not accept(word):
            return
        best = min(best, weight)
        if weight <= best + 2:
            witnesses.add(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                observe(word)

        words.sort(key=int.bit_count)
        discoveries = set(memory + words[:32])
        cutoff = best + 4
        for i, word in enumerate(words):
            for other in words[:i]:
                combined = word ^ other
                if combined and combined.bit_count() <= cutoff:
                    observe(combined)
                    discoveries.add(combined)

        short = sorted(discoveries, key=lambda v: (v.bit_count(), v))[:40]
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined and combined.bit_count() <= cutoff:
                    observe(combined)
                    discoveries.add(combined)

        reducers = words + memory
        for start in short[:8]:
            word = start
            for _ in range(4):
                improved, weight = word, word.bit_count()
                for other in reducers:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        improved, weight = trial, tw
                if improved == word:
                    break
                word = improved
                observe(word)
                discoveries.add(word)

        memory = sorted(discoveries,
                        key=lambda v: (v.bit_count(), v))[:40]
        witnesses = set(sorted(
            witnesses, key=lambda v: (v.bit_count(), v))[:64])
        if best <= 4:
            break
    short = sorted(witnesses, key=lambda v: (v.bit_count(), v))[:32]
    return best, short


def _d11_basis(rows):
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


def _d11_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _d11_small_logical(checks, stabilizers, n):
    """Exactly detect a non-stabilizer logical of weight one through four."""
    basis = _d11_basis(stabilizers)
    cols = _d11_columns(checks, n)
    representatives = {0: 0}
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        previous = representatives.get(syndrome)
        if previous is None:
            representatives[syndrome] = word
        elif _d11_outside(word ^ previous, basis):
            return True
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            previous = representatives.get(syndrome)
            if previous is None:
                representatives[syndrome] = word
            elif _d11_outside(word ^ previous, basis):
                return True
    return False


def _d11_logical_reading(words, side, hx, hz, q, n):
    """Validate new-shape embeddings, including constant-block witnesses."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    checks, stabilizers = (hx, hz) if side == "A" else (hz, hx)
    basis = _d11_basis(stabilizers)
    width = ca if side == "A" else cb
    # Even cells annihilate these component vectors. They contribute
    # only after full commutation and stabilizer membership are checked.
    seeds = [mask << (j * q) for j in range(width)]
    best = n + 1
    for word in list(words) + seeds:
        weight = word.bit_count()
        if not word or weight >= best:
            continue
        for block in range(cb if side == "A" else ca):
            if side == "A":
                full = 0
                for j in range(ca):
                    full |= ((word >> (j * q)) & mask) << (
                        (j * cb + block) * q)
            else:
                full = word << (block * cb * q)
            if any((full & row).bit_count() % 2 for row in checks):
                raise ValueError("verified component embedding failed")
            if _d11_outside(full, basis):
                best = weight
                break
    return best


def _d11_fingerprint(hx, hz, n):
    """Combined-Tanner invariant, ignoring even the X/Z check colouring.

    Different fingerprints imply nonisomorphic incidence graphs.
    Matching fingerprints conservatively reject some nonisomorphic codes.
    No label, group encoding, or protograph ordering enters this invariant.
    """
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, row in enumerate(rows):
        while row:
            low = row & -row
            incidents[low.bit_length() - 1].append(r)
            row ^= low

    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = {}
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] = square.get(t, 0) + multiplicity*other
        triangles = sum(neighbors.get(t, 0)*value
                        for t, value in square.items())
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            triangles,
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _d11_bank(G):
    """Pure deterministic H3 bank; every emitted graph has a new invariant."""
    bank, fingerprints = [], set()
    assert G.q == 27

    for slot in range(N_SLOTS):
        card = slot % 20
        mode = (card + slot // 20) % 4
        sa, sb = _d11_support(card)
        rng = random.Random(15027033 + 100003*slot)
        chosen = None
        targets = {
            "A": [(i, j) for i, row in enumerate(sa)
                  for j, count in enumerate(row) if count],
            "B": [(i, j) for i, row in enumerate(sb)
                  for j, count in enumerate(row) if count],
        }

        for batch in range(3):
            finalists = []
            for attempt in range(40):
                A, B = _d11_draw(rng, G, sa, sb, mode)
                cells = {"A": A, "B": B}
                hx, hz, n = lifted_matrices(G, A, B)
                ac, bc = _d11_components(hx, hz, G.q)
                cycle = [_d11_cycles(ac), _d11_cycles(bc)]

                # Replace whole cells without changing their even counts.
                # Both component actions are read from verified matrices.
                for step in range(16):
                    side = "A" if step % 2 == 0 else "B"
                    which = 0 if side == "A" else 1
                    i, j = rng.choice(targets[side])
                    old = cells[side][i][j]
                    cells[side][i][j] = _d11_a_cell(
                        rng, G, len(old), mode)
                    tx, tz, tn = lifted_matrices(G, A, B)
                    tac, tbc = _d11_components(tx, tz, G.q)
                    cost = _d11_cycles(tac if which == 0 else tbc)
                    if cost < cycle[which] or (
                            cost == cycle[which] and rng.random() < 0.4):
                        hx, hz, n, ac, bc = tx, tz, tn, tac, tbc
                        cycle[which] = cost
                    else:
                        cells[side][i][j] = old

                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 567 or not 73 <= k <= 99:
                    continue
                if min(_d11_floor4(ac), _d11_floor4(bc)) < 5:
                    continue
                da, wa = _d11_kernel_reading(ac, rng, 8)
                db, wb = _d11_kernel_reading(bc, rng, 8)
                cycles = sum(cycle)
                floor = min(da, db, 27)
                key = (k*floor*floor/n, floor, da + db, -cycles)
                finalists.append({
                    "key": key, "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "wa": wa, "wb": wb, "cycles": cycles,
                })
                finalists.sort(key=lambda item: item["key"], reverse=True)
                del finalists[8:]

            deep = []
            for item in finalists:
                # This is an exact component-kernel filter, not a
                # declaration of quantum distance.
                if not _h27_free6(item["ac"]):
                    continue
                if not _h27_free6(item["bc"]):
                    continue
                da, wa = _d11_kernel_reading(item["ac"], rng, 56)
                db, wb = _d11_kernel_reading(item["bc"], rng, 56)
                wa = sorted(set(wa + item["wa"]),
                            key=lambda v: (v.bit_count(), v))[:48]
                wb = sorted(set(wb + item["wb"]),
                            key=lambda v: (v.bit_count(), v))[:48]
                dz = _d11_logical_reading(
                    wa, "A", item["hx"], item["hz"], G.q, item["n"])
                dx = _d11_logical_reading(
                    wb, "B", item["hx"], item["hz"], G.q, item["n"])
                if max(dx, dz) > item["n"]:
                    continue
                # Unlike a raw classical minimum, this ceiling comes
                # exclusively from validated quantum logical witnesses.
                d = min(dx, dz)
                key = (item["k"]*d*d/item["n"], d,
                       dx + dz, -item["cycles"])
                deep.append((key, item))

            deep.sort(key=lambda pair: pair[0], reverse=True)
            joint = []
            for key, item in deep[:4]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                xs, zs = _d11_basis(hx), _d11_basis(hz)
                upper = key[1]
                dx, _ = _d11_kernel_reading(
                    _d11_columns(hz, n), rng, 16,
                    accept=lambda word, basis=xs:
                        _d11_outside(word, basis),
                    initial=upper)
                dz, _ = _d11_kernel_reading(
                    _d11_columns(hx, n), rng, 16,
                    accept=lambda word, basis=zs:
                        _d11_outside(word, basis),
                    initial=upper)
                d = min(upper, dx, dz)
                joint.append(((item["k"]*d*d/n, d,
                               key[2], -item["cycles"]), item))

            joint.sort(key=lambda pair: pair[0], reverse=True)
            for key, item in joint:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _d11_small_logical(hx, hz, n):
                    continue
                if _d11_small_logical(hz, hx, n):
                    continue
                fingerprint = _d11_fingerprint(hx, hz, n)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert 125*n <= 1000*item["k"] < 175*n
                assert all(len(cell) in (0, 2, 4)
                           and len(cell) == len(set(cell))
                           for M in (item["A"], item["B"])
                           for row in M for cell in row)
                chosen = (item["A"], item["B"], fingerprint)
                break
            if chosen is not None:
                break

        if chosen is None:
            bank.append(None)
        else:
            A, B, fingerprint = chosen
            fingerprints.add(fingerprint)
            bank.append((A, B))

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign one independently optimized, incidence-distinct card."""
    selected = _d11_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh rate-legal filling passed logical-four screening")
    A, B = selected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: D8 (2,7)x(2,8) distinct monomial relatives and short-syndrome selection did not break the distance-four plateau: all five measured readings were 4, with both logical types entirely in sector 1.
# REFUTED: Dic11 (1,2)x(3,5) degree tradeoffs and component selection yield five measured readings >=14; measured slots 11 and 28 remained at 13, while the other three measured readings were 14.
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
