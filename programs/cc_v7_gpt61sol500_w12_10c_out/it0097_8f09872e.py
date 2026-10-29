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
CONCEPT = ("GL(2,3), order 48, shapes (1,2)x(1,2): equal four-term components vanishing in the determinant-quotient algebra, screened against component and full-CSS logical witnesses, should exceed the parent's five-code fitness 38.25 in cell 0.250.")
GROUP = ("Keep the exact GL(2,F3) matrix-group rule and order 48. The parent's (1,2)x(2,5) bed produced five measured readings of twelve and fitness 38.25. That evidence motivates retaining the group, not assuming a distance advantage. The determinant homomorphism to C2 supplies a controlled source of rank surplus on the new shape.")
SHAPES = ("Target cell 0.250 using equal 96-variable/48-check components: (1,2)x(1,2) gives n=240 and kb=48, guaranteed rate 0.200. Every cell projects to zero in F2[C2], giving determinant-coset check dependencies; verified ranks, rather than an assumed surplus, enforce 0.225<=k/n<0.275. Parent measured minima were entirely sector 1, spanning both A columns and all five B columns. Replace the asymmetric B component and screen both sectors.")
FILLING = ("Twenty cards cross five determinant-occupancy architectures with four relative-order palettes. Every cell has four distinct terms, with either zero or two determinant-minus terms, so each projects to zero in F2[C2]. Both CSS row weights are exactly twelve. Exclude relative involutions and require noncommuting relatives. Optimize fresh verified lifts by short component dependencies and kernel probes; refine finalists with exact component tests through six, exact full logical exclusion through four, and stabilizer-tested full-CSS witness probes. Invariant-distinct final matrices prevent emitted orbit copies. Probe readings are upper-bound heuristics, not distance certificates.")
FAMILY_NAME = "r025_GL2F3_12x12_detquotient_fullprobe_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # GL(2,F3), encoded by row-major entries (a,b,c,d).
    if not valid_q(q):
        raise ValueError("GL(2,F3) has order 48")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = red(g)
        e,f,u,v = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a,b,c,d = red(g)
        determinant = (a*d-b*c) % 3
        # Each nonzero element of F3 is its own multiplicative inverse.
        return ((determinant*d) % 3, (-determinant*b) % 3,
                (-determinant*c) % 3, (determinant*a) % 3)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


from functools import lru_cache

# A0,A1,B0,B1 determinant-minus term counts. Each cell has four terms.
# Both quotient coefficients are even, hence zero over F2.
# Pure and mixed cells are architectural choices, not gauge copies.
_DET_CARDS = (
    (2,2,2,2),
    (0,2,2,2),
    (0,0,2,2),
    (0,2,0,2),
    (0,2,0,0),
)
_ORDER_CARDS = ((3,4), (4,8), (6,8), (3,8))


def _element_orders(G):
    orders = {}
    for g in G.elements:
        value = G.e
        for exponent in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break
        else:
            raise ValueError("element order did not divide group order")
    return orders


def _det(g):
    return (g[0]*g[3] - g[1]*g[2]) % 3


@lru_cache(maxsize=1)
def _label_data(G):
    orders = _element_orders(G)
    palettes = {
        order: tuple(g for g in G.elements if orders[g] == order)
        for order in (3,4,6,8)
    }
    compatible = {}
    for h in G.elements:
        hi = G.inv(h)
        compatible[h] = frozenset(
            g for g in G.elements
            if g != h and orders[G.mul(hi, g)] > 2)
    return orders, palettes, compatible


def _designed_grid(rng, G, mode, side, negative):
    """Four-term cells with a prescribed zero determinant projection."""
    orders, palettes, compatible = _label_data(G)
    row = []
    for j, nf in enumerate(negative):
        wanted = (_ORDER_CARDS[mode][j] if side == "A"
                  else (3,4,6,8)[(mode+j+1) % 4])
        # Order-eight relatives have determinant minus and cannot occur
        # in a cell containing only determinant-plus elements.
        if nf == 0:
            wanted = (3,4,6)[(mode+j+(side == "B")) % 3]
        selected = None
        for restart in range(192):
            anchors = [g for g in G.elements
                       if nf or _det(g) == 1]
            anchor = rng.choice(anchors)
            second = G.mul(anchor, rng.choice(palettes[wanted]))
            cell = [anchor, second]
            if second not in compatible[anchor]:
                continue
            if sum(_det(g) == 2 for g in cell) > nf:
                continue
            if sum(_det(g) == 1 for g in cell) > 4-nf:
                continue
            while len(cell) < 4:
                need_minus = nf - sum(_det(g) == 2 for g in cell)
                need_plus = 4-nf - sum(_det(g) == 1 for g in cell)
                legal = [
                    g for g in G.elements
                    if ((need_minus and _det(g) == 2)
                        or (need_plus and _det(g) == 1))
                    and all(g in compatible[h] for h in cell)
                ]
                if not legal:
                    break
                cell.append(rng.choice(legal))
            if len(cell) != 4:
                continue
            relatives = [G.mul(G.inv(anchor), g) for g in cell[1:]]
            if not any(G.mul(g,h) != G.mul(h,g)
                       for t, g in enumerate(relatives)
                       for h in relatives[:t]):
                continue
            assert sum(_det(g) == 2 for g in cell) == nf
            selected = cell
            break
        if selected is None:
            raise ValueError("determinant/order cell recipe exhausted")
        row.append(selected)
    return [row]


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _short_floor(columns):
    """Exact component-kernel minimum through four; five means unexcluded."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_reading(columns, rng, trials):
    """Find actual classical kernel words; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
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
                best = min(best, word.bit_count())
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # A few short basis triples supplement pairwise information sets.
        small = sorted(words, key=int.bit_count)[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = min(best, (pair ^ small[h]).bit_count())
        if best <= 4:
            break
    return best


def _no_kernel_le6(columns):
    """Exact component test, not a quantum-distance certificate.

    Every support of weight <=6 partitions into two different subsets
    of size <=3. A syndrome collision therefore detects such a support.
    Only existence is tested; no witness census is claimed.
    """
    seen = {0}
    for syndrome in columns:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in seen:
                return False
            seen.add(syndrome)
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
    return True


def _component_data(hx, hz, q):
    """Restrict the VERIFIED lift; do not implement regular actions."""
    mask = (1 << q) - 1
    cb = B_SHAPE[1]
    ar = [
        (row & mask) | (((row >> (cb*q)) & mask) << q)
        for row in hx[:q]
    ]
    bm = (1 << (cb*q)) - 1
    br = [row & bm for row in hz[:B_SHAPE[0]*q]]
    return ar, br, _columns(ar, 2*q), _columns(br, cb*q)


def _basis(rows):
    pivots = {}
    for v in rows:
        while v:
            p = v.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = v
                break
            v ^= old
    return pivots


def _outside(v, pivots):
    while v:
        p = v.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        v ^= old
    return False


def _logical_le4(checks, stabilizers, n):
    """Exact existence of a non-stabilizer commuting support through four."""
    cols = _columns(checks, n)
    basis = _basis(stabilizers)
    seen = {0: 0}
    for i, s in enumerate(cols):
        v = 1 << i
        old = seen.get(s)
        if old is None:
            seen[s] = v
        elif _outside(v ^ old, basis):
            return True
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            v = (1 << i) | (1 << j)
            old = seen.get(s)
            if old is None:
                seen[s] = v
            elif _outside(v ^ old, basis):
                return True
    return False


def _logical_probe(cols, rng, trials, stabilizers=()):
    """Actual classical/quantum witnesses, including both LP sectors.

    Empty stabilizers gives a classical kernel probe. Otherwise every
    counted witness is tested outside the exact stabilizer row span.
    A sentinel denotes search failure, never a distance certificate.
    """
    basis = _basis(stabilizers)
    generators = list(stabilizers)
    best = len(cols)+1
    order = list(range(len(cols)))

    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            s, v = cols[c], 1 << c
            while s:
                p = s.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s, v)
                    break
                s ^= old[0]
                v ^= old[1]
            if not s:
                dependencies.append(v)

        dependencies.sort(key=int.bit_count)
        logical = [v for v in dependencies if _outside(v, basis)]
        for v in logical:
            best = min(best, v.bit_count())
        reducers = dependencies[:48] + generators
        for v in logical[:16]:
            weight = v.bit_count()
            for sweep in range(3):
                changed = False
                for other in reducers:
                    z = v ^ other
                    zw = z.bit_count()
                    if 0 < zw < weight and _outside(z, basis):
                        v, weight = z, zw
                        changed = True
                if not changed:
                    break
            best = min(best, weight)

        small = dependencies[:24]
        for i, v in enumerate(small):
            for other in small[:i]:
                z = v ^ other
                if z and z.bit_count() < best and _outside(z, basis):
                    best = z.bit_count()
        if best <= 4:
            break
    return best


def _walk_signature(rows, n):
    """Invariant under independent check and qubit permutations."""
    cols = _columns(rows, n)
    adjacency = [{} for _ in rows]
    for s in cols:
        active = []
        while s:
            bit = s & -s
            active.append(bit.bit_length()-1)
            s ^= bit
        for a, i in enumerate(active):
            for j in active[:a]:
                adjacency[i][j] = adjacency[i].get(j, 0)+1
                adjacency[j][i] = adjacency[j].get(i, 0)+1
    profiles = tuple(sorted(
        tuple(sorted(neighbors.values())) for neighbors in adjacency))
    triangles, fourth = 0, 0
    for i, neighbors in enumerate(adjacency):
        walks = {}
        for j, ij in neighbors.items():
            for h, jh in adjacency[j].items():
                walks[h] = walks.get(h, 0)+ij*jh
        triangles += sum(ij*walks.get(j, 0)
                         for j, ij in neighbors.items())
        fourth += sum(v*v for v in walks.values())
    return (tuple(sorted(r.bit_count() for r in rows)),
            tuple(sorted(c.bit_count() for c in cols)),
            profiles, triangles, fourth)


def _code_signature(hx, hz, n, k):
    # Include a global X/Z exchange in the equivalence convention.
    return (k, tuple(sorted((_walk_signature(hx, n),
                             _walk_signature(hz, n)))))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Independently optimized cards with invariant-distinct final matrices."""
    assert G.q == 48
    sa = sb = [[4,4]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    bank, fingerprints = [], set()

    for idx in range(N_SLOTS):
        recipe = idx % 20
        mode = recipe % 4
        occupancy = _DET_CARDS[recipe // 4]
        rng = random.Random(25048121207 + 100003*idx)
        chosen = None

        def merit(item):
            d = min(item["da"], item["db"])
            return (item["floor"], item["k"]*d*d,
                    d, item["da"]+item["db"], -item["cycles"])

        for batch in range(3):
            pool = []
            quota = 192 + 32*batch
            for attempt in range(quota):
                A = _designed_grid(rng, G, mode, "A", occupancy[:2])
                B = _designed_grid(rng, G, mode, "B", occupancy[2:])
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (225*n <= 1000*k < 275*n):
                    continue
                ar, br, ac, bc = _component_data(hx, hz, G.q)
                fa, fb = _short_floor(ac), _short_floor(bc)
                item = dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k, ac=ac, bc=bc,
                    floor=min(fa, fb),
                    da=(fa if fa < 5 else _logical_probe(ac, rng, 6)),
                    db=(fb if fb < 5 else _logical_probe(bc, rng, 6)),
                    cycles=_component_cycles(ar)+_component_cycles(br))
                pool.append(item)
                if len(pool) >= 36:
                    break

            finalists = sorted(pool, key=merit, reverse=True)[:6]
            for item in finalists:
                ac, bc = item["ac"], item["bc"]
                item["clean6"] = (
                    _no_kernel_le6(ac) and _no_kernel_le6(bc))
                item["da"] = min(
                    item["da"], _logical_probe(ac, rng, 40))
                item["db"] = min(
                    item["db"], _logical_probe(bc, rng, 40))
                hx, hz, n = item["hx"], item["hz"], item["n"]
                item["clean4"] = (
                    not _logical_le4(hx, hz, n)
                    and not _logical_le4(hz, hx, n))
                # X: kernel HZ modulo span HX; Z: the reverse.
                item["dx"] = _logical_probe(
                    _columns(hz, n), rng, 16, stabilizers=hx)
                item["dz"] = _logical_probe(
                    _columns(hx, n), rng, 16, stabilizers=hz)

            def deep_merit(item):
                d = min(item["da"], item["db"],
                        item["dx"], item["dz"])
                return (item["clean4"], item["k"]*d*d,
                        d, item["clean6"],
                        min(item["dx"], item["dz"]),
                        item["da"]+item["db"], -item["cycles"])

            for item in sorted(finalists, key=deep_merit, reverse=True):
                hx, hz, n, k = (
                    item["hx"], item["hz"], item["n"], item["k"])
                fingerprint = _code_signature(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(r.bit_count() for r in hx+hz) <= W_MAX
                assert 225*n <= 1000*k < 275*n
                assert all(len(cell) == len(set(cell)) == 4
                           for grid in (item["A"], item["B"])
                           for row in grid for cell in row)
                fingerprints.add(fingerprint)
                chosen = (item["A"], item["B"], k, deep_merit(item))
                break
            if chosen is not None:
                break
        # A failed recipe voids only its own slot, not every later slot.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the card's optimized cells; never emit a relabeled earlier code."""
    chosen = _design_bank(G)[idx]
    if chosen is None:
        raise ValueError("no fresh determinant-quotient design in target cell")
    A, B, k, score = chosen
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "selection": score}


def CELL(rng, G, side, i, j, ctx):
    """All labels and counts were selected jointly in NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: Rebalanced nonuniform term counts preserve Dic21's uniform-cubic distance readings; every distinct nonuniform-count parent slot had d2s=5, whereas measured uniform-cubic slots reached d30s=9 or 10.
# REFUTED: Noncommuting order palettes and component short-six screening on GL(2,3), (1,2)x(2,5), escape the measured distance-twelve plateau; all five measured parent codes had d30s=12, with both types entirely in sector 1.
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
