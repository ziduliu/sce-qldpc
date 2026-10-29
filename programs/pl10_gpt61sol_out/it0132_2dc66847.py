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
CONCEPT = ("For semidihedral SD16=C8 semidirect C2 with action x->x^3, order 16, shapes (2,5)x(2,6), screened degree-three component columns and complementary order-four/order-eight cell differences should raise worse-basis circuit fitness above 5.0.")
GROUP = ("Jump from cyclic Z17 to nonabelian SD16: x^8=y^2=1 and y*x*y^-1=x^3. This group supplies order-four and order-eight relative directions. Its F2 group algebra is local because SD16 is a 2-group, and odd-augmentation elements are units, motivating concentrated trinomials. The related rolled order-16 bed scored about 4.0; neither nonabelianness nor the unit property establishes a circuit advantage. This SD16 bed is absent from BEDS.")
SHAPES = ("Keep (2,5)x(2,6), targeting cell 0.350 with n=544, kb=192 and rate 6/17=0.352941. Order 16 is the largest order admitted by this fixed group. A has 32 checks on 80 component variables; B has 32 checks on 96. Parent slot31 had 146/750 Z failures, including worst logical #28, and slot50 had 125/1000 Z failures versus 100/13500 X failures, including worst Z logical #33. Both affected all 204 logicals. Their Acol/Bcol enrichments were only about 1.1-1.2x, not a proven localized cause.")
FILLING = ("Twenty contracts cross five A support architectures with four B architectures and complementary binomial-order policies. Every component column has three terms; A rows total 7/8 and B rows 9/9, so HX<=11 and HZ=12. The six-round CNOT budget is 23616: additional exposure relative to the rolled parent is an explicit trade for uniform protection. Compose cached singleton actions obtained from lifted_matrices, optimize short component kernels and cycles, and rank finalists by weight-six multiplicity and capped sampled kernel witnesses. Actual full lifts require k=192, screen quantum logicals through weight four and balance same-basis cycle burden. Additional harness batches change trinomial ratio contracts. These screens target collective Z-limited failures; measured circuit pL must validate them.")
FAMILY_NAME = "r035_SD16_25x26_col3_complementary_orders_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SD16 requires order 16")
    # (a,b)=x^a*y^b; x^8=y^2=1, y*x*y^-1=x^3.
    # The action is valid since 3^2=1 modulo 8.
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (3 if b else 1) * c) % 8, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-(3 if b else 1) * a) % 8, b)

    def idx(g):
        a, b = red(g)
        return a + 8 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty architecture/order contracts; only final cells are emitted."""
    import math

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    design, batch = idx % 20, idx // 20
    atop = (
        (2, 2, 1, 1, 1),
        (0, 2, 2, 2, 1),
        (3, 0, 2, 1, 1),
        (3, 0, 0, 2, 2),
        (3, 3, 0, 0, 1),
    )[design // 4]
    btop = (
        (2, 2, 2, 1, 1, 1),
        (0, 2, 2, 2, 2, 1),
        (3, 0, 2, 2, 1, 1),
        (3, 0, 0, 2, 2, 2),
    )[design % 4]
    counts = {
        "A": [list(atop), [3 - t for t in atop]],
        "B": [list(btop), [3 - t for t in btop]],
    }
    assert _no_dead_lines(counts["A"], counts["B"])
    assert _grid_weights(counts["A"], counts["B"]) == 12

    # Complementary order sets avoid matching binomial ratio orders
    # across the two sides in the first two contracts.
    order_contracts = (
        ((8,), (4,)),
        ((4,), (8,)),
        ((8,), (4, 8)),
        ((4, 8), (8,)),
    )
    palettes = dict(zip(("A", "B"), order_contracts[design % 4]))
    tri_policy = batch % 3

    orders = {}
    for g in G.elements:
        h = G.e
        for t in range(1, q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                orders[G._key(g)] = t
                break
        else:
            raise ValueError("missing element order")
    deltas = {
        side: [g for g in G.elements
               if orders[G._key(g)] in palettes[side]]
        for side in ("A", "B")
    }

    def draw_cell(side, count):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            g = G.pick(rng)
            return [g, G.mul(g, rng.choice(deltas[side]))]
        for _ in range(1024):
            cell = rng.sample(G.elements, count)
            left = [G.mul(G.inv(u), v) for u in cell
                    for v in cell if u != v]
            right = [G.mul(u, G.inv(v)) for u in cell
                     for v in cell if u != v]
            relative = [orders[G._key(g)] for g in left]
            if min(relative) < 4:
                continue
            if len({G._key(g) for g in left}) != count * (count - 1):
                continue
            if len({G._key(g) for g in right}) != count * (count - 1):
                continue
            if tri_policy == 1 and 8 not in relative:
                continue
            if tri_policy == 2 and max(relative) != 4:
                continue
            return cell
        raise ValueError("trinomial ratio contract exhausted")

    def draw(side):
        return [[draw_cell(side, t) for t in row]
                for row in counts[side]]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    # Cache exactly the actions supplied by the verified helper.
    # No independently implemented left/right permutation convention.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        x, z, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        actions["A"][G._key(g)] = tuple(row & mask for row in x)
        actions["B"][G._key(g)] = tuple(row & mask for row in z)

    def component(side, grid):
        rows = []
        for cells in grid:
            block = [0] * q
            for j, cell in enumerate(cells):
                for g in cell:
                    term = actions[side][G._key(g)]
                    for r, bits in enumerate(term):
                        block[r] ^= bits << (j * q)
            rows.extend(block)
        return rows

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def fast_metric(rows, width):
        cols = columns(rows, width)
        assert all(c.bit_count() == 3 for c in cols)
        hist = {}
        for c in cols:
            hist[c] = hist.get(c, 0) + 1
        twos = sum(v * (v - 1) // 2 for v in hist.values())

        pair_hist = {}
        collisions = 0
        check_pairs = {}
        adjacency = [0] * len(rows)
        four = 0
        for a, col in enumerate(cols):
            for b in range(a):
                syn = col ^ cols[b]
                old = pair_hist.get(syn, 0)
                collisions += old
                pair_hist[syn] = old + 1
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a, u in enumerate(nodes):
                for v in nodes[:a]:
                    pair = (v, u)
                    old = check_pairs.get(pair, 0)
                    four += old
                    check_pairs[pair] = old + 1
                    adjacency[u] |= 1 << v
                    adjacency[v] |= 1 << u

        triangles = 0
        for u in range(len(rows)):
            later = adjacency[u] & ~((1 << (u + 1)) - 1)
            while later:
                bit = later & -later
                v = bit.bit_length() - 1
                triangles += ((adjacency[u] & adjacency[v])
                              >> (v + 1)).bit_count()
                later ^= bit
        # Exact weight-four multiplicity when columns are distinct.
        # The six-cycle expression is exact only when four==0.
        n4 = collisions // 3 if not twos else collisions
        return (twos, n4, four, max(0, triangles - width)), cols

    def kernel_six(cols, fast):
        """Exact weight-six count after weight-two/four exclusion."""
        if fast[0] or fast[1]:
            return 10**9
        triples = {}
        collisions = 0
        for a, x in enumerate(cols):
            for b in range(a):
                pair = x ^ cols[b]
                for c in range(b):
                    syn = pair ^ cols[c]
                    old = triples.get(syn, 0)
                    collisions += old
                    triples[syn] = old + 1
        # Odd column degrees exclude all odd-weight kernel words.
        # With no weight-two/four word, colliding triples are disjoint.
        assert collisions % 10 == 0
        return collisions // 10

    def kernel_probe(cols, trials):
        """Verified kernel witnesses, never a distance certificate."""
        order = list(range(len(cols)))
        minimum = len(cols) + 1
        small = set()
        for _ in range(trials):
            rng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p in pivots:
                        s, v = pivots[p]
                        syndrome ^= s
                        word ^= v
                    else:
                        pivots[p] = (syndrome, word)
                        break
                if not syndrome:
                    basis.append(word)
                    minimum = min(minimum, word.bit_count())
                    if word.bit_count() <= 8:
                        small.add(word)
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for j in range(i):
                    trial = word ^ basis[j]
                    w = trial.bit_count()
                    minimum = min(minimum, w)
                    if w <= 8:
                        small.add(trial)
                    if w < weight:
                        word, weight = trial, w
                basis[i] = word
            if minimum <= 4:
                break
        return minimum, len(small)

    def optimize(side):
        width = len(counts[side][0]) * q
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        pool, seen = [], set()

        def retain(grid, rows, metric, cols):
            sig = signature(grid)
            if sig in seen:
                return
            seen.add(sig)
            pool.append((metric, sig, grid, rows, cols))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        def energy(metric):
            twos, n4, four, six = metric
            return (10000 * twos + 400 * n4 + 40 * four + six) / q

        for restart in range(2):
            grid = draw(side)
            rows = component(side, grid)
            metric, cols = fast_metric(rows, width)
            retain(grid, rows, metric, cols)
            for step in range(360):
                i, j = rng.choice(sites)
                proposal = [list(row) for row in grid]
                proposal[i][j] = draw_cell(side, counts[side][i][j])
                pr = component(side, proposal)
                pm, pc = fast_metric(pr, width)
                delta = energy(pm) - energy(metric)
                temperature = 24.0 * (0.986 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, rows, metric, cols = proposal, pr, pm, pc
                    retain(grid, rows, metric, cols)

        finalists = []
        for fast, sig, grid, rows, cols in pool:
            if gf2_rank(rows) != len(rows):
                continue
            n6 = kernel_six(cols, fast)
            upper, multiplicity = kernel_probe(cols, 32)
            quality = (fast[0], fast[1], n6,
                       -min(upper, 10), multiplicity,
                       fast[2], fast[3])
            finalists.append((quality, sig, grid, rows))
        finalists.sort(key=lambda item: item[:2])
        if not finalists:
            raise ValueError("no full-row-rank component finalist")
        return finalists[:4]

    def small_logical(checks, stabilizers, n):
        """Exact existence of a nontrivial quantum logical of weight <=4."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        # Linear normal forms modulo the stabilizer row space.
        residues = [1 << i for i in range(n)]
        for p in sorted(pivots):
            row = pivots[p] ^ (1 << p)
            value = 0
            while row:
                bit = row & -row
                value ^= residues[bit.bit_length() - 1]
                row ^= bit
            residues[p] = value

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syn in enumerate(cols):
            value = residues[i]
            if syn in seen and seen[syn] != value:
                return True
            seen.setdefault(syn, value)
        for i, x in enumerate(cols):
            for j in range(i):
                syn = x ^ cols[j]
                value = residues[i] ^ residues[j]
                if syn in seen and seen[syn] != value:
                    return True
                seen.setdefault(syn, value)
        return False

    def four_cycles(rows):
        return sum(t * (t - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for t in [(row & other).bit_count()])

    apool, bpool = optimize("A"), optimize("B")
    options = []
    for aq, af, A, ar in apool:
        for bq, bf, B, br in bpool:
            # Worst component first, then total burden at each priority.
            pre = tuple(v for a, b in zip(aq, bq)
                        for v in (max(a, b), a + b))
            options.append((pre, af, bf, A, B, ar, br))
    options.sort(key=lambda item: item[:3])

    best = None
    for pre, af, bf, A, B, ar, br in options[:8]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 192:
            continue

        # Verify cached component actions against the final full lift.
        actual_a = []
        for i in range(ra):
            for r in range(q):
                row = hx[(i * cb) * q + r]
                packed = 0
                for j in range(ca):
                    packed |= ((row >> (j * cb * q)) & mask) << (j * q)
                actual_a.append(packed)
        actual_b = [hz[a * q + r] & ((1 << (cb * q)) - 1)
                    for a in range(rb) for r in range(q)]
        assert actual_a == ar and actual_b == br

        bad_x = small_logical(hz, hx, n)
        bad_z = small_logical(hx, hz, n)
        fx, fz = four_cycles(hx), four_cycles(hz)
        key = (int(bad_x) + int(bad_z),) + pre + (
            max(fx / len(hx), fz / len(hz)), fx + fz, af, bf)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, n, k)

    if best is None:
        raise ValueError("no exact-rate full-lift finalist")
    key, A, B, hx, hz, n, k = best
    assert n == 544 and k == 192
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 23616
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "contract": design, "batch": batch, "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit reliability: parent Z17 measured slots with (578,204,5) scored 3.691-4.347, and slot50 had 125/1000 Z failures versus 100/13500 X failures despite weight-seven Z-type witnesses.
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
