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
CONCEPT = ("For A4 x C1, order 12, shapes (2,5)x(3,8), degree-three component columns, bounded extraction loads and verified short-kernel screening should improve circuit reliability over the parent's single-row weight-three-logical designs.")
GROUP = ("Preserve the even-permutation direct-product family A4 x Cr. The new size factor 46 requires q<=13, so r=1 and q=12 are the largest admissible family parameters. A related order-12 bed scored only 0.7 at higher rate; nonabelianness alone is not the hypothesis. Increased check density and screened fillings must earn the improvement.")
SHAPES = ("Target cell 0.350 with (2,5)x(3,8): n=552, kb=180 and guaranteed rate 15/46=0.32609. Require exact 180<=k<=206, entirely within the target cell. A has 24 checks on 60 component variables; B has 36 checks on 96. Parent weight-three Z witnesses repeated in Acol2 across all four B columns, while failure enrichment remained diffuse near 1.0-1.1x; replace both single-row components rather than patching one presumed hotspot.")
FILLING = ("Five A support architectures cross four B reinforcement budgets. A columns have three terms and B columns have three or four; both CSS check weights are <=12, with 23976-25488 six-round CNOTs. Verified component projections guide label searches that prioritize small-kernel removal and short cycles. Exact weight-five/six kernel multiplicities rank component finalists; verified full ranks and stabilizer-aware logical screening through weight four select the final code. Additional harness batches change trinomial relative-order policies. Target the parent's fitness 3.3334, collective logical flips and 48 weight-three Z witnesses per measured code; these screens do not certify circuit pL.")
FAMILY_NAME = "r035_A4xC1_25x38_col3_kernel_exposure_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C1 requires order 12")
    # Same A4 x Cr direct-product construction; only r changes.
    r = q // 12
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(r)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % r,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + ((g[4] + h[4]) % r,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % r,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _designed_cell(rng, G, count, target_order, mode, orders):
    """Avoid order-two differences; prescribe binomial cycle lengths."""
    for _ in range(512):
        cell = rng.sample(G.elements, count)
        # Distinct projected permutations prevent pure-central pairs.
        if len({tuple(g[:4]) for g in cell}) != count:
            continue
        relative = [
            orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))]
            for a in range(count) for b in range(a + 1, count)
        ]
        if min(relative) < 3:
            continue
        if count == 2:
            if relative[0] != target_order:
                continue
        else:
            central_span = len({g[4] for g in cell})
            wanted = 3 if mode == 0 else 2
            if central_span != wanted:
                continue
        return cell
    raise ValueError("could not realize cell design")


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Twenty support briefs; screen only scaffold-built matrices."""
    import math

    q = G.q
    case = idx % 20
    extra = case % 4
    policy = (idx // 20) % 3
    atop = (
        (2, 2, 1, 1, 1),
        (0, 2, 2, 2, 1),
        (3, 0, 2, 1, 1),
        (3, 0, 0, 2, 2),
        (3, 3, 0, 0, 1),
    )[case // 4]
    sa = [list(atop), [3 - x for x in atop]]
    sb = [[1] * 8 for _ in range(3)]
    for r in range(extra):
        sb[r][(2 * r + idx // 20) % 8] = 2
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(sa[i][j] for i in range(2)) == 3
               for j in range(5))
    assert max(map(sum, sa)) == 8
    assert max(map(sum, sb)) <= 9

    orders = _orders(G)
    threes = [g for g in G.elements if orders[G.idx(g)] == 3]

    def draw_cell(count):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            g = G.pick(rng)
            return [g, G.mul(rng.choice(threes), g)]
        for _ in range(512):
            cell = rng.sample(G.elements, count)
            involutory = any(
                orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))] == 2
                for a in range(count) for b in range(a))
            if policy == 1 and not involutory:
                continue
            if policy == 2 and involutory:
                continue
            return cell
        raise ValueError("trinomial relative-order policy unavailable")

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def component(side, grid):
        # A one-cell opposite component leaves the desired action
        # in the initial sector-1 columns of the verified helper.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[G.e]]])[0]
        else:
            rows = lifted_matrices(G, [[[G.e]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    def fast_quality(rows, width):
        cols = columns(rows, width)
        bad = int(0 in cols or len(set(cols)) != width)
        singles = set(cols)
        pairs = {}
        triples = collisions = four = 0
        check_pairs = {}
        adj = [0] * len(rows)
        stars = 0

        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            d = len(nodes)
            stars += d * (d - 1) * (d - 2) // 6
            for a, u in enumerate(nodes):
                for v in nodes[:a]:
                    key = (v, u)
                    old = check_pairs.get(key, 0)
                    four += old
                    check_pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u

        for a, x in enumerate(cols):
            for b in range(a):
                syndrome = x ^ cols[b]
                triples += int(syndrome in singles)
                old = pairs.get(syndrome, 0)
                collisions += old
                pairs[syndrome] = old + 1

        triangles = 0
        for u in range(len(adj)):
            later = adj[u] & ~((1 << (u + 1)) - 1)
            while later:
                bit = later & -later
                v = bit.bit_length() - 1
                triangles += ((adj[u] & adj[v]) >> (v + 1)).bit_count()
                later ^= bit
        # Exact six-cycle count when four==0; otherwise a proxy.
        six = max(0, triangles - stars)
        if not bad:
            assert triples % 3 == 0 and collisions % 3 == 0
        return (bad, triples // 3, collisions // 3, four, six)

    def kernel56(rows, width, fast):
        """Exact weight-five/six counts after weights <=4 are excluded."""
        if fast[0] or fast[1] or fast[2]:
            return (10**9, 10**9)
        cols = columns(rows, width)
        pair_hist = {}
        for a, x in enumerate(cols):
            for b in range(a):
                s = x ^ cols[b]
                pair_hist[s] = pair_hist.get(s, 0) + 1
        triple_hist = {}
        five = six = 0
        for a, x in enumerate(cols):
            for b in range(a):
                pair = x ^ cols[b]
                for c in range(b):
                    s = pair ^ cols[c]
                    five += pair_hist.get(s, 0)
                    old = triple_hist.get(s, 0)
                    six += old
                    triple_hist[s] = old + 1
        # Colliding supports cannot overlap when weights <=4 are absent.
        # A five-set has ten pair/triple partitions; a six-set has
        # ten unordered triple/triple partitions.
        assert five % 10 == 0 and six % 10 == 0
        return five // 10, six // 10

    def optimize(side, counts):
        width = len(counts[0]) * q
        sites = [(i, j) for i, row in enumerate(counts)
                 for j, count in enumerate(row) if count]
        pool = []
        seen = set()

        def retain(grid, rows, cost):
            if cost[0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            pool.append((cost, signature, grid, rows))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        def energy(cost):
            bad, n3, n4, four, six = cost
            return (100000 * bad + 3000 * n3 + 400 * n4
                    + 50 * four + six)

        for restart in range(2):
            grid = [[draw_cell(t) for t in row] for row in counts]
            rows = component(side, grid)
            cost = fast_quality(rows, width)
            retain(grid, rows, cost)
            for step in range(360):
                i, j = rng.choice(sites)
                proposal = [list(row) for row in grid]
                proposal[i][j] = draw_cell(counts[i][j])
                pr = component(side, proposal)
                pc = fast_quality(pr, width)
                delta = energy(pc) - energy(cost)
                temperature = 400.0 * (0.985 ** step) + 2.0
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, rows, cost = proposal, pr, pc
                    retain(grid, rows, cost)

        if not pool:
            raise ValueError("no component with distinct nonzero syndromes")
        finalists = []
        for cost, signature, grid, rows in pool:
            n5, n6 = kernel56(rows, width, cost)
            deficit = len(rows) - gf2_rank(rows)
            quality = (cost[1], cost[2], n5, n6,
                       cost[3], cost[4], deficit)
            finalists.append((quality, signature, grid))
        finalists.sort(key=lambda item: item[:2])
        return finalists[:6]

    apool = optimize("A", sa)
    bpool = optimize("B", sb)

    def no_small_logical(checks, stabilizers, n):
        """Exact exclusion of nontrivial quantum logicals of weight <=4."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        bits = [1 << i for i in range(n)]
        residues = list(bits)
        for p in sorted(pivots):
            row = pivots[p] ^ bits[p]
            residue = 0
            while row:
                bit = row & -row
                residue ^= residues[bit.bit_length() - 1]
                row ^= bit
            residues[p] = residue

        cols = columns(checks, n)
        singles = {}
        for i, s in enumerate(cols):
            residue = residues[i]
            if s == 0 and residue:
                return False
            if s in singles and singles[s] != residue:
                return False
            singles.setdefault(s, residue)

        pairs = {}
        for a, x in enumerate(cols):
            for b in range(a):
                s = x ^ cols[b]
                residue = residues[a] ^ residues[b]
                if s == 0 and residue:
                    return False
                if s in singles and singles[s] != residue:
                    return False
                if s in pairs and pairs[s] != residue:
                    return False
                pairs.setdefault(s, residue)
        return True

    options = []
    for aq, af, A in apool:
        for bq, bf, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (180 <= k <= 206):
                continue
            pre = (max(aq[0], bq[0]), aq[0] + bq[0],
                   aq[1] + bq[1],
                   max(aq[2], bq[2]), aq[2] + bq[2],
                   max(aq[3], bq[3]), aq[3] + bq[3],
                   aq[4] + bq[4], aq[5] + bq[5], k - 180)
            options.append((pre, af, bf, A, B, hx, hz, n, k))
    options.sort(key=lambda item: item[:3])

    best = None
    for pre, af, bf, A, B, hx, hz, n, k in options[:6]:
        if not css_ok(hx, hz):
            continue
        safe_x = no_small_logical(hz, hx, n)
        safe_z = no_small_logical(hx, hz, n)
        fx = _same_basis_cycles(hx) / len(hx)
        fz = _same_basis_cycles(hz) / len(hz)
        key = (int(not safe_x) + int(not safe_z),) + pre + (
            max(fx, fz), fx + fz, af, bf)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, n, k, safe_x and safe_z)

    if best is None:
        raise ValueError("no verified filling remains in cell 0.350")
    key, A, B, hx, hz, n, k, safe = best
    assert n == 552
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    expected = 6 * q * (11 * 15 + 7 * (24 + extra))
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "small_logical_safe": safe, "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing same-basis four-cycles did not suffice for reliable A4 x C3 memory: fitness 3.2178, widespread weight-three logicals, and 112-137 Z failures per 250 measured shots.
# REFUTED: Certified unit trinomials plus degree-weighted short-logical screening did not remove the single-row bed's weakness: fitness 3.3334, 48 weight-three Z witnesses in every measured code, and Z failures including 121/250 and 157/250 shots.
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
