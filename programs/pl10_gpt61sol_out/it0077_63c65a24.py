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
CONCEPT = ("C7 semidirect C4 with inversion action, order 28, shapes (2,3)x(2,5): lowering six-round CNOTs to 23184..27552 while screening both components should improve the parent's circuit fitness 6.7896 in cell 0.150; measurement can refute this.")
GROUP = ("Keep the C_m semidirect C4 inversion construction and acting generator of order four. The new size factor 19 makes q=36 illegal (n=684); m=7 gives q=28 and n=532, the largest order with r=4 under this size window. The related order-28 (2,4)x(2,4) bed reached about 6.7, motivating this test without isolating a group advantage.")
SHAPES = ("Shapes (2,3)x(2,5) give n=532, guaranteed k>=84 and rate 3/19=0.157895. Components have 56 checks on 84 and 140 variables, with dimensions at least 28 and 84. Verified full ranks require 0.125<=k/n<0.175. Parent Z failures spread across Acol/Bcol blocks, including Acol1/Bcol1 and Bcol3, so screening is distributed rather than treating one logical qubit or block as a proven hotspot.")
FILLING = ("Twenty count architectures use column weights 3..4, HX weights <=10 and HZ weights <=12. Extra harness batches prescribe different quotient/order palettes, not transformations of saved fillings. Independently optimize verified component lifts; require isolated trinomials to be invertible, prefer exclusion of kernel words <=6, and screen sampled low-weight witness multiplicity and short cycles. Combine portfolios using exact full ranks and prefer exact exclusion of quantum logicals <=4. CNOTs fall from the parent's 30672..32832 to 23184..27552. Parent heavier slots had 24..45 Z failures versus 1..2 in lighter slots; reduced exposure is a hypothesis, not a controlled causal result or a pL certificate.")
FAMILY_NAME = "r015_dic7_23x25_gatelean_kernelportfolio"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Same inversion semidirect-product family, with smaller m.
    # C7 semidirect C4: y*x*y^-1 = x^-1, y^4 = 1.
    assert q == 28
    m, r = 7, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        sign = -1 if g[1] % 2 else 1
        return ((-sign * g[0]) % m, (-g[1]) % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# Column weights 3 or 4; row weights at most six.
# Sparse profiles remain connected at the protograph level.
_SUPPORT_PROFILES_A = (
    (1, 1, 2, 2, 2, 1),
    (1, 2, 2, 2, 2, 1),
    (2, 2, 2, 2, 2, 2),
    (0, 2, 2, 3, 1, 1),
    (0, 1, 3, 3, 2, 0),
)

# Column weights 3 or 4; row weights at most nine.
# Profile 6 retains augmentation rank two with a sparse trinomial.
_SUPPORT_PROFILES_B = (
    (1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
    (2, 1, 1, 2, 2, 2, 2, 2, 1, 1),
    (2, 2, 1, 2, 2, 2, 2, 2, 1, 1),
    (2, 2, 1, 2, 2, 2, 2, 2, 2, 1),
    (0, 2, 2, 2, 2, 3, 1, 1, 1, 2),
    (1, 1, 1, 3, 3, 2, 2, 2, 0, 0),
    (1, 1, 2, 1, 3, 2, 2, 1, 2, 0),
)

# Twenty distinct count briefs. A columns of weight four pair only
# with B rows of weight <=8, checking the HZ budget as well as HX.
_PROFILE_PAIRS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 0), (1, 1), (1, 4),
    (2, 0), (2, 1), (2, 6),
    (3, 0), (3, 1), (3, 2), (3, 3),
    (4, 0), (4, 1), (4, 2), (4, 3),
)


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _small_kernel_word(rows, ncols):
    """Exact meet-in-the-middle test for a nonzero kernel word <=6."""
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    # Each subset of size <=3 is visited once. Equal syndromes of two
    # different subsets give a nonzero kernel word of weight <=6.
    seen = {0}
    for value in columns:
        if value in seen:
            return True
        seen.add(value)
    for i in range(ncols):
        for j in range(i):
            value = columns[i] ^ columns[j]
            if value in seen:
                return True
            seen.add(value)
    for i in range(ncols):
        for j in range(i):
            pair = columns[i] ^ columns[j]
            for h in range(j):
                value = pair ^ columns[h]
                if value in seen:
                    return True
                seen.add(value)
    return False


def NEW_CTX(rng, G, idx):
    """Optimize independent constituents, then screen their actual CSS pair."""
    import math

    design = idx % len(_PROFILE_PAIRS)
    mode = (idx // len(_PROFILE_PAIRS)) % 3
    pa, pb = _PROFILE_PAIRS[design]
    shapes = (A_SHAPE, B_SHAPE)
    profiles = (_SUPPORT_PROFILES_A[pa], _SUPPORT_PROFILES_B[pb])
    counts = [
        [[p[c * i + j] for j in range(c)] for i in range(r)]
        for p, (r, c) in zip(profiles, shapes)
    ]
    assert _grid_weights(*counts) <= min(12, W_MAX)
    assert _no_dead_lines(*counts)
    assert all(0 <= w <= CELL_MAX for grid in counts for row in grid for w in row)

    cosets = [[g for g in G.elements if g[1] == b] for b in range(4)]
    blocked = [set() for _ in G.elements]
    for a, g in enumerate(G.elements):
        for b, h in enumerate(G.elements):
            rel = G.mul(G.inv(g), h)
            square = G.mul(rel, rel)
            if square == G.e or G.mul(square, rel) == G.e:
                blocked[a].add(b)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw(w):
        if not w:
            return []
        b = rng.randrange(4)
        if mode == 0:
            # Relative order seven; translations still explore C4.
            return rng.sample(cosets[b], w)
        if mode == 1:
            # Mixed quotient occupancy: relative orders four and seven.
            cell = [rng.choice(cosets[b])]
            if w >= 2:
                cell.append(rng.choice(cosets[(b + 1) % 4]))
            if w == 3:
                cell.append(rng.choice([g for g in cosets[b] if g not in cell]))
            return cell
        # Even quotient differences: relative orders seven or fourteen.
        available = [a for a, g in enumerate(G.elements)
                     if (g[1] - b) % 2 == 0]
        cell = []
        for _ in range(w):
            a = rng.choice(available)
            cell.append(G.elements[a])
            available = [t for t in available if t not in blocked[a]]
        return cell

    invertibility = {}

    def safe_draw(w, isolated):
        for _ in range(96):
            cell = draw(w)
            if not isolated:
                return cell
            # An isolated singular trinomial can itself support a short
            # kernel. Test the exact block through the verified helper.
            gi = G.inv(cell[0])
            normalized = [G.mul(gi, g) for g in cell]
            sig = tuple(sorted(G.idx(g) for g in normalized))
            if sig not in invertibility:
                rows = lifted_matrices(G, [[normalized]], [[[]]])[0]
                invertibility[sig] = gf2_rank(rows) == G.q
            if invertibility[sig]:
                return cell
        raise ValueError("isolated trinomial palette had no invertible block")

    def component(side, grid):
        # Zero probes extract the same left/right actions as the full lift.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        mask = (1 << (len(grid[0]) * G.q)) - 1
        return [row & mask for row in rows]

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            first = (row & -row).bit_length() - 1
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(i) for i in range(width)}) == 1

    def kernel_sample(rows, width):
        """Verified kernel witnesses; returned minimum is only an upper bound."""
        minimum = width + 1
        small = set()
        for _ in range(24):
            rr = list(rows)
            order = list(range(width))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                hit = next((r for r in range(rank, len(rr)) if rr[r] & bit), None)
                if hit is None:
                    continue
                rr[rank], rr[hit] = rr[hit], rr[rank]
                pivot = rr[rank]
                for r in range(len(rr)):
                    if r != rank and rr[r] & bit:
                        rr[r] ^= pivot
                pivots.append(c)
                rank += 1
                if rank == len(rr):
                    break
            occupied = set(pivots)
            basis = []
            for c in order:
                if c in occupied:
                    continue
                word = 1 << c
                for p, row in zip(pivots, rr[:rank]):
                    if (row >> c) & 1:
                        word |= 1 << p
                basis.append(word)
            for i, word in enumerate(basis):
                candidates = [word]
                candidates.extend(word ^ other for other in basis[:i])
                for v in candidates:
                    weight = v.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 10 and len(small) < 512:
                        small.add(v)
        return minimum, len(small)

    def portfolio(side, support):
        height, width = len(support), len(support[0])
        nc = width * G.q
        sites = [(i, j) for i in range(height) for j in range(width)
                 if support[i][j]]
        isolated = {
            (i, j) for i, j in sites
            if support[i][j] == 3 and
            sum(support[r][j] > 0 for r in range(height)) == 1
        }
        pool = []
        fingerprints = set()

        def remember(grid, rows, four):
            if len(pool) >= 8 and four > pool[-1][0][0]:
                return
            deficit = len(rows) - gf2_rank(rows)
            key = (four, deficit)
            if len(pool) >= 8 and key > pool[-1][0]:
                return
            sig = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
            if sig in fingerprints or not connected(rows, nc):
                return
            fingerprints.add(sig)
            pool.append((key, rng.random(), clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(4):
            grid = [
                [safe_draw(support[i][j], (i, j) in isolated)
                 for j in range(width)] for i in range(height)
            ]
            rows = component(side, grid)
            four = _four_cycles(rows)
            remember(grid, rows, four)
            for step in range(180):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = safe_draw(support[i][j], (i, j) in isolated)
                proposed_rows = component(side, grid)
                proposed_four = _four_cycles(proposed_rows)
                remember(grid, proposed_rows, proposed_four)
                delta = (proposed_four - four) / G.q
                temperature = 2.0 * (1.0 - step / 180.0) + 0.03
                if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    rows, four = proposed_rows, proposed_four
                else:
                    grid[i][j] = old

        if not pool:
            raise ValueError("no connected constituent in support brief")
        finalists = []
        for (four, deficit), ticket, grid, rows in pool:
            bad = int(_small_kernel_word(rows, nc))
            d, multiplicity = kernel_sample(rows, nc)
            six = _six_cycles(rows)
            key = (bad, four, -min(d, 12), multiplicity, six, deficit)
            finalists.append((key, grid, d))
        finalists.sort(key=lambda item: item[0])
        return finalists

    def no_small_logicals(checks, stabilizers, n):
        """Exact exclusion of nontrivial quantum logical supports <=4."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        # Linear normal forms of unit vectors modulo the stabilizer span.
        residues = [1 << i for i in range(n)]
        for p in sorted(pivots):
            row = pivots[p] ^ (1 << p)
            residue = 0
            while row:
                bit = row & -row
                residue ^= residues[bit.bit_length() - 1]
                row ^= bit
            residues[p] = residue

        columns = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        singles = {}
        for i, syndrome in enumerate(columns):
            residue = residues[i]
            if syndrome == 0 and residue:
                return False
            if syndrome in singles and residue != singles[syndrome]:
                return False
            singles.setdefault(syndrome, residue)
        pairs = {}
        for i, x in enumerate(columns):
            for j in range(i):
                syndrome = x ^ columns[j]
                residue = residues[i] ^ residues[j]
                if syndrome == 0 and residue:
                    return False
                if syndrome in singles and residue != singles[syndrome]:
                    return False
                if syndrome in pairs and residue != pairs[syndrome]:
                    return False
                pairs.setdefault(syndrome, residue)
        return True

    apool = portfolio("A", counts[0])
    bpool = portfolio("B", counts[1])
    options = []
    for ak, A, ad in apool:
        for bk, B, bd in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 8 * k >= n and 40 * k < 7 * n):
                continue
            key = (
                max(ak[0], bk[0]), ak[0] + bk[0],
                max(ak[1], bk[1]), ak[1] + bk[1],
                -min(ad, bd, 12),
                abs(min(ad, 12) - min(bd, 12)),
                ak[3] + bk[3],
                max(ak[4], bk[4]), ak[4] + bk[4],
                k - 84,
            )
            options.append((key, A, B, hx, hz, n, k))
    if not options:
        raise ValueError("constituent portfolios missed verified cell 0.150")
    options.sort(key=lambda item: item[0])

    chosen = options[0]
    logical_safe = False
    for option in options[:8]:
        key, A, B, hx, hz, n, k = option
        if (no_small_logicals(hx, hz, n) and
                no_small_logicals(hz, hx, n)):
            chosen = option
            logical_safe = True
            break

    key, A, B, hx, hz, n, k = chosen
    assert n == 532 and 84 <= k and 40 * k < 7 * n
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    return {"A": A, "B": B, "design": design, "palette": mode,
            "k": k, "cycle_score": key, "logical4_safe": logical_safe,
            "kernel_safe": key[0] == 0}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the already screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Excluding component kernel words <=6 and minimizing component four/six cycles guarantees ASC measurability: current slot 14 passed these screens but its X schedule timed out after 120 seconds.
# REFUTED: Equal d2s predicts equal circuit reliability: current slots 1 and 11 both had d2s=15, but their worse bases had zero and 13 failures in 200000 shots.
# REFUTED: d2s=16 ensures near-ceiling memory in the order-36 bed: parent slot 11 had 45 Z failures/200000 and s=6.186, whereas slot 58 had one failure per basis and s=7.225.
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
