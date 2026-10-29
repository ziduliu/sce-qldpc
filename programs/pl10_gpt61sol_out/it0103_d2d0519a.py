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
CONCEPT = ("Metacyclic Z24:Z2(s=23), order 48, shapes (1,2)x(2,5): mixed-coset components with column degree at least three and reduced small-syndrome boundaries should reduce collective circuit failures relative to the order-eight unit-row parent.")
GROUP = ("Keep the metacyclic construction rule and inversion action; resize to m=24,r=2,s=23, with 23^2=1 modulo 24. The new factor 12 would give invalid n=96 at q=8; q=48 gives n=576 and is the largest even dihedral lift admitted. The related order-48 (1,2)x(2,5) frontier bed scores 6.5, supporting this move without establishing a group-only advantage.")
SHAPES = ("Target cell 0.250 with (1,2)x(2,5): component matrices are 48x96 and 96x240, n=576, and kb=144, guaranteed rate 0.250. Verified full ranks require 0.225<=k/n<0.275 for every submitted slot. Parent failures affected essentially all 240 logicals, with diffuse 1.1-1.2x Acol/Bcol enrichment and ubiquitous sector-one weight-four witnesses; replace both components rather than repair an unproven hotspot.")
FILLING = ("Twenty support cards combine A weights 3/3, 3/4 and 4/4 with B column degrees three or four, distinct zero-cell patterns and balanced row loads. Mixed-coset labels use long rotational displacements. Verified-lift search minimizes component short words and cycles, then prefers no quantum logical through weight four and reduces small-boundary triples and possible ancilla-spread exposure. Sampled component protection saturates at ten; it is not a distance certificate. Rank-legal fallbacks remain eligible if every finalist has a short logical. CSS weights stay <=12 and six-round CNOT budgets are 25056-29952. The hypothesis is fewer measured failures, not a larger distance score.")
FAMILY_NAME = "r025_m24r2s23_12x25_mixed_boundarybudget"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires q=48")
    # Same metacyclic rule: x^24=y^2=1, y*x*y^-1=x^23.
    m, r, s = 24, 2, 23
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# B cards list (upper-cell count, lower-cell count) by column.
# Every column has degree three or four; every row remains live.
_B_SUPPORT_CARDS = (
    ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1)),
    ((0, 3), (2, 1), (2, 1), (2, 1), (1, 2)),
    ((3, 0), (0, 3), (1, 2), (1, 2), (2, 1)),
    ((3, 0), (3, 0), (0, 3), (0, 3), (1, 2)),
    ((2, 2), (1, 2), (1, 2), (2, 1), (2, 1)),
    ((0, 3), (3, 1), (2, 1), (2, 1), (1, 2)),
    ((0, 4), (2, 1), (2, 1), (2, 1), (2, 1)),
    ((3, 0), (0, 3), (2, 2), (1, 2), (2, 1)),
    ((2, 2), (2, 2), (1, 2), (1, 2), (2, 1)),
    ((2, 2), (2, 2), (2, 2), (1, 2), (2, 1)),
)

# Different support architectures, not saved-code orbit images.
# A maximum cell weight four permits only B cards with row sums <=8.
_SUPPORT_CARDS = (
    tuple(((3, 3), b) for b in range(10))
    + tuple(((3, 4), b) for b in (0, 1, 2, 4, 5, 7))
    + tuple(((4, 4), b) for b in (0, 4, 5, 6))
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


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def NEW_CTX(rng, G, idx):
    """Optimize a support card on the actual evaluator-convention lift."""
    from math import gcd

    q, m = G.q, G.q // 2
    design = idx % len(_SUPPORT_CARDS)
    ratio_mode = (idx // len(_SUPPORT_CARDS)) % 3
    aw, bcard = _SUPPORT_CARDS[design]
    sa = [list(aw)]
    profile = _B_SUPPORT_CARDS[bcard]
    sb = [[c[i] for c in profile] for i in range(2)]
    assert len(_SUPPORT_CARDS) == 20
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    # Orders 24, 12 and 8 are different ratio-order prescriptions,
    # not global automorphism images of one saved labeling.
    wanted_order = (24, 12, 8)[ratio_mode]
    asteps = [d for d in range(1, m)
              if m // gcd(m, d) == wanted_order]
    long_steps = [d for d in range(1, m)
                  if m // gcd(m, d) >= 6]

    def make_cell(count, is_a, phase):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(rng)]
        if is_a:
            # Gauge-fixed mixed support: never an all-rotation A cell.
            u, v = rng.choice(asteps), rng.randrange(m)
            cell = [G.e, (u, 0), (v, 1)]
            if count == 4:
                d = rng.choice(long_steps)
                cell.append(((v + d) % m, 1))
            return cell

        a, b = rng.randrange(m), rng.randrange(m)
        cell = [(a, 0), (b, 1)]
        if count == 3:
            d = rng.choice(long_steps)
            if phase % 2:
                cell.append(((b + d) % m, 1))
            else:
                cell.append(((a + d) % m, 0))
        elif count == 4:
            d, e = rng.choice(long_steps), rng.choice(long_steps)
            cell.extend([((a + d) % m, 0), ((b + e) % m, 1)])
        return cell

    def draw(weights, is_a, trial):
        return [[make_cell(t, is_a, idx + trial + i + j)
                 for j, t in enumerate(row)]
                for i, row in enumerate(weights)]

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= marker
                row ^= bit
        return result

    def assess(A, B, serial):
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 225 * n <= 1000 * k < 275 * n):
            return None
        ar, br = _component_rows(hx, hz, q)
        ast = _kernel_statistics(ar, A_SHAPE[1] * q)
        bst = _kernel_statistics(br, B_SHAPE[1] * q)
        # Component words can be repeated across five or two opposite
        # columns. These replication weights are only screening costs.
        key = tuple(v for t in range(3)
                    for v in (max(5 * ast[t], 2 * bst[t]),
                              5 * ast[t] + 2 * bst[t]))
        fa, fb = _four_cycles(ar), _four_cycles(br)
        key += (max(fa / len(ar), fb / len(br)),
                fa + fb, max(5 * ast[3], 2 * bst[3]))
        return key, serial, A, B, hx, hz, ar, br, n, k

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    pool, seen = [], set()

    def retain(item):
        if item is None:
            return
        sig = signature(item[2], item[3])
        if sig in seen:
            return
        seen.add(sig)
        pool.append(item)
        pool.sort(key=lambda entry: (entry[0], entry[1]))
        del pool[8:]

    for trial in range(512):
        A, B = draw(sa, True, trial), draw(sb, False, trial)
        retain(assess(A, B, trial))
        if trial >= 127 and pool:
            break
    if not pool:
        raise ValueError("No verified filling in the 0.250 rate cell")

    # Whole-cell mutations preserve the support and ratio-order card.
    # Neutral moves explore alternatives rather than orbit-copy a code.
    current = pool[0]
    for step in range(64):
        grids = [[[list(cell) for cell in row] for row in grid]
                 for grid in (current[2], current[3])]
        side = rng.randrange(2)
        weights = (sa, sb)[side]
        sites = [(i, j) for i, row in enumerate(weights)
                 for j, t in enumerate(row) if t]
        i, j = rng.choice(sites)
        grids[side][i][j] = make_cell(
            weights[i][j], side == 0, idx + step + i + j)
        item = assess(grids[0], grids[1], 512 + step)
        if item is not None and item[0] <= current[0]:
            current = item
        retain(item)

    def sampled_kernel(cs):
        """Sampled upper bound, not a lower-bound distance certificate."""
        order = list(range(len(cs)))
        best = len(cs) + 1
        for _ in range(24):
            rng.shuffle(order)
            pivots = {}
            for c in order:
                syndrome, word = cs[c], 1 << c
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
            if best < 8:
                break
        return best

    def triple_boundary(cs):
        """Orbit-anchored small-boundary triple exposure.

        One anchor per variable-block orbit avoids a full cubic census.
        Summing over all anchors is proportional to total triple load
        under the regular group action. This is a trapping-set surrogate.
        """
        total = 0
        width = len(cs)
        for a in range(0, width, q):
            x = cs[a]
            for b in range(width):
                if b == a:
                    continue
                pair = x ^ cs[b]
                for c in range(b):
                    if c == a:
                        continue
                    weight = (pair ^ cs[c]).bit_count()
                    if weight <= 3:
                        total += (4 - weight) ** 2
        return total

    def quantum_risk(checks, stabilizers, n):
        """Exact existence of logicals <=4 plus a possible-hook surrogate."""
        syndromes = columns(checks, n)
        spread = columns(stabilizers, n)
        exposure = [s.bit_count() + t.bit_count()
                    for s, t in zip(syndromes, spread)]

        # Canonical linear normal forms modulo the stabilizer row span.
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residues = [1 << c for c in range(n)]
        for p in sorted(pivots):
            tail = pivots[p] ^ (1 << p)
            value = 0
            while tail:
                bit = tail & -tail
                value ^= residues[bit.bit_length() - 1]
                tail ^= bit
            residues[p] = value

        # Equal syndromes of supports of size <=2 exhaust weight <=4.
        # Different quotient residues distinguish logical differences
        # from differences belonging to the stabilizer row space.
        seen_syndromes = {0: 0}
        short = False
        for s, r in zip(syndromes, residues):
            old = seen_syndromes.get(s)
            if old is not None and old != r:
                short = True
            else:
                seen_syndromes[s] = r

        hooks = 0
        for a, x in enumerate(syndromes):
            for b in range(a):
                s = x ^ syndromes[b]
                r = residues[a] ^ residues[b]
                old = seen_syndromes.get(s)
                if old is not None and old != r:
                    short = True
                elif old is None:
                    seen_syndromes[s] = r
                weight = s.bit_count()
                if weight <= 4:
                    common = (spread[a] & spread[b]).bit_count()
                    hooks += (common * (5 - weight) ** 2
                              * exposure[a] * exposure[b])
        # Shared checks need not realize this pair in the ASC schedule.
        return int(short), hooks

    best = None
    for coarse, serial, A, B, hx, hz, ar, br, n, k in pool:
        ac = columns(ar, A_SHAPE[1] * q)
        bc = columns(br, B_SHAPE[1] * q)
        da, db = sampled_kernel(ac), sampled_kernel(bc)
        deficit = max(0, 10 - da) + max(0, 10 - db)
        ba, bb = triple_boundary(ac), triple_boundary(bc)
        zshort, zhooks = quantum_risk(hx, hz, n)
        xshort, xhooks = quantum_risk(hz, hx, n)
        ua, ub = _six_cycles(ar), _six_cycles(br)
        score = (
            max(xshort, zshort), xshort + zshort,
            deficit,
            max(5 * ba, 2 * bb), 5 * ba + 2 * bb,
            max(xhooks, zhooks), xhooks + zhooks,
            max(ua / len(ar), ub / len(br)), ua + ub,
        ) + coarse + (k - q * 3,)
        if best is None or score < best[0]:
            best = (score, A, B, hx, hz, k)

    score, A, B, hx, hz, k = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return {"A": A, "B": B, "design": design,
            "ratio_mode": ratio_mode, "k": k, "circuit_risk": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell of the new support card."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Dispersed doubletons and component short-word/cycle optimization sufficiently suppress collective failures on the D8 high-rate bed: fitness was 3.2454, measured worse-basis failures were 117..147/250, and every measured code had abundant weight-four logicals.
# REFUTED: First-pass settlement plus <=3 and possible-hook screening makes q8 unit-row memory reliable: the parent scored 3.2422, had 119..153 worse-basis failures per 250 shots, and retained ubiquitous sector-one weight-four logicals.
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
