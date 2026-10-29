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
CONCEPT = ("On metacyclic Z19:Z3, s=7, order 57, shapes (3,4)x(3,4), selective cross-fiber terms giving both quotient components exact minimum weight six should improve the parent's five-code fitness of 105.2476; measured distances remain provisional upper bounds.")
GROUP = ("Keep the validated group and order byte-for-byte unchanged. Projection (a,b)->b onto C3 permits exact small-component diagnostics. Normal-subgroup-fiber-constant component words have weight 19 times their quotient weight; the parent's frequent weight-38 readings motivate hardening this subspace without identifying those witnesses as fiber-constant.")
SHAPES = ("Keep equal 228-variable, 171-check components, n=1425, guaranteed k>=57 and rate>=0.040, in the open-ended 0.100 cell. Parent measured minima were entirely sector 1: X witnesses included Acol1, and Z witnesses included Bcol0/1/3. Protect both components and retain an exact k/n<0.125 gate.")
FILLING = ("Twenty support-pair/mixing-budget prescriptions retain row totals 6--7 and column totals 4--5. Select quotient labelings whose verified 9x12 component kernels have exact minimum weight six, excluding fiber-constant component words below 114. Use minimal selective cross-fiber repairs, distinct relative labels, verified lifted ranks, sampled component kernel weights and four-cycles; independently rescreen finalists. None of these screens certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r010_m19r3s7_34x34_quotient6_selective_mix_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    # metacyclic Z_19 x| Z_3: y x y^-1 = x^7 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 19, 3, 7
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 4)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


def _balanced_profiles():
    """Enumerate count designs, collapsing row/column permutation images."""
    from itertools import product, permutations
    row_options = [v for v in product((1, 2, 3), repeat=4)
                   if 6 <= sum(v) <= 7]
    row_perms = list(permutations(range(3)))
    representatives = {}
    for rows in product(row_options, repeat=3):
        if not all(4 <= sum(rows[i][j] for i in range(3)) <= 5
                   for j in range(4)):
            continue
        key = min(tuple(sorted(tuple(rows[i][j] for i in p)
                               for j in range(4)))
                  for p in row_perms)
        if key not in representatives:
            representatives[key] = rows
    return sorted(representatives.values(),
                  key=lambda a: (-sum(map(sum, a)),
                                 sum(x == 1 for row in a for x in row),
                                 a))


def _flux_grid(idx, side):
    """Normalized quotient voltages: six independent interior entries."""
    number = ((131 * idx + 97) if side == "A"
              else (239 * idx + 313)) % 729
    if number == 0:
        number = 1
    flux = [[0] * 4 for _ in range(3)]
    for i in range(1, 3):
        for j in range(1, 4):
            flux[i][j] = number % 3
            number //= 3
    return flux


def _fiber_cells(rng, counts, flux):
    cells = []
    for i in range(3):
        row = []
        for j in range(4):
            size = counts[i][j]
            while True:
                rotations = rng.sample(range(19), size)
                differences = [(a - b) % 19 for a in rotations
                               for b in rotations if a != b]
                if len(set(differences)) == len(differences):
                    break
            row.append([(a, flux[i][j]) for a in rotations])
        cells.append(row)
    return cells


def _component_rows(hx, hz, q):
    """Extract component matrices from the verified, actual LP lift."""
    mask = (1 << q) - 1
    arows = []
    for i in range(3):
        for t in range(q):
            row = hx[(i * 4) * q + t]
            compressed = 0
            for j in range(4):
                compressed |= ((row >> (j * 4 * q)) & mask) << (j * q)
            arows.append(compressed)
    bmask = (1 << (4 * q)) - 1
    brows = [row & bmask for row in hz[:3 * q]]
    return arows, brows


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _kernel_reading(rows, width, rng):
    """Sample kernel words after randomized elimination; no lower bound."""
    reading = width + 1
    for _ in range(6):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapped_bits = [1 << p for p in permutation]
        pivots = {}
        for original in rows:
            remapped = 0
            remaining = original
            while remaining:
                bit = remaining & -remaining
                remapped |= mapped_bits[bit.bit_length() - 1]
                remaining ^= bit
            while remapped:
                p = remapped.bit_length() - 1
                if p in pivots:
                    remapped ^= pivots[p]
                else:
                    pivots[p] = remapped
                    break
        ordered = sorted(pivots.items())
        basis = []
        for free in range(width):
            if free in pivots:
                continue
            word = 1 << free
            for p, row in ordered:
                if (word & row).bit_count() & 1:
                    word |= 1 << p
            basis.append(word)
            reading = min(reading, word.bit_count())
        for i, word in enumerate(basis):
            for other in basis[:i]:
                reading = min(reading, (word ^ other).bit_count())
        if len(basis) >= 3:
            for _ in range(200):
                a, b, c = rng.sample(basis, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
    return reading


def NEW_CTX(rng, G, idx):
    """Harden the quotient first, then optimize fresh rotation coordinates."""
    def kernel_words(rows, width):
        """Enumerate the entire small binary kernel, not a sampled bound."""
        pivots = {}
        for original in rows:
            row = original
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items())
        basis = []
        for free in range(width):
            if free in pivots:
                continue
            word = 1 << free
            for p, row in ordered:
                if (word & row).bit_count() & 1:
                    word |= 1 << p
            basis.append(word)
        words = [0]
        for word in basis:
            words += [old ^ word for old in words]
        return words[1:]

    # A rank-three augmentation component with a weight-two kernel
    # permits quotient distance six without forcing a sparse GF4 kernel.
    eligible = []
    for profile in _balanced_profiles():
        parity_rows = [sum((count & 1) << j
                           for j, count in enumerate(row))
                       for row in profile]
        words = kernel_words(parity_rows, 4)
        if len(words) == 1 and words[0].bit_count() == 2:
            eligible.append(profile)
    if not eligible:
        raise ValueError("no balanced augmentation-weight-two profile")

    # Five support-pair prescriptions crossed with four mixing budgets.
    # Extra harness slots also change the support-pair offset.
    design = idx % 20
    pair, mode = design % 5, design // 5
    epoch = idx // 20
    sa = eligible[(2 * pair + epoch) % len(eligible)]
    sb = eligible[(3 * pair + 1 + epoch) % len(eligible)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    # Diagnostic quotient only: the evaluated group remains exactly G.
    Q = _G(_registry_ops("cyclic", 3, None), 3)

    def quotient_pool(counts, side):
        doubles = [(i, j) for i in range(3) for j in range(4)
                   if counts[i][j] == 2]
        triples = [(i, j) for i in range(3) for j in range(4)
                   if counts[i][j] == 3]
        nominal = min(len(doubles), 2 + mode)
        triple_budget = (0 if mode == 0 else
                         min(len(triples), 1) if mode == 1 else
                         (len(triples) + 1) // 2 if mode == 2 else
                         len(triples))
        pool = []
        signatures = set()
        for trial in range(320):
            # Relax only unsuccessful sparse repairs; keep the best
            # successful mixing budget, rather than returning unchecked cells.
            budget = nominal
            if trial >= 128:
                budget = min(len(doubles), nominal + 1)
            if trial >= 224:
                budget = len(doubles)
            mixed2 = set(rng.sample(doubles, budget))
            mixed3 = set(rng.sample(triples, triple_budget))
            labels = []
            quotient = []
            for i in range(3):
                label_row, quotient_row = [], []
                for j in range(4):
                    count = counts[i][j]
                    b = rng.randrange(3)
                    direction = 1 + rng.randrange(2)
                    bs = [b] * count
                    if (i, j) in mixed2:
                        bs[1] = (b + direction) % 3
                    if (i, j) in mixed3:
                        bs[2] = (b + direction) % 3
                    label_row.append(tuple(bs))
                    # Projection is in F2[C3]: repeated cosets CANCEL.
                    # lifted_matrices deduplicates cells, so cancel here first.
                    quotient_row.append([c for c in range(3)
                                         if bs.count(c) & 1])
                labels.append(label_row)
                quotient.append(quotient_row)

            signature = tuple(tuple(row) for row in labels)
            if signature in signatures:
                continue
            qhx, qhz, _ = lifted_matrices(Q, quotient, quotient)
            qar, qbr = _component_rows(qhx, qhz, 3)
            rows = qar if side == "A" else qbr
            words = kernel_words(rows, 12)
            if not words or min(w.bit_count() for w in words) != 6:
                continue

            # The weight-two augmentation kernel plus one GF4 kernel
            # gives dimension three. Reject additional quotient deficits.
            if len(words) != 7:
                continue
            key = (budget + 2 * triple_budget, _four_cycles(rows))
            pool.append((key, labels))
            signatures.add(signature)
            pool.sort(key=lambda item: item[0])
            pool = pool[:4]
            if trial >= 63 and len(pool) == 4:
                break
        if not pool:
            raise ValueError("no exact quotient-distance-six labeling")
        return [labels for _, labels in pool]

    apool = quotient_pool(sa, "A")
    bpool = quotient_pool(sb, "B")

    def materialize(labels):
        """Keep all ordered within-cell relative elements distinct."""
        grid = []
        for row in labels:
            output = []
            for bs in row:
                while True:
                    rotations = rng.sample(range(19), len(bs))
                    cell = [(a, b) for a, b in zip(rotations, bs)]
                    ratios = [G._key(G.mul(g, G.inv(h)))
                              for t, g in enumerate(cell)
                              for u, h in enumerate(cell) if t != u]
                    if len(set(ratios)) == len(ratios):
                        break
                output.append(cell)
            grid.append(output)
        return grid

    diagnostic_seed = rng.randrange(1 << 62)
    finalists = []
    for attempt in range(32):
        A = materialize(apool[attempt % len(apool)])
        B = materialize(bpool[(attempt // len(apool) + pair) % len(bpool)])
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 8 * k < n):
            continue
        ar, br = _component_rows(hx, hz, G.q)
        # Common diagnostic permutations make comparisons less dependent
        # on accidentally favorable random elimination orders.
        da = _kernel_reading(ar, 4 * G.q,
                             random.Random(diagnostic_seed))
        db = _kernel_reading(br, 4 * G.q,
                             random.Random(diagnostic_seed + 1))
        ca4, cb4 = _four_cycles(ar), _four_cycles(br)
        merit = (min(da, db), da + db,
                 -max(ca4, cb4), -(ca4 + cb4), k)
        finalists.append((merit, A, B, ar, br, da, db, k, ca4, cb4))
        finalists.sort(key=lambda item: item[0], reverse=True)
        finalists = finalists[:4]

    if not finalists:
        raise ValueError("no candidate passed exact low-rate rank screening")

    best = None
    for _, A, B, ar, br, da, db, k, ca4, cb4 in finalists:
        # Independent follow-up searches can only lower each reading.
        # These remain upper bounds on CLASSICAL component distance.
        for repeat in range(3):
            seed = diagnostic_seed + 100003 * (repeat + 1)
            da = min(da, _kernel_reading(ar, 4 * G.q,
                                        random.Random(seed)))
            db = min(db, _kernel_reading(br, 4 * G.q,
                                        random.Random(seed + 1)))
        merit = (min(da, db), da + db,
                 -max(ca4, cb4), -(ca4 + cb4), k)
        if best is None or merit > best[0]:
            best = (merit, A, B, da, db)

    hx, hz, n = lifted_matrices(G, best[1], best[2])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    return {"idx": idx, "cells": {"A": best[1], "B": best[2]},
            "quotient_component_distance": (6, 6),
            "component_upper_readings": (best[3], best[4]),
            "screen": best[0]}


def CELL(rng, G, side, i, j, ctx):
    """Return the quotient-hardened, independently rescreened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Balanced component degrees and same-fiber order-19 ratios guarantee distance above 38; the parent produced many d2s=38 upper readings despite legal row weights and k-kb values from 0 to 12.
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
