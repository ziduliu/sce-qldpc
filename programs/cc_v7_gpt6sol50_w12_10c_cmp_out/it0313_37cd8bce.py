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
CONCEPT = ("For metacyclic C13 semidirect C4 (action 5), order 52, and A=B=(1,3), selecting components with fewer rooted weight-eight relations may exceed the distance-eight readings.")
GROUP = ("Keep the measured order-52 group for a filling-only comparison: all five selected parent codes had k=208 and d30s=8.")
SHAPES = ("Keep equal (1,3)x(1,3) sides: n=520, guaranteed k/n=208/520=0.400, and equal-length three-block component codes. The parent's lightest sector-one witnesses favored Acol1 for X and Bcol1 for Z.")
FILLING = ("Use twenty indexed action-coset profiles with three distinct terms per cell and row weight 12. Select independently labelled A and B components using verified-lift pair screens and a bounded rooted census through weight eight, then check exact k on their pairings. Target the parent's numerous weight-eight witnesses; its asymmetric slots mostly read five or six.")
FAMILY_NAME = "r040_c13rt4_13x13_independent_root8_census"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 52

def custom_ops(q):
    # C_13 semidirect C_4: y*x*y^-1 = x^5; 5^2 = -1 mod 13.
    m, r, s = 13, 4, 5
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _component_cycles(G, row, left):
    """Count repeated check-node pairs, then triangles, in one component."""
    adj = [set() for _ in range(G.q)]
    edges = set()
    four = 0
    for x in G.elements:
        for cell in row:
            checks = [G.idx(G.mul(g, x) if left
                            else G.mul(x, G.inv(g))) for g in cell]
            for p in range(len(checks)):
                for t in range(p):
                    u, v = sorted((checks[t], checks[p]))
                    edge = (u, v)
                    if edge in edges:
                        four += 1
                    else:
                        edges.add(edge)
                    adj[u].add(v)
                    adj[v].add(u)
    six = sum(1 for u in range(G.q) for v in adj[u] if v > u
              for w in adj[u].intersection(adj[v]) if w > v)
    return four, six


def _motif_cell(rng, motif, side, column):
    """Three distinct elements with a prescribed C4-action occupancy."""
    profiles = (
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
        (0, 0, 1), (0, 0, 2), (0, 0, 3),
        (0, 1, 1), (0, 2, 2), (0, 3, 3),
    )
    profile = profiles[(motif + 3 * side + 2 * column) % len(profiles)]
    used = set()
    cell = []
    for b in profile:
        a = rng.randrange(13)
        while (a, b) in used:
            a = rng.randrange(13)
        used.add((a, b))
        cell.append((a, b))
    rng.shuffle(cell)
    return cell


def _verified_component_columns(rows, q, step):
    """Read a component's columns from the evaluator-equivalent lift."""
    columns = [0] * (3 * q)
    mask = (1 << q) - 1
    for r in range(q):
        for j in range(3):
            bits = (rows[r] >> (j * step * q)) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << r
                bits ^= low
    return columns


def _short_word_stats(rng, columns):
    """Count component words of weight at most four; sample weight five."""
    singles = set(columns)
    bad2 = len(columns) - len(singles) + (0 in singles)
    pairs = {}
    bad3 = bad4 = 0
    for i, x in enumerate(columns):
        for y in columns[i + 1:]:
            syndrome = x ^ y
            if syndrome in singles:
                bad3 += 1
            bad4 += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1

    bad5 = 0
    for _ in range(5000):
        i, j, k = rng.sample(range(len(columns)), 3)
        if columns[i] ^ columns[j] ^ columns[k] in pairs:
            bad5 += 1
    return bad2, bad3, bad4, bad5


def _screen_cell(rng, count, idx, side, column, trial):
    """Sample distinct labels with a controlled C4-coset occupancy."""
    if count == 2:
        # A same-coset binomial has a rotation, rather than an
        # order-four action element, as its relative group element.
        b = rng.randrange(4)
        return [(a, b) for a in rng.sample(range(13), 2)]

    if count == 4:
        profiles = ((0, 1, 2, 3), (0, 0, 1, 2),
                    (0, 1, 1, 3), (0, 1, 2, 2))
    else:
        profiles = ((0, 1, 2), (0, 1, 3), (0, 2, 3),
                    (0, 0, 1), (0, 1, 1), (0, 0, 2),
                    (0, 2, 2), (0, 1, 2))
    pattern = profiles[(idx * 7 + side * 3 + column * 2 + trial
                        + rng.randrange(len(profiles))) % len(profiles)]
    shift = rng.randrange(4)
    cell = []
    for b in sorted(set(pattern)):
        cell.extend((a, (b + shift) % 4)
                    for a in rng.sample(range(13), pattern.count(b)))
    rng.shuffle(cell)
    return cell


def _pair_grade(columns):
    """Count relations of weights at most four in extracted columns."""
    singles = set(columns)
    bad2 = len(columns) - len(singles) + int(0 in singles)
    pairs = set()
    bad3 = bad4 = 0
    for i, x in enumerate(columns):
        for y in columns[i + 1:]:
            syndrome = x ^ y
            bad3 += syndrome in singles
            bad4 += syndrome in pairs
            pairs.add(syndrome)
    return bad2, bad3, bad4


def _rooted_component_reading(columns, q, limit=7, node_cap=1400):
    """Budgeted search for short words in a regular lifted component.

    A group translation can root any component word in one of its
    three variable blocks. A reading of eight means only that this
    bounded search found no word of weight at most seven.
    """
    adjacency = [[] for _ in range(q)]
    maximum = 0
    for variable, syndrome in enumerate(columns):
        maximum = max(maximum, syndrome.bit_count())
        bits = syndrome
        while bits:
            low = bits & -bits
            adjacency[low.bit_length() - 1].append(variable)
            bits ^= low

    best = limit + 1
    for block in range(3):
        root = block * q
        budget = [node_cap]

        def visit(syndrome, used, depth):
            nonlocal best
            if not syndrome:
                best = min(best, depth)
                return
            if depth >= min(limit, best - 1):
                return
            if syndrome.bit_count() > maximum * (limit - depth):
                return
            if budget[0] <= 0:
                return
            budget[0] -= 1

            bits = syndrome
            pivot = None
            fewest = len(columns) + 1
            for _ in range(min(4, bits.bit_count())):
                low = bits & -bits
                bits ^= low
                check = low.bit_length() - 1
                available = sum(1 for v in adjacency[check]
                                if not (used >> v) & 1)
                if available < fewest:
                    pivot, fewest = check, available
            if not fewest:
                return

            options = [v for v in adjacency[pivot]
                       if not (used >> v) & 1]
            options.sort(key=lambda v:
                         (columns[v] & syndrome).bit_count(),
                         reverse=True)
            for v in options:
                next_syndrome = syndrome ^ columns[v]
                if next_syndrome.bit_count() <= maximum * (
                        limit - depth - 1):
                    visit(next_syndrome, used | (1 << v), depth + 1)
                if best <= 4 or budget[0] <= 0:
                    break

        visit(columns[root], 1 << root, 1)
        if best <= 4:
            break
    return best


def _full_pair_reading(rows, n):
    """Reject detected weight-one to weight-four full-check kernels."""
    columns = [0] * n
    for check, row in enumerate(rows):
        mark = 1 << check
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= mark
            row ^= low
    bad2, bad3, bad4 = _pair_grade(columns)
    if 0 in columns:
        return 1
    if bad2:
        return 2
    if bad3:
        return 3
    return 4 if bad4 else 8


def _profile_triplet(rng, idx, side, column):
    """An indexed coset-occupancy design with fresh, distinct labels."""
    profiles = (
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
        (0, 0, 1), (0, 1, 1), (0, 0, 2), (0, 2, 2),
        (0, 0, 3), (0, 3, 3),
    )
    design = idx % 20
    phase, pattern = divmod(design, 10)
    cosets = profiles[(pattern + 3 * side + 2 * column
                       + phase * (5 + side)) % len(profiles)]
    shift = rng.randrange(4)
    cell = []
    for b in sorted(set(cosets)):
        cell.extend((a, (b + shift) % 4)
                    for a in rng.sample(range(13), cosets.count(b)))
    rng.shuffle(cell)
    return cell


def _rooted_eight_census(columns, q, node_cap, hit_cap):
    """Bounded census of component words through weight eight.

    Regular group translation lets a word be rooted at the identity
    variable of one of the three blocks. Exhaustion is reported
    separately: reaching a node or hit cap is not a distance proof.
    """
    incident = [[] for _ in range(q)]
    for variable, syndrome in enumerate(columns):
        bits = syndrome
        while bits:
            bit = bits & -bits
            incident[bit.bit_length() - 1].append(variable)
            bits ^= bit

    minimum, hits, exhaustive = 10, 0, True
    for block in range(3):
        nodes = [0]

        def visit(syndrome, used, depth):
            nonlocal minimum, hits, exhaustive
            if not syndrome:
                minimum = min(minimum, depth)
                hits += 1
                return
            if depth >= 8 or syndrome.bit_count() > 3 * (8 - depth):
                return
            if nodes[0] >= node_cap or hits >= hit_cap:
                exhaustive = False
                return
            nodes[0] += 1

            bits = syndrome
            pivot, fewest = None, len(columns) + 1
            for _ in range(min(5, bits.bit_count())):
                bit = bits & -bits
                bits ^= bit
                check = bit.bit_length() - 1
                available = sum(1 for v in incident[check]
                                if not (used >> v) & 1)
                if available < fewest:
                    pivot, fewest = check, available
            if not fewest:
                return

            choices = [v for v in incident[pivot]
                       if not (used >> v) & 1]
            choices.sort(key=lambda v:
                         (columns[v] & syndrome).bit_count(),
                         reverse=True)
            for v in choices:
                if nodes[0] >= node_cap or hits >= hit_cap:
                    exhaustive = False
                    break
                visit(syndrome ^ columns[v],
                      used | (1 << v), depth + 1)

        visit(columns[block * q], 1 << (block * q), 1)
        if hits >= hit_cap:
            exhaustive = False
            break
    return minimum, hits, exhaustive


def _legacy_ctx(rng, G, idx):
    """Previous off-centre-binomial experiment, retained for comparison."""
    motif = idx % 20
    proposals = []
    for trial in range(20):
        if motif < 8:
            ac = bc = [3, 3, 3]
        else:
            binomial = 0 if (motif + trial) % 2 else 2
            other = 2 - binomial
            if motif < 14:
                ac = [3, 3, 3]
                ac[1], ac[binomial] = 4, 2
                bc = [3, 3, 3]
                bc[other] = 2
            else:
                ac = [3, 3, 3]
                ac[binomial] = 2
                bc = [3, 3, 3]
                bc[1], bc[other] = 4, 2

        A = [[_screen_cell(rng, ac[j], idx, 0, j, trial)
              for j in range(3)]]
        B = [[_screen_cell(rng, bc[j], idx, 1, j, trial)
              for j in range(3)]]
        hx, hz, n = lifted_matrices(G, A, B)
        acol = _verified_component_columns(hx, G.q, 3)
        bcol = _verified_component_columns(hz, G.q, 1)
        ga, gb = _pair_grade(acol), _pair_grade(bcol)
        short = (ga[0] + gb[0], ga[1] + gb[1],
                 ga[2] + gb[2], max(ga[2], gb[2]))
        proposals.append((short, trial, A, B, hx, hz, n,
                          acol, bcol))
    proposals.sort(key=lambda entry: entry[:2])

    finalists = []
    for short, trial, A, B, hx, hz, n, acol, bcol in proposals[:6]:
        da = _rooted_component_reading(acol, G.q)
        db = _rooted_component_reading(bcol, G.q)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        finalists.append((da, db, k, short, trial, A, B, hx, hz, n))

    finalists.sort(
        key=lambda entry: (195 <= entry[2] <= 220,
                           min(entry[0], entry[1]),
                           entry[2], -sum(entry[3]), -entry[4]),
        reverse=True)
    best = None
    for da, db, k, short, trial, A, B, hx, hz, n in finalists[:4]:
        full = min(_full_pair_reading(hx, n),
                   _full_pair_reading(hz, n))
        reading = min(da, db, full)
        key = (195 <= k <= 220, reading,
               k * reading * reading, -sum(short), -trial)
        if best is None or key > best[0]:
            best = (key, {"A": A, "B": B})
    return best[1]


def NEW_CTX(rng, G, idx):
    """Select fresh A and B components independently, then check their LP."""
    options = {"A": [], "B": []}
    q = G.q
    for _ in range(38):
        A = [[_profile_triplet(rng, idx, 0, j) for j in range(3)]]
        B = [[_profile_triplet(rng, idx, 1, j) for j in range(3)]]
        hx, hz, _ = lifted_matrices(G, A, B)
        for side, grid, rows, step in (
                ("A", A, hx, 3), ("B", B, hz, 1)):
            columns = _verified_component_columns(rows, q, step)
            pair_grade = _pair_grade(columns)
            # Repeated pairs of check neighbours make short component
            # relations more likely; they break ties in the cheap screen.
            overlaps = 0
            for i, x in enumerate(columns):
                for y in columns[:i]:
                    shared = (x & y).bit_count()
                    overlaps += shared * (shared - 1) // 2
            options[side].append(
                (pair_grade + (overlaps,), grid, columns))

    ranked = {}
    for side in ("A", "B"):
        preliminary = sorted(options[side], key=lambda x: x[0])[:11]
        shallow = []
        for coarse, grid, columns in preliminary:
            reading = _rooted_eight_census(columns, q, 2400, 10)
            grade = (reading[0], reading[2], -reading[1],
                     tuple(-v for v in coarse))
            shallow.append((grade, grid, columns, coarse))
        shallow.sort(key=lambda x: x[0], reverse=True)

        deep = []
        for _, grid, columns, coarse in shallow[:4]:
            reading = _rooted_eight_census(columns, q, 18000, 28)
            grade = (reading[0], reading[2], -reading[1],
                     tuple(-v for v in coarse))
            deep.append((grade, grid))
        deep.sort(key=lambda x: x[0], reverse=True)
        ranked[side] = deep[:3]

    best = None
    for a_grade, A in ranked["A"]:
        for b_grade, B in ranked["B"]:
            k = lifted_k(G, A, B)
            grade = (208 <= k <= 220,
                     min(a_grade[0], b_grade[0]),
                     a_grade[1] and b_grade[1],
                     a_grade[2] + b_grade[2],
                     -abs(k - 208), a_grade, b_grade)
            if best is None or grade > best[0]:
                best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without subsequent relabeling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform rolled supports did not reliably suppress light logicals:
# the parent's 60 distinct slots included many d2s=2 readings.
# REFUTED: Adjacent-coset 3/4 labeling removes weight-20 sector-1
# witnesses: all five measured order-116 slots had d30s=20.
# REFUTED: Three-term C4-coset motifs screened for component cycles
# exceed distance eight on this order-52 (1,3)x(1,3) bed: all five
# measured parent slots had k=208 and d30s=8.
# REFUTED: Off-centre binomials screened only through component weight
# seven improve this bed: the five measured parent slots still had
# d30s=8, while its asymmetric slots mostly had d2s=5 or 6.
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
