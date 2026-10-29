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
CONCEPT = ("Inversion-metacyclic C37:C2, order 74, shapes (1,2)x(2,3): dense noncommuting cells optimized against verified component witnesses should supply five measured distance readings >=16 and exceed the target cell's fitness 30.6.")
GROUP = ("Keep the metacyclic multiplication rule and inversion action. The new factor eight makes the parent's q=8 lift illegal at n=64, so resize within this family to m=37,r=2,s=36. Order 74 is the largest even inversion lift fitting the window, giving n=592. Related inversion beds measured fitness 45.0 and 44.7; this exact bed is untried, so no transferred distance advantage is assumed.")
SHAPES = ("Target cell 0.150 with A=(1,2), B=(2,3): n=592, kb=74 and guaranteed rate 1/8=0.125. Components have 148 variables/74 checks and 222 variables/148 checks. The shape admits q<=74. Parent weight-four minima occurred entirely in sector-one Acol/Bcol blocks across both wide components. Replace those monomial-rich components with fewer, denser blocks and require exact assembled 0.125<=k/n<0.175.")
FILLING = ("Twenty cards cross five legal support architectures with four quotient-occupancy rules, packing distinct unsigned normal-subgroup differences within each cell. Verified component lifts exclude kernels through weight four. Actual kernel witnesses guide cell repairs; independently deepen finalists and test their embedded logical classes with single/pair stabilizer dressings. Exact full ranks enforce the target cell, full CSS screening excludes logicals through four, and combined uncoloured incidence fingerprints reject emitted orbit copies. Both check-weight caps are twelve. Restricted and randomized witness searches are not distance certificates.")
FAMILY_NAME = "r015_m37r2s36_12x23_dense_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 74

def custom_ops(q):
    # Same inversion-metacyclic rule; the new factor eight requires
    # replacing q=8, which would give n=64 outside the size window.
    # y x y^-1 = x^36, and 36^2 = 1 mod 37.
    if not valid_q(q):
        raise ValueError("C37:C2 requires order 74")
    m, r, s = 37, 2, 36
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
B_SHAPE = (2, 3)               # (rb, cb)


from functools import lru_cache

# (A row sizes, B column sizes). Every card satisfies both row caps.
# Their HX/HZ maxima are respectively 12/10, 12/10, 12/11,
# 12/12, and 12/11.
_DENSE_SUPPORTS = (
    ((4, 4), ((2, 2), (2, 2), (2, 2))),
    ((4, 4), ((3, 1), (1, 3), (2, 2))),
    ((3, 5), ((3, 1), (1, 3), (2, 2))),
    ((3, 3), ((3, 3), (3, 3), (3, 3))),
    ((3, 4), ((3, 2), (2, 3), (2, 2))),
)


def _dense_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _dense_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _dense_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _dense_short4(cols, stabilizers=None):
    """Exact kernel/logical existence through four, not a distance reading."""
    basis = None if stabilizers is None else _dense_basis(stabilizers)
    seen = {0: 0}

    def insert(syndrome, word):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
            return False
        difference = word ^ old
        return bool(difference) and (
            basis is None or _dense_outside(difference, basis))

    for i, syndrome in enumerate(cols):
        if insert(syndrome, 1 << i):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            if insert(syndrome ^ cols[j], (1 << i) | (1 << j)):
                return True
    return False


def _dense_cycles(cols):
    total = 0
    for i, col in enumerate(cols):
        for other in cols[:i]:
            overlap = (col & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _dense_words(cols, rng, trials, memory=()):
    """Retain genuine nonzero kernel witnesses across information sets."""
    retained = sorted(set(memory), key=lambda w: (w.bit_count(), w))[:24]
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
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
                dependencies.append(word)

        words = sorted(set(retained + dependencies),
                       key=lambda w: (w.bit_count(), w))
        for sweep in range(2):
            reducers = sorted(set(words), key=int.bit_count)[:40]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    if combined and combined.bit_count() < weight:
                        word, weight = combined, combined.bit_count()
                reduced.append(word)
            words = sorted(set(reduced), key=lambda w: (w.bit_count(), w))

        discoveries = set(words + retained)
        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    discoveries.add(combined)
        retained = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:24]
    return retained


def _dense_cell(rng, G, size, side, i, j, mode, phase):
    if size <= 1:
        return [] if size == 0 else [G.pick(rng)]
    if side == "A":
        if mode == 0:
            outer = size // 2
        elif mode == 1:
            outer = 1
        elif mode == 2:
            outer = 1 if j == 0 else size - 1
        else:
            outer = min(size - 1, 1 + ((j + phase) % 2))
    elif mode == 0:
        outer = size // 2
    elif mode == 1:
        outer = 0 if (i + j + phase) % 2 == 0 else 1
    elif mode == 2:
        outer = 1
    else:
        outer = (i + 2*j + phase) % (size + 1)

    # Avoid repeated unsigned normal differences inside a cell.
    # This is a labeling constraint, not a substitute for matrix screening.
    for _ in range(64):
        xs = rng.sample(range(37), size - outer)
        ys = rng.sample(range(37), outer)
        differences = []
        for values in (xs, ys):
            for a, x in enumerate(values):
                for y in values[:a]:
                    d = (x - y) % 37
                    differences.append(min(d, 37 - d))
        if len(differences) != len(set(differences)):
            continue
        cell = [(a, 0) for a in xs] + [(a, 1) for a in ys]
        base = G.pick(rng)
        return [G.mul(base, g) for g in cell]
    raise ValueError("normal-difference cell packing exhausted")


def _dense_component(G, grid, side):
    """Extract components only from the scaffold's verified LP lifts."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    zero_a = [[[] for _ in range(ca)] for _ in range(ra)]
    zero_b = [[[] for _ in range(cb)] for _ in range(rb)]
    if side == "A":
        hx, hz, _ = lifted_matrices(G, grid, zero_b)
        rows = []
        for i in range(ra):
            for t in range(q):
                source = hx[(i * cb) * q + t]
                rows.append(sum(
                    ((source >> (j * cb * q)) & mask) << (j * q)
                    for j in range(ca)))
        width = ca * q
    else:
        hx, hz, _ = lifted_matrices(G, zero_a, grid)
        width = cb * q
        rows = [row & ((1 << width) - 1) for row in hz[:rb * q]]
    return rows, _dense_columns(rows, width)


def _dense_fingerprint(hx, hz, n, k):
    """Uncoloured incidence invariant, including arbitrary check mixing.

    Different fingerprints exclude Tanner isomorphism; equal ones are
    conservatively rejected rather than asserted isomorphic.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    profiles = [Counter() for _ in rows]
    overlaps = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[overlap] += 1
            if overlap:
                profiles[i][(weights[j], overlap)] += 1
                profiles[j][(weights[i], overlap)] += 1
    local = Counter(
        (weights[i], tuple(sorted(profile.items())))
        for i, profile in enumerate(profiles))
    qubits = [Counter() for _ in range(n)]
    for row, weight in zip(rows, weights):
        while row:
            bit = row & -row
            qubits[bit.bit_length() - 1][weight] += 1
            row ^= bit
    return (n, k, tuple(sorted(Counter(weights).items())),
            tuple(sorted(overlaps.items())),
            tuple(sorted(local.items())),
            tuple(sorted(Counter(
                tuple(sorted(p.items())) for p in qubits).items())))


def _dense_dressed(hx, hz, n, q, aw, bw):
    """Verified embedded logical witnesses, dressed into either LP sector."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    best = n + 1
    for side, seeds, checks, stabilizers in (
            ("A", aw, hx, hz), ("B", bw, hz, hx)):
        basis = _dense_basis(stabilizers)
        candidates = set()
        for seed in seeds[:8]:
            for fixed in range(cb if side == "A" else ca):
                word = (sum(
                    ((seed >> (j*q)) & mask) << ((j*cb + fixed)*q)
                    for j in range(ca)) if side == "A"
                    else seed << (fixed*cb*q))
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if _dense_outside(word, basis):
                    candidates.add(word)
                    best = min(best, word.bit_count())

        for word in sorted(candidates, key=int.bit_count)[:8]:
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                nearby = []
                for row in stabilizers:
                    combined = word ^ row
                    cw = combined.bit_count()
                    nearby.append((cw, row))
                    if cw < weight:
                        chosen, weight = combined, cw
                nearby.sort(key=lambda item: item[0])
                moves = [row for _, row in nearby[:16]]
                for i, row in enumerate(moves):
                    for other in moves[:i]:
                        combined = word ^ row ^ other
                        cw = combined.bit_count()
                        if cw < weight:
                            chosen, weight = combined, cw
                if chosen == word:
                    break
                word = chosen
                best = min(best, weight)
    return best


@lru_cache(maxsize=1)
def _dense_bank(G):
    """Twenty structural policies; extra slots optimize fresh distinct codes."""
    assert G.q == 74
    q = G.q
    fingerprints, bank = set(), []

    for slot in range(N_SLOTS):
        profile, mode = divmod(slot % 20, 4)
        sizes, column_counts = _DENSE_SUPPORTS[profile]
        sa = [list(sizes)]
        sb = [[column[i] for column in column_counts] for i in range(2)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED * 10007 + 74015003 + 100003*slot)
        phase = slot // 20

        def draw(counts, side):
            return [[_dense_cell(rng, G, count, side, i, j, mode, phase)
                     for j, count in enumerate(row)]
                    for i, row in enumerate(counts)]

        def optimize_component(side, counts):
            pool, seen = [], set()
            width = len(counts[0]) * q

            def key(item):
                reading = min(w.bit_count() for w in item["words"])
                return (min(reading, 24), -item["cycles"], item["rank"])

            def admit(grid):
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen:
                    return
                seen.add(signature)
                rows, cols = _dense_component(G, grid, side)
                rank = gf2_rank(rows)
                if rank < len(counts)*q - 4:
                    return
                if _dense_short4(cols):
                    return
                words = _dense_words(cols, rng, 3)
                if not words:
                    return
                pool.append(dict(grid=grid, cols=cols, words=words,
                                 rank=rank, cycles=_dense_cycles(cols)))
                pool.sort(key=key, reverse=True)
                del pool[5:]

            for attempt in range(48 if side == "A" else 64):
                admit(draw(counts, side))

            # Repairs change actual cell labels, not a common gauge.
            for step in range(20):
                if not pool:
                    admit(draw(counts, side))
                    continue
                parent = rng.choice(pool[:3])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                pressure = [
                    1 + sum(((w >> (j*q)) & ((1 << q)-1)).bit_count()
                            for w in parent["words"][:6])
                    for j in range(len(counts[0]))]
                j = rng.choices(range(len(pressure)),
                                weights=pressure, k=1)[0]
                i = rng.randrange(len(counts))
                grid[i][j] = _dense_cell(
                    rng, G, counts[i][j], side, i, j, mode, phase)
                if step % 5 == 0:
                    other = (j + 1) % len(counts[0])
                    r = rng.randrange(len(counts))
                    grid[r][other] = _dense_cell(
                        rng, G, counts[r][other], side, r, other,
                        mode, phase)
                admit(grid)

            # Equal independent rescreening budgets for all finalists.
            for item in pool:
                item["words"] = _dense_words(
                    item["cols"], rng, 32, item["words"])
            pool.sort(key=key, reverse=True)
            return pool[:3]

        ap = optimize_component("A", sa)
        bp = optimize_component("B", sb)
        proposals = []
        for a in ap:
            for b in bp:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (125*n <= 1000*k < 175*n):
                    continue
                d = min(24, a["words"][0].bit_count(),
                        b["words"][0].bit_count())
                cycles = a["cycles"] + b["cycles"]
                proposals.append((k*d*d, -cycles, A, B,
                                  hx, hz, n, k, a, b))
        proposals.sort(key=lambda item: item[:2], reverse=True)

        winner = None
        for proposal in proposals[:5]:
            _, negcycles, A, B, hx, hz, n, k, a, b = proposal
            fingerprint = _dense_fingerprint(hx, hz, n, k)
            if fingerprint in fingerprints:
                continue
            if _dense_short4(_dense_columns(hx, n), hz):
                continue
            if _dense_short4(_dense_columns(hz, n), hx):
                continue
            logical = _dense_dressed(
                hx, hz, n, q, a["words"], b["words"])
            # Component readings are conservative selection heuristics.
            # Only embedded, syndrome-checked non-stabilizers earn the
            # description "logical witness"; none certifies distance.
            d = min(24, logical, a["words"][0].bit_count(),
                    b["words"][0].bit_count())
            objective = (k*d*d, d, negcycles)
            if winner is None or objective > winner[0]:
                winner = (objective, A, B, hx, hz, k, fingerprint)

        if winner is None:
            bank.append(None)
            continue
        objective, A, B, hx, hz, k, fingerprint = winner
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert all(len(cell) <= 5 and len(cell) == len(set(cell))
                   for grid in (A, B) for row in grid for cell in row)
        fingerprints.add(fingerprint)
        bank.append(dict(cells={"A": A, "B": B}, k=k,
                         card=slot % 20, selection=objective))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the slot's jointly optimized, incidence-distinct filling."""
    selected = _dense_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh dense filling passed target-rate screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional label draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Two-check monomial-rich D8 components with extra terms and short-syndrome selection improve the distance-four plateau; all five measured parent leaders instead had k=245 and d30s=4, with thousands of sector-one weight-four witnesses.
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
