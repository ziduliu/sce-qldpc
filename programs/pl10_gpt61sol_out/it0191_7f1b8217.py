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
CONCEPT = ("On metacyclic Z9:Z1(s=1), order 9, shapes (2,7)x(2,8), odd-order distinct-voltage backbones with distributed extra terms should improve worse-basis circuit fitness beyond 3.1978 by reducing short-logical ambiguity.")
GROUP = ("Keep the metacyclic construction rule and change only m,r,s to 9,1,1. The new size factor 60 makes order 14 invalid: n=840. Order 9 gives n=540 and has no involutions. Related order-eight two-row beds scored 3.2454 and 3.2422; these support a new shape/order experiment, not a group-only advantage claim.")
SHAPES = ("Target cell 0.500 with (2,7)x(2,8): n=540, kb=270, component lengths 63 and 72, and 18 checks on each side. Final verified ranks require 0.475<=k/n<0.525. Parent failures affected essentially all 213 logicals with diffuse 1.0-1.2x Acol/Bcol and Acol/Brow enrichment, motivating replacement of both components rather than a localized repair.")
FILLING = ("Twenty contracts comprise seventeen distributed-doubleton pairs and three concentrated-support tests. Explicit row/column budgets bound both check weights by twelve; six-round CNOT budgets are 15822-18468. Gauge-fixed distinct cyclic voltages are optimized using component actions extracted exclusively from verified lifts. Full verified matrices screen rate and nontrivial weight-three logicals, then rank truncated data-single/data-pair/shared-ancilla fault ambiguities modulo stabilizers in both bases. Extra batches select different invariant profiles from each deterministic bank, never relabel a saved code. Target fewer collective, syndrome-valid failures than the parent's 110-147 X failures/250 shots. This model omits measurement faults and actual ASC ordering and is not measured pL.")
FAMILY_NAME = "r050_metacyclic_m9r1s1_27x28_odd_voltage_faultbank_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 9

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires order 9")
    # Same metacyclic rule; trivial complement, with s^r=1 modulo m.
    m, r, s = 9, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)


def _distinct_terms(rng, G, count):
    terms = []
    seen = set()
    while len(terms) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in seen:
            seen.add(key)
            terms.append(g)
    return terms


def _b_profile(rng, extra, zeros):
    """Seven live columns; fixed total terms and fixed zero-cell count."""
    for _ in range(2000):
        triple = set(rng.sample(range(7), extra))
        sparse = set(rng.sample(range(7), zeros))
        sb = [[0] * 7 for _ in range(2)]
        for j in range(7):
            degree = 3 if j in triple else 2
            if j in sparse:
                a, b = 0, degree
            else:
                a, b = 1, degree - 1
            if rng.randrange(2):
                a, b = b, a
            sb[0][j], sb[1][j] = a, b
        if 0 < sum(sb[0]) <= 9 and 0 < sum(sb[1]) <= 9:
            return sb
    raise ValueError("could not realize deliberate B profile")


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def NEW_CTX(rng, G, idx):
    """Optimize one support contract; all state is local to this pure call."""
    design, wanted = idx % 20, idx // 20
    q = G.q
    assert q == 9
    srng = random.Random(9002708 + 1009 * design)
    shapes = (A_SHAPE, B_SHAPE)
    ap = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    bp = ((0, 0), (0, 1), (1, 1))
    cards = [(a, b) for a in range(len(ap)) for b in range(len(bp))
             if a or b]
    assert len(cards) == 17

    def distributed(width, extras):
        counts = [[1] * width for _ in range(2)]
        cursor = 0
        for i, number in enumerate(extras):
            for j in range(cursor, cursor + number):
                counts[i][j] = 2
            cursor += number
        return counts

    if design < len(cards):
        a, b = cards[design]
        sa = distributed(7, ap[a])
        sb = distributed(8, bp[b])
    else:
        sa, sb = distributed(7, (0, 0)), distributed(8, (0, 0))
        if design == 17:
            # A column degree four: B rows remain eight.
            sa[0][0] = 3
        elif design == 18:
            # B column degree four: A rows remain seven.
            sb[0][0] = 3
        else:
            # A live degree-three column with a deliberate zero cell.
            sa[0][0], sa[1][0] = 3, 0
            sb[1][0] = 2

    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= CELL_MAX for g in (sa, sb)
               for row in g for w in row)

    # Cache exact singleton actions from the verified construction.
    # Linear composition below introduces no independent lift convention.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G.idx(g)] = tuple(row & mask for row in hx)
        actions[1][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    for t, word in enumerate(actions[side][G.idx(g)]):
                        out[t] ^= word << (j * q)
            rows.extend(out)
        return rows

    def draw(counts):
        width = len(counts[0])
        voltages = srng.sample(range(q), width)
        grid = []
        for i, row in enumerate(counts):
            cells = []
            for j, weight in enumerate(row):
                if not weight:
                    cells.append([])
                    continue
                origin = 0 if i == 0 else voltages[j]
                offsets = [0] + srng.sample(range(1, q), weight - 1)
                cells.append([((origin + d) % q, 0) for d in offsets])
            grid.append(cells)
        return grid

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def assess(A, B):
        ar, br = component(A, 0), component(B, 1)
        ast = _syndrome_risk(ar, 7 * q)
        bst = _syndrome_risk(br, 8 * q)
        if ast is None or bst is None:
            return None
        deficits = (2 * q - gf2_rank(ar), 2 * q - gf2_rank(br))
        return (max(ast[0], bst[0]), ast[0] + bst[0],
                max(ast[1], bst[1]), ast[1] + bst[1],
                max(deficits), sum(deficits))

    groups, seen = {}, set()

    def remember(A, B, serial):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            return None
        seen.add(signature)
        key = assess(A, B)
        if key is None:
            return None
        bucket = groups.setdefault(key, [])
        if len(bucket) < 3:
            bucket.append((serial, clone(A), clone(B)))
        return key

    leaders = []
    for trial in range(256):
        A, B = draw(sa), draw(sb)
        key = remember(A, B, trial)
        if key is not None:
            leaders.append((key, trial, A, B))
            leaders.sort(key=lambda item: item[:2])
            del leaders[4:]

    # Refinement changes relative supports or inter-row voltages.
    # It never creates a slot by applying an automorphism to a saved code.
    serial = 256
    for key, trial, A, B in leaders:
        for step in range(64):
            grids = [clone(A), clone(B)]
            side = srng.randrange(2)
            counts = (sa, sb)[side]
            sites = [(i, j) for i in range(2)
                     for j in range(shapes[side][1]) if counts[i][j]]
            i, j = srng.choice(sites)
            cell = grids[side][i][j]
            if i == 1 and srng.random() < 0.5:
                shift = srng.randrange(1, q)
                grids[side][i][j] = [((g[0] + shift) % q, 0) for g in cell]
            elif len(cell) > 1:
                position = srng.randrange(1, len(cell))
                options = [g for g in G.elements if g not in cell]
                cell[position] = srng.choice(options)
            else:
                cell[0] = G.pick(srng)
            AA, BB = grids
            proposed = remember(AA, BB, serial)
            serial += 1
            if proposed is not None and proposed <= key:
                A, B, key = AA, BB, proposed

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def quotient_columns(rows, width):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        residual = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                residual[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length() - 1]
                tail ^= bit
            residual[c] = value
        return residual

    def fault_profile(checks, stabilizers, width):
        """Exact short-logical exclusion plus a truncated fault surrogate."""
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        rates = [64 * (a.bit_count() + b.bit_count() + 1)
                 for a, b in zip(syndrome, spread)]
        hook_mass = [
            80000 // (w * (w - 1) // 2) if w > 1 else 0
            for w in (row.bit_count() for row in stabilizers)
        ]
        buckets = {0: {0: 100000000}}

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0) + mass

        for i, s in enumerate(syndrome):
            add(s, residual[i], rates[i] * 10000)

        for i, si in enumerate(syndrome):
            ri, hi = residual[i], spread[i]
            for j in range(i):
                s, label = si ^ syndrome[j], ri ^ residual[j]
                if s == 0 and label:
                    return None
                third = singles.get(s)
                if third is not None and third != label:
                    return None
                # Shared-check pairs are possible spread opportunities,
                # not certificates of an ASC hook ordering.
                common = hi & spread[j]
                mass = rates[i] * rates[j]
                while common:
                    bit = common & -common
                    mass += hook_mass[bit.bit_length() - 1]
                    common ^= bit
                add(s, label, mass)

        loss = ambiguous = worst = 0
        for classes in buckets.values():
            if len(classes) > 1:
                error = sum(classes.values()) - max(classes.values())
                loss += error
                ambiguous += 1
                worst = max(worst, error)
        return loss, ambiguous, worst

    shortlist = []
    keys = sorted(groups)
    for position in range(3):
        for key in keys:
            if position < len(groups[key]):
                shortlist.append((key,) + groups[key][position])
                if len(shortlist) == 32:
                    break
        if len(shortlist) == 32:
            break

    book = {}
    for coarse, trial, A, B in shortlist:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 540 or not (k > 0 and 19 * n <= 40 * k < 21 * n):
            continue

        # Confirm both cached component conventions against final matrices.
        actual_a = [
            sum(((hx[(i * 8) * q + t] >> (j * 8 * q)) & mask) << (j * q)
                for j in range(7))
            for i in range(2) for t in range(q)
        ]
        actual_b = [row & ((1 << (8 * q)) - 1) for row in hz[:2 * q]]
        assert actual_a == component(A, 0)
        assert actual_b == component(B, 1)
        pz = fault_profile(hx, hz, n)
        if pz is None:
            continue
        px = fault_profile(hz, hx, n)
        if px is None:
            continue
        invariant = (k, tuple(sorted((px, pz))))
        score = (max(px[0], pz[0]), px[0] + pz[0],
                 max(px[2], pz[2]), px[2] + pz[2],
                 max(px[1], pz[1]), px[1] + pz[1])
        item = (score, trial, A, B, hx, hz, k, invariant)
        previous = book.get(invariant)
        if previous is None or item[:2] < previous[:2]:
            book[invariant] = item

    # Distinct invariant values exclude Tanner-isomorphic bank entries.
    # Across different contracts, arbitrary equivalence remains subject
    # to the evaluator's authoritative deduplication.
    entries = sorted(book.values(), key=lambda item: item[:2])
    if wanted >= len(entries):
        raise ValueError("insufficient unused invariant-distinct designs")
    score, trial, A, B, hx, hz, k, invariant = entries[wanted]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    cnots = 6 * sum(row.bit_count() for row in hx + hz)
    assert cnots == 6 * q * (
        10 * sum(map(sum, sa)) + 9 * sum(map(sum, sb)))
    return {"A": A, "B": B, "k": k, "brief": (design, wanted),
            "fault_cost": score, "invariant": invariant, "cnots": cnots}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected low-load cell, including designed zeros."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Distinct rotational binomial steps and raw syndrome-ambiguity screening eliminate the immediate parent's short-logical problem: measured codes retained abundant weight-three witnesses, 108-139 X failures per 250 shots, and fitness 3.2219.
# REFUTED: Replacing weight-three witnesses with weight-four witnesses alone improves this bed's circuit fitness: the shown odd-A triangle-free-B program scored 3.2061 versus 3.2219, with 115-131 X failures per 250 shots.
# REFUTED: Unit-column low-load fault-quotient optimization sufficiently suppresses D14 high-rate failures: fitness fell from 3.2219 to 3.1978, measured X failures were 110-147/250, and weight-three sector-1 witnesses persisted despite almost universal first-pass settlement.
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
