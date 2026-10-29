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
CONCEPT = ("For metacyclic C23 semidirect C2 with inversion action, order 46, shapes (1,3)x(1,4), triangle-free rotational binomial backbones and controlled even-weight quartics should reduce worse-basis circuit failures relative to the order-8 unit-row parent while measured rank deficits place the child in cell 0.500.")
GROUP = ("Keep the metacyclic construction family and resize its parameters to m=23,r=2,s=22; inversion squares to one modulo 23. The new size factor 13 permits q=46 and n=598. Related order-46 lower-rate beds reached about 5.7, but establish no advantage on this untried shape. Pure rotational supports deliberately test component geometry rather than claim that noncommutation causes reliability.")
SHAPES = ("Change to (1,3)x(1,4): components have 138 and 184 variables with 46 checks each. The guaranteed bound is kb=276, rate 6/13; even rotational augmentation creates controlled deficits, and complete verified ranks must give k=290, rate 0.48495. Parent failures spread across essentially all 240 logicals, with diffuse 1.0-1.2x Acol/Bcol and Acol/Brow-check enrichment, so replace both components rather than reinforce an unsupported hotspot.")
FILLING = ("Twenty briefs cross four even-weight support architectures with five distinct canonical direction contracts. Triangle-free binomial backbones are ranked using verified component lifts and short-word multiplicity; quartic labels are independently optimized. Actual CSS stabilizer quotienting rejects logicals of weight <=3 and balances hook-accessible and gate-weighted weight-four ambiguities. Both row-weight sums are <=12. CNOT budgets are 17112,19872,19320 or22632: the sparse child's larger n costs more total gates than the parent, so improved geometry, not gate reduction alone, must lower its 119-153/250 worse-basis failures. Later batches consume further canonical contracts; these screens remain uncalibrated pL surrogates.")
FAMILY_NAME = "r050_m23r2s22_13x14_evenrank_trianglecontracts_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires q=46")
    # Same metacyclic rule: x^23=y^2=1, y*x*y^-1=x^22.
    m, r, s = 23, 2, 22
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


# Number of doubleton cells in each component row.
# Their columns are disjoint, keeping every column total at two or three.
_SUPPORT_PROFILES = (
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)

# Twenty deliberate profile pairs. The unequal component lengths make
# the side assignment meaningful. Labels are independently optimized,
# never generated by applying an automorphism to an earlier slot.
_PROFILE_PAIRS = (
    (0, 3), (0, 4),
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
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
    """Choose a canonical even-support contract and classify actual logicals."""
    from itertools import combinations

    assert G.q == 46
    design, batch = idx % 20, idx // 20
    architecture, marker = divmod(design, 5)
    ratio_mode = batch
    profiles = (
        ((2, 2, 2), (2, 2, 2, 2)),
        ((2, 2, 4), (2, 2, 2, 2)),
        ((2, 2, 2), (2, 2, 2, 4)),
        ((2, 4, 4), (2, 2, 2, 2)),
    )
    pa, pb = profiles[architecture]
    sa, sb = [list(pa)], [list(pb)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    def direction(value):
        value %= 23
        return min(value, 23 - value)

    def triangle_free(steps):
        # An algebraic catalogue prefilter only. Every final short-word
        # decision below is checked on scaffold-built matrices.
        connection = {v % 23 for d in steps for v in (d, -d)}
        return not any((a + b) % 23 in connection
                       for a in connection for b in connection)

    a_sets = [steps for steps in combinations(range(1, 12), 3)
              if triangle_free(steps)]
    b_sets = [steps for steps in combinations(range(1, 12), 4)
              if triangle_free(steps)]

    def canonical(a, b):
        # Simultaneous group automorphisms and component-column
        # permutations do not create additional contracts.
        return min(
            (tuple(sorted(direction(u * d) for d in a)),
             tuple(sorted(direction(u * d) for d in b)))
            for u in range(1, 23))

    contracts = sorted({canonical(a, b) for a in a_sets for b in b_sets})
    if len(contracts) < 20:
        raise ValueError("Insufficient distinct triangle-free contracts")

    component_cache = {}

    def backbone_metric(side, steps):
        identity = (side, steps)
        if identity in component_cache:
            return component_cache[identity]
        grid = [[[(0, 0), (d, 0)] for d in steps]]
        # Zero partners are probes of the verified construction, not
        # submitted codes or an independently implemented lift.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        width = len(steps) * G.q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        stats = _kernel_statistics(rows, width)
        if stats[0] or stats[1]:
            raise ValueError("Backbone failed verified triangle exclusion")
        value = stats[2], _four_cycles(rows), stats[3]
        component_cache[identity] = value
        return value

    ranked = []
    for a, b in contracts:
        am, bm = backbone_metric("A", a), backbone_metric("B", b)
        # Repetition factors are four for A and three for B.
        # This is a shortlist heuristic, not quantum distance or pL.
        key = (max(4 * am[0], 3 * bm[0]),
               4 * am[0] + 3 * bm[0],
               max(am[1], bm[1]), am[1] + bm[1],
               max(am[2], bm[2]), am[2] + bm[2])
        ranked.append((key, a, b))
    ranked.sort()
    wanted = 5 * batch + marker
    if wanted >= len(ranked):
        raise ValueError("Unused canonical contract catalogue exhausted")
    contract_key, a_steps, b_steps = ranked[wanted]

    def draw(weights, steps):
        row = []
        for count, d in zip(weights[0], steps):
            # Origins are fixed, rather than resampled column gauges.
            # Every cell has even rotation augmentation; each side
            # retains a binomial with a nonzero order-23 difference.
            values = [0, d]
            if count == 4:
                values.extend(rng.sample(
                    [a for a in range(23) if a not in values], 2))
            cell = [(a, 0) for a in values]
            assert len(set(cell)) == count
            row.append(cell)
        return [row]

    def assess(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 598 or k != 290:
            return None
        ar, br = _component_rows(hx, hz, G.q)
        ast = _kernel_statistics(ar, A_SHAPE[1] * G.q)
        bst = _kernel_statistics(br, B_SHAPE[1] * G.q)
        if ast[0] or bst[0] or ast[1] or bst[1]:
            return None
        fa, fb = _four_cycles(ar), _four_cycles(br)
        key = (max(4 * ast[2], 3 * bst[2]),
               4 * ast[2] + 3 * bst[2],
               max(fa, fb), fa + fb,
               max(ast[3], bst[3]), ast[3] + bst[3])
        return key, hx, hz, n

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def logical_risk(checks, stabilizers, n):
        """Exact <=4 support classification; circuit costs are surrogates.

        Pair partitions count shared-check ancilla-spread opportunities.
        They do not assert that ASC realizes any particular hook order.
        """
        syndrome = columns(checks, n)
        spread = columns(stabilizers, n)
        exposure = [a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        cache = {}

        def nontrivial(word):
            answer = cache.get(word)
            if answer is not None:
                return answer
            reduced = word
            while reduced:
                p = reduced.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    cache[word] = True
                    return True
                reduced ^= old
            cache[word] = False
            return False

        bits = [1 << i for i in range(n)]
        singles = {}
        for i, s in enumerate(syndrome):
            if s == 0 and nontrivial(bits[i]):
                return None
            singles.setdefault(s, []).append(i)

        pair_buckets = {}
        hook_one = weighted_two = hook_two = quartets = 0
        hits = [0] * n
        for i in range(n):
            si, bi = syndrome[i], bits[i]
            for j in range(i):
                s = si ^ syndrome[j]
                word = bi | bits[j]
                if s == 0 and nontrivial(word):
                    return None
                for t in singles.get(s, ()):
                    if t != i and t != j:
                        if nontrivial(word | bits[t]):
                            return None

                hooks = (spread[i] & spread[j]).bit_count()
                weight = exposure[i] * exposure[j]
                bucket = pair_buckets.get(s)
                if bucket is None:
                    pair_buckets[s] = [(word, hooks, weight)]
                    continue

                for other, other_hooks, other_weight in bucket:
                    if word & other:
                        continue
                    quartet = word | other
                    fresh = quartet not in cache
                    if not nontrivial(quartet):
                        continue
                    # Each quartet has three disjoint pair partitions.
                    # Across them these sums visit its six pairs once.
                    hook_one += hooks + other_hooks
                    weighted_two += weight + other_weight
                    hook_two += hooks * other_hooks
                    if fresh:
                        quartets += 1
                        remaining = quartet
                        while remaining:
                            bit = remaining & -remaining
                            hits[bit.bit_length() - 1] += 1
                            remaining ^= bit
                bucket.append((word, hooks, weight))

        return hook_one, weighted_two, hook_two, quartets, max(hits)

    # Cheap component screening precedes the more expensive quantum
    # classification. Retain distinct relative-label proposals.
    pool = []
    seen = set()
    for trial in range(1 if architecture == 0 else 256):
        A, B = draw(sa, a_steps), draw(sb, b_steps)
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        result = assess(A, B)
        if result is None:
            continue
        key, hx, hz, n = result
        pool.append((key, trial, A, B, hx, hz, n))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[12:]

    best = None
    for component_key, trial, A, B, hx, hz, n in pool:
        pz = logical_risk(hx, hz, n)
        if pz is None:
            continue
        px = logical_risk(hz, hx, n)
        if px is None:
            continue
        # Minimax protects the weaker basis. Gate exposure weights the
        # two-data-fault ambiguities; shared ancillas identify possible
        # lower-order spread faults without claiming a circuit simulation.
        score = tuple(v for t in range(5)
                      for v in (max(px[t], pz[t]), px[t] + pz[t]))
        score += component_key
        if best is None or score < best[0]:
            best = (score, A, B, hx, hz)

    if best is None:
        raise ValueError("No target-rate filling passed quantum <=3 screen")
    score, A, B, hx, hz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    expected_cnots = (17112, 19872, 19320, 22632)[architecture]
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
    assert 0.475 <= 290 / 598 < 0.525
    return {"A": A, "B": B, "design": design, "ratio_mode": ratio_mode,
            "k": 290, "circuit_risk": score,
            "contract": (architecture, wanted, a_steps, b_steps),
            "backbone_metric": contract_key}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete verified even-weight cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Dispersed doubletons and component short-word/cycle optimization sufficiently suppress collective failures on the D8 high-rate bed: fitness was 3.2454, measured worse-basis failures were 117..147/250, and every measured code had abundant weight-four logicals.
# REFUTED: Unit first rows and sparse pivot columns improve D8 high-rate memory through reduced extraction work alone: fitness was 3.2422 versus 3.2454, with 119..153/250 worse-basis failures, broad logical flips and thousands of weight-four witnesses despite mostly first-pass settlement.
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
