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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dicyclic Dic2, isomorphic to Q8, order eight, A=(3,9), B=(2,7): cyclic-subgroup A components paired with regular and mixed-degree B components should produce five novel distance-four readings and improve fitness above 7.3504 in cell 0.450.")
GROUP = ("Preserve the dicyclic construction rule. Size factor 69 excludes order 24 and permits q<=8; Dic2 has order eight and gives n=552. Its three cyclic order-four subgroups permit different component connectivity and relative subgroup couplings. The parent measured five distance-four leaders but severe duplicate collapse; subgroup structure and rank screens are hypotheses, not distance guarantees.")
SHAPES = ("A=(3,9), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.43478. Constituents have 72 variables/24 checks and 56 variables/16 checks. Parent minimum logicals broadly occupied sector-1 Acol0..2/Bcol0..6. Three A rows permit internal cycles and pairwise row matchings absent from its single-row A. Verified ranks enforce actual rate in [0.425,0.475).")
FILLING = ("Twenty prescriptions cross five degree-two A incidence profiles with four B support profiles. A rows total six and columns two; B rows total seven and columns at most three, giving HX<=9 and HZ=9. Subgroup-supported A proposals receive priority but connected alternatives remain available. Verified constituent kernels exclude weights below four; complete stabilizer-aware tests exclude quantum logicals below four. Batch optimization prioritizes actual target-band k, then constituent four-cycle burden. An immutable portfolio conservatively rejects equal rooted intrinsic Tanner invariants. Distance readings above four and novelty against the unseen archive are not assumed.")
FAMILY_NAME = "r045_dic2_q8_39x27_subgroup_incidence_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic2 requires order eight")
    # Same dicyclic rule: x^(2t)=1, y^2=x^t, y*x*y^-1=x^-1.
    # Factor 69 requires q<=8; t=2 gives Dic2, also called Q8.
    t, m = 2, 4
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + t * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else -1) * c + t * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + t) % m, 1)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 9)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


_A_INTERNAL = (
    (1, 1, 1),
    (0, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 2, 2),
)


def _incidence_filling(rng, G, card, epoch):
    """Five A incidence structures crossed with four B support structures."""
    amode, bmode = divmod(card % 20, 4)
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    })
    assert len(directions) == 3

    A = [[] for _ in range(3)]

    def a_column(entries):
        for i in range(3):
            A[i].append(list(entries.get(i, ())))

    if amode == 0 and epoch % 3 != 2:
        # In each coset of <g>, the component is K6,6.
        # The two cosets are genuine components, not different labels
        # of a connected A constituent.
        g = directions[epoch % 3]
        powers = [G.e]
        for _ in range(3):
            powers.append(G.mul(powers[-1], g))
        phases = [0, rng.randrange(2), rng.randrange(2)]
        for i in range(3):
            a_column({i: (G.e, g)})
        for i, j in ((0, 1), (0, 2), (1, 2)):
            parity = 1 ^ phases[i] ^ phases[j]
            values = [powers[parity], powers[parity + 2]]
            rng.shuffle(values)
            for voltage in values:
                a_column({i: (G.e,), j: (voltage,)})
    else:
        internal = _A_INTERNAL[amode]
        degrees = [6 - 2 * p for p in internal]
        counts = (
            (degrees[0] + degrees[1] - degrees[2]) // 2,
            (degrees[0] + degrees[2] - degrees[1]) // 2,
            (degrees[1] + degrees[2] - degrees[0]) // 2,
        )

        # Q8 has three nontrivial homomorphisms to C2.
        grade_mode = epoch % 3

        def grade(g):
            a, b = g
            return (a if grade_mode == 0 else
                    b if grade_mode == 1 else a + b) % 2

        graded = epoch % 4 != 3
        gaps = ([g for g in directions if grade(g)]
                if graded else directions)
        if any(p > len(gaps) for p in internal):
            return None
        for i, p in enumerate(internal):
            for gap in rng.sample(gaps, p):
                a_column({i: (G.e, gap)})

        phases = [0, rng.randrange(2), rng.randrange(2)]
        for (i, j), count in zip(((0, 1), (0, 2), (1, 2)), counts):
            parity = 1 ^ phases[i] ^ phases[j]
            labels = ([g for g in G.elements if grade(g) == parity]
                      if graded else list(G.elements))
            if count < 0 or count > len(labels):
                return None
            for voltage in rng.sample(labels, count):
                a_column({i: (G.e,), j: (voltage,)})

    B = [[], []]
    if bmode < 2:
        internal = 0 if bmode == 0 else 2
        for i in range(2):
            for gap in rng.sample(directions, internal):
                B[i].append([G.e, gap])
                B[1 - i].append([])
        cross = 7 - 2 * internal
        for voltage in rng.sample(list(G.elements), cross):
            B[0].append([G.e])
            B[1].append([voltage])
    else:
        # Both profiles have row sums seven and column sums 1..3.
        profiles = (
            ((1, 0), (1, 2)) + ((1, 1),) * 5,
            ((1, 0), (3, 0)) + ((1, 1),) * 3
            + ((0, 2),) * 2,
        )
        for top, bottom in profiles[bmode - 2]:
            for i, count in enumerate((top, bottom)):
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    cell = [first, G.mul(first, rng.choice(directions))]
                else:
                    cell = rng.sample(list(G.elements), count)
                B[i].append(cell)

    assert all(len(row) == 9 for row in A)
    assert all(sum(map(len, row)) == 6 for row in A)
    assert all(sum(len(A[i][j]) for i in range(3)) == 2
               for j in range(9))
    assert all(len(row) == 7 for row in B)
    assert all(sum(map(len, row)) == 7 for row in B)
    return A, B


def _extract_components(G, hx, hz):
    """Extract components from the verified lift, not a separate lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    # j=0 HZ rows already have consecutive B-column blocks.
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _column_syndromes(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _component_floor(rows, n):
    """Exact minimum through weight four; return 5 if none is found."""
    columns = _column_syndromes(rows, n)
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2
    pairs = set()
    found4 = False
    for i, first in enumerate(columns):
        for second in columns[:i]:
            syndrome = first ^ second
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    return 4 if found4 else 5


def _no_logicals_below_four(hx, hz, n):
    """Complete CSS logical test for weights one, two, and three.

    A weight-three zero syndrome is a pair/single collision.
    Row-space reduction distinguishes logicals from stabilizers.
    """
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = {}
        for row in stabilizers:
            while row:
                pivot = row.bit_length() - 1
                if pivot in basis:
                    row ^= basis[pivot]
                else:
                    basis[pivot] = row
                    break

        def nontrivial(word):
            while word:
                pivot = word.bit_length() - 1
                if pivot not in basis:
                    return True
                word ^= basis[pivot]
            return False

        columns = _column_syndromes(checks, n)
        singles = {0: 0}
        for i, syndrome in enumerate(columns):
            support = 1 << i
            previous = singles.get(syndrome)
            if previous is None:
                singles[syndrome] = support
            elif nontrivial(previous ^ support):
                return False
        for i, first in enumerate(columns):
            for j in range(i):
                previous = singles.get(first ^ columns[j])
                if previous is not None:
                    if nontrivial(previous ^ (1 << i) ^ (1 << j)):
                        return False
    return True


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _portfolio_signature(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant, not a complete iso test.

    High-degree qubits are identified by degree, never LP block labels.
    Rooted refinement retains check hyperedges and low-qubit corners.
    Unequal signatures certify inequivalence; equality is discarded.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _column_syndromes(rows, n)
    high = [j for j, c in enumerate(columns) if c.bit_count() >= 10]
    positions = {j: i for i, j in enumerate(high)}
    endpoints, tags, check_ends = [], [], []

    for row in rows:
        ends = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                ends.append(positions[j])
            bits ^= bit
        ends = tuple(sorted(ends))
        check_ends.append(ends)
        endpoints.append(ends)
        tags.append((0, row.bit_count()))

    for j, syndrome in enumerate(columns):
        if j in positions:
            continue
        ends, weights = [], []
        bits = syndrome
        while bits:
            bit = bits & -bits
            r = bit.bit_length() - 1
            ends.extend(check_ends[r])
            weights.append(rows[r].bit_count())
            bits ^= bit
        endpoints.append(tuple(sorted(ends)))
        tags.append((1, tuple(sorted(weights))))

    incident = [[] for _ in high]
    for edge, ends in enumerate(endpoints):
        for v in ends:
            incident[v].append(edge)

    def compress(signatures):
        mapping = {s: i for i, s in enumerate(sorted(set(signatures)))}
        return [mapping[s] for s in signatures]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(4):
            es = [(tag, tuple(sorted(colours[v] for v in ends)))
                  for tag, ends in zip(tags, endpoints)]
            ec = compress(es)
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            colours = compress(vs)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    # Check-intersection profiles supplement rooted contraction.
    overlaps = [Counter() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i][(rows[j].bit_count(), overlap)] += 1
            overlaps[j][(row.bit_count(), overlap)] += 1

    return (
        n,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(rooted)),
        tuple(sorted((rows[i].bit_count(),
                      tuple(sorted(overlaps[i].items())))
                     for i in range(len(rows)))),
    )


from functools import lru_cache


@lru_cache(maxsize=1)
def _incidence_portfolio(G):
    """Build one deterministic immutable portfolio, independent of call order.

    Proposals are optimized in batches. Only invariant-distinct screened
    winners enter the portfolio; no mutable cross-slot selection state
    is exposed to NEW_CTX or CELL.
    """
    local = random.Random(BASE_SEED ^ 0x8392745)
    cap = min(9, W_MAX)
    entries, graph_keys, filling_keys = [], set(), set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for batch in range(96):
        candidates = []
        # Three trials for each of the twenty structural prescriptions.
        for trial in range(60):
            attempt = 60 * batch + trial
            card, epoch = attempt % 20, attempt // 20
            proposal = _incidence_filling(local, G, card, epoch)
            if proposal is None:
                continue
            A, B = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                raise ValueError("new-shape support budget failed")
            if any(len(set(cell)) != len(cell)
                   for grid in (A, B) for row in grid for cell in row):
                raise ValueError("new-shape cell contains duplicate elements")

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            ar, br = _extract_components(G, hx, hz)
            da = _component_floor(ar, 9 * G.q)
            db = _component_floor(br, 7 * G.q)
            if min(da, db) < 4:
                continue
            cycles = _four_cycles(ar) + _four_cycles(br)
            priority = (-min(da, db), -k, cycles, attempt)
            candidates.append(
                (priority, A, B, hx, hz, n, k, da, db, rx, rz, card))

        candidates.sort(key=lambda record: record[0])
        for _, A, B, hx, hz, n, k, da, db, rx, rz, card in candidates:
            signature = _portfolio_signature(hx, hz, n)
            if signature in graph_keys:
                continue
            if not _no_logicals_below_four(hx, hz, n):
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified new-shape matrices failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            graph_keys.add(signature)
            screen = (
                ("k", k),
                ("component_floors", (da, db)),
                ("quantum_below_four_excluded", True),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("design_card", card),
            )
            entries.append((freeze(A), freeze(B), screen))
            if len(entries) == N_SLOTS:
                return tuple(entries)

    # Never recycle an earlier accepted code to fill missing positions.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete optimized invariant-distinct design."""
    portfolio = _incidence_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("distinct new-shape incidence portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete, already screened cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Twenty relative-order/quotient-support cards provide twenty inequivalent Dic6 designs: current slots 0..19 collapsed to four classes, and all five measured leaders still had sector-1 weight-four logicals.
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
