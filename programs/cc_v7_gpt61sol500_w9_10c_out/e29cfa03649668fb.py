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
CONCEPT = ("Cyclic C36, order 36, A=(2,3), B=(2,4): independently optimized mixed-degree constituents followed by verified logical-witness reduction should supply five novel measured readings >=15 in cell 0.150 and improve its 28.5 record; this is a falsifiable measurement hypothesis.")
GROUP = ("Jump from nonabelian Dic6 to cyclic C36. Related cyclic (2,3)x(2,4) beds measured fitness 28.5 at q=31, 26.7 at q=33 and 27.3 at q=35. The cyclic q=36 bed is unlisted and gives n=576. Even-order rank deficits are screened rather than assumed beneficial; neither distance nor archive novelty transfers from those beds.")
SHAPES = ("Use A=(2,3), B=(2,4): n=576, kb=72 and guaranteed rate 0.125. The size factor sixteen allows q<=37, so q=36 fits. Constituents have 108 variables/72 checks and 144 variables/72 checks. Parent measured weight-four minima broadly occupied sector-1 Acol0..2/Bcol0..6. Changing both constituents replaces that incidence topology. Verified complete ranks require 0.126<=k/n<0.175, placing every published candidate strictly in cell 0.150.")
FILLING = ("Four A and six B support profiles combine saturated and mixed-degree columns with deliberate relative-gap policies. Both CSS term-count sums must remain <=9, hence also <=12; cells contain at most three distinct elements. Independently optimize verified constituents, exactly exclude kernel weights 1..4, and give retained finalists equal information-set budgets. Pair heterogeneous finalists using actual full ranks, exact logical exclusion below four, and explicitly nontrivial embedded witnesses shortened by stabilizer pairs. Different intrinsic uncoloured Tanner invariants separate submitted designs conservatively. Larger witness readings remain upper bounds, and the evaluator's archive remains authoritative.")
FAMILY_NAME = "r015_cyclic36_23x24_independent_mixeddegree_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this cyclic family fixes order 36")
    els = [(a,) for a in range(36)]

    def red(g):
        return (int(g[0]) % 36,)

    def mul(g, h):
        return ((g[0] + h[0]) % 36,)

    def inv(g):
        return ((-g[0]) % 36,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


_QC_A_PROFILES = (
    ((1, 1, 3), (1, 3, 1)),
    ((1, 1, 3), (2, 2, 1)),
    ((1, 2, 2), (2, 1, 2)),
    ((1, 1, 2), (2, 2, 1)),
)
_QC_B_PROFILES = (
    ((1, 1, 1, 2), (1, 1, 2, 1)),
    ((1, 1, 1, 2), (1, 1, 1, 2)),
    ((0, 2, 1, 2), (2, 0, 2, 1)),
    ((1, 1, 1, 2), (0, 1, 3, 1)),
    ((1, 1, 0, 3), (1, 1, 3, 0)),
    ((1, 1, 2, 2), (2, 2, 1, 1)),
)


def _qc_draw(rng, G, profile, policy):
    """Prescribe relative gaps, not automorphism copies of a saved grid."""
    units = (1, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35)
    even = tuple(a for a in range(2, 36, 2) if a != 18)
    thirds = tuple(a for a in range(3, 36, 3) if a != 18)
    broad = tuple(a for a in range(1, 36) if a != 18)
    pools = (units, even, thirds, broad, units)
    grid = []
    for i, counts in enumerate(profile):
        row = []
        for j, count in enumerate(counts):
            if not count:
                row.append([])
                continue
            first = rng.randrange(36)
            values = [first]
            if count >= 2:
                gap = rng.choice(pools[(policy + i + j) % 5])
                values.append((first + gap) % 36)
            if count == 3:
                # A third term crosses the parity or mod-three classes
                # in the restricted-gap policies.
                choices = [
                    a for a in range(36) if a not in values
                    and (policy not in (1, 2)
                         or (a - first) % (2 if policy == 1 else 3))
                ]
                values.append(rng.choice(choices))
            row.append([G.elements[a] for a in values])
        grid.append(row)
    return grid


def _qc_component(G, grid, side):
    """Extract directly from a verified lift with a zero auxiliary side."""
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
    else:
        _, rows, _ = lifted_matrices(G, [[[]]], grid)
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _qc_probe(rng, columns, passes, seed=0):
    """Heuristic kernel upper bound with an explicit nonincreasing witness."""
    width = len(columns)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight:
            best, weight = word, w

    for _ in range(passes):
        order = list(range(width))
        rng.shuffle(order)
        pivots, basis = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)

        rng.shuffle(basis)
        for i in range(len(basis)):
            word = basis[i]
            w = word.bit_count()
            for j in range(len(basis)):
                if i != j:
                    trial = word ^ basis[j]
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w = trial, tw
            basis[i] = word
            inspect(word)

        for i, first in enumerate(basis):
            for second in basis[:i]:
                inspect(first ^ second)
        short = sorted(basis, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for third in short[:j]:
                    inspect(pair ^ third)

        for _ in range(2):
            starting = best
            for i, first in enumerate(basis):
                inspect(starting ^ first)
                for second in basis[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

    if not best:
        raise ValueError("component kernel witness absent")
    syndrome, bits = 0, best
    while bits:
        bit = bits & -bits
        syndrome ^= columns[bit.bit_length() - 1]
        bits ^= bit
    assert syndrome == 0
    return weight, best


def _qc_bank(G, side):
    """Optimize structural cards independently with equal finalist budgets."""
    profiles = _QC_A_PROFILES if side == "A" else _QC_B_PROFILES
    policies = 5 if side == "A" else 4
    width = len(profiles[0][0]) * G.q
    records, fillings = [], set()

    for pi, profile in enumerate(profiles):
        for policy in range(policies):
            local = random.Random(
                BASE_SEED ^ (0xC36150 + 1009 * pi + 7919 * policy
                             + 104729 * int(side == "B")))
            raw = []
            for trial in range(40):
                grid = _qc_draw(local, G, profile, policy)
                key = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
                if key in fillings:
                    continue
                fillings.add(key)
                rows = _qc_component(G, grid, side)
                if _component_floor(rows, width) != 5:
                    continue
                columns = _column_syndromes(rows, width)
                cost = _four_cycles(rows)
                raw.append((cost, trial, grid, rows, columns))
            raw.sort(key=lambda r: r[:2])

            shortlist = []
            for cost, trial, grid, rows, columns in raw[:5]:
                reading, witness = _qc_probe(local, columns, 2)
                shortlist.append(dict(
                    grid=grid, rows=rows, columns=columns,
                    cost=cost, trial=trial, reading=reading,
                    witness=witness, card=(pi, policy)))
            shortlist.sort(key=lambda r: (
                -r["reading"], r["cost"], r["trial"]))

            # Every retained record receives exactly 2+14 passes.
            for record in shortlist[:2]:
                reading, witness = _qc_probe(
                    local, record["columns"], 14, record["witness"])
                record["reading"], record["witness"] = reading, witness
                records.append(record)

    records.sort(key=lambda r: (
        -r["reading"], r["cost"], r["card"], r["trial"]))
    if not records:
        raise ValueError("no kernel-four-clean cyclic constituents")
    return records


def _qc_graph_key(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive."""
    from collections import Counter
    rows = list(hx) + list(hz)
    m = len(rows)
    adjacency = [0] * m
    intersections = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                intersections[i][j] = intersections[j][i] = overlap

    profiles = []
    for i in range(m):
        histogram = Counter(
            (rows[j].bit_count(),
             intersections[i].get(j, 0),
             adjacency[j].bit_count(),
             (adjacency[i] & adjacency[j]).bit_count())
            for j in range(m) if i != j)
        profiles.append((rows[i].bit_count(), adjacency[i].bit_count(),
                         tuple(sorted(histogram.items()))))
    columns = _column_syndromes(rows, n)
    return (
        n,
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(profiles)),
    )


def _qc_logical_reading(G, a, b, hx, hz, n):
    """Check and shorten actual nontrivial component-derived logicals."""
    q, ca, cb = G.q, A_SHAPE[1], B_SHAPE[1]
    overall = None
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        best, weight = 0, n + 1
        for block in range(ca if kind == "X" else cb):
            if kind == "X":
                word = b["witness"] << (block * cb * q)
            else:
                word, source = 0, a["witness"]
                while source:
                    bit = source & -source
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * cb + block) * q + h)
                    source ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if nontrivial(word) and word.bit_count() < weight:
                best, weight = word, word.bit_count()

        if not best:
            return None
        # All moves preserve the already verified nontrivial logical class.
        for _ in range(2):
            starting = best
            for i, first in enumerate(stabilizers):
                trial = starting ^ first
                tw = trial.bit_count()
                if tw < weight:
                    best, weight = trial, tw
                for second in stabilizers[:i]:
                    trial = starting ^ first ^ second
                    tw = trial.bit_count()
                    if tw < weight:
                        best, weight = trial, tw
            if best == starting:
                break
        assert best and nontrivial(best)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
        overall = weight if overall is None else min(overall, weight)
    return overall


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


from functools import lru_cache


@lru_cache(maxsize=1)
def _qc_portfolio(G):
    """Deterministic immutable output; memoization changes only build cost."""
    aa, bb = _qc_bank(G, "A"), _qc_bank(G, "B")
    cap = min(9, 12, W_MAX)
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            sa = [[len(c) for c in row] for row in a["grid"]]
            sb = [[len(c) for c in row] for row in b["grid"]]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            plans.append((
                -min(a["reading"], b["reading"]),
                -a["reading"] - b["reading"],
                a["cost"] + b["cost"], ai, bi))
    plans.sort()

    entries, graph_keys = [], set()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 576 and k > 0
                and 126 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Verify bank witnesses against constituents extracted from the
        # actual submitted pair, not only against auxiliary lifts.
        ar, br = _extract_components(G, hx, hz)
        if ar != a["rows"] or br != b["rows"]:
            raise ValueError("verified constituent extraction disagrees")
        if not _no_logicals_below_four(hx, hz, n):
            continue
        key = _qc_graph_key(hx, hz, n)
        if key in graph_keys:
            continue

        reading = _qc_logical_reading(G, a, b, hx, hz, n)
        if reading is None or reading < 5:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified cyclic lift failed CSS")
        assert lifted_k(G, A, B) == k
        assert all(len(set(cell)) == len(cell) <= 3
                   for grid in (A, B) for row in grid for cell in row)
        graph_keys.add(key)

        quality = (k * reading * reading / n, reading, k,
                   -a["cost"] - b["cost"])
        screen = (
            ("k", k),
            ("component_readings", (a["reading"], b["reading"])),
            ("logical_witness_reading", reading),
            ("component_kernel_weights_1_to_4_excluded", True),
            ("quantum_weights_1_to_3_excluded", True),
            ("design_cards", (a["card"], b["card"])),
            ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("rank_gain_above_bound", k - 72),
        )
        entries.append((quality, freeze(A), freeze(B), screen))
        # Compare a wider equally screened portfolio before publication.
        if len(entries) >= max(N_SLOTS, 5) + 28:
            break

    entries.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete optimized, invariant-separated design."""
    portfolio = _qc_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("novel target-rate cyclic portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete, already screened cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Twenty Dic6 relative-order/quotient-support cards supply twenty inequivalent designs or escape weight four: current slots 0..19 retained four classes, and all five measured leaders had d30s=4 with sector-1 witnesses.
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
