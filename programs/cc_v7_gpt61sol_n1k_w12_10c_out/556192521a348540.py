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
CONCEPT = ("A5 x C3, order 180, shapes (1,2)x(1,3): noncommuting trinomials and independently screened component kernels should yield five measured readings >=16 and beat fitness 59.4286 in cell 0.300. This is a measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from the soluble order-112 affine group to the nonsolvable direct product of A5 with central C3. This order-180 bed is absent from BEDS. The simple factor supplies noncommuting relative labels; central-orbit restrictions explicitly test another possible weak family. No measured distance transfer is assumed.")
SHAPES = ("A=(1,2), B=(1,3), q=180 give n=1260, k>=360 and guaranteed rate 2/7. Components have 360/180 and 540/180 variables/checks. Parent sector-1 minima included Acol2 across all Bcols and Bcol1/2 across all Acols; independently protect both new kernels, with deeper searches on higher-rate B. Exact ranks retain [0.275,0.325).")
FILLING = ("All cells contain three distinct labels, giving HX weight 6+3=9 and HZ weight 9+3=12. Twenty cards cross five A central-coordinate profiles with four B profiles. Independently shortlist fresh noncommuting permutation geometry by verified connectivity, kernel-3/4 counts and cycles; retest with independent kernel descent, localized restrictions and central-C3 orbit probes. Full ranks, CSS and both weight sums gate final cells. Sampled component readings are upper bounds, not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r030_A5xC3_12x13_noncommuting_independent_orbits_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 180

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A5 x C3 requires order 180")
    # First five coordinates encode an even permutation by its images.
    # Last coordinate is central C3; multiplication composes permutations.
    even = [
        p for p in permutations(range(5))
        if sum(p[i] > p[j] for i in range(5)
               for j in range(i+1, 5)) % 2 == 0
    ]
    els = [p + (z,) for z in range(3) for p in even]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) % 5 for i in range(5)) + (int(g[5]) % 3,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(5)) + ((g[5] + h[5]) % 3,)

    def inv(g):
        p = [0] * 5
        for i in range(5):
            p[g[i]] = i
        return tuple(p) + ((-g[5]) % 3,)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 4, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Profiles specify central coordinates within each trinomial.
# Every component has a mixed cell; all-one-coset components would split.
_A_CENTRAL_PLANS = (
    ((0,0,1), (0,0,1)),
    ((0,0,1), (0,1,2)),
    ((0,1,2), (0,1,2)),
    ((0,0,0), (0,0,1)),
    ((0,0,0), (0,1,2)),
)
_B_CENTRAL_PLANS = (
    ((0,0,1), (0,0,1), (0,0,1)),
    ((0,0,0), (0,0,1), (0,1,2)),
    ((0,0,1), (0,0,1), (0,1,2)),
    ((0,0,1), (0,1,2), (0,1,2)),
)


def _set_bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _component_stats(rows, ncols):
    """Four cycles and check-projection triangle excess, not distance."""
    columns = [0] * ncols
    pairs = {}
    four = 0
    for r, row in enumerate(rows):
        ns = list(_set_bits(row))
        for u in ns:
            columns[u] |= 1 << r
        for a, u in enumerate(ns):
            for v in ns[a + 1:]:
                key = u * ncols + v
                previous = pairs.get(key, 0)
                four += previous
                pairs[key] = previous + 1

    # Reject isolated variables and identical component columns:
    # the latter supply classical weight-two kernel vectors.
    if not all(columns) or len(set(columns)) != ncols:
        return None

    adjacency = [0] * len(rows)
    forced_triangles = 0
    for signature in columns:
        degree = signature.bit_count()
        forced_triangles += degree * (degree - 1) * (degree - 2) // 6
        for r in _set_bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    # Require a connected component Tanner graph.
    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _set_bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _set_bits(neighbors >> (u + 1)):
            v += u + 1
            triangles += (neighbors & adjacency[v] &
                          (-1 << (v + 1))).bit_count()

    # With no four cycles, this is the number of genuine six cycles.
    # Otherwise it remains only a secondary projection statistic.
    return four, max(0, triangles - forced_triangles)


def _verified_components(G, hx, hz):
    """Extract both classical lifts from the scaffold's verified matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            lifted_row = hx[i * cb * q + r]  # beta = 0
            ar.append(sum(((lifted_row >> (j * cb * q)) & mask)
                          << (j * q) for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + r] & bmask             # A-column j = 0
          for a in range(rb) for r in range(q)]
    return ar, br


def NEW_CTX(rng, G, idx):
    """Twenty central-profile cards with independent component searches."""
    from itertools import combinations

    q = G.q
    assert q == 180
    card, cohort = idx % 20, idx // 20
    plans = (_A_CENTRAL_PLANS[card // 4],
             _B_CENTRAL_PLANS[card % 4])
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    diagnostic_seed = rng.randrange(1 << 62)

    # Cache only the group law. Every matrix still comes from the
    # scaffold helper, using exactly the original element indexing.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    assert all(G.idx(g) == i for i, g in enumerate(G.elements))
    assert all(g[5] == i // 60 for i, g in enumerate(G.elements))

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    orders = []
    for g in range(q):
        value, order = ident, 0
        while True:
            value = table[value][g]
            order += 1
            if value == ident:
                break
            if order > q:
                raise ValueError("invalid direct-product group law")
        orders.append(order)
    relative = [
        [orders[table[inverses[g]][h]] for h in range(q)]
        for g in range(q)
    ]
    penalties = (
        {2:30, 3:0, 5:0, 6:0, 15:0},
        {2:30, 3:4, 5:1, 6:3, 15:0},
        {2:30, 3:0, 5:4, 6:0, 15:4},
        {2:30, 3:3, 5:0, 6:3, 15:1},
    )

    def fresh_cell(profile, side):
        """Distinct projected permutations and noncommuting relatives."""
        policy = (card // 4 + side + cohort) % len(penalties)
        best, best_cost = None, None
        for attempt in range(128):
            permutations = rng.sample(range(60), 3)
            zs = list(profile)
            rng.shuffle(zs)
            shift = rng.randrange(3)
            cell = [p + 60*((z + shift) % 3)
                    for p, z in zip(permutations, zs)]
            u = table[inverses[cell[0]]][cell[1]]
            v = table[inverses[cell[0]]][cell[2]]
            if table[u][v] == table[v][u]:
                continue
            cost = sum(penalties[policy][relative[g][h]]
                       for i, g in enumerate(cell) for h in cell[:i])
            if best_cost is None or cost < best_cost:
                best, best_cost = cell, cost
            if attempt >= 11 and best is not None:
                break
        if best is None:
            raise ValueError("noncommuting trinomial could not be filled")
        return best

    def component(cells, side):
        """Empty dummy sides isolate VERIFIED components, never submissions."""
        width = len(cells) * q
        if side == 0:
            rows, _, _ = lifted_matrices(probe, [cells], [[[]]])
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], [cells])
        assert len(rows) == q
        assert all(row >> width == 0 for row in rows)
        columns = [0] * width
        for r, row in enumerate(rows):
            for c in _set_bits(row):
                columns[c] |= 1 << r
        return rows, columns

    def small_counts(columns):
        """Exact kernel-3/4 counts after excluding zero/duplicate columns."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return None
        pairs = {}
        three = four = 0
        for i, x in enumerate(columns):
            for y in columns[:i]:
                value = x ^ y
                three += int(value in singles)
                previous = pairs.get(value, 0)
                four += previous
                pairs[value] = previous + 1
        assert three % 3 == 0 and four % 3 == 0
        return three // 3, four // 3

    def dependencies(columns, order):
        pivots, basis = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
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
        return basis

    def kernel_probe(columns, diagnostic, trials):
        """Actual nonzero kernel witnesses; high readings certify nothing."""
        width = len(columns)
        order = list(range(width))
        reading = width + 1
        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = dependencies(columns, order)
            if not basis:
                continue
            basis.sort(key=int.bit_count)
            reading = min(reading, basis[0].bit_count())
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            descent = list(basis)
            diagnostic.shuffle(descent)
            for initial in basis[:10]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in descent:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    reading = min(reading, weight)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    candidate = a ^ b ^ c
                    if candidate:
                        reading = min(reading, candidate.bit_count())
            if reading <= 4:
                break
        return reading

    def restricted_reading(columns, diagnostic):
        """Localized and central-orbit restrictions of verified syndromes."""
        width = len(columns)
        blocks = width // q
        reading = width + 1

        # Exact enumeration only for small single-block kernels.
        for j in range(blocks):
            restricted = columns[j*q:(j+1)*q]
            basis = dependencies(restricted, range(q))
            if not basis:
                continue
            if len(basis) <= 16:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    reading = min(reading, word.bit_count())
            else:
                d = kernel_probe(restricted, diagnostic, 8)
                if d <= q:
                    reading = min(reading, d)

        if blocks > 2:
            for a, b in combinations(range(blocks), 2):
                restricted = (columns[a*q:(a+1)*q] +
                              columns[b*q:(b+1)*q])
                d = kernel_probe(restricted, diagnostic, 8)
                if d <= len(restricted):
                    reading = min(reading, d)

        # Same permutation in all three central coordinates is one
        # disjoint size-three orbit. XOR actual columns to restrict.
        central = [
            columns[j*q+p] ^ columns[j*q+60+p] ^ columns[j*q+120+p]
            for j in range(blocks) for p in range(60)
        ]
        d = kernel_probe(central, diagnostic, 12)
        if d <= len(central):
            reading = min(reading, 3*d)
        return reading

    def merit(record):
        return (tuple(-x for x in record["small"]), record["d"],
                -record["cycles"][0], -record["cycles"][1],
                record["rank"])

    def component_pool(side):
        records = []
        signatures = set()
        # B has nominal rate 2/3 rather than A's 1/2, so give B
        # more independent geometry candidates and longer final probes.
        for trial in range(32 if side == 0 else 56):
            cells = [fresh_cell(profile, side) for profile in plans[side]]
            signature = tuple(tuple(sorted(cell)) for cell in cells)
            if signature in signatures:
                continue
            signatures.add(signature)
            rows, columns = component(cells, side)
            stats = _component_stats(rows, len(columns))
            if stats is None:
                continue
            small = small_counts(columns)
            if small is None:
                continue
            records.append(dict(
                cells=cells, rows=rows, columns=columns, small=small,
                cycles=stats, rank=gf2_rank(rows), trial=trial))
        records.sort(key=lambda r: (r["small"], r["cycles"], r["trial"]))
        shortlisted = records[:4 if side == 0 else 6]
        if not shortlisted:
            raise ValueError("empty connected component shortlist")

        for record in shortlisted:
            diagnostic = random.Random(diagnostic_seed + 100003*side)
            record["d"] = kernel_probe(
                record["columns"], diagnostic, 8 if side == 0 else 16)
        shortlisted.sort(key=merit, reverse=True)
        finalists = shortlisted[:2]
        for record in finalists:
            # Fresh common orders reduce selection on one lucky search.
            diagnostic = random.Random(
                diagnostic_seed + 700001 + 100003*side)
            record["d"] = min(
                record["d"],
                kernel_probe(record["columns"], diagnostic,
                             32 if side == 0 else 64),
                restricted_reading(record["columns"], diagnostic))
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa, bb = component_pool(0), component_pool(1)
    best = None
    for a in aa:
        for b in bb:
            xx, zz, n = lifted_matrices(
                probe, [a["cells"]], [b["cells"]])
            assert n == 7*q == 1260
            k = n - gf2_rank(xx) - gf2_rank(zz)
            if not (k >= 2*q and 275*n <= 1000*k < 325*n):
                continue
            key = (
                tuple(-(x+y) for x, y in zip(a["small"], b["small"])),
                min(a["d"], b["d"]), a["d"] + b["d"],
                -a["cycles"][0] - b["cycles"][0], k)
            if best is None or key > best[0]:
                best = (key, a, b, k, xx, zz)
    if best is None:
        raise ValueError("screened component pairs missed cell 0.300")

    key, a, b, k, xx, zz = best
    A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
    B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == xx and hz == zz
    ar, br = _verified_components(G, hx, hz)
    assert ar == a["rows"] and br == b["rows"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert n - gf2_rank(hx) - gf2_rank(hz) == k
    return {
        "cells": {"A": A, "B": B}, "design": idx, "card": card,
        "k": k, "screen": key,
        "component_readings": (a["d"], b["d"]),
        "small_kernel_counts": (a["small"], b["small"]),
        "component_cycles": (a["cycles"], b["cycles"]),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the selected distinct-element permutation-group trinomial."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A row-weight cap of 12 alone excludes abundant weight-6 logicals: the rolled C37 (2,6)x(2,6) parent had d=6 in all five measured slots, entirely in sector 1, despite k-kb ranging from 0 to 10.
# REFUTED: Cycle-only selection plus a high short-budget reading reliably excludes sub-60 logicals on this affine-field bed: parent slot 40 fell from d2s=70 to d30s=40, with sector-1 X witnesses spanning multiple Acol/Bcol blocks.
# REFUTED: A high affine-field short reading remains stable after translation-fixed kernel screening: current slot 2 dropped from 68 to 56, with a sector-1 X witness on Acol2 across all three Bcols.
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
