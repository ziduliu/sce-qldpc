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
CONCEPT = ("For metacyclic Z61:Z3(s=13), order 183, A=(1,2), B=(2,3), independently optimized degree-four component kernels should raise five-code fitness above the recorded 87.5 in cell 0.150. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Keep the split-metacyclic construction rule and GTYPE; resize only its parameters to m=61,r=3,s=13. Since 13^3=1 modulo 61 and 13!=1, the action has order three. The current order-111 bed reaches 87.5, while the order-291 C3-action bed reports 917.7 on another shape. Those provisional readings motivate a larger normal subgroup without assuming distance transfer.")
SHAPES = ("A=(1,2), B=(2,3), q=183 give n=8*183=1464 and k>=183, guaranteed rate 0.125, the included lower boundary of cell 0.150. The size window permits q<=187. Components have variables/checks 366/183 and 549/366, nominal rates 1/2 and 1/3. Relative to the parent, A grows from 222 variables and B gains checks from 333 to 366. Parent measured minima were abundant sector-1 Z witnesses at weights 22..24, using both Acols at individual Bcols. Exact full ranks retain [0.125,0.175).")
FILLING = ("Twenty cards cross four A two-cell coset-occupancy profiles with five inequivalent B support topologies. A cells have four terms; every B column has four terms and B row totals are at most eight. Thus HX=8+4=12 and HZ<=8+4=12. Independently shortlist components using verified lifts, connectedness, ranks and cycle geometry; exclude component kernels through four, then search genuine kernel words with cancellation descent and independently retest finalists. Final full ranks, CSS and row weights gate pairing. Component readings are selection heuristics, not certified quantum distances.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m61r3s13_12x23_degree4_independent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 183

def custom_ops(q):
    # Same split-metacyclic rule, resized for the L2 shape move.
    # 13^2=47 and 13^3=1 modulo 61; the action has order three.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 183")
    m, r, s = 61, 3, 13
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


# Four distinct A occupancy allocations crossed with five weighted B graphs.
# B column totals are exactly four; row totals never exceed eight.
_A_OCCUPANCIES = ((2, 1, 1), (3, 1, 0), (2, 2, 0))
_A_PROFILES = ((0, 0), (0, 1), (1, 1), (0, 2))
_B_PROFILES = (
    ((2, 2, 2), (2, 2, 2)),
    ((3, 2, 2), (1, 2, 2)),
    ((3, 1, 2), (1, 3, 2)),
    ((3, 3, 2), (1, 1, 2)),
    ((4, 1, 1), (0, 3, 3)),
)


def _bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _occupancy_cell(rng, occupancy):
    """Fresh support, not an automorphism image of a stored support."""
    fibers = list(range(3))
    rng.shuffle(fibers)
    cell = []
    for b, count in zip(fibers, occupancy):
        if not count:
            continue
        for _ in range(256):
            aa = rng.sample(range(61), count)
            differences = [(a-c) % 61 for a in aa
                           for c in aa if a != c]
            if len(set(differences)) == len(differences):
                break
        else:
            raise ValueError("within-fiber Sidon construction failed")
        cell.extend((a, b) for a in aa)
    assert len(cell) == len(set(cell))
    return cell


def _component_geometry(rows, width):
    """Verified columns, exact C4 count and projection-triangle excess."""
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _bits(row):
            columns[c] |= 1 << r
    if not all(columns) or len(set(columns)) != width:
        return None
    if any(column.bit_count() != 4 for column in columns):
        raise ValueError("component column degree is not four")

    adjacency = [0] * len(rows)
    pair_uses = {}
    four = 0
    for column in columns:
        neighbors = list(_bits(column))
        for i, a in enumerate(neighbors):
            adjacency[a] |= column ^ (1 << a)
            for b in neighbors[:i]:
                key = b * len(rows) + a
                previous = pair_uses.get(key, 0)
                four += previous
                pair_uses[key] = previous + 1

    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for a, neighbors in enumerate(adjacency):
        for b in _bits(neighbors & (-1 << (a + 1))):
            triangles += (neighbors & adjacency[b]
                          & (-1 << (b + 1))).bit_count()
    # Each degree-four variable supplies four forced triangles.
    # With C4 present the excess is only a projection statistic.
    excess = max(0, triangles - 4 * width)
    return (four, excess), columns


def _kernel4_free(columns):
    """Exact classical exclusion through four, not a quantum certificate."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    pairs = set()
    for i, a in enumerate(columns):
        for b in columns[:i]:
            syndrome = a ^ b
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _kernel_probe(columns, diagnostic, trials):
    """Search genuine nonzero kernel words in original coordinates.

    Random fundamental circuits, pair cancellation and descent yield
    sampled UPPER readings. Retained words remain in the actual kernel.
    """
    width = len(columns)
    reading = width + 1
    retained = []
    order = list(range(width))
    for _ in range(trials):
        diagnostic.shuffle(order)
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
                reading = min(reading, word.bit_count())
        if not basis:
            raise ValueError("component unexpectedly has no kernel")

        light = sorted(set(retained + basis), key=int.bit_count)[:18]
        additions = []
        for word in light:
            for other in basis:
                candidate = word ^ other
                if not candidate:
                    continue
                weight = candidate.bit_count()
                reading = min(reading, weight)
                if weight <= reading + 6:
                    additions.append(candidate)

        moves = list(retained + basis)
        diagnostic.shuffle(moves)
        for word in light[:6]:
            weight = word.bit_count()
            for sweep in range(2):
                changed = False
                for other in moves:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() < weight:
                        word, weight = candidate, candidate.bit_count()
                        changed = True
                reading = min(reading, weight)
                if not changed:
                    break
            additions.append(word)

        if len(basis) >= 3:
            for _ in range(96):
                a, b, c = diagnostic.sample(basis, 3)
                word = a ^ b ^ c
                if word:
                    reading = min(reading, word.bit_count())
                    additions.append(word)
        retained = sorted(set(retained + light + additions),
                          key=int.bit_count)[:24]
        if reading <= 8:
            break
    return reading


def NEW_CTX(rng, G, idx):
    """Twenty degree-four cards; independently optimize both components."""
    q = G.q
    assert q == 183
    card, cohort = idx % 20, idx // 20
    aprofile, bprofile = card % 4, card // 4
    sa = [[4, 4]]
    sb = [list(row) for row in _B_PROFILES[bprofile]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(sb[i][j] for i in range(2)) == 4
               for j in range(3))
    seed = rng.randrange(1 << 62)

    # Cache native GROUP OPERATIONS only. Every lift still comes from
    # lifted_matrices, with the native element order preserved exactly.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = G.idx(G.e)

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()

    def fresh(side):
        if side == "A":
            return [[
                _occupancy_cell(rng, _A_OCCUPANCIES[kind])
                for kind in _A_PROFILES[aprofile]
            ]]
        result = []
        for i, counts in enumerate(sb):
            row = []
            for j, count in enumerate(counts):
                if count == 0:
                    occupancy = (0, 0, 0)
                elif count == 1:
                    occupancy = (1, 0, 0)
                elif count == 2:
                    # Extra slots change actual fiber multiplicities,
                    # not simply quotient names or common translations.
                    homogeneous = (cohort % 3 == 1 and
                                   (i + j + aprofile) % 3 == 0)
                    occupancy = ((2, 0, 0) if homogeneous
                                 else (1, 1, 0))
                elif count == 3:
                    occupancy = ((1, 1, 1) if cohort % 3 == 2
                                 else (2, 1, 0))
                else:
                    occupancy = (2, 1, 1)
                row.append(_occupancy_cell(rng, occupancy))
            result.append(row)
        return result

    def component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        # Empty dummy sides isolate exact evaluator components.
        # These probes are not submitted as codes.
        if side == "A":
            rows, _, _ = lifted_matrices(probe, indexed, [[[]]])
        else:
            _, rows, _ = lifted_matrices(probe, [[[]]], indexed)
        width = len(grid[0]) * q
        assert len(rows) == len(grid) * q
        assert all(row >> width == 0 for row in rows)
        return rows, width

    def cycle_key(record):
        four, excess = record["cycles"]
        if cohort % 3 == 1:
            return (8 * four + excess, four, excess)
        return (four, excess)

    def select_component(side):
        records = []
        signatures = set()
        for attempt in range(2):
            for trial in range(36):
                grid = fresh(side)
                signature = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                rows, width = component(grid, side)
                geometry = _component_geometry(rows, width)
                if geometry is None:
                    continue
                cycles, columns = geometry
                records.append(dict(
                    grid=grid, rows=rows, width=width,
                    columns=columns, cycles=cycles,
                    trial=36*attempt + trial))
            records.sort(key=lambda r: (cycle_key(r), r["trial"]))

            screened = []
            for record in records:
                rows = record["rows"]
                rank = gf2_rank(rows)
                permitted_loss = 3 if side == "A" else 6
                if rank < len(rows) - permitted_loss:
                    continue
                if side == "A":
                    mask = (1 << q) - 1
                    cell_ranks = (
                        gf2_rank([row & mask for row in rows]),
                        gf2_rank([row >> q for row in rows]))
                    if min(cell_ranks) < q - 3:
                        continue
                    record["cell_ranks"] = cell_ranks
                if not _kernel4_free(record["columns"]):
                    continue
                diagnostic = random.Random(
                    seed + (17 if side == "A" else 31))
                record["d"] = _kernel_probe(
                    record["columns"], diagnostic, 6)
                record["rank"] = rank
                screened.append(record)
                if len(screened) == 5:
                    break
            if not screened:
                continue

            def merit(record):
                return (record["d"],
                        tuple(-x for x in cycle_key(record)),
                        record["rank"])

            screened.sort(key=merit, reverse=True)
            finalists = screened[:2]
            for record in finalists:
                # Common independent diagnostic seeds reduce selection
                # on one fortunate short probe. Readings only decrease.
                offset = 100003 if side == "A" else 200003
                record["d"] = min(
                    record["d"],
                    _kernel_probe(
                        record["columns"],
                        random.Random(seed + offset), 16),
                    _kernel_probe(
                        record["columns"],
                        random.Random(seed + offset + 700001), 24))
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("no connected degree-four component passed screening")

    aa = select_component("A")
    bb = select_component("B")
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"] + ab[1]["d"],
        tuple(-x-y for x, y in
              zip(cycle_key(ab[0]), cycle_key(ab[1])))),
        reverse=True)

    for a, b in proposals:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1464 and k >= 183
                and 125*n <= 1000*k < 175*n):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert _grid_weights(sa, sb) <= W_MAX
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (aprofile, bprofile), "cohort": cohort,
            "component_readings": (a["d"], b["d"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_cycles": (a["cycles"], b["cycles"]),
        }
    raise ValueError("independent component pairs missed cell 0.150")


def CELL(rng, G, side, i, j, ctx):
    """Return every new-shape cell without post-screen randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent measurements: rolled positive-cell supports on Z19:Z3 (4,2)x(5,3) reliably avoid small sector-2 logicals; measured slots 2 and 9 both contain X logicals of weight 6.
# REFUTED: Small-component-kernel and cycle-first label selection alone removes abundant sector-1 Z logicals of weight <=24 on Z37:Z3 (1,2)x(3,5); the current five measured codes returned 22,22,22,23,24 with those witnesses across both Acols.
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
