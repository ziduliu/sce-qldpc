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
CONCEPT = ("For cyclic C19, order 19, A=(2,5), B=(2,5), equal multi-row components and low-load, short-dependency-screened voltage geometries should reduce the affine parent's dominant X-memory failures in cell 0.300.")
GROUP = ("Jump from nonabelian AGL(1,7), order 42, to abelian C19. This (group,shape) bed is absent from BEDS. Nineteen is the largest prime lift allowed by size factor 29; ord_19(2)=18 gives one nontrivial binary cyclotomic factor. The C29 multi-row bed scored 7.177 and the C20 symmetric rate-0.300 bed reached 5.8, motivating but not establishing a prime-order circuit advantage.")
SHAPES = ("Equal (2,5)x(2,5) components have 38 checks and 95 variables each: n=551, kb=171, guaranteed rate 9/29=0.310345. Verified ranks keep every submitted code in [0.275,0.325). Parent X failures affected all logical qubits, with diffuse Acol/Bcol enrichment and widespread weight-eight Z witnesses; no single Acol/Brow hotspot was established. Replace the unequal single-row/multi-row components rather than privileging one block.")
FILLING = ("Twenty prescribed support pairs have column sums 3-4 and row sums <=8, giving both CSS check-weight sums <=12 and 23940..25536 six-round CNOTs versus the parent's 25704. Internal geometries exclude repeated within-row directions and obvious multiplier/permutation orbit copies; extra slots extend their geometry books. Independently optimize cell offsets on verified component lifts, then compare exact short-kernel exposure, small syndrome boundaries and capped sampled kernel witnesses. Full-code screening verifies actual rate and CSS and prefers absence of logicals <=4. Parent measured X failure caps occurred after 2500..7500 shots, versus only 3..70 Z failures in 200000 shots; balanced circuit protection, not distance alone, is the hypothesis.")
FAMILY_NAME = "r030_C19_25x25_balanced_geometry_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order 19")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def idx(g):
        return red(g)[0]

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _design_four_cycles(rows):
    """Count four-cycles within one basis, not across X and Z."""
    seen = {}
    cycles = 0
    for row in rows:
        cols = []
        while row:
            bit = row & -row
            cols.append(bit.bit_length() - 1)
            row ^= bit
        for a in range(len(cols)):
            for b in range(a):
                pair = (cols[b], cols[a])
                previous = seen.get(pair, 0)
                cycles += previous
                seen[pair] = previous + 1
    return cycles


def _design_small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1..4.

    Equal syndromes of supports of size <=2 exhaust this range.
    Reduce their difference against the opposite stabilizer row space.
    Inputs are exclusively the scaffold's verified lifted matrices.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def trivial(vector):
        while vector:
            p = vector.bit_length() - 1
            if p not in pivots:
                return False
            vector ^= pivots[p]
        return True

    syndromes = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] ^= syndrome_bit
            row ^= bit

    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(syndromes):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif not trivial(previous ^ bits[i]):
            return True

    for i in range(n):
        si, vi = syndromes[i], bits[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            vector = vi ^ bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = vector
            elif not trivial(previous ^ vector):
                return True
    return False


# Columns are term-count pairs. All columns have degree 3 or 4;
# every row has degree at most 8. Augmentation row rank is two.
_PRIME_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,2)),
    ((2,1),(2,1),(1,2),(1,2),(2,2)),
    ((2,1),(1,2),(1,2),(1,2),(3,1)),
    ((3,0),(2,1),(1,2),(1,2),(1,2)),
    ((3,0),(2,1),(2,1),(0,3),(0,3)),
    ((3,0),(0,3),(2,1),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(2,2)),
    ((3,0),(0,3),(0,3),(3,1),(2,1)),
)
_PRIME_PAIRS = (
    (0,0),(0,1),(0,2),(1,1),(1,2),(2,2),
    (0,3),(0,4),(1,3),(1,4),(2,3),(2,4),
    (3,3),(3,4),(4,4),(0,5),(1,5),(2,5),
    (0,6),(0,7),
)


def _prime_support(number):
    columns = _PRIME_PROFILES[number]
    return [[column[i] for column in columns] for i in range(2)]


def _prime_geometry(G, idx, sa, sb):
    """A deterministic internal-geometry book, independent of offset RNG.

    The signature quotients individual cell translations, cyclic
    multipliers, row/column permutations and side exchange. Distinct
    signatures exclude those obvious orbit copies; arbitrary Tanner
    isomorphism detection remains the evaluator's authority.
    """
    q = G.q

    def normalized(values):
        if not values:
            return ()
        return min(tuple(sorted((v - origin) % q for v in values))
                   for origin in values)

    def signature(grids):
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                cells = [[normalized([(u * a) % q for a in cell])
                          for cell in row] for row in grid]
                columns = tuple(sorted(
                    (cells[0][j], cells[1][j]) for j in range(5)))
                swapped = tuple(sorted((b, a) for a, b in columns))
                sides.append(min(columns, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    def draw(srng, support):
        grid = []
        for row in support:
            used = set()
            out = []
            for weight in row:
                if weight == 0:
                    out.append([])
                    continue
                for _ in range(1024):
                    values = [0] + srng.sample(range(1, q), weight - 1)
                    directions = []
                    for a in range(weight):
                        for b in range(a):
                            d = (values[a] - values[b]) % q
                            directions.append(min(d, q - d))
                    ds = set(directions)
                    if len(ds) != len(directions) or ds & used:
                        continue
                    used.update(ds)
                    out.append(values)
                    break
                else:
                    return None
            grid.append(out)
        return grid

    srng = random.Random(190019 + 1009 * (idx % 20))
    wanted = idx // 20
    seen = set()
    for _ in range(4096):
        a, b = draw(srng, sa), draw(srng, sb)
        if a is None or b is None:
            continue
        key = signature((a, b))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) == wanted + 1:
            return a, b
    raise ValueError("insufficient inequivalent internal geometries")


def _prime_grid(rng, G, geometry):
    """Independent cell offsets preserve this slot's internal geometry."""
    return [[[((a + shift) % G.q,) for a in cell]
             for cell, shift in
             ((cell, rng.randrange(G.q)) for cell in row)]
            for row in geometry]


def _affine_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _affine_component_rows(G, side, grid):
    """Probe the scaffold's exact component action, never a private lift."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _affine_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _affine_component_screen(rows, width):
    """Exact exclusion test through kernel weight four, plus connectivity."""
    columns = _affine_columns(rows, width)
    bad = int(0 in columns or len(set(columns)) != width)
    singles = set(columns)
    pairs = set()
    if not bad:
        for i, a in enumerate(columns):
            for b in columns[:i]:
                syndrome = a ^ b
                if syndrome in singles or syndrome in pairs:
                    bad = 1
                    break
                pairs.add(syndrome)
            if bad:
                break

    parent = list(range(len(rows)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for column in columns:
        if not column:
            continue
        first = (column & -column).bit_length() - 1
        remaining = column & (column - 1)
        while remaining:
            bit = remaining & -remaining
            a, b = root(first), root(bit.bit_length() - 1)
            if a != b:
                parent[b] = a
            remaining ^= bit
    components = len({root(a) for a in range(len(rows))})
    return (bad, components - 1, len(rows) - gf2_rank(rows),
            _design_four_cycles(rows)), columns


def _prime_cycles(rows, width):
    """Exact Tanner four/six cycles from the verified component rows."""
    from itertools import combinations
    neighbors = [[] for _ in range(width)]
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            neighbors[bit.bit_length() - 1].append(r)
            row ^= bit

    adjacency = [dict() for _ in rows]
    for ns in neighbors:
        for a, b in combinations(ns, 2):
            adjacency[a][b] = adjacency[a].get(b, 0) + 1
            adjacency[b][a] = adjacency[b].get(a, 0) + 1
    four = sum(v * (v - 1) // 2
               for a, ns in enumerate(adjacency)
               for b, v in ns.items() if a < b)
    six = 0
    for a, ns in enumerate(adjacency):
        for b, ab in ns.items():
            if b <= a:
                continue
            for c, bc in adjacency[b].items():
                if c > b and c in ns:
                    six += ab * bc * ns[c]
    # Inclusion-exclusion removes reused-variable triangles, including
    # stars even when four-cycles are present.
    for ns in neighbors:
        for a, b, c in combinations(ns, 3):
            six -= (adjacency[a][b] + adjacency[a][c]
                    + adjacency[b][c] - 2)
    assert six >= 0
    return four, six


def _prime_dependencies(columns):
    """Exact short-word existence signals and triple-boundary exposure.

    After excluding weights <=4, triple/pair collisions witness weight
    five and different equal-syndrome triples witness weight six.
    Multiplicities and nonzero small boundaries are screening heuristics,
    not quantum distances or measured decoder failure rates.
    """
    pairs = {columns[i] ^ columns[j]
             for i in range(len(columns)) for j in range(i)}
    triples = {}
    five = six = boundary = 0
    for i, a in enumerate(columns):
        for j in range(i):
            ab = a ^ columns[j]
            for h in range(j):
                syndrome = ab ^ columns[h]
                five += int(syndrome in pairs)
                previous = triples.get(syndrome, 0)
                six += previous
                triples[syndrome] = previous + 1
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
    return five, six, boundary


def _prime_kernel_upper(rng, columns, trials=48):
    """Sample actual component kernel words; the minimum is an upper bound."""
    width = len(columns)
    order = list(range(width))
    best = width + 1
    witnesses = set()
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in pivots:
                    s, v = pivots[p]
                    syndrome ^= s
                    word ^= v
                else:
                    pivots[p] = (syndrome, word)
                    break
            if not syndrome:
                basis.append(word)
                w = word.bit_count()
                best = min(best, w)
                if w <= 10:
                    witnesses.add(word)
        for i, word in enumerate(basis):
            for other in basis[:i]:
                v = word ^ other
                w = v.bit_count()
                best = min(best, w)
                if w <= 10:
                    witnesses.add(v)
        if best <= 6:
            break
    return best, len(witnesses)


def _prime_component_pool(rng, G, side, geometry, pressure):
    """Optimize offsets without converting a slot into an orbit copy."""
    import math
    width = 5 * G.q
    positions = [(i, j) for i, row in enumerate(geometry)
                 for j, cell in enumerate(row) if cell]
    proposals = []

    def analyze(grid):
        rows = _affine_component_rows(G, side, grid)
        four, six = _prime_cycles(rows, width)
        energy = (80.0 * four + six / 4.0) / G.q
        return energy, rows, four, six

    for restart in range(4):
        grid = _prime_grid(rng, G, geometry)
        current = analyze(grid)
        local_best = _affine_copy(grid)
        local = current
        for step in range(180):
            i, j = rng.choice(positions)
            old = grid[i][j]
            shift = rng.randrange(1, G.q)
            grid[i][j] = [((g[0] + shift) % G.q,) for g in old]
            proposed = analyze(grid)
            delta = proposed[0] - current[0]
            temperature = 18.0 * (0.975 ** step) + 0.15
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                current = proposed
                if current[0] < local[0]:
                    local_best = _affine_copy(grid)
                    local = current
            else:
                grid[i][j] = old
            if step in (89, 179):
                proposals.append(_affine_copy(grid))
        proposals.append(local_best)

    screened = []
    seen = set()
    for serial, grid in enumerate(proposals):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        rows = _affine_component_rows(G, side, grid)
        quality, columns = _affine_component_screen(rows, width)
        four, six = _prime_cycles(rows, width)
        preliminary = quality[:3] + (four, six, serial)
        screened.append((preliminary, grid, rows, columns))
    screened.sort(key=lambda item: item[0])

    finalists = []
    for preliminary, grid, rows, columns in screened[:4]:
        bad, disconnected, deficit, four, cycles6, serial = preliminary
        if bad:
            five, six, boundary = 10**9, 10**9, 10**9
        else:
            five, six, boundary = _prime_dependencies(columns)
        d, exposure = _prime_kernel_upper(rng, columns)
        refinement = ((boundary, six, cycles6) if pressure == 1
                      else (six, boundary, cycles6))
        key = (bad, disconnected, int(five > 0 or six > 0),
               deficit, five, four, -min(d, 10), refinement,
               exposure, serial)
        stats = (bad, disconnected, deficit, five, six, boundary,
                 four, cycles6, d, exposure)
        finalists.append((key, grid, stats))
    finalists.sort(key=lambda item: item[0])
    return finalists[:3]


def NEW_CTX(rng, G, idx):
    """Twenty support briefs with independent, inequivalent geometries."""
    design = idx % len(_PRIME_PAIRS)
    pa, pb = _PRIME_PAIRS[design]
    sa, sb = _prime_support(pa), _prime_support(pb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(3 <= sum(row[j] for row in s) <= 4
               for s in (sa, sb) for j in range(5))
    assert all(sum(row) <= 8 for s in (sa, sb) for row in s)

    ga, gb = _prime_geometry(G, idx, sa, sb)
    pressure = (idx // 20) % 3
    apool = _prime_component_pool(rng, G, "A", ga, pressure)
    bpool = _prime_component_pool(rng, G, "B", gb, pressure)
    options = []
    for _, A, ast in apool:
        for _, B, bst in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 275 * n <= 1000 * k < 325 * n):
                continue
            da, db = min(ast[8], 10), min(bst[8], 10)
            smalla = int(ast[3] > 0 or ast[4] > 0)
            smallb = int(bst[3] > 0 or bst[4] > 0)
            key = (
                ast[0] + bst[0],
                ast[1] + bst[1],
                max(smalla, smallb), smalla + smallb,
                k - 171,
                max(ast[3], bst[3]), ast[3] + bst[3],
                max(ast[4], bst[4]), ast[4] + bst[4],
                max(ast[6], bst[6]), ast[6] + bst[6],
                -min(da, db), abs(da - db),
                max(ast[5], bst[5]), ast[5] + bst[5],
                ast[7] + bst[7], ast[9] + bst[9],
            )
            options.append((key, A, B, hx, hz, n, k))
    options.sort(key=lambda item: item[0])

    def candidate(item):
        key, A, B, hx, hz, n, k = item
        assert n == 551
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        expected = 6 * 7 * G.q * (
            sum(map(sum, sa)) + sum(map(sum, sb)))
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected
        return {"idx": idx, "design": design, "screen": key,
                "k": k, "cells": {"A": A, "B": B}}

    # Exact short-logical exclusion is preferred, not a reason to discard
    # a rank-legal architecture if every finalist misses the preference.
    for item in options:
        _, _, _, hx, hz, n, _ = item
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        return candidate(item)
    if options:
        return candidate(options[0])

    # Rescue changes offsets only: term counts and the indexed internal
    # geometry remain this slot's deliberate design.
    for attempt in range(64):
        A = _prime_grid(rng, G, ga)
        B = _prime_grid(rng, G, gb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k > 0 and 275 * n <= 1000 * k < 325 * n:
            return candidate(((10**12, attempt), A, B, hx, hz, n, k))
    raise ValueError("no C19 filling retained the target rate")


def CELL(rng, G, side, i, j, ctx):
    """Return distinct elements of the selected complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Lower ASC depth alone guarantees better circuit fitness: current slot 42 had depth 14 and s=6.189, while slot 5 had depth 15 and s=6.657.
# REFUTED: Noncommuting affine trinomials plus degree-three component screening balance memory protection: all five measured parent codes hit X failure caps after 2500..7500 shots, while Z had only 3..70 failures in 200000 shots; worse-basis s remained 4.279..4.794.
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
