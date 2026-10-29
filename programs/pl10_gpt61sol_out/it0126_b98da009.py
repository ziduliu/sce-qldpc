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
CONCEPT = ("For cyclic C29, order 29, shapes (2,4)x(2,4), balanced degree-three components with new relative-support architectures and reduced short-word concentration should improve mean worse-basis circuit fitness above 7.1770.")
GROUP = ("Frozen under L1: retain the C29 bed currently leading cell 0.200 at fitness 7.1770. Order 29 is the largest lift admitted by these shapes; ord_29(2)=28 supports simple rank screening but is not a circuit-reliability guarantee.")
SHAPES = ("Frozen (2,4)x(2,4): n=580, guaranteed k>=116 and rate 0.200. Both components have 58 checks and 116 variables; final verified ranks require 116<=k<=130. Parent maps implicated multiple Acol/Bcol and Acol/Brow blocks, but only 1-5 failures per measured basis support those maps, so there is insufficient evidence for a unique hotspot repair.")
FILLING = ("Twenty designs cross four balanced support pairs with five relative-support cards. Component rows have six terms and columns three, so every CSS check has weight nine and six-round extraction uses 25056 CNOTs. Binomial-only cards prescribe cross-row direction repetitions; other cards introduce zero/trinomial cells. Canonical relative-support books exclude obvious orbit copies, not arbitrary Tanner isomorphisms. Verified component screens combine exact kernel exclusion through weight six, a sampled protection floor of twelve, short-kernel translation-orbit concentration and small triple boundaries. Parent slots1 and7 both had d2s=12 but scores 7.398 and 7.077; weight-ten siblings required depth13 versus depth12 for weight-nine slots. Reduced extraction load and collective ambiguity are hypotheses about pL, not substitutes for its measurement.")
FAMILY_NAME = "r020_C29_24x24_balanced_orbitexposure_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This cyclic family requires q=29")
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

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Construct and screen a slot-specific design; no shared mutable cache."""
    from itertools import combinations

    # First-row counts; the second row completes each column to three.
    # All profiles have row totals 6/6 and column totals 3/3/3/3.
    profiles = ((1, 1, 2, 2), (0, 1, 2, 3), (0, 2, 2, 2))
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    architecture = (idx // 5) % len(pairs)
    pa, pb = pairs[architecture]

    def support(p):
        first = profiles[p]
        return [list(first), [3 - value for value in first]]

    sa, sb = support(pa), support(pb)
    assert _grid_weights(sa, sb) <= min(9, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) == 6 for counts in (sa, sb) for row in counts)
    q = G.q

    def normalized(cell, multiplier):
        if not cell:
            return ()
        values = [(multiplier * value) % q for value in cell]
        return min(tuple(sorted((value - origin) % q for value in values))
                   for origin in values)

    def geometry_key(grids):
        # Quotient independent cell origins, cyclic multipliers,
        # component row/column permutations and side exchange.
        # This is not a complete Tanner-isomorphism classifier.
        images = []
        for u in range(1, q):
            sides = []
            for counts, grid in zip((sa, sb), grids):
                columns = [
                    (counts[0][j], normalized(grid[0][j], u),
                     counts[1][j], normalized(grid[1][j], u))
                    for j in range(4)
                ]
                normal = tuple(sorted(columns))
                exchanged = tuple(sorted(
                    (c, d, a, b) for a, b, c, d in columns))
                sides.append(min(normal, exchanged))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Five genuinely different repetition briefs for the binomial pair:
    # one/two cross-row repetitions in A, and zero/one/two in B.
    # Repetitions never occur twice within the same component row.
    repetition_cards = ((1, 0), (2, 0), (1, 1), (2, 1), (2, 2))
    # Four Sidon trinomial shapes, with the fifth card distinguishing
    # joint binomial/trinomial geometry rather than claiming a fifth
    # affine-equivalence class of single trinomials.
    markers = (3, 4, 5, 9, 3)
    geometry_rng = random.Random(290119 + 1031 * architecture)
    wanted = 5 * (idx // 20) + idx % 5
    book, seen_geometries = [], set()

    for attempt in range(10000):
        card = len(book) % 5
        grids = []
        legal = True
        for side, counts in enumerate((sa, sb)):
            sites = [(i, j) for i in range(2) for j in range(4)
                     if counts[i][j] == 2]
            values = geometry_rng.sample(range(1, (q + 1) // 2),
                                         len(sites))
            if architecture == 0:
                repetitions = repetition_cards[card][side]
                if repetitions:
                    values[2] = values[0]
                if repetitions == 2:
                    values[3] = values[1]
            directions = dict(zip(sites, values))
            grid = []
            for i in range(2):
                row = []
                row_differences = set()
                for j in range(4):
                    weight = counts[i][j]
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [0]
                    elif weight == 2:
                        cell = [0, directions[i, j]]
                    else:
                        step = geometry_rng.randrange(1, q)
                        marker = markers[(card + side) % len(markers)]
                        cell = [0, step, step * marker % q]
                    # Repeated ordered differences within one row force
                    # component four-cycles regardless of cell offsets.
                    for x in cell:
                        for y in cell:
                            if x == y:
                                continue
                            difference = (x - y) % q
                            if difference in row_differences:
                                legal = False
                            row_differences.add(difference)
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        if not legal:
            continue
        key = geometry_key(grids)
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
        book.append(grids)
        if len(book) > wanted:
            break

    if len(book) <= wanted:
        raise ValueError("insufficient new relative-support architectures")
    geometry = book[wanted]

    def labels(s, side):
        grid = []
        for i in range(2):
            row = []
            for j in range(4):
                origin = rng.randrange(q)
                cell = [((origin + value) % q,)
                        for value in geometry[side][i][j]]
                assert len(cell) == s[i][j]
                row.append(cell)
            grid.append(row)
        return grid

    q = G.q
    mask = (1 << q) - 1
    component_mask = (1 << (4 * q)) - 1

    def components(hx, hz):
        # Extract from the verified construction, not a separate lift.
        # Fix B-column 0 in HX to obtain the left-lifted A component.
        ar = []
        for i in range(2):
            for r in range(q):
                row = hx[(i * 4) * q + r]
                compact = 0
                for j in range(4):
                    compact |= ((row >> ((j * 4) * q)) & mask) << (j * q)
                ar.append(compact)
        # Fix A-column 0 in HZ to obtain the right-lifted B component.
        br = [row & component_mask for row in hz[:2 * q]]
        return ar, br

    def cycle_cost(rows):
        """Exact component 4/6-cycle counts and a rank-deficit penalty."""
        m = len(rows)
        columns = [[] for _ in range(4 * q)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1].append(r)
                row ^= bit

        adj = [dict() for _ in range(m)]
        for ns in columns:
            for a, b in combinations(ns, 2):
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1

        seen, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        if len(seen) != m:
            return None

        four = sum(v * (v - 1) // 2
                   for a in range(m) for b, v in adj[a].items() if a < b)
        six = 0
        for a in range(m):
            for b, ab in adj[a].items():
                if b <= a:
                    continue
                for c, bc in adj[b].items():
                    if c > b and c in adj[a]:
                        six += ab * bc * adj[a][c]
        # Remove triangles whose check-pair edges reuse a variable.
        for ns in columns:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        deficit = m - gf2_rank(rows)
        return (four, six + q * deficit, deficit)

    pools = [[], []]
    for trial in range(192):
        a, b = labels(sa, 0), labels(sb, 1)
        hx, hz, _ = lifted_matrices(G, a, b)
        ar, br = components(hx, hz)
        for side, grid, rows in ((0, a, ar), (1, b, br)):
            cost = cycle_cost(rows)
            if cost is not None:
                pools[side].append((cost, trial, grid, rows))
                pools[side].sort(key=lambda item: (item[0], item[1]))
                del pools[side][6:]

    if not all(pools):
        raise ValueError("no connected component candidate")

    def kernel_upper_bound(rows):
        """Sample kernel words using randomized information sets.

        This is only an upper-bound heuristic, not a distance certificate
        and not an estimate of the scored circuit-level failure rate.
        """
        n = 4 * q
        best = n
        small = set()
        for _ in range(64):
            mat = list(rows)
            order = list(range(n))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                found = next((r for r in range(rank, len(mat))
                              if mat[r] & bit), None)
                if found is None:
                    continue
                mat[rank], mat[found] = mat[found], mat[rank]
                pivot_row = mat[rank]
                for r in range(len(mat)):
                    if r != rank and mat[r] & bit:
                        mat[r] ^= pivot_row
                pivots.append(c)
                rank += 1
                if rank == len(mat):
                    break

            occupied = set(pivots)
            basis = []
            for c in order:
                if c in occupied:
                    continue
                bit = 1 << c
                word = bit
                for r, p in enumerate(pivots):
                    if mat[r] & bit:
                        word ^= 1 << p
                basis.append(word)
                weight = word.bit_count()
                best = min(best, weight)
                if weight <= 12:
                    small.add(word)

            for i, x in enumerate(basis):
                for j in range(i):
                    word = x ^ basis[j]
                    weight = word.bit_count()
                    best = min(best, weight)
                    if weight <= 12:
                        small.add(word)
            if best <= 6:
                break

        # Count translation orbits rather than rewarding accidental
        # resampling of the same word. This permutes witnesses only;
        # it does not construct or substitute a different lifted matrix.
        orbit_words = set()
        for word in small:
            blocks = [(word >> (j * q)) & mask for j in range(4)]
            representative = None
            for shift in range(q):
                image = 0
                for j, block in enumerate(blocks):
                    rotated = ((block << shift) |
                               (block >> (q - shift))) & mask
                    image |= rotated << (j * q)
                if representative is None or image < representative:
                    representative = image
            orbit_words.add(representative)
        loads = [0] * 4
        for word in orbit_words:
            for j in range(4):
                loads[j] += ((word >> (j * q)) & mask).bit_count()
        return best, len(orbit_words), max(loads)

    def small_kernel_word(rows):
        """Exact kernel exclusion through six, plus triple-boundary exposure.

        The boundary score is a trapping-set heuristic, not measured pL.
        Counting its load by component column tests concentration without
        claiming that one noisy failure-map block identifies a defect.
        """
        columns = [0] * (4 * q)
        for r, row in enumerate(rows):
            check_bit = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= check_bit
                row ^= bit
        seen = {0}
        for value in columns:
            if value in seen:
                return True, 10**9, 10**9
            seen.add(value)
        for i, x in enumerate(columns):
            for j in range(i):
                value = x ^ columns[j]
                if value in seen:
                    return True, 10**9, 10**9
                seen.add(value)

        exposure = 0
        loads = [0] * 4
        for i, x in enumerate(columns):
            for j in range(i):
                pair = x ^ columns[j]
                for h in range(j):
                    value = pair ^ columns[h]
                    if value in seen:
                        return True, 10**9, 10**9
                    seen.add(value)
                    weight = value.bit_count()
                    if weight <= 3:
                        penalty = (5 - weight) ** 2
                        exposure += penalty
                        loads[i // q] += penalty
                        loads[j // q] += penalty
                        loads[h // q] += penalty
        return False, exposure, max(loads)

    unit = [[[G.e]]]

    def projected(side, grid):
        # The first sector of this smaller VERIFIED identity-partner
        # construction is exactly the requested component.
        if side == 0:
            rows = lifted_matrices(G, grid, unit)[0]
        else:
            rows = lifted_matrices(G, unit, grid)[1]
        return [row & component_mask for row in rows]

    finalists = []
    for side, pool in enumerate(pools):
        entries = []
        for cost, trial, grid, rows in pool:
            # Shift an entire cell: preserve its indexed internal geometry
            # while changing inter-cell voltages and component cycles.
            for step in range(64):
                proposal = [[list(cell) for cell in row] for row in grid]
                i, j = rng.randrange(2), rng.randrange(4)
                shift = rng.randrange(1, q)
                proposal[i][j] = [((g[0] + shift) % q,)
                                  for g in proposal[i][j]]
                proposed_rows = projected(side, proposal)
                proposed_cost = cycle_cost(proposed_rows)
                if proposed_cost is not None and proposed_cost <= cost:
                    grid, rows, cost = proposal, proposed_rows, proposed_cost
            bad, boundary, boundary_peak = small_kernel_word(rows)
            d, low_orbits, word_peak = kernel_upper_bound(rows)
            entries.append((cost, d, int(bad), low_orbits, word_peak,
                            boundary, boundary_peak, trial, grid))
        finalists.append(entries)

    # Establish a coarse sampled protection floor, then minimize the
    # worse component's short-word concentration and boundary burden.
    # No extra reward is given for upper bounds above twelve.
    # Exact small-word exclusion remains a preference, not a guarantee
    # that a code passes the circuit experiment.
    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, ad, abad, alow, awpeak, abound, abpeak, at, ag = a
            bc, bd, bbad, blow, bwpeak, bbound, bbpeak, bt, bg = b
            da, db = min(ad, 12), min(bd, 12)
            key = (abad + bbad, int(min(da, db) < 12),
                   ac[0] + bc[0],
                   max(awpeak, bwpeak), max(alow, blow), alow + blow,
                   max(abpeak, bbpeak), abound + bbound,
                   ac[1] + bc[1], abs(da - db),
                   ac[2] + bc[2], at, bt)
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    for key, a, b in options:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if 116 <= k <= 130 and css_ok(hx, hz):
            return {"idx": idx, "cells": {"A": a, "B": b}, "k": k,
                    "architecture": architecture,
                    "geometry": wanted,
                    "small_kernel_safe": key[0] == 0}
    raise ValueError("no screened design remains in the 0.200 rate cell")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without drawing new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s>=8 guarantees basis-balanced memory performance; seed slot19 had X failures 100/78000 versus Z 29/200000, with scores 5.634 versus 6.509.
# REFUTED: Equal d2s and larger k guarantee no worse circuit score: current-parent slot16 had d2s=12,k=116,s=6.389, versus slot21 with d2s=12,k=112,s=6.931.
# REFUTED: Internal binomial-step orbit signatures alone ensure distinct submitted Tanner codes: the C29 parent retained only 25 distinct codes from 60 slots, including slot2 marked iso_dup_of_1.
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
