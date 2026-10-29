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
CONCEPT = ("SL(2,5), order 120, A=(1,2), B=(3,4): screening single-cell and central-fixed kernels before independent component selection should raise five-code fitness above 64.3858 in cell 0.100; no quantum-distance lower bound is claimed.")
GROUP = ("Keep the frozen SL(2,5) group and order: this bed has reached fitness 64.8. Its order-two center supplies an invariant subspace whose actual kernel columns can be extracted from verified lifts. The parent's 32-to-22 collapse motivates screening this subspace and localized kernels explicitly.")
SHAPES = ("Keep A=(1,2), B=(3,4): q=120 gives n=1320, k>=120 and guaranteed rate 1/11. Components have 240 variables/120 checks and 480 variables/360 checks. Parent measured minima were sector-1 Z witnesses at weights 22--28, usually both Acols at one Bcol, especially Bcol3; no reported minimum occupied sector 2. Exact full ranks retain k/n<0.125.")
FILLING = ("Twenty cards cross five inequivalent B count profiles with four relative-order enrichment rules. Every A has counts (4,5); every B column has three terms, including new three-term cells, so both row-weight caps are <=12. Four-cycle-free A labels receive localized-kernel, central-fixed-kernel and full-kernel screening. Independently selected A/B shortlists receive fresh longer cancellation searches before verified full pairing. Small localized kernels are exhaustively searched; other readings remain upper bounds, not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r010_SL2F5_12x34_center_local_independent_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL(2,5) has order 120")
    els = [(a,b,c,d) for a in range(5) for b in range(5)
           for c in range(5) for d in range(5)
           if (a*d-b*c) % 5 == 1]
    pos = {g:i for i,g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 5 for x in g)

    def mul(g,h):
        a,b,c,d = g
        e,f,u,v = h
        return ((a*e+b*u)%5, (a*f+b*v)%5,
                (c*e+d*u)%5, (c*f+d*v)%5)

    def inv(g):
        a,b,c,d = g
        return (d%5, (-b)%5, (-c)%5, a%5)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1,0,0,1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


def _component_reading(rng, rows, nc, trials):
    """Connectedness, four-cycles, and sampled classical kernel weights.
    Random column elimination produces actual kernel vectors, hence
    upper bounds only; this is not a quantum distance certificate."""
    columns = [0] * nc
    parent = list(range(nc))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = {}
    cycles = 0
    for r, row in enumerate(rows):
        support = []
        while row:
            bit = row & -row
            c = bit.bit_length() - 1
            support.append(c)
            columns[c] |= 1 << r
            row ^= bit
        if support:
            anchor = root(support[0])
            for c in support[1:]:
                parent[root(c)] = anchor
        for p, c in enumerate(support):
            for d in support[p+1:]:
                key = (c, d)
                previous = pairs.get(key, 0)
                cycles += previous
                pairs[key] = previous + 1

    connected = (all(columns)
                 and len({root(c) for c in range(nc)}) == 1)
    if not connected:
        return 0, cycles, False

    reading = nc + 1
    for _ in range(trials):
        order = list(range(nc))
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome = columns[c]
            vector = 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, vector)
                    break
                syndrome ^= old[0]
                vector ^= old[1]
            if not syndrome:
                reading = min(reading, _popcount(vector))
    return reading, cycles, True


def NEW_CTX(rng, G, idx):
    """Independent component selection with localized and central-fixed probes."""
    q = G.q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    profile, model = card // 4, card % 4
    diagnostic_seed = rng.randrange(1 << 62)

    # COLUMN-major triples. Different zero counts, concentrations or
    # row-degree multisets distinguish these weighted protographs.
    profiles = (
        ((1,1,1), (1,1,1), (1,1,1), (1,1,1)),
        ((2,1,0), (0,2,1), (1,0,2), (1,1,1)),
        ((2,1,0), (2,0,1), (0,2,1), (0,1,2)),
        ((3,0,0), (0,2,1), (0,1,2), (1,1,1)),
        ((2,1,0), (1,2,0), (0,1,2), (1,1,1)),
    )
    bc = profiles[profile]
    sa = [[4, 5]]
    sb = [[bc[j][i] for j in range(4)] for i in range(3)]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache native group operations locally. The screening group is only
    # an integer encoding of EXACTLY the frozen group's multiplication.
    elements = G.elements
    table = [[G.idx(G.mul(g, h)) for h in elements] for g in elements]
    inverses = [G.idx(G.inv(g)) for g in elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    orders = []
    for g in range(q):
        value = identity
        for exponent in range(1, q + 1):
            value = table[value][g]
            if value == identity:
                orders.append(exponent)
                break
        else:
            raise ValueError("invalid native element order")

    # Obtain every monomial block from the VERIFIED construction.
    # Subsequent assembly uses only GF2 linearity and disjoint columns.
    xx, zz, _ = lifted_matrices(
        probe, [[[g] for g in range(q)]], [[[]]])
    left = [[(row >> (g*q)) & mask for row in xx] for g in range(q)]
    del xx, zz
    xx, zz, _ = lifted_matrices(
        probe, [[[]]], [[[g] for g in range(q)]])
    right = [[(row >> (g*q)) & mask for row in zz] for g in range(q)]
    del xx, zz

    def assemble(grid, side):
        blocks = left if side == "A" else right
        rows = [0] * (len(grid) * q)
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                block = [0] * q
                for g in cell:
                    source = blocks[g]
                    for r in range(q):
                        block[r] ^= source[r]
                for r in range(q):
                    rows[i*q+r] |= block[r] << (j*q)
        return rows

    def columns_of(rows, width):
        columns = [0] * width
        for r, source in enumerate(rows):
            while source:
                bit = source & -source
                columns[bit.bit_length() - 1] |= 1 << r
                source ^= bit
        return columns

    # Multiplication by the central -I pairs coordinates inside each
    # regular group block. XOR paired VERIFIED columns restricts checks
    # to vectors constant on each pair; no quotient lift is implemented.
    central = G.idx((4, 0, 0, 4))
    assert central != identity and table[central][central] == identity
    assert all(table[central][g] == table[g][central] for g in range(q))
    orbit_pairs = [(g, table[central][g]) for g in range(q)
                   if g < table[central][g]]
    assert len(orbit_pairs) == q // 2

    def fixed_columns(columns):
        return [columns[j*q+a] ^ columns[j*q+b]
                for j in range(len(columns) // q)
                for a, b in orbit_pairs]

    def dependency_basis(columns, order):
        """Actual kernel words in the original coordinates of these columns."""
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

    def localized_floor(rows):
        """Exact small single-cell kernels; larger ones get upper readings."""
        readings, ranks = [], []
        for j in range(2):
            block = [(row >> (j*q)) & mask for row in rows]
            rank = gf2_rank(block)
            ranks.append(rank)
            if rank == q:
                readings.append(q + 1)
                continue
            columns = columns_of(block, q)
            basis = dependency_basis(columns, range(q))
            assert len(basis) == q - rank
            reading = min(word.bit_count() for word in basis)
            if len(basis) <= 14:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    reading = min(reading, word.bit_count())
            else:
                for i, word in enumerate(basis):
                    for other in basis[:i]:
                        reading = min(reading, (word ^ other).bit_count())
            readings.append(reading)
        return min(readings), tuple(ranks)

    def kernel_probe(columns, seed, trials):
        """Independent systematic words and nonzero cancellation descent.

        Every tested word lies in the kernel of the supplied VERIFIED
        columns. A central-fixed reading is multiplied by two by the
        caller, since its embedding repeats each bit on a coordinate pair.
        """
        diagnostic = random.Random(seed)
        width = len(columns)
        order = list(range(width))
        reading = width + 1
        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = dependency_basis(columns, order)
            if not basis:
                return width + 1
            light = sorted(basis, key=int.bit_count)[:12]
            reading = min(reading, light[0].bit_count())

            # Unlike the parent's light-pool-only pair test, inspect
            # light words against the ENTIRE independent kernel basis.
            for word in light[:8]:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            # Strictly nonzero descent explores higher-order combinations.
            descent = list(basis)
            diagnostic.shuffle(descent)
            for original in light[:4]:
                word = original
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in descent:
                        candidate = word ^ other
                        if not candidate:
                            continue
                        w = candidate.bit_count()
                        if w < weight:
                            word, weight = candidate, w
                            changed = True
                    reading = min(reading, weight)
                    if not changed:
                        break
            if len(light) >= 3:
                for _ in range(48):
                    a, b, c = diagnostic.sample(light, 3)
                    candidate = a ^ b ^ c
                    if candidate:
                        reading = min(reading, candidate.bit_count())
            if reading <= 8:
                break
        return reading

    def relatives(g, h):
        return table[g][inverses[h]], table[h][inverses[g]]

    # These are enrichment objectives, not infeasible hard order gates.
    # Extra cohorts alter the objective rather than relabeling a code.
    targets = ((5, 6, 10), (10,), (6,), (5,))
    target = targets[(model + cohort) % 4]

    def fresh_A():
        for restart in range(48):
            used, cells = set(), []
            failed = False
            for count in (4, 5):
                cell = [identity]
                while len(cell) < count:
                    choices = []
                    for g in rng.sample(range(q), 40):
                        if g in cell:
                            continue
                        ds = [d for h in cell for d in relatives(g, h)]
                        if len(set(ds)) != len(ds) or used.intersection(ds):
                            continue
                        penalty = sum(orders[d] not in target for d in ds)
                        # Avoid sacrificing all relative-order diversity
                        # merely to maximize one preferred order.
                        diversity = len({orders[d] for d in ds})
                        choices.append((penalty, -diversity, rng.random(),
                                        g, ds))
                    if not choices:
                        failed = True
                        break
                    choices.sort(key=lambda item: item[:3])
                    chosen = choices[rng.randrange(min(5, len(choices)))]
                    cell.append(chosen[3])
                    used.update(chosen[4])
                if failed:
                    break
                cells.append(cell)
            if not failed:
                return [cells]
        return None

    def fresh_B_cell(count):
        if count < 2:
            return rng.sample(range(q), count)
        for _ in range(256):
            cell = rng.sample(range(q), count)
            ds = [d for g in cell for h in cell if g != h
                  for d in (table[g][inverses[h]],)]
            if len(set(ds)) == len(ds):
                return cell
        raise ValueError("could not construct distinct-relative B cell")

    def component_record(grid, side, serial):
        rows = assemble(grid, side)
        width = (2 if side == "A" else 4) * q
        _, cycles, connected = _component_reading(
            random.Random(0), rows, width, 0)
        if not connected:
            return None
        if side == "A" and cycles:
            raise ValueError("native Sidon labels failed verified C4 check")
        columns = columns_of(rows, width)
        fixed = fixed_columns(columns)
        if side == "A":
            local, cell_ranks = localized_floor(rows)
            trials, fixed_trials, offset = 4, 3, 17
        else:
            local, cell_ranks = width + 1, ()
            trials, fixed_trials, offset = 2, 2, 31
        full = kernel_probe(columns, diagnostic_seed + offset, trials)
        center = 2 * kernel_probe(
            fixed, diagnostic_seed + offset + 101, fixed_trials)
        reading = min(local, full, center)
        rank = gf2_rank(rows)
        key = (reading, min(full, center), full + center,
               -cycles, rank, min(cell_ranks) if cell_ranks else rank)
        return dict(grid=grid, rows=rows, columns=columns, fixed=fixed,
                    local=local, full=full, center=center, d=reading,
                    cycles=cycles, rank=rank, cell_ranks=cell_ranks,
                    serial=serial, key=key)

    pools = {"A": [], "B": []}
    signatures = {"A": set(), "B": set()}

    def retain(grid, side, serial):
        if grid is None:
            return
        signature = tuple(tuple(sorted(cell)) for row in grid for cell in row)
        if signature in signatures[side]:
            return
        signatures[side].add(signature)
        record = component_record(grid, side, serial)
        if record is None:
            return
        pools[side].append(record)
        pools[side].sort(key=lambda item: item["key"], reverse=True)
        del pools[side][6:]

    # Components are generated and shortlisted independently. A promising
    # A no longer disappears because its original random B was weak.
    for trial in range(80):
        retain(fresh_A(), "A", trial)
    for trial in range(64):
        if trial >= 32 and len(pools["B"]) >= 6:
            break
        B = [[fresh_B_cell(count) for count in row] for row in sb]
        retain(B, "B", trial)

    def deepen(side):
        pool = pools[side]
        if not pool:
            raise ValueError("empty independently screened component pool")
        count = 4 if side == "A" else 3
        full_trials = 128 if side == "A" else 24
        fixed_trials = 64 if side == "A" else 24
        offset = 700001 if side == "A" else 900001
        finalists = pool[:count]
        for item in finalists:
            # Fresh common permutations reduce finalist-specific luck.
            item["full"] = min(item["full"], kernel_probe(
                item["columns"], diagnostic_seed + offset, full_trials))
            item["center"] = min(item["center"], 2 * kernel_probe(
                item["fixed"], diagnostic_seed + offset + 100003,
                fixed_trials))
            item["d"] = min(item["local"], item["full"], item["center"])
            item["key"] = (
                item["d"], min(item["full"], item["center"]),
                item["full"] + item["center"], -item["cycles"],
                item["rank"],
                min(item["cell_ranks"]) if item["cell_ranks"] else item["rank"])
        finalists.sort(key=lambda item: item["key"], reverse=True)
        return finalists

    aa, bb = deepen("A"), deepen("B")

    def pair_key(pair):
        a, b = pair
        return (min(a["d"], b["d"]), a["d"] + b["d"],
                -b["cycles"], a["rank"] + b["rank"],
                a["key"], b["key"])

    proposals = sorted(((a, b) for a in aa for b in bb),
                       key=pair_key, reverse=True)
    for a, b in proposals:
        A = [[[elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= q and 8*k < n):
            continue

        # Assert that the cached linear screens were exactly the components
        # in the authoritative final construction, including conventions.
        ar = [(row & mask) | (((row >> (4*q)) & mask) << q)
              for row in hx[:q]]
        br = [row & ((1 << (4*q)) - 1) for row in hz[:3*q]]
        assert ar == a["rows"] and br == b["rows"]
        assert n == 1320
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (profile, model, cohort), "diagnostic": pair_key((a, b)),
            "component_readings": (a["d"], b["d"]),
            "central_fixed_readings": (a["center"], b["center"]),
            "A_cell_ranks": a["cell_ranks"],
            "component_ranks": (a["rank"], b["rank"]),
        }
    raise ValueError("no independently selected pair passed exact rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the already selected full cell, with no further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Connected components alone exclude sub-12 logical witnesses: parent slots 29 and 32 had d2s=8 and slot 26 had d2s=10 despite passing the connectivity gate.
# REFUTED: A four-cycle-free A component and a short-budget reading >=32 exclude sub-24 logicals: parent slot 54 fell from d2s=32 to d30s=22, with sector-1 Z witnesses concentrated at individual Bcols despite deeper sampled component selection.
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
