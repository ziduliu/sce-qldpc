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
CONCEPT = ("For the frozen Frobenius permutation group C7 semidirect C3, order 21, shapes (2,4)x(3,5), lean B supports and rank-safe A reinforcement should reduce collective X-memory failures and improve circuit fitness above 5.8925.")
GROUP = ("Keep the exact permutation-closure group for L1. This bed scored 5.8925, while the same order-21 group reached about 6.9 on (3,4)x(3,4). These readings motivate a filling experiment, not a causal claim about the group. The fixed group uses q=21 and n=546.")
SHAPES = ("Frozen (2,4)x(3,5): n=546, kb=84 and guaranteed rate 4/26=0.15385; the size factor permits q<=23, but the frozen group has order 21. A has 42 checks on 84 variables and B has 63 checks on 105, each with dimension at least 42. Verified full ranks retain k=84..95 in cell 0.150. Parent X failures span Acol/Bcol data and Acol/Brow checks with mostly 1.1-1.2x enrichment, insufficient to identify one repair block.")
FILLING = ("Twenty contracts combine five A count architectures with lean B architectures of 15-18 terms. Most A columns have degree four; lower-exposure cards retain selected degree-three columns. Both check-weight sums remain <=12, and six-round CNOT budgets are computed exactly from counts. Quotient restrictions prevent all-doubleton A from vanishing on the entire C3 quotient. Fresh labels undergo verified-singleton-action annealing, rank and connectivity screening, exact kernel detection through weight six, then sampled short-witness multiplicity and anchored boundary screening. Complete CSS ranks decide the rate. This targets the parent's 32-101 X failures, collective flips across all logicals and substantial stage-three fall-through, versus only 0-3 Z failures. These structural screens are hypotheses about circuit pL, not reliability certificates.")
FAMILY_NAME = "r015_permF21_24x35_leanB_witnessdensity_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 21

def custom_ops(q):
    # Permutation closure in S7, using composition g after h.
    # The generators do not commute; closure has exactly 21 elements.
    assert q == 21
    e = tuple(range(7))
    generators = [
        tuple((x + 1) % 7 for x in range(7)),
        tuple((2 * x) % 7 for x in range(7)),
    ]

    def mul(g, h):
        return tuple(g[h[x]] for x in range(7))

    els = [e]
    positions = {e: 0}
    for g in els:
        for h in generators:
            gh = mul(g, h)
            if gh not in positions:
                positions[gh] = len(els)
                els.append(gh)
    assert len(els) == q

    def inv(g):
        result = [0] * 7
        for x, y in enumerate(g):
            result[y] = x
        return tuple(result)

    def red(g):
        return tuple(g)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=e, mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _component_checks(hx, hz, q):
    """Extract component checks from the verified full CSS construction."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ac = []
    for i in range(ra):
        for t in range(q):
            full = hx[(i * cb) * q + t]
            row = 0
            for j in range(ca):
                row |= ((full >> (j * cb * q)) & mask) << (j * q)
            ac.append(row)
    bmask = (1 << (cb * q)) - 1
    bc = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ac, bc


def _small_component_kernel(rows, n):
    """Exactly detect a nonzero component kernel word of weight <=6.

    A collision between syndromes of subsets of size <=3 witnesses
    such a word. This is a classical component test, not a certificate
    of quantum distance or circuit-level performance.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    seen = {0}
    for s in cols:
        if s in seen:
            return True
        seen.add(s)
    for i, a in enumerate(cols):
        for j in range(i):
            s = a ^ cols[j]
            if s in seen:
                return True
            seen.add(s)
    for i, a in enumerate(cols):
        for j in range(i):
            ab = a ^ cols[j]
            for t in range(j):
                s = ab ^ cols[t]
                if s in seen:
                    return True
                seen.add(s)
    return False


def _same_basis_cycles(rows):
    """Four-cycle count and largest overlap between distinct checks."""
    cycles, peak = 0, 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            cycles += overlap * (overlap - 1) // 2
            if overlap > peak:
                peak = overlap
    return cycles, peak


def _planned_labels(rng, G, idx, counts, affine):
    """Controlled quotient profiles, not automorphism copies of a seed."""
    mode = idx % 5
    phase = idx // 20
    quota = (1, 2, 3, 4, 2)[mode]
    grids = {}
    for side in ("A", "B"):
        grid = []
        width = len(counts[side][0])
        for i, crow in enumerate(counts[side]):
            row = []
            for j, weight in enumerate(crow):
                anchor = G.pick(rng)
                cell = [anchor]
                p = i * width + j
                cross = ((p + 2 * i + phase
                          + (1 if side == "B" else 0)) % 4 < quota)
                if mode == 4 and side == "B":
                    cross = not cross
                db = 1 + ((p + mode + phase) % 2) if cross else 0
                for term in range(1, weight):
                    # Trinomials use a 2+1 quotient occupancy, avoiding
                    # an imposed complete C3 orbit within one cell.
                    b = db
                    if term == 2:
                        b = db if (p + mode + phase) % 2 else 0
                        if db == 0:
                            b = 1 + ((p + mode + phase) % 2)
                    while True:
                        a = rng.randrange(7) if b else 1 + rng.randrange(6)
                        relative = affine[(a, b)]
                        if mode == 4 and (i + j + term) % 2:
                            g = G.mul(relative, anchor)
                        else:
                            g = G.mul(anchor, relative)
                        if g not in cell:
                            cell.append(g)
                            break
                row.append(cell)
            grid.append(row)
        grids[side] = grid
    return grids


def NEW_CTX(rng, G, idx):
    """Optimize fresh lean-support contracts, then verify the complete CSS."""
    import math

    q = G.q
    assert q == 21
    phase, design = divmod(idx, 20)
    if phase >= 3:
        raise ValueError("lean support/quotient contract book exhausted")

    a_profiles = (
        ((1, 3, 2, 2), (3, 1, 2, 2)),
        ((1, 1, 3, 3), (3, 3, 1, 1)),
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((2, 2, 1, 2), (1, 2, 2, 2)),
        ((1, 2, 2, 2), (3, 1, 2, 2)),
    )
    b_profiles = (
        ((1, 1, 1, 1, 1),
         (1, 1, 1, 1, 1),
         (1, 1, 1, 1, 1)),
        ((2, 1, 1, 1, 1),
         (1, 1, 1, 1, 1),
         (1, 1, 1, 1, 1)),
        ((2, 1, 1, 1, 1),
         (1, 2, 1, 1, 1),
         (1, 1, 1, 1, 1)),
        ((2, 1, 1, 1, 1),
         (1, 2, 1, 1, 1),
         (1, 1, 2, 1, 1)),
        ((2, 2, 1, 1, 1),
         (1, 1, 1, 1, 1),
         (1, 1, 1, 1, 1)),
    )
    # All-even A with monomial B already exceeds the target band's
    # rank budget at augmentation, so that pair is deliberately absent.
    cards = (
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (4, 0), (4, 1), (4, 2), (4, 3),
    )
    pa, pb = cards[design]
    counts = {
        "A": [list(row) for row in a_profiles[pa]],
        "B": [list(row) for row in b_profiles[pb]],
    }
    sa, sb = counts["A"], counts["B"]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= CELL_MAX
               for grid in counts.values() for row in grid for w in row)

    slope_to_b = {1: 0, 2: 1, 4: 2}
    affine = {
        (g[0], slope_to_b[(g[1] - g[0]) % 7]): g
        for g in G.elements
    }

    def doubleton_cross(side, i, j):
        if phase == 0:
            # Two independent quotient entries keep regular-doubleton A
            # from disappearing on every nontrivial C3 character.
            return side == "A" and pa == 2 and i == j
        if phase == 1:
            return (side == "B" or
                    (side == "A" and pa == 2 and i == j) or
                    (i + 2 * j + design) % 3 == 0)
        return (i + j) % 2 == 1

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if weight == 0:
            return []
        anchor = G.pick(rng)
        if weight == 1:
            return [anchor]
        if weight == 2:
            cross = doubleton_cross(side, i, j)
            b = rng.choice((1, 2)) if cross else 0
            a = rng.randrange(7) if b else rng.randrange(1, 7)
            return [anchor, G.mul(anchor, affine[a, b])]

        assert weight == 3
        if phase == 1 and (i + j) % 2 == 0:
            # A translated C7 difference triangle; its internal geometry
            # changes from the split-quotient triangles of phase zero.
            u = rng.randrange(1, 7)
            rels = [affine[0, 0], affine[u, 0],
                    affine[(3 * u) % 7, 0]]
        elif phase == 2:
            # Three quotient cosets, but not a complete C3 coset orbit.
            while True:
                r1 = affine[rng.randrange(7), 1]
                r2 = affine[rng.randrange(7), 2]
                if G.mul(r1, r1) != r2:
                    break
            rels = [G.e, r1, r2]
        else:
            # Two labels in one quotient coset and one in another.
            rels = [G.e, affine[rng.randrange(1, 7), 0],
                    affine[rng.randrange(7), rng.choice((1, 2))]]
        cell = [G.mul(anchor, h) for h in rels]
        assert len({G._key(g) for g in cell}) == weight
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j)
                 for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache evaluator-convention singleton actions locally. Composition
    # below is exact F2 linearity of these verified matrices, not a new
    # implementation of left/right lifting.
    mask = (1 << q) - 1
    unit = [[[G.e]]]
    actions = {"A": [], "B": []}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], unit)
        _, hz, _ = lifted_matrices(G, unit, [[[g]]])
        actions["A"].append(tuple(row & mask for row in hx))
        actions["B"].append(tuple(row & mask for row in hz))

    def component(side, grid):
        rows = []
        for crow in grid:
            block = [0] * q
            for j, cell in enumerate(crow):
                for g in cell:
                    term = actions[side][G.idx(g)]
                    for r, word in enumerate(term):
                        block[r] ^= word << (j * q)
            rows.extend(block)
        return rows

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    augmentation_rank = {
        side: gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in support
        ])
        for side, support in counts.items()
    }

    def assess(side, grid):
        rows = component(side, grid)
        width = len(grid[0]) * q
        cols = columns(rows, width)
        nr = len(rows)
        adjacency = [0] * nr
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adjacency[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (col.bit_count() for col in cols))
        # Exact six-cycles without C4; otherwise only a proposal proxy.
        six = max(0, triangles - stars)
        forced = len(grid) - augmentation_rank[side]
        excess = max(0, nr - gf2_rank(rows) - forced)
        energy = 400.0 * excess + 100.0 * four / q + six / (8.0 * q)
        return energy, rows, cols, excess, four, six

    def connectivity(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            first = (row & -row).bit_length() - 1
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(i) for i in range(width)}) - 1

    def sampled_exposure(side, cols):
        """Verified fundamental/pair-XOR words; no distance certificate."""
        srng = random.Random(194291 + 1009 * idx + (side == "B"))
        order = list(range(len(cols)))
        minimum = len(cols) + 1
        witnesses = set()
        threshold = 14 if side == "A" else 12

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= threshold and len(witnesses) < 16384:
                witnesses.add(word)

        for trial in range(112):
            srng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    record(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    record(word ^ other)
            if minimum <= 6:
                break

        low10 = sum(word.bit_count() <= 10 for word in witnesses)
        low12 = sum(word.bit_count() <= 12 for word in witnesses)
        loads = [0] * (len(cols) // q)
        for word in witnesses:
            if word.bit_count() > 12:
                continue
            for j in range(len(loads)):
                loads[j] += bool((word >> (j * q)) & mask)
        return minimum, low10, low12, len(witnesses), max(loads)

    def boundary_exposure(cols):
        """Anchored triple-boundary heuristic, not an exact spectrum."""
        costs = []
        for anchor in range(0, len(cols), q):
            total = 0
            for i in range(len(cols)):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ cols[i]
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 4:
                        total += (6 - weight) ** 2
            costs.append(total)
        return max(costs), sum(costs)

    def portfolio(side):
        support = counts[side]
        positions = [(i, j) for i, row in enumerate(support)
                     for j, w in enumerate(row) if w]
        pool = []
        seen = set()
        serial = 0

        def remember(grid, stats):
            nonlocal serial
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            coarse = stats[3], stats[4], stats[5]
            pool.append((coarse, serial, clone(grid), stats))
            pool.sort(key=lambda item: item[:2])
            del pool[10:]

        for restart in range(5):
            grid = draw_grid(side)
            current = assess(side, grid)
            local_grid, local = clone(grid), current
            for step in range(320):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed = assess(side, grid)
                delta = proposed[0] - current[0]
                temperature = 14.0 * (0.988 ** step) + 0.12
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    current = proposed
                    if current[0] < local[0]:
                        local_grid, local = clone(grid), current
                else:
                    grid[i][j] = old
                if step in (159, 239):
                    remember(local_grid, local)
            remember(local_grid, local)
            remember(grid, current)

        finalists = []
        for coarse, serial, grid, stats in pool:
            _, rows, cols, excess, four, six = stats
            width = len(cols)
            disconnected = connectivity(rows, width)
            bad = int(_small_component_kernel(rows, width))
            d, low10, low12, low, load = sampled_exposure(side, cols)
            peak, risk = boundary_exposure(cols)
            target = 14 if side == "A" else 12
            key = (disconnected, excess, bad, max(0, target - d),
                   low10, low12, load, low, peak, risk, four, six)
            finalists.append({
                "grid": grid, "rows": rows, "key": key,
                "disconnected": disconnected, "excess": excess,
                "bad": bad, "d": d, "low10": low10,
                "low12": low12, "low": low, "load": load,
                "peak": peak, "risk": risk,
                "four": four, "six": six,
            })
        finalists.sort(key=lambda item: item["key"])
        return finalists[:5]

    apool, bpool = portfolio("A"), portfolio("B")
    winner = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 546 and 84 <= k <= 95):
                continue
            actual_a, actual_b = _component_checks(hx, hz, q)
            assert actual_a == a["rows"] and actual_b == b["rows"]
            key = (
                a["disconnected"] + b["disconnected"],
                2 * a["bad"] + b["bad"],
                2 * max(0, 14 - a["d"]) + max(0, 12 - b["d"]),
                a["low10"], b["low10"],
                a["low12"] + b["low12"],
                max(a["load"], b["load"]),
                a["low"] + b["low"],
                max(a["peak"], b["peak"]),
                a["risk"] + b["risk"],
                a["excess"] + b["excess"],
                a["four"] + b["four"],
                a["six"] + b["six"], -k,
            )
            if winner is None or key < winner[0]:
                winner = key, A, B, hx, hz, k

    # A finite structural search can miss the target ranks. Keep the
    # same count/quotient contract and search fresh legal fillings;
    # never repair an invalid submitted slot or substitute predicted k.
    if winner is None:
        for trial in range(256):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 546 and 84 <= k <= 95):
                continue
            ar, br = _component_checks(hx, hz, q)
            af, ap = _same_basis_cycles(ar)
            bf, bp = _same_basis_cycles(br)
            key = (max(ap, bp), af + bf, -k)
            if winner is None or key < winner[0]:
                winner = key, A, B, hx, hz, k
        if winner is None:
            raise ValueError("contract missed verified rate cell 0.150")

    key, A, B, hx, hz, k = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    expected_cnots = (
        8 * sum(map(sum, sa)) + 6 * sum(map(sum, sb))
    ) * q * 6
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
    return {"idx": idx, "cells": {"A": A, "B": B},
            "contract": (pa, pb, phase), "k": k,
            "screen": key, "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected complete cell without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: rolled counts and uniform labels suffice for balanced protection on the cyclic Z23 (2,4)x(3,5) bed; parent slot 13 had 0/200000 Z failures but 108/1000 X failures and weight-six Z logicals.
# REFUTED: Balanced degree-three/four A columns alone reliably balance memory bases on this F21 bed: all five measured codes remained X-limited, with 32-101 X failures versus only 0-3 Z failures.
# REFUTED: Equal d2s implies equal circuit reliability on this F21 bed: slots 0 and 3 both had d2s=10, but X failed 40/200000 versus 101/29000 shots; differing counts and labels prevent attributing this solely to distance or weight.
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
