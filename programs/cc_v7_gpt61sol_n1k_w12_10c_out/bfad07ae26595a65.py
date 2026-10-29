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
CONCEPT = ("PSL(2,7), order 168, shapes (1,2)x(1,3): full-group-generating trinomials and witness-directed component optimization should improve measured five-code fitness beyond 44.6084 in cell 0.300. Non-improvement refutes this experiment; sampled distances remain upper bounds.")
GROUP = ("Keep the frozen PSL(2,7) group and order. On this exact bed, all measured parent slots used five trinomials and read 12,12,12,14,12; most mixed-count cards screened at seven. Relative labels generating the whole simple group exclude confinement of an individual cell to a proper subgroup, without asserting a quantum-distance theorem.")
SHAPES = ("Keep frozen A=(1,2), B=(1,3): n=1176, k>=336, guaranteed rate 2/7. Components have unequal lengths 336 and 504, with 168 checks each. Parent minima occupied sector-1 Acol/Bcol blocks; X witnesses frequently used one A column across all three B columns. Spend more optimization on the longer B component and enforce exact full rate in [0.275,0.325).")
FILLING = ("All five cells have three distinct elements: HX weight nine and HZ weight twelve. Twenty cards cross five relative-order allocations with four verified short-cycle objectives. Each trinomial excludes involutory ratios and its relative labels generate all 168 elements. Independently optimize components, mutate cells touched by surviving genuine kernel witnesses, then deepen finalist searches using independent information sets and all basis pairs. Cached single-term lifts come exclusively from lifted_matrices and final components are checked against the full verified lift. These screens are heuristic, not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r030_PSL2F7_12x13_fullgen_trinomial_witness_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 168

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("PSL(2,7) requires order 168")

    # SL(2,F7) modulo its central subgroup {I,-I}.
    # Elements encode a matrix (a,b;c,d), choosing min(M,-M).
    def red(g):
        v = tuple(int(x) % 7 for x in g)
        return min(v, tuple((-x) % 7 for x in v))

    els = []
    for a in range(7):
        for b in range(7):
            for c in range(7):
                for d in range(7):
                    g = (a, b, c, d)
                    if (a*d - b*c) % 7 == 1 and red(g) == g:
                        els.append(g)
    assert len(els) == q
    pos = {g: i for i, g in enumerate(els)}

    # Cache the exact quotient law once. Subsequent verified lifts
    # still use the scaffold, not an independently implemented lift.
    table = []
    for a, b, c, d in els:
        table.append([
            red((a*e + b*h, a*f + b*k,
                 c*e + d*h, c*f + d*k))
            for e, f, h, k in els
        ])
    inverses = {
        g: red((g[3], -g[1], -g[2], g[0])) for g in els
    }

    def mul(g, h):
        return table[pos[g]][pos[h]]

    def inv(g):
        return inverses[g]

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Five density allocations, crossed with four within-cell order spectra.
# Maximum HX/HZ weights: (9,12), (11,12), (10,12), (10,12), (9,11).
_PSL_COUNTS = (
    ((3, 3), (3, 3, 3)),
    ((4, 4), (2, 3, 3)),
    ((3, 4), (2, 3, 3)),
    ((3, 3), (2, 3, 4)),
    ((3, 3), (2, 3, 3)),
)


def _psl_bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def _psl_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _psl_bits(row):
            columns[c] |= 1 << r
    return columns


def _psl_no_kernel4(columns):
    """Exact exclusion of nonzero kernel vectors of weights one to four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = singles.copy()
    for i, a in enumerate(columns):
        for j in range(i):
            value = a ^ columns[j]
            if value in seen:
                return False
            seen.add(value)
    return True


def _psl_components(hx, hz, q):
    """Project components from the scaffold's actual LP column ordering."""
    mask = (1 << q) - 1
    ar = []
    for t in range(q):
        source = hx[t]  # Arow 0, Bcol 0.
        ar.append((source & mask)
                  | (((source >> (3*q)) & mask) << q))
    bmask = (1 << (3*q)) - 1
    br = [hz[t] & bmask for t in range(q)]  # Acol 0, Brow 0.
    return ar, br


def _psl_quality(rows, width):
    """Actual component C4 count and a projection-triangle heuristic."""
    columns = _psl_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    adjacency = [0] * len(rows)
    uses = {}
    four = forced = 0
    for signature in columns:
        ns = list(_psl_bits(signature))
        degree = len(ns)
        forced += degree * (degree - 1) * (degree - 2) // 6
        for i, u in enumerate(ns):
            adjacency[u] |= signature ^ (1 << u)
            for v in ns[:i]:
                key = v * len(rows) + u
                previous = uses.get(key, 0)
                four += previous
                uses[key] = previous + 1

    reached = frontier = 1
    while frontier:
        nxt = 0
        for u in _psl_bits(frontier):
            nxt |= adjacency[u]
        frontier = nxt & ~reached
        reached |= frontier
    if reached != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _psl_bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v]
                          & (-1 << (v + 1))).bit_count()
    return (four, max(0, triangles - forced)), columns


def _psl_circuit_reading(rng, columns, trials):
    """Find actual classical kernel words; readings are upper bounds only."""
    reading = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                lead = syndrome.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                reading = min(reading, word.bit_count())
        light = sorted(basis, key=int.bit_count)[:20]
        for a in light:
            for b in basis:
                if a != b:
                    reading = min(reading, (a ^ b).bit_count())
        if len(light) >= 3:
            for _ in range(80):
                a, b, c = rng.sample(light, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
        if reading <= 6:
            break
    return reading


def NEW_CTX(rng, G, idx):
    """Full-generating trinomials, verified cycles, and surviving witnesses."""
    import math
    from functools import lru_cache
    from itertools import combinations

    q = G.q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    recipe, mode = card // 4, card % 4
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Twenty design objectives, not twenty orbit copies. Extra cohorts
    # change A's relative-order allocation as well as the search objective.
    b_plans = (
        (0, 1, 2),
        (1, 1, 1),
        (1, 2, 2),
        (2, 2, 2),
        (1, 2, 3),
    )
    a_plans = ((1, 1), (1, 2), (2, 2), (0, 2))
    targets = {
        "A": a_plans[(mode + cohort) % 4],
        "B": b_plans[recipe],
    }
    diagnostic_seed = rng.randrange(1 << 60)

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid group element order")

    @lru_cache(None)
    def full_generated(u, v):
        # Finite-group positive closure equals the subgroup generated by
        # u,v. This concerns support geometry, not a replacement lift.
        reached = {G.e}
        frontier = [G.e]
        for h in frontier:
            for generator in (u, v):
                new = G.mul(h, generator)
                if new not in reached:
                    reached.add(new)
                    frontier.append(new)
        return len(reached) == q

    def cell_kind(cell):
        if len(set(cell)) != 3:
            return None
        relative = [
            G.mul(G.inv(cell[i]), cell[j])
            for i in range(3) for j in range(i)
        ]
        spectrum = [orders[g] for g in relative]
        if 2 in spectrum:
            return None
        u = G.mul(G.inv(cell[0]), cell[1])
        v = G.mul(G.inv(cell[0]), cell[2])
        if not full_generated(u, v):
            return None
        return spectrum.count(7)

    # An independently generated library of supports, not conjugates or
    # translations of one support. Missing exact spectrum buckets receive
    # the nearest available bucket rather than voiding a legal slot.
    buckets = {i: [] for i in range(4)}
    for _ in range(1024):
        cell = rng.sample(G.elements, 3)
        kind = cell_kind(cell)
        if kind is not None:
            buckets[kind].append(tuple(cell))
        if sum(len(v) for v in buckets.values()) >= 320:
            break
    available = [i for i in buckets if buckets[i]]
    if not available:
        raise ValueError("no full-group-generating trinomial sampled")
    actual = {
        side: tuple(min(available, key=lambda v: (abs(v-t), v))
                    for t in plan)
        for side, plan in targets.items()
    }

    def fresh(side):
        return [[list(rng.choice(buckets[kind]))
                 for kind in actual[side]]]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    @lru_cache(None)
    def verified_atom(g):
        """Both single-term component actions from the VERIFIED helper.

        Cache original bit rows, not an independently implemented group
        action. XOR and block shifts below merely assemble these verified
        linear summands. Finalists are checked against the full lift.
        """
        hx, hz, _ = lifted_matrices(G, [[[g]]], [[[g]]])
        return (tuple(row & mask for row in hx),
                tuple(row & mask for row in hz))

    def component(side, grid):
        which = int(side == "B")
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            shift = j * q
            for g in cell:
                summand = verified_atom(g)[which]
                for r, word in enumerate(summand):
                    rows[r] ^= word << shift
        return rows

    def inspect(side, grid):
        rows = component(side, grid)
        width = len(grid[0]) * q
        inspected = _psl_quality(rows, width)
        if inspected is None:
            return None
        stats, columns = inspected

        # Additional check-projection C4 statistic. This is not claimed
        # to be a Tanner C8 count, nor to bound component distance.
        adjacency = [0] * q
        for syndrome in columns:
            for r in _psl_bits(syndrome):
                adjacency[r] |= syndrome ^ (1 << r)
        projected = 0
        if (mode + cohort) % 4:
            for r, neighbors in enumerate(adjacency):
                for s in range(r):
                    common = (neighbors & adjacency[s]).bit_count()
                    projected += common * (common - 1) // 2
            projected //= 2
        multiplier = (0, 1, 4, 12)[(mode + cohort) % 4]
        cost = 100000*stats[0] + 16*stats[1] + multiplier*projected
        return {
            "cells": copy_grid(grid), "rows": rows, "cols": columns,
            "stats": stats + (projected,), "cost": cost,
        }

    def syndrome(columns, word):
        value = 0
        for c in _psl_bits(word):
            value ^= columns[c]
        return value

    def kernel_search(columns, seed, trials, deep=False):
        """Genuine null words, not lower bounds on classical distance.

        Deep searches examine every systematic-basis pair. Initial probes
        use pairs with at least one of the forty shortest basis words.
        Short triples and kernel-preserving descent broaden both searches.
        """
        diagnostic = random.Random(seed)
        order = list(range(len(columns)))
        best = len(columns) + 1
        witnesses = {}

        def note(word):
            nonlocal best, witnesses
            if not word:
                return
            weight = word.bit_count()
            if weight < best:
                best = weight
                witnesses = {
                    w: wt for w, wt in witnesses.items()
                    if wt <= best + 2
                }
            if weight <= best + 2:
                witnesses[word] = weight
                if len(witnesses) > 24:
                    witnesses = dict(sorted(
                        witnesses.items(), key=lambda item: item[1])[:12])

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (value, word)
                        break
                    value ^= previous[0]
                    word ^= previous[1]
                if not value:
                    basis.append(word)
                    note(word)

            basis.sort(key=int.bit_count)
            limit = len(basis) if deep else min(40, len(basis))
            for i in range(limit):
                word = basis[i]
                for other in basis[i+1:]:
                    note(word ^ other)
            short = basis[:(20 if deep else 12)]
            for a, b, c in combinations(short, 3):
                note(a ^ b ^ c)

            if deep and witnesses:
                # Adding another kernel vector preserves the syndrome.
                moves = list(basis)
                diagnostic.shuffle(moves)
                starts = sorted(witnesses, key=int.bit_count)[:2]
                for word in starts:
                    for sweep in range(3):
                        changed = False
                        for move in moves:
                            candidate = word ^ move
                            if candidate and (
                                    candidate.bit_count() < word.bit_count()):
                                word = candidate
                                changed = True
                        if not changed:
                            break
                    note(word)
            if best <= 8:
                break

        found = sorted(witnesses, key=int.bit_count)[:8]
        if not found:
            raise ValueError("component search found no nonzero kernel word")
        assert all(syndrome(columns, word) == 0 for word in found)
        return best, found

    def optimize(side):
        bank = set()
        width = len(actual[side]) * q
        seed = diagnostic_seed + (0 if side == "A" else 10000019)

        def remember_witnesses(words):
            nonlocal bank
            bank.update(word for word in words
                        if word.bit_count() <= 28)
            if len(bank) > 96:
                bank = set(sorted(bank, key=int.bit_count)[:64])

        def rating(record):
            reading = record["upper"]
            for word in bank:
                if word.bit_count() < reading and not syndrome(
                        record["cols"], word):
                    reading = word.bit_count()
            return (reading, -record["cost"])

        def probe(record, trials=10, offset=0, deep=False):
            reading, words = kernel_search(
                record["cols"], seed + offset, trials, deep)
            record["upper"] = min(record.get("upper", width+1), reading)
            record["words"] = words
            remember_witnesses(words)
            return record

        keepers = {}

        def retain(record):
            key = signature(record["cells"])
            old = keepers.get(key)
            if old is None or record["cost"] < old["cost"]:
                keepers[key] = record
            if len(keepers) > 8:
                worst = max(keepers, key=lambda k: keepers[k]["cost"])
                del keepers[worst]

        starts = []
        for _ in range(12 if side == "A" else 20):
            record = inspect(side, fresh(side))
            if record is not None:
                starts.append(record)
                retain(record)
        if not starts:
            raise ValueError("no connected full-generating component")
        starts.sort(key=lambda item: item["cost"])

        steps = 96 if side == "A" else 160
        restarts = 2 if side == "A" else 3
        for initial in starts[:restarts]:
            current = initial
            for step in range(steps):
                grid = copy_grid(current["cells"])
                j = rng.randrange(len(grid[0]))
                kind = actual[side][j]
                old = grid[0][j]
                if rng.random() < 0.25:
                    new = list(rng.choice(buckets[kind]))
                else:
                    new = list(old)
                    new[rng.randrange(3)] = G.pick(rng)
                    if cell_kind(new) != kind:
                        continue
                if set(new) == set(old):
                    continue
                grid[0][j] = new
                proposed = inspect(side, grid)
                if proposed is None:
                    continue
                delta = proposed["cost"] - current["cost"]
                temperature = q*(40*(1-step/steps)**2 + 0.5)
                accept = delta <= 0
                if not accept:
                    accept = rng.random() < math.exp(-delta/temperature)
                if accept:
                    current = proposed
                    retain(current)

        # Short-word exclusion is exact only through weight four.
        pool = []
        for record in keepers.values():
            if _psl_no_kernel4(record["cols"]):
                pool.append(probe(record))
        if not pool:
            # Preserve legal all-trinomial designs if this stronger
            # classical screen finds no survivor.
            pool = [probe(record) for record in
                    sorted(keepers.values(),
                           key=lambda item: item["cost"])[:4]]

        current = max(pool, key=rating)
        for step in range(12 if side == "A" else 20):
            surviving = [
                word for word in bank
                if not syndrome(current["cols"], word)
            ]
            witness = min(surviving, key=int.bit_count) if surviving else None
            active = []
            if witness:
                active = [c // q for c in _psl_bits(witness)]
            grid = copy_grid(current["cells"])
            j = (rng.choice(active) if active and rng.random() < 0.85
                 else rng.randrange(len(grid[0])))
            kind = actual[side][j]
            old = grid[0][j]
            new = None
            for _ in range(24):
                candidate = list(old)
                if step % 4 == 0:
                    candidate = list(rng.choice(buckets[kind]))
                else:
                    candidate[rng.randrange(3)] = G.pick(rng)
                if set(candidate) != set(old) and cell_kind(candidate) == kind:
                    new = candidate
                    break
            if new is None:
                continue
            grid[0][j] = new
            proposed = inspect(side, grid)
            if proposed is None:
                continue
            # Permit a modest cycle trade to remove a detected word.
            if proposed["stats"][0] > current["stats"][0] + q:
                continue
            if not _psl_no_kernel4(proposed["cols"]):
                continue
            proposed = probe(proposed)
            pk, ck = rating(proposed), rating(current)
            if pk > ck or (
                    pk[0] == ck[0] and
                    proposed["cost"] <= current["cost"] + 16*q and
                    rng.random() < 0.12):
                current = proposed
                pool.append(proposed)
                pool = sorted(pool, key=rating, reverse=True)[:6]

        # Independent orders lower readings rather than resetting them.
        first = sorted(pool, key=rating, reverse=True)[:3]
        for record in first:
            probe(record, trials=40, offset=100003, deep=True)
        final = sorted(first, key=rating, reverse=True)[:2]
        for record in final:
            probe(record, trials=96 if side == "B" else 64,
                  offset=200003, deep=True)
        final.sort(key=rating, reverse=True)
        for record in final:
            record["reading"] = rating(record)[0]
        return final

    afinal = optimize("A")
    bfinal = optimize("B")
    proposals = []
    for aa in afinal:
        for bb in bfinal:
            da, db = aa["reading"], bb["reading"]
            key = (min(da, db), da+db, -aa["cost"]-bb["cost"])
            proposals.append((key, aa, bb))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for quality, aa, bb in proposals:
        A, B = aa["cells"], bb["cells"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 1176
        ar, br = _psl_components(hx, hz, q)
        # Validate cached linear assembly against the exact evaluator lift.
        assert ar == aa["rows"] and br == bb["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= 2*q and 11*n <= 40*k < 13*n):
            continue
        assert max(row.bit_count() for row in hx + hz) <= 12
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        return {
            "idx": idx, "A": A, "B": B, "k": k,
            "card": (recipe, mode, cohort),
            "relative_order_targets": actual,
            "component_readings": (aa["reading"], bb["reading"]),
            "component_statistics": (aa["stats"], bb["stats"]),
            "quality": quality,
        }
    raise ValueError("no optimized trinomial pair passed exact rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the screened full-generating trinomial without new draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the row-weight cap suffices to eliminate sub-12 sector-1 logicals; parent D298 slots 4 and 5 had w=12 but measured distance eight, with X minima entirely in sector 1.
# REFUTED: Replacing trinomials with order-seven binomial cells preserves distance >=12 on PSL(2,7), (1,2)x(1,3); the mixed-count parent cards repeatedly had screening upper bounds of seven, while all five measured cards used only trinomials.
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
