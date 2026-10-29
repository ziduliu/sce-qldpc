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
CONCEPT = ("For SL2(F5), order 120, shapes (1,2)x(2,5), localized A-cell kernel screening followed by witness-directed optimization of both complete components should raise five-code fitness above 109.4042 in cell 0.250. The experimental target is five distance readings >=24; all readings remain provisional upper bounds.")
GROUP = ("Keep the frozen SL2(F5) matrix group, order 5*(25-1)=120. This exact bed leads cell 0.250 with measured fitness 109.4042 and five readings 22,21,21,20,20. These measurements motivate filling optimization on the existing bed, not a certified distance claim.")
SHAPES = ("Keep A=(1,2), B=(2,5): n=1440, k>=360 and guaranteed rate 0.250. The size window permits q<=124; the frozen group fixes q=120. Unequal components have 240 variables/120 checks and 600 variables/240 checks. Parent minima occupied only sector-1 Acol/Bcol blocks: X witnesses usually used Acol1 across several B columns, while Z witnesses used both A columns at individual B columns. Exact full ranks retain [0.225,0.275).")
FILLING = ("Retain A counts (4,4), B row sums eight and column sums (4,3,3,3,3), hence both quantum row-weight maxima twelve. Twenty topology/relative-order cards now screen individual A-cell kernels, exactly enumerating dimensions <=15 and heuristically probing larger kernels. Optimize complete verified components against accumulated surviving witnesses using targeted label mutations, exact small-kernel checks, near-full component ranks and independent longer finalist searches. Cached monomial blocks come only from verified lifts and are checked against the final construction. No sampled kernel reading certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_SL2F5_12x25_localized_witness_mutation_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL2(F5) requires order 120")
    p = 5
    # Encoding (a,b,c,d) represents the matrix [[a,b],[c,d]].
    els = [(a,b,c,d)
           for a in range(p) for b in range(p)
           for c in range(p) for d in range(p)
           if (a*d-b*c) % p == 1]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(x % p for x in g)

    def mul(g, h):
        a,b,c,d = g
        e,f,u,v = h
        return ((a*e+b*u) % p, (a*f+b*v) % p,
                (c*e+d*u) % p, (c*f+d*v) % p)

    def inv(g):
        a,b,c,d = g
        return (d % p, (-b) % p, (-c) % p, a % p)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty relative-order/topology cards, independently screened."""
    from functools import lru_cache

    q = G.q
    assert q == 120
    zero_budget = (idx % 20) // 5
    model = idx % 5
    cohort = idx // 20
    diagnostic_seed = rng.randrange(1 << 62)

    # Cache group operations only. All binary matrices still come from
    # lifted_matrices, with exactly the original element indexing.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)
    orders = []
    for g in range(q):
        h, order = ident, 0
        while True:
            h = table[h][g]
            order += 1
            if h == ident:
                break
            if order > q:
                raise ValueError("invalid group operation")
        orders.append(order)
    relative_order = [
        [orders[table[inverses[g]][h]] for h in range(q)]
        for g in range(q)
    ]

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

    # Preferences concern g^-1*h, invariant under common left
    # translation and conjugation. They are not coset relabelings.
    penalties = (
        {2: 30, 3: 0, 4: 0, 5: 0, 6: 0, 10: 0},
        {2: 30, 3: 4, 4: 1, 5: 0, 6: 4, 10: 0},
        {2: 30, 3: 0, 4: 4, 5: 1, 6: 0, 10: 1},
        {2: 30, 3: 2, 4: 0, 5: 2, 6: 2, 10: 2},
        {2: 30, 3: 1, 4: 1, 5: 0, 6: 1, 10: 2},
    )

    def fresh_cell(count, policy):
        if not count:
            return []
        best, best_cost = None, None
        for _ in range(8):
            cell = rng.sample(range(q), count)
            rels = [relative_order[g][h]
                    for i, g in enumerate(cell) for h in cell[:i]]
            # Never intentionally introduce a central involution pair.
            if 2 in rels:
                continue
            cost = sum(penalties[policy].get(order, 30)
                       for order in rels)
            if best_cost is None or cost < best_cost:
                best, best_cost = cell, cost
        if best is None:
            for _ in range(256):
                cell = rng.sample(range(q), count)
                if all(relative_order[g][h] != 2
                       for i, g in enumerate(cell) for h in cell[:i]):
                    best = cell
                    break
        if best is None:
            raise ValueError("relative-order cell construction failed")
        return [G.elements[g] for g in best]

    def support_b():
        degrees = (4, 3, 3, 3, 3)

        @lru_cache(None)
        def feasible(j, upper, zeros):
            if upper < 0 or zeros < 0:
                return False
            if j == len(degrees):
                return upper == zeros == 0
            return any(
                feasible(j+1, upper-a,
                         zeros-int(a == 0 or a == degrees[j]))
                for a in range(degrees[j]+1)
            )

        assert feasible(0, 8, zero_budget)
        upper, zeros = 8, zero_budget
        grid = [[], []]
        for j, degree in enumerate(degrees):
            choices = [
                a for a in range(degree+1)
                if feasible(j+1, upper-a,
                            zeros-int(a == 0 or a == degree))
            ]
            a = rng.choice(choices)
            grid[0].append(a)
            grid[1].append(degree-a)
            upper -= a
            zeros -= int(a == 0 or a == degree)
        assert [sum(row) for row in grid] == [8, 8]
        assert _grid_weights([[4, 4]], grid) == 12 <= W_MAX
        assert _no_dead_lines([[4, 4]], grid)
        return grid

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    # Obtain both monomial conventions from the authoritative helper.
    # Assembling their XORs uses GF2 linearity, not a separate group lift.
    mask = (1 << q)-1
    monomials = [[[g] for g in range(q)]]
    lx, _, _ = lifted_matrices(probe, monomials, [[[]]])
    _, rz, _ = lifted_matrices(probe, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in lx]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in rz]
              for g in range(q)],
    }
    del lx, rz

    def cell_rows(cell, side):
        result = [0]*q
        for g in cell:
            block = blocks[side][G.idx(g)]
            for r in range(q):
                result[r] ^= block[r]
        return result

    def verified_component(grid, side):
        width = len(grid[0])*q
        rows = []
        for base_row in grid:
            lifted = [cell_rows(cell, side) for cell in base_row]
            rows.extend(
                sum(block[r] << (j*q)
                    for j, block in enumerate(lifted))
                for r in range(q)
            )
        columns = [0]*width
        for check, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << check
        return rows, columns

    single_cache = {}

    def single_geometry(cell):
        """Actual localized A words; exact minimum when nullity <=15."""
        key = tuple(sorted(G.idx(g) for g in cell))
        if key in single_cache:
            return single_cache[key]
        rows = cell_rows(cell, "A")
        columns = [0]*q
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        pivots, basis = {}, []
        for c, syndrome in enumerate(columns):
            word = 1 << c
            while syndrome:
                lead = syndrome.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)

        reading, witness = q+1, None

        def observe(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        if len(basis) <= 15:
            word = 0
            for t in range(1, 1 << len(basis)):
                word ^= basis[(t & -t).bit_length()-1]
                observe(word)
        else:
            for i, word in enumerate(basis):
                observe(word)
                for other in basis[:i]:
                    observe(word ^ other)
            # Larger kernels get a heuristic, not an exact minimum.
            local = random.Random(diagnostic_seed + sum(
                (i+1)*g for i, g in enumerate(key)))
            for _ in range(256):
                word = 0
                for other in local.sample(basis, min(4, len(basis))):
                    word ^= other
                observe(word)
        if witness is not None:
            assert all((witness & row).bit_count() % 2 == 0
                       for row in rows)
        result = (reading, witness, len(pivots))
        single_cache[key] = result
        return result

    def quality(rows, columns, exact=True):
        """Exact small-word counts when requested; genuine cycle statistics."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return None
        three = four_words = 0
        if exact:
            pairs = {}
            for i, a in enumerate(columns):
                for b in columns[:i]:
                    syndrome = a ^ b
                    three += int(syndrome in singles)
                    previous = pairs.get(syndrome, 0)
                    four_words += previous
                    pairs[syndrome] = previous+1
        # Cheap cold-pool statistics are never used as exact kernel results.

        uses, four_cycles = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for column in columns:
            neighbors = list(bits(column))
            degree = len(neighbors)
            forced += degree*(degree-1)*(degree-2)//6
            for i, u in enumerate(neighbors):
                adjacency[u] |= column ^ (1 << u)
                for v in neighbors[:i]:
                    key = (v, u)
                    previous = uses.get(key, 0)
                    four_cycles += previous
                    uses[key] = previous+1

        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u+1))):
                triangles += (
                    neighbors & adjacency[v] & (-1 << (v+1))
                ).bit_count()
        return ((three//3, four_words//3),
                (four_cycles, max(0, triangles-forced)))

    def kernel_probe(rows, columns, trials, diagnostic):
        """Genuine classical kernel witnesses, never a distance certificate."""
        width = len(columns)
        order = list(range(width))
        best, witness = width+1, None
        found = {}

        def observe(word):
            nonlocal best, witness, found
            if word:
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word
                    found = {w: wt for w, wt in found.items()
                             if wt <= best+2}
                if weight <= best+2:
                    found[word] = weight
                    if len(found) > 32:
                        found = dict(sorted(
                            found.items(),
                            key=lambda item: (item[1], item[0]))[:16])

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    observe(word)

            light = sorted(basis, key=int.bit_count)[:24]
            for word in light:
                for other in basis:
                    observe(word ^ other)

            # Cancellation descent searches beyond individual basis words.
            diagnostic.shuffle(basis)
            for initial in light[:8]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word = candidate
                            weight = word.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c)
            if best <= 4:
                break

        if witness is None:
            raise ValueError("no component kernel witness")
        assert all((witness & row).bit_count() % 2 == 0
                   for row in rows)
        words = sorted(found, key=lambda w: (w.bit_count(), w))[:16]
        assert all(all((word & row).bit_count() % 2 == 0
                       for row in rows) for word in words)
        return best, words

    def component_pool(side):
        """Screen localized words, then mutate surviving complete-component words."""
        side_id = int(side == "B")
        policy = (model + side_id + cohort) % 5
        seed = diagnostic_seed + 100003*side_id
        bank = set()
        records, signatures = [], set()

        def syndrome(columns, word):
            value = 0
            for c in bits(word):
                value ^= columns[c]
            return value

        def remember(words):
            bank.update(words)
            if len(bank) > 96:
                keep = sorted(bank, key=lambda w: (w.bit_count(), w))[:80]
                bank.clear()
                bank.update(keep)

        def effective(record):
            reading = record["d"]
            for word in bank:
                if word.bit_count() < reading:
                    if syndrome(record["columns"], word) == 0:
                        reading = word.bit_count()
            return reading

        def merit(record):
            return (tuple(-x for x in record["small"]),
                    effective(record),
                    tuple(-x for x in record["cycles"]),
                    record["rank"])

        def geometry(grid, trial):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in signatures:
                return None
            signatures.add(signature)
            rows, columns = verified_component(grid, side)
            stats = quality(rows, columns, exact=False)
            if stats is None:
                return None
            localized, localized_words = len(columns)+1, []
            cell_ranks = []
            if side == "A":
                for j, cell in enumerate(grid[0]):
                    reading, word, rank = single_geometry(cell)
                    localized = min(localized, reading)
                    cell_ranks.append(rank)
                    if word is not None:
                        localized_words.append(word << (j*q))
                # Do not spend complete-component searches on an already
                # verified localized word shorter than the parent winners.
                if localized < 20:
                    return None
            return {
                "grid": grid, "rows": rows, "columns": columns,
                "cycles": stats[1], "trial": trial,
                "localized": localized,
                "localized_words": localized_words,
                "cell_ranks": tuple(cell_ranks),
            }

        def deepen(record, trials, offset):
            diagnostic = random.Random(seed + offset)
            reading, words = kernel_probe(
                record["rows"], record["columns"], trials, diagnostic)
            record["d"] = min(record["d"], reading)
            remember(words)

        def complete(record, trials, offset):
            stats = quality(record["rows"], record["columns"], exact=True)
            if stats is None or stats[0] != (0, 0):
                return None
            rank = gf2_rank(record["rows"])
            # Near-full ranks prevent selection primarily for extra k.
            if rank < len(record["rows"])-4:
                return None
            record["rank"] = rank
            record["small"] = stats[0]
            record["cycles"] = stats[1]
            record["d"] = record["localized"]
            remember(record["localized_words"])
            deepen(record, trials, offset)
            return record

        # More fresh geometries compete, but expensive exact small-word
        # and complete-kernel searches are reserved for a shortlist.
        for trial in range(48):
            support = [[4, 4]] if side == "A" else support_b()
            grid = [[fresh_cell(count, policy) for count in row]
                    for row in support]
            record = geometry(grid, trial)
            if record is not None:
                records.append(record)

        # If the localized threshold was too restrictive, preserve a valid
        # experimental slot instead of silently substituting dead blocks.
        if not records and side == "A":
            for trial in range(48, 80):
                grid = [[fresh_cell(4, policy), fresh_cell(4, policy)]]
                record = geometry(grid, trial)
                if record is not None:
                    records.append(record)
        if not records:
            raise ValueError("empty localized/cycle component pool")

        records.sort(key=lambda r: (
            -r["localized"] if side == "A" else 0,
            r["cycles"], r["trial"]))
        pool = []
        for record in records:
            record = complete(record, 6, 17)
            if record is not None:
                pool.append(record)
            if len(pool) == 8:
                break
        if not pool:
            raise ValueError("no component passed exact small-kernel/rank screen")

        # First independently retest the leading alternatives.
        for record in sorted(pool, key=merit, reverse=True)[:3]:
            deepen(record, 24, 1009)
        current = max(pool, key=merit)

        for step in range(24):
            surviving = [
                word for word in bank
                if syndrome(current["columns"], word) == 0
            ]
            witness = (min(surviving, key=lambda w: (w.bit_count(), w))
                       if surviving else None)
            active = (sorted({c//q for c in bits(witness)})
                      if witness is not None else [])
            grid = [[list(cell) for cell in row]
                    for row in current["grid"]]
            for _ in range(2 if step % 6 == 0 else 1):
                j = (rng.choice(active)
                     if active and rng.random() < 0.85
                     else rng.randrange(len(grid[0])))
                occupied = [i for i in range(len(grid)) if grid[i][j]]
                i = rng.choice(occupied)
                cell = grid[i][j]
                t = rng.randrange(len(cell))
                if step % 5 == 0:
                    # Whole-cell changes can leave single-label plateaus.
                    grid[i][j] = fresh_cell(len(cell), policy)
                else:
                    candidates = []
                    for _ in range(12):
                        g = G.pick(rng)
                        if g in cell:
                            continue
                        ids = [G.idx(g)] + [
                            G.idx(h) for u, h in enumerate(cell) if u != t
                        ]
                        rels = [
                            relative_order[x][y]
                            for u, x in enumerate(ids) for y in ids[:u]
                        ]
                        if 2 not in rels:
                            candidates.append((
                                sum(penalties[policy].get(order, 30)
                                    for order in rels), g))
                    if candidates:
                        best_cost = min(x[0] for x in candidates)
                        cell[t] = rng.choice([
                            g for cost, g in candidates if cost == best_cost
                        ])

            proposed = geometry(grid, 100+step)
            if proposed is None:
                continue
            # Witness improvement may justify a modest cycle trade.
            if proposed["cycles"][0] > current["cycles"][0] + 2*q:
                continue
            proposed = complete(proposed, 4, 17)
            if proposed is None:
                continue
            pk, ck = merit(proposed), merit(current)
            accept = pk > ck
            if not accept and pk[:2] == ck[:2]:
                accept = (
                    proposed["cycles"][0] <= current["cycles"][0]+q
                    and rng.random() < 0.12
                )
            pool.append(proposed)
            pool = sorted(pool, key=merit, reverse=True)[:8]
            if accept:
                current = proposed

        finalists = sorted(pool, key=merit, reverse=True)[:3]
        for record in finalists:
            # Accumulate independent readings: deeper searches only lower d.
            deepen(record, 64 + 8*(cohort % 3), 700001)
            deepen(record, 16, 900001)
        finalists.sort(key=merit, reverse=True)
        for record in finalists:
            record["d"] = effective(record)
        return finalists[:2]

    aa, bb = component_pool("A"), component_pool("B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        tuple(-(x+y) for x, y in
              zip(ab[0]["small"], ab[1]["small"])),
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]/(2*q)-ab[1]["cycles"][0]/(5*q)),
        reverse=True)

    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 12*q == 1440
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= 3*q and 9*n <= 40*k < 11*n):
            continue
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) == 12 <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx+hz) <= 12
        # Verify cached GF2 assemblies against the authoritative full lift.
        actual_a = [
            (row & mask) | (((row >> (5*q)) & mask) << q)
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
        assert actual_a == a["rows"]
        assert actual_b == b["rows"]
        assert css_ok(hx, hz)
        return {
            "idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (zero_budget, model),
            "component_readings": (a["d"], b["d"]),
            "component_small_kernels": (a["small"], b["small"]),
            "component_cycles": (a["cycles"], b["cycles"]),
        }
    raise ValueError("screened pairings missed the exact target-rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the jointly designed and screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-cap saturation alone prevents light sector-2 logicals; parent w=12 still produced measured X witnesses of weight 6 in slots 2 and 9.
# REFUTED: The parent's degree/coset cards supply twenty inequivalent designs per cohort; slots 1, 2 and 20..22 instead collapsed to iso_dup_of_0.
# REFUTED: A parent screen reading >=22 reliably survives longer measurement at >=22; SL2(F5) slots 28 and 34 instead fell from 22 and 23 to measured weight 20, with minima entirely in sector 1.
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
