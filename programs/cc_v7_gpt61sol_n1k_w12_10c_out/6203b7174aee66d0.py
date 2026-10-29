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
CONCEPT = ("For perfect A5, order 60, A=(1,3), B=(2,5), exclusively cubic components with generating order-spectrum cards and independently challenged witness repair should produce five measured distance readings >=12 and improve the 0.350 incumbent's fitness 35.747. Failure refutes this experiment; readings remain upper bounds.")
GROUP = ("Jump from metabelian S3 x D14, order 84, to perfect simple A5, order 60. The parent's five measured distances were eight. Related perfect SL2(F5) cubic components reached readings twelve on a different shape; A5 itself reached eight on (1,3)x(2,7), so neither comparison establishes distance transfer. This A5/shape bed is absent from BEDS. Fixed q=60 gives n=1020.")
SHAPES = ("Keep A=(1,3), B=(2,5): n=17*60=1020, k>=6*60=360, guaranteed rate 6/17=0.352941 targets cell 0.350. Components have 180 variables/60 checks and 300 variables/120 checks, nominal rates 2/3 and 3/5. Parent weight-eight minima were entirely sector-1 Acol/Bcol operators, spread across all Acols and several Bcols. Protect both forward kernels and retain the exact [0.325,0.375) rate gate.")
FILLING = ("Twenty cards cross five inequivalent cubic B concentration profiles with four A relative-order compositions. All component columns have degree three: A-row total nine, B-row totals seven/eight, HX maximum twelve and HZ maximum eleven. Generating trinomials have spectra (3,3,5), (3,5,5), or (5,5,5), distinct directed differences and no involution pair; B binomials use prescribed orders three/five. Authoritative scaffold monomial blocks supply every component. Geometry shortlisting, exact kernel-four exclusion, retained genuine null words, targeted cell repair and deeper independent parity/elimination challenges select fresh fillings. Native final lifts verify assembly, ranks, weights and CSS. Component searches do not certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_A5_13x25_cubic_spectrum_witness12_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 60

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A5 requires order 60")
    # Tuple g records the images of 0..4. Composition is g after h.
    els = [g for g in permutations(range(5))
           if sum(g[i] > g[j] for i in range(5)
                  for j in range(i+1, 5)) % 2 == 0]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) for x in g)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(5))

    def inv(g):
        answer = [0]*5
        for i, image in enumerate(g):
            answer[image] = i
        return tuple(answer)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 4),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty perfect-group spectrum/support cards with witnessed repair."""
    from math import exp

    q = G.q
    assert q == 60 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    topology, model = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    first_rows = (
        (1,1,1,2,2),
        (0,1,2,2,2),
        (0,1,1,2,3),
        (0,0,2,2,3),
        (0,0,1,3,3),
    )
    first = first_rows[topology]
    sa = [[3, 3, 3]]
    sb = [list(first), [3-x for x in first]]
    assert [sum(row) for row in sb] == [7, 8]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Cache native group operations only. Every binary action below is
    # obtained from the authoritative scaffold, not a hand-written lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

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

    P = Probe()
    mask = (1 << q)-1
    monomials = [[[g] for g in range(q)]]
    raw_a, _, _ = lifted_matrices(P, monomials, [[[]]])
    _, raw_b, _ = lifted_matrices(P, [[[]]], monomials)
    blocks = [
        [[(row >> (g*q)) & mask for row in raw]
         for g in range(q)]
        for raw in (raw_a, raw_b)
    ]
    assert all(len(action) == q for side in blocks for action in side)

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for v in bits(word):
            value ^= columns[v]
        return value

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group")

    def generates(a, b):
        reached, pending = {ident}, [ident]
        for value in pending:
            for generator in (a, b):
                image = table[value][generator]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    spectra = ((3,3,5), (3,5,5), (5,5,5))
    catalogs = [[] for _ in spectra]
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            relative = table[inverses[a]][b]
            spectrum = tuple(sorted((orders[a], orders[b],
                                     orders[relative])))
            if spectrum not in spectra or not generates(a, b):
                continue
            cell = (ident, a, b)
            left = {table[inverses[x]][y]
                    for x in cell for y in cell if x != y}
            right = {table[x][inverses[y]]
                     for x in cell for y in cell if x != y}
            if len(left) == len(right) == 6:
                catalogs[spectra.index(spectrum)].append(cell)
    assert all(catalogs)
    ratio_pools = {
        order: [g for g in range(q) if orders[g] == order]
        for order in (3, 5)
    }

    # Unordered A-spectrum compositions distinguish the first twenty
    # cards. Extra harness cohorts use additional compositions and B
    # order-three binomials rather than submitting symmetry images.
    compositions = (
        ((0,0,0), (1,1,1), (2,2,2), (0,1,2)),
        ((0,0,1), (0,0,2), (0,1,1), (0,2,2)),
        ((1,1,2), (1,2,2), (0,0,0), (2,2,2)),
    )
    aplan = compositions[cohort % 3][model]
    bbinomials = [(i, j) for i in range(2) for j in range(5)
                  if sb[i][j] == 2]
    three_sites = set(
        bbinomials[:min(len(bbinomials), cohort % 3)])

    def draw_cell(count, side, i, j):
        if not count:
            return []
        root = rng.randrange(q)
        if count == 1:
            return [root]
        if count == 2:
            order = 3 if (i, j) in three_sites else 5
            ratio = rng.choice(ratio_pools[order])
            return [root, table[root][ratio]]
        assert count == 3
        kind = (aplan[j] if side == 0 else
                (model+i+j+cohort) % 3)
        triple = rng.choice(catalogs[kind])
        return [table[root][g] for g in triple]

    def sample_grid(support, side):
        return [[draw_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def component_data(grid, side):
        """GF2 superposition of authoritative scaffold monomial blocks."""
        nr, nc = len(grid), len(grid[0])
        width = nc*q
        rows = [0]*(nr*q)
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        columns = [0]*width
        for r, row in enumerate(rows):
            for v in bits(row):
                columns[v] |= 1 << r
        assert all(column.bit_count() == 3 for column in columns)

        seen, four = {}, 0
        adjacency = [0]*len(rows)
        for column in columns:
            checks = list(bits(column))
            for t, a in enumerate(checks):
                adjacency[a] |= column ^ (1 << a)
                for b in checks[:t]:
                    key = (b, a)
                    old = seen.get(key, 0)
                    four += old
                    seen[key] = old+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        # Projection excess is only a geometry heuristic when C4 exists.
        excess = max(0, triangles-width)
        cost = (600+150*model)*four + (1+cohort % 3)*excess
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return {"grid": copy_grid(grid), "rows": rows,
                "columns": columns, "nchecks": len(rows),
                "c4": four, "cycles": (four, excess), "cost": cost,
                "signature": signature, "d": width+1, "words": set()}

    def small_kernels(columns):
        """Component null-word counts, not quantum logical counts."""
        singles = {}
        duplicates = 0
        zero = 0
        for syndrome in columns:
            zero += int(syndrome == 0)
            duplicates += singles.get(syndrome, 0)
            singles[syndrome] = singles.get(syndrome, 0) + 1
        pairs = {}
        triples = 0
        quadruples = 0
        for i, x in enumerate(columns):
            for j in range(i):
                syndrome = x ^ columns[j]
                triples += singles.get(syndrome, 0)
                previous = pairs.get(syndrome, 0)
                quadruples += previous
                pairs[syndrome] = previous + 1
        # With distinct nonzero columns, the last two counts are exactly
        # three times the numbers of weight-3 and weight-4 null words.
        # Otherwise they remain conservative collision penalties.
        return (zero, duplicates, triples, quadruples)

    def note(record, word):
        """Retain only explicitly verified, genuine component null words."""
        if not word:
            return
        weight = word.bit_count()
        if weight > record["d"]+4:
            return
        assert syndrome(record["columns"], word) == 0
        record["d"] = min(record["d"], weight)
        record["words"].add(word)
        if len(record["words"]) > 48:
            record["words"] = set(sorted(
                record["words"], key=lambda w: (w.bit_count(), w))[:24])

    def circuit_reading(record, trials, offset):
        """Independent information sets and genuine kernel-word descent."""
        diagnostic = random.Random(seed+100003*record["side"]+offset)
        columns = record["columns"]
        order = list(range(len(columns)))
        retained = sorted(record["words"], key=int.bit_count)[:20]
        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(record, word)
            light = sorted(set(retained+basis), key=int.bit_count)[:18]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= record["d"]+2:
                        note(record, candidate)
            moves = retained+basis
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for sweep in range(2):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(record, word)
            retained = sorted(record["words"], key=int.bit_count)[:20]
            if record["d"] <= 4:
                break

    def parity_challenge(record, budget, offset, cap=12):
        """Budgeted anchored completion; exhaustion never certifies absence."""
        diagnostic = random.Random(seed+200003*record["side"]+offset)
        columns = record["columns"]
        width = len(columns)
        singles, pairs = {}, {}
        incidence = [0]*record["nchecks"]
        for v, col in enumerate(columns):
            bit = 1 << v
            singles.setdefault(col, bit)
            for r in bits(col):
                incidence[r] |= bit
            for u in range(v):
                pairs.setdefault(col ^ columns[u], bit | (1 << u))
        roots = list(range(0, width, q))
        diagnostic.shuffle(roots)
        priorities = [diagnostic.random() for _ in columns]
        all_variables = (1 << width)-1
        for root in roots:
            remaining = [max(1, budget//len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    note(record, word)
                    return
                left = min(cap, record["d"])-word.bit_count()
                if left <= 0 or value.bit_count() > 3*left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    note(record, word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    note(record, word | pair)
                left = min(cap, record["d"])-word.bit_count()
                if left <= 2:
                    return
                chosen, smallest = 0, width+1
                for r in bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                choices = list(bits(chosen))
                choices.sort(key=lambda v: (
                    -(columns[v] & value).bit_count(), priorities[v]))
                live = allowed
                for v in choices:
                    bit = 1 << v
                    live &= ~bit
                    visit(value ^ columns[v], word | bit, live)
                    if remaining[0] <= 0:
                        break

            # Classical regular symmetry permits an identity-coordinate
            # anchor in the first occupied block. No image codes are emitted.
            allowed = all_variables & ~((1 << (root+1))-1)
            visit(columns[root], 1 << root, allowed)

    def component_pool(support, side):
        bank = {}

        def remember(record):
            bank.setdefault(record["signature"], record)
            if len(bank) > 8:
                worst = max(bank, key=lambda key: bank[key]["cost"])
                del bank[worst]

        starts = []
        for _ in range(32):
            record = component_data(sample_grid(support, side), side)
            starts.append(record)
            remember(record)
        starts.sort(key=lambda r: r["cost"])
        positions = [(i, j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]

        for initial in starts[:2]:
            current = initial
            for step in range(48):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(positions)
                grid[i][j] = draw_cell(support[i][j], side, i, j)
                proposed = component_data(grid, side)
                delta = proposed["cost"]-current["cost"]
                temperature = q*(7*(1-step/48.0)**2+0.12)
                if delta <= 0 or rng.random() < exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(proposed)

        def screen(record, trials, budget, offset):
            record["small"] = small_kernels(record["columns"])
            if any(record["small"]):
                return False
            record["rank"] = gf2_rank(record["rows"])
            if record["rank"] < record["nchecks"]-6:
                return False
            record["side"] = side
            circuit_reading(record, trials, offset)
            parity_challenge(record, budget, offset+500009)
            return True

        def quality(record):
            multiplicity = sum(w.bit_count() == record["d"]
                               for w in record["words"])
            return (min(16, record["d"]), -multiplicity,
                    -record["cost"], record["rank"])

        selected = []
        for record in sorted(bank.values(), key=lambda r: r["cost"]):
            if screen(record, 8, 6000, 17):
                selected.append(record)
            if len(selected) >= 6:
                break
        if not selected:
            for trial in range(64):
                record = component_data(sample_grid(support, side), side)
                if screen(record, 8, 6000, 19+trial):
                    selected.append(record)
                    if len(selected) >= 3:
                        break
        if not selected:
            raise ValueError("no A5 component passed exact rank/kernel-four gates")

        selected.sort(key=quality, reverse=True)
        current = selected[0]
        keepers = selected[:3]
        historical = set()
        for record in selected:
            historical.update(record["words"])

        for step in range(10):
            light = sorted(current["words"], key=int.bit_count)[:8]
            proposals = []
            for _ in range(3):
                grid = copy_grid(current["grid"])
                if light:
                    occupied = sorted({v//q for v in bits(rng.choice(light))})
                    j = rng.choice(occupied)
                else:
                    j = rng.randrange(len(support[0]))
                active = [i for i in range(len(support)) if support[i][j]]
                # Alternate whole-column repair with a single-cell move.
                changed = active if step % 3 == 0 else [rng.choice(active)]
                for i in changed:
                    grid[i][j] = draw_cell(support[i][j], side, i, j)
                record = component_data(grid, side)
                if record["c4"] > current["c4"]+q:
                    continue
                survivors = sum(
                    syndrome(record["columns"], word) == 0
                    for word in historical)
                proposals.append((survivors, record["cost"], record))
            proposals.sort(key=lambda item: item[:2])
            for _, _, record in proposals[:2]:
                for word in historical:
                    if syndrome(record["columns"], word) == 0:
                        note(record, word)
                if not screen(record, 6, 3500, 700001+7919*step):
                    continue
                historical.update(record["words"])
                historical = set(sorted(
                    historical, key=lambda w: (w.bit_count(), w))[:80])
                keepers.append(record)
                if quality(record) >= quality(current):
                    current = record
            keepers.sort(key=quality, reverse=True)
            del keepers[6:]

        unique = {r["signature"]: r for r in keepers+[current]}
        finalists = sorted(unique.values(), key=quality, reverse=True)[:3]
        for record in finalists:
            # Previously found words remain binding for this matrix.
            # Fresh independent challenges can only lower its reading.
            circuit_reading(record, 48, 1700021)
            parity_challenge(record, 24000, 2700011)
            assert all(syndrome(record["columns"], word) == 0
                       for word in record["words"])
        finalists.sort(key=quality, reverse=True)
        return finalists

    aa = component_pool(sa, 0)
    bb = component_pool(sb, 1)
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(16, ab[0]["d"], ab[1]["d"]),
        min(20, ab[0]["d"])+min(20, ab[1]["d"]),
        -ab[0]["cost"]-ab[1]["cost"]), reverse=True)

    best = None
    for a, b in pairs:
        hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1020 and k >= 360
                and 325*n <= 1000*k < 375*n):
            continue
        # Confirm the cached authoritative component assembly against
        # the complete construction's sector-1 block order.
        actual_a = [
            sum(((hx[t] >> (j*5*q)) & mask) << (j*q)
                for j in range(3))
            for t in range(q)
        ]
        actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        detected = min(16, a["d"], b["d"])
        key = (detected, k*detected*detected,
               min(20, a["d"])+min(20, b["d"]),
               -a["cost"]-b["cost"])
        if best is None or key > best[0]:
            best = (key, a, b, hx, hz, k)
    if best is None:
        raise ValueError("A5 cubic pairings missed the exact 0.350 rate gate")

    key, a, b, cached_hx, cached_hz, k = best
    A = [[[G.elements[g] for g in cell] for cell in row]
         for row in a["grid"]]
    B = [[[G.elements[g] for g in cell] for cell in row]
         for row in b["grid"]]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1020
    assert k == n-gf2_rank(hx)-gf2_rank(hz)
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) == 11
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "card": (topology, model, cohort),
            "A_spectrum_composition": aplan,
            "component_readings": (a["d"], b["d"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "quality": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: w=12 alone prevents weight-4 logicals; parent slots 0-5 all had w=12 and d2s=4.
# REFUTED: Preliminary distance readings are stable under longer searches; the current D74 slot 46 fell from d2s=23 to d30s=20.
# REFUTED: Full-group-generating S3 x D14 trinomials and component screening raise the 0.350 five-code fitness above 35.3; the measured program instead returned five distance-eight codes and fitness 22.947, with minimum witnesses entirely in sector 1.
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
